// GameNodes.swift — the Play-mode logic graph for LumenScene (v4).
//
// A project carries one NodeGraph (Scene.graph): boxes ("nodes")
// wired output-pin → input-pin by links. Event nodes start a signal
// (Play starts, the player enters an area, a timer fires…), Action
// nodes do things (move an object, start a particle emitter, change
// the weather…), and Logic nodes steer the signal between them.
//
// This file is the single source of truth for the graph's shape: the
// node editor builds its palette and inspector from NodeCatalog, and
// the runtime interprets exactly the same kind strings and parameter
// keys. Kind strings and parameter keys are contractual — renaming
// one silently breaks every saved project that uses it.

import Foundation
import simd

// MARK: - Parameter values

/// One parameter value on a node. Object/area references are stored
/// as UUIDs (nil = nothing picked yet); colours and choices ride in
/// `text` ("#FF9A33", "Rain", …) so the format stays simple.
enum NodeParam: Codable, Equatable {
    case number(Double)
    case bool(Bool)
    case text(String)
    case ref(UUID?)
}

// MARK: - Nodes and links

struct LogicNode: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    /// A NodeCatalog kind string, e.g. "evEnterArea" or "actMoveTo".
    var kind: String
    var params: [String: NodeParam] = [:]
    /// Canvas position in the node editor (points).
    var x: Double = 0
    var y: Double = 0

    /// Number parameter `k`, or `def` when the key is missing or
    /// holds a different case.
    func num(_ k: String, _ def: Double) -> Double {
        if let p = params[k], case .number(let v) = p { return v }
        return def
    }

    /// Bool parameter `k`, or `def` when absent/mistyped.
    func boolean(_ k: String, _ def: Bool) -> Bool {
        if let p = params[k], case .bool(let v) = p { return v }
        return def
    }

    /// Text parameter `k` (also how choices and colours are stored),
    /// or `def` when absent/mistyped.
    func txt(_ k: String, _ def: String) -> String {
        if let p = params[k], case .text(let v) = p { return v }
        return def
    }

    /// Referenced object/area id for `k`; nil when the key is
    /// missing, mistyped, or explicitly set to no reference.
    func ref(_ k: String) -> UUID? {
        if let p = params[k], case .ref(let v) = p { return v }
        return nil
    }

    /// A copy of this node with one parameter set — handy for the
    /// editor's inspector, which edits nodes as values.
    func with(_ k: String, _ v: NodeParam) -> LogicNode {
        var copy = self
        copy.params[k] = v
        return copy
    }
}

extension LogicNode {
    private enum CodingKeys: String, CodingKey {
        case id, kind, params, x, y
    }

    // Custom Decodable, defensive like the rest of the model: a
    // graph written by an early v4 build may lack keys a later one
    // adds, and that must not fail the whole project load. Every
    // field falls back to exactly the property default above
    // (`kind` had no default, so a missing kind reads as "").
    // encode(to:) stays synthesised — see Models.swift.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        kind = try c.decodeIfPresent(String.self, forKey: .kind) ?? ""
        params = try c.decodeIfPresent([String: NodeParam].self, forKey: .params) ?? [:]
        x = try c.decodeIfPresent(Double.self, forKey: .x) ?? 0
        y = try c.decodeIfPresent(Double.self, forKey: .y) ?? 0
    }
}

/// A wire from one node's output pin to another node's input pin.
/// The receiving pin (`toPin`) is "in" for a node's normal input, or
/// the name of an extra input (the particle emitter's "stop").
struct NodeLink: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    var from: UUID
    var fromPin: String
    var to: UUID
    var toPin: String
}

extension NodeLink {
    private enum CodingKeys: String, CodingKey {
        case id, from, fromPin, to, toPin
    }

    // Defensive decodeIfPresent init, same reasoning as LogicNode.
    // `from`/`to` had no defaults; a missing endpoint falls back to
    // the all-zero UUID, which matches no node and is inert.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        from = try c.decodeIfPresent(UUID.self, forKey: .from)
            ?? UUID(uuidString: "00000000-0000-0000-0000-000000000000")!
        fromPin = try c.decodeIfPresent(String.self, forKey: .fromPin) ?? "out"
        to = try c.decodeIfPresent(UUID.self, forKey: .to)
            ?? UUID(uuidString: "00000000-0000-0000-0000-000000000000")!
        toPin = try c.decodeIfPresent(String.self, forKey: .toPin) ?? "in"
    }
}

struct NodeGraph: Codable, Equatable {
    var nodes: [LogicNode] = []
    var links: [NodeLink] = []
}

extension NodeGraph {
    private enum CodingKeys: String, CodingKey {
        case nodes, links
    }

    // Defensive decodeIfPresent init, same reasoning as LogicNode.
    // (Whole-graph absence is already handled by Scene, which decodes
    // `graph` itself with decodeIfPresent.)
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        nodes = try c.decodeIfPresent([LogicNode].self, forKey: .nodes) ?? []
        links = try c.decodeIfPresent([NodeLink].self, forKey: .links) ?? []
    }
}

// MARK: - Catalogue

/// How the editor should present one parameter. `objectRef` picks a
/// scene object, `areaRef` picks a trigger area, `colorText` edits a
/// "#RRGGBB" string, and `choice` picks from `choices` in the spec.
enum NodeParamType {
    case number, bool, text, objectRef, areaRef, colorText, choice
}

/// One parameter of a node kind: its storage key, inspector label,
/// editor type, default value, and — for numbers — the sensible
/// range. A number range of 0...0 means "no range" (free entry).
struct NodeParamSpec {
    var key: String
    var label: String
    var type: NodeParamType
    var def: NodeParam
    var min: Double = 0
    var max: Double = 0
    var choices: [String] = []
}

/// One node kind: identity, palette wording, wiring shape, params.
/// `hasIn` is false for events (they start signals); `extraIns`
/// names any inputs beyond the normal "in" pin; `outs` names the
/// output pins a signal can leave by.
struct NodeKindSpec {
    var kind: String
    var title: String
    var category: String
    var blurb: String
    var hasIn: Bool
    var extraIns: [String] = []
    var outs: [String]
    var params: [NodeParamSpec] = []
}

enum NodeCatalog {
    static let kinds: [NodeKindSpec] = [

        // MARK: Events — hasIn = false, one "out" pin.

        NodeKindSpec(
            kind: "evStart", title: "Play Starts", category: "Events",
            blurb: "Runs once, the moment Play starts.",
            hasIn: false, outs: ["out"]),

        NodeKindSpec(
            kind: "evEnterArea", title: "Player Enters Area", category: "Events",
            blurb: "Runs when the player walks into the chosen area.",
            hasIn: false, outs: ["out"],
            params: [
                NodeParamSpec(key: "area", label: "Area", type: .areaRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "evExitArea", title: "Player Exits Area", category: "Events",
            blurb: "Runs when the player walks out of the chosen area.",
            hasIn: false, outs: ["out"],
            params: [
                NodeParamSpec(key: "area", label: "Area", type: .areaRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "evTap", title: "Object Tapped", category: "Events",
            blurb: "Runs when the player taps the chosen object.",
            hasIn: false, outs: ["out"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "evTimer", title: "Timer", category: "Events",
            blurb: "Runs after a delay, then keeps running on a timer if Repeat is on.",
            hasIn: false, outs: ["out"],
            params: [
                NodeParamSpec(key: "delay", label: "Delay (seconds)", type: .number, def: .number(0), min: 0, max: 600),
                NodeParamSpec(key: "interval", label: "Repeat Every (seconds)", type: .number, def: .number(5), min: 0.1, max: 600),
                NodeParamSpec(key: "repeat", label: "Repeat", type: .bool, def: .bool(true)),
            ]),

        NodeKindSpec(
            kind: "evVariable", title: "Variable Changes", category: "Events",
            blurb: "Runs whenever the named number changes.",
            hasIn: false, outs: ["out"],
            params: [
                NodeParamSpec(key: "name", label: "Variable Name", type: .text, def: .text("score")),
            ]),

        // MARK: Actions — hasIn = true, "out" pin unless noted.

        NodeKindSpec(
            kind: "actShow", title: "Show Object", category: "Actions",
            blurb: "Makes the chosen object appear.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "actHide", title: "Hide Object", category: "Actions",
            blurb: "Makes the chosen object disappear.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "actMoveTo", title: "Move Object", category: "Actions",
            blurb: "Moves an object to another object, or by an offset, over a few seconds.",
            hasIn: true, outs: ["out", "arrived"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
                NodeParamSpec(key: "mode", label: "Mode", type: .choice, def: .text("To Object"), choices: ["To Object", "By Offset"]),
                NodeParamSpec(key: "target", label: "Target Object", type: .objectRef, def: .ref(nil)),
                NodeParamSpec(key: "dx", label: "Offset X (m)", type: .number, def: .number(0), min: -50, max: 50),
                NodeParamSpec(key: "dy", label: "Offset Y (m)", type: .number, def: .number(0), min: -50, max: 50),
                NodeParamSpec(key: "dz", label: "Offset Z (m)", type: .number, def: .number(0), min: -50, max: 50),
                NodeParamSpec(key: "duration", label: "Duration (seconds)", type: .number, def: .number(1), min: 0, max: 60),
            ]),

        NodeKindSpec(
            kind: "actRotate", title: "Rotate Object", category: "Actions",
            blurb: "Spins an object at a steady speed; a duration of 0 spins it forever.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
                NodeParamSpec(key: "speed", label: "Speed (°/s)", type: .number, def: .number(45), min: -360, max: 360),
                NodeParamSpec(key: "duration", label: "Duration (seconds)", type: .number, def: .number(0), min: 0, max: 3600),
            ]),

        NodeKindSpec(
            kind: "actSetLight", title: "Set Light", category: "Actions",
            blurb: "Turns an object's light on or off and sets its brightness and colour.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
                NodeParamSpec(key: "on", label: "Light On", type: .bool, def: .bool(true)),
                NodeParamSpec(key: "intensity", label: "Intensity", type: .number, def: .number(4), min: 0, max: 40),
                NodeParamSpec(key: "color", label: "Colour", type: .colorText, def: .text("#FF9A33")),
            ]),

        NodeKindSpec(
            kind: "actSetMaterial", title: "Set Material", category: "Actions",
            blurb: "Changes an object's surface texture, or turns it into glass.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "object", label: "Object", type: .objectRef, def: .ref(nil)),
                // The editor extends this choice list at runtime with
                // the renderer's real texture presets; "none" is the
                // only choice baked into the catalogue.
                NodeParamSpec(key: "texture", label: "Texture", type: .choice, def: .text("none"), choices: ["none"]),
                NodeParamSpec(key: "glass", label: "Make Glass", type: .bool, def: .bool(false)),
            ]),

        NodeKindSpec(
            kind: "actParticles", title: "Particle Emitter", category: "Actions",
            blurb: "Sprays copies of a model that fall, bounce and settle until the emitter is stopped.",
            hasIn: true, extraIns: ["stop"], outs: ["out"],
            params: [
                NodeParamSpec(key: "source", label: "Model", type: .objectRef, def: .ref(nil)),
                NodeParamSpec(key: "mode", label: "Mode", type: .choice, def: .text("Physics"), choices: ["Physics", "Simple"]),
                NodeParamSpec(key: "rate", label: "Rate (per second)", type: .number, def: .number(6), min: 0, max: 240),
                NodeParamSpec(key: "burst", label: "Burst (at start)", type: .number, def: .number(0), min: 0, max: 500),
                NodeParamSpec(key: "duration", label: "Duration (seconds)", type: .number, def: .number(0), min: 0, max: 3600),
                NodeParamSpec(key: "lifetime", label: "Particle Lifetime (seconds)", type: .number, def: .number(12), min: 0.5, max: 120),
                NodeParamSpec(key: "size", label: "Size (m)", type: .number, def: .number(0.12), min: 0.01, max: 2),
                NodeParamSpec(key: "speed", label: "Speed (m/s)", type: .number, def: .number(0.6), min: 0, max: 20),
                NodeParamSpec(key: "spread", label: "Spread (degrees)", type: .number, def: .number(35), min: 0, max: 180),
                NodeParamSpec(key: "direction", label: "Direction", type: .choice, def: .text("Up"), choices: ["Up", "Outward", "Down"]),
                NodeParamSpec(key: "gravity", label: "Gravity", type: .number, def: .number(0.35), min: 0, max: 2),
                NodeParamSpec(key: "drag", label: "Drag", type: .number, def: .number(0.6), min: 0, max: 4),
                NodeParamSpec(key: "faces", label: "Faces per Particle", type: .number, def: .number(24), min: 4, max: 2000),
                NodeParamSpec(key: "hitbox", label: "Hitbox Shape", type: .choice, def: .text("Disc"), choices: ["Disc", "Box", "Sphere"]),
                NodeParamSpec(key: "hitboxSize", label: "Hitbox Size", type: .number, def: .number(1), min: 0.2, max: 3),
                NodeParamSpec(key: "bounce", label: "Bounce", type: .number, def: .number(0), min: 0, max: 0.9),
                NodeParamSpec(key: "drape", label: "Drape", type: .bool, def: .bool(true)),
            ]),

        NodeKindSpec(
            kind: "actLoadArea", title: "Load Area", category: "Actions",
            blurb: "Brings in everything that belongs to the chosen area.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "area", label: "Area", type: .areaRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "actUnloadArea", title: "Unload Area", category: "Actions",
            blurb: "Removes everything that belongs to the chosen area.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "area", label: "Area", type: .areaRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "actSetWeather", title: "Set Weather", category: "Actions",
            blurb: "Changes the weather to rain, snow, or off.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "mode", label: "Weather", type: .choice, def: .text("Rain"), choices: ["Off", "Rain", "Snow"]),
                NodeParamSpec(key: "clear", label: "Clear", type: .bool, def: .bool(false)),
            ]),

        NodeKindSpec(
            kind: "actTeleport", title: "Teleport Player", category: "Actions",
            blurb: "Moves the player to the chosen object, instantly.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "target", label: "Target", type: .objectRef, def: .ref(nil)),
            ]),

        NodeKindSpec(
            kind: "actMessage", title: "Show Message", category: "Actions",
            blurb: "Shows a short message on the screen for a few seconds.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "message", label: "Message", type: .text, def: .text("Hello")),
                NodeParamSpec(key: "seconds", label: "Show For (seconds)", type: .number, def: .number(3), min: 0.5, max: 30),
            ]),

        NodeKindSpec(
            kind: "actSetVariable", title: "Set Variable", category: "Actions",
            blurb: "Sets a named number to a value, or adds to the number.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "name", label: "Variable Name", type: .text, def: .text("score")),
                NodeParamSpec(key: "op", label: "Operation", type: .choice, def: .text("Set"), choices: ["Set", "Add"]),
                NodeParamSpec(key: "value", label: "Value", type: .number, def: .number(1)),
            ]),

        // MARK: Logic — hasIn = true, pins vary per kind.

        NodeKindSpec(
            kind: "logIf", title: "If Variable", category: "Logic",
            blurb: "Sends the signal to Then if the number matches, or to Else if it doesn't.",
            hasIn: true, outs: ["then", "else"],
            params: [
                NodeParamSpec(key: "name", label: "Variable Name", type: .text, def: .text("score")),
                NodeParamSpec(key: "cmp", label: "Comparison", type: .choice, def: .text(">="), choices: ["==", "!=", ">", "<", ">=", "<="]),
                NodeParamSpec(key: "value", label: "Value", type: .number, def: .number(1)),
            ]),

        NodeKindSpec(
            kind: "logDelay", title: "Delay", category: "Logic",
            blurb: "Waits a few seconds, then passes the signal on.",
            hasIn: true, outs: ["out"],
            params: [
                NodeParamSpec(key: "seconds", label: "Delay (seconds)", type: .number, def: .number(1), min: 0, max: 600),
            ]),

        NodeKindSpec(
            kind: "logToggle", title: "Toggle", category: "Logic",
            blurb: "Takes turns: the first signal goes to On, the next to Off, and so on.",
            hasIn: true, outs: ["on", "off"]),

        NodeKindSpec(
            kind: "logRandom", title: "Random Chance", category: "Logic",
            blurb: "Sends the signal to Yes a chosen percent of the time, otherwise to No.",
            hasIn: true, outs: ["yes", "no"],
            params: [
                NodeParamSpec(key: "percent", label: "Chance (%)", type: .number, def: .number(50), min: 0, max: 100),
            ]),

        NodeKindSpec(
            kind: "logCounter", title: "Counter", category: "Logic",
            blurb: "Counts signals, passes each one on, and fires Reached when the count gets to the target.",
            hasIn: true, outs: ["out", "reached"],
            params: [
                NodeParamSpec(key: "target", label: "Count To", type: .number, def: .number(5), min: 1, max: 10000),
            ]),
    ]

    /// The spec for a kind string, or nil for an unknown kind (a
    /// node from a newer build, say) — callers should leave unknown
    /// nodes untouched rather than dropping them.
    static func spec(_ kind: String) -> NodeKindSpec? {
        kinds.first { $0.kind == kind }
    }

    /// Fresh default parameters for a new node of `kind`, built
    /// from the spec so the editor and runtime never disagree.
    /// Unknown kinds get no parameters.
    static func defaultParams(for kind: String) -> [String: NodeParam] {
        guard let spec = spec(kind) else { return [:] }
        var result: [String: NodeParam] = [:]
        for p in spec.params { result[p.key] = p.def }
        return result
    }
}
