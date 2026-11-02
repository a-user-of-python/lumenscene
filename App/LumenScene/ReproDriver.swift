// ReproDriver.swift — LS-REPRO: off-device reproduction harness for
// the v4.10.x snow-accumulation-start crash (Lane A, v4.10.3).
//
// COMPLETELY INERT unless the process is launched with the
// `-LSRepro` argument (CI: `xcrun simctl launch ... com.user.
// lumenscene -LSRepro [-LSVariant <name>] [-LSToggle <name> ...]`).
// A normal launch never touches any of this: LumenSceneApp's
// init only calls bootstrapIfNeeded(), which returns immediately
// when the argument is absent.
//
// When active, the driver builds a REAL EditorSession over a
// fresh DemoScene project on the shared Renderer and pumps the
// REAL per-frame entry point — EditorSession.frameTick(view:) —
// on the main runloop at ~30 Hz, exactly as MetalViewport's
// MTKViewDelegate draw does. frameTick derives dt from wall
// clock (CACurrentMediaTime), so the pump runs in REAL time:
// weather dynamics, the quantised geometry-step trigger, the
// sheet-only scoped rebuild, the async staged static AS build +
// frame-boundary swap (swapStaticIfReady, polled at the top of
// Renderer.draw), and the GPU weather sim tick (tickWeatherSim,
// called from Renderer.draw) all execute through their live
// call paths, in the live order. There is NO parallel fake
// weather implementation here — setWeatherMode(2) is the exact
// API the Weather menu's picker binding calls.
//
// Progress is reported as `REPRO-MARKER <step>` lines on stdout
// (print + fflush, mirrored via NSLog) so they appear in
// `simctl launch --console` output. The driver's own heartbeat
// plus every distinct Renderer.weatherStep value observed after
// each pump names the last completed link before a crash.
// Success ends with `REPRO-DONE` and exit(0). An internal
// watchdog ends with `REPRO-TIMEOUT` and exit(2) — a harness
// outcome, NOT a crash: CI distinguishes the two by the crash
// report, not by exit status alone.
//
// Variants (-LSVariant):
//   default  — DemoScene.makeProject() as-is.
//   plane    — the demo with every non-plane object deleted.
//   scaled   — the demo plus a box with transform scale 0 on X
//              and a box with transform scale 50 on all axes.
//   bigplane — the demo floor plane scaled ×12 on X/Z (~120 m).
//   perfbench — v4.13 controlled perf baseline: rain scene,
//              frameTick CPU ms at rest + edit-rebuild wall
//              times as PERF metrics (host CPU; device FPS
//              stays the device's verdict).
//   gravfall / gravstack — v4.13 Applied Gravity audits
//              (fall curve, landing, stacking, negative
//              strength, save/load).
//   loopmove — v4.14 Loop audits: a Loop drives Move Object
//              and Rotate Object through repeated cycles,
//              stops dead at Play exit, and starts fresh on
//              re-entry; graph save/load round-trips.
//   playernodes — v4.14 player targets: Move Object glides
//              the player to a target; Rotate Object turns
//              the player camera and writes through to the
//              real look state (an audit-applied look drag
//              must survive — no snap-back); a Loop drives
//              the camera rotate forever.
//
// Retired with v4.13's weather removal: freeze / freeze2
// (their establishment witness was the snow build-up) and
// wallbox / walldeep / carve / refill / snowgate (the v4.12
// snow audits — snow itself is removed). The water audits
// retired with the 3D water the same day.
// Retired variant machines: deeprest / deepdrag / deepspawn
// (v4.10.4) and pile / settle / drag2 / spawn2 / roof /
// longrun (v4.11 physics) — removed with the models they
// audited.

import Foundation
import MetalKit
import UIKit

/// LS-REPRO TOGGLE registry — read ONCE from the launch
/// arguments (`-LSToggle <name>`, repeatable). Every toggle is
/// a minimal guard at one exact link in Renderer.swift, marked
/// `// LS-REPRO TOGGLE` at the call site. Default (no argument)
/// is the current shipped behaviour, bit for bit.
enum ReproToggles {
    static let active: Set<String> = {
        var out: Set<String> = []
        let args = ProcessInfo.processInfo.arguments
        var i = 0
        while i < args.count {
            if args[i] == "-LSToggle", i + 1 < args.count {
                out.insert(args[i + 1])
                i += 2
            } else {
                i += 1
            }
        }
        return out
    }()

    /// Skip GPU sim stepping entirely (heights stay at the
    /// sheet build's initial condition).
    static var simTick: Bool { active.contains("simtick") }
    /// Force the synchronous static AS build path (bypass the
    /// v4.9 async staged build + frame-boundary swap).
    static var syncAS: Bool { active.contains("syncas") }
    /// Disable the v4.9 sheet-only scoped rebuild (always full
    /// static rebuild).
    static var noScoped: Bool { active.contains("noscoped") }
    /// Dynamic layer does a full AS rebuild instead of an
    /// in-place refit.
    static var noRefit: Bool { active.contains("norefit") }
    /// Sim readback uses explicit buffer copies (simRead /
    /// simWrite) instead of the v4.9 zero-copy contents()
    /// pointer reads.
    static var copyReads: Bool { active.contains("copyreads") }
    /// v4.10.4 round 2 experiment (gate reverted in round 3
    /// after the gated/nogate repro comparison showed the
    /// gate neutral at rest and harmful to drag volume
    /// conservation): when the gate existed, this disabled
    /// it. Now unreferenced by Renderer; kept registered so
    /// old run scripts still parse.
    static var noGate: Bool { active.contains("nogate") }
}

enum ReproDriver {
    static var isActive: Bool {
        ProcessInfo.processInfo.arguments.contains("-LSRepro")
    }

    static var variant: String {
        let args = ProcessInfo.processInfo.arguments
        if let i = args.firstIndex(of: "-LSVariant"), i + 1 < args.count {
            return args[i + 1]
        }
        return "default"
    }

    static func marker(_ text: String) {
        let line = "REPRO-MARKER \(text)"
        print(line)
        fflush(stdout)
        NSLog("%@", line)
    }

    /// Called from LumenSceneApp.init — returns at once unless
    /// `-LSRepro` was passed. The 1.2 s delay lets the app's
    /// own launch (RootView seeding + first window) settle
    /// before the driver takes over the shared renderer.
    /// The live run, retained for the process lifetime (the
    /// pump chain references it weakly by design).
    private static var activeRun: ReproRun?

    static func bootstrapIfNeeded() {
        guard isActive else { return }
        print("REPRO-START variant=\(variant) toggles=\(ReproToggles.active.sorted().joined(separator: ","))")
        fflush(stdout)
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            MainActor.assumeIsolated {
                let run = ReproRun()
                activeRun = run
                run.start()
            }
        }
    }

    static func makeProject(variant: String) -> Project {
        var p = DemoScene.makeProject()
        switch variant {
        case "plane":
            // The demo minus every box/other object: only the
            // floor plane remains as support.
            p.scene.objects.removeAll { $0.primitiveKind != "plane" }
        case "perfbench":
            // Controlled perf baseline scene: a 24x24 m plane
            // (the sheet/sim lattice size class Sean's deep
            // scenes use) plus six boxes. Weather is driven by
            // the variant's own phase machine (rain).
            p.scene.objects.removeAll()
            var big = SceneObject.primitive("plane", name: "Repro bench plane")
            big.params = ["w": 24, "d": 24]
            p.scene.objects.append(big)
            for i in 0..<6 {
                var b = SceneObject.primitive("box", name: "Repro bench box \(i)")
                b.transform.position = [Double(-7 + i * 3), 0.5, Double((i % 2) * 6 - 3)]
                p.scene.objects.append(b)
            }
        case "scaled":
            var zero = SceneObject.primitive("box", name: "Repro zero-scale box")
            zero.transform.position = [0.8, 0.5, 0.8]
            zero.transform.scale = [0, 1, 1]
            p.scene.objects.append(zero)
            var huge = SceneObject.primitive("box", name: "Repro huge-scale box")
            huge.transform.position = [30, 25, 30]
            huge.transform.scale = [50, 50, 50]
            p.scene.objects.append(huge)
        case "bigplane":
            if let i = p.scene.objects.firstIndex(where: { $0.primitiveKind == "plane" }) {
                p.scene.objects[i].transform.scale = [12, 1, 12]
            }
        case "gravfall":
            // v4.13 Applied Gravity audit (a)+(c)+(e): a bare
            // 8x8 plane; a 1 m box 3 m above it wired to an
            // Applied Gravity node at 9.8; a second box at the
            // same height wired at strength 0 (must hang); a
            // third box unwired (control — must never move).
            // The phase machine turns the weather off, runs the
            // save/load audit, then enters Play and samples.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            var fall = SceneObject.primitive("box", name: "Repro fall box")
            fall.transform.position = [0, 3.5, 0]
            var floatBox = SceneObject.primitive("box", name: "Repro float box")
            floatBox.transform.position = [-2.5, 3.5, 0]
            var control = SceneObject.primitive("box", name: "Repro control box")
            control.transform.position = [2.5, 3.5, 0]
            p.scene.objects = [plane, fall, floatBox, control]
            let start = LogicNode(kind: "evStart")
            var gFall = LogicNode(kind: "actGravity", x: 220, y: 0)
            gFall.params = ["object": .ref(fall.id),
                            "strength": .number(9.8), "bounce": .number(0)]
            var gFloat = LogicNode(kind: "actGravity", x: 220, y: 140)
            gFloat.params = ["object": .ref(floatBox.id),
                             "strength": .number(0), "bounce": .number(0)]
            p.scene.graph.nodes = [start, gFall, gFloat]
            p.scene.graph.links = [
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: gFall.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: gFloat.id, toPin: "in"),
            ]
        case "gravstack":
            // v4.13 Applied Gravity audit (b)+(d) + support
            // removal: a faller lands on a lower box's top; the
            // graph itself hides the lower box after 2.5 s
            // (Delay → Hide Object) and the faller must fall
            // again to the plane. A riser under negative gravity
            // (−9.8) rises and rests against the underside of a
            // ceiling box (its bottom at 2.5).
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            var lower = SceneObject.primitive("box", name: "Repro lower box")
            lower.transform.position = [0, 0.5, 0]
            var faller = SceneObject.primitive("box", name: "Repro stack faller")
            faller.transform.position = [0, 3.5, 0]
            var ceilingBox = SceneObject.primitive("box", name: "Repro ceiling box")
            ceilingBox.transform.position = [3, 3.0, 0]
            var riser = SceneObject.primitive("box", name: "Repro riser box")
            riser.transform.position = [3, 0.5, 0]
            p.scene.objects = [plane, lower, faller, ceilingBox, riser]
            let start = LogicNode(kind: "evStart")
            var gFall = LogicNode(kind: "actGravity", x: 220, y: 0)
            gFall.params = ["object": .ref(faller.id),
                            "strength": .number(9.8), "bounce": .number(0)]
            var delay = LogicNode(kind: "logDelay", x: 220, y: 140)
            delay.params = ["seconds": .number(2.5)]
            var hide = LogicNode(kind: "actHide", x: 440, y: 140)
            hide.params = ["object": .ref(lower.id)]
            var gRise = LogicNode(kind: "actGravity", x: 220, y: 280)
            gRise.params = ["object": .ref(riser.id),
                            "strength": .number(-9.8), "bounce": .number(0)]
            var delay2 = LogicNode(kind: "logDelay", x: 220, y: 420)
            delay2.params = ["seconds": .number(4.0)]
            var hide2 = LogicNode(kind: "actHide", x: 440, y: 420)
            hide2.params = ["object": .ref(ceilingBox.id)]
            p.scene.graph.nodes = [start, gFall, delay, hide, gRise, delay2, hide2]
            p.scene.graph.links = [
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: gFall.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: delay.id, toPin: "in"),
                NodeLink(id: UUID(), from: delay.id, fromPin: "out", to: hide.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: gRise.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: delay2.id, toPin: "in"),
                NodeLink(id: UUID(), from: delay2.id, fromPin: "out", to: hide2.id, toPin: "in"),
            ]
        case "loopmove":
            // v4.14 Loop audit (object movers): one Loop drives
            // a Move Object (By Offset +1.5 m, 0.8 s) with a
            // 0.25 s repeat delay — the box must march one
            // offset per cycle, continuously, for as long as
            // Play runs; a second Loop drives a Rotate Object
            // (120°/s × 0.5 s = 60° per cycle, 0.4 s delay).
            // The phase machine also checks Play-exit stillness
            // (the snapshot restore returns the box, and nothing
            // may keep driving it) and a fresh second session.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            var march = SceneObject.primitive("box", name: "Repro march box")
            march.transform.position = [0, 0.5, 0]
            var spin = SceneObject.primitive("box", name: "Repro spin box")
            spin.transform.position = [-3, 0.5, 0]
            p.scene.objects = [plane, march, spin]
            let start = LogicNode(kind: "evStart")
            var move = LogicNode(kind: "actMoveTo", x: 440, y: 0)
            move.params = ["subject": .text("Object"),
                           "object": .ref(march.id),
                           "mode": .text("By Offset"),
                           "dx": .number(1.5), "dy": .number(0), "dz": .number(0),
                           "duration": .number(0.8)]
            var loopMove = LogicNode(kind: "logLoop", x: 220, y: 0)
            loopMove.params = ["node": .ref(move.id), "delay": .number(0.25)]
            var rotate = LogicNode(kind: "actRotate", x: 440, y: 140)
            rotate.params = ["subject": .text("Object"),
                             "object": .ref(spin.id),
                             "speed": .number(120), "duration": .number(0.5)]
            var loopSpin = LogicNode(kind: "logLoop", x: 220, y: 140)
            loopSpin.params = ["node": .ref(rotate.id), "delay": .number(0.4)]
            p.scene.graph.nodes = [start, loopMove, move, loopSpin, rotate]
            p.scene.graph.links = [
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: loopMove.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: loopSpin.id, toPin: "in"),
            ]
        case "playernodes":
            // v4.14 player-target audit: Move Object with
            // Move = Player glides the viewpoint to a target
            // box (the eye rule puts the eye at 2.2 over its
            // 0.5 centre); Rotate Object with Rotate = Player
            // Camera then turns the view +90° over 1 s (after a
            // 2 s Delay); the audit drags the look +30° itself
            // (the Viewport gesture's exact write) and the yaw
            // must KEEP the dragged value — no snap-back. From
            // 5.5 s a Loop drives a second Player Camera rotate
            // (180°/s × 0.5 s = 90°/cycle, 0.5 s delay) forever.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 12, "d": 12]
            plane.transform.position = [0, 0, 0]
            var target = SceneObject.primitive("box", name: "Repro player target")
            target.transform.position = [4, 0.5, 0]
            p.scene.objects = [plane, target]
            p.scene.camera.playPosition = [0, 1.7, 0]
            p.scene.camera.playYaw = 0
            p.scene.camera.playPitch = 0
            let start = LogicNode(kind: "evStart")
            var pMove = LogicNode(kind: "actMoveTo", x: 220, y: 0)
            pMove.params = ["subject": .text("Player"),
                            "mode": .text("To Object"),
                            "target": .ref(target.id),
                            "dx": .number(0), "dy": .number(0), "dz": .number(0),
                            "duration": .number(1.2)]
            var delay = LogicNode(kind: "logDelay", x: 220, y: 140)
            delay.params = ["seconds": .number(2.0)]
            var camRot = LogicNode(kind: "actRotate", x: 440, y: 140)
            camRot.params = ["subject": .text("Player Camera"),
                             "speed": .number(90), "duration": .number(1.0)]
            var delay2 = LogicNode(kind: "logDelay", x: 220, y: 280)
            delay2.params = ["seconds": .number(5.5)]
            var camRot2 = LogicNode(kind: "actRotate", x: 660, y: 280)
            camRot2.params = ["subject": .text("Player Camera"),
                              "speed": .number(180), "duration": .number(0.5)]
            var camLoop = LogicNode(kind: "logLoop", x: 440, y: 280)
            camLoop.params = ["node": .ref(camRot2.id), "delay": .number(0.5)]
            p.scene.graph.nodes = [start, pMove, delay, camRot, delay2, camLoop, camRot2]
            p.scene.graph.links = [
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: pMove.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: delay.id, toPin: "in"),
                NodeLink(id: UUID(), from: delay.id, fromPin: "out", to: camRot.id, toPin: "in"),
                NodeLink(id: UUID(), from: start.id, fromPin: "out", to: delay2.id, toPin: "in"),
                NodeLink(id: UUID(), from: delay2.id, fromPin: "out", to: camLoop.id, toPin: "in"),
            ]
        case "convglass", "convrefl", "convlight":
            // v4.15 convergence-audit reference scenes. Each
            // stresses one tracer path family; handleAudit
            // renders them offscreen at a fixed sample count
            // and compares per-region converged statistics
            // against the pre-optimisation baseline.
            func prim(_ kind: String, _ name: String,
                      pos: [Double], rot: [Double] = [0, 0, 0],
                      params: [String: Double] = [:],
                      material: MaterialProps) -> SceneObject {
                var o = SceneObject.primitive(kind, name: name)
                if !params.isEmpty { o.params = params }
                o.transform.position = pos
                o.transform.rotation = rot
                o.material = material
                return o
            }
            func room(_ w: Double, _ d: Double, _ h: Double,
                      floorPreset: String, floorScale: Double) -> [SceneObject] {
                var floorMat = MaterialProps()
                floorMat.texturePreset = floorPreset
                floorMat.textureScale = floorScale
                var brickMat = MaterialProps()
                brickMat.texturePreset = "brick"
                brickMat.textureScale = 2.5
                var cementMat = MaterialProps()
                cementMat.texturePreset = "cement"
                cementMat.textureScale = 2.5
                var plainMat = MaterialProps()
                plainMat.baseColor = RGB(0.45, 0.45, 0.47)
                var ceilMat = MaterialProps()
                ceilMat.baseColor = RGB(0.16, 0.16, 0.17)
                return [
                    prim("plane", "Audit floor", pos: [0, 0, 0],
                         params: ["w": w, "d": d], material: floorMat),
                    prim("box", "Audit back wall", pos: [0, h / 2, -d / 2 + 0.1],
                         params: ["w": w, "h": h, "d": 0.2], material: brickMat),
                    prim("box", "Audit left wall", pos: [-w / 2 + 0.1, h / 2, 0],
                         params: ["w": 0.2, "h": h, "d": d], material: cementMat),
                    prim("box", "Audit right wall", pos: [w / 2 - 0.1, h / 2, 0],
                         params: ["w": 0.2, "h": h, "d": d], material: plainMat),
                    prim("box", "Audit ceiling", pos: [0, h + 0.1, 0],
                         params: ["w": w, "h": 0.2, "d": d], material: ceilMat),
                ]
            }
            func emitter(_ name: String, _ kind: String,
                         pos: [Double], rot: [Double] = [0, 0, 0],
                         params: [String: Double],
                         color: RGB, intensity: Double) -> SceneObject {
                var m = MaterialProps()
                m.baseColor = RGB(0.1, 0.1, 0.1)
                m.emissiveOn = true
                m.emissiveColor = color
                m.emissiveIntensity = intensity
                return prim(kind, name, pos: pos, rot: rot,
                            params: params, material: m)
            }
            func glass(_ name: String, _ kind: String,
                       pos: [Double], rot: [Double] = [0, 0, 0],
                       params: [String: Double],
                       rough: Double, tint: RGB) -> SceneObject {
                var m = MaterialProps()
                m.glassOn = true
                m.transmission = 1.0
                m.roughness = rough
                m.ior = 1.5
                m.tint = tint
                return prim(kind, name, pos: pos, rot: rot,
                            params: params, material: m)
            }
            switch variant {
            case "convglass":
                // GLASS-heavy: three nested glass boxes around a
                // glowing core (paths cross up to 7 interfaces),
                // crossed clear panes, a frosted block, a glass
                // sphere and a deep-tinted glass cylinder.
                var objs = room(10, 8, 3.4, floorPreset: "concrete", floorScale: 2.0)
                objs.append(emitter("Audit ceiling light", "box",
                                    pos: [0, 3.32, -0.6],
                                    params: ["w": 2.4, "h": 0.06, "d": 1.2],
                                    color: RGB(1.0, 0.97, 0.9), intensity: 3.0))
                objs.append(emitter("Audit neon sign", "box",
                                    pos: [0.6, 2.05, -3.72],
                                    params: ["w": 2.4, "h": 0.8, "d": 0.1],
                                    color: RGB(1.0, 0.16, 0.5), intensity: 7.0))
                objs.append(glass("Audit glass outer", "box",
                                  pos: [0, 0.65, -0.8],
                                  params: ["w": 1.3, "h": 1.3, "d": 1.3],
                                  rough: 0.03, tint: RGB(0.90, 1.0, 0.94)))
                objs.append(glass("Audit glass middle", "box",
                                  pos: [0, 0.65, -0.8],
                                  params: ["w": 0.85, "h": 0.85, "d": 0.85],
                                  rough: 0.15, tint: RGB(1.0, 0.88, 0.70)))
                objs.append(glass("Audit glass inner", "box",
                                  pos: [0, 0.65, -0.8],
                                  params: ["w": 0.5, "h": 0.5, "d": 0.5],
                                  rough: 0.35, tint: RGB(0.75, 0.88, 1.0)))
                objs.append(emitter("Audit glass core", "box",
                                    pos: [0, 0.65, -0.8],
                                    params: ["w": 0.22, "h": 0.22, "d": 0.22],
                                    color: RGB(0.1, 0.9, 1.0), intensity: 5.0))
                objs.append(glass("Audit pane A", "box",
                                  pos: [2.3, 1.0, -0.7], rot: [0, 30, 0],
                                  params: ["w": 1.6, "h": 2.0, "d": 0.04],
                                  rough: 0.02, tint: RGB(0.95, 1.0, 0.98)))
                objs.append(glass("Audit pane B", "box",
                                  pos: [-2.3, 1.0, -0.7], rot: [0, -30, 0],
                                  params: ["w": 1.6, "h": 2.0, "d": 0.04],
                                  rough: 0.02, tint: RGB(0.95, 1.0, 0.98)))
                objs.append(glass("Audit frosted block", "box",
                                  pos: [-1.7, 0.3, 0.9],
                                  params: ["w": 0.6, "h": 0.6, "d": 0.6],
                                  rough: 0.6, tint: RGB(0.95, 0.98, 1.0)))
                objs.append(glass("Audit glass sphere", "sphere",
                                  pos: [1.9, 0.45, 1.3],
                                  params: ["r": 0.45],
                                  rough: 0.05, tint: RGB(0.96, 1.0, 0.97)))
                objs.append(glass("Audit green cylinder", "cylinder",
                                  pos: [-2.9, 0.55, -1.9],
                                  params: ["r": 0.3, "h": 1.1],
                                  rough: 0.08, tint: RGB(0.5, 0.95, 0.6)))
                var crateMat = MaterialProps()
                crateMat.baseColor = RGB(0.95, 0.45, 0.1)
                objs.append(prim("box", "Audit crate", pos: [3.0, 0.25, -2.6],
                                 rot: [0, -20, 0],
                                 params: ["w": 0.7, "h": 0.5, "d": 0.7],
                                 material: crateMat))
                p.scene.objects = objs
                p.scene.camera.target = [0, 0.85, -0.7]
                p.scene.camera.azimuth = 24
                p.scene.camera.elevation = 13
                p.scene.camera.distance = 6.8
            case "convrefl":
                // REFLECTION-heavy: a wet road-texture floor
                // (the audit forces wet = 1, so up-facing gloss
                // is 0.85 and the specular roll dominates),
                // tilted mirror panes, glossy road-texture
                // blocks, and bright emissive content to reflect.
                var objs = room(12, 10, 3.6, floorPreset: "road-normal", floorScale: 3.0)
                objs.append(emitter("Audit ceiling light", "box",
                                    pos: [0, 3.52, -0.5],
                                    params: ["w": 3.0, "h": 0.06, "d": 1.6],
                                    color: RGB(1.0, 0.97, 0.9), intensity: 3.0))
                objs.append(emitter("Audit neon sign", "box",
                                    pos: [0.8, 2.1, -4.72],
                                    params: ["w": 2.6, "h": 0.9, "d": 0.1],
                                    color: RGB(1.0, 0.16, 0.5), intensity: 7.0))
                objs.append(emitter("Audit glow cube", "box",
                                    pos: [-2.8, 0.18, -1.4],
                                    params: ["w": 0.36, "h": 0.36, "d": 0.36],
                                    color: RGB(0.1, 0.9, 1.0), intensity: 3.5))
                objs.append(glass("Audit mirror pane", "box",
                                  pos: [0, 1.5, -2.4], rot: [-14, 0, 0],
                                  params: ["w": 3.2, "h": 2.2, "d": 0.04],
                                  rough: 0.03, tint: RGB(0.96, 1.0, 0.98)))
                objs.append(glass("Audit side pane", "box",
                                  pos: [3.6, 1.3, -0.6], rot: [0, -38, 0],
                                  params: ["w": 2.4, "h": 2.0, "d": 0.04],
                                  rough: 0.03, tint: RGB(0.96, 1.0, 0.98)))
                objs.append(glass("Audit glass slab", "box",
                                  pos: [-2.6, 0.25, 1.4], rot: [0, 18, 0],
                                  params: ["w": 2.0, "h": 0.5, "d": 1.2],
                                  rough: 0.1, tint: RGB(0.94, 0.99, 1.0)))
                var roadMat = MaterialProps()
                roadMat.texturePreset = "road-normal"
                roadMat.textureScale = 1.5
                objs.append(prim("box", "Audit road block", pos: [2.4, 0.35, 1.6],
                                 rot: [0, -12, 0],
                                 params: ["w": 1.4, "h": 0.7, "d": 1.4],
                                 material: roadMat))
                var pillarMat = MaterialProps()
                pillarMat.texturePreset = "concrete"
                pillarMat.textureScale = 1.5
                objs.append(prim("cylinder", "Audit pillar", pos: [-3.4, 0.8, -2.6],
                                 params: ["r": 0.35, "h": 1.6],
                                 material: pillarMat))
                p.scene.objects = objs
                p.scene.camera.target = [0, 0.8, -0.9]
                p.scene.camera.azimuth = 18
                p.scene.camera.elevation = 7
                p.scene.camera.distance = 7.4
            default: // convlight
                // LIGHT-heavy: 18 emissive objects spanning a
                // huge area x intensity range — two 12.9 m2 dim
                // ceiling panels (intensity 0.7-0.9) vs 0.08 m
                // cubes at intensity 25. Uniform-by-area light
                // selection wastes nearly every pick on the dim
                // panels; the receivers are plain diffuse.
                var objs = room(10, 8, 3.4, floorPreset: "concrete", floorScale: 2.0)
                objs.append(emitter("Audit panel A", "box",
                                    pos: [-1.8, 3.32, -1.0],
                                    params: ["w": 3.0, "h": 0.08, "d": 2.0],
                                    color: RGB(1.0, 0.9, 0.75), intensity: 0.7))
                objs.append(emitter("Audit panel B", "box",
                                    pos: [2.0, 3.32, 0.8],
                                    params: ["w": 3.0, "h": 0.08, "d": 2.0],
                                    color: RGB(0.8, 0.9, 1.0), intensity: 0.9))
                objs.append(emitter("Audit wall panel 1", "box",
                                    pos: [-4.78, 1.9, -1.5], rot: [0, 90, 0],
                                    params: ["w": 1.2, "h": 0.8, "d": 0.06],
                                    color: RGB(1.0, 0.55, 0.15), intensity: 3.0))
                objs.append(emitter("Audit wall panel 2", "box",
                                    pos: [4.78, 1.9, 0.5], rot: [0, -90, 0],
                                    params: ["w": 1.2, "h": 0.8, "d": 0.06],
                                    color: RGB(0.3, 1.0, 0.45), intensity: 3.5))
                objs.append(emitter("Audit wall panel 3", "box",
                                    pos: [-1.5, 2.2, -3.78],
                                    params: ["w": 1.4, "h": 0.7, "d": 0.06],
                                    color: RGB(0.35, 0.5, 1.0), intensity: 4.0))
                objs.append(emitter("Audit wall panel 4", "box",
                                    pos: [2.2, 1.6, -3.78],
                                    params: ["w": 1.0, "h": 0.7, "d": 0.06],
                                    color: RGB(1.0, 0.3, 0.8), intensity: 2.5))
                let cubeCols: [RGB] = [RGB(1.0, 0.25, 0.2), RGB(0.1, 0.9, 1.0),
                                       RGB(1.0, 0.85, 0.2), RGB(0.3, 1.0, 0.4),
                                       RGB(0.3, 0.45, 1.0), RGB(1.0, 0.4, 0.75)]
                let cubePos: [[Double]] = [[-3.2, 0.09, -0.6], [-1.2, 0.09, 1.6],
                                           [0.8, 0.09, -1.8], [2.6, 0.09, 0.9],
                                           [3.6, 0.09, -2.4], [-2.4, 0.09, 2.6]]
                for i in 0..<6 {
                    objs.append(emitter("Audit bright cube \(i)", "box",
                                        pos: cubePos[i],
                                        params: ["w": 0.18, "h": 0.18, "d": 0.18],
                                        color: cubeCols[i], intensity: 12.0))
                }
                let stripPos: [[Double]] = [[0, 0.06, -3.5], [-4.5, 0.06, 0.5],
                                            [4.5, 0.06, -0.8], [1.5, 0.06, 3.4]]
                let stripRot: [[Double]] = [[0, 0, 0], [0, 90, 0], [0, 90, 0], [0, 0, 0]]
                for i in 0..<4 {
                    objs.append(emitter("Audit strip \(i)", "box",
                                        pos: stripPos[i], rot: stripRot[i],
                                        params: ["w": 1.6, "h": 0.12, "d": 0.12],
                                        color: RGB(0.9, 0.95, 1.0), intensity: 5.0))
                }
                objs.append(emitter("Audit tiny cube 1", "box",
                                    pos: [-0.5, 0.04, 0.2],
                                    params: ["w": 0.08, "h": 0.08, "d": 0.08],
                                    color: RGB(1.0, 0.95, 0.85), intensity: 25.0))
                objs.append(emitter("Audit tiny cube 2", "box",
                                    pos: [1.8, 0.04, -0.4],
                                    params: ["w": 0.08, "h": 0.08, "d": 0.08],
                                    color: RGB(0.85, 0.92, 1.0), intensity: 25.0))
                var crateMat = MaterialProps()
                crateMat.baseColor = RGB(0.95, 0.45, 0.1)
                objs.append(prim("box", "Audit crate", pos: [2.9, 0.25, -2.5],
                                 rot: [0, -20, 0],
                                 params: ["w": 0.7, "h": 0.5, "d": 0.7],
                                 material: crateMat))
                var sphereMat = MaterialProps()
                sphereMat.texturePreset = "cement"
                sphereMat.textureScale = 1.2
                objs.append(prim("sphere", "Audit sphere", pos: [-3.5, 0.42, 1.6],
                                 params: ["r": 0.42], material: sphereMat))
                var pillarMat = MaterialProps()
                pillarMat.texturePreset = "concrete"
                pillarMat.textureScale = 1.5
                objs.append(prim("cylinder", "Audit pillar", pos: [-2.9, 0.48, -2.3],
                                 params: ["r": 0.35, "h": 0.95],
                                 material: pillarMat))
                p.scene.objects = objs
                p.scene.camera.target = [0, 1.0, -0.5]
                p.scene.camera.azimuth = 30
                p.scene.camera.elevation = 16
                p.scene.camera.distance = 7.2
            }
        default:
            break
        }
        return p
    }
}

/// One repro run. Lives on the main actor (EditorSession is
/// @MainActor-isolated; the pump is a self-rescheduling
/// asyncAfter chain on the main queue so the runloop keeps
/// spinning between frames — Metal completion handlers and
/// the renderer's main-queue weather-step updates depend on
/// it). The DispatchQueue hops re-enter via assumeIsolated,
/// which is exact: asyncAfter on .main runs on the main actor.
@MainActor
final class ReproRun {
    private var session: EditorSession?
    private var renderer: Renderer?
    private var view: MTKView?
    private var pumpCount = 0
    private var startedAt = CACurrentMediaTime()
    private var lastStepTag = ""
    private var firstStepCrossed = false
    private var weatherOn = false
    private var crossedA = false
    private var finished = false

    // Gravity-audit state (v4.13 variants gravfall / gravstack).
    // Phase 0: weather off through the real funnel (start()
    // forced snow on), save/load audit on the pristine project,
    // a beat for the mode-0 rebuild, then Play via toggleMode.
    // Phase 1: sample the tracked boxes' centre heights once
    // per pump until the variant's time limit, then verdicts.
    private var isGravityVariant: Bool {
        ["gravfall", "gravstack"].contains(ReproDriver.variant)
    }
    private var isPerfVariant: Bool { ReproDriver.variant == "perfbench" }
    private var gravPhase = 0
    private var gravPrepared = false
    private var gravPhaseSince: CFTimeInterval = 0
    private var gravPlayStart: CFTimeInterval = 0
    private var gravFallSamples: [(t: Double, y: Double)] = []
    private var gravAuxSamples: [(t: Double, y: Double)] = []
    private var gravCtlSamples: [(t: Double, y: Double)] = []

    private func gravY(_ name: String) -> Double? {
        session?.project.scene.objects
            .first { $0.name == name }?.transform.position[1]
    }

    private func handleGravityAudit() {
        guard let session else { return }
        let now = CACurrentMediaTime()
        switch gravPhase {
        case 0:
            if !gravPrepared {
                gravPrepared = true
                session.setWeatherMode(0)
                runGravitySaveLoadChecks()
                gravPhaseSince = now
                return
            }
            if now - gravPhaseSince >= 0.6 {
                session.toggleMode()
                gravPlayStart = CACurrentMediaTime()
                gravPhase = 1
                ReproDriver.marker("gravity-play-entered mode=\(session.mode)")
            }
        case 1:
            let t = now - gravPlayStart
            let primary = ReproDriver.variant == "gravfall"
                ? "Repro fall box" : "Repro stack faller"
            let aux = ReproDriver.variant == "gravfall"
                ? "Repro float box" : "Repro riser box"
            if let y = gravY(primary) { gravFallSamples.append((t, y)) }
            if let y = gravY(aux) { gravAuxSamples.append((t, y)) }
            if ReproDriver.variant == "gravfall",
               let y = gravY("Repro control box") {
                gravCtlSamples.append((t, y))
            }
            let limit = ReproDriver.variant == "gravfall" ? 5.0 : 6.5
            if t >= limit {
                gravPhase = 2
                if ReproDriver.variant == "gravfall" {
                    emitGravfallChecks()
                } else {
                    emitGravstackChecks()
                }
                finishSnow()
            }
        default:
            break
        }
    }

    /// Save/load audit (both gravity variants, run pre-Play on
    /// the pristine project): the project must round-trip
    /// through the REAL ProjectStore with its Applied Gravity
    /// nodes intact (kind, strength, object refs), and the same
    /// project with the gravity nodes stripped — a v4.12-era
    /// file — must still load (the new kind is additive).
    private func runGravitySaveLoadChecks() {
        guard let session else { return }
        let project = session.project
        let expected: [Double] = ReproDriver.variant == "gravfall"
            ? [9.8, 0] : [9.8, -9.8]
        ProjectStore.saveProject(project, thumbnailPNG: nil)
        let loaded = ProjectStore.loadProject(id: project.id)
        let grav = loaded?.scene.graph.nodes.filter { $0.kind == "actGravity" } ?? []
        let strengths = grav.map { $0.num("strength", -999) }.sorted()
        let refsOK = grav.allSatisfy { node in
            guard let ref = node.ref("object") else { return false }
            return loaded?.scene.object(id: ref) != nil
        }
        check("saveload",
              loaded != nil && grav.count == expected.count
                  && strengths == expected.sorted() && refsOK,
              "nodes=\(grav.count) strengths=\(strengths) refsOK=\(refsOK)")
        var legacy = project
        legacy.id = UUID()
        let gravIDs = Set(project.scene.graph.nodes
            .filter { $0.kind == "actGravity" }.map { $0.id })
        legacy.scene.graph.nodes.removeAll { $0.kind == "actGravity" }
        legacy.scene.graph.links.removeAll {
            gravIDs.contains($0.from) || gravIDs.contains($0.to)
        }
        ProjectStore.saveProject(legacy, thumbnailPNG: nil)
        let legacyLoaded = ProjectStore.loadProject(id: legacy.id)
        let legacyGrav = legacyLoaded?.scene.graph.nodes
            .filter { $0.kind == "actGravity" }.count ?? -1
        check("legacyload",
              legacyLoaded != nil && legacyGrav == 0,
              "loaded=\(legacyLoaded != nil) gravNodes=\(legacyGrav)")
    }

    /// gravfall verdicts: (a) the fall is accelerating (the
    /// half-time fallen fraction sits in the quadratic band,
    /// not the linear 0.5), lands in the analytic time window
    /// with its base on the plane, and stays; (c) the strength-0
    /// box hangs; (e) the unwired control never moves.
    private func emitGravfallChecks() {
        let s = gravFallSamples
        guard s.count >= 10 else {
            check("samples", false, "only \(s.count) samples")
            return
        }
        // Landing = first sample at rest height (centre 0.5).
        let landIdx = s.firstIndex { $0.y <= 0.515 }
        check("landed", landIdx != nil,
              "samples=\(s.count) finalY=\(s.last?.y ?? -1)")
        guard let li = landIdx else { return }
        let landT = s[li].t
        // Analytic fall time for 3 m at 9.8 m/s² is 0.782 s;
        // the band absorbs pump granularity and Euler error.
        check("falltime", landT >= 0.55 && landT <= 1.10,
              String(format: "landT=%.3f analytic=0.782", landT))
        // Curve: fallen fraction at half the fall duration.
        // Quadratic ≈ 0.25; a linear fall would read 0.50.
        let halfT = landT / 2
        let halfSample = s.first { $0.t >= halfT } ?? s[li]
        let frac = (3.5 - halfSample.y) / 3.0
        check("curve", frac >= 0.13 && frac <= 0.42,
              String(format: "halfTimeFrac=%.3f quadratic=0.25 linear=0.50", frac))
        // Monotone descent until landing (no upward steps).
        var monotone = true
        if li > 0 {
            for i in 1...li where s[i].y > s[i - 1].y + 0.002 { monotone = false }
        }
        check("monotone", monotone, "checked \(li) fall samples")
        // Rest: exact height, and the final second is still.
        let rest = s.filter { $0.t >= landT + 0.3 }
        let restOK = !rest.isEmpty
            && rest.allSatisfy { abs($0.y - 0.5) <= 0.01 }
        let tail = s.filter { $0.t >= (s.last?.t ?? 0) - 1.0 }
        let tailSpan = (tail.map { $0.y }.max() ?? 0) - (tail.map { $0.y }.min() ?? 0)
        check("rest", restOK && tailSpan <= 0.002,
              String(format: "restY=%.4f tailSpan=%.4f", s.last?.y ?? -1, tailSpan))
        // (c) strength 0 hangs motionless at its placed height.
        let floatOK = !gravAuxSamples.isEmpty
            && gravAuxSamples.allSatisfy { abs($0.y - 3.5) <= 0.001 }
        check("float", floatOK,
              "samples=\(gravAuxSamples.count) finalY=\(gravAuxSamples.last?.y ?? -1)")
        // (e) the unwired control object never moves.
        let ctlOK = !gravCtlSamples.isEmpty
            && gravCtlSamples.allSatisfy { abs($0.y - 3.5) <= 0.001 }
        check("control", ctlOK,
              "samples=\(gravCtlSamples.count) finalY=\(gravCtlSamples.last?.y ?? -1)")
    }

    /// gravstack verdicts: (b) the faller rests on the lower
    /// box's top (centre 1.5) before the graph hides that box
    /// at 2.5 s, then falls again and rests on the plane
    /// (centre 0.5); (d) the riser under −9.8 rises and rests
    /// against the ceiling box's underside (centre 2.0).
    private func emitGravstackChecks() {
        guard let session else { return }
        let s = gravFallSamples
        // Sustained rest on the box top before the hide fires.
        var runLength = 0
        var onBox = false
        for sample in s where sample.t < 2.4 {
            if abs(sample.y - 1.5) <= 0.02 {
                runLength += 1
                if runLength >= 3 { onBox = true }
            } else {
                runLength = 0
            }
        }
        check("landonbox", onBox, "samples=\(s.count)")
        // After the support is hidden it falls again to the
        // plane and stays there.
        let late = s.filter { $0.t >= 4.5 }
        let refellOK = !late.isEmpty
            && late.allSatisfy { abs($0.y - 0.5) <= 0.01 }
        check("refell", refellOK,
              "late=\(late.count) finalY=\(s.last?.y ?? -1)")
        if let lower = session.project.scene.objects
            .first(where: { $0.name == "Repro lower box" }) {
            check("supporthidden", !session.runtimeIsObjectVisible(lower.id),
                  "lower visible at end=\(session.runtimeIsObjectVisible(lower.id))")
        } else {
            check("supporthidden", false, "lower box missing")
        }
        // The riser rose (past 1.9 by 2 s) and rests against
        // the underside (centre 2.0) for the rest of the run.
        let rose = gravAuxSamples.contains { $0.t <= 2.0 && $0.y >= 1.9 }
        let riserLate = gravAuxSamples.filter { $0.t >= 2.0 }
        let riserRestOK = !riserLate.isEmpty
            && riserLate.allSatisfy { abs($0.y - 2.0) <= 0.01 }
        check("riser", rose && riserRestOK,
              "rose=\(rose) finalY=\(gravAuxSamples.last?.y ?? -1)")
        // Tick-alive witness: the ceiling box must have been
        // hidden by the 4 s Delay branch — if the runtime
        // stopped ticking after the 2.5 s hide, it is still
        // visible at the end of the run.
        if let ceil = session.project.scene.objects
            .first(where: { $0.name == "Repro ceiling box" }) {
            check("tickwitness", !session.runtimeIsObjectVisible(ceil.id),
                  "ceiling visible at end=\(session.runtimeIsObjectVisible(ceil.id))")
        }
    }

    // MARK: - v4.14 audits (loopmove / playernodes)
    //
    // Same phase shape as the gravity audits: phase 0 turns
    // the weather off through the real funnel, runs the
    // save/load audit on the pristine project, waits a beat,
    // then enters Play via toggleMode. Phase 1 samples once
    // per pump until the variant's limit, then verdicts.
    // loopmove continues: phase 2 exits Play and samples the
    // restored scene for stillness (a leaked loop would keep
    // writing — or the restore would not hold); phase 3
    // re-enters Play and confirms the loop starts fresh from
    // the authored position (no state leaked across sessions).
    private var isLoopVariant: Bool {
        ["loopmove", "playernodes"].contains(ReproDriver.variant)
    }
    private var loopPhase = 0
    private var loopPrepared = false
    private var loopPhaseSince: CFTimeInterval = 0
    private var loopPlayStart: CFTimeInterval = 0
    private var loopMarchSamples: [(t: Double, x: Double)] = []
    private var loopSpinTotal: [(t: Double, deg: Double)] = []
    private var loopSpinLastYaw: Double? = nil
    private var loopSpinAccum: Double = 0
    private var loopMaxStep: Double = 0
    private var loopPosSamples: [(t: Double, x: Double, y: Double, z: Double)] = []
    private var loopYawSamples: [(t: Double, deg: Double)] = []
    private var loopYawLast: Double? = nil
    private var loopYawAccum: Double = 0
    private var loopDragApplied = false
    private var loopExitSamples: [Double] = []

    /// Accumulates a wrapped 0..<360 angle into an unwrapped
    /// total (per-step deltas folded into −180...180; the
    /// per-pump steps here are a few degrees, far from the
    /// fold).
    private func unwrap(_ raw: Double, last: inout Double?,
                        accum: inout Double) -> Double {
        if let prev = last {
            var d = raw - prev
            while d > 180 { d -= 360 }
            while d < -180 { d += 360 }
            accum += d
        }
        last = raw
        return accum
    }

    private func loopX(_ name: String) -> Double? {
        session?.project.scene.objects
            .first { $0.name == name }?.transform.position[0]
    }

    private func handleLoopAudit() {
        guard let session else { return }
        let now = CACurrentMediaTime()
        switch loopPhase {
        case 0:
            if !loopPrepared {
                loopPrepared = true
                session.setWeatherMode(0)
                runLoopSaveLoadChecks()
                loopPhaseSince = now
                return
            }
            if now - loopPhaseSince >= 0.6 {
                session.toggleMode()
                loopPlayStart = CACurrentMediaTime()
                loopPhase = 1
                ReproDriver.marker("loop-play-entered mode=\(session.mode)")
            }
        case 1:
            let t = now - loopPlayStart
            if ReproDriver.variant == "loopmove" {
                if let x = loopX("Repro march box") {
                    if let prev = loopMarchSamples.last {
                        loopMaxStep = max(loopMaxStep, abs(x - prev.x))
                    }
                    loopMarchSamples.append((t, x))
                }
                if let spin = session.project.scene.objects
                    .first(where: { $0.name == "Repro spin box" }),
                   spin.transform.rotation.count > 1 {
                    let total = unwrap(spin.transform.rotation[1],
                                       last: &loopSpinLastYaw,
                                       accum: &loopSpinAccum)
                    loopSpinTotal.append((t, total))
                }
                if t >= 4.6 {
                    emitLoopmoveChecks()
                    session.toggleMode()
                    loopPhaseSince = now
                    loopPhase = 2
                }
            } else {   // playernodes
                let cam = session.project.scene.camera
                if cam.playPosition.count >= 3 {
                    loopPosSamples.append((t, cam.playPosition[0],
                                           cam.playPosition[1],
                                           cam.playPosition[2]))
                }
                let total = unwrap(cam.playYaw, last: &loopYawLast,
                                   accum: &loopYawAccum)
                loopYawSamples.append((t, total))
                if t >= 3.6, !loopDragApplied {
                    loopDragApplied = true
                    // The Viewport drag-look gesture's exact
                    // write path: mutate the camera value and
                    // assign it back (Viewport.swift).
                    var dragged = session.project.scene.camera
                    dragged.playYaw += 30
                    session.project.scene.camera = dragged
                    ReproDriver.marker("loop-drag-applied yaw=\(dragged.playYaw)")
                }
                if t >= 10.2 {
                    emitPlayernodeChecks()
                    finishSnow()
                }
            }
        case 2:
            // loopmove, Play exited: the snapshot restore has
            // returned the march box to its authored x = 0.
            if let x = loopX("Repro march box") {
                loopExitSamples.append(x)
            }
            if now - loopPhaseSince >= 1.2 {
                let still = !loopExitSamples.isEmpty
                    && loopExitSamples.allSatisfy { abs($0) <= 0.001 }
                check("postexit", still,
                      "samples=\(loopExitSamples.count) last=\(loopExitSamples.last ?? -999)")
                session.toggleMode()
                loopPlayStart = CACurrentMediaTime()
                loopMarchSamples.removeAll()
                loopPhase = 3
            }
        case 3:
            let t = now - loopPlayStart
            if let x = loopX("Repro march box") {
                loopMarchSamples.append((t, x))
            }
            if t >= 1.7 {
                let x = loopMarchSamples.last?.x ?? -999
                check("restart", x >= 1.4 && x <= 3.1,
                      String(format: "x@1.7s=%.3f (fresh session: cycle 1 done, cycle 2 under way)", x))
                finishSnow()
            }
        default:
            break
        }
    }

    /// loopmove verdicts: the marched box completes at least
    /// three full +1.5 m cycles by 4.6 s (0.8 s tween + 0.25 s
    /// delay per cycle), gliding — never jumping; the spun box
    /// accumulates at least four 60° cycles (0.5 s at 120°/s +
    /// 0.4 s delay per cycle).
    private func emitLoopmoveChecks() {
        let s = loopMarchSamples
        guard s.count >= 20 else {
            check("samples", false, "only \(s.count) samples")
            return
        }
        let finalX = s.last?.x ?? -999
        check("movecycles", finalX >= 4.4 && finalX <= 7.6,
              String(format: "finalX=%.3f (>=3 cycles of +1.5 m)", finalX))
        check("glide", loopMaxStep <= 0.25,
              String(format: "maxStep=%.3f (a teleporting loop would jump 1.5)", loopMaxStep))
        let spin = loopSpinTotal.last?.deg ?? -999
        check("spincycles", spin >= 230 && spin <= 400,
              String(format: "unwrapped=%.1f deg (>=4 cycles of 60)", spin))
    }

    /// playernodes verdicts: the player glide lands on the
    /// eye-rule destination and stays; the camera rotation is
    /// gradual, lands at +90 and HOLDS it; the audit's own +30
    /// drag survives untouched (no snap-back); and the Loop ×
    /// Player Camera composition keeps turning the view at
    /// ~90° per 1 s cycle.
    private func emitPlayernodeChecks() {
        func posNear(_ t0: Double, _ t1: Double) -> Bool {
            let win = loopPosSamples.filter { $0.t >= t0 && $0.t <= t1 }
            return !win.isEmpty && win.allSatisfy {
                abs($0.x - 4) <= 0.06 && abs($0.y - 2.2) <= 0.06
                    && abs($0.z) <= 0.06
            }
        }
        check("playermove", posNear(1.7, 2.1),
              "samples=\(loopPosSamples.count) dest=(4, 2.2, 0) by the teleport eye rule")
        let lastPos = loopPosSamples.last
        check("playerstays", posNear(9.6, 10.2),
              "final=\(lastPos.map { "(\($0.x), \($0.y), \($0.z))" } ?? "none")")
        let mid = loopYawSamples.filter { $0.t >= 2.35 && $0.t <= 2.65 }
        let midOK = !mid.isEmpty
            && mid.allSatisfy { $0.deg >= 25 && $0.deg <= 65 }
        check("camspin", midOK,
              String(format: "midYaw=%.1f (half of the 90-degree turn)", mid.last?.deg ?? -999))
        func yawNear(_ t0: Double, _ t1: Double, _ want: Double) -> (Bool, Double) {
            let win = loopYawSamples.filter { $0.t >= t0 && $0.t <= t1 }
            guard let last = win.last else { return (false, -999) }
            return (win.allSatisfy { abs($0.deg - want) <= 5 }, last.deg)
        }
        let (rotOK, rotYaw) = yawNear(3.15, 3.5, 90)
        check("camrotate", rotOK,
              String(format: "yaw=%.2f want=90", rotYaw))
        let (dragOK, dragYaw) = yawNear(4.2, 5.2, 120)
        check("nosnapback", dragOK,
              String(format: "yaw=%.2f want=120 (node's 90 + the audit's own +30 drag)", dragYaw))
        let a = loopYawSamples.last { $0.t <= 5.9 }?.deg ?? 0
        let b = loopYawSamples.last { $0.t <= 9.9 }?.deg ?? 0
        check("loopcam", b - a >= 250 && b - a <= 430,
              String(format: "gain=%.1f deg over 4 s (~90 per 1 s cycle)", b - a))
    }

    /// Save/load audit (both loop variants, pre-Play on the
    /// pristine project), mirroring the gravity audit's shape:
    /// the v4.14 graph round-trips through the REAL
    /// ProjectStore with its Loop nodes, subject params and
    /// refs intact; a pre-v4.14 file (Loop nodes stripped —
    /// and, for playernodes, the subject params too, which the
    /// catalogue default merge then supplies as "Object") still
    /// loads.
    private func runLoopSaveLoadChecks() {
        guard let session else { return }
        let project = session.project
        ProjectStore.saveProject(project, thumbnailPNG: nil)
        let loaded = ProjectStore.loadProject(id: project.id)
        let nodes = loaded?.scene.graph.nodes ?? []
        if ReproDriver.variant == "loopmove" {
            let loops = nodes.filter { $0.kind == "logLoop" }
            let delays = loops.map { $0.num("delay", -1) }.sorted()
            let refsOK = loops.allSatisfy { loop in
                guard let ref = loop.ref("node") else { return false }
                return nodes.contains { $0.id == ref }
            }
            let move = nodes.first { $0.kind == "actMoveTo" }
            let rotate = nodes.first { $0.kind == "actRotate" }
            check("saveload",
                  loaded != nil && loops.count == 2 && refsOK
                      && delays == [0.25, 0.4]
                      && move?.num("dx", 0) == 1.5
                      && move?.txt("subject", "") == "Object"
                      && rotate?.num("speed", 0) == 120
                      && rotate?.txt("subject", "") == "Object",
                  "loops=\(loops.count) delays=\(delays) refsOK=\(refsOK)")
        } else {
            let loops = nodes.filter { $0.kind == "logLoop" }
            let refsOK = loops.allSatisfy { loop in
                guard let ref = loop.ref("node") else { return false }
                return nodes.contains { $0.id == ref }
            }
            let move = nodes.first { $0.kind == "actMoveTo" }
            let rotates = nodes.filter { $0.kind == "actRotate" }
            let targetOK = move.flatMap { $0.ref("target") }
                .map { id in loaded?.scene.object(id: id) != nil } ?? false
            check("saveload",
                  loaded != nil && loops.count == 1 && refsOK && targetOK
                      && move?.txt("subject", "") == "Player"
                      && rotates.count == 2
                      && rotates.allSatisfy { $0.txt("subject", "") == "Player Camera" },
                  "loops=\(loops.count) refsOK=\(refsOK) targetOK=\(targetOK)")
        }
        var legacy = project
        legacy.id = UUID()
        let loopIDs = Set(project.scene.graph.nodes
            .filter { $0.kind == "logLoop" }.map { $0.id })
        legacy.scene.graph.nodes.removeAll { $0.kind == "logLoop" }
        legacy.scene.graph.links.removeAll {
            loopIDs.contains($0.from) || loopIDs.contains($0.to)
        }
        if ReproDriver.variant == "playernodes" {
            for i in legacy.scene.graph.nodes.indices {
                legacy.scene.graph.nodes[i].params["subject"] = nil
            }
        }
        ProjectStore.saveProject(legacy, thumbnailPNG: nil)
        let legacyLoaded = ProjectStore.loadProject(id: legacy.id)
        let legacyLoops = legacyLoaded?.scene.graph.nodes
            .filter { $0.kind == "logLoop" }.count ?? -1
        check("legacyload",
              legacyLoaded != nil && legacyLoops == 0,
              "loaded=\(legacyLoaded != nil) loopNodes=\(legacyLoops)")
    }

    /// LS-REPRO metrics: computed from the renderer's sim
    /// snapshot. Surface height = support + snow at each
    /// lattice vertex. Emits exactly one REPRO-METRIC line.

    // MARK: perfbench — controlled performance baseline (rain
    // scene; valid on every build): frameTick CPU ms at rest
    // with rain active, and the wall time of a geometry edit's
    // staged rebuild, on the simulator's host CPU. Metrics only
    // — device FPS stays the device's verdict.
    private var perfPhase = 0
    private var perfFrameSamples: [Double] = []
    private var perfEditsDone = 0
    private var perfEditStart: CFTimeInterval = 0
    private var perfEditInFlight = false
    private var perfEditTimes: [Double] = []

    private func perfMetric(_ name: String, _ values: [Double]) {
        guard !values.isEmpty else {
            ReproDriver.marker("PERF \(name) n=0")
            return
        }
        let mean = values.reduce(0, +) / Double(values.count)
        let mx = values.max() ?? 0
        ReproDriver.marker(String(format: "PERF %@ meanMs=%.3f maxMs=%.3f n=%d",
                                  name, mean, mx, values.count))
    }

    private func pumpPerf() {
        guard let session else { return }
        switch perfPhase {
        case 0:
            if session.project.scene.weather.mode != 1 {
                session.setWeatherMode(1)
            }
            if session.liveWetAmount > 0.05 {
                perfFrameSamples.removeAll()
                perfPhase = 1
                ReproDriver.marker("perfbench rain-established wet=\(session.liveWetAmount)")
            }
        case 1:
            if perfFrameSamples.count >= 90 {
                perfMetric("rainRestFrameTick", perfFrameSamples)
                perfFrameSamples.removeAll()
                perfPhase = 2
            }
        case 2:
            let renderer = session.renderer
            if !perfEditInFlight {
                let boxes = session.project.scene.objects.indices.filter {
                    session.project.scene.objects[$0].primitiveKind == "box"
                }
                guard perfEditsDone < 6, !boxes.isEmpty else { perfPhase = 3; return }
                let idx = boxes[perfEditsDone % boxes.count]
                session.project.scene.objects[idx].transform.position[0] += 0.5
                session.scheduleRebuild(geometry: true)
                perfEditStart = CACurrentMediaTime()
                perfEditInFlight = true
            } else if renderer.weatherStep == "rebuild:swap"
                        || CACurrentMediaTime() - perfEditStart > 20 {
                perfEditTimes.append((CACurrentMediaTime() - perfEditStart) * 1000)
                perfEditsDone += 1
                perfEditInFlight = false
                if perfEditsDone >= 6 {
                    perfMetric("rainEditRebuild", perfEditTimes)
                    perfPhase = 3
                }
            }
        default:
            // Save/load sanity on the surviving weather model:
            // the project round-trips with rain mode intact.
            let project = session.project
            ProjectStore.saveProject(project, thumbnailPNG: nil)
            let loaded = ProjectStore.loadProject(id: project.id)
            check("saveload", loaded?.scene.weather.mode == 1,
                  "mode=\(String(describing: loaded?.scene.weather.mode))")
            finishSnow()
        }
    }

    private func finishSnow() {
        finished = true
        print("REPRO-DONE")
        fflush(stdout)
        NSLog("REPRO-DONE")
        exit(0)
    }

    // MARK: v4.15 convergence audit (convglass / convrefl /
    // convlight): weather off, a short settle for the scene
    // upload, then three offscreen renders of the variant's
    // reference scene through Renderer.renderAuditStats (fixed
    // 768 samples, still settings). Stats come from the first
    // render; the PERF line reports the trace wall times of the
    // following renders (host-measured on the CI runner).
    private var isAuditVariant: Bool {
        ["convglass", "convrefl", "convlight"].contains(ReproDriver.variant)
    }
    private var auditPhase = 0

    /// v4.15 baselines — settled-image statistics from the
    /// PRE-optimisation engine, captured on a ray-tracing-
    /// capable host (the CI simulator has no Metal RT support
    /// and skips the audit; see handleAudit). When an entry
    /// exists, the current engine must reproduce its region
    /// means within max(4% relative, 0.004 absolute) per region
    /// and 3% overall: the settled picture does not move. The
    /// table ships empty for v4.15 — the first RT-host run of
    /// the audit prints the AUDIT lines that become the
    /// capture; the settled-image invariance for v4.15 itself
    /// rests on the estimator algebra documented in
    /// VERSION-HISTORY §4.15 (every change is exact-identical
    /// or a compensated, provably unbiased roulette).
    static let auditBaselines: [String: (overall: Double, regions: [Double])] = [:]

    private func handleAudit() {
        guard let session, let renderer else { return }
        switch auditPhase {
        case 0:
            session.setWeatherMode(0)
            if pumpCount >= 45 { auditPhase = 1 }
        case 1:
            // Environment gate: the GitHub CI simulator reports
            // no Metal ray-tracing support, so no acceleration
            // structure is ever built there and the tracer
            // cannot run (the rest of the matrix is a logic
            // matrix and never needed it). On such hosts the
            // audit SKIPs explicitly; on any RT-capable host it
            // runs in full and the AUDIT lines below are the
            // numeric record.
            if !renderer.rayTracingSupported {
                skipCheck("conv", "no ray-tracing support on this host — audit runs on RT-capable hosts only")
                skipCheck("adaptimg", "no ray-tracing support on this host")
                ReproDriver.marker("AUDIT-SKIP \(ReproDriver.variant) rayTracingSupported=false")
                finishSnow()
                return
            }
            let camera = session.project.scene.camera.orbitBasis()
            let savedWet = renderer.weatherWet
            if ReproDriver.variant == "convrefl" { renderer.weatherWet = 1 }
            var walls: [Double] = []
            var stats: (wallMs: Double, regions: [Double], overall: Double)?
            for i in 0..<4 {
                let r = renderer.renderAuditStats(camera: camera, width: 384,
                                                  height: 256, samples: 768,
                                                  seedStart: 0)
                if let r {
                    if i == 0 { stats = r } else { walls.append(r.wallMs) }
                }
            }
            guard let stats else {
                check("render", false, "renderAuditStats returned nil")
                renderer.weatherWet = savedWet
                finishSnow()
                return
            }
            ReproDriver.marker(String(format: "AUDIT %@ overall=%.6f regions=%@",
                ReproDriver.variant, stats.overall,
                stats.regions.map { String(format: "%.6f", $0) }
                    .joined(separator: ",")))
            perfMetric("auditTrace768", walls)
            // v4.15 convergence audit: settled output vs the
            // pre-optimisation baseline, region by region.
            if let base = ReproDriver.auditBaselines[ReproDriver.variant] {
                var worst = 0.0
                for i in 0..<min(base.regions.count, stats.regions.count) {
                    let tol = max(0.04 * base.regions[i], 0.004)
                    worst = max(worst, abs(stats.regions[i] - base.regions[i]) / tol)
                }
                let overallDev = abs(stats.overall - base.overall)
                    / max(0.03 * base.overall, 0.004)
                check("conv", worst <= 1.0 && overallDev <= 1.0,
                      String(format: "worst region %.2f of tol, overall %.2f of tol (baseline overall %.6f)",
                             worst, overallDev, base.overall))
            } else {
                skipCheck("conv", "no Step A baseline captured yet — the AUDIT line above is the capture for this host class")
            }
            // v4.15 adaptive-sampling A/B on the LIVE path: gate
            // ON vs gate OFF (moments tracked in both arms), same
            // camera, fresh accumulation per arm. The two images
            // must agree within the same tolerance; both race
            // times are reported for the delivery record.
            if let on = renderer.renderAuditLive(camera: camera, width: 384,
                                                 height: 256, adaptive: true),
               let off = renderer.renderAuditLive(camera: camera, width: 384,
                                                  height: 256, adaptive: false) {
                var worstAB = 0.0
                for i in 0..<min(on.regions.count, off.regions.count) {
                    let tol = max(0.04 * off.regions[i], 0.004)
                    worstAB = max(worstAB, abs(on.regions[i] - off.regions[i]) / tol)
                }
                check("adaptimg", worstAB <= 1.0,
                      String(format: "worst region %.2f of tol (on: %.0f%% converged in %d dispatches; off: %.0f%% in %d)",
                             worstAB, on.convergedFrac * 100, on.dispatches,
                             off.convergedFrac * 100, off.dispatches))
                perfMetric("adaptOnMs", [on.wallMs])
                perfMetric("adaptOffMs", [off.wallMs])
            } else {
                check("adaptimg", false, "renderAuditLive returned nil")
            }
            renderer.weatherWet = savedWet
            finishSnow()
        default:
            break
        }
    }

    /// One named audit verdict line. REPRO-DONE only proves
    /// the sequence completed without crashing (the v4.11
    /// convention); the verdicts live in these lines and the
    /// coordinator reads them from the run log.
    private func check(_ name: String, _ pass: Bool, _ detail: String) {
        let line = "REPRO-CHECK \(ReproDriver.variant) \(name) \(pass ? "PASS" : "FAIL") \(detail)"
        print(line)
        fflush(stdout)
        NSLog("%@", line)
    }

    /// An environment-driven skip (not a pass, not a failure):
    /// the check could not run on this host at all.
    private func skipCheck(_ name: String, _ detail: String) {
        let line = "REPRO-CHECK \(ReproDriver.variant) \(name) SKIP \(detail)"
        print(line)
        fflush(stdout)
        NSLog("%@", line)
    }


    func start() {
        guard let renderer = Renderer.shared else {
            print("REPRO-ERROR renderer-unavailable")
            fflush(stdout)
            exit(3)
        }
        self.renderer = renderer
        ReproDriver.marker("start variant=\(ReproDriver.variant)")

        let project = ReproDriver.makeProject(variant: ReproDriver.variant)
        ReproDriver.marker("session-init-before objects=\(project.scene.objects.count)")
        let session = EditorSession(project: project, renderer: renderer)
        self.session = session
        ReproDriver.marker("session-init-after")

        // A real MTKView, attached to the app's window so
        // Renderer.draw can vend drawables exactly as in the
        // live editor. (The swap poll and the sim tick run at
        // the top of draw(), before its drawable guard, so the
        // sequence is driven even if a drawable is unavailable.)
        let view = MTKView(frame: CGRect(x: 0, y: 0, width: 480, height: 360))
        view.device = renderer.device
        view.colorPixelFormat = .bgra8Unorm
        view.drawableSize = CGSize(width: 480, height: 360)
        if let window = UIApplication.shared.connectedScenes
            .compactMap({ $0 as? UIWindowScene }).first?.windows.first {
            window.addSubview(view)
        }
        self.view = view
        session.viewport = view

        // The Weather menu path: EditorView's picker binding
        // calls session.setWeatherMode — mode 1 is Rain.
        ReproDriver.marker("weather-on-before")
        session.setWeatherMode(1)
        weatherOn = true
        ReproDriver.marker("weather-on-after mode=\(session.project.scene.weather.mode)")

        startedAt = CACurrentMediaTime()
        pump()
    }

    private func pump() {
        guard !finished, let session, let renderer, let view else { return }

        // Synthetic presence: a live user driving frames keeps
        // the idle freeze from engaging; the repro does the
        // same through the real input funnel.
        session.noteUserInput()
        let perfT0 = isPerfVariant ? CACurrentMediaTime() : 0
        session.frameTick(view: view)
        if isPerfVariant {
            perfFrameSamples.append((CACurrentMediaTime() - perfT0) * 1000)
        }
        pumpCount += 1

        // Observe the renderer's own step trail (the same tag
        // the app persists for post-crash diagnosis).
        let tag = renderer.weatherStep
        if tag != lastStepTag {
            lastStepTag = tag
            ReproDriver.marker("step \(tag)")
        }

        // v4.13 settle evidence (default + toggle variants):
        // the flat wetness accumulator crosses zero and then
        // its first quantised step (1/24) under rain, and the
        // pump has run a few seconds of frames. (Through
        // v4.12 this chain proved the snow/water sim; the sim
        // is removed and the wetness ramp is what remains.)
        let a = session.liveWetAmount
        if weatherOn, !crossedA, a > 0 {
            crossedA = true
            ReproDriver.marker("wetness-positive W=\(a)")
        }
        if weatherOn, !firstStepCrossed, a >= 0.04 {
            firstStepCrossed = true
            ReproDriver.marker("first-step-crossed W=\(a)")
        }
        if pumpCount % 15 == 0 {
            ReproDriver.marker("pump \(pumpCount) W=\(a) step=\(tag)")
        }

        if isPerfVariant { pumpPerf() }
        if isGravityVariant {
            handleGravityAudit()
            if finished { return }
        }
        if isLoopVariant {
            handleLoopAudit()
            if finished { return }
        }
        if isAuditVariant {
            handleAudit()
            if finished { return }
        }

        if !isGravityVariant, !isPerfVariant, !isLoopVariant, !isAuditVariant, crossedA, firstStepCrossed, pumpCount >= 90 {
            finished = true
            ReproDriver.marker("settle W=\(a) pumps=\(pumpCount)")
            print("REPRO-DONE")
            fflush(stdout)
            NSLog("REPRO-DONE")
            exit(0)
        }

        // Watchdog: a harness outcome, never a crash signal.
        // The snow-audit variants are long by design (carve:
        // ~18 s of drag pumps + 20 s rest; refill: a 90 s
        // refill wait) — the workflow's per-run timeout is
        // 420 s.
        let watchdogLimit: CFTimeInterval
        switch ReproDriver.variant {
        case "gravfall", "gravstack": watchdogLimit = 120
        case "convglass", "convrefl", "convlight": watchdogLimit = 360
        default: watchdogLimit = 100
        }
        if CACurrentMediaTime() - startedAt > watchdogLimit {
            finished = true
            print("REPRO-TIMEOUT pumps=\(pumpCount) W=\(a) step=\(tag) crossedA=\(crossedA) firstStep=\(firstStepCrossed)")
            fflush(stdout)
            NSLog("REPRO-TIMEOUT")
            exit(2)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(33)) { [weak self] in
            MainActor.assumeIsolated { self?.pump() }
        }
    }
}
