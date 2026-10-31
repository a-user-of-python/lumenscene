// GameRuntime.swift — the Play-mode logic runtime for LumenScene.
//
// Executes the project's node graph (GameNodes.swift) while the
// user is in Play mode: events, actions, logic nodes, variables,
// timers, area enter/exit, object tweens/rotations, and the
// particle-emitter system (a decimated copy of a source mesh,
// spawned over its source object, flown with simple physics,
// and rested / bounced / draped onto the collision geometry).
//
// The runtime never touches the scene or the renderer directly:
// every visible effect goes through EditorSession's runtime*
// funnels, which keep the edits in the renderer's dynamic layer
// so stopping Play restores the authored scene untouched. The
// only renderer reads are the collision triangles (via
// session.renderer), per the lane contract.
//
// Loop safety: trigger propagation is an iterative work queue
// (never recursion) with a hard cap of 1024 deliveries per tick.
// Determinism: a seeded splitmix64 RNG drives particle spawns,
// logRandom and cone jitter, reset on every start(), so a play
// session is reproducible.

import Foundation
import simd

@MainActor
final class GameRuntime {

    weak var session: EditorSession?
    private(set) var running = false

    // MARK: - Snapshot state (taken in start())

    /// Graph snapshot: the session's scene can be rebuilt under
    /// us, so the graph is copied at start() and indexed two ways.
    private var snapshotNodes: [LogicNode] = []
    private var nodesByID: [UUID: LogicNode] = [:]
    /// Adjacency keyed "uuid|pin" -> links leaving that pin.
    private var linksByPin: [String: [NodeLink]] = [:]
    private var hasAreaEvents = false

    /// Runtime clock: seconds since start(), advanced by tick dt.
    private var clock: Double = 0

    // MARK: - Logic state

    private var variables: [String: Double] = [:]
    private var toggleNextOn: [UUID: Bool] = [:]   // absent = next fire is "on"
    private var counterCounts: [UUID: Int] = [:]

    private struct TimerState {
        var nodeID: UUID
        var nextFire: Double
        var interval: Double
        var repeating: Bool
    }
    private var timers: [TimerState] = []

    private struct PendingDelay {
        var fireTime: Double
        var nodeID: UUID
    }
    private var pendingDelays: [PendingDelay] = []

    /// Area ids the player was inside as of the last tick.
    private var insideAreas: Set<UUID> = []

    // MARK: - Runtime object overrides (mirror of what the funnels were told)

    /// Last transform this runtime wrote per object, seeded from
    /// the scene on first use. The scene itself is NOT mutated in
    /// Play, so this is the only place "where the object is now"
    /// can be read back (MoveTo/Rotate capture, Teleport targets).
    private var trackedTransforms: [UUID: Transform3] = [:]
    private var trackedMaterials: [UUID: MaterialProps] = [:]

    private struct MoveTween {
        var nodeID: UUID
        var startPos: SIMD3<Float>
        var destPos: SIMD3<Float>
        var duration: Double
        var elapsed: Double
    }
    private var tweens: [UUID: MoveTween] = [:]   // keyed by object id

    private struct RotateState {
        var speed: Double        // degrees / second, may be negative
        var duration: Double     // 0 = forever
        var elapsed: Double
    }
    private var rotations: [UUID: RotateState] = [:]   // keyed by object id

    // MARK: - Trigger queue

    /// Pending (source node, out pin) fires awaiting delivery.
    private var fireQueue: [(from: UUID, pin: String)] = []
    /// Deliveries spent in the current entry point (tick / start /
    /// tap). Reset by the entry point, shared by every drain it
    /// triggers, capped at 1024 — excess fires are dropped.
    private var deliveriesUsed = 0
    private let deliveryCap = 1024

    // MARK: - Deterministic RNG (splitmix64)

    private struct SplitMix64 {
        var state: UInt64
        mutating func nextUInt() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
        /// Uniform Double in 0..<1.
        mutating func nextDouble() -> Double {
            Double(nextUInt() >> 11) * (1.0 / 9007199254740992.0)
        }
        mutating func nextFloat() -> Float { Float(nextDouble()) }
        mutating func range(_ lo: Float, _ hi: Float) -> Float {
            lo + (hi - lo) * nextFloat()
        }
        mutating func range(_ lo: Double, _ hi: Double) -> Double {
            lo + (hi - lo) * nextDouble()
        }
    }
    private var rng = SplitMix64(state: 0x51AB_C0DE_1234_5678)

    // MARK: - Collision grid (particle support queries)

    private struct CellKey: Hashable { var x: Int; var z: Int }
    private var collisionTris: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
    private var collisionGrid: [CellKey: [Int]] = [:]
    private let cellSize: Float = 0.5

    private struct Support {
        var height: Float
        var normal: SIMD3<Float>
    }

    // MARK: - Emitters

    /// Decimated source meshes, keyed "sourceUUID|faces".
    private var decimationCache: [String: MeshData] = [:]
    private var emitters: [UUID: EmitterState] = [:]   // keyed by actParticles node id
    /// True while the renderer currently holds particle soups from
    /// us, so the final [] that clears them is sent exactly once.
    private var soupsSent = false

    // MARK: - Lifecycle

    func start() {
        guard let session else { return }
        stopInternal(clearSoups: true)
        running = true
        clock = 0
        rng = SplitMix64(state: 0x51AB_C0DE_1234_5678)

        // Snapshot the graph and index it.
        let graph = session.project.scene.graph
        snapshotNodes = graph.nodes
        nodesByID = Dictionary(graph.nodes.map { ($0.id, $0) },
                               uniquingKeysWith: { first, _ in first })
        linksByPin = [:]
        for link in graph.links {
            linksByPin[pinKey(link.from, link.fromPin), default: []].append(link)
        }
        hasAreaEvents = graph.nodes.contains { $0.kind == "evEnterArea" || $0.kind == "evExitArea" }
        insideAreas = []

        buildCollisionGrid()

        // Arm timers: first fire at delay + interval, then every
        // interval while repeating. A non-positive interval can
        // only ever be a one-shot at delay (delay + 0 == delay).
        for node in graph.nodes where node.kind == "evTimer" {
            let delay = node.num("delay", 0)
            let interval = node.num("interval", 5)
            let repeating = node.boolean("repeat", true) && interval > 0
            timers.append(TimerState(nodeID: node.id,
                                     nextFire: delay + max(0, interval),
                                     interval: interval,
                                     repeating: repeating))
        }

        // Fire evStart.
        deliveriesUsed = 0
        for node in graph.nodes where node.kind == "evStart" {
            enqueueFire(node.id, "out")
        }
        processQueue()
    }

    func stop() {
        stopInternal(clearSoups: true)
        running = false
    }

    /// Clears every piece of runtime state. The session restores
    /// the scene itself, so no funnel writes happen here except
    /// clearing particle soups we previously handed over.
    private func stopInternal(clearSoups: Bool) {
        let hadSoups = soupsSent
        snapshotNodes = []
        nodesByID = [:]
        linksByPin = [:]
        hasAreaEvents = false
        variables = [:]
        toggleNextOn = [:]
        counterCounts = [:]
        timers = []
        pendingDelays = []
        insideAreas = []
        trackedTransforms = [:]
        trackedMaterials = [:]
        tweens = [:]
        rotations = [:]
        fireQueue = []
        deliveriesUsed = 0
        emitters = [:]
        decimationCache = [:]
        collisionTris = []
        collisionGrid = [:]
        soupsSent = false
        if clearSoups && hadSoups {
            session?.runtimeSetParticleSoups([])
        }
    }

    // MARK: - Frame need

    var needsFrame: Bool {
        guard running else { return false }
        if !tweens.isEmpty || !rotations.isEmpty { return true }
        if !pendingDelays.isEmpty || !timers.isEmpty { return true }
        for emitter in emitters.values where emitter.emitting || !emitter.particles.isEmpty {
            return true
        }
        // Walking must keep ticking so area enter/exit can fire;
        // the session draws continuously in Play while moving
        // anyway — this pin keeps the checks alive even at rest.
        if hasAreaEvents { return true }
        return false
    }

    // MARK: - Per-frame tick

    func tick(dt: Float, playerPos: SIMD3<Float>) {
        guard running, session != nil else { return }
        let step = Double(max(0, dt))
        clock += step
        deliveriesUsed = 0

        fireDueTimers()
        fireDueDelays()
        checkAreas(playerPos: playerPos)
        processQueue()

        advanceTweens(dt: step)
        advanceRotations(dt: step)
        // Tween completions enqueue "arrived" fires.
        processQueue()

        updateEmitters(dt: dt)
        buildAndSendSoups()
    }

    // MARK: - Play-mode tap

    func handleTap(objectID: UUID) {
        guard running else { return }
        deliveriesUsed = 0
        for node in snapshotNodes where node.kind == "evTap" {
            // A nil ref means "any object" (no specific target set).
            if let target = node.ref("object"), target != objectID { continue }
            enqueueFire(node.id, "out")
        }
        processQueue()
    }

    // MARK: - Collision rebuild hook

    /// Called by the session after an area load/unload rebuilds
    /// the renderer's collision set.
    func collisionDidChange() {
        buildCollisionGrid()
    }

    // MARK: - Timers / delays / areas (enqueue phase of tick)

    private func fireDueTimers() {
        guard !timers.isEmpty else { return }
        var kept: [TimerState] = []
        kept.reserveCapacity(timers.count)
        for var timer in timers {
            var fired = false
            var guardCount = 0
            while clock >= timer.nextFire {
                enqueueFire(timer.nodeID, "out")
                fired = true
                guardCount += 1
                if timer.repeating && timer.interval > 0 && guardCount < 32 {
                    timer.nextFire += timer.interval
                } else {
                    break
                }
            }
            // A repeating timer that fell far behind (long hitch)
            // resynchronises instead of bursting to catch up.
            if timer.repeating && clock >= timer.nextFire {
                timer.nextFire = clock + timer.interval
            }
            if !fired || timer.repeating {
                kept.append(timer)
            }
        }
        timers = kept
    }

    private func fireDueDelays() {
        guard !pendingDelays.isEmpty else { return }
        var kept: [PendingDelay] = []
        for delay in pendingDelays {
            if clock >= delay.fireTime {
                enqueueFire(delay.nodeID, "out")
            } else {
                kept.append(delay)
            }
        }
        pendingDelays = kept
    }

    private func checkAreas(playerPos: SIMD3<Float>) {
        guard let session, hasAreaEvents || !insideAreas.isEmpty else { return }
        var current: Set<UUID> = []
        for area in session.project.scene.areas where area.contains(playerPos) {
            current.insert(area.id)
        }
        let entered = current.subtracting(insideAreas)
        let exited = insideAreas.subtracting(current)
        insideAreas = current
        guard !entered.isEmpty || !exited.isEmpty else { return }
        for node in snapshotNodes {
            guard let areaID = node.ref("area") else { continue }
            if node.kind == "evEnterArea" && entered.contains(areaID) {
                enqueueFire(node.id, "out")
            } else if node.kind == "evExitArea" && exited.contains(areaID) {
                enqueueFire(node.id, "out")
            }
        }
    }

    // MARK: - Trigger propagation (iterative work queue)

    private func pinKey(_ id: UUID, _ pin: String) -> String {
        id.uuidString + "|" + pin
    }

    private func enqueueFire(_ nodeID: UUID, _ pin: String) {
        fireQueue.append((from: nodeID, pin: pin))
    }

    /// Drains the fire queue: each fire delivers to every link
    /// leaving that pin, executing the target node. Executions may
    /// enqueue further fires, which the same drain picks up. Past
    /// the per-entry delivery cap the remaining queue is dropped.
    private func processQueue() {
        var head = 0
        while head < fireQueue.count {
            if deliveriesUsed >= deliveryCap {
                fireQueue.removeAll()
                return
            }
            let fire = fireQueue[head]
            head += 1
            guard let links = linksByPin[pinKey(fire.from, fire.pin)] else { continue }
            for link in links {
                if deliveriesUsed >= deliveryCap {
                    fireQueue.removeAll()
                    return
                }
                deliveriesUsed += 1
                if let target = nodesByID[link.to] {
                    execute(target, inputPin: link.toPin)
                }
            }
        }
        fireQueue.removeAll()
    }

    // MARK: - Node execution

    private func execute(_ node: LogicNode, inputPin: String) {
        guard let session else { return }
        switch node.kind {

        // MARK: Actions

        case "actShow":
            if let id = node.ref("object") { session.runtimeSetVisible(id, true) }
            enqueueFire(node.id, "out")

        case "actHide":
            if let id = node.ref("object") { session.runtimeSetVisible(id, false) }
            enqueueFire(node.id, "out")

        case "actMoveTo":
            startMoveTo(node)
            enqueueFire(node.id, "out")

        case "actRotate":
            if let id = node.ref("object") {
                // Replace-on-retrigger: assigning over any old entry.
                rotations[id] = RotateState(speed: node.num("speed", 90),
                                            duration: node.num("duration", 0),
                                            elapsed: 0)
            }
            enqueueFire(node.id, "out")

        case "actSetLight":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                var material = currentMaterial(of: obj)
                material.emissiveOn = node.boolean("on", true)
                material.emissiveIntensity = node.num("intensity", 4)
                if let colour = parseHexColour(node.txt("color", "")) {
                    material.emissiveColor = colour
                } // unparseable / empty colour: keep the existing one
                trackedMaterials[id] = material
                session.runtimeSetMaterial(id, material)
            }
            enqueueFire(node.id, "out")

        case "actSetMaterial":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                var material = currentMaterial(of: obj)
                let texture = node.txt("texture", "none")
                material.texturePreset = texture
                material.glassOn = node.boolean("glass", false)
                // Roads are the known auto-fit texture family (their
                // markings live in a single tile); everything else tiles.
                material.textureFit = texture.hasPrefix("road-")
                trackedMaterials[id] = material
                session.runtimeSetMaterial(id, material)
            }
            enqueueFire(node.id, "out")

        case "actParticles":
            if inputPin == "stop" {
                emitters[node.id]?.emitting = false
            } else {
                startEmitter(node)
            }

        case "actLoadArea":
            if let id = node.ref("area") { session.runtimeLoadArea(id) }
            enqueueFire(node.id, "out")

        case "actUnloadArea":
            if let id = node.ref("area") { session.runtimeUnloadArea(id) }
            enqueueFire(node.id, "out")

        case "actSetWeather":
            let mode: Int
            switch node.txt("mode", "Off") {
            case "Rain": mode = 1
            case "Snow": mode = 2
            default: mode = 0
            }
            session.runtimeSetWeather(mode: mode, clear: node.boolean("clear", false))
            enqueueFire(node.id, "out")

        case "actTeleport":
            if let targetID = node.ref("target"),
               let target = session.project.scene.object(id: targetID) {
                let t = currentTransform(of: target)
                session.runtimeTeleportPlayer(to: SIMD3<Float>(Float(t.position[0]),
                                                               Float(t.position[1]),
                                                               Float(t.position[2])))
            }
            enqueueFire(node.id, "out")

        case "actMessage":
            session.runtimeShowMessage(node.txt("message", ""),
                                       seconds: node.num("seconds", 3))
            enqueueFire(node.id, "out")

        case "actSetVariable":
            let name = node.txt("name", "")
            if !name.isEmpty {
                let value = node.num("value", 0)
                if node.txt("op", "Set") == "Add" {
                    variables[name, default: 0] += value
                } else {
                    variables[name] = value
                }
                // Matching evVariable nodes fire on any application.
                for watcher in snapshotNodes where watcher.kind == "evVariable"
                    && watcher.txt("name", "") == name {
                    enqueueFire(watcher.id, "out")
                }
            }
            enqueueFire(node.id, "out")

        // MARK: Logic

        case "logIf":
            let name = node.txt("name", "")
            let lhs = variables[name, default: 0]
            let result = compare(lhs, node.txt("cmp", ">="), node.num("value", 0))
            enqueueFire(node.id, result ? "then" : "else")

        case "logDelay":
            pendingDelays.append(PendingDelay(fireTime: clock + node.num("seconds", 1),
                                              nodeID: node.id))

        case "logToggle":
            let on = toggleNextOn[node.id] ?? true
            toggleNextOn[node.id] = !on
            enqueueFire(node.id, on ? "on" : "off")

        case "logRandom":
            let hit = rng.nextDouble() * 100 < node.num("percent", 50)
            enqueueFire(node.id, hit ? "yes" : "no")

        case "logCounter":
            let target = max(1, Int(node.num("target", 3).rounded()))
            let count = (counterCounts[node.id] ?? 0) + 1
            enqueueFire(node.id, "out")
            if count >= target {
                counterCounts[node.id] = 0
                enqueueFire(node.id, "reached")
            } else {
                counterCounts[node.id] = count
            }

        default:
            // Event nodes (and unknown kinds) are sources, not
            // targets: a link landing on one does nothing.
            break
        }
    }

    /// Comparison for logIf. Accepts symbol and word forms so the
    /// catalogue's spelling does not matter ("≥", ">=", "Greater
    /// or Equal", "is at least", …).
    private func compare(_ lhs: Double, _ cmp: String, _ rhs: Double) -> Bool {
        let c = cmp.lowercased()
        if c.contains("!=") || c.contains("≠") || c.contains("not equal") { return lhs != rhs }
        if c.contains(">=") || c.contains("≥") || c.contains("at least")
            || c.contains("greater or") { return lhs >= rhs }
        if c.contains("<=") || c.contains("≤") || c.contains("at most")
            || c.contains("less or") { return lhs <= rhs }
        if c.contains(">") || c.contains("greater") || c.contains("more") { return lhs > rhs }
        if c.contains("<") || c.contains("less") { return lhs < rhs }
        return lhs == rhs   // "=", "==", "equal", anything else
    }

    /// "#RRGGBB" -> RGB, nil when the string is not exactly that.
    private func parseHexColour(_ text: String) -> RGB? {
        var hex = text.trimmingCharacters(in: .whitespaces)
        if hex.hasPrefix("#") { hex.removeFirst() }
        guard hex.count == 6, let value = UInt32(hex, radix: 16) else { return nil }
        return RGB(Double((value >> 16) & 0xFF) / 255,
                   Double((value >> 8) & 0xFF) / 255,
                   Double(value & 0xFF) / 255)
    }

    // MARK: - Effective (runtime) object state

    private func currentTransform(of obj: SceneObject) -> Transform3 {
        trackedTransforms[obj.id] ?? obj.transform
    }

    private func currentMaterial(of obj: SceneObject) -> MaterialProps {
        trackedMaterials[obj.id] ?? obj.material
    }

    private func applyTransform(_ id: UUID, _ t: Transform3) {
        trackedTransforms[id] = t
        session?.runtimeSetTransform(id, t)
    }

    private func position(of t: Transform3) -> SIMD3<Float> {
        SIMD3<Float>(Float(t.position[0]), Float(t.position[1]), Float(t.position[2]))
    }

    // MARK: - MoveTo tweens

    private func startMoveTo(_ node: LogicNode) {
        guard let session, let id = node.ref("object"),
              let obj = session.project.scene.object(id: id) else { return }
        let start = currentTransform(of: obj)
        let startPos = position(of: start)
        let offset = SIMD3<Float>(Float(node.num("dx", 0)),
                                  Float(node.num("dy", 0)),
                                  Float(node.num("dz", 0)))
        var base = startPos
        if node.txt("mode", "To Object") != "By Offset",
           let targetID = node.ref("target"),
           let target = session.project.scene.object(id: targetID) {
            // Destination is the target's position NOW, at
            // execution time — not wherever it ends up later.
            base = position(of: currentTransform(of: target))
        }
        // The offset is added in both modes.
        let dest = base + offset
        let duration = node.num("duration", 2)
        if duration <= 0 {
            tweens[id] = nil
            var t = start
            t.position = [Double(dest.x), Double(dest.y), Double(dest.z)]
            applyTransform(id, t)
            enqueueFire(node.id, "arrived")
        } else {
            // A new MoveTo on the same object replaces the old tween.
            tweens[id] = MoveTween(nodeID: node.id, startPos: startPos,
                                   destPos: dest, duration: duration, elapsed: 0)
        }
    }

    private func advanceTweens(dt: Double) {
        guard !tweens.isEmpty else { return }
        // Snapshot the keys: completion mutates the dictionary.
        for id in Array(tweens.keys) {
            guard var tween = tweens[id], let session,
                  let obj = session.project.scene.object(id: id) else {
                tweens[id] = nil
                continue
            }
            tween.elapsed += dt
            let raw = min(1, tween.elapsed / tween.duration)
            // Smoothstep easing.
            let s = raw * raw * (3 - 2 * raw)
            let pos = tween.startPos + (tween.destPos - tween.startPos) * Float(s)
            var t = currentTransform(of: obj)
            t.position = [Double(pos.x), Double(pos.y), Double(pos.z)]
            if raw >= 1 {
                tweens[id] = nil
                applyTransform(id, t)
                enqueueFire(tween.nodeID, "arrived")
            } else {
                tweens[id] = tween
                applyTransform(id, t)
            }
        }
    }

    private func advanceRotations(dt: Double) {
        guard !rotations.isEmpty else { return }
        for id in Array(rotations.keys) {
            guard var state = rotations[id], let session,
                  let obj = session.project.scene.object(id: id) else {
                rotations[id] = nil
                continue
            }
            state.elapsed += dt
            var t = currentTransform(of: obj)
            var yaw = t.rotation[1] + state.speed * dt
            yaw = yaw.truncatingRemainder(dividingBy: 360)
            if yaw < 0 { yaw += 360 }
            t.rotation[1] = yaw
            applyTransform(id, t)
            if state.duration > 0 && state.elapsed >= state.duration {
                rotations[id] = nil
            } else {
                rotations[id] = state
            }
        }
    }

    // MARK: - Collision grid

    private func buildCollisionGrid() {
        guard let session else { return }
        collisionTris = session.renderer.collisionTriangles
        collisionGrid = [:]
        for (index, tri) in collisionTris.enumerated() {
            let loX = min(tri.0.x, tri.1.x, tri.2.x), hiX = max(tri.0.x, tri.1.x, tri.2.x)
            let loZ = min(tri.0.z, tri.1.z, tri.2.z), hiZ = max(tri.0.z, tri.1.z, tri.2.z)
            var minCX = Int(floor(loX / cellSize)), maxCX = Int(floor(hiX / cellSize))
            var minCZ = Int(floor(loZ / cellSize)), maxCZ = Int(floor(hiZ / cellSize))
            // A huge triangle (terrain sheet) would flood the grid;
            // cap the span and register its centroid cell as well —
            // support queries also probe the exact cell, and large
            // flat sheets are additionally hit through neighbouring
            // geometry in practice. (Grid is a query accelerator,
            // not a filter: correctness leans on the barycentric
            // test, so a missed registration only loses a rest spot.)
            if maxCX - minCX > 32 { let c = (minCX + maxCX) / 2; minCX = c - 16; maxCX = c + 16 }
            if maxCZ - minCZ > 32 { let c = (minCZ + maxCZ) / 2; minCZ = c - 16; maxCZ = c + 16 }
            for cx in minCX...maxCX {
                for cz in minCZ...maxCZ {
                    collisionGrid[CellKey(x: cx, z: cz), default: []].append(index)
                }
            }
        }
    }

    /// Highest up-facing (normal.y > 0.3) triangle surface at
    /// (x, z) that lies at or below fromY + 0.05, plus its normal.
    private func support(atX x: Float, z: Float, fromY: Float) -> Support? {
        let key = CellKey(x: Int(floor(x / cellSize)), z: Int(floor(z / cellSize)))
        guard let indices = collisionGrid[key] else { return nil }
        var best: Support?
        for index in indices {
            let tri = collisionTris[index]
            let a = tri.0, b = tri.1, c = tri.2
            let normalRaw = simd_cross(b - a, c - a)
            let len = simd_length(normalRaw)
            if len < 1e-9 { continue }
            let n = normalRaw / len
            if n.y <= 0.3 { continue }
            // Barycentric test in the XZ projection.
            let denom = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z)
            if abs(denom) < 1e-9 { continue }
            let w0 = ((b.z - c.z) * (x - c.x) + (c.x - b.x) * (z - c.z)) / denom
            let w1 = ((c.z - a.z) * (x - c.x) + (a.x - c.x) * (z - c.z)) / denom
            let w2 = 1 - w0 - w1
            if w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5 { continue }
            let h = w0 * a.y + w1 * b.y + w2 * c.y
            if h <= fromY + 0.05 && (best == nil || h > best!.height) {
                best = Support(height: h, normal: n)
            }
        }
        return best
    }

    // MARK: - Particle source decimation (vertex clustering)

    private struct Cell3: Hashable { var x: Int; var y: Int; var z: Int }

    /// Clustered copy of `mesh` for one grid cell size: each cell
    /// collapses to one vertex (average position / normal / colour
    /// / uv) and triangles are rebuilt, dropping degenerates.
    private func clustered(_ mesh: MeshData, cell: Float) -> MeshData {
        guard cell > 0, mesh.vertexCount > 0 else { return mesh }
        var cellIndex: [Cell3: Int] = [:]
        var sums: [(pos: SIMD3<Float>, nrm: SIMD3<Float>, col: SIMD3<Float>, uv: SIMD2<Float>, count: Float)] = []
        var vertexMap: [Int] = []
        vertexMap.reserveCapacity(mesh.vertexCount)
        for i in 0..<mesh.vertexCount {
            let p = SIMD3<Float>(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2])
            let key = Cell3(x: Int(floor(p.x / cell)), y: Int(floor(p.y / cell)), z: Int(floor(p.z / cell)))
            let idx: Int
            if let existing = cellIndex[key] {
                idx = existing
            } else {
                idx = sums.count
                cellIndex[key] = idx
                sums.append((.zero, .zero, .zero, .zero, 0))
            }
            vertexMap.append(idx)
            sums[idx].pos += p
            sums[idx].nrm += SIMD3<Float>(mesh.normals[3 * i], mesh.normals[3 * i + 1], mesh.normals[3 * i + 2])
            if mesh.colors.count >= 3 * i + 3 {
                sums[idx].col += SIMD3<Float>(mesh.colors[3 * i], mesh.colors[3 * i + 1], mesh.colors[3 * i + 2])
            } else {
                sums[idx].col += SIMD3<Float>(1, 1, 1)
            }
            if mesh.uvs.count >= 2 * i + 2 {
                sums[idx].uv += SIMD2<Float>(mesh.uvs[2 * i], mesh.uvs[2 * i + 1])
            }
            sums[idx].count += 1
        }
        var out = MeshData()
        for s in sums {
            let inv = 1 / max(1, s.count)
            var n = s.nrm * inv
            n = simd_length(n) > 1e-6 ? simd_normalize(n) : SIMD3<Float>(0, 1, 0)
            out.addVertex(s.pos * inv, n, s.uv * inv, s.col * inv)
        }
        var i = 0
        while i + 2 < mesh.indices.count {
            let a = vertexMap[Int(mesh.indices[i])]
            let b = vertexMap[Int(mesh.indices[i + 1])]
            let c = vertexMap[Int(mesh.indices[i + 2])]
            i += 3
            if a == b || b == c || a == c { continue }
            let pa = SIMD3<Float>(out.positions[3 * a], out.positions[3 * a + 1], out.positions[3 * a + 2])
            let pb = SIMD3<Float>(out.positions[3 * b], out.positions[3 * b + 1], out.positions[3 * b + 2])
            let pc = SIMD3<Float>(out.positions[3 * c], out.positions[3 * c + 1], out.positions[3 * c + 2])
            if simd_length_squared(simd_cross(pb - pa, pc - pa)) < 1e-14 { continue }
            out.indices += [UInt32(a), UInt32(b), UInt32(c)]
        }
        return out
    }

    /// Decimate `mesh` toward `target` triangles: binary-search
    /// the clustering cell size (at most 12 evaluations) so the
    /// result is at or below ~1.3x the target and as close to it
    /// as the search finds. Results are cached per (source, faces).
    private func decimatedMesh(sourceID: UUID, source: MeshData, targetFaces: Int) -> MeshData {
        if targetFaces <= 0 || source.triangleCount <= targetFaces { return source }
        let cacheKey = sourceID.uuidString + "|" + String(targetFaces)
        if let cached = decimationCache[cacheKey] { return cached }
        let b = source.bounds
        let diag = simd_length(b.max - b.min)
        guard diag > 1e-6 else { return source }
        let ceiling = Int(1.3 * Double(targetFaces))
        var lo: Float = diag / 512
        var hi: Float = diag
        var best: MeshData?
        for _ in 0..<12 {
            let mid = (lo + hi) / 2
            let candidate = clustered(source, cell: mid)
            if candidate.triangleCount > ceiling {
                lo = mid   // too many triangles: cluster harder
            } else {
                hi = mid
                if best == nil || candidate.triangleCount > best!.triangleCount {
                    best = candidate
                }
            }
        }
        let result = best ?? clustered(source, cell: hi)
        decimationCache[cacheKey] = result
        return result
    }

    // MARK: - Emitters

    private final class EmitterState {
        var nodeID: UUID
        var sourceID: UUID
        var sourceObject: SceneObject   // snapshot for soup materials
        var mesh: MeshData              // decimated local-space mesh
        var halfHeight: Float           // local bounds half extents
        var halfWidth: Float
        var isPhysics: Bool
        var rate: Double
        var lifetime: Float
        var size: Float
        var speed: Float
        var spreadDeg: Float
        var direction: String
        var gravity: Float
        var drag: Float
        var hitbox: String
        var hitboxSize: Float
        var bounce: Float
        var drape: Bool
        var emitting = false
        var emitUntil: Double = .infinity
        var accumulator: Double = 0
        var particles: [Particle] = []

        init(nodeID: UUID, sourceID: UUID, sourceObject: SceneObject, mesh: MeshData,
             halfHeight: Float, halfWidth: Float, isPhysics: Bool, rate: Double,
             lifetime: Float, size: Float, speed: Float, spreadDeg: Float,
             direction: String, gravity: Float, drag: Float, hitbox: String,
             hitboxSize: Float, bounce: Float, drape: Bool) {
            self.nodeID = nodeID; self.sourceID = sourceID
            self.sourceObject = sourceObject; self.mesh = mesh
            self.halfHeight = halfHeight; self.halfWidth = halfWidth
            self.isPhysics = isPhysics; self.rate = rate
            self.lifetime = lifetime; self.size = size; self.speed = speed
            self.spreadDeg = spreadDeg; self.direction = direction
            self.gravity = gravity; self.drag = drag; self.hitbox = hitbox
            self.hitboxSize = hitboxSize; self.bounce = bounce; self.drape = drape
        }

        var perEmitterCap: Int { isPhysics ? 400 : 800 }
    }

    private struct Particle {
        var pos = SIMD3<Float>.zero
        var vel = SIMD3<Float>.zero
        var age: Float = 0
        var lifetime: Float = 12
        var size: Float = 1
        var yaw: Float = 0              // radians, evolves while flying
        var spinVel: Float = 0          // radians / second
        var phase: Float = 0            // flutter phase
        var resting = false
        var restPos = SIMD3<Float>.zero
        var restQuat = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    }

    private func startEmitter(_ node: LogicNode) {
        let emitter: EmitterState
        if let existing = emitters[node.id] {
            emitter = existing
        } else {
            guard let created = makeEmitter(node) else { return }
            emitters[node.id] = created
            emitter = created
        }
        emitter.emitting = true
        emitter.accumulator = 0
        let duration = node.num("duration", 0)
        emitter.emitUntil = duration > 0 ? clock + duration : .infinity
        let burst = Int(node.num("burst", 0).rounded())
        if burst > 0 { _ = spawnParticles(emitter, count: burst) }
    }

    /// Builds emitter state lazily on the node's first start
    /// trigger — including resolving and decimating the source mesh.
    private func makeEmitter(_ node: LogicNode) -> EmitterState? {
        guard let session, let sourceID = node.ref("source"),
              let obj = session.project.scene.object(id: sourceID) else { return nil }
        let rawMesh: MeshData
        if obj.source == "model" {
            guard let mesh = session.meshes[sourceID] else { return nil }
            rawMesh = mesh
        } else {
            rawMesh = MeshGen.primitive(kind: obj.primitiveKind, params: obj.params)
        }
        guard !rawMesh.isEmpty else { return nil }
        let faces = Int(node.num("faces", 200).rounded())
        let mesh = decimatedMesh(sourceID: sourceID, source: rawMesh, targetFaces: faces)
        let b = mesh.bounds
        return EmitterState(
            nodeID: node.id, sourceID: sourceID, sourceObject: obj, mesh: mesh,
            halfHeight: (b.max.y - b.min.y) / 2, halfWidth: (b.max.x - b.min.x) / 2,
            isPhysics: node.txt("mode", "Physics") != "Simple",
            rate: node.num("rate", 24),
            lifetime: Float(node.num("lifetime", 12)),
            size: Float(node.num("size", 1)),
            speed: Float(node.num("speed", 1.6)),
            spreadDeg: Float(node.num("spread", 28)),
            direction: node.txt("direction", "Up"),
            gravity: Float(node.num("gravity", 1)),
            drag: Float(node.num("drag", 0.12)),
            hitbox: node.txt("hitbox", "Disc"),
            hitboxSize: Float(node.num("hitboxSize", 1)),
            bounce: Float(node.num("bounce", 0.35)),
            drape: node.boolean("drape", true))
    }

    private func totalAliveParticles() -> Int {
        emitters.values.reduce(0) { $0 + $1.particles.count }
    }

    /// World-space AABB of an emitter's decimated mesh under the
    /// source object's current (runtime) transform.
    private func sourceWorldBounds(_ emitter: EmitterState) -> (min: SIMD3<Float>, max: SIMD3<Float>) {
        let obj = session?.project.scene.object(id: emitter.sourceID) ?? emitter.sourceObject
        let m = currentTransform(of: obj).matrix
        let b = emitter.mesh.bounds
        var lo = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var hi = -lo
        for cx in [b.min.x, b.max.x] {
            for cy in [b.min.y, b.max.y] {
                for cz in [b.min.z, b.max.z] {
                    let w = (m * SIMD4<Float>(cx, cy, cz, 1)).xyz
                    lo = simd_min(lo, w); hi = simd_max(hi, w)
                }
            }
        }
        return (lo, hi)
    }

    /// Random direction inside a cone of `spreadDeg` around `base`.
    private func coneDirection(base: SIMD3<Float>, spreadDeg: Float) -> SIMD3<Float> {
        if spreadDeg <= 0.01 { return base }
        let spreadRad = spreadDeg * .pi / 180
        let cosMax = cos(spreadRad)
        let cosT = 1 - rng.nextFloat() * (1 - cosMax)
        let sinT = sqrt(max(0, 1 - cosT * cosT))
        let phi = rng.nextFloat() * 2 * .pi
        let helper = abs(base.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
        let t = simd_normalize(simd_cross(base, helper))
        let b = simd_cross(base, t)
        return simd_normalize(base * cosT + (t * cos(phi) + b * sin(phi)) * sinT)
    }

    /// Spawns up to `count` particles, honouring the per-emitter
    /// and global alive caps. Returns how many actually spawned.
    @discardableResult
    private func spawnParticles(_ emitter: EmitterState, count: Int) -> Int {
        let capacity = min(emitter.perEmitterCap - emitter.particles.count,
                           1200 - totalAliveParticles())
        let n = min(count, max(0, capacity))
        guard n > 0 else { return 0 }
        let bounds = sourceWorldBounds(emitter)
        let centre = (bounds.min + bounds.max) / 2
        for _ in 0..<n {
            // Random point over the source's world AABB footprint,
            // at the AABB top.
            let x = rng.range(bounds.min.x, bounds.max.x)
            let z = rng.range(bounds.min.z, bounds.max.z)
            let pos = SIMD3<Float>(x, bounds.max.y, z)
            let base: SIMD3<Float>
            switch emitter.direction {
            case "Down":
                base = SIMD3<Float>(0, -1, 0)
            case "Outward":
                var radial = SIMD3<Float>(pos.x - centre.x, 0, pos.z - centre.z)
                if simd_length(radial) < 0.01 {
                    let a = rng.nextFloat() * 2 * .pi
                    radial = SIMD3<Float>(sin(a), 0, cos(a))
                }
                base = simd_normalize(radial)
            default: // "Up"
                base = SIMD3<Float>(0, 1, 0)
            }
            let dir = coneDirection(base: base, spreadDeg: emitter.spreadDeg)
            var p = Particle()
            p.pos = pos
            p.vel = dir * (emitter.speed * rng.range(0.7, 1.3))
            p.lifetime = emitter.lifetime > 0 ? emitter.lifetime : 12
            p.size = emitter.size
            p.yaw = rng.range(0, 2 * .pi)
            p.spinVel = rng.range(-0.9, 0.9)
            p.phase = rng.range(0, 2 * .pi)
            emitter.particles.append(p)
        }
        return n
    }

    private func updateEmitters(dt: Float) {
        guard !emitters.isEmpty else { return }
        for emitter in Array(emitters.values) {
            // Emission.
            if emitter.emitting {
                if clock > emitter.emitUntil {
                    emitter.emitting = false
                } else {
                    emitter.accumulator += emitter.rate * Double(dt)
                    let wanted = Int(emitter.accumulator.rounded(.down))
                    if wanted > 0 {
                        let spawned = spawnParticles(emitter, count: wanted)
                        // At a cap the backlog is dropped, not banked:
                        // emission pauses instead of bursting later.
                        emitter.accumulator = spawned < wanted ? 0
                            : emitter.accumulator - Double(wanted)
                    }
                }
            }
            integrate(emitter, dt: dt)
            emitter.particles.removeAll { $0.age >= $0.lifetime }
        }
    }

    private func integrate(_ emitter: EmitterState, dt: Float) {
        guard dt > 0 else { return }
        let isDisc = emitter.hitbox == "Disc"
        for i in emitter.particles.indices {
            var p = emitter.particles[i]
            p.age += dt
            if p.age >= p.lifetime { emitter.particles[i] = p; continue }
            if p.resting { emitter.particles[i] = p; continue }

            let prevY = p.pos.y
            // Gravity (discs fall slower — they catch the air).
            var g: Float = 9.8 * emitter.gravity
            if isDisc { g *= 0.75 }
            p.vel.y -= g * dt
            // Exponential drag.
            p.vel *= Float(exp(-Double(emitter.drag) * Double(dt)))
            // Disc flutter: horizontal sinusoidal sway.
            if isDisc {
                let amp: Float = 0.5 * (1 - emitter.gravity)
                if amp != 0 {
                    let arg = p.age * 2.2 * 2 * .pi + p.phase
                    p.vel.x += sin(arg) * amp * dt
                    p.vel.z += cos(arg) * amp * 0.6 * dt
                }
            }
            p.pos += p.vel * dt
            p.yaw += p.spinVel * dt

            // Collision: support is queried from the height the
            // particle started this step at, so a fast step that
            // crosses the surface still finds it.
            if p.vel.y <= 0, let sup = support(atX: p.pos.x, z: p.pos.z, fromY: prevY) {
                let restOffset = max(0.003, emitter.halfHeight * p.size * emitter.hitboxSize)
                if p.pos.y <= sup.height + restOffset {
                    let impact = -p.vel.y
                    if emitter.isPhysics && emitter.bounce > 0 && impact > 0.8 {
                        p.vel.y = impact * emitter.bounce
                        p.vel.x *= 0.7   // tangential * (1 - 0.3)
                        p.vel.z *= 0.7
                        p.pos.y = sup.height + restOffset
                    } else {
                        comeToRest(emitter, &p, support: sup, restOffset: restOffset)
                    }
                }
            }
            emitter.particles[i] = p
        }
    }

    /// Freezes a particle's rest pose. Stores position + orientation
    /// (not a fully baked matrix) so the final-0.4 s shrink can
    /// still scale the particle on its way out.
    private func comeToRest(_ emitter: EmitterState, _ p: inout Particle,
                            support sup: Support, restOffset: Float) {
        p.resting = true
        p.vel = .zero
        let flatPos = SIMD3<Float>(p.pos.x, sup.height + restOffset, p.pos.z)
        let drapable = emitter.isPhysics && emitter.drape
            && (emitter.hitbox == "Disc" || emitter.hitbox == "Box")
        if drapable, let draped = drapeRest(emitter, particle: p, centre: flatPos,
                                            centreSupport: sup, restOffset: restOffset) {
            p.restPos = draped.pos
            p.restQuat = draped.quat
        } else {
            // Simple rest (and physics without drape): flat on the
            // surface, aligned to the support normal, yaw kept.
            p.restPos = flatPos
            p.restQuat = flatQuat(normal: sup.normal, yaw: p.yaw)
        }
        p.pos = p.restPos
    }

    /// Orientation that keeps the particle's yaw but tilts its
    /// local +Y onto `normal`.
    private func flatQuat(normal: SIMD3<Float>, yaw: Float) -> simd_quatf {
        let yawQuat = simd_quaternion(yaw, SIMD3<Float>(0, 1, 0))
        let up = SIMD3<Float>(0, 1, 0)
        let d = simd_dot(up, normal)
        if d > 0.9999 { return yawQuat }
        if d < -0.9999 {
            return simd_quaternion(Float.pi, SIMD3<Float>(1, 0, 0)) * yawQuat
        }
        let axis = simd_normalize(simd_cross(up, normal))
        let align = simd_quaternion(acos(max(-1, min(1, d))), axis)
        return align * yawQuat
    }

    /// DRAPE: a rest-pose approximation (not cloth simulation).
    /// The particle's world right vector (from its yaw) is sampled
    /// at both edges against the support surface:
    ///  - both edges supported: stay flat on the centre normal
    ///    (returns nil so the caller uses the flat pose);
    ///  - exactly one edge unsupported (no hit, or a hit more than
    ///    0.015 below the centre support): hinge around the axis
    ///    through the SUPPORTED edge point at its support height,
    ///    rotating the unsupported side down — 55 degrees for a
    ///    no-hit edge, atan((centreH - lowH + 0.01) / hx) capped
    ///    at 65 degrees for a lower-hit edge;
    ///  - both edges unsupported (narrow perch): stay flat.
    private func drapeRest(_ emitter: EmitterState, particle p: Particle,
                           centre: SIMD3<Float>, centreSupport: Support,
                           restOffset: Float) -> (pos: SIMD3<Float>, quat: simd_quatf)? {
        let yaw = p.yaw
        // World axes of the yawed particle.
        let right = SIMD3<Float>(cos(yaw), 0, -sin(yaw))
        let forward = SIMD3<Float>(sin(yaw), 0, cos(yaw))
        let hx = max(0.005, emitter.halfWidth * p.size * emitter.hitboxSize)
        let centreH = centreSupport.height

        let sRight = support(atX: centre.x + right.x * hx, z: centre.z + right.z * hx,
                             fromY: centreH + 0.3)
        let sLeft = support(atX: centre.x - right.x * hx, z: centre.z - right.z * hx,
                            fromY: centreH + 0.3)
        func isSupported(_ s: Support?) -> Bool {
            guard let s else { return false }
            return !(s.height < centreH - 0.015)
        }
        let rightOK = isSupported(sRight)
        let leftOK = isSupported(sLeft)
        if rightOK == leftOK { return nil }   // both or neither: flat

        // Hinge on the supported edge.
        let sign: Float = rightOK ? 1 : -1
        let supportedH = (rightOK ? sRight : sLeft)!.height
        let lowSupport = rightOK ? sLeft : sRight
        let angleDeg: Float
        if let low = lowSupport {
            let drop = centreH - low.height + 0.01
            angleDeg = min(65, atan(drop / hx) * 180 / .pi)
        } else {
            angleDeg = 55
        }
        let angle = angleDeg * .pi / 180
        var pivot = centre + right * (sign * hx)
        pivot.y = supportedH + restOffset
        let centreOffset = centre - pivot
        // Pick the tilt sign that actually moves the unsupported
        // side DOWN (its offset from the pivot loses height).
        var tilt = simd_quaternion(angle, forward)
        if simd_act(tilt, centreOffset).y > centreOffset.y {
            tilt = simd_quaternion(-angle, forward)
        }
        let newCentre = pivot + simd_act(tilt, centreOffset)
        let quat = tilt * simd_quaternion(yaw, SIMD3<Float>(0, 1, 0))
        return (newCentre, quat)
    }

    // MARK: - Particle soups

    private func composeMatrix(pos: SIMD3<Float>, quat: simd_quatf, scale: Float) -> float4x4 {
        var m = float4x4(quat)
        m.columns.0 *= scale
        m.columns.1 *= scale
        m.columns.2 *= scale
        m.columns.3 = SIMD4<Float>(pos.x, pos.y, pos.z, 1)
        return m
    }

    private func appendMesh(_ src: MeshData, into dst: inout MeshData) {
        let base = UInt32(dst.vertexCount)
        dst.positions += src.positions
        dst.normals += src.normals
        dst.uvs += src.uvs
        dst.colors += src.colors
        dst.indices.reserveCapacity(dst.indices.count + src.indices.count)
        for index in src.indices { dst.indices.append(index + base) }
    }

    /// Builds one soup mesh per live emitter and hands them all to
    /// the session in a single call. Sends [] exactly once when the
    /// last particle dies so the renderer clears the soups.
    private func buildAndSendSoups() {
        guard running else { return }
        if emitters.isEmpty {
            if soupsSent { session?.runtimeSetParticleSoups([]); soupsSent = false }
            return
        }
        var soups: [(mesh: MeshData, materialObject: SceneObject)] = []
        for emitter in emitters.values where !emitter.particles.isEmpty {
            var soup = MeshData()
            for p in emitter.particles {
                var scale = p.size
                let remaining = p.lifetime - p.age
                if remaining < 0.4 { scale *= max(0, remaining / 0.4) }
                if scale <= 0.001 { continue }
                let matrix: float4x4
                if p.resting {
                    matrix = composeMatrix(pos: p.restPos, quat: p.restQuat, scale: scale)
                } else {
                    matrix = composeMatrix(pos: p.pos,
                                           quat: simd_quaternion(p.yaw, SIMD3<Float>(0, 1, 0)),
                                           scale: scale)
                }
                appendMesh(emitter.mesh.baked(by: matrix), into: &soup)
            }
            if !soup.isEmpty {
                let materialObject = session?.project.scene.object(id: emitter.sourceID)
                    ?? emitter.sourceObject
                soups.append((mesh: soup, materialObject: materialObject))
            }
        }
        if soups.isEmpty {
            if soupsSent { session?.runtimeSetParticleSoups([]); soupsSent = false }
        } else {
            session?.runtimeSetParticleSoups(soups)
            soupsSent = true
        }
    }
}
