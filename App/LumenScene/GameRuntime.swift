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
// (never recursion) with a hard cap of 1024 deliveries per drain
// phase; fires past the cap stay queued for the next drain rather
// than being dropped (F9).
// Determinism: a seeded splitmix64 RNG drives particle spawns,
// logRandom and cone jitter, reset on every start(), seeded, so
// identical frame sequences reproduce (F35).

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
    /// Watcher lists precomputed at start() (recon §10.9) so the
    /// per-tick scans never walk the whole graph snapshot.
    private var areaEventNodes: [LogicNode] = []
    private var tapNodes: [LogicNode] = []
    private var variableWatcherNodes: [LogicNode] = []
    private var proximityNodes: [LogicNode] = []
    private var fallNodes: [LogicNode] = []

    /// Runtime clock: seconds since start(), advanced by tick dt.
    private var clock: Double = 0

    // MARK: - Logic state

    private var variables: [String: Double] = [:]
    private var toggleNextOn: [UUID: Bool] = [:]   // absent = next fire is "on"
    private var counterCounts: [UUID: Int] = [:]
    private var onceFired: Set<UUID> = []          // logOnce node ids that have fired
    private var sequenceStep: [UUID: Int] = [:]    // logSequence step per node id

    /// Player position as of the most recent tick start, for code
    /// paths that run inside processQueue (recon §8.0, pinned).
    private var lastPlayerPos: SIMD3<Float> = .zero
    /// Proximity event node ids the player was inside as of the
    /// last tick (edge detection, mirroring insideAreas).
    private var proximityInside: Set<UUID> = []
    /// Fall-below event node ids currently armed (re-arm above
    /// height + 0.5, fire on the downward crossing).
    private var fallArmed: Set<UUID> = []

    // MARK: - Continuous-behaviour slots (recon §8.0)

    private enum MovementKind {
        case oscillate, follow, chase, wander
    }

    /// One Movement-slot occupant. Owns transform.position only.
    private struct MovementBehaviour {
        var kind: MovementKind
        var nodeID: UUID
        var startClock: Double = 0
        // Oscillate
        var anchor: SIMD3<Float> = .zero
        var axis: Int = 1
        var amplitude: Double = 0
        var period: Double = 1
        // Shared locomotion
        var speed: Double = 0
        var home: SIMD3<Float> = .zero
        // Follow
        var targetID: UUID? = nil
        var mode: String = "PingPong"
        var goingToTarget: Bool = true
        // Chase
        var stopDistance: Double = 0
        var fly: Bool = false
        var caughtArmed: Bool = true
        // Wander
        var radius: Double = 0
        var pause: Double = 0
        var wanderTarget: SIMD3<Float>? = nil
        var idleUntil: Double = 0
    }
    private var movement: [UUID: MovementBehaviour] = [:]   // keyed by object id

    /// One Facing-slot occupant (LookAt). Owns rotation only.
    /// Rotate stays in `rotations`; the two cancel each other.
    private struct FacingBehaviour {
        var nodeID: UUID
        var targetChoice: String   // "Player" or "Object"
        var targetObjectID: UUID? = nil
        var turnSpeed: Double = 180
    }
    private var facing: [UUID: FacingBehaviour] = [:]   // keyed by object id

    /// One Scale-slot occupant. Owns transform.scale only.
    private struct ScaleTween {
        var nodeID: UUID
        var fromScale: [Double]
        var target: Double
        var duration: Double
        var elapsed: Double
    }
    private var scaleTweens: [UUID: ScaleTween] = [:]   // keyed by object id

    private enum LightFxKind {
        case pulse, colorCycle
    }

    /// One LightFx-slot occupant. Owns the material's emissive
    /// fields only. PulseLight and ColorCycle share the slot.
    private struct LightFx {
        var kind: LightFxKind
        var nodeID: UUID
        var startClock: Double = 0
        // Pulse
        var minIntensity: Double = 0
        var maxIntensity: Double = 0
        var period: Double = 1
        // Colour cycle (HSV captured from the emissive colour at start)
        var hue: Double = 0
        var saturation: Double = 0
        var value: Double = 0
        var speed: Double = 0   // hue degrees / second
    }
    private var lightFx: [UUID: LightFx] = [:]   // keyed by object id

    // MARK: - Spawned clones (runtime owns lifetimes, recon §7.4)

    private var cloneSource: [UUID: UUID] = [:]   // cloneID -> source object id
    private var cloneExpiry: [UUID: Double] = [:] // cloneID -> clock expiry (absent = immortal)

    // MARK: - Player pose (checkpoint / respawn)

    private var checkpointPose: (position: SIMD3<Float>, yaw: Double, pitch: Double)? = nil
    private var playEntryPose: (position: SIMD3<Float>, yaw: Double, pitch: Double) =
        (SIMD3<Float>(0, 1.7, 3.5), 0, 0)

    /// Global particle gravity multiplier (actSetGravity).
    private var particleGravityScale: Double = 1

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
    /// tap). Reset ONCE by the entry point and shared by every
    /// drain it triggers (v4.2: tick's two drains draw on ONE
    /// budget — previously the post-advance drain reset it, which
    /// handed a self-sustaining fire cycle a fresh budget every
    /// phase, forever). Capped at deliveryCap; past the cap the
    /// drain stops and undelivered fires stay queued (F9).
    private var deliveriesUsed = 0
    private let deliveryCap = 1024
    /// v4.2 storm breaker: consecutive ticks whose final drain
    /// still could not empty the queue within the tick budget.
    /// A finite chain always drains within a few ticks; only a
    /// self-sustaining cycle (a graph loop that re-seeds itself
    /// every drain) saturates indefinitely — after
    /// stormTickLimit consecutive saturated ticks the backlog is
    /// cleared so no wiring can pin the main thread forever.
    private var saturatedTicks = 0
    private let stormTickLimit = 4

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
    /// Indices of triangles whose XZ span exceeds the grid cap;
    /// support() always scans these in addition to the cell list
    /// (F11), so large flat geometry is collidable everywhere.
    private var bigCollisionTris: [Int] = []
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
    /// F27: soups are rebuilt/resent only when something actually
    /// changed (spawn, death, a flying/shrinking/un-rested
    /// particle, an emitter-set change). All-resting, non-shrinking
    /// soups are left with the renderer untouched.
    private var soupsDirty = true

    // MARK: - Lifecycle

    func start() {
        guard let session else { return }
        stopInternal(clearSoups: true)
        running = true
        clock = 0
        rng = SplitMix64(state: 0x51AB_C0DE_1234_5678)

        // Snapshot the graph and index it. F7: catalogue defaults
        // are merged UNDER each node's stored params (stored wins),
        // so the runtime's per-call literal fallbacks are
        // unreachable for catalogued kinds and old/hand-edited
        // projects run with the catalogue's values.
        let graph = session.project.scene.graph
        snapshotNodes = graph.nodes.map { stored in
            var node = stored
            var merged = NodeCatalog.defaultParams(for: stored.kind)
            for (key, value) in stored.params { merged[key] = value }
            node.params = merged
            return node
        }
        nodesByID = Dictionary(snapshotNodes.map { ($0.id, $0) },
                               uniquingKeysWith: { first, _ in first })
        linksByPin = [:]
        for link in graph.links {
            linksByPin[pinKey(link.from, link.fromPin), default: []].append(link)
        }
        hasAreaEvents = snapshotNodes.contains { $0.kind == "evEnterArea" || $0.kind == "evExitArea" }
        insideAreas = []
        // Precompute per-kind watcher lists (recon §10.9).
        areaEventNodes = snapshotNodes.filter { $0.kind == "evEnterArea" || $0.kind == "evExitArea" }
        tapNodes = snapshotNodes.filter { $0.kind == "evTap" }
        variableWatcherNodes = snapshotNodes.filter { $0.kind == "evVariable" }
        proximityNodes = snapshotNodes.filter { $0.kind == "evProximity" }
        fallNodes = snapshotNodes.filter { $0.kind == "evFallBelow" }
        proximityInside = []

        // Capture the Play-entry pose BEFORE any graph runs, so
        // Respawn Player has somewhere to go before a checkpoint.
        let camera = session.project.scene.camera
        playEntryPose = (position: SIMD3<Float>(
                            Float(camera.playPosition.count > 0 ? camera.playPosition[0] : 0.0),
                            Float(camera.playPosition.count > 1 ? camera.playPosition[1] : 1.7),
                            Float(camera.playPosition.count > 2 ? camera.playPosition[2] : 3.5)),
                         yaw: camera.playYaw, pitch: camera.playPitch)
        checkpointPose = nil
        // v4.2: the player position is valid from the first tick
        // onward — a spawn wired to Play Starts can already read
        // it (Spawn at Player) before any tick has run.
        lastPlayerPos = playEntryPose.position
        // N6: a fall node starts armed only when the player starts
        // at or above its height — starting below the line is not
        // a drop. The existing re-arm logic in
        // checkProximityAndFall (rise above height + 0.5) arms it
        // later if the player climbs above the line.
        fallArmed = Set(fallNodes.filter {
            Float($0.num("height", 0)) <= playEntryPose.position.y
        }.map { $0.id })

        buildCollisionGrid()

        // Arm timers (F5): the first fire is at `delay`; after a
        // fire, fireDueTimers advances by `interval` only while
        // repeating. A non-positive interval can only ever be a
        // one-shot at delay.
        for node in snapshotNodes where node.kind == "evTimer" {
            let delay = node.num("delay", 0)
            let interval = node.num("interval", 5)
            let repeating = node.boolean("repeat", true) && interval > 0
            timers.append(TimerState(nodeID: node.id,
                                     nextFire: delay,
                                     interval: interval,
                                     repeating: repeating))
        }

        // Fire evStart.
        deliveriesUsed = 0
        for node in snapshotNodes where node.kind == "evStart" {
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
        areaEventNodes = []
        tapNodes = []
        variableWatcherNodes = []
        proximityNodes = []
        fallNodes = []
        variables = [:]
        toggleNextOn = [:]
        counterCounts = [:]
        onceFired = []
        sequenceStep = [:]
        lastPlayerPos = .zero
        proximityInside = []
        fallArmed = []
        timers = []
        pendingDelays = []
        insideAreas = []
        trackedTransforms = [:]
        trackedMaterials = [:]
        tweens = [:]
        rotations = [:]
        movement = [:]
        facing = [:]
        scaleTweens = [:]
        lightFx = [:]
        cloneSource = [:]
        cloneExpiry = [:]
        checkpointPose = nil
        playEntryPose = (position: SIMD3<Float>(0, 1.7, 3.5), yaw: 0, pitch: 0)
        particleGravityScale = 1
        fireQueue = []
        deliveriesUsed = 0
        saturatedTicks = 0
        emitters = [:]
        decimationCache = [:]
        collisionTris = []
        collisionGrid = [:]
        bigCollisionTris = []
        soupsSent = false
        soupsDirty = true
        if clearSoups && hadSoups {
            session?.runtimeSetParticleSoups([])
        }
    }

    // MARK: - Frame need

    var needsFrame: Bool {
        guard running else { return false }
        if !tweens.isEmpty || !rotations.isEmpty { return true }
        if !pendingDelays.isEmpty || !timers.isEmpty { return true }
        // F31: an emitting emitter pins the frame loop only while
        // it can still produce something — positive rate, a finite
        // emission window still open, or live particles. (A rate-0
        // burst emitter with duration 0 has emitUntil = .infinity,
        // which must not count on its own.)
        for emitter in emitters.values {
            if !emitter.particles.isEmpty { return true }
            if emitter.emitting
                && (emitter.rate > 0
                    || (emitter.emitUntil.isFinite && clock < emitter.emitUntil)) {
                return true
            }
        }
        // Continuous-behaviour slots and per-tick event watchers
        // (recon §8.0: forgetting these freezes behaviours in a
        // still scene).
        if !movement.isEmpty || !facing.isEmpty { return true }
        if !scaleTweens.isEmpty || !lightFx.isEmpty { return true }
        if !proximityNodes.isEmpty || !fallNodes.isEmpty { return true }
        if !cloneExpiry.isEmpty { return true }
        // Walking must keep ticking so area enter/exit can fire;
        // the session draws continuously in Play while moving
        // anyway — this pin keeps the checks alive even at rest.
        if hasAreaEvents { return true }
        return false
    }

    // MARK: - Per-frame tick

    func tick(dt: Float, playerPos: SIMD3<Float>) {
        guard running, session != nil else { return }
        // F22: clamp the step — after a hitch, one tick must not
        // advance timers/tweens/particles by seconds in one leap.
        // The same clamped step drives emitter integration.
        let step = min(0.1, max(0, Double(dt)))
        clock += step
        lastPlayerPos = playerPos
        deliveriesUsed = 0

        fireDueTimers()
        fireDueDelays()
        checkAreas(playerPos: playerPos)
        checkProximityAndFall(playerPos: playerPos)
        // v4.2: ONE delivery budget per tick, shared by both
        // drains. The first drain is soft-capped at 3/4 of the
        // budget so behaviour fires raised during the advance
        // phase (tween "arrived", chase "caught") always keep a
        // reserve — the F9 starvation concern — without handing
        // a self-sustaining cycle a second full budget per tick.
        processQueue(limit: deliveryCap * 3 / 4)

        advanceTweens(dt: step)
        advanceRotations(dt: step)
        advanceMovement(dt: step)
        advanceFacing(dt: step)
        advanceScaleTweens(dt: step)
        applyLightFx()
        expireClones()
        processQueue(limit: deliveryCap)
        // v4.2 storm breaker: if the post-advance drain STILL
        // ends with undelivered fires, this tick was saturated.
        // Finite chains clear the counter by draining; a cycle
        // that re-seeds itself every drain saturates every tick,
        // and after stormTickLimit in a row its backlog is
        // dropped — the one deliberate drop, so no graph wiring
        // (e.g. a caught chain looped back on itself) can freeze
        // the app. Ordinary chains never come near this: the
        // breaker needs >~4,000 undelivered deliveries sustained
        // across four consecutive ticks.
        if fireQueue.isEmpty {
            saturatedTicks = 0
        } else {
            saturatedTicks += 1
            if saturatedTicks >= stormTickLimit {
                fireQueue.removeAll()
                saturatedTicks = 0
            }
        }

        updateEmitters(dt: Float(step))
        buildAndSendSoups()
    }

    // MARK: - Play-mode tap

    func handleTap(objectID: UUID) {
        guard running else { return }
        deliveriesUsed = 0
        for node in tapNodes {
            // F17: a nil ref is INERT (no object picked yet),
            // matching area events — not a wildcard.
            guard let target = node.ref("object") else { continue }
            if target != objectID { continue }
            enqueueFire(node.id, "out")
        }
        processQueue()
    }

    // MARK: - Collision rebuild hook

    /// Called by the session after an area load/unload rebuilds
    /// the renderer's collision set.
    func collisionDidChange() {
        buildCollisionGrid()
        // F12 (runtime half): resting particles re-check their
        // support against the new grid — if the floor under one
        // moved/went away, it un-rests and falls to re-settle.
        for emitter in emitters.values {
            for i in emitter.particles.indices {
                var p = emitter.particles[i]
                guard p.resting else { continue }
                let sup = support(atX: p.restPos.x, z: p.restPos.z, fromY: p.restPos.y)
                let expectedHeight = p.restPos.y - p.restOffset
                if sup == nil || abs(sup!.height - expectedHeight) > 0.02 {
                    p.resting = false
                    p.pos = p.restPos
                    p.vel = .zero
                    soupsDirty = true
                }
                emitter.particles[i] = p
            }
        }
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
        for node in areaEventNodes {
            guard let areaID = node.ref("area") else { continue }
            if node.kind == "evEnterArea" && entered.contains(areaID) {
                enqueueFire(node.id, "out")
            } else if node.kind == "evExitArea" && exited.contains(areaID) {
                enqueueFire(node.id, "out")
            }
        }
    }

    /// Per-tick edge detection for evProximity (distance from the
    /// player to the watched object's CURRENT position) and
    /// evFallBelow (downward crossing with a +0.5 re-arm band).
    private func checkProximityAndFall(playerPos: SIMD3<Float>) {
        guard let session else { return }
        for node in proximityNodes {
            guard let objectID = node.ref("object"),
                  let obj = session.project.scene.object(id: objectID) else { continue }
            let objectPos = position(of: currentTransform(of: obj))
            let inside = simd_distance(playerPos, objectPos) <= Float(node.num("radius", 3))
            let wasInside = proximityInside.contains(node.id)
            if inside && !wasInside {
                proximityInside.insert(node.id)
                enqueueFire(node.id, "enter")
            } else if !inside && wasInside {
                proximityInside.remove(node.id)
                enqueueFire(node.id, "exit")
            }
        }
        for node in fallNodes {
            let height = Float(node.num("height", 0))
            if playerPos.y < height {
                if fallArmed.contains(node.id) {
                    fallArmed.remove(node.id)
                    enqueueFire(node.id, "out")
                }
            } else if playerPos.y > height + 0.5 {
                fallArmed.insert(node.id)
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
    /// enqueue further fires, which the same drain picks up.
    /// F9: past the delivery limit the drain STOPS and the
    /// undelivered fires stay queued for the next entry point —
    /// no fire is silently dropped (the storm breaker in tick()
    /// is the one deliberate exception), and a fire is never
    /// delivered half-way (the limit is checked against a whole
    /// fire's links before it is consumed, except for a single
    /// fire whose own link count alone exceeds the remaining
    /// budget). `limit` is the entry point's total budget;
    /// deliveriesUsed carries across the drains of one entry
    /// point, so a limit below deliveryCap acts as a soft cap
    /// that reserves the remainder for a later drain.
    private func processQueue(limit: Int = 1024) {
        var head = 0
        while head < fireQueue.count {
            let fire = fireQueue[head]
            let links = linksByPin[pinKey(fire.from, fire.pin)] ?? []
            if deliveriesUsed > 0
                && deliveriesUsed + links.count > limit {
                break   // keep this fire (and everything after it) queued
            }
            head += 1
            for link in links {
                if deliveriesUsed >= limit { break }
                deliveriesUsed += 1
                if let target = nodesByID[link.to] {
                    execute(target, inputPin: link.toPin)
                }
            }
        }
        if head > 0 { fireQueue.removeFirst(head) }
    }

    /// F1: Int(Double) traps on NaN/inf/out-of-range file values —
    /// clamp (and round) BEFORE converting, everywhere.
    private func clampedInt(_ value: Double, _ lo: Int, _ hi: Int) -> Int {
        guard value.isFinite else { return lo }
        return Int(min(Double(hi), max(Double(lo), value.rounded())))
    }

    /// N5: Int(floor(...)) traps on non-finite / extreme mesh
    /// coordinates — convert a divided coordinate to a cell
    /// index safely: non-finite -> 0, floored value clamped to
    /// ±1_000_000 before the Int conversion.
    private func safeCellIndex(_ value: Float) -> Int {
        guard value.isFinite else { return 0 }
        let floored = Double(value.rounded(.down))
        return Int(min(1_000_000.0, max(-1_000_000.0, floored)))
    }

    /// Sets a variable and fires matching evVariable watchers —
    /// but ONLY when the value actually changed (F8). Add semantics
    /// stay local to actSetVariable; every other writer
    /// (actMath, actRandomValue) goes through here too (recon §8.0).
    private func setVariable(_ name: String, _ value: Double) {
        let old = variables[name] ?? 0
        guard value != old else { return }
        variables[name] = value
        for watcher in variableWatcherNodes
        where watcher.txt("name", "score") == name {
            enqueueFire(watcher.id, "out")
        }
    }

    /// HSV -> RGB (hue in degrees, saturation/value in 0...1),
    /// for actColorCycle's LightFx slot.
    private func hsvToRGB(h: Double, s: Double, v: Double) -> RGB {
        let hue = h.truncatingRemainder(dividingBy: 360)
        let hn = hue < 0 ? hue + 360 : hue
        let c = v * s
        let x = c * (1 - abs((hn / 60).truncatingRemainder(dividingBy: 2) - 1))
        let m = v - c
        let r: Double, g: Double, b: Double
        switch hn {
        case 0..<60:    r = c; g = x; b = 0
        case 60..<120:  r = x; g = c; b = 0
        case 120..<180: r = 0; g = c; b = x
        case 180..<240: r = 0; g = x; b = c
        case 240..<300: r = x; g = 0; b = c
        default:        r = c; g = 0; b = x
        }
        return RGB(r + m, g + m, b + m)
    }

    /// RGB -> HSV (hue in degrees, saturation/value in 0...1),
    /// used to capture a light's starting colour for cycling.
    private func hsvFromRGB(_ rgb: RGB) -> (h: Double, s: Double, v: Double) {
        let maxC = max(rgb.r, rgb.g, rgb.b)
        let minC = min(rgb.r, rgb.g, rgb.b)
        let delta = maxC - minC
        var h: Double = 0
        if delta > 1e-9 {
            if maxC == rgb.r {
                h = 60 * ((rgb.g - rgb.b) / delta).truncatingRemainder(dividingBy: 6)
            } else if maxC == rgb.g {
                h = 60 * ((rgb.b - rgb.r) / delta + 2)
            } else {
                h = 60 * ((rgb.r - rgb.g) / delta + 4)
            }
        }
        if h < 0 { h += 360 }
        let s = maxC > 1e-9 ? delta / maxC : 0
        return (h, s, maxC)
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
            // F10: "out" is enqueued BEFORE the move starts, so an
            // instant (duration 0) move's consumers see out ->
            // arrived, matching the tweened path's ordering.
            enqueueFire(node.id, "out")
            startMoveTo(node)

        case "actRotate":
            if let id = node.ref("object") {
                // Slot rule: Rotate is a Facing occupant — starting
                // it cancels any LookAt on the same object.
                facing[id] = nil
                // Replace-on-retrigger: assigning over any old entry.
                rotations[id] = RotateState(speed: node.num("speed", 45),
                                            duration: node.num("duration", 0),
                                            elapsed: 0)
            }
            enqueueFire(node.id, "out")

        case "actSetLight":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                // Slot rule: a one-shot light write cancels any
                // active LightFx on the object before writing.
                lightFx[id] = nil
                var material = currentMaterial(of: obj)
                material.emissiveOn = node.boolean("on", true)
                material.emissiveIntensity = node.num("intensity", 4)
                if let colour = parseHexColour(node.txt("color", "#FF9A33")) {
                    material.emissiveColor = colour
                } // unparseable colour: keep the existing one
                trackedMaterials[id] = material
                session.runtimeSetMaterial(id, material)
                soupsDirty = true   // a soup source's material may have changed
            }
            enqueueFire(node.id, "out")

        case "actSetMaterial":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                // Slot rule: cancels LightFx before writing (§8.0).
                lightFx[id] = nil
                var material = currentMaterial(of: obj)
                let texture = node.txt("texture", "none")
                material.texturePreset = texture
                material.glassOn = node.boolean("glass", false)
                // Roads are the known auto-fit texture family (their
                // markings live in a single tile); everything else tiles.
                material.textureFit = texture.hasPrefix("road-")
                trackedMaterials[id] = material
                session.runtimeSetMaterial(id, material)
                soupsDirty = true   // a soup source's material may have changed
            }
            enqueueFire(node.id, "out")

        case "actParticles":
            // F6: "out" fires after BOTH the start and stop paths.
            if inputPin == "stop" {
                emitters[node.id]?.emitting = false
            } else {
                startEmitter(node)
            }
            enqueueFire(node.id, "out")

        case "actLoadArea":
            if let id = node.ref("area") { session.runtimeLoadArea(id) }
            enqueueFire(node.id, "out")

        case "actUnloadArea":
            if let id = node.ref("area") { session.runtimeUnloadArea(id) }
            enqueueFire(node.id, "out")

        case "actSetWeather":
            // F26: only Rain/Snow/Off map — an unknown mode string
            // (a newer build's "Fog", say) is NO weather change,
            // never a silent switch-off. "out" still fires.
            switch node.txt("mode", "Rain") {
            case "Rain":
                session.runtimeSetWeather(mode: 1, clear: node.boolean("clear", false))
            case "Snow":
                session.runtimeSetWeather(mode: 2, clear: node.boolean("clear", false))
            case "Off":
                session.runtimeSetWeather(mode: 0, clear: node.boolean("clear", false))
            default:
                break
            }
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
            session.runtimeShowMessage(node.txt("message", "Hello"),
                                       seconds: node.num("seconds", 3))
            enqueueFire(node.id, "out")

        case "actSetVariable":
            let name = node.txt("name", "score")
            if !name.isEmpty {
                let value = node.num("value", 1)
                if node.txt("op", "Set") == "Add" {
                    // Add stays local; the write (and watcher
                    // firing, only on a real change — F8) goes
                    // through setVariable.
                    setVariable(name, (variables[name] ?? 0) + value)
                } else {
                    setVariable(name, value)
                }
            }
            enqueueFire(node.id, "out")

        case "actSpawn":
            if let sourceID = node.ref("source"),
               let source = session.project.scene.object(id: sourceID) {
                // Spawn at the `at` object's current transform, or
                // the source's own when none is picked. The source's
                // CURRENT runtime material is passed explicitly as
                // the override, per the spawn contract.
                var transform = currentTransform(of: source)
                if node.boolean("atPlayer", false) {
                    // v4.2: spawn at the player — the clone keeps
                    // the source's facing/scale and lands at the
                    // player's feet (eye position − 1.7 eye
                    // height; Play locks eye height, so this is
                    // the floor the player is standing on).
                    transform.position = [Double(lastPlayerPos.x),
                                          Double(lastPlayerPos.y - 1.7),
                                          Double(lastPlayerPos.z)]
                } else if let atID = node.ref("at"),
                   let atObject = session.project.scene.object(id: atID) {
                    transform = currentTransform(of: atObject)
                }
                // v4.2: displacement — the clone lands offset
                // from the anchor instead of exactly inside it.
                applySpawnOffset(node, to: &transform)
                let material = currentMaterial(of: source)
                let ids = session.runtimeSpawnClones(
                    [(sourceID: sourceID, transform: transform, material: material)])
                let lifetime = node.num("lifetime", 0)
                for cloneID in ids {
                    cloneSource[cloneID] = sourceID
                    if lifetime > 0 { cloneExpiry[cloneID] = clock + lifetime }
                }
            }
            enqueueFire(node.id, "out")

        case "actSpawnPattern":
            if let sourceID = node.ref("source"),
               let source = session.project.scene.object(id: sourceID) {
                let template = currentTransform(of: source)
                // v4.2: the pattern centre is the source's
                // position, or the player's feet with Centre on
                // Player; the offset then shifts the whole centre
                // — it composes with the pattern's own placement
                // (every clone keeps its pattern position relative
                // to the shifted centre).
                var centre = position(of: template)
                if node.boolean("atPlayer", false) {
                    centre = SIMD3<Float>(lastPlayerPos.x,
                                          lastPlayerPos.y - 1.7,
                                          lastPlayerPos.z)
                }
                centre += spawnOffset(of: node)
                let count = clampedInt(node.num("count", 8), 1, 64)
                let spacing = Float(node.num("spacing", 1.5))
                let radius = Float(node.num("radius", 4))
                let randomYaw = node.boolean("randomYaw", true)
                // Scatter/random yaw use a LOCAL SplitMix64 seeded
                // from the node's seed — never the shared rng, so
                // patterns are reproducible and don't perturb the
                // session's random stream.
                var localRng = SplitMix64(
                    state: UInt64(clampedInt(node.num("seed", 1), 0, 100000)))
                var transforms: [Transform3] = []
                transforms.reserveCapacity(count)
                let pattern = node.txt("pattern", "Grid")
                for i in 0..<count {
                    var t = template
                    var pos = centre
                    var yawDegrees: Double? = nil
                    switch pattern {
                    case "Ring":
                        let angle = 2 * Float.pi * Float(i) / Float(count)
                        pos = SIMD3<Float>(centre.x + radius * cos(angle),
                                           centre.y,
                                           centre.z + radius * sin(angle))
                        // Face outward from the ring centre.
                        yawDegrees = Double(atan2(-cos(angle), -sin(angle))) * 180 / .pi
                    case "Line":
                        pos = SIMD3<Float>(
                            centre.x + (Float(i) - Float(count - 1) / 2) * spacing,
                            centre.y, centre.z)
                    case "Scatter":
                        let angle = localRng.range(0, 2 * Float.pi)
                        let r = radius * sqrt(localRng.nextFloat())
                        pos = SIMD3<Float>(centre.x + r * cos(angle),
                                           centre.y,
                                           centre.z + r * sin(angle))
                    default: // "Grid"
                        let cols = Int(ceil(sqrt(Double(count))))
                        let rows = (count + cols - 1) / cols
                        let col = i % cols, row = i / cols
                        pos = SIMD3<Float>(
                            centre.x + (Float(col) - Float(cols - 1) / 2) * spacing,
                            centre.y,
                            centre.z + (Float(row) - Float(rows - 1) / 2) * spacing)
                    }
                    if randomYaw && pattern != "Ring" {
                        // N13: Ring sets an outward-facing yaw
                        // above; random yaw must not overwrite it.
                        // Grid/Line/Scatter keep random facing.
                        yawDegrees = localRng.range(0.0 as Double, 360.0 as Double)
                    }
                    t.position = [Double(pos.x), Double(pos.y), Double(pos.z)]
                    if let yawDegrees, t.rotation.count >= 2 {
                        var yaw = yawDegrees.truncatingRemainder(dividingBy: 360)
                        if yaw < 0 { yaw += 360 }
                        t.rotation[1] = yaw
                    }
                    transforms.append(t)
                }
                let material = currentMaterial(of: source)
                let requests: [(sourceID: UUID, transform: Transform3,
                                 material: MaterialProps?)] =
                    transforms.map { (sourceID: sourceID, transform: $0, material: material) }
                let ids = session.runtimeSpawnClones(requests)
                let lifetime = node.num("lifetime", 0)
                for cloneID in ids {
                    cloneSource[cloneID] = sourceID
                    if lifetime > 0 { cloneExpiry[cloneID] = clock + lifetime }
                }
            }
            enqueueFire(node.id, "out")

        case "actDespawn":
            if let sourceID = node.ref("source") {
                // Iterate our own cloneSource map and despawn per
                // id, so the runtime's maps stay consistent.
                let ids = cloneSource.filter { $0.value == sourceID }.map { $0.key }
                for cloneID in ids {
                    session.runtimeDespawnClone(cloneID)
                    cloneSource[cloneID] = nil
                    cloneExpiry[cloneID] = nil
                }
            }
            enqueueFire(node.id, "out")

        case "actSetColor":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                var material = currentMaterial(of: obj)
                if let colour = parseHexColour(node.txt("color", "#FFFFFF")) {
                    material.baseColor = colour
                    trackedMaterials[id] = material
                    session.runtimeSetMaterial(id, material)
                    soupsDirty = true
                } // unparseable colour: keep the existing one
            }
            enqueueFire(node.id, "out")

        case "actPulseLight":
            if let id = node.ref("object") {
                if inputPin == "stop" {
                    // Vacate the slot; the material stays at its
                    // last written value.
                    lightFx[id] = nil
                } else if session.project.scene.object(id: id) != nil {
                    lightFx[id] = LightFx(
                        kind: .pulse, nodeID: node.id, startClock: clock,
                        minIntensity: node.num("min", 0),
                        maxIntensity: node.num("max", 6),
                        period: node.num("period", 2))
                }
            }
            enqueueFire(node.id, "out")

        case "actColorCycle":
            if let id = node.ref("object") {
                if inputPin == "stop" {
                    lightFx[id] = nil
                } else if let obj = session.project.scene.object(id: id) {
                    let hsv = hsvFromRGB(currentMaterial(of: obj).emissiveColor)
                    lightFx[id] = LightFx(
                        kind: .colorCycle, nodeID: node.id, startClock: clock,
                        hue: hsv.h, saturation: hsv.s, value: hsv.v,
                        speed: node.num("speed", 60))
                }
            }
            enqueueFire(node.id, "out")

        case "actOscillate":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                let axis: Int
                switch node.txt("axis", "Y") {
                case "X": axis = 0
                case "Z": axis = 2
                default: axis = 1   // "Y"
                }
                var behaviour = MovementBehaviour(kind: .oscillate, nodeID: node.id)
                behaviour.startClock = clock
                behaviour.anchor = position(of: currentTransform(of: obj))
                behaviour.axis = axis
                behaviour.amplitude = node.num("amplitude", 1)
                behaviour.period = node.num("period", 3)
                movement[id] = behaviour
                tweens[id] = nil   // slot rule: cancels the MoveTo tween
            }
            enqueueFire(node.id, "out")

        case "actFollowPath":
            // N7: a self-target would "arrive" every tick — do not
            // start the behaviour at all ("out" still fires below,
            // vacating silently like other invalid targets).
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id),
               node.ref("target") != id {
                var behaviour = MovementBehaviour(kind: .follow, nodeID: node.id)
                behaviour.home = position(of: currentTransform(of: obj))
                behaviour.targetID = node.ref("target")
                behaviour.speed = node.num("speed", 1.5)
                behaviour.mode = node.txt("mode", "PingPong")
                movement[id] = behaviour
                tweens[id] = nil
            }
            enqueueFire(node.id, "out")

        case "actChasePlayer":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                var behaviour = MovementBehaviour(kind: .chase, nodeID: node.id)
                behaviour.speed = node.num("speed", 2)
                behaviour.stopDistance = node.num("stopDistance", 1.2)
                behaviour.fly = node.boolean("fly", false)
                // v4.2: a chase (re)started while the pair are
                // already inside the re-arm radius starts
                // DISARMED — otherwise wiring Caught back into
                // Chase Player refires Caught on the very next
                // tick, every tick, for as long as they stay in
                // range. Arming follows the tick's own hysteresis:
                // armed only when currently beyond 1.5× the catch
                // radius (same caughtRadius as advanceMovement).
                var toPlayer = lastPlayerPos
                    - position(of: currentTransform(of: obj))
                if !behaviour.fly { toPlayer.y = 0 }
                let caughtRadius = max(Float(behaviour.stopDistance), 0.05)
                behaviour.caughtArmed = simd_length(toPlayer) > caughtRadius * 1.5
                movement[id] = behaviour
                tweens[id] = nil
            }
            enqueueFire(node.id, "out")

        case "actWander":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                var behaviour = MovementBehaviour(kind: .wander, nodeID: node.id)
                behaviour.home = position(of: currentTransform(of: obj))
                behaviour.radius = node.num("radius", 5)
                behaviour.speed = node.num("speed", 1)
                behaviour.pause = node.num("pause", 1)
                movement[id] = behaviour
                tweens[id] = nil
            }
            enqueueFire(node.id, "out")

        case "actLookAt":
            if let id = node.ref("object"),
               session.project.scene.object(id: id) != nil {
                facing[id] = FacingBehaviour(
                    nodeID: node.id,
                    targetChoice: node.txt("target", "Player"),
                    targetObjectID: node.ref("targetObject"),
                    turnSpeed: node.num("turnSpeed", 180))
                rotations[id] = nil   // slot rule: cancels Rotate
            }
            enqueueFire(node.id, "out")

        case "actStopBehaviour":
            if let id = node.ref("object") {
                switch node.txt("scope", "All") {
                case "Movement":
                    movement[id] = nil
                    tweens[id] = nil
                case "Facing":
                    facing[id] = nil
                    rotations[id] = nil
                case "Scale":
                    scaleTweens[id] = nil
                case "Light":
                    lightFx[id] = nil
                default: // "All"
                    movement[id] = nil
                    tweens[id] = nil
                    facing[id] = nil
                    rotations[id] = nil
                    scaleTweens[id] = nil
                    lightFx[id] = nil
                }
            }
            enqueueFire(node.id, "out")

        case "actScaleTo":
            if let id = node.ref("object"),
               let obj = session.project.scene.object(id: id) {
                let target = node.num("scale", 2)
                let duration = node.num("duration", 1)
                if duration <= 0 {
                    scaleTweens[id] = nil
                    var t = currentTransform(of: obj)
                    t.scale = [target, target, target]
                    applyTransform(id, t)
                    // Instant path ordering: out BEFORE arrived,
                    // matching the tweened path (see F10 pattern).
                    enqueueFire(node.id, "out")
                    enqueueFire(node.id, "arrived")
                } else {
                    // Replacing an in-flight scale tween drops the
                    // old one's "arrived" (documented slot rule).
                    var from = currentTransform(of: obj).scale
                    while from.count < 3 { from.append(1) }
                    scaleTweens[id] = ScaleTween(nodeID: node.id,
                                                 fromScale: Array(from.prefix(3)),
                                                 target: target,
                                                 duration: duration, elapsed: 0)
                    enqueueFire(node.id, "out")
                }
            } else {
                enqueueFire(node.id, "out")
            }

        case "actSetGravity":
            let scale = node.num("scale", 1)
            particleGravityScale = scale.isFinite ? min(4, max(0, scale)) : 1
            enqueueFire(node.id, "out")

        case "actImpulse":
            if let anchorID = node.ref("object"),
               let anchor = session.project.scene.object(id: anchorID) {
                let anchorPos = position(of: currentTransform(of: anchor))
                let radius = Float(node.num("radius", 3))
                let strength = Float(node.num("strength", 6))
                if radius > 0 {
                for emitter in emitters.values {
                    for i in emitter.particles.indices {
                        var p = emitter.particles[i]
                        let particlePos = p.resting ? p.restPos : p.pos
                        let delta = particlePos - anchorPos
                        let dist = simd_length(delta)
                        guard dist <= radius else { continue }
                        let dir = dist > 1e-6 ? delta / dist
                                               : SIMD3<Float>(0, 1, 0)
                        let impulse = dir * strength * (1 - dist / radius)
                        if p.resting {
                            // Un-rest: flight resumes from the
                            // visible rest spot with the impulse.
                            p.resting = false
                            p.pos = p.restPos
                            p.vel = impulse
                            soupsDirty = true
                        } else {
                            p.vel += impulse
                        }
                        emitter.particles[i] = p
                    }
                }
                }
            }
            enqueueFire(node.id, "out")

        case "actStopAllParticles":
            let clear = node.boolean("clear", false)
            for emitter in emitters.values {
                emitter.emitting = false
                if clear { emitter.particles.removeAll() }
            }
            if clear { soupsDirty = true }
            enqueueFire(node.id, "out")

        case "actAreaVisibility":
            if let areaID = node.ref("area") {
                let visible = node.boolean("visible", true)
                for obj in session.project.scene.objects where obj.areaID == areaID {
                    session.runtimeSetVisible(obj.id, visible)
                }
            }
            enqueueFire(node.id, "out")

        case "actSetPlayerSpeed":
            session.runtimeSetPlayerSpeed(node.num("speed", 2.4))
            enqueueFire(node.id, "out")

        case "actSetCheckpoint":
            let camera = session.project.scene.camera
            checkpointPose = (position: SIMD3<Float>(
                                Float(camera.playPosition.count > 0 ? camera.playPosition[0] : 0.0),
                                Float(camera.playPosition.count > 1 ? camera.playPosition[1] : 1.7),
                                Float(camera.playPosition.count > 2 ? camera.playPosition[2] : 3.5)),
                              yaw: camera.playYaw, pitch: camera.playPitch)
            enqueueFire(node.id, "out")

        case "actRespawnPlayer":
            let pose = checkpointPose ?? playEntryPose
            session.runtimeSetPlayerPose(position: pose.position,
                                         yaw: pose.yaw, pitch: pose.pitch)
            enqueueFire(node.id, "out")

        case "actCameraShake":
            session.runtimeCameraShake(amplitude: node.num("strength", 0.3),
                                       seconds: node.num("seconds", 0.5))
            enqueueFire(node.id, "out")

        case "actCutToView":
            let camera = session.project.scene.camera
            let currentPos = SIMD3<Float>(
                Float(camera.playPosition.count > 0 ? camera.playPosition[0] : 0.0),
                Float(camera.playPosition.count > 1 ? camera.playPosition[1] : 1.7),
                Float(camera.playPosition.count > 2 ? camera.playPosition[2] : 3.5))
            if node.txt("mode", "Look At Object") == "Angles" {
                session.runtimeSetPlayerPose(position: currentPos,
                                             yaw: node.num("yaw", 0),
                                             pitch: node.num("pitch", 0))
            } else if let targetID = node.ref("target"),
                      let target = session.project.scene.object(id: targetID) {
                // Position unchanged; yaw/pitch computed from the
                // player's live position toward the target's
                // current position (recon §3.5 sign convention).
                let targetPos = position(of: currentTransform(of: target))
                let dx = Double(targetPos.x - lastPlayerPos.x)
                let dy = Double(targetPos.y - lastPlayerPos.y)
                let dz = Double(targetPos.z - lastPlayerPos.z)
                let horiz = sqrt(dx * dx + dz * dz)
                let yaw = atan2(-dx, -dz) * 180 / .pi
                let pitch = min(85, max(-85, atan2(dy, horiz) * 180 / .pi))
                session.runtimeSetPlayerPose(position: currentPos,
                                             yaw: yaw, pitch: pitch)
            }
            enqueueFire(node.id, "out")

        case "actSetEnvironment":
            session.runtimeSetEnvironmentIntensity(node.num("intensity", 1))
            enqueueFire(node.id, "out")

        case "actWeatherControl":
            // Both scales always forward; the wet/snow levels
            // forward only when their "set" bool is on (nil =
            // leave that channel untouched).
            session.runtimeSetWeatherControl(
                rateScale: node.num("rateScale", 1),
                intensityScale: node.num("intensityScale", 1),
                wet: node.boolean("setWet", false) ? node.num("wet", 0.5) : nil,
                snow: node.boolean("setSnow", false) ? node.num("snow", 0.5) : nil)
            enqueueFire(node.id, "out")

        case "actShowValue":
            session.runtimeShowMessage(substituteVariables(node.txt("message", "Score: {score}")),
                                       seconds: node.num("seconds", 3))
            enqueueFire(node.id, "out")

        case "actMath":
            let name = node.txt("name", "score")
            if !name.isEmpty {
                let a = variables[name] ?? 0
                let b = node.boolean("bIsVariable", false)
                    ? (variables[node.txt("bName", "other")] ?? 0)
                    : node.num("value", 1)
                let result: Double?
                switch node.txt("op", "Add") {
                case "Subtract": result = a - b
                case "Multiply": result = a * b
                case "Divide": result = b == 0 ? nil : a / b
                case "Modulo": result = b == 0 ? nil : a.truncatingRemainder(dividingBy: b)
                case "Min": result = min(a, b)
                case "Max": result = max(a, b)
                case "Add": result = a + b
                default: result = nil   // N14: unknown op — leave the value unchanged
                }
                // Divide/Modulo by zero leaves the value unchanged.
                if let result { setVariable(name, result) }
            }
            enqueueFire(node.id, "out")

        case "actRandomValue":
            let name = node.txt("name", "score")
            if !name.isEmpty {
                var value = rng.range(node.num("min", 0), node.num("max", 10))
                if node.boolean("integer", false) { value = value.rounded() }
                setVariable(name, value)
            }
            enqueueFire(node.id, "out")

        // MARK: Logic

        case "logIf":
            let name = node.txt("name", "score")
            let lhs = variables[name, default: 0]
            let result = compare(lhs, node.txt("cmp", ">="), node.num("value", 1))
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
            let target = clampedInt(node.num("target", 5), 1, 10000)
            let count = (counterCounts[node.id] ?? 0) + 1
            enqueueFire(node.id, "out")
            if count >= target {
                counterCounts[node.id] = 0
                enqueueFire(node.id, "reached")
            } else {
                counterCounts[node.id] = count
            }

        case "logOnce":
            if inputPin == "reset" {
                onceFired.remove(node.id)
            } else if !onceFired.contains(node.id) {
                onceFired.insert(node.id)
                enqueueFire(node.id, "out")
            }

        case "logSequence":
            if inputPin == "reset" {
                sequenceStep[node.id] = 0
            } else {
                let steps = clampedInt(node.num("steps", 2), 2, 4)
                let wrap = node.boolean("wrap", true)
                let step = sequenceStep[node.id] ?? 0
                if wrap {
                    enqueueFire(node.id, "out\((step % steps) + 1)")
                    sequenceStep[node.id] = step + 1
                } else if step < steps {
                    enqueueFire(node.id, "out\(step + 1)")
                    sequenceStep[node.id] = step + 1
                }   // non-wrap: after the last step, inputs drop until reset
            }

        default:
            // Event nodes (and unknown kinds) are sources, not
            // targets: a link landing on one does nothing.
            break
        }
    }

    /// Substitutes every {name} placeholder in an actShowValue
    /// message from the variables table (unknown names read as 0).
    /// Integers print without decimals; anything else gets 2 dp.
    private func substituteVariables(_ message: String) -> String {
        var result = ""
        var rest = message[...]
        while let open = rest.firstIndex(of: "{") {
            result += String(rest[..<open])
            let afterOpen = rest.index(after: open)
            guard let close = rest[afterOpen...].firstIndex(of: "}") else {
                result += String(rest[open...])   // unterminated: keep literally
                rest = rest[rest.endIndex...]
                break
            }
            let name = String(rest[afterOpen..<close])
            let value = variables[name] ?? 0
            if value.isFinite && value == value.rounded() && abs(value) < 9.0e15 {
                result += String(Int(value))
            } else {
                result += String(format: "%.2f", value)
            }
            rest = rest[rest.index(after: close)...]
        }
        result += String(rest)
        return result
    }

    /// Comparison for logIf. Accepts symbol and word forms so the
    /// catalogue's spelling does not matter ("≥", ">=", "Greater
    /// or Equal", "is at least", …).
    private func compare(_ lhs: Double, _ cmp: String, _ rhs: Double) -> Bool {
        let c = cmp.lowercased()
        // F26: "<>" is a common not-equal spelling — check it
        // explicitly BEFORE the ">" test below can claim it.
        if c.contains("<>") { return lhs != rhs }
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

    /// v4.2: the spawn nodes' Offset X/Y/Z params as a vector
    /// (metres, world axes). Zero for nodes saved before the
    /// params existed (catalogue defaults merge at start()).
    private func spawnOffset(of node: LogicNode) -> SIMD3<Float> {
        SIMD3<Float>(Float(node.num("offsetX", 0)),
                     Float(node.num("offsetY", 0)),
                     Float(node.num("offsetZ", 0)))
    }

    /// v4.2: shift a spawn transform's position by the node's
    /// offset (no-op at the zero default).
    private func applySpawnOffset(_ node: LogicNode, to transform: inout Transform3) {
        let offset = spawnOffset(of: node)
        if offset != .zero {
            let p = position(of: transform) + offset
            transform.position = [Double(p.x), Double(p.y), Double(p.z)]
        }
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
        let duration = node.num("duration", 1)
        // Slot rule (§8.0): a MoveTo tween and a Movement behaviour
        // cancel each other — starting the tween vacates the slot.
        movement[id] = nil
        if duration <= 0 {
            tweens[id] = nil
            var t = start
            t.position = [Double(dest.x), Double(dest.y), Double(dest.z)]
            applyTransform(id, t)
            enqueueFire(node.id, "arrived")
        } else {
            // A new MoveTo on the same object replaces the old
            // tween. Slot rule (F10, documented): replacing a tween
            // DROPS the replaced tween's "arrived" — it never fires.
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

    // MARK: - Behaviour slots (advance)

    /// Movement slot: mutates transform.position only, reading the
    /// live tracked transform each application so it composes
    /// with the Facing/Scale slots (recon §8.0).
    private func advanceMovement(dt: Double) {
        guard !movement.isEmpty else { return }
        for id in Array(movement.keys) {
            guard var behaviour = movement[id], let session,
                  let obj = session.project.scene.object(id: id) else {
                movement[id] = nil
                continue
            }
            switch behaviour.kind {

            case .oscillate:
                let period = behaviour.period > 0 ? behaviour.period : 1
                let phase = sin(2 * .pi * (clock - behaviour.startClock) / period)
                var pos = behaviour.anchor
                let offset = Float(behaviour.amplitude * phase)
                switch behaviour.axis {
                case 0: pos.x += offset
                case 2: pos.z += offset
                default: pos.y += offset
                }
                var t = currentTransform(of: obj)
                t.position = [Double(pos.x), Double(pos.y), Double(pos.z)]
                applyTransform(id, t)
                movement[id] = behaviour

            case .follow:
                guard let targetID = behaviour.targetID,
                      let target = session.project.scene.object(id: targetID) else {
                    movement[id] = nil   // target gone: vacate quietly
                    continue
                }
                let targetPos = position(of: currentTransform(of: target))
                let dest = behaviour.goingToTarget ? targetPos : behaviour.home
                let cur = position(of: currentTransform(of: obj))
                let delta = dest - cur
                let dist = simd_length(delta)
                let stepLen = Float(behaviour.speed * dt)
                if dist <= max(stepLen, 0.0001) {
                    var t = currentTransform(of: obj)
                    t.position = [Double(dest.x), Double(dest.y), Double(dest.z)]
                    applyTransform(id, t)
                    switch behaviour.mode {
                    case "Once":
                        enqueueFire(behaviour.nodeID, "arrived")
                        movement[id] = nil
                    case "Loop":
                        enqueueFire(behaviour.nodeID, "arrived")
                        var home = currentTransform(of: obj)
                        home.position = [Double(behaviour.home.x),
                                         Double(behaviour.home.y),
                                         Double(behaviour.home.z)]
                        applyTransform(id, home)
                        movement[id] = behaviour
                    default: // "PingPong": each arrival at either end fires
                        enqueueFire(behaviour.nodeID, "arrived")
                        behaviour.goingToTarget.toggle()
                        movement[id] = behaviour
                    }
                } else {
                    let pos = cur + delta / dist * stepLen
                    var t = currentTransform(of: obj)
                    t.position = [Double(pos.x), Double(pos.y), Double(pos.z)]
                    applyTransform(id, t)
                    movement[id] = behaviour
                }

            case .chase:
                // v4.2: a hidden chaser neither hunts nor catches.
                // The natural Caught wiring — Show Message, then
                // Hide the chaser — must END the hunt: previously
                // the behaviour kept steering invisibly, re-armed
                // whenever the player stepped past the re-arm
                // radius, and caught them again, forever. The slot
                // is kept (a later Show resumes the chase) but
                // disarmed, so a re-show inside the catch radius
                // doesn't instantly re-fire Caught either.
                if !session.runtimeIsObjectVisible(id) {
                    behaviour.caughtArmed = false
                    movement[id] = behaviour
                    break
                }
                var cur = position(of: currentTransform(of: obj))
                var toPlayer = lastPlayerPos - cur
                if !behaviour.fly { toPlayer.y = 0 }
                var dist = simd_length(toPlayer)
                let stopDistance = Float(behaviour.stopDistance)
                // N16: the caught test needs a non-zero radius —
                // at stopDistance 0 it would re-fire every tick
                // while sitting on the player. Movement stopping
                // still uses stopDistance itself.
                let caughtRadius = max(stopDistance, 0.05)
                if dist > stopDistance && dist > 1e-6 {
                    let moveLen = min(Float(behaviour.speed * dt), dist - stopDistance)
                    if moveLen > 0 {
                        cur += toPlayer / dist * moveLen
                        var t = currentTransform(of: obj)
                        t.position = [Double(cur.x), Double(cur.y), Double(cur.z)]
                        applyTransform(id, t)
                        dist -= moveLen
                    }
                }
                if dist <= caughtRadius + 0.0001 {
                    if behaviour.caughtArmed {
                        behaviour.caughtArmed = false
                        enqueueFire(behaviour.nodeID, "caught")
                    }
                } else if dist > caughtRadius * 1.5 {
                    behaviour.caughtArmed = true   // re-arm
                }
                movement[id] = behaviour

            case .wander:
                if clock >= behaviour.idleUntil {
                    if behaviour.wanderTarget == nil {
                        let angle = rng.range(0, 2 * Double.pi)
                        let r = behaviour.radius * sqrt(rng.nextDouble())
                        behaviour.wanderTarget = SIMD3<Float>(
                            behaviour.home.x + Float(cos(angle) * r),
                            behaviour.home.y,
                            behaviour.home.z + Float(sin(angle) * r))
                    }
                    let dest = behaviour.wanderTarget!
                    let cur = position(of: currentTransform(of: obj))
                    let delta = dest - cur
                    let dist = simd_length(delta)
                    let stepLen = Float(behaviour.speed * dt)
                    var t = currentTransform(of: obj)
                    if dist <= max(stepLen, 0.0001) {
                        t.position = [Double(dest.x), Double(dest.y), Double(dest.z)]
                        behaviour.wanderTarget = nil
                        behaviour.idleUntil = clock + behaviour.pause
                    } else {
                        let pos = cur + delta / dist * stepLen
                        t.position = [Double(pos.x), Double(pos.y), Double(pos.z)]
                    }
                    applyTransform(id, t)
                }
                movement[id] = behaviour
            }
        }
    }

    /// Facing slot (LookAt): smooth-turns rotation[1] (yaw only)
    /// toward the target point at turnSpeed °/s, shortest arc.
    private func advanceFacing(dt: Double) {
        guard !facing.isEmpty else { return }
        for id in Array(facing.keys) {
            guard let behaviour = facing[id], let session,
                  let obj = session.project.scene.object(id: id) else {
                facing[id] = nil
                continue
            }
            let targetPoint: SIMD3<Float>
            if behaviour.targetChoice == "Object" {
                guard let targetID = behaviour.targetObjectID,
                      let target = session.project.scene.object(id: targetID) else {
                    facing[id] = nil   // target gone: vacate quietly
                    continue
                }
                targetPoint = position(of: currentTransform(of: target))
            } else {
                targetPoint = lastPlayerPos
            }
            var t = currentTransform(of: obj)
            guard t.rotation.count >= 2 else { continue }
            let cur = position(of: t)
            let dx = Double(targetPoint.x - cur.x)
            let dz = Double(targetPoint.z - cur.z)
            guard dx * dx + dz * dz > 1e-8 else { continue }
            var desired = atan2(-dx, -dz) * 180 / .pi
            desired = desired.truncatingRemainder(dividingBy: 360)
            if desired < 0 { desired += 360 }
            var delta = (desired - t.rotation[1]).truncatingRemainder(dividingBy: 360)
            if delta > 180 { delta -= 360 }
            if delta < -180 { delta += 360 }
            let maxTurn = behaviour.turnSpeed * dt
            let turn = min(maxTurn, max(-maxTurn, delta))
            var yaw = (t.rotation[1] + turn).truncatingRemainder(dividingBy: 360)
            if yaw < 0 { yaw += 360 }
            t.rotation[1] = yaw
            applyTransform(id, t)
        }
    }

    /// Scale slot: linear tween of all three scale components to
    /// the uniform target; completion fires "arrived" once and
    /// vacates the slot.
    private func advanceScaleTweens(dt: Double) {
        guard !scaleTweens.isEmpty else { return }
        for id in Array(scaleTweens.keys) {
            guard var tween = scaleTweens[id], let session,
                  let obj = session.project.scene.object(id: id) else {
                scaleTweens[id] = nil
                continue
            }
            tween.elapsed += dt
            let raw = tween.duration > 0 ? min(1, tween.elapsed / tween.duration) : 1
            var t = currentTransform(of: obj)
            var scale: [Double] = []
            for i in 0..<3 {
                let from = i < tween.fromScale.count ? tween.fromScale[i] : 1
                scale.append(from + (tween.target - from) * raw)
            }
            t.scale = scale
            if raw >= 1 {
                scaleTweens[id] = nil
                applyTransform(id, t)
                enqueueFire(tween.nodeID, "arrived")
            } else {
                scaleTweens[id] = tween
                applyTransform(id, t)
            }
        }
    }

    /// LightFx slot: writes ONLY the emissive fields, at most one
    /// write per object per frame (one dictionary entry each).
    private func applyLightFx() {
        guard !lightFx.isEmpty else { return }
        for id in Array(lightFx.keys) {
            guard let fx = lightFx[id], let session,
                  let obj = session.project.scene.object(id: id) else {
                lightFx[id] = nil
                continue
            }
            var material = currentMaterial(of: obj)
            let elapsed = clock - fx.startClock
            switch fx.kind {
            case .pulse:
                let mid = (fx.minIntensity + fx.maxIntensity) / 2
                let amp = (fx.maxIntensity - fx.minIntensity) / 2
                let period = fx.period > 0 ? fx.period : 1
                material.emissiveOn = true
                material.emissiveIntensity = mid + amp * sin(2 * .pi * elapsed / period)
            case .colorCycle:
                let hue = fx.hue + fx.speed * elapsed
                material.emissiveOn = true
                material.emissiveColor = hsvToRGB(h: hue, s: fx.saturation, v: fx.value)
            }
            trackedMaterials[id] = material
            session.runtimeSetMaterial(id, material)
            soupsDirty = true   // N9: a soup source's material may have changed
        }
    }

    /// Clone lifetime expiry (recon §7.4): the runtime owns
    /// lifetimes; the session owns the geometry.
    private func expireClones() {
        guard !cloneExpiry.isEmpty, let session else { return }
        let expired = cloneExpiry.filter { clock >= $0.value }.map { $0.key }
        for cloneID in expired {
            session.runtimeDespawnClone(cloneID)
            cloneExpiry[cloneID] = nil
            cloneSource[cloneID] = nil
        }
    }

    // MARK: - Collision grid

    private func buildCollisionGrid() {
        guard let session else { return }
        collisionTris = session.renderer.collisionTriangles
        collisionGrid = [:]
        bigCollisionTris = []
        for (index, tri) in collisionTris.enumerated() {
            let loX = min(tri.0.x, tri.1.x, tri.2.x), hiX = max(tri.0.x, tri.1.x, tri.2.x)
            let loZ = min(tri.0.z, tri.1.z, tri.2.z), hiZ = max(tri.0.z, tri.1.z, tri.2.z)
            let minCX = safeCellIndex(loX / cellSize), maxCX = safeCellIndex(hiX / cellSize)
            let minCZ = safeCellIndex(loZ / cellSize), maxCZ = safeCellIndex(hiZ / cellSize)
            // F11: a triangle whose XZ span exceeds the cap (a big
            // floor/terrain sheet) goes into the always-scanned
            // big list instead of the grid — registering it in
            // every overlapped cell would flood the grid, but the
            // old capped registration left it non-collidable
            // outside a central patch. Small triangles keep cell
            // registration exactly as before.
            if maxCX - minCX > 32 || maxCZ - minCZ > 32 {
                bigCollisionTris.append(index)
                continue
            }
            for cx in minCX...maxCX {
                for cz in minCZ...maxCZ {
                    collisionGrid[CellKey(x: cx, z: cz), default: []].append(index)
                }
            }
        }
    }

    /// Highest up-facing (normal.y > 0.3) triangle surface at
    /// (x, z) that lies at or below fromY + 0.05, plus its normal.
    /// Candidates are the query cell's triangles PLUS the
    /// always-scanned big-triangle list (F11).
    private func support(atX x: Float, z: Float, fromY: Float) -> Support? {
        var best: Support?
        func consider(_ index: Int) {
            let tri = collisionTris[index]
            let a = tri.0, b = tri.1, c = tri.2
            let normalRaw = simd_cross(b - a, c - a)
            let len = simd_length(normalRaw)
            if len < 1e-9 { return }
            let n = normalRaw / len
            if n.y <= 0.3 { return }
            // Barycentric test in the XZ projection.
            let denom = (b.z - c.z) * (a.x - c.x) + (c.x - b.x) * (a.z - c.z)
            if abs(denom) < 1e-9 { return }
            let w0 = ((b.z - c.z) * (x - c.x) + (c.x - b.x) * (z - c.z)) / denom
            let w1 = ((c.z - a.z) * (x - c.x) + (a.x - c.x) * (z - c.z)) / denom
            let w2 = 1 - w0 - w1
            if w0 < -1e-5 || w1 < -1e-5 || w2 < -1e-5 { return }
            let h = w0 * a.y + w1 * b.y + w2 * c.y
            if h <= fromY + 0.05 && (best == nil || h > best!.height) {
                best = Support(height: h, normal: n)
            }
        }
        let key = CellKey(x: Int(floor(x / cellSize)), z: Int(floor(z / cellSize)))
        if let indices = collisionGrid[key] {
            for index in indices { consider(index) }
        }
        for index in bigCollisionTris { consider(index) }
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
            // F3: positions/normals get the same count guards the
            // colours/uvs already had — a malformed mesh with short
            // arrays must not trap. Missing position reads as the
            // origin; a missing normal falls back to up.
            let p = mesh.positions.count >= 3 * i + 3
                ? SIMD3<Float>(mesh.positions[3 * i], mesh.positions[3 * i + 1], mesh.positions[3 * i + 2])
                : SIMD3<Float>.zero
            let key = Cell3(x: safeCellIndex(p.x / cell), y: safeCellIndex(p.y / cell), z: safeCellIndex(p.z / cell))
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
            if mesh.normals.count >= 3 * i + 3 {
                sums[idx].nrm += SIMD3<Float>(mesh.normals[3 * i], mesh.normals[3 * i + 1], mesh.normals[3 * i + 2])
            } else {
                sums[idx].nrm += SIMD3<Float>(0, 1, 0)
            }
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
            // F4: skip any triangle whose indices fall outside the
            // vertex map (corrupt/oversized stored mesh) instead of
            // trapping on the subscript.
            let rawA = Int(mesh.indices[i])
            let rawB = Int(mesh.indices[i + 1])
            let rawC = Int(mesh.indices[i + 2])
            i += 3
            guard rawA < vertexMap.count, rawB < vertexMap.count,
                  rawC < vertexMap.count else { continue }
            let a = vertexMap[rawA]
            let b = vertexMap[rawB]
            let c = vertexMap[rawC]
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
        // F13: clamp to the catalogue range BEFORE the early
        // return — a faces of 0/negative from a file must not turn
        // every particle into a full copy of the source mesh.
        let target = min(2000, max(4, targetFaces))
        if source.triangleCount <= target { return source }
        let cacheKey = sourceID.uuidString + "|" + String(target)
        if let cached = decimationCache[cacheKey] { return cached }
        let b = source.bounds
        let diag = simd_length(b.max - b.min)
        guard diag > 1e-6 else { return source }
        let ceiling = Int(1.3 * Double(target))
        var lo: Float = diag / 512
        var hi: Float = diag
        var best: MeshData?
        var coarsest: MeshData?   // F14: fewest-triangle candidate seen
        for _ in 0..<12 {
            let mid = (lo + hi) / 2
            let candidate = clustered(source, cell: mid)
            if coarsest == nil || candidate.triangleCount < coarsest!.triangleCount {
                coarsest = candidate
            }
            if candidate.triangleCount > ceiling {
                lo = mid   // too many triangles: cluster harder
            } else {
                hi = mid
                if best == nil || candidate.triangleCount > best!.triangleCount {
                    best = candidate
                }
            }
        }
        // F14: when no candidate fit under the ceiling, use the
        // COARSEST candidate the search actually produced rather
        // than an unchecked extra clustering at `hi`.
        let result: MeshData
        if let best {
            result = best
        } else if let coarsest {
            result = coarsest
        } else {
            result = clustered(source, cell: hi)
        }
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
        /// The restOffset the particle came to rest with, so a
        /// collision rebuild can verify its support height (F12).
        var restOffset: Float = 0
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
        soupsDirty = true   // emitter set changed
        let duration = node.num("duration", 0)
        emitter.emitUntil = duration > 0 ? clock + duration : .infinity
        let burst = clampedInt(node.num("burst", 0), 0, 500)   // F1
        // N3: the burst obeys the same source-visibility rule as
        // the rate path in updateEmitters — a hidden source does
        // not spray, though the emitter still becomes "emitting".
        let sourceVisible = session?.runtimeIsObjectVisible(emitter.sourceID) ?? true
        if burst > 0 && sourceVisible { _ = spawnParticles(emitter, count: burst) }
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
        // F1/F13: faces clamped to the catalogue range before use.
        // F7 (belt-and-braces): the literal fallbacks below match
        // the catalogue defaults exactly; the start() default
        // merge makes them unreachable for catalogued kinds.
        // F31: rate is clamped to >= 0 so a negative rate from a
        // file cannot grow the accumulator negatively.
        let faces = clampedInt(node.num("faces", 24), 4, 2000)
        let mesh = decimatedMesh(sourceID: sourceID, source: rawMesh, targetFaces: faces)
        let b = mesh.bounds
        return EmitterState(
            nodeID: node.id, sourceID: sourceID, sourceObject: obj, mesh: mesh,
            halfHeight: (b.max.y - b.min.y) / 2, halfWidth: (b.max.x - b.min.x) / 2,
            isPhysics: node.txt("mode", "Physics") != "Simple",
            rate: max(0, node.num("rate", 6)),
            lifetime: Float(node.num("lifetime", 12)),
            size: Float(node.num("size", 0.12)),
            speed: Float(node.num("speed", 0.6)),
            spreadDeg: Float(node.num("spread", 35)),
            direction: node.txt("direction", "Up"),
            gravity: Float(node.num("gravity", 0.35)),
            drag: Float(node.num("drag", 0.6)),
            hitbox: node.txt("hitbox", "Disc"),
            hitboxSize: Float(node.num("hitboxSize", 1)),
            bounce: Float(node.num("bounce", 0)),
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
        soupsDirty = true   // F27: a spawn changed the soups
        return n
    }

    private func updateEmitters(dt: Float) {
        guard !emitters.isEmpty else { return }
        for emitter in Array(emitters.values) {
            // Emission. F20: emission PAUSES while the source is
            // not runtime-visible — the accumulator does not
            // advance, so no backlog builds up behind a hidden
            // source. Particles already alive keep simulating.
            let sourceVisible = session?.runtimeIsObjectVisible(emitter.sourceID) ?? true
            if emitter.emitting && sourceVisible {
                if clock > emitter.emitUntil {
                    emitter.emitting = false
                } else {
                    emitter.accumulator += emitter.rate * Double(dt)
                    // F1: clamp before Int() — a huge accumulator
                    // must not trap the conversion.
                    let wanted = emitter.accumulator.isFinite
                        ? Int(min(100_000, max(0, emitter.accumulator.rounded(.down))))
                        : 0
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
            let aliveBefore = emitter.particles.count
            emitter.particles.removeAll { $0.age >= $0.lifetime }
            if emitter.particles.count != aliveBefore {
                soupsDirty = true   // F27: a death changed the soups
            }
        }
    }

    private func integrate(_ emitter: EmitterState, dt: Float) {
        guard dt > 0 else { return }
        let isDisc = emitter.hitbox == "Disc"
        for i in emitter.particles.indices {
            var p = emitter.particles[i]
            p.age += dt
            if p.age >= p.lifetime { emitter.particles[i] = p; continue }
            // F27 dirty tracking: a particle inside its final-0.4 s
            // shrink window changes the soup every tick, resting
            // or not.
            if p.lifetime - p.age < 0.4 { soupsDirty = true }
            if p.resting { emitter.particles[i] = p; continue }
            soupsDirty = true   // a flying particle moves the soup

            let prevPos = p.pos
            let prevY = p.pos.y
            // Gravity (discs fall slower — they catch the air),
            // scaled by the global actSetGravity multiplier.
            var g: Float = 9.8 * emitter.gravity * Float(particleGravityScale)
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
            // crosses the surface still finds it. F21: when the
            // step's XZ length exceeds a grid cell, the step
            // midpoint is probed too (union of candidates — the
            // higher support wins) so a fast particle cannot jump
            // clean over a narrow ledge in one step.
            var foundSupport: Support? = nil
            if p.vel.y <= 0 {
                foundSupport = support(atX: p.pos.x, z: p.pos.z, fromY: prevY)
                let stepXZ = simd_distance(SIMD2<Float>(prevPos.x, prevPos.z),
                                           SIMD2<Float>(p.pos.x, p.pos.z))
                if stepXZ > cellSize {
                    let midX = (prevPos.x + p.pos.x) / 2
                    let midZ = (prevPos.z + p.pos.z) / 2
                    if let midSup = support(atX: midX, z: midZ, fromY: prevY),
                       midSup.height > (foundSupport?.height ?? -Float.greatestFiniteMagnitude) {
                        foundSupport = midSup
                    }
                }
            }
            if p.vel.y <= 0, let sup = foundSupport {
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
        p.restOffset = restOffset   // F12: remembered for support re-checks
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
    /// F27: when nothing changed since the last send (every
    /// particle resting, none shrinking, no spawn/death/un-rest)
    /// the rebuild is skipped entirely — the renderer keeps the
    /// soups it already holds.
    private func buildAndSendSoups() {
        guard running else { return }
        if emitters.isEmpty {
            if soupsSent { session?.runtimeSetParticleSoups([]); soupsSent = false }
            soupsDirty = false
            return
        }
        let anyParticles = emitters.values.contains { !$0.particles.isEmpty }
        if anyParticles && !soupsDirty { return }   // nothing moved: skip the resend
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
        soupsDirty = false   // whatever changed has now been sent
    }
}
