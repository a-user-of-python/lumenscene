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
//   freeze   — accumulate hands-off: once A >= 0.002 (before
//              the first sheet rebuild), the REAL idle freeze
//              engages via the session's own bookkeeping (the
//              driver shortens idleFreezeSeconds — the same
//              preference the More-menu picker writes — and
//              withholds input; frameTick's engage check does
//              the rest), ~30 frozen frames pump, the real wake
//              path runs (noteUserInput -> wakeFromIdleFreeze),
//              then accumulation continues through the first
//              sheet rebuild + sim ticks.
//   freeze2  — like freeze, but armed the moment a rebuild is
//              STAGED (async AS build in flight), so the engage
//              races the frame-boundary swap: the session's
//              protected-operation check defers the engage to
//              the first clear frame after the swap.
//   deeprest — v4.10.4: bare 8x8 plane + one 1x1x1 box, deep
//              snow pre-seeded (maxDepth 2.0, A = 1.80, level
//              0.9). No drag: once the deep field is established
//              (first sheet swap + sim initialised) emit
//              REPRO-METRIC "pre", pump ~30 s more (snowfall
//              still on), emit "rest-30s", then REPRO-DONE.
//   deepdrag — same scene/pre-seed as deeprest. After "pre",
//              drag the box +X ~0.011 m/pump for ~3.3 m total,
//              emit "mid-drag" halfway and "post-drag" at the
//              end, pump 20 s more, emit "settled-20s", then
//              REPRO-DONE. Metrics are computed from the
//              renderer's LS-REPRO sim snapshot (see
//              Renderer.reproSimSnapshot).
//   deepspawn — v4.10.4: same pre-seeded deep scene as
//              deeprest/deepdrag but plane ONLY at start.
//              Once the field is established, a fresh 1x1x1
//              box is placed through the REAL Add path
//              (EditorSession.addPrimitive — exposure 0,
//              renderer map sync, rebuild) at (0, 0.5, 0):
//              makeProject sets the camera target to
//              (0, 1, 0), which is where addPrimitive drops
//              new objects. Metrics at placement
//              ("spawn-0", first metric after the add
//              rebuild) and at +5/+15/+30 s of continued
//              snowfall carry a spawn section: box-top snow
//              max/mean, ring (plane within 0.4 m outside the
//              footprint) mean/min/max, and field (plane
//              > 1 m from the footprint) mean/max.

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
        case "deeprest", "deepdrag":
            // v4.10.4 deep-snow scene: a BARE plane + one box,
            // nothing else from the demo. Weather is pre-seeded
            // BEFORE EditorSession init, so the session seeds
            // buildupSnow from snowAmount (1.80 m) and every
            // object's exposure defaults to the ambient amount
            // (absent snowExposure = fully caught up): the
            // field starts deep, exactly as authored.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            var box = SceneObject.primitive("box", name: "Repro drag box")
            box.params = ["w": 1, "h": 1, "d": 1]
            box.transform.position = [-2.5, 0.5, 0]
            p.scene.objects = [plane, box]
            p.scene.weather.maxDepth = 2.0
            p.scene.weather.snowAmount = 1.80
            p.scene.weather.snow = 0.9
        case "deepspawn":
            // Same deep pre-seed, but NO box at start — it is
            // spawned mid-run via addPrimitive (see the
            // variant list). Camera target (0, 1, 0) makes
            // addPrimitive land the box at (0, 0.5, 0): it
            // places at target X/Z with the box's resting Y.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            p.scene.objects = [plane]
            p.scene.weather.maxDepth = 2.0
            p.scene.weather.snowAmount = 1.80
            p.scene.weather.snow = 0.9
            p.scene.camera.target = [0, 1, 0]
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
    private var sawSheetAfterWeatherOn = false
    private var sawSwapAfterWeatherOn = false
    private var sawTickTag = false
    private var firstStepCrossed = false
    private var weatherOn = false
    private var crossedA = false
    private var simTickFrames = 0
    private var framesAfterSwap = 0
    private var finished = false

    // Freeze-interleave state (variants freeze / freeze2).
    private enum Phase { case accumulate, armingFreeze, frozen, resume }
    private var phase: Phase = .accumulate
    private var frozenPumps = 0
    private var armingPumps = 0
    private var freezeDone = false
    private var stagedSeen = false

    private var isFreezeVariant: Bool {
        ReproDriver.variant == "freeze" || ReproDriver.variant == "freeze2"
    }

    // Deep-snow state (variants deeprest / deepdrag /
    // deepspawn, v4.10.4).
    private enum DeepPhase { case establishing, resting, dragging, settling, spawnWait }
    private var deepPhase: DeepPhase = .establishing
    private var deepPhaseSince: CFTimeInterval = 0
    private var dragStartX: Double = -2.5
    private var dragX: Double = -2.5
    private var dragMidEmitted = false
    private var spawnCheckpointIdx = 0
    private let spawnCheckpoints: [(delay: CFTimeInterval, label: String)] =
        [(5, "spawn+5s"), (15, "spawn+15s"), (30, "spawn+30s")]

    private var isDeepVariant: Bool {
        ReproDriver.variant == "deeprest" || ReproDriver.variant == "deepdrag"
            || ReproDriver.variant == "deepspawn"
    }

    /// LS-REPRO metrics: computed from the renderer's sim
    /// snapshot. Surface height = support + snow at each
    /// lattice vertex. Emits exactly one REPRO-METRIC line.
    private func emitMetric(_ label: String) {
        guard let renderer else { return }
        let s = renderer.reproSimSnapshot()
        let n = s.vx * s.vz
        func clean(_ v: Float) -> Float { v.isFinite ? v : 0 }
        var maxH: Float = 0
        var violCount = 0
        var violMax: Float = 0
        var maxSlopeDeg: Float = 0
        var spikeCount = 0
        var spikeMax: Float = 0
        var vol: Float = 0
        if s.initialised, n > 0, s.snow.count >= n,
           s.support.count >= n, s.ceiling.count >= n, s.cs > 0 {
            var surf = [Float](repeating: 0, count: n)
            for i in 0..<n {
                let snow = clean(s.snow[i])
                surf[i] = clean(s.support[i]) + snow
                vol += snow
                if surf[i] > maxH { maxH = surf[i] }
                let excess = snow - clean(s.ceiling[i])
                if excess > 0.05 {
                    violCount += 1
                    if excess > violMax { violMax = excess }
                }
            }
            vol *= s.cs * s.cs
            // Max slope between adjacent surface vertices.
            for j in 0..<s.vz {
                for i in 0..<s.vx {
                    let a = j * s.vx + i
                    if i + 1 < s.vx {
                        let d = abs(surf[a + 1] - surf[a]) / s.cs
                        let deg = Float(atan(Double(d))) * 180 / Float.pi
                        if deg > maxSlopeDeg { maxSlopeDeg = deg }
                    }
                    if j + 1 < s.vz {
                        let d = abs(surf[a + s.vx] - surf[a]) / s.cs
                        let deg = Float(atan(Double(d))) * 180 / Float.pi
                        if deg > maxSlopeDeg { maxSlopeDeg = deg }
                    }
                }
            }
            // Spike: surface minus the median of its 3x3
            // neighbourhood (all 9 values, centre included),
            // interior vertices only.
            if s.vx >= 3, s.vz >= 3 {
                for j in 1..<(s.vz - 1) {
                    for i in 1..<(s.vx - 1) {
                        var vals: [Float] = []
                        vals.reserveCapacity(9)
                        for dj in -1...1 {
                            for di in -1...1 {
                                vals.append(surf[(j + dj) * s.vx + (i + di)])
                            }
                        }
                        vals.sort()
                        let diff = surf[j * s.vx + i] - vals[4]
                        if diff > spikeMax { spikeMax = diff }
                        if diff > 0.30 { spikeCount += 1 }
                    }
                }
            }
        }
        var line = "REPRO-METRIC \(label)"
            + String(format: " maxH=%.4f violCount=%d violMax=%.4f maxSlope=%.2f spikeCount=%d spikeMax=%.4f vol=%.4f maxPlowPos=%.4f minPlowNeg=%.4f",
                     maxH, violCount, violMax, maxSlopeDeg,
                     spikeCount, spikeMax, vol,
                     s.plowMaxPos, s.plowMinNeg)
        // Spawn section (deepspawn only): box-top / ring /
        // field snow stats from the same snapshot. Box-top =
        // vertices with support Y ~ 1.0 inside the box
        // footprint; ring = plane-supported vertices (support
        // Y ~ 0) whose XZ distance to the footprint rectangle
        // is in (0, 0.4 m]; field = plane-supported vertices
        // more than 1 m from the footprint. All counts are 0
        // and all stats 0 before the box exists (pre-spawn).
        if ReproDriver.variant == "deepspawn" {
            var topMax: Float = 0, topSum: Float = 0, topCount = 0
            var ringSum: Float = 0, ringMin: Float = 0, ringMax: Float = 0, ringCount = 0
            var fieldSum: Float = 0, fieldMax: Float = 0, fieldCount = 0
            if s.initialised, n > 0, s.snow.count >= n, s.support.count >= n,
               let box = session?.project.scene.objects.first(where: { $0.primitiveKind == "box" }) {
                let cx = Float(box.transform.position[0])
                let cz = Float(box.transform.position[2])
                for j in 0..<s.vz {
                    for i in 0..<s.vx {
                        let idx = j * s.vx + i
                        let x = s.ox + Float(i) * s.cs
                        let z = s.oz + Float(j) * s.cs
                        let sup = clean(s.support[idx])
                        let snow = clean(s.snow[idx])
                        let dx = max(abs(x - cx) - 0.5, 0)
                        let dz = max(abs(z - cz) - 0.5, 0)
                        let rectDist = sqrtf(dx * dx + dz * dz)
                        let inside = abs(x - cx) <= 0.5 && abs(z - cz) <= 0.5
                        if inside, abs(sup - 1.0) < 0.05 {
                            topCount += 1
                            topSum += snow
                            if snow > topMax { topMax = snow }
                        } else if abs(sup) < 0.05 {
                            if rectDist > 0.001, rectDist <= 0.4 {
                                if ringCount == 0 || snow < ringMin { ringMin = snow }
                                if snow > ringMax { ringMax = snow }
                                ringCount += 1
                                ringSum += snow
                            } else if rectDist > 1.0 {
                                fieldCount += 1
                                fieldSum += snow
                                if snow > fieldMax { fieldMax = snow }
                            }
                        }
                    }
                }
            }
            let topMean: Float = topCount > 0 ? topSum / Float(topCount) : 0
            let ringMean: Float = ringCount > 0 ? ringSum / Float(ringCount) : 0
            let fieldMean: Float = fieldCount > 0 ? fieldSum / Float(fieldCount) : 0
            let ratio: Float = fieldMean > 0 ? topMean / fieldMean : 0
            line += String(format: " spawnTopMax=%.4f spawnTopMean=%.4f spawnTopCount=%d ringMean=%.4f ringMin=%.4f ringMax=%.4f ringCount=%d fieldMean=%.4f fieldMax=%.4f fieldCount=%d topFieldRatio=%.4f A=%.4f",
                           topMax, topMean, topCount,
                           ringMean, ringMin, ringMax, ringCount,
                           fieldMean, fieldMax, fieldCount,
                           ratio, session?.liveSnowAmount ?? 0)
        }
        print(line)
        fflush(stdout)
        NSLog("%@", line)
    }

    private func finishDeep() {
        finished = true
        print("REPRO-DONE")
        fflush(stdout)
        NSLog("REPRO-DONE")
        exit(0)
    }

    /// Deep-variant phase machine, called once per pump AFTER
    /// the latch updates (so sawTickTag / simTickFrames reflect
    /// this pump). Establishment = a staged static build has
    /// swapped in (reproSimSnapshot().swapped, i.e.
    /// staticEverSwapped) AND the sim is initialised
    /// (reproSimSnapshot().initialised — only a sheet build's
    /// simSync initialises it, so this IS the first-sheet
    /// evidence) AND the sim has ticked 5 times.
    /// NOTE (run 41336002939): the generic settle's sampled
    /// `rebuild:swap` tag latch can NEVER fire for these
    /// variants — with A pre-seeded the sim initialises at
    /// session init, so every draw runs swapStaticIfReady
    /// (tag rebuild:swap) and then tickWeatherSim (tag
    /// tick:entry) inside the SAME draw, and the tick tag
    /// always wins the post-frameTick sample. deeprest and
    /// deepdrag both sat in .establishing for the full 200 s
    /// watchdog with simTicks > 5000 and A = 2.0 and no swap
    /// tag ever sampled; the direct checks above are the fix
    /// (driver-only — no app code involved).
    private func handleDeep() {
        guard let session else { return }
        let now = CACurrentMediaTime()
        switch deepPhase {
        case .establishing:
            // Swap witness is the snapshot's `swapped`
            // (staticEverSwapped), NOT hasGeometry: in the
            // simulator ray tracing is unsupported, the staged
            // build's AS is nil, swapStatic still runs and
            // sets the flag — but accel stays nil forever
            // (run 41338778095 sat in .establishing on
            // hasGeometry for the full watchdog, simTicks
            // 5489, exactly as the tag latch had before it).
            guard let renderer else { return }
            let snap = renderer.reproSimSnapshot()
            guard snap.initialised, snap.swapped,
                  sawTickTag, simTickFrames >= 5 else { return }
            if ReproDriver.variant == "deepspawn" {
                emitMetric("pre-spawn")
                // The REAL editor Add path: addPrimitive
                // registers the new object at exposure 0,
                // syncs the renderer's exposure map BEFORE
                // its rebuild, selects it and rebuilds —
                // exactly what the Add button does. Placement
                // is the camera target's X/Z (set to (0, 1, 0)
                // in makeProject) with the box's resting Y,
                // i.e. (0, 0.5, 0); the marker records where
                // it actually landed.
                session.addPrimitive("box")
                if let box = session.project.scene.objects.first(where: { $0.primitiveKind == "box" }) {
                    let p = box.transform.position
                    ReproDriver.marker("spawn-placed pos=[\(p[0]), \(p[1]), \(p[2])] A=\(session.liveSnowAmount)")
                } else {
                    ReproDriver.marker("spawn-placed MISSING-BOX A=\(session.liveSnowAmount)")
                }
                // First metric after the add rebuild (the
                // rebuild inside addPrimitive is synchronous:
                // the sheet pass + sim sync have already run).
                emitMetric("spawn-0")
                spawnCheckpointIdx = 0
                deepPhaseSince = now
                deepPhase = .spawnWait
                return
            }
            emitMetric("pre")
            deepPhaseSince = now
            if ReproDriver.variant == "deepdrag" {
                if let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) {
                    dragStartX = session.project.scene.objects[idx].transform.position[0]
                    dragX = dragStartX
                }
                dragMidEmitted = false
                deepPhase = .dragging
                ReproDriver.marker("deep-drag-start x=\(dragStartX)")
            } else {
                deepPhase = .resting
                ReproDriver.marker("deep-rest-start")
            }
        case .resting:
            if now - deepPhaseSince >= 30 {
                emitMetric("rest-30s")
                finishDeep()
            }
        case .dragging:
            // Scripted drag: exactly the two effects
            // EditorSession.dragObjectTo has — it writes
            // position[0]/[2] (here only [0]: the motion is
            // pure +X, so [2] is unchanged) and then calls
            // scheduleRebuild(geometry: true). The ray math
            // that derives the position from a pointer NDC is
            // the only part skipped.
            guard let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) else {
                deepPhase = .settling
                deepPhaseSince = now
                return
            }
            dragX = min(dragX + 0.011, dragStartX + 3.3)
            session.project.scene.objects[idx].transform.position[0] = dragX
            session.scheduleRebuild(geometry: true)
            if !dragMidEmitted, dragX >= dragStartX + 1.65 {
                dragMidEmitted = true
                emitMetric("mid-drag")
            }
            if dragX >= dragStartX + 3.3 - 1e-9 {
                emitMetric("post-drag")
                deepPhase = .settling
                deepPhaseSince = now
                ReproDriver.marker("deep-drag-end x=\(dragX)")
            }
        case .settling:
            if now - deepPhaseSince >= 20 {
                emitMetric("settled-20s")
                finishDeep()
            }
        case .spawnWait:
            guard spawnCheckpointIdx < spawnCheckpoints.count else { finishDeep(); return }
            let cp = spawnCheckpoints[spawnCheckpointIdx]
            if now - deepPhaseSince >= cp.delay {
                emitMetric(cp.label)
                spawnCheckpointIdx += 1
                if spawnCheckpointIdx >= spawnCheckpoints.count {
                    finishDeep()
                }
            }
        }
    }

    /// Freeze trigger, evaluated in the accumulate phase.
    /// freeze: accumulation under way but no sheet rebuild yet
    /// (the hands-off watcher crossing the 60 s boundary
    /// mid-accumulation, compressed). freeze2: a rebuild has
    /// just been staged — freeze tries to engage at the swap
    /// boundary.
    private func freezeTriggerMet() -> Bool {
        guard isFreezeVariant, !freezeDone else { return false }
        if ReproDriver.variant == "freeze" {
            // A in [0.002, 0.006): accumulation under way, the
            // first quantised step (0.00625 at the default
            // 0.3 m depth) not yet crossed — no sheet rebuild
            // can have fired.
            let a = session?.liveSnowAmount ?? 0
            return crossedA && a >= 0.002 && a < 0.006
        }
        return stagedSeen
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
        // calls session.setWeatherMode — mode 2 is Snow.
        ReproDriver.marker("weather-on-before")
        session.setWeatherMode(2)
        weatherOn = true
        ReproDriver.marker("weather-on-after mode=\(session.project.scene.weather.mode)")

        startedAt = CACurrentMediaTime()
        pump()
    }

    private func pump() {
        guard !finished, let session, let renderer, let view else { return }

        // --- Freeze-variant phase transitions (before input) ---
        if phase == .accumulate, freezeTriggerMet() {
            // Arm the REAL idle freeze through the session's own
            // bookkeeping: shorten the idle delay (the same
            // renderer preference the More-menu picker writes)
            // and stop reporting input. frameTick's own engage
            // check (idle time + protected-operation gate) does
            // the rest — no stub, no state surgery.
            session.idleFreezeSeconds = 0.25
            phase = .armingFreeze
            armingPumps = 0
            ReproDriver.marker("freeze-arm A=\(session.liveSnowAmount) step=\(renderer.weatherStep)")
        }
        if phase == .frozen {
            frozenPumps += 1
            if frozenPumps >= 30 {
                // The real wake path: every input funnel in the
                // app goes through noteUserInput, which wakes a
                // frozen session via wakeFromIdleFreeze.
                ReproDriver.marker("wake-before frozen=\(session.isIdleFrozen)")
                session.noteUserInput()
                session.idleFreezeSeconds = 60
                freezeDone = true
                phase = .resume
                ReproDriver.marker("wake-after frozen=\(session.isIdleFrozen)")
            }
        }

        // Synthetic presence: a live user driving frames keeps
        // the idle freeze from engaging; the repro does the
        // same through the real input funnel — EXCEPT while
        // arming/frozen, where withholding input is the point.
        if phase != .armingFreeze && phase != .frozen {
            session.noteUserInput()
        }
        session.frameTick(view: view)
        pumpCount += 1

        if phase == .armingFreeze {
            armingPumps += 1
            if session.isIdleFrozen {
                phase = .frozen
                frozenPumps = 0
                ReproDriver.marker("freeze-engaged step=\(renderer.weatherStep)")
            } else if armingPumps > 240 {
                // The engage never landed (a protected operation
                // never cleared?) — a named driver verdict, not
                // a silent stall.
                finished = true
                print("REPRO-TIMEOUT reason=freeze-arm-stalled pumps=\(pumpCount) step=\(renderer.weatherStep) buildInFlight=\(renderer.staticBuildInFlight)")
                fflush(stdout)
                NSLog("REPRO-TIMEOUT freeze-arm-stalled")
                exit(2)
            }
        }
        if weatherOn, renderer.staticBuildInFlight, !stagedSeen {
            stagedSeen = true
            ReproDriver.marker("as-staged step=\(renderer.weatherStep)")
        }

        // Observe the renderer's own step trail (the same tag
        // the app persists for post-crash diagnosis).
        let tag = renderer.weatherStep
        if tag != lastStepTag {
            lastStepTag = tag
            ReproDriver.marker("step \(tag)")
            if weatherOn, tag.hasPrefix("sheet:") { sawSheetAfterWeatherOn = true }
        }

        let a = session.liveSnowAmount
        if weatherOn, !crossedA, a > 0 {
            crossedA = true
            ReproDriver.marker("accumulation-positive A=\(a)")
        }
        if weatherOn, !firstStepCrossed, a >= 0.006 {
            firstStepCrossed = true
            ReproDriver.marker("first-step-crossed A=\(a)")
        }
        // Unconditional latches — NOT change-gated: the step
        // tag can round-trip entry -> sheet -> stageAS -> swap
        // inside ONE frameTick and return to exactly the value
        // the previous poll saw, so a change-gated latch misses
        // every swap after the first. A swap only counts as
        // the SHEET rebuild's swap once A has crossed the
        // first quantised step (earlier swaps are the
        // session-init builds); the tick: latch proves the sim
        // initialised (tickWeatherSim tags only past its
        // simInitialised guard).
        if weatherOn, tag == "rebuild:swap", a >= 0.006 {
            sawSwapAfterWeatherOn = true
        }
        if weatherOn, tag.hasPrefix("tick:") { sawTickTag = true }
        if sawSwapAfterWeatherOn { framesAfterSwap += 1 }

        if renderer.simStepsLastFrame > 0 {
            simTickFrames += 1
            if simTickFrames == 1 { ReproDriver.marker("sim-tick-first") }
            if simTickFrames == 5 { ReproDriver.marker("sim-tick-fifth") }
        }

        if pumpCount % 15 == 0 {
            ReproDriver.marker("pump \(pumpCount) A=\(a) step=\(tag) simTicks=\(simTickFrames)")
        }

        // Settle = every link in the accumulation-start chain
        // has demonstrably run: A crossed 0 and then the first
        // quantised step, the sheet rebuild's staged AS swapped
        // in (a swap observed at A >= step), the sim initialised
        // (a tick: tag was observed — tickWeatherSim tags only
        // past its simInitialised guard) and ticked 5 times (or,
        // with the simtick toggle, 90 post-swap frames stand in
        // for tick evidence). Freeze variants additionally
        // require the full engage -> frozen frames -> wake
        // cycle to have run.
        // Deep variants run their own phase machine and finish
        // from it (REPRO-DONE there); the generic settle below
        // would otherwise end them at establishment.
        if isDeepVariant {
            handleDeep()
            if finished { return }
        }

        let simSatisfied = ReproToggles.simTick ? framesAfterSwap >= 90
                                                : (sawTickTag && simTickFrames >= 5)
        let freezeSatisfied = !isFreezeVariant || freezeDone
        if !isDeepVariant, crossedA, firstStepCrossed, sawSwapAfterWeatherOn, simSatisfied, freezeSatisfied {
            finished = true
            ReproDriver.marker("settle A=\(a) pumps=\(pumpCount) simTicks=\(simTickFrames)")
            print("REPRO-DONE")
            fflush(stdout)
            NSLog("REPRO-DONE")
            exit(0)
        }

        // Watchdog: a harness outcome, never a crash signal.
        // Deep variants are long by design (30 s rest / ~3.3 m
        // drag + 20 s settle), so they get a 200 s watchdog
        // (the workflow's per-run timeout is 240 s).
        let watchdogLimit: CFTimeInterval = isDeepVariant ? 200 : 100
        if CACurrentMediaTime() - startedAt > watchdogLimit {
            finished = true
            print("REPRO-TIMEOUT pumps=\(pumpCount) A=\(a) step=\(tag) crossedA=\(crossedA) sheet=\(sawSheetAfterWeatherOn) swap=\(sawSwapAfterWeatherOn) simTicks=\(simTickFrames)")
            fflush(stdout)
            NSLog("REPRO-TIMEOUT")
            exit(2)
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(33)) { [weak self] in
            MainActor.assumeIsolated { self?.pump() }
        }
    }
}
