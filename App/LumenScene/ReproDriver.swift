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
//              rebuild) and at +5/+15/+30/+60/+90 s of
//              continued snowfall carry a spawn section:
//              box-top snow max/mean + box-top ceiling mean,
//              ring (plane within 0.4 m outside the
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
        case "pile", "settle", "drag2", "spawn2", "roof", "longrun":
            // v4.11 physics-audit scenes: a bare 16x16 plane
            // (plus a variant object), temperature -8 C, and a
            // whisper of pre-seeded snow (0.05 m) so the sheet
            // builds and the sim initialises at session init.
            // The real test fields are INJECTED into the sim
            // by handlePhysics once establishment is proven —
            // the in-app counterparts of the mirror audit's
            // PILE / SETTLE / DRAG / SPAWN / ROOF / LONGRUN.
            var plane = SceneObject.primitive("plane", name: "Repro physics plane")
            plane.params = ["w": 16, "d": 16]
            plane.transform.position = [0, 0, 0]
            p.scene.objects = [plane]
            if variant == "settle" || variant == "longrun" {
                var box = SceneObject.primitive("box", name: "Repro settle box")
                box.params = ["w": 1, "h": 1, "d": 1]
                box.transform.position = [0, 0.5, 0]
                p.scene.objects.append(box)
            }
            if variant == "drag2" {
                var box = SceneObject.primitive("box", name: "Repro drag box")
                box.params = ["w": 1, "h": 1, "d": 1]
                box.transform.position = [-2.5, 0.5, 0]
                p.scene.objects.append(box)
            }
            if variant == "roof" {
                // A box rolled 45 degrees: its two upper faces
                // are 45-degree slopes meeting at a ridge.
                var roof = SceneObject.primitive("box", name: "Repro roof block")
                roof.params = ["w": 1, "h": 1, "d": 1]
                roof.transform.scale = [3, 1, 2]
                roof.transform.rotation = [0, 0, 45]
                roof.transform.position = [0, 2.2, 0]
                p.scene.objects.append(roof)
            }
            if variant == "spawn2" {
                p.scene.camera.target = [0, 1, 0]
            }
            p.scene.weather.maxDepth = 2.0
            // v4.11 fix round: physics scenes run at the full
            // 2.0 m amount so the deposition GATE sits at 2.0 m —
            // at 0.05 the gate-compaction term (depth above the
            // gate densifies) ate the injected pile from the top
            // (flank 24 deg instead of repose in run 41368850510).
            // Injections overwrite the whole field, so the deep
            // initial condition this loads is immaterial.
            p.scene.weather.snowAmount = 2.0
            p.scene.weather.snow = 0.9
            p.scene.weather.temperature = -8
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
    private var dragStepCount = 0
    private var spawnCheckpointIdx = 0
    private let spawnCheckpoints: [(delay: CFTimeInterval, label: String)] =
        [(5, "spawn+5s"), (15, "spawn+15s"), (30, "spawn+30s"),
         (60, "spawn+60s"), (90, "spawn+90s")]

    private var isDeepVariant: Bool {
        ReproDriver.variant == "deeprest" || ReproDriver.variant == "deepdrag"
            || ReproDriver.variant == "deepspawn"
    }

    // Physics-audit state (v4.11 variants pile / settle /
    // drag2 / spawn2 / roof / longrun).
    private enum PhysPhase { case establishing, running, dragging, settling,
                             spawnSnow, longrunA, longrunB }
    private var physPhase: PhysPhase = .establishing
    private var physPhaseSince: CFTimeInterval = 0
    private var injectedMass: Float = 0

    private var isPhysicsVariant: Bool {
        ["pile", "settle", "drag2", "spawn2", "roof", "longrun"]
            .contains(ReproDriver.variant)
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
        var smoothMaxDeg: Float = 0
        var smoothRMSDeg: Float = 0
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
            // Same-support smoothness (v4.10.4 round 2): the
            // honest broad-field roughness number — surface
            // slope over adjacent pairs whose SUPPORT heights
            // agree (no object edge / skirt step between
            // them) and which both carry snow. maxSlope above
            // conflates support steps with snow roughness.
            var smoothPairs = 0
            var smoothSqSum: Double = 0
            for j in 0..<s.vz {
                for i in 0..<s.vx {
                    let a = j * s.vx + i
                    func pair(_ b: Int) {
                        if abs(clean(s.support[a]) - clean(s.support[b])) > 0.05 { return }
                        if clean(s.snow[a]) <= 0 || clean(s.snow[b]) <= 0 { return }
                        let d = abs(surf[b] - surf[a]) / s.cs
                        let deg = Float(atan(Double(d))) * 180 / Float.pi
                        if deg > smoothMaxDeg { smoothMaxDeg = deg }
                        smoothSqSum += Double(deg) * Double(deg)
                        smoothPairs += 1
                    }
                    if i + 1 < s.vx { pair(a + 1) }
                    if j + 1 < s.vz { pair(a + s.vx) }
                }
            }
            if smoothPairs > 0 {
                smoothRMSDeg = Float(sqrt(smoothSqSum / Double(smoothPairs)))
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
            + String(format: " maxH=%.4f violCount=%d violMax=%.4f maxSlope=%.2f smoothMax=%.2f smoothRMS=%.2f spikeCount=%d spikeMax=%.4f vol=%.4f maxPlowPos=%.4f minPlowNeg=%.4f",
                     maxH, violCount, violMax, maxSlopeDeg,
                     smoothMaxDeg, smoothRMSDeg,
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
            var topCeilSum: Float = 0
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
                            if idx < s.ceiling.count { topCeilSum += clean(s.ceiling[idx]) }
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
            let topCeilMean: Float = topCount > 0 ? topCeilSum / Float(topCount) : 0
            line += String(format: " spawnTopMax=%.4f spawnTopMean=%.4f spawnTopCeilMean=%.4f spawnTopCount=%d ringMean=%.4f ringMin=%.4f ringMax=%.4f ringCount=%d fieldMean=%.4f fieldMax=%.4f fieldCount=%d topFieldRatio=%.4f A=%.4f",
                           topMax, topMean, topCeilMean, topCount,
                           ringMean, ringMin, ringMax, ringCount,
                           fieldMean, fieldMax, fieldCount,
                           ratio, session?.liveSnowAmount ?? 0)
        }
        // Physics section (v4.11 variants): total snow MASS,
        // the TRUE max slope (8-neighbour; an edge counts only
        // when its higher endpoint carries snow — a step up to
        // a bare support is an exposed object wall), and the
        // repose-support spike residual (a column is a spike
        // only if it rises above what ANY same-support
        // neighbour could support at repose + 2 deg).
        if isPhysicsVariant, s.initialised, n > 0,
           s.mass.count >= n, s.snow.count >= n, s.support.count >= n, s.cs > 0 {
            var massTotal: Float = 0
            for i in 0..<n { massTotal += clean(s.mass[i]) }
            massTotal *= s.cs * s.cs
            var slopeTrue: Float = 0
            var spikeR: Float = 0
            let tanLim = Float(tan(36.0 * Double.pi / 180))
            for j in 0..<s.vz {
                for i in 0..<s.vx {
                    let a = j * s.vx + i
                    let surfA = clean(s.support[a]) + clean(s.snow[a])
                    for (di, dj) in [(1, 0), (0, 1), (1, 1), (1, -1)] {
                        let ni = i + di, nj = j + dj
                        if ni < 0 || nj < 0 || ni >= s.vx || nj >= s.vz { continue }
                        let b = nj * s.vx + ni
                        let surfB = clean(s.support[b]) + clean(s.snow[b])
                        let dist = s.cs * ((di != 0 && dj != 0) ? Float(1.41421356) : 1)
                        let hiSnow = surfA >= surfB ? clean(s.snow[a]) : clean(s.snow[b])
                        if hiSnow > 0.001 {
                            let deg = Float(atan(Double(abs(surfA - surfB) / dist)))
                                * 180 / Float.pi
                            if deg > slopeTrue { slopeTrue = deg }
                        }
                    }
                    if i > 0, j > 0, i + 1 < s.vx, j + 1 < s.vz {
                        var best: Float = .infinity
                        for dj in -1...1 {
                            for di in -1...1 where !(di == 0 && dj == 0) {
                                let b = (j + dj) * s.vx + (i + di)
                                if abs(clean(s.support[b]) - clean(s.support[a])) > 0.05 { continue }
                                let dist = s.cs * Float(hypot(Double(di), Double(dj)))
                                let cand = clean(s.support[b]) + clean(s.snow[b]) + tanLim * dist
                                if cand < best { best = cand }
                            }
                        }
                        if best.isFinite {
                            let r = surfA - best
                            if r > spikeR { spikeR = r }
                        }
                    }
                }
            }
            line += String(format: " mass=%.5f slopeTrue=%.2f spikeR=%.4f",
                           massTotal, slopeTrue, spikeR)
            line += " remaps=\(renderer.reproRemapCount)"
            switch ReproDriver.variant {
            case "pile":
                // Cone flank: median local edge slope along
                // the four rays from the cone centre.
                var loc: [Float] = []
                let ci = Int(((0 - s.ox) / s.cs).rounded())
                let cj = Int(((0 - s.oz) / s.cs).rounded())
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    var prev = clean(s.support[cj * s.vx + ci]) + clean(s.snow[cj * s.vx + ci])
                    for r in 1...8 {
                        let ii = ci + di * r, jj = cj + dj * r
                        if ii < 0 || jj < 0 || ii >= s.vx || jj >= s.vz { break }
                        let h = clean(s.support[jj * s.vx + ii]) + clean(s.snow[jj * s.vx + ii])
                        let drop = prev - h
                        if drop > 0.02 {
                            loc.append(Float(atan(Double(drop / s.cs))) * 180 / Float.pi)
                        }
                        prev = h
                    }
                }
                loc.sort()
                let flank: Float = loc.isEmpty ? 0 : loc[loc.count / 2]
                line += String(format: " coneFlank=%.2f", flank)
                // v4.11 diagnostic: the cone's radial surface
                // profile along +x (support+snow), 9 samples —
                // discriminates dome vs shallow-cone vs
                // flat-top shapes behind the flank reading.
                var prof: [String] = []
                for r in 0...8 {
                    let ii = ci + r
                    if ii < s.vx {
                        let h = clean(s.support[cj * s.vx + ii]) + clean(s.snow[cj * s.vx + ii])
                        prof.append(String(format: "%.3f", h))
                    }
                }
                line += " profile=" + prof.joined(separator: ",")
            case "drag2":
                var sum: Float = 0, cnt = 0
                for j in 0..<s.vz {
                    for i in 0..<s.vx {
                        let idx = j * s.vx + i
                        if abs(clean(s.support[idx])) > 0.05 { continue }
                        let x = s.ox + Float(i) * s.cs
                        let z = s.oz + Float(j) * s.cs
                        if abs(z) <= 0.4, x >= Float(dragStartX) + 0.3,
                           x <= Float(dragX) - 0.3 {
                            sum += clean(s.snow[idx]); cnt += 1
                        }
                    }
                }
                line += String(format: " trenchMean=%.4f", cnt > 0 ? sum / Float(cnt) : 0)
            case "spawn2":
                var topSum: Float = 0, topCnt = 0
                var ringSum: Float = 0, ringCnt = 0
                if let box = session?.project.scene.objects.first(where: { $0.primitiveKind == "box" }) {
                    let cx = Float(box.transform.position[0])
                    let cz = Float(box.transform.position[2])
                    for j in 0..<s.vz {
                        for i in 0..<s.vx {
                            let idx = j * s.vx + i
                            let x = s.ox + Float(i) * s.cs
                            let z = s.oz + Float(j) * s.cs
                            let sup = clean(s.support[idx])
                            let inside = abs(x - cx) <= 0.5 && abs(z - cz) <= 0.5
                            if inside, abs(sup - 2.5) < 0.05 {
                                topSum += clean(s.snow[idx]); topCnt += 1
                            } else if abs(sup) < 0.05 {
                                let dx = max(abs(x - cx) - 0.5, 0)
                                let dz = max(abs(z - cz) - 0.5, 0)
                                let rd = sqrtf(dx * dx + dz * dz)
                                if rd > 0.001, rd <= 0.4 {
                                    ringSum += clean(s.snow[idx]); ringCnt += 1
                                }
                            }
                        }
                    }
                }
                line += String(format: " spawnTopMean=%.4f ringMean=%.4f",
                               topCnt > 0 ? topSum / Float(topCnt) : 0,
                               ringCnt > 0 ? ringSum / Float(ringCnt) : 0)
            case "roof":
                var sum: Float = 0, cnt = 0
                for j in 0..<s.vz {
                    for i in 0..<s.vx {
                        let idx = j * s.vx + i
                        let x = s.ox + Float(i) * s.cs
                        let z = s.oz + Float(j) * s.cs
                        let sup = clean(s.support[idx])
                        if sup > 0.9, sup < 3.7, abs(x) <= 1.55, abs(z) <= 1.05 {
                            sum += clean(s.snow[idx]); cnt += 1
                        }
                    }
                }
                line += String(format: " roofFaceMean=%.4f roofFaceCount=%d",
                               cnt > 0 ? sum / Float(cnt) : 0, cnt)
            default:
                break
            }
        }
        print(line)
        fflush(stdout)
        NSLog("%@", line)
        // v4.10.4 round 2: field maps for the deep variants at
        // establishment end and at the final checkpoint —
        // ceiling and sim snow depth, every 2nd vertex,
        // one char per vertex (0-9a-z over the given range),
        // rows north (max z) first. Lets a human SEE whether
        // roughness lives in the ceiling landscape itself,
        // in the sim state, or only near the box.
        if (ReproDriver.variant == "deeprest" || ReproDriver.variant == "deepdrag"),
           label == "pre" || label == "rest-30s" || label == "settled-20s",
           s.initialised, n > 0, s.snow.count >= n, s.ceiling.count >= n {
            let chars = Array("0123456789abcdefghijklmnopqrstuvwxyz")
            func dump(_ name: String, _ field: [Float], _ range: Float) {
                print("REPRO-MAP \(label) \(name) range=\(range)")
                var j = s.vz - 1
                while j >= 0 {
                    var row = ""
                    var i = 0
                    while i < s.vx {
                        let v = clean(field[j * s.vx + i])
                        let q = max(0, min(35, Int(v / range * 36)))
                        row.append(chars[q])
                        i += 2
                    }
                    print("REPRO-MAP \(row)")
                    j -= 2
                }
                fflush(stdout)
            }
            dump("ceil", s.ceiling, 2.2)
            dump("snow", s.snow, 3.5)
        }
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
            dragStepCount += 1
            if dragStepCount % 30 == 0 { emitMetric("dragstep") }
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

    /// v4.11: inject a snow field into the live sim (mass at
    /// density 0.45) from a depth closure over (x, z, support).
    private func injectField(_ depthAt: (Float, Float, Float) -> Float) {
        guard let renderer else { return }
        let s = renderer.reproSimSnapshot()
        let n = s.vx * s.vz
        guard s.initialised, n > 0, s.support.count >= n else { return }
        var mass = [Float](repeating: 0, count: n)
        var total: Float = 0
        for j in 0..<s.vz {
            for i in 0..<s.vx {
                let idx = j * s.vx + i
                let x = s.ox + Float(i) * s.cs
                let z = s.oz + Float(j) * s.cs
                let d = depthAt(x, z, s.support[idx])
                if d > 0 {
                    mass[idx] = d * 0.45
                    total += mass[idx]
                }
            }
        }
        injectedMass = total * s.cs * s.cs
        renderer.reproInjectSnow(mass: mass, rho: 0.45)
        ReproDriver.marker("injected mass=\(injectedMass)")
    }

    /// v4.11 physics-variant phase machine. Establishment uses
    /// the same witnesses as the deep variants; then the field
    /// is injected, weather input is stopped (mode 0 at
    /// temperature -8 C gives zero base rate and zero sink in
    /// the session's own feed), and the variant runs its
    /// course against the REAL sim.
    private func handlePhysics() {
        guard let session else { return }
        let now = CACurrentMediaTime()
        let variant = ReproDriver.variant
        switch physPhase {
        case .establishing:
            guard let renderer else { return }
            let snap = renderer.reproSimSnapshot()
            guard snap.initialised, snap.swapped,
                  sawTickTag, simTickFrames >= 5 else { return }
            switch variant {
            case "pile":
                injectField { x, z, sup in
                    if abs(sup) > 0.05 { return 0 }
                    let r = hypotf(x, z)
                    return r < 1.2 ? 1.6 * (1 - r / 1.2) : 0
                }
                emitMetric("injected")
                session.setWeatherMode(0)
                physPhase = .running
            case "settle", "longrun":
                let fins: [(Float, Float)] = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0),
                                              (0.0, -1.0), (3.0, 2.0), (-3.0, -2.0),
                                              (2.0, -3.0), (-2.0, 3.0)]
                injectField { x, z, sup in
                    var d: Float = 0
                    if abs(sup) < 0.05 || abs(sup - 1.0) < 0.05 { d = 1.2 }
                    if d > 0 {
                        for (fx, fz) in fins where hypotf(x - fx, z - fz) < 0.15 {
                            d += 1.2
                        }
                    }
                    return d
                }
                emitMetric("injected")
                session.setWeatherMode(0)
                physPhase = variant == "longrun" ? .longrunA : .running
            case "drag2":
                injectField { _, _, sup in abs(sup) < 0.05 ? 1.8 : 0 }
                emitMetric("injected")
                session.setWeatherMode(0)
                if let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) {
                    dragStartX = session.project.scene.objects[idx].transform.position[0]
                    dragX = dragStartX
                }
                dragMidEmitted = false
                physPhase = .dragging
                ReproDriver.marker("phys-drag-start x=\(dragStartX)")
            case "spawn2":
                injectField { _, _, sup in abs(sup) < 0.05 ? 1.8 : 0 }
                emitMetric("pre-spawn")
                // The REAL Add path, then the box is stretched
                // tall (2.5 m) so its top PROTRUDES from the
                // 1.8 m field (a 1 m box would be buried by
                // flux — correct physics, wrong test).
                session.addPrimitive("box")
                if let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) {
                    session.project.scene.objects[idx].transform.scale = [1, 2.5, 1]
                    session.project.scene.objects[idx].transform.position[1] = 1.25
                    session.scheduleRebuild(geometry: true)
                    let p = session.project.scene.objects[idx].transform.position
                    ReproDriver.marker("spawn2-placed pos=[\(p[0]), \(p[1]), \(p[2])]")
                }
                emitMetric("spawn-0")
                physPhase = .spawnSnow
            case "roof":
                injectField { x, z, sup in
                    (sup > 0.9 && sup < 3.7 && abs(x) <= 1.55 && abs(z) <= 1.05) ? 0.5 : 0
                }
                emitMetric("injected")
                session.setWeatherMode(0)
                physPhase = .running
            default:
                physPhase = .running
            }
            physPhaseSince = now
        case .running:
            let dur: CFTimeInterval = variant == "pile" ? 15 : (variant == "roof" ? 25 : 20)
            if now - physPhaseSince >= dur {
                emitMetric("final")
                finishDeep()
            }
        case .dragging:
            guard let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) else {
                physPhase = .settling
                physPhaseSince = now
                return
            }
            dragX = min(dragX + 0.011, dragStartX + 3.3)
            session.project.scene.objects[idx].transform.position[0] = dragX
            session.scheduleRebuild(geometry: true)
            dragStepCount += 1
            if dragStepCount % 30 == 0 { emitMetric("dragstep") }
            if !dragMidEmitted, dragX >= dragStartX + 1.65 {
                dragMidEmitted = true
                emitMetric("mid-drag")
            }
            if dragX >= dragStartX + 3.3 - 1e-9 {
                emitMetric("post-drag")
                physPhase = .settling
                physPhaseSince = now
                ReproDriver.marker("phys-drag-end x=\(dragX)")
            }
        case .settling:
            if now - physPhaseSince >= 20 {
                emitMetric("settled")
                finishDeep()
            }
        case .spawnSnow:
            if now - physPhaseSince >= 30 {
                emitMetric("spawn+30s")
                finishDeep()
            }
        case .longrunA:
            if now - physPhaseSince >= 60 {
                emitMetric("lr-60s")
                session.setWeatherMode(2)
                physPhase = .longrunB
                physPhaseSince = now
            }
        case .longrunB:
            if now - physPhaseSince >= 60 {
                emitMetric("lr-120s")
                finishDeep()
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
        if isPhysicsVariant {
            handlePhysics()
            if finished { return }
        }

        let simSatisfied = ReproToggles.simTick ? framesAfterSwap >= 90
                                                : (sawTickTag && simTickFrames >= 5)
        let freezeSatisfied = !isFreezeVariant || freezeDone
        if !isDeepVariant, !isPhysicsVariant, crossedA, firstStepCrossed, sawSwapAfterWeatherOn, simSatisfied, freezeSatisfied {
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
        let watchdogLimit: CFTimeInterval = isDeepVariant ? 200
            : (isPhysicsVariant ? 280 : 100)
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
