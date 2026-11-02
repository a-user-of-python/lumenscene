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
//   wallbox  — v4.12 wall-contact audit: bare 8x8 plane
//              pre-seeded at 0.77 m (temperature -8). Once
//              established, a 1 m box is spawned through the
//              REAL Add path at (0, 0.5, 0) — Sean's exact
//              proportions from the v4.11 device report —
//              weather stops, the carve forms for 6 s, one
//              forced rebuild, then REPRO-CHECK lines measure
//              the RENDERED sheet against the wall (contact
//              line at field depth, nothing above field +
//              0.12 in the wall band, <= 0.05 under the box).
//              Round 2: at Sean's settings the field is drift
//              lace, so wallbox checks are LOCAL (per-vertex,
//              against the field at each vertex's own XZ);
//              walldeep (1.80 m, box aimed at the deepest
//              drift crest) runs the scalar contact checks.
//   carve    — v4.12 carve-only audit: bare 8x8 plane + 1 m
//              box, 1.80 m pre-seed. The box drags +X at
//              0.011 m/pump for 5.8 m (position write +
//              scheduleRebuild, the dragObjectTo pattern);
//              REPRO-CHECK lines verify the invariant
//              (rendered <= field + 0.101 everywhere, >= 0),
//              the carved trough, its persistence after a 20 s
//              rest, and the carved seat under the box.
//   refill   — v4.12 persistence audit: spawn, carve in for
//              6 s, teleport the box to x = 3.0, then measure
//              the abandoned patch's refill after 90 s
//              (REPRO-CHECK: >= +0.10 m, never above field).
//   snowgate — v4.12 shader-gate audit: after 30 pumps the
//              shared accumulation factor is evaluated on the
//              GPU for vertical / flat / 45-degree normals
//              (REPRO-CHECK: vertical == 0 exactly, flat
//              > 0.9, roof strictly between).
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
        case "wallbox", "walldeep":
            // v4.12 wall-contact audit scenes: a bare 8x8
            // plane; the 1 m box is spawned mid-run via
            // addPrimitive (camera target lands it). wallbox =
            // Sean's exact device settings — 0.77 m of snow of
            // 1.80 m max (his readout), box at the origin;
            // walldeep = the same scene at 1.80 m of 2.0 m,
            // with the box aimed at the deepest drift crest
            // (see pickDriftCrest). Temperature -8: mode 0
            // afterwards = no growth, no melt, so the field
            // the box meets is the field audited.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            p.scene.objects = [plane]
            p.scene.camera.target = [0, 1, 0]
            if ReproDriver.variant == "walldeep" {
                p.scene.weather.maxDepth = 2.0
                p.scene.weather.snowAmount = 1.8
                p.scene.weather.snow = 0.9
            } else {
                p.scene.weather.maxDepth = 1.8
                p.scene.weather.snowAmount = 0.77
                p.scene.weather.snow = 0.385
            }
            p.scene.weather.temperature = -8
        case "carve":
            // v4.12 carve-only audit scene: bare 8x8 plane +
            // one 1 m box at (-2.5, 0.5, 0), pre-seeded deep
            // (1.80 m). The box is dragged +X across the field
            // by the phase machine.
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
            p.scene.weather.temperature = -8
        case "refill":
            // v4.12 persistence/refill audit scene: bare 8x8
            // plane pre-seeded deep; a box is spawned mid-run,
            // carved in, then teleported away and the patch's
            // refill is measured.
            var plane = SceneObject.primitive("plane", name: "Repro bare plane")
            plane.params = ["w": 8, "d": 8]
            plane.transform.position = [0, 0, 0]
            p.scene.objects = [plane]
            p.scene.camera.target = [0, 1, 0]
            p.scene.weather.maxDepth = 2.0
            p.scene.weather.snowAmount = 1.80
            p.scene.weather.snow = 0.9
            p.scene.weather.temperature = -8
        case "snowgate":
            // v4.12 shader-gate audit: no scene requirements —
            // the gate is evaluated on the GPU directly
            // (Renderer.reproSnowGateAudit) once the run is up.
            break
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

    // Snow-audit state (v4.12 variants wallbox / carve /
    // refill / snowgate — the games-model audits that replace
    // the retired v4.10.4 deep and v4.11 physics machines).
    private enum SnowPhase { case establishing, acting, waiting, measuring }
    private var snowPhase: SnowPhase = .establishing
    private var snowPhaseSince: CFTimeInterval = 0
    private var carveDragX: Double = -2.5
    private var refillPatch0: Float = -1
    private var refillFieldMean: Float = 0

    private var isSnowAuditVariant: Bool {
        ["wallbox", "walldeep", "carve", "refill", "snowgate"]
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
        print(line)
        fflush(stdout)
        NSLog("%@", line)
    }

    private func finishSnow() {
        finished = true
        print("REPRO-DONE")
        fflush(stdout)
        NSLog("REPRO-DONE")
        exit(0)
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

    private func printTickMetric() {
        guard let renderer, let ms = renderer.reproTickMsAvg else { return }
        let line = String(format: "REPRO-METRIC %@ tickEncodeMsAvg=%.3f (CPU encode time, not GPU)",
                          ReproDriver.variant, ms)
        print(line)
        fflush(stdout)
        NSLog("%@", line)
    }

    /// Establishment witness for the snow-audit variants —
    /// the direct checks (the generic settle's tag latches
    /// cannot fire at pre-seeded amounts; see the note that
    /// has lived here since run 41336002939): swap flag from
    /// the snapshot (staticEverSwapped), sim initialised (a
    /// sheet build's simSync ran), tick evidence.
    private var snowEstablished: Bool {
        guard let renderer else { return false }
        let snap = renderer.reproSimSnapshot()
        return snap.initialised && snap.swapped && sawTickTag && simTickFrames >= 5
    }

    /// v4.12 snow-audit phase machine, called once per pump
    /// AFTER the latch updates (so sawTickTag / simTickFrames
    /// reflect this pump).
    private func handleSnowAudit() {
        let now = CACurrentMediaTime()
        switch ReproDriver.variant {
        case "wallbox", "walldeep": handleWallbox(now: now)
        case "carve": handleCarve(now: now)
        case "refill": handleRefill(now: now)
        case "snowgate": handleSnowgate(now: now)
        default: break
        }
    }

    /// wallbox / walldeep: spawn the 1 m box into snow
    /// through the REAL Add path, stop the weather, let the
    /// carve form, force one more rebuild, then measure the
    /// RENDERED sheet against the wall. wallbox uses Sean's
    /// device settings (0.77 m of 1.80 m max) and LOCAL
    /// per-vertex invariants — at those settings the authored
    /// field is drift lace (≤ ~0.21 m, patchy), so a scalar
    /// "field depth" does not exist; walldeep uses the deep
    /// scene (1.80 m), picks the deepest drift crest from the
    /// snapshot, aims the camera target at it (addPrimitive
    /// lands on the target) and runs the scalar contact
    /// checks with real depth to bite on.
    private func handleWallbox(now: CFTimeInterval) {
        guard let session else { return }
        switch snowPhase {
        case .establishing:
            guard snowEstablished else { return }
            if ReproDriver.variant == "walldeep", let renderer {
                if let crest = pickDriftCrest(renderer) {
                    session.project.scene.camera.target =
                        [Double(crest.0), 1, Double(crest.1)]
                    ReproDriver.marker("walldeep-crest x=\(crest.0) z=\(crest.1) minField=\(crest.2)")
                } else {
                    ReproDriver.marker("walldeep-crest NONE")
                }
            }
            session.addPrimitive("box")
            if let box = session.project.scene.objects.first(where: { $0.primitiveKind == "box" }) {
                let p = box.transform.position
                ReproDriver.marker("wallbox-spawned pos=[\(p[0]), \(p[1]), \(p[2])]")
            } else {
                ReproDriver.marker("wallbox-spawned MISSING-BOX")
            }
            session.setWeatherMode(0)
            snowPhase = .acting
            snowPhaseSince = now
        case .acting:
            if now - snowPhaseSince >= 6 {
                session.scheduleRebuild(geometry: true)
                snowPhase = .waiting
                snowPhaseSince = now
            }
        case .waiting:
            if now - snowPhaseSince >= 2 {
                emitWallboxChecks()
                printTickMetric()
                finishSnow()
            }
        case .measuring:
            break
        }
    }

    /// Bilinear sample of a lattice array from a sim
    /// snapshot at world (x, z), clamped to the grid.
    private func snapBilinear(_ a: [Float], _ vx: Int, _ vz: Int,
                              _ ox: Float, _ oz: Float, _ cs: Float,
                              _ x: Float, _ z: Float) -> Float {
        guard vx > 1, vz > 1, cs > 0, a.count >= vx * vz else { return 0 }
        let fx = min(max((x - ox) / cs, 0), Float(vx - 1))
        let fz = min(max((z - oz) / cs, 0), Float(vz - 1))
        let i0 = min(Int(fx), vx - 2), j0 = min(Int(fz), vz - 2)
        let tx = fx - Float(i0), tz = fz - Float(j0)
        let v00 = a[j0 * vx + i0], v10 = a[j0 * vx + i0 + 1]
        let v01 = a[(j0 + 1) * vx + i0], v11 = a[(j0 + 1) * vx + i0 + 1]
        return (v00 + (v10 - v00) * tx) * (1 - tz)
            + (v01 + (v11 - v01) * tx) * tz
    }

    /// walldeep: the footprint centre (restricted to
    /// |c| <= 1.4 m so the measurement ring stays on the
    /// plane) whose PERIMETER carries the deepest minimum
    /// authored field — the drift crest the box should land
    /// on. Nil when the snapshot has no usable ground.
    private func pickDriftCrest(_ renderer: Renderer) -> (Float, Float, Float)? {
        let s = renderer.reproSimSnapshot()
        let n = s.vx * s.vz
        guard s.initialised, n > 0, s.field.count >= n,
              s.groundHas.count >= n else { return nil }
        var best: (Float, Float, Float, Float)?   // cx, cz, min, mean
        var ci = -7
        while ci <= 7 {
            var cj = -7
            while cj <= 7 {
                let cx = Float(ci) * 0.2, cz = Float(cj) * 0.2
                var mn = Float.greatestFiniteMagnitude
                var sum: Float = 0
                var cnt = 0
                var t: Float = -0.5
                while t <= 0.501 {
                    let pts: [(Float, Float)] = [(cx + t, cz - 0.5), (cx + t, cz + 0.5),
                                                 (cx - 0.5, cz + t), (cx + 0.5, cz + t)]
                    for p in pts {
                        let v = snapBilinear(s.field, s.vx, s.vz, s.ox, s.oz, s.cs, p.0, p.1)
                        if v < mn { mn = v }
                        sum += v
                        cnt += 1
                    }
                    t += 0.1
                }
                let mean = cnt > 0 ? sum / Float(cnt) : 0
                if best == nil || mn > best!.2 || (mn == best!.2 && mean > best!.3) {
                    best = (cx, cz, mn, mean)
                }
                cj += 1
            }
            ci += 1
        }
        guard let b = best else { return nil }
        return (b.0, b.1, b.2)
    }

    private func emitWallboxChecks() {
        guard let renderer, let session else { return }
        let s = renderer.reproSimSnapshot()
        let tris = renderer.reproSheetSnapshot()
        let n = s.vx * s.vz
        guard s.initialised, n > 0, s.field.count >= n, s.groundY.count >= n,
              let box = session.project.scene.objects.first(where: { $0.primitiveKind == "box" })
        else {
            check("snapshot", false, "initialised=\(s.initialised)")
            return
        }
        let cx = Float(box.transform.position[0])
        let cz = Float(box.transform.position[2])
        func rectDist(_ x: Float, _ z: Float) -> Float {
            let dx = max(abs(x - cx) - 0.5, 0)
            let dz = max(abs(z - cz) - 0.5, 0)
            return sqrtf(dx * dx + dz * dz)
        }
        // Local field surface: median of (ground + authored
        // field) over open ground 1–2 m from the footprint —
        // MEASURED, never assumed (the drift field varies).
        var ringTops: [Float] = []
        var underMax: Float = 0
        for j in 0..<s.vz {
            for i in 0..<s.vx {
                let idx = j * s.vx + i
                if !s.groundHas[idx] { continue }
                let x = s.ox + Float(i) * s.cs
                let z = s.oz + Float(j) * s.cs
                let rd = rectDist(x, z)
                if rd > 1.0, rd <= 2.0 {
                    ringTops.append(s.groundY[idx] + s.field[idx])
                }
                if abs(x - cx) <= 0.5, abs(z - cz) <= 0.5,
                   s.snow[idx] > underMax {
                    underMax = s.snow[idx]
                }
            }
        }
        ringTops.sort()
        let fieldLocal = ringTops.isEmpty ? 0 : ringTops[ringTops.count / 2]
        let deep = ReproDriver.variant == "walldeep"
        // LOCAL per-vertex invariants — the only honest
        // reference at ANY settings: the authored field varies
        // on the drift scale (in walldeep the crest's own
        // field is 1.6–1.9 m while the 1–2 m ring median reads
        // 1.16 m; in wallbox the field is drift lace). Every
        // sheet vertex is judged against the field AT ITS OWN
        // XZ. walldeep additionally asserts its scene
        // precondition (perimeter field >= 1.2 m) so the audit
        // cannot silently degrade to a shallow scene.
        if deep {
            var perMin = Float.greatestFiniteMagnitude
            var t: Float = -0.5
            while t <= 0.501 {
                let pts: [(Float, Float)] = [(cx + t, cz - 0.5), (cx + t, cz + 0.5),
                                             (cx - 0.5, cz + t), (cx + 0.5, cz + t)]
                for p in pts {
                    let v = snapBilinear(s.field, s.vx, s.vz, s.ox, s.oz, s.cs, p.0, p.1)
                    if v < perMin { perMin = v }
                }
                t += 0.05
            }
            check("scene", perMin >= 1.2,
                  String(format: "perimeterMinField=%.4f ringMedian=%.4f",
                         perMin, fieldLocal))
        }
        var devs: [Float] = []
        var bandViol = 0
        var bandMaxExcess: Float = 0
        for v in tris {
            let gY = snapBilinear(s.groundY, s.vx, s.vz, s.ox, s.oz, s.cs, v.x, v.z)
            let f = snapBilinear(s.field, s.vx, s.vz, s.ox, s.oz, s.cs, v.x, v.z)
            if rectDist(v.x, v.z) <= 0.25 {
                let excess = v.y - (gY + f + 0.115)
                if excess > bandMaxExcess { bandMaxExcess = excess }
                if excess > 0.01 { bandViol += 1 }
            }
            let perim = max(abs(v.x - cx), abs(v.z - cz))
            if abs(perim - 0.5) <= 0.004 {
                // The clip pins wall-crossing vertices at
                // the FIELD surface (ground + authored
                // field): deviation from it is the moat /
                // float / ramp signature.
                devs.append(abs(v.y - (gY + f)))
            }
        }
        devs.sort()
        let medDev = devs.isEmpty ? 1 : devs[devs.count / 2]
        check("contact", devs.count >= 4 && medDev <= 0.03
              && (devs.last ?? 1) <= 0.10,
              String(format: "count=%d medDev=%.4f maxDev=%.4f",
                     devs.count, medDev, devs.last ?? 0))
        check("wallband", bandViol == 0,
              String(format: "viol=%d maxExcess=%.4f", bandViol, bandMaxExcess))
        check("underbox", underMax <= 0.05,
              String(format: "underMax=%.4f", underMax))
    }

    /// carve: drag the box +X through 1.80 m snow at
    /// 0.011 m/pump (the scripting pattern dragObjectTo uses:
    /// position write + scheduleRebuild), then verify the
    /// carve-only invariant + trough, rest 20 s, verify the
    /// trough persists and the box still sits in carved snow.
    private func handleCarve(now: CFTimeInterval) {
        guard let session else { return }
        switch snowPhase {
        case .establishing:
            guard snowEstablished else { return }
            session.setWeatherMode(0)
            snowPhase = .acting
            snowPhaseSince = now
            ReproDriver.marker("carve-drag-start x=\(carveDragX)")
        case .acting:
            guard let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) else {
                snowPhase = .waiting
                snowPhaseSince = now
                return
            }
            if carveDragX < 3.3 {
                carveDragX = min(3.3, carveDragX + 0.011)
                session.project.scene.objects[idx].transform.position[0] = carveDragX
                session.scheduleRebuild(geometry: true)
            } else {
                emitCarveChecks(final: false)
                snowPhase = .waiting
                snowPhaseSince = now
                ReproDriver.marker("carve-drag-end x=\(carveDragX)")
            }
        case .waiting:
            if now - snowPhaseSince >= 20 {
                emitCarveChecks(final: true)
                finishSnow()
            }
        case .measuring:
            break
        }
    }

    private func emitCarveChecks(final: Bool) {
        guard let renderer else { return }
        let s = renderer.reproSimSnapshot()
        let n = s.vx * s.vz
        guard s.initialised, n > 0, s.snow.count >= n, s.field.count >= n else {
            check("snapshot", false, "initialised=\(s.initialised)")
            return
        }
        var viol = 0
        var minR = Float.greatestFiniteMagnitude
        var pathSum: Float = 0, pathField: Float = 0, pathCnt = 0
        var underMax: Float = 0
        for j in 0..<s.vz {
            for i in 0..<s.vx {
                let idx = j * s.vx + i
                if !s.groundHas[idx] { continue }
                let r = s.snow[idx], f = s.field[idx]
                if r > f + 0.101 { viol += 1 }
                if r < minR { minR = r }
                let x = s.ox + Float(i) * s.cs
                let z = s.oz + Float(j) * s.cs
                if abs(z) <= 0.4, x >= -2.2, x <= Float(carveDragX) - 0.3 {
                    pathSum += r; pathField += f; pathCnt += 1
                }
                if final, abs(x - Float(carveDragX)) <= 0.5, abs(z) <= 0.5,
                   r > underMax {
                    underMax = r
                }
            }
        }
        let pathMean = pathCnt > 0 ? pathSum / Float(pathCnt) : 0
        let fieldMean = pathCnt > 0 ? pathField / Float(pathCnt) : 0
        if !final {
            check("invariant", viol == 0 && minR >= -0.001,
                  String(format: "viol=%d minRendered=%.4f", viol, minR))
            check("trough", pathCnt > 0 && pathMean <= 0.5 * fieldMean,
                  String(format: "pathMean=%.4f fieldMean=%.4f",
                         pathMean, fieldMean))
            printTickMetric()
        } else {
            check("persist", pathCnt > 0 && pathMean <= 0.7 * fieldMean,
                  String(format: "pathMean=%.4f fieldMean=%.4f",
                         pathMean, fieldMean))
            check("underbox", underMax <= 0.05,
                  String(format: "underMax=%.4f", underMax))
        }
    }

    /// refill: spawn the box, let it carve in (6 s), measure
    /// the compressed patch, teleport the box clear across
    /// the plane, then measure the patch's refill after 90 s.
    private func handleRefill(now: CFTimeInterval) {
        guard let session else { return }
        switch snowPhase {
        case .establishing:
            guard snowEstablished else { return }
            session.addPrimitive("box")
            session.setWeatherMode(0)
            snowPhase = .acting
            snowPhaseSince = now
        case .acting:
            if now - snowPhaseSince >= 6 {
                // A rebuild makes the sheet/sim read the
                // formed carve before it is measured.
                session.scheduleRebuild(geometry: true)
                snowPhase = .waiting
                snowPhaseSince = now
            }
        case .waiting:
            if refillPatch0 < 0 {
                // First waiting pump after the rebuild: the
                // snapshot now carries the formed carve.
                guard now - snowPhaseSince >= 1 else { return }
                let m = refillMeasure()
                refillPatch0 = m.patch
                refillFieldMean = m.field
                guard let idx = session.project.scene.objects.firstIndex(where: { $0.primitiveKind == "box" }) else {
                    check("teleport", false, "box-missing")
                    finishSnow()
                    return
                }
                session.project.scene.objects[idx].transform.position[0] = 3.0
                session.scheduleRebuild(geometry: true)
                ReproDriver.marker("refill-teleported patch0=\(m.patch) field=\(m.field)")
                snowPhaseSince = now
                snowPhase = .measuring
            }
        case .measuring:
            if now - snowPhaseSince >= 90 {
                let m = refillMeasure()
                check("refill", m.patch >= refillPatch0 + 0.10
                      && m.patch <= refillFieldMean,
                      String(format: "patch0=%.4f patch90=%.4f field=%.4f",
                             refillPatch0, m.patch, refillFieldMean))
                finishSnow()
            }
        }
    }

    /// Mean rendered depth over the ORIGINAL box footprint
    /// (0, 0) ± 0.5 plus the local field mean around it.
    private func refillMeasure() -> (patch: Float, field: Float) {
        guard let renderer else { return (0, 0) }
        let s = renderer.reproSimSnapshot()
        let n = s.vx * s.vz
        guard s.initialised, n > 0, s.snow.count >= n, s.field.count >= n
        else { return (0, 0) }
        var pSum: Float = 0, pCnt = 0
        var fSum: Float = 0, fCnt = 0
        for j in 0..<s.vz {
            for i in 0..<s.vx {
                let idx = j * s.vx + i
                if !s.groundHas[idx] { continue }
                let x = s.ox + Float(i) * s.cs
                let z = s.oz + Float(j) * s.cs
                if abs(x) <= 0.5, abs(z) <= 0.5 {
                    pSum += s.snow[idx]; pCnt += 1
                }
                let r = hypotf(x, z)
                if r > 1.5, r <= 2.5 {
                    fSum += s.field[idx]; fCnt += 1
                }
            }
        }
        return (pCnt > 0 ? pSum / Float(pCnt) : 0,
                fCnt > 0 ? fSum / Float(fCnt) : 0)
    }

    /// snowgate: the accumulation gate evaluated on the GPU —
    /// vertical EXACTLY 0, flat near 1, 45° strictly between.
    private func handleSnowgate(now: CFTimeInterval) {
        guard let renderer else { return }
        if pumpCount < 30 { return }
        guard let g = renderer.reproSnowGateAudit() else {
            check("gate", false, "audit-unavailable")
            finishSnow()
            return
        }
        check("vertical-zero", g.vertical == 0,
              String(format: "vertical=%.6f", g.vertical))
        check("flat-full", g.flat > 0.9,
              String(format: "flat=%.6f", g.flat))
        check("roof-between", g.roof > 0.3 && g.roof < g.flat,
              String(format: "roof=%.6f flat=%.6f", g.roof, g.flat))
        finishSnow()
    }

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
        // Snow-audit variants run their own phase machine and
        // finish from it (REPRO-DONE there); the generic
        // settle below would otherwise end them at
        // establishment.
        if isSnowAuditVariant {
            handleSnowAudit()
            if finished { return }
        }

        let simSatisfied = ReproToggles.simTick ? framesAfterSwap >= 90
                                                : (sawTickTag && simTickFrames >= 5)
        let freezeSatisfied = !isFreezeVariant || freezeDone
        if !isSnowAuditVariant, crossedA, firstStepCrossed, sawSwapAfterWeatherOn, simSatisfied, freezeSatisfied {
            finished = true
            ReproDriver.marker("settle A=\(a) pumps=\(pumpCount) simTicks=\(simTickFrames)")
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
        case "wallbox", "walldeep": watchdogLimit = 150
        case "carve": watchdogLimit = 240
        case "refill": watchdogLimit = 300
        case "snowgate": watchdogLimit = 90
        default: watchdogLimit = 100
        }
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
