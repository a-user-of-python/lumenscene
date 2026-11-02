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

        if !isGravityVariant, !isPerfVariant, crossedA, firstStepCrossed, pumpCount >= 90 {
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
