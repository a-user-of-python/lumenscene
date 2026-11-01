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
        let simSatisfied = ReproToggles.simTick ? framesAfterSwap >= 90
                                                : (sawTickTag && simTickFrames >= 5)
        let freezeSatisfied = !isFreezeVariant || freezeDone
        if crossedA, firstStepCrossed, sawSwapAfterWeatherOn, simSatisfied, freezeSatisfied {
            finished = true
            ReproDriver.marker("settle A=\(a) pumps=\(pumpCount) simTicks=\(simTickFrames)")
            print("REPRO-DONE")
            fflush(stdout)
            NSLog("REPRO-DONE")
            exit(0)
        }

        // Watchdog: a harness outcome, never a crash signal.
        if CACurrentMediaTime() - startedAt > 100 {
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
