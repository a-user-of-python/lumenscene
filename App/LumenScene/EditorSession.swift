// EditorSession.swift — the editor's glue: owns the open project,
// the selection, undo/redo (scene-object snapshots), the imported
// meshes, keyboard movement, play mode, saving and export. Views
// observe it; the viewport's draw loop calls frameTick each frame.

import Foundation
import SwiftUI
import MetalKit
import Photos
import simd

enum EditorMode { case edit, play }

enum MoveKey: Hashable { case w, a, s, d, q, e, up, down, left, right }

/// v4.9: dedicated observable for the tree panel's FOOTER only
/// (hint line, weather cover caption, snow-depth line,
/// temperature status line). Before v4.9 the footer lived inside
/// `treePanel`, which observes the whole session, so EVERY
/// @Published change — including the per-percent weather sync —
/// re-evaluated the footer along with everything else. The
/// footer now observes ONLY this object, whose fields are
/// written exclusively by `EditorSession.refreshFooter()` and
/// only when the displayed string actually changes (the v3.1
/// whole-percent discipline, applied to strings). Rendered
/// content is byte-identical to the pre-v4.9 footer.
@MainActor
final class FooterDisplay: ObservableObject {
    @Published private(set) var hintText: String = ""
    /// Weather cover caption ("Ground: 62% wet · Snow cover:
    /// 30%"); nil = the line is not shown.
    @Published private(set) var weatherText: String? = nil
    /// "Snow depth: A m of M m max"; nil = not shown.
    /// Temperature status line; nil = not shown.
    /// v4.10.1: post-crash diagnostic line ("Last step: …"),
    /// the weather-accumulation step the previous session
    /// last reached; nil = not shown.
    @Published private(set) var diagText: String? = nil

    /// Change-only writes: an unchanged string never fires
    /// objectWillChange, so a footer observing this object
    /// re-evaluates only when a displayed line really changes.
    func update(hint: String, weather: String?, diag: String? = nil) {
        if hintText != hint { hintText = hint }
        if weatherText != weather { weatherText = weather }
        if diagText != diag { diagText = diag }
    }
}

@MainActor
final class EditorSession: ObservableObject {
    @Published var project: Project
    @Published var selection: UUID? {
        // An object selection and an area selection are mutually
        // exclusive: the properties panel shows exactly one editor.
        didSet {
            if selection != nil, selectedArea != nil { selectedArea = nil }
            syncHighlightToRenderer()
        }
    }
    /// v4.6 (workstream 2): MULTI-SELECT. `multiSelectMode` is
    /// the tree-panel tool toggle; `multiSelection` is the set
    /// it builds. `selection` stays the PRIMARY (last tapped)
    /// member — with a set of one, everything behaves exactly
    /// as the old single selection. Leaving the tool (or
    /// entering Play) clears the set: switching tools clears,
    /// per normal editor behaviour.
    @Published var multiSelectMode: Bool = false {
        didSet {
            if !multiSelectMode { multiSelection = [] }
            syncHighlightToRenderer()
            // v4.9: the footer hint line depends on this toggle.
            refreshFooter()
        }
    }
    @Published var multiSelection: Set<UUID> = [] {
        didSet { syncHighlightToRenderer() }
    }
    /// v4.0: the trigger area selected in the tree's Areas section
    /// (edited in the properties panel instead of an object).
    @Published var selectedArea: UUID? {
        didSet { if selectedArea != nil, selection != nil { selection = nil } }
    }
    @Published var mode: EditorMode = .edit
    @Published var status: String = ""
    @Published var undoDepth: Int = 0
    @Published var redoDepth: Int = 0
    /// v4.0: transient Play-mode banner (node "Show Message"
    /// action). Display-only; frameTick clears it at expiry.
    @Published var playBanner: String = ""

    /// v4.9: the tree footer's narrow observable (see
    /// FooterDisplay). A `let`, deliberately NOT @Published:
    /// publishing it through the session would re-couple the
    /// footer to every session change. Views observe `footer`
    /// directly; the session writes it change-only via
    /// refreshFooter().
    let footer = FooterDisplay()

    /// v4.10.1: the weather-accumulation step the PREVIOUS
    /// session last reached, read back from the renderer's
    /// persisted tag at init. Shown in the footer ("Last
    /// step: …") until this session's weather path writes a
    /// step of its own — so after a crash, the relaunch names
    /// the step the crash interrupted. "idle" (no weather ran,
    /// or a healthy last step) shows nothing.
    private var previousSessionStep: String? = {
        let t = UserDefaults.standard.string(forKey: "lumenscene.weatherStep")
        return (t != nil && t != "idle") ? t : nil
    }()

    /// v4.10.2: the renderer's persisted count of non-finite
    /// sim values the sheet readback has replaced with 0, read
    /// at init and shown in the footer alongside the last-step
    /// readout — if bad snow values are occurring at all, the
    /// number grows between sessions.
    private var previousNanClamps: Int =
        UserDefaults.standard.integer(forKey: "lumenscene.nanClampCount")

    let renderer: Renderer
    /// v4.0: the node-graph game runtime. Inert in edit mode;
    /// started/stopped by toggleMode().
    let runtime: GameRuntime
    private(set) var meshes: [UUID: MeshData] = [:]

    /// v4.9: scene-objects revision for the multi-select member
    /// cache (see multiSelectionMembers()). Incremented ONLY by
    /// noteSceneObjectsChanged(), which every method that adds,
    /// removes, replaces or re-fields a scene object calls (the
    /// call sites are enumerated on that method). Never bumped
    /// per frame: frameTick does not touch it, and the per-frame
    /// runtime funnels that write ONLY transform/material
    /// (runtimeSetTransform / runtimeSetMaterial, viewport
    /// dragging) cannot stale the cache, because the cached
    /// member list is consumed only for name / visible / count.
    private var sceneRevision: Int = 0
    private var cachedMultiMembers: [SceneObject] = []
    private var cachedMultiSet: Set<UUID>? = nil
    private var cachedMultiRevision: Int = -1

    /// v4.9: mark the scene-object list as changed for the
    /// member cache. Call sites (audited against every
    /// `project.scene.objects` write in this file): restore,
    /// restorePlaySnapshot, mutateScene, objectBinding's
    /// setter, addPrimitive, deleteSelected,
    /// deleteMultiSelection, importModelFile (on success),
    /// promoteObject (writes `visible`), runtimeSetVisible's
    /// static fallback (writes `visible`), deleteArea
    /// (un-assigns areaID), syncExposuresToProject (writes
    /// `snowExposure`, save-rate). Writes that touch ONLY transform
    /// or material (dragObjectTo, runtimeSetTransform,
    /// runtimeSetMaterial) deliberately do NOT call this: the
    /// cache's consumers read name/visible/membership only, so
    /// those writes cannot make it stale.
    private func noteSceneObjectsChanged() { sceneRevision += 1 }

    /// v4.9: the multi-selection's member objects, resolved
    /// with ONE filter pass and memoised on (multiSelection,
    /// sceneRevision). Body passes driven by unrelated
    /// @Published churn (e.g. the per-percent weather sync)
    /// hit the cache instead of re-filtering the whole scene.
    /// Invalidation is the revision contract above: any write
    /// that could change a member's name/visible or the set's
    /// membership bumps the revision first.
    func multiSelectionMembers() -> [SceneObject] {
        if cachedMultiSet == multiSelection,
           cachedMultiRevision == sceneRevision {
            return cachedMultiMembers
        }
        let members = project.scene.objects.filter {
            multiSelection.contains($0.id)
        }
        cachedMultiMembers = members
        cachedMultiSet = multiSelection
        cachedMultiRevision = sceneRevision
        return members
    }

    /// v4.9: recompute the footer's four displayed lines from
    /// current state and push them into `footer` change-only.
    /// The expressions below are the pre-v4.9 EditorView ones
    /// verbatim (weatherCaption, the snow-depth format, the
    /// hint strings, temperatureStatusText), so the rendered
    /// footer is byte-identical. Called from init, setWeather-
    /// Mode, setMaxSnowDepth, setTemperature,
    /// clearWeatherBuildup, toggleMode, the multiSelectMode
    /// didSet, and once per frameTick — the per-frame call is
    /// a few short string builds whose publishes are guarded
    /// inside FooterDisplay.update, so it costs nothing unless
    /// a displayed line actually changes (snow depth at its
    /// 0.01 m format granularity, freeze status at its
    /// thresholds, the caption at whole percents).
    func refreshFooter() {
        let w = project.scene.weather
        var parts: [String] = []
        if w.wetness > 0.02 {
            parts.append("Ground: \(Int((w.wetness * 100).rounded()))% wet")
        }
        if parts.isEmpty, w.mode == 1 { parts.append("Ground: 0% wet") }
        let showWeather = w.mode != 0 || w.wetness > 0.02
        // v4.10.1: the previous session's last weather step
        // stays readable in the footer until this session's
        // own weather path writes a step (the renderer's tag
        // leaves "idle"), at which point the readout clears.
        var diag: String? = nil
        if let prev = previousSessionStep {
            if renderer.weatherStep == "idle" {
                diag = "Last step: \(prev)"
            } else {
                previousSessionStep = nil
            }
        }
        // v4.10.2: the NaN-clamp count rides the same readout.
        if previousNanClamps > 0 {
            let extra = "NaN clamps: \(previousNanClamps)"
            diag = diag.map { $0 + " · " + extra } ?? extra
        }
        footer.update(
            hint: isIdleFrozen
                ? "Scene frozen — tap, press a key, or move to resume."
                : (multiSelectMode
                ? "Multi Select is on: tap objects in the view or the list to add or remove them. Drag any selected object to move them all. Tap empty space to clear."
                : "Tap an object in the view to select it."),
            weather: showWeather ? parts.joined(separator: " · ") : nil,
            diag: diag)
    }

    private struct Snapshot: Codable { var objects: [SceneObject] }
    private var undoStack: [Data] = []
    private var redoStack: [Data] = []
    private var editGestureAt: Date = .distantPast

    // MARK: v4.0 Play-session state (node runtime + promotion)

    /// Restore point taken the moment Play is entered: the scene
    /// objects (via the same snapshot() the undo stack uses) plus
    /// the weather mode. Exiting Play restores both, so everything
    /// the runtime did — moves, material swaps, hides, area
    /// streaming — is wiped and the scene returns to exactly the
    /// state the user pressed Play on. Deliberately separate from
    /// restore(): that one is undo plumbing (selection fix-ups,
    /// stack bookkeeping) and must not run on the Play boundary.
    private var playSnapshot: Data?
    private var playWeatherMode: Int = 0
    /// v4.1 (F15): Play-entry snapshot of the weather build-up
    /// accumulators + quantised geometry steps, restored on Play
    /// exit beside `playWeatherMode` — otherwise a Play session's
    /// Set Weather / Clear (or Weather Control level set) would
    /// leak into the authored scene, because save() syncs the
    /// accumulators into the published project.
    private var playBuildupWet: Double = 0
    private var playGeoStepWet: Int = 0

    /// v4.1 (§3.4): Play player speed in m/s. Session-only (NOT
    /// persisted); Play entry resets it to the v4.0 constant.
    var playSpeed: Float = 2.4

    /// v4.1 (§3.5): camera-shake state. Session-only, non-published
    /// (same discipline as the weather accumulators). frameTick
    /// perturbs the computed camera BASIS only while a shake is
    /// active; the persisted CameraState is never written by it.
    private var shakeAmplitude: Float = 0
    private var shakeEndsAt: CFTimeInterval = 0
    private var shakeDuration: Float = 0

    /// v4.1 (§4.4): Weather Control scales. Session-only,
    /// non-published; reset to 1.0 on Play entry (and exit).
    private var weatherRateScale: Double = 1.0
    private var weatherIntensityScale: Double = 1.0

    /// v4.1 (F29): set by promoteObject instead of rebuilding
    /// immediately; frameTick performs ONE coalesced rebuild (plus
    /// runtime.collisionDidChange()) right after runtime.tick.
    private var promotionRebuildPending: Bool = false

    /// v4.1 (§7): one spawned clone. Clones are NOT scene objects:
    /// they live only in this registry + the renderer's entry list
    /// for the duration of a Play session.
    struct SpawnedClone {
        var cloneID: UUID        // fresh UUID(), also the DynEntry key
        var sourceID: UUID       // authored object the clone came from
        var transform: Transform3
        var material: MaterialProps   // snapshot at spawn time
        var triCount: Int
    }
    private var clones: [SpawnedClone] = []        // spawn order
    private var cloneEntries: [DynamicEntry] = []  // renderer mirror, clones only

    /// v4.1 (§7.2): live clone counts for the runtime's planner.
    var liveCloneCount: Int { clones.count }
    var liveCloneTriCount: Int { clones.reduce(0) { $0 + $1.triCount } }

    /// PROMOTION, the mechanism in one paragraph: the renderer
    /// draws static objects from one baked soup (rebuilt on edit)
    /// and dynamic objects from per-object entries the runtime can
    /// move/re-skin per frame without a rebuild. The first time
    /// the runtime touches an object in Play it is "promoted": a
    /// dynamic entry is created for it AND its scene `visible` is
    /// set to false, so every static rebuild (area load/unload,
    /// weather geometry) naturally excludes it — the static layer
    /// has no per-object exclusion list, and `visible == false` is
    /// the exclusion it already understands. The object's real
    /// visibility lives in these sets instead: promoted but not in
    /// `hiddenPromotedIDs` (runtime hide) and not in
    /// `areaHiddenPromotedIDs` (its area is streamed out) = its
    /// dynamic entry is on screen. The scene's false `visible`
    /// flags are never user-visible — the editor UI is hidden in
    /// Play — and the Play-exit snapshot restore undoes them all.
    var promotedIDs: Set<UUID> = []
    var hiddenPromotedIDs: Set<UUID> = []
    private var areaHiddenPromotedIDs: Set<UUID> = []
    /// Session-side mirror of the renderer's dynamic entries, so
    /// promotion can hand setDynamicEntries the full list.
    private var dynamicEntries: [DynamicEntry] = []
    /// Areas the runtime has streamed out this Play session.
    private var runtimeUnloaded: Set<UUID> = []
    private var bannerExpiry: Date?

    /// Live weather build-up (source of truth while editing).
    /// Mutating `project` every frame would fire objectWillChange
    /// 60x/s and re-evaluate the whole editor UI, so frameTick
    /// advances these plain accumulators and only syncs the
    /// published values when the displayed percentage changes.
    /// v4.3: `buildupSnow` no longer carries the 0...1 level — it
    /// carries the ABSOLUTE accumulation A in metres (at the
    /// normalised drift factor). The 0...1 progress the rest of
    /// the app speaks is derived: p = A / maxDepth (see below).
    private var buildupWet: Double = 0
    /// v4.7: cumulative snowfall clock (metres fallen, uncapped,
    /// non-published per frame like the other accumulators).
    /// v4.7: puddle freeze fraction 0...1 (non-published).

    // MARK: v4.3 snow depth / exposure / footprints / player figure

    /// The user-set target depth (live from the published
    /// weather state; the settings sheet edits it as an explicit
    /// user action, which is a legitimate published write).
    /// Step size in metres for quantised geometry: maxDepth / 24.
    /// Unclamped accumulation progress p = A / maxDepth.
    /// Clamped progress pc in 0...1 — what the persisted
    /// `weather.snow` level and the continuous shading level are.
    /// Quantised AMOUNT step of A (uncapped — past 24 while deep
    /// snow above the target compacts). This is what geoStepSnow
    /// now counts; the renderer's weatherGeoSnow is the clamped
    /// progress step derived from it.
    /// Renderer-facing quantised progress (0...24).

    /// v4.3 (§4): per-object snow exposure — each scene object
    /// AND each Play clone tracks the absolute accumulation it
    /// has personally caught up with. New/spawned objects start
    /// at 0 and gather snow only as fast as the ambient A grows;
    /// loaded objects start at their persisted value (absent =
    /// fully caught up). Synced WHOLESALE to the renderer every
    /// tick (assignment is the sync — a deleted object's entry
    /// can never linger) and written back into the scene objects
    /// only on save/export (published-write discipline).
    /// The per-object exposure steps already baked into geometry
    /// / dynamic materials, for the coalesced rebuild trigger.

    /// v4.3: player footprints in Play — transient stamps in the
    /// snow, fed to the renderer as compression spots. Nothing
    /// here is persisted; Play exit clears them before the
    /// snapshot-restore rebuild so restored snow is pristine.

    /// v4.3: player-figure driver state (session-only, visual
    /// only — playPosition / eye height / pose funnels are never
    /// touched by it). Heading is smoothed toward the movement
    /// direction (fallback: the look yaw); the walk phase
    /// advances with distance actually travelled.
    private var figureHeading: Float = 0
    private var figureWalkPhase: Float = 0
    private var figureWalkBlend: Float = 0
    private var figureLastPlayPos: SIMD3<Float>?
    private var figureLastSentPhase: Float = -.infinity
    private var figureLastSentBlend: Float = -.infinity
    private var figureLastMatrix: float4x4?
    private var editFigureSet: Bool = false
    /// v3.4: quantised build-up steps (of 24) already baked into
    /// the renderer's weather GEOMETRY (snow displacement +
    /// puddle water sheets). -1 = nothing baked yet. Geometry
    /// rebuilds are the expensive part of weather, so they run
    /// only when a step boundary is crossed, at most ~1 per
    /// 0.8 s; the shading levels stay continuous throughout.
    private var geoStepWet = -1
    private var lastGeoRebuildAt: CFTimeInterval = 0

    /// Weather dynamics (v3.2 — same non-published discipline
    /// as the build-up accumulators): a clock that runs while
    /// weather is on, the surge intensity it produces, and the
    /// gusting wind vector. The clock resets when weather is
    /// switched on, so each storm fades in and finds its phase.
    private var weatherClock: Double = 0
    private var dynIntensity: Double = 0
    private var dynWind = SIMD2<Float>(1.2, 0.4)

    var pressedKeys: Set<MoveKey> = []

    /// Touch joystick vector (Play mode): x = strafe right,
    /// y = forward, each in -1...1.
    var touchVector = SIMD2<Float>(0, 0)
    var touchActive: Bool { simd_length(touchVector) > 0.02 }

    /// The live viewport (set by the view's coordinator). Draws
    /// happen only when kicked — see kickRender().
    weak var viewport: MTKView?
    private var lastCameraActivity: CFTimeInterval = 0

    // MARK: v4.10 Idle freeze (WS1) + Final render (WS2) state

    /// Wall-clock time of the last user input (see
    /// noteUserInput). Initialised in init.
    private var lastInputAt: CFTimeInterval = 0
    /// True while the scene is idle-frozen: frameTick returns
    /// before every scene-time consumer, the frame chain is
    /// dead (wantsFrame false), and renderer.draw is not called.
    private(set) var isIdleFrozen = false
    private var frozenSince: CFTimeInterval = 0
    /// A kickRender() that arrived while frozen collapses into
    /// this flag; the wake path always kicks once, which is
    /// that collapsed kick made real.
    private var pendingKickWhileFrozen = false

    /// v4.10 (WS1): the idle-freeze delay, held on the renderer
    /// exactly like waterReflectionScale (in-memory preference,
    /// not project data). 0 = Off. Changing it takes effect
    /// immediately — frameTick compares it against the elapsed
    /// idle time every frame, so a shorter delay can engage on
    /// the very next frame of an idle period already under way,
    /// and a longer one (or Off) simply never engages.
    var idleFreezeSeconds: Double {
        get { renderer.idleFreezeSeconds }
        set { renderer.idleFreezeSeconds = newValue }
    }

    /// Every input path funnels here: viewport gestures, the
    /// window-level touch observer, camera activity, hardware
    /// keys, the Play joystick, and the discrete menu/button
    /// funnels (save / exports / Play-Pause / weather / undo /
    /// redo). Records the input; if the scene is frozen, wakes
    /// it (see wakeFromIdleFreeze for the no-time-jump rules).
    func noteUserInput() {
        lastInputAt = CACurrentMediaTime()
        if isIdleFrozen { wakeFromIdleFreeze() }
    }

    /// Wake from an idle freeze with NO scene-time jump:
    /// - lastTick is set to nil, so the wake frame's dt is 0 —
    ///   the raw dt that frameTick hands runtime.tick can never
    ///   contain frozen time (movement/weather blocks also see
    ///   dt == 0 on that frame).
    /// - Wall-clock scene clocks are rebased by the frozen
    ///   span: the camera shake's end time and the Play
    ///   banner's expiry shift by the span, and the renderer
    ///   rebases its precip phase + weather-sim clocks
    ///   (noteSceneResumed). The banner expiry is a Date, so
    ///   it shifts by the same span in wall time.
    private func wakeFromIdleFreeze() {
        guard isIdleFrozen else { return }
        isIdleFrozen = false
        let span = CACurrentMediaTime() - frozenSince
        if shakeEndsAt > frozenSince { shakeEndsAt += span }
        if let expiry = bannerExpiry {
            bannerExpiry = expiry.addingTimeInterval(span)
        }
        frozenSince = 0
        lastTick = nil
        renderer.noteSceneResumed(frozenSpan: span)
        pendingKickWhileFrozen = false
        refreshFooter()
        kickRender()
    }

    /// Operations that must never be frozen across: a static
    /// AS build completing mid-freeze would swap geometry the
    /// frozen frame cannot show (and its completion drives
    /// frame requests the freeze would swallow), and a final
    /// render reads the scene on a background queue. Save /
    /// export / load / thumbnail are NOT listed: they are
    /// main-thread synchronous and input-initiated —
    /// noteUserInput precedes them, so they can never coincide
    /// with the engage check, which only runs inside frameTick.
    private func protectedOperationInFlight() -> Bool {
        renderer.staticBuildInFlight || finalRenderActive
    }

    // MARK: v4.10 Final render state machine (WS2)

    enum FinalRenderPhase { case idle, rendering, done, failed }
    @Published var finalRenderPhase: FinalRenderPhase = .idle
    @Published var finalRenderSamplesDone: Int = 0
    @Published var finalRenderTotalSamples: Int = 4096
    @Published var finalRenderPreview: CGImage? = nil
    @Published var finalRenderImage: CGImage? = nil
    @Published var finalRenderStatusText: String = ""
    /// Drives the Render Image sheet (showNodes pattern).
    @Published var finalRenderPresented: Bool = false
    @Published var finalRenderWasCancelled: Bool = false
    /// True while a final render runs — feeds the idle-freeze
    /// protected list and frameTick's scene-time pause.
    var finalRenderActive: Bool { finalRenderPhase == .rendering }
    private let finalRenderCancelLock = NSLock()
    private var finalRenderCancelFlag = false
    private func setFinalRenderCancelFlag(_ value: Bool) {
        finalRenderCancelLock.lock(); finalRenderCancelFlag = value
        finalRenderCancelLock.unlock()
    }
    private func finalRenderIsCancelled() -> Bool {
        finalRenderCancelLock.lock(); defer { finalRenderCancelLock.unlock() }
        return finalRenderCancelFlag
    }

    /// Start a final render of the current view. Wakes a
    /// frozen scene first (the render shows the frozen state
    /// as it stands); the scene stays paused for the render's
    /// whole duration (frameTick early-returns while
    /// finalRenderActive), and the render itself runs on a
    /// background queue with progress hopping to main.
    func startFinalRender() {
        noteUserInput()
        guard finalRenderPhase != .rendering else {
            finalRenderPresented = true
            return
        }
        let basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
        let size = viewport?.drawableSize ?? CGSize(width: 2360, height: 1640)
        let width = max(1, Int(size.width))
        let height = max(1, Int(size.height))
        setFinalRenderCancelFlag(false)
        finalRenderSamplesDone = 0
        finalRenderTotalSamples = 4096
        finalRenderPreview = nil
        finalRenderImage = nil
        finalRenderWasCancelled = false
        finalRenderStatusText = "Rendering…"
        finalRenderPhase = .rendering
        finalRenderPresented = true
        let renderer = self.renderer
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            // bounces 16 (vs the interactive 5–6): a ray can
            // survive ~5 full mirror→glass→mirror round trips
            // before the budget ends it, where the interactive
            // budget truncates such chains visibly. glassBudget
            // 16 matches: 16 transmission crossings per path.
            let image = renderer.renderFinalImage(
                camera: basis, width: width, height: height,
                totalSamples: 4096, bounces: 16, glassBudget: 16,
                chunkSamples: 128,
                shouldCancel: { [weak self] in self?.finalRenderIsCancelled() ?? true },
                onProgress: { [weak self] done, snapshot in
                    DispatchQueue.main.async {
                        guard let self, self.finalRenderPhase == .rendering else { return }
                        self.finalRenderSamplesDone = done
                        if let snapshot { self.finalRenderPreview = snapshot }
                    }
                })
            DispatchQueue.main.async {
                guard let self else { return }
                self.finalRenderWasCancelled = self.finalRenderIsCancelled()
                if let image {
                    self.finalRenderImage = image
                    self.finalRenderPreview = image
                    self.finalRenderPhase = .done
                    self.finalRenderStatusText = self.finalRenderWasCancelled
                        ? "Cancelled at \(self.finalRenderSamplesDone) samples — best image kept"
                        : "Render complete — \(self.finalRenderTotalSamples) samples"
                } else {
                    self.finalRenderPhase = .failed
                    self.finalRenderStatusText = "Render failed — no image was produced."
                }
                self.kickRender()
            }
        }
    }

    /// Cancel the running render: the render loop notices
    /// between chunks (bounded by one 128-sample chunk) and
    /// finishes with the best image accumulated so far, which
    /// the completion path above keeps as the final image.
    func cancelFinalRender() {
        setFinalRenderCancelFlag(true)
    }

    /// PNG data for the finished render (renderStillPNG's
    /// encoding, via the renderer's shared helper).
    func finalRenderPNGData() -> Data? {
        guard let image = finalRenderImage else { return nil }
        return renderer.pngData(from: image)
    }

    /// Save the finished render into the Photos library
    /// (add-only authorization; NSPhotoLibraryAddUsageDescription
    /// is in the app's Info.plist).
    func saveFinalRenderToPhotos() {
        guard let data = finalRenderPNGData() else {
            finalRenderStatusText = "No finished render to save yet."
            return
        }
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { [weak self] auth in
            guard auth == .authorized || auth == .limited else {
                DispatchQueue.main.async {
                    self?.finalRenderStatusText = "Photos access not allowed — Save to Files or Share instead."
                }
                return
            }
            PHPhotoLibrary.shared().performChanges {
                let request = PHAssetCreationRequest.forAsset()
                request.addResource(with: .photo, data: data, options: nil)
            } completionHandler: { success, _ in
                DispatchQueue.main.async {
                    self?.finalRenderStatusText = success ? "Saved to Photos." : "Saving to Photos failed."
                }
            }
        }
    }

    /// Write the finished render as a PNG into
    /// Documents/Renders/<project name>-render.png — visible
    /// in the Files app via the app's file sharing.
    @discardableResult
    func saveFinalRenderToFiles() -> URL? {
        guard let data = finalRenderPNGData() else {
            finalRenderStatusText = "No finished render to save yet."
            return nil
        }
        do {
            let docs = FileManager.default.urls(for: .documentDirectory,
                                                in: .userDomainMask)[0]
            let folder = docs.appendingPathComponent("Renders", isDirectory: true)
            try FileManager.default.createDirectory(at: folder,
                                                    withIntermediateDirectories: true)
            let name = project.name.replacingOccurrences(of: "/", with: "-")
            let url = folder.appendingPathComponent("\(name.isEmpty ? "Untitled" : name)-render.png")
            try data.write(to: url, options: .atomic)
            finalRenderStatusText = "Saved to Files: Renders/\(url.lastPathComponent)"
            return url
        } catch {
            finalRenderStatusText = "Saving to Files failed."
            return nil
        }
    }

    /// Temp PNG for the Share sheet (exportPNGURL's pattern).
    func finalRenderShareURL() -> URL? {
        guard let data = finalRenderPNGData() else { return nil }
        let name = project.name.replacingOccurrences(of: "/", with: "-")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name.isEmpty ? "Untitled" : name)-render.png")
        do {
            try data.write(to: url, options: .atomic)
            return url
        } catch { return nil }
    }

    init(project: Project, renderer: Renderer) {
        self.runtime = GameRuntime()
        self.project = project
        self.renderer = renderer
        // v4.10 (WS1): the idle clock starts at session birth,
        // so a freshly opened scene gets a full delay of grace.
        lastInputAt = CACurrentMediaTime()
        runtime.session = self
        for obj in project.scene.objects where obj.source == "model" {
            if let file = obj.meshFile,
               let mesh = ProjectStore.readMesh(projectID: project.id, fileName: file) {
                meshes[obj.id] = mesh
            }
        }
        buildupWet = project.scene.weather.wetness
        // v4.3: A comes from the persisted absolute amount when
        // the file has one; v1–v4.2 files carry only the 0...1
        // level, so A = level * maxDepth (at the default 0.3 m
        // that is exactly the old build-up).
        // v4.13: snow is removed — a project saved in snow
        // mode opens with weather off; its stored snow state
        // (amount, exposures, plow layer, freeze) is ignored.
        if project.scene.weather.mode == 2 {
            self.project.scene.weather.mode = 0
        }
        renderer.onNeedsFrame = { [weak self] in self?.kickRender() }
        _ = renderer.reloadTextureArraysIfNeeded()
        geoStepWet = Int((buildupWet * 24).rounded())
        renderer.rebuild(scene: project.scene, meshes: meshes)
        // v4.3: the player figure stands idle at the saved Play
        // start in edit mode.
        setEditPlayerFigure()
        if !renderer.rayTracingSupported {
            status = "This device does not support Metal ray tracing."
        }
        // v4.9: seed the footer's narrow observable.
        refreshFooter()
    }

    // MARK: Render-on-change plumbing

    /// Ask the viewport for one draw. The viewport chains further
    /// draws itself while `wantsFrame` stays true, so a converged,
    /// untouched scene costs nothing.
    func kickRender() {
        // v4.10 (WS1): while frozen, kicks collapse into a flag
        // — the wake path issues the one real kick.
        if isIdleFrozen { pendingKickWhileFrozen = true; return }
        viewport?.setNeedsDisplay()
    }

    func wantsFrame() -> Bool {
        // v4.10 (WS1): a frozen scene never wants a frame — the
        // viewport's chained-draw loop dies at the freeze.
        if isIdleFrozen { return false }
        return renderer.frameWanted || !pressedKeys.isEmpty || touchActive
            || project.scene.weather.mode != 0
            // v4.0: the node runtime asks for frames while it has
            // live work (tweens, timers, particles); a showing
            // banner also keeps frames coming so its expiry lands.
            || (mode == .play && runtime.needsFrame)
            || (mode == .play && shakeEndsAt > CACurrentMediaTime())
            || !playBanner.isEmpty
    }

    /// Camera input happened (orbit / pan / zoom / focus / mode
    /// change): the next frames are "moving" frames and the view
    /// needs a draw.
    func noteCameraActivity() {
        // v4.10 (WS1): camera activity IS user input — it
        // records the input time and wakes a frozen scene.
        noteUserInput()
        lastCameraActivity = CACurrentMediaTime()
        kickRender()
    }

    var scene: Scene { project.scene }
    var selectedObject: SceneObject? {
        selection.flatMap { id in project.scene.objects.first { $0.id == id } }
    }

    // MARK: Undo / redo

    private func snapshot() -> Data? {
        try? JSONEncoder().encode(Snapshot(objects: project.scene.objects))
    }

    /// Push the current state before a discrete edit.
    func beginEdit() {
        if let s = snapshot() {
            undoStack.append(s)
            if undoStack.count > 60 { undoStack.removeFirst() }
            redoStack.removeAll()
            syncUndoDepths()
        }
        editGestureAt = Date()
    }

    /// For continuous controls (colour picker, sliders without an
    /// editing-changed hook): one snapshot per gesture burst.
    func beginEditIfNewGesture() {
        if Date().timeIntervalSince(editGestureAt) > 1.5 { beginEdit() }
        editGestureAt = Date()
    }

    func undo() {
        noteUserInput()
        guard let last = undoStack.popLast() else { return }
        if let cur = snapshot() { redoStack.append(cur) }
        restore(last)
        syncUndoDepths()
    }

    func redo() {
        noteUserInput()
        guard let last = redoStack.popLast() else { return }
        if let cur = snapshot() { undoStack.append(cur) }
        restore(last)
        syncUndoDepths()
    }

    private func restore(_ data: Data) {
        guard let snap = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
        project.scene.objects = snap.objects
        noteSceneObjectsChanged()
        if let sel = selection, !snap.objects.contains(where: { $0.id == sel }) {
            selection = nil
        }
        // v4.6: prune the multi set to objects that exist in
        // the restored snapshot (undoing a group delete brings
        // the objects back as scene objects; the SET itself is
        // not part of the snapshot and stays cleared/pruned).
        if !multiSelection.isEmpty {
            // v4.9: assign only on an actual change — an
            // unpruned set must not republish.
            let pruned = multiSelection.filter { id in
                snap.objects.contains { $0.id == id }
            }
            if pruned != multiSelection { multiSelection = pruned }
            if multiSelection.count == 1 {
                let only = multiSelection.first
                if selection != only { selection = only }
            }
            if multiSelection.isEmpty && selection != nil,
               !snap.objects.contains(where: { $0.id == selection }) {
                selection = nil
            }
        }
        // v4.3: objects that (re)appear via undo/redo join the
        // exposure map at their snapshot value (absent = caught
        // up); entries for vanished objects are pruned.
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
    }

    private func syncUndoDepths() {
        // v4.9: change-only — an unchanged depth must not fire
        // objectWillChange (this runs on every beginEdit).
        if undoDepth != undoStack.count { undoDepth = undoStack.count }
        if redoDepth != redoStack.count { redoDepth = redoStack.count }
    }

    var canUndo: Bool { undoDepth > 0 }
    var canRedo: Bool { redoDepth > 0 }

    // MARK: Scene mutation plumbing

    /// Live mutation (no snapshot): rebuilds the renderer. Pair with
    /// beginEdit()/beginEditIfNewGesture() at gesture start.
    func mutateScene(_ change: (inout Scene) -> Void) {
        change(&project.scene)
        noteSceneObjectsChanged()
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
    }

    func objectBinding(_ id: UUID) -> Binding<SceneObject> {
        Binding(
            get: { [self] in
                project.scene.objects.first { $0.id == id }
                    ?? SceneObject(name: "?", source: "primitive")
            },
            set: { [self] newValue in
                guard let idx = project.scene.objects.firstIndex(where: { $0.id == id })
                else { return }
                let old = project.scene.objects[idx]
                if old == newValue { return }
                project.scene.objects[idx] = newValue
                noteSceneObjectsChanged()
                // A name-only edit needs no renderer work at all.
                var a = old, b = newValue
                a.name = ""; b.name = ""
                if a == b { return }
                // Geometry edits (transform/size/visibility) need
                // the full rebuild; material edits take the cheap
                // material-only path (no AS rebuild).
                let geometry = old.transform != newValue.transform
                    || old.params != newValue.params
                    || old.visible != newValue.visible
                    || old.source != newValue.source
                    || old.primitiveKind != newValue.primitiveKind
                    || old.meshFile != newValue.meshFile
                    // v4.0: area membership decides which objects a
                    // streamed (unloaded) area's static rebuild
                    // includes, so it is a geometry-class change.
                    || old.areaID != newValue.areaID
                scheduleRebuild(geometry: geometry)
            })
    }

    /// Coalesced renderer rebuild for continuous edits (sliders,
    /// colour pickers): at most one rebuild per ~60 ms, always with
    /// the latest scene, so dragging on a heavy import stays fluid.
    /// Geometry requests dominate material-only ones.
    private var rebuildScheduled = false
    private var rebuildNeedsGeometry = false
    func scheduleRebuild(geometry: Bool) {
        rebuildNeedsGeometry = rebuildNeedsGeometry || geometry
        guard !rebuildScheduled else { return }
        rebuildScheduled = true
        Task { [weak self] in
            try? await Task.sleep(nanoseconds: 60_000_000)
            guard let self else { return }
            self.rebuildScheduled = false
            let geo = self.rebuildNeedsGeometry
            self.rebuildNeedsGeometry = false
            self.renderer.rebuild(scene: self.project.scene, meshes: self.meshes,
                                  geometry: geo)
            self.kickRender()
        }
    }

    // MARK: Manipulation preview + viewport drag-move

    /// The object currently being dragged (viewport drag or
    /// transform slider), if any. While a drag is active the
    /// render is cheap: ALL stage light is cut (renderer.lightsCut
    /// zeroes every emissive flag, so nothing casts light) and the
    /// dragged object additionally gets the v2 preview treatment
    /// (a glass object renders opaque via previewObjectID).
    /// Everything restores in full when the drag ends.
    private var dragPreviewID: UUID?

    /// Push the current drag state into the renderer and refresh
    /// through the material-only path — no AS rebuild.
    private func applyDragRenderState() {
        renderer.previewObjectID = dragPreviewID
        renderer.lightsCut = (dragPreviewID != nil)
        if !renderer.updateMaterialsOnly(scene: project.scene, meshes: meshes) {
            renderer.rebuild(scene: project.scene, meshes: meshes)
        }
        kickRender()
    }

    /// Back-compat wrapper: sets/clears the drag render state.
    func setManipulationPreview(_ id: UUID?) {
        dragPreviewID = id
        applyDragRenderState()
    }

    /// Hook for transform sliders (position / rotation / scale /
    /// size): snapshot once at drag start, preview during, full
    /// material on release.
    func transformDragChanged(_ editing: Bool, objectID: UUID) {
        if editing {
            beginEdit()
            dragPreviewID = objectID
        } else {
            dragPreviewID = nil
        }
        applyDragRenderState()
    }

    private var dragState: (id: UUID, planeY: Float, grabOffset: SIMD2<Float>)?
    /// v4.6: group-drag bookkeeping — every member's start XZ,
    /// captured at drag start; members move by the grabbed
    /// object's delta (each keeps its own height — only XZ is
    /// ever written, exactly as in a single drag).
    private var groupDragMembers: [(id: UUID, start: SIMD2<Double>)] = []
    private var groupDragPrimaryStart: SIMD2<Double>?
    var isDraggingObject: Bool { dragState != nil }

    /// Start dragging the given object across the horizontal plane
    /// through its current height. Returns false when the ray
    /// can't reach that plane (camera level with it). v4.6: when
    /// the grabbed object is a member of a multi set (count > 1),
    /// the drag moves the WHOLE set together — one beginEdit, so
    /// the group move is a single undo step, and every member
    /// stays its own snow intruder (a group drag plows per-object).
    func beginObjectDrag(id: UUID, ndc: SIMD2<Float>, aspect: Float) -> Bool {
        guard mode == .edit, let obj = project.scene.object(id: id) else { return false }
        let (origin, dir) = rayFor(ndc: ndc, aspect: aspect)
        let planeY = Float(obj.transform.position[1])
        guard abs(dir.y) > 1e-4 else { return false }
        let t = (planeY - origin.y) / dir.y
        guard t > 0 else { return false }
        let grab = origin + dir * t
        beginEdit()
        dragPreviewID = id
        applyDragRenderState()
        dragState = (id, planeY, SIMD2<Float>(Float(obj.transform.position[0]) - grab.x,
                                              Float(obj.transform.position[2]) - grab.z))
        groupDragMembers = []
        groupDragPrimaryStart = nil
        if multiSelectMode && multiSelection.count > 1 && multiSelection.contains(id) {
            groupDragPrimaryStart = SIMD2<Double>(obj.transform.position[0],
                                                  obj.transform.position[2])
            groupDragMembers = project.scene.objects
                .filter { multiSelection.contains($0.id) && $0.id != id }
                .map { (id: $0.id, start: SIMD2<Double>($0.transform.position[0],
                                                        $0.transform.position[2])) }
        }
        return true
    }

    func dragObjectTo(ndc: SIMD2<Float>, aspect: Float) {
        guard let ds = dragState,
              let idx = project.scene.objects.firstIndex(where: { $0.id == ds.id })
        else { return }
        let (origin, dir) = rayFor(ndc: ndc, aspect: aspect)
        guard abs(dir.y) > 1e-4 else { return }
        let t = (ds.planeY - origin.y) / dir.y
        guard t > 0 else { return }
        let grab = origin + dir * t
        project.scene.objects[idx].transform.position[0] = Double(grab.x + ds.grabOffset.x)
        project.scene.objects[idx].transform.position[2] = Double(grab.z + ds.grabOffset.y)
        // v4.6: group members follow the grabbed object's delta.
        if let start = groupDragPrimaryStart {
            let delta = SIMD2<Double>(
                project.scene.objects[idx].transform.position[0] - start.x,
                project.scene.objects[idx].transform.position[2] - start.y)
            for member in groupDragMembers {
                if let mi = project.scene.objects.firstIndex(where: { $0.id == member.id }) {
                    project.scene.objects[mi].transform.position[0] = member.start.x + delta.x
                    project.scene.objects[mi].transform.position[2] = member.start.y + delta.y
                }
            }
        }
        noteCameraActivity()
        scheduleRebuild(geometry: true)
    }

    func endObjectDrag() {
        guard dragState != nil else { return }
        dragState = nil
        groupDragMembers = []
        groupDragPrimaryStart = nil
        dragPreviewID = nil
        applyDragRenderState()
    }

    // MARK: Editing operations

    func addPrimitive(_ kind: String) {
        beginEdit()
        var obj = SceneObject.primitive(kind, name: uniqueName(for: kind.capitalized))
        // Place new objects in front of the orbit target, on the floor.
        let t = project.scene.camera.target
        obj.transform.position = [t[0], kind == "plane" ? 0 : defaultY(for: obj), t[2]]
        project.scene.objects.append(obj)
        noteSceneObjectsChanged()
        selection = obj.id
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
        status = "Added \(obj.name)."
    }

    private func defaultY(for obj: SceneObject) -> Double {
        switch obj.primitiveKind {
        case "sphere": return obj.params["r"] ?? 0.5
        case "cylinder": return (obj.params["h"] ?? 1) / 2
        case "box": return (obj.params["h"] ?? 1) / 2
        default: return 0
        }
    }

    private func uniqueName(for base: String) -> String {
        let names = Set(project.scene.objects.map(\.name))
        if !names.contains(base) { return base }
        var n = 2
        while names.contains("\(base) \(n)") { n += 1 }
        return "\(base) \(n)"
    }

    func deleteSelected() {
        // v4.6: with a multi set, Delete removes the whole set
        // (one undo step) — the keyboard and the properties
        // panel both land here.
        if multiSelectMode && multiSelection.count > 1 {
            deleteMultiSelection()
            return
        }
        guard let sel = selection,
              let idx = project.scene.objects.firstIndex(where: { $0.id == sel }) else { return }
        beginEdit()
        let name = project.scene.objects[idx].name
        let removedID = project.scene.objects[idx].id
        project.scene.objects.remove(at: idx)
        noteSceneObjectsChanged()
        selection = nil
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
        status = "Deleted \(name)."
    }

    func toggleVisibility(_ id: UUID) {
        beginEdit()
        mutateScene { scene in
            if let i = scene.objects.firstIndex(where: { $0.id == id }) {
                scene.objects[i].visible.toggle()
            }
        }
    }

    // MARK: Multi-select (v4.6 workstream 2)

    /// Push the effective selection highlight into the
    /// renderer (material-path tint; no rebuild, no AS work).
    /// The highlighted set is the multi set while the tool is
    /// on, otherwise the single selection — so the object you
    /// picked is marked in the 3D view in both modes.
    private func syncHighlightToRenderer() {
        let ids: Set<UUID>
        if multiSelectMode {
            ids = multiSelection
        } else if let sel = selection {
            ids = [sel]
        } else {
            ids = []
        }
        if renderer.highlightObjectIDs != ids {
            renderer.highlightObjectIDs = ids
            if !renderer.updateMaterialsOnly(scene: project.scene, meshes: meshes) {
                renderer.rebuild(scene: project.scene, meshes: meshes)
            }
            kickRender()
        }
    }

    /// One tap in Multi Select mode (viewport or tree): add
    /// the object to the set, or remove it if it's already in.
    /// The tapped object becomes the primary selection when
    /// added; removing the primary promotes another member.
    func toggleMultiSelection(_ id: UUID) {
        guard project.scene.objects.contains(where: { $0.id == id }) else { return }
        if multiSelection.contains(id) {
            multiSelection.remove(id)
            if selection == id {
                selection = multiSelection.first
            }
            status = multiSelection.isEmpty
                ? "Multi Select: nothing selected."
                : "Multi Select: \(multiSelection.count) object\(multiSelection.count == 1 ? "" : "s") selected."
        } else {
            multiSelection.insert(id)
            selection = id
            if let name = project.scene.objects.first(where: { $0.id == id })?.name {
                status = "Multi Select: \(name) added (\(multiSelection.count) selected)."
            }
        }
    }

    func clearMultiSelection() {
        // v4.9: change-only — clearing an already-empty set
        // (e.g. a tap on empty space with nothing selected)
        // must not republish either property.
        if !multiSelection.isEmpty { multiSelection = [] }
        if selection != nil { selection = nil }
    }

    /// Delete the whole set as ONE undo step.
    func deleteMultiSelection() {
        let ids = multiSelection
        guard ids.count > 1 else { deleteSelected(); return }
        beginEdit()
        project.scene.objects.removeAll { ids.contains($0.id) }
        noteSceneObjectsChanged()
        multiSelection = []
        selection = nil
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
        status = "Deleted \(ids.count) objects."
    }

    /// Group-safe visibility: hide all members, or show all
    /// if every member is hidden.
    func toggleVisibilityForMultiSelection() {
        let ids = multiSelection
        guard ids.count > 1 else { return }
        let anyVisible = project.scene.objects.contains { ids.contains($0.id) && $0.visible }
        beginEdit()
        mutateScene { scene in
            for i in scene.objects.indices where ids.contains(scene.objects[i].id) {
                scene.objects[i].visible = !anyVisible
            }
        }
        status = anyVisible ? "Hid \(ids.count) objects." : "Showed \(ids.count) objects."
    }

    /// Can a viewport drag starting on `id` move something?
    /// (The single selection, or any member of the multi set.)
    func dragSetContains(_ id: UUID) -> Bool {
        if multiSelectMode && multiSelection.count > 1 {
            return multiSelection.contains(id)
        }
        return selection == id
    }

    // MARK: Model import (STL / OBJ / USD via ModelIO)

    func importModelFile(_ url: URL) {
        status = "Importing \(url.lastPathComponent)…"
        let projectID = project.id
        // Keep the original file inside the project first.
        let assetName = ProjectStore.copyAsset(projectID: projectID, from: url)
        let assetPath = assetName.map { ProjectStore.assetURL(projectID: projectID, fileName: $0).path } ?? url.path
        Task { [weak self] in
            let result = await Task.detached(priority: .userInitiated) {
                Result { try ModelImporter.importFile(path: assetPath) }
            }.value
            guard let self else { return }
            switch result {
            case .failure(let error):
                // Don't leave the copied original behind.
                if let assetName {
                    ProjectStore.removeAsset(projectID: projectID, fileName: assetName)
                }
                self.status = "Import failed: \(error.localizedDescription)"
            case .success(let parts):
                guard !parts.isEmpty else {
                    if let assetName {
                        ProjectStore.removeAsset(projectID: projectID, fileName: assetName)
                    }
                    self.status = "Import failed: the file contained no geometry."
                    return
                }
                self.beginEdit()
                var firstID: UUID?
                for part in parts {
                    var mesh = MeshData()
                    mesh.positions = part.positions
                    mesh.normals = part.normals
                    mesh.indices = part.indices
                    mesh.uvs = [Float](repeating: 0, count: part.positions.count / 3 * 2)
                    mesh.colors = [Float](repeating: 1, count: part.positions.count)
                    // Bake the file's material colour into the
                    // vertices; per-vertex colours multiply on top.
                    if part.hasColor {
                        for v in 0..<mesh.vertexCount {
                            mesh.colors[3 * v] = Float(part.red)
                            mesh.colors[3 * v + 1] = Float(part.green)
                            mesh.colors[3 * v + 2] = Float(part.blue)
                        }
                    }
                    if let vc = part.vertexColors, vc.count == mesh.colors.count {
                        for i in 0..<mesh.colors.count {
                            mesh.colors[i] *= vc[i]
                        }
                    }
                    var obj = SceneObject(name: self.uniqueName(for: part.name), source: "model")
                    obj.material.triplanar = true
                    let file = ProjectStore.writeMesh(projectID: projectID, objectID: obj.id, mesh: mesh)
                    obj.meshFile = file
                    obj.assetFile = assetName
                    // Rest the part on the floor (min Y -> 0) and centre
                    // it near the orbit target.
                    let b = mesh.bounds
                    let cx = Double((b.min.x + b.max.x) / 2)
                    let cz = Double((b.min.z + b.max.z) / 2)
                    let t = self.project.scene.camera.target
                    obj.transform.position = [t[0] - cx, -Double(b.min.y), t[2] - cz]
                    self.meshes[obj.id] = mesh
                    self.project.scene.objects.append(obj)
                    if firstID == nil { firstID = obj.id }
                }
                self.noteSceneObjectsChanged()
                self.selection = firstID
                self.renderer.rebuild(scene: self.project.scene, meshes: self.meshes)
                self.kickRender()
                self.status = "Imported \(parts.count) part\(parts.count == 1 ? "" : "s") from \(url.lastPathComponent)."
            }
        }
    }

    // MARK: Play mode

    func toggleMode() {
        noteUserInput()
        pressedKeys.removeAll() // a held key must not survive the switch
        touchVector = .zero // a stuck joystick must not survive either
        // v4.6: multi-select is an edit-mode tool — entering
        // Play switches tools, which clears the set.
        if multiSelectMode { multiSelectMode = false }
        if mode == .edit {
            // Entering Play. FIRST take the restore snapshot —
            // before the runtime touches anything — then clear
            // all promotion/streaming state so the session starts
            // from the scene exactly as authored.
            playSnapshot = snapshot()
            playWeatherMode = project.scene.weather.mode
            // v4.1 (F15/§4.4): snapshot the weather build-up too.
            playBuildupWet = buildupWet
            playGeoStepWet = geoStepWet
            // v4.1 (Appendix B): every new session state resets
            // on Play entry.
            playSpeed = 2.4
            shakeAmplitude = 0
            shakeEndsAt = 0
            shakeDuration = 0
            weatherRateScale = 1.0
            weatherIntensityScale = 1.0
            renderer.envIntensity = 1
            clones.removeAll()
            cloneEntries.removeAll()
            promotionRebuildPending = false
            promotedIDs.removeAll()
            hiddenPromotedIDs.removeAll()
            areaHiddenPromotedIDs.removeAll()
            dynamicEntries.removeAll()
            runtimeUnloaded.removeAll()
            if !playBanner.isEmpty { playBanner = "" }
            bannerExpiry = nil
            figureHeading = Float(project.scene.camera.playYaw) * .pi / 180
            figureWalkPhase = 0
            figureWalkBlend = 0
            figureLastPlayPos = nil
            figureLastSentPhase = -.infinity
            figureLastSentBlend = -.infinity
            figureLastMatrix = nil
            editFigureSet = false
            mode = .play
            runtime.start()
        } else {
            // Exiting Play: stop the runtime, clear ALL dynamic
            // renderer state (entries, particles, area streaming),
            // then restore the pre-Play snapshot + weather mode.
            runtime.stop()
            renderer.setDynamicEntries([])
            dynamicEntries.removeAll()
            // v4.1 (§7.4): clones die with the Play session; the
            // setDynamicEntries([]) above already removed their
            // renderer entries.
            clones.removeAll()
            cloneEntries.removeAll()
            renderer.setParticleSoups([])
            renderer.unloadedAreaIDs = []
            runtimeUnloaded.removeAll()
            // v4.1 (F15/§4.4): restore the weather build-up BEFORE
            // the snapshot restore, so its rebuild bakes the
            // restored geometry levels, and reset every new
            // session state (Appendix B: resets in BOTH branches).
            buildupWet = playBuildupWet
            project.scene.weather.wetness = buildupWet
            weatherRateScale = 1.0
            weatherIntensityScale = 1.0
            playSpeed = 2.4
            shakeAmplitude = 0
            shakeEndsAt = 0
            shakeDuration = 0
            renderer.envIntensity = 1
            promotionRebuildPending = false
            restorePlaySnapshot()
            if project.scene.weather.mode != playWeatherMode {
                setWeatherMode(playWeatherMode)
            }
            promotedIDs.removeAll()
            hiddenPromotedIDs.removeAll()
            areaHiddenPromotedIDs.removeAll()
            if !playBanner.isEmpty { playBanner = "" }
            bannerExpiry = nil
            mode = .edit
            // v4.3: back in edit mode the figure returns to its
            // idle pose at the (possibly moved) Play start.
            setEditPlayerFigure()
        }
        renderer.resetAccumulation()
        noteCameraActivity()
        status = mode == .play ? "Play mode — walk around. Pause to edit."
                                : "Edit mode."
        // v4.9: Play entry/exit restores weather state — the
        // footer's displayed lines follow it here too.
        refreshFooter()
    }

    /// Undo-free restore of the Play-entry object snapshot. See
    /// `playSnapshot`: unlike restore() this does not touch the
    /// undo/redo stacks or the selection — the selection the user
    /// had when they pressed Play is still valid on return, and
    /// the undo history must read as if Play never happened.
    private func restorePlaySnapshot() {
        guard let data = playSnapshot else { return }
        playSnapshot = nil
        guard let snap = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
        project.scene.objects = snap.objects
        noteSceneObjectsChanged()
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
    }

    // MARK: v4.0 Runtime funnels (called by GameRuntime only)

    /// The object's local (unbaked) mesh: primitives regenerate
    /// from their params; imported models use the loaded mesh.
    private func localMesh(for obj: SceneObject) -> MeshData? {
        if obj.source == "primitive" {
            return MeshGen.primitive(kind: obj.primitiveKind, params: obj.params)
        }
        return meshes[obj.id]
    }

    /// Bring one object under runtime control (see the promotion
    /// note on `promotedIDs`). The dynamic entry is built from the
    /// object's CURRENT transform/material, so promotion itself
    /// never visibly changes the object; the static rebuild that
    /// drops it from the baked soup is DEFERRED (v4.1 F29) to
    /// frameTick's single coalesced rebuild after runtime.tick.
    private func promoteObject(id: UUID) {
        guard mode == .play, !promotedIDs.contains(id),
              let idx = project.scene.objects.firstIndex(where: { $0.id == id })
        else { return }
        let obj = project.scene.objects[idx]
        guard let mesh = localMesh(for: obj), !mesh.isEmpty else { return }
        promotedIDs.insert(id)
        if !obj.visible {
            // Already hidden as authored: no entry, and it counts
            // as runtime-hidden so a later show can add the entry.
            hiddenPromotedIDs.insert(id)
        }
        // The promotion mechanism: from here the static layer must
        // never draw this object again this Play session.
        project.scene.objects[idx].visible = false
        noteSceneObjectsChanged()
        syncDynamicEntry(id: id)
        // v4.1 (F29): do NOT rebuild here. A Play-start chain that
        // touches N objects in one tick would otherwise pay N full
        // static rebuilds. Defer: frameTick performs ONE rebuild
        // after runtime.tick (before the draw) when this flag is
        // set, and calls runtime.collisionDidChange() with it —
        // which also closes the promotion collision-ghost gap
        // (recon §6.6). Callers must not assume the static rebuild
        // has already happened when this returns; the dynamic
        // entry above already carries the object for this frame.
        promotionRebuildPending = true
        kickRender()
    }

    /// Reconcile one promoted object's dynamic entry with its
    /// tracked visibility: entry present iff promoted, not
    /// runtime-hidden, and not parked by an unloaded area.
    private func syncDynamicEntry(id: UUID) {
        guard promotedIDs.contains(id),
              let obj = project.scene.object(id: id) else { return }
        let shouldShow = !hiddenPromotedIDs.contains(id)
            && !areaHiddenPromotedIDs.contains(id)
        let hasEntry = dynamicEntries.contains { $0.key == id }
        var structureChanged = false
        if shouldShow, !hasEntry, let mesh = localMesh(for: obj), !mesh.isEmpty {
            dynamicEntries.append(DynamicEntry(
                key: id, mesh: mesh, matrix: obj.transform.matrix,
                material: renderer.gpuMaterialForObject(obj), objectID: id))
            // v4.1 (§7.1): the renderer's entry list is ALWAYS
            // promoted entries (in order) + clone entries.
            renderer.setDynamicEntries(dynamicEntries + cloneEntries)
            structureChanged = true
        } else if !shouldShow, hasEntry {
            dynamicEntries.removeAll { $0.key == id }
            renderer.removeDynamicEntry(key: id)
            structureChanged = true
        }
        // v4.13: the runtime's collision grid must follow entry
        // structure exactly as it follows the legacy static path
        // (F29) and the area funnels — without this, a hidden
        // promoted object kept colliding (an Applied Gravity
        // object resting on it never fell when it was hidden:
        // gravstack refell). Entry add/remove changes the
        // renderer's collision triangle set, so notify.
        if structureChanged { runtime.collisionDidChange() }
        kickRender()
    }

    /// Runtime transform write. NOTE on the @Published cost: the
    /// scene write goes through the published `project`, so during
    /// an active tween/rotation this invalidates the Play overlay
    /// per frame — the v3.1 menu-storm pattern. It is accepted
    /// here as a BOUNDED exception: it only happens while a
    /// runtime animation is actually running, only in Play (the
    /// editor body is not shown; the overlay is a pause button, a
    /// joystick and a banner), and the write keeps the scene
    /// truthful — what actually renders is the renderer's dynamic
    /// matrix below. mutateScene/scheduleRebuild are deliberately
    /// NOT used: a static rebuild would fight the promotion.
    func runtimeSetTransform(_ id: UUID, _ t: Transform3) {
        guard let idx = project.scene.objects.firstIndex(where: { $0.id == id })
        else { return }
        if !promotedIDs.contains(id) { promoteObject(id: id) }
        // v4.9: change-only — a tween/pose funnel can re-send
        // the transform the object already has (held end pose,
        // zero-length step); the published write, the static
        // rebuild and the dynamic-matrix push are all no-ops
        // then, so skip them. The dynamic entry (when promoted)
        // was built from this same transform, so it already
        // carries t.matrix.
        if project.scene.objects[idx].transform == t {
            kickRender()
            return
        }
        // v4.1 (F19): promotion can FAIL (no/empty local mesh) —
        // then there is no dynamic entry to move, and writing only
        // the scene value would be silently invisible until some
        // unrelated rebuild. Fall back to the static path instead.
        guard promotedIDs.contains(id) else {
            project.scene.objects[idx].transform = t
            renderer.rebuild(scene: project.scene, meshes: meshes)
            runtime.collisionDidChange()
            renderer.resetAccumulation()
            kickRender()
            return
        }
        project.scene.objects[idx].transform = t
        if let ei = dynamicEntries.firstIndex(where: { $0.key == id }) {
            dynamicEntries[ei].matrix = t.matrix
        }
        renderer.updateDynamicMatrix(key: id, matrix: t.matrix)
        kickRender()
    }

    func runtimeSetMaterial(_ id: UUID, _ m: MaterialProps) {
        guard let idx = project.scene.objects.firstIndex(where: { $0.id == id })
        else { return }
        if !promotedIDs.contains(id) { promoteObject(id: id) }
        // v4.9: change-only, same discipline as transforms —
        // re-sending the current material skips the published
        // write and the GPU-material rebuild below.
        if project.scene.objects[idx].material == m {
            kickRender()
            return
        }
        // v4.1 (F19): same unpromotable fallback as transforms.
        guard promotedIDs.contains(id) else {
            project.scene.objects[idx].material = m
            renderer.rebuild(scene: project.scene, meshes: meshes)
            renderer.resetAccumulation()
            kickRender()
            return
        }
        project.scene.objects[idx].material = m
        let obj = project.scene.objects[idx]
        let gpu = renderer.gpuMaterialForObject(obj)
        if let ei = dynamicEntries.firstIndex(where: { $0.key == id }) {
            dynamicEntries[ei].material = gpu
        }
        renderer.setDynamicMaterial(key: id, material: gpu)
        kickRender()
    }

    func runtimeSetVisible(_ id: UUID, _ visible: Bool) {
        guard let idx = project.scene.objects.firstIndex(where: { $0.id == id })
        else { return }
        // v4.1 (F28a): on the FIRST runtime visibility change of a
        // non-promoted object, promote it — from then on flips are
        // cheap entry add/removes instead of a full static rebuild
        // per flip. Promotion hides the object from the static
        // layer and the entry logic below carries its visibility:
        // a hide-first change ends as hiddenPromoted with no entry
        // (promoteObject adds the entry, then the hide removes it).
        if !promotedIDs.contains(id) { promoteObject(id: id) }
        if promotedIDs.contains(id) {
            // Promoted: visibility is entry presence. The scene
            // flag stays false (promotion mechanism); the tracked
            // sets carry the truth.
            if visible { hiddenPromotedIDs.remove(id) } else { hiddenPromotedIDs.insert(id) }
            syncDynamicEntry(id: id)
        } else {
            // Promotion failed (F19-class object: no local mesh):
            // legacy static path — write the flag and rebuild.
            // v4.1 (F29): the collision grid must follow, as the
            // area funnels already do.
            // v4.9: change-only — an unchanged flag needs no
            // published write, rebuild or collision rebuild.
            if project.scene.objects[idx].visible != visible {
                project.scene.objects[idx].visible = visible
                noteSceneObjectsChanged()
                renderer.rebuild(scene: project.scene, meshes: meshes)
                runtime.collisionDidChange()
            }
        }
        kickRender()
    }

    /// Particle soups arrive with a material SOURCE object; the
    /// renderer wants the baked GPU material for it.
    func runtimeSetParticleSoups(_ soups: [(mesh: MeshData, materialObject: SceneObject)]) {
        renderer.setParticleSoups(soups.map {
            (mesh: $0.mesh, material: renderer.gpuMaterialForObject($0.materialObject))
        })
        kickRender()
    }

    func runtimeLoadArea(_ id: UUID) {
        // v4.1 (F28b): no membership change → no rebuild, no
        // collision rebuild. (Loading an already-loaded area was
        // a full static rebuild on every call.)
        guard runtimeUnloaded.contains(id) else { return }
        runtimeUnloaded.remove(id)
        renderer.unloadedAreaIDs = runtimeUnloaded
        // Promoted members come back too — if they were visible.
        for obj in project.scene.objects where obj.areaID == id && promotedIDs.contains(obj.id) {
            areaHiddenPromotedIDs.remove(obj.id)
            syncDynamicEntry(id: obj.id)
        }
        // Static rebuild: non-promoted members rejoin the soup
        // (promoted members stay excluded — visible == false).
        renderer.rebuild(scene: project.scene, meshes: meshes)
        runtime.collisionDidChange()
        kickRender()
    }

    func runtimeUnloadArea(_ id: UUID) {
        // v4.1 (F28b): no membership change → no work (above).
        guard !runtimeUnloaded.contains(id) else { return }
        runtimeUnloaded.insert(id)
        renderer.unloadedAreaIDs = runtimeUnloaded
        // Hide promoted members' dynamic entries for as long as
        // the area is out; loading restores them if visible.
        for obj in project.scene.objects where obj.areaID == id && promotedIDs.contains(obj.id) {
            areaHiddenPromotedIDs.insert(obj.id)
            syncDynamicEntry(id: obj.id)
        }
        renderer.rebuild(scene: project.scene, meshes: meshes)
        runtime.collisionDidChange()
        kickRender()
    }

    /// Areas currently streamed IN: every authored area minus the
    /// runtime-unloaded set.
    var runtimeLoadedAreaIDs: Set<UUID> {
        Set(project.scene.areas.map(\.id)).subtracting(runtimeUnloaded)
    }

    func runtimeShowMessage(_ text: String, seconds: Double) {
        // v4.9: change-only published write — re-showing the
        // same message still extends its expiry below, it just
        // doesn't republish an identical string.
        if playBanner != text { playBanner = text }
        bannerExpiry = Date().addingTimeInterval(seconds)
        kickRender()
    }

    /// v4.1 (F16): the teleport eye rule, made CONTINUOUS. The old
    /// rule (eye = y + 1.7 below y = 1, eye = y at/above it) had a
    /// 1.69 m cliff at y = 1.0 and put the eye below standing
    /// height for targets between 1.0 and ~1.7. Now: below 1.0 the
    /// target is a floor point (eye = y + 1.7); from 1.0 to 2.7
    /// the added offset blends linearly down from 1.7 to 0; at or
    /// above 2.7 the target IS the eye. Shared by teleport + pose.
    private func teleportEyeY(forTargetY y: Double) -> Double {
        if y < 1.0 { return y + 1.7 }
        if y < 2.7 { return y + (2.7 - y) / 1.7 * 1.7 }
        return y
    }

    /// Teleport the Play camera. The target is an object's origin:
    /// a floor-level marker means "stand here", so the eye goes
    /// above it per the (continuous) eye rule in teleportEyeY.
    func runtimeTeleportPlayer(to t: SIMD3<Float>) {
        let eyeY = teleportEyeY(forTargetY: Double(t.y))
        // v4.9: change-only — teleporting to the spot the
        // player already occupies writes nothing.
        let target: [Double] = [Double(t.x), eyeY, Double(t.z)]
        if project.scene.camera.playPosition != target {
            project.scene.camera.playPosition = target
        }
        noteCameraActivity()
    }

    /// v4.1 (§3.4): set the Play movement speed, absolute m/s,
    /// clamped 0...20. Session-only; Play entry resets to 2.4.
    func runtimeSetPlayerSpeed(_ metresPerSecond: Double) {
        playSpeed = Float(min(20, max(0, metresPerSecond)))
        kickRender()
    }

    /// v4.1 (§3.5): full Play-camera pose set — position is an
    /// eye position, used verbatim (no teleport eye rule), plus
    /// yaw/pitch in degrees (pitch clamped ±85, matching look
    /// input).
    func runtimeSetPlayerPose(position: SIMD3<Float>, yaw: Double, pitch: Double) {
        // v4.9: per-field change-only writes — a per-frame pose
        // funnel (camera-follow nodes) re-sends held components
        // constantly; only a component that actually moved
        // writes through the published project.
        let pos: [Double] = [Double(position.x), Double(position.y), Double(position.z)]
        if project.scene.camera.playPosition != pos {
            project.scene.camera.playPosition = pos
        }
        if project.scene.camera.playYaw != yaw {
            project.scene.camera.playYaw = yaw
        }
        let clampedPitch = min(85, max(-85, pitch))
        if project.scene.camera.playPitch != clampedPitch {
            project.scene.camera.playPitch = clampedPitch
        }
        noteCameraActivity()
    }

    /// v4.1 (§3.5): camera shake. Session state only — frameTick
    /// perturbs the computed basis while the shake is active,
    /// decaying by remaining/duration; persisted camera state is
    /// never written. A non-positive duration clears the shake.
    func runtimeCameraShake(amplitude: Double, seconds: Double) {
        let clampedAmplitude = min(2, max(0, amplitude))
        let clampedSeconds = min(10, max(0.1, seconds))
        if seconds <= 0 || amplitude <= 0 {
            shakeAmplitude = 0
            shakeEndsAt = 0
            shakeDuration = 0
        } else {
            shakeAmplitude = Float(clampedAmplitude)
            shakeDuration = Float(clampedSeconds)
            shakeEndsAt = CACurrentMediaTime() + clampedSeconds
        }
        kickRender()
    }

    func runtimeSetWeather(mode weatherMode: Int, clear: Bool) {
        setWeatherMode(weatherMode)
        if clear { clearWeatherBuildup() }
    }

    /// v4.1 (§4.4, PINNED): Weather Control. Nil channels are left
    /// untouched. Scales are session-only (reset on Play entry);
    /// direct level sets write the accumulators AND the published
    /// values (a sync point, like clearWeatherBuildup), quantised
    /// geometry steps + renderer levels, then rebuild + reset.
    func runtimeSetWeatherControl(rateScale: Double?, intensityScale: Double?,
                                  wet: Double?, snow: Double?) {
        if let rateScale { weatherRateScale = min(10, max(0, rateScale)) }
        if let intensityScale { weatherIntensityScale = min(4, max(0, intensityScale)) }
        // v4.13: the snow channel is retired — old graphs may
        // still pass it; it is accepted and ignored. A wet set
        // writes the accumulator + published value and resets
        // accumulation (wetness is a shading uniform now; no
        // geometry rebuild follows it).
        if let wet {
            buildupWet = min(1, max(0, wet))
            if project.scene.weather.wetness != buildupWet {
                project.scene.weather.wetness = buildupWet
            }
            renderer.weatherWet = Float(buildupWet)
            renderer.resetAccumulation()
        }
        kickRender()
    }

    /// v4.1 (§5): environment (sky) light intensity, 0...4.
    /// A per-frame uniform — no rebuild, but accumulated history
    /// is stale the moment it changes, so reset accumulation.
    func runtimeSetEnvironmentIntensity(_ value: Double) {
        renderer.envIntensity = Float(min(4, max(0, value)))
        renderer.resetAccumulation()
        kickRender()
    }

    /// v4.1: would this object render right now? For promoted ids
    /// the entry-presence truth applies (promoted ∧ not
    /// runtime-hidden ∧ not area-parked — syncDynamicEntry keeps
    /// entry presence exactly equivalent to that); for everything
    /// else, the scene visible flag ∧ its area (if any) is loaded.
    func runtimeIsObjectVisible(_ id: UUID) -> Bool {
        guard let obj = project.scene.object(id: id) else { return false }
        if promotedIDs.contains(id) {
            return !hiddenPromotedIDs.contains(id)
                && !areaHiddenPromotedIDs.contains(id)
        }
        if !obj.visible { return false }
        if let areaID = obj.areaID, runtimeUnloaded.contains(areaID) { return false }
        return true
    }

    // MARK: v4.1 Clone spawn (§7, PINNED)

    /// Batch spawn. Each request: source object id, world
    /// transform, optional material override (nil = the source's
    /// CURRENT material). Returns the clone ids actually spawned
    /// (admission control may shorten the list; order preserved).
    /// ONE setDynamicEntries call per batch. Clone entries always
    /// carry objectID = nil (§6.5: unpickable by design — a tap on
    /// a clone must never fire the source's tap nodes).
    @discardableResult
    func runtimeSpawnClones(_ requests: [(sourceID: UUID, transform: Transform3,
                                          material: MaterialProps?)]) -> [UUID] {
        guard mode == .play else { return [] }
        // Admission caps (§7.3): the renderer's 90k dynamic budget
        // truncates silently and order-dependently, so the session
        // admits conservatively and in request order.
        let maxClones = 64
        let maxCloneTris = 45_000
        let maxSourceTris = 8_000
        var spawned: [UUID] = []
        var newClones: [SpawnedClone] = []
        var newEntries: [DynamicEntry] = []
        var triTotal = liveCloneTriCount
        for request in requests {
            if clones.count + newClones.count >= maxClones { break }
            guard let source = project.scene.object(id: request.sourceID),
                  let mesh = localMesh(for: source), !mesh.isEmpty
            else { continue }
            let tris = mesh.triangleCount
            if tris > maxSourceTris { continue }   // this request only is skipped
            if triTotal + tris > maxCloneTris { break }   // remaining requests dropped
            let cloneID = UUID()
            let snapshotMaterial = request.material ?? source.material
            var temp = source
            temp.id = cloneID
            temp.transform = request.transform
            temp.material = snapshotMaterial
            newClones.append(SpawnedClone(cloneID: cloneID, sourceID: request.sourceID,
                                          transform: request.transform,
                                          material: snapshotMaterial, triCount: tris))
            newEntries.append(DynamicEntry(
                key: cloneID, mesh: mesh, matrix: request.transform.matrix,
                material: renderer.gpuMaterialForObject(temp), objectID: nil))
            spawned.append(cloneID)
            triTotal += tris
        }
        if !spawned.isEmpty {
            clones.append(contentsOf: newClones)
            cloneEntries.append(contentsOf: newEntries)
            renderer.setDynamicEntries(dynamicEntries + cloneEntries)
            kickRender()
        }
        return spawned
    }

    /// Despawn every live clone whose sourceID matches. No-op (no
    /// renderer call) when nothing matches.
    func runtimeDespawnClones(sourceID: UUID) {
        let removed = Set(clones.filter { $0.sourceID == sourceID }.map(\.cloneID))
        guard !removed.isEmpty else { return }
        clones.removeAll { removed.contains($0.cloneID) }
        cloneEntries.removeAll { removed.contains($0.key) }
        renderer.setDynamicEntries(dynamicEntries + cloneEntries)
        kickRender()
    }

    /// Despawn one clone by its clone id (runtime expiry path).
    func runtimeDespawnClone(_ cloneID: UUID) {
        guard clones.contains(where: { $0.cloneID == cloneID }) else { return }
        clones.removeAll { $0.cloneID == cloneID }
        cloneEntries.removeAll { $0.key == cloneID }
        renderer.setDynamicEntries(dynamicEntries + cloneEntries)
        kickRender()
    }

    // MARK: v4.0 Areas & graph editing (views call these)

    /// Add a trigger area centred on the orbit target. Areas and
    /// the node graph are NOT undoable (same precedent as weather:
    /// undo snapshots cover scene objects only).
    func addArea() {
        let t = project.scene.camera.target
        let area = TriggerArea(id: UUID(), name: uniqueAreaName(),
                               center: [t[0], 1.0, t[2]], size: [4, 3, 4])
        project.scene.areas.append(area)
        selectedArea = area.id
        status = "Added trigger area \(area.name)."
        kickRender()
    }

    private func uniqueAreaName() -> String {
        let names = Set(project.scene.areas.map(\.name))
        if !names.contains("Area") { return "Area" }
        var n = 2
        while names.contains("Area \(n)") { n += 1 }
        return "Area \(n)"
    }

    func deleteArea(_ id: UUID) {
        project.scene.areas.removeAll { $0.id == id }
        if selectedArea == id { selectedArea = nil }
        runtimeUnloaded.remove(id)
        // Objects keep no dangling membership; membership feeds
        // streaming rebuilds, so un-assigning is a geometry change.
        var unassigned = false
        for i in project.scene.objects.indices where project.scene.objects[i].areaID == id {
            project.scene.objects[i].areaID = nil
            unassigned = true
        }
        if unassigned {
            noteSceneObjectsChanged()
            renderer.rebuild(scene: project.scene, meshes: meshes)
        }
        status = "Deleted trigger area."
        kickRender()
    }

    func updateArea(_ area: TriggerArea) {
        guard let idx = project.scene.areas.firstIndex(where: { $0.id == area.id })
        else { return }
        project.scene.areas[idx] = area
        // Areas are not part of the baked soup — the debug boxes
        // refreshed in frameTick are their only visuals, so no
        // rebuild is needed, just a frame.
        kickRender()
    }

    /// Node-graph edits land here: a plain assignment, saved with
    /// the project on Save, not covered by Undo.
    func updateGraph(_ graph: NodeGraph) {
        project.scene.graph = graph
    }

    // MARK: Weather

    /// Set the precipitation mode (0 off, 1 rain, 2 snow). Scene
    /// state, but NOT undoable — undo snapshots only cover objects,
    /// so no beginEdit() here.
    func setWeatherMode(_ mode: Int) {
        let mode = mode == 2 ? 0 : mode // v4.13: snow mode retired — treated as Off
        noteUserInput()
        // Accumulators stay the source of truth — do not reset
        // them on a mode change (they were initialised from the
        // project and are kept in sync with it).
        if mode != 0 { weatherClock = 0 } // dynamics fade in fresh
        // v4.9: change-only published write — re-selecting the
        // current mode still resets the dynamics clock above
        // and kicks a frame, it just doesn't republish.
        if project.scene.weather.mode != mode {
            project.scene.weather.mode = mode
        }
        kickRender()
        if mode != 0 && !renderer.precipAvailable {
            status = "Precipitation unavailable on this device — ground effects still apply."
        }
        refreshFooter()
    }

    /// Flush the live build-up accumulators into the published
    /// project. Called before anything writes the project to disk.
    func syncWeatherToProject() {
        // v4.9: every write here is change-only — the values
        // written are exactly the pre-v4.9 ones, but a field
        // that already holds its synced value no longer fires
        // objectWillChange (this runs on every save/export).
        if project.scene.weather.wetness != buildupWet {
            project.scene.weather.wetness = buildupWet
        }
        // v4.13: snow state is no longer written back — stored
        // keys in old files are left as they are (ignored on
        // load). The retired plow layer is still actively
        // cleared so new saves carry none.
        if project.scene.weather.plowCells != nil {
            project.scene.weather.plowCells = nil
        }
    }

    /// v4.3: the quantised exposure step of one A-value, in
    /// maxDepth/24 units (the same quantum as the geometry steps).

    /// v4.3: scene objects with no exposure entry join at their
    /// persisted value (absent = fully caught up). Used after
    /// loads and snapshot restores — creation funnels register
    /// their objects explicitly at 0 before this could see them.

    /// v4.3: drop exposure entries for ids that are neither scene
    /// objects nor live clones (save / despawn / delete paths).

    /// v4.3: no object can hold more snow than the ambient A —
    /// after melts, compaction, direct level sets and Play-exit
    /// restores, every entry clamps down to A.

    /// v4.3: re-bake the GPU materials of every dynamic entry
    /// (promoted scene objects + clones) so the snow factor the
    /// renderer embeds in each material follows the entry's
    /// current exposure. Called ONLY from the coalesced geometry
    /// trigger in frameTick while snow is active. Clone
    /// materials are built from a temp object exactly as the
    /// spawn funnel builds them.

    /// v4.3: write the live exposure map back into the scene
    /// objects (scene objects ONLY — clones are Play-session
    /// ephemera). Called from save()/exports via
    /// syncWeatherToProject; never per frame.

    /// v4.3 (Weather sheet): set the Max Snow Depth. An explicit
    /// user action, so the published write is legitimate (weather
    /// precedent: not undoable). A itself is PRESERVED — progress
    /// simply rescales: raising the target lets snow keep
    /// building deeper; lowering it leaves deep snow standing,
    /// compacting slowly toward the new target while it snows
    /// (frameTick), never bulldozed instantly.

    /// v4.7 (Weather sheet): set the temperature (°C), or nil
    /// to clear it back to legacy behaviour. An explicit user
    /// action, so the published write is legitimate (same
    /// precedent as setMaxSnowDepth; not undoable).

    /// v4.8 (steering 1): water reflection resolution. Held
    /// on the renderer exactly like the quality preference
    /// (renderMode) is -- a renderer-side setting chosen in
    /// the UI, not project data. Steps: 1.0 Full (default,
    /// v4.7 quality), 0.5 Half, 0.25 Quarter; the setter
    /// snaps to the nearest step. UI PLACEMENT for the
    /// coordinator: a "Water reflections" picker with
    /// Full / Half / Quarter segments in the Render settings
    /// surface, beside the existing Quality picker that binds
    /// renderer.qualityPreference (EditorView ~line 140).

    /// v4.7: live temperature accessors for the UI.
    /// Formatted temperature with the real minus sign, e.g.
    /// "+10.0 °C" / "−4.5 °C" / "0.0 °C".
    /// v4.7: footer status line wording by regime (nil when
    /// temperature is unset — legacy projects show nothing).

    /// v4.3 (Weather sheet): re-roll this project's snow pattern.
    /// A fresh non-zero seed re-bakes the renderer's field with
    /// the new pattern; the geometry rebuild picks it up.
    /// v4.6: a re-roll is also the way back to PRISTINE snow —
    /// it clears the plow layer (trails and berms) along with
    /// the pattern, in the renderer and the saved project.

    /// Read-only live snow figures for the Weather sheet.
    var liveWetAmount: Double { buildupWet }

    /// Clear rain wetness / snow cover back to dry ground.
    func clearWeatherBuildup() {
        buildupWet = 0
        project.scene.weather.wetness = 0
        renderer.weatherWet = 0
        renderer.resetAccumulation()
        kickRender()
        status = "Ground build-up cleared."
        refreshFooter()
    }

    // MARK: v4.3 footprints + player figure

    /// Drop every footprint stamp and tell the renderer.

    /// Rotation-Y matrix for the player figure: local −Z (its
    /// facing) maps to the play-forward for `heading` (radians,
    /// playYaw convention: 0 faces −Z, + turns left).
    private func figureMatrix(feet: SIMD3<Float>, heading: Float) -> float4x4 {
        let c = cos(heading), s = sin(heading)
        var m = float4x4()
        m.columns.0 = SIMD4<Float>(c, 0, -s, 0)
        m.columns.1 = SIMD4<Float>(0, 1, 0, 0)
        m.columns.2 = SIMD4<Float>(s, 0, c, 0)
        m.columns.3 = SIMD4<Float>(feet.x, feet.y, feet.z, 1)
        return m
    }

    /// Edit mode: the figure stands idle at the saved Play start
    /// (feet = eye − 1.7), facing the saved look direction.
    private func setEditPlayerFigure() {
        // v4.5: the Edit orbit camera (and Cut To View's
        // stills) are third-person — the whole figure, head
        // included, renders normally.
        renderer.hidePlayerHead = false
        let cam = project.scene.camera
        let p = cam.playPosition
        let feet = SIMD3<Float>(Float(p[0]), Float(p[1]) - 1.7, Float(p[2]))
        let heading = Float(cam.playYaw) * .pi / 180
        figureHeading = heading
        let matrix = figureMatrix(feet: feet, heading: heading)
        renderer.setPlayerFigure(mesh: PlayerModel.mesh(phase: 0, blend: 0), matrix: matrix)
        figureLastSentPhase = 0
        figureLastSentBlend = 0
        figureLastMatrix = matrix
        figureLastPlayPos = nil
        editFigureSet = true
    }

    /// Per-tick Play driver + footprints, run just before the
    /// draw so teleports/checkpoints this tick are reflected.
    /// Purely visual: playPosition is READ, never written.
    private func drivePlayerFigure(now: CFTimeInterval, dt: Float) {
        guard mode == .play else { return }
        // v4.5: in Play the camera IS the figure's eye (inside
        // the head box) — the renderer excludes head + hair
        // from primary hits for as long as this is set.
        renderer.hidePlayerHead = true
        let p = project.scene.camera.playPosition
        let eye = SIMD3<Float>(Float(p[0]), Float(p[1]), Float(p[2]))
        let feet = eye - SIMD3<Float>(0, 1.7, 0)

        var travelled: Float = 0
        var speed: Float = 0
        var velocityHeading: Float?
        if let last = figureLastPlayPos {
            let d = eye - last
            travelled = simd_length(SIMD2<Float>(d.x, d.z))
            if dt > 0 { speed = travelled / dt }
            if travelled > 1e-5 {
                // Forward for heading θ is (−sinθ, −cosθ) in XZ,
                // so θ = atan2(−dx, −dz) for a velocity (dx, dz).
                velocityHeading = atan2(-d.x, -d.z)
            }
        }
        figureLastPlayPos = eye
        let moving = speed > 0.15
        // Heading target: the movement direction while actually
        // moving, otherwise the look direction. Smoothed toward
        // it at ≤540°/s on the shortest arc — turns swing,
        // never snap.
        let desired = (moving ? velocityHeading : nil)
            ?? Float(project.scene.camera.playYaw) * .pi / 180
        if dt > 0 {
            var delta = desired - figureHeading
            while delta > .pi { delta -= 2 * .pi }
            while delta < -.pi { delta += 2 * .pi }
            let maxTurn = Float(3 * Double.pi) * dt   // 540°/s
            figureHeading += min(maxTurn, max(-maxTurn, delta))
        } else {
            figureHeading = desired
        }
        // Walk cycle: phase advances 4.2 rad per metre travelled
        // (≈ a half-cycle per 0.75 m stride for the 1.75 m
        // figure); blend ramps in at 6/s moving, out at 4/s.
        figureWalkPhase += travelled * 4.2
        if moving {
            figureWalkBlend = min(1, figureWalkBlend + 6 * dt)
        } else {
            figureWalkBlend = max(0, figureWalkBlend - 4 * dt)
        }
        let matrix = figureMatrix(feet: feet, heading: figureHeading)
        let poseChanged = abs(figureWalkPhase - figureLastSentPhase) > 0.03
            || abs(figureWalkBlend - figureLastSentBlend) > 0.03
        if poseChanged {
            renderer.setPlayerFigure(
                mesh: PlayerModel.mesh(phase: figureWalkPhase, blend: figureWalkBlend),
                matrix: matrix)
            figureLastSentPhase = figureWalkPhase
            figureLastSentBlend = figureWalkBlend
            figureLastMatrix = matrix
        } else if let lastMatrix = figureLastMatrix {
            // Cheap path: matrix only, when position or heading
            // actually moved (column 0 encodes the heading).
            if simd_distance(matrix.columns.3, lastMatrix.columns.3) > 0.0001
                || simd_distance(matrix.columns.0, lastMatrix.columns.0) > 0.0001 {
                renderer.setPlayerFigureMatrix(matrix)
                figureLastMatrix = matrix
            }
        } else {
            renderer.setPlayerFigureMatrix(matrix)
            figureLastMatrix = matrix
        }
    }

    // MARK: Per-frame: keyboard movement + camera

    func frameTick(view: MTKView) {
        let now = CACurrentMediaTime()
        defer { lastTick = now }
        // v4.10 (WS1): idle-freeze engage check, at the TOP of
        // frameTick — before the dt computation below and
        // therefore before EVERY dt consumer (movement, weather
        // build-up, runtime.tick, the player-figure driver) and
        // before renderer.draw. Engaging requires the full
        // delay elapsed since the last input and no protected
        // operation in flight; if a protected operation is in
        // flight the check simply re-runs next frame, so the
        // freeze engages on the first clear frame past delay.
        if !isIdleFrozen, renderer.idleFreezeSeconds > 0,
           (now - lastInputAt) >= renderer.idleFreezeSeconds,
           !protectedOperationInFlight() {
            isIdleFrozen = true
            frozenSince = now
            refreshFooter()
        }
        // Frozen, or a final render is running: scene time is
        // paused. The defer above still rebases lastTick to
        // now, so even a stray draw cannot accumulate a dt
        // debt; the real wake additionally sets lastTick = nil
        // (dt == 0 on the wake frame, guaranteed). While a
        // final render runs this gate also keeps the idle
        // timer from engaging mid-render (belt and braces with
        // the protected list, which independently blocks it).
        if isIdleFrozen || finalRenderActive { return }
        let dt = lastTick.map { Float(now - $0) } ?? 0
        if dt > 0 && dt < 0.25 { applyMovement(dt: dt) }
        // Weather build-up advances with real time: rain wets
        // the ground over ~75 s and melts snow slowly; snow
        // covers the ground over ~100 s; with precipitation off
        // the ground dries / thaws gently. This advances the
        // plain accumulators only — NOT the published project —
        // so a weather scene no longer re-evaluates the whole
        // SwiftUI editor body every frame.
        if dt > 0, dt <= 0.25 {
            let mode = project.scene.weather.mode
            // Dynamics (v3.2): while weather is on, a slow surge
            // cycle drives the live intensity — detuned sines
            // wander it between a lull (~0.3) and a surge (~1.2)
            // over tens of seconds, faded in over the first ~6 s
            // after switch-on. The wind heading wanders and its
            // strength gusts on a related cycle. Build-up and
            // melt rates follow the intensity, so the ground
            // answers the sky: puddles grow in surges, snow piles
            // faster in gusts of snowfall.
            if mode != 0 {
                weatherClock += Double(dt)
                let t = weatherClock
                let surge = 0.62 + 0.30 * sin(t * 0.11)
                          + 0.22 * sin(t * 0.043 + 1.7)
                          + 0.12 * sin(t * 0.31 + 0.5)
                let fadeIn = min(1.0, t / 6.0)
                dynIntensity = min(1.2, max(0.25, surge)) * fadeIn
                let heading = 0.32 + 0.50 * sin(t * 0.017 + 1.3)
                            + 0.20 * sin(t * 0.061)
                let gust = 1.15 + 0.85 * sin(t * 0.13 + 2.0)
                         + 0.45 * sin(t * 0.047)
                let wmag = max(0.15, gust) * fadeIn
                dynWind = SIMD2<Float>(Float(cos(heading) * wmag),
                                       Float(sin(heading) * wmag))
            } else {
                dynIntensity = 0
            }
            // v4.1 (§4.4): the Weather Control intensity scale
            // applies AFTER the surge clamp (and after the
            // mode-0 zeroing — 0 × scale stays 0), BEFORE the
            // renderer writes below and BEFORE the build-up
            // switch, so shading, precip and build-up scale
            // together.
            dynIntensity *= weatherIntensityScale
            // v4.13: the only build-up left — rain wets the
            // flat ground-wetness value (~75 s to soaked,
            // surge-scaled); with rain off it dries slowly
            // (~600 s). No snow, no 3D water, no geometry
            // steps: the wetness is a shading uniform.
            if mode == 1 {
                buildupWet = min(1, buildupWet + Double(dt) * weatherRateScale / 75 * dynIntensity)
            } else {
                buildupWet = max(0, buildupWet - Double(dt) * weatherRateScale / 600)
            }
        }
        // v4.13: the renderer sees the stored mode (0 off /
        // 1 rain — snow mode no longer exists) and the flat
        // wetness; streak intensity follows the surge cycle.
        renderer.weatherMode = project.scene.weather.mode
        renderer.weatherWet = Float(buildupWet)
        renderer.weatherDynTime = Float(weatherClock
            .truncatingRemainder(dividingBy: 3600))
        renderer.weatherDynIntensity = Float(dynIntensity)
        renderer.weatherDynRain = project.scene.weather.mode == 1
            ? Float(dynIntensity) : 0
        renderer.weatherWind = dynWind
        // Published sync only when the displayed (whole-percent)
        // wetness actually changes.
        let wetCrossed = Int((buildupWet * 100).rounded())
            != Int((project.scene.weather.wetness * 100).rounded())
        if wetCrossed {
            project.scene.weather.wetness = buildupWet
        }
        // The footer follows the same change-only discipline
        // via its own observable — this per-frame call is inert
        // on unchanged frames.
        refreshFooter()
        // v4.0: banner expiry — a published write only on the
        // frame where it actually lapses.
        if !playBanner.isEmpty, let expiry = bannerExpiry, Date() >= expiry {
            playBanner = ""
            bannerExpiry = nil
        }
        // v4.0: the node runtime ticks BEFORE the draw, so
        // everything it moves/shows this frame is in the frame.
        if mode == .play, runtime.running {
            let p = project.scene.camera.playPosition
            runtime.tick(dt: dt, playerPos: SIMD3<Float>(Float(p[0]), Float(p[1]), Float(p[2])))
        }
        // v4.1 (F29): ONE coalesced static rebuild for everything
        // promoted during that tick, performed here — after the
        // runtime drain, before the camera basis/draw below, so a
        // promoted object never double-draws or vanishes for a
        // frame. collisionDidChange rides along (§6.6/F12): the
        // promoted objects' static tris just left the collision
        // set, so the runtime's grid must be rebuilt from it.
        if promotionRebuildPending {
            promotionRebuildPending = false
            renderer.rebuild(scene: project.scene, meshes: meshes)
            runtime.collisionDidChange()
        }
        // v4.0: in edit mode the trigger areas draw as debug
        // boxes — a plain renderer write each tick, no published
        // churn. The selected area burns brighter.
        if mode == .edit {
            renderer.debugBoxes = project.scene.areas.map { area in
                let c = area.center, s = area.size
                let lo = SIMD3<Float>(Float(c[0] - s[0] / 2), Float(c[1] - s[1] / 2), Float(c[2] - s[2] / 2))
                let hi = SIMD3<Float>(Float(c[0] + s[0] / 2), Float(c[1] + s[1] / 2), Float(c[2] + s[2] / 2))
                let color = area.id == selectedArea
                    ? SIMD4<Float>(1.0, 0.72, 0.15, 0.95)
                    : SIMD4<Float>(0.15, 0.75, 0.9, 0.35)
                return (min: lo, max: hi, color: color)
            }
        }
        let moving = !pressedKeys.isEmpty
            || touchActive
            || (now - lastCameraActivity) < 0.35
            || isDraggingObject
        var basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
        // v4.1 (§3.5): camera shake perturbs the BASIS ONLY, in
        // Play, while a shake is active — persisted playPosition /
        // playYaw / playPitch are never written here. The offset
        // and rotation decay linearly with remaining/duration.
        if mode == .play {
            if shakeEndsAt > now, shakeDuration > 0, shakeAmplitude > 0 {
                let remaining = Float(shakeEndsAt - now)
                let decay = max(0, min(1, remaining / shakeDuration))
                let scale = shakeAmplitude * decay
                let offset = SIMD3<Float>(Float.random(in: -1...1),
                                          Float.random(in: -1...1),
                                          Float.random(in: -1...1)) * scale * 0.05
                basis.origin += offset
                let tilt = scale * 0.02   // radians at amplitude 1
                var fwd = basis.forward
                    + basis.right * Float.random(in: -1...1) * tilt
                    + basis.up * Float.random(in: -1...1) * tilt
                fwd = simd_normalize(fwd)
                var right = simd_cross(fwd, SIMD3<Float>(0, 1, 0))
                if simd_length(right) < 1e-4 { right = basis.right }
                else { right = simd_normalize(right) }
                var up = simd_cross(right, fwd)
                // Small roll around the perturbed forward axis.
                let roll = Float.random(in: -1...1) * tilt
                let cr = cos(roll), sr = sin(roll)
                let rolledRight = right * cr + up * sr
                let rolledUp = up * cr - right * sr
                right = rolledRight
                up = rolledUp
                basis.forward = fwd
                basis.right = right
                basis.up = up
            } else if shakeEndsAt != 0 {
                // Expired (or cleared) shake: drop the state so
                // wantsFrame stops chaining frames for it.
                shakeAmplitude = 0
                shakeEndsAt = 0
                shakeDuration = 0
            }
        }
        // v4.3: footprints + the player figure are driven last,
        // after movement and the runtime tick, so this frame's
        // teleports/checkpoints are already in playPosition.
        // Edit mode keeps the idle figure posed at the Play start.
        if mode == .play {
            drivePlayerFigure(now: now, dt: dt)
        } else if !editFigureSet {
            setEditPlayerFigure()
        }
        renderer.draw(view: view, camera: basis,
                      moving: moving, play: mode == .play)
    }

    private var lastTick: CFTimeInterval?

    private func applyMovement(dt: Float) {
        guard !pressedKeys.isEmpty || touchActive else { return }
        var cam = project.scene.camera
        var moved = false
        let fwdKey = pressedKeys.contains(.w) || pressedKeys.contains(.up)
        let backKey = pressedKeys.contains(.s) || pressedKeys.contains(.down)
        let leftKey = pressedKeys.contains(.a) || pressedKeys.contains(.left)
        let rightKey = pressedKeys.contains(.d) || pressedKeys.contains(.right)
        let fwdAmt = (fwdKey ? 1 : 0) - (backKey ? 1 : 0)
        let sideAmt = (rightKey ? 1 : 0) - (leftKey ? 1 : 0)
        let vertAmt = (pressedKeys.contains(.e) ? 1 : 0) - (pressedKeys.contains(.q) ? 1 : 0)

        if mode == .play {
            let speed: Float = playSpeed   // v4.1 (§3.4): was a 2.4 literal
            let yaw = Float(cam.playYaw) * .pi / 180
            let f = SIMD2<Float>(-sin(yaw), -cos(yaw))     // XZ forward
            let r = SIMD2<Float>(cos(yaw), -sin(yaw))      // XZ right... (perp of f)
            // Keys and the touch joystick combine; an over-long
            // combined vector is normalised so diagonals and a
            // full key+stick push don't run faster than `speed`,
            // while a partial stick push stays proportional.
            let fwdAmtF = Float(fwdAmt) + touchVector.y
            let sideAmtF = Float(sideAmt) + touchVector.x
            var moveVec = SIMD2<Float>(sideAmtF, fwdAmtF)
            let moveLen = simd_length(moveVec)
            if moveLen > 1 { moveVec /= moveLen }
            if simd_length(moveVec) > 0 {
                let delta = (f * moveVec.y + r * moveVec.x) * speed * dt
                cam.playPosition[0] += Double(delta.x)
                cam.playPosition[2] += Double(delta.y)
                moved = true
            }
        } else {
            let basis = cam.orbitBasis()
            var groundF = basis.forward; groundF.y = 0
            groundF = simd_normalize(groundF)
            var groundR = basis.right; groundR.y = 0
            groundR = simd_normalize(groundR)
            let speed = Float(cam.distance) * 0.9 * dt
            var delta = (groundF * Float(fwdAmt) + groundR * Float(sideAmt)) * speed
            delta.y += Float(vertAmt) * speed
            if simd_length(delta) > 0 || vertAmt != 0 {
                cam.target[0] += Double(delta.x)
                cam.target[1] += Double(delta.y)
                cam.target[2] += Double(delta.z)
                moved = true
            }
        }
        if moved {
            // v4.9: change-only — `moved` can be true with a
            // zero net delta (opposing inputs, playSpeed 0);
            // an identical camera must not republish per frame.
            if project.scene.camera != cam {
                project.scene.camera = cam
            }
            lastCameraActivity = CACurrentMediaTime()
        }
    }

    /// F key: frame the selection with the orbit camera.
    func focusSelection() {
        guard mode == .edit, let obj = selectedObject else { return }
        let mesh: MeshData?
        if obj.source == "primitive" {
            mesh = MeshGen.primitive(kind: obj.primitiveKind, params: obj.params)
        } else {
            mesh = meshes[obj.id]
        }
        guard let local = mesh else { return }
        let world = local.baked(by: obj.transform.matrix)
        let b = world.bounds
        let centre = (b.min + b.max) / 2
        let diag = simd_length(b.max - b.min)
        project.scene.camera.target = [Double(centre.x), Double(centre.y), Double(centre.z)]
        project.scene.camera.distance = Double(max(1.2, diag * 1.35))
        noteCameraActivity()
    }

    // MARK: Picking

    /// World-space ray through an edit-mode viewport point
    /// (ndc: x right, y up, both -1...1).
    func rayFor(ndc: SIMD2<Float>, aspect: Float) -> (origin: SIMD3<Float>, dir: SIMD3<Float>) {
        let basis = project.scene.camera.orbitBasis()
        let tanH = tan(renderer.fovY / 2)
        let dir = simd_normalize(basis.forward
                                 + basis.right * (ndc.x * tanH * aspect)
                                 + basis.up * (ndc.y * tanH))
        return (basis.origin, dir)
    }

    func pickAt(ndc: SIMD2<Float>, aspect: Float) -> UUID? {
        let r = rayFor(ndc: ndc, aspect: aspect)
        return renderer.pick(origin: r.origin, dir: r.dir)
    }

    func handleTap(ndc: SIMD2<Float>, aspect: Float) {
        if mode == .play {
            // Play taps pick through the FIRST-PERSON camera (the
            // orbit rig rayFor uses is meaningless in Play) and
            // forward hits to the runtime — tap-to-interact for
            // node graphs. Misses are ignored.
            let basis = project.scene.camera.playBasis()
            let tanH = tan(renderer.fovY / 2)
            let dir = simd_normalize(basis.forward
                                     + basis.right * (ndc.x * tanH * aspect)
                                     + basis.up * (ndc.y * tanH))
            if let hit = renderer.pick(origin: basis.origin, dir: dir) {
                runtime.handleTap(objectID: hit)
            }
            return
        }
        if multiSelectMode {
            // Multi Select tool: taps toggle membership; a tap
            // on empty space clears the set.
            if let hit = pickAt(ndc: ndc, aspect: aspect) {
                toggleMultiSelection(hit)
            } else {
                clearMultiSelection()
                status = "Multi Select: cleared."
            }
            return
        }
        if let hit = pickAt(ndc: ndc, aspect: aspect) {
            // v4.9: change-only — re-tapping the already
            // selected object republishes neither property.
            if selection != hit { selection = hit }
            if let name = project.scene.objects.first(where: { $0.id == hit })?.name {
                let text = "Selected \(name)."
                if status != text { status = text }
            }
        } else {
            if selection != nil { selection = nil }
        }
    }

    // MARK: Save / export

    func save() {
        noteUserInput()
        syncWeatherToProject()
        project.modifiedAt = Date()
        // The thumbnail is the view the user is actually looking
        // at — orbit camera in edit mode, first-person in Play.
        let basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
        // v4.9: static AS builds are asynchronous now — flush
        // any pending build + swap synchronously so the
        // thumbnail renders the CURRENT scene, not the last
        // completed build (pre-v4.9 synchronous behaviour).
        renderer.flushPendingStaticBuild()
        let thumb = renderer.renderStillPNG(camera: basis, width: 640, height: 480, samples: 48)
        ProjectStore.saveProject(project, thumbnailPNG: thumb)
        status = "Saved."
    }

    func exportPackageURL() -> URL? {
        noteUserInput()
        syncWeatherToProject()
        save()
        return ProjectStore.exportPackage(projectID: project.id)
    }

    func exportPNGURL() -> URL? {
        noteUserInput()
        syncWeatherToProject()
        let basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
        // v4.9: same flush as save() — the export must render
        // the current scene, not a still-pending async build.
        renderer.flushPendingStaticBuild()
        guard let png = renderer.renderStillPNG(camera: basis, width: 1600, height: 1200, samples: 128)
        else { return nil }
        let name = project.name.replacingOccurrences(of: "/", with: "-")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(name).png")
        do {
            try png.write(to: url, options: .atomic)
            return url
        } catch { return nil }
    }
}
