// EditorSession.swift — the editor's glue: owns the open project,
// the selection, undo/redo (scene-object snapshots), the imported
// meshes, keyboard movement, play mode, saving and export. Views
// observe it; the viewport's draw loop calls frameTick each frame.

import Foundation
import SwiftUI
import MetalKit
import simd

enum EditorMode { case edit, play }

enum MoveKey: Hashable { case w, a, s, d, q, e, up, down, left, right }

@MainActor
final class EditorSession: ObservableObject {
    @Published var project: Project
    @Published var selection: UUID? {
        // An object selection and an area selection are mutually
        // exclusive: the properties panel shows exactly one editor.
        didSet { if selection != nil, selectedArea != nil { selectedArea = nil } }
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

    let renderer: Renderer
    /// v4.0: the node-graph game runtime. Inert in edit mode;
    /// started/stopped by toggleMode().
    let runtime: GameRuntime
    private(set) var meshes: [UUID: MeshData] = [:]

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
    private var playBuildupSnow: Double = 0
    private var playGeoStepWet: Int = 0
    private var playGeoStepSnow: Int = 0

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
    private var buildupSnow: Double = 0

    // MARK: v4.3 snow depth / exposure / footprints / player figure

    /// The user-set target depth (live from the published
    /// weather state; the settings sheet edits it as an explicit
    /// user action, which is a legitimate published write).
    private var snowMaxDepth: Double { project.scene.weather.maxDepth }
    /// Step size in metres for quantised geometry: maxDepth / 24.
    private var snowStepMetres: Double { snowMaxDepth / 24 }
    /// Unclamped accumulation progress p = A / maxDepth.
    private var snowProgress: Double {
        snowMaxDepth > 0 ? buildupSnow / snowMaxDepth : 0
    }
    /// Clamped progress pc in 0...1 — what the persisted
    /// `weather.snow` level and the continuous shading level are.
    private var snowProgressClamped: Double { min(1, max(0, snowProgress)) }
    /// Quantised AMOUNT step of A (uncapped — past 24 while deep
    /// snow above the target compacts). This is what geoStepSnow
    /// now counts; the renderer's weatherGeoSnow is the clamped
    /// progress step derived from it.
    private var snowAmountStep: Int {
        snowMaxDepth > 0 ? Int((buildupSnow / snowStepMetres).rounded()) : 0
    }
    /// Renderer-facing quantised progress (0...24).
    private var snowProgressStep: Int { Int((snowProgressClamped * 24).rounded()) }

    /// v4.3 (§4): per-object snow exposure — each scene object
    /// AND each Play clone tracks the absolute accumulation it
    /// has personally caught up with. New/spawned objects start
    /// at 0 and gather snow only as fast as the ambient A grows;
    /// loaded objects start at their persisted value (absent =
    /// fully caught up). Synced WHOLESALE to the renderer every
    /// tick (assignment is the sync — a deleted object's entry
    /// can never linger) and written back into the scene objects
    /// only on save/export (published-write discipline).
    private var snowExposureA: [UUID: Float] = [:]
    /// The per-object exposure steps already baked into geometry
    /// / dynamic materials, for the coalesced rebuild trigger.
    private var bakedExposureSteps: [UUID: Int] = [:]

    /// v4.3: player footprints in Play — transient stamps in the
    /// snow, fed to the renderer as compression spots. Nothing
    /// here is persisted; Play exit clears them before the
    /// snapshot-restore rebuild so restored snow is pristine.
    private struct SnowStamp { var x: Float; var z: Float; var expiresAt: CFTimeInterval }
    private var snowStamps: [SnowStamp] = []
    private var lastStampFeet: SIMD2<Float>?
    private var stampParity: Int = 0
    private var lastImprintRebuildAt: CFTimeInterval = 0
    private var imprintRebuildPending: Bool = false

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
    private var geoStepSnow = -1
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

    init(project: Project, renderer: Renderer) {
        self.runtime = GameRuntime()
        self.project = project
        self.renderer = renderer
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
        buildupSnow = project.scene.weather.snowAmount
            ?? (project.scene.weather.snow * project.scene.weather.maxDepth)
        // v4.3 (§4): seed the per-object exposure map. A persisted
        // value wins; absent means the object had fully caught
        // up with the ambient snow (old projects keep their look).
        for obj in project.scene.objects {
            snowExposureA[obj.id] = obj.snowExposure.map { Float($0) } ?? Float(buildupSnow)
        }
        bakedExposureSteps = snowExposureA.mapValues { exposureStep(for: $0) }
        renderer.onNeedsFrame = { [weak self] in self?.kickRender() }
        _ = renderer.reloadTextureArraysIfNeeded()
        // v4.3: hand the renderer this project's field seed and
        // depth settings BEFORE the first rebuild, so the baked
        // field and the snow geometry match the saved project.
        renderer.weatherSeed = project.scene.weather.seed
        renderer.weatherMaxDepth = Float(project.scene.weather.maxDepth)
        renderer.weatherSnowAmount = Float(buildupSnow)
        renderer.objectSnowAmounts = snowExposureA
        renderer.snowFootprints = []
        // v3.4: seed the renderer's quantised geometry levels
        // from the saved build-up BEFORE the first rebuild, so a
        // project saved with snow/puddles opens with its solid
        // weather already in place.
        geoStepWet = Int((buildupWet * 24).rounded())
        geoStepSnow = snowAmountStep
        renderer.weatherGeoWet = Float(geoStepWet) / 24
        renderer.weatherGeoSnow = Float(snowProgressStep) / 24
        renderer.rebuild(scene: project.scene, meshes: meshes)
        // v4.3: the player figure stands idle at the saved Play
        // start in edit mode.
        setEditPlayerFigure()
        if !renderer.rayTracingSupported {
            status = "This device does not support Metal ray tracing."
        }
    }

    // MARK: Render-on-change plumbing

    /// Ask the viewport for one draw. The viewport chains further
    /// draws itself while `wantsFrame` stays true, so a converged,
    /// untouched scene costs nothing.
    func kickRender() {
        viewport?.setNeedsDisplay()
    }

    func wantsFrame() -> Bool {
        renderer.frameWanted || !pressedKeys.isEmpty || touchActive
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
        guard let last = undoStack.popLast() else { return }
        if let cur = snapshot() { redoStack.append(cur) }
        restore(last)
        syncUndoDepths()
    }

    func redo() {
        guard let last = redoStack.popLast() else { return }
        if let cur = snapshot() { undoStack.append(cur) }
        restore(last)
        syncUndoDepths()
    }

    private func restore(_ data: Data) {
        guard let snap = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
        project.scene.objects = snap.objects
        if let sel = selection, !snap.objects.contains(where: { $0.id == sel }) {
            selection = nil
        }
        // v4.3: objects that (re)appear via undo/redo join the
        // exposure map at their snapshot value (absent = caught
        // up); entries for vanished objects are pruned.
        ensureExposuresForSceneObjects()
        pruneExposures()
        renderer.rebuild(scene: project.scene, meshes: meshes)
        kickRender()
    }

    private func syncUndoDepths() {
        undoDepth = undoStack.count
        redoDepth = redoStack.count
    }

    var canUndo: Bool { undoDepth > 0 }
    var canRedo: Bool { redoDepth > 0 }

    // MARK: Scene mutation plumbing

    /// Live mutation (no snapshot): rebuilds the renderer. Pair with
    /// beginEdit()/beginEditIfNewGesture() at gesture start.
    func mutateScene(_ change: (inout Scene) -> Void) {
        change(&project.scene)
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
    var isDraggingObject: Bool { dragState != nil }

    /// Start dragging the given object across the horizontal plane
    /// through its current height. Returns false when the ray
    /// can't reach that plane (camera level with it).
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
        noteCameraActivity()
        scheduleRebuild(geometry: true)
    }

    func endObjectDrag() {
        guard dragState != nil else { return }
        dragState = nil
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
        // v4.3: a brand-new object has caught NO snow yet — if
        // snow is already down it gathers visibly from zero.
        snowExposureA[obj.id] = 0
        bakedExposureSteps[obj.id] = 0
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
        guard let sel = selection,
              let idx = project.scene.objects.firstIndex(where: { $0.id == sel }) else { return }
        beginEdit()
        let name = project.scene.objects[idx].name
        let removedID = project.scene.objects[idx].id
        project.scene.objects.remove(at: idx)
        // v4.3: prune the deleted object's exposure immediately,
        // so it never lingers in the renderer's map.
        snowExposureA.removeValue(forKey: removedID)
        bakedExposureSteps.removeValue(forKey: removedID)
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
                    // v4.3: imported parts start with no snow
                    // caught, like any newly created object.
                    self.snowExposureA[obj.id] = 0
                    self.bakedExposureSteps[obj.id] = 0
                    if firstID == nil { firstID = obj.id }
                }
                self.selection = firstID
                self.renderer.rebuild(scene: self.project.scene, meshes: self.meshes)
                self.kickRender()
                self.status = "Imported \(parts.count) part\(parts.count == 1 ? "" : "s") from \(url.lastPathComponent)."
            }
        }
    }

    // MARK: Play mode

    func toggleMode() {
        pressedKeys.removeAll() // a held key must not survive the switch
        touchVector = .zero // a stuck joystick must not survive either
        if mode == .edit {
            // Entering Play. FIRST take the restore snapshot —
            // before the runtime touches anything — then clear
            // all promotion/streaming state so the session starts
            // from the scene exactly as authored.
            playSnapshot = snapshot()
            playWeatherMode = project.scene.weather.mode
            // v4.1 (F15/§4.4): snapshot the weather build-up too.
            playBuildupWet = buildupWet
            playBuildupSnow = buildupSnow
            playGeoStepWet = geoStepWet
            playGeoStepSnow = geoStepSnow
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
            playBanner = ""
            bannerExpiry = nil
            // v4.3: footprints start empty every Play session,
            // and the figure driver starts from the look yaw
            // with a still pose (first Play tick force-sends it).
            clearSnowStamps()
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
            // v4.3: their exposure entries die with them.
            for clone in clones {
                snowExposureA.removeValue(forKey: clone.cloneID)
                bakedExposureSteps.removeValue(forKey: clone.cloneID)
            }
            clones.removeAll()
            cloneEntries.removeAll()
            // v4.3: footprints clear BEFORE the snapshot-restore
            // rebuild below, so the restored snow is pristine.
            clearSnowStamps()
            renderer.setParticleSoups([])
            renderer.unloadedAreaIDs = []
            runtimeUnloaded.removeAll()
            // v4.1 (F15/§4.4): restore the weather build-up BEFORE
            // the snapshot restore, so its rebuild bakes the
            // restored geometry levels, and reset every new
            // session state (Appendix B: resets in BOTH branches).
            buildupWet = playBuildupWet
            buildupSnow = playBuildupSnow
            geoStepWet = playGeoStepWet
            geoStepSnow = playGeoStepSnow
            renderer.weatherGeoWet = Float(geoStepWet) / 24
            renderer.weatherGeoSnow = Float(snowProgressStep) / 24
            project.scene.weather.wetness = buildupWet
            // v4.3: the published level is the clamped progress;
            // the absolute amount rides alongside it.
            project.scene.weather.snow = snowProgressClamped
            project.scene.weather.snowAmount = buildupSnow
            // v4.3: exposures gathered during Play belong to the
            // Play session's weather, which was just rolled back
            // — clamp them to the restored ambient so nothing
            // keeps more snow than the ground has.
            clampExposuresToAmbient()
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
            playBanner = ""
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
        // v4.3: restored objects carry their Play-entry snapshot
        // exposures (absent = caught up); prune anything else.
        ensureExposuresForSceneObjects()
        pruneExposures()
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
        if shouldShow, !hasEntry, let mesh = localMesh(for: obj), !mesh.isEmpty {
            dynamicEntries.append(DynamicEntry(
                key: id, mesh: mesh, matrix: obj.transform.matrix,
                material: renderer.gpuMaterialForObject(obj), objectID: id))
            // v4.1 (§7.1): the renderer's entry list is ALWAYS
            // promoted entries (in order) + clone entries.
            renderer.setDynamicEntries(dynamicEntries + cloneEntries)
        } else if !shouldShow, hasEntry {
            dynamicEntries.removeAll { $0.key == id }
            renderer.removeDynamicEntry(key: id)
        }
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
            project.scene.objects[idx].visible = visible
            renderer.rebuild(scene: project.scene, meshes: meshes)
            runtime.collisionDidChange()
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
        playBanner = text
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
        project.scene.camera.playPosition = [Double(t.x), eyeY, Double(t.z)]
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
        project.scene.camera.playPosition = [Double(position.x), Double(position.y), Double(position.z)]
        project.scene.camera.playYaw = yaw
        project.scene.camera.playPitch = min(85, max(-85, pitch))
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
        if wet != nil || snow != nil {
            if let wet { buildupWet = min(1, max(0, wet)) }
            // v4.3: the node's snow parameter is a 0...1 PROGRESS
            // (its catalogue contract is unchanged) — in absolute
            // terms that is progress * maxDepth metres.
            if let snow { buildupSnow = min(1, max(0, snow)) * snowMaxDepth }
            project.scene.weather.wetness = buildupWet
            project.scene.weather.snow = snowProgressClamped
            project.scene.weather.snowAmount = buildupSnow
            // A direct level set can only lower exposures (they
            // re-gather from falling snow, per the age rule).
            clampExposuresToAmbient()
            geoStepWet = Int((buildupWet * 24).rounded())
            geoStepSnow = snowAmountStep
            renderer.weatherGeoWet = Float(geoStepWet) / 24
            renderer.weatherGeoSnow = Float(snowProgressStep) / 24
            renderer.rebuild(scene: project.scene, meshes: meshes)
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
            // v4.3: the clone's exposure starts at ZERO and is
            // registered BEFORE its material is built below, so
            // the spawn-time material already carries the bare
            // (uncaught) snow factor — a clone spawned into deep
            // snow appears bare and gathers visibly.
            snowExposureA[cloneID] = 0
            bakedExposureSteps[cloneID] = 0
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
            renderer.objectSnowAmounts = snowExposureA
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
        // v4.3: despawned clones' exposure entries go too.
        for id in removed {
            snowExposureA.removeValue(forKey: id)
            bakedExposureSteps.removeValue(forKey: id)
        }
        renderer.objectSnowAmounts = snowExposureA
        renderer.setDynamicEntries(dynamicEntries + cloneEntries)
        kickRender()
    }

    /// Despawn one clone by its clone id (runtime expiry path).
    func runtimeDespawnClone(_ cloneID: UUID) {
        guard clones.contains(where: { $0.cloneID == cloneID }) else { return }
        clones.removeAll { $0.cloneID == cloneID }
        cloneEntries.removeAll { $0.key == cloneID }
        snowExposureA.removeValue(forKey: cloneID)
        bakedExposureSteps.removeValue(forKey: cloneID)
        renderer.objectSnowAmounts = snowExposureA
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
        if unassigned { renderer.rebuild(scene: project.scene, meshes: meshes) }
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
        // Accumulators stay the source of truth — do not reset
        // them on a mode change (they were initialised from the
        // project and are kept in sync with it).
        if mode != 0 { weatherClock = 0 } // dynamics fade in fresh
        project.scene.weather.mode = mode
        kickRender()
        if mode != 0 && !renderer.precipAvailable {
            status = "Precipitation unavailable on this device — ground effects still apply."
        }
    }

    /// Flush the live build-up accumulators into the published
    /// project. Called before anything writes the project to disk.
    func syncWeatherToProject() {
        project.scene.weather.wetness = buildupWet
        // v4.3: the level persists as clamped progress; the
        // absolute amount and the per-object exposures ride
        // along so a reopened project resumes exactly here.
        project.scene.weather.snow = snowProgressClamped
        project.scene.weather.snowAmount = buildupSnow
        syncExposuresToProject()
    }

    /// v4.3: the quantised exposure step of one A-value, in
    /// maxDepth/24 units (the same quantum as the geometry steps).
    private func exposureStep(for amount: Float) -> Int {
        snowMaxDepth > 0 ? Int((Double(amount) / snowStepMetres).rounded()) : 0
    }

    /// v4.3: scene objects with no exposure entry join at their
    /// persisted value (absent = fully caught up). Used after
    /// loads and snapshot restores — creation funnels register
    /// their objects explicitly at 0 before this could see them.
    private func ensureExposuresForSceneObjects() {
        for obj in project.scene.objects where snowExposureA[obj.id] == nil {
            snowExposureA[obj.id] = obj.snowExposure.map { Float($0) } ?? Float(buildupSnow)
            bakedExposureSteps[obj.id] = exposureStep(for: snowExposureA[obj.id] ?? 0)
        }
    }

    /// v4.3: drop exposure entries for ids that are neither scene
    /// objects nor live clones (save / despawn / delete paths).
    private func pruneExposures() {
        let live = Set(project.scene.objects.map(\.id)).union(clones.map(\.cloneID))
        snowExposureA = snowExposureA.filter { live.contains($0.key) }
        bakedExposureSteps = bakedExposureSteps.filter { live.contains($0.key) }
    }

    /// v4.3: no object can hold more snow than the ambient A —
    /// after melts, compaction, direct level sets and Play-exit
    /// restores, every entry clamps down to A.
    private func clampExposuresToAmbient() {
        let cap = Float(buildupSnow)
        for key in snowExposureA.keys where (snowExposureA[key] ?? 0) > cap {
            snowExposureA[key] = cap
        }
    }

    /// v4.3: re-bake the GPU materials of every dynamic entry
    /// (promoted scene objects + clones) so the snow factor the
    /// renderer embeds in each material follows the entry's
    /// current exposure. Called ONLY from the coalesced geometry
    /// trigger in frameTick while snow is active. Clone
    /// materials are built from a temp object exactly as the
    /// spawn funnel builds them.
    private func refreshDynamicSnowMaterials() {
        for i in dynamicEntries.indices {
            let key = dynamicEntries[i].key
            guard let obj = project.scene.object(id: key) else { continue }
            let gpu = renderer.gpuMaterialForObject(obj)
            dynamicEntries[i].material = gpu
            renderer.setDynamicMaterial(key: key, material: gpu)
        }
        for i in cloneEntries.indices {
            let key = cloneEntries[i].key
            guard let clone = clones.first(where: { $0.cloneID == key }),
                  let source = project.scene.object(id: clone.sourceID)
            else { continue }
            var temp = source
            temp.id = clone.cloneID
            temp.transform = clone.transform
            temp.material = clone.material
            let gpu = renderer.gpuMaterialForObject(temp)
            cloneEntries[i].material = gpu
            renderer.setDynamicMaterial(key: key, material: gpu)
        }
    }

    /// v4.3: write the live exposure map back into the scene
    /// objects (scene objects ONLY — clones are Play-session
    /// ephemera). Called from save()/exports via
    /// syncWeatherToProject; never per frame.
    func syncExposuresToProject() {
        pruneExposures()
        for i in project.scene.objects.indices {
            if let exp = snowExposureA[project.scene.objects[i].id] {
                project.scene.objects[i].snowExposure = Double(exp)
            }
        }
    }

    /// v4.3 (Weather sheet): set the Max Snow Depth. An explicit
    /// user action, so the published write is legitimate (weather
    /// precedent: not undoable). A itself is PRESERVED — progress
    /// simply rescales: raising the target lets snow keep
    /// building deeper; lowering it leaves deep snow standing,
    /// compacting slowly toward the new target while it snows
    /// (frameTick), never bulldozed instantly.
    func setMaxSnowDepth(_ value: Double) {
        let clamped = min(2.0, max(0.05, value))
        project.scene.weather.maxDepth = clamped
        project.scene.weather.snow = snowProgressClamped
        project.scene.weather.snowAmount = buildupSnow
        renderer.weatherMaxDepth = Float(clamped)
        geoStepSnow = snowAmountStep
        renderer.weatherGeoSnow = Float(snowProgressStep) / 24
        bakedExposureSteps = snowExposureA.mapValues { exposureStep(for: $0) }
        scheduleRebuild(geometry: true)
        kickRender()
    }

    /// v4.3 (Weather sheet): re-roll this project's snow pattern.
    /// A fresh non-zero seed re-bakes the renderer's field with
    /// the new pattern; the geometry rebuild picks it up.
    func rerollSnowSeed() {
        let seed = UInt32.random(in: 1...UInt32.max)
        project.scene.weather.seed = seed
        renderer.weatherSeed = seed
        renderer.rebuild(scene: project.scene, meshes: meshes)
        renderer.resetAccumulation()
        kickRender()
        status = "Snow pattern re-rolled."
    }

    /// Read-only live snow figures for the Weather sheet.
    var liveSnowAmount: Double { buildupSnow }
    var liveSnowProgress: Double { snowProgressClamped }

    /// Clear rain wetness / snow cover back to dry ground.
    func clearWeatherBuildup() {
        buildupWet = 0
        buildupSnow = 0
        project.scene.weather.wetness = 0
        project.scene.weather.snow = 0
        project.scene.weather.snowAmount = 0
        // v4.3: every object's caught snow goes with the ground's,
        // and any footprints vanish.
        for key in snowExposureA.keys { snowExposureA[key] = 0 }
        bakedExposureSteps = snowExposureA.mapValues { _ in 0 }
        clearSnowStamps()
        renderer.objectSnowAmounts = snowExposureA
        // v3.4: the solid weather geometry goes too — a full
        // rebuild at zero levels removes sheets + displacement.
        geoStepWet = 0
        geoStepSnow = 0
        renderer.weatherGeoWet = 0
        renderer.weatherGeoSnow = 0
        renderer.rebuild(scene: project.scene, meshes: meshes)
        renderer.resetAccumulation()
        kickRender()
        status = "Ground build-up cleared."
    }

    // MARK: v4.3 footprints + player figure

    /// Drop every footprint stamp and tell the renderer.
    private func clearSnowStamps() {
        snowStamps.removeAll()
        lastStampFeet = nil
        stampParity = 0
        imprintRebuildPending = false
        renderer.snowFootprints = []
    }

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
    private func drivePlayerAndFootprints(now: CFTimeInterval, dt: Float) {
        guard mode == .play else { return }
        let p = project.scene.camera.playPosition
        let eye = SIMD3<Float>(Float(p[0]), Float(p[1]), Float(p[2]))
        let feet = eye - SIMD3<Float>(0, 1.7, 0)

        // --- footprints -------------------------------------
        // Expire first, then stamp. A stamp lands every 0.35 m
        // of travel through snow that is actually deep here,
        // offset alternately left/right of the stride line.
        var stampsChanged = false
        let beforeExpire = snowStamps.count
        snowStamps.removeAll { $0.expiresAt <= now }
        if snowStamps.count != beforeExpire { stampsChanged = true }
        if buildupSnow > 0.005,
           renderer.snowDepthAt(x: feet.x, z: feet.z) > 0.02 {
            let feetXZ = SIMD2<Float>(feet.x, feet.z)
            let dist = lastStampFeet.map { simd_distance(feetXZ, $0) } ?? .infinity
            if dist > 0.35 {
                var dir = SIMD2<Float>(-sin(figureHeading), -cos(figureHeading))
                if let last = lastStampFeet {
                    let d = feetXZ - last
                    if simd_length(d) > 1e-4 { dir = simd_normalize(d) }
                }
                let perp = SIMD2<Float>(-dir.y, dir.x)
                let side: Float = (stampParity % 2 == 0) ? 0.09 : -0.09
                let pos = feetXZ + perp * side
                snowStamps.append(SnowStamp(x: pos.x, z: pos.y, expiresAt: now + 25))
                stampParity += 1
                lastStampFeet = feetXZ
                if snowStamps.count > 48 { snowStamps.removeFirst(snowStamps.count - 48) }
                stampsChanged = true
            }
        }
        renderer.snowFootprints = snowStamps.map { stamp in
            let fade = Float(min(1, max(0, (stamp.expiresAt - now) / 5)))
            return Renderer.SnowFootprint(x: stamp.x, z: stamp.z, radius: 0.16, compress: fade)
        }
        if stampsChanged { imprintRebuildPending = true }
        if imprintRebuildPending, now - lastImprintRebuildAt >= 0.8 {
            imprintRebuildPending = false
            lastImprintRebuildAt = now
            renderer.rebuild(scene: project.scene, meshes: meshes)
        }

        // --- figure -----------------------------------------
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
            // v4.3: snow rates are in ABSOLUTE metres. `depth`
            // is the rate scale (the target depth, with a sane
            // fallback for a degenerate 0 in a hand-edited file);
            // at the default 0.3 m every rate below reproduces
            // the v3/v4.2 level rates × 0.3 exactly.
            let depth = snowMaxDepth > 0 ? snowMaxDepth : 0.3
            let snowAmountBefore = buildupSnow
            switch mode {
            case 1:
                buildupWet = min(1, buildupWet + Double(dt) * weatherRateScale / 75 * dynIntensity)
                // Rain melts snow: the old level rate (1/300,
                // intensity-blended) × maxDepth, in A units.
                buildupSnow = max(0, buildupSnow - depth * Double(dt) * weatherRateScale / 300
                                        * (0.6 + 0.4 * dynIntensity))
            case 2:
                if buildupSnow < snowMaxDepth {
                    // 100 s to the target depth, surge-scaled —
                    // the v3 snow feel at the default depth.
                    buildupSnow = min(snowMaxDepth, buildupSnow
                        + (depth / 100) * weatherRateScale * dynIntensity * Double(dt))
                } else if buildupSnow > snowMaxDepth {
                    // The target was lowered under deep snow:
                    // slow COMPACTION toward it (900 s scale,
                    // half rate), never below it while snowing.
                    buildupSnow = max(snowMaxDepth, buildupSnow
                        - (depth / 900) * 0.5 * weatherRateScale * Double(dt))
                }
            case 0:
                buildupWet = max(0, buildupWet - Double(dt) * weatherRateScale / 600)
                // Full melt in ~15 min at the default depth.
                buildupSnow = max(0, buildupSnow - (depth / 900) * weatherRateScale * Double(dt))
            default:
                break
            }
            // v4.3 (§4): per-object exposure follows the ambient
            // A. Growth (snowing only) lifts every entry below A
            // by the same delta, capped at A — young objects lag
            // exactly by their age. Any shrink clamps entries
            // down to A. All in the non-published map.
            let dA = buildupSnow - snowAmountBefore
            if dA > 0, mode == 2 {
                let cap = Float(buildupSnow)
                let delta = Float(dA)
                for key in snowExposureA.keys {
                    let v = snowExposureA[key] ?? 0
                    if v < cap { snowExposureA[key] = min(cap, v + delta) }
                }
            } else if dA < 0 {
                clampExposuresToAmbient()
            }
        }
        renderer.weatherMode = project.scene.weather.mode
        renderer.weatherWet = Float(buildupWet)
        // v4.3: the continuous shading level is the clamped
        // PROGRESS; the absolute amount and target depth travel
        // separately, and the exposure map syncs wholesale.
        renderer.weatherSnow = Float(snowProgressClamped)
        renderer.weatherSnowAmount = Float(buildupSnow)
        renderer.weatherMaxDepth = Float(snowMaxDepth)
        renderer.objectSnowAmounts = snowExposureA
        renderer.weatherDynTime = Float(weatherClock
            .truncatingRemainder(dividingBy: 3600))
        renderer.weatherDynIntensity = Float(dynIntensity)
        renderer.weatherDynRain = project.scene.weather.mode == 1
            ? Float(dynIntensity) : 0
        renderer.weatherWind = dynWind
        // v3.4: weather GEOMETRY follows the build-up in
        // quantised steps of 1/24, coalesced to at most one
        // geometry rebuild per ~0.8 s. A step is ~4% of full
        // cover — at full snow that's about a centimetre of
        // drift growth per rebuild, arriving in the same short
        // accumulation-reset bursts scene edits already use.
        // v4.3: the snow step counts A in maxDepth/24 units
        // (uncapped, so compaction of over-target snow still
        // steps), the renderer's geo level is the CLAMPED
        // progress step, and per-object exposure steps are an
        // additional trigger into the same coalesced block.
        let stepWet = Int((buildupWet * 24).rounded())
        let stepSnow = snowAmountStep
        var exposureSteps: [UUID: Int] = [:]
        var exposureStepChanged = false
        for (id, amount) in snowExposureA {
            let s = exposureStep(for: amount)
            exposureSteps[id] = s
            if bakedExposureSteps[id] != s { exposureStepChanged = true }
        }
        if (stepWet != geoStepWet || stepSnow != geoStepSnow || exposureStepChanged),
           now - lastGeoRebuildAt >= 0.8 {
            geoStepWet = stepWet
            geoStepSnow = stepSnow
            bakedExposureSteps = exposureSteps
            lastGeoRebuildAt = now
            renderer.weatherGeoWet = Float(stepWet) / 24
            renderer.weatherGeoSnow = Float(snowProgressStep) / 24
            renderer.rebuild(scene: project.scene, meshes: meshes)
            // v4.3: dynamic entries (promoted objects + clones)
            // aren't in the static rebuild — refresh their
            // materials so their snow factor follows their
            // exposure step. Bounded: only on this coalesced
            // trigger, only while snow is in play.
            if buildupSnow > 0 || project.scene.weather.mode == 2 {
                refreshDynamicSnowMaterials()
            }
        }
        // Published sync only when the displayed (whole-percent)
        // value actually changes. The snow percent is of the
        // clamped progress pc.
        if Int((buildupWet * 100).rounded()) != Int((project.scene.weather.wetness * 100).rounded())
            || Int((snowProgressClamped * 100).rounded()) != Int((project.scene.weather.snow * 100).rounded()) {
            project.scene.weather.wetness = buildupWet
            project.scene.weather.snow = snowProgressClamped
            project.scene.weather.snowAmount = buildupSnow
        }
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
            drivePlayerAndFootprints(now: now, dt: dt)
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
            project.scene.camera = cam
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
        if let hit = pickAt(ndc: ndc, aspect: aspect) {
            selection = hit
            if let name = project.scene.objects.first(where: { $0.id == hit })?.name {
                status = "Selected \(name)."
            }
        } else {
            selection = nil
        }
    }

    // MARK: Save / export

    func save() {
        syncWeatherToProject()
        project.modifiedAt = Date()
        // The thumbnail is the view the user is actually looking
        // at — orbit camera in edit mode, first-person in Play.
        let basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
        let thumb = renderer.renderStillPNG(camera: basis, width: 640, height: 480, samples: 48)
        ProjectStore.saveProject(project, thumbnailPNG: thumb)
        status = "Saved."
    }

    func exportPackageURL() -> URL? {
        syncWeatherToProject()
        save()
        return ProjectStore.exportPackage(projectID: project.id)
    }

    func exportPNGURL() -> URL? {
        syncWeatherToProject()
        let basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
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
