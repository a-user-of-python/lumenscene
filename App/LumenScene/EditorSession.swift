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
    @Published var selection: UUID?
    @Published var mode: EditorMode = .edit
    @Published var status: String = ""
    @Published var undoDepth: Int = 0
    @Published var redoDepth: Int = 0

    let renderer: Renderer
    private(set) var meshes: [UUID: MeshData] = [:]

    private struct Snapshot: Codable { var objects: [SceneObject] }
    private var undoStack: [Data] = []
    private var redoStack: [Data] = []
    private var editGestureAt: Date = .distantPast

    /// Live weather build-up (source of truth while editing).
    /// Mutating `project` every frame would fire objectWillChange
    /// 60x/s and re-evaluate the whole editor UI, so frameTick
    /// advances these plain accumulators and only syncs the
    /// published values when the displayed percentage changes.
    private var buildupWet: Double = 0
    private var buildupSnow: Double = 0

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
        self.project = project
        self.renderer = renderer
        for obj in project.scene.objects where obj.source == "model" {
            if let file = obj.meshFile,
               let mesh = ProjectStore.readMesh(projectID: project.id, fileName: file) {
                meshes[obj.id] = mesh
            }
        }
        buildupWet = project.scene.weather.wetness
        buildupSnow = project.scene.weather.snow
        renderer.onNeedsFrame = { [weak self] in self?.kickRender() }
        _ = renderer.reloadTextureArraysIfNeeded()
        renderer.rebuild(scene: project.scene, meshes: meshes)
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
        project.scene.objects.remove(at: idx)
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
        mode = (mode == .edit) ? .play : .edit
        renderer.resetAccumulation()
        noteCameraActivity()
        status = mode == .play ? "Play mode — walk around. Pause to edit."
                                : "Edit mode."
    }

    // MARK: Weather

    /// Set the precipitation mode (0 off, 1 rain, 2 snow). Scene
    /// state, but NOT undoable — undo snapshots only cover objects,
    /// so no beginEdit() here.
    func setWeatherMode(_ mode: Int) {
        // Accumulators stay the source of truth — do not reset
        // them on a mode change (they were initialised from the
        // project and are kept in sync with it).
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
        project.scene.weather.snow = buildupSnow
    }

    /// Clear rain wetness / snow cover back to dry ground.
    func clearWeatherBuildup() {
        buildupWet = 0
        buildupSnow = 0
        project.scene.weather.wetness = 0
        project.scene.weather.snow = 0
        renderer.resetAccumulation()
        kickRender()
        status = "Ground build-up cleared."
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
            switch project.scene.weather.mode {
            case 1:
                buildupWet = min(1, buildupWet + Double(dt) / 75)
                buildupSnow = max(0, buildupSnow - Double(dt) / 300)
            case 2:
                buildupSnow = min(1, buildupSnow + Double(dt) / 100)
            case 0:
                buildupWet = max(0, buildupWet - Double(dt) / 600)
                buildupSnow = max(0, buildupSnow - Double(dt) / 900)
            default:
                break
            }
        }
        renderer.weatherMode = project.scene.weather.mode
        renderer.weatherWet = Float(buildupWet)
        renderer.weatherSnow = Float(buildupSnow)
        // Published sync only when the displayed (whole-percent)
        // value actually changes.
        if Int((buildupWet * 100).rounded()) != Int((project.scene.weather.wetness * 100).rounded())
            || Int((buildupSnow * 100).rounded()) != Int((project.scene.weather.snow * 100).rounded()) {
            project.scene.weather.wetness = buildupWet
            project.scene.weather.snow = buildupSnow
        }
        let moving = !pressedKeys.isEmpty
            || touchActive
            || (now - lastCameraActivity) < 0.35
            || isDraggingObject
        let basis = mode == .play ? project.scene.camera.playBasis()
                                  : project.scene.camera.orbitBasis()
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
            let speed: Float = 2.4
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
        guard mode == .edit else { return }
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
