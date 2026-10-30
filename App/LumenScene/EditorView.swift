// EditorView.swift — the editor: top bar (back / project name /
// undo / redo / play-pause / save / export), object tree on the
// left, ray-traced viewport in the centre, properties for the
// selection on the right. Keyboard: WASD/arrows move, Q/E down/up,
// F focus, Delete removes, Cmd+Z / Cmd+Shift+Z, Cmd+S saves,
// P or the top-bar button toggles Play, Esc leaves Play.

import SwiftUI
import UIKit
import simd

enum FieldFocus: Hashable { case projectName, objectName }

struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

struct EditorView: View {
    @ObservedObject var session: EditorSession
    var onClose: () -> Void

    @State private var showImport = false
    @State private var shareItem: ShareItem?
    @FocusState private var fieldFocus: FieldFocus?
    @FocusState private var kbFocused: Bool

    var body: some View {
        Group {
            if session.mode == .play {
                playLayout
            } else {
                editLayout
            }
        }
        .background(Color(white: 0.09))
        .focusable()
        .focused($kbFocused)
        .onAppear {
            kbFocused = true
            session.kickRender()
        }
        .onDisappear { session.pressedKeys.removeAll() }
        .onChange(of: kbFocused) { _, focused in
            // If focus is lost mid-keypress the key-up can be
            // missed; a stuck movement key would drift the camera
            // and keep the renderer awake forever.
            if !focused { session.pressedKeys.removeAll() }
        }
        .onKeyPress(phases: .all) { press in handleKey(press) }
        .sheet(isPresented: $showImport) { importSheet }
        .sheet(item: $shareItem) { item in ShareSheet(url: item.url) }
    }

    // MARK: Layouts

    /// Edit mode: the full editor chrome (top bar + panels).
    private var editLayout: some View {
        VStack(spacing: 0) {
            topBar
            Divider()
            HStack(spacing: 0) {
                treePanel
                    .frame(width: 238)
                Divider()
                viewportPanel
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                propertiesPanel
                    .frame(width: 302)
            }
        }
    }

    /// Play mode: the 3D view fills everything. The only chrome is
    /// a circular pause button (top-right) and the touch joystick
    /// (bottom-left) — no panels, no hint bars, no status text.
    private var playLayout: some View {
        ZStack {
            MetalViewport(session: session)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .overlay(alignment: .topTrailing) {
            Button { session.toggleMode() } label: {
                Image(systemName: "pause.fill")
                    .foregroundStyle(.white)
                    .frame(width: 52, height: 52)
                    .background(Color.black.opacity(0.55))
                    .clipShape(Circle())
                    .overlay(Circle().stroke(Color.white.opacity(0.25), lineWidth: 1))
            }
            .buttonStyle(.plain)
            .padding(.top, 16)
            .padding(.trailing, 16)
        }
        .overlay(alignment: .bottomLeading) {
            JoystickView(session: session)
                .padding(.leading, 22)
                .padding(.bottom, 26)
        }
    }

    private var qualityBinding: Binding<Int> {
        Binding(
            get: { session.renderer.qualityPreference },
            set: { newValue in
                session.renderer.qualityPreference = newValue
                session.renderer.resetAccumulation()
                session.kickRender()
            })
    }

    private var weatherBinding: Binding<Int> {
        Binding(
            get: { session.project.scene.weather.mode },
            set: { session.setWeatherMode($0) })
    }

    /// "Ground: 62% wet · Snow cover: 30%" style summary of the
    /// current weather build-up, for the tree panel.
    private var weatherCaption: String {
        let w = session.project.scene.weather
        var parts: [String] = []
        if w.wetness > 0.02 {
            parts.append("Ground: \(Int((w.wetness * 100).rounded()))% wet")
        }
        if w.snow > 0.02 {
            parts.append("Snow cover: \(Int((w.snow * 100).rounded()))%")
        }
        if parts.isEmpty {
            if w.mode == 1 { parts.append("Ground: 0% wet") }
            else if w.mode == 2 { parts.append("Snow cover: 0%") }
        }
        return parts.joined(separator: " · ")
    }

    // MARK: Top bar

    private var topBar: some View {
        HStack(spacing: 12) {
            Button { onClose() } label: {
                Label("Projects", systemImage: "chevron.left")
            }
            TextField("Project name", text: $session.project.name)
                .textFieldStyle(.roundedBorder)
                .frame(width: 210)
                .focused($fieldFocus, equals: .projectName)
            Spacer()
            Button { session.undo() } label: {
                Image(systemName: "arrow.uturn.backward")
            }
            .disabled(!session.canUndo)
            .help("Undo (Cmd+Z)")
            Button { session.redo() } label: {
                Image(systemName: "arrow.uturn.forward")
            }
            .disabled(!session.canRedo)
            .help("Redo (Cmd+Shift+Z)")
            Divider().frame(height: 22)
            Button { session.toggleMode() } label: {
                Label(session.mode == .edit ? "Play" : "Pause",
                      systemImage: session.mode == .edit ? "play.fill" : "pause.fill")
            }
            .buttonStyle(.borderedProminent)
            .help("Play: walk around the scene. Pause: back to editing.")
            Menu {
                Picker("Weather", selection: weatherBinding) {
                    Text("Off").tag(0)
                    Text("Rain").tag(1)
                    Text("Snow").tag(2)
                }
                if session.project.scene.weather.wetness > 0.01
                    || session.project.scene.weather.snow > 0.01 {
                    Divider()
                    Button("Clear ground build-up") { session.clearWeatherBuildup() }
                }
            } label: {
                Label("Weather", systemImage: "cloud.rain")
            }
            Menu {
                Picker("Quality", selection: qualityBinding) {
                    Text("Auto").tag(0)
                    Text("High detail").tag(1)
                    Text("Battery saver").tag(2)
                }
            } label: {
                Label("Render", systemImage: "wand.and.stars")
            }
            Button { session.save() } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            Menu {
                Button("Export project package (.zip)") {
                    if let url = session.exportPackageURL() { shareItem = ShareItem(url: url) }
                }
                Button("Export PNG image") {
                    if let url = session.exportPNGURL() { shareItem = ShareItem(url: url) }
                }
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(white: 0.13))
    }

    // MARK: Object tree (left)

    private var treePanel: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Objects").font(.headline)
                Spacer()
                Menu {
                    Button("Box") { session.addPrimitive("box") }
                    Button("Sphere") { session.addPrimitive("sphere") }
                    Button("Cylinder") { session.addPrimitive("cylinder") }
                    Button("Plane") { session.addPrimitive("plane") }
                    Divider()
                    Button("Import Model…") { showImport = true }
                } label: {
                    Label("Add", systemImage: "plus")
                }
            }
            .padding(10)
            List(selection: $session.selection) {
                ForEach(session.project.scene.objects) { obj in
                    HStack(spacing: 8) {
                        Image(systemName: obj.source == "model" ? "cube" : icon(for: obj.primitiveKind))
                            .foregroundStyle(.secondary)
                        Text(obj.name).lineLimit(1)
                        Spacer()
                        Button {
                            session.toggleVisibility(obj.id)
                        } label: {
                            Image(systemName: obj.visible ? "eye" : "eye.slash")
                                .foregroundStyle(obj.visible ? .primary : .secondary)
                        }
                        .buttonStyle(.plain)
                    }
                    .tag(obj.id)
                }
            }
            .listStyle(.plain)
            Text("Tap an object in the view to select it.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(8)
            if session.project.scene.weather.mode != 0
                || session.project.scene.weather.wetness > 0.02
                || session.project.scene.weather.snow > 0.02 {
                Text(weatherCaption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 8)
            }
        }
        .background(Color(white: 0.11))
    }

    private func icon(for kind: String) -> String {
        switch kind {
        case "sphere": return "circle"
        case "cylinder": return "cylinder"
        case "plane": return "square"
        default: return "cube"
        }
    }

    // MARK: Viewport (centre)

    private var viewportPanel: some View {
        ZStack(alignment: .bottomLeading) {
            MetalViewport(session: session)
            VStack(alignment: .leading, spacing: 6) {
                Text(session.mode == .play
                     ? "PLAY — drag to look · WASD to walk · Esc to edit"
                     : "EDIT — drag orbit · drag selected object to move it · two-finger pan · pinch zoom · WASD/QE · F focus")
                    .font(.caption.bold())
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Color.black.opacity(0.55))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                if !session.status.isEmpty {
                    Text(session.status)
                        .font(.caption)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(Color.black.opacity(0.55))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
            .padding(10)
        }
    }

    // MARK: Properties (right)

    @ViewBuilder
    private var propertiesPanel: some View {
        if let obj = session.selectedObject {
            ObjectPropertiesView(session: session, objectID: obj.id,
                                 fieldFocus: $fieldFocus)
        } else {
            VStack(spacing: 8) {
                Image(systemName: "cursorarrow.click")
                    .font(.title2)
                    .foregroundStyle(.secondary)
                Text("Select an object to edit its name, position and material.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .font(.callout)
            }
            .padding(20)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .background(Color(white: 0.11))
        }
    }

    // MARK: Import sheet (app folder, no system picker)

    private var importSheet: some View {
        let files = ProjectStore.appFolderModelFiles()
        return NavigationStack {
            Group {
                if files.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "folder.badge.plus").font(.largeTitle)
                        Text("No model files in the app folder yet.")
                            .font(.headline)
                        Text("In the Files app, copy a .stl, .obj, .usd or .usdz model file into the LumenScene folder (On My iPad → LumenScene), then reopen this sheet.")
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.secondary)
                    }
                    .padding(30)
                } else {
                    List(files, id: \.path) { url in
                        Button {
                            showImport = false
                            session.importModelFile(url)
                        } label: {
                            HStack {
                                Image(systemName: "cube")
                                VStack(alignment: .leading) {
                                    Text(url.lastPathComponent)
                                    Text(url.pathExtension.uppercased() + " model")
                                        .font(.caption).foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Import Model from App Folder")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showImport = false }
                }
            }
        }
    }

    // MARK: Keyboard

    private func handleKey(_ press: KeyPress) -> KeyPress.Result {
        if fieldFocus != nil { return .ignored }
        // `characters` is used for letters (not `key`) so Shift —
        // which uppercases the character — still matches.
        let ch = press.characters.lowercased()
        if press.modifiers.contains(.command) {
            if ch == "z" {
                if press.phase == .down {
                    if press.modifiers.contains(.shift) { session.redo() } else { session.undo() }
                }
                return .handled
            }
            if ch == "s" {
                if press.phase == .down { session.save() }
                return .handled
            }
            return .ignored
        }
        switch press.key {
        case .delete, .deleteForward:
            if press.phase == .down { session.deleteSelected() }
            return .handled
        case .escape:
            if press.phase == .down, session.mode == .play { session.toggleMode() }
            return .handled
        default:
            break
        }
        if ch == "f" {
            if press.phase == .down { session.focusSelection() }
            return .handled
        }
        if ch == "p" {
            if press.phase == .down { session.toggleMode() }
            return .handled
        }
        var mk: MoveKey?
        switch ch {
        case "w": mk = .w
        case "a": mk = .a
        case "s": mk = .s
        case "d": mk = .d
        case "q": mk = .q
        case "e": mk = .e
        default: break
        }
        if press.key == .upArrow { mk = .up }
        else if press.key == .downArrow { mk = .down }
        else if press.key == .leftArrow { mk = .left }
        else if press.key == .rightArrow { mk = .right }
        if let mk {
            if press.phase == .down { session.pressedKeys.insert(mk) }
            else if press.phase == .up { session.pressedKeys.remove(mk) }
            return .handled
        }
        return .ignored
    }
}

// MARK: - Touch joystick (Play mode)

/// Bottom-left thumb stick for walking in Play mode. A separate
/// view on top of the viewport, so touches on it never reach the
/// viewport's drag-to-look gesture; everywhere else look-around
/// keeps working. Up on screen = forward (+y in the session's
/// touch vector).
struct JoystickView: View {
    @ObservedObject var session: EditorSession
    @State private var offset: CGSize = .zero

    private let maxTravel: CGFloat = 38

    var body: some View {
        ZStack {
            Circle()
                .fill(Color.white.opacity(0.12))
                .frame(width: 124, height: 124)
                .overlay(Circle().stroke(Color.white.opacity(0.3), lineWidth: 1.5))
            Circle()
                .fill(Color(white: 0.9).opacity(0.9))
                .frame(width: 54, height: 54)
                .offset(x: offset.width, y: offset.height)
        }
        .frame(width: 124, height: 124)
        .shadow(color: Color.black.opacity(0.35), radius: 8, x: 0, y: 2)
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    var dx = value.translation.width
                    var dy = value.translation.height
                    let len = sqrt(dx * dx + dy * dy)
                    if len > maxTravel {
                        dx = dx / len * maxTravel
                        dy = dy / len * maxTravel
                    }
                    offset = CGSize(width: dx, height: dy)
                    session.touchVector = SIMD2<Float>(Float(dx / maxTravel),
                                                       Float(-dy / maxTravel))
                }
                .onEnded { _ in
                    offset = .zero
                    session.touchVector = .zero
                }
        )
    }
}

// MARK: - Properties panel for one object

struct ObjectPropertiesView: View {
    @ObservedObject var session: EditorSession
    let objectID: UUID
    var fieldFocus: FocusState<FieldFocus?>.Binding

    @State private var nameText: String = ""
    @State private var customTextures: [CustomTextureStore.CustomTextureMeta] = []

    private var obj: Binding<SceneObject> { session.objectBinding(objectID) }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                TextField("Name", text: $nameText)
                    .textFieldStyle(.roundedBorder)
                    .focused(fieldFocus, equals: .objectName)
                    .onSubmit {
                        if nameText != obj.wrappedValue.name {
                            session.beginEdit()
                            obj.wrappedValue.name = nameText
                        }
                    }

                section("Position") {
                    axisRow("X", comp(\.position, 0), -20...20)
                    axisRow("Y", comp(\.position, 1), -20...20)
                    axisRow("Z", comp(\.position, 2), -20...20)
                }
                section("Rotation") {
                    axisRow("X°", comp(\.rotation, 0), -180...180, "%.0f")
                    axisRow("Y°", comp(\.rotation, 1), -180...180, "%.0f")
                    axisRow("Z°", comp(\.rotation, 2), -180...180, "%.0f")
                }
                section("Scale") {
                    axisRow("X", comp(\.scale, 0), 0.05...5)
                    axisRow("Y", comp(\.scale, 1), 0.05...5)
                    axisRow("Z", comp(\.scale, 2), 0.05...5)
                }

                if obj.wrappedValue.source == "primitive" {
                    section("Size") { primitiveSizeRows }
                } else {
                    section("Imported") {
                        if let asset = obj.wrappedValue.assetFile {
                            Text(asset).font(.caption).foregroundStyle(.secondary)
                        }
                        if let mesh = session.meshes[objectID] {
                            Text("\(mesh.triangleCount) triangles")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }

                section("Surface") {
                    ColorPicker("Colour", selection: colorOf(
                        { $0.material.baseColor }, { $0.material.baseColor = $1 }))
                    Picker("Texture", selection: presetBinding) {
                        Text("Plain").tag("none")
                        Section("Surfaces") {
                            Text("Concrete").tag("concrete")
                            Text("Cement").tag("cement")
                            Text("Brick").tag("brick")
                        }
                        Section("Roads") {
                            ForEach(TextureGen.roadNames, id: \.self) { name in
                                Text(TextureGen.displayName(name)).tag(name)
                            }
                        }
                        if !customTextures.isEmpty {
                            Section("My Textures") {
                                ForEach(customTextures) { meta in
                                    Text(meta.name).tag("custom:\(meta.id.uuidString)")
                                }
                            }
                        }
                    }
                    if obj.wrappedValue.material.texturePreset != "none"
                        && !obj.wrappedValue.material.textureFit {
                        sliderRow("Tiling", dbl({ $0.material.textureScale },
                                                { $0.material.textureScale = $1 }),
                                  0.25...8)
                    }
                    if obj.wrappedValue.material.texturePreset != "none"
                        && obj.wrappedValue.material.textureFit {
                        Text("Auto-fit: this texture spans the whole face it is applied to.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                section("Light Emission") {
                    Toggle("Emit light", isOn: boolOf(
                        { $0.material.emissiveOn }, { $0.material.emissiveOn = $1 }))
                    if obj.wrappedValue.material.emissiveOn {
                        ColorPicker("Light colour", selection: colorOf(
                            { $0.material.emissiveColor }, { $0.material.emissiveColor = $1 }))
                        sliderRow("Intensity", dbl({ $0.material.emissiveIntensity },
                                                   { $0.material.emissiveIntensity = $1 }),
                                  0...20, "%.1f")
                        Text("The object lights the scene in this colour.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                section("Glass") {
                    Toggle("Make glass", isOn: boolOf(
                        { $0.material.glassOn }, { $0.material.glassOn = $1 }))
                    if obj.wrappedValue.material.glassOn {
                        sliderRow("Transmission", dbl({ $0.material.transmission },
                                                      { $0.material.transmission = $1 }), 0...1)
                        sliderRow("Matte", dbl({ $0.material.roughness },
                                               { $0.material.roughness = $1 }), 0...1)
                        sliderRow("IOR", dbl({ $0.material.ior },
                                             { $0.material.ior = $1 }), 1.0...2.0, "%.2f")
                        ColorPicker("Tint", selection: colorOf(
                            { $0.material.tint }, { $0.material.tint = $1 }))
                        Text("Matte frosts the glass: reflections blur and less light gets through cleanly.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                Toggle("Visible", isOn: boolOf({ $0.visible }, { $0.visible = $1 }))

                Button(role: .destructive) {
                    session.selection = objectID
                    session.deleteSelected()
                } label: {
                    Label("Delete Object", systemImage: "trash")
                }
            }
            .padding(14)
        }
        .background(Color(white: 0.11))
        .onAppear {
            nameText = obj.wrappedValue.name
            customTextures = CustomTextureStore.list()
        }
        .onChange(of: objectID) { _, _ in nameText = obj.wrappedValue.name }
    }

    // MARK: sections & rows

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.bold())
                .foregroundStyle(.secondary)
            content()
        }
    }

    private func axisRow(_ label: String, _ value: Binding<Double>,
                         _ range: ClosedRange<Double>, _ format: String = "%.2f") -> some View {
        HStack(spacing: 8) {
            Text(label).frame(width: 26, alignment: .leading)
            Slider(value: value, in: range) { editing in
                // Transform drags run the manipulation preview:
                // a moved light/glass object renders cheap until
                // the slider is released.
                session.transformDragChanged(editing, objectID: objectID)
            }
            Text(String(format: format, value.wrappedValue))
                .frame(width: 46, alignment: .trailing)
                .monospacedDigit()
        }
        .font(.callout)
    }

    private func sliderRow(_ label: String, _ value: Binding<Double>,
                           _ range: ClosedRange<Double>, _ format: String = "%.2f",
                           transform: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: format, value.wrappedValue)).monospacedDigit()
            }
            .font(.callout)
            Slider(value: value, in: range) { editing in
                if transform {
                    session.transformDragChanged(editing, objectID: objectID)
                } else if editing {
                    session.beginEdit()
                }
            }
        }
    }

    @ViewBuilder
    private var primitiveSizeRows: some View {
        let kind = obj.wrappedValue.primitiveKind
        switch kind {
        case "sphere":
            sliderRow("Radius", param("r"), 0.05...5, transform: true)
        case "cylinder":
            sliderRow("Radius", param("r"), 0.05...5, transform: true)
            sliderRow("Height", param("h"), 0.05...10, transform: true)
        case "plane":
            sliderRow("Width", param("w"), 0.5...30, transform: true)
            sliderRow("Depth", param("d"), 0.5...30, transform: true)
        default:
            sliderRow("Width", param("w"), 0.05...10, transform: true)
            sliderRow("Height", param("h"), 0.05...10, transform: true)
            sliderRow("Depth", param("d"), 0.05...10, transform: true)
        }
    }

    // MARK: binding helpers

    private func comp(_ kp: WritableKeyPath<Transform3, [Double]>, _ i: Int) -> Binding<Double> {
        Binding(
            get: { obj.wrappedValue.transform[keyPath: kp][i] },
            set: { v in
                var o = obj.wrappedValue
                o.transform[keyPath: kp][i] = v
                obj.wrappedValue = o
            })
    }

    private func param(_ key: String) -> Binding<Double> {
        Binding(
            get: { obj.wrappedValue.params[key] ?? 1 },
            set: { v in
                var o = obj.wrappedValue
                o.params[key] = v
                obj.wrappedValue = o
            })
    }

    private func dbl(_ get: @escaping (SceneObject) -> Double,
                     _ set: @escaping (inout SceneObject, Double) -> Void) -> Binding<Double> {
        Binding(get: { get(obj.wrappedValue) },
                set: { v in var o = obj.wrappedValue; set(&o, v); obj.wrappedValue = o })
    }

    private func boolOf(_ get: @escaping (SceneObject) -> Bool,
                        _ set: @escaping (inout SceneObject, Bool) -> Void) -> Binding<Bool> {
        Binding(get: { get(obj.wrappedValue) },
                set: { v in
                    session.beginEditIfNewGesture()
                    var o = obj.wrappedValue; set(&o, v); obj.wrappedValue = o
                })
    }

    private func colorOf(_ get: @escaping (SceneObject) -> RGB,
                         _ set: @escaping (inout SceneObject, RGB) -> Void) -> Binding<Color> {
        Binding(
            get: {
                let c = get(obj.wrappedValue)
                return Color(red: c.r, green: c.g, blue: c.b)
            },
            set: { newColor in
                session.beginEditIfNewGesture()
                var o = obj.wrappedValue
                set(&o, RGB(newColor))
                obj.wrappedValue = o
            })
    }

    private var presetBinding: Binding<String> {
        Binding(get: { obj.wrappedValue.material.texturePreset },
                set: { v in
                    session.beginEditIfNewGesture()
                    var o = obj.wrappedValue
                    o.material.texturePreset = v
                    o.material.textureFit = TextureGen.isRoad(v)
                    obj.wrappedValue = o
                })
    }
}

extension RGB {
    init(_ color: Color) {
        let ui = UIColor(color)
        var r: CGFloat = 0, g: CGFloat = 0, b: CGFloat = 0, a: CGFloat = 0
        ui.getRed(&r, green: &g, blue: &b, alpha: &a)
        self.init(Double(r), Double(g), Double(b))
    }
}
