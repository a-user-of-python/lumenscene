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
    @State private var showNodes = false
    @State private var showSnowSettings = false
    @State private var shareItem: ShareItem?
    /// v4.2: the top bar switches to its compact row (icon-only
    /// back button, Weather folded into the More menu, narrower
    /// title field) when the window is too narrow for the full
    /// row — driven by the bar's measured width, not a guess.
    @State private var barCompact = false
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
        .sheet(isPresented: $showNodes) { NodeEditorView(session: session) }
        .sheet(isPresented: $session.finalRenderPresented, onDismiss: {
            // v4.10 (WS2): swiping the sheet away mid-render is
            // Cancel-then-dismiss — the loop notices between
            // chunks and stops; the in-progress image is
            // discarded with the sheet.
            if session.finalRenderPhase == .rendering { session.cancelFinalRender() }
        }) { FinalRenderSheetView(session: session) }
        .sheet(isPresented: $showSnowSettings) { snowSettingsSheet }
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
                    .frame(width: 264)
                Divider()
                viewportPanel
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                propertiesPanel
                    .frame(width: 320)
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
        // v4.0: the runtime's message banner — display-only and
        // transient; the pause button and joystick remain the only
        // Play controls.
        .overlay(alignment: .top) {
            if !session.playBanner.isEmpty {
                Text(session.playBanner)
                    .font(.headline)
                    .foregroundStyle(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 18)
                    .padding(.vertical, 10)
                    .background(Color.black.opacity(0.55))
                    .clipShape(Capsule())
                    .overlay(Capsule().stroke(Color.white.opacity(0.25), lineWidth: 1))
                    .padding(.top, 16)
                    .allowsHitTesting(false)
            }
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

    /// v4.8 (steering 1): water reflection resolution —
    /// Full (1.0, v4.7 quality) / Half / Quarter, mapped to
    /// the session's stepped waterReflectionScale.
    private var waterReflectionBinding: Binding<Int> {
        Binding(
            get: {
                switch session.waterReflectionScale {
                case 0.5: return 1
                case 0.25: return 2
                default: return 0
                }
            },
            set: { newValue in
                session.setWaterReflectionScale([1.0, 0.5, 0.25][newValue])
            })
    }

    /// v4.10 (WS1): idle-freeze delay — Off / 15 s / 30 s /
    /// 1 min / 2 min / 5 min, tags 0-5 mapped to seconds.
    /// The underlying value lives on the renderer (in-memory
    /// preference, like Quality and Water reflections).
    private var idleFreezeBinding: Binding<Int> {
        Binding(
            get: {
                switch session.idleFreezeSeconds {
                case 15: return 1
                case 30: return 2
                case 60: return 3
                case 120: return 4
                case 300: return 5
                default: return 0
                }
            },
            set: { newValue in
                let steps: [Double] = [0, 15, 30, 60, 120, 300]
                session.idleFreezeSeconds = steps[min(max(newValue, 0), 5)]
            })
    }

    private var weatherBinding: Binding<Int> {
        Binding(
            get: { session.project.scene.weather.mode },
            set: { session.setWeatherMode($0) })
    }

    // v4.9: the weather caption that used to be computed here
    // moved into EditorSession.refreshFooter() (identical
    // expression), feeding the footer's narrow FooterDisplay
    // observable — see TreeFooterView below.

    // MARK: Top bar

    // v4.2 TOP BAR, rebuilt. The v4.1 bar carried ten controls
    // whose combined intrinsic width met or exceeded the bar's
    // width at Sean's 1180 pt layout, the title field was a
    // fixed, inflexible 210 pt, and NOT ONE label had lineLimit
    // or fixedSize discipline — so when SwiftUI ran out of
    // width it took its lowest-resistance path and WRAPPED the
    // label text mid-word, doubling the bar's height into tall
    // narrow pills ("Proje/cts", "No/de/s", "Ex-/port"…).
    // The fix attacks both halves: (1) less demand — the title
    // field is flexible (120–230 pt, compact 84–230) and the
    // two secondary menus (Render quality, Export) fold into
    // one trailing More menu; (2) wrapping made structurally
    // impossible — every label is lineLimit(1) and every
    // control fixedSize horizontally, so a control is always
    // its natural single-line size; (3) a compact row (icon
    // back button, Weather inside More) engages below 1020 pt
    // measured width for split-view widths. Primary actions
    // (Play, Nodes, Save, Weather in the full row) stay one
    // tap; everything else is at most one extra tap via More.
    private var topBar: some View {
        HStack(spacing: 10) {
            Button { onClose() } label: {
                if barCompact {
                    Image(systemName: "chevron.left")
                } else {
                    Label("Projects", systemImage: "chevron.left")
                        .lineLimit(1)
                }
            }
            .fixedSize(horizontal: true, vertical: false)
            .help("Back to projects")
            TextField("Project name", text: $session.project.name)
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: barCompact ? 84 : 120,
                       idealWidth: barCompact ? 130 : 200,
                       maxWidth: 230)
                .focused($fieldFocus, equals: .projectName)
            Spacer(minLength: 8)
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
                    .lineLimit(1)
            }
            .buttonStyle(.borderedProminent)
            .fixedSize(horizontal: true, vertical: false)
            .help("Play: walk around the scene. Pause: back to editing.")
            Button { showNodes = true } label: {
                Label("Nodes", systemImage: "point.3.connected.trianglepath.dotted")
                    .lineLimit(1)
            }
            .fixedSize(horizontal: true, vertical: false)
            .help("Node editor: logic that runs in Play.")
            if !barCompact {
                Menu {
                    weatherMenuItems
                } label: {
                    Label("Weather", systemImage: "cloud.rain")
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
            }
            Button { session.save() } label: {
                Label("Save", systemImage: "square.and.arrow.down")
                    .lineLimit(1)
            }
            .fixedSize(horizontal: true, vertical: false)
            Menu {
                if barCompact {
                    weatherMenuItems
                    Divider()
                }
                Picker("Quality", selection: qualityBinding) {
                    Text("Auto").tag(0)
                    Text("High detail").tag(1)
                    Text("Battery saver").tag(2)
                }
                Picker("Water reflections", selection: waterReflectionBinding) {
                    Text("Full").tag(0)
                    Text("Half").tag(1)
                    Text("Quarter").tag(2)
                }
                Picker("Idle freeze", selection: idleFreezeBinding) {
                    Text("Off").tag(0)
                    Text("15 s").tag(1)
                    Text("30 s").tag(2)
                    Text("1 min").tag(3)
                    Text("2 min").tag(4)
                    Text("5 min").tag(5)
                }
                Divider()
                Button("Render Image…") { session.startFinalRender() }
                Button("Export project package (.zip)") {
                    if let url = session.exportPackageURL() { shareItem = ShareItem(url: url) }
                }
                Button("Export PNG image") {
                    if let url = session.exportPNGURL() { shareItem = ShareItem(url: url) }
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .fixedSize(horizontal: true, vertical: false)
            .help("Render quality and export")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color(white: 0.13))
        // Width probe for the compact switch (iOS 17-compatible:
        // onGeometryChange needs iOS 18). The state write happens
        // in onAppear/onChange, never during body evaluation.
        .background(
            GeometryReader { proxy in
                Color.clear
                    .onAppear { barCompact = proxy.size.width < 1020 }
                    .onChange(of: proxy.size.width) { _, width in
                        barCompact = width < 1020
                    }
            }
        )
    }

    /// The Weather menu's contents, shared by the full row's
    /// Weather menu and the compact row's More menu.
    @ViewBuilder
    private var weatherMenuItems: some View {
        Picker("Weather", selection: weatherBinding) {
            Text("Off").tag(0)
            Text("Rain").tag(1)
            Text("Snow").tag(2)
        }
        Button("Snow settings…") { showSnowSettings = true }
        if session.project.scene.weather.wetness > 0.01
            || session.project.scene.weather.snow > 0.01 {
            Divider()
            Button("Clear ground build-up") { session.clearWeatherBuildup() }
        }
    }

    /// v4.3: binding for the Max Snow Depth slider — writes go
    /// through the session's explicit settings call (A preserved,
    /// renderer depth + geometry rescheduled there).
    private var maxSnowDepthBinding: Binding<Double> {
        Binding(
            get: { session.project.scene.weather.maxDepth },
            set: { session.setMaxSnowDepth($0) })
    }

    /// v4.7: binding for the Temperature slider — writes go
    /// through the session's explicit settings call.
    private var temperatureBinding: Binding<Double> {
        Binding(
            get: { session.project.scene.weather.temperature ?? 10.0 },
            set: { session.setTemperature($0) })
    }

    /// v4.3: the Snow settings sheet (from the Weather menu):
    /// the user-set Max Snow Depth, the per-project snow pattern
    /// seed with a re-roll, and the live depth readout.
    private var snowSettingsSheet: some View {
        NavigationStack {
            Form {
                Section {
                    if session.project.scene.weather.temperature == nil {
                        Text("Not set — rain and snow behave as they always have.")
                            .foregroundStyle(.secondary)
                        Button("Set temperature") { session.setTemperature(10.0) }
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    } else {
                        HStack {
                            Text("Temperature")
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                            Spacer()
                            Text(session.liveTemperatureText)
                                .foregroundStyle(.secondary)
                                .monospacedDigit()
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        Slider(value: temperatureBinding, in: -20...30, step: 0.5)
                        Text("At or below freezing, rain falls as snow and puddles freeze into ice. Above freezing, snow falls as rain and the snowpack melts, faster the warmer it is.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Button("Clear temperature") { session.setTemperature(nil) }
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                } header: {
                    Text("Temperature").lineLimit(1).fixedSize(horizontal: true, vertical: false)
                }
                Section {
                    HStack {
                        Text("Max Snow Depth")
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                        Spacer()
                        Text(String(format: "%.2f m", session.project.scene.weather.maxDepth))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    Slider(value: maxSnowDepthBinding, in: 0.05...2.0, step: 0.05)
                    Text("Snow keeps falling until it reaches this depth. Lowering it lets deep snow settle down gradually — it isn't removed instantly.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } header: {
                    Text("Depth").lineLimit(1).fixedSize(horizontal: true, vertical: false)
                }
                Section {
                    HStack {
                        Text("Pattern")
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                        Spacer()
                        Text(session.project.scene.weather.seed == 0
                             ? "Pattern #0 (Classic)"
                             : "Pattern #\(session.project.scene.weather.seed)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    Button("Re-roll pattern") { session.rerollSnowSeed() }
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                } header: {
                    Text("Snow Pattern").lineLimit(1).fixedSize(horizontal: true, vertical: false)
                }
                Section {
                    Text(String(format: "Snow now: %.2f m of %.2f m max (%d%%)",
                                session.liveSnowAmount,
                                session.project.scene.weather.maxDepth,
                                Int((session.liveSnowProgress * 100).rounded())))
                        .monospacedDigit()
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                } header: {
                    Text("Right Now").lineLimit(1).fixedSize(horizontal: true, vertical: false)
                }
            }
            .navigationTitle("Snow Settings")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { showSnowSettings = false }
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
            }
        }
    }

    /// v4.6: tree taps route through here — in Multi Select
    /// mode a row tap toggles membership instead of replacing
    /// the selection.
    private var treeSelectionBinding: Binding<UUID?> {
        Binding(
            get: { session.selection },
            set: { newValue in
                if session.multiSelectMode {
                    if let id = newValue { session.toggleMultiSelection(id) }
                } else {
                    session.selection = newValue
                }
            })
    }

    // MARK: Object tree (left)

    private var treePanel: some View {
        // v4.9: per-area member counts in ONE pass over the
        // scene per treePanel evaluation. Previously every
        // area row re-filtered all objects (O(areas × objects)
        // per body pass); rows now read this map.
        let areaMemberCounts: [UUID: Int] = {
            var counts: [UUID: Int] = [:]
            for obj in session.project.scene.objects {
                if let areaID = obj.areaID {
                    counts[areaID, default: 0] += 1
                }
            }
            return counts
        }()
        return VStack(spacing: 0) {
            HStack {
                Text("Objects").font(.headline).lineLimit(1).fixedSize(horizontal: true, vertical: false)
                Spacer()
                // v4.6: Multi Select tool toggle. On: taps (tree
                // or viewport) add/remove objects from the set;
                // off: the set clears (switching tools clears).
                Button {
                    session.multiSelectMode.toggle()
                } label: {
                    Label("Multi", systemImage: session.multiSelectMode
                          ? "checklist.checked" : "checklist")
                        .labelStyle(.titleAndIcon)
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .tint(session.multiSelectMode ? .accentColor : .secondary)
                .help("Multi Select: tap several objects to select them together")
                Menu {
                    Button("Box") { session.addPrimitive("box") }
                    Button("Sphere") { session.addPrimitive("sphere") }
                    Button("Cylinder") { session.addPrimitive("cylinder") }
                    Button("Plane") { session.addPrimitive("plane") }
                    Divider()
                    Button("Trigger Area") { session.addArea() }
                    Divider()
                    Button("Import Model…") { showImport = true }
                } label: {
                    Label("Add", systemImage: "plus")
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
            }
            .padding(10)
            List(selection: treeSelectionBinding) {
                ForEach(session.project.scene.objects) { obj in
                    HStack(spacing: 8) {
                        // v4.6: set membership mark in the tree —
                        // every selected object shows it, not
                        // just the primary row highlight.
                        if session.multiSelectMode
                            && session.multiSelection.contains(obj.id) {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(Color.accentColor)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        Image(systemName: obj.source == "model" ? "cube" : icon(for: obj.primitiveKind))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: true, vertical: false)
                        Text(obj.name).lineLimit(1)
                        Spacer()
                        Button {
                            session.toggleVisibility(obj.id)
                        } label: {
                            Image(systemName: obj.visible ? "eye" : "eye.slash")
                                .foregroundStyle(obj.visible ? .primary : .secondary)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        .buttonStyle(.plain)
                        .fixedSize(horizontal: true, vertical: false)
                    }
                    .tag(obj.id)
                }
            }
            .listStyle(.plain)
            Divider()
            HStack {
                Text("Areas").font(.headline).lineLimit(1).fixedSize(horizontal: true, vertical: false)
                Spacer()
                Button { session.addArea() } label: {
                    Image(systemName: "plus")
                        .fixedSize(horizontal: true, vertical: false)
                }
                .fixedSize(horizontal: true, vertical: false)
                .help("Add a trigger area")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            if session.project.scene.areas.isEmpty {
                Text("No trigger areas. Areas group objects for node logic and streaming.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.bottom, 6)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                ScrollView {
                    VStack(spacing: 2) {
                        ForEach(session.project.scene.areas, id: \.id) { area in
                            HStack(spacing: 8) {
                                Image(systemName: "cube.transparent")
                                    .foregroundStyle(.secondary)
                                    .fixedSize(horizontal: true, vertical: false)
                                Text(area.name).lineLimit(1)
                                Spacer()
                                Text("\(areaMemberCounts[area.id] ?? 0)")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                    .fixedSize(horizontal: true, vertical: false)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(session.selectedArea == area.id
                                        ? Color.accentColor.opacity(0.25) : Color.clear)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .contentShape(Rectangle())
                            .onTapGesture {
                                session.selectedArea = area.id
                                session.selection = nil
                            }
                        }
                    }
                    .padding(.horizontal, 4)
                }
                .frame(maxHeight: 132)
            }
            // v4.9: the footer (hint + weather lines) is its
            // own view observing ONLY the session's narrow
            // FooterDisplay — see TreeFooterView. Rendered
            // lines, order and styling are unchanged.
            TreeFooterView(footer: session.footer)
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
        if let areaID = session.selectedArea,
           session.project.scene.areas.contains(where: { $0.id == areaID }) {
            AreaPropertiesView(session: session, areaID: areaID)
        } else if session.multiSelectMode && session.multiSelection.count > 1 {
            MultiSelectionPropertiesView(session: session)
        } else if let obj = session.selectedObject {
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
        // v4.10 (WS1): ANY key press (down or up) is user
        // input — it wakes a frozen scene and resets the idle
        // clock, before any handling below.
        session.noteUserInput()
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
            if press.phase == .down {
                if session.mode == .play { session.toggleMode() }
                else if session.multiSelectMode { session.multiSelectMode = false }
            }
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

// MARK: - Final render sheet (v4.10, WS2)

/// The Render Image sheet: a live preview of the progressive
/// final render with sample progress and Cancel while it runs;
/// when done (or cancelled — the best image so far is kept),
/// the finished image with Save to Photos / Share / Save to
/// Files / Done. Natural proportions throughout, every label
/// lineLimit(1) + fixedSize per the v4.8 panel discipline.
struct FinalRenderSheetView: View {
    @ObservedObject var session: EditorSession
    @State private var shareItem: ShareItem?

    private var progressPercent: Int {
        guard session.finalRenderTotalSamples > 0 else { return 0 }
        return Int((Double(session.finalRenderSamplesDone)
                    / Double(session.finalRenderTotalSamples) * 100).rounded())
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                if let image = session.finalRenderImage ?? session.finalRenderPreview {
                    Image(image, scale: 1.0, label: Text("Final render"))
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 10) {
                        ProgressView()
                        Text("Rendering…")
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                if session.finalRenderPhase == .rendering {
                    ProgressView(value: Double(session.finalRenderSamplesDone),
                                 total: Double(max(session.finalRenderTotalSamples, 1)))
                    HStack {
                        Text("\(session.finalRenderSamplesDone.formatted()) / \(session.finalRenderTotalSamples.formatted()) samples")
                            .monospacedDigit()
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                        Spacer()
                        Text("\(progressPercent)%")
                            .monospacedDigit()
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                    Button("Cancel Render") { session.cancelFinalRender() }
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
                if !session.finalRenderStatusText.isEmpty {
                    Text(session.finalRenderStatusText)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
                if session.finalRenderPhase == .done {
                    HStack(spacing: 12) {
                        Button("Save to Photos") { session.saveFinalRenderToPhotos() }
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                        Button("Share") {
                            if let url = session.finalRenderShareURL() {
                                shareItem = ShareItem(url: url)
                            }
                        }
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                        Button("Save to Files") { session.saveFinalRenderToFiles() }
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                        Spacer()
                        Button("Done") { session.finalRenderPresented = false }
                            .lineLimit(1)
                            .fixedSize(horizontal: true, vertical: false)
                    }
                } else if session.finalRenderPhase == .failed {
                    Button("Done") { session.finalRenderPresented = false }
                        .lineLimit(1)
                        .fixedSize(horizontal: true, vertical: false)
                }
            }
            .padding(20)
            .navigationTitle("Render Image")
        }
        .sheet(item: $shareItem) { item in ShareSheet(url: item.url) }
    }
}

// MARK: - Tree panel footer (v4.9: narrow observation)

/// The tree panel's footer: the Multi Select hint, the weather
/// cover caption, the snow-depth line and the temperature
/// status line — in that order, with the pre-v4.9 fonts,
/// colours and paddings byte-identical (the caption's bottom
/// padding is 2 when the snow-depth line follows it, else 8;
/// the depth line's is 8 when no temperature line follows,
/// else 2). This view observes ONLY the session's
/// FooterDisplay, which the session updates change-only, so
/// unrelated @Published churn in the session no longer
/// re-evaluates the footer at all.
struct TreeFooterView: View {
    @ObservedObject var footer: FooterDisplay

    var body: some View {
        VStack(spacing: 0) {
            Text(footer.hintText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(8)
            if let weatherText = footer.weatherText {
                Text(weatherText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .padding(.horizontal, 8)
                    .padding(.bottom, footer.snowDepthText != nil ? 2 : 8)
            }
            // v4.3: the absolute depth under the cover percent —
            // "Snow depth: 0.18 m of 0.30 m max".
            if let snowDepthText = footer.snowDepthText {
                Text(snowDepthText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .padding(.horizontal, 8)
                    .padding(.bottom, footer.temperatureText == nil ? 8 : 2)
            }
            // v4.7: temperature status, only when temperature
            // is set — e.g. "−4.5 °C — puddles freezing".
            if let temperatureText = footer.temperatureText {
                Text(temperatureText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 8)
            }
        }
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
                    session.noteUserInput()
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
                    session.noteUserInput()
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
                            Text(asset).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                        }
                        if let mesh = session.meshes[objectID] {
                            Text("\(mesh.triangleCount) triangles")
                                .font(.caption).foregroundStyle(.secondary)
                                .lineLimit(1)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                    }
                }

                section("Surface") {
                    ColorPicker("Colour", selection: colorOf(
                        { $0.material.baseColor }, { $0.material.baseColor = $1 }))
                        .lineLimit(1)
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
                    .lineLimit(1)
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
                        .lineLimit(1)
                    if obj.wrappedValue.material.emissiveOn {
                        ColorPicker("Light colour", selection: colorOf(
                            { $0.material.emissiveColor }, { $0.material.emissiveColor = $1 }))
                            .lineLimit(1)
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
                        .lineLimit(1)
                    if obj.wrappedValue.material.glassOn {
                        sliderRow("Transmission", dbl({ $0.material.transmission },
                                                      { $0.material.transmission = $1 }), 0...1)
                        sliderRow("Matte", dbl({ $0.material.roughness },
                                               { $0.material.roughness = $1 }), 0...1)
                        sliderRow("IOR", dbl({ $0.material.ior },
                                             { $0.material.ior = $1 }), 1.0...2.0, "%.2f")
                        ColorPicker("Tint", selection: colorOf(
                            { $0.material.tint }, { $0.material.tint = $1 }))
                            .lineLimit(1)
                        Text("Matte frosts the glass: reflections blur and less light gets through cleanly.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                Toggle("Visible", isOn: boolOf({ $0.visible }, { $0.visible = $1 }))
                    .lineLimit(1)

                section("Area") {
                    Menu {
                        Button("None") { setObjectArea(nil) }
                        ForEach(session.project.scene.areas, id: \.id) { area in
                            Button(area.name) { setObjectArea(area.id) }
                        }
                    } label: {
                        HStack {
                            Text(currentAreaName)
                                .foregroundStyle(.primary)
                                .lineLimit(1)
                            Spacer()
                            Image(systemName: "chevron.up.chevron.down")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: true, vertical: false)
                        }
                        .font(.callout)
                        .padding(.vertical, 4)
                    }
                    Text("Areas group objects for node triggers and streaming in Play.")
                        .font(.caption).foregroundStyle(.secondary)
                }

                Button(role: .destructive) {
                    session.selection = objectID
                    session.deleteSelected()
                } label: {
                    Label("Delete Object", systemImage: "trash")
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
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
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            content()
        }
    }

    private func axisRow(_ label: String, _ value: Binding<Double>,
                         _ range: ClosedRange<Double>, _ format: String = "%.2f") -> some View {
        HStack(spacing: 8) {
            Text(label)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .frame(width: 26, alignment: .leading)
            Slider(value: value, in: range) { editing in
                // Transform drags run the manipulation preview:
                // a moved light/glass object renders cheap until
                // the slider is released.
                session.transformDragChanged(editing, objectID: objectID)
            }
            Text(String(format: format, value.wrappedValue))
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .frame(width: 54, alignment: .trailing)
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
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                Spacer()
                Text(String(format: format, value.wrappedValue))
                    .monospacedDigit()
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
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

    private var currentAreaName: String {
        guard let areaID = obj.wrappedValue.areaID else { return "None" }
        return session.project.scene.areas.first { $0.id == areaID }?.name ?? "None"
    }

    private func setObjectArea(_ areaID: UUID?) {
        session.beginEdit()
        var o = obj.wrappedValue
        o.areaID = areaID
        obj.wrappedValue = o
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

// MARK: - Properties panel for one trigger area

/// Area editor shown in the properties panel while an area is
/// selected: name, centre and size, member count, delete. Areas
/// are not undoable (session precedent), so edits write straight
/// through session.updateArea — no beginEdit().
/// v4.6: the properties panel for a multi-selection (2+
/// objects). Deliberately small: what is group-safe (move by
/// dragging in the view, visibility, delete, clear) is here;
/// single-object editing (name, transform numbers, material,
/// rotate) is NOT — it applies to one object at a time, and
/// pretending otherwise would silently edit only the primary.
/// The panel says so in plain words.
struct MultiSelectionPropertiesView: View {
    @ObservedObject var session: EditorSession

    var body: some View {
        let count = session.multiSelection.count
        // v4.9: single filter pass, memoised in the session on
        // (multiSelection, sceneRevision) — repeated body
        // passes from unrelated published churn reuse the
        // resolved members instead of re-filtering the scene.
        let members = session.multiSelectionMembers()
        let anyVisible = members.contains { $0.visible }
        return ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Label("\(count) objects selected", systemImage: "checklist.checked")
                    .font(.headline)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                Text(members.map(\.name).joined(separator: ", "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(4)
                Divider()
                Text("Drag any selected object in the view to move them all together. Each one still pushes snow on its own when dragged through it.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Button {
                    session.toggleVisibilityForMultiSelection()
                } label: {
                    Label(anyVisible ? "Hide All" : "Show All",
                          systemImage: anyVisible ? "eye.slash" : "eye")
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
                Button(role: .destructive) {
                    session.deleteMultiSelection()
                } label: {
                    Label("Delete All \(count) Objects", systemImage: "trash")
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
                Button {
                    session.clearMultiSelection()
                } label: {
                    Label("Clear Selection", systemImage: "xmark.circle")
                        .lineLimit(1)
                }
                .fixedSize(horizontal: true, vertical: false)
                Divider()
                Text("Name, position, rotation and material editing work on one object at a time. Turn Multi off (or tap a single object with Multi off) to edit those.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .background(Color(white: 0.11))
    }
}

struct AreaPropertiesView: View {
    @ObservedObject var session: EditorSession
    let areaID: UUID

    private var area: TriggerArea? {
        session.project.scene.areas.first { $0.id == areaID }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let area {
                    TextField("Name", text: nameBinding(area))
                        .textFieldStyle(.roundedBorder)

                    section("Centre") {
                        axisRow("X", comp(area, \.center, 0), -40...40)
                        axisRow("Y", comp(area, \.center, 1), -10...40)
                        axisRow("Z", comp(area, \.center, 2), -40...40)
                    }
                    section("Size") {
                        axisRow("X", comp(area, \.size, 0), 0.2...40)
                        axisRow("Y", comp(area, \.size, 1), 0.2...40)
                        axisRow("Z", comp(area, \.size, 2), 0.2...40)
                    }

                    let members = session.project.scene.objects.filter { $0.areaID == areaID }
                    section("Members") {
                        if members.isEmpty {
                            Text("No objects yet. Assign objects from their properties panel (Area picker).")
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            ForEach(members) { member in
                                Text(member.name).font(.callout).lineLimit(1)
                            }
                        }
                    }

                    Text("The box is drawn in the viewport while editing. Node triggers fire when the player is inside it.")
                        .font(.caption).foregroundStyle(.secondary)

                    Button(role: .destructive) {
                        session.deleteArea(areaID)
                    } label: {
                        Label("Delete Area", systemImage: "trash")
                            .lineLimit(1)
                    }
                    .fixedSize(horizontal: true, vertical: false)
                } else {
                    Text("This area no longer exists.")
                        .foregroundStyle(.secondary)
                }
            }
            .padding(14)
        }
        .background(Color(white: 0.11))
    }

    @ViewBuilder
    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title.uppercased())
                .font(.caption.bold())
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
            content()
        }
    }

    /// Same shape as ObjectPropertiesView's axisRow: label, slider,
    /// value — duplicated here because that helper is private to
    /// the object editor.
    private func axisRow(_ label: String, _ value: Binding<Double>,
                         _ range: ClosedRange<Double>) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .frame(width: 26, alignment: .leading)
            Slider(value: value, in: range)
            Text(String(format: "%.2f", value.wrappedValue))
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                .frame(width: 54, alignment: .trailing)
                .monospacedDigit()
        }
        .font(.callout)
    }

    private func nameBinding(_ area: TriggerArea) -> Binding<String> {
        Binding(
            get: { area.name },
            set: { v in
                var a = area
                a.name = v
                session.updateArea(a)
            })
    }

    private func comp(_ area: TriggerArea,
                      _ kp: WritableKeyPath<TriggerArea, [Double]>, _ i: Int) -> Binding<Double> {
        Binding(
            get: { area[keyPath: kp][i] },
            set: { v in
                var a = area
                a[keyPath: kp][i] = v
                session.updateArea(a)
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
