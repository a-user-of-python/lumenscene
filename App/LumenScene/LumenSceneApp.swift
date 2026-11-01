// LumenSceneApp.swift — app entry and the root switch between the
// project home screen and the full-screen editor.

import SwiftUI
import UIKit

@main
struct LumenSceneApp: App {
    init() {
        // LS-REPRO: inert unless launched with `-LSRepro`
        // (see ReproDriver.swift).
        ReproDriver.bootstrapIfNeeded()
        // Gate proof marker, emitted at bootstrap like the
        // repro driver's own markers: a headless console
        // launch may never present a window, so the notice
        // view's onAppear is not a reliable signal there.
        // On iPad this never prints.
        if DeviceGate.isPhone {
            print("GATE-MARKER ipad-required idiom=phone")
            fflush(stdout)
            NSLog("GATE-MARKER ipad-required idiom=phone")
        }
    }

    var body: some SwiftUI.Scene {
        WindowGroup {
            // Device gate (Sean, 2026-11-01): LumenScene is an
            // iPad app. On an iPhone it stops here, at a clear
            // notice, before any editor UI exists. On an iPad
            // this is exactly the pre-gate root, untouched —
            // the gate sits in front of the idle-freeze and
            // final-render paths without intersecting them.
            if DeviceGate.isPhone {
                IPadRequiredView()
            } else {
                RootView()
            }
        }
    }
}

/// The launch device check: true only on an iPhone-idiom
/// device. The repro driver is unaffected — it bootstraps in
/// `init` and its rig runs on iPad-idiom simulators, where
/// this is always false.
enum DeviceGate {
    static var isPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }
}

/// The full-screen notice shown on iPhones instead of the
/// app. Headline is Sean's wording; one plain supporting
/// line; high contrast, no clutter.
struct IPadRequiredView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "ipad")
                .font(.system(size: 64))
            Text("You need an iPad for this to work")
                .font(.title)
                .fontWeight(.semibold)
                .multilineTextAlignment(.center)
            Text("LumenScene is designed for the iPad's larger screen.")
                .font(.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(32)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .preferredColorScheme(.dark)
        .onAppear {
            // Same marker as the bootstrap emission (the
            // view-level signal for interactive launches).
            print("GATE-MARKER ipad-required idiom=phone")
            fflush(stdout)
            NSLog("GATE-MARKER ipad-required idiom=phone")
        }
    }
}

struct RootView: View {
    @State private var openProjectID: UUID?
    @State private var seeded = false

    var body: some View {
        Group {
            if Renderer.shared == nil {
                VStack(spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.largeTitle)
                    Text("Metal is unavailable on this device, so LumenScene can't render.")
                        .multilineTextAlignment(.center)
                }
                .padding()
            } else if let id = openProjectID,
                      let project = ProjectStore.loadProject(id: id) {
                EditorContainer(project: project) { openProjectID = nil }
                    .id(id)
            } else {
                HomeView(onOpen: { openProjectID = $0 })
            }
        }
        .preferredColorScheme(.dark)
        .task {
            guard !seeded else { return }
            seeded = true
            if ProjectStore.listProjects().isEmpty {
                let demo = DemoScene.makeProject()
                ProjectStore.saveProject(demo, thumbnailPNG: nil)
                // Give the demo a real thumbnail immediately —
                // the home screen shouldn't show a placeholder
                // for it until someone happens to save. No editor
                // is open yet and opening a project rebuilds the
                // renderer, so borrowing it here is safe. The
                // demo is all primitives, hence the empty mesh map.
                if let renderer = Renderer.shared {
                    renderer.rebuild(scene: demo.scene, meshes: [:])
                    if let thumb = renderer.renderStillPNG(
                        camera: demo.scene.camera.orbitBasis(),
                        width: 640, height: 480, samples: 48) {
                        ProjectStore.saveProject(demo, thumbnailPNG: thumb)
                    }
                }
            }
        }
    }
}

/// Creates the session for one open project. Keyed by the editor's
/// `.id(projectID)` in RootView so switching projects rebuilds it.
struct EditorContainer: View {
    @StateObject private var session: EditorSession
    let onClose: () -> Void

    init(project: Project, onClose: @escaping () -> Void) {
        // RootView only constructs this when Renderer.shared != nil.
        _session = StateObject(wrappedValue: EditorSession(
            project: project, renderer: Renderer.shared!))
        self.onClose = onClose
    }

    var body: some View {
        EditorView(session: session, onClose: onClose)
    }
}
