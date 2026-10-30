// LumenSceneApp.swift — app entry and the root switch between the
// project home screen and the full-screen editor.

import SwiftUI

@main
struct LumenSceneApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup {
            RootView()
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
