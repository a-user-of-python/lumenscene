// HomeView.swift — the app's front page: the project list. Each
// card shows the project's name and the thumbnail rendered from
// the editor's view at its last save.

import SwiftUI

struct HomeView: View {
    var onOpen: (UUID) -> Void

    @State private var metas: [ProjectMeta] = []
    @State private var showDesigner = false
    @State private var renameTarget: ProjectMeta?
    @State private var renameText: String = ""
    @State private var deleteTarget: ProjectMeta?

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("LumenScene")
                        .font(.largeTitle.bold())
                    Text("Ray-traced 3D scenes")
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    showDesigner = true
                } label: {
                    Label("Texture Designer", systemImage: "paintpalette")
                }
                .buttonStyle(.bordered)
                Button {
                    let p = Project(name: "Untitled Project")
                    ProjectStore.saveProject(p, thumbnailPNG: nil)
                    reload()
                    onOpen(p.id)
                } label: {
                    Label("New Project", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(20)

            if metas.isEmpty {
                Spacer()
                Text("No projects yet. Create one, or open the Demo Stage.")
                    .foregroundStyle(.secondary)
                Spacer()
            } else {
                ScrollView {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 230), spacing: 16)],
                              spacing: 16) {
                        ForEach(metas) { meta in
                            card(meta)
                        }
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 20)
                }
            }
        }
        .background(Color(white: 0.07))
        .onAppear { reload() }
        .sheet(isPresented: $showDesigner) {
            TextureDesignerView()
        }
        .alert("Rename Project", isPresented: renameBinding) {
            TextField("Name", text: $renameText)
            Button("Save") {
                if let target = renameTarget {
                    ProjectStore.renameProject(id: target.id, to: renameText)
                    reload()
                }
                renameTarget = nil
            }
            Button("Cancel", role: .cancel) { renameTarget = nil }
        }
        .confirmationDialog("Delete this project?",
                            isPresented: deleteBinding,
                            titleVisibility: .visible) {
            Button("Delete Project", role: .destructive) {
                if let target = deleteTarget {
                    ProjectStore.deleteProject(id: target.id)
                    reload()
                }
                deleteTarget = nil
            }
            Button("Cancel", role: .cancel) { deleteTarget = nil }
        } message: {
            Text("The project and its imported files will be removed.")
        }
    }

    private var renameBinding: Binding<Bool> {
        Binding(get: { renameTarget != nil }, set: { if !$0 { renameTarget = nil } })
    }
    private var deleteBinding: Binding<Bool> {
        Binding(get: { deleteTarget != nil }, set: { if !$0 { deleteTarget = nil } })
    }

    private func reload() { metas = ProjectStore.listProjects() }

    @ViewBuilder
    private func card(_ meta: ProjectMeta) -> some View {
        Button { onOpen(meta.id) } label: {
            VStack(alignment: .leading, spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(LinearGradient(colors: [Color(white: 0.16), Color(white: 0.1)],
                                             startPoint: .top, endPoint: .bottom))
                    if let url = meta.thumbnailURL,
                       let img = UIImage(contentsOfFile: url.path) {
                        Image(uiImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        Image(systemName: "cube.transparent")
                            .font(.system(size: 44))
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(height: 150)
                .clipShape(RoundedRectangle(cornerRadius: 10))
                Text(meta.name)
                    .font(.headline)
                    .foregroundStyle(.primary)
                    .lineLimit(1)
                Text(meta.modifiedAt, style: .date)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(10)
            .background(Color(white: 0.12))
            .clipShape(RoundedRectangle(cornerRadius: 14))
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button {
                renameText = meta.name
                renameTarget = meta
            } label: { Label("Rename", systemImage: "pencil") }
            Button(role: .destructive) {
                deleteTarget = meta
            } label: { Label("Delete", systemImage: "trash") }
        }
    }
}
