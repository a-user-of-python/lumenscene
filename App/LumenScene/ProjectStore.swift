// ProjectStore.swift — on-disk layout for projects.
//
//   Application Support/LumenScene/Projects/<id>/
//     project.json   the Project value (scene + camera)
//     thumb.png      viewport render from the last Save
//     assets/        original model files copied in at import time
//     meshes/        tessellated geometry (.meshbin, LSM1 format)
//
// Imported meshes and assets are written at IMPORT time (not save
// time) so undo snapshots, which reference them by file name, are
// always consistent. The Documents folder (visible in Files via
// UIFileSharingEnabled) is the drop zone for model files (STL/OBJ/USD);
// it is only ever read by the import sheet.

import Foundation

enum ProjectStore {
    static var rootURL: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
        return base.appendingPathComponent("LumenScene/Projects", isDirectory: true)
    }

    static var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func projectDir(_ id: UUID) -> URL {
        rootURL.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    static func ensureDirs(_ id: UUID) {
        let fm = FileManager.default
        for sub in ["", "assets", "meshes"] {
            try? fm.createDirectory(at: projectDir(id).appendingPathComponent(sub),
                                    withIntermediateDirectories: true)
        }
    }

    // MARK: Listing / loading

    static func listProjects() -> [ProjectMeta] {
        let fm = FileManager.default
        guard let dirs = try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil) else {
            return []
        }
        var metas: [ProjectMeta] = []
        for dir in dirs {
            let json = dir.appendingPathComponent("project.json")
            guard let data = try? Data(contentsOf: json),
                  let p = try? JSONDecoder().decode(Project.self, from: data) else { continue }
            let thumb = dir.appendingPathComponent("thumb.png")
            metas.append(ProjectMeta(id: p.id, name: p.name, modifiedAt: p.modifiedAt,
                                     thumbnailURL: fm.fileExists(atPath: thumb.path) ? thumb : nil))
        }
        return metas.sorted { $0.modifiedAt > $1.modifiedAt }
    }

    static func loadProject(id: UUID) -> Project? {
        let json = projectDir(id).appendingPathComponent("project.json")
        guard let data = try? Data(contentsOf: json) else { return nil }
        return try? JSONDecoder().decode(Project.self, from: data)
    }

    static func saveProject(_ project: Project, thumbnailPNG: Data?) {
        ensureDirs(project.id)
        var p = project
        p.modifiedAt = Date()
        let json = projectDir(project.id).appendingPathComponent("project.json")
        if let data = try? JSONEncoder().encode(p) {
            try? data.write(to: json, options: .atomic)
        }
        if let thumbnailPNG {
            let thumb = projectDir(project.id).appendingPathComponent("thumb.png")
            try? thumbnailPNG.write(to: thumb, options: .atomic)
        }
    }

    static func deleteProject(id: UUID) {
        try? FileManager.default.removeItem(at: projectDir(id))
    }

    static func renameProject(id: UUID, to name: String) {
        guard var p = loadProject(id: id) else { return }
        p.name = name
        saveProject(p, thumbnailPNG: nil)
    }

    // MARK: Mesh / asset files inside a project

    static func writeMesh(projectID: UUID, objectID: UUID, mesh: MeshData) -> String {
        ensureDirs(projectID)
        let name = objectID.uuidString + ".meshbin"
        let url = projectDir(projectID).appendingPathComponent("meshes/\(name)")
        try? mesh.encoded().write(to: url, options: .atomic)
        return name
    }

    static func readMesh(projectID: UUID, fileName: String) -> MeshData? {
        let url = projectDir(projectID).appendingPathComponent("meshes/\(fileName)")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return MeshData(data: data)
    }

    static func copyAsset(projectID: UUID, from source: URL) -> String? {
        ensureDirs(projectID)
        let fm = FileManager.default
        var dest = projectDir(projectID).appendingPathComponent("assets/\(source.lastPathComponent)")
        if fm.fileExists(atPath: dest.path) {
            let base = source.deletingPathExtension().lastPathComponent
            let ext = source.pathExtension
            dest = projectDir(projectID).appendingPathComponent(
                "assets/\(base)-\(UUID().uuidString.prefix(6)).\(ext)")
        }
        do {
            try fm.copyItem(at: source, to: dest)
            return dest.lastPathComponent
        } catch {
            return nil
        }
    }

    static func assetURL(projectID: UUID, fileName: String) -> URL {
        projectDir(projectID).appendingPathComponent("assets/\(fileName)")
    }

    // MARK: The app-folder drop zone (Documents)

    /// Model files dropped into the app folder via Files (STL / OBJ / USD / USDZ).
    static func appFolderModelFiles() -> [URL] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
            at: documentsURL, includingPropertiesForKeys: nil) else { return [] }
        return items.filter { ["stl", "obj", "usd", "usdz"].contains($0.pathExtension.lowercased()) }
            .sorted { $0.lastPathComponent.lowercased() < $1.lastPathComponent.lowercased() }
    }

    // MARK: Export — project package (.zip, stored entries)

    static func exportPackage(projectID: UUID) -> URL? {
        let dir = projectDir(projectID)
        let fm = FileManager.default
        guard let files = fm.enumerator(at: dir, includingPropertiesForKeys: nil) else { return nil }
        var entries: [(name: String, data: Data)] = []
        let projectName = (loadProject(id: projectID)?.name ?? "LumenScene")
            .replacingOccurrences(of: "/", with: "-")
        for case let url as URL in files {
            if url.hasDirectoryPath { continue }
            let rel = url.path.replacingOccurrences(of: dir.path + "/", with: "")
            if let data = try? Data(contentsOf: url) {
                entries.append(("\(projectName)/\(rel)", data))
            }
        }
        guard let zip = ZipWriter.make(entries: entries) else { return nil }
        let out = fm.temporaryDirectory.appendingPathComponent("\(projectName).lumenscene.zip")
        do {
            try zip.write(to: out, options: .atomic)
            return out
        } catch { return nil }
    }
}

// MARK: - Minimal ZIP writer (stored, no compression)

enum ZipWriter {
    static func make(entries: [(name: String, data: Data)]) -> Data? {
        var out = Data()
        var central = Data()
        for (name, data) in entries {
            guard let nameBytes = name.data(using: .utf8) else { return nil }
            let crc = crc32(data)
            let offset = UInt32(out.count)
            func w16(_ v: UInt16) { var x = v.littleEndian; out.append(Data(bytes: &x, count: 2)) }
            func w32(_ v: UInt32) { var x = v.littleEndian; out.append(Data(bytes: &x, count: 4)) }
            w32(0x04034b50); w16(20); w16(0x0800); w16(0)
            w16(0); w16(0) // time, date
            w32(crc); w32(UInt32(data.count)); w32(UInt32(data.count))
            w16(UInt16(nameBytes.count)); w16(0)
            out.append(nameBytes)
            out.append(data)
            func c16(_ v: UInt16) { var x = v.littleEndian; central.append(Data(bytes: &x, count: 2)) }
            func c32(_ v: UInt32) { var x = v.littleEndian; central.append(Data(bytes: &x, count: 4)) }
            c32(0x02014b50); c16(20); c16(20); c16(0x0800); c16(0)
            c16(0); c16(0)
            c32(crc); c32(UInt32(data.count)); c32(UInt32(data.count))
            c16(UInt16(nameBytes.count)); c16(0); c16(0); c16(0); c16(0)
            c32(0); c32(offset)
            central.append(nameBytes)
        }
        let cdOffset = UInt32(out.count)
        out.append(central)
        func e32(_ v: UInt32) { var x = v.littleEndian; out.append(Data(bytes: &x, count: 4)) }
        func e16(_ v: UInt16) { var x = v.littleEndian; out.append(Data(bytes: &x, count: 2)) }
        e32(0x06054b50); e16(0); e16(0)
        e16(UInt16(entries.count)); e16(UInt16(entries.count))
        e32(UInt32(central.count)); e32(cdOffset); e16(0)
        return out
    }

    static func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB8_8320 : crc >> 1
            }
        }
        return crc ^ 0xFFFF_FFFF
    }
}
