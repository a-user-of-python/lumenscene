//  ModelImport.swift
//  LumenScene
//
//  Model import through Apple's ModelIO (MDLAsset): STL (binary
//  and ASCII), OBJ, and USD / USDZ. (STEP is not supported — no
//  iOS system library reads it; per the 2026-10-30 redirect the
//  Cascade/OCCT engine was removed from this app entirely.)
//
//  Files are expected in CAD convention: Z-up, and — for STL/OBJ —
//  millimetres. Everything is converted exactly once, here, into
//  the app's space (metres, Y-up, right-handed):
//
//      position: (x, y, z) -> (x, z, -y) * scale
//      normal:   (x, y, z) -> (x, z, -y), re-normalized
//
//  with scale = 0.001 for .stl/.obj (mm) and 1.0 for .usd/.usdz
//  (USD's unit is the metre). The mapping is a proper rotation
//  (determinant +1), so winding and handedness are preserved.
//
//  Each (mesh, submesh) pair becomes one ModelPart with its own
//  self-contained buffers. The part colour comes from the
//  submesh material's base colour when the format carries one
//  (OBJ/USD); STL carries no colour, so STL parts are uncoloured.
//  A per-vertex colour attribute, when present, rides along in
//  `vertexColors` and is multiplied with the part colour by the
//  caller.

import Foundation
import ModelIO
import simd

/// One piece of an imported model file, ready to become a scene
/// object. Geometry is in metres, Y-up (already converted).
struct ModelPart {
    var name: String
    var red: Double; var green: Double; var blue: Double  // material colour, 0..1
    var hasColor: Bool                                     // false when the file gave none
    var positions: [Float]   // 3 per vertex, METERS, Y-up
    var normals: [Float]     // 3 per vertex, converted + normalized
    var indices: [UInt32]
    var vertexColors: [Float]?  // 3 per vertex when the file carries them
}

enum ModelImportError: Error { case failed(String) }

enum ModelImporter {

    /// Imports the model at `path` and returns one part per
    /// (mesh, submesh). Throws ModelImportError.failed with a
    /// plain-words message when the file can't be read or holds
    /// no triangle geometry.
    static func importFile(path: String) throws -> [ModelPart] {
        let url = URL(fileURLWithPath: path)
        let ext = url.pathExtension.lowercased()
        let scale: Double = (ext == "usd" || ext == "usdz") ? 1.0 : 0.001

        let asset = MDLAsset(url: url)
        let meshes = (asset.childObjects(of: MDLMesh.self) as? [MDLMesh]) ?? []
        guard !meshes.isEmpty else {
            throw ModelImportError.failed(
                "No meshes could be read from this file. LumenScene imports STL, OBJ and USD/USDZ models.")
        }

        let fileName = (path as NSString).lastPathComponent
        let baseName = (fileName as NSString).deletingPathExtension
        let partBaseName = baseName.isEmpty ? "Model" : baseName

        var parts: [ModelPart] = []
        for mesh in meshes {
            // ---- Vertex attributes (position / normal / colour)
            guard let raw = readVertices(mesh) else { continue }
            let submeshes = (mesh.submeshes as? [MDLSubmesh]) ?? []
            guard !submeshes.isEmpty else { continue }

            for (subIndex, sub) in submeshes.enumerated() {
                guard sub.topology == .triangle else { continue }
                let indices = readIndices(sub)
                guard indices.count >= 3 else { continue }

                // ---- Material colour (OBJ/USD carry one; STL not)
                var red = 1.0, green = 1.0, blue = 1.0
                var hasColor = false
                if let prop = sub.material?.property(with: .baseColor) {
                    let c = prop.float3Value
                    red = Double(c.x); green = Double(c.y); blue = Double(c.z)
                    hasColor = true
                }

                // ---- Rebuild a self-contained part through the
                // submesh's index buffer.
                var positions: [Float] = []
                var normals: [Float] = []
                var colors: [Float]? = raw.colors != nil ? [] : nil
                var outIndices: [UInt32] = []
                positions.reserveCapacity(indices.count * 3)
                normals.reserveCapacity(indices.count * 3)
                outIndices.reserveCapacity(indices.count)
                var next: UInt32 = 0

                func corner(_ vi: Int) -> (SIMD3<Double>, SIMD3<Double>) {
                    let p = SIMD3<Double>(raw.positions[vi])
                    let cp = SIMD3<Double>(p.x * scale, p.z * scale, -p.y * scale)
                    if let n = raw.normals?[vi] {
                        let nd = SIMD3<Double>(n)
                        var rn = SIMD3<Double>(nd.x, nd.z, -nd.y)
                        let len = simd_length(rn)
                        if len > 1e-12 { rn /= len }
                        return (cp, rn)
                    }
                    return (cp, SIMD3<Double>(0, 1, 0))
                }

                if raw.normals != nil {
                    for vi in indices {
                        guard vi >= 0, vi < raw.positions.count else { continue }
                        let (p, n) = corner(vi)
                        positions.append(Float(p.x)); positions.append(Float(p.y)); positions.append(Float(p.z))
                        normals.append(Float(n.x)); normals.append(Float(n.y)); normals.append(Float(n.z))
                        if colors != nil, let c = raw.colors?[vi] {
                            colors!.append(c.x); colors!.append(c.y); colors!.append(c.z)
                        }
                        outIndices.append(next); next += 1
                    }
                } else {
                    // No normals in the file: flat shading — one
                    // face normal per triangle, corners duplicated.
                    var t = 0
                    while t + 2 < indices.count {
                        let ia = indices[t], ib = indices[t + 1], ic = indices[t + 2]
                        guard ia < raw.positions.count, ib < raw.positions.count,
                              ic < raw.positions.count else { t += 3; continue }
                        let (pa, _) = corner(ia)
                        let (pb, _) = corner(ib)
                        let (pc, _) = corner(ic)
                        var fn = simd_cross(pb - pa, pc - pa)
                        let len = simd_length(fn)
                        fn = len > 1e-15 ? fn / len : SIMD3<Double>(0, 1, 0)
                        for (p, vi) in [(pa, ia), (pb, ib), (pc, ic)] {
                            positions.append(Float(p.x)); positions.append(Float(p.y)); positions.append(Float(p.z))
                            normals.append(Float(fn.x)); normals.append(Float(fn.y)); normals.append(Float(fn.z))
                            if colors != nil, let c = raw.colors?[vi] {
                                colors!.append(c.x); colors!.append(c.y); colors!.append(c.z)
                            }
                            outIndices.append(next); next += 1
                        }
                        t += 3
                    }
                }
                guard !outIndices.isEmpty else { continue }

                let name: String
                if meshes.count + submeshes.count > 2 {
                    let meshName = mesh.name.isEmpty ? partBaseName : mesh.name
                    name = submeshes.count > 1
                        ? "\(meshName) — part \(subIndex + 1)"
                        : meshName
                } else {
                    name = partBaseName
                }
                parts.append(ModelPart(
                    name: name, red: red, green: green, blue: blue,
                    hasColor: hasColor, positions: positions,
                    normals: normals, indices: outIndices,
                    vertexColors: colors))
            }
        }

        guard !parts.isEmpty else {
            throw ModelImportError.failed(
                "The file contained no triangle geometry LumenScene can use.")
        }
        return parts
    }

    // MARK: - Raw buffer reads

    private struct RawVertices {
        var positions: [SIMD3<Float>]
        var normals: [SIMD3<Float>]?
        var colors: [SIMD3<Float>]?
    }

    /// Reads a mesh's interleaved vertex buffer by attribute name.
    /// Returns nil when there is no usable position attribute.
    private static func readVertices(_ mesh: MDLMesh) -> RawVertices? {
        let vd = mesh.vertexDescriptor
        let attrs = (vd.attributes as? [MDLVertexAttribute]) ?? []
        func attr(_ name: String) -> MDLVertexAttribute? {
            attrs.first { $0.name == name }
        }
        guard let pa = attr(MDLVertexAttributePosition) else { return nil }
        let na = attr(MDLVertexAttributeNormal)
        let ca = attr(MDLVertexAttributeColor)
        let layouts = (vd.layouts as? [MDLVertexBufferLayout]) ?? []

        func readAll(_ a: MDLVertexAttribute) -> [SIMD3<Float>]? {
            guard a.format == .float3 || a.format == .float4 else { return nil }
            guard a.bufferIndex >= 0, a.bufferIndex < mesh.vertexBuffers.count,
                  a.bufferIndex < layouts.count else { return nil }
            let stride = layouts[a.bufferIndex].stride
            guard stride > 0 else { return nil }
            let buffer = mesh.vertexBuffers[a.bufferIndex]
            let bytes = buffer.map().bytes
            let comps = a.format == .float4 ? 4 : 3
            var out: [SIMD3<Float>] = []
            out.reserveCapacity(mesh.vertexCount)
            for v in 0..<mesh.vertexCount {
                let base = bytes.advanced(by: v * stride + a.offset)
                let fp = base.assumingMemoryBound(to: Float.self)
                out.append(SIMD3<Float>(fp[0], fp[1], fp[2]))
                _ = comps
            }
            return out
        }

        guard let positions = readAll(pa) else { return nil }
        return RawVertices(positions: positions,
                           normals: na.flatMap { readAll($0) },
                           colors: ca.flatMap { readAll($0) })
    }

    /// A submesh's index buffer as [Int] vertex indices.
    private static func readIndices(_ sub: MDLSubmesh) -> [Int] {
        let buffer = sub.indexBuffer
        let bytes = buffer.map().bytes
        var out: [Int] = []
        out.reserveCapacity(sub.indexCount)
        switch sub.indexType {
        case .uInt32:
            let p = bytes.assumingMemoryBound(to: UInt32.self)
            for i in 0..<sub.indexCount { out.append(Int(p[i])) }
        case .uInt16:
            let p = bytes.assumingMemoryBound(to: UInt16.self)
            for i in 0..<sub.indexCount { out.append(Int(p[i])) }
        default:
            break
        }
        return out
    }
}
