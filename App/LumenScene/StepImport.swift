//  StepImport.swift
//  LumenScene
//
//  STEP (.step / .stp) import through the Open CASCADE C façade
//  (occt_facade.h, exposed to Swift via the bridging header).
//
//  The façade speaks in millimetres, Z-up, right-handed. The app
//  speaks in metres, Y-up, right-handed. This file is the only
//  place where that conversion happens, and it happens exactly
//  once, on the way in:
//
//      position: (x, y, z) mm  ->  (x, z, -y) * 0.001   (metres)
//      normal:   (x, y, z)     ->  (x, z, -y), re-normalized
//
//  That mapping is a proper rotation (determinant +1): it swaps
//  the Y and Z axes and negates the new Z, so handedness and
//  triangle winding are preserved.
//
//  The tessellation returned by the façade is one flat vertex /
//  index buffer for the whole shape, with vertices duplicated per
//  B-rep face and a per-face triangle range table. Callers of this
//  file never see that layout: faces are grouped by their STEP
//  colour and each group is rebuilt as a self-contained StepPart
//  with its own fresh buffers and sequential indices.

import Foundation

/// One colour-homogeneous piece of an imported STEP file, ready to
/// become a scene object. Geometry is in metres, Y-up (already
/// converted from the façade's millimetre Z-up space).
struct StepPart {
    var name: String
    var red: Double; var green: Double; var blue: Double  // the group's STEP colour, 0..1
    var hasColor: Bool                                     // false when the file gave no colour
    var positions: [Float]   // 3 per vertex, METERS, Y-up (already converted)
    var normals: [Float]     // 3 per vertex, converted with the same rotation, normalized
    var indices: [UInt32]
}

enum StepImportError: Error { case failed(String) }

enum StepImporter {

    /// Imports the STEP file at `path` and returns one part per
    /// distinct face colour, in first-face order.
    ///
    /// - Throws: `StepImportError.failed` with the façade's error
    ///   text (or a fallback description) on any failure: unreadable
    ///   file, OCCT error, or a shape with no faces / no triangles.
    static func importFile(path: String) throws -> [StepPart] {

        // MARK: - Import the STEP file

        var solidCount: Int32 = 0
        var importErr: Int32 = 0
        var shape: UnsafeMutablePointer<MCFacadeShape>? = nil
        path.withCString { cPath in
            shape = mc_import_step(cPath, &solidCount, &importErr)
        }
        guard let shape = shape else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "Could not import STEP file."))
        }
        // The shape handle is freed on every path from here on.
        defer { mc_free_shape(shape) }
        guard importErr == 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "Could not import STEP file."))
        }

        // MARK: - Census (faces / edges / vertices)

        var faceCount32: Int32 = 0
        var edgeCount32: Int32 = 0
        var vertexCount32: Int32 = 0
        var countsErr: Int32 = 0
        let countsRC = mc_shape_counts(
            shape, &faceCount32, &edgeCount32, &vertexCount32, &countsErr)
        guard countsRC == 0, countsErr == 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "Could not read the STEP shape."))
        }
        guard faceCount32 > 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "The STEP file contains no faces."))
        }
        let faceCount = Int(faceCount32)

        // MARK: - Per-face colours (census / face order)

        var faceRGB = [Double](repeating: 0, count: faceCount * 3)
        var faceHasColor = [Int32](repeating: 0, count: faceCount)
        var colorsErr: Int32 = 0
        let colorsRC = mc_shape_face_colors(
            shape, &faceRGB, &faceHasColor, &colorsErr)
        guard colorsRC == 0, colorsErr == 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "Could not read the STEP face colours."))
        }

        // MARK: - Tessellate (0.1 mm deflection)

        var mesh = MCFacadeMesh(
            positions: nil, normals: nil, indices: nil,
            vertexCount: 0, triangleCount: 0,
            faceTriStart: nil, faceTriCount: nil, faceCount: 0)
        // The mesh buffers are freed on every path from here on,
        // whether tessellation succeeded, failed, or a later step
        // throws while rebuilding the parts.
        defer { mc_free_mesh(&mesh) }

        var tessErr: Int32 = 0
        let tessRC = mc_tessellate(shape, 0.1, &mesh, &tessErr)
        guard tessRC == 0, tessErr == 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "Could not tessellate the STEP shape."))
        }
        guard mesh.triangleCount > 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "The STEP file produced no triangles."))
        }
        guard mesh.vertexCount > 0, mesh.faceCount > 0 else {
            throw StepImportError.failed(
                lastErrorMessage(fallback: "The STEP file produced no geometry."))
        }
        // Face colours are reported in census order and the mesh's
        // per-face triangle ranges are in that same order; if the
        // two face counts ever disagreed the colours could not be
        // trusted to line up with the triangles, so refuse.
        guard Int(mesh.faceCount) == faceCount else {
            throw StepImportError.failed(
                "The tessellation's face count does not match the shape's face count.")
        }
        guard let positionsPtr = mesh.positions,
              let normalsPtr = mesh.normals,
              let indicesPtr = mesh.indices,
              let faceTriStartPtr = mesh.faceTriStart,
              let faceTriCountPtr = mesh.faceTriCount else {
            throw StepImportError.failed(
                "The tessellation returned incomplete buffers.")
        }

        let vertexCount = Int(mesh.vertexCount)
        let triangleCount = Int(mesh.triangleCount)
        let positions = UnsafeBufferPointer(
            start: positionsPtr, count: vertexCount * 3)
        let normals = UnsafeBufferPointer(
            start: normalsPtr, count: vertexCount * 3)
        let indices = UnsafeBufferPointer(
            start: indicesPtr, count: triangleCount * 3)
        let faceTriStart = UnsafeBufferPointer(
            start: faceTriStartPtr, count: faceCount)
        let faceTriCount = UnsafeBufferPointer(
            start: faceTriCountPtr, count: faceCount)

        // MARK: - Group faces by colour
        //
        // Two faces belong to the same part when they agree on the
        // hasColor flag and, if coloured, on RGB quantized to
        // 1/255 steps (so near-identical STEP colours that differ
        // only below display precision still merge). Uncoloured
        // faces carry no meaningful RGB, so they all share one key
        // and form a single part. Groups are kept in first-face
        // order, which is also the part naming order below.

        var groupOrder: [ColourKey] = []
        var groupFaces: [ColourKey: [Int]] = [:]
        for face in 0..<faceCount {
            let coloured = faceHasColor[face] != 0
            let key: ColourKey
            if coloured {
                key = ColourKey(
                    hasColor: true,
                    red: quantized(faceRGB[face * 3]),
                    green: quantized(faceRGB[face * 3 + 1]),
                    blue: quantized(faceRGB[face * 3 + 2]))
            } else {
                key = ColourKey(hasColor: false, red: 0, green: 0, blue: 0)
            }
            if groupFaces[key] == nil {
                groupOrder.append(key)
            }
            groupFaces[key, default: []].append(face)
        }

        // MARK: - Rebuild one self-contained part per group
        //
        // For every face in the group, walk its triangle range in
        // the global tessellation and copy each corner's position
        // and normal out through the index buffer, appending a
        // fresh sequential index. Parts therefore share nothing
        // with the façade buffers (freed by the defer above) or
        // with each other.

        let fileName = (path as NSString).lastPathComponent
        let baseName = (fileName as NSString).deletingPathExtension
        let partBaseName = baseName.isEmpty ? "Part" : baseName

        var parts: [StepPart] = []
        parts.reserveCapacity(groupOrder.count)

        for (groupIndex, key) in groupOrder.enumerated() {
            let faces = groupFaces[key] ?? []

            // Reserve roughly the right capacity up front.
            var groupTriangles = 0
            for face in faces {
                groupTriangles += Int(faceTriCount[face])
            }

            var partPositions: [Float] = []
            var partNormals: [Float] = []
            var partIndices: [UInt32] = []
            partPositions.reserveCapacity(groupTriangles * 9)
            partNormals.reserveCapacity(groupTriangles * 9)
            partIndices.reserveCapacity(groupTriangles * 3)
            var nextIndex: UInt32 = 0

            for face in faces {
                let triStart = Int(faceTriStart[face])
                let triCount = Int(faceTriCount[face])
                guard triStart >= 0, triCount >= 0,
                      triStart + triCount <= triangleCount else {
                    throw StepImportError.failed(
                        "The tessellation's per-face triangle range is out of bounds.")
                }
                for triangle in triStart..<(triStart + triCount) {
                    for corner in 0..<3 {
                        let source = Int(indices[triangle * 3 + corner])
                        guard source >= 0, source < vertexCount else {
                            throw StepImportError.failed(
                                "The tessellation contains an out-of-bounds vertex index.")
                        }
                        let offset = source * 3

                        // Position: (x, y, z) mm -> (x, z, -y) m.
                        let px = Double(positions[offset])
                        let py = Double(positions[offset + 1])
                        let pz = Double(positions[offset + 2])
                        partPositions.append(Float(px * 0.001))
                        partPositions.append(Float(pz * 0.001))
                        partPositions.append(Float(-py * 0.001))

                        // Normal: same rotation, no scale, then
                        // re-normalize (the rotation alone already
                        // preserves length; this guards against
                        // non-unit normals coming from OCCT).
                        let nx = Double(normals[offset])
                        let ny = Double(normals[offset + 1])
                        let nz = Double(normals[offset + 2])
                        let rx = nx, ry = nz, rz = -ny
                        let length = (rx * rx + ry * ry + rz * rz).squareRoot()
                        if length > 1e-12 {
                            partNormals.append(Float(rx / length))
                            partNormals.append(Float(ry / length))
                            partNormals.append(Float(rz / length))
                        } else {
                            partNormals.append(Float(rx))
                            partNormals.append(Float(ry))
                            partNormals.append(Float(rz))
                        }

                        partIndices.append(nextIndex)
                        nextIndex += 1
                    }
                }
            }

            // Part colour: the quantized group colour (so the value
            // reported here is exactly the value faces were grouped
            // by). Uncoloured parts report white with hasColor off.
            let red: Double
            let green: Double
            let blue: Double
            if key.hasColor {
                red = Double(key.red) / 255.0
                green = Double(key.green) / 255.0
                blue = Double(key.blue) / 255.0
            } else {
                red = 1; green = 1; blue = 1
            }

            let name: String
            if groupOrder.count > 1 {
                name = "\(partBaseName) — part \(groupIndex + 1)"
            } else {
                name = partBaseName
            }

            parts.append(StepPart(
                name: name,
                red: red, green: green, blue: blue,
                hasColor: key.hasColor,
                positions: partPositions,
                normals: partNormals,
                indices: partIndices))
        }

        return parts
    }

    // MARK: - Helpers

    /// Grouping key for face colours: the hasColor flag plus, for
    /// coloured faces, RGB quantized to 1/255 steps (0...255).
    private struct ColourKey: Hashable {
        var hasColor: Bool
        var red: Int
        var green: Int
        var blue: Int
    }

    /// Quantizes a 0...1 colour component to the nearest 1/255
    /// step, clamped to 0...255.
    private static func quantized(_ value: Double) -> Int {
        let step = Int((value * 255.0).rounded())
        return min(255, max(0, step))
    }

    /// The façade's last error text, or `fallback` when the façade
    /// reports nothing (null or empty string).
    private static func lastErrorMessage(fallback: String) -> String {
        if let cString = mc_last_error() {
            let text = String(cString: cString)
            if !text.isEmpty {
                return text
            }
        }
        return fallback
    }
}
