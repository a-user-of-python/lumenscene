// MeshGen.swift — CPU-side triangle meshes: primitive generation,
// transform baking, and the binary mesh-file format used to persist
// imported model geometry inside a project folder.

import Foundation
import simd

struct MeshData {
    var positions: [Float] = []   // 3 per vertex, metres, Y-up
    var normals: [Float] = []     // 3 per vertex
    var uvs: [Float] = []         // 2 per vertex
    var colors: [Float] = []      // 3 per vertex (imported material colours; white otherwise)
    var indices: [UInt32] = []

    var vertexCount: Int { positions.count / 3 }
    var triangleCount: Int { indices.count / 3 }
    var isEmpty: Bool { indices.isEmpty }

    mutating func addVertex(_ p: SIMD3<Float>, _ n: SIMD3<Float>,
                            _ uv: SIMD2<Float>, _ c: SIMD3<Float> = SIMD3<Float>(1, 1, 1)) {
        positions += [p.x, p.y, p.z]
        normals += [n.x, n.y, n.z]
        uvs += [uv.x, uv.y]
        colors += [c.x, c.y, c.z]
    }

    mutating func addQuad(_ a: Int, _ b: Int, _ c: Int, _ d: Int) {
        indices += [UInt32(a), UInt32(b), UInt32(c), UInt32(a), UInt32(c), UInt32(d)]
    }

    /// Apply a model matrix, returning a world-space copy. Normals use
    /// the inverse-transpose so non-uniform scale stays correct.
    func baked(by m: float4x4) -> MeshData {
        var out = MeshData()
        out.positions.reserveCapacity(positions.count)
        out.normals.reserveCapacity(normals.count)
        out.uvs = uvs
        out.colors = colors
        out.indices = indices
        let nmat = float3x3(m.columns.0.xyz, m.columns.1.xyz, m.columns.2.xyz).inverse.transpose
        for i in 0..<vertexCount {
            let p = SIMD3<Float>(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2])
            let wp = (m * SIMD4<Float>(p, 1)).xyz
            out.positions += [wp.x, wp.y, wp.z]
            let n = SIMD3<Float>(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2])
            let wn = simd_normalize(nmat * n)
            out.normals += [wn.x, wn.y, wn.z]
        }
        return out
    }

    /// Axis-aligned bounds (for camera focus / picking helpers).
    var bounds: (min: SIMD3<Float>, max: SIMD3<Float>) {
        guard vertexCount > 0 else { return (SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0)) }
        var lo = SIMD3<Float>(Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude, Float.greatestFiniteMagnitude)
        var hi = -lo
        for i in 0..<vertexCount {
            let p = SIMD3<Float>(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2])
            lo = simd_min(lo, p); hi = simd_max(hi, p)
        }
        return (lo, hi)
    }

    // MARK: Binary persistence ("LSM1": header + raw arrays)
    //
    // Layout: magic(4) version(1 as UInt32) counts(4 × UInt32:
    // vertexCount, indexCount, 0, 0) then positions, normals, uvs,
    // colors (Float32 each) and indices (UInt32). All payload arrays
    // are 4-byte elements so offsets stay 4-byte aligned.

    func encoded() -> Data {
        var d = Data()
        d.append(contentsOf: [0x4C, 0x53, 0x4D, 0x31]) // "LSM1"
        func u32(_ v: UInt32) { var x = v; d.append(Data(bytes: &x, count: 4)) }
        u32(1)
        u32(UInt32(vertexCount)); u32(UInt32(indices.count)); u32(0); u32(0)
        func floats(_ a: [Float]) {
            a.withUnsafeBytes { raw in
                if let base = raw.baseAddress {
                    d.append(base.assumingMemoryBound(to: UInt8.self), count: raw.count)
                }
            }
        }
        floats(positions); floats(normals); floats(uvs); floats(colors)
        indices.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                d.append(base.assumingMemoryBound(to: UInt8.self), count: raw.count)
            }
        }
        return d
    }

    init?(data: Data) {
        let bytes = [UInt8](data)
        guard bytes.count >= 24, bytes[0] == 0x4C, bytes[1] == 0x53,
              bytes[2] == 0x4D, bytes[3] == 0x31 else { return nil }
        func u32at(_ off: Int) -> Int {
            Int(bytes[off]) | Int(bytes[off + 1]) << 8 | Int(bytes[off + 2]) << 16 | Int(bytes[off + 3]) << 24
        }
        let vc = u32at(8), ic = u32at(12)
        let need = 24 + (vc * 11 + ic) * 4
        guard bytes.count >= need else { return nil }
        var off = 24
        func readFloats(_ n: Int) -> [Float] {
            defer { off += n * 4 }
            return (0..<n).map { i in
                let b = off + 4 * i
                let bits = UInt32(bytes[b]) | UInt32(bytes[b + 1]) << 8 | UInt32(bytes[b + 2]) << 16 | UInt32(bytes[b + 3]) << 24
                return Float(bitPattern: bits)
            }
        }
        positions = readFloats(vc * 3)
        normals = readFloats(vc * 3)
        uvs = readFloats(vc * 2)
        colors = readFloats(vc * 3)
        indices = (0..<ic).map { i in
            let b = off + 4 * i
            return UInt32(bytes[b]) | UInt32(bytes[b + 1]) << 8 | UInt32(bytes[b + 2]) << 16 | UInt32(bytes[b + 3]) << 24
        }
    }

    init() {}
}

// MARK: - Primitive generation

enum MeshGen {
    static func primitive(kind: String, params: [String: Double]) -> MeshData {
        switch kind {
        case "sphere": return sphere(radius: Float(params["r"] ?? 0.5))
        case "cylinder": return cylinder(radius: Float(params["r"] ?? 0.5), height: Float(params["h"] ?? 1))
        case "plane": return plane(width: Float(params["w"] ?? 4), depth: Float(params["d"] ?? 4))
        default: return box(w: Float(params["w"] ?? 1), h: Float(params["h"] ?? 1), d: Float(params["d"] ?? 1))
        }
    }

    /// Box centred at the origin (extends ±w/2, ±h/2, ±d/2).
    static func box(w: Float, h: Float, d: Float) -> MeshData {
        var m = MeshData()
        let x = w / 2, y = h / 2, z = d / 2
        // Each face: 4 corners (CCW seen from outside), normal, uvs 0..1.
        let faces: [(SIMD3<Float>, [SIMD3<Float>])] = [
            (SIMD3(0, 0, 1), [SIMD3(-x, -y, z), SIMD3(x, -y, z), SIMD3(x, y, z), SIMD3(-x, y, z)]),
            (SIMD3(0, 0, -1), [SIMD3(x, -y, -z), SIMD3(-x, -y, -z), SIMD3(-x, y, -z), SIMD3(x, y, -z)]),
            (SIMD3(1, 0, 0), [SIMD3(x, -y, z), SIMD3(x, -y, -z), SIMD3(x, y, -z), SIMD3(x, y, z)]),
            (SIMD3(-1, 0, 0), [SIMD3(-x, -y, -z), SIMD3(-x, -y, z), SIMD3(-x, y, z), SIMD3(-x, y, -z)]),
            (SIMD3(0, 1, 0), [SIMD3(-x, y, z), SIMD3(x, y, z), SIMD3(x, y, -z), SIMD3(-x, y, -z)]),
            (SIMD3(0, -1, 0), [SIMD3(-x, -y, -z), SIMD3(x, -y, -z), SIMD3(x, -y, z), SIMD3(-x, -y, z)]),
        ]
        let uvs = [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)]
        for (n, corners) in faces {
            let base = m.vertexCount
            for i in 0..<4 { m.addVertex(corners[i], n, uvs[i]) }
            m.addQuad(base, base + 1, base + 2, base + 3)
        }
        return m
    }

    static func sphere(radius r: Float, latSegs: Int = 24, lonSegs: Int = 40) -> MeshData {
        var m = MeshData()
        for la in 0...latSegs {
            let theta = Float(la) / Float(latSegs) * .pi          // 0..π from +Y
            for lo in 0...lonSegs {
                let phi = Float(lo) / Float(lonSegs) * 2 * .pi
                let p = SIMD3<Float>(r * sin(theta) * sin(phi), r * cos(theta), r * sin(theta) * cos(phi))
                let n = simd_normalize(p)
                m.addVertex(p, n, SIMD2<Float>(Float(lo) / Float(lonSegs), 1 - Float(la) / Float(latSegs)))
            }
        }
        let row = lonSegs + 1
        for la in 0..<latSegs {
            for lo in 0..<lonSegs {
                let a = la * row + lo, b = a + 1, c = a + row, d = c + 1
                m.indices += [UInt32(a), UInt32(c), UInt32(b), UInt32(b), UInt32(c), UInt32(d)]
            }
        }
        return m
    }

    static func cylinder(radius r: Float, height h: Float, segs: Int = 40) -> MeshData {
        var m = MeshData()
        let y = h / 2
        // Side.
        for s in 0...segs {
            let a = Float(s) / Float(segs) * 2 * .pi
            let dir = SIMD3<Float>(sin(a), 0, cos(a))
            let u = Float(s) / Float(segs)
            m.addVertex(SIMD3<Float>(dir.x * r, -y, dir.z * r), dir, SIMD2<Float>(u, 0))
            m.addVertex(SIMD3<Float>(dir.x * r, y, dir.z * r), dir, SIMD2<Float>(u, 1))
        }
        for s in 0..<segs {
            let a = 2 * s, b = a + 1, c = a + 2, d = a + 3
            m.indices += [UInt32(a), UInt32(b), UInt32(c), UInt32(b), UInt32(d), UInt32(c)]
        }
        // Caps (fans with their own centre + ring vertices).
        for (cy, ny) in [(y, Float(1)), (-y, Float(-1))] {
            let centre = m.vertexCount
            m.addVertex(SIMD3<Float>(0, cy, 0), SIMD3<Float>(0, ny, 0), SIMD2<Float>(0.5, 0.5))
            let ringStart = m.vertexCount
            for s in 0..<segs {
                let a = Float(s) / Float(segs) * 2 * .pi
                let p = SIMD3<Float>(sin(a) * r, cy, cos(a) * r)
                m.addVertex(p, SIMD3<Float>(0, ny, 0),
                            SIMD2<Float>(0.5 + 0.5 * sin(a), 0.5 + 0.5 * cos(a)))
            }
            for s in 0..<segs {
                let a = ringStart + s, b = ringStart + (s + 1) % segs
                if ny > 0 { m.indices += [UInt32(centre), UInt32(b), UInt32(a)] }
                else { m.indices += [UInt32(centre), UInt32(a), UInt32(b)] }
            }
        }
        return m
    }

    /// Horizontal plane in XZ centred at the origin, facing +Y.
    static func plane(width w: Float, depth d: Float) -> MeshData {
        var m = MeshData()
        let x = w / 2, z = d / 2, n = SIMD3<Float>(0, 1, 0)
        m.addVertex(SIMD3<Float>(-x, 0, -z), n, SIMD2<Float>(0, 1))
        m.addVertex(SIMD3<Float>(x, 0, -z), n, SIMD2<Float>(1, 1))
        m.addVertex(SIMD3<Float>(x, 0, z), n, SIMD2<Float>(1, 0))
        m.addVertex(SIMD3<Float>(-x, 0, z), n, SIMD2<Float>(0, 0))
        m.indices = [0, 2, 1, 0, 3, 2]
        return m
    }
}

extension SIMD4 {
    var xyz: SIMD3<Scalar> { SIMD3<Scalar>(x, y, z) }
}
