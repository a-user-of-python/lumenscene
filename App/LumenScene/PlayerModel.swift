// PlayerModel.swift — the procedural low-poly player figure.
//
// A ~1.75 m humanoid built from coloured boxes (MeshGen primitives),
// posed by a procedural walk cycle. Used as the visible Play-mode /
// spawn figure: the caller asks for `mesh(phase:blend:)` each frame
// (or once, at phase 0 / blend 0, for a static stance) and renders the
// result like any other MeshData — all colours are per-vertex, so the
// figure needs no special material.
//
// Local space: feet at the origin (y = 0 at the soles), facing −Z —
// the app's camera convention (CameraState.playBasis: yaw 0 looks
// along −Z, the character's right hand is on +X, so its left is −X).
// The eye line (y ≈ 1.7) sits inside the head box.
//
// Rotation-sign convention (derived, and matching Transform3.matrix /
// playBasis, which both use right-handed simd_quaternion axes):
// a rotation about +X by angle a maps
//     +Y → (0, cos a,  sin a)     −Y → (0, −cos a, −sin a)
// Check against playBasis: pitching the −Z forward vector by a gives
// (0, sin a, −cos a) — positive pitch looks up, exactly as playBasis
// has it. Consequence for limbs: a limb hangs along −Y from its
// pivot, so POSITIVE rotX carries its free end toward −Z — a FORWARD
// swing. Legs therefore use +sin(phase …) (positive half-cycle =
// stepping forward) and arms −sin(phase …) to counter-swing against
// the same-side leg. The torso is the opposite case — it extends
// along +Y from the pelvis-top pivot — so tipping its top forward
// (−Z) takes a NEGATIVE rotX: the lean below is −0.07 × blend.

import Foundation
import simd

enum PlayerModel {

    // MARK: Palette (per-vertex colours, linear-ish RGB as authored)

    private static let jacket = SIMD3<Float>(0.27, 0.38, 0.58)  // slate blue
    private static let pants  = SIMD3<Float>(0.16, 0.17, 0.22)  // dark trousers
    private static let skin   = SIMD3<Float>(0.87, 0.66, 0.52)
    private static let hairC  = SIMD3<Float>(0.23, 0.16, 0.11)  // dark brown
    private static let bootC  = SIMD3<Float>(0.08, 0.08, 0.09)  // near-black

    /// The figure posed at walk-cycle `phase` (radians; 0 = neutral
    /// stance, a full stride is 2π) with `blend` 0 = idle stance,
    /// 1 = full walk. The part list is fixed, so the vertex and
    /// triangle counts are CONSTANT for every (phase, blend):
    /// 12 boxes = 288 vertices, 144 triangles.
    static func mesh(phase: Float, blend: Float) -> MeshData {
        let b = min(1, max(0, blend))

        // Whole-body bob: two footfalls per cycle, so |sin| — it lifts
        // EVERY part (legs included) by up to 4.5 cm at full walk.
        let bob = abs(sin(phase)) * 0.045 * b
        let bobT = translation(SIMD3<Float>(0, bob, 0))

        // Torso group: forward lean + lateral sway, both pivoting
        // about the pelvis top (y = 0.95) so the pelvis itself stays
        // planted and only what sits on it tips. Applied to the torso,
        // head, hair and the arm pivots (arms inherit the lean/sway,
        // then add their own swing at the shoulder). The sway (rotZ)
        // is applied before the lean (rotX) — rightmost first, the
        // same composition order as Transform3.matrix.
        let pelvisTop = SIMD3<Float>(0, 0.95, 0)
        let lean = -0.07 * b                    // negative: +Y top tips toward −Z
        let sway = sin(phase) * 0.03 * b
        let torsoGroup = bobT
            * translation(pelvisTop)
            * rotationX(lean) * rotationZ(sway)
            * translation(-pelvisTop)

        var parts: [MeshData] = []
        parts.reserveCapacity(12)

        // Pelvis — bob only; it is the base the torso group pivots on.
        // 0.38 × 0.18 × 0.23, centre y 0.86 (spans 0.77–0.95).
        add(&parts, box: (0.38, 0.18, 0.23), colour: pants,
            matrix: bobT * translation(SIMD3<Float>(0, 0.86, 0)))

        // Torso — 0.42 × 0.60 × 0.24, centre y 1.21 (spans 0.91–1.51).
        add(&parts, box: (0.42, 0.60, 0.24), colour: jacket,
            matrix: torsoGroup * translation(SIMD3<Float>(0, 1.21, 0)))

        // Head — 0.24 × 0.25 × 0.25, centre y 1.615 (spans 1.49–1.74;
        // height trimmed from 0.27 so the hair top lands at ≈1.75).
        // The bottom overlaps the torso top slightly — no neck gap
        // when the torso group leans.
        add(&parts, box: (0.24, 0.25, 0.25), colour: skin,
            matrix: torsoGroup * translation(SIMD3<Float>(0, 1.615, 0)))

        // Hair cap — 0.26 × 0.07 × 0.27, slight overhang on every side,
        // centre y = head top (1.74) − 0.02 = 1.72, so the figure's
        // total height is 1.755 ≈ 1.75 m.
        add(&parts, box: (0.26, 0.07, 0.27), colour: hairC,
            matrix: torsoGroup * translation(SIMD3<Float>(0, 1.72, 0)))

        // Arms — pivot at the shoulder (±0.265, 1.455; arm inner face
        // flush with the torso side). Built hanging along −Y in
        // pivot-local space: sleeve+upper arm as one jacket box, then
        // a skin hand overlapping its bottom. Swing is rotX about the
        // shoulder, counter-phased to the same-side leg; the constant
        // rotZ tilts the hand slightly outward (applied first, then
        // the swing — rightmost first, as in Transform3).
        for isLeft in [true, false] {
            let side: Float = isLeft ? -1 : 1        // left = −X, right = +X
            let swing = -sin(phase + (isLeft ? 0 : .pi)) * 0.55 * b + 0.05
            let outward: Float = isLeft ? -0.07 : 0.07
            // Outward sign: rotZ by a maps the hanging −Y to
            // x = +L·sin a, so a takes the sign of the side it must
            // fall away toward (+0.07 on +X, −0.07 on −X).
            let shoulder = torsoGroup
                * translation(SIMD3<Float>(side * 0.265, 1.455, 0))
                * rotationX(swing) * rotationZ(outward)
            // Sleeve/arm: 0.11 × 0.58 × 0.11, centre y −0.29
            // (spans 0 to −0.58 below the shoulder).
            add(&parts, box: (0.11, 0.58, 0.11), colour: jacket,
                matrix: shoulder * translation(SIMD3<Float>(0, -0.29, 0)))
            // Hand: 0.10 × 0.13 × 0.10, centre y −0.63
            // (spans −0.565 to −0.695, overlapping the sleeve bottom).
            add(&parts, box: (0.10, 0.13, 0.10), colour: skin,
                matrix: shoulder * translation(SIMD3<Float>(0, -0.63, 0)))
        }

        // Legs — pivot at the hip (±0.105, 0.92), single-piece (no
        // knee at this scale). Legs are NOT in the torso group: they
        // get the bob and their own swing only. Positive rotX swings
        // the foot forward (−Z), opposite phase to the same-side arm.
        for isLeft in [true, false] {
            let side: Float = isLeft ? -1 : 1
            let swing = sin(phase + (isLeft ? 0 : .pi)) * 0.62 * b
            let hip = bobT
                * translation(SIMD3<Float>(side * 0.105, 0.92, 0))
                * rotationX(swing)
            // Leg: 0.15 × 0.84 × 0.15, centre y −0.44
            // (spans −0.02 to −0.86 below the hip).
            add(&parts, box: (0.15, 0.84, 0.15), colour: pants,
                matrix: hip * translation(SIMD3<Float>(0, -0.44, 0)))
            // Boot: 0.15 × 0.09 × 0.27, centre (y −0.875, z −0.05) —
            // sole exactly at local y = 0 in idle stance, toe
            // overhanging toward −Z.
            add(&parts, box: (0.15, 0.09, 0.27), colour: bootC,
                matrix: hip * translation(SIMD3<Float>(0, -0.875, -0.05)))
        }

        return merged(parts)
    }

    // MARK: - Part assembly

    /// Builds one MeshGen box, overwrites every vertex colour with
    /// `colour` (MeshGen writes white), bakes `matrix` into it and
    /// appends it to `parts`.
    private static func add(_ parts: inout [MeshData],
                            box size: (w: Float, h: Float, d: Float),
                            colour: SIMD3<Float>, matrix: float4x4) {
        var m = MeshGen.box(w: size.w, h: size.h, d: size.d)
        for v in 0..<m.vertexCount {
            m.colors[3 * v] = colour.x
            m.colors[3 * v + 1] = colour.y
            m.colors[3 * v + 2] = colour.z
        }
        parts.append(m.baked(by: matrix))
    }

    /// Concatenates baked parts into one mesh, offsetting each part's
    /// indices by the running vertex count.
    private static func merged(_ parts: [MeshData]) -> MeshData {
        var out = MeshData()
        for p in parts {
            let base = UInt32(out.vertexCount)
            out.positions += p.positions
            out.normals += p.normals
            out.uvs += p.uvs
            out.colors += p.colors
            out.indices += p.indices.map { $0 + base }
        }
        return out
    }

    // MARK: - Matrix helpers (mirroring Transform3.matrix conventions)

    private static func translation(_ v: SIMD3<Float>) -> float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(v, 1)
        return m
    }

    private static func rotationX(_ angle: Float) -> float4x4 {
        float4x4(simd_quaternion(angle, SIMD3<Float>(1, 0, 0)))
    }

    private static func rotationZ(_ angle: Float) -> float4x4 {
        float4x4(simd_quaternion(angle, SIMD3<Float>(0, 0, 1)))
    }
}
