// Textures.swift
// LumenScene
//
// Fully procedural, deterministic PBR texture generation.
//
// Every preset produces a square 512x512 texture set:
//   - albedo: RGBA8 base colour
//   - normal: RGBA8 tangent-space normal map (rgb = n * 0.5 + 0.5, a = 255)
//   - data:   RGBA8 packing (R = roughness, G = height, B = 0, A = 255)
//
// Design notes:
//   * Deterministic: a small self-contained SplitMix64 PRNG with fixed
//     per-preset seeds is the only source of randomness. No system RNG,
//     no dates, no hashing of addresses — the same preset always yields
//     byte-identical output.
//   * Tileable: all noise is periodic value noise whose lattice
//     coordinates wrap modulo the octave period, filtering is wrap-aware,
//     per-pixel speckle is indexed by tile position, the Sobel pass wraps
//     around the edges, and the brick layout uses integer brick counts
//     (4 per row, 8 rows) so the pattern closes on itself exactly.
//   * Performance: noise is evaluated from small precomputed grids and
//     the pixel loops are plain arithmetic, so a set generates in well
//     under ~2 s on an A-series CPU.

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// MARK: - Public contract

struct PBRTextureSet {
    let size: Int          // square, e.g. 512
    let albedo: Data       // RGBA8, size*size*4 bytes
    let normal: Data       // RGBA8 tangent-space normal map (rgb = n*0.5+0.5, a=255)
    let data: Data         // RGBA8: R = roughness*255, G = height*255 (0..1), B = 0, A = 255
}

enum TextureGen {
    static let presetNames = ["concrete", "cement", "brick"]

    /// Road designs. These are fit-mapped (MaterialProps.textureFit):
    /// the whole 512x512 design spans one face, so unlike the material
    /// presets they are not required to tile. Road direction runs
    /// along v — top to bottom of the texture.
    static let roadNames = ["road-highway", "road-normal", "road-intersection", "road-crosswalk", "road-roundabout"]

    /// Every slice the renderer should build a texture array from:
    /// material presets first, then roads.
    static func allSliceNames() -> [String] { presetNames + roadNames }

    static func isRoad(_ name: String) -> Bool { roadNames.contains(name) }

    static func displayName(_ name: String) -> String {
        switch name {
        case "concrete": return "Concrete"
        case "cement": return "Cement"
        case "brick": return "Brick"
        case "road-highway": return "Highway"
        case "road-normal": return "Road"
        case "road-intersection": return "Intersection"
        case "road-crosswalk": return "Crosswalk"
        case "road-roundabout": return "Roundabout"
        default: return name
        }
    }

    /// Builds the texture set for `preset` (one of `presetNames` or
    /// `roadNames`). Unknown names fall back to "concrete"; callers are
    /// expected to pass a value from `allSliceNames()`.
    static func makeSet(preset: String) -> PBRTextureSet {
        switch preset {
        case "cement":
            return makeCement()
        case "brick":
            return makeBrick()
        case "road-highway":
            return makeRoadHighway()
        case "road-normal":
            return makeRoadNormal()
        case "road-intersection":
            return makeRoadIntersection()
        case "road-crosswalk":
            return makeRoadCrosswalk()
        case "road-roundabout":
            return makeRoadRoundabout()
        default: // "concrete", plus any unexpected name
            return makeConcrete()
        }
    }

    /// Builds a PBR set from caller-supplied per-pixel fields (the
    /// Texture Designer's painted channels), deriving normals from
    /// `height` exactly like the procedural presets do.
    /// - Parameters:
    ///   - albedo: interleaved RGB floats in 0..1, `size*size*3` values.
    ///   - height: height floats in 0..1, `size*size` values.
    ///   - roughness: roughness floats in 0..1, `size*size` values.
    static func makeCustomSet(size: Int,
                              albedo: [Double],
                              height: [Double],
                              roughness: [Double],
                              normalStrength: Double) -> PBRTextureSet {
        buildSet(size: size, albedo: albedo, height: height,
                 roughness: roughness, normalStrength: normalStrength)
    }

    // MARK: - PRNG (SplitMix64)

    /// SplitMix64: tiny, fast, well-distributed 64-bit PRNG.
    /// A fixed seed gives a fixed stream, which is what makes the
    /// generated textures byte-for-byte reproducible.
    private struct SplitMix64 {
        var state: UInt64

        init(seed: UInt64) {
            state = seed
        }

        mutating func nextUInt64() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }

        /// Uniform Double in [0, 1) built from the top 53 bits.
        mutating func nextDouble() -> Double {
            Double(nextUInt64() >> 11) * (1.0 / 9_007_199_254_740_992.0) // 2^53
        }
    }

    // MARK: - Periodic value noise

    /// A square grid of random values with period `period`, sampled with
    /// smoothstep-interpolated bilinear filtering. Lattice coordinates
    /// wrap modulo `period`, so sampling a field over u, v in [0, 1)
    /// tiles perfectly in both axes.
    private struct ValueGrid {
        let period: Int
        let values: [Double] // period * period, row-major

        init(period: Int, rng: inout SplitMix64) {
            self.period = period
            var grid = [Double]()
            grid.reserveCapacity(period * period)
            for _ in 0 ..< (period * period) {
                grid.append(rng.nextDouble())
            }
            values = grid
        }

        /// Samples the grid at tile coordinates (u, v), both in [0, 1).
        func sample(u: Double, v: Double) -> Double {
            // Position in lattice space, in [0, period).
            let x = u * Double(period)
            let y = v * Double(period)
            // x, y are non-negative, so truncation is floor here.
            let x0 = Int(x)
            let y0 = Int(y)
            let xf = x - Double(x0)
            let yf = y - Double(y0)

            let xi0 = x0 % period
            let yi0 = y0 % period
            let xi1 = (xi0 + 1) % period
            let yi1 = (yi0 + 1) % period

            // Smoothstep weights keep derivatives continuous across cells.
            let sx = xf * xf * (3.0 - 2.0 * xf)
            let sy = yf * yf * (3.0 - 2.0 * yf)

            let v00 = values[yi0 * period + xi0]
            let v10 = values[yi0 * period + xi1]
            let v01 = values[yi1 * period + xi0]
            let v11 = values[yi1 * period + xi1]

            let top = v00 + (v10 - v00) * sx
            let bottom = v01 + (v11 - v01) * sx
            return top + (bottom - top) * sy
        }
    }

    /// Fractal Brownian motion: a stack of value-noise octaves, each with
    /// double the period and half the amplitude of the previous one.
    /// The summed result is normalised back into [0, 1].
    private struct FBMNoise {
        let grids: [ValueGrid]
        let amplitudes: [Double]
        let amplitudeSum: Double

        init(basePeriod: Int, octaves: Int, rng: inout SplitMix64) {
            var builtGrids: [ValueGrid] = []
            var builtAmplitudes: [Double] = []
            var sum = 0.0
            var period = basePeriod
            var amplitude = 1.0
            for _ in 0 ..< octaves {
                builtGrids.append(ValueGrid(period: period, rng: &rng))
                builtAmplitudes.append(amplitude)
                sum += amplitude
                period *= 2
                amplitude *= 0.5
            }
            grids = builtGrids
            amplitudes = builtAmplitudes
            amplitudeSum = sum
        }

        /// Samples the fbm field at tile coordinates (u, v) in [0, 1).
        func sample(u: Double, v: Double) -> Double {
            var accumulated = 0.0
            for index in grids.indices {
                accumulated += grids[index].sample(u: u, v: v) * amplitudes[index]
            }
            return accumulated / amplitudeSum
        }
    }

    // MARK: - Small numeric helpers

    private static func clamp01(_ value: Double) -> Double {
        min(1.0, max(0.0, value))
    }

    private static func clamp(_ value: Double, _ low: Double, _ high: Double) -> Double {
        min(high, max(low, value))
    }

    /// Hermite smoothstep between `edge0` and `edge1`.
    private static func smoothstep(_ edge0: Double, _ edge1: Double, _ value: Double) -> Double {
        let t = clamp01((value - edge0) / (edge1 - edge0))
        return t * t * (3.0 - 2.0 * t)
    }

    /// Converts a 0..1 float to a byte, clamping first.
    private static func toByte(_ value: Double) -> UInt8 {
        UInt8((clamp01(value) * 255.0).rounded())
    }

    /// Per-pixel speckle for a whole tile: one random value per pixel,
    /// generated in row-major order. Indexed purely by tile position, so
    /// it tiles trivially (the tile *is* the period).
    private static func makeSpeckle(count: Int, rng: inout SplitMix64) -> [Double] {
        var speckle = [Double]()
        speckle.reserveCapacity(count)
        for _ in 0 ..< count {
            speckle.append(rng.nextDouble())
        }
        return speckle
    }

    // MARK: - Set assembly (packing + Sobel normal map)

    /// Packs per-pixel fields into a `PBRTextureSet`, deriving the normal
    /// map from the height field with a 3x3 Sobel operator.
    ///
    /// - Parameters:
    ///   - albedo: interleaved RGB floats in 0..1, `size*size*3` values.
    ///   - height: height floats in 0..1, `size*size` values.
    ///   - roughness: roughness floats in 0..1, `size*size` values.
    ///   - normalStrength: scales the height gradient. Brick uses a large
    ///     value (deep mortar joints), cement a small one (nearly flat).
    static func buildSet(size: Int,
                                 albedo: [Double],
                                 height: [Double],
                                 roughness: [Double],
                                 normalStrength: Double) -> PBRTextureSet {
        let count = size * size
        var albedoBytes = [UInt8](repeating: 0, count: count * 4)
        var normalBytes = [UInt8](repeating: 0, count: count * 4)
        var dataBytes = [UInt8](repeating: 0, count: count * 4)

        // Height lookup with wrap-around indexing, so the normal map is
        // as tileable as the height field it is derived from.
        func heightAt(_ x: Int, _ y: Int) -> Double {
            let wrappedX = ((x % size) + size) % size
            let wrappedY = ((y % size) + size) % size
            return height[wrappedY * size + wrappedX]
        }

        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let byteIndex = index * 4

                // Albedo.
                albedoBytes[byteIndex] = toByte(albedo[index * 3])
                albedoBytes[byteIndex + 1] = toByte(albedo[index * 3 + 1])
                albedoBytes[byteIndex + 2] = toByte(albedo[index * 3 + 2])
                albedoBytes[byteIndex + 3] = 255

                // Data packing: R = roughness, G = height, B = 0, A = 255.
                dataBytes[byteIndex] = toByte(roughness[index])
                dataBytes[byteIndex + 1] = toByte(height[index])
                dataBytes[byteIndex + 2] = 0
                dataBytes[byteIndex + 3] = 255

                // Sobel gradients of the height field.
                let topLeft = heightAt(x - 1, y - 1)
                let top = heightAt(x, y - 1)
                let topRight = heightAt(x + 1, y - 1)
                let left = heightAt(x - 1, y)
                let right = heightAt(x + 1, y)
                let bottomLeft = heightAt(x - 1, y + 1)
                let bottom = heightAt(x, y + 1)
                let bottomRight = heightAt(x + 1, y + 1)

                let dx = (topRight + 2.0 * right + bottomRight)
                    - (topLeft + 2.0 * left + bottomLeft)
                // dy is the gradient along image rows (downwards).
                let dy = (bottomLeft + 2.0 * bottom + bottomRight)
                    - (topLeft + 2.0 * top + topRight)

                // Tangent-space normal: n = (-dh/du, -dh/dv, 1).
                // Image rows run opposite to v, so -dh/dv = +dh/drow = dy.
                let nx = -dx * normalStrength
                let ny = dy * normalStrength
                let nz = 1.0
                let inverseLength = 1.0 / (nx * nx + ny * ny + nz * nz).squareRoot()

                normalBytes[byteIndex] = toByte(nx * inverseLength * 0.5 + 0.5)
                normalBytes[byteIndex + 1] = toByte(ny * inverseLength * 0.5 + 0.5)
                normalBytes[byteIndex + 2] = toByte(nz * inverseLength * 0.5 + 0.5)
                normalBytes[byteIndex + 3] = 255
            }
        }

        return PBRTextureSet(size: size,
                             albedo: Data(albedoBytes),
                             normal: Data(normalBytes),
                             data: Data(dataBytes))
    }

    // MARK: - Concrete

    /// Mid-grey cast concrete: large soft fbm blotches, fine speckle
    /// grain, a few darker pits and stains, high roughness throughout.
    private static func makeConcrete() -> PBRTextureSet {
        let size = 512
        let count = size * size
        var rng = SplitMix64(seed: 0xC09C_E57E_0000_0001)

        let blotches = FBMNoise(basePeriod: 4, octaves: 4, rng: &rng)  // 4, 8, 16, 32
        let medium = FBMNoise(basePeriod: 16, octaves: 3, rng: &rng)   // 16, 32, 64
        let stains = FBMNoise(basePeriod: 6, octaves: 2, rng: &rng)    // 6, 12
        let pitGrid = ValueGrid(period: 32, rng: &rng)
        let speckle = makeSpeckle(count: count, rng: &rng)

        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)

        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)

                let blotch = blotches.sample(u: u, v: v)
                let mid = medium.sample(u: u, v: v)
                let grain = speckle[index]

                // Base ~0.5 grey, pushed around by the blotches and grain.
                var grey = 0.5
                    + (blotch - 0.5) * 0.22
                    + (mid - 0.5) * 0.08
                    + (grain - 0.5) * 0.10

                // Broad darker stains where the stain field runs high.
                let stain = smoothstep(0.62, 0.85, stains.sample(u: u, v: v))
                grey -= stain * 0.15

                // Sparse pits: small soft spots where the pit field peaks.
                // They darken the surface and dent the height field.
                let pit = smoothstep(0.78, 0.95, pitGrid.sample(u: u, v: v))
                grey -= pit * 0.18

                grey = clamp01(grey)
                albedo[index * 3] = grey
                albedo[index * 3 + 1] = grey
                albedo[index * 3 + 2] = grey * 0.99 // faintest cool cast

                // Low-amplitude height: blotches dominate, pits dig in.
                height[index] = clamp01(0.5
                    + (blotch - 0.5) * 0.35
                    + (mid - 0.5) * 0.25
                    + (grain - 0.5) * 0.12
                    - pit * 0.35
                    - stain * 0.05)

                // Roughness ~0.85-0.97, tracking the blotches.
                roughness[index] = clamp(0.91
                    + (blotch - 0.5) * 0.12
                    + (grain - 0.5) * 0.04, 0.85, 0.97)
            }
        }

        return buildSet(size: size,
                        albedo: albedo,
                        height: height,
                        roughness: roughness,
                        normalStrength: 1.6) // medium
    }

    // MARK: - Cement

    /// Smooth, light cement: very fine grain and only faint large-scale
    /// tonal variation. Low height amplitude and weak normals.
    private static func makeCement() -> PBRTextureSet {
        let size = 512
        let count = size * size
        var rng = SplitMix64(seed: 0xCE3E_5700_0000_0002)

        let tone = FBMNoise(basePeriod: 4, octaves: 3, rng: &rng)   // 4, 8, 16
        let fine = FBMNoise(basePeriod: 64, octaves: 2, rng: &rng)  // 64, 128
        let speckle = makeSpeckle(count: count, rng: &rng)

        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)

        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)

                let largeTone = tone.sample(u: u, v: v)
                let fineGrain = fine.sample(u: u, v: v)
                let grain = speckle[index]

                // Base ~0.62 grey with only faint variation.
                let grey = clamp01(0.62
                    + (largeTone - 0.5) * 0.08
                    + (fineGrain - 0.5) * 0.04
                    + (grain - 0.5) * 0.05)

                albedo[index * 3] = grey
                albedo[index * 3 + 1] = grey
                albedo[index * 3 + 2] = grey * 0.995

                // Very low-amplitude height.
                height[index] = clamp01(0.5
                    + (largeTone - 0.5) * 0.15
                    + (fineGrain - 0.5) * 0.10
                    + (grain - 0.5) * 0.06)

                // Roughness ~0.78-0.90.
                roughness[index] = clamp(0.84
                    + (largeTone - 0.5) * 0.08
                    + (grain - 0.5) * 0.03, 0.78, 0.90)
            }
        }

        return buildSet(size: size,
                        albedo: albedo,
                        height: height,
                        roughness: roughness,
                        normalStrength: 0.7) // weak
    }

    // MARK: - Brick

    /// Running-bond brickwork: exactly 4 bricks per row and 8 rows per
    /// tile, alternate rows offset by half a brick. Integer counts and a
    /// half-brick offset that divides the tile width make it wrap
    /// exactly; a brick split by the tile edge recombines because its
    /// identity is computed in offset (wrapped) brick space.
    private static func makeBrick() -> PBRTextureSet {
        let size = 512
        let count = size * size
        var rng = SplitMix64(seed: 0xB21C_6000_0000_0003)

        let bricksPerRow = 4
        let rows = 8
        let brickWidth = size / bricksPerRow  // 128 px
        let brickHeight = size / rows         // 64 px
        // Mortar joints ~6% of brick height: 0.06 * 64 = 3.84, so 4 px.
        let mortarPx = 4

        let faceNoise = FBMNoise(basePeriod: 8, octaves: 3, rng: &rng) // 8, 16, 32
        let mortarNoise = ValueGrid(period: 64, rng: &rng)
        let speckle = makeSpeckle(count: count, rng: &rng)

        // Per-brick tint variation, precomputed in a fixed order so the
        // result is deterministic. Each entry varies hue and lightness
        // around the red-brown base colour.
        struct BrickTint {
            let redFactor: Double
            let greenFactor: Double
            let blueFactor: Double
            let lightness: Double
        }
        var tints: [BrickTint] = []
        tints.reserveCapacity(bricksPerRow * rows)
        for _ in 0 ..< (bricksPerRow * rows) {
            tints.append(BrickTint(redFactor: 0.92 + rng.nextDouble() * 0.16,
                                   greenFactor: 0.85 + rng.nextDouble() * 0.30,
                                   blueFactor: 0.85 + rng.nextDouble() * 0.30,
                                   lightness: 0.85 + rng.nextDouble() * 0.30))
        }

        // Red-brown brick base (linear-space-ish) and mortar colour.
        let baseRed = 0.55
        let baseGreen = 0.23
        let baseBlue = 0.16
        let mortarRed = 0.64
        let mortarGreen = 0.62
        let mortarBlue = 0.58

        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)

        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let grain = speckle[index]

                let row = y / brickHeight
                let localY = y % brickHeight
                // Alternate rows shift by half a brick (64 px).
                let offset = (row % 2 == 0) ? 0 : brickWidth / 2
                let brickX = (x + offset) % size
                let column = brickX / brickWidth
                let localX = brickX % brickWidth

                let isMortar = localX < mortarPx || localY < mortarPx

                if isMortar {
                    // Light warm-grey mortar, low and very rough.
                    let variation = 0.96
                        + mortarNoise.sample(u: u, v: v) * 0.06
                        + (grain - 0.5) * 0.04
                    albedo[index * 3] = clamp01(mortarRed * variation)
                    albedo[index * 3 + 1] = clamp01(mortarGreen * variation)
                    albedo[index * 3 + 2] = clamp01(mortarBlue * variation)
                    height[index] = clamp01(0.1 + (grain - 0.5) * 0.03)
                    roughness[index] = 0.98
                } else {
                    let tint = tints[row * bricksPerRow + column]
                    let face = faceNoise.sample(u: u, v: v)

                    // Per-brick tint + broad face variation + speckle.
                    let shade = tint.lightness
                        * (1.0 + (face - 0.5) * 0.12 + (grain - 0.5) * 0.14)
                    albedo[index * 3] = clamp01(baseRed * tint.redFactor * shade)
                    albedo[index * 3 + 1] = clamp01(baseGreen * tint.greenFactor * shade)
                    albedo[index * 3 + 2] = clamp01(baseBlue * tint.blueFactor * shade)

                    // Slight dome across the brick face: a parabolic
                    // profile that falls to zero at every face edge.
                    let faceU = Double(localX - mortarPx) / Double(brickWidth - mortarPx)
                    let faceV = Double(localY - mortarPx) / Double(brickHeight - mortarPx)
                    let domeU = 1.0 - (2.0 * faceU - 1.0) * (2.0 * faceU - 1.0)
                    let domeV = 1.0 - (2.0 * faceV - 1.0) * (2.0 * faceV - 1.0)
                    let dome = domeU * domeV

                    var faceHeight = 0.7
                        + dome * 0.12
                        + (face - 0.5) * 0.08
                        + (grain - 0.5) * 0.05

                    // Slight chamfer on the two pixels nearest the
                    // mortar, so joints read as bevelled, not sliced.
                    let edgeDistance = min(localX - mortarPx,
                                           localY - mortarPx,
                                           brickWidth - 1 - localX,
                                           brickHeight - 1 - localY)
                    if edgeDistance < 2 {
                        faceHeight -= Double(2 - edgeDistance) * 0.08
                    }

                    height[index] = clamp01(faceHeight)
                    roughness[index] = clamp(0.9 + (grain - 0.5) * 0.04, 0.86, 0.94)
                }
            }
        }

        return buildSet(size: size,
                        albedo: albedo,
                        height: height,
                        roughness: roughness,
                        normalStrength: 3.0) // strong: deep mortar joints
    }

    // MARK: - Roads
    //
    // Roads share an asphalt field and a worn-paint marking pass.
    // They are fit-mapped (one whole design per face), so they do not
    // need to tile. Coordinate convention: u runs right, v runs DOWN
    // the texture (v = 0 is the far side); the road direction runs
    // along v, top to bottom.

    /// Per-pixel fields for a road, plus a `wear` noise field used to
    /// erode painted markings so the paint looks worn, not printed.
    private struct RoadFields {
        var albedo: [Double]
        var height: [Double]
        var roughness: [Double]
        var wear: [Double]
    }

    /// Asphalt: ~0.16–0.22 grey with fbm tone and speckle,
    /// roughness 0.92–0.97, height ~0.5 ± speckle.
    private static func makeRoadFields(seed: UInt64) -> RoadFields {
        let size = 512
        let count = size * size
        var rng = SplitMix64(seed: seed)
        let tone = FBMNoise(basePeriod: 4, octaves: 3, rng: &rng)
        let wearNoise = FBMNoise(basePeriod: 8, octaves: 2, rng: &rng)
        let speckle = makeSpeckle(count: count, rng: &rng)

        var fields = RoadFields(albedo: [Double](repeating: 0, count: count * 3),
                                height: [Double](repeating: 0, count: count),
                                roughness: [Double](repeating: 0, count: count),
                                wear: [Double](repeating: 0, count: count))
        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = (Double(x) + 0.5) / Double(size)
                let v = (Double(y) + 0.5) / Double(size)
                let t = tone.sample(u: u, v: v)
                let grain = speckle[index]
                let grey = clamp(0.19 + (t - 0.5) * 0.05 + (grain - 0.5) * 0.04, 0.16, 0.22)
                fields.albedo[index * 3] = grey
                fields.albedo[index * 3 + 1] = grey
                fields.albedo[index * 3 + 2] = grey * 0.99
                fields.height[index] = clamp01(0.5 + (grain - 0.5) * 0.1)
                fields.roughness[index] = clamp(0.945 + (t - 0.5) * 0.04, 0.92, 0.97)
                fields.wear[index] = wearNoise.sample(u: u, v: v)
            }
        }
        return fields
    }

    /// Paints one marking pixel, worn: the paint only survives where
    /// the wear noise clears the threshold, so edges look eroded.
    /// Paint is smoother (roughness 0.72) and slightly raised (+0.03).
    private static func paintMarking(_ fields: inout RoadFields, index: Int,
                                     red: Double, green: Double, blue: Double) {
        guard fields.wear[index] > 0.25 else { return }
        fields.albedo[index * 3] = red
        fields.albedo[index * 3 + 1] = green
        fields.albedo[index * 3 + 2] = blue
        fields.roughness[index] = 0.72
        fields.height[index] = clamp01(fields.height[index] + 0.03)
    }

    private static func paintWhite(_ fields: inout RoadFields, index: Int) {
        paintMarking(&fields, index: index, red: 0.85, green: 0.84, blue: 0.8)
    }

    private static func paintYellow(_ fields: inout RoadFields, index: Int) {
        paintMarking(&fields, index: index, red: 0.85, green: 0.65, blue: 0.15)
    }

    /// True when pixel coordinate `p` lies in a band of `width` px
    /// centred at `centre` (in px).
    private static func inBand(_ p: Int, _ centre: Double, _ width: Double) -> Bool {
        abs(Double(p) - centre) < width / 2.0
    }

    private static func roadSet(_ fields: RoadFields) -> PBRTextureSet {
        buildSet(size: 512, albedo: fields.albedo, height: fields.height,
                 roughness: fields.roughness, normalStrength: 0.8)
    }

    /// Two lanes along v: dashed white centreline, solid white edges.
    private static func makeRoadNormal() -> PBRTextureSet {
        let size = 512
        var fields = makeRoadFields(seed: 0x20AD_0000_0000_0004)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let centreline = inBand(x, 256, 6) && (y % 44) < 24
                let edge = inBand(x, 0.06 * 512, 6) || inBand(x, 0.94 * 512, 6)
                if centreline || edge { paintWhite(&fields, index: index) }
            }
        }
        return roadSet(fields)
    }

    /// Highway: concrete median with yellow edge lines, two lanes
    /// each side with dashed white dividers and solid shoulders.
    private static func makeRoadHighway() -> PBRTextureSet {
        let size = 512
        var fields = makeRoadFields(seed: 0x20AD_0000_0000_0005)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = Double(x) / Double(size)
                // Median strip: lighter concrete-ish grey.
                if u >= 0.44 && u < 0.56 {
                    fields.albedo[index * 3] = clamp01(fields.albedo[index * 3] + 0.12)
                    fields.albedo[index * 3 + 1] = clamp01(fields.albedo[index * 3 + 1] + 0.12)
                    fields.albedo[index * 3 + 2] = clamp01(fields.albedo[index * 3 + 2] + 0.12)
                    fields.roughness[index] = 0.9
                    fields.height[index] = clamp01(fields.height[index] + 0.02)
                }
                if inBand(x, 0.44 * 512, 5) || inBand(x, 0.56 * 512, 5) {
                    paintYellow(&fields, index: index)
                } else if (inBand(x, 0.22 * 512, 5) || inBand(x, 0.78 * 512, 5)) && (y % 54) < 30 {
                    paintWhite(&fields, index: index)
                } else if inBand(x, 0.03 * 512, 5) || inBand(x, 0.97 * 512, 5) {
                    paintWhite(&fields, index: index)
                }
            }
        }
        return roadSet(fields)
    }

    /// Crosswalk: zebra stripes across the full road width, plus the
    /// usual solid edge lines.
    private static func makeRoadCrosswalk() -> PBRTextureSet {
        let size = 512
        var fields = makeRoadFields(seed: 0x20AD_0000_0000_0006)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = Double(x) / Double(size)
                let stripe = (y % 52) < 26 && u >= 0.05 && u < 0.95
                let edge = inBand(x, 0.04 * 512, 6) || inBand(x, 0.96 * 512, 6)
                if stripe || edge { paintWhite(&fields, index: index) }
            }
        }
        return roadSet(fields)
    }

    /// Two roads crossing: vertical road u in [0.28, 0.72], horizontal
    /// road v in [0.28, 0.72]; darker corner blocks outside the cross,
    /// stop lines and crosswalks at the four entries, dashed
    /// centrelines on the approaches.
    private static func makeRoadIntersection() -> PBRTextureSet {
        let size = 512
        var fields = makeRoadFields(seed: 0x20AD_0000_0000_0007)
        let stripeBands = [56, 78, 100, 400, 422, 444] // 12 px stripes, 3 per entry
        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let u = Double(x) / Double(size)
                let v = Double(y) / Double(size)
                let inVertical = u >= 0.28 && u < 0.72
                let inHorizontal = v >= 0.28 && v < 0.72
                // Corner blocks: darker, rougher surround asphalt.
                if !inVertical && !inHorizontal {
                    fields.albedo[index * 3] *= 0.72
                    fields.albedo[index * 3 + 1] *= 0.72
                    fields.albedo[index * 3 + 2] *= 0.72
                    fields.roughness[index] = clamp01(fields.roughness[index] + 0.03)
                }
                // Crosswalk stripes just outside each entry.
                let zebraAcross = u >= 0.30 && u < 0.70
                    && stripeBands.contains { y >= $0 && y < $0 + 12 }
                let zebraAlong = v >= 0.30 && v < 0.70
                    && stripeBands.contains { x >= $0 && x < $0 + 12 }
                // Stop lines at the four entries.
                let stopAcross = u >= 0.30 && u < 0.70
                    && ((v >= 0.24 && v < 0.28) || (v >= 0.72 && v < 0.76))
                let stopAlong = v >= 0.30 && v < 0.70
                    && ((u >= 0.24 && u < 0.28) || (u >= 0.72 && u < 0.76))
                // Dashed centrelines on the approaches only.
                let centreV = inBand(x, 256, 5) && !inHorizontal && (y % 44) < 24
                let centreH = inBand(y, 256, 5) && !inVertical && (x % 44) < 24
                if zebraAcross || zebraAlong || stopAcross || stopAlong || centreV || centreH {
                    paintWhite(&fields, index: index)
                }
            }
        }
        return roadSet(fields)
    }

    /// Roundabout: raised island, brick-tint apron, circulating
    /// roadway with a dashed circular lane line, four entry/exit
    /// stubs, dark surround in the corners.
    private static func makeRoadRoundabout() -> PBRTextureSet {
        let size = 512
        var fields = makeRoadFields(seed: 0x20AD_0000_0000_0008)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let index = y * size + x
                let dx = Double(x) - 256
                let dy = Double(y) - 256
                let r = (dx * dx + dy * dy).squareRoot()
                let angle = atan2(dy, dx)
                let onStub = abs(dy) <= 55 || abs(dx) <= 55
                if r < 78 {
                    // Island: light concrete, raised.
                    fields.albedo[index * 3] = 0.55
                    fields.albedo[index * 3 + 1] = 0.55
                    fields.albedo[index * 3 + 2] = 0.54
                    fields.roughness[index] = 0.9
                    fields.height[index] = 0.62
                } else if r < 96 {
                    // Apron ring: brick-ish tint.
                    fields.albedo[index * 3] = 0.46
                    fields.albedo[index * 3 + 1] = 0.29
                    fields.albedo[index * 3 + 2] = 0.23
                    fields.roughness[index] = 0.9
                    fields.height[index] = 0.56
                } else if r > 196 && !onStub {
                    // Corners: dark asphalt/gravel surround.
                    fields.albedo[index * 3] = 0.09
                    fields.albedo[index * 3 + 1] = 0.11
                    fields.albedo[index * 3 + 2] = 0.08
                    fields.roughness[index] = 1.0
                    fields.height[index] = 0.45
                }
                // Dashed circular lane line at r = 146 (24-segment dash).
                let segment = Int(((angle + .pi) / (2.0 * .pi)) * 24.0)
                if abs(r - 146) < 2.5 && segment % 2 == 0 {
                    paintWhite(&fields, index: index)
                }
                // Outer edge: solid white circle at r = 196.
                if abs(r - 196) < 2.5 {
                    paintWhite(&fields, index: index)
                }
                // Dashed centrelines on the four stubs.
                let stubH = abs(dy) < 2.5 && abs(dx) > 196 && (Int(abs(dx)) % 44) < 24
                let stubV = abs(dx) < 2.5 && abs(dy) > 196 && (Int(abs(dy)) % 44) < 24
                if stubH || stubV {
                    paintWhite(&fields, index: index)
                }
            }
        }
        return roadSet(fields)
    }
}

// MARK: - Custom texture store
//
// User-painted textures from the Texture Designer live on disk as
// PNG triplets, one folder per texture:
//
//   ProjectStore.texturesRootURL/<uuidString>/
//     albedo.png  normal.png  data.png  meta.json  ({"name", "size"})
//
// The renderer enumerates them with `list()` and loads the set for a
// material whose texturePreset is "custom:<uuidString>".

enum CustomTextureStore {
    struct CustomTextureMeta: Identifiable, Equatable {
        var id: UUID
        var name: String
    }

    private struct MetaFile: Codable {
        var name: String
        var size: Int
    }

    private static func dir(_ id: UUID) -> URL {
        ProjectStore.texturesRootURL.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private static func meta(_ id: UUID) -> MetaFile? {
        let url = dir(id).appendingPathComponent("meta.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(MetaFile.self, from: data)
    }

    /// All saved custom textures, sorted by name (ties: uuidString).
    /// Unreadable or malformed entries are skipped.
    static func list() -> [CustomTextureMeta] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: ProjectStore.texturesRootURL, includingPropertiesForKeys: nil) else { return [] }
        var metas: [CustomTextureMeta] = []
        for entry in entries {
            guard let id = UUID(uuidString: entry.lastPathComponent),
                  let m = meta(id) else { continue }
            metas.append(CustomTextureMeta(id: id, name: m.name))
        }
        return metas.sorted {
            $0.name != $1.name ? $0.name < $1.name : $0.id.uuidString < $1.id.uuidString
        }
    }

    /// Loads the saved PNG triplet for `id`; nil on any failure.
    static func loadSet(id: UUID) -> PBRTextureSet? {
        let size = meta(id)?.size ?? 512
        guard size > 0,
              let albedo = readPNG(dir(id).appendingPathComponent("albedo.png"), size: size),
              let normal = readPNG(dir(id).appendingPathComponent("normal.png"), size: size),
              let data = readPNG(dir(id).appendingPathComponent("data.png"), size: size)
        else { return nil }
        return PBRTextureSet(size: size, albedo: albedo, normal: normal, data: data)
    }

    /// Saves `set` under a fresh id; returns the id, or nil on failure
    /// (the partial folder is removed).
    static func save(name: String, set: PBRTextureSet) -> UUID? {
        let id = UUID()
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: dir(id), withIntermediateDirectories: true)
        } catch { return nil }
        let ok = writePNG(set.albedo, size: set.size,
                          to: dir(id).appendingPathComponent("albedo.png"))
            && writePNG(set.normal, size: set.size,
                        to: dir(id).appendingPathComponent("normal.png"))
            && writePNG(set.data, size: set.size,
                        to: dir(id).appendingPathComponent("data.png"))
        guard ok,
              let metaData = try? JSONEncoder().encode(MetaFile(name: name, size: set.size)),
              (try? metaData.write(to: dir(id).appendingPathComponent("meta.json"),
                                   options: .atomic)) != nil
        else {
            try? fm.removeItem(at: dir(id))
            return nil
        }
        return id
    }

    static func delete(id: UUID) {
        try? FileManager.default.removeItem(at: dir(id))
    }

    // MARK: PNG IO (RGBA8, .premultipliedLast both ways so the
    // round-trip is symmetric; every texture here has alpha = 255,
    // so premultiplication is lossless)

    private static func writePNG(_ rgba: Data, size: Int, to url: URL) -> Bool {
        guard rgba.count == size * size * 4,
              let provider = CGDataProvider(data: rgba as CFData),
              let image = CGImage(width: size, height: size,
                                  bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: size * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(
                                    rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil,
                                  shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(
                url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(dest, image, nil)
        return CGImageDestinationFinalize(dest)
    }

    private static func readPNG(_ url: URL, size: Int) -> Data? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return nil }
        let storage = NSMutableData(length: size * size * 4)
        guard let bytes = storage?.mutableBytes,
              let context = CGContext(data: bytes,
                                      width: size, height: size,
                                      bitsPerComponent: 8, bytesPerRow: size * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        // Flip vertically while drawing so the buffer's first row is
        // the image's top row — the same order writePNG stored.
        context.translateBy(x: 0, y: CGFloat(size))
        context.scaleBy(x: 1, y: -1)
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
        return Data(bytes: bytes, count: size * size * 4)
    }
}
