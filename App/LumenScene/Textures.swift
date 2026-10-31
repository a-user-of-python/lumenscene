// Textures.swift
// LumenScene
//
// Fully procedural, deterministic PBR texture generation.
//
// Material presets produce square 512x512 texture sets; the road
// designs produce square 1024x1024 sets (TextureGen.roadSize):
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
    /// the whole 1024x1024 design spans one face, so unlike the material
    /// presets they are not required to tile. Road direction runs
    /// along v — top to bottom of the texture.
    static let roadNames = ["road-highway", "road-normal", "road-intersection", "road-crosswalk", "road-roundabout"]

    /// Resolution of every road texture set. Material presets stay at
    /// 512; roads get 1024 so lane markings, cracks and paint wear
    /// survive at glancing angles in the path tracer.
    static let roadSize = 1024

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
    // Roads share a parameterised asphalt base and a worn-paint pass.
    // They are fit-mapped (one whole 1024x1024 design spans one face),
    // so they do not need to tile. Coordinate convention: u runs right,
    // v runs DOWN the texture (v = 0 is the far side); the road
    // direction runs along v, top to bottom.
    //
    // Generation cost structure: every noise-derived quantity is
    // precomputed ONCE per set into a flat [Double] field (large tone,
    // mid tone, paint wear, oil, crack lines, tar-snake lines, two
    // speckles, patch rectangles rasterised into tone/edge fields).
    // Low-frequency fields are sampled on a small grid and bilinearly
    // upsampled; only the crack field is evaluated at full resolution.
    // Each road type then makes ONE compose pass over the pixels,
    // reading those arrays and adding cheap analytic marking shapes
    // (distance bands / intervals with ~1.5 px anti-aliasing). No
    // per-pixel noise evaluation happens in the compose passes.

    /// Precomputed per-pixel fields shared by every road design.
    private struct RoadFields {
        let large: [Double]     // broad fbm tone: repairs, age, patchiness
        let mid: [Double]       // medium fbm tone / aggregate shading
        let wear: [Double]      // paint-wear field (markings)
        let oil: [Double]       // low-frequency oil-stain field (raw)
        let crack: [Double]     // thin ridge-noise crack lines, 0...1
        let snake: [Double]     // wandering tar-snake lines, 0...1
        let speckleA: [Double]  // per-pixel aggregate grain
        let speckleB: [Double]  // second per-pixel grain / paint break-up
        let patchTone: [Double] // per-pixel tone delta from patch rectangles
        let patchEdge: [Double] // 0...1 near a patch rectangle's tar-sealed edge
    }

    /// Samples `noise` on a `resolution`-square grid and bilinearly
    /// upsamples it to `size`-square. The underlying value noise is
    /// periodic, and the upsample wraps, so the field stays periodic.
    private static func roadUpsampledField(_ noise: FBMNoise, resolution: Int, size: Int) -> [Double] {
        var low = [Double](repeating: 0, count: resolution * resolution)
        for yy in 0 ..< resolution {
            for xx in 0 ..< resolution {
                low[yy * resolution + xx] = noise.sample(u: (Double(xx) + 0.5) / Double(resolution),
                                                         v: (Double(yy) + 0.5) / Double(resolution))
            }
        }
        var out = [Double](repeating: 0, count: size * size)
        let scale = Double(resolution) / Double(size)
        for y in 0 ..< size {
            let gy = (Double(y) + 0.5) * scale - 0.5
            let y0f = gy.rounded(.down)
            let fy = gy - y0f
            let y0 = ((Int(y0f) % resolution) + resolution) % resolution
            let y1 = (y0 + 1) % resolution
            for x in 0 ..< size {
                let gx = (Double(x) + 0.5) * scale - 0.5
                let x0f = gx.rounded(.down)
                let fx = gx - x0f
                let x0 = ((Int(x0f) % resolution) + resolution) % resolution
                let x1 = (x0 + 1) % resolution
                let top = low[y0 * resolution + x0]
                    + (low[y0 * resolution + x1] - low[y0 * resolution + x0]) * fx
                let bottom = low[y1 * resolution + x0]
                    + (low[y1 * resolution + x1] - low[y1 * resolution + x0]) * fx
                out[y * size + x] = top + (bottom - top) * fy
            }
        }
        return out
    }

    /// Samples `noise` at full texture resolution.
    private static func roadFullField(_ noise: FBMNoise, size: Int) -> [Double] {
        var out = [Double](repeating: 0, count: size * size)
        for y in 0 ..< size {
            for x in 0 ..< size {
                out[y * size + x] = noise.sample(u: (Double(x) + 0.5) / Double(size),
                                                 v: (Double(y) + 0.5) / Double(size))
            }
        }
        return out
    }

    /// Builds every shared field for one road set, deterministically
    /// from `seed` (fixed per road type, so output is byte-identical
    /// on every launch).
    private static func makeRoadFields(seed: UInt64) -> RoadFields {
        let size = roadSize
        let count = size * size
        var rng = SplitMix64(seed: seed)

        // Noise sources, consumed from the PRNG in a fixed order.
        let largeNoise = FBMNoise(basePeriod: 4, octaves: 4, rng: &rng)   // 4, 8, 16, 32
        let midNoise = FBMNoise(basePeriod: 16, octaves: 3, rng: &rng)    // 16, 32, 64
        let wearNoise = FBMNoise(basePeriod: 8, octaves: 3, rng: &rng)    // 8, 16, 32
        let oilNoise = FBMNoise(basePeriod: 5, octaves: 2, rng: &rng)     // 5, 10
        let crackNoise = FBMNoise(basePeriod: 13, octaves: 2, rng: &rng)  // 13, 26
        let snakeNoise = FBMNoise(basePeriod: 5, octaves: 2, rng: &rng)   // 5, 10

        let large = roadUpsampledField(largeNoise, resolution: 256, size: size)
        let mid = roadUpsampledField(midNoise, resolution: 512, size: size)
        let wear = roadUpsampledField(wearNoise, resolution: 512, size: size)
        let oil = roadUpsampledField(oilNoise, resolution: 256, size: size)
        let snakeRaw = roadUpsampledField(snakeNoise, resolution: 512, size: size)
        let crackRaw = roadFullField(crackNoise, size: size)
        let speckleA = makeSpeckle(count: count, rng: &rng)
        let speckleB = makeSpeckle(count: count, rng: &rng)

        // Ridge transform: a noise contour becomes a thin soft line.
        // Cracks hug the 0.50 contour of a fine field (thin, branching
        // feel); tar snakes hug the 0.56 contour of a coarse field, so
        // only a few long wandering repairs appear per face.
        var crack = [Double](repeating: 0, count: count)
        var snake = [Double](repeating: 0, count: count)
        for i in 0 ..< count {
            let cd = clamp01(1.0 - abs(crackRaw[i] - 0.50) / 0.030)
            crack[i] = cd * cd * (3.0 - 2.0 * cd)
            let sd = clamp01(1.0 - abs(snakeRaw[i] - 0.56) / 0.034)
            snake[i] = sd * sd * (3.0 - 2.0 * sd)
        }

        // Patch rectangles: resurfaced areas with a slightly different
        // tone, rasterised here so the compose pass is a plain lookup.
        // The outer 4 px of each rectangle is its tar-sealed edge.
        var patchTone = [Double](repeating: 0, count: count)
        var patchEdge = [Double](repeating: 0, count: count)
        for _ in 0 ..< 6 {
            let pw = 110 + Int(rng.nextDouble() * 170.0)
            let ph = 90 + Int(rng.nextDouble() * 190.0)
            let px = 36 + Int(rng.nextDouble() * Double(size - pw - 72))
            let py = 36 + Int(rng.nextDouble() * Double(size - ph - 72))
            let delta = (rng.nextDouble() - 0.5) * 0.055
            for yy in py ..< (py + ph) {
                for xx in px ..< (px + pw) {
                    let idx = yy * size + xx
                    patchTone[idx] = delta
                    let dEdge = min(xx - px, px + pw - 1 - xx, yy - py, py + ph - 1 - yy)
                    if dEdge < 4 {
                        patchEdge[idx] = Double(4 - dEdge) / 4.0
                    }
                }
            }
        }

        return RoadFields(large: large, mid: mid, wear: wear, oil: oil,
                          crack: crack, snake: snake,
                          speckleA: speckleA, speckleB: speckleB,
                          patchTone: patchTone, patchEdge: patchEdge)
    }

    // MARK: Road shape helpers (all cheap, analytic, anti-aliased)

    /// Coverage of a band centred at `centre` with half-width
    /// `halfWidth`, at coordinate `coord`. Falls 1 -> 0 over ~1.5 px.
    private static func roadBand(_ coord: Double, _ centre: Double, _ halfWidth: Double) -> Double {
        clamp01((halfWidth - abs(coord - centre)) / 1.5 + 0.5)
    }

    /// Coverage of the interval [lo, hi] at `coord`, ~1.5 px edges.
    private static func roadInterval(_ coord: Double, _ lo: Double, _ hi: Double) -> Double {
        clamp01(min(coord - lo, hi - coord) / 1.5 + 0.5)
    }

    /// One centred dash per `period` (dash length `dash`), repeating
    /// along `coord`. The lane-marking periods used here (64, 128)
    /// divide 1024 exactly, so a gap — never a clipped dash —
    /// straddles both texture borders. (The 24 px rumble grooves are
    /// surface texture, not markings, and simply run off the edge.)
    private static func roadDash(_ coord: Double, _ period: Double, _ dash: Double) -> Double {
        let phase = coord - (coord / period).rounded(.down) * period
        return roadInterval(phase, (period - dash) / 2.0, (period + dash) / 2.0)
    }

    /// Polished wheel-track profile along one axis: 1 at each track
    /// centre, falling smoothly to 0 at `halfWidth`. One entry per
    /// pixel coordinate, so the compose pass is an array lookup.
    private static func roadWheelProfile(centres: [Double], halfWidth: Double, size: Int) -> [Double] {
        var out = [Double](repeating: 0, count: size)
        for i in 0 ..< size {
            let coord = Double(i) + 0.5
            var best = 0.0
            for centre in centres {
                let d = abs(coord - centre)
                if d < halfWidth {
                    let t = 1.0 - (d / halfWidth) * (d / halfWidth)
                    let v = t * t
                    if v > best { best = v }
                }
            }
            out[i] = best
        }
        return out
    }

    /// The shared asphalt surface at one pixel.
    /// - wheel: 0...1 polished wheel-track factor (darker, shinier).
    /// - edge: 0...1 proximity to the texture border (deterioration).
    private static func roadAsphalt(_ fields: RoadFields, index: Int,
                                    wheel: Double, edge: Double)
        -> (r: Double, g: Double, b: Double, h: Double, rough: Double) {
        let large = fields.large[index]
        let mid = fields.mid[index]
        let speckA = fields.speckleA[index]
        let speckB = fields.speckleB[index]
        let oilF = smoothstep(0.66, 0.88, fields.oil[index])
        let crack = fields.crack[index]
        let snake = fields.snake[index]
        let patchEdge = fields.patchEdge[index]

        // Multi-scale tone: broad age/repair patches, medium shading,
        // then per-pixel aggregate stone (both light and dark grains).
        var grey = 0.190
            + (large - 0.5) * 0.085
            + (mid - 0.5) * 0.050
            + (speckA - 0.5) * 0.055
            + (speckB - 0.5) * 0.028
        grey += smoothstep(0.93, 1.0, speckA) * 0.07  // pale stones
        grey -= smoothstep(0.93, 1.0, speckB) * 0.06  // dark stones
        grey += fields.patchTone[index]               // resurfaced patches
        grey -= patchEdge * 0.045                     // tar-sealed patch edges
        grey -= oilF * 0.055                          // oil stains
        grey -= crack * 0.065                         // cracks read dark
        grey -= snake * 0.050                         // tar snakes are near-black
        grey -= wheel * 0.028                         // polished, tyre-darkened tracks
        grey += edge * 0.018                          // faded, dusty deterioration at borders
        grey = clamp(grey, 0.12, 0.28)

        // Height: aggregate micro-relief dominates; cracks dip, tar
        // snakes and sealed patch edges sit slightly proud, edges of
        // the plot crumble down a touch.
        var height = 0.50
            + (mid - 0.5) * 0.07
            + (speckA - 0.5) * 0.085
            + (speckB - 0.5) * 0.045
        height -= crack * 0.11
        height += snake * 0.045
        height += patchEdge * 0.018
        height -= oilF * 0.012
        height -= edge * 0.015

        // Roughness: asphalt is rough, but wheel tracks, oil and tar
        // are polished smooth — that contrast is what plays in the
        // light. Paint (applied later) sits between the two.
        var rough = 0.94
            + (large - 0.5) * 0.035
            + (speckA - 0.5) * 0.022
        rough -= wheel * 0.17
        rough -= oilF * 0.28
        rough -= snake * 0.22
        rough -= patchEdge * 0.18
        if fields.patchTone[index] != 0 { rough -= 0.025 }
        rough += crack * 0.015
        rough += edge * 0.03

        return (grey, grey, grey * 0.99,
                clamp01(height), clamp(rough, 0.55, 0.985))
    }

    /// Blends one worn-paint colour over the surface at one pixel.
    /// Coverage is multiplied by a wear mask built from the wear field
    /// plus speckle break-up, and reduced further inside wheel tracks,
    /// so paint looks scrubbed away where tyres run and fresher
    /// between them. Surviving paint is slightly raised and smoother
    /// than asphalt (roughness pulled toward 0.70), never glossy.
    private static func roadPaint(_ red: inout Double, _ green: inout Double, _ blue: inout Double,
                                  _ height: inout Double, _ roughness: inout Double,
                                  coverage: Double, fields: RoadFields, index: Int, wheel: Double,
                                  paintRed: Double, paintGreen: Double, paintBlue: Double) {
        if coverage <= 0 { return }
        let wearRaw = fields.wear[index] * 0.78 + fields.speckleB[index] * 0.22
        let mask = smoothstep(0.30, 0.56, wearRaw - wheel * 0.24)
        let eff = coverage * mask
        if eff <= 0.003 { return }
        // Grime variation across the surviving paint.
        let tint = 0.90 + fields.speckleA[index] * 0.16
        red += (paintRed * tint - red) * eff
        green += (paintGreen * tint - green) * eff
        blue += (paintBlue * tint - blue) * eff
        height += 0.04 * eff
        roughness += (0.70 - roughness) * eff
    }

    private static func roadWhitePaint(_ red: inout Double, _ green: inout Double, _ blue: inout Double,
                                       _ height: inout Double, _ roughness: inout Double,
                                       coverage: Double, fields: RoadFields, index: Int, wheel: Double) {
        roadPaint(&red, &green, &blue, &height, &roughness, coverage: coverage,
                  fields: fields, index: index, wheel: wheel,
                  paintRed: 0.86, paintGreen: 0.85, paintBlue: 0.81)
    }

    private static func roadYellowPaint(_ red: inout Double, _ green: inout Double, _ blue: inout Double,
                                        _ height: inout Double, _ roughness: inout Double,
                                        coverage: Double, fields: RoadFields, index: Int, wheel: Double) {
        roadPaint(&red, &green, &blue, &height, &roughness, coverage: coverage,
                  fields: fields, index: index, wheel: wheel,
                  paintRed: 0.83, paintGreen: 0.62, paintBlue: 0.14)
    }

    /// Border proximity 0...1: 1 at the texture edge, 0 from 48 px in.
    private static func roadEdgeFactor(x: Int, y: Int, size: Int) -> Double {
        let d = min(x, y, size - 1 - x, size - 1 - y)
        return clamp01(1.0 - Double(d) / 48.0)
    }

    private static func roadSet(albedo: [Double], height: [Double], roughness: [Double]) -> PBRTextureSet {
        buildSet(size: roadSize, albedo: albedo, height: height,
                 roughness: roughness, normalStrength: 1.15)
    }

    /// Highway: three lanes, one way. Solid white edge lines at
    /// x = 108 / 916, dashed white dividers at x = 377 / 647 (period
    /// 128, dash 72 — gaps straddle the borders), rumble strips in
    /// both shoulders, two polished wheel tracks per lane.
    private static func makeRoadHighway() -> PBRTextureSet {
        let size = roadSize
        let count = size * size
        let fields = makeRoadFields(seed: 0x20AD_0000_0000_0005)
        let wheelX = roadWheelProfile(centres: [180, 305, 450, 574, 719, 844],
                                      halfWidth: 42, size: size)
        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)
        for y in 0 ..< size {
            let yd = Double(y) + 0.5
            // Rumble grooves run across the shoulders: a 24 px period
            // along v, shared by both shoulders this row.
            let groove = roadDash(yd, 24, 12)
            for x in 0 ..< size {
                let index = y * size + x
                let xd = Double(x) + 0.5
                let wheel = wheelX[x]
                let pixel = roadAsphalt(fields, index: index, wheel: wheel,
                                        edge: roadEdgeFactor(x: x, y: y, size: size))
                var r = pixel.r, g = pixel.g, b = pixel.b
                var h = pixel.h, rough = pixel.rough

                // Rumble strips: outside the edge lines only.
                let rumble = max(roadInterval(xd, 20, 92), roadInterval(xd, 932, 1004))
                if rumble > 0 {
                    h -= rumble * groove * 0.055
                    rough += rumble * groove * 0.02
                }

                let edgeCov = max(roadBand(xd, 108, 6), roadBand(xd, 916, 6))
                roadWhitePaint(&r, &g, &b, &h, &rough, coverage: edgeCov,
                               fields: fields, index: index, wheel: wheel)
                let dividerCov = max(roadBand(xd, 377, 6), roadBand(xd, 647, 6)) * roadDash(yd, 128, 72)
                roadWhitePaint(&r, &g, &b, &h, &rough, coverage: dividerCov,
                               fields: fields, index: index, wheel: wheel)

                albedo[index * 3] = clamp01(r)
                albedo[index * 3 + 1] = clamp01(g)
                albedo[index * 3 + 2] = clamp01(b)
                height[index] = clamp01(h)
                roughness[index] = clamp01(rough)
            }
        }
        return roadSet(albedo: albedo, height: height, roughness: roughness)
    }

    /// Two-lane road, one lane each way: solid white edge lines at
    /// x = 92 / 932 and a dashed yellow centre line (period 128,
    /// dash 72). Two wheel tracks per lane.
    private static func makeRoadNormal() -> PBRTextureSet {
        let size = roadSize
        let count = size * size
        let fields = makeRoadFields(seed: 0x20AD_0000_0000_0004)
        let wheelX = roadWheelProfile(centres: [232, 372, 652, 792],
                                      halfWidth: 44, size: size)
        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)
        for y in 0 ..< size {
            let yd = Double(y) + 0.5
            let centreDash = roadDash(yd, 128, 72)
            for x in 0 ..< size {
                let index = y * size + x
                let xd = Double(x) + 0.5
                let wheel = wheelX[x]
                let pixel = roadAsphalt(fields, index: index, wheel: wheel,
                                        edge: roadEdgeFactor(x: x, y: y, size: size))
                var r = pixel.r, g = pixel.g, b = pixel.b
                var h = pixel.h, rough = pixel.rough

                let edgeCov = max(roadBand(xd, 92, 6), roadBand(xd, 932, 6))
                roadWhitePaint(&r, &g, &b, &h, &rough, coverage: edgeCov,
                               fields: fields, index: index, wheel: wheel)
                let centreCov = roadBand(xd, 512, 7) * centreDash
                roadYellowPaint(&r, &g, &b, &h, &rough, coverage: centreCov,
                                fields: fields, index: index, wheel: wheel)

                albedo[index * 3] = clamp01(r)
                albedo[index * 3 + 1] = clamp01(g)
                albedo[index * 3 + 2] = clamp01(b)
                height[index] = clamp01(h)
                roughness[index] = clamp01(rough)
            }
        }
        return roadSet(albedo: albedo, height: height, roughness: roughness)
    }

    /// Intersection: a vertical road (x in [256, 768]) crossing a
    /// horizontal road (y in [256, 768]). The centre box carries NO
    /// lane lines. On each approach, from the box outward: a
    /// continental crosswalk (bars parallel to traffic, zones 96 px
    /// deep, bars on a 64 px period over x/y in [288, 736]), then a
    /// 16 px stop line, then a short dashed yellow centre line to the
    /// texture edge. Corners are lighter concrete with expansion
    /// joints and a worn circular patch at each inner corner.
    private static func makeRoadIntersection() -> PBRTextureSet {
        let size = roadSize
        let count = size * size
        let fields = makeRoadFields(seed: 0x20AD_0000_0000_0007)
        // Wheel tracks for the vertical road (x) and horizontal road (y):
        // two lanes each, centres 384 / 640, tracks offset +/-64.
        let wheelTracks = roadWheelProfile(centres: [320, 448, 576, 704],
                                           halfWidth: 42, size: size)
        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)
        for y in 0 ..< size {
            let yd = Double(y) + 0.5
            let inHorizontal = yd >= 256 && yd < 768
            for x in 0 ..< size {
                let index = y * size + x
                let xd = Double(x) + 0.5
                let inVertical = xd >= 256 && xd < 768
                var wheel = 0.0
                if inVertical { wheel = max(wheel, wheelTracks[x]) }
                if inHorizontal { wheel = max(wheel, wheelTracks[y]) }
                let pixel = roadAsphalt(fields, index: index, wheel: wheel,
                                        edge: roadEdgeFactor(x: x, y: y, size: size))
                var r = pixel.r, g = pixel.g, b = pixel.b
                var h = pixel.h, rough = pixel.rough

                if !inVertical && !inHorizontal {
                    // Corner block: light concrete sidewalk, slightly
                    // raised and very rough, with expansion joints.
                    let concrete = 0.44
                        + (fields.large[index] - 0.5) * 0.06
                        + (fields.speckleA[index] - 0.5) * 0.04
                    r += (concrete - r) * 0.9
                    g += (concrete - g) * 0.9
                    b += (concrete * 0.99 - b) * 0.9
                    h += 0.035
                    rough = 0.96
                    let joint = max(max(roadBand(xd, 128, 1.5), roadBand(xd, 896, 1.5)),
                                    max(roadBand(yd, 128, 1.5), roadBand(yd, 896, 1.5)))
                    r -= joint * 0.04
                    g -= joint * 0.04
                    b -= joint * 0.04
                    h -= joint * 0.03
                    // Worn patch around the inner corner point.
                    let cdx = min(abs(xd - 256), abs(xd - 768))
                    let cdy = min(abs(yd - 256), abs(yd - 768))
                    let cornerDist = (cdx * cdx + cdy * cdy).squareRoot()
                    let patch = clamp01(1.0 - cornerDist / 110.0)
                    r -= patch * 0.05
                    g -= patch * 0.05
                    b -= patch * 0.05
                    h -= patch * 0.01
                } else {
                    // Continental crosswalk bars on all four approaches.
                    let barAcross = roadDash(xd - 288, 64, 32) * roadInterval(xd, 288, 736)
                    let zebraBottom = barAcross * roadInterval(yd, 780, 876)
                    let zebraTop = barAcross * roadInterval(yd, 148, 244)
                    let barAlong = roadDash(yd - 288, 64, 32) * roadInterval(yd, 288, 736)
                    let zebraRight = barAlong * roadInterval(xd, 780, 876)
                    let zebraLeft = barAlong * roadInterval(xd, 148, 244)
                    let zebra = max(max(zebraBottom, zebraTop), max(zebraRight, zebraLeft))
                    roadWhitePaint(&r, &g, &b, &h, &rough, coverage: zebra,
                                   fields: fields, index: index, wheel: wheel)

                    // Stop lines, just outside the crosswalks.
                    let stopBottom = roadInterval(yd, 892, 908) * roadInterval(xd, 288, 736)
                    let stopTop = roadInterval(yd, 116, 132) * roadInterval(xd, 288, 736)
                    let stopRight = roadInterval(xd, 892, 908) * roadInterval(yd, 288, 736)
                    let stopLeft = roadInterval(xd, 116, 132) * roadInterval(yd, 288, 736)
                    let stop = max(max(stopBottom, stopTop), max(stopRight, stopLeft))
                    roadWhitePaint(&r, &g, &b, &h, &rough, coverage: stop,
                                   fields: fields, index: index, wheel: 0)

                    // Dashed yellow centrelines on the short approach
                    // stubs beyond the stop lines (never in the box).
                    let centreV = roadBand(xd, 512, 6)
                        * max(roadInterval(yd, 908, 1024), roadInterval(yd, 0, 116))
                        * roadDash(yd, 128, 72)
                    let centreH = roadBand(yd, 512, 6)
                        * max(roadInterval(xd, 908, 1024), roadInterval(xd, 0, 116))
                        * roadDash(xd, 128, 72)
                    roadYellowPaint(&r, &g, &b, &h, &rough, coverage: max(centreV, centreH),
                                    fields: fields, index: index, wheel: wheel)
                }

                albedo[index * 3] = clamp01(r)
                albedo[index * 3 + 1] = clamp01(g)
                albedo[index * 3 + 2] = clamp01(b)
                height[index] = clamp01(h)
                roughness[index] = clamp01(rough)
            }
        }
        return roadSet(albedo: albedo, height: height, roughness: roughness)
    }

    /// Crosswalk: a full zebra crossing. Wide white stripes (64 px,
    /// on a 128 px period so gaps straddle the top/bottom borders)
    /// run across the whole texture, perpendicular to travel, between
    /// solid white edge lines at x = 56 / 968. Two wheel tracks
    /// (x = 352 / 672) polish the asphalt and scrub the paint almost
    /// away where tyres cross the stripes.
    private static func makeRoadCrosswalk() -> PBRTextureSet {
        let size = roadSize
        let count = size * size
        let fields = makeRoadFields(seed: 0x20AD_0000_0000_0006)
        let wheelX = roadWheelProfile(centres: [352, 672], halfWidth: 70, size: size)
        let gapX = roadWheelProfile(centres: [352, 672], halfWidth: 58, size: size)
        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)
        for y in 0 ..< size {
            let yd = Double(y) + 0.5
            let stripe = roadDash(yd, 128, 64)
            for x in 0 ..< size {
                let index = y * size + x
                let xd = Double(x) + 0.5
                let wheel = wheelX[x]
                let pixel = roadAsphalt(fields, index: index, wheel: wheel,
                                        edge: roadEdgeFactor(x: x, y: y, size: size))
                var r = pixel.r, g = pixel.g, b = pixel.b
                var h = pixel.h, rough = pixel.rough

                // Stripes span between the edge lines; their paint
                // wear is driven by the (narrower) wheel-gap profile,
                // so the two tyre paths read as gaps in the zebra.
                let stripeCov = stripe * roadInterval(xd, 72, 952)
                roadWhitePaint(&r, &g, &b, &h, &rough, coverage: stripeCov,
                               fields: fields, index: index, wheel: gapX[x])
                let edgeCov = max(roadBand(xd, 56, 6), roadBand(xd, 968, 6))
                roadWhitePaint(&r, &g, &b, &h, &rough, coverage: edgeCov,
                               fields: fields, index: index, wheel: 0)

                albedo[index * 3] = clamp01(r)
                albedo[index * 3 + 1] = clamp01(g)
                albedo[index * 3 + 2] = clamp01(b)
                height[index] = clamp01(h)
                roughness[index] = clamp01(rough)
            }
        }
        return roadSet(albedo: albedo, height: height, roughness: roughness)
    }

    /// Roundabout: a raised concrete island (r < 180, with a proud
    /// curb ring at its edge), a lighter warm-tinted apron ring
    /// (180...235) with radial joints every 30 degrees and one
    /// concentric joint, and an annular asphalt driving lane
    /// (235...430) with a polished circulating wheel band at r = 330
    /// and a dashed circular lane line on the same radius (24 arc
    /// segments, suppressed at the four entries). The solid outer
    /// edge circle at r = 430 breaks at the cardinal entry/exit
    /// stubs, which run to the texture edges with their own edge
    /// lines and dashed centrelines. Dashed yield arcs sit just
    /// outside the circle at r = 452 across each entry. Corners
    /// outside the circle and stubs are dark gravel surround.
    private static func makeRoadRoundabout() -> PBRTextureSet {
        let size = roadSize
        let count = size * size
        let fields = makeRoadFields(seed: 0x20AD_0000_0000_0008)
        // Wheel tracks along the straight stubs (either axis).
        let stubWheel = roadWheelProfile(centres: [457, 567], halfWidth: 40, size: size)
        var albedo = [Double](repeating: 0, count: count * 3)
        var height = [Double](repeating: 0, count: count)
        var roughness = [Double](repeating: 0, count: count)
        for y in 0 ..< size {
            let yd = Double(y) + 0.5
            let dy = yd - 512
            let inStubH = abs(dy) <= 110
            for x in 0 ..< size {
                let index = y * size + x
                let xd = Double(x) + 0.5
                let dx = xd - 512
                let inStubV = abs(dx) <= 110
                let r = (dx * dx + dy * dy).squareRoot()

                // Wheel polish: circulating band in the annulus,
                // straight tracks on the stubs.
                var wheel = 0.0
                if r >= 235 && r <= 430 {
                    let dr = abs(r - 330)
                    if dr < 48 {
                        let t = 1.0 - (dr / 48) * (dr / 48)
                        wheel = t * t
                    }
                }
                if r > 424 && inStubH { wheel = max(wheel, stubWheel[y]) }
                if r > 424 && inStubV { wheel = max(wheel, stubWheel[x]) }

                let pixel = roadAsphalt(fields, index: index, wheel: wheel,
                                        edge: roadEdgeFactor(x: x, y: y, size: size))
                var rr = pixel.r, g = pixel.g, b = pixel.b
                var h = pixel.h, rough = pixel.rough

                if r < 235 {
                    // Island and apron: concrete surfaces, raised.
                    // Angle is only needed in here (radial joints), so
                    // atan2 runs on ~18% of pixels, not the full face.
                    let variation = (fields.large[index] - 0.5) * 0.07
                        + (fields.speckleA[index] - 0.5) * 0.05
                    if r < 180 {
                        let grey = 0.54 + variation
                        rr = grey; g = grey; b = grey * 0.99
                        h = 0.65
                        rough = 0.90
                    } else {
                        let tint = 1.0 + variation * 1.6
                        rr = 0.47 * tint; g = 0.405 * tint; b = 0.355 * tint
                        h = 0.575
                        rough = 0.92
                    }
                    // Proud curb ring where island meets apron.
                    h += roadBand(r, 180, 4) * 0.03
                    if r > 24 {
                        let theta = atan2(dy, dx)
                        let sectorPos = (theta + .pi) / (2.0 * .pi) * 12.0
                        let frac = sectorPos - sectorPos.rounded(.down)
                        let arcToJoint = min(frac, 1.0 - frac) * (2.0 * .pi * r / 12.0)
                        let joint = clamp01((2.0 - arcToJoint) / 1.5 + 0.5)
                        rr -= joint * 0.07
                        g -= joint * 0.07
                        b -= joint * 0.07
                        h -= joint * 0.025
                    }
                    if r >= 180 {
                        let ringJoint = roadBand(r, 207, 1.8)
                        rr -= ringJoint * 0.06
                        g -= ringJoint * 0.06
                        b -= ringJoint * 0.06
                        h -= ringJoint * 0.02
                    }
                } else if r > 430 && !inStubH && !inStubV {
                    // Corner surround: dark gravel, low and very rough.
                    let grey = 0.085
                        + (fields.mid[index] - 0.5) * 0.03
                        + (fields.speckleA[index] - 0.5) * 0.04
                    rr = grey * 0.9; g = grey * 1.15; b = grey * 0.85
                    h = 0.44
                    rough = 1.0
                } else {
                    // Annulus and stubs: markings.
                    // Dashed circular lane line at r = 330, breaking
                    // where the stubs join (either coordinate small
                    // relative to the radius puts the point in a stub
                    // mouth, and the angular test keeps the break
                    // centred on the cardinal directions).
                    let laneBand = roadBand(r, 330, 6)
                    if laneBand > 0 {
                        let theta = atan2(dy, dx)
                        let segPos = (theta + .pi) / (2.0 * .pi) * 24.0
                        let seg = Int(segPos.rounded(.down))
                        // Angular distance to the nearest cardinal.
                        let quarter = (theta / (.pi / 2.0)).rounded()
                        let toCardinal = abs(theta - quarter * (.pi / 2.0))
                        if seg % 2 == 0 && toCardinal > 0.30 {
                            roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: laneBand,
                                           fields: fields, index: index, wheel: wheel)
                        }
                    }
                    // Yield arcs at r = 452, only across the entries
                    // (points there lie inside a stub's width), dashed
                    // into 40 arc segments.
                    let yieldBand = roadBand(r, 452, 5)
                    if yieldBand > 0 && (abs(dy) < 104 || abs(dx) < 104) {
                        let theta = atan2(dy, dx)
                        let segPos = (theta + .pi) / (2.0 * .pi) * 40.0
                        let seg = Int(segPos.rounded(.down))
                        if seg % 2 == 0 {
                            roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: yieldBand,
                                           fields: fields, index: index, wheel: 0)
                        }
                    }
                    // Solid outer edge circle, broken at the stubs.
                    if !(inStubH || inStubV) {
                        let edgeCov = roadBand(r, 430, 6)
                        roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: edgeCov,
                                       fields: fields, index: index, wheel: 0)
                    }
                    // Stub markings, fading in just outside the circle.
                    let stubFade = clamp01((r - 424) / 8.0)
                    if stubFade > 0 {
                        if inStubH {
                            let stubEdge = max(roadBand(yd, 402, 5), roadBand(yd, 622, 5))
                            roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: stubEdge * stubFade,
                                           fields: fields, index: index, wheel: 0)
                            let stubCentre = roadBand(yd, 512, 5) * roadDash(xd, 128, 72)
                            roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: stubCentre * stubFade,
                                           fields: fields, index: index, wheel: wheel)
                        }
                        if inStubV {
                            let stubEdge = max(roadBand(xd, 402, 5), roadBand(xd, 622, 5))
                            roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: stubEdge * stubFade,
                                           fields: fields, index: index, wheel: 0)
                            let stubCentre = roadBand(xd, 512, 5) * roadDash(yd, 128, 72)
                            roadWhitePaint(&rr, &g, &b, &h, &rough, coverage: stubCentre * stubFade,
                                           fields: fields, index: index, wheel: wheel)
                        }
                    }
                }

                albedo[index * 3] = clamp01(rr)
                albedo[index * 3 + 1] = clamp01(g)
                albedo[index * 3 + 2] = clamp01(b)
                height[index] = clamp01(h)
                roughness[index] = clamp01(rough)
            }
        }
        return roadSet(albedo: albedo, height: height, roughness: roughness)
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
