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

// MARK: - Public contract

struct PBRTextureSet {
    let size: Int          // square, e.g. 512
    let albedo: Data       // RGBA8, size*size*4 bytes
    let normal: Data       // RGBA8 tangent-space normal map (rgb = n*0.5+0.5, a=255)
    let data: Data         // RGBA8: R = roughness*255, G = height*255 (0..1), B = 0, A = 255
}

enum TextureGen {
    static let presetNames = ["concrete", "cement", "brick"]

    /// Builds the texture set for `preset` (one of `presetNames`).
    /// Unknown names fall back to "concrete"; callers are expected to
    /// pass a value from `presetNames`.
    static func makeSet(preset: String) -> PBRTextureSet {
        switch preset {
        case "cement":
            return makeCement()
        case "brick":
            return makeBrick()
        default: // "concrete", plus any unexpected name
            return makeConcrete()
        }
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
    private static func buildSet(size: Int,
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
}
