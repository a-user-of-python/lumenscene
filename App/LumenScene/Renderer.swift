// Renderer.swift — owns the Metal device, the merged scene buffers,
// the ray-tracing acceleration structure, and the accumulation
// render loop. All scene triangles are baked into world space on
// the CPU (modest scenes; edits reset accumulation anyway), which
// keeps the GPU side to one BLAS and one material per object.
//
// V2 notes (performance + quality):
//   * Render-on-change: the view only draws while `frameWanted` —
//     a static, converged scene costs zero GPU/CPU.
//   * Rebuild split: material-only edits (colour, emissive, glass
//     sliders) rewrite the material + emissive-CDF buffers from the
//     cached CPU geometry and never touch the acceleration
//     structure — the AS is rebuilt only when geometry changes.
//   * Temporal reprojection: accumulation lives in ping-pong
//     textures with a matching pair of "guide" textures (primary
//     hit world positions). Camera moves reproject the converged
//     history instead of restarting from noise; the settled image
//     is rendered at full resolution with a generous sample cap.
//   * Adaptive quality: resolution scale, samples/frame, bounce
//     and glass budgets follow motion, the user's quality
//     preference, and the device thermal state (emergency brake
//     only — the resting image is always full quality).
//   * Manipulation preview: `previewObjectID` renders one object
//     cheap (emissive stops emitting, glass turns opaque) through
//     the material-only path, so drags stay fluid.
//
// V3 notes:
//   * Firefly clamp in the trace kernel caps each sample against
//     the pixel's own running average; a spatial cross-bilateral
//     denoise pass (second guide texture: surface normals) cleans
//     interactive and Play frames while the settled image is
//     still displayed raw from the accumulation.
//   * Texture slices now come from TextureGen.allSliceNames()
//     (presets + roads) plus the user's custom texture library;
//     `reloadTextureArraysIfNeeded()` rebuilds the arrays when
//     that library changes. Material flag bit3 = texture FIT.
//   * Weather: wet/snow levels ride the trace uniforms (up-facing
//     surfaces darken / whiten and gain/lose gloss, with a sheen
//     bounce); rain/snow precipitation is a blended overlay pass
//     on the live draw only. `lightsCut` suppresses every
//     emissive flag for stage-lighting previews.

import Foundation
import Metal
import MetalKit
import QuartzCore
import simd
import ImageIO
import UniformTypeIdentifiers

// GPU struct mirrors — must match Shaders.metal byte-for-byte.
struct GPUUniforms {
    var camOrigin = SIMD4<Float>(0, 0, 0, 0)
    var camRight = SIMD4<Float>(1, 0, 0, 0)
    var camUp = SIMD4<Float>(0, 1, 0, 0)
    var camForward = SIMD4<Float>(0, 0, -1, 0)
    var tanFovAspect = SIMD2<Float>(0.7, 1)
    var frameSeed: UInt32 = 0
    var maxBounces: UInt32 = 5
    var triCount: UInt32 = 0
    var emissiveCount: UInt32 = 0
    var totalEmissiveArea: Float = 0
    var envIntensity: Float = 1
    var historyCap: Float = 768
    var glassRefractBudget: UInt32 = 4
    var wetLevel: Float = 0
    var snowLevel: Float = 0
    var dynTime: Float = 0   // v3.2: weather clock (retained for layout; v3.3: unused by the trace kernel)
    var rainFrac: Float = 0  // v3.2: live rain intensity (retained for layout; v3.3: unused by the trace kernel)
}

struct ReprojUniforms {
    var prevOrigin = SIMD4<Float>(0, 0, 0, 0)
    var prevRight = SIMD4<Float>(1, 0, 0, 0)
    var prevUp = SIMD4<Float>(0, 1, 0, 0)
    var prevForward = SIMD4<Float>(0, 0, -1, 0)
    var tanFovAspect = SIMD2<Float>(0.7, 1)
    var reuseCap: Float = 0
    var pad: Float = 0
}

struct GPUMaterial {
    var baseColor = SIMD4<Float>(1, 1, 1, 0)
    var emissive = SIMD4<Float>(0, 0, 0, 0)      // rgb, a = intensity
    var glassParams = SIMD4<Float>(0, 0, 1.5, 0) // transmission, roughness, ior
    var tint = SIMD4<Float>(1, 1, 1, 1)          // rgb, a = textureScale
    var texIndex: UInt32 = 0
    var flags: UInt32 = 0
    var pad = SIMD2<UInt32>(0, 0)
}

struct DenoiseUniforms {
    var camOrigin = SIMD4<Float>(0, 0, 0, 0) // xyz used; w pads float3
    var strength: Float = 1
    var pad0: Float = 0
    var pad1: Float = 0
}

struct PrecipUniforms {
    var camOrigin = SIMD4<Float>(0, 0, 0, 0)
    var camRight = SIMD4<Float>(1, 0, 0, 0)
    var camUp = SIMD4<Float>(0, 1, 0, 0)
    var camForward = SIMD4<Float>(0, 0, -1, 0)
    var tanFovAspect = SIMD2<Float>(0.7, 1)
    var time: Float = 0
    var mode: UInt32 = 0
    var intensity: Float = 1 // v3.2: live intensity 0..~1.2
    var windXZ = SIMD2<Float>(1.2, 0.4) // v3.2: time-varying wind (m/s)
    var precipPad: Float = 0
}

struct TonemapParams {
    var samples: UInt32 = 0
    var exposure: Float = 1.15
    var shimmerTime: Float = 0  // v3.3: weather clock (s) — display ripple shimmer
    var shimmerRain: Float = 0  // v3.3: live rain intensity (0 unless raining)
    var wetLevel: Float = 0     // v3.3: ground wetness, for the puddle mask
    var pad: Float = 0
}

typealias CameraBasis = (origin: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>)

final class Renderer {
    /// One renderer for the whole app (building the procedural
    /// texture arrays takes a moment; never rebuild it per view).
    static let shared: Renderer? = Renderer()

    let device: MTLDevice
    let queue: MTLCommandQueue
    let rayTracingSupported: Bool
    private var tracePipeline: MTLComputePipelineState?
    private var guidePipeline: MTLComputePipelineState?
    private var reprojectPipeline: MTLComputePipelineState?
    private var denoisePipeline: MTLComputePipelineState?
    private var tonemapPipeline: MTLRenderPipelineState?
    private var precipPipeline: MTLRenderPipelineState?
    private var precipSeedBuffer: MTLBuffer?
    private var fieldPipeline: MTLComputePipelineState?
    /// Baked weather field (v3.3): level-independent, baked
    /// once at init — see bakeWeatherField().
    private var weatherFieldTex: MTLTexture?

    // Scene GPU state.
    private var accel: MTLAccelerationStructure?
    private var posBuffer: MTLBuffer?
    private var nrmBuffer: MTLBuffer?
    private var uvBuffer: MTLBuffer?
    private var colBuffer: MTLBuffer?
    private var idxBuffer: MTLBuffer?
    private var triMatBuffer: MTLBuffer?
    private var matBuffer: MTLBuffer?
    private var emisTriBuffer: MTLBuffer?
    private var emisCDFBuffer: MTLBuffer?
    private var triCount: Int = 0
    private var emissiveCount: Int = 0
    private var totalEmissiveArea: Float = 0

    // CPU copies for picking + material-only updates.
    private var cpuPositions: [Float] = []
    private var cpuIndices: [UInt32] = []
    private var cpuTriObject: [UUID] = []
    private var cpuTriMats: [UInt32] = []
    private var cachedObjectOrder: [UUID] = []

    // Texture arrays: built-in presets + roads plus the custom
    // texture library. `sliceNames` is kept in lockstep with the
    // array slices (slice i holds sliceNames[i]).
    private var texAlbedo: MTLTexture?
    private var texNormal: MTLTexture?
    private var texData: MTLTexture?
    private(set) var sliceNames: [String] = []
    /// Fingerprint (custom library ids+names) the arrays were
    /// built with; see reloadTextureArraysIfNeeded().
    private var builtTextureFingerprint: String = ""

    // Accumulation + guide textures, ping-ponged per frame. The
    // accumulation alpha channel carries the effective sample
    // count; guides carry primary-hit world positions (rgba32F)
    // and guide-normals carry the interpolated surface normals
    // (rgba16F, denoise guide). `displayTex` is the single
    // (non-ping-ponged) denoise output the tonemap can read.
    private var accumTex: [MTLTexture?] = [nil, nil]
    private var guideTex: [MTLTexture?] = [nil, nil]
    private var guideNormalTex: [MTLTexture?] = [nil, nil]
    private var displayTex: MTLTexture?
    private var curIdx: Int = 0
    private var historyValid: Bool = false
    private var prevCamera: CameraBasis?
    private var prevTanAspect = SIMD2<Float>(0.7, 1)
    private var frameSeedCounter: UInt32 = 0
    private var wasMoving: Bool = false

    private(set) var sampleCount: UInt32 = 0
    private var lastMaxSettle: UInt32 = 768
    var exposure: Float = 1.15
    var fovY: Float = 55 * .pi / 180

    /// Manipulation preview: this object (if any) renders cheap —
    /// emissive objects stop emitting, glass renders opaque.
    var previewObjectID: UUID?

    /// Stage-lighting cut: when true, NO object is flagged
    /// emissive (emissive surfaces shade with their emissive
    /// colour as base, like the per-object preview), so the
    /// emissive light index — and its CDF — comes out empty.
    var lightsCut: Bool = false

    /// Weather, set by the session every frame tick: mode
    /// 0 = dry, 1 = rain, 2 = snow; wet/snow are 0…1 build-up
    /// levels that shade up-facing surfaces in the trace kernel.
    var weatherMode: Int = 0
    var weatherWet: Float = 0
    var weatherSnow: Float = 0
    /// v3.4: QUANTISED build-up levels (steps of 1/24) that the
    /// current geometry was baked for. rebuild() turns these into
    /// real snow displacement + puddle water sheets; the session
    /// updates them only when a step boundary is crossed, so the
    /// expensive geometry path runs in short, coalesced bursts
    /// while the shading levels above stay continuous.
    var weatherGeoWet: Float = 0
    var weatherGeoSnow: Float = 0
    /// True when the last geometry rebuild emitted water sheets —
    /// the material table then carries the sheet material last,
    /// and updateMaterialsOnly must reproduce it exactly.
    private var sceneHasSheetMaterial = false
    // v3.4 dynamic resolution (motion frames only): a multiplier
    // on the tier's motion scale, driven by the delivered frame
    // cadence from command-buffer completion deltas. 1.0 is the
    // tier's nominal scale — it only ever trims WHILE MOVING, in
    // small steps with a cooldown, when frames are demonstrably
    // missing the 60 fps cadence, and it resets to 1.0 when the
    // scene settles. The settled image is always full quality.
    private var dynScale: Float = 1.0
    private var dynCadenceEMA: Double = 0
    private var dynLastCompletion: CFTimeInterval = 0
    private var dynFramesSinceAdjust = 0
    private let dynLock = NSLock()
    /// Weather dynamics (v3.2), written by the session every
    /// tick alongside the levels: a slow surge clock, the live
    /// intensity it produces, the rain-only intensity the trace
    /// kernel ripples puddles with, and the gusting wind vector
    /// the precipitation pass slants with.
    var weatherDynTime: Float = 0
    var weatherDynIntensity: Float = 1
    var weatherDynRain: Float = 0
    var weatherWind = SIMD2<Float>(1.2, 0.4)
    var precipAvailable: Bool { precipPipeline != nil }
    /// Weather levels used by the previous draw; a change softens
    /// reprojection history so the ground visibly wets/dries.
    private var lastWet: Float = 0
    private var lastSnow: Float = 0
    private var lastTracedWet: Float = -1
    private var lastTracedSnow: Float = -1

    /// 0 = Auto, 1 = High detail, 2 = Battery saver.
    var qualityPreference: Int = 0
    /// ProcessInfo thermal state raw value (0 nominal … 3 critical).
    private(set) var thermalTier: Int = 0
    private var thermalObserver: NSObjectProtocol?
    /// A frame must be (re)presented even if converged (thermal or
    /// quality change, scene rebuild, first appearance…).
    private var presentationDirty: Bool = true
    /// Called when an async event (thermal change) needs a redraw.
    var onNeedsFrame: (() -> Void)?

    /// True while another draw would change the picture: history
    /// not built, samples below the settle cap, or a presentation
    /// is owed. The viewport's frame chain runs only then.
    var frameWanted: Bool {
        if presentationDirty { return true }
        guard accel != nil else { return false }
        if weatherMode != 0 { return true } // precipitation animates
        return !historyValid || sampleCount < lastMaxSettle
    }

    var converged: Bool { sampleCount >= lastMaxSettle }
    var hasGeometry: Bool { accel != nil }

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
        self.rayTracingSupported = device.supportsRaytracing
        self.thermalTier = ProcessInfo.processInfo.thermalState.rawValue
        guard let library = device.makeDefaultLibrary() else { return nil }
        if let fn = library.makeFunction(name: "pathTraceKernel") {
            tracePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "guideKernel") {
            guidePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "reprojectKernel") {
            reprojectPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "denoiseKernel") {
            denoisePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let fn = library.makeFunction(name: "weatherFieldKernel") {
            fieldPipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let vfn = library.makeFunction(name: "tonemapVertex"),
           let ffn = library.makeFunction(name: "tonemapFragment") {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfn
            desc.fragmentFunction = ffn
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            tonemapPipeline = try? device.makeRenderPipelineState(descriptor: desc)
        }
        if let vfn = library.makeFunction(name: "precipVertex"),
           let ffn = library.makeFunction(name: "precipFragment") {
            // Blended overlay, no depth — drawn over the
            // tonemapped drawable with a .load pass.
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfn
            desc.fragmentFunction = ffn
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            desc.colorAttachments[0].isBlendingEnabled = true
            desc.colorAttachments[0].rgbBlendOperation = .add
            desc.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
            desc.colorAttachments[0].alphaBlendOperation = .add
            desc.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
            desc.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
            precipPipeline = try? device.makeRenderPipelineState(descriptor: desc)
        }
        // Precipitation seeds: 2048 x SIMD4<Float>, each component
        // in [0,1), from a deterministic LCG so frames are stable.
        do {
            var seeds: [SIMD4<Float>] = []
            seeds.reserveCapacity(2048)
            var state: UInt32 = 12345
            for _ in 0..<2048 {
                state = state &* 1664525 &+ 1013904223
                let x = Float(state >> 8) * (1.0 / 16777216.0)
                state = state &* 1664525 &+ 1013904223
                let y = Float(state >> 8) * (1.0 / 16777216.0)
                state = state &* 1664525 &+ 1013904223
                let z = Float(state >> 8) * (1.0 / 16777216.0)
                state = state &* 1664525 &+ 1013904223
                let w = Float(state >> 8) * (1.0 / 16777216.0)
                seeds.append(SIMD4<Float>(x, y, z, w))
            }
            precipSeedBuffer = seeds.withUnsafeBytes { raw in
                guard let base = raw.baseAddress else { return nil }
                return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
            }
        }
        thermalObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil, queue: .main) { [weak self] _ in
                guard let self else { return }
                let tier = ProcessInfo.processInfo.thermalState.rawValue
                if tier != self.thermalTier {
                    self.thermalTier = tier
                    self.presentationDirty = true
                    self.onNeedsFrame?()
                }
            }
        buildTextureArrays()
        bakeWeatherField()
    }

    /// One-shot bake of the level-independent weather field
    /// (v3.3): 2048x2048 RGBA16F over a 64 m world window —
    /// R puddle metaball potential, G wet pooling, B snow
    /// drift, A broad pooling. Wet/snow levels are thresholds
    /// applied analytically in the trace kernel, so the bake
    /// never needs refreshing when build-up changes. If the
    /// bake is unavailable, a 1x1 neutral texture stands in
    /// (global wet/snow shading still works; no puddles).
    private func bakeWeatherField() {
        let size = 2048
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: size, height: size, mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        if let fieldPipeline, let tex = device.makeTexture(descriptor: desc),
           let cmd = queue.makeCommandBuffer(),
           let enc = cmd.makeComputeCommandEncoder() {
            enc.setComputePipelineState(fieldPipeline)
            enc.setTexture(tex, index: 0)
            dispatch(enc, pipeline: fieldPipeline, width: size, height: size)
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()
            weatherFieldTex = tex
            return
        }
        let ndesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        ndesc.usage = [.shaderRead]
        ndesc.storageMode = .shared
        if let tex = device.makeTexture(descriptor: ndesc) {
            var px: [Float16] = [0, 0.5, 0.5, 0.489]
            tex.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                        withBytes: &px, bytesPerRow: 8)
            weatherFieldTex = tex
        }
    }

    deinit {
        if let thermalObserver {
            NotificationCenter.default.removeObserver(thermalObserver)
        }
    }

    // MARK: Texture arrays

    /// Fingerprint of the custom texture library (ids + names):
    /// any library edit changes it, which is the rebuild signal.
    private func textureLibraryFingerprint() -> String {
        CustomTextureStore.list()
            .map { "\($0.id.uuidString):\($0.name)" }
            .joined(separator: "|")
    }

    private func buildTextureArrays() {
        let customMetas = CustomTextureStore.list()
        builtTextureFingerprint = customMetas
            .map { "\($0.id.uuidString):\($0.name)" }
            .joined(separator: "|")
        let candidateNames = TextureGen.allSliceNames()
            + customMetas.map { "custom:\($0.id.uuidString)" }
        var builtNames: [String] = []
        var albedos: [Data] = [], normals: [Data] = [], datas: [Data] = []
        var size = 0
        for name in candidateNames {
            let set: PBRTextureSet?
            if name.hasPrefix("custom:") {
                let idString = String(name.dropFirst("custom:".count))
                guard let id = UUID(uuidString: idString),
                      let loaded = CustomTextureStore.loadSet(id: id) else { continue }
                set = loaded
            } else {
                set = TextureGen.makeSet(preset: name)
            }
            guard let set else { continue }
            // All slices of one array share a size; a set that
            // doesn't match the established size is skipped (its
            // name too — names and slices stay in lockstep).
            if size == 0 {
                size = set.size
            } else if set.size != size {
                continue
            }
            builtNames.append(name)
            albedos.append(set.albedo); normals.append(set.normal); datas.append(set.data)
        }
        sliceNames = builtNames
        guard !builtNames.isEmpty, size > 0 else {
            texAlbedo = nil; texNormal = nil; texData = nil
            return
        }
        func makeArray(_ slices: [Data]) -> MTLTexture? {
            let desc = MTLTextureDescriptor()
            desc.textureType = .type2DArray
            desc.pixelFormat = .rgba8Unorm
            desc.width = size; desc.height = size
            desc.arrayLength = slices.count
            desc.usage = [.shaderRead]
            desc.storageMode = .shared
            guard let tex = device.makeTexture(descriptor: desc) else { return nil }
            for (i, data) in slices.enumerated() {
                data.withUnsafeBytes { raw in
                    if let base = raw.baseAddress {
                        tex.replace(region: MTLRegionMake2D(0, 0, size, size),
                                    mipmapLevel: 0, slice: i,
                                    withBytes: base,
                                    bytesPerRow: size * 4,
                                    bytesPerImage: size * size * 4)
                    }
                }
            }
            return tex
        }
        texAlbedo = makeArray(albedos)
        texNormal = makeArray(normals)
        texData = makeArray(datas)
    }

    /// Rebuilds the texture arrays when the custom texture
    /// library changed since they were built. Returns true when
    /// a rebuild happened (accumulation is reset — slice indices
    /// may have moved).
    func reloadTextureArraysIfNeeded() -> Bool {
        guard textureLibraryFingerprint() != builtTextureFingerprint else { return false }
        buildTextureArrays()
        resetAccumulation()
        return true
    }

    /// Texture array slice for a texture name (0 = none in
    /// materials). Resolves built-in presets, roads, and
    /// "custom:<uuid>" names against this renderer's slice list.
    func texIndexValue(forPreset preset: String) -> UInt32 {
        guard let i = sliceNames.firstIndex(of: preset) else { return 0 }
        return UInt32(i + 1)
    }

    /// Texture array slice for a preset name (0 = none in
    /// materials), resolved via the shared renderer's slices.
    static func texIndex(forPreset preset: String) -> UInt32 {
        guard let shared = Renderer.shared else { return 0 }
        return shared.texIndexValue(forPreset: preset)
    }

    // MARK: Scene building

    func rebuild(scene: Scene, meshes: [UUID: MeshData]) {
        rebuild(scene: scene, meshes: meshes, geometry: true)
    }

    /// Full geometry rebuild, or — when `geometry` is false — a
    /// material-only update if the object set is unchanged.
    func rebuild(scene: Scene, meshes: [UUID: MeshData], geometry: Bool) {
        if !geometry, updateMaterialsOnly(scene: scene, meshes: meshes) {
            return
        }
        var positions: [Float] = []
        var normals: [Float] = []
        var uvArr: [Float] = []
        var colArr: [Float] = []
        var indices: [UInt32] = []
        var triMats: [UInt32] = []
        var materials: [GPUMaterial] = []
        var triObjects: [UUID] = []
        var objectOrder: [UUID] = []

        // Gather the included objects first: the puddle sheet
        // material (when any sheet is emitted) must sit at a
        // deterministic index — right after the last object
        // material — so updateMaterialsOnly can reproduce the
        // exact same material table without the geometry.
        var included: [(obj: SceneObject, world: MeshData)] = []
        for obj in scene.objects where obj.visible {
            let local: MeshData
            if obj.source == "primitive" {
                local = MeshGen.primitive(kind: obj.primitiveKind, params: obj.params)
            } else if let meshFile = obj.meshFile, let m = meshes[obj.id] {
                _ = meshFile
                local = m
            } else {
                continue
            }
            if local.isEmpty { continue }
            included.append((obj, local.baked(by: obj.transform.matrix)))
        }
        let sheetMatIndex = UInt32(included.count)
        let weatherGeoActive = weatherGeoWet > 0.001 || weatherGeoSnow > 0.001
        var triBudget = 220_000
        var sheetEmitted = false
        for (obj, world) in included {
            let matIndex = UInt32(materials.count)
            materials.append(gpuMaterial(for: obj))
            objectOrder.append(obj.id)
            if weatherGeoActive {
                appendWeatherGeometry(world: world, objectID: obj.id,
                                      matIndex: matIndex, sheetIndex: sheetMatIndex,
                                      positions: &positions, normals: &normals,
                                      uvs: &uvArr, colors: &colArr,
                                      indices: &indices, triMats: &triMats,
                                      triObjects: &triObjects,
                                      budget: &triBudget, sheetEmitted: &sheetEmitted)
            } else {
                let vBase = UInt32(positions.count / 3)
                positions += world.positions
                normals += world.normals
                uvArr += world.uvs
                colArr += world.colors
                for t in 0..<world.triangleCount {
                    let a = world.indices[3 * t], b = world.indices[3 * t + 1], c = world.indices[3 * t + 2]
                    indices += [vBase + a, vBase + b, vBase + c]
                    triMats.append(matIndex)
                    triObjects.append(obj.id)
                }
            }
        }
        if sheetEmitted {
            materials.append(sheetMaterial())
        }
        sceneHasSheetMaterial = sheetEmitted

        triCount = indices.count / 3
        cpuPositions = positions
        cpuIndices = indices
        cpuTriObject = triObjects
        cpuTriMats = triMats
        cachedObjectOrder = objectOrder

        uploadEmissiveIndex(materials: materials)
        matBuffer = makeBuffer(materials)

        posBuffer = makeBuffer(positions)
        nrmBuffer = makeBuffer(normals)
        uvBuffer = makeBuffer(uvArr)
        colBuffer = makeBuffer(colArr)
        idxBuffer = makeBuffer(indices)
        triMatBuffer = makeBuffer(triMats)

        buildAccelerationStructure()
        resetAccumulation()
    }

    /// Material-only refresh: same objects, same geometry — rewrite
    /// just the material buffer and the emissive light index (CDF)
    /// from the cached CPU triangles. Returns false when the object
    /// set changed and a full rebuild is required.
    @discardableResult
    func updateMaterialsOnly(scene: Scene, meshes: [UUID: MeshData]) -> Bool {
        guard accel != nil, !cpuIndices.isEmpty else { return false }
        // The included-object list must match the cached geometry
        // exactly (same inclusion rule as the full rebuild).
        var candidate: [UUID] = []
        for obj in scene.objects where obj.visible {
            if obj.source == "primitive" {
                candidate.append(obj.id)
            } else if let m = meshes[obj.id], !m.isEmpty {
                candidate.append(obj.id)
            }
        }
        guard candidate == cachedObjectOrder else { return false }
        var materials: [GPUMaterial] = []
        materials.reserveCapacity(candidate.count)
        for id in candidate {
            guard let obj = scene.object(id: id) else { return false }
            materials.append(gpuMaterial(for: obj))
        }
        // v3.4: reproduce the geometry rebuild's material table
        // exactly — the puddle sheet material rides last.
        if sceneHasSheetMaterial { materials.append(sheetMaterial()) }
        uploadEmissiveIndex(materials: materials)
        matBuffer = makeBuffer(materials)
        resetAccumulation()
        return true
    }

    /// Builds the emissive triangle list + area CDF into GPU
    /// buffers from the cached CPU geometry and a material table.
    private func uploadEmissiveIndex(materials: [GPUMaterial]) {
        var emisTris: [UInt32] = []
        var emisAreas: [Float] = []
        if triCount > 0 {
            for t in 0..<triCount {
                let m = materials[Int(cpuTriMats[t])]
                guard (m.flags & 1) != 0 else { continue }
                let a = cpuIndices[3 * t], b = cpuIndices[3 * t + 1], c = cpuIndices[3 * t + 2]
                let p0 = SIMD3<Float>(cpuPositions[3 * Int(a)], cpuPositions[3 * Int(a) + 1], cpuPositions[3 * Int(a) + 2])
                let p1 = SIMD3<Float>(cpuPositions[3 * Int(b)], cpuPositions[3 * Int(b) + 1], cpuPositions[3 * Int(b) + 2])
                let p2 = SIMD3<Float>(cpuPositions[3 * Int(c)], cpuPositions[3 * Int(c) + 1], cpuPositions[3 * Int(c) + 2])
                let area = 0.5 * simd_length(simd_cross(p1 - p0, p2 - p0))
                if area > 1e-9 {
                    emisTris.append(UInt32(t))
                    emisAreas.append(area)
                }
            }
        }
        emissiveCount = emisTris.count
        var cdf: [Float] = []
        var acc: Float = 0
        for a in emisAreas { acc += a; cdf.append(acc) }
        totalEmissiveArea = acc
        emisTriBuffer = makeBuffer(emisTris)
        emisCDFBuffer = makeBuffer(cdf)
    }

    private func makeBuffer<T>(_ array: [T]) -> MTLBuffer? {
        if array.isEmpty { return device.makeBuffer(length: 4, options: .storageModeShared) }
        return array.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }
    }

    private func gpuMaterial(for obj: SceneObject) -> GPUMaterial {
        let m = obj.material
        var g = GPUMaterial()
        var base = m.baseColor.simd
        var flags: UInt32 = 0
        let previewing = (obj.id == previewObjectID)
        if m.emissiveOn {
            if previewing || lightsCut {
                // Manipulation preview / lights cut: no light cast
                // (no emissive flag — the light index keys off it,
                // so lightsCut yields an empty CDF); the surface
                // takes the emissive colour so it still reads as
                // the same object.
                base = m.emissiveColor.simd
            } else {
                flags |= 1
            }
        }
        if m.glassOn {
            if previewing {
                // Manipulation preview: opaque, tinted toward the
                // glass colour.
                base = base + (m.tint.simd - base) * 0.4
            } else {
                flags |= 2
            }
        }
        if m.triplanar { flags |= 4 }
        if m.textureFit { flags |= 8 }
        g.baseColor = SIMD4<Float>(base, 0)
        g.emissive = SIMD4<Float>(m.emissiveColor.simd, Float(m.emissiveIntensity))
        g.glassParams = SIMD4<Float>(Float(m.transmission), Float(m.roughness), Float(m.ior), 0)
        g.tint = SIMD4<Float>(m.tint.simd, Float(m.textureScale))
        g.texIndex = texIndexValue(forPreset: m.texturePreset)
        g.flags = flags
        return g
    }

    /// v3.4: the puddle water material — one shared entry appended
    /// after all object materials whenever the geometry contains
    /// water sheets. FLAG_SHEET (16) routes it to the water
    /// shading branch of the trace kernel; the dark blue-grey is
    /// the water body colour seen where the mirror reflection is
    /// weak (steep view angles).
    private func sheetMaterial() -> GPUMaterial {
        var g = GPUMaterial()
        g.baseColor = SIMD4<Float>(0.030, 0.045, 0.060, 0)
        g.emissive = SIMD4<Float>(0, 0, 0, 0)
        g.glassParams = SIMD4<Float>(0, 0.05, 1.33, 0)
        g.tint = SIMD4<Float>(1, 1, 1, 1)
        g.texIndex = 0
        g.flags = 16
        return g
    }

    // MARK: - Weather geometry (v3.4: snow and puddles become solid)
    //
    // Through v3.3, build-up weather was shading only: snow
    // whitened albedo, puddles were a gloss mask painted on the
    // floor. In a path tracer that has no silhouette, no
    // parallax and no occlusion — obvious at grazing angles
    // (device verdict 2026-10-30: "Snow isn't 3D — it's just
    // whiter and whiter, and neither are puddles"). This pass
    // makes both REAL, inside rebuild(), from the same fields
    // the shader shades with:
    //
    //   * SNOW: up-facing triangles are subdivided (target edge
    //     ~0.22 m) and their vertices displaced upward by the
    //     drift-shaped snow depth, up to ~0.30 m at full cover.
    //     A barycentric edge fade pins every original triangle
    //     boundary to zero displacement, so displaced patches
    //     meet undisplaced neighbours with no cracks, and objects
    //     standing on snowy ground read as sitting IN the snow.
    //     Vertex normals come from the heightfield gradient, so
    //     drifts shade like solids — and since the displaced
    //     triangles live in the acceleration structure, they
    //     occlude, reflect and cast real shadows.
    //   * PUDDLES: where the mirrored puddle mask is strong at a
    //     subdivided triangle's centre, a water SHEET triangle
    //     is emitted above the surface — up to ~13 mm at the
    //     pool centre by depth, rim vertices tucked 2.5 mm
    //     BELOW the surface, so the edge is a tiny sloped wall
    //     that catches light at grazing angles instead of a
    //     painted outline. Sheets carry the host object's id
    //     (tapping a puddle selects its floor) and the shared
    //     FLAG_SHEET material.
    //
    // The field functions below are exact CPU mirrors of the
    // shader's baked-field formulas (same hashes, same
    // constants), so geometry and shading agree on where the
    // drifts and pools are. Levels arrive quantised to steps of
    // 1/24 (the session rebuilds only when a step is crossed),
    // and a scene-wide budget caps added triangles; past the
    // budget a surface keeps the v3.3 shading-only look rather
    // than blowing the frame budget. Per-frame cost of all of
    // this is zero: the result is ordinary static geometry in
    // the same single acceleration structure.

    private func wgfHash(_ x0: UInt32) -> UInt32 {
        var x = x0
        x ^= x >> 16; x = x &* 0x7feb352d
        x ^= x >> 15; x = x &* 0x846ca68b
        x ^= x >> 16
        return x
    }

    private func wgfLattice(_ ix: Int32, _ iy: Int32) -> Float {
        let h = wgfHash((UInt32(bitPattern: ix) &* 0x8da6b343)
                        ^ (UInt32(bitPattern: iy) &* 0xd8163841))
        return Float(h >> 8) * (1.0 / 16777216.0)
    }

    private func wgfNoise(_ px: Float, _ py: Float) -> Float {
        let flx = px.rounded(.down), fly = py.rounded(.down)
        let ix = Int32(flx), iy = Int32(fly)
        let fx = px - flx, fy = py - fly
        let sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy)
        let v00 = wgfLattice(ix, iy), v10 = wgfLattice(ix + 1, iy)
        let v01 = wgfLattice(ix, iy + 1), v11 = wgfLattice(ix + 1, iy + 1)
        let lo = v00 + (v10 - v00) * sx
        let hi = v01 + (v11 - v01) * sx
        return lo + (hi - lo) * sy
    }

    private func wgfFbm(_ px: Float, _ py: Float) -> Float {
        0.55 * wgfNoise(px, py)
            + 0.30 * wgfNoise(px * 2, py * 2)
            + 0.15 * wgfNoise(px * 4, py * 4)
    }

    private func wgfSmooth(_ e0: Float, _ e1: Float, _ v: Float) -> Float {
        let t = min(1, max(0, (v - e0) / (e1 - e0)))
        return t * t * (3 - 2 * t)
    }

    private func wgfClamp(_ v: Float, _ lo: Float, _ hi: Float) -> Float {
        min(hi, max(lo, v))
    }

    /// Window fade of the baked field (1 inside ±29 m, 0 at ±32 m).
    private func wgfFade(_ x: Float, _ z: Float) -> Float {
        (1 - wgfSmooth(29, 32, abs(x))) * (1 - wgfSmooth(29, 32, abs(z)))
    }

    /// Drift field (baked texture B channel), window-faded.
    private func wgfDrift(_ x: Float, _ z: Float) -> Float {
        if abs(x) > 40 || abs(z) > 40 { return 0.5 }
        let raw = wgfFbm(x * 0.5 + 9.17, z * 0.5 + 27.3)
        return 0.5 + (raw - 0.5) * wgfFade(x, z)
    }

    /// Broad pooling field (baked texture A channel), faded.
    private func wgfBroad(_ x: Float, _ z: Float) -> Float {
        if abs(x) > 40 || abs(z) > 40 { return 0.489 }
        let raw = wgfFbm(x * 0.32 + 17.31, z * 0.32 + 3.97)
        return 0.489 + (raw - 0.489) * wgfFade(x, z)
    }

    /// Metaball puddle potential (baked texture R channel), faded.
    private func wgfPotential(_ x: Float, _ z: Float) -> Float {
        if abs(x) > 40 || abs(z) > 40 { return 0 }
        let px = x / 1.15, py = z / 1.15
        let wx = wgfFbm(px * 0.16 + 3.1, py * 0.16 + 7.7)
        let wy = wgfFbm(px * 0.16 + 11.9, py * 0.16 + 1.3)
        let qx = px + (wx - 0.5), qy = py + (wy - 0.5)
        let fqx = qx.rounded(.down), fqy = qy.rounded(.down)
        let ix = Int32(fqx), iy = Int32(fqy)
        let fx = qx - fqx, fy = qy - fqy
        var sum: Float = 0
        for j in -1...1 {
            for i in -1...1 {
                let cx = ix + Int32(i), cy = iy + Int32(j)
                let h = wgfHash((UInt32(bitPattern: cx) &* 0x8da6b343)
                                ^ (UInt32(bitPattern: cy) &* 0xd8163841))
                let r0 = Float(h >> 8) * (1.0 / 16777216.0)
                let h2 = wgfHash(h ^ 0x9e3779b9)
                let r1 = Float(h2 >> 8) * (1.0 / 16777216.0)
                let h3 = wgfHash(h2 ^ 0x85ebca6b)
                let cr = Float(h3 >> 8) * (1.0 / 16777216.0)
                let a = powf(wgfClamp((0.58 - cr) / 0.58, 0, 1), 0.7)
                if a <= 0.001 { continue }
                let ddx = (Float(i) + r0) - fx
                let ddy = (Float(j) + r1) - fy
                let d = sqrtf(ddx * ddx + ddy * ddy)
                let s = wgfClamp(1 - (d / 1.45) * (d / 1.45), 0, 1)
                sum += a * s * s
            }
        }
        return sum * wgfFade(x, z)
    }

    /// Snow depth (metres) the geometry carries at (x, z) for a
    /// quantised level: drift-shaped, and zero exactly where the
    /// shader's snow coverage is zero, so the silhouette and the
    /// whitening agree.
    private func wgfSnowHeight(_ x: Float, _ z: Float, level: Float) -> Float {
        let d = level * (0.3 + 1.4 * wgfDrift(x, z))
        let cover = wgfSmooth(0.45, 0.75, d)
        return 0.30 * cover * min(d / 1.35, 1)
    }

    private func wgfSnowCover(_ x: Float, _ z: Float, level: Float) -> Float {
        wgfSmooth(0.45, 0.75, level * (0.3 + 1.4 * wgfDrift(x, z)))
    }

    /// The trace kernel's puddle mask, mirrored: (mask, depth01)
    /// where depth01 says how deep into the pool the point sits.
    private func wgfPuddle(_ x: Float, _ z: Float, y: Float, geoNY: Float,
                           level: Float) -> (mask: Float, depth: Float) {
        let upFace = wgfSmooth(0.5, 0.85, geoNY)
        if upFace <= 0 || level <= 0.001 { return (0, 0) }
        let elevBias = wgfClamp(0.10 - y * 0.55, -0.22, 0.38)
        let slopeShed = wgfClamp((geoNY - 0.70) / 0.30, 0, 1)
        let thr = (1.15 + (0.08 - 1.15) * powf(level, 1.15)) - elevBias * 0.45
        let pot = wgfPotential(x, z)
        var m = wgfSmooth(thr - 0.14, thr + 0.14, pot)
            * (0.78 + 0.45 * wgfBroad(x, z))
        m = wgfClamp(m, 0, 1) * slopeShed * upFace
            * wgfFade(x, z) * wgfSmooth(0.03, 0.15, level)
        return (m, wgfClamp((pot - thr) / 0.5, 0, 1))
    }

    /// Append one object's world mesh with weather geometry
    /// applied (see the section comment above).
    private func appendWeatherGeometry(world: MeshData, objectID: UUID,
                                       matIndex: UInt32, sheetIndex: UInt32,
                                       positions: inout [Float], normals: inout [Float],
                                       uvs: inout [Float], colors: inout [Float],
                                       indices: inout [UInt32], triMats: inout [UInt32],
                                       triObjects: inout [UUID],
                                       budget: inout Int, sheetEmitted: inout Bool) {
        let wet = weatherGeoWet
        let snow = weatherGeoSnow

        func wPos(_ idx: UInt32) -> SIMD3<Float> {
            let i = Int(idx) * 3
            return SIMD3<Float>(world.positions[i], world.positions[i + 1], world.positions[i + 2])
        }
        func wNrm(_ idx: UInt32) -> SIMD3<Float> {
            let i = Int(idx) * 3
            return SIMD3<Float>(world.normals[i], world.normals[i + 1], world.normals[i + 2])
        }
        func wCol(_ idx: UInt32) -> SIMD3<Float> {
            let i = Int(idx) * 3
            return SIMD3<Float>(world.colors[i], world.colors[i + 1], world.colors[i + 2])
        }
        func wUV(_ idx: UInt32) -> SIMD2<Float> {
            let i = Int(idx) * 2
            return SIMD2<Float>(world.uvs[i], world.uvs[i + 1])
        }
        // One undisplaced triangle with its source attributes —
        // the v3.3 shading-only look.
        func emitFlat(_ a: UInt32, _ b: UInt32, _ c: UInt32) {
            let base = UInt32(positions.count / 3)
            for v in [a, b, c] {
                let p = wPos(v), n = wNrm(v), t = wUV(v), cl = wCol(v)
                positions.append(contentsOf: [p.x, p.y, p.z])
                normals.append(contentsOf: [n.x, n.y, n.z])
                uvs.append(contentsOf: [t.x, t.y])
                colors.append(contentsOf: [cl.x, cl.y, cl.z])
            }
            indices.append(contentsOf: [base, base + 1, base + 2])
            triMats.append(matIndex)
            triObjects.append(objectID)
        }

        for t in 0..<world.triangleCount {
            let ia = world.indices[3 * t]
            let ib = world.indices[3 * t + 1]
            let ic = world.indices[3 * t + 2]
            let p0 = wPos(ia), p1 = wPos(ib), p2 = wPos(ic)
            let cr = simd_cross(p1 - p0, p2 - p0)
            let cl = simd_length(cr)
            if cl < 1e-12 { emitFlat(ia, ib, ic); continue }
            let geoN = cr / cl
            let longest = max(simd_length(p1 - p0),
                              max(simd_length(p2 - p1), simd_length(p0 - p2)))
            if geoN.y <= 0.62 || budget <= 0 { emitFlat(ia, ib, ic); continue }
            // Cheap reject: a small triangle whose corners and
            // centre are all snow-free and pool-free stays flat.
            // Displaced neighbours pin their shared border to
            // zero height, so this cannot open a crack.
            if longest <= 2.0 {
                var maxH: Float = 0
                var maxM: Float = 0
                for q in [p0, p1, p2, (p0 + p1 + p2) / 3] {
                    if snow > 0.001 {
                        maxH = max(maxH, wgfSnowHeight(q.x, q.z, level: snow))
                    }
                    if wet > 0.001 {
                        maxM = max(maxM, wgfPuddle(q.x, q.z, y: q.y,
                                                   geoNY: geoN.y, level: wet).mask)
                    }
                }
                if maxH < 0.001 && maxM < 0.08 { emitFlat(ia, ib, ic); continue }
            }
            var n = Int((longest / 0.22).rounded(.up))
            n = max(1, min(n, 56))
            while n > 1 && n * n > budget { n -= 1 }
            if n < 2 { emitFlat(ia, ib, ic); continue }
            budget -= n * n

            // Triangular lattice: vertex (i, j) carries
            // barycentric weights (1-(i+j)/n, i/n, j/n) over
            // (p0, p1, p2), indexed row-major by gIdx.
            let nf = Float(n)
            var gPos: [SIMD3<Float>] = []
            var gUV: [SIMD2<Float>] = []
            var gMask: [Float] = []
            var gDepth: [Float] = []
            var gSnow: [Float] = []
            gPos.reserveCapacity((n + 1) * (n + 2) / 2)
            let vBase = UInt32(positions.count / 3)
            for i in 0...n {
                for j in 0...(n - i) {
                    let wb = Float(i) / nf, wc = Float(j) / nf
                    let wa = 1 - wb - wc
                    let pos = p0 * wa + p1 * wb + p2 * wc
                    var nrm = wNrm(ia) * wa + wNrm(ib) * wb + wNrm(ic) * wc
                    let nl = simd_length(nrm)
                    if nl > 1e-6 { nrm /= nl }
                    var disp = pos
                    var snowC: Float = 0
                    if snow > 0.001 {
                        snowC = wgfSnowCover(pos.x, pos.z, level: snow)
                        let edgeFade = wgfSmooth(0, 0.18, min(wa, min(wb, wc)))
                        let h = wgfSnowHeight(pos.x, pos.z, level: snow) * edgeFade
                        if h > 0.0005 {
                            disp.y += h
                            let e: Float = 0.05
                            let gx = (wgfSnowHeight(pos.x + e, pos.z, level: snow)
                                      - wgfSnowHeight(pos.x - e, pos.z, level: snow)) / (2 * e)
                            let gz = (wgfSnowHeight(pos.x, pos.z + e, level: snow)
                                      - wgfSnowHeight(pos.x, pos.z - e, level: snow)) / (2 * e)
                            let sn = simd_normalize(SIMD3<Float>(-gx, 1, -gz))
                            let blend = min(h / 0.03, 1)
                            nrm = simd_normalize(nrm * (1 - blend) + sn * blend)
                        }
                    }
                    gSnow.append(snowC)
                    var mask: Float = 0, depth01: Float = 0
                    if wet > 0.001 {
                        let pm = wgfPuddle(pos.x, pos.z, y: pos.y,
                                           geoNY: geoN.y, level: wet)
                        mask = pm.mask; depth01 = pm.depth
                    }
                    gMask.append(mask); gDepth.append(depth01)
                    gPos.append(disp)
                    let tuv = wUV(ia) * wa + wUV(ib) * wb + wUV(ic) * wc
                    gUV.append(tuv)
                    let col = wCol(ia) * wa + wCol(ib) * wb + wCol(ic) * wc
                    positions.append(contentsOf: [disp.x, disp.y, disp.z])
                    normals.append(contentsOf: [nrm.x, nrm.y, nrm.z])
                    uvs.append(contentsOf: [tuv.x, tuv.y])
                    colors.append(contentsOf: [col.x, col.y, col.z])
                }
            }
            func gIdx(_ i: Int, _ j: Int) -> Int {
                i * (n + 1) - i * (i - 1) / 2 + j
            }
            // One water sheet triangle over a subdivided
            // triangle: centre vertices ride up to ~13 mm above
            // the surface by pool depth; rim vertices tuck
            // 2.5 mm below it, so the edge is a tiny sloped
            // wall — a real rim, not a painted outline.
            func emitSheet(_ a: Int, _ b: Int, _ c: Int) {
                var order = [a, b, c]
                if simd_cross(gPos[b] - gPos[a], gPos[c] - gPos[a]).y < 0 {
                    order = [a, c, b]
                }
                let base = UInt32(positions.count / 3)
                for k in order {
                    let gp = gPos[k]
                    let off: Float = gMask[k] < 0.45 ? -0.0025
                        : 0.0035 + 0.0095 * gDepth[k]
                    positions.append(contentsOf: [gp.x, gp.y + off, gp.z])
                    normals.append(contentsOf: [0, 1, 0])
                    uvs.append(contentsOf: [gUV[k].x, gUV[k].y])
                    colors.append(contentsOf: [1, 1, 1])
                }
                indices.append(contentsOf: [base, base + 1, base + 2])
                triMats.append(sheetIndex)
                triObjects.append(objectID)
                sheetEmitted = true
            }
            func emitSub(_ a: Int, _ b: Int, _ c: Int) {
                var bb = b, cc = c
                // Winding repair against the host face:
                // displacement must never flip a subtriangle
                // inside-out.
                if simd_dot(simd_cross(gPos[bb] - gPos[a],
                                       gPos[cc] - gPos[a]), geoN) < 0 {
                    swap(&bb, &cc)
                }
                indices.append(contentsOf: [vBase + UInt32(a),
                                            vBase + UInt32(bb),
                                            vBase + UInt32(cc)])
                triMats.append(matIndex)
                triObjects.append(objectID)
                if wet > 0.001 {
                    let mC = (gMask[a] + gMask[bb] + gMask[cc]) / 3
                    let sC = (gSnow[a] + gSnow[bb] + gSnow[cc]) / 3
                    if mC * (1 - sC) > 0.5 { emitSheet(a, bb, cc) }
                }
            }
            for i in 0..<n {
                for j in 0..<(n - i) {
                    emitSub(gIdx(i, j), gIdx(i + 1, j), gIdx(i, j + 1))
                }
                for j in 0..<(n - 1 - i) {
                    emitSub(gIdx(i + 1, j + 1), gIdx(i, j + 1), gIdx(i + 1, j))
                }
            }
        }
    }

    private func buildAccelerationStructure() {
        accel = nil
        guard rayTracingSupported, triCount > 0,
              let posBuffer, let idxBuffer else { return }
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = posBuffer
        geom.vertexStride = 12
        geom.vertexFormat = .float3
        geom.indexBuffer = idxBuffer
        geom.indexType = .uint32
        geom.triangleCount = triCount
        geom.opaque = true
        let desc = MTLPrimitiveAccelerationStructureDescriptor()
        desc.geometryDescriptors = [geom]
        let sizes = device.accelerationStructureSizes(descriptor: desc)
        guard let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize, options: .storageModePrivate),
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeAccelerationStructureCommandEncoder() else { return }
        enc.build(accelerationStructure: structure, descriptor: desc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        accel = structure
    }

    // MARK: Accumulation state

    /// Hard reset: the next frame starts from an empty accumulation
    /// (scene content changed, quality changed, …). Camera moves do
    /// NOT call this — they reproject (see draw).
    func resetAccumulation() {
        sampleCount = 0
        historyValid = false
        presentationDirty = true
    }

    private func makeAccumTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: max(width, 1), height: max(height, 1),
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    private func makeGuideTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: max(width, 1), height: max(height, 1),
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    /// Guide-normal texture (v3): interpolated surface normals
    /// for the denoise pass — half precision is plenty for a
    /// bilateral weight.
    private func makeGuideNormalTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: max(width, 1), height: max(height, 1),
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    /// Zero the accumulation texture (alpha = 0: no samples yet).
    /// The trace kernel reads the previous contents before adding
    /// its sample, so a brand-new texture MUST be cleared first.
    private func clearAccum(_ accum: MTLTexture, cmd: MTLCommandBuffer) {
        let clearDesc = MTLRenderPassDescriptor()
        clearDesc.colorAttachments[0].texture = accum
        clearDesc.colorAttachments[0].loadAction = .clear
        clearDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        clearDesc.colorAttachments[0].storeAction = .store
        if let cenc = cmd.makeRenderCommandEncoder(descriptor: clearDesc) {
            cenc.endEncoding()
        }
    }

    // MARK: Adaptive quality

    private struct FrameTargets {
        var scale: Float
        var spp: Int
        var bounces: UInt32
        var glassBudget: UInt32
        var maxSettle: UInt32
        /// History weight the reprojection may carry across a
        /// camera move (at rest: everything — reprojection is the
        /// identity and accumulation simply continues).
        var reuseCap: Float
    }

    private func targets(moving: Bool, play: Bool) -> FrameTargets {
        var scale: Float = 1.0 // resting: full resolution
        var spp = 2
        var bounces: UInt32 = 5
        var glass: UInt32 = 4
        var maxSettle: UInt32 = 768
        switch qualityPreference {
        case 1: // High detail
            spp = 3; bounces = 6; glass = 6; maxSettle = 1536
        case 2: // Battery saver (user-chosen)
            // v3.4: glass floor raised (was 3) — see the motion
            // cap below; the mode's savings live in resolution,
            // spp and the settle cap, not in starving glass.
            scale = 0.8; spp = 1; bounces = 4; glass = 4; maxSettle = 256
        default:
            break
        }
        if play {
            maxSettle = qualityPreference == 1 ? 192 : (qualityPreference == 2 ? 64 : 128)
        }
        var reuseCap = Float(maxSettle)
        if moving {
            scale = min(scale, qualityPreference == 2 ? 0.55 : 0.65)
            spp = qualityPreference == 2 ? 3 : 4
            bounces = min(bounces, 4)
            // v3.4: Battery Saver keeps a higher glass floor in
            // motion (3 refractions, not 2) — the old cap hit
            // glass hardest and it went visibly fuzzy there,
            // while the mode still saves real power through
            // resolution (0.55), spp and settle caps. Other
            // modes keep the v2 motion budget of 2.
            glass = min(glass, qualityPreference == 2 ? 3 : 2)
            reuseCap = play ? 16 : 24
        }
        // Thermal state is an emergency brake only.
        switch thermalTier {
        case 1:
            scale *= 0.9
            maxSettle = min(maxSettle, 512)
        case 2:
            scale *= 0.75
            maxSettle = min(maxSettle, 192)
            bounces = min(bounces, 4)
            if !moving { spp = min(spp, 2) }
        case 3:
            scale *= 0.6
            maxSettle = min(maxSettle, 96)
            bounces = min(bounces, 3)
            glass = min(glass, 2)
            spp = min(spp, moving ? 2 : 1)
        default:
            break
        }
        if moving { reuseCap = min(reuseCap, play ? 16 : 24) }
        return FrameTargets(scale: scale, spp: spp, bounces: bounces,
                            glassBudget: glass, maxSettle: maxSettle,
                            reuseCap: reuseCap)
    }

    // MARK: Encoding

    private func uniforms(camera: CameraBasis, aspect: Float, seed: UInt32,
                          bounces: UInt32, cap: Float, glassBudget: UInt32) -> GPUUniforms {
        var u = GPUUniforms()
        u.camOrigin = SIMD4<Float>(camera.origin, 0)
        u.camRight = SIMD4<Float>(camera.right, 0)
        u.camUp = SIMD4<Float>(camera.up, 0)
        u.camForward = SIMD4<Float>(camera.forward, 0)
        u.tanFovAspect = SIMD2<Float>(tan(fovY / 2), aspect)
        u.frameSeed = seed
        u.maxBounces = bounces
        u.triCount = UInt32(triCount)
        u.emissiveCount = UInt32(emissiveCount)
        u.totalEmissiveArea = totalEmissiveArea
        u.envIntensity = 1.0
        u.historyCap = cap
        u.glassRefractBudget = glassBudget
        u.wetLevel = weatherWet
        u.snowLevel = weatherSnow
        u.dynTime = weatherDynTime
        u.rainFrac = weatherDynRain
        return u
    }

    private func dispatch(_ enc: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          width: Int, height: Int) {
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
    }

    private func encodeGuide(guide: MTLTexture, guideNormal: MTLTexture,
                             camera: CameraBasis,
                             cmd: MTLCommandBuffer) -> Bool {
        guard let guidePipeline, let accel,
              let posBuffer, let nrmBuffer, let idxBuffer, let triMatBuffer, let matBuffer,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(guidePipeline)
        enc.setTexture(guide, index: 0)
        enc.setTexture(guideNormal, index: 1)
        enc.setBuffer(posBuffer, offset: 0, index: 1)
        enc.setBuffer(nrmBuffer, offset: 0, index: 2)
        enc.setBuffer(idxBuffer, offset: 0, index: 5)
        enc.setBuffer(triMatBuffer, offset: 0, index: 6)
        enc.setBuffer(matBuffer, offset: 0, index: 7)
        enc.setAccelerationStructure(accel, bufferIndex: 10)
        let aspect = Float(guide.width) / Float(max(guide.height, 1))
        var u = uniforms(camera: camera, aspect: aspect, seed: 0,
                         bounces: 0, cap: 0, glassBudget: 0)
        enc.setBytes(&u, length: MemoryLayout<GPUUniforms>.stride, index: 0)
        dispatch(enc, pipeline: guidePipeline, width: guide.width, height: guide.height)
        enc.endEncoding()
        return true
    }

    private func encodeReproject(prevAccum: MTLTexture, prevGuide: MTLTexture,
                                 curGuide: MTLTexture, curAccum: MTLTexture,
                                 reuseCap: Float, cmd: MTLCommandBuffer) -> Bool {
        guard let reprojectPipeline, let prevCamera,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(reprojectPipeline)
        enc.setTexture(prevAccum, index: 0)
        enc.setTexture(prevGuide, index: 1)
        enc.setTexture(curGuide, index: 2)
        enc.setTexture(curAccum, index: 3)
        var u = ReprojUniforms()
        u.prevOrigin = SIMD4<Float>(prevCamera.origin, 0)
        u.prevRight = SIMD4<Float>(prevCamera.right, 0)
        u.prevUp = SIMD4<Float>(prevCamera.up, 0)
        u.prevForward = SIMD4<Float>(prevCamera.forward, 0)
        u.tanFovAspect = prevTanAspect
        u.reuseCap = reuseCap
        enc.setBytes(&u, length: MemoryLayout<ReprojUniforms>.stride, index: 0)
        dispatch(enc, pipeline: reprojectPipeline, width: curGuide.width, height: curGuide.height)
        enc.endEncoding()
        return true
    }

    /// Encode `count` path-tracing samples into `accum`. Returns false
    /// when there is nothing to trace.
    private func encodeTrace(accum: MTLTexture, camera: CameraBasis,
                             count: Int, seedStart: UInt32,
                             bounces: UInt32, cap: Float, glassBudget: UInt32,
                             cmd: MTLCommandBuffer) -> Bool {
        guard let tracePipeline, let accel,
              let posBuffer, let nrmBuffer, let uvBuffer, let colBuffer,
              let idxBuffer, let triMatBuffer, let matBuffer,
              let emisTriBuffer, let emisCDFBuffer,
              let texAlbedo, let texNormal, let texData,
              let weatherFieldTex,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(tracePipeline)
        enc.setTexture(accum, index: 0)
        enc.setTexture(texAlbedo, index: 1)
        enc.setTexture(texNormal, index: 2)
        enc.setTexture(texData, index: 3)
        enc.setTexture(weatherFieldTex, index: 4)
        enc.setBuffer(posBuffer, offset: 0, index: 1)
        enc.setBuffer(nrmBuffer, offset: 0, index: 2)
        enc.setBuffer(uvBuffer, offset: 0, index: 3)
        enc.setBuffer(colBuffer, offset: 0, index: 4)
        enc.setBuffer(idxBuffer, offset: 0, index: 5)
        enc.setBuffer(triMatBuffer, offset: 0, index: 6)
        enc.setBuffer(matBuffer, offset: 0, index: 7)
        enc.setBuffer(emisTriBuffer, offset: 0, index: 8)
        enc.setBuffer(emisCDFBuffer, offset: 0, index: 9)
        enc.setAccelerationStructure(accel, bufferIndex: 10)
        let aspect = Float(accum.width) / Float(max(accum.height, 1))
        for i in 0..<count {
            var u = uniforms(camera: camera, aspect: aspect,
                             seed: seedStart + UInt32(i),
                             bounces: bounces, cap: cap, glassBudget: glassBudget)
            enc.setBytes(&u, length: MemoryLayout<GPUUniforms>.stride, index: 0)
            dispatch(enc, pipeline: tracePipeline, width: accum.width, height: accum.height)
        }
        enc.endEncoding()
        return true
    }

    /// Spatial denoise of the current accumulation into the
    /// display texture (interactive / Play frames only — the
    /// settled frame is always shown raw).
    private func encodeDenoise(accum: MTLTexture, guide: MTLTexture,
                               guideNormal: MTLTexture, output: MTLTexture,
                               strength: Float, camera: CameraBasis,
                               cmd: MTLCommandBuffer) -> Bool {
        guard let denoisePipeline,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(denoisePipeline)
        enc.setTexture(accum, index: 0)
        enc.setTexture(guide, index: 1)
        enc.setTexture(guideNormal, index: 2)
        enc.setTexture(output, index: 3)
        var u = DenoiseUniforms()
        u.camOrigin = SIMD4<Float>(camera.origin, 0)
        u.strength = strength
        enc.setBytes(&u, length: MemoryLayout<DenoiseUniforms>.stride, index: 0)
        dispatch(enc, pipeline: denoisePipeline, width: accum.width, height: accum.height)
        enc.endEncoding()
        return true
    }

    private func cameraChanged(_ a: CameraBasis, _ b: CameraBasis) -> Bool {
        simd_length(a.origin - b.origin) > 1e-5
            || simd_length(a.forward - b.forward) > 1e-5
            || simd_length(a.right - b.right) > 1e-5
            || simd_length(a.up - b.up) > 1e-5
    }

    /// Precipitation overlay (live draw only): a fresh pass
    /// over the presented drawable, loading what the tonemap
    /// wrote. Rain = 1600 instances, snow = 1200 instances.
    private func encodePrecip(cmd: MTLCommandBuffer, drawable: CAMetalDrawable, camera: CameraBasis, view: MTKView) {
        if weatherMode == 1 || weatherMode == 2, let precipPipeline, let precipSeedBuffer {
            let precipPass = MTLRenderPassDescriptor()
            precipPass.colorAttachments[0].texture = drawable.texture
            precipPass.colorAttachments[0].loadAction = .load
            precipPass.colorAttachments[0].storeAction = .store
            if let penc = cmd.makeRenderCommandEncoder(descriptor: precipPass) {
                penc.setRenderPipelineState(precipPipeline)
                var pu = PrecipUniforms()
                pu.camOrigin = SIMD4<Float>(camera.origin, 0)
                pu.camRight = SIMD4<Float>(camera.right, 0)
                pu.camUp = SIMD4<Float>(camera.up, 0)
                pu.camForward = SIMD4<Float>(camera.forward, 0)
                pu.tanFovAspect = SIMD2<Float>(
                    tan(fovY / 2),
                    Float(view.drawableSize.width) / Float(max(view.drawableSize.height, 1)))
                pu.time = Float(CACurrentMediaTime().truncatingRemainder(dividingBy: 3600))
                pu.mode = UInt32(weatherMode)
                pu.intensity = weatherDynIntensity
                pu.windXZ = weatherWind
                penc.setVertexBytes(&pu, length: MemoryLayout<PrecipUniforms>.stride, index: 0)
                penc.setVertexBuffer(precipSeedBuffer, offset: 0, index: 1)
                // Draw count breathes with the surge intensity
                // (never below ~45% — a lull still reads as
                // weather; never above the 2048-seed buffer).
                let inten = max(0.35, min(weatherDynIntensity, 1.15))
                let baseCount = weatherMode == 1 ? 1600 : 1200
                let particleCount = Int(Float(baseCount) * (0.45 + 0.55 * inten))
                penc.drawPrimitives(type: .triangle, vertexStart: 0,
                                    vertexCount: particleCount * 6)
                penc.endEncoding()
            }
        }
    }

    /// Tonemap parameters for a live frame. With `shimmer` set,
    /// the fragment adds the display-rate rain-ripple shimmer
    /// on puddle pixels (v3.3); stills pass false so exports
    /// stay clean.
    private func tonemapParams(shimmer: Bool) -> TonemapParams {
        var p = TonemapParams(samples: max(sampleCount, 1), exposure: exposure)
        if shimmer {
            p.shimmerTime = weatherDynTime
            p.shimmerRain = weatherMode == 1 ? weatherDynRain : 0
            p.wetLevel = weatherWet
        }
        return p
    }

    /// One frame for the live MTKView. Returns true when another
    /// frame would still change the picture (the caller chains the
    /// next draw only then — a converged, untouched scene sleeps).
    @discardableResult
    func draw(view: MTKView, camera: CameraBasis, moving: Bool, play: Bool) -> Bool {
        presentationDirty = false
        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer(),
              view.drawableSize.width > 0, view.drawableSize.height > 0
        else { return false }

        let t = targets(moving: moving, play: play)
        lastMaxSettle = t.maxSettle
        // Weather levels moved since the last presented frame:
        // reprojection history gets capped hard for one frame so
        // the ground visibly wets/dries instead of staying frozen
        // in pre-change samples.
        let weatherLevelsMoved = abs(weatherWet - lastWet) > 0.004
            || abs(weatherSnow - lastSnow) > 0.004
        // v3.3: rain traces exactly like snow. The v3.2
        // `rippling` special case (settled path disabled +
        // history reuse capped at 12 while it rained) is gone —
        // ripples are a display-time shimmer in the tonemap
        // pass now, so a rainy scene settles and sleeps like
        // any other.
        let weatherLevelsQuiet = abs(weatherWet - lastTracedWet) <= 0.004
            && abs(weatherSnow - lastTracedSnow) <= 0.004
        // v3.4 dynamic resolution: while moving, the tier scale
        // is multiplied by dynScale (see the completion handler
        // at the bottom of draw) so a scene that cannot hold the
        // 60 fps cadence trades a little motion resolution —
        // never rest resolution — instead of dropping frames.
        dynLock.lock(); let dScale = dynScale; dynLock.unlock()
        let effScale = t.scale * (moving ? dScale : 1.0)
        let pixelW = max(1, Int(Float(view.drawableSize.width) * effScale))
        let pixelH = max(1, Int(Float(view.drawableSize.height) * effScale))
        if accumTex[0] == nil || accumTex[0]!.width != pixelW || accumTex[0]!.height != pixelH {
            accumTex = [makeAccumTexture(width: pixelW, height: pixelH),
                        makeAccumTexture(width: pixelW, height: pixelH)]
            guideTex = [makeGuideTexture(width: pixelW, height: pixelH),
                        makeGuideTexture(width: pixelW, height: pixelH)]
            guideNormalTex = [makeGuideNormalTexture(width: pixelW, height: pixelH),
                              makeGuideNormalTexture(width: pixelW, height: pixelH)]
            displayTex = makeAccumTexture(width: pixelW, height: pixelH)
            historyValid = false
            sampleCount = 0
        }
        let dst = curIdx
        let src = 1 - curIdx
        guard let curAccum = accumTex[dst], let curGuide = guideTex[dst] else { return false }
        let curGuideNormal = guideNormalTex[dst]

        // Settled fast path: history valid, camera identical to
        // the last presented frame, sample cap reached — just
        // present the finished image (which lives in the other
        // ping-pong slot) and sleep. Also serves weather-on
        // frames when the wet/snow levels have not moved since
        // the last trace (the settled image already includes
        // that shading); precipitation is overlaid on top and
        // the frame chain stays alive so it keeps animating.
        if accel != nil, (weatherMode == 0 || weatherLevelsQuiet), historyValid, sampleCount >= t.maxSettle,
           let pc = prevCamera, !cameraChanged(pc, camera),
           let settled = accumTex[src] {
            if let tonemapPipeline,
               let renc = cmd.makeRenderCommandEncoder(descriptor: passDesc) {
                var params = tonemapParams(shimmer: true)
                renc.setRenderPipelineState(tonemapPipeline)
                renc.setFragmentTexture(settled, index: 0)
                renc.setFragmentTexture(guideTex[src], index: 1)
                renc.setFragmentTexture(guideNormalTex[src], index: 2)
                renc.setFragmentTexture(weatherFieldTex, index: 3)
                renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
                renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                renc.endEncoding()
            }
            if weatherMode != 0 {
                encodePrecip(cmd: cmd, drawable: drawable, camera: camera, view: view)
            }
            // Settled: dynamic resolution stands down — the next
            // motion starts from the tier's full motion scale.
            dynLock.lock()
            dynScale = 1.0; dynCadenceEMA = 0; dynLastCompletion = 0
            dynFramesSinceAdjust = 0
            dynLock.unlock()
            cmd.present(drawable)
            cmd.commit()
            wasMoving = moving
            lastWet = weatherWet
            lastSnow = weatherSnow
            return weatherMode != 0
        }

        var traced = false
        if accel != nil {
            // Guide pass for the current camera (becomes next
            // frame's "previous"), then history: reproject or clear.
            if let curGuideNormal {
                _ = encodeGuide(guide: curGuide, guideNormal: curGuideNormal,
                                camera: camera, cmd: cmd)
            }
            let camMoved = prevCamera.map { cameraChanged($0, camera) } ?? true
            if historyValid, let prevAccum = accumTex[src], let prevGuide = guideTex[src],
               prevCamera != nil {
                var reuse = t.reuseCap
                if let pc = prevCamera {
                    let dOrigin = simd_length(camera.origin - pc.origin)
                    let cosA = simd_dot(simd_normalize(camera.forward),
                                        simd_normalize(pc.forward))
                    if dOrigin > 1.5 || cosA < 0.86 {
                        reuse = 0 // teleport: no smear
                        sampleCount = 0 // the image really is fresh
                    }
                }
                if weatherLevelsMoved {
                    reuse = min(reuse, 8) // ground is changing state
                }
                _ = encodeReproject(prevAccum: prevAccum, prevGuide: prevGuide,
                                    curGuide: curGuide, curAccum: curAccum,
                                    reuseCap: reuse, cmd: cmd)
                if wasMoving && !moving {
                    // Settle bookkeeping: per-pixel history after
                    // motion is roughly the moving reuse cap.
                    sampleCount = min(sampleCount, play ? 16 : 24)
                }
            } else {
                clearAccum(curAccum, cmd: cmd)
                sampleCount = 0
            }
            // Trace more samples unless settled and the camera
            // has not moved since the last presented frame.
            // Active weather always traces (at the non-moving
            // tier's spp): precipitation frames keep refreshing
            // the wet/snow shading instead of freezing.
            if moving || camMoved || sampleCount < t.maxSettle || weatherMode != 0 {
                if encodeTrace(accum: curAccum, camera: camera, count: t.spp,
                               seedStart: frameSeedCounter, bounces: t.bounces,
                               cap: Float(t.maxSettle), glassBudget: t.glassBudget,
                               cmd: cmd) {
                    traced = true
                    lastTracedWet = weatherWet
                    lastTracedSnow = weatherSnow
                    sampleCount += UInt32(t.spp)
                    frameSeedCounter &+= UInt32(t.spp)
                } else {
                    sampleCount = t.maxSettle
                }
            }
            prevCamera = camera
            prevTanAspect = SIMD2<Float>(tan(fovY / 2),
                                         Float(curAccum.width) / Float(max(curAccum.height, 1)))
            historyValid = true
            curIdx = src
        } else if !historyValid {
            clearAccum(curAccum, cmd: cmd)
            historyValid = true // empty scene is now presented
            sampleCount = 0
        }

        // Display source: interactive / Play / still-converging
        // frames go through the spatial denoise pass; the settled
        // frame (sampleCount >= maxSettle) is shown raw.
        var displaySource = curAccum
        if accel != nil, moving || play || sampleCount < 160,
           denoisePipeline != nil, let displayTex, let curGuideNormal {
            let strength: Float = (play || moving) ? 1.0 : 0.65
            if encodeDenoise(accum: curAccum, guide: curGuide,
                             guideNormal: curGuideNormal, output: displayTex,
                             strength: strength, camera: camera, cmd: cmd) {
                displaySource = displayTex
            }
        }

        if let tonemapPipeline,
           let renc = cmd.makeRenderCommandEncoder(descriptor: passDesc) {
            var params = tonemapParams(shimmer: true)
            renc.setRenderPipelineState(tonemapPipeline)
            renc.setFragmentTexture(displaySource, index: 0)
            renc.setFragmentTexture(curGuide, index: 1)
            renc.setFragmentTexture(curGuideNormal, index: 2)
            renc.setFragmentTexture(weatherFieldTex, index: 3)
            renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
            renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renc.endEncoding()
        }

        // Precipitation overlay (live draw only).
        encodePrecip(cmd: cmd, drawable: drawable, camera: camera, view: view)

        // v3.4 dynamic resolution feedback: the delivered cadence
        // of traced frames (completion-to-completion) is the
        // ground truth for "is this holding 60 fps". While moving,
        // a cadence EMA clearly above the 16.7 ms vsync quantum
        // means frames are landing on 33 ms steps — trim the
        // motion resolution a notch (floor 0.70x of the tier
        // scale); a locked 60 fps cadence earns it back. Steps
        // are at most one per 45 traced frames, so the scale
        // never hunts; rest frames never feed the EMA.
        if traced {
            let frameMoving = moving
            cmd.addCompletedHandler { [weak self] _ in
                guard let self else { return }
                let now = ProcessInfo.processInfo.systemUptime
                self.dynLock.lock()
                if frameMoving {
                    if self.dynLastCompletion > 0 {
                        let dt = now - self.dynLastCompletion
                        if dt > 0.004, dt < 0.25 {
                            self.dynCadenceEMA = self.dynCadenceEMA == 0
                                ? dt
                                : self.dynCadenceEMA * 0.9 + dt * 0.1
                        }
                    }
                    self.dynLastCompletion = now
                    self.dynFramesSinceAdjust += 1
                    if self.dynFramesSinceAdjust >= 45, self.dynCadenceEMA > 0 {
                        if self.dynCadenceEMA > 0.0195, self.dynScale > 0.70 {
                            self.dynScale = max(0.70, self.dynScale - 0.08)
                            self.dynFramesSinceAdjust = 0
                        } else if self.dynCadenceEMA < 0.0169, self.dynScale < 1.0 {
                            self.dynScale = min(1.0, self.dynScale + 0.05)
                            self.dynFramesSinceAdjust = 0
                        }
                    }
                } else {
                    self.dynLastCompletion = 0
                }
                self.dynLock.unlock()
            }
        }

        cmd.present(drawable)
        cmd.commit()
        wasMoving = moving
        lastWet = weatherWet
        lastSnow = weatherSnow
        guard accel != nil else { return false }
        return moving || weatherMode != 0 || (traced && sampleCount < t.maxSettle)
    }

    /// Render the current scene offscreen and return a PNG (used for
    /// project thumbnails and PNG export). Always full quality:
    /// full bounce depth, generous glass budget, no history cap.
    func renderStillPNG(camera: CameraBasis, width: Int, height: Int, samples: Int) -> Data? {
        guard let accum = makeAccumTexture(width: width, height: height),
              let cmd = queue.makeCommandBuffer() else { return nil }
        clearAccum(accum, cmd: cmd)
        let traced = encodeTrace(accum: accum, camera: camera, count: samples,
                                 seedStart: 0, bounces: 6, cap: 1_000_000,
                                 glassBudget: 6, cmd: cmd)
        guard traced else { return nil }
        // Tonemap into a readable BGRA texture.
        let targetDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        targetDesc.usage = [.renderTarget, .shaderRead]
        targetDesc.storageMode = .shared
        guard let target = device.makeTexture(descriptor: targetDesc),
              let tonemapPipeline else { return nil }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        guard let renc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return nil }
        var params = TonemapParams(samples: UInt32(samples), exposure: exposure)
        renc.setRenderPipelineState(tonemapPipeline)
        renc.setFragmentTexture(accum, index: 0)
        // Inert placeholders: the fragment's shimmer block is
        // gated on shimmerRain > 0 (0 here) and never samples
        // indices 1-3 for stills.
        renc.setFragmentTexture(weatherFieldTex, index: 1)
        renc.setFragmentTexture(weatherFieldTex, index: 2)
        renc.setFragmentTexture(weatherFieldTex, index: 3)
        renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
        renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { raw in
            target.getBytes(raw.baseAddress!, bytesPerRow: width * 4,
                            from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue
                                      | CGBitmapInfo.byteOrder32Little.rawValue)
        guard let image = CGImage(width: width, height: height, bitsPerComponent: 8,
                                  bitsPerPixel: 32, bytesPerRow: width * 4,
                                  space: cs, bitmapInfo: bitmapInfo,
                                  provider: provider, decode: nil,
                                  shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }

    // MARK: Picking (CPU raycast over the merged world triangles)

    func pick(origin: SIMD3<Float>, dir: SIMD3<Float>) -> UUID? {
        guard cpuIndices.count >= 3 else { return nil }
        var best: Float = .greatestFiniteMagnitude
        var bestObj: UUID? = nil
        let triTotal = cpuIndices.count / 3
        for t in 0..<triTotal {
            let ia = Int(cpuIndices[3 * t]), ib = Int(cpuIndices[3 * t + 1]), ic = Int(cpuIndices[3 * t + 2])
            let p0 = SIMD3<Float>(cpuPositions[3 * ia], cpuPositions[3 * ia + 1], cpuPositions[3 * ia + 2])
            let p1 = SIMD3<Float>(cpuPositions[3 * ib], cpuPositions[3 * ib + 1], cpuPositions[3 * ib + 2])
            let p2 = SIMD3<Float>(cpuPositions[3 * ic], cpuPositions[3 * ic + 1], cpuPositions[3 * ic + 2])
            if let d = rayTriangle(origin: origin, dir: dir, p0: p0, p1: p1, p2: p2), d < best {
                best = d
                bestObj = cpuTriObject[t]
            }
        }
        return bestObj
    }

    private func rayTriangle(origin: SIMD3<Float>, dir: SIMD3<Float>,
                             p0: SIMD3<Float>, p1: SIMD3<Float>, p2: SIMD3<Float>) -> Float? {
        let e1 = p1 - p0, e2 = p2 - p0
        let pvec = simd_cross(dir, e2)
        let det = simd_dot(e1, pvec)
        if abs(det) < 1e-9 { return nil }
        let inv = 1 / det
        let tvec = origin - p0
        let u = simd_dot(tvec, pvec) * inv
        if u < 0 || u > 1 { return nil }
        let qvec = simd_cross(tvec, e1)
        let v = simd_dot(dir, qvec) * inv
        if v < 0 || u + v > 1 { return nil }
        let d = simd_dot(e2, qvec) * inv
        return d > 1e-4 ? d : nil
    }
}
