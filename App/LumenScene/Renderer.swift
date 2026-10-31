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
//   * Texture slices: the MAIN arrays (512) come from
//     TextureGen.presetNames plus the user's custom texture
//     library; roads live in a SEPARATE 1024 array set (v4.1)
//     encoded as texIndex >= 4096. `reloadTextureArraysIfNeeded()`
//     rebuilds both array sets when the library changes.
//     Material flag bit3 = texture FIT.
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
    var dynTriCount: UInt32 = 0  // v4: dynamic-layer triangles (unified ids start at triCount)
    var padDyn: UInt32 = 0
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
    var glassParams = SIMD4<Float>(0, 0, 1.5, 0) // transmission, roughness, ior, w = mean texture roughness (v4.1 F3, guide gloss only)
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

struct LineUniforms {
    var camOrigin = SIMD4<Float>(0, 0, 0, 0)
    var camRight = SIMD4<Float>(1, 0, 0, 0)
    var camUp = SIMD4<Float>(0, 1, 0, 0)
    var camForward = SIMD4<Float>(0, 0, -1, 0)
    var tanFovAspect = SIMD2<Float>(0.7, 1)
    var pad0: Float = 0
    var pad1: Float = 0
}

/// Sentinel object id for dynamic triangles that carry no scene
/// object (particle soups, anonymous dynamic entries). pick()
/// never returns it: a nearer sentinel hit occludes and yields
/// nil, exactly like a hit on unpickable geometry.
private let dynNilObjectSentinel = UUID(uuidString: "00000000-0000-0000-0000-000000000000")!

typealias CameraBasis = (origin: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>)

final class Renderer {
    /// v4: one dynamic-layer entry — a local mesh baked by
    /// `matrix` on compose/flush, shaded with `material`, and
    /// reported by pick() as `objectID` (nil = unpickable).
    struct DynEntry {
        var key: UUID
        var mesh: MeshData
        var matrix: float4x4
        var material: GPUMaterial
        var objectID: UUID?
    }

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
    private var linePipeline: MTLRenderPipelineState?
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

    // v4 dynamic layer: a second triangle soup beside the static
    // one, in its own buffers (slots 11-17) and its own AS.
    // Unified ids: static 0..<triCount, dynamic triCount... .
    private var dynAccel: MTLAccelerationStructure?
    private var dynPosBuffer: MTLBuffer?
    private var dynNrmBuffer: MTLBuffer?
    private var dynUVBuffer: MTLBuffer?
    private var dynColBuffer: MTLBuffer?
    private var dynIdxBuffer: MTLBuffer?
    private var dynTriMatBuffer: MTLBuffer?
    private var dynTriCount: Int = 0
    /// v4.1 (F9): true only when the dynamic AS built AND every
    /// dynamic buffer exists. When false, uniforms force
    /// dynTriCount = 0 for both kernels (the dummy/static
    /// bindings in bindDynamicBuffers are then safe).
    private var dynUsable: Bool = false
    private var dynDummyBuffer: MTLBuffer?
    private var cpuDynPositions: [Float] = []
    private var cpuDynNormals: [Float] = []
    private var cpuDynUVs: [Float] = []
    private var cpuDynColors: [Float] = []
    private var cpuDynIndices: [UInt32] = []
    private var cpuDynTriMats: [UInt32] = []
    /// Per-dynamic-tri offset into the dynamic material section
    /// (entry order, then soup order); cpuDynTriMats is this plus
    /// the dynamic material base, refreshed when the static
    /// material count moves.
    private var cpuDynMatOffsets: [Int] = []
    private var cpuDynTriObject: [UUID] = []
    private var dynEntries: [DynEntry] = []
    private var particleSoupsStored: [(mesh: MeshData, material: GPUMaterial)] = []
    private var dynDirtyKeys: Set<UUID> = []
    private struct DynSlice {
        var vertexStart: Int = 0
        var vertexCount: Int = 0
        var triStart: Int = 0
        var triCount: Int = 0
        /// v4.1 (F17): true when compose truncated this soup and
        /// remapped its vertices — flushDynamic must not re-bake
        /// it in world order (see composeDynamic).
        var remapped: Bool = false
    }
    private var dynSlices: [UUID: DynSlice] = [:]
    /// Static object materials from the last rebuild/material
    /// update (WITHOUT the sheet/dynamic/particle tail) — the
    /// full GPU table is composed from this plus renderer state.
    private var cachedStaticMaterials: [GPUMaterial] = []
    /// Emissive tris contributed by the dynamic layer in the
    /// last full emissive build (drives the rescan skip).
    private var dynEmissiveTriCount: Int = 0
    /// v4.1 (F4ii): cached STATIC portion of the emissive index
    /// (tris/areas/centroids from the last full static scan).
    /// composeDynamic / setDynamicMaterial reuse it when static
    /// geometry+materials are untouched since that scan; full
    /// rebuilds and material-only updates rescan and refill it.
    private var cachedStaticEmisTris: [UInt32] = []
    private var cachedStaticEmisAreas: [Float] = []
    private var cachedStaticEmisCentroids: [SIMD3<Float>] = []
    private var staticEmisCacheValid: Bool = false
    /// v4.1 (F11): unified emissive id -> index in the full
    /// emissive arrays, built by uploadEmissiveIndex so
    /// flushDynamic centroid refresh is O(1) per tri.
    private var fullEmisIndexByID: [UInt32: Int] = [:]

    // v4 light culling: full emissive set + culled set. The
    // active pair is what emisTriBuffer/emisCDFBuffer and
    // emissiveCount/totalEmissiveArea currently describe.
    private var fullEmisTris: [UInt32] = []
    private var fullEmisCDF: [Float] = []
    private var fullEmisAreas: [Float] = []
    private var fullEmisCentroids: [SIMD3<Float>] = []
    private var fullTotalEmissiveArea: Float = 0
    private var culledEmisTris: [UInt32] = []
    private var culledEmisCDF: [Float] = []
    private var culledTotalEmissiveArea: Float = 0
    private var cullingActive: Bool = false
    private var lastCullCamera: CameraBasis?
    private var lastCullTime: CFTimeInterval = 0
    /// When false, NEE always samples the full emissive set.
    var lightCullingEnabled: Bool = true

    /// v4 area streaming: scene objects whose areaID is in this
    /// set are excluded by rebuild/updateMaterialsOnly.
    var unloadedAreaIDs: Set<UUID> = []
    /// v4 edit-mode area-box overlay (world-space AABBs).
    var debugBoxes: [(min: SIMD3<Float>, max: SIMD3<Float>, color: SIMD4<Float>)] = []
    private var _collisionTriangles: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
    /// World-space triangles from BOTH layers (v4.1 F5: static
    /// post-weather geometry plus the dynamic layer, world-space
    /// as rendered), refreshed by rebuild(), composeDynamic()
    /// and flushDynamic() from the CPU mirrors.
    var collisionTriangles: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] {
        _collisionTriangles
    }

    // CPU copies for picking + material-only updates.
    private var cpuPositions: [Float] = []
    private var cpuIndices: [UInt32] = []
    private var cpuTriObject: [UUID] = []
    private var cpuTriMats: [UInt32] = []
    private var cachedObjectOrder: [UUID] = []

    // Texture arrays: the MAIN arrays (512) hold built-in
    // presets plus the custom texture library. `sliceNames` is
    // kept in lockstep with the main array slices (slice i holds
    // sliceNames[i]). v4.1: roads live in a SEPARATE 1024 array
    // set (roadAlbedo/roadNormal/roadData, slices in
    // TextureGen.roadNames order), encoded in materials as
    // texIndex = 4096 + roadSlice + 1 so main-array slice
    // indices for existing content never move.
    private var texAlbedo: MTLTexture?
    private var texNormal: MTLTexture?
    private var texData: MTLTexture?
    private var roadTexAlbedo: MTLTexture?
    private var roadTexNormal: MTLTexture?
    private var roadTexData: MTLTexture?
    /// v4.1 (N1): true only when ALL three road arrays were
    /// created non-nil in buildTextureArrays. When false, road
    /// materials degrade to untextured colour (texIndexValue
    /// returns 0 for road presets) instead of gating all tracing
    /// on the road arrays' allocation.
    private var roadsAvailable: Bool = false
    /// v4.1 (N1): lazily created 1x1 dummy texture2d_array bound
    /// at trace slots 5-7 when a road array is missing. Never
    /// sampled once roadsAvailable == false (no material then
    /// encodes >= 4096).
    private var roadDummyTex: MTLTexture?
    private(set) var sliceNames: [String] = []
    private var roadSliceNames: [String] = []
    /// v4.1 (F3): per-slice MEAN data-channel roughness, keyed by
    /// the material texIndex encoding (main: slice+1; road:
    /// 4096+slice+1). Baked into GPUMaterial.glassParams.w so
    /// the guide kernel can price textured gloss without
    /// sampling the data texture.
    private var texMeanRoughness: [UInt32: Float] = [:]
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
    /// v4.1 (recon §5): environment (sky) light intensity,
    /// set by the session funnel; uniforms() feeds it to the
    /// trace kernel instead of a hardcoded 1.0.
    var envIntensity: Float = 1.0
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
    /// v4.1 (F22): when the camera last stopped moving, for the
    /// governor's ~2 s stand-down timer (guarded by dynLock).
    private var dynStillSince: CFTimeInterval = 0
    private let dynLock = NSLock()
    /// v4.1 (F12): frames since motion last stopped (reset on
    /// any moving draw, counted up otherwise). The denoise gate
    /// stays open for ~45 frames after motion so newly exposed,
    /// still-young pixels keep their denoise.
    private var framesSinceMove: Int = 10_000
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
        if let vfn = library.makeFunction(name: "lineVertex"),
           let ffn = library.makeFunction(name: "lineFragment") {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfn
            desc.fragmentFunction = ffn
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            linePipeline = try? device.makeRenderPipelineState(descriptor: desc)
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

    /// Fingerprint of everything the texture arrays are built
    /// from — the road set (v4.1: roads are their own array set,
    /// so the road names are part of the rebuild signal) plus
    /// the custom texture library (ids + names): any change is
    /// the rebuild signal.
    private func textureLibraryFingerprint() -> String {
        let roads = TextureGen.roadNames.joined(separator: ",")
        let customs = CustomTextureStore.list()
            .map { "\($0.id.uuidString):\($0.name)" }
            .joined(separator: "|")
        return "roads:\(roads)|\(customs)"
    }

    /// Nearest-neighbour RGBA8 resample (v4.1 F10): a set whose
    /// size differs from its array's established size is
    /// resampled into the array instead of being silently
    /// dropped, so names and slices stay in lockstep and the
    /// material still renders textured.
    private func resampleRGBA8(_ data: Data, from srcSize: Int, to dstSize: Int) -> Data {
        guard srcSize > 0, dstSize > 0, srcSize != dstSize,
              data.count >= srcSize * srcSize * 4 else { return data }
        var out = Data(count: dstSize * dstSize * 4)
        data.withUnsafeBytes { srcRaw in
            out.withUnsafeMutableBytes { dstRaw in
                guard let s = srcRaw.baseAddress?.assumingMemoryBound(to: UInt8.self),
                      let d = dstRaw.baseAddress?.assumingMemoryBound(to: UInt8.self)
                else { return }
                for y in 0..<dstSize {
                    let sy = min(srcSize - 1, y * srcSize / dstSize)
                    for x in 0..<dstSize {
                        let sx = min(srcSize - 1, x * srcSize / dstSize)
                        let si = (sy * srcSize + sx) * 4
                        let di = (y * dstSize + x) * 4
                        d[di] = s[si]; d[di + 1] = s[si + 1]
                        d[di + 2] = s[si + 2]; d[di + 3] = s[si + 3]
                    }
                }
            }
        }
        return out
    }

    /// Mean of the data channel's R (roughness) over a set, in
    /// 0…1 (v4.1 F3: feeds GPUMaterial.glassParams.w via
    /// texMeanRoughness).
    private func meanDataRoughness(_ data: Data, size: Int) -> Float {
        let pixels = min(size * size, data.count / 4)
        guard pixels > 0 else { return 0.9 }
        var sum: UInt64 = 0
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress?.assumingMemoryBound(to: UInt8.self)
            else { return }
            for i in 0..<pixels { sum += UInt64(base[i * 4]) }
        }
        return Float(sum) / (Float(pixels) * 255)
    }

    private func buildTextureArrays() {
        let customMetas = CustomTextureStore.list()
        builtTextureFingerprint = textureLibraryFingerprint()
        func makeArray(_ slices: [Data], size: Int) -> MTLTexture? {
            guard !slices.isEmpty, size > 0 else { return nil }
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
        var means: [UInt32: Float] = [:]
        // MAIN arrays: presets + customs ONLY (v4.1 — roads are
        // excluded, so main slice indices for existing content
        // are unchanged from v4.0).
        let candidateNames = TextureGen.presetNames
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
            // doesn't match the established size is RESAMPLED
            // into it (v4.1 F10) — name and slices stay in
            // lockstep, nothing is silently dropped.
            var albedoData = set.albedo, normalData = set.normal, dataData = set.data
            if size == 0 {
                size = set.size
            } else if set.size != size {
                albedoData = resampleRGBA8(set.albedo, from: set.size, to: size)
                normalData = resampleRGBA8(set.normal, from: set.size, to: size)
                dataData = resampleRGBA8(set.data, from: set.size, to: size)
            }
            means[UInt32(builtNames.count + 1)] = meanDataRoughness(dataData, size: size)
            builtNames.append(name)
            albedos.append(albedoData); normals.append(normalData); datas.append(dataData)
        }
        sliceNames = builtNames
        if !builtNames.isEmpty, size > 0 {
            texAlbedo = makeArray(albedos, size: size)
            texNormal = makeArray(normals, size: size)
            texData = makeArray(datas, size: size)
        } else {
            texAlbedo = nil; texNormal = nil; texData = nil
        }
        // ROAD arrays (v4.1): separate type2DArray set at
        // TextureGen.roadSize, slices in roadNames order —
        // materials encode them as 4096 + slice + 1.
        let roadSize = TextureGen.roadSize
        var rAlbedos: [Data] = [], rNormals: [Data] = [], rDatas: [Data] = []
        for (i, name) in TextureGen.roadNames.enumerated() {
            let set = TextureGen.makeSet(preset: name)
            var albedoData = set.albedo, normalData = set.normal, dataData = set.data
            if set.size != roadSize {
                albedoData = resampleRGBA8(set.albedo, from: set.size, to: roadSize)
                normalData = resampleRGBA8(set.normal, from: set.size, to: roadSize)
                dataData = resampleRGBA8(set.data, from: set.size, to: roadSize)
            }
            means[UInt32(4096 + i + 1)] = meanDataRoughness(dataData, size: roadSize)
            rAlbedos.append(albedoData); rNormals.append(normalData); rDatas.append(dataData)
        }
        roadSliceNames = TextureGen.roadNames
        roadTexAlbedo = makeArray(rAlbedos, size: roadSize)
        roadTexNormal = makeArray(rNormals, size: roadSize)
        roadTexData = makeArray(rDatas, size: roadSize)
        // v4.1 (N1): roads are usable only when every array
        // allocated; a single failure degrades roads alone.
        roadsAvailable = roadTexAlbedo != nil && roadTexNormal != nil
            && roadTexData != nil
        texMeanRoughness = means
    }

    /// v4.1 (N1): the dummy road array bound when a real road
    /// array is missing (see roadDummyTex). Created lazily, once.
    private func roadDummyTexture() -> MTLTexture? {
        if let d = roadDummyTex { return d }
        let desc = MTLTextureDescriptor()
        desc.textureType = .type2DArray
        desc.pixelFormat = .rgba8Unorm
        desc.width = 1; desc.height = 1
        desc.arrayLength = 1
        desc.usage = [.shaderRead]
        desc.storageMode = .shared
        let t = device.makeTexture(descriptor: desc)
        roadDummyTex = t
        return t
    }

    /// Rebuilds the texture arrays when the custom texture
    /// library changed since they were built. Returns true when
    /// a rebuild happened (accumulation is reset — slice indices
    /// may have moved).
    ///
    /// CONTRACT (v4.1 F14): this must be called BEFORE the first
    /// rebuild of a session, or the caller must recompute
    /// materials afterwards — GPUMaterial texIndex values are
    /// baked into the material buffer at rebuild/material-update
    /// time, and a slice remap under live materials would make
    /// them sample the wrong slice. (Today's only caller is the
    /// session init, which runs before the first rebuild.)
    func reloadTextureArraysIfNeeded() -> Bool {
        guard textureLibraryFingerprint() != builtTextureFingerprint else { return false }
        buildTextureArrays()
        resetAccumulation()
        return true
    }

    /// Texture encoding for a texture name (0 = none in
    /// materials). Built-in presets and "custom:<uuid>" names
    /// resolve against the MAIN slice list (slice+1); roads
    /// resolve to the separate road arrays as
    /// 4096 + roadSlice + 1 (v4.1).
    func texIndexValue(forPreset preset: String) -> UInt32 {
        if TextureGen.isRoad(preset) {
            // v4.1 (N1): no road arrays -> road materials fall
            // back to untextured colour (index 0); everything
            // else traces normally.
            guard roadsAvailable,
                  let i = TextureGen.roadNames.firstIndex(of: preset) else { return 0 }
            return UInt32(4096 + i + 1)
        }
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

    /// v4.1 (F16): THE inclusion predicate for scene objects —
    /// shared verbatim by rebuild() and updateMaterialsOnly()
    /// so their object lists can never diverge. Returns the
    /// object's local mesh when the object is included (visible,
    /// area loaded, mesh resolvable, non-empty), else nil.
    private func includedLocalMesh(for obj: SceneObject,
                                   meshes: [UUID: MeshData]) -> MeshData? {
        guard obj.visible, !isUnloadedArea(obj) else { return nil }
        let local: MeshData
        if obj.source == "primitive" {
            local = MeshGen.primitive(kind: obj.primitiveKind, params: obj.params)
        } else if obj.meshFile != nil, let m = meshes[obj.id] {
            local = m
        } else {
            return nil
        }
        return local.isEmpty ? nil : local
    }

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
        for obj in scene.objects {
            guard let local = includedLocalMesh(for: obj, meshes: meshes) else { continue }
            included.append((obj, local.baked(by: obj.transform.matrix)))
        }
        let sheetMatIndex = UInt32(included.count)
        let weatherGeoActive = weatherGeoWet > 0.001 || weatherGeoSnow > 0.001
        var triBudget = 220_000
        var sheetEmitted = false
        for (obj, world) in included {
            let matIndex = UInt32(materials.count)
            materials.append(gpuMaterialForObject(obj))
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
        sceneHasSheetMaterial = sheetEmitted
        cachedStaticMaterials = materials

        triCount = indices.count / 3
        cpuPositions = positions
        cpuIndices = indices
        cpuTriObject = triObjects
        cpuTriMats = triMats
        cachedObjectOrder = objectOrder
        _collisionTriangles = computeCollisionTriangles()
        // The dynamic material base may have moved with the new
        // static count; the dynamic layer itself survives.
        refreshDynTriMats()
        let fullMats = fullMaterialTable()
        uploadEmissiveIndex(materials: fullMats)
        matBuffer = makeBuffer(fullMats)

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
        // exactly — via the ONE shared inclusion predicate
        // (v4.1 F16: includedLocalMesh, also used by rebuild).
        var candidate: [UUID] = []
        for obj in scene.objects {
            if includedLocalMesh(for: obj, meshes: meshes) != nil {
                candidate.append(obj.id)
            }
        }
        guard candidate == cachedObjectOrder else { return false }
        var materials: [GPUMaterial] = []
        materials.reserveCapacity(candidate.count)
        for id in candidate {
            guard let obj = scene.object(id: id) else { return false }
            materials.append(gpuMaterialForObject(obj))
        }
        // v3.4/v4: reproduce the geometry rebuild's material
        // table exactly — static objects, the puddle sheet
        // material (when the geometry emitted sheets), then the
        // renderer's dynamic-entry and particle-soup materials.
        cachedStaticMaterials = materials
        refreshDynTriMats()
        let fullMats = fullMaterialTable()
        uploadEmissiveIndex(materials: fullMats)
        matBuffer = makeBuffer(fullMats)
        resetAccumulation()
        return true
    }

    /// Builds the emissive triangle list + area CDF into GPU
    /// buffers from the cached CPU geometry and a material table.
    /// v4: the index spans BOTH triangle sets with unified ids
    /// (static 0..<triCount, dynamic triCount..<triCount+
    /// dynTriCount; positions come from the respective CPU
    /// arrays). The result becomes the FULL set — light culling
    /// (see draw) may later swap a culled subset in as the
    /// active set. Optimisation: on per-frame particle-soup
    /// uploads, when no dynamic/particle material is emissive
    /// and the dynamic layer contributed no emissive tris last
    /// time either, the dynamic tris are not rescanned at all.
    /// v4.1 (F4): culling state SURVIVES an upload — if culling
    /// was active, the culled set is recomputed for the saved
    /// camera against the new full set and re-activated (no
    /// accumulation reset here; callers own reset logic), instead
    /// of the old wipe that forced a re-cull + reset next draw.
    /// v4.1 (F4ii): `rescanStatic` false reuses the cached
    /// static portion of the index — for callers that changed
    /// only dynamic content (composeDynamic, setDynamicMaterial)
    /// while static geometry+materials are untouched since the
    /// last full scan.
    private func uploadEmissiveIndex(materials: [GPUMaterial], rescanStatic: Bool = true) {
        let wasCulling = cullingActive
        let savedCullCamera = lastCullCamera
        let savedCullTime = lastCullTime
        var emisTris: [UInt32] = []
        var emisAreas: [Float] = []
        var emisCentroids: [SIMD3<Float>] = []
        if rescanStatic || !staticEmisCacheValid {
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
                        emisCentroids.append((p0 + p1 + p2) / 3)
                    }
                }
            }
            cachedStaticEmisTris = emisTris
            cachedStaticEmisAreas = emisAreas
            cachedStaticEmisCentroids = emisCentroids
            staticEmisCacheValid = true
        } else {
            emisTris = cachedStaticEmisTris
            emisAreas = cachedStaticEmisAreas
            emisCentroids = cachedStaticEmisCentroids
        }
        var dynEmis = 0
        if dynTriCount > 0 {
            let dynHasEmissive = dynEntries.contains { ($0.material.flags & 1) != 0 }
                || particleSoupsStored.contains { ($0.material.flags & 1) != 0 }
            let skipDynScan = !dynHasEmissive && dynEmissiveTriCount == 0
            if !skipDynScan {
                for dt in 0..<dynTriCount {
                    let m = materials[Int(cpuDynTriMats[dt])]
                    guard (m.flags & 1) != 0 else { continue }
                    let a = cpuDynIndices[3 * dt], b = cpuDynIndices[3 * dt + 1], c = cpuDynIndices[3 * dt + 2]
                    let p0 = SIMD3<Float>(cpuDynPositions[3 * Int(a)], cpuDynPositions[3 * Int(a) + 1], cpuDynPositions[3 * Int(a) + 2])
                    let p1 = SIMD3<Float>(cpuDynPositions[3 * Int(b)], cpuDynPositions[3 * Int(b) + 1], cpuDynPositions[3 * Int(b) + 2])
                    let p2 = SIMD3<Float>(cpuDynPositions[3 * Int(c)], cpuDynPositions[3 * Int(c) + 1], cpuDynPositions[3 * Int(c) + 2])
                    let area = 0.5 * simd_length(simd_cross(p1 - p0, p2 - p0))
                    if area > 1e-9 {
                        emisTris.append(UInt32(triCount + dt))
                        emisAreas.append(area)
                        emisCentroids.append((p0 + p1 + p2) / 3)
                        dynEmis += 1
                    }
                }
            }
        }
        dynEmissiveTriCount = dynEmis
        var cdf: [Float] = []
        var acc: Float = 0
        for a in emisAreas { acc += a; cdf.append(acc) }
        fullEmisTris = emisTris
        fullEmisCDF = cdf
        fullEmisAreas = emisAreas
        fullEmisCentroids = emisCentroids
        fullTotalEmissiveArea = acc
        // v4.1 (F11): unified-id -> full-set index map for
        // flushDynamic's centroid refresh.
        var indexByID: [UInt32: Int] = [:]
        indexByID.reserveCapacity(emisTris.count)
        for (i, t) in emisTris.enumerated() { indexByID[t] = i }
        fullEmisIndexByID = indexByID
        // v4.1 (F4i): preserve culling across the upload. If
        // culling was active, recompute the culled set for the
        // saved camera against the NEW full set and re-activate
        // it WITHOUT resetting accumulation (a genuinely changed
        // set is a legitimate content change — the caller's own
        // reset logic covers it; an unchanged set is invisible).
        if wasCulling, let cam = savedCullCamera {
            let culled = computeCulledSet(camera: cam)
            culledEmisTris = culled.tris
            culledEmisCDF = culled.cdf
            culledTotalEmissiveArea = culled.total
            lastCullCamera = cam
            lastCullTime = savedCullTime
            activateCulledEmissive()
        } else {
            culledEmisTris = []
            culledEmisCDF = []
            culledTotalEmissiveArea = 0
            cullingActive = false
            lastCullCamera = nil
            lastCullTime = 0
            applyEmissiveSet(tris: emisTris, cdf: cdf, total: acc)
        }
    }

    /// Point the active emissive buffers/counts at one set.
    private func applyEmissiveSet(tris: [UInt32], cdf: [Float], total: Float) {
        emissiveCount = tris.count
        totalEmissiveArea = total
        emisTriBuffer = makeBuffer(tris)
        emisCDFBuffer = makeBuffer(cdf)
    }

    /// Restore the full emissive set as active (no accumulation
    /// reset here — callers decide).
    private func restoreFullEmissive() {
        cullingActive = false
        applyEmissiveSet(tris: fullEmisTris, cdf: fullEmisCDF,
                         total: fullTotalEmissiveArea)
    }

    /// Re-activate the stored culled set (used around stills).
    private func activateCulledEmissive() {
        cullingActive = true
        applyEmissiveSet(tris: culledEmisTris, cdf: culledEmisCDF,
                         total: culledTotalEmissiveArea)
    }

    /// Compute the culled CDF for a camera: keep every emissive
    /// tri (static + dynamic, via the unified full set) whose
    /// centroid is within 7 m of the camera OR inside the
    /// expanded frustum (angle to the view forward < 75°).
    private func computeCulledSet(camera: CameraBasis)
        -> (tris: [UInt32], cdf: [Float], total: Float) {
        var tris: [UInt32] = []
        var cdf: [Float] = []
        var acc: Float = 0
        let fwd = simd_normalize(camera.forward)
        let cosLimit: Float = 0.258819 // cos(75°)
        for i in 0..<fullEmisTris.count {
            let c = fullEmisCentroids[i]
            let rel = c - camera.origin
            let dist = simd_length(rel)
            var keep = dist <= 7
            if !keep, dist > 1e-5 {
                keep = simd_dot(rel / dist, fwd) > cosLimit
            }
            if keep {
                tris.append(fullEmisTris[i])
                acc += fullEmisAreas[i]
                cdf.append(acc)
            }
        }
        return (tris, cdf, acc)
    }

    private func makeBuffer<T>(_ array: [T]) -> MTLBuffer? {
        if array.isEmpty { return device.makeBuffer(length: 4, options: .storageModeShared) }
        return array.withUnsafeBytes { raw in
            // v4.1 (F24): no force-unwrap — a nil base address
            // degrades to no buffer instead of a crash.
            guard let base = raw.baseAddress else { return nil }
            return device.makeBuffer(bytes: base, length: raw.count, options: .storageModeShared)
        }
    }

    func gpuMaterialForObject(_ obj: SceneObject) -> GPUMaterial {
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
        let texIdx = texIndexValue(forPreset: m.texturePreset)
        // v4.1 (F3): glassParams.w carries the material's slice
        // MEAN texture roughness (0 when untextured) for the
        // guide kernel's gloss estimate. Verified unused by
        // every kernel before this (only .x/.y/.z are read).
        g.glassParams = SIMD4<Float>(Float(m.transmission), Float(m.roughness),
                                     Float(m.ior), texMeanRoughness[texIdx] ?? 0)
        g.tint = SIMD4<Float>(m.tint.simd, Float(m.textureScale))
        g.texIndex = texIdx
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
                // v4.1 (F6): sheets are charged against the
                // same 220k weather budget as the subdivision
                // that spawned them; when it is exhausted no
                // further sheets are emitted.
                guard budget > 0 else { return }
                budget -= 1
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

    // MARK: - Dynamic layer, areas, light-culling support (v4)
    //
    // The dynamic layer is a second world-space triangle soup
    // beside the static one: dynamic entries (keyed, matrix-
    // driven) plus up to 8 particle soups, composed into CPU
    // mirrors + GPU buffers (slots 11-17) + its own AS. The
    // material table is contractual everywhere:
    //   [static object materials…, sheetMaterial if emitted,
    //    dynamic entry materials in entry order,
    //    particle soup materials in soup order]
    // Dynamic/particle materials are renderer state — static
    // rebuilds and material-only updates APPEND them, so they
    // survive static rebuilds. Total dynamic budget: 90,000
    // triangles; entries/soups are added in order and the first
    // one that would overflow the cap is truncated to the
    // remaining budget (the rest of it dropped), after which
    // nothing further is added.

    /// The object's area id (SceneObject.areaID, v4).
    private func objectAreaID(_ obj: SceneObject) -> UUID? {
        obj.areaID
    }

    private func isUnloadedArea(_ obj: SceneObject) -> Bool {
        guard let aid = objectAreaID(obj) else { return false }
        return unloadedAreaIDs.contains(aid)
    }

    /// Index of the first dynamic material in the full table.
    private var dynMatBase: Int {
        cachedStaticMaterials.count + (sceneHasSheetMaterial ? 1 : 0)
    }

    /// The full GPU material table, in the contractual order.
    private func fullMaterialTable() -> [GPUMaterial] {
        var mats = cachedStaticMaterials
        if sceneHasSheetMaterial { mats.append(sheetMaterial()) }
        for entry in dynEntries { mats.append(entry.material) }
        for soup in particleSoupsStored { mats.append(soup.material) }
        return mats
    }

    /// Recompute cpuDynTriMats (+ its GPU buffer) from the
    /// stored per-tri material offsets and the current base.
    private func refreshDynTriMats() {
        let base = dynMatBase
        cpuDynTriMats = cpuDynMatOffsets.map { UInt32(base + $0) }
        dynTriMatBuffer = makeBuffer(cpuDynTriMats)
        // v4.1 (N2): this reassigns dynTriMatBuffer outside the
        // AS build — a nil buffer here (makeBuffer failure) must
        // not leave dynUsable stale-true. Recompute it with the
        // same predicate the AS build uses (refreshDynUsable).
        refreshDynUsable()
    }

    private func computeCollisionTriangles()
        -> [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] {
        var out: [(SIMD3<Float>, SIMD3<Float>, SIMD3<Float>)] = []
        out.reserveCapacity(max(triCount, 0) + max(dynTriCount, 0))
        if triCount > 0 {
            for t in 0..<triCount {
                let ia = Int(cpuIndices[3 * t]), ib = Int(cpuIndices[3 * t + 1]), ic = Int(cpuIndices[3 * t + 2])
                out.append((
                    SIMD3<Float>(cpuPositions[3 * ia], cpuPositions[3 * ia + 1], cpuPositions[3 * ia + 2]),
                    SIMD3<Float>(cpuPositions[3 * ib], cpuPositions[3 * ib + 1], cpuPositions[3 * ib + 2]),
                    SIMD3<Float>(cpuPositions[3 * ic], cpuPositions[3 * ic + 1], cpuPositions[3 * ic + 2])))
            }
        }
        // v4.1 (F5): the dynamic layer collides too — its CPU
        // mirrors are already world-space as rendered.
        if dynTriCount > 0 {
            for t in 0..<dynTriCount {
                let ia = Int(cpuDynIndices[3 * t]), ib = Int(cpuDynIndices[3 * t + 1]), ic = Int(cpuDynIndices[3 * t + 2])
                out.append((
                    SIMD3<Float>(cpuDynPositions[3 * ia], cpuDynPositions[3 * ia + 1], cpuDynPositions[3 * ia + 2]),
                    SIMD3<Float>(cpuDynPositions[3 * ib], cpuDynPositions[3 * ib + 1], cpuDynPositions[3 * ib + 2]),
                    SIMD3<Float>(cpuDynPositions[3 * ic], cpuDynPositions[3 * ic + 1], cpuDynPositions[3 * ic + 2])))
            }
        }
        return out
    }

    /// Full replace of the dynamic entries (ordered).
    func setDynamicEntries(_ entries: [DynEntry]) {
        let structureChanged = entries.map { $0.key } != dynEntries.map { $0.key }
        dynEntries = entries
        dynDirtyKeys.removeAll()
        composeDynamic(structureChanged: structureChanged)
    }

    /// Cheap matrix update: marks the entry dirty; the re-bake
    /// happens in flushDynamic() at the top of draw(). Does NOT
    /// reset accumulation (reprojection rejects stale history
    /// by position, as with camera moves).
    func updateDynamicMatrix(key: UUID, matrix: float4x4) {
        guard let idx = dynEntries.firstIndex(where: { $0.key == key }) else { return }
        dynEntries[idx].matrix = matrix
        dynDirtyKeys.insert(key)
    }

    func setDynamicMaterial(key: UUID, material: GPUMaterial) {
        guard let idx = dynEntries.firstIndex(where: { $0.key == key }) else { return }
        dynEntries[idx].material = material
        let fullMats = fullMaterialTable()
        matBuffer = makeBuffer(fullMats)
        // v4.1 (F4ii): only a dynamic material changed — reuse
        // the cached static emissive scan.
        uploadEmissiveIndex(materials: fullMats, rescanStatic: false)
        resetAccumulation()
    }

    func removeDynamicEntry(key: UUID) {
        guard dynEntries.contains(where: { $0.key == key }) else { return }
        dynEntries.removeAll { $0.key == key }
        dynDirtyKeys.remove(key)
        composeDynamic(structureChanged: true)
    }

    /// Particle soups (at most 8 kept), appended after the
    /// dynamic entries in both geometry and the material table.
    /// May be called per frame: when neither the soup count nor
    /// the total dynamic tri count changes, compose does not
    /// reset accumulation.
    func setParticleSoups(_ soups: [(mesh: MeshData, material: GPUMaterial)]) {
        let capped = Array(soups.prefix(8))
        var structureChanged = capped.count != particleSoupsStored.count
        // v4.1 (N4): count alone misses a material-only change —
        // a soup at the same index whose emissive signature
        // (flags bit0 + the full emissive SIMD4, compared
        // component-wise; GPUMaterial is not Equatable) changed
        // must also reset accumulation, or settled frames keep
        // converging against the old emissive set.
        if !structureChanged {
            for i in capped.indices {
                let oldM = particleSoupsStored[i].material
                let newM = capped[i].material
                if (oldM.flags & 1) != (newM.flags & 1)
                    || oldM.emissive.x != newM.emissive.x
                    || oldM.emissive.y != newM.emissive.y
                    || oldM.emissive.z != newM.emissive.z
                    || oldM.emissive.w != newM.emissive.w {
                    structureChanged = true
                    break
                }
            }
        }
        particleSoupsStored = capped
        composeDynamic(structureChanged: structureChanged)
    }

    /// Re-bake every dynamic entry + particle soup into the CPU
    /// mirrors, rebuild the dynamic GPU buffers and AS, refresh
    /// the material table and the unified emissive index.
    /// Resets accumulation once when the composition changed
    /// structurally (entries/soups added/removed) or the total
    /// dynamic tri count changed.
    private func composeDynamic(structureChanged: Bool) {
        let oldDynTriCount = dynTriCount
        let budget = 90_000
        var positions: [Float] = []
        var normals: [Float] = []
        var uvArr: [Float] = []
        var colArr: [Float] = []
        var indices: [UInt32] = []
        var matOffsets: [Int] = []
        var triObjects: [UUID] = []
        var slices: [UUID: DynSlice] = [:]
        var totalTris = 0
        var capped = false

        func appendSoup(world: MeshData, matOffset: Int, objectID: UUID?) -> DynSlice {
            let available = max(budget - totalTris, 0)
            let included = min(world.triangleCount, available)
            var slice = DynSlice(vertexStart: positions.count / 3, vertexCount: 0,
                                 triStart: totalTris, triCount: included)
            if included < world.triangleCount {
                // v4.1 (F17): truncated soup — append ONLY the
                // vertices the included triangles reference
                // (remapped), instead of the full vertex payload.
                // (flushDynamic's vertexCount check then skips
                // matrix re-bakes for this slice: acceptable for
                // an over-budget soup, which is the pathological
                // case only.)
                var remap: [UInt32: UInt32] = [:]
                remap.reserveCapacity(included * 3)
                func mapVertex(_ vi: UInt32) -> UInt32 {
                    if let r = remap[vi] { return r }
                    let r = UInt32(positions.count / 3)
                    let i3 = Int(vi) * 3, i2 = Int(vi) * 2
                    positions.append(contentsOf: [world.positions[i3], world.positions[i3 + 1], world.positions[i3 + 2]])
                    normals.append(contentsOf: [world.normals[i3], world.normals[i3 + 1], world.normals[i3 + 2]])
                    uvArr.append(contentsOf: [world.uvs[i2], world.uvs[i2 + 1]])
                    colArr.append(contentsOf: [world.colors[i3], world.colors[i3 + 1], world.colors[i3 + 2]])
                    remap[vi] = r
                    return r
                }
                for t in 0..<included {
                    let a = mapVertex(world.indices[3 * t])
                    let b = mapVertex(world.indices[3 * t + 1])
                    let c = mapVertex(world.indices[3 * t + 2])
                    indices += [a, b, c]
                    matOffsets.append(matOffset)
                    triObjects.append(objectID ?? dynNilObjectSentinel)
                }
                slice.vertexCount = positions.count / 3 - slice.vertexStart
                slice.remapped = true
            } else {
                let vBase = UInt32(positions.count / 3)
                positions += world.positions
                normals += world.normals
                uvArr += world.uvs
                colArr += world.colors
                slice.vertexCount = world.vertexCount
                for t in 0..<included {
                    let a = world.indices[3 * t], b = world.indices[3 * t + 1], c = world.indices[3 * t + 2]
                    indices += [vBase + a, vBase + b, vBase + c]
                    matOffsets.append(matOffset)
                    triObjects.append(objectID ?? dynNilObjectSentinel)
                }
            }
            totalTris += included
            if included < world.triangleCount { capped = true }
            return slice
        }

        for (i, pair) in dynEntries.enumerated() {
            if capped { break }
            let world = pair.local.baked(by: pair.matrix)
            slices[pair.key] = appendSoup(world: world, matOffset: i,
                                          objectID: pair.objectID)
        }
        for (j, soup) in particleSoupsStored.enumerated() {
            if capped { break }
            _ = appendSoup(world: soup.mesh,
                           matOffset: dynEntries.count + j, objectID: nil)
        }

        cpuDynPositions = positions
        cpuDynNormals = normals
        cpuDynUVs = uvArr
        cpuDynColors = colArr
        cpuDynIndices = indices
        cpuDynMatOffsets = matOffsets
        cpuDynTriObject = triObjects
        dynSlices = slices
        dynTriCount = totalTris
        // v4.1 (F5): the collision export tracks the dynamic
        // layer's composition too.
        _collisionTriangles = computeCollisionTriangles()

        dynPosBuffer = makeBuffer(positions)
        dynNrmBuffer = makeBuffer(normals)
        dynUVBuffer = makeBuffer(uvArr)
        dynColBuffer = makeBuffer(colArr)
        dynIdxBuffer = makeBuffer(indices)
        refreshDynTriMats()
        buildDynamicAccelerationStructure()
        let fullMats = fullMaterialTable()
        matBuffer = makeBuffer(fullMats)
        // v4.1 (F4ii): compose changes only dynamic content —
        // reuse the cached static emissive scan.
        uploadEmissiveIndex(materials: fullMats, rescanStatic: false)
        if structureChanged || totalTris != oldDynTriCount {
            resetAccumulation()
        }
    }

    /// Re-bake dirty entries' vertices (matrix updates only)
    /// into the dynamic mirrors, re-upload the position/normal
    /// buffers and rebuild the dynamic AS. Runs at the top of
    /// draw(); never resets accumulation.
    private func flushDynamic() {
        guard !dynDirtyKeys.isEmpty else { return }
        let dirty = dynDirtyKeys
        dynDirtyKeys.removeAll()
        var anyBaked = false
        for key in dirty {
            guard let pair = dynEntries.first(where: { $0.key == key }),
                  let slice = dynSlices[key], slice.vertexCount > 0 else { continue }
            let world = pair.local.baked(by: pair.matrix)
            guard world.vertexCount == slice.vertexCount, !slice.remapped else { continue }
            let start = slice.vertexStart * 3
            let count = slice.vertexCount * 3
            if start + count <= cpuDynPositions.count {
                cpuDynPositions.replaceSubrange(start..<(start + count), with: world.positions)
                cpuDynNormals.replaceSubrange(start..<(start + count), with: world.normals)
                anyBaked = true
            }
            // An emissive entry moved: refresh the full-set
            // centroids of its tris in place (areas are unchanged
            // by the matrix in the common rigid case; the CDF —
            // and therefore the GPU buffers — stay as they are).
            if (pair.material.flags & 1) != 0, slice.triCount > 0 {
                for dt in slice.triStart..<(slice.triStart + slice.triCount) {
                    let unified = UInt32(triCount + dt)
                    // v4.1 (F11): O(1) map lookup, not a linear
                    // scan of the full emissive list per tri.
                    guard let fi = fullEmisIndexByID[unified],
                          fi < fullEmisCentroids.count else { continue }
                    let a = cpuDynIndices[3 * dt], b = cpuDynIndices[3 * dt + 1], c = cpuDynIndices[3 * dt + 2]
                    let p0 = SIMD3<Float>(cpuDynPositions[3 * Int(a)], cpuDynPositions[3 * Int(a) + 1], cpuDynPositions[3 * Int(a) + 2])
                    let p1 = SIMD3<Float>(cpuDynPositions[3 * Int(b)], cpuDynPositions[3 * Int(b) + 1], cpuDynPositions[3 * Int(b) + 2])
                    let p2 = SIMD3<Float>(cpuDynPositions[3 * Int(c)], cpuDynPositions[3 * Int(c) + 1], cpuDynPositions[3 * Int(c) + 2])
                    fullEmisCentroids[fi] = (p0 + p1 + p2) / 3
                }
            }
        }
        guard anyBaked else { return }
        dynPosBuffer = makeBuffer(cpuDynPositions)
        dynNrmBuffer = makeBuffer(cpuDynNormals)
        buildDynamicAccelerationStructure()
        // v4.1 (F5): moved dynamic geometry moves its collision
        // triangles with it.
        _collisionTriangles = computeCollisionTriangles()
        // Keep an active culled set honest about moved lights —
        // silently (no accumulation reset for matrix updates,
        // and no buffer swap when the culled set is unchanged).
        if cullingActive, let lc = lastCullCamera {
            let culled = computeCulledSet(camera: lc)
            if culled.tris != culledEmisTris {
                culledEmisTris = culled.tris
                culledEmisCDF = culled.cdf
                culledTotalEmissiveArea = culled.total
                activateCulledEmissive()
            }
        }
    }

    /// v4.1 (F9): dynUsable = dynamic AS built AND all dynamic
    /// buffers present. Recomputed at the end of every dynamic
    /// AS build (the only place dynAccel changes) and at the end
    /// of refreshDynTriMats (v4.1 N2: that path reassigns
    /// dynTriMatBuffer outside the AS build).
    private func refreshDynUsable() {
        dynUsable = dynTriCount > 0 && dynAccel != nil
            && dynPosBuffer != nil && dynNrmBuffer != nil
            && dynUVBuffer != nil && dynColBuffer != nil
            && dynIdxBuffer != nil && dynTriMatBuffer != nil
    }

    private func buildDynamicAccelerationStructure() {
        dynAccel = nil
        dynUsable = false
        guard rayTracingSupported, dynTriCount > 0,
              let dynPosBuffer, let dynIdxBuffer else { return }
        let geom = MTLAccelerationStructureTriangleGeometryDescriptor()
        geom.vertexBuffer = dynPosBuffer
        geom.vertexStride = 12
        geom.vertexFormat = .float3
        geom.indexBuffer = dynIdxBuffer
        geom.indexType = .uint32
        geom.triangleCount = dynTriCount
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
        dynAccel = structure
        refreshDynUsable()
    }

    private func dynamicDummyBuffer() -> MTLBuffer? {
        if let d = dynDummyBuffer { return d }
        // v4.1 (F24): tolerate a nil allocation — bindings take
        // optionals, and F9's dynUsable keeps the kernels safe.
        let b = device.makeBuffer(length: 4, options: .storageModeShared)
        dynDummyBuffer = b
        return b
    }

    /// Bind the dynamic layer at slots 11-17 for a compute
    /// encoder. With no dynamic triangles, slots 11-16 get a
    /// dummy 4-byte buffer and slot 17 gets the static AS (the
    /// kernels skip it via dynTriCount == 0, but the slot must
    /// still be bound).
    private func bindDynamicBuffers(_ enc: MTLComputeCommandEncoder) {
        let dummy = dynamicDummyBuffer()
        // v4.1 (F9): only bind the real dynamic set when it is
        // usable; otherwise the dummy/static bindings pair with
        // the forced dynTriCount = 0 in the uniforms.
        if dynUsable {
            enc.setBuffer(dynPosBuffer ?? dummy, offset: 0, index: 11)
            enc.setBuffer(dynNrmBuffer ?? dummy, offset: 0, index: 12)
            enc.setBuffer(dynUVBuffer ?? dummy, offset: 0, index: 13)
            enc.setBuffer(dynColBuffer ?? dummy, offset: 0, index: 14)
            enc.setBuffer(dynIdxBuffer ?? dummy, offset: 0, index: 15)
            enc.setBuffer(dynTriMatBuffer ?? dummy, offset: 0, index: 16)
            if let dynAccel {
                enc.setAccelerationStructure(dynAccel, bufferIndex: 17)
            } else if let accel {
                enc.setAccelerationStructure(accel, bufferIndex: 17)
            }
        } else {
            for i in 11...16 { enc.setBuffer(dummy, offset: 0, index: i) }
            if let accel {
                enc.setAccelerationStructure(accel, bufferIndex: 17)
            }
        }
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
        u.envIntensity = envIntensity
        u.historyCap = cap
        u.glassRefractBudget = glassBudget
        u.wetLevel = weatherWet
        u.snowLevel = weatherSnow
        u.dynTime = weatherDynTime
        u.rainFrac = weatherDynRain
        // v4.1 (F9): an unusable dynamic layer reports 0 tris
        // to BOTH kernels, so a failed AS/buffer can never be
        // traversed against the dynamic attribute arrays.
        u.dynTriCount = UInt32(dynUsable ? dynTriCount : 0)
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
              let weatherFieldTex,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(guidePipeline)
        enc.setTexture(guide, index: 0)
        enc.setTexture(guideNormal, index: 1)
        // v4.1 (F7): the guide samples the baked weather field
        // (same slot as the trace kernel) for its gloss model.
        enc.setTexture(weatherFieldTex, index: 4)
        enc.setBuffer(posBuffer, offset: 0, index: 1)
        enc.setBuffer(nrmBuffer, offset: 0, index: 2)
        enc.setBuffer(idxBuffer, offset: 0, index: 5)
        enc.setBuffer(triMatBuffer, offset: 0, index: 6)
        enc.setBuffer(matBuffer, offset: 0, index: 7)
        enc.setAccelerationStructure(accel, bufferIndex: 10)
        bindDynamicBuffers(enc)
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
        // v4.1 (N1): the road arrays are NOT a gate on tracing —
        // a missing road array binds the dummy array instead.
        // The dummy is never sampled: roadsAvailable is
        // all-or-nothing, so any road-array failure makes
        // texIndexValue return 0 for every road preset, and no
        // material encodes >= 4096.
        let roadAlbedoBind = roadTexAlbedo ?? roadDummyTexture()
        let roadNormalBind = roadTexNormal ?? roadDummyTexture()
        let roadDataBind = roadTexData ?? roadDummyTexture()
        guard let tracePipeline, let accel,
              let posBuffer, let nrmBuffer, let uvBuffer, let colBuffer,
              let idxBuffer, let triMatBuffer, let matBuffer,
              let emisTriBuffer, let emisCDFBuffer,
              let texAlbedo, let texNormal, let texData,
              let roadAlbedoBind, let roadNormalBind, let roadDataBind,
              let weatherFieldTex,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(tracePipeline)
        enc.setTexture(accum, index: 0)
        enc.setTexture(texAlbedo, index: 1)
        enc.setTexture(texNormal, index: 2)
        enc.setTexture(texData, index: 3)
        enc.setTexture(weatherFieldTex, index: 4)
        // v4.1: road arrays (1024) at texture indices 5-7
        // (dummy array when a road array failed to allocate).
        enc.setTexture(roadAlbedoBind, index: 5)
        enc.setTexture(roadNormalBind, index: 6)
        enc.setTexture(roadDataBind, index: 7)
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
        bindDynamicBuffers(enc)
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

    /// Area-box overlay (v4, edit mode only): expands
    /// debugBoxes into a line list (24 vertices / 12 edges per
    /// box) with per-vertex colours and draws it over the
    /// presented drawable in its own .load pass, like precip.
    private func encodeLineOverlay(cmd: MTLCommandBuffer, drawable: CAMetalDrawable,
                                   camera: CameraBasis, view: MTKView) {
        guard !debugBoxes.isEmpty, let linePipeline else { return }
        var verts: [SIMD4<Float>] = []
        var cols: [SIMD4<Float>] = []
        verts.reserveCapacity(debugBoxes.count * 24)
        cols.reserveCapacity(debugBoxes.count * 24)
        let edges: [(Int, Int)] = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                                   (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
        for box in debugBoxes {
            var corners: [SIMD3<Float>] = []
            for i in 0..<8 {
                corners.append(SIMD3<Float>(
                    (i & 1) == 0 ? box.min.x : box.max.x,
                    (i & 2) == 0 ? box.min.y : box.max.y,
                    (i & 4) == 0 ? box.min.z : box.max.z))
            }
            for (a, b) in edges {
                for c in [corners[a], corners[b]] {
                    verts.append(SIMD4<Float>(c, 1))
                    cols.append(box.color)
                }
            }
        }
        guard let vBuf = makeBuffer(verts), let cBuf = makeBuffer(cols) else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = drawable.texture
        pass.colorAttachments[0].loadAction = .load
        pass.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        enc.setRenderPipelineState(linePipeline)
        var lu = LineUniforms()
        lu.camOrigin = SIMD4<Float>(camera.origin, 0)
        lu.camRight = SIMD4<Float>(camera.right, 0)
        lu.camUp = SIMD4<Float>(camera.up, 0)
        lu.camForward = SIMD4<Float>(camera.forward, 0)
        lu.tanFovAspect = SIMD2<Float>(
            tan(fovY / 2),
            Float(view.drawableSize.width) / Float(max(view.drawableSize.height, 1)))
        enc.setVertexBytes(&lu, length: MemoryLayout<LineUniforms>.stride, index: 0)
        enc.setVertexBuffer(vBuf, offset: 0, index: 1)
        enc.setVertexBuffer(cBuf, offset: 0, index: 2)
        enc.drawPrimitives(type: .line, vertexStart: 0, vertexCount: verts.count)
        enc.endEncoding()
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
        flushDynamic()
        // v4.1 (F12): denoise "recently moved" bookkeeping.
        if moving {
            framesSinceMove = 0
        } else if framesSinceMove < 10_000 {
            framesSinceMove += 1
        }
        presentationDirty = false
        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer(),
              view.drawableSize.width > 0, view.drawableSize.height > 0
        else { return false }

        // v4 light culling: while MOVING, NEE samples only the
        // emissive tris near/ahead of the camera. This changes
        // only WHICH lights are sampled during interaction —
        // when motion stops the full set is restored and settled
        // frames converge with every light. v4.1 (F2): Play is
        // NOT a culling condition of its own — a still Play
        // camera restores the full set like any other settle.
        // v4.1 (F1): a re-cull resets accumulation ONLY when the
        // culled tri set actually changes, and there is no
        // pure-time re-cull trigger anymore (the camera-move
        // triggers below are the only re-cull sources).
        let wantCull = lightCullingEnabled && moving
        if wantCull {
            let now = CACurrentMediaTime()
            var needCull = !cullingActive
            if let lc = lastCullCamera {
                if simd_length(camera.origin - lc.origin) > 1.5 { needCull = true }
                let cosA = simd_dot(simd_normalize(camera.forward),
                                    simd_normalize(lc.forward))
                if cosA < 0.9781476 { needCull = true } // rotated > ~12°
            } else {
                needCull = true
            }
            if needCull {
                lastCullCamera = camera
                lastCullTime = now
                if !fullEmisTris.isEmpty {
                    let culled = computeCulledSet(camera: camera)
                    let activeTris = cullingActive ? culledEmisTris : fullEmisTris
                    let changed = culled.tris != activeTris
                    culledEmisTris = culled.tris
                    culledEmisCDF = culled.cdf
                    culledTotalEmissiveArea = culled.total
                    activateCulledEmissive()
                    if changed { resetAccumulation() }
                }
            }
        } else if cullingActive {
            restoreFullEmissive()
            resetAccumulation()
        }

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
            if !play {
                encodeLineOverlay(cmd: cmd, drawable: drawable, camera: camera, view: view)
            }
            // Settled: dynamic resolution stands down — the next
            // motion starts from the tier's full motion scale.
            dynLock.lock()
            dynScale = 1.0; dynCadenceEMA = 0; dynLastCompletion = 0
            dynFramesSinceAdjust = 0; dynStillSince = 0
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
                    // v4.1 (F8): a failed encode is NOT
                    // convergence — leave sampleCount untouched,
                    // keep the frame wanted so the failure is
                    // retried instead of sleeping on a black
                    // image, and log it.
                    print("LumenScene: encodeTrace failed — frame kept wanted for retry")
                    presentationDirty = true
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
        // v4.1 (F12): the gate also stays open for ~45 frames
        // after motion stops — per-pixel alpha (the shader's
        // own fade) is the real signal, and newly exposed young
        // pixels need the pass even when the GLOBAL sampleCount
        // is already high.
        if accel != nil, moving || play || sampleCount < 160 || framesSinceMove < 45,
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

        // Area-box overlay (edit mode only), after precip.
        if !play {
            encodeLineOverlay(cmd: cmd, drawable: drawable, camera: camera, view: view)
        }

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
                    self.dynStillSince = 0
                    if self.dynLastCompletion > 0 {
                        let dt = now - self.dynLastCompletion
                        // v4.1 (F22): hitch frames (completion
                        // delta > 40 ms) do NOT feed the EMA —
                        // they are stalls, not cadence.
                        if dt > 0.004, dt <= 0.040 {
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
                    // v4.1 (F22): stand-down timer — ~2 s of
                    // not-moving returns dynScale to 1.0 even
                    // when the settled fast path was never
                    // reached (stop-and-go motion).
                    if self.dynStillSince == 0 {
                        self.dynStillSince = now
                    } else if now - self.dynStillSince >= 2.0, self.dynScale != 1.0 {
                        self.dynScale = 1.0
                        self.dynCadenceEMA = 0
                        self.dynFramesSinceAdjust = 0
                    }
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
        // Stills always use the FULL emissive set: force-restore
        // it around the trace, then put the culled set back (the
        // command buffer is waited on below before any swap-back
        // could race the GPU reads). v4.1 (F13): the restore is
        // a defer, so EVERY exit path — target failure, encoder
        // failure, traced == false, success — leaves the culling
        // state consistent. On the success path the defer runs
        // at return, i.e. after waitUntilCompleted, as before;
        // on failure paths the command buffer was never
        // committed, so no GPU work can observe the swap.
        let wasCulling = cullingActive
        if wasCulling { restoreFullEmissive() }
        defer { if wasCulling { activateCulledEmissive() } }
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
        // v4.1 (N5): no force-unwrap — a nil base address (zero-
        // size request) returns nil, in the F24 style.
        let gotBytes: Bool = bytes.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress else { return false }
            target.getBytes(base, bytesPerRow: width * 4,
                            from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
            return true
        }
        guard gotBytes else { return nil }
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
        guard cpuIndices.count >= 3 || cpuDynIndices.count >= 3 else { return nil }
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
        // v4: dynamic mirrors too — nearest hit overall wins. A
        // dynamic hit with the sentinel id is unpickable: it
        // still occludes (best is set) and returns nil.
        let dynTotal = cpuDynIndices.count / 3
        for t in 0..<dynTotal {
            let ia = Int(cpuDynIndices[3 * t]), ib = Int(cpuDynIndices[3 * t + 1]), ic = Int(cpuDynIndices[3 * t + 2])
            let p0 = SIMD3<Float>(cpuDynPositions[3 * ia], cpuDynPositions[3 * ia + 1], cpuDynPositions[3 * ia + 2])
            let p1 = SIMD3<Float>(cpuDynPositions[3 * ib], cpuDynPositions[3 * ib + 1], cpuDynPositions[3 * ib + 2])
            let p2 = SIMD3<Float>(cpuDynPositions[3 * ic], cpuDynPositions[3 * ic + 1], cpuDynPositions[3 * ic + 2])
            if let d = rayTriangle(origin: origin, dir: dir, p0: p0, p1: p1, p2: p2), d < best {
                best = d
                let oid = cpuDynTriObject[t]
                bestObj = (oid == dynNilObjectSentinel) ? nil : oid
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

/// Top-level aliases so callers can write the entry type
/// unqualified under either name; the struct itself is nested
/// in Renderer (Renderer.DynEntry). `DynamicEntry` is the name
/// the editor session uses for its entry mirror.
typealias DynEntry = Renderer.DynEntry
typealias DynamicEntry = Renderer.DynEntry

extension Renderer.DynEntry {
    /// The entry's local-space mesh (alias of the stored
    /// `mesh`), for callers written against the `local` name.
    var local: MeshData {
        get { mesh }
        set { mesh = newValue }
    }

    /// Construct without an explicit key (a fresh key is
    /// minted); the entry is then anonymous to the key-based
    /// update/remove calls unless the caller reads `key` back.
    init(local: MeshData, matrix: float4x4, material: GPUMaterial, objectID: UUID?) {
        self.init(key: UUID(), mesh: local, matrix: matrix,
                  material: material, objectID: objectID)
    }

    /// Construct with an explicit key and the `local` name.
    init(key: UUID, local: MeshData, matrix: float4x4,
         material: GPUMaterial, objectID: UUID?) {
        self.init(key: key, mesh: local, matrix: matrix,
                  material: material, objectID: objectID)
    }
}
