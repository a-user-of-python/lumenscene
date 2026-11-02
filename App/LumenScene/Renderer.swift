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
    var hidePlayerHead: UInt32 = 0  // v4.5: Play-eye primary-ray head exclusion (was padDyn; layout unchanged)
    var freezeFrac: Float = 0  // v4.7: puddle freeze fraction 0...1 (tail field; mirrors Shaders.metal Uniforms)
    var waterReflScale: Float = 1  // v4.8: water reflection resolution (tail; mirrors Uniforms)
    var frameCounter: UInt32 = 0   // v4.8: rotating reflection pattern (tail; mirrors Uniforms)
    var reflPad: UInt32 = 0
}

/// v4.8: parameters for the weather simulation kernels (mirrors
/// SimParams in Shaders.metal byte-for-byte).

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
    // v4.2: per-class denoise hand-off (effective samples where
    // the blend reaches zero) — reusable/diffuse class, then
    // the non-reusable (glass/reflection) class. Same offsets
    // as the former pad0/pad1; the Metal mirror matches.
    var fadeEnd: Float = 0
    var fadeEndSpec: Float = 0
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

/// v4.3: parameters for the weather-field bake kernel — the
/// seed-derived field offset (metres), the wind heading as a
/// literal (cos, sin) pair, and whether a non-zero seed is in
/// play (seed 0 = the legacy v4.2 field, bit-exact). Mirror of
/// WeatherFieldParams in Shaders.metal.

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

/// v4.3: the reserved dynamic-layer key of the player figure
/// (see setPlayerFigure). Fixed so the figure's slice, dirty
/// tracking and material slot are stable across pose swaps.
private let playerFigureKey = UUID(uuidString: "00000000-0000-0000-0000-0000000000F1")!

/// v4.5: the reserved key of the figure's HEAD entry (head +
/// hair, split from the body by PlayerModel.splitHead) — its
/// material carries FLAG_PLAYER_HEAD (64) so the trace/guide
/// kernels can exclude it from primary hits while the camera
/// is the Play first-person eye.
private let playerHeadKey = UUID(uuidString: "00000000-0000-0000-0000-0000000000F2")!

/// v4.5: GPUMaterial flag for the player head entry (mirrors
/// FLAG_PLAYER_HEAD in Shaders.metal).
private let flagPlayerHead: UInt32 = 64

/// v4.3: one player footprint stamp in snow (world XZ,
/// metres) — the struct is Renderer.SnowFootprint (nested,
/// like DynEntry); a top-level typealias at the bottom of
/// Renderer.swift keeps the unqualified name working too.

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

    /// v4.3: one player footprint stamp in snow (world XZ,
    /// metres). The session appends these as the player walks;
    /// the weather-geometry pass compresses the snowfield
    /// under each stamp deterministically at rebuild time —
    /// nothing is stored in the scene itself.

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
    /// v4.10 (WS2 steering): the seed field in CPU form (the
    /// same deterministic LCG array the GPU buffer holds), and
    /// a snapshot of the LAST precipitation pass actually
    /// encoded — time / intensity / wind / draw count as the
    /// user last saw them. The final render rebuilds its rain
    /// drops from exactly this snapshot, so the still's drops
    /// sit where the frozen live view showed them.
    private var precipSeeds: [SIMD4<Float>] = []
    private var lastPrecipTime: Float? = nil
    private var lastPrecipIntensity: Float = 1
    private var lastPrecipWind = SIMD2<Float>(1.2, 0.4)
    private var lastPrecipCount: Int = 0
    private var linePipeline: MTLRenderPipelineState?
    /// v4.13: the baked weather field is gone with the snow
    /// and the 3D water; the trace/guide/tonemap kernels keep
    /// their weatherField texture slot (binding layout is
    /// contractual), bound to this inert 1x1 texture and
    /// never sampled.
    private var weatherFieldTex: MTLTexture?
    /// Baked weather field (v3.3): level-independent, baked
    /// once at init — see bakeWeatherField().

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

    // MARK: - v4.9 optimisation state
    //
    // (1b) Static-build staging: rebuild() stages the complete
    // new static state in a PreparedStatic and builds its AS
    // asynchronously; the live properties above swap atomically
    // at a frame boundary once the build completes, so a frame
    // never mixes an old AS with new buffers or vice versa.
    private struct PreparedStatic {
        var cpuPositions: [Float] = []
        var cpuIndices: [UInt32] = []
        var cpuTriObject: [UUID] = []
        var cpuTriMats: [UInt32] = []
        var objectOrder: [UUID] = []
        var materials: [GPUMaterial] = []
        var sheet: Bool = false
        var snow: Bool = false
        var triCount: Int = 0
        var posBuffer: MTLBuffer?
        var nrmBuffer: MTLBuffer?
        var uvBuffer: MTLBuffer?
        var colBuffer: MTLBuffer?
        var idxBuffer: MTLBuffer?
        var triMatBuffer: MTLBuffer?
        var accel: MTLAccelerationStructure?
        var cmd: MTLCommandBuffer?
        /// v4.10.2: the AS build could not be encoded
        /// (allocation/encoder failure on a non-empty scene).
        /// A failed stage must never be swapped in — see
        /// staticStageFailed.
        var buildFailed = false
    }
    private var pendingStatic: PreparedStatic?
    private var deferredStatic: PreparedStatic?
    /// v4.10 (WS1): read-only query for the session's idle-freeze
    /// protected-operations check — a static build is in flight
    /// while a stage is pending (building) or deferred behind it.
    var staticBuildInFlight: Bool { pendingStatic != nil || deferredStatic != nil }
    private var staticEverSwapped = false
    /// The last presented frame's command buffer: the bounded
    /// fence for CPU writes into shared buffers (see
    /// waitForPreviousFrame).
    private var lastFrameCommandBuffer: MTLCommandBuffer?

    // (5) Material table: one persistent shared buffer, written
    // per changed slot at a single flush point (top of draw).
    private var matBufferCapacity = 0 // in GPUMaterial slots
    private var matBufferUploaded: [GPUMaterial] = []
    private var matPendingTable: [GPUMaterial] = []
    private var matTableDirty = false
    /// Emissive-index gating: signature of the emissive-
    /// relevant material fields at the last uploadEmissiveIndex.
    private var lastEmissiveSignature: UInt64 = 0
    private var lastEmissiveSignatureValid = false

    // (1c) Sheet-only scoped rebuild: fingerprint of everything
    // except the sheet segment + the cached static prefix.
    private struct StaticFingerprint: Equatable {
        var ids: [UUID] = []
        var baked: [UInt64] = []
        var mats: [UInt64] = []
        var xf: [UInt64] = []
    }
    private struct PrefixCache {
        var fingerprint: StaticFingerprint
        var positions: [Float]
        var normals: [Float]
        var uvs: [Float]
        var colors: [Float]
        var indices: [UInt32]
        var triMats: [UInt32]
        var triObjects: [UUID]
        var materials: [GPUMaterial]
        var objectOrder: [UUID]
    }
    private var cachedPrefix: PrefixCache?
    private var scopedVerified: Set<UInt64> = []
    private var scopedDisabled: Set<UInt64> = []
    // (2b) Support-index cache for appendSnowSheet, keyed by a
    // hash of the included world meshes + per-object amounts
    // (the only inputs buildSupportIndex reads).
    // v4.12: the ground-only support index (plane primitives)
    // the snow lattice lives on — same cache key (its inputs
    // are a subset of the same included set).
    // (1a) Dynamic-layer persistent vertex capacity (vertices)
    // + cached refit scratch.
    private var dynVertCapacity = 0
    private var dynRefitScratch: MTLBuffer?
    private var dynRefitScratchSize = 0
    private var dynTriMatCapacity = 0
    private var dynTriMatUploaded: [UInt32] = []
    // (4) Sim per-kind content knowledge: set by creation paths
    // (lattice sync, recorder edits, active sources), cleared
    // only by a full settle-scan read. Never guessed.

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
    /// v4.3: the player figure's reserved dynamic entry, kept
    /// SEPARATE from dynEntries so setDynamicEntries (the
    /// session's promoted/clone list) can never remove it.
    /// Every dynamic-layer composition appends it LAST (after
    /// the particle soups) in geometry and the material table.
    private var playerFigureEntry: DynEntry?
    /// v4.5: the figure's head entry (head + hair), composed
    /// immediately after the body entry with the
    /// FLAG_PLAYER_HEAD material; same matrix, same pose cadence.
    private var playerHeadEntry: DynEntry?
    /// v4.5: the unsplit figure mesh, kept for the snow
    /// intruder (its bounds are the whole figure's).
    private var playerFigureFullMesh: MeshData?
    /// v4.5: session-set — true while the camera is the Play
    /// first-person eye. uniforms() forwards it to the trace
    /// and guide kernels, which exclude the figure's head from
    /// primary hits only. Never set in Edit (the orbit camera
    /// sees the whole figure) or for stills from Edit.
    var hidePlayerHead = false
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
    /// v4.6: objects carrying the selection highlight (the
    /// multi-select set, or the single selection). A material-
    /// path tint ONLY — flags are untouched, so a highlighted
    /// emissive still casts its light and a highlighted glass
    /// object still transmits; the session syncs this on every
    /// selection change and refreshes via updateMaterialsOnly.
    var highlightObjectIDs: Set<UUID> = []

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
    /// v4.3: per-project weather-field seed. 0 reproduces the
    /// v4.2 field exactly; any other value re-derives the field
    /// offset + wind heading (see weatherSeedDerived) and
    /// RE-BAKES the field texture immediately, so the texture
    /// and the CPU mirrors below — which derive from the same
    /// seed at call time — can never disagree. The session
    /// observes the change through the existing rebuild flow
    /// (accumulation is reset here; it calls rebuild() after
    /// setting the seed, and the geometry pass re-derives from
    /// the new seed).
    /// v4.3: user-set maximum snow depth (metres, UI range
    /// 0.05…2, default 0.3 = the v3.4/v4.2 fixed cap).
    /// v4.3: ambient ABSOLUTE snow accumulation A (metres at
    /// the normalised drift factor). Snow progress is
    /// p = A / weatherMaxDepth; the displaced height is
    /// cover(p) × min(d/1.35, 1) × max(maxDepth, A) — see
    /// wgfSnowHeight.
    /// v4.3: per-object accumulated snow A_obj (metres), keyed
    /// by scene object id. Objects absent from the map use the
    /// ambient `weatherSnowAmount`. New/spawned objects start
    /// at 0 in the session's map, so they land bare and bury
    /// gradually instead of appearing pre-covered.
    /// v4.3: player footprint stamps in snow, newest last
    /// (the session caps the list). Consumed by the imprint
    /// pass in appendSnowSheet at rebuild time.
    /// v4.6: the snow DISPLACEMENT layer — the first piece of
    /// snow STATE (everything before it was a pure function of
    /// transforms + weather). Keyed by cell on a FIXED world
    /// grid (0.20 m cells, cell (0, 0) spans x ∈ [0, 0.20),
    /// z ∈ [0, 0.20)); the value is the signed height delta
    /// (metres) a plow left in the cell plus the ambient
    /// accumulation A when it was last touched. The plow
    /// recorder in appendSnowSheet writes it whenever an
    /// intruder moves through snow it touches; the sheet pass
    /// applies it (faded by |A_now − a|) under the repose
    /// relaxation, so trails and berms are real geometry that
    /// persists across rebuilds — and, via the session, across
    /// saves (WeatherState.plowCells). The session pushes the
    /// saved layer in at load and pulls it back out on save.
    /// v4.7: cumulative snowfall total (metres), written by the
    /// session every tick alongside weatherSnowAmount. Plow
    /// trails fade as this advances, even at the depth cap.
    /// v4.7: puddle freeze fraction 0...1, written every tick.

    // MARK: - v4.10.1 accumulation-path step instrumentation
    //
    // Sean's device crash (2026-11-01): "as soon as snow started
    // accumulating the app crashed" on the v4.10 build — the
    // accumulation path (sheet build + GPU sim) had never been
    // device-verified since the sim shipped in v4.8, and a
    // whole-path static audit pinned no single trap for clean
    // geometry, so the path is hardened (v4.10.1) AND tagged:
    // every accumulation-path step writes its name here on
    // entry. The tag is (a) readable live via `weatherStep`
    // (the coordinator wires footer display), and (b) persisted
    // to UserDefaults on CHANGE only — after a crash, the
    // persisted value names the last step the path reached.
    // Allocation-light by design: call sites pass string
    // literals, and the UserDefaults write fires only when the
    // tag actually changes.
    private(set) var weatherStep: String = "idle"
    func noteWeatherStep(_ tag: String) {
        guard tag != weatherStep else { return }
        weatherStep = tag
        UserDefaults.standard.set(tag, forKey: "lumenscene.weatherStep")
    }

    // v4.10.2: how many non-finite sim values the sheet
    // readback has had to replace with 0 (see appendSnowSheet).
    // Round 2's NaN hunt found NO finite-guard between the sim
    // buffers and the sheet vertices/AS build: one bad depth
    // (a corrupt persisted exposure amount, a singular
    // intruder inverse, a raced buffer) previously flowed
    // straight into AS geometry. The clamp converts that into
    // a one-vertex visual glitch; this counter — persisted,
    // flushed once per sheet build, and surfaced in the panel
    // footer by the session — records whether bad values are
    // occurring at all.
    private var nanClampCount: Int =
        UserDefaults.standard.integer(forKey: "lumenscene.nanClampCount")
    private var nanClampDirty = false
    func noteNanClamp() {
        nanClampCount += 1
        nanClampDirty = true
    }
    private func flushNanClampCount() {
        guard nanClampDirty else { return }
        nanClampDirty = false
        UserDefaults.standard.set(nanClampCount,
                                  forKey: "lumenscene.nanClampCount")
    }

    /// v4.10.1 hardening: floor-divide a world coordinate into a
    /// cell index WITHOUT the Int(Float) trap — a non-finite or
    /// out-of-range quotient (degenerate imported mesh, extreme
    /// transform) clamps to a bounded index instead of trapping.
    /// Identical to Int((v / cell).rounded(.down)) for every
    /// finite, in-range input, which covers all real scenes.
    private func cellIndexFloor(_ v: Float, _ cell: Float) -> Int {
        let f = (v / cell).rounded(.down)
        guard f.isFinite else { return 0 }
        return Int(min(max(f, -1_000_000), 1_000_000))
    }

    // MARK: - v4.8 weather simulation state
    //
    // Snow and water depths are PERSISTENT SIMULATION STATE in
    // GPU buffers (one Float per lattice vertex), stepped by
    // the sim kernels in Shaders.metal -- they are no longer
    // re-derived from the closed-form height function at
    // rebuilds. The closed form survives ONLY as (a) the
    // per-vertex deposition ceiling + drift-rate field baked
    // into simSnowCeil / simSnowRate at lattice build, and
    // (b) the load-time initial condition of the depth buffers.
    // The buffers are .storageModeShared, so the CPU skin in
    // appendSnowSheet reads the same memory the GPU steps --
    // and the plow recorder's edit transfers are applied by
    // direct writes into that shared memory (see
    // applySnowEdits), which is the legitimate shared-buffer
    // mechanism; simApplyEditsKernel exists for the same
    // operation when a caller already has a command buffer.
    /// Reflection resolution for water (steering 1): 1.0 =
    /// Full (v4.7 quality, the default), 0.5 = Half, 0.25 =
    /// Quarter. Mirrors how qualityPreference is held (a
    /// renderer-side setting chosen in the UI, not project
    /// data); the session setter is setWaterReflectionScale.
    var waterReflectionScale: Float = 1.0
    /// v4.10 (WS1): idle-freeze delay in seconds (0 = Off).
    /// Held exactly like qualityPreference / waterReflectionScale
    /// — a renderer-side, in-memory preference chosen in the UI,
    /// NOT project data and NOT UserDefaults.
    var idleFreezeSeconds: Double = 60
    /// v4.10 (WS1): cumulative wall-clock span removed from the
    /// precipitation overlay's phase clock on every wake from an
    /// idle freeze, so frozen spans do not move the rain/snow
    /// phase. See noteSceneResumed(frozenSpan:).
    var precipTimeOffset: CFTimeInterval = 0
    /// Incremented once per presented frame; feeds the
    /// rotating reflection-selection pattern in the trace
    /// kernel so sub-full scales cycle coverage across frames.
    private var reflFrameCounter: UInt32 = 0
    // v4.12 snow (games model) pipelines + state buffers. The
    // v4.11 buffer slots are reused with new meanings:
    // simSnowMassBuf now holds the authored FIELD depth and
    // simSnowRhoBuf the CARVE depth (see the v4.12 design
    // record); the age / gate / snapshot / edge buffers are
    // retired, replaced by the ground-support buffers.
    /// Host identity per vertex from the last full sync
    /// (bookkeeping for diagnostics; the v4.12 remap is purely
    /// positional and does not need it).
    /// The last intruder set a sheet build computed — the tick
    /// rasterises THESE into the occupancy buffers each frame
    /// (rebuilds refresh it whenever anything moves).
    /// Footprint press bookkeeping: keys of the stamps already
    /// pressed (a stamp presses ONCE, when it first appears)
    /// and presses queued for the next tick.
    /// Lattice identity the sim buffers currently describe.
    /// Host identity per vertex from the last full sync.
    /// Previous lattice geometry for the state remap: the
    /// v4.12 remap samples the old field/carve/water by world
    /// position (identity on an unchanged grid).
    /// Session-fed sim rates (written every frameTick).
    /// v4.12: the project's Temperature setting (nil = legacy
    /// projects), fed by the session alongside the rates. It
    /// flavours the carve refill time constant (wet snow
    /// slumps back faster; deep-cold snow keeps its tracks).
    /// v4.11: the project's Max Snow Depth (uniform cap). Fed
    /// by the session with the temperature; in v4.12 it is
    /// legacy bookkeeping only (the authored ceilings already
    /// carry the cap through the drift field).
    /// v4.10.2: the last committed sim-tick command buffer.
    /// The tick is fire-and-forget; this handle lets CPU
    /// readers/writers of the shared sim buffers fence on it
    /// (waitForSimTick) and lets the tick's completion handler
    /// report GPU errors into the weather-step trail.
    /// True while depths are still changing enough to matter.

    /// Pack a displacement-layer cell coordinate into its
    /// dictionary key (mirrors supportKey's packing).

    /// The layer as sparse Codable cells for the project file
    /// (sorted for a deterministic save; cells whose faded
    /// delta is negligible at the CURRENT accumulation are
    /// dropped — the trail has weathered away).
    /// True when the last geometry rebuild emitted water sheets —
    /// the material table then carries the sheet material last,
    /// and updateMaterialsOnly must reproduce it exactly.
    private var sceneHasSheetMaterial = false
    /// v4.4: same discipline for the snow sheet's material,
    /// which sits right after the sheet material in the table.
    private var sceneHasSnowMaterial = false
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
    private var lastTracedWet: Float = -1

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
            precipSeeds = seeds
            let wfDesc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
            wfDesc.usage = [.shaderRead]
            weatherFieldTex = device.makeTexture(descriptor: wfDesc)
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
    }

    /// v4.3: seed derivation (SplitMix32 from `weatherSeed`),
    /// shared by the bake and every CPU field mirror so both
    /// sides see the SAME field. Seed 0 is special-cased to
    /// the legacy field: offset (0,0), wind (1,0), seeded =
    /// false — the bake kernel and wgfDrift then run their
    /// original v4.2 formulas bit-exactly. For seed != 0, two
    /// SplitMix32 outputs in [0,1) map to a field offset with
    /// each component in [-28, 28) metres, and the wind index
    /// (seed >> 8) % 8 picks a heading in 45° steps from a
    /// LITERAL (cos, sin) table — the same literal Float
    /// constants are written out in Shaders.metal's bake, so
    /// no transcendental function is ever evaluated on either
    /// side.

    /// One-shot bake of the level-independent weather field
    /// (v3.3): 2048x2048 RGBA16F over a 64 m world window —
    /// R puddle metaball potential, G wet pooling, B snow
    /// drift, A broad pooling. Wet/snow levels are thresholds
    /// applied analytically in the trace kernel, so the bake
    /// never needs refreshing when build-up changes. v4.3:
    /// RE-RUNNABLE — it bakes with the CURRENT `weatherSeed`'s
    /// derived parameters, and `weatherSeed`'s didSet calls it
    /// whenever the seed changes. If the bake is unavailable,
    /// a 1x1 neutral texture stands in (global wet/snow
    /// shading still works; no puddles).

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
        noteWeatherStep("rebuild:entry")
        if !geometry, updateMaterialsOnly(scene: scene, meshes: meshes) {
            return
        }
        swapStaticIfReady()
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
        // v4.3: the LOCAL mesh is kept alongside the baked
        // world mesh — the snow-imprint pass needs each
        // object's local AABB + inverse transform.
        var included: [(obj: SceneObject, local: MeshData, world: MeshData)] = []
        for obj in scene.objects {
            guard let local = includedLocalMesh(for: obj, meshes: meshes) else { continue }
            included.append((obj, local, local.baked(by: obj.transform.matrix)))
        }
        // v4.13: the weather sheet/sim are removed (snow out,
        // rain = streaks + flat wetness) — a rebuild is now
        // purely the scene's own geometry; the v4.9 scoped
        // sheet-rebuild cache below is retired with it (its
        // fingerprint machinery stays inert).
        let fp = staticFingerprint(included: included)
        let fpHash = fingerprintHash(fp)
        let verifyScope = false
        let useScoped = false
        var triBudget = 220_000
        for (obj, _, world) in included {
            let matIndex = UInt32(materials.count)
            materials.append(gpuMaterialForObject(obj))
            objectOrder.append(obj.id)
            // v4.4: host geometry is appended FLAT, always.
            // Snow and puddle water are no longer carved out
            // of the host's own triangles — through v4.3
            // appendWeatherGeometry subdivided up-facing host
            // triangles and displaced them, which bound the
            // snow surface to the host's tessellation (a
            // 2-triangle plane grew crystalline facets and
            // zero-depth trenches along its own edges at deep
            // accumulation) and to the host's material (snow
            // on glass shaded as glass). The weather sheet
            // pass below now owns all of it.
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
        // v4.9 (1b): stage the complete new static state (all
        // buffers + CPU mirrors + counts) and build its AS
        // asynchronously. The live properties swap atomically
        // at the next frame boundary (see stageStaticBuild /
        // swapStatic); until then rendering continues on the
        // previous complete state.
        var prepared = PreparedStatic()
        prepared.cpuPositions = positions
        prepared.cpuIndices = indices
        prepared.cpuTriObject = triObjects
        prepared.cpuTriMats = triMats
        prepared.objectOrder = objectOrder
        prepared.materials = materials
        prepared.sheet = false
        prepared.snow = false
        prepared.triCount = indices.count / 3
        prepared.posBuffer = makeBuffer(positions)
        prepared.nrmBuffer = makeBuffer(normals)
        prepared.uvBuffer = makeBuffer(uvArr)
        prepared.colBuffer = makeBuffer(colArr)
        prepared.idxBuffer = makeBuffer(indices)
        prepared.triMatBuffer = makeBuffer(triMats)
        noteWeatherStep("rebuild:stageAS")
        stageStaticBuild(prepared)
    }

    /// Material-only refresh: same objects, same geometry — rewrite
    /// just the material buffer and the emissive light index (CDF)
    /// from the cached CPU triangles. Returns false when the object
    /// set changed and a full rebuild is required.
    @discardableResult
    func updateMaterialsOnly(scene: Scene, meshes: [UUID: MeshData]) -> Bool {
        swapStaticIfReady()
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
        // v4.9 (item 5): geometry is unchanged by construction
        // here (the candidate list matched the cached geometry),
        // so the emissive rescan is signature-gated and the
        // material table writes only changed slots.
        uploadEmissiveIndexGated(materials: fullMats, rescanStatic: true,
                                 geometryUnchanged: true)
        uploadMaterialTable(fullMats)
        flushMaterialTable()
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
                || ((playerFigureEntry?.material.flags ?? 0) & 1) != 0
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

    // MARK: - v4.9 helpers (hashing, material table, fences)

    private func fnv1a(_ h: inout UInt64, _ v: UInt64) {
        h = (h ^ v) &* 1099511628211
    }

    /// Content hash of one baked world mesh (all five arrays,
    /// counts + values). Used by the scoped-rebuild fingerprint
    /// and the support-index cache key: when in doubt, the
    /// fingerprint covers MORE, never less.
    private func bakedHash(_ world: MeshData) -> UInt64 {
        var h: UInt64 = 1469598103934665603
        fnv1a(&h, UInt64(world.positions.count))
        for v in world.positions { fnv1a(&h, UInt64(v.bitPattern)) }
        fnv1a(&h, UInt64(world.normals.count))
        for v in world.normals { fnv1a(&h, UInt64(v.bitPattern)) }
        fnv1a(&h, UInt64(world.uvs.count))
        for v in world.uvs { fnv1a(&h, UInt64(v.bitPattern)) }
        fnv1a(&h, UInt64(world.colors.count))
        for v in world.colors { fnv1a(&h, UInt64(v.bitPattern)) }
        fnv1a(&h, UInt64(world.indices.count))
        for v in world.indices { fnv1a(&h, UInt64(v)) }
        return h
    }

    /// Field-wise hash of a GPUMaterial (explicit fields, never
    /// raw struct bytes — padding is not deterministic).
    private func materialHash(_ m: GPUMaterial) -> UInt64 {
        var h: UInt64 = 1469598103934665603
        for v in [m.baseColor.x, m.baseColor.y, m.baseColor.z, m.baseColor.w,
                  m.emissive.x, m.emissive.y, m.emissive.z, m.emissive.w,
                  m.glassParams.x, m.glassParams.y, m.glassParams.z, m.glassParams.w,
                  m.tint.x, m.tint.y, m.tint.z, m.tint.w] {
            fnv1a(&h, UInt64(v.bitPattern))
        }
        fnv1a(&h, UInt64(m.texIndex))
        fnv1a(&h, UInt64(m.flags))
        fnv1a(&h, UInt64(m.pad.x))
        fnv1a(&h, UInt64(m.pad.y))
        return h
    }

    private func gpuMaterialEqual(_ a: GPUMaterial, _ b: GPUMaterial) -> Bool {
        a.baseColor == b.baseColor && a.emissive == b.emissive
            && a.glassParams == b.glassParams && a.tint == b.tint
            && a.texIndex == b.texIndex && a.flags == b.flags && a.pad == b.pad
    }

    private func materialsEqual(_ a: [GPUMaterial], _ b: [GPUMaterial]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { p in gpuMaterialEqual(p.0, p.1) }
    }

    private func prefixEqual(_ a: PrefixCache, _ b: PrefixCache) -> Bool {
        a.positions == b.positions && a.normals == b.normals
            && a.uvs == b.uvs && a.colors == b.colors
            && a.indices == b.indices && a.triMats == b.triMats
            && a.triObjects == b.triObjects && a.objectOrder == b.objectOrder
            && materialsEqual(a.materials, b.materials)
    }

    /// v4.9 (1c): fingerprint of everything EXCEPT the sheet
    /// segment: included object ids in order, per-object baked
    /// world-mesh content, per-object material-table content,
    /// and per-object transforms. The sheet pass reads other
    /// inputs too (amounts, weather levels, the plow layer,
    /// intruders) — those re-run with the sheet and are NOT
    /// part of the prefix, so they are excluded here; anything
    /// that feeds the prefix arrays is included.
    private func staticFingerprint(
        included: [(obj: SceneObject, local: MeshData, world: MeshData)]
    ) -> StaticFingerprint {
        var fp = StaticFingerprint()
        for (obj, _, world) in included {
            fp.ids.append(obj.id)
            fp.baked.append(bakedHash(world))
            fp.mats.append(materialHash(gpuMaterialForObject(obj)))
            var h: UInt64 = 1469598103934665603
            let m = obj.transform.matrix
            for col in [m.columns.0, m.columns.1, m.columns.2, m.columns.3] {
                for v in [col.x, col.y, col.z, col.w] {
                    fnv1a(&h, UInt64(v.bitPattern))
                }
            }
            fp.xf.append(h)
        }
        return fp
    }

    private func fingerprintHash(_ fp: StaticFingerprint) -> UInt64 {
        var h: UInt64 = 1469598103934665603
        for id in fp.ids {
            for byte in id.uuidString.utf8 { fnv1a(&h, UInt64(byte)) }
        }
        for v in fp.baked { fnv1a(&h, v) }
        for v in fp.mats { fnv1a(&h, v) }
        for v in fp.xf { fnv1a(&h, v) }
        return h
    }

    /// v4.9 (2b): cache key for buildSupportIndex — its only
    /// inputs are the included world meshes (positions, normals,
    /// indices) and each object's snow amount.

    /// v4.9: bounded fence for CPU writes into .storageModeShared
    /// buffers the GPU may still be reading (dynamic vertex
    /// slices, the material table, the dynamic tri-mat buffer).
    /// Waits ONLY on the previous presented frame's command
    /// buffer — never on anything encoded for the current frame
    /// — so the bound is one frame of GPU work, and at vsync
    /// cadence that buffer is normally already complete.
    private func waitForPreviousFrame() {
        if let cb = lastFrameCommandBuffer, cb.status != .completed {
            cb.waitUntilCompleted()
        }
    }

    /// v4.9 (item 5): stage a new material table. The actual
    /// buffer write happens once at the flush point (top of
    /// draw / still / swap), with per-slot diffing — N cheap
    /// setDynamicMaterial calls in a row cost ONE flush.
    private func uploadMaterialTable(_ mats: [GPUMaterial]) {
        matPendingTable = mats
        matTableDirty = true
    }

    /// Write the staged material table into the persistent
    /// shared matBuffer, changing only the slots whose bytes
    /// differ from what was last uploaded. Buffer CONTENTS at
    /// any draw are exactly the staged table — only the write
    /// path changed, never the values.
    private func flushMaterialTable() {
        guard matTableDirty else { return }
        matTableDirty = false
        let mats = matPendingTable
        let stride = MemoryLayout<GPUMaterial>.stride
        var reallocated = false
        if matBuffer == nil || matBufferCapacity < mats.count {
            let cap = max(mats.count, 1)
            matBuffer = device.makeBuffer(length: max(cap * stride, 4),
                                          options: .storageModeShared)
            matBufferCapacity = (matBuffer != nil) ? cap : 0
            matBufferUploaded = []
            reallocated = true
        }
        guard let buf = matBuffer, matBufferCapacity > 0 else {
            matBufferUploaded = []
            return
        }
        let fullRewrite = matBufferUploaded.count != mats.count
        var anyDiff = fullRewrite
        if !anyDiff {
            for i in mats.indices where !gpuMaterialEqual(mats[i], matBufferUploaded[i]) {
                anyDiff = true
                break
            }
        }
        guard anyDiff else {
            matBufferUploaded = mats
            return
        }
        // In-place writes into a buffer an in-flight frame may
        // still be reading: fence on the previous frame first.
        // A freshly allocated buffer is referenced by nothing
        // in flight, so it needs no fence.
        if !reallocated { waitForPreviousFrame() }
        let ptr = buf.contents().bindMemory(to: GPUMaterial.self,
                                            capacity: matBufferCapacity)
        if fullRewrite {
            for i in mats.indices { ptr[i] = mats[i] }
        } else {
            for i in mats.indices where !gpuMaterialEqual(mats[i], matBufferUploaded[i]) {
                ptr[i] = mats[i]
            }
        }
        matBufferUploaded = mats
    }

    /// Emissive-relevant signature of a material table: the
    /// emissive flag bit + the full emissive SIMD4 per slot,
    /// plus the triangle counts the scan ranges over. Anything
    /// else in a material (base colour, snow pad, glass…) can
    /// change without moving a single emissive byte.
    private func emissiveSignature(_ mats: [GPUMaterial]) -> UInt64 {
        var h: UInt64 = 1469598103934665603
        fnv1a(&h, UInt64(triCount))
        fnv1a(&h, UInt64(dynTriCount))
        fnv1a(&h, UInt64(mats.count))
        for m in mats {
            fnv1a(&h, UInt64(m.flags & 1))
            fnv1a(&h, UInt64(m.emissive.x.bitPattern))
            fnv1a(&h, UInt64(m.emissive.y.bitPattern))
            fnv1a(&h, UInt64(m.emissive.z.bitPattern))
            fnv1a(&h, UInt64(m.emissive.w.bitPattern))
        }
        return h
    }

    /// v4.9 (item 5d): gated emissive upload. When the caller
    /// knows geometry did not change and the emissive signature
    /// is unchanged since the last upload, the index (tris,
    /// areas, centroids, CDF, GPU buffers) is provably identical
    /// to what is already uploaded, so the rescan AND the
    /// re-upload are skipped entirely.
    private func uploadEmissiveIndexGated(materials: [GPUMaterial],
                                          rescanStatic: Bool,
                                          geometryUnchanged: Bool) {
        let sig = emissiveSignature(materials)
        if geometryUnchanged, lastEmissiveSignatureValid,
           sig == lastEmissiveSignature {
            return
        }
        uploadEmissiveIndex(materials: materials, rescanStatic: rescanStatic)
        lastEmissiveSignature = sig
        lastEmissiveSignatureValid = true
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
        // v4.6: selection highlight — a warm tint blended into
        // the base colour so every selected object reads at a
        // glance in the 3D view (the tree row alone was easy to
        // lose). Applied after the preview branch and never to
        // flags: lighting behaviour is unchanged. Glass keeps
        // transmitting (its base colour tints the transmission,
        // so the mark shows through the glass look).
        if highlightObjectIDs.contains(obj.id) {
            let tint = SIMD3<Float>(1.0, 0.60, 0.16)
            base = base + (tint - base) * 0.38
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
        // v4.3: pad.x carries this object's snow-exposure
        // factor as a Float bit pattern (the trace + guide
        // kernels reinterpret Material.pad.x and multiply
        // their snow cover by it): the object's own progress
        // p_obj over the ambient progress p_amb, clamped 0…1,
        // so a freshly spawned object (A_obj = 0) shades bare
        // while long-standing objects are fully covered. When
        // the ambient amount is ~0 the factor is 1 (nothing
        // to expose against; also guards the division).
        // v4.13: Material.pad.x carried the per-object snow
        // exposure factor until v4.12; snow shading is removed
        // and the field is now always zero.
        g.pad = SIMD2<UInt32>(0, 0)
        return g
    }

    /// v3.4: the puddle water material — one shared entry appended
    /// after all object materials whenever the geometry contains
    /// water sheets. FLAG_SHEET (16) routes it to the water
    /// shading branch of the trace kernel; the dark blue-grey is
    /// the water body colour seen where the mirror reflection is
    /// weak (steep view angles).

    /// v4.4: the snow material — one shared entry appended
    /// after the puddle sheet material whenever the snow
    /// sheet emitted geometry. FLAG_SNOW (32) routes it to
    /// the snow branch of the trace kernel: opaque, matte,
    /// blue-white — snow shades as snow whatever the object
    /// underneath is made of (glass, emissive, textured).
    /// Through v4.3 the snow geometry carried the HOST
    /// object's material index, so snow lying on a glass box
    /// took the glass path and refracted like glass (Sean's
    /// device report 2026-10-31: "snow still turns to glass
    /// on glass objects").

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
    //   * SNOW (v3.4–v4.3 form): up-facing triangles were
    //     subdivided and displaced upward by the drift-shaped
    //     snow depth, with a barycentric edge fade pinning
    //     every original triangle boundary to zero. v4.4
    //     REPLACED that construction with the independent
    //     lattice sheet further below (see "Snow support
    //     field + weather sheet"): the fade carved full-depth
    //     trenches along host edges at the new metre-scale
    //     depths, the displaced surface inherited the host's
    //     tessellation AND its material, and Sean's device
    //     verdicts followed. The height FIELD below (drift,
    //     coverage, accumulation) is unchanged — only the
    //     surface built from it changed.
    //   * PUDDLES: where the mirrored puddle mask is strong at
    //     a lattice cell's centre, a water SHEET quad is
    //     emitted above the surface — up to ~13 mm at the
    //     pool centre by depth, rim vertices tucked 2.5 mm
    //     BELOW the surface, so the edge is a tiny sloped wall
    //     that catches light at grazing angles instead of a
    //     painted outline. Sheets carry the host object's id
    //     (tapping a puddle selects its floor) and the shared
    //     FLAG_SHEET material. (v3.4 emitted them per
    //     subdivided host triangle; v4.4 emits them from the
    //     same lattice as the snow sheet.)
    //
    // The field functions below are exact CPU mirrors of the
    // shader's baked-field formulas (same hashes, same
    // constants), so geometry and shading agree on where the
    // drifts and pools are. Levels arrive quantised to steps of
    // 1/24 (the session rebuilds only when a step is crossed),
    // and a scene-wide budget caps added triangles; past the
    // budget the sheet simply stops growing rather than
    // blowing the frame budget. Per-frame cost of all of
    // this is zero: the result is ordinary static geometry in
    // the same single acceleration structure.







    /// Window fade of the baked field (1 inside ±29 m, 0 at ±32 m).

    /// Drift field (baked texture B channel), window-faded.
    /// v4.3: for a non-zero seed the sample point is first
    /// shifted by the seed offset, then ROTATED by the seed's
    /// wind heading, and the channel becomes multi-scale and
    /// wind-anisotropic — the exact operations, constants and
    /// order of the bake kernel in Shaders.metal. The fade is
    /// still evaluated at the UNshifted (x, z): it models the
    /// texture window at the lookup position, exactly as the
    /// trace kernel mixes the baked sample toward neutral.
    /// Seed 0 keeps the original single-scale formula.

    /// Broad pooling field (baked texture A channel), faded.
    /// v4.3: seed offset only — the formula is unchanged.

    /// Metaball puddle potential (baked texture R channel), faded.
    /// v4.3: seed offset only — the formula is unchanged.

    /// Snow depth (metres) the geometry carries at (x, z) for
    /// an ABSOLUTE accumulation A (v4.3, replacing the fixed
    /// 0.30 m cap + quantised level): progress p = A / maxDepth
    /// (uncapped), pc = min(p, 1); drifted depth
    /// d = pc × (0.3 + 1.4 × drift); height =
    /// cover(d) × min(d / 1.35, 1) × max(maxDepth, A) — the
    /// last factor keeps already-deep snow continuous when the
    /// user lowers the max-depth setting (no bulldozing).
    /// Zero exactly where the coverage is zero, so the
    /// silhouette and the shader whitening agree. At
    /// A = level × 0.3 with maxDepth 0.3 this reduces exactly
    /// to the v3.4 formula.


    /// The snow depth (metres) at a world XZ — v4.11: the
    /// SIMULATED depth where the sim is live (the session's
    /// footprint gate must see the trench the player is
    /// actually standing in, not the ambient closed form),
    /// falling back to the ambient accumulation before the
    /// sim initialises. The 4-point sample reads the shared
    /// depth buffer directly; a torn read against an
    /// in-flight tick is harmless for a stamping gate.

    /// The trace kernel's puddle mask, mirrored: (mask, depth01)
    /// where depth01 says how deep into the pool the point sits.

    // MARK: Snow imprints (v4.3: objects crush the snowfield)
    //
    // An intruder is anything resting in the snow: an included
    // static object, a dynamic entry (promoted objects and
    // spawned clones), or the player figure. Its footprint is
    // its LOCAL XZ bounds — a circle for sphere primitives
    // (centre = local XZ centre, radius = local XZ half-extent
    // x), an OBB rect otherwise — tested per lattice vertex by
    // transforming the vertex into the intruder's local space
    // with the inverse matrix. Where the intruder's world
    // bottom reaches into the snow, the field height is
    // compressed toward (bottom − 0.02 m) with a soft inner
    // lip — INSIDE the footprint only (v4.5: outside it the
    // snow is never compressed; it BANKS against the object,
    // +18% of field height at the wall easing out over
    // 0.8 m — see imprintedSnowHeight). Everything is
    // derived from transforms at rebuild time — drag troughs
    // appear for free because a drag rebuilds at the object's
    // live transform, and nothing sculpted is ever stored.

    /// v4.6: one intruder's footprint as the plow recorder
    /// last saw it. The recorder is rebuild-to-rebuild
    /// memory: an intruder whose world footprint CENTRE moved
    /// since the previous rebuild AND whose body reaches the
    /// snow plows the span between the two positions.

    /// v4.6: the previous rebuild's intruder footprints, keyed
    /// by intruder identity (see plowIdentityKey). Replaced
    /// wholesale at the end of every sheet pass.
    private var prevPlowIntruders: [Int: PlowPrev] = [:]

    /// Identity for the plow recorder: statics by their
    /// object id; dynamic entries by the hash of the key the
    /// session gave them — an entry whose key changes between
    /// rebuilds simply starts fresh (no trail that step).



    /// Gather this rebuild's intruders: static objects (with
    /// their baked world meshes), dynamic entries, and the
    /// player figure.

    // MARK: - Snow plowing (v4.6: displacement with memory)
    //
    // The imprint system above is a pure function of CURRENT
    // transforms: snow conforms to a block's footprint and
    // heals to pristine field the moment the block leaves.
    // Sean (2026-10-31 16:55): a dragged block must PUSH the
    // snow out of the way — a bow wave while it moves and a
    // carved trail with side berms after it passes. That is
    // displacement WITH MEMORY, so v4.6 adds the one piece of
    // snow state: the displacement layer (`snowPlow`).
    //
    // The recorder runs once per sheet pass, BEFORE the
    // columns are computed (so a trail is visible in the same
    // rebuild that carves it). For each intruder it compares
    // the footprint centre against the previous rebuild's
    // (prevPlowIntruders). An intruder plows when it moved
    // (≥ 2 cm), when its body overlaps some snow column of
    // the sheet extent, and when that snow is actually in its
    // way: it can only move snow at or below the snow-top
    // CEILING recorded last rebuild (its top then — a block
    // resting on a floor cannot carve the snow sheet of a
    // table standing over it; a carried block, whose body was
    // above every column last rebuild, carves nothing).
    //
    // Carving works on the sheet lattice itself. For every
    // lattice vertex inside the SWEPT footprint (the shape
    // the footprint swept between the two rebuilds — a
    // polygon through both positions' corners, or a capsule
    // for a circle), the snow above the intruder's underside
    // is removed: the vertex's layer delta is driven so the
    // total height equals the plow depth (underside − 0.02,
    // the same resting depth the imprint uses). The removed
    // VOLUME is then deposited: 50 % into a band ahead of the
    // footprint's leading edge (the bow wave), 25 % + 25 %
    // into bands along the swept path's two side edges (the
    // berms) — direction from the motion vector; a step along
    // the path's own axis puts nearly everything ahead. The
    // sheet's repose relaxation runs AFTER the layer is
    // applied, so berms slump to natural slopes exactly like
    // any other snow. The layer stores ONLY recorder writes
    // (relaxation never feeds back into it — see the note at
    // the relaxation), so the trail is a true fixed point.
    // Volume is conserved at the recorder level: removal is
    // measured against the base WITHOUT the mover's own
    // transient banking, and the full removed volume is
    // distributed over the front/side bands (no caps — the
    // relaxation does the shaping).
    /// The recorder's view of this rebuild's sheet extent:
    /// the lattice parameters appendSnowSheet will use
    /// (kept in ONE place so the recorder and the sheet can
    /// never disagree about the grid).

    /// The layer delta stored for a world point: the value
    /// of the FIXED 0.20 m cell containing the point, faded
    /// by |A_now − a_touch| (gone at 0.25 m of weather change
    /// — snowfall refills, melt lowers). This is the ONLY
    /// read the sheet makes of the layer, per vertex.
    ///
    /// The read is deliberately NEAREST-CELL, not bilinear:
    /// the audit's first design interpolated over the four
    /// surrounding cells and wrote them back with splat
    /// solves — overlapping writes coupled through the shared
    /// corners and the carve/write-back loop diverged (layer
    /// values reached 10²⁵ m in the host port). Nearest-cell
    /// reads make every write exact and local: a cell's value
    /// is what every vertex inside it reads, full stop. The
    /// 0.20 m quantisation never reaches the screen raw — the
    /// layer sums with the smooth drift field and the repose
    /// relaxation smooths the sum every rebuild.

    /// v4.6: set the layer cell containing (x, z) to `value`
    /// (raw delta, stamped at aNow so its fade is 1). Cells
    /// under 2 mm are removed — the layer stays sparse.

    /// v4.6: add `increment` to the layer cell containing
    /// (x, z), composing on the cell's FADED current value
    /// (depositing never resurrects weathered snow).

    /// Snapshot one intruder's footprint for the NEXT
    /// rebuild's recorder (local-space corners/centre + the
    /// world anchors the gates read).

    /// Run the plow recorder for this rebuild. `baseH` /
    /// `supportY` / `hasSupport` describe the lattice BEFORE
    /// the layer is applied (field + imprints + footprints,
    /// exactly the v4.5 column heights) — the recorder reads
    /// removal from them and writes only the layer. Mutates
    /// `snowPlow` and `prevPlowIntruders`.
    /// Returns the lattice indices carved THIS pass — the
    /// sheet pins them for this pass's relaxation (blade-
    /// compacted snow; see the call site in appendSnowSheet).

    /// The snow height at one point AFTER imprints: `fieldH`
    /// is the un-imprinted field height AT THIS POINT (the
    /// caller evaluates wgfSnowHeight per tap, so the gradient
    /// taps follow the imprint exactly like the centre vertex).
    /// `hostY` is the host surface's world y at the vertex and
    /// `hostID` the host object (its own intruder is skipped).
    /// Intruders apply sequentially — each sees the previous
    /// result as its input height and re-tests contact against
    /// its own bottom. Footprint stamps apply last,
    /// multiplicatively, with no rim.

    // MARK: - Snow support field + weather sheet (v4.4)
    //
    // The snow surface no longer derives from the host mesh.
    // A regular lattice in world XZ (0.20 m cells, coarsened
    // only if the support extent would break the triangle
    // envelope) is draped over the scene's snow-holding
    // surfaces: at each lattice vertex the SUPPORT — the
    // topmost up-facing surface under that point, sampled
    // from the real triangles, so a sphere gets a true curved
    // cap and a rooftop its own height — plus the v4.3 snow
    // HEIGHT function (drift field, absolute accumulation vs
    // maxDepth, per-object exposure via the support host's
    // own amount, imprints, footprints) gives the sheet's
    // top. Because the height is a single-valued function of
    // (x, z), the sheet is continuous by construction: no
    // barycentric edge fade, no trenches along host triangle
    // borders, no folds (a heightfield graph's geometric
    // normal always has a positive Y component). Vertex
    // normals are taps of the SAME total height function
    // (support + snow), so the shading can never disagree
    // with the surface the way the v4.3 fade-vs-taps mismatch
    // did (audit: mean normal error 34 degrees inside the old
    // fade band, 0.6 degrees for the tap scheme on the total
    // function). Snow triangles carry the shared FLAG_SNOW
    // material — never the host's — and puddle water sheets
    // are emitted from the same lattice with the shared
    // FLAG_SHEET material. Perimeter skirts drop from any
    // sheet edge that was CUT at depth (a support ending in
    // mid-air, a footprint cliff) to the support below, so
    // deep snow never shows a torn open edge. Picking a snow
    // or water triangle still selects the object whose
    // surface carries it.

    /// One snow-holding host triangle, registered in the
    /// support index.

    /// The support sampled at one point: the topmost
    /// snow-holding surface there.


    /// XZ spatial hash (0.5 m cells) of every included static
    /// object's snow-holding triangles (flat normal Y > 0.5 —
    /// steeper than ~60 degrees cannot hold snow or a puddle;
    /// the shedding taper handles the rest).

    /// The topmost snow-holding support at (x, z): among the
    /// candidate triangles containing the point (XZ
    /// barycentric), the highest interpolated surface wins.

    /// The full snow column at (x, z): sampled support + the
    /// v4.3 height function for the support host's own
    /// accumulation, shed by the support's steepness, then
    /// imprinted. `pre` lets a lattice vertex reuse the
    /// support it already sampled at its exact position;
    /// gradient taps pass nil and sample their own point, so
    /// the taps and the vertex heights are the same function.

    /// The v4.4 weather sheet pass: lattice over the support
    /// union, snow sheet + puddle water sheets, charged
    /// against the shared weather budget like the v3.4/v4.3
    /// subdivision was.

    // MARK: - v4.8 simulation buffers



    /// Host-masked bilinear sample: like simSampleBilinear but
    /// only taps whose OLD host equals `want` contribute
    /// (weights renormalised; nil when no tap matches). Used
    /// for snow that rides a MOVING host — the unmasked sample
    /// at the host-motion offset also reads whatever other
    /// host's snow lies at that world position (the deep
    /// ground field ahead of a dragged box), importing foreign
    /// mass on every rebuild (in-app DRAG audit: +0.5% over
    /// one drag).

    /// Bilinear sample of a previous lattice field at world XZ
    /// (0 outside). Used by the object-move remap.

    /// (Re)initialise the sim buffers for a lattice build.
    /// `targetSnow` is the v4.7 surface (closed form + imprints
    /// + plow layer) for this build -- the load-time initial
    /// condition, so a freshly loaded project's first skin is
    /// EXACTLY the v4.7 surface (the audit measures this diff).
    /// On a lattice/support change, existing vertices are
    /// REMAPPED: a vertex whose support host moved samples the
    /// old field in the host's frame (its snow travels with
    /// it); other vertices sample at their world position when
    /// the old topmost host there matches, and otherwise fall
    /// back to the closed-form equilibrium for the support now
    /// beneath them (targetSnow). Patch mass is conserved to
    /// bilinear-sampling accuracy (audit tolerance 5%).
    /// v4.10.2: the sim tick is fire-and-forget on the serial
    /// queue (Lane-A finding): CPU code that reads or writes
    /// the shared sim buffers must first know the last tick
    /// finished, or it races the GPU kernels on the same
    /// allocations. Bounded the same way as
    /// waitForPreviousFrame — the tick is small and ahead of
    /// the previous frame's trace in queue order.

    /// v4.12: CPU prep for one tick's snow step — rasterise
    /// the live intruder set into the occupancy buffers.
    /// Called after waitForSimTick and before any encoder
    /// exists, so the shared-buffer writes cannot race a
    /// tick. Returns true only when some ground column's snow
    /// ACTUALLY overlaps an intruder's volume right now (a
    /// resting object whose footprint snow has already been
    /// carved away is not activity — otherwise the sim could
    /// never sleep in a scene with anything sitting in snow).
    /// v4.12: the test runs against the GROUND column (ground
    /// support + rendered depth) — the only place snow exists.

    /// v4.12: stamp the queued footprint presses into the
    /// press buffer (zeroed first). Called once per tick that
    /// will actually step, after the activity decision and
    /// before the encoder exists; the field kernel consumes
    /// the stamps on the first substep and a clear dispatch
    /// follows it, so a press lands exactly once.


    /// Apply recorder edit transfers (vertex delta pairs) to
    /// the snow state by direct shared-buffer writes, and
    /// return the resulting depths for the skin.

    // LS-REPRO: metrics snapshot for the repro driver
    // (v4.12 snow-audit variants). INERT in normal use — no
    // call site outside ReproDriver, and a normal launch never
    // calls it. Returns the CURRENT rendered snow-depth array,
    // the (topmost) support array, the per-vertex authored
    // ceiling array, the authored FIELD and CARVE state arrays,
    // and the ground-support arrays (all one value per lattice
    // vertex, row-major vz rows x vx columns), the lattice
    // parameters, and the plow layer's stored-delta extrema
    // (legacy; the layer is dead code since v4.11). Fences on
    // the last sim tick first (the same waitForSimTick
    // discipline the sheet readback uses) so the copies are
    // not racing a GPU tick. When the sim is not initialised
    // the arrays are empty and `initialised` is false.
    // `swapped` mirrors `staticEverSwapped` (set by
    // swapStatic): the frame-boundary swap witness — the AS
    // itself can be nil where ray tracing is unsupported, so
    // this flag, not hasGeometry, proves a swap occurred.
    /// v4.11 diagnostics: full-lattice remaps since launch
    /// (kept in v4.12; the carve audit reads it).
    var reproRemapCount = 0

    /// LS-REPRO (v4.12): the snow sheet triangles the last
    /// sheet build emitted (positions only, 3 vertices per
    /// triangle, in emission order — snow cells + wall clips +
    /// skirts). The wall-contact audit measures the RENDERED
    /// surface against the box from these. Stashed only while
    /// ReproDriver.isActive; empty otherwise.

    /// LS-REPRO (v4.12): mean CPU encode time of the snow
    /// tick's dispatch section (EMA, milliseconds), or nil
    /// when no tick has been measured. CPU ENCODE time only —
    /// NOT GPU execution time (labelled as such in the record).

    /// LS-REPRO (v4.12 shader-gate audit): evaluate the shared
    /// snowAccumFactor on the GPU for the three probe normals
    /// (vertical face / flat top / 45-degree roof) at full snow
    /// level. Runs on its OWN command buffer with a blocking
    /// wait — safe between frame pumps, never on the render
    /// path. Returns nil if the pipeline/queue is unavailable.

    /// Drop all sim state (Clear build-up / Re-roll): the next
    /// sheet build re-initialises from the then-current target.

    /// v4.10 (WS1): the session woke from an idle freeze spanning
    /// `frozenSpan` wall-clock seconds. Rebase every wall-clock
    /// scene clock the renderer owns so that span does not leak
    /// into the simulation or the precipitation phase:
    /// - precipTimeOffset grows by the span (encodePrecip
    ///   subtracts it), so rain/snow hang exactly where they
    ///   froze instead of jumping ahead.
    /// - simLastTick = 0 makes the next tickWeatherSim call take
    ///   its rebase branch (it sets simLastTick = now and steps
    ///   nothing), so none of the up-to-4 substeps that a raw dt
    ///   could otherwise produce run for frozen time.
    /// - simLastSettleScan = 0 likewise restarts the settle-scan
    ///   cadence from the wake frame.
    /// Nothing else changes: rates, substep size and all sim
    /// math are untouched.
    func noteSceneResumed(frozenSpan: CFTimeInterval) {
        precipTimeOffset += frozenSpan
    }

    /// Step the simulation from the render loop. Fixed substeps
    /// of 1/30 s, at most 4 per frame (excess time is dropped,
    /// never spiralled). Per substep: the v4.8 water system
    /// (source/sink + levelling flow) runs untouched; the
    /// v4.11 snow physics runs sources → flux → displacement →
    /// writeback. Skipped entirely when nothing can change: no
    /// weather input, no live intrusion or press, and the
    /// settle scan (every 0.5 s, CPU scan of the shared
    /// buffers) reports no slope past the dynamic repose.


    /// v4.9 (1b): encode (and commit) an AS build for a staged
    /// static state WITHOUT waiting. Returns the structure and
    /// its command buffer (both nil for an empty scene, which
    /// needs no build and is immediately swappable).
    /// v4.10.2: `failed` distinguishes "nothing to build"
    /// (empty scene / no RT) from "a build was required but
    /// could not be encoded" (allocation/encoder failure) —
    /// round 1 returned (nil, nil) for both, and a valid
    /// structure was silently discarded with the failure.
    private func makeStaticAS(posBuffer: MTLBuffer?, idxBuffer: MTLBuffer?,
                              triCount: Int)
        -> (structure: MTLAccelerationStructure?, cmd: MTLCommandBuffer?,
            failed: Bool) {
        guard rayTracingSupported, triCount > 0,
              let posBuffer, let idxBuffer else {
            return (nil, nil, triCount > 0 && rayTracingSupported)
        }
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
              let enc = cmd.makeAccelerationStructureCommandEncoder() else { return (nil, nil, true) }
        enc.build(accelerationStructure: structure, descriptor: desc,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        return (structure, cmd, false)
    }

    /// v4.10.2 (round-2 Lane-B F1): a staged static build
    /// whose GPU work failed — encode failure, or a build
    /// command buffer that ended in .error — must NEVER be
    /// installed as the live state. Round 1's paths waited for
    /// completion but never checked the result, and the frame
    /// path treated only .completed as done: an errored build
    /// wedged every later swap, and the synchronous flush
    /// (thumbnails / export / final render) installed the
    /// never-built AS outright — the next trace then
    /// intersected a corrupt structure and faulted the
    /// process. The first sheet build is the largest AS the
    /// app ever builds, i.e. the one most likely to error.
    private func staticStageFailed(_ p: PreparedStatic) -> Bool {
        if p.buildFailed { return true }
        if let c = p.cmd, c.status == .error { return true }
        return false
    }

    /// Kick the AS build for a staged state and hold it as the
    /// pending build. A frame is owed (the swap happens at a
    /// frame boundary), so presentation is marked dirty.
    private func kickStatic(_ p: inout PreparedStatic) {
        let (structure, cmd, failed) = makeStaticAS(posBuffer: p.posBuffer,
                                                    idxBuffer: p.idxBuffer,
                                                    triCount: p.triCount)
        p.accel = structure
        p.cmd = cmd
        p.buildFailed = failed
        pendingStatic = p
        presentationDirty = true
    }

    /// Stage a complete new static state. The FIRST build ever
    /// completes synchronously (there is no previous state to
    /// keep presenting, and callers may read hasGeometry /
    /// picking state immediately). Later builds are async:
    /// if a build is already in flight, this stage is held as
    /// the deferred stage — coalescing: only the LATEST stage
    /// is ever built next, the in-flight build's state is
    /// discarded unswapped when it completes, and there is no
    /// queue beyond (current, pending, deferred).
    private func stageStaticBuild(_ prepared: PreparedStatic) {
        var prepared = prepared
        // LS-REPRO TOGGLE `syncas`: force the synchronous static
        // AS build path (the pre-v4.9 behaviour) — build this
        // stage now, wait for it, and swap it in immediately,
        // bypassing the async staged build + frame-boundary
        // swap. Any in-flight stage is completed and dropped
        // unswapped (its scene is older than this stage).
        // Default OFF = shipped behaviour.
        if ReproToggles.syncAS {
            if let p = pendingStatic { p.cmd?.waitUntilCompleted(); pendingStatic = nil }
            deferredStatic = nil
            let (structure, cmd, failed) = makeStaticAS(posBuffer: prepared.posBuffer,
                                                        idxBuffer: prepared.idxBuffer,
                                                        triCount: prepared.triCount)
            prepared.accel = structure
            prepared.cmd = cmd
            prepared.buildFailed = failed
            cmd?.waitUntilCompleted()
            if staticStageFailed(prepared) {
                noteWeatherStep("rebuild:stageError")
                return
            }
            swapStatic(prepared)
            return
        }
        if !staticEverSwapped, pendingStatic == nil, deferredStatic == nil {
            let (structure, cmd, failed) = makeStaticAS(posBuffer: prepared.posBuffer,
                                                        idxBuffer: prepared.idxBuffer,
                                                        triCount: prepared.triCount)
            prepared.accel = structure
            prepared.cmd = cmd
            prepared.buildFailed = failed
            cmd?.waitUntilCompleted()
            // v4.10.2: never install a failed first build —
            // keep the (empty) prior state; a later rebuild
            // retries through this same branch.
            if staticStageFailed(prepared) {
                noteWeatherStep("rebuild:stageError")
                return
            }
            swapStatic(prepared)
            return
        }
        if pendingStatic == nil {
            kickStatic(&prepared)
        } else {
            deferredStatic = prepared
            presentationDirty = true
        }
    }

    /// Atomically install a staged static state as the live
    /// state: GPU buffers + AS + CPU mirrors + counts + flags
    /// all move together, so no frame (and no pick) can ever
    /// observe a mix of old and new. The dynamic layer's
    /// material base, the collision export, the material table
    /// and the emissive index are refreshed against the new
    /// state here, in the same order the old synchronous
    /// rebuild used.
    private func swapStatic(_ p: PreparedStatic) {
        noteWeatherStep("rebuild:swap")
        // v4.10.2 note (Lane-B F3): dropping the old AS and
        // buffers here has no explicit fence — GPU safety
        // rests on Metal's guarantee that a committed command
        // buffer retains every resource its encoders
        // referenced until it completes, plus the fact that
        // the static buffers are never CPU-written after
        // creation. Any future CPU write into a static buffer
        // or CPU-side AS query voids that reliance.
        accel = p.accel
        posBuffer = p.posBuffer
        nrmBuffer = p.nrmBuffer
        uvBuffer = p.uvBuffer
        colBuffer = p.colBuffer
        idxBuffer = p.idxBuffer
        triMatBuffer = p.triMatBuffer
        triCount = p.triCount
        cpuPositions = p.cpuPositions
        cpuIndices = p.cpuIndices
        cpuTriObject = p.cpuTriObject
        cpuTriMats = p.cpuTriMats
        cachedObjectOrder = p.objectOrder
        cachedStaticMaterials = p.materials
        sceneHasSheetMaterial = p.sheet
        sceneHasSnowMaterial = p.snow
        staticEverSwapped = true
        _collisionTriangles = computeCollisionTriangles()
        // The dynamic material base may have moved with the new
        // static count; the dynamic layer itself survives.
        refreshDynTriMats()
        let fullMats = fullMaterialTable()
        uploadEmissiveIndexGated(materials: fullMats, rescanStatic: true,
                                 geometryUnchanged: false)
        uploadMaterialTable(fullMats)
        flushMaterialTable()
        resetAccumulation()
    }

    /// Frame-boundary swap check (top of draw): if the pending
    /// build has completed, swap it in — unless a newer stage
    /// was deferred behind it, in which case the completed
    /// build is discarded unswapped and the newest is kicked.
    private func swapStaticIfReady() {
        guard let pending = pendingStatic else {
            if let d = deferredStatic {
                deferredStatic = nil
                var d = d
                kickStatic(&d)
            }
            return
        }
        // v4.10.2: an errored build is FINISHED too — round 1
        // tested only .completed, so an errored pending build
        // wedged this check forever and no sheet ever swapped
        // in. A failed stage is discarded unswapped (the
        // previous live state keeps presenting) and the
        // deferred stage, if any, is kicked in its place.
        let status = pending.cmd?.status
        let done = pending.cmd == nil || status == .completed || status == .error
        guard done else { return }
        pendingStatic = nil
        if let d = deferredStatic {
            deferredStatic = nil
            var d = d
            kickStatic(&d)
            if staticStageFailed(pending) {
                noteWeatherStep("rebuild:stageError")
            }
            return
        }
        if staticStageFailed(pending) {
            noteWeatherStep("rebuild:stageError")
            return
        }
        swapStatic(pending)
    }

    /// Synchronously complete any pending static build + swap.
    /// Capture/export paths (stills, thumbnails) call this so
    /// they always render the latest staged geometry — the
    /// same guarantee the old synchronous rebuild gave them.
    func flushPendingStaticBuild() {
        while pendingStatic != nil || deferredStatic != nil {
            if pendingStatic == nil, let d = deferredStatic {
                deferredStatic = nil
                var d = d
                kickStatic(&d)
                continue
            }
            guard let pending = pendingStatic else { break }
            pending.cmd?.waitUntilCompleted()
            pendingStatic = nil
            // v4.10.2: waitUntilCompleted returns for an
            // ERRORED buffer exactly as for a completed one —
            // round 1 never checked, and this flush (called by
            // save thumbnails, PNG export, and the final
            // render) installed the never-built AS, so the
            // next trace faulted the process. A failed stage
            // is dropped unswapped; the previous live state is
            // what the capture renders.
            if staticStageFailed(pending) {
                noteWeatherStep("rebuild:stageError")
                if let d = deferredStatic {
                    deferredStatic = nil
                    var d = d
                    kickStatic(&d)
                }
                continue
            }
            if let d = deferredStatic {
                deferredStatic = nil
                var d = d
                kickStatic(&d)
                continue
            }
            swapStatic(pending)
        }
    }

    // MARK: - Dynamic layer, areas, light-culling support (v4)
    //
    // The dynamic layer is a second world-space triangle soup
    // beside the static one: dynamic entries (keyed, matrix-
    // driven) plus up to 8 particle soups, composed into CPU
    // mirrors + GPU buffers (slots 11-17) + its own AS. The
    // material table is contractual everywhere:
    //   [static object materials…, sheetMaterial if emitted,
    //    snowMaterial if emitted (v4.4),
    //    dynamic entry materials in entry order,
    //    particle soup materials in soup order,
    //    the player figure's material last (v4.3)]
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
        cachedStaticMaterials.count
    }

    /// The full GPU material table, in the contractual order.
    private func fullMaterialTable() -> [GPUMaterial] {
        var mats = cachedStaticMaterials
        for entry in dynEntries { mats.append(entry.material) }
        for soup in particleSoupsStored { mats.append(soup.material) }
        // v4.3: the player figure's material is ALWAYS last —
        // the same order composeDynamic bakes its triangles in.
        // v4.5: the head entry's material follows the body's.
        if let fig = playerFigureEntry { mats.append(fig.material) }
        if let head = playerHeadEntry { mats.append(head.material) }
        return mats
    }

    /// Recompute cpuDynTriMats (+ its GPU buffer) from the
    /// stored per-tri material offsets and the current base.
    private func refreshDynTriMats() {
        let base = dynMatBase
        cpuDynTriMats = cpuDynMatOffsets.map { UInt32(base + $0) }
        // v4.9: persistent shared buffer, rewritten in place
        // only when the values actually changed.
        var reallocated = false
        if dynTriMatBuffer == nil || dynTriMatCapacity < cpuDynTriMats.count {
            dynTriMatBuffer = device.makeBuffer(
                length: max(cpuDynTriMats.count * 4, 4), options: .storageModeShared)
            dynTriMatCapacity = (dynTriMatBuffer != nil) ? cpuDynTriMats.count : 0
            dynTriMatUploaded = []
            reallocated = true
        }
        if let buf = dynTriMatBuffer, dynTriMatUploaded != cpuDynTriMats,
           !cpuDynTriMats.isEmpty {
            if !reallocated { waitForPreviousFrame() }
            let ptr = buf.contents().bindMemory(to: UInt32.self,
                                                capacity: dynTriMatCapacity)
            for i in cpuDynTriMats.indices { ptr[i] = cpuDynTriMats[i] }
            dynTriMatUploaded = cpuDynTriMats
        }
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
        // v4.9 (item 5): cheap — stage the table and let the
        // single flush point (top of draw) write only the
        // changed slots; the emissive rescan is signature-gated
        // (a snow-factor pad change moves no emissive byte, so
        // a per-entry snow trigger no longer rescans anything).
        // v4.1 (F4ii): only a dynamic material changed — reuse
        // the cached static emissive scan when a rescan IS due.
        let fullMats = fullMaterialTable()
        uploadMaterialTable(fullMats)
        uploadEmissiveIndexGated(materials: fullMats, rescanStatic: false,
                                 geometryUnchanged: true)
        resetAccumulation()
    }

    func removeDynamicEntry(key: UUID) {
        guard dynEntries.contains(where: { $0.key == key }) else { return }
        dynEntries.removeAll { $0.key == key }
        dynDirtyKeys.remove(key)
        composeDynamic(structureChanged: true)
    }

    // MARK: Player figure (v4.3)
    //
    // The figure is a reserved DynEntry under playerFigureKey,
    // stored separately from dynEntries (so setDynamicEntries
    // can never remove it) and composed LAST, after the
    // particle soups, in geometry and the material table.
    // Its material is plain white with flags 0 — the mesh's
    // vertex colours carry the look — and pad.x = 1.0 so it
    // never loses snow shading. objectID is nil: pick() maps
    // its triangles through the unpickable sentinel.

    /// Install, replace, or (with mesh = nil) remove the
    /// player figure. A first install (or a mesh whose vertex
    /// count no longer matches the baked slice — outside the
    /// PlayerModel constant-count contract, kept as a safety
    /// net) composes the dynamic layer; a same-count pose
    /// swap only marks the figure's slice dirty and
    /// flushDynamic() re-bakes its vertices at the next draw.
    /// v4.5: the mesh is split (PlayerModel.splitHead) into a
    /// BODY entry under playerFigureKey (material flags 0, as
    /// before) and a HEAD entry under playerHeadKey whose
    /// material carries FLAG_PLAYER_HEAD — same matrix, posed
    /// in lockstep, composed immediately after the body.
    func setPlayerFigure(mesh: MeshData?, matrix: float4x4) {
        guard let mesh else {
            if playerFigureEntry != nil || playerHeadEntry != nil {
                playerFigureEntry = nil
                playerHeadEntry = nil
                playerFigureFullMesh = nil
                dynDirtyKeys.remove(playerFigureKey)
                dynDirtyKeys.remove(playerHeadKey)
                composeDynamic(structureChanged: true)
            }
            return
        }
        playerFigureFullMesh = mesh
        let split = PlayerModel.splitHead(mesh)
        var needsCompose = false
        needsCompose = upsertFigurePart(
            entry: &playerFigureEntry, key: playerFigureKey,
            mesh: split.body, matrix: matrix, flags: 0) || needsCompose
        needsCompose = upsertFigurePart(
            entry: &playerHeadEntry, key: playerHeadKey,
            mesh: split.head, matrix: matrix, flags: flagPlayerHead) || needsCompose
        if needsCompose { composeDynamic(structureChanged: true) }
    }

    /// One half of the split figure: same-count pose swaps
    /// mark the slice dirty (returns false); a first install
    /// or count mismatch asks the caller to compose (true).
    private func upsertFigurePart(entry: inout DynEntry?, key: UUID,
                                  mesh: MeshData, matrix: float4x4,
                                  flags: UInt32) -> Bool {
        if var existing = entry {
            existing.mesh = mesh
            existing.matrix = matrix
            entry = existing
            if let slice = dynSlices[key],
               slice.vertexCount == mesh.vertexCount, !slice.remapped {
                dynDirtyKeys.insert(key)
                return false
            }
            return true
        }
        var mat = GPUMaterial()
        mat.baseColor = SIMD4<Float>(1, 1, 1, 0)
        mat.emissive = SIMD4<Float>(0, 0, 0, 0)
        mat.glassParams = SIMD4<Float>(0, 0, 1.5, 0)
        mat.tint = SIMD4<Float>(1, 1, 1, 1)
        mat.texIndex = 0
        mat.flags = flags
        mat.pad = SIMD2<UInt32>(0x3F800000, 0) // exposure 1.0
        entry = DynEntry(key: key, mesh: mesh, matrix: matrix,
                         material: mat, objectID: nil)
        return true
    }

    /// Cheap figure move: update the matrix only and mark the
    /// slices dirty — the re-bake happens in flushDynamic() at
    /// the top of draw(), exactly like updateDynamicMatrix.
    func setPlayerFigureMatrix(_ m: float4x4) {
        guard playerFigureEntry != nil else { return }
        playerFigureEntry?.matrix = m
        dynDirtyKeys.insert(playerFigureKey)
        playerHeadEntry?.matrix = m
        if playerHeadEntry != nil { dynDirtyKeys.insert(playerHeadKey) }
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
        swapStaticIfReady()
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
        // v4.3: the player figure composes LAST (after the
        // particle soups) — the one consistent order used by
        // fullMaterialTable, the pick mapping and the emissive
        // scan. objectID nil → the sentinel → unpickable.
        // v4.5: two soups — body, then head — with consecutive
        // material offsets matching fullMaterialTable's order.
        if !capped, let fig = playerFigureEntry {
            let world = fig.mesh.baked(by: fig.matrix)
            slices[fig.key] = appendSoup(
                world: world,
                matOffset: dynEntries.count + particleSoupsStored.count,
                objectID: nil)
            if let head = playerHeadEntry {
                let headWorld = head.mesh.baked(by: head.matrix)
                slices[head.key] = appendSoup(
                    world: headWorld,
                    matOffset: dynEntries.count + particleSoupsStored.count + 1,
                    objectID: nil)
            }
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

        // v4.9 (1a): position/normal buffers are persistent —
        // (re)allocated only when the composed vertex count
        // outgrows them, otherwise rewritten in place (fenced
        // on the previous frame when reusing live buffers).
        let dynVerts = positions.count / 3
        if dynVertCapacity < dynVerts || dynPosBuffer == nil || dynNrmBuffer == nil {
            dynPosBuffer = device.makeBuffer(length: max(dynVerts * 12, 4),
                                             options: .storageModeShared)
            dynNrmBuffer = device.makeBuffer(length: max(dynVerts * 12, 4),
                                             options: .storageModeShared)
            dynVertCapacity = (dynPosBuffer != nil && dynNrmBuffer != nil) ? dynVerts : 0
        } else {
            waitForPreviousFrame()
        }
        if let b = dynPosBuffer, !positions.isEmpty {
            let p = b.contents().bindMemory(to: Float.self, capacity: dynVerts * 3)
            positions.withUnsafeBufferPointer { src in
                if let s = src.baseAddress { p.update(from: s, count: positions.count) }
            }
        }
        if let b = dynNrmBuffer, !normals.isEmpty {
            let p = b.contents().bindMemory(to: Float.self, capacity: dynVerts * 3)
            normals.withUnsafeBufferPointer { src in
                if let s = src.baseAddress { p.update(from: s, count: normals.count) }
            }
        }
        dynUVBuffer = makeBuffer(uvArr)
        dynColBuffer = makeBuffer(colArr)
        dynIdxBuffer = makeBuffer(indices)
        refreshDynTriMats()
        buildDynamicAccelerationStructure()
        let fullMats = fullMaterialTable()
        uploadMaterialTable(fullMats)
        // v4.1 (F4ii): compose changes only dynamic content —
        // reuse the cached static emissive scan.
        uploadEmissiveIndexGated(materials: fullMats, rescanStatic: false,
                                 geometryUnchanged: false)
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
        var baked: [(slice: DynSlice, world: MeshData)] = []
        for key in dirty {
            // v4.3: the player figure flushes through the same
            // path — its key resolves to the reserved entry
            // when it is not a session dynamic entry.
            guard let pair = dynEntries.first(where: { $0.key == key })
                    ?? (key == playerFigureKey ? playerFigureEntry : nil)
                    ?? (key == playerHeadKey ? playerHeadEntry : nil),
                  let slice = dynSlices[key], slice.vertexCount > 0 else { continue }
            let world = pair.local.baked(by: pair.matrix)
            guard world.vertexCount == slice.vertexCount, !slice.remapped else { continue }
            let start = slice.vertexStart * 3
            let count = slice.vertexCount * 3
            if start + count <= cpuDynPositions.count {
                cpuDynPositions.replaceSubrange(start..<(start + count), with: world.positions)
                cpuDynNormals.replaceSubrange(start..<(start + count), with: world.normals)
                baked.append((slice: slice, world: world))
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
        // v4.9 (1a): write ONLY the dirty slices into the
        // persistent position/normal buffers, then REFIT the
        // dynamic AS — a matrix-only flush changes no topology,
        // buffer, or triangle count, so refit is the designed
        // operation (same triangles => same intersection
        // results; only BVH organisation may differ, exactly as
        // it did between v4.8's per-flush full rebuilds). The
        // refit command buffer commits on the serial queue
        // BEFORE this frame's trace command buffer, so no wait
        // is needed on it. The CPU slice writes are the one
        // real hazard: they are fenced on the PREVIOUS frame's
        // command buffer (never the one about to be encoded).
        waitForPreviousFrame()
        if let posB = dynPosBuffer, let nrmB = dynNrmBuffer,
           dynVertCapacity * 3 >= cpuDynPositions.count {
            let posPtr = posB.contents().bindMemory(to: Float.self,
                                                    capacity: dynVertCapacity * 3)
            let nrmPtr = nrmB.contents().bindMemory(to: Float.self,
                                                    capacity: dynVertCapacity * 3)
            for (slice, world) in baked {
                let start = slice.vertexStart * 3
                let count = slice.vertexCount * 3
                // v4.10.1 hardening: the CPU-mirror write above
                // bounds-checks start + count; the GPU write now
                // checks the same range against the persistent
                // buffers' capacity before touching the pointers
                // (a stale slice could otherwise write past the
                // allocation while its AS refit is in flight).
                guard start >= 0, count >= 0,
                      start + count <= dynVertCapacity * 3 else { continue }
                world.positions.withUnsafeBufferPointer { src in
                    if let s = src.baseAddress {
                        (posPtr + start).update(from: s, count: count)
                    }
                }
                world.normals.withUnsafeBufferPointer { src in
                    if let s = src.baseAddress {
                        (nrmPtr + start).update(from: s, count: count)
                    }
                }
            }
            // LS-REPRO TOGGLE `norefit`: skip the in-place
            // refit and always take the full dynamic AS build.
            // Default OFF = shipped behaviour.
            if ReproToggles.noRefit || !refitDynamicAccelerationStructure() {
                // Refit unavailable (no AS, allocation failure):
                // the v4.8 full build remains as the fallback.
                buildDynamicAccelerationStructure()
            }
        } else {
            // Persistent buffers missing/undersized (compose
            // normally owns them): fresh upload + full build.
            dynPosBuffer = makeBuffer(cpuDynPositions)
            dynNrmBuffer = makeBuffer(cpuDynNormals)
            dynVertCapacity = cpuDynPositions.count / 3
            buildDynamicAccelerationStructure()
        }
        // v4.1 (F5): moved dynamic geometry moves its collision
        // triangles with it. v4.9: only the dirty entries'
        // triangle ranges are recomputed (the collision array
        // mirrors the dynamic slices at triCount + triStart);
        // every other entry is byte-identical by construction.
        patchCollisionTriangles(dirtySlices: baked.map { $0.slice })
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

    /// v4.9 (1a): recompute ONLY the given dynamic slices'
    /// ranges of the collision export. The array layout is
    /// [static tris..., dynamic tris...], mirroring the
    /// dynamic slices at triCount + slice.triStart — patching
    /// those ranges from the same CPU mirrors produces exactly
    /// what computeCollisionTriangles() would return.
    private func patchCollisionTriangles(dirtySlices: [DynSlice]) {
        guard _collisionTriangles.count == triCount + dynTriCount else {
            _collisionTriangles = computeCollisionTriangles()
            return
        }
        for slice in dirtySlices where slice.triCount > 0 {
            for dt in slice.triStart..<(slice.triStart + slice.triCount) {
                let ia = Int(cpuDynIndices[3 * dt])
                let ib = Int(cpuDynIndices[3 * dt + 1])
                let ic = Int(cpuDynIndices[3 * dt + 2])
                _collisionTriangles[triCount + dt] = (
                    SIMD3<Float>(cpuDynPositions[3 * ia], cpuDynPositions[3 * ia + 1], cpuDynPositions[3 * ia + 2]),
                    SIMD3<Float>(cpuDynPositions[3 * ib], cpuDynPositions[3 * ib + 1], cpuDynPositions[3 * ib + 2]),
                    SIMD3<Float>(cpuDynPositions[3 * ic], cpuDynPositions[3 * ic + 1], cpuDynPositions[3 * ic + 2]))
            }
        }
    }

    /// The dynamic layer's AS descriptor — ONE builder shared
    /// by the full build and the refit so their geometry can
    /// never drift apart. usage includes .refit: the AS is
    /// created refit-capable so flushDynamic can refit it.
    private func dynamicASDescriptor() -> MTLPrimitiveAccelerationStructureDescriptor? {
        guard let dynPosBuffer, let dynIdxBuffer else { return nil }
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
        desc.usage = [.refit]
        return desc
    }

    /// v4.9 (1a): refit the dynamic AS in place after a
    /// matrix-only flush. No waitUntilCompleted: the refit
    /// command buffer is committed on the renderer's serial
    /// queue before the frame's trace command buffer is even
    /// created, so queue order runs it first. Returns false
    /// when refit is unavailable and the caller must fall back
    /// to the full build.
    private func refitDynamicAccelerationStructure() -> Bool {
        guard rayTracingSupported, dynTriCount > 0,
              let dynAccel, let desc = dynamicASDescriptor() else { return false }
        let sizes = device.accelerationStructureSizes(descriptor: desc)
        if dynRefitScratch == nil || dynRefitScratchSize < sizes.refitScratchBufferSize {
            dynRefitScratch = device.makeBuffer(
                length: max(sizes.refitScratchBufferSize, 4), options: .storageModePrivate)
            dynRefitScratchSize = (dynRefitScratch != nil) ? sizes.refitScratchBufferSize : 0
        }
        guard let scratch = dynRefitScratch,
              let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeAccelerationStructureCommandEncoder() else { return false }
        enc.refit(sourceAccelerationStructure: dynAccel,
                  descriptor: desc,
                  destinationAccelerationStructure: dynAccel,
                  scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.endEncoding()
        cmd.commit()
        return true
    }

    private func buildDynamicAccelerationStructure() {
        dynAccel = nil
        dynUsable = false
        guard rayTracingSupported, dynTriCount > 0,
              let desc = dynamicASDescriptor() else { return }
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
        u.snowLevel = 0 // v4.13: snow removed; field kept for layout
        u.dynTime = weatherDynTime
        u.rainFrac = weatherDynRain
        // v4.1 (F9): an unusable dynamic layer reports 0 tris
        // to BOTH kernels, so a failed AS/buffer can never be
        // traversed against the dynamic attribute arrays.
        u.dynTriCount = UInt32(dynUsable ? dynTriCount : 0)
        // v4.5: Play-eye head exclusion, forwarded to both
        // kernels through the one uniforms() both consume.
        u.hidePlayerHead = hidePlayerHead ? 1 : 0
        u.freezeFrac = 0 // v4.13: puddle freeze removed with the 3D water
        u.waterReflScale = waterReflectionScale
        u.frameCounter = reflFrameCounter
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
    /// display texture (display-only — the accumulation, the
    /// settled frame and stills are never filtered; see the
    /// kernel's header comment in Shaders.metal).
    private func encodeDenoise(accum: MTLTexture, guide: MTLTexture,
                               guideNormal: MTLTexture, output: MTLTexture,
                               strength: Float, fadeEnd: Float, fadeEndSpec: Float,
                               camera: CameraBasis,
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
        u.fadeEnd = fadeEnd
        u.fadeEndSpec = fadeEndSpec
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
                pu.time = Float((CACurrentMediaTime() - precipTimeOffset).truncatingRemainder(dividingBy: 3600))
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
                // v4.10 (WS2 steering): remember the pass just
                // encoded — the final render's rain drops are
                // rebuilt from this exact snapshot.
                lastPrecipTime = pu.time
                lastPrecipIntensity = pu.intensity
                lastPrecipWind = pu.windXZ
                lastPrecipCount = particleCount
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
            p.shimmerRain = (weatherMode == 1 ? weatherDynRain : 0)
            p.wetLevel = weatherWet
        }
        return p
    }

    /// One frame for the live MTKView. Returns true when another
    /// frame would still change the picture (the caller chains the
    /// next draw only then — a converged, untouched scene sleeps).
    @discardableResult
    func draw(view: MTKView, camera: CameraBasis, moving: Bool, play: Bool) -> Bool {
        // v4.9 (1b/5): frame boundary — swap any completed
        // static build atomically BEFORE anything encodes
        // against renderer state, then flush staged material
        // slots, then the dynamic layer.
        swapStaticIfReady()
        flushMaterialTable()
        flushDynamic()
        // v4.13: the weather sim is removed; the reflection
        // pattern counter still advances (uniform layout kept).
        reflFrameCounter &+= 1
        // v4.2: the v4.1 framesSinceMove denoise bookkeeping is
        // gone — the pass now keys off this frame having traced
        // (see the display-source selection below) and the
        // per-pixel fade handles young pixels directly.
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
        // v3.3: rain traces exactly like snow. The v3.2
        // `rippling` special case (settled path disabled +
        // history reuse capped at 12 while it rained) is gone —
        // ripples are a display-time shimmer in the tonemap
        // pass now, so a rainy scene settles and sleeps like
        // any other.
        let weatherLevelsQuiet = abs(weatherWet - lastTracedWet) <= 0.004
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
            lastFrameCommandBuffer = cmd
            wasMoving = moving
            lastWet = weatherWet
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

        // Display source (v4.2 rework): any frame that TRACED
        // goes through the spatial denoise pass; the settled
        // frame never reaches here (the settled fast path above
        // presents the raw accumulation and returns), and a
        // static re-present (nothing traced) shows raw too.
        // The v3.1/v4.1 gate (pass off at 160 global samples)
        // existed because the old fade was zero past 96 samples
        // anyway; with the v4.2 class-aware fade the pass earns
        // its keep across the whole settle tail — per-pixel
        // alpha decides each pixel's blend, so mature pixels
        // are untouched while young/noisy ones are cleaned.
        // Cost: one bilateral pass per traced frame (it already
        // ran on every moving/Play frame); at settle the frame
        // chain stops, so the resting cost is zero.
        var displaySource = curAccum
        if accel != nil, traced,
           denoisePipeline != nil, let displayTex, let curGuideNormal {
            let strength: Float = (play || moving) ? 1.0 : 0.8
            // Hand-off ends scale with the tier's settle cap:
            // diffuse hands over at 60% of the cap, the
            // reflection class rides the denoise to the cap
            // itself (its fade is exactly 0 there, and past
            // the cap the settled path shows raw regardless).
            let capF = Float(t.maxSettle)
            if encodeDenoise(accum: curAccum, guide: curGuide,
                             guideNormal: curGuideNormal, output: displayTex,
                             strength: strength,
                             fadeEnd: capF * 0.6, fadeEndSpec: capF,
                             camera: camera, cmd: cmd) {
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
        lastFrameCommandBuffer = cmd
        wasMoving = moving
        lastWet = weatherWet
        guard accel != nil else { return false }
        return moving || weatherMode != 0 || (traced && sampleCount < t.maxSettle)
    }

    // MARK: v4.10 Final render (WS2)

    /// Shared BGRA readback target for still tonemapping.
    private func makeStillTarget(width: Int, height: Int) -> MTLTexture? {
        let targetDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        targetDesc.usage = [.renderTarget, .shaderRead]
        targetDesc.storageMode = .shared
        return device.makeTexture(descriptor: targetDesc)
    }

    /// Tonemap `source` (an accumulation-format texture whose
    /// alpha is its sample count) into `target` and read it back
    /// as a CGImage. This is renderStillPNG's tonemap + readback
    /// tail, factored out so the final render can snapshot its
    /// progressive accumulation after every chunk and produce
    /// its final image through exactly the same path.
    private func cgImageByTonemapping(source: MTLTexture, samples: Int,
                                      width: Int, height: Int,
                                      target: MTLTexture) -> CGImage? {
        guard let tonemapPipeline, let cmd = queue.makeCommandBuffer() else { return nil }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0].storeAction = .store
        guard let renc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return nil }
        var params = TonemapParams(samples: UInt32(max(samples, 1)), exposure: exposure)
        renc.setRenderPipelineState(tonemapPipeline)
        renc.setFragmentTexture(source, index: 0)
        // Inert placeholders, exactly as in renderStillPNG: the
        // fragment's shimmer block is gated on shimmerRain > 0
        // (0 here) and never samples indices 1-3 for stills.
        renc.setFragmentTexture(weatherFieldTex, index: 1)
        renc.setFragmentTexture(weatherFieldTex, index: 2)
        renc.setFragmentTexture(weatherFieldTex, index: 3)
        renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
        renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
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
        return CGImage(width: width, height: height, bitsPerComponent: 8,
                       bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: cs, bitmapInfo: bitmapInfo,
                       provider: provider, decode: nil,
                       shouldInterpolate: false, intent: .defaultIntent)
    }

    /// Encode a CGImage as PNG (renderStillPNG's tail encoding,
    /// shared with the session's final-render output actions).
    func pngData(from image: CGImage) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }

    /// v4.10 (WS2 steering): refractive rain drops for the
    /// final render — HOW MANY and HOW BIG, stated plainly:
    /// at most 384 drops (the nearest to the camera of the
    /// drops the live overlay was actually drawing), each a
    /// 36-triangle lathe (6 sides, 3 rings + 2 poles) at the
    /// live streak's own size (0.6 m long, 0.032 m wide), so a
    /// full drop set is 13,824 triangles — inside the dynamic
    /// layer's 90,000-triangle budget alongside the scene's
    /// own dynamic entries. Positions are a Swift port of
    /// precipVertex's placement for mode 1, evaluated with
    /// the LAST ENCODED pass's time / intensity / wind /
    /// draw count (the frozen view the user is looking at)
    /// around the render camera, keeping only drops the
    /// overlay would show (centre in front, z >= 0.3, and
    /// within the 30 m far-fade distance where its fade
    /// reaches zero). Snow mode gets NO geometry (rain only).
    private static let finalRenderDropCap = 384

    private func finalRenderRainDropMesh(camera: CameraBasis) -> MeshData? {
        guard weatherMode == 1, let time = lastPrecipTime,
              !precipSeeds.isEmpty else { return nil }
        let count = min(lastPrecipCount, precipSeeds.count, 1600)
        guard count > 0 else { return nil }
        let box = SIMD3<Float>(36, 22, 36)
        let wind = SIMD3<Float>(lastPrecipWind.x, 0, lastPrecipWind.y)
        var drops: [(centre: SIMD3<Float>, speed: Float, dist: Float)] = []
        for i in 0..<count {
            let seed = precipSeeds[i]
            // precipVertex (rain): speed from the seed's w.
            let speed = (10 + seed.w * 4) * (0.85 + 0.30 * lastPrecipIntensity)
            var p = SIMD3<Float>(seed.x * box.x, seed.y * box.y, seed.z * box.z)
            p.y = (p.y - time * speed).truncatingRemainder(dividingBy: box.y)
            if p.y < 0 { p.y += box.y }
            // Wind carry, rain factor 0.25 (precipVertex).
            p.x = (p.x + time * wind.x * 0.25).truncatingRemainder(dividingBy: box.x)
            if p.x < 0 { p.x += box.x }
            p.z = (p.z + time * wind.z * 0.25).truncatingRemainder(dividingBy: box.z)
            if p.z < 0 { p.z += box.z }
            let centre = SIMD3<Float>(camera.origin.x + (p.x - box.x * 0.5),
                                      camera.origin.y - 2 + p.y,
                                      camera.origin.z + (p.z - box.z * 0.5))
            let rel = centre - camera.origin
            let dist = simd_length(rel)
            guard simd_dot(rel, camera.forward) >= 0.3, dist <= 30 else { continue }
            drops.append((centre, speed, dist))
        }
        drops.sort { $0.dist < $1.dist }
        let chosen = drops.prefix(Self.finalRenderDropCap)
        guard !chosen.isEmpty else { return nil }
        // One lathe per drop, baked in world space. Winding
        // follows MeshGen's cylinder verbatim (bands
        // (a,b,c)/(b,d,c) along the axis, cap fans
        // (pole,k,k+1) at the tail and (pole,k+1,k) at the
        // head), so front-facing determination matches the
        // shipped glass primitives exactly.
        var mesh = MeshData()
        let radial = 6
        let halfLen: Float = 0.3
        let maxR: Float = 0.016
        // Ring profile along the fall axis: (offset, radius
        // factor) — a stretched teardrop, fuller at the
        // leading (lower, +axis) end.
        let profile: [(s: Float, r: Float)] = [(-0.18, 0.70), (0.0, 1.0), (0.18, 0.74)]
        for drop in chosen {
            var axis = SIMD3<Float>(wind.x, -drop.speed, wind.z)
            axis = simd_normalize(axis)
            let helper = abs(axis.y) < 0.9 ? SIMD3<Float>(0, 1, 0) : SIMD3<Float>(1, 0, 0)
            let side = simd_normalize(simd_cross(helper, axis))
            let other = simd_normalize(simd_cross(axis, side))
            let pole0 = mesh.vertexCount
            mesh.addVertex(drop.centre - axis * halfLen, -axis, SIMD2<Float>(0, 0))
            var ringStart: [Int] = []
            for prof in profile {
                ringStart.append(mesh.vertexCount)
                for k in 0..<radial {
                    let th = Float(k) / Float(radial) * 2 * .pi
                    let dir = side * cos(th) + other * sin(th)
                    mesh.addVertex(drop.centre + axis * prof.s + dir * (maxR * prof.r),
                                   dir, SIMD2<Float>(0, 0))
                }
            }
            let pole1 = mesh.vertexCount
            mesh.addVertex(drop.centre + axis * halfLen, axis, SIMD2<Float>(0, 0))
            for k in 0..<radial {
                let k1 = (k + 1) % radial
                // Tail fan (cylinder bottom-cap convention).
                mesh.indices += [UInt32(pole0), UInt32(ringStart[0] + k), UInt32(ringStart[0] + k1)]
                // Head fan (cylinder top-cap convention).
                let last = ringStart[ringStart.count - 1]
                mesh.indices += [UInt32(pole1), UInt32(last + k1), UInt32(last + k)]
            }
            for band in 0..<(ringStart.count - 1) {
                let a0 = ringStart[band], b0 = ringStart[band + 1]
                for k in 0..<radial {
                    let k1 = (k + 1) % radial
                    let a = a0 + k, b = b0 + k, c = a0 + k1, d = b0 + k1
                    mesh.indices += [UInt32(a), UInt32(b), UInt32(c),
                                     UInt32(b), UInt32(d), UInt32(c)]
                }
            }
        }
        return mesh
    }

    /// The drop material: a glass-class GPUMaterial the
    /// EXISTING transmission machinery traces (FLAG_GLASS,
    /// transmission 1, roughness 0, IOR 1.33 — water — and a
    /// white tint, so Beer–Lambert absorption is zero and the
    /// drop is clear water). No new shader path exists.
    private func finalRenderDropMaterial() -> GPUMaterial {
        var g = GPUMaterial()
        g.baseColor = SIMD4<Float>(1, 1, 1, 0)
        g.emissive = SIMD4<Float>(0, 0, 0, 0)
        g.glassParams = SIMD4<Float>(1, 0, 1.33, 0)
        g.tint = SIMD4<Float>(1, 1, 1, 1)
        g.texIndex = 0
        g.flags = 2 // FLAG_GLASS
        g.pad = SIMD2<UInt32>(0x3F800000, 0) // exposure 1.0
        return g
    }

    /// v4.10 (WS2): progressive offline-quality still — the
    /// "Blender render" of the current view. Built on
    /// renderStillPNG's pattern: pending static builds and the
    /// material table are flushed first, the FULL emissive set
    /// is force-restored for the whole render (light culling
    /// off) and re-activated by defer on every exit path, and
    /// samples accumulate chunk by chunk into ONE accumulation
    /// texture via repeated encodeTrace calls with advancing
    /// seedStart (encodeTrace dispatches `count` samples with
    /// seeds seedStart+i into the existing accum). One command
    /// buffer per chunk; after each chunk the accumulation is
    /// tonemapped into the shared BGRA target and reported via
    /// onProgress as a CGImage snapshot. shouldCancel() is
    /// checked between chunks, so a cancel is bounded by one
    /// chunk (128 samples) and keeps the best image so far.
    /// At finish — complete OR cancelled — the accumulation is
    /// denoised (see below) and tonemapped for the returned
    /// image; if the denoise textures/pipeline are unavailable
    /// the raw accumulation is tonemapped instead (fallback
    /// path). Returns nil only on hard failure (no textures,
    /// no guide, or the first trace chunk fails to encode).
    func renderFinalImage(camera: CameraBasis, width: Int, height: Int,
                          totalSamples: Int, bounces: UInt32, glassBudget: UInt32,
                          chunkSamples: Int, shouldCancel: () -> Bool,
                          onProgress: (Int, CGImage?) -> Void) -> CGImage? {
        flushPendingStaticBuild()
        flushMaterialTable()
        guard width > 0, height > 0, totalSamples > 0,
              let accum = makeAccumTexture(width: width, height: height),
              let guide = makeGuideTexture(width: width, height: height),
              let guideNormal = makeGuideNormalTexture(width: width, height: height),
              let target = makeStillTarget(width: width, height: height)
        else { return nil }
        // Full emissive set for the whole render, exactly as
        // renderStillPNG: force-restore now, defer re-activate
        // the culled set on every exit path (after the waits
        // below on success paths; uncommitted on failure paths).
        let wasCulling = cullingActive
        if wasCulling { restoreFullEmissive() }
        defer { if wasCulling { activateCulledEmissive() } }
        // v4.10 (WS2 steering): TRUE REFRACTIVE RAIN. When the
        // effective weather mode is rain, the drops are built
        // as real glass-class geometry in the DYNAMIC layer
        // (its own AS, which encodeTrace / encodeGuide already
        // intersect alongside the static AS, with materials
        // from the same table) for the duration of this render
        // only. They are static across every sample — the
        // placement snapshot never changes during the loop —
        // so each drop refracts the scene behind it at water
        // IOR 1.33 through the existing transmission path
        // (the render's glassBudget of 16 covers the two
        // crossings per drop plus scene glass). RESTORE
        // DISCIPLINE: the pre-render dynamic entry list is
        // saved here and restored verbatim by the defer below
        // (LIFO: drops come out before the culling restore),
        // recomposing the dynamic layer to exactly its
        // pre-render state on EVERY exit path — success,
        // cancel, or failure. The only lasting side effect is
        // the accumulation reset composeDynamic performs,
        // the same transient the culling restore already
        // causes in renderStillPNG: the live view simply
        // re-accumulates afterwards. Snow gets no geometry.
        let savedDynEntriesForDrops = dynEntries
        var dropsInstalled = false
        if let dropMesh = finalRenderRainDropMesh(camera: camera) {
            dynEntries = savedDynEntriesForDrops + [DynEntry(
                key: UUID(), mesh: dropMesh, matrix: float4x4(),
                material: finalRenderDropMaterial(), objectID: nil)]
            composeDynamic(structureChanged: true)
            // composeDynamic only STAGES the material table —
            // flush it so the drop's glass material slot is in
            // the buffer encodeTrace actually reads (no draw
            // intervenes during a still to flush it for us).
            flushMaterialTable()
            dropsInstalled = true
        }
        defer {
            if dropsInstalled {
                dynEntries = savedDynEntriesForDrops
                composeDynamic(structureChanged: true)
                flushMaterialTable()
            }
        }
        // Guide pass once (the camera never moves during a
        // still) + a cleared accumulation to start from.
        guard let guideCmd = queue.makeCommandBuffer() else { return nil }
        guard encodeGuide(guide: guide, guideNormal: guideNormal,
                          camera: camera, cmd: guideCmd) else { return nil }
        clearAccum(accum, cmd: guideCmd)
        guideCmd.commit()
        guideCmd.waitUntilCompleted()
        var done = 0
        let chunkSize = max(1, chunkSamples)
        while done < totalSamples {
            if done > 0, shouldCancel() { break }
            let chunk = min(chunkSize, totalSamples - done)
            guard let cmd = queue.makeCommandBuffer() else { return nil }
            let traced = encodeTrace(accum: accum, camera: camera, count: chunk,
                                     seedStart: UInt32(done), bounces: bounces,
                                     cap: 1_000_000, glassBudget: glassBudget,
                                     cmd: cmd)
            guard traced else { return done > 0 ? cgImageByTonemapping(
                source: accum, samples: done, width: width, height: height,
                target: target) : nil }
            cmd.commit()
            cmd.waitUntilCompleted()
            done += chunk
            let snapshot = cgImageByTonemapping(source: accum, samples: done,
                                                width: width, height: height,
                                                target: target)
            onProgress(done, snapshot)
            if shouldCancel() { break }
        }
        guard done > 0 else { return nil }
        // Final denoise, mirroring the LIVE settled denoise
        // invocation in draw(): a still, non-Play frame uses
        // strength 0.8, fadeEnd = cap * 0.6, fadeEndSpec = cap
        // with the Auto settle cap (768) as the cap. (At the
        // final render's 4,096 samples the per-pixel fade in
        // the kernel has already handed over to the raw
        // accumulation for both classes, so this pass is the
        // same no-op-at-maturity the live path applies; on a
        // cancelled early render it cleans the young image
        // exactly as the live denoise would.) If the denoise
        // output texture or pipeline is unavailable, fall back
        // to tonemapping the raw accumulation.
        var finalSource = accum
        if let display = makeAccumTexture(width: width, height: height),
           let cmd = queue.makeCommandBuffer(),
           encodeDenoise(accum: accum, guide: guide, guideNormal: guideNormal,
                         output: display, strength: 0.8,
                         fadeEnd: 768 * 0.6, fadeEndSpec: 768,
                         camera: camera, cmd: cmd) {
            cmd.commit()
            cmd.waitUntilCompleted()
            finalSource = display
        }
        return cgImageByTonemapping(source: finalSource, samples: done,
                                    width: width, height: height, target: target)
    }

    /// Render the current scene offscreen and return a PNG (used for
    /// project thumbnails and PNG export). Always full quality:
    /// full bounce depth, generous glass budget, no history cap.
    func renderStillPNG(camera: CameraBasis, width: Int, height: Int, samples: Int) -> Data? {
        // v4.9 (1b/5): a still must show the LATEST staged
        // geometry and materials (the old synchronous rebuild
        // guaranteed this) — complete any pending static build
        // + swap and flush the material table first.
        flushPendingStaticBuild()
        flushMaterialTable()
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
        // v4.10.2 (Lane-B F2): snapshot the CPU mirrors into
        // locals before looping. They are value types, so the
        // loops below are immune to a static swap replacing
        // them mid-pick — same-thread swaps can't interleave,
        // but the final render drives swaps from a background
        // queue, and a smaller replacement scene would have
        // subscripted the loop out of bounds.
        let indices = cpuIndices, positions = cpuPositions, triObjects = cpuTriObject
        let dynIndices = cpuDynIndices, dynPositions = cpuDynPositions, dynTriObjects = cpuDynTriObject
        var best: Float = .greatestFiniteMagnitude
        var bestObj: UUID? = nil
        let triTotal = indices.count / 3
        for t in 0..<triTotal {
            let ia = Int(indices[3 * t]), ib = Int(indices[3 * t + 1]), ic = Int(indices[3 * t + 2])
            let p0 = SIMD3<Float>(positions[3 * ia], positions[3 * ia + 1], positions[3 * ia + 2])
            let p1 = SIMD3<Float>(positions[3 * ib], positions[3 * ib + 1], positions[3 * ib + 2])
            let p2 = SIMD3<Float>(positions[3 * ic], positions[3 * ic + 1], positions[3 * ic + 2])
            if let d = rayTriangle(origin: origin, dir: dir, p0: p0, p1: p1, p2: p2), d < best {
                best = d
                bestObj = triObjects[t]
            }
        }
        // v4: dynamic mirrors too — nearest hit overall wins. A
        // dynamic hit with the sentinel id is unpickable: it
        // still occludes (best is set) and returns nil.
        let dynTotal = dynIndices.count / 3
        for t in 0..<dynTotal {
            let ia = Int(dynIndices[3 * t]), ib = Int(dynIndices[3 * t + 1]), ic = Int(dynIndices[3 * t + 2])
            let p0 = SIMD3<Float>(dynPositions[3 * ia], dynPositions[3 * ia + 1], dynPositions[3 * ia + 2])
            let p1 = SIMD3<Float>(dynPositions[3 * ib], dynPositions[3 * ib + 1], dynPositions[3 * ib + 2])
            let p2 = SIMD3<Float>(dynPositions[3 * ic], dynPositions[3 * ic + 1], dynPositions[3 * ic + 2])
            if let d = rayTriangle(origin: origin, dir: dir, p0: p0, p1: p1, p2: p2), d < best {
                best = d
                let oid = dynTriObjects[t]
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
