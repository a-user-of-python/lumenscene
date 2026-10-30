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

import Foundation
import Metal
import MetalKit
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

struct TonemapParams {
    var samples: UInt32 = 0
    var exposure: Float = 1.15
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
    private var tonemapPipeline: MTLRenderPipelineState?

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

    // Procedural texture arrays (built once).
    private var texAlbedo: MTLTexture?
    private var texNormal: MTLTexture?
    private var texData: MTLTexture?

    // Accumulation + guide textures, ping-ponged per frame. The
    // accumulation alpha channel carries the effective sample
    // count; guides carry primary-hit world positions (rgba32F).
    private var accumTex: [MTLTexture?] = [nil, nil]
    private var guideTex: [MTLTexture?] = [nil, nil]
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
        if let vfn = library.makeFunction(name: "tonemapVertex"),
           let ffn = library.makeFunction(name: "tonemapFragment") {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfn
            desc.fragmentFunction = ffn
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            tonemapPipeline = try? device.makeRenderPipelineState(descriptor: desc)
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

    deinit {
        if let thermalObserver {
            NotificationCenter.default.removeObserver(thermalObserver)
        }
    }

    // MARK: Texture arrays

    private func buildTextureArrays() {
        let names = TextureGen.presetNames
        guard !names.isEmpty else { return }
        var albedos: [Data] = [], normals: [Data] = [], datas: [Data] = []
        var size = 0
        for name in names {
            let set = TextureGen.makeSet(preset: name)
            size = set.size
            albedos.append(set.albedo); normals.append(set.normal); datas.append(set.data)
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

    /// Texture array slice for a preset name (0 = none in materials).
    static func texIndex(forPreset preset: String) -> UInt32 {
        guard let i = TextureGen.presetNames.firstIndex(of: preset) else { return 0 }
        return UInt32(i + 1)
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
            let world = local.baked(by: obj.transform.matrix)
            let matIndex = UInt32(materials.count)
            materials.append(gpuMaterial(for: obj))
            objectOrder.append(obj.id)
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
            if previewing {
                // Manipulation preview: no light cast; the surface
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
        g.baseColor = SIMD4<Float>(base, 0)
        g.emissive = SIMD4<Float>(m.emissiveColor.simd, Float(m.emissiveIntensity))
        g.glassParams = SIMD4<Float>(Float(m.transmission), Float(m.roughness), Float(m.ior), 0)
        g.tint = SIMD4<Float>(m.tint.simd, Float(m.textureScale))
        g.texIndex = Renderer.texIndex(forPreset: m.texturePreset)
        g.flags = flags
        return g
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
            scale = 0.8; spp = 1; bounces = 4; glass = 3; maxSettle = 256
        default:
            break
        }
        if play {
            maxSettle = qualityPreference == 1 ? 128 : (qualityPreference == 2 ? 64 : 96)
        }
        var reuseCap = Float(maxSettle)
        if moving {
            scale = min(scale, qualityPreference == 2 ? 0.55 : 0.65)
            spp = qualityPreference == 2 ? 3 : 4
            bounces = min(bounces, 4)
            glass = min(glass, 2)
            reuseCap = play ? 12 : 24
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
        if moving { reuseCap = min(reuseCap, play ? 12 : 24) }
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

    private func encodeGuide(guide: MTLTexture, camera: CameraBasis,
                             cmd: MTLCommandBuffer) -> Bool {
        guard let guidePipeline, let accel,
              let posBuffer, let idxBuffer, let triMatBuffer, let matBuffer,
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(guidePipeline)
        enc.setTexture(guide, index: 0)
        enc.setBuffer(posBuffer, offset: 0, index: 1)
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
              let enc = cmd.makeComputeCommandEncoder() else { return false }
        enc.setComputePipelineState(tracePipeline)
        enc.setTexture(accum, index: 0)
        enc.setTexture(texAlbedo, index: 1)
        enc.setTexture(texNormal, index: 2)
        enc.setTexture(texData, index: 3)
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

    private func cameraChanged(_ a: CameraBasis, _ b: CameraBasis) -> Bool {
        simd_length(a.origin - b.origin) > 1e-5
            || simd_length(a.forward - b.forward) > 1e-5
            || simd_length(a.right - b.right) > 1e-5
            || simd_length(a.up - b.up) > 1e-5
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
        let pixelW = max(1, Int(Float(view.drawableSize.width) * t.scale))
        let pixelH = max(1, Int(Float(view.drawableSize.height) * t.scale))
        if accumTex[0] == nil || accumTex[0]!.width != pixelW || accumTex[0]!.height != pixelH {
            accumTex = [makeAccumTexture(width: pixelW, height: pixelH),
                        makeAccumTexture(width: pixelW, height: pixelH)]
            guideTex = [makeGuideTexture(width: pixelW, height: pixelH),
                        makeGuideTexture(width: pixelW, height: pixelH)]
            historyValid = false
            sampleCount = 0
        }
        let dst = curIdx
        let src = 1 - curIdx
        guard let curAccum = accumTex[dst], let curGuide = guideTex[dst] else { return false }

        // Settled fast path: history valid, camera identical to
        // the last presented frame, sample cap reached — just
        // present the finished image (which lives in the other
        // ping-pong slot) and sleep.
        if accel != nil, historyValid, sampleCount >= t.maxSettle,
           let pc = prevCamera, !cameraChanged(pc, camera),
           let settled = accumTex[src] {
            if let tonemapPipeline,
               let renc = cmd.makeRenderCommandEncoder(descriptor: passDesc) {
                var params = TonemapParams(samples: max(sampleCount, 1), exposure: exposure)
                renc.setRenderPipelineState(tonemapPipeline)
                renc.setFragmentTexture(settled, index: 0)
                renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
                renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                renc.endEncoding()
            }
            cmd.present(drawable)
            cmd.commit()
            wasMoving = moving
            return false
        }

        var traced = false
        if accel != nil {
            // Guide pass for the current camera (becomes next
            // frame's "previous"), then history: reproject or clear.
            _ = encodeGuide(guide: curGuide, camera: camera, cmd: cmd)
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
                _ = encodeReproject(prevAccum: prevAccum, prevGuide: prevGuide,
                                    curGuide: curGuide, curAccum: curAccum,
                                    reuseCap: reuse, cmd: cmd)
                if wasMoving && !moving {
                    // Settle bookkeeping: per-pixel history after
                    // motion is roughly the moving reuse cap.
                    sampleCount = min(sampleCount, play ? 12 : 24)
                }
            } else {
                clearAccum(curAccum, cmd: cmd)
                sampleCount = 0
            }
            // Trace more samples unless settled and the camera
            // has not moved since the last presented frame.
            if moving || camMoved || sampleCount < t.maxSettle {
                if encodeTrace(accum: curAccum, camera: camera, count: t.spp,
                               seedStart: frameSeedCounter, bounces: t.bounces,
                               cap: Float(t.maxSettle), glassBudget: t.glassBudget,
                               cmd: cmd) {
                    traced = true
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

        if let tonemapPipeline,
           let renc = cmd.makeRenderCommandEncoder(descriptor: passDesc) {
            var params = TonemapParams(samples: max(sampleCount, 1), exposure: exposure)
            renc.setRenderPipelineState(tonemapPipeline)
            renc.setFragmentTexture(curAccum, index: 0)
            renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
            renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renc.endEncoding()
        }
        cmd.present(drawable)
        cmd.commit()
        wasMoving = moving
        guard accel != nil else { return false }
        return moving || (traced && sampleCount < t.maxSettle)
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
