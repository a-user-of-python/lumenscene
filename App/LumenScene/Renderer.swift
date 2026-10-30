// Renderer.swift — owns the Metal device, the merged scene buffers,
// the ray-tracing acceleration structure, and the accumulation
// render loop. All scene triangles are baked into world space on
// the CPU (modest scenes; edits reset accumulation anyway), which
// keeps the GPU side to one BLAS and one material per object.

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

    // CPU copies for picking.
    private var cpuPositions: [Float] = []
    private var cpuIndices: [UInt32] = []
    private var cpuTriObject: [UUID] = []

    // Procedural texture arrays (built once).
    private var texAlbedo: MTLTexture?
    private var texNormal: MTLTexture?
    private var texData: MTLTexture?

    // Accumulation target for the live view.
    private var accumTexture: MTLTexture?
    private(set) var sampleCount: UInt32 = 0
    var samplesPerFrame: Int = 2
    /// Play mode: clear the accumulation every frame (moving camera).
    var resetEachFrame: Bool = false
    var maxBounces: UInt32 = 5
    var exposure: Float = 1.15
    var renderScale: Float = 0.75
    var maxSamples: UInt32 = 512
    var fovY: Float = 55 * .pi / 180

    var converged: Bool { sampleCount >= maxSamples }
    var hasGeometry: Bool { accel != nil }

    init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
        self.rayTracingSupported = device.supportsRaytracing
        guard let library = device.makeDefaultLibrary() else { return nil }
        if let fn = library.makeFunction(name: "pathTraceKernel") {
            tracePipeline = try? device.makeComputePipelineState(function: fn)
        }
        if let vfn = library.makeFunction(name: "tonemapVertex"),
           let ffn = library.makeFunction(name: "tonemapFragment") {
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = vfn
            desc.fragmentFunction = ffn
            desc.colorAttachments[0].pixelFormat = .bgra8Unorm
            tonemapPipeline = try? device.makeRenderPipelineState(descriptor: desc)
        }
        buildTextureArrays()
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
        var positions: [Float] = []
        var normals: [Float] = []
        var uvArr: [Float] = []
        var colArr: [Float] = []
        var indices: [UInt32] = []
        var triMats: [UInt32] = []
        var materials: [GPUMaterial] = []
        var triObjects: [UUID] = []
        var emisTris: [UInt32] = []
        var emisAreas: [Float] = []

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

        // Emissive triangle list + area CDF (for next-event estimation).
        if triCount > 0 {
            for t in 0..<triCount {
                let m = materials[Int(triMats[t])]
                guard (m.flags & 1) != 0 else { continue }
                let a = indices[3 * t], b = indices[3 * t + 1], c = indices[3 * t + 2]
                let p0 = SIMD3<Float>(positions[3 * Int(a)], positions[3 * Int(a) + 1], positions[3 * Int(a) + 2])
                let p1 = SIMD3<Float>(positions[3 * Int(b)], positions[3 * Int(b) + 1], positions[3 * Int(b) + 2])
                let p2 = SIMD3<Float>(positions[3 * Int(c)], positions[3 * Int(c) + 1], positions[3 * Int(c) + 2])
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

        func buf<T>(_ array: [T]) -> MTLBuffer? {
            if array.isEmpty { return device.makeBuffer(length: 4, options: .storageModeShared) }
            return array.withUnsafeBytes { raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }
        }
        posBuffer = buf(positions)
        nrmBuffer = buf(normals)
        uvBuffer = buf(uvArr)
        colBuffer = buf(colArr)
        idxBuffer = buf(indices)
        triMatBuffer = buf(triMats)
        matBuffer = buf(materials)
        emisTriBuffer = buf(emisTris)
        emisCDFBuffer = buf(cdf)

        buildAccelerationStructure()
        resetAccumulation()
    }

    private func gpuMaterial(for obj: SceneObject) -> GPUMaterial {
        let m = obj.material
        var g = GPUMaterial()
        g.baseColor = SIMD4<Float>(m.baseColor.simd, 0)
        g.emissive = SIMD4<Float>(m.emissiveColor.simd, Float(m.emissiveIntensity))
        g.glassParams = SIMD4<Float>(Float(m.transmission), Float(m.roughness), Float(m.ior), 0)
        g.tint = SIMD4<Float>(m.tint.simd, Float(m.textureScale))
        g.texIndex = Renderer.texIndex(forPreset: m.texturePreset)
        var flags: UInt32 = 0
        if m.emissiveOn { flags |= 1 }
        if m.glassOn { flags |= 2 }
        if m.triplanar { flags |= 4 }
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
        let desc = MTLAccelerationStructureDescriptor.primitiveAccelerationStructureDescriptor()
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

    // MARK: Accumulation / drawing

    func resetAccumulation() {
        sampleCount = 0
        accumTexture = nil // force recreation (clears old samples)
    }

    private func makeAccumTexture(width: Int, height: Int) -> MTLTexture? {
        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba16Float, width: max(width, 1), height: max(height, 1),
            mipmapped: false)
        desc.usage = [.shaderRead, .shaderWrite]
        desc.storageMode = .private
        return device.makeTexture(descriptor: desc)
    }

    /// Zero the accumulation texture. The trace kernel reads the
    /// previous contents before adding its sample, so a brand-new
    /// (uninitialised) texture MUST be cleared first.
    private func clearAccum(_ accum: MTLTexture, cmd: MTLCommandBuffer) {
        let clearDesc = MTLRenderPassDescriptor()
        clearDesc.colorAttachments[0].texture = accum
        clearDesc.colorAttachments[0].loadAction = .clear
        clearDesc.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        clearDesc.colorAttachments[0].storeAction = .store
        if let cenc = cmd.makeRenderCommandEncoder(descriptor: clearDesc) {
            cenc.endEncoding()
        }
    }

    private func uniforms(camera: CameraBasis, aspect: Float, seed: UInt32) -> GPUUniforms {
        var u = GPUUniforms()
        u.camOrigin = SIMD4<Float>(camera.origin, 0)
        u.camRight = SIMD4<Float>(camera.right, 0)
        u.camUp = SIMD4<Float>(camera.up, 0)
        u.camForward = SIMD4<Float>(camera.forward, 0)
        u.tanFovAspect = SIMD2<Float>(tan(fovY / 2), aspect)
        u.frameSeed = seed
        u.maxBounces = maxBounces
        u.triCount = UInt32(triCount)
        u.emissiveCount = UInt32(emissiveCount)
        u.totalEmissiveArea = totalEmissiveArea
        u.envIntensity = 1.0
        return u
    }

    /// Encode `count` path-tracing samples into `accum`. Returns false
    /// when there is nothing to trace.
    private func encodeTrace(accum: MTLTexture, camera: CameraBasis,
                             count: Int, seedStart: UInt32,
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
        let w = tracePipeline.threadExecutionWidth
        let h = tracePipeline.maxTotalThreadsPerThreadgroup / w
        let group = MTLSize(width: w, height: h, depth: 1)
        let grid = MTLSize(width: accum.width, height: accum.height, depth: 1)
        for i in 0..<count {
            var u = uniforms(camera: camera, aspect: aspect, seed: seedStart + UInt32(i))
            enc.setBytes(&u, length: MemoryLayout<GPUUniforms>.stride, index: 0)
            enc.dispatchThreads(grid, threadsPerThreadgroup: group)
        }
        enc.endEncoding()
        return true
    }

    /// One frame for the live MTKView.
    func draw(view: MTKView, camera: CameraBasis) {
        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let cmd = queue.makeCommandBuffer() else { return }
        let pixelW = Int(Float(view.drawableSize.width) * renderScale)
        let pixelH = Int(Float(view.drawableSize.height) * renderScale)
        if accumTexture == nil || accumTexture!.width != pixelW || accumTexture!.height != pixelH {
            accumTexture = makeAccumTexture(width: pixelW, height: pixelH)
            sampleCount = 0
            if let accum = accumTexture { clearAccum(accum, cmd: cmd) }
        }
        if resetEachFrame, let accum = accumTexture {
            clearAccum(accum, cmd: cmd)
            sampleCount = 0
        }
        if let accum = accumTexture, !converged, sampleCount < maxSamples {
            let n = min(samplesPerFrame, Int(maxSamples - sampleCount))
            if encodeTrace(accum: accum, camera: camera, count: n,
                           seedStart: sampleCount, cmd: cmd) {
                sampleCount += UInt32(n)
            } else {
                sampleCount = maxSamples // nothing to trace; stop trying
            }
        }
        if let tonemapPipeline, let accum = accumTexture,
           let renc = cmd.makeRenderCommandEncoder(descriptor: passDesc) {
            var params = TonemapParams(samples: max(sampleCount, 1), exposure: exposure)
            renc.setRenderPipelineState(tonemapPipeline)
            renc.setFragmentTexture(accum, index: 0)
            renc.setFragmentBytes(&params, length: MemoryLayout<TonemapParams>.stride, index: 0)
            renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            renc.endEncoding()
        }
        cmd.present(drawable)
        cmd.commit()
    }

    /// Render the current scene offscreen and return a PNG (used for
    /// project thumbnails and PNG export).
    func renderStillPNG(camera: CameraBasis, width: Int, height: Int, samples: Int) -> Data? {
        guard let accum = makeAccumTexture(width: width, height: height),
              let cmd = queue.makeCommandBuffer() else { return nil }
        clearAccum(accum, cmd: cmd)
        let traced = encodeTrace(accum: accum, camera: camera, count: samples,
                                 seedStart: 0, cmd: cmd)
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
