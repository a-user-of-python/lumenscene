// DemoScene.swift — the first-launch demo project. Every headline
// feature is visible in one small stage: a coloured emissive sign
// lighting a brick wall, a clear glass pane, a frosted glass block,
// procedural concrete / cement / brick surfaces, and a ceiling
// light panel, so Sean can judge the renderer immediately.

import Foundation

enum DemoScene {
    static func makeProject() -> Project {
        var p = Project(name: "Demo Stage")
        var objs: [SceneObject] = []

        func prim(_ kind: String, _ name: String,
                  pos: [Double], rot: [Double] = [0, 0, 0],
                  params: [String: Double] = [:],
                  material: MaterialProps) -> SceneObject {
            var o = SceneObject.primitive(kind, name: name)
            if !params.isEmpty { o.params = params }
            o.transform.position = pos
            o.transform.rotation = rot
            o.material = material
            return o
        }

        // Floor — concrete with depth texture.
        var floorMat = MaterialProps()
        floorMat.texturePreset = "concrete"
        floorMat.textureScale = 2.0
        objs.append(prim("plane", "Concrete floor", pos: [0, 0, 0],
                         params: ["w": 10, "d": 8], material: floorMat))

        // Back wall — brick.
        var brickMat = MaterialProps()
        brickMat.texturePreset = "brick"
        brickMat.textureScale = 2.5
        objs.append(prim("box", "Brick wall", pos: [0, 1.7, -3.9],
                         params: ["w": 10, "h": 3.4, "d": 0.2], material: brickMat))

        // Left wall — cement.
        var cementMat = MaterialProps()
        cementMat.texturePreset = "cement"
        cementMat.textureScale = 2.5
        objs.append(prim("box", "Cement wall", pos: [-4.9, 1.7, 0],
                         params: ["w": 0.2, "h": 3.4, "d": 8], material: cementMat))

        // Ceiling light panel — warm white emitter.
        var ceilMat = MaterialProps()
        ceilMat.baseColor = RGB(0.2, 0.2, 0.2)
        ceilMat.emissiveOn = true
        ceilMat.emissiveColor = RGB(1.0, 0.97, 0.9)
        ceilMat.emissiveIntensity = 3.0
        objs.append(prim("box", "Ceiling light", pos: [0, 3.32, -0.6],
                         params: ["w": 2.4, "h": 0.06, "d": 1.2], material: ceilMat))

        // The sign — hot pink emitter proud of the brick wall; its
        // light spills onto the wall and the floor.
        var signMat = MaterialProps()
        signMat.baseColor = RGB(0.15, 0.02, 0.06)
        signMat.emissiveOn = true
        signMat.emissiveColor = RGB(1.0, 0.16, 0.5)
        signMat.emissiveIntensity = 7.0
        objs.append(prim("box", "Neon sign", pos: [0.6, 2.05, -3.72],
                         params: ["w": 2.4, "h": 0.8, "d": 0.1], material: signMat))

        // Small teal glow cube on the floor.
        var cubeMat = MaterialProps()
        cubeMat.baseColor = RGB(0.02, 0.1, 0.12)
        cubeMat.emissiveOn = true
        cubeMat.emissiveColor = RGB(0.1, 0.9, 1.0)
        cubeMat.emissiveIntensity = 3.5
        objs.append(prim("box", "Glow cube", pos: [-2.6, 0.18, -1.2],
                         params: ["w": 0.36, "h": 0.36, "d": 0.36], material: cubeMat))

        // Clear glass pane, angled, standing on the floor.
        var glassMat = MaterialProps()
        glassMat.glassOn = true
        glassMat.transmission = 1.0
        glassMat.roughness = 0.04
        glassMat.ior = 1.5
        glassMat.tint = RGB(0.92, 1.0, 0.96)
        objs.append(prim("box", "Glass pane", pos: [1.9, 0.95, -1.4], rot: [0, 28, 0],
                         params: ["w": 1.5, "h": 1.9, "d": 0.05], material: glassMat))

        // Frosted (matte) glass block.
        var frostMat = MaterialProps()
        frostMat.glassOn = true
        frostMat.transmission = 1.0
        frostMat.roughness = 0.6
        frostMat.ior = 1.5
        frostMat.tint = RGB(0.95, 0.98, 1.0)
        objs.append(prim("box", "Frosted glass block", pos: [-1.7, 0.28, -0.5],
                         params: ["w": 0.55, "h": 0.55, "d": 0.55], material: frostMat))

        // Concrete pillar (cylinder).
        var pillarMat = MaterialProps()
        pillarMat.texturePreset = "concrete"
        pillarMat.textureScale = 1.5
        objs.append(prim("cylinder", "Concrete pillar", pos: [-2.9, 0.48, -2.3],
                         params: ["r": 0.35, "h": 0.95], material: pillarMat))

        // Cement sphere.
        var sphereMat = MaterialProps()
        sphereMat.texturePreset = "cement"
        sphereMat.textureScale = 1.2
        objs.append(prim("sphere", "Cement sphere", pos: [3.1, 0.42, 0.9],
                         params: ["r": 0.42], material: sphereMat))

        // Plain coloured crate.
        var crateMat = MaterialProps()
        crateMat.baseColor = RGB(0.95, 0.45, 0.1)
        objs.append(prim("box", "Orange crate", pos: [2.9, 0.25, -2.5], rot: [0, -20, 0],
                         params: ["w": 0.7, "h": 0.5, "d": 0.7], material: crateMat))

        p.scene.objects = objs
        p.scene.camera.target = [0, 1.05, -0.8]
        p.scene.camera.azimuth = 32
        p.scene.camera.elevation = 19
        p.scene.camera.distance = 7.6
        p.scene.camera.playPosition = [0.4, 1.7, 3.4]
        p.scene.camera.playYaw = -6
        p.scene.camera.playPitch = -3
        return p
    }
}
