// Models.swift — the scene data model for LumenScene.
//
// Everything here is value-type and Codable: a Project is saved as
// JSON, and undo/redo is snapshot-based on the Scene value. Imported
// geometry lives in binary mesh files next to the JSON (see
// ProjectStore); primitives are regenerated from their parameters.

import Foundation
import simd

// MARK: - Colour

struct RGB: Codable, Equatable {
    var r: Double
    var g: Double
    var b: Double

    init(_ r: Double, _ g: Double, _ b: Double) {
        self.r = r; self.g = g; self.b = b
    }

    static let white = RGB(1, 1, 1)
    static let black = RGB(0, 0, 0)

    var simd: SIMD3<Float> { SIMD3<Float>(Float(r), Float(g), Float(b)) }

    init(simd v: SIMD3<Float>) {
        self.init(Double(v.x), Double(v.y), Double(v.z))
    }
}

// MARK: - Transform

/// Position in metres, rotation as Euler angles in degrees
/// (x = pitch, y = yaw, z = roll, applied Z then X then Y),
/// scale as per-axis multipliers.
struct Transform3: Codable, Equatable {
    var position: [Double] = [0, 0, 0]
    var rotation: [Double] = [0, 0, 0]
    var scale: [Double] = [1, 1, 1]

    var matrix: float4x4 {
        let qx = simd_quaternion(Float(rotation[0]) * .pi / 180, SIMD3<Float>(1, 0, 0))
        let qy = simd_quaternion(Float(rotation[1]) * .pi / 180, SIMD3<Float>(0, 1, 0))
        let qz = simd_quaternion(Float(rotation[2]) * .pi / 180, SIMD3<Float>(0, 0, 1))
        var m = float4x4(qy * qx * qz)
        m.columns.0 *= Float(scale[0])
        m.columns.1 *= Float(scale[1])
        m.columns.2 *= Float(scale[2])
        m.columns.3 = SIMD4<Float>(Float(position[0]), Float(position[1]), Float(position[2]), 1)
        return m
    }
}

// MARK: - Material

struct MaterialProps: Codable, Equatable {
    /// Base colour. For imported models this multiplies the per-part
    /// colours baked into the mesh vertices (white = untouched).
    var baseColor: RGB = .white
    /// Emissive (light-emitting) surface. The emitted light takes
    /// `emissiveColor` and really illuminates the scene.
    var emissiveOn: Bool = false
    var emissiveColor: RGB = RGB(1, 0.6, 0.2)
    var emissiveIntensity: Double = 4.0
    /// Glass behaviour: reflection + refraction with Fresnel,
    /// tinted transmission, and a matte (roughness) control that
    /// frosts both the reflection and the through-light.
    var glassOn: Bool = false
    var transmission: Double = 1.0   // 0 = opaque surface, 1 = fully glass
    var roughness: Double = 0.05     // 0 = clear / mirror-sharp, 1 = fully frosted
    var ior: Double = 1.5
    var tint: RGB = .white
    /// Procedural texture preset: "none", "concrete", "cement", "brick".
    var texturePreset: String = "none"
    /// Tiling: UV repeats per object face, or tiles per metre for
    /// triplanar-mapped imports.
    var textureScale: Double = 1.0
    /// True for imported models (no UVs in the file): the
    /// renderer samples textures triplanar from world position.
    var triplanar: Bool = false
    /// Auto-fit: the texture spans the face exactly once instead of
    /// tiling by `textureScale`. Used by road textures, whose whole
    /// design (markings, crossings) lives in a single tile.
    var textureFit: Bool = false
}

extension MaterialProps {
    private enum CodingKeys: String, CodingKey {
        case baseColor, emissiveOn, emissiveColor, emissiveIntensity
        case glassOn, transmission, roughness, ior, tint
        case texturePreset, textureScale, triplanar, textureFit
    }

    // Custom Decodable for backward compatibility: project.json files
    // written by v1/v2 lack `textureFit` (and may lack any later key),
    // so every field is decoded with decodeIfPresent and falls back to
    // exactly the property default above. encode(to:) stays
    // synthesised — providing init(from:) in an extension does not
    // suppress it, and the synthesised encoder uses the same key names.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        baseColor = try c.decodeIfPresent(RGB.self, forKey: .baseColor) ?? .white
        emissiveOn = try c.decodeIfPresent(Bool.self, forKey: .emissiveOn) ?? false
        emissiveColor = try c.decodeIfPresent(RGB.self, forKey: .emissiveColor) ?? RGB(1, 0.6, 0.2)
        emissiveIntensity = try c.decodeIfPresent(Double.self, forKey: .emissiveIntensity) ?? 4.0
        glassOn = try c.decodeIfPresent(Bool.self, forKey: .glassOn) ?? false
        transmission = try c.decodeIfPresent(Double.self, forKey: .transmission) ?? 1.0
        roughness = try c.decodeIfPresent(Double.self, forKey: .roughness) ?? 0.05
        ior = try c.decodeIfPresent(Double.self, forKey: .ior) ?? 1.5
        tint = try c.decodeIfPresent(RGB.self, forKey: .tint) ?? .white
        texturePreset = try c.decodeIfPresent(String.self, forKey: .texturePreset) ?? "none"
        textureScale = try c.decodeIfPresent(Double.self, forKey: .textureScale) ?? 1.0
        triplanar = try c.decodeIfPresent(Bool.self, forKey: .triplanar) ?? false
        textureFit = try c.decodeIfPresent(Bool.self, forKey: .textureFit) ?? false
    }
}

// MARK: - Scene objects

struct SceneObject: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    var name: String
    /// "primitive" or "model".
    var source: String
    /// For source == "primitive": "box" | "sphere" | "cylinder" | "plane".
    var primitiveKind: String = "box"
    /// Primitive dimensions in metres (box: w/h/d, sphere: r,
    /// cylinder: r/h, plane: w/d).
    var params: [String: Double] = [:]
    /// For source == "model": file inside the project's meshes/ folder.
    var meshFile: String? = nil
    /// For source == "model": file inside the project's assets/ folder
    /// (the original model file, kept with the project).
    var assetFile: String? = nil
    var visible: Bool = true
    var transform: Transform3 = Transform3()
    var material: MaterialProps = MaterialProps()
    /// The trigger area this object belongs to, if any (v4). Objects
    /// with an area are loaded/unloaded with it by the node graph;
    /// nil means the object is always present.
    var areaID: UUID? = nil

    static func primitive(_ kind: String, name: String) -> SceneObject {
        var o = SceneObject(name: name, source: "primitive", primitiveKind: kind)
        switch kind {
        case "box": o.params = ["w": 1, "h": 1, "d": 1]
        case "sphere": o.params = ["r": 0.5]
        case "cylinder": o.params = ["r": 0.5, "h": 1]
        case "plane": o.params = ["w": 4, "d": 4]
        default: o.params = [:]
        }
        return o
    }
}

extension SceneObject {
    private enum CodingKeys: String, CodingKey {
        case id, name, source, primitiveKind, params, meshFile, assetFile
        case visible, transform, material, areaID
    }

    // Custom Decodable for backward compatibility: SceneObject used
    // to rely on the synthesised conformance, which throws on any
    // missing key — so a v1–v3 project.json (no `areaID`, and possibly
    // missing any field added since) would fail to load at all. Every
    // field is therefore decoded with decodeIfPresent and falls back
    // to exactly the property default above. `name` and `source` had
    // no property defaults (they were required at construction), so a
    // missing key falls back to "Object" / "primitive". encode(to:)
    // stays synthesised — see the note on MaterialProps above.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? "Object"
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? "primitive"
        primitiveKind = try c.decodeIfPresent(String.self, forKey: .primitiveKind) ?? "box"
        params = try c.decodeIfPresent([String: Double].self, forKey: .params) ?? [:]
        meshFile = try c.decodeIfPresent(String.self, forKey: .meshFile) ?? nil
        assetFile = try c.decodeIfPresent(String.self, forKey: .assetFile) ?? nil
        visible = try c.decodeIfPresent(Bool.self, forKey: .visible) ?? true
        transform = try c.decodeIfPresent(Transform3.self, forKey: .transform) ?? Transform3()
        material = try c.decodeIfPresent(MaterialProps.self, forKey: .material) ?? MaterialProps()
        areaID = try c.decodeIfPresent(UUID.self, forKey: .areaID) ?? nil
    }
}

// MARK: - Camera

/// Both camera rigs are persisted so a project reopens where it was.
struct CameraState: Codable, Equatable {
    // Edit-mode orbit rig.
    var target: [Double] = [0, 1, 0]
    var azimuth: Double = 35      // degrees
    var elevation: Double = 22    // degrees
    var distance: Double = 7
    // Play-mode first-person rig.
    var playPosition: [Double] = [0, 1.7, 3.5]
    var playYaw: Double = 0       // degrees; 0 looks along -Z, + turns left
    var playPitch: Double = 0

    /// Orbit camera basis: (origin, forward, right, up).
    func orbitBasis() -> (origin: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>) {
        let az = Float(azimuth) * .pi / 180
        let el = Float(elevation) * .pi / 180
        let dir = SIMD3<Float>(cos(el) * sin(az), sin(el), cos(el) * cos(az))
        let t = SIMD3<Float>(Float(target[0]), Float(target[1]), Float(target[2]))
        let origin = t + dir * Float(distance)
        let forward = simd_normalize(t - origin)
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        let up = simd_cross(right, forward)
        return (origin, forward, right, up)
    }

    /// Play camera basis from yaw/pitch (yaw 0 = looking -Z... we use
    /// yaw measured so that 180 faces -Z when standing at +Z; forward
    /// is computed directly from yaw/pitch below).
    func playBasis() -> (origin: SIMD3<Float>, forward: SIMD3<Float>, right: SIMD3<Float>, up: SIMD3<Float>) {
        let yaw = Float(playYaw) * .pi / 180
        let pitch = Float(playPitch) * .pi / 180
        // Yaw 0 looks along -Z; positive yaw turns left (toward -X).
        let forward = SIMD3<Float>(-sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))
        let right = simd_normalize(simd_cross(forward, SIMD3<Float>(0, 1, 0)))
        let up = simd_cross(right, forward)
        let origin = SIMD3<Float>(Float(playPosition[0]), Float(playPosition[1]), Float(playPosition[2]))
        return (origin, forward, right, up)
    }
}

// MARK: - Weather

/// Scene-wide weather state. `mode`: 0 = off, 1 = rain, 2 = snow.
/// `wetness` (0...1) darkens / slickens surfaces in rain; `snow`
/// (0...1) is accumulation in snow mode.
struct WeatherState: Codable, Equatable {
    var mode: Int = 0
    var wetness: Double = 0
    var snow: Double = 0
}

// MARK: - Trigger areas

/// A named box region of the scene (v4). The node graph watches the
/// player entering/leaving areas (see GameNodes.swift), and objects
/// whose `areaID` matches can be loaded and unloaded as a group.
/// `center` and `size` are in metres; `size` is the full extent on
/// each axis, so the box reaches centre ± size/2.
struct TriggerArea: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    var name: String = "Area"
    var center: [Double] = [0, 1.5, 0]   // metres
    var size: [Double] = [4, 3, 4]       // full extents

    /// True when `p` is inside the box (edges included):
    /// |p − centre| ≤ size/2 on every axis.
    func contains(_ p: SIMD3<Float>) -> Bool {
        guard center.count >= 3, size.count >= 3 else { return false }
        let c = SIMD3<Float>(Float(center[0]), Float(center[1]), Float(center[2]))
        let half = SIMD3<Float>(Float(size[0]), Float(size[1]), Float(size[2])) * 0.5
        let d = simd_abs(p - c)
        return d.x <= half.x && d.y <= half.y && d.z <= half.z
    }
}

// MARK: - Scene & project

struct Scene: Codable, Equatable {
    var objects: [SceneObject] = []
    var camera: CameraState = CameraState()
    var weather: WeatherState = WeatherState()
    /// Trigger areas (v4); see TriggerArea above.
    var areas: [TriggerArea] = []
    /// The Play-mode logic graph (v4); see GameNodes.swift.
    var graph: NodeGraph = NodeGraph()

    func object(id: UUID) -> SceneObject? { objects.first { $0.id == id } }
    func area(id: UUID) -> TriggerArea? { areas.first { $0.id == id } }
}

extension Scene {
    private enum CodingKeys: String, CodingKey {
        case objects, camera, weather, areas, graph
    }

    // Custom Decodable for backward compatibility: v1/v2 project.json
    // files have no `weather` key, and pre-v4 files have no `areas`
    // or `graph` keys. Every field falls back to exactly the property
    // default above. encode(to:) stays synthesised (see the note on
    // MaterialProps above).
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        objects = try c.decodeIfPresent([SceneObject].self, forKey: .objects) ?? []
        camera = try c.decodeIfPresent(CameraState.self, forKey: .camera) ?? CameraState()
        weather = try c.decodeIfPresent(WeatherState.self, forKey: .weather) ?? WeatherState()
        areas = try c.decodeIfPresent([TriggerArea].self, forKey: .areas) ?? []
        graph = try c.decodeIfPresent(NodeGraph.self, forKey: .graph) ?? NodeGraph()
    }
}

struct Project: Codable, Equatable, Identifiable {
    var id: UUID = UUID()
    var name: String
    var createdAt: Date = Date()
    var modifiedAt: Date = Date()
    var scene: Scene = Scene()
}

/// Lightweight listing info for the home screen.
struct ProjectMeta: Identifiable {
    var id: UUID
    var name: String
    var modifiedAt: Date
    var thumbnailURL: URL?
}
