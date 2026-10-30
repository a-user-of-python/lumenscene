# LumenScene

A ray-traced 3D scene editor for iPad (M3 and later), built around
Metal's hardware ray tracing. Scenes are lit by real light: any
object can be turned into an area light that illuminates its
surroundings in its own colour, glass refracts and reflects, and
procedural concrete / cement / brick surfaces carry normal, height
and roughness data so light rakes across their depth.

## Features

- **Monte Carlo path tracing in a Metal compute kernel** —
  merged-scene primitive acceleration structure, one sample per
  dispatch accumulated progressively. Next-event estimation samples
  emissive triangles as area lights (ray-traced shadows), plus
  cosine-weighted indirect bounces, so emissive surfaces genuinely
  light the room.
- **Glass** — dielectric Fresnel reflection/refraction with
  adjustable transmission, matte (roughness), tint and IOR; matte
  frosts both the reflection and the transmitted light.
- **Model import** — STL, OBJ and USD/USDZ files copied into the
  app's Files folder are imported in-app (no system document
  picker) using Apple's ModelIO framework, split into scene
  objects per submesh with per-submesh material colours where the
  format carries them (STL carries none). Imported surfaces
  support triplanar procedural texturing. STEP import is not
  supported: no iOS library reads STEP without a separate CAD
  engine, and this app does not include one.
- **Procedural PBR textures** — concrete, cement and brick sets
  (albedo + tangent-space normal + roughness/height) generated
  on-device; tiled UVs on primitives with normal mapping and
  parallax.
- **Editor** — project home screen with save-time thumbnails,
  object tree, transform/material properties, undo/redo,
  primitives (box, sphere, cylinder, plane).
- **Play mode** — one tap switches to a first-person walk-around
  (WASD + drag to look); Pause returns to editing untouched.
  Edit mode flies with the same keys, Q/E for down/up, F focuses
  the selection.
- **Save / export** — projects persist as JSON plus their source
  model assets and meshes; export a project package (.zip) or a
  high-sample PNG render via the share sheet.

## Building

The Xcode project is a static project stored in the repo at
`App/LumenScene.xcodeproj` — there is no project generator and no
external engine to build first:

```
xcodebuild -project App/LumenScene.xcodeproj -scheme LumenScene ...
```

CI (`.github/workflows/build-ipa.yml`) is a single job on a
standard macos-26 runner that builds the unsigned app and
packages the IPA.

The app requires iOS 17+ and a Metal ray-tracing-capable iPad.

## Third-party code

- Model import uses ModelIO, an Apple system framework; there is
  no third-party CAD engine in the app.
- Everything else is original to this project.
