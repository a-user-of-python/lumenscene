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
- **STEP import** — `.step` / `.stp` files copied into the app's
  Files folder are imported in-app (no system document picker),
  tessellated with Open CASCADE and split into scene objects by the
  colours stored in the file. Imported surfaces support triplanar
  procedural texturing.
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
  STEP assets and meshes; export a project package (.zip) or a
  high-sample PNG render via the share sheet.

## Building

The Xcode project is generated, not stored:

```
python3 App/gen_project.py            # needs cascade-engine/ next to App/
xcodebuild -project LumenScene.xcodeproj -scheme LumenScene ...
```

CI (`.github/workflows/build-ipa.yml`) does the whole job on a
standard macOS runner: it builds the CascadeKit engine (Open
CASCADE iOS port, pinned by commit) once, caches it, then builds
the unsigned app. `gen_project.py --engine-only` produces just the
engine project; `LUMEN_ENGINE_DIST=<dir with libCascadeKit.a +
occt_facade.h>` builds the app against a prebuilt engine.

The app requires iOS 17+ and a Metal ray-tracing-capable iPad.

## Third-party code

- Open CASCADE Technology (OCCT) 8.0.1, LGPL 2.1 with the OCCT
  exception — used for STEP import only. The license texts ship in
  the app bundle.
- Everything else is original to this project.
