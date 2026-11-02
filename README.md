# LumenScene

[![Build IPA](https://github.com/a-user-of-python/lumenscene/actions/workflows/build-ipa.yml/badge.svg)](https://github.com/a-user-of-python/lumenscene/actions/workflows/build-ipa.yml)
[![Repro (snow accumulation)](https://github.com/a-user-of-python/lumenscene/actions/workflows/repro.yml/badge.svg)](https://github.com/a-user-of-python/lumenscene/actions/workflows/repro.yml)

**A ray-traced 3D scene editor for iPad** — a real Metal hardware
path tracer you can edit in: any object can become a light, glass
refracts, weather is a live GPU simulation, and a 55-kind node
system turns scenes into little games.

LumenScene targets the iPad Air M3 (2360×1640) and renders with
Metal's hardware ray tracing: scenes are lit by actual light, not
tricks — emissive objects genuinely illuminate their surroundings
in their own colour, glass bends and reflects, and snow and water
are simulated on the GPU, piling up, ploughing aside and freezing
over time.

## Screenshots

_Screenshots are coming — slots are reserved below. Drop the images
into [`docs/screenshots/`](docs/screenshots/) using the filenames
listed in [its README](docs/screenshots/README.md) and uncomment the
image lines._

<!-- Once the files exist in docs/screenshots/, remove this comment block's markers:
![The editor](docs/screenshots/editor.png)
![Play mode](docs/screenshots/play.png)
![Deep snow](docs/screenshots/weather-snow.png)
![The node editor](docs/screenshots/nodes.png)
![A final render](docs/screenshots/final-render.png)
-->

| Slot | Shows |
| --- | --- |
| `editor.png` | The editor: object tree, viewport, properties |
| `play.png` | Play mode: first-person walk-around |
| `weather-snow.png` | A scene with deep snow / weather running |
| `nodes.png` | The node editor with a working graph |
| `final-render.png` | A finished Final Render |

## Features

### The editor

- Home screen = project list with thumbnails (the view at last save).
- Object tree on the left, path-traced viewport in the centre,
  properties on the right; top bar carries the project name,
  Undo/Redo, Play, Nodes, Weather, Save and a More menu.
- Primitives (box, sphere, cylinder, plane and more), multi-select
  with group drag/delete, full transform and material editing,
  keyboard navigation in both Edit and Play modes.
- **Model import** — STL, OBJ and USD/USDZ via Apple's ModelIO,
  imported in-app from the app's Files folder (no system document
  picker), split per submesh. STEP is not supported: no iOS library
  reads STEP without a separate CAD engine, and this app ships none.
- **Procedural PBR textures** — concrete, cement and brick sets with
  albedo, normal, height and roughness, plus a built-in Texture
  Designer; parallax/normal depth so light rakes across surfaces.

### Materials & light

- **Emissive anything** — any object can be emissive with its own
  colour and adjustable intensity, and it really lights the room:
  next-event estimation samples emissive triangles as area lights
  with ray-traced shadows.
- **Glass** — dielectric Fresnel reflection/refraction with
  adjustable transmission, matte↔clear roughness (frosting both the
  reflection and the transmitted light), tint and IOR.

### Play mode

- One tap switches to a first-person walk-around: virtual joystick
  and drag-look on touch, WASD/QE + F with a keyboard. A procedural
  humanoid player (eye line 1.7 m) walks the scene; Pause returns to
  editing untouched.

### The node system

- **55 node kinds**, wired in a visual editor over named variables:
  spawn objects and seeded patterns, behaviours (including Chase
  Player), trigger areas with area streaming, checkpoints and
  respawn, weather control, gravity, sequences, physics and simple
  particles (with cloth-like drape), messages and value displays.
- Trigger areas are invisible zones that group objects for node
  logic — walk into one and the graph reacts.

### Weather simulation

- A **persistent GPU compute simulation**, not a growing mesh:
  snow accumulates on a ~0.20 m lattice with per-object exposure
  (fresh objects start clean and gather snow at the ambient rate),
  a multi-scale drift field with a per-project seed, a settable
  maximum depth (0.05–2.0 m), 37° angle-of-repose flow, and
  volume-conserving ploughing — drag something through deep snow
  and it carves a trench and banks the snow aside; trails heal
  under fresh snowfall.
- **Water** pools where it can collect — round puddles confined to
  up-facing support, never climbing vertical faces or forming
  ledges — and a **temperature slider** (−20…+30 °C) drives the
  phase behaviour: rain turns to snow at or below freezing,
  puddles freeze into pale mirror ice, warm air melts the pack.
- Procedural precipitation: soft round snowflakes and capsule rain
  streaks — no square silhouettes at any angle. A water-reflections
  quality picker (Full/Half/Quarter) trades reflection sharpness
  for speed.

### Rendering

- Monte Carlo path tracing in a Metal compute kernel over a merged
  acceleration structure, progressively accumulated, with temporal
  reprojection and a class-aware denoise fade as the image settles;
  dynamic resolution applies only while the camera moves.
- Quality picker (Auto / High / Battery saver); the settled picture
  is never traded away for speed — that is a standing rule of the
  project.
- **Final Render** ("Render Image…" in the More menu): 16 bounces,
  4,096 samples per pixel accumulated progressively at native
  2360×1640, with true refractive rain (the nearest raindrops
  become real glass geometry for the still). Cancel keeps the
  best-so-far image; save to Photos, Share, or Files.
- **Idle freeze**: leave the app alone and rendering and scene time
  stop completely (configurable: Off / 15 s / 30 s / 1 min / 2 min /
  5 min) — rain hangs mid-air rather than burning battery — and the
  next touch or keypress wakes it with no time jump.

## Documentation

- [Manual](docs/MANUAL.md) — the editor, rendering, weather, nodes
  and particles, and the known limits, honestly.
- [Version History](docs/VERSION-HISTORY.md) — the full per-version
  record, including the bug hunts.

## Version history

| Version | Date | Summary |
| --- | --- | --- |
| 1.0 | 2026-10-30 | The editor: path-traced viewport, emissive area lights, glass, procedural PBR, ModelIO import, Play walk mode |
| 2.0 | 2026-10-30 | Temporal reprojection, manipulation previews, higher resting quality, render-quality menu |
| 3.0 | 2026-10-30 | Weather build-up, Texture Designer, roads, touch Play |
| 3.1 | 2026-10-30 | Menu-freeze fix; precipitation rebuilt |
| 3.2 | 2026-10-30 | Puddles on flat surfaces, slope/dip response |
| 3.3 | 2026-10-30 | Weather performance: baked field, metaball puddles |
| 3.4 | 2026-10-31 | Snow and puddles become real 3D surfaces; reflection ghosting fixed |
| 4.0 | 2026-10-31 | The node system, dynamic layer, physics particles, trigger areas |
| 4.1 | 2026-10-31 | 55 node kinds, spawn patterns, behaviours, checkpoints, new roads and icon |
| 4.2 | 2026-10-31 | Top bar rebuilt, settle-fuzz denoise fade, spawn offsets, chase freeze fixed |
| 4.3 | 2026-10-31 | Living snow: drift field + seed, settable depth, per-object exposure, humanoid player |
| 4.4 | 2026-10-31 | Snow becomes its own lattice surface with a true snow material |
| 4.5 | 2026-10-31 | Play camera out of the player's head; snow banks and slumps at 37° |
| 4.6 | 2026-10-31 | Volume-conserving ploughing; multi-select |
| 4.7 | 2026-10-31 | Trails heal under snowfall; temperature with freeze/melt phases |
| 4.8 | 2026-10-31 | Snow and water become a persistent GPU simulation; panel layout sweep |
| 4.9 | 2026-10-31 | Full-app optimisation pass — same picture, less waiting |
| 4.10 | 2026-10-31 | Idle freeze + Final Render with refractive rain |
| 4.10.1 | 2026-11-01 | Crash-hardening hotfix (8 hazards) + step instrumentation; panel header fix |
| 4.10.2 | 2026-11-01 | Crash round 2: failed-build swap chain, NaN guards, sim-tick fencing |
| 4.10.3 | 2026-11-01 | The snow crash reproduced in the simulator and fixed for real; device-confirmed |
| 4.10.4 | 2026-11-01 | Drag-through-snow and spawn snow fixes; on iPhone the app now shows an iPad-required notice |
| 4.11 | 2026-11-02 | Snow rebuilt as a granular physics simulation (mass, density, repose flow) with a physics audit suite — retired on device evidence the same day; the games-standard model is in development as 4.12 |

Full details in the [Version History](docs/VERSION-HISTORY.md).

## Building

The Xcode project is static and lives in the repo at
`App/LumenScene.xcodeproj` — no project generator, no external
engine to build first:

```
xcodebuild -project App/LumenScene.xcodeproj -scheme LumenScene ...
```

CI ([`.github/workflows/build-ipa.yml`](.github/workflows/build-ipa.yml))
is a single job on a standard `macos-26` runner that builds the
unsigned app and packages an IPA artifact. Install the unsigned IPA
with any sideloading/signing tool.

Requires iOS 17+ and an iPad with Metal hardware ray tracing
(M3 and later).

## How it's verified

Weather and lifecycle changes are gated on host-side audits, and
crashes get reproduced rather than guessed at:

- **Weather-sim audit** — 33 checks over the simulation's kernel
  math (conservation, repose, pooling, phase timing).
- **Idle-freeze audit** — 51 checks (a 600-second freeze advances
  scene time by exactly zero, wake input is never swallowed, and
  more).
- **Simulator repro** ([`.github/workflows/repro.yml`](.github/workflows/repro.yml))
  — the `-LSRepro` driver runs real accumulation sequences in the
  iOS Simulator on CI. It is how the v4.10.3 snow crash was pinned:
  reproduced in 10 of 11 configurations before the fix, passing in
  all of them after.

## Project layout

```
App/
  LumenScene.xcodeproj        the static Xcode project
  Assets.xcassets/            app icon (a Snell-traced rainbow fan)
  LumenScene/
    LumenSceneApp.swift       app entry point
    HomeView.swift            project home (list + thumbnails)
    EditorView.swift          editor shell (tree / viewport / properties)
    EditorSession.swift       editor state, undo, weather controls
    Viewport.swift            Metal viewport and input
    Renderer.swift            path tracer + weather-simulation host
    Shaders.metal             trace, denoise and weather kernels
    Models.swift              scene model
    ProjectStore.swift        project persistence (JSON + assets)
    MeshGen.swift             primitive meshes
    ModelImport.swift         ModelIO import (STL/OBJ/USD/USDZ)
    Textures.swift            procedural PBR texture sets
    TextureDesigner.swift     the Texture Designer
    GameNodes.swift           node definitions (55 kinds)
    NodeEditorView.swift      the visual node editor
    GameRuntime.swift         node runtime, particles, areas
    PlayerModel.swift         the procedural humanoid player
    DemoScene.swift           the starter scene
    ReproDriver.swift         simulator repro harness (inert in normal use)
    Info.plist
docs/
  MANUAL.md                   the manual
  VERSION-HISTORY.md          the full version record
  screenshots/                screenshot drop-in slots (see its README)
.github/workflows/
  build-ipa.yml               CI: unsigned IPA on macos-26
  repro.yml                   CI: simulator repro matrix
```

## Third-party code

- Model import uses ModelIO, an Apple system framework; there is
  no third-party CAD engine in the app.
- Everything else is original to this project.
