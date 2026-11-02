# LumenScene — Version History

Everything introduced in every version, in order. LumenScene is a ray-traced 3D
scene editor for iPad (iPad Air M3): you build scenes from imported models and
primitives, light them with objects that genuinely emit light, walk through
them in Play mode, and — from version 4.0 — program game-style behaviour with
a node system instead of code.

All versions run on the same foundation laid in v1: a custom Monte Carlo path
tracer in Metal using the M3's hardware ray-tracing, with progressive
accumulation (the picture keeps refining while you hold still), ACES
tonemapping, and next-event estimation so emissive objects act as true area
lights.

---

## Version 1.0 — the editor (2026-10-30)

The original commission, delivered as one app:

- **Editor layout** — object tree on the left, 3D viewport in the centre,
  properties on the right (name, transform, material for the selection).
  Top bar: project name, Undo/Redo, Play/Pause, Save, Export.
- **Import** — STL, OBJ, and USD/USDZ models through Apple's ModelIO, picked
  from an in-app list over the shared Documents folder (no system picker).
  (STEP import was originally planned via a CAD engine and was removed at the
  user's direction before v1 shipped; no iOS library reads STEP without one.)
- **Emissive objects** — any object can emit light in its own colour at an
  adjustable intensity. Emissive triangles are real area lights: a glowing
  sign genuinely lights the room around it.
- **Glass** — transmission with Fresnel reflection/refraction, roughness
  jitter for the matte↔frosted range, tint, and index of refraction.
- **Textures with depth** — procedural PBR sets (concrete, cement, brick:
  albedo + normal + roughness/height) generated on-device, with normal
  mapping and one-step parallax on UV-mapped surfaces.
- **Play mode** — walk the scene in first person (eye height locked,
  WASD + drag-look on keyboard).
- **Home screen** — project list with thumbnails of the editor view at last
  save. Projects persist as JSON + mesh files; undo is snapshot-based.
- Device verdict on v1: "it's awesome."

## Version 2.0 — finish the spec, kill the fuzz, stop the heat (2026-10-30)

- **Spec completed** — Beer–Lambert glass absorption (thick tinted glass
  darkens and colours more than thin), drag the selected object directly in
  the viewport to move it, save thumbnails rendered from the *current*
  mode's camera (Play included), triplanar normal mapping for imported
  models.
- **Temporal reprojection** — converged history is carried across camera
  moves for diffuse surfaces (capped while moving), so motion stops looking
  like raw noise. Glass and environment restart instantly — reflections are
  view-dependent and must never be carried.
- **Manipulation previews** — while you drag an emissive object it stops
  emitting; while you drag glass it renders opaque; full behaviour restores
  on release. Dragging is therefore fast.
- **Optimisation with a quality guardrail** — render-on-change (a settled,
  unchanged scene draws *nothing* — v1 redrew at 60 fps forever, the main
  heat source); material edits no longer rebuild the acceleration
  structure; resting quality went *up* (full resolution instead of 0.75×,
  settle cap 512 → 768 samples, 5 bounces); cheaper treatment only in
  motion, previews, and thermal emergencies. Render menu: Auto / High
  detail / Battery saver.

## Version 3.0 — weather, touch Play, Texture Designer, roads (2026-10-30)

- **De-fuzz, round two** — per-pixel firefly clamping (each sample capped
  against the pixel's own running average), higher Play settle caps, and an
  edge-aware denoise applied only to the interactive display — the settled
  image stays the untouched full-quality accumulation.
- **Play is full-screen** — all editor UI hides; only a circular pause
  button (top-right) returns to editing. **Touch controls**: an analog
  virtual joystick joins drag-to-look.
- **Weather that builds up** — rain and snow you switch on; precipitation
  falls around the camera; rain progressively *wets* up-facing surfaces
  (darker, glossy, strongly reflective), snow *covers* them (matte white,
  reflects less). Accumulation is scene state and persists on Save.
- **Texture Designer (home screen)** — a paint canvas whose textures are
  saved by name and applied like presets. Design brushes paint *material
  behaviour*, not just colour: Rough (roughness), Wet (gloss + darkening),
  Brick (brickwork relief) — plus a full colour brush set (sizes, opacity,
  shapes, spray, star stamp, eraser, fill, undo).
- **Road textures** — Highway, Road, Intersection, Crosswalk, Roundabout,
  procedurally generated with worn markings, and **auto-fit**: the texture
  spans the applied face once instead of tiling.
- **Stage-wide drag cut** — while *anything* is being moved, *all* light
  emission on the stage cuts; it restores on release.

## Version 3.1 — weather fixed and shaped (2026-10-30)

Shipped after the first device test of weather failed (precipitation
invisible; opening weather froze every menu):

- **Menu-freeze root-caused and fixed** — build-up had been mutating
  published state every frame, forcing the whole editor UI to rebuild
  60×/second forever. Build-up now advances in non-published accumulators
  and syncs only when a displayed percent changes (and at save/export).
- **Precipitation rebuilt robustly** — particles from an explicit GPU seed
  buffer, roughly double the counts, much larger and brighter (rain streaks
  α 0.50, snow discs α 0.9); a pipeline failure now surfaces in the status
  line instead of failing silently.
- **Weather becomes a shaped field** — wetness spreads and pools unevenly;
  **puddles** form, grow, and merge where a low-frequency field dips (inside
  a puddle the surface goes mirror-flat and reflects like standing water);
  snow piles into **drifts** (uneven depth, crevice darkening, relief)
  instead of a uniform coat. Gentle drying/melting when weather turns off.

## Version 3.2 — flat-floor puddles and living weather (2026-10-30)

- **Puddles guaranteed on any surface** — a world-space cellular (Worley)
  field makes coverage a deterministic function of wetness, so puddles
  appear even on a completely flat floor (the demo floor is the main case).
- **…but water still answers to geometry** — depressions, holes, and slope
  feet fill first and deepest; slants shed standing water; wet runnels
  streak downhill on slanted ground.
- **Dynamic weather** — intensity surges and eases over tens of seconds
  (no flat constant rate); gusting wind slants the falling rain and snow
  and carries the particle field sideways; raindrop **ripples** play across
  puddles while it rains.

## Version 3.3 — the great optimisation (2026-10-30)

Device report on v3.2: puddles work — but they're square slabs, and
extremely laggy. Two structural fixes:

- **The weather field is baked** — the entire shaped field (puddle
  potential, wet pooling, drift shape) is computed once at app start into a
  2048×2048 texture. The path tracer now does *one texture fetch* per hit
  instead of ~75 noise evaluations on every bounce of every sample.
  The bake never refreshes: wet/snow levels are thresholds applied in the
  shader, not baked values.
- **Organic puddles** — the puddle outline is a threshold sweep of a smooth
  "metaball" potential whose pools sum and merge across neighbours, so
  outlines are rounded and irregular by construction — square corners are
  impossible. The blown bright fringe at slab edges went with them.
- **Ripples moved to the display pass** — a brightness shimmer on puddle
  pixels only, at display rate, costing zero extra tracing. Rain now uses
  the same settled fast path as snow instead of retracing forever.

## Version 3.4 — weather becomes real 3D (2026-10-31)

Device verdict on v3.3: "Snow isn't 3D — it's just whiter and whiter, and
neither are puddles." Correct — both were shading effects. Now they're
geometry:

- **Snow with height** — up-facing surfaces are subdivided and physically
  displaced by the drift field (up to ~30 cm at full cover), with the
  displacement using the same coverage threshold the shader whitens with,
  so silhouette and colour agree. The displaced triangles live in the ray
  tracer's acceleration structure: drifts occlude, reflect, and cast real
  shadows, and objects read as sitting *in* snow. Rebuilds are quantised
  to build-up steps (~1 cm at full cover) and coalesced — shading levels
  stay continuous.
- **Puddles as water sheets** — each pool is a thin sheet of geometry
  floating millimetres above the floor (deeper pools sit higher), with a
  tiny sloped rim where it meets dry ground. At grazing angles the sheet
  has an edge, catches light physically, and reflects with true parallax
  against the floor.
- **Reflection ghosting fixed** — moving the camera had left ~1-second
  after-images of lights in reflections: reprojection history was anchored
  to surface *position*, which is wrong for view-dependent content.
  Reflections (water sheets, near-mirror gloss, wet ground) now restart
  history like glass always did; diffuse surfaces keep full smoothing.
- **Performance** — the weather geometry is static, so it costs *nothing
  per frame*; motion frames additionally get a dynamic-resolution governor
  that trims only when frames demonstrably miss the 60 fps cadence and
  earns resolution back when they lock. Trade, stated plainly: snow rises
  in ~1 cm steps rather than perfectly continuously.
- **Battery Saver glass floor** — glass refraction budgets raised in
  Battery Saver so glass stays clean; the mode still saves power through
  resolution, samples, and settle caps.

## Version 4.0 — the game-engine update: nodes, physics particles, culling (2026-10-31)

- **A node system instead of code** — a "Nodes" sheet in the editor:
  category-coloured cards (Events / Actions / Logic) on a canvas; drag to
  arrange; tap an output pin, then an input pin, to connect. Every node's
  parameters get real editors (sliders, toggles, text, colour picker,
  object/area/texture menus). Graphs save with the project and run in Play.
  24 node kinds — the full reference is in the Manual.
- **Physics particles from imported models** — an emitter node can use any
  object (an imported model or a primitive) as its particle. Per-particle
  settings follow engine depth: **Faces** (model detail, via decimation)
  and **Hitbox** (Disc/Box/Sphere + size) — plus gravity, drag, bounce,
  lifetime, size, speed, spread, direction, rate. Flat particles flutter
  and sway in flight like leaves; on landing, a particle resting with one
  edge unsupported **hinges down over the edge** (the leaf-drooping-off-a-
  block case). Simple ballistic model particles exist alongside.
- **Trigger Areas and streaming** — areas are scene objects you place;
  objects are assigned to areas; nodes can load/unload areas, and Player
  Enters/Exits Area events make walk-in loading a two-node wiring.
- **A dynamic layer for moving things** — node-driven objects are promoted
  out of the static geometry into a second acceleration structure, so
  animating an object never rebuilds the static scene. Leaving Play
  restores the exact pre-Play scene.
- **Culling (light-set, interactive frames)** — while moving or in Play,
  the light set used for direct-light sampling keeps only emitters in
  front of the camera (75° cone) or near it (7 m) — the "scene with a
  bunch of lights" case. Settled frames and stills restore the full light
  set, so culling never costs final image quality. Geometry is *not*
  view-culled (reflections and shadows legitimately need off-screen
  geometry); geometry leaves the world through area unloading instead.
- **Budgets so the new systems can't eat the frame** — dynamic layer capped
  at 90,000 triangles; particle caps per emitter and globally with
  emission pausing (not bursting) at the cap; the settled quality floor
  (full resolution, 768-sample settle cap, High-detail tier above) is
  untouched, and everything from v3.1–v3.4 carries forward unchanged.

## Version 4.1 — the vast node expansion, the great bug hunt, new roads, new icon (2026-10-31)

- **55 node kinds** (31 new). New Events: Player Near Object, Player Falls
  Below. **Spawning**: Spawn Object, Spawn Pattern (Grid/Ring/Scatter/Line,
  reproducible seed), Despawn Objects. **Animation behaviours** (slot-based,
  mutually cancelling): Oscillate, Follow Object, Chase Player, Wander,
  Look At, Scale Object. **Light/material animation**: Pulse Light, Cycle
  Light Colour, Set Object Colour. **Player/camera**: Set Player Speed,
  Set Checkpoint / Respawn Player, Camera Shake, Cut To View. **World**:
  Weather Control (rates, intensity, direct build-up levels), Set
  Environment Light, Set Gravity, Impulse Burst, Show/Hide Area, Stop All
  Particles. **Variables/display**: Variable Maths, Random Value, Show
  Value (`{name}` substitution). **Logic**: Fire Once, Sequence.
- **Custom nodes: deferred, on evidence** — the editor has no
  multi-selection, variables share one global namespace, and pin-mapping
  for flattened composites is ambiguous. The recorded design is
  flatten-at-start macros; a separate node-builder app is the planned
  vehicle.
- **Multi-pass bug hunt** — two independent passes over the node system
  (36 findings) and the renderer (24), a verification pass on the fixes,
  and straggler fixes before shipping. Headlines: light culling had been
  resetting accumulation ~4×/second in Play and never restoring the full
  light set (Play now settles with *all* lights — a quality increase);
  the guide kernel misread gloss on textured surfaces, silently disabling
  reprojection for all of them (fixed — textured puddles no longer
  ghost); timers armed one interval late; the emitter node never fired
  its output; collision ignored objects in the dynamic layer; a Respawn
  bug lifted the player 1 m into the air (caught in pass 2, fixed before
  shipping). Deferred items are recorded in the project log, not hidden.
- **Node-system optimisation** — dictionary-based graph lookups (was
  repeated linear scans), precomputed watcher lists, spawn/promotion
  coalescing, soup-rebuild dirty flags; renderer: culling resets only
  when the culled set changes, O(n²) centroid scans replaced, hitch
  frames excluded from the resolution governor.
- **Roads redrawn at 1024×1024** (was 512) — polished wheel tracks, tar
  snakes, cracks, oil stains, sealed patch rectangles, border decay, and
  worn paint scrubbed hardest in the wheel paths; per-type geometry
  (Highway rumble strips; Intersection concrete corners; Roundabout
  island, apron, yield arcs, gravel corners). Roads moved to their own
  texture array so the higher resolution costs nothing elsewhere.
- **App icon fixed** — the prism now does real optics: the beam rises
  ~22° into the left face (a dead-horizontal beam would trap inside by
  total internal reflection), bends, and exits deviated ~27° as a
  **continuous rainbow fan** (red least, violet most) instead of three
  colour lines.

## Version 4.2 — top bar rebuilt, settle fuzz attacked, spawn offsets, chase freeze fixed (2026-10-31)

- **Top bar rebuilt** — device evidence showed its buttons crushed into tall
  pills with mid-word wrapped labels ("No/de/s"): ten controls ≈ the bar's
  full width *and* zero wrap discipline. Now every label is single-line and
  fixed-size (wrapping is structurally impossible), the title flexes, and
  Render/Export live under a trailing "…" menu; under 1020 pt a compact
  icon row takes over. Play / Nodes / Save stay one tap in both layouts.
- **Fuzz: the display chain, not the tracer** — the denoise faded out
  completely by 96 samples and switched off entirely past 160 when still,
  leaving the 160→768 settle tail displayed raw (exactly the window the
  device screenshot sat in). Now the fade is class-aware: diffuse hands
  off at 0.6× the settle cap, reflections/gloss ride the denoise to the
  cap with wider luminance tolerance. The settled image is bit-identical
  to v4.1 — the same picture, reached sooner.
- **Spawn nodes gain Offset + Spawn at Player** — clones land displaced
  from their anchor by an X/Y/Z offset, or at the player's feet in Play.
- **Chase Player freeze fixed** — three defects matching the device repro
  (Caught → Show Message → Hide, message shown, then freeze): a *hidden*
  chaser kept hunting and re-catching forever (behaviours now idle while
  their object is invisible); the anti-storm budget reset at every drain
  phase, letting a self-sustaining cycle execute ~123k times/second
  (now one budget per tick + a storm breaker); a re-triggered Chase
  started already-armed in range (now starts armed only beyond the
  re-arm radius).

## Version 4.3 — living snow and a proper player (2026-10-31)

- **Snow is different every time** — v4.2's cover was one smooth
  deterministic blanket. The drift field is now multi-scale (broad
  wind-stretched swells, medium lumps, fine detail) and driven by a
  **per-project random seed** stored with the project: two scenes snow
  differently, low cover arrives as patchy drifts in hollows and
  sheltered spots first, and Weather → Snow settings… can **Re-roll**
  a project's pattern. (Projects from before v4.3 keep the classic
  pattern until re-rolled.)
- **Snow depth is yours to set** — accumulation no longer stops at a
  fixed ~30 cm. **Max Snow Depth** (0.05–2 m, default 0.3 m) is the
  target the varied field builds to — drifts overshoot it, exposed
  spots undershoot it. Raising the setting resumes build-up; lowering
  it never bulldozes: deep snow compacts slowly under snowfall and
  melts normally otherwise. "Snow cover %" now reads as progress
  toward the set depth.
- **Objects crush snow** — anything resting in a drift compresses the
  snow under its footprint (smooth falloff, raised rim beside it) and
  sits *in* the snow; dragging parts a live trough that settles into
  the imprint on release; walking in Play presses footprints and a
  compressed path. Resting/drag imprints are derived from object
  positions at rebuild time — nothing sculpted is stored, so
  save/load reproduces them exactly; footprints are session-only and
  fade away.
- **Snow has a memory of who's new** — cover is per-object exposure:
  spawn or place an object at full cover and it appears **clean**,
  whitening and deepening at the scene's build-up rate until it
  catches up. Long-present objects keep their cover.
- **The player is a character now** — a procedural low-poly humanoid
  (~1.75 m, swinging arms/legs, bob, faces its movement) replaces the
  plain sphere: visible standing at the play start in the editor,
  walking with you in Play, and correct in reflections, shadows, and
  node-camera views. Purely visual — eye height (1.7 m), checkpoints,
  respawn, and Chase catch distances are untouched.

## Version 4.4 — snow becomes its own surface (2026-10-31)

Three device findings on v4.3, one fix each:

- **Snow is always snow now** — v4.3's snow was displaced host
  geometry and inherited the host's material, so snow on a glass box
  rendered *as glass*. Snow geometry now carries a dedicated snow
  material (opaque white, rough) whatever it lies on — glass,
  emissive, textured — and the renderer treats the snow sheet as
  finished snow on the shading side too (no wetness, gloss, or second
  snow layer computed over it).
- **Snow no longer follows the shape's triangles** — The user's
  diagnosis: the old snow subdivided the host mesh, so a plain
  two-triangle plane produced huge angular facets. The snow surface
  is now an independent sheet on a regular ~0.20 m lattice draped
  over the scene's up-facing supports (floors, rooftops, box tops at
  their own heights), with the same v4.3 height model underneath —
  same drift field and seeds, same Max Snow Depth, same per-object
  exposure, same crush imprints and footprints. Piles read as piles
  on any geometry. Normals come from the height function itself, and
  cut edges where a support ends get closed sides (skirts), so deep
  snow stays a sealed solid.
- **The dark patches are gone at the source** — host-side analysis
  found the "sometimes snow turns dark" cause: the old construction
  pinned snow height to *zero* along every original host-triangle
  edge (a seam-hiding trick from the 30 cm era). At 1.9 m that carved
  full-depth, light-starved trenches straight through the drifts —
  the dark triangular patches in the screenshots — and the shading
  normals disagreed with the trenched positions by up to ~105°,
  darkening a wide band around every host edge. The lattice sheet
  has no host edges, so neither defect exists. The same audit fixed
  a second crush bug: the ground plane's own footprint was silently
  compressing snow on top of every raised object standing on it.

---

## Version 4.5 — see out of your own head; snow slumps and banks (2026-10-31)

Three device findings on v4.4, fixed in the renderer only (game
logic, eye position, checkpoints, and chase behaviour untouched):

- **Play no longer puts the camera inside the character's head** —
  the v4.3 character's head surrounds the eye point, so first-person
  Play rendered the inside of the head. The figure's head and hair
  now carry a dedicated marker the path tracer honours for the
  camera's own rays only, in Play: the view steps over them. Every
  other ray — reflections, mirrors, shadows, node-camera views —
  still meets the complete character, head included, and looking
  down in Play you see your own torso, arms, and legs. The denoise
  guide pass applies the same rule, so no head-shaped ghost can
  appear in the displayed image.
- **Snow slumps to its angle of repose (~37°)** — v4.4's sheet
  could hold vertical interior faces: lift a block out of deep
  snow and the hole kept box-shaped walls. The sheet build now
  relaxes the snow surface after the imprints are computed —
  neighbouring columns steeper than the repose angle trade height
  until they settle (columns an object is physically pressing on
  are pinned; genuine support steps like a roof edge are the
  object's silhouette and are not relaxed away). Host-side audit
  of the user's scene shape: worst interior slope 81° → 37°, and the
  pit left by a lifted block recovers from flat-bottomed 0 m to a
  smooth 0.76 m hollow (snow volume is conserved — it slumps, it
  does not vanish or fill flat). The relaxation is a pure function
  of the current scene state at each rebuild: deterministic,
  nothing new is stored, save/load is unchanged.
- **Snow banks against objects instead of ringing them with a
  moat** — the old imprint blend compressed snow *outside* the
  footprint too, holding it at the fully-crushed height right at
  the wall and recovering over 30 cm: a flat cleared trench around
  every object (audit: 0.000 m at the wall against a 0.631 m
  drift). Compression now applies only under the footprint, and
  outside it the drift rises toward the wall — about +18% of the
  local drift height at the wall, easing to nothing ~0.8 m out.
  Same audit wall: 0.745 m at the wall, 1.093 m at 0.3 m out, open
  drift beyond.
- **Remaining dark snow, named** — after v4.4 the darkness the user
  still saw concentrated on the steep side walls of the snow mass.
  Two causes, both addressed: (1) those walls were largely the
  near-vertical interior faces the repose relaxation now removes;
  (2) the sheet's true boundary sides (skirts) were shaded with
  purely horizontal normals, which under the gradient sky receive
  only ~70% of the light a snow top gets (measured by integrating
  the sky model) — they read as dark slate. Skirt normals now tilt
  up like a steep snow slope (~95% response) and an extra blanket
  darkening factor on skirts is removed. Sheet normals generally
  now come from the relaxed surface itself, so shading and
  silhouette agree exactly.

---

## Version 4.6 — snow remembers being pushed; multi-select (2026-10-31)

Three asks from the same afternoon, one build:

- **Plowing — displacement with memory.** Dragging an object through
  snow used to only dent the snow around its *current* footprint;
  the drift healed pristine behind it. Now a recorder watches every
  object that moves while touching snow: the swept path is carved
  to the block's underside depth, and the removed snow is deposited
  ahead of the motion (the bow wave) and along both sides (berms) —
  volume for volume (host audit: 18.31 m³ removed, 18.31 m³
  deposited; the finished trail measures a full-depth trough —
  0.00 m against a 1.24–1.60 m field — with ridges +0.2…+0.7 m on
  both edges). The trail is the app's first piece of snow *state*:
  a sparse displacement layer on a fixed 0.20 m world grid, saved
  in the project file (a hard plow of a big scene adds ~30–60 KB),
  still there after save + reopen. Fresh snowfall gradually fills
  trails (gone after ~0.25 m of weather change); **Re-roll** and
  **Clear build-up** restore pristine snow. The packed trail floor
  holds its depth; loose berms slump to the usual 37°. Teleports
  (spawn-at-player, undo jumps) plow nothing — only genuine drags
  carve. Editor drags are the full case; in Play, objects moved by
  node behaviours carve only when a rebuild happens to observe the
  motion. The audit trail for this one was long — three feedback
  designs for the layer diverged or ratcheted on paper before the
  shipped one (recorder-writes-only, nearest-cell reads) proved
  stable; the numbers above are from the final archived script
  (`v46-snow-audit.py`).
- **New blocks land clean — a real defect, found and fixed.** The
  per-object snow-ageing from v4.3 registered new objects at zero,
  but the renderer's copy of the exposure map wasn't updated until
  the next frame — and both snow paths (the sheet's support
  attribution and the shading factor) fall back to the *ambient*
  amount for an object the renderer doesn't know yet. A block added
  in deep snow appeared with a full snow cap and full whitening.
  Both registration sites (Add and Import) now sync the renderer's
  map before the rebuild the placement triggers. Placement audit:
  0.000 m of snow on sheet and shading paths at placement, full
  cover after the usual build-up.
- **Accumulation is independent of height — verified, no fix
  needed.** The snow-depth function reads only (x, z, amount): no
  code path ties an object's snow to the ground, so a clean block
  raised 5 m stacks the full set depth on its own schedule, and a
  lifted snowy block's depth is bit-identical before and after the
  move (audit-checked all three cases).
- **Multi Select tool.** A **Multi** toggle in the object-tree
  header: tap objects in the view or the tree to build a selection
  set — members are tinted in the 3D view and checkmarked in the
  tree. Dragging any member moves the whole set (each member still
  plows its own trail through snow); Delete removes the set as a
  single Undo step. The properties panel shows "N objects
  selected" with Hide/Show All, Delete All and Clear. Name,
  transform and material editing remain single-object (the panel
  says so). Tapping empty space, switching the tool off, or
  entering Play clears the set. No permanent grouping, no
  multi-select in Play or the node editor. (Cmd+click was asked
  for "if it falls out naturally" — it doesn't: UIKit tap
  recognizers carry no modifier flags, so the tool is the
  mechanism.)

---

## Version 4.7 — trails heal at full depth; temperature (2026-10-31)

Two weather asks, one build:

- **Plow trails heal while it is actively snowing, even at max
  depth.** In v4.6 a trail faded only when the accumulation A
  *changed* by 0.25 m — but at Max Snow Depth A stops changing,
  so a trail carved at full depth was permanent while snow kept
  falling on it. The app now keeps a **cumulative snowfall clock**
  (total metres fallen, uncapped, saved with the project): every
  plow cell is stamped with it, and a trail's fade is driven by
  whichever has moved more — the depth change or the snowfall
  since the trail was made. At 2 m depth and full intensity the
  clock runs at 0.02 m/s, so a trail is gone after **12.5 s** of
  snowfall; with the snow off, a trail at full depth is untouched
  after an hour. Below the cap nothing changes: depth and
  snowfall advance together, so the fade is exactly v4.6's.
  Trails saved by v4.6 load with their heal clock starting at
  load. Host audit (`v47-weather-audit.py`): 57/57 checks pass.
- **Temperature.** A new slider in **Snow settings**
  (**−20…+30 °C**), deliberately **unset by default** — an unset
  temperature is the old behaviour, bit for bit, and old projects
  open unchanged. Set it and temperature governs three things:
  **phase** (rain falls as snow at/below 0 °C; snow falls as rain
  above it — including weather driven by nodes), **snowpack
  melt** (a full pack lasts ~900 s at +2.5 °C, 450 s at +5 °C,
  225 s at +10 °C, ~112 s at +20 °C, and the meltwater wets the
  ground), and **puddle freeze** (puddles ice over in 240 s at
  0 °C, 120 s at −5 °C, 48 s at −20 °C; frozen puddles shade as
  pale mirror ice and stop rippling; thaw takes ~164 s at
  +1 °C, 90 s at +10 °C, 60 s at +20 °C). The objects panel shows
  a live line — "−4.5 °C — puddles freezing", "+10.0 °C — snow
  melting" — only while a temperature is set. **Clear
  temperature** in the sheet returns the scene to the legacy
  behaviour. Temperature, snowfall total and freeze fraction all
  save with the project (and stay out of the file entirely while
  they're at their unset/zero values).

---

---

## Version 4.8 — snow and water become a GPU simulation (2026-10-31)

The user's verdict on v4.7's deep snow was "really messy" — spiky,
jagged masses — with a direction: replace the growing mesh with
a simulation on the GPU, for rain as well as snow. Five steering
items came with it.

- **Persistent GPU simulation.** Snow depth and water depth now
  live in per-vertex GPU buffers on the same 0.20 m lattice,
  stepped every frame by compute kernels: deposition toward a
  per-vertex ceiling (the old height function's one remaining
  job, besides providing the load-time starting surface),
  temperature/legacy melt sinks, compaction, and pairwise
  red-black flow transfers — snow slumps toward the 37° repose
  angle plus a small rounding diffusion (0.8% per visit) that
  relaxes piles into soft mounds; water levels its free surface,
  is confined to the pooling field, drains at support edges and
  across steps, and stops flowing when frozen. Flow is
  mass-conserving by construction (audit: 0.00 relative error
  per step). The rendered sheets are skins over the sim state.
  **Stated limitation:** the ray-tracing acceleration structure
  is still rebuilt from the sim state at the existing quantised
  cadence (>= 0.8 s), not refit per frame — the sim itself steps
  on the GPU every frame. Host audit (`v48-sim-audit.py`,
  33/33): at ~1.1 m the old surface measured slope 75°, spike
  0.357 m, ledge 0.761 m; the sim measures 16°, 0.008 m,
  0.057 m. At 2.0 m: 83° / 1.568 m / 1.710 m before, 20° /
  0.009 m / 0.073 m after. Drift contrast is retained (std
  0.459 -> 0.311 at deposition equilibrium); 10 sim-minutes at
  max rates stays bounded with no NaNs; a freshly loaded
  project's first surface is the v4.7 surface exactly
  (max-abs-diff 0.000 m), and a lifted block's snow patch
  remaps with it (interior mass within 1.4%).
- **Puddles round, water stays down.** Pool outlines are level
  sets of the simulated depth with a millimetre rim fade:
  isoperimetric quotient 1.23 (v4.7's square-ish mask) -> 1.16
  (a circle is 1.0). Water exists only on up-facing support:
  zero water on unsupported cells, edge ring drained to 0.8 mm,
  pool surface slope <= 0.3°.
- **Water reflection resolution.** New setting (Full / Half /
  Quarter; Full = v4.7 quality, the default). Below Full, a
  rotating per-frame pattern traces the true reflection on
  that fraction of water pixels and fills the rest from the
  environment — water reflection rays per frame scale by
  roughly the chosen factor.
- **Precipitation redrawn.** Rain alpha is a capsule
  signed-distance shape and snow a radial disc reaching
  exactly zero by 0.85 of the half-extent: alpha is zero near
  every quad border for both, so no square silhouette can
  show. Counts, sizes and motion are unchanged.
- Everything else is preserved behaviour-for-behaviour:
  drift pattern + seed + Re-roll, Max Snow Depth (0.05-2.0 m),
  per-object exposure (fresh objects start clean at any
  height), plowing with 100% volume deposit + snowfall
  healing + melt fade, footprints, banking, temperature
  (unset = v4.7 exactly), ice look, and the save format
  (no new keys — the sim re-derives from saved state).

## Version 4.9 — the optimisation pass: same picture, less waiting (2026-10-31)

A full-app performance pass under the standing rule: **the settled
image does not change** — v4.9 computes exactly what v4.8 computed
(same tracer, denoiser, sim math; the sim audit passes 33/33 with
identical results) and only removes stalls, redundant uploads, and
UI churn.

- **No more rebuild stalls.** Scene (re)builds used to block the
  frame thread while the ray-tracing acceleration structure was
  rebuilt (`waitUntilCompleted` on every rebuild). The structure
  now builds in the background and swaps in atomically between
  frames — a frame never sees a half-updated scene, and picking
  always matches what's on screen. Rapid rebuilds coalesce to the
  latest state. Save thumbnails and PNG exports still render the
  newest geometry (they synchronously finish any pending build
  first, as before).
- **Play motion is incremental.** Node-driven objects used to be
  re-uploaded whole (both vertex buffers, every frame) with a full
  acceleration-structure rebuild + CPU wait per frame. Now only
  the moved objects' vertices are rewritten in place and the
  structure is *refitted* (Metal's in-place update for moved
  geometry — same triangles, same hit results), fenced only by the
  previous frame's completion. Collision data patches the same
  way, per moved object instead of scene-wide.
- **Cheaper weather skins.** The ~0.8 s weather-cadence rebuilds
  (snow skin, imprints, plow trails) previously re-baked every
  object in the scene. They now verify a content fingerprint and
  reuse the unchanged scene geometry verbatim, re-running only
  the weather sheet; the support sampling is cached the same way,
  the plow recorder reuses the sheet's own samples instead of
  re-sampling per vertex, and the sim's settle scan reads its
  buffers in place instead of copying ~3 MB per scan. A one-shot
  self-check compares the shortcut's first result against a full
  rebuild and disables the shortcut permanently if they ever
  disagree. (The ray-traced snow silhouette still refreshes on
  the same cadence as v4.8 — what changed is that it no longer
  stalls or re-bakes the scene to do it.)
- **Simulation bookkeeping halved.** The snow/water sim tick
  merged its paired dispatches (snow/water flows are disjoint):
  worst case 52 dispatches + 52 barriers per frame → 28 + 27,
  0 when neither field is active. Every transfer's arithmetic and
  ordering is unchanged — this is dispatch organisation, not new
  math.
- **Material updates are surgical.** Exposure refresh (the
  per-snow-trigger update) used to re-upload the entire material
  table per dynamic object, plus a full emissive-light rescan +
  CDF re-upload each time. The material buffer is now persistent
  with per-slot diffing — a snow refresh writes only the slots
  whose values changed (one 80-byte material each) — and the
  emissive index is skipped entirely when no emissive-relevant
  value changed.
- **Calmer panels.** The side panels observed the whole editing
  session, so weather readouts re-evaluated the object tree and
  properties on every tick. The footer now observes a dedicated
  display object that only publishes when a displayed string
  actually changes (same strings, same styling); per-frame state
  writes throughout the session are change-guarded (a node
  re-sending an object's current transform/material is now a
  no-op instead of a publish + rebuild); area member counts are
  computed in one pass instead of per-row scans, and the
  multi-select member list is memoised between edits.
- **Examined and deliberately NOT changed:** sleeping the frame
  chain when "nothing is falling" — precipitation animates at
  every intensity by design (a 0.35 visibility floor in the
  shader and the draw count), so frames are never actually
  identical while weather is on; gating them would freeze rain
  mid-air. Making that gate possible would require removing the
  floors when the intensity scale is exactly 0 — a behaviour
  change, recorded here as an offered choice, not shipped.

## Version 4.10 — idle freeze + the final render (2026-10-31)

Two features, one theme: the scene's time and the scene's best
picture are now both under explicit control.

- **Idle freeze.** After an adjustable delay with no detected
  input, the scene freezes until the next input — scene time
  stops entirely (no weather advance, no node timers, no
  movement debt) and the frame chain stops with it, so a frozen
  scene costs no draws at all. The delay lives in the More menu
  (**Idle freeze: Off / 15 s / 30 s / 1 min / 2 min / 5 min**,
  default 1 min) as an in-memory preference beside Quality and
  Water reflections, and changing it applies to an idle period
  already under way. Input is detected comprehensively: viewport
  gestures, camera activity, hardware keys, the Play joystick,
  the discrete actions (Save, exports, Play/Pause, Weather,
  Undo/Redo), and a window-level touch observer that listens to
  every touch in the editor window without ever intercepting one
  (its one honest boundary: the on-screen keyboard lives in a
  separate window — its keystrokes are not observed; hardware
  keys are). The wake is engineered to have **no time jump**:
  the wake frame's delta is exactly zero, and every wall-clock
  scene clock (camera shake, message banner, precipitation
  phase, weather-sim clocks) is rebased by the frozen span, so
  frozen rain hangs mid-air and resumes from the same phase.
  Freezing never interrupts a scene rebuild in flight or a
  final render — it engages on the first clear frame after
  them. A host audit mirrors the state machine
  (`v410-idle-audit.py`, 51/51): no engage before the delay,
  protected operations block and then release it, zero scene
  time across a scripted 600 s freeze, wake-frame dt exactly 0.
- **Render Image…** (More menu) is the offline-quality still:
  the current view at native resolution, the full light set,
  **16 bounces / 16 glass crossings / 4,096 samples**, built
  progressively in 128-sample chunks into one accumulation with
  a live preview, sample count, and percentage in its sheet.
  Cancel is bounded by one chunk and keeps the best-so-far
  image (which then goes through the same final denoise +
  tonemap as a completed render). Finished renders save to
  **Photos**, **Share**, or **Files** (`Documents/Renders/`).
  The live scene is paused for the render's duration, and a
  render in flight blocks the idle freeze. No 2× resolution
  option: at native resolution 4,096 samples is already
  minutes-class, and 2× would quadruple it.
- **Rain refracts in the final render.** The still's drops are
  true geometry, not the live overlay's sprites: the exact
  drops the frozen live view was showing (same seed field,
  same placement math, same last-encoded time/intensity/wind)
  are rebuilt as elongated glass droplets — the **384 nearest**
  drops, **36 triangles each** at the live streak's size —
  carried in the dynamic layer for the render's duration only
  and traced through the existing glass transmission path at
  **water IOR 1.33**, so each drop bends the scene behind it.
  The pre-render dynamic state is restored verbatim afterwards
  on every exit path. Snow gets no still geometry (rain only);
  interactive precipitation is untouched.
- Anchors held: the game runtime and the shader source are
  byte-identical to v4.9; the top bar and panel layout are
  unchanged apart from the two new More-menu entries; weather
  rates and tracer/denoise math are untouched (the v4.8 sim
  audit still passes 33/33).

## Version 4.10.1 — hotfix (2026-11-01)

A device report on 4.10 ("as soon as snow started accumulating the app
crashed", plus the tree-panel header still overflowing its panel) sent
this hotfix back the same morning:

- **Snow-accumulation crash hunt — hardened + instrumented.** A full
  static audit of the accumulation path (GPU sim buffers/kernels, sheet
  build, plow recorder, session trigger, zero-copy buffer reads) found
  no single trap that fires for clean geometry the moment snow starts,
  so every plausible hazard on the path was hardened instead: raw
  `Int()` conversions of world coordinates (which trap on NaN or
  extreme values) in the support index, plow layer and recorder now
  clamp or skip non-finite input; the support index no longer registers
  a scene-spanning triangle in every grid cell it crosses; sim buffer
  reads/writes clamp to the buffers' real lengths and the sim tick
  refuses to step on a mismatched buffer set; a NaN support height can
  no longer poison the sheet; persisted snow amounts/exposures decoded
  from a project file can't trap the step computations; three exposure
  loops no longer write into a dictionary while iterating its keys; and
  the dynamic layer's GPU slice writes are range-checked like their CPU
  mirrors already were. None of these change the simulation's math,
  rates, or any rendered value on the normal path (the v4.8 sim audit
  still passes 33/33, the idle-freeze audit 51/51). And the path is now
  **instrumented**: each step writes a short tag ("sheet:simSync",
  "tick:dispatch", …) that persists across a crash — after a relaunch,
  the panel footer shows **"Last step: …"** naming where the previous
  session stopped, until this session runs weather itself.
- **Tree-panel header fits its panel.** The "Objects" heading and the
  Multi / Add controls shared one row whose total width exceeded the
  264 pt panel, so the centred row overflowed both edges (the heading
  clipped to "bjects", the buttons spilling under the viewport). The
  header is now two lines — heading on its own line, Multi and Add on
  the next — at the same natural sizes. Every other panel header/footer
  row was re-checked by composed-width arithmetic against its panel;
  the tree header was the only one over.
- **Areas, explained in the panel.** The Areas helper text now says
  what an area is with a buildable example: an invisible zone over a
  doorway, grouped with objects, where a Player Enters Area node can
  start a Chase Player node when you walk in.

## Version 4.10.2 — crash round 2 (2026-11-01)

The 4.10.1 hotfix did not stop the crash — snow starting still killed
the app — so this round audited the failure modes a code read of the
Swift side cannot see: what the GPU does with the builds and buffers it
is handed. Three independent audit lanes (sim dispatch, acceleration-
structure build/swap, first-tick behaviour) found the following, all
fixed:

- **A failed geometry build could be installed as if it had
  succeeded.** The ray-tracing acceleration structure is built on the
  GPU in the background and swapped in when done — but nothing checked
  whether the build had actually succeeded. A build whose GPU work
  errored was treated as "not done yet" forever by the live path (the
  snow sheet would simply never appear), and the synchronous flush used
  by save-thumbnails, PNG export and the final render installed the
  never-built structure outright — the next traced frame then
  intersected a corrupt structure and the process died. The first snow
  build is the largest structure the app ever builds, so it is the one
  most exposed to a build error. Failed builds are now recognised,
  discarded unswapped, tagged in the step trail ("rebuild:stageError"),
  and retried by the next rebuild; build encoding failures are
  distinguished from "empty scene, nothing to build".
- **Bad snow values could reach the geometry unguarded.** Between the
  simulation buffers and the sheet vertices there was a length check
  but no finiteness check: one NaN or infinite depth — from a corrupt
  persisted per-object exposure amount, or from a zero-scale object
  whose inverse transform is infinite (such an object renders fine
  while dry and only touches the snow code once accumulation starts) —
  flowed into sheet vertices, their normals, and the structure build,
  and the simulation's own flow would spread it to neighbouring cells
  first. The readback now replaces any non-finite value with 0 and
  counts replacements (the footer can show "NaN clamps: N"); persisted
  exposure amounts are sanitised where they are read; singular
  intruders are rejected before they can poison an imprint.
- **The simulation tick now reports itself, and the CPU no longer
  races it.** The tick's command buffer previously had no completion
  handling at all — a GPU fault in it was indistinguishable from any
  other silent death. It now records "tick:completed" (or
  "tick:gpuError") into the same persisted step trail, and CPU code
  that reads or writes the shared simulation buffers first waits for
  the last tick to finish instead of racing its kernels.
- Also closed, from the same audits: the picking raycast snapshots the
  scene arrays before looping (a background swap could otherwise
  replace them mid-pick), and a currently-unused simulation kernel's
  one unbounded GPU write is guarded before anyone wires it up. The
  audits also *cleared* ground round 1 could not: the sim kernels'
  index arithmetic is provably in-bounds at every lattice size, the
  build-swap ordering against the draw loop is correct, and no
  same-thread count disagreement exists in the sheet rebuild.

Simulation math and rates are unchanged (the v4.8 sim audit still
passes 33/33; the idle-freeze audit 51/51). If the crash ever recurs,
the step trail is now fine enough to name the failing link directly:
the footer distinguishes a GPU error in the tick, a failed structure
build, a swap, and each sheet stage.

## Version 4.10.3 — the snow crash, found by running it (2026-11-01)

4.10.2 did not stop the crash either. Both hotfix rounds had been
code *reading*; this round changed method: the app was driven,
off-device, through the exact accumulation-start sequence — demo
scene, weather set to snow, real frame ticks pumped in the iOS
Simulator — and the crash **reproduced on the first try**, dying at
the same point in 10 of 11 configurations with the same Metal
assertion in every crash report:

> `Command encoder released without endEncoding` — the process
> aborts when a GPU command encoder is discarded unfinished.

The guilty code was in the weather simulation tick. Each frame the
tick created its GPU command buffer and compute encoder *first*,
and only afterwards decided whether any snow or water was actually
active; when neither was, it returned — leaving the encoder
un-ended and un-committed, and Metal killed the process when the
encoder was released. At accumulation start that "neither active"
case is guaranteed: the first snow-sheet build fires while every
depth ceiling is still zero (fresh snow has not yet earned any
depth anywhere), so the second tick after the sheet appears always
took the leaking branch. That is why the app died the moment snow
started accumulating, and why the pre-simulation build (v4.7 and
earlier) never crashed: it had no tick to leak.

**The fix is structural, not a clamp:** the tick now makes the
activity decision *before* any GPU object exists — when neither
kind is active it returns having created nothing — and no return
path sits between an encoder's creation and its `endEncoding`
anywhere in the function. The simulation's math, rates, and
dispatch organisation are untouched.

**Proof, both directions.** Without the fix, the simulator matrix
crashed 10 of 11 runs at the identical marker with the identical
assertion (only the run with the sim tick disabled survived —
which is what isolated the tick). With the fix, the same matrix
passes: demo scene, plane-only, zero-scale and huge objects, a
large plane, an idle-freeze/wake cycle mid-accumulation, and each
bisect toggle individually — snow accumulates to a settled sheet
in every run, and no crash report or assertion appears in any
log. (One variant, which arms idle freeze on the exact frame the
geometry build swaps, ran past its test driver's own time budget
without crashing — 1,081 sim ticks and accumulation still
progressing when the driver stopped waiting.) The v4.8 sim audit
still passes 33/33 and the idle-freeze audit 51/51.

The bisect toggles and the scripted repro driver stay in the app,
inert unless launched with a special argument, so any future
weather crash can be bisected the same mechanical way instead of
by reading code.

## Version 4.10.4 — drag and spawn fixes; the iPhone gate (2026-11-01)

Two snow behaviours that the simulator audits had proven on paper
get their final form. **Dragging an object through deep snow** no
longer misbehaves: the displaced snow travels with the drag the way
the model intends instead of snapping or leaving artefacts behind.
And an object **spawned into existing snow** now lands with its top
genuinely blank — the spawn-blank behaviour the exposure model
always promised — displacing its own volume into banks around its
base rather than starting pre-buried or pre-covered.

This is also the version where the app stops pretending to be a
phone app. On an **iPhone** it now shows a single notice — "You
need an iPad for this to work" — and nothing else; the project was
always built for the iPad's screen, and a phone build that half-ran
the editor helped nobody. (The package still declares both device
families so the one build serves both; the gate is the app's own.)

**One honest correction to earlier wording:** the 2.00 m **Max Snow
Depth** setting is a *per-cell ceiling*, not a hard bound on the
snow surface — drifts and banks could legitimately stand at the
ceiling in one cell while the surface as drawn passed above or
below it elsewhere. The setting page's promise is narrower than
earlier versions implied.

**Device note, and it matters:** on the iPad itself, the verdict
came back that the snow *still messed up*. The simulator-proven
fixes had landed; the behaviour on the real device in a real scene
had not matched them. That report — not any failing audit — is
what sent snow back to the drawing board for v4.11.

## Version 4.11 — snow becomes a granular simulation (2026-11-02)

The entire rule stack that had grown from v4.2 to v4.10.4 —
ceilings, exposure clocks, the plough layer, drag caps, relaxation
patches — is **retired in one move** and replaced by a
physics-first granular simulation on the same ~0.20 m lattice.
Each column now carries snow **mass**, **density** (ρ from 0.30 to
1.00) and a sinter **age**; depth is derived (height = mass /
density). Thirty times a second the model runs its substeps:
snowfall enters as an input flux only (the old ceiling survives
only as a deposition gate, never as a shape), melt and compaction
apply, and gravity moves mass between neighbouring columns in
pairwise, exactly-conserving transfers limited by a
**temperature-dependent angle of repose** — about 34° for cold
snow, up to 38° near freezing. Objects are displacing volume
boundaries, not decoration: placing one pushes its own volume of
snow aside into repose-shaped banks and lands with a bare top;
removing one lets the hole collapse; roofs shed to repose. (A
particle method — MPM — was costed and rejected: whole-scene snow
at particle spacing runs to roughly 5–6 million particles, which
does not fit an M3 alongside a hardware path tracer at 60 FPS.
The model's honest limits: one height per column means no
overhangs or cornices, no airborne grains, and no wind transport
beyond the deposition drift field.)

**The audits were rebuilt for it** — a new in-app physics suite,
plus a line-for-line float32 mirror of the kernel math checked
host-side (20/20). On the final build: a dropped pile's cone flank
settles at **33.97°** against a 34° ± 3 bar with mass exact to the
last digit; injected spikes settle to a maximum slope of 34.08°
with zero spike remaining (the retired model scores 79.7° on the
same test); dragging a box through 1.8 m of snow carves its trench
with total mass drift of −0.0003%; a spawn lands with its top at
exactly zero and banks at 34.13°; a 45° roof sheds from 0.50 m to
0.000 while a flat roof keeps its snow. **Zero crashes across all
22 CI runs**, and the water system was not touched — its kernels
are byte-identical and its v4.8 audit re-ran 33/33. The simulation
step's cost was not directly instrumented; the repro harness pumps
full ticks at interactive rates in the simulator, and the device
remains the judge of performance.

**Device note — the honest end of this model.** On the iPad, the
user placed a 1 m box into 0.77 m of snow and the snow formed a
**big smooth ramp climbing the box's vertical face**, meeting the
wall far above the field depth instead of at it. The audits had
passed because they were structurally blind to exactly this: the
slope metric excluded edges ending at a bare object wall, and no
check measured bank height against exposure. The mechanism, once
dug out: an object sitting fully on the ground gives the
displacement code no overlap to compact against, so **all** of the
snow that was under its footprint is transferred into the single
ring of cells around its base — at these proportions that ring
reaches about the box's full height — and the snow sheet then
renders the step up to the box's top as one smooth slope. Two
real defects, compounding.

That report triggered a research pass into how shipped games
actually do snow, with Red Dead Redemption 2 as the reference.
The finding: almost none of them simulate mass-conserving flow at
all. The industry stack is **authored snow depth**, **carve-only
deformation** (feet and objects push snow *down*; no operation
ever raises it), and **shader accumulation gated by how up-facing
a surface is** — which is exactly zero on a vertical face, so snow
cannot climb a wall *by construction*. Real snow agrees: in still
air the contact line sits at field depth, and measured equilibrium
drift profiles run at 7–14°, not at a granular repose angle.

So the granular model is **retired**, one day after it shipped.
**v4.12, in development**, replaces it with the games-standard
model: authored depth, carve-only deformation with persistent
trails, and up-facing-gated shader accumulation with RDR2-style
glint and caked-snow coating. This entry stays in the record
because the rebuild was real work, honestly audited — and because
the device outranks the audit, every time.

---

## Version 4.12 — snow, the way games do it (2026-11-02)

The snow model is replaced wholesale. v4.11's physics-first
granular simulation (per-column mass, density and repose flow)
passed its host audits but failed on the device in the way
that matters: a 1 m box placed in 0.77 m of snow wore a big
smooth ramp of snow climbing its vertical face. The user's
direction was "do what games typically do", then named the
reference: **Red Dead Redemption 2**. Research into production
snow (Arkham Origins, Rise of the Tomb Raider, God of War,
RDR2) found the industry stack, and v4.12 is that stack:

- **Authored drift field, no flow.** Ground snow is a depth
  field on the existing 0.20 m lattice over ground supports,
  evolving toward the drift-pattern ceiling exactly as the
  field always has (growth, compaction, melt unchanged).
  Nothing moves sideways; there is no repose angle and no
  banking — the two mechanisms that built v4.11's wall ramp
  (a support-switch transfer with no under-object reservoir,
  and a sheet that interpolated the support step into a
  slope) are deleted by construction.
- **Carve-only deformation.** Objects and the player only
  ever REMOVE rendered snow: occupancy compresses the sheet
  toward the ground, dragging carves a trough, and the player
  wades a body-depth trough through deep snow in Play. The
  only permitted raise is a small visual lip along a
  depression's edge, capped at 0.10 m, confined to vertices
  that keep at least half their own snow, and never above the
  authored depth plus that cap; nothing piles ahead of or
  beside a moving object. A placed object sits IN the snow —
  contact line at the local field depth, no bank, no moat —
  and removing it leaves a compressed patch. Depressions
  refill gradually (faster while snow falls;
  temperature-flavoured), and are session state: projects
  save exactly as before, and a reopened project starts from
  the pristine field, objects re-seating within seconds.
- **Snow on objects is a coating, not geometry.** A material
  layer gated by up-facing-ness x exposure with a
  noise-broken snowline: flat tops cake over patchily, steep
  roofs take little, vertical faces take **exactly zero** —
  by construction in the shared accumulation function, and
  proven on the GPU by a dedicated audit kernel. Lattice snow
  no longer rides on object tops; the coating is its
  replacement.
- **RDR2 touches.** The sheet keeps detailed two-scale
  normals so troughs and tracks read at glancing angles; the
  snow material (sheet and coating) gains the glint/sparkle
  stage — sparse view-dependent sparkles in the light; the
  coating reads as caked snow, broken up by noise rather than
  painted on uniformly. (RDR2's hardware tessellation stage
  and its clothing-snow system are not ported; the sheet is
  displaced on the same 0.20 m lattice as before.)
- **Audits**, aimed at exactly the holes v4.11's suite had.
  In-simulator verdicts (repro run 41451080339): wall contact
  at the user's own settings (0.77 m of 1.80 m) — 17 contact
  vertices, median deviation from the local field surface
  0.0015 m, worst 0.0614 m; wall band violations 0; snow
  under the box 0.0000 m. The deep-field variant (1.80 m,
  box placed on the deepest drift crest, perimeter field
  1.6119 m) — 119 contact vertices, median deviation
  0.0015 m, worst 0.0954 m; violations 0; under-box 0.0000 m.
  Drag-carve: invariant violations 0 across a 5.8 m drag in
  1.80 m snow; trough mean 0.0409 m against a 1.3560 m field;
  after a 20 s rest the trough persists at 0.1308 m; under-box
  0.0000 m. Refill: an abandoned patch recovers 0.0000 ->
  0.4028 m in 90 s (field 1.2683 m, no overshoot). Shader
  gate on the GPU: vertical face exactly 0.000000, flat top
  1.000000, 45-degree roof 0.636056. Host mirrors: the new
  snow kernels 15/15, the untouched water audit still 33/33.
  Crash matrix: every audit variant and lifecycle run
  completed without a crash report (one freeze variant's
  arming observation timed out on the runner three rounds
  running — the app alive and ticking each time, the freeze
  machinery itself passing in its sibling variant); the
  iPhone gate check passed on both idioms. Snow tick cost in
  the harness measured 0.012-0.026 ms of CPU encode time per
  tick (encode time only, not GPU or device frame time —
  the device remains the judge of smoothness, as always).

## Version 4.13 — Applied Gravity; snow and 3D water removed (2026-11-02)

Two commissions landed, and one system left.

**Applied Gravity node (56 kinds).** A new Actions node: wire
an object to it and in Play the object falls from where it
was placed, accelerates at the node's Gravity strength
(m/s², default 9.8, editable −50…50 — 0 floats, negative
falls up), and lands on the highest support beneath it —
ground or another object's top — coming to rest exactly on
the surface. It is substepped (≤1/120 s) and swept against
the runtime collision grid, so it cannot tunnel through
thin supports; resting objects re-verify their support
every tick and fall again if it is removed mid-Play (Hide a
box out from under a landed box and the landed box falls).
**out** fires when gravity takes hold, **Landed** when the
object comes to rest after a real fall (>1 cm of travel);
an optional Bounce (default 0) rebounds off hard impacts.
Editor state is untouched — Play works on the snapshot, as
with every other behaviour. Host audits: a box dropped 3 m
lands at the analytic fall time (0.783 s measured vs
0.782 s), follows the quadratic curve (half-time fraction
0.262), rests with zero drift; strength 0 hangs motionless;
an unwired control never moves; a −9.8 box rises and rests
under a ceiling; box-on-box landing and the
support-removed refall both pass; graphs using the node
save and reload intact, and pre-4.13 projects load
unchanged (the kind is purely additive).

**Snow and 3D water are removed.** v4.12's snow was
accepted in the morning ("Looks good") and condemned the
same morning by harder use: at maximum depth the sheet
tore around a buried box being put back (dark slits, a
stray fin), a slit appeared in the open field after the
box was lifted out, walking through deep snow left a chain
of jagged craters instead of a wading trough, and the
whole scene was, in the user's words, "laggy as hell". The
user's call: "just remove snow entirely until we can get a
good working version — there is so much wrong with it."
The 3D pooling/puddle water went the same way an hour
later ("make rain just a slowly sheening wet ground… no
3D") — it had the same climbing instincts. So v4.13
deletes both systems outright (the sheet, the lattice
simulation, the coating, temperature and its phases, the
puddle model and its reflection controls) rather than
patching them. A good snow version is being proven in a
separate test app first. Rain is now exactly two things:
the falling streaks (unchanged) and one flat ground-wetness
value that ramps while it rains and dries when it stops,
darkening and glossing up-facing surfaces. Projects saved
by the snow-era versions open normally — their stored snow
state is ignored and the weather starts off.

**Optimisation is the third headline.** The deep-snow lag
had a structural cause: weather build-up was quantised
into geometry steps, and every step rebuilt the scene's
geometry and acceleration structures — dozens of full
rebuilds a minute while weather ran, more while objects
moved. With the weather systems gone, weather never
touches geometry at all: the per-frame weather cost in the
rain bench scene is 0.16 ms of CPU (mean frame-tick time),
and a geometry edit costs one rebuild (35 ms in the bench
scene) only when an edit actually happens. The removal
itself is the rest of the optimisation: Renderer.swift
shrinks from ~7,400 to ~4,480 lines and the shader from
2,254 to 1,403, and the per-frame tick no longer runs a
snow simulation, a sheet pass, or a weather-field bake.
Settled image quality is unchanged — same samples,
bounces and resolution at rest; the savings are all
work-not-done, not quality-not-rendered. True device FPS
remains the device's verdict.

One cross-cutting fix fell out of the gravity audits: the
runtime collision grid now follows promoted-object
visibility (Hide/Show in Play). It previously only
followed the legacy static path, so a hidden box kept
colliding — nothing could fall through where it had stood.

---

*Every version above was verified by automated build and inspection before
delivery; the final judge of each was the device it runs on, and several
versions (3.1, 3.2, 3.3, 3.4) exist because device reports sent the work
back. Build-verified is never the same claim as device-verified.*
