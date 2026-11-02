# LumenScene — Manual

What the app does, how it works, and what every node does.

LumenScene is a 3D scene editor for iPad whose pictures are computed by a
real path tracer running on the M3's hardware ray-tracing units — not a
game-engine rasteriser with faked lighting. Objects you mark as emissive
*are* the lights. Glass genuinely refracts. Weather builds up in the scene
and, since v3.4, has real physical height. Since v4.0, a node system gives
scenes behaviour — particles, triggers, streaming areas — with no code;
v4.1 expanded it to **55 node kinds** built to modify and shape the scene
in meaningful ways.

---

## 1. The editor

- **Home screen** — your projects, each with a thumbnail of the view at
  last save, plus the **Texture Designer**.
- **Layout** — object tree (left), 3D viewport (centre), properties
  (right). Top bar: project name, Undo/Redo, **Nodes**, Play/Pause, Save,
  Export, Render menu, Weather menu.
- **Adding things** — primitives from the Add menu, models via Import
  (STL / OBJ / USD / USDZ) from the in-app file list, and Trigger Areas
  from the same Add menu.
- **Moving things** — select in the tree or viewport; drag the selected
  object in the viewport or use the transform rows. While *anything* is
  being dragged, all light emission on the stage cuts (and the dragged
  emissive/glass object previews cheaply); everything restores on release.
- **Multi Select (v4.6)** — the **Multi** toggle at the top of the object
  tree turns on multi-select: tap objects (in the view or the tree) to
  add or remove them from the selection set. Selected objects are
  tinted in the 3D view, checkmarked in the tree, and the properties
  panel shows how many are selected. Dragging any selected object
  moves them **all** together; **Delete** removes them all (one Undo
  brings them all back). The panel offers Hide/Show All and Clear.
  Name, position, rotation and material editing stay single-object —
  turn Multi off to edit those. Tapping empty space clears the set;
  turning Multi off or entering Play clears it too. Multi-select is
  an editor tool only: it doesn't group objects permanently, and it
  doesn't apply in Play or the node editor.
- **Materials** — colour, emissive (toggle, colour, intensity), glass
  (transmission, matte↔clear roughness, tint, IOR), and textures:
  procedural presets (concrete, cement, brick), the road family, and your
  own designed textures under "My Textures".

## 2. How the rendering works

- **Path tracing** — every pixel's colour is estimated by tracing light
  paths through the scene with hardware ray tracing. Emissive triangles
  are sampled directly as area lights (next-event estimation), so a neon
  sign lights a room the way a real sign would — colour included.
- **Progressive accumulation** — the image refines the longer you hold
  still, up to the settle cap (768 samples in Auto; more in High detail).
  A converged, unchanged scene draws *nothing at all* — that's the big
  battery saver.
- **Temporal reprojection** — while the camera moves, converged history
  follows diffuse surfaces so motion stays readable. View-dependent
  content — glass, environment, reflections, water — *restarts* instead
  of being carried; carrying it caused visible ghosting (fixed in v3.4,
  and extended to textured surfaces in v4.1).
- **Glass** — Fresnel reflection/refraction with roughness jitter
  (matte = frosted) and Beer–Lambert absorption: the thicker the tinted
  glass, the deeper the colour.
- **Render modes** — Auto balances quality and heat; **High detail**
  raises resolution headroom, settle cap, and bounce depth; **Battery
  saver** trades resolution/samples/settle cap for runtime (glass keeps a
  quality floor since v3.4). A thermal brake trims quality only under
  genuine heat pressure, and motion frames have a dynamic-resolution
  governor that defends the 60 fps cadence without touching the settled
  image.
- **Performance (v4.9)** — a whole-app optimisation pass changed
  *when* work happens, never *what* is computed: scene rebuilds no
  longer stall the frame thread (the ray-tracing structure builds
  in the background and swaps in between frames), objects that
  move in Play update incrementally instead of being re-uploaded
  whole, weather-skin rebuilds reuse the unchanged parts of the
  scene, and panels refresh only when something they display
  actually changes. The settled picture, the weather rates, and
  every control behave exactly as in v4.8 — it should simply feel
  smoother and run cooler, most noticeably while dragging objects
  and while node-driven objects move in Play.

- **Idle freeze (v4.10)** — when no input is detected for a while,
  the scene **freezes** exactly as it stands until the next input:
  scene time stops (weather, node timers, the player — nothing
  advances), the frozen picture stays on screen, and the app draws
  nothing at all while frozen. Any input — a tap anywhere, a drag,
  a key press, the joystick, a menu action — wakes it instantly,
  and the wake is seamless: no time jump, so nothing lurches,
  teleports, or fast-forwards to "catch up". Rain hanging
  mid-air in the frozen frame is the freeze working as intended.
  The delay is adjustable in the **More (…) menu → Idle freeze**:
  **Off / 15 s / 30 s / 1 min (default) / 2 min / 5 min**, and a
  change applies immediately, even to an idle period already
  under way. A freeze never lands in the middle of work that must
  finish — a scene rebuild in flight or a final render (below)
  completes first, and the freeze engages right after.
- **Render Image — the final render (v4.10)** — in the **More (…)
  menu**, **Render Image…** produces a deliberate, offline-quality
  still of the current view (Edit orbit camera or Play
  first-person), at the view's full native resolution. Unlike the
  interactive picture it uses **every light** (no motion culling),
  **16 bounces** (the interactive budget is 5–6, so long
  mirror→glass→mirror chains survive instead of being cut), and
  **4,096 samples per pixel**, accumulated progressively in
  128-sample chunks: the sheet shows the picture refining, with a
  live sample count and percentage. **Cancel** stops at the next
  chunk boundary and **keeps the best image accumulated so far**.
  While a render runs, the live scene is paused. When it
  finishes: **Save to Photos**, **Share**, or **Save to Files**
  (a PNG in the app's `Renders` folder, visible in Files). It is
  not fast — thousands of full-quality samples at native
  resolution are minutes-class work on any scene with real
  glass; the preview tells you how far along it is, and Cancel
  always leaves you the best-so-far picture.
- **Rain in the final render (v4.10)** — live rain is a light
  overlay of streak sprites (fast, and it animates). A final
  render is a frozen still, so its rain is **real geometry**:
  the drops the live view was showing at the moment you started
  the render — same deterministic field, same positions — are
  rebuilt as small elongated glass droplets (**384 nearest drops,
  36 triangles each**) and traced with the same transmission
  machinery as scene glass, at **water IOR 1.33**. Each drop
  genuinely refracts the scene behind it and catches highlights;
  there are no flat streak cards in a finished still. The ordinary PNG
  export and save thumbnails are unchanged — they show no
  precipitation at all, as before.

## 3. Weather

- **Rain** falls around the camera (GPU precipitation) while
  Rain is on in the Weather menu. Intensity **surges and eases**
  over tens of seconds; gusting wind slants the fall. The
  streaks are thin and soft — no square silhouettes at any
  angle.
- **Ground wetness** is the whole ground effect: one flat
  value that ramps up while it rains (~75 s to fully wet,
  faster in a surge) and dries slowly when the rain stops
  (~10 min). As it rises, up-facing ground and objects get
  **darker, glossier and more reflective** — a wet sheen,
  strongest at glancing angles — and while it rains, faint
  ripple rings shimmer across the wet ground. There is no
  standing water: nothing pools, nothing climbs, nothing is
  simulated in 3D.
- Wetness is scene state and saves with the project; the
  footer shows "Ground: N% wet". **Clear ground build-up**
  (Weather menu, shown when anything is wet) resets it
  instantly. Nodes can steer it in Play (§6, Weather
  Control) — Play-time changes never leak into the saved
  scene.
- **Snow is gone (v4.13).** The snow system (accumulation,
  drifts, carving, the coating on objects, temperature and
  its phases) and the 3D puddle system were removed in
  v4.13 after device testing showed deep-snow scenes
  tearing around moved objects and both systems trying to
  climb surfaces they shouldn't. A good snow version is
  being worked out in a separate test app before it comes
  back. Projects saved with snow (or a temperature) still
  open normally — the stored snow state is simply ignored,
  and such projects open with the weather off.

## 4. Textures and the Texture Designer

- **Presets** — concrete, cement, brick: full PBR sets (colour, normal,
  roughness/height) generated procedurally on-device, with normal mapping
  and parallax depth.
- **Roads** — Highway, Road, Intersection, Crosswalk, Roundabout,
  redrawn in v4.1 at **1024×1024**: polished wheel tracks in the
  roughness channel, tar snakes, cracks, oil stains, patch rectangles
  with tar-sealed edges, border deterioration, and paint worn hardest
  where tyres run. Highway gets 3 lanes and rumble strips; Intersection
  a marked box with concrete corners; Roundabout an island, apron,
  radial joints, yield arcs, and gravel corners. They **auto-fit**: the
  texture spans the face you apply it to exactly once.
- **Texture Designer (home screen)** — paint on a 512 px canvas with
  channel previews (Colour / Roughness / Height). Brushes: colour
  (round/square shapes, spray, star stamp, palette + colour picker,
  size/opacity), **Rough** (paints roughness), **Wet** (low roughness +
  darkened colour = a permanently damp material), **Brick** (brickwork
  across all channels), Eraser, Fill, 20-step Undo. Save by name; your
  textures appear in the material picker under **My Textures** and behave
  like any preset. (Note: custom textures live in the app's library — a
  project exported to another device without that library falls back to
  plain colour for those objects.)

## 5. Play mode

Play hides *all* editor UI — the scene goes full-screen with only a
circular pause button (top-right) to return to editing. Walk with the
**analog joystick** (bottom-left) and drag-to-look, or WASD + mouse/keyboard.
In Play, the node runtime is alive: areas trigger, emitters emit, objects
move, checkpoints catch you. Leaving Play restores the exact pre-Play
scene.

**Your character** — the player is a low-poly character (~1.75 m tall),
not a marker: it stands at the play start in the editor so you can see
where you'll spawn, and in Play it walks with you — legs and arms
swinging, turning to face the way you move. Because the scene is
ray-traced you can catch it in reflections and shadows, and it appears
properly in views a node camera cuts to. It is purely visual: eye
height, checkpoints, respawn, and everything that measures distance to
the player work exactly as before. In Play the camera sits at the
character's eyes, so the head never blocks your view (it is skipped
for the view itself only — reflections, mirrors, and shadows still
show the complete character, head included), and looking down you see
your own torso, arms, and legs.

## 6. The node system

Open **Nodes** from the top bar. The graph is a canvas of colour-coded
cards — **Events (orange)** start things, **Actions (blue)** do things,
**Logic (purple)** decides. Drag a card's header to arrange it. Tap an
output pin (valid inputs light up green), then tap an input pin to
connect. Select a node to get its inspector: every parameter is a real
editor — sliders with ranges, toggles, text fields, a colour picker,
object/area menus, a texture menu. Graphs **save with the project**;
graph edits aren't covered by Undo (the sheet's footer says so). A graph
reading left-to-right always means: *when this happens → do that.*

There are deliberately **no value wires**: values travel through named
**variables** (Set Variable / Variable Maths / If Variable / Variable
Changes), which keeps graphs readable at a glance. The runtime is safe
against wiring loops — deliveries are queued and budgeted per tick, so a
bad loop can't hang the frame.

**Behaviour slots.** Continuous movement behaviours (Move Object,
Oscillate, Follow, Chase, Wander, Look At, Rotate, Scale) share one
*movement slot* per object: starting one cancels the previous, so two
behaviours never fight over the same object. Light effects (Pulse, Cycle)
share a *light slot* the same way. **Stop Behaviour** vacates a slot on
demand. Objects the graph animates are promoted to a dynamic render
layer, so the static scene is never rebuilt for animation.

### Events (start a chain)

| Node | Fires when… | Key parameters |
|---|---|---|
| Play Starts | Play mode begins | — |
| Player Enters Area | you walk into a Trigger Area | Area |
| Player Exits Area | you walk out of a Trigger Area | Area |
| Player Near Object | you come within range of an object (separate enter/exit outputs) | Object, Radius |
| Player Falls Below | your height drops below a line (re-arms above it) | Height |
| Object Tapped | the chosen object is tapped in Play | Object |
| Timer | a delay passes, once or repeating | Delay, Interval, Repeat |
| Variable Changes | a named variable's value changes | Variable name |

### Actions — objects and spawning

| Node | Does… | Key parameters |
|---|---|---|
| Show / Hide | makes an object appear / disappear | Object |
| Move Object | tweens an object to a captured destination; its **Arrived** output fires on arrival | Object, destination (captured live), duration |
| Rotate Object | rotates an object (movement slot) | Object, angle, duration |
| Scale Object | grows/shrinks an object to a uniform scale; **Arrived** on completion | Object, scale, duration |
| Spawn Object | clones an object into the world at another object's position | Source, At object, Lifetime, Offset X/Y/Z, Spawn at Player |
| Spawn Pattern | clones in a **Grid, Ring, Scatter, or Line** — one batch, reproducible | Source, pattern, Count (≤64), Spacing, Radius, Seed, Random yaw, Offset X/Y/Z, Centre on Player |

Both spawn nodes take an **Offset** (X/Y/Z, −50…50) so clones land displaced from their anchor instead of inside it, and a **Spawn at Player / Centre on Player** option that spawns at your feet in Play (offset applied on top).
| Despawn Objects | removes spawned clones | Source / all |
| Stop Behaviour | cancels an object's running behaviour slot | Object, slot |

### Actions — animation behaviours (continuous, movement slot)

| Node | Behaviour | Key parameters |
|---|---|---|
| Oscillate Object | swings an object back and forth along an axis | Axis, Amplitude, Period |
| Follow Object | steers one object after another; **Arrived** output | Target, Mode (Once / Ping-Pong / Loop) |
| Chase Player | an object hunts you; **Caught** fires in range | Speed, Stop distance, Fly |
| Wander | an object roams randomly around a home area | Radius, Speed, Pause |
| Look At | an object turns to face the player or another object | Target, Turn speed |
| Applied Gravity | gives an object real gravity in Play: it falls from where it was placed, lands on the surface below (ground or another object's top) and rests there; if the support is removed it falls again. **Landed** fires when it comes to rest after a real fall | Gravity (m/s², default 9.8 — 0 floats, negative falls up), Bounce |

### Actions — light, material, display

| Node | Does… | Key parameters |
|---|---|---|
| Set Light | switches and sets an emissive object | Object, On, Intensity, Colour |
| Pulse Light | breathes an emissive object's intensity | Min/Max intensity, Period, Stop input |
| Cycle Light Colour | rotates a light's hue continuously | Hue speed, Stop input |
| Set Object Colour | sets an object's colour directly | Object, Colour |
| Set Material | restyles an object | Object, Texture (presets/roads/yours), Glass on/off |
| Show Message | puts text on screen in Play | Text |
| Show Value | shows text with a variable substituted (`{name}`) | Text template, Variable |

### Actions — player and camera

| Node | Does… | Key parameters |
|---|---|---|
| Teleport Player | moves the Play viewpoint | Destination |
| Set Player Speed | changes walk speed | Speed |
| Set Checkpoint | remembers where you stand | — |
| Respawn Player | returns you to the checkpoint (or the Play-entry pose) | — |
| Camera Shake | shakes the view briefly | Strength, Seconds |
| Cut To View | snaps the view to look at an object, or to fixed angles | Target / Angles |

### Actions — world

| Node | Does… | Key parameters |
|---|---|---|
| Load Area / Unload Area | streams an area's objects in / out of the world | Area |
| Show Area / Hide Area | shows/hides an area's objects without unloading | Area |
| Set Weather | changes weather from a graph | Mode (Off/Rain) |
| Weather Control | the deep version: build-up rate, intensity scale, direct wet level | Rate, Intensity, Wet |
| Set Environment Light | sets ambient environment brightness (0–4) | Level |
| Set Gravity | scales gravity for physics particles (0–4) | Scale |
| Impulse Burst | a radial shove that wakes and throws resting particles | Anchor, Radius, Strength |
| Stop All Particles | halts emitters (optionally clearing live particles) | Clear |

### Actions — variables

| Node | Does… | Key parameters |
|---|---|---|
| Set Variable | writes a named variable | Name, Value |
| Variable Maths | + / − / × / ÷ / modulo / min / max; the second operand is a number or another variable (divide/modulo by zero leaves the value unchanged) | Variable, Op, Operand |
| Random Value | writes a random number (integer mode available) | Min, Max, Integer |
| Particle Emitter | starts/stops an emitter (has **Start** and **Stop** inputs) | See §7 |

### Logic (decide and shape the flow)

| Node | Behaviour | Outputs |
|---|---|---|
| If Variable | compares a variable and routes the flow | Then / Else |
| Delay | waits before passing the flow on | Out (after the wait) |
| Toggle | alternates between two outputs each time it fires | On / Off |
| Random Chance | fires its output with a set probability | Yes / No |
| Counter | counts firings; fires **Reached** at the target | Out (each count), Reached |
| Fire Once | passes the flow exactly once, until its **Reset** input fires | Out, Reset input |
| Sequence | steps through 2–4 outputs in order, optionally wrapping | Out 1…4, Reset input |

### Trigger Areas

Areas are placed from the Add menu, drawn as line boxes in the editor
(amber when selected), and objects are assigned to an area in Properties.
All areas start **loaded**. The walk-in loading pattern is two wirings:
*Play Starts → Unload Area* (the room starts absent) and *Player Enters
Area (the doorway) → Load Area*. Unloaded areas' objects leave the
rendered world entirely — that, not camera tricks, is how LumenScene
streams.

### Custom nodes — where they stand

User-defined nodes are **not** in v4.1. The honest blockers: the graph
editor has no multi-selection yet, variables share one global namespace
(so composites would collide), and mapping a composite's pins when it's
flattened into a graph is ambiguous. The recorded design is
flatten-at-start macros, and a separate node-builder app is the planned
vehicle for authoring them.

## 7. Particles — simple and physics

The **Particle Emitter** node turns any object — an imported model or a
primitive — into a particle model. Two modes:

- **Simple** — ballistic flight under gravity, flat landing. Cheap; good
  for debris, sparks-with-shape, falling props.
- **Physics** — particles collide with up-facing surfaces (including
  objects the graph has moved — the dynamic layer collides too) and come
  to rest. Flat particles **flutter and sway** as they fall. On landing, a
  particle whose footprint hangs partly off an edge **hinges down over
  that edge** (55–65°) — the leaf-drooping-off-a-block look. It is a
  rest-pose approximation, not cloth simulation: it reads correctly from
  normal distances but won't ripple or fold, and a particle balanced on a
  narrow perch with *both* edges unsupported stays flat. An **Impulse
  Burst** (§6) can wake resting particles and throw them again.

Emitter parameters (the depth standard every node follows):

| Parameter | Meaning |
|---|---|
| Model | the object used as the particle |
| Mode | Simple / Physics |
| Faces | target face count for the particle model (decimated and cached) — detail vs cost, your choice |
| Hitbox | Disc / Box / Sphere, plus a size multiplier — the shape it collides and rests with |
| Gravity scale, Drag | how it falls and how air resists it (global gravity is node-adjustable too) |
| Bounce | energy kept on impact |
| Lifetime | seconds before a particle ages out (resting ones included; default 12 s) |
| Size, Speed, Spread, Direction | initial motion |
| Rate / Burst / Duration | how emission runs |

Caps keep the frame safe: at most 400 physics (800 simple) particles per
emitter, 1,200 globally, 8 emitters' worth of particle geometry, spawned
clones capped at 64 live / 45,000 triangles, all inside a 90,000-triangle
dynamic-layer budget shared with node-moved objects. Emission *pauses* at
a cap instead of bursting past it.

**Worked example — the tree.** Import a tree, give it your textures, put a
crate or block beneath a branch. Nodes: *Play Starts → Particle Emitter*
with Model = the tree's leaf object (or any leaf-shaped import), Mode =
Physics, Hitbox = Disc, Faces ≈ 24 for cheap leaves or a few hundred for
hero leaves. In Play the tree sheds leaves that flutter down, land on the
block and the ground — and the one that lands across the block's edge
droops over it.

## 8. Culling — what it does and doesn't do

While the camera moves or you're in Play, direct-light sampling uses a
**culled light set**: only emitters within a 75° cone of where you're
looking, or within 7 m of the camera, are sampled. In a scene full of
lights this is the difference between smooth and slideshow. The culled
set only rebuilds when it actually changes, and the moment you hold
still, the **full light set returns** and the image settles with *every*
light contributing — culling costs nothing in final quality.

Geometry is deliberately **not** view-culled: reflections and shadows
legitimately need geometry behind you, and the hardware ray tracer
already skips what primary rays can't reach. Geometry leaves the world
through area unloading (§6) instead.

## 9. Projects, files, compatibility

- Projects live in the app's storage as JSON + mesh files with a
  thumbnail; **Export** packages a project for sharing.
- Models are imported from the shared Documents folder via the in-app
  list (the app has full file-sharing enabled, so you can drop models in
  from Files/Finder).
- Projects from **every earlier version (v1–v4.0) still open** — loading
  is defensive: new fields default sensibly when absent, and even a
  damaged top-level field falls back to its default instead of failing
  the whole project.
- Objects animated by nodes live in a separate dynamic layer; editing is
  unaffected, and exiting Play restores the scene exactly as it was.
- Projects saved by the snow-era versions (through v4.12) open
  normally: their stored snow, temperature and puddle state is
  ignored (v4.13 removed those systems), and the weather starts
  off; everything else — scenes, graphs, textures — is unchanged.

## 10. Known limits, honestly

- STEP files can't be imported (no CAD engine ships in the app).
- Custom nodes aren't in yet (see §6) — the builder app is the plan.
- Particle collision is with up-facing surfaces only — no wall bounces.
- Leaf drape is a hinge approximation, not cloth.
- Node graphs pass values through variables only — no value wires.
- Custom textures aren't embedded in exported projects (see §4).
- There is no snow (v4.13) — it was removed after device
  testing showed the deep-snow sheet tearing; a replacement is
  being proven in a separate test app before it returns.
- Everything in this manual describes the software as built and
  build-verified; the final word on feel — joystick, node-editor
  dragging, drape believability — belongs to the device in your hands.
