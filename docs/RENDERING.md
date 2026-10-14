# Rendering

Aether3D renders images **with its own implementation**. This guide covers
cameras, lights, the shading model and the default renderer: a deterministic,
headless CPU rasterizer written in Python and NumPy. It turns any Aether3D
scene into a correctly projected, depth-tested, lit PNG. Scenes can contain
static meshes, imported glTF meshes, meshes made with the editing engine, and
procedural geometry. No external renderer, GPU, OpenGL context or GUI is
involved, and the PNG encoder is Aether3D's own.

Since 0.6, the same scenes also render on graphics hardware with
`backend="gpu"`: real GPU rasterization through wgpu (Vulkan, Metal,
Direct3D 12), with MSAA and directional shadow maps. See [GPU.md](GPU.md).
The CPU renderer stays the default and the reference.

![The procedural Minecraft-inspired castle, rendered by Aether3D](../examples/output/minecraft_castle_render.png)

*The Stage 4 procedural castle, rendered by Aether3D's CPU renderer at
1920×1080 with 3×3 samples per pixel
([`examples/minecraft_castle_render.py`](../examples/minecraft_castle_render.py)).*

- [Quick start](#quick-start)
- [The pipeline](#the-pipeline)
- [Coordinate conventions](#coordinate-conventions)
- [Cameras](#cameras)
- [Lights and the shading model](#lights-and-the-shading-model)
- [Color](#color)
- [Rasterization](#rasterization)
- [What renders](#what-renders)
- [Python API](#python-api)
- [Command line](#command-line)
- [Persistence](#persistence)
- [How correctness is verified](#how-correctness-is-verified)
- [Performance](#performance)
- [The first renders](#the-first-renders)
- [Error and diagnostic codes](#error-and-diagnostic-codes)
- [Limitations](#limitations)
- [Toward later stages](#toward-later-stages)

## Quick start

```python
from aether3d import Scene

scene = Scene("Studio")
scene.add_plane("Ground", size=20).color = "#6f9b4f"
crate = scene.add_cube("Crate", size=2, position=(0, 0, 1))
crate.color = "#c0392b"
ball = scene.add_sphere("Ball", radius=0.8, position=(2.2, 0.5, 0.8))
ball.appearance.configure(color="#2e86c1", specular=0.5, shininess=60)

camera = scene.add_camera("Camera", fov=40)  # the first camera becomes active
camera.frame(direction=(-0.5, 1.0, -0.45))  # fit all visible geometry
scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1.0), intensity=2.0)
scene.world.configure(background="#9cc6e8", ambient_intensity=0.25)

result = scene.render(width=640, height=360)  # samples=2 (2x2 antialiasing) by default
result.save("studio.png")
print(result.stats["timings"]["total"], [o["name"] for o in result.objects])
```

From the terminal, the same steps work without any Python; see
[Command line](#command-line).

## The pipeline

```text
Scene ──extract──▶ RenderScene ──CPU backend──────────────────────────────────▶ RenderResult
        (read-only)  meshes (shared arrays, procedural = current evaluation)
                     instances (world + normal matrices, linear colors)
                     lights (world space), camera (frame + projection)
                     background, ambient

CPU backend (aether3d/rendering/cpu.py, raster.py):
  geometry   vertices → world (M) → clip space (P·V·M); normals with M⁻ᵀ;
             world-space facing; back/front-face culling; frustum rejection
  clipping   Sutherland–Hodgman against 6 planes in homogeneous clip space
  raster     perspective divide → viewport → 8-bit fixed point → exact
             integer edge functions per sample row (top-left rule)
             → visibility buffer (packed depth + triangle id, np.minimum.at)
  shade      once per covered sample: perspective-correct barycentrics,
             world position, interpolated normal, lights
  resolve    average the s×s samples of each pixel in linear light
output:      exposure → clamp → sRGB transfer function → 8 bits → PNG (own encoder)
```

| Module | Responsibility |
| --- | --- |
| `rendering/camera.py` | `Camera` parameters, projection matrices, camera frame from a world matrix |
| `rendering/light.py` | `Light` parameters |
| `rendering/settings.py` | `Appearance`, `World`, `RenderSettings`, `RenderLimits` |
| `rendering/color.py` | sRGB ↔ linear transfer functions, color parsing |
| `rendering/scene_data.py` | `RenderScene` extraction (backend-independent) |
| `rendering/raster.py` | clipping, fixed-point rasterization, visibility buffer |
| `rendering/cpu.py` | the CPU backend: geometry, raster, shading, resolve |
| `rendering/renderer.py` | `render()` and `RenderResult` |
| `rendering/framing.py` | exact camera framing |
| `rendering/png.py` | PNG encoder |
| `rendering/script.py` | declarative scene-setup scripts |

Rendering never modifies the scene. Extraction reads objects, transforms and
the *current evaluated* procedural meshes (through the procedural cache), and
everything after extraction works on immutable `RenderScene` data.

## Coordinate conventions

- **World:** right-handed, **+Z up**, 1 unit = 1 meter. An asset's front
  faces **−Y**.
- **Cameras (and lights) look along their local −Y axis, with local +Z up.**
  This is the same "front faces −Y" convention as every asset, so
  `obj.look_at(target)` aims cameras, directional lights and models alike.
  Seen through the camera, image right is the camera's local **−X**. An
  unrotated camera at `(0, 10, 0)` looks along −Y, toward the origin.
- **Camera frame.** The renderer derives the frame from the camera's world
  matrix:
  - forward is the normalized world direction of local −Y;
  - up is the world direction of local +Z, made orthogonal to forward
    (Gram–Schmidt);
  - right = forward × up.

  Scale, shear or mirroring inherited from parents therefore never distort or
  mirror the image; they are reported as `CAMERA_SCALE_IGNORED`.
- **View space:** x right, y up, the camera looking down −z (the usual
  right-handed convention).
- **Clip space and NDC:** x and y in [−1, 1] (y up); depth z in **[0, 1]**,
  with the near plane at 0 and the far plane at 1.
- **Images:** row 0 is the top. Pixel `(x, y)` covers `[x, x+1] × [y, y+1]`,
  and its center is at `(x + 0.5, y + 0.5)`. NDC (−1, 1) is the top-left
  corner of the image.
- **Other tools:**
  - A Blender camera (looking along local −Z, +Y up) with the same view has
    the rotation matrix whose columns are (right, up, −forward).
  - glTF cameras look along −Z in glTF's Y-up space. Under Aether3D's exact
    axis conversion that is world +Y, so a glTF camera node corresponds to an
    Aether3D camera turned 180° about Z.

## Cameras

```python
cam = scene.add_camera(
    "Main",
    projection="perspective",
    fov=40,
    fov_axis="vertical",
    near=0.1,
    far=1000,
    position=(8, -12, 6),
    look_at=(0, 0, 1),
)
cam.camera.fov = 35  # validated immediately
cam.camera.configure(near=500, far=5000)  # several values at once, all or nothing
```

| Parameter | Meaning | Valid values (default) |
| --- | --- | --- |
| `projection` | `perspective` or `orthographic` | (`perspective`) |
| `fov` | Field of view in degrees, measured along `fov_axis` | 0.01–179 (40) |
| `fov_axis` | `vertical`, `horizontal`, or `auto` (the larger image dimension) | (`vertical`) |
| `ortho_scale` | Orthographic view size in world units along `fov_axis` | > 0 (10) |
| `near`, `far` | Clipping distances along the view direction | perspective: 0 < near < far, far/near ≤ 10⁹; orthographic: 0 ≤ near < far (0.1, 1000) |
| `aspect_ratio` | The frame shape (width / height) the camera is meant for, or `None` | 10⁻³–10³ (None) |

**Aspect ratio.**
- Pixels are always square, so the projection uses the image's aspect
  (width / height).
- The field of view stays fixed along `fov_axis`, and the other extent
  follows the image.
- A camera's `aspect_ratio` gives the missing dimension when a render only
  specifies a width or a height, and framing uses it.
- If the image's shape differs from `aspect_ratio`, the render reports
  `ASPECT_MISMATCH` (info).

**Projection matrices.** With `tx`, `ty` the tangents of the half angles
(perspective), `hx`, `hy` the half extents (orthographic), and `n`, `f` the
clipping distances:

```text
perspective                                  orthographic
[1/tx  0     0          0        ]           [1/hx  0     0          0         ]
[0     1/ty  0          0        ]           [0     1/hy  0          0         ]
[0     0     f/(n−f)    n·f/(n−f)]           [0     0     −1/(f−n)   −n/(f−n)  ]
[0     0     −1         0        ]           [0     0     0          1         ]
```

Both map view depth −n to NDC z 0 and −f to 1. For perspective, the depth at
distance d along the view direction is `z = f (d − n) / (d (f − n))`, and
`x_ndc` is inversely proportional to d (foreshortening). Orthographic
projection has no foreshortening.

**Aiming.**
- `obj.look_at(target, up=(0, 0, 1))` turns any object so its front (local
  −Y) faces a point or another object, keeping local +Z as close to `up` as
  possible. `obj.look_along(direction)` takes a direction instead.
- Both are exact under rotated and non-uniformly scaled parents and under
  the object's own negative scale. The local rotation is solved through the
  inverse parent transform, so the resulting *world* forward and up axes are
  exactly the requested ones.
- When the direction is parallel to `up`, the image's up becomes the world
  axis least aligned with the direction, preferring Y, then X, then Z. Its
  sign is chosen so that looking straight down puts +Y at the top, like a top
  view, and looking straight up puts −Y at the top.
- Aiming an object at its own position raises `ValidationError`.
- `obj.set_world_position(p)` places an object in world space through its
  parents.

**Active camera.**
- `scene.add_camera()` makes the new camera active when there is no active
  camera yet, or when `active=True`.
- `scene.active_camera = cam` (an object, id or name) or `None` changes it.
- Removing the active camera clears it.
- `scene.render()` uses `camera=` if given, else the active camera. Without
  either it raises `NO_CAMERA`, with a hint listing the scene's cameras.
- `render(frame=True)` renders without any camera: it uses a temporary one,
  as described under framing below.

**Framing.** `cam.frame(targets=None, direction=None, margin=0.05,
fit_clipping=True, aspect=None)` moves a camera so the targets (default:
every visible mesh object) exactly fill the frame. For the camera's view
direction and the targets' world-space **vertices** (not a bounding sphere),
it computes the closest position along that direction at which every vertex
projects inside the frame minus the margin. Each image axis is centered
independently. The closed-form solution is in
[`rendering/framing.py`](../src/aether3d/rendering/framing.py).

- Orthographic cameras get the `ortho_scale` that fits.
- With `fit_clipping`, near and far are set to enclose the geometry.
- `direction` turns the camera first.
- Framing is atomic: if it fails (for example `NOTHING_TO_FRAME`), the camera
  is unchanged.
- `render(frame=True)` frames for that render only. If the scene has no
  camera, a temporary perspective camera looks from the front (−Y), right
  (+X) and above, along `(−0.6, 1.0, −0.55)`.

## Lights and the shading model

This is a deliberately simple, **non-physically-based** model. Intensities are
unitless multipliers, not photometric units. The equations are exactly what
the renderer computes, in linear light:

```text
radiance = base ⊙ ambient
         + Σ_lights base ⊙ Lᵢ · aᵢ · max(0, n·lᵢ)
         + Σ_lights Lᵢ · aᵢ · specular · max(0, n·hᵢ)^shininess   (only where n·lᵢ > 0)
```

| Term | Definition |
| --- | --- |
| `base` | The object's color in linear light |
| `ambient` | World `ambient_color` (linear) × `ambient_intensity` |
| `Lᵢ` | Light color (linear) × `intensity` |
| `lᵢ` | Unit vector toward the light |
| `hᵢ` | Unit half vector between `lᵢ` and the direction toward the camera (Blinn–Phong) |
| `n` | The shading normal |
| `aᵢ` | Attenuation |

| Light | Placement | Attenuation `aᵢ` |
| --- | --- | --- |
| `directional` | Travels along the light object's front (local −Y); its position does not matter | 1 |
| `point` | At the object's world position | `1/d²` (`falloff="inverse_square"`, default) or 1 (`falloff="none"`), with d clamped to ≥ 10⁻⁴; with a `range`, also multiplied by `clamp(1 − (d/range)⁴, 0, 1)²` (the smooth window recommended by glTF's KHR_lights_punctual) |
| ambient | `scene.world`: uniform, no direction | — |

What the numbers mean:
- A white directional light of intensity 1 shining straight onto a surface
  makes it appear exactly its base color, before exposure.
- A point light of intensity *I* does the same at distance √*I*.
- Lights add up linearly.
- **There are no shadows**: lights are not occluded. There are no
  reflections and no indirect light.
- Hidden light objects (`visible = False`) do not shine.
- If a scene has no light objects and `default_lights` is on (the default), a
  built-in white key light of intensity 1 shines from the camera's upper left
  and behind it (`DEFAULT_LIGHTING`, info). Turn it off to render with
  ambient light only (`NO_LIGHTS`, warning).

**Appearance.** Every mesh object has `obj.appearance`:

| Field | Meaning | Valid values (default) |
| --- | --- | --- |
| `color` | Base color | display sRGB (light gray `(0.8, 0.8, 0.8)`) |
| `specular` | Highlight strength | 0–1 (0, matte) |
| `shininess` | Highlight exponent | 1–10000 (32) |

`obj.color` is a shortcut for `obj.appearance.color`. The highlight is a
simple, non-energy-conserving Blinn–Phong term, in the light's color. This is
not a material system: there are no textures, PBR parameters or shaders yet
(see [Toward later stages](#toward-later-stages)).

**Normals.**
- Meshes with normals are shaded with the vertex normals, interpolated
  perspective-correctly and renormalized per sample. Aether3D meshes split
  vertices at sharp edges, so flat-shaded and smooth-shaded meshes both look
  as authored.
- Normals are transformed by the **inverse transpose** of the world matrix's
  linear part. This handles non-uniform scale, and it keeps outward normals
  outward under mirroring; the plain cofactor matrix would flip them.
- Meshes without normals (allowed by glTF) are shaded flat with geometric
  face normals.
- Where interpolated normals cancel out, the geometric normal is used.
- A triangle's outward side is its counter-clockwise side, reversed for
  mirrored instances (negative world determinant).
- **Two-sided lighting:** triangles seen from behind are lit with their
  normals turned toward the camera, so open surfaces and interiors render
  sensibly.
- `cull="back"` or `"front"` removes back-facing or front-facing triangles
  instead.

## Color

- **Inputs are display colors.** Object, light, background and ambient colors
  are sRGB-encoded, as in color pickers and hex codes. They are given as
  `"#rrggbb"`, `"#rgb"`, or three numbers in [0, 1].
- **Lighting happens in linear light.** Inputs are converted with the exact
  IEC 61966-2-1 sRGB transfer function. Light, antialiasing and exposure
  then add up linearly.
- **Output:**
  1. Linear values are multiplied by `2^exposure` (EV stops).
  2. They are clamped to [0, 1] (no tone mapping).
  3. They are encoded with the sRGB transfer function.
  4. They are quantized to 8 bits, rounding to nearest with ties to even.

  The PNG declares sRGB with an `sRGB` chunk. `result.linear` keeps the
  linear values after exposure (above 1 where they will clip). A background
  color comes out exactly as given (for example `#336699` → `(51, 102, 153)`).
- Half of a pixel covered by white and half by black averages to 0.5 in
  linear light. That displays as sRGB 188, not 128: antialiasing is
  physically consistent.

## Rasterization

- **Clipping.** Triangles are clipped against all six frustum planes, near
  plane first, in homogeneous clip space, where linear interpolation is exact
  for perspective. A clipped triangle becomes one or two triangles with the
  same winding. Triangles entirely outside one plane are rejected first.
  Geometry behind the camera therefore never wraps around, and extreme
  coordinates (10⁶–10¹² units) are safe.
- **Fixed point.**
  - After the perspective divide, vertices are snapped to 1/256 of a sample
    (8 sub-pixel bits).
  - Edge functions are evaluated **exactly in 64-bit integers** at sample
    centers.
  - The covered span of every sample row is computed directly; no candidate
    pixels are tested and rejected.
- **Fill rule.** A sample exactly on an edge belongs to the triangle for
  which that edge is a top or left edge. Triangles sharing an edge therefore
  cover each sample on it **exactly once**: no cracks, no double coverage.
  This is tested with random meshes and with fans around vertices placed
  exactly on sample centers and pixel corners. Zero-area triangles after
  snapping are skipped (counted as `degenerate`).
- **Visibility buffer.**
  - Depth (NDC z, affine in screen space) is quantized to `63 − b` bits and
    packed with the triangle's index in its `b` low bits.
  - `np.minimum.at` keeps the nearest fragment of every sample.
  - Occlusion therefore does not depend on draw order.
  - Exact depth ties go to the lowest triangle index (scene order), so
    results are deterministic.
- **Shading.**
  - Each covered sample is shaded once.
  - Perspective-correct barycentrics come from the **original** triangle in
    homogeneous 2D coordinates: `b ∝ adj(M)·(x, y, 1)` with
    `M = [c₀ c₁ c₂]` in clip (x, y, w). This works even for triangles that
    were clipped at the near plane.
  - The view depth is `w = det(M) / Σb′`, and the world position is
    reconstructed on the sample's view ray.
- **Antialiasing.** `samples = s` renders `s × s` ordered-grid supersamples
  per pixel (1–4; default 2) and averages them in linear light (box filter).
  `result.depth` and `result.object_index` describe one representative sample
  per pixel: the center sample for odd `s`.
- **Precision.**
  - Geometry and shading arithmetic is float64. Mesh positions are stored as
    float32, like all Aether3D meshes.
  - Per-vertex and per-sample arithmetic is written out explicitly rather
    than delegated to BLAS, so identical inputs give identical bits across
    platforms.
  - The remaining cross-platform differences are possible last-bit
    differences in C-library `tan`/`pow`. They are rare and can only flip
    samples lying within 1/512 pixel of an edge.

## What renders

| Content | Rendered as |
| --- | --- |
| Static meshes, imported glTF meshes, meshes committed by editing sessions | Their mesh |
| Procedural objects | **The current evaluation**, through the procedural cache. Nothing is baked, and the modifier stack, parameters and base meshes are untouched. A geometry that fails to evaluate raises `GEOMETRY_EVALUATION_FAILED` with the object's location; hide the object (`visible = False`) to render without it. |
| Objects sharing a mesh | Separate instances with their own transform, color and visibility. The mesh is extracted once. |
| Hierarchies | Each object uses its full world matrix, including non-uniform, sheared and mirrored transforms. |
| Hidden objects (`visible = False`), cameras, lights, empties | Nothing. Visibility is not inherited by children. |
| An empty scene | The background, with `NOTHING_TO_RENDER` (warning). |

## Python API

| Call | Description |
| --- | --- |
| `scene.add_camera(name=None, *, projection, fov, fov_axis, ortho_scale, near, far, aspect_ratio, parent, position, rotation, look_at, up, active)` | Add a camera object |
| `scene.add_light(type="directional", name=None, *, color, intensity, range, falloff, parent, position, rotation, direction, look_at, up)` | Add a light object; also `add_directional_light(name, direction=...)` and `add_point_light(name, position=...)` |
| `scene.cameras`, `scene.lights`, `scene.active_camera` | Lookup and active camera (settable) |
| `scene.world` | `World`: `background`, `ambient_color`, `ambient_intensity` |
| `scene.render_settings` | `RenderSettings`: `width`, `height`, `samples`, `exposure`, `cull`, `default_lights`; saved with the project |
| `scene.render(*, camera, width, height, samples, exposure, cull, default_lights, frame, margin, limits, backend, adapter, msaa, supersample, shadows)` | Render; arguments override the saved settings for this render. Giving only `width` or `height` keeps the camera's `aspect_ratio` (or the saved shape). `backend="gpu"`/`"auto"` and the GPU options are described in [GPU.md](GPU.md). Returns a `RenderResult`. |
| `obj.camera`, `obj.light`, `obj.kind` | Components (`Camera`/`Light` or `None`); `"mesh"`, `"procedural"`, `"camera"`, `"light"` or `"empty"`. An object holds at most one of mesh, camera and light. |
| `obj.appearance`, `obj.color`, `obj.visible` | Appearance and render visibility |
| `obj.look_at(target, *, up)`, `obj.look_along(direction, *, up)`, `obj.world_forward`, `obj.set_world_position(p)` | Aiming and placement |
| `cam.frame(targets=None, *, direction, up, margin, fit_clipping, aspect)` | Exact framing; returns `Framing` |
| `aether3d.render(scene, ...)` | Same as `scene.render(...)` |
| `aether3d.rendering.extract_render_scene(scene, camera)` | The backend-independent `RenderScene` |
| `run_scene_script(scene, script)` | Declarative setup scripts (see below) |

**`RenderResult`:**

| Field or method | Contents |
| --- | --- |
| `image` | `(h, w, 3) uint8` sRGB |
| `linear` | `float32`, linear light after exposure |
| `depth` | View depth; `inf` for background |
| `coverage` | Fraction of each pixel's samples covered by geometry |
| `object_index` | Index into `objects`; −1 for background |
| `objects` | Id, name, triangles and `visible_samples` of every rendered object |
| `camera` | Parameters, frame, and the horizontal and vertical field of view used |
| `settings` | The settings used |
| `stats` | Counts and timings |
| `diagnostics` | `Diagnostic` list |
| `output` | Set by `save()` |
| `save(path)` | Atomic PNG write; `.png` only |
| `png_bytes()`, `pixel(x, y)`, `to_dict()` | Encoded PNG, one pixel, JSON-compatible report |

`stats` holds:
- counts: `instances`, `meshes`, `lights`, `triangles`, `culled_backface`,
  `culled_frustum`, `clipped`, `rasterized`, `degenerate`, `fragments`,
  `samples`, `samples_covered`, `pixels_covered`;
- `timings` in seconds: `evaluate`, `extract`, `geometry`, `raster`,
  `shade`, `output`, `total`, plus `encode` and `write` after `save`.

**`RenderLimits`** are checked before any large allocation (`RESOURCE_LIMIT`):
- `max_dimension`: 16384 pixels per side;
- `max_samples`: 40,000,000 samples (width × height × samples²);
- `max_triangles`: 20,000,000 triangles, counting instances.

Errors raise `RenderError` (code, message, location, hint, details), and
nothing is written or changed.

## Command line

A complete terminal-only workflow:

```console
$ aether3d camera add castle.aether --name Main --frame --frame-direction -0.55 1 -0.6 --fov 30
$ aether3d light add castle.aether --type directional --direction 0.55 0.6 -0.8 --intensity 2.4 --color "#fff1d6"
$ aether3d light add castle.aether --type point --name Torch --position -2.6 -14 3 --intensity 6 --range 10
$ aether3d object set castle.aether "Curtain Wall" --color "#8f8f88"
$ aether3d scene set castle.aether --background "#8ec5f2" --ambient-intensity 0.32 --width 1920 --height 1080 --samples 3
$ aether3d render castle.aether -o castle.png --json
$ aether3d render model.glb -o preview.png --frame          # any project or glTF file, no setup
$ aether3d camera list castle.aether ; aether3d light list castle.aether
```

| Command | Purpose |
| --- | --- |
| `render PATH [-o OUT.png] [--width W] [--height H] [--camera REF] [--samples N] [--exposure EV] [--cull MODE] [--background C] [--no-default-lights] [--frame] [--margin M] [--force]` | Render a `.aether` project, or a `.glb`/`.gltf` file with `--frame`. The output defaults to `PATH.png`. Existing files need `--force`. The project is never modified. |
| `camera add PATH [--name] [--projection] [--fov] [--fov-axis] [--ortho-scale] [--near] [--far] [--aspect-ratio] [--parent] [--position X Y Z] [--rotation-euler X Y Z] [--look-at X Y Z \| --look-at-object REF] [--up X Y Z] [--active \| --inactive] [--frame] [--frame-direction X Y Z] [--margin]` | Add a camera |
| `camera set PATH CAMERA [...] [--activate]`, `camera frame PATH [CAMERA] [--object REF]... [--direction X Y Z] [--margin] [--no-fit-clipping]`, `camera activate PATH CAMERA\|none`, `camera list PATH` | Change, frame, activate, list cameras |
| `light add PATH [--type directional\|point] [--name] [--color] [--intensity] [--range R\|none] [--falloff] [--direction X Y Z] [--position X Y Z] [--look-at ...] [--parent]`, `light set PATH LIGHT [...]`, `light list PATH` | Lights (list also shows ambient light) |
| `object set PATH OBJECT [--position] [--rotation-euler] [--scale] [--look-at ...] [--color] [--specular] [--shininess] [--visible true\|false]`, `object remove PATH OBJECT` | Any object |
| `scene set PATH [--background] [--ambient-color] [--ambient-intensity] [--width] [--height] [--samples] [--exposure] [--cull] [--default-lights true\|false] [--active-camera REF\|none]` | World and saved render settings |
| `scene run PATH SCRIPT.json\|-` | Apply a declarative scene script |

How the commands behave:
- Every changing command accepts `--output PATH` and `--dry-run`, and writes
  the project only if everything succeeded.
- Every command accepts `--json`. The JSON `command` field names the
  subcommand (for example `"camera add"`).
- Exit codes: `0` success, `1` failure (a `RenderError` or `ScriptError` with
  `code` and `location`), `2` usage error.
- Images and projects are written atomically.
- `inspect --json` now reports each object's `kind`, `camera`, `light`,
  `appearance` and `visible`.

**Scene scripts** (`aether3d.scene-script` v1) are data, never code. They are
the declarative form of every setup operation, and the CLI commands above are
built on them:

```json
{
  "format": "aether3d.scene-script",
  "version": 1,
  "operations": [
    {"op": "add_camera", "name": "Main", "fov": 35, "frame": true, "frame_direction": [-0.5, 1, -0.45]},
    {"op": "add_light", "type": "directional", "name": "Sun", "direction": [-0.4, 0.6, -1], "intensity": 2},
    {"op": "add_light", "type": "point", "name": "Bulb", "position": [2, -2, 3], "intensity": 6, "range": 12},
    {"op": "set_object", "object": "Crate", "color": "#c0392b", "specular": 0.3},
    {"op": "set_world", "background": "#9cc6e8", "ambient_intensity": 0.25},
    {"op": "set_render_settings", "width": 1280, "height": 720, "samples": 2}
  ]
}
```

| Operation | Keys |
| --- | --- |
| `add_camera` | Camera parameters, `name`, `parent`, `position`, `rotation_euler`, `look_at` (point or object), `up`, `active`, and `frame` with `frame_targets`, `frame_direction`, `frame_margin` |
| `add_light` | `type`, light parameters, `name`, `parent`, `position`, `rotation_euler`, `direction`, `look_at`, `up` |
| `set_camera`, `set_light` | `camera`/`light` plus parameters |
| `set_object` | `object`, `position`, `rotation_euler`, `scale`, `look_at` or `look_along`, `up`, `color`, `specular`, `shininess`, `visible` |
| `frame_camera` | `camera` (default: active), `targets`, `direction`, `up`, `margin`, `fit_clipping` |
| `set_active_camera` | `camera` or `null` |
| `set_world` | `background`, `ambient_color`, `ambient_intensity` |
| `set_render_settings` | `width`, `height`, `samples`, `exposure`, `cull`, `default_lights` |
| `remove` | `object` (and its children) |
| `info` | Read-only summary |

A script is atomic. If any operation fails:
- created objects are removed;
- removed objects are restored in place;
- every changed value is restored *in place*, so held handles stay valid;
- the error names the operation and the element, for example
  `operations[1]: objects[Main].camera.fov`.

These operations are designed to be exposed unchanged to AI agents through
the planned MCP server (Stage 11).

## Persistence

Projects are saved as `.aether` **format version 3**; see
[FILE_FORMAT.md](FILE_FORMAT.md).

- `scene.active_camera`, `scene.world` and `scene.render` (the render
  settings) are saved.
- Every object has `camera`, `light`, `appearance` (`null` = default) and
  `visible`.
- Versions 1 and 2 load unchanged: they have no cameras, the default world
  and the default settings. Saving writes version 3.
- Rendering a reloaded procedural project still evaluates its definitions.
  Nothing is baked, and parameters stay editable.

## How correctness is verified

1. **Analytic tests** compare against exact expected values at every pixel:
   - projection of frustum corners, depth formula, foreshortening ratios;
   - Lambert shading at several light angles;
   - point lights with inverse-square, range-window and no falloff;
   - the specular peak at the half-vector point;
   - exposure, exact background colors, and a half-covered pixel resolving
     to 0.5 linear;
   - watertight coverage and the exact top-left fill rule;
   - culling, two-sided lighting and mirror-image symmetry.
2. **An independent reference renderer**
   ([`tests/render_reference.py`](../tests/render_reference.py)) is a
   brute-force Möller–Trumbore ray caster. It shares no projection, clipping,
   rasterization or shading code with the renderer. It is compared with the
   renderer on scenes with:
   - non-uniform scale and mirrored shared meshes;
   - rotated, scaled parents of objects and of the camera;
   - geometry crossing the near plane, a camera inside a closed room;
   - perspective and orthographic cameras;
   - glTF-imported and edited meshes.

   Documented tolerances:
   - The visible object agrees at every pixel center. The only exceptions
     are samples within the 1/256-pixel snapping band of a silhouette, which
     must lie on an object boundary.
   - Depth agrees within 10⁻⁵ relative.
   - Linear radiance agrees within 10⁻⁴: within float32 precision in
     interiors, and up to about 10⁻⁵ where barycentrics are clamped at
     edges. Samples exactly on a crease between two faces may legitimately
     be shaded by either face.
3. **Regression images.**
   [`tests/data/renders`](../tests/data/renders) holds small renders,
   regenerated deliberately with `python tests/render_goldens.py --update`.
   They are compared with a cross-platform tolerance: at most 0.5% of pixels
   may differ by more than 2 codes, and the mean difference must be below
   0.25 codes.
4. **External verification (optional, not part of the image pipeline).**
   `python examples/minecraft_castle_render.py --blender` exports the scene,
   renders the same camera in Blender's Workbench engine, and compares
   silhouettes. On Blender 5.2, the 1920×1080 castle agreed on all but
   **2 of 2,073,600 pixels** (IoU 1.00000). The published PNGs are always
   Aether3D's own renders.

CI runs the renderer and checks real PNGs on Linux, macOS and Windows, and
uploads the renders as workflow artifacts.

## Performance

`python benchmarks/bench_render.py` gives the following timings. All are the
best of 3 runs at 2×2 antialiasing (4 samples per pixel), on an Intel Core
i9-12900H, single-threaded, Linux, Python 3.14, NumPy 2.5. Peak memory is
measured with `tracemalloc`.

| Scene | Triangles | 640×360 | 1280×720 | 1920×1080 |
| --- | --- | --- | --- | --- |
| Procedural castle (as in the published render) | 6,804 | 0.16 s / 92 MiB | 0.58 s / 141 MiB | 1.24 s / 221 MiB |
| UV sphere + ground | 65,026 | 0.11 s / 109 MiB | 0.36 s / 164 MiB | 0.81 s / 249 MiB |
| UV sphere + ground | 1,046,530 | 0.46 s / 248 MiB | 0.73 s / 305 MiB | 1.19 s / 395 MiB |
| 400 instanced cubes (one shared mesh) | 4,800 | 0.18 s / 84 MiB | 0.66 s / 135 MiB | 1.45 s / 214 MiB |

The castle at 1920×1080 takes 1.24 s in total:

| Stage | Time | Notes |
| --- | --- | --- |
| Evaluation and extraction | under 0.01 s | Procedural results are cached |
| Geometry | 0.002 s | |
| Rasterization | 0.31 s | 10.8 million fragments |
| Shading | 0.64 s | |
| Resolve and sRGB | 0.08 s | |
| PNG encoding | 0.18 s | Measured separately |

The published 3×3-sample castle render takes about 2.2 s.

- Rasterization costs about 30 ns per fragment, including overdraw.
- Shading costs about 0.15 µs per covered sample.
- Geometry costs about 0.17 µs per triangle. Frustum tests and screen setup
  are done once per vertex, using Cohen–Sutherland outcodes.
- Memory is roughly 8 bytes per sample (the visibility buffer) plus about
  90 bytes per pixel, with work done in bounded batches.
- The renderer is single-threaded NumPy, with no compiled extensions.

## The first renders

| Image | Script |
| --- | --- |
| ![castle](../examples/output/minecraft_castle_render.png) | [`examples/minecraft_castle_render.py`](../examples/minecraft_castle_render.py): Stage 4 procedural castle (walls, battlements, arched gate, towers, keep, watchtower, red roof, trees, path, torches), 1920×1080 with 3×3 samples per pixel |
| ![first](../examples/output/aether3d_first_render.png) | [`examples/first_render.py`](../examples/first_render.py): static and procedural primitives, sun, fill light and point light, 1280×720 |
| ![hero](../examples/output/blocky_hero_render.png) | [`examples/hero_render.py`](../examples/hero_render.py): the Stage 3 hero and sword, modeled with the editing engine, 1280×720 |

To reproduce them:

```console
$ python examples/minecraft_castle_render.py      # writes examples/output/minecraft_castle_render.png (+ .aether)
$ aether3d render examples/output/minecraft_castle_render.aether -o castle.png   # same image from the CLI
$ python examples/first_render.py ; python examples/hero_render.py
```

Each script checks its own result:
- every part is visible;
- colors are distinct;
- the saved project re-renders identically;
- for the castle, `aether3d render` reproduces the API's pixels exactly.

## Error and diagnostic codes

Errors (`RenderError.code`):

| Code | Meaning |
| --- | --- |
| `NO_CAMERA`, `NOT_A_CAMERA`, `OBJECT_NOT_FOUND` | No usable camera, or an unknown reference |
| `INVALID_CAMERA`, `INVALID_CAMERA_TRANSFORM` | A bad camera parameter, or a degenerate camera world transform |
| `INVALID_LIGHT`, `INVALID_APPEARANCE`, `INVALID_WORLD`, `INVALID_COLOR`, `INVALID_SETTINGS`, `INVALID_FRAMING` | Bad values |
| `RESOURCE_LIMIT` | Image size, samples or triangles beyond `RenderLimits` |
| `GEOMETRY_EVALUATION_FAILED` | Procedural geometry could not be evaluated |
| `NONFINITE_TRANSFORM` | A world matrix overflowed |
| `NOTHING_TO_FRAME` | Framing with no geometry |
| `INVALID_OUTPUT`, `OUTPUT_ERROR`, `INVALID_IMAGE` | Bad output path, write failure, bad PNG input |
| `RENDER_FAILED` | Non-finite pixel values; this should not happen |

Diagnostics (in `result.diagnostics`):

| Code | Severity | Meaning |
| --- | --- | --- |
| `DEFAULT_LIGHTING` | info | No light objects, so the built-in key light was used |
| `NO_LIGHTS` | warning | No light objects, and default lights are off |
| `NOTHING_TO_RENDER` | warning | No visible mesh objects |
| `CAMERA_FRAMED` | info | `frame=True` placed the camera for this render |
| `CAMERA_SCALE_IGNORED` | info | The camera's world transform has scale, shear or mirroring |
| `ASPECT_MISMATCH` | info | The image shape differs from the camera's `aspect_ratio` |
| `ZERO_WORLD_SCALE` | warning | An object collapses (singular world matrix) and was skipped |
| `LIGHT_DIRECTION_UNDEFINED` | warning | A directional light with zero forward scale was skipped |
| `NONFINITE_GEOMETRY` | warning | Triangles with non-finite coordinates were skipped |

## Limitations

- **Lighting:** the CPU renderer draws no shadows (the GPU backend renders
  directional-light shadow maps, see [GPU.md](GPU.md#shadows)). There are no
  reflections or indirect light, no transparency, and no textures or
  materials beyond one solid color and a simple highlight per object.
- **Antialiasing:** ordered-grid supersampling only. Specular highlights on
  coarse tessellations can still alias.
- **Speed:** single-threaded CPU rendering. Previews take seconds at 1080p,
  not real time. The GPU backend ([GPU.md](GPU.md)) renders the same scenes in
  milliseconds on graphics hardware.
- **Limits:** perspective cameras need `far / near ≤ 10⁹`, which keeps the
  packed depth precise. There is no tone mapping beyond exposure and
  clamping.
- **glTF export:** cameras and lights are written as plain nodes (their
  transforms are kept, and the files pass the Khronos validator). glTF
  cameras and `KHR_lights_punctual` are not written yet, and imported glTF
  cameras and lights are still skipped (reported as before).
- **Colors stay in Aether3D:** exported glTF files do not carry object colors
  (no materials yet).

## Toward later stages

- **Stage 6 (GPU rendering) is implemented.** The GPU backend consumes the
  same `RenderScene` and returns the same `RenderResult`, so `scene.render`,
  the CLI and the CPU tests stayed unchanged. See [GPU.md](GPU.md).
- **Stage 7 (materials and textures).** `Appearance` is the single place
  where surface properties enter shading, and shading is one function over
  per-sample data. A material system replaces `Appearance` and extends the
  per-triangle table (UVs are already carried by meshes) without touching
  projection, clipping or visibility.
- **Stage 8 (quality).** The visibility buffer and deferred shading are the
  natural base for CPU shadows (shadow maps from the same rasterizer, fitted
  by the backend-neutral `rendering/shadows.py` the GPU already uses), better
  antialiasing and post-processing.
- **Later: path tracing.** The camera frame and ray generation documented
  here, already implemented independently in the test reference ray caster,
  together with the `RenderScene` data, are what a path tracer needs.
