# Aether3D

[![Tests](https://github.com/Murisixz/Aether3D/actions/workflows/tests.yml/badge.svg)](https://github.com/Murisixz/Aether3D/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Aether3D** is an open-source, AI-native, code-first 3D engine. Every operation
is available through a small, predictable Python API and a JSON-speaking CLI,
so it can be driven reliably by people and AI coding agents alike, with no
graphical interface required.

> **Status: v0.6.0 (the GPU rendering engine).** Aether3D provides:
>
> - a scene graph with transforms and hierarchy;
> - the native `.aether` project format;
> - standards-compliant **glTF 2.0 / GLB import and export**;
> - structured **geometry inspection**;
> - a programmable **polygon modeling engine** (selection, transforms,
>   extrude, inset, bevel, loop cuts, subdivision, merging, splitting,
>   validation, undo);
> - a **non-destructive procedural geometry engine**: parametric primitives,
>   modifier stacks (transform, mirror, array, solidify, Catmull–Clark
>   subdivision, bevel, solid booleans), typed parameters with safe
>   expressions and rules, dependency tracking, caching, and undo;
> - **cameras, lights and Aether3D's own CPU renderer**: perspective and
>   orthographic cameras, directional, point and ambient light, solid colors,
>   clipping, depth testing, antialiasing, and PNG output, all without
>   Blender, a GPU or a GUI;
> - a **GPU rendering backend** (optional): real, headless GPU rasterization
>   through wgpu on Vulkan, Metal or Direct3D 12, with MSAA, directional
>   shadow maps, adapter discovery and selection, and automatic CPU fallback.
>
> All of it is scriptable from Python and from declarative JSON on the CLI.
> Materials and textures, animation, physics and other systems are
> intentionally **not** part of this stage. See the [changelog](CHANGELOG.md).

## Quick start

```python
from aether3d import Scene

scene = Scene("First Scene")

cube = scene.add_cube(name="Cube", size=2)
cube.position = (0, 0, 1)

sphere = scene.add_sphere(name="Sphere", radius=1)
sphere.position = (3, 0, 1)

scene.save("example.aether")

loaded_scene = Scene.load("example.aether")
```

Hierarchy and world-space queries:

```python
pivot = scene.add_object(name="Pivot", position=(3, 0, 1))
pivot.rotation_euler = (0, 0, 90)  # degrees, XYZ order
moon = scene.add_sphere(name="Moon", radius=0.25, parent=pivot, position=(2, 0, 0))

moon.world_position  # ≈ (3, 2, 1)
moon.set_parent(None, keep_world_transform=True)
scene.get(moon.id) is scene.get_by_name("Moon")  # True
```

A complete, runnable example lives in [`examples/first_scene.py`](examples/first_scene.py).

### glTF import/export and geometry inspection

```python
from aether3d import Scene, import_gltf

result = scene.export("world.glb")  # glTF 2.0 binary (or "world.gltf")
imported = Scene.import_file("world.glb")  # object ids survive the round trip

report = imported.inspect_geometry()  # bounds, areas, degeneracies, topology...
report.status  # "ok" | "warning" | "error"
report.to_dict()  # JSON-compatible

result = import_gltf("model.gltf")  # scene + node map + diagnostics
for d in result.diagnostics:
    print(d.code, d.severity, d.location, d.message)
```

- **Exact conversion.** Aether3D's +Z-up space converts exactly to glTF's
  +Y-up space, using the same convention as Blender.
- **Unsupported content is reported, never silently dropped.**
  - Non-geometric content (materials, animations, cameras, …) is skipped and
    listed in the diagnostics.
  - Content whose loss would change geometry (skins, morph targets, points
    and lines) is refused unless you pass `allow_lossy=True`.
- **Verified exports.** Output is checked with the official Khronos glTF
  Validator.

See [docs/GLTF.md](docs/GLTF.md) and [docs/INSPECTION.md](docs/INSPECTION.md).
[`examples/gltf_roundtrip.py`](examples/gltf_roundtrip.py) builds a town
scene, exports it, validates it, imports it back, and verifies every property.

### Mesh editing

```python
from aether3d import Scene

scene = Scene()
crate = scene.add_cube("Crate", size=2)

with crate.edit_mesh() as edit:  # commits on success, discards on error
    top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
    edit.inset_faces(top, 0.2)
    edit.extrude_faces(top, -0.1)  # a recessed lid
    edge = edit.select_edges(nearest=(1, 1, 0))  # exactly one edge, or an error
    edit.bevel_edges(edge, 0.05)
    report = edit.validate()  # topology checks before committing
```

- **Editable topology.** The engine works on BMesh-style polygon topology with
  stable vertex, edge and face ids.
- **Safe operations.** Every operation is atomic and undoable, reports exactly
  which ids it created and removed, and fails with a stable error code instead
  of producing broken geometry.
- **Faithful attributes.** UVs and normals are preserved or interpolated, and
  any loss is reported.
- **Instancing respected.** Editing one object never changes another object
  that shares its mesh, unless asked to.

See [docs/EDITING.md](docs/EDITING.md).
[`examples/minecraft_hero.py`](examples/minecraft_hero.py) models a blocky
hero and sword entirely with editing operations, validates them, exports a
GLB, and checks it with the Khronos validator and, optionally, Blender.

### Procedural geometry

```python
from aether3d import Expr, Scene

scene = Scene()
scene.procedural.add_parameter("count", 5, min=1, max=50)
crate = scene.add_geometry("Crates", "box", width=1, depth=1, height=1)
crate.add_modifier("bevel", width=0.08, limit="all")
crate.add_modifier("array", count=Expr("count"), relative_offset=(1.2, 0, 0))
row = scene.add_object("Row", geometry=crate)  # row.mesh is the evaluated result

scene.procedural.set_parameter("count", 8)  # regenerates; invalid values are refused
scene.procedural.undo()
```

- **Non-destructive.** Objects use the *evaluated* result of a geometry
  definition (a source plus a modifier stack); the base definitions are
  never modified. Stacks can be reordered, disabled, edited and baked.
- **Parametric.** Typed, range-checked parameters; safe expressions
  (`{"expr": "floor(width / 2)"}`) that are never Python `eval`; and rules
  that reject invalid combinations.
- **Safe and fast.** Every change is an atomic, undoable transaction.
  Dependency cycles are detected. Content-addressed caching recomputes only
  what changed, and shares meshes between instances.
- **Persistent and exportable.** Definitions are saved in `.aether` format
  version 2 (version 1 projects still load). Exports always use the current
  evaluation.
- **Booleans** use [Manifold](https://github.com/elalish/manifold), an
  optional dependency: `pip install "aether3d[boolean]"`.

See [docs/PROCEDURAL.md](docs/PROCEDURAL.md).
[`examples/minecraft_castle.py`](examples/minecraft_castle.py) builds a
parametric Minecraft-style castle (configurable width, wall height, tower
count and radius, gate size, and detail density), then saves, reloads,
modifies, validates and exports it, and checks the GLB with the Khronos
validator and, optionally, Blender.

### Rendering

![A procedural Minecraft-style castle rendered by Aether3D](examples/output/minecraft_castle_render.png)

*Rendered by Aether3D's own CPU rasterizer:
[`examples/minecraft_castle_render.py`](examples/minecraft_castle_render.py),
1920×1080, about 2 s.*

```python
from aether3d import Scene

scene = Scene()
scene.add_plane("Ground", size=20).color = "#6f9b4f"
scene.add_cube("Crate", size=2, position=(0, 0, 1)).color = "#c0392b"
camera = scene.add_camera("Camera", fov=40)  # becomes the active camera
camera.frame(direction=(-0.5, 1.0, -0.45))  # fit everything in view
scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1.0), intensity=2.0)
result = scene.render(width=640, height=360)
result.save("crate.png")  # a real PNG, written by Aether3D
```

- **A real renderer.** The pipeline runs camera projection, homogeneous
  clipping, exact watertight fixed-point rasterization, an order-independent
  depth test, and perspective-correct deferred shading. Supersampling is
  averaged in linear light, and the PNG encoder is Aether3D's own.
- **Scenes from every stage render.** That includes static meshes, imported
  glTF, edited meshes, and procedural objects, which render their current
  evaluation without being baked.
- **Persistent setup.** Cameras, lights, colors, the background and render
  settings are saved in the project (`.aether` format version 3).
- **Verified.** Correctness is checked against analytic results and an
  independent ray caster. Blender, as an external check, agrees with the
  castle's silhouette on all but 2 of 2,073,600 pixels.

See [docs/RENDERING.md](docs/RENDERING.md).
[`examples/first_render.py`](examples/first_render.py) and
[`examples/hero_render.py`](examples/hero_render.py) render more scenes.

### GPU rendering

![The castle: CPU renderer | GPU | GPU with shadows](examples/output/castle_comparison.png)

*The same scene and camera, rendered by Aether3D's CPU renderer, its GPU
backend, and the GPU backend with a sun shadow map
([`examples/gpu_render.py`](examples/gpu_render.py)).*

```python
result = scene.render(backend="gpu", width=1920, height=1080, samples=2, shadows=True)
print(result.backend["adapter"]["name"])  # e.g. NVIDIA GeForce RTX 4060 Laptop GPU
```

- **Real GPU rendering.** WGSL shaders, instanced draw calls, a hardware depth
  buffer, MSAA, shadow maps and framebuffer readback run on Vulkan, Metal or
  Direct3D 12 through [wgpu](https://github.com/pygfx/wgpu-py). No window or
  display is needed.
- **Same scenes, same results.** The GPU renders everything the CPU renderer
  does. It matches the CPU pixel for pixel in parity mode (the CPU's sample
  positions) up to sub-pixel edge snapping. The castle at 1080p shows the
  same object at 100% of pixels.
- **Fast.** Measured on an RTX 4060 Laptop GPU with 4 samples per pixel:
  - the castle at 1080p renders in 44 ms instead of 1.24 s (28×);
  - at 4K, 89 ms instead of 4.5 s (51×);
  - a 1M-triangle mesh at 1080p takes 47 ms (22×);
  - 10,000 objects take 0.21 s (9×).

  Shadows cost a few milliseconds.
- **Honest fallback and discovery.**
  - `backend="auto"` uses the GPU when it works, else the CPU, and says why.
  - `backend="gpu"` never silently falls back.
  - `aether3d gpu info --check` lists the adapters, selects one, renders a
    verified test image and reports the result.

See [docs/GPU.md](docs/GPU.md) for installation (`pip install "aether3d[gpu]"`,
Python 3.11+), adapters, shadows, verification and benchmarks.

## Command-line interface

```console
$ aether3d new world.aether --name "My World"
$ aether3d validate world.aether
$ aether3d inspect examples/first_scene.aether
Scene "First Scene" (5 objects, format v1)
├── Cube  [d0da5f96-…]  pos=(0, 0, 1)  mesh=24v/12t
├── Sphere  [f9788aa5-…]  pos=(3, 0, 1)  mesh=561v/960t
├── Ground  [efddd1b2-…]  pos=(0, 0, 0)  mesh=4v/2t
└── Pivot  [c1181286-…]  pos=(3, 0, 1)  rot_euler=(0, 0, 90)
    └── Moon  [8122a0f9-…]  pos=(2, 0, 0)  mesh=561v/960t
```

Interchange and geometry analysis:

```console
$ aether3d export world.aether world.glb       # .aether (or .gltf) -> .glb / .gltf
$ aether3d import model.gltf model.aether      # add --allow-lossy to drop unsupported content
$ aether3d topology world.aether --object Crate --json     # polygon topology + validation
$ aether3d edit world.aether script.json --json            # declarative JSON edit script
$ aether3d procedural run castle.aether build.json         # declarative procedural script
$ aether3d procedural set castle.aether tower_count=8      # change parameters (validated)
$ aether3d procedural inspect castle.aether --evaluate     # parameters, nodes, stacks
$ aether3d procedural types --json                         # primitive/modifier schemas
$ aether3d camera add castle.aether --frame --fov 30      # camera framed on everything
$ aether3d light add castle.aether --direction 0.5 0.6 -0.8 --intensity 2.4
$ aether3d object set castle.aether Keep --color "#8c7356"
$ aether3d render castle.aether -o castle.png --width 1920 --height 1080
$ aether3d render model.glb -o preview.png --frame          # any project or glTF, no setup
$ aether3d gpu info --check                                 # GPU adapters + a verified test render
$ aether3d render castle.aether -o castle.png --backend gpu --shadows --width 1920 --height 1080
$ aether3d inspect model.glb                   # validate/inspect also read .glb/.gltf
$ aether3d geometry examples/first_scene.aether
Geometry of "First Scene": OK
  objects: 5 (4 with meshes), unique meshes: 4
  triangles: 1934 unique, 1934 instanced
  world bounds: (-5, -5, 0) .. (5, 5, 2)
  mesh 0: 24v/12t, area 24, closed, volume 8 [ok] used by 1 object(s)
  ...
```

Add `--json` to any command for machine-readable output. Errors point at the
exact element that is wrong:

```console
$ aether3d validate broken.aether --json
{
  "ok": false,
  "command": "validate",
  "error": {
    "type": "ProjectFormatError",
    "message": "expected an array of 3 numbers",
    "location": "scene.objects[1].transform.scale"
  }
}
```

Exit codes:

- `0`: success
- `1`: failure (invalid file, missing file, unsupported content, …)
- `2`: usage error
- `3`: `geometry`, `topology` or `procedural validate` found error-severity
  problems
`python -m aether3d` works the same as `aether3d`.

## Installation

### Arch Linux

```bash
sudo pacman -S --needed git python
git clone https://github.com/Murisixz/Aether3D.git
cd Aether3D
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # editable install + pytest, ruff, trimesh, manifold3d, pillow, wgpu
pytest                      # run the test suite
```

Optional: to run the official Khronos glTF Validator locally (tests use it
when present), install Node.js and the pinned validator:

```bash
sudo pacman -S --needed nodejs npm
npm ci --prefix tools/gltf-validator
```

Arch marks the system Python as externally managed (PEP 668), so install
Aether3D into a virtual environment as shown rather than with a system-wide
`pip install`. See [docs/INSTALL.md](docs/INSTALL.md) for details and other
platforms.

### Other platforms

Aether3D is pure Python (3.10+) with NumPy as its only required runtime
dependency, and is tested on Linux, macOS and Windows:

```bash
pip install -e ".[dev]"
```

Procedural booleans additionally need the optional `boolean` extra
(`pip install "aether3d[boolean]"`, which installs manifold3d, Apache-2.0).
GPU rendering needs the optional `gpu` extra (`pip install "aether3d[gpu]"`,
Python 3.11+, which installs wgpu, BSD-2-Clause, with wgpu-native, MIT or
Apache-2.0) and a Vulkan, Metal or Direct3D 12 driver; see
[docs/GPU.md](docs/GPU.md#installing-gpu-support).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/API.md](docs/API.md) | Python API reference and conventions |
| [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md) | The versioned `.aether` project format |
| [docs/GLTF.md](docs/GLTF.md) | glTF 2.0 import/export: features, axis conversion, limitations, safety |
| [docs/INSPECTION.md](docs/INSPECTION.md) | Geometry inspection reports and diagnostic codes |
| [docs/EDITING.md](docs/EDITING.md) | Mesh editing: topology, ids, sessions, operations, scripts, limitations |
| [docs/PROCEDURAL.md](docs/PROCEDURAL.md) | Procedural geometry: parameters, expressions, modifiers, caching, scripts, CLI, limitations |
| [docs/RENDERING.md](docs/RENDERING.md) | Cameras, lights, the CPU renderer: pipeline, conventions, shading model, API, CLI, verification, performance, limitations |
| [docs/GPU.md](docs/GPU.md) | The GPU backend: installation, adapters, pipeline, antialiasing, shadows, memory, CPU/GPU differences, benchmarks, debugging, the backend decision |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design principles, project structure, extension points |
| [docs/INSTALL.md](docs/INSTALL.md) | Installation details |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [AGENTS.md](AGENTS.md) | Orientation for AI coding agents working on this repository |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Conventions at a glance

- Right-handed coordinates, **+Z up**, 1 unit = 1 meter; an asset's front faces −Y.
- glTF (+Y up) conversion: `glTF (x, y, z) = (x, z, −y)`; UV origin is bottom-left (`v` flipped for glTF).
- Rotations are unit quaternions `(x, y, z, w)`; `rotation_euler` offers XYZ Euler angles in degrees.
- Transforms compose as `T @ R @ S`; `world = parent.world @ local`.
- Object ids are UUID strings that never change, including across save/load.
- Object names are unique within a scene.
- Faces wind counter-clockwise seen from the front; editing ids are stable within a session.
- Procedural geometry nodes and modifiers have stable UUID ids; objects reference nodes by id.
- Cameras and lights look along their local −Y axis with +Z up (the same "front faces −Y" rule).
- User-facing colors are display sRGB (`"#rrggbb"`); lighting happens in linear light.
- Images: row 0 at the top, pixel centers at `+0.5`; NDC depth runs from 0 (near) to 1 (far)
  (the GPU backend uses reversed depth internally, for precision).

## License

Aether3D is licensed under the [Apache License 2.0](LICENSE).
