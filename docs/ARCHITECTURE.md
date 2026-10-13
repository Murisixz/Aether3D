# Architecture

This document describes how the Aether3D foundation is organized, the
principles behind it, and where future systems are expected to plug in.

## Principles

- **Code-first.** Every capability is a Python call. The CLI is a thin layer
  over the same API, with no logic of its own beyond formatting.
- **AI-native.** Behaviour is predictable and failures are explicit:
  - inputs are validated eagerly, with messages that name the bad argument;
  - operations either fully succeed or leave state unchanged;
  - nothing is silently corrected (an explicit duplicate name is an error, an
    unknown key in a project file is an error);
  - identifiers are stable, names are unique, and output is deterministic;
  - every CLI command can answer in JSON, including errors with locations.
- **Small and replaceable modules.** Each module has one job and depends only
  on modules below it (see the dependency graph).
- **Data, not behaviour, at the boundaries.** Meshes are plain arrays, transforms
  are plain tuples, project files are plain JSON. This keeps the door open for
  native code and other tools without an API rewrite.

## Project structure

```text
Aether3D/
├── src/aether3d/          The Python package
│   ├── __init__.py        Public API surface (__all__)
│   ├── _version.py        Package version (single source of truth)
│   ├── errors.py          Exception hierarchy
│   ├── diagnostics.py     Diagnostic: structured findings (code, severity, location, ...)
│   ├── _fileio.py         Atomic file writes shared by all writers
│   ├── math3d.py          Stateless vector/quaternion/matrix helpers (NumPy)
│   ├── transform.py       Transform: position / rotation / scale
│   ├── mesh.py            Mesh: immutable indexed triangle mesh
│   ├── primitives.py      Cube, sphere and plane generators
│   ├── scene.py           Scene and SceneObject: the scene graph
│   ├── serialization.py   .aether reader/writer, validation, migrations
│   ├── inspection.py      Geometry analysis of meshes and scenes
│   ├── editing/           Polygon mesh editing engine (see docs/EDITING.md)
│   │   ├── core.py        EditableMesh storage, journaled primitives, transactions
│   │   ├── selection.py   Selection queries, loops, rings, conversions
│   │   ├── transform.py   Local/world transforms, mirroring, normal transforms
│   │   ├── topology.py    Create/delete/merge/split/dissolve/triangulate
│   │   ├── extrude.py     Region/individual extrusion and inset
│   │   ├── bevel.py       Edge chamfer
│   │   ├── subdivide.py   Subdivision and loop cuts
│   │   ├── normals.py     Shading model and corner-normal recomputation
│   │   ├── validation.py  Polygon-level topology validation
│   │   ├── session.py     EditSession: commit/discard on a scene object
│   │   ├── script.py      Declarative JSON edit scripts (CLI, future MCP)
│   │   └── _polygon.py    Newell normals, ear clipping, simplicity, affine fits
│   ├── procedural/        Non-destructive procedural geometry (see docs/PROCEDURAL.md)
│   │   ├── polymesh.py    PolyMesh: immutable CSR polygon mesh flowing through stacks
│   │   ├── params.py      Parameters, safe expressions (Expr), rules, schemas
│   │   ├── primitives.py  Parametric primitive types (box, grid, cylinder, cone, ...)
│   │   ├── modifiers.py   Modifier types (transform, mirror, array, ..., boolean)
│   │   ├── graph.py       ProceduralSystem: nodes, dependencies, evaluation, cache, undo
│   │   ├── session.py     Editing a node's stored base mesh with the editing engine
│   │   └── script.py      Declarative JSON procedural scripts
│   ├── rendering/         Cameras, lights and the CPU renderer (see docs/RENDERING.md)
│   │   ├── camera.py      Camera parameters, projection matrices, camera frames
│   │   ├── light.py       Light parameters
│   │   ├── settings.py    Appearance, World, RenderSettings, RenderLimits
│   │   ├── color.py       sRGB <-> linear transfer functions, color parsing
│   │   ├── scene_data.py  RenderScene extraction (backend-independent)
│   │   ├── raster.py      Clipping, fixed-point rasterization, visibility buffer
│   │   ├── cpu.py         CPU backend: geometry, raster, deferred shading, resolve
│   │   ├── renderer.py    render() and RenderResult
│   │   ├── framing.py     Exact camera framing
│   │   ├── png.py         PNG encoder (zlib from the standard library)
│   │   └── script.py      Declarative scene-setup scripts
│   ├── gltf/              glTF 2.0 interchange (no third-party dependencies)
│   │   ├── axes.py        Exact Z-up <-> Y-up conversion
│   │   ├── writer.py      Export: scene -> glTF document + buffer -> .glb/.gltf
│   │   └── reader.py      Import: validated, resource-limited, path-confined
│   ├── cli.py             `aether3d` command-line interface
│   └── __main__.py        `python -m aether3d`
├── tests/                 pytest suite, one file per module
│   └── data/              Khronos samples (pinned), v0.1.0/v0.4.0 projects, reference renders
├── examples/              Runnable examples, a sample project, and the first renders (output/)
├── tools/gltf-validator/  Pinned Khronos glTF Validator runner (Node.js, dev only)
├── benchmarks/            Editing, procedural and rendering benchmarks (bench_*.py)
├── docs/                  API, file formats, glTF, inspection, architecture, install
└── .github/workflows/     Continuous integration
```

### Module dependencies

| Module | Imports (internal) |
| --- | --- |
| `errors`, `diagnostics`, `_fileio` | none |
| `math3d` | `errors` |
| `mesh` | `errors` |
| `transform` | `math3d` |
| `primitives` | `mesh`, `math3d` |
| `scene` | `transform`, `mesh`, `primitives`, `math3d`, `errors` |
| `serialization` | `scene`, `transform`, `mesh`, `errors`, `_fileio` |
| `inspection` | `mesh`, `math3d`, `diagnostics` |
| `gltf.axes` | `math3d` |
| `gltf.writer` | `gltf.axes`, `mesh`, `diagnostics`, `errors`, `_fileio` |
| `gltf.reader` | `gltf.axes`, `gltf.writer` (constants), `scene`, `mesh`, `transform`, `math3d`, `diagnostics`, `errors` |
| `editing.*` | `mesh`, `math3d`, `diagnostics`, `errors`, `primitives` (box layout), `inspection` (session commit reports); `scene` only for type hints and the session |
| `procedural.*` | `mesh`, `math3d`, `diagnostics`, `errors`, `inspection`, `editing` (bevel, base-mesh sessions, polygon helpers); `scene` only for type hints; `manifold3d` (optional, lazily, boolean modifier only) |
| `serialization` (v2) | additionally `procedural.params` (value encoding), and `procedural` lazily when reading |
| `rendering.camera`, `.light`, `.settings`, `.color` | `errors`, `diagnostics` (value types; imported by `scene` and `serialization`) |
| `rendering.*` (others) | the above, `mesh`, `math3d`, `_fileio`; `scene` only for type hints and lazily for lookups |
| `cli` | `editing`, `procedural`, `rendering`, `gltf`, `serialization`, `scene`, `errors` |

Lower layers never import higher ones. `scene` provides convenience methods
(`save`/`load`, `export`/`import_file`, `inspect_geometry`, `edit_mesh`) by importing
`serialization`, `gltf`, `inspection` and `editing` lazily inside those methods, and
`inspection`/`gltf.writer` refer to `Scene` only for type hints.

## Core concepts

### Scene graph

A `Scene` owns a set of `SceneObject`s and indexes them by id and by name. Each
object holds a parent pointer and an ordered list of children; the scene keeps
an ordered list of roots. Iteration is depth-first, which is also the order
objects are written to disk, so files and CLI output are stable.

Objects are created only by the scene (`add_object`, `add_cube`, …), which
guarantees every object has a valid id, a unique name, and a consistent place
in the hierarchy. Removed objects are detached (`obj.scene is None`) and refuse
further hierarchy operations.

`SceneObject` is deliberately a generic node: today it carries a transform and
an optional mesh. Future capabilities (lights, cameras, materials, physics
bodies, …) are expected to attach to objects as additional optional data, and
the decision of how (dedicated attributes vs. a component registry) is left
until the first such capability exists, so the abstraction matches real needs.

### Identity

Ids are UUID4 strings, generated at creation and preserved by save/load. They
are the durable way to refer to an object (from scripts, files, or future
systems such as animation tracks). Names are unique per scene and are meant for
humans and convenient lookups; they can change, ids cannot.

### Transforms

`Transform` stores position, a unit quaternion, and scale as immutable tuples
of floats. Matrices are computed on demand (`T @ R @ S`), and world matrices
are the product of local matrices up the parent chain. Computing on demand
avoids cache-invalidation bugs; caching can be added inside `SceneObject`
later without changing the API.

`math3d` holds all the math as stateless functions over tuples and NumPy
arrays. It is the natural first candidate for a native implementation.

### Geometry

`Mesh` is an immutable indexed triangle mesh: `float32` positions, optional
normals and UVs, `uint32` triangle indices. These contiguous NumPy arrays
match what GPUs and native libraries consume, and they expose the buffer
protocol for zero-copy handoff to Rust/C++ extensions. Immutability makes
sharing a mesh between objects (instancing) safe.

New vertex attributes (colors, tangents, …) can be added as further optional
arrays without breaking existing code. Primitive generators are plain
functions returning a `Mesh`; future procedural tools can follow the same shape.

### Persistence

`.aether` files are versioned JSON (see [FILE_FORMAT.md](FILE_FORMAT.md)). The
reader validates everything and reports errors with a precise location. A
`MIGRATIONS` table upgrades old files one version at a time, and files from
newer versions are refused rather than partially read. Saves are atomic
(write to a temporary file, then rename).

Meshes are stored in a table and referenced by index, so shared meshes are
stored once. JSON keeps files transparent for agents and code review. If very
large meshes later need binary storage, that becomes a new format version
(e.g. a buffer reference in place of inline arrays) with a migration.

## Extending Aether3D

| Future system | Expected integration point |
| --- | --- |
| Rendering | Implemented in v0.5.0 (`rendering/`): backends consume a `RenderScene` from `extract_render_scene` (see below). |
| Animation | Targets objects by `id`; writes transform components through the public setters. |
| Physics | Reads/writes world transforms; stores per-object data alongside the transform. |
| Procedural modeling | Implemented in v0.4.0 (`procedural/`): new primitive and modifier types are registry entries (see below). |
| Materials / shaders | Replace `Appearance` (the only place surface properties enter shading) with materials; extend the per-triangle shading table. Format version bump. |
| Asset management / interop | Importers/exporters that convert to/from `Scene` and `Mesh`, outside the core. |
| Native acceleration | Replace `math3d` functions or mesh generation with Rust/C++ (e.g. PyO3) behind the same Python signatures. |

Ground rules for extensions:

1. The core (`scene`, `transform`, `mesh`, `math3d`, `serialization`) must
   not import optional subsystems.
2. Any change to the file format bumps `FORMAT_VERSION`, adds a migration, and
   updates `docs/FILE_FORMAT.md`.
3. Public API additions are exported from `aether3d/__init__.py` and documented
   in `docs/API.md`.

## Interchange and inspection (v0.2.0)

Interchange code is an adapter around the core, not part of it. `gltf`
converts between external documents and the existing `Scene`/`Mesh` types,
and the core never imports it except lazily for convenience methods. Future
formats (OBJ, USD, FBX, ...) can be added the same way, as sibling packages
with their own reader and writer.

Several design decisions keep conversions predictable:

- **No third-party glTF library.** The container format is small. A direct
  reader and writer over `json`, `struct` and NumPy keep full control over
  validation, error locations, resource limits and exactness. Correctness is
  established externally instead: the official Khronos validator, an
  independent parser (trimesh, test-only), and pinned Khronos sample assets.
- **Exact axis conversion.** The Z-up → Y-up map is a signed axis
  permutation applied to TRS components, mesh data and every node.
  Conversion is therefore lossless and hierarchy-consistent, and needs no
  extra root node (see [GLTF.md](GLTF.md)).
- **One `Diagnostic` type for every report.** Inspection, import and export
  all report findings the same way: a stable `code`, a `severity`, a
  `location` and a `lossy` flag.
- **Lossy conversions are opt-in.** Content that a conversion would have to
  drop or approximate makes it fail with `UnsupportedFeatureError` listing
  every item, unless the caller passes `allow_lossy=True`. Non-geometric
  content that is merely skipped is reported but does not fail.
- **No partial state.** Imports build a new `Scene` and return it only on
  success. Exports build the complete file in memory, then write it
  atomically.
- **Inspection is read-only and vectorized.** It never mutates scenes, and
  each distinct mesh is analysed once. Its topology claims are limited to
  what it actually computes on exactly-welded vertices (see
  [INSPECTION.md](INSPECTION.md)).

## Mesh editing (v0.3.0)

The editing engine is a layer on top of the core: it consumes and produces
immutable `Mesh` objects and never changes how they are stored. The core
design decisions are explained in [EDITING.md](EDITING.md). Most importantly:

- **BMesh-style topology in id-keyed tables.** Polygons, non-manifold
  geometry and per-corner attributes, with stable integer ids, stored as
  NumPy positions plus dictionaries. The alternatives that were considered
  are compared in EDITING.md.
- **All mutation goes through journaled primitives.** Operations are atomic
  (exact rollback on any exception) and undoable, without copying meshes.
  `OperationResult` id lists are derived from the journal, so they are always
  complete.
- **Operations are plain methods on mixins** (`TransformMixin`,
  `ExtrudeMixin`, ...) composed into `EditableMesh`. A new operation is one
  `@operation` method in the right module; it is then automatically
  undoable, listed by `EditableMesh.operations()`, and callable from JSON
  scripts.
- **Attributes are generic corner layers.** Interpolation code is
  layer-agnostic, so vertex colours, extra UV sets or a per-face material
  index can be added without touching the operations.
- **Sessions bind an editable mesh to a scene object.** Committing is the
  only way geometry reaches the scene: instancing stays intact and failed
  edits never leak.
- **Scripts are data, not code.** `run_edit_script` is the single entry point
  for declarative edits. The CLI uses it today; a future MCP server can expose
  it unchanged.

## Procedural geometry (v0.4.0)

The procedural engine sits beside the editing engine, above the core. Objects
reference **geometry nodes**, and `SceneObject.mesh` returns the node's
evaluated `Mesh`. Everything downstream (inspection, glTF, the CLI) is
therefore unchanged and always sees current geometry. The design is explained
in [PROCEDURAL.md](PROCEDURAL.md); the key decisions:

- **A value type for evaluation.** Modifiers map an immutable `PolyMesh` to a
  new one: CSR polygons with per-corner UV and normal layers, smooth flags and
  sharp edges, as read-only NumPy arrays. Evaluation can therefore share
  arrays freely, caching is safe, and modifiers are vectorized. The editing
  engine's `EditableMesh` (id-keyed, journaled) is built only where an
  operation needs it (bevel, editing a base mesh).
- **Definitions are data.** A node is a source plus a list of `(id, name,
  type, enabled, params)` entries, and parameter values are literals or `Expr`
  objects. Serialization, undo snapshots, `info()` and JSON scripts all
  operate on that data. Nothing is executed as code.
- **Content-addressed caching.** Cache keys are SHA-256 hashes of fully
  resolved inputs, chained through the stack. Invalidation is implicit: a
  changed input gives a different key, so results can never be stale. There
  is no dependency bookkeeping to get wrong, and identical sub-evaluations are
  shared across nodes. The cache is a byte-bounded LRU.
- **Transactions validate by evaluating.** After any change, rules, schemas
  and cycles are checked and every node is evaluated (mostly cache hits); on
  failure, the definitions are restored *into the existing handle objects*.
  Errors therefore appear at the change that caused them, and handles held by
  callers stay valid across rollback, undo and redo.
- **Registries for extension.** `PRIMITIVES` and `MODIFIER_TYPES` map names
  to small classes that declare a parameter schema (`ParamSpec`) and a pure
  function. A new primitive or modifier is one class: it is then validated,
  serialized, listed by `types()`, and usable from scripts and the CLI
  without further changes. Modifiers that read other geometry declare it via
  `references()`, which feeds the dependency graph and cycle detection.
- **Booleans through Manifold.** Solid booleans need robust geometric
  predicates that are impractical to reimplement correctly in pure Python.
  The boolean modifier is the only place that uses an optional third-party
  library: `manifold3d` (Apache-2.0), imported lazily, validated on input,
  and reported clearly when absent. The core and every other feature work
  without it.

## Rendering (v0.5.0)

The renderer is a layer above the core, beside editing and procedural
geometry. The scene gains only *data*: camera and light components on
objects, an appearance and a visibility flag, a world and render settings.
All rendering logic lives in `rendering/`. The design is explained in
[RENDERING.md](RENDERING.md); the key decisions:

- **Extraction separates scenes from backends.** `extract_render_scene` walks
  the scene once and produces an immutable `RenderScene`:
  - shared mesh arrays (procedural objects contribute their current cached
    evaluation);
  - per-instance world and normal matrices and linear colors;
  - world-space lights and the camera frame.

  The CPU backend consumes only this, and a GPU backend (Stage 6) can
  consume the same data, so `scene.render`, the CLI and the tests do not
  change. Rendering never modifies the scene.
- **A visibility buffer, then deferred shading.**
  - Rasterization writes one packed `int64` per sample (quantized depth
    above, triangle index below), using `np.minimum.at`.
  - Visibility is therefore order-independent and deterministic, and every
    covered sample is shaded exactly once.
  - Batches bound memory to about 8 bytes per sample plus per-pixel buffers.
- **Exact integer rasterization.** Vertices are snapped to 1/256 of a sample.
  Covered spans come directly from integer edge functions with the top-left
  rule, so the result is watertight (each sample of a shared edge is covered
  once) and no candidate pixels are wasted.
- **Homogeneous everything.** Clipping happens in clip space. Shading
  barycentrics come from the original triangle in homogeneous 2D
  coordinates, so near-plane clipping never distorts attributes. The world
  position is reconstructed exactly on the view ray.
- **Conventions shared with the rest of Aether3D.**
  - Cameras and lights use the same "front faces −Y, +Z up" convention as
    assets, so one `look_at` aims everything.
  - Colors follow one policy: sRGB in, linear for lighting, sRGB out.
  - The clip and NDC conventions (depth [0, 1], pixel centers at +0.5)
    match modern GPU APIs, ready for a GPU backend.
- **Data-only scripts.** `run_scene_script` is the single entry point for
  declarative setup changes. CLI commands build one-operation scripts, and
  the planned MCP server can expose it unchanged. Rollback is exact and keeps
  callers' handles valid.
- **Independent verification.** The test suite's reference renderer is a
  brute-force ray caster that shares no projection, clipping, rasterization
  or shading code with the renderer. Analytic tests and a cross-check of
  silhouettes with Blender, an external tool, complement it.
