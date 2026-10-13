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
│   ├── gltf/              glTF 2.0 interchange (no third-party dependencies)
│   │   ├── axes.py        Exact Z-up <-> Y-up conversion
│   │   ├── writer.py      Export: scene -> glTF document + buffer -> .glb/.gltf
│   │   └── reader.py      Import: validated, resource-limited, path-confined
│   ├── cli.py             `aether3d` command-line interface
│   └── __main__.py        `python -m aether3d`
├── tests/                 pytest suite, one file per module
│   └── data/              Khronos sample assets (pinned) and a v0.1.0 project
├── examples/              Runnable examples and a sample .aether project
├── tools/gltf-validator/  Pinned Khronos glTF Validator runner (Node.js, dev only)
├── benchmarks/            Editing-engine benchmarks (bench_editing.py)
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
| `cli` | `editing`, `gltf`, `serialization`, `scene`, `errors` |

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
| Rendering | Consumes `Scene` iteration, `world_matrix`, and `Mesh` arrays read-only; lives in its own package/module and is not imported by the core. |
| Animation | Targets objects by `id`; writes transform components through the public setters. |
| Physics | Reads/writes world transforms; stores per-object data alongside the transform. |
| Procedural modeling | Functions that return `Mesh` (like `primitives`), or that build objects through `Scene`. |
| Materials / shaders | New per-object data plus a format version bump. |
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
