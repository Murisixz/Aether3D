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
│   ├── math3d.py          Stateless vector/quaternion/matrix helpers (NumPy)
│   ├── transform.py       Transform: position / rotation / scale
│   ├── mesh.py            Mesh: immutable indexed triangle mesh
│   ├── primitives.py      Cube, sphere and plane generators
│   ├── scene.py           Scene and SceneObject: the scene graph
│   ├── serialization.py   .aether reader/writer, validation, migrations
│   ├── cli.py             `aether3d` command-line interface
│   └── __main__.py        `python -m aether3d`
├── tests/                 pytest suite, one file per module
├── examples/              Runnable examples and a sample .aether project
├── docs/                  API, file format, architecture, installation
└── .github/workflows/     Continuous integration
```

### Module dependencies

| Module | Imports (internal) |
| --- | --- |
| `errors` | none |
| `math3d` | `errors` |
| `mesh` | `errors` |
| `transform` | `math3d` |
| `primitives` | `mesh`, `math3d` |
| `scene` | `transform`, `mesh`, `primitives`, `math3d`, `errors` |
| `serialization` | `scene`, `transform`, `mesh`, `errors` |
| `cli` | `serialization`, `scene`, `errors` |

Lower layers never import higher ones (`scene` imports `serialization` lazily, only inside
`Scene.save`/`Scene.load`, to provide the convenience methods).

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
