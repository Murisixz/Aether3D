# Notes for AI coding agents

This file orients AI agents working **on** the Aether3D codebase. Human
contributors should read [CONTRIBUTING.md](CONTRIBUTING.md); the rules there
apply to agents too.

## Essentials

- Package source: `src/aether3d/`. Tests: `tests/`. Docs: `docs/`.
- Setup: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- Verify every change with `ruff check . && ruff format --check . && pytest`.
- Architecture and module boundaries: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- Public API: [docs/API.md](docs/API.md). File format: [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md).
- glTF interchange: [docs/GLTF.md](docs/GLTF.md). Geometry reports: [docs/INSPECTION.md](docs/INSPECTION.md).
- Mesh editing: [docs/EDITING.md](docs/EDITING.md) (topology, ids, sessions, operations, scripts).
- Optional: `npm ci --prefix tools/gltf-validator` enables the Khronos validator tests locally.

## Conventions to preserve

- +Z up, right-handed; quaternions are `(x, y, z, w)`; Euler angles are XYZ in degrees.
- Matrices are column-vector 4x4 `float64`; local = `T @ R @ S`; world = `parent.world @ local`.
- Public getters return tuples or fresh arrays, never mutable internal state.
- Validate all inputs before mutating; raise `AetherError` subclasses with specific messages.
- File format changes require a `FORMAT_VERSION` bump, a migration, a test, and doc updates.
- Z-up <-> glTF Y-up conversion lives only in `gltf/axes.py`; it must stay exact (see its docstring).
- Never silently drop or alter content in a conversion: emit a `Diagnostic`, and mark it
  `lossy=True` (which makes the conversion fail unless `allow_lossy=True`) if geometry would change.
- Treat imported files as untrusted: validate before allocating, keep file access inside the
  asset's directory, raise `AssetFormatError` with a `code` and JSON-pointer `location`.
- Editing: mutate an `EditableMesh` only through the journaled primitives in
  `editing/core.py`; new operations are `@operation` methods that validate first and raise
  `EditError` with a stable code (never leave a partial edit; never approximate an operation).
  Run `tests/test_editing_robustness.py` after touching topology code.

## Using Aether3D from an agent

- Prefer ids (`obj.id`) over names for durable references.
- Use `aether3d inspect PATH --json` to read a project's structure and
  `aether3d validate PATH --json` after editing a `.aether` file by hand.
  Error responses include a `location` such as `scene.objects[2].transform`.
- `aether3d import`/`export` convert between `.aether` and `.glb`/`.gltf`; read the
  `diagnostics` in the JSON output, and only pass `--allow-lossy` deliberately.
- `aether3d geometry PATH --json` reports geometric problems; match on diagnostic `code`
  values, not messages. Exit code `3` means error-severity findings.
- Object ids survive glTF round trips (stored in `node.extras.aether3d.id`); names may
  be suffixed (`.001`) if an external file repeats them.
- Model geometry with `obj.edit_mesh()` (Python) or `aether3d edit PROJECT script.json`
  (JSON). Use `expect=` on selections, prefer `nearest=`/`normal=`/`box=` over guessing ids,
  read `created`/`removed` from each result, and run `validate` before committing.
- `aether3d topology PROJECT --object NAME --elements --json` lists element ids, positions and
  face loops; ids from a fresh session are deterministic for the same mesh.
