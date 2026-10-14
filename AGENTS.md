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
- Procedural geometry: [docs/PROCEDURAL.md](docs/PROCEDURAL.md) (parameters, expressions, modifiers,
  caching, transactions, scripts).
- Cameras, lights and rendering: [docs/RENDERING.md](docs/RENDERING.md) (conventions, pipeline,
  shading model, API, CLI, verification); the GPU backend: [docs/GPU.md](docs/GPU.md).
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
- Procedural: modifiers are pure functions `PolyMesh -> PolyMesh` registered in
  `procedural/modifiers.py` with a `ParamSpec` schema; never mutate inputs (arrays are
  read-only), raise `ProceduralError` with a stable code instead of returning damaged
  geometry, and report noteworthy results as `Diagnostic`s. Cache keys must cover every
  input a result depends on. Run `tests/test_procedural_randomized.py` after changes.
- Expressions are evaluated by the AST whitelist in `procedural/params.py`; never use
  `eval`/`exec` or widen it to attribute access, indexing or arbitrary calls.
- Rendering: images are produced by Aether3D's own code (`rendering/`). Never shell out to Blender
  or another renderer to make an image; Blender may only be used for optional external checks.
  Backends consume a `RenderScene` from `extract_render_scene` and must not touch scene objects.
  Rendering must never modify the scene or bake procedural geometry.
- Cameras and lights look along local -Y with +Z up; user colors are display sRGB and lighting is
  linear; keep per-vertex/per-sample math explicit (no BLAS) so results stay bitwise reproducible.
  Validate new shading or projection code against `tests/render_reference.py` (an independent ray
  caster) and analytic values; regenerate `tests/data/renders` only for intended changes
  (`python tests/render_goldens.py --update`).
- GPU backend (`rendering/gpu/`): only this package imports wgpu, and only lazily -- `import
  aether3d` and the CPU renderer must keep working without it. Shaders come from the bundled
  `wgsl/` files only (never compile text from scenes, projects or the CLI). Keep the host-side
  byte layouts in `gpu/resources.py` in sync with the WGSL structs (`tests/test_gpu_data.py`
  checks them). Lighting changes must stay identical in `cpu.py` and `wgsl/scene.wgsl`; check
  them with the parity tests in `tests/test_gpu_render.py` (`supersample=s, msaa=1`) and shadow
  changes with the ray-traced references in `tests/test_gpu_shadows.py`. Never report a GPU as
  used or available without a created device, and label software adapters as such.
- Run GPU tests locally with `AETHER3D_REQUIRE_GPU=1 pytest tests/test_gpu_*.py` (optionally
  `AETHER3D_TEST_GPU_ADAPTER=llvmpipe` or `=nvidia`); `AETHER3D_GPU_DEBUG=1` enables validation.

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

- Build or change procedural geometry with `scene.procedural` (Python) or
  `aether3d procedural run PROJECT script.json` (JSON). Read `aether3d procedural types --json`
  for schemas, wrap related changes in one `batch()`/script, and match on
  `ProceduralError.code` and `location`. Procedural objects' meshes are read-only: change
  the definition, `bake()`, or edit the base mesh with `geometry.edit_source()`.
- `aether3d procedural validate PROJECT --json` evaluates every node; exit code `3` means a
  node failed. `aether3d procedural set PROJECT name=value` changes parameters atomically.
- To see a scene: `aether3d render PROJECT -o out.png --json` (add `--frame` if it has no camera;
  works on `.glb`/`.gltf` too). Read `objects[].visible_samples` to check what is visible and
  `diagnostics` (e.g. `DEFAULT_LIGHTING`, `NOTHING_TO_RENDER`). Set up cameras, lights and colors
  with `aether3d camera add --frame`, `light add`, `object set --color`, `scene set`, or one
  atomic `aether3d scene run PROJECT script.json`. In Python: `scene.add_camera(...).frame()`,
  `scene.add_directional_light(...)`, `scene.render(...)`, `result.save("x.png")`.
- GPU: `aether3d gpu info --check --json` says whether GPU rendering works here and which adapter
  is used (exit code 3 if it does not). Render with `--backend gpu` (fails with a `GPU_*` code if
  unavailable -- never a silent CPU render) or `--backend auto` (read `backend.name` and the
  `BACKEND_FALLBACK` diagnostic). Add `--shadows` for sun shadows; `backend.adapter.software: true`
  means a CPU-emulated GPU (correct, not hardware speed). Python: `scene.render(backend="gpu",
  shadows=True)`, `aether3d.gpu_status()`, `aether3d.list_adapters()`.
