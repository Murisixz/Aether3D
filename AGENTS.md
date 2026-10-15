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
  shading model, API, CLI, verification); the GPU backend: [docs/GPU.md](docs/GPU.md); the
  path tracer: [docs/RENDER_QUALITY.md](docs/RENDER_QUALITY.md) (BVH, light transport, BSDF,
  lights, environment, AO, sampling, color management, passes, performance).
- Materials, textures and procedural shaders: [docs/MATERIALS.md](docs/MATERIALS.md) (models and
  formulas, slots, images, coordinates, node graphs, presets, face groups, CLI, security).
- Undo, redo and transactions: [docs/HISTORY.md](docs/HISTORY.md) (what is recorded, design,
  history files, CLI).
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
- Materials (`materials/`, `rendering/materials.py`, `rendering/surface.py`,
  `rendering/gpu/materials.py`, `wgsl/material.wgsl`):
  - **Graphs are data.** Procedural graphs are validated against the node schema and evaluated
    by the NumPy evaluator or the fixed WGSL interpreter; never generate Python or WGSL from them.
  - **Both backends compute the same thing.** Sampling, noise and shading changes must stay
    identical in `sampling.py`/`noise.py`/`graph.py`/`surface.py` and `material.wgsl`. Check
    them with `tests/test_gpu_materials.py`, and keep the packer's layout constants in sync
    with the shader (`test_shader_constants_match_the_packer`). A new node type needs a schema,
    CPU evaluation, a GPU opcode and parity tests.
  - **Objects without a material render bit-identically to 0.6.** The `simple` model's
    arithmetic is frozen; the render goldens and examples depend on it.
  - **Untrusted images.** Decoders check sizes before allocating and raise `MaterialError`
    with a code. Texture paths stay inside the project's directory, textually and after
    resolving symbolic links.
  - **Face groups.** Editing operations and procedural modifiers must carry face groups to the
    faces they create (`tests/test_material_groups.py`).

- Path tracer (`rendering/raytrace/`, `rendering/pathtrace/`):
  - **CPU only, and say so.** It must never touch wgpu or claim GPU use; reports carry
    `"device": "cpu"`. It must never change what the rasterizers render.
  - **One material semantics.** Evaluate materials with `surface.evaluate_material`; do not fork
    material logic into the path tracer. Lights keep the raster-equivalent units documented in
    `pathtrace/lights.py`.
  - **Counter-based randomness only.** Draw random numbers with
    `sampling.uniform(keys, dimension(vertex, slot))`, a distinct slot per decision; never
    `np.random`. Images must not depend on batch sizes or `workers`
    (`test_renders_repeat_bit_for_bit_and_do_not_depend_on_batching`,
    `test_worker_processes_give_the_same_image_bit_for_bit`).
  - **Verify against references.** Check radiometry changes against the analytic tests in
    `tests/test_pathtrace.py`, BSDF changes against the quadrature tests in
    `tests/test_pathtrace_bsdf.py`, and BVH or traversal changes against the brute-force
    and watertightness tests in `tests/test_raytrace.py`. Never loosen a tolerance without
    deriving the new noise level.
  - **Every estimator stays unbiased** unless it is a documented, opt-in bias (`clamp`). An
    emitter must be counted once (MIS between light and BSDF sampling).
- GPU path tracer (`rendering/gpu/pathtrace.py`, `gpu/wgsl/pathtrace.wgsl`):
  - **It is a port of the CPU path tracer, kept in step.** Any change to the estimator,
    its constants or its random-number slots goes into both, and
    `tests/test_gpu_pathtrace.py` must still show the two tracing the same paths (same
    random numbers: `test_the_gpu_draws_the_cpu_path_tracers_random_numbers`). Only
    float32 margins may differ (documented in the kernel's header).
  - **One material evaluation.** `evaluate_surface` is called at exactly one place in the
    kernel (the state machine's loop): D3D12's FXC inlines every call site of the
    texture-program interpreter.
  - **Honest reports.** It runs on the adapter's shader cores with software BVH
    traversal: never claim hardware ray tracing, and report `"device": "gpu"` with the
    adapter. Without a usable GPU it fails with a `GPU_*` error, never a silent CPU render.

- Operations history (`history.py`):
  - **Every change of a scene is recorded.** A new public method that changes the
    scene runs as an operation (`scene._history._op(label, ...)`) and *touches* every
    aspect it writes before writing it (`history._touch(ASPECT, target)`); value objects
    go through their owner (`recording(...)`). Test it with the catalog in
    `tests/test_history.py`: undo must give back the same project document and the same
    Python objects, and redo the changed ones.
  - **Strict mode is on in the tests** (`history.STRICT`, conftest.py): a write recorded
    outside an operation fails, and every undo, redo and rollback re-checks the scene.
  - **Never write private attributes from outside their module** to change a scene:
    such changes bypass the history.
  - **A new kind of state needs an aspect** (capture, apply, its encoding in
    `_history_file.py`) rather than ad-hoc rollback code. Loaders build scenes with
    recording suspended (`history._suspended()`).

## Using Aether3D from an agent

- Prefer ids (`obj.id`) over names for durable references.
- Every change can be undone: `scene.history.undo()`, `with scene.transaction("label"):`
  for several changes as one step (all or nothing), `scene.history.checkpoint(name)` and
  `goto(name)` to try something and come back. On the command line, `aether3d history
  start PROJECT` makes a project keep a history file, then `aether3d history undo
  PROJECT` and `show`, `redo`, `checkpoint`, `goto` and `verify` work across commands.
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
- Materials:
  - Discover with `aether3d material schema --json` (parameters, slots, binding options, node
    types) and `aether3d material presets --json`.
  - Build with `material create`, `material assign` (`--group K` for a face group) and
    `material texture` (`--image`, `--procedural`, `--texture`).
  - Check with `aether3d material validate PROJECT --json` (exit code 3 on errors), then render
    and read `diagnostics` (`TEXTURE_MISSING`, `MISSING_UVS`, ...).
  - Python: `scene.create_material(...)`, `material.set_texture(slot, path_or_texture, ...)`,
    `material.set_procedural_texture(...)`, `scene.assign_material(obj, material, group=k)`.
    Match on `MaterialError.code`.
  - Keep image files inside the project's folder, or embed them (`--embed`, `texture.embed()`).
    Procedural meshes have no UVs: map their textures with `coordinates="object"`.
- Path tracing: `aether3d render PROJECT -o out.png --backend path --spp 32 --bounces 4 --json`
  (CPU), or `--backend gpu_path` for the same path tracer on the GPU (much faster; read
  `backend.adapter`).
  Read `backend.device` (`"cpu"`), `stats.noise.mean_relative_error` (more `--spp` lowers
  it), `stats.rays` and `stats.timings`. It is much slower than the rasterizers: try small
  sizes and few samples first. `--ao-only` is a fast AO preview, `--passes DIR` writes the
  passes (`.pfm`), `-o out.hdr` keeps linear HDR, `--tonemap neutral` tames highlights. Give
  lights a size (`light set --radius/--angle`) for soft shadows and set the sky with
  `scene set --environment sky`. Python: `scene.render(backend="path", path={"samples": 32},
  ao=True)`, then `result.passes`.
- GPU: `aether3d gpu info --check --json` says whether GPU rendering works here and which adapter
  is used (exit code 3 if it does not). Render with `--backend gpu` (fails with a `GPU_*` code if
  unavailable -- never a silent CPU render) or `--backend auto` (read `backend.name` and the
  `BACKEND_FALLBACK` diagnostic). Add `--shadows` for sun shadows; `backend.adapter.software: true`
  means a CPU-emulated GPU (correct, not hardware speed). Python: `scene.render(backend="gpu",
  shadows=True)`, `aether3d.gpu_status()`, `aether3d.list_adapters()`.
