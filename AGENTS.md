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
- Mesh editing: [docs/EDITING.md](docs/EDITING.md) (topology, ids, sessions, operations, scripts);
  advanced modeling and sculpting: [docs/MODELING.md](docs/MODELING.md) (bevels, cuts,
  bridges, sweeps; brushes, dynamic topology, remeshing, symmetrize, validity tiers,
  sculpt scripts).
- UV mapping, texture painting and baking: [docs/TEXTURING.md](docs/TEXTURING.md)
  (unwrapping, projections, island tools, UV diagnostics, MikkTSpace, painted textures,
  brushes, baking, texture transfer, texture scripts).
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
- Spatial queries, tags and metadata: [docs/SPATIAL.md](docs/SPATIAL.md) (conventions,
  semantics of every query, caching, JSON and CLI, errors, limitations).
- Scene plans: [docs/SCENE_PLANS.md](docs/SCENE_PLANS.md) (the format, patterns, operations,
  errors, idempotency, chunks, limits, performance); the MCP server for AI agents:
  [docs/MCP.md](docs/MCP.md) (tools, resources, Claude Code, security).
- Visual feedback and automatic corrections: [docs/FEEDBACK.md](docs/FEEDBACK.md)
  (observations, views, expectations, findings and their basis, vision evaluations,
  corrections, verification, the loop, safety, design, performance).
- Precision modeling and CAD: [docs/CAD.md](docs/CAD.md) (the CAD document, units and
  tolerances, sketches and the solver, features and persistent names, the B-rep kernel,
  booleans, assemblies, measurements, drawings, STEP/DXF/IFC, CAD scripts); buildings and
  BIM: [docs/BUILDINGS.md](docs/BUILDINGS.md); the design record:
  [research/2026-10-stage15/DESIGN.md](research/2026-10-stage15/DESIGN.md).
- Animation, rigging and cinematics: [docs/ANIMATION.md](docs/ANIMATION.md) (authored and
  evaluated state, time, paths, F-curves, rotations, clips and NLA, rigs, skins, IK and
  constraints, generators, contacts, cameras, frames, glTF, scripts); the design record:
  [research/2026-10-stage17/DESIGN.md](research/2026-10-stage17/DESIGN.md).
- Character rigging: [docs/CHARACTER_RIGGING.md](docs/CHARACTER_RIGGING.md) (three
  workflows, the rig document, automatic rigging and landmarks, weights, deformation,
  structural editing, IK/FK and pose commands, diagnostics and verified correction,
  rigid-part characters); the design record:
  [research/2026-10-stage18/DESIGN.md](research/2026-10-stage18/DESIGN.md).
- Procedural worlds: [docs/WORLDS.md](docs/WORLDS.md) (definitions and determinism,
  terrain, hydrology, climate and biomes, features, water, caves, block worlds,
  vegetation, structures, streaming and memory, rendering, queries, files, world scripts,
  limits); the design record:
  [research/2026-10-stage16/DESIGN.md](research/2026-10-stage16/DESIGN.md).
- Optional: `npm ci --prefix tools/gltf-validator` enables the Khronos validator tests locally.
- Performance: [docs/PERFORMANCE.md](docs/PERFORMANCE.md) (how to measure with
  `benchmarks/bench_engine.py`, the correctness guards, implementation switches, the record
  of every phase of the performance modernization).

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
  Run `tests/test_editing_robustness.py` after touching topology code. Building in bulk
  (`MeshCore._bulk_faces`, behind `AETHER3D_EDITING_BULK`) must give the tables adding one
  face at a time gives, in the same insertion and iteration orders, sharing one Python int
  per id (`tests/test_editing_bulk.py` compares them, and their memory).
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
  Rendering must never modify the scene or bake procedural geometry. A `RenderScene` is
  immutable: its arrays may be shared with later frames (the retained render world makes them
  read-only), so backends never write into them.
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
    Hardware ray queries were prototyped outside the package
    (`research/2026-10-modernization/hw_pathtrace/`) and deferred; see
    docs/PERFORMANCE.md (Phase 5) before reopening that question.

- Sculpting (`sculpt/`):
  - **Every change goes through the journal.** Write sculpt mesh state only through the
    journaled primitives (`_move`, `_set_mask`, `apply_topology`, `_replace`) inside a
    `@sculpt_operation`; an operation that raises is rolled back, and `undo()` must give
    back the previous state bit for bit (`tests/test_sculpt_core.py`,
    `tests/test_sculpt_topology.py` compare raw arrays).
  - **Validate before changing.** Check every argument (types, ranges, finiteness) before
    the first write and raise `ValidationError` or `SculptError` with a stable code; the
    sculpt-script fuzz test (`tests/test_sculpt_schema.py`) feeds mutated scripts to the
    executor and fails on any other exception. A new setting needs its checker, the JSON
    Schema (`sculpt/schema.py`) and docs.
  - **Topology changes keep features.** Splits, collapses, flips and relaxation never
    cross boundaries, UV seams, hard edges, face-group borders or masked vertices; new
    vertices interpolate UVs within each face. Check validity (closed, manifold,
    oriented, no self-intersection) after every new topology pass in tests.
  - **Decide sides exactly.** Use `spatial/predicates.py` (`orient3d`,
    `segment_crosses_triangle`) for in/out and crossing decisions; never a float
    determinant near zero.
  - **Measure what you report.** Remeshers and dynamic topology report counts and
    statistics computed on the result; validity tiers are measured (`solid` only with the
    exact self-intersection test). Never call a result valid without checking it.

- UVs, painting and baking (`uv/`, `texturing/`, `materials/painted.py`):
  - **NumPy twins for native kernels.** The sparse LDL^T factorization and the packing
    placement have Rust kernels (`native/src/ldl.rs`, `pack.rs`) and NumPy versions that
    must give the same bits (`tests/test_native.py`); change both or neither, and bump
    `ABI` in `_native.py` and `native/src/lib.rs` when a signature changes.
  - **MikkTSpace is a bit-exact port.** `uv/mikktspace.py` reproduces the reference C
    code in float32 (`tests/data/mikktspace`, regenerated only from the reference harness
    described there); the renderers' tangent convention (interpolated, unnormalized `T`,
    `B = sign x cross(N, T)`) and the baker's exact inverse must change together
    (`tests/test_tangents.py`, `tests/test_bake.py`).
  - **Painted textures change through journaled tile replacements** (`replace_tiles`,
    `replace_masks`, layer methods), each one history step; incremental level-0 and mip
    updates must equal a full rebuild bit for bit (`tests/test_painted.py`). A stroke,
    fill or bake commits once, after computing everything, so a failure changes nothing.
  - **Validate before changing; scripts never touch files.** Texture scripts pass only
    allowlisted arguments to each operation and resolve textures by id or name (never a
    path); the fuzz tests (`tests/test_texture_script.py`) fail on any exception other
    than `TexturingError`. A new operation or argument needs its allowlist entry, the
    schema (`texturing/schema.py`), the guide (`texturing/guide.md`) and docs.
  - **Diagnostics are exact or say what they are.** UV overlaps and flips use exact
    predicates; stacked islands are reported apart from unintended overlaps; bake misses
    and UV overlaps are counted, never hidden.

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

- Spatial queries (`spatial/`):
  - **Read-only.** Queries never change a scene, record history or bump `scene.revision`
    (`test_queries_are_deterministic_and_change_nothing` runs every query). The index is a
    cache kept on the scene (`Scene._derived()`) and keyed by its revision; do not add
    invalidation hooks, add state to the history instead.
  - **Exact or labeled.** Report facts decided on triangles; anything that rests on boxes
    or sampling must say so (`exact=False`, `"bounds"`, `visible_from`'s samples). Never
    call a box overlap a collision.
  - **One ray core.** Rays go through `rendering/raytrace` (`SceneAccel`, the watertight
    test); rendered surfaces through `PathScene.trace`. Use the renderers' world matrices
    (`index.worlds`), so queries, picks and renders agree.
  - **Verify against references.** Check kernels with `tests/test_spatial_kernels.py`
    (independent references in `tests/spatial_helpers.py`) and queries with the brute-force
    tests in `tests/test_spatial_queries.py`; a new query also needs a JSON request
    (`spatial/request.py`), a case in `tests/test_spatial_cli.py` and docs.

- Scene plans (`plans/`):
  - **Plans are data.** A plan is parsed and checked as a whole (`model.py`), bound to the
    scene (`executor.py`: patterns, names, references), then applied; nothing in it is
    ever run as code, and expressions go through the procedural whitelist. Report every
    problem with a stable code, a JSON pointer and a hint, before changing anything.
  - **All or nothing.** A plan is one transaction: a failure, a timeout or a
    cancellation rolls back completely. Only the opt-in `chunk_size` gives up atomicity,
    with a checkpoint and resumable chunks.
  - **One format, three descriptions in step.** A new plan feature needs the checker, the
    JSON Schema (`schema.py`), the guide (`guide.md`) and docs; `tests/test_plans_schema.py`
    fuzzes the checker against the schema and the executor against crashes.
  - **Bulk, but equivalent.** Objects are created with `Scene._create_many` (one history
    step); it must give the same document as the per-object API
    (`test_create_many_gives_the_document_of_the_per_object_api`).
  - **Measure, do not assume.** `benchmarks/bench_plans.py` measures; regression tests
    (`tests/test_plans_performance.py`) compare times on the same machine with generous
    margins, never against fixed budgets. Time them with `tests/timing.py` (the garbage
    collector paused): a full collection walks the whole test session, whose size
    depends on the tests that ran before.
- MCP server (`mcp/`):
  - **Optional and isolated.** Only `aether3d.mcp` imports the MCP SDK; `import aether3d`
    must never load it (`test_importing_aether3d_does_not_load_the_mcp_sdk`).
  - **stdout is the protocol.** Log to stderr only; nothing else may print to stdout.
  - **Every path is untrusted.** Read and write files only through `Workspace.path`
    (containment after resolving links, suffixes, no writes through links, `overwrite`,
    read-only mode). Never open a network connection or run a subprocess from a tool.
  - **Honest replies.** Errors are structured (code, message, hint) and show no paths of
    the machine; a render reports the backend that drew it, never a GPU for a CPU image.
  - Test tools through a real server process (`tests/test_mcp.py` uses the SDK client and
    raw JSON-RPC).

- Precision modeling (`cad/`) and buildings (`arch/`):
  - **Exact or labeled.** Bodies are B-reps on exact surfaces; meshes are derived from
    them within a stated chord and are never the model. Never call a tessellation exact,
    never claim exact surface-surface booleans (Manifold gives the topology, the B-rep is
    reconstructed; traced edges report their deviation in `notes`), and compute mass
    properties and measurements on the exact geometry.
  - **Definitions are data, regeneration is atomic.** An item kind checks its definition
    (`check`) before anything changes, reads only what it declares (`references`, via
    `ctx.inputs`), and raises `CadError` with a stable code and a location
    (`part[Bracket].features[2](h)`). A content key must cover every input of a result
    (definitions, referenced results, parameters, units, tolerances): cached results are
    reused across undo and redo.
  - **No scattered epsilons.** Take tolerances from `cad/tolerance.py`
    (`Tolerance.for_scale`, `rank_threshold`, tessellation tolerances) and say in the
    result which one decided. Lengths in definitions are in the document's unit and
    internally metres; sketch point positions are stored in metres (the solver's state),
    and CAD scripts convert them.
  - **Names, never indices.** Features and mates reference faces and edges by persistent
    names (`cad/naming.py`); a new feature names every face it makes
    (`"<feature>:<role>"`).
  - **Solver changes keep the analysis exact.** DOF, redundancy and conflicts come from
    the numerical rank (`sketch/solver.py`); the block path must agree with the dense one
    (`test_large_sketches_block_analysis_and_incremental_solving_match_the_dense_path`).
    Conflicts name their constraints, and a failed solve changes nothing.
  - **Untrusted files.** STEP, DXF and IFC readers are strict tokenizers with size and
    entity limits that never evaluate anything; unsupported content is refused or
    skipped and reported, never guessed. Open CASCADE, IfcOpenShell and ezdxf are
    validators in the tests only (`AETHER3D_VALIDATOR_PYTHON`): never import them in the
    package.
  - **Buildings check the user's criteria, not codes.** Stair and other checks compare
    against values the building definition gives; never claim building-code compliance.
  - **Verify against closed forms.** Feature, quantity and drawing tests compare with
    analytic values (volumes, areas, distances), not with snapshots of our own output.
  - The ear-clipping triangulation has a Rust twin (`native/src/earcut.rs`): change
    `cad/triangulate.py` and it together, bitwise (`tests/test_native.py`).

- Animation (`animation/`, `gltf/anim_export.py`, `gltf/anim_import.py`):
  - **Evaluation never writes the scene.** Poses, overlays and rigging results are
    computed from the authored scene and a time; only `apply_pose` writes (one undo
    step). Rendering at a time must equal rendering after `apply_pose`
    (`tests/test_animation_system.py`).
  - **Typed paths only.** Channels resolve through the registry in `paths.py`; never look
    up attributes by name. A new animatable property needs its registry entry (kind, unit,
    range), evaluation, the schema, the guide and docs.
  - **Batches equal singles.** `CurveBatch`, `RotationBatch` and `nla.StripTable` give
    exactly the bits one curve, track or strip gives; tests compare them bitwise, and a
    faster path must keep it.
  - **Caches of authored state** (rest values, authored world matrices, scene order) key
    on `evaluate.scene_key(scene)`: the revision, or nothing while an operation is open.
  - **Exact or labelled.** Continuity, extrema and overshoot come from closed forms;
    trajectories, contacts and fitted errors are sampled and say so. Generated motion is
    measured (`foot_contacts`), never assumed.
  - **Scripts are data, atomic.** A new animation-script operation needs its entry in the
    operation table of `script.py` (whitelisted arguments), the schema, `guide.md` and
    docs; a failing script changes nothing (`tests/test_animation_script.py` fuzzes it).
  - **The skinning twin.** `rig.skin_linear` and `native/src/skin.rs` stay bitwise equal;
    change both or neither (and bump `ABI` for a signature change).
  - **glTF: exact or fitted, never silently approximate.** A channel is exported exactly
    only when glTF expresses the same function; otherwise it is fitted within the stated
    tolerance and its error reported. Exports stay Khronos-validator clean (0 errors,
    0 warnings).
  - **No fake cinematography.** No renderer simulates depth of field and nothing claims
    it; motion blur is real temporal sampling; every frame is rendered by Aether3D.

- Character rigging (`animation/autorig.py`, `landmarks.py`, `voxels.py`, `weights.py`,
  `rig_edit.py`, `controls.py`, `limits.py`, `rig_report.py`, `rig_preview.py`,
  `rigid.py`; `native/src/dqs.rs`, `geodesic.rs`):
  - **Extend Stage 17, never fork it.** One `Rig`, one skin representation, one set of
    solvers and timelines: new joint, control or corrective data goes into the immutable
    rig document with its validation (`parse_rig`), `to_json`, a file-format migration if
    needed, glTF handling (exported, kept as extras, or reported) and docs.
  - **Requests are data.** Rigging operations are allowlisted script operations; fix
    proposals coming back from agents are checked call by call (`_check_fix`) before
    anything changes; nothing is evaluated as code.
  - **Twins.** `rig.skin_dual_quaternion` and `native/src/dqs.rs`, `voxels.geodesic` and
    `native/src/geodesic.rs` give the same bits (`tests/test_native.py`); change both or
    neither, and bump `ABI` for a signature change.
  - **Measure and say how.** Landmarks report `confidence` and `method`; diagnostics are
    numeric; a fix is kept only when re-measurement shows it helped and nothing got worse.
    Never claim anatomical correctness, physically based deformation or GPU execution.
  - **Untouched means bit for bit.** Weight edits leave the rows they do not select
    exactly as they were; switching IK/FK must not move any joint
    (`tests/test_rig_controls.py`).
  - **Rigid-part characters stay rigid.** Nothing adds a skeleton to a character unless
    asked (`convert_character`); Stage 17's blocky motions are pinned by
    `tests/data/v0.18.0/nova_motion.json`.

- Procedural worlds (`worlds/`, `rendering/atmosphere.py`, `native/src/hydro.rs`):
  - **Determinism is a contract.** Generation uses integer hashing (`noise.derive`,
    SplitMix64) for every random decision and only correctly rounded operations (`+ - *
    /`, `sqrt`, `floor`, comparisons): no `np.exp`, `np.sin`, `np.cos` or `**` with
    non-integer powers in generation (use `noise.cos_sin_deg` and the curves in
    `noise.py`), and no BLAS -- no `@`, `np.dot`, `einsum` or `tensordot`, whose results
    depend on batch size and CPU; write explicit ordered sums (`noise.blend`). The golden
    hashes of `tests/test_worlds_determinism.py` change only with a deliberate generator
    change (bump `definition["generator"]` and document it).
  - **Pure functions of addresses.** A tile, chunk, plant tile or volume cube depends only
    on the definition, the edits and its address -- never on the camera, generation order,
    batch composition or threads (batched placement equals tile-by-tile placement
    bitwise). Only *selection* depends on the camera.
  - **Authoritative versus drawn.** Queries, structures, paths and exports of exact data
    use the fields (`WorldModel.evaluate`, `Volumes.solid`, blocks with edits); levels of
    detail are for drawing, with measured error. Do not answer a query from a mesh.
  - **The hydrology twins.** `hydrology.py`'s NumPy sweeps and `native/src/hydro.rs`
    stay bitwise equal (`tests/test_native.py`); change both or neither, and bump `ABI`
    for a signature change.
  - **Caches never own authored data.** Generated data lives in byte-budgeted LRU caches
    keyed by content; definitions and edits are immutable and are the only state saved
    and recorded in history.
  - **Definitions and scripts are validated data.** A new definition key needs its
    validation (type, range, size), default, docs and guide; a new world-script operation
    its table entry, schema, guide and docs; a failing script changes nothing
    (`tests/test_worlds_script.py` fuzzes it). No input reaches shader source or code.
  - **Original assets only.** Block textures (`blocks.py`), plants (`assets.py`) and
    structures are drawn procedurally by Aether3D; never add textures, models or names
    from other games.

- Visual feedback (`feedback/`):
  - **Facts and judgments stay apart.** Every `Finding` names its `basis` (`geometry`,
    `render`, `scene`, `heuristic`, `vision`, `agent`) and whether it is `exact`. Never
    give a heuristic or a model's judgment the basis of a measurement, and never let it
    count toward acceptance unless `require_vision` asks for it.
  - **No fake vision.** The engine contains no model and makes no network request. A
    vision evaluation's provenance is *declared* by whoever wrote it; the engine records
    only what it delivered itself (`image_delivered_by_aether3d`). Never write that a
    model saw an image, and give `basis: "vision"` only to findings whose evaluation
    declares `image_input: true`.
  - **Ground before use.** Evaluation findings are checked against their observation's
    id and pixel digest (`OBSERVATION_MISMATCH`), marked stale after the scene changed,
    and grounded in the object index; an `ungrounded` target is never corrected, and
    evaluation-driven proposals are always `approval`.
  - **Corrections are allowlisted actions compiled to plans and scripts.** A new action
    needs its checker (`corrections.py`), its compilation, its policy class, the schema
    (`schema.py`, fuzzed by `tests/test_feedback_schema.py`), the guide (`guide.md`) and
    docs. Never add deletion, mesh editing, file or network access as an action.
  - **Verify every automatic change.** `apply` runs one history transaction, diagnoses
    again and undoes the step (`history.undo()`) on a regression or no improvement; the
    loop stops with an explicit `stop_reason`. Keep previews and diagnoses inside open
    transactions out of every cache.
  - **Observations never change the scene.** Views are virtual cameras, not scene
    objects; observing adds no object, history step or revision. Observation caches are
    keyed by the revision and the full view and render options.
  - Test a planner's correction by applying it and diagnosing again on a real render (the
    tests in `tests/test_feedback_corrections.py`); stub critics only to test the
    provider contract, and label them so.

- Performance work:
  - **Measure first and after** with `benchmarks/bench_engine.py` (`--compare` against a
    saved run on the same machine); never report an improvement that was not measured end
    to end.
  - **Caches derived from a scene live in `Scene._derived()`**, keyed by `scene.revision`
    or the history's own bookkeeping. Never key a module-level dictionary by a scene: a
    cache entry that refers to its scene keeps it alive (`tests/test_scene_lifecycle.py`).
  - **Incremental caches follow the change journal.** `scene.history._changes_since(revision)`
    names the parts changed after a revision (`None`: recompute everything). Every scene
    change must go through the history (`_touch` before writing) or such caches go stale.
    Inside an open operation (`history.in_progress`) changes are not journaled yet: compute
    without the cache and keep nothing. A journaled aspect a cache does not know must make it
    recompute everything (`rendering/retained.py` is the model; its tests compare it with the
    full extraction along random edits, `tests/test_retained_world.py`).
  - **Identical results.** A cached or faster path must give exactly what the path it
    replaces gives (`tests/parity.py`); `tests/test_invariants.py` checks identity, undo,
    redo and cache coherence against a fresh copy of the scene along random edit chains.
  - **Switchable replacements.** A new implementation of something that works keeps the
    old one behind a switch (`aether3d/_features.py`, documented in docs/PERFORMANCE.md)
    with a test comparing the two, until it has proven itself.
  - **Native kernels** (`native/`, the optional `aether3d-native` package; loaded by
    `aether3d/_native.py`):
    - **NumPy stays the reference.** Every kernel has a NumPy implementation in the
      package that runs without it (or with `AETHER3D_NATIVE=0`); `import aether3d` must
      never require the native package.
    - **Bitwise the same.** A kernel performs the NumPy code's operations in its order:
      no fused multiply-adds, no `target-cpu`, no fast-math, no reassociation.
      `tests/test_native.py` compares each kernel with NumPy bit for bit on random and
      adversarial inputs; run the suite with and without the package.
    - **Safe at the boundary.** Check every array (type, contiguity, size) and every
      index a kernel follows and raise `ValueError`, never read out of bounds; release
      the GIL while computing; keep no reference to Python objects.
    - **One calling convention.** Changing a kernel's signature or meaning bumps `ABI` in
      both `native/src/lib.rs` and `_native.py` (a mismatched package is not used).
    - Lint the crate with `cargo fmt --check` and `cargo clippy --release -- -D warnings`
      in `native/` (CI does).

## Using Aether3D from an agent

- **Build with scene plans.** One plan creates or changes many objects in one call and one
  undo step: `scene.execute_plan(plan)`, `aether3d execute-plan PROJECT plan.json`, or
  the `execute_scene_plan` tool of `aether3d mcp serve`. Check first with `aether3d plan
  validate plan.json` (or `mode="dry_run"`), read every issue's `code` and `pointer`, pass
  the revision you read as `expected_revision`, and give retried plans an
  `idempotency_key`. Put the spatial checks in the plan's `queries`.

- **Check what a render shows, then correct it.** State what must hold as an
  `aether3d.expectations` document (`aether3d feedback schema expectations`), then
  `aether3d feedback check PROJECT --expect FILE --json` (exit code 3: not accepted) and
  `aether3d feedback correct PROJECT --expect FILE --preview|--apply` or `feedback loop`;
  over MCP: `observe_scene` (look at the image), `diagnose_scene`, `propose_corrections`,
  `apply_corrections`, `feedback_loop`. Read each finding's `code`, `basis` and
  `magnitude`; `safe` corrections apply automatically, `verify` ones are kept only if a new
  render shows improvement, and `approval` ones wait for `approve=[ids]`. If you evaluate
  an image yourself, send an `aether3d.vision-evaluation` with `image_input: true` only if
  you actually received the image. Python: `FeedbackSession(scene, expectations)`,
  `.diagnose()`, `.propose()`, `.apply()`, `.run()`.

- **Animate with scripts and measure the result.** Write an `aether3d.animation-script`
  (`aether3d animation run PROJECT script.json --json`, or MCP `run_script`; schema:
  `aether3d animation schema`, guide: `aether3d://animation/guide`): keys, curves, clips,
  layers, rigs, skins, IK, generated motions, cameras and shots, as one undo step. Then
  measure instead of guessing: `inspect_animation` / `aether3d animation report` with
  `curve` (discontinuities, overshoot, velocity and jerk), `contacts` (foot sliding and
  penetration) or `diagnose`; fix corners with automatic handles (`auto_clamped` never
  overshoots) and sliding with `plant_feet` or by retiming the whole cycle; render a
  `time` to look (`render_preview` with `time`, if you can see images). Express checks as
  `foot_contacts` and `animation_continuity` expectations; animation corrections need the
  policy's `allow_animation`. Match on error `code` and `location`.
- **Rig characters with scripts, then measure before fixing.** Look first
  (`inspect_rig` `landmarks`, or `aether3d rig landmarks`); rig with an animation script
  (`autorig`, with `parts` for clothes and hard pieces and `landmarks` overrides); pose
  with `pose_character`, switch with `ik_fk`; check with `inspect_rig` `diagnose` and
  previews (`stretch`, `weights`, `skeleton`); apply `fixes` with `apply_rig_fix` and read
  what was `applied` and `reverted`. Joint rotation channels are absolute (rest
  included). Minecraft-style characters need no bones: `generate_rigid`. Guide:
  `aether3d://rigging/guide`.
- **Build worlds with world scripts and query them.** Write an `aether3d.world-script`
  (`aether3d world run PROJECT script.json --json`, MCP `run_script`, or the scene-plan
  operation `world`; schema: `aether3d world schema`, guide: `aether3d world guide` or
  `aether3d://worlds/guide`): create a world from a preset, add features, structures,
  settlements and roads, edit blocks -- one undo step. Ask exact questions before placing
  things (`inspect_world` / `aether3d world sample`: heights, slopes, biomes, water;
  `sites`, `route`), check `info()`'s cost estimates before large regions, and keep
  cameras above the ground with `ground_path(clearance=...)`. Renders include worlds
  (`render_preview` frames them by their ground).
- Prefer ids (`obj.id`) over names for durable references.
- Every change can be undone: `scene.history.undo()`, `with scene.transaction("label"):`
  for several changes as one step (all or nothing), `scene.history.checkpoint(name)` and
  `goto(name)` to try something and come back. On the command line, `aether3d history
  start PROJECT` makes a project keep a history file, then `aether3d history undo
  PROJECT` and `show`, `redo`, `checkpoint`, `goto` and `verify` work across commands.
- Ask where things are instead of guessing from transforms: `scene.raycast`,
  `closest_point`, `nearest_objects`, `contains_point`, `overlaps`, `query_region`,
  `pick`, `line_of_sight`, `relation`, or `aether3d query QUERY PROJECT KEY=VALUE --json`
  (`aether3d query describe PROJECT --json` gives a compact summary of every object).
  Tag objects (`--add-tag`, `obj.tags`) and find them with `find_objects` / `query find`.
  Match on `QueryError.code`; `contains_point` says `"undefined"` for open meshes.
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

- Sculpt with a session (`with obj.sculpt() as s: s.stroke({...})`) or a sculpt script
  (`aether3d sculpt run PROJECT script.json`, MCP `run_script` with format
  `aether3d.sculpt-script`): brush radius and dynamic-topology detail are world units,
  strokes snap to the surface by default, `"symmetry": "x"` mirrors across the object's
  local plane. Give each sculpt a `"require"` tier (`closed`, or `solid` for the exact
  self-intersection test) so a broken result is refused instead of committed; run
  `voxel_remesh` then `remesh` for an even base before fine strokes. Measure with
  `aether3d sculpt check --require TIER` or MCP `inspect_mesh`; look with
  `observe_scene` and check with a `mesh_validity` expectation. Read the guide
  `aether3d://sculpt/guide` and match on error `code` and `location`.
- Texture with `obj.unwrap()`, `obj.paint(...)`, `scene.bake(...)` (Python) or a texture
  script (`aether3d texture run PROJECT script.json`, MCP `run_script` with format
  `aether3d.texture-script`). Unwrap first and check with `aether3d texture report
  --require-clean` or MCP `inspect_uvs` (`image: "overlaps"`): no flipped or overlapping
  triangles. Paint with world-space points and world-unit radii (like sculpt strokes);
  bind with `"factor": "#ffffff"` so painted colors show as painted; bake with
  `apply: true` and read `miss_fraction`. Check with a `uv_quality` expectation; read the
  guide `aether3d://texturing/guide` and match on error `code` and `location`.
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
