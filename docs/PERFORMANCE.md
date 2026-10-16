# Performance

How Aether3D is measured, what guards its correctness while it gets faster, and the
record of the performance modernization (an architecture-first plan from the
[October 2026 audit](../research/2026-10-architecture-audit/REPORT.md): fix the data
layout and recomputation first, then a short list of native kernels; hardware ray
tracing as a separate decision).

- [Measuring](#measuring)
- [Correctness guards](#correctness-guards)
- [Implementation switches](#implementation-switches)
- [Phase 0: foundation and baseline](#phase-0-foundation-and-baseline)
- [Phase 1: quick wins in the current architecture](#phase-1-quick-wins-in-the-current-architecture)
- [Phase 2: the retained render world](#phase-2-the-retained-render-world)
- [Phase 3: native kernels (Rust)](#phase-3-native-kernels-rust)
- [Phase 4: mesh editing](#phase-4-mesh-editing)
- [Phase 5: hardware ray tracing (prototype, deferred)](#phase-5-hardware-ray-tracing-prototype-deferred)
- [Visual feedback (0.12.0)](#visual-feedback-0120)

## Measuring

`benchmarks/bench_engine.py` measures the engine-wide workloads, each group in its
own Python process (caches, compiled GPU pipelines and memory peaks stay separate;
every group reports its process's peak RSS):

```sh
python benchmarks/bench_engine.py --label NAME --json benchmarks/results/NAME.json
python benchmarks/bench_engine.py --only frame,extract --compare benchmarks/results/phase0-baseline.json
python benchmarks/bench_engine.py --table benchmarks/results/*.json --metrics KEY,KEY
python benchmarks/bench_engine.py --profile extract --quick      # cProfile one group
```

| Group | What it times (10,000 objects unless stated) |
| --- | --- |
| `core` | one plan creating 10,000 cubes; per-object `add_object`; position edits alone and in a transaction; reading `position` and `world_matrix`; `world_matrices`; undo/redo of the plan; memory per object |
| `queries` | the first raycast, a warm one, the first after an edit; the first and a warm `overlaps`; a warm `nearest_objects` |
| `extract` | the CPU side of a frame without a GPU: `extract_render_scene` when nothing, one object, all objects or the camera changed |
| `frame` | complete GPU frames at 1280x720 for the same four cases: wall clock, every CPU stage of `stats["timings"]`, and the GPU's own pass times from timestamp queries (`stats["gpu_timings"]`) |
| `raytrace` | CPU closest-hit and any-hit throughput (100k-triangle sphere, camera and diffuse rays); BVH build |
| `editing` | `EditableMesh.from_mesh` (with attributes and quads; topology only, 200k triangles); extrude, inset, bevel (one edge, every edge), subdivide, `to_mesh` |
| `images` | decoding and encoding (zlib levels 6 and 1) a 1080p PNG render |
| `io` | saving, loading and glTF export |
| `memory` | 100,000 objects: plan, first raycast, RSS |
| `lifecycle` | 12 cycles of create, query, render and drop: scenes left alive, RSS growth |

Method: medians after warm-up; Python's cyclic garbage collector is paused during each
timed run (as `timeit` does: a full collection walks the whole process); the frame
scenarios apply their change outside the timed region. The GPU frame's stages are the
renderer's own timings: `extract`, `textures` (material resolution), `upload` (packing
and buffer creation), `prepare` (command encoding), `gpu` (submission until the outputs
can be mapped, which includes synchronization and copying them to the readable
buffer), `readback` (decoding into NumPy arrays); `gpu_passes` are the GPU's execution
times of the shadow, scene and finalize passes, measured on the GPU.

`benchmarks/bench_feedback.py` measures the visual feedback workflow
([FEEDBACK.md](FEEDBACK.md#performance)): observations on both rasterizers (cold, after
one move, reused; the retained world against the full extraction), the checks of a
diagnosis, previews, verified corrections and whole loops (`--quick`; `--smoke` only
checks that it runs).

Compare runs on the same machine only. The saved runs in `benchmarks/results/` come
from the development machine: Intel Core i9-12900H (20 threads), NVIDIA GeForce RTX
4060 Laptop GPU (Vulkan, driver 615.78.08), Arch Linux (kernel 7.2.9), CPython 3.14.8,
NumPy 2.5.4, wgpu-py 0.32.0 (wgpu-native 29).

## Correctness guards

Getting faster must never change what Aether3D computes or keep memory alive.

- **Scene lifetime** (`tests/test_scene_lifecycle.py`): a scene is freed after every
  kind of use -- each spatial query, plans with idempotency keys and queries, CPU and
  GPU rendering and path tracing, editing, procedural geometry, history, saving,
  loading, glTF. Caches derived from a scene live in `Scene._derived()`, never in a
  module-level dictionary keyed by the scene (see Phase 0).
- **Identity, exact undo/redo and coherent caches** (`tests/test_invariants.py`): along
  random chains of changes (objects, hierarchy, transforms, materials, procedural
  parameters, plans, mesh edits), undoing to any state gives the same document and the
  same Python objects, redoing too; and every cached answer (spatial queries, frame
  extraction) equals the answer of a fresh copy of the scene rebuilt from its
  document, after every change, undo and redo.
- **Determinism**: the renderer's world matrices equal a per-object reference with the
  same arithmetic bit for bit, also after reparenting; extracting a frame twice gives
  identical render scenes. `tests/parity.py` compares render scenes, arrays and query
  answers exactly (bytes, ids, order, diagnostics) and reports the first difference.
- **Retained state equals recomputed state** (Phase 2): `tests/test_retained_world.py`
  compares the retained render world's render scenes, world matrices and framing points
  with the full computations after every change of random chains (and errors with the
  full extraction's errors); `retained.STRICT` makes its unexpected errors fail tests
  instead of falling back. GPU frames run with `VERIFY_KEPT`: everything a frame kept or
  updated from earlier frames is compared with a computation from scratch, bit for bit,
  and `tests/test_gpu_kept_frames.py` compares frame sequences with renders made without
  the retained world.
- **The existing suites**: render goldens (`tests/data/renders`), CPU/GPU parity,
  path-tracer bit-for-bit tests, the history catalog, the spatial brute-force
  references and the editing robustness tests run unchanged.

Two world-matrix arithmetics exist and both are kept: the renderers, spatial queries
and picks use `world_matrices` (top-down, `(root @ parent) @ local`, explicit
fixed-order sums), while the public `SceneObject.world_matrix` multiplies with `@`
from the object up (`root @ (parent @ local)`). For nested objects they can differ in
the last bits; neither may change silently.

## Implementation switches

A new implementation that replaces a working one stays switchable until it has proven
itself: tests run and compare both, and anyone can switch back without a release.
Switches are environment variables registered in `aether3d/_features.py`
(`1`/`on`/`true`/`yes`, `0`/`off`/`false`/`no`; unset means the default; any other value
is an error naming the variable). Benchmark records include their state.

| Variable | Default | Selects |
| --- | --- | --- |
| `AETHER3D_MATERIAL_CACHE` | on | reuse materials resolved for earlier renders of a scene (`rendering/materials.py`, `MaterialCache`); `0` resolves every material for every render |
| `AETHER3D_GPU_PERSISTENT_BUFFERS` | on | keep the GPU's per-frame data buffers between renders and write only what changed (`rendering/gpu/resources.py`, `FrameBuffers`); `0` creates fresh buffers for every render |
| `AETHER3D_RETAINED_RENDER_WORLD` | on | keep what extraction computed between renders and update it from the history's change journal (`rendering/retained.py`, `RenderWorld`; it needs the material cache); `0` walks the whole scene for every render |
| `AETHER3D_NATIVE` | on | use the native kernels of `aether3d-native` when installed (`aether3d/_native.py`: ray traversal, transform hierarchies, PNG filters); `0` uses the NumPy implementations, which give the same bits |
| `AETHER3D_NATIVE_THREADS` | `1` | threads of the native ray traversal (`0`: one per CPU); results do not depend on it (a count, not an on/off switch) |
| `AETHER3D_EDITING_BULK` | on | build an `EditableMesh`'s faces, edges and corner layers from a `Mesh` in bulk with NumPy (`editing/core.py`, `MeshCore._bulk_faces`); `0` adds one face at a time |

## Phase 0: foundation and baseline

**Spatial queries kept their scenes alive (fixed).** The spatial index was cached in a
module-level `WeakKeyDictionary` keyed by the scene; its values are strong references
and the index refers to its scene, so every scene that ever answered a query was never
freed -- with its index, world matrices and BVHs. Measured: 20 scenes of 1,000 objects
that ran two queries each stayed resident (+80 MiB RSS). Scenes now hold the caches
derived from them (`Scene._derived()`); the two other weak dictionaries keyed by
scenes (plan idempotency records, the CLI's history notes) hold plain data and were
checked to free their scenes. On the old code 24 of the 31 lifecycle cases fail.

**Baseline** (`benchmarks/results/phase0-baseline.json`, after the leak fix):

| metric | phase 0 |
| --- | ---: |
| `core.plan_create_10k_ms` | 89.1 |
| `core.set_position_in_transaction_us` | 3.132 |
| `core.read_world_matrix_us` | 11.45 |
| `core.world_matrices_10k_ms` | 8.046 |
| `core.bytes_per_object_with_history` | 935 |
| `queries.raycast_warm_ms` | 7.385 |
| `queries.raycast_after_edit_ms` | 242.7 |
| `queries.overlaps_first_ms` | 49.3 |
| `extract.extract_unchanged_ms` | 69.08 |
| `extract.extract_one_moved_ms` | 69.65 |
| `extract.extract_all_moved_ms` | 72.34 |
| `frame.unchanged_wall_ms` | 115.5 |
| `frame.one_moved_wall_ms` | 113.4 |
| `frame.all_moved_wall_ms` | 116.2 |
| `frame.camera_moved_wall_ms` | 117.3 |
| `frame.unchanged_stages_ms.extract` | 49.42 |
| `frame.unchanged_stages_ms.textures` | 16.53 |
| `frame.unchanged_stages_ms.upload` | 27.93 |
| `frame.unchanged_stages_ms.gpu` | 4.357 |
| `frame.unchanged_stages_ms.readback` | 8.009 |
| `frame.unchanged_gpu_passes_ms.scene` | 1.548 |
| `frame.unchanged_gpu_passes_ms.finalize` | 0.224 |
| `raytrace.closest_camera_mrays_s` | 0.308 |
| `raytrace.closest_bounce_mrays_s` | 0.107 |
| `raytrace.blas_build_100k_tris_ms` | 267.9 |
| `editing.from_mesh_attributes_s` | 0.844 |
| `editing.from_mesh_topology_s` | 0.702 |
| `editing.bevel_all_edges_ms` | 660.7 |
| `editing.inset_500_faces_ms` | 125.2 |
| `images.png_decode_1080p_ms` | 311.5 |
| `images.png_encode_1080p_level6_ms` | 595.3 |
| `io.save_10k_s` | 0.4624 |
| `io.load_10k_s` | 0.4525 |
| `memory.first_raycast_s` | 2.68 |
| `memory.peak_rss_mib` | 311.2 |
| `lifecycle.scenes_still_alive` | 0 |

What it shows:

- **A frame is an O(N) rebuild.** A 10,000-object GPU frame takes 113-117 ms whether
  nothing, one object, every object or the camera changed. The GPU executes its passes
  in 1.8 ms; the rest is CPU work repeated every frame: extraction (49 ms), material
  resolution (17 ms), packing and creating buffers (28 ms), synchronization and the
  output copy (4 ms), reading 28 bytes per pixel back (8 ms).
- **Any edit rebuilds the whole spatial index**: the first raycast after moving one
  object takes 243 ms at 10,000 objects (2.7 s at 100,000); warm queries take 7 ms, most
  of it a per-query pass over all objects.
- **Scalar kernels in NumPy**: CPU ray tracing at 0.1-0.3 M rays/s, a 100k-triangle BVH
  in 268 ms, PNG decoding at 6.7 MP/s.
- **Editing is per-element Python**: bevelling every edge of a 2,048-face sphere takes
  0.66 s.

## Phase 1: quick wins in the current architecture

Optimizations inside the existing Python/NumPy/wgpu architecture. Every one gives
exactly the results of the code it replaces; the replaced loops are kept as references
in `tests/test_vectorized_parity.py`, and the caches are compared with fresh
computations in `tests/test_invariants.py`, `tests/test_render_caches.py` and
`tests/test_gpu_frame_buffers.py`.

**What changed**

- **Spatial index** (`spatial/index.py`, `rendering/raytrace/accel.py`). Entries compute
  `singular` once (it was recomputed for every entry on every query); the acceleration
  structure cache is keyed by the selection itself (a tuple built and hashed in C); an
  index reuses, from the previous revision's index, every entry whose object, mesh,
  visibility and world matrix are unchanged, with what it built (world vertices,
  per-object BVHs, boxes) and the per-mesh topology facts; determinants, inverses and
  instance bounds are computed for all objects at once with exactly the per-object
  LAPACK/BLAS operations (bitwise, checked against the old loop on random transforms,
  including singular, tiny and non-finite ones).
- **Materials between renders** (`rendering/materials.py`, `MaterialCache`): a library
  material's resolution is kept for one `MaterialLibrary.version` and while the images it
  read are unchanged (every render reads them again, so files changed on disk are used);
  its diagnostics are reported again at its first use in each render; an object's own
  color is keyed by the exact bytes of its values (`-0.0` and `0.0` stay distinct, as their
  `repr` keys were). Switch: `AETHER3D_MATERIAL_CACHE`.
- **Extraction** checks all world matrices for finiteness at once and builds an object's
  location text only when it reports something; the render scene carries its instances
  as arrays (`RenderScene.arrays`, `InstanceArrays`: row `k` is `instances[k]`, exactly).
- **GPU frame preparation** reads those arrays: the draw order is one `lexsort`, instance
  records, material tables and instance corners are indexed, `triangle_count` is summed by
  index. The float32-range check that skips objects too far from the camera first bounds
  every instance's corners (`|translation - eye| + |linear| @ max(|lo|, |hi|)`, with a margin
  far above rounding); when every bound is in range the exact per-corner test would keep
  everything and is skipped, otherwise it runs as before.
- **Persistent GPU buffers** (`rendering/gpu/resources.py`, `FrameBuffers`): instance
  records, material tables, lights, shadow maps and uniforms live in buffers kept between
  renders; each render's data is compared with the buffer's copy and only changed ranges
  are written, compared per 32-bit word (merged when closer than 256 bytes, at most 32
  writes per buffer). An unchanged frame writes nothing; moving one object along one axis
  wrote one 4-byte word (2,000 objects, measured with the `frame_buffer_writes` and
  `frame_buffer_bytes_written` statistics). Switch: `AETHER3D_GPU_PERSISTENT_BUFFERS`.
- **Selective readback**: `render(outputs=...)` returns only some per-pixel outputs; the
  GPU then copies and reads back only those (4 bytes per pixel for the image alone,
  instead of 28). The non-finite check (`RENDER_FAILED`) and the covered-pixel count
  that needed the full readback are counted on the GPU by the finalize pass (the
  non-finite test reads the float bits, which shader compilers cannot fold). MCP
  previews read back only the image; `render_image` and the CLI's `render` read back what
  the file needs.
- **Images**: MCP previews and inline MCP images are encoded at zlib level 1
  (`RenderResult.png_bytes(compression=...)`; the default 6 and every written file are
  unchanged); choosing each PNG row's filter works in 16-bit integers with a lookup
  table, the same filters and bytes 1.8x faster.
- **Scene plans** draw new object ids from one read of the system's random source
  (`new_object_ids`, `str(uuid.uuid4())`'s format) and check all records' transforms at
  once with the per-record tests (per-record checks still run when a batch fails, for the
  same first error).

**Measured** (`benchmarks/results/phase1.json`, same machine and method as Phase 0):

| metric | phase 0 | phase 1 |
| --- | ---: | ---: |
| `frame.unchanged_wall_ms` | 115.5 | 71.78 |
| `frame.one_moved_wall_ms` | 113.4 | 71.88 |
| `frame.all_moved_wall_ms` | 116.2 | 75.54 |
| `frame.camera_moved_wall_ms` | 117.3 | 75.12 |
| `frame.unchanged_image_only_wall_ms` |  | 67.71 |
| `frame.unchanged_stages_ms.extract` | 49.42 | 41.25 |
| `frame.unchanged_stages_ms.textures` | 16.53 | 11.22 |
| `frame.unchanged_stages_ms.upload` | 27.93 | 2.46 |
| `frame.unchanged_stages_ms.gpu` | 4.357 | 4.506 |
| `frame.unchanged_stages_ms.readback` | 8.009 | 6.064 |
| `frame.unchanged_image_only_stages_ms.gpu` |  | 2.454 |
| `frame.unchanged_image_only_stages_ms.readback` |  | 2.616 |
| `frame.unchanged_gpu_passes_ms.scene` | 1.548 | 1.552 |
| `frame.unchanged_gpu_passes_ms.finalize` | 0.224 | 0.224 |
| `frame.preview_640x360_png_ms` |  | 76.56 |
| `extract.extract_unchanged_ms` | 69.08 | 54.49 |
| `queries.raycast_first_ms` | 245.6 | 63.27 |
| `queries.raycast_warm_ms` | 7.385 | 2.778 |
| `queries.raycast_after_edit_ms` | 242.7 | 59.35 |
| `queries.overlaps_first_ms` | 49.3 | 26.83 |
| `memory.first_raycast_s` | 2.68 | 0.805 |
| `memory.peak_rss_mib` | 311.2 | 374.9 |
| `core.plan_create_10k_ms` | 89.1 | 72.88 |
| `images.png_encode_1080p_level6_ms` | 595.3 | 484.8 |
| `images.png_encode_1080p_level1_ms` | 309.4 | 173.1 |
| `lifecycle.scenes_still_alive` | 0 | 0 |

- A 10,000-object GPU frame takes 72-76 ms instead of 113-117 (68 ms reading back only
  the image). The GPU's own pass times are unchanged (1.8 ms): the gain is CPU work.
  Uploads went from 28 ms to 2.5; the synchronized `gpu` stage and the readback halve when
  only the image is read back.
- What remains is still O(N) per frame: extraction (41 ms) and material resolution
  (11 ms) visit every object whatever changed. That is Phase 2's subject.
- The first spatial query after an edit takes 59 ms instead of 243; at 100,000 objects
  0.8 s instead of 2.7. A warm query takes 2.8 ms instead of 7.4.
- Peak memory of the 100,000-object case rose from 311 to 375 MiB, with the vectorized
  spatial build, which holds arrays for all objects at once: the first raycast's own
  allocation peak (tracemalloc) is 169 MiB instead of 115.
- Extraction also builds the instance arrays now, which costs 3-5 ms at 10,000 objects
  (alternating runs of the material cache commit and of this phase's last one: 45-48 ms
  against 50-51 ms for `extract.extract_unchanged_ms`); the GPU's packing reads them
  instead of visiting instances, which took the upload stage from 28 ms to 10.6 (and the
  persistent buffers, from 10.6 to 2.5).

## Phase 2: the retained render world

Renders no longer visit every object of the scene: what a frame computes is kept with the
scene and brought up to date with what changed. Every retained result equals the full
computation it replaces (see [Correctness guards](#correctness-guards)).

**What changed**

- **The change journal** (`history.py`, `History._changes_since(revision)`): which parts
  of the scene (aspect, target) each revision changed -- commits, undo and redo -- for
  the last 256 revisions and at most 65,536 parts; it starts when a cache first asks and
  never keeps the states themselves. A cache asks what changed since the revision it was
  computed at; `None` (the journal does not reach back) means recompute everything.
- **The retained render world** (`rendering/retained.py`, `RenderWorld`; switch
  `AETHER3D_RETAINED_RENDER_WORLD`): the objects in depth-first order as rows, with
  NumPy columns for local and world matrices, finiteness, normal matrices and
  determinants, colors and material keys, and per row the evaluated mesh and the render
  instance. A transform recomputes its subtree's world matrices level by level with the
  full extraction's arithmetic; an edited object is classified again; a change of
  structure orders the objects again and keeps what each unchanged one had (world
  matrices compared bit for bit); procedural and geometry changes evaluate procedural
  objects again; library materials are resolved for every render through the material
  cache, which reads their images again. The mesh and material tables keep their order of
  first use. Inside open operations, with the material cache off, under contention and
  when a render fails, the full extraction runs (it reports the errors).
- **Kept GPU frame data** (`RenderScene.shared_cache()`, `rendering/gpu/renderer.py`):
  render scenes of a retained world share one dict while their instances, meshes and
  materials are the same objects; the GPU backend keeps there what it computes from them
  -- draw order, instance records, material tables, the float32 range check -- and, when
  only some instances' transforms or colors changed, updates it: their records again
  (one moved object: one record), and only the camera-relative translations when the
  camera moved (with packing's float64 subtraction and float32 rounding). Read-only kept
  arrays are uploaded without comparison. `RenderResult.objects` rows are kept too.
- **Kept world matrices and framing points**: `world_matrices(scene)` (spatial index,
  framing, scripts, framed renders) and `framing.target_points(scene)` come from the
  world of a scene that was rendered, up to date; each object's framing points (its
  world-space vertices) are recomputed only when its mesh or matrix changes, within 64 MiB
  of points per scene (`FRAME_CACHE_BYTES`; larger scenes compute them for every framed
  render, as before). Scenes never rendered compute everything as before.
- **Bulk changes**: instances' matrices are views into one read-only block per build
  (the world compacts when blocks hold more than twice the rows in use, plus 1,024);
  structural changes order objects in one pass and match them to their old rows through
  sorted id arrays; `_local_matrices` gathers transforms with C-level iteration.
- **A change of the sign of a zero is a change** (`history._same`): the history recorded
  nothing when a value went from `0.0` to `-0.0` (they compare equal), so undo could not
  restore it and every cache keyed by the revision kept the other zero (the spatial index
  since 0.10, the render world now). Transforms compare their ten floats as bytes.

**Measured** (`benchmarks/results/phase2.json`; 10,000 objects; GPU frames at 1280x720 on
the RTX 4060; median wall times):

| metric | phase 0 | phase 1 | phase 2 |
| --- | ---: | ---: | ---: |
| `frame.unchanged_image_only_wall_ms` |  | 67.71 | 7.906 |
| `frame.one_moved_image_only_wall_ms` |  | 68.17 | 10.29 |
| `frame.unchanged_wall_ms` | 115.5 | 71.78 | 15.88 |
| `frame.one_moved_wall_ms` | 113.4 | 71.88 | 17.61 |
| `frame.camera_moved_wall_ms` | 117.3 | 75.12 | 15.47 |
| `frame.one_added_wall_ms` |  | (74.77) | 26.52 |
| `frame.all_moved_wall_ms` | 116.2 | 75.54 | 43.66 |
| `frame.preview_640x360_png_ms` |  | 76.56 | 17.72 |
| `frame.preview_framed_unchanged_640x360_png_ms` |  | (183.3) | 46.43 |
| `frame.preview_framed_one_moved_640x360_png_ms` |  | (184.1) | 58.62 |
| `extract.extract_unchanged_ms` | 69.08 | 54.49 | 0.21 |
| `extract.extract_one_moved_ms` | 69.65 | 54.04 | 0.659 |
| `extract.extract_camera_moved_ms` | 71.94 | 55.48 | 0.285 |
| `extract.extract_one_added_ms` |  | (55.28) | 8.692 |
| `extract.extract_all_moved_ms` | 72.34 | 56.35 | 25.01 |
| `extract.extract_all_moved_full_ms` |  |  | 53.73 |
| `extract.retained_bytes_per_object` |  |  | 1721 |
| `core.plan_create_per_object_us` | 8.91 | 7.288 | 8.114 |
| `core.set_position_in_transaction_us` | 3.132 | 3.035 | 3.853 |
| `queries.raycast_after_edit_ms` | 242.7 | 59.35 | 59.91 |
| `lifecycle.scenes_still_alive` | 0 | 0 | 0 |

Values in parentheses are new metrics measured on the Phase 1 commit with the Phase 2
benchmark script, the same day. `*_full_ms` is the full extraction, which still runs
when the retained world cannot.

- **GPU frames.** With nothing changed, a frame takes 7.9 ms reading back only the image
  (15.9 ms reading back everything): the GPU stage (2.2-4.1 ms) and the readback
  (2.8 ms for the image, 6.6-8.9 ms for everything) are most of it. One moved object:
  10.3 ms (17.6 ms). The audit estimated about 6 ms for a frame with one changed object
  and image-only readback (section 8 of its report); measured, it is 10.3 ms -- the
  extraction (1.1 ms), uploads (0.9 ms), the result's per-object list and Python
  overhead account for the difference.
- **Previews.** An MCP-sized preview (640x360, image only, PNG at zlib level 1) takes
  17.7 ms instead of 76.6; framed on the geometry (`frame=True`, previews of scenes
  without a camera), 46.4 ms instead of 183.3. What remains there is the exact camera
  fit over every vertex (240,000 points) and the PNG encoding.
- **Costs.** A rendered scene keeps 1,721 bytes per object for its world (17 MB at
  10,000 objects, measured with tracemalloc). Setting a transform to the value it
  already has costs 3.8-4.0 us instead of 3.3-3.4 (the exact comparison; alternating
  runs of the Phase 1 commit and this code); real changes cost the same. After the first
  cycle of the lifecycle benchmark, RSS is about 4 MiB higher than with Phase 1 (code and
  caches loaded once) and does not grow over 36 cycles; no scene stays alive.
- **Drift.** The run was made on a laptop running slower than for the Phase 1 record:
  groups whose code did not change measure 1-14 % slower (ray tracing 10-14 %, editing
  2-10 %, images 2-6 %, I/O 1-4 %). Plan creation (8.1 against 7.3 us per object) and the first
  queries match the Phase 1 commit run alternately with this code on the same machine
  state (76-81 against 79-81 ms per 10,000-object plan).

**Targets of the audit not met** (its phase plan gave estimates, marked as such):

- *All objects moved within 25 ms*: 43.7 ms (extraction 25 ms). Every moved object gets
  a new `RenderInstance` -- the render scene `extract_render_scene` returns (public API)
  is a tuple of instance objects -- and a normal matrix through LAPACK, bitwise the full
  extraction's and the goldens': about 11 ms and 4 ms of those 25 at 10,000 objects.
  Instances as lazy views over the columns would meet the estimate but change what the
  public function returns.
- *Creation within 5 us per object*: 8.1 us per object for plans. The estimate assumed
  transforms stored in columns with column deltas in the history. Authoring stays
  object-based: `SceneObject` and `Transform` handles, undo and redo restoring the same
  Python objects, and per-object history aspects are the scene model's contract, and the
  render side's columns already remove the per-frame cost that columnar storage aimed at.
- *Camera rebase on the GPU*: investigated and declined, see
  [research/2026-10-modernization](../research/2026-10-modernization/README.md): it
  would change 0.23 % of instance translations in their last bit, with twice the CPU
  rebase's worst error, to save about 1 ms per camera move.
- *The spatial index as a journal consumer, with a refit of its top level*: not done.
  The first query after an edit takes 59.9 ms at 10,000 objects, about half to index the
  objects again (vectorized since Phase 1) and half to build the top-level BVH over all
  of them; a refit would leave the query results unchanged (they are ordered by distance,
  scene order and triangle, not by traversal) and is the next step for edit-and-query
  loops.
- *Readback of every output*: 6.6-8.9 ms, varying between runs and frame kinds on this
  laptop (the image alone: a steady 2.8 ms); nothing in this phase changed it.

## Phase 3: native kernels (Rust)

One Rust crate, `native/` (`aether3d-native`, PyO3 0.30 and maturin, Apache-2.0; PyO3 is
MIT or Apache-2.0), implements the kernels the audit found native code helps most, with
exactly the arithmetic of their NumPy implementations. Aether3D uses it through
`aether3d/_native.py` when it is installed; otherwise, when it cannot be loaded or speaks
another calling convention (`ABI`), and with `AETHER3D_NATIVE=0`, the NumPy code runs --
the default for the 17 portable CI jobs, which do not install it.

**The kernels** (in the audit's order of priority)

- **Ray traversal** (`SceneAccel.closest_hit`, `occluded`: the CPU path tracer and the
  spatial queries): `SceneAccel._trace`'s state machine one ray at a time, with the same
  slab tests, instance spaces, watertight triangle test, visiting order and choice of
  hits; the hits and the traversal statistics are bitwise NumPy's. `AETHER3D_NATIVE_THREADS`
  (default 1; 0: one per CPU) traces with threads -- results do not depend on it; the
  path tracer already runs several processes on Linux.
- **Transform hierarchies**: local matrices from transforms (`_local_matrices`) and the
  top-down world stack (`_world_stack`, the retained world's rebuilds), with
  `_mat4_mul`'s fixed sums.
- **PNG row filters**: the encoder's choice of filter per row and the decoder's
  reconstruction (all five filters).

**Exactness.** Rust never contracts `a * b + c` into a fused multiply-add on its own, and
nothing in the crate asks it to (no `target-cpu`, no fast-math): every kernel performs
the NumPy code's operations in its order. `tests/test_native.py` compares each kernel with
NumPy bit for bit on random and adversarial inputs (axis-aligned and non-finite rays,
mirrored, singular and nested instances, skip pairs, signed zeros and extreme scales,
every PNG filter and pixel size), checks that threads change nothing, and the whole suite
-- render goldens, the path tracer's bit-for-bit images -- runs with the kernels in the
`native` CI jobs (Linux, macOS, Windows).

**Safety.** Every function checks its arrays (element types, contiguity, sizes, every
index the traversal follows, outputs sharing memory with another array) and raises
`ValueError` instead of reading out of bounds; panics become Python exceptions. The GIL is
released while a kernel runs (tested for the traversal), and no reference or memory is
kept (tested).

**Packaging.** A separate distribution: Aether3D stays pure Python (hatchling) and
installs everywhere; `pip install ./native` builds the kernels with a Rust toolchain
(1.85 or later). Its wheels use CPython's stable ABI from 3.11 (one wheel per platform);
on Python 3.10 the NumPy code runs. CI builds it on Linux, macOS and Windows and lints it
(rustfmt, clippy with warnings as errors).

**Not done.** Embree, an optional accelerator in the audit's plan, is not integrated: its
float32 traversal would not give the exact hits the CPU path tracer's bit-for-bit images
and the spatial queries rely on, and the exact native traversal already traces several
times faster.

**Measured** (`benchmarks/results/phase3.json`, the same machine; `*_numpy` is the same
run with `AETHER3D_NATIVE=0`; `*_8_threads` with `AETHER3D_NATIVE_THREADS=8`):

| metric | phase 2 | phase 3 |
| --- | ---: | ---: |
| `raytrace.closest_camera_numpy_mrays_s` |  | 0.305 |
| `raytrace.closest_camera_mrays_s` | 0.293 | 3.049 |
| `raytrace.closest_camera_8_threads_mrays_s` |  | 12.12 |
| `raytrace.closest_bounce_numpy_mrays_s` |  | 0.104 |
| `raytrace.closest_bounce_mrays_s` | 0.102 | 0.757 |
| `raytrace.closest_bounce_8_threads_mrays_s` |  | 4.084 |
| `raytrace.any_bounce_numpy_mrays_s` |  | 0.093 |
| `raytrace.any_bounce_mrays_s` | 0.102 | 0.804 |
| `raytrace.any_bounce_8_threads_mrays_s` |  | 3.964 |
| `images.png_decode_1080p_numpy_ms` |  | 324.4 |
| `images.png_decode_1080p_ms` | 327.7 | 44.66 |
| `images.png_encode_1080p_level6_numpy_ms` |  | 535.8 |
| `images.png_encode_1080p_level6_ms` | 515.6 | 433.2 |
| `images.png_encode_1080p_level1_numpy_ms` |  | 189.9 |
| `images.png_encode_1080p_level1_ms` | 182.7 | 104 |
| `core.world_matrices_10k_numpy_ms` |  | 6.646 |
| `core.world_matrices_10k_ms` | 6.855 | 6.427 |
| `queries.raycast_warm_ms` | 2.98 | 1.42 |
| `frame.preview_640x360_png_ms` | 17.72 | 10.91 |
| `frame.preview_framed_unchanged_640x360_png_ms` | 46.43 | 38.84 |
| `lifecycle.scenes_still_alive` | 0 | 0 |

- **Traversal.** On the 100,000-triangle scene of the raytrace group, one thread traces
  camera rays 10.0 times faster than NumPy and bounce rays 7.3 (closest hit) and 8.6
  (any hit) times faster; eight threads, 39-43 times. The audit measured 9-22 times per
  thread for the bare kernel; here each call also prepares its arrays in Python, and the
  wheel targets every x86-64 CPU (no `target-cpu`).
- **The path tracer, end to end** (160x120, 8 samples, 3 bounces, 30 spheres and a
  plane, one process; measured separately): 0.45 s instead of 1.85 s, 4.1 times faster,
  with bitwise the same image, linear light and depth. Shading, in NumPy, is most of what
  remains.
- **PNG.** Decoding a 1080p image takes 45 ms instead of 324 (the reconstruction was a
  Python loop over anti-diagonals); encoding is now mostly zlib (level 6: 433 ms, of
  which choosing the filters takes a few). MCP-sized previews take 10.9 ms instead of 17.7.
- **World matrices** barely change (6.4 against 6.6 ms for 10,000 flat objects): gathering
  the transforms from Python objects costs more than the arithmetic, and renders reuse
  the retained world's matrices anyway.
- **Not affected**: GPU frames (the GPU traces nothing here), extraction, editing.

## Phase 4: mesh editing

The audit proposed rebuilding `EditableMesh` on arrays (half-edge tables and corner
layers) with native operators. Its geometric decisions -- which triangles join into quads
(`from_mesh`, `join_triangles`), convexity, the shading inferred from normals, the points
an operation computes -- compare values from `np.dot`, `np.linalg.norm`, `np.arccos` and
`np.log`: BLAS and libm (or SVML) calls whose last bits depend on the platform. A vectorized
or native rewrite of those computations could not promise the same quads and the same
results, so this phase changed only what computes exactly what it replaces.

**What changed**

- **Bulk construction** (`MeshCore._bulk_faces`; switch `AETHER3D_EDITING_BULK`):
  `EditableMesh.from_mesh` builds its faces, edges and corner layers with NumPy -- the
  same ids, the same tables in the same insertion orders, each vertex's edge set filled in
  increasing edge id as building one face at a time fills it (so even the iteration order
  of sets matches), and one Python int per id shared by every table, as before.
- **Quad joining** computes each triangle's normal once instead of once per edge.
- **`poly.cross3`** (bevel, inset, extrude): `np.cross`'s operations for two float64
  3-vectors, without its per-call cost: 0.8 us per call instead of 11.9 (timeit on the
  development machine; in `np.cross`, mostly axis bookkeeping).

`tests/test_editing_bulk.py` compares meshes built both ways with a strict snapshot (every
table in its order, every set in its iteration order) on primitives, face groups, a random
soup with welding, signed zeros, degenerate and non-manifold triangles and an empty mesh,
with both welds and both quad settings; runs a chain of operations on both; compares quad
joining with a copy of the previous implementation; `cross3` with `np.cross` (signed
zeros, infinities, NaN, extreme magnitudes); and the builder's memory with the old one.

**A regression found and fixed on the way.** The first bulk builder gave every table its
own Python int for each id and kept its temporaries to the end: building the
199,808-triangle mesh peaked at 438 MiB (tracemalloc) instead of 251, and the mesh held
305 MiB instead of 226. The benchmark's peak memory showed it (517 MiB for the editing
group instead of 322); sharing the ids and freeing temporaries early brought the build to
248 MiB at peak, the mesh to 205 MiB, and the group back to 328 MiB.

**Measured** (`benchmarks/results/phase4.json`):

| metric | phase 3 | phase 4 |
| --- | ---: | ---: |
| `editing.from_mesh_topology_s` (199,808 triangles) | 0.732 | 0.348 |
| `editing.from_mesh_attributes_s` (16,128 triangles) | 0.875 | 0.585 |
| `editing.bevel_all_edges_ms` (2,048 faces) | 691.1 | 502.6 |
| `editing.inset_500_faces_ms` | 133.5 | 99.05 |
| `editing.bevel_edge_ms` | 1.388 | 1.171 |
| `editing.extrude_region_ms` | 5.909 | 5.739 |
| `editing.subdivide_quads_ms` | 118.1 | 115.9 |
| `editing.to_mesh_ms` | 35.17 | 33.79 |
| `editing.peak_rss_mib` | 321.7 | 328.4 |

**Targets of the audit not met**: opening a 200,000-triangle mesh *with* UVs and normals
within 1 s (here 16,128 such triangles take 0.59 s, most of it joining triangle pairs into
quads, each pair's decision taken with the scalar geometry described above), and
bevelling 51,000 faces within 1 s (2,048 faces take 0.50 s: per-vertex geometry and
topology updates in Python). Meeting them needs the redesign the audit planned -- array
topology and native operators -- with geometric predicates that no longer depend on the
platform's BLAS and libm, which would change some quads and the last bits of computed
points relative to 0.11 on some platforms. That is a deliberate behavior change for a
release of its own, with the editing corpus as its guard; it was not made silently here.

## Phase 5: hardware ray tracing (prototype, deferred)

The audit's last phase was optional: hardware ray tracing through a Rust wgpu host,
prototyped end to end first, and integrated only if showcase path renders became at least
twice as fast end to end with at least 99.99 % of rays hitting the same triangle as the
software traversal. wgpu-py cannot build acceleration structures (wgpu-native's C API has
none), so ray queries need a Rust host that owns its own GPU device.

**The prototype** ([research/2026-10-modernization/hw_pathtrace](../research/2026-10-modernization/README.md#phase-5-hardware-ray-queries-in-the-gpu-path-tracer-hw_pathtrace))
captures what `render_scene` binds and dispatches for each showcase scene (the settings of
`benchmarks/bench_gpu_pathtrace.py`) and replays it in a Rust host (wgpu 30.0.1, Vulkan):
with the production kernel unchanged, or with only `trace_ray` replaced by ray queries over
a BLAS per mesh and a TLAS of the instances. The replacement keeps the software
traversal's semantics -- hits with tmin < t < tmax, never the triangle a ray leaves from
(an opaque query, then, only when it returns that triangle or a hit at t <= tmin, a
non-opaque query confirming candidates one by one), any hit for shadow and AO rays. The
host's software renders are bitwise `render_scene`'s (all 22 accumulation planes, every
scene), so the comparison is Aether3D's path tracer with and without hardware traversal.

**Measured** (RTX 4060 Laptop GPU, Vulkan; trace: the sample passes, wall and GPU time;
end to end: the Python render's measured CPU stages with the host's GPU stages, and for
ray queries without the CPU BVH build):

| Scene | Size, samples | Triangles (instances) | Software trace (GPU) | Ray queries (GPU) | Trace | End to end |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cornell | 640x640, 256 | 36 (8) | 8.99 s (8.62 s) | 7.76 s (7.42 s) | 1.16x | 9.07 -> 7.84 s: 1.16x |
| metals | 960x540, 128 | 36,122 (7) | 2.63 s (2.44 s) | 1.70 s (1.52 s) | 1.55x | 2.74 -> 1.80 s: 1.52x |
| window-room | 960x640, 512 | 102 (16) | 40.56 s (38.95 s) | 25.03 s (23.51 s) | 1.62x | 40.70 -> 25.17 s: 1.62x |
| castle | 1920x1080, 64 | 6,842 (25) | 17.16 s (16.58 s) | 10.13 s (9.57 s) | 1.70x | 17.86 -> 10.82 s: 1.65x |
| hero | 1280x720, 64 | 458 (8) | 2.28 s (2.00 s) | 2.05 s (1.76 s) | 1.11x | 2.58 -> 2.34 s: 1.10x |
| castle-ao | 1920x1080, 8 | 6,842 (25) | 0.36 s (0.30 s) | 0.33 s (0.27 s) | 1.08x | 0.86 -> 0.83 s: 1.04x |

| Scene | Camera rays: same triangle | Bounce rays: same triangle | Any-hit: same verdict | Rays that differ (same point / one hit / elsewhere) | Image: difference / sample noise | Mean bias | Host = render_scene |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| cornell | 100.0000 % of 409,600 | 100.0000 % of 409,600 | 100.0000 % of 409,600 | camera 0 / 0 / 0; bounce 0 / 0 / 0 | 6.6e-04 | 0.0e+00 | bitwise |
| metals | 100.0000 % of 518,400 | 100.0000 % of 395,706 | 100.0000 % of 395,706 | camera 0 / 0 / 0; bounce 0 / 0 / 0 | 7.8e-04 | -2.8e-07 | bitwise |
| window-room | 100.0000 % of 614,400 | 100.0000 % of 582,094 | 100.0000 % of 582,094 | camera 0 / 0 / 0; bounce 0 / 0 / 0 | 3.8e-03 | 2.8e-07 | bitwise |
| castle | 99.9999 % of 2,073,600 | 100.0000 % of 2,073,599 | 100.0000 % of 2,073,599 | camera 1 / 1 / 1; bounce 1 / 0 / 0 | 1.7e-02 | 9.9e-07 | bitwise |
| hero | 99.9962 % of 921,600 | 99.9956 % of 497,043 | 100.0000 % of 497,043 | camera 35 / 0 / 0; bounce 21 / 1 / 0 | 2.1e-02 | -2.8e-06 | bitwise |
| castle-ao | 99.9999 % of 2,073,600 | 100.0000 % of 2,073,599 | 100.0000 % of 2,073,599 | camera 1 / 1 / 1; bounce 1 / 0 / 0 | 6.7e-03 | 2.4e-07 | bitwise |

(The AO-only castle traces the castle's agreement rays again: the agreement kernel does
not depend on the integrator.)

- **Agreement holds.** At least 99.995 % of camera and bounce rays -- the path tracer's
  own rays, traced with both traversals -- hit the same triangle, and every any-hit
  verdict agrees; almost every ray that differs hits the same point on another triangle
  (a shared edge or coincident faces). Images differ by 0.07-2.1 % of the sample noise,
  without bias (within 3e-6 of the mean). The pixel-center pass (the depth and instance
  outputs) agrees on at least 99.997 % of the pixels, except in the Cornell box, whose
  differing 0.084 % all lie on the image's diagonals: rays along the box's corner edges,
  where the software traversal slips between two walls (152 pixels) or hits the other
  wall at the same depth (194), and ray queries hit a wall every time.
- **The 2x target is missed.** The trace is 1.08-1.70 times faster and renders 1.04-1.65
  times faster end to end; building the acceleration structures takes about 1 ms. The
  GPU time ray queries remove is a lower bound for what the software traversal costs
  (if the rest of the kernel costs the same in both): 10-42 % of the kernel. The rest --
  material evaluation, light sampling, the state machine of one large kernel and its
  divergence -- is untouched by ray queries.

**Decision: deferred; the software traversal stays the only one.** Integration would add
a second GPU device owned by a Rust extension (scene data and textures uploaded twice, the
kernel compiled twice), wgpu's experimental ray-query API (measured on Vulkan only) and a
large native dependency, for the gains above. It becomes worth measuring again (the
prototype is kept for that) after shading is restructured -- a wavefront path tracer
sorting hits by material, where traversal could dominate -- or when wgpu-native's C API
gains acceleration structures, which would remove the second device.

## Visual feedback (0.12.0)

Stage 12 added no kernel: observations reuse the renderers, the retained render world and
selective readback; per-object statistics are NumPy reductions over the buffers the
renderers return; geometry reuses the spatial queries' exact kernels. The measurements
(`benchmarks/bench_feedback.py`, saved as `benchmarks/results/stage12-feedback.json`; the
full table is in [FEEDBACK.md](FEEDBACK.md#performance)), at 640 x 360:

- One GPU observation after one object moved: 24.0 ms (34 objects), 26.8 ms (2,190),
  46.2 ms (11,270); with the full extraction instead of the retained world 24.6, 36.8 and
  96.4 ms. On the CPU rasterizer 1.1-3.1 s, about 99 % of it rendering.
- An observation's measurements (every object's pixels, depth and luminance; the image's
  statistics): 11-15 ms. An unchanged view is reused from the store in 0.01 ms.
- The checks of a diagnosis: 240 ms for the furnished room (6 checks), 312 ms for the
  castle (4 checks) -- mostly exact triangle tests deciding contacts and separations.
- A whole loop from the first draft to accepted on the GPU: 390 ms for the room (1
  iteration, 2 renders) and 179 ms for the garden view (2 iterations, 3 renders).

One measured change: the image statistics reduced the last axis of 8-bit images
(`max`/`any` over RGB), which NumPy does slowly; elementwise operations over the three
channels give the same results, 2.5 times faster on the room's render (9.31 -> 3.71 ms).

## Advanced modeling and sculpting (0.13.0)

Stage 13 added no native kernel and no GPU code. The design record
([research/2026-10-stage13/DESIGN.md](../research/2026-10-stage13/DESIGN.md)) measured
prototypes first and chose NumPy, data-oriented, for sculpting, dynamic topology and
remeshing, and Python on the editing engine for the modeling operators. The measurements
(`benchmarks/bench_sculpt.py`, saved as `benchmarks/results/stage13-sculpt.json`; the
same file's `baseline` group on `main` 35e5a23 before the stage is
`benchmarks/results/stage13-baseline.json`) run every group and size in its own process,
on UV spheres of about 10k, 100k and 1M vertices (an i9-12900H laptop, Python 3.14, NumPy 2.5; commit 5323c44):

| | 10k vertices | 100k | 1M |
| --- | ---: | ---: | ---: |
| mesh (vertices / triangles) | 10,011 / 19,320 | 101,025 / 199,808 | 1,001,820 / 1,996,568 |
| `SculptMesh.from_mesh` | 0.01 s | 0.16 s | 1.89 s |
| commit after one dab (`to_mesh`) | 1.06 ms | 9.52 ms | 99.9 ms |
| commit after a dynamic-topology dab | 1.67 ms | 18.6 ms | 188 ms |
| one `draw` dab, radius 0.03 (vertices moved) | 0.74 ms (2) | 1.53 ms (26) | 2.68 ms (249) |
| one `draw` dab, radius 0.1 (vertices moved) | 1.39 ms (26) | 2.33 ms (271) | 7.30 ms (2,715) |
| one `draw` dab, radius 0.3 (vertices moved) | 1.86 ms (241) | 6.62 ms (2,562) | 44.9 ms (23,675) |
| 100-dab `draw` stroke, radius 0.1, per dab | 0.92 ms | 1.57 ms | 6.54 ms |
| 100-dab `clay` stroke, radius 0.1, per dab | 0.78 ms | 1.46 ms | 5.57 ms |
| 100-dab `smooth` stroke, radius 0.1, per dab | 2.80 ms | 6.00 ms | 21.1 ms |
| 100-dab `grab` stroke, radius 0.1, per dab | 0.26 ms | 0.95 ms | 6.77 ms |
| undo of a 100-dab stroke | 1.28 ms | 4.05 ms | 23.5 ms |
| commit after a 100-dab stroke | 1.46 ms | 12.5 ms | 121 ms |
| journal of four strokes | 0.4 MiB | 3.1 MiB | 27.4 MiB |
| symmetry none: per dab | 0.93 ms | 1.63 ms | 6.68 ms |
| symmetry x: per dab | 1.45 ms | 2.96 ms | 12.5 ms |
| symmetry xy: per dab | 2.04 ms | 4.66 ms | 19.3 ms |
| dynamic topology, brush scaled to the density: per dab (radius) | 13.3 ms (0.15) | 16.0 ms (0.0466) | 20.1 ms (0.0148) |
| dynamic topology, radius 0.15: per dab | 13.2 ms | 85.7 ms | 870 ms |
| ... triangles before -> after 30 dabs | 19,320 -> 20,136 | 199,808 -> 206,958 | 1,996,568 -> 2,064,030 |
| ... undo | 3.08 ms | 16.3 ms | 137 ms |
| isotropic `remesh` (iterations) | 1.77 s (3) | 27.39 s (3) | 199 s (1) |
| `check_mesh` | 0.04 s | 0.49 s | 6.11 s |
| `check_mesh` with the exact self-intersection test | 0.36 s | 3.78 s | 47.56 s |
| peak memory, MiB (dab / stroke / remesh / validity) | 61 / 61 / 166 / 89 | 139 / 138 / 544 / 239 | 886 / 885 / 3440 / 1702 |

Voxel remeshing of the 100k sphere:

| Voxel remesh of the 100k sphere | voxel size | triangles out | time | result | peak MiB |
| --- | ---: | ---: | ---: | --- | ---: |
| about 10k triangles | 0.06 | 10,436 | 3.92 s | closed | 416 |
| about 100k triangles | 0.019 | 104,348 | 5.58 s | closed | 460 |
| about 1M triangles | 0.006 | 1,047,020 | 14.39 s | closed | 1224 |

The modeling operators on the editing engine:

| Modeling operator (`EditableMesh`) | 10k faces | 100k faces |
| --- | ---: | ---: |
| 3-segment bevel of a latitude ring (edges) | 69.3 ms (140) | 411 ms (448) |
| `bisect` | 93.3 ms | 1.39 s |
| `bisect` with `clear` and `fill` | 165 ms | 2.12 s |
| `knife_project` across the top | 204 ms | 2.00 s |
| `cleanup` | 61.0 ms | 775 ms |
| `recalculate_normals` | 76.6 ms | 881 ms |
| peak memory, MiB | 104 | 656 |

Against what 0.12 offered for the same work (the editing API: select the vertices in a
sphere, move them along their normals with no falloff, commit):

| Against 0.12 (same machine) | 0.12 (editing API) | 0.13 (sculpt mesh) | factor |
| --- | ---: | ---: | ---: |
| open a 100k-vertex mesh | 12.43 s | 0.16 s | 76x |
| a dab of radius 0.1 at the pole (3,137 / 3,137 vertices moved) | 66.9 ms | 7.33 ms | 9x |
| commit it | 3.56 s | 15.8 ms | 226x |

Measured choices:

- **The voxel band is scattered from the triangles.** A first version asked a BVH for
  each grid point's nearest triangle; each triangle now measures the grid points of its
  own box instead, and a scatter keeps the minimum per point (lowest triangle id among
  ties). The same surfaces (equal band points, triangle counts and volumes) at 3.7-4.5
  times the speed: 28.3 s -> 6.2 s for a 65k-triangle sphere at a voxel size giving 377k
  triangles, 138 s -> 37 s for a 517k-triangle one giving 1.5M. Narrowing the band from
  2 to 1.75 voxels (what marching cubes needs: every corner of a crossed cell lies within
  `sqrt(3)` voxels of the surface; the output is bit for bit the same, tested) and
  scattering minima instead of sorting each batch brought these to 4.8 s and 26.8 s.
- **Collapses choose their independent sets in rounds.** One pass that kept each
  candidate whose priority (its length) was the smallest around it collapsed only 48 of
  13,536 short edges of a dense sphere: neighbouring candidates have nearly equal
  lengths, so few win. Validity is now checked for every candidate first, and the set is
  chosen in Luby-style rounds with scrambled priorities: 640 collapses in the first pass,
  and 1,938 of the sphere's 9,024 triangles left after ten passes, against 7,926 after
  twelve before.
- **Exact predicates cost little.** `orient3d` takes 19.2 ms per 200,000 random rows,
  3.3 times a plain float determinant (5.8 ms): Shewchuk's static filter decides almost
  every row in float64, and the rational fallback (29 microseconds per row) runs only for
  nearly coplanar configurations -- where a float answer can be wrong.
- **Strokes read the mesh's extent from the chunk boxes.** The extent (for the mirror
  planes' tolerance and ray offsets) came from a copy of every vertex: 41 of the 45 ms of a
  one-dab stroke at 1M. The chunk boxes hold every vertex of a triangle and are tight
  between operations; a one-dab stroke at 1M now takes 2.7 ms.
- **Isotropic remeshing runs on a detached copy.** Its whole-mesh passes journaled every
  element they touched, the replaced rows stayed in the arrays, and each check
  materialized every candidate's neighbourhood at once: 471 s and a 9.3 GB peak for one
  iteration at 1M, 43.3 s and 1,155 MiB for three at 100k (the same benchmark at commit
  b2232f8: `benchmarks/results/stage13-sculpt-b2232f8.json`). The passes now run on a
  compact copy that records nothing (compacted again when dead rows pile up), the
  collapse, flip and UV re-sampling checks run in batches -- each on the candidates the
  cheaper checks left, with the cached triangle cross products -- and projections search
  within twice the split length first. The batched checks and the bounded projection
  give the same remesh, bit for bit (compared in one process with the previous passes);
  the compact copy renumbers the vertices, which changes which of the equally valid
  independent sets the scrambled priorities pick, so the result is another valid remesh
  of the same quality (the tests check validity and quality, not ids). The mesh is
  replaced once, as one journaled step: 199 s and 3.4 GB at 1M, 27.4 s and 544 MiB at
  100k.
