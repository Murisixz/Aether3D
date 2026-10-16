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
