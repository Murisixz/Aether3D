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
