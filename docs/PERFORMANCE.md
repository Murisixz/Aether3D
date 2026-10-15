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
| (none yet) | | |

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
