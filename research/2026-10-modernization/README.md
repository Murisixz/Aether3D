# Performance modernization: investigations (October 2026)

Measurements made during the performance modernization ([docs/PERFORMANCE.md](../../docs/PERFORMANCE.md))
to decide questions its phases raised. Like the audit next to it
([2026-10-architecture-audit](../2026-10-architecture-audit/README.md)), this is research
code: not part of the package, excluded from the linters, kept as it ran. Run the scripts
from the repository root with the project's virtual environment.

## Phase 2: camera rebase on the GPU? (`camera_rebase.py`)

The GPU rasterizer draws with float32 positions relative to the camera. Aether3D computes
each instance's camera-relative translation on the CPU, `float32(translation - eye)` with
the subtraction in float64, so far geometry keeps its precision near the camera. When the
camera moves, every instance record changes: with the kept frame data of Phase 2, only
the translations are rewritten (about 1 ms per camera move for 10,000 objects on the
RTX 4060, measured), against nothing for an unchanged frame.

The alternative is the usual GPU technique: upload each translation once as a float32
pair (`hi = float32(t)`, `lo = float32(t - hi)`), and the eye the same way every frame,
and subtract on the GPU, `(t_hi - e_hi) + (t_lo - e_lo)` in float32. The script compares
the two on 2,000,000 random translations per scale, with a camera among the objects
(output on the machine of the audit):

```
objects within +-10 m:  0.23% of translations differ in bits; max error vs exact: CPU 1.49e-08 m, GPU hi/lo 2.95e-08 m
objects within +-1000 m:  0.23% of translations differ in bits; max error vs exact: CPU 1.91e-06 m, GPU hi/lo 3.75e-06 m
objects within +-100000 m:  0.23% of translations differ in bits; max error vs exact: CPU 0.000239 m, GPU hi/lo 0.000459 m
objects within +-1e+07 m:  0.23% of translations differ in bits; max error vs exact: CPU 0.0156 m, GPU hi/lo 0.031 m
```

**Decision: keep the rebase on the CPU.** The GPU version would change 0.23% of instance
translations in their last bit -- renders would no longer be bitwise those of the CPU
reference packing, nor of earlier releases -- and its worst error is twice the CPU's,
to save about 1 ms per camera move at 10,000 objects.

## Phase 5: hardware ray queries in the GPU path tracer (`hw_pathtrace/`)

The audit's optional last phase was hardware ray tracing, with a prototype first and two
conditions: at least 99.99 % of rays hitting the same triangle as the software traversal
(with image-level tolerances), and showcase path renders at least twice as fast end to
end. The audit measured traversal alone (E7: ray queries 6.5-15.1 times faster than
`trace_ray`) and left open what that gives inside the real path tracer, whose shading it
does not touch. This prototype measures exactly that.

wgpu-py cannot build acceleration structures (wgpu-native's C API has none: the audit's
E7), so ray queries need a Rust program on wgpu (30.0.1, Vulkan) that owns its own GPU
device. The prototype keeps everything else Aether3D's:

- **`export.py`** renders a showcase scene with `backend="gpu_path"` at the settings of
  `benchmarks/bench_gpu_pathtrace.py` (each scene's CPU record) and captures what
  `render_scene` (`rendering/gpu/pathtrace.py`) binds and dispatches: every buffer, the
  parameters, the tiles and samples per dispatch, the accumulation planes it reads back
  and its stage timings (a first render with the BLAS cache empty and the pipeline
  compiling, and a second, warm one). It writes the production kernel unchanged
  (`sw.wgsl`), the same module with only `trace_ray` on ray queries (`hw.wgsl`), and an
  agreement kernel with both traversals (`agree.wgsl`).
- **`host/`** (Rust) replays exactly what `render_scene` dispatches -- the same buffers and
  bind groups, the same tiles and samples per dispatch, the counters cleared before and
  read back after every dispatch, the pixel-center pass, the planes read back -- with
  `sw.wgsl`, or with `hw.wgsl` over a BLAS per mesh and a TLAS of the instances
  (camera-relative transforms, the instance index as custom data). GPU time comes from
  timestamp queries around each dispatch.
- **The ray-query `trace_ray` keeps the software traversal's semantics**: the closest hit
  (or any hit, for shadow and AO rays) with tmin < t < tmax, never the triangle the ray
  leaves from. An opaque query finds the hit; only when that hit is the skipped triangle
  or lies at t <= tmin does a second, non-opaque query confirm the candidates one by one
  ("fallback rays"). `Hit.b` holds the three corner weights, as the software test's.
- **The agreement kernel** traces, per pixel, the path tracer's first camera ray, a
  cosine-distributed bounce ray leaving the software camera hit, and a short any-hit ray
  (5 % of the scene's size) from there, with both traversals.
- **`compare.py`** compares: the host's software render with `render_scene`'s planes; the
  ray-query render with the software one (the image mean, the per-pixel difference against
  the sample noise -- a software render of the next `samples` samples, independent
  paths --, the pixel centers' instances and depths); the agreement kernel's rays; and the
  timings. **`run.py`** runs all of it per scene (the first host render of each kernel is
  a warm-up; two timed renders of each alternate), **`summarize.py`** prints the tables
  below from `results/*.json`.

```sh
cd research/2026-10-modernization/hw_pathtrace/host && cargo build --release && cd ../../../..
python research/2026-10-modernization/hw_pathtrace/run.py OUT \
    research/2026-10-modernization/hw_pathtrace/host/target/release/hwpt
python research/2026-10-modernization/hw_pathtrace/summarize.py
```

It needs an adapter with ray queries (`Features::EXPERIMENTAL_RAY_QUERY`; the RTX 4060
here), `aether3d[gpu,boolean]`, and about 1 GB of scratch space per 1080p scene (freed
after each scene).

**Results** (RTX 4060 Laptop GPU, driver 615.78.08, Vulkan; `run.py`, one host process per scene):

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

| Scene | Fallback rays | Pass-through | AS build | Software traversal share (lower bound) |
| --- | ---: | --- | ---: | ---: |
| cornell | 16,981 | yes | 0.81 ms | 14 % |
| metals | 0 | no | 0.82 ms | 37 % |
| window-room | 0 | no | 0.89 ms | 40 % |
| castle | 166 | no | 0.92 ms | 42 % |
| hero | 0 | no | 0.63 ms | 12 % |
| castle-ao | 0 | no | 1.02 ms | 10 % |

(The AO-only castle traces the castle's agreement rays again: the agreement kernel does
not depend on the integrator. "Software traversal share": the GPU time ray queries
remove, a lower bound for what the software traversal costs if the rest of the kernel
costs the same in both.)

- **The host is faithful.** Its software renders are bitwise `render_scene`'s -- all 22
  accumulation planes of every scene. The same dispatches take 2-9 % longer from Python
  (one warm Python render against the host's median; Python reads the counters back
  through wgpu-py after every dispatch), so both kernels are compared in the host.
- **Agreement holds.** At least 99.995 % of camera and bounce rays hit the same triangle,
  and every any-hit verdict agrees; almost every ray that differs hits the same point on
  another triangle (a shared edge or coincident faces). Images differ by 0.07-2.1 % of
  the sample noise, without bias (within 3e-6 of the mean). The pixel-center pass agrees
  on at least 99.997 % of the pixels, except in the Cornell box, whose differing 0.084 %
  all lie on the image's diagonals: rays along the box's corner edges, where the software
  traversal slips between two walls (152 pixels) or hits the other wall at the same depth
  (194), and ray queries hit a wall at all of them.
- **Speed falls far short.** Ray queries make the trace 1.08-1.70 times faster and
  renders 1.04-1.65 times faster end to end ("end to end": the Python render's measured
  CPU stages -- extraction, setup, finishing -- with the host's GPU stages, and for ray
  queries without the CPU BVH build). They remove 10-42 % of the kernel's GPU time;
  shading -- material evaluation (the castle's procedural textures), light sampling, the
  state machine of one large kernel and its divergence -- is the rest. Even the AO-only
  castle, which shades almost nothing, traces only 1.08 times faster: its short AO rays
  and coherent camera rays were already cheap in software (279 M rays per second).

**Decision: deferred.** Integrating ray queries would make the GPU path tracer depend on
a second GPU device owned by a Rust extension (wgpu-py cannot share buffers with it, so
scene data and textures would be uploaded twice and the kernel compiled twice), on
wgpu's experimental ray-query API (measured on Vulkan only), and on a large native
dependency, for 1.04-1.65 times on the showcase scenes. The software traversal stays the
GPU path tracer's only traversal. What would change the answer: a restructured shading
stage (wavefront path tracing, sorting hits by material), after which traversal could
dominate and this prototype can measure again; or acceleration structures in
wgpu-native's C API, which would remove the second device.
