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
