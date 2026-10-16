"""Camera rebase on the CPU (float64 subtraction rounded to float32, what Aether3D packs)
versus a GPU rebase from hi/lo float32 pairs: (t_hi - e_hi) + (t_lo - e_lo) in float32."""
import numpy as np

rng = np.random.default_rng(7)
f32 = np.float32
for extent in (10.0, 1e3, 1e5, 1e7):
    t = rng.uniform(-extent, extent, 2_000_000)
    e = t + rng.normal(0.0, extent * 1e-2, t.size)  # a camera among the objects
    cpu = (t - e).astype(f32)
    t_hi = t.astype(f32); t_lo = (t - t_hi.astype(np.float64)).astype(f32)
    e_hi = e.astype(f32); e_lo = (e - e_hi.astype(np.float64)).astype(f32)
    gpu = (t_hi - e_hi) + (t_lo - e_lo)  # float32 arithmetic, as WGSL would do it
    differ = np.count_nonzero(cpu.view(np.uint32) != gpu.view(np.uint32))
    exact = t - e
    err_cpu = np.max(np.abs(cpu.astype(np.float64) - exact))
    err_gpu = np.max(np.abs(gpu.astype(np.float64) - exact))
    print(f"objects within +-{extent:g} m: {differ / t.size:6.2%} of translations differ in bits; "
          f"max error vs exact: CPU {err_cpu:.3g} m, GPU hi/lo {err_gpu:.3g} m")
