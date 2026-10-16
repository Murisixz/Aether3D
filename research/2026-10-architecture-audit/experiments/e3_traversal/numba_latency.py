import sys, time
t0 = time.perf_counter(); import numba; t_imp = time.perf_counter() - t0
sys.path.insert(0, sys.argv[1])
from e3common import scenes
import trace_numba
(accel, batches) = next(iter(scenes().values()))
from e3common import flat
arrs = flat(accel); rb = batches["camera 640x480"]
outs = rb.outputs()
t0 = time.perf_counter(); trace_numba.closest_hit(arrs, rb, outs); t_first = time.perf_counter() - t0
t0 = time.perf_counter(); trace_numba.closest_hit(arrs, rb, outs); t_second = time.perf_counter() - t0
t0 = time.perf_counter(); trace_numba.closest_hit(arrs, rb, outs, parallel=True); t_pfirst = time.perf_counter() - t0
print(f"import numba {t_imp*1e3:.0f} ms; first call (compile + run) {t_first:.2f} s; warm call {t_second*1e3:.0f} ms; first parallel call {t_pfirst:.2f} s")
