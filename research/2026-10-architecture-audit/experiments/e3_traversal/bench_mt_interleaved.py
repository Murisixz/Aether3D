"""E3 multi-threaded rerun: 20 threads, implementations interleaved in random order per round, 7 rounds, median."""
import ctypes, json, random, statistics, sys, time
import numba
sys.path.insert(0, sys.argv[1])
from e3common import Accel, Rays, flat, accel_struct, scenes, reference, compare
import trace_numba
E = sys.argv[1]
libs = {}
for label, so in (("C (gcc)", "libtrace_c_omp.so"), ("Rust", "libtrace_rs.so"), ("Zig", "libtrace_zig.so"), ("Go", "libtrace_go.so"), ("C# AOT", "libtrace_cs.so")):
    lib = ctypes.CDLL(f"{E}/{so}"); lib.closest_hit.argtypes = [ctypes.POINTER(Accel), ctypes.POINTER(Rays), ctypes.c_int64]; lib.closest_hit.restype = ctypes.c_int
    if hasattr(lib, "set_lengths"):
        lib.set_lengths.argtypes = [ctypes.c_void_p]
    libs[label] = lib
numba.set_num_threads(20)
random.seed(1)
out = {}
for sname, (accel, batches) in scenes().items():
    arrs = flat(accel); A = accel_struct(arrs, accel.stack_size)
    import numpy as np
    lengths = np.array([arrs[k].size for k in ("child", "child_lo", "kind", "v0", "tri_prim", "slot_inst", "inst_lo", "inst_root", "to_object")], np.int64)
    for lib in libs.values():
        if hasattr(lib, "set_lengths"): lib.set_lengths(lengths.ctypes.data)
    for bname, rb in batches.items():
        ref = reference(accel, rb)
        calls = {}
        for label, lib in libs.items():
            outs = rb.outputs(); R = rb.struct(outs)
            calls[label] = (lambda lib=lib, R=R: lib.closest_hit(ctypes.byref(A), ctypes.byref(R), 20), outs)
        o_nb = rb.outputs()
        calls["Numba"] = (lambda o=o_nb: trace_numba.closest_hit(arrs, rb, o, parallel=True), o_nb)
        times = {k: [] for k in calls}
        for k, (fn, _) in calls.items():
            fn()
        for _ in range(7):
            order = list(calls); random.shuffle(order)
            for k in order:
                t = time.perf_counter(); calls[k][0](); times[k].append(time.perf_counter() - t)
        row = {k: round(rb.n / statistics.median(v) / 1e6, 1) for k, v in times.items()}
        row["parity"] = all(compare(ref, o)["mismatched_rays"] == 0 for _, o in calls.values())
        out[f"{sname} | {bname}"] = row
        print(sname, "|", bname, row, flush=True)
json.dump(out, open(f"{E}/e3_mt_interleaved.json", "w"), indent=1)
