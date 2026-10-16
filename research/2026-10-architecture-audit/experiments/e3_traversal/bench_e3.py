"""E3: closest-hit BVH traversal, same algorithm and data, every implementation checked bitwise."""
import ctypes, json, statistics, sys, time
import numpy as np
E = sys.argv[1]
sys.path.insert(0, E)
from e3common import Accel, Rays, flat, accel_struct, scenes, reference, compare
import numba
import trace_numba

libs = {}
for label, so in (("C (gcc -O3)", "libtrace_c_omp.so"), ("C (clang -O3)", "libtrace_clang.so"), ("Rust 1.99", "libtrace_rs.so"),
                  ("Zig 0.17", "libtrace_zig.so"), ("Go 1.27", "libtrace_go.so"), ("C# .NET 10 AOT", "libtrace_cs.so")):
    lib = ctypes.CDLL(f"{E}/{so}")
    lib.closest_hit.argtypes = [ctypes.POINTER(Accel), ctypes.POINTER(Rays), ctypes.c_int64]; lib.closest_hit.restype = ctypes.c_int
    if hasattr(lib, "set_lengths"):
        lib.set_lengths.argtypes = [ctypes.c_void_p]; lib.set_lengths.restype = None
    libs[label] = lib
THREADS = (1, 8, 20)
mt_ok = {"C (gcc -O3)", "Rust 1.99", "Zig 0.17", "Go 1.27", "C# .NET 10 AOT"}

def timed(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

results = {}
for sname, (accel, batches) in scenes().items():
    arrs = flat(accel); A = accel_struct(arrs, accel.stack_size)
    lengths = np.array([arrs[k].size for k in ("child", "child_lo", "kind", "v0", "tri_prim", "slot_inst", "inst_lo", "inst_root", "to_object")], np.int64)
    for lib in libs.values():
        if hasattr(lib, "set_lengths"):
            lib.set_lengths(lengths.ctypes.data)
    for bname, rb in batches.items():
        key = f"{sname} | {bname}"
        accel.traced.__init__()
        t = time.perf_counter(); ref = reference(accel, rb); t_np = time.perf_counter() - t
        visits = accel.traced.node_visits / rb.n
        row = {"rays": rb.n, "node_visits_per_ray": round(visits, 1), "NumPy (current)": {"1": round(rb.n / t_np / 1e6, 3)}, "parity": {}}
        for label, lib in libs.items():
            row[label] = {}
            for nt in THREADS:
                if nt > 1 and label not in mt_ok:
                    continue
                outs = rb.outputs(); R = rb.struct(outs)
                rc = lib.closest_hit(ctypes.byref(A), ctypes.byref(R), nt)
                assert rc == 0, (label, rc)
                cmp = compare(ref, outs)
                row["parity"][f"{label} x{nt}"] = cmp["mismatched_rays"] == 0 and cmp["bary_bitwise"]
                sec = timed(lambda: lib.closest_hit(ctypes.byref(A), ctypes.byref(R), nt), 5 if nt > 1 else 3)
                row[label][str(nt)] = round(rb.n / sec / 1e6, 2)
        row["Numba 0.68"] = {}
        for nt in THREADS:
            outs = rb.outputs()
            if nt > 1:
                numba.set_num_threads(nt)
            fn = lambda: trace_numba.closest_hit(arrs, rb, outs, parallel=nt > 1)
            sec = timed(fn, 5 if nt > 1 else 3)
            cmp = compare(ref, outs)
            row["parity"][f"Numba x{nt}"] = cmp["mismatched_rays"] == 0 and cmp["bary_bitwise"]
            row["Numba 0.68"][str(nt)] = round(rb.n / sec / 1e6, 2)
        results[key] = row
        print(key, json.dumps({k: v for k, v in row.items() if k != "parity"}), "all parity:", all(row["parity"].values()), flush=True)
json.dump(results, open(f"{E}/e3_results.json", "w"), indent=1)
