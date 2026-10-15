import ctypes, json, statistics, sys, time
sys.path.insert(0, sys.argv[1])
from e3common import Accel, Rays, flat, accel_struct, scenes, reference, compare
res = {}
libs = {}
for label, so in (("C (gcc)", "libtrace_c_omp.so"), ("C++ (g++)", "libtrace_cpp.so")):
    lib = ctypes.CDLL(f"{sys.argv[1]}/{so}"); lib.closest_hit.argtypes = [ctypes.POINTER(Accel), ctypes.POINTER(Rays), ctypes.c_int64]; lib.closest_hit.restype = ctypes.c_int
    libs[label] = lib
for sname, (accel, batches) in scenes().items():
    A = accel_struct(flat(accel), accel.stack_size)
    for bname, rb in batches.items():
        ref = reference(accel, rb)
        row = {}
        for label, lib in libs.items():
            for nt in (1, 20):
                outs = rb.outputs(); R = rb.struct(outs)
                lib.closest_hit(ctypes.byref(A), ctypes.byref(R), nt)
                ok = compare(ref, outs)["mismatched_rays"] == 0
                ts = []
                for _ in range(5):
                    t = time.perf_counter(); lib.closest_hit(ctypes.byref(A), ctypes.byref(R), nt); ts.append(time.perf_counter() - t)
                row[f"{label} x{nt}"] = (round(rb.n / statistics.median(ts) / 1e6, 2), ok)
        print(sname, "|", bname, row, flush=True)
