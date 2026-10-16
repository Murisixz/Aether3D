"""E4a: per-call overhead from CPython 3.14 into native code (noop and 3-double add), median of 7 x 1M calls."""
import ctypes, json, statistics, sys, timeit
sys.path.insert(0, sys.argv[1])
import numba, numpy as np
import capi_mod, cy_mod, nb_mod, pyo3_mod, cffi_api_mod
from cffi import FFI
lib = ctypes.CDLL(sys.argv[1] + "/libffi_c.so")
lib.noop.argtypes = []; lib.noop.restype = None
lib.add3.argtypes = [ctypes.c_double] * 3; lib.add3.restype = ctypes.c_double
ffi = FFI(); ffi.cdef("void noop(void); double add3(double, double, double);"); abi = ffi.dlopen(sys.argv[1] + "/libffi_c.so")
def py_noop(): pass
def py_add3(a, b, c): return a + b + c
@numba.njit
def nb_noop(): pass
@numba.njit
def nb_add3(a, b, c): return a + b + c
nb_noop(); nb_add3(1.0, 2.0, 3.0)
impls = {
    "pure Python function": (py_noop, py_add3),
    "CPython C API (METH_NOARGS / METH_FASTCALL)": (capi_mod.noop, capi_mod.add3),
    "Cython 3.3 def": (cy_mod.noop, cy_mod.add3),
    "nanobind 3.1": (nb_mod.noop, nb_mod.add3),
    "PyO3 0.30": (pyo3_mod.noop, pyo3_mod.add3),
    "cffi 2.1 API mode (compiled)": (cffi_api_mod.lib.noop, cffi_api_mod.lib.add3),
    "cffi 2.1 ABI mode (dlopen)": (abi.noop, abi.add3),
    "ctypes": (lib.noop, lib.add3),
    "Numba 0.68 njit dispatch": (nb_noop, nb_add3),
}
N = 1_000_000
empty = min(timeit.repeat("pass", number=N, repeat=7)) / N * 1e9
out = {"empty loop ns": round(empty, 1)}
for name, (noop, add3) in impls.items():
    a = statistics.median(timeit.repeat("f()", globals={"f": noop}, number=N, repeat=7)) / N * 1e9
    b = statistics.median(timeit.repeat("f(1.0, 2.0, 3.0)", globals={"f": add3}, number=N, repeat=7)) / N * 1e9
    out[name] = {"noop ns": round(a, 1), "add3(float, float, float) ns": round(b, 1)}
    print(f"{name:45s} noop {a:7.1f} ns   add3 {b:7.1f} ns", flush=True)
json.dump(out, open(sys.argv[1] + "/ffi.json", "w"), indent=1)
