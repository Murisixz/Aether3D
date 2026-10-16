"""E1: world matrices of a hierarchy, same arithmetic in every implementation."""
import ctypes, json, statistics, sys, time
import numpy as np

E = sys.argv[1]
sys.path.insert(0, E)

def make(n, seed=1, max_depth=8):
    rng = np.random.default_rng(seed)
    parent = np.full(n, -1, dtype=np.int64); depth = np.zeros(n, dtype=np.int64)
    for k in range(1, n):
        if rng.random() < 0.9:
            p = int(rng.integers(0, k))
            if depth[p] < max_depth:
                parent[k] = p; depth[k] = depth[p] + 1
    pos = rng.uniform(-10, 10, (n, 3))
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    scale = rng.uniform(0.5, 2.0, (n, 3))
    return parent, np.ascontiguousarray(pos), np.ascontiguousarray(q), np.ascontiguousarray(scale), depth

def numpy_level(parent, pos, quat, scale, depth):
    """Aether3D's algorithm (rendering/scene_data.py) on flat arrays."""
    n = len(parent)
    x, y, z, w = quat.T
    r = np.empty((n, 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z); r[:, 0, 1] = 2 * (x * y - z * w); r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w); r[:, 1, 1] = 1 - 2 * (x * x + z * z); r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w); r[:, 2, 1] = 2 * (y * z + x * w); r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    m = np.zeros((n, 4, 4)); m[:, 3, 3] = 1.0
    m[:, :3, :3] = r * scale[:, None, :]; m[:, :3, 3] = pos
    for level in range(1, int(depth.max()) + 1):
        rows = np.flatnonzero(depth == level)
        a, b = m[parent[rows]], m[rows]
        out = np.empty_like(b)
        for i in range(4):
            for j in range(4):
                out[:, i, j] = a[:, i, 0] * b[:, 0, j] + a[:, i, 1] * b[:, 1, j] + a[:, i, 2] * b[:, 2, j] + a[:, i, 3] * b[:, 3, j]
        m[rows] = out
    return m

def pure_python(parent, pos, quat, scale, depth):
    P, Q, S, par = pos.tolist(), quat.tolist(), scale.tolist(), parent.tolist()
    out = [None] * len(par)
    for k in range(len(par)):
        x, y, z, w = Q[k]; s0, s1, s2 = S[k]; p = P[k]
        loc = [(1 - 2 * (y * y + z * z)) * s0, (2 * (x * y - z * w)) * s1, (2 * (x * z + y * w)) * s2, p[0],
               (2 * (x * y + z * w)) * s0, (1 - 2 * (x * x + z * z)) * s1, (2 * (y * z - x * w)) * s2, p[1],
               (2 * (x * z - y * w)) * s0, (2 * (y * z + x * w)) * s1, (1 - 2 * (x * x + y * y)) * s2, p[2],
               0.0, 0.0, 0.0, 1.0]
        if par[k] < 0:
            out[k] = loc; continue
        a = out[par[k]]
        out[k] = [a[4 * i] * loc[j] + a[4 * i + 1] * loc[4 + j] + a[4 * i + 2] * loc[8 + j] + a[4 * i + 3] * loc[12 + j]
                  for i in range(4) for j in range(4)]
    return np.array(out).reshape(-1, 4, 4)

import numba
@numba.njit(cache=False, fastmath=False)
def _numba_kernel(parent, pos, quat, scale, out):
    loc = np.empty(16)
    for k in range(parent.shape[0]):
        x, y, z, w = quat[k, 0], quat[k, 1], quat[k, 2], quat[k, 3]
        s0, s1, s2 = scale[k, 0], scale[k, 1], scale[k, 2]
        loc[0] = (1 - 2 * (y * y + z * z)) * s0; loc[1] = (2 * (x * y - z * w)) * s1; loc[2] = (2 * (x * z + y * w)) * s2; loc[3] = pos[k, 0]
        loc[4] = (2 * (x * y + z * w)) * s0; loc[5] = (1 - 2 * (x * x + z * z)) * s1; loc[6] = (2 * (y * z - x * w)) * s2; loc[7] = pos[k, 1]
        loc[8] = (2 * (x * z - y * w)) * s0; loc[9] = (2 * (y * z + x * w)) * s1; loc[10] = (1 - 2 * (x * x + y * y)) * s2; loc[11] = pos[k, 2]
        loc[12] = 0.0; loc[13] = 0.0; loc[14] = 0.0; loc[15] = 1.0
        p = parent[k]
        if p < 0:
            for t in range(16):
                out[k, t // 4, t % 4] = loc[t]
            continue
        for i in range(4):
            for j in range(4):
                out[k, i, j] = out[p, i, 0] * loc[j] + out[p, i, 1] * loc[4 + j] + out[p, i, 2] * loc[8 + j] + out[p, i, 3] * loc[12 + j]

def numba_impl(parent, pos, quat, scale, depth):
    out = np.empty((len(parent), 4, 4)); _numba_kernel(parent, pos, quat, scale, out); return out

import world_cy
def cython_impl(parent, pos, quat, scale, depth):
    out = np.empty((len(parent), 4, 4)); world_cy.world_matrices(parent, pos, quat, scale, out); return out

def native(libname):
    lib = ctypes.CDLL(f"{E}/{libname}")
    f = lib.world_matrices
    f.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 5; f.restype = None
    def impl(parent, pos, quat, scale, depth):
        out = np.empty((len(parent), 4, 4))
        f(len(parent), parent.ctypes.data, pos.ctypes.data, quat.ctypes.data, scale.ctypes.data, out.ctypes.data)
        return out
    return impl, lib

impls = {"numpy (Aether3D algorithm, flat arrays)": numpy_level, "pure Python": pure_python,
         "Numba 0.68": numba_impl, "Cython 3.3": cython_impl}
libs = {}
for label, so in [("C gcc -O3 (no FMA contraction)", "libworld_c.so"), ("C clang -O3 (no FMA contraction)", "libworld_clang.so"),
                  ("C gcc -O3 FMA contraction allowed", "libworld_c_fma.so"), ("Rust 1.99 (release, LTO)", "libworld_rs.so"),
                  ("Zig 0.17 ReleaseFast", "libworld_zig.so"), ("Go 1.27 (cgo c-shared)", "libworld_go.so"),
                  ("C# .NET 10 Native AOT (x86-64-v3)", "world_cs4.so")]:
    impls[label], libs[label] = native(so)

results = {}
for n in (10_000, 100_000):
    data = make(n)
    ref = numpy_level(*data)
    rows = {}
    for label, fn in impls.items():
        if label == "pure Python" and n > 10_000:
            continue
        fn(*data)  # warm-up (JIT for Numba)
        reps = 3 if label == "pure Python" else 15
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); out = fn(*data); ts.append(time.perf_counter() - t)
        diff = float(np.max(np.abs(out - ref)))
        rows[label] = {"median_ms": round(statistics.median(ts) * 1e3, 3), "min_ms": round(min(ts) * 1e3, 3),
                       "bitwise_equal": bool(np.array_equal(out, ref)), "max_abs_diff": diff}
    results[n] = rows
# per-call FFI overhead of an empty function through ctypes
noop = {}
for label, lib in libs.items():
    f = lib.noop; f.argtypes = []; f.restype = None
    for _ in range(1000): f()
    t = time.perf_counter()
    for _ in range(200000): f()
    noop[label] = round((time.perf_counter() - t) / 200000 * 1e9)
results["ctypes_noop_call_ns"] = noop
print(json.dumps(results, indent=1))
