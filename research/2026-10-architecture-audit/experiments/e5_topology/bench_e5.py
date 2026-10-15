"""E5: edge topology of a ~200k-triangle mesh: EditableMesh.from_mesh vs NumPy vs C vs Rust, same numbering."""
import ctypes, json, statistics, sys, time
import numpy as np
from aether3d import create_sphere, Mesh
from aether3d.editing import EditableMesh
E = sys.argv[1]
src = create_sphere(1.0, 448, 224)
mesh = Mesh(src.positions, src.triangles)  # topology only: no normals/uvs
out = {"triangles": int(mesh.triangles.shape[0]), "vertices": int(mesh.positions.shape[0])}

def weld(mesh):
    """EditableMesh.from_mesh's exact weld (editing/core.py:312-329)."""
    keys = np.ascontiguousarray(mesh.positions + np.float32(0.0))
    keys = keys.view(np.dtype((np.void, keys.dtype.itemsize * 3))).ravel()
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first, kind="stable"); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    tris = rank[inverse][mesh.triangles.astype(np.int64)]
    bad = (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
    return np.ascontiguousarray(tris[~bad])

def numpy_edges(loops):
    F = loops.shape[0]
    a = loops; b = np.roll(loops, -1, axis=1)
    lo = np.minimum(a, b).ravel(); hi = np.maximum(a, b).ravel()
    key = (lo << 32) | hi
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable"); rank = np.empty_like(order); rank[order] = np.arange(order.size)
    eou = rank[inv.reshape(-1)]
    E_ = order.size
    perm = np.argsort(eou, kind="stable")
    start = np.zeros(E_ + 1, np.int64); np.cumsum(np.bincount(eou, minlength=E_), out=start[1:])
    return E_, eou, lo[first[order]], hi[first[order]], start, perm // 3

def native(path, extra=()):
    lib = ctypes.CDLL(path); f = lib.build_edges
    f.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 6 + [ctypes.c_int32] * len(extra); f.restype = ctypes.c_int64
    def run(loops):
        F = loops.shape[0]; m = 3 * F
        eou, lo, hi, start, faces = np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m, np.int64), np.empty(m + 1, np.int64), np.empty(m, np.int64)
        n = f(F, loops.ctypes.data, eou.ctypes.data, lo.ctypes.data, hi.ctypes.data, start.ctypes.data, faces.ctypes.data, *extra)
        return n, eou, lo[:n], hi[:n], start[:n + 1], faces
    return run

def med(fn, k):
    fn(); ts = []
    for _ in range(k):
        t = time.perf_counter(); r = fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts), r

# Aether3D (current), topology only and the default path with normals/uvs/quads
t_a3d, em = med(lambda: EditableMesh.from_mesh(mesh, quads=False), 3)
out["EditableMesh.from_mesh(quads=False) s"] = round(t_a3d, 3)
t_full, _ = med(lambda: EditableMesh.from_mesh(src), 1)
out["EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s"] = round(t_full, 3)
ref_lo = np.array([em._edge_verts[e][0] for e in sorted(em._edge_verts)]); ref_hi = np.array([em._edge_verts[e][1] for e in sorted(em._edge_verts)])
assert sorted(em._edge_verts) == list(range(len(ref_lo)))
ref_faces = [em._edge_faces[e] for e in range(len(ref_lo))]
t_weld, loops = med(lambda: weld(mesh), 5)
out["weld (NumPy, shared) ms"] = round(t_weld * 1e3, 2)
impls = {"NumPy (vectorized)": numpy_edges, "C (open addressing)": native(f"{E}/libtopo_c.so"),
         "Rust (std HashMap, SipHash)": native(f"{E}/libtopo_rs.so", (0,)), "Rust (std HashMap, fast hasher)": native(f"{E}/libtopo_rs.so", (1,))}
for name, fn in impls.items():
    t, (n, eou, lo, hi, start, faces) = med(lambda: fn(loops), 7)
    ok = n == len(ref_lo) and np.array_equal(lo, ref_lo) and np.array_equal(hi, ref_hi) and \
         all(list(faces[start[e]:start[e + 1]]) == ref_faces[e] for e in range(0, n, 97))
    out[f"{name} ms"] = round(t * 1e3, 2); out[f"{name} matches Aether3D edge ids/faces"] = bool(ok)
print(json.dumps(out, indent=1)); json.dump(out, open(E + "/e5.json", "w"), indent=1)
