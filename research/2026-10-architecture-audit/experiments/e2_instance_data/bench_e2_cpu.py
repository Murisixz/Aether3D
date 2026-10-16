"""E2 (CPU): per-frame instance preparation, current path vs flat arrays, full vs dirty."""
import ctypes, json, statistics, sys, time
import numpy as np
import numba
from aether3d import Scene
from aether3d.rendering.scene_data import extract_render_scene, world_matrices
from aether3d.rendering.gpu.resources import draw_order, pack_instances, INSTANCE_FLOATS

E = sys.argv[1]
def med(fn, k=9):
    fn(); ts = []
    for _ in range(k):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

s = Scene("s")
s.execute_plan({"format": "aether3d.scene-plan", "version": 1, "meshes": {"block": {"primitive": "cube", "size": 0.9}},
                "objects": {"blocks": {"mesh": "@block", "grid": {"count": [100, 100]}, "position": [0, 0, 0.45]}}})
s.add_directional_light("Sun", direction=(-0.4, 0.6, -1))
cam = s.add_camera("Camera", fov=50); cam.frame(direction=(-0.5, 0.8, -0.6), aspect=16 / 9)
out = {}
out["world_matrices_ms"] = med(lambda: world_matrices(s)) * 1e3
out["extract_render_scene_ms"] = med(lambda: extract_render_scene(s, cam)) * 1e3
rs = extract_render_scene(s, cam)
eye = rs.camera.frame.position
out["draw_order_ms"] = med(lambda: draw_order(rs, "back")) * 1e3
order, draws = draw_order(rs, "back")
out["pack_instances_ms"] = med(lambda: pack_instances(rs, order, eye)) * 1e3
ref = pack_instances(rs, order, eye)
n = len(rs.instances)
# flat arrays (what a retained, data-oriented scene would already hold)
world = np.ascontiguousarray(np.stack([i.world for i in rs.instances]))
normal = np.ascontiguousarray(np.stack([i.normal_matrix for i in rs.instances]))
color = np.ascontiguousarray(np.stack([i.base_color for i in rs.instances]).astype(np.float32))
spec = np.array([i.specular for i in rs.instances], dtype=np.float32)
shin = np.array([i.shininess for i in rs.instances], dtype=np.float32)
flags = ref.view(np.uint32)[:, 29].copy()
flags_by_instance = np.empty(n, dtype=np.uint32); flags_by_instance[order] = flags
mesh = np.array([i.mesh for i in rs.instances], dtype=np.int64)

def numpy_pack():
    o = np.argsort(mesh, kind="stable")  # draw order: by mesh (cull/mirror keys equal here)
    d = np.zeros((n, INSTANCE_FLOATS), dtype=np.float32)
    w = world[o]
    model = w[:, :3, :4].copy(); model[:, :, 3] -= eye
    d[:, 0:12] = model.reshape(n, 12)
    nm = normal[o]
    for row in range(3):
        d[:, 4 * (3 + row): 4 * (3 + row) + 3] = nm[:, row]
    d[:, 24:27] = color[o]; d[:, 27] = spec[o]; d[:, 28] = shin[o]
    wd = d.view(np.uint32); wd[:, 29] = flags_by_instance[o]; wd[:, 30] = o
    return d

@numba.njit(cache=False)
def _numba_pack(order, world, normal, color, spec, shin, flags, eye, d, wd):
    for k in range(order.shape[0]):
        s_ = order[k]
        for i in range(3):
            for j in range(4):
                v = world[s_, i, j]
                if j == 3:
                    v -= eye[i]
                d[k, 4 * i + j] = np.float32(v)
            for j in range(3):
                d[k, 12 + 4 * i + j] = np.float32(normal[s_, i, j])
        d[k, 24] = color[s_, 0]; d[k, 25] = color[s_, 1]; d[k, 26] = color[s_, 2]
        d[k, 27] = spec[s_]; d[k, 28] = shin[s_]
        wd[k, 29] = flags[s_]; wd[k, 30] = np.uint32(s_)
def numba_pack():
    o = np.argsort(mesh, kind="stable")
    d = np.zeros((n, INSTANCE_FLOATS), dtype=np.float32)
    _numba_pack(o, world, normal, color, spec, shin, flags_by_instance, eye, d, d.view(np.uint32))
    return d

lib = ctypes.CDLL(f"{E}/libpack.so")
lib.pack_all.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 9; lib.pack_all.restype = None
lib.pack_dirty.argtypes = [ctypes.c_int64] + [ctypes.c_void_p] * 10; lib.pack_dirty.restype = None
eye64 = np.ascontiguousarray(eye, dtype=np.float64)
def c_pack():
    o = np.argsort(mesh, kind="stable")
    d = np.empty((n, INSTANCE_FLOATS), dtype=np.float32)
    lib.pack_all(n, o.ctypes.data, world.ctypes.data, normal.ctypes.data, color.ctypes.data, spec.ctypes.data,
                 shin.ctypes.data, flags_by_instance.ctypes.data, eye64.ctypes.data, d.ctypes.data)
    return d
for label, fn in (("numpy", numpy_pack), ("numba", numba_pack), ("c", c_pack)):
    got = fn()
    assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), label
    out[f"full_pack_{label}_ms"] = med(fn) * 1e3
# dirty updates into retained records (slot = position in draw order)
slot_of = np.empty(n, dtype=np.int64); slot_of[order] = np.arange(n)
retained = ref.copy()
rng = np.random.default_rng(0)
for m in (1, 10, 100, 1000):
    idx = np.sort(rng.choice(n, m, replace=False)).astype(np.int64); slots = slot_of[idx]
    def numpy_dirty():
        w = world[idx]; model = w[:, :3, :4].copy(); model[:, :, 3] -= eye
        retained[slots, 0:12] = model.reshape(m, 12)
        for row in range(3):
            retained[slots, 4 * (3 + row): 4 * (3 + row) + 3] = normal[idx, row]
    def c_dirty():
        lib.pack_dirty(m, idx.ctypes.data, slots.ctypes.data, world.ctypes.data, normal.ctypes.data, color.ctypes.data,
                       spec.ctypes.data, shin.ctypes.data, flags_by_instance.ctypes.data, eye64.ctypes.data, retained.ctypes.data)
    out[f"dirty_{m}_numpy_us"] = med(numpy_dirty) * 1e6
    out[f"dirty_{m}_c_us"] = med(c_dirty) * 1e6
assert np.array_equal(retained.view(np.uint32), ref.view(np.uint32))
print(json.dumps({k: round(v, 3) for k, v in out.items()}, indent=1))
