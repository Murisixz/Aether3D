"""Shared E3 setup: Aether3D's own benchmark scenes, rays and accel arrays, plus ctypes structs."""
import ctypes, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "benchmarks"))
from bench_raytrace import sphere_scene, instances_scene, camera_rays, surface_rays  # noqa: E402
from aether3d.rendering.raytrace import BLASCache, SceneAccel  # noqa: E402
from aether3d.rendering.scene_data import extract_render_scene  # noqa: E402

P = ctypes.c_void_p
class Accel(ctypes.Structure):
    _fields_ = [(k, P) for k in ("child", "child_lo", "child_hi", "kind", "first", "count", "v0", "v1", "v2",
                                 "tri_prim", "slot_inst", "inst_lo", "inst_hi", "inst_root", "to_object",
                                 "root_lo", "root_hi")] + [("stack_size", ctypes.c_int64)]
class Rays(ctypes.Structure):
    _fields_ = [("n", ctypes.c_int64)] + [(k, P) for k in ("o", "d", "tmin", "tmax", "skip_inst", "skip_tri",
                                                            "t", "inst", "prim", "b")]
DT = {"child": np.int64, "child_lo": np.float32, "child_hi": np.float32, "kind": np.int8, "first": np.int64,
      "count": np.int64, "v0": np.float32, "v1": np.float32, "v2": np.float32, "tri_prim": np.int64,
      "slot_inst": np.int64, "inst_lo": np.float64, "inst_hi": np.float64, "inst_root": np.int64,
      "to_object": np.float64, "root_lo": np.float64, "root_hi": np.float64}

def flat(accel):
    arrs = {}
    for k, dt in DT.items():
        src = getattr(accel, k)
        a = np.ascontiguousarray(src, dtype=dt)
        assert np.array_equal(a, src), k  # no value changed by the conversion
        arrs[k] = a
    return arrs

def accel_struct(arrs, stack_size):
    s = Accel(**{k: a.ctypes.data for k, a in arrs.items()}, stack_size=stack_size)
    return s

class RayBatch:
    def __init__(self, o, d, skip_inst=None, skip_tri=None):
        n = o.shape[0]
        self.o = np.ascontiguousarray(o, dtype=np.float64); self.d = np.ascontiguousarray(d, dtype=np.float64)
        self.tmin = np.zeros(n); self.tmax = np.full(n, np.inf)
        self.skip_inst = None if skip_inst is None else np.ascontiguousarray(skip_inst, dtype=np.int64)
        self.skip_tri = None if skip_tri is None else np.ascontiguousarray(skip_tri, dtype=np.int64)
        self.n = n
    def outputs(self):
        n = self.n
        return np.empty(n), np.empty(n, np.int64), np.empty(n, np.int64), np.empty((n, 3))
    def struct(self, outs):
        t, inst, prim, b = outs
        return Rays(self.n, self.o.ctypes.data, self.d.ctypes.data, self.tmin.ctypes.data, self.tmax.ctypes.data,
                    None if self.skip_inst is None else self.skip_inst.ctypes.data,
                    None if self.skip_tri is None else self.skip_tri.ctypes.data,
                    t.ctypes.data, inst.ctypes.data, prim.ctypes.data, b.ctypes.data)

LAST_RS = {}
def scenes():
    out = {}
    for name, build, eye in (("sphere 100k tris", lambda: sphere_scene(316), (0, -4, 0)),
                             ("400 cube instances", instances_scene, (0, -25, 0)),
                             ("sphere 1M tris", lambda: sphere_scene(1000), (0, -4, 0))):
        scene = build()
        rs = extract_render_scene(scene, scene.active_camera)
        accel = SceneAccel.build(rs, BLASCache())
        LAST_RS[name] = rs
        o, d = camera_rays(eye, 640, 480)
        so, sd, si, sp_ = surface_rays(accel, o, d, np.random.default_rng(0))
        out[name] = (accel, {"camera 640x480": RayBatch(o, d), "bounce (diffuse)": RayBatch(so, sd, si, sp_)})
    return out

def reference(accel, rb):
    h = accel.closest_hit(rb.o, rb.d, rb.tmin, rb.tmax, rb.skip_inst, rb.skip_tri)
    return h.t, h.inst, h.prim, h.b

def compare(ref, got):
    t0, i0, p0, b0 = ref; t1, i1, p1, b1 = got
    return {"t_bitwise": bool(np.array_equal(t0.view(np.int64), t1.view(np.int64))),
            "inst_equal": bool(np.array_equal(i0, i1)), "prim_equal": bool(np.array_equal(p0, p1)),
            "bary_bitwise": bool(np.array_equal(b0.view(np.int64), b1.view(np.int64))),
            "mismatched_rays": int(np.count_nonzero((i0 != i1) | (p0 != p1) | (t0.view(np.int64) != t1.view(np.int64))))}
