"""Embree 4.4.1 on the E3 scenes and rays: throughput, build time and agreement with Aether3D's float64 tracer."""
import ctypes, json, statistics, sys, time
import numpy as np
E = sys.argv[1]
sys.path.insert(0, E)
from e3common import scenes, reference
from aether3d.rendering.raytrace import BLASCache, SceneAccel  # noqa
lib = ctypes.CDLL(f"{E}/libembree_bench.so")
V = ctypes.c_void_p
lib.em_device.restype = V
lib.em_mesh.restype = V; lib.em_mesh.argtypes = [V, ctypes.c_int64, V, ctypes.c_int64, V, ctypes.c_int]
lib.em_instances.restype = V; lib.em_instances.argtypes = [V, ctypes.c_int64, V, V, V, ctypes.c_int]
lib.em_trace.restype = None; lib.em_trace.argtypes = [V, ctypes.c_int64, V, V, V, V, ctypes.c_int, V, V, V, ctypes.c_int]
dev = lib.em_device()
out = {}
for sname, (accel, batches) in scenes().items():
    from aether3d.rendering.scene_data import extract_render_scene  # noqa
    # rebuild the same RenderScene the accel came from
    import e3common
    rs_meshes = [b for b in accel.blas]
    # mesh data: positions/triangles are not kept in MeshBLAS; recover from the scene builders
    out[sname] = {}
    scene_rs = e3common.LAST_RS[sname]
    meshes = []
    t0 = time.perf_counter()
    for m in scene_rs.meshes:
        pos = np.ascontiguousarray(m.positions, dtype=np.float32)
        tri = np.ascontiguousarray(m.triangles, dtype=np.uint32)
        meshes.append(lib.em_mesh(dev, pos.shape[0], pos.ctypes.data, tri.shape[0], tri.ctypes.data, 1))
    build_mesh = time.perf_counter() - t0
    xfm = np.ascontiguousarray(np.stack([i.world[:3, :4] for i in scene_rs.instances]), dtype=np.float32)
    mesh_of = np.array([i.mesh for i in scene_rs.instances], dtype=np.int64)
    mesh_arr = (V * len(meshes))(*meshes)
    t0 = time.perf_counter()
    top = lib.em_instances(dev, len(mesh_of), mesh_of.ctypes.data, mesh_arr, xfm.ctypes.data, 1)
    build_top = time.perf_counter() - t0
    out[sname]["build_seconds (all threads, TBB)"] = round(build_mesh + build_top, 4)
    out[sname]["aether3d_blas_build_seconds (NumPy, 1 thread)"] = round(sum(b.build_seconds for b in accel.blas), 3)
    for bname, rb in batches.items():
        ref_t, ref_i, ref_p, _ = reference(accel, rb)
        row = {}
        for nt in (1, 8, 20):
            t = np.empty(rb.n); inst = np.empty(rb.n, np.int64); prim = np.empty(rb.n, np.int64)
            si = None if rb.skip_inst is None else rb.skip_inst.ctypes.data
            st = None if rb.skip_tri is None else rb.skip_tri.ctypes.data
            call = lambda: lib.em_trace(top, rb.n, rb.o.ctypes.data, rb.d.ctypes.data, si, st, 1, t.ctypes.data, inst.ctypes.data, prim.ctypes.data, nt)
            call(); ts = []
            for _ in range(5 if nt > 1 else 3):
                a = time.perf_counter(); call(); ts.append(time.perf_counter() - a)
            row[f"x{nt}"] = round(rb.n / statistics.median(ts) / 1e6, 2)
        hit_ref = ref_i >= 0; hit_em = inst >= 0
        both = hit_ref & hit_em
        rel = np.abs(t[both] - ref_t[both]) / ref_t[both]
        row["agreement"] = {"hit/miss same": round(float(np.mean(hit_ref == hit_em)), 6),
                            "same triangle (of rays both hit)": round(float(np.mean((ref_p[both] == prim[both]) & (ref_i[both] == inst[both]))), 6),
                            "t rel. diff median": float(np.median(rel)) if rel.size else 0.0, "t rel. diff max": float(rel.max()) if rel.size else 0.0}
        out[sname][bname] = row
        print(sname, bname, json.dumps(row), flush=True)
json.dump(out, open(f"{E}/embree_results.json", "w"), indent=1)
