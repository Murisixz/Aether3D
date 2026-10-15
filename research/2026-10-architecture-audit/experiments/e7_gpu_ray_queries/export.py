"""Export E7 inputs: Aether3D's GPU BVH words (pathtrace layout, eye = 0), meshes, instances, rays, WGSL."""
import ctypes, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent / "e3_traversal"))
sys.path.insert(0, str(ROOT / "benchmarks"))
from bench_raytrace import sphere_scene, instances_scene, camera_rays  # noqa
from e3common import Accel, Rays, flat, accel_struct, RayBatch, compare  # noqa
from aether3d.rendering.raytrace import BLASCache, SceneAccel  # noqa
from aether3d.rendering.raytrace.accel import KIND_INNER, _round_out  # noqa
from aether3d.rendering.scene_data import extract_render_scene  # noqa
from aether3d.rendering.gpu.pathtrace import HEADER_WORDS, INST_WORDS  # noqa

SRC = (ROOT / "src/aether3d/rendering/gpu/wgsl/pathtrace.wgsl").read_text()

def words(accel, rs):
    """The first six regions of pack_scene_words (rendering/gpu/pathtrace.py) with eye = 0."""
    n = accel.child.shape[0]
    nodes = np.zeros((n, 4), np.uint32)
    inner = accel.kind == KIND_INNER
    nodes[:, 0] = np.where(inner, accel.child[:, 0], accel.first)
    nodes[:, 1] = np.where(inner, accel.child[:, 1], accel.count)
    nodes[:, 2] = accel.kind
    lo32, hi32 = accel.child_lo.copy(), accel.child_hi.copy()
    top = accel.top_nodes
    lo32[:top], hi32[:top] = _round_out(accel.child_lo[:top].astype(np.float64), accel.child_hi[:top].astype(np.float64))
    boxes = np.concatenate([lo32[:, 0], hi32[:, 0], lo32[:, 1], hi32[:, 1]], axis=1).astype(np.float32)
    tris = np.concatenate([accel.v0, accel.v1, accel.v2], axis=1).astype(np.float32)
    k = len(rs.instances)
    inst = np.zeros((k, INST_WORDS), np.uint32); fi = inst.view(np.float32)
    fi[:, 0:12] = accel.to_object.reshape(k, 12)
    blo, bhi = _round_out(accel.inst_lo, accel.inst_hi)
    fi[:, 12:15] = blo; fi[:, 15:18] = bhi
    inst[:, 18] = accel.inst_root
    regions = [nodes.ravel(), boxes.ravel().view(np.uint32), tris.ravel().view(np.uint32),
               accel.tri_prim.astype(np.uint32), accel.slot_inst.astype(np.uint32), inst.ravel(),
               np.zeros(24, np.uint32), np.zeros(12, np.uint32), np.zeros(1, np.uint32), np.full(1, 0xFFFFFFFF, np.uint32)]
    header = np.zeros(HEADER_WORDS, np.uint32); off = HEADER_WORDS
    for i, r in enumerate(regions):
        header[i] = off; off += r.size
    return np.concatenate([header, *regions]).astype(np.uint32)

def sw_shader():
    consts = "\n".join(l for l in SRC.splitlines() if l.startswith("const "))
    start = SRC.index("fn ptf(i: u32)"); end = SRC.index("// -- random numbers")
    helpers = SRC[start:end]
    start = SRC.index("// -- the BVH: two-level traversal"); end = SRC.index("// -- surface points")
    trace = SRC[start:end]
    return f"""// Benchmark kernel: Aether3D's own trace_ray (copied verbatim from pathtrace.wgsl).
{consts}
struct Params {{ root_lo: vec3<f32>, instance_count: u32, root_hi: vec3<f32>, n_rays: u32 }};
@group(0) @binding(0) var<storage, read> pt: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> hits: array<vec4<u32>>;
{helpers}
{trace}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = gid.x;
  if (i >= params.n_rays) {{ return; }}
  let o = rays[2u * i];
  let d = rays[2u * i + 1u];
  let h = trace_ray(o.xyz, d.xyz, o.w, d.w, PT_NONE, PT_NONE, false);
  hits[i] = vec4<u32>(h.inst, h.prim, bitcast<u32>(h.t), 0u);
}}
"""

HW = """enable wgpu_ray_query;
const PT_NONE: u32 = 0xffffffffu;
struct Params { root_lo: vec3<f32>, instance_count: u32, root_hi: vec3<f32>, n_rays: u32 };
@group(0) @binding(0) var acc: acceleration_structure;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> hits: array<vec4<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n_rays) { return; }
  let o = rays[2u * i];
  let d = rays[2u * i + 1u];
  var rq: ray_query;
  rayQueryInitialize(&rq, acc, RayDesc(RAY_FLAG_FORCE_OPAQUE, 0xffu, o.w, d.w, o.xyz, d.xyz));
  rayQueryProceed(&rq);
  let x = rayQueryGetCommittedIntersection(&rq);
  if (x.kind == RAY_QUERY_INTERSECTION_NONE) {
    hits[i] = vec4<u32>(PT_NONE, PT_NONE, bitcast<u32>(d.w), 0u);
  } else {
    hits[i] = vec4<u32>(x.instance_custom_data, x.primitive_index, bitcast<u32>(x.t), 0u);
  }
}
"""
(OUT / "sw.wgsl").write_text(sw_shader()); (OUT / "hw.wgsl").write_text(HW)

lib = ctypes.CDLL(str(HERE.parent / "e3_traversal" / "libtrace_c_omp.so"))
lib.closest_hit.argtypes = [ctypes.POINTER(Accel), ctypes.POINTER(Rays), ctypes.c_int64]; lib.closest_hit.restype = ctypes.c_int
def cpu(accel, rb):
    arrs = flat(accel); A = accel_struct(arrs, accel.stack_size); outs = rb.outputs()
    assert lib.closest_hit(ctypes.byref(A), ctypes.byref(rb.struct(outs)), 20) == 0
    return outs

TMAX = 3.0e38
manifest = {}
for name, build, eye in (("sphere1m", lambda: sphere_scene(1000), (0, -4, 0)), ("cubes400", instances_scene, (0, -25, 0)),
                         ("sphere100k", lambda: sphere_scene(316), (0, -4, 0))):
    scene = build(); rs = extract_render_scene(scene, scene.active_camera)
    accel = SceneAccel.build(rs, BLASCache())
    d = OUT / name; d.mkdir(exist_ok=True)
    words(accel, rs).tofile(d / "pt.bin")
    rlo, rhi = _round_out(accel.root_lo, accel.root_hi)
    params = np.zeros(8, np.uint32); pf = params.view(np.float32)
    pf[0:3] = rlo; params[3] = len(rs.instances); pf[4:7] = rhi
    params.tofile(d / "params.bin")
    meshes = []
    for j, m in enumerate(rs.meshes):
        np.ascontiguousarray(m.positions, np.float32).tofile(d / f"mesh{j}_pos.bin")
        np.ascontiguousarray(m.triangles, np.uint32).tofile(d / f"mesh{j}_idx.bin")
        meshes.append({"vertices": int(m.positions.shape[0]), "triangles": int(m.triangles.shape[0])})
    mesh_of = np.array([i.mesh for i in rs.instances], np.uint32); mesh_of.tofile(d / "inst_mesh.bin")
    np.ascontiguousarray(np.stack([i.world[:3, :4] for i in rs.instances]), np.float32).tofile(d / "inst_xfm.bin")
    # rays: 1920x1080 camera rays, then diffuse bounces leaving the camera hits (tmin 1e-4, no skip)
    o, dd = camera_rays(eye, 1920, 1080)
    cam = RayBatch(o, dd); cam.tmax[:] = TMAX
    t, inst, prim, _ = cpu(accel, cam)
    k = np.flatnonzero(inst >= 0)
    p = o[k] + dd[k] * t[k, None]
    rng = np.random.default_rng(0); dirs = rng.normal(size=p.shape); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    bounce = RayBatch(p, dirs); bounce.tmin[:] = 1e-4; bounce.tmax[:] = TMAX
    sets = {}
    for rname, rb in (("camera", cam), ("bounce", bounce)):
        packed = np.zeros((rb.n, 2, 4), np.float32)
        packed[:, 0, :3] = rb.o; packed[:, 0, 3] = rb.tmin; packed[:, 1, :3] = rb.d; packed[:, 1, 3] = rb.tmax
        packed.tofile(d / f"rays_{rname}.bin")
        # float64 reference on the float32-rounded rays the GPU sees
        r32 = RayBatch(packed[:, 0, :3].astype(np.float64), packed[:, 1, :3].astype(np.float64))
        r32.tmin[:] = packed[:, 0, 3]; r32.tmax[:] = packed[:, 1, 3]
        rt, ri, rp, _ = cpu(accel, r32)
        np.stack([np.where(ri >= 0, ri, 0xFFFFFFFF), np.where(rp >= 0, rp, 0xFFFFFFFF)], axis=1).astype(np.uint32).tofile(d / f"ref_{rname}.bin")
        rt.astype(np.float64).tofile(d / f"reft_{rname}.bin")
        sets[rname] = int(rb.n)
    manifest[name] = {"meshes": meshes, "instances": len(rs.instances), "rays": sets, "pt_words": int((d / "pt.bin").stat().st_size // 4),
                      "triangles": int(sum(m["triangles"] for m in meshes))}
    print(name, manifest[name], flush=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
