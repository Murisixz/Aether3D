"""Phase 5 prototype, step 1: one showcase render's GPU path tracer inputs and reference.

Renders a showcase scene with ``backend="gpu_path"`` at the settings of
``benchmarks/bench_gpu_pathtrace.py`` (the size and samples of its CPU record) and
captures what ``render_scene`` (``rendering/gpu/pathtrace.py``) binds and dispatches:
every buffer, the base parameters, the tiles and the samples per dispatch, the
accumulation planes it reads back and its stage timings -- of a first render (the BLAS
cache empty, the pipeline compiled) and of a second, warm one. Writes them to OUT/SCENE
with the kernels the Rust host (``host/``) runs:

- ``sw.wgsl``: ``module_source("pathtrace")``, unchanged -- the production kernel;
- ``hw.wgsl``: the same module with ``trace_ray`` on wgpu ray queries and nothing else
  changed (``HW_TRACE`` below);
- ``agree.wgsl``: both traversals in one module and an entry point that traces the same
  rays with each (camera rays, bounce rays leaving the camera hits, short any-hit rays);

and each mesh's float32 vertices and indices and each instance's camera-relative
transform for the acceleration structures.

    python export.py OUT SCENE [--quick]

SCENE is one of ``bench_gpu_pathtrace.CASES`` (cornell, metals, window-room, castle,
hero) or ``castle-ao`` (that benchmark's AO-only castle render: camera and AO rays).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))

import bench_gpu_pathtrace as bench  # noqa: E402  (puts examples/ on the path)

import aether3d  # noqa: E402
from aether3d.rendering.gpu import pathtrace as gp  # noqa: E402
from aether3d.rendering.gpu.materials import image_key  # noqa: E402
from aether3d.rendering.gpu.shaders import module_source  # noqa: E402

# render_scene's dispatch(): the parameter words it sets per dispatch, and the modes.
WORD_FIRST, WORD_COUNT, WORD_MODE, WORD_TILE = 40, 41, 47, 54
MODE_PATHS, MODE_CENTER = 0, 1
FALLBACK_SLOT = 7  # counters[7]: unused by the kernel (ray counters are 0-6, C_VISIBLE 8+)
assert len(gp.RAY_COUNTERS) == FALLBACK_SLOT < gp.COUNTER_SLOTS

HW_TRACE = """
// Phase 5 prototype: trace_ray on wgpu ray queries, with the semantics of the software
// traversal it replaces -- the closest hit (or any hit) with tmin < t < tmax, never the
// triangle the ray leaves from. An opaque query finds the hit; only when that hit is the
// skipped triangle or lies at t <= tmin does a second, non-opaque query confirm the
// candidates one by one (counters[FALLBACK_SLOT] counts those rays). Hit.b holds the
// weights of the triangle's three corners, as tri_hit's.
fn trace_ray(o_w: vec3<f32>, d_w: vec3<f32>, tmin: f32, tmax: f32, skip_inst: u32, skip_prim: u32, any_hit: bool) -> Hit {
  var best = Hit(tmax, PT_NONE, PT_NONE, vec3<f32>(0.0));
  if (params.instance_count == 0u) {
    return best;
  }
  var first = RAY_FLAG_FORCE_OPAQUE;
  var exact = RAY_FLAG_FORCE_NO_OPAQUE;
  if (any_hit) {
    first = first | RAY_FLAG_TERMINATE_ON_FIRST_HIT;
    exact = exact | RAY_FLAG_TERMINATE_ON_FIRST_HIT;
  }
  var rq: ray_query;
  rayQueryInitialize(&rq, acc, RayDesc(first, 0xffu, tmin, tmax, o_w, d_w));
  rayQueryProceed(&rq);
  var x = rayQueryGetCommittedIntersection(&rq);
  if (x.kind != RAY_QUERY_INTERSECTION_NONE
      && ((x.instance_custom_data == skip_inst && x.primitive_index == skip_prim) || !(x.t > tmin))) {
    atomicAdd(&counters[FALLBACK_SLOTu], 1u);
    var rq2: ray_query;
    rayQueryInitialize(&rq2, acc, RayDesc(exact, 0xffu, tmin, tmax, o_w, d_w));
    while (rayQueryProceed(&rq2)) {
      let c = rayQueryGetCandidateIntersection(&rq2);
      if (!(c.instance_custom_data == skip_inst && c.primitive_index == skip_prim) && c.t > tmin) {
        rayQueryConfirmIntersection(&rq2);
      }
    }
    x = rayQueryGetCommittedIntersection(&rq2);
  }
  if (x.kind == RAY_QUERY_INTERSECTION_TRIANGLE) {
    let w = x.barycentrics;
    best = Hit(x.t, x.instance_custom_data, x.primitive_index, vec3<f32>(1.0 - w.x - w.y, w.x, w.y));
  }
  return best;
}
""".replace("FALLBACK_SLOT", str(FALLBACK_SLOT))

HW_HEADER = """enable wgpu_ray_query;
@group(2) @binding(0) var acc: acceleration_structure;
"""

AGREE_MAIN = """
// The agreement kernel: the same rays through trace_ray_sw (the production traversal)
// and trace_ray (ray queries). Per pixel: the camera ray of the path tracer's first
// sample (jittered as path_main's); then, from the software camera hit, a
// cosine-distributed bounce ray and a short any-hit ray (5 % of the scene's size), both
// leaving that triangle as the path tracer's do. (The pixel centers' hits are compared
// by the center pass of the renders.)
fn agree_write(k: u32, pixel: u32, h: Hit) {
  accum[plane(k, pixel)] = bitcast<f32>(h.inst);
  accum[plane(k + 1u, pixel)] = bitcast<f32>(h.prim);
  accum[plane(k + 2u, pixel)] = h.t;
}

@compute @workgroup_size(8, 8)
fn agree_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.tile_w || gid.y >= params.tile_h) {
    return;
  }
  let x = params.tile_x + gid.x;
  let y = params.tile_y + gid.y;
  let pixel = y * params.width + x;
  let key = path_key(pixel, 0u);
  var st = new_state(key);
  camera_ray(&st, RAY_CAMERA, x, y, rnd(key, 0u, SLOT_PIXEL), rnd(key, 0u, SLOT_PIXEL + 1u), false);
  let a = trace_ray_sw(st.ro, st.rd, st.rtmin, st.rtmax, PT_NONE, PT_NONE, false);
  let b = trace_ray(st.ro, st.rd, st.rtmin, st.rtmax, PT_NONE, PT_NONE, false);
  agree_write(0u, pixel, a);
  agree_write(3u, pixel, b);
  if (a.inst == PT_NONE) {
    return;
  }
  let zero = vec3<f32>(0.0);
  let it = interaction(a, a.t, st.rd, false, zero, zero, zero, zero, 0.0, 0.0);
  let side = it.ng * it.toward;
  let o = offset_origin(it.p, side);
  let frame = basis(side);
  let d = to_world(cosine_hemisphere(rnd(key, 1u, SLOT_BSDF), rnd(key, 1u, SLOT_BSDF + 1u)), frame, side);
  agree_write(6u, pixel, trace_ray_sw(o, d, 0.0, PT_INF, it.inst, it.prim, false));
  agree_write(9u, pixel, trace_ray(o, d, 0.0, PT_INF, it.inst, it.prim, false));
  let reach = 0.05 * length(params.root_hi - params.root_lo);
  let e = to_world(cosine_hemisphere(rnd(key, 1u, SLOT_AO), rnd(key, 1u, SLOT_AO + 1u)), frame, side);
  let f = trace_ray_sw(o, e, 0.0, reach, it.inst, it.prim, true);
  let g = trace_ray(o, e, 0.0, reach, it.inst, it.prim, true);
  accum[plane(12u, pixel)] = select(0.0, 1.0, f.inst != PT_NONE);
  accum[plane(13u, pixel)] = select(0.0, 1.0, g.inst != PT_NONE);
  accum[plane(14u, pixel)] = 1.0;
}
"""


def kernels() -> dict[str, str]:
    """sw.wgsl, hw.wgsl and agree.wgsl from the production module."""
    sw = module_source("pathtrace")
    start = sw.index("// Closest hit in (tmin, tmax) -- or any hit -- skipping the triangle")
    end = sw.index("// -- surface points")
    original = sw[start:end]
    assert original.count("fn trace_ray(") == 1
    hw = HW_HEADER + sw[:start] + HW_TRACE + "\n" + sw[end:]
    renamed = original.replace("fn trace_ray(", "fn trace_ray_sw(")
    agree = HW_HEADER + sw[:start] + renamed + HW_TRACE + "\n" + sw[end:] + AGREE_MAIN
    return {"sw": sw, "hw": hw, "agree": agree}


def cases() -> dict[str, tuple]:
    out = {}
    for name, module, record, path, options in bench.CASES:
        out[name.replace(" ", "-")] = (module, record, path, options, None)
    module, record, _, options, _ = out["castle"]
    out["castle-ao"] = (module, record, {"integrator": "ao"}, options, {"samples": 4, "distance": 1.0})
    return out


def texel_words(images) -> tuple[np.ndarray, dict]:
    """TexturePool.prepare's buffer contents for ``images`` (rendering/gpu/materials.py)."""
    chains = {}
    for image in images:
        chains.setdefault(image_key(image), image.chain)
    total = sum(sum(lv.shape[0] * lv.shape[1] for lv in c.levels) for c in chains.values())
    data = np.zeros(max(4 * total, 256) // 4, dtype=np.uint32)
    offsets, at = {}, 0
    for key, chain in chains.items():
        offsets[key] = at
        for level in chain.levels:
            t = level.reshape(-1, 4).astype(np.uint32)
            packed = t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16) | (t[:, 3] << 24)
            data[at : at + packed.shape[0]] = packed
            at += packed.shape[0]
    return data, offsets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("scene", choices=sorted(cases()))
    ap.add_argument("--quick", action="store_true", help="a quarter of the size, 1/8 samples")
    args = ap.parse_args()
    module, (file, image), path, options, ao = cases()[args.scene]
    cpu = bench.cpu_record(file, image)
    width, height, samples = cpu["width"], cpu["height"], cpu["samples"]
    if args.scene == "castle-ao":
        samples = max(1, samples // 8)
    if args.quick:
        width, height, samples = width // 4, height // 4, max(1, samples // 8)
    out = args.out / args.scene
    out.mkdir(parents=True, exist_ok=True)
    scene = bench.build(module, width, height)
    settings = dict(
        width=width, height=height, backend="gpu_path", path={"samples": samples, **path}, **options
    )
    if ao is not None:
        settings["ao"] = ao

    cap: dict = {}
    original = {
        name: getattr(gp, name)
        for name in ("render_scene", "pack_meshes", "pack_instances", "pack_material_data",
                     "pack_scene_words", "base_params", "_frame")
    }  # fmt: skip

    def keep(name):
        def wrapped(*a, **k):
            result = original[name](*a, **k)
            cap[name] = result
            if name == "base_params":
                cap["base_words"] = result.words.copy()
            return result

        return wrapped

    def frame(planes, done, *a, **k):
        cap["planes"] = np.array(planes, copy=True)
        cap["done"] = done
        return original["_frame"](planes, done, *a, **k)

    def render_scene(context, rs, *a, **k):
        cap["rs"] = rs
        pool = context.textures
        prepare = pool.prepare

        def checked(images):
            result = prepare(images)
            data, offsets = texel_words(images)
            assert result[1] == offsets, "the texture pool held other images: run in a new process"
            cap["texels"] = data
            return result

        pool.prepare = checked
        try:
            result = original["render_scene"](context, rs, *a, **k)
        finally:
            del pool.prepare
        cap["timings"] = dict(result.timings)
        cap["dispatch"] = dict(result.info["dispatch"])
        return result

    def timed_render():
        t0 = time.perf_counter()
        r = scene.render(**settings)
        return r, time.perf_counter() - t0

    for name in original:
        if name not in ("render_scene", "_frame"):
            setattr(gp, name, keep(name))
    gp.render_scene, gp._frame = render_scene, frame
    try:
        first, first_s = timed_render()
        captured = dict(cap)
        cap.clear()
        second, second_s = timed_render()
    finally:
        for name, fn in original.items():
            setattr(gp, name, fn)
    assert np.array_equal(captured["planes"].view(np.uint32), cap["planes"].view(np.uint32))
    warm_timings, warm_dispatch = cap["timings"], cap["dispatch"]

    rs = captured["rs"]
    meshes, tri_base = captured["pack_meshes"]
    instances = captured["pack_instances"]
    materials = captured["pack_material_data"]
    scene_words = captured["pack_scene_words"]
    per = captured["dispatch"]["samples_per_dispatch"]
    tiles = gp._tiles(width, height)
    assert captured["dispatch"] == warm_dispatch and len(tiles) == captured["dispatch"]["tiles"]
    assert captured["done"] == samples

    def put(name: str, array: np.ndarray) -> None:
        (out / f"{name}.bin").write_bytes(np.ascontiguousarray(array).tobytes())

    put("instances", instances)
    put("materials", materials.words)
    put("texels", captured["texels"])
    put("scene", scene_words)
    put("params", captured["base_words"])
    for name, a in zip(gp.MESH_BUFFERS, meshes, strict=True):
        put(f"mesh_{name}", a)
    for j, mesh in enumerate(rs.meshes):
        put(f"blas{j}_pos", np.asarray(mesh.positions, np.float32).reshape(-1, 3))
        put(f"blas{j}_idx", np.asarray(mesh.triangles, np.uint32).reshape(-1, 3))
    put("inst_mesh", np.array([i.mesh for i in rs.instances], np.uint32))
    put("inst_xfm", instances[: len(rs.instances), 0:12])  # camera-relative object -> world rows
    put("planes_python", captured["planes"])
    for name, text in kernels().items():
        (out / f"{name}.wgsl").write_text(text)
    plan = {
        "width": [width],
        "height": [height],
        "samples": [samples],
        "per": [per],
        "acc_planes": [gp.ACC_PLANES],
        "counter_slots": [gp.COUNTER_SLOTS],
        "fallback_slot": [FALLBACK_SLOT],
        "instances": [len(rs.instances)],
        "meshes": [len(rs.meshes)],
        "words": [WORD_FIRST, WORD_COUNT, WORD_MODE, WORD_TILE],
        "modes": [MODE_PATHS, MODE_CENTER],
        "tiles": [len(tiles), *[v for t in tiles for v in t]],
    }
    (out / "plan.txt").write_text(
        "".join(f"{k} {' '.join(str(int(v)) for v in vs)}\n" for k, vs in plan.items())
    )

    def python_run(r, seconds, timings):
        return {
            "wall_s": round(seconds, 4),
            "render_scene_s": {k: round(v, 5) for k, v in timings.items()},
            "rays": r.stats["rays"],
            "mrays_per_s": round(r.stats["rays_per_second"] / 1e6, 2),
        }

    meta = {
        "scene": args.scene,
        "aether3d": aether3d.__version__,
        "adapter": first.backend["adapter"]["name"],
        "settings": {k: v for k, v in settings.items() if k != "backend"},
        "triangles": int(rs.triangle_count),
        "instances": len(rs.instances),
        "meshes": len(rs.meshes),
        "tiles": len(tiles),
        "samples_per_dispatch": per,
        "flags": int(captured["base_words"][35]),  # base_params: word 35, FLAG_PASS_THROUGH = 2
        "pass_through": bool(captured["base_words"][35] & 2),
        "cull": int(captured["base_words"][46]),  # CULLS: none 0, back 1, front 2
        "python_cold": python_run(first, first_s, captured["timings"]),
        "python_warm": python_run(second, second_s, warm_timings),
        "python_stats": {
            k: v for k, v in first.stats.items() if isinstance(v, (int, float, str, bool))
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("scene", "triangles", "instances", "python_warm")}))


if __name__ == "__main__":
    main()
