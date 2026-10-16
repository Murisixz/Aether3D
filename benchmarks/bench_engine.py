"""Engine-wide performance baseline: the workloads the performance modernization targets.

Run from the repository root:

    python benchmarks/bench_engine.py [--json OUT.json] [--label NAME] [--quick]
        [--only core,queries,extract,frame,raytrace,editing,images,io,memory,lifecycle]
        [--adapter nvidia] [--compare BASELINE.json] [--profile GROUP]

Every group runs in its own Python process (``--worker``), so caches, compiled
pipelines and memory peaks of one group never influence another; each process's
peak RSS (``VmHWM`` on Linux) is reported with its results. Timings are medians
of repeated runs after warm-up, with Python's cyclic garbage collector paused
during each timed run (as ``timeit`` does; a full collection would walk the
whole process). ``--compare`` prints every metric next to a saved run.

Groups:

* ``core`` -- object creation (one plan, per-object calls), transform edits,
  world matrices, undo/redo, memory per object;
* ``queries`` -- spatial queries at 10,000 objects (warm, after an edit, first
  overlaps);
* ``extract`` -- the CPU side of a frame without any GPU: ``extract_render_scene``
  for an unchanged scene, one moved object, all objects moved, a moved camera;
* ``frame`` -- complete GPU frames at 1280x720 (wall clock, every CPU stage of
  ``stats["timings"]`` and the GPU passes from timestamp queries);
* ``raytrace`` -- CPU closest-hit throughput and BVH builds;
* ``editing`` -- opening meshes for editing and representative operations;
* ``images`` -- PNG decoding and encoding of a 1080p render;
* ``io`` -- saving, loading and glTF export at 10,000 objects;
* ``memory`` -- peak RSS of a 100,000-object plan with a query;
* ``lifecycle`` -- memory growth over repeated create/query/render/drop cycles.

Results depend on the machine: compare runs on the same machine only.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GROUPS = (
    "core",
    "queries",
    "extract",
    "frame",
    "raytrace",
    "editing",
    "images",
    "io",
    "memory",
    "lifecycle",
)


# ---------------------------------------------------------------------------------------
# Helpers


def median_time(fn, repeat: int, warmup: int = 1) -> float:
    """Median wall time of ``fn()`` in seconds, the collector paused while it runs."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        enabled = gc.isenabled()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        finally:
            if enabled:
                gc.enable()
    return statistics.median(times)


def peak_rss_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 1024, 1)


def rss_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def ms(seconds: float) -> float:
    return round(seconds * 1e3, 3)


def blocks_plan(n: int) -> dict:
    """``n`` cubes on a grid (100 per row), one shared mesh: the Stage 11 benchmark scene."""
    rows = max(1, n // 100)
    return {
        "format": "aether3d.scene-plan",
        "version": 1,
        "meshes": {"block": {"primitive": "cube", "size": 0.9}},
        "objects": {
            "blocks": {"mesh": "@block", "grid": {"count": [100, rows]}, "position": [0, 0, 0.45]}
        },
    }


def blocks_scene(n: int, camera: bool = True):
    from aether3d import Scene

    scene = Scene("Blocks")
    scene.execute_plan(blocks_plan(n))
    if camera:
        scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1.0))
        cam = scene.add_camera("Camera", fov=50)
        cam.frame(direction=(-0.5, 0.8, -0.6), aspect=16 / 9)
    scene.history.clear()
    return scene


def mesh_objects(scene) -> list:
    return [o for o in scene if o.mesh is not None]


# ---------------------------------------------------------------------------------------
# Groups (each returns a flat dict of metrics; units in the names)


def group_core(quick: bool) -> dict[str, Any]:
    import tracemalloc

    from aether3d import Scene, create_cube
    from aether3d.rendering.scene_data import world_matrices

    out: dict[str, Any] = {}
    n = 10_000
    repeat = 3 if quick else 5

    def plan_once():
        Scene("s").execute_plan(blocks_plan(n))

    out["plan_create_10k_ms"] = ms(median_time(plan_once, repeat))
    out["plan_create_per_object_us"] = round(out["plan_create_10k_ms"] / n * 1e3, 3)

    k = 1000 if quick else 2000
    cube = create_cube(0.9)

    def per_object():
        s = Scene("s")
        for i in range(k):
            s.add_object(f"o{i}", mesh=cube, position=(float(i), 0.0, 0.0))

    out["add_object_per_object_us"] = round(median_time(per_object, 3) / k * 1e6, 3)

    scene = blocks_scene(n, camera=False)
    objs = list(scene)
    first = objs[:k]

    def moves():
        with scene.transaction("move"):
            for i, o in enumerate(first):
                o.position = (float(i), 1.0, 0.0)

    out["set_position_in_transaction_us"] = round(median_time(moves, 3) / k * 1e6, 3)

    def single_moves():
        for i, o in enumerate(first[:500]):
            o.position = (float(i), 2.0, 0.0)

    out["set_position_alone_us"] = round(median_time(single_moves, 3) / 500 * 1e6, 3)
    target = objs[len(objs) // 2]
    out["read_position_ns"] = round(
        median_time(lambda: [target.position for _ in range(10_000)], 3) / 10_000 * 1e9, 1
    )
    out["read_world_matrix_us"] = round(
        median_time(lambda: [target.world_matrix for _ in range(1000)], 3) / 1000 * 1e6, 3
    )
    out["world_matrices_10k_ms"] = ms(median_time(lambda: world_matrices(scene), repeat))

    s = Scene("s")
    s.execute_plan(blocks_plan(n))

    def undo_redo():
        s.history.undo()
        s.history.redo()

    out["undo_redo_plan_10k_ms"] = ms(median_time(undo_redo, repeat) / 2)

    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    s2 = Scene("m")
    s2.execute_plan(blocks_plan(n))
    out["bytes_per_object_with_history"] = round((tracemalloc.get_traced_memory()[0] - base) / n)
    tracemalloc.stop()
    del s2
    return out


def group_queries(quick: bool) -> dict[str, Any]:
    from aether3d.spatial.request import run_query

    out: dict[str, Any] = {}
    scene = blocks_scene(10_000, camera=False)
    ray = {"query": "raycast", "origin": [50.3, -10.0, 0.4], "direction": [0.0, 1.0, 0.0]}
    mover = scene.get_by_name(next(iter(scene)).name)
    out["raycast_first_ms"] = ms(median_time(lambda: run_query(scene, ray), 1, warmup=0))
    out["raycast_warm_ms"] = ms(median_time(lambda: run_query(scene, ray), 9))
    toggle = [0.0]

    def after_edit():
        toggle[0] = 1.0 - toggle[0]
        mover.position = (0.0, toggle[0], 0.45)
        run_query(scene, ray)

    out["raycast_after_edit_ms"] = ms(median_time(after_edit, 5))
    fresh = blocks_scene(10_000, camera=False)
    first = next(iter(fresh))
    out["overlaps_first_ms"] = ms(median_time(lambda: fresh.overlaps(first), 1, warmup=0))
    out["overlaps_warm_ms"] = ms(median_time(lambda: fresh.overlaps(first), 5))
    out["nearest_warm_ms"] = ms(median_time(lambda: fresh.nearest_objects((5, 5, 0), count=4), 5))
    return out


def _frame_edits(scene):
    """Callables changing the scene the way the frame scenarios need."""
    objs = mesh_objects(scene)
    one = objs[len(objs) // 2]
    camera = scene.get_by_name("Camera")
    state = {"k": 0}

    def unchanged():
        pass

    def one_moved():
        state["k"] += 1
        one.position = (one.position[0], one.position[1], 0.45 + 0.01 * (state["k"] % 2))

    def all_moved():
        state["k"] += 1
        dz = 0.01 * (state["k"] % 2)
        with scene.transaction("move all"):
            for o in objs:
                x, y, _ = o.position
                o.position = (x, y, 0.45 + dz)

    def camera_moved():
        state["k"] += 1
        x, y, z = camera.position
        camera.position = (x + (0.05 if state["k"] % 2 else -0.05), y, z)

    def one_added():  # a change of structure: one object more (then fewer again)
        if "added" in state:
            scene.remove(state.pop("added"))
        else:
            state["added"] = scene.add_object("bench added", mesh=one.mesh, position=(0, -3, 0.45))

    return {
        "unchanged": unchanged,
        "one_moved": one_moved,
        "one_added": one_added,
        "all_moved": all_moved,
        "camera_moved": camera_moved,
    }


def group_extract(quick: bool) -> dict[str, Any]:
    import tracemalloc

    from aether3d.rendering import scene_data
    from aether3d.rendering.scene_data import extract_render_scene

    # Since 0.11.1 extract_render_scene keeps a retained render world; _extract_full is
    # the full walk it replaces (and falls back to), measured too for comparison.
    full = getattr(scene_data, "_extract_full", None)
    out: dict[str, Any] = {}
    scene = blocks_scene(10_000)
    camera = scene.get_by_name("Camera")
    for name, edit in _frame_edits(scene).items():
        for label, extract in (("", extract_render_scene), ("_full", full)):
            if extract is None:
                continue
            # Only the extraction is timed: the edit itself is excluded.
            times = []
            for k in range(6 if quick else 9):
                edit()
                gc.disable()
                t0 = time.perf_counter()
                extract(scene, camera, aspect=16 / 9)
                dt = time.perf_counter() - t0
                gc.enable()
                if k:
                    times.append(dt)
            out[f"extract_{name}{label}_ms"] = ms(statistics.median(times))
    if full is not None:  # what the retained world keeps, per object
        fresh = blocks_scene(10_000)
        camera = fresh.get_by_name("Camera")
        gc.collect()
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        extract_render_scene(fresh, camera, aspect=16 / 9)
        gc.collect()
        held = tracemalloc.get_traced_memory()[0] - before
        tracemalloc.stop()
        out["retained_bytes_per_object"] = round(held / len(fresh))
    return out


def group_frame(quick: bool, adapter: str | None) -> dict[str, Any]:
    from aether3d.rendering.gpu import gpu_status

    status = gpu_status(adapter=adapter)
    if not status["available"]:
        return {"skipped": f"no GPU: {status['reason']}"}
    out: dict[str, Any] = {"adapter": status["selected"]["adapter"]["name"]}
    scene = blocks_scene(10_000)
    options = {"width": 1280, "height": 720, "backend": "gpu", "adapter": adapter}
    for _ in range(3):
        scene.render(**options)
    edits = _frame_edits(scene)
    repeat = 5 if quick else 9
    for name, edit in edits.items():
        walls, stages, passes = [], {}, {}
        for _ in range(repeat):
            edit()
            gc.disable()
            t0 = time.perf_counter()
            result = scene.render(**options)
            walls.append(time.perf_counter() - t0)
            gc.enable()
            for key, value in result.stats["timings"].items():
                stages.setdefault(key, []).append(value)
            for key, value in result.stats.get("gpu_timings", {}).items():
                passes.setdefault(key, []).append(value)
        out[f"{name}_wall_ms"] = ms(statistics.median(walls))
        out[f"{name}_stages_ms"] = {k: ms(statistics.median(v)) for k, v in stages.items()}
        out[f"{name}_gpu_passes_ms"] = {k: ms(statistics.median(v)) for k, v in passes.items()}
    # The same frames reading back only the image (render(outputs=...), since 0.11.1).
    if "outputs" in inspect.signature(scene.render.__func__).parameters or _accepts_outputs():
        for name in ("unchanged", "one_moved"):
            edit = edits[name]
            walls, stages = [], {}
            for _ in range(repeat):
                edit()
                gc.disable()
                t0 = time.perf_counter()
                result = scene.render(outputs=("image",), **options)
                walls.append(time.perf_counter() - t0)
                gc.enable()
                for key, value in result.stats["timings"].items():
                    stages.setdefault(key, []).append(value)
            out[f"{name}_image_only_wall_ms"] = ms(statistics.median(walls))
            out[f"{name}_image_only_stages_ms"] = {
                k: ms(statistics.median(v)) for k, v in stages.items()
            }
        # An MCP preview: 640x360, image only, PNG at zlib level 1.
        walls = []
        for _ in range(repeat):
            gc.disable()
            t0 = time.perf_counter()
            result = scene.render(
                width=640, height=360, backend="gpu", adapter=adapter, outputs=("image",)
            )
            result.png_bytes(compression=1)
            walls.append(time.perf_counter() - t0)
            gc.enable()
        out["preview_640x360_png_ms"] = ms(statistics.median(walls))
        # The same preview framed on the geometry (frame=True: MCP previews of scenes
        # without a camera), unchanged and after one object moved.
        for name in ("unchanged", "one_moved"):
            walls = []
            for _ in range(repeat):
                edits[name]()
                gc.disable()
                t0 = time.perf_counter()
                result = scene.render(
                    width=640,
                    height=360,
                    backend="gpu",
                    adapter=adapter,
                    outputs=("image",),
                    frame=True,
                )
                result.png_bytes(compression=1)
                walls.append(time.perf_counter() - t0)
                gc.enable()
            out[f"preview_framed_{name}_640x360_png_ms"] = ms(statistics.median(walls))
    return out


def _accepts_outputs() -> bool:
    from aether3d.rendering.renderer import render

    return "outputs" in inspect.signature(render).parameters


def group_raytrace(quick: bool) -> dict[str, Any]:
    import numpy as np

    sys.path.insert(0, str(HERE))
    from bench_raytrace import camera_rays, sphere_scene, surface_rays

    from aether3d.rendering.raytrace import BLASCache, SceneAccel
    from aether3d.rendering.scene_data import extract_render_scene

    out: dict[str, Any] = {}
    scene = sphere_scene(316)  # ~100k triangles
    rs = extract_render_scene(scene, scene.active_camera)
    build = median_time(lambda: SceneAccel.build(rs, BLASCache()), 1 if quick else 3, warmup=0)
    out["blas_build_100k_tris_ms"] = ms(build)
    accel = SceneAccel.build(rs, BLASCache())
    o, d = camera_rays((0, -4, 0), 320, 240)
    so, sd, si, sp = surface_rays(accel, o, d, np.random.default_rng(0))
    t = median_time(lambda: accel.closest_hit(o, d), 1 if quick else 3)
    out["closest_camera_mrays_s"] = round(o.shape[0] / t / 1e6, 3)
    t = median_time(lambda: accel.closest_hit(so, sd, skip_inst=si, skip_tri=sp), 1 if quick else 3)
    out["closest_bounce_mrays_s"] = round(so.shape[0] / t / 1e6, 3)
    t = median_time(lambda: accel.occluded(so, sd, skip_inst=si, skip_tri=sp), 1 if quick else 3)
    out["any_bounce_mrays_s"] = round(so.shape[0] / t / 1e6, 3)
    return out


def group_editing(quick: bool) -> dict[str, Any]:
    from aether3d import Mesh, create_sphere
    from aether3d.editing import EditableMesh

    out: dict[str, Any] = {}
    attrs = create_sphere(1.0, 96 if quick else 128, 48 if quick else 64)
    out["from_mesh_attributes_tris"] = int(attrs.triangle_count)
    out["from_mesh_attributes_s"] = round(
        median_time(lambda: EditableMesh.from_mesh(attrs), 1, warmup=0), 3
    )
    big = create_sphere(1.0, 448, 224)
    topo = Mesh(big.positions, big.triangles)
    out["from_mesh_topology_tris"] = int(topo.triangle_count)
    out["from_mesh_topology_s"] = round(
        median_time(lambda: EditableMesh.from_mesh(topo, quads=False), 1, warmup=0), 3
    )
    base = EditableMesh.from_mesh(create_sphere(1.0, 64, 32))
    top = base.select_faces(normal=((0, 0, 1), 30))
    some = base.face_ids()[::7][:500]
    quads = [f for f in some if len(base.face_vertices(f)) == 4]
    thickness = 0.05 * min(base.edge_length(e) for e in base.edge_ids())

    def op(fn):
        def go():
            m = base.copy()
            gc.disable()
            t0 = time.perf_counter()
            fn(m)
            dt = time.perf_counter() - t0
            gc.enable()
            return dt

        return round(statistics.median(go() for _ in range(3)) * 1e3, 3)

    out["extrude_region_ms"] = op(lambda m: m.extrude_faces(top, 0.2))
    out["inset_500_faces_ms"] = op(lambda m: m.inset_faces(some, thickness))
    edge = base.edge_ids()[len(base.edge_ids()) // 3]
    out["bevel_edge_ms"] = op(lambda m: m.bevel_edges([edge], 0.2 * thickness))
    out["subdivide_quads_ms"] = op(lambda m: m.subdivide_faces(quads, 1))
    out["bevel_all_edges_faces"] = int(base.face_count)
    out["bevel_all_edges_ms"] = op(lambda m: m.bevel_edges(m.edge_ids(), 0.2 * thickness))
    out["to_mesh_ms"] = op(lambda m: m.to_mesh())
    return out


def group_images(quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d.materials.images import decode_png
    from aether3d.rendering.png import encode_png

    out: dict[str, Any] = {}
    path = ROOT / "examples" / "output" / "castle_path.png"
    data = path.read_bytes()
    decoded = decode_png(data)
    h, w = decoded.pixels.shape[:2]
    t = median_time(lambda: decode_png(data), 3)
    out["png_decode_1080p_ms"] = ms(t)
    out["png_decode_mpix_s"] = round(h * w / t / 1e6, 2)
    rgb = np.ascontiguousarray(decoded.pixels[..., :3])
    out["png_encode_1080p_level6_ms"] = ms(median_time(lambda: encode_png(rgb), 3))
    out["png_encode_1080p_level1_ms"] = ms(median_time(lambda: encode_png(rgb, compression=1), 3))
    return out


def group_io(quick: bool) -> dict[str, Any]:
    import tempfile

    from aether3d import Scene

    out: dict[str, Any] = {}
    scene = blocks_scene(10_000)
    folder = Path(tempfile.mkdtemp(prefix="aether3d-bench-"))
    project = folder / "blocks.aether"
    out["save_10k_s"] = round(median_time(lambda: scene.save(project), 3), 4)
    out["file_mb"] = round(project.stat().st_size / 1e6, 2)
    out["load_10k_s"] = round(median_time(lambda: Scene.load(project), 3), 4)
    glb = folder / "blocks.glb"
    out["export_glb_10k_s"] = round(median_time(lambda: scene.export(glb), 3), 4)
    return out


def group_memory(quick: bool) -> dict[str, Any]:
    from aether3d import Scene

    n = 20_000 if quick else 100_000
    scene = Scene("big")
    t0 = time.perf_counter()
    scene.execute_plan(blocks_plan(n))
    out: dict[str, Any] = {"objects": n, "plan_s": round(time.perf_counter() - t0, 3)}
    t0 = time.perf_counter()
    scene.raycast((50.3, -10.0, 0.4), (0.0, 1.0, 0.0))
    out["first_raycast_s"] = round(time.perf_counter() - t0, 3)
    out["rss_after_mib"] = rss_mib()
    return out


def group_lifecycle(quick: bool) -> dict[str, Any]:
    import weakref

    from aether3d import Scene

    cycles = 6 if quick else 12
    gc.collect()
    start = rss_mib()
    alive = 0
    for k in range(cycles):
        s = Scene(f"c{k}")
        s.execute_plan(blocks_plan(1000))
        s.raycast((5.0, -10.0, 0.4), (0.0, 1.0, 0.0))
        s.add_camera("Camera").frame(aspect=1.0)
        s.render(width=48, height=48, backend="cpu")
        ref = weakref.ref(s)
        del s
        gc.collect()
        alive += ref() is not None
    return {
        "cycles": cycles,
        "scenes_still_alive": alive,
        "rss_growth_mib": None if start is None else round(rss_mib() - start, 1),
    }


def run_group(name: str, quick: bool, adapter: str | None) -> dict[str, Any]:
    group = globals()[f"group_{name}"]
    result = group(quick, adapter) if name == "frame" else group(quick)
    result["peak_rss_mib"] = peak_rss_mib()
    return result


# ---------------------------------------------------------------------------------------
# Driver


def machine() -> dict[str, Any]:
    import numpy as np

    import aether3d

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    info: dict[str, Any] = {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aether3d": aether3d.__version__,
    }
    from aether3d import _features

    info["switches"] = _features.report()
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        info["git"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    try:
        import wgpu

        info["wgpu"] = wgpu.__version__
    except ImportError:
        pass
    return info


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = value


def compare(current: dict, baseline: dict) -> None:
    rows: dict[str, Any] = {}
    flatten("", current["groups"], rows)
    base: dict[str, Any] = {}
    flatten("", baseline["groups"], base)
    width = max((len(k) for k in rows), default=10)
    label = baseline.get("label") or "baseline"
    print(f"\n{'metric':{width}s} {label:>14s} {current.get('label') or 'now':>14s}   ratio")
    for key, value in rows.items():
        before = base.get(key)
        if isinstance(value, (int, float)) and isinstance(before, (int, float)) and before:
            print(f"{key:{width}s} {before:14.4g} {value:14.4g}   {value / before:6.3f}x")
        elif before is not None or isinstance(value, (int, float)):
            print(f"{key:{width}s} {before!s:>14.14s} {value!s:>14.14s}")


def table(paths: list[Path], keys: list[str] | None) -> None:
    """Print a Markdown table of the numeric metrics of saved runs, one column per run."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    flat = []
    for run in runs:
        values: dict[str, Any] = {}
        flatten("", run["groups"], values)
        flat.append(values)
    names = keys or [
        k
        for k in flat[0]
        if isinstance(flat[0][k], (int, float)) and not k.endswith("peak_rss_mib")
    ]
    labels = [run.get("label") or p.stem for run, p in zip(runs, paths, strict=True)]
    print("| metric | " + " | ".join(labels) + " |")
    print("| --- |" + " ---: |" * len(labels))
    for name in names:
        cells = []
        for values in flat:
            value = values.get(name)
            cells.append(
                "" if value is None else f"{value:,.4g}" if isinstance(value, float) else str(value)
            )
        print(f"| `{name}` | " + " | ".join(cells) + " |")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join(GROUPS))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--profile", default=None, help="run one group under cProfile")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--table", type=Path, nargs="+", help="print saved runs as Markdown")
    parser.add_argument("--metrics", default=None, help="with --table: comma-separated keys")
    args = parser.parse_args(argv)

    if args.table:
        table(args.table, args.metrics.split(",") if args.metrics else None)
        return 0

    if args.worker:
        result = run_group(args.worker, args.quick, args.adapter)
        print("BENCH-RESULT " + json.dumps(result))
        return 0
    if args.profile:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        run_group(args.profile, args.quick, args.adapter)
        profiler.disable()
        pstats.Stats(profiler).sort_stats("tottime").print_stats(30)
        return 0

    groups = [g for g in args.only.split(",") if g]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        parser.error(f"unknown groups: {', '.join(unknown)}")
    data: dict[str, Any] = {"label": args.label, "machine": machine(), "groups": {}}
    for group in groups:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", group]
        if args.quick:
            command.append("--quick")
        if args.adapter:
            command += ["--adapter", args.adapter]
        t0 = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BENCH-RESULT ")]
        if proc.returncode != 0 or not lines:
            data["groups"][group] = {"error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"{group}: FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        data["groups"][group] = json.loads(lines[-1][len("BENCH-RESULT ") :])
        print(f"{group} ({time.perf_counter() - t0:.0f} s): {json.dumps(data['groups'][group])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        compare(data, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
