"""Measure current Aether3D subsystems at 10,000 objects (one process, warm where noted)."""
import gc, json, os, statistics, sys, tempfile, time, tracemalloc
from pathlib import Path
from aether3d import Scene
from aether3d.spatial.request import run_query

def blocks(n):
    return {"format": "aether3d.scene-plan", "version": 1, "meshes": {"block": {"primitive": "cube", "size": 0.9}},
            "objects": {"blocks": {"mesh": "@block", "grid": {"count": [100, n // 100]}, "position": [0, 0, 0.45]}}}

def med(fn, k=5):
    ts = []
    for _ in range(k):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

out = {}
# memory per object
gc.collect(); tracemalloc.start()
s = Scene("mem"); s.execute_plan(blocks(10000))
cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
out["bytes_per_object_incl_history"] = round(cur / 10000)
s.history.clear() if hasattr(s.history, "clear") else None
gc.collect()
# per-property access costs
o = s.get_by_name("blocks 5")
out["read_position_ns"] = round(med(lambda: [o.position for _ in range(100000)], 3) / 100000 * 1e9)
out["read_world_matrix_us"] = round(med(lambda: [o.world_matrix for _ in range(1000)], 3) / 1000 * 1e6, 2)
def setpos():
    for i in range(2000):
        o.position = (float(i), 0.0, 0.0)
out["set_position_us"] = round(med(setpos, 3) / 2000 * 1e6, 2)
def tx_setpos():
    with s.transaction("move many"):
        for i, ob in enumerate(list(s)[:2000]):
            ob.position = (float(i), 1.0, 0.0)
out["set_position_in_tx_us"] = round(med(tx_setpos, 3) / 2000 * 1e6, 2)
# save / load
d = Path(tempfile.mkdtemp())
p = d / "b.aether"
out["save_s"] = round(med(lambda: s.save(p), 3), 3)
out["file_mb"] = round(p.stat().st_size / 1e6, 2)
out["load_s"] = round(med(lambda: Scene.load(p), 3), 3)
# glTF export
g = d / "b.glb"
out["export_glb_s"] = round(med(lambda: s.export(g), 3), 3)
out["glb_mb"] = round(g.stat().st_size / 1e6, 2)
# spatial index: first query after a change, and warm
def first_query():
    s.get_by_name("blocks 0").position = (0.0, 0.0, 0.0)  # bump the revision
    run_query(s, {"query": "raycast", "origin": [50.3, -10, 0.4], "direction": [0, 1, 0]})
out["spatial_first_query_after_edit_ms"] = round(med(first_query, 3) * 1e3, 1)
out["spatial_warm_raycast_ms"] = round(med(lambda: run_query(s, {"query": "raycast", "origin": [50.3, -10, 0.4], "direction": [0, 1, 0]}), 5) * 1e3, 2)
# render: one object moved between frames (is the frame O(N) anyway?)
s.add_directional_light("Sun", direction=(-0.4, 0.6, -1))
s.add_camera("Camera", fov=50).frame(direction=(-0.5, 0.8, -0.6), aspect=16 / 9)
for _ in range(3): s.render(width=1280, height=720, backend="gpu")
same = med(lambda: s.render(width=1280, height=720, backend="gpu"), 5)
def moved():
    s.get_by_name("blocks 7").position = (7.0, 0.0, 0.5)
    s.render(width=1280, height=720, backend="gpu")
one = med(moved, 5)
out["gpu_frame_unchanged_ms"] = round(same * 1e3, 1)
out["gpu_frame_one_object_moved_ms"] = round(one * 1e3, 1)
r = s.render(width=1280, height=720, backend="gpu")
out["gpu_stage_ms"] = {k: round(v * 1e3, 1) for k, v in r.stats["timings"].items()}
print(json.dumps(out, indent=1))
