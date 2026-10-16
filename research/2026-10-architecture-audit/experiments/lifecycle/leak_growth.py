import gc, os, sys, tracemalloc
from aether3d import Scene
def vmrss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"): return int(line.split()[1]) / 1024
plan = {"format": "aether3d.scene-plan", "version": 1, "meshes": {"m": {"primitive": "sphere", "radius": 0.4, "segments": 24}},
        "objects": {"g": {"mesh": "@m", "grid": {"count": [20, 50]}}}}
tracemalloc.start()
gc.collect(); base = tracemalloc.get_traced_memory()[0]; rss0 = vmrss()
for k in range(int(sys.argv[1])):
    s = Scene(f"s{k}"); s.execute_plan(plan); s.raycast((0, -5, 0.1), (0.01, 1, 0)); s.overlaps(next(o for o in s if o.mesh is not None))
    del s; gc.collect()
    if k % 5 == 4:
        print(f"after {k+1:3d} scenes: traced +{(tracemalloc.get_traced_memory()[0]-base)/2**20:7.1f} MiB, RSS +{vmrss()-rss0:7.1f} MiB", flush=True)
