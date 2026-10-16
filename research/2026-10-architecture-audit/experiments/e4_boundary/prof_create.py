import cProfile, pstats, io, time, gc, statistics, uuid
from aether3d import Scene
PLAN = {"format": "aether3d.scene-plan", "version": 1, "meshes": {"block": {"primitive": "cube", "size": 0.9}},
        "objects": {"blocks": {"mesh": "@block", "grid": {"count": [100, 100]}, "position": [0, 0, 0.45]}}}
ts = []
for _ in range(5):
    s = Scene("s"); t = time.perf_counter(); s.execute_plan(PLAN); ts.append(time.perf_counter() - t)
print(f"plan 10k blocks: median {statistics.median(ts)*1e3:.1f} ms")
s = Scene("s")
pr = cProfile.Profile(); pr.enable(); s.execute_plan(PLAN); pr.disable()
st = pstats.Stats(pr); st.sort_stats("tottime")
buf = io.StringIO(); st.stream = buf; st.print_stats(14); print("\n".join(buf.getvalue().splitlines()[:40]))
# floors: what bare Python objects cost per object
class T:
    __slots__ = ("_position", "_rotation", "_scale", "_owner")
class O:
    __slots__ = tuple(f"a{i}" for i in range(17))
    def __init__(self, i):
        for k in range(17):
            setattr(self, f"a{k}", None)
def floor(n=10000):
    d = {}
    for i in range(n):
        t = T(); t._position = (0.0, 0.0, 0.0); t._rotation = (0.0, 0.0, 0.0, 1.0); t._scale = (1.0, 1.0, 1.0); t._owner = None
        o = O.__new__(O); oid = uuid.uuid4().hex; d[oid] = o
    return d
gc.disable(); t = time.perf_counter(); floor(); dt = time.perf_counter() - t; gc.enable()
print(f"floor (slotted object + transform + uuid4 + dict insert): {dt/10000*1e6:.2f} us/object")
t = time.perf_counter(); [uuid.uuid4().hex for _ in range(10000)]; print(f"uuid4().hex alone: {(time.perf_counter()-t)/10000*1e6:.2f} us")
