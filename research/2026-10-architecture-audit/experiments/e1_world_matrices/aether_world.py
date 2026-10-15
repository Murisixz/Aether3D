import statistics, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from e1_data import make, numpy_level
from aether3d import Scene
from aether3d.scene import NewObject
from aether3d.rendering.scene_data import world_matrices
for n in (10_000, 100_000):
    parent, pos, quat, scale, depth = make(n)
    s = Scene("w"); P, Q, S = pos.tolist(), quat.tolist(), scale.tolist()
    recs = [NewObject(f"o{k}", parent=(int(parent[k]) if parent[k] >= 0 else None),
                      position=tuple(P[k]), rotation=tuple(Q[k]), scale=tuple(S[k])) for k in range(n)]
    s._create_many(recs, "build")
    ref = numpy_level(parent, pos, quat, scale, depth)
    ids = {o.name: o.id for o in s}
    world_matrices(s)
    ts = []
    for _ in range(7):
        t = time.perf_counter(); w = world_matrices(s); ts.append(time.perf_counter() - t)
    ok = all(np.array_equal(w[ids[f"o{k}"]], ref[k]) for k in range(0, n, max(1, n // 2000)))
    print(f"N={n}: Aether3D world_matrices(scene) median {statistics.median(ts)*1e3:.2f} ms (min {min(ts)*1e3:.2f}); matches flat-array reference: {ok}")
