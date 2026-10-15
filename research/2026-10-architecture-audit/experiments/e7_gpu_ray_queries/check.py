import json, sys
from pathlib import Path
import numpy as np
D = Path(sys.argv[1])
NONE = 0xFFFFFFFF
res = {}
for scene in ("sphere100k", "cubes400", "sphere1m"):
    for r in ("camera", "bounce"):
        ref = np.fromfile(D / scene / f"ref_{r}.bin", np.uint32).reshape(-1, 2)
        reft = np.fromfile(D / scene / f"reft_{r}.bin", np.float64)
        row = {}
        outs = {}
        for k in ("sw", "hw"):
            o = np.fromfile(D / scene / f"out_{r}_{k}.bin", np.uint32).reshape(-1, 4)
            outs[k] = o
            t = o[:, 2].view(np.float32).astype(np.float64)
            hit_ref = ref[:, 0] != NONE; hit = o[:, 0] != NONE
            both = hit_ref & hit
            rel = np.abs(t[both] - reft[both]) / np.maximum(reft[both], 1e-30)
            row[k] = {"hit/miss same as f64 ref": round(float(np.mean(hit_ref == hit)), 6),
                      "same triangle as ref (both hit)": round(float(np.mean((o[both, 0] == ref[both, 0]) & (o[both, 1] == ref[both, 1]))), 6),
                      "t rel diff median": float(np.median(rel)), "t rel diff p99.99": float(np.quantile(rel, 0.9999)),
                      "hits": int(hit.sum()), "ref hits": int(hit_ref.sum())}
        a, b = outs["sw"], outs["hw"]
        row["sw vs hw same (inst, prim)"] = round(float(np.mean((a[:, 0] == b[:, 0]) & (a[:, 1] == b[:, 1]))), 6)
        res[f"{scene} {r}"] = row
        print(scene, r, json.dumps(row))
json.dump(res, open(D / "agreement.json", "w"), indent=1)
