"""Measure deformations of a case with geometry-only metrics (the same vertex sets and
formulas for every binding): Aether3D (computed here) and Blender (loaded)."""
import json, math, sys
from pathlib import Path
import numpy as np
from aether3d.animation import rotation as R
from aether3d.animation.rig import skin_linear, skin_dual_quaternion


def frames(case):
    J = case["joints"]
    idx = {j["name"]: k for k, j in enumerate(J)}
    rest = []
    for j in J:
        m = np.eye(4)
        m[:3, :3] = np.array(j["frame"])
        m[:3, 3] = j["head"]
        rest.append(m)
    return J, idx, np.array(rest)


def posed_model(case, rotations):
    J, idx, rest = frames(case)
    out = rest.copy()
    local = []
    for k, j in enumerate(J):
        p = idx[j["parent"]] if j["parent"] else -1
        loc = rest[k] if p < 0 else np.linalg.solve(rest[p], rest[k])
        q = rotations.get(j["name"])
        if q is not None:
            r = np.eye(4)
            r[:3, :3] = R.matrices(np.array(q)[None])[0]
            loc = loc @ r
        local.append(loc)
    for k, j in enumerate(J):
        p = idx[j["parent"]] if j["parent"] else -1
        out[k] = local[k] if p < 0 else out[p] @ local[k]
    return out


def vertex_normals(p, t):
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    n = np.zeros_like(p)
    for c in range(3):
        np.add.at(n, t[:, c], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


def edges_of(t):
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0)


def dihedral(p, t):
    """Per interior edge: the angle (degrees) between the two face normals."""
    fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    f = np.tile(np.arange(len(t)), 3)
    key = np.minimum(e[:, 0], e[:, 1]) * (p.shape[0] + 1) + np.maximum(e[:, 0], e[:, 1])
    o = np.argsort(key, kind="stable")
    key, f = key[o], f[o]
    pair = np.flatnonzero(key[1:] == key[:-1])
    a, b = f[pair], f[pair + 1]
    d = np.einsum("ij,ij->i", fn[a], fn[b])
    return np.degrees(np.arccos(np.clip(d, -1, 1))), key[pair]


def measure(p0, p1, t, region, pivot0, pivot1, rest_dih):
    e = edges_of(t)
    l0 = np.linalg.norm(p0[e[:, 0]] - p0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(p1[e[:, 0]] - p1[e[:, 1]], axis=1)
    good = l0 > 0.35 * np.median(l0)
    r = np.where(good, l1 / np.maximum(l0, 1e-12), 1.0)
    inreg = region[e[:, 0]] | region[e[:, 1]]
    dih, _ = dihedral(p1, t)
    sharp = int((dih > 60).sum()) - int((rest_dih > 60).sum())
    fn0 = np.cross(p0[t[:, 1]] - p0[t[:, 0]], p0[t[:, 2]] - p0[t[:, 0]])
    fn1 = np.cross(p1[t[:, 1]] - p1[t[:, 0]], p1[t[:, 2]] - p1[t[:, 0]])
    # a fold: a triangle facing against the average normal of its posed neighbourhood
    vn1 = vertex_normals(p1, t)
    nv = vn1[t].mean(axis=1)
    a0 = 0.5 * np.linalg.norm(fn0, axis=1)
    ok = a0 > 0.2 * np.median(a0)
    folded = int((ok & (np.einsum("ij,ij->i", fn1, nv) < 0)).sum()) - int(
        (ok & (np.einsum("ij,ij->i", fn0, vertex_normals(p0, t)[t].mean(axis=1)) < 0)).sum())
    d0 = np.linalg.norm(p0[region] - pivot0, axis=1)
    d1 = np.linalg.norm(p1[region] - pivot1, axis=1)
    return {
        "kept": round(float(d1.mean() / d0.mean()), 4),
        "kept_min_p5": round(float(np.quantile(d1 / np.maximum(d0, 1e-9), 0.05)), 4),
        "worst_stretch": round(float(r[inreg].max()), 3),
        "worst_compress": round(float(r[inreg].min()), 3),
        "edges_over_1.4": int((r > 1.4).sum()),
        "edges_under_0.7": int((r < 0.7).sum()),
        "sharp_new": sharp,
        "folded": max(folded, 0),
        "p99_dihedral": round(float(np.quantile(dih, 0.999)), 1),
    }


def main(case_dir, extra_dirs=()):
    case_dir = Path(case_dir)
    case = json.loads((case_dir / "case.json").read_text())
    p0 = np.load(case_dir / "positions.npy")
    t = np.load(case_dir / "triangles.npy")
    n0 = np.load(case_dir / "normals.npy")
    J, idx, rest = frames(case)
    names = case["weight_joints"]
    pal_idx = [idx[n] for n in names]
    inv_bind = np.linalg.inv(rest[pal_idx])
    rest_dih, _ = dihedral(p0, t)
    weights = {"aether_v019": np.load(case_dir / "weights_aether_v019.npy")}
    for f in sorted(case_dir.glob("weights_*.npy")):
        tag = f.stem[len("weights_"):]
        if tag not in ("given",):
            weights.setdefault(tag, np.load(f))
    results = {}
    for pose in case["poses"]:
        k = idx[pose["joint"]]
        pm = posed_model(case, pose["rotations"])
        pivot0, pivot1 = rest[k][:3, 3], pm[k][:3, 3]
        # region: vertices within 1.6x the 15% quantile distance of the joint's limb surface
        d = np.linalg.norm(p0 - pivot0, axis=1)
        r15 = np.quantile(np.sort(d)[:400], 0.15)
        region = d <= max(2.2 * r15, 0.06)
        row = {}
        pal = pm[pal_idx] @ inv_bind
        for tag in [k for k in weights if k.startswith("aether")]:
            W = weights[tag]
            order = np.argsort(-W, axis=1)[:, :4]
            w4 = np.take_along_axis(W, order, axis=1)
            w4 /= w4.sum(1, keepdims=True)
            p_l, _ = skin_linear(p0.astype(np.float32), None, order.astype(np.int32), w4, pal)
            p_d, _ = skin_dual_quaternion(p0.astype(np.float32), None, order.astype(np.int32), w4, pal)
            row[f"{tag}:lbs"] = measure(p0, p_l.astype(np.float64), t, region, pivot0, pivot1, rest_dih)
            row[f"{tag}:dqs"] = measure(p0, p_d.astype(np.float64), t, region, pivot0, pivot1, rest_dih)
            gl = case_dir / "posed_given" / f"{pose['name']}__lbs.npy"
            gl = gl if tag == "aether_v019" else case_dir / "nonexistent"
            if gl.exists():
                diff = np.abs(np.load(gl) - p_l).max()
                row[f"{tag}:lbs"]["blender_same_weights_maxdiff"] = float(diff)
                for v in ("lbs_cs", "dqs_cs"):
                    row[f"{tag}:blender_{v}"] = measure(p0, np.load(case_dir / "posed_given" / f"{pose['name']}__{v}.npy"), t, region, pivot0, pivot1, rest_dih)
        for v in ("lbs", "dqs", "lbs_cs", "dqs_cs"):
            f = case_dir / "posed_blender_heat" / f"{pose['name']}__{v}.npy"
            if f.exists():
                row[f"blender_heat:{v}"] = measure(p0, np.load(f), t, region, pivot0, pivot1, rest_dih)
        for ed in extra_dirs:
            for f in sorted(Path(ed).glob(f"{pose['name']}__*.npy")):
                row[f"{Path(ed).name}:{f.stem.split('__')[1]}"] = measure(p0, np.load(f), t, region, pivot0, pivot1, rest_dih)
        results[pose["name"]] = row
    # weight similarity
    sim = {}
    if "blender_heat" in weights:
        A, B = weights["aether_v019"], weights["blender_heat"]
        Bn = B / np.maximum(B.sum(1, keepdims=True), 1e-12)
        sim["aether_v019_vs_blender_heat"] = {
            "mean_abs_diff_per_vertex": round(float(np.abs(A - Bn).sum(1).mean() / 2), 4),
            "dominant_agreement": round(float((A.argmax(1) == Bn.argmax(1)).mean()), 4),
        }
    return results, sim


if __name__ == "__main__":
    res, sim = main(sys.argv[1], sys.argv[2:])
    out = Path(sys.argv[1]) / "metrics.json"
    out.write_text(json.dumps({"poses": res, "weights": sim}, indent=1))
    print(json.dumps(sim, indent=1))
    cols = ["kept", "worst_stretch", "edges_over_1.4", "sharp_new", "folded", "p99_dihedral"]
    for pose, row in res.items():
        print(f"== {pose}")
        for tag, m in row.items():
            extra = f"  same-weights maxdiff {m['blender_same_weights_maxdiff']:.2e}" if "blender_same_weights_maxdiff" in m else ""
            print(f"  {tag:28s} " + " ".join(f"{c}={m[c]}" for c in cols) + extra)
