"""Rig diagnostics and verified corrections.

:func:`diagnose_rig` measures a rig and everything bound to it -- the skeleton (bone
lengths, frames, symmetry, IK poles), the skins (normalization, leakage, cross-side
weights, helper joints, the bind against the current rest) and the deformation under a
**diagnostic pose suite** (elbows and knees bent 120 degrees, hips flexed 90, arms raised
90 to the side, the torso and the wrists twisted, the head turned): the collapse of each
bent joint's cross-section, interpenetration of the child segment into the parent's,
stretched, compressed, collapsed and flipped triangles, the volume of closed meshes --
and, over the timeline, IK residuals, knee and elbow flips and foot sliding. Every finding
has a ``code``, a ``severity``, a ``basis`` (``numeric``: measured; nothing here is a
judgment of taste), the joints, meshes, poses or times, sample vertex ids and a
magnitude. Nothing is changed.

:func:`propose_rig_fix` turns findings into proposals: concrete operations (weight
edits, a rebind, blended skinning on a collapsing joint, smoothing, pole moves, joint
mirroring, foot planting...) with a safety class (``safe``: exact repairs; ``verify``:
kept only if the measurement improves; ``approval``: a choice for a person or an agent).
:func:`apply_rig_fix` applies chosen proposals as **one undo step**, re-measures, and
undoes everything when a ``verify`` fix does not improve what it targets or makes
something else worse (``dry_run`` reports the after-measurements and changes nothing).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import rotation as R
from .rig import METHODS

if TYPE_CHECKING:
    from .system import AnimationSystem

SEVERITIES = ("error", "warning", "info")
COLLAPSE_WARN = 0.75  # a bent joint's cross-section keeps less than this of its size
COLLAPSE_ERROR = 0.5
STRETCH = 1.6
COMPRESS = 0.5
FIX_CALLS = ("edit_weights", "edit_rig", "skin_method", "rebind", "pole", "plant_feet")


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


# --------------------------------------------------------------------------------------- #
# The diagnostic pose suite
# --------------------------------------------------------------------------------------- #


def _role(rig, role: str) -> int | None:
    from .generators import find_roles

    k = rig.role_index(role)
    if k is not None:
        return k
    name = find_roles(rig).get(role)
    return None if name is None else rig.index(name)


def _bone_dir(rig, k: int) -> np.ndarray:
    model = rig.rest_model()
    tails = rig.tails()
    end = (model[k] @ np.append(tails[k], 1.0))[:3]
    d = end - model[k][:3, 3]
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def pose_suite(rig) -> list[dict[str, Any]]:
    """The diagnostic poses this rig supports: ``{name, joint, turns: {joint index:
    (armature-space axis, degrees)}}`` (joints found by role)."""
    fwd = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}.get(
        rig.forward_axis("-y"), (0, -1, 0)
    )
    fwd = np.array(fwd, np.float64)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    poses = []

    def turn(k, toward, deg):
        d = _bone_dir(rig, k)
        axis = np.cross(d, toward)
        n = float(np.linalg.norm(axis))
        return None if n < 1e-6 else (axis / n, deg)

    for side, out in (("L", -right), ("R", right)):
        for role, name, toward, deg in (
            ("lower_arm", "elbow_bend_120", fwd, 120.0),
            ("shin", "knee_bend_120", -fwd, 120.0),
            ("thigh", "hip_flex_90", fwd, 90.0),
            ("upper_arm", "shoulder_raise_90", up, 90.0),
        ):
            k = _role(rig, f"{role}.{side}")
            if k is None:
                continue
            t = turn(k, toward if role != "upper_arm" else up, deg)
            if role == "upper_arm":
                d = _bone_dir(rig, k)
                # raise sideways: from the rest direction toward up, in the plane of out
                axis = np.cross(d, up) if abs(d @ up) < 0.95 else np.cross(d, out)
                t = (axis / np.linalg.norm(axis), deg)
            if t is not None:
                poses.append({"name": f"{name}.{side}", "joint": k, "turns": {k: t}})
        k = _role(rig, f"hand.{side}")
        if k is not None:
            poses.append(
                {
                    "name": f"wrist_twist_90.{side}",
                    "joint": k,
                    "turns": {k: (_bone_dir(rig, k), 90.0)},
                }
            )
    spine = [k for k in (_role(rig, "spine"), _role(rig, "chest")) if k is not None]
    if spine:
        poses.append(
            {
                "name": "torso_twist_45",
                "joint": spine[0],
                "turns": {k: (up, 45.0 / len(spine)) for k in spine},
            }
        )
        poses.append(
            {
                "name": "spine_bend_45",
                "joint": spine[0],
                "turns": {k: (np.cross(up, fwd), 45.0 / len(spine)) for k in spine},
            }
        )
    k = _role(rig, "head")
    if k is not None:
        poses.append({"name": "head_turn_60", "joint": k, "turns": {k: (up, 60.0)}})
    return poses


def _posed(system, arm, rig, turns: dict[int, tuple[np.ndarray, float]], time: float):
    """The rigging pass with joints turned from rest about armature axes (``turns``) --
    a synthetic pose: the scene is not changed."""
    from .rigging import evaluate_rigging

    pose = system.evaluate(time)
    model = rig.rest_model()
    joints = {k: dict(v) for k, v in pose.joints.items()}
    joints[arm._id] = {}  # (the suite poses the rest pose: no animation, no IK)
    mine = joints[arm._id]
    influences = {k: dict(v) for k, v in pose.influences.items()}
    influences[arm._id] = {c.name: 0.0 for c in rig.constraints if c.type != "twist"}
    for k, (axis, deg) in turns.items():
        local_axis = np.linalg.solve(model[k][:3, :3], axis)
        local_axis /= np.linalg.norm(local_axis)
        qd = R.axis_angle(local_axis, math.radians(deg))[0]
        rest = np.array(rig.joints[k].rotation, np.float64)
        q = R.multiply(rest[None], qd[None])[0]
        name = rig.joints[k].name
        old = mine.get(name, (None, None, None))
        mine[name] = (old[0], q / np.linalg.norm(q), old[2])
    synthetic = dataclasses.replace(pose, joints=joints, influences=influences, extra={})
    return evaluate_rigging(system, synthetic)


# --------------------------------------------------------------------------------------- #
# Measurements
# --------------------------------------------------------------------------------------- #


def _tri_measures(
    p0: np.ndarray,
    p1: np.ndarray,
    tris: np.ndarray,
    n1: np.ndarray | None,
    n0: np.ndarray | None = None,
):
    a0, b0, c0 = p0[tris[:, 0]], p0[tris[:, 1]], p0[tris[:, 2]]
    a1, b1, c1 = p1[tris[:, 0]], p1[tris[:, 1]], p1[tris[:, 2]]
    cr0 = np.cross(b0 - a0, c0 - a0)
    cr1 = np.cross(b1 - a1, c1 - a1)
    area0 = 0.5 * np.linalg.norm(cr0, axis=1)
    area1 = 0.5 * np.linalg.norm(cr1, axis=1)
    # Slivers (a voxel mesher's, a modeler's) say nothing: only triangles of typical size.
    ok = area0 > max(0.2 * float(np.median(area0)), 1e-14)
    ratio = np.where(ok, area1 / np.where(ok, area0, 1.0), 1.0)
    collapsed = ok & (ratio < 0.15)
    flipped = np.zeros(len(tris), bool)
    if n1 is not None:
        nv = n1[tris].mean(axis=1)
        dots = np.einsum("ij,ij->i", cr1, nv)
        flipped = ok & (area1 > 1e-14) & (dots < 0)
        if n0 is not None:
            # Only what the pose turned over: a triangle that already disagrees with its
            # vertex normals at rest (a crease of the modeled surface) is not a fold.
            flipped &= np.einsum("ij,ij->i", cr0, n0[tris].mean(axis=1)) > 0
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e0 = np.linalg.norm(p0[edges[:, 0]] - p0[edges[:, 1]], axis=1)
    e1 = np.linalg.norm(p1[edges[:, 0]] - p1[edges[:, 1]], axis=1)
    good = e0 > max(0.35 * float(np.median(e0)), 1e-12)
    er = np.where(good, e1 / np.where(good, e0, 1.0), 1.0)
    return {
        "collapsed": np.flatnonzero(collapsed),
        "flipped": np.flatnonzero(flipped),
        "stretched": edges[good & (er > STRETCH)],
        "compressed": edges[good & (er < COMPRESS)],
        "worst_stretch": float(er.max()) if er.size else 1.0,
        "worst_compress": float(er.min()) if er.size else 1.0,
    }


def _volume(p: np.ndarray, tris: np.ndarray) -> float:
    a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _closed(tris: np.ndarray) -> bool:
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


class _Skinned:
    """A skinned mesh's data for measuring (world space at rest)."""

    def __init__(self, system, oid: str, skin, rig, worlds) -> None:
        obj = system.scene._objects[oid]
        self.obj = obj
        self.skin = skin
        mesh = obj.mesh
        self.tris = np.asarray(mesh.triangles, np.int64)
        m = worlds[oid]
        self.m = m
        self.p0 = mesh.positions.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        dense = skin.dense()
        self.dense = dense
        self.palette = [rig.index_of_id(j) for j in skin.joints]
        self.dominant = np.array([self.palette[c] for c in np.argmax(dense, axis=1)])
        self.closed = _closed(self.tris)
        self.volume0 = _volume(self.p0, self.tris) if self.closed else None
        self.n0 = None if mesh.normals is None else mesh.normals.astype(np.float64) @ m[:3, :3].T

    def posed(self, result) -> tuple[np.ndarray, np.ndarray | None]:
        mesh = result.meshes.get(self.obj._id)
        if mesh is None:
            return self.p0, None
        p = mesh.positions.astype(np.float64) @ self.m[:3, :3].T + self.m[:3, 3]
        n = None if mesh.normals is None else mesh.normals.astype(np.float64) @ self.m[:3, :3].T
        return p, n


def _subtree(rig, k: int) -> set[int]:
    return {k, *rig.descendants(k)}


def _joint_ring(sk: _Skinned, rig, k: int, a_world: np.ndarray) -> np.ndarray:
    """Vertices around joint ``k`` at rest: weighted to its parent's bone or its own,
    within the band where the two blend (or a limb radius of the joint)."""
    model = a_world @ rig.rest_model()
    pivot = model[k][:3, 3]
    parent = rig.joints[k].parent
    near = np.isin(sk.dominant, [k] + ([parent] if parent >= 0 else []))
    d = np.linalg.norm(sk.p0 - pivot, axis=1)
    if not near.any():
        return np.zeros(0, np.int64)
    r = float(np.quantile(d[near], 0.15)) if near.sum() > 10 else float(d[near].min())
    band = near & (d <= max(1.6 * r, 1e-6))
    return np.flatnonzero(band)


# --------------------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------------------- #


def diagnose_rig(
    system: AnimationSystem,
    armature: Any,
    *,
    poses: Any = "suite",
    time: float | None = None,
    animation: bool = True,
    max_findings: int = 50,
    max_vertices: int = 12,
) -> dict[str, Any]:
    """Measure ``armature``'s rig, skins and deformation (see the module docstring)."""
    from ..rendering.scene_data import world_matrices

    arm = system._object(armature, "armature")
    rig = system.rig(arm)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    t = system.time if time is None else float(time)
    findings: list[dict[str, Any]] = []

    def add(code, severity, message, **extra):
        findings.append(
            {"code": code, "severity": severity, "basis": "numeric", "message": message, **extra}
        )

    worlds = world_matrices(system.scene)
    a_world = worlds[arm._id]
    model = rig.rest_model()
    height = float(np.ptp(model[:, 2, 3])) or 1.0
    # -- skeleton -----------------------------------------------------------------------------
    for k, j in enumerate(rig.joints):
        kids = [c for c in range(len(rig.joints)) if rig.joints[c].parent == k]
        if kids and all(np.linalg.norm(model[c][:3, 3] - model[k][:3, 3]) < 1e-5 for c in kids):
            add(
                "ZERO_LENGTH_BONE",
                "warning",
                f"joint {j.name!r} sits on its children",
                joints=[j.name],
            )
    aligned = []
    for k, _j in enumerate(rig.joints):
        kids = [c for c in range(len(rig.joints)) if rig.joints[c].parent == k]
        if len(kids) == 1:
            d = model[kids[0]][:3, 3] - model[k][:3, 3]
            if np.linalg.norm(d) > 1e-6:
                y = model[k][:3, 1] / np.linalg.norm(model[k][:3, 1])
                aligned.append(
                    (
                        k,
                        math.degrees(
                            math.acos(min(1.0, max(-1.0, float(y @ d / np.linalg.norm(d)))))
                        ),
                    )
                )
    if aligned and np.median([a for _, a in aligned]) < 5.0:  # a bone-frame rig
        for k, ang in aligned:
            if ang > 10.0:
                add(
                    "BONE_AXIS_MISALIGNED",
                    "info",
                    f"{rig.joints[k].name!r}'s +Y is {ang:.1f} degrees off its bone",
                    joints=[rig.joints[k].name],
                    magnitude=round(ang, 3),
                    unit="deg",
                )
    seen = set()
    for k, j in enumerate(rig.joints):
        other = rig.mirror_name(j.name)
        o = rig.index(other) if other else None
        if o is None or o == k or (o, k) in seen:
            continue
        seen.add((k, o))
        pa = model[k][:3, 3]
        pb = model[o][:3, 3] * np.array([-1.0, 1.0, 1.0])
        diff = float(np.linalg.norm(pa - pb))
        if diff > 0.01 * height:
            add(
                "ASYMMETRIC_JOINTS",
                "warning",
                f"{j.name!r} and {other!r} are {diff:.4f} from mirror images",
                joints=[j.name, other],
                magnitude=round(diff, 6),
                unit="m",
            )
    for c in rig.constraints:
        if c.type != "ik" or not c.p.get("pole") or len(c.p["chain"]) < 3:
            continue
        rr = system.rigging(t)
        _, posed_model = rr.joints[arm._id]
        idx = [rig.index(n) for n in c.p["chain"]]
        a = (a_world @ posed_model[idx[0]])[:3, 3]
        b = (a_world @ posed_model[idx[-1]])[:3, 3]
        pole = None
        pt = c.p["pole"]
        if "object" in pt:
            obj = system.scene._objects.get(pt["object"])
            if obj is not None:
                pole = worlds[obj._id][:3, 3]
        if pole is not None:
            ab = b - a
            off = pole - a - ((pole - a) @ ab) / max(ab @ ab, 1e-12) * ab
            if np.linalg.norm(off) < 0.02 * max(float(np.linalg.norm(ab)), 1e-9):
                add(
                    "POLE_ON_LINE",
                    "warning",
                    f"the pole of {c.name!r} lies on the chain's line: the bend is undefined",
                    constraint=c.name,
                )
    for d in system.rigging(t).diagnostics:
        if d.location and arm.name in d.location:
            add(d.code, "warning", d.message)
    # -- skins --------------------------------------------------------------------------------
    rig_item = system._owned[("rig", arm._id)]
    skinned = []
    for (kind, oid), item in system._owned.items():
        if (
            kind != "skin"
            or item._definition.rig != rig_item.id
            or oid not in system.scene._objects
        ):
            continue
        sk = _Skinned(system, oid, item._definition, rig, worlds)
        skinned.append(sk)
        rep = system.inspect_weights(sk.obj, max_list=max_vertices)
        for f in rep["findings"]:
            add(
                f["code"],
                f["severity"],
                f.get("message", f["code"].lower().replace("_", " ")),
                mesh=sk.obj.name,
                **{k: v for k, v in f.items() if k in ("vertices", "count", "joint", "joints")},
            )
        # The bind against the current rest: at rest every vertex is where it was modeled.
        pal = np.linalg.inv(sk.m) @ a_world @ model[sk.palette] @ sk.skin.inverse_bind
        dev = float(np.abs(pal - np.eye(4)[None]).max())
        if dev > 1e-6:
            from .rig import skin_linear

            moved, _ = skin_linear(
                sk.obj.mesh.positions, None, sk.skin.indices, sk.skin.weights, pal
            )
            disp = np.linalg.norm(moved.astype(np.float64) - sk.obj.mesh.positions, axis=1)
            if disp.max() > 1e-4:
                worst = np.argsort(-disp)[:max_vertices]
                add(
                    "BIND_MISMATCH",
                    "warning",
                    f"{sk.obj.name!r} moves up to {disp.max():.4f} at rest: the rig's rest "
                    "changed after binding",
                    mesh=sk.obj.name,
                    vertices=worst.tolist(),
                    magnitude=round(float(disp.max()), 6),
                    unit="m",
                )
    # -- the pose suite -----------------------------------------------------------------------
    suite = pose_suite(rig) if poses == "suite" else ([] if not poses else _named(poses, rig))
    pose_report: dict[str, Any] = {}
    for pose in suite:
        result = _posed(system, arm, rig, pose["turns"], t)
        entry: dict[str, Any] = {}
        k = pose["joint"]
        for sk in skinned:
            p1, n1 = sk.posed(result)
            m = _tri_measures(sk.p0, p1, sk.tris, n1, sk.n0)
            rec: dict[str, Any] = {
                "collapsed_triangles": int(m["collapsed"].size),
                "flipped_triangles": int(m["flipped"].size),
                "stretched_edges": int(len(m["stretched"])),
                "compressed_edges": int(len(m["compressed"])),
                "worst_stretch": round(m["worst_stretch"], 4),
            }
            if sk.closed:
                v1 = _volume(p1, sk.tris)
                rec["volume_change"] = round((v1 - sk.volume0) / sk.volume0, 5)
            ring = _joint_ring(sk, rig, k, a_world)
            if ring.size >= 8:
                posed_model = result.joints[arm._id][1]
                pivot0 = (a_world @ model[k])[:3, 3]
                pivot1 = (worlds[arm._id] @ posed_model[k])[:3, 3]
                r0 = np.linalg.norm(sk.p0[ring] - pivot0, axis=1)
                r1 = np.linalg.norm(p1[ring] - pivot1, axis=1)
                keep = float(np.mean(r1) / max(np.mean(r0), 1e-12))
                rec["joint_ring"] = round(keep, 4)
                # Interpenetration: the child side inside the parent segment's radius.
                parent = rig.joints[k].parent
                child_side = np.isin(sk.dominant, list(_subtree(rig, k)))
                if parent >= 0 and child_side.any():
                    a0 = (worlds[arm._id] @ posed_model[parent])[:3, 3]
                    b0 = pivot1
                    seg = b0 - a0
                    tt = np.clip((p1 - a0) @ seg / max(seg @ seg, 1e-12), 0.0, 1.0)
                    dist = np.linalg.norm(p1 - (a0 + tt[:, None] * seg), axis=1)
                    parent_side = sk.dominant == parent
                    if parent_side.any():
                        rad = float(np.median(dist[parent_side]))
                        inside = child_side & (dist < 0.6 * rad) & (tt > 0.05) & (tt < 0.95)
                        rec["interpenetrating"] = int(inside.sum())
                        if inside.sum() > 0.01 * child_side.sum():
                            add(
                                "INTERPENETRATION",
                                "info",
                                f"{pose['name']}: {int(inside.sum())} vertices of {sk.obj.name!r} "
                                "pass inside the parent segment",
                                mesh=sk.obj.name,
                                pose=pose["name"],
                                vertices=np.flatnonzero(inside)[:max_vertices].tolist(),
                                magnitude=int(inside.sum()),
                                unit="vertices",
                            )
                if keep < COLLAPSE_WARN:
                    add(
                        "JOINT_COLLAPSE",
                        "error" if keep < COLLAPSE_ERROR else "warning",
                        f"{pose['name']}: the skin around {rig.joints[k].name!r} keeps "
                        f"{keep:.0%} of its size ({sk.skin.method} skinning)",
                        mesh=sk.obj.name,
                        pose=pose["name"],
                        joints=[rig.joints[k].name],
                        vertices=ring[np.argsort(r1 / np.maximum(r0, 1e-12))][
                            :max_vertices
                        ].tolist(),
                        magnitude=round(keep, 4),
                        unit="fraction",
                    )
            if m["flipped"].size:
                add(
                    "FLIPPED_TRIANGLES",
                    "warning",
                    f"{pose['name']}: {m['flipped'].size} triangles of {sk.obj.name!r} turn "
                    "inside out",
                    mesh=sk.obj.name,
                    pose=pose["name"],
                    triangles=m["flipped"][:max_vertices].tolist(),
                    magnitude=int(m["flipped"].size),
                    unit="triangles",
                )
            if len(m["stretched"]) > 0.002 * len(sk.tris):
                add(
                    "STRETCHED_EDGES",
                    "info",
                    f"{pose['name']}: {len(m['stretched'])} edges of {sk.obj.name!r} stretch "
                    f"over {STRETCH:g}x (worst {m['worst_stretch']:.2f}x)",
                    mesh=sk.obj.name,
                    pose=pose["name"],
                    vertices=np.unique(m["stretched"])[:max_vertices].tolist(),
                    magnitude=round(m["worst_stretch"], 4),
                    unit="ratio",
                )
            entry[sk.obj.name] = rec
        pose_report[pose["name"]] = entry
    # -- animation over the timeline ----------------------------------------------------------
    timeline: dict[str, Any] = {}
    if animation and system.binding(arm) is not None:
        timeline = _timeline_checks(system, arm, rig, add)
    findings.sort(key=lambda f: SEVERITIES.index(f["severity"]))
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITIES}
    return {
        "armature": arm.name,
        "time": t,
        "joints": len(rig.joints),
        "meshes": [sk.obj.name for sk in skinned],
        "summary": counts,
        "findings": findings[:max_findings],
        "truncated": len(findings) > max_findings,
        "poses": pose_report,
        "timeline": timeline,
        "notes": [
            "every finding is measured (basis: numeric); how a pose should look is not judged",
            "joint_ring: the mean distance of the skin around a bent joint to its pivot, "
            "posed over rest (1: kept; below 0.75: collapsing)",
        ],
    }


def _named(poses: Any, rig) -> list[dict[str, Any]]:
    suite = {p["name"]: p for p in pose_suite(rig)}
    if not isinstance(poses, list) or not all(isinstance(p, str) for p in poses):
        raise _err("INVALID_ARGUMENT", "poses is 'suite', none or a list of pose names", "poses")
    out = []
    for p in poses:
        if p not in suite:
            raise _err(
                "INVALID_ARGUMENT", f"no diagnostic pose {p!r}; one of {sorted(suite)}", "poses"
            )
        out.append(suite[p])
    return out


def _timeline_checks(system, arm, rig, add) -> dict[str, Any]:
    start, end = system.start, system.end
    times = np.linspace(start, end, max(2, int((end - start) * 30) + 1))
    ik = [c for c in rig.constraints if c.type == "ik" and len(c.p["chain"]) == 3]
    flips: dict[str, list[float]] = {}
    worst: dict[str, tuple[float, float]] = {}
    prev: dict[str, np.ndarray] = {}
    for t in times.tolist():
        rr = system.rigging(t)
        found = rr.joints.get(arm._id)
        if found is None:
            continue
        _, model = found
        for c in ik:
            res = rr.residuals.get(f"{arm.name}/{c.name}")
            if res is not None and res > worst.get(c.name, (0.0, 0.0))[0]:
                worst[c.name] = (res, t)
            idx = [rig.index(n) for n in c.p["chain"]]
            d1 = model[idx[1]][:3, 3] - model[idx[0]][:3, 3]
            d2 = model[idx[2]][:3, 3] - model[idx[1]][:3, 3]
            n = np.cross(d1, d2)
            bend = float(np.linalg.norm(n) / max(np.linalg.norm(d1) * np.linalg.norm(d2), 1e-12))
            if bend < math.sin(math.radians(10)):
                prev.pop(c.name, None)
                continue
            n /= np.linalg.norm(n)
            if c.name in prev and float(n @ prev[c.name]) < 0.0:
                flips.setdefault(c.name, []).append(round(t, 4))
            prev[c.name] = n
    for name, ts in flips.items():
        add(
            "IK_FLIP",
            "warning",
            f"{name!r} flips its bend at {ts[:5]}",
            constraint=name,
            times=ts[:20],
            magnitude=len(ts),
            unit="flips",
        )
    for name, (res, t) in worst.items():
        if res > 0.01:
            add(
                "IK_UNREACHABLE",
                "warning",
                f"{name!r} misses its target by {res:.3f} at {t:.3f} s",
                constraint=name,
                magnitude=round(res, 6),
                unit="m",
                times=[round(t, 4)],
            )
    feet = [
        rig.joints[k].name for k in (_role(rig, "foot.L"), _role(rig, "foot.R")) if k is not None
    ]
    contacts = {}
    if feet:
        rep = system.foot_contacts(arm, feet, rate=60.0)
        for foot, d in rep["feet"].items():
            contacts[foot] = {
                "sliding": round(d["sliding_max"], 5),
                "penetration": round(d["penetration"], 5),
            }
            if d["sliding_max"] > 0.01:
                add(
                    "FOOT_SLIDING",
                    "warning",
                    f"{foot} slides {d['sliding_max']:.4f} m while planted",
                    joints=[foot],
                    magnitude=round(d["sliding_max"], 6),
                    unit="m",
                )
    return {
        "samples": int(times.size),
        "ik_flips": flips,
        "ik_worst": {
            k: {"residual": round(v[0], 6), "time": round(v[1], 4)} for k, v in worst.items()
        },
        "feet": contacts,
    }


# --------------------------------------------------------------------------------------- #
# Proposals and verified application
# --------------------------------------------------------------------------------------- #


def _fix_id(spec: dict[str, Any]) -> str:
    text = json.dumps(spec, sort_keys=True, default=str)
    return "fix-" + hashlib.sha256(text.encode()).hexdigest()[:12]


def propose_rig_fix(
    system: AnimationSystem, armature: Any, *, diagnosis: dict[str, Any] | None = None, **opts
) -> dict[str, Any]:
    """Proposals for ``armature``'s findings (from ``diagnosis`` or a fresh one)."""
    arm = system._object(armature, "armature")
    rig = system.rig(arm)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    diag = diagnosis or diagnose_rig(system, arm, **opts)
    proposals: list[dict[str, Any]] = []
    by_mesh: dict[str, Any] = {}
    for f in diag["findings"]:
        mesh = f.get("mesh")
        code = f["code"]
        if code == "WEIGHTS_NOT_NORMALIZED":
            ops = [{"call": "edit_weights", "mesh": mesh, "operations": [{"op": "normalize"}]}]
            proposals.append(_proposal(code, "safe", f"Normalize {mesh!r}'s weights", ops, f))
        elif code == "HELPER_JOINT_WEIGHTED":
            k = rig.index(f["joint"])
            p = rig.joints[k].parent
            while p >= 0 and not rig.joints[p].deform:
                p = rig.joints[p].parent
            if p >= 0:
                ops = [
                    {
                        "call": "edit_weights",
                        "mesh": mesh,
                        "operations": [
                            {"op": "replace", "from": f["joint"], "to": rig.joints[p].name}
                        ],
                    }
                ]
                proposals.append(
                    _proposal(
                        code,
                        "safe",
                        f"Move {f['joint']!r}'s weights to {rig.joints[p].name!r}",
                        ops,
                        f,
                    )
                )
        elif code in ("WEIGHT_LEAK", "CROSS_SIDE_WEIGHTS") and f.get("vertices"):
            ops = [
                {
                    "call": "edit_weights",
                    "mesh": mesh,
                    "operations": [
                        {"op": "rigid", "per": "vertex", "select": {"vertices": f["vertices"]}},
                        {"op": "smooth", "iterations": 2, "select": {"vertices": f["vertices"]}},
                    ],
                }
            ]
            proposals.append(
                _proposal(
                    code,
                    "verify",
                    f"Give {len(f['vertices'])} vertices of {mesh!r} back to their dominant joint",
                    ops,
                    f,
                )
            )
        elif code == "BIND_MISMATCH":
            ops = [{"call": "rebind", "mesh": mesh}]
            proposals.append(
                _proposal(code, "verify", f"Rebind {mesh!r} at the current rest", ops, f)
            )
        elif code == "JOINT_COLLAPSE" and mesh is not None:
            joint = f["joints"][0]
            sk = system._owned[("skin", system._object(mesh, "mesh")._id)]._definition
            if sk.method == "linear" or (sk.method == "blended"):
                ops = [
                    {
                        "call": "skin_method",
                        "mesh": mesh,
                        "method": "blended",
                        "dq": {"joint": joint, "share": "auto", "pose": f["pose"]},
                    }
                ]
                proposals.append(
                    _proposal(
                        code,
                        "verify",
                        f"Blend dual quaternion skinning into {mesh!r} around "
                        f"{joint!r} (keeps volume under bends and twists)",
                        ops,
                        f,
                    )
                )
            else:
                ops = [
                    {
                        "call": "edit_weights",
                        "mesh": mesh,
                        "operations": [
                            {
                                "op": "smooth",
                                "iterations": 4,
                                "select": {"near_joint": joint, "radius": 0.15},
                            }
                        ],
                    }
                ]
                proposals.append(
                    _proposal(code, "verify", f"Smooth {mesh!r}'s weights around {joint!r}", ops, f)
                )
        elif code in ("INTERPENETRATION", "STRETCHED_EDGES") and f.get("vertices") and mesh:
            ops = [
                {
                    "call": "edit_weights",
                    "mesh": mesh,
                    "operations": [
                        {
                            "op": "smooth",
                            "iterations": 3,
                            "strength": 0.5,
                            "select": {"vertices": f["vertices"]},
                        }
                    ],
                }
            ]
            proposals.append(
                _proposal(
                    code,
                    "verify",
                    f"Smooth {mesh!r} where {f['pose']} {code.lower().replace('_', ' ')}",
                    ops,
                    f,
                )
            )
        elif code == "ASYMMETRIC_JOINTS":
            side = (
                "L" if f["joints"][0].endswith(".L") or f["joints"][0].startswith("Left") else "R"
            )
            ops = [
                {
                    "call": "edit_rig",
                    "operations": [{"op": "mirror_joints", "joints": [f["joints"][0]]}],
                }
            ]
            proposals.append(
                _proposal(
                    code,
                    "approval",
                    f"Mirror {f['joints'][0]!r} onto {f['joints'][1]!r} (which "
                    f"side is right is a choice; this keeps side {side})",
                    ops,
                    f,
                )
            )
        elif code == "POLE_ON_LINE":
            ops = [{"call": "pole", "constraint": f["constraint"]}]
            proposals.append(
                _proposal(
                    code, "safe", f"Move {f['constraint']!r}'s pole off the chain's line", ops, f
                )
            )
        elif code == "FOOT_SLIDING":
            foot = f["joints"][0]
            k = rig.index(foot)
            knee = rig.joints[k].parent
            hip = rig.joints[knee].parent if knee >= 0 else -1
            if hip >= 0:
                ops = [
                    {
                        "call": "plant_feet",
                        "legs": {foot: [rig.joints[hip].name, rig.joints[knee].name, foot]},
                    }
                ]
                proposals.append(
                    _proposal(code, "verify", f"Plant {foot} (keyed IK per contact)", ops, f)
                )
        by_mesh.setdefault(mesh, 0)
    seen = set()
    unique = []
    for p in proposals:
        if p["id"] not in seen:
            seen.add(p["id"])
            unique.append(p)
    return {
        "armature": arm.name,
        "proposals": unique,
        "findings": len(diag["findings"]),
        "revision": system.scene.revision,
    }


def _proposal(code: str, safety: str, title: str, ops: list, finding: dict) -> dict[str, Any]:
    spec = {"code": code, "operations": ops}
    target = {k: finding[k] for k in ("mesh", "pose", "joints", "constraint") if k in finding}
    return {
        "id": _fix_id({**spec, "target": target}),
        "code": code,
        "safety": safety,
        "title": title,
        "operations": ops,
        "target": target,
        "before": {"magnitude": finding.get("magnitude"), "unit": finding.get("unit")},
    }


def apply_rig_fix(
    system: AnimationSystem,
    armature: Any,
    fixes: Any,
    *,
    dry_run: bool = False,
    verify: bool = True,
) -> dict[str, Any]:
    """Apply proposals (their dicts, or ``"safe"`` / ``"verify"``: every proposal of that
    safety and below from a fresh diagnosis) as one undo step; re-measure; undo when a
    ``verify`` proposal did not improve its target or anything got worse."""
    arm = system._object(armature, "armature")
    if fixes in ("safe", "verify"):
        allowed = ("safe",) if fixes == "safe" else ("safe", "verify")
        fixes = [p for p in propose_rig_fix(system, arm)["proposals"] if p["safety"] in allowed]
    if not isinstance(fixes, list) or len(fixes) > MAX_FIXES:
        raise _err(
            "INVALID_ARGUMENT",
            f"fixes is a list of at most {MAX_FIXES} proposals, 'safe' or 'verify'",
            "fixes",
        )
    for i, f in enumerate(fixes):
        _check_fix(system, arm, f, f"fixes[{i}]")
    if not fixes:
        return {"armature": arm.name, "applied": [], "reverted": [], "before": {}, "after": {}}
    before = diagnose_rig(system, arm, max_findings=1000)
    report: dict[str, Any] = {
        "armature": arm.name,
        "applied": [],
        "reverted": [],
        "dry_run": bool(dry_run),
    }

    class _Undo(Exception):
        pass

    after = before
    why: dict[str, Any] = {}
    try:
        with system._op(f"Fix the rig of {arm.name!r}"):
            # Each fix on its own: applied, measured, kept or undone (a nested operation
            # rolls back alone); all that is kept is one undo step.
            for f in fixes:
                current = after
                try:
                    with system._op(f.get("title") or f["code"]):
                        for op in f["operations"]:
                            _run_fix_op(system, arm, op)
                        measured = diagnose_rig(system, arm, max_findings=1000)
                        worse, unimproved = _compare(current, measured, [f])
                        if verify and f.get("safety") != "approval" and (worse or unimproved):
                            why[f.get("id")] = {"worse": worse, "unimproved": bool(unimproved)}
                            raise _Undo()
                except _Undo:
                    report["reverted"].append(f.get("id"))
                    continue
                after = measured
                report["applied"].append(f.get("id"))
            report["changes"] = _delta(before, after)
            if dry_run:
                raise _Undo()
    except _Undo:
        pass
    if why:
        report["why"] = why
    report["before"] = before["summary"]
    report["after"] = after["summary"]
    return report


MAX_FIXES = 200
PROPOSAL_KEYS = ("id", "code", "safety", "title", "operations", "target", "before")
_FIX_FIELDS = {
    "edit_weights": ("mesh", "operations"),
    "edit_rig": ("operations",),
    "rebind": ("mesh",),
    "skin_method": ("mesh", "method", "dq"),
    "pole": ("constraint",),
    "plant_feet": ("legs",),
}


def _check_fix(system, arm, f: Any, where: str) -> None:
    """A proposal's shape, checked before anything changes (proposals may come back from
    an agent: they are data, and each call's own arguments are validated when it runs)."""
    if not isinstance(f, dict) or not isinstance(f.get("operations"), list):
        raise _err("INVALID_ARGUMENT", "a fix is a proposal {code, operations, ...}", where)
    unknown = sorted(set(f) - set(PROPOSAL_KEYS))
    if unknown:
        raise _err("INVALID_ARGUMENT", f"unknown proposal key {unknown[0]!r}", where)
    if not isinstance(f.get("code"), str) or not f["code"]:
        raise _err("INVALID_ARGUMENT", "a proposal names its finding's code", f"{where}.code")
    if f.get("safety", "verify") not in ("safe", "verify", "approval"):
        raise _err("INVALID_ARGUMENT", "safety is safe, verify or approval", f"{where}.safety")
    if not 1 <= len(f["operations"]) <= 16:
        raise _err("INVALID_ARGUMENT", "a proposal has 1 to 16 operations", f"{where}.operations")
    rig = system.rig(arm)
    for j, op in enumerate(f["operations"]):
        at = f"{where}.operations[{j}]"
        call = op.get("call") if isinstance(op, dict) else None
        if call not in FIX_CALLS:
            raise _err(
                "INVALID_ARGUMENT", f"a fix operation's call is one of {', '.join(FIX_CALLS)}", at
            )
        allowed = _FIX_FIELDS[call]
        extra = sorted(set(op) - {"call", *allowed})
        if extra:
            raise _err("INVALID_ARGUMENT", f"{call} takes no {extra[0]!r}", at)
        required = tuple(k for k in allowed if k != "dq")
        for key in required:
            if key not in op:
                raise _err("INVALID_ARGUMENT", f"{call} needs {key!r}", at)
        if "mesh" in op:
            obj = system._object(op["mesh"], f"{at}.mesh")
            if ("skin", obj._id) not in system._owned:
                raise _err("INVALID_ARGUMENT", f"{obj.name!r} is not skinned", f"{at}.mesh")
        if "operations" in op and not isinstance(op["operations"], list):
            raise _err("INVALID_ARGUMENT", "operations is a list", f"{at}.operations")
        if call == "skin_method":
            if op["method"] not in METHODS:
                raise _err(
                    "INVALID_ARGUMENT", f"method is one of {', '.join(METHODS)}", f"{at}.method"
                )
            dq = op.get("dq", {})
            if not isinstance(dq, dict) or set(dq) - {"joint", "share", "pose"}:
                raise _err("INVALID_ARGUMENT", "dq is {joint, share, pose}", f"{at}.dq")
            share = dq.get("share", 1.0)
            number = isinstance(share, (int, float)) and not isinstance(share, bool)
            if not (share == "auto" or (number and 0.0 <= share <= 1.0)):
                raise _err(
                    "INVALID_ARGUMENT", "share is a number in 0..1 or 'auto'", f"{at}.dq.share"
                )
            if share == "auto" and not ("joint" in dq and "pose" in dq):
                raise _err(
                    "INVALID_ARGUMENT", "share 'auto' measures a joint in a pose", f"{at}.dq"
                )
            if "pose" in dq and dq["pose"] not in {q["name"] for q in pose_suite(rig)}:
                raise _err("INVALID_ARGUMENT", "no such diagnostic pose", f"{at}.dq.pose")
            if "joint" in dq and (
                not isinstance(dq["joint"], str) or rig.index(dq["joint"]) is None
            ):
                raise _err("INVALID_ARGUMENT", "no such joint", f"{at}.dq.joint")
        if call == "pole":
            c = next((c for c in rig.constraints if c.name == op["constraint"]), None)
            if c is None or not isinstance(c.p.get("pole"), dict) or "object" not in c.p["pole"]:
                raise _err(
                    "INVALID_ARGUMENT", "names an IK constraint with a pole object",
                    f"{at}.constraint",
                )  # fmt: skip
        if call == "plant_feet" and not isinstance(op["legs"], dict):
            raise _err("INVALID_ARGUMENT", "legs is {foot: [hip, knee, foot]}", f"{at}.legs")


def _run_fix_op(system, arm, op: dict[str, Any]) -> None:
    call = op["call"]
    if call == "edit_weights":
        system.edit_weights(op["mesh"], op["operations"])
    elif call == "edit_rig":
        system.edit_rig(arm, op["operations"])
    elif call == "rebind":
        _rebind(system, arm, op["mesh"])
    elif call == "skin_method":
        dq = op.get("dq") or {}
        if dq.get("share") == "auto":
            _blend_measured(system, arm, op["mesh"], op["method"], dq)
        else:
            _blend_around(system, op["mesh"], op["method"], dq)
    elif call == "pole":
        _move_pole(system, arm, op["constraint"])
    elif call == "plant_feet":
        system.plant_feet(arm, op["legs"])


def _rebind(system, arm, mesh) -> None:
    """New inverse binds at the current rest (the mesh stays as modeled at rest)."""
    from ..rendering.scene_data import world_matrices

    obj = system._object(mesh, "mesh")
    item = system._owned[("skin", obj._id)]
    sk = item._definition
    rig = system.rig(arm)
    worlds = world_matrices(system.scene)
    model = rig.rest_model()
    idx = [rig.index_of_id(j) for j in sk.joints]
    ib = np.linalg.inv(model[idx]) @ np.linalg.inv(worlds[arm._id]) @ worlds[obj._id]
    ib.flags.writeable = False
    with system._op(f"Rebind {obj.name!r}"):
        system._set(item, definition=sk.replace(inverse_bind=ib))


def _blend_around(system, mesh, method: str, dq: dict[str, Any]) -> None:
    """Blended skinning with a DQS share of ``share`` on the vertices a joint (and its
    parent) weigh on, fading out over the rest."""
    obj = system._object(mesh, "mesh")
    item = system._owned[("skin", obj._id)]
    sk = item._definition
    rig = system._items[sk.rig]._definition
    share = float(dq.get("share", 1.0))
    current = np.zeros(sk.vertex_count) if sk.dq is None else sk.dq.copy()
    joint = dq.get("joint")
    k = rig.index(joint) if isinstance(joint, str) else None
    if k is None:
        current[:] = share
    else:
        ids = {rig.joints[k].id}
        if rig.joints[k].parent >= 0:
            ids.add(rig.joints[rig.joints[k].parent].id)
        cols = [c for c, j in enumerate(sk.joints) if j in ids]
        w = sk.dense()[:, cols].sum(axis=1)
        current = np.maximum(current, np.clip(w * 2.0, 0.0, 1.0) * share)
    current = np.clip(current, 0.0, 1.0)
    current.flags.writeable = False
    with system._op(f"Blend the skinning of {obj.name!r}"):
        system._set(item, definition=sk.replace(method=method, dq=current))


SHARES = (0.25, 0.5, 0.75, 1.0)


def _blend_measured(system, arm, mesh, method: str, dq: dict[str, Any]) -> None:
    """The smallest dual quaternion share (of :data:`SHARES`) around ``dq["joint"]`` that
    keeps the skin there from collapsing in ``dq["pose"]`` -- measured, one share at a
    time (dual quaternions keep volume but bulge: no more of them than needed)."""
    obj = system._object(mesh, "mesh")

    class _Retry(Exception):
        pass

    for share in SHARES:
        try:
            with system._op("Blend the skinning"):  # (an exception rolls the attempt back)
                _blend_around(system, mesh, method, {**dq, "share": share})
                d = diagnose_rig(
                    system, arm, poses=[dq["pose"]], animation=False, max_findings=1000
                )
                collapsed = any(
                    f["code"] == "JOINT_COLLAPSE"
                    and f.get("mesh") == obj.name
                    and dq["joint"] in (f.get("joints") or ())
                    for f in d["findings"]
                )
                if collapsed and share != SHARES[-1]:
                    raise _Retry()
        except _Retry:
            continue
        return


def _move_pole(system, arm, name: str) -> None:
    from ..rendering.scene_data import world_matrices

    rig = system.rig(arm)
    c = next(c for c in rig.constraints if c.name == name)
    pole = c.p["pole"]
    obj = system.scene._objects[pole["object"]]
    worlds = world_matrices(system.scene)
    model = worlds[arm._id] @ rig.rest_model()
    idx = [rig.index(n) for n in c.p["chain"]]
    a, mid, b = (model[i][:3, 3] for i in idx)
    fwd = {"-y": (0, -1, 0), "+y": (0, 1, 0), "-x": (-1, 0, 0), "+x": (1, 0, 0)}[
        rig.forward_axis("-y")
    ]
    side = worlds[arm._id][:3, :3] @ np.array(fwd, float)
    ab = b - a
    perp = side - (side @ ab) / max(ab @ ab, 1e-12) * ab
    if np.linalg.norm(perp) < 1e-9:
        perp = np.cross(ab, [0.0, 0.0, 1.0])
    target = mid + perp / np.linalg.norm(perp) * float(np.linalg.norm(ab))
    parent = obj.parent
    local = (
        target if parent is None else np.linalg.solve(worlds[parent.id], np.append(target, 1))[:3]
    )
    with system._op(f"Move the pole of {name!r}"):
        obj.position = tuple(float(v) for v in local)


def _key(f: dict[str, Any]) -> tuple:
    return (
        f["code"],
        f.get("mesh"),
        f.get("pose"),
        tuple(f.get("joints") or ()),
        f.get("constraint"),
    )


def _compare(before, after, fixes) -> tuple[list, list]:
    """What got worse anywhere, and which fixes' targets did not improve."""
    rank = {"error": 2, "warning": 1, "info": 0}
    b = {_key(f): f for f in before["findings"]}
    a = {_key(f): f for f in after["findings"]}
    worse = []
    for k, f in a.items():
        if f["severity"] == "info":
            continue
        old = b.get(k)
        if old is None or rank[f["severity"]] > rank[old["severity"]]:
            worse.append(f["code"])
        else:
            new_m, old_m = _amount(f), _amount(old)
            if new_m is None or old_m is None:
                continue
            if f["code"] == "JOINT_COLLAPSE":
                if new_m < old_m - 0.02:
                    worse.append(f["code"])
            elif new_m > 1.1 * old_m + 1e-9:
                worse.append(f["code"])
    unimproved = []
    for fix in fixes:
        if fix.get("safety") != "verify":
            continue
        tgt = fix.get("target", {})
        key = (
            fix["code"],
            tgt.get("mesh"),
            tgt.get("pose"),
            tuple(tgt.get("joints") or ()),
            tgt.get("constraint"),
        )
        if key not in b:
            continue
        old = b[key].get("magnitude")
        new = a.get(key, {}).get("magnitude") if key in a else None
        if new is None:
            continue  # resolved
        better = (new > old) if fix["code"] == "JOINT_COLLAPSE" else (new < old)
        if old is not None and not better:
            unimproved.append(fix.get("id"))
    return sorted(set(worse)), unimproved


def _amount(f: dict[str, Any]) -> float | None:
    for key in ("magnitude", "count"):
        v = f.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _delta(before, after) -> dict[str, Any]:
    out = {}
    for pose, meshes in after["poses"].items():
        for mesh, rec in meshes.items():
            old = before["poses"].get(pose, {}).get(mesh, {})
            if (
                "joint_ring" in rec
                and "joint_ring" in old
                and rec["joint_ring"] != old["joint_ring"]
            ):
                out[f"{pose}/{mesh}/joint_ring"] = [old["joint_ring"], rec["joint_ring"]]
    return out
