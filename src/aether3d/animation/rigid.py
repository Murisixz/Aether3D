"""Rigid-part characters: animation without bones, and optional conversions (Stage 18).

Three separate workflows, chosen by the user:

1. **Rigid transforms** (this module): a Minecraft-style character is a hierarchy of
   solid parts -- head, torso, arms, legs -- moved by their own object transforms. No
   armature, no skin, no weights: :func:`generate_rigid` writes walk, run, jump, idle,
   wave... as ordinary ``transform`` clips on the part objects (and travel on the root),
   and rendering pays for nothing but object matrices.
2. **Bone-attached rigid parts** (Stage 17's ``attach``): blocks follow the joints of an
   armature -- for generators that need a skeleton's IK, retargeting, constraints.
3. **Skinned rigs** (``autorig``, ``skin``): organic characters deform smoothly.

:func:`convert_character` converts between the first two on request: a rigid hierarchy
to an armature with attached copies of its parts (``to="skeleton"``), or an armature's
attached parts to a rigid hierarchy of pivots (``to="rigid"``) -- each with its motion
baked and verified sample by sample. The original is never changed (only hidden, when
asked) and an undo removes the conversion.

**Parts.** :func:`find_parts` takes the roles ``head``, ``torso``, ``arm.L``, ``arm.R``,
``leg.L``, ``leg.R`` from ``parts`` (``{role: object}``) or from object names (``head``;
``torso``/``body``/``chest``; ``arm``; ``leg``) under the root; the side is measured --
left is ``up x forward`` (``+X`` for a character facing ``-Y``) -- and a name that says
otherwise is reported (``SIDE_NAME_MISMATCH``). Joint pivots come from each part's box:
shoulders half an arm's width below the top (a Minecraft shoulder), hips at the top of
the legs, the neck under the head, the waist under the torso -- or the part's own origin
when it is already there (within a quarter of the part's length).

**How a rigid motion is made.** The motion generators (``generators``) run unchanged on
a skeleton built in a private scratch scene from the parts' pivots; the parts there are
attached proxies. Their exact motion is then sampled (240 per second) and written as each
real part's local transform -- rotation, and position where its origin is not at its
pivot -- with keys chosen until every sample is within ``tolerance`` (the generators' own
fitting). The report gives the fitting errors and a verification: the parts' world
matrices in the scene against the proxies' at 49 times.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import rotation as R

if TYPE_CHECKING:
    from .system import AnimationSystem

PART_ROLES = ("head", "torso", "arm.L", "arm.R", "leg.L", "leg.R")
JOINTS = {
    "hips": "Hips",
    "torso": "Spine",
    "head": "Head",
    "arm.L": "UpperArm.L",
    "arm.R": "UpperArm.R",
    "leg.L": "Thigh.L",
    "leg.R": "Thigh.R",
}
JOINT_ROLES = {
    "hips": "hips",
    "torso": "spine",
    "head": "head",
    "arm.L": "upper_arm.L",
    "arm.R": "upper_arm.R",
    "leg.L": "thigh.L",
    "leg.R": "thigh.R",
}
_WORDS = {
    "head": ("head",),
    "torso": ("torso", "body", "chest", "trunk"),
    "arm": ("arm",),
    "leg": ("leg",),
}
_FORWARD = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}
VERIFY_SAMPLES = 49
MAX_PART_OBJECTS = 2000


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


# --------------------------------------------------------------------------------------- #
# Parts
# --------------------------------------------------------------------------------------- #


@dataclass
class Part:
    role: str
    obj: Any
    center: np.ndarray  # armature space (the root's local frame)
    pivot: np.ndarray
    tail: np.ndarray  # the far end, relative to the pivot
    source: str  # "origin" or "bounds"
    length: float


@dataclass
class PartSet:
    root: Any
    forward: str
    parts: dict[str, Part]
    hips: np.ndarray
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def report(self, world: np.ndarray) -> dict[str, Any]:
        def w(p: np.ndarray) -> list[float]:
            return [round(float(v), 6) for v in (world @ np.append(p, 1.0))[:3]]

        return {
            "root": self.root.name,
            "forward": self.forward,
            "parts": {
                role: {
                    "object": p.obj.name,
                    "pivot": w(p.pivot),
                    "pivot_source": p.source,
                    "end": w(p.pivot + p.tail),
                    "length": round(p.length, 6),
                }
                for role, p in self.parts.items()
            },
            "hips": w(self.hips),
            "diagnostics": self.diagnostics,
        }


def _words(name: str) -> list[str]:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [w for w in re.split(r"[^a-z0-9]+", spaced.lower()) if w]


def _named_side(name: str) -> str | None:
    words = _words(name)
    if "left" in words or "l" in words or re.search(r"[._-]l$", name.lower()):
        return "L"
    if "right" in words or "r" in words or re.search(r"[._-]r$", name.lower()):
        return "R"
    return None


def _kind_of(name: str) -> str | None:
    words = _words(name)
    found = [k for k, keys in _WORDS.items() if any(w in keys for w in words)]
    if len(found) == 1:
        return found[0]
    # (joined words: "leftarm", "RightLeg" without case change)
    low = re.sub(r"[^a-z]", "", name.lower())
    found = [k for k, keys in _WORDS.items() if any(low.endswith(w) for w in keys)]
    return found[0] if len(found) == 1 else None


def _subtree(obj: Any) -> list[Any]:
    return list(obj.iter_descendants())


def _local_box(obj: Any, exclude: set[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """The part's box in its own frame: its mesh, else its descendants' meshes (other
    parts excluded)."""
    if obj.mesh is not None and obj.mesh.vertex_count:
        lo, hi = obj.mesh.bounds()
        return np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    inv = np.linalg.inv(obj.world_matrix)
    pts = []
    stack = list(obj.children)
    while stack:
        d = stack.pop()
        if d.id in exclude:
            continue
        stack.extend(d.children)
        if d.mesh is None or not d.mesh.vertex_count:
            continue
        lo, hi = (np.asarray(v, np.float64) for v in d.mesh.bounds())
        corners = np.array(
            [[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        pts.append((inv @ d.world_matrix @ corners.T).T[:, :3])
    if not pts:
        return None
    allp = np.concatenate(pts)
    return allp.min(axis=0), allp.max(axis=0)


class _Box:
    """A part's box in armature space: faces (center, outward normal), extents."""

    def __init__(self, to_arm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> None:
        self.m = to_arm
        mid = 0.5 * (lo + hi)
        self.center = (to_arm @ np.append(mid, 1.0))[:3]
        axes = to_arm[:3, :3]
        norms = np.linalg.norm(axes, axis=0)
        self.sizes = (hi - lo) * norms
        self.faces = []
        for i in range(3):
            for sign, v in ((1.0, hi[i]), (-1.0, lo[i])):
                c = mid.copy()
                c[i] = v
                n = sign * axes[:, i] / max(norms[i], 1e-12)
                self.faces.append(((to_arm @ np.append(c, 1.0))[:3], n, i))
        self.long = int(np.argmax(self.sizes))

    def face_toward(self, point: np.ndarray, axis: int | None = None):
        faces = [f for f in self.faces if axis is None or f[2] == axis]
        return min(faces, key=lambda f: float(np.linalg.norm(f[0] - point)))

    def face_along(self, direction: np.ndarray):
        return max(self.faces, key=lambda f: float(f[1] @ direction))

    def opposite(self, face):
        c, n, i = face
        return next(f for f in self.faces if f[2] == i and float(f[1] @ n) < 0)


def find_parts(
    system: AnimationSystem,
    root: Any,
    *,
    parts: dict[str, Any] | None = None,
    forward: str = "-y",
) -> PartSet:
    """The rigid parts of the character under ``root`` (see the module docstring)."""
    R_obj = system._object(root, "root")
    if forward not in _FORWARD:
        raise _err("INVALID_ARGUMENT", "forward is +x, -x, +y or -y", "forward")
    if system.rig(R_obj) is not None:
        raise _err(
            "INVALID_ARGUMENT",
            f"{R_obj.name!r} is an armature (it has a rig)",
            "root",
            hint="rigid workflows take the root of a hierarchy of parts",
        )
    tree = _subtree(R_obj)
    if len(tree) > MAX_PART_OBJECTS:
        raise _err("LIMIT_EXCEEDED", f"more than {MAX_PART_OBJECTS} objects under the root", "root")
    in_tree = {o.id: o for o in tree}
    diagnostics: list[dict[str, Any]] = []
    chosen: dict[str, Any] = {}
    if parts is not None:
        if not isinstance(parts, dict) or not parts:
            raise _err("INVALID_ARGUMENT", "parts is {role: object}", "parts")
        for role, ref in parts.items():
            if role not in PART_ROLES:
                raise _err(
                    "INVALID_ARGUMENT",
                    f"unknown role {role!r} (roles: {', '.join(PART_ROLES)})",
                    f"parts.{role}",
                )
            obj = system._object(ref, f"parts.{role}")
            if obj.id not in in_tree:
                raise _err(
                    "INVALID_ARGUMENT", f"{obj.name!r} is not under {R_obj.name!r}", f"parts.{role}"
                )
            chosen[role] = obj
        named = list(chosen.items())
    else:
        found: dict[str, list[Any]] = {"head": [], "torso": [], "arm": [], "leg": []}
        depth = {}
        for o in tree:
            kind = _kind_of(o.name)
            if kind is not None:
                found[kind].append(o)
                depth[o.id] = sum(1 for _ in o.iter_ancestors())
        for kind, objs in found.items():
            # Shallowest first: a part's own pieces ("Head Visor") are inside it.
            objs.sort(key=lambda o: depth[o.id])
            keep = [
                o for o in objs if not any(a.id in {x.id for x in objs} for a in o.iter_ancestors())
            ]
            want = 2 if kind in ("arm", "leg") else 1
            if len(keep) > want:
                raise _err(
                    "AMBIGUOUS_PART",
                    f"{len(keep)} objects look like a {kind}: "
                    + ", ".join(o.name for o in keep[:6]),
                    "parts",
                    hint="name the parts: parts={role: object}",
                )
            found[kind] = keep
        named = [("head", o) for o in found["head"]] + [("torso", o) for o in found["torso"]]
        named += [("arm", o) for o in found["arm"]] + [("leg", o) for o in found["leg"]]
    if not any(r.startswith(("leg", "arm", "head", "torso")) for r, _ in named):
        raise _err(
            "PARTS_NOT_FOUND",
            f"no head, torso, arm or leg under {R_obj.name!r}",
            "parts",
            hint="name the parts: parts={role: object}",
        )
    ids = {o.id for _, o in named}
    to_arm_root = np.linalg.inv(R_obj.world_matrix)
    boxes: dict[str, _Box] = {}
    objs: dict[str, Any] = {}
    for role, o in named:
        lb = _local_box(o, ids - {o.id})
        if lb is None:
            raise _err("PART_EMPTY", f"{o.name!r} has no geometry", "parts")
        key = role if role not in ("arm", "leg") else f"{role}:{o.id}"
        boxes[key] = _Box(to_arm_root @ o.world_matrix, *lb)
        objs[key] = o
    up = np.array([0.0, 0.0, 1.0])
    fwd = np.asarray(_FORWARD[forward], np.float64)
    left = np.cross(up, fwd)
    torso = boxes.get("torso")
    body_center = (
        torso.center if torso is not None else np.mean([b.center for b in boxes.values()], axis=0)
    )
    # Sides, measured.
    roles: dict[str, str] = {}
    for kind in ("arm", "leg"):
        keys = [k for k in boxes if k.startswith(kind + ":")] + [
            k for k in boxes if k in (f"{kind}.L", f"{kind}.R")
        ]
        explicit = [k for k in keys if "." in k and ":" not in k]
        for k in explicit:
            roles[k] = k
        auto = [k for k in keys if ":" in k]
        if not auto:
            continue
        sides = {k: float((boxes[k].center - body_center) @ left) for k in auto}
        if len(auto) == 2:
            a, b = sorted(auto, key=lambda k: sides[k])
            if sides[a] == sides[b]:
                raise _err(
                    "AMBIGUOUS_SIDE",
                    f"both {kind}s are on the same side",
                    "parts",
                    hint="name them: parts={'arm.L': ..., 'arm.R': ...}",
                )
            assign = {b: f"{kind}.L", a: f"{kind}.R"}
        else:
            k = auto[0]
            assign = {k: f"{kind}.L" if sides[k] >= 0 else f"{kind}.R"}
        for k, role in assign.items():
            roles[k] = role
            said = _named_side(objs[k].name)
            if said is not None and said != role[-1]:
                diagnostics.append(
                    {
                        "code": "SIDE_NAME_MISMATCH",
                        "severity": "warning",
                        "message": f"{objs[k].name!r} is named {('left', 'right')[said == 'R']} "
                        f"but is on the character's {('left', 'right')[role[-1] == 'R']} (left is "
                        f"up x forward; forward {forward}): the geometry decides",
                        "object": objs[k].name,
                    }
                )
    for k in boxes:
        if k in ("head", "torso"):
            roles[k] = k
    out: dict[str, Part] = {}
    torso_top = torso.face_along(up)[0] if torso is not None else None
    legs = [k for k in boxes if roles[k].startswith("leg")]
    for k, box in boxes.items():
        role = roles[k]
        o = objs[k]
        origin = (to_arm_root @ o.world_matrix)[:3, 3]
        if role.startswith("arm"):
            ref = torso_top if torso_top is not None else box.center + up
            near = box.face_toward(ref, axis=box.long)
            thick = float(min(np.delete(box.sizes, box.long)))
            pivot = near[0] - near[1] * (0.5 * thick)
            far = box.opposite(near)
        elif role.startswith("leg"):
            ref = torso.center if torso is not None else box.center + up
            near = box.face_toward(ref, axis=box.long)
            pivot = near[0]
            far = box.opposite(near)
        elif role == "head":
            near = box.face_toward(torso_top) if torso_top is not None else box.face_along(-up)
            pivot = near[0]
            far = box.opposite(near)
        else:  # torso
            if legs:
                ref = np.mean([boxes[k2].face_along(up)[0] for k2 in legs], axis=0)
                near = box.face_toward(ref)
            else:
                near = box.face_along(-up)
            pivot = near[0]
            far = box.opposite(near)
        length = float(np.linalg.norm(far[0] - near[0]))
        source = "bounds"
        if float(np.linalg.norm(origin - pivot)) <= 0.25 * length:
            pivot, source = origin, "origin"
        out[role] = Part(role, o, box.center, pivot, far[0] - pivot, source, length)
    leg_parts = [p for r, p in out.items() if r.startswith("leg")]
    if leg_parts:
        hips = np.mean([p.pivot for p in leg_parts], axis=0)
    elif "torso" in out:
        hips = out["torso"].pivot.copy()
    else:
        hips = np.zeros(3)
    for role in PART_ROLES:
        if role not in out:
            diagnostics.append(
                {
                    "code": "PART_MISSING",
                    "severity": "info",
                    "message": f"no {role} part (motions skip it)",
                }
            )
    return PartSet(R_obj, forward, out, hips, diagnostics)


def _joints(ps: PartSet) -> list[dict[str, Any]]:
    """A skeleton through the parts' pivots (rest rotations identity: armature axes)."""
    pv = {r: p.pivot for r, p in ps.parts.items()}
    joints = [
        {"name": "Hips", "position": ps.hips.tolist(), "role": "hips", "tail": [0.0, 0.0, 0.05]}
    ]
    spine_parent = "Hips"
    if "torso" in ps.parts:
        p = ps.parts["torso"]
        joints.append(
            {
                "name": "Spine",
                "parent": "Hips",
                "role": "spine",
                "position": (p.pivot - ps.hips).tolist(),
                "tail": p.tail.tolist(),
            }
        )
        spine_parent = "Spine"
    base = pv.get("torso", ps.hips)
    for role in ("head", "arm.L", "arm.R"):
        if role in ps.parts:
            p = ps.parts[role]
            joints.append(
                {
                    "name": JOINTS[role],
                    "parent": spine_parent,
                    "role": JOINT_ROLES[role],
                    "position": (p.pivot - base).tolist(),
                    "tail": p.tail.tolist(),
                }
            )
    for role in ("leg.L", "leg.R"):
        if role in ps.parts:
            p = ps.parts[role]
            joints.append(
                {
                    "name": JOINTS[role],
                    "parent": "Hips",
                    "role": JOINT_ROLES[role],
                    "position": (p.pivot - ps.hips).tolist(),
                    "tail": p.tail.tolist(),
                }
            )
    return joints


def describe_parts(
    system: AnimationSystem,
    root: Any,
    *,
    parts: dict[str, Any] | None = None,
    forward: str = "-y",
) -> dict[str, Any]:
    """What :func:`find_parts` measured: roles, pivots (world), ends, diagnostics."""
    ps = find_parts(system, root, parts=parts, forward=forward)
    out = ps.report(ps.root.world_matrix)
    animated = [o.name for o in [ps.root, *_subtree(ps.root)] if system.action(o) is not None]
    out["animated"] = animated
    out["rig"] = False
    return out


# --------------------------------------------------------------------------------------- #
# Sampling and fitting
# --------------------------------------------------------------------------------------- #


def _decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (T, 3), quaternions (T, 4) and scales (T, 3) of affine matrices."""
    basis = m[:, :3, :3]
    scale = np.linalg.norm(basis, axis=1)
    det = np.linalg.det(basis)
    scale[:, 0] = np.where(det < 0, -scale[:, 0], scale[:, 0])
    rot = basis / scale[:, None, :]
    u, _, vt = np.linalg.svd(rot)
    rot = u @ vt
    return m[:, :3, 3].copy(), R.from_matrices(rot), scale


def _fit_local(
    times: np.ndarray,
    local: np.ndarray,
    rest: np.ndarray,
    tol_deg: float,
    pos_tol: float,
    cyclic: bool,
    prefix: str = "transform",
) -> tuple[dict[str, Any], dict[str, float], int]:
    """Curves for sampled local matrices (channels that never leave the rest are left
    out): positions and rotations (XYZ Euler, or a quaternion track near gimbal lock)."""
    from .generators import GIMBAL_MARGIN, _fit_scalar, _fit_track

    pos, quat, scale = _decompose(local)
    rpos, rquat, rscale = _decompose(rest[None])
    extrap = "cycle" if cyclic else "constant"
    channels: dict[str, Any] = {}
    errors = {"degrees": 0.0, "units": 0.0}
    keys = 0
    if not np.allclose(np.abs(np.sum(quat * rquat, axis=1)), 1.0, atol=1e-12):
        q = quat.copy()
        for i in range(1, len(q)):
            q[i] = R.align(q[i - 1], q[i])
        e = R.euler_filter(R.to_euler(q))
        if cyclic and not np.allclose(e[-1], e[0], atol=1e-6):
            e[-1] = e[0]
        if np.max(np.abs(e[:, 1])) > GIMBAL_MARGIN:
            track, err, nk = _fit_track(times, q, tol_deg, cyclic)
            channels[f"{prefix}.rotation"] = track
            errors["degrees"] = max(errors["degrees"], err)
            keys += nk
        else:
            for c, axis in enumerate("xyz"):
                curve, err, nk = _fit_scalar(times, e[:, c], tol_deg, cyclic, extrap)
                channels[f"{prefix}.rotation_euler.{axis}"] = curve
                errors["degrees"] = max(errors["degrees"], err)
                keys += nk
    for field_name, values, base, tol in (
        ("position", pos, rpos[0], pos_tol),
        ("scale", scale, rscale[0], 1e-6),
    ):
        for c, axis in enumerate("xyz"):
            v = values[:, c]
            if np.allclose(v, base[c], atol=1e-9):
                continue
            curve, err, nk = _fit_scalar(times, v, tol, cyclic, extrap)
            channels[f"{prefix}.{field_name}.{axis}"] = curve
            if field_name == "position":
                errors["units"] = max(errors["units"], err)
            keys += nk
    return channels, errors, keys


def _posed_worlds(system: AnimationSystem, times: np.ndarray, ids: list[str]) -> np.ndarray:
    """World matrices (T, len(ids), 4, 4) of objects over ``times`` (the evaluated pose)."""
    from ..rendering.scene_data import world_matrices

    out = np.empty((len(times), len(ids), 4, 4))
    for i, t in enumerate(times.tolist()):
        w = world_matrices(system.scene, system.overlay(float(t)))
        for j, oid in enumerate(ids):
            out[i, j] = w[oid]
    return out


def _parent_world(obj: Any) -> np.ndarray:
    return obj.parent.world_matrix if obj.parent is not None else np.eye(4)


def _posed_parent(system: AnimationSystem, obj: Any, times: np.ndarray) -> np.ndarray:
    """The world matrices (T, 4, 4) of ``obj``'s parent over ``times`` (identity at the
    scene root)."""
    if obj.parent is None:
        return np.broadcast_to(np.eye(4), (len(times), 4, 4)).copy()
    return _posed_worlds(system, times, [obj.parent.id])[:, 0]


def _set_local(obj: Any, m: np.ndarray) -> float:
    """Give ``obj`` the local matrix ``m`` (as TRS); returns the representation error."""
    from ..math3d import compose_matrix, decompose_matrix

    p, q, s = decompose_matrix(m)
    obj.position, obj.rotation, obj.scale = p, q, s
    return float(np.abs(compose_matrix(p, q, s) - m).max())


def _verify(
    expected: np.ndarray, got: np.ndarray, names: list[str]
) -> dict[str, Any]:  # (T, n, 4, 4) each
    dp = np.linalg.norm(expected[..., :3, 3] - got[..., :3, 3], axis=-1)
    ra = expected[..., :3, :3] / np.linalg.norm(expected[..., :3, :3], axis=-2, keepdims=True)
    rb = got[..., :3, :3] / np.linalg.norm(got[..., :3, :3], axis=-2, keepdims=True)
    tr = np.einsum("...ij,...ij->...", ra, rb)
    ang = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
    worst = int(np.unravel_index(int(dp.argmax()), dp.shape)[1]) if dp.size else 0
    return {
        "samples": int(expected.shape[0]),
        "objects": len(names),
        "position": round(float(dp.max()) if dp.size else 0.0, 9),
        "rotation_deg": round(float(ang.max()) if ang.size else 0.0, 6),
        "worst": names[worst] if names else None,
    }


# --------------------------------------------------------------------------------------- #
# Generation without bones
# --------------------------------------------------------------------------------------- #


def _scratch(ps: PartSet):
    """A private scene with a skeleton through the parts and attached proxies (world
    frame: the root's parent's)."""
    from ..scene import Scene

    R_obj = ps.root
    scratch = Scene("rigid parts (scratch)")
    arm = scratch.add_object(
        "Character", position=R_obj.position, rotation=R_obj.rotation, scale=R_obj.scale
    )
    a = scratch.animation
    a.add_rig(arm, _joints(ps))
    base = np.linalg.inv(_parent_world(R_obj))
    proxies: dict[str, Any] = {}
    for role, p in ps.parts.items():
        proxy = scratch.add_object(f"proxy {role}")
        _set_local(proxy, base @ p.obj.world_matrix)
        a.attach(proxy, arm, JOINTS[role])
        proxies[role] = proxy
    return scratch, arm, proxies


def _scratch_motion(sa, arm, clip, times: np.ndarray, proxies: dict[str, Any]):
    """The scratch armature's world matrices (T, 4, 4) and the proxies' (role -> (T, 4, 4))
    over ``times``, from the clip's curves (its skeleton has no constraints or limits:
    forward kinematics is the whole evaluation; the scene check in the report confirms)."""
    from ..math3d import compose_matrix
    from .rig import forward_kinematics, local_matrices

    rig = sa.rig(arm)
    chans = dict(clip.channels)
    n, J = len(times), len(rig.joints)
    pos0, rot0, scl0 = rig.rest_trs()
    pos = np.broadcast_to(pos0, (n, J, 3)).copy()
    rot = np.broadcast_to(rot0, (n, J, 4)).copy()
    for k, joint in enumerate(rig.joints):
        key = f'joints["{joint.name}"]'
        track = chans.get(f"{key}.rotation")
        if track is not None:
            rot[:, k] = track.evaluate(times)
        elif any(f"{key}.rotation_euler.{a}" in chans for a in "xyz"):
            e0 = R.to_euler(rot0[k][None])[0]
            e = np.stack(
                [
                    chans[f"{key}.rotation_euler.{a}"].evaluate(times)
                    if f"{key}.rotation_euler.{a}" in chans
                    else np.full(n, e0[c])
                    for c, a in enumerate("xyz")
                ],
                axis=1,
            )
            rot[:, k] = R.from_euler(e)
        for c, a in enumerate("xyz"):
            curve = chans.get(f"{key}.position.{a}")
            if curve is not None:
                pos[:, k, c] = curve.evaluate(times)
    scl = np.broadcast_to(scl0, (n, J, 3))
    local = local_matrices(pos.reshape(-1, 3), rot.reshape(-1, 4), scl.reshape(-1, 3))
    local = local.reshape(n, J, 4, 4)
    model = np.stack([forward_kinematics(local[i], rig.parents, rig.depth) for i in range(n)])
    apos = np.broadcast_to(np.asarray(arm.position, np.float64), (n, 3)).copy()
    for c, a in enumerate("xyz"):
        curve = chans.get(f"transform.position.{a}")
        if curve is not None:
            apos[:, c] = curve.evaluate(times)
    A0 = compose_matrix(arm.position, arm.rotation, arm.scale)
    A = np.broadcast_to(A0, (n, 4, 4)).copy()
    A[:, :3, 3] = apos
    model0 = rig.rest_model()
    out = {}
    for role, proxy in proxies.items():
        k = rig.index(JOINTS[role])
        # The attachment rule: W' = A' J' (A J)^-1 W.
        rest = np.linalg.inv(A0 @ model0[k]) @ proxy.world_matrix
        out[role] = A @ model[:, k] @ rest
    return A, out


def _chain(ps: PartSet) -> list[Any]:
    """The parts and every object between them and the root, ancestors first."""
    keep: dict[str, Any] = {}
    for p in ps.parts.values():
        node = p.obj
        while node is not None and node.id != ps.root.id:
            keep[node.id] = node
            node = node.parent
    depth = {oid: sum(1 for _ in o.iter_ancestors()) for oid, o in keep.items()}
    return sorted(keep.values(), key=lambda o: depth[o.id])


def generate_rigid(
    system: AnimationSystem,
    root: Any,
    kind: str,
    *,
    parts: dict[str, Any] | None = None,
    name: str | None = None,
    forward: str = "-y",
    start: float = 0.0,
    travel: bool | None = None,
    tolerance: float = 0.05,
    bind: bool = True,
    **params: Any,
) -> dict[str, Any]:
    """A motion for a rigid-part character: ordinary ``transform`` clips on its parts
    (and travel on the root), no armature in the scene (see the module docstring).
    One undo step."""
    from .clip import Clip
    from .generators import RATE

    ps = find_parts(system, root, parts=parts, forward=forward)
    R_obj = ps.root
    scratch, arm, proxies = _scratch(ps)
    sa = scratch.animation
    gen = sa.generate(
        arm,
        kind,
        name="motion",
        forward=forward,
        start=start,
        travel=travel,
        tolerance=tolerance,
        bind=True,
        **params,
    )
    duration, cyclic = float(gen["duration"]), bool(gen["cyclic"])
    count = int(round(duration * RATE))
    times = start + np.linspace(0.0, duration, count + 1)
    motion = sa._clip_item(gen["clip"])._definition
    A_t, proxy_worlds = _scratch_motion(sa, arm, motion, times, proxies)
    worlds = {R_obj.id: A_t}
    for r, w in proxy_worlds.items():
        worlds[ps.parts[r].obj.id] = w
    chain = _chain(ps)
    for o in chain:  # (objects between the root and the parts follow their parents)
        if o.id not in worlds:
            worlds[o.id] = worlds[o.parent.id] @ o.local_matrix
    leg = max((p.length for r, p in ps.parts.items() if r.startswith("leg")), default=1.0)
    pos_tol = tolerance / 1000.0 * leg * 20.0
    clips: dict[str, Clip] = {}
    fit = {"degrees": 0.0, "units": 0.0}
    keys = 0
    meta = {
        "generator": {
            "kind": kind,
            "version": 1,
            "rigid": True,
            "forward": forward,
            "travel": gen["travel"] > 0,
            "tolerance": tolerance,
            "start": start,
            "params": dict(params),
        }
    }
    text = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False)
    from . import paths as P

    for p in ps.parts.values():
        o = p.obj
        local = np.linalg.inv(worlds[o.parent.id]) @ worlds[o.id]
        channels, err, nk = _fit_local(times, local, o.local_matrix, tolerance, pos_tol, cyclic)
        fit = {k: max(fit[k], err[k]) for k in fit}
        keys += nk
        if channels:
            clips[o.id] = Clip(P.OBJECT, tuple(sorted(channels.items())), (), None, text)
    root_channels = {path: ch for path, ch in motion.channels if path.startswith("transform.")}
    if root_channels:
        clips[R_obj.id] = Clip(P.OBJECT, tuple(sorted(root_channels.items())), (), None, text)
        keys += sum(len(ch) for ch in root_channels.values())
    base_name = name or system._unique_clip_name(kind.replace("_", " ").title())
    names: dict[str, str] = {}
    cleared: list[str] = []
    scene = system.scene
    with system._op(f"Generate {kind} for {R_obj.name!r} (rigid parts)"):
        for oid, clip in clips.items():
            obj = scene._objects[oid]
            clip_name = f"{base_name} {obj.name}"
            if clip_name in system._clip_names:
                clip_name = system._unique_clip_name(clip_name)
            handle = system.add_clip(clip_name, clip)
            names[obj.name] = handle.name
            if bind:
                system.set_action(obj, handle.name)
        if bind:
            # Parts (and the root) this motion leaves at rest: a previous generated clip
            # would keep playing on them (an in-place motion must not travel, a leg the
            # motion does not move must not keep walking); clips of the user's own stay.
            for obj in [R_obj, *(p.obj for p in ps.parts.values())]:
                if obj.id in clips:
                    continue
                current = system.action(obj)
                if current is None:
                    continue
                if _is_rigid_clip(current.data):
                    system.set_action(obj, None)
                    cleared.append(obj.name)
                else:
                    ps.diagnostics.append(
                        {
                            "code": "PART_ANIMATED",
                            "severity": "info",
                            "message": f"{obj.name!r} keeps its own action {current.name!r}",
                        }
                    )
    feet = [JOINTS[r] for r in ("leg.L", "leg.R") if r in ps.parts]
    contacts = None
    if feet and kind in ("walk", "run", "jump", "land"):
        rep = sa.foot_contacts(arm, feet, start=start, end=start + duration, rate=120.0)
        contacts = {
            f: {
                "sliding_max": round(d["sliding_max"], 6),
                "penetration": round(d["penetration"], 6),
                "contacts": len(d["contacts"]),
            }
            for f, d in rep["feet"].items()
        }
    report: dict[str, Any] = {
        "root": R_obj.name,
        "kind": kind,
        "clips": names,
        "bound": bool(bind),
        "parts": {r: p.obj.name for r, p in ps.parts.items()},
        "pivots": {r: p.source for r, p in ps.parts.items()},
        "channels": sum(len(c.channels) for c in clips.values()),
        "keys": keys,
        "samples": int(times.size),
        "max_error": fit,
        "duration": duration,
        "cyclic": cyclic,
        "travel": gen["travel"],
        "ik_residual": gen["ik_residual"],
        "diagnostics": ps.diagnostics,
        "armature": None,  # (no armature, no skin: object transforms only)
        "cleared": cleared,
    }
    for k in ("speed", "period", "stride", "height", "takeoff", "landing", "air_time"):
        if k in gen:
            report[k] = gen[k]
    if contacts is not None:
        report["feet"] = contacts
    if bind:
        report["verification"] = _verify_generated(system, ps, times, worlds, chain)
    return report


def _is_rigid_clip(clip: Any) -> bool:
    try:
        meta = json.loads(clip.metadata or "null")
    except (TypeError, ValueError):
        return False
    return isinstance(meta, dict) and bool((meta.get("generator") or {}).get("rigid"))


def _verify_generated(system, ps: PartSet, times, worlds, chain) -> dict[str, Any]:
    """The parts as the scene now evaluates them, relative to the root (whatever moves
    the root), against the generated motion."""
    idx = np.linspace(0, len(times) - 1, min(VERIFY_SAMPLES, len(times))).round().astype(int)
    objs = [p.obj for p in ps.parts.values()]
    got = _posed_worlds(system, times[idx], [o.id for o in objs] + [ps.root.id])
    root_got = np.linalg.inv(got[:, -1])
    root_exp = np.linalg.inv(worlds[ps.root.id][idx])
    expected = np.stack([root_exp @ worlds[o.id][idx] for o in objs], axis=1)
    actual = np.stack([root_got @ got[:, j] for j in range(len(objs))], axis=1)
    return _verify(expected, actual, [o.name for o in objs])


# --------------------------------------------------------------------------------------- #
# Conversions
# --------------------------------------------------------------------------------------- #


def _animated_range(system: AnimationSystem, objs: list[Any]) -> tuple[float, float] | None:
    lo, hi = math.inf, -math.inf
    for o in objs:
        action = system.action(o)
        if action is None:
            continue
        t0, t1 = action.data.frame_range()
        lo, hi = min(lo, t0), max(hi, t1)
    return (lo, hi) if hi > lo else None


def _copy(scene, obj: Any, suffix: str, parent: Any, skip: set[str]) -> dict[str, Any]:
    """Copy ``obj`` (and its descendants not in ``skip``) under ``parent``, each named
    ``"<name> <suffix>"``."""
    top = scene.duplicate(
        obj, scene._unique_name(f"{obj.name} {suffix}"), parent=parent, children=False
    )
    made = {obj.id: top}
    for child in obj.children:
        if child.id in skip:
            continue
        made.update(_copy(scene, child, suffix, top, skip))
    return made


def convert_character(
    system: AnimationSystem,
    source: Any,
    to: str,
    *,
    parts: dict[str, Any] | None = None,
    forward: str = "-y",
    name: str | None = None,
    hide_original: bool = True,
    start: float | None = None,
    end: float | None = None,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Convert a character between workflows (see the module docstring): ``to``
    ``"skeleton"`` (a rigid hierarchy -> an armature with attached copies) or ``"rigid"``
    (an armature's attached parts -> a hierarchy of pivots). The source stays as it was
    (hidden with ``hide_original``); motion is baked and verified. One undo step."""
    if to not in ("skeleton", "rigid"):
        raise _err("INVALID_ARGUMENT", "to is 'skeleton' or 'rigid'", "to")
    if not isinstance(hide_original, bool):
        raise _err("INVALID_ARGUMENT", "hide_original is true or false", "hide_original")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not (1e-4 <= tolerance <= 10)
    ):
        raise _err("INVALID_ARGUMENT", "tolerance is 0.0001..10 degrees", "tolerance")
    for key, value in (("start", start), ("end", end)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise _err("INVALID_ARGUMENT", f"{key} is a finite time", key)
    if name is not None and system.scene._by_name.get(name) is not None:
        raise _err("DUPLICATE_NAME", f"an object is already named {name!r}", "name")
    with system._op(f"Convert {system._object(source, 'source').name!r} to {to}"):
        if to == "skeleton":
            return _to_skeleton(
                system, source, parts, forward, name, hide_original, start, end, tolerance
            )
        return _to_rigid(system, source, name, hide_original, start, end, tolerance)


def _times(system, objs, start, end) -> np.ndarray | None:
    from .generators import RATE

    found = _animated_range(system, objs)
    if found is None and start is None:
        return None
    t0 = start if start is not None else found[0] if found else system.start
    t1 = end if end is not None else found[1] if found else system.end
    if not t1 > t0:
        raise _err("INVALID_ARGUMENT", "end must come after start", "end")
    if t1 - t0 > 600.0:
        raise _err("LIMIT_EXCEEDED", "conversions bake at most 600 s", "end")
    return t0 + np.linspace(0.0, t1 - t0, int(round((t1 - t0) * RATE)) + 1)


def _to_skeleton(system, source, parts, forward, name, hide, start, end, tolerance):
    from . import paths as P
    from .clip import Clip

    ps = find_parts(system, source, parts=parts, forward=forward)
    scene = system.scene
    R_obj = ps.root
    arm = scene.add_object(
        name or scene._unique_name(f"{R_obj.name} Rig"),
        parent=R_obj.parent,
        position=R_obj.position,
        rotation=R_obj.rotation,
        scale=R_obj.scale,
    )
    system.add_rig(arm, _joints(ps))
    rig = system.rig(arm)
    part_ids = {p.obj.id for p in ps.parts.values()}
    copies: dict[str, Any] = {}
    flattened = []
    for role, p in ps.parts.items():
        made = _copy(scene, p.obj, "(rig)", R_obj.parent, part_ids)
        top = made[p.obj.id]
        _set_local(top, np.linalg.inv(_parent_world(R_obj)) @ p.obj.world_matrix)
        system.attach(top, arm, JOINTS[role])
        copies.update(made)
    chain_ids = {o.id for o in _chain(ps)}
    for child in R_obj.children:  # (what hangs on the root moves with the armature)
        if child.id in part_ids:
            continue
        if child.id in chain_ids:
            flattened.append(child.name)
            continue
        copies.update(_copy(scene, child, "(rig)", arm, part_ids))
    tree = [R_obj, *_subtree(R_obj)]
    times = _times(system, tree, start, end)
    report: dict[str, Any] = {
        "to": "skeleton",
        "source": R_obj.name,
        "armature": arm.name,
        "joints": [j.name for j in rig.joints],
        "copies": {scene._objects[o].name: c.name for o, c in copies.items()},
        "parts": {r: p.obj.name for r, p in ps.parts.items()},
        "diagnostics": list(ps.diagnostics),
        "clip": None,
    }
    if flattened:
        report["diagnostics"].append(
            {
                "code": "STRUCTURE_FLATTENED",
                "severity": "info",
                "message": "objects between the root and the parts are "
                "not copied (the joints carry the parts): " + ", ".join(flattened[:8]),
            }
        )
    pairs = [(p.obj, copies[p.obj.id]) for p in ps.parts.values()]
    if times is not None:
        roles = list(ps.parts)
        ids = [R_obj.id] + [ps.parts[r].obj.id for r in roles]
        sampled = _posed_worlds(system, times, ids)
        parent_rest = _parent_world(R_obj)
        A0 = parent_rest @ R_obj.local_matrix
        rest_model = rig.rest_model()
        frames = {}  # joint world frames over time
        R_t = sampled[:, 0]
        frames["Hips"] = R_t @ np.linalg.inv(A0) @ (A0 @ rest_model[rig.index("Hips")])
        for k, role in enumerate(roles):
            o = ps.parts[role].obj
            j = JOINTS[role]
            frames[j] = (
                sampled[:, k + 1] @ np.linalg.inv(o.world_matrix) @ (A0 @ rest_model[rig.index(j)])
            )
        leg = max((p.length for r, p in ps.parts.items() if r.startswith("leg")), default=1.0)
        pos_tol = tolerance / 1000.0 * leg * 20.0
        channels: dict[str, Any] = {}
        arm_local = np.linalg.inv(_posed_parent(system, R_obj, times)) @ R_t
        ch, _, _ = _fit_local(times, arm_local, R_obj.local_matrix, tolerance, pos_tol, False)
        channels.update(ch)
        for k, jt in enumerate(rig.joints):
            # Joint transforms are local to the parent joint (the root joint: to the
            # armature, which moves as the source root did).
            above = R_t if jt.parent < 0 else frames[rig.joints[jt.parent].name]
            local = np.linalg.inv(above) @ frames[jt.name]
            rest = (
                rest_model[k]
                if jt.parent < 0
                else (np.linalg.inv(rest_model[jt.parent]) @ rest_model[k])
            )
            ch, _, _ = _fit_local(
                times, local, rest, tolerance, pos_tol, False, prefix=f'joints["{jt.name}"]'
            )
            channels.update(ch)
        if channels:
            clip_name = system._unique_clip_name(f"{arm.name} Converted")
            meta = json.dumps(
                {"converted": {"from": "rigid", "source": R_obj.name}},
                sort_keys=True,
                separators=(",", ":"),
            )
            system.add_clip(
                clip_name, Clip(P.OBJECT, tuple(sorted(channels.items())), (), None, meta)
            )
            system.set_action(arm, clip_name)
            report["clip"] = clip_name
        check = times[
            np.linspace(0, len(times) - 1, min(VERIFY_SAMPLES, len(times))).round().astype(int)
        ]
    else:
        check = np.array([system.time])
    got = _posed_worlds(system, check, [c.id for _, c in pairs] + [o.id for o, _ in pairs])
    n = len(pairs)
    report["verification"] = _verify(got[:, n:], got[:, :n], [o.name for o, _ in pairs])
    if hide:
        for o in tree:
            if o.visible:
                o.visible = False
        report["hidden"] = [o.name for o in tree]
    return report


def _to_rigid(system, source, name, hide, start, end, tolerance):
    from . import paths as P
    from .clip import Clip

    scene = system.scene
    arm = system._object(source, "source")
    rig = system.rig(arm)
    if rig is None:
        raise _err(
            "RIG_MISSING",
            f"{arm.name!r} has no rig",
            "source",
            hint="rigid hierarchies convert with to='skeleton'",
        )
    attached: dict[int, list[Any]] = {}
    for oid, jid in rig.attachments:
        if oid in scene._objects:
            attached.setdefault(rig.index_of_id(jid), []).append(scene._objects[oid])
    if not attached:
        raise _err(
            "NOTHING_ATTACHED",
            f"no object is attached to {arm.name!r}'s joints",
            "source",
            hint="skinned meshes deform: they have no rigid equivalent",
        )
    diagnostics = []
    for (kind, oid), item in system._owned.items():
        if kind == "skin" and item._definition.rig == system._owned[("rig", arm._id)].id:
            diagnostics.append(
                {
                    "code": "SKINNED_MESH_SKIPPED",
                    "severity": "warning",
                    "message": f"{scene._objects[oid].name!r} is skinned: a rigid "
                    "hierarchy cannot deform it (not converted)",
                }
            )
    root = scene.add_object(
        name or scene._unique_name(f"{arm.name} Parts"),
        parent=arm.parent,
        position=arm.position,
        rotation=arm.rotation,
        scale=arm.scale,
    )
    A0 = arm.world_matrix
    model = rig.rest_model()
    pivots: dict[int, Any] = {}
    rest_world: dict[int, np.ndarray] = {}
    for k in sorted(attached):
        p = rig.joints[k].parent
        while p >= 0 and p not in attached:
            p = rig.joints[p].parent
        parent = pivots[p] if p >= 0 else root
        pw = rest_world[p] if p >= 0 else root.world_matrix
        pivot = scene.add_object(
            scene._unique_name(f"{root.name} {rig.joints[k].name}"), parent=parent
        )
        rest_world[k] = A0 @ model[k]
        _set_local(pivot, np.linalg.inv(pw) @ rest_world[k])
        pivots[k] = pivot
    copies = {}
    for k, objs in attached.items():
        for o in objs:
            made = _copy(scene, o, "(parts)", pivots[k], set())
            _set_local(made[o.id], np.linalg.inv(rest_world[k]) @ o.world_matrix)
            copies.update(made)
    originals = [o for objs in attached.values() for o in objs]
    times = _times(system, [arm], start, end)
    report: dict[str, Any] = {
        "to": "rigid",
        "source": arm.name,
        "root": root.name,
        "pivots": {rig.joints[k].name: p.name for k, p in pivots.items()},
        "copies": {scene._objects[o].name: c.name for o, c in copies.items()},
        "diagnostics": diagnostics,
        "clips": {},
    }
    if times is not None:
        ids = [arm.id] + [attached[k][0].id for k in sorted(attached)]
        sampled = _posed_worlds(system, times, ids)
        frames = {}
        for i, k in enumerate(sorted(attached)):
            o = attached[k][0]
            frames[k] = sampled[:, i + 1] @ np.linalg.inv(o.world_matrix) @ rest_world[k]
        A_t = sampled[:, 0]
        parent_t = _posed_parent(system, arm, times)
        pos_tol = tolerance / 1000.0 * float(np.ptp(model[:, 2, 3]) or 1.0) * 10.0
        locals_ = {root.id: (np.linalg.inv(parent_t) @ A_t, root.local_matrix)}
        for k, pivot in pivots.items():
            p = rig.joints[k].parent
            while p >= 0 and p not in attached:
                p = rig.joints[p].parent
            pw = frames[p] if p >= 0 else A_t
            locals_[pivot.id] = (np.linalg.inv(pw) @ frames[k], pivot.local_matrix)
        for oid, (local, rest) in locals_.items():
            channels, _, _ = _fit_local(times, local, rest, tolerance, pos_tol, False)
            if not channels:
                continue
            obj = scene._objects[oid]
            clip_name = system._unique_clip_name(f"{obj.name} Converted")
            meta = json.dumps(
                {"converted": {"from": "skeleton", "source": arm.name}},
                sort_keys=True,
                separators=(",", ":"),
            )
            system.add_clip(
                clip_name, Clip(P.OBJECT, tuple(sorted(channels.items())), (), None, meta)
            )
            system.set_action(obj, clip_name)
            report["clips"][obj.name] = clip_name
        check = times[
            np.linspace(0, len(times) - 1, min(VERIFY_SAMPLES, len(times))).round().astype(int)
        ]
    else:
        check = np.array([system.time])
    got = _posed_worlds(
        system, check, [copies[o.id].id for o in originals] + [o.id for o in originals]
    )
    n = len(originals)
    report["verification"] = _verify(got[:, n:], got[:, :n], [o.name for o in originals])
    if hide:
        for o in originals:
            o.visible = False
        report["hidden"] = [o.name for o in originals]
    return report
