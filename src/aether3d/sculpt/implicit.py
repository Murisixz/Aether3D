"""Implicit surfaces: organic meshes from smoothly blended shapes.

:func:`implicit_surface` evaluates the signed distance of a set of **shapes** -- spheres,
ellipsoids, tapered capsules ("round cones") and rounded boxes -- combined by a smooth
union (each shape's ``blend`` radius rounds the crease where it meets the others), on a
grid of spacing ``voxel_size``, and extracts the zero level set with the voxel
remesher's marching cubes (closed, manifold, consistently oriented). Normals are the
distance field's gradient (central differences), so the surface shades smoothly.

This is a modeling tool for base meshes -- bodies, creatures, rounded props -- built
from a few numbers; :func:`humanoid_shapes` and :func:`quadruped_shapes` are recipes for
unrigged characters (Z up, facing -Y, feet on ``z = 0``) with adjustable proportions and
rest poses, used by Aether3D's rigging tests and showcases.

Distances: round cones are exact (the closed form of two spheres' convex hull), spheres
and rounded boxes exact, ellipsoids a bound (``k0 (k0 - 1) / k1``); the smooth union is
the polynomial smooth minimum. The surface is therefore accurate where shapes do not
blend, and smooth where they do.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import ValidationError

KINDS = ("sphere", "ellipsoid", "capsule", "box")
MAX_SHAPES = 512
MAX_POINTS = 60_000_000


def _vec(value: Any, n: int, what: str) -> np.ndarray:
    try:
        v = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValidationError(f"{what} is {n} numbers") from None
    if v.shape != (n,) or not np.all(np.isfinite(v)):
        raise ValidationError(f"{what} is {n} finite numbers")
    return v


def _pos(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValidationError(f"{what} must be a positive number")
    if not math.isfinite(float(value)):
        raise ValidationError(f"{what} must be finite")
    return float(value)


def _check(shape: Any, i: int, blend: float) -> dict[str, Any]:
    if not isinstance(shape, dict) or shape.get("type") not in KINDS:
        raise ValidationError(f"shapes[{i}] is {{type: {' | '.join(KINDS)}, ...}}")
    kind = shape["type"]
    out: dict[str, Any] = {"type": kind}
    b = shape.get("blend", blend)
    if isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 <= b < 10:
        raise ValidationError(f"shapes[{i}].blend is 0..10")
    out["blend"] = float(b)
    allowed = {"type", "blend", "subtract"}
    if kind == "sphere":
        allowed |= {"center", "radius"}
        out["center"] = _vec(shape.get("center"), 3, f"shapes[{i}].center")
        out["radius"] = _pos(shape.get("radius"), f"shapes[{i}].radius")
    elif kind == "ellipsoid":
        allowed |= {"center", "radii", "rotation"}
        out["center"] = _vec(shape.get("center"), 3, f"shapes[{i}].center")
        out["radii"] = _vec(shape.get("radii"), 3, f"shapes[{i}].radii")
        if np.any(out["radii"] <= 0):
            raise ValidationError(f"shapes[{i}].radii must be positive")
        q = _vec(shape.get("rotation", (0, 0, 0, 1)), 4, f"shapes[{i}].rotation")
        out["rotation"] = q / np.linalg.norm(q)
    elif kind == "capsule":
        allowed |= {"a", "b", "radius", "radius_b"}
        out["a"] = _vec(shape.get("a"), 3, f"shapes[{i}].a")
        out["b"] = _vec(shape.get("b"), 3, f"shapes[{i}].b")
        out["r0"] = _pos(shape.get("radius"), f"shapes[{i}].radius")
        out["r1"] = _pos(shape.get("radius_b", out["r0"]), f"shapes[{i}].radius_b")
    else:
        allowed |= {"center", "size", "radius", "rotation"}
        out["center"] = _vec(shape.get("center"), 3, f"shapes[{i}].center")
        out["size"] = _vec(shape.get("size"), 3, f"shapes[{i}].size")
        if np.any(out["size"] <= 0):
            raise ValidationError(f"shapes[{i}].size must be positive")
        r = shape.get("radius", 0.0)
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not r >= 0:
            raise ValidationError(f"shapes[{i}].radius must be >= 0")
        out["radius"] = float(r)
        q = _vec(shape.get("rotation", (0, 0, 0, 1)), 4, f"shapes[{i}].rotation")
        out["rotation"] = q / np.linalg.norm(q)
    if set(shape) - allowed:
        raise ValidationError(f"shapes[{i}]: unknown fields {sorted(set(shape) - allowed)}")
    sub = shape.get("subtract", False)
    if not isinstance(sub, bool):
        raise ValidationError(f"shapes[{i}].subtract is true or false")
    out["subtract"] = sub
    return out


def _rot_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _distance(s: dict[str, Any], p: np.ndarray) -> np.ndarray:
    kind = s["type"]
    if kind == "sphere":
        return np.linalg.norm(p - s["center"], axis=1) - s["radius"]
    if kind == "ellipsoid":
        local = (p - s["center"]) @ _rot_matrix(s["rotation"])
        r = s["radii"]
        k0 = np.linalg.norm(local / r, axis=1)
        k1 = np.linalg.norm(local / (r * r), axis=1)
        return np.where(k1 > 0, k0 * (k0 - 1.0) / np.where(k1 > 0, k1, 1.0), -float(r.min()))
    if kind == "capsule":
        return _round_cone(p, s["a"], s["b"], s["r0"], s["r1"])
    local = (p - s["center"]) @ _rot_matrix(s["rotation"])
    q = np.abs(local) - (s["size"] / 2.0 - s["radius"])
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside - s["radius"]


def _round_cone(p: np.ndarray, a: np.ndarray, b: np.ndarray, r0: float, r1: float):
    """Exact distance to the convex hull of the spheres ``(a, r0)`` and ``(b, r1)``."""
    ba = b - a
    l2 = float(ba @ ba)
    if l2 < 1e-24:
        return np.linalg.norm(p - a, axis=1) - max(r0, r1)
    rr = r0 - r1
    a2 = l2 - rr * rr
    if a2 <= 0:  # one sphere holds the other
        big = (a, r0) if r0 >= r1 else (b, r1)
        return np.linalg.norm(p - big[0], axis=1) - big[1]
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    perp = pa * l2 - y[:, None] * ba[None]
    x2 = np.sum(perp * perp, axis=1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2 if rr != 0 else np.zeros_like(x2)
    end_b = np.sign(z) * a2 * z2 > k
    end_a = np.sign(y) * a2 * y2 < k
    side = (np.sqrt(np.maximum(x2 * a2 * il2, 0.0)) + y * rr) * il2 - r0
    return np.where(
        end_b,
        np.sqrt(x2 + z2) * il2 - r1,
        np.where(end_a, np.sqrt(x2 + y2) * il2 - r0, side),
    )


def _combine(d: np.ndarray, ds: np.ndarray, s: dict[str, Any], first: bool) -> np.ndarray:
    """``d`` joined with a shape's distances ``ds``: a smooth union (or difference)."""
    if first:
        return ds if not s["subtract"] else -ds
    k = s["blend"]
    if s["subtract"]:
        if k > 0:
            h = np.maximum(k - np.abs(-ds - d), 0.0) / k
            return np.maximum(d, -ds) + h * h * k / 4.0
        return np.maximum(d, -ds)
    if k > 0:
        with np.errstate(invalid="ignore"):
            h = np.maximum(k - np.abs(ds - d), 0.0) / k
        h = np.where(np.isfinite(h), h, 0.0)
        return np.minimum(d, ds) - h * h * k / 4.0
    return np.minimum(d, ds)


def _field(shapes: list[dict[str, Any]], p: np.ndarray) -> np.ndarray:
    d = np.full(p.shape[0], np.inf)
    for i, s in enumerate(shapes):
        d = _combine(d, _distance(s, p), s, i == 0)
    return d


def _bounds(shapes: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for s in shapes:
        if s["subtract"]:
            continue
        g = s["blend"]  # (a smooth union can bulge by up to a quarter of the blend radius)
        if s["type"] == "sphere":
            pts, r = [s["center"]], s["radius"]
        elif s["type"] == "ellipsoid":
            pts, r = [s["center"]], float(s["radii"].max())
        elif s["type"] == "capsule":
            pts, r = [s["a"], s["b"]], max(s["r0"], s["r1"])
        else:
            pts, r = [s["center"]], float(np.linalg.norm(s["size"])) / 2.0
        for e in pts:
            lo, hi = np.minimum(lo, e - r - g), np.maximum(hi, e + r + g)
    return lo, hi


def implicit_surface(
    shapes: list[Any], voxel_size: float, *, blend: float = 0.0, max_points: int = MAX_POINTS
) -> Any:
    """A closed mesh of the smooth union of ``shapes`` (see the module docstring), each
    ``{type: sphere | ellipsoid | capsule | box, ..., blend, subtract}``; ``blend`` is the
    default blend radius."""
    from ..mesh import Mesh
    from .voxel import marching_cubes

    if not isinstance(shapes, list) or not 1 <= len(shapes) <= MAX_SHAPES:
        raise ValidationError(f"shapes is a list of 1..{MAX_SHAPES}")
    h = _pos(voxel_size, "voxel_size")
    if isinstance(blend, bool) or not isinstance(blend, (int, float)) or not 0 <= blend < 10:
        raise ValidationError("blend is 0..10")
    checked = [_check(s, i, float(blend)) for i, s in enumerate(shapes)]
    if all(s["subtract"] for s in checked):
        raise ValidationError("at least one shape adds material")
    lo, hi = _bounds(checked)
    lo, hi = lo - 2 * h, hi + 2 * h
    dims = np.ceil((hi - lo) / h).astype(np.int64) + 1
    total = int(np.prod(dims))
    if total > max_points:
        raise ValidationError(f"{total} grid points exceed {max_points}: use a larger voxel_size")
    nx, ny, nz = (int(d) for d in dims)
    # Each shape changes the field only near itself (within its blend radius): it is
    # evaluated on its own block of the grid and combined there.
    field = np.full((nx, ny, nz), np.inf)
    first = True
    for shape in checked:
        blo, bhi = _bounds([{**shape, "subtract": False}])
        margin = shape["blend"] + 3 * h
        i0 = np.clip(np.floor((blo - margin - lo) / h).astype(np.int64), 0, dims - 1)
        i1 = np.clip(np.ceil((bhi + margin - lo) / h).astype(np.int64) + 1, 1, dims)
        if shape["subtract"] or first:  # (a subtraction or the first shape: everywhere)
            i0, i1 = np.zeros(3, np.int64), dims.copy()
        gx = lo[0] + h * np.arange(i0[0], i1[0])
        gy = lo[1] + h * np.arange(i0[1], i1[1])
        gz = lo[2] + h * np.arange(i0[2], i1[2])
        pts = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
        block = field[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]].reshape(-1)
        merged = _combine(block, _distance(shape, pts), shape, first)
        field[i0[0] : i1[0], i0[1] : i1[1], i0[2] : i1[2]] = merged.reshape(
            int(i1[0] - i0[0]), int(i1[1] - i0[1]), int(i1[2] - i0[2])
        )
        first = False
    field = np.where(np.isfinite(field), field, 4.0 * h)
    # The band the surface crosses: the points near it, and every corner of every cell
    # whose corners change sign -- the blended field is not an exact distance (a thin
    # finger's gradient can exceed 1), and a crossing cell with a corner outside the band
    # would be skipped, leaving a hole.
    band = np.abs(field) < 2.5 * h
    neg = field < 0
    nx1, ny1, nz1 = nx - 1, ny - 1, nz - 1
    corner = [(i & 1, (i >> 1) & 1, (i >> 2) & 1) for i in range(8)]
    first_corner = neg[:nx1, :ny1, :nz1]
    mixed = np.zeros((nx1, ny1, nz1), bool)
    for dx, dy, dz in corner[1:]:
        mixed |= neg[dx : nx1 + dx, dy : ny1 + dy, dz : nz1 + dz] != first_corner
    for dx, dy, dz in corner:
        band[dx : nx1 + dx, dy : ny1 + dy, dz : nz1 + dz] |= mixed
    values = field.reshape(-1, order="F")
    keys = np.flatnonzero(band.reshape(-1, order="F"))
    positions, triangles = marching_cubes(keys, values[keys], dims, lo, h)
    if len(triangles) == 0:
        raise ValidationError("the shapes enclose no volume at this voxel size")
    eps = 0.25 * h
    grad = np.zeros_like(positions)
    for axis in range(3):
        off = np.zeros(3)
        off[axis] = eps
        grad[:, axis] = _field(checked, positions + off) - _field(checked, positions - off)
    nrm = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
    return Mesh(
        positions.astype(np.float32), triangles.astype(np.uint32), normals=nrm.astype(np.float32)
    )


# --------------------------------------------------------------------------------------- #
# Character recipes
# --------------------------------------------------------------------------------------- #

HUMANOID_DEFAULTS: dict[str, float] = {
    "height": 1.75,
    "head": 1.0,  # head size factor (stylized characters: 1.4 .. 2)
    "limbs": 1.0,  # limb thickness factor
    "shoulders": 1.0,  # shoulder width factor
    "hips": 1.0,  # hip width factor
    "legs": 1.0,  # leg length factor (relative; the height stays)
    "arms": 1.0,  # arm length factor
    "belly": 1.0,  # belly size factor
    "arm_angle": 30.0,  # degrees from hanging straight down (0) to a T-pose (90)
    "elbow_bend": 8.0,  # degrees the forearm bends forward at rest
    "knee_bend": 3.0,  # degrees the shin bends back at rest
    "stance": 0.0,  # extra distance between the feet (m)
}


def humanoid_shapes(**options: Any) -> list[dict[str, Any]]:
    """Shapes of an unrigged humanoid (facing -Y, Z up, feet on ``z = 0``): torso, neck,
    head with a nose, arms with hands and thumbs, legs with feet and toes. Options (see
    :data:`HUMANOID_DEFAULTS`) change proportions and the rest pose (A-pose by default;
    ``arm_angle=90`` is a T-pose)."""
    o = dict(HUMANOID_DEFAULTS)
    unknown = set(options) - set(o)
    if unknown:
        raise ValidationError(f"unknown humanoid options {sorted(unknown)}")
    for k, v in options.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValidationError(f"{k} is a number")
        o[k] = float(v)
    H = o["height"]
    s = H / 1.75  # (sizes below are for a 1.75 m adult)
    limb = o["limbs"]
    # Heights as fractions of the stature (Drillis and Contini's segment proportions);
    # the leg factor moves the hips, the head factor sizes the head, the torso fills in.
    ankle_z = 0.039 * H
    hip_z = ankle_z + (0.51 * H - ankle_z) * o["legs"]
    knee_z = ankle_z + 0.49 * (hip_z - ankle_z)
    head_rz = 0.066 * H * o["head"]
    head_c = H - head_rz
    chin = head_c - 0.85 * head_rz
    neck_base = chin - 0.045 * H
    shoulder_z = neck_base - 0.012 * H
    pelvis_z = hip_z + 0.035 * H
    hw = 0.052 * H * o["hips"]  # hip joint half spacing
    sw = 0.105 * H * o["shoulders"]  # shoulder joint half spacing
    shapes: list[dict[str, Any]] = []

    def capsule(a, b, r0, r1=None, blend=0.03 * s):
        rb = r0 if r1 is None else r1
        shapes.append(
            {
                "type": "capsule",
                "a": list(map(float, a)),
                "b": list(map(float, b)),
                "radius": r0,
                "radius_b": rb,
                "blend": blend,
            }
        )

    def ell(c, r, blend=0.04 * s, rot=None):
        d = {
            "type": "ellipsoid",
            "center": list(map(float, c)),
            "radii": list(map(float, r)),
            "blend": blend,
        }
        if rot is not None:
            d["rotation"] = list(rot)
        shapes.append(d)

    # Torso: a backbone capsule (the waist), pelvis, belly, chest and shoulder girdle.
    chest_c = shoulder_z - 0.085 * H
    capsule(
        [0, 0.0, pelvis_z],
        [0, 0.0, neck_base - 0.02 * H],
        0.075 * H * o["belly"] ** 0.5,
        0.07 * H,
        blend=0.06 * s,
    )
    ell([0, 0.0, pelvis_z], [0.1 * H * o["hips"], 0.065 * H, 0.07 * H], 0.06 * s)
    belly_c = 0.55 * pelvis_z + 0.45 * chest_c
    ell(
        [0, -0.006 * H * o["belly"], belly_c],
        [0.085 * H * max(o["belly"], 0.6), 0.06 * H * o["belly"], 0.09 * H],
        0.07 * s,
    )
    ell([0, 0.0, chest_c], [0.1 * H * o["shoulders"], 0.065 * H, 0.1 * H], 0.07 * s)
    capsule(
        [-sw * 0.9, 0.0, shoulder_z - 0.006 * H],
        [sw * 0.9, 0.0, shoulder_z - 0.006 * H],
        0.032 * H * limb,
        blend=0.06 * s,
    )
    # Neck and head (a nose and a jaw make the face readable and the facing measurable).
    capsule(
        [0, 0.004 * H, neck_base - 0.02 * H],
        [0, 0.0, chin + 0.02 * H],
        0.03 * H * limb,
        0.027 * H * limb,
        blend=0.04 * s,
    )
    head_rx, head_ry = 0.8 * head_rz, 0.9 * head_rz
    ell([0, 0.0, head_c], [head_rx, head_ry, head_rz], 0.03 * s)
    ell(
        [0, -head_ry * 0.35, head_c - head_rz * 0.5],
        [head_rx * 0.7, head_ry * 0.6, head_rz * 0.5],
        0.03 * s,
    )
    # (a soft ridge rather than a blob: a blob's underside faces away from any light from
    # above and reads as a dark pit)
    ell(
        [0, -head_ry * 0.98 + head_rz * 0.01, head_c - head_rz * 0.04],
        [head_rz * 0.08, head_rz * 0.12, head_rz * 0.32],
        0.025 * s,
    )
    # Arms: shoulder -> elbow -> wrist -> hand, out from hanging by arm_angle.
    upper = 0.186 * H * o["arms"]
    fore = 0.146 * H * o["arms"]
    hand = 0.108 * H * o["arms"]
    ang = math.radians(o["arm_angle"])
    bend = math.radians(o["elbow_bend"])
    fwd = np.array([0.0, -1.0, 0.0])
    for side in (1.0, -1.0):
        sh = np.array([side * sw, 0.0, shoulder_z - 0.01 * H])
        down = np.array([side * math.sin(ang), 0.0, -math.cos(ang)])
        elbow = sh + upper * down
        d2 = down * math.cos(bend) + fwd * math.sin(bend)
        wrist = elbow + fore * d2
        capsule(sh, elbow, 0.03 * H * limb, 0.024 * H * limb, blend=0.035 * s)
        capsule(elbow, wrist, 0.0235 * H * limb, 0.017 * H * limb, blend=0.025 * s)
        lat = np.cross(d2, fwd)
        lat = lat / max(np.linalg.norm(lat), 1e-9)
        palm = wrist + d2 * hand * 0.32
        ell(palm, [0.026 * H * limb, 0.012 * H * limb, 0.034 * H], 0.02 * s, _basis_quat(lat, d2))
        capsule(
            palm + d2 * 0.02 * H,
            wrist + d2 * hand,
            0.017 * H * limb,
            0.012 * H * limb,
            blend=0.02 * s,
        )
        thumb0 = wrist + d2 * 0.022 * H + fwd * 0.014 * H
        capsule(
            thumb0,
            thumb0 + d2 * 0.028 * H + fwd * 0.017 * H,
            0.009 * H * limb,
            0.0075 * H * limb,
            blend=0.012 * s,
        )
    # Legs: hip -> knee -> ankle -> foot and toes (feet point forward, -Y).
    kb = math.radians(o["knee_bend"])
    for side in (1.0, -1.0):
        x_foot = side * (hw + o["stance"] / 2)
        hip = np.array([side * hw, 0.0, hip_z])
        knee = np.array([0.5 * (hip[0] + x_foot), -math.sin(kb) * 0.1 * H, knee_z])
        ankle = np.array([x_foot, 0.0, ankle_z])
        capsule(hip + [0, 0, 0.03 * H], knee, 0.05 * H * limb, 0.032 * H * limb, blend=0.05 * s)
        capsule(knee, ankle, 0.031 * H * limb, 0.02 * H * limb, blend=0.03 * s)
        heel = ankle + [0, 0.02 * H, -ankle_z + 0.02 * H]
        ball = ankle + [0, -0.075 * H, -ankle_z + 0.017 * H]
        capsule(heel, ball, 0.02 * H * limb, 0.0195 * H * limb, blend=0.03 * s)
        capsule(
            ball,
            ball + [0, -0.035 * H, -0.003 * H],
            0.017 * H * limb,
            0.014 * H * limb,
            blend=0.015 * s,
        )
    return shapes


def _basis_quat(x_axis: np.ndarray, z_axis: np.ndarray) -> list[float]:
    z = z_axis / np.linalg.norm(z_axis)
    x = x_axis - (x_axis @ z) * z
    x = x / max(np.linalg.norm(x), 1e-12)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    from ..animation import rotation as R

    return R.from_matrices(m[None])[0].tolist()


def quadruped_shapes(*, length: float = 1.2, height: float = 0.8, tail: bool = True):
    """Shapes of an unrigged four-legged creature (facing -Y, Z up, feet on ``z = 0``)."""
    L = _pos(length, "length")
    Hh = _pos(height, "height")
    s = L / 1.2
    body_z = Hh * 0.68
    shapes: list[dict[str, Any]] = [
        {
            "type": "ellipsoid",
            "center": [0, 0, body_z],
            "radii": [0.17 * s, 0.42 * s, 0.17 * s],
            "blend": 0.06 * s,
        },
        {
            "type": "capsule",
            "a": [0, -0.36 * s, body_z + 0.05 * s],
            "b": [0, -0.55 * s, body_z + 0.25 * s],
            "radius": 0.08 * s,
            "radius_b": 0.065 * s,
            "blend": 0.05 * s,
        },
        {
            "type": "ellipsoid",
            "center": [0, -0.64 * s, body_z + 0.3 * s],
            "radii": [0.085 * s, 0.13 * s, 0.085 * s],
            "blend": 0.04 * s,
        },
    ]
    for y in (-0.3 * s, 0.3 * s):
        for side in (1.0, -1.0):
            top = np.array([side * 0.11 * s, y, body_z - 0.05 * s])
            knee = top + [0, 0.03 * s * (1 if y < 0 else -1), -(body_z - 0.05 * s) * 0.5]
            foot = np.array([side * 0.11 * s, y, 0.035 * s])
            shapes += [
                {
                    "type": "capsule",
                    "a": list(top),
                    "b": list(knee),
                    "radius": 0.06 * s,
                    "radius_b": 0.042 * s,
                    "blend": 0.05 * s,
                },
                {
                    "type": "capsule",
                    "a": list(knee),
                    "b": list(foot),
                    "radius": 0.04 * s,
                    "radius_b": 0.032 * s,
                    "blend": 0.03 * s,
                },
                {
                    "type": "ellipsoid",
                    "center": list(foot + [0, -0.03 * s, -0.005 * s]),
                    "radii": [0.04 * s, 0.06 * s, 0.03 * s],
                    "blend": 0.02 * s,
                },
            ]
    if tail:
        shapes.append(
            {
                "type": "capsule",
                "a": [0, 0.38 * s, body_z + 0.04 * s],
                "b": [0, 0.75 * s, body_z + 0.18 * s],
                "radius": 0.04 * s,
                "radius_b": 0.015 * s,
                "blend": 0.03 * s,
            }
        )
    return shapes
