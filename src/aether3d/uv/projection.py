"""Projection mappings: planar, box, cylindrical and spherical UVs.

Coordinates are taken in the mesh's own space (object space). ``axis`` is the projection
axis (``"x"``, ``"y"``, ``"z"`` or a 3-vector); a right-handed basis ``(u_dir, v_dir,
axis)`` is built with ``v_dir`` along the world +Z projected on the plane (or +Y when the
axis is vertical), so images are upright.

* **planar** -- ``(u, v) = (p . u_dir, p . v_dir)``: one island per connected piece.
* **box** -- every triangle uses the plane of the dominant axis of its normal (+X, -X, +Y,
  -Y, +Z, -Z), each seen from outside so nothing is mirrored; faces of one direction that
  share edges form an island.
* **cylindrical** -- ``u = angle around the axis / 2 pi``, ``v = height along the axis /
  (2 pi radius_scale)``... in world units: ``u`` is the arc length at the mean radius,
  ``v`` the height, so a cylinder's side keeps its proportions. A triangle crossing the
  seam (the ``-u_dir`` side) is unwrapped continuously (its corners on the far side move
  by one turn), so the seam is a cut, not a wrap-around.
* **spherical** -- ``u`` the longitude arc at the mean radius, ``v`` the latitude arc;
  the same seam rule; triangles touching a pole get the pole's longitude from their other
  corners (a pole is a point: its UVs are reported as singular).

Results are in world units and packed like :func:`~aether3d.uv.unwrap` unless
``pack=False`` (then fitted into the unit square, keeping proportions).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import TexturingError
from ..mesh import Mesh
from . import pack as packing
from .topology import Topology, dense_labels
from .unwrap import UnwrapResult, _area, _check_resolution, _rebuild

KINDS = ("planar", "box", "cylindrical", "spherical")
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def _axis(axis: Any) -> np.ndarray:
    if isinstance(axis, str):
        if axis not in AXES:
            raise TexturingError(
                "INVALID_ARGUMENT", "axis must be 'x', 'y', 'z' or a 3-vector", "axis"
            )
        return np.array(AXES[axis])
    try:
        a = np.asarray(axis, dtype=np.float64).reshape(3)
    except (TypeError, ValueError):
        raise TexturingError(
            "INVALID_ARGUMENT", "axis must be 'x', 'y', 'z' or a 3-vector", "axis"
        ) from None
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n == 0:
        raise TexturingError("INVALID_ARGUMENT", "axis must be a finite non-zero vector", "axis")
    return a / n


def basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(u_dir, v_dir)`` completing ``axis`` to a right-handed frame, v as upright as
    possible (world +Z, or +Y for a vertical axis)."""
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    v = up - axis * float(axis @ up)
    v /= np.linalg.norm(v)
    u = np.cross(v, axis)
    return u, v


def project(
    mesh: Mesh,
    kind: str,
    *,
    axis: Any = "z",
    center: Any = None,
    pack: bool = True,
    resolution: Any = 1024,
    padding: int = 4,
    rotate: bool = False,
    texels_per_unit: float | None = None,
) -> UnwrapResult:
    """UVs by projection (see the module docstring); returns a new mesh like ``unwrap``."""
    if not isinstance(mesh, Mesh):
        raise TexturingError("INVALID_ARGUMENT", "project needs a Mesh", "mesh")
    if kind not in KINDS:
        raise TexturingError("INVALID_ARGUMENT", f"kind must be one of {', '.join(KINDS)}", "kind")
    if mesh.triangle_count == 0:
        raise TexturingError("EMPTY_MESH", "the mesh has no triangles", "mesh")
    res = _check_resolution(resolution)
    if (
        isinstance(padding, bool)
        or not isinstance(padding, (int, np.integer))
        or not 0 <= padding <= 256
    ):
        raise TexturingError(
            "INVALID_ARGUMENT", "padding must be an integer in [0, 256]", "padding"
        )
    a = _axis(axis)
    topo = Topology.from_mesh(mesh)
    p = topo.positions[topo.triangles]  # (T, 3, 3)
    if center is None:
        lo, hi = topo.positions.min(axis=0), topo.positions.max(axis=0)
        c = 0.5 * (lo + hi)
    else:
        try:
            c = np.asarray(center, dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            raise TexturingError(
                "INVALID_ARGUMENT", "center must be a 3-vector", "center"
            ) from None
        if not np.all(np.isfinite(c)):
            raise TexturingError("INVALID_ARGUMENT", "center must be finite", "center")
    diagnostics: list[Diagnostic] = []
    u_dir, v_dir = basis(a)
    rel = p - c
    T = topo.triangle_count
    if kind == "planar":
        uv = np.stack([rel @ u_dir, rel @ v_dir], axis=-1)
        group = np.zeros(T, dtype=np.int64)
    elif kind == "box":
        uv, group = _box(rel, topo.face_normal)
    else:
        uv, group, singular = _wrapped(rel, a, u_dir, v_dir, kind)
        if singular:
            diagnostics.append(
                Diagnostic(
                    "UV_SINGULARITY",
                    WARNING,
                    f"{singular} corner(s) lie on the projection axis (a pole): their u is "
                    "taken from the rest of their triangle",
                    None,
                )
            )
    # Islands: faces of one projection group connected through edges whose two sides got
    # the same UVs.
    corner_id, island = _islands(topo, uv, group)
    count = int(island.max()) + 1
    if pack:
        groups = [np.flatnonzero(island == i) for i in range(count)]
        placed, result = packing.pack(
            [uv[g] for g in groups],
            res,
            padding=int(padding),
            rotate=rotate,
            texels_per_unit=texels_per_unit,
        )
        for i, g in enumerate(groups):
            uv[g] = placed[i]
        pack_report = {
            "resolution": list(res),
            "padding": int(padding),
            "texels_per_unit": result.texels_per_unit,
            "utilization": float(sum(_area(t) for t in placed)),
        }
    else:
        lo = uv.reshape(-1, 2).min(axis=0)
        span = float((uv.reshape(-1, 2).max(axis=0) - lo).max())
        uv = (uv - lo) / (span if span > 0 else 1.0)
        pack_report = None
        diagnostics.append(
            Diagnostic(
                "UVS_NOT_PACKED", INFO, "islands were fitted into the unit square together", None
            )
        )
    new = _rebuild(mesh, topo, corner_id, uv)
    report = {
        "kind": kind,
        "axis": a.tolist(),
        "center": c.tolist(),
        "islands": count,
        "pack": pack_report,
    }
    return UnwrapResult(new, island, report, diagnostics)


def _box(rel: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal
    dom = np.argmax(np.abs(n), axis=1)
    sign = np.where(n[np.arange(n.shape[0]), dom] < 0, 1, 0)
    group = dom * 2 + sign
    uv = np.zeros(rel.shape[:2] + (2,))
    for g in range(6):
        sel = group == g
        if not sel.any():
            continue
        axis = np.zeros(3)
        axis[g // 2] = -1.0 if g % 2 else 1.0
        u_dir, v_dir = basis(axis)
        uv[sel, :, 0] = rel[sel] @ u_dir
        uv[sel, :, 1] = rel[sel] @ v_dir
    return uv, group


def _wrapped(rel, axis, u_dir, v_dir, kind):
    x = rel @ u_dir
    y = rel @ v_dir
    h = rel @ axis
    ang = np.arctan2(y, x)  # (-pi, pi], seam at -u_dir
    radius_xy = np.hypot(x, y)
    singular = 0
    if kind == "cylindrical":
        scale = float(radius_xy.mean()) if radius_xy.size else 1.0
        v = h
        pole = radius_xy <= 1e-12 * max(scale, 1e-300)
    else:
        r3 = np.sqrt(radius_xy**2 + h**2)
        scale = float(r3.mean()) if r3.size else 1.0
        v = np.arctan2(h, radius_xy) * scale
        pole = radius_xy <= 1e-12 * max(scale, 1e-300)
    singular = int(pole.sum())
    # Continuity across the seam: every corner within half a turn of the triangle's first
    # corner off the axis; corners on the axis take their triangle's mean longitude.
    ang = np.where(pole, np.nan, ang)
    first = np.argmax(~pole, axis=1)
    ref = ang[np.arange(ang.shape[0]), first][:, None]
    ang = np.where(ang - ref > np.pi, ang - 2 * np.pi, ang)
    ang = np.where(ang - ref < -np.pi, ang + 2 * np.pi, ang)
    if singular:
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(np.where(pole, np.nan, ang), axis=1)
        ang = np.where(pole, np.where(np.isfinite(mean), mean, 0.0)[:, None], ang)
    u = ang * scale
    uv = np.stack([u, v], axis=-1)
    group = np.zeros(rel.shape[0], dtype=np.int64)
    return uv, group, singular


def _islands(topo: Topology, uv: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner ids (one per distinct (welded vertex, UV) within a connected island) and
    the island of every face."""
    T = topo.triangle_count
    flat = uv.reshape(-1, 2)
    h = np.flatnonzero(topo.mate >= 0)
    m = topo.mate[h]
    fh, kh, fm, km = h // 3, h % 3, m // 3, m % 3
    a1, a2 = 3 * fh + kh, 3 * fm + (km + 1) % 3
    b1, b2 = 3 * fh + (kh + 1) % 3, 3 * fm + km
    same = (
        (group[fh] == group[fm])
        & np.all(flat[a1] == flat[a2], axis=1)
        & np.all(flat[b1] == flat[b2], axis=1)
    )
    from .topology import union_labels

    faces = union_labels(T, fh[same], fm[same])
    island, _ = dense_labels(faces)
    corners = union_labels(3 * T, np.r_[a1[same], b1[same]], np.r_[a2[same], b2[same]])
    cid, _ = dense_labels(corners)
    return cid.reshape(T, 3), island
