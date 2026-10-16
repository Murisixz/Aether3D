"""Operations on existing UV islands: transform, align, normalize, texel density, repack,
stitch.

Islands are found as in :func:`~aether3d.uv.report.uv_report` and numbered by their
lowest triangle. Every operation returns a new :class:`~aether3d.Mesh` (vertices are
split where islands that shared a vertex move apart) and never touches positions,
normals, triangles or face groups.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import TexturingError
from ..mesh import Mesh
from . import pack as packing
from .islands import island_labels
from .report import _check_mesh
from .topology import Topology
from .unwrap import UnwrapResult, _area, _check_resolution, _rebuild


def _islands(mesh: Mesh) -> tuple[Topology, np.ndarray, int, np.ndarray, np.ndarray]:
    topo = Topology.from_mesh(_check_mesh(mesh))
    labels, count = island_labels(topo)
    corner_uv = topo.uvs[topo.triangles].copy()
    corner_id = labels[:, None] * topo.positions.shape[0] + topo.triangles
    return topo, labels, count, corner_uv, corner_id


def _select(islands: Any, count: int) -> np.ndarray:
    if islands is None:
        return np.ones(count, dtype=bool)
    arr = np.asarray(islands)
    if arr.size and (
        not np.issubdtype(arr.dtype, np.integer) or arr.min() < 0 or arr.max() >= count
    ):
        raise TexturingError(
            "INVALID_ARGUMENT", f"islands must be island numbers in [0, {count - 1}]", "islands"
        )
    sel = np.zeros(count, dtype=bool)
    sel[arr.astype(np.int64).reshape(-1)] = True
    return sel


def _vec2(value: Any, name: str, default: tuple[float, float]) -> np.ndarray:
    if value is None:
        return np.array(default, dtype=np.float64)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value, value)
    try:
        v = np.asarray(value, dtype=np.float64).reshape(2)
    except (TypeError, ValueError):
        raise TexturingError(
            "INVALID_ARGUMENT", f"{name} must be a number or [u, v]", name
        ) from None
    if not np.all(np.isfinite(v)):
        raise TexturingError("INVALID_ARGUMENT", f"{name} must be finite", name)
    return v


def transform_islands(
    mesh: Mesh,
    islands: Any = None,
    *,
    translate: Any = None,
    rotate: float = 0.0,
    scale: Any = None,
    mirror: str | None = None,
    pivot: Any = "center",
) -> UnwrapResult:
    """Move, rotate (degrees, counter-clockwise), scale and mirror (``"u"`` or ``"v"``)
    islands, each about its own bounding-box center (``pivot="center"``) or a common
    ``[u, v]`` pivot. Order: mirror, scale, rotate, translate."""
    topo, labels, count, uv, cid = _islands(mesh)
    sel = _select(islands, count)
    t = _vec2(translate, "translate", (0.0, 0.0))
    s = _vec2(scale, "scale", (1.0, 1.0))
    if (
        isinstance(rotate, bool)
        or not isinstance(rotate, (int, float))
        or not math.isfinite(rotate)
    ):
        raise TexturingError("INVALID_ARGUMENT", "rotate must be a number of degrees", "rotate")
    if mirror not in (None, "u", "v"):
        raise TexturingError("INVALID_ARGUMENT", "mirror must be 'u', 'v' or None", "mirror")
    if np.any(s == 0):
        raise TexturingError("INVALID_ARGUMENT", "scale must not be zero", "scale")
    a = math.radians(float(rotate))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    for i in np.flatnonzero(sel):
        f = labels == i
        pts = uv[f]
        if isinstance(pivot, str) and pivot == "center":
            lo, hi = pts.reshape(-1, 2).min(0), pts.reshape(-1, 2).max(0)
            pv = 0.5 * (lo + hi)
        else:
            pv = _vec2(pivot, "pivot", (0.0, 0.0))
        q = pts - pv
        if mirror == "u":
            q[..., 0] = -q[..., 0]
        elif mirror == "v":
            q[..., 1] = -q[..., 1]
        q = q * s
        q = q @ rot.T
        uv[f] = q + pv + t
    return UnwrapResult(_rebuild(mesh, topo, cid, uv), labels, {"islands": int(sel.sum())})


def align_islands(mesh: Mesh, islands: Any = None, *, mode: str = "min_area") -> UnwrapResult:
    """Rotate islands about their centers: ``"min_area"`` to their minimum-area bounding
    rectangle (wider than tall), ``"u"``/``"v"`` so their longest boundary-hull edge is
    horizontal/vertical."""
    if mode not in ("min_area", "u", "v"):
        raise TexturingError("INVALID_ARGUMENT", "mode must be 'min_area', 'u' or 'v'", "mode")
    topo, labels, count, uv, cid = _islands(mesh)
    sel = _select(islands, count)
    for i in np.flatnonzero(sel):
        f = labels == i
        pts = uv[f].reshape(-1, 2)
        if mode == "min_area":
            ang = packing.min_area_angle(pts)
        else:
            hull = packing._hull(pts)
            if hull.shape[0] < 2:
                continue
            e = np.roll(hull, -1, axis=0) - hull
            k = int(np.argmax(np.hypot(e[:, 0], e[:, 1])))
            ang = -math.atan2(e[k, 1], e[k, 0]) + (math.pi / 2 if mode == "v" else 0.0)
        lo, hi = pts.min(0), pts.max(0)
        c = 0.5 * (lo + hi)
        q = packing._rotate(pts - c, ang) + c
        uv[f] = q.reshape(-1, 3, 2)
    return UnwrapResult(_rebuild(mesh, topo, cid, uv), labels, {"islands": int(sel.sum())})


def normalize_islands(mesh: Mesh, islands: Any = None, *, keep_aspect: bool = True) -> UnwrapResult:
    """Scale and move the selected islands together so they fill the unit square
    (keeping proportions unless ``keep_aspect=False``)."""
    topo, labels, count, uv, cid = _islands(mesh)
    sel = _select(islands, count)
    f = sel[labels]
    if not f.any():
        raise TexturingError("INVALID_ARGUMENT", "no island selected", "islands")
    pts = uv[f].reshape(-1, 2)
    lo, hi = pts.min(0), pts.max(0)
    span = hi - lo
    if keep_aspect:
        span = np.array([span.max(), span.max()])
    span = np.where(span > 0, span, 1.0)
    uv[f] = (uv[f] - lo) / span
    return UnwrapResult(_rebuild(mesh, topo, cid, uv), labels, {"islands": int(sel.sum())})


def set_texel_density(
    mesh: Mesh,
    texels_per_unit: float,
    *,
    resolution: Any = 1024,
    islands: Any = None,
) -> UnwrapResult:
    """Scale each selected island about its center so its texel density (texels per world
    unit at ``resolution``) is ``texels_per_unit``. Islands may then overlap or leave the
    unit square: repack with :func:`pack_islands` (which keeps one density for all)."""
    if (
        isinstance(texels_per_unit, bool)
        or not isinstance(texels_per_unit, (int, float))
        or not (math.isfinite(texels_per_unit) and texels_per_unit > 0)
    ):
        raise TexturingError(
            "INVALID_ARGUMENT", "texels_per_unit must be positive", "texels_per_unit"
        )
    w, h = _check_resolution(resolution)
    topo, labels, count, uv, cid = _islands(mesh)
    sel = _select(islands, count)
    for i in np.flatnonzero(sel):
        f = labels == i
        uv_area = _area(uv[f])
        area3 = float(topo.face_area[f].sum())
        if uv_area <= 0 or area3 <= 0:
            continue
        current = math.sqrt(uv_area * w * h / area3)
        k = texels_per_unit / current
        pts = uv[f]
        c = 0.5 * (pts.reshape(-1, 2).min(0) + pts.reshape(-1, 2).max(0))
        uv[f] = (pts - c) * k + c
    return UnwrapResult(_rebuild(mesh, topo, cid, uv), labels, {"islands": int(sel.sum())})


def pack_islands(
    mesh: Mesh,
    *,
    resolution: Any = 1024,
    padding: int = 4,
    rotate: bool = True,
    texels_per_unit: float | None = None,
    keep_density: bool = False,
) -> UnwrapResult:
    """Repack the existing islands into the unit square without overlaps. Islands are first
    scaled to one texel density (their shapes kept), unless ``keep_density=True`` keeps
    their current relative sizes."""
    res = _check_resolution(resolution)
    topo, labels, count, uv, cid = _islands(mesh)
    parts = []
    for i in range(count):
        f = labels == i
        t = uv[f].copy()
        if not keep_density:
            uv_area = _area(t)
            area3 = float(topo.face_area[f].sum())
            if uv_area > 0 and area3 > 0:
                t = t * math.sqrt(area3 / uv_area)
        parts.append(t)
    placed, result = packing.pack(
        parts, res, padding=int(padding), rotate=rotate, texels_per_unit=texels_per_unit
    )
    for i in range(count):
        uv[labels == i] = placed[i]
    report = {
        "islands": count,
        "resolution": list(res),
        "padding": int(padding),
        "texels_per_unit": result.texels_per_unit,
        "utilization": float(sum(_area(t) for t in placed)),
    }
    return UnwrapResult(_rebuild(mesh, topo, cid, uv), labels, report)


def stitch(mesh: Mesh, edges: Any, *, allow_overlap: bool = False) -> UnwrapResult:
    """Weld UV seams along ``edges`` (``[vertex, vertex]`` pairs of the mesh's edges whose
    two sides lie in different islands): each pair of islands is joined by moving the
    second (the higher-numbered) rigidly -- rotation and translation, mirrored when the two
    sides have opposite orientations -- so its side of the first stitched edge lands on the
    first island's; every listed edge between the two islands is then welded exactly.
    Refused (``STITCH_OVERLAP``) when the joined islands would overlap, unless
    ``allow_overlap``."""
    topo, labels, count, uv, cid = _islands(mesh)
    arr = np.asarray(edges)
    if arr.size == 0:
        raise TexturingError("INVALID_ARGUMENT", "no edge to stitch", "edges")
    if not np.issubdtype(arr.dtype, np.integer) or arr.ndim != 2 or arr.shape[1] != 2:
        raise TexturingError("INVALID_SEAM", "edges must be [vertex, vertex] pairs", "edges")
    n = topo.positions.shape[0]
    if arr.min() < 0 or arr.max() >= n:
        raise TexturingError("INVALID_SEAM", f"edge vertices must be in [0, {n - 1}]", "edges")
    wa, wb = topo.weld[arr[:, 0]], topo.weld[arr[:, 1]]
    a = topo.corners.reshape(-1)
    b = topo.corners[:, [1, 2, 0]].reshape(-1)
    # Half-edges of each requested edge, mated (an interior edge on a seam).
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, y in zip(wa.tolist(), wb.tolist(), strict=True):
        h = np.flatnonzero((a == x) & (b == y))
        if h.size == 0:
            h = np.flatnonzero((a == y) & (b == x))
        h = h[topo.mate[h] >= 0] if h.size else h
        if h.size != 1:
            raise TexturingError(
                "INVALID_SEAM", f"edge ({x}, {y}) is not an interior edge", "edges"
            )
        h0 = int(h[0])
        m0 = int(topo.mate[h0])
        i1, i2 = int(labels[h0 // 3]), int(labels[m0 // 3])
        if i1 == i2:
            continue  # already one island
        if i1 > i2:
            h0, m0, i1, i2 = m0, h0, i2, i1
        pairs.setdefault((i1, i2), []).append((h0, m0))
    if not pairs:
        raise TexturingError("NOTHING_TO_STITCH", "the edges already join their islands", "edges")
    flat = uv.reshape(-1, 2)
    merged = labels.copy()
    for _pair, hs in sorted(pairs.items()):
        h0, m0 = hs[0]
        # h0 runs a -> b in island i1; its mate runs b -> a in island i2.
        pa, pb = flat[3 * (h0 // 3) + h0 % 3], flat[3 * (h0 // 3) + (h0 % 3 + 1) % 3]
        qb, qa = flat[3 * (m0 // 3) + m0 % 3], flat[3 * (m0 // 3) + (m0 % 3 + 1) % 3]
        f2 = merged == merged[m0 // 3]
        orient1 = np.sign(_signed(uv[merged == merged[h0 // 3]]))
        orient2 = np.sign(_signed(uv[f2]))
        pts = uv[f2].reshape(-1, 2)
        if orient1 != orient2 and orient1 != 0 and orient2 != 0:
            pts = pts * np.array([-1.0, 1.0])
            qa, qb = qa * np.array([-1.0, 1.0]), qb * np.array([-1.0, 1.0])
        d1, d2 = pb - pa, qb - qa
        l1, l2 = float(np.hypot(*d1)), float(np.hypot(*d2))
        if l1 == 0 or l2 == 0:
            raise TexturingError("INVALID_SEAM", "a stitched edge has zero UV length", "edges")
        ang = math.atan2(d1[1], d1[0]) - math.atan2(d2[1], d2[0])
        k = l1 / l2
        moved = packing._rotate((pts - qa) * k, ang) + pa
        uv[f2] = moved.reshape(-1, 3, 2)
        flat = uv.reshape(-1, 2)
        # Weld every listed edge between these islands exactly (to island i1's values):
        # every corner of the moved island at a stitched vertex takes the value.
        weld = topo.corners.reshape(-1)
        moved_corner = np.repeat(f2, 3)
        for h, _m in hs:
            ca, cb = 3 * (h // 3) + h % 3, 3 * (h // 3) + (h % 3 + 1) % 3
            for c in (ca, cb):
                flat[moved_corner & (weld == weld[c])] = flat[c]
        merged[f2] = merged[h0 // 3]
    uv = flat.reshape(-1, 3, 2)
    moved = np.isin(merged, [merged[hs[0][0] // 3] for hs in pairs.values()])
    if not allow_overlap and _overlaps_involving(uv, moved):
        raise TexturingError(
            "STITCH_OVERLAP",
            "the stitched islands would overlap themselves or other islands",
            "edges",
            hint="stitch fewer edges, or pass allow_overlap=True",
        )
    out_labels, _ = np.unique(merged, return_inverse=True)
    del out_labels
    new_mesh = _rebuild(mesh, topo, _stitched_ids(topo, uv, merged), uv)
    return UnwrapResult(
        new_mesh,
        np.unique(merged, return_inverse=True)[1].reshape(-1),
        {"stitched_pairs": len(pairs)},
    )


def _signed(t: np.ndarray) -> float:
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float((e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum())


def _overlaps_involving(corner: np.ndarray, involved: np.ndarray) -> bool:
    """Whether a triangle of an ``involved`` face overlaps another triangle (positive area)."""
    from .metrics import segment_pairs
    from .report import _triangles_overlap

    e1, e2 = corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]
    signed = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    live = np.flatnonzero(signed != 0)
    if live.size < 2:
        return False
    lo, hi = corner[live].min(1), corner[live].max(1)
    size = np.maximum(hi - lo, 0).max(axis=1)
    cell = np.array([max(float(np.median(size)) * 2.0, 1e-9)])
    a, b = segment_pairs(lo, hi, np.zeros(live.shape[0], dtype=np.int64), cell)
    ta, tb = live[a], live[b]
    keep = involved[ta] | involved[tb]
    ta, tb = ta[keep], tb[keep]
    if ta.size == 0:
        return False
    return bool(_triangles_overlap(corner[ta], corner[tb]).any())


def _stitched_ids(topo: Topology, uv: np.ndarray, merged: np.ndarray) -> np.ndarray:
    """Corner ids: corners of one welded vertex in one (merged) island with equal UVs share
    an id."""
    flat = uv.reshape(-1, 2)
    w = topo.corners.reshape(-1)
    isl = np.repeat(merged, 3)
    key = np.stack([isl, w], axis=1).astype(np.float64)
    rows = np.concatenate([key, flat], axis=1)
    _, inv = np.unique(rows, axis=0, return_inverse=True)
    return inv.reshape(-1, 3).astype(np.int64)
