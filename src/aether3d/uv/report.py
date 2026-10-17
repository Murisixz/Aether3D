"""UV diagnostics of a mesh: islands, orientation, overlaps, stretch, texel density, packing.

Every measure is a fact about the map, computed exactly or with a stated resolution:

* **islands** -- faces connected through edges whose two sides agree on their UVs;
* **orientation** -- a triangle is *flipped* when its UV winding disagrees with the
  majority of its island; an island whose triangles are all reversed is *mirrored*
  (intentional on symmetric models), which is reported separately;
* **degenerate** triangles have zero UV area while their 3D area is not zero;
* **overlaps** -- triangles whose UVs overlap (positive-area intersection), found exactly
  with the predicates of :mod:`aether3d.spatial.predicates`: within an island (a fold) or
  between two islands; an overlapping pair of islands
  with the same number of triangles and equal UV areas is reported as *stacked*
  (copies or mirrors placed on each other, a deliberate texel-sharing technique):
  ``overlapping_triangles`` counts the triangles of unintended overlaps only, and
  ``stacked_triangles`` those overlapping only through stacked pairs;
* **stretch** -- Sander et al.'s L2 and Linf stretch per island (see
  :mod:`aether3d.uv.metrics`);
* **texel density** -- texels per world unit at ``resolution`` (per island: sqrt of UV
  area x texels / 3D area); the spread is the coefficient of variation;
* **packing** -- the fraction of the unit square covered by islands (exact area), islands
  outside ``[0, 1]^2``, and the smallest gap between islands in texels (at
  ``resolution``, measured by dilation).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import TexturingError
from ..mesh import Mesh
from ..spatial.predicates import orient2d
from . import param
from .islands import island_labels
from .metrics import segment_pairs
from .topology import Topology


def _check_mesh(mesh: Any) -> Mesh:
    if not isinstance(mesh, Mesh):
        raise TexturingError("INVALID_ARGUMENT", "expected a Mesh", "mesh")
    if mesh.uvs is None:
        raise TexturingError("NO_UVS", "the mesh has no UVs", "mesh", hint="unwrap it first")
    return mesh


def uv_report(mesh: Mesh, *, resolution: Any = 1024, islands_limit: int = 64) -> dict[str, Any]:
    """The UV diagnostics of ``mesh`` (see the module docstring) as a JSON-ready dict."""
    from .unwrap import _check_resolution

    mesh = _check_mesh(mesh)
    width, height = _check_resolution(resolution)
    topo = Topology.from_mesh(mesh)
    T = topo.triangle_count
    labels, count = island_labels(topo)
    tri = topo.triangles
    uv = topo.uvs
    corner = uv[tri]
    e1 = corner[:, 1] - corner[:, 0]
    e2 = corner[:, 2] - corner[:, 0]
    signed = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    area3 = topo.face_area
    live = area3 > 0
    degenerate = live & (signed == 0)
    # Orientation per island: the sign of its total signed area.
    island_signed = np.bincount(labels, signed, count)
    island_sign = np.where(island_signed < 0, -1.0, 1.0)
    flipped = live & ~degenerate & (np.sign(signed) != island_sign[labels])
    mirrored = island_sign < 0
    finite = np.isfinite(corner).all(axis=(1, 2))
    # Stretch per island via the parameterization machinery (UVs scaled to texels below).
    cs = param.ChartSet(
        np.arange(3 * T).reshape(T, 3),
        topo.positions[tri].reshape(-1, 3),
        labels,
        np.repeat(labels, 3),
        count,
    )
    uv_flat = corner.reshape(-1, 2).copy()
    # Mirrored islands are measured as their mirror image (orientation is reported apart).
    uv_flat[:, 0] *= np.repeat(island_sign[labels], 3)
    from .metrics import stretch

    l2, linf = stretch(cs, uv_flat)
    uv_area = np.bincount(labels, np.abs(signed), count)
    island_area3 = np.bincount(labels, area3, count)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.sqrt(uv_area * width * height / island_area3)
    weights = island_area3 / max(float(island_area3.sum()), 1e-300)
    dens_ok = np.isfinite(density) & (island_area3 > 0)
    mean_density = (
        float((density[dens_ok] * weights[dens_ok]).sum() / max(weights[dens_ok].sum(), 1e-300))
        if dens_ok.any()
        else 0.0
    )
    spread = (
        float(
            np.sqrt(
                (weights[dens_ok] * (density[dens_ok] - mean_density) ** 2).sum()
                / max(weights[dens_ok].sum(), 1e-300)
            )
            / mean_density
        )
        if mean_density > 0
        else 0.0
    )
    lo = corner.reshape(-1, 2).min(axis=0) if T else np.zeros(2)
    hi = corner.reshape(-1, 2).max(axis=0) if T else np.zeros(2)
    tri_lo = corner.min(axis=1)
    tri_hi = corner.max(axis=1)
    outside = np.zeros(count, dtype=bool)
    out_tri = (tri_lo < 0).any(axis=1) | (tri_hi > 1).any(axis=1)
    outside[np.unique(labels[out_tri])] = True
    ta, tb = overlapping_triangles(corner, signed)
    inner = labels[ta] == labels[tb]
    folded = np.unique(labels[ta[inner]])
    diff = ~inner
    la, lb = labels[ta[diff]], labels[tb[diff]]
    keys = np.unique(np.minimum(la, lb) * count + np.maximum(la, lb))
    pairs = [(int(k // count), int(k % count)) for k in keys]
    stacked = []
    overlapping = []
    tri_count = np.bincount(labels, minlength=count)
    for a, b in pairs:
        same = tri_count[a] == tri_count[b] and math.isclose(
            uv_area[a], uv_area[b], rel_tol=1e-6, abs_tol=1e-12
        )
        (stacked if same else overlapping).append([int(a), int(b)])
    # Triangles of stacked island pairs overlap on purpose; every other overlap does not.
    stacked_keys = np.array([a * count + b for a, b in stacked], dtype=np.int64)
    pair_key = np.minimum(labels[ta], labels[tb]) * count + np.maximum(labels[ta], labels[tb])
    on_purpose = ~inner & np.isin(pair_key, stacked_keys)
    involved = np.unique(np.r_[ta[~on_purpose], tb[~on_purpose]])
    stacked_tris = np.setdiff1d(np.unique(np.r_[ta[on_purpose], tb[on_purpose]]), involved)
    gap = min_gap(corner, labels, count, (width, height)) if count > 1 else None
    finite_l2 = l2[np.isfinite(l2)]
    report = {
        "triangles": T,
        "islands": int(count),
        "flipped_triangles": int(flipped.sum()),
        "degenerate_triangles": int(degenerate.sum()),
        "nonfinite_triangles": int((~finite).sum()),
        "mirrored_islands": int(mirrored.sum()),
        "overlapping_triangles": int(involved.shape[0]),
        "stacked_triangles": int(stacked_tris.shape[0]),
        "self_overlapping_islands": int(folded.shape[0]),
        "overlapping_pairs": overlapping[:islands_limit],
        "overlapping_pair_count": len(overlapping),
        "stacked_pairs": stacked[:islands_limit],
        "stacked_pair_count": len(stacked),
        "stretch_l2": {
            "max": float(finite_l2.max()) if finite_l2.size else None,
            "mean": float(
                np.average(l2[np.isfinite(l2)], weights=island_area3[np.isfinite(l2)] + 1e-300)
            )
            if finite_l2.size
            else None,
            "invalid_islands": int((~np.isfinite(l2)).sum()),
        },
        "stretch_max": float(linf[np.isfinite(linf)].max()) if np.isfinite(linf).any() else None,
        "texel_density": {
            "resolution": [width, height],
            "mean": mean_density,
            "min": float(density[dens_ok].min()) if dens_ok.any() else 0.0,
            "max": float(density[dens_ok].max()) if dens_ok.any() else 0.0,
            "spread": spread,
        },
        "utilization": float(np.abs(signed).sum()),
        "bounds": {"min": lo.tolist(), "max": hi.tolist()},
        "islands_outside_unit_square": int(outside.sum()),
        "min_gap_texels": gap,
        "per_island": [
            {
                "island": int(i),
                "triangles": int(tri_count[i]),
                "uv_area": float(uv_area[i]),
                "area": float(island_area3[i]),
                "stretch_l2": float(l2[i]) if np.isfinite(l2[i]) else None,
                "texel_density": float(density[i]) if dens_ok[i] else None,
                "mirrored": bool(mirrored[i]),
            }
            for i in range(min(count, islands_limit))
        ],
    }
    return report


def overlapping_triangles(corner: np.ndarray, signed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs ``(i, j)`` of triangles whose UVs share positive area, exactly: an edge of one
    properly crosses an edge of the other, a vertex (or the centroid) of one lies strictly
    inside the other, or they coincide."""
    T = corner.shape[0]
    live = np.flatnonzero((signed != 0) & np.isfinite(corner).all(axis=(1, 2)))
    if live.size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    lo = corner[live].min(axis=1)
    hi = corner[live].max(axis=1)
    size = np.maximum(hi - lo, 0).max(axis=1)
    cell = np.array([max(float(np.median(size)) * 2.0, 1e-9)])
    a, b = segment_pairs(lo, hi, np.zeros(live.shape[0], dtype=np.int64), cell)
    if a.size == 0:
        return a, b
    ta, tb = live[a], live[b]
    boxes = np.all(corner[ta].min(1) < corner[tb].max(1), axis=1) & np.all(
        corner[tb].min(1) < corner[ta].max(1), axis=1
    )
    ta, tb = ta[boxes], tb[boxes]
    out_a, out_b = [], []
    step = 1 << 18
    for k in range(0, ta.shape[0], step):
        sa, sb = ta[k : k + step], tb[k : k + step]
        hit = _triangles_overlap(corner[sa], corner[sb])
        out_a.append(sa[hit])
        out_b.append(sb[hit])
    del T
    if not out_a:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(out_a), np.concatenate(out_b)


def island_overlaps(
    corner: np.ndarray, labels: np.ndarray, count: int, signed: np.ndarray
) -> list[tuple[int, int]]:
    """Pairs of different islands with overlapping UV triangles (see
    :func:`overlapping_triangles`)."""
    ta, tb = overlapping_triangles(corner, signed)
    diff = labels[ta] != labels[tb]
    la, lb = labels[ta[diff]], labels[tb[diff]]
    keys = np.unique(np.minimum(la, lb) * count + np.maximum(la, lb))
    return [(int(k // count), int(k % count)) for k in keys]


def _triangles_overlap(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Whether 2D triangles ``p`` and ``q`` (``(n, 3, 2)``) share interior area, exactly."""
    out = np.zeros(p.shape[0], dtype=bool)
    # Proper edge crossings.
    for i in range(3):
        a1, b1 = p[:, i], p[:, (i + 1) % 3]
        for j in range(3):
            a2, b2 = q[:, j], q[:, (j + 1) % 3]
            d1 = orient2d(a2, b2, a1)
            d2 = orient2d(a2, b2, b1)
            d3 = orient2d(a1, b1, a2)
            d4 = orient2d(a1, b1, b2)
            out |= (d1 * d2 < 0) & (d3 * d4 < 0)
    # A vertex strictly inside the other triangle, or coincident triangles.
    out |= _strictly_inside(p, q) | _strictly_inside(q, p)
    same = np.zeros_like(out)
    for k in range(3):
        same |= np.all(np.all(p == np.roll(q, k, axis=1), axis=2), axis=1)
        same |= np.all(np.all(p == np.roll(q[:, ::-1], k, axis=1), axis=2), axis=1)
    return out | same


def _strictly_inside(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Whether some vertex of ``q`` lies strictly inside ``p`` (or ``q``'s centroid does)."""
    s = orient2d(p[:, 0], p[:, 1], p[:, 2])
    out = np.zeros(p.shape[0], dtype=bool)
    points = [q[:, 0], q[:, 1], q[:, 2], q.mean(axis=1)]
    for v in points:
        o1 = orient2d(p[:, 0], p[:, 1], v)
        o2 = orient2d(p[:, 1], p[:, 2], v)
        o3 = orient2d(p[:, 2], p[:, 0], v)
        inside = np.where(s > 0, (o1 > 0) & (o2 > 0) & (o3 > 0), (o1 < 0) & (o2 < 0) & (o3 < 0))
        out |= (s != 0) & inside
    return out


def min_gap(corner: np.ndarray, labels: np.ndarray, count: int, resolution: tuple[int, int]):
    """The smallest number of free texels between two islands' conservative footprints
    (every texel a triangle touches) at ``resolution``, along rows, columns or diagonals
    (Chebyshev distance minus one); 0 when islands share or touch texels; ``None`` when no
    two islands are within 64 texels."""
    from .pack import footprint

    w, h = resolution
    pad = 34
    scaled = corner * np.array([w, h], dtype=np.float64)
    label = np.full((h + 2 * pad, w + 2 * pad), -1, dtype=np.int64)
    for c in range(count):
        mask, org = footprint(scaled[labels == c], 0)
        ys, xs = np.nonzero(mask)
        ys = np.clip(ys + org[1] + pad, 0, h + 2 * pad - 1)
        xs = np.clip(xs + org[0] + pad, 0, w + 2 * pad - 1)
        clash = (label[ys, xs] >= 0) & (label[ys, xs] != c)
        if clash.any():
            return 0
        label[ys, xs] = c
    dist = np.where(label >= 0, 0, np.iinfo(np.int64).max // 4)
    shifts = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for ring in range(1, 33):
        grown = label.copy()
        for dy, dx in shifts:
            src = np.roll(np.roll(label, dy, axis=0), dx, axis=1)
            take = (grown < 0) & (src >= 0)
            grown[take] = src[take]
        newly = (label < 0) & (grown >= 0)
        dist[newly] = ring
        label = grown
    best = None
    for dy, dx in shifts[::2] + ((1, 1), (1, -1)):
        nl = np.roll(np.roll(label, dy, axis=0), dx, axis=1)
        nd = np.roll(np.roll(dist, dy, axis=0), dx, axis=1)
        meet = (label >= 0) & (nl >= 0) & (label != nl)
        if meet.any():
            d = int((dist[meet] + nd[meet]).min())
            best = d if best is None else min(best, d)
    return best
