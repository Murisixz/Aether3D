"""Distances along the surface for geodesic brush falloff.

``surface_distances`` solves the eikonal equation ``|grad d| = 1`` on the triangles near
a dab with the fast iterative method on triangulated surfaces (Fu, Jeong, Pan, Kirby and
Whitaker, SIAM J. Sci. Comput. 33(5), 2011): all triangles update their vertices at once
(Jacobi style) until no distance changes, which suits NumPy. Each update unfolds a
triangle into its plane: the two known corners ``A``, ``B`` with distances ``dA``, ``dB``
define a virtual source on the far side of ``AB``; the third corner gets the distance to
that source when the straight path crosses ``AB``, and the shorter of ``dA + |AC|`` and
``dB + |BC|`` otherwise. Distances are measured in world space.

Unlike Euclidean distance, nothing reaches vertices that are not connected to the dab
through the nearby triangles (a finger next to another, a lip over a chin).
"""

from __future__ import annotations

import numpy as np


def surface_distances(sm, ids: np.ndarray, region: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Surface distances (world units) from the dab center ``c`` (local) to the vertices
    ``ids`` over the ``region`` triangles; ``inf`` where unreachable."""
    linear = np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3]
    tri = sm._tri[region]
    verts = np.unique(tri.ravel())
    local = np.full(sm._nv, -1, dtype=np.int64)
    local[verts] = np.arange(len(verts))
    t = local[tri]
    p = sm._pos[verts] @ linear.T  # world positions
    center = linear @ c
    d = np.full(len(verts), np.inf)
    # Seed: the corners of the triangle nearest to the center get their straight distance.
    from ..spatial.geometry import closest_points_on_triangles

    cp, _ = closest_points_on_triangles(center, p[t[:, 0]], p[t[:, 1]], p[t[:, 2]])
    gap = np.sqrt(((cp - center) ** 2).sum(axis=1))
    seed = t[int(np.argmin(gap))]
    d[seed] = np.sqrt(((p[seed] - center) ** 2).sum(axis=1))
    a_idx = np.concatenate([t[:, 0], t[:, 1], t[:, 2]])
    b_idx = np.concatenate([t[:, 1], t[:, 2], t[:, 0]])
    c_idx = np.concatenate([t[:, 2], t[:, 0], t[:, 1]])
    pa, pb, pc = p[a_idx], p[b_idx], p[c_idx]
    ab = pb - pa
    e = np.sqrt((ab * ab).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        u = ab / e[:, None]
    ac = pc - pa
    cx = (ac * u).sum(axis=1)
    cy = np.sqrt(np.maximum((ac * ac).sum(axis=1) - cx * cx, 0.0))
    lac = np.sqrt((ac * ac).sum(axis=1))
    bc = pc - pb
    lbc = np.sqrt((bc * bc).sum(axis=1))
    for _ in range(4 * len(verts) + 8):
        da, db = d[a_idx], d[b_idx]
        known = np.isfinite(da) & np.isfinite(db)
        cand = np.minimum(da + lac, db + lbc)  # along the edges
        with np.errstate(invalid="ignore", divide="ignore"):
            sx = (da * da - db * db + e * e) / (2 * e)
            sy2 = da * da - sx * sx
            sy = -np.sqrt(np.maximum(sy2, 0.0))
            through = sx + (cx - sx) * (-sy) / (cy - sy)
            straight = np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2)
        ok = known & (sy2 >= 0) & (e > 0) & (through >= 0) & (through <= e) & (cy > 0)
        cand = np.where(ok, np.minimum(cand, straight), cand)
        new = d.copy()
        np.minimum.at(new, c_idx, cand)
        if np.array_equal(new, d):  # a fixed point: every update is a minimum
            break
        d = new
    out = np.full(len(ids), np.inf)
    inside = local[ids] >= 0
    out[inside] = d[local[ids[inside]]]
    return out
