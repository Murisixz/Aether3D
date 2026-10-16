"""Isotropic remeshing and dynamic topology, built on :mod:`aether3d.sculpt.topology`.

Isotropic remeshing (Botsch and Kobbelt, "A Remeshing Approach to Multiresolution
Modeling", SGP 2004, section 4) repeats, for ``iterations`` rounds:

1. split every edge longer than ``4/3 L`` (repeated until none is left);
2. collapse edges shorter than ``4/5 L`` (independent sets, a few passes);
3. flip edges to bring valences toward 6;
4. relax vertices tangentially toward the area-weighted centroid of their one-ring;
5. project the relaxed vertices back onto the surface the remesh started from.

Features (boundaries, seams, hard edges, face-group borders, non-manifold edges),
masked vertices and degenerate triangles are kept as they are (see the topology
module); splits still refine feature edges. The result is measured, not assumed: the
operation returns the edge-length statistics and the number of operations of each kind.

The passes run on a compact copy of the mesh that keeps no history (whole-mesh passes
would otherwise journal every element and keep every replaced row), compacted again when
dead rows pile up; the result replaces the mesh in one journaled step. The checks of
each pass run in batches of candidates, so memory follows the mesh, not the candidates
times their neighbourhoods.

Dynamic topology (``brush.dyntopo``) runs the same passes inside a brush's sphere before
each dab: splits down to the detail size, collapses (``refine`` = ``"subdivide"``,
``"collapse"`` or ``"both"``), Delaunay flips and a tangential relaxation (``relax``,
default 0.5; 0 turns it off). Detail is ``"constant"`` (world units) or ``"relative"``
(a fraction of the brush radius).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ValidationError
from .topology import (
    COLLAPSE_RATIO,
    SPLIT_RATIO,
    build_region,
    collapse_pass,
    flip_pass,
    relax_pass,
    split_pass,
    world_lengths,
)

MAX_SPLIT_PASSES = 8
"""Split passes per round (each halves the longest edges)."""

MAX_COLLAPSE_PASSES = 24
"""Collapse passes per round (each collapses an independent set)."""

MAX_FLIP_PASSES = 8
"""Flip passes per round."""


def edge_statistics(sm) -> dict[str, float]:
    tris = sm.triangle_ids()
    tri = sm._tri[tris]
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    if not len(e):
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    length = world_lengths(sm, e[:, 0], e[:, 1])
    return {
        "min": float(length.min()),
        "mean": float(length.mean()),
        "max": float(length.max()),
        "std": float(length.std()),
    }


def isotropic_remesh(
    sm,
    target: float,
    *,
    iterations: int = 5,
    flips: str = "valence",
    relax: float = 0.5,
    project: bool = True,
    max_triangles: int | None = None,
) -> dict[str, Any]:
    """Remesh the whole sculpt mesh toward edges of world length ``target``."""
    from ..spatial.surface import TriangleSurface

    if not (target > 0 and np.isfinite(target)):
        raise ValidationError("target length must be a finite number > 0")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 50:
        raise ValidationError("iterations must be an integer in 1..50")
    if flips not in ("valence", "delaunay", "none"):
        raise ValidationError("flips must be 'valence', 'delaunay' or 'none'")
    if not 0.0 <= relax <= 1.0:
        raise ValidationError("relax must be in [0, 1]")
    budget = max_triangles if max_triangles is not None else sm._limit
    # Refuse early when the target would explode the mesh (area / equilateral area).
    area = _surface_area(sm)
    expected = int(area / (0.433 * target * target)) if target > 0 else 0
    if expected > budget:
        from ..errors import SculptError

        raise SculptError(
            "REMESH_TOO_DENSE",
            f"about {expected} triangles would be needed (limit {budget})",
            hint="use a larger target length",
        )
    # The passes run on a compact copy without history (whole-mesh passes would journal
    # every element and keep their old rows); the result replaces the mesh in one
    # journaled step. Stopping (should_stop) leaves the mesh as it was.
    work = sm._detached()
    reference = None
    if project:
        reference = TriangleSurface(work._pos[: work._nv], work._tri[work.triangle_ids()])
    hi, lo = SPLIT_RATIO * target, COLLAPSE_RATIO * target
    # Splits, collapses and relaxation keep vertices within about an edge of the surface:
    # projections search twice the split length (local units) and ask again without a
    # bound for the rare vertex with nothing that near.
    sigma = np.linalg.svd(np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3], compute_uv=False)
    bound = 2.0 * hi / max(float(sigma.min()), 1e-300)
    counts = {"splits": 0, "collapses": 0, "flips": 0, "relaxed": 0}

    def phase(run, limit: int, key: str) -> None:
        nonlocal work
        for _ in range(limit):
            sm._checkpoint()
            done = run(work, _whole(work))
            counts[key] += done
            if not done:
                break
        if work._nt > 1.5 * work.triangle_count + 1024:  # drop the rows passes left dead
            work = work._detached()

    for _ in range(iterations):
        phase(
            lambda w, region: split_pass(w, region, hi, max_new=budget), MAX_SPLIT_PASSES, "splits"
        )
        phase(
            lambda w, region: collapse_pass(w, region, lo, hi, max_ops=budget),
            MAX_COLLAPSE_PASSES,
            "collapses",
        )
        if flips != "none":
            phase(
                lambda w, region: flip_pass(w, region, flips, max_ops=budget),
                MAX_FLIP_PASSES,
                "flips",
            )
        if relax > 0:
            for _ in range(2):
                sm._checkpoint()
                region = _whole(work)
                counts["relaxed"] += relax_pass(work, region, relax)
            if reference is not None:
                _project(work, region, reference, bound)
    sm._checkpoint()
    sm._adopt(work)
    return {**counts, "edge_length": edge_statistics(sm), "triangles": sm.triangle_count}


def _surface_area(sm) -> float:
    tris = sm.triangle_ids()
    return float(0.5 * np.sqrt((sm._cross[tris] ** 2).sum(axis=1)).sum()) * _area_scale(sm)


def _area_scale(sm) -> float:
    m = np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3]
    s = np.linalg.svd(m, compute_uv=False)
    return float(s[0] * s[1])


def _whole(sm):
    return build_region(sm, sm.triangle_ids(), sm.vertex_ids())


PROJECT_BATCH = 1 << 15
"""Points per lockstep traversal when projecting (wide: projections ask for every free
vertex at once)."""


def _project(sm, region, reference, bound: float = np.inf) -> None:
    """Move the free vertices onto the reference surface (exact nearest points), searching
    within ``bound`` first (local units; the same nearest point when it lies that near)
    and without a bound for the vertices that found nothing."""
    free = region.verts[~region.locked]
    if not len(free):
        return
    points = sm._pos[free]
    hit = reference.nearest(points, max_distance=bound, batch=PROJECT_BATCH)
    missed = np.flatnonzero(hit.triangles < 0)
    if len(missed) and np.isfinite(bound):
        again = reference.nearest(points[missed], batch=PROJECT_BATCH)
        hit.points[missed] = again.points
        hit.triangles[missed] = again.triangles
    ok = hit.triangles >= 0
    if ok.any():
        sm._move(free[ok], hit.points[ok], region.tris)


def dyntopo_dab(
    sm, center: np.ndarray, local_radius: float, world_radius: float, spec: dict
) -> dict:
    """Refine (and coarsen) the topology inside a dab's sphere (see the module docs)."""
    detail = float(spec["detail"])
    length = detail * world_radius if spec.get("mode", "relative") == "relative" else detail
    refine = spec.get("refine", "both")
    hi, lo = SPLIT_RATIO * length, COLLAPSE_RATIO * length
    max_new = int(spec.get("max_splits", 20_000))
    counts = {"splits": 0, "collapses": 0, "flips": 0}
    m = np.asarray(sm.world_matrix, dtype=np.float64)[:3, :3]

    def region_now():
        # Splits stay inside the brush; collapses, flips and relaxation also reach a ring
        # of one radius around it, so the transition to the coarse surface is improved
        # rather than left as slivers between refined and locked vertices.
        ids, _d, tris = sm.vertices_in_sphere(center, 2.0 * local_radius)
        if not len(ids):
            return None
        rel = (sm._pos[ids] - center) @ m.T
        dist = np.sqrt((rel * rel).sum(axis=1))
        inside = ids[dist <= world_radius]
        if not len(inside):
            return None
        return build_region(sm, tris, ids[dist <= 2.0 * world_radius], inside)

    if refine in ("subdivide", "both"):
        for _ in range(MAX_SPLIT_PASSES):
            region = region_now()
            if region is None:
                break
            done = split_pass(sm, region, hi, max_new)
            counts["splits"] += done
            if not done:
                break
    if refine in ("collapse", "both"):
        region = region_now()
        if region is not None:
            counts["collapses"] += collapse_pass(sm, region, lo, hi, max_new)
    if spec.get("flip", True):
        region = region_now()
        if region is not None:
            counts["flips"] += flip_pass(sm, region, "delaunay", max_new)
    strength = float(spec.get("relax", 0.5))
    if strength > 0:
        region = region_now()
        if region is not None:
            counts["relaxed"] = relax_pass(sm, region, strength)
    return counts


def check_dyntopo(spec: Any) -> dict:
    """Validate a brush's ``dyntopo`` settings."""
    if not isinstance(spec, dict):
        raise ValidationError("dyntopo is an object")
    allowed = {"detail", "mode", "refine", "flip", "relax", "max_splits"}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        raise ValidationError(f"unknown dyntopo settings: {', '.join(unknown)}")
    if "detail" not in spec:
        raise ValidationError("dyntopo needs a detail size")
    detail = spec["detail"]
    if isinstance(detail, bool) or not isinstance(detail, (int, float)) or not detail > 0:
        raise ValidationError("dyntopo detail must be a number > 0")
    if not np.isfinite(float(detail)):
        raise ValidationError("dyntopo detail must be finite")
    if spec.get("mode", "relative") not in ("relative", "constant"):
        raise ValidationError("dyntopo mode must be 'relative' or 'constant'")
    if spec.get("mode", "relative") == "relative" and not 0.01 <= float(detail) <= 1.0:
        raise ValidationError("a relative detail is a fraction of the radius in [0.01, 1]")
    if spec.get("refine", "both") not in ("subdivide", "collapse", "both"):
        raise ValidationError("dyntopo refine must be 'subdivide', 'collapse' or 'both'")
    flip = spec.get("flip", True)
    if not isinstance(flip, bool):
        raise ValidationError("dyntopo flip must be true or false")
    relax = spec.get("relax", 0.5)
    if isinstance(relax, bool) or not isinstance(relax, (int, float)) or not 0 <= relax <= 1:
        raise ValidationError("dyntopo relax must be a number in [0, 1]")
    ms = spec.get("max_splits", 20_000)
    if isinstance(ms, bool) or not isinstance(ms, int) or not 1 <= ms <= 1_000_000:
        raise ValidationError("dyntopo max_splits must be an integer in 1..1,000,000")
    return dict(spec)
