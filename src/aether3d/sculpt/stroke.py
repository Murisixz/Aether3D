"""Strokes: a path sampled into dabs, each dab applying a brush (docs/MODELING.md).

A stroke is plain data: a :class:`~aether3d.sculpt.brushes.Brush`, a path of samples
(positions with an optional pressure) and how to place them:

* ``space``: ``"world"`` (default) or the object's ``"local"`` coordinates;
* ``project``: ``"nearest"`` snaps every dab center to the closest point of the current
  surface (default), ``"ray"`` casts it along ``view`` onto the surface (dabs that miss
  are skipped and counted), ``"none"`` uses the samples as they are;
* ``resample``: place dabs every ``brush.spacing * radius`` (world units) along the
  path, with pressure interpolated (default), or use the samples as dabs.

**Symmetry** mirrors each dab across the object's local planes through its origin
(``brush.symmetry``, any of ``"xyz"``; negative scales and parents included, since the
mirror is taken in local space). All mirrored dabs of one sample are computed from the
same positions and their displacements added, so the result does not depend on their
order and a symmetric mesh stays exactly symmetric; vertices lying on a symmetry plane
(within ``1e-9`` of the mesh size) stay on it.

**Distances and directions are world quantities.** Candidates come from the mesh's
chunks with a local radius large enough for any transform (``radius / smallest
singular value``); distances, normals and displacements are evaluated in world space
and mapped back with the inverse of the object's linear transform, so brushes behave
the same under non-uniform scale. Normals are the local (outward) normals mapped by the
inverse transpose, which keeps them outward under mirroring transforms too.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import SculptError, ValidationError
from .brushes import Brush, kelvinlet

MAX_SAMPLES = 100_000
"""Most samples one stroke may have."""

MAX_DABS = 200_000
"""Most dabs one stroke may make after resampling."""


@dataclass(frozen=True)
class Stroke:
    """A brush and the path it follows (see the module documentation)."""

    brush: Brush
    points: np.ndarray  # (n, 3)
    pressure: np.ndarray  # (n,)
    space: str = "world"
    project: str = "nearest"
    view: tuple[float, float, float] | None = None
    resample: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stroke:
        if not isinstance(data, dict):
            raise ValidationError("a stroke is an object")
        allowed = {"brush", "path", "space", "project", "view", "resample"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValidationError(f"unknown stroke settings: {', '.join(unknown)}")
        if "brush" not in data or "path" not in data:
            raise ValidationError("a stroke needs a brush and a path")
        brush = (
            data["brush"] if isinstance(data["brush"], Brush) else Brush.from_dict(data["brush"])
        )
        points, pressure = parse_path(data["path"])
        view = data.get("view")
        if view is not None:
            if not isinstance(view, (list, tuple, np.ndarray)) or not all(
                isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)
                for x in view
            ):
                raise ValidationError("view must be a non-zero 3D direction")
            v = np.asarray(view, dtype=np.float64)
            if v.shape != (3,) or not np.all(np.isfinite(v)) or not np.any(v):
                raise ValidationError("view must be a non-zero 3D direction")
            view = tuple(float(x) for x in v / np.linalg.norm(v))
        if not isinstance(data.get("resample", True), bool):
            raise ValidationError("resample must be true or false")
        stroke = cls(
            brush,
            points,
            pressure,
            space=data.get("space", "world"),
            project=data.get("project", "nearest"),
            view=view,
            resample=bool(data.get("resample", True)),
        )
        stroke.check()
        return stroke

    def check(self) -> None:
        if self.space not in ("world", "local"):
            raise ValidationError("space must be 'world' or 'local'")
        if self.project not in ("nearest", "ray", "none"):
            raise ValidationError("project must be 'nearest', 'ray' or 'none'")
        needs_view = (
            self.project == "ray"
            or self.brush.front_faces_only
            or self.brush.falloff_shape == "projected"
            or self.brush.normal == "view"
        )
        if needs_view and self.view is None:
            raise ValidationError("this stroke needs a view direction ('view')")

    def to_dict(self) -> dict[str, Any]:
        return {
            "brush": self.brush.to_dict(),
            "path": [
                {"position": p.tolist(), "pressure": float(q)}
                for p, q in zip(self.points, self.pressure, strict=True)
            ],
            "space": self.space,
            "project": self.project,
            "view": None if self.view is None else list(self.view),
            "resample": self.resample,
        }


def parse_path(path: Any) -> tuple[np.ndarray, np.ndarray]:
    """Points and pressures of a path: a list of ``[x, y, z]`` or of
    ``{"position": [x, y, z], "pressure": p}`` (pressure in ``[0, 1]``, default 1)."""
    if not isinstance(path, (list, tuple)) or not path:
        raise ValidationError("a path is a non-empty list of points")
    if len(path) > MAX_SAMPLES:
        raise ValidationError(f"a path has at most {MAX_SAMPLES} samples")
    points, pressure = [], []
    for k, item in enumerate(path):
        if isinstance(item, dict):
            extra = set(item) - {"position", "pressure"}
            if extra or "position" not in item:
                raise ValidationError(f"path[{k}] needs a position (and an optional pressure)")
            pos, pres = item["position"], item.get("pressure", 1.0)
        else:
            pos, pres = item, 1.0
        number = (int, float, np.integer, np.floating)
        if (
            not isinstance(pos, (list, tuple, np.ndarray))
            or not all(isinstance(x, number) and not isinstance(x, bool) for x in pos)
            or isinstance(pres, bool)
            or not isinstance(pres, number)
        ):
            raise ValidationError(f"path[{k}] is not a point")
        p = np.asarray(pos, dtype=np.float64)
        q = float(pres)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValidationError(f"path[{k}] must be three finite numbers")
        if not 0.0 <= q <= 1.0:
            raise ValidationError(f"path[{k}].pressure must be in [0, 1]")
        points.append(p)
        pressure.append(q)
    return np.array(points), np.array(pressure)


def dab_positions(
    points: np.ndarray, pressure: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dabs every ``step`` along the polyline (the first and last points included)."""
    if len(points) == 1:
        return points.copy(), pressure.copy()
    seg = np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total == 0.0:
        return points[:1].copy(), pressure[:1].copy()
    count = int(math.floor(total / step)) + 1
    if count > MAX_DABS:
        raise SculptError(
            "TOO_MANY_DABS",
            f"the stroke would make {count} dabs (at most {MAX_DABS})",
            hint="increase the brush's spacing or radius, or split the stroke",
        )
    t = np.arange(count) * step
    if total - t[-1] > 1e-9 * total:
        t = np.append(t, total)
    out = np.stack([np.interp(t, arc, points[:, k]) for k in range(3)], axis=1)
    return out, np.interp(t, arc, pressure)


@dataclass
class _Variant:
    """One mirror image of the stroke: its local mirror and the state its brush keeps."""

    mirror: np.ndarray  # (3,) of +-1
    state: dict = field(default_factory=dict)


class StrokeRunner:
    """Applies one stroke to a :class:`~aether3d.sculpt.SculptMesh` (inside its
    operation). Use :meth:`SculptMesh.stroke`."""

    def __init__(self, sm, stroke: Stroke) -> None:
        self.sm = sm
        self.stroke = stroke
        self.brush = stroke.brush
        m = np.asarray(sm.world_matrix, dtype=np.float64)
        self.linear = m[:3, :3].copy()
        self.translation = m[:3, 3].copy()
        det = float(np.linalg.det(self.linear))
        if not abs(det) > 1e-300:
            raise SculptError(
                "SINGULAR_WORLD_TRANSFORM", "the object's world transform is singular"
            )
        self.inv = np.linalg.inv(self.linear)
        sv = np.linalg.svd(self.linear, compute_uv=False)
        self.min_scale = float(sv.min())
        axes = [ord_axis(c) for c in stroke.brush.symmetry]
        mirrors = [np.ones(3)]
        for a in axes:
            mirrors += [m_ * np.where(np.arange(3) == a, -1.0, 1.0) for m_ in mirrors]
        self.variants = [_Variant(mm) for mm in mirrors]
        self.sym_axes = axes
        self.plane_tol = 1e-9 * sm.scale()
        self.view_world = None if stroke.view is None else np.asarray(stroke.view, dtype=np.float64)
        self.timings = {"query": 0.0, "deform": 0.0, "apply": 0.0}
        self.skipped = 0
        self.max_displacement = 0.0
        self.topology: dict[str, int] = {}

    # -- placement -------------------------------------------------------------------

    def to_local(self, p: np.ndarray) -> np.ndarray:
        return self.inv @ (p - self.translation)

    def dabs(self) -> list[tuple[np.ndarray, float]]:
        """Dab centers (local) and pressures, before projection."""
        st = self.stroke
        pts = st.points
        world = pts if st.space == "world" else pts @ self.linear.T + self.translation
        if st.resample:
            step = self.brush.spacing * self.brush.radius
            world, pres = dab_positions(world, st.pressure, step)
        else:
            pres = st.pressure
        local = (world - self.translation) @ self.inv.T
        return [(local[i], float(pres[i])) for i in range(len(local))]

    def project(self, c_local: np.ndarray) -> np.ndarray | None:
        mode = self.stroke.project
        if mode == "none":
            return c_local
        if mode == "nearest":
            hit = self.sm.nearest_point(c_local)
            return None if hit is None else hit[0]
        d_local = self.inv @ self.view_world
        hit = self.sm.raycast(
            c_local - 1e6 * self.sm.scale() * d_local / np.linalg.norm(d_local), d_local
        )
        if hit is None:
            return None
        t, tri, bary = hit
        p = self.sm._pos[self.sm._tri[tri]]
        return bary @ p

    # -- the stroke --------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        sm, brush = self.sm, self.brush
        dabs = self.dabs()
        applied = 0
        for k, (c_raw, pressure) in enumerate(dabs):
            sm._checkpoint()
            c = self.project(c_raw)
            if c is None:
                self.skipped += 1
                continue
            radius = brush.radius * (pressure if brush.pressure in ("radius", "both") else 1.0)
            strength = brush.strength * (
                pressure if brush.pressure in ("strength", "both") else 1.0
            )
            if radius <= 0 or (strength <= 0 and brush.kind not in ("grab", "elastic")):
                continue
            if brush.dyntopo is not None:
                from .remesh import dyntopo_dab

                t_topo = time.perf_counter()
                for variant in self.variants:
                    counts = dyntopo_dab(
                        sm, c * variant.mirror, radius / self.min_scale, radius, brush.dyntopo
                    )
                    for key, value in counts.items():
                        self.topology[key] = self.topology.get(key, 0) + value
                self.timings["topology"] = self.timings.get("topology", 0.0) + (
                    time.perf_counter() - t_topo
                )
            moves: list[tuple[np.ndarray, np.ndarray]] = []
            regions: list[np.ndarray] = []
            for variant in self.variants:
                out = self.dab(variant, c * variant.mirror, radius, strength, k)
                if out is None:
                    continue
                ids, disp, region = out
                moves.append((ids, disp))
                regions.append(region)
            if not moves:
                continue
            t0 = time.perf_counter()
            ids = np.concatenate([m[0] for m in moves])
            disp = np.concatenate([m[1] for m in moves])
            uniq, inverse = np.unique(ids, return_inverse=True)
            total = np.zeros((len(uniq), 3))
            np.add.at(total, inverse, disp)
            if self.sym_axes:
                on_plane = uniq
                for a in self.sym_axes:
                    flat = np.abs(sm._pos[on_plane, a]) <= self.plane_tol
                    total[flat, a] = 0.0
            moving = np.any(total != 0.0, axis=1)
            if moving.any():
                uniq, total = uniq[moving], total[moving]
                region = regions[0] if len(regions) == 1 else np.unique(np.concatenate(regions))
                new = sm._pos[uniq] + total
                world_disp = np.sqrt(((total @ self.linear.T) ** 2).sum(axis=1))
                self.max_displacement = max(self.max_displacement, float(world_disp.max()))
                sm._move(uniq, new, region)
                applied += 1
            self.timings["apply"] += time.perf_counter() - t0
        moved = int((sm._stamp[: sm._nv] == sm._serial).sum())
        out = {
            "dabs": len(dabs),
            "applied_dabs": applied,
            "skipped_dabs": self.skipped,
            "moved_vertices": moved,
            "max_displacement": self.max_displacement,
        }
        if brush.dyntopo is not None:
            out["topology"] = dict(self.topology)
            out["triangles"] = sm.triangle_count
        return out

    # -- one dab -----------------------------------------------------------------------

    def dab(self, variant: _Variant, c: np.ndarray, radius: float, strength: float, k: int):
        sm, brush = self.sm, self.brush
        kind = brush.kind
        t0 = time.perf_counter()
        if kind == "elastic":
            return self._elastic(variant, c, radius, k)
        if kind == "grab" and "ids" in variant.state:
            return self._grab_follow(variant, c)
        local_r = radius / self.min_scale
        ids, _dist, region = sm.vertices_in_sphere(c, local_r)
        self.timings["query"] += time.perf_counter() - t0
        t1 = time.perf_counter()
        if not len(ids):
            return None
        rel = (sm._pos[ids] - c) @ self.linear.T  # world offsets from the dab center
        mirror = variant.mirror
        view = (
            None
            if self.view_world is None
            else self.linear @ (mirror * (self.inv @ self.view_world))
        )
        if view is not None:
            view = view / np.linalg.norm(view)
        if brush.falloff_shape == "projected":
            along = rel @ view
            dist = np.sqrt(np.maximum((rel * rel).sum(axis=1) - along * along, 0.0))
        else:
            dist = np.sqrt((rel * rel).sum(axis=1))
        inside = dist <= radius
        if brush.geodesic and inside.any():
            from .geodesic import surface_distances

            dist = surface_distances(sm, ids, region, c)
            inside &= dist <= radius
        ids, rel, dist = ids[inside], rel[inside], dist[inside]
        if not len(ids):
            return None
        w = brush.weights(dist / radius)
        if brush.use_mask:
            w = w * (1.0 - sm._mask[ids].astype(np.float64))
        normals = self._world_normals(ids)
        if brush.front_faces_only:
            w = np.where((normals @ view) < 0, w, 0.0)
        if brush.automask:
            w = w * self._automask(ids, rel, region)
        if not np.any(w > 0):
            return None
        disp = self._displacement(variant, ids, rel, normals, w, radius, strength, c, region, view)
        self.timings["deform"] += time.perf_counter() - t1
        if disp is None:
            return None
        return ids, disp @ self.inv.T, region

    def _world_normals(self, ids: np.ndarray) -> np.ndarray:
        # Local normals follow the local winding (outward); the inverse transpose maps
        # them to the outward world normals under any invertible transform, mirrors
        # included (the inside of a mirrored solid is still its inside).
        n_local = self.sm.vertex_normals(ids)
        n = n_local @ self.inv  # inverse transpose: (inv.T @ n^T)^T = n @ inv
        length = np.sqrt((n * n).sum(axis=1))
        ok = length > 0
        n[ok] /= length[ok, None]
        return n

    def _area(self, rel, normals, w, radius) -> tuple[np.ndarray, np.ndarray]:
        dist = np.sqrt((rel * rel).sum(axis=1))
        near = dist <= self.brush.normal_radius * radius
        weights = w * near if np.any(w * near > 0) else w
        n = (normals * weights[:, None]).sum(axis=0)
        length = float(np.linalg.norm(n))
        if length == 0:
            n = normals.mean(axis=0)
            length = float(np.linalg.norm(n))
        n = n / length if length > 0 else np.array([0.0, 0.0, 1.0])
        center = (rel * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-300)
        return n, center

    def _displacement(self, variant, ids, rel, normals, w, radius, strength, c, region, view):
        brush = self.brush
        kind = brush.kind
        sign = float(brush.direction)
        if kind == "mask":
            self.sm._set_mask(ids, self.sm._mask[ids] + sign * strength * w)
            return None
        n_area, p_area = self._area(rel, normals, w, radius)
        if brush.normal == "view" and view is not None:
            n_area = -view
        if kind == "draw":
            direction = normals if brush.normal == "vertex" else n_area[None, :]
            return sign * strength * radius * w[:, None] * direction
        if kind == "inflate":
            return sign * strength * radius * w[:, None] * normals
        if kind in ("flatten", "fill", "scrape", "clay"):
            plane_n = n_area * (sign if kind == "clay" else 1.0)
            offset = 0.5 * strength * radius if kind == "clay" else 0.0
            height = (rel - p_area) @ plane_n - offset
            if kind == "fill":
                height = np.minimum(height, 0.0)
            elif kind == "scrape":
                height = np.maximum(height, 0.0)
            elif kind == "clay":
                height = 0.5 * np.minimum(height, 0.0)
                return -(w * height)[:, None] * plane_n[None, :]
            return -(min(strength, 1.0) * w * height)[:, None] * plane_n[None, :]
        if kind in ("pinch", "crease"):
            to_center = -rel
            tangential = to_center - np.outer(to_center @ n_area, n_area)
            factor = min(strength, 1.0) * (0.5 if kind == "crease" else 1.0)
            disp = factor * w[:, None] * tangential
            if kind == "crease":
                disp = disp + (sign * 0.5 * strength * radius) * w[:, None] * n_area[None, :]
            return disp
        if kind == "twist":
            angle = sign * strength * w * (math.pi / 2)
            along = rel @ n_area
            radial = rel - np.outer(along, n_area)
            cross = np.cross(n_area[None, :], radial)
            cos, sin = np.cos(angle), np.sin(angle)
            rotated = radial * cos[:, None] + cross * sin[:, None]
            return rotated - radial
        if kind == "layer":
            state = variant.state.setdefault("layer", {})
            base_p = state.setdefault("p0", {})
            base_n = state.setdefault("n0", {})
            for i, v in enumerate(ids.tolist()):
                if v not in base_p:
                    base_p[v] = self.sm._pos[v].copy()
                    base_n[v] = normals[i].copy()
            p0 = np.array([base_p[v] for v in ids.tolist()])
            n0 = np.array([base_n[v] for v in ids.tolist()])
            target = sign * strength * radius
            now = (((self.sm._pos[ids] - p0) @ self.linear.T) * n0).sum(axis=1)
            goal = np.maximum(now, w * target) if sign > 0 else np.minimum(now, w * target)
            return (goal - now)[:, None] * n0
        if kind == "grab":
            variant.state.update(ids=ids, p0=self.sm._pos[ids].copy(), w=w, c0=c.copy())
            return np.zeros((len(ids), 3))
        if kind == "snake_hook":
            previous = variant.state.get("c")
            variant.state["c"] = c.copy()
            if previous is None:
                return None
            delta = (c - previous) @ self.linear.T
            return w[:, None] * delta[None, :]
        if kind == "smooth":
            return self._smooth(ids, w, strength, region) @ self.linear.T
        raise AssertionError(kind)  # pragma: no cover

    def _grab_follow(self, variant, c):
        st = variant.state
        ids, p0, w, c0 = st["ids"], st["p0"], st["w"], st["c0"]
        target = p0 + w[:, None] * (c - c0)[None, :]
        disp_local = target - self.sm._pos[ids]
        region = self.sm.region(np.mean(np.vstack([p0, target]), axis=0), _radius_of(p0, target))
        return ids, disp_local @ self.linear.T, region

    def _elastic(self, variant, c, radius, k):
        sm = self.sm
        st = variant.state
        if "p0" not in st:
            ids = sm.vertex_ids()
            st.update(ids=ids, p0=sm._pos[ids].copy(), c0=c.copy())
            return None
        ids, p0, c0 = st["ids"], st["p0"], st["c0"]
        delta = (c - c0) @ self.linear.T
        rel = (p0 - c0) @ self.linear.T
        u = kelvinlet(rel, delta, self.brush.radius, self.brush.poisson_ratio)
        if self.brush.use_mask:
            u = u * (1.0 - sm._mask[ids].astype(np.float64))[:, None]
        target_world = rel + u
        current = (sm._pos[ids] - c0) @ self.linear.T
        return ids, target_world - current, sm.triangle_ids()

    def _automask(self, ids, rel, region):
        sm = self.sm
        keep = np.ones(len(ids))
        tri = sm._tri[region]
        if "boundary" in self.brush.automask:
            e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
            uniq, counts = np.unique(e, axis=0, return_counts=True)
            border = np.unique(uniq[counts == 1].ravel())
            keep = keep * ~np.isin(ids, border)
        if "topology" in self.brush.automask or "face_groups" in self.brush.automask:
            center_vertex = ids[int(np.argmin((rel * rel).sum(axis=1)))]
            if "topology" in self.brush.automask:
                reach = _component(tri, center_vertex)
                keep = keep * np.isin(ids, reach)
            if "face_groups" in self.brush.automask:
                groups = sm._tgroup[region]
                at_center = np.any(tri == center_vertex, axis=1)
                group = int(np.min(groups[at_center])) if at_center.any() else 0
                allowed = np.unique(tri[groups == group].ravel())
                keep = keep * np.isin(ids, allowed)
        return keep

    def _smooth(self, ids, w, strength, region):
        sm = self.sm
        tri = sm._tri[region]
        e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        uniq, counts = np.unique(e, axis=0, return_counts=True)
        boundary = counts == 1
        border = np.zeros(sm._nv, dtype=bool)
        border[uniq[boundary].ravel()] = True
        local = np.full(sm._nv, -1, dtype=np.int64)
        local[ids] = np.arange(len(ids))
        # Directed neighbour pairs (a -> b) for a among ids; a boundary vertex only
        # follows its boundary neighbours, so open borders keep their shape.
        a = np.concatenate([uniq[:, 0], uniq[:, 1]])
        b = np.concatenate([uniq[:, 1], uniq[:, 0]])
        bnd = np.concatenate([boundary, boundary])
        use = (local[a] >= 0) & np.where(border[a], bnd, True)
        rows, nbr = local[a[use]], b[use]
        num = np.bincount(rows, minlength=len(ids)).astype(np.float64)
        p = sm._pos

        def laplacian() -> np.ndarray:
            sums = np.zeros((len(ids), 3))
            for k in range(3):
                sums[:, k] = np.bincount(rows, weights=p[nbr, k], minlength=len(ids))
            out = np.zeros((len(ids), 3))
            ok = num > 0
            out[ok] = sums[ok] / num[ok, None] - p[ids[ok]]
            return out

        lam = min(strength, 1.0) * w
        method = self.brush.smooth_method
        if method == "laplacian":
            return lam[:, None] * laplacian()
        if method == "relax":
            lap = laplacian()
            n = sm.vertex_normals(ids)
            return lam[:, None] * (lap - (lap * n).sum(axis=1)[:, None] * n)
        # Taubin: a shrinking step (lambda), then an inflating one (mu = -1.04 lambda)
        # evaluated at the shrunk positions; positions are restored exactly afterwards.
        first = lam[:, None] * laplacian()
        saved = p[ids].copy()
        p[ids] = saved + first
        try:
            second = laplacian()
        finally:
            p[ids] = saved
        return first - 1.04 * lam[:, None] * second


def ord_axis(c: str) -> int:
    return "xyz".index(c)


def _radius_of(a: np.ndarray, b: np.ndarray) -> float:
    pts = np.vstack([a, b])
    center = pts.mean(axis=0)
    return float(np.sqrt(((pts - center) ** 2).sum(axis=1)).max()) + 1e-12


def _component(tri: np.ndarray, seed: int) -> np.ndarray:
    """Vertices connected to ``seed`` through the given triangles."""
    verts = np.unique(tri.ravel())
    index = {v: i for i, v in enumerate(verts.tolist())}
    parent = np.arange(len(verts))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, c in tri.tolist():
        ra, rb, rc = find(index[a]), find(index[b]), find(index[c])
        root = min(ra, rb, rc)
        parent[ra] = parent[rb] = parent[rc] = root
    roots = np.array([find(i) for i in range(len(verts))])
    return verts[roots == roots[index[seed]]]


def stroke_diagnostics(data: dict[str, Any]) -> list[Diagnostic]:
    out = []
    if data.get("skipped_dabs"):
        out.append(
            Diagnostic(
                "DABS_SKIPPED",
                INFO,
                "dabs whose ray missed the surface were skipped",
                None,
                int(data["skipped_dabs"]),
                (),
            )
        )
    if data.get("applied_dabs") == 0:
        out.append(
            Diagnostic(
                "STROKE_NO_EFFECT", INFO, "no vertex was within reach of any dab", None, 0, ()
            )
        )
    return out
