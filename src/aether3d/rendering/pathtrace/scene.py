"""What the path tracer knows about a scene: geometry for shading, materials, ray queries.

:class:`PathScene` wraps a :class:`~aether3d.rendering.scene_data.RenderScene`
with its ray tracing structure (:class:`~aether3d.rendering.raytrace.SceneAccel`)
and the world-space arrays that material evaluation reads -- exactly the
:class:`~aether3d.rendering.surface.ShadeContext` the CPU rasterizer builds,
so textures, procedural graphs, normal and bump maps, triplanar mapping and
face-group materials are evaluated by the same code for both renderers.

**Surface points.** A hit's world position and normals are interpolated
from the triangle's world-space corners with the hit's barycentrics. The
geometric normal is the triangle's (outward, corrected for mirroring
instances); the shading normal is the interpolated vertex normal (or the
geometric one for meshes without normals), then the material's normal map.
Both are turned toward the side the ray arrived from, so both faces of
double-sided surfaces shade alike, as on the rasterizers.

**Texture filtering.** Material evaluation needs the derivatives of the
barycentrics across one sample's footprint, which select mip levels and
filter procedural textures. Camera rays carry ray differentials (Igehy
1999) for one pixel step, scaled by ``1 / sqrt(samples)`` (pbrt's
convention: many samples per pixel resolve detail finer than one pixel);
the differential is transferred to the hit's plane. Later rays carry a ray
cone (Akenine-Moller et al. 2019): a width that grows with distance by a
spread angle, widened at each bounce by the sampled lobe (``2 alpha`` for
the specular lobe, one radian for the diffuse lobe). The footprint becomes
two orthogonal vectors on the surface, projected along the ray.

**Pass-through surfaces.** Rays continue through alpha-masked texels below
the material's cutoff and through the back of single-sided materials (both
invisible on the rasterizers too); camera rays also skip faces removed by
the render's ``cull`` setting. Shadow and AO rays treat such surfaces the
same way, so cut-out leaves cast leaf-shaped shadows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .. import surface
from ..raytrace import BLAS_CACHE, Hits, SceneAccel
from ..scene_data import RenderScene
from .environment import EnvironmentModel

MAX_LAYERS = 256
"""Most pass-through surfaces one ray crosses; rays still passing through after that are
treated as leaving the scene (a ``PASS_THROUGH_LIMIT`` diagnostic counts them)."""
OFFSET = 1e-9
"""Ray origins move this far (times the scale of the point) off surfaces."""


@dataclass
class Interaction:
    """Hit points of a subset of rays (``rows`` of the batch)."""

    rows: np.ndarray
    gtri: np.ndarray  # global triangle id (into the shade context)
    inst: np.ndarray
    prim: np.ndarray
    b: np.ndarray  # (m, 3) barycentrics
    t: np.ndarray  # (m,) distance along the (unit) ray
    p: np.ndarray  # (m, 3) world position
    ng: np.ndarray  # (m, 3) geometric normal, outward
    ns: np.ndarray  # (m, 3) interpolated shading normal, outward
    toward: np.ndarray  # (m,) +1 / -1 turns outward normals toward the ray's origin
    d: np.ndarray  # (m, 3) unit ray direction
    db_dx: np.ndarray
    db_dy: np.ndarray
    width: np.ndarray  # (m,) footprint width (for the next ray cone)

    @property
    def front(self) -> np.ndarray:
        return self.toward > 0.0

    def samples(self, sel=None) -> surface.Samples:
        s = surface.Samples(
            self.gtri, self.inst, self.b, self.db_dx, self.db_dy, self.ns, self.toward
        )
        return s if sel is None else s.subset(sel)


@dataclass
class Shading:
    """Material values at interactions (all ``(m, ...)``)."""

    base: np.ndarray
    metallic: np.ndarray
    roughness: np.ndarray
    specular: np.ndarray  # dielectric reflectance factor (F0 = 0.08 x specular)
    has_specular: np.ndarray  # bool
    emissive: np.ndarray
    normal: np.ndarray  # outward shading normal (normal-mapped)


class PathScene:
    """A render scene prepared for ray queries and surface shading (see the module docstring)."""

    def __init__(self, rs: RenderScene, cache=None) -> None:
        t0 = time.perf_counter()
        self.rs = rs
        self.accel = SceneAccel.build(rs, BLAS_CACHE if cache is None else cache)
        t1 = time.perf_counter()
        positions, normals, tri_vertices, tri_instance, tri_material = [], [], [], [], []
        po, uv = [], []
        offsets = np.zeros(len(rs.instances) + 1, dtype=np.int64)
        vertex_base = 0
        textured = any(m.textures for m in rs.materials)
        for k, inst in enumerate(rs.instances):
            mesh = rs.meshes[inst.mesh]
            local = mesh.positions.astype(np.float64)
            m = inst.world
            positions.append(local @ m[:3, :3].T + m[:3, 3])
            if mesh.normals is not None:
                n = mesh.normals.astype(np.float64) @ inst.normal_matrix.T
                length = np.linalg.norm(n, axis=1, keepdims=True)
                normals.append(np.divide(n, length, out=np.zeros_like(n), where=length > 1e-300))
            else:
                normals.append(np.zeros_like(local))
            if textured:
                po.append(local)
                uv.append(
                    mesh.uvs.astype(np.float64)
                    if mesh.uvs is not None
                    else np.zeros((local.shape[0], 2))
                )
            count = mesh.triangles.shape[0]
            tri_vertices.append(mesh.triangles.astype(np.int64) + vertex_base)
            tri_instance.append(np.full(count, k, dtype=np.int32))
            tri_material.append(inst.material_of_groups(mesh.groups, count))
            offsets[k + 1] = offsets[k] + count
            vertex_base += local.shape[0]
        self.P = np.concatenate(positions) if positions else np.zeros((0, 3))
        self.N = np.concatenate(normals) if normals else np.zeros((0, 3))
        self.TRI = np.concatenate(tri_vertices) if tri_vertices else np.zeros((0, 3), np.int64)
        self.INST = np.concatenate(tri_instance) if tri_instance else np.zeros(0, np.int32)
        self.MAT = np.concatenate(tri_material) if tri_material else np.zeros(0, np.int32)
        self.tri_offset = offsets
        self.has_normals = np.array(
            [rs.meshes[i.mesh].normals is not None for i in rs.instances], dtype=bool
        )
        self.mirrored = np.array([i.mirrored for i in rs.instances], dtype=bool)
        frame = rs.camera.frame
        self.ctx = surface.ShadeContext(
            rs,
            self.P,
            np.concatenate(po) if po else np.zeros((0, 3)),
            np.concatenate(uv) if uv else np.zeros((0, 2)),
            self.TRI,
            self.INST,
            self.MAT,
            np.array([rs.meshes[i.mesh].uvs is not None for i in rs.instances], dtype=bool),
            np.array([i.world[:3, :3] for i in rs.instances], dtype=np.float64).reshape(-1, 3, 3),
            frame.position,
            frame.forward,
            rs.camera.camera.is_perspective,
            0.0,
            0.0,
        )
        mats = rs.materials
        self.double_sided = np.array([m.double_sided for m in mats], dtype=bool)
        self.masked = np.array([m.masked for m in mats], dtype=bool)
        self.pass_through = bool((~self.double_sided).any() or self.masked.any())
        self.environment = EnvironmentModel.build(rs.environment, rs.background, rs.ambient)
        self.timings = {"bvh": t1 - t0, "geometry": time.perf_counter() - t1}
        self.pass_through_rays = 0  # extra ray segments behind transparent hits
        self.layer_limited = 0  # rays stopped by MAX_LAYERS

    # ------------------------------------------------------------------ geometry --

    @property
    def triangle_count(self) -> int:
        return int(self.TRI.shape[0])

    def corners(self, gtri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tv = self.TRI[gtri]
        return self.P[tv[:, 0]], self.P[tv[:, 1]], self.P[tv[:, 2]]

    def geometric_normal(self, gtri: np.ndarray, inst: np.ndarray):
        """Outward unit geometric normals and triangle areas (world space)."""
        p0, p1, p2 = self.corners(gtri)
        g = np.cross(p1 - p0, p2 - p0)
        length = np.linalg.norm(g, axis=1)
        g = np.divide(g, length[:, None], out=np.zeros_like(g), where=length[:, None] > 0.0)
        g = np.where(self.mirrored[inst][:, None], -g, g)
        return g, 0.5 * length

    def interaction(
        self,
        rows: np.ndarray,
        hits: Hits,
        o: np.ndarray,
        d: np.ndarray,
        diff: tuple | None = None,
        cone: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> Interaction:
        """Surface data of the hit rays ``rows`` (``hits``, ``o``, ``d`` indexed by ``rows``).

        ``diff``: camera ray differentials ``(do_dx, do_dy, dd_dx, dd_dy)``
        (per row or broadcastable) in the units of ``d``; ``cone``:
        ``(width, spread)`` of the arriving ray cones.
        """
        inst = hits.inst
        gtri = self.tri_offset[inst] + hits.prim
        b = hits.b
        p0, p1, p2 = self.corners(gtri)
        p = b[:, 0:1] * p0 + b[:, 1:2] * p1 + b[:, 2:3] * p2
        e1, e2 = p1 - p0, p2 - p0
        g = np.cross(e1, e2)
        length = np.linalg.norm(g, axis=1, keepdims=True)
        ng = np.divide(g, length, out=np.zeros_like(g), where=length > 0.0)
        ng = np.where(self.mirrored[inst][:, None], -ng, ng)
        tv = self.TRI[gtri]
        ns = (
            b[:, 0:1] * self.N[tv[:, 0]]
            + b[:, 1:2] * self.N[tv[:, 1]]
            + b[:, 2:3] * self.N[tv[:, 2]]
        )
        n_len = np.linalg.norm(ns, axis=1, keepdims=True)
        use = self.has_normals[inst] & (n_len[:, 0] > 1e-6)
        ns = np.where(use[:, None], ns / np.maximum(n_len, 1e-300), ng)
        toward = np.where((ng * d).sum(axis=1) < 0.0, 1.0, -1.0)
        t = hits.t
        if diff is not None:
            do_dx, do_dy, dd_dx, dd_dy = diff
            dp = []
            for do, dd in ((do_dx, dd_dx), (do_dy, dd_dy)):
                v = do + t[:, None] * dd
                denom = (d * ng).sum(axis=1)
                safe = np.where(np.abs(denom) > 1e-300, denom, 1.0)
                dt = -(v * ng).sum(axis=1) / safe
                dp.append(v + dt[:, None] * d)
            dp_dx, dp_dy = dp
            width = np.maximum(np.linalg.norm(dp_dx, axis=1), np.linalg.norm(dp_dy, axis=1))
        else:
            w, spread = cone if cone is not None else (np.zeros(t.shape), np.zeros(t.shape))
            width = w + spread * t
            dp_dx, dp_dy = _cone_footprint(d, ng, width)
        db_dx = _bary_derivative(e1, e2, dp_dx)
        db_dy = _bary_derivative(e1, e2, dp_dy)
        return Interaction(
            rows, gtri, inst, hits.prim, b, t, p, ng, ns, toward, d, db_dx, db_dy, width
        )

    def offset_origin(self, p: np.ndarray, ng_side: np.ndarray) -> np.ndarray:
        """``p`` moved off its surface along ``ng_side`` (a normal on the leaving side)."""
        scale = OFFSET * (1.0 + np.abs(p).max(axis=1, keepdims=True))
        return p + ng_side * scale

    # ----------------------------------------------------------------- materials --

    def shade(self, it: Interaction, sel: np.ndarray | None = None) -> Shading:
        """Material values at the interactions (``sel``: a subset, else all)."""
        s = it.samples(sel)
        mat = self.MAT[s.tri]
        m = s.tri.shape[0]
        out = Shading(
            np.zeros((m, 3)),
            np.zeros(m),
            np.ones(m),
            np.zeros(m),
            np.zeros(m, dtype=bool),
            np.zeros((m, 3)),
            s.normal.copy(),
        )
        for index in np.unique(mat):
            rows = np.flatnonzero(mat == index)
            material = self.rs.materials[int(index)]
            sv = surface.evaluate_material(self.ctx, s.subset(rows), material)
            out.base[rows] = sv.base
            out.emissive[rows] = sv.emissive
            if "normal" in material.textures:
                out.normal[rows] = sv.normal
            if material.model == "simple":
                out.metallic[rows] = 0.0
                if material.specular > 0.0:
                    out.has_specular[rows] = True
                    out.specular[rows] = material.specular
                    out.roughness[rows] = (2.0 / (material.shininess + 2.0)) ** 0.25
            else:
                out.metallic[rows] = sv.metallic
                out.roughness[rows] = sv.roughness
                out.specular[rows] = material.specular
                out.has_specular[rows] = True
        return out

    def transparent(self, it: Interaction, camera_cull: str = "none") -> np.ndarray:
        """Whether rays pass through these hits (cut-out texels, backs of single-sided
        materials, culled faces for camera rays)."""
        mat = self.MAT[it.gtri]
        through = ~it.front & ~self.double_sided[mat]
        if camera_cull == "back":
            through |= ~it.front
        elif camera_cull == "front":
            through |= it.front
        masked = self.masked[mat] & ~through
        if masked.any():
            rows = np.flatnonzero(masked)
            s = it.samples(rows)
            sub_mat = mat[rows]
            for index in np.unique(sub_mat):
                pick = np.flatnonzero(sub_mat == index)
                material = self.rs.materials[int(index)]
                sv = surface.evaluate_material(self.ctx, s.subset(pick), material, alpha_only=True)
                through[rows[pick]] = sv.alpha < material.alpha_cutoff
        return through

    # ---------------------------------------------------------------- ray queries --

    def trace(
        self,
        o: np.ndarray,
        d: np.ndarray,
        tmin: np.ndarray | float = 0.0,
        tmax: np.ndarray | float = np.inf,
        skip_inst: np.ndarray | None = None,
        skip_tri: np.ndarray | None = None,
        cull: str = "none",
        diff: tuple | None = None,
        cone: tuple[np.ndarray, np.ndarray] | None = None,
        counter: dict | None = None,
    ) -> tuple[Hits, Interaction | None]:
        """Nearest surface the rays stop at (passing through transparent hits).

        Returns the hits (``t`` from the original origins) and the
        interaction of the rays that hit something.
        """
        n = o.shape[0]
        tmin = np.broadcast_to(np.asarray(tmin, dtype=np.float64), (n,)).copy()
        tmax = np.broadcast_to(np.asarray(tmax, dtype=np.float64), (n,)).copy()
        out = Hits(
            np.full(n, np.inf), np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.zeros((n, 3))
        )
        if skip_inst is None:
            skip_inst = np.full(n, -1, np.int64)
            skip_tri = np.full(n, -1, np.int64)
        else:
            skip_inst = skip_inst.copy()
            skip_tri = skip_tri.copy()
        origin = o.copy()
        travelled = np.zeros(n)
        pending = np.arange(n)
        result_it: list[Interaction] = []
        filtering = self.pass_through or cull != "none"
        for _layer in range(MAX_LAYERS):
            if not pending.size:
                break
            h = self.accel.closest_hit(
                origin[pending],
                d[pending],
                tmin[pending],
                tmax[pending],
                skip_inst[pending],
                skip_tri[pending],
            )
            if counter is not None:
                counter["rays"] = counter.get("rays", 0) + int(pending.size)
            self.pass_through_rays += int(pending.size) if _layer else 0
            hit = np.flatnonzero(h.hit)
            if not hit.size:
                break
            rows = pending[hit]
            sub = Hits(h.t[hit] + travelled[rows], h.inst[hit], h.prim[hit], h.b[hit])
            sub_diff = None
            if diff is not None:
                sub_diff = tuple(x if np.ndim(x) == 1 else x[rows] for x in diff)
            sub_cone = None if cone is None else (cone[0][rows], cone[1][rows])
            it = self.interaction(rows, sub, o[rows], d[rows], sub_diff, sub_cone)
            through = self.transparent(it, cull) if filtering else np.zeros(rows.size, bool)
            keep = ~through
            if keep.any():
                k = np.flatnonzero(keep)
                r = rows[k]
                out.t[r] = sub.t[k]
                out.inst[r] = sub.inst[k]
                out.prim[r] = sub.prim[k]
                out.b[r] = sub.b[k]
                result_it.append(_subset(it, k))
            if not through.any():
                break
            k = np.flatnonzero(through)
            r = rows[k]
            # Continue from the transparent hit, past its triangle.
            step = h.t[hit[k]]
            origin[r] = origin[r] + d[r] * step[:, None]
            travelled[r] += step
            tmin[r] = 0.0
            tmax[r] = tmax[r] - step
            skip_inst[r] = sub.inst[k]
            skip_tri[r] = sub.prim[k]
            pending = r
        else:  # pragma: no cover - needs more than MAX_LAYERS stacked cut-outs
            # Rays still passing through after MAX_LAYERS surfaces count as leaving the
            # scene (they most likely would); the render reports how many there were.
            self.layer_limited += int(pending.size)
        return out, _concat(result_it)

    def visible(
        self,
        o: np.ndarray,
        d: np.ndarray,
        tmax: np.ndarray,
        skip_inst: np.ndarray,
        skip_tri: np.ndarray,
        counter: dict | None = None,
    ) -> np.ndarray:
        """Whether nothing opaque lies between ``o`` and ``o + tmax d`` (shadow/AO rays)."""
        if counter is not None:
            counter["rays"] = counter.get("rays", 0) + int(o.shape[0])
        if not self.pass_through:
            return ~self.accel.occluded(o, d, 0.0, tmax, skip_inst, skip_tri)
        hits, _ = self.trace(o, d, 0.0, tmax, skip_inst, skip_tri)
        return ~hits.hit


def _cone_footprint(d: np.ndarray, ng: np.ndarray, width: np.ndarray):
    """Two footprint vectors of length ``width`` across the ray, projected on the plane."""
    t = np.cross(d, np.where(np.abs(d[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]]))
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-300)
    b = np.cross(d, t)
    denom = (d * ng).sum(axis=1)
    safe = np.where(np.abs(denom) > 1e-6, denom, np.where(denom < 0.0, -1e-6, 1e-6))
    out = []
    for v in (t, b):
        v = v * width[:, None]
        out.append(v - d * ((v * ng).sum(axis=1) / safe)[:, None])
    return out[0], out[1]


def _bary_derivative(e1: np.ndarray, e2: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Barycentric change ``(db0, db1, db2)`` for an in-plane offset ``dp``."""
    a = (e1 * e1).sum(axis=1)
    b = (e1 * e2).sum(axis=1)
    c = (e2 * e2).sum(axis=1)
    det = a * c - b * b
    safe = np.where(np.abs(det) > 1e-300, det, 1.0)
    p1 = (e1 * dp).sum(axis=1)
    p2 = (e2 * dp).sum(axis=1)
    db1 = np.where(np.abs(det) > 1e-300, (c * p1 - b * p2) / safe, 0.0)
    db2 = np.where(np.abs(det) > 1e-300, (a * p2 - b * p1) / safe, 0.0)
    return np.stack([-db1 - db2, db1, db2], axis=1)


def _subset(it: Interaction, k: np.ndarray) -> Interaction:
    return Interaction(
        it.rows[k],
        it.gtri[k],
        it.inst[k],
        it.prim[k],
        it.b[k],
        it.t[k],
        it.p[k],
        it.ng[k],
        it.ns[k],
        it.toward[k],
        it.d[k],
        it.db_dx[k],
        it.db_dy[k],
        it.width[k],
    )


def _concat(parts: list[Interaction]) -> Interaction | None:
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    names = [
        "rows",
        "gtri",
        "inst",
        "prim",
        "b",
        "t",
        "p",
        "ng",
        "ns",
        "toward",
        "d",
        "db_dx",
        "db_dy",
        "width",
    ]
    merged = Interaction(*(np.concatenate([getattr(p, f) for p in parts]) for f in names))
    order = np.argsort(merged.rows, kind="stable")
    return _subset(merged, order)
