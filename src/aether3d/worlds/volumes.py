"""Volumetric terrain near the camera: caves, arches, holes and overhangs.

A heightfield cannot hold a cave or an overhang, so near the camera
(``streaming.volume_distance``) a terrain world is also a signed distance field,
negative in rock::

    phi(x, y, z) = max(z - h(x + wx, y + wy), -a(x, y, z))

where ``h`` is the authoritative ground height, ``(wx, wy)`` a 3D noise warp that only acts
on steep slopes (``caves.overhangs``: cliffs lean out over their foot) and ``a`` the cave
field, negative in cave air:

* noise caves (``caves.enabled``): tunnels, where two noise fields are both near zero
  (``caves.tunnels``, ``caves.radius``), and caverns, where a fractal noise exceeds a
  threshold (``caves.caverns``), sampled every :data:`LATTICE` metres on a global lattice
  and interpolated trilinearly. They stay between ``caves.depth[0]`` and ``caves.depth[1]``
  metres below the ground (default 80 to 4), except around hashed entrance points
  (``caves.entrances``: the probability per :data:`ENTRANCE_CELL`-metre cell) where they
  may break through -- never under water;
* cave features, exact: ``tunnel`` and ``arch`` (a 3D polyline ``[x, y, z]`` with a radius
  going to ``radius_end``), ``cavern`` (an ellipsoid under a circle, its top about half a
  radius below the ground) and ``hole`` (a shaft down from the ground into a cavern);
  ``noise`` roughens their walls.

The field is sampled every :data:`VOXEL` metres in cubes of :data:`CHUNK` metres (the
same global grid for every cube, so neighbours meet exactly) and meshed with marching
cubes. A cube is *active* if cave air (or an overhang) may lie in it; a *surface* cube if
the air comes within :data:`SURFACE_MARGIN` metres of the ground or it holds an overhang.
Surface cubes draw the ground of their footprint themselves: the terrain tiles leave that
footprint out (refined to level 0, whose vertices are the volume's: on vertical cube
edges the field is linear in ``z``, so the marching-cubes vertex lies exactly on ``h``) and
hang skirts along the hole's edge. Overhang warps fade to nothing toward footprint edges
that border the heightfield, so both surfaces agree there. Ground near the surface takes
the terrain's colours; deeper walls are the biomes' rock, darker and damper with depth.

Closed caves (active cubes that are not surface cubes) are drawn within
:data:`UNDERGROUND_DISTANCE` of the camera; surface cubes within ``volume_distance``.
Beyond, the heightfield is closed: entrances and overhangs farther than
``volume_distance`` are not drawn (a documented level of detail; queries such as
:meth:`Volumes.solid` always answer from the exact field).
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ..mesh import Mesh
from . import noise as nz

CHUNK = 32.0
"""Metres per side of a volume cube."""
VOXEL = 1.0
"""Metres between samples of the distance field."""
LATTICE = 4.0
"""Metres between the samples of the cave noise (interpolated trilinearly)."""
SURFACE_MARGIN = 3.0
"""Cave air this close below the ground makes a cube a surface cube."""
ENTRANCE_CELL = 96.0
"""Metres per side of the cells that may hold one cave entrance."""
ENTRANCE_RADIUS = 36.0
"""Metres around an entrance point where caves may reach the surface."""
UNDERGROUND_DISTANCE = 160.0
"""Closed caves are drawn within this distance of a camera below the ground."""
OVERHANG_DISTANCE = 160.0
"""Overhangs are drawn within this distance of the camera (beyond, the heightfield)."""
CAVE_SURFACE = 1
OVERHANG = 2
OVERHANG_AMPLITUDE = 6.0
"""Metres the overhang warp moves the ground at ``caves.overhangs = 1`` on cliffs."""
TAPER = 8.0
"""Metres over which overhang warps fade toward footprint edges bordering the heightfield."""
MAX_CUBES = 4000
"""Cube meshes kept in memory (least recently used dropped first)."""
_N = int(round(CHUNK / VOXEL))  # voxels per cube side


def _features(model: Any) -> list[dict]:
    return [c.spec for c in model.final_features if c.spec["type"] == "cave"]


class Volumes:
    """The volumetric terrain of a world (per runtime; see the module docstring)."""

    def __init__(self, rt: Any) -> None:
        self.rt = rt
        self._columns: OrderedDict[tuple, list] = OrderedDict()
        self._cubes: OrderedDict[tuple, Mesh | None] = OrderedDict()
        self._anchors: dict[tuple, float] = {}
        self._lock = threading.Lock()
        self.counters = {"columns": 0, "cubes_meshed": 0, "triangles": 0, "seconds": 0.0}

    # -- settings --------------------------------------------------------------------

    def enabled(self) -> bool:
        d = self.rt.definition
        if d["mode"] != "terrain" or d["streaming"]["volume_distance"] <= 0.0:
            return False
        cv = d["caves"]
        noise = cv["enabled"] and (cv["tunnels"] > 0.0 or cv["caverns"] > 0.0)
        return noise or cv["overhangs"] > 0.0 or bool(_features(self.rt.model()))

    def _key(self) -> str:
        return self.rt.model().key

    # -- the field -------------------------------------------------------------------

    def _noise_nodes(self, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        """The noise cave field (metres, negative in air) at lattice nodes, before the depth
        band: ``min(tunnel, cavern)``."""
        d = self.rt.definition
        cv = d["caves"]
        seed = int(d["seed"])
        x, y, z = ix * LATTICE, iy * LATTICE, iz * LATTICE
        sc = cv["scale"]
        out = np.full(x.shape, np.inf)
        if cv["tunnels"] > 0.0:
            fx, fy, fz = x / (sc * 0.8), y / (sc * 0.8), z / (sc * 0.5)
            t1 = nz.gradient3(nz.derive(seed, "caves", "tunnel-a"), fx, fy, fz)
            t2 = nz.gradient3(nz.derive(seed, "caves", "tunnel-b"), fx, fy, fz)
            radius = cv["radius"] * math.sqrt(cv["tunnels"] / 0.4)
            out = np.minimum(out, np.maximum(np.abs(t1), np.abs(t2)) * (sc * 0.4) - radius)
        if cv["caverns"] > 0.0:
            n = nz.fbm3(
                nz.derive(seed, "caves", "cavern"), x / sc, y / sc, z / (sc * 0.55), octaves=3
            )
            out = np.minimum(out, ((0.5 - 0.25 * cv["caverns"]) - n) * (sc * 0.3))
        return out

    def _warp(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The overhang warp directions at points (about [-1, 1]), trilinear from the
        lattice like the cave noise."""
        seed = int(self.rt.definition["seed"])

        def nodes(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
            px, py, pz = ix * LATTICE / 14.0, iy * LATTICE / 14.0, iz * LATTICE / 9.0
            a = nz.gradient3(nz.derive(seed, "caves", "warp-x"), px, py, pz)
            b = nz.gradient3(nz.derive(seed, "caves", "warp-y"), px, py, pz)
            return np.stack([a, b], axis=-1)

        w = _trilinear(nodes, x, y, z)
        return w[:, 0], w[:, 1]

    def _noise(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """The noise cave field at points, trilinear between lattice nodes (each node a
        function of its coordinates only, so every caller gets the same values)."""
        return _trilinear(self._noise_nodes, x, y, z)

    def _band(self, x: np.ndarray, y: np.ndarray, h: np.ndarray, wet: np.ndarray) -> np.ndarray:
        """The highest the noise caves may reach at columns (metres): ``depth[1]`` below the
        ground, up to 2 m above it near entrances on dry ground."""
        d = self.rt.definition
        cv = d["caves"]
        top = h + cv["depth"][1]
        if cv["entrances"] <= 0.0:
            return top
        seed = int(d["seed"])
        out = top.copy()
        ci0 = np.floor(x / ENTRANCE_CELL).astype(np.int64)
        cj0 = np.floor(y / ENTRANCE_CELL).astype(np.int64)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ci, cj = ci0 + di, cj0 + dj
                has = nz.random2(nz.derive(seed, "caves", "entrance"), ci, cj) < cv["entrances"]
                ex = ci + 0.2 + 0.6 * nz.random2(nz.derive(seed, "caves", "ex"), ci, cj)
                ey = cj + 0.2 + 0.6 * nz.random2(nz.derive(seed, "caves", "ey"), ci, cj)
                dist = np.hypot(x - ex * ENTRANCE_CELL, y - ey * ENTRANCE_CELL)
                f = nz.smoothstep(ENTRANCE_RADIUS, ENTRANCE_RADIUS / 3.0, dist) * has
                out = np.maximum(out, top + f * (h + 2.0 - top))
        return np.where(wet, top, out)

    def _anchor(self, fid: str, x: float, y: float) -> float:
        key = (self._key(), fid)
        found = self._anchors.get(key)
        if found is None:
            found = self._anchors[key] = float(
                self.rt.model().height(np.array([x]), np.array([y]))[0]
            )
        return found

    def _feature_field(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Exact cave features (metres, negative in air)."""
        out = np.full(x.shape, np.inf)
        seed = int(self.rt.definition["seed"])
        for f in _features(self.rt.model()):
            kind, r0 = f["kind"], f["radius"]
            r1 = f["radius_end"] if f["radius_end"] is not None else r0
            shape = f["shape"]
            if "line" in shape:
                pts = np.asarray(shape["line"], dtype=np.float64)
                dist, t = _polyline_distance(np.stack([x, y, z], axis=1), pts)
                field = dist - (r0 + (r1 - r0) * t)
                rad = np.maximum(r0, r1)
            else:
                cx, cy, r = shape["circle"]
                ground = self._anchor(f["id"], cx, cy)
                if kind == "hole":
                    shaft = np.hypot(x - cx, y - cy) - r0
                    bottom = ground - 4.0 * r
                    shaft = np.maximum(shaft, np.maximum(bottom - z, z - ground - 5.0))
                    bulb = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - bottom) ** 2) - 1.6 * r
                    field = np.minimum(shaft, bulb)
                else:  # cavern: an ellipsoid below the ground
                    zc = ground - 0.5 * r - 0.6 * r
                    q = np.sqrt(
                        ((x - cx) / r) ** 2 + ((y - cy) / r) ** 2 + ((z - zc) / (0.6 * r)) ** 2
                    )
                    field = (q - 1.0) * 0.6 * r
                rad = r
            if f["noise"] > 0.0:
                s = nz.derive(seed, "caves", "feature", f["id"])
                k = 1.0 / max(rad, 0.5)
                field = field + f["noise"] * 0.35 * rad * nz.gradient3(s, x * k, y * k, z * k)
            out = np.minimum(out, field)
        return out

    def cave_field(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray, h: np.ndarray, wet: np.ndarray
    ) -> np.ndarray:
        """The cave field at points (metres, negative in cave air); ``h`` and ``wet`` are the
        ground height and whether there is water, at the points' columns."""
        d = self.rt.definition
        cv = d["caves"]
        out = np.full(x.shape, np.inf)
        if cv["enabled"] and (cv["tunnels"] > 0.0 or cv["caverns"] > 0.0):
            noise = self._noise(x, y, z)
            band = np.maximum(z - self._band(x, y, h, wet), (h + cv["depth"][0]) - z)
            out = np.maximum(noise, band)
        if _features(self.rt.model()):
            out = np.minimum(out, self._feature_field(x, y, z))
        return out

    def solid(self, points: Any) -> np.ndarray:
        """Whether points ``[[x, y, z], ...]`` are in rock (exact field; overhangs are not
        counted -- they are a level of detail of the drawn surface)."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        s = self.rt.model().evaluate(p[:, 0], p[:, 1])
        h = s["height"]
        wet = ~np.isnan(s["water"])
        a = self.cave_field(p[:, 0], p[:, 1], p[:, 2], h, wet)
        return (p[:, 2] < h) & ~(a < 0.0)

    # -- which cubes -----------------------------------------------------------------

    def column(self, ci: int, cj: int) -> list[tuple[int, int]]:
        """``(ck, flags)`` of the active cubes above footprint ``(ci, cj)`` (cached):
        :data:`CAVE_SURFACE` if cave air reaches the ground there, :data:`OVERHANG` if it
        holds a cliff steep enough to lean out, 0 for closed caves."""
        key = (self._key(), ci, cj)
        with self._lock:
            found = self._columns.get(key)
            if found is not None:
                self._columns.move_to_end(key)
                return found
        out = self._classify(ci, cj)
        with self._lock:
            self._columns[key] = out
            while len(self._columns) > 50_000:
                self._columns.popitem(last=False)
            self.counters["columns"] += 1
        return out

    def _classify(self, ci: int, cj: int) -> list[tuple[int, int]]:
        d = self.rt.definition
        cv = d["caves"]
        model = self.rt.model()
        x0, y0 = ci * CHUNK, cj * CHUNK
        m = int(CHUNK / LATTICE) + 1
        g = np.arange(m) * LATTICE
        X, Y = np.meshgrid(x0 + g, y0 + g)
        X, Y = X.reshape(-1), Y.reshape(-1)
        s = model.evaluate(X, Y)
        h = s["height"]
        wet = ~np.isnan(s["water"])
        found: dict[int, int] = {}
        # Overhangs: cliffs (steeper than about 60 degrees) in the footprint.
        if cv["overhangs"] > 0.0:
            H = h.reshape(m, m)
            gx = np.abs(np.diff(H, axis=1)).max(initial=0.0) / LATTICE
            gy = np.abs(np.diff(H, axis=0)).max(initial=0.0) / LATTICE
            if max(gx, gy) > 1.7:
                for ck in range(
                    math.floor((h.min() - OVERHANG_AMPLITUDE) / CHUNK),
                    math.floor((h.max() + OVERHANG_AMPLITUDE) / CHUNK) + 1,
                ):
                    found[ck] = found.get(ck, 0) | OVERHANG
        noise = cv["enabled"] and (cv["tunnels"] > 0.0 or cv["caverns"] > 0.0)
        lo_z = float((h + cv["depth"][0]).min()) if noise else np.inf
        hi_z = float(self._band(X, Y, h, wet).max()) + 2.0 if noise else -np.inf
        for f in _features(model):
            fb = _feature_box(self, f)
            if fb[0] <= x0 + CHUNK and fb[2] >= x0 and fb[1] <= y0 + CHUNK and fb[3] >= y0:
                lo_z, hi_z = min(lo_z, fb[4]), max(hi_z, fb[5])
        if lo_z > hi_z:
            return sorted(found.items())
        zs = np.arange(math.floor(lo_z / LATTICE), math.floor(hi_z / LATTICE) + 2) * LATTICE
        PX = np.repeat(X, zs.size)
        PY = np.repeat(Y, zs.size)
        PZ = np.tile(zs, X.size)
        hz = np.repeat(h, zs.size)
        a = self.cave_field(PX, PY, PZ, hz, np.repeat(wet, zs.size))
        # Conservative for activity: a node closer than a lattice step to air may have air
        # near it, in the cubes within a lattice step of it. Exact for the surface: air at
        # a node within a metre of the ground.
        air = (a < LATTICE) & (hz + LATTICE > PZ)
        at_ground = (a < 0.0) & (np.abs(PZ - hz) <= LATTICE * 0.5 + 1.0)
        for z, flag in ((PZ[air], 0), (PZ[at_ground], CAVE_SURFACE)):
            for lo_k, hi_k in zip(
                np.floor((z - LATTICE) / CHUNK).astype(np.int64).tolist(),
                np.floor((z + LATTICE) / CHUNK).astype(np.int64).tolist(),
                strict=True,
            ):
                for ck in range(lo_k, hi_k + 1):
                    found[ck] = found.get(ck, 0) | flag
        return sorted(found.items())

    def drawn(self, view: Any) -> tuple[list[tuple[int, int, int, int]], set[tuple[int, int]]]:
        """The cubes to draw from a view ``(ci, cj, ck, flags)`` and the footprints the
        terrain tiles leave out: cave entrances within ``volume_distance``, overhangs within
        :data:`OVERHANG_DISTANCE`; closed caves within :data:`UNDERGROUND_DISTANCE` of a
        camera below the ground, or under and beside a drawn entrance."""
        if not self.enabled():
            return [], set()
        d = self.rt.definition
        eye = np.asarray(view.eye, dtype=np.float64)
        reach = min(d["streaming"]["volume_distance"], view.distance)
        ground = float(self.rt.model().height(eye[:1], eye[1:2])[0])
        below = eye[2] < ground
        e = self.rt.model().extent()
        surface: dict[tuple[int, int, int], int] = {}
        closed: list[tuple[int, int, int]] = []
        i0, i1 = math.floor((eye[0] - reach) / CHUNK), math.floor((eye[0] + reach) / CHUNK)
        j0, j1 = math.floor((eye[1] - reach) / CHUNK), math.floor((eye[1] + reach) / CHUNK)
        for cj in range(j0, j1 + 1):
            for ci in range(i0, i1 + 1):
                bx, by = ci * CHUNK, cj * CHUNK
                if e is not None and (
                    bx + CHUNK <= e[0] or bx >= e[2] or by + CHUNK <= e[1] or by >= e[3]
                ):
                    continue
                cx = min(max(eye[0], bx), bx + CHUNK)
                cy = min(max(eye[1], by), by + CHUNK)
                flat = math.hypot(cx - eye[0], cy - eye[1])
                if flat > reach:
                    continue
                for ck, flags in self.column(ci, cj):
                    cz = min(max(eye[2], ck * CHUNK), (ck + 1) * CHUNK)
                    dist = math.hypot(flat, cz - eye[2])
                    if flags & OVERHANG and dist > OVERHANG_DISTANCE:
                        flags &= ~OVERHANG
                    if _cube_outside(view, bx, by, ck * CHUNK):
                        continue
                    if flags:
                        surface[(ci, cj, ck)] = flags
                    elif below and dist <= UNDERGROUND_DISTANCE or not below:
                        closed.append((ci, cj, ck))
        out = [(*key, flags) for key, flags in surface.items()]
        # Seen from above, closed caves only show through entrances: the cubes under an
        # entrance and beside it (at its level and one below).
        near = set()
        for (ci, cj, ck), flags in surface.items():
            if flags & CAVE_SURFACE:
                near.update({(ci, cj, ck - 1), (ci, cj, ck - 2)})
                for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    near.update({(ci + a, cj + b, ck), (ci + a, cj + b, ck - 1)})
        for key in closed:
            if below or key in near:
                out.append((*key, 0))
        holes = {(ci, cj) for ci, cj, _ in surface}
        return sorted(out), holes

    # -- meshes ----------------------------------------------------------------------

    def cube(self, ci: int, cj: int, ck: int, open_sides: int) -> Mesh | None:
        """The mesh of cube ``(ci, cj, ck)`` in cube-local metres; ``open_sides`` (bits
        -x, +x, -y, +y) are the footprint edges bordering other drawn surface cubes, where
        overhang warps do not fade."""
        key = (self._key(), ci, cj, ck, open_sides)
        with self._lock:
            if key in self._cubes:
                self._cubes.move_to_end(key)
                return self._cubes[key]
        import time

        t0 = time.perf_counter()
        mesh = self._mesh(ci, cj, ck, open_sides)
        with self._lock:
            self._cubes[key] = mesh
            while len(self._cubes) > MAX_CUBES:
                self._cubes.popitem(last=False)
            self.counters["cubes_meshed"] += 1
            self.counters["triangles"] += 0 if mesh is None else mesh.triangle_count
            self.counters["seconds"] += time.perf_counter() - t0
        return mesh

    def _mesh(self, ci: int, cj: int, ck: int, open_sides: int) -> Mesh | None:
        from .surface import colors as surface_colors

        d = self.rt.definition
        cv = d["caves"]
        model = self.rt.model()
        x0, y0, z0 = ci * CHUNK, cj * CHUNK, ck * CHUNK
        # Columns with a margin wide enough for the warp (bilinear heights in between).
        pad = int(math.ceil(OVERHANG_AMPLITUDE * cv["overhangs"])) + 2
        g = np.arange(-pad, _N + pad + 1) * VOXEL
        CX, CY = np.meshgrid(x0 + g, y0 + g)
        s = model.evaluate(CX.reshape(-1), CY.reshape(-1))
        nc = g.size
        H = s["height"].reshape(nc, nc)
        WET = (~np.isnan(s["water"])).reshape(nc, nc)
        # Samples: the cube's grid plus one sample around (normals, closing the surface).
        k = np.arange(-1, _N + 2)
        lx = k * VOXEL
        Z, Y, X = np.meshgrid(z0 + lx, y0 + lx, x0 + lx, indexing="ij")
        ii = (k + pad)[None, None, :]
        jj = (k + pad)[None, :, None]
        h = np.broadcast_to(H[jj, ii], X.shape)
        wet = np.broadcast_to(WET[jj, ii], X.shape)
        z = Z
        phi_t = z - h
        if cv["overhangs"] > 0.0:
            gxh = (H[1:-1, 2:] - H[1:-1, :-2]) / (2 * VOXEL)
            gyh = (H[2:, 1:-1] - H[:-2, 1:-1]) / (2 * VOXEL)
            steep = np.zeros_like(H)
            steep[1:-1, 1:-1] = nz.smoothstep(0.9, 2.2, np.sqrt(gxh * gxh + gyh * gyh))
            taper = _taper(lx, open_sides)  # (n, n) over [y, x]
            amp = OVERHANG_AMPLITUDE * cv["overhangs"] * steep[jj, ii][0] * taper
            wx, wy = self._warp(X.reshape(-1), Y.reshape(-1), Z.reshape(-1))
            wx, wy = wx.reshape(X.shape), wy.reshape(X.shape)
            hw = _bilinear(
                H, (X - x0) / VOXEL + pad + wx * amp[None], (Y - y0) / VOXEL + pad + wy * amp[None]
            )
            phi_t = z - hw
        a = self.cave_field(
            X.reshape(-1), Y.reshape(-1), Z.reshape(-1), h.reshape(-1), wet.reshape(-1)
        ).reshape(X.shape)
        phi = np.maximum(phi_t, -a)
        if not (np.any(phi < 0) and np.any(phi > 0)):
            return None
        origin = np.array([-VOXEL, -VOXEL, z0 - VOXEL])
        pos, tris = _marching_cubes(phi, origin)
        if not len(tris):
            return None
        # Keep the triangles inside the cube (the extra samples only gave neighbours).
        cen = pos[tris].mean(axis=1)
        inside = (
            (cen[:, 0] >= 0)
            & (cen[:, 0] <= CHUNK)
            & (cen[:, 1] >= 0)
            & (cen[:, 1] <= CHUNK)
            & (cen[:, 2] >= z0)
            & (cen[:, 2] <= z0 + CHUNK)
        )
        tris = tris[inside]
        if not len(tris):
            return None
        used, tris = np.unique(tris, return_inverse=True)
        tris = tris.reshape(-1, 3)
        pos = pos[used]
        # Normals: the field's gradient (trilinear central differences), toward the air.
        grad = _gradient(phi, pos, origin)
        nrm = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
        # Outward winding.
        fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
        if np.sum(fn * nrm[tris].mean(axis=1)) < 0:
            tris = tris[:, [0, 2, 1]]
        # Colours: the terrain's surface colours near the ground, rock below.
        wx_, wy_ = pos[:, 0] + x0, pos[:, 1] + y0
        smp = model.evaluate(wx_, wy_)
        depth = smp["height"] - pos[:, 2]
        surf = surface_colors(model, smp, wx_, wy_, nrm)
        rock = smp["weights"] @ model.colors["rock"]
        tone = 0.75 + 0.25 * nz.fbm2(
            nz.derive(int(d["seed"]), "caves", "tone"),
            wx_ * 0.3,
            wy_ * 0.3 + pos[:, 2] * 0.5,
            octaves=2,
        )
        damp = np.clip(1.0 - depth / 120.0, 0.55, 1.0)
        cave = rock * (tone * damp)[:, None]
        f = nz.smoothstep(0.4, 1.6, depth)[:, None]
        col = np.clip(surf + f * (cave - surf), 0.0, 1.0)
        # Near-ground vertices take the heightfield's normals, so shading meets the tiles.
        hn = np.stack(
            [-_bilinear_grad(H, pos, pad, 0), -_bilinear_grad(H, pos, pad, 1), np.ones(len(pos))],
            axis=1,
        )
        hn /= np.linalg.norm(hn, axis=1, keepdims=True)
        on_ground = (np.abs(depth) < 0.05)[:, None] & (nrm[:, 2:3] > 0.3)
        nrm = np.where(on_ground, hn, nrm)
        return Mesh._from_arrays(
            pos.astype(np.float32),
            tris.astype(np.int64),
            nrm.astype(np.float32),
            None,
            None,
            np.concatenate([col, np.ones((len(col), 1))], axis=1).astype(np.float32),
        )

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self.counters,
                "seconds": round(self.counters["seconds"], 4),
                "cubes_cached": len(self._cubes),
            }

    def clear(self) -> None:
        with self._lock:
            self._columns.clear()
            self._cubes.clear()
            self._anchors.clear()


def _marching_cubes(phi: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes (``sculpt.voxel``) over a dense grid ``phi[z, y, x]``, given only the
    corners of the cells that change sign."""
    from ..sculpt.voxel import marching_cubes

    nzs, nys, nxs = phi.shape
    s = phi < 0.0
    corners = [
        s[a : nzs - 1 + a, b : nys - 1 + b, c : nxs - 1 + c]
        for a in (0, 1)
        for b in (0, 1)
        for c in (0, 1)
    ]
    anyin = np.logical_or.reduce(corners)
    allin = np.logical_and.reduce(corners)
    kz, ky, kx = np.nonzero(anyin & ~allin)
    if not kz.size:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    base = kx + nxs * (ky + nys * kz)
    offsets = np.array(
        [(c & 1) + ((c >> 1) & 1) * nxs + ((c >> 2) & 1) * nxs * nys for c in range(8)]
    )
    keys = np.unique((base[:, None] + offsets[None, :]).reshape(-1))
    values = phi.reshape(-1)[keys]
    return marching_cubes(keys, values, np.array([nxs, nys, nzs]), origin, VOXEL)


def _trilinear(nodes: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Values at points, trilinear between the global lattice nodes around them;
    ``nodes(ix, iy, iz)`` gives the values at integer lattice coordinates. Dense points
    read a table over their box; scattered ones only the nodes they need."""
    u = np.stack([x, y, z]) / LATTICE
    i0 = np.floor(u).astype(np.int64)
    t = u - i0
    lo = i0.min(axis=1)
    span = i0.max(axis=1) - lo + 2
    n = x.shape[0]
    corners = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    if int(np.prod(span)) <= 8 * n + 64:
        axes = (np.arange(lo[k], lo[k] + span[k]) for k in range(3))
        table = nodes(*np.meshgrid(*axes, indexing="ij"))
        a, b, c = (i0[k] - lo[k] for k in range(3))
        vals = [table[a + dx, b + dy, c + dz] for dx, dy, dz in corners]
    else:
        pts = np.concatenate([(i0 + np.array(o)[:, None]).T for o in corners])  # (8n, 3)
        uniq, inv = np.unique(pts, axis=0, return_inverse=True)
        table = nodes(uniq[:, 0], uniq[:, 1], uniq[:, 2])
        vals = [table[inv.reshape(-1)[k * n : (k + 1) * n]] for k in range(8)]
    tx, ty, tz = t
    v = 0.0
    for (dx, dy, dz), val in zip(corners, vals, strict=True):
        w = (tx if dx else 1.0 - tx) * (ty if dy else 1.0 - ty) * (tz if dz else 1.0 - tz)
        v = v + val * (w[:, None] if val.ndim == 2 else w)
    return v


def _polyline_distance(p: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from points (n, 3) to a 3D polyline and the parameter along it in [0, 1]."""
    seg = pts[1:] - pts[:-1]
    lens = np.linalg.norm(seg, axis=1)
    total = max(float(lens.sum()), 1e-12)
    start = np.concatenate([[0.0], np.cumsum(lens)[:-1]])
    best = np.full(p.shape[0], np.inf)
    where = np.zeros(p.shape[0])
    for k in range(seg.shape[0]):
        ll = float(seg[k] @ seg[k])
        r = p - pts[k]
        t = np.clip((r @ seg[k]) / ll, 0.0, 1.0) if ll > 0.0 else np.zeros(p.shape[0])
        q = r - t[:, None] * seg[k]
        dist = np.sqrt(np.sum(q * q, axis=1))
        closer = dist < best
        best = np.where(closer, dist, best)
        where = np.where(closer, (start[k] + t * lens[k]) / total, where)
    return best, where


def _feature_box(vol: Volumes, f: dict) -> tuple[float, float, float, float, float, float]:
    """``(x0, y0, x1, y1, z0, z1)`` around a cave feature."""
    shape = f["shape"]
    r = max(f["radius"], f["radius_end"] or 0.0) * (1.0 + f["noise"]) + LATTICE
    if "line" in shape:
        pts = np.asarray(shape["line"], dtype=np.float64)
        lo, hi = pts.min(axis=0) - r, pts.max(axis=0) + r
        return (lo[0], lo[1], hi[0], hi[1], lo[2], hi[2])
    cx, cy, rc = shape["circle"]
    ground = vol._anchor(f["id"], cx, cy)
    rr = max(rc, r) * 1.7
    if f["kind"] == "hole":
        return (cx - rr, cy - rr, cx + rr, cy + rr, ground - 6.0 * rc, ground + 6.0)
    return (cx - rr, cy - rr, cx + rr, cy + rr, ground - 2.0 * rc, ground + 2.0)


def _taper(lx: np.ndarray, open_sides: int) -> np.ndarray:
    """1 inside a footprint, fading to 0 within :data:`TAPER` metres of the edges that are
    not open (bits -x, +x, -y, +y)."""
    fx = np.ones(lx.size)
    fy = np.ones(lx.size)
    if not open_sides & 1:
        fx = fx * nz.smoothstep(0.0, TAPER, lx)
    if not open_sides & 2:
        fx = fx * nz.smoothstep(0.0, TAPER, CHUNK - lx)
    if not open_sides & 4:
        fy = fy * nz.smoothstep(0.0, TAPER, lx)
    if not open_sides & 8:
        fy = fy * nz.smoothstep(0.0, TAPER, CHUNK - lx)
    return np.clip(fy[:, None] * fx[None, :], 0.0, 1.0)


def _bilinear(H: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear samples of a column grid ``H[v, u]`` at fractional indices (clamped)."""
    u = np.clip(u, 0.0, H.shape[1] - 1.000001)
    v = np.clip(v, 0.0, H.shape[0] - 1.000001)
    i = np.floor(u).astype(np.int64)
    j = np.floor(v).astype(np.int64)
    a, b = u - i, v - j
    return (H[j, i] * (1 - a) + H[j, i + 1] * a) * (1 - b) + (
        H[j + 1, i] * (1 - a) + H[j + 1, i + 1] * a
    ) * b


def _bilinear_grad(H: np.ndarray, pos: np.ndarray, pad: int, axis: int) -> np.ndarray:
    """The central-difference slope of the column heights along ``axis`` at points."""
    gx = np.zeros_like(H)
    if axis == 0:
        gx[:, 1:-1] = (H[:, 2:] - H[:, :-2]) / (2 * VOXEL)
    else:
        gx[1:-1, :] = (H[2:, :] - H[:-2, :]) / (2 * VOXEL)
    return _bilinear(gx, pos[:, 0] / VOXEL + pad, pos[:, 1] / VOXEL + pad)


def _gradient(phi: np.ndarray, pos: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """The gradient of a field sampled on the cube grid ``phi[z, y, x]``, at points
    (trilinear central differences)."""
    g = np.zeros(phi.shape + (3,))
    g[:, :, 1:-1, 0] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * VOXEL)
    g[:, 1:-1, :, 1] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * VOXEL)
    g[1:-1, :, :, 2] = (phi[2:] - phi[:-2]) / (2 * VOXEL)
    u = (pos - origin) / VOXEL
    n = phi.shape[0]
    i0 = np.clip(np.floor(u).astype(np.int64), 0, n - 2)
    t = np.clip(u - i0, 0.0, 1.0)
    out = np.zeros((len(pos), 3))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (
                    (t[:, 0] if dx else 1 - t[:, 0])
                    * (t[:, 1] if dy else 1 - t[:, 1])
                    * (t[:, 2] if dz else 1 - t[:, 2])
                )
                out += g[i0[:, 2] + dz, i0[:, 1] + dy, i0[:, 0] + dx] * w[:, None]
    return out


def _cube_outside(view: Any, x0: float, y0: float, z0: float) -> bool:
    from .tiles import outside

    return outside(view, (x0, y0, x0 + CHUNK, y0 + CHUNK), z0, z0 + CHUNK)


def volumes_of(rt: Any) -> Volumes:
    v = getattr(rt, "_volumes", None)
    if v is None:
        v = rt._volumes = Volumes(rt)
    return v


def render_parts(rt: Any, view: Any, ctx: Any = None) -> tuple[list, set]:
    """The drawn cubes' meshes ``(id, name, mesh, origin, "terrain")`` and the footprints
    the terrain tiles leave out."""
    vol = volumes_of(rt)
    cubes, holes = vol.drawn(view)
    if not cubes:
        return [], set()
    world = rt.world
    out = []
    for ci, cj, ck, _surface in cubes:
        if ctx is not None:
            ctx.check("volumetric terrain")
        open_sides = sum(
            bit
            for bit, (a, b) in zip((1, 2, 4, 8), ((-1, 0), (1, 0), (0, -1), (0, 1)), strict=True)
            if (ci + a, cj + b) in holes
        )
        mesh = vol.cube(ci, cj, ck, open_sides)
        if mesh is None:
            continue
        tag = f"{ci},{cj},{ck}"
        out.append(
            (
                f"world:{world.id}:volume/{tag}",
                f"{world.name} volume {tag}",
                mesh,
                np.array([ci * CHUNK, cj * CHUNK, 0.0]),
                "terrain",
            )
        )
    return out, holes
