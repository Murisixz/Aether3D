"""Parametric primitive sources: polygon meshes generated from validated parameters.

All primitives are centered at the origin with +Z up, are built from quads
(plus n-gon caps or pole triangles where natural), wind counter-clockwise
seen from outside, and carry per-corner UVs and exact (analytic) normals.
Curved surfaces are smooth-shaded with sharp edges where flat caps meet them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .params import ParamSpec, ProceduralError, validate_values
from .polymesh import PolyMesh

MAX_SEGMENTS = 1024


@dataclass(frozen=True)
class PrimitiveType:
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    build: Callable[..., PolyMesh]
    check: Callable[[dict], None] | None = None

    def schema(self) -> dict:
        return {
            "type": self.name,
            "description": self.description,
            "params": [p.to_json() for p in self.params],
        }

    def generate(self, values: dict, location: str) -> PolyMesh:
        resolved = validate_values(self.params, values, location)
        if self.check is not None:
            try:
                self.check(resolved)
            except ProceduralError as exc:
                raise ProceduralError(exc.code, exc.message, location, hint=exc.hint) from None
        return self.build(**resolved)


def _positive(name: str, desc: str, default: float = 1.0) -> ParamSpec:
    return ParamSpec(name, "float", default, min=1e-9, max=1e9, description=desc)


def _segments(name: str, default: int, minimum: int, desc: str) -> ParamSpec:
    return ParamSpec(name, "int", default, min=minimum, max=MAX_SEGMENTS, description=desc)


# --------------------------------------------------------------------------- #
# Box and grid
# --------------------------------------------------------------------------- #

_BOX_FACES = (  # (normal, u axis, v axis) with u x v == normal (same layout as create_cube)
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
)
_QUAD = ((-1, -1), (1, -1), (1, 1), (-1, 1))
_QUAD_UV = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def _box(width: float, depth: float, height: float) -> PolyMesh:
    half = np.array([width, depth, height]) / 2.0
    index: dict[tuple, int] = {}
    positions, faces, uvs, normals = [], [], [], []
    for normal, u, v in _BOX_FACES:
        n, uu, vv = (np.array(a) for a in (normal, u, v))
        loop = []
        for cu, cv in _QUAD:
            key = tuple((n + cu * uu + cv * vv).astype(int).tolist())
            if key not in index:
                index[key] = len(positions)
                positions.append(np.array(key) * half)
            loop.append(index[key])
        faces.append(loop)
        uvs.append(list(_QUAD_UV))
        normals.append([tuple(map(float, normal))] * 4)
    return PolyMesh.from_faces(np.array(positions), faces, uvs=uvs, normals=normals)


def _grid(x_size: float, y_size: float, x_segments: int, y_segments: int) -> PolyMesh:
    nx, ny = x_segments, y_segments
    xs = (np.arange(nx + 1) / nx - 0.5) * x_size
    ys = (np.arange(ny + 1) / ny - 0.5) * y_size
    gx, gy = np.meshgrid(xs, ys)
    positions = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    i, j = i.ravel(), j.ravel()
    k = j * (nx + 1) + i
    fv = np.stack([k, k + 1, k + nx + 2, k + nx + 1], axis=1).ravel()
    uv_i = np.stack([i, i + 1, i + 1, i], axis=1).ravel() / nx
    uv_j = np.stack([j, j, j + 1, j + 1], axis=1).ravel() / ny
    return PolyMesh(
        positions,
        np.arange(0, 4 * nx * ny + 1, 4),
        fv,
        uvs=np.stack([uv_i, uv_j], axis=1),
        normals=np.tile([0.0, 0.0, 1.0], (4 * nx * ny, 1)),
    )


# --------------------------------------------------------------------------- #
# Surfaces of revolution: cylinder / cone
# --------------------------------------------------------------------------- #


def _cone_like(
    radius_bottom: float,
    radius_top: float,
    depth: float,
    segments: int,
    caps: bool,
    smooth: bool,
) -> PolyMesh:
    n = segments
    theta = 2 * np.pi * np.arange(n) / n
    cos, sin = np.cos(theta), np.sin(theta)
    z0, z1 = -depth / 2.0, depth / 2.0
    positions: list = []
    faces: list[list[int]] = []
    uvs: list[list[tuple]] = []
    normals: list[list[tuple]] = []
    flags: list[bool] = []
    sharp: list[tuple[int, int]] = []

    def ring(r: float, z: float) -> list[int] | int:
        start = len(positions)
        if r == 0:
            positions.append((0.0, 0.0, z))
            return start
        positions.extend(zip((r * cos).tolist(), (r * sin).tolist(), [z] * n, strict=True))
        return list(range(start, start + n))

    bottom, top = ring(radius_bottom, z0), ring(radius_top, z1)
    slope = radius_bottom - radius_top

    def side_normal(t: float) -> tuple:
        v = np.array([depth * math.cos(t), depth * math.sin(t), slope])
        return tuple((v / np.linalg.norm(v)).tolist())

    for j in range(n):
        k = (j + 1) % n
        ta, tb = theta[j], theta[j] + 2 * np.pi / n
        u0, u1 = j / n, (j + 1) / n
        if isinstance(bottom, list) and isinstance(top, list):
            loop = [bottom[j], bottom[k], top[k], top[j]]
            uv = [(u0, 0.0), (u1, 0.0), (u1, 1.0), (u0, 1.0)]
            nr = [side_normal(ta), side_normal(tb), side_normal(tb), side_normal(ta)]
        elif isinstance(bottom, list):  # apex on top
            loop = [bottom[j], bottom[k], top]
            uv = [(u0, 0.0), (u1, 0.0), ((u0 + u1) / 2, 1.0)]
            nr = [side_normal(ta), side_normal(tb), side_normal((ta + tb) / 2)]
        else:  # apex at the bottom
            loop = [bottom, top[k], top[j]]
            uv = [((u0 + u1) / 2, 0.0), (u1, 1.0), (u0, 1.0)]
            nr = [side_normal((ta + tb) / 2), side_normal(tb), side_normal(ta)]
        faces.append(loop)
        uvs.append(uv)
        normals.append(nr if smooth else [None] * len(loop))
        flags.append(smooth)
    if caps:
        for ids, z_sign in ((top, 1.0), (bottom, -1.0)):
            if not isinstance(ids, list):
                continue
            loop = ids if z_sign > 0 else ids[::-1]
            faces.append(loop)
            ring_cos = cos if z_sign > 0 else cos[::-1]
            ring_sin = sin if z_sign > 0 else sin[::-1]
            uvs.append(
                list(
                    zip(
                        ((ring_cos + 1) / 2).tolist(),
                        ((ring_sin * z_sign + 1) / 2).tolist(),
                        strict=True,
                    )
                )
            )
            normals.append([(0.0, 0.0, z_sign)] * n)
            flags.append(False)
            if smooth:
                sharp += [(ids[j], ids[(j + 1) % n]) for j in range(n)]
    pm = PolyMesh.from_faces(
        np.array(positions, dtype=np.float64),
        faces,
        uvs=uvs,
        smooth=np.array(flags),
        sharp_edges=sharp or None,
    )
    face_unit = pm.face_normals / np.linalg.norm(pm.face_normals, axis=1)[:, None]
    corner_normals = [
        nn if nn is not None else tuple(face_unit[f]) for f, row in enumerate(normals) for nn in row
    ]
    return pm.with_(normals=np.array(corner_normals), normals_valid=np.ones(pm.face_count, bool))


def _cylinder(radius: float, depth: float, segments: int, caps: bool, smooth: bool) -> PolyMesh:
    return _cone_like(radius, radius, depth, segments, caps, smooth)


def _cone(
    radius_bottom: float, radius_top: float, depth: float, segments: int, caps: bool, smooth: bool
) -> PolyMesh:
    return _cone_like(radius_bottom, radius_top, depth, segments, caps, smooth)


def _cone_check(v: dict) -> None:
    if v["radius_bottom"] == 0 and v["radius_top"] == 0:
        raise ProceduralError(
            "INVALID_COMBINATION", "radius_bottom and radius_top cannot both be 0"
        )


# --------------------------------------------------------------------------- #
# Torus and UV sphere
# --------------------------------------------------------------------------- #


def _torus(
    major_radius: float, minor_radius: float, major_segments: int, minor_segments: int
) -> PolyMesh:
    nu, nv = major_segments, minor_segments
    theta = 2 * np.pi * np.arange(nu) / nu
    phi = 2 * np.pi * np.arange(nv) / nv
    t, p = np.meshgrid(theta, phi, indexing="ij")  # (nu, nv)
    ring = major_radius + minor_radius * np.cos(p)
    positions = np.stack(
        [ring * np.cos(t), ring * np.sin(t), minor_radius * np.sin(p)], -1
    ).reshape(-1, 3)
    normal_grid = np.stack([np.cos(p) * np.cos(t), np.cos(p) * np.sin(t), np.sin(p)], -1).reshape(
        -1, 3
    )
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    i, j = i.ravel(), j.ravel()
    i1, j1 = (i + 1) % nu, (j + 1) % nv
    fv = np.stack([i * nv + j, i1 * nv + j, i1 * nv + j1, i * nv + j1], axis=1).ravel()
    u = np.stack([i, i + 1, i + 1, i], axis=1).ravel() / nu
    v = np.stack([j, j, j + 1, j + 1], axis=1).ravel() / nv
    return PolyMesh(
        positions,
        np.arange(0, 4 * nu * nv + 1, 4),
        fv,
        uvs=np.stack([u, v], 1),
        normals=normal_grid[fv],
        smooth=np.ones(nu * nv, bool),
    )


def _torus_check(v: dict) -> None:
    if v["minor_radius"] >= v["major_radius"]:
        raise ProceduralError(
            "INVALID_COMBINATION",
            "minor_radius must be smaller than major_radius "
            "(otherwise the torus intersects itself)",
        )


def _uv_sphere(radius: float, segments: int, rings: int) -> PolyMesh:
    n, m = segments, rings
    theta = np.pi * np.arange(1, m) / m  # interior rings (poles are single vertices)
    phi = 2 * np.pi * np.arange(n) / n
    t, p = np.meshgrid(theta, phi, indexing="ij")
    ring_pts = np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], -1).reshape(
        -1, 3
    )
    unit = np.vstack([[0.0, 0.0, 1.0], ring_pts, [0.0, 0.0, -1.0]])
    north, south = 0, len(unit) - 1

    def vid(r: int, s: int) -> int:  # r in 1..m-1
        return 1 + (r - 1) * n + (s % n)

    faces, uvs = [], []
    for s in range(n):
        u0, u1 = s / n, (s + 1) / n
        faces.append([north, vid(1, s), vid(1, s + 1)])
        uvs.append([((u0 + u1) / 2, 1.0), (u0, 1 - 1 / m), (u1, 1 - 1 / m)])
        for r in range(1, m - 1):
            faces.append([vid(r, s), vid(r + 1, s), vid(r + 1, s + 1), vid(r, s + 1)])
            uvs.append(
                [(u0, 1 - r / m), (u0, 1 - (r + 1) / m), (u1, 1 - (r + 1) / m), (u1, 1 - r / m)]
            )
        faces.append([vid(m - 1, s), south, vid(m - 1, s + 1)])
        uvs.append([(u0, 1 / m), ((u0 + u1) / 2, 0.0), (u1, 1 / m)])
    pm = PolyMesh.from_faces(unit * radius, faces, uvs=uvs, smooth=np.ones(len(faces), bool))
    return pm.with_(normals=unit[pm.face_vertices], normals_valid=np.ones(pm.face_count, bool))


PRIMITIVES: dict[str, PrimitiveType] = {
    t.name: t
    for t in (
        PrimitiveType(
            "box",
            "Axis-aligned box of 6 quads (per-face UVs, flat shading)",
            (
                _positive("width", "size along X"),
                _positive("depth", "size along Y"),
                _positive("height", "size along Z"),
            ),
            _box,
        ),
        PrimitiveType(
            "grid",
            "Flat grid of quads in the XY plane facing +Z (UVs 0..1)",
            (
                _positive("x_size", "size along X", 2.0),
                _positive("y_size", "size along Y", 2.0),
                _segments("x_segments", 1, 1, "quads along X"),
                _segments("y_segments", 1, 1, "quads along Y"),
            ),
            _grid,
        ),
        PrimitiveType(
            "cylinder",
            "Cylinder along Z: quad sides (smooth) and n-gon caps (flat, sharp-edged)",
            (
                _positive("radius", "radius"),
                _positive("depth", "height along Z", 2.0),
                _segments("segments", 32, 3, "sides around Z"),
                ParamSpec("caps", "bool", True, description="close the ends with n-gons"),
                ParamSpec("smooth", "bool", True, description="smooth-shade the sides"),
            ),
            _cylinder,
        ),
        PrimitiveType(
            "cone",
            "Cone or frustum along Z; a zero radius makes an apex",
            (
                ParamSpec(
                    "radius_bottom",
                    "float",
                    1.0,
                    min=0,
                    max=1e9,
                    description="radius at z = -depth/2",
                ),
                ParamSpec(
                    "radius_top", "float", 0.0, min=0, max=1e9, description="radius at z = +depth/2"
                ),
                _positive("depth", "height along Z", 2.0),
                _segments("segments", 32, 3, "sides around Z"),
                ParamSpec("caps", "bool", True, description="close the ends with n-gons"),
                ParamSpec("smooth", "bool", True, description="smooth-shade the sides"),
            ),
            _cone,
            _cone_check,
        ),
        PrimitiveType(
            "torus",
            "Torus around Z (all quads, smooth)",
            (
                _positive("major_radius", "radius of the ring"),
                _positive("minor_radius", "radius of the tube", 0.25),
                _segments("major_segments", 48, 3, "segments around the ring"),
                _segments("minor_segments", 12, 3, "segments around the tube"),
            ),
            _torus,
            _torus_check,
        ),
        PrimitiveType(
            "uv_sphere",
            "UV sphere with poles on Z: quads plus pole triangles, smooth",
            (
                _positive("radius", "radius"),
                _segments("segments", 32, 3, "divisions around Z"),
                _segments("rings", 16, 2, "divisions pole to pole"),
            ),
            _uv_sphere,
        ),
    )
}
