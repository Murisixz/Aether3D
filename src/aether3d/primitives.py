"""Generators for basic primitive meshes.

All primitives are centered at the origin of their local space, use the
engine's +Z-up convention, carry per-vertex normals and UVs, and wind their
triangles counter-clockwise when viewed from outside.
"""

from __future__ import annotations

import numpy as np

from .math3d import int_at_least, positive_number
from .mesh import Mesh

# (normal, u axis, v axis) for each cube face, with u x v == normal so that
# the quad corners below come out counter-clockwise seen from outside.
_CUBE_FACES = (
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
)
_QUAD_CORNERS = ((-1, -1), (1, -1), (1, 1), (-1, 1))
_QUAD_UVS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def create_cube(size: float = 1.0) -> Mesh:
    """Axis-aligned cube with edge length ``size``.

    Each face has its own 4 vertices (24 total) so normals and UVs stay flat
    per face; 12 triangles.
    """
    half = positive_number(size, "size") / 2.0
    positions, normals, uvs, triangles = [], [], [], []
    for normal, u_axis, v_axis in _CUBE_FACES:
        n, u, v = (np.array(a, dtype=np.float64) for a in (normal, u_axis, v_axis))
        base = len(positions)
        for (cu, cv), uv in zip(_QUAD_CORNERS, _QUAD_UVS, strict=True):
            positions.append((n + cu * u + cv * v) * half)
            normals.append(n)
            uvs.append(uv)
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return Mesh(positions, triangles, normals=normals, uvs=uvs)


def create_sphere(radius: float = 1.0, segments: int = 32, rings: int = 16) -> Mesh:
    """UV sphere of the given ``radius``, with its poles on the Z axis.

    ``segments`` is the number of divisions around the Z axis (>= 3) and
    ``rings`` the number of divisions from pole to pole (>= 2). A seam column
    of vertices is duplicated so UVs wrap cleanly.
    """
    r = positive_number(radius, "radius")
    segments = int_at_least(segments, 3, "segments")
    rings = int_at_least(rings, 2, "rings")

    theta = np.linspace(0.0, np.pi, rings + 1)  # polar angle, from +Z down
    phi = np.linspace(0.0, 2.0 * np.pi, segments + 1)  # azimuth around +Z
    t, p = np.meshgrid(theta, phi, indexing="ij")
    sin_t = np.sin(t)
    sin_t[0, :] = sin_t[-1, :] = 0.0  # exact poles
    normals = np.stack([sin_t * np.cos(p), sin_t * np.sin(p), np.cos(t)], axis=-1).reshape(-1, 3)
    uvs = np.stack(
        [np.broadcast_to(np.arange(segments + 1) / segments, t.shape), 1.0 - t / np.pi], axis=-1
    ).reshape(-1, 2)

    stride = segments + 1
    triangles = []
    for i in range(rings):
        for j in range(segments):
            a = i * stride + j  # upper-left of the quad (seen from outside)
            b, c = a + 1, a + stride
            d = c + 1
            if i != 0:  # the top row collapses a-b onto the north pole
                triangles.append((a, d, b))
            if i != rings - 1:  # the bottom row collapses c-d onto the south pole
                triangles.append((a, c, d))
    return Mesh(normals * r, triangles, normals=normals, uvs=uvs)


def create_plane(size: float = 1.0) -> Mesh:
    """Square of edge length ``size`` in the XY plane, facing +Z."""
    half = positive_number(size, "size") / 2.0
    positions = [(cu * half, cv * half, 0.0) for cu, cv in _QUAD_CORNERS]
    normals = [(0.0, 0.0, 1.0)] * 4
    return Mesh(positions, [(0, 1, 2), (0, 2, 3)], normals=normals, uvs=_QUAD_UVS)
