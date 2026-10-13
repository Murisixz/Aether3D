"""Corner normals: preserved where geometry is unchanged, recomputed where it changed.

Shading model (as in Blender): every face is *flat* or *smooth*, and edges
may be *sharp*. A flat face's corners use the face normal. A smooth face's
corner at vertex ``v`` uses the angle-weighted average of the normals of the
smooth faces around ``v`` that can be reached without crossing a sharp,
boundary or non-manifold edge (Thürmer-Wüthrich weighting, which does not
depend on how the faces are triangulated).

Only corners touching a face whose geometry changed ("dirty" faces) are
recomputed; all other stored normals are returned unchanged.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from . import _polygon as poly

if TYPE_CHECKING:
    from .core import EditableMesh

FALLBACK = (0.0, 0.0, 1.0)


def _corner_angle(points: np.ndarray, k: int) -> float:
    n = len(points)
    a = points[k - 1] - points[k]
    b = points[(k + 1) % n] - points[k]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (na * nb)))))


def corner_normals(
    mesh: EditableMesh, *, recompute_all: bool = False
) -> dict[int, list[tuple[float, float, float]]]:
    """Per-face lists of corner normals (in loop order) for every face."""
    stored = mesh._layers.get("normal")
    result: dict[int, list[tuple]] = {}
    if stored is not None and not recompute_all:
        result = {fid: list(stored.data[fid]) for fid in mesh._faces}
        targets: set[int] = set()
        for fid in mesh._dirty_faces:
            if fid in mesh._faces:
                targets.update(mesh._faces[fid])
    else:
        result = {fid: [FALLBACK] * len(loop) for fid, loop in mesh._faces.items()}
        targets = set(mesh.vertex_ids())
    if not targets:
        return result

    face_normal: dict[int, np.ndarray | None] = {}

    def normal_of(fid: int) -> np.ndarray | None:
        if fid not in face_normal:
            face_normal[fid] = poly.unit(poly.newell_normal(mesh._pos[mesh._faces[fid]]))
        return face_normal[fid]

    for v in targets:
        for fid in mesh._faces_of_vertex(v):
            loop = mesh._faces[fid]
            k = loop.index(v)
            if not mesh._smooth.get(fid, False):
                n = normal_of(fid)
            else:
                n = _smooth_normal(mesh, v, fid, normal_of)
            if n is None:  # zero-area face: fall back to the vertex's other faces
                total = np.zeros(3)
                for other in mesh._faces_of_vertex(v):
                    on = normal_of(other)
                    if on is not None:
                        total += on
                n = poly.unit(total)
            result[fid][k] = tuple(n.tolist()) if n is not None else FALLBACK
    return result


def _smooth_normal(mesh: EditableMesh, v: int, start: int, normal_of) -> np.ndarray | None:
    fan = {start}
    stack = [start]
    while stack:
        fid = stack.pop()
        loop = mesh._faces[fid]
        k = loop.index(v)
        for other_vertex in (loop[k - 1], loop[(k + 1) % len(loop)]):
            eid = mesh._edge_id(v, other_vertex)
            if eid in mesh._sharp:
                continue
            faces = mesh._edge_faces[eid]
            if len(faces) != 2:
                continue
            g = faces[0] if faces[1] == fid else faces[1]
            if g not in fan and mesh._smooth.get(g, False):
                fan.add(g)
                stack.append(g)
    total = np.zeros(3)
    for fid in fan:
        n = normal_of(fid)
        if n is None:
            continue
        loop = mesh._faces[fid]
        total += n * _corner_angle(mesh._pos[loop], loop.index(v))
    return poly.unit(total)
