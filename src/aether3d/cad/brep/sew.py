"""Sewing solids along a shared planar face.

Two solids that meet exactly on a face -- the end cap of one sweep segment and the start
cap of the next -- are joined by dropping both caps and identifying their boundary
vertices and edges (same positions; same curve, possibly run the other way). Unlike a
boolean, sewing never re-intersects anything: a joint where the segments meet
tangentially (a straight run flowing into a bend) stays exact, where tessellated
booleans would fragment the nearly coincident faces.

The caps must match: the same loops of edges (by end points and midpoints, within the
tolerance). Anything else is ``SEW_MISMATCH``.
"""

from __future__ import annotations

import numpy as np

from ...errors import CadError
from .topology import Body, Edge, Face


def _cap_edges(body: Body, fi: int) -> list[int]:
    return [ei for loop in body.faces[fi].loops for ei, _ in loop]


def sew(a: Body, cap_a: int, b: Body, cap_b: int, tol: float) -> Body:
    """``a`` and ``b`` joined across face ``cap_a`` of ``a`` and ``cap_b`` of ``b``."""
    ea, eb = _cap_edges(a, cap_a), _cap_edges(b, cap_b)
    if len(ea) != len(eb):
        raise CadError("SEW_MISMATCH", "the faces to join have different boundaries")
    # Vertices of b mapped onto a's (the caps' vertices coincide).
    vmap: dict[int, int] = {}
    cap_v_a = {v for ei in ea for v in (a.edges[ei].v0, a.edges[ei].v1)}
    for ei in eb:
        for v in (b.edges[ei].v0, b.edges[ei].v1):
            if v in vmap:
                continue
            d = {va: float(np.linalg.norm(a.vertices[va] - b.vertices[v])) for va in cap_v_a}
            best = min(d, key=d.get)
            if d[best] > tol:
                raise CadError("SEW_MISMATCH", "the faces to join have different corners")
            vmap[v] = best
    # Edges of b's cap mapped onto a's: same end vertices and midpoint.
    emap: dict[int, tuple[int, bool]] = {}
    for ei in eb:
        e = b.edges[ei]
        m = e.midpoint()
        for ej in ea:
            f = a.edges[ej]
            same = {vmap[e.v0], vmap[e.v1]} == {f.v0, f.v1}
            if same and float(np.linalg.norm(f.midpoint() - m)) <= tol:
                emap[ei] = (ej, vmap[e.v0] == f.v0)
                break
        else:
            raise CadError("SEW_MISMATCH", "an edge of the joint has no partner")
    nva = len(a.vertices)
    keep_v = [v for v in range(len(b.vertices)) if v not in vmap]
    new_index = {v: nva + k for k, v in enumerate(keep_v)}
    vindex = {**{v: vmap[v] for v in vmap}, **new_index}
    verts = np.concatenate([a.vertices, b.vertices[keep_v]]) if keep_v else a.vertices.copy()
    vtol = np.concatenate([a.vertex_tol, b.vertex_tol[keep_v]]) if keep_v else a.vertex_tol.copy()
    edges = list(a.edges)
    eindex: dict[int, tuple[int, bool]] = {}
    for ei, e in enumerate(b.edges):
        if ei in emap:
            eindex[ei] = emap[ei]
        else:
            eindex[ei] = (len(edges), True)
            edges.append(Edge(e.curve, e.t0, e.t1, vindex[e.v0], vindex[e.v1], e.name, e.tol))
    faces = [f for i, f in enumerate(a.faces) if i != cap_a]
    for i, f in enumerate(b.faces):
        if i == cap_b:
            continue
        loops = tuple(
            tuple((eindex[ei][0], fwd if eindex[ei][1] else not fwd) for ei, fwd in lp)
            for lp in f.loops
        )
        faces.append(Face(f.surface, f.sense, loops, f.name, f.group, f.window))
    # Drop edges and vertices no face uses any more? The joint's edges are still used
    # by the side faces on both sides, so nothing is orphaned.
    return Body(verts, tuple(edges), tuple(faces), (tuple(range(len(faces))),), "solid", vtol)


def face_named(body: Body, suffix: str) -> int:
    hits = [i for i, f in enumerate(body.faces) if f.name.endswith(suffix)]
    if len(hits) != 1:
        raise CadError("SEW_MISMATCH", f"expected one face named *{suffix}, found {len(hits)}")
    return hits[0]
