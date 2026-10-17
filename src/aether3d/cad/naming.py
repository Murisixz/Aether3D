"""Persistent names of faces and edges, and resolving references to them.

**Faces** are named by what generated them: ``"<feature>:<role>"`` --
``"base:side/l3"`` (the side swept by sketch line ``l3`` in feature ``base``),
``"base:end"``, ``"h1:rev/wall"``, ``"fl1:round"``. Booleans keep the names of the faces
their result faces lie on; a face split into several pieces gets ``"#0"``, ``"#1"``, ...
ordered by position (lowest centroid first, deterministically).

**Edges** are named by the two faces they separate: ``{"edge": ["base:end",
"base:side/l3"]}`` (or the string ``"base:end|base:side/l3"``).

A reference may carry ``"near": [x, y, z]`` (in document units): when a name matches
several pieces, the piece closest to that point is taken -- if it is clearly the closest
(its distance is less than half of the next candidate's).
Otherwise resolution fails explicitly:

* ``REFERENCE_LOST`` -- nothing carries the name any more (the generating feature or
  sketch entity is gone); the error lists the faces that exist;
* ``REFERENCE_AMBIGUOUS`` -- several pieces match and no ``near`` point decides; the error
  lists each candidate with its centroid so the reference can be made precise.

Triangle or face *indices* are never used as references: they change whenever the model
regenerates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import CadError
from .brep.topology import Body, face_sample


def _face_center(body: Body, fi: int) -> np.ndarray:
    p, _ = face_sample(body, body.faces[fi])
    return p


def face_candidates(body: Body, name: str) -> list[int]:
    exact = [i for i, f in enumerate(body.faces) if f.name == name]
    if exact:
        return exact
    return [i for i, f in enumerate(body.faces) if f.name.split("#")[0] == name]


def parse_face_ref(ref: Any, location: str) -> tuple[str, np.ndarray | None]:
    if isinstance(ref, str):
        return ref, None
    if isinstance(ref, dict) and isinstance(ref.get("face"), str):
        near = ref.get("near")
        return ref["face"], None if near is None else np.asarray(near, dtype=np.float64)
    raise CadError(
        "INVALID_REFERENCE",
        'a face reference is a name or {"face": name, "near": [x, y, z]}',
        location,
    )


def resolve_face(body: Body, ref: Any, location: str, near_scale: float = 1.0) -> int:
    name, near = parse_face_ref(ref, location)
    cands = face_candidates(body, name)
    if not cands:
        raise CadError(
            "REFERENCE_LOST",
            f"no face named {name!r} in the current shape",
            location,
            hint="faces: " + ", ".join(sorted({f.name for f in body.faces})[:40]),
            details={"reference": name},
        )
    if len(cands) == 1:
        return cands[0]
    return _pick(
        body, cands, near, near_scale, name, location, _face_center, lambda c: body.faces[c].name
    )


def _pick(body, cands, near, near_scale, name, location, center_of, label_of) -> int:
    centers = {c: center_of(body, c) for c in cands}
    if near is not None:
        p = near * near_scale
        dist = sorted((float(np.linalg.norm(centers[c] - p)), c) for c in cands)
        if len(dist) == 1 or dist[0][0] < 0.5 * dist[1][0]:
            return dist[0][1]
    raise CadError(
        "REFERENCE_AMBIGUOUS",
        f"{len(cands)} pieces match {name!r}",
        location,
        hint='add "near": [x, y, z] to choose one, e.g. one of the centroids listed',
        details={
            "candidates": [
                {"name": label_of(c), "point": (centers[c] / near_scale).tolist()} for c in cands
            ]
        },
    )


def parse_edge_ref(ref: Any, location: str) -> tuple[str, str, np.ndarray | None]:
    near = None
    if isinstance(ref, str) and "|" in ref:
        a, b = ref.split("|", 1)
    elif (
        isinstance(ref, dict)
        and isinstance(ref.get("edge"), (list, tuple))
        and len(ref["edge"]) == 2
    ):
        a, b = ref["edge"]
        if ref.get("near") is not None:
            near = np.asarray(ref["near"], dtype=np.float64)
    else:
        raise CadError(
            "INVALID_REFERENCE",
            'an edge reference is "faceA|faceB" or {"edge": [faceA, faceB], "near": [x, y, z]}',
            location,
        )
    if not isinstance(a, str) or not isinstance(b, str):
        raise CadError("INVALID_REFERENCE", "edge reference faces must be names", location)
    return a.strip(), b.strip(), near


def resolve_edge(body: Body, ref: Any, location: str, near_scale: float = 1.0) -> int:
    a, b, near = parse_edge_ref(ref, location)
    fa, fb = set(face_candidates(body, a)), set(face_candidates(body, b))
    for name, found in ((a, fa), (b, fb)):
        if not found:
            raise CadError(
                "REFERENCE_LOST",
                f"no face named {name!r} in the current shape",
                location,
                hint="faces: " + ", ".join(sorted({f.name for f in body.faces})[:40]),
                details={"reference": name},
            )
    uses = body.edge_faces()
    cands = [
        ei
        for ei, fs in uses.items()
        if len(fs) == 2
        and ({fs[0][0], fs[1][0]} & fa)
        and ({fs[0][0], fs[1][0]} & fb)
        and fs[0][0] != fs[1][0]
    ]
    if not cands:
        raise CadError(
            "REFERENCE_LOST",
            f"the faces {a!r} and {b!r} no longer share an edge",
            location,
            details={"reference": [a, b]},
        )
    if len(cands) == 1:
        return cands[0]

    def mid(body_, ei):
        return body_.edges[ei].midpoint()

    return _pick(body, cands, near, near_scale, f"{a}|{b}", location, mid, lambda e: f"{a}|{b}")


def describe_faces(body: Body, scale: float = 1.0) -> list[dict[str, Any]]:
    """Every face's name, surface kind and a centroid (document units)."""
    out = []
    for i, f in enumerate(body.faces):
        c = _face_center(body, i) / scale
        out.append({"name": f.name, "surface": f.surface.kind, "point": [round(v, 9) for v in c]})
    return out


def edge_name(body: Body, ei: int) -> str:
    uses = body.edge_faces().get(ei, [])
    names = sorted(body.faces[f].name for f, _ in uses)
    return "|".join(names)
