"""Shading and orientation: auto smooth, consistent outward winding, weighted normals.

* ``set_auto_smooth(angle)`` marks faces smooth and an edge sharp exactly when its two
  faces' normals differ by more than ``angle`` degrees (the shading model of
  EDITING.md then gives flat-looking creases and smooth curvature).
* ``recalculate_normals()`` makes the winding consistent within each connected piece
  (faces joined by manifold edges): a breadth-first walk from the piece's lowest-id face
  flips every face that runs a shared edge the same way as its neighbour. A closed piece
  is then turned outward (positive signed volume); an open piece has no inside, so its
  orientation follows the area-weighted majority of its faces facing away from its
  centroid, which is reported as a heuristic (``ORIENTATION_HEURISTIC``). A piece that
  cannot be oriented (a Moebius strip) is left as it is and reported (``NON_ORIENTABLE``).
* ``set_weighted_normals()`` writes custom corner normals: within each smooth fan
  (bounded by sharp, boundary and non-manifold edges, as for computed normals) the
  normal is the sum of the fan's face normals weighted by ``area ** (weight / 50)``
  (``"face_area"``), ``corner angle ** (weight / 50)`` (``"corner_angle"``) or their
  product (``"face_area_angle"``). ``weight=50`` is plain area weighting; larger weights
  let big faces dominate, which keeps large flat faces of hard-surface models flat next
  to small bevels. The normals are stored and kept at commit; a later geometric edit of
  a face recomputes its normals as usual.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import EditError, ValidationError
from . import _polygon as poly
from .core import NORMAL, Elements, Selection, operation


class ShadingMixin:
    @operation
    def set_auto_smooth(self, angle: float = 30.0, *, faces: Elements | None = None):
        """Smooth-shade faces and mark edges sharp where faces meet at more than ``angle``."""
        a = float(angle)
        if not 0.0 <= a <= 180.0:
            raise ValidationError("angle must be between 0 and 180 degrees")
        fids = self.face_ids() if faces is None else self._elements(faces, "faces")
        pool = set(fids)
        cos_limit = math.cos(math.radians(a))
        for f in fids:
            if not self._smooth.get(f, False):
                self._set_smooth(f, True)
        sharp, smooth = [], []
        for e in sorted({e for f in fids for e in self.face_edges(f)}):
            fs = self._edge_faces[e]
            if len(fs) != 2 or not set(fs) <= pool:
                continue
            n0, n1 = self._face_normal_vec(fs[0]), self._face_normal_vec(fs[1])
            crease = n0 is None or n1 is None or float(np.dot(n0, n1)) < cos_limit - 1e-12
            if crease and e not in self._sharp:
                self._set_sharp(e, True)
                sharp.append(e)
            elif not crease and e in self._sharp:
                self._set_sharp(e, False)
                smooth.append(e)
        self._mark_dirty(fids)
        return {
            "sharp_edges": Selection.of(edges=sharp),
            "smoothed_edges": Selection.of(edges=smooth),
        }

    @operation
    def recalculate_normals(self, *, faces: Elements | None = None, inside: bool = False):
        """Make winding consistent and outward (``inside=True``: inward) per connected piece."""
        fids = self.face_ids() if faces is None else self._elements(faces, "faces")
        pool = set(fids)
        flipped: list[int] = []
        heuristic, non_orientable = [], []
        seen: set[int] = set()
        for seed in fids:
            if seed in seen:
                continue
            # Breadth-first: orientation of each face relative to the seed (+1 keep, -1 flip).
            orient = {seed: 1}
            queue = deque([seed])
            bad = False
            while queue:
                f = queue.popleft()
                loop = self._faces[f]
                m = len(loop)
                for k in range(m):
                    a, b = loop[k], loop[(k + 1) % m]
                    if orient[f] < 0:
                        a, b = b, a
                    e = self._edge_id(a, b)
                    fs = self._edge_faces[e]
                    if len(fs) != 2:
                        continue
                    g = fs[0] if fs[1] == f else fs[1]
                    if g not in pool:
                        continue
                    lg = self._faces[g]
                    runs_same = lg[(lg.index(a) + 1) % len(lg)] == b
                    want = -1 if runs_same else 1
                    if g not in orient:
                        orient[g] = want
                        queue.append(g)
                    elif orient[g] != want:
                        bad = True
            seen.update(orient)
            piece = sorted(orient)
            if bad:
                non_orientable.extend(piece)
                continue
            # Which way out? Closed pieces: the signed volume; open ones: a majority vote.
            closed = all(len(self._edge_faces[e]) == 2 for f in piece for e in self.face_edges(f))
            centroid = np.mean([self._pos[v] for f in piece for v in self._faces[f]], axis=0)
            score = 0.0
            for f in piece:
                pts = self._pos[self._faces[f]]
                nv = poly.newell_normal(pts) * orient[f]
                if closed:
                    score += float(np.dot(pts[0], nv))  # 6 x volume (fan around pts[0])
                else:
                    score += float(np.dot(pts.mean(axis=0) - centroid, nv))
            if not closed:
                heuristic.extend(piece)
            outward = score >= 0
            if inside:
                outward = not outward
            flip = [f for f in piece if (orient[f] < 0) == outward]
            if flip:
                self.flip_faces(flip)
                flipped.extend(flip)
        if heuristic:
            self._diagnose(
                Diagnostic(
                    "ORIENTATION_HEURISTIC",
                    INFO,
                    "open pieces have no inside; they face away from their centroid (majority)",
                    None,
                    len(heuristic),
                    tuple(heuristic[:10]),
                )
            )
        if non_orientable:
            self._diagnose(
                Diagnostic(
                    "NON_ORIENTABLE",
                    WARNING,
                    "pieces that cannot be oriented consistently were left unchanged",
                    None,
                    len(non_orientable),
                    tuple(non_orientable[:10]),
                )
            )
        return {"flipped": sorted(flipped)}

    @operation
    def set_weighted_normals(
        self,
        *,
        faces: Elements | None = None,
        mode: str = "face_area",
        weight: float = 50.0,
        keep_sharp: bool = True,
    ):
        """Write custom corner normals weighted toward large faces (see the module docs)."""
        if NORMAL not in self._layers:
            raise EditError(
                "NO_NORMAL_LAYER", "the mesh stores no normals", hint="open it with normals"
            )
        if mode not in ("face_area", "corner_angle", "face_area_angle"):
            raise ValidationError(
                f"mode must be face_area, corner_angle or face_area_angle, got {mode!r}"
            )
        w = float(weight)
        if not 1.0 <= w <= 100.0:
            raise ValidationError("weight must be between 1 and 100")
        power = w / 50.0
        fids = self.face_ids() if faces is None else self._elements(faces, "faces")
        normal: dict[int, np.ndarray | None] = {}
        area: dict[int, float] = {}
        around = {g for f in fids for v in self._faces[f] for g in self._faces_of_vertex(v)}
        for g in around:
            nv = poly.newell_normal(self._pos[self._faces[g]])
            normal[g] = poly.unit(nv)
            area[g] = float(np.linalg.norm(nv)) / 2.0
        for f in fids:
            loop = self._faces[f]
            values = []
            for v in loop:
                fan = self._fan_around(v, f, keep_sharp) if self._smooth.get(f, False) else [f]
                total = np.zeros(3)
                for g in fan:
                    ng = normal[g]
                    if ng is None:
                        continue
                    lg = self._faces[g]
                    kg = lg.index(v)
                    angle = _corner_angle(self._pos[lg], kg)
                    if mode == "face_area":
                        wt = area[g] ** power
                    elif mode == "corner_angle":
                        wt = angle**power
                    else:
                        wt = (area[g] * angle) ** power
                    total += wt * ng
                n = poly.unit(total)
                if n is None:
                    n = normal[f] if normal[f] is not None else np.array([0.0, 0.0, 1.0])
                values.append(tuple(float(x) for x in n))
            self._set_corners(f, NORMAL, values)
        self._mark_clean(fids)
        return {"faces": Selection.of(faces=fids)}

    def _fan_around(self, v: int, start: int, keep_sharp: bool) -> list[int]:
        """Smooth faces around ``v`` reachable from ``start`` without crossing a sharp
        (if ``keep_sharp``), boundary or non-manifold edge."""
        fan = {start}
        stack = [start]
        while stack:
            f = stack.pop()
            loop = self._faces[f]
            k = loop.index(v)
            for u in (loop[k - 1], loop[(k + 1) % len(loop)]):
                e = self._edge_id(v, u)
                if keep_sharp and e in self._sharp:
                    continue
                fs = self._edge_faces[e]
                if len(fs) != 2:
                    continue
                g = fs[0] if fs[1] == f else fs[1]
                if g not in fan and self._smooth.get(g, False):
                    fan.add(g)
                    stack.append(g)
        return sorted(fan)


def _corner_angle(points: np.ndarray, k: int) -> float:
    n = len(points)
    a = points[k - 1] - points[k]
    b = points[(k + 1) % n] - points[k]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, float(np.dot(a, b)) / (na * nb))))
