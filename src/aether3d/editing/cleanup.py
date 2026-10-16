"""Dissolving, connecting, and cleaning up meshes with an auditable report.

* ``dissolve_vertices`` removes vertices and merges the faces around each into one: the
  interior edges at the vertex are dissolved (``dissolve_edges``' rules), then the vertex,
  now inside one face's outline with two edges, leaves the loop.
* ``dissolve_faces`` merges each connected region of faces into one polygon; the region
  must be a disk (one outline), otherwise ``DISSOLVE_INVALID_FACE``.
* ``limited_dissolve`` dissolves edges between nearly coplanar faces (smallest angle
  first) and then vertices in the middle of straight runs, never across face-group
  borders, UV seams or sharp edges unless asked.
* ``connect_vertex_path`` and ``connect_vertices`` split faces between selected vertices.
* ``cleanup`` runs explicit repairs (merge by distance, zero-length edges, duplicate faces,
  loose vertices, winding, holes) and returns what each step changed and what it could not.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import EditError, ValidationError
from . import _polygon as poly
from .core import Elements, Selection, nested_data, operation


class CleanupMixin:
    # -- dissolve --------------------------------------------------------------------

    @operation
    def dissolve_vertices(self, vertices: Elements):
        """Remove vertices, merging the faces around each into one polygon.

        Each vertex must have one manifold fan (closed, or open on a boundary); a vertex
        on two edges of one face simply leaves the loop. ``data["faces"]`` are the merged
        faces (each keeps its lowest face id).
        """
        vids = self._elements(vertices, "vertices")
        merged: set[int] = set()
        for v in vids:
            if not self.has_vertex(v):
                continue  # removed with an earlier vertex's faces
            merged.add(self._dissolve_vertex(v))
        return {"faces": Selection.of(faces=[f for f in merged if f in self._faces])}

    def _dissolve_vertex(self, v: int) -> int:
        edges = sorted(self._vert_edges[v])
        faces = sorted(self._faces_of_vertex(v))
        if len(edges) == 2:
            # A vertex in the middle of two edges: it simply leaves its faces' loops.
            a, b = (x for e in edges for x in self._edge_verts[e] if x != v)
            if self.edge_between(a, b) is not None:
                raise EditError(
                    "DISSOLVE_INVALID_FACE",
                    f"vertices {a} and {b} are already joined by an edge",
                    elements={"vertices": [v]},
                )
            for f in faces:
                loop = self._faces[f]
                if len(loop) <= 3:
                    raise EditError(
                        "DISSOLVE_INVALID_FACE",
                        f"removing vertex {v} would leave face {f} with fewer than 3 vertices",
                        elements={"faces": [f]},
                    )
                k = loop.index(v)
                corners = self._corners_of(f)
                self._replace_face(
                    f, loop[:k] + loop[k + 1 :], self._split_corners(corners[:k] + corners[k + 1 :])
                )
            self._delete_vertex(v)
            return faces[0]
        fan, _closed = self._vertex_fan(v)
        # Each face's outline without v runs from its q (after v) round to its p (before
        # v); the next face's outline ends where this one starts, so concatenating them
        # in reverse fan order gives the merged polygon, wound like the faces.
        loop: list[int] = []
        corners: list[dict] = []
        for f, _e_in, _e_out, _p, _q in reversed(fan):
            fl, fc = self._faces[f], self._corners_of(f)
            k = fl.index(v)
            path = [(k + j) % len(fl) for j in range(1, len(fl))]  # q .. p
            if loop and loop[-1] == fl[path[0]]:
                path = path[1:]
            loop += [fl[j] for j in path]
            corners += [fc[j] for j in path]
        if len(loop) > 1 and loop[0] == loop[-1]:
            loop.pop()
            corners.pop()
        if len(loop) < 3 or len(set(loop)) != len(loop):
            raise EditError(
                "DISSOLVE_INVALID_FACE",
                f"the faces around vertex {v} touch elsewhere; they cannot become one polygon",
                elements={"vertices": [v]},
            )
        pts = self._pos[loop]
        nrm = poly.newell_normal(pts)
        if poly.unit(nrm) is None or not poly.is_simple(poly.project(pts, nrm)):
            raise EditError(
                "DISSOLVE_INVALID_FACE",
                f"the merged face around vertex {v} would be degenerate or self-intersecting",
                elements={"vertices": [v]},
            )
        keep = min(f for f, *_ in fan)
        groups = {self._group.get(f, 0) for f, *_ in fan}
        if len(groups) > 1:
            self._diagnose(
                Diagnostic(
                    "FACE_GROUP_CHANGED",
                    WARNING,
                    f"faces of groups {sorted(groups)} were merged into face {keep}, which "
                    f"keeps group {self._group.get(keep, 0)}",
                    None,
                    len(fan),
                    tuple(sorted(f for f, *_ in fan)[:10]),
                )
            )
        for f, *_ in fan:
            if f != keep:
                self._delete_face(f)
        self._replace_face(keep, loop, self._split_corners(corners))
        if self._alive[v] and not self._vert_edges[v]:
            self._delete_vertex(v)
        return keep

    @operation
    def dissolve_faces(self, faces: Elements):
        """Merge each connected region of faces into one polygon: its outline.

        The region must be a disk: one outline that visits no vertex twice
        (``DISSOLVE_INVALID_FACE`` otherwise). The merged face keeps the region's lowest
        face id and group; vertices inside the region are removed.
        """
        fids = self._elements(faces, "faces")
        out = []
        for region in self._components(fids):
            if len(region) == 1:
                out.append(region[0])
                continue
            boundary = self._region_boundary_directed(region)  # (a, b, face) runs a -> b
            nxt = {}
            for a, b, f in boundary:
                if a in nxt:
                    raise EditError(
                        "DISSOLVE_INVALID_FACE",
                        f"the region's outline passes vertex {a} twice",
                        elements={"vertices": [a]},
                    )
                nxt[a] = (b, f)
            if not nxt:
                raise EditError(
                    "DISSOLVE_INVALID_FACE",
                    "the region is a closed shell; it has no outline",
                    elements={"faces": region[:20]},
                )
            start = min(nxt)
            loop, owners = [start], [nxt[start][1]]
            v = nxt[start][0]
            while v != start:
                if v not in nxt or len(loop) > len(nxt):
                    raise EditError(  # pragma: no cover - a consistent outline closes
                        "DISSOLVE_INVALID_FACE", "the region's outline does not close"
                    )
                loop.append(v)
                owners.append(nxt[v][1])
                v = nxt[v][0]
            if len(loop) != len(nxt):
                raise EditError(
                    "DISSOLVE_INVALID_FACE",
                    "the region has holes (several outlines)",
                    elements={"faces": region[:20]},
                )
            corners = [
                self._corners_of(f)[self._faces[f].index(x)]
                for x, f in zip(loop, owners, strict=True)
            ]
            pts = self._pos[loop]
            nrm = poly.newell_normal(pts)
            if poly.unit(nrm) is None or not poly.is_simple(poly.project(pts, nrm)):
                raise EditError(
                    "DISSOLVE_INVALID_FACE",
                    "the merged face would be degenerate or self-intersecting",
                    elements={"faces": region[:20]},
                )
            keep = region[0]
            groups = {self._group.get(f, 0) for f in region}
            if len(groups) > 1:
                self._diagnose(
                    Diagnostic(
                        "FACE_GROUP_CHANGED",
                        WARNING,
                        f"faces of groups {sorted(groups)} were merged into face {keep}, which "
                        f"keeps group {self._group.get(keep, 0)}",
                        None,
                        len(region),
                        tuple(region[:10]),
                    )
                )
            inner_vertices = {x for f in region for x in self._faces[f]} - set(loop)
            for f in region[1:]:
                self._delete_face(f)
            self._replace_face(keep, loop, self._split_corners(corners))
            self._delete_loose(inner_vertices)
            out.append(keep)
        return {"faces": Selection.of(faces=out)}

    @operation
    def limited_dissolve(
        self,
        angle: float = 5.0,
        *,
        faces: Elements | None = None,
        delimit: Iterable[str] = ("groups", "seams", "sharp"),
    ):
        """Dissolve edges between faces whose normals differ by at most ``angle`` degrees,
        then vertices with two edges forming a straight line within ``angle``.

        ``delimit`` keeps edges between different face groups (``"groups"``), across UV
        or normal seams (``"seams"``) and sharp edges (``"sharp"``).
        """
        a = float(angle)
        if not 0.0 <= a <= 180.0:
            raise ValidationError("angle must be between 0 and 180 degrees")
        limits = set(delimit)
        unknown = limits - {"groups", "seams", "sharp"}
        if unknown:
            raise ValidationError(f"unknown delimit values: {sorted(unknown)}")
        pool = set(self.face_ids() if faces is None else self._elements(faces, "faces"))
        cos_limit = math.cos(math.radians(a))
        candidates = []
        for e in sorted({e for f in pool for e in self.face_edges(f)}):
            fs = self._edge_faces[e]
            if len(fs) != 2 or not set(fs) <= pool:
                continue
            if "sharp" in limits and e in self._sharp:
                continue
            if "groups" in limits and self._group.get(fs[0], 0) != self._group.get(fs[1], 0):
                continue
            if "seams" in limits and not self._corner_values_agree(e):
                continue
            n0, n1 = self._face_normal_vec(fs[0]), self._face_normal_vec(fs[1])
            if n0 is None or n1 is None:
                continue
            c = max(-1.0, min(1.0, float(np.dot(n0, n1))))
            if c >= cos_limit - 1e-12:
                candidates.append((math.acos(c), e))
        dissolved = []
        for _, e in sorted(candidates):
            if e not in self._edge_verts or len(self._edge_faces[e]) != 2:
                continue
            try:
                self._dissolve_edge(e)
            except EditError:
                continue
            dissolved.append(e)
        removed = []
        for v in self.vertex_ids():
            if len(self._vert_edges[v]) != 2:
                continue
            ea, eb = sorted(self._vert_edges[v])
            if self._edge_faces[ea] != self._edge_faces[eb]:
                continue
            others = [x for e in (ea, eb) for x in self._edge_verts[e] if x != v]
            d0 = poly.unit(self._pos[others[0]] - self._pos[v])
            d1 = poly.unit(self._pos[others[1]] - self._pos[v])
            if d0 is None or d1 is None:
                continue
            if float(np.dot(d0, -d1)) < cos_limit - 1e-12:
                continue
            fs = self._edge_faces[ea]
            if any(len(self._faces[f]) <= 3 for f in fs):
                continue
            try:
                for f in fs:
                    loop = self._faces[f]
                    k = loop.index(v)
                    corners = self._corners_of(f)
                    self._replace_face(
                        f,
                        loop[:k] + loop[k + 1 :],
                        self._split_corners(corners[:k] + corners[k + 1 :]),
                    )
            except (EditError, AssertionError):
                raise
            if not self._vert_edges[v]:
                self._delete_vertex(v)
                removed.append(v)
        return {"dissolved_edges": dissolved, "dissolved_vertices": removed}

    def _corner_values_agree(self, e: int) -> bool:
        f, g = self._edge_faces[e]
        lf, lg = self._faces[f], self._faces[g]
        for v in self._edge_verts[e]:
            kf, kg = lf.index(v), lg.index(v)
            for layer in self._layers.values():
                x, y = layer.data[f][kf], layer.data[g][kg]
                if any(abs(p - q) > 1e-9 for p, q in zip(x, y, strict=True)):
                    return False
        return True

    # -- connect ---------------------------------------------------------------------

    @operation
    def connect_vertex_path(self, vertices: Iterable[int]):
        """Join consecutive vertices of an ordered path with new edges, splitting the face
        each pair shares (pairs already joined by an edge are skipped)."""
        if isinstance(vertices, (int, np.integer, Selection)):
            raise ValidationError("vertices must be an ordered list of vertex ids")
        path = [int(v) for v in vertices]
        self._require_vertices(path)
        if len(path) < 2:
            raise EditError("TOO_FEW_ELEMENTS", "a path needs at least 2 vertices")
        edges = []
        for a, b in zip(path, path[1:], strict=False):
            e = self.edge_between(a, b)
            if e is not None:
                edges.append(e)
                continue
            shared = sorted(self._faces_of_vertex(a) & self._faces_of_vertex(b))
            if not shared:
                raise EditError(
                    "CONNECT_NO_SHARED_FACE",
                    f"vertices {a} and {b} share no face",
                    elements={"vertices": [a, b]},
                    hint="insert vertices on the edges between them first (split_edge)",
                )
            if len(shared) > 1:
                raise EditError(
                    "AMBIGUOUS_CONNECTION",
                    f"vertices {a} and {b} share {len(shared)} faces",
                    elements={"faces": shared},
                )
            out = nested_data(self.split_face(shared[0], a, b))
            edges.append(out["edge"])
        return {"edges": Selection.of(edges=edges)}

    @operation
    def connect_vertices(self, vertices: Elements):
        """Split every face holding exactly two of the selected vertices (not already
        joined by an edge) between them. A face holding more raises
        ``AMBIGUOUS_CONNECTION`` (use :meth:`connect_vertex_path`)."""
        vids = set(self._elements(vertices, "vertices"))
        plans = []
        for f in sorted({f for v in vids for f in self._faces_of_vertex(v)}):
            inside = [v for v in self._faces[f] if v in vids]
            if len(inside) > 2:
                raise EditError(
                    "AMBIGUOUS_CONNECTION",
                    f"face {f} holds {len(inside)} of the selected vertices",
                    elements={"faces": [f]},
                    hint="use connect_vertex_path with an ordered path",
                )
            if len(inside) == 2 and self.edge_between(*inside) is None:
                plans.append((f, inside[0], inside[1]))
        edges = []
        for f, a, b in plans:
            edges.append(nested_data(self.split_face(f, a, b))["edge"])
        return {"edges": Selection.of(edges=edges)}

    # -- cleanup ---------------------------------------------------------------------

    @operation
    def cleanup(
        self,
        *,
        merge_distance: float | None = None,
        zero_length: bool = True,
        duplicates: bool = True,
        loose: bool = True,
        fix_winding: bool = False,
        fill_holes: bool = False,
    ):
        """Explicit repairs, in this order, each reported in ``data["report"]``:

        ``merge_distance`` (weld vertices closer than it), ``zero_length`` (collapse
        zero-length edges), ``duplicates`` (delete faces with the same vertices as a
        lower-id face), ``loose`` (delete vertices without faces), ``fix_winding``
        (:meth:`recalculate_normals`), ``fill_holes`` (close boundary loops). Each entry
        says how many elements it changed (``fixed``, with ids) and how many problems of
        its kind remain (``remaining``). Zero-area faces and non-manifold elements are
        only counted (``remaining``): repairing them needs a decision this method does
        not guess.
        """
        report = []
        if merge_distance is not None:
            out = nested_data(self.merge_by_distance(float(merge_distance)))
            report.append(
                {
                    "step": "merge_by_distance",
                    "fixed": out["merged_count"],
                    "ids": [g[1:] for g in out["clusters"]][:20],
                }
            )
        if zero_length:
            tiny = 1e-12 * self._scale()
            short = [
                e
                for e in self.edge_ids()
                if np.linalg.norm(
                    self._pos[self._edge_verts[e][0]] - self._pos[self._edge_verts[e][1]]
                )
                <= tiny
            ]
            if short:
                self.collapse_edges(short)
            report.append({"step": "zero_length_edges", "fixed": len(short), "ids": short[:20]})
        if duplicates:
            seen: dict[frozenset, int] = {}
            dup = []
            for f in self.face_ids():
                key = frozenset(self._faces[f])
                if key in seen:
                    dup.append(f)
                else:
                    seen[key] = f
            if dup:
                self._delete_faces_and_loose(dup, keep_loose=True)
            report.append({"step": "duplicate_faces", "fixed": len(dup), "ids": dup[:20]})
        if fix_winding:
            out = nested_data(self.recalculate_normals())
            report.append(
                {
                    "step": "fix_winding",
                    "fixed": len(out["flipped"]),
                    "ids": out["flipped"][:20],
                }
            )
        if fill_holes:
            boundary = sorted(e for e, fs in self._edge_faces.items() if len(fs) == 1)
            made = (
                nested_data(self.fill_holes(boundary))["faces"].faces if boundary else frozenset()
            )
            report.append({"step": "fill_holes", "fixed": len(made), "ids": sorted(made)[:20]})
        if loose:
            lone = [v for v in self.vertex_ids() if not self._vert_edges[v]]
            for v in lone:
                self._delete_vertex(v)
            report.append({"step": "loose_vertices", "fixed": len(lone), "ids": lone[:20]})
        eps = self._area_eps()
        zero = [f for f in self.face_ids() if poly.polygon_area(self._pos[self._faces[f]]) <= eps]
        non_manifold = sorted(e for e, fs in self._edge_faces.items() if len(fs) > 2)
        report.append(
            {"step": "zero_area_faces", "fixed": 0, "remaining": len(zero), "ids": zero[:20]}
        )
        report.append(
            {
                "step": "non_manifold_edges",
                "fixed": 0,
                "remaining": len(non_manifold),
                "ids": non_manifold[:20],
            }
        )
        if zero or non_manifold:
            self._diagnose(
                Diagnostic(
                    "CLEANUP_UNRESOLVED",
                    WARNING,
                    "zero-area faces or non-manifold edges remain (cleanup reports them only)",
                    None,
                    len(zero) + len(non_manifold),
                    tuple((zero + non_manifold)[:10]),
                )
            )
        else:
            self._diagnose(Diagnostic("CLEANUP_DONE", INFO, "cleanup finished", None, 0, ()))
        return {"report": report}
