"""Topology operations: creation, deletion, merging, splitting, dissolving, triangulation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import int_at_least, scale3, vec3
from ..primitives import _CUBE_FACES, _QUAD_CORNERS, _QUAD_UVS
from . import _polygon as poly
from .core import NORMAL, UV, Elements, Selection, operation


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return [sorted(g) for g in sorted(out.values()) if len(g) > 1]


class TopologyMixin:
    # ------------------------------------------------------------------ #
    # Creation
    # ------------------------------------------------------------------ #

    @operation
    def create_vertex(self, position: Iterable[float], *, space: str = "local"):
        """Add a loose vertex; returns ``data["vertex"]``."""
        p = np.asarray(vec3(position, "position"))
        if space == "world":
            p = self._world_to_local(p[None])[0]
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        return {"vertex": self._new_vertex(p)}

    @operation
    def create_face(
        self,
        vertices: Iterable[int],
        *,
        smooth: bool | None = None,
        allow_non_manifold: bool = False,
    ):
        """Create a polygon from existing vertices given in counter-clockwise (front) order.

        Rejects degenerate or self-intersecting polygons, duplicates of existing
        faces, and -- unless ``allow_non_manifold=True`` -- faces that would put
        a third face on an edge or traverse an existing edge in the same
        direction as its face (inconsistent winding; reverse the order).
        """
        if isinstance(vertices, (int, np.integer, Selection)):
            raise ValidationError("vertices must be an ordered list of vertex ids")
        loop = [int(v) for v in vertices]
        self._require_vertices(loop)
        if len(loop) < 3 or len(set(loop)) != len(loop):
            raise EditError("INVALID_POLYGON", "a face needs at least 3 distinct vertices")
        self._check_polygon(loop, "the new face")
        for f in self._faces_of_vertex(loop[0]):
            if set(self._faces[f]) == set(loop):
                raise EditError(
                    "DUPLICATE_FACE",
                    "a face with these vertices already exists",
                    elements={"faces": [f]},
                )
        if not allow_non_manifold:
            self._check_new_loop_edges(loop)
        neighbours = sorted(
            {
                g
                for key in self._loop_edges(loop)
                if key in self._edge_by_pair
                for g in self._edge_faces[self._edge_by_pair[key]]
            }
        )
        if smooth is None:
            smooth = bool(neighbours) and self._smooth[neighbours[0]]
        corners: dict[str, list[tuple]] = {}
        if UV in self._layers:
            uvs, undefined = [], []
            for v in loop:
                values = {
                    self._layers[UV].data[f][self._faces[f].index(v)]
                    for f in self._faces_of_vertex(v)
                }
                if len(values) == 1:
                    uvs.append(values.pop())
                else:
                    uvs.append((0.0, 0.0))
                    undefined.append(v)
            corners[UV] = uvs
            if undefined:
                self._diagnose(
                    Diagnostic(
                        "UVS_UNDEFINED",
                        WARNING,
                        "the new face's UVs could not be derived for some corners (vertex has no "
                        "single existing UV); they were set to (0, 0)",
                        None,
                        len(undefined),
                        tuple(undefined[:10]),
                    )
                )
        fid = self._new_face(loop, corners, smooth=smooth)
        return {"face": fid}

    @operation
    def add_box(
        self,
        size: float | Iterable[float] = 1.0,
        *,
        center: Iterable[float] = (0.0, 0.0, 0.0),
    ):
        """Add a closed box of 6 quads (8 shared vertices), flat-shaded, with per-face UVs.

        The layout matches :func:`aether3d.create_cube`. Returns ``data["faces"]``
        and ``data["vertices"]`` selections.
        """
        half = np.asarray(scale3(size, "size")) / 2.0
        if np.any(half <= 0):
            raise ValidationError("box size must be positive")
        c = np.asarray(vec3(center, "center"))
        vertex_at: dict[tuple, int] = {}
        faces = []
        for normal, u_axis, v_axis in _CUBE_FACES:
            n, u, v = (np.array(a, dtype=np.float64) for a in (normal, u_axis, v_axis))
            loop = []
            for cu, cv in _QUAD_CORNERS:
                key = tuple((n + cu * u + cv * v).astype(int).tolist())
                if key not in vertex_at:
                    vertex_at[key] = self._new_vertex(c + np.asarray(key) * half)
                loop.append(vertex_at[key])
            corners = {UV: list(_QUAD_UVS)} if UV in self._layers else {}
            if NORMAL in self._layers:
                corners[NORMAL] = [tuple(n.tolist())] * 4
            faces.append(self._new_face(loop, corners, smooth=False))
        return {"faces": Selection.of(faces=faces), "vertices": Selection.of(vertex_at.values())}

    @operation
    def add_grid(
        self,
        x_segments: int = 1,
        y_segments: int = 1,
        size: float | Iterable[float] = 1.0,
        *,
        center: Iterable[float] = (0.0, 0.0, 0.0),
    ):
        """Add a flat grid of quads in the XY plane facing +Z, UVs spanning 0..1."""
        nx = int_at_least(x_segments, 1, "x_segments")
        ny = int_at_least(y_segments, 1, "y_segments")
        sx, sy, _ = scale3(size if not isinstance(size, (list, tuple)) else (*size, 1.0), "size")
        if sx <= 0 or sy <= 0:
            raise ValidationError("grid size must be positive")
        if (nx + 1) * (ny + 1) > self.max_elements:
            raise EditError("TOO_MANY_ELEMENTS", "the grid is too large")
        c = np.asarray(vec3(center, "center"))
        ids = [
            [
                self._new_vertex(c + ((i / nx - 0.5) * sx, (j / ny - 0.5) * sy, 0.0))
                for i in range(nx + 1)
            ]
            for j in range(ny + 1)
        ]
        faces = []
        for j in range(ny):
            for i in range(nx):
                loop = [ids[j][i], ids[j][i + 1], ids[j + 1][i + 1], ids[j + 1][i]]
                corners = {}
                if UV in self._layers:
                    corners[UV] = [
                        (i / nx, j / ny),
                        ((i + 1) / nx, j / ny),
                        ((i + 1) / nx, (j + 1) / ny),
                        (i / nx, (j + 1) / ny),
                    ]
                faces.append(self._new_face(loop, corners, smooth=False))
        return {
            "faces": Selection.of(faces=faces),
            "vertices": Selection.of(v for row in ids for v in row),
        }

    # ------------------------------------------------------------------ #
    # Deletion, orientation and shading
    # ------------------------------------------------------------------ #

    @operation
    def delete_faces(self, faces: Elements, *, keep_loose_vertices: bool = False):
        """Delete faces; their vertices are deleted too if no other face uses them."""
        fids = self._elements(faces, "faces")
        self._delete_faces_and_loose(fids, keep_loose_vertices)
        return {}

    @operation
    def delete_edges(self, edges: Elements, *, keep_loose_vertices: bool = False):
        """Delete edges and every face that uses them."""
        eids = self._elements(edges, "edges")
        fids = sorted({f for e in eids for f in self._edge_faces[e]})
        self._delete_faces_and_loose(fids, keep_loose_vertices)
        return {}

    @operation
    def delete_vertices(self, vertices: Elements):
        """Delete vertices and every face that uses them."""
        vids = self._elements(vertices, "vertices")
        fids = sorted({f for v in vids for f in self._faces_of_vertex(v)})
        affected = self._delete_faces_and_loose(fids, keep_loose=True)
        self._delete_loose(affected | set(vids))
        return {}

    @operation
    def flip_faces(self, faces: Elements):
        """Reverse the winding (front side) of faces; their stored normals are negated."""
        fids = self._elements(faces, "faces")
        for f in fids:
            loop, corners = self._faces[f], self._corners_of(f)
            order = [0] + list(range(len(loop) - 1, 0, -1))
            new_corners = [dict(corners[i]) for i in order]
            if NORMAL in self._layers:
                for c in new_corners:
                    c[NORMAL] = tuple(-x for x in c[NORMAL])
            self._replace_face(
                f, [loop[i] for i in order], self._split_corners(new_corners), dirty=False
            )
        return {"faces": Selection.of(faces=fids)}

    @operation
    def set_smooth(self, faces: Elements, smooth: bool = True):
        """Mark faces smooth- or flat-shaded (their normals are recomputed)."""
        fids = self._elements(faces, "faces")
        for f in fids:
            self._set_smooth(f, smooth)
        self._mark_dirty(fids)
        return {}

    @operation
    def set_sharp(self, edges: Elements, sharp: bool = True):
        """Mark edges sharp (normals do not blend across them) or smooth."""
        eids = self._elements(edges, "edges")
        for e in eids:
            self._set_sharp(e, sharp)
        self._mark_dirty({f for e in eids for f in self._edge_faces[e]})
        return {}

    # ------------------------------------------------------------------ #
    # Merging
    # ------------------------------------------------------------------ #

    @operation
    def merge_vertices(
        self,
        vertices: Elements,
        *,
        at: str | Iterable[float] = "center",
        keep: int | None = None,
    ):
        """Merge vertices into one.

        The surviving vertex is ``keep`` (default: the lowest id). It is placed
        at ``at``: ``"center"`` (mean position), ``"keep"`` (the survivor's own
        position) or an explicit point. Faces that collapse below 3 vertices
        are deleted; corner attributes of merged corners are averaged. Raises
        ``MERGE_INVALID_FACE`` if a face would visit the merged vertex twice
        (e.g. merging opposite corners of a quad).
        """
        vids = self._elements(vertices, "vertices")
        if len(vids) < 2:
            raise EditError(
                "TOO_FEW_ELEMENTS", "merging needs at least 2 vertices", elements={"vertices": vids}
            )
        survivor = vids[0] if keep is None else int(keep)
        if survivor not in vids:
            raise ValidationError("keep must be one of the merged vertices")
        if isinstance(at, str):
            if at == "center":
                target = self._pos[vids].mean(axis=0)
            elif at == "keep":
                target = self._pos[survivor].copy()
            else:
                raise ValidationError(f"at must be 'center', 'keep' or a point, got {at!r}")
        else:
            target = np.asarray(vec3(at, "at"))
        removed = self._merge_group(vids, survivor, target)
        return {"vertex": survivor, "removed_faces": removed}

    @operation
    def merge_by_distance(self, distance: float, vertices: Elements | None = None):
        """Merge vertices closer than ``distance`` (clusters are transitive), at their centers."""
        d = float(distance)
        if not d >= 0:
            raise ValidationError("distance must be >= 0")
        vids = self.vertex_ids() if vertices is None else self._vertices_of(vertices)
        uf = _UnionFind()
        cell = d if d > 0 else 1.0
        grid: dict[tuple[int, int, int], list[int]] = {}
        keys = np.floor(self._pos[vids] / cell).astype(np.int64)
        for v, key in zip(vids, map(tuple, keys.tolist()), strict=True):
            grid.setdefault(key, []).append(v)
        offsets = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        for key, members in grid.items():
            near = [
                u
                for o in offsets
                for u in grid.get((key[0] + o[0], key[1] + o[1], key[2] + o[2]), ())
            ]
            near_pos = self._pos[near]
            for v in members:
                close = np.linalg.norm(near_pos - self._pos[v], axis=1) <= d
                for u in np.asarray(near)[close]:
                    if int(u) != v:
                        uf.union(v, int(u))
        groups = uf.groups()
        for group in groups:
            self._merge_group(group, group[0], self._pos[group].mean(axis=0))
        return {
            "clusters": [list(g) for g in groups],
            "merged_count": sum(len(g) - 1 for g in groups),
        }

    @operation
    def collapse_edges(self, edges: Elements):
        """Collapse edges to points; connected edges merge into one vertex at their center."""
        eids = self._elements(edges, "edges")
        uf = _UnionFind()
        for e in eids:
            uf.union(*self._edge_verts[e])
        groups = uf.groups()
        for group in groups:
            if all(self.has_vertex(v) for v in group):
                self._merge_group(group, group[0], self._pos[group].mean(axis=0))
        return {"vertices": [g[0] for g in groups]}

    def _merge_group(self, vids: list[int], survivor: int, target: np.ndarray) -> list[int]:
        merged = set(vids)
        faces = sorted({f for v in vids for f in self._faces_of_vertex(v)})
        plans: list[tuple[int, list[int] | None, list[dict]]] = []
        for f in faces:
            loop, corners = self._faces[f], self._corners_of(f)
            new_loop: list[int] = []
            new_corners: list[list[dict]] = []
            for v, c in zip(loop, corners, strict=True):
                v2 = survivor if v in merged else v
                if new_loop and new_loop[-1] == v2:
                    new_corners[-1].append(c)
                else:
                    new_loop.append(v2)
                    new_corners.append([c])
            if len(new_loop) > 1 and new_loop[0] == new_loop[-1]:
                new_corners[0] = new_corners.pop() + new_corners[0]
                new_loop.pop()
            if len(new_loop) < 3:
                plans.append((f, None, []))
                continue
            if len(set(new_loop)) != len(new_loop):
                raise EditError(
                    "MERGE_INVALID_FACE",
                    f"face {f} would use the merged vertex more than once",
                    elements={"faces": [f]},
                    hint="split or delete that face first, or merge a different set of vertices",
                )
            mixed = [self._mix([(c, 1.0 / len(group)) for c in group]) for group in new_corners]
            plans.append((f, new_loop, mixed))
        removed_faces = []
        for f, new_loop, mixed in plans:
            if new_loop is None:
                self._delete_face(f)
                removed_faces.append(f)
            else:
                self._replace_face(f, new_loop, self._split_corners(mixed))
        for v in vids:
            if v != survivor and self._alive[v]:
                if self._vert_edges[v]:  # pragma: no cover - guarded by the plan above
                    raise AssertionError("merged vertex still in use")
                self._delete_vertex(v)
        self._set_positions([survivor], np.asarray(target)[None])
        self._report_merge_side_effects({f for f, loop, _ in plans if loop is not None})
        return removed_faces

    def _report_merge_side_effects(self, faces: set[int]) -> None:
        eps = self._area_eps()
        zero = sorted(f for f in faces if poly.polygon_area(self._pos[self._faces[f]]) <= eps)
        if zero:
            self._diagnose(
                Diagnostic(
                    "ZERO_AREA_FACES",
                    WARNING,
                    "the merge left zero-area faces",
                    None,
                    len(zero),
                    tuple(zero[:10]),
                )
            )
        seen: dict[frozenset, int] = {}
        dup = []
        for f in sorted(faces):
            key = frozenset(self._faces[f])
            for g in self._faces_of_vertex(self._faces[f][0]):
                if g != f and frozenset(self._faces[g]) == key:
                    dup.append(f)
                    break
            seen[key] = f
        if dup:
            self._diagnose(
                Diagnostic(
                    "DUPLICATE_FACES",
                    WARNING,
                    "the merge made faces that duplicate other faces",
                    None,
                    len(dup),
                    tuple(dup[:10]),
                )
            )
        nm = sorted(
            {
                self._edge_by_pair[key]
                for f in faces
                for key in self._loop_edges(self._faces[f])
                if len(self._edge_faces[self._edge_by_pair[key]]) > 2
            }
        )
        if nm:
            self._diagnose(
                Diagnostic(
                    "NON_MANIFOLD_EDGES",
                    WARNING,
                    "the merge produced edges shared by more than two faces",
                    None,
                    len(nm),
                    tuple(nm[:10]),
                )
            )

    # ------------------------------------------------------------------ #
    # Dissolving
    # ------------------------------------------------------------------ #

    @operation
    def dissolve_edges(self, edges: Elements):
        """Remove edges between two faces, merging each pair of faces into one polygon.

        The merged face keeps the lower face id. Each edge must have exactly
        two consistently-wound faces that share no other vertex, and the
        merged polygon must be valid. Edges are processed in id order.
        """
        eids = self._elements(edges, "edges")
        merged: dict[int, int] = {}
        for e in eids:
            if e not in self._edge_verts:
                raise EditError(
                    "INVALID_ELEMENT",
                    f"edge {e} no longer exists (a previous dissolve "
                    "in this call merged its faces)",
                    elements={"edges": [e]},
                )
            merged[e] = self._dissolve_edge(e)
        return {"faces": Selection.of(faces=set(merged.values()) & set(self._faces))}

    def _dissolve_edge(self, e: int, check: bool = True) -> int:
        faces = self._edge_faces[e]
        if len(faces) != 2:
            raise EditError(
                "EDGE_NOT_INTERIOR",
                f"edge {e} has {len(faces)} faces; dissolving needs exactly 2",
                elements={"edges": [e]},
            )
        f, g = sorted(faces)
        a, b = self._edge_verts[e]
        lf, lg = self._faces[f], self._faces[g]
        if lf[(lf.index(a) + 1) % len(lf)] != b:
            a, b = b, a  # make f traverse a -> b
        if lg[(lg.index(b) + 1) % len(lg)] != a:
            raise EditError(
                "INCONSISTENT_WINDING",
                f"faces {f} and {g} are wound inconsistently across edge {e}",
                elements={"faces": [f, g]},
                hint="flip one of the faces first",
            )
        cf, cg = self._corners_of(f), self._corners_of(g)
        i = lf.index(b)
        f_rot = [lf[(i + k) % len(lf)] for k in range(len(lf))]
        cf_rot = [cf[(i + k) % len(lf)] for k in range(len(lf))]
        j = lg.index(a)
        g_rot = [lg[(j + k) % len(lg)] for k in range(len(lg))]
        cg_rot = [cg[(j + k) % len(lg)] for k in range(len(lg))]
        loop = f_rot + g_rot[1:-1]
        corners = cf_rot + cg_rot[1:-1]
        if len(set(loop)) != len(loop):
            raise EditError(
                "DISSOLVE_INVALID_FACE",
                f"faces {f} and {g} share more than edge {e}; the merged face would "
                "repeat a vertex",
                elements={"faces": [f, g]},
            )
        try:
            if check:
                poly.triangulate(self._pos[loop])
        except ValueError:
            raise EditError(
                "DISSOLVE_INVALID_FACE",
                f"merging faces {f} and {g} gives an invalid polygon",
                elements={"faces": [f, g]},
            ) from None
        self._delete_face(g)
        self._replace_face(f, loop, self._split_corners(corners), dirty=False)
        return f

    @operation
    def join_triangles(self, faces: Elements | None = None, *, max_angle: float = 1.0):
        """Merge pairs of adjacent triangles into quads (the inverse of triangulation).

        Pairs qualify when their normals differ by at most ``max_angle``
        degrees, the quad is convex, and their UVs/normals agree along the
        shared edge (so no attribute is lost). Best-shaped pairs are joined
        first; ties resolve by edge id.
        """
        pool = set(self.face_ids() if faces is None else self._elements(faces, "faces"))
        joined = self._join_pairs(pool, max_angle)
        return {"faces": Selection.of(faces=joined), "joined_count": len(joined)}

    def _join_pairs(self, pool: set[int], max_angle: float) -> list[int]:
        """Join qualifying triangle pairs within ``pool``; returns the resulting quads."""
        tris = {f for f in pool if len(self._faces[f]) == 3}
        candidates = []
        for e in sorted(
            {self._edge_by_pair[k] for f in tris for k in self._loop_edges(self._faces[f])}
        ):
            fs = self._edge_faces[e]
            if len(fs) != 2 or not (fs[0] in tris and fs[1] in tris):
                continue
            n0, n1 = self._face_normal_vec(fs[0]), self._face_normal_vec(fs[1])
            if n0 is None or n1 is None:
                continue
            angle = float(np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1, 1))))
            if angle > max_angle or not self._attributes_agree_across(e):
                continue
            quad = self._joined_loop(e)
            if quad is None:
                continue
            quad_normal = poly.newell_normal(self._pos[quad])
            if poly.unit(quad_normal) is None:
                continue
            pts = poly.project(self._pos[quad], quad_normal)
            crosses = [
                (pts[(k + 1) % 4][0] - pts[k][0]) * (pts[(k + 2) % 4][1] - pts[k][1])
                - (pts[(k + 1) % 4][1] - pts[k][1]) * (pts[(k + 2) % 4][0] - pts[k][0])
                for k in range(4)
            ]
            if min(crosses) <= 1e-12 * self._scale() ** 2:
                continue  # not strictly convex
            sides = [
                np.linalg.norm(self._pos[quad[(k + 1) % 4]] - self._pos[quad[k]]) for k in range(4)
            ]
            shape = abs(np.log(max(sides) / max(min(sides), 1e-300)))
            candidates.append((angle + shape, e))
        used: set[int] = set()
        joined = []
        for _, e in sorted(candidates):
            fs = self._edge_faces.get(e)
            if not fs or used & set(fs):
                continue
            used.update(fs)
            joined.append(self._dissolve_edge(e, check=False))  # strictly convex: valid
        return joined

    def _joined_loop(self, e: int) -> list[int] | None:
        f, g = self._edge_faces[e]
        a, b = self._edge_verts[e]
        lf, lg = self._faces[f], self._faces[g]
        if lf[(lf.index(a) + 1) % 3] != b:
            a, b = b, a
        if lg[(lg.index(b) + 1) % 3] != a:
            return None
        c = next(v for v in lf if v not in (a, b))
        d = next(v for v in lg if v not in (a, b))
        return None if c == d else [a, d, b, c]

    def _attributes_agree_across(self, e: int) -> bool:
        f, g = self._edge_faces[e]
        lf, lg = self._faces[f], self._faces[g]
        for v in self._edge_verts[e]:
            kf, kg = lf.index(v), lg.index(v)
            for layer in self._layers.values():
                x, y = layer.data[f][kf], layer.data[g][kg]
                if any(abs(a - b) > 1e-9 for a, b in zip(x, y, strict=True)):
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #

    @operation
    def split_edges(self, edges: Elements, cuts: int = 1):
        """Insert ``cuts`` evenly spaced vertices on each edge (adjacent faces gain them).

        New vertices are ordered from the edge's lower vertex id to its higher one.
        Returns ``data["vertices"]`` mapping each edge id to its new vertex ids.
        """
        eids = self._elements(edges, "edges")
        cuts = int_at_least(cuts, 1, "cuts")
        if cuts > 1000:
            raise ValidationError("cuts must be at most 1000")
        params = [k / (cuts + 1) for k in range(1, cuts + 1)]
        return {"vertices": {e: self._split_edge(e, self._edge_verts[e][0], params) for e in eids}}

    @operation
    def split_edge(self, edge: int, factor: float = 0.5, *, from_vertex: int | None = None):
        """Insert one vertex on ``edge`` at ``factor`` (0..1) from ``from_vertex``.

        ``from_vertex`` defaults to the edge's lower vertex id.
        """
        (e,) = self._elements(edge, "edges")
        t = float(factor)
        if not 0.0 < t < 1.0:
            raise ValidationError("factor must be strictly between 0 and 1")
        a, b = self._edge_verts[e]
        start = a if from_vertex is None else int(from_vertex)
        if start not in (a, b):
            raise ValidationError(f"from_vertex must be {a} or {b}")
        (v,) = self._split_edge(e, start, [t])
        return {"vertex": v}

    def _split_edge(self, e: int, start: int, params: list[float]) -> list[int]:
        a, b = self._edge_verts[e]
        end = b if start == a else a
        pa, pb = self._pos[start], self._pos[end]
        new = [self._new_vertex(pa + t * (pb - pa)) for t in params]
        for f in list(self._edge_faces[e]):
            loop, corners = self._faces[f], self._corners_of(f)
            i = loop.index(start)
            n = len(loop)
            if loop[(i + 1) % n] == end:  # face runs start -> end
                k, ts, ids, c0, c1 = i, params, new, corners[i], corners[(i + 1) % n]
            else:  # face runs end -> start
                k = loop.index(end)
                ts, ids = [1 - t for t in reversed(params)], list(reversed(new))
                c0, c1 = corners[k], corners[(k + 1) % n]
            inserted = [self._mix([(c0, 1 - t), (c1, t)]) for t in ts]
            new_loop = loop[: k + 1] + ids + loop[k + 1 :]
            new_corners = corners[: k + 1] + inserted + corners[k + 1 :]
            self._replace_face(f, new_loop, self._split_corners(new_corners), dirty=False)
        return new

    @operation
    def split_face(self, face: int, vertex_a: int, vertex_b: int):
        """Split a face in two with a new edge between two of its non-adjacent vertices.

        The part from ``vertex_a`` to ``vertex_b`` (in loop order) keeps the
        face id; ``data["face"]`` is the new face.
        """
        (f,) = self._elements(face, "faces")
        loop = self._faces[f]
        a, b = int(vertex_a), int(vertex_b)
        if a not in loop or b not in loop or a == b:
            raise EditError(
                "INVALID_ELEMENT",
                f"vertices {a} and {b} must be two different vertices of face {f}",
                elements={"faces": [f]},
            )
        n = len(loop)
        i, j = loop.index(a), loop.index(b)
        if (j - i) % n in (1, n - 1):
            raise EditError(
                "VERTICES_ADJACENT",
                f"vertices {a} and {b} are already joined by an edge of face {f}",
                elements={"vertices": [a, b]},
            )
        if self.edge_between(a, b) is not None:
            raise EditError(
                "NON_MANIFOLD_RESULT",
                f"an edge {a}-{b} already exists; splitting would give it a third face",
                elements={"vertices": [a, b]},
            )
        if poly.polygon_area(self._pos[loop]) <= self._area_eps():
            raise EditError("DEGENERATE_INPUT", f"face {f} has zero area", elements={"faces": [f]})
        pts = poly.project(self._pos[loop], poly.newell_normal(self._pos[loop]))
        if not poly.chord_inside(pts, i, j):
            raise EditError(
                "CHORD_OUTSIDE_FACE",
                f"the segment {a}-{b} does not lie inside face {f}",
                elements={"faces": [f]},
            )
        corners = self._corners_of(f)
        first = [(i + k) % n for k in range((j - i) % n + 1)]
        second = [(j + k) % n for k in range((i - j) % n + 1)]
        for part in (first, second):
            if poly.polygon_area(self._pos[[loop[k] for k in part]]) <= self._area_eps():
                raise EditError(
                    "DEGENERATE_RESULT",
                    "splitting would create a zero-area face",
                    elements={"faces": [f]},
                )
        self._replace_face(
            f,
            [loop[k] for k in first],
            self._split_corners([corners[k] for k in first]),
            dirty=False,
        )
        new = self._new_face(
            [loop[k] for k in second],
            self._split_corners([corners[k] for k in second]),
            smooth=self._smooth[f],
            dirty=False,
        )
        return {"face": new, "edge": self.edge_between(a, b)}

    @operation
    def rip_vertices(self, vertices: Elements):
        """Split each vertex so every face around it gets its own copy (faces disconnect there).

        The lowest-id face keeps the original vertex. ``data["vertices"]``
        maps each original vertex to its new copies.
        """
        vids = self._elements(vertices, "vertices")
        out = {}
        for v in vids:
            copies = []
            for f in sorted(self._faces_of_vertex(v))[1:]:
                c = self._new_vertex(self._pos[v])
                loop = [c if u == v else u for u in self._faces[f]]
                self._replace_face(f, loop, self._split_corners(self._corners_of(f)), dirty=False)
                copies.append(c)
            out[v] = copies
        return {"vertices": out}

    @operation
    def separate_faces(self, faces: Elements):
        """Disconnect a region of faces from the rest of the mesh (vertices are duplicated).

        ``data["vertices"]`` maps each duplicated vertex to its copy now used by the region.
        """
        fids = self._elements(faces, "faces")
        mapping = self._detach_region(fids)
        return {"vertices": mapping}

    def _detach_region(self, fids: list[int], also: set[int] | None = None) -> dict[int, int]:
        """Give the region its own copies of vertices it shares with other faces (and ``also``)."""
        region = set(fids)
        shared = sorted(
            {
                v
                for f in fids
                for v in self._faces[f]
                if any(g not in region for g in self._faces_of_vertex(v))
            }
        )
        shared = sorted(set(shared) | (also or set()))
        mapping = {v: self._new_vertex(self._pos[v]) for v in shared}
        for f in fids:
            loop = self._faces[f]
            if any(v in mapping for v in loop):
                self._replace_face(
                    f,
                    [mapping.get(v, v) for v in loop],
                    self._split_corners(self._corners_of(f)),
                    dirty=False,
                )
        return mapping

    @operation
    def triangulate_faces(self, faces: Elements | None = None):
        """Split faces with more than 3 vertices into triangles (ear clipping, winding kept).

        The first triangle keeps the face id. ``data["faces"]`` maps each
        original face to all of its triangles.
        """
        fids = self.face_ids() if faces is None else self._elements(faces, "faces")
        out: dict[int, list[int]] = {}
        for f in fids:
            loop = self._faces[f]
            if len(loop) == 3:
                out[f] = [f]
                continue
            try:
                tris = poly.triangulate(self._pos[loop])
            except ValueError as exc:
                raise EditError(
                    "TRIANGULATION_FAILED", f"face {f}: {exc}", elements={"faces": [f]}
                ) from None
            corners = self._corners_of(f)
            pieces = [f]
            self._replace_face(
                f,
                [loop[k] for k in tris[0]],
                self._split_corners([corners[k] for k in tris[0]]),
                dirty=False,
            )
            for tri in tris[1:]:
                pieces.append(
                    self._new_face(
                        [loop[k] for k in tri],
                        self._split_corners([corners[k] for k in tri]),
                        smooth=self._smooth[f],
                        dirty=False,
                    )
                )
            out[f] = pieces
        return {"faces": out}

    # ------------------------------------------------------------------ #
    # Shared checks
    # ------------------------------------------------------------------ #

    def _check_polygon(self, loop: list[int], what: str) -> None:
        pts3 = self._pos[loop]
        normal = poly.newell_normal(pts3)
        if np.linalg.norm(normal) / 2 <= self._area_eps():
            raise EditError("DEGENERATE_RESULT", f"{what} would have zero area")
        if not poly.is_simple(poly.project(pts3, normal)):
            raise EditError(
                "SELF_INTERSECTING_FACE",
                f"{what} would intersect itself",
                hint="check the vertex order",
            )

    def _check_new_loop_edges(self, loop: list[int]) -> None:
        n = len(loop)
        for k in range(n):
            a, b = loop[k], loop[(k + 1) % n]
            e = self.edge_between(a, b)
            if e is None:
                continue
            faces = self._edge_faces[e]
            if len(faces) >= 2:
                raise EditError(
                    "NON_MANIFOLD_RESULT",
                    f"edge {a}-{b} already has {len(faces)} faces",
                    elements={"edges": [e]},
                    hint="pass allow_non_manifold=True to force",
                )
            g = faces[0]
            lg = self._faces[g]
            if lg[(lg.index(a) + 1) % len(lg)] == b:
                raise EditError(
                    "INCONSISTENT_WINDING",
                    f"face {g} already runs {a}->{b}",
                    elements={"faces": [g]},
                    hint="reverse the vertex order",
                )

    def _delete_faces_and_loose(self, fids: list[int], keep_loose: bool) -> set[int]:
        """Delete faces and return their vertices (loose ones are deleted unless ``keep_loose``)."""
        affected = {v for f in fids for v in self._faces[f]}
        for f in fids:
            self._delete_face(f)
        if not keep_loose:
            self._delete_loose(affected)
        return affected

    def _delete_loose(self, vids: set[int]) -> None:
        for v in sorted(vids):
            if self._alive[v] and not self._vert_edges[v]:
                self._delete_vertex(v)
