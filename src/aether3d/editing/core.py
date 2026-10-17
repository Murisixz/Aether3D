"""The editable polygon mesh: storage, journaled mutation, transactions, queries.

Topology representation
-----------------------
``EditableMesh`` stores BMesh-style topology (vertices, edges, faces, and
per-face *corners*) in integer-keyed tables instead of pointer cycles:

* **vertices** -- positions in a growable ``float64`` NumPy array indexed by
  vertex id, plus an "alive" mask and, per vertex, the set of incident edges;
* **edges** -- an unordered vertex pair ``(a, b)`` with ``a < b``, an id, and
  the list of faces using it (its *radial* list: 1 face = boundary edge,
  2 = manifold, more = non-manifold);
* **faces** -- an ordered vertex loop (counter-clockwise seen from the front,
  so the front normal follows the right-hand rule), with per-corner
  attributes (UV, normal) stored alongside the loop, and per-face attributes.

Edges exist exactly while at least one face uses them. Arbitrary n-gons,
boundaries, non-manifold edges and vertices, and attribute seams (different
UVs/normals at the same vertex in different faces) are all representable.

Every change goes through a handful of journaled primitives. An operation
runs inside a transaction: if it raises, the journal is replayed backwards and
the mesh is exactly as before (same ids, same data). Successful operations
are recorded in a bounded history supporting ``undo()``/``redo()``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import _features
from ..diagnostics import INFO, Diagnostic
from ..errors import EditError, ValidationError
from ..mesh import Mesh
from . import _polygon as poly

UV = "uv"
NORMAL = "normal"


@dataclass
class CornerLayer:
    """A per-corner attribute (one value per face corner).

    ``normalize`` layers (normals) are renormalized after interpolation.
    Future corner attributes (vertex colors, extra UV sets) are added as new
    layers and are carried through every operation automatically.
    """

    name: str
    dim: int
    normalize: bool = False
    default: tuple[float, ...] | None = None
    data: dict[int, list[tuple[float, ...]]] = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    """An immutable set of element ids of one editing session.

    Ids are only meaningful for the :class:`EditableMesh` they came from.
    Combine selections with ``|``, ``&`` and ``-``.
    """

    vertices: frozenset[int] = frozenset()
    edges: frozenset[int] = frozenset()
    faces: frozenset[int] = frozenset()

    @classmethod
    def of(
        cls,
        vertices: Iterable[int] = (),
        edges: Iterable[int] = (),
        faces: Iterable[int] = (),
    ) -> Selection:
        return cls(frozenset(vertices), frozenset(edges), frozenset(faces))

    def __or__(self, other: Selection) -> Selection:
        return Selection(
            self.vertices | other.vertices, self.edges | other.edges, self.faces | other.faces
        )

    def __and__(self, other: Selection) -> Selection:
        return Selection(
            self.vertices & other.vertices, self.edges & other.edges, self.faces & other.faces
        )

    def __sub__(self, other: Selection) -> Selection:
        return Selection(
            self.vertices - other.vertices, self.edges - other.edges, self.faces - other.faces
        )

    def __len__(self) -> int:
        return len(self.vertices) + len(self.edges) + len(self.faces)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "vertices": sorted(self.vertices),
            "edges": sorted(self.edges),
            "faces": sorted(self.faces),
        }


Elements = Selection | Iterable[int] | int


@dataclass
class OperationResult:
    """Outcome of one editing operation.

    ``created``/``removed`` list element ids by kind (computed from the
    operation's journal, so they are always complete); ``modified_faces`` are
    faces that kept their id but changed their vertex loop. ``data`` holds
    operation-specific outputs (e.g. ``"side_faces"`` for extrusion), and
    ``diagnostics`` any findings (e.g. attributes that could not be preserved).
    """

    operation: str
    created: dict[str, list[int]]
    removed: dict[str, list[int]]
    modified_faces: list[int]
    data: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def selection(self, key: str = "created") -> Selection:
        """Created elements (default), or ``data[key]`` interpreted by kind."""
        if key == "created":
            return Selection.of(
                self.created["vertices"], self.created["edges"], self.created["faces"]
            )
        value = self.data[key]
        if isinstance(value, Selection):
            return value
        raise KeyError(f"result data {key!r} is not a selection")

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, Selection):
                return value.to_dict()
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set, frozenset)):
                items = sorted(value) if isinstance(value, (set, frozenset)) else value
                return [plain(v) for v in items]
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "operation": self.operation,
            "created": {k: list(v) for k, v in self.created.items()},
            "removed": {k: list(v) for k, v in self.removed.items()},
            "modified_faces": list(self.modified_faces),
            "data": plain(self.data),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass
class _HistoryEntry:
    name: str
    args: tuple
    kwargs: dict
    journal: list
    counters: tuple[int, int, int]
    result: OperationResult


def nested_data(result: Any) -> dict:
    """The ``data`` of an operation's result. Operations called from inside another
    operation share its transaction and return their ``data`` dict directly."""
    return result if isinstance(result, dict) else result.data


def operation(func: Callable) -> Callable:
    """Make a method an atomic, undoable operation returning an OperationResult.

    The wrapped function returns ``data`` (a dict) and may append diagnostics to
    ``self._op_diagnostics``. Nested operations share the outer transaction.
    """
    name = func.__name__

    @functools.wraps(func)
    def wrapper(self: MeshCore, *args: Any, **kwargs: Any) -> OperationResult:
        if self._journal is not None:  # called from inside another operation
            return func(self, *args, **kwargs)
        if self._closed:
            raise EditError(
                "SESSION_CLOSED",
                "this editable mesh belongs to an editing session that was committed or discarded",
                hint="start a new session with obj.edit_mesh()",
            )
        self._journal = []
        self._op_diagnostics = []
        counters = (self._nv, self._next_face, self._next_edge)
        try:
            data = func(self, *args, **kwargs) or {}
            self._check_element_limits()
        except BaseException:
            journal, self._journal = self._journal, None
            self._rollback(journal, counters)
            raise
        journal, self._journal = self._journal, None
        result = self._summarize(name, journal, data, self._op_diagnostics)
        self._history.append(_HistoryEntry(name, args, kwargs, journal, counters, result))
        if len(self._history) > self.max_history:
            del self._history[0]
        if not self._redoing:
            self._redo.clear()
        return result

    wrapper.is_operation = True  # type: ignore[attr-defined]
    return wrapper


class MeshCore:
    """Storage, journaling, transactions and queries shared by :class:`EditableMesh`.

    An editable polygon mesh with stable element ids and transactional operations.

    Create one with :meth:`from_mesh` (from an immutable :class:`Mesh`), or
    empty (``EditableMesh()``) and add geometry with :meth:`add_box`,
    :meth:`add_grid`, :meth:`create_vertex` and :meth:`create_face`. Convert
    back with :meth:`to_mesh`. Inside a scene, use ``obj.edit_mesh()`` to get
    an :class:`EditSession` that commits the result to the object.

    Element ids are non-negative integers, unique per element kind within this
    mesh and never reused by later operations. Ids of untouched elements never
    change; every operation reports the ids it created, removed and modified.
    Undo restores the previous ids exactly.
    """

    max_history: int = 256
    max_elements: int = 20_000_000

    def __init__(self, *, uvs: bool = True, normals: bool = True, groups: bool = False) -> None:
        self._pos = np.zeros((16, 3), dtype=np.float64)
        self._alive = np.zeros(16, dtype=bool)
        self._nv = 0
        self._vert_edges: list[set[int]] = []
        self._faces: dict[int, list[int]] = {}
        self._next_face = 0
        self._edge_by_pair: dict[tuple[int, int], int] = {}
        self._edge_verts: dict[int, tuple[int, int]] = {}
        self._edge_faces: dict[int, list[int]] = {}
        self._next_edge = 0
        self._sharp: set[int] = set()
        self._smooth: dict[int, bool] = {}
        self._group: dict[int, int] = {}  # face group of every face (materials per group)
        self._has_groups = groups  # the mesh carries face groups (Mesh.groups)
        self._layers: dict[str, CornerLayer] = {}
        if uvs:
            self._layers[UV] = CornerLayer(UV, 2)
        if normals:
            self._layers[NORMAL] = CornerLayer(NORMAL, 3, normalize=True)
        self._dirty_faces: set[int] = set()
        self._journal: list | None = None
        self._op_diagnostics: list[Diagnostic] = []
        self._history: list[_HistoryEntry] = []
        self._redo: list[tuple[str, tuple, dict]] = []
        self._redoing = False
        self.world_matrix = np.eye(4)
        self.source_diagnostics: list[Diagnostic] = []
        self._closed = False
        self._bbox_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #

    @classmethod
    def from_mesh(cls, mesh: Mesh, *, weld: str = "exact", quads: bool = True):
        """Build an editable mesh from an immutable triangle :class:`Mesh`.

        ``weld="exact"`` (default) merges vertices whose positions are
        bit-identical (``-0.0 == 0.0``), so seams that exist only to carry
        different normals/UVs become shared topology with per-corner
        attributes; ``weld="none"`` keeps every vertex separate. Vertex ids
        follow the first occurrence of each (welded) vertex in the source
        vertex order, and face ids follow the triangle order, so opening the
        same Mesh twice yields the same ids.

        Triangles that become degenerate (a repeated vertex) are dropped and
        reported in :attr:`source_diagnostics`; their area is zero.
        Shading is inferred from the normals: a face is *smooth* if its
        corner normals differ from its face normal, and an edge is *sharp* if
        the normals of its two faces differ across it. Face groups
        (:attr:`Mesh.groups`) carry over to the faces, and every editing
        operation keeps them (new faces take the group of the face they come
        from).

        ``quads=True`` (default) rebuilds quads from pairs of triangles that are
        coplanar (within 1 degree), form a convex quad and agree on UVs and
        normals along their shared edge -- exactly :meth:`join_triangles` -- so
        triangulated primitives become editable quads without changing the
        surface. ``quads=False`` keeps one face per triangle.
        """
        if not isinstance(mesh, Mesh):
            raise ValidationError(f"expected a Mesh, got {type(mesh).__name__}")
        if weld not in ("exact", "none"):
            raise ValidationError(f"weld must be 'exact' or 'none', got {weld!r}")
        em = cls(
            uvs=mesh.uvs is not None,
            normals=mesh.normals is not None,
            groups=mesh.groups is not None,
        )
        n = mesh.vertex_count
        positions = mesh.positions.astype(np.float64)
        if weld == "exact" and n:
            keys = np.ascontiguousarray(mesh.positions + np.float32(0.0))
            keys = keys.view(np.dtype((np.void, keys.dtype.itemsize * 3))).ravel()
            _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
            inverse = inverse.reshape(-1)
            order = np.argsort(first, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            vertex_of = rank[inverse]
            unique_positions = positions[first[order]]
        else:
            vertex_of = np.arange(n)
            unique_positions = positions
        em._bulk_vertices(unique_positions)

        tris = vertex_of[mesh.triangles.astype(np.int64)]
        degenerate = (
            (tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2]) | (tris[:, 0] == tris[:, 2])
        )
        uv = mesh.uvs.astype(np.float64) if mesh.uvs is not None else None
        nor = mesh.normals.astype(np.float64) if mesh.normals is not None else None
        source = mesh.triangles
        groups = mesh.groups
        if _features.enabled("AETHER3D_EDITING_BULK"):
            kept = np.flatnonzero(~degenerate)
            corners_of = {}
            if uv is not None:
                corners_of[UV] = uv[source[kept]]
            if nor is not None:
                corners_of[NORMAL] = nor[source[kept]]
            em._bulk_faces(tris[kept], corners_of, None if groups is None else groups[kept])
        else:
            for t in np.flatnonzero(~degenerate):
                corners = {}
                if uv is not None:
                    corners[UV] = [tuple(uv[i]) for i in source[t]]
                if nor is not None:
                    corners[NORMAL] = [tuple(nor[i]) for i in source[t]]
                group = int(groups[t]) if groups is not None else 0
                em._new_face([int(v) for v in tris[t]], corners, smooth=False, group=group)
        if degenerate.any():
            em.source_diagnostics.append(
                Diagnostic(
                    "DEGENERATE_TRIANGLES_DROPPED",
                    INFO,
                    "triangles that repeat a (welded) vertex have zero area and were not "
                    "converted to faces",
                    None,
                    int(degenerate.sum()),
                    tuple(int(i) for i in np.flatnonzero(degenerate)[:10]),
                )
            )
        if nor is not None:
            em._infer_shading()
        if quads:  # same rules as join_triangles(); no history is needed while converting
            em._join_pairs(set(em._faces), 1.0)
        em._dirty_faces.clear()
        return em

    def _bulk_faces(
        self,
        loops: np.ndarray,
        corners: dict[str, np.ndarray],
        groups: np.ndarray | None,
    ) -> None:
        """``_new_face(loop, corners, smooth=False, group=...)`` for every row of ``loops``
        (``(n, k)`` vertex ids, ``k >= 3`` distinct per row), in order, into a mesh without
        faces yet: the same ids, tables and insertion orders, computed with NumPy.

        Edges get ids in order of first use; each edge's faces are listed in face order;
        each vertex's edge set receives its edge ids in increasing order, as one face after
        the other adds them (so even iteration orders match). ``corners`` maps every layer
        to ``(n, k, dim)`` values.
        """
        if self._faces or self._edge_by_pair or self._journal is not None:
            raise AssertionError("internal error: bulk faces go into an empty mesh")
        n, k = loops.shape
        if n == 0:
            return
        # One Python int per id, shared by every table that holds it (as building one face
        # at a time shares them): the tables cost what they cost before, not a copy per use.
        vertex = list(range(self._nv)).__getitem__
        first = self._next_face
        fids = list(range(first, first + n))
        flat = list(map(vertex, loops.reshape(-1).tolist()))
        self._faces.update(zip(fids, (flat[i : i + k] for i in range(0, n * k, k)), strict=True))
        del flat
        # The edges of every loop, (loop[j], loop[j + 1]) as (lower, higher) vertex id.
        nxt = np.roll(loops, -1, axis=1)
        lo = np.minimum(loops, nxt).reshape(-1)
        hi = np.maximum(loops, nxt).reshape(-1)
        del nxt
        keys = lo * np.int64(max(self._nv, 1)) + hi
        unique, first_use, inverse = np.unique(keys, return_index=True, return_inverse=True)
        del keys
        by_first = np.argsort(first_use, kind="stable")
        rank = np.empty(len(unique), dtype=np.int64)
        rank[by_first] = np.arange(len(unique), dtype=np.int64)
        edge_of = rank[inverse.reshape(-1)]  # each loop edge's edge, 0 = first used
        del unique, inverse, rank
        m = len(by_first)
        e0 = self._next_edge
        eids = list(range(e0, e0 + m))
        ends_lo = lo[first_use[by_first]]
        ends_hi = hi[first_use[by_first]]
        del lo, hi, first_use, by_first
        pairs = list(
            zip(
                map(vertex, ends_lo.tolist()),
                map(vertex, ends_hi.tolist()),
                strict=True,
            )
        )
        self._edge_by_pair.update(zip(pairs, eids, strict=True))
        self._edge_verts.update(zip(eids, pairs, strict=True))
        del pairs
        # Each edge's faces, in face order.
        order = np.argsort(edge_of, kind="stable")
        cuts = np.zeros(m + 1, dtype=np.int64)
        np.cumsum(np.bincount(edge_of, minlength=m), out=cuts[1:])
        del edge_of
        faces = list(map(fids.__getitem__, (order // k).tolist()))
        del order
        cut = cuts.tolist()
        self._edge_faces.update(
            zip(eids, (faces[cut[i] : cut[i + 1]] for i in range(m)), strict=True)
        )
        del faces, cut, cuts
        # Each vertex's edges, in increasing id (creation) order.
        ends = np.concatenate((ends_lo, ends_hi))
        owners = np.concatenate((np.arange(m), np.arange(m)))
        del ends_lo, ends_hi
        by_vertex = np.lexsort((owners, ends))
        vertex_sorted = ends[by_vertex]
        edge_sorted = list(map(eids.__getitem__, owners[by_vertex].tolist()))
        del ends, owners, by_vertex
        starts = np.flatnonzero(np.concatenate(([True], vertex_sorted[1:] != vertex_sorted[:-1])))
        bounds = np.append(starts, len(vertex_sorted)).tolist()
        vertex_edges = self._vert_edges
        for j, v in enumerate(vertex_sorted[starts].tolist()):
            vertex_edges[v].update(edge_sorted[bounds[j] : bounds[j + 1]])
        del edge_sorted, vertex_sorted
        # Corner values (tuples of Python floats), smoothness, groups.
        for name, layer in self._layers.items():
            values = list(map(tuple, corners[name].reshape(n * k, layer.dim).tolist()))
            layer.data.update(
                zip(fids, (values[i * k : (i + 1) * k] for i in range(n)), strict=True)
            )
            del values
        self._smooth.update(dict.fromkeys(fids, False))
        self._group.update(zip(fids, [0] * n if groups is None else groups.tolist(), strict=True))
        self._dirty_faces.update(fids)
        self._next_face = first + n
        self._next_edge = e0 + m

    def _bulk_vertices(self, positions: np.ndarray) -> None:
        n = len(positions)
        self._ensure_capacity(n)
        self._pos[:n] = positions
        self._alive[:n] = True
        self._bbox_cache = None
        self._nv = n
        self._vert_edges = [set() for _ in range(n)]

    def _infer_shading(self) -> None:
        normals = self._layers[NORMAL].data
        for fid, loop in self._faces.items():
            fn = poly.unit(poly.newell_normal(self._pos[loop]))
            corner = normals[fid]
            self._smooth[fid] = fn is None or any(float(np.dot(fn, c)) < 1.0 - 1e-6 for c in corner)
        for eid, faces in self._edge_faces.items():
            if len(faces) != 2:
                continue
            a, b = self._edge_verts[eid]
            f, g = faces
            for v in (a, b):
                nf = normals[f][self._faces[f].index(v)]
                ng = normals[g][self._faces[g].index(v)]
                if float(np.dot(nf, ng)) < 1.0 - 1e-6:
                    self._sharp.add(eid)
                    break

    def to_mesh(self) -> Mesh:
        """Triangulate into an immutable :class:`Mesh` (see :meth:`build_mesh`)."""
        return self.build_mesh()[0]

    def build_mesh(self, *, compute_normals: bool | None = None) -> tuple[Mesh, list[Diagnostic]]:
        """Triangulate into a :class:`Mesh`; returns ``(mesh, diagnostics)`` (see
        :meth:`_build_mesh`)."""
        mesh, diagnostics, _ = self._build_mesh(compute_normals=compute_normals)
        return mesh, diagnostics

    def _build_mesh(
        self, *, compute_normals: bool | None = None
    ) -> tuple[Mesh, list[Diagnostic], list[tuple[int, int, int, int]]]:
        """Triangulate into a :class:`Mesh`; returns ``(mesh, diagnostics, sources)`` with
        ``sources[t] = (face id, corner a, corner b, corner c)`` for every triangle.

        Faces are triangulated by ear clipping (deterministic, winding
        preserved). Output vertices are the distinct ``(vertex, normal, uv)``
        combinations, in order of first use by faces sorted by id. Loose
        vertices are omitted (reported). Normals of corners affected by
        geometric edits are recomputed; others are kept exactly.
        ``compute_normals=True`` adds normals to a mesh that has none
        (``False`` omits them).
        """
        from .normals import corner_normals

        diagnostics: list[Diagnostic] = []
        want_normals = NORMAL in self._layers if compute_normals is None else compute_normals
        normals = (
            corner_normals(self, recompute_all=NORMAL not in self._layers) if want_normals else None
        )
        uvs = self._layers[UV].data if UV in self._layers else None
        key_index: dict[tuple, int] = {}
        out_vertex: list[int] = []
        out_normal: list[tuple] = []
        out_uv: list[tuple] = []
        triangles: list[tuple[int, int, int]] = []
        sources: list[tuple[int, int, int, int]] = []
        triangle_groups: list[int] = []
        for fid in sorted(self._faces):
            loop = self._faces[fid]
            try:
                local = poly.triangulate(self._pos[loop])
            except ValueError as exc:
                raise EditError(
                    "TRIANGULATION_FAILED",
                    f"face {fid} cannot be triangulated: {exc}",
                    elements={"faces": [fid]},
                    hint="fix the face (e.g. split_face, dissolve, or move its vertices) first",
                ) from None
            corner_out = []
            for k, v in enumerate(loop):
                nk = normals[fid][k] if normals is not None else ()
                uk = uvs[fid][k] if uvs is not None else ()
                key = (v, nk, uk)
                index = key_index.get(key)
                if index is None:
                    index = key_index[key] = len(out_vertex)
                    out_vertex.append(v)
                    out_normal.append(nk)
                    out_uv.append(uk)
                corner_out.append(index)
            triangles.extend((corner_out[a], corner_out[b], corner_out[c]) for a, b, c in local)
            sources.extend((fid, a, b, c) for a, b, c in local)
            triangle_groups.extend([self._group.get(fid, 0)] * len(local))
        loose = [v for v in self.vertex_ids() if not self._vert_edges[v]]
        if loose:
            diagnostics.append(
                Diagnostic(
                    "LOOSE_VERTICES_OMITTED",
                    INFO,
                    "vertices not used by any face are not part of a triangle mesh",
                    None,
                    len(loose),
                    tuple(loose[:10]),
                )
            )
        mesh = Mesh(
            self._pos[out_vertex] if out_vertex else np.zeros((0, 3)),
            np.array(triangles, dtype=np.int64).reshape(-1, 3),
            normals=np.array(out_normal).reshape(-1, 3) if normals is not None else None,
            uvs=np.array(out_uv).reshape(-1, 2) if uvs is not None else None,
            groups=np.array(triangle_groups, dtype=np.int64) if self._has_groups else None,
        )
        return mesh, diagnostics, sources

    def copy(self):
        """An independent deep copy (without history)."""
        other = type(self)(uvs=False, normals=False)
        other._pos = self._pos.copy()
        other._alive = self._alive.copy()
        other._nv = self._nv
        other._vert_edges = [set(s) for s in self._vert_edges]
        other._faces = {k: list(v) for k, v in self._faces.items()}
        other._next_face = self._next_face
        other._edge_by_pair = dict(self._edge_by_pair)
        other._edge_verts = dict(self._edge_verts)
        other._edge_faces = {k: list(v) for k, v in self._edge_faces.items()}
        other._next_edge = self._next_edge
        other._sharp = set(self._sharp)
        other._smooth = dict(self._smooth)
        other._group = dict(self._group)
        other._has_groups = self._has_groups
        other._layers = {
            name: CornerLayer(
                layer.name,
                layer.dim,
                layer.normalize,
                layer.default,
                {k: list(v) for k, v in layer.data.items()},
            )
            for name, layer in self._layers.items()
        }
        other._dirty_faces = set(self._dirty_faces)
        other.world_matrix = self.world_matrix.copy()
        other._bbox_cache = self._bbox_cache
        return other

    # ------------------------------------------------------------------ #
    # Read-only queries
    # ------------------------------------------------------------------ #

    @property
    def vertex_count(self) -> int:
        return int(self._alive[: self._nv].sum())

    @property
    def edge_count(self) -> int:
        return len(self._edge_verts)

    @property
    def face_count(self) -> int:
        return len(self._faces)

    @property
    def has_uvs(self) -> bool:
        return UV in self._layers

    @property
    def has_normals(self) -> bool:
        return NORMAL in self._layers

    @property
    def corner_layers(self) -> tuple[str, ...]:
        return tuple(self._layers)

    def vertex_ids(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self._alive[: self._nv])]

    def edge_ids(self) -> list[int]:
        return sorted(self._edge_verts)

    def face_ids(self) -> list[int]:
        return sorted(self._faces)

    def has_vertex(self, vid: int) -> bool:
        return isinstance(vid, (int, np.integer)) and 0 <= vid < self._nv and bool(self._alive[vid])

    def has_edge(self, eid: int) -> bool:
        return eid in self._edge_verts

    def has_face(self, fid: int) -> bool:
        return fid in self._faces

    def position(self, vid: int) -> tuple[float, float, float]:
        self._require_vertices([vid])
        x, y, z = self._pos[vid].tolist()
        return (x, y, z)

    def positions(self, vids: Iterable[int] | None = None) -> np.ndarray:
        """``(n, 3)`` positions (a copy) of the given vertices (default: all, by id)."""
        ids = self.vertex_ids() if vids is None else self._require_vertices(vids)
        return self._pos[ids].copy()

    def world_positions(self, vids: Iterable[int] | None = None) -> np.ndarray:
        p = self.positions(vids)
        m = self.world_matrix
        return p @ m[:3, :3].T + m[:3, 3]

    def face_vertices(self, fid: int) -> tuple[int, ...]:
        self._require_faces([fid])
        return tuple(self._faces[fid])

    def face_edges(self, fid: int) -> tuple[int, ...]:
        """Edge ids of a face in loop order (edge k joins corner k and k+1)."""
        self._require_faces([fid])
        loop = self._faces[fid]
        return tuple(self._edge_id(loop[k], loop[(k + 1) % len(loop)]) for k in range(len(loop)))

    def edge_vertices(self, eid: int) -> tuple[int, int]:
        self._require_edges([eid])
        return self._edge_verts[eid]

    def edge_faces(self, eid: int) -> tuple[int, ...]:
        self._require_edges([eid])
        return tuple(self._edge_faces[eid])

    def vertex_edges(self, vid: int) -> tuple[int, ...]:
        self._require_vertices([vid])
        return tuple(sorted(self._vert_edges[vid]))

    def vertex_faces(self, vid: int) -> tuple[int, ...]:
        self._require_vertices([vid])
        return tuple(sorted(self._faces_of_vertex(vid)))

    def edge_between(self, a: int, b: int) -> int | None:
        """Id of the edge joining vertices ``a`` and ``b``, or ``None``."""
        return self._edge_by_pair.get((min(a, b), max(a, b)))

    def face_normal(self, fid: int) -> tuple[float, float, float]:
        """Unit front normal (Newell); ``(0, 0, 0)`` for a zero-area face."""
        self._require_faces([fid])
        n = poly.unit(poly.newell_normal(self._pos[self._faces[fid]]))
        x, y, z = (n if n is not None else np.zeros(3)).tolist()
        return (x, y, z)

    def face_center(self, fid: int) -> tuple[float, float, float]:
        self._require_faces([fid])
        x, y, z = self._pos[self._faces[fid]].mean(axis=0).tolist()
        return (x, y, z)

    def face_area(self, fid: int) -> float:
        self._require_faces([fid])
        return poly.polygon_area(self._pos[self._faces[fid]])

    def edge_length(self, eid: int) -> float:
        a, b = self.edge_vertices(eid)
        return float(np.linalg.norm(self._pos[a] - self._pos[b]))

    def edge_midpoint(self, eid: int) -> tuple[float, float, float]:
        a, b = self.edge_vertices(eid)
        x, y, z = ((self._pos[a] + self._pos[b]) / 2.0).tolist()
        return (x, y, z)

    def corner_values(self, fid: int, layer: str) -> list[tuple[float, ...]]:
        """Per-corner values of ``layer`` (``"uv"``/``"normal"``) for a face, in loop order."""
        self._require_faces([fid])
        if layer not in self._layers:
            raise ValidationError(f"mesh has no {layer!r} layer (layers: {list(self._layers)})")
        return list(self._layers[layer].data[fid])

    @property
    def has_groups(self) -> bool:
        """Whether the mesh carries face groups (they become :attr:`Mesh.groups`)."""
        return self._has_groups

    def face_group(self, fid: int) -> int:
        """The face group of face ``fid`` (0 unless set; see :meth:`set_group`)."""
        self._require_faces([fid])
        return self._group.get(fid, 0)

    def is_smooth(self, fid: int) -> bool:
        self._require_faces([fid])
        return self._smooth.get(fid, False)

    def is_sharp(self, eid: int) -> bool:
        self._require_edges([eid])
        return eid in self._sharp

    def bounds(self, space: str = "local") -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        if self.vertex_count == 0:
            return None
        p = self.world_positions() if space == "world" else self.positions()
        return tuple(p.min(axis=0).tolist()), tuple(p.max(axis=0).tolist())

    def summary(self) -> dict[str, Any]:
        """Counts and face-size histogram (cheap; see :meth:`validate` for checks)."""
        sizes: dict[str, int] = {}
        for loop in self._faces.values():
            key = str(len(loop)) if len(loop) < 5 else "5+"
            sizes[key] = sizes.get(key, 0) + 1
        radial = [len(f) for f in self._edge_faces.values()]
        return {
            "vertex_count": self.vertex_count,
            "edge_count": self.edge_count,
            "face_count": self.face_count,
            "face_sizes": dict(sorted(sizes.items())),
            "triangle_count": sum(len(loop) - 2 for loop in self._faces.values()),
            "boundary_edge_count": sum(1 for r in radial if r == 1),
            "non_manifold_edge_count": sum(1 for r in radial if r > 2),
            "layers": list(self._layers),
            "bounds": self.bounds(),
        }

    def describe(
        self, kinds: Iterable[str] = ("vertices", "edges", "faces"), limit: int | None = None
    ):
        """Element listing as plain data (ids, positions, loops, normals ...)."""
        out: dict[str, Any] = {}
        kinds = set(kinds)
        if "vertices" in kinds:
            ids = self.vertex_ids()[:limit]
            out["vertices"] = [{"id": v, "position": list(self._pos[v].tolist())} for v in ids]
        if "edges" in kinds:
            ids = self.edge_ids()[:limit]
            out["edges"] = [
                {"id": e, "vertices": list(self._edge_verts[e]), "faces": list(self._edge_faces[e])}
                for e in ids
            ]
        if "faces" in kinds:
            ids = self.face_ids()[:limit]
            out["faces"] = [
                {
                    "id": f,
                    "vertices": list(self._faces[f]),
                    "normal": list(self.face_normal(f)),
                    "center": list(self.face_center(f)),
                    "area": self.face_area(f),
                }
                for f in ids
            ]
        return out

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> list[dict[str, Any]]:
        """Applied operations (oldest first) as ``{"operation", "result"}`` dicts."""
        return [{"operation": h.name, "result": h.result.to_dict()} for h in self._history]

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str:
        """Revert the most recent operation exactly (data and ids); returns its name."""
        if self._journal is not None:
            raise EditError("UNDO_IN_OPERATION", "cannot undo while an operation is running")
        if not self._history:
            raise EditError("NOTHING_TO_UNDO", "there is no operation to undo")
        entry = self._history.pop()
        self._rollback(entry.journal, entry.counters)
        self._redo.append((entry.name, entry.args, entry.kwargs))
        return entry.name

    def redo(self) -> OperationResult:
        """Re-apply the most recently undone operation (same arguments, same ids)."""
        if not self._redo:
            raise EditError("NOTHING_TO_REDO", "there is no undone operation to redo")
        name, args, kwargs = self._redo.pop()
        self._redoing = True
        try:
            return getattr(self, name)(*args, **kwargs)
        finally:
            self._redoing = False

    @property
    def has_changes(self) -> bool:
        return bool(self._history) or bool(self._dirty_faces)

    # ------------------------------------------------------------------ #
    # Element argument helpers
    # ------------------------------------------------------------------ #

    def _ids(self, elements: Elements, kind: str) -> list[int]:
        if isinstance(elements, Selection):
            ids = set(getattr(elements, kind))
            others = [
                k for k in ("vertices", "edges", "faces") if k != kind and getattr(elements, k)
            ]
            if not ids and others:
                raise EditError(
                    "WRONG_ELEMENT_KIND",
                    f"expected {kind}, but the selection contains only {', '.join(others)}",
                    hint=f"convert it, e.g. mesh.to_{kind}(selection)",
                )
        elif isinstance(elements, (int, np.integer)) and not isinstance(elements, bool):
            ids = {int(elements)}
        else:
            try:
                ids = set()
                for i in elements:
                    if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
                        raise TypeError
                    ids.add(int(i))
            except TypeError:
                raise ValidationError(
                    f"{kind} must be a Selection, an id, or an iterable of integer ids"
                ) from None
        return sorted(ids)

    def _require(self, ids: list[int], kind: str, exists: Callable[[int], bool]) -> list[int]:
        missing = [i for i in ids if not exists(i)]
        if missing:
            raise EditError(
                "INVALID_ELEMENT",
                f"{len(missing)} {kind} do not exist in this mesh",
                elements={kind: missing},
                hint="ids may have been removed by an earlier operation; check its result",
            )
        return ids

    def _require_vertices(self, vids: Iterable[int]) -> list[int]:
        return self._require(self._ids(vids, "vertices"), "vertices", self.has_vertex)

    def _require_edges(self, eids: Iterable[int]) -> list[int]:
        return self._require(self._ids(eids, "edges"), "edges", self.has_edge)

    def _require_faces(self, fids: Iterable[int]) -> list[int]:
        return self._require(self._ids(fids, "faces"), "faces", self.has_face)

    def _elements(self, elements: Elements, kind: str, *, allow_empty: bool = False) -> list[int]:
        ids = self._require(
            self._ids(elements, kind),
            kind,
            {"vertices": self.has_vertex, "edges": self.has_edge, "faces": self.has_face}[kind],
        )
        if not ids and not allow_empty:
            raise EditError("EMPTY_SELECTION", f"no {kind} given", hint=f"select some {kind} first")
        return ids

    def _vertices_of(self, elements: Elements) -> list[int]:
        """All vertices referenced by a selection or an iterable of vertex ids."""
        if isinstance(elements, Selection):
            vids = set(self._require_vertices(elements.vertices))
            for e in self._require_edges(elements.edges):
                vids.update(self._edge_verts[e])
            for f in self._require_faces(elements.faces):
                vids.update(self._faces[f])
            return sorted(vids)
        return self._require_vertices(elements)

    # ------------------------------------------------------------------ #
    # Internal topology helpers
    # ------------------------------------------------------------------ #

    def _edge_id(self, a: int, b: int) -> int:
        return self._edge_by_pair[(a, b) if a < b else (b, a)]

    def _faces_of_vertex(self, vid: int) -> set[int]:
        faces: set[int] = set()
        for e in self._vert_edges[vid]:
            faces.update(self._edge_faces[e])
        return faces

    def _loop_edges(self, loop: list[int]) -> list[tuple[int, int]]:
        n = len(loop)
        return [
            (loop[k], loop[(k + 1) % n])
            if loop[k] < loop[(k + 1) % n]
            else (loop[(k + 1) % n], loop[k])
            for k in range(n)
        ]

    def _face_normal_vec(self, fid: int) -> np.ndarray | None:
        return poly.unit(poly.newell_normal(self._pos[self._faces[fid]]))

    def _scale(self) -> float:
        """Largest extent of the mesh, used to make tolerances scale-independent.

        The bounding box is cached and grown as vertices are added or moved (it
        never shrinks until a rollback), so tolerance checks stay O(1) inside
        operations that move many vertices. A slightly stale, larger extent only
        makes the 1e-12-relative tolerances marginally more conservative.
        """
        if self._bbox_cache is None:
            p = self._pos[: self._nv][self._alive[: self._nv]]
            if len(p) == 0:
                return 1.0
            self._bbox_cache = (p.min(axis=0), p.max(axis=0))
        lo, hi = self._bbox_cache
        return max(float(np.max(hi - lo)), 1e-12)

    def _grow_bbox(self, points: np.ndarray) -> None:
        if self._bbox_cache is not None and len(points):
            lo, hi = self._bbox_cache
            self._bbox_cache = (
                np.minimum(lo, points.min(axis=0)),
                np.maximum(hi, points.max(axis=0)),
            )

    def _area_eps(self) -> float:
        return 1e-12 * self._scale() ** 2

    def _check_element_limits(self) -> None:
        if self._nv > self.max_elements or len(self._faces) > self.max_elements:
            raise EditError(
                "TOO_MANY_ELEMENTS",
                f"the mesh would exceed {self.max_elements} vertices or faces",
                hint="reduce cuts/segments or edit a smaller region",
            )

    def _ensure_capacity(self, n: int) -> None:
        if n <= len(self._pos):
            return
        size = max(n, 2 * len(self._pos))
        pos = np.zeros((size, 3), dtype=np.float64)
        alive = np.zeros(size, dtype=bool)
        pos[: len(self._pos)] = self._pos
        alive[: len(self._alive)] = self._alive
        self._pos, self._alive = pos, alive

    def _corner_defaults(self, loop: list[int]) -> dict[str, list[tuple]]:
        return {
            name: [layer.default or (0.0,) * layer.dim for _ in loop]
            for name, layer in self._layers.items()
        }

    def _corners_of(self, fid: int) -> list[dict[str, tuple]]:
        """Per-corner attribute dicts of a face, in loop order."""
        layers = self._layers
        return [
            {name: layer.data[fid][k] for name, layer in layers.items()}
            for k in range(len(self._faces[fid]))
        ]

    def _mix(self, values: list[tuple[dict[str, tuple], float]]) -> dict[str, tuple]:
        """Weighted sum of corner attribute dicts (normalized layers renormalized)."""
        out: dict[str, tuple] = {}
        for name, layer in self._layers.items():
            total = np.zeros(layer.dim)
            for attrs, weight in values:
                total += weight * np.asarray(attrs[name], dtype=np.float64)
            if layer.normalize:
                unit_total = poly.unit(total)
                total = unit_total if unit_total is not None else np.asarray(values[0][0][name])
            out[name] = tuple(float(x) for x in total)
        return out

    def _affine_attrs(self, fid: int, points: np.ndarray) -> list[dict[str, tuple]]:
        """Attributes at points inside face ``fid`` by an affine fit over its corners."""
        loop = self._faces[fid]
        out: list[dict[str, tuple]] = [{} for _ in range(len(points))]
        for name, layer in self._layers.items():
            values = np.asarray(layer.data[fid], dtype=np.float64)
            fitted = poly.affine_fit(self._pos[loop], values)(points)
            for i, row in enumerate(fitted):
                if layer.normalize:
                    unit_row = poly.unit(row)
                    row = unit_row if unit_row is not None else values[0]
                out[i][name] = tuple(float(x) for x in row)
        return out

    @staticmethod
    def _split_corners(corners: list[dict[str, tuple]]) -> dict[str, list[tuple]]:
        """Convert a list of per-corner attribute dicts into per-layer lists."""
        if not corners:
            return {}
        return {name: [c[name] for c in corners] for name in corners[0]}

    # ------------------------------------------------------------------ #
    # Journaled primitives (all mutation goes through these)
    # ------------------------------------------------------------------ #

    def _log(self, entry: tuple) -> None:
        if self._journal is not None:
            self._journal.append(entry)

    def _mark_dirty(self, fids: Iterable[int]) -> None:
        added = [f for f in fids if f not in self._dirty_faces]
        if added:
            self._dirty_faces.update(added)
            self._log(("dirty", added))

    def _mark_clean(self, fids: Iterable[int]) -> None:
        """Keep the stored corner normals of faces at commit (an operation wrote them)."""
        removed = [f for f in fids if f in self._dirty_faces]
        if removed:
            self._dirty_faces.difference_update(removed)
            self._log(("clean", removed))

    def _new_vertex(self, position: Iterable[float]) -> int:
        vid = self._nv
        self._ensure_capacity(vid + 1)
        self._pos[vid] = np.asarray(position, dtype=np.float64)
        self._grow_bbox(self._pos[vid : vid + 1])
        self._alive[vid] = True
        self._vert_edges.append(set())
        self._nv += 1
        self._log(("v+", vid))
        return vid

    def _delete_vertex(self, vid: int) -> None:
        if self._vert_edges[vid]:
            raise AssertionError("internal error: deleting a vertex that still has edges")
        self._alive[vid] = False
        self._log(("v-", vid))

    def _set_positions(self, vids: list[int], positions: np.ndarray, *, dirty: bool = True) -> None:
        index = np.asarray(vids, dtype=np.int64)
        self._log(("mv", index, self._pos[index].copy()))
        self._pos[index] = positions
        self._grow_bbox(self._pos[index])
        if dirty:
            faces: set[int] = set()
            for v in vids:
                faces |= self._faces_of_vertex(v)
            self._mark_dirty(faces)

    def _add_edge_use(self, a: int, b: int, fid: int, created: list, reuse: dict | None) -> None:
        key = (a, b) if a < b else (b, a)
        eid = self._edge_by_pair.get(key)
        if eid is None:
            eid = (reuse or {}).get(key)
            if eid is None:
                eid = self._next_edge
                self._next_edge += 1
            self._edge_by_pair[key] = eid
            self._edge_verts[eid] = key
            self._edge_faces[eid] = []
            self._vert_edges[key[0]].add(eid)
            self._vert_edges[key[1]].add(eid)
            created.append(eid)
        self._edge_faces[eid].append(fid)

    def _drop_edge_use(self, key: tuple[int, int], fid: int, removed: list) -> None:
        eid = self._edge_by_pair[key]
        faces = self._edge_faces[eid]
        index = faces.index(fid)
        del faces[index]
        if not faces:
            removed.append((eid, key, eid in self._sharp, index))
            del self._edge_by_pair[key], self._edge_verts[eid], self._edge_faces[eid]
            self._vert_edges[key[0]].discard(eid)
            self._vert_edges[key[1]].discard(eid)
            self._sharp.discard(eid)
        else:
            removed.append((None, key, False, index))

    def _new_face(
        self,
        loop: list[int],
        corners: dict[str, list[tuple]] | None = None,
        *,
        smooth: bool = False,
        group: int = 0,
        fid: int | None = None,
        reuse_edges: dict | None = None,
        dirty: bool = True,
    ) -> int:
        if len(loop) < 3 or len(set(loop)) != len(loop):
            raise AssertionError(f"internal error: invalid face loop {loop}")
        if fid is None:
            fid = self._next_face
            self._next_face += 1
        self._faces[fid] = list(loop)
        created: list[int] = []
        for a, b in self._loop_edges(loop):
            self._add_edge_use(a, b, fid, created, reuse_edges)
        corners = corners or {}
        defaults = None
        for name, layer in self._layers.items():
            values = corners.get(name)
            if values is None:
                defaults = defaults or self._corner_defaults(loop)
                values = defaults[name]
            layer.data[fid] = [tuple(float(x) for x in v) for v in values]
        self._smooth[fid] = bool(smooth)
        self._group[fid] = int(group)
        self._log(("f+", fid, created))
        if dirty:
            self._mark_dirty([fid])
        return fid

    def _delete_face(self, fid: int) -> None:
        loop = self._faces.pop(fid)
        removed: list = []
        for key in self._loop_edges(loop):
            self._drop_edge_use(key, fid, removed)
        corners = {name: layer.data.pop(fid) for name, layer in self._layers.items()}
        smooth = self._smooth.pop(fid)
        group = self._group.pop(fid, 0)
        was_dirty = fid in self._dirty_faces
        self._dirty_faces.discard(fid)
        self._log(("f-", fid, loop, corners, smooth, removed, was_dirty, group))

    def _replace_face(
        self,
        fid: int,
        loop: list[int],
        corners: dict[str, list[tuple]] | None = None,
        *,
        dirty: bool = True,
    ) -> None:
        """Give face ``fid`` a new loop (and corners); edges common to both keep their ids.

        ``dirty=False`` is for edits that do not change the face's surface (inserting
        vertices on its edges, re-ordering), so stored normals stay valid.
        """
        if len(loop) < 3 or len(set(loop)) != len(loop):
            raise AssertionError(f"internal error: invalid face loop {loop}")
        old_loop = self._faces[fid]
        old_corners = {name: layer.data[fid] for name, layer in self._layers.items()}
        old_keys, new_keys = self._loop_edges(old_loop), self._loop_edges(loop)
        keep = set(old_keys) & set(new_keys)
        removed: list = []
        for key in old_keys:
            if key not in keep:
                self._drop_edge_use(key, fid, removed)
        created: list[int] = []
        added: list[tuple[int, int]] = []
        for a, b in new_keys:
            if (a, b) not in keep:
                self._add_edge_use(a, b, fid, created, None)
                added.append((a, b))
        self._faces[fid] = list(loop)
        corners = corners or {}
        defaults = None
        for name, layer in self._layers.items():
            values = corners.get(name)
            if values is None:
                defaults = defaults or self._corner_defaults(loop)
                values = defaults[name]
            layer.data[fid] = [tuple(float(x) for x in v) for v in values]
        self._log(("f~", fid, old_loop, old_corners, removed, added, created))
        if dirty:
            self._mark_dirty([fid])

    def _add_layer(self, name: str, dim: int, default: tuple[float, ...]) -> None:
        """Create a corner layer with ``default`` at every corner (journaled)."""
        self._log(("layer+", name))
        self._layers[name] = CornerLayer(
            name, dim, data={fid: [default] * len(loop) for fid, loop in self._faces.items()}
        )

    def _set_corners(self, fid: int, name: str, values: list[tuple]) -> None:
        layer = self._layers[name]
        self._log(("c=", fid, name, layer.data[fid]))
        layer.data[fid] = [tuple(float(x) for x in v) for v in values]

    def _set_smooth(self, fid: int, smooth: bool) -> None:
        self._log(("s=", fid, self._smooth[fid]))
        self._smooth[fid] = bool(smooth)

    def _set_group(self, fid: int, group: int) -> None:
        self._log(("g=", fid, self._group.get(fid, 0)))
        self._group[fid] = int(group)

    def _set_sharp(self, eid: int, sharp: bool) -> None:
        self._log(("e=", eid, eid in self._sharp))
        if sharp:
            self._sharp.add(eid)
        else:
            self._sharp.discard(eid)

    # -- rollback ---------------------------------------------------------

    def _restore_edge_uses(self, fid: int, removed: list) -> None:
        for eid, key, sharp, index in reversed(removed):
            if eid is not None:
                self._edge_by_pair[key] = eid
                self._edge_verts[eid] = key
                self._edge_faces[eid] = []
                self._vert_edges[key[0]].add(eid)
                self._vert_edges[key[1]].add(eid)
                if sharp:
                    self._sharp.add(eid)
            self._edge_faces[self._edge_by_pair[key]].insert(index, fid)

    def _remove_created_edge_uses(self, fid: int, keys: list[tuple[int, int]]) -> None:
        for key in keys:
            eid = self._edge_by_pair[key]
            self._edge_faces[eid].remove(fid)
            if not self._edge_faces[eid]:
                del self._edge_by_pair[key], self._edge_verts[eid], self._edge_faces[eid]
                self._vert_edges[key[0]].discard(eid)
                self._vert_edges[key[1]].discard(eid)
                self._sharp.discard(eid)

    def _rollback(self, journal: list, counters: tuple[int, int, int]) -> None:
        self._bbox_cache = None
        for entry in reversed(journal):
            kind = entry[0]
            if kind == "v+":
                vid = entry[1]
                self._alive[vid] = False
                self._pos[vid] = 0.0
                self._vert_edges.pop()
                self._nv -= 1
            elif kind == "v-":
                self._alive[entry[1]] = True
            elif kind == "mv":
                self._pos[entry[1]] = entry[2]
            elif kind == "f+":
                fid = entry[1]
                loop = self._faces.pop(fid)
                self._remove_created_edge_uses(fid, self._loop_edges(loop))
                for layer in self._layers.values():
                    layer.data.pop(fid, None)
                self._smooth.pop(fid, None)
                self._group.pop(fid, None)
                self._dirty_faces.discard(fid)
            elif kind == "f-":
                _, fid, loop, corners, smooth, removed, was_dirty, group = entry
                self._faces[fid] = loop
                self._restore_edge_uses(fid, removed)
                for name, values in corners.items():
                    self._layers[name].data[fid] = values
                self._smooth[fid] = smooth
                self._group[fid] = group
                if was_dirty:
                    self._dirty_faces.add(fid)
            elif kind == "f~":
                _, fid, old_loop, old_corners, removed, added, _created = entry
                self._remove_created_edge_uses(fid, added)
                self._restore_edge_uses(fid, removed)
                self._faces[fid] = old_loop
                for name, values in old_corners.items():
                    self._layers[name].data[fid] = values
            elif kind == "c=":
                self._layers[entry[2]].data[entry[1]] = entry[3]
            elif kind == "layer+":
                del self._layers[entry[1]]
            elif kind == "s=":
                self._smooth[entry[1]] = entry[2]
            elif kind == "g=":
                self._group[entry[1]] = entry[2]
            elif kind == "groups":
                self._has_groups = entry[1]
            elif kind == "e=":
                if entry[2]:
                    self._sharp.add(entry[1])
                else:
                    self._sharp.discard(entry[1])
            elif kind == "dirty":
                self._dirty_faces.difference_update(entry[1])
            elif kind == "clean":
                self._dirty_faces.update(entry[1])
            else:  # pragma: no cover
                raise AssertionError(f"unknown journal entry {kind}")
        self._nv, self._next_face, self._next_edge = counters

    # ------------------------------------------------------------------ #
    # Result summaries
    # ------------------------------------------------------------------ #

    def _summarize(
        self, name: str, journal: list, data: dict, diagnostics: list[Diagnostic]
    ) -> OperationResult:
        cv, rv, cf, rf, ce, re_, mf = set(), set(), set(), set(), set(), set(), set()
        for entry in journal:
            kind = entry[0]
            if kind == "v+":
                cv.add(entry[1])
            elif kind == "v-":
                (cv.discard(entry[1]) if entry[1] in cv else rv.add(entry[1]))
            elif kind == "f+":
                cf.add(entry[1])
                ce.update(entry[2])
            elif kind == "f-":
                (cf.discard(entry[1]) if entry[1] in cf else rf.add(entry[1]))
                for eid, *_ in entry[5]:
                    if eid is not None:
                        (ce.discard(eid) if eid in ce else re_.add(eid))
            elif kind == "f~":
                mf.add(entry[1])
                for eid, *_ in entry[4]:
                    if eid is not None:
                        (ce.discard(eid) if eid in ce else re_.add(eid))
                ce.update(entry[6])
        ce = {e for e in ce if e in self._edge_verts}
        mf = {f for f in mf if f in self._faces and f not in cf}
        return OperationResult(
            operation=name,
            created={"vertices": sorted(cv), "edges": sorted(ce), "faces": sorted(cf)},
            removed={"vertices": sorted(rv), "edges": sorted(re_), "faces": sorted(rf)},
            modified_faces=sorted(mf),
            data=data,
            diagnostics=list(diagnostics),
        )

    def _diagnose(self, diagnostic: Diagnostic) -> None:
        self._op_diagnostics.append(diagnostic)

    # ------------------------------------------------------------------ #
    # Internal consistency (used by validate() and the test suite)
    # ------------------------------------------------------------------ #

    def _consistency_errors(self) -> list[str]:
        errors: list[str] = []
        for fid, loop in self._faces.items():
            if len(loop) < 3 or len(set(loop)) != len(loop):
                errors.append(f"face {fid} has an invalid loop {loop}")
            for v in loop:
                if not self.has_vertex(v):
                    errors.append(f"face {fid} uses missing vertex {v}")
            for key in self._loop_edges(loop):
                eid = self._edge_by_pair.get(key)
                if eid is None or fid not in self._edge_faces.get(eid, []):
                    errors.append(f"face {fid} edge {key} is not registered")
            for name, layer in self._layers.items():
                if len(layer.data.get(fid, ())) != len(loop):
                    errors.append(f"face {fid} has the wrong number of {name} corners")
        for key, eid in self._edge_by_pair.items():
            if self._edge_verts.get(eid) != key:
                errors.append(f"edge {eid} vertex pair mismatch")
            faces = self._edge_faces.get(eid, [])
            if not faces:
                errors.append(f"edge {eid} has no faces")
            for f in faces:
                if f not in self._faces or key not in self._loop_edges(self._faces[f]):
                    errors.append(f"edge {eid} lists face {f} which does not use it")
            for v in key:
                if not self.has_vertex(v) or eid not in self._vert_edges[v]:
                    errors.append(f"edge {eid} is not registered at vertex {v}")
        for v in range(self._nv):
            for e in self._vert_edges[v]:
                if e not in self._edge_verts or v not in self._edge_verts[e]:
                    errors.append(f"vertex {v} lists stale edge {e}")
            if not self._alive[v] and self._vert_edges[v]:
                errors.append(f"deleted vertex {v} still has edges")
        if not np.all(np.isfinite(self._pos[: self._nv][self._alive[: self._nv]])):
            errors.append("non-finite vertex positions")
        return errors


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between two non-zero vectors."""
    ua, ub = poly.unit(np.asarray(a, float)), poly.unit(np.asarray(b, float))
    if ua is None or ub is None:
        return math.nan
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(ua, ub))))))
