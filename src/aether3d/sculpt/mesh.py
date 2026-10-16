"""SculptMesh: the data-oriented triangle mesh that sculpting works on.

Representation (see docs/MODELING.md, "The sculpt mesh")
---------------------------------------------------------
An immutable :class:`~aether3d.Mesh` splits a vertex wherever its UVs or normals differ
(a seam). Sculpting must move such vertices together, so the sculpt mesh separates
geometry from attributes:

* **welded vertices** -- one per distinct position (bit-identical float32 positions of
  the source mesh, ``-0.0 == 0.0``), in growable ``float64`` arrays: ``_pos``;
* **render vertices** -- the source mesh's vertices, each pointing at its welded vertex
  (``_rgeo``) and carrying its UV (``_ruv``), its stored normal (``_rnorm``) and its
  *normal group* (``_rng``): render vertices of one welded vertex with bit-identical
  normals share a group, so a smooth surface stays smooth across a UV seam and a hard
  edge stays hard;
* **triangles** -- welded corners (``_tri``) and render corners (``_trv``), the face
  group (``_tgroup``), an alive flag.

Committing (:meth:`SculptMesh.to_mesh`) writes the render vertices back: positions come
from the welded vertices; normals of groups whose geometry changed are recomputed (angle
weighted over the group's triangles), all others are the stored ones bit for bit; UVs,
triangles and groups are reused as they are. A mesh nothing changed commits as the very
object it was made from.

Spatial chunks
--------------
Triangles are partitioned into **chunks** of at most :data:`CHUNK_TRIANGLES` by the
Morton (Z-order) codes of their centroids, split at the highest differing bit, so each
chunk is an octree cell (no chunk spans a jump of the Z curve). Every chunk keeps a box
containing all vertices of its triangles, so the chunks a sphere around a point
reaches hold *every* triangle incident to a vertex inside the sphere. Boxes only grow as
vertices move and are tightened again at the end of each operation.

Normals for brushes are kept incrementally: each triangle's cross product
``(b - a) x (c - a)`` and, per vertex, the sum of its triangles' cross products
(area-weighted normal). Moving vertices updates both by the change; the journal keeps
the previous values, so undo restores them bit for bit.

History
-------
Every change goes through journaled primitives inside an operation (a stroke, a mask
change, a remesh): positions, masks, the "changed" flags and the derived normal data.
If an operation raises, its journal is replayed backwards. :meth:`undo` replays the last
operation's journal backwards; :meth:`redo` runs it again with the same arguments,
which gives the same result (everything here is deterministic).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import SculptError, ValidationError
from ..mesh import Mesh
from ..rendering.raytrace.bvh import morton_codes

CHUNK_TRIANGLES = 1024
"""Most triangles in one spatial chunk."""

MAX_ELEMENTS = 20_000_000
"""Most vertices or triangles a sculpt mesh may hold."""

_VOID12 = np.dtype((np.void, 12))


def _grow(a: np.ndarray, n: int, fill: Any = 0) -> np.ndarray:
    """``a`` with room for at least ``n`` rows (doubling), new rows set to ``fill``."""
    if n <= len(a):
        return a
    size = max(n, 2 * len(a), 16)
    out = np.empty((size,) + a.shape[1:], dtype=a.dtype)
    out[: len(a)] = a
    out[len(a) :] = fill
    return out


@dataclass
class SculptResult:
    """What one sculpt operation did.

    ``data`` holds operation-specific outputs (for strokes: dabs, moved vertices, the
    largest displacement); ``timings`` the seconds spent per stage; ``diagnostics`` the
    findings (stable codes).
    """

    operation: str
    data: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def plain(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(k): plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(v) for v in value]
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "operation": self.operation,
            "data": plain(self.data),
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


@dataclass
class _Entry:
    name: str
    args: tuple
    kwargs: dict
    journal: list
    nbytes: int
    result: SculptResult


def sculpt_operation(func: Callable) -> Callable:
    """Make a method an atomic, undoable sculpt operation returning a :class:`SculptResult`.

    The wrapped function returns ``data`` and may append diagnostics to
    ``self._op_diagnostics`` and timings to ``self._op_timings``. Nested operations share
    the outer transaction.
    """
    name = func.__name__

    @functools.wraps(func)
    def wrapper(self: SculptMesh, *args: Any, **kwargs: Any) -> SculptResult:
        if self._journal is not None:
            return func(self, *args, **kwargs)
        if self._closed:
            raise SculptError(
                "SESSION_CLOSED",
                "this sculpt mesh belongs to a session that was committed or discarded",
                hint="start a new session with obj.sculpt()",
            )
        self._journal = []
        self._op_diagnostics = []
        self._op_timings = {}
        self._serial += 1
        try:
            data = func(self, *args, **kwargs) or {}
            self._tighten_boxes()
        except BaseException:
            journal, self._journal = self._journal, None
            self._rollback(journal)
            raise
        journal, self._journal = self._journal, None
        result = SculptResult(name, data, dict(self._op_timings), list(self._op_diagnostics))
        entry = _Entry(name, args, kwargs, journal, _journal_bytes(journal), result)
        self._history.append(entry)
        self._trim_history()
        if not self._redoing:
            self._redo.clear()
        return result

    wrapper.is_sculpt_operation = True  # type: ignore[attr-defined]
    return wrapper


def _validity(summary: dict, triangles: np.ndarray, vertex_count: int) -> dict:
    """The validity metrics a remesh reports (from a :func:`check_mesh` summary)."""
    from ..spatial.geometry import component_labels

    labels = component_labels(triangles, vertex_count)
    keys = (
        "closed",
        "edge_manifold",
        "vertex_manifold",
        "consistently_oriented",
        "boundary_edge_count",
        "non_manifold_edge_count",
        "degenerate_triangle_count",
        "euler_characteristic",
        "min_angle_degrees",
    )
    out = {k: summary[k] for k in keys}
    out["volume"] = summary.get("volume")
    out["components"] = int(labels.max()) + 1 if len(labels) and labels.max() >= 0 else 0
    out["triangles"] = int(summary["triangle_count"])
    out["vertices"] = int(summary["vertex_count"])
    return out


def _journal_bytes(journal: list) -> int:
    def size(item: Any) -> int:
        if isinstance(item, np.ndarray):
            return item.nbytes
        if isinstance(item, dict):
            return sum(size(x) for x in item.values())
        if isinstance(item, (list, tuple)):
            return sum(size(x) for x in item)
        return 0

    return sum(size(item) for entry in journal for item in entry[1:])


class SculptMesh:
    """A triangle mesh prepared for sculpting (see the module documentation).

    Make one with :meth:`from_mesh`; :meth:`to_mesh` gives the committed
    :class:`~aether3d.Mesh`. In a scene, ``obj.sculpt()`` opens a
    :class:`~aether3d.sculpt.SculptSession` that commits to the object.
    """

    max_history: int = 256
    max_history_bytes: int = 512 * 2**20

    def __init__(self) -> None:
        self._pos = np.zeros((0, 3))
        self._valive = np.zeros(0, dtype=bool)
        self._nv = 0
        self._tri = np.zeros((0, 3), dtype=np.int64)
        self._trv = np.zeros((0, 3), dtype=np.int64)
        self._talive = np.zeros(0, dtype=bool)
        self._tgroup = np.zeros(0, dtype=np.int64)
        self._nt = 0
        self._rgeo = np.zeros(0, dtype=np.int64)
        self._ruv: np.ndarray | None = None
        self._rnorm: np.ndarray | None = None
        self._rng = np.zeros(0, dtype=np.int64)
        self._nr = 0
        self._has_groups = False
        self._mask = np.zeros(0, dtype=np.float32)
        self._dirty = np.zeros(0, dtype=bool)  # welded vertices moved since the source
        self._cross = np.zeros((0, 3))  # per triangle (b - a) x (c - a)
        self._vsum = np.zeros((0, 3))  # per vertex: the sum of its triangles' crosses
        self._stamp = np.zeros(0, dtype=np.int64)  # last operation that journaled a vertex
        self._chunk_of = np.zeros(0, dtype=np.int64)
        self._chunks: list[np.ndarray] = []
        self._chunk_lo = np.zeros((0, 3))
        self._chunk_hi = np.zeros((0, 3))
        self._loose_boxes: set[int] = set()
        self._topology_changed = False
        self._source: Mesh | None = None
        self._journal: list | None = None
        self._op_diagnostics: list[Diagnostic] = []
        self._op_timings: dict[str, float] = {}
        self._history: list[_Entry] = []
        self._redo: list[tuple[str, tuple, dict]] = []
        self._redoing = False
        self._serial = 0
        self._closed = False
        self.world_matrix = np.eye(4)
        self.source_diagnostics: list[Diagnostic] = []

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> SculptMesh:
        """A sculpt mesh of ``mesh``: vertices with bit-identical positions are welded."""
        if not isinstance(mesh, Mesh):
            raise ValidationError(f"expected a Mesh, got {type(mesh).__name__}")
        if mesh.vertex_count > MAX_ELEMENTS or mesh.triangle_count > MAX_ELEMENTS:
            raise SculptError(
                "TOO_MANY_ELEMENTS", f"meshes of more than {MAX_ELEMENTS} elements are refused"
            )
        sm = cls()
        sm._source = mesh
        n = mesh.vertex_count
        if n:
            keys = np.ascontiguousarray(mesh.positions + np.float32(0.0)).view(_VOID12).ravel()
            _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
            order = np.argsort(first, kind="stable")
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            rgeo = rank[inverse.reshape(-1)].astype(np.int64)
            positions = mesh.positions[first[order]].astype(np.float64)
        else:
            rgeo = np.zeros(0, dtype=np.int64)
            positions = np.zeros((0, 3))
        nv = len(positions)
        sm._pos = positions.copy()
        sm._valive = np.ones(nv, dtype=bool)
        sm._nv = nv
        sm._mask = np.zeros(nv, dtype=np.float32)
        sm._dirty = np.zeros(nv, dtype=bool)
        sm._stamp = np.zeros(nv, dtype=np.int64)
        sm._rgeo = rgeo
        sm._nr = n
        sm._ruv = None if mesh.uvs is None else mesh.uvs.astype(np.float64)
        sm._rnorm = None if mesh.normals is None else mesh.normals.astype(np.float64)
        if mesh.normals is not None and n:
            nkeys = np.concatenate(
                [rgeo[:, None].astype(np.float64).view(np.uint64), mesh.normals.view(np.uint32)],
                axis=1,
            )
            _, ngroup = np.unique(nkeys, axis=0, return_inverse=True)
            sm._rng = ngroup.reshape(-1).astype(np.int64)
        else:
            sm._rng = rgeo.copy()
        trv = mesh.triangles.astype(np.int64)
        sm._trv = trv.copy()
        sm._tri = rgeo[trv] if len(trv) else np.zeros((0, 3), dtype=np.int64)
        nt = len(trv)
        sm._nt = nt
        sm._talive = np.ones(nt, dtype=bool)
        sm._has_groups = mesh.groups is not None
        sm._tgroup = (
            mesh.groups.astype(np.int64) if mesh.groups is not None else np.zeros(nt, np.int64)
        )
        degenerate = (
            (sm._tri[:, 0] == sm._tri[:, 1])
            | (sm._tri[:, 1] == sm._tri[:, 2])
            | (sm._tri[:, 0] == sm._tri[:, 2])
        )
        if degenerate.any():
            sm.source_diagnostics.append(
                Diagnostic(
                    "DEGENERATE_TRIANGLES",
                    INFO,
                    "triangles that repeat a (welded) vertex are kept as they are; brushes move "
                    "their vertices with the rest",
                    None,
                    int(degenerate.sum()),
                    tuple(int(i) for i in np.flatnonzero(degenerate)[:10]),
                )
            )
        sm._cross = sm._triangle_cross(np.arange(nt))
        sm._vsum = np.zeros((nv, 3))
        for k in range(3):
            np.add.at(sm._vsum, sm._tri[:nt, k], sm._cross)
        sm._build_chunks()
        return sm

    def to_mesh(self) -> Mesh:
        """The committed immutable :class:`Mesh` (see the module documentation)."""
        if not self.has_changes and self._source is not None:
            return self._source
        if self._topology_changed:
            return self._compact_mesh()
        rgeo = self._rgeo[: self._nr]
        positions = self._pos[rgeo].astype(np.float32)
        normals = None
        if self._rnorm is not None:
            normals = self._render_normals()
        source = self._source
        return Mesh._from_arrays(
            positions,
            source.triangles if source is not None else self._trv[: self._nt].astype(np.uint32),
            normals,
            source.uvs if source is not None else None,
            source.groups if source is not None else None,
        )

    def _render_normals(self) -> np.ndarray:
        """Render-vertex normals: stored ones where nothing changed, recomputed (angle
        weighted over each normal group's triangles) where the geometry did."""
        nr = self._nr
        stored = self._rnorm[:nr]
        out = stored.astype(np.float32)
        nt = self._nt
        alive = self._talive[:nt]
        tri, trv = self._tri[:nt], self._trv[:nt]
        dirty_v = self._dirty[: self._nv]
        if not dirty_v.any():
            return out
        # Triangles with a moved vertex change the normals of all three corners' groups.
        touched = alive & dirty_v[tri].any(axis=1)
        groups = np.unique(self._rng[trv[touched]])
        if not len(groups):
            return out
        need = np.zeros(int(self._rng[:nr].max()) + 1, dtype=bool)
        need[groups] = True
        corner_groups = self._rng[trv]  # (t, 3)
        rows = np.flatnonzero(alive & need[corner_groups].any(axis=1))
        p = self._pos[tri[rows]]  # (r, 3, 3)
        cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        length = np.sqrt((cross**2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(length[:, None] > 0, cross / length[:, None], 0.0)
        sums = np.zeros((len(need), 3))
        for k in range(3):
            a = p[:, (k + 1) % 3] - p[:, k]
            b = p[:, (k + 2) % 3] - p[:, k]
            la = np.sqrt((a**2).sum(axis=1))
            lb = np.sqrt((b**2).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                cosine = np.clip((a * b).sum(axis=1) / (la * lb), -1.0, 1.0)
            angle = np.where((la > 0) & (lb > 0), np.arccos(cosine), 0.0)
            g = corner_groups[rows, k]
            pick = need[g]
            np.add.at(sums, g[pick], unit[pick] * angle[pick, None])
        length = np.sqrt((sums**2).sum(axis=1))
        ok = length > 0
        fresh = np.zeros_like(sums)
        fresh[ok] = sums[ok] / length[ok, None]
        recompute = need[self._rng[:nr]] & ok[self._rng[:nr]]
        out[recompute] = fresh[self._rng[:nr][recompute]].astype(np.float32)
        return out

    def _compact_mesh(self) -> Mesh:
        """Commit after topology changes: alive elements renumbered in order."""
        nt, nr = self._nt, self._nr
        alive = np.flatnonzero(self._talive[:nt])
        trv = self._trv[alive]
        used = np.zeros(nr, dtype=bool)
        used[trv.ravel()] = True
        remap = np.full(nr, -1, dtype=np.int64)
        keep = np.flatnonzero(used)
        remap[keep] = np.arange(len(keep))
        positions = self._pos[self._rgeo[keep]].astype(np.float32)
        normals = None
        if self._rnorm is not None:
            normals = self._render_normals()[keep]
        uvs = None if self._ruv is None else self._ruv[keep].astype(np.float32)
        groups = self._tgroup[alive] if self._has_groups else None
        return Mesh(positions, remap[trv], normals=normals, uvs=uvs, groups=groups)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    @property
    def vertex_count(self) -> int:
        return int(self._valive[: self._nv].sum())

    @property
    def triangle_count(self) -> int:
        return int(self._talive[: self._nt].sum())

    @property
    def has_changes(self) -> bool:
        return bool(self._dirty[: self._nv].any()) or self._topology_changed

    @property
    def has_uvs(self) -> bool:
        return self._ruv is not None

    def positions(self) -> np.ndarray:
        """``(n, 3)`` positions of the alive vertices, by id (a copy)."""
        return self._pos[: self._nv][self._valive[: self._nv]].copy()

    def vertex_ids(self) -> np.ndarray:
        return np.flatnonzero(self._valive[: self._nv])

    def triangles(self) -> np.ndarray:
        """``(m, 3)`` welded vertex ids of the alive triangles (a copy)."""
        return self._tri[: self._nt][self._talive[: self._nt]].copy()

    def triangle_ids(self) -> np.ndarray:
        return np.flatnonzero(self._talive[: self._nt])

    def mask(self) -> np.ndarray:
        """Mask value of every alive vertex (0 free, 1 frozen), by id."""
        return self._mask[: self._nv][self._valive[: self._nv]].copy()

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        p = self.positions()
        if not len(p):
            return None
        return tuple(p.min(axis=0).tolist()), tuple(p.max(axis=0).tolist())

    def scale(self) -> float:
        b = self.bounds()
        if b is None:
            return 1.0
        return max(float(np.max(np.subtract(b[1], b[0]))), 1e-12)

    def vertex_normals(self, ids: np.ndarray | None = None) -> np.ndarray:
        """Area-weighted unit vertex normals (zero where undefined)."""
        idx = self.vertex_ids() if ids is None else np.asarray(ids, dtype=np.int64)
        s = self._vsum[idx]
        length = np.sqrt((s**2).sum(axis=1))
        out = np.zeros_like(s)
        ok = length > 0
        out[ok] = s[ok] / length[ok, None]
        return out

    def chunk_count(self) -> int:
        return len(self._chunks)

    def region(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Ids of the alive triangles of every chunk whose box meets the sphere: a superset
        of the triangles incident to the vertices inside it."""
        if not self._chunks:
            return np.zeros(0, dtype=np.int64)
        c = np.asarray(center, dtype=np.float64)
        d = np.maximum(np.maximum(self._chunk_lo - c, c - self._chunk_hi), 0.0)
        hit = np.flatnonzero((d * d).sum(axis=1) <= radius * radius)
        if not len(hit):
            return np.zeros(0, dtype=np.int64)
        tris = np.concatenate([self._chunks[i] for i in hit])
        return tris[self._talive[tris]]

    def vertices_in_sphere(
        self, center: np.ndarray, radius: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(vertex ids, distances, region triangles)`` within ``radius`` of ``center``."""
        tris = self.region(center, radius)
        if not len(tris):
            return np.zeros(0, dtype=np.int64), np.zeros(0), tris
        corners = self._tri[tris].ravel()
        d = self._pos[corners] - center
        near = (d * d).sum(axis=1) <= radius * radius
        verts = np.unique(corners[near])
        d = self._pos[verts] - center
        return verts, np.sqrt((d * d).sum(axis=1)), tris

    def raycast(
        self, origin: np.ndarray, direction: np.ndarray, tmax: float = math.inf
    ) -> tuple[float, int, np.ndarray] | None:
        """The closest hit of one ray on the current surface: ``(t, triangle, bary)``."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if not self._chunks:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            t0 = (self._chunk_lo - o) * inv
            t1 = (self._chunk_hi - o) * inv
        near = np.nanmax(np.where(np.isnan(t0), -np.inf, np.minimum(t0, t1)), axis=1)
        far = np.nanmin(np.where(np.isnan(t0), np.inf, np.maximum(t0, t1)), axis=1)
        near = np.maximum(near, 0.0)
        far = np.minimum(far, tmax)
        hit = np.flatnonzero(near <= far * (1 + 1e-12) + 1e-300)
        if not len(hit):
            return None
        best: tuple[float, int, np.ndarray] | None = None
        for ci in hit[np.argsort(near[hit], kind="stable")]:
            if best is not None and near[ci] > best[0]:
                break
            tris = self._chunks[ci]
            tris = tris[self._talive[tris]]
            if not len(tris):
                continue
            p = self._pos[self._tri[tris]]
            t, u, v = _ray_triangles(o, d, p[:, 0], p[:, 1], p[:, 2])
            ok = (t > 0) & (t <= tmax)
            if not ok.any():
                continue
            k = np.flatnonzero(ok)
            j = k[np.lexsort((tris[k], t[k]))[0]]
            if best is None or t[j] < best[0]:
                best = (float(t[j]), int(tris[j]), np.array([1 - u[j] - v[j], u[j], v[j]]))
        return best

    def nearest_point(self, point: np.ndarray) -> tuple[np.ndarray, float, int] | None:
        """The closest point of the current surface: ``(point, distance, triangle)``.

        The nearest vertex of the chunk with the nearest box bounds the distance; every
        triangle (of a chunk whose box is no farther) whose own box is within that bound
        is then measured at once. Exact: a nearer triangle passes both filters."""
        from ..spatial.geometry import closest_points_on_triangles

        if not self._chunks:
            return None
        q = np.asarray(point, dtype=np.float64)
        dbox = np.maximum(np.maximum(self._chunk_lo - q, q - self._chunk_hi), 0.0)
        box_d = np.sqrt((dbox * dbox).sum(axis=1))
        bound = np.inf
        for i in np.argsort(box_d, kind="stable"):
            tris = self._chunks[i]
            tris = tris[self._talive[tris]]
            if len(tris):
                v = self._pos[self._tri[tris].ravel()] - q
                bound = float(np.sqrt((v * v).sum(axis=1).min()))
                break
        if not np.isfinite(bound):
            return None
        chunks = np.flatnonzero(box_d <= bound)
        tris = np.concatenate([self._chunks[i] for i in chunks])
        tris = tris[self._talive[tris]]
        p = self._pos[self._tri[tris]]  # (t, 3, 3)
        lo, hi = p.min(axis=1), p.max(axis=1)
        gap = np.maximum(np.maximum(lo - q, q - hi), 0.0)
        near = (gap * gap).sum(axis=1) <= bound * bound * (1 + 1e-12)
        tris, p = tris[near], p[near]
        cp, _ = closest_points_on_triangles(q, p[:, 0], p[:, 1], p[:, 2])
        dist = np.sqrt(((cp - q) ** 2).sum(axis=1))
        j = int(np.lexsort((tris, dist))[0])
        return cp[j], float(dist[j]), int(tris[j])

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #

    @sculpt_operation
    def stroke(self, stroke: Any):
        """Apply a brush stroke (a :class:`~aether3d.sculpt.Stroke` or its JSON form) as
        one undoable step. ``data``: dabs made, applied and skipped, the vertices moved
        and the largest displacement (world units)."""
        from .stroke import Stroke, StrokeRunner, stroke_diagnostics

        st = stroke if isinstance(stroke, Stroke) else Stroke.from_dict(stroke)
        st.check()
        runner = StrokeRunner(self, st)
        data = runner.run()
        for key, value in runner.timings.items():
            self._op_timings[key] = self._op_timings.get(key, 0.0) + value
        for d in stroke_diagnostics(data):
            self._diagnose(d)
        return data

    @sculpt_operation
    def remesh(
        self,
        target_length: float,
        *,
        iterations: int = 5,
        flips: str = "valence",
        relax: float = 0.5,
        project: bool = True,
    ):
        """Isotropic remeshing toward edges of world length ``target_length`` (Botsch and
        Kobbelt 2004; see :mod:`aether3d.sculpt.remesh`), keeping features. ``data``:
        operations done and the resulting edge-length statistics."""
        from .remesh import isotropic_remesh

        return isotropic_remesh(
            self,
            float(target_length),
            iterations=iterations,
            flips=flips,
            relax=float(relax),
            project=project,
        )

    @sculpt_operation
    def voxel_remesh(
        self,
        voxel_size: float,
        *,
        mode: str = "solid",
        thickness: float | None = None,
        max_voxels: int | None = None,
    ):
        """Rebuild the whole mesh from a voxel grid of spacing ``voxel_size`` (world
        units): a signed distance field on a narrow band, then marching cubes (see
        :mod:`aether3d.sculpt.voxel`). ``mode="solid"`` needs closed, manifold,
        consistently oriented parts (overlapping parts fuse; inward-facing ones are
        cavities); ``mode="shell"`` wraps any mesh in a closed shell of ``thickness``.

        Every vertex and triangle is new. Kept: face groups (each new triangle takes the
        group of the nearest old triangle) and the mask (interpolated at the nearest old
        point). Not kept, and reported: UVs (``VOXEL_UVS_DROPPED``) and stored normals
        (recomputed smooth). ``data`` holds validity metrics of the input and the result
        (closed, manifold, oriented, volume, Euler characteristic, components, smallest
        angle) and the grid's size."""
        from ..spatial.geometry import component_labels
        from ..spatial.surface import TriangleSurface
        from .validate import check_mesh
        from .voxel import MAX_VOXELS, check_voxel_args, voxel_surface

        limit = MAX_VOXELS if max_voxels is None else max_voxels
        check_voxel_args(voxel_size, mode, thickness, limit)
        tris = self.triangle_ids()
        if not len(tris):
            raise SculptError("EMPTY_MESH", "there are no triangles to remesh")
        used = np.unique(self._tri[tris])
        remap = np.full(self._nv, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        t = remap[self._tri[tris]]
        m = np.asarray(self.world_matrix, dtype=np.float64)
        world = self._pos[used] @ m[:3, :3].T + m[:3, 3]
        before = check_mesh(world, t, welded=True)
        component = outward = None
        if mode == "solid":
            s = before.summary
            if not (
                s["closed"]
                and s["vertex_manifold"]
                and s["consistently_oriented"]
                and s["degenerate_triangle_count"] == 0
            ):
                raise SculptError(
                    "VOXEL_NOT_WATERTIGHT",
                    "a solid voxel remesh needs closed, manifold, consistently oriented parts "
                    f"without degenerate triangles; this mesh has {s['boundary_edge_count']} "
                    f"boundary edges, {s['non_manifold_edge_count']} non-manifold edges, "
                    f"{s['non_manifold_vertex_count']} non-manifold vertices and "
                    f"{s['degenerate_triangle_count']} degenerate triangles"
                    + ("" if s["consistently_oriented"] else ", and inconsistent winding"),
                    hint="close the holes first (EditableMesh.fill_holes), or use "
                    "mode='shell' with a thickness",
                )
            component = component_labels(t, len(world))
            a, b, c = world[t[:, 0]], world[t[:, 1]], world[t[:, 2]]
            volume = np.bincount(component[t[:, 0]], weights=(a * np.cross(b, c)).sum(axis=1) / 6.0)
            outward = volume >= 0
            if not outward.any():
                raise SculptError(
                    "VOXEL_INSIDE_OUT",
                    "every closed part faces inward (negative volume)",
                    hint="flip the normals outward first (recalculate_normals)",
                )
        positions, triangles, stats = voxel_surface(
            world,
            t,
            float(voxel_size),
            mode=mode,
            thickness=None if thickness is None else float(thickness),
            max_voxels=limit,
            component=component,
            outward=outward,
        )
        if not len(triangles):
            raise SculptError(
                "VOXEL_EMPTY",
                "no surface is left at this voxel size (the mesh is thinner than about two voxels)",
                hint="use a smaller voxel_size",
            )
        inverse = np.linalg.inv(m)
        local = positions @ inverse[:3, :3].T + inverse[:3, 3]
        source = TriangleSurface(world, t)
        groups = np.zeros(len(triangles), dtype=np.int64)
        if self._has_groups:
            centre = positions[triangles].mean(axis=1)
            groups = self._tgroup[tris][source.nearest(centre, batch=1 << 15).triangles]
        mask = np.zeros(len(positions), dtype=np.float32)
        old_mask = self._mask[used]
        if old_mask.any():
            near = source.nearest(positions, batch=1 << 15)
            corner = old_mask[t[near.triangles]].astype(np.float64)
            mask = np.clip((corner * near.bary).sum(axis=1), 0.0, 1.0).astype(np.float32)
        if self._ruv is not None:
            self._op_diagnostics.append(
                Diagnostic(
                    "VOXEL_UVS_DROPPED",
                    WARNING,
                    "UVs cannot follow a new topology: the remeshed mesh has none (unwrap it "
                    "again, or map textures with object coordinates)",
                    None,
                    int(self._nr),
                    (),
                )
            )
        if self._rnorm is not None:
            self._op_diagnostics.append(
                Diagnostic(
                    "VOXEL_NORMALS_RECOMPUTED",
                    INFO,
                    "stored normals were replaced by smooth normals of the new surface",
                    None,
                    0,
                    (),
                )
            )
        self._replace(local, triangles, groups, mask, normals=self._rnorm is not None)
        after = check_mesh(positions, triangles, welded=True)
        return {
            "voxel_size": float(voxel_size),
            "mode": mode,
            "grid": stats["grid"],
            "band_points": stats["band_points"],
            "vertices": int(len(positions)),
            "triangles": int(len(triangles)),
            "input": _validity(before.summary, t, len(world)),
            "output": _validity(after.summary, triangles, len(positions)),
        }

    @sculpt_operation
    def set_mask(
        self,
        value: float = 1.0,
        *,
        vertices: Any = None,
        sphere: Any = None,
        box: Any = None,
        groups: Any = None,
        mode: str = "set",
    ):
        """Set (``mode="set"``), raise (``"add"``) or lower (``"subtract"``) the mask of
        vertices: given ids, inside a ``sphere`` ``(center, radius)``, inside a ``box``
        ``(min, max)``, or on faces of the face ``groups`` (all vertices without any of
        these). Coordinates are local. Masked vertices (1) do not move."""
        v = float(value)
        if not 0.0 <= v <= 1.0:
            raise ValidationError("mask value must be in [0, 1]")
        if mode not in ("set", "add", "subtract"):
            raise ValidationError("mode must be 'set', 'add' or 'subtract'")
        ids = self._select(vertices=vertices, sphere=sphere, box=box, groups=groups)
        current = self._mask[ids].astype(np.float64)
        new = {"set": np.full(len(ids), v), "add": current + v, "subtract": current - v}[mode]
        self._set_mask(ids, new)
        return {"vertices": len(ids)}

    @sculpt_operation
    def invert_mask(self):
        """Mask becomes ``1 - mask`` everywhere."""
        ids = self.vertex_ids()
        self._set_mask(ids, 1.0 - self._mask[ids].astype(np.float64))
        return {"vertices": len(ids)}

    @sculpt_operation
    def clear_mask(self):
        """Unmask every vertex."""
        ids = self.vertex_ids()
        self._set_mask(ids, np.zeros(len(ids)))
        return {"vertices": len(ids)}

    def _select(self, *, vertices=None, sphere=None, box=None, groups=None) -> np.ndarray:
        ids = self.vertex_ids()
        keep = np.ones(len(ids), dtype=bool)
        if vertices is not None:
            chosen = np.asarray(sorted({int(x) for x in vertices}), dtype=np.int64)
            bad = chosen[(chosen < 0) | (chosen >= self._nv)]
            if len(bad) or (len(chosen) and not self._valive[chosen].all()):
                raise SculptError(
                    "INVALID_ELEMENT",
                    "some vertices do not exist",
                    elements={"vertices": bad.tolist()},
                )
            keep &= np.isin(ids, chosen)
        if sphere is not None:
            center, radius = sphere
            c = np.asarray(center, dtype=np.float64)
            r = float(radius)
            if c.shape != (3,) or not np.all(np.isfinite(c)) or not r >= 0:
                raise ValidationError("sphere is ((x, y, z), radius >= 0)")
            keep &= ((self._pos[ids] - c) ** 2).sum(axis=1) <= r * r
        if box is not None:
            lo, hi = (np.asarray(x, dtype=np.float64) for x in box)
            if lo.shape != (3,) or hi.shape != (3,):
                raise ValidationError("box is ((min x, y, z), (max x, y, z))")
            keep &= np.all((self._pos[ids] >= lo) & (self._pos[ids] <= hi), axis=1)
        if groups is not None:
            wanted = np.asarray(list(groups), dtype=np.int64)
            nt = self._nt
            rows = np.flatnonzero(self._talive[:nt] & np.isin(self._tgroup[:nt], wanted))
            on = np.zeros(self._nv, dtype=bool)
            on[self._tri[rows].ravel()] = True
            keep &= on[ids]
        return ids[keep]

    # ------------------------------------------------------------------ #
    # Chunks
    # ------------------------------------------------------------------ #

    def _build_chunks(self) -> None:
        nt = self._nt
        alive = np.flatnonzero(self._talive[:nt])
        self._chunk_of = np.full(len(self._talive), -1, dtype=np.int64)
        if not len(alive):
            self._chunks = []
            self._chunk_lo = np.zeros((0, 3))
            self._chunk_hi = np.zeros((0, 3))
            return
        p = self._pos[self._tri[alive]]
        centroid = p.mean(axis=1)
        codes = morton_codes(centroid)
        order = np.argsort(codes, kind="stable")
        codes = codes[order]
        sorted_tris = alive[order]
        # Split ranges at the highest bit their first and last codes differ in.
        starts, ends = [0], [len(order)]
        final: list[tuple[int, int]] = []
        while starts:
            s_arr, e_arr = np.array(starts), np.array(ends)
            big = (e_arr - s_arr) > CHUNK_TRIANGLES
            final += list(zip(s_arr[~big].tolist(), e_arr[~big].tolist(), strict=True))
            s_arr, e_arr = s_arr[big], e_arr[big]
            if not len(s_arr):
                break
            first, last = codes[s_arr], codes[e_arr - 1]
            diff = first ^ last
            mid = (s_arr + e_arr) // 2
            split = mid.copy()
            has = diff > 0
            if has.any():
                bit = np.floor(np.log2(diff[has].astype(np.float64))).astype(np.int64)
                # float64 log2 may round up for codes near a power of two: correct it.
                bit = np.where((np.uint64(1) << bit.astype(np.uint64)) > diff[has], bit - 1, bit)
                threshold = ((first[has] >> bit.astype(np.uint64)) | np.uint64(1)) << bit.astype(
                    np.uint64
                )
                cut = np.searchsorted(codes, threshold, side="left")
                split[has] = np.clip(cut, s_arr[has] + 1, e_arr[has] - 1)
            starts = np.concatenate([s_arr, split]).tolist()
            ends = np.concatenate([split, e_arr]).tolist()
        final.sort()
        self._chunks = [np.sort(sorted_tris[s:e]) for s, e in final]
        for i, tris in enumerate(self._chunks):
            self._chunk_of[tris] = i
        self._chunk_lo = np.empty((len(self._chunks), 3))
        self._chunk_hi = np.empty((len(self._chunks), 3))
        self._refit_chunks(np.arange(len(self._chunks)))

    def _refit_chunks(self, chunk_ids: np.ndarray) -> None:
        for i in np.asarray(chunk_ids, dtype=np.int64).tolist():
            tris = self._chunks[i]
            tris = tris[self._talive[tris]]
            if not len(tris):
                self._chunk_lo[i] = np.inf
                self._chunk_hi[i] = -np.inf
                continue
            p = self._pos[self._tri[tris].ravel()]
            self._chunk_lo[i] = p.min(axis=0)
            self._chunk_hi[i] = p.max(axis=0)

    def _grow_boxes(self, tris: np.ndarray) -> None:
        """Grow the boxes of the chunks of ``tris`` to hold their current vertices (they
        are tightened again when the operation ends)."""
        if not len(tris):
            return
        chunks = self._chunk_of[tris]
        order = np.argsort(chunks, kind="stable")
        chunks = chunks[order]
        p = self._pos[self._tri[tris[order]]]
        starts = np.flatnonzero(np.r_[True, chunks[1:] != chunks[:-1]])
        ids = chunks[starts]
        lo = np.minimum.reduceat(p.min(axis=1), starts, axis=0)
        hi = np.maximum.reduceat(p.max(axis=1), starts, axis=0)
        self._chunk_lo[ids] = np.minimum(self._chunk_lo[ids], lo)
        self._chunk_hi[ids] = np.maximum(self._chunk_hi[ids], hi)
        self._loose_boxes.update(ids.tolist())

    def _tighten_boxes(self) -> None:
        if self._loose_boxes:
            count = len(self._chunks)
            ids = [i for i in sorted(self._loose_boxes) if i < count]  # (rollback drops chunks)
            if ids:
                self._refit_chunks(np.array(ids))
            self._loose_boxes.clear()

    # ------------------------------------------------------------------ #
    # Journaled primitives
    # ------------------------------------------------------------------ #

    def _triangle_cross(self, tris: np.ndarray) -> np.ndarray:
        p = self._pos[self._tri[tris]]
        return np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])

    def _log(self, entry: tuple) -> None:
        if self._journal is not None:
            self._journal.append(entry)

    def _move(self, ids: np.ndarray, new_positions: np.ndarray, region: np.ndarray) -> None:
        """Move vertices ``ids`` (unique) to ``new_positions``. ``region`` must contain
        every alive triangle incident to them (:meth:`region` of a sphere holding them)."""
        if not len(ids):
            return
        if not np.all(np.isfinite(new_positions)):
            raise SculptError("NON_FINITE_RESULT", "a brush produced non-finite positions")
        serial = self._serial
        first = ids[self._stamp[ids] != serial]
        if len(first):
            self._log(("pos", first, self._pos[first].copy()))
            self._stamp[first] = serial
            clean = first[~self._dirty[first]]
            if len(clean):
                self._log(("dirty", clean))
                self._dirty[clean] = True
        mark = self._scratch_mark()
        mark[ids] = True
        tris = region[mark[self._tri[region]].any(axis=1)]
        mark[ids] = False
        corner = self._tri[tris]
        touched = np.unique(corner)
        self._log(("cross", tris, self._cross[tris].copy()))
        self._log(("vsum", touched, self._vsum[touched].copy()))
        self._pos[ids] = new_positions
        new = self._triangle_cross(tris)
        delta = new - self._cross[tris]
        self._cross[tris] = new
        # Scatter the changes onto the touched vertices (bincount over a local index).
        local = self._scratch_index()
        local[touched] = np.arange(len(touched))
        rows = local[corner].ravel()
        weights = np.repeat(delta, 3, axis=0)
        for k in range(3):
            self._vsum[touched, k] += np.bincount(
                rows, weights=weights[:, k], minlength=len(touched)
            )
        local[touched] = -1
        self._grow_boxes(tris)

    def _scratch_mark(self) -> np.ndarray:
        if len(getattr(self, "_mark", ())) < self._nv:
            self._mark = np.zeros(len(self._pos), dtype=bool)
        return self._mark

    def _scratch_index(self) -> np.ndarray:
        if len(getattr(self, "_index", ())) < self._nv:
            self._index = np.full(len(self._pos), -1, dtype=np.int64)
        return self._index

    def _set_mask(self, ids: np.ndarray, values: np.ndarray) -> None:
        if not len(ids):
            return
        self._log(("mask", ids, self._mask[ids].copy()))
        self._mask[ids] = np.clip(values, 0.0, 1.0).astype(np.float32)

    def _rollback(self, journal: list) -> None:
        for entry in reversed(journal):
            kind = entry[0]
            if kind == "pos":
                self._pos[entry[1]] = entry[2]
            elif kind == "dirty":
                self._dirty[entry[1]] = False
            elif kind == "cross":
                self._cross[entry[1]] = entry[2]
                self._loose_boxes.update(np.unique(self._chunk_of[entry[1]]).tolist())
            elif kind == "vsum":
                self._vsum[entry[1]] = entry[2]
            elif kind == "mask":
                self._mask[entry[1]] = entry[2]
            elif kind == "topology":
                self._restore_topology(entry[1])
            elif kind == "uv":
                self._ruv[entry[1]] = entry[2]
            elif kind == "state":
                for name, value in entry[1].items():
                    setattr(self, name, value)
                self._loose_boxes.clear()
            else:  # pragma: no cover
                raise AssertionError(f"unknown journal entry {kind}")
        self._tighten_boxes()

    def _restore_topology(self, state: dict) -> None:
        from .topology import restore_topology

        restore_topology(self, state)

    _limit = MAX_ELEMENTS

    _STATE_FIELDS = (
        "_pos",
        "_valive",
        "_nv",
        "_tri",
        "_trv",
        "_talive",
        "_tgroup",
        "_nt",
        "_rgeo",
        "_ruv",
        "_rnorm",
        "_rng",
        "_nr",
        "_mask",
        "_dirty",
        "_cross",
        "_vsum",
        "_stamp",
        "_chunk_of",
        "_chunks",
        "_chunk_lo",
        "_chunk_hi",
        "_topology_changed",
    )

    def _replace(
        self,
        positions: np.ndarray,
        triangles: np.ndarray,
        groups: np.ndarray,
        mask: np.ndarray,
        *,
        normals: bool,
    ) -> None:
        """Replace the whole mesh (a destructive remesh): one render vertex per vertex, no
        UVs, smooth normals recomputed at commit if the mesh had normals. Journaled by
        keeping the previous arrays themselves: nothing writes into them afterwards."""
        nv, nt = len(positions), len(triangles)
        if nv > self._limit or nt > self._limit:
            raise SculptError(
                "TOO_MANY_ELEMENTS", f"the mesh would exceed {self._limit} vertices or triangles"
            )
        self._log(("state", {name: getattr(self, name) for name in self._STATE_FIELDS}))
        self._pos = np.array(positions, dtype=np.float64)
        self._valive = np.ones(nv, dtype=bool)
        self._nv = nv
        self._tri = np.array(triangles, dtype=np.int64).reshape(-1, 3)
        self._trv = self._tri.copy()
        self._talive = np.ones(nt, dtype=bool)
        self._tgroup = np.array(groups, dtype=np.int64)
        self._nt = nt
        self._rgeo = np.arange(nv, dtype=np.int64)
        self._ruv = None
        self._rnorm = np.zeros((nv, 3)) if normals else None
        self._rng = np.arange(nv, dtype=np.int64)
        self._nr = nv
        self._mask = np.array(mask, dtype=np.float32)
        self._dirty = np.ones(nv, dtype=bool)
        self._stamp = np.zeros(nv, dtype=np.int64)
        self._cross = self._triangle_cross(np.arange(nt))
        self._vsum = np.zeros((nv, 3))
        for k in range(3):
            np.add.at(self._vsum, self._tri[:, k], self._cross)
        self._topology_changed = True
        self._loose_boxes.clear()
        self._build_chunks()

    def _ensure(self, nv: int, nt: int, nr: int) -> None:
        """Room for ``nv`` vertices, ``nt`` triangles and ``nr`` render vertices."""
        if nv > len(self._pos):
            self._pos = _grow(self._pos, nv)
            self._valive = _grow(self._valive, nv, False)
            self._mask = _grow(self._mask, nv)
            self._dirty = _grow(self._dirty, nv, False)
            self._stamp = _grow(self._stamp, nv)
            self._vsum = _grow(self._vsum, nv)
        if nt > len(self._tri):
            self._tri = _grow(self._tri, nt)
            self._trv = _grow(self._trv, nt)
            self._talive = _grow(self._talive, nt, False)
            self._tgroup = _grow(self._tgroup, nt)
            self._cross = _grow(self._cross, nt)
            self._chunk_of = _grow(self._chunk_of, nt, -1)
        if nr > len(self._rgeo):
            self._rgeo = _grow(self._rgeo, nr)
            self._rng = _grow(self._rng, nr)
            if self._ruv is not None:
                self._ruv = _grow(self._ruv, nr)
            if self._rnorm is not None:
                self._rnorm = _grow(self._rnorm, nr)

    def _split_big_chunks(self, chunk_ids: np.ndarray) -> None:
        """Halve chunks that grew past twice the chunk size (by Morton order of their
        triangles' centroids), so queries stay local after refinement."""
        work = [int(i) for i in np.asarray(chunk_ids).tolist()]
        while work:
            i = work.pop()
            tris = self._chunks[i]
            tris = tris[self._talive[tris]]
            if len(tris) <= 2 * CHUNK_TRIANGLES:
                self._chunks[i] = tris
                continue
            centroid = self._pos[self._tri[tris]].mean(axis=1)
            order = np.argsort(morton_codes(centroid), kind="stable")
            half = len(order) // 2
            keep, move = np.sort(tris[order[:half]]), np.sort(tris[order[half:]])
            self._chunks[i] = keep
            new_id = len(self._chunks)
            self._chunks.append(move)
            self._chunk_of[move] = new_id
            self._chunk_lo = np.vstack([self._chunk_lo, np.full((1, 3), np.inf)])
            self._chunk_hi = np.vstack([self._chunk_hi, np.full((1, 3), -np.inf)])
            work += [i, new_id]

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> list[dict[str, Any]]:
        return [{"operation": e.name, "result": e.result.to_dict()} for e in self._history]

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str:
        """Revert the most recent operation exactly; returns its name."""
        if self._journal is not None:
            raise SculptError("UNDO_IN_OPERATION", "cannot undo while an operation is running")
        if not self._history:
            raise SculptError("NOTHING_TO_UNDO", "there is no operation to undo")
        entry = self._history.pop()
        self._rollback(entry.journal)
        self._redo.append((entry.name, entry.args, entry.kwargs))
        return entry.name

    def redo(self) -> SculptResult:
        """Run the most recently undone operation again (same arguments, same result)."""
        if not self._redo:
            raise SculptError("NOTHING_TO_REDO", "there is no undone operation to redo")
        name, args, kwargs = self._redo.pop()
        self._redoing = True
        try:
            return getattr(self, name)(*args, **kwargs)
        finally:
            self._redoing = False

    def _trim_history(self) -> None:
        while len(self._history) > self.max_history or (
            len(self._history) > 1 and sum(e.nbytes for e in self._history) > self.max_history_bytes
        ):
            del self._history[0]

    @property
    def history_nbytes(self) -> int:
        """Bytes the undo journals hold."""
        return sum(e.nbytes for e in self._history)

    def _diagnose(self, diagnostic: Diagnostic) -> None:
        self._op_diagnostics.append(diagnostic)

    # ------------------------------------------------------------------ #
    # Consistency (tests)
    # ------------------------------------------------------------------ #

    def _consistency_errors(self) -> list[str]:
        errors = []
        nt, nv = self._nt, self._nv
        alive = np.flatnonzero(self._talive[:nt])
        tri = self._tri[alive]
        if len(tri) and (tri.min() < 0 or tri.max() >= nv or not self._valive[tri].all()):
            errors.append("a triangle uses a missing vertex")
        if len(alive) and not np.array_equal(self._rgeo[self._trv[alive]], tri):
            errors.append("render corners disagree with welded corners")
        cross = self._triangle_cross(alive)
        if not np.allclose(cross, self._cross[alive], rtol=1e-9, atol=1e-12 * self.scale() ** 2):
            errors.append("stored triangle cross products are stale")
        vsum = np.zeros((nv, 3))
        for k in range(3):
            np.add.at(vsum, tri[:, k], cross)
        scale = max(float(np.abs(vsum).max()) if len(vsum) else 1.0, 1e-300)
        if not np.allclose(vsum, self._vsum[:nv], atol=1e-9 * scale):
            errors.append("stored vertex normal sums are stale")
        for i, tris in enumerate(self._chunks):
            tris = tris[self._talive[tris]]
            if not len(tris):
                continue
            if np.any(self._chunk_of[tris] != i):
                errors.append(f"chunk {i} lists triangles of another chunk")
            p = self._pos[self._tri[tris].ravel()]
            if np.any(p < self._chunk_lo[i] - 1e-12) or np.any(p > self._chunk_hi[i] + 1e-12):
                errors.append(f"chunk {i}'s box does not hold its triangles")
        listed = np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.int64)
        listed = listed[self._talive[listed]]
        if len(listed) != len(alive) or not np.array_equal(np.sort(listed), alive):
            errors.append("chunks do not partition the alive triangles")
        if not np.all(np.isfinite(self._pos[:nv][self._valive[:nv]])):
            errors.append("non-finite vertex positions")
        return errors


def _ray_triangles(o, d, a, b, c):
    e1, e2 = b - a, c - a
    pvec = np.cross(d, e2)
    det = (e1 * pvec).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tvec = o - a
        u = (tvec * pvec).sum(axis=1) * inv
        qvec = np.cross(tvec, e1)
        v = (d * qvec).sum(axis=1) * inv
        t = (e2 * qvec).sum(axis=1) * inv
        miss = (det == 0) | ~np.isfinite(t) | (u < 0) | (v < 0) | (u + v > 1)
    return np.where(miss, np.inf, t), u, v
