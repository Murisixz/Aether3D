"""The spatial index of a scene: its geometry in world space, ready for queries.

A :class:`SpatialIndex` holds every object with geometry -- static meshes and the
current evaluation of procedural ones -- with its world matrix (parents, rotations,
non-uniform and mirrored scales included) and, built on demand and kept:

* world-space vertices and triangle corners (float64, from the meshes' float32
  positions);
* a world-space :class:`~aether3d.rendering.raytrace.BVH` over each object's
  triangles, for region, overlap and distance queries;
* the mesh's topology (closed, manifold, oriented) for containment;
* for ray queries, the Stage 8 two-level :class:`~aether3d.rendering.raytrace.SceneAccel`
  over the objects asked about (bottom levels come from the renderers' shared,
  content-keyed BLAS cache).

**Caching.** :func:`get_index` keeps one index per scene, valid while
``scene.revision`` is unchanged. The history increases the revision with every
change -- creation, removal, transforms, edits, procedural and material changes,
imports, undo and redo -- so a cached index never outlives the geometry it
describes. Inside an open operation or transaction the revision has not been
increased yet, so queries there get a fresh index. Building an index reads the
scene and changes nothing in it (procedural evaluations land in the procedural
system's own cache, which is not scene state).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, QueryError
from ..rendering.raytrace.accel import BLAS_CACHE, SceneAccel
from ..rendering.raytrace.bvh import build_bvh
from ..rendering.scene_data import RenderMesh, world_matrices

if TYPE_CHECKING:
    from ..mesh import Mesh
    from ..scene import Scene, SceneObject

MAX_ACCELS = 8
"""Ray acceleration structures kept per index (one per distinct set of objects)."""


@dataclass(eq=False)
class Entry:
    """One object with geometry, in world space."""

    order: int  # position in the scene's depth-first order (stable sorting key)
    obj: SceneObject
    mesh: Mesh
    world: np.ndarray  # (4, 4)
    visible: bool
    det: float  # determinant of the world matrix's linear part
    _vertices: np.ndarray | None = field(default=None, repr=False)
    _bvh: Any = field(default=None, repr=False)
    _box: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    _nodes: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    _used: np.ndarray | None = field(default=None, repr=False)
    _tri_box: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    @property
    def linear(self) -> np.ndarray:
        return self.world[:3, :3]

    @property
    def singular(self) -> bool:
        return not np.isfinite(self.det) or abs(self.det) < 1e-300

    @property
    def orientation(self) -> float:
        """+1, or -1 for a mirrored world transform (it reverses the triangles' winding)."""
        return -1.0 if self.det < 0 else 1.0

    @property
    def vertices(self) -> np.ndarray:
        """World-space vertices ``(n, 3)`` float64."""
        if self._vertices is None:
            local = self.mesh.positions.astype(np.float64)
            self._vertices = local @ self.linear.T + self.world[:3, 3]
        return self._vertices

    def corners(self, triangles: np.ndarray | slice | None = None):
        """World-space corners ``a``, ``b``, ``c`` of the mesh's triangles (or some)."""
        tri = self.mesh.triangles if triangles is None else self.mesh.triangles[triangles]
        v = self.vertices
        return v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]

    @property
    def used(self) -> np.ndarray:
        """Indices of the vertices the triangles use (the geometry's points)."""
        if self._used is None:
            self._used = np.unique(self.mesh.triangles)
        return self._used

    @property
    def points(self) -> np.ndarray:
        """World positions of the vertices the triangles use."""
        return self.vertices[self.used]

    @property
    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """The world-space axis-aligned bounds of the geometry (exact: min/max of its points)."""
        if self._box is None:
            pts = self.points
            self._box = (pts.min(axis=0), pts.max(axis=0))
        return self._box

    @property
    def normal_matrix(self) -> np.ndarray:
        """Inverse transpose of the linear part (carries normals; outward stays outward)."""
        return np.linalg.inv(self.linear).T

    @property
    def node_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds of every node of :attr:`bvh` (``(nodes, 3)`` each)."""
        if self._nodes is None:
            bvh = self.bvh
            lo = np.empty((bvh.node_count, 3))
            hi = np.empty((bvh.node_count, 3))
            lo[0], hi[0] = bvh.lo, bvh.hi
            inner = np.flatnonzero(bvh.child[:, 0] >= 0)
            for j in (0, 1):
                lo[bvh.child[inner, j]] = bvh.child_lo[inner, j]
                hi[bvh.child[inner, j]] = bvh.child_hi[inner, j]
            self._nodes = (lo, hi)
        return self._nodes

    @property
    def bvh(self):
        """A world-space BVH over the triangles (built on first use)."""
        if self._bvh is None:
            lo, hi = self.triangle_boxes
            finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
            if not finite.all():
                keep = np.flatnonzero(finite)
                bvh = build_bvh(lo[keep], hi[keep])
                object.__setattr__(bvh, "order", keep[bvh.order])
            else:
                bvh = build_bvh(lo, hi)
            self._bvh = bvh
        return self._bvh

    @property
    def triangle_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space bounds of every triangle (``(n, 3)`` each)."""
        if self._tri_box is None:
            a, b, c = self.corners()
            self._tri_box = (np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c))
        return self._tri_box


class _Instance:
    __slots__ = ("mesh", "world")

    def __init__(self, mesh: int, world: np.ndarray) -> None:
        self.mesh = mesh
        self.world = world


class _Instances:
    """The two fields of a RenderScene that SceneAccel.build reads."""

    def __init__(self, meshes: list[RenderMesh], instances: list[_Instance]) -> None:
        self.meshes = meshes
        self.instances = instances


class SpatialIndex:
    """Every object with geometry in a scene, in world space (see the module docstring)."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.revision = scene.revision
        worlds = world_matrices(scene)
        self.worlds = worlds
        self.order: dict[str, int] = {}
        self.entries: list[Entry] = []
        self.by_id: dict[str, Entry] = {}
        self._topology: dict[int, dict[str, Any]] = {}
        self._accels: dict[tuple[int, ...], tuple[SceneAccel, list[Entry]]] = {}
        self._rendered: dict[tuple[int, ...], Any] = {}  # PathScenes (rays.rendered_scene)
        self._render_scene: Any = None
        self._users: dict[int, list[Entry]] | None = None
        self._representatives: dict[int, np.ndarray] = {}
        # Objects whose procedural geometry fails to evaluate: an error for the queries
        # they take part in (as renders fail only for the objects they draw).
        self.failed: dict[str, QueryError] = {}
        for k, obj in enumerate(scene):
            self.order[obj.id] = k
            if obj._mesh is None and obj._geometry is None:
                continue
            try:
                mesh = obj.mesh  # procedural objects: the current evaluation (cached)
            except AetherError as exc:
                code = getattr(exc, "code", type(exc).__name__)
                self.failed[obj.id] = QueryError(
                    "GEOMETRY_EVALUATION_FAILED",
                    f"the geometry of {obj.name!r} could not be evaluated: [{code}] "
                    + getattr(exc, "message", str(exc)),
                    f"objects[{obj.name}]",
                    hint="fix the procedural definition, hide the object, or leave it out "
                    "(exclude=...)",
                )
                continue
            if mesh is None or mesh.triangle_count == 0:
                continue
            world = worlds[obj.id]
            if not np.all(np.isfinite(world)):
                continue
            det = float(np.linalg.det(world[:3, :3]))
            entry = Entry(k, obj, mesh, world, obj.visible, det)
            self.entries.append(entry)
            self.by_id[obj.id] = entry

    # -- selection --------------------------------------------------------------

    def select(
        self,
        objects: Any = None,
        exclude: Any = None,
        include_hidden: bool = False,
    ) -> list[Entry]:
        """The entries of ``objects`` (default: all), minus ``exclude``, visible only unless
        ``include_hidden``; in scene order."""
        ids = (
            None if objects is None else {self.resolve(o, "objects").id for o in _as_list(objects)}
        )
        drop = (
            set() if exclude is None else {self.resolve(o, "exclude").id for o in _as_list(exclude)}
        )
        for oid, error in self.failed.items():
            obj = self.scene._objects[oid]
            if (ids is None or oid in ids) and oid not in drop and (include_hidden or obj.visible):
                raise error
        chosen = self.entries if ids is None else [e for e in self.entries if e.obj.id in ids]
        if drop:
            chosen = [e for e in chosen if e.obj.id not in drop]
        if not include_hidden:
            chosen = [e for e in chosen if e.visible]
        return chosen

    def resolve(self, ref: Any, location: str = "object") -> SceneObject:
        """A scene object by object, id or name (``QueryError`` if unknown)."""
        from ..scene import SceneObject

        scene = self.scene
        if isinstance(ref, SceneObject):
            if ref.scene is not scene:
                raise QueryError(
                    "OBJECT_NOT_FOUND", f"{ref!r} does not belong to this scene", location
                )
            return ref
        if isinstance(ref, str):
            if ref in scene._objects:
                return scene._objects[ref]
            if ref in scene._by_name:
                return scene._by_name[ref]
        raise QueryError(
            "OBJECT_NOT_FOUND",
            f"no object with id or name {ref!r} in this scene",
            location,
        )

    def entry(self, ref: Any, location: str = "object") -> Entry:
        """The entry of an object with geometry (``QueryError("NO_GEOMETRY")`` otherwise)."""
        obj = self.resolve(ref, location)
        if obj.id in self.failed:
            raise self.failed[obj.id]
        found = self.by_id.get(obj.id)
        if found is None:
            raise QueryError(
                "NO_GEOMETRY",
                f"object {obj.name!r} has no geometry ({obj.kind})",
                location,
                hint="spatial queries on surfaces need mesh or procedural objects",
            )
        return found

    # -- shared data --------------------------------------------------------------

    def mesh_users(self, entry: Entry) -> list[Entry]:
        """The entries whose geometry is the same mesh as ``entry``'s (instances), in order."""
        if self._users is None:
            users: dict[int, list[Entry]] = {}
            for e in self.entries:
                users.setdefault(id(e.mesh), []).append(e)
            self._users = users
        return self._users[id(entry.mesh)]

    def representatives(self, entry: Entry) -> np.ndarray:
        """One vertex index of every connected component of the entry's mesh."""
        from .geometry import component_representatives

        key = id(entry.mesh)
        found = self._representatives.get(key)
        if found is None:
            found = component_representatives(entry.mesh.positions, entry.mesh.triangles)
            self._representatives[key] = found
        return found

    def topology(self, entry: Entry) -> dict[str, Any]:
        """``closed``, ``edge_manifold``, ``consistently_oriented`` of an entry's mesh."""
        key = id(entry.mesh)
        found = self._topology.get(key)
        if found is None:
            from ..inspection import inspect_mesh

            topo = inspect_mesh(entry.mesh).topology
            found = {
                "closed": bool(topo.get("closed")),
                "edge_manifold": bool(topo.get("edge_manifold")),
                "consistently_oriented": bool(topo.get("consistently_oriented")),
            }
            found["solid"] = (
                found["closed"] and found["edge_manifold"] and found["consistently_oriented"]
            )
            self._topology[key] = found
        return found

    def accel(self, entries: list[Entry]) -> tuple[SceneAccel, list[Entry]]:
        """The ray acceleration structure of these entries (cached)."""
        usable = [e for e in entries if not e.singular]
        key = tuple(e.order for e in usable)
        found = self._accels.get(key)
        if found is None:
            meshes: list[RenderMesh] = []
            index: dict[int, int] = {}
            instances: list[_Instance] = []
            for e in usable:
                k = index.get(id(e.mesh))
                if k is None:
                    k = index[id(e.mesh)] = len(meshes)
                    m = e.mesh
                    meshes.append(RenderMesh(m.positions, m.triangles, m.normals, m.uvs, m.groups))
                instances.append(_Instance(k, e.world))
            accel = SceneAccel.build(_Instances(meshes, instances), BLAS_CACHE)
            if len(self._accels) >= MAX_ACCELS:
                self._accels.pop(next(iter(self._accels)))
            found = self._accels[key] = (accel, usable)
        return found


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        return [value]
    return list(value)


_CACHE: weakref.WeakKeyDictionary[Scene, SpatialIndex] = weakref.WeakKeyDictionary()
"""The last index of every scene (a cache, not scene state: it dies with the scene)."""


def get_index(scene: Scene) -> SpatialIndex:
    """The scene's spatial index, reused while ``scene.revision`` is unchanged."""
    if scene.history.in_progress:  # inside an operation: its changes are not counted yet
        return SpatialIndex(scene)
    cached = _CACHE.get(scene)
    if cached is not None and cached.revision == scene.revision:
        return cached
    index = SpatialIndex(scene)
    _CACHE[scene] = index
    return index
