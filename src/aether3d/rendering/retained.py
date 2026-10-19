"""The retained render world: extraction kept up to date between renders.

The full extraction (:func:`~aether3d.rendering.scene_data._extract_full`) walks every
object of a scene for every render: world matrices, normal matrices, evaluated meshes,
materials and one render instance per object, in Python. A :class:`RenderWorld`, kept
among the scene's derived caches (``Scene._derived``), holds what that walk computes --
arrays over the objects in depth-first order (their *rows*) and the render instances
themselves -- and brings it up to date from the history's change journal
(:meth:`~aether3d.history.History._changes_since`):

* a transform recomputes its object's local matrix and the world matrices of its subtree,
  one hierarchy level at a time with the full extraction's arithmetic;
* a change of an object's visibility, appearance, materials, geometry, name, camera or
  light classifies that object again;
* a change of the hierarchy (objects created, removed or reparented) orders the objects
  again; per object, what was computed before is kept unless that object changed (world
  matrices are compared bit for bit);
* a procedural or geometry change evaluates the procedural objects again; library
  materials are resolved again for every render through the material cache, which reads
  their images again as before (files changed on disk are picked up).

Nothing else is visited: an unchanged frame does no per-object work, and moving one object
rebuilds one render instance. The render scene is exactly the one the full extraction
returns -- the same meshes, instances, arrays, materials, lights and diagnostics, in the
same order (the mesh and material tables keep their order of first use) --
``tests/test_retained_world.py`` compares them along random edits. The full extraction
runs instead whenever this one cannot run or could differ: inside an open operation or
transaction (its changes are not journaled yet), with the material cache switched off,
while another thread updates the world, and when the scene fails to render (a non-finite
transform, a failing procedural evaluation, a bad camera: the full extraction reports the
error).

The arrays a render scene shares with the world (each instance's matrices and color, the
instance arrays) are read-only: render scenes are immutable, and these are reused by later
frames.

Switch: ``AETHER3D_RETAINED_RENDER_WORLD`` (``docs/PERFORMANCE.md``).
"""

from __future__ import annotations

import operator
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .. import history as _h
from ..errors import AetherError
from .color import to_linear
from .materials import (
    MaterialCache,
    MaterialTable,
    RenderMaterial,
    _appearance_key,
    _material,
    _missing_uvs,
    _prepare,
    appearance_material,
)
from .scene_data import (
    InstanceArrays,
    RenderCamera,
    RenderInstance,
    RenderLight,
    RenderMesh,
    RenderScene,
    _evaluated_mesh,
    _finish,
    _light,
    _local_matrices,
    _mat4_mul_many,
    _material_cache,
    _render_camera,
    _render_mesh,
    _with_fields,
    _world_from_local,
    _zero_scale,
    normal_matrices,
)

if TYPE_CHECKING:
    from ..diagnostics import Diagnostic
    from ..mesh import Mesh
    from ..scene import Scene, SceneObject

STRICT = False
"""Testing aid: let unexpected errors of the retained world propagate, instead of falling
back to the full extraction (which would hide them)."""

_DERIVED = "render_world"

# What each journaled aspect affects.
_STRUCTURE = frozenset((_h.ALIVE, _h.PARENT, _h.CHILDREN, _h.ROOTS))
_CONTENT = frozenset(
    (_h.VISIBLE, _h.APPEARANCE, _h.MATERIALS, _h.DATA, _h.NAME, _h.CAMERA, _h.LIGHT)
)
_LIBRARY = frozenset((_h.LIBRARY_MATERIALS, _h.LIBRARY_TEXTURES, _h.MATERIAL, _h.TEXTURE))
# CAD definitions reach renders only through the meshes and materials of their linked
# objects (journaled as DATA and MATERIALS), never directly.
_UNUSED = frozenset(
    (
        _h.ANNOTATIONS,
        _h.SCENE_NAME,
        _h.ACTIVE_CAMERA,
        _h.WORLD,
        _h.RENDER,
        _h.CAD_ITEM,
        _h.CAD_SETTINGS,
        # Animation reaches renders through the evaluated pose of each render's time.
        _h.ANIMATION_ITEM,
        _h.ANIMATION_SETTINGS,
    )
)

# Row kinds.
_NONE, _LIGHT, _MESH = 0, 1, 2


class _Fallback(Exception):
    """The full extraction must run (to report the scene's error, or because only it can)."""


def retained_render_scene(
    scene: Scene,
    camera: SceneObject | RenderCamera,
    *,
    default_lights: bool,
    aspect: float,
    overlay: Any = None,
) -> RenderScene | None:
    """The render scene, from the scene's retained world -- or ``None`` when the full
    extraction must run instead (see the module docstring). An ``overlay`` (an animated
    pose) is applied on top of the authored rows: posed rows get their local matrices
    and subtrees' world matrices again, rows with animated content are classified again
    -- the render scene the full extraction returns for that overlay. Animated material
    parameters go through the full extraction."""
    history = scene.history
    if history.in_progress or history._building:
        return None
    if overlay is not None and overlay.materials:
        return None
    cache = _material_cache(scene)
    if cache is None:  # AETHER3D_MATERIAL_CACHE=0: every render resolves every material
        return None
    derived = scene._derived()
    world = derived.get(_DERIVED)
    if world is None:
        world = derived[_DERIVED] = RenderWorld()
    if not world.lock.acquire(blocking=False):
        return None  # another thread is updating it
    try:
        start = time.perf_counter()
        try:
            world.sync(scene, cache)
            world.apply_overlay(overlay, cache)
        except AetherError:  # a failing evaluation, ...: the full extraction reports it
            derived.pop(_DERIVED, None)  # possibly half updated: start again next time
            return None
        except Exception:
            derived.pop(_DERIVED, None)
            if STRICT:
                raise
            return None
        try:
            return world.render_scene(scene, camera, cache, default_lights, aspect, start, overlay)
        except (_Fallback, AetherError):  # nothing changed in the world: it is kept
            return None
        except Exception:
            derived.pop(_DERIVED, None)
            if STRICT:
                raise
            return None
    finally:
        world.lock.release()


def _depth_first(scene: Scene) -> tuple[list[SceneObject], list[int], list[int]]:
    """The scene's objects in ``iter(scene)``'s order -- each root, then its descendants
    depth-first in order -- with each one's parent row (-1 for roots) and depth."""
    roots = scene._roots
    parents_at = [i for i, root in enumerate(roots) if root._children]
    if not parents_at:  # a flat scene
        return list(roots), [-1] * len(roots), [0] * len(roots)
    objects: list[SceneObject] = []
    parents: list[int] = []
    depths: list[int] = []
    start = 0
    for i in [*parents_at, len(roots)]:
        run = roots[start:i]  # roots without children
        objects += run
        parents += [-1] * len(run)
        depths += [0] * len(run)
        if i == len(roots):
            break
        stack: list[tuple[SceneObject, int, int]] = [(roots[i], -1, 0)]
        while stack:
            obj, parent, depth = stack.pop()
            k = len(objects)
            objects.append(obj)
            parents.append(parent)
            depths.append(depth)
            for child in reversed(obj._children):
                stack.append((child, k, depth + 1))
        start = i + 1
    return objects, parents, depths


def _old_rows(ids: np.ndarray, old_ids: np.ndarray) -> np.ndarray:
    """For each id, its row among ``old_ids`` (-1 if absent)."""
    out = np.full(len(ids), -1, dtype=np.int64)
    if not len(old_ids) or not len(ids):
        return out
    order = np.argsort(old_ids, kind="stable")
    sorted_ids = old_ids[order]
    at = np.minimum(np.searchsorted(sorted_ids, ids), len(sorted_ids) - 1)
    found = sorted_ids[at] == ids
    out[found] = order[at[found]]
    return out


def _carry_items(items: list, kept: np.ndarray, was: np.ndarray, n: int) -> list:
    """A list of ``n`` items: ``items[was[i]]`` at ``kept[i]``, ``None`` elsewhere."""
    m = len(was)
    if m == 0:
        return [None] * n
    picked = operator.itemgetter(*was.tolist())(items) if m > 1 else (items[int(was[0])],)
    if int(kept[-1]) == m - 1:  # kept rows come first (objects added after them)
        return [*picked, *([None] * (n - m))]
    out = np.empty(n, dtype=object)
    out[kept] = np.fromiter(picked, dtype=object, count=m)
    return out.tolist()


FRAME_CACHE_BYTES = 64 << 20
"""Framing points (float64 world-space vertices) a world keeps for later framed renders;
scenes with more vertices compute them for every framed render, as before."""


def _existing_world(scene: Scene) -> tuple[RenderWorld, MaterialCache] | None:
    """The scene's retained world, if it has one (it was rendered) and it can be used now."""
    history = scene.history
    if history.in_progress or history._building:
        return None
    world = scene._derived().get(_DERIVED)
    cache = _material_cache(scene)
    if world is None or cache is None:
        return None
    return world, cache


def retained_world_matrices(scene: Scene) -> dict[str, np.ndarray] | None:
    """``world_matrices(scene)`` from the retained world, when the scene has one (and no
    animated pose is applied to it: authored queries get authored matrices)."""
    found = _existing_world(scene)
    if found is None:
        return None
    world, cache = found
    return _with_world(scene, world, cache, world.world_matrices)


def retained_target_points(scene: Scene) -> np.ndarray | None:
    """``target_points(scene)`` from the retained world, when the scene has one."""
    found = _existing_world(scene)
    if found is None:
        return None
    world, cache = found
    return _with_world(scene, world, cache, world.target_points)


def _with_world(scene: Scene, world: RenderWorld, cache: MaterialCache, use: Any) -> Any:
    """``use()`` with ``world`` brought up to date, or ``None`` if it cannot be."""
    if not world.lock.acquire(blocking=False):
        return None
    try:
        try:
            world.sync(scene, cache)
        except AetherError:
            scene._derived().pop(_DERIVED, None)
            return None
        except Exception:
            scene._derived().pop(_DERIVED, None)
            if STRICT:
                raise
            return None
        if world.overlay is not None:  # it shows an animated pose: compute authored ones
            return None
        return use()
    finally:
        world.lock.release()


def _first_use(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a sequence of keys: where each distinct key first occurs, in order of first use,
    and the rank of each element's key in that order."""
    if keys.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(unique), dtype=np.int64)
    rank[order] = np.arange(len(unique), dtype=np.int64)
    return first[order], rank[inverse.reshape(-1)]


def _frozen(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class _Tables:
    """One frame's ordering of meshes and materials, its instances and arrays."""

    __slots__ = (
        "valid_rows",
        "meshes",
        "mesh_index",
        "keys",
        "positions",
        "default_index",
        "group_index",
        "library",
        "materials",
        "uv_slots",
        "instances",
        "arrays",
        "materials_tuple",
        "shared",
    )

    valid_rows: np.ndarray  # rows of the objects with an instance, in order
    meshes: tuple[RenderMesh, ...]
    mesh_index: np.ndarray  # per valid row
    keys: list[int]  # material keys, in order of first use
    positions: list[int]  # the row of each material's first use
    default_index: np.ndarray  # per valid row
    group_index: dict[int, tuple[tuple[int, int], ...]]  # row -> sorted (group, index)
    library: list[tuple[int, str]]  # (table index, material id) of library materials
    materials: list[RenderMaterial]
    uv_slots: list[bool]  # per material: some texture maps by UV
    instances: tuple[RenderInstance, ...] | None
    arrays: InstanceArrays | None
    materials_tuple: tuple[RenderMaterial, ...] | None  # (None when the list changed)
    shared: dict[str, Any] | None  # the render scenes' shared cache (RenderScene.cache)


class RenderWorld:
    """What extraction computes for one scene, kept up to date (see the module docstring)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.revision = -1
        self.objects: list[SceneObject] = []
        self.rows: dict[int, int] = {}  # id(object) -> row
        self.ids = np.zeros(0, dtype=np.int64)  # id(object) of each row
        self.parent = np.zeros(0, dtype=np.int64)
        self.depth = np.zeros(0, dtype=np.int64)
        self.local = np.zeros((0, 4, 4))
        self.world = np.zeros((0, 4, 4))
        self.finite = np.zeros(0, dtype=bool)
        self.kind = np.zeros(0, dtype=np.int8)
        self.procedural = np.zeros(0, dtype=bool)
        # Mesh rows (kind _MESH): the evaluated mesh, appearance, materials, normal matrix.
        self.mesh: list[Mesh | None] = []
        # Framing (target_points): the meshes with vertices of visible objects, and their
        # world-space vertices once computed (within FRAME_CACHE_BYTES).
        self.frame_mesh: list[Mesh | None] = []
        self.frame_points: list[np.ndarray | None] = []
        self.frame_all: np.ndarray | None = None
        self.world_items: list[tuple[str, np.ndarray]] | None = None  # world_matrices
        self.mesh_id = np.zeros(0, dtype=np.int64)  # id() of the mesh (it is held above)
        self.no_uvs = np.zeros(0, dtype=bool)
        self.look: list[tuple[Any, Any] | None] = []  # (specular, shininess) as given
        self.base_color = np.zeros((0, 3))
        self.specular = np.zeros(0)
        self.shininess = np.zeros(0)
        self.mat_key = np.zeros(0, dtype=np.int64)
        self.group_keys: list[tuple[tuple[int, int], ...]] = []  # (group, key), in order
        self.ngroups = np.zeros(0, dtype=np.int64)
        self.normal = np.zeros((0, 3, 3))
        self.det = np.zeros(0)
        self.valid = np.zeros(0, dtype=bool)
        # The render instance of each mesh row, and the indices it was built with.
        self.instance: list[RenderInstance | None] = []
        self.inst_mesh = np.zeros(0, dtype=np.int64)
        self.inst_material = np.zeros(0, dtype=np.int64)
        self.dirty = np.zeros(0, dtype=bool)  # the instance must be built again
        # Instances' matrices and colors live in read-only blocks: block -> [rows, in use].
        self.inst_block = np.zeros(0, dtype=np.int64)  # each row's instance's block, or -1
        self.blocks: dict[int, list[int]] = {}
        self.block_rows = 0
        self.next_block = 0
        # Material keys: ("library", material id) or ("color", appearance bytes) -> int.
        self.keys: dict[tuple[str, Any], int] = {}
        self.key_info: list[tuple[str, Any]] = []  # ("library", id) | ("color", material)
        self.tables: _Tables | None = None
        self.reorder = True  # the mesh and material orders must be computed again
        self.render_meshes: dict[int, RenderMesh] = {}
        self.evaluate_time = 0.0
        # (previous instances tuple, indices of the instances built again) of this frame
        self.lineage: tuple[tuple[RenderInstance, ...], np.ndarray] | None = None
        # An applied overlay (animated pose): authored local matrices are kept apart from
        # the effective ones; posed rows and rows classified with animated content.
        self.authored = np.zeros((0, 4, 4))
        self.overlay: Any = None
        self.overlay_rows = np.zeros(0, dtype=np.int64)
        self.content_rows: set[int] = set()
        self.by_oid: dict[str, int] | None = None

    # ------------------------------------------------------------------ #
    # Bringing the world up to date
    # ------------------------------------------------------------------ #

    def sync(self, scene: Scene, cache: MaterialCache) -> None:
        """Apply the changes journaled since the world was last brought up to date."""
        self.evaluate_time = 0.0
        history = scene.history
        changes = history._changes_since(self.revision)  # (the first call starts it)
        if changes != [] and self.overlay is not None:
            self._drop_overlay(cache)  # authored changes apply to the authored rows
        if changes is None:  # unknown changes: nothing computed before can be kept
            self._rebuild(scene, cache, {}, {}, reevaluate=False, keep=False)
        elif changes:
            self._apply(scene, cache, changes)
        self.revision = history.revision

    def _apply(self, scene: Scene, cache: MaterialCache, changes: list) -> None:
        moved: dict[int, SceneObject] = {}
        content: dict[int, SceneObject] = {}
        structure = reevaluate = False
        for aspect, target in changes:
            if aspect is _h.TRANSFORM:
                moved[id(target)] = target
            elif aspect in _CONTENT:
                content[id(target)] = target
                if aspect is _h.DATA:
                    reevaluate = True  # a procedural object may read other geometry
            elif aspect in _STRUCTURE:
                structure = True
            elif aspect is _h.PROCEDURAL:
                reevaluate = True
            elif aspect in _LIBRARY or aspect in _UNUSED:
                continue  # library materials are resolved again for every render
            else:  # an aspect this module does not know: recompute everything
                self._rebuild(scene, cache, {}, {}, reevaluate=False, keep=False)
                return
        rows = self.rows
        if structure or any(k not in rows for k in moved) or any(k not in rows for k in content):
            self._rebuild(scene, cache, moved, content, reevaluate)
            return
        if moved:
            self._move(sorted(rows[k] for k in moved))
        dirty = {rows[k] for k in content}
        if reevaluate:
            dirty.update(np.flatnonzero(self.procedural).tolist())
        if dirty:
            self._classify(sorted(dirty), cache)

    def _rebuild(
        self,
        scene: Scene,
        cache: MaterialCache,
        moved: dict[int, SceneObject],
        content: dict[int, SceneObject],
        reevaluate: bool,
        keep: bool = True,
    ) -> None:
        """Order the objects again. With ``keep``, what was computed for an object is kept
        unless the journal says it changed (``moved``, ``content``, ``reevaluate``);
        without, everything is computed again."""
        objects, parents, depths = _depth_first(scene)  # the order of iter(scene)
        n = len(objects)
        ids = np.fromiter(map(id, objects), dtype=np.int64, count=n)
        rows = dict(zip(ids.tolist(), range(n), strict=True))
        parent = np.array(parents, dtype=np.int64).reshape(n)
        depth = np.array(depths, dtype=np.int64).reshape(n)
        old = _old_rows(ids, self.ids if keep else np.zeros(0, dtype=np.int64))
        kept = np.flatnonzero(old >= 0)
        was = old[kept]
        fresh = old < 0
        if moved:
            fresh_local = fresh | np.fromiter((id(o) in moved for o in objects), bool, count=n)
        else:
            fresh_local = fresh

        local = np.empty((n, 4, 4), dtype=np.float64)
        local[kept] = self.local[was]
        recompute = np.flatnonzero(fresh_local)
        if recompute.size:
            local[recompute] = _local_matrices([objects[k] for k in recompute.tolist()])
        world = _world_from_local(local.copy(), parent, depth)
        changed = fresh_local.copy()
        if kept.size:
            changed[kept] |= np.any(
                world[kept].view(np.uint64) != self.world[was].view(np.uint64), axis=(1, 2)
            )

        def carry(array: np.ndarray, fill: Any) -> np.ndarray:
            out = np.full((n, *array.shape[1:]), fill, dtype=array.dtype)
            out[kept] = array[was]
            return out

        def carry_list(items: list) -> list:
            return _carry_items(items, kept, was, n)

        self.kind = carry(self.kind, _NONE)
        self.procedural = carry(self.procedural, False)
        self.mesh = carry_list(self.mesh)
        self.frame_mesh = carry_list(self.frame_mesh)
        self.frame_points = carry_list(self.frame_points)
        for k in np.flatnonzero(changed).tolist():
            self.frame_points[k] = None
        self.frame_all = None
        self.world_items = None
        self.mesh_id = carry(self.mesh_id, 0)
        self.no_uvs = carry(self.no_uvs, False)
        self.look = carry_list(self.look)
        self.base_color = carry(self.base_color, 0.0)
        self.specular = carry(self.specular, 0.0)
        self.shininess = carry(self.shininess, 0.0)
        self.mat_key = carry(self.mat_key, -1)
        self.group_keys = [g if g is not None else () for g in carry_list(self.group_keys)]
        self.ngroups = carry(self.ngroups, 0)
        self.normal = carry(self.normal, 0.0)
        self.det = carry(self.det, 0.0)
        self.valid = carry(self.valid, False)
        for r in np.setdiff1d(np.flatnonzero(self.inst_block >= 0), was).tolist():
            self._release(r)  # (objects removed, or nothing kept)
        self.instance = carry_list(self.instance)
        self.inst_block = carry(self.inst_block, -1)
        self.inst_mesh = carry(self.inst_mesh, -1)
        self.inst_material = carry(self.inst_material, -1)
        self.dirty = carry(self.dirty, True)
        self.objects, self.rows, self.ids = objects, rows, ids
        self.parent, self.depth = parent, depth
        self.local, self.world = local, world
        self.authored = local.copy()
        self.overlay = None
        self.overlay_rows = np.zeros(0, dtype=np.int64)
        self.content_rows = set()
        self.by_oid = None
        self.finite = np.all(np.isfinite(world), axis=(1, 2)).reshape(n)
        self.reorder = True

        classify = fresh.copy()
        if content:
            classify |= np.fromiter((id(o) in content for o in objects), bool, count=n)
        if reevaluate:
            classify |= self.procedural
        meshes = np.flatnonzero(changed & ~classify & (self.kind == _MESH))
        self._normals(meshes)
        self._classify(np.flatnonzero(classify).tolist(), cache)

    def _move(self, rows: list[int]) -> None:
        """New local matrices for ``rows``; new world matrices for their subtrees."""
        objects = self.objects
        self.local[rows] = _local_matrices([objects[r] for r in rows])
        self.authored[rows] = self.local[rows]
        self._subtrees(rows)

    def _subtrees(self, rows: list[int]) -> None:
        """New world matrices for the subtrees of ``rows`` (their local matrices set)."""
        objects = self.objects
        index = self.rows
        affected = set(rows)
        for r in rows:
            obj = objects[r]
            if obj._children:
                affected.update(index[id(d)] for d in obj.iter_descendants())
        at = np.array(sorted(affected), dtype=np.int64)
        levels = self.depth[at]
        for level in np.unique(levels).tolist():
            these = at[levels == level]
            parents = self.parent[these]
            roots = these[parents < 0]
            self.world[roots] = self.local[roots]
            inner = these[parents >= 0]
            if inner.size:
                self.world[inner] = _mat4_mul_many(
                    self.world[self.parent[inner]], self.local[inner]
                )
        self.finite[at] = np.all(np.isfinite(self.world[at]), axis=(1, 2))
        for r in at.tolist():
            self.frame_points[r] = None
        self.frame_all = None
        self.world_items = None
        self._normals(at[self.kind[at] == _MESH])

    def _normals(self, rows: np.ndarray) -> None:
        """Normal matrices and determinants of mesh ``rows`` (their instances are rebuilt)."""
        if rows.size == 0:
            return
        normal, valid, det = normal_matrices(np.ascontiguousarray(self.world[rows][:, :3, :3]))
        self.normal[rows] = normal
        self.valid[rows] = valid
        self.det[rows] = det
        self.dirty[rows] = True

    def _classify(self, rows: list[int], cache: MaterialCache) -> None:
        """What each of ``rows`` renders as (as the full extraction decides it; with the
        applied overlay's visibility, appearance and meshes)."""
        if not rows:
            return
        self.reorder = True
        objects = self.objects
        overlay = self.overlay
        meshes = []
        for r in rows:
            obj = objects[r]
            kind = _NONE
            mesh = None
            self.procedural[r] = obj._geometry is not None
            visible = obj.visible if overlay is None else overlay.is_visible(obj)
            if obj.light is not None:
                kind = _LIGHT
            elif (
                obj.camera is None
                and visible
                and (obj._mesh is not None or obj._geometry is not None)
            ):
                t0 = time.perf_counter()
                mesh = _evaluated_mesh(obj, overlay)
                self.evaluate_time += time.perf_counter() - t0
                if mesh is not None and mesh.triangle_count > 0:
                    kind = _MESH
            self.kind[r] = kind
            self.dirty[r] = True
            self.frame_mesh[r] = mesh if mesh is not None and mesh.vertex_count > 0 else None
            self.frame_points[r] = None
            self.frame_all = None
            if kind != _MESH:
                self.mesh[r] = None
                self.look[r] = None
                self.group_keys[r] = ()
                self.ngroups[r] = 0
                self.instance[r] = None
                self._release(r)
                continue
            assert mesh is not None
            meshes.append(r)
            self.mesh[r] = mesh
            self.mesh_id[r] = id(mesh)
            self.no_uvs[r] = mesh.uvs is None
            look = obj.appearance
            if overlay is not None:
                values = overlay.field_values(obj._id, "appearance")
                if values:
                    look = _with_fields(look, values)
            self.look[r] = (look.specular, look.shininess)
            self.base_color[r] = to_linear(look.color)
            self.specular[r] = look.specular
            self.shininess[r] = look.shininess
            if obj._material is not None:
                self.mat_key[r] = self._library_key(obj._material)
            else:
                akey = _appearance_key(look)
                key = self.keys.get(("color", akey))
                if key is None:
                    material = cache.appearances.get(akey)
                    if material is None:
                        name = "appearance:" + repr((look.color, look.specular, look.shininess))
                        material = cache.appearances[akey] = appearance_material(look, name)
                    key = self._new_key(("color", akey), ("color", material))
                self.mat_key[r] = key
            groups = tuple(
                (group, self._library_key(ref)) for group, ref in obj._group_materials.items()
            )
            self.group_keys[r] = groups
            self.ngroups[r] = len(groups)
        self._normals(np.array(meshes, dtype=np.int64))

    def _library_key(self, material_id: str) -> int:
        key = self.keys.get(("library", material_id))
        return key if key is not None else self._new_key(("library", material_id), None)

    def _new_key(self, key: tuple[str, Any], info: tuple[str, Any] | None) -> int:
        k = len(self.key_info)
        self.keys[key] = k
        self.key_info.append(info if info is not None else key)
        return k

    # ------------------------------------------------------------------ #
    # World matrices and framing (kept for the scene's later renders)
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Overlays (animated poses)
    # ------------------------------------------------------------------ #

    def apply_overlay(self, overlay: Any, cache: MaterialCache) -> None:
        """Make the effective rows those of ``overlay`` (``None``: the authored ones)."""
        if overlay is self.overlay:
            return
        if self.by_oid is None:
            self.by_oid = {o._id: r for r, o in enumerate(self.objects)}
        by_oid = self.by_oid
        if overlay is not None and overlay.ids:
            pairs = [(by_oid[oid], k) for k, oid in enumerate(overlay.ids) if oid in by_oid]
            new_rows = np.array([p[0] for p in pairs], dtype=np.int64)
            new_local = overlay.local[[p[1] for p in pairs]] if pairs else np.zeros((0, 4, 4))
        else:
            new_rows = np.zeros(0, dtype=np.int64)
            new_local = np.zeros((0, 4, 4))
        old_rows = self.overlay_rows
        touched = np.union1d(old_rows, new_rows)
        before = self.local[touched].copy()
        restore = np.setdiff1d(old_rows, new_rows)
        if restore.size:
            self.local[restore] = self.authored[restore]
        if new_rows.size:
            self.local[new_rows] = new_local
        changed = (
            touched[
                np.any(self.local[touched].view(np.uint64) != before.view(np.uint64), axis=(1, 2))
            ]
            if touched.size
            else touched
        )
        self.overlay_rows = new_rows
        # Content: rows with animated visibility, appearance, light fields or meshes.
        content: set[int] = set()
        if overlay is not None:
            for oid in overlay.visible:
                if oid in by_oid:
                    content.add(by_oid[oid])
            for oid, group in overlay.fields:
                if group == "appearance" and oid in by_oid:
                    content.add(by_oid[oid])
            for oid in overlay.meshes:
                if oid in by_oid:
                    content.add(by_oid[oid])
        reclassify = sorted(content | self.content_rows)
        self.content_rows = content
        self.overlay = overlay
        if changed.size:
            self._subtrees(changed.tolist())
        if reclassify:
            self._classify(reclassify, cache)

    def _drop_overlay(self, cache: MaterialCache) -> None:
        """Back to the authored rows (before applying journaled authored changes)."""
        self.apply_overlay(None, cache)

    def world_matrices(self) -> dict[str, np.ndarray]:
        """``scene_data.world_matrices``: read-only rows of a copy of the world matrices."""
        items = self.world_items
        if items is None:
            copy = _frozen(self.world.copy())
            items = self.world_items = list(zip([o._id for o in self.objects], copy, strict=True))
        return dict(items)

    def target_points(self) -> np.ndarray | None:
        """``framing.target_points(scene)``: the world-space vertices of every visible mesh, in
        scene order, with its arithmetic (``None``: the full computation runs, to report
        that there is nothing to frame or that a position is not finite)."""
        if self.frame_all is not None:
            return self.frame_all
        rows = [r for r, mesh in enumerate(self.frame_mesh) if mesh is not None]
        if not rows:
            return None
        keep = sum(self.frame_mesh[r].vertex_count for r in rows) * 24 <= FRAME_CACHE_BYTES  # type: ignore[union-attr]
        chunks = []
        for r in rows:
            p = self.frame_points[r]
            if p is None:
                m = self.world[r]
                mesh = self.frame_mesh[r]
                assert mesh is not None
                p = _frozen(mesh.positions.astype(np.float64) @ m[:3, :3].T + m[:3, 3])
                if keep:
                    self.frame_points[r] = p
            chunks.append(p)
        if not keep:
            for r in rows:
                self.frame_points[r] = None
        points = np.concatenate(chunks)
        if not np.all(np.isfinite(points)):
            return None
        points = _frozen(points)
        if keep:
            self.frame_all = points
        return points

    # ------------------------------------------------------------------ #
    # One frame
    # ------------------------------------------------------------------ #

    def render_scene(
        self,
        scene: Scene,
        camera: SceneObject | RenderCamera,
        cache: MaterialCache,
        default_lights: bool,
        aspect: float,
        start: float,
        overlay: Any = None,
    ) -> RenderScene:
        diagnostics: list[Diagnostic] = []
        if isinstance(camera, RenderCamera):
            render_camera = camera
        else:
            row = self.rows.get(id(camera))
            if row is None:
                raise _Fallback

            def world_of(oid: str) -> np.ndarray | None:
                obj = scene._objects.get(oid)
                r = None if obj is None else self.rows.get(id(obj))
                return None if r is None else self.world[r]

            render_camera, camera_diags = _render_camera(
                camera, self.world[row].copy(), aspect, overlay, world_of
            )
            diagnostics += camera_diags
        kind = self.kind
        mesh_rows = np.flatnonzero(kind == _MESH)
        if not np.all(self.finite[mesh_rows]):
            raise _Fallback  # the full extraction reports the first object in error
        ordered: list[tuple[int, Diagnostic]] = []
        lights: list[RenderLight] = []
        for r in np.flatnonzero(kind == _LIGHT).tolist():
            obj = self.objects[r]
            if not (obj.visible if overlay is None else overlay.is_visible(obj)):
                continue
            if not self.finite[r]:
                raise _Fallback
            light = _light(obj, self.world[r], True, r, ordered, overlay)
            if light is not None:
                lights.append(light)

        valid_rows = mesh_rows[self.valid[mesh_rows]]
        tables = self.tables
        if (
            tables is None
            or self.reorder
            or tables.valid_rows.shape != valid_rows.shape
            or not np.array_equal(tables.valid_rows, valid_rows)
        ):
            tables = self._order(valid_rows)
        t0 = time.perf_counter()
        library_materials = self._library_materials(scene, cache, tables, ordered)
        if library_materials is not tables.materials:  # a material changed: new list
            tables.materials = library_materials
            tables.materials_tuple = None
            tables.uv_slots = [_uv_slots(m) for m in library_materials]
        if any(tables.uv_slots):
            ordered += self._missing_uvs(tables)
        texture_time = time.perf_counter() - t0
        for r in mesh_rows[~self.valid[mesh_rows]].tolist():
            ordered.append((r, _zero_scale(self.objects[r])))
        self._instances(tables)
        self.tables = tables
        assert tables.instances is not None and tables.arrays is not None
        materials = tables.materials_tuple
        if materials is None:
            materials = tables.materials_tuple = tuple(tables.materials)
        shared = tables.shared
        if (
            shared is None
            or shared["instances"] is not tables.instances
            or (shared["materials"] is not materials)
        ):  # what backends kept is for other instances or materials
            previous = shared
            shared = tables.shared = {
                "instances": tables.instances,
                "meshes": tables.meshes,
                "materials": materials,
            }
            lineage = self.lineage
            if (
                previous is not None
                and lineage is not None
                and previous["instances"] is lineage[0]
                and previous["meshes"] is tables.meshes
                and previous["materials"] is materials
            ):  # backends may update what they kept for the previous frame
                previous.pop("previous", None)  # (one frame back at most)
                shared["previous"] = previous
                shared["changed"] = lineage[1]
        self.lineage = None
        return _finish(
            scene,
            meshes=tables.meshes,
            instances=tables.instances,
            lights=lights,
            camera=render_camera,
            diagnostics=diagnostics,
            ordered=ordered,
            materials=materials,
            arrays=tables.arrays,
            default_lights=default_lights,
            evaluate_time=self.evaluate_time,
            texture_time=texture_time,
            start=start,
            cache=shared,
            overlay=overlay,
        )

    def _order(self, valid_rows: np.ndarray) -> _Tables:
        """The mesh and material tables, in order of first use, and each row's indices."""
        self._prune_keys()
        self.reorder = False
        t = _Tables()
        t.valid_rows = valid_rows
        # Meshes: shared by every instance that uses them, in order of first use.
        firsts, t.mesh_index = _first_use(self.mesh_id[valid_rows])
        previous = self.render_meshes
        render_meshes: dict[int, RenderMesh] = {}
        meshes = []
        for r in valid_rows[firsts].tolist():
            mesh = self.mesh[r]
            assert mesh is not None
            rm = previous.get(id(mesh))
            if rm is None or not _same_arrays(rm, mesh):  # (an id can be reused)
                rm = _render_mesh(mesh)
            render_meshes[id(mesh)] = rm
            meshes.append(rm)
        self.render_meshes = render_meshes
        t.meshes = tuple(meshes)
        # Materials: each object's default, then those of its face groups in its order.
        counts = 1 + self.ngroups[valid_rows]
        starts = np.zeros(len(valid_rows), dtype=np.int64)
        np.cumsum(counts[:-1], out=starts[1:])
        sequence = np.empty(int(counts.sum()), dtype=np.int64)
        sequence[starts] = self.mat_key[valid_rows]
        grouped = np.flatnonzero(self.ngroups[valid_rows]).tolist()
        for j in grouped:
            groups = self.group_keys[int(valid_rows[j])]
            s = int(starts[j]) + 1
            sequence[s : s + len(groups)] = [key for _, key in groups]
        firsts, rank = _first_use(sequence)
        t.keys = sequence[firsts].tolist()
        t.positions = np.repeat(valid_rows, counts)[firsts].tolist()
        t.default_index = rank[starts]
        t.group_index = {}
        for j in grouped:
            r = int(valid_rows[j])
            s = int(starts[j]) + 1
            t.group_index[r] = tuple(
                sorted((group, int(rank[s + i])) for i, (group, _) in enumerate(self.group_keys[r]))
            )
        materials: list[Any] = []
        t.library = []
        for index, key in enumerate(t.keys):
            kind, payload = self.key_info[key]
            if kind == "library":
                t.library.append((index, payload))
                materials.append(None)  # resolved for every render
            else:
                materials.append(payload)
        t.materials = materials
        t.uv_slots = [m is not None and _uv_slots(m) for m in materials]
        t.instances = None
        t.arrays = None
        t.materials_tuple = None
        t.shared = None
        return t

    def _library_materials(
        self,
        scene: Scene,
        cache: MaterialCache,
        tables: _Tables,
        ordered: list[tuple[int, Diagnostic]],
    ) -> list[RenderMaterial]:
        """The table's materials with its library ones resolved for this render (as the
        full extraction does: in order of first use, reporting their diagnostics at their
        first use). The same list if none changed."""
        if not tables.library:
            return tables.materials
        library = scene.materials
        _prepare(cache, library)
        table = MaterialTable()
        this_render: dict[str, RenderMaterial] = {}
        materials = tables.materials
        out = materials
        for index, material_id in tables.library:
            m = _material(
                library.material(material_id),
                cache,
                this_render,
                table,
                tables.positions[index],
            )
            if m is not materials[index]:
                if out is materials:
                    out = list(materials)
                out[index] = m
        ordered += table.diagnostics
        return out

    def _missing_uvs(self, tables: _Tables) -> list[tuple[int, Diagnostic]]:
        """``MISSING_UVS`` of the objects without UVs whose materials map textures by UV."""
        rows = tables.valid_rows
        uv_slots = np.array(tables.uv_slots, dtype=bool)
        hit = self.no_uvs[rows] & uv_slots[tables.default_index]
        for j in np.flatnonzero(self.no_uvs[rows] & (self.ngroups[rows] > 0)).tolist():
            r = int(rows[j])
            hit[j] |= any(uv_slots[i] for _, i in tables.group_index[r])
        out: list[tuple[int, Diagnostic]] = []
        materials = tables.materials
        for j in np.flatnonzero(hit).tolist():
            r = int(rows[j])
            used = [materials[int(tables.default_index[j])]]
            if r in tables.group_index:  # the face groups' materials, in the object's order
                by_group = dict(tables.group_index[r])
                used += [materials[by_group[group]] for group, _ in self.group_keys[r]]
            out += _missing_uvs(self.objects[r], used, r)
        return out

    def _instances(self, tables: _Tables) -> None:
        """Build the instances that changed; the tuple and arrays if anything did."""
        rows = tables.valid_rows
        mesh_index = tables.mesh_index
        default_index = tables.default_index
        rebuild = (
            self.dirty[rows]
            | (self.inst_mesh[rows] != mesh_index)
            | (self.inst_material[rows] != default_index)
        )
        group_index = tables.group_index
        for j in np.flatnonzero(self.ngroups[rows]).tolist():
            r = int(rows[j])
            inst = self.instance[r]
            if inst is None or inst.group_materials != group_index[r]:
                rebuild[j] = True
        built = np.flatnonzero(rebuild)
        if built.size:
            self._build(rows, built, mesh_index, default_index, group_index)
            if self.block_rows > 2 * len(rows) + 1024:  # too many rows kept for few instances
                self._compact(rows)
                built = np.arange(len(rows))
                self._build(rows, built, mesh_index, default_index, group_index)
            self.inst_mesh[rows[built]] = mesh_index[built]
            self.inst_material[rows[built]] = default_index[built]
        self.dirty[rows] = False
        self.lineage = None
        before = tables.instances
        if before is not None and not built.size:
            return  # unchanged: the previous frame's tuple and arrays
        instances = self.instance
        if before is None or tables.arrays is None or built.size == len(rows):
            tables.instances = tuple([instances[r] for r in rows.tolist()])  # type: ignore[misc]
            tables.arrays = InstanceArrays(
                world=_frozen(self.world[rows]),
                normal_matrix=_frozen(self.normal[rows]),
                mirrored=_frozen(self.det[rows] < 0.0),
                mesh=_frozen(mesh_index.astype(np.int64)),
                base_color=_frozen(self.base_color[rows]),
                specular=_frozen(self.specular[rows]),
                shininess=_frozen(self.shininess[rows]),
                material=_frozen(default_index.astype(np.int64)),
                has_groups=_frozen(self.ngroups[rows] > 0),
            )
            if before is not None and len(before) == len(rows):
                self._note_lineage(before, tables.instances, built)
            return
        # The same rows as the previous frame: only the instances built again change (every
        # other row has the previous frame's values, or it would have been built again).
        changed = rows[built]
        items = list(before)
        for j in built.tolist():
            items[j] = instances[int(rows[j])]
        tables.instances = tuple(items)  # type: ignore[arg-type]
        old = tables.arrays

        def patch(array: np.ndarray, values: np.ndarray) -> np.ndarray:
            out = array.copy()
            out[built] = values
            return _frozen(out)

        tables.arrays = InstanceArrays(
            world=patch(old.world, self.world[changed]),
            normal_matrix=patch(old.normal_matrix, self.normal[changed]),
            mirrored=patch(old.mirrored, self.det[changed] < 0.0),
            mesh=patch(old.mesh, mesh_index[built]),
            base_color=patch(old.base_color, self.base_color[changed]),
            specular=patch(old.specular, self.specular[changed]),
            shininess=patch(old.shininess, self.shininess[changed]),
            material=patch(old.material, default_index[built]),
            has_groups=patch(old.has_groups, self.ngroups[changed] > 0),
        )
        self._note_lineage(before, tables.instances, built)

    def _note_lineage(
        self,
        before: tuple[RenderInstance, ...],
        after: tuple[RenderInstance, ...],
        built: np.ndarray,
    ) -> None:
        """Lineage for backends (``RenderScene.shared_cache``) when the instances built again
        changed only in their transform, color or highlight: the same mesh, materials, name
        and id."""
        if all(
            a.mesh == b.mesh
            and a.material == b.material
            and a.group_materials == b.group_materials
            and a.name == b.name
            and a.object_id == b.object_id
            for a, b in ((before[j], after[j]) for j in built.tolist())
        ):
            self.lineage = (before, _frozen(built.astype(np.int64)))

    def _build(
        self,
        rows: np.ndarray,
        built: np.ndarray,
        mesh_index: np.ndarray,
        default_index: np.ndarray,
        group_index: dict[int, tuple[tuple[int, int], ...]],
    ) -> None:
        """The render instances of ``rows[built]``. Their matrices and colors are views into
        one read-only block of these rows (as the full extraction's are into its stack); the
        world counts the rows of each block still in use (``_compact``)."""
        at = rows[built]
        rows_list = at.tolist()
        self._release_rows(at)
        world = _frozen(self.world[at])
        normal = _frozen(self.normal[at])
        base = _frozen(self.base_color[at])
        block = self.next_block
        self.next_block += 1
        self.blocks[block] = [len(at), len(at)]
        self.block_rows += len(at)
        self.inst_block[at] = block
        objects = [self.objects[r] for r in rows_list]
        looks = [self.look[r] for r in rows_list]
        groups = [group_index.get(r, ()) for r in rows_list] if group_index else [()] * len(at)
        built_instances = [
            RenderInstance(obj._id, obj._name, m, w, nm, mirrored, b, look[0], look[1], material, g)
            for obj, m, w, nm, mirrored, b, look, material, g in zip(
                objects,
                mesh_index[built].tolist(),
                world,
                normal,
                (self.det[at] < 0.0).tolist(),
                base,
                looks,
                default_index[built].tolist(),
                groups,
                strict=True,
            )
        ]
        instance = self.instance
        for r, inst in zip(rows_list, built_instances, strict=True):
            instance[r] = inst

    def _release_rows(self, at: np.ndarray) -> None:
        """``_release`` for many rows at once."""
        blocks = self.inst_block[at]
        used = blocks[blocks >= 0]
        if not used.size:
            return
        self.inst_block[at] = -1
        ids, counts = np.unique(used, return_counts=True)
        for block, count in zip(ids.tolist(), counts.tolist(), strict=True):
            entry = self.blocks[block]
            entry[1] -= count
            if entry[1] == 0:
                self.block_rows -= entry[0]
                del self.blocks[block]

    def _release(self, r: int) -> None:
        """Row ``r`` no longer uses its instance's block (forgotten when none does)."""
        block = int(self.inst_block[r])
        if block < 0:
            return
        self.inst_block[r] = -1
        entry = self.blocks[block]
        entry[1] -= 1
        if entry[1] == 0:
            self.block_rows -= entry[0]
            del self.blocks[block]

    def _compact(self, rows: np.ndarray) -> None:
        """Forget every instance: the next build puts all of ``rows`` in one block, so the
        blocks kept never hold more than about twice the rows in use."""
        drawn = set(rows.tolist())
        for r in np.flatnonzero(self.inst_block >= 0).tolist():
            self._release(r)
            if r not in drawn:  # (an instance not drawn now is built again when it is)
                self.instance[r] = None
                self.dirty[r] = True

    def _prune_keys(self) -> None:
        """Forget material keys no object uses any more (colors that changed, ...)."""
        mesh_rows = np.flatnonzero(self.kind == _MESH)
        live = set(self.mat_key[mesh_rows].tolist())
        grouped = mesh_rows[self.ngroups[mesh_rows] > 0].tolist()
        for r in grouped:
            live.update(key for _, key in self.group_keys[r])
        if len(self.key_info) <= 2 * len(live) + 1024:
            return
        remap = np.full(len(self.key_info), -1, dtype=np.int64)
        keys: dict[tuple[str, Any], int] = {}
        info: list[tuple[str, Any]] = []
        for key, k in self.keys.items():
            if k in live:
                remap[k] = len(info)
                keys[key] = len(info)
                info.append(self.key_info[k])
        self.mat_key[mesh_rows] = remap[self.mat_key[mesh_rows]]
        for r in grouped:
            self.group_keys[r] = tuple((g, int(remap[key])) for g, key in self.group_keys[r])
        self.keys, self.key_info = keys, info


def _uv_slots(material: RenderMaterial) -> bool:
    return any(t.needs_uvs for t in material.textures.values())


def _same_arrays(rm: RenderMesh, mesh: Mesh) -> bool:
    return (
        rm.positions is mesh.positions
        and rm.triangles is mesh.triangles
        and rm.normals is mesh.normals
        and rm.uvs is mesh.uvs
        and rm.groups is mesh.groups
        and rm.colors is mesh.colors
    )
