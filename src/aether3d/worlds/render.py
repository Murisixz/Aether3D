"""Worlds in renders: what a world adds to a render scene for one camera.

``render()`` extracts the scene's objects as always, then asks each world for its
**contribution** for that frame's camera (:func:`contribute`):

* terrain worlds: the LOD-selected, stitched terrain tiles and their water surfaces;
* block worlds: the section meshes within the view distance (:mod:`.voxels`);
* both: plants and rocks near the camera (:mod:`.vegetation`), when present.

They are appended to the render scene as meshes and instances -- not scene objects:
nothing is added to the scene, its history or its files -- with materials resolved from
the scene's library (the world's layers, ``definition["materials"]``). Every backend then
renders them like any other geometry (instanced draws, shadows, path tracing). Their
render rows carry synthetic ids, ``world:<world id>:<part>``, and names
``<world> <part>``. The selection depends only on the frame's camera and the world, never
on earlier frames, so frames can be rendered in any order.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from ..rendering.materials import MaterialTable, _material, _prepare, appearance_material
from ..rendering.scene_data import InstanceArrays, RenderInstance, RenderMesh, RenderScene
from ..rendering.settings import Appearance
from .tiles import view_from_camera

SHADOW_MARGIN = 600.0
"""Metres the view frustum is widened by when choosing where to keep detail (tiles just
outside the view still cast detailed shadows into it)."""


def contribute(
    scene: Any, rs: RenderScene, height: int, aspect: float, ctx: Any = None
) -> RenderScene:
    """``rs`` with every alive world's geometry for its camera appended (see the module
    docstring)."""
    worlds = scene._worlds
    if worlds is None or not len(worlds):
        return rs
    t0 = time.perf_counter()
    cam = rs.camera
    # (id, name, mesh, origin (3,) or matrix (4, 4), layers): a layer name, a tuple (the
    # layer of face group 0, 1, ...) or a dict {face group: layer}
    parts: list[tuple[str, str, Any, np.ndarray, Any]] = []
    diagnostics: list[Diagnostic] = []
    for world in worlds:
        rt = world.runtime
        d = world._definition
        view = view_from_camera(
            cam.camera, cam.frame, height, aspect, d["streaming"]["view_distance"], SHADOW_MARGIN
        )
        t1 = time.perf_counter()
        if d["mode"] == "blocks":
            from .voxels import render_parts

            got = render_parts(rt, view, ctx)
        else:
            got = _terrain_parts(world, rt, view, ctx)
        from .vegetation import render_parts as plant_parts

        got += plant_parts(rt, view, ctx)
        rt.last_frame = {
            "parts": len(got),
            "triangles": int(sum(m.triangle_count for _, _, m, _, _ in got)),
            "seconds": round(time.perf_counter() - t1, 4),
        }
        parts += [p for p in got if p[2].triangle_count > 0]
    if not parts:
        return rs
    # Materials: the world layers, resolved like library materials (cached per library
    # version), appended after the objects' materials.
    table = MaterialTable(materials=list(rs.materials))
    table._index = {m.key: k for k, m in enumerate(rs.materials)}
    from ..rendering.scene_data import _material_cache

    cache = _material_cache(scene)
    if cache is None:
        from ..rendering.materials import MaterialCache

        cache = MaterialCache()
    _prepare(cache, scene.materials)
    this_render: dict[str, Any] = {}
    layer_index: dict[tuple[str, str], int] = {}
    fallback = None
    library = scene.materials
    meshes = list(rs.meshes)
    mesh_index: dict[int, int] = {}
    instances = list(rs.instances)
    rows_mesh, rows_mat, rows_groups = [], [], []

    def layer_material(wkey: str, layer: str, position: int) -> int:
        nonlocal fallback
        key = (wkey, layer)
        mi = layer_index.get(key)
        if mi is None:
            world = worlds.get(wkey)
            ref = world._definition["materials"].get(layer)
            material = library._materials.get(ref) if ref is not None else None
            if material is None:
                if fallback is None:
                    fallback = table.add(appearance_material(Appearance(), "world:fallback"))
                diagnostics.append(
                    Diagnostic(
                        "WORLD_MATERIAL_MISSING",
                        WARNING,
                        f"world {world.name!r} has no material for its {layer!r} layer; it is "
                        "drawn grey",
                        f"{world.location}.definition.materials.{layer}",
                    )
                )
                mi = fallback
            else:
                mi = table.add(_material(material, cache, this_render, table, position))
            layer_index[key] = mi
        return mi

    # World matrices (a translation or a full matrix per part) and their normal matrices,
    # computed together.
    n = len(parts)
    worlds_m = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    full = np.zeros(n, dtype=bool)
    for k, part in enumerate(parts):
        place = np.asarray(part[3], dtype=np.float64)
        if place.shape == (3,):
            worlds_m[k, :3, 3] = place
        else:
            worlds_m[k] = place
            full[k] = True
    normals = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    if np.any(full):
        normals[full] = np.transpose(np.linalg.inv(worlds_m[full, :3, :3]), (0, 2, 1))
    worlds_m.flags.writeable = False
    normals.flags.writeable = False
    mats_of: dict[tuple, tuple[int, tuple[tuple[int, int], ...]]] = {}
    for wid, (oid, name, mesh, _place, layers) in enumerate(parts):
        wkey = oid.split(":")[1]
        if isinstance(layers, str):
            layers = {0: layers}
        elif not isinstance(layers, dict):
            layers = dict(enumerate(layers))
        key = (wkey, tuple(sorted(layers.items())))
        found = mats_of.get(key)
        if found is None:
            by_group = {g: layer_material(wkey, layer, 10**9 + wid) for g, layer in key[1]}
            base = by_group[min(by_group)]
            found = mats_of[key] = (
                base,
                tuple((g, m) for g, m in sorted(by_group.items()) if m != base),
            )
        base, groups = found
        k = mesh_index.get(id(mesh))
        if k is None:
            k = mesh_index[id(mesh)] = len(meshes)
            meshes.append(
                RenderMesh(
                    mesh.positions, mesh.triangles, mesh.normals, mesh.uvs, mesh.groups, mesh.colors
                )
            )
        instances.append(
            RenderInstance(
                oid, name, k, worlds_m[wid], normals[wid], False, _WHITE, 0.0, 32.0, base, groups
            )
        )
        rows_mesh.append(k)
        rows_mat.append(base)
        rows_groups.append(bool(groups))
    diagnostics += [d for _, d in table.diagnostics]
    kept = tuple(d for d in rs.diagnostics if d.code != "NOTHING_TO_RENDER")  # (worlds render)
    extra = InstanceArrays(
        world=worlds_m,
        normal_matrix=normals,
        mirrored=np.zeros(n, dtype=bool),
        mesh=np.array(rows_mesh, dtype=np.int64),
        base_color=np.ones((n, 3)),
        specular=np.zeros(n),
        shininess=np.full(n, 32.0),
        material=np.array(rows_mat, dtype=np.int64),
        has_groups=np.array(rows_groups, dtype=bool),
    )
    a = rs.instance_arrays
    arrays = InstanceArrays(
        *(
            np.concatenate([getattr(a, f.name), getattr(extra, f.name)])
            for f in dataclasses.fields(InstanceArrays)
        )
    )
    timings = dict(rs.timings)
    timings["worlds"] = time.perf_counter() - t0
    return dataclasses.replace(
        rs,
        meshes=tuple(meshes),
        instances=tuple(instances),
        materials=tuple(table.materials),
        arrays=arrays,
        diagnostics=kept + tuple(diagnostics),
        timings=timings,
        cache=None,
    )


_IDENTITY3 = np.eye(3)
_IDENTITY3.flags.writeable = False
_WHITE = np.ones(3)
_WHITE.flags.writeable = False


def _terrain_parts(world: Any, rt: Any, view: Any, ctx: Any) -> list:
    from .volumes import render_parts as volume_parts

    out, holes = volume_parts(rt, view, ctx)
    for key, _mask, td in rt.select(view, ctx, holes):
        origin = np.array([td.origin[0], td.origin[1], 0.0])
        out.append(
            (
                f"world:{world.id}:terrain/{key}",
                f"{world.name} terrain {key}",
                td.mesh,
                origin,
                "terrain",
            )
        )
        if td.water is not None:
            out.append(
                (
                    f"world:{world.id}:water/{key}",
                    f"{world.name} water {key}",
                    td.water,
                    origin,
                    "water",
                )
            )
    return out
