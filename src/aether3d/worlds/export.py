"""Export a finite region of a world to glTF (``World.export_region``).

The region's generated geometry becomes ordinary objects of a temporary scene, which the
glTF writer exports (:func:`aether3d.gltf.writer.export_gltf`), so the file is like any
other Aether3D export -- +Y up, ids in ``extras``, validator-clean:

* terrain worlds: terrain tiles at one level of detail over the region (``level`` 0 is the
  finest; all tiles of one level, so no stitching is needed), their water surfaces, cave
  and overhang volumes (with the tiles' holes) when ``caves`` is on, plants and rocks
  (every plant of the region at the chosen ``plant_lod``, instanced: one mesh per
  species, variant and level), structures and bridges;
* block worlds: the block chunks (greedy-meshed, textured) and water.

The region snaps outward to whole tiles or chunks; the result reports the bounds actually
exported. Materials are copied from the scene's library; what glTF cannot express (the
terrain's world-space detail texture) is reported by the writer and left out, unless
``bake_textures`` bakes procedural textures (see the writer). Vertex colours (glTF
``COLOR_0``) carry the ground's, plants' and blocks' colours.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import WorldError

MAX_REGION = 8192.0
"""Largest side of an exported region (metres)."""
MAX_TRIANGLES = 20_000_000
MAX_PLANTS = 200_000
INCLUDE = ("terrain", "water", "caves", "vegetation", "structures")


def _err(code: str, message: str, location: str, hint: str | None = None) -> WorldError:
    return WorldError(code, message, location, hint=hint)


def export_region(
    world: Any,
    path: Any,
    region: Any,
    *,
    level: int = 0,
    include: Any = INCLUDE,
    plant_lod: int = 1,
    allow_lossy: bool = True,
    bake_textures: int | None = None,
    max_triangles: int = 4_000_000,
) -> dict[str, Any]:
    """Export ``region`` ``[x0, y0, x1, y1]`` (metres) of ``world`` to ``path`` (``.glb``
    or ``.gltf``). Returns what was written: bounds, objects, triangles, the writer's
    diagnostics."""
    from ..gltf.writer import export_gltf
    from ..scene import Scene

    loc = f"{world.location}.export"
    r = np.asarray(region, dtype=np.float64) if isinstance(region, (list, tuple)) else None
    if (
        r is None
        or r.shape != (4,)
        or not np.all(np.isfinite(r))
        or not (r[2] > r[0] and r[3] > r[1])
    ):
        raise _err(
            "INVALID_ARGUMENT",
            "region is [x0, y0, x1, y1] with x1 > x0 and y1 > y0",
            f"{loc}.region",
        )
    if max(r[2] - r[0], r[3] - r[1]) > MAX_REGION:
        raise _err(
            "LIMIT_EXCEEDED",
            f"a region's side is at most {MAX_REGION:.0f} m",
            f"{loc}.region",
            hint="export several regions",
        )
    include = tuple(include)
    for k in include:
        if k not in INCLUDE:
            raise _err(
                "INVALID_ARGUMENT", f"unknown part {k!r}", f"{loc}.include", hint=", ".join(INCLUDE)
            )
    if not isinstance(max_triangles, int) or not 1 <= max_triangles <= MAX_TRIANGLES:
        raise _err(
            "INVALID_ARGUMENT", f"max_triangles is in [1, {MAX_TRIANGLES}]", f"{loc}.max_triangles"
        )
    rt = world.runtime
    rt.regional()
    d = world._definition
    src = world._system.scene
    out = Scene(f"{world.name} region")
    ids = out.materials._copy_from(src.materials)
    layer_mats = {layer: ids.get(ref) for layer, ref in d["materials"].items()}
    parts: list[tuple[str, Any, np.ndarray, dict]] = []  # (name, mesh, matrix, {group: layer})
    bounds = [float(r[0]), float(r[1]), float(r[2]), float(r[3])]
    if d["mode"] == "blocks":
        bounds = _blocks(rt, r, parts)
    else:
        if "terrain" in include or "water" in include:
            bounds = _terrain(rt, r, level, include, parts)
        if "vegetation" in include:
            _plants(rt, bounds, plant_lod, parts)
        if "structures" in include:
            _structures(rt, bounds, parts)
    triangles = sum(m.triangle_count for _, m, _, _ in parts)
    if triangles > max_triangles:
        raise _err(
            "LIMIT_EXCEEDED",
            f"the region has {triangles} triangles (max_triangles {max_triangles})",
            f"{loc}.region",
            hint="export a smaller region, a coarser level or fewer parts",
        )
    with out.transaction("Region"):
        for name, mesh, matrix, layers in parts:
            s = float(np.linalg.norm(matrix[:3, 2]))
            half = math.atan2(matrix[1, 0], matrix[0, 0]) / 2.0
            obj = out.add_object(
                name,
                mesh=mesh,
                position=tuple(float(v) for v in matrix[:3, 3]),
                rotation=(0.0, 0.0, math.sin(half), math.cos(half)),
                scale=s,
            )
            groups = sorted(layers)
            base = layers[groups[0]]
            if layer_mats.get(base) is not None:
                out.assign_material(obj, layer_mats[base])
            for g in groups:
                m = layer_mats.get(layers[g])
                if m is not None and layers[g] != base:
                    out.assign_material(obj, m, group=g)
    result = export_gltf(out, path, allow_lossy=allow_lossy, bake_textures=bake_textures)
    return {
        "path": str(result.path),
        "format": result.format,
        "bytes": result.byte_length,
        "bounds": [round(b, 3) for b in bounds],
        "objects": len(parts),
        "meshes": result.mesh_count,
        "triangles": int(triangles),
        "diagnostics": [
            {"code": x.code, "severity": x.severity, "message": x.message}
            for x in result.diagnostics
        ],
    }


def _translation(x: float, y: float, z: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _terrain(rt: Any, r: np.ndarray, level: int, include: tuple, parts: list) -> list[float]:
    from .tiles import TileKey
    from .volumes import CHUNK as VCHUNK
    from .volumes import volumes_of

    grid = rt.grid()
    if not isinstance(level, int) or not 0 <= level <= rt.max_level():
        raise _err(
            "INVALID_ARGUMENT", f"level is an integer in [0, {rt.max_level()}]", "export.level"
        )
    size = grid.size(level)
    i0, i1 = math.floor(r[0] / size), math.ceil(r[2] / size) - 1
    j0, j1 = math.floor(r[1] / size), math.ceil(r[3] / size) - 1
    if (i1 - i0 + 1) * (j1 - j0 + 1) > 4096:
        raise _err(
            "LIMIT_EXCEEDED",
            "more than 4096 tiles: use a coarser level or a smaller region",
            "export.level",
        )
    holes: list = []
    vol = volumes_of(rt)
    if "caves" in include and vol.enabled():
        for ci in range(math.floor(i0 * size / VCHUNK), math.ceil((i1 + 1) * size / VCHUNK)):
            for cj in range(math.floor(j0 * size / VCHUNK), math.ceil((j1 + 1) * size / VCHUNK)):
                cubes = vol.column(ci, cj)
                surface = any(flags for _, flags in cubes)
                if surface:
                    holes.append((ci * VCHUNK, cj * VCHUNK, (ci + 1) * VCHUNK, (cj + 1) * VCHUNK))
                for ck, _flags in cubes:
                    mesh = vol.cube(ci, cj, ck, 15)
                    if mesh is not None:
                        parts.append(
                            (
                                f"cave {ci},{cj},{ck}",
                                mesh,
                                _translation(ci * VCHUNK, cj * VCHUNK),
                                {0: "terrain"},
                            )
                        )
    from .terrain_mesh import cut_holes

    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            key = TileKey(level, i, j)
            td = rt.tile(key, 0)
            mesh = td.mesh
            mine = [
                h
                for h in holes
                if h[0] < (i + 1) * size
                and h[2] > i * size
                and h[1] < (j + 1) * size
                and h[3] > j * size
            ]
            if mine:
                mesh = cut_holes(mesh, td.origin, mine)
            m = _translation(td.origin[0], td.origin[1])
            if "terrain" in include and mesh.triangle_count:
                parts.append((f"terrain {level}/{i},{j}", mesh, m, {0: "terrain"}))
            if "water" in include and td.water is not None:
                parts.append((f"water {level}/{i},{j}", td.water, m, {0: "water"}))
    return [i0 * size, j0 * size, (i1 + 1) * size, (j1 + 1) * size]


def _plants(rt: Any, b: list[float], lod: int, parts: list) -> None:
    from .assets import asset
    from .vegetation import CLASSES, SPECIES_ORDER, TILE, vegetation_of

    if not isinstance(lod, int) or not 0 <= lod <= 2:
        raise _err("INVALID_ARGUMENT", "plant_lod is 0, 1 or 2", "export.plant_lod")
    veg = vegetation_of(rt)
    requests = [
        (cls, i, j, 0)
        for cls in CLASSES
        for i in range(math.floor(b[0] / TILE), math.ceil(b[2] / TILE))
        for j in range(math.floor(b[1] / TILE), math.ceil(b[3] / TILE))
    ]
    count = 0
    for ps in veg.tiles(requests):
        inside = (
            (ps.pos[:, 0] >= b[0])
            & (ps.pos[:, 0] < b[2])
            & (ps.pos[:, 1] >= b[1])
            & (ps.pos[:, 1] < b[3])
        )
        for k in np.flatnonzero(inside):
            count += 1
            if count > MAX_PLANTS:
                raise _err(
                    "LIMIT_EXCEEDED",
                    f"more than {MAX_PLANTS} plants in the region",
                    "export.region",
                    hint="export a smaller region or leave vegetation out",
                )
            name = SPECIES_ORDER[int(ps.species[k])]
            a = asset(name, int(ps.variant[k]), lod, 0)
            c, s = math.cos(float(ps.yaw[k])), math.sin(float(ps.yaw[k]))
            sc = float(ps.scale[k])
            m = np.eye(4)
            m[0, 0], m[0, 1], m[1, 0], m[1, 1], m[2, 2] = c * sc, -s * sc, s * sc, c * sc, sc
            m[:3, 3] = ps.pos[k]
            parts.append((f"{name} {ps.ids()[k]}", a.mesh, m, dict(enumerate(a.layers))))


def _structures(rt: Any, b: list[float], parts: list) -> None:
    from . import noise as nz
    from .structures import LAYERS, _bridge, structure_mesh

    model = rt.model()
    for item in rt.definition["structures"]["items"]:
        if item["kind"] == "bridge":
            (ax, ay), (bx, by) = item["line"]
            if max(ax, bx) < b[0] or min(ax, bx) > b[2] or max(ay, by) < b[1] or min(ay, by) > b[3]:
                continue
            mesh, origin = _bridge(rt, item)
            parts.append((f"bridge {item['id']}", mesh, _translation(*origin), dict(LAYERS)))
            continue
        x, y = item["at"]
        if not (b[0] <= x < b[2] and b[1] <= y < b[3]):
            continue
        z = model.pad_height(f"structure:{item['id']}")
        c, s = nz.cos_sin_deg(item["rotation"])
        m = np.eye(4)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        m[:3, 3] = (x, y, z)
        parts.append((f"{item['kind']} {item['id']}", structure_mesh(item), m, dict(LAYERS)))


def _blocks(rt: Any, r: np.ndarray, parts: list) -> list[float]:
    from .voxels import CHUNK, _layers_of
    from .voxels import _blocks as block_world

    size = rt.definition["blocks"]["size"]
    span = CHUNK * size
    i0, i1 = math.floor(r[0] / span), math.ceil(r[2] / span) - 1
    j0, j1 = math.floor(r[1] / span), math.ceil(r[3] / span) - 1
    coords = [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    if len(coords) > 16384:
        raise _err(
            "LIMIT_EXCEEDED", "more than 16384 chunks: export a smaller region", "export.region"
        )
    bw = block_world(rt)
    for (i, j), (mesh, water) in zip(coords, bw.meshes(coords), strict=True):
        m = _translation(i * span, j * span)
        if mesh is not None:
            parts.append((f"chunk {i},{j}", mesh, m, _layers_of(mesh)))
        if water is not None:
            parts.append((f"water {i},{j}", water, m, {0: "water"}))
    return [i0 * span, j0 * span, (i1 + 1) * span, (j1 + 1) * span]
