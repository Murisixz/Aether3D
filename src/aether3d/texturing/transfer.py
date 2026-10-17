"""Texture transfer: carry an object's painted look onto another mesh of the same shape.

Remeshing (``voxel_remesh``, or a retopology made elsewhere) replaces every vertex, so UVs
and the textures painted through them do not survive. Keep a copy of the object before the
change (``scene.duplicate(obj)``), then :func:`transfer_textures` unwraps the new mesh
(when it has no UVs) and bakes the old object's evaluated material -- base color,
roughness, metallic and its shading normals as a tangent-space normal map, by default --
into new painted textures on the new UVs, bound to a material of the target's own: all
one step of the scene's history.

The maps hold the evaluated values (factor x texture) in 8/16-bit texture encodings: base
color and emission are clamped to [0, 1] (an emission stronger than 1 is not carried).
"""

from __future__ import annotations

from typing import Any

from ..errors import TexturingError
from .bake import _object, apply_bake, bake

DEFAULT_MAPS = ("base_color", "roughness", "metallic", "normal")
COPIED_PARAMS = ("model", "double_sided", "alpha_mode", "alpha_cutoff", "normal_strength")


def transfer_textures(
    scene: Any,
    source: Any,
    target: Any,
    maps: Any = DEFAULT_MAPS,
    *,
    resolution: Any = 1024,
    unwrap: Any = "auto",
    own_material: bool = True,
    label: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Bake ``source``'s material (``maps``) onto ``target`` through ``target``'s UVs.

    ``unwrap``: ``"auto"`` unwraps the target only when it has no UVs, ``True`` always,
    ``False`` never (a target without UVs is then refused). ``own_material``: bind the maps
    to a new material (copying the source material's model, sidedness and alpha settings)
    when the target has none or shares its material with any other object, so the source
    keeps its look; ``False`` binds them to the target's material as it is. ``options``
    go to :func:`~aether3d.texturing.bake` (``samples``, ``cage``, ``depth``, ``padding``,
    ``backend``, ...). One history step; returns the maps, textures, material and the
    bake's report."""
    src = _object(scene, source, "source")
    dst = _object(scene, target, "target")
    if src is dst:
        raise TexturingError(
            "INVALID_ARGUMENT", "source and target must be different objects", "target"
        )
    if unwrap not in ("auto", True, False):
        raise TexturingError("INVALID_ARGUMENT", "unwrap must be 'auto', true or false", "unwrap")
    if not isinstance(own_material, bool):
        raise TexturingError(
            "INVALID_ARGUMENT", "own_material must be true or false", "own_material"
        )
    if dst.mesh is None or dst.mesh.triangle_count == 0:
        raise TexturingError("NOT_A_MESH_OBJECT", f"{dst.name!r} has no triangles", "target")
    if "apply" in options:
        raise TexturingError("INVALID_ARGUMENT", "transfer_textures always applies", "apply")
    names = [maps] if isinstance(maps, str) else list(maps)
    unwrapped = False
    with scene.transaction(label or f"Transfer textures {src.name!r} -> {dst.name!r}"):
        if unwrap is True or (unwrap == "auto" and dst.mesh.uvs is None):
            dst.unwrap(resolution=resolution)
            unwrapped = True
        result = bake(scene, dst, [src], names, resolution=resolution, **options)
        material = dst.material
        shared = material is not None and any(
            o is not dst and o.material is material for o in scene.objects
        )
        if own_material and (material is None or shared):
            material = scene.create_material(
                scene.materials._unique_material_name(f"{dst.name} transferred")
            )
            if src.material is not None:
                for param in COPIED_PARAMS:
                    if hasattr(src.material, param):
                        setattr(material, param, getattr(src.material, param))
            dst.material = material
        applied = apply_bake(scene, dst, result, material=material)
    return {
        "source": src.name,
        "target": dst.name,
        "maps": list(result.maps),
        "unwrapped": unwrapped,
        "material": applied["material"].name,
        "textures": {k: t.name for k, t in applied["textures"].items()},
        "report": result.report,
        "diagnostics": [d.to_dict() for d in result.diagnostics],
    }
