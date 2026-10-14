"""Materials, textures and procedural texture graphs (``docs/MATERIALS.md``).

Quick start::

    from aether3d import Scene

    scene = Scene()
    wall = scene.add_cube("Wall", size=2)
    stone = scene.create_material("Stone", preset="stone")       # procedural textures
    wall.material = stone

    crate = scene.create_material("Crate", roughness=0.7)
    crate.set_texture("base_color", "textures/crate.png")          # PNG or JPEG
    crate.set_procedural_texture("roughness", "fbm", scale=8, octaves=4)
    scene.assign_material(["Box 1", "Box 2"], crate)
    print(crate.inspect())

Modules: :mod:`.material` (materials, slots, bindings), :mod:`.textures`
(image and procedural textures, the image cache), :mod:`.graph` (procedural
node graphs: schemas, CPU evaluation, GPU programs), :mod:`.noise`
(reproducible hashes and noise), :mod:`.sampling` (mipmaps, filtering,
wrapping), :mod:`.images` (safe PNG/JPEG decoding), :mod:`.library` (a
scene's materials) and :mod:`.presets`.
"""

from .graph import NODE_TYPES, TextureGraph, node_types
from .images import DecodedImage, ImageLimits, decode_image, jpeg_available
from .library import MaterialLibrary
from .material import (
    ALPHA_MODES,
    COORDINATES,
    MODELS,
    PROJECTIONS,
    SLOTS,
    Material,
    TextureBinding,
    material_schema,
)
from .presets import preset_info, preset_names
from .textures import IMAGE_CACHE, ImageTexture, ProceduralTexture, Texture

__all__ = [
    "ALPHA_MODES",
    "COORDINATES",
    "IMAGE_CACHE",
    "MODELS",
    "NODE_TYPES",
    "PROJECTIONS",
    "SLOTS",
    "DecodedImage",
    "ImageLimits",
    "ImageTexture",
    "Material",
    "MaterialLibrary",
    "ProceduralTexture",
    "Texture",
    "TextureBinding",
    "TextureGraph",
    "decode_image",
    "jpeg_available",
    "material_schema",
    "node_types",
    "preset_info",
    "preset_names",
]
