"""Materials on the GPU: one storage buffer of material data and a cached pool of texels.

:func:`pack_material_data` writes everything ``wgsl/material.wgsl`` reads, as
``uint32`` words (floats as their bit patterns):

* a header of region offsets (``HEADER_*``);
* per drawn instance (draw order): its default material and its face-group
  table;
* material records (``MATERIAL_WORDS`` each): flags, linear factors and the
  binding of each slot;
* texture bindings (``BINDING_WORDS`` each): coordinates, projection,
  channel, triplanar sharpness, transform and the texture's program;
* texture programs: the instructions of every compiled procedural graph (an
  image texture is a one-instruction program), with operands relocated into
  one shared parameter region;
* image descriptors (``IMAGE_WORDS`` each): where an image's levels start in
  the texel pool, its size, level count and sampler;
* the 256-entry sRGB decoding table.

:class:`TexturePool` keeps the levels of every image of the last render in
one storage buffer (RGBA8, one texel per ``uint32``). It is keyed by image
content (SHA-256 digest), decoding mode and mipmaps, so it is uploaded again
only when the set of images changes -- an edited file has a new digest and can
never leave a stale image on the GPU.

Nothing here comes from user input as code: the shader is fixed and these
are numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ...errors import RenderError
from ...materials.graph import INSTRUCTION_WORDS, REGISTER_BIT, TextureGraph, compile_graph
from ...materials.sampling import SRGB_TABLE
from ..materials import RenderImage, RenderMaterial, RenderTexture
from ..scene_data import RenderScene

if TYPE_CHECKING:
    from .device import GPUContext

HEADER_WORDS = 16
(
    HEADER_INSTANCES,
    HEADER_MATERIALS,
    HEADER_BINDINGS,
    HEADER_PROGRAMS,
    HEADER_PARAMS,
    HEADER_IMAGES,
    HEADER_LUT,
    HEADER_GROUPS,
) = range(8)
INSTANCE_WORDS = 4
MATERIAL_WORDS = 24
BINDING_WORDS = 20
IMAGE_WORDS = 8
NO_BINDING = 0xFFFFFFFF

MATERIAL_STANDARD = 1
MATERIAL_MASK = 2
MATERIAL_DOUBLE_SIDED = 4
MATERIAL_TEXTURED = 8
SLOT_ORDER = ("base_color", "opacity", "metallic", "roughness", "normal", "emissive", "occlusion")
COORDINATES = {"uv": 0, "object": 1, "world": 2}
PROJECTIONS = {"triplanar": 0, "xy": 1, "xz": 2, "yz": 3, "volume": 4}
CHANNELS = {"rgb": 0, "r": 1, "g": 2, "b": 3, "a": 4, "luminance": 5}
BINDING_IMAGE = 4096
WRAPS = {"repeat": 0, "clamp": 1, "mirror": 2}
OP_IMAGE = 16

CONSTANTS = {
    "HEADER_INSTANCES": HEADER_INSTANCES,
    "HEADER_MATERIALS": HEADER_MATERIALS,
    "HEADER_BINDINGS": HEADER_BINDINGS,
    "HEADER_PROGRAMS": HEADER_PROGRAMS,
    "HEADER_PARAMS": HEADER_PARAMS,
    "HEADER_IMAGES": HEADER_IMAGES,
    "HEADER_LUT": HEADER_LUT,
    "HEADER_GROUPS": HEADER_GROUPS,
    "INSTANCE_WORDS": INSTANCE_WORDS,
    "MATERIAL_WORDS": MATERIAL_WORDS,
    "BINDING_WORDS": BINDING_WORDS,
    "IMAGE_WORDS": IMAGE_WORDS,
    "INSTRUCTION_WORDS": INSTRUCTION_WORDS,
    "MATERIAL_STANDARD": MATERIAL_STANDARD,
    "MATERIAL_MASK": MATERIAL_MASK,
    "MATERIAL_DOUBLE_SIDED": MATERIAL_DOUBLE_SIDED,
    "MATERIAL_TEXTURED": MATERIAL_TEXTURED,
    "BINDING_IMAGE": BINDING_IMAGE,
    "OP_IMAGE": OP_IMAGE,
}
"""Constants shared with ``wgsl/material.wgsl`` (a test checks the shader uses the same values)."""


def _f32_bits(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32)


def image_key(image: RenderImage) -> tuple[str, str, bool]:
    return (image.digest, image.chain.mode, image.mipmaps)


# --------------------------------------------------------------------------- #
# Programs
# --------------------------------------------------------------------------- #


@dataclass
class _Program:
    first: int  # first instruction (index in the program region)
    count: int


@dataclass
class _Builder:
    instructions: list[np.ndarray] = field(default_factory=list)  # (k * 8,) uint32 each
    params: list[np.ndarray] = field(default_factory=list)  # float32 arrays
    param_count: int = 0
    instruction_count: int = 0
    images: list[RenderImage] = field(default_factory=list)
    image_index: dict[tuple, int] = field(default_factory=dict)
    programs: dict[tuple, _Program] = field(default_factory=dict)

    def image(self, image: RenderImage) -> int:
        key = (*image_key(image), image.wrap, image.filter)
        index = self.image_index.get(key)
        if index is None:
            index = self.image_index[key] = len(self.images)
            self.images.append(image)
        return index

    def program(self, rt: RenderTexture) -> _Program:
        key = (rt.texture_id, rt.image.chain.mode if rt.image is not None else "")
        found = self.programs.get(key)
        if found is not None:
            return found
        if rt.kind == "image":
            # An image texture is a graph with one image node.
            graph = TextureGraph.from_dict({"type": "image", "texture": rt.texture_id})
            images = [rt.image]
        else:
            graph = rt.graph
            images = None
        compiled = compile_graph(graph, f"textures[{rt.name}]")
        words = compiled.words.reshape(-1, INSTRUCTION_WORDS).copy()
        base = self.param_count
        for row in words:
            for k in range(2, 6):  # operand constants: offsets into the parameter region
                if (row[0] & 0xFF) != 32 and not row[k] & REGISTER_BIT:
                    row[k] = row[k] + base
            row[6] = row[6] + base
            if (row[0] & 0xFF) == OP_IMAGE:
                if images is not None:
                    image = images[0]
                else:
                    ref, space = compiled.images[int(row[7])]
                    image = rt.images[(ref, space)]
                row[7] = self.image(image)
        self.params.append(compiled.params)
        self.param_count += int(compiled.params.shape[0])
        program = _Program(self.instruction_count, words.shape[0])
        self.instructions.append(words.reshape(-1))
        self.instruction_count += words.shape[0]
        self.programs[key] = program
        return program


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


@dataclass
class MaterialData:
    """The packed words and what they reference."""

    words: np.ndarray  # uint32
    images: list[RenderImage]
    textured: int  # materials with textures
    bindings: int
    instructions: int


def pack_material_data(
    rs: RenderScene, order: np.ndarray, image_offsets: dict[tuple, int] | None = None
) -> MaterialData:
    """Material data for ``rs`` with instances in draw ``order`` (see the module docstring).

    ``image_offsets`` maps :func:`image_key` to the texel offset of each image
    in the pool; when ``None`` the descriptors get offset 0 (for layout tests).
    """
    builder = _Builder()
    materials: list[np.ndarray] = []
    bindings: list[np.ndarray] = []
    textured = 0
    for m in rs.materials:
        record, slot_bindings = _material_record(m)
        for slot, rt in m.textures.items():
            slot_bindings[SLOT_ORDER.index(slot)] = len(bindings)
            bindings.append(_binding_record(rt, builder.program(rt)))
        record[16:23] = slot_bindings
        textured += bool(m.textures)
        materials.append(record)
    if not materials:  # a valid (unused) record keeps the buffer layout simple
        materials.append(np.zeros(MATERIAL_WORDS, dtype=np.uint32))

    # Per-instance tables in draw order, and the face-group tables they point to.
    instance_table = np.zeros((max(len(order), 1), INSTANCE_WORDS), dtype=np.uint32)
    groups: list[int] = []
    for row, index in enumerate(order):
        inst = rs.instances[int(index)]
        instance_table[row, 0] = inst.material
        if inst.group_materials:
            count = max(g for g, _ in inst.group_materials) + 1
            table = [inst.material] * count
            for g, m in inst.group_materials:
                table[g] = m
            instance_table[row, 1] = len(groups)
            instance_table[row, 2] = count
            groups.extend(table)

    images = np.zeros((max(len(builder.images), 1), IMAGE_WORDS), dtype=np.uint32)
    for i, image in enumerate(builder.images):
        chain = image.chain
        flags = WRAPS[image.wrap[0]] | (WRAPS[image.wrap[1]] << 2)
        flags |= 16 if image.filter == "linear" else 0
        flags |= 32 if chain.mode == "srgb" else 0
        flags |= 64 if image.mipmaps and len(chain.levels) > 1 else 0
        offset = 0 if image_offsets is None else image_offsets[image_key(image)]
        images[i, :5] = (offset, chain.width, chain.height, len(chain.levels), flags)

    regions = [
        instance_table.reshape(-1),
        np.concatenate(materials),
        np.concatenate(bindings) if bindings else np.zeros(BINDING_WORDS, dtype=np.uint32),
        np.concatenate(builder.instructions)
        if builder.instructions
        else np.zeros(INSTRUCTION_WORDS, dtype=np.uint32),
        _f32_bits(np.concatenate(builder.params)) if builder.params else np.zeros(4, np.uint32),
        images.reshape(-1),
        _f32_bits(SRGB_TABLE),
        np.array(groups or [0], dtype=np.uint32),
    ]
    header = np.zeros(HEADER_WORDS, dtype=np.uint32)
    offset = HEADER_WORDS
    for k, region in enumerate(regions):
        header[k] = offset
        offset += region.shape[0]
    words = np.concatenate([header, *regions]).astype(np.uint32)
    return MaterialData(words, builder.images, textured, len(bindings), builder.instruction_count)


def _material_record(m: RenderMaterial) -> tuple[np.ndarray, np.ndarray]:
    record = np.zeros(MATERIAL_WORDS, dtype=np.uint32)
    flags = MATERIAL_STANDARD if m.model == "standard" else 0
    flags |= MATERIAL_MASK if m.masked else 0
    flags |= MATERIAL_DOUBLE_SIDED if m.double_sided else 0
    flags |= MATERIAL_TEXTURED if m.textures else 0
    record[0] = flags
    floats = np.zeros(15, dtype=np.float32)
    floats[0:3] = m.base_color
    floats[3] = m.opacity
    floats[4] = m.alpha_cutoff
    floats[5] = m.metallic
    floats[6] = m.roughness
    floats[7] = m.specular
    floats[8] = m.shininess
    floats[9:12] = m.emissive
    floats[12] = m.normal_strength
    floats[13] = m.occlusion_strength
    record[1:16] = floats.view(np.uint32)
    return record, np.full(7, NO_BINDING, dtype=np.uint32)


def _binding_record(rt: RenderTexture, program: _Program) -> np.ndarray:
    record = np.zeros(BINDING_WORDS, dtype=np.uint32)
    flags = COORDINATES[rt.coordinates] | (PROJECTIONS[rt.projection] << 4)
    flags |= CHANNELS[rt.channel] << 8
    flags |= BINDING_IMAGE if rt.kind == "image" else 0
    record[0] = flags
    record[1] = program.first
    record[2] = program.count
    record[3] = _f32_bits([rt.blend])[0]
    record[4:16] = _f32_bits(rt.transform.reshape(-1))
    return record


# --------------------------------------------------------------------------- #
# The texel pool
# --------------------------------------------------------------------------- #


class TexturePool:
    """The texels of the last render's images, in one storage buffer (see the module docstring)."""

    def __init__(self, context: GPUContext) -> None:
        self.context = context
        self.buffer: Any = None
        self.keys: frozenset = frozenset()
        self.offsets: dict[tuple, int] = {}
        self.nbytes = 0
        self.uploads = 0
        self.upload_bytes = 0
        self.upload_seconds = 0.0

    def clear(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
        self.buffer = None
        self.keys = frozenset()
        self.offsets = {}
        self.nbytes = 0

    def stats(self) -> dict[str, Any]:
        return {
            "images": len(self.offsets),
            "bytes": self.nbytes,
            "uploads": self.uploads,
            "upload_bytes": self.upload_bytes,
        }

    def prepare(self, images: list[RenderImage]) -> tuple[Any, dict[tuple, int], int]:
        """The texel buffer holding ``images``, their offsets, and bytes uploaded now."""
        chains = {}
        for image in images:
            chains.setdefault(image_key(image), image.chain)
        keys = frozenset(chains)
        if self.buffer is not None and keys <= self.keys:
            return self.buffer, self.offsets, 0
        total = sum(
            sum(level.shape[0] * level.shape[1] for level in c.levels) for c in chains.values()
        )
        nbytes = max(4 * total, 256)
        limit = self.context.limits.get("max-storage-buffer-binding-size", 128 * 2**20)
        if nbytes > limit:
            raise RenderError(
                "GPU_LIMIT_EXCEEDED",
                f"the scene's textures need {nbytes / 2**20:,.1f} MiB of texels; "
                f"{self.context.info.name} binds at most {limit / 2**20:,.0f} MiB",
                "textures",
                hint="use smaller images, or render with backend='cpu'",
                details={"texel_bytes": nbytes, "limit": limit},
            )
        t0 = time.perf_counter()
        data = np.zeros(nbytes // 4, dtype=np.uint32)
        offsets: dict[tuple, int] = {}
        at = 0
        for key, chain in chains.items():
            offsets[key] = at
            for level in chain.levels:
                texel = level.reshape(-1, 4).astype(np.uint32)
                packed = (
                    texel[:, 0] | (texel[:, 1] << 8) | (texel[:, 2] << 16) | (texel[:, 3] << 24)
                )
                data[at : at + packed.shape[0]] = packed  # (any host byte order)
                at += packed.shape[0]
        self.clear()
        wgpu = self.context.wgpu
        self.buffer = self.context.device.create_buffer_with_data(
            label="aether3d-texels", data=data, usage=wgpu.BufferUsage.STORAGE
        )
        self.keys, self.offsets, self.nbytes = keys, offsets, nbytes
        self.uploads += 1
        self.upload_bytes += nbytes
        self.upload_seconds += time.perf_counter() - t0
        return self.buffer, offsets, nbytes
