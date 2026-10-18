"""glTF 2.0 export (``.glb`` binary and ``.gltf`` with an embedded buffer).

Materials are written as glTF metallic-roughness materials, with image
textures embedded in the buffer (see :mod:`aether3d.gltf.materials`). A mesh
with face groups becomes one primitive per group (sharing the vertex
attributes), each with its group's material; objects that share a mesh but
not materials get separate glTF meshes over the same accessors.
"""

from __future__ import annotations

import base64
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._fileio import write_atomic
from .._version import __version__
from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import UnsupportedFeatureError, ValidationError
from ..mesh import Mesh
from . import axes

if TYPE_CHECKING:
    from ..scene import Scene

GLB_MAGIC = 0x46546C67  # "glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A  # "JSON"
CHUNK_BIN = 0x004E4942  # "BIN\0"

FLOAT = 5126
UNSIGNED_SHORT = 5123
UNSIGNED_INT = 5125
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
TRIANGLES = 4

EXTRAS_KEY = "aether3d"
"""Key under a node's ``extras`` holding Aether3D metadata (``{"id": ...}``)."""

NORMAL_RENORMALIZE_TOLERANCE = 1e-5
"""Exported normals whose length differs from 1 by more than this are renormalized."""


@dataclass
class ExportResult:
    """Outcome of an export. ``to_dict()`` gives a JSON-compatible summary."""

    path: Path
    format: str
    byte_length: int
    node_count: int
    mesh_count: int
    nodes: list[dict[str, Any]]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    animations: list[dict[str, Any]] = field(default_factory=list)
    skins: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "format": self.format,
            "byte_length": self.byte_length,
            "node_count": self.node_count,
            "mesh_count": self.mesh_count,
            "nodes": [dict(n) for n in self.nodes],
            "animations": [dict(a) for a in self.animations],
            "skins": self.skins,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def output_format(path: str | os.PathLike[str]) -> str:
    """``"glb"`` or ``"gltf"`` from the file extension; raises ``ValidationError`` otherwise."""
    suffix = Path(path).suffix.lower()
    if suffix not in (".glb", ".gltf"):
        raise ValidationError(
            f"unsupported export format {suffix or '(no extension)'!r}; "
            "supported: .glb, .gltf (use Scene.save() for .aether projects)"
        )
    return suffix[1:]


class _BufferBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def add_bytes(self, raw: bytes) -> int:
        """A buffer view holding ``raw`` (images); returns its index."""
        while len(self.data) % 4:
            self.data.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)})
        self.data += raw
        return len(self.views) - 1

    def add(
        self, array: np.ndarray, component: int, kind: str, target: int | None, minmax: bool
    ) -> int:
        """An accessor over a new buffer view (``target`` ``None`` for data that is neither
        vertices nor indices: animation keys, inverse bind matrices)."""
        while len(self.data) % 4:
            self.data.append(0)
        raw = (
            np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"), copy=False).tobytes()
        )
        view: dict[str, Any] = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        self.data += raw
        accessor: dict[str, Any] = {
            "bufferView": len(self.views) - 1,
            "componentType": component,
            "count": int(array.shape[0]),
            "type": kind,
        }
        if minmax:
            accessor["min"] = [float(v) for v in array.min(axis=0)]
            accessor["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _export_mesh(
    mesh: Mesh,
    location: str,
    buffers: _BufferBuilder,
    diagnostics: list[Diagnostic],
    tangents: bool = False,
) -> dict[str, Any] | None:
    """Encode ``mesh``'s accessors: ``{"attributes": ..., "groups": {group: index accessor}}``;
    ``None`` if glTF cannot represent it."""
    if mesh.triangle_count == 0:
        diagnostics.append(
            Diagnostic(
                "EMPTY_MESH_NOT_EXPORTED",
                WARNING,
                "glTF cannot store a mesh without triangles; objects using it are exported "
                "without geometry",
                location,
                lossy=True,
            )
        )
        return None
    tangent = None
    source = None  # (when vertices are split: each new vertex's original)
    if tangents and mesh.normals is not None and mesh.uvs is not None:
        from .materials import split_for_tangents_mapped

        before = mesh.vertex_count
        mesh, tangent, source = split_for_tangents_mapped(mesh)
        if mesh.vertex_count > before:
            diagnostics.append(
                Diagnostic(
                    "VERTICES_SPLIT_FOR_TANGENTS",
                    INFO,
                    "vertices whose corners have different MikkTSpace tangents (mirrored UVs, "
                    "UV seams of one position) were duplicated for glTF's per-vertex TANGENT",
                    location,
                    mesh.vertex_count - before,
                )
            )
    attributes = {
        "POSITION": buffers.add(
            axes.vectors_to_gltf(mesh.positions), FLOAT, "VEC3", ARRAY_BUFFER, True
        )
    }
    if mesh.normals is not None:
        normals = axes.vectors_to_gltf(mesh.normals).astype(np.float64)
        length = np.linalg.norm(normals, axis=1)
        if np.any(length < 1e-12):
            diagnostics.append(
                Diagnostic(
                    "ZERO_NORMALS_NOT_EXPORTED",
                    WARNING,
                    "mesh has zero-length normals, which glTF forbids; its normals are omitted",
                    location,
                    int(np.sum(length < 1e-12)),
                    lossy=True,
                )
            )
        else:
            off = np.abs(length - 1.0) > NORMAL_RENORMALIZE_TOLERANCE
            if off.any():
                diagnostics.append(
                    Diagnostic(
                        "NORMALS_RENORMALIZED",
                        WARNING,
                        "glTF requires unit normals; non-unit normals were renormalized "
                        "(directions are unchanged)",
                        location,
                        int(off.sum()),
                    )
                )
                normals[off] /= length[off, None]
            attributes["NORMAL"] = buffers.add(
                normals.astype(np.float32), FLOAT, "VEC3", ARRAY_BUFFER, False
            )
    if mesh.uvs is not None:
        attributes["TEXCOORD_0"] = buffers.add(
            axes.flip_uvs(mesh.uvs), FLOAT, "VEC2", ARRAY_BUFFER, False
        )
    if mesh.colors is not None:  # linear RGBA, as Mesh.colors (glTF's COLOR_0 semantics)
        attributes["COLOR_0"] = buffers.add(
            np.ascontiguousarray(mesh.colors, dtype=np.float32), FLOAT, "VEC4", ARRAY_BUFFER, False
        )
    if tangent is not None and "NORMAL" in attributes:
        # MikkTSpace on Aether3D's V-up UVs gives glTF's tangent and sign directly (the axes
        # change is a rotation; flipping V for TEXCOORD_0 is accounted for by the sign
        # convention: bitangent = cross(normal, tangent) * w points up the image).
        xyz = axes.vectors_to_gltf(tangent[:, :3].astype(np.float32)).astype(np.float64)
        xyz /= np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-30)
        values = np.concatenate([xyz, tangent[:, 3:]], axis=1).astype(np.float32)
        attributes["TANGENT"] = buffers.add(values, FLOAT, "VEC4", ARRAY_BUFFER, False)
    # 65535 is reserved as the primitive-restart value for 16-bit indices.
    if mesh.vertex_count <= 65535:
        indices, component = mesh.triangles.astype(np.uint16), UNSIGNED_SHORT
    else:
        indices, component = mesh.triangles.astype(np.uint32), UNSIGNED_INT
    groups: dict[int, int] = {}
    for group in mesh.group_ids:
        part = indices if mesh.groups is None else indices[mesh.groups == group]
        groups[group] = buffers.add(
            part.reshape(-1, 1), component, "SCALAR", ELEMENT_ARRAY_BUFFER, False
        )
    return {
        "attributes": attributes,
        "groups": groups,
        "grouped": mesh.groups is not None,
        "source": source,
    }


def build_gltf(
    scene: Scene,
    *,
    bake_textures: int | None = None,
    animation: str = "auto",
    animation_options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes, list[Diagnostic], list[dict[str, Any]]]:
    """Build the glTF JSON document and binary buffer for ``scene`` (no I/O).

    Returns ``(document, buffer_bytes, diagnostics, node_map)``. The buffer is
    referenced as buffer 0 without a ``uri``; callers add one for ``.gltf``.
    ``bake_textures`` (a size in pixels) bakes procedural textures to PNG;
    ``animation`` and ``animation_options`` see :mod:`aether3d.gltf.anim_export`.
    """
    from . import anim_export
    from .materials import MaterialExporter

    objects = list(scene)  # depth-first: node index == position in this list
    node_of = {obj.id: index for index, obj in enumerate(objects)}
    buffers = _BufferBuilder()
    diagnostics: list[Diagnostic] = []
    exporter = MaterialExporter(buffers, diagnostics, bake_textures)
    encoded_meshes: dict[int, dict[str, Any] | None] = {}  # id(mesh) -> accessors
    mesh_index: dict[tuple, int] = {}  # (id(mesh), materials) -> glTF mesh
    meshes: list[dict[str, Any]] = []
    # Meshes drawn with a UV normal map get TANGENT attributes (glTF's portable tangent space).
    normal_mapped = set()
    for obj in objects:
        if obj._mesh is None and obj._geometry is None:
            continue
        refs = ([obj._material] if obj._material else []) + list(obj._group_materials.values())
        for ref in refs:
            material = scene.materials.find_material(ref)
            binding = material.textures.get("normal") if material is not None else None
            if binding is not None and binding.coordinates == "uv":
                normal_mapped.add(id(obj.mesh))
    nodes: list[dict[str, Any]] = []
    node_map: list[dict[str, Any]] = []
    rigging = anim_export.collect(scene)
    carried: list[tuple[Any, int]] = []  # skinned objects: (object, glTF mesh)

    def gltf_mesh(obj: Any, encoded: dict[str, Any], own: bool) -> int:
        """The glTF mesh of ``obj`` (shared unless it carries a skin or morph targets)."""
        groups = encoded["groups"]
        materials = tuple(exporter.for_object(obj, g) for g in groups)
        key = (id(obj.mesh), materials, obj.id if own else None)
        if key not in mesh_index:
            attributes = dict(encoded["attributes"])
            skin = rigging.skins.get(obj.id) if own else None
            if skin is not None:
                attributes.update(anim_export.skin_attributes(skin, encoded["source"], buffers))
            morph = rigging.morphs.get(obj.id) if own else None
            targets = None
            if morph is not None:
                targets = anim_export.morph_targets(
                    morph, encoded["source"], buffers, "NORMAL" in attributes
                )
            primitives = []
            for (group, indices), material in zip(groups.items(), materials, strict=True):
                primitive: dict[str, Any] = {
                    "attributes": attributes,
                    "indices": indices,
                    "mode": TRIANGLES,
                }
                if material is not None:
                    primitive["material"] = material
                if targets is not None:
                    primitive["targets"] = targets
                if encoded["grouped"]:
                    primitive["extras"] = {EXTRAS_KEY: {"group": group}}
                primitives.append(primitive)
            entry: dict[str, Any] = {"primitives": primitives}
            if morph is not None:
                entry["weights"] = [float(w) for w in morph.defaults]
                entry["extras"] = {"targetNames": list(morph.names)}
            mesh_index[key] = len(meshes)
            meshes.append(entry)
        return mesh_index[key]

    for obj in objects:
        node: dict[str, Any] = {"name": obj.name}
        t = obj.transform
        if t.position != (0.0, 0.0, 0.0):
            node["translation"] = list(axes.vector_to_gltf(t.position))
        if t.rotation != (0.0, 0.0, 0.0, 1.0):
            node["rotation"] = list(axes.rotation_to_gltf(t.rotation))
        if t.scale != (1.0, 1.0, 1.0):
            node["scale"] = list(axes.scale_to_gltf(t.scale))
        mesh_ref = None
        mesh = obj.mesh
        if mesh is not None:
            if id(mesh) not in encoded_meshes:
                encoded_meshes[id(mesh)] = _export_mesh(
                    mesh,
                    f"objects[{obj.id}].mesh",
                    buffers,
                    diagnostics,
                    tangents=id(mesh) in normal_mapped,
                )
            encoded = encoded_meshes[id(mesh)]
            if encoded is not None:
                skinned = obj.id in rigging.skins
                own = skinned or obj.id in rigging.morphs
                mesh_ref = gltf_mesh(obj, encoded, own)
                if skinned:  # carried by a root node of its own (see anim_export)
                    carried.append((obj, mesh_ref))
                else:
                    node["mesh"] = mesh_ref
                    rigging.mesh_node[obj.id] = len(nodes)
        if obj.children:
            node["children"] = [node_of[child.id] for child in obj.children]
        stored: dict[str, Any] = {"id": obj.id}
        if obj.tags:
            stored["tags"] = list(obj.tags)
        metadata = obj.metadata
        if metadata:
            stored["metadata"] = metadata
        node["extras"] = {EXTRAS_KEY: stored}
        rigging.object_node[obj.id] = len(nodes)
        nodes.append(node)
        node_map.append({"node": len(nodes) - 1, "id": obj.id, "name": obj.name, "mesh": mesh_ref})

    roots = [node_of[root.id] for root in scene.roots]
    for obj, mesh_ref in carried:
        rigging.carriers[obj.id] = len(nodes)
        rigging.mesh_node[obj.id] = len(nodes)
        roots.append(len(nodes))
        nodes.append(
            {
                "name": f"{obj.name} (skinned mesh)",
                "mesh": mesh_ref,
                "extras": {EXTRAS_KEY: {"skinned_mesh_of": obj.id}},
            }
        )
    anim_export.add_skeletons(scene, rigging, nodes, node_of, roots, EXTRAS_KEY)
    skins = anim_export.add_skins(scene, rigging, nodes, node_of, buffers)
    anim_export.rig_diagnostics(scene, rigging, diagnostics)
    node_parent = {c: p for p, n in enumerate(nodes) for c in n.get("children", [])}
    options = dict(animation_options or {})
    animations = anim_export.build_animations(
        scene, rigging, node_parent, buffers, diagnostics, mode=animation, **options
    )

    gltf_scene: dict[str, Any] = {"name": scene.name}
    if roots:
        gltf_scene["nodes"] = roots
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": f"Aether3D {__version__}"},
        "scene": 0,
        "scenes": [gltf_scene],
    }
    if nodes:
        document["nodes"] = nodes
    if meshes:
        document["meshes"] = meshes
        document.update(exporter.document_parts())
    if skins:
        document["skins"] = skins
    if animations:
        document["animations"] = animations
    if buffers.accessors:
        document["accessors"] = buffers.accessors
        document["bufferViews"] = buffers.views
        while len(buffers.data) % 4:
            buffers.data.append(0)
        document["buffers"] = [{"byteLength": len(buffers.data)}]
    if exporter.extensions:
        document["extensionsUsed"] = sorted(exporter.extensions)
    return document, bytes(buffers.data), diagnostics, node_map


def encode_glb(document: dict[str, Any], binary: bytes) -> bytes:
    """Pack a document and its buffer into a GLB container."""
    json_chunk = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    chunks = struct.pack("<II", len(json_chunk), CHUNK_JSON) + json_chunk
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        chunks += struct.pack("<II", len(binary), CHUNK_BIN) + binary
    return struct.pack("<III", GLB_MAGIC, GLB_VERSION, 12 + len(chunks)) + chunks


def encode_gltf(document: dict[str, Any], binary: bytes) -> bytes:
    """Encode a ``.gltf`` JSON file with the buffer embedded as a base64 data URI."""
    document = dict(document)
    if binary:
        buffer = dict(document["buffers"][0])
        buffer["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binary).decode(
            "ascii"
        )
        document["buffers"] = [buffer]
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def export_gltf(
    scene: Scene,
    path: str | os.PathLike[str],
    *,
    allow_lossy: bool = False,
    bake_textures: int | None = None,
    animation: str = "auto",
    fps: Any = None,
    position_tolerance: float = 1e-4,
    rotation_tolerance: float = 0.01,
    weight_tolerance: float = 1e-4,
) -> ExportResult:
    """Export ``scene`` to a ``.glb`` or ``.gltf`` file (chosen by extension).

    Objects become nodes (depth-first order, so node indices are
    deterministic); each object's id is stored in ``node.extras.aether3d.id``.
    Meshes shared by several objects are written once and instanced.
    Coordinates are converted from Aether3D's +Z up to glTF's +Y up (see
    :mod:`aether3d.gltf.axes`).

    If the scene contains something glTF cannot represent (a mesh without
    triangles, zero-length normals), :class:`UnsupportedFeatureError` is
    raised and nothing is written, unless ``allow_lossy=True``, in which case
    the content is omitted and reported in ``ExportResult.diagnostics``.
    Materials are exported as glTF materials (see :mod:`aether3d.gltf.materials`);
    what glTF cannot express -- procedural textures, textures mapped by object
    or world coordinates -- is left out with a warning, unless
    ``bake_textures`` (a size in pixels, 16-8192) asks to bake procedural
    textures into PNG images. The file is written atomically.

    Rigs, skins, morph targets and animations are exported too (see
    :mod:`aether3d.gltf.anim_export`): ``animation`` is ``"auto"``, ``"clips"``,
    ``"actions"``, ``"bake"`` (sampled at ``fps``, default the timeline's) or
    ``"none"``; fitted channels stay within ``position_tolerance`` (scene units),
    ``rotation_tolerance`` (degrees) and ``weight_tolerance``.
    """
    from . import anim_export

    fmt = output_format(path)
    anim_export.check_mode(animation)
    options = {
        "fps": fps,
        "position_tolerance": anim_export.tolerance_check("position_tolerance", position_tolerance),
        "rotation_tolerance": anim_export.tolerance_check("rotation_tolerance", rotation_tolerance),
        "weight_tolerance": anim_export.tolerance_check("weight_tolerance", weight_tolerance),
    }
    if bake_textures is not None and (
        isinstance(bake_textures, bool)
        or not isinstance(bake_textures, int)
        or not 16 <= bake_textures <= 8192
    ):
        raise ValidationError(
            f"bake_textures must be an image size in [16, 8192] pixels, got {bake_textures!r}"
        )
    document, binary, diagnostics, node_map = build_gltf(
        scene, bake_textures=bake_textures, animation=animation, animation_options=options
    )
    if scene._worlds is not None and len(scene._worlds):
        diagnostics.append(
            Diagnostic(
                "WORLDS_NOT_EXPORTED",
                WARNING,
                "procedural worlds are generated content, not scene objects, and are not "
                "exported with the scene; export a finite region with world.export_region "
                "(aether3d world export)",
                "scene.worlds",
                len(scene._worlds),
            )
        )
    lossy = [d for d in diagnostics if d.lossy]
    if lossy and not allow_lossy:
        raise UnsupportedFeatureError(
            "the scene contains content glTF cannot represent: "
            + ", ".join(sorted({d.code for d in lossy}))
            + "; pass allow_lossy=True to export without it",
            lossy,
        )
    data = encode_glb(document, binary) if fmt == "glb" else encode_gltf(document, binary)
    written = write_atomic(path, data)
    return ExportResult(
        path=written,
        format=fmt,
        byte_length=len(data),
        node_count=len(node_map),
        mesh_count=len(document.get("meshes", [])),
        nodes=node_map,
        diagnostics=diagnostics,
        animations=[
            {
                "name": a["name"],
                "channels": len(a["channels"]),
                **a.get("extras", {}).get("aether3d", {}),
            }
            for a in document.get("animations", [])
        ],
        skins=len(document.get("skins", [])),
    )
