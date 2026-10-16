"""Exact comparison helpers for parity tests: old vs new implementations, cached vs fresh.

Parity in Aether3D means *identical*, not close: the same bytes in every array (NaN
payloads included), the same ids, names, order and diagnostics. These helpers turn
render scenes and query results into canonical, hashable fingerprints and report the
first difference, so a test failure says where two paths diverge.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def array_bytes(a: Any) -> tuple | None:
    """``(dtype, shape, bytes)`` of an array (``None`` stays ``None``): equal iff bitwise equal."""
    if a is None:
        return None
    arr = np.ascontiguousarray(np.asarray(a))
    return (arr.dtype.str, arr.shape, arr.tobytes())


def assert_bitwise(a: Any, b: Any, what: str = "arrays") -> None:
    """``a`` and ``b`` have the same dtype, shape and bytes."""
    fa, fb = array_bytes(a), array_bytes(b)
    if fa != fb:
        detail = ""
        if fa is not None and fb is not None and fa[:2] == fb[:2]:
            x, y = np.asarray(a), np.asarray(b)
            diff = np.argwhere(
                x.view(np.uint8).reshape(x.size, -1) != y.view(np.uint8).reshape(y.size, -1)
            )
            if diff.size:
                k = int(diff[0, 0])
                left, right = x.reshape(-1)[k], y.reshape(-1)[k]
                detail = f"; first difference at flat index {k}: {left!r} vs {right!r}"
        else:
            detail = f"; {None if fa is None else fa[:2]} vs {None if fb is None else fb[:2]}"
        raise AssertionError(f"{what} differ{detail}")


def _diagnostics(items) -> list:
    return [(d.code, d.severity, d.message, d.location) for d in items]


def _texture(t) -> tuple:
    image = t.image
    return (
        t.slot,
        t.kind,
        t.texture_id,
        t.name,
        t.channel,
        t.coordinates,
        t.projection,
        array_bytes(t.transform),
        t.blend,
        None
        if image is None
        else (image.name, image.digest, image.chain.mode, image.wrap, image.filter, image.mipmaps),
        None if t.graph is None else json.dumps(t.graph.to_dict(), sort_keys=True, default=repr),
        tuple(sorted((key, img.digest) for key, img in t.images.items())),
    )


def _instance_arrays(rs) -> tuple:
    a = rs.instance_arrays
    return tuple(
        array_bytes(getattr(a, f))
        for f in (
            "world",
            "normal_matrix",
            "mirrored",
            "mesh",
            "base_color",
            "specular",
            "shininess",
            "material",
            "has_groups",
        )
    )


def render_scene_fingerprint(rs) -> dict[str, Any]:
    """Everything a backend reads from a ``RenderScene``, as comparable plain data."""
    camera = rs.camera
    frame = camera.frame
    return {
        "meshes": [
            tuple(
                array_bytes(getattr(m, f))
                for f in ("positions", "triangles", "normals", "uvs", "groups")
            )
            for m in rs.meshes
        ],
        "instances": [
            (
                i.object_id,
                i.name,
                i.mesh,
                array_bytes(i.world),
                array_bytes(i.normal_matrix),
                i.mirrored,
                array_bytes(i.base_color),
                i.specular,
                i.shininess,
                i.material,
                i.group_materials,
            )
            for i in rs.instances
        ],
        "lights": [
            (
                light.object_id,
                light.name,
                light.type,
                array_bytes(light.radiance),
                array_bytes(light.position),
                array_bytes(light.direction),
                light.range,
                light.falloff,
                light.radius,
                light.angle,
            )
            for light in rs.lights
        ],
        "camera": (
            camera.object_id,
            camera.name,
            json.dumps(camera.camera.to_dict(), sort_keys=True),
            tuple(array_bytes(getattr(frame, f)) for f in ("position", "forward", "up", "right")),
        ),
        "background": array_bytes(rs.background),
        "ambient": array_bytes(rs.ambient),
        "diagnostics": _diagnostics(rs.diagnostics),
        "materials": [
            (
                m.key,
                m.name,
                m.material_id,
                m.model,
                array_bytes(m.base_color),
                m.opacity,
                m.alpha_mode,
                m.alpha_cutoff,
                m.metallic,
                m.roughness,
                m.specular,
                m.shininess,
                array_bytes(m.emissive),
                m.normal_strength,
                m.occlusion_strength,
                m.double_sided,
                tuple(_texture(m.textures[slot]) for slot in sorted(m.textures)),
            )
            for m in rs.materials
        ],
        "instance_arrays": _instance_arrays(rs),
        "environment": None
        if rs.environment is None
        else json.dumps(rs.environment.to_dict(), sort_keys=True),
    }


def assert_render_scenes_equal(a, b, what: str = "render scenes") -> None:
    """Two ``RenderScene``s give a backend exactly the same input."""
    fa, fb = render_scene_fingerprint(a), render_scene_fingerprint(b)
    for key in fa:
        if fa[key] != fb[key]:
            left, right = fa[key], fb[key]
            if isinstance(left, list) and isinstance(right, list):
                if len(left) != len(right):
                    raise AssertionError(
                        f"{what}: {key} counts differ ({len(left)} vs {len(right)})"
                    )
                k = next(k for k, (x, y) in enumerate(zip(left, right, strict=True)) if x != y)
                raise AssertionError(
                    f"{what}: {key}[{k}] differ:\n  {left[k]!r:.400}\n  {right[k]!r:.400}"
                )
            raise AssertionError(f"{what}: {key} differ")


def _without(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {k: _without(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_without(v, keys) for v in value]
    return value


def query_fingerprint(
    scene, requests: list[dict[str, Any]], ignore: tuple[str, ...] = ()
) -> list[str]:
    """The JSON answer (or the error code) of every spatial query request.

    ``ignore`` drops keys (at any depth) that legitimately differ between scenes with the
    same content -- ``"revision"`` when comparing with a copy rebuilt from the document.
    """
    from aether3d.errors import AetherError
    from aether3d.spatial.request import run_query

    out = []
    for request in requests:
        try:
            answer = _without(run_query(scene, request), ignore)
            out.append(json.dumps(answer, sort_keys=True))
        except AetherError as exc:
            out.append(f"error:{getattr(exc, 'code', type(exc).__name__)}")
    return out
