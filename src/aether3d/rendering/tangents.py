"""MikkTSpace tangents for rendering: per-corner, object space, cached per mesh.

UV normal maps are interpreted in the MikkTSpace tangent basis (``docs/TEXTURING.md``,
"Tangent space"), the basis glTF 2.0 specifies and that baking tools -- Aether3D's own
baker included -- write normal maps for. A renderer reads, per shading point:

* ``N``: the interpolated vertex normal (not renormalized; the face normal on meshes
  without normals), ``T``: the interpolated corner tangent (not renormalized), and the
  bitangent ``B = sign * cross(N, T)`` (``sign``: the tangent's ``w``, negated on mirrored
  instances);
* the mapped normal ``normalize(x T + y B + z N)`` for the map's ``(x, y, z)``.

This is the pixel-shader convention stated in the reference ``mikktspace.h``; the baker
uses its exact inverse. Tangents are computed from the render mesh's own arrays
(positions, normals or flat face normals, UVs) by :func:`aether3d.uv.mikktspace.mikktspace`
and cached by the identity of those read-only arrays.
"""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

from ..uv.mikktspace import face_normals, mikktspace

_CACHE: dict[tuple[int, ...], np.ndarray] = {}


def corner_tangents(mesh: Any) -> np.ndarray | None:
    """``(T, 3, 4)`` float32 MikkTSpace tangents of a render mesh (``None`` without UVs)."""
    uvs = mesh.uvs
    if uvs is None:
        return None
    arrays = [mesh.positions, mesh.triangles, mesh.normals, uvs]
    live = [a for a in arrays if a is not None]
    cacheable = not any(a.flags.writeable for a in live)
    key = tuple(id(a) if a is not None else 0 for a in arrays)
    if cacheable:
        found = _CACHE.get(key)
        if found is not None:
            return found
    t = mesh.triangles.astype(np.int64)
    p = mesh.positions[t]
    if mesh.normals is not None:
        n = mesh.normals[t]
    else:
        n = np.repeat(face_normals(mesh.positions, t)[:, None, :], 3, axis=1)
    out = mikktspace(p, n, uvs[t])
    out.flags.writeable = False
    if cacheable:
        _CACHE[key] = out
        weakref.finalize(mesh.triangles, _CACHE.pop, key, None)
    return out


def uv_aligned(transform: np.ndarray) -> bool:
    """Whether a UV binding's transform keeps the tangent basis: a positive uniform scale
    (with any offset). Rotated, mirrored or non-uniformly scaled bindings fall back to each
    triangle's UV-gradient frame."""
    m = np.asarray(transform, dtype=np.float64)
    a, b, c, d = m[0, 0], m[0, 1], m[1, 0], m[1, 1]
    return bool(a > 0 and a == d and b == 0 and c == 0)


def meshes_needing_tangents(rs: Any) -> set[int]:
    """Indices of ``rs.meshes`` drawn with a material whose normal map is read in the
    MikkTSpace basis (and that have UVs)."""
    mikk = [
        bool(m.textures.get("normal") is not None and m.textures["normal"].mikktspace)
        for m in rs.materials
    ]
    out: set[int] = set()
    if not any(mikk):
        return out
    for inst in rs.instances:
        if rs.meshes[inst.mesh].uvs is None:
            continue
        used = [inst.material, *(index for _, index in inst.group_materials)]
        if any(mikk[i] for i in used):
            out.add(inst.mesh)
    return out
