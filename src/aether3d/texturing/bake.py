"""Baking: transferring surface detail from high-resolution sources into a mesh's texture.

``bake(scene, target, sources, maps, ...)`` fills texture maps of ``target`` (an object
with UVs) from the surfaces of ``sources`` (its own surface by default):

1. **Texels.** The target's UV triangles are rasterized exactly (:mod:`.raster`) with
   ``samples x samples`` stratified samples per texel; each sample stands for a point
   ``P`` of the target's surface (world space), with its interpolated normal ``N`` and
   MikkTSpace tangent frame.
2. **Projection.** A ray starts on the **cage** -- ``P`` pushed out by ``cage`` along the
   target's *averaged* normal (the vertex normals averaged over equal positions, so hard
   edges do not open gaps in the cage) or, with ``cage_mesh``, the matching point of an
   explicit cage of the same topology -- and runs back through ``P`` until ``cage +
   depth`` past it. Its first hit on the sources is the surface point the sample takes its
   values from. A ray that hits nothing is a **miss**: reported, and filled from the
   neighbouring texels only by the padding.
3. **Maps** (what each sample computes at the hit ``H``, with the source's interpolated
   normal ``n``):

   * ``normal`` -- ``n`` in the target's tangent space, the exact inverse of the
     renderers' MikkTSpace transform (:mod:`aether3d.rendering.tangents`), so the map
     reproduces ``n`` when rendered on the target; encoded ``n * 0.5 + 0.5``;
   * ``object_normal`` / ``world_normal`` -- ``n`` in the target's object space or world
     space, encoded the same way;
   * ``height`` -- the signed distance from ``P`` to ``H`` along the outward cage
     direction, mapped to ``0.5 + 0.5 * d / height_range`` (clamped);
   * ``ao`` -- ambient occlusion: the cosine-weighted fraction of ``ao_rays`` directions of
     the hemisphere around ``n`` along which nothing is hit within ``ao_distance``
     (stratified, reproducible with ``seed``);
   * ``thickness`` -- the mean distance to the next surface along ``thickness_rays``
     cosine-weighted directions *into* the source (misses count as ``thickness_distance``),
     divided by ``thickness_distance``: 1 is thick, 0 paper-thin;
   * ``curvature`` -- the source's mean curvature at ``H`` (the cotangent Laplacian of its
     welded vertices, Meyer et al. 2003, interpolated), signed positive on convex
     surfaces, mapped to ``0.5 + 0.5 * k * curvature_radius`` (clamped): a sphere of radius
     ``curvature_radius`` maps to 1;
   * ``position`` -- ``H`` in the target's object space, normalized to its box;
   * ``base_color``, ``roughness``, ``metallic``, ``emissive``, ``opacity`` -- the source's
     own material evaluated at ``H`` (textures included);
   * ``material_id``, ``object_id``, ``group_id`` -- a distinct color per source material,
     object or face group (a hash of its id).
4. **Texels** average their covered samples (unit vectors are averaged and renormalized;
   colors are averaged in linear light and stored as display sRGB); uncovered texels get
   ``padding`` rings of dilation.

Everything is deterministic: the same inputs give the same bits on every run with the same
ray backend and device (CPU and GPU traversals differ by float32 rounding).
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import TexturingError
from .dilate import dilate
from .raster import rasterize_uv

MAPS: dict[str, str] = {
    "normal": "normal",
    "object_normal": "normal",
    "world_normal": "normal",
    "height": "data",
    "ao": "data",
    "thickness": "data",
    "curvature": "data",
    "position": "data",
    "base_color": "color",
    "roughness": "data",
    "metallic": "data",
    "emissive": "color",
    "opacity": "data",
    "material_id": "color",
    "object_id": "color",
    "group_id": "color",
}
"""Every bake map and the encoding of its texture (``"normal"``, ``"color"``, ``"data"``)."""

MAX_SAMPLES = 1 << 26
"""Most texel samples one bake may take (``resolution^2 x samples^2``)."""
MAX_RAYS = 1 << 32
"""Most rays one bake may trace (samples x (1 + AO and thickness rays))."""
CHUNK = 1 << 17
"""Samples traced per batch (bounds memory; progress and cancellation between batches)."""


@dataclass
class BakeResult:
    """Baked maps (``(height, width, 4)`` float64 in their texture encoding, alpha 1 where
    a sample hit), coverage and a report."""

    maps: dict[str, np.ndarray]
    covered: np.ndarray  # (H, W) fraction of a texel's samples on the target's UVs
    hit: np.ndarray  # (H, W) fraction of its samples whose ray hit a source
    report: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_texture(self, scene: Any, map_name: str, name: str | None = None, **sampler: Any):
        """A painted texture holding the map (one paint layer, the right encoding)."""
        if map_name not in self.maps:
            raise TexturingError("INVALID_ARGUMENT", f"no baked map {map_name!r}", "map")
        encoding = MAPS[map_name]
        return scene.materials.add_painted_texture(
            name,
            image=self.maps[map_name],
            encoding=encoding,
            background=(0.5, 0.5, 1.0, 1.0) if encoding == "normal" else (0.0, 0.0, 0.0, 1.0),
            **sampler,
        )

    def save_png(self, map_name: str, path: Any, *, bits: int = 8) -> Any:
        from pathlib import Path

        from .._fileio import write_atomic
        from ..rendering.png import encode_png

        values = self.maps[map_name]
        top = 65535 if bits == 16 else 255
        pixels = np.round(np.clip(values, 0.0, 1.0) * top).astype(
            np.uint16 if bits == 16 else np.uint8
        )
        target = Path(path).absolute()
        write_atomic(target, encode_png(pixels, srgb=MAPS[map_name] == "color"))
        return target


def _bad(message: str, location: str) -> TexturingError:
    return TexturingError("INVALID_ARGUMENT", message, location)


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _bad(f"{name} must be a finite number", name)
    if value < 0 or (value == 0 and not allow_zero):
        raise _bad(f"{name} must be positive", name)
    return float(value)


def _count(value: Any, name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise _bad(f"{name} must be an integer in [{lo}, {hi}]", name)
    return int(value)


# --------------------------------------------------------------------------- #
# Sources: a render scene of the source objects
# --------------------------------------------------------------------------- #


def source_scene(scene: Any, objects: list[Any]):
    """A :class:`~aether3d.rendering.pathtrace.scene.PathScene` of ``objects`` (their meshes
    in world space, their materials), for ray queries and material evaluation."""
    from ..rendering.camera import Camera, frame_from_matrix
    from ..rendering.color import to_linear
    from ..rendering.materials import resolve_materials
    from ..rendering.pathtrace.scene import PathScene
    from ..rendering.scene_data import RenderCamera, RenderInstance, RenderMesh, RenderScene

    candidates = [(obj, obj.mesh, k) for k, obj in enumerate(objects)]
    table, per_object = resolve_materials(scene, candidates)
    meshes: list[RenderMesh] = []
    index: dict[int, int] = {}
    instances = []
    for (obj, mesh, _), (default, groups) in zip(candidates, per_object, strict=True):
        key = id(mesh)
        if key not in index:
            index[key] = len(meshes)
            meshes.append(
                RenderMesh(mesh.positions, mesh.triangles, mesh.normals, mesh.uvs, mesh.groups)
            )
        world = np.asarray(obj.world_matrix, dtype=np.float64)
        lin = world[:3, :3]
        det = float(np.linalg.det(lin))
        if not math.isfinite(det) or abs(det) < 1e-300:
            raise TexturingError(
                "INVALID_SOURCE",
                f"object {obj.name!r} has a singular transform",
                f"objects[{obj.name}]",
            )
        look = obj.appearance
        instances.append(
            RenderInstance(
                obj.id,
                obj.name,
                index[key],
                world,
                np.linalg.inv(lin).T,
                det < 0,
                to_linear(look.color),
                look.specular,
                look.shininess,
                default,
                tuple(sorted(groups.items())),
            )
        )
    frame, _ = frame_from_matrix(np.eye(4))
    camera = RenderCamera(None, "bake", Camera(), frame)
    rs = RenderScene(
        tuple(meshes),
        tuple(instances),
        (),
        camera,
        np.zeros(3),
        np.zeros(3),
        materials=tuple(table.materials),
    )
    return PathScene(rs)


# --------------------------------------------------------------------------- #
# The target: its surface per texel sample
# --------------------------------------------------------------------------- #


class _Target:
    """World-space data of the target mesh per corner (positions, normals, tangents,
    averaged cage normals)."""

    def __init__(self, obj: Any, cage_mesh: Any = None) -> None:
        from ..rendering.scene_data import RenderMesh
        from ..rendering.tangents import corner_tangents
        from ..uv.topology import weld_positions

        mesh = obj.mesh
        self.mesh = mesh
        world = np.asarray(obj.world_matrix, dtype=np.float64)
        self.world = world
        lin = world[:3, :3]
        self.lin = lin
        self.mirrored = float(np.linalg.det(lin)) < 0
        tri = mesh.triangles.astype(np.int64)
        self.tri = tri
        p = mesh.positions.astype(np.float64) @ lin.T + world[:3, 3]
        self.p = p  # (V, 3) world
        corners = p[tri]
        g = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        if self.mirrored:
            g = -g
        area = np.linalg.norm(g, axis=1)
        face = np.divide(g, area[:, None], out=np.zeros_like(g), where=area[:, None] > 0)
        normal_matrix = np.linalg.inv(lin).T
        if mesh.normals is not None:
            n = mesh.normals.astype(np.float64) @ normal_matrix.T
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
            self.cn = n[tri]  # (T, 3, 3) corner normals, unit, outward
        else:
            self.cn = np.repeat(face[:, None, :], 3, axis=1)
        # Averaged normals for the cage: area-weighted face normals summed over welded
        # positions (smooth across hard edges, so the cage has no gaps).
        weld, count = weld_positions(mesh.positions)
        acc = np.zeros((count, 3))
        for k in range(3):
            np.add.at(acc, weld[tri[:, k]], g)
        acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-300)
        self.cage_n = acc[weld][tri]  # (T, 3, 3)
        self.cage_p = None
        if cage_mesh is not None:
            cp = cage_mesh.positions.astype(np.float64) @ lin.T + world[:3, 3]
            self.cage_p = cp[cage_mesh.triangles.astype(np.int64)]
        t = corner_tangents(
            RenderMesh(mesh.positions, mesh.triangles, mesh.normals, mesh.uvs, mesh.groups)
        )
        v = t[..., :3].astype(np.float64) @ lin.T
        v /= np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-300)
        self.ct = v
        self.cs = np.where(t[..., 3] < 0, -1.0, 1.0) * (-1.0 if self.mirrored else 1.0)
        self.inv_world = np.linalg.inv(world)

    def frame(self, tri: np.ndarray, b: np.ndarray):
        """Interpolated position, normal (unnormalized), tangent, sign, cage normal."""
        p = np.einsum("nk,nkc->nc", b, self.p[self.tri[tri]])
        n = np.einsum("nk,nkc->nc", b, self.cn[tri])
        t = np.einsum("nk,nkc->nc", b, self.ct[tri])
        s = np.where((b * self.cs[tri]).sum(axis=1) < 0.0, -1.0, 1.0)
        c = np.einsum("nk,nkc->nc", b, self.cage_n[tri])
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-300)
        return p, n, t, s, c


def tangent_space(
    n_world: np.ndarray, t: np.ndarray, n: np.ndarray, sign: np.ndarray
) -> np.ndarray:
    """``n_world`` in the MikkTSpace frame ``(T, B = sign cross(N, T), N)``: the exact inverse
    of the renderers' ``normalize(x T + y B + z N)`` (reference ``mikktspace.h``)."""
    b = sign[:, None] * np.cross(n, t)
    row0 = np.cross(b, n)
    row1 = np.cross(n, t)
    row2 = np.cross(t, b)
    f = np.where((t * row0).sum(axis=1) < 0.0, -1.0, 1.0)[:, None]
    v = f * np.stack([(n_world * row0).sum(1), (n_world * row1).sum(1), (n_world * row2).sum(1)], 1)
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(length > 1e-300, v / np.maximum(length, 1e-300), np.array([0.0, 0.0, 1.0]))


# --------------------------------------------------------------------------- #
# Curvature of the sources
# --------------------------------------------------------------------------- #


def vertex_mean_curvature(positions: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Signed mean curvature at every vertex (positive on convex surfaces): ``-(L x) . n /
    2`` with the cotangent Laplacian ``L`` over mixed Voronoi areas (Meyer et al. 2003),
    computed on positions welded by equality (UV and normal seams do not split it)."""
    from ..uv.topology import weld_positions

    weld, count = weld_positions(positions)
    wp = np.zeros((count, 3))
    wp[weld] = positions.astype(np.float64)
    tri = weld[triangles.astype(np.int64)]
    lap = np.zeros((count, 3))
    area = np.zeros(count)
    normal = np.zeros((count, 3))
    p = wp[tri]
    g = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    tri_area = 0.5 * np.linalg.norm(g, axis=1)
    for k in range(3):
        np.add.at(normal, tri[:, k], g)
    for k in range(3):
        i, j, o = tri[:, k], tri[:, (k + 1) % 3], tri[:, (k + 2) % 3]
        # Edge (i, j) is opposite corner o: weight cot(angle at o).
        u = wp[i] - wp[o]
        v = wp[j] - wp[o]
        cross = np.linalg.norm(np.cross(u, v), axis=1)
        cot = np.divide((u * v).sum(axis=1), cross, out=np.zeros_like(cross), where=cross > 1e-300)
        d = wp[j] - wp[i]
        np.add.at(lap, i, cot[:, None] * d)
        np.add.at(lap, j, -cot[:, None] * d)
    # Mixed areas: Voronoi for non-obtuse triangles, else area/2 at the obtuse corner and
    # area/4 at the others.
    for k in range(3):
        i, j, o = tri[:, k], tri[:, (k + 1) % 3], tri[:, (k + 2) % 3]
        a = wp[j] - wp[i]
        b = wp[o] - wp[i]
        cos_i = (a * b).sum(1)
        obtuse_i = cos_i < 0
        obtuse_other = np.zeros_like(obtuse_i)
        for m in ((j, i, o), (o, i, j)):
            x, y, z = m
            obtuse_other |= ((wp[y] - wp[x]) * (wp[z] - wp[x])).sum(1) < 0

        # Voronoi part at i: (|e_ij|^2 cot(o) + |e_io|^2 cot(j)) / 8.
        def cot_at(c, e1, e2):
            u_ = wp[e1] - wp[c]
            v_ = wp[e2] - wp[c]
            cr = np.linalg.norm(np.cross(u_, v_), axis=1)
            return np.divide((u_ * v_).sum(1), cr, out=np.zeros_like(cr), where=cr > 1e-300)

        vor = ((a * a).sum(1) * cot_at(o, i, j) + (b * b).sum(1) * cot_at(j, i, o)) / 8.0
        mixed = np.where(obtuse_i, tri_area / 2.0, np.where(obtuse_other, tri_area / 4.0, vor))
        np.add.at(area, i, mixed)
    lap = 0.5 * lap  # (1/2) sum (cot a + cot b)(x_j - x_i): the cotangent Laplacian
    n_len = np.linalg.norm(normal, axis=1, keepdims=True)
    unit = np.divide(normal, n_len, out=np.zeros_like(normal), where=n_len > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_mean = -(lap * unit).sum(axis=1) / np.where(area > 0, area, np.inf) / 2.0
    return np.nan_to_num(k_mean)[weld]


# --------------------------------------------------------------------------- #
# Sampling directions
# --------------------------------------------------------------------------- #


def _radical_inverse(i: np.ndarray) -> np.ndarray:
    """Van der Corput radical inverse in base 2 of ``i`` (uint32)."""
    x = i.astype(np.uint64) & 0xFFFFFFFF
    x = ((x << 16) | (x >> 16)) & 0xFFFFFFFF
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1)
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2)
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4)
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8)
    return x.astype(np.float64) * (1.0 / 4294967296.0)


def hemisphere_shifts(keys: np.ndarray, slot: int) -> np.ndarray:
    """``(n, 2)`` random Cranley-Patterson shifts of each sample's direction set."""
    from ..rendering.pathtrace.sampling import uniform

    return np.stack([uniform(keys, slot), uniform(keys, slot + 1)], axis=1)


def hemisphere_rays(normal: np.ndarray, shifts: np.ndarray, count: int) -> np.ndarray:
    """``(n, count, 3)`` cosine-weighted directions around ``normal``: a Hammersley set of
    ``count`` points rotated by each sample's random shift (Cranley-Patterson), so every
    sample gets a stratified, reproducible set (``wgsl/bake.wgsl`` generates the same)."""
    from ..rendering.pathtrace.sampling import cosine_hemisphere, frame

    n = normal.shape[0]
    i = np.arange(count)
    base1 = (i + 0.5) / count
    base2 = _radical_inverse(i)
    s1 = shifts[:, 0]
    s2 = shifts[:, 1]
    u1 = np.mod(base1[None, :] + s1[:, None], 1.0)
    u2 = np.mod(base2[None, :] + s2[:, None], 1.0)
    local = cosine_hemisphere(u1.reshape(-1), u2.reshape(-1)).reshape(n, count, 3)
    t, b = frame(normal)
    return (
        t[:, None, :] * local[..., 0:1]
        + b[:, None, :] * local[..., 1:2]
        + normal[:, None, :] * local[..., 2:3]
    )


def id_color(value: str) -> np.ndarray:
    """A distinct, reproducible display color for an id (hue from its SHA-256)."""
    h = hashlib.sha256(value.encode("utf-8")).digest()
    hue = int.from_bytes(h[:4], "little") / 2**32
    sat = 0.55 + 0.4 * h[4] / 255
    val = 0.65 + 0.3 * h[5] / 255
    i = int(hue * 6) % 6
    f = hue * 6 - int(hue * 6)
    p, q, t = val * (1 - sat), val * (1 - f * sat), val * (1 - (1 - f) * sat)
    rgb = [(val, t, p), (q, val, p), (p, val, t), (p, q, val), (t, p, val), (val, p, q)][i]
    return np.array(rgb)


# --------------------------------------------------------------------------- #
# The bake
# --------------------------------------------------------------------------- #


def bake(
    scene: Any,
    target: Any,
    sources: Any = None,
    maps: Any = ("normal",),
    *,
    resolution: Any = 1024,
    samples: int = 2,
    cage: float | None = None,
    depth: float | None = None,
    cage_mesh: Any = None,
    padding: int = 8,
    ao_rays: int = 64,
    ao_distance: float | None = None,
    thickness_rays: int = 32,
    thickness_distance: float | None = None,
    curvature_radius: float | None = None,
    height_range: float | None = None,
    seed: int = 0,
    backend: str = "auto",
    adapter: Any = None,
    should_stop=None,
    progress=None,
) -> BakeResult:
    """Bake ``maps`` of ``target`` from ``sources`` (see the module docstring).

    ``cage`` and ``depth`` (world units; default 2% of the sources' and target's bounding
    diagonal each) are the ray distances outside and inside the target's surface;
    ``ao_distance`` and ``thickness_distance`` default to 10% of that diagonal,
    ``curvature_radius`` to 2% of it and ``height_range`` to ``max(cage, depth)``.
    ``backend``: ``"cpu"``, ``"gpu"`` or ``"auto"`` (the GPU when one works, for the rays
    only: :mod:`.gpu_bake`; results agree to float32 precision, and AO and thickness may
    differ by the rare ray whose hit float32 rounding flips); ``adapter`` selects the GPU
    adapter as for ``scene.render``.
    """
    from ..uv.unwrap import _check_resolution

    t_start = time.perf_counter()
    target_obj = _object(scene, target, "target")
    mesh = target_obj.mesh
    if mesh is None or mesh.triangle_count == 0:
        raise TexturingError("NOT_A_MESH_OBJECT", f"{target_obj.name!r} has no triangles", "target")
    if mesh.uvs is None:
        raise TexturingError(
            "NO_UVS",
            f"{target_obj.name!r} has no UVs to bake into",
            "target",
            hint="unwrap it first (obj.unwrap())",
        )
    if sources is None:
        source_objs = [target_obj]
    else:
        items = sources if isinstance(sources, (list, tuple)) else [sources]
        source_objs = [_object(scene, s, "sources") for s in items]
        if not source_objs:
            raise _bad("sources must name at least one object", "sources")
    for obj in source_objs:
        if obj.mesh is None or obj.mesh.triangle_count == 0:
            raise TexturingError("NOT_A_MESH_OBJECT", f"{obj.name!r} has no triangles", "sources")
    names = [maps] if isinstance(maps, str) else list(maps)
    if not names:
        raise _bad("maps must name at least one map", "maps")
    for m in names:
        if m not in MAPS:
            raise _bad(f"unknown map {m!r} (maps: {', '.join(MAPS)})", "maps")
    names = list(dict.fromkeys(names))
    width, height = _check_resolution(resolution)
    n = _count(samples, "samples", 1, 8)
    padding = _count(padding, "padding", 0, 256)
    seed = _count(seed, "seed", 0, 2**63 - 1)
    if backend not in ("auto", "cpu", "gpu"):
        raise _bad("backend must be 'auto', 'cpu' or 'gpu'", "backend")
    if width * height * n * n > MAX_SAMPLES:
        raise TexturingError(
            "BAKE_TOO_LARGE",
            f"{width}x{height} texels x {n * n} samples exceeds {MAX_SAMPLES:,} samples",
            "resolution",
            hint="lower the resolution or the samples",
        )
    if cage_mesh is not None:
        from ..mesh import Mesh

        if not isinstance(cage_mesh, Mesh) or cage_mesh.triangle_count != mesh.triangle_count:
            raise _bad("cage_mesh must be a Mesh with the target's triangles", "cage_mesh")
    t0 = time.perf_counter()
    tgt = _Target(target_obj, cage_mesh)
    ps = source_scene(scene, source_objs)
    lo = np.minimum(tgt.p.min(axis=0), ps.P.min(axis=0)) if ps.P.size else tgt.p.min(axis=0)
    hi = np.maximum(tgt.p.max(axis=0), ps.P.max(axis=0)) if ps.P.size else tgt.p.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    cage = 0.02 * diag if cage is None else _positive(cage, "cage", allow_zero=True)
    depth = 0.02 * diag if depth is None else _positive(depth, "depth", allow_zero=True)
    if cage + depth <= 0:
        raise _bad("cage + depth must be positive", "depth")
    ao_rays = _count(ao_rays, "ao_rays", 1, 1024)
    thickness_rays = _count(thickness_rays, "thickness_rays", 1, 1024)
    ao_distance = 0.1 * diag if ao_distance is None else _positive(ao_distance, "ao_distance")
    thickness_distance = (
        0.1 * diag
        if thickness_distance is None
        else _positive(thickness_distance, "thickness_distance")
    )
    curvature_radius = (
        0.02 * diag if curvature_radius is None else _positive(curvature_radius, "curvature_radius")
    )
    height_range = (
        max(cage, depth) if height_range is None else _positive(height_range, "height_range")
    )
    extra = (ao_rays if "ao" in names else 0) + (thickness_rays if "thickness" in names else 0)
    if width * height * n * n * (1 + extra) > MAX_RAYS:
        raise TexturingError(
            "BAKE_TOO_LARGE",
            f"the bake would trace more than {MAX_RAYS:,} rays",
            "ao_rays",
            hint="lower the resolution, samples or ray counts",
        )
    tracer = _tracer(ps, backend, adapter, hemisphere=bool({"ao", "thickness"} & set(names)))
    t_setup = time.perf_counter() - t0
    # Texel samples.
    t0 = time.perf_counter()
    uv = mesh.uvs.astype(np.float64)[tgt.tri]
    cov = rasterize_uv(uv, width, height, n)
    t_raster = time.perf_counter() - t0
    texel = cov.texel
    total = texel.shape[0]
    curvature = None
    if "curvature" in names:
        curvature = [vertex_mean_curvature(m.positions, m.triangles) for m in ps.rs.meshes]
    channels = {name: np.zeros((height * width, 4)) for name in names}
    weight = np.zeros(height * width)
    covered = np.bincount(texel, minlength=height * width).astype(np.float64)
    hits_total = 0
    t_trace = 0.0
    t_shade = 0.0
    for start in range(0, total, CHUNK):
        if should_stop is not None and should_stop():
            raise TexturingError("CANCELLED", "the bake was cancelled")
        sl = slice(start, min(start + CHUNK, total))
        tri, b, tex = cov.triangle[sl], cov.bary[sl], texel[sl]
        p, nrm, tan, sign, cage_n = tgt.frame(tri, b)
        if tgt.cage_p is not None:
            origin = np.einsum("nk,nkc->nc", b, tgt.cage_p[tri])
            d = p - origin
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            direction = np.divide(d, dist, out=-cage_n.copy(), where=dist > 1e-300)
            tmax = dist[:, 0] + depth
        else:
            origin = p + cage * cage_n
            direction = -cage_n
            tmax = np.full(p.shape[0], cage + depth)
        ta = time.perf_counter()
        hits = tracer.closest(origin, direction, tmax)
        t_trace += time.perf_counter() - ta
        ok = hits.inst >= 0
        rows = np.flatnonzero(ok)
        hits_total += int(rows.size)
        if rows.size == 0:
            continue
        ta = time.perf_counter()
        sub = type(hits)(hits.t[rows], hits.inst[rows], hits.prim[rows], hits.b[rows])
        it = ps.interaction(rows, sub, origin[rows], direction[rows])
        n_side = it.ns  # the source's outward shading normal
        shading = None
        needs_shading = {
            "normal",
            "object_normal",
            "world_normal",
            "base_color",
            "roughness",
            "metallic",
            "emissive",
            "opacity",
        } & set(names)
        if needs_shading:
            shading = ps.shade(it)
        keys = None
        if "ao" in names or "thickness" in names:
            from ..rendering.pathtrace.sampling import path_keys

            keys = path_keys(seed, tex[rows], start + rows)
        values: dict[str, np.ndarray] = {}
        for name in names:
            if name in ("normal", "object_normal", "world_normal"):
                nw = shading.normal  # outward, with the source's own normal map
                if name == "normal":
                    v = tangent_space(nw, tan[rows], nrm[rows], sign[rows])
                elif name == "object_normal":
                    v = nw @ tgt.lin  # (M^-T)^-1 = M^T: world normals to object space
                    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-300)
                else:
                    v = nw
                values[name] = np.concatenate([v, np.ones((v.shape[0], 1))], axis=1)
            elif name == "height":
                travelled = hits.t[rows] - (tmax[rows] - depth)
                h = 0.5 - 0.5 * np.clip(travelled / height_range, -1.0, 1.0)
                values[name] = _gray(h)
            elif name == "position":
                local = (it.p - tgt.world[:3, 3]) @ np.linalg.inv(tgt.lin).T
                mlo, mhi = mesh.bounds()
                span = np.maximum(np.asarray(mhi) - np.asarray(mlo), 1e-300)
                values[name] = np.concatenate(
                    [np.clip((local - np.asarray(mlo)) / span, 0, 1), np.ones((local.shape[0], 1))],
                    1,
                )
            elif name == "ao":
                values[name] = _gray(_ao(tracer, ps, it, n_side, keys, ao_rays, ao_distance))
            elif name == "thickness":
                values[name] = _gray(
                    _thickness(tracer, ps, it, n_side, keys, thickness_rays, thickness_distance)
                )
            elif name == "curvature":
                k = _curvature_at(ps, it, curvature)
                values[name] = _gray(0.5 + 0.5 * np.clip(k * curvature_radius, -1.0, 1.0))
            elif name == "base_color":
                values[name] = np.concatenate([shading.base, np.ones((rows.size, 1))], 1)
            elif name == "emissive":
                values[name] = np.concatenate([shading.emissive, np.ones((rows.size, 1))], 1)
            elif name in ("roughness", "metallic"):
                values[name] = _gray(getattr(shading, name))
            elif name == "opacity":
                values[name] = _gray(_opacity(ps, it))
            else:  # ids
                values[name] = _ids(ps, it, name)
        # Sum per texel over this chunk's texels only (not the whole image).
        texels, inv = np.unique(tex[rows], return_inverse=True)
        for name, v in values.items():
            for c in range(4):
                channels[name][texels, c] += np.bincount(inv, v[:, c], texels.size)
        weight[texels] += np.bincount(inv, minlength=texels.size)
        t_shade += time.perf_counter() - ta
        if progress is not None:
            progress(min(start + CHUNK, total), total)
    # Texel averages, encodings and padding.
    t0 = time.perf_counter()
    hit_mask = weight > 0
    out: dict[str, np.ndarray] = {}
    for name in names:
        acc = channels[name]
        mean = np.divide(acc, weight[:, None], out=np.zeros_like(acc), where=hit_mask[:, None])
        kind = MAPS[name]
        if kind == "normal":
            v = mean[:, :3]
            length = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.divide(v, length, out=np.zeros_like(v), where=length > 1e-300)
            mean[:, :3] = v * 0.5 + 0.5
        elif kind == "color" and name not in ("material_id", "object_id", "group_id"):
            from ..rendering.color import linear_to_srgb

            mean[:, :3] = linear_to_srgb(np.clip(mean[:, :3], 0.0, 1.0))
        mean[:, 3] = np.where(hit_mask, 1.0, 0.0)
        image = mean.reshape(height, width, 4)
        if padding:
            image, _ = dilate(image, hit_mask.reshape(height, width), padding)
        out[name] = np.clip(image, 0.0, 1.0)
    t_finish = time.perf_counter() - t0
    covered_texels = int(np.count_nonzero(covered))
    missed = int(np.count_nonzero((covered > 0) & ~hit_mask))
    diagnostics: list[Diagnostic] = []
    if total and hits_total < total:
        miss_fraction = 1.0 - hits_total / total
        diagnostics.append(
            Diagnostic(
                "BAKE_MISSES",
                WARNING if miss_fraction > 0.01 else INFO,
                f"{total - hits_total:,} of {total:,} samples ({miss_fraction:.2%}) found no "
                f"source surface within the cage ({cage:g}) and depth ({depth:g}); "
                f"{missed:,} texels got no value and were filled by padding",
                "cage",
            )
        )
    if cov.overlaps:
        diagnostics.append(
            Diagnostic(
                "UV_OVERLAP",
                WARNING,
                f"{cov.overlaps:,} samples lie in overlapping UV triangles; each took the "
                "lowest-numbered triangle",
                "target",
            )
        )
    report = {
        "target": target_obj.name,
        "sources": [o.name for o in source_objs],
        "maps": names,
        "resolution": [width, height],
        "samples": n,
        "cage": cage,
        "depth": depth,
        "padding": padding,
        "backend": tracer.name,
        "texels_covered": covered_texels,
        "texel_samples": int(total),
        "hits": int(hits_total),
        "miss_fraction": 0.0 if not total else 1.0 - hits_total / total,
        "texels_missed": missed,
        "uv_overlap_samples": int(cov.overlaps),
        "rays": int(tracer.rays),
        "seconds": {
            "setup": t_setup,
            "rasterize": t_raster,
            "trace": t_trace,
            "shade": t_shade,
            "finish": t_finish,
            "total": time.perf_counter() - t_start,
        },
    }
    if "ao" in names:
        report["ao"] = {"rays": ao_rays, "distance": ao_distance, "seed": seed}
    if "thickness" in names:
        report["thickness"] = {"rays": thickness_rays, "distance": thickness_distance}
    if "curvature" in names:
        report["curvature_radius"] = curvature_radius
    if "height" in names:
        report["height_range"] = height_range
    hit_fraction = np.divide(weight, covered, out=np.zeros_like(weight), where=covered > 0)
    return BakeResult(
        out,
        (covered / (n * n)).reshape(height, width),
        hit_fraction.reshape(height, width),
        report,
        diagnostics,
    )


def _gray(v: np.ndarray) -> np.ndarray:
    return np.stack([v, v, v, np.ones_like(v)], axis=1)


def _object(scene: Any, ref: Any, where: str):
    from ..scene import SceneObject

    if isinstance(ref, SceneObject):
        if ref.scene is not scene:
            raise _bad("the object belongs to another scene", where)
        return ref
    from ..errors import ObjectNotFoundError

    try:
        return scene.get(ref)
    except (ObjectNotFoundError, TypeError):
        try:
            return scene.get_by_name(ref)
        except (ObjectNotFoundError, TypeError):
            raise TexturingError("OBJECT_NOT_FOUND", f"no object {ref!r}", where) from None


class _CPUTracer:
    name = "cpu"

    def __init__(self, ps: Any) -> None:
        import os

        self.ps = ps
        self.rays = 0
        # A bake is one process: trace with every core unless AETHER3D_NATIVE_THREADS says.
        if not os.environ.get("AETHER3D_NATIVE_THREADS", "").strip():
            ps.accel.threads = max(1, os.cpu_count() or 1)

    def hemisphere(self, mode, origin, normal, shifts, count, distance, skip_inst, skip_tri):
        """Mode 1: the visible fraction of ``count`` cosine-weighted rays around ``normal``;
        mode 2: their mean distance to a hit (misses: ``distance``) over ``distance``."""
        m = origin.shape[0]
        d = hemisphere_rays(normal, shifts, count).reshape(-1, 3)
        o = np.repeat(origin, count, axis=0)
        si = np.repeat(skip_inst, count)
        st = np.repeat(skip_tri, count)
        tmax = np.full(o.shape[0], distance)
        if mode == 1:
            blocked = self.occluded(o, d, tmax, si, st)
            return 1.0 - blocked.reshape(m, count).mean(axis=1)
        hits = self.closest(o, d, tmax, si, st)
        t = np.where(hits.inst >= 0, np.minimum(hits.t, distance), distance)
        return t.reshape(m, count).mean(axis=1) / distance

    def closest(self, origin, direction, tmax, skip_inst=None, skip_tri=None):
        self.rays += origin.shape[0]
        return self.ps.accel.closest_hit(origin, direction, 0.0, tmax, skip_inst, skip_tri)

    def occluded(self, origin, direction, tmax, skip_inst=None, skip_tri=None):
        self.rays += origin.shape[0]
        return self.ps.accel.occluded(origin, direction, 0.0, tmax, skip_inst, skip_tri)


def _tracer(ps: Any, backend: str, adapter: Any = None, *, hemisphere: bool = True):
    """The ray tracer of a bake. ``"auto"`` takes the GPU only for bakes with hemisphere
    maps (AO, thickness: many rays per sample, where it was measured about 7 times faster);
    one projection ray per sample is not worth the transfers (measured slower at 4K)."""
    if backend == "cpu":
        return _CPUTracer(ps)
    import os

    if backend == "auto" and (not hemisphere or os.environ.get("AETHER3D_GPU_TESTS") == "skip"):
        return _CPUTracer(ps)  # (portable test runs never touch the GPU)
    try:
        from .gpu_bake import GPUTracer

        return GPUTracer(ps, adapter)
    except TexturingError:
        if backend == "gpu":
            raise
    return _CPUTracer(ps)


def _ao(tracer, ps, it, n_side, keys, count, distance) -> np.ndarray:
    from ..rendering.pathtrace.sampling import SLOT_AO

    origin = ps.offset_origin(it.p, it.ng)
    shifts = hemisphere_shifts(keys, SLOT_AO)
    return tracer.hemisphere(1, origin, n_side, shifts, count, distance, it.inst, it.prim)


def _thickness(tracer, ps, it, n_side, keys, count, distance) -> np.ndarray:
    origin = ps.offset_origin(it.p, -it.ng)
    shifts = hemisphere_shifts(keys, 64)
    return tracer.hemisphere(2, origin, -n_side, shifts, count, distance, it.inst, it.prim)


def _curvature_at(ps, it, curvature) -> np.ndarray:
    out = np.zeros(it.p.shape[0])
    for j, inst in enumerate(ps.rs.instances):
        sel = it.inst == j
        if not sel.any():
            continue
        mesh = ps.rs.meshes[inst.mesh]
        tri = mesh.triangles.astype(np.int64)[it.prim[sel]]
        k = curvature[inst.mesh][tri]
        # Curvature scales as 1 / length: uniform scale of the instance divides it.
        scale = abs(float(np.linalg.det(inst.world[:3, :3]))) ** (1.0 / 3.0)
        out[sel] = (it.b[sel] * k).sum(axis=1) / scale
    return out


def _opacity(ps, it) -> np.ndarray:
    from ..rendering import surface

    s = it.samples()
    out = np.ones(s.tri.shape[0])
    mat = ps.MAT[s.tri]
    for m in np.unique(mat).tolist():
        sel = mat == m
        sv = surface.evaluate_material(ps.ctx, s.subset(sel), ps.rs.materials[m], alpha_only=True)
        out[sel] = np.clip(sv.alpha, 0.0, 1.0)
    return out


def _ids(ps, it, name: str) -> np.ndarray:
    m = it.p.shape[0]
    out = np.ones((m, 4))
    if name == "object_id":
        keys = [inst.object_id for inst in ps.rs.instances]
        idx = it.inst
    elif name == "material_id":
        keys = [mat.key for mat in ps.rs.materials]
        idx = ps.MAT[it.gtri]
    else:
        groups = np.zeros(m, dtype=np.int64)
        for j, inst in enumerate(ps.rs.instances):
            sel = it.inst == j
            g = ps.rs.meshes[inst.mesh].groups
            if sel.any() and g is not None:
                groups[sel] = g[it.prim[sel]]
        uniq = np.unique(groups)
        keys = [f"group:{g}" for g in uniq.tolist()]
        idx = np.searchsorted(uniq, groups)
    palette = np.array([id_color(k) for k in keys]) if keys else np.zeros((0, 3))
    out[:, :3] = palette[idx]
    return out


SLOT_OF_MAP = {
    "normal": ("normal", None),
    "ao": ("occlusion", "r"),
    "base_color": ("base_color", None),
    "roughness": ("roughness", "r"),
    "metallic": ("metallic", "r"),
    "emissive": ("emissive", None),
    "opacity": ("opacity", "r"),
}
"""The material slot (and channel) each bakeable map drives."""

NEUTRAL_FACTORS = {
    "base_color": {"base_color": (1.0, 1.0, 1.0)},
    "roughness": {"roughness": 1.0},
    "metallic": {"metallic": 1.0},
    "emissive": {"emissive": (1.0, 1.0, 1.0), "emissive_strength": 1.0},
    "opacity": {"opacity": 1.0},
}
"""Factors set where a baked map is bound: the maps hold the evaluated values (factor x
texture), so the factor they multiply must be neutral."""


def apply_bake(
    scene: Any,
    target: Any,
    result: BakeResult,
    *,
    material: Any = None,
    names: dict[str, str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Store every map of ``result`` as a painted texture and bind the ones that drive a
    material slot (``normal``, ``ao`` -> occlusion, ``base_color``, ``roughness``,
    ``metallic``, ``emissive``, ``opacity``) to ``material`` (by default the target's
    material, created if it has none), setting the factors those maps multiply to neutral
    values (:data:`NEUTRAL_FACTORS`: the maps already hold factor x texture) -- one step
    of the scene's history. Returns ``{map: texture}`` and the material."""
    obj = _object(scene, target, "target")
    names = names or {}
    with scene.transaction(label or f"Apply bake to {obj.name!r}"):
        if material is None:
            material = obj.material
            if material is None:
                material = scene.create_material(
                    scene.materials._unique_material_name(f"{obj.name} baked")
                )
                obj.material = material
        elif not hasattr(material, "set_texture"):
            material = scene.materials.material(material)
        textures = {}
        for map_name in result.maps:
            tex_name = names.get(map_name) or scene.materials._unique_texture_name(
                f"{obj.name} {map_name}"
            )
            tex = result.to_texture(scene, map_name, tex_name)
            textures[map_name] = tex
            slot = SLOT_OF_MAP.get(map_name)
            if slot is not None:
                options = {} if slot[1] is None else {"channel": slot[1]}
                material.set_texture(slot[0], tex, **options)
                for param, value in NEUTRAL_FACTORS.get(map_name, {}).items():
                    setattr(material, param, value)
    return {"textures": textures, "material": material}
