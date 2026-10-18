"""The CPU rasterization backend: RenderScene in, linear-light image out.

Stages (timed separately, see :class:`CPUFrame`):

1. **geometry** -- vertices to world space (``M``) and normals with the inverse
   transpose; per-instance clip coordinates (``P V M``); per-triangle facing
   from the world-space outward normal (mirrored instances flip their
   winding); back/front-face culling; frustum rejection and clipping.
2. **raster** -- fixed-point rasterization into a visibility buffer of
   ``(width * s) x (height * s)`` samples holding, per sample, the nearest
   triangle (see :mod:`aether3d.rendering.raster`).
3. **shade** -- deferred shading of every covered sample exactly once:
   perspective-correct barycentrics from the original triangle in homogeneous
   2D coordinates (valid even for triangles clipped at the near plane) and
   their exact screen-space derivatives, world position and normal
   interpolation, material evaluation (textures, normal maps; see
   :mod:`aether3d.rendering.surface`), two-sided lighting.

Materials also act before shading: back faces of single-sided materials are
culled with the triangles, and alpha-masked materials test every fragment's
opacity before the depth test, so surfaces behind cut-out holes show.
4. **resolve** -- per-pixel average of its ``s x s`` samples in linear light.

All per-vertex and per-sample arithmetic is written out explicitly
(multiplies and adds in a fixed order) rather than delegated to BLAS, so the
same inputs give bitwise identical results on every platform.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from ..errors import RenderError
from . import atmosphere as atmo
from . import raster, surface
from .raster import EMPTY, RasterStats
from .scene_data import RenderScene, _mat4_mul
from .settings import RenderLimits

SHADE_BATCH = 131_072
"""Samples shaded per batch (bounds temporary memory)."""
SCAN_CHUNK = 1_048_576
"""Visibility-buffer entries scanned per chunk while shading."""
TRIANGLE_BATCH = 131_072
"""Triangles per geometry/raster batch."""
MIN_LIGHT_DISTANCE = 1e-4
"""Point-light distances are clamped to at least this (avoids division by zero)."""


@dataclass
class CPUFrame:
    """Output of :func:`render_scene`: linear-light buffers at the final resolution."""

    linear: np.ndarray  # (H, W, 3) float64, scene-referred (before exposure)
    coverage: np.ndarray  # (H, W) float32: fraction of samples covered by geometry
    depth: np.ndarray  # (H, W) float32: view depth at the representative sample (inf: none)
    instance: np.ndarray  # (H, W) int32: instance at the representative sample (-1: none)
    visible_samples: np.ndarray  # (instances,) int64
    counts: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _transform_points(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    x, y, z = (p[:, i].astype(np.float64) for i in range(3))
    out = np.empty((p.shape[0], 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = m[i, 0] * x + m[i, 1] * y + m[i, 2] * z + m[i, 3]
    return out


def _transform_normals(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    x, y, z = (n[:, i].astype(np.float64) for i in range(3))
    out = np.empty((n.shape[0], 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = m[i, 0] * x + m[i, 1] * y + m[i, 2] * z
    length = np.sqrt(out[:, 0] * out[:, 0] + out[:, 1] * out[:, 1] + out[:, 2] * out[:, 2])
    return np.divide(out, length[:, None], out=np.zeros_like(out), where=length[:, None] > 1e-300)


def _clip_coords(p: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.empty((p.shape[0], 4), dtype=np.float64)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    for i in range(4):
        out[:, i] = m[i, 0] * x + m[i, 1] * y + m[i, 2] * z + m[i, 3]
    return out


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _normalize(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = np.sqrt(_dot(v, v))
    unit = np.divide(v, length[..., None], out=np.zeros_like(v), where=length[..., None] > 1e-300)
    return unit, length


def render_scene(
    rs: RenderScene,
    width: int,
    height: int,
    samples: int,
    cull: str,
    limits: RenderLimits | None = None,
) -> CPUFrame:
    """Rasterize and shade ``rs`` into linear-light buffers (see the module docstring)."""
    limits = limits or RenderLimits()
    n_samples = width * height * samples * samples
    if width > limits.max_dimension or height > limits.max_dimension:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"image size {width}x{height} exceeds the limit of {limits.max_dimension} pixels "
            "per side",
            hint="lower width or height",
        )
    if n_samples > limits.max_samples:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"{width}x{height} pixels with {samples}x{samples} antialiasing needs {n_samples:,} "
            f"samples (limit {limits.max_samples:,})",
            hint="lower the resolution or the samples setting",
        )
    if rs.triangle_count > limits.max_triangles:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"the scene has {rs.triangle_count:,} triangles to render "
            f"(limit {limits.max_triangles:,})",
            hint="hide objects or reduce subdivision levels",
        )
    sw, sh = width * samples, height * samples
    aspect = width / height
    camera = rs.camera
    view = camera.frame.view_matrix()
    proj = camera.camera.projection_matrix(aspect)
    vp = _mat4_mul(proj, view)
    perspective = camera.camera.is_perspective
    eye = camera.frame.position
    forward = camera.frame.forward
    timings: dict[str, float] = {}
    diagnostics: list[Diagnostic] = []

    # ---- geometry --------------------------------------------------------------------
    t0 = time.perf_counter()
    positions, normals, has_normals, tri_vertices, tri_instance = [], [], [], [], []
    tri_material = []
    vertex_base = 0
    for k, inst in enumerate(rs.instances):
        mesh = rs.meshes[inst.mesh]
        pw = _transform_points(mesh.positions, inst.world)
        positions.append(pw)
        if mesh.normals is not None:
            normals.append(_transform_normals(mesh.normals, inst.normal_matrix))
        else:
            normals.append(np.zeros((pw.shape[0], 3)))
        has_normals.append(mesh.normals is not None)
        tri_vertices.append(mesh.triangles.astype(np.int64) + vertex_base)
        tri_instance.append(np.full(mesh.triangles.shape[0], k, dtype=np.int32))
        tri_material.append(inst.material_of_groups(mesh.groups, mesh.triangles.shape[0]))
        vertex_base += pw.shape[0]
    P = np.concatenate(positions) if positions else np.zeros((0, 3))
    N = np.concatenate(normals) if normals else np.zeros((0, 3))
    TRI = np.concatenate(tri_vertices) if tri_vertices else np.zeros((0, 3), dtype=np.int64)
    INST = np.concatenate(tri_instance) if tri_instance else np.zeros(0, dtype=np.int32)
    MAT = np.concatenate(tri_material) if tri_material else np.zeros(0, dtype=np.int32)
    has_normals_arr = np.array(has_normals, dtype=bool)
    mirrored = np.array([i.mirrored for i in rs.instances], dtype=bool)
    single_sided = np.array([not m.double_sided for m in rs.materials], dtype=bool)
    masked = np.array([m.masked for m in rs.materials], dtype=bool)
    ctx = _shade_context(rs, P, TRI, INST, MAT, eye, forward, perspective, sw, sh)
    n_tris = TRI.shape[0]
    id_bits = max(1, int(n_tris).bit_length())
    counts = {
        "instances": len(rs.instances),
        "meshes": len(rs.meshes),
        "lights": len(rs.lights),
        "triangles": int(n_tris),
        "culled_backface": 0,
        "culled_frustum": 0,
        "clipped": 0,
        "nonfinite": 0,
        "alpha_discarded": 0,
    }
    clip = _clip_coords(P, vp) if P.shape[0] else np.zeros((0, 4))
    # Per-vertex work, shared by every triangle using the vertex: finiteness,
    # frustum outcodes and (for triangles that need no clipping) screen setup.
    vertex_finite = np.all(np.isfinite(clip), axis=1)
    codes = raster.outcodes(clip)
    VX, VY, VZ = raster.vertex_screen(clip, sw, sh)
    t_geometry = time.perf_counter() - t0

    # ---- raster ---------------------------------------------------------------------
    keys = np.full(sw * sh, EMPTY, dtype=np.int64)
    rstats = RasterStats()
    t_raster = 0.0
    keep_fragment = None
    if masked.any():
        masked_tris = np.flatnonzero(masked[MAT])
        if masked_tris.size:
            alpha_table = _triangle_table(
                masked_tris,
                P,
                N,
                TRI,
                INST,
                MAT,
                has_normals_arr,
                mirrored,
                vp,
                eye,
                forward,
                perspective,
            )
            keep_fragment = _alpha_test(ctx, alpha_table, masked, sw, sh, counts)
    for start in range(0, n_tris, TRIANGLE_BATCH):
        ta = time.perf_counter()
        ids = np.arange(start, min(start + TRIANGLE_BATCH, n_tris), dtype=np.int64)
        tv = TRI[ids]
        finite = vertex_finite[tv].all(axis=1)
        if not finite.all():
            counts["nonfinite"] += int(np.count_nonzero(~finite))
            ids, tv = ids[finite], tv[finite]
        if cull != "none" and ids.size:
            front = _front_facing(P, tv, mirrored[INST[ids]], eye, forward, perspective)
            keep = ~front if cull == "front" else front
            counts["culled_backface"] += int(np.count_nonzero(~keep))
            ids, tv = ids[keep], tv[keep]
        if single_sided.any() and ids.size:
            single = single_sided[MAT[ids]]
            if single.any():  # single-sided materials: their back faces are not drawn
                keep = np.ones(ids.shape[0], dtype=bool)
                keep[single] = _front_facing(
                    P, tv[single], mirrored[INST[ids[single]]], eye, forward, perspective
                )
                counts["culled_backface"] += int(np.count_nonzero(~keep))
                ids, tv = ids[keep], tv[keep]
        oc = codes[tv]
        outside = (oc[:, 0] & oc[:, 1] & oc[:, 2]) != 0
        counts["culled_frustum"] += int(np.count_nonzero(outside))
        ids, tv, oc = ids[~outside], tv[~outside], oc[~outside]
        inside = (oc[:, 0] | oc[:, 1] | oc[:, 2]) == 0
        counts["clipped"] += int(np.count_nonzero(~inside))
        tin = tv[inside]
        X, Y, z = VX[tin], VY[tin], VZ[tin]
        ids_in = ids[inside]
        if not inside.all():
            cpos, cids = raster.clip_triangles(clip[tv[~inside]], ids[~inside])
            cX, cY, cz, _ = raster.screen_setup(cpos, sw, sh)
            X, Y, z = np.concatenate([X, cX]), np.concatenate([Y, cY]), np.concatenate([z, cz])
            ids_in = np.concatenate([ids_in, cids])
        A2 = raster.signed_area(X, Y)
        tb = time.perf_counter()
        t_geometry += tb - ta
        raster.rasterize(X, Y, z, A2, ids_in, sw, sh, keys, id_bits, rstats, keep_fragment)
        t_raster += time.perf_counter() - tb
    timings["geometry"] = t_geometry
    timings["raster"] = t_raster
    if counts["nonfinite"]:
        diagnostics.append(
            Diagnostic(
                "NONFINITE_GEOMETRY",
                WARNING,
                f"{counts['nonfinite']} triangles have non-finite coordinates after "
                "transformation (extreme scales or positions) and were skipped",
                count=counts["nonfinite"],
            )
        )

    # ---- shade + resolve -------------------------------------------------------------
    t0 = time.perf_counter()
    n_pixels = width * height
    accum = [np.zeros(n_pixels, dtype=np.float64) for _ in range(3)]
    covered_per_pixel = np.zeros(n_pixels, dtype=np.int64)
    depth = np.full(n_pixels, np.inf, dtype=np.float32)
    instance_map = np.full(n_pixels, -1, dtype=np.int32)
    visible = np.zeros(len(rs.instances), dtype=np.int64)
    mask = (1 << id_bits) - 1
    # Pass 1 (chunked): which triangles are visible in at least one sample.
    present = np.zeros(TRI.shape[0], dtype=bool)
    for c0 in range(0, keys.size, SCAN_CHUNK):
        chunk = keys[c0 : c0 + SCAN_CHUNK]
        present[chunk[chunk != EMPTY] & mask] = True
    table = _triangle_table(
        np.flatnonzero(present),
        P,
        N,
        TRI,
        INST,
        MAT,
        has_normals_arr,
        mirrored,
        vp,
        eye,
        forward,
        perspective,
    )
    hx, hy = camera.camera.half_extents(aspect)
    rep = samples // 2  # representative sample of each pixel (its center for odd s)
    n_covered = 0
    # Pass 2 (chunked): shade every covered sample once.
    for c0 in range(0, keys.size, SCAN_CHUNK):
        chunk = keys[c0 : c0 + SCAN_CHUNK]
        local = np.flatnonzero(chunk != EMPTY)
        n_covered += local.size
        for s0 in range(0, local.size, SHADE_BATCH):
            sel = local[s0 : s0 + SHADE_BATCH]
            idx = sel + c0
            ct = table.compact[chunk[sel] & mask]
            row, col = np.divmod(idx, sw)
            radiance, view_depth, inst = _shade(
                rs,
                table,
                ct,
                (col + 0.5) * (2.0 / sw) - 1.0,
                1.0 - (row + 0.5) * (2.0 / sh),
                camera.frame,
                hx,
                hy,
                perspective,
                ctx,
            )
            pixel = (row // samples) * width + (col // samples)
            for c in range(3):
                np.add.at(accum[c], pixel, radiance[c])
            np.add.at(covered_per_pixel, pixel, 1)
            np.add.at(visible, inst, 1)
            is_rep = ((row % samples) == rep) & ((col % samples) == rep)
            depth[pixel[is_rep]] = view_depth[is_rep]
            instance_map[pixel[is_rep]] = inst[is_rep]
    del keys
    uncovered = samples * samples - covered_per_pixel
    inv = 1.0 / float(samples * samples)
    background = _background(rs, width, height, hx, hy, perspective)
    for c in range(3):  # in place: no full-image temporaries
        accum[c] += (rs.background[c] if background is None else background[:, c]) * uncovered
        accum[c] *= inv
    linear = np.stack(accum, axis=-1)
    del accum
    timings["shade"] = time.perf_counter() - t0
    counts.update(rstats.to_dict())
    counts["samples"] = int(n_samples)
    counts["samples_covered"] = int(n_covered)
    return CPUFrame(
        linear=linear.reshape(height, width, 3),
        coverage=(covered_per_pixel / float(samples * samples))
        .astype(np.float32)
        .reshape(height, width),
        depth=depth.reshape(height, width),
        instance=instance_map.reshape(height, width),
        visible_samples=visible,
        counts=counts,
        timings=timings,
        diagnostics=diagnostics,
    )


def _background(
    rs: RenderScene, width: int, height: int, hx: float, hy: float, perspective: bool
) -> np.ndarray | None:
    """Per-pixel background radiance ``(pixels, 3)`` with the atmosphere (its sky at the
    pixel centre's view direction, through the fog), or ``None`` (the flat background)."""
    model = _atmosphere(rs)
    if model is None or not (model.sky_background or model.fog):
        return None
    frame = rs.camera.frame
    row, col = np.divmod(np.arange(width * height), width)
    xn = (col + 0.5) * (2.0 / width) - 1.0
    yn = 1.0 - (row + 0.5) * (2.0 / height)
    if perspective:
        d = (
            frame.forward[None, :]
            + (xn * hx)[:, None] * frame.right[None, :]
            + (yn * hy)[:, None] * frame.up[None, :]
        )
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
    else:
        d = np.broadcast_to(frame.forward, (width * height, 3)).copy()
    if model.sky_background:
        color = atmo.sky(model, d)
    else:
        color = np.broadcast_to(rs.background, d.shape).copy()
    return atmo.fog(model, color, d, np.full(d.shape[0], np.inf))


def _front_facing(
    P: np.ndarray,
    tv: np.ndarray,
    mirrored: np.ndarray,
    eye: np.ndarray,
    forward: np.ndarray,
    perspective: bool,
) -> np.ndarray:
    """Whether each triangle's outward side faces the camera (world-space test)."""
    p0, p1, p2 = P[tv[:, 0]], P[tv[:, 1]], P[tv[:, 2]]
    g = _cross(p1 - p0, p2 - p0)
    g = np.where(mirrored[:, None], -g, g)
    to_eye = (eye - p0) if perspective else np.broadcast_to(-forward, p0.shape)
    return _dot(g, to_eye) > 0.0


@dataclass
class _TriangleTable:
    """Per-triangle shading data for the triangles visible in at least one sample."""

    compact: np.ndarray  # (all triangles,) -> row in this table (only valid for visible ones)
    adj: np.ndarray  # (T, 9): rows of adj(M) for the homogeneous barycentrics
    det: np.ndarray  # (T,): det(M)
    vdepth: np.ndarray  # (T, 3): view depth of each vertex
    geo: np.ndarray  # (T, 3): unit geometric normal, turned toward the camera
    normals: np.ndarray  # (T, 9): vertex normals, turned toward the camera (zeros if none)
    has_normals: np.ndarray  # (T,) bool
    instance: np.ndarray  # (T,) int32
    sign: np.ndarray  # (T,) +1 / -1: what turned the normals toward the camera
    triangle: np.ndarray  # (T,) global triangle id
    material: np.ndarray  # (T,) material index


def _triangle_table(
    present: np.ndarray,
    P: np.ndarray,
    N: np.ndarray,
    TRI: np.ndarray,
    INST: np.ndarray,
    MAT: np.ndarray,
    has_normals: np.ndarray,
    mirrored: np.ndarray,
    vp: np.ndarray,
    eye: np.ndarray,
    forward: np.ndarray,
    perspective: bool,
) -> _TriangleTable:
    compact = np.zeros(TRI.shape[0], dtype=np.int32 if TRI.shape[0] < 2**31 else np.int64)
    compact[present] = np.arange(present.size)
    tv = TRI[present]
    inst = INST[present]
    p = [P[tv[:, k]] for k in range(3)]
    # Clip-space columns c_k = (x, y, w); the homogeneous barycentrics of a
    # sample s = (xn, yn, 1) are proportional to adj(M) s with M = [c0 c1 c2],
    # whose rows are c1 x c2, c2 x c0 and c0 x c1.
    cols = []
    for k in range(3):
        x, y, z = p[k][:, 0], p[k][:, 1], p[k][:, 2]
        cols.append(
            np.stack(
                [
                    vp[0, 0] * x + vp[0, 1] * y + vp[0, 2] * z + vp[0, 3],
                    vp[1, 0] * x + vp[1, 1] * y + vp[1, 2] * z + vp[1, 3],
                    vp[3, 0] * x + vp[3, 1] * y + vp[3, 2] * z + vp[3, 3],
                ],
                axis=-1,
            )
        )
    r0, r1, r2 = _cross(cols[1], cols[2]), _cross(cols[2], cols[0]), _cross(cols[0], cols[1])
    adj = np.concatenate([r0, r1, r2], axis=1)
    det = _dot(cols[0], r0)
    vdepth = np.stack([_dot(p[k] - eye, np.broadcast_to(forward, p[k].shape)) for k in range(3)], 1)
    geo = _cross(p[1] - p[0], p[2] - p[0])
    geo = np.where(mirrored[inst][:, None], -geo, geo)
    to_eye = (eye - p[0]) if perspective else np.broadcast_to(-forward, p[0].shape)
    sign = np.where(_dot(geo, to_eye) > 0.0, 1.0, -1.0)  # two-sided: face the camera
    geo_unit, _ = _normalize(geo)
    geo_unit *= sign[:, None]
    normals = np.concatenate([N[tv[:, k]] for k in range(3)], axis=1) * sign[:, None]
    return _TriangleTable(
        compact,
        adj,
        det,
        vdepth,
        geo_unit,
        normals,
        has_normals[inst],
        inst.astype(np.int32),
        sign,
        present,
        MAT[present] if MAT.size else np.zeros(0, dtype=np.int32),
    )


def _shade(
    rs: RenderScene,
    table: _TriangleTable,
    ct: np.ndarray,
    xn: np.ndarray,
    yn: np.ndarray,
    frame,
    hx: float,
    hy: float,
    perspective: bool,
    ctx: surface.ShadeContext,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Radiance (list of 3 component arrays), view depth and instance of each sample."""
    a = table.adj[ct]
    b0 = a[:, 0] * xn + a[:, 1] * yn + a[:, 2]
    b1 = a[:, 3] * xn + a[:, 4] * yn + a[:, 5]
    b2 = a[:, 6] * xn + a[:, 7] * yn + a[:, 8]
    total = b0 + b1 + b2
    safe = np.where(np.abs(total) > 1e-300, total, 1.0)
    # Normalized perspective-correct barycentrics (clamped: samples on an edge may
    # round slightly outside).
    b0, b1, b2 = (np.maximum(b / safe, 0.0) for b in (b0, b1, b2))
    norm = np.maximum(b0 + b1 + b2, 1e-300)
    b0, b1, b2 = b0 / norm, b1 / norm, b2 / norm
    ex, ey, ez = frame.position
    fx, fy, fz = frame.forward
    ux, uy, uz = frame.up
    rx, ry, rz = frame.right
    if perspective:
        w = table.det[ct] / safe  # view depth along the camera's forward axis
        sx, sy = xn * hx, yn * hy
        px = ex + w * (fx + sx * rx + sy * ux)
        py = ey + w * (fy + sx * ry + sy * uy)
        pz = ez + w * (fz + sx * rz + sy * uz)
        view_depth = w
    else:
        d = table.vdepth[ct]
        view_depth = b0 * d[:, 0] + b1 * d[:, 1] + b2 * d[:, 2]
        sx, sy = xn * hx, yn * hy
        px = ex + view_depth * fx + sx * rx + sy * ux
        py = ey + view_depth * fy + sx * ry + sy * uy
        pz = ez + view_depth * fz + sx * rz + sy * uz

    nv = table.normals[ct]
    nx = b0 * nv[:, 0] + b1 * nv[:, 3] + b2 * nv[:, 6]
    ny = b0 * nv[:, 1] + b1 * nv[:, 4] + b2 * nv[:, 7]
    nz = b0 * nv[:, 2] + b1 * nv[:, 5] + b2 * nv[:, 8]
    n_len = np.sqrt(nx * nx + ny * ny + nz * nz)
    g = table.geo[ct]
    use = table.has_normals[ct] & (n_len > 1e-6)
    inv = np.where(use, 1.0 / np.where(use, n_len, 1.0), 0.0)
    nx = np.where(use, nx * inv, g[:, 0])
    ny = np.where(use, ny * inv, g[:, 1])
    nz = np.where(use, nz * inv, g[:, 2])

    inst = table.instance[ct]
    mat = table.material[ct]
    materials = rs.materials
    plain = np.array([m.model == "simple" and m.plain for m in materials], dtype=bool)
    radiance = [np.empty(ct.shape[0]) for _ in range(3)]
    simple_rows = plain[mat] if len(materials) else np.ones(ct.shape[0], dtype=bool)
    if ctx.has_colors is not None:  # vertex colours shade through the material path
        simple_rows &= ~ctx.has_colors[inst]
    if simple_rows.any():
        # Untextured simple materials (every object color): Stage 5 shading, unchanged.
        rows = None if simple_rows.all() else simple_rows
        pick = (lambda a: a) if rows is None else (lambda a: a[rows])
        base_tab = np.array([m.base_color for m in materials]).reshape(-1, 3)
        spec_tab = np.array([m.specular for m in materials], dtype=np.float64)
        shin_tab = np.array([m.shininess for m in materials], dtype=np.float64)
        m_rows = pick(mat)
        spec = spec_tab[m_rows]
        view = None
        if bool(np.any(spec > 0.0)):
            view = surface.view_vectors(ctx, pick(px), pick(py), pick(pz))
        part = surface.shade_simple(
            rs,
            base_tab[m_rows],
            spec,
            shin_tab[m_rows],
            None,
            (pick(nx), pick(ny), pick(nz)),
            (pick(px), pick(py), pick(pz)),
            view,
        )
        for c in range(3):
            if rows is None:
                radiance[c] = part[c]
            else:
                radiance[c][rows] = part[c]
    if not simple_rows.all():
        _shade_materials(
            rs, ctx, table, ct, xn, yn, ~simple_rows, mat, (nx, ny, nz), (px, py, pz), radiance
        )
    model = _atmosphere(rs)
    if model is not None and model.fog:  # aerial perspective along the camera rays
        dx, dy, dz = px - ex, py - ey, pz - ez
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        safe = np.maximum(dist, 1e-30)
        d = np.stack([dx / safe, dy / safe, dz / safe], axis=1)
        fogged = atmo.fog(model, np.stack(radiance, axis=1), d, dist)
        radiance = [fogged[:, c] for c in range(3)]
    return radiance, view_depth, inst


def _atmosphere(rs: RenderScene):
    atm = rs.atmosphere
    if atm is None or not atm["settings"].enabled:
        return None
    return atmo.resolve(rs)


def _shade_context(
    rs: RenderScene,
    P: np.ndarray,
    TRI: np.ndarray,
    INST: np.ndarray,
    MAT: np.ndarray,
    eye: np.ndarray,
    forward: np.ndarray,
    perspective: bool,
    sw: int,
    sh: int,
) -> surface.ShadeContext:
    """The per-render arrays material evaluation reads (object positions and UVs only if
    some material has textures)."""
    textured = any(m.textures for m in rs.materials)
    PO = UV = np.zeros((0, 3))
    if textured and rs.instances:
        meshes = [rs.meshes[i.mesh] for i in rs.instances]
        PO = np.concatenate([m.positions.astype(np.float64) for m in meshes])
        UV = np.concatenate(
            [
                m.uvs.astype(np.float64)
                if m.uvs is not None
                else np.zeros((m.positions.shape[0], 2))
                for m in meshes
            ]
        )
    has_uvs = np.array([rs.meshes[i.mesh].uvs is not None for i in rs.instances], dtype=bool)
    linear = np.array([i.world[:3, :3] for i in rs.instances], dtype=np.float64).reshape(-1, 3, 3)
    CO, has_colors = vertex_colors(rs)
    return surface.ShadeContext(
        rs,
        P,
        PO,
        UV,
        TRI,
        INST,
        MAT,
        has_uvs,
        linear,
        eye,
        forward,
        perspective,
        2.0 / sw,
        -2.0 / sh,
        CO=CO,
        has_colors=has_colors,
    )


def vertex_colors(rs: RenderScene) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(V, 4)`` vertex colours of every instance's vertices, concatenated in instance
    order (ones for meshes without), and which instances have them; ``(None, None)`` when no
    instance's mesh has colours."""
    has = np.array([rs.meshes[i.mesh].colors is not None for i in rs.instances], dtype=bool)
    if not has.any():
        return None, None
    parts = []
    for inst in rs.instances:
        m = rs.meshes[inst.mesh]
        if m.colors is not None:
            parts.append(m.colors.astype(np.float64))
        else:
            parts.append(np.ones((m.positions.shape[0], 4)))
    return np.concatenate(parts), has


def _surface_samples(
    ctx: surface.ShadeContext, table: _TriangleTable, ct: np.ndarray, xn: np.ndarray, yn: np.ndarray
) -> surface.Samples:
    """Barycentrics with derivatives and outward shading normals of samples (for materials)."""
    a = table.adj[ct]
    b, dbx, dby = surface.barycentric_derivatives(a, xn, yn, ctx.step_x, ctx.step_y)
    b = np.maximum(b, 0.0)
    b = b / np.maximum(b.sum(axis=1, keepdims=True), 1e-300)
    nv = table.normals[ct].reshape(-1, 3, 3)
    n = np.einsum("nk,nkc->nc", b, nv)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    use = table.has_normals[ct] & (length[:, 0] > 1e-6)
    toward = table.sign[ct]
    n = np.where(use[:, None], n / np.maximum(length, 1e-300), table.geo[ct])
    outward = n * toward[:, None]
    return surface.Samples(table.triangle[ct], table.instance[ct], b, dbx, dby, outward, toward)


def _shade_materials(
    rs: RenderScene,
    ctx: surface.ShadeContext,
    table: _TriangleTable,
    ct: np.ndarray,
    xn: np.ndarray,
    yn: np.ndarray,
    rows: np.ndarray,
    mat: np.ndarray,
    normal: tuple[np.ndarray, np.ndarray, np.ndarray],
    position: tuple[np.ndarray, np.ndarray, np.ndarray],
    radiance: list[np.ndarray],
) -> None:
    """Shade the samples in ``rows`` whose material has textures, emission or is standard."""
    index = np.flatnonzero(rows)
    samples = _surface_samples(ctx, table, ct[index], xn[index], yn[index])
    p_all = np.stack([c[index] for c in position], axis=1)
    n_all = np.stack([c[index] for c in normal], axis=1)
    for m_index in np.unique(mat[index]):
        sel = mat[index] == m_index
        material = rs.materials[int(m_index)]
        s = samples.subset(sel)
        sv = surface.evaluate_material(ctx, s, material)
        p = p_all[sel]
        # A normal map gives an outward normal; otherwise keep the interpolated normal exactly.
        n = sv.normal * s.toward[:, None] if "normal" in material.textures else n_all[sel]
        view = np.stack(surface.view_vectors(ctx, p[:, 0], p[:, 1], p[:, 2]), axis=1)
        if material.model == "simple":
            spec = np.full(p.shape[0], material.specular)
            part = surface.shade_simple(
                rs,
                sv.base,
                spec,
                np.full(p.shape[0], material.shininess),
                sv.occlusion if "occlusion" in material.textures else None,
                (n[:, 0], n[:, 1], n[:, 2]),
                (p[:, 0], p[:, 1], p[:, 2]),
                (view[:, 0], view[:, 1], view[:, 2]),
            )
            out = np.stack(part, axis=1)
        else:
            out = surface.shade_standard(rs, sv, material.specular, n, p, view)
        out = out + sv.emissive
        target = index[sel]
        for c in range(3):
            radiance[c][target] = out[:, c]


def _alpha_test(
    ctx: surface.ShadeContext,
    table: _TriangleTable,
    masked: np.ndarray,
    sw: int,
    sh: int,
    counts: dict[str, int],
):
    """``keep_fragment`` for the rasterizer: alpha-masked fragments below the cutoff are dropped."""
    rs = ctx.rs

    def keep_fragment(ids: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray | None:
        test = masked[ctx.MAT[ids]]
        if not test.any():
            return None
        keep = np.ones(ids.shape[0], dtype=bool)
        index = np.flatnonzero(test)
        ct = table.compact[ids[index]]
        xn = (cols[index] + 0.5) * (2.0 / sw) - 1.0
        yn = 1.0 - (rows[index] + 0.5) * (2.0 / sh)
        samples = _surface_samples(ctx, table, ct, xn, yn)
        mats = ctx.MAT[ids[index]]
        for m_index in np.unique(mats):
            sel = mats == m_index
            material = rs.materials[int(m_index)]
            sv = surface.evaluate_material(ctx, samples.subset(sel), material, alpha_only=True)
            keep[index[sel]] = sv.alpha >= material.alpha_cutoff
        counts["alpha_discarded"] += int(np.count_nonzero(~keep))
        return keep

    return keep_fragment
