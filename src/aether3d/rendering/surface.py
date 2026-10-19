"""CPU surface shading: material evaluation (textures, normal maps, alpha) and lighting.

The CPU renderer shades every covered sample once (deferred). For each
sample it knows the triangle, the perspective-correct barycentrics ``b`` and
their exact screen-space derivatives ``db/dx``, ``db/dy`` (per sample of the
supersampled grid; see :func:`barycentric_derivatives`). Any attribute
interpolated from the triangle's corners -- UVs, object or world positions --
therefore comes with its derivatives, which select the mip level of image
textures and position bump-map samples.

Texture coordinates (per binding, :class:`~aether3d.materials.TextureBinding`):

* ``uv`` -- the mesh's UVs, transformed by the binding;
* ``object`` / ``world`` -- positions in that space, through a projection:
  ``xy``/``xz``/``yz`` (one plane), ``volume`` (the 3D point) or
  ``triplanar``: three planar projections -- x: ``(s_x y, z)``,
  y: ``(-s_y x, z)``, z: ``(s_z x, y)`` with ``s`` the sign of the
  triangle's geometric normal on that axis, so no projection is mirrored --
  blended with weights ``|n_axis|^blend`` of the surface normal.

Normal maps are tangent-space maps (+X right, +Y up the image, as glTF). A map read
through the mesh's UVs (with at most a uniform scale and an offset) uses the
MikkTSpace basis (:mod:`aether3d.rendering.tangents`): interpolated corner tangent T
and normal N, ``B = sign * cross(N, T)``, ``n = normalize(x T + y B + z N)``. Other
projections derive their tangent frame per triangle from its corner positions and
texture coordinates (the classic UV-gradient construction), made orthonormal against
the interpolated surface normal (Gram-Schmidt). Triplanar normal maps blend the three
perturbed normals.
Back faces flip the final normal (as glTF's sample viewer does for
double-sided materials).

Lighting (linear light; ``E`` = light color x intensity x attenuation):

* ``simple`` -- Stage 5 exactly: ``base (A ao + sum E max(n.l, 0)) + sum E
  specular max(n.h, 0)^shininess`` (the arithmetic is unchanged, so object
  colors render bit-identically to Aether3D 0.6).
* ``standard`` -- glTF 2.0's metallic-roughness BRDF, multiplied by pi so a
  white diffuse surface matches the simple model:
  ``E (n.l) ((1 - F) c_diff + pi F D V)`` with ``c_diff = base (1 - metallic)``,
  ``F0 = mix(0.08 specular, base, metallic)``, Schlick ``F``, GGX ``D`` and the
  height-correlated Smith ``V`` (``alpha = roughness^2``, roughness >= 0.045).
  Ambient light ``A`` is treated as a uniform environment: ``A ao (c_diff +
  F0 s + b)`` with Karis' analytic split-sum fit ``(s, b)``.
* Emission is added unlit, after lighting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..materials.graph import EvalContext, evaluate
from ..materials.sampling import lod, sample
from .materials import RenderImage, RenderMaterial, RenderTexture
from .scene_data import RenderScene

MIN_ROUGHNESS = 0.045
"""Perceptual roughness floor (a perfect mirror would make point highlights vanish)."""
TRIPLANAR_EPSILON = 1e-4
"""Triplanar projections with a smaller weight are skipped (on both backends)."""
MIN_LIGHT_DISTANCE = 1e-4
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])


@dataclass
class ShadeContext:
    """Per-render arrays the shader reads (all indexed by global vertex / triangle ids)."""

    rs: RenderScene
    P: np.ndarray  # (V, 3) world positions
    PO: np.ndarray  # (V, 3) object-space positions
    UV: np.ndarray  # (V, 2) texture coordinates (zeros where a mesh has none)
    TRI: np.ndarray  # (T, 3) vertex ids
    INST: np.ndarray  # (T,) instance of each triangle
    MAT: np.ndarray  # (T,) material of each triangle
    has_uvs: np.ndarray  # (instances,) bool
    linear: np.ndarray  # (instances, 3, 3) linear part of each world matrix
    eye: np.ndarray
    forward: np.ndarray
    perspective: bool
    step_x: float  # d(xn) per sample column
    step_y: float  # d(yn) per sample row (negative: rows go down)
    frames: Any = None  # MikkTSpace corner frames, built on first use (see world_frames)
    CO: Any = None  # (V, 4) linear vertex colours (ones where a mesh has none), or None
    has_colors: Any = None  # (instances,) bool, or None (no mesh has colours)


@dataclass
class Samples:
    """Surface data of a set of samples (all arrays have ``n`` rows)."""

    tri: np.ndarray  # global triangle ids
    inst: np.ndarray
    b: np.ndarray  # (n, 3) barycentrics
    db_dx: np.ndarray  # (n, 3)
    db_dy: np.ndarray  # (n, 3)
    normal: np.ndarray  # (n, 3) outward unit shading normal (world)
    toward: np.ndarray  # (n,) +1 or -1: turns outward normals toward the camera

    def subset(self, sel: np.ndarray) -> Samples:
        return Samples(
            self.tri[sel],
            self.inst[sel],
            self.b[sel],
            self.db_dx[sel],
            self.db_dy[sel],
            self.normal[sel],
            self.toward[sel],
        )


def barycentric_derivatives(
    a: np.ndarray, xn: np.ndarray, yn: np.ndarray, step_x: float, step_y: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barycentrics and their derivatives per sample column (x) and row (y).

    ``a`` holds the rows of ``adj(M)`` (``(n, 9)``): the unnormalized
    homogeneous barycentrics are ``B_k = a_k . (xn, yn, 1)`` and ``b_k = B_k /
    sum(B)``, so ``db_k/dxn = (a_k.x - b_k sum_j a_j.x) / sum(B)``.
    """
    B = np.stack(
        [a[:, 3 * k] * xn + a[:, 3 * k + 1] * yn + a[:, 3 * k + 2] for k in range(3)], axis=1
    )
    total = B.sum(axis=1)
    safe = np.where(np.abs(total) > 1e-300, total, 1.0)
    b = B / safe[:, None]
    ax = np.stack([a[:, 0], a[:, 3], a[:, 6]], axis=1)
    ay = np.stack([a[:, 1], a[:, 4], a[:, 7]], axis=1)
    dbx = (ax - b * ax.sum(axis=1, keepdims=True)) / safe[:, None] * step_x
    dby = (ay - b * ay.sum(axis=1, keepdims=True)) / safe[:, None] * step_y
    return b, dbx, dby


# --------------------------------------------------------------------------- #
# Texture coordinates
# --------------------------------------------------------------------------- #


@dataclass
class Projection:
    """Texture coordinates of one projection for a set of samples."""

    weight: np.ndarray | None  # (n,) triplanar weight, None for a single projection
    p: np.ndarray  # (n, 3) coordinates after the binding transform
    dx: np.ndarray  # (n, 3)
    dy: np.ndarray  # (n, 3)
    corners: np.ndarray  # (n, 3, 3) the triangle's corner coordinates (for tangent frames)


def _interpolate(corners: np.ndarray, s: Samples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.einsum("nk,nkc->nc", s.b, corners)
    dx = np.einsum("nk,nkc->nc", s.db_dx, corners)
    dy = np.einsum("nk,nkc->nc", s.db_dy, corners)
    return p, dx, dy


def _apply(transform: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """``transform`` (3 x 4) applied to ``(n, 3, 3)`` corner points."""
    return np.einsum("ij,nkj->nki", transform[:, :3], corners) + transform[:, 3]


TRIPLANAR_AXES = ((1, 2), (0, 2), (0, 1))
"""Coordinates of the x, y and z projections (sign-corrected, see the module docstring)."""


def _planar(corners: np.ndarray, plane: str) -> np.ndarray:
    a, b = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
    out = np.zeros_like(corners)
    out[..., 0] = corners[..., a]
    out[..., 1] = corners[..., b]
    return out


def projections(ctx: ShadeContext, s: Samples, rt: RenderTexture) -> list[Projection]:
    """The coordinates ``rt`` reads at ``s`` (one projection, or up to three triplanar)."""
    tv = ctx.TRI[s.tri]
    if rt.coordinates == "uv":
        uv = ctx.UV[tv]  # (n, 3, 2)
        corners = np.zeros((tv.shape[0], 3, 3))
        corners[..., :2] = uv
        corners = _apply(rt.transform, corners)
        return [Projection(None, *_interpolate(corners, s), corners)]
    points = ctx.PO[tv] if rt.coordinates == "object" else ctx.P[tv]
    if rt.projection == "volume":
        corners = _apply(rt.transform, points)
        return [Projection(None, *_interpolate(corners, s), corners)]
    if rt.projection != "triplanar":
        corners = _apply(rt.transform, _planar(points, rt.projection))
        return [Projection(None, *_interpolate(corners, s), corners)]
    # Triplanar: weights from the shading normal, signs from the geometric normal
    # (both in the coordinate space).
    n = s.normal if rt.coordinates == "world" else _to_object(ctx, s.inst, s.normal)
    g = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    sign = np.where(g >= 0.0, 1.0, -1.0)
    w = np.abs(n) ** rt.blend
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    out = []
    for axis, (i, j) in enumerate(TRIPLANAR_AXES):
        keep = w[:, axis] > TRIPLANAR_EPSILON
        corners = np.zeros_like(points)
        flip = -sign[:, axis] if axis == 1 else sign[:, axis]
        corners[..., 0] = flip[:, None] * points[..., i]
        corners[..., 1] = points[..., j]
        corners = _apply(rt.transform, corners)
        p, dx, dy = _interpolate(corners, s)
        out.append(Projection(np.where(keep, w[:, axis], 0.0), p, dx, dy, corners))
    return out


def _to_object(ctx: ShadeContext, inst: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """World normals to object space (``M^T n``, normalized): for triplanar weights."""
    m = ctx.linear[inst]
    v = np.einsum("nji,nj->ni", m, normal)
    length = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 1e-300)


# --------------------------------------------------------------------------- #
# Texture values
# --------------------------------------------------------------------------- #


def _sample_image(image: RenderImage, p: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    lam = lod(image.chain, dx[:, 0], dx[:, 1], dy[:, 0], dy[:, 1]) if image.mipmaps else None
    return sample(image.chain, p[:, 0], p[:, 1], wrap=image.wrap, filter=image.filter, lam=lam)


def _graph_sampler(rt: RenderTexture):
    def sample_image(ref: str, color_space: str, p, dpdx, dpdy):
        return _sample_image(rt.images[(ref, color_space)], p, dpdx, dpdy)

    return sample_image


def texture_rgba(rt: RenderTexture, proj: Projection, sel: np.ndarray | None = None) -> np.ndarray:
    """``(n, 4)`` linear RGBA of ``rt`` at a projection's coordinates (rows ``sel`` only)."""
    p, dx, dy = proj.p, proj.dx, proj.dy
    if sel is not None:
        p, dx, dy = p[sel], dx[sel], dy[sel]
    if rt.kind == "image":
        return _sample_image(rt.image, p, dx, dy)
    return evaluate(rt.graph, EvalContext(p, dx, dy, _graph_sampler(rt)))


def channel(rgba: np.ndarray, name: str) -> np.ndarray:
    if name == "luminance":
        return rgba[:, :3] @ LUMINANCE
    return rgba[:, "rgba".index(name)]


def blended_rgba(rt: RenderTexture, projs: list[Projection]) -> np.ndarray:
    """A texture's value, blending triplanar projections by their weights."""
    if len(projs) == 1:
        return texture_rgba(rt, projs[0])
    n = projs[0].p.shape[0]
    out = np.zeros((n, 4))
    for proj in projs:
        keep = proj.weight > 0.0
        if keep.any():
            out[keep] += texture_rgba(rt, proj, keep) * proj.weight[keep, None]
    return out


def _tangent_frame(
    ctx: ShadeContext, s: Samples, corners: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal ``(T, B)`` per sample from the triangle's texture-coordinate gradients,
    and whether the frame is valid (non-degenerate coordinates)."""
    tv = ctx.TRI[s.tri]
    p0, p1, p2 = ctx.P[tv[:, 0]], ctx.P[tv[:, 1]], ctx.P[tv[:, 2]]
    du1 = corners[:, 1, :2] - corners[:, 0, :2]
    du2 = corners[:, 2, :2] - corners[:, 0, :2]
    det = du1[:, 0] * du2[:, 1] - du2[:, 0] * du1[:, 1]
    valid = np.abs(det) > 1e-20
    inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)[:, None]
    e1, e2 = p1 - p0, p2 - p0
    t = (e1 * du2[:, 1:2] - e2 * du1[:, 1:2]) * inv
    b = (e2 * du1[:, 0:1] - e1 * du2[:, 0:1]) * inv
    t = t - normal * (t * normal).sum(axis=1, keepdims=True)
    t_len = np.linalg.norm(t, axis=1, keepdims=True)
    valid &= t_len[:, 0] > 1e-20
    t = np.divide(t, t_len, out=np.zeros_like(t), where=t_len > 1e-20)
    bitangent = np.cross(normal, t)
    handed = np.where((bitangent * b).sum(axis=1) < 0.0, -1.0, 1.0)
    return t, bitangent * handed[:, None], valid


def world_frames(ctx: ShadeContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-space MikkTSpace frames of every triangle corner of the render, built once:
    ``(tangents (T, 3, 3) unit, signs (T, 3), normals (T, 3, 3) unit outward)`` and which
    triangles have them (``(T,)`` bool: meshes with UVs)."""
    if ctx.frames is not None:
        return ctx.frames
    from .tangents import corner_tangents

    total = ctx.TRI.shape[0]
    tan = np.zeros((total, 3, 3))
    sign = np.ones((total, 3))
    nrm = np.zeros((total, 3, 3))
    ok = np.zeros(total, dtype=bool)
    start = 0
    for inst in ctx.rs.instances:
        mesh = ctx.rs.meshes[inst.mesh]
        count = mesh.triangles.shape[0]
        sl = slice(start, start + count)
        start += count
        t = corner_tangents(mesh)
        if t is None:
            continue
        lin = inst.world[:3, :3]
        v = t[..., :3].astype(np.float64) @ lin.T
        length = np.linalg.norm(v, axis=2, keepdims=True)
        tan[sl] = np.divide(v, length, out=np.zeros_like(v), where=length > 1e-300)
        flip = -1.0 if inst.mirrored else 1.0
        sign[sl] = np.where(t[..., 3] < 0, -1.0, 1.0) * flip
        tri = mesh.triangles.astype(np.int64)
        if mesh.normals is not None:
            n = mesh.normals.astype(np.float64)[tri] @ inst.normal_matrix.T
        else:
            p = mesh.positions.astype(np.float64)[tri] @ lin.T
            g = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]) * flip
            n = np.repeat(g[:, None, :], 3, axis=1)
        length = np.linalg.norm(n, axis=2, keepdims=True)
        nrm[sl] = np.divide(n, length, out=np.zeros_like(n), where=length > 1e-300)
        ok[sl] = True
    ctx.frames = (tan, sign, nrm, ok)
    return ctx.frames


def mikk_normal(
    ctx: ShadeContext, s: Samples, rgba: np.ndarray, strength: float
) -> tuple[np.ndarray, np.ndarray]:
    """The normal mapped in the MikkTSpace basis (see the module docstring) and where the
    basis exists (non-zero tangent)."""
    tan, sign, nrm, ok = world_frames(ctx)
    t = np.einsum("nk,nkc->nc", s.b, tan[s.tri])
    n = np.einsum("nk,nkc->nc", s.b, nrm[s.tri])
    w = np.where((s.b * sign[s.tri]).sum(axis=1) < 0.0, -1.0, 1.0)
    b = w[:, None] * np.cross(n, t)
    tn = rgba[:, :3] * 2.0 - 1.0
    tn[:, :2] *= strength
    mapped = t * tn[:, 0:1] + b * tn[:, 1:2] + n * tn[:, 2:3]
    length = np.linalg.norm(mapped, axis=1, keepdims=True)
    valid = ok[s.tri] & (np.linalg.norm(t, axis=1) > 1e-12) & (length[:, 0] > 1e-12)
    return np.where(valid[:, None], mapped / np.maximum(length, 1e-300), s.normal), valid


def perturbed_normal(
    ctx: ShadeContext, s: Samples, rt: RenderTexture, projs: list[Projection], strength: float
) -> np.ndarray:
    """The outward normal tilted by a tangent-space normal map (blended for triplanar)."""
    n = s.normal
    out = np.zeros_like(n)
    for proj in projs:
        keep = np.ones(n.shape[0], dtype=bool) if proj.weight is None else proj.weight > 0.0
        if not keep.any():
            continue
        sub = s.subset(keep)
        rgba = texture_rgba(rt, proj, keep)
        if rt.coordinates == "uv" and rt.mikktspace:
            mapped, _ = mikk_normal(ctx, sub, rgba, strength)
            w = 1.0 if proj.weight is None else proj.weight[keep, None]
            out[keep] += mapped * w
            continue
        t, b, valid = _tangent_frame(ctx, sub, proj.corners[keep], sub.normal)
        tn = rgba[:, :3] * 2.0 - 1.0
        tn[:, :2] *= strength
        mapped = t * tn[:, 0:1] + b * tn[:, 1:2] + sub.normal * tn[:, 2:3]
        length = np.linalg.norm(mapped, axis=1, keepdims=True)
        ok = valid & (length[:, 0] > 1e-12)
        mapped = np.where(ok[:, None], mapped / np.maximum(length, 1e-300), sub.normal)
        w = 1.0 if proj.weight is None else proj.weight[keep, None]
        out[keep] += mapped * w
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(length > 1e-12, out / np.maximum(length, 1e-300), n)


# --------------------------------------------------------------------------- #
# Materials
# --------------------------------------------------------------------------- #


@dataclass
class SurfaceValues:
    base: np.ndarray  # (n, 3) linear
    alpha: np.ndarray  # (n,)
    metallic: np.ndarray
    roughness: np.ndarray
    occlusion: np.ndarray  # (n,) multiplies ambient light
    emissive: np.ndarray  # (n, 3)
    normal: np.ndarray  # (n, 3) outward


def _usable(ctx: ShadeContext, s: Samples, rt: RenderTexture) -> np.ndarray | None:
    """Rows where ``rt`` applies (``None``: all): UV bindings need meshes with UVs."""
    if rt.needs_uvs:
        ok = ctx.has_uvs[s.inst]
        return None if ok.all() else ok
    return None


def evaluate_material(
    ctx: ShadeContext, s: Samples, m: RenderMaterial, *, alpha_only: bool = False
) -> SurfaceValues:
    """Factors x textures of ``m`` at samples ``s`` (all of them use ``m``)."""
    n = s.tri.shape[0]
    values = SurfaceValues(
        base=np.tile(m.base_color, (n, 1)),
        alpha=np.full(n, m.opacity),
        metallic=np.full(n, m.metallic),
        roughness=np.full(n, m.roughness),
        occlusion=np.ones(n),
        emissive=np.tile(m.emissive, (n, 1)),
        normal=s.normal,
    )
    if ctx.CO is not None:
        rows = ctx.has_colors[s.inst]
        if rows.any():
            tv = ctx.TRI[s.tri[rows]]
            b = s.b[rows]
            co = ctx.CO
            c = b[:, 0:1] * co[tv[:, 0]] + b[:, 1:2] * co[tv[:, 1]] + b[:, 2:3] * co[tv[:, 2]]
            values.base[rows] *= c[:, :3]
            values.alpha[rows] *= c[:, 3]
    slots = ("base_color", "opacity") if alpha_only else tuple(m.textures)
    cache: dict[tuple, np.ndarray] = {}
    for slot in slots:
        rt = m.textures.get(slot)
        if rt is None:
            continue
        rows = _usable(ctx, s, rt)
        if rows is not None and not rows.any():
            continue
        sub = s if rows is None else s.subset(rows)
        projs = projections(ctx, sub, rt)
        if slot == "normal":
            mapped = perturbed_normal(ctx, sub, rt, projs, m.normal_strength)
            if rows is None:
                values.normal = mapped
            else:
                values.normal = values.normal.copy()
                values.normal[rows] = mapped
            continue
        key = rt.key
        rgba = cache.get(key) if rows is None else None
        if rgba is None:
            rgba = blended_rgba(rt, projs)
            if rows is None:
                cache[key] = rgba
        target = slice(None) if rows is None else rows
        if slot == "base_color":
            values.base[target] *= rgba[:, :3]
            if rt.kind == "image":
                values.alpha[target] *= rgba[:, 3]
        elif slot == "emissive":
            values.emissive[target] *= rgba[:, :3]
        elif slot == "opacity":
            values.alpha[target] *= channel(rgba, rt.channel)
        elif slot == "metallic":
            values.metallic[target] *= channel(rgba, rt.channel)
        elif slot == "roughness":
            values.roughness[target] *= channel(rgba, rt.channel)
        elif slot == "occlusion":
            ao = channel(rgba, rt.channel)
            values.occlusion[target] = 1.0 + m.occlusion_strength * (ao - 1.0)
    return values


# --------------------------------------------------------------------------- #
# Lighting
# --------------------------------------------------------------------------- #


def _light_vectors(light, px, py, pz):
    """Unit direction to the light and attenuation (None: 1) at points (point and spot
    lights; a spot's cone masks the attenuation)."""
    if light.type == "directional":
        lx, ly, lz = (-light.direction).tolist()
        return lx, ly, lz, None
    lx, ly, lz = light.position[0] - px, light.position[1] - py, light.position[2] - pz
    dist = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = (c / np.maximum(dist, 1e-300) for c in (lx, ly, lz))
    dist = np.maximum(dist, MIN_LIGHT_DISTANCE)
    atten = 1.0 / (dist * dist) if light.falloff == "inverse_square" else np.ones_like(dist)
    if light.range is not None:
        ratio = dist / light.range
        atten = atten * np.clip(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0) ** 2
    if light.type == "spot":
        dx, dy, dz = light.direction.tolist()
        atten = atten * light.spot_mask(-(lx * dx + ly * dy + lz * dz))
    return lx, ly, lz, atten


AREA_POLYGON = 16
"""Corners of the polygon a disk or ellipse light is integrated as on the rasterizers."""
AREA_SCALE = math.sqrt(math.pi / (0.5 * AREA_POLYGON * math.sin(2.0 * math.pi / AREA_POLYGON)))
"""The polygon's corners at this many radii: its area is then the ellipse's."""


def area_corners(light) -> np.ndarray:
    """``(k, 3)`` world corners of an area light's polygon, in order (4 for squares and
    rectangles, :data:`AREA_POLYGON` on disks and ellipses)."""
    c, ax, ay = light.position, light.axis_x, light.axis_y
    if light.shape in ("disk", "ellipse"):
        t = np.arange(AREA_POLYGON) * (2.0 * math.pi / AREA_POLYGON)
        # Enlarged to the ellipse's own area (pi ab): the polygon carries all of its light.
        k = AREA_SCALE
        return c + (k * np.cos(t))[:, None] * ax + (k * np.sin(t))[:, None] * ay
    return np.array([c - ax - ay, c + ax - ay, c + ax + ay, c - ax + ay])


SPREAD_QUADRATURE = 12
"""Points per side of the midpoint rule spread area lights are integrated with (rasterizers)."""


def spread_irradiance(light, px, py, pz, nx, ny, nz) -> np.ndarray:
    """``G`` of a spread area light (its radiance factor included): the midpoint rule on a
    :data:`SPREAD_QUADRATURE`-square grid over the part of the light that reaches each
    point (``RenderLight.spread_window``), each grid point weighted by the spread factor
    toward the point. An
    approximation of the integral (the path tracers sample it without one); cells across
    the edge of a disk or ellipse count the share of them inside it."""
    p = np.stack([px, py, pz], axis=1)
    u0, u1, w0, w1, area, ok = light.spread_window(p)
    k = SPREAD_QUADRATURE
    du, dw = (u1 - u0) / k, (w1 - w0) / k
    c, ax, ay, nl = light.position, light.axis_x, light.axis_y, light.direction
    round_ = light.shape in ("disk", "ellipse")
    total = np.zeros(px.shape[0])
    for i in range(k):
        u = u0 + (i + 0.5) * du
        for j in range(k):
            w = w0 + (j + 0.5) * dw
            vx = c[0] + u * ax[0] + w * ay[0] - px
            vy = c[1] + u * ax[1] + w * ay[1] - py
            vz = c[2] + u * ax[2] + w * ay[2] - pz
            d2 = np.maximum(vx * vx + vy * vy + vz * vz, MIN_LIGHT_DISTANCE * MIN_LIGHT_DISTANCE)
            d = np.sqrt(d2)
            cos_p = np.maximum((vx * nx + vy * ny + vz * nz) / d, 0.0)
            cos_l = -(vx * nl[0] + vy * nl[1] + vz * nl[2]) / d
            term = light.spread_factor(cos_l) * np.maximum(cos_l, 0.0) * cos_p / d2
            if round_:
                # The share of the cell inside the ellipse (the unit circle here), from the
                # signed distance of its centre to the edge over its width across the edge.
                r = np.sqrt(u * u + w * w)
                rs = np.maximum(r, 1e-300)
                across = np.abs(u / rs) * du + np.abs(w / rs) * dw
                term = term * np.clip(0.5 - (r - 1.0) / np.maximum(across, 1e-300), 0.0, 1.0)
            total += term
    return np.where(ok, total * (area / (k * k)), 0.0)


def area_irradiance(light, px, py, pz, nx, ny, nz) -> np.ndarray:
    """``G``: the irradiance an area light of unit radiance gives the points (normals
    ``n``), unshadowed -- Lambert's polygon formula ``1/2 sum theta_i (n . gamma_i)`` over
    the polygon clipped by the points' tangent planes (exact for squares and rectangles;
    disks and ellipses as :data:`AREA_POLYGON`-gons). Zero behind the one-sided light.
    Spread lights: :func:`spread_irradiance`."""
    if light.spread_tan > 0.0:
        return spread_irradiance(light, px, py, pz, nx, ny, nz)
    corners = area_corners(light)
    k = corners.shape[0]
    count = px.shape[0]
    rel = [
        (corners[i, 0] - px, corners[i, 1] - py, corners[i, 2] - pz) for i in range(k)
    ]  # vectors from the points to the corners
    dist = [vx * nx + vy * ny + vz * nz for vx, vy, vz in rel]  # heights above the planes
    total = np.zeros(count)
    exit_x = [np.zeros(count) for _ in range(3)]
    entry_x = [np.zeros(count) for _ in range(3)]
    clipped = np.zeros(count, bool)

    def edge(a, b, live):
        # theta (n . gamma) of the edge a -> b, where live; stable for small angles
        ax_, ay_, az_ = a
        bx_, by_, bz_ = b
        cx = ay_ * bz_ - az_ * by_
        cy = az_ * bx_ - ax_ * bz_
        cz = ax_ * by_ - ay_ * bx_
        cross = np.sqrt(cx * cx + cy * cy + cz * cz)
        dot = ax_ * bx_ + ay_ * by_ + az_ * bz_
        theta = np.arctan2(cross, dot)
        ngam = (cx * nx + cy * ny + cz * nz) / np.maximum(cross, 1e-300)
        return np.where(live & (cross > 0.0), theta * ngam, 0.0)

    for i in range(k):
        j = (i + 1) % k
        a, b = rel[i], rel[j]
        da, db = dist[i], dist[j]
        above_a, above_b = da > 0.0, db > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(above_a != above_b, da / np.where(da != db, da - db, 1.0), 0.0)
        x = tuple(a[c] + f * (b[c] - a[c]) for c in range(3))
        # the part of the edge above the plane
        start = tuple(np.where(above_a, a[c], x[c]) for c in range(3))
        end = tuple(np.where(above_b, b[c], x[c]) for c in range(3))
        total += edge(start, end, above_a | above_b)
        leaving = above_a & ~above_b
        entering = ~above_a & above_b
        for c in range(3):
            exit_x[c] = np.where(leaving, x[c], exit_x[c])
            entry_x[c] = np.where(entering, x[c], entry_x[c])
        clipped |= leaving
    total += edge(tuple(exit_x), tuple(entry_x), clipped)  # along the horizon
    # One-sided: only points in front of the light (the side its normal faces).
    cx_, cy_, cz_ = light.position
    nl = light.direction
    front = (px - cx_) * nl[0] + (py - cy_) * nl[1] + (pz - cz_) * nl[2] > 0.0
    return np.where(front, 0.5 * np.abs(total), 0.0)


def area_highlight(light, px, py, pz, nx, ny, nz, vx, vy, vz):
    """The representative point of an area light for glossy highlights (Karis 2013):
    where the mirror direction meets the light's plane, clamped into its shape. Returns
    the unit direction to it, ``cos`` at the light (one-sided) over the squared distance,
    and the distance (for widening the lobe). Spread lights: clamped into the part of the
    light that reaches the point, and weighted by the spread factor there."""
    ndv = nx * vx + ny * vy + nz * vz
    rx, ry, rz = 2.0 * ndv * nx - vx, 2.0 * ndv * ny - vy, 2.0 * ndv * nz - vz
    c = light.position
    nl = light.direction
    ox, oy, oz = c[0] - px, c[1] - py, c[2] - pz  # point -> centre
    denom = rx * nl[0] + ry * nl[1] + rz * nl[2]
    t = (ox * nl[0] + oy * nl[1] + oz * nl[2]) / np.where(np.abs(denom) > 1e-12, denom, 1e-12)
    t = np.where((denom < -1e-12) & (t > 0.0), t, 0.0)  # toward the front of the light
    qx, qy, qz = rx * t - ox, ry * t - oy, rz * t - oz  # hit point relative to the centre
    ax, ay = light.axis_x, light.axis_y
    la, lb = float(ax @ ax), float(ay @ ay)
    u = (qx * ax[0] + qy * ax[1] + qz * ax[2]) / la
    w = (qx * ay[0] + qy * ay[1] + qz * ay[2]) / lb
    if light.spread_tan > 0.0:
        u0, u1, w0, w1, _area, ok = light.spread_window(np.stack([px, py, pz], axis=1))
        u, w = np.clip(u, u0, u1), np.clip(w, w0, w1)
    if light.shape in ("disk", "ellipse"):
        r = np.sqrt(u * u + w * w)
        scale = np.where(r > 1.0, 1.0 / np.maximum(r, 1e-300), 1.0)
        u, w = u * scale, w * scale
    else:
        u, w = np.clip(u, -1.0, 1.0), np.clip(w, -1.0, 1.0)
    lx = ox + u * ax[0] + w * ay[0]
    ly = oy + u * ax[1] + w * ay[1]
    lz = oz + u * ax[2] + w * ay[2]
    d2 = np.maximum(lx * lx + ly * ly + lz * lz, MIN_LIGHT_DISTANCE * MIN_LIGHT_DISTANCE)
    d = np.sqrt(d2)
    lx, ly, lz = lx / d, ly / d, lz / d
    cos_l = np.maximum(-(lx * nl[0] + ly * nl[1] + lz * nl[2]), 0.0)
    if light.spread_tan > 0.0:
        return lx, ly, lz, np.where(ok, light.spread_factor(cos_l) * cos_l / d2, 0.0), d
    return lx, ly, lz, cos_l / d2, d


def _hemisphere(rs: RenderScene, nz: np.ndarray, n=None) -> np.ndarray | None:
    """Per-sample ambient light: an image environment's (its spherical harmonics at the
    normals ``n``), the hemisphere ambient's, or ``None`` (uniform ambient)."""
    sh = getattr(rs, "ambient_sh", None)
    if sh is not None:
        from .envimage import sh_ambient

        return sh_ambient(sh, np.stack(n, axis=1))
    atm = getattr(rs, "atmosphere", None)
    if atm is None or not atm["settings"].enabled or atm["settings"].ambient != "hemisphere":
        return None
    from .atmosphere import ambient_at, resolve

    return ambient_at(resolve(rs), rs.ambient, nz)


def linked(light, inst: np.ndarray | None) -> np.ndarray | None:
    """Light linking: whether ``light`` lights each sample (of instances ``inst``), or
    ``None`` when it lights them all."""
    if light.receivers is None or inst is None:
        return None
    return light.receivers[inst]


def shade_simple(
    rs: RenderScene,
    base: np.ndarray,
    spec: np.ndarray,
    shin: np.ndarray,
    occlusion: np.ndarray | None,
    n: tuple[np.ndarray, np.ndarray, np.ndarray],
    p: tuple[np.ndarray, np.ndarray, np.ndarray],
    v: tuple | None,
    inst: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Stage 5 shading, with the same arithmetic in the same order (bitwise stable).
    ``inst``: the samples' instances, for light linking."""
    nx, ny, nz = n
    px, py, pz = p
    count = nx.shape[0]
    any_spec = bool(np.any(spec > 0.0))
    hemi = _hemisphere(rs, nz, (nx, ny, nz))
    if hemi is not None:  # (scene.world.atmosphere ambient="hemisphere")
        diffuse = [hemi[:, c] if occlusion is None else hemi[:, c] * occlusion for c in range(3)]
    elif occlusion is None:
        diffuse = [np.full(count, rs.ambient[c]) for c in range(3)]
    else:
        diffuse = [rs.ambient[c] * occlusion for c in range(3)]
    highlight = [np.zeros(count) for _ in range(3)] if any_spec else None
    if any_spec:
        vx, vy, vz = v
    for light in rs.lights:
        mask = linked(light, inst)
        if mask is not None and not mask.any():
            continue
        if light.type == "area":
            # Diffuse: the polygon's irradiance (G / A per unit intensity, radiance pi I / A);
            # highlight: toward the representative point, as a point light there.
            g = area_irradiance(light, px, py, pz, nx, ny, nz) / light.area
            if mask is not None:
                g = np.where(mask, g, 0.0)
            for c in range(3):
                diffuse[c] += light.radiance[c] * g
            if any_spec:
                lx, ly, lz, atten, _d = area_highlight(light, px, py, pz, nx, ny, nz, vx, vy, vz)
                ndl = np.maximum(nx * lx + ny * ly + nz * lz, 0.0)
                hx_, hy_, hz_ = lx + vx, ly + vy, lz + vz
                h_len = np.maximum(np.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_), 1e-300)
                ndh = np.maximum((nx * hx_ + ny * hy_ + nz * hz_) / h_len, 0.0)
                term = spec * np.where(ndl > 0.0, np.power(ndh, shin), 0.0) * atten
                if mask is not None:
                    term = np.where(mask, term, 0.0)
                for c in range(3):
                    highlight[c] += light.radiance[c] * term
            continue
        lx, ly, lz, atten = _light_vectors(light, px, py, pz)
        ndl = np.maximum(nx * lx + ny * ly + nz * lz, 0.0)
        weight = ndl if atten is None else ndl * atten
        if mask is not None:
            weight = np.where(mask, weight, 0.0)
        for c in range(3):
            diffuse[c] += light.radiance[c] * weight
        if any_spec:
            hx_, hy_, hz_ = lx + vx, ly + vy, lz + vz
            h_len = np.maximum(np.sqrt(hx_ * hx_ + hy_ * hy_ + hz_ * hz_), 1e-300)
            ndh = np.maximum((nx * hx_ + ny * hy_ + nz * hz_) / h_len, 0.0)
            term = spec * np.where(ndl > 0.0, np.power(ndh, shin), 0.0)
            if atten is not None:
                term = term * atten
            if mask is not None:
                term = np.where(mask, term, 0.0)
            for c in range(3):
                highlight[c] += light.radiance[c] * term
    radiance = [base[:, c] * diffuse[c] for c in range(3)]
    if any_spec:
        radiance = [radiance[c] + highlight[c] for c in range(3)]
    return radiance


def env_brdf(roughness: np.ndarray, ndv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Karis' analytic fit of the split-sum environment BRDF: ``F0 * scale + bias``."""
    r = roughness
    c0 = (-1.0, -0.0275, -0.572, 0.022)
    c1 = (1.0, 0.0425, 1.04, -0.04)
    rx = r * c0[0] + c1[0]
    ry = r * c0[1] + c1[1]
    rz = r * c0[2] + c1[2]
    rw = r * c0[3] + c1[3]
    a004 = np.minimum(rx * rx, np.exp2(-9.28 * ndv)) * rx + ry
    return a004 * -1.04 + rz, a004 * 1.04 + rw


def shade_standard(
    rs: RenderScene,
    sv: SurfaceValues,
    specular: float | np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    inst: np.ndarray | None = None,
) -> np.ndarray:
    """``(n, 3)`` radiance of the metallic-roughness model (see the module docstring);
    ``inst``: the samples' instances, for light linking."""
    metallic = np.clip(sv.metallic, 0.0, 1.0)[:, None]
    rough = np.clip(sv.roughness, MIN_ROUGHNESS, 1.0)
    alpha = rough * rough
    a2 = (alpha * alpha)[:, None]
    base = sv.base
    c_diff = base * (1.0 - metallic)
    f0 = (0.08 * np.asarray(specular, dtype=np.float64)) * (1.0 - metallic) + base * metallic
    ndv = np.maximum((n * v).sum(axis=1), 1e-4)[:, None]
    scale, bias = env_brdf(rough, ndv[:, 0])
    hemi = _hemisphere(rs, n[:, 2], (n[:, 0], n[:, 1], n[:, 2]))
    amb = rs.ambient if hemi is None else hemi
    out = amb * sv.occlusion[:, None] * (c_diff + f0 * scale[:, None] + bias[:, None])
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    for light in rs.lights:
        mask = linked(light, inst)
        if mask is not None and not mask.any():
            continue
        if light.type == "area":
            part = _area_standard(light, n, p, v, c_diff, f0, alpha, a2, ndv)
            out += part if mask is None else np.where(mask[:, None], part, 0.0)
            continue
        lx, ly, lz, atten = _light_vectors(light, px, py, pz)
        if atten is None:  # directional: one direction for every sample
            l_vec = np.broadcast_to(np.array([lx, ly, lz], dtype=np.float64), n.shape)
        else:
            l_vec = np.stack([lx, ly, lz], axis=1)
        ndl = (n * l_vec).sum(axis=1)[:, None]
        lit = ndl[:, 0] > 0.0
        if mask is not None:
            lit &= mask
        if not lit.any():
            continue
        h = l_vec + v
        h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
        ndh = np.maximum((n * h).sum(axis=1), 0.0)[:, None]
        vdh = np.maximum((v * h).sum(axis=1), 0.0)[:, None]
        ndl = np.maximum(ndl, 0.0)
        fresnel = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
        d = ndh * ndh * (a2 - 1.0) + 1.0
        dist = a2 / (math.pi * d * d)
        vis = 0.5 / np.maximum(
            ndl * np.sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * np.sqrt(ndl * ndl * (1.0 - a2) + a2),
            1e-300,
        )
        brdf = (1.0 - fresnel) * c_diff + math.pi * fresnel * dist * vis
        weight = ndl if atten is None else ndl * np.asarray(atten)[:, None]
        out += np.where(lit[:, None], light.radiance * weight * brdf, 0.0)
    return out


def _area_standard(light, n, p, v, c_diff, f0, alpha, a2, ndv) -> np.ndarray:
    """An area light on the metallic-roughness model: the exact unshadowed diffuse
    irradiance of its polygon, and a GGX highlight toward its representative point with
    the lobe widened by the light's apparent size (Karis 2013, an approximation; the
    path tracer integrates it exactly)."""
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    g = area_irradiance(light, px, py, pz, nx, ny, nz) / light.area
    lx, ly, lz, atten, d = area_highlight(light, px, py, pz, nx, ny, nz, vx, vy, vz)
    l_vec = np.stack([lx, ly, lz], axis=1)
    h = l_vec + v
    h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-300)
    ndl = np.maximum((n * l_vec).sum(axis=1), 0.0)[:, None]
    ndh = np.maximum((n * h).sum(axis=1), 0.0)[:, None]
    vdh = np.maximum((v * h).sum(axis=1), 0.0)[:, None]
    fresnel = f0 + (1.0 - f0) * (1.0 - vdh) ** 5
    # A source of radius sqrt(A / pi) at distance d widens the lobe: alpha' (energy kept
    # by (alpha / alpha')^2).
    a = alpha[:, None]
    wide = np.clip(a + math.sqrt(light.area / math.pi) / (2.0 * d)[:, None], 0.0, 1.0)
    a2w = wide * wide
    dd = ndh * ndh * (a2w - 1.0) + 1.0
    dist = a2w / (math.pi * dd * dd) * (a * a) / np.maximum(a2w, 1e-300)
    vis = 0.5 / np.maximum(
        ndl * np.sqrt(ndv * ndv * (1.0 - a2) + a2) + ndv * np.sqrt(ndl * ndl * (1.0 - a2) + a2),
        1e-300,
    )
    fresnel_v = f0 + (1.0 - f0) * (1.0 - ndv) ** 5  # diffuse: the view's Fresnel
    diffuse = (1.0 - fresnel_v) * c_diff * g[:, None]
    spec = math.pi * fresnel * dist * vis * ndl * atten[:, None]
    lit = ndl[:, 0] > 0.0
    return light.radiance * diffuse + np.where(lit[:, None], light.radiance * spec, 0.0)


def view_vectors(
    ctx: ShadeContext, px: np.ndarray, py: np.ndarray, pz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ctx.perspective:
        ex, ey, ez = ctx.eye
        vx, vy, vz = ex - px, ey - py, ez - pz
        v_len = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-300)
        return vx / v_len, vy / v_len, vz / v_len
    fx, fy, fz = ctx.forward
    return (np.full_like(px, -fx), np.full_like(px, -fy), np.full_like(px, -fz))
