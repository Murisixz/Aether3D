"""Directional-light shadow maps: which lights cast shadows and how each map is fitted.

This module is backend-neutral (NumPy, ``float64``): it turns a
:class:`~aether3d.rendering.scene_data.RenderScene` and
:class:`~aether3d.rendering.settings.ShadowSettings` into a
:class:`ShadowPlan`. The GPU backend renders one depth map per planned light
and compares against it while shading (see ``docs/GPU.md``, "Shadows").

Fitting (for each shadowed directional light, all in world units):

1. **Light space.** ``forward`` is the light's direction of travel, ``right =
   forward x up_hint`` (``up_hint`` is world +Z, or +Y for a vertical light)
   and ``up = right x forward``. Coordinates are measured from the camera's
   eye, the same camera-relative origin the GPU uses for everything else.
2. **Casters.** Every rendered instance's bounding box (its mesh's box
   transformed by its world matrix: 8 corners) is projected to light space;
   the union is the caster box ``C``.
3. **Receivers.** The camera's view frustum, cut off at the farthest
   instance corner in front of the camera (and at the camera's far plane),
   is projected to light space too (box ``F``). Only receivers inside both
   ``C`` and ``F`` can show a shadow, so the map covers ``C ∩ F`` across
   the light (``x``, ``y``) -- the part of the scene the camera can see --
   while its depth range covers all of ``C``, so casters outside the view
   still cast into it.
4. **Square texels.** The covered area is made square (the larger of its two
   sides), widened by ``pcf_radius + 2`` texels on every side for filtering,
   and divided into ``map_size x map_size`` texels.

Bias (against self-shadowing, "acne"): the receiving point moves along its
face normal, toward the light, by ``(sqrt(2) * (pcf_radius + 1) +
normal_offset)`` texels times the sine of the angle between the normal and
the light, and its depth is compared with a constant ``bias`` (in texels)
subtracted. For a flat receiver at angle ``theta``, an offset of ``k`` texels
times ``sin(theta)`` lifts the point ``k * tan(theta)`` texels above the
receiver's own depth wherever the filter looks within ``k`` texels -- and the
filter reads at most ``sqrt(2) * (pcf_radius + 1)`` texels away -- so flat
receivers never shadow themselves, at any angle. Shadows still start within
a few texels of their casters (little "peter-panning").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import RenderError
from .scene_data import RenderScene
from .settings import MAX_SHADOW_MAPS, ShadowSettings

PAD_TEXELS = 2
"""Extra texels around the fitted area on every side (beyond the PCF radius)."""
FOOTPRINT = math.sqrt(2.0)
"""Texels the filter reads beyond each tap (bilinear comparison, diagonally)."""


@dataclass(frozen=True)
class ShadowMap:
    """One directional light's shadow map: placement and bias, ready for a backend."""

    light: int  # index into RenderScene.lights
    name: str
    object_id: str | None
    size: int  # texels per side
    matrix: np.ndarray  # (4, 4) float64: camera-relative world -> shadow clip (x, y in [-1, 1])
    texel_size: float  # world units per texel
    depth_range: float  # world units mapped to shadow depth [0, 1]
    bias: float  # constant depth bias, in shadow depth units
    normal_offset: float  # receiver offset along its normal, world units (times sin(angle))
    extent: tuple[float, float, float]  # covered width, height and depth in world units

    def to_dict(self) -> dict[str, Any]:
        return {
            "light": self.name,
            "id": self.object_id,
            "map_size": self.size,
            "texel_size": float(self.texel_size),
            "extent": [float(v) for v in self.extent],
            "depth_range": float(self.depth_range),
        }


@dataclass(frozen=True)
class ShadowPlan:
    """The shadow maps to render for one image, and what was decided about them."""

    settings: ShadowSettings
    maps: tuple[ShadowMap, ...]
    diagnostics: tuple[Diagnostic, ...] = ()

    def layer_of(self, light_index: int) -> int:
        """The shadow-map layer of a light, or ``-1`` if it casts no shadow."""
        for layer, shadow in enumerate(self.maps):
            if shadow.light == light_index:
                return layer
        return -1

    def to_dict(self) -> dict[str, Any]:
        return {**self.settings.to_dict(), "maps": [m.to_dict() for m in self.maps]}


def _luminance(rgb: np.ndarray) -> float:
    return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])


def select_shadow_lights(
    rs: RenderScene, settings: ShadowSettings
) -> tuple[list[int], list[Diagnostic]]:
    """Indices (into ``rs.lights``) of the lights that cast shadows, and diagnostics."""
    diagnostics: list[Diagnostic] = []
    directional = [i for i, light in enumerate(rs.lights) if light.type == "directional"]
    points = [light.name for light in rs.lights if light.type == "point"]
    if settings.lights is None:
        chosen = sorted(directional, key=lambda i: (-_luminance(rs.lights[i].radiance), i))
        if len(chosen) > MAX_SHADOW_MAPS:
            left = [rs.lights[i].name for i in chosen[MAX_SHADOW_MAPS:]]
            diagnostics.append(
                Diagnostic(
                    "SHADOW_LIGHT_LIMIT",
                    WARNING,
                    f"only the {MAX_SHADOW_MAPS} brightest directional lights cast shadows; "
                    f"these do not: {', '.join(left)}",
                    "shadows.lights",
                    count=len(left),
                )
            )
            chosen = chosen[:MAX_SHADOW_MAPS]
        chosen.sort()
        if points:
            diagnostics.append(
                Diagnostic(
                    "POINT_LIGHT_SHADOWS_UNSUPPORTED",
                    INFO,
                    "point lights do not cast shadows yet: " + ", ".join(points),
                    "shadows",
                    count=len(points),
                )
            )
        return chosen, diagnostics
    chosen = []
    for k, ref in enumerate(settings.lights):
        loc = f"shadows.lights[{k}]"
        matches = [i for i, light in enumerate(rs.lights) if ref in (light.object_id, light.name)]
        if not matches:
            raise RenderError(
                "INVALID_SHADOWS",
                f"no visible light with id or name {ref!r}",
                loc,
                hint="name a light object of the scene (hidden lights do not shine)",
                details={"lights": [light.name for light in rs.lights]},
            )
        index = matches[0]
        if rs.lights[index].type != "directional":
            raise RenderError(
                "INVALID_SHADOWS",
                f"light {rs.lights[index].name!r} is a point light; only directional lights "
                "cast shadows in this version",
                loc,
            )
        if index not in chosen:
            chosen.append(index)
    chosen.sort()
    return chosen, diagnostics


def _light_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = direction / np.linalg.norm(direction)
    hint = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def mesh_bounds(rs: RenderScene) -> np.ndarray:
    """``(meshes, 2, 3)`` float64 bounding boxes (min, max) of every mesh."""
    bounds = np.empty((len(rs.meshes), 2, 3), dtype=np.float64)
    for k, mesh in enumerate(rs.meshes):
        bounds[k, 0] = mesh.positions.min(axis=0)
        bounds[k, 1] = mesh.positions.max(axis=0)
    return bounds


_CORNERS = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=bool)


def instance_corners(rs: RenderScene, bounds: np.ndarray | None = None) -> np.ndarray:
    """``(instances, 8, 3)`` world-space corners of every instance's bounding box.

    ``bounds`` -- the meshes' boxes (:func:`mesh_bounds`), if already known.
    """
    n = len(rs.instances)
    if n == 0:
        return np.zeros((0, 8, 3), dtype=np.float64)
    if bounds is None:
        bounds = mesh_bounds(rs)
    mesh = np.fromiter((inst.mesh for inst in rs.instances), dtype=np.int64, count=n)
    world = np.stack([inst.world for inst in rs.instances])
    boxes = np.where(_CORNERS[None], bounds[mesh, 1][:, None, :], bounds[mesh, 0][:, None, :])
    return np.einsum("nij,nkj->nki", world[:, :3, :3], boxes) + world[:, None, :3, 3]


def _frustum_corners(rs: RenderScene, aspect: float, far: float) -> np.ndarray:
    """Camera-relative corners of the view frustum between the near plane and ``far``."""
    cam = rs.camera
    hx, hy = cam.camera.half_extents(aspect)
    f, r, u = cam.frame.forward, cam.frame.right, cam.frame.up
    near = cam.camera.near
    points = []
    for depth in (near, far):
        sx, sy = (hx * depth, hy * depth) if cam.camera.is_perspective else (hx, hy)
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                points.append(depth * f + a * sx * r + b * sy * u)
    return np.array(points)


def plan_shadows(rs: RenderScene, settings: ShadowSettings, aspect: float) -> ShadowPlan:
    """Choose the shadowed lights and fit one shadow map to each (see the module docstring)."""
    chosen, diagnostics = select_shadow_lights(rs, settings)
    if not chosen or not rs.instances:
        return ShadowPlan(settings, (), tuple(diagnostics))
    eye = rs.camera.frame.position
    corners = instance_corners(rs).reshape(-1, 3) - eye  # camera-relative
    view_depth = corners @ rs.camera.frame.forward
    far = min(rs.camera.camera.far, float(view_depth.max()))
    far = max(far, rs.camera.camera.near)
    frustum = _frustum_corners(rs, aspect, far)
    size = settings.map_size
    maps = []
    for index in chosen:
        light = rs.lights[index]
        right, up, forward = _light_basis(light.direction)
        basis = np.stack([right, up, forward])
        caster = corners @ basis.T
        view = frustum @ basis.T
        lo, hi = caster.min(axis=0), caster.max(axis=0)
        x0, x1 = max(lo[0], view[:, 0].min()), min(hi[0], view[:, 0].max())
        y0, y1 = max(lo[1], view[:, 1].min()), min(hi[1], view[:, 1].max())
        if not (x1 > x0 and y1 > y0):  # the camera sees no receiver: cover every caster
            x0, x1, y0, y1 = lo[0], hi[0], lo[1], hi[1]
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        scale = max(float(np.abs(caster).max()), 1e-30)
        half = max(0.5 * max(x1 - x0, y1 - y0), 1e-9 * scale, 1e-30)
        # Widen by exactly pcf_radius + PAD_TEXELS texels per side (texels of the final map).
        half /= 1.0 - 2.0 * (settings.pcf_radius + PAD_TEXELS) / size
        texel = 2.0 * half / size
        depth_margin = max(texel, 1e-6 * scale)
        z0, z1 = lo[2] - depth_margin, hi[2] + depth_margin
        depth_range = z1 - z0
        m = np.zeros((4, 4), dtype=np.float64)
        m[0, :3], m[0, 3] = right / half, -cx / half
        m[1, :3], m[1, 3] = up / half, -cy / half
        m[2, :3], m[2, 3] = forward / depth_range, -z0 / depth_range
        m[3, 3] = 1.0
        if not np.all(np.isfinite(m)):
            diagnostics.append(
                Diagnostic(
                    "SHADOW_FIT_FAILED",
                    WARNING,
                    f"light {light.name!r}: the scene is too large or too small to fit a "
                    "shadow map; it renders without shadows",
                    "shadows",
                )
            )
            continue
        maps.append(
            ShadowMap(
                light=index,
                name=light.name,
                object_id=light.object_id,
                size=size,
                matrix=m,
                texel_size=texel,
                depth_range=depth_range,
                bias=settings.bias * texel / depth_range,
                normal_offset=(FOOTPRINT * (settings.pcf_radius + 1) + settings.normal_offset)
                * texel,
                extent=(2.0 * half, 2.0 * half, depth_range),
            )
        )
    return ShadowPlan(settings, tuple(maps), tuple(diagnostics))
