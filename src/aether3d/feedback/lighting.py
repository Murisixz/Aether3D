"""How a view's subject is lit (0.20.0): what the ``lighting`` check measures.

**Direct light at the visible surface.** The subject's pixels in an observation are
sampled (up to :data:`MAX_POINTS`, evenly in raster order, and up to :data:`RING_POINTS`
more just inside its outline); a ray through each pixel's centre (the path tracer's ray
core and camera arithmetic) finds the surface point and its shading normal. Each light's
irradiance there is the rasterizers' diffuse term on a white surface -- the polygon
formula of area lights (with their spread), the cone and falloff of spots and points --
so ``1`` is what makes a white surface appear white at exposure 0, the unit of
:func:`aether3d.rendering.studio.light_rig`'s ``level``. Shadows are decided by rays
toward the light (from :data:`AREA_SAMPLES` x :data:`AREA_SAMPLES` points spread over an
area light, toward the centre of the others; a light with ``shadow`` off is never
blocked), and light linking is honored. The ambient term is the rasterizers' (uniform,
hemisphere, or an image environment's harmonics), unoccluded. Light bounced off other
surfaces is not counted: this describes the lighting design rather than the final image,
and it is sampled (findings say ``exact: false``).

**Lighting ratio.** The *key* is the light that gives the subject the most. Its lit side
is the points it lights at half its strength or more (half its 95th percentile there), its
shadow side the points it does not reach (2% or less: turned away or shadowed); the ratio
is the median irradiance of the lit side over the shadow side's -- key plus fill against
fill, as a photographer compares the two sides of a face. It is measured on the visible
surface, not on cards facing each light: :func:`~aether3d.rendering.studio.light_rig`'s
nominal ratio holds at the subject's centre, and a round subject shows about twice it (its
shadow side turns away from the fill). With no shadow side in view (less than
:data:`MIN_SIDE` of the points: a frontal key) the ratio is the 90th over the 10th
percentile of the irradiance (``ratio_method``); with an unlit shadow side it is infinite
(``shadow_side_unlit``).

**Separation.** Measured on the displayed image: the outline is split into
:data:`SECTORS` angular sectors around the subject's centroid; in each, the median
lightness (CIE L*, 0 black to 100 white, of the displayed pixels) of a ring of pixels just
inside the outline is compared with a ring just outside (``r`` pixels wide, about 0.6% of
the image's larger side). A sector whose two medians differ by less than the check's
``min_separation`` is *merged*; ``merged_fraction`` weighs sectors by their outline
pixels. Lightness rather than stops: two near-black tones are many stops apart and still
look the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MAX_POINTS = 1024
"""Subject pixels sampled for the irradiance (evenly in raster order)."""
RING_POINTS = 256
"""More points sampled just inside the outline (where rim lights and separation act)."""
AREA_SAMPLES = 3
"""Per axis: an area light's shadow rays leave from this many x this many points on it."""
SECTORS = 16
"""Angular sectors of the outline in which separation is measured."""
MIN_SIDE = 0.03
"""Fraction of the points the key's lit and shadow sides each need for a key-side ratio."""
DEFAULT_SEPARATION = 10.0
"""L* difference below which an outline sector merges with what surrounds it."""
LUMA = (0.2126, 0.7152, 0.0722)
MIN_RING_PIXELS = 4


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * LUMA[0] + rgb[..., 1] * LUMA[1] + rgb[..., 2] * LUMA[2]


@dataclass
class Lighting:
    """The measurement (arrays per sampled point; :meth:`summary` gives the JSON)."""

    points: np.ndarray  # (n, 3) world positions
    normals: np.ndarray  # (n, 3) shading normals, toward the camera
    xy: np.ndarray  # (n, 2) their pixels (column, row)
    even: np.ndarray  # (n,) bool: in the even sample of the subject's pixels (statistics)
    on_outline: np.ndarray  # (n,) bool: just inside the outline
    total: np.ndarray  # (n,) luminance of the irradiance: lights + ambient
    ambient: np.ndarray  # (n,) the ambient term's luminance
    albedo: np.ndarray  # (n,) the base color's luminance (materials and textures)
    contribution: np.ndarray  # (n, L) each light's, linked and shadowed
    unshadowed: np.ndarray  # (n, L) each light's, linked, without shadows
    lights: list[dict[str, Any]]  # one per column: name, id, type, ...
    separation: dict[str, Any] | None  # the outline's (None: no outline to measure)
    centre: np.ndarray  # the sampled surface's centroid (world)
    eye: np.ndarray  # the camera's position
    pixels: int  # the subject's pixels in the image
    surround: dict[str, np.ndarray] | None = None  # points just outside the outline: xy,
    # points, normals, total, contribution, albedo (as above; pixels showing the
    # background have no row)
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Points of the even sample (the statistics are theirs)."""
        return int(np.count_nonzero(self.even))

    def key_index(self) -> int | None:
        """The light that gives the subject the most (``None``: no light reaches it)."""
        if not self.lights or not self.count:
            return None
        share = self.contribution[self.even].mean(axis=0)
        k = int(np.argmax(share))
        return k if share[k] > 0.0 else None

    def sides(self) -> tuple[np.ndarray, np.ndarray, str]:
        """The points (indices into the arrays) of the lit and the shadow side, and how they
        were chosen: ``"key"`` (the key's reach) or ``"percentile"`` (the top and bottom
        tenths)."""
        rows = np.flatnonzero(self.even)
        k = self.key_index()
        if k is not None:
            c = self.contribution[rows, k]
            top = float(np.quantile(c, 0.95))
            lit, shadow = rows[c >= 0.5 * top], rows[c <= 0.02 * top]
            if min(lit.size, shadow.size) >= MIN_SIDE * rows.size:
                return lit, shadow, "key"
        total = self.total[rows]
        hi, lo = np.quantile(total, [0.9, 0.1])
        return rows[total >= hi], rows[total <= lo], "percentile"

    def ratio(self) -> tuple[float | None, str]:
        """The lighting ratio (``inf``: an unlit shadow side; ``None``: nothing measured or
        no light) and its method (module docstring)."""
        if not self.count:
            return None, "none"
        lit, shadow, method = self.sides()
        hi, lo = float(np.median(self.total[lit])), float(np.median(self.total[shadow]))
        if hi <= 0.0:
            return None, method
        return (math.inf if lo <= 1e-9 * hi else hi / lo), method

    def summary(self) -> dict[str, Any]:
        even = self.even
        mean_total = float(self.total[even].mean()) if self.count else 0.0
        lights = []
        for k, info in enumerate(self.lights):
            c, u = self.contribution[even, k], self.unshadowed[even, k]
            facing = u > 0.0
            row = {
                **{key: info[key] for key in ("name", "id", "type") if key in info},
                "share": float(c.mean() / mean_total) if mean_total > 0.0 else 0.0,
                "reach": float(np.mean(c > 0.0)) if self.count else 0.0,
                "blocked": float(np.mean(facing & (c <= 0.0))) if self.count else 0.0,
            }
            for key in ("unlinked", "role", "default", "environment", "blockers"):
                if info.get(key):
                    row[key] = info[key]
            lights.append(row)
        ratio, method = self.ratio()
        q = np.quantile(self.total[even], [0.1, 0.5, 0.9]) if self.count else [0.0, 0.0, 0.0]
        key = self.key_index()
        out: dict[str, Any] = {
            "points": self.count,
            "pixels": self.pixels,
            "ratio": None if ratio is None or math.isinf(ratio) else ratio,
            "ratio_method": method,
            "shadow_side_unlit": ratio is not None and math.isinf(ratio),
            "irradiance": {
                "p10": float(q[0]),
                "median": float(q[1]),
                "p90": float(q[2]),
                "mean": mean_total,
            },  # fmt: skip
            "ambient_share": float(self.ambient[even].mean() / mean_total)
            if mean_total > 0
            else 0.0,
            "key": None if key is None else self.lights[key]["name"],
            "lights": lights,
            "separation": self.separation,
            "basis": "direct light of the scene's lights at the subject's visible surface "
            "(pixel centres; shadows by rays) and the rasterizers' unoccluded ambient term; "
            "light bounced off other surfaces is not counted",
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out


# --------------------------------------------------------------------------- #
# Masks and rings
# --------------------------------------------------------------------------- #


def _box_any(mask: np.ndarray, r: int) -> np.ndarray:
    """Pixels with a set pixel within ``r`` (a square neighbourhood; outside is unset)."""
    h, w = mask.shape
    s = np.zeros((h + 1, w + 1), np.int64)
    s[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0, dtype=np.int64), axis=1)
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    total = s[y1][:, x1] - s[y0][:, x1] - s[y1][:, x0] + s[y0][:, x0]  # (the neighbourhood's count)
    return total > 0


def rings(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """The pixels just inside and just outside a mask's outline, and the ring width."""
    h, w = mask.shape
    r = max(2, int(round(0.006 * max(h, w))))
    # (outside the image counts as neither: a subject cut by the frame has no outline there)
    inner = mask & _box_any(~mask, r)
    outer = ~mask & _box_any(mask, r)
    return inner, outer, r


def lightness(image: np.ndarray) -> np.ndarray:
    """CIE L* (0 to 100) of displayed sRGB pixels (``(h, w, 3) uint8``)."""
    c = image[..., :3].astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return y_to_lstar(_luma(lin))


def separation(
    image: np.ndarray, mask: np.ndarray, min_separation: float = DEFAULT_SEPARATION
) -> dict[str, Any] | None:
    """The outline's separation from what surrounds it in the displayed ``image``
    (module docstring); ``None`` when the subject has no outline in the image."""
    inner, outer, r = rings(mask)
    if inner.sum() < MIN_RING_PIXELS or outer.sum() < MIN_RING_PIXELS:
        return None
    lstar = lightness(image)
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5
    sectors = []
    for ring, side in ((inner, "in"), (outer, "out")):
        yy, xx = np.nonzero(ring)
        # Image angles: 0 to the right, 90 up.
        ang = np.degrees(np.arctan2(cy - (yy + 0.5), xx + 0.5 - cx)) % 360.0
        k = np.minimum((ang / (360.0 / SECTORS)).astype(np.int64), SECTORS - 1)
        values = lstar[yy, xx]
        for s in range(SECTORS):
            if side == "in":
                sectors.append({"sector": s, "in": values[k == s]})
            else:
                sectors[s]["out"] = values[k == s]
    rows, merged_px, valid_px = [], 0, 0
    for sec in sectors:
        a, b = sec["in"], sec["out"]
        if a.size < MIN_RING_PIXELS or b.size < MIN_RING_PIXELS:
            continue
        med_in, med_out = float(np.median(a)), float(np.median(b))
        merged = abs(med_in - med_out) < min_separation
        valid_px += int(b.size)
        merged_px += int(b.size) if merged else 0
        rows.append({
            "angle": (sec["sector"] + 0.5) * 360.0 / SECTORS,
            "inside": med_in,
            "outside": med_out,
            "difference": med_in - med_out,
            "merged": merged,
            "pixels": int(b.size),
        })  # fmt: skip
    if not rows:
        return None
    diff = np.abs(np.array([row["difference"] for row in rows]))
    return {
        "min_separation": min_separation,
        "unit": "L*",
        "merged_fraction": merged_px / valid_px if valid_px else 0.0,
        "median_difference": float(np.median(diff)),
        "min_difference": float(diff.min()),
        "ring_pixels": int(r),
        "centroid": [float(cx), float(cy)],
        "sectors": rows,
    }


def sector_of(xy: np.ndarray, centroid: list[float]) -> np.ndarray:
    """The outline sector of pixels ``(column, row)`` around the subject's centroid."""
    ang = np.degrees(np.arctan2(centroid[1] - (xy[:, 1] + 0.5), xy[:, 0] + 0.5 - centroid[0]))
    return np.minimum(((ang % 360.0) / (360.0 / SECTORS)).astype(np.int64), SECTORS - 1)


def lstar_to_y(lstar: np.ndarray | float) -> np.ndarray:
    """The luminance (0 to 1) of a CIE lightness."""
    lstar = np.asarray(lstar, np.float64)
    f = (lstar + 16.0) / 116.0
    return np.where(lstar > 8.0, f * f * f, lstar * (27.0 / 24389.0))


def y_to_lstar(y: np.ndarray | float) -> np.ndarray:
    y = np.clip(np.asarray(y, np.float64), 0.0, 1.0)
    return np.where(y > 216.0 / 24389.0, 116.0 * np.cbrt(y) - 16.0, y * (24389.0 / 27.0))


# --------------------------------------------------------------------------- #
# The irradiance
# --------------------------------------------------------------------------- #


def unshadowed(light: Any, p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """``(m, 3)`` irradiance of ``light`` at points ``p`` with normals ``n``, in white-
    surface units, without shadows or links: the rasterizers' diffuse term."""
    from ..rendering.surface import _light_vectors, area_irradiance

    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    if light.type == "area":
        g = area_irradiance(light, px, py, pz, nx, ny, nz) / light.area
        return light.radiance[None, :] * g[:, None]
    lx, ly, lz, atten = _light_vectors(light, px, py, pz)
    ndl = np.maximum(nx * lx + ny * ly + nz * lz, 0.0)
    weight = ndl if atten is None else ndl * atten
    return light.radiance[None, :] * weight[:, None]


def _emitters(light: Any) -> np.ndarray:
    """Points shadow rays aim at: spread over an area light, else its centre."""
    if light.type != "area":
        return light.position[None, :]
    u = (np.arange(AREA_SAMPLES) + 0.5) * (2.0 / AREA_SAMPLES) - 1.0
    uu, vv = (a.ravel() for a in np.meshgrid(u, u))
    if light.shape in ("disk", "ellipse"):  # the same points, pulled inside the ellipse
        radius = np.maximum(np.hypot(uu, vv), 1e-300)
        scale = np.minimum(1.0, 0.9 / radius)
        uu, vv = uu * scale, vv * scale
    return light.position[None, :] + uu[:, None] * light.axis_x + vv[:, None] * light.axis_y


def _visibility(ps: Any, it: Any, light: Any, rows: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """The fraction of ``light`` the points ``rows`` see (shadow rays; the others get 0),
    and the instances that block their central rays most often."""
    from ..rendering.pathtrace.scene import _subset

    vis = np.zeros(it.p.shape[0])
    if not light.shadow:
        vis[rows] = 1.0
        return vis, []
    sub = _subset(it, rows)
    m = rows.size
    origin = ps.offset_origin(sub.p, sub.ng * sub.toward[:, None])
    if light.type == "directional":
        d = np.broadcast_to(-light.direction, (m, 3)).copy()
        seen = ps.visible(origin, d, np.full(m, np.inf), sub.inst, sub.prim)
        hits, _ = ps.trace(origin[~seen], d[~seen], 0.0, np.inf, sub.inst[~seen],
                           sub.prim[~seen])  # fmt: skip
        vis[rows] = seen
        return vis, _blockers(hits.inst)
    total = np.zeros(m)
    blockers: list[int] = []
    targets = _emitters(light)
    for j, q in enumerate(targets):
        v = q[None, :] - origin
        dist = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])
        d = v / np.maximum(dist, 1e-300)[:, None]
        seen = ps.visible(origin, d, dist * (1.0 - 1e-7), sub.inst, sub.prim)
        total += seen
        if j == len(targets) // 2 and (~seen).any():  # the centre's blockers
            hits, _ = ps.trace(origin[~seen], d[~seen], 0.0, dist[~seen] * (1.0 - 1e-7),
                               sub.inst[~seen], sub.prim[~seen])  # fmt: skip
            blockers = _blockers(hits.inst)
    vis[rows] = total / len(targets)
    return vis, blockers


def _blockers(inst: np.ndarray) -> list[int]:
    inst = inst[inst >= 0]
    if not inst.size:
        return []
    ids, counts = np.unique(inst, return_counts=True)
    return [int(i) for i in ids[np.argsort(-counts, kind="stable")][:3]]


def _render_scene(obs: Any):
    from ..rendering.overlay import animation_overlay
    from ..rendering.scene_data import extract_render_scene, raster_environment

    scene = obs.scene
    overlay = animation_overlay(scene, obs.time) if obs.time is not None else None
    aspect = obs.width / obs.height
    rs = extract_render_scene(
        scene,
        obs.view.target,
        default_lights=scene.render_settings.default_lights,
        aspect=aspect,
        overlay=overlay,
    )
    if scene._worlds is not None and len(scene._worlds):
        from ..worlds.render import contribute

        rs = contribute(scene, rs, obs.height, aspect)
    return raster_environment(rs)


def _even(rows: np.ndarray, count: int) -> np.ndarray:
    if rows.size <= count:
        return rows
    return rows[np.linspace(0, rows.size - 1, count).round().astype(np.int64)]


def _irradiance(ps: Any, rs: Any, it: Any, lights: list[dict[str, Any]] | None = None):
    """Per-light irradiance luminance at interactions ``it`` (normals turned toward the
    rays' origins): ``(contribution, unshadowed, ambient, normals)``. With ``lights``
    (the measurement's rows), records links and blockers there."""
    from ..rendering.surface import _hemisphere, linked

    n = it.ns * it.toward[:, None]
    m = it.p.shape[0]
    contribution = np.zeros((m, len(rs.lights)))
    flat = np.zeros((m, len(rs.lights)))
    for k, light in enumerate(rs.lights):
        u = _luma(unshadowed(light, it.p, n))
        allow = linked(light, it.inst)
        if allow is not None:
            u = np.where(allow, u, 0.0)
            if lights is not None and not allow.any():
                lights[k]["unlinked"] = True
        flat[:, k] = u
        reached = np.flatnonzero(u > 0.0)
        if not reached.size:
            continue
        vis, blockers = _visibility(ps, it, light, reached)
        contribution[:, k] = u * vis
        if lights is not None and blockers:
            lights[k]["blockers"] = [rs.instances[b].name for b in blockers]
    hemi = _hemisphere(rs, n[:, 2], (n[:, 0], n[:, 1], n[:, 2]))
    if hemi is not None:
        ambient = _luma(hemi)
    else:
        ambient = np.full(m, float(_luma(np.asarray(rs.ambient, np.float64))))
    return contribution, flat, ambient, n


def measure(
    obs: Any, objects: list[Any], *, min_separation: float = DEFAULT_SEPARATION
) -> Lighting:
    """How ``objects`` (one subject: their geometry and their descendants') are lit in
    ``obs`` (module docstring). Cached on the observation; never changes the scene."""
    from ..rendering.pathtrace.scene import PathScene, _subset
    from ..spatial.queries import View
    from .observe import parts_of

    key = "|".join(o.id for o in objects) + f"|{float(min_separation).hex()}"
    cached = obs._lighting.get(key)
    if cached is not None:
        return cached
    rows = sorted({k for o in objects for k in obs.instances_of(o)})
    mask = np.isin(obs.index, rows) if rows else np.zeros(obs.index.shape, bool)
    pixels = int(mask.sum())
    sep = separation(obs.image, mask, min_separation) if pixels else None
    rs = _render_scene(obs)
    subject_ids = {p.id for o in objects for p in parts_of(o)}
    inside = np.array([inst.object_id in subject_ids for inst in rs.instances], bool)
    lights = []
    for light in rs.lights:
        info: dict[str, Any] = {"name": light.name, "id": light.object_id, "type": light.type}
        if light.object_id is None:
            info["environment" if light.name == "environment sun" else "default"] = True
        else:
            obj = obs.scene._objects.get(light.object_id)
            if obj is not None and "light_rig" in obj.tags:
                info["role"] = light.name.split()[-1].lower() if light.name else None
        lights.append(info)
    width = obs.width
    sample = _even(np.flatnonzero(mask.ravel()), MAX_POINTS)
    even = np.ones(sample.size, bool)
    outer = np.zeros(0, np.int64)
    if sep is not None:
        inner_ring, outer_ring, _ = rings(mask)
        ring = np.flatnonzero(inner_ring.ravel())
        extra = np.setdiff1d(_even(ring, RING_POINTS), sample)
        sample = np.concatenate([sample, extra])
        even = np.concatenate([even, np.zeros(extra.size, bool)])
        outer = _even(np.flatnonzero(outer_ring.ravel()), RING_POINTS)
        on_ring = np.isin(sample, ring)
    else:
        on_ring = np.zeros(sample.size, bool)
    n_lights = len(rs.lights)
    zeros = np.zeros((0, n_lights))
    empty = Lighting(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), np.int64),
                     np.zeros(0, bool), np.zeros(0, bool), np.zeros(0), np.zeros(0),
                     np.zeros(0), zeros, zeros, lights, sep, np.zeros(3), np.zeros(3),
                     pixels)  # fmt: skip
    if not sample.size or not inside.any():
        obs._lighting[key] = empty
        return empty
    view = View(obs.scene, obs.view.target, obs.width, obs.height)
    ps = PathScene(rs)
    traced = []
    for pix in (sample, outer):
        ys, xs = np.divmod(pix, width)
        o, d, tmin, tmax = view.rays(xs + 0.5, ys + 0.5)
        _, it = ps.trace(o, d, tmin, tmax) if pix.size else (None, None)
        traced.append(it)
    it = traced[0]
    if it is None:
        obs._lighting[key] = empty
        return empty
    keep = np.flatnonzero(inside[it.inst])
    it = _subset(it, keep[np.argsort(it.rows[keep], kind="stable")])
    contribution, flat, ambient, n = _irradiance(ps, rs, it, lights)
    xy = np.stack([sample[it.rows] % width, sample[it.rows] // width], axis=1)
    surround = None
    if traced[1] is not None:
        out_it = traced[1]
        order = np.argsort(out_it.rows, kind="stable")
        out_it = _subset(out_it, order)
        c_out, _, a_out, _ = _irradiance(ps, rs, out_it)
        surround = {
            "xy": np.stack([outer[out_it.rows] % width, outer[out_it.rows] // width], axis=1),
            "points": out_it.p,
            "normals": out_it.ns * out_it.toward[:, None],
            "total": c_out.sum(axis=1) + a_out,
            "contribution": c_out,
            "albedo": _luma(ps.shade(out_it).base),
        }
    total = contribution.sum(axis=1) + ambient
    notes = []
    if any(info.get("default") for info in lights):
        notes.append("the scene has no lights: the renderers' built-in default light is measured")
    albedo = _luma(ps.shade(it).base)
    out = Lighting(it.p, n, xy, even[it.rows], on_ring[it.rows], total, ambient, albedo,
                   contribution, flat, lights, sep, it.p[even[it.rows]].mean(axis=0),
                   view.eye, pixels, surround, notes)  # fmt: skip
    obs._lighting[key] = out
    return out
