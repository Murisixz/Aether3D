"""The path tracing integrator: camera rays, the bounce loop, passes and accumulation.

Every pixel's value is the average of ``samples`` independent path
estimates of the radiance arriving through a uniformly random point of the
pixel (a box filter). A path (the textbook unidirectional estimator with
next-event estimation; Veach 1997, Pharr et al. "Physically Based
Rendering"):

1. starts at the camera; camera rays only see geometry between the
   camera's ``near`` and ``far`` distances, like the rasterizers;
2. at every surface it hits, adds the surface's emission -- weighted
   against next-event estimation by the power heuristic when the surface was
   found by BSDF sampling -- and, at vertices before the last, samples the
   lights (next-event estimation, :mod:`.lights`) and the BSDF
   (:mod:`.bsdf`) for the next direction;
3. adds the environment when a ray leaves the scene (camera rays see the
   world background in ``"ambient"`` mode);
4. stops after ``bounces + 1`` surface vertices, when the BSDF sample is
   invalid (below the surface), or by Russian roulette from vertex
   ``russian_roulette`` on: it continues with probability ``q = min(1,
   max(throughput))`` (at least 0.05) and divides its throughput by ``q``.

``bounces = 0`` is therefore direct lighting only: what the camera sees
directly, lit by every light and emitter (and the environment, through the
BSDF ray of the first vertex). The **direct** pass holds exactly that;
the **indirect** pass everything after it (``beauty = direct + indirect``).

All paths are independent and their random numbers are counter-based
(:mod:`.sampling`), so images do not depend on batching. Per pixel, samples
are summed in sample order, which makes renders bitwise repeatable on a
given platform.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ...diagnostics import INFO, WARNING, Diagnostic
from ...errors import RenderError
from ..scene_data import RenderScene
from ..settings import AOSettings, PathSettings, RenderLimits
from ..surface import LUMINANCE
from .bsdf import BSDF, MIN_COS
from .lights import SHADOW_END, LightSet, mis_weight_emission
from .sampling import (
    SLOT_AO,
    SLOT_BSDF,
    SLOT_LOBE,
    SLOT_PIXEL,
    SLOT_RR,
    cosine_hemisphere,
    dimension,
    frame,
    path_keys,
    power_heuristic,
    to_world,
    uniform,
)
from .scene import MAX_LAYERS, PathScene

BATCH = 65_536
"""Paths traced together (bounds temporary memory)."""
DIFFUSE_SPREAD = 1.0
"""Ray-cone spread (radians) added by a diffuse bounce."""
RR_MIN = 0.05
"""Smallest Russian-roulette survival probability."""
PASSES = (
    "beauty",
    "direct",
    "indirect",
    "albedo",
    "normal",
    "depth",
    "ao",
    "sample_count",
    "variance",
)


@dataclass
class PathFrame:
    """Output of :func:`render_scene` (linear, scene-referred, before exposure)."""

    linear: np.ndarray  # (H, W, 3) beauty
    passes: dict[str, np.ndarray]
    coverage: np.ndarray  # (H, W) float32: fraction of camera samples that hit geometry
    depth: np.ndarray  # (H, W) float32: view depth at the pixel center (inf: none)
    instance: np.ndarray  # (H, W) int32: instance at the pixel center (-1: none)
    visible_samples: np.ndarray  # (instances,) int64: camera samples per instance
    counts: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    info: dict = field(default_factory=dict)


@dataclass
class _Camera:
    eye: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    hx: float
    hy: float
    perspective: bool
    near: float
    far: float
    width: int
    height: int
    scale: float  # ray-differential scale (1 / sqrt(samples))

    def rays(self, col, row, u, v):
        """Unit camera rays through ``(col + u, row + v)``; ``(o, d, tmin, tmax, diff)``."""
        w, h = self.width, self.height
        xn = (col + u) * (2.0 / w) - 1.0
        yn = 1.0 - (row + v) * (2.0 / h)
        sx = (xn * self.hx)[:, None]
        sy = (yn * self.hy)[:, None]
        n = xn.shape[0]
        step_x = (2.0 / w) * self.hx * self.scale
        step_y = -(2.0 / h) * self.hy * self.scale
        if self.perspective:
            d = self.forward + sx * self.right + sy * self.up
            length = np.linalg.norm(d, axis=1, keepdims=True)
            unit = d / length
            o = np.broadcast_to(self.eye, (n, 3)).copy()
            # d(unit)/dx = (I - u u^T) dd/dx / |d|
            dd = []
            for vec in (self.right * step_x, self.up * step_y):
                raw = np.broadcast_to(vec, (n, 3))
                dd.append((raw - unit * (unit @ vec)[:, None]) / length)
            zero = np.zeros(3)
            diff = (zero, zero, dd[0], dd[1])
            tmin = self.near * length[:, 0]
            tmax = self.far * length[:, 0]
            return o, unit, tmin, tmax, diff
        o = self.eye + sx * self.right + sy * self.up
        d = np.broadcast_to(self.forward, (n, 3)).copy()
        zero = np.zeros(3)
        diff = (self.right * step_x, self.up * step_y, zero, zero)
        return o, d, np.full(n, self.near), np.full(n, self.far), diff


def _camera(rs: RenderScene, width: int, height: int, samples: int) -> _Camera:
    cam = rs.camera.camera
    frame_ = rs.camera.frame
    hx, hy = cam.half_extents(width / height)
    return _Camera(
        np.asarray(frame_.position, dtype=np.float64),
        np.asarray(frame_.forward, dtype=np.float64),
        np.asarray(frame_.right, dtype=np.float64),
        np.asarray(frame_.up, dtype=np.float64),
        hx,
        hy,
        cam.is_perspective,
        cam.near,
        cam.far,
        width,
        height,
        1.0 / math.sqrt(samples),
    )


def estimate_work(rs: RenderScene, width: int, height: int, path: PathSettings, ao) -> dict:
    """Upper estimates of the rays and memory of a render (for :class:`RenderLimits`)."""
    paths = width * height * path.samples
    lights = len(rs.lights) + (1 if any(np.any(m.emissive > 0) for m in rs.materials) else 0)
    ao_rays = 0 if ao is None else ao.samples
    if path.integrator == "ao":
        rays = paths * (1 + ao_rays)
    else:
        vertices = path.bounces + 2
        rays = paths * (vertices + (vertices - 1) * lights * path.light_samples + ao_rays)
    tris = rs.triangle_count
    memory = (
        width * height * 200  # accumulators
        + tris * 250  # world-space shading arrays and the BVH
        + BATCH * 4096  # one batch of paths in flight
    )
    return {"paths": int(paths), "rays": int(rays), "memory": int(memory)}


def render_scene(
    rs: RenderScene,
    width: int,
    height: int,
    path: PathSettings,
    ao: AOSettings | None,
    cull: str = "none",
    limits: RenderLimits | None = None,
    progress=None,
) -> PathFrame:
    """Path trace ``rs`` (see the module docstring)."""
    limits = limits or RenderLimits()
    work = estimate_work(rs, width, height, path, ao)
    if work["rays"] > limits.max_path_rays:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"this path-traced render would trace up to {work['rays']:,} rays "
            f"(limit {limits.max_path_rays:,})",
            "path",
            hint="lower the resolution, samples, bounces or light_samples",
        )
    if work["memory"] > limits.max_path_memory:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"this path-traced render needs about {work['memory'] / 2**20:,.0f} MiB "
            f"(limit {limits.max_path_memory / 2**20:,.0f} MiB)",
            "path",
            hint="lower the resolution or the scene's triangle count",
        )
    if rs.triangle_count > limits.max_triangles:
        raise RenderError(
            "RESOURCE_LIMIT",
            f"the scene has {rs.triangle_count:,} triangles to render "
            f"(limit {limits.max_triangles:,})",
            hint="hide objects or reduce subdivision levels",
        )
    t_start = time.perf_counter()
    ps = PathScene(rs)
    lights = LightSet.build(ps)
    t_setup = time.perf_counter() - t_start
    if path.integrator == "ao" and ao is None:
        ao = AOSettings()
    cam = _camera(rs, width, height, path.samples)
    tracer = _Tracer(ps, lights, cam, path, ao, cull)
    n_pix = width * height
    acc = {
        "beauty": np.zeros((n_pix, 3)),
        "squares": np.zeros((n_pix, 3)),
        "direct": np.zeros((n_pix, 3)),
        "indirect": np.zeros((n_pix, 3)),
        "albedo": np.zeros((n_pix, 3)),
        "normal": np.zeros((n_pix, 3)),
        "ao": np.zeros(n_pix),
        "hits": np.zeros(n_pix),
    }
    visible = np.zeros(len(rs.instances), dtype=np.int64)
    total = n_pix * path.samples
    passes_done = 0
    stopped = False
    paths = 0
    t_trace = time.perf_counter()
    for start in range(0, total, BATCH):
        idx = np.arange(start, min(total, start + BATCH), dtype=np.int64)
        sample, pixel = np.divmod(idx, n_pix)
        out = tracer.trace(pixel, sample)
        end = int(idx[-1]) + 1
        for s in np.unique(sample):  # per pixel, add samples in sample order
            sel = np.flatnonzero(sample == s)
            px = pixel[sel]
            beauty = out["direct"][sel] + out["indirect"][sel]
            acc["beauty"][px] += beauty
            acc["squares"][px] += beauty * beauty
            for name in ("direct", "indirect", "albedo", "normal"):
                acc[name][px] += out[name][sel]
            acc["ao"][px] += out["ao"][sel]
            acc["hits"][px] += out["hit"][sel]
            hit = out["hit"][sel]
            visible += np.bincount(out["inst"][sel][hit], minlength=visible.size)
            paths += sel.size
            if (int(s) + 1) * n_pix > end:
                continue  # this sample pass continues in the next batch
            passes_done = int(s) + 1
            if (
                progress is not None
                and progress(passes_done, path.samples, time.perf_counter() - t_start) is False
                and passes_done < path.samples
            ):
                stopped = True
                break
        if stopped:
            break
    t_trace = time.perf_counter() - t_trace
    spp = passes_done
    if spp == 0:  # pragma: no cover - every batch ends at least one pass or continues
        raise RenderError("RENDER_FAILED", "no samples were rendered")
    t0 = time.perf_counter()
    depth, instance = tracer.center(n_pix)
    t_center = time.perf_counter() - t0
    inv = 1.0 / spp
    mean = acc["beauty"] * inv
    if spp > 1:
        variance = np.maximum(acc["squares"] * inv - mean * mean, 0.0) * (1.0 / (spp - 1))
    else:
        variance = np.full((n_pix, 3), np.inf)
    shape = (height, width)
    passes = {
        "beauty": mean.reshape(*shape, 3),
        "direct": (acc["direct"] * inv).reshape(*shape, 3),
        "indirect": (acc["indirect"] * inv).reshape(*shape, 3),
        "albedo": (acc["albedo"] * inv).reshape(*shape, 3),
        "normal": (acc["normal"] * inv).reshape(*shape, 3),
        "depth": depth.reshape(shape),
        "sample_count": np.full(shape, spp, dtype=np.int32),
        "variance": variance.reshape(*shape, 3),
    }
    if ao is not None:
        passes["ao"] = (acc["ao"] * inv).reshape(shape)
    if path.integrator == "ao":
        passes["beauty"] = passes["direct"]
    coverage = (acc["hits"] * inv).astype(np.float32).reshape(shape)
    diagnostics = []
    if stopped:
        diagnostics.append(
            Diagnostic(
                "RENDER_STOPPED",
                INFO,
                f"the render stopped after {spp} of {path.samples} samples per pixel "
                "(progress callback)",
                "path",
            )
        )
    lum = (mean @ LUMINANCE).reshape(-1)
    err = np.sqrt(np.maximum((variance @ (LUMINANCE * LUMINANCE)).reshape(-1), 0.0))
    lit = lum > 1e-6
    measured = bool(lit.any()) and spp > 1
    noise = {
        "mean_standard_error": float(err[lit].mean()) if measured else None,
        "mean_relative_error": float((err[lit] / lum[lit]).mean()) if measured else None,
    }
    counts = dict(tracer.counts)
    counts.update(
        {
            "triangles": ps.triangle_count,
            "instances": len(rs.instances),
            "lights": lights.count,
            "emissive_triangles": 0 if lights.emitters is None else lights.emitters.count,
            "paths": int(paths),
            "samples_per_pixel": int(spp),
        }
    )
    counts["pass_through_rays"] = ps.pass_through_rays
    if ps.layer_limited:
        diagnostics.append(
            Diagnostic(
                "PASS_THROUGH_LIMIT",
                WARNING,
                f"{ps.layer_limited} rays crossed more than {MAX_LAYERS} cut-out or "
                "single-sided surfaces; they were treated as leaving the scene",
                "path",
            )
        )
    return PathFrame(
        linear=passes["beauty"].copy(),
        passes=passes,
        coverage=coverage,
        depth=depth.reshape(shape).astype(np.float32),
        instance=instance.reshape(shape).astype(np.int32),
        visible_samples=visible,
        counts=counts,
        timings={
            "setup": t_setup,
            "bvh": ps.timings["bvh"],
            "trace": t_trace,
            "center": t_center,
            **{f"trace_{k}": v for k, v in tracer.timings.items()},
        },
        diagnostics=diagnostics,
        info={
            "bvh": ps.accel.stats(),
            "lights": lights.to_dict(),
            "environment": ps.environment.to_dict(),
            "noise": noise,
            "work_estimate": work,
        },
    )


class _Tracer:
    """Traces batches of paths (one instance per render)."""

    def __init__(self, ps, lights, cam, path, ao, cull):
        self.ps = ps
        self.lights = lights
        self.cam = cam
        self.path = path
        self.ao = ao
        self.cull = cull
        self.counts = {
            "camera_rays": 0,
            "bounce_rays": 0,
            "shadow_rays": 0,
            "ao_rays": 0,
            "center_rays": 0,
        }
        self.timings = {"camera": 0.0, "shade": 0.0, "lights": 0.0, "bounce": 0.0, "ao": 0.0}

    # -------------------------------------------------------------------- paths --

    def trace(self, pixel: np.ndarray, sample: np.ndarray) -> dict[str, np.ndarray]:
        ps, cam, path = self.ps, self.cam, self.path
        n = pixel.shape[0]
        keys = path_keys(path.seed, pixel, sample)
        row, col = np.divmod(pixel, cam.width)
        u = uniform(keys, dimension(0, SLOT_PIXEL))
        v = uniform(keys, dimension(0, SLOT_PIXEL + 1))
        t0 = time.perf_counter()
        o, d, tmin, tmax, diff = cam.rays(col, row, u, v)
        hits, it = ps.trace(o, d, tmin, tmax, cull=self.cull, diff=diff)
        self.counts["camera_rays"] += n
        self.timings["camera"] += time.perf_counter() - t0
        out = {
            "direct": np.zeros((n, 3)),
            "indirect": np.zeros((n, 3)),
            "albedo": np.zeros((n, 3)),
            "normal": np.zeros((n, 3)),
            "ao": np.ones(n),
            "hit": hits.hit,
            "inst": np.where(hits.hit, hits.inst, 0),
        }
        miss = ~hits.hit
        if miss.any():
            out["direct"][miss] = ps.environment.camera(d[miss])
        if it is None:
            return out
        if self.ao is not None:
            out["ao"][it.rows] = self._ao(it, keys[it.rows])
        if path.integrator == "ao":
            out["direct"][it.rows] = out["ao"][it.rows, None]
            return out
        throughput = np.ones((n, 3))
        prev_pdf = np.zeros(n)
        spread = np.zeros(n)
        last = path.bounces + 1
        vertex = 0
        while it is not None:
            rows = it.rows
            t0 = time.perf_counter()
            sh = ps.shade(it)
            self.timings["shade"] += time.perf_counter() - t0
            target = out["direct"] if vertex <= 1 else out["indirect"]
            # Emission seen at this vertex (MIS-weighted against NEE after a bounce).
            glow = sh.emissive.any(axis=1)
            if glow.any():
                le = sh.emissive[glow]
                r = rows[glow]
                if vertex > 0:
                    sub = _rows_of(it, glow)
                    le = le * mis_weight_emission(self.lights, sub, prev_pdf[r])[:, None]
                target[r] += throughput[r] * le
            n_shade = sh.normal * it.toward[:, None]
            ng_side = it.ng * it.toward[:, None]
            wo = -it.d
            if vertex == 0:
                out["normal"][rows] = n_shade
            if vertex == last:
                break
            bad = (n_shade * wo).sum(axis=1) < MIN_COS
            n_shade = np.where(bad[:, None], ng_side, n_shade)
            bsdf = BSDF.build(
                n_shade, wo, sh.base, sh.metallic, sh.roughness, sh.specular, sh.has_specular
            )
            if vertex == 0:
                out["albedo"][rows] = bsdf.albedo()
            kv = keys[rows]
            # Next-event estimation.
            t0 = time.perf_counter()
            direct = self._direct(it, bsdf, ng_side, kv, vertex)
            target = out["direct"] if vertex == 0 else out["indirect"]
            target[rows] += throughput[rows] * direct
            self.timings["lights"] += time.perf_counter() - t0
            # BSDF sampling for the next ray.
            t0 = time.perf_counter()
            u_lobe = uniform(kv, dimension(vertex, SLOT_LOBE))
            u1 = uniform(kv, dimension(vertex, SLOT_BSDF))
            u2 = uniform(kv, dimension(vertex, SLOT_BSDF + 1))
            wi, weight, pdf, valid, spec = bsdf.sample(u_lobe, u1, u2)
            valid &= (wi * ng_side).sum(axis=1) > 0.0
            throughput[rows] *= weight
            if vertex >= path.russian_roulette:
                q = np.clip(throughput[rows].max(axis=1), RR_MIN, 1.0)
                survive = uniform(kv, dimension(vertex, SLOT_RR)) < q
                valid &= survive
                throughput[rows] /= q[:, None]
            keep = np.flatnonzero(valid)
            if not keep.size:
                break
            r = rows[keep]
            prev_pdf[r] = pdf[keep]
            spread[r] = spread[r] + np.where(spec[keep], 2.0 * bsdf.alpha[keep], DIFFUSE_SPREAD)
            origin = ps.offset_origin(it.p[keep], ng_side[keep])
            direction = wi[keep]
            hits, nxt = ps.trace(
                origin,
                direction,
                0.0,
                np.inf,
                it.inst[keep].astype(np.int64),
                it.prim[keep].astype(np.int64),
                cone=(it.width[keep], spread[r]),
            )
            self.counts["bounce_rays"] += keep.size
            self.timings["bounce"] += time.perf_counter() - t0
            vertex += 1
            target = out["direct"] if vertex == 1 else out["indirect"]
            miss = ~hits.hit
            if miss.any():
                m = r[miss]
                target[m] += throughput[m] * ps.environment.radiance(direction[miss])
            if nxt is None:
                break
            nxt.rows = r[nxt.rows]
            it = nxt
        if path.clamp > 0.0:
            peak = out["indirect"].max(axis=1)
            scale = np.where(peak > path.clamp, path.clamp / np.maximum(peak, 1e-300), 1.0)
            out["indirect"] *= scale[:, None]
        return out

    # ---------------------------------------------------------- direct lighting --

    def _direct(self, it, bsdf: BSDF, ng_side, keys, vertex: int) -> np.ndarray:
        """Next-event estimate at the interactions: ``(m, 3)`` (before throughput)."""
        ps, lights = self.ps, self.lights
        m = it.rows.shape[0]
        out = np.zeros((m, 3))
        ns = self.path.light_samples
        origin = ps.offset_origin(it.p, ng_side)
        rays_o, rays_d, rays_t, rays_row, rays_c = [], [], [], [], []
        index = 0
        for light in range(lights.count):
            for _ in range(ns):
                wi, target, scale = lights.sample_analytic(light, it.p, keys, vertex, index)
                index += 1
                f, _, cos = bsdf.eval(wi)
                ok = (cos > 0.0) & ((wi * ng_side).sum(axis=1) > 0.0)
                contrib = f * scale * (cos / ns)[:, None]
                ok &= contrib.any(axis=1)
                if not ok.any():
                    continue
                k = np.flatnonzero(ok)
                rays_row.append(k)
                rays_c.append(contrib[k])
                rays_o.append(origin[k])
                if target is None:
                    rays_d.append(wi[k])
                    rays_t.append(np.full(k.size, np.inf))
                else:
                    rays_d.append(target[k] - origin[k])
                    rays_t.append(np.full(k.size, SHADOW_END))
        if lights.emitters is not None:
            for _ in range(ns):
                wi, x, le_over_pdf, pdf, _, _ = lights.sample_emitter(ps, it.p, keys, vertex, index)
                index += 1
                f, pdf_bsdf, cos = bsdf.eval(wi)
                ok = (cos > 0.0) & (pdf > 0.0) & ((wi * ng_side).sum(axis=1) > 0.0)
                w = power_heuristic(pdf, pdf_bsdf)
                contrib = f * le_over_pdf * (cos * w / ns)[:, None]
                ok &= contrib.any(axis=1)
                if not ok.any():
                    continue
                k = np.flatnonzero(ok)
                rays_row.append(k)
                rays_c.append(contrib[k])
                rays_o.append(origin[k])
                rays_d.append(x[k] - origin[k])
                rays_t.append(np.full(k.size, SHADOW_END))
        if not rays_row:
            return out
        row = np.concatenate(rays_row)
        so = np.concatenate(rays_o)
        sd = np.concatenate(rays_d)
        st = np.concatenate(rays_t)
        contrib = np.concatenate(rays_c)
        lit = ps.visible(
            so,
            sd,
            st,
            it.inst[row].astype(np.int64),
            it.prim[row].astype(np.int64),
        )
        self.counts["shadow_rays"] += row.size
        np.add.at(out, row[lit], contrib[lit])
        return out

    # ------------------------------------------------------------------------ AO --

    def _ao(self, it, keys) -> np.ndarray:
        """``1 - strength x occlusion`` at camera hits (cosine-weighted, within ``distance``)."""
        t0 = time.perf_counter()
        ps, ao = self.ps, self.ao
        m = it.rows.shape[0]
        ng_side = it.ng * it.toward[:, None]
        origin = ps.offset_origin(it.p, ng_side)
        t, b = frame(ng_side)
        blocked = np.zeros(m)
        dirs, rows = [], []
        for j in range(ao.samples):
            u1 = uniform(keys, dimension(0, SLOT_AO + 2 * j))
            u2 = uniform(keys, dimension(0, SLOT_AO + 2 * j + 1))
            dirs.append(to_world(cosine_hemisphere(u1, u2), t, b, ng_side))
            rows.append(np.arange(m))
        row = np.concatenate(rows)
        d = np.concatenate(dirs)
        open_ = ps.visible(
            origin[row],
            d,
            np.full(row.size, ao.distance),
            it.inst[row].astype(np.int64),
            it.prim[row].astype(np.int64),
        )
        self.counts["ao_rays"] += row.size
        np.add.at(blocked, row, (~open_).astype(np.float64))
        self.timings["ao"] += time.perf_counter() - t0
        return 1.0 - ao.strength * blocked / ao.samples

    # -------------------------------------------------------------------- center --

    def center(self, n_pix: int) -> tuple[np.ndarray, np.ndarray]:
        """View depth and instance at every pixel center (one ray each)."""
        cam = self.cam
        depth = np.full(n_pix, np.inf)
        instance = np.full(n_pix, -1, dtype=np.int64)
        for start in range(0, n_pix, BATCH):
            pixel = np.arange(start, min(n_pix, start + BATCH), dtype=np.int64)
            row, col = np.divmod(pixel, cam.width)
            half = np.full(pixel.size, 0.5)
            o, d, tmin, tmax, _ = cam.rays(col, row, half, half)
            hits, _ = self.ps.trace(o, d, tmin, tmax, cull=self.cull)
            self.counts["center_rays"] += pixel.size
            hit = hits.hit
            depth[pixel[hit]] = hits.t[hit] * (d[hit] @ cam.forward)
            instance[pixel[hit]] = hits.inst[hit]
        return depth, instance


def _rows_of(it, sel: np.ndarray):
    """A view of interaction ``it`` restricted to ``sel`` (for MIS densities)."""
    from .scene import _subset

    return _subset(it, np.flatnonzero(sel))
