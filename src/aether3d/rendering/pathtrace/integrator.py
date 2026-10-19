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
from . import parallel
from .bsdf import BSDF, MIN_COS
from .lights import SHADOW_END, LightSet, mis_weight_emission
from .sampling import (
    SLOT_AO,
    SLOT_BSDF,
    SLOT_LENS,
    SLOT_LIGHTS,
    SLOT_LOBE,
    SLOT_PIXEL,
    SLOT_RR,
    cosine_hemisphere,
    dimension,
    filter_offset,
    filter_table,
    frame,
    power_heuristic,
    sampler_keys,
    to_world,
    uniform,
)
from .scene import MAX_LAYERS, PathScene

BATCH = 65_536
"""Paths traced together (bounds temporary memory)."""
PATH_BYTES = 1536
"""Upper estimate of the temporary memory of one path in flight (measured: ~800 bytes)."""
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
    ox: float = 0.0  # lens shift (NDC offset of the view's centre)
    oy: float = 0.0
    lens: float = 0.0  # aperture radius (m); 0: a pinhole
    focus: float = 10.0  # view depth of the plane in focus (m)
    blades: int = 0
    rotation: float = 0.0  # degrees
    ratio: float = 1.0

    def rays(self, col, row, u, v, lu=None, lv=None):
        """Unit camera rays through ``(col + u, row + v)``; ``(o, d, tmin, tmax, diff)``.
        With an aperture (``lens > 0``) and its uniform numbers ``lu, lv``, a ray leaves
        a point of the thin lens toward where the pinhole ray meets the plane in focus
        (the ray differentials stay the pinhole's)."""
        w, h = self.width, self.height
        xn = (col + u) * (2.0 / w) - 1.0 + self.ox
        yn = 1.0 - (row + v) * (2.0 / h) + self.oy
        sx = (xn * self.hx)[:, None]
        sy = (yn * self.hy)[:, None]
        n = xn.shape[0]
        step_x = (2.0 / w) * self.hx * self.scale
        step_y = -(2.0 / h) * self.hy * self.scale
        if self.perspective:
            d = self.forward + sx * self.right + sy * self.up
            if self.lens > 0.0 and lu is not None:
                from ..camera import aperture_points

                ax, ay = aperture_points(lu, lv, self.blades, self.rotation, self.ratio)
                offset = (ax * self.lens)[:, None] * self.right + (ay * self.lens)[
                    :, None
                ] * self.up
                # (d has a unit forward component: d * focus lies on the plane in focus)
                lens_d = d * self.focus - offset
                length = np.linalg.norm(lens_d, axis=1, keepdims=True)
                unit = lens_d / length
                along = (unit @ self.forward)[:, None]  # cosine to the view axis
                o = self.eye + offset
                length_p = np.linalg.norm(d, axis=1, keepdims=True)
                unit_p = d / length_p
                dd = []
                for vec in (self.right * step_x, self.up * step_y):
                    raw = np.broadcast_to(vec, (n, 3))
                    dd.append((raw - unit_p * (unit_p @ vec)[:, None]) / length_p)
                zero = np.zeros(3)
                diff = (zero, zero, dd[0], dd[1])
                return o, unit, self.near / along[:, 0], self.far / along[:, 0], diff
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
    ox, oy = cam.ndc_offset(width / height)
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
        ox,
        oy,
        cam.aperture_radius(),
        rs.camera.focus_distance,
        cam.aperture_blades,
        cam.aperture_rotation,
        cam.aperture_ratio,
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
    workers, _ = parallel.plan_workers(path.workers, rays)
    memory = (
        width * height * 200  # accumulators and passes
        + tris * 250  # world-space shading arrays and the BVH
        + workers * BATCH * PATH_BYTES  # one batch of paths in flight per process
    )
    return {"paths": int(paths), "rays": int(rays), "memory": int(memory), "workers": workers}


ADAPTIVE_FLOOR = 0.05
"""Luminance below which adaptive sampling measures absolute error (as if 0.05)."""


def adaptive_rounds(path: PathSettings) -> list[tuple[int, int]]:
    """Sample ranges of adaptive sampling's rounds: the first ``adaptive_min_samples``
    (default ``max(4, samples // 16)``), then doubling, up to ``samples``."""
    first = path.adaptive_min_samples or max(4, path.samples // 16)
    first = max(2, min(first, path.samples))
    rounds = [(0, first)]
    while rounds[-1][1] < path.samples:
        s0 = rounds[-1][1]
        rounds.append((s0, min(path.samples, 2 * s0)))
    return rounds


def converged(
    mean: np.ndarray, variance_of_mean: np.ndarray, width: int, height: int, threshold: float
) -> np.ndarray:
    """Pixels whose estimate -- and every 3 x 3 neighbour's -- has a relative luminance
    error below ``threshold`` (``mean``, ``variance_of_mean``: ``(n_pix, 3)``)."""
    lum = np.maximum(mean @ LUMINANCE, ADAPTIVE_FLOOR)
    err = np.sqrt(np.maximum(variance_of_mean @ (LUMINANCE * LUMINANCE), 0.0)) / lum
    err = np.where(np.isfinite(err), err, np.inf).reshape(height, width)
    padded = np.pad(err, 1, mode="edge")
    worst = np.zeros_like(err)
    for dy in range(3):
        for dx in range(3):
            worst = np.maximum(worst, padded[dy : dy + height, dx : dx + width])
    return (worst < threshold).reshape(-1)


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
    workers, why = parallel.plan_workers(path.workers, work["rays"])
    acc = parallel.Accumulators(n_pix, shared=workers > 1)
    visible = np.zeros(len(rs.instances), dtype=np.int64)
    # One round of all samples; with a progress callback, one pass per round; adaptive
    # sampling: doubling rounds over the pixels that have not converged.
    adaptive = path.adaptive_threshold > 0.0 and path.integrator == "path"
    if adaptive:
        rounds = adaptive_rounds(path)
    elif progress is None:
        rounds = [(0, path.samples)]
    else:
        rounds = [(s, s + 1) for s in range(path.samples)]
    active = np.arange(n_pix, dtype=np.int64)
    passes_done = 0
    stopped = False
    paths = 0
    busy = 0.0
    pids: set[int] = set()
    # Totals of the tasks' own counts (tasks may run in this process or in workers).
    ray_counts: dict[str, int] = {}
    trace_timings: dict[str, float] = {}
    pass_through = layer_limited = 0
    t_trace = time.perf_counter()
    try:
        with parallel.Runner(tracer, acc, workers, BATCH) as runner:
            for s0, s1 in rounds:
                if adaptive and s0 > 0:  # drop the pixels that have converged
                    n_now = np.array(acc["samples"])  # (a copy: the shared sums close later)
                    safe = np.maximum(n_now, 1.0)[:, None]
                    mean_now = acc["beauty"] / safe
                    var_now = (
                        np.maximum(acc["squares"] / safe - mean_now * mean_now, 0.0)
                        / (np.maximum(n_now - 1.0, 1.0)[:, None])
                    )
                    done_mask = converged(mean_now, var_now, width, height,
                                          path.adaptive_threshold)  # fmt: skip
                    active = np.flatnonzero(~done_mask)
                    if not active.size:
                        break
                n_act = int(active.size)
                count = parallel.tile_count(n_act, s1 - s0, workers, BATCH)
                tiles = parallel.tiles(n_act, count)
                if adaptive:
                    tasks = [(a, b, s0, s1, active[a:b]) for a, b in tiles]
                else:
                    tasks = [(a, b, s0, s1) for a, b in tiles]
                for result in runner.run(tasks):
                    for k, v in result["counts"].items():
                        ray_counts[k] = ray_counts.get(k, 0) + v
                    for k, v in result["timings"].items():
                        trace_timings[k] = trace_timings.get(k, 0.0) + v
                    pass_through += result["pass_through_rays"]
                    layer_limited += result["layer_limited"]
                    visible += result["visible"]
                    paths += result["paths"]
                    busy += result["seconds"]
                    pids.add(result["pid"])
                passes_done = s1
                if (
                    progress is not None
                    and progress(passes_done, path.samples, time.perf_counter() - t_start) is False
                    and passes_done < path.samples
                ):
                    stopped = True
                    break
        sums = acc.copy()
    finally:
        acc.close()
    t_trace = time.perf_counter() - t_trace
    spp = passes_done
    if spp == 0:  # pragma: no cover - every round completes at least one pass
        raise RenderError("RENDER_FAILED", "no samples were rendered")
    acc = sums
    t0 = time.perf_counter()
    before = (ps.pass_through_rays, ps.layer_limited, tracer.counts["center_rays"])
    depth, instance = tracer.center(n_pix)
    pass_through += ps.pass_through_rays - before[0]
    layer_limited += ps.layer_limited - before[1]
    ray_counts["center_rays"] = tracer.counts["center_rays"] - before[2]
    t_center = time.perf_counter() - t0
    if adaptive:  # every pixel has its own count
        n_px = np.maximum(acc["samples"], 1.0)
        inv = (1.0 / n_px)[:, None]
        mean = acc["beauty"] * inv
        variance = (
            np.maximum(acc["squares"] * inv - mean * mean, 0.0)
            / np.maximum(n_px - 1.0, 1.0)[:, None]
        )
        variance[n_px < 2] = np.inf
    else:
        inv = 1.0 / spp
        mean = acc["beauty"] * inv
        if spp > 1:
            variance = np.maximum(acc["squares"] * inv - mean * mean, 0.0) * (1.0 / (spp - 1))
        else:
            variance = np.full((n_pix, 3), np.inf)
    inv1 = inv[:, 0] if adaptive else inv
    shape = (height, width)
    passes = {
        "beauty": mean.reshape(*shape, 3),
        "direct": (acc["direct"] * inv).reshape(*shape, 3),
        "indirect": (acc["indirect"] * inv).reshape(*shape, 3),
        "albedo": (acc["albedo"] * inv).reshape(*shape, 3),
        "normal": (acc["normal"] * inv).reshape(*shape, 3),
        "depth": depth.reshape(shape),
        "sample_count": (
            acc["samples"].astype(np.int32).reshape(shape)
            if adaptive
            else np.full(shape, spp, dtype=np.int32)
        ),
        "variance": variance.reshape(*shape, 3),
    }
    if ao is not None:
        passes["ao"] = (acc["ao"] * inv1).reshape(shape)
    if path.integrator == "ao":
        passes["beauty"] = passes["direct"]
    coverage = (acc["hits"] * inv1).astype(np.float32).reshape(shape)
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
    counts = {k: ray_counts.get(k, 0) for k in tracer.counts}
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
    if adaptive:
        counts["samples_mean"] = float(acc["samples"].mean())
        counts["pixels_converged_early"] = int(np.count_nonzero(acc["samples"] < spp))
    counts["pass_through_rays"] = pass_through
    if layer_limited:
        diagnostics.append(
            Diagnostic(
                "PASS_THROUGH_LIMIT",
                WARNING,
                f"{layer_limited} rays crossed more than {MAX_LAYERS} cut-out or "
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
            # CPU seconds of the tracing stages, summed over the worker processes:
            **{f"trace_{k}": v for k, v in trace_timings.items()},
        },
        diagnostics=diagnostics,
        info={
            "workers": len(pids),
            "workers_planned": workers,
            "parallel": why,
            "cpu_seconds": busy,
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
        """Trace the camera paths of ``(pixel, sample)`` pairs (see the module docstring);
        with the atmosphere's fog, every path's first segment is seen through it."""
        out, d, first = self._trace(pixel, sample)
        model = self.ps.atmosphere
        if model is not None and model.fog:
            from ..atmosphere import fog, transmittance

            out["direct"] = fog(model, out["direct"], d, first)
            out["indirect"] = out["indirect"] * transmittance(model, d, first)[:, None]
        return out

    def _trace(self, pixel: np.ndarray, sample: np.ndarray):
        ps, cam, path = self.ps, self.cam, self.path
        n = pixel.shape[0]
        keys = sampler_keys(path.sampler, path.seed, pixel, sample)
        row, col = np.divmod(pixel, cam.width)
        u = uniform(keys, dimension(0, SLOT_PIXEL))
        v = uniform(keys, dimension(0, SLOT_PIXEL + 1))
        if path.filter != "box":  # filter importance sampling: offsets from the pixel's centre
            table = filter_table(path.filter, path.filter_width)
            u = 0.5 + filter_offset(table, u)
            v = 0.5 + filter_offset(table, v)
        lu = lv = None
        if cam.lens > 0.0:
            lu = uniform(keys, dimension(0, SLOT_LENS))
            lv = uniform(keys, dimension(0, SLOT_LENS + 1))
        t0 = time.perf_counter()
        o, d, tmin, tmax, diff = cam.rays(col, row, u, v, lu, lv)
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
        first = np.full(n, np.inf)  # distance to the first surface seen (fog)
        if it is None:
            return out, d, first
        diff_p = it.p - cam.eye
        first[it.rows] = np.sqrt(np.sum(diff_p * diff_p, axis=1))
        if self.ao is not None:
            out["ao"][it.rows] = self._ao(it, keys[it.rows])
        if path.integrator == "ao":
            out["direct"][it.rows] = out["ao"][it.rows, None]
            return out, d, first
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
                n_shade,
                wo,
                sh.base,
                sh.metallic,
                sh.roughness,
                sh.specular,
                sh.has_specular,
                sh.transmission,
                sh.ior,
                it.toward > 0.0,
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
            # Reflections leave on the geometric surface's side, refractions through it.
            through = (wi * n_shade).sum(axis=1) < 0.0
            valid &= ((wi * ng_side).sum(axis=1) > 0.0) != through
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
            side = np.where(through[keep, None], -ng_side[keep], ng_side[keep])
            origin = ps.offset_origin(it.p[keep], side)
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
                le = ps.environment.radiance(direction[miss])
                if ps.environment.sampled:  # also sampled as a light: weight both strategies
                    w = power_heuristic(pdf[keep][miss], ps.environment.image.pdf(direction[miss]))
                    le = le * w[:, None]
                target[m] += throughput[m] * le
            if self.lights.sized:  # suns and light spheres the BSDF sample reaches
                target[r] += throughput[r] * self.lights.hit_sized(
                    origin, direction, hits.t, pdf[keep], it.inst[keep]
                )
            if nxt is None:
                break
            nxt.rows = r[nxt.rows]
            it = nxt
        if path.clamp > 0.0:
            peak = out["indirect"].max(axis=1)
            scale = np.where(peak > path.clamp, path.clamp / np.maximum(peak, 1e-300), 1.0)
            out["indirect"] *= scale[:, None]
        return out, d, first

    # ---------------------------------------------------------- direct lighting --

    def _direct(self, it, bsdf: BSDF, ng_side, keys, vertex: int) -> np.ndarray:
        """Next-event estimate at the interactions: ``(m, 3)`` (before throughput)."""
        ps, lights = self.ps, self.lights
        m = it.rows.shape[0]
        out = np.zeros((m, 3))
        ns = self.path.light_samples
        origin = ps.offset_origin(it.p, ng_side)
        transmits = bsdf.transmits
        if transmits.any():  # lights seen through glass leave from its other side
            inside = ps.offset_origin(it.p, -ng_side)
        rays_o, rays_d, rays_t, rays_row, rays_c, rays_test = [], [], [], [], [], []
        index = 0
        for light in range(lights.count):
            occluded = lights.shadow is None or bool(lights.shadow[light])
            receivers = lights.receivers_at(light, it.inst)  # light linking
            for _ in range(ns):
                wi, target, scale, pdf = lights.sample_analytic(light, it.p, keys, vertex, index)
                index += 1
                f, pdf_bsdf, cos = bsdf.eval(wi)
                above = (wi * ng_side).sum(axis=1) > 0.0
                ok = (cos > 0.0) & (above | transmits)
                if receivers is not None:
                    ok &= receivers
                # Sized lights are also found by BSDF sampling: weight both strategies.
                finite = np.isfinite(pdf)
                w = np.ones(pdf.shape[0])
                w[finite] = power_heuristic(pdf[finite], pdf_bsdf[finite])
                contrib = f * scale * (cos * w / ns)[:, None]
                ok &= contrib.any(axis=1)
                if not ok.any():
                    continue
                k = np.flatnonzero(ok)
                rays_row.append(k)
                rays_c.append(contrib[k])
                o = (
                    origin[k]
                    if not transmits.any()
                    else np.where(above[k, None], origin[k], inside[k])
                )
                rays_o.append(o)
                rays_test.append(np.full(k.size, occluded))
                if target is None:
                    rays_d.append(wi[k])
                    rays_t.append(np.full(k.size, np.inf))
                else:
                    rays_d.append(target[k] - o)
                    rays_t.append(np.full(k.size, SHADOW_END))
        env = ps.environment
        if env.sampled:  # an image environment: directions by its luminance
            for _ in range(ns):
                slot = SLOT_LIGHTS + 3 * index
                index += 1
                u0 = uniform(keys, dimension(vertex, slot))
                u1 = uniform(keys, dimension(vertex, slot + 1))
                wi, pdf = env.image.sample(u0, u1)
                f, pdf_bsdf, cos = bsdf.eval(wi)
                above = (wi * ng_side).sum(axis=1) > 0.0
                ok = (cos > 0.0) & (pdf > 0.0) & (above | transmits)
                if not ok.any():
                    continue
                w = power_heuristic(pdf, pdf_bsdf)
                le = env.radiance(wi)
                contrib = f * le * (cos * w / (np.where(ok, pdf, 1.0) * ns))[:, None]
                ok &= contrib.any(axis=1)
                k = np.flatnonzero(ok)
                rays_row.append(k)
                rays_c.append(contrib[k])
                rays_o.append(origin[k] if not transmits.any() else
                              np.where(above[k, None], origin[k], inside[k]))  # fmt: skip
                rays_test.append(np.ones(k.size, bool))
                rays_d.append(wi[k])
                rays_t.append(np.full(k.size, np.inf))
        if lights.emitters is not None:
            for _ in range(ns):
                wi, x, le_over_pdf, pdf, _, _ = lights.sample_emitter(ps, it.p, keys, vertex, index)
                index += 1
                f, pdf_bsdf, cos = bsdf.eval(wi)
                above = (wi * ng_side).sum(axis=1) > 0.0
                ok = (cos > 0.0) & (pdf > 0.0) & (above | transmits)
                w = power_heuristic(pdf, pdf_bsdf)
                contrib = f * le_over_pdf * (cos * w / ns)[:, None]
                ok &= contrib.any(axis=1)
                if not ok.any():
                    continue
                k = np.flatnonzero(ok)
                rays_row.append(k)
                rays_c.append(contrib[k])
                o = (
                    origin[k]
                    if not transmits.any()
                    else np.where(above[k, None], origin[k], inside[k])
                )
                rays_o.append(o)
                rays_test.append(np.ones(k.size, bool))
                rays_d.append(x[k] - o)
                rays_t.append(np.full(k.size, SHADOW_END))
        if not rays_row:
            return out
        row = np.concatenate(rays_row)
        so = np.concatenate(rays_o)
        sd = np.concatenate(rays_d)
        st = np.concatenate(rays_t)
        contrib = np.concatenate(rays_c)
        test = np.concatenate(rays_test)
        lit = np.ones(row.size, bool)  # lights without shadows are never occluded
        if test.any():
            tested = np.flatnonzero(test)
            lit[tested] = ps.visible(
                so[tested],
                sd[tested],
                st[tested],
                it.inst[row[tested]].astype(np.int64),
                it.prim[row[tested]].astype(np.int64),
            )
            self.counts["shadow_rays"] += tested.size
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
