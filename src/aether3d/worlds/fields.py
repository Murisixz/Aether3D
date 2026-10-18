"""The world model: a definition compiled into pure field functions of world coordinates.

:class:`WorldModel` evaluates, for any float64 points ``(x, y)`` in world metres:

* the **shape height** -- continents, mountain ranges, ridged mountains, hills, valleys,
  plateaus, canyons, the optional height curve and shape-stage features
  (:meth:`WorldModel.shape`); the regional stage (:mod:`.hydrology`) erodes this surface;
* the **climate** -- temperature at sea level and moisture (with gradients, water
  influence and rain shadows) on a global lattice of :data:`CLIMATE_SPACING` metres,
  interpolated bilinearly; temperature then drops with each point's own elevation;
* the **biome weights** -- smooth Whittaker-style windows in (temperature, moisture),
  normalized, with boundary noise and ``biome`` overrides;
* the **final height** -- the eroded shape plus biome terrain modifiers (dunes, swamp
  flattening, roughness), detail noise, river channels and final-stage features
  (:meth:`WorldModel.evaluate`).

Everything is a pure function of the definition and the point: there is no state that
depends on evaluation order (the climate lattice and the regional stage are memoized,
deterministic computations). Only correctly rounded float operations are used (see
:mod:`.noise`), so a point evaluates to the same bits anywhere.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import WorldError
from . import noise as nz
from .definition import biome_table, digest
from .features import (
    apply_final,
    apply_shape,
    biome_overrides,
    compile_features,
)

CLIMATE_SPACING = 64.0
"""Metres between climate lattice points (climate varies over kilometres)."""
CLIMATE_TILE = 32
"""Lattice points per side of a cached climate block."""
UPWIND = (1500.0, 4000.0, 9000.0)
"""Distances (m) at which upwind relief is sampled for rain shadows."""
MACRO_CELLS = 4.0
"""Fractal octaves with wavelengths of at least this many hydrology cells are eroded by
the regional stage; shorter ones are detail added on top of the eroded surface."""


def _rgb(hex_color: str) -> np.ndarray:
    """A display colour as linear RGB (the sRGB transfer function, exactly as rendering)."""
    from ..rendering.color import to_linear

    return np.asarray(to_linear(tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))))


@dataclass
class Context:
    """What long generation steps may consult: cancellation, progress and a deadline."""

    should_stop: Any = None
    progress: Any = None
    deadline: float | None = None

    def check(self, where: str = "generation") -> None:
        import time

        if self.should_stop is not None and self.should_stop():
            raise WorldError("GENERATION_CANCELLED", f"{where} was cancelled", where)
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise WorldError(
                "TIME_LIMIT",
                f"{where} exceeded its time budget",
                where,
                hint="raise the time limit, lower the resolution or generate a smaller area",
            )


class Sample(dict):
    """Fields at points (a dict of arrays): ``height``, ``shape``, ``temperature``,
    ``moisture``, ``weights`` (points x biomes), ``water`` (water level or ``nan``),
    ``water_kind`` (0 none, 1 sea, 2 lake, 3 river, 4 authored), ``river`` (distance to the
    nearest river centre, ``inf`` if none), ``road`` (0..1), ``land`` (0..1)."""


class WorldModel:
    """A world definition compiled into field functions (see the module docstring)."""

    def __init__(self, definition: dict) -> None:
        self.d = definition
        self.key = digest(definition)
        self.shape_key = digest(
            {k: definition[k] for k in ("seed", "generator", "extent", "sea_level", "terrain")},
            [f for f in definition["features"] if f["type"] in _SHAPE_TYPES],
        )
        self.seed = int(definition["seed"])
        self.sea = float(definition["sea_level"])
        s = self.seed
        self.seeds = {
            name: nz.derive(s, "terrain", name)
            for name in (
                "warp",
                "continents",
                "continents-warp",
                "ranges",
                "ranges-warp",
                "mountains",
                "mountains-warp",
                "hills",
                "valleys",
                "plateaus",
                "plateau-zone",
                "canyons",
                "canyon-zone",
                "dunes",
                "detail",
                "temperature",
                "moisture",
                "boundary-t",
                "boundary-m",
            )
        }
        t = definition["terrain"]
        self.curve_continents = nz.MonotoneCurve(t["continents"]["curve"])
        self.curve_final = None if t["curve"] is None else nz.MonotoneCurve(t["curve"])
        from .structures import pad_features

        self.features = compile_features(
            definition["features"] + pad_features(definition["structures"]["items"])
        )
        self._auto: dict[str, float] | None = None  # resolved automatic heights by feature id
        self._auto_lock = threading.Lock()
        self.shape_features = [f for f in self.features if f.stage == "shape"]
        self.final_features = [f for f in self.features if f.stage == "final"]
        # biomes
        table = biome_table(definition)
        self.biome_names = list(table)
        self.biomes = table
        b = list(table.values())
        self.b_t = np.array([v["temperature"] for v in b], dtype=np.float64)  # (B, 2)
        self.b_m = np.array([v["moisture"] for v in b], dtype=np.float64)
        self.b_dunes = np.array([v["terrain"]["dunes"] for v in b], dtype=np.float64)
        self.b_flatten = np.array([v["terrain"]["flatten"] for v in b], dtype=np.float64)
        self.b_rough = np.array([v["terrain"]["roughness"] for v in b], dtype=np.float64)
        self.colors = {
            key: np.stack([_rgb(v["colors"][key]) for v in b])
            for key in ("grass", "soil", "rock", "sand", "foliage", "water")
        }
        self._climate: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._regional: Any = None
        self._regional_lock = threading.Lock()
        hy = definition["hydrology"]
        self.cutoff = MACRO_CELLS * hy["cell"] if hy["mode"] == "drainage" else None

    # ------------------------------------------------------------------------------- #
    # Bounds
    # ------------------------------------------------------------------------------- #

    @property
    def bounded(self) -> bool:
        return self.d["extent"] is not None

    def extent(self) -> tuple[float, float, float, float] | None:
        e = self.d["extent"]
        return None if e is None else (e["min"][0], e["min"][1], e["max"][0], e["max"][1])

    def inside(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        e = self.extent()
        if e is None:
            return np.ones(np.shape(x), dtype=bool)
        return (x >= e[0]) & (x <= e[2]) & (y >= e[1]) & (y <= e[3])

    # ------------------------------------------------------------------------------- #
    # Shape
    # ------------------------------------------------------------------------------- #

    def shape(self, x: Any, y: Any, aux: dict | None = None) -> np.ndarray:
        """The shape height at points (before erosion, detail and final features)."""
        return self.shape_parts(x, y, aux)[1]

    def _split(self, scale: float, octaves: int) -> int | None:
        """How many octaves of a fractal of feature size ``scale`` are *macro*: wavelength
        at least :data:`MACRO_CELLS` hydrology cells (``None``: no regional stage)."""
        if self.cutoff is None:
            return None
        k = 0
        wavelength = scale
        while k < octaves and wavelength >= self.cutoff:
            k += 1
            wavelength /= 2.0
        return k

    def shape_parts(
        self, x: Any, y: Any, aux: dict | None = None, features: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(macro, full)`` shape heights at points. *Macro* keeps only the octaves of
        every fractal whose wavelength is at least :data:`MACRO_CELLS` hydrology cells: it
        is what the regional stage erodes; ``full - macro`` is the detail added on top of
        the eroded surface (the same arrays when there is no regional stage)."""
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
        t = self.d["terrain"]
        S = t["scale"]
        HS = t["height_scale"]
        sd = self.seeds
        w = t["warp"]
        wx, wy = nz.warp2(
            sd["warp"],
            x,
            y,
            amplitude=w["amplitude"] * S,
            frequency=1.0 / (w["scale"] * S),
            octaves=3,
        )
        # Continents: the macro land/ocean field through its height curve.
        ct = t["continents"]
        cs = ct["scale"] * S
        cx, cy = nz.warp2(
            sd["continents-warp"],
            wx,
            wy,
            amplitude=ct["warp"] * cs * 0.25,
            frequency=1.0 / cs,
            octaves=2,
        )
        c = nz.fbm2(sd["continents"], cx, cy, octaves=ct["octaves"], frequency=1.0 / cs)
        c = c * 1.6 + (ct["land"] - 0.5) * 1.6
        e = self.extent()
        if e is not None and ct["edge"] > 0.0:
            inner = np.minimum(np.minimum(x - e[0], e[2] - x), np.minimum(y - e[1], e[3] - y))
            c = c - (1.0 - nz.smoothstep(0.0, ct["edge"], inner)) * 2.2
        base = self.sea + self.curve_continents(c) * HS
        hm = base.copy()  # macro
        hf = base  # full
        land = nz.smoothstep(-0.08, 0.22, c)
        # Mountain ranges: chains along the zero lines of a warped fractal.
        r = t["ranges"]
        rs = r["scale"] * S
        rx, ry = nz.warp2(
            sd["ranges-warp"], wx, wy, amplitude=r["warp"] * rs * 0.3, frequency=1.5 / rs, octaves=2
        )
        rv = np.abs(nz.fbm2(sd["ranges"], rx, ry, octaves=3, frequency=1.0 / rs))
        # A band of full mountains around each zero line (its core), falling off over the
        # rest of the band; sharpness narrows the fall-off (steeper range fronts).
        tau = 0.02 + 0.32 * r["coverage"]
        core = tau * (0.25 + 0.5 * r["sharpness"])
        ranges = (1.0 - nz.smoothstep(core, tau, rv)) * land
        # Mountains: broad massifs carrying ridged and eroded fractal detail.
        mt = t["mountains"]
        if mt["height"] > 0.0 and r["coverage"] > 0.0:
            ms = mt["scale"] * S
            mx, my = nz.warp2(
                sd["mountains-warp"],
                wx,
                wy,
                amplitude=mt["warp"] * ms * 0.5,
                frequency=1.0 / ms,
                octaves=2,
            )
            massif = 0.5 + 0.75 * nz.fbm2(
                sd["mountains"] ^ 2, mx, my, octaves=3, frequency=0.45 / ms
            )
            massif = nz.smoothstep(0.15, 0.95, massif)
            split = self._split(ms, mt["octaves"])
            rid = nz.ridged2(
                sd["mountains"],
                mx,
                my,
                octaves=mt["octaves"],
                frequency=1.0 / ms,
                gain=mt["roughness"],
                sharpness=mt["sharpness"],
                split=split,
            )
            rid_m, rid_f = rid if split is not None else (rid, rid)
            if mt["ridged"] < 1.0:  # the rest: eroded fBm (smooth flanks, gullies)
                ero = nz.eroded2(
                    sd["mountains"] ^ 1,
                    mx,
                    my,
                    octaves=mt["octaves"],
                    frequency=1.0 / ms,
                    gain=mt["roughness"],
                    split=split,
                )
                ero_m, ero_f = ero if split is not None else (ero, ero)
                k = mt["ridged"]
                rid_m = k * rid_m + (1.0 - k) * np.clip(0.5 + 1.6 * ero_m, 0.0, 1.0)
                rid_f = k * rid_f + (1.0 - k) * np.clip(0.5 + 1.6 * ero_f, 0.0, 1.0)
            amp = mt["height"] * HS * ranges
            for part, rv_ in ((0, rid_m), (1, rid_f)):
                shape = massif * (0.35 + 0.65 * rv_)
                add = amp * shape * (0.6 + 0.4 * shape)
                if part == 0:
                    hm = hm + add
                else:
                    hf = hf + add
        # Hills.
        hl = t["hills"]
        if hl["height"] > 0.0:
            hs = hl["scale"] * S
            split = self._split(hs, hl["octaves"])
            hv = nz.fbm2(
                sd["hills"], wx, wy, octaves=hl["octaves"], frequency=1.0 / hs, split=split
            )
            hv_m, hv_f = hv if split is not None else (hv, hv)
            k = hl["height"] * HS * land * (1.0 - 0.5 * ranges)
            hm = hm + k * (hv_m + 0.35)
            hf = hf + k * (hv_f + 0.35)
        # Valleys: winding carves along the zero lines of another fractal.
        vl = t["valleys"]
        vv = None
        if vl["depth"] > 0.0 or self.d["hydrology"]["mode"] == "contour":
            vv = np.abs(
                nz.fbm2(sd["valleys"], wx, wy, octaves=4, frequency=1.0 / (vl["scale"] * S))
            )
            if vl["depth"] > 0.0:
                carve = vl["depth"] * HS * (1.0 - nz.smoothstep(0.0, vl["width"], vv)) * land
                hm = hm - carve
                hf = hf - carve
        # Plateaus (raised, terraced zones) and canyons.
        pl = t["plateaus"]
        pzone = None
        if pl["coverage"] > 0.0:
            pv = 0.5 + 0.8 * nz.fbm2(
                sd["plateau-zone"], wx, wy, octaves=3, frequency=1.0 / (pl["scale"] * S)
            )
            k = 1.0 - pl["coverage"]
            pzone = nz.smoothstep(k - 0.04, k + 0.04, pv) * land
            hm = hm + pl["height"] * HS * pzone
            hf = hf + pl["height"] * HS * pzone
        cn = t["canyons"]
        if cn["coverage"] > 0.0:
            zv = 0.5 + 0.8 * nz.fbm2(
                sd["canyon-zone"], wx, wy, octaves=2, frequency=0.5 / (cn["scale"] * S)
            )
            k = 1.0 - cn["coverage"]
            zone = nz.smoothstep(k - 0.1, k + 0.1, zv) * land
            cv = np.abs(
                nz.fbm2(sd["canyons"], wx, wy, octaves=4, frequency=1.0 / (cn["scale"] * S))
            )
            prof = 1.0 - nz.smoothstep(cn["width"] * 0.35, cn["width"], cv)
            cut = cn["depth"] * HS * prof * zone
            hm = hm - cut
            hf = hf - cut
        out = []
        for h in (hm, hf):
            if pzone is not None and pl["step"] > 0.0:
                stepped = nz.terrace(h - self.sea, pl["step"] * HS, pl["sharpness"]) + self.sea
                h = h + pzone * (stepped - h)
            if self.curve_final is not None:
                h = self.sea + self.curve_final(h - self.sea)
            if self.shape_features and features:
                h = apply_shape(self.shape_features, x, y, h, self.sea, self.seed)
            out.append(h)
        if aux is not None:
            aux["land"] = land
            aux["continent"] = c
            aux["ranges"] = ranges
            if vv is not None:
                aux["valley"] = vv
        return out[0], out[1]

    # ------------------------------------------------------------------------------- #
    # Climate (on a global lattice)
    # ------------------------------------------------------------------------------- #

    def _climate_block(self, tx: int, ty: int) -> np.ndarray:
        key = (tx, ty)
        with self._lock:
            found = self._climate.get(key)
            if found is not None:
                self._climate.move_to_end(key)
                return found
        n = CLIMATE_TILE + 1
        j, i = np.mgrid[0:n, 0:n]
        x = (tx * CLIMATE_TILE + i).astype(np.float64).reshape(-1) * CLIMATE_SPACING
        y = (ty * CLIMATE_TILE + j).astype(np.float64).reshape(-1) * CLIMATE_SPACING
        values = self._climate_points(x, y).reshape(2, n, n)
        values.flags.writeable = False
        with self._lock:
            self._climate[key] = values
            while len(self._climate) > 512:
                self._climate.popitem(last=False)
        return values

    def _climate_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``(2, N)``: temperature at sea level and moisture at lattice points."""
        c = self.d["climate"]
        sd = self.seeds
        fs = 1.0 / c["scale"]
        temp = (
            c["temperature"]
            + (c["gradient"][0] * x + c["gradient"][1] * y) / 1000.0
            + c["variation"] * 1.5 * nz.fbm2(sd["temperature"], x, y, octaves=3, frequency=fs)
        )
        moist = (
            c["moisture"]
            + (c["moisture_gradient"][0] * x + c["moisture_gradient"][1] * y) / 1000.0
            + c["moisture_variation"] * 1.6 * nz.fbm2(sd["moisture"], x, y, octaves=3, frequency=fs)
        )
        bn = c["boundary_noise"]
        if bn > 0.0:
            temp = temp + bn * 14.0 * nz.fbm2(sd["boundary-t"], x, y, octaves=3, frequency=6.0 * fs)
            moist = moist + bn * 0.7 * nz.fbm2(
                sd["boundary-m"], x, y, octaves=3, frequency=6.0 * fs
            )
        h = self.shape(x, y)
        if c["water_influence"] > 0.0:
            dist = self.water_distance(x, y, h)
            moist = moist + c["water_influence"] * (
                1.0 - nz.smoothstep(0.0, c["water_distance"], dist)
            )
        if c["rain_shadow"] > 0.0:
            # The wind comes FROM ``wind`` degrees (counter-clockwise from +x).
            ux, uy = nz.cos_sin_deg(c["wind"])
            # (The procedural relief only: authored features change the climate where
            # they are, not kilometres downwind, so a feature stays a local change.)
            upwind = np.zeros_like(h)
            for dist in UPWIND:
                hu = self.shape_parts(x + ux * dist, y + uy * dist, features=False)[1]
                upwind = np.maximum(upwind, hu)
            relief = np.clip((upwind - np.maximum(h, self.sea)) / 900.0, 0.0, 1.0)
            moist = moist - c["rain_shadow"] * 0.6 * relief
        return np.stack([temp, np.clip(moist, 0.0, 1.0)])

    def climate(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Temperature at sea level and moisture at points (bilinear on the lattice)."""
        gx = x / CLIMATE_SPACING
        gy = y / CLIMATE_SPACING
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        fx = gx - ix
        fy = gy - iy
        tx = np.floor_divide(ix, CLIMATE_TILE)
        ty = np.floor_divide(iy, CLIMATE_TILE)
        lx = ix - tx * CLIMATE_TILE
        ly = iy - ty * CLIMATE_TILE
        temp = np.empty(x.shape)
        moist = np.empty(x.shape)
        keys = tx * 1_000_003 + ty
        for key in np.unique(keys):
            sel = keys == key
            bx = int(tx[sel][0])
            by = int(ty[sel][0])
            block = self._climate_block(bx, by)
            a, b = lx[sel], ly[sel]
            u, v = fx[sel], fy[sel]
            for k, out in ((0, temp), (1, moist)):
                g = block[k]
                v00 = g[b, a]
                v10 = g[b, a + 1]
                v01 = g[b + 1, a]
                v11 = g[b + 1, a + 1]
                top = v00 + u * (v10 - v00)
                bot = v01 + u * (v11 - v01)
                out[sel] = top + v * (bot - top)
        return temp, moist

    def water_distance(self, x: np.ndarray, y: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Distance (m) to water: from the regional stage where it exists, else estimated
        from the depth of the land below the sea and the continent field (heuristic)."""
        reg = self.regional() if self.d["hydrology"]["mode"] == "drainage" else None
        if reg is not None:
            return reg.water_distance(x, y)
        above = np.maximum(h - self.sea, 0.0)
        return above * 40.0

    # ------------------------------------------------------------------------------- #
    # Biomes
    # ------------------------------------------------------------------------------- #

    def biome_weights(self, temp: np.ndarray, moist: np.ndarray) -> np.ndarray:
        """``(N, B)`` weights from temperature (at the point's height) and moisture."""
        blend = self.d["biomes"]["blend"]
        bt = blend * 20.0
        bm = blend
        t = temp[:, None]
        m = moist[:, None]
        t0, t1 = self.b_t[None, :, 0], self.b_t[None, :, 1]
        m0, m1 = self.b_m[None, :, 0], self.b_m[None, :, 1]
        wt = nz.smoothstep(0.0, 1.0, (t - t0 + bt) / (2.0 * bt)) * (
            1.0 - nz.smoothstep(0.0, 1.0, (t - t1 + bt) / (2.0 * bt))
        )
        wm = nz.smoothstep(0.0, 1.0, (m - m0 + bm) / (2.0 * bm)) * (
            1.0 - nz.smoothstep(0.0, 1.0, (m - m1 + bm) / (2.0 * bm))
        )
        w = wt * wm
        # A faint nearest-envelope term keeps weights smooth in gaps between envelopes.
        dt = (np.maximum(np.maximum(t0 - t, t - t1), 0.0)) / 20.0
        dm = np.maximum(np.maximum(m0 - m, m - m1), 0.0)
        d2 = dt * dt + dm * dm
        q = 1.0 / (1.0 + d2 / 0.0025)
        w = w + 1e-4 * q * q
        return w / w.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------------------- #
    # Regional stage
    # ------------------------------------------------------------------------------- #

    def regional_if_ready(self) -> Any:
        return self._regional

    def regional(self, ctx: Context | None = None) -> Any:
        """The regional stage's results (drainage mode), computed once (thread-safe)."""
        if self.d["hydrology"]["mode"] != "drainage":
            return None
        if self._regional is not None:
            return self._regional
        with self._regional_lock:
            if self._regional is None:
                from .hydrology import compute_regional

                self._regional = compute_regional(self, ctx or Context())
        return self._regional

    def attach_regional(self, regional: Any) -> None:
        """Use a regional result computed elsewhere (a cache) for this definition."""
        self._regional = regional

    # ------------------------------------------------------------------------------- #
    # Everything
    # ------------------------------------------------------------------------------- #

    def natural(self, x: Any, y: Any, ctx: Context | None = None) -> np.ndarray:
        """The natural ground at points: the height before final features (pads, roads,
        lakes, rivers...) -- what structures and roads are fitted to."""
        return self.evaluate(x, y, ctx, final=False)["height"]

    def pad_height(self, fid: str) -> float:
        """The height of a pad or flatten feature (authored, or resolved automatically)."""
        self._resolve()
        for c in self.final_features:
            if c.spec["id"] == fid:
                h = c.spec.get("height")
                return float(h if h is not None else c.data["height"])
        raise KeyError(fid)

    def _resolve(self) -> None:
        """Resolve automatic heights once per model, from the natural ground: pads and
        flattens without a height (the median over their footprint for structures, the
        centre otherwise) and roads given as ``[x, y]`` lines (densified every 4 m and
        given smoothed natural heights)."""
        if self._auto is not None:
            return
        with self._auto_lock:
            if self._auto is not None:
                return
            from .structures import auto_height

            items = {f"structure:{i['id']}": i for i in self.d["structures"]["items"]}
            done: dict[str, float] = {}
            for c in self.final_features:
                s = c.spec
                if s["type"] in ("pad", "flatten") and s.get("height") is None:
                    item = items.get(s["id"])
                    if item is not None:
                        h = auto_height(self, item)
                    else:
                        cx, cy = _centre_of(s["shape"])
                        h = float(self.natural(np.array([cx]), np.array([cy]))[0])
                    c.data["height"] = round(h, 4)
                    done[s["id"]] = c.data["height"]
                elif s["type"] == "road" and "z" not in c.data:
                    line = _densify(c.data["line"], 4.0)
                    z = self.natural(line[:, 0], line[:, 1])
                    k = min(9, len(z))
                    if k >= 3:  # moving average (ends kept)
                        pad = np.concatenate([np.full(k // 2, z[0]), z, np.full(k // 2, z[-1])])
                        z = np.convolve(pad, np.ones(k) / k, mode="valid")[: len(line)]
                    c.data["line"] = line
                    c.data["z"] = z
            self._auto = done

    def evaluate(self, x: Any, y: Any, ctx: Context | None = None, final: bool = True) -> Sample:
        """Every field at points (see :class:`Sample`); ``final=False`` leaves out the final
        features (the natural ground)."""
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        y = np.ascontiguousarray(y, dtype=np.float64).reshape(-1)
        aux: dict = {}
        shape = self.shape(x, y, aux)
        h = shape
        reg = self.regional(ctx)
        atten = None
        if reg is not None:
            h = h + reg.delta(x, y)
            atten = reg.attenuation(x, y)
        t_sea, moist = self.climate(x, y)
        temp = t_sea - self.d["climate"]["lapse_rate"] * np.maximum(h - self.sea, 0.0) / 1000.0
        weights = self.biome_weights(temp, moist)
        if self.final_features:
            weights = biome_overrides(self.final_features, x, y, weights, self.biome_names)
        t = self.d["terrain"]
        S = t["scale"]
        HS = t["height_scale"]
        # Biome terrain modifiers: dunes, swamp flattening, roughness of the detail.
        dunes = weights @ self.b_dunes
        dn = t["dunes"]
        if dn["height"] > 0.0 and np.any(dunes > 1e-3):
            ux, uy = nz.cos_sin_deg(dn["direction"])
            u = (x * ux + y * uy) / (dn["scale"] * S)
            u = u + 0.6 * nz.fbm2(
                self.seeds["dunes"], x, y, octaves=3, frequency=1.0 / (dn["scale"] * S * 3.0)
            )
            f = u - np.floor(u)
            ridge = np.where(f < 0.72, f / 0.72, (1.0 - f) / 0.28)
            ridge = ridge * ridge * (3.0 - 2.0 * ridge)
            land = aux["land"]
            h = h + dn["height"] * HS * dunes * ridge * land
        flat = weights @ self.b_flatten
        if np.any(flat > 1e-3):
            target = self.sea + 0.35
            near = 1.0 - nz.smoothstep(4.0, 30.0, np.abs(h - target))
            h = h + flat * near * (target - h) * 0.85
        rough = weights @ self.b_rough
        dt = t["detail"]
        detail = None
        if dt["height"] > 0.0:
            detail = (
                dt["height"]
                * HS
                * rough
                * nz.fbm2(
                    self.seeds["detail"],
                    x,
                    y,
                    octaves=dt["octaves"],
                    frequency=1.0 / (dt["scale"] * S),
                )
            )
            if atten is not None:
                detail = detail * atten
            h = h + detail
        water = np.full(x.shape, np.nan)
        kind = np.zeros(x.shape, dtype=np.int8)
        river = np.full(x.shape, np.inf)
        # Rivers and lakes from the regional stage, or along noise valleys (contour mode).
        if reg is not None:
            h, river, rlevel = reg.carve_rivers(x, y, h)
            has = ~np.isnan(rlevel)
            water[has] = rlevel[has]
            kind[has] = 3
            lake = reg.lake_level(x, y)
            on = ~np.isnan(lake) & (np.isnan(water) | (lake > water))
            water[on] = lake[on]
            kind[on] = 2
        elif self.d["hydrology"]["mode"] == "contour" and "valley" in aux:
            hy = self.d["hydrology"]
            cw = hy["rivers"]["contour_width"]
            vv = aux["valley"]
            lowland = 1.0 - nz.smoothstep(self.sea + 6.0, self.sea + 40.0, shape)
            inside = (vv < cw) & (lowland > 0.0)
            depth = hy["rivers"]["depth"] + 1.0
            u = np.clip(vv / cw, 0.0, 1.0)
            bed = self.sea - depth * (1.0 - u * u)
            bank = 1.0 - nz.smoothstep(cw, cw * 3.5, vv)
            target = np.where(inside, bed, self.sea + 0.6)
            pull = np.where(inside, 1.0, bank) * lowland
            lower = h > target
            h = np.where(lower, h + pull * (target - h), h)
            river = np.where(vv < cw * 3.5, vv / max(cw, 1e-9), np.inf)
            on = inside & (h < self.sea)
            water[on] = self.sea
            kind[on] = 3
        road = None
        if final and self.final_features:
            self._resolve()
            h, extras = apply_final(self.final_features, x, y, h, detail)
            wa = extras.get("water")
            if wa is not None:
                on = ~np.isnan(wa)
                water[on] = wa[on]
                kind[on] = 4
            road = extras.get("road")
        sea = (h < self.sea) & (np.isnan(water) | (water < self.sea))
        water[sea] = self.sea
        kind[sea] = 1
        out = Sample(
            height=h,
            shape=shape,
            temperature=temp,
            moisture=moist,
            weights=weights,
            water=water,
            water_kind=kind,
            river=river,
            land=aux["land"],
            road=np.zeros(x.shape) if road is None else road,
        )
        return out

    def height(self, x: Any, y: Any, ctx: Context | None = None) -> np.ndarray:
        """The authoritative final height at points."""
        return self.evaluate(x, y, ctx)["height"]

    def biome_index(self, weights: np.ndarray) -> np.ndarray:
        return np.argmax(weights, axis=1)


_SHAPE_TYPES = ("raise", "lower", "scale", "mountain", "ridge", "valley")


def _centre_of(shape: dict) -> tuple[float, float]:
    if "circle" in shape:
        return shape["circle"][0], shape["circle"][1]
    if "rect" in shape:
        x0, y0, x1, y1 = shape["rect"]
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    pts = np.asarray(shape.get("polygon") or shape.get("line"), dtype=np.float64)[:, :2]
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _densify(line: np.ndarray, step: float) -> np.ndarray:
    """A polyline with extra points so no segment is longer than ``step``."""
    out = [line[0]]
    for a, b in zip(line[:-1], line[1:], strict=True):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out, dtype=np.float64)
