"""Surface colours of terrain: what the ground looks like at each vertex.

The colour of a terrain vertex (linear RGB, stored as the mesh's vertex colour and
multiplied by the terrain material's detail texture) blends, in order:

1. **The biomes' ground** -- each biome's ground (grass, sand, snow, mud, gravel, ...)
   in its own colours, weighted by the biome weights; grass dries toward straw where the
   moisture is low, with two octaves of tone noise so meadows are not flat paint;
2. **soil** on moderate slopes and **rock** on steep ones (the ``surface.cliff_slope``
   angle, with a noisy band), in the biomes' rock colours, varied over a few hundred
   metres and darker in gullies (the regional drainage area) than on ridges;
3. **sand** on beaches (near the sea, on gentle slopes);
4. **snow** where the temperature at the vertex's own height is below
   ``surface.snow_temperature`` (a noisy margin; not on slopes too steep to hold it);
5. **wet banks** along rivers and lakes, and the **bed** colours under water;
6. **roads** where road features lie.

Only correctly rounded operations are used (see :mod:`.noise`), so colours are as
reproducible as heights.
"""

from __future__ import annotations

import numpy as np

from . import noise as nz

SNOW = np.array([0.82, 0.86, 0.9])
"""Linear albedo of snow (slightly blue)."""
WET = 0.62
"""Soil darkening factor where wet."""
ROAD_COLORS = {
    "gravel": np.array([0.32, 0.29, 0.25]),
    "dirt": np.array([0.24, 0.17, 0.11]),
    "cobble": np.array([0.26, 0.25, 0.24]),
    "asphalt": np.array([0.05, 0.05, 0.055]),
    "path": np.array([0.3, 0.24, 0.16]),
}


def _ground_table(model) -> np.ndarray:
    """Per biome, the linear colour of its ground kind."""
    rows = []
    for k, b in enumerate(model.biomes.values()):
        g = b["ground"]
        if g == "grass":
            rows.append(model.colors["grass"][k])
        elif g in ("sand", "red_sand"):
            rows.append(model.colors["sand"][k])
        elif g == "snow":
            rows.append(SNOW)
        elif g in ("dirt", "mud"):
            rows.append(model.colors["soil"][k])
        elif g == "ice":
            rows.append(np.array([0.55, 0.7, 0.82]))
        else:  # gravel, rock
            rows.append(0.5 * (model.colors["rock"][k] + model.colors["soil"][k]))
    return np.stack(rows)


def colors(model, sample: dict, x: np.ndarray, y: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """``(N, 3)`` linear ground colours at points (``sample`` from
    :meth:`WorldModel.evaluate`, ``normals`` unit, up = +z)."""
    w = sample["weights"]
    h = sample["height"]
    nz_ = np.clip(normals[:, 2], -1.0, 1.0)
    s = model.d["surface"]
    seed = model.seed
    # 1. Biome grounds, with dryness and tone noise.
    ground = nz.blend(w, _ground_table(model))
    grass_like = nz.blend(
        w, np.array([1.0 if b["ground"] == "grass" else 0.0 for b in model.biomes.values()])
    )
    dry = np.clip(1.0 - sample["moisture"] * 1.6, 0.0, 1.0) * grass_like
    straw = np.array([0.36, 0.3, 0.12])
    ground = ground + (dry * 0.45)[:, None] * (straw - ground)
    tone = 0.92 + 0.1 * nz.fbm2(
        nz.derive(seed, "surface", "tone"), x, y, octaves=2, frequency=1.0 / 23.0
    )
    tone += 0.06 * nz.gradient2(nz.derive(seed, "surface", "patch"), x / 7.0, y / 7.0)
    ground = ground * tone[:, None]
    soil = nz.blend(w, model.colors["soil"])
    rock = nz.blend(w, model.colors["rock"])
    sand = nz.blend(w, model.colors["sand"])
    # Rock varies over a few hundred metres (warmer and cooler outcrops); gullies, where
    # water collects (the regional drainage area, the same at every level of detail), are
    # darker and ridges lighter.
    broad = nz.fbm2(nz.derive(seed, "surface", "outcrop"), x, y, octaves=2, frequency=1.0 / 220.0)
    rock = rock * (0.9 + 0.22 * broad)[:, None] * np.array([1.0, 0.985, 0.96])
    reg = model.regional_if_ready()
    if reg is not None:
        wetness = np.clip((np.log10(np.maximum(reg.flow(x, y), 1.0)) - 3.6) / 2.4, 0.0, 1.0)
        rock = rock * (1.06 - 0.3 * wetness)[:, None]
        soil = soil * (1.04 - 0.2 * wetness)[:, None]
    # 2. Soil and rock by slope (cos of the slope angle = n_z), with a noisy band; above the
    # tree line (cold) bare rock starts 12 degrees earlier, in warm lowlands 8 later.
    temp = sample["temperature"]
    cold = nz.smoothstep(s["treeline_temperature"] + 4.0, s["treeline_temperature"] - 2.0, temp)
    lo_c, _ = nz.cos_sin_deg(min(s["cliff_slope"] + 8.0, 85.0))
    hi_c, _ = nz.cos_sin_deg(max(s["cliff_slope"] - 12.0, 5.0))
    cliff_c = lo_c + cold * (hi_c - lo_c)
    soil_lo, _ = nz.cos_sin_deg(min(s["cliff_slope"], 85.0))
    soil_hi, _ = nz.cos_sin_deg(max(s["cliff_slope"] - 18.0, 5.0))
    soil_c = soil_lo + cold * (soil_hi - soil_lo)
    jitter = 0.05 * nz.gradient2(nz.derive(seed, "surface", "slope"), x / 11.0, y / 11.0)
    # Soil shows through in patches just below the rock (not as bands along contours).
    patches = nz.smoothstep(
        -0.05,
        0.45,
        nz.fbm2(nz.derive(seed, "surface", "soil"), x, y, octaves=2, frequency=1.0 / 17.0),
    )
    soil_f = nz.smoothstep(soil_c + 0.04, soil_c - 0.08, nz_ + jitter) * patches
    rock_f = nz.smoothstep(cliff_c + 0.05, cliff_c - 0.07, nz_ + jitter)
    color = ground + soil_f[:, None] * (soil - ground)
    rock_tone = 0.85 + 0.25 * (
        0.5
        + 0.5
        * nz.fbm2(
            nz.derive(seed, "surface", "strata"), x, y + h * 3.0, octaves=2, frequency=1.0 / 9.0
        )
    )
    color = color + rock_f[:, None] * (rock * rock_tone[:, None] - color)
    # 3. Beaches.
    sea = model.sea
    beach = (1.0 - nz.smoothstep(s["beach_height"] * 0.6, s["beach_height"], h - sea)) * (
        nz.smoothstep(0.85, 0.95, nz_)
    )
    color = color + beach[:, None] * (sand - color)
    # 4. Snow by temperature (at the vertex's own height), held by slopes up to ~55 degrees.
    margin = 1.6 * nz.gradient2(nz.derive(seed, "surface", "snow"), x / 37.0, y / 37.0)
    snow = nz.smoothstep(s["snow_temperature"] + 1.0, s["snow_temperature"] - 0.8, temp + margin)
    snow *= nz.smoothstep(0.5, 0.7, nz_)
    color = color + snow[:, None] * (SNOW - color)
    # 5. Water: wet banks above, beds below.
    water = sample["water"]
    wet_level = np.where(np.isnan(water), -np.inf, water)
    depth = wet_level - h
    bank = nz.smoothstep(1.5, 0.0, h - wet_level) * (1.0 - snow)
    color = color * (1.0 - (1.0 - WET) * bank)[:, None]
    under = depth > 0.0
    if np.any(under):
        bed = np.where(sample["water_kind"][:, None] == 3, 0.6 * rock + 0.4 * sand, sand * 0.8)
        f = nz.smoothstep(0.0, 1.2, depth)[:, None]
        color = np.where(under[:, None], color + f * (bed * 0.7 - color), color)
    # 6. Roads.
    road = sample.get("road")
    if road is not None and np.any(road > 0.0):
        rc = ROAD_COLORS["gravel"] * (0.9 + 0.2 * tone[:, None] - 0.1)
        color = color + road[:, None] * (rc - color)
    return np.clip(color, 0.0, 1.0)


def water_colors(model, sample: dict, depth: np.ndarray) -> np.ndarray:
    """``(N, 3)`` linear colours of water surfaces by depth: clear shallows over the bed,
    darker deep water, tinted by the biomes' water colours."""
    w = sample["weights"]
    deep = nz.blend(w, model.colors["water"]) * 0.55
    shallow = deep * 0.4 + np.array([0.18, 0.25, 0.22])
    t = nz.smoothstep(0.0, 6.0, depth)
    return np.clip(shallow + t[:, None] * (deep - shallow), 0.0, 1.0)
