"""Showcase D (Stage 16): a continent spanning climates -- ice in the north, taiga,
temperate forests, plains, savanna, badlands, desert, tropical forest and swamp in the south
-- with smooth transitions, generated and rendered by Aether3D.

Run from the repository root:

    python examples/worlds_biomes.py [OUTPUT_DIR] [--quick]

Writes to ``OUTPUT_DIR`` (default ``examples/output/worlds/d_biomes``): ``continent.png``
(the whole 32 km continent from above), one still per biome ``biome_<name>.png`` (where
that biome dominates), ``transition.png`` (an oblique view across climate zones), their
previews, ``biomes_sheet.png``, ``continent.aether`` and ``report.json`` (where each still
was taken, its biome weights and timings). Stills are rasterized on the GPU (1920x1080,
cascaded shadows).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from worlds_common import (  # noqa: E402
    OUT,
    frame_size,
    light,
    machine,
    preview,
    render_still,
    set_backend,
    sheet,
    write_report,
)

from aether3d import Scene  # noqa: E402

SEED = 3
BIOMES = (
    "snowy_taiga",
    "taiga",
    "temperate_forest",
    "plains",
    "savanna",
    "badlands",
    "desert",
    "tropical_forest",
    "swamp",
)


def find(world, biome: str) -> tuple[float, float] | None:
    """A dry, gentle point where ``biome`` weighs most (on a 64 x 64 grid)."""
    e = world.extent
    xs = np.linspace(e[0] + 2500, e[2] - 2500, 64)
    X, Y = np.meshgrid(xs, xs)
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    s = world.sample(pts, fields=["biome_weights", "water", "slope"])
    best, where = -1.0, None
    for (x, y), weights, water, slope in zip(
        pts, s["biome_weights"], s["water"], s["slope"], strict=True
    ):
        if water is not None or slope > 20:
            continue
        w = dict(weights).get(biome, 0.0)
        if w > best:
            best, where = w, (float(x), float(y))
    return where if best > 0.45 else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(OUT / "d_biomes"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                        help="cpu: the CPU rasterizer for every image (tests)")  # fmt: skip
    args = parser.parse_args(argv)
    set_backend(args.backend)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = frame_size(args.quick)
    t0 = time.perf_counter()
    scene = Scene("Biomes")
    world = scene.worlds.create("Continent", "biome_continent", seed=SEED)
    world.runtime.regional()
    report: dict = {
        "machine": machine(),
        "backend": args.backend,
        "size": [w, h],
        "regional": world.runtime.regional().summary(),
    }
    light(scene, "midday", fog=0.00004, fog_height=0.0, falloff=0.0004)
    e = world.extent
    span = e[2] - e[0]
    top = scene.add_camera(
        "Top",
        position=(0.0, -1.0, 52000.0),
        look_at=(0.0, 0.0, 0.0),
        fov=2 * math.degrees(math.atan(span / 2 / 52000.0)) * 1.12,
        far=90000,
    )
    world.configure(streaming={"view_distance": 90000.0}, label="map view")
    stills = {
        "continent": render_still(
            scene,
            out / "continent.png",
            camera=top,
            width=h if args.quick else 1080,
            height=h if args.quick else 1080,
            preset="midday",
            distance=60000.0,
        )
    }
    world.configure(streaming={"view_distance": 14000.0}, label="ground views")
    places = {}
    for biome in BIOMES:
        at = find(world, biome)
        if at is None:
            continue
        x, y = at
        g = world.height(x, y)
        # Look across the biome from 60 m up, toward the higher ground nearby.
        best = max(
            range(8),
            key=lambda k: world.height(
                x + 600 * math.cos(k * math.pi / 4), y + 600 * math.sin(k * math.pi / 4)
            ),
        )
        a = best * math.pi / 4
        eye = (x - 220 * math.cos(a), y - 220 * math.sin(a), g + 60.0)
        target = (x + 500 * math.cos(a), y + 500 * math.sin(a), g + 10.0)
        weights = world.sample([[x, y]], fields=["biome_weights", "temperature", "moisture"])
        places[biome] = {
            "at": [round(x, 1), round(y, 1)],
            "weights": weights["biome_weights"][0],
            "temperature": round(weights["temperature"][0], 2),
            "moisture": round(weights["moisture"][0], 3),
        }
        if args.backend == "cpu" and len(places) > 2:
            continue  # (the test mode renders the first two)
        cam = scene.add_camera(f"Biome {biome}", position=eye, look_at=target, fov=58, far=40000)
        stills[f"biome_{biome}"] = render_still(
            scene,
            out / f"biome_{biome}.png",
            camera=cam,
            width=w,
            height=h,
            preset="midday",
            distance=5000.0,
        )
        print(
            f"{biome}: {stills[f'biome_{biome}']['seconds']} s at {places[biome]['at']}", flush=True
        )
    # A transition: from high above the middle of the continent, looking south.
    g = world.height(0.0, 0.0)
    cam = scene.add_camera(
        "Transition",
        position=(0.0, 4000.0, g + 2600.0),
        look_at=(0.0, -9000.0, 0.0),
        fov=60,
        far=90000,
    )
    world.configure(streaming={"view_distance": 40000.0}, label="transition view")
    stills["transition"] = render_still(
        scene,
        out / "transition.png",
        camera=cam,
        width=w,
        height=h,
        preset="midday",
        distance=30000.0,
    )
    for name in stills:
        preview(out / f"{name}.png", out / f"{name}_preview.png")
    names = [n for n in stills if n.startswith("biome_")]
    sheet(
        [out / f"{n}.png" for n in names],
        out / "biomes_sheet.png",
        labels=[n[6:] for n in names],
        columns=3,
    )
    report["stills"] = stills
    report["places"] = places
    report["seconds"] = round(time.perf_counter() - t0, 1)
    scene.save(out / "continent.aether")
    write_report(out / "report.json", report)
    print(f"{out}: {report['seconds']:.0f} s, done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
