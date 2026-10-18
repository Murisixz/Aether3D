"""Showcase E (Stage 16): caves and overhangs -- volumetric terrain (a signed distance
field meshed with marching cubes near the camera) with cave entrances, caverns lit by
lanterns and cliffs leaning out over their feet, generated and rendered by Aether3D.

Run from the repository root:

    python examples/worlds_caves.py [OUTPUT_DIR] [--quick] [--spp N]

Writes to ``OUTPUT_DIR`` (default ``examples/output/worlds/e_caves``): ``entrance.png``,
``cavern.png``, ``overhang.png`` (1920x1080, GPU path tracer) with previews,
``caves_sheet.png``, ``caves.aether`` and ``report.json`` (where the cameras were placed,
how many volume cubes and triangles each view drew, timings).
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
from aether3d.worlds.volumes import CHUNK, volumes_of  # noqa: E402

SEED = 3


def entrance(world, vol, reach: int = 12) -> tuple[float, float, int]:
    """The opening (a ground point over cave air) of the cube near the origin whose ground
    opens most -- the lattice flags candidates conservatively, the field decides."""
    import numpy as np

    model = world.runtime.model()
    best = None
    for ci in range(-reach, reach):
        for cj in range(-reach, reach):
            if not any(flags & 1 for _, flags in vol.column(ci, cj)):
                continue
            gx, gy = np.meshgrid(
                np.arange(ci * CHUNK, (ci + 1) * CHUNK, 1.0),
                np.arange(cj * CHUNK, (cj + 1) * CHUNK, 1.0),
            )
            gx, gy = gx.ravel(), gy.ravel()
            s = model.evaluate(gx, gy)
            air = vol.cave_field(gx, gy, s["height"] - 0.5, s["height"], ~np.isnan(s["water"]))
            n = int((air < 0).sum())
            if n and (best is None or n > best[0]):
                k = int(np.argmin(air))
                best = (n, float(gx[k]), float(gy[k]))
    assert best is not None, "no cave opening found"
    return best[1], best[2], best[0]


def open_space(
    world, vol, x0: float, y0: float, radius: float = 120.0
) -> tuple[np.ndarray, np.ndarray]:
    """The most open point of cave air below ``(x0, y0)`` and its longest free direction."""
    rng = np.random.default_rng(5)
    model = world.runtime.model()
    best, where = -1e9, None
    for _ in range(6):
        p = np.column_stack(
            [
                x0 + rng.uniform(-radius, radius, 400),
                y0 + rng.uniform(-radius, radius, 400),
                np.zeros(400),
            ]
        )
        s = model.evaluate(p[:, 0], p[:, 1])
        p[:, 2] = s["height"] - rng.uniform(8.0, 50.0, 400)
        a = vol.cave_field(p[:, 0], p[:, 1], p[:, 2], s["height"], ~np.isnan(s["water"]))
        k = int(np.argmin(a))
        if -a[k] > best:
            best, where = float(-a[k]), p[k]
    dirs = [
        np.array([math.cos(t), math.sin(t), -0.12])
        for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)
    ]
    free = []
    for d in dirs:
        steps = np.arange(1.0, 60.0, 1.0)
        pts = where[None] + steps[:, None] * d[None]
        solid = vol.solid(pts)
        free.append(float(steps[np.argmax(solid)] if solid.any() else 60.0))
    return where, dirs[int(np.argmax(free))]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("output", nargs="?", default=str(OUT / "e_caves"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                        help="cpu: the CPU rasterizer for every image (tests)")  # fmt: skip
    parser.add_argument("--spp", type=int, default=384)
    args = parser.parse_args(argv)
    set_backend(args.backend)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    w, h = frame_size(args.quick)
    spp = 8 if args.quick else args.spp
    t0 = time.perf_counter()
    scene = Scene("Caves")
    overrides: dict = {"vegetation": {"density": 0.4}}
    if args.backend == "cpu":
        overrides["streaming"] = {"volume_distance": 150.0}  # (the test mode: fewer cubes)
    world = scene.worlds.create("Highlands", "cavernous_highlands", seed=SEED, overrides=overrides)
    world.runtime.regional()
    vol = volumes_of(world.runtime)
    light(scene, "morning", fog=0.0002, fog_height=world.height(0.0, 0.0), falloff=0.002)
    report: dict = {"machine": machine(), "backend": args.backend, "size": [w, h]}
    stills = {}
    # 1. A cave entrance from outside: aim at the widest opening.
    ex, ey, opening = entrance(world, vol)
    world.add_feature(
        {
            "type": "clear",
            "id": "entrance",
            "shape": {"circle": [ex, ey, 30.0]},
            "what": ["trees"],
            "falloff": 12.0,
        }
    )
    g = world.height(ex, ey)
    cam = scene.add_camera(
        "Entrance",
        position=(ex - 17.0, ey - 21.0, g + 11.0),
        look_at=(ex, ey, g - 5.0),
        fov=60,
        far=8000,
    )
    stills["entrance"] = render_still(
        scene,
        out / "entrance.png",
        camera=cam,
        width=w,
        height=h,
        path_tracing=spp,
        preset="morning",
    )
    stills["entrance"]["at"] = [ex, ey]
    stills["entrance"]["world"] = dict(world.runtime.last_frame)
    print("entrance", stills["entrance"]["seconds"], flush=True)
    # 2. Inside a cavern -- an authored hall (a cave feature, 60 m across) -- lit by
    # lanterns on its floor; the sun does not reach it.
    hx, hy, r = ex - 170.0, ey + 110.0, 30.0
    world.add_feature(
        {
            "type": "cave",
            "id": "hall",
            "kind": "cavern",
            "radius": r,
            "noise": 0.35,
            "shape": {"circle": [hx, hy, r]},
        }
    )
    hz = world.height(hx, hy) - 1.1 * r  # the hall's centre (see docs/WORLDS.md)

    def floor(x: float, y: float) -> float:
        """The hall's floor below (x, y): the first rock under its centre height."""
        zs = np.arange(hz, hz - 1.2 * r, -0.5)
        solid = vol.solid(np.column_stack([np.full(zs.size, x), np.full(zs.size, y), zs]))
        return float(zs[np.argmax(solid)]) if solid.any() else hz - 0.6 * r

    eye = np.array([hx - 0.55 * r, hy - 0.2 * r, 0.0])
    eye[2] = floor(eye[0], eye[1]) + 2.5
    for k, (u, v) in enumerate(((-0.15, 0.1), (0.25, -0.2), (0.5, 0.25), (0.05, 0.45))):
        q = np.array([hx + u * r, hy + v * r, 0.0])
        q[2] = floor(q[0], q[1]) + 1.2
        lamp = scene.add_point_light(f"Lantern {k}", position=tuple(q), intensity=260.0, range=70.0)
        lamp.light.configure(color=(1.0, 0.84, 0.62), radius=0.25)
    cam = scene.add_camera(
        "Cavern",
        position=tuple(eye),
        look_at=(hx + 0.5 * r, hy + 0.15 * r, hz - 6.0),
        fov=74,
        far=4000,
    )
    stills["cavern"] = render_still(
        scene, out / "cavern.png", camera=cam, width=w, height=h, path_tracing=spp, preset="morning"
    )
    stills["cavern"]["at"] = [round(float(v), 2) for v in eye]
    stills["cavern"]["world"] = dict(world.runtime.last_frame)
    print("cavern", stills["cavern"]["seconds"], flush=True)
    # 3. Overhangs: a mesa (a raised pad with near-vertical sides) whose cliffs lean out
    # and are undercut, seen along its south face so the profile stands against the sky.
    mx, my = ex + 160.0, ey - 120.0
    top = world.height(mx, my) + 28.0
    world.add_feature(
        {
            "type": "pad",
            "id": "mesa",
            "shape": {"rect": [mx - 40.0, my - 28.0, mx + 40.0, my + 28.0]},
            "height": round(top, 2),
            "falloff": 5.0,
        }
    )
    world.add_feature(
        {
            "type": "clear",
            "id": "mesa-foot",
            "shape": {"circle": [mx, my, 90.0]},
            "what": ["trees"],
            "falloff": 20.0,
        }
    )
    face = my - 28.0
    corner = (mx - 40.0 - 30.0, face - 10.0)  # (the morning sun is behind the camera)
    gz = world.height(*corner)
    cam = scene.add_camera(
        "Overhang",
        position=(corner[0], corner[1], gz + 2.5),
        look_at=(mx + 25.0, face - 1.0, gz + 13.0),
        fov=58,
        far=8000,
    )
    stills["overhang"] = render_still(
        scene,
        out / "overhang.png",
        camera=cam,
        width=w,
        height=h,
        path_tracing=spp,
        preset="morning",
    )
    stills["overhang"]["at"] = [mx, my]
    stills["overhang"]["world"] = dict(world.runtime.last_frame)
    print("overhang", stills["overhang"]["seconds"], flush=True)
    for name in stills:
        preview(out / f"{name}.png", out / f"{name}_preview.png")
    sheet(
        [out / f"{n}.png" for n in stills], out / "caves_sheet.png", labels=list(stills), columns=3
    )
    report["stills"] = stills
    report["volumes"] = vol.stats()
    report["seconds"] = round(time.perf_counter() - t0, 1)
    scene.save(out / "caves.aether")
    write_report(out / "report.json", report)
    print(f"{out}: {report['seconds']:.0f} s, done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
