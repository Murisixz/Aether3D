"""Multi-channel PBR painting, rendered alike by the four renderers of Aether3D.

Run from the repository root:

    python examples/texturing_pbr.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

A metal orb is painted channel by channel, with *material strokes* -- one stroke laying
base color, roughness, metallic and emission at once, on four painted textures:

* brushed steel: the textures' backgrounds (base color #b8bcc2, roughness 0.32,
  metallic 1);
* rust patches: soft strokes of dull orange-brown, roughness 0.85, metallic 0 (rust is
  not a metal), at seeded random points;
* a gold band around the equator: one stroke around it (gold, roughness 0.2,
  metallic 1);
* glowing runes: thin strokes on the emission texture only (and a smoother roughness
  under them).

The textures' encodings are honoured everywhere: base color and emission are display
sRGB (decoded to linear before shading), roughness and metallic are linear data.

Then the orb is rendered by the CPU and GPU rasterizers and the CPU and GPU path
tracers (those that run here), and their images are compared: the rasterizers must agree
closely (the same shading code on both: compared after a 3x3 box filter, since their
antialiasing samples differ), and so must the path tracers (the same
estimator and random numbers). A rasterizer and a path tracer differ by design (no
reflections in a rasterizer).

Writes ``texturing_pbr.png`` (top: base color, roughness, metallic and emission as
painted; bottom: the renders) and ``texturing_pbr.json`` (differences, timings).
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import RenderError, Scene, __version__, create_sphere
from aether3d.rendering.png import encode_png
from aether3d.texturing import stroke

HERE = Path(__file__).resolve().parent


def build(res: int):
    scene = Scene("PBR orb")
    scene.world.configure(
        background="#9ec4e8",
        ambient_color="#cfe0ff",
        ambient_intensity=0.35,
        environment={"type": "sky", "zenith_color": "#2f63b8", "horizon_color": "#d6e2ee",
                     "ground_color": "#5a5248", "intensity": 0.9},
    )  # fmt: skip
    floor = scene.add_plane("Floor", size=20)
    tiles = scene.create_material("Tiles", roughness=0.6)
    tiles.set_procedural_texture("base_color", "checker", color1="#3a3a3a", color2="#bdbdbd",
                                 coordinates="world", projection="xy", scale=1.0)  # fmt: skip
    floor.material = tiles
    orb = scene.add_object("Orb", mesh=create_sphere(1.0, 128, 64), position=(0, 0, 1.0))
    scene.add_directional_light("Sun", direction=(0.55, 0.65, -0.7), color="#fff1dc",
                                intensity=2.6, angle=2.0)  # fmt: skip
    cam = scene.add_camera("Cam", position=(2.8, -3.6, 2.2), look_at=(0, 0, 0.9))
    cam.frame(["Orb"], margin=0.25, aspect=1.0)
    report = orb.unwrap(resolution=res, padding=8)

    color = scene.add_painted_texture("Orb color", res, res, background="#b8bcc2")
    rough = scene.add_painted_texture("Orb roughness", res, res, encoding="data",
                                      background=[0.32, 0.32, 0.32, 1])  # fmt: skip
    metal = scene.add_painted_texture("Orb metallic", res, res, encoding="data",
                                      background=[1.0, 1.0, 1.0, 1])  # fmt: skip
    glow = scene.add_painted_texture("Orb emission", res, res, background="#000000")
    for doc in (color, rough, metal, glow):
        doc.add_layer("Paint")
    m = scene.create_material("Orb", roughness=1.0, metallic=1.0, emissive="#ffffff",
                              emissive_strength=3.0)  # fmt: skip
    m.set_texture("base_color", color)
    m.set_texture("roughness", rough, channel="r")
    m.set_texture("metallic", metal, channel="r")
    m.set_texture("emissive", glow)
    m.base_color = (1.0, 1.0, 1.0)
    orb.material = m

    # Rust: material strokes at seeded random points of the upper half.
    rng = np.random.default_rng(21)
    rust = [{"texture": color, "color": "#7a3d1c"}, {"texture": rough, "color": [0.85]},
            {"texture": metal, "color": [0.0]}]  # fmt: skip
    for _ in range(14):
        d = rng.normal(size=3)
        d[2] = abs(d[2]) + 0.3
        d /= np.linalg.norm(d)
        e = d + rng.normal(scale=0.25, size=3)
        e /= np.linalg.norm(e)
        stroke(scene, orb, rust, {"radius": 0.12 + 0.12 * rng.random(), "hardness": 0.1,
                                  "strength": 0.7}, [d + [0, 0, 1.0], e + [0, 0, 1.0]])  # fmt: skip
    # The gold band: one material stroke around the equator (soft-edged, unlike a fill of
    # whole triangles).
    ring = [[np.cos(a), np.sin(a), 1.0] for a in np.linspace(0, 2 * np.pi, 97)]
    stroke(scene, orb, [{"texture": color, "color": "#d4a640"}, {"texture": rough,
           "color": [0.2]}, {"texture": metal, "color": [1.0]}],
           {"radius": 0.08, "hardness": 0.85, "spacing": 0.1}, ring)  # fmt: skip
    # Runes: glowing zigzags above the band, on the emission (and roughness) only.
    for k in range(6):
        a = 2 * np.pi * k / 6
        pts = []
        for j in range(5):
            phi = a + 0.08 * j
            z = 0.18 + (0.12 if j % 2 else 0.0)
            r = np.sqrt(1 - z * z)
            pts.append([r * np.cos(phi), r * np.sin(phi), z + 1.0])
        stroke(scene, orb, [{"texture": glow, "color": "#3fd8ff"},
                            {"texture": rough, "color": [0.1]}],
               {"radius": 0.018, "hardness": 0.8}, pts)  # fmt: skip
    return scene, cam, (color, rough, metal, glow), report


def try_render(scene, cam, size, backend, **extra):
    try:
        t0 = time.perf_counter()
        r = scene.render(width=size, height=size, camera=cam, backend=backend, **extra)
        return r, time.perf_counter() - t0
    except RenderError as exc:
        if exc.code in ("GPU_UNAVAILABLE", "BACKEND_UNSUPPORTED", "GPU_ADAPTER_NOT_FOUND"):
            return None, exc.message
        raise


def rms(a, b, box: int = 0) -> float:
    """RMS difference in 0..1; ``box`` > 0 first averages both over (2 box + 1)^2 pixels,
    so that antialiasing patterns (2 jittered samples on the CPU, 4x MSAA on the GPU)
    do not count, only shading."""
    a = a.astype(np.float64) / 255
    b = b.astype(np.float64) / 255
    if box:
        k = 2 * box + 1

        def blur(x):
            p = np.pad(x, ((box, box), (box, box), (0, 0)), mode="edge")
            c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
            return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)

        a, b = blur(a), blur(b)
    d = a - b
    return float(np.sqrt((d * d).mean()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="smaller textures and images")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    size = 240 if args.quick else 480
    res = 512 if args.quick else 2048
    spp = 16 if args.quick else 256
    failures: list[str] = []
    t0 = time.perf_counter()
    scene, cam, docs, unwrap = build(res)
    build_seconds = time.perf_counter() - t0

    path = {"samples": spp, "bounces": 4, "seed": 7}
    renders, times = {}, {}
    plan = [("cpu", {}), ("path", {"path": path})]
    if args.backend != "cpu":
        plan[1:1] = [("gpu", {})]
        plan.append(("gpu_path", {"path": path}))
    for name, extra in plan:
        r, t = try_render(scene, cam, size, name, **extra)
        if r is None:
            times[name] = f"not available: {t}"
            continue
        renders[name], times[name] = r, t
    diffs = {}
    if "gpu" in renders:
        diffs["cpu_vs_gpu_raster"] = rms(renders["cpu"].image, renders["gpu"].image)
        diffs["cpu_vs_gpu_raster_3x3"] = rms(renders["cpu"].image, renders["gpu"].image, 1)
        if diffs["cpu_vs_gpu_raster_3x3"] > 0.01:
            failures.append(f"rasterizers disagree: RMS {diffs['cpu_vs_gpu_raster_3x3']:.4f}")
    if "gpu_path" in renders:
        diffs["cpu_vs_gpu_path"] = rms(renders["path"].image, renders["gpu_path"].image)
        if diffs["cpu_vs_gpu_path"] > 0.03:
            failures.append(f"path tracers disagree: RMS {diffs['cpu_vs_gpu_path']:.4f}")
    diffs["raster_vs_path"] = rms(renders["cpu"].image, renders["path"].image)

    def panel(doc):
        img = doc.read()[..., :3]
        rows = np.minimum((np.arange(size) + 0.5) * img.shape[0] / size, img.shape[0] - 1)
        cols = np.minimum((np.arange(size) + 0.5) * img.shape[1] / size, img.shape[1] - 1)
        return np.round(img[rows.astype(int)][:, cols.astype(int)] * 255).astype(np.uint8)

    top = np.concatenate([panel(d) for d in docs], axis=1)
    shots = [renders[k].image for k in ("cpu", "gpu", "path", "gpu_path") if k in renders]
    while len(shots) < 4:
        shots.append(np.full((size, size, 3), 32, np.uint8))
    bottom = np.concatenate(shots, axis=1)
    png = encode_png(np.concatenate([top, bottom], axis=0),
                     text={"Software": f"Aether3D {__version__}"})  # fmt: skip
    (out / "texturing_pbr.png").write_bytes(png)
    report = {
        "aether3d": __version__,
        "quick": args.quick,
        "unwrap": {k: v for k, v in unwrap.items() if k != "diagnostics"},
        "textures": {d.name: d.info() for d in docs},
        "renders": {k: renders[k].backend for k in renders},
        "seconds": {"build_and_paint": build_seconds, **times},
        "rms": diffs,
        "path": path,
        "failures": failures,
    }
    (out / "texturing_pbr.json").write_text(json.dumps(report, indent=2, default=str))
    print(
        f"renders: {', '.join(renders)}; RMS {', '.join(f'{k} {v:.4f}' for k, v in diffs.items())}"
    )
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
