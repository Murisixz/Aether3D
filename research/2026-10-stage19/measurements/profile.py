"""What Stage 19's new NumPy code costs at the sizes it is used at (Part eight: languages).

    .venv/bin/python research/2026-10-stage19/measurements/profile.py OUT.json

Each entry is the best of three runs (one process, nothing else heavy running), next to
what it serves (a frame's render, a curve's evaluation), so the choice of language rests
on measured shares rather than guesses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import aether3d
from aether3d import Scene
from aether3d.animation.curves import FCurve
from aether3d.rendering.denoise import denoise
from aether3d.rendering.envimage import EnvironmentImage, builtin
from aether3d.rendering.exr import write_exr
from aether3d.rendering.tonemap import display


def best(fn, repeat: int = 3) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def curves() -> dict:
    rng = np.random.default_rng(19)
    out = {}
    for n in (100, 10_000):
        t = np.cumsum(rng.uniform(0.2, 1.5, n))
        v = rng.normal(size=n).cumsum()
        grid = np.linspace(t[0], t[-1], 100_000)
        for mode in ("local", "blender", "continuous"):
            build = best(lambda m=mode: FCurve.from_arrays(t, v, handles="auto_clamped",
                                                           smoothing=m))  # fmt: skip
            c = FCurve.from_arrays(t, v, handles="auto_clamped", smoothing=mode)
            ev = best(lambda c=c: c.evaluate(grid, order=2))
            out[f"{mode}_{n}_keys"] = {"build_s": build, "evaluate_100k_order2_s": ev}
    return out


def motion() -> dict:
    scene = Scene("Motion")
    rng = np.random.default_rng(3)
    anim = scene.animation
    anim.set_timeline(fps=60, start=0.0, end=10.0)
    names = []
    with scene.transaction("Objects"):
        for i in range(100):
            obj = scene.add_cube(f"C{i}", size=0.2, position=(i * 0.5, 0.0, 0.0))
            names.append(obj.name)
            times = np.linspace(0.0, 10.0, 11)
            for axis in ("x", "y", "z"):
                keys = [(float(tt), float(vv)) for tt, vv in zip(times, rng.normal(size=11))]
                anim.set_channel(obj, f"transform.position.{axis}", keys)
    seconds = best(lambda: anim.motion_report(names, rate=240), repeat=2)
    return {"objects": 100, "duration_s": 10.0, "rate": 240, "motion_report_s": seconds}


def lighting() -> dict:
    from aether3d.feedback import FeedbackSession

    scene = Scene("Lighting")
    scene.add_plane("Floor", size=10)
    ball = scene.add_sphere("Ball", radius=0.5, position=(0, 0, 0.5), segments=64, rings=32)
    scene.add_camera("Cam", position=(0, -4, 1.2), look_at=(0, 0, 0.5))
    scene.light_rig(ball, "three_point")
    expect = {"format": "aether3d.expectations", "version": 1, "defaults": False,
              "checks": [{"id": "l", "check": "lighting", "objects": "Ball",
                          "ratio": {"min": 2, "max": 8}}]}  # fmt: skip
    from aether3d.feedback.lighting import measure

    session = FeedbackSession(scene, expect, width=640, height=360)
    obs = session.observe()

    def lit():
        obs._lighting.clear()  # (measurements are cached on the observation)
        measure(obs, [ball])

    seconds = best(lambda: FeedbackSession(scene, expect, width=640, height=360).diagnose(),
                   repeat=2)  # fmt: skip
    render = best(lambda: scene.render(width=640, height=360, backend="cpu"), repeat=2)
    return {"size": [640, 360], "measure_s": best(lit), "diagnose_s": seconds,
            "cpu_render_s": render}  # fmt: skip


def frame_outputs() -> dict:
    h, w = 1080, 1920
    rng = np.random.default_rng(5)
    beauty = rng.random((h, w, 3), dtype=np.float32)
    passes = {
        "beauty": beauty,
        "albedo": np.full((h, w, 3), 0.6, np.float32),
        "normal": np.tile(np.array([0, 0, 1], np.float32), (h, w, 1)),
        "depth": np.full((h, w), 3.0, np.float32),
        "variance": (0.01 * rng.random((h, w, 3))).astype(np.float32),
    }
    out = {"size": [w, h]}
    out["denoise_s"] = best(lambda: denoise(passes), repeat=2)
    out["display_film_8bit_s"] = best(lambda: display(beauty, "film"))
    out["display_film_16bit_s"] = best(lambda: display(beauty, "film", bits=16))
    tmp = Path(sys.argv[1]).with_suffix(".exr")
    channels = {"R": beauty[..., 0], "G": beauty[..., 1], "B": beauty[..., 2],
                "Z": passes["depth"], **{f"albedo.{c}": passes["albedo"][..., i]
                                         for i, c in enumerate("RGB")}}  # fmt: skip
    out["exr_write_7_channels_s"] = best(lambda: write_exr(tmp, channels))
    out["exr_bytes"] = tmp.stat().st_size
    tmp.unlink()
    return out


def environment() -> dict:
    draw = best(lambda: builtin.__wrapped__("studio", 2048))  # (builtin() is cached)
    pixels = builtin("studio", 2048)
    u = np.random.default_rng(1).random((1000, 2))

    def tables():  # (a fresh image builds its sampling tables on first use)
        EnvironmentImage.build(pixels, 0.0, 1.0, "builtin:studio").sample(u[:, 0], u[:, 1])

    def raster():
        EnvironmentImage.build(pixels, 0.0, 1.0, "builtin:studio").raster_lighting()

    return {"size": list(pixels.shape[1::-1]), "draw_builtin_s": draw,
            "sampling_tables_s": best(tables), "raster_lighting_s": best(raster)}  # fmt: skip


def motion_vectors() -> dict:
    scene = Scene("Vectors")
    scene.add_plane("Floor", size=20)
    anim = scene.animation
    with scene.transaction("Objects"):
        for i in range(20):
            obj = scene.add_sphere(f"S{i}", radius=0.3, position=(i - 10.0, 2.0, 0.3))
            anim.set_channel(obj, "transform.position.y", [(0.0, 2.0), (1.0, -2.0)])
    scene.add_camera("Cam", position=(0, -12, 4), look_at=(0, 0, 0))
    plain = best(lambda: scene.render(width=1920, height=1080, backend="cpu", time=0.5),
                 repeat=2)  # fmt: skip
    vec = best(lambda: scene.render(width=1920, height=1080, backend="cpu", time=0.5,
                                    motion=1 / 60), repeat=2)  # fmt: skip
    return {"size": [1920, 1080], "cpu_render_s": plain, "with_motion_s": vec,
            "motion_pass_s": vec - plain}  # fmt: skip


def main() -> None:
    out = {"aether3d": aether3d.__version__}
    for name, fn in (("curves", curves), ("motion_report", motion), ("lighting", lighting),
                     ("frame_outputs", frame_outputs), ("environment", environment),
                     ("motion_vectors", motion_vectors)):  # fmt: skip
        t0 = time.perf_counter()
        out[name] = fn()
        print(name, round(time.perf_counter() - t0, 1), "s", json.dumps(out[name])[:300],
              flush=True)  # fmt: skip
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
