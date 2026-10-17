"""Hard-surface texturing: UVs, texel density and a painted, baked material, by Aether3D.

Run from the repository root:

    python examples/texturing_hard_surface.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

The power cell of ``examples/hard_surface.py`` (built by the same code) is textured end to
end, every step measured:

1. **UVs.** The body is unwrapped: charts are cut at its hard edges (auto-smooth creases)
   and face-group borders, flattened without flips and packed with 8 texels of padding.
   The knob and the bezel are unwrapped at the body's texel density (``texels_per_unit``)
   so a checker -- or a scratch -- has the same size on every part. ``uv_report`` checks
   that no triangle is flipped or overlaps another and measures the density spread.
2. **Bakes.** Ambient occlusion and curvature of the body, baked from itself.
3. **Painting.** Painted textures with layers, not one flat material per face group: the
   base color is filled per face group (paint, recess, glowing panel, duct), crevices are
   darkened by an AO-masked multiply layer, convex edges are worn to bare metal by a
   curvature-masked layer laid on base color, metallic and roughness alike, rivets and a
   hazard stripe are painted with surface strokes and a gradient, and the panel glows
   through a painted emission texture.
4. **Render.** The textured cell (the GPU path tracer when available).

Writes ``texturing_hard_surface.png`` (top: UV islands, texel density, checker through the
UVs; bottom: the baked AO and curvature over the painted roughness and metallic, the
painted base color over its UV layout, the final render) and
``texturing_hard_surface.json`` (UV reports, bake reports, timings).
Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import Scene, __version__, uv
from aether3d.rendering.png import encode_png
from aether3d.texturing import bake, fill, gradient, stroke

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hard_surface import Steps, build_body, build_knob  # noqa: E402

RES = 1024


def render(scene, camera, width, height, backend, samples):
    """The final render: the GPU path tracer when asked for or available, else the
    rasterizer of the backend."""
    from aether3d import RenderError

    options = {"width": width, "height": height, "camera": camera}
    if backend != "cpu":
        try:
            return scene.render(**options, backend="gpu_path", path={"samples": samples})
        except RenderError as exc:
            if backend == "gpu" and exc.code not in ("GPU_UNAVAILABLE", "BACKEND_UNSUPPORTED"):
                raise
    return scene.render(**options, backend="cpu")


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fit(image: np.ndarray, size: int) -> np.ndarray:
    """``image`` scaled (nearest) into a ``size`` square."""
    h, w = image.shape[:2]
    rows = np.minimum((np.arange(size) + 0.5) * h / size, h - 1).astype(int)
    cols = np.minimum((np.arange(size) + 0.5) * w / size, w - 1).astype(int)
    return image[rows][:, cols]


def to8(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="smaller textures and images")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    res = 256 if args.quick else RES
    panel = 300 if args.quick else 600
    samples = 32 if args.quick else 512
    failures: list[str] = []
    timings: dict[str, float] = {}
    report: dict = {"aether3d": __version__, "failures": failures, "timings": timings}

    scene = Scene("Power cell, textured")
    steps = Steps()
    body, _ = build_body(scene, steps)
    knob, bezel = build_knob(scene, steps)

    # 1. UVs.
    t0 = time.perf_counter()
    body_uv = body.unwrap(resolution=res, padding=max(2, res // 128), split_by=["hard", "groups"])
    timings["unwrap_body"] = time.perf_counter() - t0
    body_report = body.uv_report(resolution=res)
    density = body_report["texel_density"]["mean"]
    parts = {"body": body_report}
    for obj in (knob, bezel):
        t0 = time.perf_counter()
        obj.unwrap(resolution=res, padding=max(2, res // 128), texels_per_unit=density)
        timings[f"unwrap_{obj.name.lower()}"] = time.perf_counter() - t0
        parts[obj.name.lower()] = obj.uv_report(resolution=res)
    report["unwrap"] = {k: v for k, v in body_uv.items() if k != "diagnostics"}
    report["uv"] = {
        name: {
            "islands": r["islands"],
            "flipped_triangles": r["flipped_triangles"],
            "overlapping_triangles": r["overlapping_triangles"],
            "stretch_l2_max": r["stretch_l2"]["max"],
            "texel_density": r["texel_density"],
            "utilization": r["utilization"],
            "min_gap_texels": r["min_gap_texels"],
        }
        for name, r in parts.items()
    }
    for name, r in parts.items():
        if r["flipped_triangles"] or r["overlapping_triangles"]:
            failures.append(f"{name}: flipped or overlapping UVs")
    for name in ("knob", "bezel"):
        d = parts[name]["texel_density"]["mean"]
        if abs(d / density - 1) > 0.02:
            failures.append(f"{name}: texel density {d:.1f}, body {density:.1f}")

    # Lights, floor, camera.
    floor = scene.add_plane("Floor", size=12, position=(0, 0, -0.4))
    scene.assign_material(floor, scene.create_material("Floor", base_color="#8d939a"))
    scene.add_directional_light("Key", direction=(-0.5, 0.7, -1.0), intensity=2.4)
    scene.add_directional_light("Fill", direction=(0.8, 0.3, -0.4), intensity=0.6)
    scene.add_directional_light("Back light", direction=(0.4, -0.9, -0.5), intensity=1.2)
    cam = scene.add_camera("Front")
    cam.frame(["Power cell", "Knob"], direction=(-3.2, 3.4, -2.2), margin=0.06, aspect=1.0)

    # A checker through every part's UVs: squares stay square (low distortion) and keep
    # their size from part to part (one texel density).
    checker = scene.create_material("Checker", roughness=0.6)
    square = 0.08  # meters: the checker's squares, the same size on every part
    checker.set_procedural_texture(
        "base_color", "checker", scale=res / (density * square), color1="#2b3440",
        color2="#e9edf2",
    )  # fmt: skip
    for obj in (body, knob, bezel):
        obj.material = checker
    checker_view = scene.render(width=panel, height=panel, camera=cam, backend="cpu").image

    # 2. Bakes (the body from itself).
    t0 = time.perf_counter()
    baked = bake(
        scene,
        body,
        None,
        ["ao", "curvature"],
        resolution=res,
        samples=2,
        ao_rays=32 if args.quick else 128,
        curvature_radius=0.04,
        backend="cpu" if args.backend == "cpu" else "auto",
    )
    timings["bake_ao_curvature"] = time.perf_counter() - t0
    report["bake"] = {
        k: baked.report[k] for k in ("maps", "resolution", "backend", "miss_fraction", "rays")
    }
    report["bake"]["seconds"] = baked.report["seconds"]
    ao = baked.maps["ao"][..., 0]
    curvature = baked.maps["curvature"][..., 0]
    if baked.report["miss_fraction"] > 1e-5:  # float32 GPU traversal: a few in a million
        failures.append(f"self-bake missed {baked.report['miss_fraction']:.2%} of samples")

    # 3. Painting.
    t0 = time.perf_counter()
    color = scene.add_painted_texture("Cell color", res, res, background="#d9822b")
    rough = scene.add_painted_texture("Cell roughness", res, res, encoding="data",
                                      background=[0.38, 0.38, 0.38, 1])  # fmt: skip
    metal = scene.add_painted_texture("Cell metallic", res, res, encoding="data",
                                      background=[0.2, 0.2, 0.2, 1])  # fmt: skip
    glow = scene.add_painted_texture("Cell glow", res, res, background="#000000")
    with scene.transaction("Paint the power cell"):
        for doc in (color, rough, metal, glow):
            doc.add_layer("Parts")
        # Face groups: 1 recess, 2 glowing panel, 3 duct.
        fill(scene, body, color, "#2b2f36", where={"groups": [1]})
        fill(scene, body, color, "#1b2a30", where={"groups": [2]})
        fill(scene, body, color, "#5a6068", where={"groups": [3]})
        fill(scene, body, rough, [0.7], where={"groups": [1]})
        fill(scene, body, rough, [0.3], where={"groups": [3]})
        fill(scene, body, metal, [0.9], where={"groups": [3]})
        fill(scene, body, glow, "#38c6e8", where={"groups": [2]})
        # A hazard stripe across the top, a gradient shading the duct, rivets on the front.
        color.add_layer("Stripe")
        stroke(scene, body, {"texture": color, "layer": "Stripe", "color": "#f2c230"},
               {"radius": 0.05, "hardness": 0.9, "falloff": "smooth", "spacing": 0.1},
               [[-0.88, -0.45, 0.4], [-0.88, 0.45, 0.4]])  # fmt: skip
        color.add_layer("Shade")
        gradient(scene, body, {"texture": color, "layer": "Shade"}, [0, 0, -0.4], [0, 0, 0.4],
                 [[0.0, 0.0, 0.0, 0.35], [0.0, 0.0, 0.0, 0.0]])  # fmt: skip
        color.add_layer("Rivets")
        for x in (-0.55, -0.2, 0.2, 0.55):
            for z in (-0.32, 0.22):
                stroke(scene, body, {"texture": color, "layer": "Rivets", "color": "#1e2126"},
                       {"radius": 0.018, "hardness": 0.8}, [[x, -0.62, z]])  # fmt: skip
        # Grime: crevices darkened by a multiply layer masked by (1 - AO).
        color.add_layer("Grime", kind="fill", blend="multiply", value=[0.55, 0.5, 0.45, 1])
        color.set_mask("Grime", smoothstep(0.15, 0.7, 1.0 - ao))
        rough.add_layer("Grime", kind="fill", value=[0.75, 0.75, 0.75, 1], opacity=0.6)
        rough.set_mask("Grime", smoothstep(0.15, 0.7, 1.0 - ao))
        # Wear: convex edges to bare metal on every channel.
        wear = smoothstep(0.56, 0.68, curvature)
        for doc, value in ((color, [0.78, 0.76, 0.72, 1]), (rough, [0.22, 0.22, 0.22, 1]),
                           (metal, [1.0, 1.0, 1.0, 1])):  # fmt: skip
            doc.add_layer("Wear", kind="fill", value=value)
            doc.set_mask("Wear", wear)
        material = scene.create_material("Cell paint", roughness=1.0, metallic=1.0)
        material.set_texture("base_color", color)
        material.set_texture("roughness", rough, channel="r")
        material.set_texture("metallic", metal, channel="r")
        material.set_texture("emissive", glow)
        material.base_color = (1.0, 1.0, 1.0)
        material.emissive = (1.0, 1.0, 1.0)
        material.emissive_strength = 1.5
        body.material = material
        knob.material = scene.create_material("Rubber", base_color="#1d1d1f", roughness=0.8)
        bezel.material = scene.create_material("Chrome", base_color="#d8dde3", metallic=1.0,
                                               roughness=0.15)  # fmt: skip
    timings["paint"] = time.perf_counter() - t0
    report["textures"] = {doc.name: doc.info() for doc in (color, rough, metal, glow)}
    report["wear_texels"] = int((wear > 0.5).sum())
    if report["wear_texels"] == 0:
        failures.append("the curvature mask found no convex edge")

    # 4. Render.
    t0 = time.perf_counter()
    final = render(scene, cam, panel, panel, args.backend, samples)
    timings["render"] = time.perf_counter() - t0
    report["render"] = final.backend

    # The picture.
    islands = uv.layout_image(body.mesh, panel, mode="islands")
    dens = uv.layout_image(body.mesh, panel, mode="density")
    half = panel // 2

    def gray(x):
        return fit(np.repeat(to8(x)[..., None], 3, axis=2), half)

    maps = np.concatenate(
        [
            np.concatenate([gray(ao), gray(curvature)], axis=1),
            np.concatenate([gray(rough.read()[..., 0]), gray(metal.read()[..., 0])], axis=1),
        ],
        axis=0,
    )
    painted = uv.layout_image(body.mesh, panel, mode="texture", texture=color.read())
    top = np.concatenate([islands, dens, checker_view], axis=1)
    bottom = np.concatenate([maps, painted, final.image], axis=1)
    png = encode_png(
        np.concatenate([top, bottom], axis=0), text={"Software": f"Aether3D {__version__}"}
    )
    (out / "texturing_hard_surface.png").write_bytes(png)
    (out / "texturing_hard_surface.json").write_text(json.dumps(report, indent=2, default=str))
    for k, v in report["uv"].items():
        print(f"{k}: {v['islands']} islands, density {v['texel_density']['mean']:.1f} texels/m, "
              f"stretch {v['stretch_l2_max']:.3f}")  # fmt: skip
    print(f"bake: {baked.report['seconds']['total']:.2f} s ({baked.report['backend']}); "
          f"render: {timings['render']:.2f} s ({final.backend.get('name')})")  # fmt: skip
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
