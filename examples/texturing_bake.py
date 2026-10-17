"""Baking a high-resolution sculpt onto a low-resolution mesh, by Aether3D.

Run from the repository root:

    python examples/texturing_bake.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

1. **High.** The creature of ``examples/organic_sculpt.py`` (same block-out and strokes),
   remeshed finely, then covered with small sculpted warts and a scaly ridge -- detail a
   coarse mesh cannot carry.
2. **Low.** The same surface remeshed coarsely (isotropic, about 1/20 of the triangles),
   unwrapped.
3. **Bake** from high to low through the low mesh's UVs: tangent-space normals (matching
   the renderers' MikkTSpace tangents), ambient occlusion and curvature, 2x2 samples per
   texel, padded; ``apply`` binds them to the low mesh's material.
4. **Measure.** Renders of the high mesh, the plain low mesh and the low mesh with its
   maps, from the same camera and light; the RMS difference of each low render to the
   high render (over the pixels the creature covers) must drop with the maps. Misses
   are reported.

Writes ``texturing_bake.png`` (top: high, low, low + maps; bottom: the normal, AO and
curvature maps) and ``texturing_bake.json``. Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import Scene, __version__
from aether3d.rendering.png import encode_png
from aether3d.sculpt import SculptMesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from organic_sculpt import HORN, SMOOTH, blockout, strokes  # noqa: E402


def creature(target: float) -> SculptMesh:
    sm = SculptMesh.from_mesh(blockout())
    sm.voxel_remesh(target * 0.9)
    sm.remesh(target, iterations=4)
    sm.set_mask(1.0, box=((-2, -2, -3), (2, 2, -1.45)))
    for _, s in strokes():
        sm.stroke(s)
    sm.stroke(HORN)
    sm.symmetrize("x", direction="+")
    sm.stroke(SMOOTH)
    sm.clear_mask()
    return sm


def detail(sm: SculptMesh, warts: int, seed: int = 3) -> None:
    """Warts at random points of the upper head and a ridge of scales down the back."""
    from aether3d.spatial.surface import TriangleSurface

    mesh = sm.to_mesh()
    p = mesh.positions[mesh.triangles].astype(np.float64)
    c = p.mean(axis=1)
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    weights = area * (c[:, 2] > -0.5)
    rng = np.random.default_rng(seed)
    for t in rng.choice(len(area), size=warts, p=weights / weights.sum()):
        sm.stroke({"brush": {"kind": "draw", "radius": 0.035 + 0.03 * rng.random(),
                             "strength": 0.5}, "path": [c[t].tolist()]})  # fmt: skip
    surface = TriangleSurface(mesh.positions.astype(np.float64), mesh.triangles)
    guide = np.array([[0, -0.6, 0.95], [0, 0.0, 1.08], [0, 0.6, 0.8], [0, 0.75, 0.1],
                      [0, 0.6, -0.8]])  # fmt: skip
    ridge = surface.nearest(guide).points
    for a, b in zip(ridge[:-1], ridge[1:], strict=True):
        for f in np.linspace(0, 1, 6, endpoint=False):
            sm.stroke({"brush": {"kind": "draw", "radius": 0.05, "strength": 0.6},
                       "path": [(a + (b - a) * f).tolist()]})  # fmt: skip


def masked_rms(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    d = (a.astype(np.float64) - b.astype(np.float64))[mask] / 255.0
    return float(np.sqrt((d * d).mean()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="coarser meshes, smaller maps")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    size = 320 if args.quick else 640
    res = 512 if args.quick else 2048
    fine, coarse, warts = (0.03, 0.12, 40) if args.quick else (0.012, 0.06, 120)
    failures: list[str] = []
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    sm = creature(fine)
    detail(sm, warts)
    high_mesh = sm.to_mesh()
    timings["high"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    sm.remesh(coarse, iterations=5)
    low_mesh = sm.to_mesh()
    timings["low"] = time.perf_counter() - t0

    scene = Scene("Bake demo")
    high = scene.add_object("High", mesh=high_mesh)
    low = scene.add_object("Low", mesh=low_mesh)
    t0 = time.perf_counter()
    unwrap = low.unwrap(resolution=res, padding=8)
    timings["unwrap"] = time.perf_counter() - t0
    clay = dict(base_color="#b98f6c", roughness=0.6)
    high.material = scene.create_material("High clay", **clay)
    low.material = scene.create_material("Low clay", **clay)
    scene.add_directional_light("Key", direction=(0.5, 0.8, -0.7), intensity=2.4)
    scene.add_directional_light("Rim", direction=(-0.6, -0.4, -0.3), intensity=0.7)
    cam = scene.add_camera("Cam", position=(2.6, -4.2, 1.6), look_at=(0, 0, 0))
    cam.frame(["High"], margin=0.12, aspect=1.0)
    backend = "cpu" if args.backend == "cpu" else "auto"

    def shot(visible: str):
        high.visible = visible == "High"
        low.visible = visible == "Low"
        return scene.render(width=size, height=size, camera=cam, backend="cpu")

    t0 = time.perf_counter()
    r_high = shot("High")
    r_low = shot("Low")
    timings["renders_before"] = time.perf_counter() - t0

    # Bake (High is hidden from the low renders, but bakes read the sources regardless).
    t0 = time.perf_counter()
    result = scene.bake(low, ["High"], ["normal", "ao", "curvature"], resolution=res,
                        samples=2, ao_rays=32 if args.quick else 128, padding=8,
                        backend=backend, apply=True)  # fmt: skip
    timings["bake"] = time.perf_counter() - t0
    r_baked = shot("Low")

    mask = (r_high.object_index >= 0) | (r_low.object_index >= 0)
    rms_plain = masked_rms(r_low.image, r_high.image, mask)
    rms_baked = masked_rms(r_baked.image, r_high.image, mask)
    if not rms_baked < 0.8 * rms_plain:
        failures.append(f"maps did not bring the low render closer: {rms_plain} -> {rms_baked}")
    miss = result.report["miss_fraction"]
    if miss > 0.001:
        failures.append(f"{miss:.2%} of the bake's samples missed the high mesh")

    def tile(name: str) -> np.ndarray:
        m = result.maps[name][..., :3]
        rows = np.minimum((np.arange(size) + 0.5) * m.shape[0] / size, m.shape[0] - 1)
        cols = np.minimum((np.arange(size) + 0.5) * m.shape[1] / size, m.shape[1] - 1)
        return np.round(m[rows.astype(int)][:, cols.astype(int)] * 255).astype(np.uint8)

    top = np.concatenate([r_high.image, r_low.image, r_baked.image], axis=1)
    bottom = np.concatenate([tile("normal"), tile("ao"), tile("curvature")], axis=1)
    png = encode_png(np.concatenate([top, bottom], axis=0),
                     text={"Software": f"Aether3D {__version__}"})  # fmt: skip
    (out / "texturing_bake.png").write_bytes(png)
    report = {
        "aether3d": __version__,
        "quick": args.quick,
        "high_triangles": high_mesh.triangle_count,
        "low_triangles": low_mesh.triangle_count,
        "unwrap": {k: v for k, v in unwrap.items() if k != "diagnostics"},
        "bake": {k: v for k, v in result.report.items() if k != "applied"},
        "applied": {k: t.name for k, t in result.report["applied"]["textures"].items()},
        "rms_to_high": {"low": rms_plain, "low_with_maps": rms_baked},
        "timings": timings,
        "failures": failures,
    }
    (out / "texturing_bake.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"high {high_mesh.triangle_count} triangles, low {low_mesh.triangle_count}; bake "
          f"{result.report['seconds']['total']:.2f} s ({result.report['backend']}), misses "
          f"{miss:.2e}; RMS to high: low {rms_plain:.4f}, low + maps {rms_baked:.4f}")  # fmt: skip
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
