"""Generate the example textures in ``examples/textures`` (original, made by Aether3D).

Run from the repository root:

    python examples/make_textures.py [OUTPUT_DIR] [--size N]

The images are *baked* from Aether3D procedural texture graphs -- an
explicit, optional output; the materials in the examples use the images to
demonstrate image textures (PNG and JPEG, color and normal maps). Nothing
here comes from a third party:

* ``stone_bricks_color.png`` / ``stone_bricks_normal.png`` -- weathered gray
  stone bricks with mortar grooves (a tangent-space normal map);
* ``wood_planks.jpg`` -- planks with growth rings (JPEG; needs Pillow);
* ``uv_grid.png`` -- a colored 8 x 8 grid whose top-left cell is white, to
  check texture orientation.

Every image tiles seamlessly: the brick and plank layouts repeat exactly
once per texture, and the noise in them is cross-faded across the borders
(``tileable``). The same command always produces the same pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from aether3d.materials.graph import EvalContext, TextureGraph, evaluate
from aether3d.rendering.color import linear_to_srgb, quantize
from aether3d.rendering.png import write_png

HERE = Path(__file__).resolve().parent

# 4 x 8 bricks per tile: rows alternate by half a brick, so the layout repeats every tile.
BRICKS = {
    "type": "bricks",
    "brick_width": 0.25,
    "brick_height": 0.125,
    "mortar": 0.012,
    "smoothness": 0.8,
    "variation": 0.8,
    "color1": "#7d7b74",
    "color2": "#a19e95",
    "mortar_color": "#5b5852",
    "seed": 11,
}
STONE_COLOR = {
    "output": "color",
    "nodes": {
        "bricks": BRICKS,
        "grime": {"type": "fbm", "scale": 6.0, "octaves": 5, "seed": 3},
        "shade": {
            "type": "ramp",
            "factor": {"node": "grime"},
            "stops": [[0.25, 0.62], [0.75, 1.05]],
        },
        "speckle": {"type": "noise", "kind": "value", "scale": 90.0, "seed": 5},
        "grain": {
            "type": "ramp",
            "factor": {"node": "speckle"},
            "stops": [[0.0, 0.92], [1.0, 1.04]],
        },
        "dirty": {
            "type": "mix",
            "a": {"node": "bricks"},
            "b": {"node": "shade"},
            "factor": 1.0,
            "mode": "multiply",
        },
        "color": {
            "type": "mix",
            "a": {"node": "dirty"},
            "b": {"node": "grain"},
            "factor": 1.0,
            "mode": "multiply",
        },
    },
}
STONE_HEIGHT = {
    "output": "height",
    "nodes": {
        "bricks": BRICKS,
        "solid": {"type": "math", "op": "invert", "a": {"node": "bricks", "output": "mortar"}},
        "pits": {"type": "fbm", "scale": 24.0, "octaves": 4, "seed": 7},
        "rough": {"type": "math", "op": "multiply", "a": {"node": "pits"}, "b": 0.18},
        "height": {"type": "math", "op": "add", "a": {"node": "solid"}, "b": {"node": "rough"}},
    },
}
PLANKS = {
    "output": "color",
    "nodes": {
        # Eight planks across a tile; each plank's grain is offset by its own random value.
        "boards": {
            "type": "stripes",
            "axis": "y",
            "scale": 8.0,
            "width": 0.94,
            "color1": 1.0,
            "color2": 0.35,
        },
        "wood": {
            "type": "wood",
            "scale": [2.0, 9.0, 2.0],
            "rings": 3.0,
            "distortion": 0.8,
            "grain": 0.45,
            "axis": "x",
            "color1": "#c9965e",
            "color2": "#7b4f2a",
            "seed": 4,
        },
        "color": {
            "type": "mix",
            "a": {"node": "wood"},
            "b": {"node": "boards"},
            "factor": 1.0,
            "mode": "multiply",
        },
    },
}


def tileable(graph: dict, size: int) -> np.ndarray:
    """A seamless tile: four shifted copies cross-faded so opposite edges match.

    ``f(u, v) = sum w * T(u - a, v - b)`` over ``a, b in {0, 1}`` with bilinear
    weights; the brick/plank layouts repeat every unit, so only noise blends.
    """

    def shifted(du: float, dv: float) -> np.ndarray:
        g = TextureGraph.from_dict(graph)
        u = (np.arange(size) + 0.5) / size
        v = 1.0 - (np.arange(size) + 0.5) / size
        uu, vv = np.meshgrid(u, v)
        p = np.stack([uu.ravel() - du, vv.ravel() - dv, np.zeros(uu.size)], axis=1)
        zero = np.zeros_like(p)
        return evaluate(g, EvalContext(p, zero, zero)).reshape(size, size, 4)

    u = ((np.arange(size) + 0.5) / size)[None, :, None]
    v = (1.0 - (np.arange(size) + 0.5) / size)[:, None, None]
    return (
        (1 - u) * (1 - v) * shifted(0, 0)
        + u * (1 - v) * shifted(1, 0)
        + (1 - u) * v * shifted(0, 1)
        + u * v * shifted(1, 1)
    )


def normal_map(height: np.ndarray, depth: float) -> np.ndarray:
    """Tangent-space normals (+X right, +Y up the image) from a periodic height field."""
    size = height.shape[0]
    # Central differences that wrap around: the normal map tiles like the height.
    dhdu = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * (size / 2.0)
    dhdv = (np.roll(height, 1, axis=0) - np.roll(height, -1, axis=0)) * (size / 2.0)  # rows go down
    n = np.stack([-depth * dhdu, -depth * dhdv, np.ones_like(height)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def to_srgb8(linear: np.ndarray) -> np.ndarray:
    out = np.empty(linear.shape[:2] + (4,), np.uint8)
    out[..., :3] = quantize(linear_to_srgb(linear[..., :3]))
    out[..., 3] = 255
    return out


def make(out_dir: Path, size: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    color = tileable(STONE_COLOR, size)
    written.append(write_png(out_dir / "stone_bricks_color.png", to_srgb8(color)))
    height = tileable(STONE_HEIGHT, size)[..., 0]
    normals = normal_map(height, depth=0.012)
    encoded = np.empty((size, size, 4), np.uint8)
    encoded[..., :3] = quantize(normals * 0.5 + 0.5)
    encoded[..., 3] = 255
    written.append(write_png(out_dir / "stone_bricks_normal.png", encoded))
    planks = tileable(PLANKS, size)
    try:
        from PIL import Image
    except ImportError:
        print("   Pillow is not installed: wood_planks.jpg (JPEG) was not written")
    else:
        path = out_dir / "wood_planks.jpg"
        Image.fromarray(to_srgb8(planks)[..., :3]).save(
            path, format="JPEG", quality=92, optimize=True
        )
        written.append(path)
    grid = np.zeros((size, size, 4), np.uint8)
    cell = size // 8
    y, x = np.mgrid[0:size, 0:size]
    grid[..., 0] = (x // cell) * 32 + 16
    grid[..., 1] = (y // cell) * 32 + 16
    grid[..., 2] = np.where(((x // cell) + (y // cell)) % 2 == 0, 220, 70)
    grid[..., 3] = 255
    grid[:cell, :cell, :3] = 255  # top-left (u = 0, v = 1) is white
    written.append(write_png(out_dir / "uv_grid.png", grid))
    return written


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "textures"))
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args(argv)
    for path in make(Path(args.output), args.size):
        print(f"   wrote {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
