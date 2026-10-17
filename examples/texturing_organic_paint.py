"""Painting a sculpt in 3D: the sculpted creature, unwrapped and painted, by Aether3D.

Run from the repository root:

    python examples/texturing_organic_paint.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

1. **Sculpt.** The creature bust of ``examples/organic_sculpt.py``, made by the same
   block-out and strokes (voxel remesh, isotropic remesh, symmetric strokes, a horn
   symmetrized).
2. **Unwrap.** Charts are flattened isometrically (SLIM) and split until their stretch is
   at most 1.15; ``uv_report`` checks for flips and overlaps.
3. **Paint** -- in 3D, never in the UV square, each stroke one undo step:

   * skin: the painted texture's background; a lighter belly: one soft, wide stroke
     under the snout and down the throat;
   * spots: a hundred dabs at random points of the head, mirrored (``symmetry: "x"``);
   * a dorsal stripe from the snout over the head down the neck: one surface stroke
     that crosses several UV seams without a visible edge;
   * the horns, bone white: ``fill`` of the triangles above the brow; the eyes dark;
   * war paint: a stroke in **view space**, drawn across the camera's image of the snout:
     only what the camera sees is painted (the far side and the hidden flanks are not);
   * roughness: shinier horns and eyes (a second, data texture painted by the same
     strokes as material strokes).

4. **Checks.** No flipped or overlapping UVs; one history step per stroke; the spots layer
   mirror-symmetric (the texel colors at mirrored surface points agree); the view stroke
   reached the visible side only; one undo removes the last stroke exactly.

Writes ``texturing_organic_paint.png`` (the clay sculpt, the painted sculpt, a second
view, and the painted texture over its UV layout) and ``texturing_organic_paint.json``.
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
from aether3d.spatial.surface import TriangleSurface
from aether3d.texturing import PaintView, fill, stroke, texel_map

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from organic_sculpt import HORN, SMOOTH, blockout, strokes  # noqa: E402


def render(scene, camera, width, height, backend):
    from aether3d import RenderError

    options = {"width": width, "height": height, "camera": camera, "backend": backend}
    if backend != "cpu":
        try:
            return scene.render(**options, shadows=True)
        except RenderError as exc:
            if exc.code != "BACKEND_UNSUPPORTED":
                raise
    return scene.render(**options)


def sculpt(scene: Scene, voxel: float, target: float):
    head = scene.add_object("Creature", mesh=blockout())
    with head.sculpt() as session:
        session.voxel_remesh(voxel)
        session.remesh(target, iterations=4)
    with head.sculpt() as session:
        session.set_mask(1.0, box=((-2, -2, -3), (2, 2, -1.45)))
        for _, s in strokes():
            session.stroke(s)
        session.stroke(HORN)
        session.symmetrize("x", direction="+")
        session.stroke(SMOOTH)
        session.clear_mask()
    return head


def texel_values(doc, layer, obj, points):
    """A layer's texel (RGBA, 0..1) at the surface texels nearest ``points``."""
    tm = texel_map(obj.mesh, doc.width, doc.height)
    pos, idx = [], []
    for t in range(tm.tiles_x * tm.tiles_y):
        tx = tm.texels(t)
        if tx.local.size == 0:
            continue
        x0, y0, _, _ = tm.tile_rect(t)
        ly, lx = np.divmod(tx.local, 64)
        pos.append(tx.position)
        idx.append(np.stack([y0 + ly, x0 + lx], axis=1))
    pos = np.concatenate(pos)
    idx = np.concatenate(idx)
    pixels = doc.read_layer(layer)
    out = []
    for p in points:
        k = int(np.argmin(np.linalg.norm(pos - p, axis=1)))
        out.append(pixels[idx[k, 0], idx[k, 1]])
    return np.array(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="coarser mesh, smaller images")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    size = 320 if args.quick else 640
    res = 512 if args.quick else 2048
    voxel, target = (0.06, 0.07) if args.quick else (0.035, 0.04)
    failures: list[str] = []
    timings: dict[str, float] = {}

    scene = Scene("Painted creature")
    t0 = time.perf_counter()
    head = sculpt(scene, voxel, target)
    timings["sculpt"] = time.perf_counter() - t0
    clay = scene.create_material("Clay", base_color="#c49a74", roughness=0.75, specular=0.3)
    head.material = clay
    ground = scene.add_plane("Ground", size=12, position=(0, 0, -1.8))
    scene.assign_material(ground, scene.create_material("Ground", base_color="#6f7378"))
    scene.add_directional_light("Key", direction=(0.5, 0.8, -0.7), intensity=2.4)
    scene.add_directional_light("Rim", direction=(-0.6, -0.4, -0.3), intensity=0.7)
    front = scene.add_camera("Front", position=(2.6, -4.2, 1.2), look_at=(0, 0, -0.1))
    front.frame(["Creature"], margin=0.3, aspect=1.0)
    side = scene.add_camera("Side", position=(-4.4, 1.4, 1.6), look_at=(0, 0, -0.1))
    side.frame(["Creature"], margin=0.3, aspect=1.0)
    before = render(scene, front, size, size, args.backend)

    # 2. Unwrap.
    t0 = time.perf_counter()
    unwrap = head.unwrap(resolution=res, padding=8)
    timings["unwrap"] = time.perf_counter() - t0
    uvr = head.uv_report(resolution=res)
    if uvr["flipped_triangles"] or uvr["overlapping_triangles"]:
        failures.append("flipped or overlapping UVs")

    # 3. Paint.
    color = scene.add_painted_texture("Skin", res, res, background="#6f8a4c")
    rough = scene.add_painted_texture("Skin roughness", res, res, encoding="data",
                                      background=[0.62, 0.62, 0.62, 1])  # fmt: skip
    for name in ("Belly", "Spots", "Stripe", "Horns", "War paint"):
        color.add_layer(name)
    rough.add_layer("Gloss")
    skin = scene.create_material("Skin", roughness=1.0, specular=0.4)
    skin.set_texture("base_color", color)
    skin.set_texture("roughness", rough, channel="r")
    skin.base_color = (1.0, 1.0, 1.0)
    head.material = skin
    history_before = len(scene.history.entries)
    t0 = time.perf_counter()
    reports = {}
    # Spots: random surface points of the head (seeded), mirrored.
    mesh = head.mesh
    surface = TriangleSurface(mesh.positions.astype(np.float64), mesh.triangles)
    rng = np.random.default_rng(14)
    p = mesh.positions[mesh.triangles].astype(np.float64)
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    top = p.mean(axis=1)[:, 2] > -0.6
    weights = area * top
    picks = rng.choice(len(area), size=50 if args.quick else 100, p=weights / weights.sum())
    r1, r2 = rng.random(len(picks)), rng.random(len(picks))
    s = np.sqrt(r1)
    spots = (1 - s)[:, None] * p[picks, 0] + (s * (1 - r2))[:, None] * p[picks, 1]
    spots += (s * r2)[:, None] * p[picks, 2]
    # Belly: under the snout and down the front of the throat, a soft, wide stroke.
    throat = surface.nearest(np.array([[0, -1.1, -0.6], [0, -0.8, -0.9], [0, -0.5, -1.2],
                                       [0, -0.45, -1.5]])).points  # fmt: skip
    reports["belly"] = stroke(
        scene, head, {"texture": color, "layer": "Belly", "color": "#d8cc8a"},
        {"radius": 0.38, "hardness": 0.0, "strength": 0.35, "spacing": 0.1}, throat,
    ).to_dict()  # fmt: skip
    spot_dabs = 0
    for i, point in enumerate(spots):
        rep = stroke(scene, head, {"texture": color, "layer": "Spots", "color": "#3a4a26"},
                     {"radius": 0.03 + 0.04 * rng.random(), "hardness": 0.6,
                      "symmetry": "x"}, [point])  # fmt: skip
        spot_dabs += rep.dabs
        if i == 0:
            reports["first_spot"] = rep.to_dict()
    # Dorsal stripe: from the snout over the head and down the back of the neck.
    guide = np.array(
        [[0, -1.2, 0.0], [0, -0.9, 0.6], [0, -0.4, 1.0], [0, 0.2, 1.05], [0, 0.7, 0.7],
         [0, 0.75, 0.0], [0, 0.6, -0.8], [0, 0.55, -1.3]]
    )  # fmt: skip
    path = surface.nearest(guide).points
    dense = np.concatenate(
        [np.linspace(a, b, 12, endpoint=False) for a, b in zip(path[:-1], path[1:], strict=True)]
    )
    dense = surface.nearest(dense).points
    reports["stripe"] = stroke(
        scene, head, [{"texture": color, "layer": "Stripe", "color": "#2c3a1c"},
                      {"texture": rough, "layer": "Gloss", "color": [0.45]}],
        {"radius": 0.09, "hardness": 0.4, "strength": 0.9}, dense,
    ).to_dict()  # fmt: skip
    # Horns: every triangle above the brow and out to the side.
    c = mesh.positions[mesh.triangles].mean(axis=1)
    horn_tris = np.flatnonzero((np.abs(c[:, 0]) > 0.5) & (c[:, 2] > 0.97))
    fill(scene, head, {"texture": color, "layer": "Horns"}, "#e8dcc0",
         where={"triangles": horn_tris.tolist()})  # fmt: skip
    fill(scene, head, {"texture": rough, "layer": "Gloss"}, [0.3],
         where={"triangles": horn_tris.tolist()})  # fmt: skip
    # Eyes: dark and glossy, mirrored.
    eye = surface.nearest(np.array([[0.32, -0.95, 0.22]])).points[0]
    stroke(scene, head, [{"texture": color, "layer": "Horns", "color": "#141210"},
                         {"texture": rough, "layer": "Gloss", "color": [0.12]}],
           {"radius": 0.07, "hardness": 0.8, "symmetry": "x"}, [eye])  # fmt: skip
    # War paint: drawn across the camera's picture of the snout.
    view = PaintView.render(scene, front, size, size)
    snout = view.project(
        surface.nearest(np.array([[-0.35, -1.05, 0.05], [0.35, -1.05, 0.05]])).points
    )[0]
    reports["war_paint"] = stroke(
        scene, head, {"texture": color, "layer": "War paint", "color": "#b8402c"},
        {"radius": size * 0.025, "hardness": 0.7}, snout, space="view", view=view,
    ).to_dict()  # fmt: skip
    timings["paint"] = time.perf_counter() - t0
    strokes_made = len(scene.history.entries) - history_before

    # 4. Checks.
    # One undo step per operation: belly, spots, stripe, 2 fills, eyes, war paint.
    expected = 1 + len(spots) + 1 + 2 + 1 + 1
    if strokes_made != expected:
        failures.append(f"{strokes_made} history steps for {expected} paint operations")
    probes = spots[:20]
    left = texel_values(color, "Spots", head, probes)
    right = texel_values(color, "Spots", head, probes * [-1, 1, 1])
    symmetry = float(np.abs(left[:, 3] - right[:, 3]).mean())
    if symmetry > 0.1:
        failures.append(f"spots not mirrored: mean alpha difference {symmetry:.3f}")
    far_side = texel_values(color, "War paint", head, [[0.0, 1.0, 0.2], [0.9, 0.3, 0.0]])
    if far_side[:, 3].max() > 0:
        failures.append("the view stroke painted surfaces the camera does not see")
    before_undo = color.read_layer("War paint").copy()
    scene.history.undo()
    if color.read_layer("War paint").any():
        failures.append("undo did not remove the view stroke")
    scene.history.redo()
    if not np.array_equal(color.read_layer("War paint"), before_undo):
        failures.append("redo did not restore the view stroke exactly")

    after = render(scene, front, size, size, args.backend)
    second = render(scene, side, size, size, args.backend)
    layout = uv.layout_image(head.mesh, size, mode="texture", texture=color.read())
    png = encode_png(
        np.concatenate([before.image, after.image, second.image, layout], axis=1),
        text={"Software": f"Aether3D {__version__}"},
    )
    (out / "texturing_organic_paint.png").write_bytes(png)
    report = {
        "aether3d": __version__,
        "quick": args.quick,
        "mesh": {"triangles": head.mesh.triangle_count, "vertices": head.mesh.vertex_count},
        "unwrap": {k: v for k, v in unwrap.items() if k != "diagnostics"},
        "uv": {
            k: uvr[k]
            for k in (
                "islands",
                "flipped_triangles",
                "overlapping_triangles",
                "stretch_l2",
                "texel_density",
                "utilization",
            )
        },  # fmt: skip
        "texture": color.info(),
        "paint": {"history_steps": strokes_made, "spot_dabs": spot_dabs, **reports},
        "spot_symmetry_alpha_difference": symmetry,
        "timings": timings,
        "render": after.backend,
        "failures": failures,
    }
    (out / "texturing_organic_paint.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"{head.mesh.triangle_count} triangles, {uvr['islands']} islands, stretch "
          f"{uvr['stretch_l2']['max']:.3f}; {strokes_made} paint steps in "
          f"{timings['paint']:.1f} s; spot symmetry {symmetry:.4f}")  # fmt: skip
    for f in failures:
        print("FAILED:", f)
    print("done")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
