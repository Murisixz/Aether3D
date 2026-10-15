"""Spatial inspection of the Minecraft castle: what is where, answered exactly (Stage 10).

Run from the repository root (booleans need ``pip install 'aether3d[boolean]'``):

    python examples/castle_spatial.py [OUTPUT_DIR] [--quick]

1. Builds the castle of ``examples/minecraft_castle_render.py`` and annotates its
   parts with tags and metadata (``tree``, ``bush``, ``fortification``, ...).
2. Asks Aether3D about it. Every answer is computed from the triangles in world
   space and changes nothing (no history step, same revision):

   * ``describe`` -- a compact summary of every object (for tools and agents);
   * ``bounds`` -- the site, and the keep with its watchtower and roof
     (``children=True``: one box around the hierarchy);
   * ``find_objects`` -- the trees by tag, the objects inside the courtyard;
   * a height map -- one ray straight down per cell over the site;
   * ``overlaps`` -- exact contacts (trunks in their leaves, bushes sunk into
     the ground) next to the box-only candidates, which are not collisions;
   * ``relation`` -- the watchtower stands on the keep, the keep on the ground;
   * ``line_of_sight`` -- through the gate, and into the wall beside it;
   * ``visible_from`` -- how much of the keep the camera sees;
   * ``pick`` -- the object under a grid of pixels, compared with the
     rasterizer's object ids.
3. Writes cheap diagnostic images next to a JSON report:

   * ``castle_spatial_heights.png`` -- the height map (dark: low, bright: high);
   * ``castle_spatial_view.png`` -- the rasterized view with the keep's
     hierarchy box, the picked points and the sight lines drawn in.
4. Checks every answer against facts about the castle; exits with status 1 if one
   fails. ``--quick`` uses a coarser height map and pick grid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from minecraft_castle_render import build_scene as build_castle  # noqa: E402
from overlay import box, cross, id_image, segment  # noqa: E402
from showcase import caption_strip, write  # noqa: E402

from aether3d.spatial import Box  # noqa: E402

PARTS = {
    "fortification": ["Curtain Wall", "Towers", "Keep", "Watchtower", "Watchtower Roof"],
    "landscape": ["Ground", "Path"],
}


def annotate(scene) -> None:
    """Tags and metadata: one undo step, saved with the project."""
    with scene.transaction("Annotate the castle"):
        for tag, names in PARTS.items():
            for name in names:
                scene.get_by_name(name).add_tags(tag)
        for obj in scene.find_objects(name="Trunk *"):
            obj.add_tags("tree", "trunk")
        for obj in scene.find_objects(name="Leaves *"):
            obj.add_tags("tree", "leaves")
        for obj in scene.find_objects(name="Bush *"):
            obj.add_tags("bush")
        scene.get_by_name("Keep").metadata = {"floors": 3, "role": "residence"}
        scene.get_by_name("Watchtower").set_metadata("lookout", True)


def height_map(scene, cells: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Top surface height and object of every cell (rays straight down over the site)."""
    site = scene.bounds()
    (x0, y0, _), (x1, y1, z1) = site.aabb.min, site.aabb.max
    heights = np.full((cells, cells), np.nan)
    owners = np.full((cells, cells), -1)
    names: list[str] = []
    for row in range(cells):
        y = y1 - (row + 0.5) * (y1 - y0) / cells  # image rows run from +y (top) to -y
        for col in range(cells):
            x = x0 + (col + 0.5) * (x1 - x0) / cells
            hit = scene.raycast((x, y, z1 + 1.0), (0, 0, -1))
            if hit is not None:
                heights[row, col] = hit.position[2]
                if hit.object.name not in names:
                    names.append(hit.object.name)
                owners[row, col] = names.index(hit.object.name)
    return heights, owners, names


def height_image(heights: np.ndarray, scale: int) -> np.ndarray:
    lo, hi = np.nanmin(heights), np.nanmax(heights)
    level = np.where(np.isnan(heights), 0.0, (heights - lo) / max(hi - lo, 1e-9))
    gray = (40 + 215 * level).astype(np.uint8)
    image = np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)
    return np.stack([image] * 3, axis=-1)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="coarser height map and pick grid")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (480, 270) if args.quick else (960, 540)
    cells = 32 if args.quick else 96
    step = 24 if args.quick else 12

    scene = build_castle(width, height, 1)
    annotate(scene)
    revision, steps = scene.revision, len(scene.history)
    report: dict = {"scene": scene.name}
    problems: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)
        print(("ok    " if ok else "FAIL  ") + what)

    t0 = time.perf_counter()
    summary = scene.describe()
    report["describe"] = summary
    site = scene.bounds()
    keep = scene.bounds("Keep", children=True)
    report["bounds"] = {"site": site.to_dict(), "keep_with_children": keep.to_dict()}
    roof = scene.get_by_name("Watchtower Roof").bounds()
    check(keep.objects == 3, "the keep's hierarchy box covers the keep, watchtower and roof")
    check(keep.aabb.max[2] == roof.aabb.max[2] == site.aabb.max[2], "the roof is the top")

    trees = scene.find_objects(tags="tree")
    courtyard = Box((-11.0, -11.0, -1.0), (11.0, 11.0, 30.0))
    inside = scene.query_region(courtyard, mode="inside")
    report["find"] = {
        "trees": [o.name for o in trees],
        "inside_the_courtyard": [m.object.name for m in inside],
    }
    check(len(trees) == 10, "ten tree parts are tagged (five trunks, five crowns)")
    check(
        {m.object.name for m in inside}
        >= {"Keep", "Watchtower", "Watchtower Roof", "Bush 1", "Bush 2", "Bush 3", "Bush 4"},
        "the keep, its tower and the bushes stand inside the courtyard",
    )

    heights, owners, names = height_map(scene, cells)
    top = np.unravel_index(np.nanargmax(heights), heights.shape)
    report["height_map"] = {
        "cells": cells,
        "max": float(heights[top]),
        "highest": names[owners[top]],
        "objects_seen_from_above": names,
    }
    check(
        names[owners[top]] == "Watchtower Roof" and heights[top] <= roof.aabb.max[2],
        "the height map peaks on the watchtower roof",
    )

    exact = scene.overlaps()
    candidates = scene.overlaps(exact=False)
    contacts = {(o.a.name, o.b.name) for o in exact}
    report["overlaps"] = {
        "exact": [o.to_dict() for o in exact],
        "bounds_only_candidates": len(candidates),
    }
    check(
        all((f"Trunk {k}", f"Leaves {k}") in contacts for k in range(1, 6)),
        "every trunk reaches into its leaves",
    )
    check(
        all(("Ground", f"Bush {k}") in contacts for k in range(1, 5)),
        "the bushes are sunk into the ground",
    )
    check(len(candidates) > len(exact), "box overlaps include pairs that do not touch")

    on_keep = scene.relation("Watchtower", "Keep")
    on_ground = scene.relation("Keep", "Ground")
    report["relations"] = {
        "watchtower_keep": on_keep.to_dict(),
        "keep_ground": on_ground.to_dict(),
    }
    check(on_keep.above and on_keep.intersecting, "the watchtower stands on the keep")
    check(on_ground.above and not on_ground.below, "the keep stands on the ground")

    door = (0.0, -0.25, 1.0)  # just in front of the keep (its front face is at y = 0)
    gate = scene.line_of_sight((0.0, -18.0, 1.0), door)
    wall = scene.line_of_sight((6.0, -18.0, 1.0), door)
    report["line_of_sight"] = {"through_the_gate": gate.to_dict(), "beside_it": wall.to_dict()}
    check(gate.clear, "the path sees the keep through the gate")
    check(
        not wall.clear and wall.blocker.object.name in ("Curtain Wall", "Towers"),
        "beside the gate the wall blocks the view",
    )

    seen = scene.visible_from("Keep", scene.active_camera, samples=256, width=width, height=height)
    report["visible_from"] = seen.to_dict()
    check(0.0 < seen.fraction < 1.0, "the camera sees part of the keep, not all of it")

    render = scene.render(width=width, height=height, samples=1)
    ids = render.object_index
    names = {o["index"]: o["name"] for o in render.objects}
    agree = total = 0
    picks = []
    for row in range(step // 2, height, step):
        for col in range(step // 2, width, step):
            block = ids[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2]
            hit = scene.pick(col + 0.5, row + 0.5, width=width, height=height)
            picks.append((col + 0.5, row + 0.5, hit))
            if np.any(block != block[1, 1]):
                continue  # on a silhouette the raster and the ray may sample differently
            total += 1
            agree += (hit.object.name if hit else None) == names.get(int(ids[row, col]))
    report["pick"] = {"interior_samples": total, "agreeing_with_the_rasterizer": agree}
    check(total > 100 and agree == total, "picking agrees with the rasterizer's object ids")
    report["seconds"] = time.perf_counter() - t0

    check((scene.revision, len(scene.history)) == (revision, steps), "the queries changed nothing")

    view = render.image.copy()
    corners = np.asarray(keep.obb.corners())
    box(scene, view, corners, (255, 220, 0))
    for x, y, hit in picks:
        if hit is not None and hit.object.name in ("Keep", "Watchtower", "Watchtower Roof"):
            cross(view, x, y, (255, 60, 60), size=2)
    for line, color in ((gate, (80, 255, 80)), (wall, (255, 80, 80))):
        a = scene.project(line.start, width=width, height=height)
        end = line.end if line.clear else line.blocker.position
        b = scene.project(end, width=width, height=height)
        segment(view, (a.x, a.y), (b.x, b.y), color)
    scale = max(1, (height - 34) // cells)
    panels = [
        ("Rasterized view, keep box and sight lines", view),
        ("Heights from above (ray per cell)", height_image(heights, scale)),
        ("Rasterizer ids", id_image(ids)),
    ]
    write(out / "castle_spatial_view.png", caption_strip(panels[:1]))
    write(out / "castle_spatial_heights.png", caption_strip(panels[1:]))
    report["problems"] = problems
    (out / "castle_spatial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out / 'castle_spatial.json'} and two images ({report['seconds']:.1f} s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
