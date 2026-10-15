"""Picking and probing a character scene: the blocky hero, a crate and a ball (Stage 10).

Run from the repository root:

    python examples/hero_picking.py [OUTPUT_DIR] [--quick]

1. Builds the hero scene of ``examples/hero_render.py`` (a hierarchy: Hero ->
   Torso -> Head, arms -> Sword) and sets a crate and a ball beside the hero.
2. Uses the spatial queries the way an editor or an agent would, every answer
   exact and read-only:

   * ``pick`` -- what is under the cursor, at a few "clicks" (head, sword,
     crate, ground, sky), with the surface point, normal, UV and material;
   * a picked id map: one pick per pixel cell, compared with the rasterizer's
     object ids (``result.object_index``);
   * ``raycast`` -- drop the crate onto whatever is below a point (placement),
     then ``overlaps`` to check it only touches the ground;
   * ``bounds(children=True)`` -- one box around the whole hero hierarchy;
   * ``nearest_objects`` / ``closest_point`` -- what is closest to the sword's
     tip;
   * ``overlaps`` -- exact contacts: the feet and the sword's tip go into the
     grass block, which the box-only candidates cannot tell from near misses;
   * ``project`` and ``visible_from`` -- where the sword's tip appears in the
     image and how much of the head the camera sees.
3. Writes ``hero_picking.png`` (the view with the clicks and the hero's box drawn
   in, the rasterizer's ids and the picked ids side by side) and
   ``hero_picking.json``.
4. Checks the answers; exits with status 1 if one fails. ``--quick`` is smaller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hero_render import build_scene as build_hero  # noqa: E402
from overlay import box, cross, id_image  # noqa: E402
from showcase import caption_strip, write  # noqa: E402


def build_scene(width: int, height: int):
    scene = build_hero(width, height, 1)
    crate = scene.add_cube(name="Crate", size=0.6, position=(1.4, 0.6, 3.0))
    crate.color = "#a0703a"
    crate.tags = ["prop", "movable"]
    ball = scene.add_sphere(name="Ball", radius=0.3, position=(-1.2, 0.4, 0.3))
    ball.color = "#d03a2f"
    ball.tags = ["prop"]
    scene.get_by_name("Hero").tags = ["character"]
    scene.get_by_name("Sword").set_metadata("damage", 7)
    return scene


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="a smaller image and pick grid")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (320, 180) if args.quick else (640, 360)
    scene = build_scene(width, height)
    problems: list[str] = []
    report: dict = {}

    def check(ok: bool, what: str) -> None:
        if not ok:
            problems.append(what)
        print(("ok    " if ok else "FAIL  ") + what)

    # Placement: drop the crate onto the surface below it (one undoable move).
    crate = scene.get_by_name("Crate")
    below = scene.raycast(crate.world_position, (0, 0, -1), exclude=crate)
    lowest = scene.bounds(crate).aabb.min[2] - crate.world_position[2]
    crate.position = (crate.position[0], crate.position[1], below.position[2] - lowest)
    report["placement"] = {"surface": below.to_dict(), "crate_z": crate.position[2]}
    check(below.object.name == "Grass Block", "the crate lands on the grass block")
    # (pairs come in scene order: the grass block was added before the crate)
    touching = {(o.a.name, o.b.name) for o in scene.overlaps("Crate", against=scene.objects)}
    check(touching == {("Grass Block", "Crate")}, "the placed crate touches only the ground")

    revision = scene.revision
    render = scene.render(width=width, height=height, samples=1)
    names = {o["index"]: o["name"] for o in render.objects}
    index_of = {name: k for k, name in names.items()}

    def screen(name: str):
        point = scene.bounds(name).aabb.center
        p = scene.project(point, width=width, height=height)
        return p.x, p.y

    clicks = {
        "head": screen("Head"),
        "sword": scene.project(
            scene.closest_point((0.4, -0.4, 1.3), objects="Sword").position,
            width=width,
            height=height,
        ),
        "crate": screen("Crate"),
        "ground": (width * 0.15, height * 0.92),
        "sky": (width * 0.5, 2.0),
    }
    clicks["sword"] = (clicks["sword"].x, clicks["sword"].y)
    picked = {}
    for label, (x, y) in clicks.items():
        hit = scene.pick(x, y, width=width, height=height)
        picked[label] = None if hit is None else hit.to_dict()
        print(f"  click {label:6s} ({x:6.1f}, {y:6.1f}) -> {hit.object.name if hit else 'nothing'}")
    report["clicks"] = picked
    check(picked["head"]["object"]["name"] == "Head", "a click on the head picks the head")
    check(picked["sword"]["object"]["name"] == "Sword", "a click on the sword picks the sword")
    check(picked["crate"]["object"]["name"] == "Crate", "a click on the crate picks the crate")
    check(picked["ground"]["object"]["name"] == "Grass Block", "the ground is the grass block")
    check(picked["sky"] is None, "the sky picks nothing")
    check(picked["head"]["front_face"] is True, "picked surfaces face the camera")

    # A picked id map: one pick per cell, against the rasterizer's ids.
    step = 4 if args.quick else 2
    ids = render.object_index
    picks = np.full((height // step, width // step), -1, dtype=np.int64)
    agree = total = 0
    for r in range(picks.shape[0]):
        for c in range(picks.shape[1]):
            col, row = c * step + step // 2, r * step + step // 2
            hit = scene.pick(col + 0.5, row + 0.5, width=width, height=height)
            picks[r, c] = -1 if hit is None else index_of.get(hit.object.name, -1)
            block = ids[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2]
            if np.all(block == ids[row, col]):
                total += 1
                agree += picks[r, c] == ids[row, col]
    report["id_map"] = {"interior_samples": total, "agreeing": int(agree)}
    check(total > 500 and agree == total, "picked ids match the rasterizer away from silhouettes")

    hero = scene.bounds("Hero", children=True)
    report["hero_bounds"] = hero.to_dict()
    check(hero.objects == 7, "the hero's box covers its seven parts")

    tip = scene.bounds("Sword").aabb.min  # the blade points down
    nearest = scene.nearest_objects(scene.closest_point(tip, objects="Sword").position, count=3)
    report["near_the_sword_tip"] = [h.to_dict() for h in nearest]
    contacts = {(o.a.name, o.b.name) for o in scene.overlaps()}
    report["contacts"] = sorted(contacts)
    check(("Sword", "Grass Block") in contacts, "the sword's tip goes into the grass block")
    check(
        ("Right Leg", "Grass Block") in contacts and ("Left Leg", "Grass Block") in contacts,
        "both feet are sunk slightly into the grass block",
    )
    loose = scene.overlaps(exact=False)
    check(len(loose) > len(contacts), "some box overlaps are near misses, not contacts")

    head = scene.visible_from("Head", scene.active_camera, samples=128, width=width, height=height)
    report["head_visibility"] = head.to_dict()
    check(0.2 < head.fraction < 0.8, "the camera sees part of the head (its front)")
    check(scene.revision == revision, "the queries changed nothing")

    view = render.image.copy()
    box(scene, view, np.asarray(hero.obb.corners()), (255, 220, 0))
    for label, (x, y) in clicks.items():
        cross(view, x, y, (255, 255, 255) if picked[label] else (255, 80, 80), size=5)
    picked_image = id_image(np.repeat(np.repeat(picks, step, axis=0), step, axis=1))
    strip = caption_strip(
        [
            ("Clicks and the hero's box", view),
            ("Rasterizer ids", id_image(ids)),
            ("Picked ids (ray per cell)", picked_image),
        ]
    )
    write(out / "hero_picking.png", strip)
    report["problems"] = problems
    (out / "hero_picking.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out / 'hero_picking.png'} and {out / 'hero_picking.json'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
