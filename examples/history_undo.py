"""Undo, redo, transactions, checkpoints and a saved history (Stage 9).

Run from the repository root:

    python examples/history_undo.py [OUTPUT_DIR] [--quick]

Builds a small courtyard -- a procedural tower, a stone wall, a crate, a camera and
a sun -- and then edits it the way a person or an agent would, with every change
recorded in ``scene.history``:

1. single changes (a move, a color, a procedural parameter, a material, a mesh edit,
   a duplicate), each one undo step;
2. a transaction that builds a gate in one step, and one that fails half-way and is
   rolled back as a whole;
3. a checkpoint, more changes, and a return to the checkpoint;
4. a glTF file merged into the scene (an undoable import);
5. undo all the way back. The render then equals the very first one, pixel for pixel,
   and redo brings the edited scene back the same way;
6. the project saved with its history, loaded again, and its history replayed and
   checked (``history.verify()``).

Writes ``history_undo.png`` (the first render, the edited scene, and the scene after
undoing everything) and ``history_undo.json`` (the steps and checks). Exits with
status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from showcase import caption_strip, write  # noqa: E402

import aether3d as a3  # noqa: E402


def build_scene() -> a3.Scene:
    scene = a3.Scene("Courtyard")
    with scene.history.disabled():  # building is not an edit: start with an empty history
        ground = scene.add_plane(name="Ground", size=14)
        ground.color = "#7d8a5a"
        stone = scene.create_material("Stone", preset="stone", scale=2.0)
        tower = scene.add_procedural(
            "Tower", "cylinder", radius=1.2, depth=5, segments=12, position=(-3, 2, 2.5)
        )
        tower.geometry.add_modifier("bevel", width=0.08, limit="angle", angle=40)
        tower.material = stone
        wall = scene.add_cube(name="Wall", position=(1.5, 3, 1))
        wall.scale = (5, 0.6, 2)
        wall.material = stone
        crate = scene.add_cube(name="Crate", size=1.2, position=(1, -1, 0.6))
        crate.color = "#a0703c"
        scene.add_camera(name="Camera", position=(9, -11, 7), look_at=(0, 1, 1.5), fov=38)
        scene.add_directional_light(name="Sun", direction=(-0.4, 0.6, -1.0), intensity=2.5)
    return scene


def gltf_asset(folder: Path) -> Path:
    """A small separate asset (a barrel and a lamp post) exported to glTF."""
    asset = a3.Scene("Props")
    barrel = asset.add_object("Barrel", mesh=a3.create_sphere(0.5, 24, 12))
    barrel.scale = (1, 1, 1.4)
    barrel.color = "#5b3a21"
    post = asset.add_cube(name="Post", size=0.3)
    post.scale = (1, 1, 10)
    post.position = (1.2, 0, 1.5)
    path = folder / "props.glb"
    asset.export(path, allow_lossy=True)
    return path


def render(scene: a3.Scene, size: tuple[int, int]) -> np.ndarray:
    return scene.render(width=size[0], height=size[1], samples=2).image


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--quick", action="store_true", help="small images")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    size = (240, 150) if args.quick else (640, 400)
    problems: list[str] = []

    def check(ok: bool, what: str) -> None:
        print(f"   {'ok' if ok else 'FAILED'}: {what}")
        if not ok:
            problems.append(what)

    scene = build_scene()
    history = scene.history
    first = render(scene, size)
    first_doc = json.dumps(scene.to_dict(), sort_keys=True)

    print("1. Single changes, one step each")
    crate = scene.get_by_name("Crate")
    crate.position = (2.5, -1.5, 0.6)
    crate.color = "#c0392b"
    scene.procedural.geometry("Tower").set_source(segments=6)
    scene.materials["Stone"].roughness = 0.35
    with scene.get_by_name("Wall").edit_mesh() as edit:
        top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
        edit.extrude_faces(top, 0.4)
    scene.duplicate(crate, "Crate 2").position = (3.6, -0.2, 0.6)
    check(len(history) == 7, f"7 steps recorded ({len(history)})")

    print("2. Transactions")
    with scene.transaction("Build the gate") as tx:
        for x in (-1.3, 0.3):
            post = scene.add_cube(name=f"Gate post {x:+.1f}", size=0.5, position=(x, 1.2, 1.4))
            post.scale = (1, 1, 5.6)
        lintel = scene.add_cube(name="Lintel", size=0.5, position=(-0.5, 1.2, 2.95))
        lintel.scale = (4.2, 1, 0.6)
    check(tx.entry is not None and len(history) == 8, f"the gate is one step: {tx.entry.label!r}")
    before_failure = json.dumps(scene.to_dict(), sort_keys=True)
    try:
        with scene.transaction("Demolish"):
            scene.remove(scene.get_by_name("Tower"))
            scene.world.background = "#000000"
            raise RuntimeError("changed our mind half-way")
    except RuntimeError:
        pass
    check(
        json.dumps(scene.to_dict(), sort_keys=True) == before_failure and len(history) == 8,
        "the failed transaction left nothing behind",
    )

    print("3. A checkpoint")
    history.checkpoint("gate")
    scene.get_by_name("Sun").light.intensity = 0.8
    scene.world.background = "#d07040"  # dusk
    at_dusk = json.dumps(scene.to_dict(), sort_keys=True)
    history.goto("gate")
    check(scene.get_by_name("Sun").light.intensity == 2.5, "back at the checkpoint")
    history.redo(2)
    check(json.dumps(scene.to_dict(), sort_keys=True) == at_dusk, "and forward again")

    print("4. Importing a glTF file into the scene (undoable)")
    with tempfile.TemporaryDirectory() as tmp:
        props = scene.merge(gltf_asset(Path(tmp)), parent=None, allow_lossy=True)
    for obj in props:
        obj.position = (obj.position[0] - 4, obj.position[1] - 3, obj.position[2] + 0.7)
    check(len(props) == 2, f"merged {', '.join(o.name for o in props)}")
    edited = render(scene, size)
    edited_doc = json.dumps(scene.to_dict(), sort_keys=True)
    steps = len(history)

    print("5. Undo everything, then redo it")
    for entry in history.entries:
        print(f"      {entry.index:3d}  {entry.label}")
    history.undo(steps)
    restored = render(scene, size)
    check(json.dumps(scene.to_dict(), sort_keys=True) == first_doc, "the document is the first one")
    check(np.array_equal(restored, first), "the render equals the first one, pixel for pixel")
    history.redo(steps)
    check(json.dumps(scene.to_dict(), sort_keys=True) == edited_doc, "redo restores the edits")
    check(np.array_equal(render(scene, size), edited), "and their render")

    print("6. Saved with its history, loaded again, replayed")
    with tempfile.TemporaryDirectory() as tmp:
        project = Path(tmp) / "courtyard.aether"
        scene.save(project, history=True)
        loaded = a3.Scene.load(project, history=True)
        report = loaded.history.verify()
        check(report["ok"], f"the loaded history replays exactly ({report['steps']} steps)")
        loaded.history.undo(steps)
        check(
            json.dumps(loaded.to_dict(), sort_keys=True) == first_doc,
            "undo in the new session gives the first document",
        )
        history_bytes = (Path(tmp) / "courtyard.aether.history").stat().st_size

    strip = caption_strip(
        [
            ("First render", first),
            (f"Edited ({steps} steps)", edited),
            ("All undone: identical", restored),
        ]
    )
    image = write(out / "history_undo.png", strip)
    data = {
        "aether3d": a3.__version__,
        "steps": [e.to_dict() for e in history.entries],
        "checkpoints": history.checkpoints,
        "history_bytes_in_memory": history.nbytes,
        "history_file_bytes": history_bytes,
        "problems": problems,
    }
    (out / "history_undo.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {image} and history_undo.json; {history.nbytes / 1024:.0f} KiB of history")
    print("done" if not problems else f"{len(problems)} check(s) failed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
