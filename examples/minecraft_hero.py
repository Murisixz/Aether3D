"""A Minecraft-inspired blocky hero holding a sword, modelled with the editing engine.

Every mesh is built by editing operations -- add_box, bevel, inset, extrude,
loop cuts, subdivision, vertex scaling and merging -- never by writing vertex
arrays by hand. No textures or third-party assets are used.

Run from the repository root:

    python examples/minecraft_hero.py [OUTPUT_DIR] [--require-validator] [--blender]

Steps: model the sword and the character, validate every mesh's topology,
export ``OUTPUT_DIR/blocky_hero.glb`` (default ``./minecraft_hero_output``),
validate it with the official Khronos glTF Validator (if installed, see
``tools/gltf-validator``), import it back and compare, and -- with
``--blender`` -- open it in Blender (headless) and compare what Blender sees.
Exits with status 1 if any check fails.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from aether3d import Scene, import_gltf, inspect_mesh

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "tools" / "gltf-validator"
UP, FRONT = (0, 0, 1), (0, -1, 0)  # an asset's front faces -Y in Aether3D


def log(step: str, result=None) -> None:
    extra = ""
    if result is not None:
        created = result.created
        extra = f"  (+{len(created['vertices'])}v +{len(created['faces'])}f)"
        for d in result.diagnostics:
            extra += f" [{d.code}]"
    print(f"   - {step}{extra}")


def build_sword(scene: Scene, parent) -> None:
    sword = scene.add_object("Sword", parent=parent, position=(0, -0.05, -0.62))
    sword.rotation_euler = (-90, 0, 0)  # the blade points forward along the arm
    with sword.edit_mesh() as m:
        log("pommel: a 0.24 cube", m.add_box(0.24))
        log("bevel every pommel edge", m.bevel_edges(m.select_all(), 0.03))
        top = m.select_faces(normal=(UP, 1), expect=1)
        log("inset the top face (grip outline)", m.inset_faces(top, 0.035))
        log("extrude the grip", m.extrude_faces(top, 0.42))
        grip = next(iter(m.select_edges(nearest=(0.055, 0.055, 0.33)).edges))
        log("three loop cuts: grip wrapping bands", m.loop_cut(grip, cuts=3))
        log("extrude the guard core", m.extrude_faces(top, 0.1))
        guard_band = ((-1, -1, 0.5), (1, 1, 0.65))
        arms = m.select_faces(normal=((1, 0, 0), 1), box=guard_band, expect=1) | m.select_faces(
            normal=((-1, 0, 0), 1), box=guard_band, expect=1
        )
        log("extrude both crossguard arms", m.extrude_faces(arms, 0.28, mode="individual"))
        log("bevel the arm tips", m.bevel_edges(m.to_edges(arms), 0.02))
        log("inset the guard top (blade root)", m.inset_faces(top, 0.01))
        log("flatten the blade root (vertex scaling)", m.scale(top, (1, 0.45, 1)))
        log("extrude the blade", m.extrude_faces(top, 1.25))
        blade = next(iter(m.select_edges(nearest=(0.065, 0.025, 1.3)).edges))
        log("four loop cuts along the blade", m.loop_cut(blade, cuts=4))
        fuller = m.select_faces(
            normal=(FRONT, 1), box=((-1, -1, 0.95), (1, 1, 1.75))
        ) | m.select_faces(normal=((0, 1, 0), 1), box=((-1, -1, 0.95), (1, 1, 1.75)))
        log("inset + sink the fullers (face editing)", m.inset_faces(fuller, 0.012, depth=-0.006))
        log("extrude the tip", m.extrude_faces(top, 0.16))
        tip = m.to_vertices(top)
        log("merge the tip vertices into a point", m.merge_vertices(tip, at="center"))
        report = m.validate()
        assert report.ok, report.to_dict()


def build_hero(scene: Scene):
    hero = scene.add_object("Hero")
    torso = scene.add_object("Torso", parent=hero, position=(0, 0, 1.125))
    with torso.edit_mesh() as m:
        log("torso box", m.add_box((0.5, 0.25, 0.75)))
        belt = next(iter(m.select_edges(nearest=(0.25, 0.125, 0.0)).edges))
        log("loop cut for the belt", m.loop_cut(belt, factor=0.3))

    head = scene.add_object("Head", parent=torso, position=(0, 0, 0.625))
    with head.edit_mesh() as m:
        log("head cube", m.add_box(0.5))
        face = m.select_faces(normal=(FRONT, 1), expect=1)
        log("subdivide the face 3x3", m.subdivide_faces(face, cuts=2))
        eyes = m.select_faces(nearest=(-0.0833, -0.25, 0.0833)) | m.select_faces(
            nearest=(0.0833, -0.25, 0.0833)
        )
        log("inset and sink the eyes", m.inset_faces(eyes, 0.02, depth=-0.03))
        nose = m.select_faces(nearest=(0, -0.25, 0), expect=1)
        log("extrude the nose", m.extrude_faces(nose, 0.05))
        mouth = m.select_faces(nearest=(0, -0.25, -0.1667), expect=1)
        log("sink the mouth", m.inset_faces(mouth, 0.02, depth=-0.015))

    right_arm = scene.add_object("Right Arm", parent=torso, position=(0.375, 0, 0.3))
    with right_arm.edit_mesh() as m:
        log("arm box, pivot at the shoulder", m.add_box((0.25, 0.25, 0.75), center=(0, 0, -0.3)))
        log("soften the arm edges", m.bevel_edges(m.select_all(), 0.015))
    left_arm = scene.add_object(
        "Left Arm", parent=torso, position=(-0.375, 0, 0.3), mesh=right_arm.mesh
    )
    right_leg = scene.add_object("Right Leg", parent=hero, position=(0.125, 0, 0.75))
    with right_leg.edit_mesh() as m:
        log("leg box, pivot at the hip", m.add_box((0.25, 0.25, 0.75), center=(0, 0, -0.375)))
        sole = m.select_faces(normal=((0, 0, -1), 1), expect=1)
        log("inset the sole", m.inset_faces(sole, 0.03, depth=0.02))
    left_leg = scene.add_object(
        "Left Leg", parent=hero, position=(-0.125, 0, 0.75), mesh=right_leg.mesh
    )

    # A sleeve on the right arm only: the default session scope keeps the left arm's
    # (shared) mesh untouched, so instancing is broken only where it was edited.
    with right_arm.edit_mesh() as m:
        # The bevelled corner has two chamfer edges; pick one unambiguously.
        cuff = next(iter(m.select_edges(nearest=(0.125, 0.11, -0.3)).edges))
        r = m.loop_cut(cuff, factor=0.75)
        log("loop cut for a sleeve (right arm only)", r)
    assert right_arm.mesh is not left_arm.mesh and left_leg.mesh is right_leg.mesh

    # Pose (articulation through the hierarchy).
    right_arm.rotation_euler = (-70, 0, 0)
    left_arm.rotation_euler = (20, 0, 0)
    right_leg.rotation_euler = (15, 0, 0)
    left_leg.rotation_euler = (-15, 0, 0)
    head.rotation_euler = (0, 0, -10)
    return right_arm


def khronos(path: Path) -> dict | None:
    if not (shutil.which("node") and (VALIDATOR / "node_modules" / "gltf-validator").is_dir()):
        return None
    out = subprocess.run(
        ["node", str(VALIDATOR / "validate.mjs"), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return json.loads(out.stdout.splitlines()[0])


BLENDER_SCRIPT = """
import bpy, json, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=sys.argv[-1])
out = {}
for o in bpy.context.scene.objects:
    if o.type != "MESH":
        continue
    pts = [o.matrix_world @ v.co for v in o.data.vertices]
    out[o.name] = {
        "polygons": len(o.data.polygons),
        "min": [min(p[i] for p in pts) for i in range(3)],
        "max": [max(p[i] for p in pts) for i in range(3)],
    }
print("AETHER3D_BLENDER " + json.dumps(out))
"""


def blender_command() -> list[str] | None:
    if shutil.which("blender"):
        return ["blender"]
    if shutil.which("flatpak"):
        listed = subprocess.run(
            ["flatpak", "list", "--app", "--columns=application"],
            capture_output=True,
            text=True,
            check=False,
        )
        if "org.blender.Blender" in listed.stdout.split():
            return ["flatpak", "run", f"--filesystem={ROOT}", "org.blender.Blender"]
    return None


def check_in_blender(scene: Scene, glb: Path) -> list[str]:
    command = blender_command()
    if command is None:
        return ["Blender not found"]
    out = subprocess.run(
        [
            *command,
            "--background",
            "--factory-startup",
            "--python-expr",
            BLENDER_SCRIPT,
            "--",
            str(glb.resolve()),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    line = next((x for x in out.stdout.splitlines() if x.startswith("AETHER3D_BLENDER ")), None)
    if line is None:
        return ["Blender did not report: " + out.stderr[-500:]]
    seen = json.loads(line.split(" ", 1)[1])
    problems = []
    for obj in scene:
        if obj.mesh is None:
            continue
        if obj.name not in seen:
            problems.append(f"Blender is missing {obj.name!r}")
            continue
        m = obj.world_matrix
        pts = obj.mesh.positions.astype(float) @ m[:3, :3].T + m[:3, 3]
        if not (
            np.allclose(seen[obj.name]["min"], pts.min(axis=0), atol=1e-4)
            and np.allclose(seen[obj.name]["max"], pts.max(axis=0), atol=1e-4)
        ):
            problems.append(f"{obj.name}: world bounds differ in Blender")
        print(f"   Blender sees {obj.name!r}: {seen[obj.name]['polygons']} polygons, bounds match")
    return problems


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    out_dir = Path(args[0]) if args else Path("minecraft_hero_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = Scene("Blocky Hero")

    print("1. Modelling the character")
    right_arm = build_hero(scene)
    print("2. Modelling the sword")
    build_sword(scene, right_arm)

    print("3. Validating every mesh")
    failures = []
    for obj in scene:
        if obj.mesh is None:
            continue
        with obj.edit_mesh() as edit:
            report = edit.validate()
            edit.discard()
        topo = inspect_mesh(obj.mesh).topology
        ok = report.ok and topo["closed"] and topo["edge_manifold"] and topo["vertex_manifold"]
        ok = ok and topo["consistently_oriented"] and topo["volume"] > 0
        summary = report.summary
        print(
            f"   {obj.name:10s} {summary['face_count']:3d} faces {summary['face_sizes']}  "
            f"closed manifold outward volume={topo['volume']:.5f}"
            if ok
            else f"   {obj.name}: FAILED"
        )
        if not ok:
            failures.append(f"{obj.name} is not a valid closed solid: {report.to_dict()}")

    print("4. Exporting GLB")
    glb = out_dir / "blocky_hero.glb"
    exported = scene.export(glb)
    size = f"{exported.byte_length} bytes"
    print(f"   {glb}: {size}, {exported.node_count} nodes, {exported.mesh_count} meshes")

    print("5. Khronos glTF Validator")
    report = khronos(glb)
    if report is None:
        print("   skipped (not installed)")
        if "--require-validator" in argv:
            failures.append("validator required but not installed")
    else:
        counts = f"{report['errors']} errors, {report['warnings']} warnings"
        print(f"   {report['validatorVersion']}: {counts}")
        if not report["ok"]:
            failures.append("Khronos validator reported problems")

    print("6. Importing the GLB back")
    back = import_gltf(glb)
    for obj in scene:
        other = back.scene.get(obj.id)
        if obj.mesh is not None and not np.array_equal(other.mesh.positions, obj.mesh.positions):
            failures.append(f"{obj.name}: geometry changed in the round trip")
        if not np.array_equal(other.world_matrix, obj.world_matrix):
            failures.append(f"{obj.name}: transform changed in the round trip")
    print(f"   {len(back.scene)} objects, ids restored, geometry and transforms identical")

    if "--blender" in argv:
        print("7. Opening the GLB in Blender (headless)")
        failures += check_in_blender(scene, glb)

    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
