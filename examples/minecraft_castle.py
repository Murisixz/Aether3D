"""A Minecraft-inspired castle built entirely from procedural, non-destructive geometry.

Nothing here is modelled by hand: every mesh is the evaluated result of a
procedural geometry node -- primitives plus modifier stacks (transform, array,
mirror, solidify, subdivision, bevel and exact booleans) driven by scene
parameters and expressions. Change a parameter and everything that depends on
it regenerates; the base definitions are never modified.

Run from the repository root (booleans need ``pip install 'aether3d[boolean]'``):

    python examples/minecraft_castle.py [OUTPUT_DIR] [--require-validator] [--blender]

Steps: build the castle, validate every evaluated mesh (closed, manifold,
outward-facing solids), check its structure (tower count, gate opening,
bounds), save the ``.aether`` project, reload it and compare, modify parameters
(and check that invalid combinations are rejected without changing anything),
export ``OUTPUT_DIR/castle.glb`` (default ``./minecraft_castle_output``),
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

from aether3d import Expr, ProceduralError, Scene, import_gltf, inspect_mesh

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "tools" / "gltf-validator"

PARAMETERS = {
    # name: (default, options)
    "castle_width": (
        24.0,
        {
            "min": 12.0,
            "max": 80.0,
            "unit": "m",
            "description": "outer size of the square curtain wall",
        },
    ),
    "wall_height": (6.0, {"min": 3.0, "max": 20.0, "unit": "m"}),
    "wall_thickness": (1.2, {"min": 0.5, "max": 3.0, "unit": "m"}),
    "tower_count": (6, {"min": 4, "max": 16, "description": "towers on the side walls (even)"}),
    "tower_radius": (2.2, {"min": 1.0, "max": 6.0, "unit": "m"}),
    "gate_width": (3.0, {"min": 1.0, "max": 10.0, "unit": "m"}),
    "gate_height": (4.0, {"min": 1.5, "max": 16.0, "unit": "m"}),
    "detail_density": (2, {"min": 0, "max": 3, "description": "battlements, windows, segments"}),
}

RULES = [
    ("even_towers", "tower_count % 2 == 0", "tower_count must be even (two rows of towers)"),
    (
        "towers_fit",
        "2 * tower_radius + 1 <= castle_width / (tower_count / 2 - 1)",
        "towers would overlap: reduce tower_radius or tower_count, or widen the castle",
    ),
    (
        "gate_fits_width",
        "gate_width + 2 * tower_radius + 2 <= castle_width",
        "the gate is too wide for the front wall",
    ),
    ("gate_fits_height", "gate_height + 1 <= wall_height", "the gate must be lower than the wall"),
    ("arch_fits", "gate_width / 2 < gate_height", "the gate arch is taller than the gate"),
]


def build_castle(scene: Scene) -> None:
    """Define every parameter, geometry node and object (one atomic transaction)."""
    proc = scene.procedural
    with proc.batch():
        for name, (value, options) in PARAMETERS.items():
            proc.add_parameter(name, value, **options)
        # Derived values are parameters too: expressions over other parameters.
        proc.add_parameter("towers_per_side", Expr("tower_count // 2"), type="int", min=2)
        proc.add_parameter("tower_height", Expr("wall_height * 1.4"), type="float", min=1)
        proc.add_parameter(
            "merlons_per_wall",
            Expr("max(3, floor(castle_width * (detail_density + 1) / 6))"),
            type="int",
            min=3,
        )
        for name, expr, message in RULES:
            proc.add_rule(name, expr, message)

        # Ground: a single quad given thickness (top surface at z = 0).
        ground = scene.add_geometry(
            "Ground", "grid", x_size=Expr("castle_width * 1.6"), y_size=Expr("castle_width * 1.6")
        )
        ground.add_modifier("solidify", thickness=1.0, offset=-1.0)

        # Curtain wall: front/back and left/right wall pairs, mirrored, unioned, gate cut out.
        walls = scene.add_geometry(
            "Curtain Wall",
            "box",
            width=Expr("castle_width + wall_thickness"),
            depth=Expr("wall_thickness"),
            height=Expr("wall_height"),
        )
        walls.add_modifier("transform", translation=Expr("(0, castle_width / 2, wall_height / 2)"))
        walls.add_modifier("mirror", axes="y")
        side_walls = scene.add_geometry(
            "Side Walls",
            "box",
            width=Expr("wall_thickness"),
            depth=Expr("castle_width + wall_thickness"),
            height=Expr("wall_height"),
        )
        side_walls.add_modifier(
            "transform", translation=Expr("(castle_width / 2, 0, wall_height / 2)")
        )
        side_walls.add_modifier("mirror", axes="x")

        # Battlements: one merlon, arrayed along a wall, mirrored to the opposite wall.
        merlons_x = scene.add_geometry(
            "Merlons X",
            "box",
            width=Expr("castle_width / merlons_per_wall / 2"),
            depth=Expr("wall_thickness"),
            height=0.9,
        )
        merlons_x.add_modifier(
            "transform",
            translation=Expr(
                "(-castle_width / 2 + castle_width / merlons_per_wall / 2,"
                " castle_width / 2, wall_height + 0.4)"
            ),
        )
        merlons_x.add_modifier(
            "array",
            count=Expr("merlons_per_wall"),
            relative_offset=(0, 0, 0),
            constant_offset=Expr("(castle_width / merlons_per_wall, 0, 0)"),
        )
        merlons_x.add_modifier("mirror", axes="y")
        merlons_y = scene.add_geometry(
            "Merlons Y", source=merlons_x, after="Mirror"
        )  # the same battlement row...
        merlons_y.add_modifier("transform", rotation=(0, 0, 90))  # ...turned to the side walls

        walls.add_modifier(
            "boolean", name="Join Side Walls", operation="union", operand={"geometry": "Side Walls"}
        )
        walls.add_modifier(
            "boolean", name="Battlements X", operation="union", operand={"geometry": "Merlons X"}
        )
        walls.add_modifier(
            "boolean", name="Battlements Y", operation="union", operand={"geometry": "Merlons Y"}
        )

        # Gate: a box plus a half-cylinder arch, cut through the front wall.
        arch = scene.add_geometry(
            "Gate Arch",
            "cylinder",
            radius=Expr("gate_width / 2"),
            depth=Expr("wall_thickness * 3"),
            segments=16,
            smooth=False,
        )
        arch.add_modifier(
            "transform",
            rotation=(90, 0, 0),
            translation=Expr("(0, -castle_width / 2, gate_height - gate_width / 2)"),
        )
        gate = scene.add_geometry(
            "Gate Cutter",
            "box",
            width=Expr("gate_width"),
            depth=Expr("wall_thickness * 3"),
            height=Expr("gate_height - gate_width / 2 + 1"),
        )
        gate.add_modifier(
            "transform",
            translation=Expr("(0, -castle_width / 2, (gate_height - gate_width / 2 + 1) / 2 - 1)"),
        )
        gate.add_modifier("boolean", operation="union", operand={"geometry": "Gate Arch"})
        walls.add_modifier(
            "boolean", name="Cut Gate", operation="difference", operand={"geometry": "Gate Cutter"}
        )

        # Tower: an octagonal prism with arrow slits cut all around and a crenellated crown.
        tower = scene.add_geometry(
            "Tower",
            "cylinder",
            radius=Expr("tower_radius"),
            depth=Expr("tower_height"),
            segments=Expr("8 * (1 + detail_density // 2)"),
            smooth=False,
            parameters={"height": {"value": Expr("tower_height"), "type": "float", "min": 1}},
        )
        tower.set_source(depth=Expr("height"))
        tower.add_modifier("transform", translation=Expr("(0, 0, height / 2)"))
        slit = scene.add_geometry(
            "Arrow Slit", "box", width=0.35, depth=Expr("tower_radius * 0.8"), height=1.2
        )
        slit.add_modifier("transform", translation=Expr("(0, tower_radius, wall_height * 0.9)"))
        slit.add_modifier(
            "array",
            count=Expr("detail_density + 2"),
            relative_offset=(0, 0, 0),
            rotation=Expr("(0, 0, 360 / (detail_density + 2))"),
        )
        tower.add_modifier(
            "boolean",
            name="Arrow Slits",
            operation="difference",
            operand={"geometry": "Arrow Slit"},
        )
        # Merlons are sized from the radius so neighbours never overlap (booleans need
        # operands whose pieces do not intersect each other).
        crown = scene.add_geometry(
            "Crown Merlon",
            "box",
            width=Expr("tower_radius * 0.25"),
            depth=Expr("tower_radius * 0.25"),
            height=1.0,
        )
        crown.add_modifier("transform", translation=Expr("(tower_radius * 0.82, 0, 0.4)"))
        crown.add_modifier(
            "array",
            count=Expr("8 * (1 + detail_density // 2)"),
            relative_offset=(0, 0, 0),
            rotation=Expr("(0, 0, 45 / (1 + detail_density // 2))"),
        )
        tower.add_modifier(
            "boolean",
            name="Crown",
            operation="union",
            operand={"geometry": "Crown Merlon"},
            operand_translation=Expr("(0, 0, height)"),
        )

        # Towers: the tower repeated along one side wall, then mirrored to the other.
        towers = scene.add_geometry("Towers", source=tower)
        towers.add_modifier(
            "transform", translation=Expr("(castle_width / 2, -castle_width / 2, 0)")
        )
        towers.add_modifier(
            "array",
            count=Expr("towers_per_side"),
            relative_offset=(0, 0, 0),
            constant_offset=Expr("(0, castle_width / (towers_per_side - 1), 0)"),
        )
        towers.add_modifier("mirror", axes="x")

        # Keep: a bevelled block; the watchtower on top reuses Tower with overrides and a roof.
        keep = scene.add_geometry(
            "Keep",
            "box",
            width=Expr("castle_width * 0.3"),
            depth=Expr("castle_width * 0.3"),
            height=Expr("wall_height * 1.5"),
        )
        keep.add_modifier(
            "transform", translation=Expr("(0, castle_width * 0.15, wall_height * 0.75)")
        )
        keep.add_modifier("bevel", width=0.3, limit="all")
        roof = scene.add_geometry(
            "Roof",
            "cone",
            radius_bottom=Expr("tower_radius * 1.25"),
            radius_top=0,
            depth=Expr("tower_radius * 1.8"),
            segments=Expr("8 * (1 + detail_density // 2)"),
            smooth=False,
        )
        watchtower = scene.add_geometry(
            "Watchtower", source=tower, overrides={"height": Expr("wall_height * 1.1")}
        )
        watchtower.add_modifier(
            "boolean",
            name="Roof",
            operation="union",
            operand={"geometry": roof},
            operand_translation=Expr("(0, 0, wall_height * 1.1 + tower_radius * 0.9 + 0.5)"),
        )

        # Bushes: a subdivided block, shared by several objects (one mesh, many instances).
        bush = scene.add_geometry("Bush", "box", width=1.6, depth=1.6, height=1.2)
        bush.add_modifier("subdivision", levels=Expr("min(2, detail_density)"))
        bush.add_modifier("transform", translation=(0, 0, 0.5))

        scene.add_object("Ground", geometry=ground)
        scene.add_object("Curtain Wall", geometry=walls)
        scene.add_object("Towers", geometry=towers)
        keep_obj = scene.add_object("Keep", geometry=keep)
        scene.add_object("Watchtower", geometry=watchtower, parent=keep_obj)
        for i, (x, y) in enumerate([(-6, -6), (6, -6), (-7, 2), (7, 2)]):
            scene.add_object(f"Bush {i + 1}", geometry=bush, position=(x, y, 0))
    place_watchtower(scene)


def place_watchtower(scene: Scene) -> None:
    """Objects are placed by transforms (not parameters); keep the watchtower on the keep."""
    values = scene.procedural.info()["parameters"]
    width = values["castle_width"]["resolved"]
    height = values["wall_height"]["resolved"]
    scene.get_by_name("Watchtower").position = (0, width * 0.15, height * 1.5)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def components(mesh) -> int:
    """Number of connected pieces (vertex-connected triangles)."""
    parent = np.arange(mesh.vertex_count)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    positions = np.round(mesh.positions.astype(np.float64), 5)
    _, weld = np.unique(positions, axis=0, return_inverse=True)
    for a, b, c in weld.reshape(-1)[mesh.triangles]:
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    used = np.unique(weld.reshape(-1)[mesh.triangles])
    return len({find(int(v)) for v in used})


def winding_number(mesh, point) -> float:
    """Generalized winding number: ~1 inside a closed outward mesh, ~0 outside."""
    tri = mesh.positions.astype(np.float64)[mesh.triangles] - np.asarray(point, dtype=np.float64)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    den = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return float(np.sum(2 * np.arctan2(det, den)) / (4 * np.pi))


def check_solids(scene: Scene) -> list[str]:
    failures = []
    for obj in scene:
        if obj.mesh is None:
            continue
        topo = inspect_mesh(obj.mesh).topology
        ok = topo["closed"] and topo["edge_manifold"] and topo["vertex_manifold"]
        ok = ok and topo["consistently_oriented"] and (topo["volume"] or 0) > 0
        mesh = obj.mesh
        status = f"closed solid, volume {topo['volume']:.3f}" if ok else "NOT A VALID SOLID"
        print(f"   {obj.name:13s} {mesh.vertex_count:6d}v {mesh.triangle_count:6d}t  {status}")
        if not ok:
            failures.append(f"{obj.name} is not a closed, manifold, outward solid: {topo}")
    return failures


def check_structure(scene: Scene) -> list[str]:
    values = {k: v["resolved"] for k, v in scene.procedural.info()["parameters"].items()}
    width, wall_h, thick = values["castle_width"], values["wall_height"], values["wall_thickness"]
    failures = []
    towers = components(scene.get_by_name("Towers").mesh)
    print(f"   towers: {towers} (tower_count = {values['tower_count']})")
    if towers != values["tower_count"]:
        failures.append(f"expected {values['tower_count']} towers, found {towers}")
    walls = scene.get_by_name("Curtain Wall").mesh
    gate_point = (0.0, -width / 2, min(1.0, values["gate_height"] / 2))
    wall_point = (width / 4, -width / 2, wall_h / 2)
    inside_gate = winding_number(walls, gate_point)
    inside_wall = winding_number(walls, wall_point)
    print(f"   gate opening: winding {inside_gate:.3f} (open); wall: {inside_wall:.3f} (solid)")
    if abs(inside_gate) > 1e-3 or abs(inside_wall - 1) > 1e-3:
        failures.append("the gate is not cut through the front wall")
    lo, hi = walls.positions.min(axis=0), walls.positions.max(axis=0)
    half = (width + thick) / 2
    if not (np.allclose(lo[:2], -half, atol=1e-4) and np.allclose(hi[:2], half, atol=1e-4)):
        failures.append(f"curtain wall bounds {lo}..{hi} do not match castle_width")
    if not np.isclose(hi[2], wall_h + 0.85, atol=1e-4):
        failures.append(f"battlements top {hi[2]} != wall_height + 0.85")
    print(f"   curtain wall bounds {np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}")
    return failures


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
    if command[0] == "flatpak":  # the sandbox must be allowed to read the output directory
        command.insert(2, f"--filesystem={glb.parent.resolve()}")
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


def snapshot(scene: Scene) -> dict[str, tuple]:
    return {
        o.name: (o.mesh.positions.copy(), o.mesh.triangles.copy())
        for o in scene
        if o.mesh is not None
    }


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1]) for k in a
    )


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    out_dir = Path(args[0]) if args else Path("minecraft_castle_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    scene = Scene("Procedural Castle")

    print("1. Defining the procedural castle")
    build_castle(scene)
    proc = scene.procedural
    print(
        f"   {len(proc.parameters)} parameters, {len(proc.rules)} rules, "
        f"{len(proc.geometries)} geometry nodes, {len(scene)} objects"
    )
    for g in proc.geometries:
        stack = " -> ".join(m.name for m in g.modifiers) or "(no modifiers)"
        print(f"   {g.name:13s} {stack}")

    print("2. Validating every evaluated mesh")
    failures += check_solids(scene)
    report = proc.validate()
    warnings = sorted(
        {
            f"{e['geometry']['name']}: {d['code']}"
            for e in report["geometry"]
            for d in e.get("diagnostics", [])
            if d["severity"] == "warning"
        }
    )
    print(f"   procedural validation: {report['status']} {warnings or ''}")
    if report["status"] == "error":
        failures.append("procedural validation reported errors")

    print("3. Checking the structure")
    failures += check_structure(scene)

    print("4. Saving, reloading and comparing")
    project = out_dir / "castle.aether"
    scene.save(project)
    original = snapshot(scene)
    reloaded = Scene.load(project)
    if not same(original, snapshot(reloaded)):
        failures.append("the reloaded project evaluates to different geometry")
    print(f"   {project}: {project.stat().st_size} bytes; reloaded geometry identical")

    print("5. Modifying parameters")
    before = proc.cache_info
    with proc.batch():
        proc.set_parameter("castle_width", 32.0)
        proc.set_parameter("tower_count", 8)
        proc.set_parameter("detail_density", 1)
    place_watchtower(scene)
    after = proc.cache_info
    print(
        f"   castle_width 32, tower_count 8, detail_density 1: regenerated "
        f"{after['misses'] - before['misses']} steps, reused {after['hits'] - before['hits']}"
    )
    failures += check_solids(scene)
    failures += check_structure(scene)
    for name, value, code in (
        ("tower_count", 7, "CONSTRAINT_VIOLATED"),
        ("gate_height", 9.0, "CONSTRAINT_VIOLATED"),
        ("tower_radius", 9.0, "PARAMETER_OUT_OF_RANGE"),
    ):
        revision = proc.revision
        try:
            proc.set_parameter(name, value)
            failures.append(f"{name}={value} was accepted")
        except ProceduralError as exc:
            ok = exc.code == code and proc.revision == revision
            print(f"   {name}={value} rejected: [{exc.code}] {exc.message}")
            if not ok:
                failures.append(f"{name}={value}: expected {code}, state unchanged")
    reloaded.procedural.set_parameters(
        {"castle_width": 32.0, "tower_count": 8, "detail_density": 1}
    )
    place_watchtower(reloaded)
    if not same(snapshot(scene), snapshot(reloaded)):
        failures.append("the reloaded project regenerates differently")
    print("   the reloaded project regenerates identically")

    print("6. Exporting GLB")
    glb = out_dir / "castle.glb"
    exported = scene.export(glb)
    print(
        f"   {glb}: {exported.byte_length} bytes, {exported.node_count} nodes, "
        f"{exported.mesh_count} meshes (bushes share one)"
    )
    scene.save(project)

    print("7. Khronos glTF Validator")
    report = khronos(glb)
    if report is None:
        print("   skipped (not installed)")
        if "--require-validator" in argv:
            failures.append("validator required but not installed")
    else:
        print(
            f"   {report['validatorVersion']}: {report['errors']} errors, "
            f"{report['warnings']} warnings"
        )
        if not report["ok"] or report["warnings"]:
            failures.append(f"Khronos validator reported problems: {report['messages']}")

    print("8. Importing the GLB back")
    back = import_gltf(glb)
    for obj in scene:
        other = back.scene.get(obj.id)
        if obj.mesh is not None and not np.array_equal(other.mesh.positions, obj.mesh.positions):
            failures.append(f"{obj.name}: geometry changed in the round trip")
        if not np.allclose(other.world_matrix, obj.world_matrix, atol=1e-6):
            failures.append(f"{obj.name}: transform changed in the round trip")
    print(f"   {len(back.scene)} objects, geometry and transforms identical")

    print("9. Undo")
    proc.undo()
    place_watchtower(scene)
    if not same(original, snapshot(scene)):
        failures.append("undo did not restore the original castle")
    print("   undo restored the original castle exactly")

    if "--blender" in argv:
        print("10. Opening the GLB in Blender (headless)")
        failures += check_in_blender(back.scene, glb)  # what was exported (before undo)

    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
