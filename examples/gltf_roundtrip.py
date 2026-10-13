"""Build a scene, export it to GLB, validate it, import it back, and verify it.

Run from the repository root:

    python examples/gltf_roundtrip.py [OUTPUT_DIR] [--require-validator]

Steps:

1. build a multi-object scene (hierarchy, rotations, non-uniform and mirrored
   scale, instanced meshes, a custom mesh);
2. export it to ``OUTPUT_DIR/town.glb`` (default: ``./gltf_roundtrip_output``);
3. validate the file with the official Khronos glTF Validator if Node.js and
   ``tools/gltf-validator`` are installed (``npm ci --prefix tools/gltf-validator``);
   otherwise this step is reported as skipped -- or fails with ``--require-validator``;
4. import the GLB back into a new scene;
5. inspect the geometry of both scenes;
6. verify that everything glTF can carry survived, within these tolerances:
   ids, names, hierarchy, transforms, world matrices, vertex positions, normals
   and triangles are bit-exact; UVs within 2.5e-7 (the v-flip between
   bottom-left and top-left UV origins rounds in float32).

Exits with status 0 if every check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from aether3d import Mesh, Scene, import_gltf

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "tools" / "gltf-validator"
UV_TOLERANCE = 2.5e-7


def build_scene() -> Scene:
    scene = Scene("Little Town")
    scene.add_plane(name="Ground", size=40)

    # Buildings: one shared cube mesh, non-uniformly scaled and rotated.
    block = scene.add_cube(name="Town Hall", size=1, position=(0, 0, 3), scale=(8, 6, 6))
    for i, (x, y, height, turn) in enumerate([(-12, 6, 4, 15), (-12, -6, 7, -10), (12, 8, 10, 30)]):
        house = scene.add_object(
            name=f"House {i + 1}",
            mesh=block.mesh,
            position=(x, y, height / 2),
            scale=(4, 5, height),
        )
        house.rotation_euler = (0, 0, turn)

    # A nested, rotated tower: base -> shaft -> beacon, each with its own transform.
    tower = scene.add_object(name="Tower", position=(10, -10, 0))
    tower.rotation_euler = (0, 0, 45)
    shaft = scene.add_cube(name="Shaft", parent=tower, position=(0, 0, 6), scale=(1, 1, 12))
    beacon = scene.add_sphere(name="Beacon", radius=0.8, parent=shaft, position=(0, 0, 0.55))
    beacon.scale = (1, 1, 1 / 12)  # undo the shaft's stretch so the beacon stays round
    antenna = scene.add_cube(name="Antenna", size=0.2, parent=beacon, position=(0, 0, 1.2))
    antenna.rotation_euler = (20, -35, 10)

    # Instanced trees: two shared meshes reused by every tree.
    trunk_mesh = scene.add_cube(name="Tree 1 Trunk", size=0.5, position=(-4, 12, 1)).mesh
    scene.get_by_name("Tree 1 Trunk").scale = (1, 1, 4)
    crown_mesh = scene.add_sphere(
        name="Tree 1 Crown", radius=1.5, segments=16, rings=8, position=(-4, 12, 3.5)
    ).mesh
    for i, (x, y) in enumerate([(0, 12), (4, 12), (8, 14)], start=2):
        tree = scene.add_object(name=f"Tree {i}", position=(x, y, 0))
        tree.rotation_euler = (0, 0, 30 * i)
        scene.add_object(
            name=f"Tree {i} Trunk",
            mesh=trunk_mesh,
            parent=tree,
            position=(0, 0, 1),
            scale=(1, 1, 4),
        )
        scene.add_object(name=f"Tree {i} Crown", mesh=crown_mesh, parent=tree, position=(0, 0, 3.5))

    # A custom closed mesh (a tetrahedron) placed with a mirrored scale.
    tetra = Mesh(
        positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        triangles=[(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)],
    )
    scene.add_object(name="Sculpture", mesh=tetra, position=(-3, -8, 0), scale=(-2, 2, 3))
    return scene


def run_khronos_validator(path: Path) -> dict | None:
    if shutil.which("node") is None or not (VALIDATOR / "node_modules" / "gltf-validator").is_dir():
        return None
    result = subprocess.run(
        ["node", str(VALIDATOR / "validate.mjs"), str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return json.loads(result.stdout.splitlines()[0])


def world_points(obj) -> np.ndarray:
    m = obj.world_matrix
    return obj.mesh.positions.astype(np.float64) @ m[:3, :3].T + m[:3, 3]


def verify(original: Scene, imported: Scene) -> list[str]:
    """Return a list of human-readable failures (empty if everything matches)."""
    failures = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(imported.name == original.name, "scene name changed")
    check([o.id for o in imported] == [o.id for o in original], "object ids/order changed")
    for a in original:
        if a.id not in imported:
            failures.append(f"{a.name}: missing after import")
            continue
        b = imported.get(a.id)
        check(b.name == a.name, f"{a.name}: name changed to {b.name}")
        check((b.parent and b.parent.id) == (a.parent and a.parent.id), f"{a.name}: parent changed")
        check(b.transform == a.transform, f"{a.name}: local transform changed")
        check(np.array_equal(b.world_matrix, a.world_matrix), f"{a.name}: world matrix changed")
        if a.mesh is None:
            check(b.mesh is None, f"{a.name}: gained a mesh")
            continue
        check(np.array_equal(world_points(b), world_points(a)), f"{a.name}: world vertices moved")
        check(np.array_equal(b.mesh.triangles, a.mesh.triangles), f"{a.name}: triangles changed")
        if a.mesh.normals is not None:
            check(np.array_equal(b.mesh.normals, a.mesh.normals), f"{a.name}: normals changed")
        if a.mesh.uvs is not None:
            error = float(np.abs(b.mesh.uvs - a.mesh.uvs).max())
            check(error <= UV_TOLERANCE, f"{a.name}: UV error {error} > {UV_TOLERANCE}")

    # Instancing: objects that shared a mesh still share one.
    for a in original:
        for c in original:
            if a.mesh is not None and a.mesh is c.mesh:
                check(
                    imported.get(a.id).mesh is imported.get(c.id).mesh,
                    f"{a.name}/{c.name}: no longer share a mesh",
                )

    # Geometry analysis agrees (bounds, areas, topology, volumes, diagnostics).
    before = original.inspect_geometry().to_dict()
    after = imported.inspect_geometry().to_dict()
    for key in ("status", "summary", "world_bounds", "objects", "diagnostics"):
        check(after[key] == before[key], f"geometry report differs in {key!r}")
    for m_before, m_after in zip(before["meshes"], after["meshes"], strict=True):
        for key in ("vertex_count", "triangle_count", "bounds", "surface_area", "topology"):
            check(m_after[key] == m_before[key], f"mesh {m_before['index']}: {key} differs")
    return failures


def main(argv: list[str]) -> int:
    require_validator = "--require-validator" in argv
    args = [a for a in argv if not a.startswith("--")]
    out_dir = Path(args[0]) if args else Path("gltf_roundtrip_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. Building the scene")
    scene = build_scene()
    print(f"   {len(scene)} objects, {len(scene.roots)} roots")

    print("2. Exporting to GLB")
    glb = out_dir / "town.glb"
    export = scene.export(glb)
    print(
        f"   {export.path.name}: {export.byte_length} bytes, {export.node_count} nodes, "
        f"{export.mesh_count} meshes, {len(export.diagnostics)} export diagnostics"
    )

    print("3. Validating with the Khronos glTF Validator")
    report = run_khronos_validator(glb)
    if report is None:
        print("   skipped: Node.js or tools/gltf-validator is not installed")
        if require_validator:
            print("FAILED: --require-validator was given")
            return 1
    else:
        print(
            f"   validator {report['validatorVersion']}: {report['errors']} errors, "
            f"{report['warnings']} warnings, {report['infos']} infos"
        )
        if not report["ok"]:
            print("FAILED: the validator reported problems:", *report["messages"], sep="\n   ")
            return 1

    print("4. Importing the GLB back")
    result = import_gltf(glb)
    imported = result.scene
    restored = sum(1 for o in result.objects if o["id_source"] == "extras")
    print(
        f"   {len(imported)} objects ({restored} ids restored from extras), "
        f"{len(result.diagnostics)} import diagnostics"
    )

    print("5. Inspecting geometry")
    geometry = imported.inspect_geometry().to_dict()
    summary = geometry["summary"]
    print(
        f"   status {geometry['status']}: {summary['unique_mesh_count']} unique meshes, "
        f"{summary['unique_triangle_count']} unique / {summary['instanced_triangle_count']} "
        "instanced triangles"
    )
    for mesh in geometry["meshes"]:
        topo = mesh["topology"]
        volume = f", volume {topo['volume']:.4f}" if topo["volume"] is not None else ""
        print(
            f"   mesh {mesh['index']}: {mesh['triangle_count']} triangles, "
            f"{'closed' if topo['closed'] else 'open'}{volume}, used by {len(mesh['used_by'])}"
        )
    lo, hi = geometry["world_bounds"]["min"], geometry["world_bounds"]["max"]
    print(f"   world bounds {tuple(round(v, 4) for v in lo)} .. {tuple(round(v, 4) for v in hi)}")

    print("6. Verifying the round trip")
    failures = verify(scene, imported)
    if failures:
        print("FAILED:", *failures, sep="\n   ")
        return 1
    print("   all checks passed: ids, names, hierarchy, transforms, world matrices, vertices,")
    print("   normals and triangles are bit-exact; UVs within", UV_TOLERANCE)
    print("   instancing and the geometry analysis are unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
