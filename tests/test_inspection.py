import json

import numpy as np
import pytest
from scenes import challenging_scene

from aether3d import (
    Mesh,
    Scene,
    create_cube,
    create_plane,
    create_sphere,
    inspect_mesh,
    inspect_scene,
)

TRI = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]


def codes(report):
    return {d.code: d for d in report.diagnostics}


def test_cube_is_closed_with_exact_area_and_volume():
    report = inspect_mesh(create_cube(2))
    assert report.status == "ok" and report.diagnostics == []
    assert report.vertex_count == 24 and report.triangle_count == 12
    assert report.attributes == {"positions": True, "normals": True, "uvs": True}
    assert report.bounds == {
        "min": [-1.0, -1.0, -1.0],
        "max": [1.0, 1.0, 1.0],
        "size": [2.0, 2.0, 2.0],
        "center": [0.0, 0.0, 0.0],
    }
    assert report.surface_area == pytest.approx(24.0)
    topo = report.topology
    assert topo["welded_vertex_count"] == 8 and topo["edge_count"] == 18
    assert topo["closed"] and topo["edge_manifold"] and topo["consistently_oriented"]
    assert topo["volume"] == pytest.approx(8.0)


def test_sphere_is_closed_and_approaches_analytic_values():
    report = inspect_mesh(create_sphere(radius=2, segments=128, rings=64))
    assert report.topology["closed"] and report.topology["consistently_oriented"]
    assert report.topology["volume"] == pytest.approx(4 / 3 * np.pi * 8, rel=2e-3)
    assert report.surface_area == pytest.approx(4 * np.pi * 4, rel=2e-3)
    # The UV sphere has two unused duplicate pole vertices; reported, not hidden.
    assert codes(report)["UNREFERENCED_VERTICES"].count == 2
    assert report.status == "ok"  # info findings do not change the status


def test_plane_is_open():
    report = inspect_mesh(create_plane(3))
    assert not report.topology["closed"]
    assert report.topology["volume"] is None
    assert codes(report)["OPEN_BOUNDARY"].count == 4
    assert report.surface_area == pytest.approx(9.0)


def test_degenerate_and_duplicate_triangles():
    mesh = Mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0)],
        [(0, 1, 2), (0, 0, 1), (0, 1, 3), (2, 1, 0), (0, 1, 2)],
    )
    found = codes(inspect_mesh(mesh, location="meshes[7]"))
    assert found["DEGENERATE_TRIANGLES_REPEATED_INDEX"].samples == (1,)
    assert found["DEGENERATE_TRIANGLES_ZERO_AREA"].samples == (2,)  # collinear points
    assert found["DUPLICATE_TRIANGLES"].samples == (3, 4)
    assert all(d.location == "meshes[7]" for d in found.values())


def test_normal_checks():
    mesh = Mesh(TRI, [(0, 1, 2)], normals=[(0, 0, -1), (0, 0, 0), (0, 0, -3)])
    found = codes(inspect_mesh(mesh))
    assert found["ZERO_LENGTH_NORMALS"].samples == (1,)
    assert found["NON_UNIT_NORMALS"].samples == (2,)
    assert found["NORMALS_OPPOSE_WINDING"].samples == (0,)
    clean = inspect_mesh(Mesh(TRI, [(0, 1, 2)], normals=[(0, 0, 1)] * 3))
    assert {d.code for d in clean.diagnostics} == {"OPEN_BOUNDARY"}  # a lone triangle is open


def test_inside_out_and_inconsistent_orientation():
    cube = create_cube()
    flipped = Mesh(cube.positions, cube.triangles[:, ::-1], normals=-cube.normals)
    report = inspect_mesh(flipped)
    assert report.topology["volume"] == pytest.approx(-1.0)
    assert "INSIDE_OUT" in codes(report)

    tris = cube.triangles.copy()
    tris[0] = tris[0][::-1]
    report = inspect_mesh(Mesh(cube.positions, tris))
    assert not report.topology["consistently_oriented"]
    assert report.topology["volume"] is None  # never reported for inconsistent meshes
    assert codes(report)["INCONSISTENT_ORIENTATION"].count == 3


def test_non_manifold_edges():
    mesh = Mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)],
        [(0, 1, 2), (1, 0, 3), (0, 1, 4)],
    )
    report = inspect_mesh(mesh)
    assert not report.topology["edge_manifold"]
    assert codes(report)["NON_MANIFOLD_EDGES"].count == 1


def test_welding_is_exact_and_treats_negative_zero_as_zero():
    a = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (-0.0, 0, 0), (1, 0, 0), (0, -1, 0)],
             [(0, 1, 2), (3, 5, 4)])  # fmt: skip
    assert inspect_mesh(a).topology["welded_vertex_count"] == 4
    b = Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1e-7, 0, 0)], [(0, 1, 2), (3, 1, 2)])
    assert inspect_mesh(b).topology["welded_vertex_count"] == 4  # no tolerance applied


def test_empty_mesh():
    report = inspect_mesh(Mesh([], []))
    assert report.bounds is None and report.surface_area == 0.0
    assert set(codes(report)) == {"EMPTY_MESH"}
    assert report.status == "ok"


def test_report_is_json_serializable_with_stable_keys():
    data = json.loads(json.dumps(inspect_mesh(create_sphere(segments=8, rings=4)).to_dict()))
    assert list(data) == [
        "status",
        "vertex_count",
        "triangle_count",
        "attributes",
        "bounds",
        "surface_area",
        "uv_bounds",
        "topology",
        "diagnostics",
    ]
    assert set(data["diagnostics"][0]) == {
        "code",
        "severity",
        "message",
        "location",
        "count",
        "samples",
        "lossy",
    }


def test_inspect_mesh_rejects_non_meshes():
    with pytest.raises(TypeError):
        inspect_mesh("cube")


# --------------------------------------------------------------------------- #
# Scenes
# --------------------------------------------------------------------------- #


def test_scene_report_counts_instances_and_unique_meshes():
    scene = challenging_scene()
    report = scene.inspect_geometry()
    assert report.scene_name == "Challenging"
    s = report.summary
    assert s["object_count"] == len(scene)
    assert s["unique_mesh_count"] == 5
    assert s["mesh_object_count"] == 7
    assert s["instanced_triangle_count"] > s["unique_triangle_count"]
    data = report.to_dict()
    moon = scene.get_by_name("Moon")
    entry = next(o for o in data["objects"] if o["id"] == moon.id)
    assert data["meshes"][entry["mesh"]]["used_by"] == [
        moon.id,
        scene.get_by_name("Mirrored Moon").id,
    ]
    json.dumps(data)


def test_world_bounds_and_area_use_world_transforms():
    scene = Scene()
    scene.add_cube("Big", size=2, position=(10, 0, 0), scale=(2, 1, 1))
    report = inspect_scene(scene)
    entry = report.objects[0]
    assert entry["world_bounds"]["min"] == [8.0, -1.0, -1.0]
    assert entry["world_bounds"]["max"] == [12.0, 1.0, 1.0]
    # A 4 x 2 x 2 box: 2 * (4*2) + 2 * (4*2) + 2 * (2*2)
    assert entry["world_surface_area"] == pytest.approx(40.0)
    assert report.world_bounds["min"] == [8.0, -1.0, -1.0]


def test_transform_diagnostics():
    scene = challenging_scene()
    found = {(d.code, d.location) for d in scene.inspect_geometry().diagnostics}
    mirrored = scene.get_by_name("Mirrored Moon").id
    assert ("MIRRORED_WORLD_TRANSFORM", f"objects[{mirrored}]") in found
    hand = scene.get_by_name("Hand").id  # non-uniform Arm scale under a rotated Cube
    assert ("SHEARED_WORLD_TRANSFORM", f"objects[{hand}]") in found

    flat = Scene()
    flat.add_cube("Flat", scale=(1, 0, 1))
    report = flat.inspect_geometry()
    assert [d.code for d in report.diagnostics] == ["ZERO_WORLD_SCALE"]
    assert report.status == "warning"


def test_empty_scene_report():
    data = Scene("Void").inspect_geometry().to_dict()
    assert data["status"] == "ok" and data["world_bounds"] is None
    assert data["summary"]["object_count"] == 0 and data["meshes"] == []


def test_inspection_does_not_modify_the_scene():
    scene = challenging_scene()
    before = scene.to_dict()
    scene.inspect_geometry()
    assert scene.to_dict() == before
