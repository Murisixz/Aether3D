"""Face groups (per-face materials) survive editing, edit scripts and procedural modifiers."""

import importlib.util

import numpy as np
import pytest

from aether3d import EditableMesh, Mesh, Scene, create_cube, run_edit_script
from aether3d.procedural.polymesh import PolyMesh

HAS_MANIFOLD = importlib.util.find_spec("manifold3d") is not None


def grouped_cube(size: float = 2.0) -> Mesh:
    """A cube whose six faces are face groups 0-5 (two triangles each)."""
    cube = create_cube(size)
    return Mesh(
        cube.positions, cube.triangles, cube.normals, cube.uvs, groups=np.repeat(np.arange(6), 2)
    )


def group_areas(mesh: Mesh) -> dict[int, float]:
    p = mesh.positions.astype(np.float64)
    t = mesh.triangles
    area = 0.5 * np.linalg.norm(np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]]), axis=1)
    groups = mesh.groups if mesh.groups is not None else np.zeros(len(t), int)
    return {int(g): float(area[groups == g].sum()) for g in np.unique(groups)}


def test_editable_meshes_round_trip_groups_and_only_join_quads_within_a_group():
    mesh = grouped_cube()
    em = EditableMesh.from_mesh(mesh)
    assert em.has_groups and em.face_count == 6  # triangle pairs joined into quads per group
    assert sorted(em.face_group(f) for f in em.face_ids()) == [0, 1, 2, 3, 4, 5]
    back = em.to_mesh()
    assert group_areas(back) == pytest.approx(group_areas(mesh))
    mixed = Mesh(mesh.positions, mesh.triangles, mesh.normals, mesh.uvs, groups=np.arange(12) % 2)
    em_mixed = EditableMesh.from_mesh(mixed)
    assert em_mixed.face_count == 12  # the two triangles of each side differ: no quads
    plain = EditableMesh.from_mesh(create_cube())
    assert not plain.has_groups and plain.to_mesh().groups is None


def test_editing_operations_give_new_faces_their_source_group():
    em = EditableMesh.from_mesh(grouped_cube())
    top = em.select("faces", normal=((0, 0, 1), 1), expect=1)
    top_group = em.face_group(min(top.faces))
    result = em.extrude_faces(top, distance=0.5)
    created = result.created["faces"]
    assert created and all(em.face_group(f) == top_group for f in created)
    em.inset_faces(top, thickness=0.1)
    em.subdivide_faces(em.select("faces", normal=((1, 0, 0), 1)), cuts=2)
    side = min(em.select("faces", normal=((0, -1, 0), 1)).faces)
    side_group = em.face_group(side)
    loop = em.face_vertices(side)
    em.split_face(side, loop[0], loop[2])
    em.triangulate_faces()
    mesh = em.to_mesh()
    areas = group_areas(mesh)
    assert set(areas) == {0, 1, 2, 3, 4, 5} and areas[side_group] == pytest.approx(4.0)
    # Every group but the extruded top (which gained side walls) keeps its area.
    original = group_areas(grouped_cube())
    for g in set(areas) - {top_group}:
        assert areas[g] == pytest.approx(original[g])
    assert areas[top_group] == pytest.approx(original[top_group] + 4 * 2.0 * 0.5)


def test_bevel_faces_and_created_faces_get_neighbour_groups():
    em = EditableMesh.from_mesh(grouped_cube())
    edges = em.select("edges", box=((-2, -2, 0.9), (2, 2, 1.1)))  # the top rim
    em.bevel_edges(edges, 0.1)
    assert {em.face_group(f) for f in em.face_ids()} <= {0, 1, 2, 3, 4, 5}
    open_mesh = EditableMesh.from_mesh(grouped_cube())
    top = open_mesh.select("faces", normal=((0, 0, 1), 1), expect=1)
    loop = open_mesh.face_vertices(min(top.faces))
    open_mesh.delete_faces(top)
    face = open_mesh.create_face(loop)  # fills the hole: takes the lowest-id neighbour's group
    neighbours = sorted(
        {g for e in open_mesh.face_edges(face.data["face"]) for g in open_mesh.edge_faces(e)}
        - {face.data["face"]}
    )
    assert open_mesh.face_group(face.data["face"]) == open_mesh.face_group(neighbours[0])


def test_set_group_undo_and_merging_different_groups_is_reported():
    em = EditableMesh.from_mesh(create_cube(2.0))
    assert not em.has_groups
    top = em.select("faces", normal=((0, 0, 1), 1), expect=1)
    em.set_group(top, 3)
    top_face = min(top.faces)
    assert em.has_groups and em.face_group(top_face) == 3
    assert sorted(set(group_areas(em.to_mesh()))) == [0, 3]
    em.undo()
    assert not em.has_groups and em.to_mesh().groups is None
    em.redo()
    assert em.face_group(top_face) == 3
    with pytest.raises(Exception, match="group must be"):
        em.set_group(top, 70000)
    # Dissolving the edge between groups 0 and 3 keeps one group and says so.
    side = min(em.select("faces", normal=((1, 0, 0), 1), expect=1).faces)
    shared = [e for e in em.face_edges(top_face) if e in em.face_edges(side)]
    result = em.dissolve_edges(shared)
    assert "FACE_GROUP_CHANGED" in [d.code for d in result.diagnostics]
    survivor = min(top_face, side)
    assert em.face_group(survivor) == (3 if survivor == top_face else 0)


def test_edit_sessions_keep_groups_and_their_materials():
    scene = Scene()
    obj = scene.add_object("Box", mesh=grouped_cube())
    red = scene.create_material("Red", model="simple", base_color="#ff0000")
    scene.assign_material(obj, red, group=4)  # +z
    with obj.edit_mesh() as edit:
        top = edit.select("faces", normal=((0, 0, 1), 1), expect=1)
        edit.extrude_faces(top, distance=1.0)
    assert obj.mesh.groups is not None and set(np.unique(obj.mesh.groups)) == set(range(6))
    scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=4,
        position=(0, 0, 9),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )
    image = scene.render(width=16, height=16, default_lights=False)
    assert image.pixel(8, 8)[0] > 0 and image.pixel(8, 8)[1] == 0  # the extruded top is still red


def test_edit_scripts_select_and_set_groups():
    scene = Scene()
    scene.add_object("Box", mesh=create_cube(2.0))
    script = {
        "format": "aether3d.edit-script",
        "version": 1,
        "edits": [
            {
                "object": "Box",
                "operations": [
                    {
                        "op": "select",
                        "kind": "faces",
                        "normal": [[0, 0, 1], 1],
                        "expect": 1,
                        "as": "top",
                    },
                    {"op": "set_group", "faces": {"selection": "top"}, "group": 2},
                    {"op": "select", "kind": "faces", "group": 2, "expect": 1, "as": "again"},
                ],
            }
        ],
    }
    run_edit_script(scene, script)
    groups = scene.get_by_name("Box").mesh.groups
    assert sorted(np.unique(groups).tolist()) == [0, 2] and int((groups == 2).sum()) == 2


def poly_cube() -> PolyMesh:
    return PolyMesh.from_mesh(grouped_cube())


@pytest.mark.parametrize(
    "modifier,params",
    [
        ("transform", {"translation": (1, 2, 3), "scale": (2, 1, 1)}),
        ("mirror", {"axes": "x", "origin": (2, 0, 0)}),
        ("array", {"count": 3, "relative_offset": (1.5, 0, 0)}),
        ("subdivision", {"levels": 2}),
        ("solidify", {"thickness": 0.1}),
        ("bevel", {"width": 0.1, "limit": "all"}),
    ],
)
def test_procedural_modifiers_carry_groups(modifier, params):
    scene = Scene()
    geometry = scene.add_geometry("G", mesh=grouped_cube())
    geometry.add_modifier(modifier, **params)
    obj = scene.add_object("O", geometry=geometry)
    mesh = obj.mesh
    assert mesh.groups is not None, modifier
    assert set(np.unique(mesh.groups).tolist()) == set(range(6)), modifier
    if modifier == "transform":
        assert group_areas(mesh)[0] == pytest.approx(
            4.0 * 1.0
        )  # the +x face is not scaled in area? (y/z)
    if modifier in ("mirror", "array"):
        copies = 2 if modifier == "mirror" else 3
        assert group_areas(mesh)[0] == pytest.approx(copies * 4.0)


@pytest.mark.skipif(not HAS_MANIFOLD, reason="booleans need manifold3d")
def test_booleans_keep_the_groups_of_both_operands():
    scene = Scene()
    base = scene.add_geometry("Base", mesh=grouped_cube(2.0))
    cutter_mesh = create_cube(1.0)
    cutter = scene.add_geometry(
        "Cutter",
        mesh=Mesh(
            cutter_mesh.positions,
            cutter_mesh.triangles,
            cutter_mesh.normals,
            cutter_mesh.uvs,
            groups=np.full(12, 9),
        ),
    )
    base.add_modifier(
        "boolean",
        operation="difference",
        operand={"geometry": cutter.id},
        operand_translation=(1, 1, 1),
    )
    obj = scene.add_object("O", geometry=base)
    groups = set(np.unique(obj.mesh.groups).tolist())
    assert groups == {0, 1, 2, 3, 4, 5, 9}  # the cut's walls come from the cutter's faces


def test_meshes_without_groups_stay_without_groups():
    scene = Scene()
    geometry = scene.add_geometry("G", mesh=create_cube())
    geometry.add_modifier("subdivision", levels=1)
    assert scene.add_object("O", geometry=geometry).mesh.groups is None
    polymesh = PolyMesh.from_mesh(create_cube())
    assert polymesh.groups is None and polymesh.to_mesh()[0].groups is None
    assert poly_cube().content_hash() != PolyMesh.from_mesh(create_cube(2.0)).content_hash()
