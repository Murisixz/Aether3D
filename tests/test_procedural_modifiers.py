"""Primitive types and every modifier type, checked against exact expectations."""

import math

import numpy as np
import pytest
from procedural_helpers import apply, codes, is_solid, prim, topology, volume

from aether3d import ProceduralError
from aether3d.procedural import PRIMITIVES, PolyMesh
from aether3d.procedural.modifiers import MODIFIER_TYPES, _select_faces, compact

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #


def polygon_area(n, r):
    return 0.5 * n * r * r * math.sin(2 * math.pi / n)


@pytest.mark.parametrize(
    ("name", "values", "expected_volume"),
    [
        ("box", {"width": 2, "depth": 3, "height": 4}, 24.0),
        ("cylinder", {"radius": 1, "depth": 2, "segments": 6}, polygon_area(6, 1) * 2),
        (
            "cone",
            {"radius_bottom": 1, "radius_top": 0, "depth": 3, "segments": 8},
            polygon_area(8, 1) * 3 / 3,
        ),
        (
            "cone",
            {"radius_bottom": 2, "radius_top": 1, "depth": 1, "segments": 4},
            (
                polygon_area(4, 2)
                + polygon_area(4, 1)
                + math.sqrt(polygon_area(4, 2) * polygon_area(4, 1))
            )
            / 3,
        ),
        ("uv_sphere", {"radius": 1, "segments": 64, "rings": 32}, None),
        (
            "torus",
            {"major_radius": 2, "minor_radius": 0.5, "major_segments": 64, "minor_segments": 32},
            None,
        ),
    ],
)
def test_closed_primitives_are_valid_solids(name, values, expected_volume):
    poly = prim(name, **values)
    assert is_solid(poly)
    report = poly.validate()
    assert report.ok, report.to_dict()
    if expected_volume is not None:
        assert volume(poly) == pytest.approx(expected_volume, rel=1e-6)
    elif name == "uv_sphere":
        assert volume(poly) == pytest.approx(4 / 3 * math.pi, rel=0.01)
    else:
        assert volume(poly) == pytest.approx(2 * math.pi**2 * 2 * 0.25, rel=0.01)
    expected_euler = 0 if name == "torus" else 2
    assert report.summary["euler_characteristic"] == expected_euler


def test_grid_and_open_cylinder():
    grid = prim("grid", x_size=4, y_size=2, x_segments=4, y_segments=2)
    assert grid.face_count == 8 and grid.vertex_count == 15
    t = topology(grid)
    assert not t["closed"] and t["consistently_oriented"]
    assert np.allclose(grid.face_normals[:, 2] > 0, True)
    tube = prim("cylinder", segments=12, caps=False)
    assert tube.face_count == 12 and not topology(tube)["closed"]


def test_primitives_carry_uvs_and_normals():
    for name in PRIMITIVES:
        poly = prim(name)
        assert poly.uvs is not None and poly.uvs.shape == (poly.corner_count, 2)
        mesh, _ = poly.to_mesh()
        assert mesh.normals is not None and np.allclose(
            np.linalg.norm(mesh.normals, axis=1), 1, atol=1e-5
        )


@pytest.mark.parametrize(
    ("name", "values", "code"),
    [
        ("box", {"width": 0}, "PARAMETER_OUT_OF_RANGE"),
        ("box", {"width": -1}, "PARAMETER_OUT_OF_RANGE"),
        ("cylinder", {"segments": 2}, "PARAMETER_OUT_OF_RANGE"),
        ("cone", {"radius_bottom": 0, "radius_top": 0}, "INVALID_COMBINATION"),
        ("torus", {"major_radius": 1, "minor_radius": 1}, "INVALID_COMBINATION"),
        ("grid", {"x_segments": 0}, "PARAMETER_OUT_OF_RANGE"),
        ("box", {"size": 1}, "UNKNOWN_PARAMETER"),
        ("box", {"width": "wide"}, "INVALID_PARAMETER"),
    ],
)
def test_invalid_primitive_parameters(name, values, code):
    with pytest.raises(ProceduralError) as info:
        prim(name, **values)
    assert info.value.code == code


def test_type_schemas_are_complete():
    for registry in (PRIMITIVES, MODIFIER_TYPES):
        for t in registry.values():
            schema = t.schema()
            assert schema["type"] == t.name and schema["description"]
            for p in schema["params"]:
                assert p["description"] and p["type"]
                assert "default" in p or p.get("required")


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #


def test_transform_translate_rotate_scale_about_pivot():
    box = prim("box", width=2, depth=2, height=2)
    out, diags = apply(
        "transform",
        box,
        translation=(1, 2, 3),
        rotation=(0, 0, 90),
        scale=(2, 1, 1),
        pivot=(1, 1, 1),
    )
    assert diags == []
    assert volume(out) == pytest.approx(16)
    # Point (1,1,1) is the pivot: unchanged by scale/rotation, then translated.
    p = np.array([1.0, 1.0, 1.0])
    i = np.flatnonzero(np.all(np.isclose(box.positions, p), axis=1))[0]
    assert np.allclose(out.positions[i], p + (1, 2, 3))
    assert np.array_equal(box.positions[i], p)  # input untouched (immutable)


def test_mirroring_transform_rewinds_faces_and_transforms_normals():
    sphere = prim("uv_sphere", segments=16, rings=8)
    out, _ = apply("transform", sphere, scale=(-1, 2, 1))
    assert is_solid(out) and volume(out) > 0
    mesh, _ = out.to_mesh()
    # Stored normals still point outward of the (stretched) sphere.
    centroid_dirs = mesh.positions / np.linalg.norm(mesh.positions, axis=1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", mesh.normals, centroid_dirs) > 0)


def test_transform_rejects_zero_scale():
    with pytest.raises(ProceduralError) as info:
        apply("transform", prim("box"), scale=(1, 0, 1))
    assert info.value.code == "DEGENERATE_TRANSFORM"


# --------------------------------------------------------------------------- #
# Mirror
# --------------------------------------------------------------------------- #


def half_box():
    """A unit-ish box spanning x in [0, 2] with its -X face removed (open on the plane)."""
    box, _ = apply("transform", prim("box", width=2, depth=2, height=2), translation=(1, 0, 0))
    keep = ~(box.face_normals[:, 0] < -1e-9)
    return compact(_select_faces(box, keep))


def test_mirror_merge_closes_the_seam():
    half = half_box()
    assert not topology(half)["closed"]
    out, diags = apply("mirror", half, axes="x")
    assert is_solid(out) and volume(out) == pytest.approx(16)
    assert diags == []
    assert out.vertex_count == 12  # 8 + 4 new, the 4 seam vertices shared


def test_mirror_merge_removes_interior_cap_faces():
    # A closed half (its cap face lies on the plane): the cap would end up inside.
    box, _ = apply("transform", prim("box", width=2, depth=2, height=2), translation=(1, 0, 0))
    out, diags = apply("mirror", box, axes="x")
    assert is_solid(out) and volume(out) == pytest.approx(16) and out.face_count == 10
    assert codes(diags) == ["MIRROR_INTERIOR_FACES_REMOVED"]
    assert [d.count for d in diags] == [1]
    # Without merging nothing is removed: two separate boxes whose caps coincide.
    out, diags = apply("mirror", box, axes="x", merge=False)
    assert out.face_count == 12 and codes(diags) == ["MIRROR_COINCIDENT_FACES"]


def test_mirror_without_merge_leaves_the_seam_open():
    out, _ = apply("mirror", half_box(), axes="x", merge=False)
    # The halves only touch: polygon topology has boundary edges along the seam.
    assert not out.validate().summary["closed"] and out.vertex_count == 16


def test_mirror_merge_threshold_snaps_vertices_onto_the_plane():
    half = half_box()
    nudged = half.with_(
        positions=half.positions + np.where(half.positions[:, :1] == 0, 3e-5, 0) * [1, 0, 0]
    )
    out, _ = apply("mirror", nudged, axes="x", merge_threshold=1e-4)
    assert is_solid(out)
    out, _ = apply("mirror", nudged, axes="x", merge_threshold=1e-6)
    assert not out.validate().summary["closed"]


def test_mirror_multiple_axes_and_origin():
    box = prim("box", width=1, depth=1, height=1)
    out, diags = apply("mirror", box, axes="xyz", origin=(2, 2, 2), merge=False)
    assert out.face_count == 48 and volume(out) == pytest.approx(8)
    assert np.allclose(out.bounds()[1], (4.5, 4.5, 4.5))
    assert diags == []


def test_mirror_diagnostics_for_overlap_and_faces_on_plane():
    box = prim("box", width=2, depth=2, height=2)  # spans the X plane
    _, diags = apply("mirror", box, axes="x")
    assert "MIRROR_OVERLAP" in codes(diags)
    grid = prim("grid")  # lies in the Z plane
    out, diags = apply("mirror", grid, axes="z")
    assert "MIRROR_FACES_ON_PLANE" in codes(diags) and out.face_count == 1


def test_mirror_uvs():
    half = half_box()
    plain, _ = apply("mirror", half, axes="x")
    flipped, _ = apply("mirror", half, axes="x", mirror_u=True)
    n = half.corner_count
    assert np.allclose(plain.uvs[n:, 0], half.uvs[_rewound(half), 0])
    assert np.allclose(flipped.uvs[n:, 0], 1 - half.uvs[_rewound(half), 0])
    assert np.allclose(flipped.uvs[:n], half.uvs)


def _rewound(mesh):
    from aether3d.procedural.modifiers import reverse_corners

    return reverse_corners(mesh)


# --------------------------------------------------------------------------- #
# Array
# --------------------------------------------------------------------------- #


def test_array_relative_and_constant_offsets():
    box = prim("box", width=2, depth=1, height=1)
    out, _ = apply("array", box, count=4, relative_offset=(1.5, 0, 0), constant_offset=(0, 0, 1))
    assert out.face_count == 24 and volume(out) == pytest.approx(8)
    lo, hi = out.bounds()
    assert np.allclose(lo, (-1, -0.5, -0.5)) and np.allclose(hi, (1 + 3 * 3, 0.5, 3.5))


def test_array_rotation_makes_radial_copies():
    box, _ = apply(
        "transform", prim("box", width=0.5, depth=0.5, height=0.5), translation=(3, 0, 0)
    )
    out, _ = apply("array", box, count=6, relative_offset=(0, 0, 0), rotation=(0, 0, 60))
    centers = np.add.reduceat(out.positions, np.arange(0, 48, 8)) / 8
    angles = np.degrees(np.arctan2(centers[:, 1], centers[:, 0])) % 360
    assert np.allclose(np.sort(angles), [0, 60, 120, 180, 240, 300], atol=1e-6)
    assert np.allclose(np.linalg.norm(centers[:, :2], axis=1), 3)


def test_array_merge_welds_touching_boundaries():
    # Two open half boxes facing each other merge into one closed box.
    half = half_box()
    out, diags = apply(
        "array", half, count=2, relative_offset=(0, 0, 0), scale=(-1, 1, 1), merge=True
    )
    assert "ARRAY_MERGED" in codes(diags)
    assert is_solid(out) and volume(out) == pytest.approx(16)
    # A strip of grid quads merges into one connected grid.
    grid = prim("grid", x_size=1, y_size=1)
    out, _ = apply("array", grid, count=5, merge=True)
    assert out.vertex_count == 12 and topology(out)["consistently_oriented"]
    unmerged, _ = apply("array", grid, count=5)
    assert unmerged.vertex_count == 20


def test_array_count_one_and_limits():
    box = prim("box")
    out, _ = apply("array", box, count=1)
    assert out is box
    with pytest.raises(ProceduralError) as info:
        apply("array", prim("uv_sphere", segments=256, rings=128), count=100_000)
    assert info.value.code == "PROCEDURAL_LIMIT"
    with pytest.raises(ProceduralError):
        apply("array", box, count=0)


# --------------------------------------------------------------------------- #
# Solidify
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("offset", "z_range"), [(-1, (-0.5, 0)), (1, (0, 0.5)), (0, (-0.25, 0.25))]
)
def test_solidify_open_surface_makes_a_slab(offset, z_range):
    grid = prim("grid", x_size=2, y_size=3, x_segments=2, y_segments=3)
    out, diags = apply("solidify", grid, thickness=0.5, offset=offset)
    assert is_solid(out) and volume(out) == pytest.approx(3.0)
    lo, hi = out.bounds()
    assert np.isclose(lo[2], z_range[0]) and np.isclose(hi[2], z_range[1])
    assert "UV_DEGENERATE_FACES" in codes(diags)
    assert out.face_count == 2 * 6 + 10  # two shells + one rim quad per boundary edge


def test_solidify_closed_mesh_makes_a_hollow_shell():
    box = prim("box", width=2, depth=2, height=2)
    out, _ = apply("solidify", box, thickness=0.1, even_thickness=True)
    t = topology(out)
    assert t["closed"] and t["consistently_oriented"]
    # Outer surface kept (offset -1); the inner one is 0.1 inside with even thickness.
    assert volume(out) == pytest.approx(8 - 1.8**3, rel=1e-6)


def test_solidify_without_rim_and_errors():
    grid = prim("grid")
    out, diags = apply("solidify", grid, thickness=0.2, rim=False)
    assert out.face_count == 2 and "SOLIDIFY_OPEN" in codes(diags)
    with pytest.raises(ProceduralError) as info:
        apply("solidify", grid, thickness=0)
    assert info.value.code == "INVALID_PARAMETER"
    fin = PolyMesh.from_faces(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0)], [(0, 1, 2), (1, 0, 3), (0, 1, 4)]
    )
    with pytest.raises(ProceduralError) as info:
        apply("solidify", fin, thickness=0.1)
    assert info.value.code == "UNSUPPORTED_TOPOLOGY"


# --------------------------------------------------------------------------- #
# Subdivision surface
# --------------------------------------------------------------------------- #


def test_catmull_clark_cube_matches_the_known_result():
    box = prim("box", width=2, depth=2, height=2)
    out, _ = apply("subdivision", box, levels=1)
    assert out.face_count == 24 and out.vertex_count == 26 and is_solid(out)
    corners = out.positions[:8]
    assert np.allclose(np.abs(corners), 5 / 9)  # classic Catmull-Clark cube corner
    edge_points = out.positions[8:20]
    assert np.allclose(np.sort(np.abs(edge_points), axis=1), [[0, 0.75, 0.75]] * 12)
    assert out.smooth.all()


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_subdivision_levels_preserve_topology(levels):
    torus = prim("torus", major_segments=8, minor_segments=6)
    out, _ = apply("subdivision", torus, levels=levels)
    assert out.face_count == torus.face_count * 4**levels
    report = out.validate()
    assert report.ok and report.summary["euler_characteristic"] == 0
    assert np.all(out.face_sizes == 4)


def test_subdivision_boundary_rule_on_open_grid():
    grid = prim("grid", x_size=2, y_size=2, x_segments=2, y_segments=2)
    out, _ = apply("subdivision", grid, levels=1)
    assert out.face_count == 16
    # Boundary vertices stay on the boundary line; corners are kept.
    assert np.allclose(out.bounds()[0], (-1, -1, 0)) and np.allclose(out.bounds()[1], (1, 1, 0))
    assert topology(out)["consistently_oriented"]


def test_subdivision_uvs_and_errors():
    box = prim("box")
    out, _ = apply("subdivision", box, levels=2, shade_smooth=False)
    assert out.uvs is not None and out.uvs.shape == (out.corner_count, 2)
    assert not out.smooth.any()
    assert apply("subdivision", box, levels=0)[0] is box
    with pytest.raises(ProceduralError):
        apply("subdivision", box, levels=7)
    with pytest.raises(ProceduralError) as info:
        apply("subdivision", prim("uv_sphere", segments=512, rings=256), levels=6)
    assert info.value.code == "PROCEDURAL_LIMIT"


# --------------------------------------------------------------------------- #
# Bevel
# --------------------------------------------------------------------------- #


def test_bevel_all_edges_of_a_cube_matches_the_stage3_formula():
    box = prim("box", width=2, depth=2, height=2)
    out, _ = apply("bevel", box, width=0.2, limit="all")
    # Each of 12 edges loses a 0.2x0.2/2 prism (length 2 - 2*0.2), each corner a tetra-like cap.
    assert volume(out) == pytest.approx(8 - 12 * 0.02 * 1.6 - 8 * (5 / 6) * 0.008, rel=1e-6)
    assert out.face_count == 26 and is_solid(out)


def test_bevel_angle_limit_and_no_edges():
    cyl = prim("cylinder", segments=32, smooth=True)
    out, _ = apply("bevel", cyl, width=0.05, limit="angle", angle=30)
    # Only the 64 cap rim edges are sharper than 30 degrees.
    assert out.face_count == cyl.face_count + 64 and is_solid(out)
    _, diags = apply("bevel", prim("grid"), width=0.1, limit="angle")
    assert codes(diags) == ["BEVEL_NO_EDGES"]


def test_bevel_reports_stage3_errors():
    with pytest.raises(ProceduralError) as info:
        apply("bevel", prim("box", width=1, depth=1, height=1), width=0.6, limit="all")
    assert info.value.code == "BEVEL_WIDTH_TOO_LARGE"


# --------------------------------------------------------------------------- #
# Boolean (manifold3d)
# --------------------------------------------------------------------------- #


def boolean(op, a, b, **params):
    return apply("boolean", a, operand_mesh=b, operation=op, operand={"geometry": "B"}, **params)


def test_boolean_backend_missing_is_reported(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_manifold(name, *args, **kwargs):
        if name == "manifold3d":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_manifold)
    with pytest.raises(ProceduralError) as info:
        boolean("union", prim("box"), prim("box"))
    assert info.value.code == "BOOLEAN_BACKEND_UNAVAILABLE"
    assert "aether3d[boolean]" in info.value.hint


def test_boolean_operand_parameter_is_validated():
    for operand in (
        "B",
        {"name": "B"},
        {"geometry": "B", "extra": 1},
        {"geometry": "B", "parameters": 3},
    ):
        with pytest.raises(ProceduralError) as info:
            apply("boolean", prim("box"), operand=operand)
        assert info.value.code == "INVALID_PARAMETER"
    with pytest.raises(ProceduralError) as info:
        apply("boolean", prim("box"))
    assert info.value.code == "MISSING_PARAMETER"
