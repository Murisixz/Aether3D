"""The anatomical test figure (``aether3d.sculpt.figure``, 0.19.1): its options, its mesh
(symmetric, standing on the floor, legs apart below the crotch, closed when remeshed), its
body parts and garments."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d.errors import ValidationError
from aether3d.sculpt.figure import (
    CROTCH_BELOW_HIP,
    FIGURE_DEFAULTS,
    PARTS,
    _figure_shapes,
    figure_garment,
    figure_joints,
    figure_mesh,
    figure_shapes,
)


@pytest.fixture(scope="module")
def mesh():
    return figure_mesh(0.02)


def test_options_are_checked():
    for bad in ({"height": -1}, {"arm_angle": 120}, {"fingers": 0.5}, {"wings": 2},
                {"finger_spread": 40}, {"muscle": "big"}):  # fmt: skip
        with pytest.raises(ValidationError):
            figure_shapes(**bad)
    assert figure_shapes() == figure_shapes()  # (deterministic)
    j = figure_joints(height=1.2)
    assert j["head_top"][2] == pytest.approx(1.2)


def test_the_figure_stands_symmetric_on_the_floor_with_its_legs_apart(mesh):
    p = mesh.positions.astype(np.float64)
    H = FIGURE_DEFAULTS["height"]
    assert p[:, 2].min() == pytest.approx(0.0, abs=1e-3)  # (flat soles on z = 0)
    assert p[:, 2].max() == pytest.approx(H, abs=0.02)
    # Mirror symmetric: every vertex has a mirror image on the surface within a voxel.
    from aether3d.animation.weights import nearest_points

    d = nearest_points(p * [-1.0, 1.0, 1.0], p)[1][:, 0]
    assert np.quantile(d, 0.99) < 0.02
    # The thighs, knees and feet are two surfaces below the crotch (no web between them).
    crotch = figure_joints()["hip.L"][2] - CROTCH_BELOW_HIP * H
    below = (p[:, 2] > 0.05) & (p[:, 2] < crotch - 0.03)
    assert np.abs(p[below, 0]).min() > 0.002
    # Closed even where the blended field is steeper than a distance (thin fingertips):
    # every cell the surface crosses is meshed (it had three boundary edges before 0.19.1).
    from aether3d.sculpt.validate import check_mesh

    assert check_mesh(mesh.positions, mesh.triangles).summary["boundary_edge_count"] == 0


def test_remeshed_the_figure_is_one_closed_surface():
    from aether3d.animation.voxels import mesh_components
    from aether3d.sculpt.validate import check_mesh

    m = figure_mesh(0.009, remesh=0.009)  # (coarser, thin fingertips part or fuse)
    report = check_mesh(m.positions, m.triangles)
    assert report.summary["closed"] and report.summary["euler_characteristic"] == 2
    assert mesh_components(m.vertex_count, m.triangles)[1] == 1
    assert m.normals is not None and np.allclose(np.linalg.norm(m.normals, axis=1), 1.0, atol=1e-4)


def test_every_shape_belongs_to_a_part_and_garments_cover_theirs():
    tags: list[str] = []
    shapes = _figure_shapes({}, 0.0, tags)
    assert len(tags) == len(shapes)
    assert {t.split(".")[0] for t in tags} == set(PARTS) | {"cut"}
    assert {t for t in tags if t.startswith("hand")} == {"hand.L", "hand.R"}
    H = FIGURE_DEFAULTS["height"]
    jacket = figure_garment(("torso", "upper_arm", "forearm"), 0.012, 0.024)
    j = jacket.positions
    assert j[:, 2].min() > 0.4 * H and j[:, 2].max() < 0.9 * H  # (no legs, no head)
    # Trousers are meshed a side at a time, each half clipped at the mirror plane (so
    # their inner faces cover the thighs whatever their voxels), and cut at a height.
    from aether3d.animation.voxels import mesh_components

    cut = figure_garment(("torso", "hips", "thigh", "shin"), 0.01, 0.024, top=1.05)
    labels, count = mesh_components(cut.vertex_count, cut.triangles)
    assert count == 2 and cut.positions[:, 2].max() <= 1.05 + 1e-6
    for k in range(2):
        x = cut.positions[labels == k, 0]
        assert x.min() >= -1e-6 or x.max() <= 1e-6  # (each component on its own side)
    jacket_cut = figure_garment(("torso",), 0.012, 0.024, bottom=1.0).positions
    assert jacket_cut[:, 2].min() >= 1.0 - 1e-6
    with pytest.raises(ValidationError):
        figure_garment(("torso",), 0.01, top="waist")
    left = figure_garment("hand.L", 0.005, 0.024).positions
    assert left[:, 0].min() > 0.1  # (one side only)
    for bad in (("wings",), ("torso", "tail.L")):
        with pytest.raises(ValidationError):
            figure_garment(bad, 0.01)
    with pytest.raises(ValidationError):
        figure_garment(("torso",), -0.01)
