"""Stage 12 feedback meets Stage 14 texturing: the ``uv_quality`` expectation check (exact
flips and overlaps, stacked islands allowed, stretch, density and gap limits, never
corrected)."""

import numpy as np
import pytest

from aether3d import FeedbackError, FeedbackSession, Mesh, Scene, create_sphere

EXPECT = {"format": "aether3d.expectations", "version": 1}


def diagnose(scene: Scene, *checks):
    doc = {**EXPECT, "checks": list(checks), "defaults": False}
    return FeedbackSession(scene, doc, width=96, height=54).diagnose()


def scene_with_balls():
    scene = Scene()
    good = scene.add_object("Good", mesh=create_sphere(1.0, 24, 12))
    good.unwrap(resolution=256, padding=4)
    ball = create_sphere(1.0, 24, 12)
    uvs = ball.uvs.copy()
    uvs[ball.triangles[100]] = uvs[ball.triangles[300]] + 0.01  # an unintended overlap
    scene.add_object("Bad", mesh=Mesh(ball.positions, ball.triangles, normals=ball.normals,
                                      uvs=uvs), position=(3, 0, 0))  # fmt: skip
    scene.add_object("Bare", mesh=Mesh(ball.positions, ball.triangles), position=(6, 0, 0))
    return scene


def test_uv_quality_finds_overlaps_and_missing_uvs_exactly():
    scene = scene_with_balls()
    d = diagnose(scene, {"check": "uv_quality", "objects": ["Good", "Bad", "Bare"]})
    codes = sorted((f.code, f.subjects[0]["name"]) for f in d.findings)
    assert ("UV_OVERLAP", "Bad") in codes and ("NO_UVS", "Bare") in codes
    assert not any(name == "Good" for _, name in codes)
    overlap = next(f for f in d.findings if f.code == "UV_OVERLAP")
    assert overlap.basis == "geometry" and overlap.exact and overlap.unit == "triangles"
    assert d.checks[0].measured["Good"]["overlapping_triangles"] == 0
    assert d.checks[0].measured["Bare"] is None
    assert not d.accepted


def test_uv_quality_limits_and_stacked_islands():
    scene = Scene()
    ball = scene.add_object("Ball", mesh=create_sphere(1.0, 24, 12))
    ball.unwrap(resolution=256, padding=4)
    d = diagnose(scene, {"check": "uv_quality", "objects": "Ball", "max_stretch": 1.0001,
                         "max_density_spread": 0.0, "min_gap": 64})  # fmt: skip
    assert {f.code for f in d.findings} == {"UV_STRETCH", "UV_DENSITY_SPREAD", "UV_GAP"}
    d = diagnose(scene, {"check": "uv_quality", "objects": "Ball", "max_stretch": 2.0,
                         "max_density_spread": 1.0, "min_gap": 1})  # fmt: skip
    assert not d.findings and d.accepted
    # Two copies of the same island laid on each other: stacked (deliberate), not an error.
    m = ball.mesh
    t = np.concatenate([m.triangles, m.triangles + len(m.positions)])
    p = np.concatenate([m.positions, m.positions + np.float32([3, 0, 0])])
    twin = Mesh(p, t, normals=np.concatenate([m.normals] * 2), uvs=np.concatenate([m.uvs] * 2))
    scene.add_object("Twins", mesh=twin, position=(0, 4, 0))
    d = diagnose(scene, {"check": "uv_quality", "objects": "Twins"})
    assert not d.findings
    assert d.checks[0].measured["Twins"]["stacked_pairs"] > 0


@pytest.mark.parametrize(
    "bad", [{"resolution": 3}, {"max_flipped": -1}, {"max_stretch": 0.5}, {"min_gap": "x"}]
)
def test_uv_quality_arguments_are_checked(bad):
    scene = scene_with_balls()
    with pytest.raises(FeedbackError):
        diagnose(scene, {"check": "uv_quality", "objects": "Good", **bad})
