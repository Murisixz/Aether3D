"""Stage 12 feedback meets Stage 13 sculpting: the ``mesh_validity`` expectation check
(exact, geometry basis, never corrected) on open, closed, intersecting, mirrored and
sculpted meshes."""

import numpy as np

from aether3d import FeedbackSession, Mesh, Scene, create_sphere, run_sculpt_script

EXPECT = {"format": "aether3d.expectations", "version": 1}


def diagnose(scene: Scene, *checks):
    doc = {**EXPECT, "checks": list(checks), "defaults": False}
    return FeedbackSession(scene, doc, width=96, height=54).diagnose()


def test_open_meshes_are_below_closed_and_findings_are_exact_geometry():
    scene = Scene()
    scene.add_plane("Floor", size=4)
    scene.add_cube("Box", size=1, position=(0, 0, 0.5))
    d = diagnose(scene, {"check": "mesh_validity", "objects": ["Floor", "Box"]})
    assert [f.code for f in d.findings] == ["MESH_BELOW_TIER"]
    finding = d.findings[0]
    assert finding.subjects[0]["name"] == "Floor"
    assert finding.basis == "geometry" and finding.exact
    assert finding.evidence["tier"] == "manifold" and finding.evidence["boundary_edge_count"] == 4
    assert d.checks[0].measured == {"Floor": "manifold", "Box": "closed"}
    d = diagnose(scene, {"check": "mesh_validity", "objects": ["Floor", "Box"], "tier": "manifold"})
    assert not d.findings and d.accepted


def test_solid_needs_no_self_intersection():
    ball = create_sphere(1.0, 24, 12)
    p = np.concatenate([ball.positions, ball.positions + np.float32([0.8, 0, 0])])
    t = np.concatenate([ball.triangles, ball.triangles + len(ball.positions)])
    scene = Scene()
    scene.add_object("Pair", mesh=Mesh(p, t))
    scene.add_object("Ball", mesh=ball, position=(5, 0, 0))
    d = diagnose(scene, {"check": "mesh_validity", "objects": ["Pair", "Ball"], "tier": "solid"})
    assert [f.subjects[0]["name"] for f in d.findings] == ["Pair"]
    assert d.findings[0].evidence["self_intersecting_pairs"] > 0
    assert d.findings[0].evidence["components"] == 2


def test_mirrored_objects_keep_their_orientation():
    scene = Scene()
    scene.add_object("Ball", mesh=create_sphere(1.0, 24, 12), scale=(-1, 1, 1))
    d = diagnose(scene, {"check": "mesh_validity", "objects": "Ball", "tier": "solid"})
    assert not d.findings


def test_a_sculpted_object_is_checked_after_the_script():
    scene = Scene()
    scene.add_object("Head", mesh=create_sphere(1.0, 32, 16))
    run_sculpt_script(
        scene,
        {
            "format": "aether3d.sculpt-script",
            "version": 1,
            "sculpts": [
                {
                    "object": "Head",
                    "require": "solid",
                    "operations": [
                        {
                            "op": "stroke",
                            "brush": {
                                "kind": "draw",
                                "radius": 0.4,
                                "strength": 0.3,
                                "symmetry": "x",
                                "dyntopo": {"detail": 0.2},
                            },
                            "path": [[0.3, -0.95, 0.0], [0.3, -0.9, 0.4]],
                        }
                    ],
                }
            ],
        },
    )
    d = diagnose(scene, {"check": "mesh_validity", "objects": "Head", "tier": "solid"})
    assert not d.findings and d.checks[0].measured == {"Head": "solid"}
