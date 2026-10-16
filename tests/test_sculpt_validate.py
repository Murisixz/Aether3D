"""check_mesh: topology, degeneracy and self-intersections of triangle meshes."""

import numpy as np

from aether3d import create_cube, create_sphere
from aether3d.sculpt import check_mesh
from aether3d.sculpt.validate import weld


def test_a_closed_sphere_is_clean():
    m = create_sphere(1.0, 32, 16)
    r = check_mesh(m.positions, m.triangles, self_intersections=True)
    assert r.ok and r.status == "ok"
    s = r.summary
    assert s["closed"] and s["edge_manifold"] and s["vertex_manifold"]
    assert s["consistently_oriented"] and s["euler_characteristic"] == 2
    assert 0 < s["volume"] < 4 / 3 * np.pi and s["self_intersecting_pairs"] == 0


def test_render_seams_are_welded_before_checking():
    cube = create_cube(2)
    p, t = weld(cube.positions, cube.triangles)
    assert len(p) == 8
    assert check_mesh(cube.positions, cube.triangles).summary["closed"]
    unwelded = check_mesh(cube.positions, cube.triangles, welded=True)
    assert unwelded.summary["boundary_edge_count"] > 0


def test_non_manifold_vertices_edges_and_winding():
    # Two tetrahedra sharing one vertex: a non-manifold vertex.
    p = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float
    )
    t = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3], [0, 4, 5], [0, 6, 4], [0, 5, 6], [4, 6, 5]]
    )
    r = check_mesh(p, t)
    assert r.summary["non_manifold_vertex_count"] == 1 and "NON_MANIFOLD_VERTICES" in r.codes()
    # Three triangles on one edge.
    q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], float)
    r = check_mesh(q, np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]))
    assert r.summary["non_manifold_edge_count"] == 1
    # A flipped triangle in a closed mesh.
    sphere = create_sphere(1.0, 16, 8)
    p2, t2 = weld(sphere.positions, sphere.triangles)
    t2 = t2.copy()
    t2[5] = t2[5][::-1]
    r = check_mesh(p2, t2, welded=True)
    assert not r.summary["consistently_oriented"] and "volume" not in r.summary


def test_degenerate_triangles_and_self_intersections():
    p = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [0, -1, -1],
            [0, 1, -1],
            [0, 1, 1],
            [0, -1, 1],
        ],
        float,
    )
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    r = check_mesh(p, t, self_intersections=True)
    assert not r.ok and r.summary["self_intersecting_pairs"] == 4
    flat = check_mesh(np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float), np.array([[0, 1, 2]]))
    assert flat.summary["degenerate_triangle_count"] == 1
    lazy = check_mesh(p, t)
    assert "self-intersections (self_intersections=True checks them)" in lazy.not_evaluated
    poked = create_sphere(1.0, 32, 16)
    q, tq = weld(poked.positions, poked.triangles)
    q = q.copy()
    q[np.argmax(q[:, 2])] = (0, 0, -1.5)  # the north pole pushed out through the south
    r = check_mesh(q, tq, self_intersections=True, welded=True)
    assert r.summary["self_intersecting_pairs"] > 0
