"""Randomized robustness: any sequence of operations keeps the mesh consistent.

Every step either succeeds -- leaving consistent adjacency -- or raises an
AetherError and leaves the mesh bit-identical. Undo restores the exact
previous state. Meshes that validate without errors always triangulate.
"""

import numpy as np
import pytest
from editing_helpers import snapshot

from aether3d import AetherError, create_sphere, inspect_mesh
from aether3d.editing import EditableMesh


def mesh_box():
    m = EditableMesh()
    m.add_box(2)
    return m


def mesh_grid():
    m = EditableMesh()
    m.add_grid(4, 3, (4, 3))
    return m


def mesh_sphere():
    return EditableMesh.from_mesh(create_sphere(1, 10, 6))


def mesh_book():
    """Three quads sharing one edge (non-manifold) plus a separate box."""
    m = EditableMesh()
    a = m.create_vertex((0, 0, 0)).data["vertex"]
    b = m.create_vertex((0, 0, 1)).data["vertex"]
    for x, y in ((1, 0), (-1, 0), (0, 1)):
        c = m.create_vertex((x, y, 1)).data["vertex"]
        d = m.create_vertex((x, y, 0)).data["vertex"]
        m.create_face([a, b, c, d] if x >= 0 else [b, a, d, c], allow_non_manifold=True)
    m.add_box(1, center=(5, 0, 0))
    return m


def mesh_degenerate():
    """A box with a collapsed (zero-area) face and coincident vertices."""
    m = mesh_box()
    top = m.select_faces(normal=((0, 0, 1), 1))
    m.scale(top, (1, 0, 1), allow_degenerate=True)
    return m


MESHES = [mesh_box, mesh_grid, mesh_sphere, mesh_book, mesh_degenerate]


def random_step(m: EditableMesh, rng: np.random.Generator):
    faces, edges, verts = m.face_ids(), m.edge_ids(), m.vertex_ids()
    if not faces:
        return "add_box", (), {"size": float(rng.uniform(0.5, 2))}

    def some(ids, k=3):
        k = min(len(ids), int(rng.integers(1, k + 1)))
        return [int(i) for i in rng.choice(ids, size=k, replace=False)]

    op = rng.choice(
        [
            "translate", "rotate", "scale", "extrude_faces", "inset_faces", "bevel_edges",
            "subdivide_faces", "loop_cut", "collapse_edges", "split_edges", "dissolve_edges",
            "triangulate_faces", "join_triangles", "split_face", "delete_faces", "flip_faces",
            "merge_by_distance", "rip_vertices", "move_along_normals", "merge_vertices",
            "separate_faces", "split_edge",
        ]
    )  # fmt: skip
    if op == "translate":
        return op, (some(verts), rng.normal(size=3) * 0.3), {}
    if op == "rotate":
        return op, (some(verts), float(rng.uniform(-90, 90)), rng.normal(size=3)), {}
    if op == "scale":
        return op, (some(verts), rng.uniform(-1.5, 1.5, size=3)), {}
    if op == "extrude_faces":
        return (
            op,
            (some(faces), float(rng.choice([-1, 1]) * rng.uniform(0.05, 0.5))),
            {"mode": str(rng.choice(["region", "individual"]))},
        )
    if op == "inset_faces":
        return (
            op,
            (some(faces), float(rng.uniform(0.01, 0.4))),
            {
                "mode": str(rng.choice(["region", "individual"])),
                "depth": float(rng.choice([0, rng.uniform(-0.2, 0.2)])),
            },
        )
    if op == "bevel_edges":
        return op, (some(edges, 4), float(rng.uniform(0.01, 0.3))), {}
    if op == "subdivide_faces":
        if len(faces) > 300:
            return "triangulate_faces", (some(faces),), {}
        return op, (some(faces, 2), int(rng.integers(1, 3))), {}
    if op == "loop_cut":
        return op, (int(rng.choice(edges)),), {"cuts": int(rng.integers(1, 3))}
    if op in ("collapse_edges", "split_edges", "dissolve_edges"):
        return op, (some(edges, 2),), {}
    if op in ("triangulate_faces", "delete_faces", "flip_faces", "separate_faces"):
        return op, (some(faces, 2),), {}
    if op == "join_triangles":
        return op, (), {"max_angle": float(rng.uniform(0, 30))}
    if op == "split_face":
        f = int(rng.choice(faces))
        loop = m.face_vertices(f)
        a, b = rng.choice(len(loop), size=2, replace=False)
        return op, (f, loop[a], loop[b]), {}
    if op == "merge_by_distance":
        return op, (float(rng.uniform(0, 0.3)),), {}
    if op == "rip_vertices":
        return op, (some(verts, 2),), {}
    if op == "move_along_normals":
        return op, (some(verts), float(rng.uniform(-0.2, 0.2))), {}
    if op == "merge_vertices":
        return op, (some(verts, 3),), {}
    a, b = m.edge_vertices(int(rng.choice(edges)))
    return "split_edge", (m.edge_between(a, b), float(rng.uniform(0.05, 0.95))), {}


@pytest.mark.parametrize("make", MESHES, ids=[f.__name__ for f in MESHES])
@pytest.mark.parametrize("seed", range(6))
def test_random_operation_sequences(make, seed):
    rng = np.random.default_rng(seed * 101 + MESHES.index(make))  # stable across runs
    m = make()
    successes = 0
    for _step in range(40):
        name, args, kwargs = random_step(m, rng)
        before = snapshot(m)
        try:
            getattr(m, name)(*args, **kwargs)
        except AetherError:
            assert snapshot(m) == before, f"{name} failed but changed the mesh"
            continue
        successes += 1
        errors = m._consistency_errors()
        assert errors == [], f"{name}{args}{kwargs} broke adjacency: {errors[:3]}"
        if rng.random() < 0.3:
            after = snapshot(m)
            m.undo()
            assert snapshot(m) == before, f"undo of {name} is not exact"
            m.redo()
            assert snapshot(m) == after, f"redo of {name} is not exact"
        if m.face_count > 2000:
            break
    assert successes > 0
    report = m.validate()
    if report.ok and m.face_count:
        mesh = m.to_mesh()  # meshes that validate always triangulate
        inspect_mesh(mesh)
