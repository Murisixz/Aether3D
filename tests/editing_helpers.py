"""Helpers shared by the mesh-editing tests."""

import numpy as np

from aether3d import inspect_mesh
from aether3d.editing import EditableMesh


def snapshot(m: EditableMesh) -> tuple:
    """The complete internal state, for exact before/after comparisons."""
    return (
        m._pos[: m._nv].tobytes(),
        m._alive[: m._nv].tobytes(),
        m._nv,
        m._next_face,
        m._next_edge,
        {k: list(v) for k, v in m._faces.items()},
        dict(m._edge_by_pair),
        {k: list(v) for k, v in m._edge_faces.items()},
        [sorted(s) for s in m._vert_edges],
        frozenset(m._sharp),
        dict(m._smooth),
        {n: {k: list(v) for k, v in layer.data.items()} for n, layer in m._layers.items()},
        frozenset(m._dirty_faces),
    )


def box(size=2.0, **kwargs) -> EditableMesh:
    m = EditableMesh(**kwargs)
    m.add_box(size)
    return m


def grid(nx=4, ny=4, size=4.0) -> EditableMesh:
    m = EditableMesh()
    m.add_grid(nx, ny, size)
    return m


def assert_healthy(m: EditableMesh) -> None:
    """Internal adjacency is consistent and nothing is broken."""
    assert m._consistency_errors() == []
    report = m.validate()
    assert "BROKEN_ADJACENCY" not in report.codes()


def volume(m: EditableMesh) -> float | None:
    return inspect_mesh(m.to_mesh()).topology["volume"]


def euler(m: EditableMesh) -> int:
    return m.vertex_count - m.edge_count + m.face_count


def outward_closed(m: EditableMesh) -> bool:
    """Closed, edge-manifold, consistently wound, positive volume."""
    t = inspect_mesh(m.to_mesh()).topology
    return bool(
        t["closed"] and t["edge_manifold"] and t["consistently_oriented"] and t["volume"] > 0
    )


def face_with_normal(m: EditableMesh, direction) -> int:
    (f,) = m.select_faces(normal=(direction, 1), expect=1).faces
    return f


def corner_map(m: EditableMesh, layer: str) -> dict[tuple[int, int], tuple]:
    """{(face, vertex): value} for a corner layer."""
    return {
        (f, v): m.corner_values(f, layer)[k]
        for f in m.face_ids()
        for k, v in enumerate(m.face_vertices(f))
    }


def total_area(m: EditableMesh) -> float:
    return float(sum(m.face_area(f) for f in m.face_ids()))


def trimesh_volume(m: EditableMesh) -> float:
    import trimesh

    mesh = m.to_mesh()
    tm = trimesh.Trimesh(mesh.positions.astype(float), mesh.triangles.astype(int), process=False)
    tm.merge_vertices()
    assert tm.is_watertight and tm.is_winding_consistent
    return float(tm.volume)


def np_close(a, b, tol=1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))
