"""Building an EditableMesh in bulk gives exactly the mesh built one face at a time.

``EditableMesh.from_mesh`` builds its faces, edges and corner layers with NumPy
(``MeshCore._bulk_faces``); ``AETHER3D_EDITING_BULK=0`` adds one face at a time as before.
The two must produce the same internal state -- ids, tables, and the insertion order of
every table and even the iteration order of every set, since later operations iterate
them -- and so the same results for every operation that follows.
"""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Mesh, create_cube, create_plane, create_sphere
from aether3d.editing import EditableMesh


def strict_snapshot(m: EditableMesh) -> tuple:
    """Every table of the mesh in its order: ids, insertion and iteration orders included."""
    return (
        m._pos[: m._nv].tobytes(),
        m._alive[: m._nv].tobytes(),
        m._nv,
        m._next_face,
        m._next_edge,
        list(m._faces.items()),
        list(m._edge_by_pair.items()),
        list(m._edge_verts.items()),
        list(m._edge_faces.items()),
        [list(s) for s in m._vert_edges],
        list(m._sharp),
        list(m._smooth.items()),
        list(m._group.items()),
        m._has_groups,
        [(n, layer.dim, list(layer.data.items())) for n, layer in m._layers.items()],
        list(m._dirty_faces),
        [d.to_dict() for d in m.source_diagnostics],
    )


def both(mesh: Mesh, monkeypatch, **options) -> tuple[EditableMesh, EditableMesh]:
    bulk = EditableMesh.from_mesh(mesh, **options)
    with monkeypatch.context() as m:
        m.setenv("AETHER3D_EDITING_BULK", "0")
        one_by_one = EditableMesh.from_mesh(mesh, **options)
    return bulk, one_by_one


def meshes():
    rng = np.random.default_rng(3)
    cube = create_cube(2.0)
    yield "cube", cube
    yield "sphere", create_sphere(1.0, 24, 12)
    yield "coarse sphere", create_sphere(0.5, 5, 3)
    yield "plane", create_plane(4.0)
    yield "no attributes", Mesh(cube.positions, cube.triangles)
    yield (
        "groups",
        Mesh(
            cube.positions,
            cube.triangles,
            cube.normals,
            cube.uvs,
            groups=np.repeat(np.arange(6), 2),
        ),
    )
    # A soup of random triangles over shared vertices: non-manifold edges, repeats,
    # degenerate triangles (repeated vertices), duplicate positions for welding.
    positions = rng.integers(-3, 4, (40, 3)).astype(np.float32)
    positions[30:] = positions[:10]  # bit-identical copies: welded
    positions[35, 0] = -0.0 if positions[5, 0] == 0.0 else positions[35, 0]
    triangles = rng.integers(0, 40, (300, 3))
    triangles[::17, 1] = triangles[::17, 0]  # degenerate
    uvs = rng.random((40, 2)).astype(np.float32)
    normals = rng.normal(size=(40, 3)).astype(np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    yield "soup", Mesh(positions, triangles.astype(np.uint32), normals, uvs)
    yield "soup without attributes", Mesh(positions, triangles.astype(np.uint32))
    yield "empty", Mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))  # fmt: skip


@pytest.mark.parametrize("name,mesh", list(meshes()), ids=[n for n, _ in meshes()])
@pytest.mark.parametrize("weld", ["exact", "none"])
@pytest.mark.parametrize("quads", [True, False])
def test_the_same_mesh_built_in_bulk(name, mesh, weld, quads, monkeypatch):
    bulk, one_by_one = both(mesh, monkeypatch, weld=weld, quads=quads)
    assert strict_snapshot(bulk) == strict_snapshot(one_by_one)


def test_operations_after_a_bulk_build_give_the_same_results(monkeypatch):
    from aether3d.errors import AetherError

    sphere = create_sphere(1.0, 32, 16)
    bulk, one_by_one = both(sphere, monkeypatch)

    def steps(m: EditableMesh):
        short = 0.05 * min(m.edge_length(e) for e in m.edge_ids())
        yield lambda: m.extrude_faces(m.face_ids()[::9], 0.1)
        yield lambda: m.inset_faces(m.face_ids()[::11], 0.01)
        yield lambda: m.bevel_edges(m.edge_ids()[::13], short)
        yield lambda: m.bevel_edges(m.edge_ids()[::29], 0.2 * short)
        yield lambda: m.subdivide_faces(
            [f for f in m.face_ids()[::7] if len(m.face_vertices(f)) == 4], 1
        )
        yield lambda: m.undo()

    for a, b in zip(steps(bulk), steps(one_by_one), strict=True):
        outcomes = []
        for step in (a, b):
            try:
                step()
                outcomes.append("ok")
            except AetherError as exc:
                outcomes.append(getattr(exc, "code", type(exc).__name__))
        assert outcomes[0] == outcomes[1]
        assert strict_snapshot(bulk) == strict_snapshot(one_by_one)
    x, y = bulk.to_mesh(), one_by_one.to_mesh()
    for field in ("positions", "triangles", "normals", "uvs"):
        p, q = getattr(x, field), getattr(y, field)
        assert (p is None and q is None) or (p.dtype == q.dtype and np.array_equal(p, q))


def _reference_join_pairs(self, pool, max_angle):
    """``_join_pairs`` as it was before its triangle normals were cached (0.11)."""
    from aether3d.editing import _polygon as poly

    tris = {f for f in pool if len(self._faces[f]) == 3}
    candidates = []
    for e in sorted(
        {self._edge_by_pair[k] for f in tris for k in self._loop_edges(self._faces[f])}
    ):
        fs = self._edge_faces[e]
        if len(fs) != 2 or not (fs[0] in tris and fs[1] in tris):
            continue
        n0, n1 = self._face_normal_vec(fs[0]), self._face_normal_vec(fs[1])
        if n0 is None or n1 is None:
            continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(n0, n1), -1, 1))))
        if angle > max_angle or not self._attributes_agree_across(e):
            continue
        quad = self._joined_loop(e)
        if quad is None:
            continue
        quad_normal = poly.newell_normal(self._pos[quad])
        if poly.unit(quad_normal) is None:
            continue
        pts = poly.project(self._pos[quad], quad_normal)
        crosses = [
            (pts[(k + 1) % 4][0] - pts[k][0]) * (pts[(k + 2) % 4][1] - pts[k][1])
            - (pts[(k + 1) % 4][1] - pts[k][1]) * (pts[(k + 2) % 4][0] - pts[k][0])
            for k in range(4)
        ]
        if min(crosses) <= 1e-12 * self._scale() ** 2:
            continue
        sides = [
            np.linalg.norm(self._pos[quad[(k + 1) % 4]] - self._pos[quad[k]]) for k in range(4)
        ]
        shape = abs(np.log(max(sides) / max(min(sides), 1e-300)))
        candidates.append((angle + shape, e))
    used = set()
    joined = []
    for _, e in sorted(candidates):
        fs = self._edge_faces.get(e)
        if not fs or used & set(fs):
            continue
        used.update(fs)
        joined.append(self._dissolve_edge(e, check=False))
    return joined


@pytest.mark.parametrize("name,mesh", list(meshes()), ids=[n for n, _ in meshes()])
def test_joining_triangles_into_quads_is_unchanged(name, mesh, monkeypatch):
    joined = EditableMesh.from_mesh(mesh)
    with monkeypatch.context() as m:
        m.setattr(EditableMesh, "_join_pairs", _reference_join_pairs)
        reference = EditableMesh.from_mesh(mesh)
    assert strict_snapshot(joined) == strict_snapshot(reference)
    if not joined.face_count:
        return
    for angle in (0.0, 5.0, 45.0):  # join_triangles with other limits
        a, b = joined.copy(), reference.copy()
        a.triangulate_faces(a.face_ids())
        b.triangulate_faces(b.face_ids())
        a.join_triangles(max_angle=angle)
        with monkeypatch.context() as m:
            m.setattr(EditableMesh, "_join_pairs", _reference_join_pairs)
            b.join_triangles(max_angle=angle)
        assert strict_snapshot(a) == strict_snapshot(b)


def test_cross3_is_np_cross():
    from aether3d.editing._polygon import cross3

    rng = np.random.default_rng(12)
    special = [0.0, -0.0, 1.0, -1.0, 1e-300, -1e300, 3.5, np.inf, -np.inf, np.nan]
    vectors = list(rng.normal(size=(2000, 3)) * rng.choice([1e-8, 1.0, 1e8], (2000, 1)))
    vectors += [np.array(rng.choice(special, 3)) for _ in range(2000)]
    for a, b in zip(vectors, vectors[1:] + vectors[:1], strict=True):
        with np.errstate(invalid="ignore", over="ignore"):
            x, y = cross3(a, b), np.cross(a, b)
        assert x.dtype == y.dtype and x.tobytes() == y.tobytes(), (a, b, x, y)
