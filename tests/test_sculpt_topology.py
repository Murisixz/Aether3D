"""Dynamic topology and isotropic remeshing: validity after every change, exact attribute
interpolation on splits, features kept, quality reached, undo exact."""

import numpy as np
import pytest

from aether3d import Mesh, SculptError, create_cube, create_sphere
from aether3d.sculpt import SculptMesh, check_mesh
from aether3d.sculpt.mesh import sculpt_operation
from aether3d.sculpt.topology import build_region, split_pass
from aether3d.sculpt.validate import weld


def plain_sphere(segments=24, rings=12) -> Mesh:
    base = create_sphere(1.0, segments, rings)
    p, t = weld(base.positions, base.triangles)
    return Mesh(p, t)


def angles(mesh: Mesh) -> np.ndarray:
    p = mesh.positions.astype(np.float64)
    t = mesh.triangles
    out = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = p[t[:, j]] - p[t[:, i]], p[t[:, k]] - p[t[:, i]]
        cos = (u * v).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.stack(out, axis=1)


def valid(sm: SculptMesh) -> dict:
    assert sm._consistency_errors() == []
    out = sm.to_mesh()
    report = check_mesh(out.positions, out.triangles, self_intersections=True)
    s = report.summary
    assert report.ok and s["degenerate_triangle_count"] == 0
    assert s["edge_manifold"] and s["vertex_manifold"] and s["consistently_oriented"]
    assert s["self_intersecting_pairs"] == 0
    return s


def state(sm: SculptMesh) -> tuple:
    n, t, r = sm._nv, sm._nt, sm._nr
    return (
        n,
        t,
        r,
        sm._pos[:n].tobytes(),
        sm._valive[:n].tobytes(),
        sm._talive[:t].tobytes(),
        sm._tri[:t].tobytes(),
        sm._trv[:t].tobytes(),
        sm._vsum[:n].tobytes(),
        sm._dirty[:n].tobytes(),
        tuple(c.tobytes() for c in sm._chunks),
        sm._chunk_lo.tobytes(),
        None if sm._ruv is None else sm._ruv[:r].tobytes(),
    )


class _Splitter(SculptMesh):
    """Test-only: one operation that only splits (every vertex active)."""

    @sculpt_operation
    def split_all(self, limit):
        done = 0
        for _ in range(8):
            region = build_region(self, self.triangle_ids(), self.vertex_ids())
            n = split_pass(self, region, limit, 10**7)
            done += n
            if not n:
                break
        return {"splits": done}


def test_splits_interpolate_uvs_exactly_and_keep_the_surface():
    source = create_sphere(1.0, 16, 8)
    sm = _Splitter.from_mesh(source)
    r = sm.split_all(0.15)
    assert r.data["splits"] > 0
    s = valid(sm)
    assert s["closed"] and s["euler_characteristic"] == 2
    out = sm.to_mesh()
    # Every new vertex lies on an original edge, at its midpoint, with the UV halfway
    # between that edge's UVs in the same face: the texture mapping is unchanged. Check it
    # by locating each output triangle's centroid in the source's faces.
    from aether3d.spatial.surface import TriangleSurface

    src = TriangleSurface(source.positions, source.triangles)
    centroid = out.positions[out.triangles].mean(axis=1)
    near = src.nearest(centroid)
    assert near.distance.max() < 1e-6  # still on the faceted source surface
    st = source.triangles[near.triangles]
    for corner in range(3):
        p = out.positions[out.triangles[:, corner]].astype(np.float64)
        a, b, c = (source.positions[st[:, k]].astype(np.float64) for k in range(3))
        # Barycentric coordinates of the corner in its centroid's source triangle.
        v0, v1, v2 = b - a, c - a, p - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        bu = 1 - bv - bw
        uv = (
            bu[:, None] * source.uvs[st[:, 0]]
            + bv[:, None] * source.uvs[st[:, 1]]
            + bw[:, None] * source.uvs[st[:, 2]]
        )
        assert np.abs(uv - out.uvs[out.triangles[:, corner]]).max() < 1e-5


def test_remesh_reaches_the_target_with_good_triangles():
    sm = SculptMesh.from_mesh(plain_sphere(48, 24))
    r = sm.remesh(0.08, iterations=6)
    s = valid(sm)
    assert s["closed"] and s["euler_characteristic"] == 2
    e = r.data["edge_length"]
    assert 0.07 < e["mean"] < 0.085 and e["std"] / e["mean"] < 0.2
    a = angles(sm.to_mesh())
    assert a.min() > 20 and (a.min(axis=1) < 30).mean() < 0.01
    assert s["volume"] == pytest.approx(4.18879, rel=0.01)  # the sphere's volume


def test_remesh_coarsens_dense_meshes():
    sm = SculptMesh.from_mesh(plain_sphere(96, 48))
    before = sm.triangle_count
    r = sm.remesh(0.2, iterations=5)
    assert r.data["triangles"] < before / 4 and r.data["collapses"] > 0
    assert valid(sm)["closed"]


def test_remesh_keeps_hard_edges_and_corners():
    sm = SculptMesh.from_mesh(create_cube(2))
    sm.remesh(0.25)
    out = sm.to_mesh()
    s = valid(sm)
    assert s["volume"] == pytest.approx(8.0)
    p = out.positions.astype(np.float64)
    assert np.allclose(np.abs(p).max(axis=1), 1.0)  # every vertex on the cube's surface
    corners = {tuple(c) for c in np.round(p[np.all(np.abs(p) > 0.999, axis=1)], 6)}
    assert len(corners) == 8


def test_remesh_keeps_uv_seams_and_face_group_borders():
    source = create_sphere(1.0, 32, 16)
    groups = (source.positions[source.triangles].mean(axis=1)[:, 2] > 0.3).astype(np.int64)
    mesh = Mesh(
        source.positions, source.triangles, normals=source.normals, uvs=source.uvs, groups=groups
    )
    sm = SculptMesh.from_mesh(mesh)
    sm.remesh(0.12, iterations=4)
    out = sm.to_mesh()
    valid(sm)
    # No triangle spans the UV seam (the u = 0 / u = 1 cut stays a cut).
    uv = out.uvs[out.triangles]
    assert (np.ptp(uv[:, :, 0], axis=1) < 0.5).all()
    # The border between the two groups still runs along z = cos(theta) of a ring.
    border_z = []
    p = out.positions.astype(np.float64)
    t = out.triangles
    g = out.groups
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    owner = np.tile(np.arange(len(t)), 3)
    key = np.sort(e, axis=1)
    pw, pt = weld(p, t)
    del pw, pt
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks, os_ = key[order], owner[order]
    same = np.all(ks[1:] == ks[:-1], axis=1)
    for i in np.flatnonzero(same):
        if g[os_[i]] != g[os_[i + 1]]:
            border_z += p[ks[i], 2].tolist()
    assert border_z and np.ptp(border_z) < 0.25


def test_masked_regions_keep_their_topology():
    sm = SculptMesh.from_mesh(plain_sphere(32, 16))
    sm.set_mask(1.0, box=((-2, -2, -2), (2, 2, 0)))
    tris_before = {tuple(sorted(x)) for x in sm.triangles().tolist() if True}
    lower = {t for t in tris_before if all(sm._pos[v, 2] < -0.1 for v in t)}
    sm.remesh(0.08, iterations=3)
    after = {tuple(sorted(x)) for x in sm.triangles().tolist()}
    assert lower <= after
    valid(sm)


def test_dyntopo_strokes_refine_locally_and_undo_exactly():
    sm = SculptMesh.from_mesh(create_sphere(1.0, 24, 12))
    start = state(sm)
    count = sm.triangle_count
    r = sm.stroke(
        {
            "brush": {
                "kind": "draw",
                "radius": 0.4,
                "strength": 0.1,
                "dyntopo": {"detail": 0.05, "mode": "constant"},
            },
            "path": [[0, 0, 1], [0.5, 0, 0.85]],
        }
    )
    topo = r.data["topology"]
    assert topo["splits"] > 0 and r.data["triangles"] > count
    s = valid(sm)
    assert s["closed"]
    # Refined near the stroke only: far triangles keep their size.
    out = sm.to_mesh()
    c = out.positions[out.triangles].mean(axis=1)
    far = c[:, 2] < -0.5
    assert (
        far.sum()
        == (
            create_sphere(1.0, 24, 12)
            .positions[create_sphere(1.0, 24, 12).triangles]
            .mean(axis=1)[:, 2]
            < -0.5
        ).sum()
    )
    done = state(sm)
    sm.undo()
    assert state(sm) == start
    sm.redo()
    assert state(sm) == done


def test_dyntopo_settings_are_checked_and_growth_is_bounded():
    from aether3d import ValidationError

    for bad in (
        {"mode": "relative"},
        {"detail": 0, "mode": "constant"},
        {"detail": 2.0, "mode": "relative"},
        {"detail": 0.1, "refine": "everything"},
        {"detail": 0.1, "unknown": 1},
    ):
        with pytest.raises(ValidationError):
            SculptMesh.from_mesh(plain_sphere()).stroke(
                {"brush": {"kind": "draw", "radius": 0.3, "dyntopo": bad}, "path": [[0, 0, 1]]}
            )
    with pytest.raises(ValidationError):
        SculptMesh.from_mesh(plain_sphere()).stroke(
            {
                "brush": {"kind": "grab", "radius": 0.3, "dyntopo": {"detail": 0.1}},
                "path": [[0, 0, 1]],
            }
        )
    sm = SculptMesh.from_mesh(plain_sphere())
    with pytest.raises(SculptError, match="REMESH_TOO_DENSE"):
        sm.remesh(1e-4)
    assert not sm.has_changes


def test_remesh_is_deterministic():
    a = SculptMesh.from_mesh(plain_sphere(32, 16))
    b = SculptMesh.from_mesh(plain_sphere(32, 16))
    a.remesh(0.1, iterations=3)
    b.remesh(0.1, iterations=3)
    assert a.to_mesh() == b.to_mesh()
