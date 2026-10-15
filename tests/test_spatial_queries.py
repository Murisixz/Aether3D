"""Stage 10 spatial queries against brute force and analytic truth (aether3d.spatial)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from spatial_helpers import ref_closest_on_triangle, ref_triangle_meets_box, ref_triangles_intersect

import aether3d as a3
from aether3d import Scene
from aether3d.errors import QueryError
from aether3d.procedural import Expr
from aether3d.spatial import Box, Sphere

# --------------------------------------------------------------------------- #
# Fixtures and brute-force references
# --------------------------------------------------------------------------- #


def tricky_scene() -> Scene:
    """Hierarchy, rotations, non-uniform and mirrored scales, shared meshes (instances), a
    procedural object with a modifier, a hidden object, a degenerate triangle, an open mesh."""
    s = Scene("Tricky")
    base = s.add_cube(name="Base", size=2)
    base.rotation_euler = (10, 20, 30)
    base.scale = (1.5, 0.7, 1.2)
    child = s.add_sphere(name="Child", radius=0.6, parent=base, position=(0, 0, 1.8))
    child.scale = (1.0, 2.0, 0.5)
    mirror = s.add_cube(name="Mirror", size=1, position=(3, 1, 0.5))
    mirror.scale = (-1.0, 1.3, 0.8)
    copy = s.duplicate("Mirror", "Copy")  # shares Mirror's mesh
    copy.position = (-3, -1, 0.2)
    copy.rotation_euler = (0, 0, 45)
    row = s.add_geometry("Row", "box", width=0.5, depth=0.5, height=0.5)
    row.add_modifier("array", count=3, relative_offset=(1.5, 0, 0))
    proc = s.add_object("Row", geometry=row, position=(-1, 3, 0))
    proc.rotation_euler = (0, 0, -20)
    hidden = s.add_cube(name="Hidden", size=1, position=(0, -3, 0))
    hidden.visible = False
    s.add_plane(name="Floor", size=12, position=(0, 0, -1.5))
    sliver = a3.Mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0]],
        [[0, 1, 2], [1, 3, 4]],  # the second triangle is degenerate (collinear)
    )
    s.add_object("Sliver", mesh=sliver, position=(1, -2, 2))
    return s


def world_triangles(scene: Scene, include_hidden=False):
    """Every triangle in world space with its object name and triangle index (float64)."""
    out = []
    for obj in scene:
        if obj.mesh is None or (not include_hidden and not obj.visible):
            continue
        m = obj.world_matrix
        v = obj.mesh.positions.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        t = obj.mesh.triangles
        for k in range(t.shape[0]):
            out.append((obj.name, k, v[t[k, 0]], v[t[k, 1]], v[t[k, 2]]))
    return out


def brute_ray(tris, o, d):
    """All (distance, object, triangle) hits of a ray, Moller-Trumbore in float64."""
    d = d / np.linalg.norm(d)
    hits = []
    for name, k, a, b, c in tris:
        e1, e2 = b - a, c - a
        h = np.cross(d, e2)
        det = e1 @ h
        if abs(det) < 1e-14:
            continue
        f = 1.0 / det
        s = o - a
        u = f * (s @ h)
        q = np.cross(s, e1)
        v = f * (d @ q)
        t = f * (e2 @ q)
        if u >= -1e-12 and v >= -1e-12 and u + v <= 1 + 1e-12 and t > 1e-12:
            hits.append((t, name, k))
    return sorted(hits)


def brute_closest(tris, p):
    best = None
    for name, k, a, b, c in tris:
        q = ref_closest_on_triangle(p, a, b, c)
        d = float(np.linalg.norm(q - p))
        if best is None or d < best[0] - 1e-12:
            best = (d, name, k, q)
    return best


def random_rays(rng, n, center=(0, 0, 0), radius=6.0):
    origins = rng.normal(size=(n, 3))
    origins = origins / np.linalg.norm(origins, axis=1, keepdims=True) * radius + center
    targets = rng.uniform(-2.5, 2.5, size=(n, 3)) + center
    return origins, targets - origins


# --------------------------------------------------------------------------- #
# Rays
# --------------------------------------------------------------------------- #


def test_raycasts_match_brute_force_through_every_transform():
    s = tricky_scene()
    tris = world_triangles(s)
    rng = np.random.default_rng(1)
    origins, directions = random_rays(rng, 150)
    hits_seen = 0
    for o, d in zip(origins, directions, strict=True):
        want = brute_ray(tris, o, d)
        got = s.raycast_all(o, d)
        assert [(h.object.name, h.triangle) for h in got] == [(w[1], w[2]) for w in want]
        assert np.allclose([h.distance for h in got], [w[0] for w in want], atol=1e-9)
        first = s.raycast(o, d)
        if want:
            hits_seen += 1
            assert first == got[0]
            unit = d / np.linalg.norm(d)
            assert np.allclose(first.position, o + first.distance * unit, atol=1e-9)
        else:
            assert first is None
    assert hits_seen > 60


def test_ray_hits_carry_normals_uvs_materials_and_instances():
    s = tricky_scene()
    brick = s.create_material("Brick", base_color="#a04030")
    s.get_by_name("Mirror").material = brick
    hit = s.raycast((3, 1, 5), (0, 0, -1))  # straight down onto Mirror's top
    assert hit.object.name == "Mirror" and hit.front_face is True
    assert np.allclose(hit.normal, (0, 0, 1), atol=1e-12)  # outward despite the mirroring
    assert np.allclose(hit.shading_normal, (0, 0, 1), atol=1e-6)
    assert hit.material.name == "Brick" and hit.uv is not None
    assert hit.instance_of.name == "Mirror" and hit.instance_count == 2
    copy_hit = s.raycast((-3, -1, 5), (0, 0, -1))
    assert copy_hit.object.name == "Copy" and copy_hit.instance_of.name == "Mirror"
    assert copy_hit.material is None or copy_hit.material.name == "Brick"
    # From inside a closed mesh the ray meets the back of a face.
    inside = s.raycast((3, 1, 0.5), (0, 0, 1))
    assert inside.object.name == "Mirror" and inside.front_face is False
    data = json.loads(json.dumps(hit.to_dict()))
    assert data["object"]["name"] == "Mirror" and len(data["barycentric"]) == 3


def test_rays_through_shared_edges_and_vertices_hit_once():
    s = Scene()
    s.add_cube(name="Cube", size=2)  # each face is split along a diagonal through its center
    hits = s.raycast_all((0, 0, 5), (0, 0, -1))  # through both diagonals (top and bottom)
    assert [(h.object.name, h.distance) for h in hits] == [("Cube", 4.0), ("Cube", 6.0)]
    corner = s.raycast_all((3, 3, 3), (-1, -1, -1))  # through two corners (shared vertices)
    assert [h.object.name for h in corner] == ["Cube", "Cube"]
    assert np.isclose(corner[0].distance, math.sqrt(12)) and np.isclose(
        corner[1].distance, math.sqrt(48)
    )
    # The first hit is the lowest triangle index among those meeting at the point.
    tri = s.get_by_name("Cube").mesh.triangles
    first = corner[0]
    touching = [
        k
        for k in range(tri.shape[0])
        if np.any(np.all(s.get_by_name("Cube").mesh.positions[tri[k]] == (1, 1, 1), axis=1))
    ]
    assert first.triangle == min(touching)


def test_grazing_and_near_parallel_rays():
    s = Scene()
    s.add_cube(name="Cube", size=2)
    assert s.raycast((-5, 0, 1.0 + 1e-9), (1, 0, 0)) is None  # just above the top face
    below = s.raycast_all((-5, 0, 1.0 - 1e-9), (1, 0, 0))  # just below: through the sides
    assert [round(h.distance, 6) for h in below] == [4.0, 6.0]
    along = s.raycast_all((-5, 0, 1.0), (1, 0, 0))  # in the top face's plane: side edges
    assert [round(h.distance, 6) for h in along] == [4.0, 6.0]
    tilted = s.raycast((-5, 0, 1.0), (1, 0, -1e-12))
    assert tilted is not None and np.isclose(tilted.distance, 4.0)


def test_ray_ranges_selection_and_hidden_objects():
    s = tricky_scene()
    down = ((0, -3, 5), (0, 0, -1))
    assert s.raycast(*down).object.name == "Floor"  # Hidden is skipped
    assert s.raycast(*down, include_hidden=True).object.name == "Hidden"
    assert s.raycast(*down, exclude="Floor") is None
    assert s.raycast(*down, objects=["Floor"]).object.name == "Floor"
    assert s.raycast(*down, max_distance=6.0) is None  # the floor is 6.5 away
    assert s.raycast(*down, max_distance=6.6).object.name == "Floor"
    hits = s.raycast_all((0, 0, 8), (0, 0, -1), include_hidden=True)
    assert s.raycast_all((0, 0, 8), (0, 0, -1), include_hidden=True, limit=2) == hits[:2]
    assert s.raycast_all((0, 0, 8), (0, 0, -1), min_distance=hits[0].distance)[0] == hits[1]
    with pytest.raises(QueryError) as exc:
        s.raycast((0, 0, 0), (0, 0, 0))
    assert exc.value.code == "INVALID_ARGUMENT" and exc.value.location == "direction"
    with pytest.raises(QueryError) as exc:
        s.raycast((0, 0, 0), (0, 0, 1), exclude="Nope")
    assert exc.value.code == "OBJECT_NOT_FOUND"


def test_rendered_surfaces_pass_through_alpha_masks_and_single_sided_backs(tmp_path):
    from aether3d.rendering.png import write_png

    s = Scene()
    floor = s.add_plane(name="Floor", size=6, position=(0, 0, -1))
    screen = s.add_plane(name="Screen", size=2)
    mask = np.zeros((2, 2, 4), dtype=np.uint8)
    mask[..., :3] = 255
    mask[0, :, 3] = 255  # top half (v near 1 in the image's first row) opaque
    write_png(tmp_path / "mask.png", mask)
    lace = s.create_material("Lace", base_color="#ffffff", alpha_mode="mask", alpha_cutoff=0.5)
    lace.set_texture(
        "base_color",
        s.add_image_texture("Mask", tmp_path / "mask.png", filter="nearest", mipmaps=False),
    )
    screen.material = lace
    hits = {}
    for y in (-0.5, 0.5):
        geo = s.raycast((0, y, 3), (0, 0, -1))
        ren = s.raycast((0, y, 3), (0, 0, -1), surfaces="rendered")
        hits[y] = (geo.object.name, ren.object.name)
    assert hits[-0.5][0] == hits[0.5][0] == "Screen"  # geometry: always the screen
    assert {hits[-0.5][1], hits[0.5][1]} == {"Screen", "Floor"}  # rendered: one half is cut out
    all_rendered = s.raycast_all((0, 0.5, 3), (0, 0, -1), surfaces="rendered")
    all_geometry = s.raycast_all((0, 0.5, 3), (0, 0, -1))
    assert len(all_geometry) == 2 and len(all_rendered) in (1, 2)
    # A single-sided plane seen from behind does not render.
    floor.material = s.create_material("Front", base_color="#808080", double_sided=False)
    up = s.raycast((2.5, 0, -3), (0, 0, 1), surfaces="rendered")
    assert up is None
    assert s.raycast((2.5, 0, -3), (0, 0, 1)).object.name == "Floor"
    with pytest.raises(QueryError):
        s.raycast((0, 0, 3), (0, 0, -1), surfaces="rendered", include_hidden=True)


# --------------------------------------------------------------------------- #
# Nearest points and distances
# --------------------------------------------------------------------------- #


def test_closest_points_match_brute_force():
    s = tricky_scene()
    tris = world_triangles(s)
    rng = np.random.default_rng(2)
    for p in rng.uniform(-5, 5, size=(60, 3)):
        want = brute_closest(tris, p)
        got = s.closest_point(p)
        assert np.isclose(got.distance, want[0], atol=1e-9)
        assert np.isclose(np.linalg.norm(np.array(got.position) - p), got.distance)
        if got.object.name != want[1]:  # only on exact ties between objects
            assert np.isclose(got.distance, want[0], atol=1e-12)


def test_nearest_objects_order_and_limits():
    s = tricky_scene()
    p = (0.5, 0.5, 4.0)
    every = s.nearest_objects(p)
    tris = world_triangles(s)
    per_object = {}
    for name, k, a, b, c in tris:
        per_object.setdefault(name, []).append((name, k, a, b, c))
    expected = sorted((brute_closest(t, np.array(p))[0], n) for n, t in per_object.items())
    assert [h.object.name for h in every] == [n for _, n in expected]
    assert np.allclose([h.distance for h in every], [d for d, _ in expected], atol=1e-9)
    assert s.nearest_objects(p, count=2) == every[:2]
    limit = every[2].distance
    assert [h.object.name for h in s.nearest_objects(p, max_distance=limit)] == [
        h.object.name for h in every[:3]
    ]
    assert s.closest_point(p, max_distance=every[0].distance * 0.5) is None


def test_object_distances_match_brute_force_pairs():
    s = Scene()
    a = s.add_sphere(name="A", radius=1.0, segments=12, rings=6)
    b = s.add_cube(name="B", size=1.0, position=(2.2, 0.3, 0.1))
    b.rotation_euler = (15, 25, 35)
    result = s.distance("A", "B")
    ta = world_triangles(s)
    pa = [t for t in ta if t[0] == "A"]
    pb = [t for t in ta if t[0] == "B"]
    from aether3d.spatial import geometry as geo

    a1 = np.array([t[2] for t in pa for _ in pb])
    b1 = np.array([t[3] for t in pa for _ in pb])
    c1 = np.array([t[4] for t in pa for _ in pb])
    a2 = np.array([t[2] for _ in pa for t in pb])
    b2 = np.array([t[3] for _ in pa for t in pb])
    c2 = np.array([t[4] for _ in pa for t in pb])
    brute, _, _ = geo.triangle_distances(a1, b1, c1, a2, b2, c2)
    assert np.isclose(result.distance, brute.min(), atol=1e-12)
    assert not result.intersecting and result.distance > 0.1
    assert np.isclose(np.linalg.norm(np.subtract(result.point_a, result.point_b)), result.distance)
    b.position = (1.2, 0.3, 0.1)  # now they cross
    crossing = s.distance(a, b)
    assert crossing.intersecting and crossing.distance == 0.0
    assert crossing.point_a == crossing.point_b
    assert any(
        ref_triangles_intersect(np.array(x[2:]), np.array(y[2:]))
        for x in world_triangles(s)
        if x[0] == "A"
        for y in world_triangles(s)
        if y[0] == "B"
    )
    with pytest.raises(QueryError):
        s.distance("A", "A")


# --------------------------------------------------------------------------- #
# Containment
# --------------------------------------------------------------------------- #


def test_containment_of_closed_mirrored_and_open_meshes():
    s = tricky_scene()
    mirror = s.get_by_name("Mirror")
    inside = s.contains_point(mirror, (3, 1, 0.5))
    assert inside.state == "inside" and inside.inside is True and np.isclose(inside.winding, 1)
    assert s.contains_point(mirror, (3, 1, 3)).state == "outside"
    on = s.contains_point(mirror, (3, 1, 0.5 + 0.4))  # its top face (half height 0.4)
    assert on.state == "surface" and on.distance < 1e-6
    sphere_center = s.get_by_name("Child").world_position
    assert s.contains_point("Child", sphere_center).state == "inside"
    floor = s.contains_point("Floor", (0, 0, 0))
    assert floor.state == "undefined" and floor.inside is None and not floor.closed
    # In front of an open surface the winding number is a negative fraction.
    assert "open" in floor.reason and -0.5 < floor.winding < 0.0
    assert s.contains_point("Row", s.get_by_name("Row").world_position).state == "inside"
    with pytest.raises(QueryError) as exc:
        s.contains_point(s.add_camera("Cam"), (0, 0, 0))
    assert exc.value.code == "NO_GEOMETRY"


def test_containment_of_non_manifold_and_inconsistent_meshes():
    s = Scene()
    fan = a3.Mesh(  # three triangles on one edge: non-manifold
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [-1, 0, 0], [0, 1, 0]],
        [[0, 1, 2], [1, 0, 3], [0, 1, 4]],
    )
    s.add_object("Fan", mesh=fan)
    result = s.contains_point("Fan", (0.1, 0.1, 0.5))
    assert result.state == "undefined" and "non-manifold" in result.reason
    cube = a3.create_cube(1.0)
    flipped = cube.triangles.copy()
    flipped[0] = flipped[0, ::-1]  # one triangle wound the other way
    s.add_object("Bad", mesh=a3.Mesh(cube.positions, flipped))
    assert s.contains_point("Bad", (0, 0, 0)).state == "undefined"


# --------------------------------------------------------------------------- #
# Overlaps and regions
# --------------------------------------------------------------------------- #


def test_overlaps_are_exact_and_never_just_boxes():
    s = Scene()
    s.add_sphere(name="A", radius=1.0)
    s.add_sphere(name="B", radius=1.0, position=(1.6, 1.6, 0))  # boxes overlap, spheres do not
    s.add_cube(name="Big", size=4, position=(10, 0, 0))
    s.add_cube(name="Small", size=1, position=(10, 0, 0))  # inside Big
    s.add_cube(name="Touching", size=2, position=(10, 0, 3))  # rests on Big's top face
    exact = {(o.a.name, o.b.name): o for o in s.overlaps()}
    assert ("A", "B") not in exact
    assert (
        exact[("Big", "Small")].contained == "b_in_a" and not exact[("Big", "Small")].intersecting
    )
    assert exact[("Big", "Touching")].intersecting is True
    loose = {(o.a.name, o.b.name): o for o in s.overlaps(exact=False)}
    assert loose[("A", "B")].bounds_overlap and loose[("A", "B")].intersecting is None
    assert not loose[("A", "B")].exact
    near = {(o.a.name, o.b.name) for o in s.overlaps(tolerance=0.3)}
    assert ("A", "B") in near  # the gap is 2 * 1.6 * sqrt(2) / 2 ... about 0.26
    assert [(o.a.name, o.b.name) for o in s.overlaps("Small", against=["Big", "A"])] == [
        ("Big", "Small")
    ]


def test_region_queries_match_brute_force():
    s = tricky_scene()
    region = Box((-1.0, -1.5, -2.0), (2.0, 1.0, 0.5))
    got = {m.object.name: m.relation for m in s.query_region(region)}
    lo, hi = np.array(region.min), np.array(region.max)
    for obj in s:
        if obj.mesh is None or not obj.visible:
            continue
        tris = [t for t in world_triangles(s) if t[0] == obj.name]
        pts = np.array([v for t in tris for v in t[2:]])
        inside = bool(np.all((pts >= lo) & (pts <= hi)))
        meets = any(ref_triangle_meets_box(t[2:], lo, hi) for t in tris)
        if inside:
            assert got.get(obj.name) == "inside"
        elif meets:
            assert got.get(obj.name) == "intersects"
        else:
            assert got.get(obj.name) in (None, "contains")
    assert "Hidden" not in got
    inside_only = {m.object.name for m in s.query_region(region, mode="inside")}
    assert inside_only == {n for n, r in got.items() if r == "inside"}
    bounds_only = s.query_region(region, mode="bounds")
    assert all(not m.exact and m.relation == "bounds" for m in bounds_only)
    assert set(got) <= {m.object.name for m in bounds_only}


def test_region_queries_with_spheres_points_and_enclosing_solids():
    s = Scene()
    s.add_cube(name="Room", size=10)
    s.add_sphere(name="Ball", radius=0.5, position=(2, 0, 0))
    s.add_point_light(name="Lamp", position=(2.2, 0, 0.3))
    s.add_camera("Cam", position=(-20, 0, 0))
    got = {m.object.name: m.relation for m in s.query_region(Sphere((2, 0, 0), 1.0))}
    assert got == {"Room": "contains", "Ball": "inside", "Lamp": "inside"}
    edge = {m.object.name: m.relation for m in s.query_region(Sphere((2.9, 0, 0), 0.5))}
    assert edge["Ball"] == "intersects" and edge["Room"] == "contains"
    assert s.query_region(Sphere((30, 0, 0), 1.0)) == []
    with pytest.raises(QueryError):
        s.query_region("everywhere")
    with pytest.raises(QueryError):
        Box((1, 0, 0), (0, 1, 1))


# --------------------------------------------------------------------------- #
# Bounds and discovery
# --------------------------------------------------------------------------- #


def test_bounds_are_exact_in_world_and_local_space():
    s = tricky_scene()
    base = s.get_by_name("Base")
    for obj in s:
        if obj.mesh is None:
            continue
        b = obj.bounds()
        pts = np.array([v for t in world_triangles(s, True) if t[0] == obj.name for v in t[2:]])
        assert np.allclose(b.aabb.min, pts.min(axis=0)) and np.allclose(b.aabb.max, pts.max(axis=0))
        assert np.all(np.linalg.norm(pts - b.sphere.center, axis=1) <= b.sphere.radius + 1e-9)
        corners = b.obb.corners()  # the oriented box encloses every point
        center = np.asarray(b.obb.center)
        axes = np.asarray(b.obb.half_axes)
        coords = (np.linalg.pinv(axes.T) @ (pts - center).T).T  # flat meshes: a zero axis
        assert np.allclose(coords @ axes + center, pts, atol=1e-9)
        assert np.all(np.abs(coords) <= 1 + 1e-9) and corners.shape == (8, 3)
    local = base.bounds(space="local")
    assert local.aabb.min == (-1.0, -1.0, -1.0) and local.aabb.max == (1.0, 1.0, 1.0)
    assert base.bounds().obb.orthogonal is True
    tree = base.bounds(children=True)
    child = s.get_by_name("Child").bounds()
    assert tree.objects == 2 and tree.aabb == base.bounds().aabb.union(child.aabb)
    assert tree.triangles == base.mesh.triangle_count + s.get_by_name("Child").mesh.triangle_count
    whole = s.bounds()
    assert whole.object is None and whole.objects == 7  # Hidden left out
    assert s.bounds(include_hidden=True).objects == 8
    assert s.bounds(s.add_camera("Cam")) is None
    assert Scene().bounds() is None
    with pytest.raises(QueryError):
        s.bounds(space="local")


def test_find_objects_filters():
    s = tricky_scene()
    s.get_by_name("Base").tags = ["castle", "stone"]
    s.get_by_name("Child").tags = ["castle"]
    s.get_by_name("Mirror").metadata = {"floors": 3, "open": True}
    s.get_by_name("Copy").metadata = {"floors": 3.0, "open": 1}
    brick = s.create_material("Brick")
    s.get_by_name("Floor").material = brick

    def names(**filters):
        return [o.name for o in s.find_objects(**filters)]

    assert names(name="Mirror") == ["Mirror"]
    assert names(name="*i*") == ["Child", "Mirror", "Hidden", "Sliver"]
    assert names(kind="procedural") == ["Row"]
    assert "Row" in names(kind="geometry") and names(kind=["camera", "light"]) == []
    assert names(tags="castle") == ["Base", "Child"]
    assert names(tags=["castle", "stone"]) == ["Base"]
    assert names(any_tags=["stone", "nope"]) == ["Base"]
    assert names(metadata={"floors": 3}) == ["Mirror", "Copy"]  # 3 == 3.0
    assert names(metadata={"open": True}) == ["Mirror"]  # but true is not 1
    assert names(metadata_keys=["open"]) == ["Mirror", "Copy"]
    assert names(parent="Base") == ["Child"] and names(ancestor="Base") == ["Child"]
    assert names(visible=False) == ["Hidden"]
    assert names(material="Brick") == ["Floor"]
    assert names(region=Sphere((3, 1, 0.5), 0.1)) == ["Mirror"]
    assert names(kind="mesh", visible=True, region=Box((4, 4, -2), (20, 20, -1))) == ["Floor"]
    with pytest.raises(QueryError) as exc:
        s.find_objects(kind="tree")
    assert exc.value.location == "kind"
    with pytest.raises(QueryError):
        s.find_objects(material="Nope")


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def camera_scene() -> Scene:
    s = Scene("View")
    s.add_plane(name="Floor", size=20)
    s.add_cube(name="Crate", size=1, position=(0, 0, 0.5))
    s.add_sphere(name="Ball", radius=0.5, position=(2.5, 1, 0.5))
    s.add_cube(name="Behind", size=1, position=(0, 30, 0.5))  # outside the far plane
    s.add_cube(name="Back", size=1, position=(0, -12, 0.5))  # behind the camera
    s.add_camera("Cam", position=(0, -8, 3), look_at=(0, 0, 0.5), fov=40, far=25)
    s.add_directional_light("Sun", direction=(-0.3, 0.4, -1))
    return s


def test_camera_rays_and_projection_are_inverse():
    s = camera_scene()
    for x, y in ((0.5, 0.5), (320.0, 180.0), (17.25, 300.75), (639.5, 359.5)):
        ray = s.camera_ray(x, y, width=640, height=360)
        point = np.add(ray.origin, np.multiply(ray.direction, 7.0))
        screen = s.project(point, width=640, height=360)
        assert np.isclose(screen.x, x) and np.isclose(screen.y, y) and screen.in_view
    behind = s.project((0, -20, 3), width=640, height=360)
    assert behind.x is None and behind.depth < 0 and not behind.in_view
    many = s.project([(0, 0, 0.5), (0, 40, 0)], width=640, height=360)
    assert [p.in_view for p in many] == [True, False]  # the second is beyond the far plane
    with pytest.raises(QueryError) as exc:
        Scene().camera_ray(1, 1)
    assert exc.value.code == "NO_CAMERA"


def test_camera_rays_are_the_path_tracers_rays_bit_for_bit():
    from aether3d.rendering.pathtrace.integrator import _camera
    from aether3d.rendering.scene_data import extract_render_scene

    s = camera_scene()
    s.get_by_name("Cam").set_parent(s.get_by_name("Crate"), keep_world_transform=True)
    for projection in ("perspective", "orthographic"):
        s.get_by_name("Cam").camera.configure(projection=projection)
        rs = extract_render_scene(s, s.active_camera, aspect=80 / 60)
        traced = _camera(rs, 80, 60, 1)
        for col, row in ((0, 0), (39, 29), (79, 59), (12, 47)):
            o, d, tmin, tmax, _ = traced.rays(
                np.array([col]), np.array([row]), np.array([0.5]), np.array([0.5])
            )
            ray = s.camera_ray(col + 0.5, row + 0.5, width=80, height=60)
            assert ray.origin == tuple(o[0]) and ray.direction == tuple(d[0])
            assert (ray.min_distance, ray.max_distance) == (tmin[0], tmax[0])


def test_picking_matches_the_rasterizers_object_ids():
    s = camera_scene()
    width, height = 96, 54
    result = s.render(width=width, height=height, samples=1)
    ids = result.object_index
    names = {o["index"]: o["name"] for o in result.objects}
    checked = 0
    for row in range(1, height - 1, 3):
        for col in range(1, width - 1, 3):
            block = ids[row - 1 : row + 2, col - 1 : col + 2]
            if np.any(block != block[0, 0]):
                continue  # silhouettes: sampling positions differ slightly
            hit = s.pick(col + 0.5, row + 0.5, width=width, height=height)
            expected = names.get(int(ids[row, col]))
            assert (hit.object.name if hit else None) == expected, (col, row)
            checked += 1
    assert checked > 200


def test_objects_in_view_agree_with_renders():
    s = camera_scene()
    view = {m.object.name: m for m in s.objects_in_view(width=320, height=180)}
    assert set(view) == {"Floor", "Crate", "Ball"}
    assert view["Crate"].visibility == "inside" and view["Floor"].visibility == "partial"
    rendered = s.render(width=320, height=180, samples=1)
    seen = {o["name"] for o in rendered.objects if o["visible_samples"] > 0}
    assert seen <= set(view)
    loose = {m.object.name for m in s.objects_in_view(width=320, height=180, exact=False)}
    assert set(view) <= loose
    cam = s.get_by_name("Cam")
    cam.camera.configure(projection="orthographic", ortho_scale=2.0)
    ortho = {m.object.name for m in s.objects_in_view(width=320, height=180)}
    assert "Crate" in ortho and "Ball" not in ortho  # 2 x 3.6 units: the ball is outside


# --------------------------------------------------------------------------- #
# Lines of sight, visibility, relations
# --------------------------------------------------------------------------- #


def test_line_of_sight_and_visibility():
    s = Scene()
    s.add_cube(name="Wall", size=1, position=(0, 0, 1))
    s.get_by_name("Wall").scale = (4, 0.2, 2)
    target = s.add_sphere(name="Target", radius=0.5, position=(0, 3, 1))
    blocked = s.line_of_sight((0, -3, 1), (0, 3, 1))
    assert not blocked.clear and blocked.blocker.object.name == "Wall"
    assert s.line_of_sight((0, -3, 1), (0, 3, 1), exclude="Wall").clear is False  # the target
    assert s.line_of_sight((0, -3, 1), target).clear is False  # an object end never blocks
    assert s.line_of_sight((0, -3, 3), (0, 3, 3)).clear  # over the wall
    assert s.line_of_sight((0, -3, 1), (0, -3, 1)).clear
    # On the target's surface: the surface itself does not count as a blocker.
    top = s.raycast((0, 3, 5), (0, 0, -1)).position
    assert s.line_of_sight((0, 3, 5), top).clear
    far = s.visible_from("Target", (0, 3, 50), samples=256)
    assert far.blockers == ((far.target, far.samples - far.visible),)  # its own far side
    cap = (1 - 0.5 / 49) / 2  # the visible fraction of a sphere's area from distance d
    assert abs(far.fraction - cap) < 0.08
    hidden = s.visible_from("Target", (0, -3, 1), samples=64)
    assert hidden.visible == 0 and dict((o.name, n) for o, n in hidden.blockers)["Wall"] > 0
    cam = s.add_camera("Cam", position=(0, -3, 1), look_at=(0, 3, 1))
    behind = s.visible_from("Target", cam, samples=32)
    assert behind.visible == 0
    away = s.add_camera("Away", position=(0, 10, 1), look_at=(0, 20, 1))
    out = s.visible_from("Target", away, samples=32)
    assert out.visible == 0 and out.outside_view == 32


def test_relations():
    s = Scene()
    s.add_cube(name="Table", size=2, position=(0, 0, 1))
    s.add_cube(name="Cup", size=0.5, position=(0.3, 0.2, 2.25))  # resting on the table
    s.add_cube(name="Box", size=6, position=(0, 0, 1))
    s.add_cube(name="Far", size=1, position=(10, 0, 0.5))
    s.add_cube(name="Front", size=1, position=(0, -5, 1))
    cup = s.relation("Cup", "Table", near_distance=0.1)
    assert cup.above and not cup.below and cup.intersecting and cup.near and cup.distance == 0.0
    assert s.relation("Table", "Cup").below
    inner = s.relation("Table", "Box")
    assert inner.inside is True and inner.contains is False and not inner.intersecting
    assert s.relation("Box", "Table").contains is True
    far = s.relation("Far", "Table", near_distance=1.0)
    assert not far.near and not far.above and np.isclose(far.distance, 8.5)
    # Objects face -Y: Front is in front of the table, the table behind it.
    assert s.relation("Front", "Table").in_front_of and s.relation("Table", "Front").behind
    viewed = s.relation("Front", "Table", viewer=(0, -20, 1))
    assert viewed.viewer == "a_nearer"
    cam = s.add_camera("Cam", position=(0, 20, 1), look_at=(0, 0, 1))
    assert s.relation("Front", "Table", viewer=cam).viewer == "b_nearer"
    open_floor = s.add_plane(name="Floor", size=30)
    assert s.relation("Table", "Floor").inside is None  # a plane bounds no solid
    data = json.loads(json.dumps(s.relation("Cup", open_floor).to_dict()))
    assert data["above"] is True


# --------------------------------------------------------------------------- #
# Degenerate inputs, empty scenes, errors
# --------------------------------------------------------------------------- #


def test_empty_scenes_and_degenerate_geometry():
    empty = Scene()
    assert (
        empty.raycast((0, 0, 0), (1, 0, 0)) is None
        and empty.raycast_all((0, 0, 0), (1, 0, 0)) == []
    )
    assert empty.closest_point((0, 0, 0)) is None and empty.nearest_objects((0, 0, 0)) == []
    assert empty.overlaps() == [] and empty.query_region(Box((0, 0, 0), (1, 1, 1))) == []
    assert empty.describe()["objects"] == [] and empty.describe()["bounds"] is None
    s = Scene()
    flat = s.add_cube(name="Flat", size=1)
    flat.scale = (1, 0, 1)  # collapsed to a square: zero volume
    assert s.raycast((0, 5, 0), (0, -1, 0)) is None  # singular objects cannot be hit
    near = s.closest_point((0, 3, 0))
    assert near.object.name == "Flat" and np.isclose(near.distance, 3.0)
    assert s.bounds(flat).aabb.size == (1.0, 0.0, 1.0)
    point = a3.Mesh([[0, 0, 0]] * 3, [[0, 1, 2]])  # one triangle collapsed to a point
    s.add_object("Dot", mesh=point, position=(5, 0, 0))
    assert s.raycast((5, 5, 0), (0, -1, 0)) is None
    assert np.isclose(s.closest_point((5, 1, 0), objects="Dot").distance, 1.0)


def test_failing_procedural_geometry_is_a_query_error():
    from aether3d.procedural.modifiers import MODIFIER_TYPES

    s = Scene()
    bad = s.add_geometry("Bad", "box")
    s.add_object("Bad", geometry=bad)
    original = type(MODIFIER_TYPES["mirror"]).apply
    bad.add_modifier("mirror")
    s.procedural.clear_cache()
    try:
        type(MODIFIER_TYPES["mirror"]).apply = lambda *a: (_ for _ in ()).throw(RuntimeError("x"))
        with pytest.raises(QueryError) as exc:
            s.raycast((0, 0, 5), (0, 0, -1))
    finally:
        type(MODIFIER_TYPES["mirror"]).apply = original
    assert exc.value.code == "GEOMETRY_EVALUATION_FAILED" and exc.value.location == "objects[Bad]"


def test_query_errors_carry_codes_locations_and_hints():
    s = tricky_scene()
    cases = [
        (lambda: s.raycast((0, 0), (0, 0, 1)), "INVALID_ARGUMENT", "origin"),
        (
            lambda: s.raycast((0, 0, 0), (0, 0, 1), max_distance=-1),
            "INVALID_ARGUMENT",
            "max_distance",
        ),
        (
            lambda: s.raycast((0, 0, 0), (0, 0, 1), surfaces="pretty"),
            "INVALID_ARGUMENT",
            "surfaces",
        ),
        (
            lambda: s.closest_point((0, 0, 0), objects=["Base", "Nope"]),
            "OBJECT_NOT_FOUND",
            "objects",
        ),
        (lambda: s.contains_point("Nope", (0, 0, 0)), "OBJECT_NOT_FOUND", "object"),
        (lambda: s.visible_from("Base", (0, 0, 9), samples=0), "INVALID_ARGUMENT", "samples"),
        (lambda: s.pick(1, 1, camera="Base"), "NOT_A_CAMERA", "camera"),
        (lambda: s.overlaps(tolerance=math.nan), "INVALID_ARGUMENT", "tolerance"),
    ]
    for call, code, location in cases:
        with pytest.raises(QueryError) as exc:
            call()
        assert (exc.value.code, exc.value.location) == (code, location)
        assert isinstance(exc.value, ValueError) and exc.value.to_dict()["code"] == code


# --------------------------------------------------------------------------- #
# Determinism, caching and history
# --------------------------------------------------------------------------- #


def every_query(s: Scene) -> list:
    """One of each query, as JSON (for determinism and no-mutation checks); the scene has a
    camera named QueryCam."""
    out = [
        s.bounds(),
        s.bounds("Base", children=True),
        s.find_objects(kind="mesh"),
        s.raycast((0, 0, 8), (0.1, 0.05, -1)),
        s.raycast_all((0, 0, 8), (0.1, 0.05, -1)),
        s.closest_point((4, 4, 4)),
        s.nearest_objects((1, 1, 1), count=3),
        s.contains_point("Mirror", (3, 1, 0.5)),
        s.distance("Base", "Mirror"),
        s.overlaps(),
        s.query_region(Box((-2, -2, -2), (2, 2, 2))),
        s.camera_ray(10, 10, camera="QueryCam", width=64, height=48),
        s.pick(32, 24, camera="QueryCam", width=64, height=48),
        s.project([(0, 0, 0), (1, 2, 3)], camera="QueryCam", width=64, height=48),
        s.objects_in_view(camera="QueryCam", width=64, height=48),
        s.line_of_sight((0, -8, 1), (0, 8, 1)),
        s.visible_from("Base", (0, -8, 3), samples=16),
        s.relation("Child", "Base", viewer="QueryCam"),
        s.describe(),
    ]
    from aether3d.spatial.request import to_json

    return json.loads(json.dumps(to_json(out)))


def test_queries_are_deterministic_and_change_nothing():
    s = tricky_scene()
    s.add_camera("QueryCam", position=(0, -12, 4), look_at=(0, 0, 0))
    document = json.dumps(s.to_dict(), sort_keys=True)
    revision, steps = s.revision, len(s.history)
    first = every_query(s)
    assert (s.revision, len(s.history)) == (revision, steps)
    assert json.dumps(s.to_dict(), sort_keys=True) == document
    assert every_query(s) == first
    # A fresh copy of the scene (new caches) answers identically (but has its own revision).
    copy = Scene.from_dict(s.to_dict())
    again = every_query(copy)
    assert again[-1].pop("revision") == 0 and first[-1].pop("revision") == revision
    assert again == first


def test_the_index_is_cached_per_revision_and_follows_undo_redo_and_rollback():
    from aether3d.spatial import get_index

    s = Scene()
    cube = s.add_cube(name="Cube", size=2)
    index = get_index(s)
    assert get_index(s) is index  # unchanged scene: reused
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 4.0
    cube.position = (0, 0, 1)
    assert get_index(s) is not index
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 3.0
    s.history.undo()
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 4.0
    s.history.redo()
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 3.0
    assert not s.history.in_progress
    with s.transaction("Move"):
        assert s.history.in_progress
        cube.position = (0, 0, 2)
        assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 2.0  # sees the open change
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 2.0
    with pytest.raises(RuntimeError), s.transaction("Abandoned"):
        cube.position = (0, 0, -3)
        assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 7.0
        raise RuntimeError("roll back")
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 2.0  # rolled back
    cube.visible = False
    assert s.raycast((0, 0, 5), (0, 0, -1)) is None
    s.history.undo()
    with cube.edit_mesh() as edit:
        edit.translate(edit.select_all(), (0, 0, 1))
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 1.0
    s.history.undo()
    assert s.raycast((0, 0, 5), (0, 0, -1)).distance == 2.0


def test_procedural_parameter_changes_reach_queries():
    s = Scene()
    s.procedural.add_parameter("count", 2, min=1, max=6)
    row = s.add_geometry("Row", "box", width=1, depth=1, height=1)
    row.add_modifier("array", count=Expr("count"), relative_offset=(2, 0, 0))
    s.add_object("Row", geometry=row)
    assert s.raycast((4, 0, 5), (0, 0, -1)) is None
    s.procedural.set_parameter("count", 3)
    assert s.raycast((4, 0, 5), (0, 0, -1)).object.name == "Row"
    s.history.undo()
    assert s.raycast((4, 0, 5), (0, 0, -1)) is None


def test_describe_is_compact_and_complete():
    s = tricky_scene()
    s.get_by_name("Base").tags = ["castle"]
    data = s.describe()
    assert data["format"] == "aether3d.spatial-scene" and data["version"] == 1
    names = [o["name"] for o in data["objects"]]
    assert names == [o.name for o in s]
    base = data["objects"][0]
    assert base["tags"] == ["castle"] and base["closed"] is True and "metadata" not in base
    assert next(o for o in data["objects"] if o["name"] == "Hidden")["visible"] is False
    assert next(o for o in data["objects"] if o["name"] == "Floor")["closed"] is False
    assert len(s.describe(include_hidden=False)["objects"]) == len(names) - 1
    assert data["frame"]["up"] == [0.0, 0.0, 1.0]
