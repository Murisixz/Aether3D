"""Cameras: parameters, exact projection/view mathematics, aiming, hierarchy and framing."""

import math

import numpy as np
import pytest

from aether3d import Camera, RenderError, Scene, ValidationError
from aether3d.math3d import look_rotation_matrix
from aether3d.rendering.camera import frame_from_matrix
from aether3d.rendering.framing import compute_framing


def project(camera: Camera, aspect: float, view_point) -> np.ndarray:
    clip = camera.projection_matrix(aspect) @ np.append(np.asarray(view_point, float), 1.0)
    return clip[:3] / clip[3]


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


def test_defaults_and_round_trip():
    cam = Camera()
    assert cam.to_dict() == {
        "projection": "perspective",
        "fov": 40.0,
        "fov_axis": "vertical",
        "ortho_scale": 10.0,
        "near": 0.1,
        "far": 1000.0,
        "aspect_ratio": None,
        # the lens (0.20.0): a pinhole by default
        "sensor": 36.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "fstop": 0.0,
        "focus_distance": 10.0,
        "focus_object": None,
        "aperture_blades": 0,
        "aperture_rotation": 0.0,
        "aperture_ratio": 1.0,
    }
    assert Camera.from_dict(cam.to_dict()) == cam and cam.copy() == cam
    old = {
        k: v
        for k, v in cam.to_dict().items()
        if k in ("projection", "fov", "fov_axis", "ortho_scale", "near", "far", "aspect_ratio")
    }
    assert Camera.from_dict(old) == cam  # a document from before the lens fields


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"projection": "fisheye"}, "projection"),
        ({"fov": 0}, "fov"),
        ({"fov": 180}, "fov"),
        ({"fov": float("nan")}, "fov"),
        ({"fov": "wide"}, "fov"),
        ({"fov": True}, "fov"),
        ({"fov_axis": "diagonal"}, "fov_axis"),
        ({"ortho_scale": 0}, "ortho_scale"),
        ({"near": 0}, "near"),
        ({"near": -1}, "near"),
        ({"far": 0.05}, "far"),
        ({"near": 1e-6, "far": 1e4}, "far"),  # far / near beyond 1e9
        ({"aspect_ratio": 0}, "aspect_ratio"),
        ({"aspect_ratio": float("inf")}, "aspect_ratio"),
        ({"sensor": 0}, "sensor"),
        ({"shift_x": 11}, "shift_x"),
        ({"fstop": 0.05}, "fstop"),
        ({"fstop": -1}, "fstop"),
        ({"focus_distance": 0}, "focus_distance"),
        ({"aperture_blades": 2}, "aperture_blades"),
        ({"aperture_blades": 1.5}, "aperture_blades"),
        ({"aperture_ratio": 0}, "aperture_ratio"),
        ({"focus_object": 3}, "focus_object"),
        ({"focal_length": 0}, "focal_length"),
    ],
)
def test_invalid_parameters_are_rejected_atomically(changes, field):
    cam = Camera()
    before = cam.to_dict()
    with pytest.raises(RenderError) as info:
        cam.configure(**changes)
    assert info.value.code == "INVALID_CAMERA"
    assert info.value.location == f"camera.{field}"
    assert cam.to_dict() == before


def test_orthographic_allows_zero_near_and_configure_is_atomic():
    cam = Camera(projection="orthographic", near=0.0, far=50.0)
    assert cam.near == 0.0
    perspective = Camera()
    with pytest.raises(RenderError) as info:
        perspective.far = 0.05  # far < near
    assert "configure" in info.value.hint
    perspective.configure(near=500, far=5000)  # both at once is fine
    assert (perspective.near, perspective.far) == (500.0, 5000.0)
    with pytest.raises(RenderError) as info:
        perspective.configure(zoom=50)
    assert "unknown" in info.value.message


# --------------------------------------------------------------------------- #
# Projection mathematics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("aspect", [16 / 9, 1.0, 0.5])
def test_perspective_matrix_maps_the_frustum_exactly(aspect):
    cam = Camera(fov=60, near=0.5, far=40)
    ty = math.tan(math.radians(30))
    tx = ty * aspect
    for depth in (0.5, 3.0, 40.0):
        top_right = project(cam, aspect, (tx * depth, ty * depth, -depth))
        assert np.allclose(top_right[:2], (1.0, 1.0), atol=1e-12)
    assert np.isclose(project(cam, aspect, (0, 0, -0.5))[2], 0.0, atol=1e-12)
    assert np.isclose(project(cam, aspect, (0, 0, -40.0))[2], 1.0, atol=1e-12)
    # Depth is monotonic and NDC z = f (d - n) / (d (f - n)).
    d = 7.0
    assert np.isclose(project(cam, aspect, (0, 0, -d))[2], 40 * (d - 0.5) / (d * 39.5), atol=1e-12)
    # Perspective foreshortening: x_ndc is inversely proportional to depth.
    near_x = project(cam, aspect, (1, 0, -2))[0]
    far_x = project(cam, aspect, (1, 0, -4))[0]
    assert np.isclose(near_x / far_x, 2.0)


@pytest.mark.parametrize(
    ("axis", "aspect", "horizontal_is_fov"),
    [
        ("vertical", 2.0, False),
        ("horizontal", 2.0, True),
        ("auto", 2.0, True),
        ("auto", 0.5, False),
    ],
)
def test_fov_axis(axis, aspect, horizontal_is_fov):
    cam = Camera(fov=50, fov_axis=axis)
    hx, hy = cam.half_extents(aspect)
    t = math.tan(math.radians(25))
    if horizontal_is_fov:
        assert np.isclose(hx, t) and np.isclose(hy, t / aspect)
    else:
        assert np.isclose(hy, t) and np.isclose(hx, t * aspect)


def test_orthographic_matrix_has_no_foreshortening():
    cam = Camera(projection="orthographic", ortho_scale=8, near=0, far=20)
    a = project(cam, 2.0, (8.0, 4.0, -1.0))
    b = project(cam, 2.0, (8.0, 4.0, -19.0))
    assert np.allclose(a[:2], (1.0, 1.0)) and np.allclose(b[:2], (1.0, 1.0))
    assert np.isclose(a[2], 1 / 20) and np.isclose(b[2], 19 / 20)


# --------------------------------------------------------------------------- #
# Camera frame and aiming
# --------------------------------------------------------------------------- #


def test_unrotated_camera_looks_along_minus_y_with_z_up():
    frame, diags = frame_from_matrix(np.eye(4))
    assert np.allclose(frame.forward, (0, -1, 0)) and np.allclose(frame.up, (0, 0, 1))
    assert np.allclose(frame.right, (-1, 0, 0)) and diags == []
    view = frame.view_matrix()
    # A point straight ahead lands on the -z view axis; world -X is on the right (+x view).
    assert np.allclose(view @ (0, -5, 0, 1), (0, 0, -5, 1))
    assert np.allclose(view @ (-1, -5, 0, 1), (1, 0, -5, 1))


def test_scaled_or_mirrored_cameras_do_not_distort_the_frame():
    world = np.diag([3.0, -2.0, 0.5, 1.0])  # non-uniform and mirrored
    frame, diags = frame_from_matrix(world, "objects[Cam]")
    assert np.allclose(frame.forward, (0, 1, 0)) and np.allclose(frame.up, (0, 0, 1))
    assert np.allclose(np.cross(frame.forward, frame.up), frame.right)
    assert [d.code for d in diags] == ["CAMERA_SCALE_IGNORED"]
    with pytest.raises(RenderError) as info:
        frame_from_matrix(np.diag([1.0, 0.0, 1.0, 1.0]))
    assert info.value.code == "INVALID_CAMERA_TRANSFORM"
    with pytest.raises(RenderError):
        frame_from_matrix(np.full((4, 4), np.nan))


@pytest.mark.parametrize(
    ("direction", "up", "right"),
    [
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),  # front view
        ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),  # straight down: like a top view
        ((0, 0, 1), (0, -1, 0), (1, 0, 0)),  # straight up
    ],
)
def test_look_rotation(direction, up, right):
    rot = look_rotation_matrix(direction)
    assert np.isclose(np.linalg.det(rot), 1.0) and np.allclose(rot.T @ rot, np.eye(3))
    forward, up_vec = -rot[:, 1], rot[:, 2]
    assert np.allclose(forward, direction) and np.allclose(up_vec, up)
    assert np.allclose(np.cross(forward, up_vec), right)
    with pytest.raises(ValidationError):
        look_rotation_matrix((0, 0, 0))


def test_look_at_is_exact_under_scaled_rotated_parents_and_mirrored_objects():
    scene = Scene()
    parent = scene.add_object("Rig", position=(1, 2, 3), scale=(2.0, 0.5, 3.0))
    parent.rotation_euler = (20, -35, 60)
    cam = scene.add_camera("Cam", parent=parent, position=(0.3, -0.2, 0.1))
    cam.scale = (1, -1, 1)  # even a mirrored camera object
    target = np.array([4.0, -6.0, 0.5])
    cam.look_at(target)
    world = cam.world_matrix
    frame, _ = frame_from_matrix(world)
    to_target = target - world[:3, 3]
    assert np.allclose(frame.forward, to_target / np.linalg.norm(to_target), atol=1e-12)
    # Image up is as close to world +Z as possible: it lies in the plane of forward and Z.
    assert abs(np.dot(np.cross(frame.forward, (0, 0, 1)), frame.up)) < 1e-12
    assert frame.up[2] > 0
    # World forward property agrees, and aiming at an object works too.
    assert np.allclose(cam.world_forward, frame.forward)
    other = scene.add_cube("Box", position=(-5, 5, 2))
    cam.look_at(other)
    assert np.allclose(
        cam.world_forward,
        (np.array(other.world_position) - cam.world_position)
        / np.linalg.norm(np.array(other.world_position) - cam.world_position),
    )
    cam.set_world_position((9, 9, 9))
    assert np.allclose(cam.world_position, (9, 9, 9))
    with pytest.raises(ValidationError):
        cam.look_at(cam.world_position)


def test_camera_hierarchy_renders_like_an_equivalent_root_camera():
    def scene_with(camera_setup):
        s = Scene()
        s.add_cube("Cube", size=2).color = "#cc3333"
        s.add_sphere("Ball", radius=0.7, position=(2, 1, 0.5))
        camera_setup(s)
        return s.render(width=96, height=64, samples=1).image

    def nested(s):
        rig = s.add_object("Rig", position=(5, -5, 2), scale=(3, 3, 3))
        rig.rotation_euler = (0, 0, 45)
        cam = s.add_camera("Cam", parent=rig)
        cam.look_at((0, 0, 0))

    def root(s):
        s.add_camera("Cam", position=(5, -5, 2), look_at=(0, 0, 0))

    a, b = scene_with(nested), scene_with(root)
    assert np.count_nonzero(np.any(a != b, axis=-1)) <= 2  # identical up to rounding at edges


# --------------------------------------------------------------------------- #
# Active camera
# --------------------------------------------------------------------------- #


def test_active_camera_selection_and_errors():
    scene = Scene()
    scene.add_cube("Cube")
    with pytest.raises(RenderError) as info:
        scene.render(width=8, height=8)
    assert info.value.code == "NO_CAMERA" and "add_camera" in info.value.hint
    first = scene.add_camera("First", position=(0, -5, 0))
    second = scene.add_camera("Second", position=(0, 5, 0), look_at=(0, 0, 0))
    assert scene.active_camera is first and scene.cameras == (first, second)
    third = scene.add_camera("Third", active=True, position=(5, 0, 0), look_at=(0, 0, 0))
    assert scene.active_camera is third
    scene.active_camera = "Second"
    assert scene.active_camera is second
    with pytest.raises(ValidationError):
        scene.active_camera = "Cube"
    scene.remove(second)
    assert scene.active_camera is None
    assert scene.render(camera="Third", width=8, height=8).camera["name"] == "Third"
    with pytest.raises(RenderError) as info:
        scene.render(camera="Cube", width=8, height=8)
    assert info.value.code == "NOT_A_CAMERA"
    with pytest.raises(RenderError) as info:
        scene.render(camera="Missing", width=8, height=8)
    assert info.value.code == "OBJECT_NOT_FOUND"
    # A camera object cannot also hold geometry.
    with pytest.raises(ValidationError):
        third.mesh = scene.get_by_name("Cube").mesh


def test_invalid_camera_arguments_create_nothing():
    scene = Scene()
    with pytest.raises(RenderError):
        scene.add_camera("Bad", fov=500)
    with pytest.raises(ValidationError):
        scene.add_camera("Bad", position=(0, 0, 0), look_at=(0, 0, 0))
    assert len(scene) == 0 and scene.active_camera is None


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def _ndc(scene, cam, points, aspect):
    frame, _ = frame_from_matrix(cam.world_matrix)
    view = frame.view_matrix()
    clip = (cam.camera.projection_matrix(aspect) @ view @ np.c_[points, np.ones(len(points))].T).T
    return clip[:, :3] / clip[:, 3:4]


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
@pytest.mark.parametrize("aspect", [16 / 9, 0.6])
def test_framing_fits_all_vertices_tightly(projection, aspect):
    scene = Scene()
    scene.add_cube("A", size=2, position=(3, 1, 1))
    scene.add_sphere("B", radius=1.5, position=(-4, 2, 0.5))
    cube = scene.add_cube("C", size=1, position=(0, -3, 4))
    cube.rotation_euler = (30, 40, 50)
    cam = scene.add_camera("Cam", projection=projection, fov=35)
    framing = cam.frame(direction=(-0.5, 1.0, -0.4), margin=0.1, aspect=aspect)
    points = np.concatenate(
        [
            o.mesh.positions.astype(float) @ o.world_matrix[:3, :3].T + o.world_matrix[:3, 3]
            for o in scene
            if o.mesh is not None
        ]
    )
    ndc = _ndc(scene, cam, points, aspect)
    assert np.all(np.abs(ndc[:, :2]) <= 0.9 + 1e-9)  # inside the frame minus the margin
    assert np.all((ndc[:, 2] >= 0) & (ndc[:, 2] <= 1))  # within near/far
    # Tight: the extreme vertices touch the margin on both sides of at least one axis.
    extent = [ndc[:, i].max() for i in (0, 1)] + [-ndc[:, i].min() for i in (0, 1)]
    assert max(extent) == pytest.approx(0.9, abs=1e-9)
    tight_axis = 0 if np.isclose(extent[0], 0.9) or np.isclose(extent[2], 0.9) else 1
    assert ndc[:, tight_axis].max() == pytest.approx(0.9, abs=1e-6)
    assert -ndc[:, tight_axis].min() == pytest.approx(0.9, abs=1e-6)
    assert np.allclose(
        cam.world_forward, np.array([-0.5, 1.0, -0.4]) / np.linalg.norm([-0.5, 1, -0.4])
    )
    assert framing.distance > 0


def test_compute_framing_is_minimal_along_the_view_direction():
    cam = Camera(fov=40)
    frame, _ = frame_from_matrix(np.eye(4))
    points = np.random.default_rng(5).uniform(-3, 3, (200, 3))
    framing = compute_framing(cam, frame, points, 1.5, margin=0.0, fit_clipping=False)
    # Moving the camera any closer pushes some vertex out of the frame.
    for shift in (1e-6, 1e-3):
        closer = framing.position + shift * frame.forward
        rel = points - closer
        x, y, z = rel @ frame.right, rel @ frame.up, rel @ frame.forward
        hx, hy = cam.half_extents(1.5)
        assert np.any((np.abs(x) > z * hx + 1e-12) | (np.abs(y) > z * hy + 1e-12))


def test_framing_errors_change_nothing():
    scene = Scene()
    cam = scene.add_camera("Cam", position=(1, 2, 3))
    before = (cam.transform.copy(), cam.camera.to_dict())
    with pytest.raises(RenderError) as info:
        cam.frame()
    assert info.value.code == "NOTHING_TO_FRAME"
    with pytest.raises(RenderError):
        scene.add_cube("Cube")
        cam.frame(margin=2.0)
    assert cam.transform == before[0] and cam.camera.to_dict() == before[1]
    with pytest.raises(RenderError) as info:
        scene.get_by_name("Cube").frame()
    assert info.value.code == "NOT_A_CAMERA"


# --------------------------------------------------------------------------- #
# The lens (0.20.0)
# --------------------------------------------------------------------------- #


def test_focal_length_sensor_and_aperture():
    cam = Camera(focal_length=50.0, fov_axis="horizontal")
    assert cam.fov == pytest.approx(2 * math.degrees(math.atan(18.0 / 50.0)))  # 39.6
    assert cam.focal_length == pytest.approx(50.0)
    cam.configure(sensor=24.0)  # a smaller sensor at the same field of view: a shorter lens
    assert cam.focal_length == pytest.approx(50.0 * 24.0 / 36.0)
    cam.configure(focal_length=85.0, sensor=36.0)
    assert cam.focal_length == pytest.approx(85.0)
    assert cam.aperture_radius() == 0.0 and not cam.has_depth_of_field
    cam.configure(fstop=2.0)
    assert cam.aperture_radius() == pytest.approx(0.085 / 4.0)  # f / (2 N), metres
    ortho = Camera(projection="orthographic", fstop=2.0)
    assert ortho.focal_length is None and ortho.aperture_radius() == 0.0


def test_lens_shift_is_an_off_centre_frustum_in_units_of_the_larger_dimension():
    # Measured in Blender: shift 0.1 moves the image by a tenth of its larger dimension.
    for aspect in (16 / 9, 9 / 16):
        plain = Camera(focal_length=50.0, fov_axis="auto")
        shifted = Camera(focal_length=50.0, fov_axis="auto", shift_x=0.1, shift_y=-0.05)
        p = (0.3, 0.2, -4.0)
        a, b = project(plain, aspect, p), project(shifted, aspect, p)
        larger_x = 2.0 * max(1.0, 1.0 / aspect)  # NDC width of the larger dimension
        larger_y = 2.0 * max(1.0, aspect)
        assert b[0] - a[0] == pytest.approx(-0.1 * larger_x)
        assert b[1] - a[1] == pytest.approx(0.05 * larger_y)
        assert b[2] == pytest.approx(a[2])  # depth unchanged
        assert shifted.ndc_offset(aspect) == pytest.approx((0.1 * larger_x, -0.05 * larger_y))
    ortho = Camera(projection="orthographic", shift_x=0.25)
    assert project(ortho, 1.0, (0, 0, -3))[0] == pytest.approx(-0.5)


def test_aperture_points_are_uniform_round_or_polygonal():
    from aether3d.rendering.camera import aperture_points

    rng = np.random.default_rng(3)
    u1, u2 = rng.random(200_000), rng.random(200_000)
    x, y = aperture_points(u1, u2, 0, 0.0, 1.0)
    r2 = x * x + y * y
    assert r2.max() <= 1.0 + 1e-12
    assert np.mean(r2 < 0.25) == pytest.approx(0.25, abs=0.005)  # uniform by area
    x, y = aperture_points(u1, u2, 6, 0.0, 1.0)
    assert y.max() == pytest.approx(1.0, abs=0.01)  # a corner up (corners are thin)
    assert x.max() == pytest.approx(math.cos(math.radians(30)), abs=2e-3)  # a flat side
    x, y = aperture_points(u1, u2, 0, 0.0, 2.0)
    assert x.max() == pytest.approx(0.5, abs=2e-3) and y.max() == pytest.approx(1.0, abs=2e-3)
    x, y = aperture_points(u1, u2, 5, 36.0, 1.0)  # turned by 36 degrees: a corner down
    assert y.min() == pytest.approx(-1.0, abs=0.01)
