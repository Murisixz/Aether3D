"""The camera's lens (0.20.0): lens shift and depth of field on every backend, measured
against the thin-lens law Blender's renders follow (research/2026-10-stage19/blender/
e5_camera: a point at distance d blurs into a disk of (f / N) |1/d - 1/focus|)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from gpu_support import GPU, GPU_PATH

from aether3d import Scene


def stage(**lens):
    scene = Scene()
    glow = scene.create_material("Glow", emissive="#ffffff", emissive_strength=200.0,
                                 base_color="#000000")  # fmt: skip
    dot = scene.add_sphere("Dot", radius=0.01, position=(0, 4, 0))
    dot.material = glow
    cam = scene.add_camera("Cam", position=(0, 0, 0), look_at=(0, 4, 0), fov_axis="horizontal",
                           focal_length=50.0, **lens)  # fmt: skip
    scene.world.ambient_intensity = 0.0
    scene.world.background = "#000000"
    return scene, cam


def blur_extent(result) -> tuple[int, int]:
    img = result.linear[..., 0]
    ys, xs = np.nonzero(img > 0.02 * img.max())
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def expected_diameter(width: int, fstop: float, d: float, focus: float) -> float:
    aperture = 0.050 / fstop  # metres: focal length / f-number
    return aperture * abs(1 / d - 1 / focus) * width / (2 * 0.36)  # tan(hfov / 2) = 18 / 50


@pytest.mark.parametrize(
    "backend", ["path", "cpu", *(["gpu"] if GPU else []), *(["gpu_path"] if GPU_PATH else [])]
)
def test_the_blur_disk_is_the_thin_lens_law(backend):
    scene, _cam = stage(fstop=2.0, focus_distance=0.5)
    options = {"width": 160, "height": 160, "default_lights": False}
    if backend in ("path", "gpu_path"):
        options["path"] = {"samples": 64, "bounces": 0}
    else:
        options["dof_samples"] = 64
    w, h = blur_extent(scene.render(backend=backend, **options))
    want = expected_diameter(160, 2.0, 4.0, 0.5)  # 9.7 px
    assert want - 1.5 <= w <= want + 3.0 and want - 1.5 <= h <= want + 3.0
    in_focus, _ = stage(fstop=2.0, focus_distance=4.0)
    w, _ = blur_extent(in_focus.render(backend=backend, **options))
    assert w <= 3  # the plane in focus stays sharp


def test_apertures_shape_the_bokeh():
    blades, _ = stage(fstop=2.0, focus_distance=0.5, aperture_blades=6)
    ratio, _ = stage(fstop=2.0, focus_distance=0.5, aperture_ratio=2.0)
    opts = {"width": 240, "height": 240, "backend": "path", "default_lights": False,
            "path": {"samples": 128, "bounces": 0}}  # fmt: skip
    w6, h6 = blur_extent(blades.render(**opts))
    assert w6 < h6  # a hexagon with a corner up: narrower across its flats
    wr, hr = blur_extent(ratio.render(**opts))
    assert wr == pytest.approx(hr / 2, abs=2)  # anamorphic: half as wide


def test_lens_shift_lands_where_the_projection_says_on_every_backend():
    scene = Scene()
    scene.add_cube("Box", size=0.4, position=(0.0, 4.0, 0.0))
    cam = scene.add_camera("Cam", position=(0, 0, 0), look_at=(0, 4, 0), focal_length=35.0,
                           fov_axis="auto", shift_x=0.15, shift_y=-0.1)  # fmt: skip
    width, height = 192, 108
    ox, oy = cam.camera.ndc_offset(width / height)
    # The box's centre is on the view axis: shifted, it lands at NDC (-ox, -oy).
    expect = ((1 - ox) * width / 2, (1 + oy) * height / 2)
    backends = ["cpu", "path", *(["gpu"] if GPU else []), *(["gpu_path"] if GPU_PATH else [])]
    for backend in backends:
        opts = {"path": {"samples": 4, "bounces": 0}} if "path" in backend else {}
        r = scene.render(width=width, height=height, backend=backend, **opts)
        ys, xs = np.nonzero(r.object_index >= 0)
        assert xs.mean() + 0.5 == pytest.approx(expect[0], abs=1.0), backend
        assert ys.mean() + 0.5 == pytest.approx(expect[1], abs=1.0), backend
    # Picks agree with what renders: the pixel at the box's image shows the box.
    hit = scene.pick(expect[0], expect[1], width=width, height=height)
    assert hit is not None and hit.object.name == "Box"


def test_focus_objects_set_the_focus_every_frame():
    scene, cam = stage(fstop=2.0)
    target = scene.add_object("Focus", position=(0, 0.5, 0))
    cam.camera.focus_object = target.id
    near_focus = blur_extent(scene.render(width=160, height=160, backend="path",
                                          path={"samples": 64, "bounces": 0},
                                          default_lights=False))[0]  # fmt: skip
    scene.animation.set_channel(target, "transform.position.y", [(0, 0.5), (1, 4.0)])
    sharp = scene.render(width=160, height=160, backend="path", time=1.0,
                         path={"samples": 64, "bounces": 0}, default_lights=False)  # fmt: skip
    assert blur_extent(sharp)[0] <= 3 < near_focus
    gone = scene.render(width=64, height=64, backend="cpu", dof_samples=2, default_lights=False,
                        time=0.0)  # fmt: skip
    assert "DEPTH_OF_FIELD" in {d.code for d in gone.diagnostics}


def test_lens_parameters_animate_and_scripts_set_them():
    from aether3d.rendering.script import run_scene_script

    scene, cam = stage(fstop=2.0, focus_distance=0.5)
    scene.animation.set_channel(cam, "camera.focus_distance", [(0, 0.5), (1, 4.0)])
    opts = {"width": 120, "height": 120, "backend": "path", "default_lights": False,
            "path": {"samples": 48, "bounces": 0}}  # fmt: skip
    assert (
        blur_extent(scene.render(time=0.0, **opts))[0]
        > blur_extent(scene.render(time=1.0, **opts))[0]
    )
    run_scene_script(scene, {
        "format": "aether3d.scene-script", "version": 1,
        "operations": [{"op": "set_camera", "camera": "Cam", "focal_length": 85.0,
                        "focus_object": "Dot", "aperture_blades": 7}],
    })  # fmt: skip
    assert cam.camera.focal_length == pytest.approx(85.0)
    assert cam.camera.focus_object == scene.get_by_name("Dot").id
    assert cam.camera.aperture_blades == 7
    assert math.isfinite(cam.camera.aperture_radius())


def test_shot_presets_frame_the_subject_as_named():
    scene = Scene()
    scene.render_settings.configure(width=320, height=180)
    body = scene.add_cube("Body", size=1.0, position=(2.0, 1.0, 0.9))
    body.scale = (0.5, 0.3, 1.8)  # a person-sized box, 1.8 m tall, facing -Y
    cam = scene.add_camera("Cam", position=(0, -5, 1.6))
    anim = scene.animation
    sizes = {}
    for preset in ("wide", "full", "medium", "close_up"):
        report = anim.frame_shot(cam, body, preset)
        r = scene.render(width=320, height=180, backend="cpu", time=0.0)
        ys, xs = np.nonzero(r.object_index >= 0)
        sizes[preset] = (ys.max() - ys.min() + 1) / 180.0
        assert cam.position[1] < 1.0  # in front of the subject (its -Y side)
        if preset in ("wide", "full"):
            assert sizes[preset] == pytest.approx(report["fill"], abs=0.08)
        assert xs.mean() == pytest.approx(160, abs=4)  # centred by default
    assert sizes["wide"] < sizes["full"] < 1.0
    third = anim.frame_shot(cam, body, "full", screen=[-1 / 3, 0.0], fstop=2.8)
    assert cam.camera.focus_object == body.id and third["depth_of_field"]
    r = scene.render(width=320, height=180, backend="cpu", time=0.0, dof_samples=1)
    ys, xs = np.nonzero(r.object_index >= 0)
    assert (xs.mean() + 0.5) / 320 == pytest.approx(1 / 3, abs=0.03)  # on the left third
    assert [c["name"] for c in anim.constraints(cam)] == ["Shot Aim"]  # replaced, not stacked
    scene.history.undo()
    assert cam.camera.fstop == 0.0
