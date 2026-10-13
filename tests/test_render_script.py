"""Declarative scene-setup scripts: every operation, validation and exact rollback."""

import json

import numpy as np
import pytest

from aether3d import Scene, ScriptError, run_scene_script

FORMAT = {"format": "aether3d.scene-script", "version": 1}


def script(*operations):
    return {**FORMAT, "operations": list(operations)}


def base_scene() -> Scene:
    scene = Scene("Script")
    scene.add_cube("Crate", size=2, position=(0, 0, 1))
    scene.add_plane("Ground", size=20)
    return scene


SETUP = [
    {"op": "add_camera", "name": "Main", "position": [6, -8, 5], "look_at": [0, 0, 1], "fov": 45},
    {
        "op": "add_camera",
        "name": "Top",
        "projection": "orthographic",
        "ortho_scale": 10,
        "position": [0, 0, 15],
        "rotation_euler": [0, 0, 0],
        "near": 0,
        "far": 40,
        "active": False,
    },
    {
        "op": "add_light",
        "type": "directional",
        "name": "Sun",
        "direction": [-0.3, 0.6, -1],
        "intensity": 2,
        "color": "#fff2d6",
    },
    {
        "op": "add_light",
        "type": "point",
        "name": "Bulb",
        "position": [2, -2, 3],
        "intensity": 6,
        "range": 12,
    },
    {"op": "set_object", "object": "Crate", "color": "#c0392b", "specular": 0.3},
    {"op": "set_object", "object": "Top", "look_along": [0, 0, -1]},
    {"op": "set_world", "background": "#8ec5f2", "ambient_intensity": 0.25},
    {"op": "set_render_settings", "width": 160, "height": 90, "samples": 1},
    {"op": "info"},
]


def test_setup_script_builds_a_renderable_scene():
    scene = base_scene()
    result = run_scene_script(scene, script(*SETUP))
    assert [r["op"] for r in result["operations"]] == [op["op"] for op in SETUP]
    info = result["operations"][-1]
    assert info["active_camera"]["name"] == "Main"
    assert [c["name"] for c in info["cameras"]] == ["Main", "Top"]
    assert info["lights"][1]["range"] == 12.0
    assert scene.get_by_name("Crate").color == pytest.approx((0xC0 / 255, 0x39 / 255, 0x2B / 255))
    image = scene.render()
    assert (image.width, image.height) == (160, 90) and image.stats["lights"] == 2
    top = scene.render(camera="Top")
    assert top.stats["pixels_covered"] == 160 * 90  # the 20 x 20 ground fills the 17.8 x 10 view


def test_frame_camera_and_add_camera_framing():
    scene = base_scene()
    result = run_scene_script(
        scene,
        script(
            {
                "op": "add_camera",
                "name": "Auto",
                "frame": True,
                "frame_direction": [-1, 1, -0.5],
                "frame_targets": ["Crate"],
                "frame_margin": 0.1,
            },
            {"op": "frame_camera", "camera": "Auto", "targets": ["Crate", "Ground"]},
            {"op": "frame_camera"},  # the active camera (Auto)
        ),
    )
    assert "framing" in result["operations"][0] and "framing" in result["operations"][1]
    assert scene.render(width=48, height=27).stats["pixels_covered"] > 0


def snapshot(scene: Scene) -> str:
    doc = scene.to_dict()
    return json.dumps(doc, sort_keys=True)


@pytest.mark.parametrize(
    ("operations", "location", "code"),
    [
        (
            [{"op": "set_camera", "camera": "Main", "fov": 500}],
            "operations[1]: objects[Main].camera.fov",
            "INVALID_CAMERA",
        ),
        (
            [{"op": "set_light", "light": "Sun", "intensity": -2}],
            "operations[1]: objects[Sun].light.intensity",
            "INVALID_LIGHT",
        ),
        (
            [{"op": "set_object", "object": "Crate", "color": "pink"}],
            "operations[1]",
            "INVALID_APPEARANCE",
        ),
        ([{"op": "set_camera", "camera": "Crate", "fov": 30}], "operations[1]", "NOT_A_CAMERA"),
        ([{"op": "set_active_camera", "camera": "Sun"}], "operations[1]", "INVALID_ARGUMENT"),
        ([{"op": "set_render_settings", "samples": 9}], "operations[1]", "INVALID_SETTINGS"),
        (
            [{"op": "add_camera", "position": [1, 1, 1], "look_at": [1, 1, 1]}],
            "operations[1]",
            "INVALID_ARGUMENT",
        ),
        (
            [{"op": "frame_camera", "camera": "Main", "targets": ["Missing"]}],
            "operations[1]",
            "OBJECT_NOT_FOUND",
        ),
        ([{"op": "teleport"}], "operations[1].op", "UNKNOWN_OPERATION"),
        ([{"op": "set_world", "fog": 1}], "operations[1]", "INVALID_SCRIPT"),
    ],
)
def test_failures_roll_back_everything(operations, location, code):
    scene = base_scene()
    run_scene_script(scene, script(*SETUP))
    camera_handle = scene.get_by_name("Main").camera
    transform_handle = scene.get_by_name("Crate").transform
    before = snapshot(scene)
    with pytest.raises(ScriptError) as info:
        run_scene_script(
            scene,
            script(
                # A change that must be undone too: a new object.
                {"op": "add_light", "type": "point", "name": "Extra", "position": [0, 0, 5]},
                *operations,
            ),
        )
    assert info.value.code == code and info.value.location.startswith(location), (
        info.value.code,
        info.value.location,
    )
    assert snapshot(scene) == before
    assert scene.get_by_name("Main").camera is camera_handle  # handles stay valid
    assert scene.get_by_name("Crate").transform is transform_handle


def test_rollback_restores_removed_subtrees_values_and_active_camera():
    scene = base_scene()
    run_scene_script(scene, script(*SETUP))
    rig = scene.add_object("Rig")
    child = scene.add_cube("Child", parent=rig)
    before = snapshot(scene)
    order = [o.name for o in scene]
    with pytest.raises(ScriptError):
        run_scene_script(
            scene,
            script(
                {"op": "remove", "object": "Rig"},
                {"op": "remove", "object": "Main"},  # the active camera
                {"op": "set_object", "object": "Crate", "position": [5, 5, 5], "visible": False},
                {"op": "set_camera", "camera": "Top", "ortho_scale": 3},
                {"op": "set_world", "background": "#000000"},
                {"op": "set_render_settings", "width": 999},
                {"op": "set_light", "light": "Sun", "type": "point"},
                {"op": "remove", "object": "Missing"},
            ),
        )
    assert snapshot(scene) == before and [o.name for o in scene] == order
    assert scene.active_camera.name == "Main" and child.parent is rig and child.scene is scene


def test_removal_and_visibility_operations():
    scene = base_scene()
    run_scene_script(scene, script(*SETUP))
    result = run_scene_script(
        scene,
        script(
            {"op": "remove", "object": "Main"},
            {"op": "set_active_camera", "camera": "Top"},
            {"op": "set_object", "object": "Ground", "visible": False},
        ),
    )
    assert scene.active_camera.name == "Top" and len(result["operations"][0]["removed"]) == 1
    r = scene.render()
    assert [o["name"] for o in r.objects] == ["Crate"]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {"format": "aether3d.edit-script", "version": 1, "operations": [{"op": "info"}]},
        {**FORMAT, "version": 2, "operations": [{"op": "info"}]},
        {**FORMAT, "operations": []},
        {**FORMAT, "operations": ["info"]},
        {**FORMAT, "operations": [{"op": "info"}], "x": 1},
    ],
)
def test_malformed_scripts(bad):
    with pytest.raises(ScriptError):
        run_scene_script(Scene(), bad)


def test_scripts_are_data_not_code():
    scene = base_scene()
    with pytest.raises(ScriptError):
        run_scene_script(
            scene,
            script(
                {"op": "set_object", "object": "Crate", "color": "__import__('os').system('true')"}
            ),
        )
    assert np.isfinite(scene.render(width=8, height=8, frame=True).linear).all()
