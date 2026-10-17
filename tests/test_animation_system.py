"""The animation document: keyframing, NLA composition, history, persistence, and
evaluation inside the scene (renders, the retained world, spatial queries, observations).

Expected values are closed forms of the keys and strips set up here, and invariants:
evaluating never changes the scene, rendering at a time equals writing the pose into the
scene and rendering that, and the retained render world equals the full extraction.
"""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest
from parity import assert_render_scenes_equal

from aether3d import Scene
from aether3d.animation import rotation as R
from aether3d.errors import AnimationError, ProjectFormatError
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import _extract_full, extract_render_scene
from aether3d.serialization import scene_from_dict


def quat_z(deg: float) -> list[float]:
    h = math.radians(deg) / 2
    return [0.0, 0.0, math.sin(h), math.cos(h)]


def euler_z(q) -> float:
    return float(R.to_euler(np.asarray(q))[0][2])


@pytest.fixture
def scene() -> Scene:
    s = Scene("Anim")
    root = s.add_object("Root")
    s.add_cube(name="Cube", parent=root, position=(0, 0, 1))
    s.add_sphere(name="Ball", position=(3, 0, 1))
    s.add_plane(name="Ground", size=20)
    cam = s.add_camera(name="Cam", position=(8, -8, 5), look_at=(0, 0, 0))
    s.active_camera = cam
    s.add_point_light(name="Lamp", position=(2, 2, 6))
    return s


# --------------------------------------------------------------------------------------- #
# Keyframing and history
# --------------------------------------------------------------------------------------- #


def test_keyframing_writes_actions_and_never_the_authored_scene(scene):
    a = scene.animation
    cube = scene.get_by_name("Cube")
    a.keyframe(cube, "transform.position", 0.0, (0, 0, 1), interpolation="linear")
    a.keyframe(cube, "transform.position", 2.0, (4, 2, 1), interpolation="linear")
    pose = a.evaluate(0.5)
    k = pose.ids.index(cube.id)
    assert pose.position[k].tolist() == [1.0, 0.5, 1.0]
    assert cube.position == (0.0, 0.0, 1.0)  # authored: the rest pose
    clip = a.action(cube)
    assert clip.name == "Cube Action" and set(clip.paths) == {
        "transform.position.x",
        "transform.position.y",
        "transform.position.z",
    }
    # Each keyframe call is one undo step; undo restores the definitions exactly.
    assert len(scene.history.entries) >= 2
    before = a.channel(cube, "transform.position.x")
    a.keyframe(cube, "transform.position.x", 1.0, 9.0)
    scene.history.undo()
    assert a.channel(cube, "transform.position.x") is before
    scene.history.redo()
    assert a.channel(cube, "transform.position.x").values.tolist() == [0.0, 9.0, 4.0]


def test_evaluation_changes_nothing_and_does_not_depend_on_order(scene):
    a = scene.animation
    cube = scene.get_by_name("Cube")
    for t, z in ((0, 0), (1, 90), (3, -45)):
        a.keyframe(cube, "transform.rotation", t, quat_z(z))
    doc = json.dumps(scene.to_dict(), sort_keys=True)
    revision = scene.revision
    entries = len(scene.history.entries)
    times = [2.2, 0.1, 1.7, 2.2, 0.0, 3.5, 0.1]
    first = {t: a.evaluate(t) for t in times}
    for t in reversed(times):
        p = a.evaluate(t)
        assert p.rotation.tobytes() == first[t].rotation.tobytes()
    fresh = Scene.from_dict(json.loads(doc)).animation
    for t in times:
        assert fresh.evaluate(t).rotation.tobytes() == first[t].rotation.tobytes()
    assert json.dumps(scene.to_dict(), sort_keys=True) == doc
    assert scene.revision == revision and len(scene.history.entries) == entries


def test_moving_the_current_time_is_not_an_edit(scene):
    a = scene.animation
    a.keyframe("Cube", "transform.position.z", 0.0, 1.0)
    a.keyframe("Cube", "transform.position.z", 1.0, 3.0)
    revision, entries = scene.revision, len(scene.history.entries)
    doc = scene.to_dict()
    a.time = 0.75
    a.frame = 12
    assert a.time == pytest.approx(0.5)
    assert scene.revision == revision and len(scene.history.entries) == entries
    assert scene.to_dict() == doc  # not saved either
    with pytest.raises(AnimationError):
        a.time = float("inf")


def test_grouped_bool_color_camera_light_and_material_channels(scene):
    a = scene.animation
    ball = scene.get_by_name("Ball")
    a.keyframe(ball, "visible", 0.0, True)
    a.keyframe(ball, "visible", 1.0, False)
    a.keyframe(ball, "appearance.color", 0.0, (1, 0, 0), interpolation="linear")
    a.keyframe(ball, "appearance.color", 2.0, (0, 0, 1), interpolation="linear")
    a.keyframe("Cam", "camera.fov", 0.0, 50.0, interpolation="linear")
    a.keyframe("Cam", "camera.fov", 2.0, 30.0, interpolation="linear")
    a.keyframe("Lamp", "light.intensity", 0.0, 1.0, interpolation="linear")
    a.keyframe("Lamp", "light.intensity", 2.0, 3.0, interpolation="linear")
    steel = scene.create_material("Steel", base_color=(0.5, 0.5, 0.5))
    a.keyframe(steel, "roughness", 0.0, 0.2, interpolation="linear")
    a.keyframe(steel, "roughness", 2.0, 0.8, interpolation="linear")
    p = a.evaluate(1.5)
    assert p.visible[ball.id] is False and a.evaluate(0.5).visible[ball.id] is True
    assert p.fields[(ball.id, "appearance")]["color"] == pytest.approx([0.25, 0.0, 0.75])
    assert p.fields[(scene.get_by_name("Cam").id, "camera")]["fov"] == pytest.approx(35.0)
    assert p.fields[(scene.get_by_name("Lamp").id, "light")]["intensity"] == pytest.approx(2.5)
    assert p.materials[steel.id]["roughness"] == pytest.approx(0.65)
    assert steel.roughness == 0.5  # untouched


def test_out_of_range_values_are_clamped_and_inapplicable_channels_reported(scene):
    a = scene.animation
    a.keyframe("Cam", "camera.fov", 0.0, 40.0, interpolation="linear")
    a.keyframe("Cam", "camera.fov", 1.0, 400.0, interpolation="linear")
    a.keyframe("Cube", "camera.fov", 0.0, 10.0)  # a cube has no lens
    p = a.evaluate(1.0)
    assert p.fields[(scene.get_by_name("Cam").id, "camera")]["fov"] == 179.9
    codes = {d.code for d in p.diagnostics}
    assert {"VALUE_CLAMPED", "CHANNEL_NOT_APPLICABLE"} <= codes


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("transform.position.w", "UNKNOWN_PROPERTY"),
        ("__class__", "UNKNOWN_PROPERTY"),
        ("camera", "UNKNOWN_PROPERTY"),
    ],
)
def test_unknown_properties_are_refused(scene, path, code):
    with pytest.raises(AnimationError) as err:
        scene.animation.keyframe("Cube", path, 0.0, 1.0)
    assert err.value.code == code
    assert scene.animation.binding("Cube") is None  # nothing half done


def test_partial_euler_channels_use_the_rest_rotation_for_the_other_axes(scene):
    cube = scene.get_by_name("Cube")
    cube.rotation_euler = (30.0, 0.0, 0.0)
    a = scene.animation
    a.keyframe(cube, "transform.rotation_euler.z", 0.0, 0.0, interpolation="linear")
    a.keyframe(cube, "transform.rotation_euler.z", 1.0, 90.0, interpolation="linear")
    q = a.evaluate(0.5).rotation[0]
    e = R.to_euler(q)[0]
    assert e[0] == pytest.approx(30.0) and e[2] == pytest.approx(45.0)


# --------------------------------------------------------------------------------------- #
# Non-linear animation
# --------------------------------------------------------------------------------------- #


def ramp_clip(a, name: str, path: str, v0: float, v1: float, length: float = 1.0):
    clip = a.add_clip(name)
    clip.set_channel(path, [(0.0, v0), (length, v1)], interpolation="linear")
    return clip


def test_strip_time_mapping_trim_speed_repeat_reverse_and_hold(scene):
    a = scene.animation
    ramp_clip(a, "Ramp", "transform.position.x", 0.0, 10.0, length=2.0)
    a.add_layer("Ball", "Main")
    a.add_strip(
        "Ball", "Main", "Ramp", start=1.0, speed=2.0, repeat=2.0, before="hold", after="hold"
    )
    x = lambda t: float(a.evaluate(t).position[0][0])  # noqa: E731
    assert x(0.0) == pytest.approx(0.0)  # held first frame
    assert x(1.5) == pytest.approx(10.0 * (0.5 * 2.0) / 2.0)  # 1 clip second in
    assert x(2.25) == pytest.approx(2.5)  # second cycle
    assert x(3.0) == pytest.approx(10.0) and x(9.0) == pytest.approx(10.0)  # end held
    a.update_strip("Ball", "Main", 0, reverse=True, repeat=1.0, before="nothing", after="nothing")
    assert x(1.5) == pytest.approx(5.0) and x(1.25) == pytest.approx(7.5)
    assert x(0.5) == pytest.approx(3.0)  # outside, not held: the rest value
    a.update_strip("Ball", "Main", 0, reverse=False, clip_start=0.5, clip_end=1.5, speed=1.0)
    assert x(1.0) == pytest.approx(2.5) and x(1.9) == pytest.approx(7.0)


def test_crossfades_layer_weights_masks_and_additive_layers(scene):
    a = scene.animation
    ramp_clip(a, "Up", "transform.position.z", 0.0, 4.0)
    ramp_clip(a, "Down", "transform.position.z", 4.0, 0.0)
    a.add_layer("Ball", "Base")
    a.add_strip("Ball", "Base", "Up", start=0.0, after="hold", blend_out=0.5)
    a.add_strip("Ball", "Base", "Down", start=0.5, blend_in=0.5)
    z = lambda t: float(a.evaluate(t).position[0][2])  # noqa: E731
    # At 0.6: "Up" (4 * 0.6, weight (1 - 0.6) / 0.5 = 0.8) and "Down" (4 - 4 * 0.1,
    # weight 0.1 / 0.5 = 0.2) crossfade: their weighted average, no rest pose mixed in
    # (the weights sum to 1).
    up, down = 2.4, 3.6
    assert z(0.6) == pytest.approx(0.8 * up + 0.2 * down)
    assert z(0.25) == pytest.approx(1.0)  # only "Up", at full weight: 4 * 0.25
    # Additive layer on top: adds w * (value - reference) where reference = clip start.
    bob = a.add_clip("Bob")
    bob.set_channel("transform.position.z", [(0.0, 0.0), (1.0, 0.5)], interpolation="linear")
    a.add_layer("Ball", "Bob", mode="additive", weight=0.5)
    a.add_strip("Ball", "Bob", "Bob", start=0.0)
    assert z(0.25) == pytest.approx(1.0 + 0.5 * 0.125)
    # A mask that lists another property leaves z alone.
    a.update_layer("Ball", "Bob", mask=[{"path": "transform.position.x"}])
    assert z(0.25) == pytest.approx(1.0)


def test_additive_rotation_layers_compose_on_top_of_the_base(scene):
    a = scene.animation
    cube = scene.get_by_name("Cube")
    a.keyframe(cube, "transform.rotation", 0.0, quat_z(10))
    a.keyframe(cube, "transform.rotation", 1.0, quat_z(10))
    turn = a.add_clip("Turn")
    turn.set_channel("transform.rotation", [(0.0, quat_z(0)), (1.0, quat_z(40))])
    a.add_layer(cube, "Add", mode="additive", weight=0.5)
    a.add_strip(cube, "Add", "Turn", start=0.0, after="hold")
    assert euler_z(a.evaluate(1.0).rotation[0]) == pytest.approx(10 + 20)
    assert euler_z(a.evaluate(0.5).rotation[0]) == pytest.approx(10 + 10)


def test_accumulating_strips_keep_walking_and_turning(scene):
    a = scene.animation
    step = a.add_clip("Step")
    step.set_channel("transform.position.y", [(0.0, 0.0), (1.0, 2.0)], interpolation="linear")
    step.set_channel(
        "transform.rotation_euler.z", [(0.0, 0.0), (1.0, 15.0)], interpolation="linear"
    )
    a.add_layer("Ball", "Walk")
    a.add_strip("Ball", "Walk", "Step", start=0.0, repeat=4.0, accumulate=True)
    p = a.evaluate(2.5)
    assert p.position[0][1] == pytest.approx(5.0)
    assert euler_z(p.rotation[0]) == pytest.approx(37.5)


def test_removing_a_clip_removes_its_strips_and_actions(scene):
    a = scene.animation
    ramp_clip(a, "Ramp", "transform.position.x", 0.0, 1.0)
    a.add_layer("Ball", "L")
    a.add_strip("Ball", "L", "Ramp")
    a.set_action("Cube", "Ramp")
    a.remove_clip("Ramp")
    assert a.binding("Ball").layers[0].strips == () and a.binding("Cube").action is None
    scene.history.undo()
    assert len(a.binding("Ball").layers[0].strips) == 1 and a.binding("Cube").action is not None


# --------------------------------------------------------------------------------------- #
# Scene hooks, persistence
# --------------------------------------------------------------------------------------- #


def test_bindings_go_with_their_objects_and_materials_and_come_back_on_undo(scene):
    a = scene.animation
    a.keyframe("Ball", "transform.position.x", 0.0, 1.0)
    steel = scene.create_material("Steel")
    a.keyframe(steel, "metallic", 0.0, 1.0)
    scene.remove(scene.get_by_name("Ball"))
    assert len(a.bindings()) == 1
    scene.materials.remove_material(steel)
    assert a.bindings() == []
    scene.history.undo()
    scene.history.undo()
    assert len(a.bindings()) == 2


def test_projects_save_and_load_animation_bit_for_bit(scene, tmp_path):
    a = scene.animation
    a.set_timeline(fps="30000/1001", start=0.0, end=4.0)
    a.add_marker("contact", 1.25, {"foot": "left"})
    a.keyframe("Cube", "transform.rotation", 0.0, quat_z(0), timing={"easing": "sine"})
    a.keyframe("Cube", "transform.rotation", 2.0, quat_z(120))
    a.keyframe("Ball", "transform.position", 0.0, (3, 0, 1))
    a.keyframe("Ball", "transform.position", 2.0, (3, 4, 2), interpolation="back", ease="out")
    ramp_clip(a, "Ramp", "transform.position.x", 0.0, 1.0)
    a.add_layer("Ball", "Extra", mode="additive", weight=0.25)
    a.add_strip("Ball", "Extra", "Ramp", start=0.5, repeat=3.0)
    path = tmp_path / "anim.aether"
    scene.save(path, history=True)
    loaded = Scene.load(path, history=True)
    la = loaded.animation
    assert la.fps.to_json() == "30000/1001" and la.markers == a.markers
    for t in (0.0, 0.33, 1.0, 1.999, 3.2):
        p, q = a.evaluate(t), la.evaluate(t)
        assert p.ids == q.ids
        for name in ("position", "rotation", "scale"):
            assert getattr(p, name).tobytes() == getattr(q, name).tobytes()
    assert json.dumps(loaded.to_dict(), sort_keys=True) == json.dumps(
        scene.to_dict(), sort_keys=True
    )
    loaded.history.undo()  # the history file came along
    assert loaded.animation.binding("Ball").layers[0].strips == ()


def test_version_8_projects_migrate_and_bad_documents_are_refused(scene):
    doc = scene.to_dict()
    old = json.loads(json.dumps(doc))
    old["format_version"] = 8
    del old["scene"]["animation"]
    migrated = scene_from_dict(old)
    assert migrated.animation.bindings() == []
    bad = json.loads(json.dumps(doc))
    bad["scene"]["animation"]["items"] = [
        {"id": "x", "kind": "binding", "name": "Ghost", "definition": {"object": "missing-id"}}
    ]
    with pytest.raises(ProjectFormatError):
        scene_from_dict(bad)
    bad["scene"]["animation"] = {"settings": {}, "items": [], "extra": 1}
    with pytest.raises(ProjectFormatError):
        scene_from_dict(bad)


# --------------------------------------------------------------------------------------- #
# Inside the scene: renders, retained world, queries, observations
# --------------------------------------------------------------------------------------- #


def animate(scene: Scene) -> None:
    a = scene.animation
    root, cube, ball = (scene.get_by_name(n) for n in ("Root", "Cube", "Ball"))
    a.keyframe(cube, "transform.position", 0.0, (0, 0, 1))
    a.keyframe(cube, "transform.position", 2.0, (3, 1, 2))
    a.keyframe(root, "transform.rotation_euler.z", 0.0, 0.0)
    a.keyframe(root, "transform.rotation_euler.z", 2.0, 120.0)
    a.keyframe(ball, "visible", 0.0, True)
    a.keyframe(ball, "visible", 1.0, False)
    a.keyframe(cube, "appearance.color", 0.0, (1, 0, 0))
    a.keyframe(cube, "appearance.color", 2.0, (0, 0, 1))
    a.keyframe("Lamp", "light.intensity", 0.0, 1.0)
    a.keyframe("Lamp", "light.intensity", 2.0, 4.0)
    a.keyframe("Cam", "camera.fov", 0.0, 40.0)
    a.keyframe("Cam", "camera.fov", 2.0, 28.0)


def test_rendering_at_a_time_equals_rendering_the_applied_pose(scene):
    animate(scene)
    doc = scene.to_dict()
    for t in (0.4, 1.3, 2.6):
        image = scene.render(time=t, width=80, height=60).image
        posed = Scene.from_dict(json.loads(json.dumps(doc)))
        posed.animation.apply_pose(t)
        posed.animation.enabled = False
        assert np.array_equal(posed.render(width=80, height=60).image, image)
    assert scene.to_dict() == doc


def test_the_retained_world_equals_the_full_extraction_along_random_edits_and_times(scene):
    animate(scene)
    cam = scene.get_by_name("Cam")
    rng = random.Random(4)
    for _ in range(40):
        roll = rng.random()
        if roll < 0.15:
            scene.get_by_name("Ground").position = (0, 0, rng.uniform(-1, 1))
        elif roll < 0.25:
            scene.animation.keyframe(
                "Ball", "transform.position.x", rng.uniform(0, 2), rng.uniform(-2, 2)
            )
        elif roll < 0.3:
            scene.get_by_name("Ball").visible = not scene.get_by_name("Ball").visible
        elif roll < 0.35 and scene.history.can_undo:
            scene.history.undo()
        t = rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, rng.uniform(-1, 3)])
        overlay = animation_overlay(scene, t)
        retained = extract_render_scene(scene, cam, aspect=1.25, overlay=overlay)
        full = _extract_full(scene, cam, aspect=1.25, overlay=overlay)
        assert_render_scenes_equal(retained, full)


def test_spatial_queries_and_observations_follow_the_current_time(scene):
    animate(scene)
    a = scene.animation
    a.time = 0.0
    assert scene.raycast((0, -20, 1.0), (0, 1, 0)).object.name == "Cube"
    a.time = 2.0
    hit = scene.raycast((0, -20, 1.0), (0, 1, 0))
    assert hit is None or hit.object.name != "Cube"  # it moved away
    obs = scene.observe({"camera": "Cam"}, width=64, height=48)
    assert not obs.stale and obs.time == 2.0
    a.time = 0.5
    assert obs.stale
    assert scene.get_by_name("Cube").position == (0.0, 0.0, 1.0)
