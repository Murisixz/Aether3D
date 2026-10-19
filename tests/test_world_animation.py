"""The world as an animation target (0.20.0): the background, ambient light, environment
and atmosphere keyed like any property -- with the sun's light, a day-night cycle.

Invariants as for every animated property: evaluating never changes the scene, rendering
at a time equals writing the pose into the scene and rendering that, the retained render
world equals the full extraction, history and files keep the keys, and glTF (which has no
world) reports what it cannot carry.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
from gpu_support import GPU
from parity import assert_render_scenes_equal

from aether3d import Scene
from aether3d.animation.script import AnimationScriptError, run_animation_script
from aether3d.errors import AnimationError, UnsupportedFeatureError
from aether3d.rendering.color import to_linear
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import _extract_full, extract_render_scene

WORLD = {"world": True}


@pytest.fixture
def scene() -> Scene:
    s = Scene("Day")
    s.add_plane("Ground", size=40)
    s.add_cube("House", size=2, position=(0, 0, 1))
    cam = s.add_camera("Cam", position=(9, -9, 4), look_at=(0, 0, 1))
    s.active_camera = cam
    s.add_directional_light("Sun", direction=(0.3, 0.4, -1.0), intensity=2.0)
    s.world.configure(environment={"type": "sky"}, atmosphere={"enabled": True, "sun": "Sun"})
    return s


def day_night(s: Scene) -> None:
    """Noon to night in four seconds: the sun sets and dims, the sky darkens."""
    a = s.animation
    sun = s.get_by_name("Sun")
    a.set_timeline(fps=24, start=0.0, end=4.0)
    a.keyframe(sun, "transform.rotation_euler.x", 0.0, 0.0, interpolation="linear")
    a.keyframe(sun, "transform.rotation_euler.x", 4.0, 80.0, interpolation="linear")
    a.keyframe(sun, "light.intensity", 0.0, 2.0, interpolation="linear")
    a.keyframe(sun, "light.intensity", 4.0, 0.05, interpolation="linear")
    a.keyframe(WORLD, "background", 0.0, (0.55, 0.7, 0.95), interpolation="linear")
    a.keyframe(WORLD, "background", 4.0, (0.02, 0.03, 0.08), interpolation="linear")
    a.keyframe(s.world, "ambient_intensity", 0.0, 0.3, interpolation="linear")
    a.keyframe(s.world, "ambient_intensity", 4.0, 0.02, interpolation="linear")
    a.keyframe(WORLD, "environment.zenith_color", 0.0, (0.2, 0.45, 0.9))
    a.keyframe(WORLD, "environment.zenith_color", 4.0, (0.01, 0.01, 0.04))
    a.keyframe(WORLD, "atmosphere.fog_density", 0.0, 0.0, interpolation="linear")
    a.keyframe(WORLD, "atmosphere.fog_density", 4.0, 0.02, interpolation="linear")
    a.keyframe(WORLD, "atmosphere.fog_color", 2.0, (0.9, 0.6, 0.4))  # (the sunset)
    a.keyframe(WORLD, "atmosphere.fog_color", 4.0, (0.02, 0.02, 0.05))


def test_world_channels_evaluate_and_never_write_the_scene(scene):
    day_night(scene)
    before = scene.to_dict()
    pose = scene.animation.evaluate(1.0)
    world = pose.world
    assert world["world"]["ambient_intensity"] == pytest.approx(0.3 + 0.25 * (0.02 - 0.3))
    assert world["world"]["background"] == pytest.approx([0.4175, 0.5325, 0.7325])
    assert world["atmosphere"]["fog_density"] == pytest.approx(0.005)
    assert set(world) == {"world", "environment", "atmosphere"}
    assert scene.to_dict() == before  # evaluation writes nothing
    assert scene.world.ambient_intensity == 0.1  # the authored value
    binding = scene.animation.binding(WORLD)
    assert binding is not None and binding.target == ("world", "world")
    assert scene.animation.binding(scene.world) == binding


def test_rendering_at_a_time_equals_rendering_the_applied_pose(scene):
    day_night(scene)
    doc = scene.to_dict()
    for t in (0.0, 1.3, 2.7, 4.0):
        image = scene.render(time=t, width=64, height=48).image
        posed = Scene.from_dict(json.loads(json.dumps(doc)))
        posed.animation.apply_pose(t)
        posed.animation.enabled = False
        assert np.array_equal(posed.render(width=64, height=48).image, image), t
    assert scene.to_dict() == doc
    # The sky the camera sees changes with the keys: noon is bright, night is dark.
    noon, night = (scene.render(time=t, width=64, height=48).image for t in (0.0, 4.0))
    assert noon[0].mean() > 4 * night[0].mean()


def test_the_path_tracer_renders_the_animated_world(scene):
    day_night(scene)
    opts = {"width": 24, "height": 18, "backend": "path", "path": {"samples": 2}}
    doc = scene.to_dict()
    for t in (0.5, 3.5):
        a = scene.render(time=t, **opts).linear
        posed = Scene.from_dict(json.loads(json.dumps(doc)))
        posed.animation.apply_pose(t)
        posed.animation.enabled = False
        assert np.array_equal(posed.render(**opts).linear, a), t


@pytest.mark.skipif(not GPU, reason="needs a GPU")
def test_the_gpu_rasterizer_renders_the_animated_world(scene):
    day_night(scene)
    for t in (0.5, 3.5):
        cpu = scene.render(time=t, width=48, height=36, samples=1).linear
        gpu = scene.render(time=t, width=48, height=36, backend="gpu", supersample=1,
                           msaa=1).linear  # fmt: skip
        assert np.median(np.abs(cpu - gpu)) < 2e-3, t


def test_the_retained_world_equals_the_full_extraction_along_times_and_edits(scene):
    day_night(scene)
    cam = scene.get_by_name("Cam")
    rng = random.Random(7)
    for _ in range(24):
        roll = rng.random()
        if roll < 0.2:
            scene.animation.keyframe(WORLD, "ambient_intensity", rng.uniform(0, 4),
                                     rng.uniform(0, 0.5))  # fmt: skip
        elif roll < 0.3:
            scene.world.ambient_color = (rng.random(), rng.random(), rng.random())
        elif roll < 0.4 and scene.history.can_undo:
            scene.history.undo()
        t = rng.choice([0.0, 1.0, 2.0, rng.uniform(-1, 5)])
        overlay = animation_overlay(scene, t)
        retained = extract_render_scene(scene, cam, aspect=1.25, overlay=overlay)
        full = _extract_full(scene, cam, aspect=1.25, overlay=overlay)
        assert_render_scenes_equal(retained, full)
        if overlay is not None and overlay.world.get("ambient_intensity") is not None:
            w = scene.world
            expected = to_linear(w.ambient_color) * overlay.world["ambient_intensity"]
            assert np.array_equal(full.ambient, expected)


def test_history_and_files_keep_world_animation(scene):
    a = scene.animation
    a.keyframe(WORLD, "environment.intensity", 0.0, 1.0)
    a.keyframe(WORLD, "environment.intensity", 2.0, 3.0)
    assert a.evaluate(2.0).world["environment"]["intensity"] == pytest.approx(3.0)
    again = Scene.from_dict(json.loads(json.dumps(scene.to_dict())))
    assert again.animation.evaluate(2.0).world == a.evaluate(2.0).world
    scene.history.undo()
    assert a.evaluate(2.0).world["environment"]["intensity"] == pytest.approx(1.0)
    scene.history.undo()
    assert a.binding(WORLD) is None or not a.evaluate(2.0).world
    scene.history.redo()
    scene.history.redo()
    assert a.evaluate(2.0).world["environment"]["intensity"] == pytest.approx(3.0)
    count = a.apply_pose(2.0)
    assert count == 0 and scene.world.environment.intensity == pytest.approx(3.0)
    scene.history.undo()
    assert scene.world.environment.intensity == 1.0


def test_values_out_of_range_are_clamped_and_reported(scene):
    a = scene.animation
    a.keyframe(WORLD, "atmosphere.fog_density", 0.0, 0.0, interpolation="linear")
    a.keyframe(WORLD, "atmosphere.fog_density", 1.0, 40.0, interpolation="linear")
    pose = a.evaluate(1.0)
    assert pose.world["atmosphere"]["fog_density"] == 10.0  # the atmosphere's limit
    assert "VALUE_CLAMPED" in {d.code for d in pose.diagnostics}


@pytest.mark.parametrize(
    "path",
    ["environment.type", "environment.image", "atmosphere.enabled", "atmosphere.sun",
     "sky", "environment", "background.w", 'background["x"]', "world.background",
     "transform.position.x"],
)  # fmt: skip
def test_only_numeric_world_properties_animate(scene, path):
    with pytest.raises(AnimationError) as err:
        scene.animation.keyframe(WORLD, path, 0.0, 1.0)
    assert err.value.code == "UNKNOWN_PROPERTY"
    assert scene.animation.binding(WORLD) is None  # nothing half done


def test_world_targets_are_explicit(scene):
    scene.add_object("world")  # an object may be named "world": it stays an object
    scene.animation.keyframe("world", "transform.position.z", 0.0, 1.0)
    assert scene.animation.binding("world").target[0] == "object"
    with pytest.raises(AnimationError):
        scene.animation.keyframe({"world": False}, "ambient_intensity", 0.0, 1.0)
    other = Scene()
    with pytest.raises(AnimationError):
        scene.animation.keyframe(other.world, "ambient_intensity", 0.0, 1.0)


def test_animation_scripts_animate_the_world(scene):
    script = {
        "format": "aether3d.animation-script",
        "version": 1,
        "operations": [
            {"op": "set_channel", "target": {"world": True}, "path": "environment.rotation",
             "keys": [[0, 0], [4, 90]], "interpolation": "linear"},
            {"op": "keyframe", "target": {"world": True}, "path": "ambient_color",
             "time": 1, "value": [1, 0.8, 0.6]},
        ],
    }  # fmt: skip
    run_animation_script(scene, script)
    pose = scene.animation.evaluate(2.0)
    assert pose.world["environment"]["rotation"] == pytest.approx(45.0)
    assert pose.world["world"]["ambient_color"] == pytest.approx([1, 0.8, 0.6])
    bad = json.loads(json.dumps(script))
    bad["operations"][0]["target"] = {"world": "yes"}
    with pytest.raises(AnimationScriptError):
        run_animation_script(scene, bad)


def test_gltf_reports_the_world_animation_it_cannot_carry(scene, tmp_path):
    day_night(scene)
    with pytest.raises(UnsupportedFeatureError) as err:  # lossy: refused unless allowed
        scene.export(tmp_path / "day.glb", animation="actions")
    assert "ANIMATED_PROPERTY_NOT_EXPORTED" in str(err.value)
    report = scene.export(tmp_path / "day.glb", animation="actions", allow_lossy=True)
    codes = {d.code for d in report.diagnostics}
    assert "ANIMATED_PROPERTY_NOT_EXPORTED" in codes
