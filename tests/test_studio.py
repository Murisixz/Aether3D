"""Studio light rigs (0.20.0): classical setups placed on a subject with solved intensities;
what the report says is what a render of the subject's centre measures."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import Scene
from aether3d.rendering.script import run_scene_script
from aether3d.rendering.studio import PRESETS


def subject_scene() -> Scene:
    scene = Scene()
    scene.add_cube("Hero", size=1.0, position=(0, 0, 0.9)).scale = (0.5, 0.3, 1.8)
    scene.add_camera("Cam", position=(0, -6, 1.2), look_at=(0, 0, 0.9))
    scene.world.ambient_intensity = 0.0
    return scene


def card_value(scene: Scene, toward: np.ndarray, only: str) -> float:
    """Linear value of a tiny white card at the subject's centre facing ``toward``, lit by
    the light ``only`` (the others switched off for the measurement)."""
    saved = {}
    for obj in scene:
        if obj.light is not None and obj.name != only:
            saved[obj.name] = obj.light.intensity
            obj.light.intensity = 0.0
    hero = scene.get_by_name("Hero")
    hero.visible = False
    card = scene.add_cube("Card", size=0.02, position=(0, 0, 0.9))
    card.color = "#ffffff"
    card.look_at(tuple(np.array([0, 0, 0.9]) - (toward - np.array([0, 0, 0.9]))))
    cam = scene.add_camera("Probe", position=tuple(toward * 0.0 + np.array([0, 0, 0.9]) +
                                                   0.3 * (toward - np.array([0, 0, 0.9])) /
                                                   np.linalg.norm(toward - np.array([0, 0, 0.9]))),
                           look_at=(0, 0, 0.9), fov=2.0, near=0.01)  # fmt: skip
    r = scene.render(width=9, height=9, backend="cpu", camera=cam, samples=1,
                     default_lights=False)  # fmt: skip
    value = float(r.linear[4, 4, 0])
    scene.remove(card)
    scene.remove(cam)
    hero.visible = True
    for name, intensity in saved.items():
        scene.get_by_name(name).light.intensity = intensity
    return value


def test_three_point_lights_the_subject_in_the_stated_ratio():
    scene = subject_scene()
    report = scene.light_rig("Hero", "three_point")
    names = [light["name"] for light in report["lights"]]
    assert names == ["Key", "Fill", "Rim"] and report["ratio"] == pytest.approx(4.0)
    assert all("light_rig" in scene.get_by_name(n).tags for n in names)
    key, fill = (np.array(scene.get_by_name(n).world_matrix[:3, 3]) for n in ("Key", "Fill"))
    # A card facing each light shows that light's solved share: key 3/4, fill 1/4, rim as
    # strong as the key.
    assert card_value(scene, key, "Key") == pytest.approx(0.75, rel=0.01)
    assert card_value(scene, fill, "Fill") == pytest.approx(0.25, rel=0.01)
    rim_at = np.array(scene.get_by_name("Rim").world_matrix[:3, 3])
    assert card_value(scene, rim_at, "Rim") == pytest.approx(0.75, rel=0.01)
    # The key is on the camera's left, raised; the rim behind the subject.
    cam = np.array(scene.get_by_name("Cam").world_matrix[:3, 3])
    assert key[0] < 0 < fill[0] and key[2] > 0.9
    rim = np.array(scene.get_by_name("Rim").world_matrix[:3, 3])
    assert np.dot(rim[:2], cam[:2]) < 0


def test_rigs_update_in_place_and_undo_as_one_step():
    scene = subject_scene()
    before = len(list(scene))
    scene.light_rig("Hero", "three_point")
    report = scene.light_rig("Hero", "low_key", rim=0.5)  # updates Key, Fill, Rim
    assert len(list(scene)) == before + 3 and report["ratio"] == pytest.approx(16.0)
    scene.history.undo()
    assert scene.get_by_name("Key").light.intensity == pytest.approx(
        scene.light_rig("Hero", "three_point")["lights"][0]["intensity"]
    )


def test_switching_presets_removes_only_the_rigs_unused_lights():
    scene = subject_scene()
    scene.light_rig("Hero", "three_point")
    report = scene.light_rig("Hero", "silhouette")
    assert report["removed"] == ["Key", "Fill", "Rim"]
    lights = sorted(o.name for o in scene if o.light is not None)
    assert lights == ["Rim 1", "Rim 2"]
    scene.history.undo()  # one step: the three-point rig is back
    assert sorted(o.name for o in scene if o.light is not None) == ["Fill", "Key", "Rim"]
    # A light the rig did not make (untagged) is never removed, even with a rig's name.
    scene = subject_scene()
    scene.add_point_light("Fill", position=(2, -2, 2))
    report = scene.light_rig("Hero", "split")
    assert report["removed"] == [] and scene.get_by_name("Fill").light.type == "point"


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_every_preset_lights_a_head_from_its_side(preset):
    scene = Scene()
    scene.add_sphere("Head", radius=0.12, position=(0, 0, 1.6))
    scene.add_camera("Cam", position=(0, -1.5, 1.6), look_at=(0, 0, 1.6))
    scene.world.ambient_intensity = 0.0
    run_scene_script(scene, {"format": "aether3d.scene-script", "version": 1, "operations": [
        {"op": "light_rig", "subject": "Head", "preset": preset, "temperature": 5600},
    ]})  # fmt: skip
    r = scene.render(width=64, height=64, backend="cpu", default_lights=False)
    seen = r.linear[..., 1][r.object_index >= 0]
    if preset == "silhouette":  # rims only: the outline is lit, the camera's side is dark
        ys, xs = np.nonzero((r.linear[..., 1] > 0.5 * seen.max()) & (r.object_index >= 0))
        disk = np.sqrt((r.object_index >= 0).sum() / np.pi)
        assert np.hypot(xs + 0.5 - 32, ys + 0.5 - 32).min() > 0.6 * disk
        return
    lit_x = np.nonzero((r.linear[..., 1] > 0.05) & (r.object_index >= 0))[1]
    assert seen.max() > 0.05, preset
    key = PRESETS[preset]["key"]
    if key is not None and key[0] > 10:  # keys on the camera's left light the image's left
        assert lit_x.mean() < 32, preset


def test_rig_arguments_are_checked():
    scene = subject_scene()
    with pytest.raises(Exception, match="preset"):
        scene.light_rig("Hero", "spooky")
    with pytest.raises(Exception, match="ratio"):
        scene.light_rig("Hero", "three_point", ratio=0.5)
