"""Format 13 (0.20.0): v12 projects load and render unchanged -- cameras get a pinhole
lens, in-and-out back easing keeps its shape, curves keep the local automatic handles."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from aether3d import Scene
from aether3d.serialization import FORMAT_VERSION, migrate, scene_from_dict, scene_to_dict


def old_back_in_out(u: np.ndarray, s: float) -> np.ndarray:
    """v0.19's back in_out: the two halves of f(u) = (s + 1) u^3 - s u^2, no 1.525."""
    f = lambda x: (s + 1.0) * x**3 - s * x**2  # noqa: E731
    return np.where(u < 0.5, 0.5 * f(2 * u), 1.0 - 0.5 * f(2 - 2 * u))


def v12_document() -> dict:
    scene = Scene()
    cam = scene.add_camera("Cam", position=(0, -5, 1), look_at=(0, 0, 0))
    box = scene.add_cube("Box", size=0.5)
    anim = scene.animation
    anim.set_channel(box, "transform.position.x", [
        {"time": 0, "value": 0, "interpolation": "back", "ease": "in_out", "back": 2.0},
        {"time": 1, "value": 3, "interpolation": "back", "ease": "in_out"},
        {"time": 2, "value": 1},
    ], smoothing="local")  # fmt: skip
    anim.set_channel(box, "transform.position.z", [(0, 0), (1, 1), (2, 0.5)], smoothing="local")
    doc = scene_to_dict(scene)
    doc["format_version"] = 12
    for obj in doc["scene"]["objects"]:
        if obj.get("camera"):
            for key in (
                "sensor",
                "shift_x",
                "shift_y",
                "fstop",
                "focus_distance",
                "focus_object",
                "aperture_blades",
                "aperture_rotation",
                "aperture_ratio",
            ):
                del obj["camera"][key]  # fmt: skip
    # v12 curves had no smoothing field, and their back values meant 0.19's in_out.
    for item in doc["scene"]["animation"]["items"]:
        for ch in item["definition"].get("channels", {}).values():
            ch.pop("smoothing", None)
            keys = ch["keys"]
            if keys.get("interpolation") != "back" and "back" not in keys:
                continue
            keys["back"] = [2.0, 1.70158, 1.70158]  # what 0.19 wrote
    del cam
    return doc


def test_v12_documents_load_unchanged():
    doc = v12_document()
    before = copy.deepcopy(doc)
    upgraded = migrate(doc)
    assert doc == before  # never modified in place
    assert upgraded["format_version"] == FORMAT_VERSION == 13
    scene = scene_from_dict(doc)
    cam = scene.get_by_name("Cam").camera
    assert cam.fstop == 0.0 and cam.shift_x == 0.0 and cam.sensor == 36.0
    box = scene.get_by_name("Box")
    x = scene.animation.channel(box, "transform.position.x")
    t = np.linspace(0.0, 1.0, 101)
    expect = 3.0 * old_back_in_out(t, 2.0)
    assert np.allclose(x.evaluate(t), expect, atol=1e-12)
    t2 = np.linspace(1.0, 2.0, 101)
    expect2 = 3.0 - 2.0 * old_back_in_out(t2 - 1.0, 1.70158)
    assert np.allclose(x.evaluate(t2), expect2, atol=1e-12)
    z = scene.animation.channel(box, "transform.position.z")
    assert z.smoothing == "local"
    from aether3d.animation.curves import FCurve  # the local rule: what 0.19 computed

    assert z.same(FCurve.from_keys([(0, 0), (1, 1), (2, 0.5)], smoothing="local"))


def test_saving_writes_the_new_fields():
    scene = scene_from_dict(v12_document())
    doc = scene_to_dict(scene)
    assert doc["format_version"] == 13
    cam = next(o for o in doc["scene"]["objects"] if o.get("camera"))["camera"]
    assert cam["fstop"] == 0.0 and cam["focus_object"] is None
    again = scene_from_dict(doc)
    box = again.get_by_name("Box")
    a = scene.animation.channel(scene.get_by_name("Box"), "transform.position.x")
    b = again.animation.channel(box, "transform.position.x")
    assert b.same(a)
    assert pytest.approx(b.evaluate(0.25)) == a.evaluate(0.25)
