"""The lighting check (0.20.0): direct light measured at the subject's visible surface (what
the rasterizers shade), the lighting ratio, separation from the background, lights that do
not reach the subject, and verified corrections of fills and rims."""

from __future__ import annotations

import numpy as np
import pytest

from aether3d import FeedbackError, FeedbackSession, Scene
from aether3d.feedback import check_actions
from aether3d.feedback.lighting import _luma, measure
from aether3d.feedback.observe import observe

EXPECT = {"format": "aether3d.expectations", "version": 1, "defaults": False,
          "views": {"main": {"camera": "Cam"}}}  # fmt: skip


def session(scene, *checks, **kw):
    return FeedbackSession(scene, {**EXPECT, "checks": list(checks)}, width=240, height=160, **kw)


def lighting(scene, **params):
    return {"check": "lighting", "id": "light", "objects": "Head", "view": "main", **params}


def head_scene(preset: str | None = "low_key", wall: str | None = None) -> Scene:
    scene = Scene()
    scene.add_sphere("Head", radius=0.5, position=(0, 0, 1.0), segments=48, rings=24)
    scene.add_plane("Floor", size=10)
    if wall is not None:
        back = scene.add_cube("Wall", size=1.0, position=(0, 2.5, 1.5))
        back.scale = (10, 0.1, 6)
        back.color = wall
    scene.add_camera("Cam", position=(0, -4, 1.0), look_at=(0, 0, 1.0))
    scene.world.ambient_intensity = 0.0
    if preset is not None:
        scene.light_rig("Head", preset)
    return scene


def test_the_measured_light_is_what_the_rasterizer_shades():
    scene = Scene()
    ball = scene.add_sphere("Ball", radius=0.5, position=(0, 0, 1.0), segments=64, rings=32)
    ball.appearance.configure(color="#ffffff", specular=0.0)
    scene.add_camera("Cam", position=(0, -4, 1.2), look_at=(0, 0, 1.0))
    scene.add_point_light("P", position=(-1.5, -1.5, 2.0), intensity=3.0)
    scene.add_spot_light("S", position=(1.5, -1.0, 1.5), look_at=(0, 0, 1.0), intensity=2.0)
    scene.add_directional_light("D", direction=(0.3, 0.5, -0.8), intensity=0.6)
    scene.world.ambient_intensity = 0.1
    obs = observe(scene, None, width=240, height=160, samples=1, linear=True)
    revision = scene.revision
    m = measure(obs, [ball])
    assert scene.revision == revision and m.count == 1024  # (measuring changes nothing)
    # Where no light is shadowed (the rasterizer draws no shadows), a white surface shows
    # exactly the measured irradiance at the same pixel centres.
    open_ = m.even & np.all(m.contribution == m.unshadowed, axis=1)
    xy = m.xy[open_]
    shaded = _luma(obs.linear[xy[:, 1], xy[:, 0]].astype(np.float64))
    rel = np.abs(shaded - m.total[open_]) / np.maximum(shaded, 1e-9)
    # (the rasterizer interpolates in float32; a spot's smooth edge amplifies that at a
    # pixel or two)
    assert open_.sum() > 900 and np.quantile(rel, 0.99) < 1e-6 and rel.max() < 5e-3
    shares = {row["name"]: row["share"] for row in m.summary()["lights"]}
    assert sum(shares.values()) + m.summary()["ambient_share"] == pytest.approx(1.0)


def test_the_ratio_of_two_opposite_suns_on_a_cube_is_their_intensity_ratio():
    # A cube turned 45 degrees: one visible face sees only the key, the other only the fill,
    # both at 45 degrees -- the key's lit side over its shadow side is exactly K / F.
    scene = Scene()
    scene.add_cube("Head", size=1.0, position=(0, 0, 1.0)).rotation_euler = (0, 0, 45)
    scene.add_camera("Cam", position=(0, -5, 1.0), look_at=(0, 0, 1.0))
    scene.add_directional_light("Key", direction=(1, 0, 0), intensity=3.0)
    scene.add_directional_light("Fill", direction=(-1, 0, 0), intensity=1.0)
    scene.world.ambient_intensity = 0.0
    fs = session(scene, lighting(scene, ratio={"max": 2.5}))
    d = fs.diagnose()
    measured = d.checks[0].measured
    assert measured["ratio"] == pytest.approx(3.0, rel=1e-9)
    assert measured["ratio_method"] == "key" and measured["key"] == "Key"
    (f,) = d.findings
    assert f.code == "LIGHTING_RATIO_HIGH" and f.basis == "geometry" and not f.exact
    assert f.magnitude == pytest.approx(np.log2(3.0 / 2.5))


def test_lights_that_do_not_reach_the_subject_say_why():
    scene = head_scene(None)
    scene.add_point_light("Blocked", position=(0, 0, 3.5), intensity=5.0)
    scene.add_cube("Screen", size=0.2, position=(0, 0, 2.2)).scale = (10, 10, 0.5)
    scene.add_spot_light("Away", position=(2, 0, 1), look_at=(5, 0, 1), intensity=5.0)
    linked = scene.add_point_light("Linked", position=(-2, -2, 2), intensity=5.0)
    linked.light.configure(exclude=[scene.get_by_name("Head").id])
    scene.add_directional_light("Sun", direction=(0.2, 0.6, -0.7), intensity=1.0)
    fs = session(scene, lighting(scene, lights=["Blocked", "Away", "Linked", "Sun"]))
    d = fs.diagnose()
    why = {f.subjects[1]["name"]: f.evidence["reason"] for f in d.findings}
    assert set(why) == {"Blocked", "Away", "Linked"}  # the sun reaches it
    # (the main blocker first; the head's own facets shadow a few points at its terminator)
    assert why["Blocked"].startswith("it is blocked by 'Screen'")
    assert why["Linked"] == "light linking excludes the subject"
    assert "points away" in why["Away"]
    assert all(f.code == "LIGHT_NO_EFFECT" and not f.exact for f in d.findings)
    assert fs.propose().to_dict()["proposals"] == []  # (a decision: never moved for you)


def test_separation_is_measured_on_the_displayed_image():
    # A white card lit straight on shows white to its edges: it merges with a white
    # background (L* 100 on both sides) and stands out of a black one.
    out = {}
    for background in ("#ffffff", "#000000"):
        scene = Scene()
        card = scene.add_plane("Head", size=1.0, position=(0, 0, 1.0))
        card.rotation_euler = (90, 0, 0)  # (its +Z faces the camera, -Y)
        card.color = "#ffffff"
        scene.add_camera("Cam", position=(0, -4, 1.0), look_at=(0, 0, 1.0))
        scene.add_directional_light("Sun", direction=(0, 1, 0), intensity=1.0)
        scene.world.ambient_intensity = 0.0
        scene.world.background = background
        fs = session(scene, lighting(scene, max_merged_fraction=0.25))
        out[background] = fs.diagnose()
    white, black = (out[b].checks[0].measured["separation"] for b in ("#ffffff", "#000000"))
    assert white["unit"] == "L*" and white["merged_fraction"] == 1.0
    # (the inner ring, two pixels wide, holds the antialiased edge: its median is ~88 L*)
    assert black["merged_fraction"] == 0.0 and black["median_difference"] > 75.0
    (f,) = out["#ffffff"].findings
    assert f.code == "POOR_SEPARATION" and f.basis == "render" and f.magnitude == 0.75
    assert not out["#000000"].findings


def test_a_harsh_ratio_is_corrected_by_raising_the_fill_and_verified():
    scene = head_scene("low_key")
    fill = scene.get_by_name("Fill")
    before = fill.light.intensity
    fs = session(scene, lighting(scene, ratio={"min": 2, "max": 6}))
    d = fs.diagnose()
    assert [f.code for f in d.findings] == ["LIGHTING_RATIO_HIGH"]
    (p,) = fs.propose().to_dict()["proposals"]
    assert p["safety"] == "verify" and p["actions"][0]["action"] == "set_light"
    assert p["actions"][0]["light"] == fill.id
    result = fs.apply(ids=[p["id"]]).to_dict()
    assert result["status"] == "applied" and result["verification"]["decision"] == "kept"
    assert fill.light.intensity > before
    after = fs.diagnose()
    assert after.accepted and 2 <= after.checks[0].measured["ratio"] <= 6
    scene.history.undo()
    assert fill.light.intensity == before


def test_a_merged_outline_gets_a_rim_light_only_with_approval():
    scene = head_scene("split", wall="#151515")
    scene.get_by_name("Head").color = "#6a5a50"
    fs = session(scene, lighting(scene, max_merged_fraction=0.2))
    d = fs.diagnose()
    merged = d.checks[0].measured["separation"]["merged_fraction"]
    (p,) = fs.propose().to_dict()["proposals"]
    action = p["actions"][0]
    assert p["safety"] == "approval" and action["action"] == "add_light"
    assert action["type"] == "spot" and "look_at" in action
    assert fs.apply().to_dict()["status"] in ("nothing_to_apply", "rejected")  # not automatic
    result = fs.apply(ids=[p["id"]], approve=[p["id"]]).to_dict()
    assert result["status"] == "applied"
    v = result["verification"]
    assert v["decision"] == "kept" and (v["targeted_resolved"] or v["targeted_improved"])
    rim = scene.get_by_name("Feedback rim light")
    assert rim.light.type == "spot"
    assert fs.diagnose().checks[0].measured["separation"]["merged_fraction"] < merged


@pytest.mark.parametrize(
    "params, pointer",
    [
        ({"ratio": {"min": 0.5}}, "/checks/0/ratio/min"),
        ({"min_separation": 120}, "/checks/0/min_separation"),
        ({"max_merged_fraction": 2}, "/checks/0/max_merged_fraction"),
        ({"lights": 3}, "/checks/0/lights"),
    ],
)
def test_lighting_checks_are_validated(params, pointer):
    scene = head_scene()
    with pytest.raises(FeedbackError) as exc:
        session(scene, lighting(scene, **params))
    assert exc.value.issues[0]["pointer"] == pointer


@pytest.mark.parametrize(
    "action",
    [
        {"action": "add_light", "type": "spot", "direction": [0, 0, -1]},
        {"action": "add_light", "type": "area", "position": [0, 0, 2]},
        {"action": "add_light", "type": "area", "position": [0, 0, 2], "look_at": [0, 0, 0],
         "direction": [0, 0, -1]},
        {"action": "add_light", "type": "point", "position": [0, 0, 2], "size": 1.0},
        {"action": "add_light", "type": "directional", "direction": [0, 0, -1],
         "position": [0, 0, 1]},
    ],
)  # fmt: skip
def test_added_lights_are_placed_completely(action):
    with pytest.raises(FeedbackError):
        check_actions([action])


def test_added_spot_and_area_lights_aim_where_asked():
    scene = head_scene(None)
    fs = session(scene)
    actions = [
        {"action": "add_light", "type": "area", "name": "Soft", "position": [2, -2, 2],
         "look_at": [0, 0, 1], "size": 0.5, "intensity": 2},
        {"action": "add_light", "type": "spot", "name": "Rim", "position": [0, 2, 2],
         "look_at": "Head", "intensity": 3},
    ]  # fmt: skip
    assert fs.apply(actions=actions, verify=False).status == "applied"
    soft, rim = scene.get_by_name("Soft"), scene.get_by_name("Rim")
    assert soft.light.type == "area" and soft.light.size == 0.5
    for light, at in ((soft, (2, -2, 2)), (rim, (0, 2, 2))):
        forward = -np.asarray(light.world_matrix[:3, 1])  # lights look along local -Y
        toward = np.array([0, 0, 1.0]) - np.asarray(at, float)
        assert forward @ toward / np.linalg.norm(toward) == pytest.approx(1.0)


def test_an_animated_light_is_not_corrected_through_its_authored_intensity():
    # The fill's intensity is keyed: renders show the keyed value, so scaling the
    # authored one would change nothing. The planner leaves it alone and proposes another
    # change (here: a fill added for approval).
    scene = head_scene("low_key")
    fill = scene.get_by_name("Fill")
    scene.animation.keyframe(fill, "light.intensity", 0.0, fill.light.intensity)
    fs = session(scene, lighting(scene, ratio={"min": 2, "max": 6}))
    assert [f.code for f in fs.diagnose().findings] == ["LIGHTING_RATIO_HIGH"]
    proposals = fs.propose().to_dict()["proposals"]
    assert all(
        a.get("light") != fill.id for p in proposals for a in p["actions"]
    ), proposals  # fmt: skip
