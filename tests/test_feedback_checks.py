"""Expectations and diagnosis: every check kind against scenes with known answers (Stage 12).

Placement is checked against analytic gaps, penetrations and tilts (also under rotated,
scaled and mirrored parents); visibility against cameras placed so the answer is known;
exposure against renders with known lighting; default checks for false positives. All
renders use the CPU rasterizer.
"""

from __future__ import annotations

import json
import math

import pytest

from aether3d import FeedbackError, FeedbackSession, Scene
from aether3d.feedback import parse_expectations
from aether3d.serialization import scene_to_dict

EXPECT = {"format": "aether3d.expectations", "version": 1}


def doc(*checks, **extra) -> dict:
    return {**EXPECT, "checks": list(checks), **extra}


def diagnose(scene: Scene, *checks, defaults=False, **extra):
    session = FeedbackSession(scene, doc(*checks, defaults=defaults, **extra), width=160, height=90)
    return session.diagnose()


def codes(d) -> list[str]:
    return [f.code for f in d.findings]


def room() -> Scene:
    scene = Scene("Room")
    scene.add_plane("Floor", size=10)
    table = scene.add_cube("Table", size=1, position=(0, 0, 0.5))
    table.tags = ["furniture"]
    scene.add_camera("Cam", position=(3, -5, 3), look_at=(0, 0, 0.6))
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=1.0)
    return scene


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def test_every_problem_of_a_document_is_reported_with_its_pointer():
    bad = {
        "format": "aether3d.expectations",
        "version": 2,
        "views": {"main": {"camera": "Cam", "zoom": 3}},
        "checks": [
            {"check": "rests_on", "support": "Table"},
            {"check": "visibel", "objects": "A"},
            {"check": "framing", "objects": "A", "region": [0.5, 0, 0.2, 1], "view": "side"},
            {"check": "exposure", "mean_luminance": {"min": 2, "max": 1}},
            {"check": "exists", "objects": "A", "id": "x"},
            {"check": "exists", "objects": "B", "id": "x"},
            {"check": "upright", "objects": "@sym"},
            {"check": "no_overlap", "objects": "A", "tolerance": -1, "fix": {"move": "up"}},
        ],
        "extra": 1,
    }
    with pytest.raises(FeedbackError) as exc:
        parse_expectations(bad)
    found = {(i["code"], i["pointer"]) for i in exc.value.issues}
    assert ("UNSUPPORTED_VERSION", "/version") in found
    assert ("UNKNOWN_KEY", "/views/main/zoom") in found
    assert ("UNKNOWN_KEY", "/extra") in found
    assert ("MISSING_KEY", "/checks/0/objects") in found
    assert ("INVALID_VALUE", "/checks/1/check") in found
    assert ("INVALID_VALUE", "/checks/2/region") in found
    assert ("UNKNOWN_VIEW", "/checks/2/view") in found
    assert ("INVALID_VALUE", "/checks/3/mean_luminance") in found
    assert ("DUPLICATE_ID", "/checks/5") in found
    assert ("INVALID_REFERENCE", "/checks/6/objects") in found
    assert ("OUT_OF_RANGE", "/checks/7/tolerance") in found
    assert ("INVALID_VALUE", "/checks/7/fix/move") in found
    assert exc.value.code == exc.value.issues[0]["code"]
    hints = {i["pointer"]: i["hint"] for i in exc.value.issues}
    assert hints["/checks/1/check"] == "did you mean 'visible'?"


def test_documents_as_json_text_and_limits():
    exp = parse_expectations(json.dumps(doc({"check": "exists", "objects": "Table"})))
    assert exp.checks[0].id == "exists-1" and exp.checks[0].severity == "error"
    for text, code in (
        ("{not json", "INVALID_JSON"),
        ('{"format": "x"}', "INVALID_FORMAT"),
        ("[1]", "INVALID_TYPE"),
    ):
        with pytest.raises(FeedbackError) as exc:
            parse_expectations(text)
        assert exc.value.code == code
    with pytest.raises(FeedbackError) as exc:
        parse_expectations(doc(*[{"check": "exists", "objects": "A"}] * 501))
    assert exc.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(FeedbackError) as exc:
        parse_expectations(doc({"check": "exists", "objects": "A", "label": "x\x00y"}))
    assert exc.value.code == "INVALID_VALUE"


def test_view_checks_use_the_default_view():
    exp = parse_expectations(
        {
            **EXPECT,
            "views": {"main": {"camera": "Cam"}, "top": {"preset": "top"}},
            "checks": [{"check": "visible", "objects": "A"}],
        }
    )
    assert exp.default_view == "main" and exp.checks[0].view == "main"
    with pytest.raises(FeedbackError) as exc:
        parse_expectations(
            {
                **EXPECT,
                "views": {"a": {}, "b": {"preset": "top"}},
                "checks": [{"check": "visible", "objects": "A"}],
            }
        )
    assert exc.value.code == "MISSING_KEY"


# --------------------------------------------------------------------------- #
# Existence, materials
# --------------------------------------------------------------------------- #


def test_exists_reports_missing_elements_and_counts():
    scene = room()
    for k in range(3):
        scene.add_cube(f"Chair {k}", size=0.4, position=(k - 1, -1.5, 0.2)).tags = ["chair"]
    d = diagnose(
        scene,
        {"id": "lamp", "check": "exists", "objects": "Lamp"},
        {"id": "chairs", "check": "exists", "objects": {"tags": ["chair"]}, "count": {"min": 4}},
        {"id": "few", "check": "exists", "objects": {"tags": ["chair"]}, "count": {"max": 2}},
        {"id": "ok", "check": "exists", "objects": ["Table", "Floor"]},
    )
    by = {f.check: f for f in d.findings}
    assert by["lamp"].code == "MISSING_ELEMENT" and by["lamp"].basis == "scene"
    assert by["chairs"].code == "MISSING_ELEMENT" and by["chairs"].magnitude == 1.0
    assert by["few"].code == "TOO_MANY" and by["few"].magnitude == 1.0
    assert "ok" not in by
    assert not d.accepted and d.errors == 3


def test_material_checks():
    scene = room()
    scene.create_material("Oak", base_color="#8a5a2b")
    d = diagnose(
        scene,
        {"id": "oak", "check": "material", "objects": "Table", "material": "Oak"},
        {"id": "gold", "check": "material", "objects": "Table", "material": "Gold"},
    )
    by = {f.check: f for f in d.findings}
    assert by["oak"].code == "WRONG_MATERIAL" and by["oak"].evidence["current"] is None
    assert by["gold"].code == "MISSING_MATERIAL"
    scene.assign_material("Table", "Oak")
    d = diagnose(scene, {"id": "oak", "check": "material", "objects": "Table", "material": "Oak"})
    assert d.accepted and not d.findings


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("z", "code", "magnitude"),
    [(1.35, "FLOATING", 0.25), (1.06, "SUNKEN", 0.04), (1.1, None, None)],
)
def test_rests_on_measures_gaps_and_penetrations(z, code, magnitude):
    scene = room()
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, z))
    d = diagnose(scene, {"check": "rests_on", "objects": "Vase", "support": "Table"})
    if code is None:
        assert d.accepted and not d.findings
        return
    (f,) = d.findings
    assert f.code == code and f.basis == "geometry" and f.exact
    assert f.magnitude == pytest.approx(magnitude, abs=2e-6)
    assert [s["role"] for s in f.subjects] == ["subject", "support"]


def test_rests_on_through_rotated_scaled_and_mirrored_parents():
    scene = room()
    pivot = scene.add_object("Pivot", position=(0.1, 0.0, 1.0))
    pivot.rotation_euler = (0, 0, 37)
    pivot.scale = (-1.5, 1.5, 1.5)  # mirrored and scaled
    vase = scene.add_cube("Vase", size=0.1, parent=pivot, position=(0, 0, 0.1))
    # World: the vase is 0.15 tall, centered 0.15 above the table top: a gap of 0.075.
    d = diagnose(scene, {"check": "rests_on", "objects": "Vase", "support": "Table"})
    (f,) = d.findings
    assert f.code == "FLOATING" and f.magnitude == pytest.approx(0.075, abs=1e-6)
    lo = vase.bounds().aabb.min[2]
    assert f.magnitude == pytest.approx(lo - 1.0, abs=1e-6)


def test_rests_on_without_a_named_support_uses_what_is_below():
    scene = room()
    scene.add_cube("Book", size=0.2, position=(0, 0, 1.3))  # 0.2 above the table
    scene.add_cube("Box", size=0.4, position=(3, 3, 0.5))  # 0.3 above the floor
    d = diagnose(scene, {"check": "rests_on", "objects": ["Book", "Box"]})
    by = {f.subjects[0]["name"]: f for f in d.findings}
    assert by["Book"].subjects[1]["name"] == "Table"
    assert by["Book"].magnitude == pytest.approx(0.2, abs=1e-6)
    assert by["Box"].subjects[1]["name"] == "Floor"
    assert by["Box"].magnitude == pytest.approx(0.3, abs=1e-6)


def test_rests_on_reports_objects_not_above_their_support():
    scene = room()
    scene.add_cube("Vase", size=0.2, position=(3, 0, 1.1))
    d = diagnose(scene, {"check": "rests_on", "objects": "Vase", "support": "Table"})
    assert codes(d) == ["NOT_ON_SUPPORT"]
    scene.add_cube("High", size=0.2, position=(0, 30, 50))
    d = diagnose(scene, {"check": "rests_on", "objects": "High"})
    assert codes(d) == ["NO_SUPPORT"]


def test_no_overlap_tells_contact_from_penetration():
    scene = room()
    scene.add_cube("Touching", size=0.5, position=(0.75, 0, 0.25))  # face to face at x = 0.5
    scene.add_cube("Inside", size=0.4, position=(0.6, 0.6, 0.5))  # 0.1 into the table
    d = diagnose(
        scene, {"check": "no_overlap", "objects": ["Touching", "Inside"], "against": "Table"}
    )
    (f,) = d.findings
    assert (
        f.code == "INTERSECTION"
        and f.subjects[0]["name"] == "Table"
        or f.subjects[1]["name"] == "Inside"
    )
    assert f.magnitude == pytest.approx(0.1, abs=1e-4)
    roles = {s["name"]: s["role"] for s in f.subjects}
    assert roles == {"Table": "against", "Inside": "subject"}
    measured = d.checks[0].measured
    assert measured["contacts"] == 1 and measured["penetrating"] == 1
    d = diagnose(
        scene,
        {"check": "no_overlap", "objects": "Touching", "against": "Table", "allow_contact": False},
    )
    assert codes(d) == ["INTERSECTION"]


def test_no_overlap_reports_containment():
    scene = room()
    scene.add_cube("Ghost", size=0.2, position=(0, 0, 0.5))  # wholly inside the table
    d = diagnose(scene, {"check": "no_overlap", "objects": "Ghost", "against": "Table"})
    assert codes(d) == ["CONTAINED"]


def test_inside_region_distance_relation_upright():
    scene = room()
    lamp = scene.add_cube("Lamp", size=0.3, position=(4.0, 0, 0.15))
    lamp.rotation_euler = (12, 0, 33)
    d = diagnose(
        scene,
        {
            "id": "region",
            "check": "inside_region",
            "objects": "Lamp",
            "region": {"box": {"min": [-3, -3, 0], "max": [3, 3, 3]}},
        },
        {"id": "dist", "check": "distance", "a": "Lamp", "b": "Table", "max": 2.0},
        {"id": "rel", "check": "relation", "a": "Lamp", "b": "Table", "predicate": "above"},
        {"id": "up", "check": "upright", "objects": "Lamp", "max_tilt": 1},
    )
    by = {f.check: f for f in d.findings}
    assert by["region"].code == "OUT_OF_REGION"
    assert by["region"].evidence["fits"] is True
    assert by["dist"].code == "TOO_FAR" and by["dist"].magnitude > 0
    assert by["rel"].code == "RELATION_VIOLATED" and not by["rel"].exact  # box-based predicate
    assert by["up"].code == "TILTED"
    assert by["up"].evidence["tilt"] == pytest.approx(12.0, abs=1e-9)
    assert by["up"].magnitude == pytest.approx(11.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def test_visibility_hidden_out_of_view_occluded_too_small():
    scene = room()
    scene.add_cube("Hidden", size=0.3, position=(-1, 0, 0.15)).visible = False
    scene.add_cube("Behind", size=0.3, position=(6, -10, 4))  # behind the camera
    scene.add_cube("Back", size=0.3, position=(-0.05, 0.1, 1.15))  # on the table, then hidden:
    scene.add_cube("Wall", size=1, position=(1.2, -2, 1.1)).scale = (3, 0.1, 3)
    scene.add_cube("Speck", size=0.005, position=(-3, 3, 0.0025))
    d = diagnose(
        scene,
        {
            "check": "visible",
            "objects": ["Hidden", "Behind", "Back", "Speck", "Table"],
            "min_pixels": 4,
        },
        views={"main": {"camera": "Cam"}},
    )
    by = {f.subjects[0]["name"]: f for f in d.findings}
    assert by["Hidden"].code == "HIDDEN"
    assert by["Behind"].code == "OUT_OF_VIEW" and "behind the camera" in by["Behind"].message
    assert by["Back"].code == "OCCLUDED"
    assert by["Back"].evidence["occlusion"]["occluders"][0]["name"] == "Wall"
    assert by["Speck"].code == "TOO_SMALL"
    assert "Table" not in by or by["Table"].code == "OCCLUDED"


def test_partial_visibility_and_frame_borders():
    scene = room()
    scene.get_by_name("Cam").look_at((4, 0, 0.6))  # the table at the left border
    d = diagnose(
        scene,
        {"check": "visible", "objects": "Table", "in_frame": True},
        views={"main": {"camera": "Cam"}},
    )
    (f,) = d.findings
    assert f.code == "PARTIALLY_OUT_OF_FRAME" and f.region is not None
    assert 0.0 < f.magnitude < 1.0


def test_framing_center_fill_and_height():
    scene = room()
    d = diagnose(
        scene,
        {"id": "center", "check": "framing", "objects": "Table", "region": [0.0, 0.0, 0.3, 0.3]},
        {"id": "fill", "check": "framing", "objects": "Table", "fill": {"min": 0.5}},
        {"id": "height", "check": "framing", "objects": "Table", "height": {"max": 0.05}},
        views={"main": {"camera": "Cam"}},
    )
    by = {f.check: f for f in d.findings}
    assert by["center"].code == "OFF_CENTER"
    assert by["fill"].code == "TOO_SMALL_IN_FRAME"
    assert by["height"].code == "TOO_LARGE_IN_FRAME"
    m = d.checks[0].measured
    assert 0 < m["center"][0] < 1 and 0 < m["fill"] < 0.5


def test_exposure_checks_measure_linear_light():
    scene = room()
    d = diagnose(
        scene,
        {"id": "dark", "check": "exposure", "mean_luminance": {"min": 5.0}},
        {"id": "bright", "check": "exposure", "mean_luminance": {"max": 0.001}},
        views={"main": {"camera": "Cam"}},
    )
    by = {f.check: f for f in d.findings}
    mean = d.checks[0].measured["mean_luminance"]
    assert by["dark"].code == "UNDEREXPOSED"
    assert by["dark"].magnitude == pytest.approx(math.log2(5.0 / mean))
    assert by["bright"].code == "OVEREXPOSED"
    assert d.checks[0].measured["luminance_source"] == "linear"
    scene.render_settings.exposure = 1.0
    d2 = diagnose(
        scene,
        {"id": "dark", "check": "exposure", "mean_luminance": {"min": 5.0}},
        views={"main": {"camera": "Cam"}},
    )
    assert d2.checks[0].measured["mean_luminance"] == pytest.approx(2 * mean, rel=1e-6)


def test_camera_clipping():
    scene = room()
    scene.get_by_name("Cam").camera.configure(near=6.4, far=7.0)
    d = diagnose(
        scene, {"check": "camera_clipping", "objects": "Table"}, views={"main": {"camera": "Cam"}}
    )
    assert sorted(codes(d)) == ["FAR_CLIPPED", "NEAR_CLIPPED"]


# --------------------------------------------------------------------------- #
# Default checks, determinism, read-only
# --------------------------------------------------------------------------- #


def test_default_checks_catch_black_and_empty_images():
    scene = room()
    scene.remove(scene.get_by_name("Sun"))
    scene.render_settings.default_lights = False
    scene.world.ambient_intensity = 0.0
    scene.world.background = "#000000"
    d = diagnose(scene, defaults=True)
    assert "BLACK_IMAGE" in codes(d) and not d.accepted
    scene = room()
    scene.get_by_name("Cam").look_at((0, -20, 3))  # facing away
    d = diagnose(scene, defaults=True)
    assert "EMPTY_VIEW" in codes(d)


def test_default_checks_do_not_flag_a_sound_scene():
    scene = room()
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, 1.1))  # resting: contact only
    d = diagnose(scene, defaults=True)
    assert d.accepted and d.errors == 0 and d.warnings == 0
    assert all(f.severity == "info" for f in d.findings)
    assert not [f for f in d.findings if f.code == "POSSIBLE_PENETRATION"]


def test_default_checks_label_heuristics_and_penetrations_as_such():
    scene = room()
    scene.get_by_name("Sun").light.intensity = 40.0
    scene.add_cube("Stuck", size=0.4, position=(0.6, 0.6, 0.5))
    d = diagnose(scene, defaults=True)
    over = [f for f in d.findings if f.code == "OVEREXPOSED"]
    assert over and over[0].basis == "heuristic" and over[0].severity == "warning"
    pen = [f for f in d.findings if f.code == "POSSIBLE_PENETRATION"]
    assert pen and pen[0].severity == "info"
    assert d.accepted  # warnings and info do not block the default acceptance


def test_diagnosis_is_deterministic_and_changes_nothing():
    scene = room()
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, 1.35))
    scene.add_cube("Stuck", size=0.4, position=(0.6, 0.6, 0.5))
    checks = [
        {"check": "rests_on", "objects": "Vase", "support": "Table"},
        {"check": "no_overlap", "objects": "Stuck", "against": "Table"},
        {"check": "visible", "objects": ["Vase", "Stuck"], "min_visible_fraction": 0.2},
    ]
    before = scene_to_dict(scene)
    revision, steps = scene.revision, len(scene.history)
    a = diagnose(scene, *checks, views={"main": {"camera": "Cam"}}, defaults=True).to_dict()
    b = diagnose(scene, *checks, views={"main": {"camera": "Cam"}}, defaults=True).to_dict()
    copy = Scene.from_dict(scene_to_dict(scene))
    c = diagnose(copy, *checks, views={"main": {"camera": "Cam"}}, defaults=True).to_dict()
    for x in (a, b, c):
        x.pop("timings")
    assert a == b

    def strip(x):
        return [(f["code"], f.get("magnitude"), f["message"]) for f in x["findings"]]

    assert strip(a) == strip(c)
    assert scene.revision == revision and len(scene.history) == steps
    assert scene_to_dict(scene) == before


def test_findings_point_at_their_region_in_the_default_view():
    scene = room()
    scene.add_cube("Vase", size=0.2, position=(0.2, 0.1, 1.35))
    d = diagnose(
        scene,
        {"check": "rests_on", "objects": "Vase", "support": "Table"},
        views={"main": {"camera": "Cam"}},
        defaults=True,  # the default view is observed: findings get their regions in it
    )
    (f,) = [f for f in d.findings if f.code == "FLOATING"]
    obs = d.observed["main"]
    assert f.observation == obs.id and f.view == "main"
    x0, y0, x1, y1 = f.region
    inside = obs.objects_in_rect((x0, y0, x1, y1))
    assert inside[0]["name"] in ("Vase", "Table")
    assert any(o["name"] == "Vase" for o in inside)


# --------------------------------------------------------------------------- #
# Groups (prefabs, imported hierarchies) are one object
# --------------------------------------------------------------------------- #


def tower(scene: Scene, name: str, position, levels: int = 3) -> object:
    """A group of blocks: an empty parent with stacked 1 m cubes (a Minecraft-like tower)."""
    group = scene.add_object(name, position=position)
    for k in range(levels):
        scene.add_cube(f"{name} block {k}", size=1, parent=group, position=(0, 0, 0.5 + k))
    return group


def test_a_group_rests_and_is_corrected_as_one_object():
    scene = room()
    group = tower(scene, "Tower", (3, 2, 0.4))  # floats 0.4 above the floor
    d = diagnose(scene, {"check": "rests_on", "objects": "Tower", "support": "Floor"})
    (f,) = d.findings
    assert f.code == "FLOATING" and f.subjects[0]["name"] == "Tower"
    assert f.magnitude == pytest.approx(0.4, abs=1e-6) and f.evidence["parts"] == 3
    fs = FeedbackSession(scene, doc({"check": "rests_on", "objects": "Tower", "support": "Floor"},
                                    defaults=False), width=160, height=90)  # fmt: skip
    result = fs.apply(fs.propose())
    assert result.status == "applied" and group.position[2] == pytest.approx(0.0, abs=1e-6)
    assert scene.get_by_name("Tower block 2").world_position[2] == pytest.approx(2.5, abs=1e-6)


def test_groups_intersect_as_wholes_never_with_their_own_parts():
    scene = room()
    tower(scene, "Left", (3, 0, 0))
    tower(scene, "Right", (3.6, 0, 0))  # 0.4 into the left tower
    d = diagnose(scene, {"check": "no_overlap", "objects": ["Left", "Right"]})
    (f,) = d.findings
    assert f.code == "INTERSECTION"
    assert {s["name"] for s in f.subjects} == {"Left", "Right"}  # the groups, not blocks
    assert f.magnitude == pytest.approx(0.4, abs=1e-4)
    alone = diagnose(scene, {"check": "no_overlap", "objects": "Left"})
    assert not alone.findings  # its stacked blocks touch each other: one object


def test_groups_are_seen_and_framed_as_wholes():
    scene = room()
    tower(scene, "Tower", (0, 2, 0))
    d = diagnose(
        scene,
        {"check": "visible", "objects": "Tower", "min_pixels": 5},
        {"check": "framing", "objects": "Tower", "fill": {"max": 0.01}},
        {"check": "inside_region", "objects": "Tower",
         "region": {"box": {"min": [-1, 1, 0], "max": [1, 3, 2]}}},
        views={"main": {"camera": "Cam"}},
    )  # fmt: skip
    codes_by_check = {c.kind: c for c in d.checks}
    assert codes_by_check["visible"].status == "pass"
    obs = d.observed["main"]
    record = obs.record("Tower")
    assert record["parts"] == 3 and record["visible_pixels"] == sum(
        obs.record(f"Tower block {k}", geometry=False)["visible_pixels"] for k in range(3)
    )
    assert record["geometry"]["parts"] == 3
    assert [f.code for f in d.findings] == ["TOO_LARGE_IN_FRAME", "OUT_OF_REGION"]
    region = [f for f in d.findings if f.code == "OUT_OF_REGION"][0]
    assert region.magnitude == pytest.approx(1.0)  # 3 m tall in a 2 m box
