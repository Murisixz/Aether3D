"""Rigid-part characters (Stage 18, requirement 10): Minecraft-style characters animated
by their own object transforms -- no armature, no skin, no weights -- Stage 17's
bone-attached blocks unchanged, and optional, verified conversions between the two."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene
from aether3d.animation.script import AnimationScriptError, run_animation_script
from aether3d.cli import main
from aether3d.errors import AnimationError
from aether3d.rendering.png import encode_png
from aether3d.rendering.scene_data import world_matrices

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "examples"))

from animation_common import blocky_character, pixel_skin  # noqa: E402


def box(scene, name, size, center, parent=None, position=(0, 0, 0)):
    o = scene.add_object(name, parent=parent, position=position)
    with o.edit_mesh() as m:
        m.add_box(size, center=center)
    return o


def block_hero(scene=None, *, swap_names=False):
    """A rigid-part hero: pivots at the shoulders and hips (as modeled), the head and the
    torso pivoting about their centres, a sword in a hand."""
    s = scene or Scene("Blocks")
    hero = s.add_object("Hero")
    torso = box(s, "Torso", (0.5, 0.25, 0.75), (0, 0, 0), parent=hero, position=(0, 0, 1.125))
    box(s, "Head", 0.5, (0, 0, 0), parent=torso, position=(0, 0, 0.625))
    right, left = ("Left", "Right") if swap_names else ("Right", "Left")
    arm = box(s, f"{right} Arm", (0.25, 0.25, 0.75), (0, 0, -0.3), parent=torso,
              position=(-0.375, 0, 0.3))  # fmt: skip
    box(s, f"{left} Arm", (0.25, 0.25, 0.75), (0, 0, -0.3), parent=torso, position=(0.375, 0, 0.3))
    box(s, f"{right} Leg", (0.25, 0.25, 0.75), (0, 0, -0.375), parent=hero,
        position=(-0.125, 0, 0.75))  # fmt: skip
    box(s, f"{left} Leg", (0.25, 0.25, 0.75), (0, 0, -0.375), parent=hero,
        position=(0.125, 0, 0.75))  # fmt: skip
    box(s, "Sword", (0.05, 0.05, 0.9), (0, 0, 0.45), parent=arm, position=(0, -0.05, -0.62))
    return s, hero


def no_bones(scene):
    a = scene.animation
    return not any(a.rig(o) is not None for o in scene.objects) and not any(
        kind == "skin" for kind, _ in a._owned
    )


def test_parts_are_found_by_name_with_sides_and_pivots_measured():
    s, hero = block_hero()
    rep = s.animation.rigid_parts(hero)
    parts = rep["parts"]
    assert {r: p["object"] for r, p in parts.items()} == {
        "head": "Head", "torso": "Torso", "arm.R": "Right Arm", "arm.L": "Left Arm",
        "leg.R": "Right Leg", "leg.L": "Left Leg",
    }  # fmt: skip
    # Shoulders and hips where they were modeled; the neck and the waist from the boxes.
    assert parts["arm.L"]["pivot_source"] == "origin" and parts["leg.L"]["pivot_source"] == "origin"
    assert parts["head"]["pivot"] == pytest.approx([0, 0, 1.5])
    assert parts["torso"]["pivot"] == pytest.approx([0, 0, 0.75])
    assert parts["leg.L"]["end"] == pytest.approx([0.125, 0, 0.0])
    assert rep["hips"] == pytest.approx([0, 0, 0.75]) and rep["diagnostics"] == []
    # A left arm named "Right Arm" (the viewer's right): the geometry decides, and says so.
    s2, hero2 = block_hero(swap_names=True)
    rep2 = s2.animation.rigid_parts(hero2)
    assert rep2["parts"]["arm.L"]["object"] == "Right Arm"
    assert {d["code"] for d in rep2["diagnostics"]} == {"SIDE_NAME_MISMATCH"}
    # Explicit roles, and the errors.
    a = s.animation
    named = a.rigid_parts(hero, parts={"head": "Head", "leg.L": "Left Leg"})
    assert set(named["parts"]) == {"head", "leg.L"}
    for bad, code in (
        ({"parts": {"tail": "Head"}}, "INVALID_ARGUMENT"),
        ({"parts": {"head": "Nope"}}, None),
        ({"forward": "up"}, "INVALID_ARGUMENT"),
    ):
        with pytest.raises(AnimationError) as e:
            a.rigid_parts(hero, **bad)
        assert code is None or e.value.code == code
    box(s, "Spare Head", 0.3, (0, 0, 0), parent=hero, position=(1, 0, 1))
    with pytest.raises(AnimationError) as e:
        a.rigid_parts(hero)
    assert e.value.code == "AMBIGUOUS_PART"


@pytest.mark.parametrize("kind", ["walk", "run", "jump", "land", "idle", "wave", "head_turn"])
def test_rigid_motions_are_transform_clips_without_bones(kind):
    s, hero = block_hero()
    a = s.animation
    steps = len(s.history)
    rep = a.generate_rigid(hero, kind)
    assert len(s.history) == steps + 1 and no_bones(s) and rep["armature"] is None
    for obj_name, clip_name in rep["clips"].items():
        clip = a.clip(clip_name).data
        assert all(path.startswith("transform.") for path, _ in clip.channels), obj_name
        assert a.action(s.get_by_name(obj_name)).name == clip_name
    v = rep["verification"]
    assert v["samples"] == 49 and v["position"] < 2e-3 and v["rotation_deg"] < 0.1, v
    assert rep["max_error"]["degrees"] <= 0.05 + 1e-9
    if kind in ("walk", "run"):
        assert rep["travel"] > 0 and hero.name in rep["clips"]
        assert all(f["sliding_max"] < 3e-3 for f in rep["feet"].values()), rep["feet"]
    s.history.undo()
    assert len(s.history) == steps and not a.clips()


def test_an_in_place_motion_stops_the_previous_travel():
    s, hero = block_hero()
    a = s.animation
    a.generate_rigid(hero, "walk")
    assert a.action(hero) is not None
    rep = a.generate_rigid(hero, "idle")
    assert a.action(hero) is None and "Hero" in rep["cleared"]
    legs = [s.get_by_name("Left Leg"), s.get_by_name("Right Leg")]
    # Idle leaves the legs at rest: the walk's leg clips stop too.
    assert all(a.action(o) is None or a.action(o).name in rep["clips"].values() for o in legs)
    # A clip of the user's own on the root is kept (and said).
    a.set_channel(hero, "transform.position.y", [(0, 0.0), (2, -1.0)], interpolation="linear")
    rep = a.generate_rigid(hero, "wave")
    assert a.action(hero) is not None
    assert any(d["code"] == "PART_ANIMATED" for d in rep["diagnostics"])
    assert rep["verification"]["position"] < 2e-3  # (measured relative to the moving root)


def test_rigid_characters_skip_the_rigging_pass():
    s, hero = block_hero()
    a = s.animation
    a.generate_rigid(hero, "run")
    pose = a.evaluate(0.3)
    assert not pose.joints and not pose.morphs  # (no skeleton to pose, nothing to deform)
    assert set(pose.ids) <= {o.id for o in [hero, *hero.iter_descendants()]}
    ov = a.overlay(0.3)
    w = world_matrices(s, ov)
    assert np.isfinite(w[s.get_by_name("Sword").id]).all()


def test_nova_stage17_blocky_motions_match_v0_18_0():
    """Stage 17's bone-attached blocky character, unchanged: its generated motions move
    its parts exactly as v0.18.0 did (fixture computed by the v0.18.0 sources)."""
    fixture = json.loads((ROOT / "tests/data/v0.18.0/nova_motion.json").read_text())
    assert fixture["version"] == "0.18.0"
    for kind, opts in (("walk", {}), ("run", {}), ("jump", {"height": 0.55}), ("idle", {})):
        s = Scene("Nova")
        nova = blocky_character(s, "Nova", encode_png(pixel_skin()))
        a = s.animation
        rep = a.generate(nova, kind, name=kind, forward="-y", **opts)
        data = fixture["motions"][kind]
        clip = a.clip(rep["clip"]).data
        assert {path: len(ch) for path, ch in clip.channels} == data["keys"], kind
        parts = [o for o in s.objects if o.name.startswith("Nova ")]
        assert [o.name for o in parts] == fixture["parts"]
        for t, expected in zip(data["times"], data["matrices"], strict=True):
            w = world_matrices(s, a.overlay(t))
            got = np.array([w[o.id][:3, :] for o in parts])
            np.testing.assert_allclose(got, np.array(expected), atol=1e-9, err_msg=kind)


def test_conversions_are_optional_verified_and_leave_the_source():
    s, hero = block_hero()
    a = s.animation
    a.generate_rigid(hero, "walk")
    before = {o.name: (o.local_matrix.copy(), o.mesh) for o in [hero, *hero.iter_descendants()]}
    actions = {o.name: a.action(o).name for o in s.objects if a.action(o) is not None}
    steps = len(s.history)
    rep = a.convert_character(hero, "skeleton")
    assert len(s.history) == steps + 1
    arm = s.get_by_name(rep["armature"])
    rig = a.rig(arm)
    assert [j.role for j in rig.joints] == ["hips", "spine", "head", "upper_arm.L",
                                            "upper_arm.R", "thigh.L", "thigh.R"]  # fmt: skip
    assert a.action(arm).name == rep["clip"]
    v = rep["verification"]
    assert v["position"] < 2e-3 and v["rotation_deg"] < 0.1, v
    # The source: untouched, hidden.
    for name, (m, mesh) in before.items():
        o = s.get_by_name(name)
        assert np.array_equal(o.local_matrix, m) and o.mesh is mesh and not o.visible
    assert {o.name: a.action(o).name for o in s.objects
            if a.action(o) is not None and o.name in actions} == actions  # fmt: skip
    assert s.get_by_name("Sword (rig)").parent.name == "Right Arm (rig)"
    s.history.undo()
    assert len(s.history) == steps and all(o.visible for o in s.objects)
    assert rep["armature"] not in {o.name for o in s.objects}

    # Stage 17's Nova (attached blocks) -> a rigid hierarchy, and back.
    s2 = Scene("Nova")
    nova = blocky_character(s2, "Nova", encode_png(pixel_skin()))
    a2 = s2.animation
    a2.generate(nova, "run", forward="-y")
    rigid = a2.convert_character(nova, "rigid")
    assert rigid["verification"]["position"] < 2e-3, rigid["verification"]
    root = s2.get_by_name(rigid["root"])
    assert a2.rig(root) is None and set(rigid["clips"]) >= {rigid["root"]}
    back = a2.convert_character(root, "skeleton", name="Nova Again")
    assert back["verification"]["position"] < 3e-3, back["verification"]
    assert a2.rig(s2.get_by_name("Nova Again")) is not None
    with pytest.raises(AnimationError) as e:
        a2.convert_character(root, "rigid")
    assert e.value.code == "RIG_MISSING"
    with pytest.raises(AnimationError):
        a2.convert_character(nova, "bones")


def test_rigid_scripts_and_the_command_line(tmp_path, capsys):
    s, hero = block_hero()
    out = run_animation_script(s, {"format": "aether3d.animation-script", "version": 1,
                                   "operations": [
        {"op": "rigid_parts", "root": "Hero"},
        {"op": "generate_rigid", "root": {"object": "Hero"}, "motion": "run", "name": "Dash",
         "stride": 1.2},
        {"op": "convert_character", "source": "Hero", "to": "skeleton", "hide_original": False,
         "roles": {"head": "Head", "torso": "Torso", "arm.L": "Left Arm", "arm.R": "Right Arm",
                   "leg.L": "Left Leg", "leg.R": "Right Leg"}},
    ]})  # fmt: skip
    ops = out["operations"]
    assert set(ops[0]["result"]["parts"]) == {"head", "torso", "arm.L", "arm.R", "leg.L", "leg.R"}
    assert ops[1]["result"]["clips"]["Hero"] == "Dash Hero"
    assert ops[2]["result"]["verification"]["position"] < 2e-3
    rev = s.revision
    for bad, where in (
        ({"op": "generate_rigid", "root": "Hero", "motion": "fly"}, "operations[0]"),
        ({"op": "generate_rigid", "root": "Hero", "motion": "walk", "roles": ["Head"]},
         "operations[0].roles"),
        ({"op": "generate_rigid", "root": "Hero", "motion": "walk", "code": "x"}, "operations[0]"),
        ({"op": "convert_character", "source": "Hero", "to": "mesh"}, "operations[0]"),
    ):  # fmt: skip
        with pytest.raises(AnimationScriptError) as e:
            run_animation_script(s, {"format": "aether3d.animation-script", "version": 1,
                                     "operations": [bad]})  # fmt: skip
        assert e.value.location.startswith(where)
        assert s.revision == rev
    s3, _ = block_hero()
    path = tmp_path / "hero.aether"
    s3.save(path)

    def run_json(*argv):
        code = main([*argv, "--json"])
        return code, json.loads(capsys.readouterr().out)

    code, rep = run_json("rig", "parts", str(path), "--root", "Hero")
    assert code == 0 and rep["parts"]["head"]["object"] == "Head"
    code, rep = run_json("rig", "convert", str(path), "--source", "Hero", "--to", "skeleton")
    assert code == 0 and rep["armature"] == "Hero Rig" and rep["output"]
