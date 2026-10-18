"""glTF and Stage 18 rigs (requirements 11 and 16): a character of several independently
modeled meshes -- a body, a jacket bound by weight transfer, a helmet attached rigidly --
exports what glTF can say (skeleton, skins, attachments, baked motion and corrective
shapes), says what it cannot (non-linear skinning, authoring data), passes the Khronos
validator and comes back numerically."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from test_animation_gltf import assert_valid

from aether3d import Scene, import_gltf
from aether3d.errors import UnsupportedFeatureError
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import world_matrices
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface


def _inflated(shapes, indices, by):
    out = []
    for i in indices:
        sh = copy.deepcopy(shapes[i])
        for key in ("radius", "radius_b"):
            if key in sh:
                sh[key] += by
        if "radii" in sh:
            sh["radii"] = [r + by for r in sh["radii"]]
        out.append(sh)
    return out


@pytest.fixture(scope="module")
def meshes():
    shapes = humanoid_shapes()
    body = implicit_surface(shapes, 0.02)
    jacket = implicit_surface(_inflated(shapes, (0, 2, 3, 4, 9, 14), 0.015), 0.02)
    helmet = implicit_surface([{"type": "sphere", "center": [0.0, 0.0, 1.64], "radius": 0.125}],
                              0.02)  # fmt: skip
    return body, jacket, helmet


def character(meshes):
    body_mesh, jacket_mesh, helmet_mesh = meshes
    s = Scene("Outfit")
    body = s.add_object("Body", mesh=body_mesh)
    s.add_object("Jacket", mesh=jacket_mesh)
    s.add_object("Helmet", mesh=helmet_mesh)
    a = s.animation
    rep = a.autorig([body, "Jacket", "Helmet"], name="Rig",
                    parts={"Jacket": "transfer", "Helmet": "attach"})  # fmt: skip
    assert rep["weights"]["Jacket"]["mode"] == "transfer"
    a.generate("Rig", "walk", forward="-y")
    a.pose_character("Rig", [{"do": "look_at", "target": [0.6, -2.0, 1.5]}], time=0.5)
    return s


def deformed(scene, name, t):
    obj = scene.get_by_name(name)
    ov = animation_overlay(scene, t)
    mesh = ov.meshes.get(obj.id, obj.mesh) if ov is not None else obj.mesh
    w = world_matrices(scene, ov)[obj.id]
    return mesh.positions.astype(np.float64) @ w[:3, :3].T + w[:3, 3]


def test_a_multi_mesh_character_round_trips_through_gltf(meshes, tmp_path):
    s = character(meshes)
    a = s.animation
    out = tmp_path / "outfit.glb"
    report = s.export(out, animation="bake").to_dict()
    assert_valid(out)
    codes = {d["code"] for d in report["diagnostics"]}
    assert "RIG_AUTHORING_DATA" in codes and "SKINNING_METHOD_LOSSY" not in codes
    info = next(d for d in report["diagnostics"] if d["code"] == "RIG_AUTHORING_DATA")
    assert "joint roles" in info["message"] and "control definitions" in info["message"]
    result = import_gltf(out)
    t_scene = result.scene
    b = t_scene.animation
    rig, t_rig = a.rig(s.get_by_name("Rig")), None
    t_arm = next(o for o in t_scene.objects if b.rig(o) is not None)
    t_rig = b.rig(t_arm)
    # The skeleton (the file may order joints otherwise): names, hierarchy, rest pose,
    # and the Stage 18 semantics (extras).
    assert sorted(j.name for j in t_rig.joints) == sorted(j.name for j in rig.joints)
    model_a, model_b = rig.rest_model(), t_rig.rest_model()
    for j in rig.joints:
        k = t_rig.index(j.name)
        jb = t_rig.joints[k]
        parent_a = None if j.parent < 0 else rig.joints[j.parent].name
        parent_b = None if jb.parent < 0 else t_rig.joints[jb.parent].name
        assert parent_a == parent_b, j.name
        np.testing.assert_allclose(model_b[k], model_a[rig.index(j.name)], atol=1e-5)
        assert (jb.role, jb.deform, jb.limits) == (j.role, j.deform, j.limits), j.name
    # Two skins (body, jacket) and the attached helmet.
    assert result.animation["skins"] == 2
    for name in ("Body", "Jacket"):
        src, dst = s.get_by_name(name), t_scene.get_by_name(name)
        ska = a._owned[("skin", src._id)]._definition
        skb = b._owned[("skin", dst._id)]._definition
        assert skb.vertex_count == dst.mesh.vertex_count
        # Same vertices at rest (the file may reorder or split them): match by position.
        at = {p.tobytes(): i for i, p in enumerate(src.mesh.positions.astype(np.float32))}
        idx = np.array([at[p.tobytes()] for p in dst.mesh.positions.astype(np.float32)])
        wa = ska.dense()[idx]
        names_a = [rig.joints[rig.index_of_id(j)].name for j in ska.joints]
        names_b = [t_rig.joints[t_rig.index_of_id(j)].name for j in skb.joints]
        wb = np.zeros_like(wa)
        dense_b = skb.dense()
        for col, nm in enumerate(names_b):
            wb[:, names_a.index(nm)] += dense_b[:, col]
        np.testing.assert_allclose(wb, wa, atol=2e-6, err_msg=name)
        # And they move alike: the motion is baked at the frames (keys within the export
        # tolerance of every frame; glTF interpolates linearly between them).
        fps = a.fps.fps
        for frame in (0, 7, 12, 19):
            t = frame / fps
            got = deformed(t_scene, name, t)
            want = deformed(s, name, t)[idx]
            assert np.abs(got - want).max() < 2e-3, (name, t, np.abs(got - want).max())
    helmet_a, helmet_b = s.get_by_name("Helmet"), t_scene.get_by_name("Helmet")
    assert b.rig(t_arm).attachments and helmet_b.mesh.vertex_count == helmet_a.mesh.vertex_count
    for t in (0.0, 0.5):
        wa = world_matrices(s, animation_overlay(s, t))[helmet_a.id]
        wb = world_matrices(t_scene, animation_overlay(t_scene, t))[helmet_b.id]
        assert np.abs(wa - wb).max() < 3e-3, t


def test_non_linear_skinning_is_a_reported_loss(meshes, tmp_path):
    s = character(meshes)
    a = s.animation
    rep = a.apply_rig_fix("Rig", [{"code": "JOINT_COLLAPSE", "safety": "approval", "operations": [
        {"call": "skin_method", "mesh": "Body", "method": "blended",
         "dq": {"joint": "ForeArm.L", "share": 1.0}}]}])  # fmt: skip
    assert rep["applied"]
    with pytest.raises(UnsupportedFeatureError) as e:
        s.export(tmp_path / "dq.glb", animation="bake")
    assert "SKINNING_METHOD_LOSSY" in str(e.value)
    report = s.export(tmp_path / "dq.glb", animation="bake", allow_lossy=True).to_dict()
    lossy = [d for d in report["diagnostics"] if d["code"] == "SKINNING_METHOD_LOSSY"]
    assert len(lossy) == 1 and lossy[0]["lossy"] and "'Body'" in lossy[0]["message"]
    assert_valid(tmp_path / "dq.glb")


def test_corrective_shapes_bake_into_morph_weights(meshes, tmp_path):
    s = character(meshes)
    a = s.animation
    body = s.get_by_name("Body")
    pos = body.mesh.positions.astype(np.float64)
    near = np.linalg.norm(pos - [0.35, 0.0, 1.14], axis=1) < 0.08  # (the left elbow)
    delta = np.zeros_like(pos)
    delta[near] = [0.0, 0.01, 0.0]
    a.set_morph_targets(body, [{"name": "elbow_fix", "positions": delta}])
    rig = a.rig("Rig")
    raw = rig.to_json()
    raw["correctives"] = [{"name": "elbow.L", "mesh": body.id, "target": "elbow_fix",
                           "joint": "ForeArm.L", "driver": "swing", "range": [0, 90]}]  # fmt: skip
    a.set_rig("Rig", raw)
    a.pose_character("Rig", [{"do": "rotate", "joint": "ForeArm.L", "euler": [80, 0, 0]}],
                     time=0.25)  # fmt: skip
    report = s.export(tmp_path / "fix.glb", animation="auto").to_dict()  # (bakes: the shapes)
    assert any(d["code"] == "ANIMATION_BAKED" and "corrective shapes" in d["message"]
               for d in report["diagnostics"])  # fmt: skip
    assert_valid(tmp_path / "fix.glb")
    t_scene = import_gltf(tmp_path / "fix.glb").scene
    t_body = t_scene.get_by_name("Body")
    clip = t_scene.animation.action(t_body)
    curve = clip.channel('morph["elbow_fix"]')
    pose = a.evaluate(0.25)
    want = a.rigging(0.25).morph_extra[body.id]["elbow_fix"] + pose.morphs.get(body.id, {}).get(
        "elbow_fix", 0.0
    )
    assert want > 0.5 and abs(curve.evaluate(0.25) - want) < 0.02
    clips_report = s.export(tmp_path / "clips.glb", animation="clips", allow_lossy=True).to_dict()
    assert any(d["code"] == "ANIMATION_NOT_BAKED" and "corrective shapes" in d["message"]
               for d in clips_report["diagnostics"])  # fmt: skip
