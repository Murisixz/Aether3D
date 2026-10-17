"""glTF animation interchange: skeletons, skins, morph targets and animations -- against
independent evaluations of the files, round trips, exactness classes, the Khronos
validator and malformed input."""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_animation_motion import blocky, humanoid
from test_animation_rig import cylinder

from aether3d import Scene, import_gltf
from aether3d.animation import FCurve
from aether3d.animation import rotation as R
from aether3d.animation.tracks import RotationTrack
from aether3d.errors import AssetFormatError, ValidationError
from aether3d.gltf import axes
from aether3d.rendering.overlay import animation_overlay
from aether3d.rendering.scene_data import world_matrices

ROOT = Path(__file__).resolve().parent.parent
KHRONOS = Path(__file__).parent / "data" / "khronos"
TOOL = ROOT / "tools" / "gltf-validator"
C4 = np.eye(4)
C4[:3, :3] = axes.AXIS_MATRIX


def validator_available() -> bool:
    return shutil.which("node") is not None and (TOOL / "node_modules" / "gltf-validator").is_dir()


def assert_valid(*paths: Path) -> None:
    """Every file passes the Khronos glTF Validator with no errors and no warnings (when it
    is installed; ``AETHER3D_REQUIRE_GLTF_VALIDATOR=1``, as in CI, requires it)."""
    if not validator_available():
        if os.environ.get("AETHER3D_REQUIRE_GLTF_VALIDATOR") == "1":
            pytest.fail("Khronos glTF Validator required but not installed "
                        "(run `npm ci --prefix tools/gltf-validator`)", pytrace=False)  # fmt: skip
        return
    result = subprocess.run(
        ["node", str(TOOL / "validate.mjs"), *map(str, paths)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )  # fmt: skip
    lines = result.stdout.splitlines()
    assert len(lines) == len(paths), result.stdout + result.stderr
    for line in lines:
        report = json.loads(line)
        assert report["errors"] == 0 and report["warnings"] == 0, report["messages"]


# --------------------------------------------------------------------------------------- #
# An independent glTF evaluator (NumPy only), for the Khronos samples
# --------------------------------------------------------------------------------------- #


class Doc:
    def __init__(self, path: Path) -> None:
        self.doc = json.loads(path.read_text())
        self.bufs = [base64.b64decode(b["uri"].split(",", 1)[1]) for b in self.doc["buffers"]]

    def accessor(self, i: int) -> np.ndarray:
        acc = self.doc["accessors"][i]
        view = self.doc["bufferViews"][acc["bufferView"]]
        dtype = {5126: np.float32, 5123: np.uint16, 5121: np.uint8}[acc["componentType"]]
        width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
        item = np.dtype(dtype).itemsize * width
        stride = view.get("byteStride", item)
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = self.bufs[view["buffer"]]
        rows = [np.frombuffer(raw, dtype, width, start + k * stride) for k in range(acc["count"])]
        return np.array(rows, np.float64)


def trs(node: dict, rotation=None) -> np.ndarray:
    t = np.array(node.get("translation", [0, 0, 0]), float)
    q = np.array(node.get("rotation", [0, 0, 0, 1]) if rotation is None else rotation, float)
    s = np.array(node.get("scale", [1, 1, 1]), float)
    m = np.eye(4)
    m[:3, :3] = R.matrices(q[None])[0] * s
    m[:3, 3] = t
    return m


def test_simple_skin_deforms_as_the_file_says():
    path = KHRONOS / "SimpleSkin/glTF-Embedded/SimpleSkin.gltf"
    f = Doc(path)
    times = f.accessor(5).ravel()
    quats = f.accessor(6)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)  # (the file's 0.707s: unit, as glTF says)
    ibm = f.accessor(4).reshape(-1, 4, 4).transpose(0, 2, 1)
    pos = f.accessor(1)
    joints = f.accessor(2).astype(int)
    weights = f.accessor(3)
    result = import_gltf(path)
    scene = result.scene
    skinned = scene.get_by_name("Node")
    for t in (0.0, 0.8, 1.7, 3.3, 5.5):
        # glTF: slerp (shortest path) between the sampler's keys; joint 2 under joint 1.
        k = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
        u = min(max((t - times[k]) / (times[k + 1] - times[k]), 0.0), 1.0)
        q = R.slerp(quats[k][None], quats[k + 1][None], np.array([u]), shortest=True)[0]
        nodes = f.doc["nodes"]
        g1 = trs(nodes[1])
        g2 = g1 @ trs(nodes[2], q)
        mats = [g1 @ ibm[0], g2 @ ibm[1]]
        want = np.zeros((len(pos), 3))
        for v in range(len(pos)):
            m = sum(weights[v, i] * mats[joints[v, i]] for i in range(4))
            want[v] = (m @ np.array([*pos[v], 1.0]))[:3]
        want = want @ axes.AXIS_MATRIX  # glTF -> Aether3D: C^T p (rows)
        overlay = animation_overlay(scene, t)
        local = overlay.meshes[skinned.id].positions.astype(np.float64)
        world = world_matrices(scene, overlay)[skinned.id]
        got = local @ world[:3, :3].T + world[:3, 3]
        assert np.allclose(got, want, atol=2e-6), t


def test_animated_morph_cube_blends_its_targets_as_the_file_says():
    path = KHRONOS / "AnimatedMorphCube/glTF-Binary/AnimatedMorphCube.glb"
    result = import_gltf(path)
    scene = result.scene
    cube = scene.get_by_name("AnimatedMorphCube")
    morph = scene.animation.morph_targets(cube)
    assert morph.names == ("Target 0", "Target 1")  # (no mesh.extras.targetNames in this file)
    clip = scene.animation.action(cube)
    curves = [clip.channel(f'morph["{n}"]') for n in morph.names]
    for t in np.linspace(0, 2, 9):
        w = [c.evaluate(t) for c in curves]
        want = cube.mesh.positions.astype(np.float64) + sum(
            wk * morph.positions[k].astype(np.float64) for k, wk in enumerate(w)
        )
        overlay = animation_overlay(scene, float(t))  # (None: nothing changes then)
        got = None if overlay is None else overlay.meshes.get(cube.id)
        base = cube.mesh.positions if got is None else got.positions
        assert np.allclose(base, want, atol=1e-6)


# --------------------------------------------------------------------------------------- #
# Exactness: what glTF can say is said exactly
# --------------------------------------------------------------------------------------- #


def roundtrip(scene: Scene, tmp_path: Path, **options):
    out = tmp_path / "out.glb"
    report = scene.export(out, **options)
    assert_valid(out)
    return report, import_gltf(out)


@pytest.mark.parametrize(
    "keys, interp, sampler",
    [
        ([(0, 0.0), (1, 2.0), (2.5, -1.0)], "linear", "LINEAR"),
        ([(0, 0.0), (1, 2.0), (2.5, -1.0)], "constant", "STEP"),
        ([(0, 0.0), (1, 2.0), (2.5, -1.0), (3, 0.5)], "bezier", "CUBICSPLINE"),
    ],
)
def test_linear_step_and_hermite_curves_export_exactly(tmp_path, keys, interp, sampler):
    s = Scene("Exact")
    cube = s.add_cube(name="Cube")
    s.animation.set_channel(cube, "transform.position.x", keys, interpolation=interp)
    report, back = roundtrip(s, tmp_path)
    anim = report.animations[0]
    assert anim["fitted_channels"] == 0 and anim["max_error"]["translation"] <= 1e-12
    doc = json.loads(_json_chunk(tmp_path / "out.glb"))
    assert doc["animations"][0]["samplers"][0]["interpolation"] == sampler
    src = s.animation.channel(cube, "transform.position.x")
    got = back.scene.animation.channel(back.scene.get_by_name("Cube"), "transform.position.x")
    t = np.linspace(-0.5, 3.5, 801)
    assert np.allclose(got.evaluate(t), src.evaluate(t), rtol=1e-6, atol=1e-6)  # (float32)


def test_slerp_tracks_export_as_linear_and_long_arcs_are_split(tmp_path):
    s = Scene("Slerp")
    cube = s.add_cube(name="Cube")
    q0 = [0.0, 0.0, 0.0, 1.0]
    q1 = R.axis_angle([0, 0, 1], np.array([math.radians(300.0)]))[0]  # 300 degrees
    track = RotationTrack.from_keys(
        [{"time": 0.0, "rotation": q0, "path": "stored"}, {"time": 2.0, "rotation": q1.tolist()}]
    )
    s.animation.set_channel(cube, "transform.rotation", track)
    report, back = roundtrip(s, tmp_path)
    assert report.animations[0]["fitted_channels"] == 0
    got = back.scene.animation.channel(back.scene.get_by_name("Cube"), "transform.rotation")
    t = np.linspace(0, 2, 41)
    a, b = track.evaluate(t), got.evaluate(t)
    assert np.allclose(np.abs(np.sum(a * b, axis=1)), 1.0, atol=1e-6)
    assert len(got) == 3  # split at the middle: each half under 180 degrees


def test_easings_and_euler_rotations_are_fitted_within_tolerance(tmp_path):
    s = Scene("Fitted")
    cube = s.add_cube(name="Cube")
    a = s.animation
    a.keyframe(cube, "transform.position.z", 0.0, 0.0, interpolation="elastic", ease="out")
    a.keyframe(cube, "transform.position.z", 2.0, 3.0)
    a.keyframe(cube, "transform.rotation_euler", 0.0, [0.0, 10.0, 0.0])
    a.keyframe(cube, "transform.rotation_euler", 2.0, [80.0, 60.0, 170.0])
    report, back = roundtrip(s, tmp_path, position_tolerance=1e-4, rotation_tolerance=0.01)
    anim = report.animations[0]
    assert anim["fitted_channels"] == 2
    assert anim["max_error"]["translation"] <= 1e-4 and anim["max_error"]["rotation"] <= 0.01
    other = back.scene.get_by_name("Cube")
    for t in np.linspace(0, 2, 201):
        wa = world_matrices(s, animation_overlay(s, float(t)))[cube.id]
        wb = world_matrices(back.scene, animation_overlay(back.scene, float(t)))[other.id]
        assert abs(wa[2, 3] - wb[2, 3]) <= 1e-4 + 1e-6
        ang = R.angle(R.from_matrices(wa[:3, :3][None]), R.from_matrices(wb[:3, :3][None]))[0]
        assert math.degrees(ang) <= 0.01 + 1e-4


def _json_chunk(path: Path) -> bytes:
    data = path.read_bytes()
    length = int.from_bytes(data[12:16], "little")
    return data[20 : 20 + length]


# --------------------------------------------------------------------------------------- #
# Round trips
# --------------------------------------------------------------------------------------- #


def rigged_scene() -> Scene:
    s = Scene("Rigged")
    steve = blocky(s)
    a = s.animation
    rig = a.rig(steve)
    model = rig.rest_model()
    for j in rig.joints:
        k = rig.index(j.name)
        box = s.add_cube(name=f"{j.name} Box", position=tuple(model[k][:3, 3] + [0, 0, 0.2]),
                         scale=(0.2, 0.2, 0.4))  # fmt: skip
        a.attach(box, steve, j.name)
    a.generate(steve, "walk")
    a.generate(steve, "jump", bind=False)
    arm = s.add_object("Arm", position=(2, 0, 0))
    a.add_rig(
        arm,
        [
            {"name": "Root"},
            {"name": "Mid", "parent": "Root", "position": [0, 1, 0]},
            {"name": "Tip", "parent": "Mid", "position": [0, 1, 0], "tail": [0, 0.2, 0]},
        ],
    )
    holder = s.add_object("Holder", position=(0, 1, 0))
    tube = s.add_object("Tube", mesh=cylinder(), position=(2, -1, 0), parent=holder)
    a.skin(tube, arm)
    bend = a.add_clip("Bend")
    bend.keyframe('joints["Mid"].rotation_euler.z', 0.0, 0.0)
    bend.keyframe('joints["Mid"].rotation_euler.z', 1.0, 60.0, interpolation="back", ease="out")
    bend.keyframe('joints["Mid"].rotation_euler.z', 2.0, 0.0)
    a.set_action(arm, bend)
    n = tube.mesh.vertex_count
    a.set_morph_targets(
        tube, [{"name": "Bulge", "positions": (tube.mesh.positions * [1, 0, 1]).tolist()}]
    )
    a.set_channel(tube, 'morph["Bulge"]', [(0, 0.0), (1, 0.5), (2, 0.0)])
    assert n > 0
    a.set_timeline(start=0, end=2.5)
    return s


def test_rigged_skinned_morphed_scenes_round_trip(tmp_path):
    s = rigged_scene()
    report, result = roundtrip(s, tmp_path, animation="clips")
    t_scene = result.scene
    assert not result.lossy
    assert {r["armature"] for r in result.animation["rigs"]} == {"Steve", "Arm"}
    assert result.animation["skins"] == 1 and result.animation["morph_targets"] == 1
    names = {c.name for c in t_scene.animation.clips()}
    assert {"Walk", "Jump", "Bend"} <= names
    assert t_scene.animation.action(t_scene.get_by_name("Steve")).name == "Walk"
    # the skinned tube keeps its parent and transform; its mesh rides a carrier node
    tube = t_scene.get_by_name("Tube")
    assert tube.parent.name == "Holder" and tube.position == (2.0, -1.0, 0.0)
    for t in np.linspace(0, 2.5, 11):
        oa, ob = animation_overlay(s, float(t)), animation_overlay(t_scene, float(t))
        wa, wb = world_matrices(s, oa), world_matrices(t_scene, ob)
        for obj in s.objects:
            other = t_scene.get_by_name(obj.name)
            assert np.allclose(wa[obj.id], wb[other.id], atol=2e-4), (obj.name, t)
        ma = oa.meshes[s.get_by_name("Tube").id].positions
        mb = ob.meshes[tube.id].positions
        assert np.allclose(ma, mb, atol=2e-4)
    # clips keep their loops (glTF holds; extras carry Aether3D's extrapolation)
    walk = t_scene.animation.clip("Walk")
    curve = next(ch for p, ch in walk.data.channels if p.startswith('joints["Leg.L"]'))
    assert curve.extrapolation[0] == "cycle"


def test_baked_export_carries_constraints_and_ik(tmp_path):
    s = Scene("Baked")
    target = s.add_cube(name="Target", position=(0.3, 0.6, 0.4))
    arm = humanoid(s)
    a = s.animation
    a.set_timeline(start=0, end=1, fps=24)
    a.set_channel(target, "transform.position.y", [(0, 0.6), (1, -0.2)])
    a.add_constraint(arm, {"name": "reach", "type": "ik", "chain": ["Thigh.R", "Shin.R", "Foot.R"],
                           "target": {"object": target.id}})  # fmt: skip
    report, result = roundtrip(s, tmp_path)  # auto -> bake
    assert "ANIMATION_BAKED" in {d.code for d in report.diagnostics}
    t_scene = result.scene
    other = t_scene.get_by_name("Hero")
    for frame in range(0, 25, 4):
        t = frame / 24
        pa = a.pose_joints(arm, t, "world")["Foot.R"][:3, 3]
        pb = t_scene.animation.pose_joints(other, t, "world")["Foot.R"][:3, 3]
        assert np.allclose(pa, pb, atol=1e-3), t


def test_animation_options_are_checked(tmp_path):
    s = rigged_scene()
    with pytest.raises(ValidationError):
        s.export(tmp_path / "x.glb", animation="sometimes")
    with pytest.raises(ValidationError):
        s.export(tmp_path / "x.glb", rotation_tolerance=0)
    none = s.export(tmp_path / "none.glb", animation="none")
    assert none.animations == [] and none.skins == 1
    assert_valid(tmp_path / "none.glb")


# --------------------------------------------------------------------------------------- #
# Malformed files
# --------------------------------------------------------------------------------------- #


def _write(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "bad.gltf"
    path.write_text(json.dumps(doc))
    return path


def _sample() -> dict:
    return json.loads((KHRONOS / "SimpleSkin/glTF-Embedded/SimpleSkin.gltf").read_text())


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda d: d["animations"][0]["channels"].append(d["animations"][0]["channels"][0]),
         "ANIMATION_DUPLICATE_TARGET"),
        (lambda d: d["accessors"][6].update(count=11), "ANIMATION_OUTPUT_COUNT"),
        (lambda d: d["skins"][0].update(joints=[1, 1]), "INVALID_SKIN"),
        (lambda d: d["animations"][0]["channels"][0].update(sampler=7), "INVALID_REFERENCE"),
        (lambda d: d["animations"][0]["samplers"][0].update(interpolation="SMOOTH"), None),
    ],
)  # fmt: skip
def test_malformed_animations_and_skins_are_refused(tmp_path, mutate, code):
    doc = _sample()
    mutate(doc)
    with pytest.raises(AssetFormatError) as info:
        import_gltf(_write(tmp_path, doc))
    if code is not None:
        assert info.value.code == code


def test_non_increasing_times_are_refused(tmp_path):
    doc = _sample()
    times = Doc(KHRONOS / "SimpleSkin/glTF-Embedded/SimpleSkin.gltf").accessor(5).ravel()
    times[3] = times[2]
    raw = bytearray(base64.b64decode(doc["buffers"][3]["uri"].split(",", 1)[1]))
    raw[0:48] = times.astype(np.float32).tobytes()
    doc["buffers"][3]["uri"] = (
        "data:application/octet-stream;base64," + base64.b64encode(bytes(raw)).decode()
    )
    with pytest.raises(AssetFormatError) as info:
        import_gltf(_write(tmp_path, doc))
    assert info.value.code == "ANIMATION_INPUT_NON_INCREASING"


def test_curves_keep_their_extrapolation_only_for_aether3d(tmp_path):
    s = Scene("Loop")
    cube = s.add_cube(name="Cube")
    curve = FCurve.from_keys([(0, 0.0), (1, 1.0), (2, 0.0)]).replace(extrapolation="cycle")
    s.animation.set_channel(cube, "transform.position.x", curve)
    _, back = roundtrip(s, tmp_path)
    got = back.scene.animation.channel(back.scene.get_by_name("Cube"), "transform.position.x")
    assert got.extrapolation == ("cycle", "cycle")
    assert got.evaluate(2.5) == pytest.approx(curve.evaluate(2.5), abs=1e-6)
