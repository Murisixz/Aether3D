"""glTF import of skeletons, skins, morph targets and animations.

**Skeletons.** The joints of the file's skins -- with any nodes between them -- become a
rig on an *armature object*: the joints' common parent node (Blender's "Armature"), or
a new root object named ``Armature`` when the joints are roots. Joint nodes are not
scene objects; their rest pose is their node TRS (glTF's default pose; a skin's bind
pose lives in its inverse bind matrices). Other nodes below joints become objects
attached to the joint (rigid parts), parented to the armature. Joint names are kept
when the rig allows them (unique, no quotes or backslashes); others are reported.

**Skins** bind with the file's inverse bind matrices (glTF's meaning is Aether3D's:
mesh space at bind to joint space; the skinned mesh node's own transform is ignored by
both) and up to eight influences per vertex (the strongest, renormalized, when there
are more). **Morph targets** (``POSITION`` and ``NORMAL`` offsets) are named from
``mesh.extras.targetNames`` (else ``Target n``) with the node's or the mesh's
``weights`` as defaults.

**Animations** become clips: one per glTF animation and animated object (the
armature's clip holds its joints' channels). ``STEP`` keys become constant keys,
``LINEAR`` linear keys (rotations: shortest-path slerp, glTF's rule), ``CUBICSPLINE``
Bezier keys whose slopes are the tangents (rotations: ``cubic`` tracks) -- the same
curves, exactly. The first animation plays as the objects' actions; the others are
clips ready to play. Times must increase; every count and reference is checked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import INFO
from ..errors import AssetFormatError, ResourceLimitError
from . import axes

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

_C4 = np.eye(4)
_C4[:3, :3] = axes.AXIS_MATRIX
PATHS = ("translation", "rotation", "scale", "weights")
INTERPOLATIONS = ("LINEAR", "STEP", "CUBICSPLINE")


def matrix_from_gltf(m: np.ndarray) -> np.ndarray:
    return _C4.T @ m @ _C4


def safe_name(name: Any, fallback: str, used: set[str]) -> str:
    """A rig-safe, unique name (quotes, backslashes and control characters replaced)."""
    base = name.strip() if isinstance(name, str) and name.strip() else fallback
    base = re.sub(r'["\\\x00-\x1f]', "_", base)[:120] or fallback
    out, k = base, 1
    while out in used:
        k += 1
        out = f"{base}.{k:03d}"
    used.add(out)
    return out


@dataclass
class SkeletonPlan:
    joints: set[int] = field(default_factory=set)  # joint nodes
    top: dict[int, int] = field(default_factory=dict)  # joint -> its skeleton's top joint
    armature_node: dict[int, int | None] = field(default_factory=dict)  # top -> armature node
    # filled while the scene is built:
    armature_obj: dict[int | None, SceneObject] = field(default_factory=dict)
    rigs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # armature id -> joints
    joint_name: dict[int, str] = field(default_factory=dict)
    joint_armature: dict[int, str] = field(default_factory=dict)  # joint node -> armature id
    model: dict[int, np.ndarray] = field(default_factory=dict)  # joint -> rest (armature space)
    names_used: dict[str, set[str]] = field(default_factory=dict)
    attachments: list[tuple[SceneObject, str, str]] = field(default_factory=list)
    carriers: dict[int, str] = field(default_factory=dict)  # node -> object id it carries for
    skinned: list[tuple[SceneObject, int, int]] = field(default_factory=list)  # obj, skin, mesh
    morphed: list[tuple[SceneObject, int, dict[str, Any]]] = field(default_factory=list)


def plan_skeletons(reader: Any, parent: dict[int, int]) -> SkeletonPlan:
    """Which nodes are joints, and the armature of each skeleton."""
    plan = SkeletonPlan()
    skins = reader.array("skins")
    skin_joints: set[int] = set()
    for s_index in range(len(skins)):
        skin, where = reader.item("skins", s_index, "/skins")
        joints = skin.get("joints")
        if not isinstance(joints, list) or not joints:
            raise AssetFormatError("joints must be a non-empty array", f"{where}/joints")
        if len(joints) > 4096:
            raise ResourceLimitError(f"{where}: more than 4096 joints")
        for j in joints:
            reader.item("nodes", j, f"{where}/joints")
        if len(set(joints)) != len(joints):
            raise AssetFormatError("a skin lists a joint twice", f"{where}/joints", "INVALID_SKIN")
        skin_joints.update(joints)
    # Aether3D's own rigs need no skin: their joint nodes say so.
    from .writer import EXTRAS_KEY

    for index, node in enumerate(reader.array("nodes")):
        extras = node.get("extras") if isinstance(node, dict) else None
        stored = extras.get(EXTRAS_KEY) if isinstance(extras, dict) else None
        if isinstance(stored, dict) and isinstance(stored.get("joint"), str):
            skin_joints.add(index)
    if not skin_joints:
        return plan
    joints = set(skin_joints)
    for j in skin_joints:  # nodes between joints are joints too
        chain = []
        p = parent.get(j)
        while p is not None:
            chain.append(p)
            p = parent.get(p)
        top = max((i for i, a in enumerate(chain) if a in skin_joints), default=-1)
        joints.update(chain[:top])
    plan.joints = joints
    for j in joints:
        t = j
        while parent.get(t) in joints:
            t = parent[t]
        plan.top[j] = t
        plan.armature_node[t] = parent.get(t)
    return plan


def joint_rest(reader: Any, node: dict[str, Any], where: str) -> tuple[Any, Any, Any]:
    t = reader.transform(node, where)
    return t.position, t.rotation, t.scale


def compose(pos: Any, rot: Any, scl: Any) -> np.ndarray:
    from ..math3d import compose_matrix

    return np.array(compose_matrix(tuple(pos), tuple(rot), tuple(scl)), np.float64)


# --------------------------------------------------------------------------------------- #
# Skins and morph targets
# --------------------------------------------------------------------------------------- #


def bind_skins(reader: Any, scene: Scene, plan: SkeletonPlan) -> int:
    from ..animation.rig import MAX_INFLUENCES

    count = 0
    for obj, s_index, m_index in plan.skinned:
        skin, where = reader.item("skins", s_index, "/skins")
        nodes = skin["joints"]
        arms = {plan.joint_armature.get(j) for j in nodes}
        if None in arms or len(arms) != 1:
            reader.lossy(
                "SKIN_SPANS_RIGS",
                "the skin's joints belong to more than one skeleton; imported unskinned",
                where,
            )
            continue
        arm_id = arms.pop()
        weights = reader.mesh_skin.get(m_index)
        if weights is None:
            reader.lossy(
                "SKIN_WITHOUT_WEIGHTS",
                "a skinned mesh without JOINTS_0/WEIGHTS_0; imported unskinned",
                where,
            )
            continue
        idx, w = weights
        if idx.size and int(idx.max()) >= len(nodes):
            raise AssetFormatError(
                f"JOINTS index {int(idx.max())} is beyond the skin's {len(nodes)} joints",
                where,
                "INVALID_SKIN",
            )
        if "inverseBindMatrices" in skin:
            ibm = reader.accessor(
                skin["inverseBindMatrices"], f"{where}/inverseBindMatrices", "IBM"
            )
            if len(ibm) != len(nodes):
                raise AssetFormatError(
                    f"{len(ibm)} inverse bind matrices for {len(nodes)} joints",
                    f"{where}/inverseBindMatrices",
                    "INVALID_SKIN",
                )
            mats = ibm.astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)  # column-major
        else:
            mats = np.tile(np.eye(4), (len(nodes), 1, 1))
        if not np.all(np.isfinite(mats)):
            raise AssetFormatError("inverse bind matrices must be finite", where, "INVALID_SKIN")
        mats = np.array([matrix_from_gltf(m) for m in mats])
        w = np.where(w > 0, w, 0.0)
        if w.shape[1] > MAX_INFLUENCES:
            order = np.argsort(-w, axis=1, kind="stable")[:, :MAX_INFLUENCES]
            idx = np.take_along_axis(idx, order, axis=1)
            w = np.take_along_axis(w, order, axis=1)
            reader.lossy(
                "INFLUENCES_TRIMMED",
                f"vertices with more than {MAX_INFLUENCES} influences keep the strongest",
                where,
            )
        empty = w.sum(axis=1) <= 0
        if np.any(empty):
            idx = idx.copy()
            w = w.copy()
            idx[empty, 0], w[empty, 0] = 0, 1.0
            reader.lossy(
                "UNWEIGHTED_VERTICES",
                "vertices with no weight follow the skin's first joint",
                where,
                int(empty.sum()),
            )
        names = [plan.joint_name[j] for j in nodes]
        armature = scene._objects[arm_id]
        try:
            scene.animation.skin(
                obj,
                armature,
                weights={
                    "indices": idx.astype(np.int64),
                    "weights": w,
                    "joints": names,
                    "inverse_bind": mats,
                },
            )
        except Exception as exc:  # (a malformed skin the checks above did not catch)
            raise AssetFormatError(f"invalid skin: {exc}", where, "INVALID_SKIN") from None
        count += 1
    return count


def add_morphs(reader: Any, scene: Scene, plan: SkeletonPlan) -> int:
    count = 0
    for obj, m_index, node in plan.morphed:
        targets = reader.mesh_targets.get(m_index)
        if not targets or obj.mesh is None:
            continue
        gmesh = reader.array("meshes")[m_index]
        names_raw = None
        extras = gmesh.get("extras") if isinstance(gmesh, dict) else None
        if isinstance(extras, dict) and isinstance(extras.get("targetNames"), list):
            names_raw = extras["targetNames"]
        weights = node.get("weights", gmesh.get("weights") if isinstance(gmesh, dict) else None)
        if weights is not None and (
            not isinstance(weights, list)
            or len(weights) != len(targets)
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in weights)
        ):
            raise AssetFormatError(
                f"weights must list {len(targets)} numbers", f"/meshes/{m_index}/weights"
            )
        used: set[str] = set()
        out = []
        for k, t in enumerate(targets):
            raw = names_raw[k] if names_raw is not None and k < len(names_raw) else None
            name = safe_name(raw, f"Target {k}", used)
            entry: dict[str, Any] = {"name": name, "positions": t["POSITION"]}
            if t.get("NORMAL") is not None and obj.mesh.normals is not None:
                entry["normals"] = t["NORMAL"]
            if weights is not None and float(weights[k]) != 0.0:
                entry["default"] = float(weights[k])
            out.append(entry)
        if any("normals" in e for e in out) and not all("normals" in e for e in out):
            for e in out:
                e.setdefault("normals", np.zeros_like(e["positions"]))
        scene.animation.set_morph_targets(obj, out)
        reader.morph_names[obj.id] = [e["name"] for e in out]
        count += 1
    return count


# --------------------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------------------- #


def hermite_curve(times: np.ndarray, values: np.ndarray, ins: np.ndarray, outs: np.ndarray):
    """An F-curve of unweighted Bezier keys with these slopes: the cubic Hermite spline
    through the keys (glTF's ``CUBICSPLINE``), exactly."""
    from ..animation.curves import FCurve

    n = times.shape[0]
    curve = FCurve.from_arrays(times, values, interpolation="bezier", handles="free")
    if n < 2:
        return curve
    d = np.diff(times)
    left = np.concatenate([[d[0]], d]) / 3.0
    right = np.concatenate([d, [d[-1]]]) / 3.0
    return curve.replace(
        {"left_dt": -left, "left_dv": -left * ins, "right_dt": right, "right_dv": right * outs}
    )


_EXTRAPOLATIONS = ("constant", "linear", "cycle", "cycle_offset", "mirror")


def _extrapolated(mode: tuple[str, str] | None, channel: Any) -> Any:
    """``channel`` repeating as Aether3D authored it (``extras``), when it can."""
    if mode is None or tuple(channel.extrapolation) == mode:
        return channel
    if len(channel) < 2 or not channel.times[-1] > channel.times[0]:
        return channel
    return channel.replace(extrapolation=list(mode))


def _curves(times, values, interp, ins, outs):
    from ..animation.curves import FCurve

    if interp == "CUBICSPLINE":
        return hermite_curve(times, values, ins, outs)
    kind = "linear" if interp == "LINEAR" else "constant"
    return FCurve.from_arrays(times, values, interpolation=kind, handles="free")


def import_animations(
    reader: Any, scene: Scene, plan: SkeletonPlan, node_object: dict[int, str]
) -> list[dict[str, Any]]:
    """Clips from the file's animations (see the module docstring)."""
    from ..animation import paths as P
    from ..animation.clip import Clip
    from ..animation.tracks import RotationTrack

    anims = reader.array("animations")
    if not anims:
        return []
    system = scene.animation
    summary = []
    total = 0
    used_names: set[str] = set(system._clip_names)
    for a_index in range(len(anims)):
        anim, where = reader.item("animations", a_index, "/animations")
        channels = anim.get("channels")
        samplers = anim.get("samplers")
        if not isinstance(channels, list) or not isinstance(samplers, list):
            raise AssetFormatError("channels and samplers must be arrays", where)
        seen: set[tuple[int, str]] = set()
        per_object: dict[str, dict[str, Any]] = {}
        ignored = 0
        for c_index, ch in enumerate(channels):
            loc = f"{where}/channels/{c_index}"
            if not isinstance(ch, dict) or not isinstance(ch.get("target"), dict):
                raise AssetFormatError("a channel has a target object", loc)
            target = ch["target"]
            path = target.get("path")
            if "node" not in target or path not in PATHS:
                ignored += 1  # (KHR_animation_pointer and unknown paths)
                continue
            node = target["node"]
            reader.item("nodes", node, f"{loc}/target/node")
            if (node, path) in seen:
                raise AssetFormatError(
                    "two channels animate the same node and path", loc, "ANIMATION_DUPLICATE_TARGET"
                )
            seen.add((node, path))
            s_index = ch.get("sampler")
            if (
                not isinstance(s_index, int)
                or isinstance(s_index, bool)
                or not 0 <= s_index < len(samplers)
            ):
                raise AssetFormatError(
                    f"invalid sampler {s_index!r}", f"{loc}/sampler", "INVALID_REFERENCE"
                )
            sampler = samplers[s_index]
            sloc = f"{where}/samplers/{s_index}"
            if not isinstance(sampler, dict):
                raise AssetFormatError("a sampler is an object", sloc)
            interp = sampler.get("interpolation", "LINEAR")
            if interp not in INTERPOLATIONS:
                raise AssetFormatError(f"unknown interpolation {interp!r}", f"{sloc}/interpolation")
            times = (
                reader.accessor(sampler.get("input"), f"{sloc}/input", "anim_input")
                .astype(np.float64)
                .ravel()
            )
            if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
                raise AssetFormatError(
                    "animation times must be finite and strictly increasing",
                    f"{sloc}/input",
                    "ANIMATION_INPUT_NON_INCREASING",
                )
            purpose = {"translation": "anim_vec3", "scale": "anim_vec3",
                       "rotation": "anim_rotation", "weights": "anim_weights"}[path]  # fmt: skip
            out = reader.accessor(sampler.get("output"), f"{sloc}/output", purpose).astype(
                np.float64
            )
            n = times.size
            total += n
            if total > reader.limits.max_animation_keys:
                raise ResourceLimitError(
                    f"{loc}: more than {reader.limits.max_animation_keys} animation keys in total"
                )
            width = {"translation": 3, "scale": 3, "rotation": 4}.get(path)
            if path == "weights":
                morph = reader.morph_names.get(node_object.get(node, ""))
                if morph is None:
                    ignored += 1
                    continue
                width = len(morph)
            per_key = 3 if interp == "CUBICSPLINE" else 1
            flat = out.reshape(-1)
            if flat.size != n * per_key * width:
                raise AssetFormatError(
                    f"output holds {flat.size} values; {n} keys need {n * per_key * width}",
                    f"{sloc}/output",
                    "ANIMATION_OUTPUT_COUNT",
                )
            block = flat.reshape(n, per_key, width)
            if per_key == 3:
                ins, vals, outs = block[:, 0], block[:, 1], block[:, 2]
            else:
                ins = outs = None
                vals = block[:, 0]
            extrap = None
            c_extras = ch.get("extras")
            c_stored = c_extras.get("aether3d") if isinstance(c_extras, dict) else None
            if isinstance(c_stored, dict) and isinstance(c_stored.get("extrapolation"), list):
                pair = c_stored["extrapolation"]
                if len(pair) == 2 and all(m in _EXTRAPOLATIONS for m in pair):
                    extrap = tuple(pair)
            kind = node_object.get(node)
            joint = plan.joint_name.get(node)
            if path == "weights":
                owner, prefix = kind, ""
            elif joint is not None:
                owner = plan.joint_armature[node]
                prefix = f'joints["{joint}"].'
            elif kind is not None and node not in plan.carriers and kind not in reader.skinned_ids:
                owner, prefix = kind, "transform."
            else:
                ignored += 1  # (nodes not imported; a skinned mesh's transform moves nothing)
                continue
            items = per_object.setdefault(owner, {})
            if path == "weights":
                names = reader.morph_names[kind]
                for k, nm in enumerate(names):
                    items[f'morph["{nm}"]'] = _extrapolated(extrap, _curves(
                        times, vals[:, k], interp,
                        None if ins is None else ins[:, k], None if outs is None else outs[:, k],
                    ))  # fmt: skip
                continue
            if path == "rotation":
                conv = np.stack([vals[:, 0], -vals[:, 2], vals[:, 1], vals[:, 3]], axis=1) + 0.0
                norms = np.linalg.norm(conv, axis=1)
                if np.any(norms < 1e-6):
                    raise AssetFormatError("zero rotation in an animation", f"{sloc}/output")
                if np.any(np.abs(norms - 1.0) > 1e-3):
                    reader.note(
                        "ROTATION_NOT_UNIT",
                        INFO,
                        "animated rotations were normalized",
                        f"{sloc}/output",
                    )
                if interp == "STEP":
                    track = RotationTrack.from_arrays(times, conv, interpolation="constant")
                elif interp == "LINEAR":
                    track = RotationTrack.from_arrays(times, conv, interpolation="slerp")
                else:
                    tin = np.stack([ins[:, 0], -ins[:, 2], ins[:, 1], ins[:, 3]], axis=1) + 0.0
                    tout = np.stack([outs[:, 0], -outs[:, 2], outs[:, 1], outs[:, 3]], axis=1) + 0.0
                    track = RotationTrack.from_arrays(times, conv, interpolation="cubic")
                    track = track.replace({"in": tin, "out": tout, "quats": conv})
                items[f"{prefix}rotation"] = _extrapolated(
                    extrap if extrap is None or "linear" not in extrap else None, track
                )
                continue
            if path == "translation":
                order, signs = (0, 2, 1), (1.0, -1.0, 1.0)
            else:
                order, signs = (0, 2, 1), (1.0, 1.0, 1.0)
            field = "position" if path == "translation" else "scale"
            for c, axis in enumerate("xyz"):
                src, sign = order[c], signs[c]
                items[f"{prefix}{field}.{axis}"] = _extrapolated(extrap, _curves(
                    times, sign * vals[:, src] + 0.0, interp,
                    None if ins is None else sign * ins[:, src] + 0.0,
                    None if outs is None else sign * outs[:, src] + 0.0,
                ))  # fmt: skip
        if ignored:
            reader.note(
                "ANIMATION_CHANNELS_IGNORED",
                INFO,
                f"{ignored} channels animate nothing Aether3D imported (pointer paths, "
                "nodes outside the scene, skinned mesh nodes, weights without targets)",
                where,
                ignored,
            )
        raw_name = anim.get("name")
        base = (
            raw_name if isinstance(raw_name, str) and raw_name.strip() else f"Animation {a_index}"
        )
        # Aether3D's own files say which objects played the clip as their action.
        actions = None
        extras = anim.get("extras")
        stored = extras.get("aether3d") if isinstance(extras, dict) else None
        if isinstance(stored, dict) and isinstance(stored.get("actions"), list):
            actions = {a for a in stored["actions"] if isinstance(a, str)}
        clips = []
        for owner, items in per_object.items():
            obj = scene._objects[owner]
            label = base if len(per_object) == 1 else f"{base} ({obj.name})"
            name = safe_name(label, f"Animation {a_index}", used_names)
            clip = Clip(P.OBJECT)
            for path, ch in sorted(items.items()):
                clip = clip.with_channel(path, ch)
            meta = json.dumps({"gltf": {"animation": a_index, "name": base}}, sort_keys=True,
                              separators=(",", ":"))  # fmt: skip
            clip = Clip(clip.target, clip.channels, clip.events, clip.range, meta)
            system.add_clip(name, clip)
            if (owner in actions) if actions is not None else a_index == 0:
                system.set_action(obj, name)
            clips.append({"clip": name, "object": obj.name, "channels": len(items)})
        summary.append({"animation": a_index, "name": base, "clips": clips})
    return summary
