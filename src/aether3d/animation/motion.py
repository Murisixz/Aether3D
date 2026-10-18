"""Motion editing: mirroring, root motion, foot contacts and planting, retargeting.

* :func:`mirror_clip` swaps left and right (by the rig's name pairs: ``Arm.L``/``Arm.R``,
  ``LeftArm``/``RightArm``, ...) and reflects across the armature's YZ plane: positions
  ``x -> -x``, quaternions ``(x, y, z, w) -> (x, -y, -z, w)``, Euler ``(rx, -ry, -rz)``.
  It assumes the rig's left and right joints have mirrored rest frames (symmetric rigs).
* :func:`extract_root_motion` moves a root joint's horizontal travel (and optionally its
  yaw) to the armature object's own channels, leaving the joint in place -- or drops it
  (``mode="in_place"``) for cycles that should not travel.
* :func:`foot_contacts` finds when feet are planted by sampling the evaluated motion --
  near the ground (``height``) and moving no faster than ``speed`` relative to their
  stance's steady velocity: zero for a planted foot, the skating velocity when a body
  outruns its steps -- and measures each contact's **sliding** (how far the foot moves
  while planted) and any **penetration** below the ground.
* :func:`plant_feet` corrects sliding (after Kovar, Schreiner and Gleicher, 2002): for
  each foot it creates a target object pinned during each contact in the pose the foot
  touched down with (lifted if a heel or toe of it was below the ground), keyed to follow
  the foot between contacts, and an IK constraint on the leg chain whose influence is
  keyed 1 during contacts and blends out around them. Everything it makes is ordinary,
  editable data; the report gives the sliding before and after.
* :func:`retarget` transfers a clip between skeletons with a joint mapping (by name by
  default): each mapped joint gets the source's rotation *relative to its rest pose* in
  armature space applied to the target's rest pose (so A-pose and T-pose rigs with the
  same intent transfer), the root's translation is scaled by the ratio of the rigs' root
  heights, and every other joint keeps the target's own bone lengths. Unmapped joints and
  rest poses that differ a lot (over 30 degrees between mapped bone directions) are
  reported -- arbitrary skeletons are not interchangeable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import paths as P
from . import rotation as R
from .clip import Clip
from .curves import FCurve
from .tracks import RotationTrack

if TYPE_CHECKING:
    from .system import AnimationSystem


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


# --------------------------------------------------------------------------------------- #
# Mirroring
# --------------------------------------------------------------------------------------- #


def _negate_curve(curve: FCurve) -> FCurve:
    a = curve.arrays()
    a["values"] = -a["values"]
    a["left_dv"] = -a["left_dv"]
    a["right_dv"] = -a["right_dv"]
    return curve.replace(a)


def _mirror_track(track: RotationTrack) -> RotationTrack:
    a = track.arrays()
    q = a["quats"].copy()
    q[:, 1] = -q[:, 1]
    q[:, 2] = -q[:, 2]
    a["quats"] = q
    for k in ("in", "out"):
        t = a[k].copy()
        t[:, 1] = -t[:, 1]
        t[:, 2] = -t[:, 2]
        a[k] = t
    return track.replace(a)


def mirror_clip(clip: Clip, rig: Any) -> Clip:
    """``clip`` with left and right swapped and reflected (see the module docstring)."""
    items: dict[str, Any] = {}
    for path, ch in clip.channels:
        spec = P.parse(path, clip.target)
        assert not isinstance(spec, list)
        new_path = path
        if spec.group == "joint":
            other = rig.mirror_name(spec.name) if rig is not None else None
            if other is not None:
                new_path = path.replace(f'joints["{spec.name}"]', f'joints["{other}"]', 1)
        new = ch
        flips = (spec.field == "position" and spec.component == 0) or (
            spec.field == "rotation_euler" and spec.component in (1, 2)
        )
        if flips:
            new = _negate_curve(ch)
        elif spec.field == "rotation":
            new = _mirror_track(ch)
        items[new_path] = new
    return Clip(clip.target, tuple(sorted(items.items())), clip.events, clip.range, clip.metadata)


# --------------------------------------------------------------------------------------- #
# Root motion
# --------------------------------------------------------------------------------------- #


def extract_root_motion(
    clip: Clip, joint: str, *, mode: str = "extract", yaw: bool = False
) -> tuple[Clip, dict[str, Any]]:
    """Move (``extract``) or drop (``in_place``) ``joint``'s horizontal translation (and
    with ``yaw``, its rotation about Z) out of the joint channels."""
    if mode not in ("extract", "in_place"):
        raise _err("INVALID_ARGUMENT", "mode is 'extract' or 'in_place'", "mode")
    items = dict(clip.channels)
    moved = {}
    for axis in ("x", "y"):
        path = f'joints["{joint}"].position.{axis}'
        ch = items.get(path)
        if ch is None:
            continue
        first = float(ch.values[0])
        flat = FCurve.from_keys([(float(ch.times[0]), first)], interpolation="constant")
        items[path] = flat
        if mode == "extract":
            a = ch.arrays()
            a["values"] = a["values"] - first
            items[f"transform.position.{axis}"] = ch.replace(a)
        moved[axis] = float(ch.values[-1] - first)
    if yaw:
        path = f'joints["{joint}"].rotation'
        tr = items.get(path)
        if tr is not None:
            q = tr.quats
            swing, twist = R.swing_twist(q, np.array([[0.0, 0.0, 1.0]]))
            a = tr.arrays()
            a["quats"] = R.normalize(swing)
            items[path] = tr.replace(a)
            if mode == "extract":
                yaw_deg = np.degrees(2 * np.arctan2(twist[:, 2], twist[:, 3]))
                yaw_deg = R.unwrap_degrees(yaw_deg)
                items["transform.rotation_euler.z"] = FCurve.from_arrays(
                    tr.times, yaw_deg, handles="auto_clamped"
                )
            moved["yaw"] = True
    return Clip(
        clip.target, tuple(sorted(items.items())), clip.events, clip.range, clip.metadata
    ), {"moved": moved, "mode": mode}


# --------------------------------------------------------------------------------------- #
# Contacts
# --------------------------------------------------------------------------------------- #


def _contact_points(rig: Any, joint: str) -> dict[str, np.ndarray]:
    """Points of a foot joint that touch the ground, in the joint's own space: its bone's
    end (the toe, or a one-joint leg's end) and, when that end is below the joint, the
    point under the joint at the toe's height (the heel)."""
    k = rig.index(joint)
    tail = np.asarray(rig.tails()[k], np.float64)
    model = rig.rest_model()[k]
    toe = (model @ np.array([*tail, 1.0]))[:3]
    origin = model[:3, 3]
    points = {"toe": tail}
    if toe[2] < origin[2] - 1e-6:
        heel = np.array([origin[0], origin[1], toe[2], 1.0])
        points["heel"] = (np.linalg.inv(model) @ heel)[:3]
    return points


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` index of every run of True in ``mask``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(int(a), int(b) - 1) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def foot_contacts(
    system: AnimationSystem,
    armature: Any,
    feet: list[str],
    *,
    start: float | None = None,
    end: float | None = None,
    rate: float = 120.0,
    ground: float = 0.0,
    height: float = 0.03,
    speed: float = 0.3,
    min_duration: float = 0.05,
) -> dict[str, Any]:
    """Planted intervals of each foot and their sliding and penetration (sampled).

    Each foot is watched at its ground points (the toe -- its bone's end -- and the heel
    under the joint, see :func:`_contact_points`): a point is planted while it is within
    ``height`` of the ground (world ``z = ground``) and moves slower than ``speed``
    horizontally than the steady velocity of its stance. A stance is a run of samples near
    the ground and slower than the body (the rig's root joint; at least ``speed``) -- a
    swinging foot moves about twice as fast as its body -- and its steady velocity is the
    run's median: zero for a planted foot, so this is the plain ``speed`` test, and the
    skating velocity for a body outrunning its steps, whose sliding is measured rather than
    missed. A foot is planted while either point is.
    **Sliding** is how far a planted point moves horizontally while it stays planted (a
    heel-to-toe roll is not sliding); **penetration** how far any point goes below the
    ground. The defaults are the feedback check's (``foot_contacts`` expectations), so
    every interface measures alike."""
    from .report import sample_joints

    obj = system._object(armature, "armature")
    rig = system.rig(obj)
    if rig is None:
        raise _err("RIG_MISSING", f"{obj.name!r} has no rig", "armature")
    s = system.start if start is None else float(start)
    e = system.end if end is None else float(end)
    if not (math.isfinite(s) and math.isfinite(e) and e > s):
        raise _err("INVALID_RANGE", "contacts need start < end", "range")
    for key, value in (("rate", rate), ("height", height), ("speed", speed), ("ground", ground)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise _err("INVALID_ARGUMENT", f"{key} must be a finite number", key)
    if not (1.0 <= rate <= 2000.0 and height > 0 and speed > 0):
        raise _err("INVALID_ARGUMENT", "rate 1..2000; height and speed positive", "parameters")
    n = int(min(20_000, max(8, round((e - s) * rate) + 1)))
    ts = np.linspace(s, e, n)
    dt = ts[1] - ts[0]
    out: dict[str, Any] = {
        "ground": ground,
        "height": height,
        "speed": speed,
        "feet": {},
        "basis": "sampled",
    }
    root = next(j.name for j in rig.joints if j.parent < 0)
    for foot in feet:
        if rig.index(foot) is None:
            raise _err("JOINT_MISSING", f"no joint {foot!r}", "feet")
    sampled = sample_joints(system, obj, ts, list(dict.fromkeys([root, *feet])))
    body = sampled[root][:, :2, 3]
    stance_bound = np.maximum(speed, np.linalg.norm(np.gradient(body, dt, axis=0), axis=1))
    for foot in feet:
        w = sampled[foot]
        local = _contact_points(rig, foot)
        pts = {name: (w @ np.array([*p, 1.0]))[:, :3] for name, p in local.items()}
        ankle = w[:, :3, 3]
        planted_by = {}
        for name, pos in pts.items():
            vel = np.gradient(pos[:, :2], dt, axis=0)
            stance = (pos[:, 2] - ground <= height) & (np.linalg.norm(vel, axis=1) <= stance_bound)
            planted_by[name] = np.zeros(n, bool)
            for a, b in _runs(stance):
                steady = np.median(vel[a : b + 1], axis=0)
                planted_by[name][a : b + 1] = (
                    np.linalg.norm(vel[a : b + 1] - steady, axis=1) <= speed
                )
        planted = np.zeros(n, bool)
        for m in planted_by.values():
            planted |= m
        lowest = min(float((pos[:, 2] - ground).min()) for pos in pts.values())
        intervals = []
        k = 0
        while k < n:
            if not planted[k]:
                k += 1
                continue
            j = k
            while j + 1 < n and planted[j + 1]:
                j += 1
            if ts[j] - ts[k] >= min_duration:
                slide = 0.0
                for name, pos in pts.items():
                    m = planted_by[name][k : j + 1]
                    # each run of this point being planted inside the interval
                    a = 0
                    while a < m.size:
                        if not m[a]:
                            a += 1
                            continue
                        b = a
                        while b + 1 < m.size and m[b + 1]:
                            b += 1
                        seg = pos[k + a : k + b + 1, :2]
                        slide = max(slide, float(np.max(np.linalg.norm(seg - seg[0], axis=1))))
                        a = b + 1
                intervals.append(
                    {
                        "start": float(ts[k]),
                        "end": float(ts[j]),
                        "sliding": slide,
                        "anchor": [float(ankle[k, 0]), float(ankle[k, 1]), float(ankle[k, 2])],
                    }
                )
            k = j + 1
        out["feet"][foot] = {
            "contacts": intervals,
            "sliding_max": max((c["sliding"] for c in intervals), default=0.0),
            "penetration": max(0.0, -lowest),
            "lowest": lowest,
        }
    return out


def _influence_keys(
    spans: list[tuple[float, float]], start: float, end: float, blend: float
) -> list[tuple[float, float]]:
    """Keys of an influence that is 1 during ``spans`` and 0 away from them, ramping over
    ``blend`` seconds (spans closer than two ramps stay joined at 1)."""
    merged: list[list[float]] = []
    for a, b in sorted(spans):
        if merged and a - merged[-1][1] <= 2.0 * blend:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if not merged:
        return [(start, 0.0)]
    keys: list[tuple[float, float]] = []
    for a, b in merged:
        if a - blend > start:
            if not keys:
                keys.append((start, 0.0))
            keys.append((a - blend, 0.0))
        keys += [(a, 1.0), (b, 1.0)] if b > a else [(a, 1.0)]
        if b + blend < end:
            keys.append((b + blend, 0.0))
    if keys[-1][0] < end and keys[-1][1] == 0.0:
        keys.append((end, 0.0))
    out: dict[float, tuple[float, float]] = {}
    for k in keys:
        out.setdefault(k[0], k)
    return sorted(out.values())


def plant_feet(
    system: AnimationSystem,
    armature: Any,
    legs: dict[str, list[str]],
    *,
    start: float | None = None,
    end: float | None = None,
    rate: float = 30.0,
    ground: float = 0.0,
    height: float = 0.03,
    speed: float = 0.3,
    blend: float = 0.08,
    pole: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plant feet during their contacts (see the module docstring). ``legs``: foot joint
    -> its leg chain (three joints, hip to foot). One undo step; returns the contacts and
    the sliding before and after. Each foot is pinned in the pose it touched down with --
    its position and orientation at the contact's start, together (its IK chain ends at
    the target's rotation) -- lifted if a heel or toe point of that pose was below the
    ground, so planting never pushes a foot into it. The contacts are found as
    :func:`foot_contacts` finds them (a skating stance included); the sliding before and
    after is measured with its defaults."""
    arm = system._object(armature, "armature")
    rig = system.rig(arm)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    feet = list(legs)
    stances = foot_contacts(
        system,
        arm,
        feet,
        start=start,
        end=end,
        rate=rate * 2,
        ground=ground,
        height=height,
        speed=speed,
    )
    before = foot_contacts(system, arm, feet, start=start, end=end, ground=ground)
    s = system.start if start is None else float(start)
    e = system.end if end is None else float(end)
    from .report import sample_world

    n = int(max(8, round((e - s) * rate) + 1))
    ts = np.linspace(s, e, n)
    with system._op(f"Plant the feet of {arm.name!r}"):
        for foot, chain in legs.items():
            if len(chain) != 3 or chain[-1] != foot:
                raise _err(
                    "INVALID_ARGUMENT",
                    "a leg is [hip, knee, foot] ending at the foot",
                    f"legs[{foot}]",
                )
            w = sample_world(system, arm, ts, joint=foot)
            pos = w[:, :3, 3].copy()
            lin = w[:, :3, :3]
            rot = R.from_matrices(lin / np.linalg.norm(lin, axis=1, keepdims=True))
            contacts = stances["feet"][foot]["contacts"]
            if contacts:
                # The touch-down poses, each sampled at its contact's start (one matrix:
                # the position and the orientation of the same instant).
                down = sample_world(system, arm, np.array([c["start"] for c in contacts]),
                                    joint=foot)  # fmt: skip
                points = np.array([[*p, 1.0] for p in _contact_points(rig, foot).values()])
            for k, c in enumerate(contacts):
                inside = np.flatnonzero((ts >= c["start"]) & (ts <= c["end"]))
                if inside.size == 0:
                    continue
                lowest = float(np.min((down[k] @ points.T)[2]))
                lift = max(0.0, ground - lowest)  # (no heel or toe below the ground)
                pos[inside] = down[k][:3, 3] + np.array([0.0, 0.0, lift])
                lin0 = down[k][:3, :3]
                rot[inside] = R.from_matrices((lin0 / np.linalg.norm(lin0, axis=0))[None])[0]
            for i in range(1, n):
                rot[i] = R.align(rot[i - 1], rot[i])
            target = system.scene.add_object(
                system.scene.unique_name(f"{arm.name} Plant {foot}"),
                position=tuple(pos[0].tolist()),
            )
            for axis, i in (("x", 0), ("y", 1), ("z", 2)):
                system.set_channel(
                    target,
                    f"transform.position.{axis}",
                    FCurve.from_arrays(ts, pos[:, i], interpolation="linear"),
                )
            system.set_channel(target, "transform.rotation", RotationTrack.from_arrays(ts, rot))
            name = f"plant {foot}"
            constraint = {
                "name": name,
                "type": "ik",
                "chain": list(chain),
                "target": {"object": target.id},
                "end": "target",  # the foot keeps its planted orientation too
            }
            if pole and foot in pole:
                constraint["pole"] = pole[foot]
            system.add_constraint(arm, constraint)
            # Influence: 1 in contacts, blending over `blend` seconds around them.
            keys = _influence_keys([(c["start"], c["end"]) for c in contacts], s, e, blend)
            system.set_channel(
                arm,
                f'constraints["{name}"].influence',
                FCurve.from_keys(keys, interpolation="sine", handles="auto_clamped"),
            )
    after = foot_contacts(system, arm, feet, start=start, end=end, ground=ground)
    return {
        "before": {f: before["feet"][f]["sliding_max"] for f in feet},
        "after": {f: after["feet"][f]["sliding_max"] for f in feet},
        "contacts": {f: stances["feet"][f]["contacts"] for f in feet},
        "basis": "sampled",
    }


# --------------------------------------------------------------------------------------- #
# Retargeting
# --------------------------------------------------------------------------------------- #

_PREFIXES = ("mixamorig:", "mixamorig_", "bip01 ", "bip01_", "def-", "orig_")


def _norm_name(name: str) -> str:
    low = name.lower()
    for p in _PREFIXES:
        if low.startswith(p):
            low = low[len(p) :]
    return low.replace(" ", "").replace("_", "").replace("-", "")


def auto_mapping(source: Any, target: Any) -> dict[str, str]:
    """Source joint -> target joint by equal names (case, spaces, underscores and common
    prefixes ignored)."""
    tgt = {_norm_name(j.name): j.name for j in target.joints}
    return {j.name: tgt[_norm_name(j.name)] for j in source.joints if _norm_name(j.name) in tgt}


def retarget(
    system: AnimationSystem,
    source_armature: Any,
    target_armature: Any,
    clip: Any,
    *,
    mapping: dict[str, str] | None = None,
    root: str | None = None,
    fps: Any = None,
    start: float | None = None,
    end: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """A new clip for the target rig (see the module docstring); returns ``{clip,
    mapped, unmapped_source, unmapped_target, rest_mismatch, root_scale}``."""
    from .rig import forward_kinematics, local_matrices
    from .time import FrameRate, frame_range

    src_obj = system._object(source_armature, "source_armature")
    tgt_obj = system._object(target_armature, "target_armature")
    src, tgt = system.rig(src_obj), system.rig(tgt_obj)
    if src is None or tgt is None:
        raise _err("RIG_MISSING", "both armatures need rigs", "armature")
    handle = system.clip(clip)
    data = handle.data
    m = auto_mapping(src, tgt) if mapping is None else dict(mapping)
    for s_name, t_name in m.items():
        if src.index(s_name) is None or tgt.index(t_name) is None:
            raise _err(
                "INVALID_ARGUMENT",
                f"mapping {s_name!r} -> {t_name!r} names a missing joint",
                "mapping",
            )
    if not m:
        raise _err("RETARGET_UNMAPPED", "no joint maps between the rigs", "mapping")
    root_name = root or next((j.name for j in src.joints if j.parent < 0 and j.name in m), None)
    rate = system.fps if fps is None else FrameRate(fps, "fps")
    crange = data.frame_range()
    a = crange[0] if start is None else float(start)
    b = crange[1] if end is None else float(end)
    frames = frame_range(rate, a, b)
    if frames.size < 2:
        raise _err("INVALID_RANGE", "retargeting needs two frames or more", "range")
    times = rate.frame_times(frames)
    src_rest_model = src.rest_model()
    tgt_rest_model = tgt.rest_model()

    def rot_of(mats):
        lin = mats[:, :3, :3] / np.linalg.norm(mats[:, :3, :3], axis=1, keepdims=True)
        return R.from_matrices(lin)

    src_rest_rot = rot_of(src_rest_model)
    tgt_rest_rot = rot_of(tgt_rest_model)
    # Rest-pose agreement: angle between mapped bone directions in armature space.
    mismatch = {}
    src_tails, tgt_tails = src.tails(), tgt.tails()
    for s_name, t_name in m.items():
        si, ti = src.index(s_name), tgt.index(t_name)
        ds = src_rest_model[si][:3, :3] @ src_tails[si]
        dt_ = tgt_rest_model[ti][:3, :3] @ tgt_tails[ti]
        if np.linalg.norm(ds) > 1e-9 and np.linalg.norm(dt_) > 1e-9:
            cosang = float(ds @ dt_ / (np.linalg.norm(ds) * np.linalg.norm(dt_)))
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
            if ang > 30.0:
                mismatch[s_name] = round(ang, 3)
    root_scale = 1.0
    if root_name is not None:
        hs = float(src_rest_model[src.index(root_name)][2, 3])
        ht = float(tgt_rest_model[tgt.index(m[root_name])][2, 3])
        if abs(hs) > 1e-9:
            root_scale = ht / hs
    # Evaluate the source clip's joint channels directly (the clip alone, as authored).
    pos, rot, scl = (x.copy() for x in src.rest_trs())
    n_src = len(src.joints)
    t_rot = np.zeros((len(times), len(tgt.joints), 4))
    root_pos = np.zeros((len(times), 3))
    for fi, t in enumerate(times.tolist()):
        p, q, sc = pos.copy(), rot.copy(), scl.copy()
        for k in range(n_src):
            jn = src.joints[k].name
            pre = f'joints["{jn}"].'
            for axis, ci in (("x", 0), ("y", 1), ("z", 2)):
                ch = data.channel(f"{pre}position.{axis}")
                if ch is not None:
                    p[k, ci] = float(ch.evaluate(t))
                ch = data.channel(f"{pre}scale.{axis}")
                if ch is not None:
                    sc[k, ci] = float(ch.evaluate(t))
            tr = data.channel(f"{pre}rotation")
            if tr is not None:
                q[k] = tr.evaluate(t)
            elif any(data.channel(f"{pre}rotation_euler.{ax}") is not None for ax in "xyz"):
                e = R.to_euler(q[k][None])[0]
                for ax, ci in (("x", 0), ("y", 1), ("z", 2)):
                    ch = data.channel(f"{pre}rotation_euler.{ax}")
                    if ch is not None:
                        e[ci] = float(ch.evaluate(t))
                q[k] = R.from_euler(e)[0]
        model = forward_kinematics(local_matrices(p, q, sc), src.parents, src.depth)
        src_rot = rot_of(model)
        # Target: start from rest; mapped joints take the source's rest-relative rotation.
        tgt_model_rot = tgt_rest_rot.copy()
        for s_name, t_name in m.items():
            si, ti = src.index(s_name), tgt.index(t_name)
            delta = R.multiply(src_rot[si][None], R.conjugate(src_rest_rot[si])[None])[0]
            tgt_model_rot[ti] = R.multiply(delta[None], tgt_rest_rot[ti][None])[0]
        # Unmapped joints follow their parents rigidly (their rest local rotation).
        for k, j in enumerate(tgt.joints):
            if j.name in m.values() or j.parent < 0:
                continue
            tgt_model_rot[k] = R.multiply(
                tgt_model_rot[j.parent][None],
                R.multiply(R.conjugate(tgt_rest_rot[j.parent])[None], tgt_rest_rot[k][None]),
            )[0]
        for k, j in enumerate(tgt.joints):
            parent_rot = np.array([0.0, 0.0, 0.0, 1.0]) if j.parent < 0 else tgt_model_rot[j.parent]
            t_rot[fi, k] = R.normalize(
                R.multiply(R.conjugate(parent_rot)[None], tgt_model_rot[k][None])
            )[0]
        if root_name is not None:
            si = src.index(root_name)
            root_pos[fi] = (p[si] - src.rest_trs()[0][si]) * root_scale
    out_name = name or system._unique_clip_name(f"{handle.name} -> {tgt_obj.name}")
    new = Clip(P.OBJECT)
    for k, j in enumerate(tgt.joints):
        qs = t_rot[:, k].copy()
        for i in range(1, qs.shape[0]):
            qs[i] = R.align(qs[i - 1], qs[i])
        if np.allclose(qs, np.array(j.rotation)[None], atol=1e-12):
            continue
        new = new.with_channel(f'joints["{j.name}"].rotation', RotationTrack.from_arrays(times, qs))
    if root_name is not None:
        ti = tgt.index(m[root_name])
        rest = tgt.rest_trs()[0][ti]
        for axis, ci in (("x", 0), ("y", 1), ("z", 2)):
            vals = rest[ci] + root_pos[:, ci]
            if not np.allclose(vals, rest[ci]):
                new = new.with_channel(
                    f'joints["{tgt.joints[ti].name}"].position.{axis}',
                    FCurve.from_arrays(times, vals, handles="auto_clamped"),
                )
    new = new.with_events([e.to_json() for e in data.events])
    created = system.add_clip(out_name, new)
    return {
        "clip": created.name,
        "mapped": dict(m),
        "unmapped_source": sorted(j.name for j in src.joints if j.name not in m),
        "unmapped_target": sorted(j.name for j in tgt.joints if j.name not in m.values()),
        "rest_mismatch": mismatch,
        "root_scale": root_scale,
        "frames": int(frames.size),
    }
