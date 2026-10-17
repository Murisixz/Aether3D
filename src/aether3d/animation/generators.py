"""Editable motion generators: procedural motions written as ordinary keys.

A generator *computes* a motion and *writes keys*: the result is a normal clip (named
after the motion) whose curves an animator or an agent edits like any other -- move a
key, change a handle, scale the timing, layer a correction on top. The clip's metadata
records the generator and its parameters, so a motion can also be regenerated with
other parameters. Nothing is evaluated procedurally at playback.

**Rigs and roles.** Generators drive *roles* -- ``hips``, ``spine``, ``chest``, ``neck``,
``head``, ``upper_arm``/``lower_arm``/``hand`` and ``thigh``/``shin``/``foot`` with sides
``.L``/``.R`` -- found by joint name (``Hips``, ``Pelvis``, ``LeftUpLeg``, ``Leg.L``,
``mixamorig:LeftForeArm``, ``ArmR``, ...; ``mapping`` overrides) and by the hierarchy
(``Leg`` under ``UpLeg`` is a shin). Minimal rigs work: a blocky character with one joint
per limb (no knees, elbows or feet) walks with straight legs. The rest pose must stand
upright, Z up, facing ``forward`` (default ``+y``); arms may rest down, in an A-pose or a
T-pose (they are brought down first).

**How motions are made.** Each generator evaluates a parametric motion densely (240
samples per second): body offsets and anatomical rotations (flexion about the lateral
axis, rotation about the vertical, side bend about the forward axis), and for legs
**foot trajectories** -- planted during stance (heel strike and toe-off rock about the
heel and the toes), lifted along an arc during swing -- reached by analytic two-bone IK
(knees bend forward; one-joint legs aim at the foot point). Walking and running hips
height follows the stance legs' reach (an inverted pendulum, smoothed). The armature
travels by its own ``transform.position`` (root motion; ``travel=False`` walks in place)
at exactly the speed the planted feet move backward, so **feet do not slide** beyond the
fitting tolerance. Each joint's local rotation is then written as XYZ Euler curves (the
graph-editable form) -- or as a quaternion track (``cubic``) when its Euler angles would
come within 15 degrees of gimbal lock -- with keys chosen greedily until every sample is
within ``tolerance`` (degrees; positions within ``tolerance`` thousandths of the leg
length): few keys where the motion is simple, more where it turns. Each key's slope is
the motion's own, so curves are C1. Cycles (idle, walk, run) loop seamlessly
(``cycle`` extrapolation; root motion ``cycle_offset``).

Motions: ``idle`` (breathing, weight shift, glances), ``walk``, ``run``, ``jump``
(anticipation, ballistic flight under ``gravity``, landing absorption), ``land``
(``style="superhero"``: a fall into one knee and a fist down, a pause, a look up and a
rise; ``"soft"``: a plain landing), ``wave``, ``head_turn``; for cameras
:func:`follow_camera` (a critically damped follow of a target, fitted into keys, aimed
by a look-at constraint). Camera orbits are :meth:`AnimationSystem.camera_orbit`.
"""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import rotation as R
from .curves import FCurve
from .fitting import _reduce
from .ik import chain_rotations, rotation_between, two_bone
from .rig import forward_kinematics, local_matrices
from .tracks import RotationTrack

if TYPE_CHECKING:
    from .system import AnimationSystem

KINDS = ("idle", "walk", "run", "jump", "land", "wave", "head_turn")
ROLES = (
    "hips",
    "spine",
    "chest",
    "neck",
    "head",
    "upper_arm.L",
    "lower_arm.L",
    "hand.L",
    "upper_arm.R",
    "lower_arm.R",
    "hand.R",
    "thigh.L",
    "shin.L",
    "foot.L",
    "thigh.R",
    "shin.R",
    "foot.R",
)
RATE = 240.0  # samples per second
GIMBAL_MARGIN = 75.0  # degrees: Euler middle angles beyond this use quaternion tracks
MAX_DURATION = 120.0
_ID = np.array([0.0, 0.0, 0.0, 1.0])


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


# --------------------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------------------- #

_PREFIX = re.compile(r"^(mixamorig\d*[:_]|bip0?1[ _]|def[-_]|orig[-_]|org[-_])", re.I)
_BASES = {
    "hips": ("hips", "hip", "pelvis"),
    "spine": ("spine", "body", "torso", "abdomen", "waist", "chest", "upperchest"),
    "neck": ("neck",),
    "head": ("head",),
    "upper_arm": ("upperarm", "arm", "uparm"),
    "lower_arm": ("forearm", "lowerarm", "elbow", "loarm"),
    "hand": ("hand", "wrist"),
    "thigh": ("thigh", "upleg", "upperleg", "upleg"),
    "leg": ("leg",),
    "shin": ("shin", "calf", "lowerleg", "knee", "loleg"),
    "foot": ("foot", "ankle"),
}


def _side_base(name: str) -> tuple[str | None, str]:
    """``("L" | "R" | None, normalized base)`` of a joint name."""
    s = _PREFIX.sub("", name)
    side = None
    m = re.match(r"^(left|right)(.*)$", s, re.I)
    if m:
        side, s = ("L" if m.group(1).lower() == "left" else "R"), m.group(2)
    else:
        m = re.match(r"^([lr])[._\- ](.+)$", s, re.I)
        if m:
            side, s = m.group(1).upper(), m.group(2)
        else:
            m = re.match(r"^(.+?)[._\- ]?(left|right)$", s, re.I)
            if m:
                side, s = ("L" if m.group(2).lower() == "left" else "R"), m.group(1)
            else:
                m = re.match(r"^(.+?)[._\- ]([lr])$", s, re.I)
                if m:
                    side, s = m.group(2).upper(), m.group(1)
                else:
                    m = re.match(r"^(.*[a-z0-9])([LR])$", s)
                    if m:
                        side, s = m.group(2), m.group(1)
    base = re.sub(r"[^a-z0-9]", "", s.lower())
    return side, base


def _match(base: str, kind: str) -> bool:
    words = _BASES[kind]
    stripped = base.rstrip("0123456789")
    return base in words or stripped in words


def find_roles(rig: Any, mapping: dict[str, str] | None = None) -> dict[str, str]:
    """Role -> joint name for ``rig`` (see the module docstring); ``mapping`` (role ->
    joint name) takes precedence."""
    roles: dict[str, str] = {}
    joints = rig.joints
    depth = rig.depth
    info = [(k, j.name, *_side_base(j.name)) for k, j in enumerate(joints)]

    def descends(k: int, ancestor: int) -> bool:
        p = joints[k].parent
        while p >= 0:
            if p == ancestor:
                return True
            p = joints[p].parent
        return False

    centre = [(k, name, base) for k, name, side, base in info if side is None]
    hips = [x for x in centre if _match(x[2], "hips")]
    if hips:
        roles["hips"] = min(hips, key=lambda x: depth[x[0]])[1]
    else:
        roots = [k for k, j in enumerate(joints) if j.parent < 0]
        if len(roots) == 1:
            roles["hips"] = joints[roots[0]].name
    spine = sorted(
        (x for x in centre if _match(x[2], "spine") and x[1] != roles.get("hips")),
        key=lambda x: depth[x[0]],
    )
    if spine:
        roles["spine"] = spine[0][1]
        if len(spine) > 1:
            roles["chest"] = spine[-1][1]
    for role in ("neck", "head"):
        found = [x for x in centre if _match(x[2], role)]
        if found:
            roles[role] = min(found, key=lambda x: depth[x[0]])[1]
    for side in ("L", "R"):
        mine = [(k, name, base) for k, name, s, base in info if s == side]

        def first(kind: str, among=mine) -> tuple[int, str, str] | None:
            found = [x for x in among if _match(x[2], kind)]
            return min(found, key=lambda x: depth[x[0]]) if found else None

        arm = first("upper_arm")
        if arm:
            roles[f"upper_arm.{side}"] = arm[1]
            fore = [x for x in mine if _match(x[2], "lower_arm") and descends(x[0], arm[0])]
            if fore:
                roles[f"lower_arm.{side}"] = min(fore, key=lambda x: depth[x[0]])[1]
            hand = [x for x in mine if _match(x[2], "hand") and descends(x[0], arm[0])]
            if hand:
                roles[f"hand.{side}"] = min(hand, key=lambda x: depth[x[0]])[1]
        thigh = first("thigh")
        legs = sorted((x for x in mine if _match(x[2], "leg")), key=lambda x: depth[x[0]])
        if thigh is None and legs:
            thigh = legs.pop(0)
        if thigh:
            roles[f"thigh.{side}"] = thigh[1]
            shins = [
                x for x in mine if (_match(x[2], "shin") or x in legs) and descends(x[0], thigh[0])
            ]
            if shins:
                shin = min(shins, key=lambda x: depth[x[0]])
                roles[f"shin.{side}"] = shin[1]
            feet = [x for x in mine if _match(x[2], "foot") and descends(x[0], thigh[0])]
            if feet:
                roles[f"foot.{side}"] = min(feet, key=lambda x: depth[x[0]])[1]
    if mapping:
        if not isinstance(mapping, dict):
            raise _err("INVALID_ARGUMENT", "mapping is {role: joint name}", "mapping")
        for role, name in mapping.items():
            if role not in ROLES:
                raise _err("INVALID_ARGUMENT", f"unknown role {role!r}", f"mapping[{role}]",
                           hint=f"roles: {', '.join(ROLES)}")  # fmt: skip
            if name is None:
                roles.pop(role, None)
                continue
            if not isinstance(name, str) or rig.index(name) is None:
                raise _err("JOINT_MISSING", f"no joint {name!r}", f"mapping[{role}]")
            roles[role] = name
    return roles


# --------------------------------------------------------------------------------------- #
# The pose solver
# --------------------------------------------------------------------------------------- #

_AXES = {
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
}


# Single quaternions and vectors, in plain floats: the pose solver handles thousands of
# them per motion, where NumPy's per-call overhead would dominate (same formulas as
# :mod:`.rotation`).


def _axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    x, y, z = (float(c) for c in axis)
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    h = 0.5 * math.radians(degrees)
    s = math.sin(h) / n
    return np.array([x * s, y * s, z * s, math.cos(h)])


def _qmul(*qs: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = qs[0].tolist()
    for q in qs[1:]:
        bx, by, bz, bw = q.tolist()
        ax, ay, az, aw = (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    return np.array([ax, ay, az, aw])


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    ux, uy, uz, w = q.tolist()
    vx, vy, vz = (float(c) for c in v)
    tx = 2.0 * (uy * vz - uz * vy)
    ty = 2.0 * (uz * vx - ux * vz)
    tz = 2.0 * (ux * vy - uy * vx)
    return np.array(
        [
            vx + w * tx + (uy * tz - uz * ty),
            vy + w * ty + (uz * tx - ux * tz),
            vz + w * tz + (ux * ty - uy * tx),
        ]
    )


def _between(a: np.ndarray, b: np.ndarray, fraction: float = 1.0) -> np.ndarray:
    """The shortest rotation taking unit direction ``a`` to ``b`` (not opposite), scaled
    to ``fraction`` of its angle."""
    ax, ay, az = (float(c) for c in a)
    bx, by, bz = (float(c) for c in b)
    axis = np.array([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
    sin = math.sqrt(float(axis @ axis))
    if sin < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    angle = math.atan2(sin, ax * bx + ay * by + az * bz)
    return _axis_angle(axis, fraction * math.degrees(angle))


class _Body:
    """Rest geometry of a rig in anatomical terms, and a pose solver."""

    def __init__(self, rig: Any, roles: dict[str, str], forward: str) -> None:
        if forward not in _AXES:
            raise _err("INVALID_ARGUMENT", "forward is +y, -y, +x or -x", "forward")
        self.rig = rig
        self.k = {role: rig.index(name) for role, name in roles.items()}
        self.f = np.array(_AXES[forward])
        self.u = np.array([0.0, 0.0, 1.0])
        self.r = np.cross(self.f, self.u)  # the character's right
        pos, rot, scl = rig.rest_trs()
        self.pos0, self.rot0, self.scl0 = np.array(pos), np.array(rot), np.array(scl)
        self.model0 = np.array(rig.rest_model())
        lin = self.model0[:, :3, :3]
        self.qm0 = R.normalize(R.from_matrices(lin / np.linalg.norm(lin, axis=1, keepdims=True)))
        self.parents = rig.parents
        self.depth = rig.depth
        self.tails = rig.tails()
        if "hips" not in self.k:
            raise _err("ROLE_MISSING", "the rig has no hips (root) joint", "roles")
        self.hips = self.k["hips"]
        # Legs.
        self.legs: dict[str, dict[str, Any]] = {}
        for side in ("L", "R"):
            th = self.k.get(f"thigh.{side}")
            if th is None:
                continue
            sh = self.k.get(f"shin.{side}")
            ft = self.k.get(f"foot.{side}") if sh is not None else None
            hip = self.model0[th, :3, 3]
            if sh is not None:
                ankle = self.model0[ft, :3, 3] if ft is not None else self.end(self.model0, sh)
                length = float(
                    np.linalg.norm(self.model0[sh, :3, 3] - hip)
                    + np.linalg.norm(ankle - self.model0[sh, :3, 3])
                )
            else:
                ankle = self.end(self.model0, th)
                length = float(np.linalg.norm(ankle - hip))
            leg = {"thigh": th, "shin": sh, "foot": ft, "hip": hip, "ankle": ankle.copy(),
                   "length": length}  # fmt: skip
            if ft is not None:
                toe = self.end(self.model0, ft)
                foot_len = float(np.linalg.norm((toe - ankle) - ((toe - ankle) @ self.u) * self.u))
                foot_len = foot_len if foot_len > 1e-6 else 0.25 * length
                ground = (
                    float(toe[2]) if toe[2] < ankle[2] - 1e-6 else float(ankle[2]) - 0.08 * length
                )
                toe_pt = toe.copy()  # the bone's end, on the ground (as diagnostics see it)
                toe_pt[2] = ground
                heel = ankle.copy()  # below the ankle, on the ground (as diagnostics see it)
                heel[2] = ground
                leg.update(foot_len=foot_len, ground=ground, toe=toe_pt, heel=heel,
                           toe_rel=toe - ankle)  # fmt: skip
            else:
                leg.update(foot_len=0.0, ground=float(ankle[2]), toe=None, heel=None)
            self.legs[side] = leg
        self.leg_length = (
            float(np.mean([leg["length"] for leg in self.legs.values()]))
            if self.legs
            else float(self.model0[self.hips, 2, 3]) or 1.0
        )
        if not self.leg_length > 1e-6:
            raise _err("INVALID_RIG", "the rig's legs have no length", "rig")
        # Arms: the rotation bringing each upper arm from rest to hanging (slightly out).
        self.arms: dict[str, dict[str, Any]] = {}
        for side in ("L", "R"):
            ua = self.k.get(f"upper_arm.{side}")
            if ua is None:
                continue
            la = self.k.get(f"lower_arm.{side}")
            sh = self.model0[ua, :3, 3]
            nxt = self.model0[la, :3, 3] if la is not None else self.end(self.model0, ua)
            d = nxt - sh
            out = -self.r if side == "L" else self.r
            hang = -self.u * math.cos(math.radians(4.0)) + out * math.sin(math.radians(4.0))
            base = rotation_between(d, hang) if np.linalg.norm(d) > 1e-9 else _ID.copy()
            reach = float(np.linalg.norm(d))
            if la is not None:
                hd = self.k.get(f"hand.{side}")
                tip = self.model0[hd, :3, 3] if hd is not None else self.end(self.model0, la)
                reach += float(np.linalg.norm(tip - nxt))
            self.arms[side] = {"upper": ua, "lower": la, "hand": self.k.get(f"hand.{side}"),
                               "base": base, "length": float(np.linalg.norm(d)),
                               "reach": reach}  # fmt: skip

    def end(self, model: np.ndarray, k: int) -> np.ndarray:
        """Where joint ``k``'s bone ends (its tail) under ``model``."""
        return (model[k] @ np.array([*self.tails[k], 1.0]))[:3]

    def side_out(self, side: str) -> np.ndarray:
        return -self.r if side == "L" else self.r

    def solve(
        self,
        offset: np.ndarray,
        deltas: dict[int, np.ndarray],
        legs: dict[str, dict[str, Any]] | None = None,
        arms: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Local joint rotations, the hips' local position and the largest IK residual."""
        rot = self.rot0.copy()
        for k, d in deltas.items():
            rot[k] = _qmul(self.rot0[k], _conj(self.qm0[k]), d, self.qm0[k])
        pos = self.pos0.copy()
        h = self.hips
        ph = int(self.parents[h])
        q_parent = self.qm0[ph] if ph >= 0 else _ID
        pos[h] = self.pos0[h] + _rot(_conj(q_parent), offset)
        model = forward_kinematics(local_matrices(pos, rot, self.scl0), self.parents, self.depth)
        lin = model[:, :3, :3]
        qm = R.normalize(R.from_matrices(lin / np.linalg.norm(lin, axis=1, keepdims=True)))
        residual = 0.0

        def parent_q(k: int) -> np.ndarray:
            p = int(self.parents[k])
            return qm[p] if p >= 0 else _ID

        for side, spec in (legs or {}).items():
            leg = self.legs.get(side)
            if leg is None or spec is None:
                continue
            th, sh, ft = leg["thigh"], leg["shin"], leg["foot"]
            target = np.asarray(spec["target"], np.float64)
            hip = model[th, :3, 3]
            if sh is not None:
                knee = model[sh, :3, 3]
                ankle = model[ft, :3, 3] if ft is not None else self.end(model, sh)
                pole = knee + self.f * self.leg_length
                new_knee, new_ankle, res = two_bone(hip, knee, ankle, target, pole)
                q1, q2 = chain_rotations(
                    np.array([hip, knee, ankle]), np.array([hip, new_knee, new_ankle])
                )
                qt = _qmul(q1, qm[th])
                qs = _qmul(q2, q1, qm[sh])
                rot[th] = _qmul(_conj(parent_q(th)), qt)
                rot[sh] = _qmul(_conj(qt), qs)
                if ft is not None:
                    want = _qmul(_axis_angle(self.r, spec.get("pitch", 0.0)), self.qm0[ft])
                    rot[ft] = _qmul(_conj(qs), want)
            else:
                end = self.end(model, th)
                length = leg["length"]
                splay = spec.get("splay")
                if splay:  # slide the target along the ground until the leg reaches it
                    drop = float((hip - target) @ self.u)
                    reach = math.sqrt(max(length**2 - drop * drop, 0.0))
                    flat = target - hip - ((target - hip) @ self.u) * self.u
                    side_off = (flat @ self.r) * self.r
                    fwd = math.sqrt(max(reach * reach - float(side_off @ side_off), 0.0))
                    target = hip - drop * self.u + side_off + math.copysign(fwd, splay) * self.f
                aim = target - hip
                n = float(np.linalg.norm(aim))
                if n > 1e-12:
                    # A straight leg never goes through the ground: tilt it (keeping its
                    # heading) until its end is on the ground.
                    above = float(hip @ self.u) - leg["ground"]
                    if float(aim @ self.u) / n * length < -above:
                        flat = aim - (aim @ self.u) * self.u
                        nf = float(np.linalg.norm(flat))
                        heading = flat / nf if nf > 1e-12 else self.f
                        c = min(max(above / length, 0.0), 1.0)
                        aim = heading * math.sqrt(1.0 - c * c) - self.u * c
                q = rotation_between(end - hip, aim)
                rot[th] = _qmul(_conj(parent_q(th)), q, qm[th])
                res = abs(float(np.linalg.norm(target - hip)) - float(np.linalg.norm(end - hip)))
            if spec.get("planted", True):
                residual = max(residual, res)
        for side, spec in (arms or {}).items():
            arm = self.arms.get(side)
            if arm is None or spec is None:
                continue
            ua, la = arm["upper"], arm["lower"]
            sh_pt = model[ua, :3, 3]
            if "target" in spec:
                target = np.asarray(spec["target"], np.float64)
            else:  # a point on the ground relative to the posed shoulder, blended in
                u = self.u
                natural = sh_pt - u * arm["reach"]
                ground = sh_pt - ((sh_pt @ u) - spec["ground"]) * u
                lean = spec.get("ahead", 0.0) * self.f + spec.get("out", 0.0) * self.side_out(side)
                if la is None:  # a one-joint arm: the ground point at exactly its length
                    drop = float(sh_pt @ u) - spec["ground"]
                    flat = float(np.linalg.norm(lean))
                    if flat > 1e-12 and drop < arm["reach"]:
                        lean = lean / flat * math.sqrt(arm["reach"] ** 2 - drop * drop)
                ground = ground + lean
                blend = spec.get("blend", 1.0)
                target = natural + (ground - natural) * blend
                if blend < 0.999:  # on the way: the hand goes as far as the arm reaches
                    d = target - sh_pt
                    n = float(np.linalg.norm(d))
                    if n > 0.999 * arm["reach"]:
                        target = sh_pt + d * (0.999 * arm["reach"] / n)
            if la is not None:
                el = model[la, :3, 3]
                hd = model[arm["hand"], :3, 3] if arm["hand"] is not None else self.end(model, la)
                pole = el + np.asarray(spec.get("pole", -self.f), np.float64) * arm["length"]
                new_el, new_hd, res = two_bone(sh_pt, el, hd, target, pole)
                q1, q2 = chain_rotations(
                    np.array([sh_pt, el, hd]), np.array([sh_pt, new_el, new_hd])
                )
                qu = _qmul(q1, qm[ua])
                ql = _qmul(q2, q1, qm[la])
                rot[ua] = _qmul(_conj(parent_q(ua)), qu)
                rot[la] = _qmul(_conj(qu), ql)
            else:
                end = self.end(model, ua)
                q = rotation_between(end - sh_pt, target - sh_pt)
                rot[ua] = _qmul(_conj(parent_q(ua)), q, qm[ua])
                res = abs(
                    float(np.linalg.norm(target - sh_pt)) - float(np.linalg.norm(end - sh_pt))
                )
            if spec.get("planted", True):
                residual = max(residual, res)
        # Feet never go through the ground: a toe that would is lifted by easing its pitch.
        for side, spec in (legs or {}).items():
            leg = self.legs.get(side)
            if leg is None or spec is None or leg["foot"] is None or leg["shin"] is None:
                continue
            pitch = spec.get("pitch", 0.0)
            if pitch >= 0.0:
                continue
            ft, sh = leg["foot"], leg["shin"]
            ankle_z = self._ankle_z(rot, pos, side)

            def toe_z(
                angle: float, ankle_z: float = ankle_z, rel: np.ndarray = leg["toe_rel"]
            ) -> float:
                return ankle_z + float(_rot(_axis_angle(self.r, angle), rel) @ self.u)

            if toe_z(pitch) >= leg["ground"]:
                continue
            lo, hi = pitch, 0.0  # toe_z(hi) >= ground > toe_z(lo)
            if toe_z(hi) < leg["ground"]:
                continue  # (the ankle itself is too low: nothing a pitch can do)
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                if toe_z(mid) < leg["ground"]:
                    lo = mid
                else:
                    hi = mid
            want = _qmul(_axis_angle(self.r, hi), self.qm0[ft])
            shin_q = self._model_q(rot, pos, sh)
            rot[ft] = _qmul(_conj(shin_q), want)
        return rot, pos[h], residual

    def _model_q(self, rot: np.ndarray, pos: np.ndarray, k: int) -> np.ndarray:
        model = forward_kinematics(local_matrices(pos, rot, self.scl0), self.parents, self.depth)
        lin = model[k, :3, :3]
        return R.normalize(
            R.from_matrices((lin / np.linalg.norm(lin, axis=0, keepdims=True))[None])
        )[0]

    def _ankle_z(self, rot: np.ndarray, pos: np.ndarray, side: str) -> float:
        model = forward_kinematics(local_matrices(pos, rot, self.scl0), self.parents, self.depth)
        return float(model[self.legs[side]["foot"], 2, 3])

    def model_at(self, offset: np.ndarray, deltas: dict[int, np.ndarray]) -> np.ndarray:
        """Joint matrices with the body deltas only (legs and arms at rest)."""
        rot = self.rot0.copy()
        for k, d in deltas.items():
            rot[k] = _qmul(self.rot0[k], _conj(self.qm0[k]), d, self.qm0[k])
        pos = self.pos0.copy()
        h = self.hips
        ph = int(self.parents[h])
        q_parent = self.qm0[ph] if ph >= 0 else _ID
        pos[h] = self.pos0[h] + _rot(_conj(q_parent), offset)
        return forward_kinematics(local_matrices(pos, rot, self.scl0), self.parents, self.depth)


# --------------------------------------------------------------------------------------- #
# Smooth helpers
# --------------------------------------------------------------------------------------- #


def _smooth(u: float) -> float:
    """Quintic smoothstep on [0, 1] (zero velocity and acceleration at the ends)."""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _ramp(t: float, a: float, b: float) -> float:
    return _smooth((t - a) / (b - a)) if b > a else float(t >= a)


def _keys(t: float, pairs: list[tuple[float, float]]) -> float:
    """Piecewise quintic-smooth interpolation through ``(time, value)`` pairs (flat at
    each pair: poses held for an instant, as an animator's key poses)."""
    if t <= pairs[0][0]:
        return pairs[0][1]
    for (a, va), (b, vb) in zip(pairs[:-1], pairs[1:], strict=True):
        if t <= b:
            return va + (vb - va) * _ramp(t, a, b)
    return pairs[-1][1]


def _softmin(a: float, b: float, k: float) -> float:
    """A smooth minimum (``k``: the blending width)."""
    h = min(max(0.5 + 0.5 * (b - a) / k, 0.0), 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def _cyc(phi: float) -> float:
    return phi - math.floor(phi)


# --------------------------------------------------------------------------------------- #
# Motions: each returns (duration, cyclic, frame(t) -> pose spec)
# --------------------------------------------------------------------------------------- #


def _locomotion(body: _Body, p: dict[str, Any], run: bool):
    L = body.leg_length
    f, u, r = body.f, body.u, body.r
    cadence = float(p.get("cadence", 2.8 if run else 1.8))  # steps per second
    step = float(p.get("stride", 1.5 if run else 0.65)) * L  # metres per step
    straight = all(leg["shin"] is None for leg in body.legs.values())  # no knees
    # Straight legs walk a compass gait: one foot down at a time (no double support).
    duty = float(p.get("duty", 0.38 if run else (0.5 if straight else 0.62)))
    lift = float(p.get("lift", 0.32 if run else 0.13)) * L
    amp = float(p.get("amplitude", 1.0))
    if not (0.2 <= cadence <= 6.0 and 0.0 < step <= 3.0 * L and 0.2 <= duty <= 0.8):
        raise _err(
            "INVALID_ARGUMENT",
            "cadence 0.2..6 steps/s, stride 0..3 leg lengths, duty 0.2..0.8",
            "parameters",
        )
    period = 2.0 / cadence
    speed = step * cadence
    travel = 2.0 * step  # per cycle
    stance = travel * duty  # how far a planted foot moves back (armature space) in stance
    ahead = stance * (0.45 if run else 0.5)  # contact point ahead of the hip
    reach = 0.975 if run else 0.985
    a_to = 22.0 if run else 18.0  # toe-off pitch
    a_hs = 8.0 if run else 12.0  # heel-strike pitch
    heel_frac, toe_frac = (0.0 if run else 0.12), 0.25

    hips0 = body.model0[body.hips, :3, 3]

    def foot_state(side: str, phase: float) -> tuple[np.ndarray, float, bool]:
        """The ankle target (armature space), the foot pitch and whether it is planted."""
        leg = body.legs[side]
        rest = leg["ankle"]
        has_foot = leg["toe"] is not None
        lateral = (rest - hips0) @ r

        def at(x: float, z: float) -> np.ndarray:  # x ahead of the hips' rest position
            return (
                hips0
                - (hips0 @ u) * u
                - (hips0 @ f) * f
                + (hips0 @ f + x) * f
                + lateral * r
                + z * u
            )

        def rocked(x: float, pitch: float, pivot: str) -> np.ndarray:
            a = at(x, rest @ u)
            if not has_foot or pitch == 0.0:
                return a
            piv_rest = leg["toe"] if pivot == "toe" else leg["heel"]
            return a + (piv_rest - rest) + _rot(_axis_angle(r, pitch), rest - piv_rest)

        x_c, x_to = ahead, ahead - stance
        if phase < duty:  # stance: planted (moving back at the ground's speed)
            x = x_c - stance * phase / duty
            if has_foot and heel_frac > 0 and phase < heel_frac * duty:
                pitch = a_hs * (1.0 - _smooth(phase / (heel_frac * duty)))
                return rocked(x, pitch, "heel"), pitch, True
            if has_foot and phase > (1.0 - toe_frac) * duty:
                pitch = -a_to * _smooth((phase - (1.0 - toe_frac) * duty) / (toe_frac * duty))
                return rocked(x, pitch, "toe"), pitch, True
            return at(x, rest @ u), 0.0, True
        # Swing: Hermite in x matching the ground's speed at both ends, a lift arc in z.
        w = (phase - duty) / (1.0 - duty)
        strike = a_hs if (has_foot and heel_frac > 0) else 0.0
        p0 = rocked(x_to, -a_to if has_foot else 0.0, "toe")
        p1 = rocked(x_c, strike, "heel")
        x0, x1 = (p0 - hips0) @ f, (p1 - hips0) @ f
        m = -stance / duty * (1.0 - duty)  # dx/dw of the ground
        h00, h10 = 2 * w**3 - 3 * w**2 + 1, w**3 - 2 * w**2 + w
        h01, h11 = -2 * w**3 + 3 * w**2, w**3 - w**2
        x = h00 * x0 + h10 * m + h01 * x1 + h11 * m
        z = (1 - _smooth(w)) * (p0 @ u) + _smooth(w) * (p1 @ u) + lift * 16 * w * w * (1 - w) ** 2
        pitch = (-a_to + (a_to + strike) * _smooth(w)) if has_foot else 0.0
        return at(x, z), pitch, False

    def pendulum(side: str, phase: float) -> float:
        """The hips' offset that puts a straight leg's end on its foot point."""
        leg = body.legs[side]
        pt = foot_state(side, phase)[0]
        d = pt - leg["hip"]
        horiz = math.hypot(d @ f, d @ r)
        return math.sqrt(max(leg["length"] ** 2 - horiz * horiz, 0.0)) - (leg["hip"] - pt) @ u

    def straight_height(phi: float) -> float:
        phases = {
            side: _cyc(phi - shift) for side, shift in (("L", 0.0), ("R", 0.5)) if side in body.legs
        }
        planted = [side for side, ph in phases.items() if ph < duty]
        if planted:
            return min(pendulum(side, phases[side]) for side in planted)
        if len(phases) < 2:
            return 0.0
        # Flight: a cubic between the take-off and the landing (heights and slopes).
        up = max(phases, key=lambda side: -phases[side] if phases[side] >= duty else -9.0)
        down = "R" if up == "L" else "L"
        a = phi - (phases[up] - duty)
        b = phi + (1.0 - phases[down])
        e = 1e-5
        h0 = pendulum(up, duty - e)
        d0 = (h0 - pendulum(up, duty - 2 * e)) / e
        h1 = pendulum(down, 0.0)
        d1 = (pendulum(down, e) - h1) / e
        span = b - a
        w = (phi - a) / span
        return ((2 * w**3 - 3 * w**2 + 1) * h0 + (w**3 - 2 * w**2 + w) * span * d0
                + (-2 * w**3 + 3 * w**2) * h1 + (w**3 - w**2) * span * d1)  # fmt: skip

    def frame(t: float) -> dict[str, Any]:
        phi = _cyc(t / period)
        s2 = math.sin(2 * math.pi * phi)
        c2 = math.cos(2 * math.pi * phi)
        legs, contact = {}, {}
        for side, shift in (("L", 0.0), ("R", 0.5)):
            if side in body.legs:
                pt, pitch, planted = foot_state(side, _cyc(phi - shift))
                legs[side] = {"target": pt, "pitch": pitch, "planted": planted}
                contact[side] = (pt, planted)
        # Hips height: the intended bob, limited by the stance legs' reach (smoothly).
        if run:
            bob = -0.06 * L - 0.05 * L * amp * (1 + math.cos(4 * math.pi * (phi - duty / 2))) / 2
        else:
            bob = -0.015 * L
        # Every foot limits the hips (a lifted one loosely): continuous through contacts.
        limit = math.inf
        for side, (pt, _planted) in contact.items():
            leg = body.legs[side]
            hip = leg["hip"]
            d = pt - hip
            horiz = math.hypot(d @ f, d @ r) + 0.012 * L  # (pelvis turn and list)
            rl = reach * leg["length"]
            limit = min(limit, math.sqrt(max(rl * rl - horiz * horiz, 0.0)) - (hip - pt) @ u)
        dz = bob if limit == math.inf else _softmin(bob, limit, 0.04 * L)
        # (Straight legs: the exact pendulum height, and a pelvis that neither sways nor
        # turns -- either would move the hips off the legs' reach.)
        pelvis = 0.0 if straight else amp
        if straight:
            dz = straight_height(phi)
        sway = (0.012 if run else 0.025) * L * pelvis
        offset = dz * u - sway * s2 * r
        yaw_h = -(8.0 if run else 5.0) * pelvis * c2
        list_h = (3.0 if run else 4.0) * pelvis * s2
        lean = -(10.0 if run else 3.0)
        deltas: dict[int, np.ndarray] = {}
        k = body.k
        deltas[body.hips] = _qmul(_axis_angle(u, yaw_h), _axis_angle(f, list_h))
        if "spine" in k:
            deltas[k["spine"]] = _qmul(
                _axis_angle(u, -yaw_h * (1.0 if "chest" in k else 1.6)),
                _axis_angle(r, lean),
                _axis_angle(f, -0.6 * list_h),
            )
        if "chest" in k:
            deltas[k["chest"]] = _qmul(_axis_angle(u, -yaw_h * 0.8), _axis_angle(r, lean * 0.3))
        head_yaw = -yaw_h * 0.15
        head_pitch = -lean * 0.8 + (2.0 if run else 1.0) * math.cos(4 * math.pi * phi)
        if "neck" in k:
            deltas[k["neck"]] = _qmul(
                _axis_angle(u, head_yaw * 0.4), _axis_angle(r, head_pitch * 0.4)
            )
        if "head" in k:
            share = 0.6 if "neck" in k else 1.0
            deltas[k["head"]] = _qmul(
                _axis_angle(u, head_yaw * share), _axis_angle(r, head_pitch * share)
            )
        arm_amp = (32.0 if run else 16.0) * amp
        for side, sgn in (("L", -1.0), ("R", 1.0)):
            arm = body.arms.get(side)
            if arm is None:
                continue
            swing = sgn * arm_amp * c2 + (12.0 if run else 2.0)
            abduct = (10.0 if run else 6.0) * (1.0 if side == "L" else -1.0)
            deltas[arm["upper"]] = _qmul(_axis_angle(r, swing), _axis_angle(f, abduct), arm["base"])
            if arm["lower"] is not None:
                fwd = (1.0 + sgn * c2) / 2.0  # 1 when this arm is forward
                bend = (78.0 + 18.0 * fwd) if run else (10.0 + 14.0 * fwd)
                deltas[arm["lower"]] = _axis_angle(r, bend)
        return {"offset": offset, "deltas": deltas, "legs": legs}

    info = {"period": period, "speed": speed, "step": step, "cadence": cadence}
    return period, True, frame, info, travel


def _idle(body: _Body, p: dict[str, Any]):
    L = body.leg_length
    f, u, r = body.f, body.u, body.r
    amp = float(p.get("amplitude", 1.0))
    period = float(p.get("period", 8.0))
    if not 2.0 <= period <= 60.0:
        raise _err("INVALID_ARGUMENT", "period is 2..60 s", "period")
    breaths = max(1, round(period / 4.0))
    straight = all(leg["shin"] is None for leg in body.legs.values())
    pelvis = 0.0 if straight else amp  # (straight legs cannot follow a shifting pelvis)

    def frame(t: float) -> dict[str, Any]:
        b = math.sin(2 * math.pi * breaths * t / period)
        w = math.sin(2 * math.pi * t / period)
        g = math.sin(2 * math.pi * t / period * 2 + 0.7)
        k = body.k
        deltas: dict[int, np.ndarray] = {}
        deltas[body.hips] = _axis_angle(f, 2.0 * pelvis * w)
        if "spine" in k:
            deltas[k["spine"]] = _qmul(
                _axis_angle(r, 0.8 * amp * b), _axis_angle(f, -1.6 * amp * w)
            )
        if "chest" in k:
            deltas[k["chest"]] = _axis_angle(r, 1.2 * amp * b)
        if "head" in k:
            deltas[k["head"]] = _qmul(_axis_angle(u, 9.0 * amp * g), _axis_angle(r, -1.5 * amp * b))
        for side in ("L", "R"):
            arm = body.arms.get(side)
            if arm is None:
                continue
            out = 1.0 if side == "L" else -1.0
            deltas[arm["upper"]] = _qmul(_axis_angle(f, out * (3.0 + 1.0 * amp * b)), arm["base"])
            if arm["lower"] is not None:
                deltas[arm["lower"]] = _axis_angle(r, 8.0 + 1.5 * amp * b)
        legs = {}
        for side, leg in body.legs.items():
            legs[side] = {"target": leg["ankle"].copy(), "pitch": 0.0}
        sink = 0.0 if straight else 0.02 * L
        offset = -sink * u - 0.02 * L * pelvis * w * r
        return {"offset": offset, "deltas": deltas, "legs": legs}

    return period, True, frame, {"period": period}, 0.0


def _jump(body: _Body, p: dict[str, Any]):
    L = body.leg_length
    f, u, r = body.f, body.u, body.r
    g = float(p.get("gravity", 9.81))
    height = float(p.get("height", 0.45)) * L
    distance = float(p.get("distance", 0.0)) * L
    if not (0.1 <= g <= 100.0 and 0.0 < height <= 3.0 * L and -2 * L <= distance <= 4 * L):
        raise _err(
            "INVALID_ARGUMENT",
            "gravity 0.1..100, height 0..3 and distance -2..4 leg lengths",
            "parameters",
        )
    t_crouch, t_off = 0.42, 0.66
    v0 = math.sqrt(2.0 * g * height)
    flight = 2.0 * v0 / g
    t_land = t_off + flight
    t_absorb, t_end = t_land + 0.16, t_land + 0.75
    # Straight legs cannot crouch without their feet sliding: they jump from standing.
    crouch = -0.24 * L if all(leg["shin"] is not None for leg in body.legs.values()) else 0.0

    def frame(t: float) -> dict[str, Any]:
        k = body.k
        # Hips: crouch, extend, ballistic flight, absorb, recover.
        if t <= t_off:
            dz = _keys(t, [(0.0, 0.0), (t_crouch, crouch), (t_off, 0.0)])
            dx = 0.0
        elif t <= t_land:
            tau = t - t_off
            dz = v0 * tau - 0.5 * g * tau * tau
            dx = distance * tau / flight
        else:
            dz = _keys(t, [(t_land, 0.0), (t_absorb, crouch * 0.8), (t_end, 0.0)])
            dx = distance
        air = t_off < t < t_land
        tuck = math.sin(math.pi * (t - t_off) / flight) if air else 0.0
        lean = _keys(t, [(0.0, 0.0), (t_crouch, -24.0), (t_off, -6.0), (t_land, -4.0),
                         (t_absorb, -20.0), (t_end, 0.0)])  # fmt: skip
        arms = _keys(t, [(0.0, 0.0), (t_crouch, -50.0), (t_off + 0.08, 150.0),
                         (t_land - 0.1, 60.0), (t_absorb, 20.0), (t_end, 0.0)])  # fmt: skip
        deltas: dict[int, np.ndarray] = {body.hips: _axis_angle(r, lean * 0.4)}
        if "spine" in k:
            deltas[k["spine"]] = _axis_angle(r, lean * 0.6)
        if "head" in k:
            deltas[k["head"]] = _axis_angle(r, -lean * 0.7)
        for side in ("L", "R"):
            arm = body.arms.get(side)
            if arm is None:
                continue
            out = 1.0 if side == "L" else -1.0
            deltas[arm["upper"]] = _qmul(
                _axis_angle(r, arms), _axis_angle(f, out * 8.0), arm["base"]
            )
            if arm["lower"] is not None:
                deltas[arm["lower"]] = _axis_angle(r, 15.0 + 25.0 * tuck)
        legs = {}
        for side, leg in body.legs.items():
            home = leg["ankle"].copy()
            if t <= t_off:
                pitch = -18.0 * _ramp(t, t_crouch, t_off) if leg["toe"] is not None else 0.0
                target = home
                if pitch:
                    target = (
                        home + leg["toe"] - home + _rot(_axis_angle(r, pitch), home - leg["toe"])
                    )
                legs[side] = {"target": target, "pitch": pitch}
                if leg["shin"] is None:  # a straight leg splays to stay on the ground
                    legs[side]["splay"] = 1.0 if side == "L" else -1.0
            elif air:
                # Feet follow the hips, tucked up and back in the middle of the flight.
                rel = home + (dz + 0.0) * u + dx * f + tuck * (0.28 * L * u - 0.12 * L * f)
                legs[side] = {
                    "target": rel,
                    "pitch": -25.0 * tuck if leg["toe"] is not None else 0.0,
                    "planted": False,
                }
            else:
                legs[side] = {"target": home + distance * f, "pitch": 0.0}
                if leg["shin"] is None:
                    legs[side]["splay"] = 1.0 if side == "L" else -1.0
        return {"offset": dz * u + dx * f, "deltas": deltas, "legs": legs}

    info = {"duration": t_end, "takeoff": t_off, "landing": t_land, "apex": t_off + flight / 2,
            "speed": v0}  # fmt: skip
    return t_end, False, frame, info, 0.0


def _land(body: _Body, p: dict[str, Any]):
    L = body.leg_length
    f, u, r = body.f, body.u, body.r
    g = float(p.get("gravity", 9.81))
    fall = float(p.get("height", 2.5))
    style = p.get("style", "superhero")
    if style not in ("superhero", "soft"):
        raise _err("INVALID_ARGUMENT", "style is 'superhero' or 'soft'", "style")
    if not (0.1 <= g <= 100.0 and 0.0 <= fall <= 50.0):
        raise _err("INVALID_ARGUMENT", "gravity 0.1..100, height 0..50", "parameters")
    side_k = p.get("knee", "R")  # the knee that goes down (and the fist on that side)
    if side_k not in ("L", "R"):
        raise _err("INVALID_ARGUMENT", "knee is 'L' or 'R'", "knee")
    side_f = "L" if side_k == "R" else "R"
    if side_k not in body.legs or side_f not in body.legs:
        raise _err("ROLE_MISSING", "a landing needs two legs", "roles")
    hero = style == "superhero"
    jointed = all(leg["shin"] is not None for leg in body.legs.values())
    hips0 = body.model0[body.hips, :3, 3]
    ground = min(leg["ground"] for leg in body.legs.values())
    front, back = body.legs[side_f], body.legs[side_k]
    lat = {side: (leg["ankle"] - hips0) @ r for side, leg in body.legs.items()}

    def spot(side: str, x: float, z: float) -> np.ndarray:
        """A foot point ``x`` ahead of the hips' rest position, at height ``z``."""
        return (hips0 @ f + x) * f + (hips0 @ r + lat[side]) * r + z * u

    # The landing pose: where the feet touch down and how low the hips settle.
    if hero and jointed:
        low = -0.71 * L
        x_f = 0.42 * L
        thigh = float(np.linalg.norm(body.model0[back["shin"], :3, 3] - back["hip"]))
        shin = back["length"] - thigh
        h_s = max(back["hip"][2] + low - (back["ground"] + 0.03 * L), 0.05 * L)
        knee_back = math.sqrt(max(thigh * thigh - h_s * h_s, 0.0))
        rise = min(0.7 * max(back["foot_len"], 0.05 * L), 0.9 * shin)
        x_b = (
            ((back["hip"] - hips0) @ f) - knee_back - math.sqrt(max(shin * shin - rise * rise, 0.0))
        )
        z_b = back["ground"] + 0.03 * L + rise
    elif hero:  # straight legs splay wide: the hips sit at a third of the leg's length
        h_s = 0.34 * L
        low = h_s - (front["hip"][2] - front["ground"])
        x_f = math.sqrt(L * L - h_s * h_s)
        x_b, z_b = -x_f, back["ground"]
    else:
        low = -0.22 * L if jointed else 0.0
        x_f, x_b, z_b = 0.1 * L, -0.1 * L, back["ankle"][2]
        if not jointed:
            x_f, x_b = 0.0, 0.0
    z_f = front["ankle"][2] if front["shin"] is not None else front["ground"]
    land_f, land_b = spot(side_f, x_f, z_f), spot(side_k, x_b, z_b)

    def reach_dz(pt: np.ndarray, leg: dict[str, Any], x_h: float = 0.0) -> float:
        hip = leg["hip"] + x_h * f
        d = pt - hip
        horiz = math.hypot(d @ f, d @ r)
        rl = (0.98 if leg["shin"] is not None else 1.0) * leg["length"]
        return math.sqrt(max(rl * rl - horiz * horiz, 0.0)) - (hip - pt) @ u

    impact = min(reach_dz(land_f, front), reach_dz(land_b, back), 0.0) if (hero or jointed) else 0.0
    impact = max(impact, low)
    t_hit = math.sqrt(2.0 * max(fall - impact, 0.0) / g)
    t_spread = max(0.0, t_hit - 0.35)
    t_low, t_set = t_hit + 0.12, t_hit + 0.34
    t_look, t_hold = t_hit + 1.15, t_hit + 1.7
    t_step0, t_step1, t_end = t_hold + 0.25, t_hold + 0.85, t_hold + 1.1
    if not jointed:  # a straight back leg cannot stay down while the hips rise: it steps now
        t_step0, t_step1 = t_hold, t_hold + 0.75
    # The lean: the least (from 35 degrees) that lets the fist reach the ground.
    fist = body.arms.get(side_k)
    wrist_h = 0.04 * L
    if fist is not None and fist["hand"] is not None:
        wrist_h = float(np.linalg.norm(body.tails[fist["hand"]]))
    lean_max = 18.0
    if hero:
        lean_max = 70.0
        if fist is not None:
            for lean in range(35, 71):
                deltas = {body.hips: _axis_angle(r, -lean * 0.35)}
                if "spine" in body.k:
                    deltas[body.k["spine"]] = _axis_angle(r, -lean * 0.65)
                m = body.model_at(low * u, deltas)
                gap = m[fist["upper"], :3, 3] @ u - (ground + wrist_h) - 0.96 * fist["reach"]
                if gap <= 0.0:
                    lean_max = float(lean)
                    break

    def hips_at(t: float) -> tuple[float, float]:
        """(dz, x) of the hips."""
        if t <= t_hit:
            return fall - 0.5 * g * t * t, 0.0
        if t <= t_hold:
            if not jointed:
                return impact, 0.0
            return _keys(t, [(t_hit, impact), (t_low, low - 0.03 * L), (t_set, low)]), 0.0
        # Rise over the front foot (the hips end above it, standing).
        w = _ramp(t, t_hold, t_end)
        x = x_f * w
        if not jointed:  # straight legs: exactly the height the front leg gives
            return reach_dz(land_f, front, x), x
        return low + (0.0 - low) * w, x

    def frame(t: float) -> dict[str, Any]:
        k = body.k
        dz, xh = hips_at(t)
        air = t <= t_hit
        settle = 0.0 if air else _keys(t, [(t_hit, 0.0), (t_low, 1.0), (t_hold, 1.0), (t_end, 0.0)])
        lean = -8.0 if air else -lean_max * settle
        look = 6.0 if air else _keys(t, [(t_hit, 0.0), (t_low, -18.0), (t_set, -22.0),
                                         (t_look - 0.3, -22.0), (t_look, 30.0), (t_hold, 30.0),
                                         (t_end, 0.0)])  # fmt: skip
        deltas: dict[int, np.ndarray] = {body.hips: _axis_angle(r, lean * 0.35)}
        if "spine" in k:
            deltas[k["spine"]] = _axis_angle(r, lean * 0.65)
        if "head" in k:
            deltas[k["head"]] = _axis_angle(r, -lean * 0.6 + look)
        legs: dict[str, Any] = {}
        arms: dict[str, Any] = {}
        spread = _ramp(t, t_spread, t_hit) if air else 1.0
        for side, leg in body.legs.items():
            home = leg["ankle"].copy()
            landing = land_f if side == side_f else land_b
            if air:  # hanging, then reaching for the landing spots (relative to the hips)
                hang = home + 0.06 * L * u
                rel = (
                    hang
                    + (landing - home - (impact if hero or jointed else 0.0) * u - 0.06 * L * u)
                    * spread
                )
                pitch = -10.0 * (1 - spread) if leg["toe"] is not None else 0.0
                legs[side] = {"target": rel + dz * u, "pitch": pitch, "planted": False}
                continue
            target = landing
            pitch = 0.0
            if side == side_k and hero and leg["toe"] is not None:
                pitch = -90.0 * settle  # on the toes (eased so they stay on the ground)
            if side == side_k and t > t_step0:  # the back foot steps forward beside the front
                w = _ramp(t, t_step0, t_step1)
                end_pt = spot(side, x_f, home[2])
                up = min(max((t - t_step0) / (t_step1 - 0.1 - t_step0), 0.0), 1.0)
                lift = 0.18 * L * math.sin(math.pi * up)  # (linear start: off the ground at once)
                target = landing + (end_pt - landing) * w + lift * u
                planted = w <= 0.0 or w >= 1.0
                legs[side] = {"target": target, "pitch": pitch * (1 - w), "planted": planted}
                continue
            legs[side] = {"target": target, "pitch": pitch}
        for side in ("L", "R"):
            arm = body.arms.get(side)
            if arm is None:
                continue
            out = 1.0 if side == "L" else -1.0
            if air:
                deltas[arm["upper"]] = _qmul(_axis_angle(f, out * 70.0), arm["base"])
                continue
            if hero and side == side_k:  # the fist on the ground, ahead and to the side
                deltas[arm["upper"]] = arm["base"]
                arms[side] = {"ground": ground + wrist_h, "ahead": 0.1 * L, "out": 0.06 * L,
                              "blend": settle, "pole": -f, "planted": settle >= 0.999}  # fmt: skip
            else:
                lift = (55.0 if hero else 25.0) * settle
                spread_a = out * (35.0 if hero else 15.0) * settle
                deltas[arm["upper"]] = _qmul(
                    _axis_angle(r, -lift), _axis_angle(f, spread_a), arm["base"]
                )
                if arm["lower"] is not None:
                    deltas[arm["lower"]] = _axis_angle(r, 20.0 * settle)
        return {"offset": dz * u + xh * f, "deltas": deltas, "legs": legs, "arms": arms}

    info = {"duration": t_end, "impact": t_hit, "style": style, "displacement": x_f}
    return t_end, False, frame, info, 0.0


def _wave(body: _Body, p: dict[str, Any]):
    f, u, r = body.f, body.u, body.r
    side = p.get("side", "R")
    if side not in ("L", "R") or side not in body.arms:
        raise _err("ROLE_MISSING", f"no {side} arm to wave with", "side")
    waves = int(p.get("waves", 3))
    rate = float(p.get("rate", 2.4))
    if not (1 <= waves <= 20 and 0.5 <= rate <= 6.0):
        raise _err("INVALID_ARGUMENT", "waves 1..20, rate 0.5..6 per second", "parameters")
    t_up = 0.45
    t_wave = t_up + waves / rate
    t_end = t_wave + 0.5
    arm = body.arms[side]
    out = 1.0 if side == "L" else -1.0

    def frame(t: float) -> dict[str, Any]:
        k = body.k
        up = _keys(t, [(0.0, 0.0), (t_up, 1.0), (t_wave, 1.0), (t_end, 0.0)])
        osc = math.sin(2 * math.pi * rate * max(0.0, t - t_up)) * _keys(
            t, [(t_up - 0.1, 0.0), (t_up + 0.15, 1.0), (t_wave - 0.15, 1.0), (t_wave, 0.0)]
        )
        deltas: dict[int, np.ndarray] = {}
        if arm["lower"] is not None:
            # In the hanging-arm frame (after ``base``): the shoulder abducts 80 degrees,
            # a little forward; the forearm turns from the upper arm's direction to
            # vertical (leaning 10 degrees forward), swinging +-22 degrees side to side.
            # A child's posed rotation is ``D_upper * D_lower`` (deltas compose down the
            # chain), so the forearm's delta is that absolute turn, seen from the shoulder.
            shoulder = _qmul(_axis_angle(f, out * 80.0 * up), _axis_angle(r, 15.0 * up))
            upper_dir = _rot(shoulder, -u)
            lean = _rot(_axis_angle(r, -10.0), u)  # (about r, +angles turn up to -f)
            fore_dir = _rot(_axis_angle(f, out * 22.0 * osc), lean)
            elbow = _between(upper_dir, fore_dir, up)
            base = arm["base"]
            deltas[arm["upper"]] = _qmul(shoulder, base)
            deltas[arm["lower"]] = _qmul(_conj(base), _conj(shoulder), elbow, shoulder, base)
        else:
            deltas[arm["upper"]] = _qmul(
                _axis_angle(f, out * (150.0 * up + 22.0 * osc)), arm["base"]
            )
        other = "R" if side == "L" else "L"
        if other in body.arms:
            deltas[body.arms[other]["upper"]] = body.arms[other]["base"]
        if "head" in k:
            deltas[k["head"]] = _qmul(
                _axis_angle(f, out * 7.0 * up), _axis_angle(u, out * 10.0 * up)
            )
        if "spine" in k:
            deltas[k["spine"]] = _axis_angle(f, -out * 3.0 * up)
        legs = {s: {"target": leg["ankle"].copy(), "pitch": 0.0} for s, leg in body.legs.items()}
        return {"offset": np.zeros(3), "deltas": deltas, "legs": legs}

    return t_end, False, frame, {"duration": t_end, "side": side}, 0.0


def _head_turn(body: _Body, p: dict[str, Any]):
    u, r = body.u, body.r
    yaw = float(p.get("yaw", 60.0))
    pitch = float(p.get("pitch", 0.0))
    turn = float(p.get("turn", 0.7))
    hold = float(p.get("hold", 1.0))
    back = bool(p.get("back", True))
    if not (-120 <= yaw <= 120 and -60 <= pitch <= 60 and 0.1 <= turn <= 10 and 0 <= hold <= 30):
        raise _err(
            "INVALID_ARGUMENT",
            "yaw -120..120, pitch -60..60 degrees; turn 0.1..10 s, hold 0..30 s",
            "parameters",
        )
    if "head" not in body.k:
        raise _err("ROLE_MISSING", "the rig has no head joint", "roles")
    t_end = turn + hold + (turn if back else 0.0)

    def frame(t: float) -> dict[str, Any]:
        k = body.k
        pairs = [(0.0, 0.0), (turn, 1.0), (turn + hold, 1.0)]
        if back:
            pairs.append((t_end, 0.0))
        w = _keys(t, pairs)
        # The head leads; the neck and chest follow less and later (by a fraction).
        lag = _keys(max(0.0, t - 0.08), pairs)
        deltas: dict[int, np.ndarray] = {}
        shares = {"head": 0.65 if "neck" in k else 0.85, "neck": 0.25, "chest": 0.1, "spine": 0.15}
        for role, share in shares.items():
            if role not in k or (role == "spine" and "chest" in k):
                continue
            ww = w if role == "head" else lag
            deltas[k[role]] = _qmul(
                _axis_angle(u, yaw * share * ww), _axis_angle(r, -pitch * share * ww)
            )
        for side in ("L", "R"):
            arm = body.arms.get(side)
            if arm is not None:
                deltas[arm["upper"]] = arm["base"]
        legs = {s: {"target": leg["ankle"].copy(), "pitch": 0.0} for s, leg in body.legs.items()}
        return {"offset": np.zeros(3), "deltas": deltas, "legs": legs}

    return t_end, False, frame, {"duration": t_end, "yaw": yaw}, 0.0


_PARAMS = {
    "idle": ("amplitude", "period"),
    "walk": ("cadence", "stride", "duty", "lift", "amplitude"),
    "run": ("cadence", "stride", "duty", "lift", "amplitude"),
    "jump": ("gravity", "height", "distance"),
    "land": ("gravity", "height", "style", "knee"),
    "wave": ("side", "waves", "rate"),
    "head_turn": ("yaw", "pitch", "turn", "hold", "back"),
}

_MOTIONS = {
    "idle": _idle,
    "walk": lambda b, p: _locomotion(b, p, False),
    "run": lambda b, p: _locomotion(b, p, True),
    "jump": _jump,
    "land": _land,
    "wave": _wave,
    "head_turn": _head_turn,
}


# --------------------------------------------------------------------------------------- #
# Fitting sampled motion into keys
# --------------------------------------------------------------------------------------- #


def _slopes(t: np.ndarray, v: np.ndarray, cyclic: bool) -> np.ndarray:
    if cyclic:  # periodic central differences (the last sample repeats the first)
        h = t[1] - t[0]
        core = v[:-1]
        d = (np.roll(core, -1, axis=0) - np.roll(core, 1, axis=0)) / (2 * h)
        return np.concatenate([d, d[:1]], axis=0)
    return np.gradient(v, t, axis=0)


def _fit_scalar(t, v, tol, cyclic, extrapolation):
    from .fitting import _hermite_curve

    d = _slopes(t, v, cyclic)
    idx, _ = _reduce(t, v, d, tol, t.size, np.zeros(0, np.int64))
    curve = _hermite_curve(t[idx], v[idx], d[idx]).replace(extrapolation=extrapolation)
    worst = float(np.max(np.abs(curve.evaluate(t) - v)))  # (measured on the built curve)
    return curve, worst, int(idx.size)


def _quat_hermite_error(t, q, dq, a, b):
    """Largest angle (radians) between the samples strictly between keys ``a`` and ``b``
    and glTF's cubic (component-wise Hermite, normalized) through them."""
    if b - a < 2:
        return 0.0, -1
    span = t[b] - t[a]
    s = ((t[a + 1 : b] - t[a]) / span)[:, None]
    s2, s3 = s * s, s * s * s
    h = (
        (2 * s3 - 3 * s2 + 1) * q[a]
        + (s3 - 2 * s2 + s) * span * dq[a]
        + (-2 * s3 + 3 * s2) * q[b]
        + (s3 - s2) * span * dq[b]
    )
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    err = 2.0 * np.arccos(np.clip(np.abs(np.sum(h * q[a + 1 : b], axis=1)), 0.0, 1.0))
    k = int(err.argmax())
    return float(err[k]), a + 1 + k


def _fit_track(t, q, tol_deg, cyclic):
    """Cubic (glTF) quaternion keys within ``tol_deg`` of every sample: keys added where the
    spline deviates most, one interval at a time (as :func:`.fitting._reduce`)."""
    import heapq

    q = q.copy()
    for i in range(1, q.shape[0]):
        q[i] = R.align(q[i - 1], q[i])
    dq = _slopes(t, q, cyclic)
    tol = math.radians(tol_deg)
    n = t.size
    heap = []
    e, k = _quat_hermite_error(t, q, dq, 0, n - 1)
    heapq.heappush(heap, (-e, 0, n - 1, k))
    keys = {0, n - 1}
    while heap and -heap[0][0] > tol:
        _, a, b, k = heapq.heappop(heap)
        if k < 0:
            continue
        keys.add(k)
        for lo, hi in ((a, k), (k, b)):
            e, at = _quat_hermite_error(t, q, dq, lo, hi)
            heapq.heappush(heap, (-e, lo, hi, at))
    idx = np.array(sorted(keys))
    track = RotationTrack.from_arrays(t[idx], q[idx], interpolation="cubic")
    track = track.replace(
        {"in": dq[idx], "out": dq[idx]}, extrapolation="cycle" if cyclic else "constant"
    )
    ev = track.evaluate(t)
    err = 2.0 * np.arccos(np.clip(np.abs(np.sum(ev * q, axis=1)), 0.0, 1.0))
    return track, math.degrees(float(err.max())), int(idx.size)


def generate(
    system: AnimationSystem,
    armature: Any,
    kind: str,
    *,
    name: str | None = None,
    start: float = 0.0,
    mapping: dict[str, str] | None = None,
    forward: str = "+y",
    travel: bool | None = None,
    tolerance: float = 0.05,
    bind: bool = True,
    **params: Any,
) -> dict[str, Any]:
    """Generate a motion on ``armature`` as a new clip (see the module docstring).

    Returns ``{clip, kind, roles, missing, channels, keys, max_error, quaternion_joints,
    ik_residual, ...}`` (the motion's own facts: period, speed, takeoff, ...). With
    ``bind`` the clip becomes the armature's action. One undo step."""
    if kind not in _MOTIONS:
        raise _err(
            "INVALID_ARGUMENT",
            f"unknown motion {kind!r}",
            "kind",
            hint=f"motions: {', '.join(KINDS)}",
        )
    obj = system._object(armature, "armature")
    rig = system.rig(obj)
    if rig is None:
        raise _err("RIG_MISSING", f"{obj.name!r} has no rig", "armature")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not 1e-4 <= tolerance <= 10
    ):
        raise _err("INVALID_ARGUMENT", "tolerance is 0.0001..10 degrees", "tolerance")
    start = float(start)
    if not math.isfinite(start) or abs(start) > 1e6:
        raise _err("INVALID_ARGUMENT", "start must be a finite time", "start")
    for key, value in params.items():
        if key not in _PARAMS[kind]:
            raise _err("INVALID_ARGUMENT", f"{kind} has no parameter {key!r}", key,
                       hint=f"{kind} takes: {', '.join(_PARAMS[kind]) or 'nothing'}")  # fmt: skip
        if isinstance(value, bool) or (isinstance(value, str) and len(value) <= 32):
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise _err("INVALID_ARGUMENT", f"{key} must be a finite number", key)
    roles = find_roles(rig, mapping)
    body = _Body(rig, roles, forward)
    if kind in ("walk", "run", "jump", "land") and not body.legs:
        raise _err("ROLE_MISSING", f"{kind} needs legs (thigh joints)", "roles",
                   hint="name them Thigh.L/Leg.L/LeftUpLeg or pass mapping")  # fmt: skip
    duration, cyclic, frame, info, travel_dist = _MOTIONS[kind](body, params)
    if not 0 < duration <= MAX_DURATION:
        raise _err("INVALID_ARGUMENT", f"motions last up to {MAX_DURATION:g} s", "parameters")
    travel = (kind in ("walk", "run")) if travel is None else bool(travel)
    count = int(round(duration * RATE))
    ts = np.linspace(0.0, duration, count + 1)
    n = len(rig.joints)
    rots = np.empty((ts.size, n, 4))
    hips = np.empty((ts.size, 3))
    worst_ik = 0.0
    for i, t in enumerate(ts.tolist()):
        spec = frame(t)
        rot, hp, res = body.solve(
            spec["offset"], spec["deltas"], spec.get("legs"), spec.get("arms")
        )
        rots[i], hips[i] = rot, hp
        worst_ik = max(worst_ik, res)
    times = ts + start
    pos_tol = tolerance / 1000.0 * body.leg_length * 20.0  # 0.05 -> a thousandth of the leg
    channels: dict[str, Any] = {}
    errors = {"degrees": 0.0, "units": 0.0}
    keys = 0
    quaternion_joints = []
    extrap = "cycle" if cyclic else "constant"
    for k, joint in enumerate(rig.joints):
        q = rots[:, k]
        if np.allclose(np.abs(np.sum(q * body.rot0[k], axis=1)), 1.0, atol=1e-12):
            continue  # never moves
        e = R.euler_filter(R.to_euler(q))
        if cyclic and not np.allclose(e[-1], e[0], atol=1e-6):
            e[-1] = e[0]  # (identical rotations; keep the loop closed)
        if np.max(np.abs(e[:, 1])) > GIMBAL_MARGIN:
            track, err, nk = _fit_track(times, q, tolerance, cyclic)
            channels[f'joints["{joint.name}"].rotation'] = track
            errors["degrees"] = max(errors["degrees"], err)
            keys += nk
            quaternion_joints.append(joint.name)
            continue
        for c, axis in enumerate("xyz"):
            curve, err, nk = _fit_scalar(times, e[:, c], tolerance, cyclic, extrap)
            channels[f'joints["{joint.name}"].rotation_euler.{axis}'] = curve
            errors["degrees"] = max(errors["degrees"], err)
            keys += nk
    hj = rig.joints[body.hips].name
    for c, axis in enumerate("xyz"):
        v = hips[:, c]
        if np.allclose(v, body.pos0[body.hips, c], atol=1e-12):
            continue
        curve, err, nk = _fit_scalar(times, v, pos_tol, cyclic, extrap)
        channels[f'joints["{hj}"].position.{axis}'] = curve
        errors["units"] = max(errors["units"], err)
        keys += nk
    if travel and travel_dist > 0:
        # Root motion: the armature moves along its own forward direction.
        # (in the parent's frame, where the object's own position and rotation live)
        direction = _rot(np.asarray(obj.rotation, np.float64), body.f)
        base = np.asarray(obj.position, np.float64)
        for c, axis in enumerate("xyz"):
            if abs(direction[c]) < 1e-12:
                continue
            curve = FCurve.from_keys(
                [
                    (start, float(base[c])),
                    (start + duration, float(base[c] + direction[c] * travel_dist)),
                ],
                interpolation="linear",
            ).replace(extrapolation="cycle_offset")
            channels[f"transform.position.{axis}"] = curve
            keys += 2
    from . import paths as P
    from .clip import Clip

    meta = {"generator": {"kind": kind, "version": 1, "forward": forward, "travel": travel,
                          "tolerance": tolerance, "start": start,
                          "params": {k: v for k, v in params.items()}}}  # fmt: skip
    text = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False)
    clip = Clip(P.OBJECT, tuple(sorted(channels.items())), (), None, text)
    clip_name = name or system._unique_clip_name(kind.replace("_", " ").title())
    with system._op(f"Generate {kind} for {obj.name!r}"):
        handle = system.add_clip(clip_name, clip)
        if bind:
            system.set_action(obj, handle.name)
    report = {
        "clip": handle.name,
        "kind": kind,
        "roles": roles,
        "missing": [r for r in ROLES if r not in roles],
        "channels": len(channels),
        "keys": keys,
        "samples": int(ts.size),
        "max_error": errors,
        "quaternion_joints": quaternion_joints,
        "ik_residual": worst_ik,
        "cyclic": cyclic,
        "duration": duration,
        "travel": travel_dist if travel else 0.0,
        "leg_length": body.leg_length,
    }
    report.update(info)
    return report


# --------------------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------------------- #


def follow_camera(
    system: AnimationSystem,
    camera: Any,
    target: Any,
    *,
    start: float | None = None,
    end: float | None = None,
    offset: Any = (0.0, -6.0, 2.5),
    lag: float = 0.35,
    aim: Any = (0.0, 0.0, 0.0),
    tolerance: float = 0.005,
    name: str = "follow",
) -> dict[str, Any]:
    """Key ``camera`` to follow ``target`` at ``offset`` (world axes) with a critically
    damped lag of ``lag`` seconds (0: rigid), fitted into position keys within
    ``tolerance``, and aim it at the target (+ ``aim``) with a look-at constraint."""
    from .report import sample_world

    cam = system._object(camera, "camera")
    tgt = system._object(target, "target")
    if cam.camera is None:
        raise _err("INVALID_TARGET", f"{cam.name!r} is not a camera", "camera")
    s = system.start if start is None else float(start)
    e = system.end if end is None else float(end)
    if not (math.isfinite(s) and math.isfinite(e) and e > s and e - s <= 600.0):
        raise _err("INVALID_RANGE", "a follow needs 0 < end - start <= 600 s", "range")
    off = np.asarray(offset, np.float64).reshape(-1)
    aim_v = np.asarray(aim, np.float64).reshape(-1)
    if (
        off.shape != (3,)
        or aim_v.shape != (3,)
        or not (np.all(np.isfinite(off)) and np.all(np.isfinite(aim_v)))
    ):
        raise _err("INVALID_ARGUMENT", "offset and aim are finite [x, y, z]", "offset")
    if isinstance(lag, bool) or not isinstance(lag, (int, float)) or not 0.0 <= lag <= 10.0:
        raise _err("INVALID_ARGUMENT", "lag is 0..10 s", "lag")
    count = int(min(20_000, max(16, round((e - s) * 60.0) + 1)))
    ts = np.linspace(s, e, count)
    goal = sample_world(system, tgt, ts)[:, :3, 3] + off
    pos = goal.copy()
    if lag > 0:
        # Critically damped spring, integrated exactly per step (piecewise-linear goal).
        w = 2.0 / lag
        x, v = goal[0].copy(), np.zeros(3)
        for i in range(1, count):
            h = ts[i] - ts[i - 1]
            g0, g1 = goal[i - 1], goal[i]
            gv = (g1 - g0) / h
            y0 = x - g0 + 2.0 * gv / w  # error relative to the moving goal (ramp)
            yd0 = v - gv + 0.0
            ex = math.exp(-w * h)
            c1 = y0
            c2 = yd0 + w * y0
            y = (c1 + c2 * h) * ex
            yd = (c2 - w * (c1 + c2 * h)) * ex
            x = g1 - 2.0 * gv / w + y
            v = gv + yd
            pos[i] = x
    from .fitting import fit_samples

    worst = 0.0
    with system._op(f"Follow {tgt.name!r} with {cam.name!r}"):
        for c, axis in enumerate("xyz"):
            curve, rep = fit_samples(ts, pos[:, c], tolerance)
            worst = max(worst, rep["max_error"])
            system.set_channel(cam, f"transform.position.{axis}", curve)
        existing = [c for c in system.constraints(cam) if c["name"] == name]
        if existing:
            system.remove_constraint(cam, name)
        system.add_constraint(cam, {"name": name, "type": "look_at", "target": {"object": tgt.id},
                                    "target_offset": aim_v.tolist()})  # fmt: skip
    return {"camera": cam.name, "target": tgt.name, "samples": count, "max_error": worst,
            "lag": lag, "basis": "sampled"}  # fmt: skip
