"""An original anatomical figure: implicit shapes built on a table of joint centres.

:func:`figure_joints` lays out a standing human's joint centres (Z up, facing -Y, the
character's left +X, feet on ``z = 0``) from anthropometric proportions -- spine and neck
along the back, clavicles from the sternum, shoulders, elbows, wrists, the knuckles and
phalanges of five fingers, hips, knees, ankles, heels, balls and toe tips -- in a chosen
rest pose (arms down, A-pose or T-pose; bent or straight elbows, knees and fingers).
:func:`figure_shapes` builds the body on those centres with blended capsules and
ellipsoids: a pelvis and glutes, abdomen, ribcage, pectorals and back muscles, trapezius
and deltoids, biceps and triceps, forearm muscles, palms, fingers with knuckles, a thumb
with its thenar pad, quadriceps, hamstrings, kneecaps, calves, ankle bones, feet with
flat soles, a neck, and a head with a jaw, chin, nose, brow and ears.
:func:`figure_mesh` meshes it (marching cubes), optionally remeshes it to clean,
evenly sized triangles and projects the result back onto the implicit surface, with
normals from the field's gradient.

Because the joints come first and the body is modeled around them, every joint centre
is known exactly: Aether3D's rigging tests measure automatic landmarks against them.
The figure is Aether3D's own (no scanned or third-party asset).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import ValidationError

FIGURE_DEFAULTS: dict[str, float] = {
    "height": 1.75,
    "head": 1.0,  # head size factor (stylized characters: 1.3 .. 1.8)
    "shoulders": 1.0,  # shoulder width factor
    "hips": 1.0,  # hip width factor
    "legs": 1.0,  # leg length factor (the stature stays)
    "arms": 1.0,  # arm length factor
    "limbs": 1.0,  # limb thickness factor
    "muscle": 1.0,  # muscle mass factor (0.6 slim .. 1.5 athletic)
    "belly": 1.0,  # belly size factor
    "chest": 1.0,  # chest depth factor
    "arm_angle": 35.0,  # degrees from hanging straight down (0) to a T-pose (90)
    "elbow_bend": 10.0,  # degrees the forearm bends forward at rest
    "knee_bend": 4.0,  # degrees the shin bends back at rest (the knee comes forward)
    "finger_curl": 12.0,  # degrees each finger joint bends toward the palm at rest
    "finger_spread": 6.0,  # degrees between neighbouring fingers
    # Extra distance between the feet (metres): a little apart, as a rigging rest pose
    # stands, so the thighs part below the groin instead of pressing together (with the
    # feet under the hips they fuse down to mid-thigh, and the cut between them leaves
    # flat inner thighs that show in every stride).
    "stance": 0.08,
    "fingers": 1.0,  # 1: separate fingers; 0: a mitten (fingers fused into one paddle)
}

CROTCH_BELOW_HIP = 0.048  # of the stature: the crotch below the hip joints' centres
LEG_GAP = 0.005  # of the stature: the gap left between the thighs, knees and feet

FINGERS = ("index", "middle", "ring", "pinky")
# Per finger, in units of the stature: metacarpal (wrist to knuckle), proximal, middle and
# distal phalanx lengths, the knuckle's offset across the palm (+ toward the thumb), and the
# finger's radius at its base.
_FINGER = {
    "index": (0.0470, 0.0250, 0.0145, 0.0120, 0.0190, 0.00500),
    "middle": (0.0480, 0.0275, 0.0165, 0.0125, 0.0065, 0.00520),
    "ring": (0.0455, 0.0255, 0.0158, 0.0122, -0.0060, 0.00490),
    "pinky": (0.0415, 0.0205, 0.0115, 0.0110, -0.0170, 0.00430),
}


def _options(options: dict[str, Any]) -> dict[str, float]:
    o = dict(FIGURE_DEFAULTS)
    unknown = set(options) - set(o)
    if unknown:
        raise ValidationError(f"unknown figure options {sorted(unknown)}")
    for k, v in options.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValidationError(f"{k} is a number")
        o[k] = float(v)
    for k in ("height", "head", "shoulders", "hips", "legs", "arms", "limbs", "muscle",
              "belly", "chest"):  # fmt: skip
        if not o[k] > 0:
            raise ValidationError(f"{k} must be positive")
    if not 0.0 <= o["arm_angle"] <= 100.0:
        raise ValidationError("arm_angle is 0..100 degrees")
    for k in ("elbow_bend", "knee_bend", "finger_curl"):
        if not -10.0 <= o[k] <= 120.0:
            raise ValidationError(f"{k} is -10..120 degrees")
    if not 0.0 <= o["finger_spread"] <= 30.0:
        raise ValidationError("finger_spread is 0..30 degrees")
    if o["fingers"] not in (0.0, 1.0):
        raise ValidationError("fingers is 1 (separate) or 0 (a mitten)")
    return o


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _rotate(v: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    """``v`` turned about the unit ``axis`` by ``degrees`` (Rodrigues)."""
    a = math.radians(degrees)
    k = _unit(axis)
    return v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * float(k @ v) * (1 - math.cos(a))


def figure_joints(**options: Any) -> dict[str, np.ndarray]:
    """The figure's joint centres (see the module docstring), keyed by the landmark names
    of :func:`aether3d.animation.landmarks.detect` (``hip.L``, ``knee.L``, ...) plus
    ``upper_chest`` and the fingers (``thumb.1.L`` .. ``thumb.tip.L``, ``index.1.L`` (the
    knuckle) .. ``index.tip.L``, likewise ``middle``, ``ring``, ``pinky``)."""
    o = _options(options)
    H = o["height"]
    J: dict[str, np.ndarray] = {}
    # Midline: the spine runs behind the torso's centre with a gentle S curve.
    ankle_z = 0.039 * H
    hip_z = ankle_z + (0.530 * H - ankle_z) * o["legs"]
    head_h = 0.130 * H * o["head"]  # chin to crown
    crown = H
    chin = crown - head_h
    neck_z = chin - 0.040 * H  # the neck's base (C7)
    shoulder_z = neck_z - 0.022 * H
    torso = neck_z - hip_z
    J["pelvis"] = np.array([0.0, 0.012 * H, hip_z + 0.03 * H])
    J["spine"] = np.array([0.0, 0.026 * H, hip_z + 0.27 * torso])
    J["chest"] = np.array([0.0, 0.022 * H, hip_z + 0.52 * torso])
    J["upper_chest"] = np.array([0.0, 0.016 * H, hip_z + 0.76 * torso])
    J["neck"] = np.array([0.0, 0.016 * H, neck_z])
    J["head"] = np.array([0.0, 0.004 * H, chin + 0.30 * head_h])
    J["head_top"] = np.array([0.0, 0.0, crown])
    # Arms: built hanging (palm toward the thigh, thumb forward), then raised by arm_angle
    # in the frontal plane and bent at the elbow.
    fwd = np.array([0.0, -1.0, 0.0])
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sw = 0.102 * H * o["shoulders"]
        J[f"clavicle.{side}"] = np.array([sx * 0.012 * H, -0.030 * H, neck_z - 0.012 * H])
        sh = np.array([sx * sw, 0.008 * H, shoulder_z - 0.004 * H])
        J[f"shoulder.{side}"] = sh
        down = np.array([sx * math.sin(math.radians(o["arm_angle"])), 0.0,
                         -math.cos(math.radians(o["arm_angle"]))])  # fmt: skip
        palm_n = np.array([-sx * math.cos(math.radians(o["arm_angle"])), 0.0,
                           -math.sin(math.radians(o["arm_angle"]))])  # fmt: skip
        thumb_dir = fwd.copy()
        upper = 0.180 * H * o["arms"]
        fore = 0.146 * H * o["arms"]
        elbow = sh + upper * down
        J[f"elbow.{side}"] = elbow
        bend_axis = _unit(np.cross(down, fwd))
        hand_dir = _rotate(down, bend_axis, o["elbow_bend"])
        palm_n = _rotate(palm_n, bend_axis, o["elbow_bend"])
        thumb_dir = _rotate(thumb_dir, bend_axis, o["elbow_bend"])
        wrist = elbow + fore * hand_dir
        J[f"wrist.{side}"] = wrist
        s = o["arms"]
        # Rotating about the palm's normal by a positive angle fans toward the thumb when
        # cross(n, d) points to the thumb's side.
        fan = 1.0 if float(np.cross(palm_n, hand_dir) @ thumb_dir) > 0 else -1.0
        for i, f in enumerate(FINGERS):
            meta, p1, p2, p3, across, _r = _FINGER[f]
            knuckle = wrist + hand_dir * meta * H * s + thumb_dir * across * H * s
            d = _rotate(hand_dir, palm_n, fan * (1.5 - i) * o["finger_spread"])
            pts = [knuckle]
            for k, length in enumerate((p1, p2, p3)):
                # (a positive turn about d x n brings d toward the palm's side)
                d = _rotate(d, np.cross(d, palm_n), o["finger_curl"] * (1.0 if k else 0.6))
                pts.append(pts[-1] + d * length * H * s)
            for k, name in enumerate(("1", "2", "3", "tip")):
                J[f"{f}.{name}.{side}"] = pts[k]
        # The thumb: from its carpometacarpal joint near the wrist, forward and toward the
        # palm's side, curling across it.
        cmc = wrist + hand_dir * 0.012 * H * s + thumb_dir * 0.014 * H * s + palm_n * 0.008 * H * s
        td = _unit(0.75 * hand_dir + 0.55 * thumb_dir + 0.30 * palm_n)
        pts = [cmc]
        for k, length in enumerate((0.026, 0.018, 0.014)):
            if k:
                td = _rotate(td, np.cross(td, palm_n), o["finger_curl"] * 0.7)
            pts.append(pts[-1] + td * length * H * s)
        for k, name in enumerate(("1", "2", "3", "tip")):
            J[f"thumb.{name}.{side}"] = pts[k]
        J[f"hand_tip.{side}"] = J[f"middle.tip.{side}"]
    # Legs.
    kb = math.radians(o["knee_bend"])
    for side, sx in (("L", 1.0), ("R", -1.0)):
        hx = sx * 0.050 * H * o["hips"]
        fx = hx + sx * o["stance"] / 2.0
        hip = np.array([hx, 0.004 * H, hip_z])
        ankle = np.array([fx, 0.012 * H, ankle_z])
        knee = hip + 0.505 * (ankle - hip)
        knee[1] = -0.008 * H - math.sin(kb) * 0.12 * H
        knee[2] = ankle_z + 0.497 * (hip_z - ankle_z)
        J[f"hip.{side}"] = hip
        J[f"knee.{side}"] = knee
        J[f"ankle.{side}"] = ankle
        out = sx * math.tan(math.radians(7.0))  # feet turn out a little
        J[f"heel.{side}"] = np.array([fx, 0.046 * H, 0.0])
        J[f"ball.{side}"] = np.array([fx + out * 0.072 * H, -0.072 * H, 0.011 * H])
        J[f"toe_tip.{side}"] = np.array([fx + out * 0.112 * H, -0.112 * H, 0.006 * H])
    return J


def figure_shapes(**options: Any) -> list[dict[str, Any]]:
    """The figure's implicit shapes (for :func:`aether3d.sculpt.implicit.implicit_surface`);
    see the module docstring and :data:`FIGURE_DEFAULTS`."""
    return _figure_shapes(options, 0.0)


PARTS = ("torso", "hips", "neck", "head", "upper_arm", "forearm", "hand", "thigh", "knee",
         "shin", "foot")  # fmt: skip
"""The body parts the figure's shapes belong to (``figure_garment`` dresses some of them;
the limbs' are sided: ``upper_arm.L``...)."""


def _figure_shapes(
    options: dict[str, Any], min_gap: float, tags: list[str] | None = None
) -> list[dict[str, Any]]:
    o = _options(options)
    J = figure_joints(**options)
    H = o["height"]
    m = o["muscle"]
    limb = o["limbs"]
    shapes: list[dict[str, Any]] = []
    marks: list[tuple[int, str]] = []

    def mark(part: str) -> None:  # (the shapes from here on belong to ``part``)
        marks.append((len(shapes), part))

    def capsule(a, b, r0, r1=None, blend=0.012 * H):
        shapes.append({"type": "capsule", "a": [float(v) for v in a], "b": [float(v) for v in b],
                       "radius": float(r0), "radius_b": float(r0 if r1 is None else r1),
                       "blend": float(blend)})  # fmt: skip

    def ell(c, r, blend=0.012 * H, axis=None, side=None):
        d = {"type": "ellipsoid", "center": [float(v) for v in c],
             "radii": [float(v) for v in r], "blend": float(blend)}  # fmt: skip
        if axis is not None:
            d["rotation"] = _quat(axis, side)
        shapes.append(d)

    def along(a, b, t):
        return a + t * (b - a)

    # -- torso ------------------------------------------------------------------------------
    mark("torso")
    spine, chest, upper = J["spine"], J["chest"], J["upper_chest"]
    neck = J["neck"]
    hip_z = J["hip.L"][2]
    hw = o["hips"]
    sw = o["shoulders"]
    # Big masses blend widely; muscles blend narrowly, so they read as relief on the masses
    # (a smooth union bulges by up to a quarter of its blend radius where shapes overlap).
    capsule([0, 0.006 * H, hip_z + 0.02 * H], [0, 0.008 * H, neck[2] - 0.03 * H],
            0.060 * H, 0.054 * H, blend=0.03 * H)  # fmt: skip
    mark("hips")
    ell([0, 0.004 * H, hip_z + 0.034 * H], [0.078 * H * hw, 0.052 * H, 0.060 * H], 0.03 * H)
    for sx in (1.0, -1.0):  # glutes
        ell([sx * 0.040 * H * hw, 0.036 * H, hip_z + 0.002 * H],
            [0.042 * H * hw * m ** 0.3, 0.032 * H * m ** 0.5, 0.054 * H], 0.016 * H)  # fmt: skip
    mark("torso")
    belly = o["belly"]
    ell(
        [0, -0.008 * H * belly, spine[2] - 0.012 * H],
        [0.074 * H * max(belly, 0.7) ** 0.5, 0.045 * H * belly**0.7, 0.074 * H],
        0.03 * H,
    )
    cd = o["chest"]
    ell([0, 0.004 * H, chest[2] + 0.02 * H], [0.084 * H * sw, 0.055 * H * cd, 0.098 * H],
        0.03 * H)  # ribcage  # fmt: skip
    # the upper back, wider than the ribcage below it: the torso's taper
    ell([0, 0.014 * H, upper[2] + 0.004 * H], [0.094 * H * sw, 0.044 * H, 0.050 * H], 0.03 * H)
    for sx in (1.0, -1.0):
        sh = J[f"shoulder.{'L' if sx > 0 else 'R'}"]
        # the shoulder girdle and the trapezius' slope (a mass: wide blends)
        capsule([sx * 0.03 * H, 0.006 * H, sh[2] + 0.000 * H],
                [sx * 0.080 * H * sw, 0.008 * H, sh[2] + 0.006 * H], 0.026 * H,
                0.025 * H, blend=0.025 * H)  # fmt: skip
        capsule([sx * 0.02 * H, 0.020 * H, neck[2] + 0.004 * H],
                [sx * 0.082 * H * sw, 0.012 * H, sh[2] + 0.014 * H],
                0.020 * H * m ** 0.4, 0.017 * H, blend=0.022 * H)  # fmt: skip
        # pectorals: flat, their long axis rising toward the shoulder (relief)
        ell([sx * 0.040 * H * sw, -0.029 * H * cd, upper[2] - 0.012 * H],
            [0.034 * H * sw, 0.014 * H * m ** 0.5 * cd, 0.046 * H], 0.010 * H,
            axis=np.array([sx * 1.0, 0.0, 0.35]), side=np.array([0.0, 0.0, 1.0]))  # fmt: skip
    # -- neck and head ----------------------------------------------------------------------
    head = J["head"]
    hs = o["head"]
    mark("neck")
    capsule([0, 0.012 * H, neck[2] - 0.012 * H], [0, 0.002 * H, head[2] + 0.004 * H],
            0.031 * H * limb ** 0.5, 0.027 * H, blend=0.02 * H)  # fmt: skip
    mark("head")
    hr = 0.13 * H * hs  # head height (chin to crown)
    top = J["head_top"][2]
    ell([0, 0.012 * hr, top - 0.42 * hr], [0.355 * hr, 0.42 * hr, 0.43 * hr], 0.02 * H)  # cranium
    ell([0, -0.10 * hr, top - 0.66 * hr], [0.285 * hr, 0.30 * hr, 0.28 * hr], 0.02 * H)  # face
    ell([0, -0.19 * hr, top - 0.86 * hr], [0.18 * hr, 0.17 * hr, 0.11 * hr], 0.015 * H)  # jaw
    ell([0, -0.395 * hr, top - 0.58 * hr], [0.042 * hr, 0.06 * hr, 0.10 * hr], 0.008 * H,
        axis=np.array([0.0, -0.35, 1.0]))  # nose  # fmt: skip
    ell([0, -0.34 * hr, top - 0.37 * hr], [0.24 * hr, 0.05 * hr, 0.045 * hr], 0.008 * H)  # brow
    for sx in (1.0, -1.0):
        ell([sx * 0.36 * hr, 0.05 * hr, top - 0.50 * hr], [0.04 * hr, 0.09 * hr, 0.13 * hr],
            0.006 * H)  # ears  # fmt: skip
    for sx in (1.0, -1.0):  # shallow eye sockets, then the eyes in them
        shapes.append({"type": "sphere", "center": [sx * 0.13 * hr, -0.455 * hr, top - 0.46 * hr],
                       "radius": 0.08 * hr, "blend": 0.008 * H, "subtract": True})  # fmt: skip
    for sx in (1.0, -1.0):
        shapes.append({"type": "sphere", "center": [sx * 0.13 * hr, -0.335 * hr, top - 0.46 * hr],
                       "radius": 0.055 * hr, "blend": 0.003 * H})  # fmt: skip
    # -- arms -------------------------------------------------------------------------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        sh, el, wr = J[f"shoulder.{side}"], J[f"elbow.{side}"], J[f"wrist.{side}"]
        ua = _unit(el - sh)
        fa = _unit(wr - el)
        fwd = np.array([0.0, -1.0, 0.0])
        front = _unit(fwd - (fwd @ ua) * ua)  # the arm's front (biceps side)
        r_up = 0.028 * H * limb
        mark(f"upper_arm.{side}")
        capsule(sh, el, r_up, 0.021 * H * limb, blend=0.012 * H)
        ell(
            sh + ua * 0.022 * H + np.array([sx * 0.008 * H, 0, 0.006 * H]),
            [0.033 * H * m**0.4, 0.034 * H * m**0.4, 0.048 * H],
            0.014 * H,
            axis=ua,
        )  # deltoid
        ell(
            along(sh, el, 0.55) + front * 0.011 * H * m**0.5,
            [0.018 * H * m**0.5, 0.017 * H * m**0.5, 0.050 * H],
            0.010 * H,
            axis=ua,
        )
        ell(
            along(sh, el, 0.45) - front * 0.009 * H * m**0.5,
            [0.019 * H * m**0.5, 0.017 * H * m**0.5, 0.056 * H],
            0.010 * H,
            axis=ua,
        )
        mark(f"forearm.{side}")
        ell(el - front * 0.012 * H, [0.011 * H, 0.011 * H, 0.012 * H], 0.010 * H)  # olecranon
        capsule(el, wr, 0.022 * H * limb, 0.0145 * H * limb, blend=0.012 * H)
        f_front = _unit(fwd - (fwd @ fa) * fa)
        ell(
            along(el, wr, 0.28) + f_front * 0.003 * H,
            [0.022 * H * m**0.4, 0.020 * H * m**0.4, 0.050 * H],
            0.010 * H,
            axis=fa,
        )
        # -- the hand: a palm, knuckles, fingers and a thumb --------------------------------
        mark(f"hand.{side}")
        mid1 = J[f"middle.1.{side}"]
        idx1, pky1 = J[f"index.1.{side}"], J[f"pinky.1.{side}"]
        across = _unit(idx1 - pky1)  # toward the thumb
        palm_n = sx * _unit(np.cross(_unit(mid1 - wr), across))  # the way the palm faces
        palm_c = along(wr, mid1, 0.52)
        ell(wr + (mid1 - wr) * 0.08, [0.017 * H * limb, 0.011 * H * limb, 0.012 * H], 0.010 * H,
            axis=_unit(mid1 - wr), side=across)  # wrist  # fmt: skip
        ell(palm_c, [0.0245 * H, 0.0098 * H, 0.030 * H], 0.010 * H, axis=_unit(mid1 - wr),
            side=across)  # fmt: skip
        if o["fingers"] == 0.0:  # a mitten: one paddle to the fingertips
            tip = J[f"middle.tip.{side}"]
            ell(along(mid1, tip, 0.45), [0.022 * H, 0.0082 * H, 0.032 * H], 0.010 * H,
                axis=_unit(tip - mid1), side=across)  # fmt: skip
        else:
            for f in FINGERS:
                r0 = _FINGER[f][5] * H * limb
                p = [J[f"{f}.{k}.{side}"] for k in ("1", "2", "3", "tip")]
                ell(p[0], [r0 * 1.15, r0 * 1.1, r0 * 1.2], 0.004 * H)  # knuckle
                radii = (r0, r0 * 0.9, r0 * 0.82, r0 * 0.72)
                for k in range(3):
                    end = p[k + 1] - _unit(p[k + 1] - p[k]) * (radii[k + 1] if k == 2 else 0)
                    capsule(p[k], end, radii[k], radii[k + 1], blend=0.0035 * H)
        t = [J[f"thumb.{k}.{side}"] for k in ("1", "2", "3", "tip")]
        capsule(t[0], t[1], 0.0068 * H * limb, 0.0060 * H * limb, blend=0.008 * H)
        capsule(t[1], t[2], 0.0058 * H * limb, 0.0053 * H * limb, blend=0.003 * H)
        capsule(t[2], t[3] - _unit(t[3] - t[2]) * 0.0045 * H, 0.0052 * H * limb,
                0.0045 * H * limb, blend=0.003 * H)  # fmt: skip
        ell(along(t[0], t[1], 0.35) + palm_n * 0.003 * H, [0.011 * H, 0.008 * H, 0.017 * H],
            0.010 * H, axis=_unit(t[1] - t[0]))  # thenar pad  # fmt: skip
    # -- legs -------------------------------------------------------------------------------
    for side, sx in (("L", 1.0), ("R", -1.0)):
        hip, knee, ankle = J[f"hip.{side}"], J[f"knee.{side}"], J[f"ankle.{side}"]
        th = _unit(knee - hip)
        sh_ = _unit(ankle - knee)
        mark(f"thigh.{side}")
        capsule(hip + np.array([0, 0, 0.025 * H]), knee, 0.050 * H * limb, 0.033 * H * limb,
                blend=0.03 * H)  # fmt: skip
        ell(
            along(hip, knee, 0.45) + np.array([0, -0.020 * H, 0]),
            [0.040 * H * m**0.4, 0.034 * H * m**0.5, 0.115 * H],
            0.025 * H,
            axis=th,
        )
        ell(
            along(hip, knee, 0.42) + np.array([sx * -0.004 * H, 0.010 * H, 0]),
            [0.034 * H, 0.034 * H, 0.10 * H],
            0.025 * H,
            axis=th,
        )  # adductors (kept off the mirror plane: the inner thighs stay round)
        ell(
            along(hip, knee, 0.48) + np.array([0, 0.022 * H, 0]),
            [0.034 * H, 0.030 * H * m**0.5, 0.10 * H],
            0.025 * H,
            axis=th,
        )  # hamstrings
        mark(f"knee.{side}")
        ell(knee + np.array([0, -0.022 * H, 0.004 * H]), [0.012 * H, 0.008 * H, 0.015 * H],
            0.014 * H)  # kneecap  # fmt: skip
        mark(f"shin.{side}")
        capsule(knee, ankle, 0.031 * H * limb, 0.0175 * H * limb, blend=0.015 * H)
        ell(
            along(knee, ankle, 0.30) + np.array([sx * 0.002 * H, 0.020 * H, 0]),
            [0.030 * H * m**0.4, 0.030 * H * m**0.5, 0.075 * H],
            0.02 * H,
            axis=sh_,
        )  # calf
        for side_off in (1.0, -1.0):  # ankle bones
            ell(ankle + np.array([side_off * 0.014 * H, 0.002 * H, -0.002 * H]),
                [0.005 * H, 0.006 * H, 0.007 * H], 0.010 * H)  # fmt: skip
        heel, ball, toe = J[f"heel.{side}"], J[f"ball.{side}"], J[f"toe_tip.{side}"]
        mark(f"foot.{side}")
        hc = np.array([ankle[0], heel[1] - 0.016 * H, 0.020 * H])
        ell(hc, [0.020 * H, 0.024 * H, 0.022 * H], 0.012 * H)  # heel
        capsule(hc + np.array([0, 0, 0.004 * H]), ball + np.array([0, 0, 0.006 * H]),
                0.020 * H, 0.017 * H, blend=0.015 * H)  # arch  # fmt: skip
        ell(ball + np.array([0, 0, 0.002 * H]), [0.028 * H, 0.020 * H, 0.013 * H], 0.010 * H)
        # five toes from the ball's line, the big toe on the inside
        fdir = _unit(toe - ball)
        lat = _unit(np.cross(fdir, np.array([0.0, 0.0, 1.0])))  # toward the foot's outside
        lat = lat if float(lat[0] * sx) > 0 else -lat
        for k, (off, length, r) in enumerate(((-0.0125, 0.040, 0.0075), (-0.0035, 0.037, 0.0056),
                                              (0.0045, 0.034, 0.0052), (0.0118, 0.030, 0.0048),
                                              (0.0185, 0.025, 0.0044))):  # fmt: skip
            base = ball + lat * off * H + np.array([0, 0, 0.0035 * H]) - fdir * 0.006 * H
            tip = base + fdir * (length * H - r * H) - np.array([0, 0, 0.003 * H])
            tip[2] = max(tip[2], r * H)
            capsule(base, tip, r * H * 1.1, r * H, blend=0.006 * H)
            del k
        capsule(ankle + np.array([0, 0.0, 0.01 * H]), hc + np.array([0, -0.006 * H, 0.014 * H]),
                0.018 * H, 0.018 * H, blend=0.015 * H)  # fmt: skip
    # The legs apart below the crotch: thighs that touch (most do) are two surfaces, not a
    # web the other leg would drag along (a 100-degree hip flexion stretched it 11 times).
    crotch = hip_z - CROTCH_BELOW_HIP * H
    mark("cut.gap")
    shapes.append({"type": "box", "center": [0.0, 0.0, 0.5 * crotch - 0.25 * H],
                   "size": [max(LEG_GAP * H, min_gap), H, crotch + 0.5 * H],
                   "blend": 0.004 * H, "subtract": True})  # fmt: skip
    # Flat soles: everything below the floor is cut away (last: it subtracts from all).
    mark("cut.floor")
    shapes.append({"type": "box", "center": [0.0, 0.0, -0.5 * H], "size": [4 * H, 4 * H, H],
                   "blend": 0.004 * H, "subtract": True})  # fmt: skip
    if tags is not None:
        bounds = [i for i, _ in marks] + [len(shapes)]
        for (start, part), end in zip(marks, bounds[1:], strict=True):
            tags.extend([part] * (end - start))
    return shapes


LEGS = ("thigh", "knee", "shin", "foot")


def figure_garment(
    parts: Any,
    inflate: float,
    voxel_size: float = 0.008,
    *,
    top: float | None = None,
    bottom: float | None = None,
    **options: Any,
) -> Any:
    """A garment (or armour) shell over some of the figure's ``parts`` (:data:`PARTS`, a
    limb's with or without its side: ``"upper_arm"`` is both, ``"upper_arm.L"`` one): their
    shapes inflated by ``inflate`` metres and meshed like the figure, flat at the floor and
    cut flat above ``top`` and below ``bottom`` (heights in metres: a jacket ends below
    the waist, trousers start above it). A garment over the legs is meshed a side at a
    time, each clipped to its own half (the trousers' legs meet on the mirror plane): the
    figure's legs are apart below the crotch by a gap of its own voxels, which a coarser
    garment could not cut as finely -- its inner faces would lie inside the thighs."""
    import copy

    from ..mesh import Mesh
    from .implicit import implicit_surface

    wanted = {parts} if isinstance(parts, str) else set(parts)
    for p_ in wanted:
        if p_.split(".")[0] not in PARTS:
            raise ValidationError(f"unknown figure part {p_!r}; one of {', '.join(PARTS)}")
    if isinstance(inflate, bool) or not isinstance(inflate, (int, float)) or inflate < 0:
        raise ValidationError("inflate is a non-negative number of metres")
    for name, value in (("top", top), ("bottom", bottom)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValidationError(f"{name} is a height in metres")
    tags: list[str] = []
    shapes = _figure_shapes(options, 2.2 * float(voxel_size), tags)
    H = _options(options)["height"]
    cuts = []  # (flat: what lies above the top, below the bottom)
    if top is not None:
        cuts.append({"type": "box", "center": [0.0, 0.0, float(top) + 2.0 * H],
                     "size": [4.0 * H, 4.0 * H, 4.0 * H], "subtract": True})  # fmt: skip
    if bottom is not None:
        cuts.append({"type": "box", "center": [0.0, 0.0, float(bottom) - 2.0 * H],
                     "size": [4.0 * H, 4.0 * H, 4.0 * H], "subtract": True})  # fmt: skip

    def dressed(side: str | None) -> list[dict[str, Any]]:
        out = []
        for sh, tag in zip(shapes, tags, strict=True):
            if tag.startswith("cut"):
                if tag == "cut.floor" or side is None:
                    out.append(copy.deepcopy(sh))
                continue
            sided = "." in tag
            if side is not None and sided and not tag.endswith(f".{side}"):
                continue
            if sh.get("subtract") or not (tag in wanted or tag.split(".")[0] in wanted):
                continue
            sh = copy.deepcopy(sh)
            for key in ("radius", "radius_b"):
                if key in sh:
                    sh[key] += inflate
            if "radii" in sh:
                sh["radii"] = [r + inflate for r in sh["radii"]]
            out.append(sh)
        if side is not None:  # (everything across the mirror plane cut away)
            sign = 1.0 if side == "L" else -1.0
            out.append({"type": "box", "center": [-sign * 2.0 * H, 0.0, 0.5 * H],
                        "size": [4.0 * H, 4.0 * H, 4.0 * H], "subtract": True})  # fmt: skip
        return out + [dict(c) for c in cuts]

    if not any(p_.split(".")[0] in LEGS for p_ in wanted):
        return implicit_surface(dressed(None), voxel_size)
    halves = []
    for side in ("L", "R"):
        chosen = dressed(side)
        if any(not sh.get("subtract") for sh in chosen):
            halves.append(implicit_surface(chosen, voxel_size))
    if len(halves) == 1:
        return halves[0]
    a, b = halves
    return Mesh(
        np.concatenate([a.positions, b.positions]),
        np.concatenate([a.triangles, b.triangles + a.vertex_count]),
        normals=np.concatenate([a.normals, b.normals]),
    )


def _quat(axis: np.ndarray, side: np.ndarray | None = None) -> list[float]:
    """A rotation taking local +Z to ``axis`` (and local +X toward ``side`` when given)."""
    from ..animation import rotation as R

    z = _unit(np.asarray(axis, np.float64))
    hint = np.asarray(side, np.float64) if side is not None else np.array([1.0, 0.0, 0.0])
    x = hint - (hint @ z) * z
    if np.linalg.norm(x) < 1e-6:
        hint = np.array([0.0, 1.0, 0.0])
        x = hint - (hint @ z) * z
    x = _unit(x)
    y = np.cross(z, x)
    return R.from_matrices(np.stack([x, y, z], axis=1)[None])[0].tolist()


def figure_mesh(voxel_size: float = 0.006, *, remesh: float | None = None, **options: Any):
    """The figure as a closed mesh: marching cubes at ``voxel_size``; with ``remesh``, an
    isotropic remesh toward edges of that length (:mod:`aether3d.sculpt.remesh`), each
    vertex then projected back onto the implicit surface (Newton steps along the field's
    gradient). Normals are the field's gradient either way."""
    from ..mesh import Mesh
    from .implicit import _check, _field, implicit_surface

    shapes = _figure_shapes(options, 2.2 * float(voxel_size))  # (a gap the voxels resolve)
    mesh = implicit_surface(shapes, voxel_size)
    if remesh is None:
        return mesh
    from .mesh import SculptMesh

    sm = SculptMesh.from_mesh(mesh)
    sm.remesh(float(remesh), iterations=5)
    out = sm.to_mesh()
    checked = [_check(sh, i, 0.0) for i, sh in enumerate(shapes)]
    p = out.positions.astype(np.float64)
    eps = 0.25 * voxel_size
    for _ in range(3):  # project onto the surface
        d = _field(checked, p)
        g = _gradient(checked, p, eps)
        g2 = np.maximum(np.sum(g * g, axis=1), 1e-12)
        p = p - (d / g2)[:, None] * g
    g = _gradient(checked, p, eps)
    n = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    return Mesh(p.astype(np.float32), out.triangles, normals=n.astype(np.float32))


def _gradient(shapes, p: np.ndarray, eps: float) -> np.ndarray:
    from .implicit import _field

    g = np.zeros_like(p)
    for axis in range(3):
        off = np.zeros(3)
        off[axis] = eps
        g[:, axis] = (_field(shapes, p + off) - _field(shapes, p - off)) / (2 * eps)
    return g


__all__ = ["FIGURE_DEFAULTS", "FINGERS", "figure_joints", "figure_mesh", "figure_shapes"]
