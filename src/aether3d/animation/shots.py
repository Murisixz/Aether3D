"""Shot design: framing a subject by a named shot size, without a viewport.

:func:`frame_shot` places a camera for one of the usual shot sizes, measured on the subject
(the object and its descendants, posed at ``time``: skins deformed, attachments carried):

======================  ==================================  ======  ====  ==========
preset                  frames (from the subject's bottom)  fills   lens  elevation
======================  ==================================  ======  ====  ==========
``extreme_wide``        all of it                           15 %    24    8 deg
``wide``                all of it                           45 %    28    5 deg
``full``                all of it                           85 %    35    3 deg
``medium``              the top 55 % (from the waist)       85 %    50    2 deg
``medium_close``        the top 40 % (from the chest)       85 %    65    1 deg
``close_up``            the top 25 % (head and shoulders)   90 %    85    0 deg
``extreme_close_up``    the top 12 %                        100 %   100   0 deg
``low_angle``           all of it                           80 %    28    -18 deg
``high_angle``          all of it                           70 %    35    35 deg
======================  ==================================  ======  ====  ==========

"Fills" is the share of the frame's height the framed part covers. The camera stands in
front of the subject (its local -Y, the convention every Aether3D asset faces), turned by
``azimuth`` degrees about the world's up and raised by ``elevation``; its distance makes the
framed part fill the frame for the camera's vertical field of view and the frame's shape.
It aims with a ``look_at`` constraint named ``Shot Aim`` (replaced if it exists) at the
middle of the framed part, placed at ``screen`` (composition: ``[-1/3, 0]`` puts the subject
on the left third). ``fstop`` turns on depth of field focused on the subject (its
``focus_object``). Everything written is ordinary, editable data -- the camera's place, its
lens and the constraint -- in one undo step; the report says what was solved.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import AnimationError

PRESETS: dict[str, dict[str, float]] = {
    "extreme_wide": {"from": 0.0, "to": 1.0, "fill": 0.15, "lens": 24.0, "elevation": 8.0},
    "wide": {"from": 0.0, "to": 1.0, "fill": 0.45, "lens": 28.0, "elevation": 5.0},
    "full": {"from": 0.0, "to": 1.0, "fill": 0.85, "lens": 35.0, "elevation": 3.0},
    "medium": {"from": 0.45, "to": 1.0, "fill": 0.85, "lens": 50.0, "elevation": 2.0},
    "medium_close": {"from": 0.6, "to": 1.0, "fill": 0.85, "lens": 65.0, "elevation": 1.0},
    "close_up": {"from": 0.75, "to": 1.0, "fill": 0.9, "lens": 85.0, "elevation": 0.0},
    "extreme_close_up": {"from": 0.88, "to": 1.0, "fill": 1.0, "lens": 100.0, "elevation": 0.0},
    "low_angle": {"from": 0.0, "to": 1.0, "fill": 0.8, "lens": 28.0, "elevation": -18.0},
    "high_angle": {"from": 0.0, "to": 1.0, "fill": 0.7, "lens": 35.0, "elevation": 35.0},
}
AIM = "Shot Aim"


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _num(value: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _err("INVALID_ARGUMENT", f"{name} must be a finite number", name)
    if not lo <= float(value) <= hi:
        raise _err("INVALID_ARGUMENT", f"{name} must be within {lo:g}..{hi:g}", name)
    return float(value)


def subject_bounds(system: Any, subject: Any, time: float) -> tuple[np.ndarray, np.ndarray]:
    """World box ``(lo, hi)`` of an object and its descendants' meshes, posed at ``time``."""
    scene = system.scene
    root = system._object(subject, "subject")
    posed = system._posed_worlds(time)
    overlay = system.overlay(time)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    stack = [root]
    while stack:
        obj = stack.pop()
        stack.extend(obj.children)
        mesh = obj.mesh if overlay is None else overlay.meshes.get(obj._id, obj.mesh)
        if mesh is None or not len(mesh.positions) or not obj._visible:
            continue
        w = posed[obj._id]
        p = np.asarray(mesh.positions, np.float64) @ w[:3, :3].T + w[:3, 3]
        lo = np.minimum(lo, p.min(axis=0))
        hi = np.maximum(hi, p.max(axis=0))
    if not np.all(np.isfinite(lo)):
        here = posed[root._id][:3, 3]
        return here.copy(), here.copy()  # (no geometry: a point)
    del scene
    return lo, hi


def frame_shot(
    system: Any,
    camera: Any,
    subject: Any,
    preset: str = "medium",
    *,
    time: float | None = None,
    azimuth: float = 0.0,
    elevation: float | None = None,
    screen: Any = None,
    fill: float | None = None,
    focal_length: float | None = None,
    fstop: float | None = None,
) -> dict[str, Any]:
    """Place ``camera`` for a shot of ``subject`` (see the module docstring)."""
    if preset not in PRESETS:
        raise _err("INVALID_ARGUMENT", f"preset is one of {', '.join(PRESETS)}", "preset")
    spec = PRESETS[preset]
    cam = system._object(camera, "camera")
    if cam.camera is None or not cam.camera.is_perspective:
        raise _err("INVALID_ARGUMENT", "frame_shot needs a perspective camera", "camera")
    subj = system._object(subject, "subject")
    t = system.start if time is None else _num(time, "time", -1e9, 1e9)
    az = _num(azimuth, "azimuth", -360.0, 360.0)
    el = spec["elevation"] if elevation is None else _num(elevation, "elevation", -89.0, 89.0)
    fl = spec["lens"] if focal_length is None else _num(focal_length, "focal_length", 1.0, 2000.0)
    share = spec["fill"] if fill is None else _num(fill, "fill", 0.01, 4.0)
    comp = [0.0, 0.0]
    if screen is not None:
        if not isinstance(screen, (list, tuple)) or len(screen) != 2:
            raise _err("INVALID_ARGUMENT", "screen is [x, y] (NDC)", "screen")
        comp = [_num(v, f"screen[{i}]", -1.0, 1.0) for i, v in enumerate(screen)]
    if fstop is not None:
        fstop = _num(fstop, "fstop", 0.1, 1e4)
    lo, hi = subject_bounds(system, subj, t)
    height = float(hi[2] - lo[2])
    centre = 0.5 * (lo + hi)
    framed_lo = lo[2] + spec["from"] * height
    framed_hi = lo[2] + spec["to"] * height
    span = max(framed_hi - framed_lo, 1e-3)
    aim = np.array([centre[0], centre[1], 0.5 * (framed_lo + framed_hi)])
    # The subject faces its local -Y: the camera stands there, turned by the azimuth.
    posed = system._posed_worlds(t)
    front = -posed[subj._id][:3, 1]
    front[2] = 0.0
    if np.linalg.norm(front) < 1e-9:
        front = np.array([0.0, -1.0, 0.0])
    front = front / np.linalg.norm(front)
    a = math.radians(az)
    heading = np.array(
        [front[0] * math.cos(a) - front[1] * math.sin(a),
         front[0] * math.sin(a) + front[1] * math.cos(a), 0.0]
    )  # fmt: skip
    e = math.radians(el)
    direction = math.cos(e) * heading + np.array([0.0, 0.0, math.sin(e)])
    # The lens: vertical field of view for the frame's shape.
    lens = cam.camera.copy()
    lens._owner = None
    lens.configure(focal_length=fl)
    r = system.scene.render_settings
    aspect = lens.aspect_ratio or r.width / r.height
    _hx, hy = lens.half_extents(aspect)
    depth = 0.5 * float(np.hypot(hi[0] - lo[0], hi[1] - lo[1]))  # half the subject's width
    distance = span / 2.0 / (share * hy) + depth
    position = aim + direction * distance
    changes: dict[str, Any] = {"focal_length": fl}
    if fstop is not None:
        changes.update(fstop=fstop, focus_object=subj._id)
    subject_origin = posed[subj._id][:3, 3]
    constraint = {
        "name": AIM,
        "type": "look_at",
        "target": {"object": subj._id},
        "target_offset": [float(v) for v in aim - subject_origin],
        "screen": comp,
    }
    with system._op(f"Frame {preset} shot of {subj.name!r}"):
        cam.camera.configure(**changes)
        parent = cam._parent
        if parent is None:
            cam.position = tuple(float(v) for v in position)
        else:
            cam.set_world_position(tuple(float(v) for v in position))
        existing = [c for c in system.constraints(cam) if c["name"] != AIM]
        system.set_constraints(cam, [*existing, constraint])
    return {
        "preset": preset,
        "camera": cam.name,
        "subject": subj.name,
        "position": [float(v) for v in position],
        "distance": float(distance),
        "focal_length": float(fl),
        "fov": float(cam.camera.fov),
        "framed": [float(framed_lo), float(framed_hi)],
        "fill": float(share),
        "elevation": float(el),
        "azimuth": float(az),
        "screen": comp,
        "depth_of_field": fstop is not None,
    }
