"""Studio lighting: classical light setups placed on a subject, without a viewport.

:func:`light_rig` creates (or updates) ordinary area lights around a subject -- the object
and its descendants, measured posed at ``time`` -- from where the camera sees it. Each
light stands at an azimuth (degrees from the camera's side of the subject, positive to the
camera's left) and an elevation, ``distance`` from the subject's centre, aimed at it:

==============  ==========================================  ==========================
preset          lights (azimuth, elevation)                 lighting ratio (key+fill):fill
==============  ==========================================  ==========================
``three_point`` key (45, 35), fill (-50, 15), rim (160, 40)  4:1 (two stops)
``loop``        key (35, 30), fill (-40, 10), rim (150, 35)  3:1
``rembrandt``   key (50, 45), fill (-45, 10)                 8:1
``butterfly``   key (0, 45), fill (0, -15), rim (180, 45)    3:1 (fill from below)
``split``       key (90, 10)                                 no fill
``high_key``    key (30, 30), fill (-30, 10), rim (180, 45)  2:1
``low_key``     key (60, 30), fill (-60, 5), rim (150, 30)   16:1
``silhouette``  rim (150, 25), rim (-150, 25)                rims only
==============  ==========================================  ==========================

Keys and fills are square soft boxes (``0.5`` and ``0.75`` of the distance wide), rims
strips (``0.15 x 0.6``). Intensities are solved, not guessed: each light's irradiance at
the subject's centre (on a surface facing it, unoccluded, with the rasterizers' exact
polygon formula) is computed, and the intensities scaled so that the key and the fill
together give ``level`` (1: a white surface facing them appears white at exposure 0) in
the preset's ratio; rims give ``rim`` times the key. The report gives what was solved and
measured. Everything written is ordinary, editable data (light objects tagged
``light_rig``), in one undo step; calling it again updates the same lights.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..errors import RenderError

PRESETS: dict[str, dict[str, Any]] = {
    "three_point": {"key": (45, 35), "fill": (-50, 15), "rims": [(160, 40)], "ratio": 4.0},
    "loop": {"key": (35, 30), "fill": (-40, 10), "rims": [(150, 35)], "ratio": 3.0},
    "rembrandt": {"key": (50, 45), "fill": (-45, 10), "rims": [], "ratio": 8.0},
    "butterfly": {"key": (0, 45), "fill": (0, -15), "rims": [(180, 45)], "ratio": 3.0},
    "split": {"key": (90, 10), "fill": None, "rims": [], "ratio": None},
    "high_key": {"key": (30, 30), "fill": (-30, 10), "rims": [(180, 45)], "ratio": 2.0},
    "low_key": {"key": (60, 30), "fill": (-60, 5), "rims": [(150, 30)], "ratio": 16.0},
    "silhouette": {"key": None, "fill": None, "rims": [(150, 25), (-150, 25)], "ratio": None},
}
TAG = "light_rig"
SIZES = {"key": (0.5, 0.5), "fill": (0.75, 0.75), "rim": (0.15, 0.6)}  # x distance


def _err(message: str, location: str) -> RenderError:
    return RenderError("INVALID_LIGHT_RIG", message, location)


def _num(value: Any, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _err(f"{name} must be a finite number", name)
    if not lo <= float(value) <= hi:
        raise _err(f"{name} must be within {lo:g}..{hi:g}", name)
    return float(value)


def light_rig(
    scene: Any,
    subject: Any,
    preset: str = "three_point",
    *,
    camera: Any = None,
    time: float | None = None,
    distance: float | None = None,
    level: float = 1.0,
    ratio: float | None = None,
    rim: float = 1.0,
    temperature: float | None = None,
    prefix: str = "",
) -> dict[str, Any]:
    """Light ``subject`` with a classical setup (see the module docstring)."""
    from ..animation.shots import subject_bounds
    from .scene_data import RenderLight
    from .surface import area_irradiance

    if preset not in PRESETS:
        raise _err(f"preset is one of {', '.join(PRESETS)}", "preset")
    spec = PRESETS[preset]
    system = scene.animation
    subj = system._object(subject, "subject")
    t = system.start if time is None else _num(time, "time", -1e9, 1e9)
    level = _num(level, "level", 0.0, 1e6)
    rim = _num(rim, "rim", 0.0, 1e6)
    if ratio is not None:
        ratio = _num(ratio, "ratio", 1.0, 1e6)
    if temperature is not None:
        temperature = _num(temperature, "temperature", 800.0, 12000.0)
    if not isinstance(prefix, str):
        raise _err("prefix must be a string", "prefix")
    lo, hi = subject_bounds(system, subj, t)
    centre = 0.5 * (lo + hi)
    size = float(np.max(hi - lo))
    d = (1.25 * max(size, 1e-3)) if distance is None else _num(distance, "distance", 1e-4, 1e7)
    # Where the camera sees the subject from (its horizontal direction), else its front.
    cam = scene.active_camera if camera is None else scene._resolve(camera)
    if cam is not None:
        toward = np.asarray(cam.world_matrix[:3, 3], float) - centre
    else:
        toward = -np.asarray(system._posed_worlds(t)[subj._id][:3, 1], float)
    toward[2] = 0.0
    if np.linalg.norm(toward) < 1e-9:
        toward = np.array([0.0, -1.0, 0.0])
    toward /= np.linalg.norm(toward)
    left = np.cross(toward, np.array([0.0, 0.0, 1.0]))  # the camera's left (image left)

    def place(az: float, el: float) -> np.ndarray:
        a, e = math.radians(az), math.radians(el)
        direction = math.cos(e) * (math.cos(a) * toward + math.sin(a) * left)
        direction = direction + np.array([0.0, 0.0, math.sin(e)])
        return centre + d * direction

    roles: list[tuple[str, str, tuple[float, float]]] = []
    if spec["key"] is not None:
        roles.append(("key", "Key", spec["key"]))
    if spec["fill"] is not None:
        roles.append(("fill", "Fill", spec["fill"]))
    for k, angles in enumerate(spec["rims"]):
        roles.append(("rim", "Rim" if len(spec["rims"]) == 1 else f"Rim {k + 1}", angles))

    # Irradiance of each light at the centre per unit intensity (aether units: radiance
    # pi I / A), on a surface facing it, by the polygon formula: E = G pi / A.
    unit = []
    for role, _name, (az, el) in roles:
        pos = place(az, el)
        w, h = SIZES[role][0] * d, SIZES[role][1] * d
        normal = (centre - pos) / np.linalg.norm(centre - pos)
        # The light's local X is horizontal; its local Z completes the frame.
        ax = np.cross(normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(ax) < 1e-9:
            ax = np.cross(normal, toward)
        ax /= np.linalg.norm(ax)
        ay = np.cross(ax, normal)
        light = RenderLight(None, "", "area", np.ones(3), pos, normal, None, "inverse_square",
                            axis_x=ax * w / 2, axis_y=ay * h / 2, shape="rectangle",
                            area=w * h)  # fmt: skip
        n = -normal
        g = area_irradiance(light, *(np.array([c]) for c in centre), *(np.array([c]) for c in n))[0]
        unit.append(float(g) * math.pi / (w * h))  # irradiance per unit intensity
    r = spec["ratio"] if ratio is None else ratio
    targets = {}
    if spec["key"] is not None:
        key_share = level if spec["fill"] is None or r is None else level * (1.0 - 1.0 / r)
        targets["key"] = key_share
    if spec["fill"] is not None and r is not None:
        targets["fill"] = level / r
    key_target = targets.get("key", level)
    created = []
    with scene.transaction(f"Light rig {preset!r} on {subj.name!r}"):
        for (role, name, (az, el)), e1 in zip(roles, unit, strict=True):
            want = targets.get(role, rim * key_target)  # irradiance (base colours) at the centre
            intensity = want * math.pi / max(e1, 1e-300)  # E = pi x (radiance factor)
            pos = place(az, el)
            w, h = SIZES[role][0] * d, SIZES[role][1] * d
            full = f"{prefix}{name}"
            params = {"intensity": intensity, "shape": "rectangle", "size": w, "size_y": h}
            if temperature is not None:
                params.update(temperature=temperature, use_temperature=True)
            existing = scene._by_name.get(full)
            if existing is not None and existing.light is not None and TAG in existing.tags:
                existing.light.configure(type="area", **params)
                existing.position = tuple(float(v) for v in pos)
                existing.look_at(tuple(float(v) for v in centre))
                obj = existing
            else:
                obj = scene.add_area_light(full, position=tuple(float(v) for v in pos),
                                           look_at=tuple(float(v) for v in centre),
                                           **params)  # fmt: skip
                obj.tags = (*obj.tags, TAG)
            created.append(
                {
                    "name": obj.name,
                    "id": obj.id,
                    "role": role,
                    "azimuth": float(az),
                    "elevation": float(el),
                    "distance": float(d),
                    "size": [float(w), float(h)],
                    "intensity": float(intensity),
                    "irradiance": float(want),  # at the centre, facing the light (pi = white)
                }
            )
    out = {
        "preset": preset,
        "subject": subj.name,
        "centre": [float(v) for v in centre],
        "distance": float(d),
        "lights": created,
        "level": level,
    }
    if "key" in targets and "fill" in targets:
        out["ratio"] = float((targets["key"] + targets["fill"]) / targets["fill"])
    return out
