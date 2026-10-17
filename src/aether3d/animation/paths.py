"""Animatable properties: typed channel paths (no attribute traversal, ever).

A clip's channels are named by *paths* relative to what the clip is bound to. Every path
is parsed by a strict grammar and resolved against this registry; anything else is
refused (``UNKNOWN_PROPERTY``). Paths of an **object**:

=====================================  ======  =====================  ==================
path                                   kind    unit                   valid values
=====================================  ======  =====================  ==================
``transform.position.x|y|z``           scalar  scene units (m)        finite
``transform.rotation``                 rotation quaternion track      unit quaternions
``transform.rotation_euler.x|y|z``     scalar  degrees (XYZ)          finite
``transform.scale.x|y|z``              scalar  factor                 finite (0 hides)
``visible``                            bool    on above 0.5           --
``camera.fov``                         scalar  degrees                0.01 .. 179.9
``camera.ortho_scale``                 scalar  scene units            > 0
``camera.near`` / ``camera.far``       scalar  scene units            > 0, near < far
``light.intensity``                    scalar  light units            >= 0
``light.color.r|g|b``                  scalar  display sRGB           0 .. 1
``light.range`` / ``radius``/``angle`` scalar  units / degrees        >= 0
``appearance.color.r|g|b``             scalar  display sRGB           0 .. 1
``appearance.specular``                scalar  0..1                   0 .. 1
``appearance.shininess``               scalar  exponent               1 .. 10000
``joints["NAME"].position.x|y|z``      scalar  armature units         finite
``joints["NAME"].rotation``            rotation joint local           unit quaternions
``joints["NAME"].rotation_euler.x|y|z`` scalar degrees                finite
``joints["NAME"].scale.x|y|z``         scalar  factor                 finite
``morph["NAME"]``                      scalar  weight                 finite
``constraints["NAME"].influence``      scalar  0..1                   0 .. 1
=====================================  ======  =====================  ==================

Paths of a **material**: ``base_color.r|g|b``, ``opacity``, ``metallic``, ``roughness``,
``specular``, ``shininess``, ``emissive.r|g|b``, ``emissive_strength``,
``normal_strength``, ``occlusion_strength`` (the material's own ranges). Evaluated values
outside a property's range are clamped and reported (``VALUE_CLAMPED``), never written
into the scene.

Grouped paths (``transform.position``, ``transform.rotation_euler``, ``transform.scale``,
``light.color``, ``appearance.color``, ``base_color``, ``emissive`` and their joint
forms) name all their components at once in authoring calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..errors import AnimationError

SCALAR, ROTATION, BOOL = "scalar", "rotation", "bool"
OBJECT, MATERIAL = "object", "material"
MAX_NAME = 128

_INF = math.inf


@dataclass(frozen=True)
class PathSpec:
    """A resolved channel path."""

    path: str  # canonical text
    target: str  # OBJECT | MATERIAL
    kind: str  # SCALAR | ROTATION | BOOL
    group: str  # transform, visible, camera, light, appearance, joint, morph, constraint, material
    field: str  # position, rotation, rotation_euler, scale, fov, color, ...
    component: int | None  # 0..2 for vector and colour components
    name: str | None  # the joint, morph target or constraint named in brackets
    unit: str
    lo: float
    hi: float

    @property
    def is_transform(self) -> bool:
        return self.group in ("transform", "joint") and self.field in (
            "position",
            "rotation",
            "rotation_euler",
            "scale",
        )


_VEC = {"x": 0, "y": 1, "z": 2}
_RGB = {"r": 0, "g": 1, "b": 2}

# (group, field) -> (kind, components table or None, unit, lo, hi)
_OBJECT: dict[tuple[str, str], tuple[str, dict[str, int] | None, str, float, float]] = {
    ("transform", "position"): (SCALAR, _VEC, "m", -_INF, _INF),
    ("transform", "rotation"): (ROTATION, None, "quaternion", -_INF, _INF),
    ("transform", "rotation_euler"): (SCALAR, _VEC, "deg", -_INF, _INF),
    ("transform", "scale"): (SCALAR, _VEC, "1", -_INF, _INF),
    ("visible", ""): (BOOL, None, "on > 0.5", -_INF, _INF),
    ("camera", "fov"): (SCALAR, None, "deg", 0.01, 179.9),
    ("camera", "ortho_scale"): (SCALAR, None, "m", 1e-9, _INF),
    ("camera", "near"): (SCALAR, None, "m", 1e-9, _INF),
    ("camera", "far"): (SCALAR, None, "m", 1e-9, _INF),
    ("light", "intensity"): (SCALAR, None, "1", 0.0, _INF),
    ("light", "color"): (SCALAR, _RGB, "sRGB", 0.0, 1.0),
    ("light", "range"): (SCALAR, None, "m", 0.0, _INF),
    ("light", "radius"): (SCALAR, None, "m", 0.0, _INF),
    ("light", "angle"): (SCALAR, None, "deg", 0.0, 180.0),
    ("appearance", "color"): (SCALAR, _RGB, "sRGB", 0.0, 1.0),
    ("appearance", "specular"): (SCALAR, None, "1", 0.0, 1.0),
    ("appearance", "shininess"): (SCALAR, None, "1", 1.0, 10000.0),
}
_JOINT = {
    "position": (SCALAR, _VEC, "m"),
    "rotation": (ROTATION, None, "quaternion"),
    "rotation_euler": (SCALAR, _VEC, "deg"),
    "scale": (SCALAR, _VEC, "1"),
}
_MATERIAL: dict[str, tuple[str, dict[str, int] | None, str, float, float]] = {
    "base_color": (SCALAR, _RGB, "sRGB", 0.0, 1.0),
    "opacity": (SCALAR, None, "1", 0.0, 1.0),
    "metallic": (SCALAR, None, "1", 0.0, 1.0),
    "roughness": (SCALAR, None, "1", 0.0, 1.0),
    "specular": (SCALAR, None, "1", 0.0, 1.0),
    "shininess": (SCALAR, None, "1", 1.0, 10000.0),
    "emissive": (SCALAR, _RGB, "sRGB", 0.0, 1.0),
    "emissive_strength": (SCALAR, None, "1", 0.0, 1e6),
    "normal_strength": (SCALAR, None, "1", -10.0, 10.0),
    "occlusion_strength": (SCALAR, None, "1", 0.0, 1.0),
}

_TOKEN = re.compile(r'([a-z_]+)(?:\["([^"\\]{1,' + str(MAX_NAME) + r'})"\])?')


def _bad(path: Any, message: str, location: str) -> AnimationError:
    return AnimationError(
        "UNKNOWN_PROPERTY",
        f"{path!r} is not an animatable property: {message}",
        location,
        hint="see aether3d.animation.paths (or the 'properties' of an animation script's "
        "capabilities) for the supported paths",
    )


def _tokens(path: str, location: str) -> list[tuple[str, str | None]]:
    if not isinstance(path, str) or not path or len(path) > 300:
        raise _bad(path, "a path is a non-empty string", location)
    out = []
    pos = 0
    while pos < len(path):
        m = _TOKEN.match(path, pos)
        if m is None:
            raise _bad(path, f"unexpected text at {path[pos:]!r}", location)
        out.append((m.group(1), m.group(2)))
        pos = m.end()
        if pos < len(path):
            if path[pos] != ".":
                raise _bad(path, f"expected '.' at {path[pos:]!r}", location)
            pos += 1
            if pos == len(path):
                raise _bad(path, "a path cannot end with '.'", location)
    return out


def parse(
    path: str, target: str = OBJECT, *, grouped: bool = False, location: str = "path"
) -> PathSpec | list[PathSpec]:
    """Resolve ``path`` for a ``target`` kind (``object`` or ``material``).

    With ``grouped=True``, a group path (``transform.position``) returns its component
    specs (a list); a single property returns a one-element list."""
    toks = _tokens(path, location)
    specs = _resolve(path, toks, target, location)
    if isinstance(specs, list):
        if not grouped:
            raise _bad(path, "a grouped property: name a component (.x, .r, ...)", location)
        return specs
    return [specs] if grouped else specs


def _resolve(
    path: str, toks: list[tuple[str, str | None]], target: str, location: str
) -> PathSpec | list[PathSpec]:
    names = [t[0] for t in toks]
    subs = [t[1] for t in toks]
    if target == MATERIAL:
        if any(subs):
            raise _bad(path, "material paths have no [names]", location)
        field = names[0]
        if field not in _MATERIAL or len(names) > 2:
            raise _bad(path, f"materials animate {', '.join(sorted(_MATERIAL))}", location)
        kind, comps, unit, lo, hi = _MATERIAL[field]
        return _finish(
            path, MATERIAL, kind, "material", field, comps, names[1:], None, unit, lo, hi, location
        )
    if target != OBJECT:
        raise AnimationError("INVALID_TARGET", f"unknown target kind {target!r}", location)
    head, sub = toks[0]
    if head == "joints":
        if sub is None or len(names) < 2 or len(names) > 3:
            raise _bad(
                path,
                'joint paths are joints["NAME"].position|rotation|rotation_euler|scale[.x]',
                location,
            )
        field = names[1]
        if field not in _JOINT or any(subs[1:]):
            raise _bad(
                path, "joints animate position, rotation, rotation_euler and scale", location
            )
        kind, comps, unit = _JOINT[field]
        return _finish(
            path, OBJECT, kind, "joint", field, comps, names[2:], sub, unit, -_INF, _INF, location
        )
    if head in ("morph", "constraints"):
        if sub is None:
            raise _bad(path, f'{head} paths name their target: {head}["NAME"]', location)
        if head == "morph":
            if len(names) != 1:
                raise _bad(path, 'morph weights are morph["NAME"]', location)
            return PathSpec(
                _canonical(toks), OBJECT, SCALAR, "morph", "weight", None, sub, "1", -_INF, _INF
            )
        if names[1:] != ["influence"] or any(subs[1:]):
            raise _bad(path, 'constraints animate constraints["NAME"].influence', location)
        return PathSpec(
            _canonical(toks), OBJECT, SCALAR, "constraint", "influence", None, sub, "1", 0.0, 1.0
        )
    if any(subs):
        raise _bad(path, "only joints, morph and constraints take [names]", location)
    if names == ["visible"]:
        return PathSpec("visible", OBJECT, BOOL, "visible", "", None, None, "on > 0.5", -_INF, _INF)
    if len(names) < 2:
        raise _bad(path, "name a property (transform.position.x, camera.fov, ...)", location)
    key = (names[0], names[1])
    if key not in _OBJECT:
        raise _bad(path, f"unknown property {names[0]}.{names[1]}", location)
    kind, comps, unit, lo, hi = _OBJECT[key]
    return _finish(
        path, OBJECT, kind, names[0], names[1], comps, names[2:], None, unit, lo, hi, location
    )


def _finish(path, target, kind, group, field, comps, rest, name, unit, lo, hi, location):
    prefix = (f'joints["{name}"].' if group == "joint" else "") + (
        "" if group in ("material", "joint") else f"{group}."
    )
    base = f"{prefix}{field}"
    if comps is None:
        if rest:
            raise _bad(path, f"{base} has no components", location)
        return PathSpec(base, target, kind, group, field, None, name, unit, lo, hi)
    if not rest:
        return [
            PathSpec(f"{base}.{c}", target, kind, group, field, i, name, unit, lo, hi)
            for c, i in comps.items()
        ]
    if len(rest) != 1 or rest[0] not in comps:
        raise _bad(path, f"components of {base} are {', '.join(comps)}", location)
    c = rest[0]
    return PathSpec(f"{base}.{c}", target, kind, group, field, comps[c], name, unit, lo, hi)


def _canonical(toks: list[tuple[str, str | None]]) -> str:
    return ".".join(f'{n}["{s}"]' if s is not None else n for n, s in toks)


def check_name(value: Any, what: str, location: str) -> str:
    """A joint, morph target, constraint, clip or layer name."""
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_NAME:
        raise AnimationError(
            "INVALID_NAME",
            f"{what} must be a non-empty string of at most {MAX_NAME} characters",
            location,
        )
    if '"' in value or "\\" in value or any(ord(ch) < 32 for ch in value):
        raise AnimationError(
            "INVALID_NAME",
            f"{what} cannot contain quotes, backslashes or control characters",
            location,
        )
    return value


def joint_path(joint: str, field: str, component: str | None = None) -> str:
    """``joints["JOINT"].FIELD[.COMPONENT]``."""
    check_name(joint, "a joint name", "joint")
    return f'joints["{joint}"].{field}' + (f".{component}" if component else "")


def group_of(spec: PathSpec) -> str:
    """The grouped path a component belongs to (``transform.position.x`` ->
    ``transform.position``)."""
    return spec.path.rsplit(".", 1)[0] if spec.component is not None else spec.path
