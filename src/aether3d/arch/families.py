"""Door and window families: parametric opening fillers built from exact boxes.

A family is a named set of parameters in the building definition::

    "Door 900": {"kind": "door", "width": "900 mm", "height": "2100 mm",
                 "frame": "50 mm", "leaf": "40 mm"},
    "Window 1200": {"kind": "window", "width": "1200 mm", "height": "1400 mm",
                    "sill": "900 mm", "frame": "60 mm", "mullions": 1, "glass": "24 mm"}

An instance is built in its opening's local frame -- x along the wall (centred on the
opening), y across the wall (from its right face to its left face), z up from the
opening's bottom -- so it follows the wall's thickness and any move of the wall or the
opening. Geometry is a multi-lump body of boxes (frames: material group 0, glass:
group 1, door leaves: group 2); instances with the same parameters and wall thickness
share one display mesh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..cad.brep.build import box
from ..cad.brep.topology import Body, merge  # noqa: F401  (merge: used here, re-exported)
from ..cad.geom3d import Frame
from ..errors import CadError

FRAME, GLASS, LEAF = 0, 1, 2
FAMILY_KINDS = ("door", "window", "opening")


@dataclass(frozen=True)
class OpeningSpec:
    kind: str
    width: float
    height: float
    sill: float
    frame: float
    leaf: float
    glass: float
    mullions: int
    transoms: int


def _box(x0, x1, y0, y1, z0, z1, name, group) -> Body:
    c = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]
    fr = Frame(np.array(c, float), np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    return box((x1 - x0, y1 - y0, z1 - z0), frame=fr, prefix=name, group=group)


def build_filler(spec: OpeningSpec, y_right: float, y_left: float) -> Body | None:
    """The door or window in the opening's local frame (None for a bare opening)."""
    w, h, f = spec.width, spec.height, spec.frame
    if spec.kind == "opening":
        return None
    depth = y_left - y_right
    if not (f > 0 and 2 * f < w and f < h):
        raise CadError("INVALID_FAMILY", "the frame is too wide for the opening")
    parts = []
    hw = w / 2
    if spec.kind == "door":
        parts.append(_box(-hw, -hw + f, y_right, y_left, 0, h, "frame/jamb_left", FRAME))
        parts.append(_box(hw - f, hw, y_right, y_left, 0, h, "frame/jamb_right", FRAME))
        parts.append(_box(-hw + f, hw - f, y_right, y_left, h - f, h, "frame/head", FRAME))
        lt = min(spec.leaf, depth * 0.8)
        yc = 0.5 * (y_right + y_left)
        parts.append(_box(-hw + f, hw - f, yc - lt / 2, yc + lt / 2, 0.0, h - f, "leaf", LEAF))
    else:
        parts.append(_box(-hw, hw, y_right, y_left, 0, f, "frame/sill", FRAME))
        parts.append(_box(-hw, hw, y_right, y_left, h - f, h, "frame/head", FRAME))
        parts.append(_box(-hw, -hw + f, y_right, y_left, f, h - f, "frame/jamb_left", FRAME))
        parts.append(_box(hw - f, hw, y_right, y_left, f, h - f, "frame/jamb_right", FRAME))
        n = max(0, spec.mullions)
        inner = w - 2 * f
        bar = min(f, inner / (2 * n + 2)) if n else 0.0
        cells = n + 1
        cell = (inner - n * bar) / cells
        gt = min(spec.glass, depth * 0.6)
        yc = 0.5 * (y_right + y_left)
        x = -hw + f
        for k in range(cells):
            parts.append(_box(x, x + cell, yc - gt / 2, yc + gt / 2, f, h - f, f"glass.{k}", GLASS))
            x += cell
            if k < n:
                parts.append(
                    _box(
                        x,
                        x + bar,
                        y_right + depth * 0.25,
                        y_left - depth * 0.25,
                        f,
                        h - f,
                        f"frame/mullion.{k}",
                        FRAME,
                    )
                )
                x += bar
    return merge(parts)


def spec_from(family: dict, overrides: dict, length, location: str) -> OpeningSpec:
    kind = family.get("kind", "opening")
    if kind not in FAMILY_KINDS:
        raise CadError("INVALID_FAMILY", f"a family's kind is one of {FAMILY_KINDS}", location)

    def get(key, default):
        raw = overrides.get(key, family.get(key, default))
        return length(raw, key)

    width = get("width", None)
    height = get("height", None)
    sill = get("sill", 0 if kind == "door" else "900 mm")
    frame = get("frame", "50 mm")
    leaf = get("leaf", "40 mm")
    glass = get("glass", "24 mm")
    mullions = int(overrides.get("mullions", family.get("mullions", 0)))
    if not (width > 0 and height > 0 and sill >= 0):
        raise CadError("INVALID_OPENING", "width and height must be positive, sill >= 0", location)
    if not 0 <= mullions <= 20:
        raise CadError("INVALID_OPENING", "mullions must be 0..20", location)
    return OpeningSpec(kind, width, height, sill, frame, leaf, glass, mullions, 0)


def door_swing(spec: OpeningSpec) -> float:
    """The door leaf's width (the swing radius drawn on plans)."""
    return spec.width - 2 * spec.frame


__all__ = ["build_filler", "door_swing", "merge", "spec_from", "math"]
