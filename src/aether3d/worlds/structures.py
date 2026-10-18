"""Structures and settlements: houses, towers, castles, ruins, wells and bridges; villages
and the roads between them.

A world's structures are **authored items** of its definition (``structures.items``), so
they are saved, undone and redone like everything else and survive regeneration:

* ``{"id", "kind", "at": [x, y], "rotation", "size": [width, depth], "floors", "style",
  "seed"}`` for ``house``, ``tower``, ``castle``, ``ruin`` and ``well`` (``size`` and
  ``floors`` scale the kind's proportions; ``style`` is ``timber``, ``stone`` or ``brick``);
* ``{"id", "kind": "bridge", "line": [[x, y], [x, y]], "width"}``.

**Foundations.** Every item but a bridge implies a ``pad`` (its footprint, a rotated
rectangle, flattened with a falloff) whose height is resolved when the world is
evaluated: the median of the *natural* ground -- the terrain before final features -- at
nine points of the footprint. A structure therefore stays on the ground whatever changes
around it (a new seed, other features, erosion settings); its plinth reaches below the
pad. Roads given as ``[x, y]`` lines are resolved the same way: densified and given the
natural ground's heights, smoothed, so they cut a level bed across slopes. Bridges stand
on the natural ground at their ends, with piers down to the bed.

**Settlements** (:func:`find_sites`, :func:`plan_settlement`, :func:`plan_road`) are
computed from the definition and the natural terrain, then *frozen* into items and road
features by one operation (``World.build_settlements``): a village is a plaza with a well,
one or two streets along the contour (road features) and houses facing them, never on
water, too steep ground or each other; roads between villages follow the terrain (A* on a
grid with a grade penalty) and cross rivers on bridges.

Meshes are procedural and original (no imported assets): plinth, walls, timber frames,
gable roofs, chimneys, doors and windows for houses; towers with crenellations or a
conical roof; castles with curtain walls, corner towers, a gatehouse and a keep. Their
face groups map to the world's material layers ``stone``, ``plaster``, ``timber``,
``roof`` and ``wood``; vertex colours vary each building a little. Block worlds stamp the
same structures in blocks (stone bricks, cobblestone, planks, glass, logs).
"""

from __future__ import annotations

import heapq
import math
from functools import lru_cache
from typing import Any

import numpy as np

from ..errors import WorldError
from ..mesh import Mesh
from . import noise as nz

KINDS = ("house", "tower", "castle", "ruin", "well", "bridge")
STYLES = ("timber", "stone", "brick")
MAX_ITEMS = 4096
PAD_FALLOFF = 3.0
"""Metres over which a structure's pad blends into the ground."""
LAYERS = {0: "stone", 1: "plaster", 2: "timber", 3: "roof", 4: "wood"}
STONE, PLASTER, TIMBER, ROOF, WOOD = range(5)
_DEFAULT_SIZE = {
    "house": (7.0, 5.5),
    "tower": (7.0, 7.0),
    "castle": (40.0, 40.0),
    "ruin": (7.0, 5.0),
    "well": (2.4, 2.4),
}


def _err(message: str, location: str, code: str = "INVALID_DEFINITION", hint: str | None = None):
    return WorldError(code, message, location, hint=hint)


# --------------------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------------------- #


def check_item(v: Any, loc: str) -> dict:
    """One structure item (see the module docstring), completed with defaults."""
    from .definition import FEATURE_ID, MAX_COORDINATE, _number, _vec

    if not isinstance(v, dict):
        raise _err("expected a structure object", loc)
    kind = v.get("kind")
    if kind not in KINDS:
        raise _err(
            f"unknown structure kind {kind!r}", f"{loc}.kind", hint="kinds: " + ", ".join(KINDS)
        )
    sid = v.get("id")
    if not isinstance(sid, str) or not FEATURE_ID.match(sid):
        raise _err("every structure needs an id (letters, digits, _ . -, at most 64)", f"{loc}.id")
    if kind == "bridge":
        allowed = {"id", "kind", "line", "width"}
    else:
        allowed = {"id", "kind", "at", "rotation", "size", "floors", "style", "seed"}
    for key in v:
        if key not in allowed:
            raise _err(
                f"unknown key {key!r}", f"{loc}.{key}", hint="keys: " + ", ".join(sorted(allowed))
            )
    out: dict[str, Any] = {"id": sid, "kind": kind}
    if kind == "bridge":
        line = v.get("line")
        if not isinstance(line, (list, tuple)) or len(line) != 2:
            raise _err("a bridge is a line of two [x, y] points", f"{loc}.line")
        out["line"] = [
            _vec(p, 2, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}.line[{i}]")
            for i, p in enumerate(line)
        ]
        span = math.dist(*out["line"])
        if not 2.0 <= span <= 400.0:
            raise _err(f"a bridge spans 2 to 400 m, not {span:.1f}", f"{loc}.line")
        out["width"] = _number(v.get("width", 4.0), 1.0, 30.0, f"{loc}.width")
        return out
    if "at" not in v:
        raise _err("a structure needs a position: at [x, y]", f"{loc}.at")
    out["at"] = _vec(v["at"], 2, -MAX_COORDINATE, MAX_COORDINATE, f"{loc}.at")
    out["rotation"] = _number(v.get("rotation", 0.0), -360.0, 360.0, f"{loc}.rotation")
    size = v.get("size", list(_DEFAULT_SIZE[kind]))
    out["size"] = _vec(size, 2, 1.0, 200.0, f"{loc}.size")
    out["floors"] = _number(v.get("floors", 1), 1, 6, f"{loc}.floors", integer=True)
    style = v.get("style", "stone" if kind in ("tower", "castle", "ruin", "well") else "timber")
    if style not in STYLES:
        raise _err(f"unknown style {style!r}", f"{loc}.style", hint="styles: " + ", ".join(STYLES))
    out["style"] = style
    out["seed"] = _number(v.get("seed", 0), 0, 2**31 - 1, f"{loc}.seed", integer=True)
    return out


def check_items(v: Any, loc: str) -> list[dict]:
    if not isinstance(v, (list, tuple)) or len(v) > MAX_ITEMS:
        raise _err(f"expected a list of at most {MAX_ITEMS} structures", loc)
    out = [check_item(x, f"{loc}[{i}]") for i, x in enumerate(v)]
    seen: set[str] = set()
    for i, item in enumerate(out):
        if item["id"] in seen:
            raise _err(f"duplicate structure id {item['id']!r}", f"{loc}[{i}].id")
        seen.add(item["id"])
    return out


def footprint(item: dict, grow: float = 0.0) -> list[list[float]]:
    """The four corners ``[x, y]`` of an item's footprint (counter-clockwise)."""
    w, d = item["size"]
    if item["kind"] == "castle":
        w, d = w + 8.0, d + 8.0  # the corner towers stand out
    elif item["kind"] == "tower":
        w = d = max(w, d)
    c, s = nz.cos_sin_deg(item["rotation"])
    hw, hd = w / 2.0 + grow, d / 2.0 + grow
    x, y = item["at"]
    return [
        [x + c * a - s * b, y + s * a + c * b]
        for a, b in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd))
    ]


def pad_features(items: list[dict]) -> list[dict]:
    """The implied foundation pads (``structure:<id>``, automatic heights)."""
    out = []
    for item in items:
        if item["kind"] == "bridge":
            continue
        out.append(
            {
                "type": "pad",
                "id": f"structure:{item['id']}",
                "shape": {"polygon": footprint(item, 1.0)},
                "enabled": True,
                "height": None,
                "falloff": PAD_FALLOFF,
                "structure": None,
            }
        )
    return out


def auto_height(model: Any, item: dict) -> float:
    """The pad height of an item: the median natural ground at nine footprint points."""
    pts = np.asarray(footprint(item, -0.5))
    centre = pts.mean(axis=0)
    samples = np.concatenate([pts, (pts + centre) / 2.0, centre[None]])
    h = model.natural(samples[:, 0], samples[:, 1])
    return float(np.median(h))


# --------------------------------------------------------------------------------------- #
# Meshes
# --------------------------------------------------------------------------------------- #


class _Build:
    def __init__(self) -> None:
        self.p: list[np.ndarray] = []
        self.n: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.t: list[np.ndarray] = []
        self.g: list[np.ndarray] = []
        self.count = 0

    def quad(self, a, b, c, d, group: int, color) -> None:
        """A planar quad a-b-c-d (counter-clockwise seen from its front)."""
        pts = np.array([a, b, c, d], dtype=np.float64)
        n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        self.p.append(pts)
        self.n.append(np.repeat(n[None], 4, axis=0))
        self.c.append(np.repeat(np.asarray(color, dtype=np.float64)[None], 4, axis=0))
        self.t.append(np.array([[0, 1, 2], [0, 2, 3]]) + self.count)
        self.g.append(np.array([group, group]))
        self.count += 4

    def tri(self, a, b, c, group: int, color) -> None:
        pts = np.array([a, b, c], dtype=np.float64)
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        self.p.append(pts)
        self.n.append(np.repeat(n[None], 3, axis=0))
        self.c.append(np.repeat(np.asarray(color, dtype=np.float64)[None], 3, axis=0))
        self.t.append(np.array([[0, 1, 2]]) + self.count)
        self.g.append(np.array([group]))
        self.count += 3

    def box(self, x0, y0, z0, x1, y1, z1, group: int, color, bottom: bool = False) -> None:
        c = color
        self.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), group, c)  # top
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), group, c)  # -y
        self.quad((x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), group, c)  # +y
        self.quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1), group, c)  # +x
        self.quad((x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), group, c)  # -x
        if bottom:
            self.quad((x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0), group, c)

    def prism(self, cx, cy, z0, z1, r, sides, group: int, color, top: bool = True) -> None:
        a = [2 * math.pi * (k + 0.5) / sides for k in range(sides)]
        ring = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in a]
        for k in range(sides):
            (xa, ya), (xb, yb) = ring[k], ring[(k + 1) % sides]
            self.quad((xa, ya, z0), (xb, yb, z0), (xb, yb, z1), (xa, ya, z1), group, color)
        if top:
            for k in range(sides):
                (xa, ya), (xb, yb) = ring[k], ring[(k + 1) % sides]
                self.tri((cx, cy, z1), (xa, ya, z1), (xb, yb, z1), group, color)

    def cone(self, cx, cy, z0, height, r, sides, group: int, color) -> None:
        a = [2 * math.pi * (k + 0.5) / sides for k in range(sides)]
        ring = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in a]
        for k in range(sides):
            (xa, ya), (xb, yb) = ring[k], ring[(k + 1) % sides]
            self.tri((xa, ya, z0), (xb, yb, z0), (cx, cy, z0 + height), group, color)

    def mesh(self) -> Mesh:
        p = np.concatenate(self.p)
        c = np.clip(np.concatenate(self.c), 0.0, 1.0)
        return Mesh._from_arrays(
            p.astype(np.float32),
            np.concatenate(self.t),
            np.concatenate(self.n).astype(np.float32),
            None,
            np.concatenate(self.g).astype(np.int64),
            np.concatenate([c, np.ones((len(c), 1))], axis=1).astype(np.float32),
        )


def _rnd(item: dict) -> Any:
    seed = nz.derive(int(item["seed"]), "structure", item["kind"])

    def u(k: int, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * float(nz.random2(seed, np.array([k]), np.array([0]))[0])

    return u


def _tone(u: Any, k: int, amount: float = 0.08) -> np.ndarray:
    return np.array([1.0 + u(k * 3 + j, -amount, amount) for j in range(3)])


def _crenellations(b: _Build, x0, y0, x1, y1, z, group, color, merlon=1.0, gap=1.0) -> None:
    """Merlons along the four edges of a rectangle at height ``z``."""
    step = merlon + gap
    for (ax, ay), (bx, by) in (
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ):
        length = math.hypot(bx - ax, by - ay)
        n = max(1, int(length // step))
        dx, dy = (bx - ax) / length, (by - ay) / length
        for k in range(n):
            t0 = k * step + gap / 2
            px, py = ax + dx * t0, ay + dy * t0
            qx, qy = px + dx * merlon, py + dy * merlon
            lo_x, hi_x = min(px, qx) - 0.3 * abs(dy), max(px, qx) + 0.3 * abs(dy)
            lo_y, hi_y = min(py, qy) - 0.3 * abs(dx), max(py, qy) + 0.3 * abs(dx)
            b.box(lo_x, lo_y, z, hi_x, hi_y, z + 1.0, group, color)


def _house(b: _Build, item: dict, ruin: bool = False) -> None:
    u = _rnd(item)
    w, d = item["size"]
    floors = item["floors"]
    style = item["style"]
    hw, hd = w / 2.0, d / 2.0
    wall_h = 2.9 * floors
    tone = _tone(u, 1)
    b.box(-hw - 0.2, -hd - 0.2, -2.0, hw + 0.2, hd + 0.2, 0.55, STONE, 0.85 * tone)  # plinth
    if ruin:
        # Broken walls: segments of random heights, no roof.
        for k, (ax, ay, bx, by) in enumerate(
            ((-hw, -hd, hw, -hd), (hw, -hd, hw, hd), (hw, hd, -hw, hd), (-hw, hd, -hw, -hd))
        ):
            n = 4
            for j in range(n):
                t0, t1 = j / n, (j + 1) / n
                top = 0.55 + wall_h * u(10 + 4 * k + j, 0.15, 1.0)
                x0, y0 = ax + (bx - ax) * t0, ay + (by - ay) * t0
                x1, y1 = ax + (bx - ax) * t1, ay + (by - ay) * t1
                b.box(
                    min(x0, x1) - 0.3,
                    min(y0, y1) - 0.3,
                    0.55,
                    max(x0, x1) + 0.3,
                    max(y0, y1) + 0.3,
                    top,
                    STONE,
                    0.75 * tone,
                )
        return
    wall_group = PLASTER if style == "timber" else STONE
    wall_color = tone * (0.95 if style != "brick" else 1.0)
    if style == "brick":
        wall_color = np.array([0.62, 0.32, 0.24]) * tone / np.array([0.6, 0.58, 0.54])
    b.box(-hw, -hd, 0.55, hw, hd, 0.55 + wall_h, wall_group, wall_color)
    if style == "timber":  # a frame of beams on the facades
        beam = 0.18
        for z in [0.55 + 2.9 * f for f in range(floors + 1)]:
            b.box(
                -hw - 0.05,
                -hd - 0.05,
                z - beam / 2,
                hw + 0.05,
                hd + 0.05,
                z + beam / 2,
                TIMBER,
                tone,
            )
        for x in np.linspace(-hw, hw, max(3, int(w // 1.8))):
            b.box(
                x - beam / 2,
                -hd - 0.06,
                0.55,
                x + beam / 2,
                -hd + 0.02,
                0.55 + wall_h,
                TIMBER,
                tone,
            )
            b.box(
                x - beam / 2, hd - 0.02, 0.55, x + beam / 2, hd + 0.06, 0.55 + wall_h, TIMBER, tone
            )
    # Door and windows (dark insets) on the front (-y) and back.
    dark = np.array([0.18, 0.15, 0.13])
    b.box(-0.55, -hd - 0.08, 0.55, 0.55, -hd + 0.02, 2.65, WOOD, np.array([0.45, 0.3, 0.18]))
    for f in range(floors):
        z0 = 1.45 + 2.9 * f
        for x in np.linspace(-hw + 1.2, hw - 1.2, max(2, int(w // 2.6))):
            if f == 0 and abs(x) < 1.2:
                continue
            b.box(x - 0.45, -hd - 0.07, z0, x + 0.45, -hd + 0.02, z0 + 1.0, WOOD, dark)
            b.box(x - 0.45, hd - 0.02, z0, x + 0.45, hd + 0.07, z0 + 1.0, WOOD, dark)
    # Gable roof along x: ridge over y = 0.
    over = 0.45
    z_eave = 0.55 + wall_h
    rise = (hd + over) * 0.85
    roof = np.array([0.95, 0.9, 0.9]) * _tone(u, 7, 0.1)
    xa, xb = -hw - over, hw + over
    b.quad(
        (xa, -hd - over, z_eave - 0.3),
        (xb, -hd - over, z_eave - 0.3),
        (xb, 0.0, z_eave + rise),
        (xa, 0.0, z_eave + rise),
        ROOF,
        roof,
    )
    b.quad(
        (xb, hd + over, z_eave - 0.3),
        (xa, hd + over, z_eave - 0.3),
        (xa, 0.0, z_eave + rise),
        (xb, 0.0, z_eave + rise),
        ROOF,
        roof,
    )
    # Undersides (seen from below the eaves) and gables.
    b.quad(
        (xa, 0.0, z_eave + rise - 0.2),
        (xb, 0.0, z_eave + rise - 0.2),
        (xb, -hd - over, z_eave - 0.5),
        (xa, -hd - over, z_eave - 0.5),
        TIMBER,
        tone * 0.6,
    )
    b.quad(
        (xb, 0.0, z_eave + rise - 0.2),
        (xa, 0.0, z_eave + rise - 0.2),
        (xa, hd + over, z_eave - 0.5),
        (xb, hd + over, z_eave - 0.5),
        TIMBER,
        tone * 0.6,
    )
    for sx in (-1.0, 1.0):
        x = sx * hw
        a, c = (x, -hd, z_eave), (x, hd, z_eave)
        top = (x, 0.0, z_eave + rise - 0.25)
        if sx > 0:
            b.tri(a, c, top, wall_group, wall_color)
        else:
            b.tri(c, a, top, wall_group, wall_color)
    # Chimney.
    cxp = hw * u(20, -0.5, 0.5)
    b.box(cxp - 0.35, 0.6, z_eave, cxp + 0.35, 1.3, z_eave + rise + 1.0, STONE, 0.8 * tone)


def _tower(b: _Build, item: dict) -> None:
    u = _rnd(item)
    w, d = item["size"]
    r = max(w, d) / 2.0
    height = 10.0 + 4.0 * item["floors"] + u(1, 0.0, 3.0)
    tone = _tone(u, 2)
    b.prism(0.0, 0.0, -2.0, height, r, 8, STONE, 0.85 * tone, top=True)
    # Arrow slits.
    for k in range(4):
        a = math.pi / 2 * k + math.pi / 8
        for z in np.arange(3.0, height - 2.0, 3.5):
            x, y = (r + 0.02) * math.cos(a), (r + 0.02) * math.sin(a)
            tx, ty = -math.sin(a), math.cos(a)
            b.quad(
                (x - 0.12 * tx, y - 0.12 * ty, z),
                (x + 0.12 * tx, y + 0.12 * ty, z),
                (x + 0.12 * tx, y + 0.12 * ty, z + 1.1),
                (x - 0.12 * tx, y - 0.12 * ty, z + 1.1),
                WOOD,
                np.array([0.12, 0.1, 0.09]),
            )
    if u(3) < 0.5:
        b.cone(0.0, 0.0, height, r * 1.3, r * 1.25, 8, ROOF, _tone(u, 4))
    else:
        b.prism(0.0, 0.0, height, height + 0.4, r + 0.4, 8, STONE, 0.8 * tone)
        for k in range(8):
            a = 2 * math.pi * k / 8
            x, y = (r - 0.1) * math.cos(a), (r - 0.1) * math.sin(a)
            b.box(
                x - 0.45,
                y - 0.45,
                height + 0.4,
                x + 0.45,
                y + 0.45,
                height + 1.5,
                STONE,
                0.8 * tone,
            )


def _castle(b: _Build, item: dict) -> None:
    u = _rnd(item)
    w, d = item["size"]
    hw, hd = w / 2.0, d / 2.0
    tone = _tone(u, 3)
    stone = 0.82 * tone
    wall_h, thick = 9.0 + 2.0 * (item["floors"] - 1), 2.4
    # Curtain walls (the front, -y, has a gate).
    gate = 2.4
    b.box(-hw, hd - thick, -2.0, hw, hd, wall_h, STONE, stone)
    b.box(-hw, -hd, -2.0, -hw + thick, hd, wall_h, STONE, stone)
    b.box(hw - thick, -hd, -2.0, hw, hd, wall_h, STONE, stone)
    b.box(-hw, -hd, -2.0, -gate, -hd + thick, wall_h, STONE, stone)
    b.box(gate, -hd, -2.0, hw, -hd + thick, wall_h, STONE, stone)
    b.box(-gate, -hd, 4.6, gate, -hd + thick, wall_h, STONE, stone)  # over the gate
    b.box(
        -gate + 0.1, -hd + 0.4, 0.0, gate - 0.1, -hd + 0.6, 4.6, WOOD, np.array([0.35, 0.24, 0.14])
    )
    _crenellations(b, -hw, -hd, hw, hd, wall_h, STONE, stone)
    # Gatehouse towers and corner towers.
    for x, y, r, h in (
        (-hw, -hd, 4.0, wall_h + 5),
        (hw, -hd, 4.0, wall_h + 5),
        (-hw, hd, 4.0, wall_h + 5),
        (hw, hd, 4.0, wall_h + 5),
        (-gate - 2.0, -hd, 2.4, wall_h + 3),
        (gate + 2.0, -hd, 2.4, wall_h + 3),
    ):
        b.prism(x, y, -2.0, h, r, 10, STONE, stone * 0.97)
        if r > 3.0:
            b.cone(x, y, h, r * 1.2, r * 1.2, 10, ROOF, _tone(u, 5))
        else:
            _crenellations(b, x - r, y - r, x + r, y + r, h, STONE, stone, 0.8, 0.8)
    # Keep.
    kw, kd = min(14.0, w * 0.35), min(12.0, d * 0.3)
    ky = hd * 0.25
    kh = wall_h + 9.0
    b.box(-kw / 2, ky - kd / 2, -2.0, kw / 2, ky + kd / 2, kh, STONE, stone * 0.94)
    _crenellations(b, -kw / 2, ky - kd / 2, kw / 2, ky + kd / 2, kh, STONE, stone)
    dark = np.array([0.12, 0.1, 0.09])
    for z in (wall_h - 2.0, wall_h + 3.0):
        for x in (-kw / 4, kw / 4):
            b.box(x - 0.3, ky - kd / 2 - 0.06, z, x + 0.3, ky - kd / 2 + 0.02, z + 1.4, WOOD, dark)
    # Courtyard ground (packed earth).
    b.quad(
        (-hw + thick, -hd + thick, 0.05),
        (hw - thick, -hd + thick, 0.05),
        (hw - thick, hd - thick, 0.05),
        (-hw + thick, hd - thick, 0.05),
        WOOD,
        np.array([0.38, 0.32, 0.24]),
    )


def _well(b: _Build, item: dict) -> None:
    u = _rnd(item)
    r = max(item["size"]) / 2.0
    tone = _tone(u, 1)
    b.prism(0.0, 0.0, -1.0, 0.9, r, 10, STONE, 0.85 * tone, top=False)
    b.prism(0.0, 0.0, -1.0, 0.75, r * 0.72, 10, STONE, 0.5 * tone, top=False)
    b.quad(
        (-r * 0.7, -r * 0.7, 0.2),
        (r * 0.7, -r * 0.7, 0.2),
        (r * 0.7, r * 0.7, 0.2),
        (-r * 0.7, r * 0.7, 0.2),
        WOOD,
        np.array([0.1, 0.16, 0.2]),
    )
    for x in (-r * 0.85, r * 0.85):
        b.box(x - 0.1, -0.1, 0.9, x + 0.1, 0.1, 2.6, TIMBER, tone)
    b.quad(
        (-r - 0.3, -r * 0.8, 2.3),
        (r + 0.3, -r * 0.8, 2.3),
        (r + 0.3, 0.0, 3.1),
        (-r - 0.3, 0.0, 3.1),
        ROOF,
        tone,
    )
    b.quad(
        (r + 0.3, r * 0.8, 2.3),
        (-r - 0.3, r * 0.8, 2.3),
        (-r - 0.3, 0.0, 3.1),
        (r + 0.3, 0.0, 3.1),
        ROOF,
        tone,
    )


@lru_cache(maxsize=1024)
def _mesh_cached(key: str) -> Mesh:
    import json

    item = json.loads(key)
    b = _Build()
    kind = item["kind"]
    if kind == "house":
        _house(b, item)
    elif kind == "ruin":
        _house(b, item, ruin=True)
    elif kind == "tower":
        _tower(b, item)
    elif kind == "castle":
        _castle(b, item)
    elif kind == "well":
        _well(b, item)
    return b.mesh()


def structure_mesh(item: dict) -> Mesh:
    """The mesh of a structure in its local frame (origin at its pad, +z up; rotated and
    placed by its matrix). Cached by everything but its position and rotation."""
    import json

    shape = {k: item[k] for k in ("kind", "size", "floors", "style", "seed")}
    return _mesh_cached(json.dumps(shape, sort_keys=True))


def bridge_mesh(item: dict, z0: float, z1: float, bed: np.ndarray) -> tuple[Mesh, np.ndarray]:
    """A bridge deck from end to end (heights ``z0``, ``z1``) with railings and piers down
    to the ground below (``bed``: heights at the piers); world-space, its origin first."""
    (ax, ay), (bx, by) = item["line"]
    length = math.hypot(bx - ax, by - ay)
    dx, dy = (bx - ax) / length, (by - ay) / length
    nx, ny = -dy, dx
    hw = item["width"] / 2.0
    b = _Build()
    wood = np.array([0.62, 0.48, 0.34])
    segs = max(2, int(length // 4.0))
    arch = min(2.0, length * 0.06)
    for k in range(segs):
        t0, t1 = k / segs, (k + 1) / segs
        za = z0 + (z1 - z0) * t0 + arch * 4 * t0 * (1 - t0)
        zb = z0 + (z1 - z0) * t1 + arch * 4 * t1 * (1 - t1)
        p0 = (dx * length * t0, dy * length * t0)
        p1 = (dx * length * t1, dy * length * t1)
        corners = [
            (p0[0] - nx * hw, p0[1] - ny * hw, za),
            (p1[0] - nx * hw, p1[1] - ny * hw, zb),
            (p1[0] + nx * hw, p1[1] + ny * hw, zb),
            (p0[0] + nx * hw, p0[1] + ny * hw, za),
        ]
        b.quad(*corners, WOOD, wood)
        low = [(x, y, z - 0.45) for x, y, z in corners]
        b.quad(low[3], low[2], low[1], low[0], TIMBER, wood * 0.7)
        b.quad(low[0], low[1], corners[1], corners[0], TIMBER, wood * 0.8)
        b.quad(low[2], low[3], corners[3], corners[2], TIMBER, wood * 0.8)
        for side in (-1.0, 1.0):  # railings
            s0 = (p0[0] + side * nx * (hw - 0.1), p0[1] + side * ny * (hw - 0.1))
            s1 = (p1[0] + side * nx * (hw - 0.1), p1[1] + side * ny * (hw - 0.1))
            b.quad(
                (s0[0], s0[1], za + 0.95),
                (s1[0], s1[1], zb + 0.95),
                (s1[0], s1[1], zb + 1.05),
                (s0[0], s0[1], za + 1.05),
                TIMBER,
                wood * 0.75,
            )
            b.quad(
                (s1[0], s1[1], zb + 0.95),
                (s0[0], s0[1], za + 0.95),
                (s0[0], s0[1], za + 1.05),
                (s1[0], s1[1], zb + 1.05),
                TIMBER,
                wood * 0.75,
            )
            b.box(
                s0[0] - 0.08,
                s0[1] - 0.08,
                za,
                s0[0] + 0.08,
                s0[1] + 0.08,
                za + 1.05,
                TIMBER,
                wood * 0.7,
            )
    for k, zb in enumerate(bed):
        t = (k + 1) / (len(bed) + 1)
        px, py = dx * length * t, dy * length * t
        top = z0 + (z1 - z0) * t + arch * 4 * t * (1 - t) - 0.45
        if top - zb > 0.5:
            b.box(
                px - 0.5,
                py - 0.5,
                zb - 1.0,
                px + 0.5,
                py + 0.5,
                top,
                STONE,
                np.array([0.8, 0.8, 0.8]),
            )
    return b.mesh(), np.array([ax, ay, 0.0])


# --------------------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------------------- #


@lru_cache(maxsize=512)
def _cells_cached(key: str) -> tuple:
    """Block cells ``(x, y, z, block)`` of a structure in its local frame (rotation in
    quarter turns applied by the caller), for block worlds."""
    import json

    item = json.loads(key)
    kind = item["kind"]
    w, d = (max(2, int(round(v))) for v in item["size"])
    floors = item["floors"]
    out: list[tuple[int, int, int, str]] = []

    def box(x0, y0, z0, x1, y1, z1, block, hollow=False):
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    if hollow and x0 < x < x1 and y0 < y < y1:
                        continue
                    out.append((x, y, z, block))

    hw, hd = w // 2, d // 2
    if kind in ("house", "ruin"):
        base = "cobblestone" if kind == "house" else "mossy_stone_bricks"
        wall = {"timber": "oak_planks", "stone": "stone_bricks", "brick": "bricks"}[item["style"]]
        if kind == "ruin":
            wall = "mossy_stone_bricks"
        h = 4 * floors
        box(-hw, -hd, -3, hw, hd, -1, "cobblestone")
        box(-hw, -hd, 0, hw, hd, 0, base, hollow=True)
        box(-hw, -hd, 1, hw, hd, h, wall, hollow=True)
        if kind == "house":
            for x in (-hw, hw):
                for y in (-hd, hd):
                    box(x, y, 1, x, y, h, "oak_log")
            for z in range(2, h, 4):
                for x in range(-hw + 2, hw - 1, 3):
                    out.append((x, -hd, z, "glass"))
                    out.append((x, hd, z, "glass"))
            out += [(0, -hd, 1, "air"), (0, -hd, 2, "air")]
            for k in range(hd + 2):  # stepped gable roof along x
                box(
                    -hw - 1, -hd - 1 + k, h + 1 + k, hw + 1, -hd - 1 + k, h + 1 + k, "spruce_planks"
                )
                box(-hw - 1, hd + 1 - k, h + 1 + k, hw + 1, hd + 1 - k, h + 1 + k, "spruce_planks")
            out.append((0, 0, h - 1, "lantern_block"))
        else:
            seed = int(item["seed"])
            out = [
                c for c in out if c[2] <= 1 or ((c[0] * 7 + c[1] * 13 + seed) % 5 + 2) * 1.0 > c[2]
            ]
    elif kind == "tower":
        r = max(w, d) // 2
        h = 10 + 4 * floors
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if x * x + y * y <= r * r + r:
                    inner = x * x + y * y < (r - 1) * (r - 1)
                    for z in range(-3, h + 1):
                        if not inner or z <= 0 or z == h:
                            out.append((x, y, z, "stone_bricks"))
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                ring = (r - 1) * (r - 1) <= x * x + y * y <= r * r + r
                if ring and (x + y) % 2 == 0:
                    out.append((x, y, h + 1, "stone_bricks"))
    elif kind == "castle":
        h = 9 + 2 * (floors - 1)
        box(-hw, -hd, -3, hw, hd, h, "stone_bricks", hollow=True)
        for x in range(-hw, hw + 1, 2):
            out += [(x, -hd, h + 1, "stone_bricks"), (x, hd, h + 1, "stone_bricks")]
        for y in range(-hd, hd + 1, 2):
            out += [(-hw, y, h + 1, "stone_bricks"), (hw, y, h + 1, "stone_bricks")]
        for z in range(1, 6):
            for x in range(-2, 3):
                out.append((x, -hd, z, "air"))
        for cx, cy in ((-hw, -hd), (hw, -hd), (-hw, hd), (hw, hd)):
            box(cx - 3, cy - 3, -3, cx + 3, cy + 3, h + 5, "stone_bricks", hollow=True)
            for x in range(cx - 3, cx + 4, 2):
                out += [(x, cy - 3, h + 6, "stone_bricks"), (x, cy + 3, h + 6, "stone_bricks")]
        kw, kd = max(4, w // 6), max(4, d // 6)
        box(-kw, -kd + hd // 3, -3, kw, kd + hd // 3, h + 9, "stone_bricks", hollow=True)
        for x in range(-kw, kw + 1, 2):
            out += [
                (x, -kd + hd // 3, h + 10, "stone_bricks"),
                (x, kd + hd // 3, h + 10, "stone_bricks"),
            ]
        out += [(x, -hd + 1, 6, "lantern_block") for x in (-3, 3)]
    elif kind == "well":
        box(-1, -1, -3, 1, 1, 0, "cobblestone", hollow=True)
        out += [(0, 0, z, "water") for z in range(-2, 0)]
        out += [
            (-1, 0, 1, "oak_log"),
            (1, 0, 1, "oak_log"),
            (-1, 0, 2, "oak_log"),
            (1, 0, 2, "oak_log"),
        ]
        box(-1, -1, 3, 1, 1, 3, "spruce_planks")
    return tuple(out)


def structure_cells(item: dict) -> tuple:
    import json

    shape = {k: item[k] for k in ("kind", "size", "floors", "style", "seed")}
    return _cells_cached(json.dumps(shape, sort_keys=True))


def quarter_turns(item: dict) -> int:
    return int(round(item.get("rotation", 0.0) / 90.0)) % 4


def rotate_cells(cells: tuple, turns: int) -> list[tuple[int, int, int, str]]:
    out = []
    for x, y, z, block in cells:
        for _ in range(turns):
            x, y = -y, x
        out.append((x, y, z, block))
    return out


# --------------------------------------------------------------------------------------- #
# Sites, settlements and roads
# --------------------------------------------------------------------------------------- #


def find_sites(
    model: Any,
    count: int,
    *,
    radius: float = 45.0,
    max_slope: float | None = None,
    biomes: list[str] | None = None,
    spacing: float | None = None,
    region: tuple[float, float, float, float] | None = None,
    seed: int = 0,
    kind: str = "village",
) -> list[dict]:
    """Up to ``count`` sites, best first: one hashed candidate per ``spacing`` cell of the
    region (default: the extent, or 8 km around the origin), on dry land in the allowed
    biomes, judged on the natural ground: flatness within ``radius`` (``max_slope`` degrees
    at most; castles and towers prefer high ground), at least ``spacing`` apart."""
    d = model.d
    st = d["structures"]
    max_slope = st["max_slope"] if max_slope is None else max_slope
    spacing = st["spacing"] if spacing is None else spacing
    allowed = set(st["biomes"] if biomes is None else biomes)
    if region is None:
        e = model.extent()
        region = e if e is not None else (-4000.0, -4000.0, 4000.0, 4000.0)
    cell = max(spacing / 2.0, 2.0 * radius)
    i0, i1 = math.floor(region[0] / cell), math.floor(region[2] / cell)
    j0, j1 = math.floor(region[1] / cell), math.floor(region[3] / cell)
    if (i1 - i0 + 1) * (j1 - j0 + 1) > 40_000:
        raise WorldError(
            "TOO_LARGE",
            "the site search region is too large for its spacing",
            "structures",
            hint="give a smaller region or a larger spacing",
        )
    ci, cj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
    ci, cj = ci.reshape(-1), cj.reshape(-1)
    s = nz.derive(int(d["seed"]), "sites", kind, seed)
    x = (ci + 0.2 + 0.6 * nz.random2(nz.derive(s, "x"), ci, cj)) * cell
    y = (cj + 0.2 + 0.6 * nz.random2(nz.derive(s, "y"), ci, cj)) * cell
    ok = (
        (x >= region[0] + radius)
        & (x <= region[2] - radius)
        & (y >= region[1] + radius)
        & (y <= region[3] - radius)
    )
    x, y = x[ok], y[ok]
    if not x.size:
        return []
    ring = [(0.0, 0.0)] + [
        (radius * math.cos(a), radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 9)[:-1]
    ]
    ring += [
        (0.5 * radius * math.cos(a), 0.5 * radius * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, 7)[:-1]
    ]
    ox = np.array([p[0] for p in ring])
    oy = np.array([p[1] for p in ring])
    px = (x[:, None] + ox[None]).reshape(-1)
    py = (y[:, None] + oy[None]).reshape(-1)
    smp = model.evaluate(px, py)
    hn = model.natural(px, py).reshape(len(x), -1)
    wet = (~np.isnan(smp["water"])).reshape(len(x), -1)
    names = list(model.biome_names)
    biome = np.array(names)[np.argmax(smp["weights"].reshape(len(x), -1, len(names))[:, 0], axis=1)]
    spread = hn.max(axis=1) - hn.min(axis=1)
    grade = np.abs(hn[:, 1:9] - hn[:, :1]).max(axis=1) / radius
    c, sn = nz.cos_sin_deg(max_slope)
    slope_ok = grade <= sn / max(c, 1e-9)
    good = (
        slope_ok
        & ~wet.any(axis=1)
        & np.array([b in allowed for b in biome])
        & (hn[:, 0] > model.sea + 2.0)
    )
    score = spread.copy()
    if kind in ("castle", "tower"):  # high ground: prefer sites above their surroundings
        around = model.natural(
            (x[:, None] + 3 * ox[None, 1:9]).reshape(-1),
            (y[:, None] + 3 * oy[None, 1:9]).reshape(-1),
        ).reshape(len(x), -1)
        score = spread - 0.5 * (hn[:, 0] - around.mean(axis=1))
    order = np.argsort(score + np.where(good, 0.0, 1e9), kind="stable")
    out: list[dict] = []
    for k in order.tolist():
        if not good[k]:
            break
        if any(math.hypot(x[k] - o["at"][0], y[k] - o["at"][1]) < spacing for o in out):
            continue
        out.append(
            {
                "at": [round(float(x[k]), 2), round(float(y[k]), 2)],
                "height": round(float(hn[k, 0]), 2),
                "biome": str(biome[k]),
                "slope": round(math.degrees(math.atan(float(grade[k]))), 2),
                "score": round(float(score[k]), 3),
            }
        )
        if len(out) >= count:
            break
    return out


def plan_settlement(
    model: Any,
    at: list[float],
    houses: int,
    *,
    prefix: str,
    seed: int = 0,
    max_slope: float | None = None,
) -> dict:
    """A village around ``at``: road features (streets) and items (houses facing the
    streets, a well on the plaza, a tower in larger villages), deterministic from the
    seed; houses are skipped where the ground is wet or too steep or they would overlap."""
    d = model.d
    max_slope = d["structures"]["max_slope"] if max_slope is None else max_slope
    cx, cy = at
    s = nz.derive(int(d["seed"]), "settlement", prefix, seed)

    def u(k: int, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * float(nz.random2(s, np.array([k]), np.array([0]))[0])

    # The main street runs along the contour (across the slope).
    e = 6.0
    h = model.natural(np.array([cx + e, cx - e, cx, cx]), np.array([cy, cy, cy + e, cy - e]))
    gx, gy = (h[0] - h[1]) / (2 * e), (h[2] - h[3]) / (2 * e)
    angle = math.degrees(math.atan2(gx, -gy)) if math.hypot(gx, gy) > 1e-4 else u(1, 0.0, 180.0)
    angle = round(angle, 2)
    streets = [angle] + ([angle + 90.0] if houses > 8 else [])
    length = 26.0 + 9.0 * math.sqrt(houses)
    roads, items, rects = [], [], []
    items.append(
        {
            "id": f"{prefix}-well",
            "kind": "well",
            "at": [round(cx, 2), round(cy, 2)],
            "rotation": 0.0,
            "size": [2.4, 2.4],
            "floors": 1,
            "style": "stone",
            "seed": seed,
        }
    )
    rects.append(footprint(items[-1], 2.0))
    for k, a in enumerate(streets):
        ca, sa = nz.cos_sin_deg(a)
        half = length / 2.0 if k == 0 else length / 3.0
        roads.append(
            {
                "type": "road",
                "id": f"{prefix}-street-{k}",
                "shape": {
                    "line": [
                        [round(cx - ca * half, 2), round(cy - sa * half, 2)],
                        [round(cx + ca * half, 2), round(cy + sa * half, 2)],
                    ],
                    "width": 4.0,
                },
                "falloff": 3.0,
                "material": "dirt",
            }
        )
    placed = 0
    slot = 0
    for k, a in enumerate(streets):
        ca, sa = nz.cos_sin_deg(a)
        half = length / 2.0 if k == 0 else length / 3.0
        for t in np.arange(-half + 6.0, half - 3.0, 12.5):
            for side in (-1.0, 1.0):
                if placed >= houses:
                    break
                slot += 1
                w = round(u(10 * slot, 6.0, 9.0), 1)
                dd = round(u(10 * slot + 1, 5.0, 6.8), 1)
                off = 2.0 + dd / 2.0 + 2.2
                hx = cx + ca * (t + u(10 * slot + 2, -1.5, 1.5)) - sa * side * off
                hy = cy + sa * (t + u(10 * slot + 3, -1.5, 1.5)) + ca * side * off
                rot = a + (0.0 if side < 0 else 180.0)  # the front (-y) faces the street
                item = {
                    "id": f"{prefix}-house-{slot}",
                    "kind": "house",
                    "at": [round(hx, 2), round(hy, 2)],
                    "rotation": round(((rot + 180.0) % 360.0) - 180.0, 2),
                    "size": [w, dd],
                    "floors": 2 if u(10 * slot + 4) < 0.3 else 1,
                    "style": "timber"
                    if u(10 * slot + 5) < 0.65
                    else ("stone" if u(10 * slot + 6) < 0.6 else "brick"),
                    "seed": int(
                        nz.hash2(s, np.array([slot]), np.array([7]))[0] % np.uint64(2**31 - 1)
                    ),
                }
                poly = footprint(item, 1.2)
                if any(_overlap(poly, r) for r in rects):
                    continue
                pts = np.asarray(footprint(item, 0.0))
                hn = model.natural(pts[:, 0], pts[:, 1])
                smp = model.evaluate(pts[:, 0], pts[:, 1])
                if np.any(~np.isnan(smp["water"])) or hn.max() - hn.min() > max(w, dd) * math.tan(
                    math.radians(max_slope)
                ):
                    continue
                items.append(item)
                rects.append(poly)
                placed += 1
    # A clearing around the village (meadows between the houses and the forest).
    roads.append(
        {
            "type": "clear",
            "id": f"{prefix}-clearing",
            "shape": {"circle": [round(cx, 2), round(cy, 2), round(length * 0.65 + 20.0, 1)]},
            "what": ["trees"],
            "falloff": 25.0,
        }
    )
    if houses >= 10:
        ca, sa = nz.cos_sin_deg(angle)
        tower = {
            "id": f"{prefix}-tower",
            "kind": "tower",
            "at": [round(cx + sa * 9.0, 2), round(cy - ca * 9.0, 2)],
            "rotation": 0.0,
            "size": [6.0, 6.0],
            "floors": 2,
            "style": "stone",
            "seed": seed,
        }
        if not any(_overlap(footprint(tower, 1.0), r) for r in rects):
            items.append(tower)
    return {"roads": roads, "items": items, "houses": placed}


def _overlap(a: list, b: list) -> bool:
    """Whether two convex polygons overlap (separating axes)."""
    pa, pb = np.asarray(a), np.asarray(b)
    for poly in (pa, pb):
        for k in range(len(poly)):
            e = poly[(k + 1) % len(poly)] - poly[k]
            n = np.array([-e[1], e[0]])
            ra, rb = pa @ n, pb @ n
            if ra.max() < rb.min() or rb.max() < ra.min():
                return False
    return True


def plan_road(
    model: Any,
    a: list[float],
    b: list[float],
    *,
    prefix: str,
    cell: float = 16.0,
    max_cells: int = 250_000,
    bridges: bool = True,
) -> dict:
    """A road from ``a`` to ``b`` following the natural terrain: A* over a grid of ``cell``
    metres (cost: distance, times a penalty growing with the square of the grade; water is
    crossed only where narrow, by a bridge), simplified, split at water into road features
    and bridge items."""
    ax, ay = a
    bx, by = b
    margin = max(200.0, 0.35 * math.hypot(bx - ax, by - ay))
    x0, y0 = min(ax, bx) - margin, min(ay, by) - margin
    x1, y1 = max(ax, bx) + margin, max(ay, by) + margin
    e = model.extent()
    if e is not None:  # roads stay inside the world
        x0, y0, x1, y1 = max(x0, e[0]), max(y0, e[1]), min(x1, e[2]), min(y1, e[3])
        if not (
            e[0] <= ax <= e[2] and e[1] <= ay <= e[3] and e[0] <= bx <= e[2] and e[1] <= by <= e[3]
        ):
            raise WorldError(
                "OUTSIDE_WORLD", "a road's ends must be inside the world", "structures"
            )
    nx = int(math.ceil((x1 - x0) / cell)) + 1
    ny = int(math.ceil((y1 - y0) / cell)) + 1
    if nx * ny > max_cells:
        raise WorldError(
            "TOO_LARGE",
            f"a road over {nx * ny} grid cells (at most {max_cells})",
            "structures",
            hint="connect nearer places or use a larger cell",
        )
    gx, gy = np.meshgrid(x0 + np.arange(nx) * cell, y0 + np.arange(ny) * cell)
    hn = model.natural(gx.reshape(-1), gy.reshape(-1)).reshape(ny, nx)
    water = ~np.isnan(model.evaluate(gx.reshape(-1), gy.reshape(-1))["water"]).reshape(ny, nx)
    start = (int(round((ay - y0) / cell)), int(round((ax - x0) / cell)))
    goal = (int(round((by - y0) / cell)), int(round((bx - x0) / cell)))
    moves = [
        (dj, di, math.hypot(di, dj) * cell) for dj in (-1, 0, 1) for di in (-1, 0, 1) if di or dj
    ]
    best = {start: 0.0}
    prev: dict = {}
    heap = [(0.0, start)]
    while heap:
        f, node = heapq.heappop(heap)
        if node == goal:
            break
        g = best[node]
        if f - math.hypot(node[0] - goal[0], node[1] - goal[1]) * cell > g + 1e-9:
            continue
        j, i = node
        for dj, di, dist in moves:
            nj, ni = j + dj, i + di
            if not (0 <= nj < ny and 0 <= ni < nx):
                continue
            grade = abs(hn[nj, ni] - hn[j, i]) / dist
            step = dist * (1.0 + (grade / 0.07) ** 2 * 3.0)
            if water[nj, ni]:
                step += 600.0 + dist * 20.0
            cost = g + step
            if cost < best.get((nj, ni), math.inf):
                best[(nj, ni)] = cost
                prev[(nj, ni)] = node
                heapq.heappush(
                    heap, (cost + math.hypot(nj - goal[0], ni - goal[1]) * cell, (nj, ni))
                )
    if goal not in best:
        raise WorldError("NO_ROUTE", "no route between the two places", "structures")
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    nodes = np.array([(x0 + i * cell, y0 + j * cell) for j, i in path], dtype=np.float64)
    nodes[0], nodes[-1] = (ax, ay), (bx, by)
    # Water along the path, sampled every 2 m (rivers narrower than the grid included).
    seg = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, float(dist[-1]), max(2, int(dist[-1] / 2.0) + 1))
    sx, sy = np.interp(t, dist, nodes[:, 0]), np.interp(t, dist, nodes[:, 1])
    wet = ~np.isnan(model.evaluate(sx, sy)["water"])
    wet[0] = wet[-1] = False  # (the ends are on land by construction or are kept)
    # Runs on land become roads (simplified); runs over water become bridges from bank to
    # bank, two metres onto each.
    roads, bridges_found = [], []
    k = 0
    while k < len(t):
        if not wet[k]:
            k += 1
            continue
        end = k
        while end < len(t) and wet[end]:
            end += 1
        lo, hi = max(0.0, t[k - 1] - 2.0), min(float(dist[-1]), t[min(end, len(t) - 1)] + 2.0)
        bridges_found.append((lo, hi))
        k = end
    cuts = [0.0]
    for lo, hi in bridges_found:
        cuts += [lo, hi]
    cuts.append(float(dist[-1]))
    for n in range(0, len(cuts), 2):
        a0, a1 = cuts[n], cuts[n + 1]
        if a1 - a0 < 1.0:
            continue
        inner = [
            (float(x), float(y))
            for x, y, d in zip(nodes[:, 0], nodes[:, 1], dist, strict=True)
            if a0 < d < a1
        ]
        line = [
            (float(np.interp(a0, dist, nodes[:, 0])), float(np.interp(a0, dist, nodes[:, 1]))),
            *inner,
            (float(np.interp(a1, dist, nodes[:, 0])), float(np.interp(a1, dist, nodes[:, 1]))),
        ]
        roads.append(_simplify(line))
    if bridges_found and not bridges:
        raise WorldError(
            "NO_ROUTE",
            "the road would cross water and structures.bridges is false",
            "structures.bridges",
        )
    items = []
    for lo, hi in bridges_found:
        p = (float(np.interp(lo, dist, nodes[:, 0])), float(np.interp(lo, dist, nodes[:, 1])))
        q = (float(np.interp(hi, dist, nodes[:, 0])), float(np.interp(hi, dist, nodes[:, 1])))
        if 2.0 <= math.dist(p, q) <= 400.0:
            items.append(
                {
                    "id": f"{prefix}-bridge-{len(items)}",
                    "kind": "bridge",
                    "line": [[round(p[0], 2), round(p[1], 2)], [round(q[0], 2), round(q[1], 2)]],
                    "width": 4.0,
                }
            )
    features = [
        {
            "type": "road",
            "id": f"{prefix}-road-{n}",
            "shape": {"line": [[round(x, 2), round(y, 2)] for x, y in line], "width": 4.0},
            "falloff": 4.0,
            "material": "gravel",
        }
        for n, line in enumerate(roads)
        if len(line) >= 2
    ]
    return {"roads": features, "items": items, "length": round(float(dist[-1]), 1)}


def _simplify(run: list[tuple[float, float]], tol: float = 6.0) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification of a polyline."""
    if len(run) <= 2:
        return run
    a, b = np.asarray(run[0]), np.asarray(run[-1])
    ab = b - a
    ll = float(ab @ ab)
    best, idx = -1.0, 0
    for k in range(1, len(run) - 1):
        p = np.asarray(run[k]) - a
        t = 0.0 if ll == 0.0 else float(np.clip(p @ ab / ll, 0.0, 1.0))
        dist = float(np.linalg.norm(p - t * ab))
        if dist > best:
            best, idx = dist, k
    if best <= tol:
        return [run[0], run[-1]]
    return _simplify(run[: idx + 1], tol)[:-1] + _simplify(run[idx:], tol)


# --------------------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------------------- #


def render_parts(rt: Any, view: Any, ctx: Any = None) -> list:
    """Structure meshes within the view distance (and the frustum)."""
    d = rt.definition
    items = d["structures"]["items"]
    if not items or d["mode"] != "terrain":
        return []
    model = rt.model()
    rt.regional()
    eye = np.asarray(view.eye, dtype=np.float64)
    reach = min(view.distance, max(1500.0, d["vegetation"]["distance"]))
    world = rt.world
    out = []
    from .tiles import outside

    for item in items:
        if item["kind"] == "bridge":
            (ax, ay), (bx, by) = item["line"]
            box = (min(ax, bx) - 10, min(ay, by) - 10, max(ax, bx) + 10, max(ay, by) + 10)
        else:
            x, y = item["at"]
            r = max(item["size"]) + 12.0
            box = (x - r, y - r, x + r, y + r)
        cx = min(max(eye[0], box[0]), box[2])
        cy = min(max(eye[1], box[1]), box[3])
        if math.hypot(cx - eye[0], cy - eye[1]) > reach or outside(view, box, -1e4, 1e4):
            continue
        if ctx is not None:
            ctx.check("structures")
        oid = f"world:{world.id}:structure/{item['id']}"
        name = f"{world.name} {item['kind']} {item['id']}"
        if item["kind"] == "bridge":
            mesh, origin = _bridge(rt, item)
            out.append((oid, name, mesh, origin, LAYERS))
            continue
        z = model.pad_height(f"structure:{item['id']}")
        c, s = nz.cos_sin_deg(item["rotation"])
        m = np.eye(4)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        m[:3, 3] = (item["at"][0], item["at"][1], z)
        out.append((oid, name, structure_mesh(item), m, LAYERS))
    return out


def _bridge(rt: Any, item: dict) -> tuple[Mesh, np.ndarray]:
    cache = rt.__dict__.setdefault("_bridges", {})
    key = (rt.model().key, item["id"])
    found = cache.get(key)
    if found is None:
        model = rt.model()
        (ax, ay), (bx, by) = item["line"]
        ends = model.natural(np.array([ax, bx]), np.array([ay, by]))
        span = math.hypot(bx - ax, by - ay)
        piers = max(0, int(span // 9.0))
        t = (np.arange(piers) + 1) / (piers + 1)
        bed = (
            model.evaluate(ax + (bx - ax) * t, ay + (by - ay) * t)["height"]
            if piers
            else np.zeros(0)
        )
        found = cache[key] = bridge_mesh(item, float(ends[0]) + 0.35, float(ends[1]) + 0.35, bed)
    return found
