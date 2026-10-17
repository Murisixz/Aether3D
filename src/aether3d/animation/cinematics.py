"""Cinematics: paths, path following, shot sequences and camera moves.

**Paths** (items of kind ``path``) are curves in world space through ``points``:
``catmull_rom`` (centripetal, alpha 1/2: no cusps or self-intersections within a
segment, Yuksel et al. 2011), ``bezier`` (each point with ``in``/``out`` handle offsets)
or ``linear``; ``closed`` paths loop; a ``circle`` (``center``, ``radius``, ``normal``,
``start`` angle in degrees) is exact. Every other segment is a cubic, and positions are found
**by arc length**: the length of each segment is integrated with 16-point
Gauss-Legendre quadrature (Guenter and Parent 1990) and a distance is turned into a
segment parameter by safeguarded Newton steps, so moving the offset uniformly moves at a
constant speed.

**follow_path** (an object constraint) puts its owner on a path at ``offset`` (a fraction
of the length, animatable as ``constraints["NAME"].offset``; ``loop`` wraps it), and with
``follow`` turns its ``axis`` along the path's tangent, ``up`` toward world +Z.

**Shot sequences** (one item of kind ``sequence``): shots ``{camera, start, end, name,
transition}`` decide the camera at each time; a shot starting while another is on cuts
to it, or with ``transition: {"type": "blend", "duration": d}`` blends from the previous
shot's camera over ``d`` seconds (position linearly, orientation by slerp, field of view
linearly) through a virtual camera. Renders that name no camera use the sequence.

**Camera moves** (``AnimationSystem.camera_move``/``camera_orbit``) write ordinary keys
and constraints -- editable data, not effects: dolly (along the view), truck (sideways),
pedestal/crane (vertically), pan (about world Z), tilt (about the camera's X), roll,
zoom (field of view), and orbits (a circular path, a follow_path constraint whose offset
is keyed with an easing, and a look_at constraint on the target).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import paths as P
from . import rotation as R

if TYPE_CHECKING:
    from .system import AnimationSystem

PATH_KINDS = ("catmull_rom", "bezier", "linear", "circle")
MAX_PATH_POINTS = 100_000
MAX_SHOTS = 10_000
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _points(raw: Any, location: str, minimum: int = 2) -> np.ndarray:
    try:
        p = np.asarray(raw, np.float64)
    except (TypeError, ValueError):
        raise _err("INVALID_PATH", "points are [[x, y, z], ...]", location) from None
    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not minimum <= p.shape[0] <= MAX_PATH_POINTS
        or not np.all(np.isfinite(p))
    ):
        raise _err(
            "INVALID_PATH", f"points are {minimum}..{MAX_PATH_POINTS} finite [x, y, z]", location
        )
    return p


@dataclass(frozen=True)
class Path3D:
    """An immutable path (see the module docstring)."""

    points: np.ndarray  # (k, 3)
    kind: str = "catmull_rom"
    closed: bool = False
    handles: np.ndarray | None = None  # (k, 2, 3): in, out offsets (bezier)
    circle: tuple[tuple[float, ...], float, tuple[float, ...], float] | None = None
    _cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def to_json(self) -> dict[str, Any]:
        if self.circle is not None:
            c, r, nrm, start = self.circle
            return {
                "kind": "circle",
                "center": list(c),
                "radius": r,
                "normal": list(nrm),
                "start": start,
            }
        out: dict[str, Any] = {"points": self.points.tolist(), "kind": self.kind}
        if self.closed:
            out["closed"] = True
        if self.handles is not None:
            out["handles"] = self.handles.tolist()
        return out

    # -- segments as cubic Hermite (power basis) ------------------------------------------

    def coefficients(self) -> np.ndarray:
        """``(segments, 4, 3)`` power-basis cubics ``((a u + b) u + c) u + d``, u in 0..1."""
        found = self._cache.get("coef")
        if found is not None:
            return found
        p = self.points
        k = p.shape[0]
        n = k if self.closed else k - 1
        idx = np.arange(n)
        p1 = p[idx]
        p2 = p[(idx + 1) % k]
        if self.kind == "linear":
            m1 = m2 = p2 - p1
        elif self.kind == "bezier":
            h = self.handles
            m1 = 3.0 * h[idx, 1]
            m2 = -3.0 * h[(idx + 1) % k, 0]
        else:  # centripetal Catmull-Rom
            if self.closed:
                p0 = p[(idx - 1) % k]
                p3 = p[(idx + 2) % k]
            else:
                ext = np.concatenate([[2 * p[0] - p[1]], p, [2 * p[-1] - p[-2]]])
                p0 = ext[idx]
                p3 = ext[idx + 3]

            def knot(a, b):
                return np.sqrt(np.maximum(np.linalg.norm(b - a, axis=1), 1e-12))[:, None]

            t01, t12, t23 = knot(p0, p1), knot(p1, p2), knot(p2, p3)
            m1 = ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12) + (p2 - p1) / t12) * t12
            m2 = ((p2 - p1) / t12 - (p3 - p1) / (t12 + t23) + (p3 - p2) / t23) * t12
        a = 2 * p1 - 2 * p2 + m1 + m2
        b = -3 * p1 + 3 * p2 - 2 * m1 - m2
        c = m1
        d = p1
        coef = np.stack([a, b, c, d], axis=1)
        coef.flags.writeable = False
        self._cache["coef"] = coef
        return coef

    def lengths(self) -> np.ndarray:
        """Arc length of each segment (16-point Gauss-Legendre) and their running sum."""
        found = self._cache.get("lengths")
        if found is None:
            coef = self.coefficients()
            u = _GL_X
            d = (3 * coef[:, None, 0] * u[None, :, None] + 2 * coef[:, None, 1]) * u[
                None, :, None
            ] + coef[:, None, 2]
            seg = np.sum(np.linalg.norm(d, axis=2) * _GL_W[None, :], axis=1)
            found = self._cache["lengths"] = (seg, np.concatenate([[0.0], np.cumsum(seg)]))
        return found

    @property
    def length(self) -> float:
        if self.circle is not None:
            return 2.0 * math.pi * self.circle[1]
        return float(self.lengths()[1][-1])

    def _circle_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        found = self._cache.get("circle_basis")
        if found is None:
            found = self._cache["circle_basis"] = self._make_circle_basis()
        return found

    def _make_circle_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c, _, nrm, _ = self.circle  # type: ignore[misc]
        n = np.asarray(nrm, np.float64)
        n = n / np.linalg.norm(n)
        e1 = np.cross(n, [0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0])
        e1 = np.cross(e1, n)
        e1 /= np.linalg.norm(e1)
        if abs(n[2]) >= 0.9:  # a horizontal circle starts along +X
            e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
            e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        out = (np.array(c, np.float64), e1, e2)
        for a in out:
            a.flags.writeable = False  # (shared by every call)
        return out

    def _segment_length(self, s: int, u: np.ndarray) -> np.ndarray:
        """Arc length from the segment's start to ``u`` (Gauss-Legendre on [0, u])."""
        coef = self.coefficients()[s]
        x = u[:, None] * _GL_X[None, :]
        d = (3 * coef[0] * x[..., None] + 2 * coef[1]) * x[..., None] + coef[2]
        return u * np.sum(np.linalg.norm(d, axis=2) * _GL_W[None, :], axis=1)

    def locate(self, fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(segment, u)`` of length fractions (0..1, clamped)."""
        seg_len, cum = self.lengths()
        total = cum[-1]
        dist = np.clip(np.asarray(fraction, np.float64), 0.0, 1.0) * total
        s = np.clip(np.searchsorted(cum, dist, side="right") - 1, 0, len(seg_len) - 1)
        local = dist - cum[s]
        u = np.where(seg_len[s] > 0, local / np.where(seg_len[s] > 0, seg_len[s], 1.0), 0.0)
        for _ in range(8):  # Newton on the arc length, kept in [0, 1]
            for seg in np.unique(s).tolist():
                sel = s == seg
                f = self._segment_length(seg, u[sel]) - local[sel]
                coef = self.coefficients()[seg]
                uu = u[sel]
                d = np.linalg.norm(
                    (3 * coef[0] * uu[:, None] + 2 * coef[1]) * uu[:, None] + coef[2], axis=1
                )
                step = np.where(d > 1e-12, f / np.where(d > 1e-12, d, 1.0), 0.0)
                u[sel] = np.clip(uu - step, 0.0, 1.0)
        return s, u

    def point(self, fraction: Any) -> tuple[np.ndarray, np.ndarray]:
        """Positions and unit tangents at length fractions."""
        f = np.atleast_1d(np.asarray(fraction, np.float64))
        if self.circle is not None:
            c, e1, e2 = self._circle_basis()
            r, start = self.circle[1], self.circle[3]
            theta = np.radians(start) + 2.0 * math.pi * f
            cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
            return c + r * (cos * e1 + sin * e2), -sin * e1 + cos * e2
        s, u = self.locate(f)
        coef = self.coefficients()[s]
        uu = u[:, None]
        pos = ((coef[:, 0] * uu + coef[:, 1]) * uu + coef[:, 2]) * uu + coef[:, 3]
        tan = (3 * coef[:, 0] * uu + 2 * coef[:, 1]) * uu + coef[:, 2]
        n = np.linalg.norm(tan, axis=1)
        tan = tan / np.where(n > 0, n, 1.0)[:, None]
        return pos, tan


def parse_path(raw: Any, location: str) -> Path3D:
    if isinstance(raw, Path3D):
        return raw
    if not isinstance(raw, dict) or set(raw) - {
        "points",
        "kind",
        "closed",
        "handles",
        "center",
        "radius",
        "normal",
        "start",
    }:
        raise _err(
            "INVALID_PATH",
            "a path is {points, kind, closed, handles} "
            "(circles: {kind, center, radius, normal, start})",
            location,
        )
    kind = raw.get("kind", "catmull_rom")
    if kind not in PATH_KINDS:
        raise _err("INVALID_PATH", f"kind is one of {', '.join(PATH_KINDS)}", f"{location}.kind")
    if kind == "circle":
        if set(raw) - {"kind", "center", "radius", "normal", "start"}:
            raise _err("INVALID_PATH", "a circle is {center, radius, normal, start}", location)
        center = _points([raw.get("center")], f"{location}.center", 1)[0]
        normal = _points([raw.get("normal", [0.0, 0.0, 1.0])], f"{location}.normal", 1)[0]
        radius = raw.get("radius")
        start = raw.get("start", 0.0)
        if (
            isinstance(radius, bool)
            or not isinstance(radius, (int, float))
            or not 0 < float(radius) < 1e9
        ):
            raise _err("INVALID_PATH", "radius is a positive number", f"{location}.radius")
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(float(start))
        ):
            raise _err("INVALID_PATH", "start is an angle in degrees", f"{location}.start")
        if np.linalg.norm(normal) < 1e-12:
            raise _err("INVALID_PATH", "normal must not be zero", f"{location}.normal")
        pts = np.zeros((2, 3))
        pts.flags.writeable = False
        return Path3D(
            pts,
            "circle",
            True,
            None,
            (tuple(center.tolist()), float(radius), tuple(normal.tolist()), float(start)),
        )
    closed = raw.get("closed", False)
    if not isinstance(closed, bool):
        raise _err("INVALID_PATH", "closed is true or false", f"{location}.closed")
    pts = _points(raw.get("points"), f"{location}.points", 3 if closed else 2)
    handles = None
    if kind == "bezier":
        h = raw.get("handles")
        try:
            handles = np.asarray(h, np.float64)
        except (TypeError, ValueError):
            handles = None
        if (
            handles is None
            or handles.shape != (pts.shape[0], 2, 3)
            or not np.all(np.isfinite(handles))
        ):
            raise _err(
                "INVALID_PATH",
                "a bezier path has handles [[in, out], ...] per point",
                f"{location}.handles",
            )
        handles.flags.writeable = False
    elif raw.get("handles") is not None:
        raise _err("INVALID_PATH", "only bezier paths have handles", f"{location}.handles")
    pts.flags.writeable = False
    path = Path3D(pts, kind, closed, handles)
    if not path.length > 0:
        raise _err("INVALID_PATH", "a path must have a positive length", location)
    return path


def follow_path(
    system: AnimationSystem, constraint: Any, world: np.ndarray, offset: float
) -> np.ndarray | None:
    """The owner's world matrix on the constraint's path at ``offset`` (a length
    fraction), or ``None`` when the path is gone."""
    from .constraints import decompose, look_rotation

    p = constraint.p
    item = system._items.get(p["path"])
    if item is None or not item._alive or item.kind != "path":
        return None
    path: Path3D = item._definition
    if p["loop"]:
        offset = offset - math.floor(offset)
    pos, tan = path.point(np.array([offset]))
    out = world.copy()
    out[:3, 3] = pos[0]
    if p["follow"]:
        _, _, s = decompose(world)
        out[:3, :3] = look_rotation(tan[0], np.array([0.0, 0.0, 1.0]), p["axis"], p["up"]) * s
    return out


# --------------------------------------------------------------------------------------- #
# Shot sequences
# --------------------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Shot:
    camera: str
    start: float
    end: float
    name: str = "shot"
    blend: float = 0.0  # seconds of blend from the previous shot (0: a cut)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "camera": self.camera,
            "start": self.start,
            "end": self.end,
            "name": self.name,
        }
        if self.blend > 0:
            out["transition"] = {"type": "blend", "duration": self.blend}
        return out


@dataclass(frozen=True)
class Sequence:
    shots: tuple[Shot, ...]

    @property
    def object(self) -> str:  # (one sequence per scene: keyed like the owned items)
        return "sequence"

    def to_json(self) -> dict[str, Any]:
        return {"shots": [s.to_json() for s in self.shots]}

    def at(self, t: float) -> tuple[Shot | None, Shot | None, float]:
        """``(shot, previous, blend weight)`` at ``t``: the latest shot started (``start <=
        t < end``), the shot it blends from and how far the blend is (1: done)."""
        live = [s for s in self.shots if s.start <= t < s.end]
        if not live:
            last = [s for s in self.shots if s.end == t]
            live = last[-1:] if last else []
        if not live:
            return None, None, 1.0
        shot = max(live, key=lambda s: (s.start, self.shots.index(s)))
        if shot.blend > 0 and t < shot.start + shot.blend:
            before = [s for s in self.shots if s is not shot and s.start <= shot.start < s.end]
            prev = max(before, key=lambda s: s.start) if before else None
            if prev is not None:
                x = (t - shot.start) / shot.blend
                return shot, prev, x * x * (3.0 - 2.0 * x)  # smoothstep
        return shot, None, 1.0


def parse_sequence(raw: Any, location: str) -> Sequence:
    if isinstance(raw, Sequence):
        return raw
    if not isinstance(raw, dict) or set(raw) != {"shots"}:
        raise _err("INVALID_SEQUENCE", "a sequence is {shots}", location)
    shots_raw = raw["shots"]
    if not isinstance(shots_raw, list) or len(shots_raw) > MAX_SHOTS:
        raise _err(
            "INVALID_SEQUENCE", f"shots is a list of at most {MAX_SHOTS}", f"{location}.shots"
        )
    shots = []
    for i, s in enumerate(shots_raw):
        loc = f"{location}.shots[{i}]"
        if (
            not isinstance(s, dict)
            or set(s) - {"camera", "start", "end", "name", "transition"}
            or not {"camera", "start", "end"} <= set(s)
        ):
            raise _err("INVALID_SEQUENCE", "a shot is {camera, start, end, name, transition}", loc)
        if not isinstance(s["camera"], str) or not s["camera"]:
            raise _err(
                "INVALID_SEQUENCE", "camera is a camera object (id or name)", f"{loc}.camera"
            )
        start, end = (
            float(s[k])
            if isinstance(s[k], (int, float)) and not isinstance(s[k], bool)
            else math.nan
            for k in ("start", "end")
        )
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise _err("INVALID_SEQUENCE", "a shot needs finite start < end", loc)
        blend = 0.0
        tr = s.get("transition")
        if tr is not None:
            if (
                not isinstance(tr, dict)
                or tr.get("type") not in ("cut", "blend")
                or set(tr) - {"type", "duration"}
            ):
                raise _err(
                    "INVALID_SEQUENCE",
                    "transition is {type: cut | blend, duration}",
                    f"{loc}.transition",
                )
            if tr["type"] == "blend":
                d = tr.get("duration")
                if (
                    isinstance(d, bool)
                    or not isinstance(d, (int, float))
                    or not 0 < float(d) <= end - start
                ):
                    raise _err(
                        "INVALID_SEQUENCE",
                        "a blend's duration is within the shot",
                        f"{loc}.transition.duration",
                    )
                blend = float(d)
        name = P.check_name(s.get("name", f"shot {i + 1}"), "a shot name", f"{loc}.name")
        shots.append(Shot(s["camera"], start, end, name, blend))
    return Sequence(tuple(sorted(shots, key=lambda s: (s.start, s.end))))


def shot_camera(
    system: AnimationSystem, t: float, worlds: dict[str, np.ndarray], fields: dict
) -> Any:
    """The camera of the active sequence at ``t``: a camera object id (a cut), a virtual
    :class:`~aether3d.rendering.scene_data.RenderCamera` (a blend), or ``None``."""
    from ..rendering.camera import frame_from_matrix
    from ..rendering.scene_data import RenderCamera, _with_fields

    item = system._owned.get(("sequence", "sequence"))
    if item is None:
        return None
    shot, prev, w = item._definition.at(t)
    if shot is None:
        return None
    scene = system.scene
    cam = scene._objects.get(shot.camera)
    if cam is None or cam._camera is None:
        return None
    if prev is None:
        return cam._id
    other = scene._objects.get(prev.camera)
    if other is None or other._camera is None:
        return cam._id

    def lens(obj):
        values = fields.get((obj._id, "camera"))
        return _with_fields(obj._camera, values) if values else obj._camera

    la, lb = lens(other), lens(cam)
    wa, wb = worlds[other._id], worlds[cam._id]
    from .constraints import blend_matrices

    m = blend_matrices(wa, wb, w)
    blended = la.copy()
    blended._owner = None
    changes = {}
    for f in ("fov", "ortho_scale", "near", "far"):
        changes[f] = getattr(la, f) + w * (getattr(lb, f) - getattr(la, f))
    if la.projection == lb.projection:
        blended.configure(**changes)
    frame, _ = frame_from_matrix(m, f"objects[{cam.name}]")
    return RenderCamera(None, f"{prev.name} -> {shot.name}", blended, frame)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


__all__ = [
    "PATH_KINDS",
    "Path3D",
    "Sequence",
    "Shot",
    "follow_path",
    "parse_path",
    "parse_sequence",
    "shot_camera",
    "R",
]
