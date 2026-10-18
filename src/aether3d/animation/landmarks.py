"""Landmarks of unrigged characters: where the joints of a template belong.

:func:`detect_landmarks` measures one or more meshes (a body with its clothing, armour
and accessories, in any number of parts) and returns named points -- hips, knees,
ankles, heels, toes, shoulders, elbows, wrists, hand and head tips, the spine, the neck
-- each with a **confidence** (0..1), the **method** that placed it and a sentence
explaining why, plus diagnostics for whatever was ambiguous. Nothing is guessed silently:
a landmark the geometry cannot decide comes from a proportion prior and says so.

How (humanoids; see the module code for the details):

1. **Frame.** The up axis is given (``+z`` by default). The facing is measured: the
   vertical plane of best mirror symmetry of the voxelized solid gives the lateral
   axis; feet protruding ahead of the legs (else a nose, else the convention ``-y``)
   give its sign. Landmarks are measured in a canonical frame (up ``+z``, facing ``-y``,
   the character's left ``+x``, the symmetry plane at ``x = 0``, the soles at ``z = 0``).
2. **Extremities.** The thickest point of the torso is the *core*; geodesic distances
   inside the solid from it (:func:`.voxels.geodesic`) have local maxima at the
   extremities: the head's top, the hands' and the toes' tips. They are found by
   farthest-point sampling with suppression and classified by height and side.
3. **Limb centrelines** (a Reeb graph): for each extremity, the solid is cut into thin
   slices of equal geodesic distance from it; each slice's connected pieces are nodes,
   pieces of consecutive slices that touch are linked, and the chain of single pieces
   from the extremity up to the first branching or sudden widening is the limb -- its
   pieces' centroids are the centreline, their sizes the limb's radius. Bent limbs,
   A-, T- and arms-down poses, odd proportions and blocky limbs all give a centreline.
4. **Joints on the centrelines**: the limb's junction with the body (shoulder, hip, neck
   base), the narrowest section between hand and forearm (wrist) or head and torso
   (neck), the strongest bend (elbow, knee) when the rest pose bends, else the
   anatomical proportion of the limb's measured length; the ankle where the leg's
   centreline turns forward into the foot; heels and toes on the sole. The spine
   follows the centroids of the torso's horizontal sections.
5. **Symmetry.** Left and right landmarks are averaged across the symmetry plane when
   they agree within 6 % of the height (``symmetry="auto"``), else kept and reported
   as an asymmetry.

``landmarks=`` overrides any point (world space) before the joints are built, so a
person or an agent corrects one landmark and rebuilds; overridden points have method
``override`` and confidence 1. Detection is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import voxels as V

if TYPE_CHECKING:
    from .system import AnimationSystem

AXES = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}
TEMPLATES = ("humanoid", "stylized", "blocky", "quadruped")
# Proportion priors (fractions of the stature; Drillis and Contini, after Winter 2009).
PRIOR = {
    "ankle": 0.039,
    "knee": 0.285,
    "hip": 0.51,
    "shoulder": 0.818,
    "chin": 0.87,
    "upper_arm": 0.186,
    "forearm": 0.146,
    "hand": 0.108,
    "hip_half_width": 0.052,
    "shoulder_half_width": 0.105,
}


def _err(code: str, message: str, location: str, hint: str | None = None) -> AnimationError:
    return AnimationError(code, message, location, hint=hint)


@dataclass
class Landmark:
    name: str
    position: np.ndarray  # canonical frame while measuring; world in reports
    confidence: float
    method: str
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "position": [round(float(v), 6) for v in self.position],
            "confidence": round(float(self.confidence), 3),
            "method": self.method,
            "note": self.note,
        }


@dataclass
class Detection:
    """The measured character: landmarks in the canonical frame, the frame itself."""

    template: str
    rotation: np.ndarray  # canonical axes (rows) in world space
    origin: np.ndarray  # world position of the canonical origin
    height: float
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def world(self, p: np.ndarray) -> np.ndarray:
        return self.origin + np.asarray(p, np.float64) @ self.rotation

    def canonical(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, np.float64) - self.origin) @ self.rotation.T

    def point(self, name: str) -> np.ndarray:
        return self.landmarks[name].position

    def add(self, name: str, p: Any, confidence: float, method: str, note: str) -> None:
        self.landmarks[name] = Landmark(name, np.asarray(p, np.float64), confidence, method, note)

    def diag(self, code: str, severity: str, message: str, **extra: Any) -> None:
        self.diagnostics.append({"code": code, "severity": severity, "message": message, **extra})

    def report(self) -> dict[str, Any]:
        lm = {}
        for name, l_ in self.landmarks.items():
            j = l_.to_json()
            j["position"] = [round(float(v), 6) for v in self.world(l_.position)]
            lm[name] = j
        conf = [l_.confidence for l_ in self.landmarks.values()]
        info = {k: v for k, v in self.info.items() if not k.startswith("_")}
        return {
            "template": self.template,
            "height": round(self.height, 6),
            "frame": {
                "up": [round(float(v), 6) for v in self.rotation[2]],
                "forward": [round(float(v), 6) for v in -self.rotation[1]],
                "left": [round(float(v), 6) for v in self.rotation[0]],
                "origin": [round(float(v), 6) for v in self.origin],
            },
            "confidence": round(float(min(conf)) if conf else 0.0, 3),
            "landmarks": lm,
            "diagnostics": self.diagnostics,
            **info,
        }


# --------------------------------------------------------------------------------------- #
# Geometry gathering and the frame
# --------------------------------------------------------------------------------------- #


def gather(system: AnimationSystem, meshes: Any) -> tuple[np.ndarray, np.ndarray, list[Any]]:
    """World-space positions and triangles of the given mesh objects (one soup)."""
    from ..rendering.scene_data import world_matrices

    refs = meshes if isinstance(meshes, (list, tuple)) else [meshes]
    if not refs:
        raise _err("INVALID_ARGUMENT", "give one or more mesh objects", "meshes")
    worlds = world_matrices(system.scene)
    pos, tris, objs, base = [], [], [], 0
    for i, ref in enumerate(refs):
        obj = system._object(ref, f"meshes[{i}]")
        mesh = obj.mesh
        if mesh is None or mesh.triangle_count == 0:
            raise _err("INVALID_ARGUMENT", f"{obj.name!r} has no triangles", f"meshes[{i}]")
        m = worlds[obj._id]
        p = mesh.positions.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
        pos.append(p)
        tris.append(np.asarray(mesh.triangles, np.int64) + base)
        base += p.shape[0]
        objs.append(obj)
    return np.concatenate(pos), np.concatenate(tris), objs


def _frame_from(up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Rows: the canonical axes (left, backward, up) in world space."""
    u = up / np.linalg.norm(up)
    f = forward - (forward @ u) * u
    f = f / np.linalg.norm(f)
    left = np.cross(u, f)
    return np.stack([left, -f, u])


def _symmetry(points: np.ndarray, occupied, h: float, height: float) -> tuple[float, float, float]:
    """``(angle, offset, score)``: the vertical mirror plane (normal at ``angle`` radians
    in the horizontal plane, through the centroid moved by ``offset`` along it) that maps
    the most solid voxels onto solid voxels."""
    c = points.mean(axis=0)
    best = (0.0, 0.0, -1.0)

    def score(theta: float, off: float) -> float:
        n = np.array([math.cos(theta), math.sin(theta), 0.0])
        q = points - 2.0 * (((points - c) @ n) - off)[:, None] * n[None]
        return float(occupied(q).mean())

    for theta in np.radians(np.arange(0.0, 180.0, 3.0)):
        s = score(theta, 0.0)
        if s > best[2]:
            best = (float(theta), 0.0, s)
    theta0 = best[0]
    for theta in theta0 + np.radians(np.arange(-3.0, 3.01, 0.5)):
        for off in np.linspace(-0.04, 0.04, 9) * height:
            s = score(float(theta), float(off))
            if s > best[2] + 1e-12:
                best = (float(theta), float(off), s)
    return best


# --------------------------------------------------------------------------------------- #
# Reeb chains along geodesic slices
# --------------------------------------------------------------------------------------- #

_HALF = V._OFFSETS[: len(V._OFFSETS) // 2]  # one of each opposite pair of neighbours


@dataclass
class Chain:
    centroids: np.ndarray  # (k, 3) canonical, from the extremity inward
    sizes: np.ndarray  # voxels per slice piece
    junction: np.ndarray | None  # where the branch splits off the body (None: not found)
    reason: str


class Reeb:
    """The Reeb graph of the solid under the geodesic distance from the core: slices of
    equal distance, cut into connected pieces (nodes), pieces of consecutive slices that
    touch linked. Limbs are the branches that end at extremities; they leave the body
    where a piece splits into several (crotch, armpits, neck base)."""

    def __init__(self, grid, centers, solid_flat, dist, slice_units: int, min_piece: int = 3):
        region = np.flatnonzero(dist >= 0)
        bins = dist[region] // slice_units
        nx, ny, _ = grid.dims
        member = np.full(grid.sid().size, -1, np.int64)
        member[solid_flat[region]] = np.arange(region.size)
        same_a, same_b, next_a, next_b = [], [], [], []
        flat_r = solid_flat[region]
        for off in _HALF:
            o = int(off[0] + nx * (off[1] + ny * off[2]))
            nb = member[flat_r + o]
            ok = nb >= 0
            a, b = np.flatnonzero(ok), nb[ok]
            same = bins[a] == bins[b]
            same_a.append(a[same])
            same_b.append(b[same])
            step = np.abs(bins[a] - bins[b]) == 1
            next_a.append(a[step])
            next_b.append(b[step])
        labels = V.label_graph(region.size, np.concatenate(same_a), np.concatenate(same_b))
        ids, comp = np.unique(labels, return_inverse=True)
        n = ids.size
        self.size = np.bincount(comp, minlength=n)
        self.centroid = (
            np.stack([np.bincount(comp, centers[region, i], n) for i in range(3)], axis=1)
            / np.maximum(self.size, 1)[:, None]
        )
        self.bin = np.zeros(n, np.int64)
        self.bin[comp] = bins
        na, nb_ = np.concatenate(next_a), np.concatenate(next_b)
        links = np.unique(np.stack([comp[na], comp[nb_]], axis=1), axis=0)
        self.inward: list[list[int]] = [[] for _ in range(n)]
        self.outward: list[list[int]] = [[] for _ in range(n)]
        for a, b in links.tolist():
            if a == b:
                continue
            lo, hi = (a, b) if self.bin[a] < self.bin[b] else (b, a)
            self.outward[lo].append(hi)
            self.inward[hi].append(lo)
        self.min_piece = min_piece
        self.comp_of = np.full(dist.size, -1, np.int64)
        self.comp_of[region] = comp
        # How far outward each piece's subtree reaches (in slices): a split counts only
        # when the other way leads somewhere (a limb), not to a thumb or a crumb.
        reach = self.bin.copy()
        for c in np.argsort(-self.bin, kind="stable").tolist():
            for o in self.outward[c]:
                if reach[o] > reach[c]:
                    reach[c] = reach[o]
        self.reach = reach
        self.slice_m = slice_units * grid.h / 3.0

    def _real(self, nodes: list[int]) -> list[int]:
        return [c for c in nodes if self.size[c] >= self.min_piece]

    def branch(self, voxel: int, min_reach: float = 0.0) -> Chain:
        """The branch from a solid voxel (an extremity) inward to where it leaves the body:
        the set of pieces of each slice that lead to it (near tips a slice can break into
        crumbs), until a piece of the next slice inward also leads elsewhere (a split:
        the branch has reached the body) or the core is reached."""
        front = {int(self.comp_of[voxel])}
        cents, sizes = [], []
        reason, junction = "the core", None
        while front:
            members = np.array(sorted(front))
            sz = self.size[members]
            cents.append((self.centroid[members] * sz[:, None]).sum(axis=0) / max(sz.sum(), 1))
            sizes.append(float(sz.sum()))
            inner: set[int] = set()
            for c in front:
                inner.update(self.inward[c])
            if not inner:
                break
            others = set()
            for p in inner:
                others.update(self._real(self.outward[p]))
            others -= front
            # A split counts when the other way leads at least ``min_reach`` farther out
            # (another limb; not a thumb, an ear or a crumb of the voxelization).
            big = [o for o in others if (self.reach[o] - self.bin[o]) * self.slice_m >= min_reach]
            if big:
                reason = "branching"
                junction = self.centroid[max(inner, key=lambda p: self.size[p])]
                break
            front = inner
        return Chain(np.array(cents), np.array(sizes), junction, reason)


def _arc(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])


def _at(points: np.ndarray, s: np.ndarray, value: float) -> np.ndarray:
    """The point at arc length ``value`` along a polyline (clamped)."""
    value = min(max(value, float(s[0])), float(s[-1]))
    return np.stack([np.interp(value, s, points[:, i]) for i in range(3)])


def _smooth(points: np.ndarray, k: int = 2) -> np.ndarray:
    if len(points) <= 2 * k:
        return points
    out = points.copy()
    for i in range(k, len(points) - k):
        out[i] = points[i - k : i + k + 1].mean(axis=0)
    return out


def _bend(points: np.ndarray, s: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """``(arc length, degrees)`` of the strongest bend of a polyline between arc lengths
    ``lo`` and ``hi`` (the angle between the directions a limb segment before and after)."""
    best = (0.5 * (lo + hi), 0.0)
    span = max(0.12 * (s[-1] - s[0]), 1e-6)
    for v in np.linspace(lo, hi, 25):
        a = _at(points, s, v - span)
        m = _at(points, s, v)
        b = _at(points, s, v + span)
        d1, d2 = m - a, b - m
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        ang = math.degrees(math.acos(min(1.0, max(-1.0, float(d1 @ d2) / (n1 * n2)))))
        if ang > best[1]:
            best = (float(v), ang)
    return best


# --------------------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------------------- #


def detect(
    positions: np.ndarray,
    triangles: np.ndarray,
    *,
    template: str = "humanoid",
    up: str | Any = "+z",
    forward: str | Any = "auto",
    resolution: int = 128,
    symmetry: str = "auto",
    overrides: dict[str, Any] | None = None,
) -> Detection:
    """Landmarks of a character mesh (world space); see the module docstring."""
    if template not in TEMPLATES:
        raise _err("INVALID_ARGUMENT", f"template is one of {', '.join(TEMPLATES)}", "template")
    if symmetry not in ("auto", "off"):
        raise _err("INVALID_ARGUMENT", "symmetry is 'auto' or 'off'", "symmetry")
    upv = _axis(up, "up")
    p = np.asarray(positions, np.float64)
    # Provisional frame (up known): measure in a frame with up = +z.
    helper = AXES["-y"] if abs(upv @ AXES["-y"]) < 0.9 else AXES["+x"]
    rot0 = _frame_from(upv, helper)
    lo_up = float((p @ upv).min())
    height = float((p @ upv).max()) - lo_up
    if height <= 0:
        raise _err("INVALID_MESH", "the meshes have no height along the up axis", "up")
    origin0 = upv * lo_up
    q0 = (p - origin0) @ rot0.T
    grid = V.voxelize(q0, triangles, resolution=resolution)
    if grid.stats["shell_only"]:
        grid = V.voxelize(q0, triangles, resolution=resolution, close=2)
    solid_flat = np.flatnonzero(grid.solid_flat())
    nx, ny, _ = grid.dims
    ijk = np.stack([solid_flat % nx, (solid_flat // nx) % ny, solid_flat // (nx * ny)], axis=1)
    centers = grid.center(ijk)
    solid = grid.solid_flat()

    def occupied(q: np.ndarray) -> np.ndarray:
        idx = np.floor((q - grid.origin) / grid.h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(grid.dims)), axis=1)
        out = np.zeros(len(q), bool)
        f = idx[ok, 0] + nx * (idx[ok, 1] + ny * idx[ok, 2])
        out[ok] = solid[f]
        return out

    det = Detection(template, rot0, origin0, height)
    det.info["voxel_size"] = grid.h
    det.info["grid"] = grid.stats["dims"]
    # Keep the main part (the largest connected piece); report detached parts.
    labels, ncomp = V.components(grid.solid)
    lab = labels.transpose(2, 1, 0).reshape(-1)[solid_flat]
    counts = np.bincount(lab, minlength=ncomp)
    main = int(np.argmax(counts))
    if ncomp > 1:
        det.diag("DETACHED_PARTS", "info",
                 f"{ncomp - 1} parts do not touch the body; they are not measured",
                 parts=int(ncomp - 1))  # fmt: skip
    keep = lab == main
    # Facing and symmetry plane.
    sub = centers[keep][:: max(1, int(keep.sum()) // 40000)]
    theta, off, sym = _symmetry(sub, occupied, grid.h, height)
    det.info["symmetry"] = round(sym, 4)
    n = np.array([math.cos(theta), math.sin(theta), 0.0])
    plane_c = sub.mean(axis=0) + off * n
    if forward == "auto":
        f_c, how, conf_f = _facing(centers[keep], n, height, template)
        det.info["facing"] = how
        if conf_f < 0.6:
            det.diag("FORWARD_AMBIGUOUS", "warning",
                     f"the facing is uncertain ({how}); pass forward= to fix it",
                     confidence=round(conf_f, 2))  # fmt: skip
    else:
        fw = _axis(forward, "forward")
        f_c = fw @ rot0.T
        f_c[2] = 0.0
        if np.linalg.norm(f_c) < 1e-6:
            raise _err("INVALID_ARGUMENT", "forward must not be along up", "forward")
        f_c /= np.linalg.norm(f_c)
        det.info["facing"] = "given"
        conf_f = 1.0
    # Final canonical frame: left = up x forward, origin on the symmetry plane at the soles.
    rot_c = _frame_from(np.array([0.0, 0.0, 1.0]), f_c)  # within the provisional frame
    rot = rot_c @ rot0
    plane_world = origin0 + plane_c @ rot0
    proj = (plane_world - origin0) @ rot0.T
    origin = origin0 + np.array([proj[0], proj[1], 0.0]) @ rot0
    det.rotation, det.origin = rot, origin
    to_c = rot_c  # provisional -> canonical (rotation only, plus the origin shift)
    shift = np.array([proj[0], proj[1], 0.0])
    c_centers = (centers - shift) @ to_c.T
    if sym < 0.75:
        det.diag("ASYMMETRIC_BODY", "info",
                 f"only {sym:.0%} of the body mirrors onto itself; left and right are measured "
                 "separately", symmetry=round(sym, 3))  # fmt: skip
    if template == "blocky":
        _blocky(det, grid, c_centers, solid_flat, keep, height)
    elif template == "quadruped":
        _quadruped(det, grid, c_centers, solid_flat, keep, height)
    else:
        _humanoid(det, grid, c_centers, solid_flat, keep, height, template, (to_c, shift))
    if symmetry == "auto" and sym >= 0.75:
        _symmetrize(det, height)
    if overrides:
        for name, value in overrides.items():
            if not isinstance(name, str):
                raise _err("INVALID_ARGUMENT", "landmark names are strings", "landmarks")
            w = np.asarray(value, np.float64).reshape(-1)
            if w.shape != (3,) or not np.all(np.isfinite(w)):
                raise _err(
                    "INVALID_ARGUMENT", "a landmark is [x, y, z] (world)", f"landmarks.{name}"
                )
            det.add(name, det.canonical(w), 1.0, "override", "given")
    return det


def _axis(value: Any, what: str) -> np.ndarray:
    if isinstance(value, str):
        if value not in AXES:
            raise _err("INVALID_ARGUMENT", f"{what} is one of {', '.join(AXES)} or [x, y, z]", what)
        return AXES[value].copy()
    v = np.asarray(value, np.float64).reshape(-1)
    if v.shape != (3,) or not np.all(np.isfinite(v)) or np.linalg.norm(v) < 1e-9:
        raise _err("INVALID_ARGUMENT", f"{what} is a non-zero direction", what)
    return v / np.linalg.norm(v)


def _facing(centers: np.ndarray, lateral: np.ndarray, height: float, template: str):
    """``(forward (provisional frame), how, confidence)``: perpendicular to the lateral
    axis, toward where the feet point (else the nose, else ``-y``)."""
    f = np.cross(np.array([0.0, 0.0, 1.0]), lateral)  # one of the two candidates
    f /= np.linalg.norm(f)
    z = centers[:, 2]
    if template == "quadruped":
        top = centers[z > np.quantile(z, 0.9)]
        lean = float((top.mean(axis=0) - centers.mean(axis=0)) @ f)
        if abs(lean) > 0.02 * height:
            return (f if lean > 0 else -f), "head end", 0.8
        return np.array([0.0, -1.0, 0.0]), "convention (-y)", 0.3
    feet = centers[z < 0.035 * height]
    legs = centers[(z > 0.09 * height) & (z < 0.2 * height)]
    if len(feet) > 10 and len(legs) > 10:
        d = float((feet.mean(axis=0) - legs.mean(axis=0)) @ f)
        if abs(d) > 0.008 * height:
            return (f if d > 0 else -f), "feet point forward", min(0.95, 0.6 + abs(d) / height * 8)
    head = centers[z > 0.9 * height]
    if len(head) > 10:
        mid = head.mean(axis=0)
        ext = head @ f - mid @ f
        lean = float(np.quantile(ext, 0.98) + np.quantile(ext, 0.02))
        if abs(lean) > 0.01 * height:
            return (f if lean > 0 else -f), "the face protrudes forward", 0.55
    cand = np.array([0.0, -1.0, 0.0])
    return (f if cand @ f >= 0 else -f), "convention (-y)", 0.3


def _symmetrize(det: Detection, height: float) -> None:
    pairs = sorted({n[:-2] for n in det.landmarks if n.endswith((".L", ".R"))})
    flip = np.array([-1.0, 1.0, 1.0])
    for base in pairs:
        left, right = f"{base}.L", f"{base}.R"
        if left not in det.landmarks or right not in det.landmarks:
            continue
        a, b = det.landmarks[left], det.landmarks[right]
        diff = float(np.linalg.norm(a.position - b.position * flip))
        if diff <= 0.06 * height:
            mean = 0.5 * (a.position + b.position * flip)
            conf = min(a.confidence, b.confidence)
            a.position, b.position = mean, mean * flip
            a.confidence = b.confidence = conf
        else:
            det.diag("ASYMMETRIC_LANDMARK", "info",
                     f"{base} differs between sides by {diff:.3f} (kept as measured)",
                     landmark=base, difference=round(diff, 4))  # fmt: skip


# --------------------------------------------------------------------------------------- #
# Humanoids
# --------------------------------------------------------------------------------------- #


def _humanoid(det, grid, centers, solid_flat, keep, H, template, frame) -> None:
    unit = grid.h / 3.0
    sid = grid.sid()
    to_c, shift = frame
    nx, ny, _ = grid.dims
    # Depth inside the solid (distance to the surface), from the full solid.
    surface_flat = np.flatnonzero(grid.surface.transpose(2, 1, 0).reshape(-1))
    depth = V.geodesic(grid, [surface_flat], solid_flat)[:, 0].astype(np.float64)

    def probe(points: np.ndarray) -> np.ndarray:
        """Depth (m) inside the solid at canonical points (0 outside)."""
        prov = np.atleast_2d(points) @ to_c + shift
        idx = np.floor((prov - grid.origin) / grid.h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(grid.dims)), axis=1)
        out = np.zeros(len(prov))
        f = idx[ok, 0] + nx * (idx[ok, 1] + ny * idx[ok, 2])
        si = sid[f]
        good = si >= 0
        vals = np.zeros(si.size)
        vals[good] = np.maximum(depth[si[good]], 0.0) * unit
        out[ok] = vals
        return out

    det.info["_probe"] = probe
    # The analysis solid: eroded by a voxel, so limbs that merely touch (a hand on a
    # thigh) separate -- unless that would break a substantial part off the body.
    agrid, a_flat, a_centers, contacts = _analysis_solid(grid, solid_flat, centers, keep)
    det.info["contacts_broken"] = contacts
    z = a_centers[:, 2]
    a_depth = V.geodesic(agrid, [np.flatnonzero(agrid.surface.transpose(2, 1, 0).reshape(-1))],
                         a_flat)[:, 0].astype(np.float64)  # fmt: skip
    band = (z > 0.45 * H) & (z < 0.75 * H) & (np.abs(a_centers[:, 0]) < 0.12 * H)
    if not band.any():
        band = np.ones(len(z), bool)
    core = int(np.flatnonzero(band)[np.argmax(a_depth[band])])
    det.info["core"] = [round(float(v), 4) for v in a_centers[core]]
    g_units = V.geodesic(agrid, [a_flat[[core]]], a_flat)[:, 0].astype(np.int64)
    g_core = np.where(g_units >= 0, g_units * unit, -1.0)
    # Extremities: local maxima of the distance from the core (farthest-point sampling
    # with suppression; a candidate with a farther point near it is not a tip).
    alive = g_core >= 0
    extremities: list[tuple[int, np.ndarray]] = []
    for _ in range(14):
        cand = np.where(alive, g_core, -1.0)
        e = int(np.argmax(cand))
        if cand[e] < 0.25 * H:
            break
        ge = V.geodesic(agrid, [a_flat[[e]]], a_flat)[:, 0]
        near = (ge >= 0) & (ge * unit < 0.1 * H)
        if g_core[near].max() <= g_core[e] + 2 * grid.h:
            extremities.append((e, ge))
        alive &= ~((ge >= 0) & (ge * unit < 0.3 * H))
    det.info["extremities"] = len(extremities)
    feet = [x for x in extremities if z[x[0]] < 0.15 * H]
    others = [x for x in extremities if z[x[0]] >= 0.25 * H]
    head = None
    centred = [x for x in others if abs(a_centers[x[0], 0]) < 0.12 * H and z[x[0]] > 0.7 * H]
    if centred:
        head = max(centred, key=lambda x: z[x[0]])
        others = [x for x in others if x[0] != head[0]]
    hands = {}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        cand = [
            x for x in others if sign * a_centers[x[0], 0] > 0.06 * H and g_core[x[0]] > 0.3 * H
        ]
        if cand:
            hands[side] = max(cand, key=lambda x: g_core[x[0]])
            others = [x for x in others if x[0] != hands[side][0]]
    foot = {}
    for side, sign in (("L", 1.0), ("R", -1.0)):
        cand = [x for x in feet if sign * a_centers[x[0], 0] > 0]
        if cand:
            foot[side] = max(cand, key=lambda x: g_core[x[0]])
    for x in others:
        det.diag("EXTRA_EXTREMITY", "info",
                 "a protrusion that is not a head, hand or foot (a tail, a weapon, hair...)",
                 position=[round(float(v), 3) for v in det.world(a_centers[x[0]])])  # fmt: skip
    slice_units = 5  # (a slice is 5/3 voxels thick)
    reeb = Reeb(agrid, a_centers, a_flat, g_units, slice_units)
    debug = det.info.setdefault("_debug", {"chains": {}})
    debug.update(centers=centers[keep], extremities=[a_centers[e] for e, _ in extremities],
                 core=a_centers[core])  # fmt: skip

    def branch(voxel: int, label: str) -> Chain:
        ch = reeb.branch(voxel, 0.1 * H)
        debug["chains"][label] = ch
        return ch

    sections = _Sections(centers[keep], grid.h)
    # Head and neck first (shoulders hang from the neck's base).
    if head is not None:
        e, _ = head
        _head(det, branch(e, "head"), _tip(a_centers[e], branch(e, "head"), grid.h), H)
    else:
        det.diag("HEAD_NOT_FOUND", "warning", "no head top found; the neck and head use priors")
        det.add("head_top", [0, 0, H], 0.4, "proportion", "the top of the mesh")
        det.add("head", [0, 0, PRIOR["chin"] * H], 0.4, "proportion", "chin height prior")
        det.add("neck", [0, 0, (PRIOR["shoulder"] + 0.03) * H], 0.4, "proportion", "prior")
    # Legs.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        if side not in foot:
            det.diag("FOOT_NOT_FOUND", "warning", f"no {side} foot tip; the leg uses priors")
            _leg_prior(det, side, sign, H)
            continue
        e, _ = foot[side]
        ch = branch(e, f"leg.{side}")
        _leg(det, side, sign, ch, _tip(a_centers[e], ch, grid.h), centers[keep], sections, H,
             float(a_centers[core][2]))  # fmt: skip
    # Arms.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        if side not in hands:
            msg = f"no {side} hand tip (arms touching the body?); priors place the arm"
            det.diag("HAND_NOT_FOUND", "warning", msg)
            _arm_prior(det, side, sign, H)
            continue
        e, _ = hands[side]
        ch = branch(e, f"arm.{side}")
        _arm(det, side, sign, ch, _tip(a_centers[e], ch, grid.h), sections, H)
    _spine(det, centers, keep, H)
    sh, wr = det.landmarks.get("shoulder.L"), det.landmarks.get("wrist.L")
    if sh is not None and wr is not None:
        d = wr.position - sh.position
        ang = math.degrees(math.atan2(abs(d[0]), -d[2])) if np.linalg.norm(d) > 0 else 0.0
        det.info["arm_angle"] = round(ang, 1)
        det.info["pose"] = "T-pose" if ang > 70 else ("A-pose" if ang > 20 else "arms down")


def _analysis_solid(grid, solid_flat, centers, keep):
    """``(grid, solid flat indices, canonical centres, contacts broken?)`` of the solid
    the extremities and limbs are measured on (see :func:`_humanoid`)."""
    nx, ny, _ = grid.dims
    full = np.zeros(grid.solid_flat().size, bool)
    full[solid_flat[keep]] = True
    solid3 = full.reshape(grid.solid.shape[::-1]).transpose(2, 1, 0)
    eroded = V.erode(solid3, 1) & solid3
    lab, n = V.components(eroded)
    flat_lab = lab.transpose(2, 1, 0).reshape(-1)
    inside = flat_lab >= 0
    counts = np.bincount(flat_lab[inside], minlength=max(n, 1))
    use = n >= 1 and counts.max() >= 0.985 * counts.sum() and counts.max() > 0.3 * keep.sum()
    if not use:
        g = V.VoxelGrid(grid.origin, grid.h, solid3, grid.surface & solid3)
        return g, solid_flat[keep], centers[keep], False
    main = int(np.argmax(counts))
    mask = lab == main
    surf = mask & ~V.erode(mask, 1)
    g = V.VoxelGrid(grid.origin, grid.h, mask, surf)
    flat = np.flatnonzero(g.solid_flat())
    pos = {int(f): i for i, f in enumerate(solid_flat.tolist())} if False else None
    del pos
    lookup = np.full(full.size, -1, np.int64)
    lookup[solid_flat] = np.arange(solid_flat.size)
    a_centers = centers[lookup[flat]]
    return g, flat, a_centers, bool(n > 1 or counts.max() < keep.sum())


def _tip(point: np.ndarray, ch: Chain, h: float) -> np.ndarray:
    """An extremity measured on the eroded solid, moved back out by a voxel."""
    if len(ch.centroids) < 2:
        return point
    d = ch.centroids[0] - ch.centroids[min(3, len(ch.centroids) - 1)]
    n = float(np.linalg.norm(d))
    return point if n < 1e-9 else point + d / n * h


class _Sections:
    """Horizontal sections of the solid (canonical voxel centres): widths about x = 0."""

    def __init__(self, centers: np.ndarray, h: float) -> None:
        self.c = centers
        self.h = h

    def half_width(self, z: float, y: float | None = None, band: float | None = None) -> float:
        """Half width of the central run of solid at height ``z`` (and depth ``y``): the
        torso's, not an arm hanging beside it."""
        b = self.h if band is None else band
        sel = self.c[np.abs(self.c[:, 2] - z) <= b]
        if y is not None:
            sel = sel[np.abs(sel[:, 1] - y) <= 2 * self.h]
        if not len(sel):
            return 0.0
        xs = np.unique(np.round(sel[:, 0] / self.h).astype(np.int64))
        out = []
        for direction in (1, -1):
            k = 0
            while (direction * (k + 1)) in set(xs.tolist()) or (k == 0 and 0 in set(xs.tolist())):
                k += 1
                if k > 10_000:
                    break
            out.append(k * self.h)
        return float(np.mean(out))

    def widest(self, z0: float, z1: float, y: float = 0.0) -> tuple[float, float]:
        """``(z, half width)`` of the widest central section between ``z0`` and ``z1``."""
        best = (0.5 * (z0 + z1), 0.0)
        for z in np.arange(z0, z1, self.h):
            w = self.half_width(float(z), y)
            if w > best[1]:
                best = (float(z), w)
        return best


def _enter(
    pts: np.ndarray,
    s: np.ndarray,
    radius: float,
    probe,
    *,
    frac: float = 0.35,
    factor: float = 1.3,
    most: float = 4.0,
) -> tuple[np.ndarray, float, str]:
    """Where a limb's axis (the line through the last ``frac`` of its centreline) enters
    the body: the first point beyond the limb's end, going inward, at least ``factor``
    limb radii deep. Returns ``(point, radii travelled, how)``."""
    start, d = _axis_line(pts, s, frac)
    steps = np.linspace(0.0, most * radius, 33)
    q = start[None] + steps[:, None] * d[None]
    deep = probe(q)
    hit = np.flatnonzero(deep >= factor * radius)
    if hit.size:
        k = int(hit[0])
        return q[k], float(steps[k] / max(radius, 1e-9)), "the limb's axis entering the body"
    k = int(np.argmax(deep))
    return q[k], float(steps[k] / max(radius, 1e-9)), "the deepest point on the limb's axis"


def _axis_line(pts: np.ndarray, s: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """``(point at the limb's end, inward direction)`` of the line fitted to the last
    ``frac`` of a centreline."""
    tail = pts[s >= s[-1] - frac * (s[-1] - s[0])]
    if len(tail) < 3:
        tail = pts[-3:]
    c = tail.mean(axis=0)
    _, _, vt = np.linalg.svd(tail - c)
    d = vt[0]
    if d @ (tail[-1] - tail[0]) < 0:
        d = -d
    return c + ((pts[-1] - c) @ d) * d, d


def _leg(det, side, sign, ch: Chain, toe_tip, centers, sections, H, core_z: float) -> None:
    pts = _smooth(ch.centroids)
    s = _arc(pts)
    vs = det.info["voxel_size"]
    radius = np.sqrt(np.maximum(ch.sizes, 1.0)) * vs / 1.6
    # The ankle: where the centreline rises above the foot (at least the prior height).
    rise = pts[:, 2]
    ankle_z = max(PRIOR["ankle"] * H, float(np.median(rise[: max(2, len(rise) // 6)])) * 1.6)
    upward = np.flatnonzero(rise >= ankle_z)
    k_a = int(upward[0]) if upward.size else len(pts) // 5
    ankle = pts[k_a].copy()
    ankle_conf = 0.8 if upward.size else 0.5
    # The hip: on the thigh's axis entering the pelvis -- unless the legs fuse below the
    # hips (thick thighs): then at the widest section of the hips.
    r_leg = float(np.median(radius[len(radius) // 3 :]))
    top = float(pts[-1][2])
    widest_z, _ = sections.widest(min(top, core_z - 0.05 * H), core_z)
    fused = ch.junction is None or top < 0.6 * widest_z
    if not fused:
        hip, travel, how = _enter(pts, s, r_leg, det.info["_probe"])
        hip_conf, hip_how, hip_method = 0.75, f"{how}, {travel:.1f} thigh radii up", "junction"
    else:
        start, d = _axis_line(pts, s, 0.35)
        t = (widest_z - 0.035 * H - start[2]) / d[2] if abs(d[2]) > 1e-6 else 0.0
        hip = start + max(t, 0.0) * d
        hip_conf, hip_how, hip_method = (
            0.55,
            "the legs join below the hips: below the hips' widest section",
            "section",
        )
        det.diag("LEGS_FUSED", "info",
                 f"the {side} leg joins the body below the hips; the hip sits below the "
                 "hips' widest section", leg=side)  # fmt: skip
    # The knee: the strongest bend if the rest pose bends, else the proportion by height.
    frac = (PRIOR["knee"] - PRIOR["ankle"]) / (PRIOR["hip"] - PRIOR["ankle"])
    k_h = int(np.argmin(np.abs(pts[:, 2] - hip[2]))) if pts[-1][2] >= hip[2] else len(pts) - 1
    lo, hi = s[k_a] + 0.3 * (s[k_h] - s[k_a]), s[k_a] + 0.75 * (s[k_h] - s[k_a])
    bend_at, bend = _bend(pts, s, lo, hi)
    if bend > 6.0:
        knee, knee_conf, knee_how, knee_method = (
            _at(pts, s, bend_at),
            0.85,
            f"the leg bends {bend:.0f} degrees here",
            "bend",
        )
    else:
        target = ankle[2] + frac * (hip[2] - ankle[2])
        legpts = pts[k_a:]
        if len(legpts) >= 2 and legpts[-1][2] >= target:
            k = int(np.argmin(np.abs(legpts[:, 2] - target)))
            knee = legpts[k].copy()
            knee[2] = target
        else:
            knee = ankle + frac * (hip - ankle)
        knee_conf, knee_how, knee_method = (
            0.7,
            "straight leg: at the anatomical fraction of its height",
            "proportion_of_measured",
        )
    # Heel and ball on the sole.
    foot_vox = centers[(centers[:, 2] < ankle[2]) & (np.abs(centers[:, 0] - ankle[0]) < 0.08 * H)]
    if len(foot_vox):
        back = foot_vox[np.argmax(foot_vox[:, 1])]
        heel = np.array([ankle[0], back[1], 0.0])
    else:
        heel = np.array([ankle[0], ankle[1] + 0.03 * H, 0.0])
    ball = toe_tip + 0.3 * (heel - toe_tip)
    ball[2] = max(ball[2], 0.3 * ankle[2])
    det.add(f"toe_tip.{side}", toe_tip, 0.9, "extremity", "the farthest point of the foot")
    det.add(
        f"ball.{side}", ball, 0.7, "proportion_of_measured", "30 % from the toe tip to the heel"
    )
    det.add(f"heel.{side}", heel, 0.75, "sole", "the back of the foot at the ground")
    det.add(
        f"ankle.{side}", ankle, ankle_conf, "turn", "where the leg's centreline turns into the foot"
    )
    det.add(f"knee.{side}", knee, knee_conf, knee_method, knee_how)
    det.add(f"hip.{side}", hip, hip_conf, hip_method, hip_how)


def _leg_prior(det, side, sign, H) -> None:
    x = sign * PRIOR["hip_half_width"] * H
    det.add(f"hip.{side}", [x, 0, PRIOR["hip"] * H], 0.3, "proportion", "prior")
    det.add(f"knee.{side}", [x, 0, PRIOR["knee"] * H], 0.3, "proportion", "prior")
    det.add(f"ankle.{side}", [x, 0, PRIOR["ankle"] * H], 0.3, "proportion", "prior")
    det.add(f"heel.{side}", [x, 0.03 * H, 0], 0.3, "proportion", "prior")
    det.add(f"ball.{side}", [x, -0.08 * H, 0.01 * H], 0.3, "proportion", "prior")
    det.add(f"toe_tip.{side}", [x, -0.12 * H, 0.01 * H], 0.3, "proportion", "prior")


def _arm(det, side, sign, ch: Chain, tip, sections, H) -> None:
    pts = _smooth(ch.centroids)
    s = _arc(pts)
    vs = det.info["voxel_size"]
    radius = np.sqrt(np.maximum(ch.sizes, 1.0)) * vs / 1.6
    # The wrist: the narrowest section 0.05..0.17 H from the tip.
    window = (s > 0.05 * H) & (s < 0.17 * H)
    if window.sum() >= 3:
        k_w = int(np.flatnonzero(window)[np.argmin(radius[window])])
        wrist, w_conf, w_how = pts[k_w], 0.75, "the narrowest section above the hand"
    else:
        wrist = _at(pts, s, PRIOR["hand"] * H)
        w_conf, w_how = 0.5, "a hand length from the tip (prior)"
    # The shoulder: on the upper arm's axis, nearest an anchor beside the neck's base
    # (the joint is under the shoulder's top, at the chest's width).
    neck = det.landmarks["neck"].position
    start, d = _axis_line(pts, s, 0.35)
    z_a = float(neck[2]) - 0.025 * H
    chest_z = float(neck[2]) - 0.12 * H
    w_c = sections.half_width(chest_z, float(np.median(pts[:, 1])), 2 * vs)
    if w_c <= 0:
        w_c = PRIOR["shoulder_half_width"] * H
    anchor = np.array([sign * w_c, start[1], z_a])
    t = float((anchor - start) @ d)
    shoulder = start + max(t, 0.0) * d
    s_conf = 0.7 if ch.junction is not None else 0.45
    s_how = f"on the upper arm's axis beside the neck's base (chest half width {w_c:.3f})"
    # The elbow: the strongest bend, else the anatomical fraction from wrist to shoulder.
    k_w = int(np.argmin(np.linalg.norm(pts - wrist, axis=1)))
    poly = np.vstack([pts[k_w:], shoulder[None]])
    sp = _arc(poly)
    frac = PRIOR["forearm"] / (PRIOR["forearm"] + PRIOR["upper_arm"])
    bend_at, bend = _bend(poly, sp, 0.3 * sp[-1], 0.65 * sp[-1])
    if bend > 8.0:
        elbow, e_conf, e_how, e_method = (
            _at(poly, sp, bend_at),
            0.85,
            f"the arm bends {bend:.0f} degrees here",
            "bend",
        )
    else:
        elbow = _at(poly, sp, frac * sp[-1])
        e_conf, e_how, e_method = (
            0.7,
            "straight arm: at the anatomical fraction",
            "proportion_of_measured",
        )
    det.add(f"hand_tip.{side}", tip, 0.9, "extremity", "the farthest point of the hand")
    det.add(f"wrist.{side}", wrist, w_conf, "narrowest", w_how)
    det.add(f"elbow.{side}", elbow, e_conf, e_method, e_how)
    det.add(f"shoulder.{side}", shoulder, s_conf, "axis_anchor", s_how)


def _arm_prior(det, side, sign, H) -> None:
    x = sign * PRIOR["shoulder_half_width"] * H
    z = PRIOR["shoulder"] * H
    det.add(f"shoulder.{side}", [x, 0, z], 0.3, "proportion", "prior (arms down)")
    det.add(f"elbow.{side}", [x, 0, z - PRIOR["upper_arm"] * H], 0.3, "proportion", "prior")
    det.add(f"wrist.{side}", [x, 0, z - (PRIOR["upper_arm"] + PRIOR["forearm"]) * H], 0.3,
            "proportion", "prior")  # fmt: skip
    det.add(f"hand_tip.{side}",
            [x, 0, z - (PRIOR["upper_arm"] + PRIOR["forearm"] + PRIOR["hand"]) * H], 0.3,
            "proportion", "prior")  # fmt: skip


def _head(det, ch: Chain, tip, H) -> None:
    pts = _smooth(ch.centroids)
    vs = det.info["voxel_size"]
    radius = np.sqrt(np.maximum(ch.sizes, 1.0)) * vs / 1.6
    n = len(pts)
    # The neck: the narrowest section below the head (the lower part of the chain).
    lower = (np.arange(n) >= n // 4) & (pts[:, 2] < tip[2] - 0.06 * H)
    if not lower.any():
        lower = np.arange(n) >= n // 4
    k_n = int(np.flatnonzero(lower)[np.argmin(radius[lower])]) if lower.any() else n - 1
    r_n = radius[k_n]
    # The head joint (skull base): going up from the neck, where the section widens 30 %.
    k_h = k_n
    for k in range(k_n, -1, -1):
        if radius[k] > 1.3 * r_n:
            k_h = k
            break
    head = 0.5 * (pts[k_h] + pts[k_n])
    # The neck's base: going down from the neck, where the section widens by half.
    k_b = n - 1
    for k in range(k_n, n):
        if radius[k] > 1.5 * r_n:
            k_b = k
            break
    neck = pts[k_b].copy()
    det.add("head_top", tip, 0.9, "extremity", "the top of the head")
    det.add("head", head, 0.7, "narrowest", "the top of the neck, where the head widens")
    det.add("neck", neck, 0.7 if k_b < n - 1 else 0.5, "widening",
            "the neck's base, where the section widens into the shoulders")  # fmt: skip


def _spine(det, centers, keep, H) -> None:
    lm = det.landmarks
    hips = 0.5 * (lm["hip.L"].position + lm["hip.R"].position)
    neck = lm["neck"].position
    z0, z1 = float(hips[2]), float(neck[2])
    torso = keep.copy()
    sh_w = 0.6 * max(abs(lm["shoulder.L"].position[0]), abs(lm["shoulder.R"].position[0]), 0.08 * H)
    torso &= np.abs(centers[:, 0]) < sh_w
    out = []
    for f in (0.0, 0.35, 0.7):
        zz = z0 + f * (z1 - z0) + 0.025 * H * (f == 0.0)
        sl = centers[torso & (np.abs(centers[:, 2] - zz) < 0.02 * H)]
        y = float(np.median(sl[:, 1])) if len(sl) > 5 else float(hips[1])
        out.append(np.array([0.0, y, zz]))
    det.add(
        "pelvis", out[0], 0.75, "section_centroid", "between the hip joints, centred in the pelvis"
    )
    det.add("spine", out[1], 0.65, "section_centroid", "35 % up the torso, centred in its section")
    det.add("chest", out[2], 0.65, "section_centroid", "70 % up the torso, centred in its section")
    for side, sign in (("L", 1.0), ("R", -1.0)):
        sh = lm[f"shoulder.{side}"].position
        base = np.array([sign * 0.02 * H, 0.5 * (neck[1] + sh[1]) - 0.005 * H,
                         neck[2] - 0.035 * H])  # fmt: skip
        det.add(
            f"clavicle.{side}", base, 0.6, "between", "beside the neck's base, toward the shoulder"
        )


def _blocky(det, grid, centers, solid_flat, keep, H) -> None:
    """Minecraft-style characters: head, body, two arms and two legs as separate blocks
    (or touching boxes); pivots at the top of each limb block and the base of the head."""
    z, x = centers[:, 2], centers[:, 0]
    legs = keep & (z < 0.375 * H)
    body = keep & (z >= 0.375 * H) & (z < 0.75 * H)
    width = float(np.quantile(np.abs(x[body]), 0.98)) if body.any() else 0.2 * H
    core = body & (np.abs(x) < 0.5 * width)
    torso_w = float(np.quantile(np.abs(x[core]), 0.98)) if core.any() else 0.12 * H
    leg_top = float(z[legs].max()) if legs.any() else 0.375 * H
    body_top = float(z[body].max()) if body.any() else 0.75 * H
    det.add("pelvis", [0, 0, leg_top], 0.8, "block", "the top of the legs")
    det.add("chest", [0, 0, 0.5 * (leg_top + body_top)], 0.6, "block", "the body's middle")
    det.add("neck", [0, 0, body_top], 0.85, "block", "the top of the body block")
    det.add("head", [0, 0, body_top], 0.85, "block", "the head turns about the body's top")
    det.add("head_top", [0, 0, H], 0.9, "extremity", "the top of the head block")
    for side, sign in (("L", 1.0), ("R", -1.0)):
        arm = body & (sign * x > torso_w * 0.98)
        if arm.any():
            ax = float(np.median(x[arm]))
            det.add(f"shoulder.{side}", [ax, 0, body_top - 0.05 * H], 0.8, "block",
                    "the top of the arm block, on its axis")  # fmt: skip
            det.add(f"hand_tip.{side}", [ax, 0, float(z[arm].min())], 0.8, "block",
                    "the bottom of the arm block")  # fmt: skip
        else:
            det.add(f"shoulder.{side}", [sign * torso_w * 1.25, 0, body_top - 0.05 * H], 0.3,
                    "proportion", "no arm block found")  # fmt: skip
            det.add(f"hand_tip.{side}", [sign * torso_w * 1.25, 0, leg_top], 0.3, "proportion",
                    "no arm block found")  # fmt: skip
        leg = legs & (sign * x > 0)
        lx = float(np.median(x[leg])) if leg.any() else sign * 0.5 * torso_w
        det.add(f"hip.{side}", [lx, 0, leg_top], 0.8 if leg.any() else 0.3, "block",
                "the top of the leg block")  # fmt: skip
        det.add(f"toe_tip.{side}", [lx, -0.06 * H, 0.0], 0.6, "block", "the front of the leg block")
        det.add(f"heel.{side}", [lx, 0.06 * H, 0.0], 0.6, "block", "the back of the leg block")
    del grid, solid_flat


def _quadruped(det, grid, centers, solid_flat, keep, H) -> None:
    """Four-legged creatures facing -y: four feet (the lowest extremities), the head (the
    front, highest), the tail (the back); spine along the body's top."""
    z, y = centers[:, 2], centers[:, 1]
    body = keep & (z > 0.45 * H)
    front = float(np.quantile(y[body], 0.03)) if body.any() else -0.3 * H
    back = float(np.quantile(y[body], 0.97)) if body.any() else 0.3 * H
    torso = body & (y > front + 0.25 * (back - front)) & (y < back - 0.15 * (back - front))
    body_z = float(np.median(z[torso])) if torso.any() else 0.6 * H
    length = back - front
    det.add("pelvis", [0, front + 0.78 * length, body_z], 0.6, "body", "the back of the trunk")
    det.add("spine", [0, front + 0.55 * length, body_z], 0.6, "body", "the middle of the trunk")
    det.add("chest", [0, front + 0.3 * length, body_z], 0.6, "body", "the front of the trunk")
    headpart = keep & (y < front + 0.15 * length) & (z > body_z)
    if headpart.any():
        hp = centers[headpart]
        det.add("head", hp.mean(axis=0) + [0, 0.25 * (hp[:, 1].max() - hp[:, 1].min()), 0], 0.6,
                "body", "the head's back")  # fmt: skip
        det.add("head_top", hp[np.argmin(hp[:, 1])], 0.7, "extremity", "the snout")
    else:
        det.add("head", [0, front, body_z + 0.2 * H], 0.3, "proportion", "no head found")
        det.add("head_top", [0, front - 0.1 * H, body_z + 0.2 * H], 0.3, "proportion", "prior")
    det.add("neck", [0, front + 0.18 * length, body_z + 0.1 * H], 0.5, "body", "the trunk's front")
    tail = keep & (y > back - 0.12 * length) & (z > 0.5 * body_z)
    if tail.any():
        tp = centers[tail]
        det.add("tail_tip", tp[np.argmax(tp[:, 1])], 0.6, "extremity", "the tail's tip")
    for end, (y0, y1) in (("front", (front, front + 0.45 * length)),
                          ("back", (front + 0.55 * length, back))):  # fmt: skip
        for side, sign in (("L", 1.0), ("R", -1.0)):
            leg = keep & (z < 0.45 * H) & (y >= y0) & (y <= y1) & (sign * centers[:, 0] > 0)
            if leg.any():
                lp = centers[leg]
                foot = lp[np.argmin(lp[:, 2])]
                top = np.array([float(np.median(lp[:, 0])), float(np.median(lp[:, 1])), body_z])
                det.add(f"{end}_hip.{side}", top, 0.6, "body", "the leg under the trunk")
                knee = top + 0.5 * (foot - top)
                det.add(f"{end}_knee.{side}", knee, 0.5, "proportion_of_measured", "mid leg")
                det.add(f"{end}_foot.{side}", foot, 0.7, "extremity", "the lowest point of the leg")
            else:
                det.diag("LEG_NOT_FOUND", "warning", f"no {end} {side} leg")
    del grid, solid_flat
