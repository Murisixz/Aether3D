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

    def score(theta: float, off: float, pts: np.ndarray) -> float:
        n = np.array([math.cos(theta), math.sin(theta), 0.0])
        q = pts - 2.0 * (((pts - c) @ n) - off)[:, None] * n[None]
        return float(occupied(q).mean())

    # The coarse sweep on a fixed sample of the voxels (an eighth: the score is a
    # fraction), the refinement on half of them, the final choice on all.
    coarse, fine = points[::8], points[::2]
    for theta in np.radians(np.arange(0.0, 180.0, 3.0)):
        s = score(theta, 0.0, coarse)
        if s > best[2]:
            best = (float(theta), 0.0, s)
    theta0 = best[0]
    ranked = []
    for theta in theta0 + np.radians(np.arange(-3.0, 3.01, 0.5)):
        for off in np.linspace(-0.04, 0.04, 9) * height:
            ranked.append((score(float(theta), float(off), fine), float(theta), float(off)))
    ranked.sort(key=lambda r: -r[0])
    # The coarse winner first: a candidate replaces it only by scoring better on all the
    # voxels (ties keep the plane the sweep found).
    best = (theta0, 0.0, score(theta0, 0.0, points))
    for _, theta, off in ranked[:6]:
        s = score(theta, off, points)
        if s > best[2] + 1e-12:
            best = (theta, off, s)
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
    components: tuple = ()  # the Reeb pieces the branch is made of


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
        visited: list[int] = []
        while front:
            visited.extend(front)
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
        return Chain(np.array(cents), np.array(sizes), junction, reason, tuple(visited))


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
        det.diag(
            "DETACHED_PARTS",
            "info",
            f"{ncomp - 1} parts do not touch the body; they are not measured",
            parts=int(ncomp - 1),
        )
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
            det.diag(
                "FORWARD_AMBIGUOUS",
                "warning",
                f"the facing is uncertain ({how}); pass forward= to fix it",
                confidence=round(conf_f, 2),
            )
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
        det.diag(
            "ASYMMETRIC_BODY",
            "info",
            f"only {sym:.0%} of the body mirrors onto itself; left and right are measured "
            "separately",
            symmetry=round(sym, 3),
        )
    if template == "blocky":
        _blocky(det, grid, c_centers, solid_flat, keep, height)
    elif template == "quadruped":
        _quadruped(det, grid, c_centers, solid_flat, keep, height)
    else:
        mesh_c = (det.canonical(p), np.asarray(triangles, np.int64))
        _humanoid(det, grid, c_centers, solid_flat, keep, height, template, (to_c, shift), mesh_c)
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
            det.diag(
                "ASYMMETRIC_LANDMARK",
                "info",
                f"{base} differs between sides by {diff:.3f} (kept as measured)",
                landmark=base,
                difference=round(diff, 4),
            )


# --------------------------------------------------------------------------------------- #
# Humanoids
# --------------------------------------------------------------------------------------- #


def _humanoid(det, grid, centers, solid_flat, keep, H, template, frame, mesh_c) -> None:
    from .anatomy import Probe

    to_c, shift = frame
    full = grid.solid  # (nx, ny, nz), provisional frame

    def inside(points: np.ndarray) -> np.ndarray:
        """Whether canonical points lie in the voxelized solid."""
        prov = np.atleast_2d(points) @ to_c + shift
        idx = np.floor((prov - grid.origin) / grid.h).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.array(grid.dims)), axis=1)
        out = np.zeros(len(prov), bool)
        out[ok] = full[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out

    probe = Probe(mesh_c[0], mesh_c[1], inside)
    det.info["_probe"] = probe
    det.info["_mesh_c"] = mesh_c
    # The analysis solid: eroded by a voxel, so limbs that merely touch (a hand on a
    # thigh) separate -- unless that would break a substantial part off the body.
    agrid, a_flat, a_centers, contacts = _analysis_solid(grid, solid_flat, centers, keep)
    det.info["contacts_broken"] = contacts
    unit = grid.h / 3.0
    z = a_centers[:, 2]
    a_depth = V.geodesic(
        agrid, [np.flatnonzero(agrid.surface.transpose(2, 1, 0).reshape(-1))], a_flat
    )[:, 0].astype(np.float64)
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
        # (hanging arms touch the hips: the path from the chest to a hand may then be
        # shorter than the arm)
        cand = [
            x for x in others if sign * a_centers[x[0], 0] > 0.06 * H and g_core[x[0]] > 0.2 * H
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
        det.diag(
            "EXTRA_EXTREMITY",
            "info",
            "a protrusion that is not a head, hand or foot (a tail, a weapon, hair...)",
            position=[round(float(v), 3) for v in det.world(a_centers[x[0]])],
        )
    slice_units = 5  # (a slice is 5/3 voxels thick)
    reeb = Reeb(agrid, a_centers, a_flat, g_units, slice_units)
    debug = det.info.setdefault("_debug", {"chains": {}})
    debug.update(
        centers=centers[keep],
        extremities=[a_centers[e] for e, _ in extremities],
        core=a_centers[core],
    )

    regions: dict[str, Any] = {}

    def branch(voxel: int, label: str) -> Chain:
        ch = reeb.branch(voxel, 0.1 * H)
        debug["chains"][label] = ch
        if label not in regions and ch.components:
            members = np.isin(reeb.comp_of, np.array(ch.components))
            r = float(np.median(np.sqrt(np.maximum(ch.sizes[len(ch.sizes) // 3 :], 1.0))))
            regions[label] = {
                "voxels": a_flat[members],
                "end": ch.centroids[-1],
                "radius": r * grid.h / 1.6,
            }
        return ch

    pts_all = mesh_c[0]
    # Head and neck first (shoulders and clavicles hang from the neck's base).
    if head is not None:
        e, _ = head
        ch = branch(e, "head")
        _head_neck(det, ch, pts_all, probe, H)
    else:
        det.diag("HEAD_NOT_FOUND", "warning", "no head top found; the neck and head use priors")
        det.add("head_top", [0, 0, H], 0.4, "proportion", "the top of the mesh")
        det.add("head", [0, 0, PRIOR["chin"] * H + 0.04 * H], 0.4, "proportion",
                "the skull's base above the chin height prior")  # fmt: skip
        det.add("neck", [0, 0, (PRIOR["shoulder"] + 0.015) * H], 0.4, "proportion", "prior")
    # Legs.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        if side not in foot:
            det.diag("FOOT_NOT_FOUND", "warning", f"no {side} foot tip; the leg uses priors")
            _leg_prior(det, side, sign, H)
            continue
        e, _ = foot[side]
        ch = branch(e, f"leg.{side}")
        _leg(det, side, sign, ch, _tip(a_centers[e], ch, grid.h), centers[keep], probe, H)
    # Arms.
    for side, sign in (("L", 1.0), ("R", -1.0)):
        if side not in hands:
            msg = f"no {side} hand tip (arms touching the body?); priors place the arm"
            det.diag("HAND_NOT_FOUND", "warning", msg)
            _arm_prior(det, side, sign, H)
            continue
        e, _ = hands[side]
        ch = branch(e, f"arm.{side}")
        _arm(det, side, sign, ch, _tip(a_centers[e], ch, grid.h), probe, H)
    _spine(det, probe, H)
    det.info["_regions"] = {
        "parts": regions,
        "torso": a_flat,
        "solid": solid_flat[keep],
        "grid": (grid.origin.copy(), grid.h, grid.dims, to_c, shift),
    }
    sh, wr = det.landmarks.get("shoulder.L"), det.landmarks.get("wrist.L")
    if sh is not None and wr is not None:
        d = wr.position - sh.position
        ang = math.degrees(math.atan2(abs(d[0]), -d[2])) if np.linalg.norm(d) > 0 else 0.0
        det.info["arm_angle"] = round(ang, 1)
        det.info["pose"] = "T-pose" if ang > 70 else ("A-pose" if ang > 20 else "arms down")


def _analysis_solid(grid, solid_flat, centers, keep):
    """``(grid, solid flat indices, canonical centres, contacts broken?)`` of the solid
    the extremities and limbs are measured on (see :func:`_humanoid`)."""
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


def _axis_line(pts: np.ndarray, s: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    """``(point at the limb's end, inward direction)`` of the line fitted to the last
    ``frac`` of a centreline."""
    tail = pts[s >= s[-1] - frac * (s[-1] - s[0])]
    if len(tail) < 3:
        tail = pts[-3:]
    return _fit_line(tail)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(the last point projected on the line, the direction from the first point
    toward the last)`` of the least-squares line through ``points``."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c)
    d = vt[0]
    if d @ (points[-1] - points[0]) < 0:
        d = -d
    return c + ((points[-1] - c) @ d) * d, d


def _tangents(pts: np.ndarray) -> np.ndarray:
    """Unit directions of a polyline at its points (central differences)."""
    d = np.gradient(pts, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(n > 1e-12, d / np.maximum(n, 1e-12), np.array([0.0, 0.0, 1.0]))


# --------------------------------------------------------------------------------------- #
# Head, neck, spine: horizontal sections through the mirror plane
# --------------------------------------------------------------------------------------- #

# The skull's pivot (atlanto-occipital joint) lies about 30 % of the head's height above
# the chin (the rule of thirds puts the nose's base 27 % up; the occipital condyles sit
# just above it); the neck's base (C7) about 1.2 % of the stature under the height where
# the horizontal section widens into the shoulders.
HEAD_PIVOT = 0.30
NECK_BELOW_SHOULDERS = 0.012
# Spine joints lie toward the back: this fraction of the way from a section's centre to
# its back surface (Rigify's metarig and the vertebral column alike sit behind the
# centre of the torso).
SPINE_DEPTH = 0.35


def _head_neck(det, ch: Chain, pts_c: np.ndarray, probe, H: float) -> None:
    from .anatomy import Midline

    chain = ch.centroids
    # The crown: the highest point of the mesh near the head's chain, on the mirror plane.
    near = np.linalg.norm(pts_c[:, :2] - chain[0][:2], axis=1) < 0.12 * H
    z_top = float(pts_c[near, 2].max()) if near.any() else float(chain[0][2])
    step = 0.002 * H
    zs = np.arange(z_top - 0.42 * H, z_top - 0.004 * H, step)
    order = np.argsort(chain[:, 2])
    y0 = np.interp(zs, chain[order, 2], chain[order, 1])
    sec = Midline(probe, H).section(zs, y0)
    hw, front, depth = sec["half_width"], sec["front"], sec["back"] - sec["front"]
    ok = sec["inside"]
    # The head's widest section, then down to where the section widens into the
    # shoulders (2.5 times the narrowest width of the neck above it).
    # (the head's own centreline, from the crown down to where it joins the shoulders: the
    # shoulders are deeper than the head)
    head_band = ok & (zs > max(float(chain[:, 2].min()), z_top - 0.2 * H))
    k_head = int(np.flatnonzero(head_band)[np.argmax(depth[head_band])]) if head_band.any() else (
        len(zs) - 1
    )  # fmt: skip
    k_wide, running = None, np.inf
    for k in range(k_head, -1, -1):
        if not ok[k]:
            continue
        running = min(running, hw[k])
        if hw[k] > 2.5 * running and running < 0.06 * H:
            k_wide = k
            break
    if k_wide is None:
        det.diag("NECK_NOT_FOUND", "warning", "no shoulders under the head: the neck uses priors")
        z_wide = z_top - 0.16 * H
        k_wide = int(np.argmin(np.abs(zs - z_wide)))
        neck_conf = 0.4
    else:
        z_wide = float(zs[k_wide])
        neck_conf = 0.75
    # The throat: the front surface's most retreated point between the shoulders and the
    # head; the chin: the first height above it where the front steps forward fastest.
    lo, hi = k_wide + 2, k_head
    if hi - lo >= 4:
        kk = np.arange(lo, hi)
        kk = kk[ok[kk]]
        k_throat = int(kk[np.argmax(front[kk])]) if kk.size else lo
        win = np.arange(k_throat, min(hi, k_throat + int(0.08 * H / step)))
        fwd = -np.diff(front[win]) if win.size > 1 else np.zeros(0)
        if fwd.size and fwd.max() > 0.0005 * H:
            k_chin = int(win[np.flatnonzero(fwd >= 0.5 * fwd.max())[0]])
            chin_conf, chin_how = 0.8, "where the front of the jaw steps out over the throat"
        else:
            k_chin = int(win[len(win) // 3]) if win.size else k_throat
            chin_conf, chin_how = 0.5, "a third up from the throat (no jaw line measured)"
    else:
        k_chin = min(len(zs) - 1, k_wide + 3)
        chin_conf, chin_how = 0.4, "just above the shoulders (no neck measured)"
    z_chin = float(zs[k_chin])
    z_head = z_chin + HEAD_PIVOT * (z_top - z_chin)
    z_neck = z_wide - NECK_BELOW_SHOULDERS * H
    upper = Midline(probe, H).section(
        np.array([z_head, z_neck, 0.5 * (z_head + z_top)]),
        np.interp([z_head, z_neck, 0.5 * (z_head + z_top)], chain[order, 2], chain[order, 1]),
    )

    def depth_y(i: int, frac: float) -> float:
        c, b = float(upper["centre"][i]), float(upper["back"][i])
        return c + frac * (b - c) if upper["inside"][i] else c

    det.add("head_top", [0.0, depth_y(2, 0.0), z_top], 0.9, "extremity",
            "the top of the head, on the mirror plane")  # fmt: skip
    det.add("chin", [0.0, float(sec["front"][k_chin]), z_chin], chin_conf, "profile", chin_how)
    det.add("head", [0.0, depth_y(0, 0.0), z_head], min(chin_conf, 0.8), "section",
            f"the skull's pivot, {HEAD_PIVOT:.0%} of the head's height above the chin")  # fmt: skip
    det.add("neck", [0.0, depth_y(1, SPINE_DEPTH), z_neck], neck_conf, "section",
            "the neck's base, just under where the section widens into the shoulders")  # fmt: skip
    det.info["_neck"] = {"z_wide": z_wide, "front": float(np.interp(z_neck, zs, front))}


def _spine(det, probe, H: float) -> None:
    from .anatomy import Midline

    lm = det.landmarks
    hips = 0.5 * (lm["hip.L"].position + lm["hip.R"].position)
    neck = lm["neck"].position
    z0, z1 = float(hips[2]), float(neck[2])
    fracs = {"pelvis": None, "spine": 0.27, "chest": 0.52, "upper_chest": 0.76}
    zs = np.array([z0 + 0.03 * H] + [z0 + f * (z1 - z0) for f in (0.27, 0.52, 0.76)])
    sec = Midline(probe, H).section(zs, np.full(zs.size, float(hips[1])))
    notes = {
        "pelvis": "above the hip joints, toward the back of the pelvis's section",
        "spine": "27 % up the torso, toward the back of its section",
        "chest": "52 % up the torso, toward the back of its section",
        "upper_chest": "76 % up the torso, toward the back of its section",
    }
    for i, name in enumerate(fracs):
        c, b = float(sec["centre"][i]), float(sec["back"][i])
        y = c + SPINE_DEPTH * (b - c) if sec["inside"][i] else float(hips[1])
        det.add(name, [0.0, y, float(zs[i])], 0.75 if sec["inside"][i] else 0.4, "section",
                notes[name])  # fmt: skip
    # Clavicles: from beside the sternal notch (the front of the neck's base) to the
    # shoulder joint.
    info = det.info.get("_neck", {})
    front = info.get("front", float(sec["centre"][-1]) - 0.04 * H)
    for side, sign in (("L", 1.0), ("R", -1.0)):
        det.add(f"clavicle.{side}", [sign * 0.012 * H, front + 0.012 * H, z1 - 0.012 * H], 0.6,
                "section", "beside the sternal notch, at the front of the neck's base")  # fmt: skip


# --------------------------------------------------------------------------------------- #
# Legs
# --------------------------------------------------------------------------------------- #

# The knee splits the measured hip-to-ankle length in about equal parts (thigh 0.245 and
# shank 0.246 of the stature: Drillis and Contini, after Winter 2009).
KNEE_FRACTION = 0.50


def _leg(det, side, sign, ch: Chain, toe_tip, centers, probe, H) -> None:
    pts = _smooth(ch.centroids)
    # The ankle: where the centreline rises above the foot (at least the prior height).
    rise = pts[:, 2]
    ankle_z = max(PRIOR["ankle"] * H, float(np.median(rise[: max(2, len(rise) // 6)])) * 1.6)
    upward = np.flatnonzero(rise >= ankle_z)
    k_a = int(upward[0]) if upward.size else len(pts) // 5
    ankle = pts[k_a].copy()
    ankle_conf = 0.8 if upward.size else 0.5
    # The leg's slant in the frontal plane, from its free centreline (thighs that touch
    # down to the knees leave only the shin: the slant is the shin's).
    free = pts[pts[:, 2] >= ankle[2] + 0.04 * H]
    if len(free) >= 6 and free[-1, 2] - ankle[2] > 0.3 * H:  # (the thigh's own slant)
        free = free[free[:, 2] >= 0.5 * (free[0, 2] + free[-1, 2])]
    slope = float(np.polyfit(free[:, 2], free[:, 0], 1)[0]) if len(free) >= 3 else 0.0
    slope = float(np.clip(slope, -0.4, 0.4))
    r_top = float(np.sqrt(max(ch.sizes[-1], 1.0))) * det.info["voxel_size"] / 1.6
    hip, hip_conf, hip_method, hip_how = _hip(det, pts[-1], slope, side, sign, probe, H, r_top)
    # The knee: the strongest bend if the rest pose bends, else the proportion of the
    # measured leg.
    poly = np.vstack([pts[pts[:, 2] < hip[2]], hip[None]])
    sp = _arc(poly)
    k0 = int(np.argmin(np.linalg.norm(poly - ankle, axis=1)))
    lo = sp[k0] + 0.3 * (sp[-1] - sp[k0])
    hi_ = sp[k0] + 0.70 * (sp[-1] - sp[k0])
    # (bends measured on the leg's own centreline: the step from where the thighs part to
    # the hip joint above is no knee)
    measured = sp[-2] if len(sp) > 2 else sp[-1]
    bend_at, bend = _bend(poly[:-1], sp[:-1], lo, min(hi_, measured - 0.03 * H))
    if not lo < measured - 0.03 * H:
        bend = 0.0
    # (a bend away from where a knee can be -- a centreline curving where the thighs
    # merge -- is not one)
    frac = (bend_at - sp[k0]) / max(sp[-1] - sp[k0], 1e-9)
    if abs(frac - KNEE_FRACTION) > 0.15:
        bend = 0.0
    if bend > 6.0:
        knee, knee_conf, knee_method = _at(poly, sp, bend_at), 0.85, "bend"
        knee_how = f"the leg bends {bend:.0f} degrees here"
    else:
        knee = _at(poly, sp, sp[k0] + KNEE_FRACTION * (sp[-1] - sp[k0]))
        knee_conf, knee_method = 0.7, "proportion_of_measured"
        knee_how = "straight leg: halfway along the measured hip-to-ankle length"
    # Heel and ball on the sole.
    foot_vox = centers[(centers[:, 2] < ankle[2]) & (np.abs(centers[:, 0] - ankle[0]) < 0.08 * H)]
    if len(foot_vox):
        back = foot_vox[np.argmax(foot_vox[:, 1])]
        heel = np.array([ankle[0], back[1], 0.0])
    else:
        heel = np.array([ankle[0], ankle[1] + 0.03 * H, 0.0])
    ball = toe_tip + 0.27 * (heel - toe_tip)
    ball[2] = max(ball[2], 0.3 * ankle[2])
    det.add(f"toe_tip.{side}", toe_tip, 0.9, "extremity", "the farthest point of the foot")
    det.add(
        f"ball.{side}", ball, 0.7, "proportion_of_measured", "27 % from the toe tip to the heel"
    )
    det.add(f"heel.{side}", heel, 0.75, "sole", "the back of the foot at the ground")
    det.add(
        f"ankle.{side}", ankle, ankle_conf, "turn", "where the leg's centreline turns into the foot"
    )
    det.add(f"knee.{side}", knee, knee_conf, knee_method, knee_how)
    det.add(f"hip.{side}", hip, hip_conf, hip_method, hip_how)


def _column(probe, x: np.ndarray, zs: np.ndarray, H: float) -> dict[str, np.ndarray]:
    """Front-to-back sections of vertical lines through the body at ``x`` (per height):
    an interior point (the middle of the longest run of the voxel solid along y), and
    from it the exact front, back and outer surfaces."""
    ys = np.arange(-0.35 * H, 0.35 * H, 0.004 * H)
    n = zs.size
    grid = np.stack([np.repeat(x, ys.size), np.tile(ys, n), np.repeat(zs, ys.size)], axis=1)
    ins = probe.inside(grid).reshape(n, ys.size)
    mid = np.full(n, np.nan)
    for i in range(n):
        row = ins[i]
        if not row.any():
            continue
        edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        k = int(np.argmax(ends - starts))
        mid[i] = 0.5 * (ys[starts[k]] + ys[ends[k] - 1])
    ok = np.isfinite(mid)
    pts = np.stack([x, np.where(ok, mid, 0.0), zs], axis=1)
    most = 0.5 * H
    back = probe.reach(pts, np.broadcast_to(np.array([0.0, 1.0, 0.0]), pts.shape), most)
    front = probe.reach(pts, np.broadcast_to(np.array([0.0, -1.0, 0.0]), pts.shape), most)
    ok &= (back > 0) & (front > 0)
    return {"ok": ok, "points": pts, "back": pts[:, 1] + back, "front": pts[:, 1] - front}


# Where the legs part high enough to show a crotch, the hip joint lies about this many
# thigh radii above it (a femoral head is about a thigh's radius above the perineum).
CROTCH_TO_HIP = 0.8


def _hip(det, top, slope: float, side, sign, probe, H, r_thigh: float = 0.0):
    """The hip joint, from up to three measured cues: the height of the buttock's most
    prominent part (the gluteal mass covers the femoral head); just under where the hip
    is widest from the leg's axis (the greater trochanter); and, when the legs part high
    enough to show it, the crotch plus :data:`CROTCH_TO_HIP` thigh radii. The buttock
    decides when it is there, else the median of the others; across, the hip lies on the
    leg's slant from its free centreline, in depth in the middle of the body's section."""
    zs = np.arange(max(0.36 * H, float(top[2]) - 0.02 * H), 0.62 * H, 0.004 * H)
    x = float(top[0]) + slope * (zs - float(top[2]))
    col = _column(probe, x, zs, H) if zs.size else None
    cues = {}
    if col is not None and col["ok"].sum() >= 6:
        ok = col["ok"]
        y_back = np.where(ok, col["back"], -np.inf)
        k = int(np.argmax(y_back))
        top_k = np.flatnonzero(ok & (y_back >= y_back[k] - 0.0025 * H))
        above = ok & (zs > zs[k])
        # A buttock: the back surface recedes again above its most prominent part.
        if above.any() and y_back[above].min() < y_back[k] - 0.008 * H:
            cues["buttock"] = float(zs[top_k].mean())
        pts = col["points"]
        lateral = probe.reach(pts, np.broadcast_to(np.array([sign, 0.0, 0.0]), pts.shape), 0.5 * H)
        good = ok & (lateral > 0)
        if good.any():  # (a hand or forearm against the hip is not the hip's width)
            good &= lateral < 1.6 * float(np.median(lateral[good]))
        # (measured from the leg's slanted axis: a wide stance slants the thighs out)
        x_lat = np.where(good, lateral, -np.inf)
        k2 = int(np.argmax(x_lat))
        top2 = np.flatnonzero(good & (x_lat >= x_lat[k2] - 0.002 * H))
        if top2.size:
            cues["trochanter"] = float(zs[top2].min()) - 0.01 * H
    if float(top[2]) > 0.40 * H and r_thigh > 0:
        cues["crotch"] = float(top[2]) + CROTCH_TO_HIP * r_thigh
    if "buttock" in cues:
        z = cues["buttock"]
        conf, method = 0.75, "buttock"
        how = "level with the buttock's most prominent part, on the leg's slant"
    elif cues:
        z = float(np.median(list(cues.values())))
        conf, method = 0.6, "+".join(sorted(cues))
        how = "between where the hip is widest and the crotch below it, on the leg's slant"
    else:
        z = PRIOR["hip"] * H
        conf, method = 0.35, "proportion"
        how = "the stature's proportion (no hip measured)"
        det.diag("HIP_NOT_MEASURED", "warning", f"the {side} hip uses a proportion prior")
    xh = float(top[0]) + slope * (z - float(top[2]))
    if col is not None and zs.size:
        k = int(np.argmin(np.abs(zs - z)))
        yh = 0.5 * (col["front"][k] + col["back"][k]) if col["ok"][k] else float(top[1])
    else:
        yh = float(top[1])
    det.info.setdefault("_hip_cues", {})[side] = {k_: round(v, 4) for k_, v in cues.items()}
    return np.array([xh, yh, z]), conf, method, how


def plateau_start_index(values: np.ndarray, fraction: float) -> int:
    from .anatomy import plateau_start

    return plateau_start(values, fraction)


def _leg_prior(det, side, sign, H) -> None:
    x = sign * PRIOR["hip_half_width"] * H
    det.add(f"hip.{side}", [x, 0, PRIOR["hip"] * H], 0.3, "proportion", "prior")
    det.add(f"knee.{side}", [x, 0, PRIOR["knee"] * H], 0.3, "proportion", "prior")
    det.add(f"ankle.{side}", [x, 0, PRIOR["ankle"] * H], 0.3, "proportion", "prior")
    det.add(f"heel.{side}", [x, 0.03 * H, 0], 0.3, "proportion", "prior")
    det.add(f"ball.{side}", [x, -0.08 * H, 0.01 * H], 0.3, "proportion", "prior")
    det.add(f"toe_tip.{side}", [x, -0.12 * H, 0.01 * H], 0.3, "proportion", "prior")


# --------------------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------------------- #

# The elbow splits the shoulder-to-wrist length as the upper arm (0.186 of the stature)
# and the forearm (0.146) do.
ELBOW_FRACTION = 0.186 / (0.186 + 0.146)
# Raised less than about 40 degrees from hanging, the upper arm's axis leaves the body
# over the shoulder: the joint lies SHOULDER_EXIT arm radii before that point. Raised
# more, the axis runs on into the chest and the joint lies above the armpit (where the
# reach under the arm jumps to the torso), by ARMPIT[0] - ARMPIT[1] (angle - 35) arm
# radii along the axis (measured on both of Aether3D's test characters from 35 to 88
# degrees: 1.26 radii in an A-pose, about none at 72 degrees, -0.4 in a T-pose).
SHOULDER_EXIT = 1.6
ARMPIT = (1.26, 0.031)
EXIT_POSE = (38.0, 50.0)


def _arm(det, side, sign, ch: Chain, tip, probe, H) -> None:
    pts = _smooth(ch.centroids)
    # The hand's tip, measured on the mesh: thin fingers vanish from the voxel solid the
    # centreline comes from, so the farthest vertex beyond the centreline's end, along
    # the hand's direction, is the tip.
    out = pts[0] - pts[min(3, len(pts) - 1)]
    out = out / max(float(np.linalg.norm(out)), 1e-12)
    mesh = det.info["_mesh_c"][0]
    rel = mesh - pts[0]
    along = rel @ out
    across = np.linalg.norm(rel - along[:, None] * out[None], axis=1)
    # (within a cylinder about the hand's axis: not a thigh the hand hangs beside)
    near = (along > -0.02 * H) & (along < 0.16 * H) & (across < 0.045 * H)
    tip_how = "the farthest point of the hand"
    if near.any() and along[near].max() > 0:
        tip = mesh[near][int(np.argmax(along[near]))].copy()
        tip_how = "the farthest point of the hand along its direction"
    # The hand and the arm as one polyline from the tip, resampled finely; the section
    # radii along it (mean and widest reach around the local direction).
    poly = np.vstack([tip[None], pts])
    sp = _arc(poly)
    step = 0.004 * H
    s = np.arange(0.0, float(sp[-1]), step)
    line = np.stack([np.interp(s, sp, poly[:, i]) for i in range(3)], axis=1)
    tan = _tangents(_smooth(line, 3))
    most = 0.2 * H
    radius = probe.radius(line, tan, np.array([0.0, 0.0, 1.0]), most)
    widest = _widest(probe, line, tan, most)
    # The wrist: the narrowest section after the palm (the hand's widest part), going in.
    w_conf, w_how = 0.5, "a hand's length from the tip (prior)"
    k_w = int(np.argmin(np.abs(s - PRIOR["hand"] * H)))
    palm = np.flatnonzero((s > 0.02 * H) & (s < 0.10 * H) & (radius > 0))
    if palm.size >= 3:
        k_palm = int(palm[np.argmax(widest[palm])])
        win = np.flatnonzero((s > s[k_palm] + 0.008 * H) & (s < s[k_palm] + 0.07 * H)
                             & (radius > 0))  # fmt: skip
        if win.size >= 3:
            k = int(win[np.argmin(radius[win])])
            if radius[k] < 0.95 * radius[win].max():
                k_w, w_conf = k, 0.8
                w_how = "the narrowest section between the palm and the forearm"
    wrist = line[k_w].copy()
    # The upper arm's axis (through the centreline between the wrist's side and the
    # armpit), refined once the elbow is known.
    k_end = len(line) - 1
    rest = s[k_end] - s[k_w]
    if rest >= 0.20 * H:
        sel = s >= s[k_w] + 0.55 * rest
    else:
        # A short free arm (hanging, its upper arm against the chest): the forearm's
        # axis, extended, stands for the upper arm's (a nearly straight arm).
        sel = s >= s[k_w] + 0.15 * rest
        det.diag("ARM_AGAINST_BODY", "info", f"the {side} upper arm touches the body; its "
                 "axis continues the forearm's", side=side)  # fmt: skip
    fused = rest < 0.20 * H
    a0, d = _fit_line(line[sel] if sel.sum() >= 3 else line[-3:])
    shoulder, s_conf, s_how, r_arm = _shoulder_on_axis(det, side, line, s, sel, a0, d, probe,
                                                        radius, H, fused)  # fmt: skip
    arm = np.vstack([line[k_w:], shoulder[None]])
    sa = _arc(arm)
    bend_at, bend = _bend(arm, sa, 0.3 * sa[-1], 0.7 * sa[-1])
    if abs(bend_at / max(sa[-1], 1e-9) - (1.0 - ELBOW_FRACTION)) > 0.15:
        bend = 0.0
    if bend > 8.0:
        elbow, e_conf, e_method = _at(arm, sa, bend_at), 0.85, "bend"
        e_how = f"the arm bends {bend:.0f} degrees here"
    else:
        elbow = _at(arm, sa, (1.0 - ELBOW_FRACTION) * sa[-1])
        e_conf, e_method = 0.7, "proportion_of_measured"
        e_how = "straight arm: at the anatomical fraction of the measured arm"
    # Refit the upper arm's axis between the elbow and the armpit, and place the
    # shoulder again on it.
    k_e = int(np.argmin(np.linalg.norm(line - elbow, axis=1)))
    sel2 = np.arange(len(line)) >= k_e + max(1, (k_end - k_e) // 5)
    if sel2.sum() >= 3:
        a0, d = _fit_line(np.vstack([elbow[None], line[sel2]]))
        shoulder, s_conf, s_how, r_arm = _shoulder_on_axis(det, side, line, s, sel2, a0, d,
                                                            probe, radius, H, fused)  # fmt: skip
        if bend <= 8.0:
            arm = np.vstack([line[k_w:], shoulder[None]])
            sa = _arc(arm)
            elbow = _at(arm, sa, (1.0 - ELBOW_FRACTION) * sa[-1])
    det.add(f"hand_tip.{side}", tip, 0.9, "extremity", tip_how)
    det.add(f"wrist.{side}", wrist, w_conf, "narrowest", w_how)
    det.add(f"elbow.{side}", elbow, e_conf, e_method, e_how)
    det.add(f"shoulder.{side}", shoulder, s_conf, "axis", s_how)
    det.info.setdefault("_arm_radius", {})[side] = r_arm
    det.info.setdefault("_hand", {})[side] = {"line": line[: k_w + 1], "radius": radius[: k_w + 1]}


def _widest(probe, points, axes, most, n: int = 8) -> np.ndarray:
    """The widest reach of each point around its axis (a flat palm's half width)."""
    a = axes / np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-12)
    ref = np.array([0.0, 0.0, 1.0])
    u = ref - (a @ ref)[:, None] * a
    bad = np.linalg.norm(u, axis=1) < 1e-6
    u[bad] = np.cross(a[bad], np.array([1.0, 0.0, 0.0]))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(a, u)
    best = np.zeros(points.shape[0])
    for k in range(n):
        ang = np.pi * k / n  # (opposite directions together: the full width)
        dd = np.cos(ang) * u + np.sin(ang) * v
        both = probe.reach(points, dd, most) + probe.reach(points, -dd, most)
        best = np.maximum(best, both)
    return 0.5 * best


def _shoulder_on_axis(det, side, pts, s, sel, a0, d, probe, radius, H, fused=False):
    """The shoulder joint on the upper arm's axis (``a0``: a point of it near the armpit,
    ``d``: toward the torso). ``fused``: the upper arm lies against the chest, so its
    radius is measured from the axis outward, forward and back (away from the torso)."""
    r_sel = radius[sel]
    r_arm = float(np.median(r_sel[r_sel > 0])) if np.any(r_sel > 0) else 0.03 * H
    # From the elbow's side: hanging = 0 degrees, T-pose = 90.
    angle = float(np.degrees(np.arccos(np.clip(d[2], -1.0, 1.0))))
    t = np.arange(-0.05 * H, 0.35 * H, 0.002 * H)
    line = a0[None] + t[:, None] * d[None]
    inside = probe.inside(line)
    first_in = np.flatnonzero(inside)
    exit_pt = None
    if first_in.size:
        after = np.flatnonzero(~inside & (np.arange(t.size) > first_in[0]))
        if after.size:
            exit_pt = line[after[0]]
            if fused:
                sign = 1.0 if pts[0][0] > 0 else -1.0
                out = np.array([sign, 0.0, 0.0]) - sign * d[0] * d
                out /= max(float(np.linalg.norm(out)), 1e-12)
                fwd = np.cross(d, out)
                band = (t > t[after[0]] - 0.14 * H) & (t < t[after[0]] - 0.05 * H)
                q = line[band]
                if len(q):
                    r3 = np.stack([probe.reach(q, np.broadcast_to(v, q.shape), 0.3 * H)
                                   for v in (out, fwd, -fwd)], axis=1)  # fmt: skip
                    good = np.all(r3 > 0, axis=1)
                    if good.any():
                        r_arm = float(np.median(r3[good].mean(axis=1)))
    down = np.array([0.0, 0.0, -1.0]) - (-d[2]) * d
    nd = float(np.linalg.norm(down))
    armpit = None
    if nd > 1e-6:
        down /= nd
        reach = probe.reach(line, np.broadcast_to(down, line.shape), 0.5 * H)
        start = np.flatnonzero(inside)
        if start.size:
            jump = np.flatnonzero((np.arange(t.size) >= start[0]) & (reach > 2.5 * r_arm))
            if jump.size:
                armpit = line[jump[0]]
    by_exit = None if exit_pt is None else exit_pt - SHOULDER_EXIT * r_arm * d
    if armpit is not None:
        armpit = armpit + (ARMPIT[0] - ARMPIT[1] * (angle - 35.0)) * r_arm * d
    lo, hi = EXIT_POSE
    w = 0.0 if angle <= lo else 1.0 if angle >= hi else (angle - lo) / (hi - lo)
    if by_exit is not None and armpit is not None:
        w = w * w * (3 - 2 * w)
        p = (1 - w) * by_exit + w * armpit
        how = (
            f"on the upper arm's axis, {SHOULDER_EXIT:g} arm radii inside where it leaves the "
            "shoulder"
            if w < 0.5
            else "on the upper arm's axis above the armpit"
        )
        return p, 0.75, how, r_arm
    if by_exit is not None and w < 1.0:
        return by_exit, 0.7, (f"on the upper arm's axis, {SHOULDER_EXIT:g} arm radii inside "
                              "where it leaves the shoulder"), r_arm  # fmt: skip
    if armpit is not None:
        return armpit, 0.65, "on the upper arm's axis above the armpit", r_arm
    det.diag("SHOULDER_NOT_MEASURED", "warning", f"the {side} shoulder is at the arm's end")
    return pts[-1].copy(), 0.4, "the end of the arm's centreline", r_arm


def _arm_prior(det, side, sign, H) -> None:
    x = sign * PRIOR["shoulder_half_width"] * H
    z = PRIOR["shoulder"] * H
    det.add(f"shoulder.{side}", [x, 0, z], 0.3, "proportion", "prior (arms down)")
    det.add(f"elbow.{side}", [x, 0, z - PRIOR["upper_arm"] * H], 0.3, "proportion", "prior")
    det.add(
        f"wrist.{side}",
        [x, 0, z - (PRIOR["upper_arm"] + PRIOR["forearm"]) * H],
        0.3,
        "proportion",
        "prior",
    )
    det.add(
        f"hand_tip.{side}",
        [x, 0, z - (PRIOR["upper_arm"] + PRIOR["forearm"] + PRIOR["hand"]) * H],
        0.3,
        "proportion",
        "prior",
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
            det.add(
                f"shoulder.{side}",
                [ax, 0, body_top - 0.05 * H],
                0.8,
                "block",
                "the top of the arm block, on its axis",
            )
            det.add(
                f"hand_tip.{side}",
                [ax, 0, float(z[arm].min())],
                0.8,
                "block",
                "the bottom of the arm block",
            )
        else:
            det.add(
                f"shoulder.{side}",
                [sign * torso_w * 1.25, 0, body_top - 0.05 * H],
                0.3,
                "proportion",
                "no arm block found",
            )
            det.add(
                f"hand_tip.{side}",
                [sign * torso_w * 1.25, 0, leg_top],
                0.3,
                "proportion",
                "no arm block found",
            )
        leg = legs & (sign * x > 0)
        lx = float(np.median(x[leg])) if leg.any() else sign * 0.5 * torso_w
        det.add(
            f"hip.{side}",
            [lx, 0, leg_top],
            0.8 if leg.any() else 0.3,
            "block",
            "the top of the leg block",
        )
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
        det.add(
            "head",
            hp.mean(axis=0) + [0, 0.25 * (hp[:, 1].max() - hp[:, 1].min()), 0],
            0.6,
            "body",
            "the head's back",
        )
        det.add("head_top", hp[np.argmin(hp[:, 1])], 0.7, "extremity", "the snout")
    else:
        det.add("head", [0, front, body_z + 0.2 * H], 0.3, "proportion", "no head found")
        det.add("head_top", [0, front - 0.1 * H, body_z + 0.2 * H], 0.3, "proportion", "prior")
    det.add("neck", [0, front + 0.18 * length, body_z + 0.1 * H], 0.5, "body", "the trunk's front")
    tail = keep & (y > back - 0.12 * length) & (z > 0.5 * body_z)
    if tail.any():
        tp = centers[tail]
        det.add("tail_tip", tp[np.argmax(tp[:, 1])], 0.6, "extremity", "the tail's tip")
    for end, (y0, y1) in (
        ("front", (front, front + 0.45 * length)),
        ("back", (front + 0.55 * length, back)),
    ):
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
