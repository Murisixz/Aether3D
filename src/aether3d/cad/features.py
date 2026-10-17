"""Part features: an ordered list regenerated into named bodies.

A part's definition is ``{"features": [...], "materials": {...}, "color": ...}``. Every
feature has a unique ``id`` (its faces are named after it) and a ``type``; features may
be ``"suppressed": true``. Regeneration replays the features in order on a set of named
bodies (``"main"`` by default). Each feature's output is cached by a key over its own
definition, everything it reads (sketch and frame results) and the previous feature's key,
so changing feature *k* re-runs features *k* onwards only, and changing a sketch re-runs
the features from the first one that uses it.

Feature types (lengths: numbers in the document's unit, quantity text or ``"=expr"``):

==============  ============================================================================
``extrude``     ``sketch``, ``regions`` (``"auto"`` or points), ``extent`` (``{"type":
                "blind"|"symmetric"|"two_sided"|"through_all", "distance"|"d1"/"d2"}``),
                ``reverse``, ``draft``, ``operation`` (``new``/``add``/``cut``/
                ``intersect``), ``body``
``revolve``     ``sketch``, ``regions``, ``axis`` (a sketch line id, or ``{"point":
                [x, y], "direction": [dx, dy]}`` in the sketch), ``angle``, ``operation``
``sweep``       ``profile`` (sketch), ``path`` (sketch of lines and arcs, tangent at its
                joints), ``operation``
``loft``        ``sections`` (2+ sketches with matching loops), ruled between them
``hole``        ``face``, ``points`` ([[x, y, z], ...]) or ``sketch`` + ``point_ids``,
                ``diameter``, ``depth`` (``"through"`` or a length), ``kind`` (``simple``,
                ``counterbore`` + ``cbore_diameter``/``cbore_depth``, ``countersink`` +
                ``csink_diameter``/``csink_angle``), ``tip_angle`` (drill point)
``fillet``      ``edges`` (references), ``radius`` -- straight edges between two planar
                faces, not meeting each other
``chamfer``     ``edges``, ``distance`` (and optional ``distance2``) -- same class
``shell``       ``feature`` (an extrude or revolve), ``thickness``, ``open`` (caps
                removed: ``"start"``, ``"end"``) -- an exact profile offset
``pattern``     ``features``, ``kind`` (``linear``: ``direction``, ``count``,
                ``spacing``; ``circular``: ``axis`` {``point``, ``direction``},
                ``count``, ``angle``)
``mirror``      ``features`` or ``body``, ``plane`` {``point``, ``normal``}
``transform``   ``body``, ``translate``, ``rotate`` {``axis``, ``angle``, ``point``}
``boolean``     ``op`` (``union``/``cut``/``intersect``), ``target``, ``tool``
``box``, ``cylinder``, ``sphere``, ``cone``, ``torus``  primitives at ``frame``
==============  ============================================================================

Failures name the feature (``part[Bracket].features[3](fillet1)``) and leave the
document unchanged; unsupported configurations are refused with the condition that failed,
never approximated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import CadError
from .brep import build
from .brep.boolean import boolean
from .brep.topology import Body, Face
from .document import content_key
from .geom2d import Arc, Segment
from .geom3d import Circle3, Frame, Line3, Plane
from .naming import resolve_edge, resolve_face
from .sketch.profile import Loop, Piece, Region
from .units import ANGLE, LENGTH, RATIO, unit

MAX_FEATURES = 2000
MAX_PATTERN = 1000
OPERATIONS = ("new", "add", "cut", "intersect")
FEATURE_TYPES = (
    "extrude",
    "revolve",
    "sweep",
    "loft",
    "hole",
    "fillet",
    "chamfer",
    "shell",
    "pattern",
    "mirror",
    "transform",
    "boolean",
    "box",
    "cylinder",
    "sphere",
    "cone",
    "torus",
)


@dataclass
class FeatureOutput:
    bodies: dict[str, Body]
    info: dict[str, Any] = field(default_factory=dict)
    tool: Body | None = None  # for features that add or cut a tool (patterns, mirrors)
    op: str | None = None
    target: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # region/frame/extent for shells


@dataclass
class PartResult:
    bodies: dict[str, Body]
    features: list[dict[str, Any]]
    outputs: dict[str, FeatureOutput]

    def body(self, name: str = "main") -> Body:
        if name not in self.bodies:
            raise CadError(
                "UNKNOWN_BODY", f"the part has no body {name!r}", hint=", ".join(self.bodies)
            )
        return self.bodies[name]


class _Part:
    """Evaluation helpers for one part regeneration."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.doc = ctx.doc
        self.scale = float(unit(self.doc.length_unit).factor)

    def length(self, raw, loc) -> float:
        return self.ctx.length(raw, loc)

    def angle(self, raw, loc) -> float:
        return self.ctx.angle(raw, loc)

    def number(self, raw, loc) -> float:
        return self.ctx.value(raw, RATIO, loc)

    def count(self, raw, loc, lo=1, hi=MAX_PATTERN) -> int:
        v = self.number(raw, loc)
        if v != int(v) or not lo <= v <= hi:
            raise CadError(
                "INVALID_VALUE", f"must be a whole number {lo}..{hi}", f"{self.ctx.location}.{loc}"
            )
        return int(v)

    def point(self, raw, loc) -> np.ndarray:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise CadError("INVALID_VALUE", "a point is [x, y, z]", f"{self.ctx.location}.{loc}")
        return np.array([self.length(v, f"{loc}[{i}]") for i, v in enumerate(raw)])

    def direction(self, raw, loc) -> np.ndarray:
        if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
            raise CadError(
                "INVALID_VALUE", "a direction is [dx, dy, dz]", f"{self.ctx.location}.{loc}"
            )
        v = np.array([self.number(x, f"{loc}[{i}]") for i, x in enumerate(raw)], dtype=np.float64)
        n = float(np.linalg.norm(v))
        if not n > 0:
            raise CadError(
                "INVALID_VALUE", "a direction cannot be zero", f"{self.ctx.location}.{loc}"
            )
        return v / n

    def sketch(self, ref, loc):
        item = self.doc.item(ref, "sketch")
        return self.ctx.inputs[item.id]

    def frame(self, raw, loc) -> Frame:
        if raw is None:
            return Frame.world()
        if isinstance(raw, str):
            item = self.doc.item(raw, ("frame", "sketch"))
            res = self.ctx.inputs[item.id]
            return res.frame if hasattr(res, "frame") else res
        if isinstance(raw, dict):
            origin = self.point(raw.get("origin", [0, 0, 0]), f"{loc}.origin")
            z = self.direction(raw.get("z", [0, 0, 1]), f"{loc}.z")
            x = raw.get("x")
            return Frame.from_axes(origin, z, None if x is None else self.direction(x, f"{loc}.x"))
        raise CadError(
            "INVALID_VALUE",
            "a frame is a frame item name or {origin, z, x}",
            f"{self.ctx.location}.{loc}",
        )


def references(doc, defn: dict) -> list[str]:
    """Items a part definition reads: sketches and frames (and parameters by name)."""
    refs: list[str] = []
    for f in defn.get("features", []):
        if not isinstance(f, dict):
            continue
        for key in ("sketch", "profile", "path"):
            v = f.get(key)
            if isinstance(v, str):
                refs.append(doc.item(v, "sketch").id)
        for v in f.get("sections", []) or []:
            if isinstance(v, str):
                refs.append(doc.item(v, "sketch").id)
        fr = f.get("frame")
        if isinstance(fr, str):
            refs.append(doc.item(fr, ("frame", "sketch")).id)
    refs += parameter_refs(doc, defn)
    return refs


def parameter_refs(doc, value: Any) -> list[str]:
    """Parameter items named in ``"=expression"`` strings anywhere in ``value``."""
    from .expr import Expression

    out: list[str] = []

    def walk(v):
        if isinstance(v, str) and v.strip().startswith("="):
            for name in Expression(v.strip()[1:]).names:
                item = doc._by_name.get(name)
                if item is None or item.kind != "parameter":
                    raise CadError("UNKNOWN_PARAMETER", f"unknown parameter {name!r} in {v!r}")
                out.append(item.id)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(value)
    return out


def check(doc, defn: dict, location: str) -> None:
    feats = defn.get("features", [])
    if not isinstance(feats, list):
        raise CadError("INVALID_PART", "features must be a list", location)
    if len(feats) > MAX_FEATURES:
        raise CadError("LIMIT_EXCEEDED", f"more than {MAX_FEATURES} features", location)
    seen: set[str] = set()
    for k, f in enumerate(feats):
        loc = f"{location}.features[{k}]"
        if not isinstance(f, dict):
            raise CadError("INVALID_FEATURE", "a feature must be an object", loc)
        fid = f.get("id")
        if not isinstance(fid, str) or not fid or len(fid) > 64 or any(c in fid for c in ":|#/ "):
            raise CadError("INVALID_FEATURE", "a feature needs an id (no ':|#/' or spaces)", loc)
        if fid in seen:
            raise CadError("DUPLICATE_ID", f"feature id {fid!r} is used twice", loc)
        seen.add(fid)
        t = f.get("type")
        if t not in FEATURE_TYPES:
            raise CadError(
                "INVALID_FEATURE", f"unknown feature type {t!r}", loc, hint=", ".join(FEATURE_TYPES)
            )
        op = f.get("operation")
        if op is not None and op not in OPERATIONS:
            raise CadError("INVALID_FEATURE", f"operation must be one of {OPERATIONS}", loc)
        for key in ("features",):
            refs = f.get(key)
            if refs is not None and (
                not isinstance(refs, list)
                or not all(isinstance(r, str) and r in seen for r in refs)
            ):
                raise CadError(
                    "INVALID_FEATURE",
                    f"{key} must list ids of earlier features",
                    f"{loc}.{key}",
                )
    references(doc, defn)


# --------------------------------------------------------------------------- #
# Regeneration
# --------------------------------------------------------------------------- #


def regenerate(ctx, defn: dict, cache) -> PartResult:
    part = _Part(ctx)
    bodies: dict[str, Body] = {}
    prev = "start"
    infos = []
    outputs: dict[str, FeatureOutput] = {}
    input_keys = {r: ctx.doc._keys.get(r) for r in ctx.item._refs}
    for k, f in enumerate(defn.get("features", [])):
        loc = f"features[{k}]({f['id']})"
        if f.get("suppressed"):
            infos.append({"id": f["id"], "type": f["type"], "suppressed": True})
            continue
        ctx.check()
        key = content_key(prev, f, input_keys, ctx.doc.length_unit, ctx.doc._settings["angle_unit"])
        hit = cache.get(key)
        if hit is None:
            try:
                out = FEATURES[f["type"]](part, f, dict(bodies), outputs, loc)
            except CadError as exc:
                where = f"{ctx.location}.{loc}"
                if exc.location and exc.location.startswith(loc):
                    where = f"{ctx.location}.{exc.location}"
                elif exc.location and not exc.location.startswith(ctx.location):
                    where = f"{where}.{exc.location}"
                raise CadError(
                    exc.code, exc.message, where, hint=exc.hint, details=exc.details
                ) from None
            cache[key] = out
            if len(cache) > 4096:
                cache.pop(next(iter(cache)))
        else:
            out = hit
        bodies = out.bodies
        outputs[f["id"]] = out
        prev = key
        infos.append({"id": f["id"], "type": f["type"], **out.info})
    return PartResult(bodies, infos, outputs)


def _apply(state: dict[str, Body], target: str, tool: Body, op: str, loc: str) -> dict[str, Body]:
    out = dict(state)
    if op == "new" or (op == "add" and target not in state):
        if op == "new" and target in state:
            raise CadError(
                "BODY_EXISTS",
                f"the part already has a body {target!r}",
                loc,
                hint='use "operation": "add" or another "body" name',
            )
        out[target] = tool
        return out
    if target not in state:
        raise CadError("NO_BODY", f"there is no body {target!r} to {op}", loc)
    kind = {"add": "union", "cut": "difference", "intersect": "intersection"}[op]
    try:
        out[target] = boolean(state[target], tool, kind)
    except CadError as exc:
        if exc.code == "EMPTY_RESULT" and op == "cut":
            raise CadError("EMPTY_RESULT", "the cut removes the whole body", loc) from None
        raise
    return out


def _regions(part: _Part, sk, select, loc) -> list[Region]:
    if select in (None, "auto"):
        regions = sk.regions(None)
    else:
        if not isinstance(select, list) or not select:
            raise CadError(
                "INVALID_VALUE", 'regions must be "auto" or a list of [x, y] points', loc
            )
        pts = []
        for i, p in enumerate(select):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise CadError("INVALID_VALUE", "a region point is [x, y] (sketch)", f"{loc}[{i}]")
            pts.append([part.length(v, f"{loc}[{i}]") for v in p])
        regions = sk.regions(pts)
    if not regions:
        raise CadError("EMPTY_PROFILE", "the sketch has no closed region", loc)
    return regions


def _extent(part: _Part, f: dict, loc: str, size: float) -> tuple[float, float]:
    ext = f.get("extent", {"type": "blind", "distance": f.get("distance", 10)})
    if not isinstance(ext, dict):
        raise CadError("INVALID_VALUE", "extent must be an object", f"{loc}.extent")
    t = ext.get("type", "blind")
    if t == "blind":
        d = part.length(ext.get("distance"), f"{loc}.extent.distance")
        if d == 0:
            raise CadError("INVALID_VALUE", "the distance cannot be zero", f"{loc}.extent.distance")
        lo, hi = (0.0, d) if d > 0 else (d, 0.0)
    elif t == "symmetric":
        d = part.length(ext.get("distance"), f"{loc}.extent.distance")
        if d <= 0:
            raise CadError(
                "INVALID_VALUE", "the distance must be positive", f"{loc}.extent.distance"
            )
        lo, hi = -d / 2, d / 2
    elif t == "two_sided":
        d1 = part.length(ext.get("d1"), f"{loc}.extent.d1")
        d2 = part.length(ext.get("d2"), f"{loc}.extent.d2")
        lo, hi = -d2, d1
        if hi <= lo:
            raise CadError("INVALID_VALUE", "the two sides leave no thickness", f"{loc}.extent")
    elif t == "through_all":
        lo, hi = -size, size
    else:
        raise CadError("INVALID_VALUE", f"unknown extent type {t!r}", f"{loc}.extent.type")
    if f.get("reverse"):
        lo, hi = -hi, -lo
    return lo, hi


def _state_size(state: dict[str, Body], frame: Frame | None = None) -> float:
    size = 1.0
    for b in state.values():
        lo, hi = b.bounds
        size = max(
            size, float(np.linalg.norm(hi - lo)) + float(np.max(np.abs(np.concatenate([lo, hi]))))
        )
    if frame is not None:
        size += float(np.linalg.norm(frame.origin))
    return 2.0 * size


def f_extrude(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    sk = part.sketch(f.get("sketch"), f"{loc}.sketch")
    regions = _regions(part, sk, f.get("regions"), f"{loc}.regions")
    lo, hi = _extent(part, f, loc, _state_size(state, sk.frame))
    draft = part.angle(f.get("draft", 0), f"{loc}.draft") if f.get("draft") else 0.0
    group = int(f.get("group", 0))
    tool = None
    for i, reg in enumerate(regions):
        b = build.extrude(reg, sk.frame, lo, hi, draft=draft, prefix=f["id"], group=group)
        if i:
            b = b.renamed({f["id"]: f["id"]})
        tool = b if tool is None else boolean(tool, b, "union")
    op = f.get("operation", "new" if not state else "add")
    target = f.get("body", "main")
    out = _apply(state, target, tool, op, loc)
    return FeatureOutput(
        out,
        {"operation": op, "body": target, "regions": len(regions), "extent_m": [lo, hi]},
        tool,
        op,
        target,
        {"regions": regions, "frame": sk.frame, "start": lo, "end": hi, "kind": "extrude"},
    )


def _axis_2d(part: _Part, sk, raw, loc):
    if isinstance(raw, str):
        e = sk.entity(raw)
        if e is None or e["type"] != "line":
            raise CadError("INVALID_VALUE", f"{raw!r} is not a line of the sketch", loc)
        a = np.asarray(sk.point2(e["start"]))
        b = np.asarray(sk.point2(e["end"]))
        return a, b - a
    if isinstance(raw, dict):
        p = raw.get("point", [0, 0])
        d = raw.get("direction", [1, 0])
        pt = np.array([part.length(p[0], f"{loc}.point[0]"), part.length(p[1], f"{loc}.point[1]")])
        dv = np.array(
            [part.number(d[0], f"{loc}.direction[0]"), part.number(d[1], f"{loc}.direction[1]")]
        )
        return pt, dv
    raise CadError("INVALID_VALUE", "axis is a sketch line id or {point, direction}", loc)


def f_revolve(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    sk = part.sketch(f.get("sketch"), f"{loc}.sketch")
    regions = _regions(part, sk, f.get("regions"), f"{loc}.regions")
    ap, ad = _axis_2d(part, sk, f.get("axis"), f"{loc}.axis")
    angle = part.angle(f.get("angle", "360 deg"), f"{loc}.angle")
    tool = None
    for reg in regions:
        b = build.revolve(
            reg, sk.frame, ap, ad, angle, prefix=f["id"], group=int(f.get("group", 0))
        )
        tool = b if tool is None else boolean(tool, b, "union")
    op = f.get("operation", "new" if not state else "add")
    target = f.get("body", "main")
    return FeatureOutput(
        _apply(state, target, tool, op, loc),
        {"operation": op, "body": target},
        tool,
        op,
        target,
        {
            "regions": regions,
            "frame": sk.frame,
            "axis": (ap, ad),
            "angle": angle,
            "kind": "revolve",
        },
    )


def _primitive(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    t = f["type"]
    frame = part.frame(f.get("frame"), f"{loc}.frame")
    g = int(f.get("group", 0))
    if t == "box":
        size = f.get("size")
        if not isinstance(size, list) or len(size) != 3:
            raise CadError("INVALID_VALUE", "size is [x, y, z]", f"{loc}.size")
        dims = [part.length(v, f"{loc}.size[{i}]") for i, v in enumerate(size)]
        tool = build.box(dims, frame=frame, prefix=f["id"], group=g)
    elif t == "cylinder":
        tool = build.cylinder(
            part.length(f.get("radius"), f"{loc}.radius"),
            part.length(f.get("height"), f"{loc}.height"),
            frame=frame,
            prefix=f["id"],
            group=g,
        )
    elif t == "sphere":
        tool = build.sphere(
            part.length(f.get("radius"), f"{loc}.radius"), frame=frame, prefix=f["id"], group=g
        )
    elif t == "cone":
        tool = build.cone(
            part.length(f.get("radius1"), f"{loc}.radius1"),
            part.length(f.get("radius2", 0), f"{loc}.radius2"),
            part.length(f.get("height"), f"{loc}.height"),
            frame=frame,
            prefix=f["id"],
            group=g,
        )
    else:
        tool = build.torus(
            part.length(f.get("major"), f"{loc}.major"),
            part.length(f.get("minor"), f"{loc}.minor"),
            frame=frame,
            prefix=f["id"],
            group=g,
        )
    op = f.get("operation", "new" if not state else "add")
    target = f.get("body", "main")
    return FeatureOutput(
        _apply(state, target, tool, op, loc), {"operation": op, "body": target}, tool, op, target
    )


# --------------------------------------------------------------------------- #
# Holes
# --------------------------------------------------------------------------- #


def f_hole(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    target = f.get("body", "main")
    if target not in state:
        raise CadError("NO_BODY", f"there is no body {target!r} to drill", loc)
    body = state[target]
    fi = resolve_face(body, f.get("face"), f"{loc}.face", part.scale)
    face = body.faces[fi]
    if not isinstance(face.surface, Plane):
        raise CadError("UNSUPPORTED_FACE", "holes are placed on planar faces", f"{loc}.face")
    n = face.surface.frame.z if face.sense else -face.surface.frame.z  # outward
    pts = _hole_points(part, f, loc)
    d = part.length(f.get("diameter"), f"{loc}.diameter")
    if d <= 0:
        raise CadError("INVALID_VALUE", "the diameter must be positive", f"{loc}.diameter")
    r = d / 2
    size = _state_size(state)
    depth_raw = f.get("depth", "through")
    through = depth_raw in ("through", "through_all")
    depth = size if through else part.length(depth_raw, f"{loc}.depth")
    if depth <= 0:
        raise CadError("INVALID_VALUE", "the depth must be positive", f"{loc}.depth")
    kind = f.get("kind", "simple")
    margin = max(0.05 * d, 1e-4)
    prof = [(0.0, -margin), (r, -margin)]
    names = ["top"]
    if kind == "counterbore":
        rc = part.length(f.get("cbore_diameter"), f"{loc}.cbore_diameter") / 2
        dc = part.length(f.get("cbore_depth"), f"{loc}.cbore_depth")
        if not (rc > r and 0 < dc < depth):
            raise CadError(
                "INVALID_VALUE", "a counterbore must be wider than the hole and shallower", loc
            )
        prof = [(0.0, -margin), (rc, -margin), (rc, dc), (r, dc)]
        names = ["top", "cbore", "cbore_floor"]
    elif kind == "countersink":
        rs = part.length(f.get("csink_diameter"), f"{loc}.csink_diameter") / 2
        ang = part.angle(f.get("csink_angle", "90 deg"), f"{loc}.csink_angle")
        if not (rs > r and 0 < ang < math.pi):
            raise CadError(
                "INVALID_VALUE", "a countersink must be wider than the hole, angle 0..180", loc
            )
        h = (rs - r) / math.tan(ang / 2)
        if h >= depth:
            raise CadError("INVALID_VALUE", "the countersink is deeper than the hole", loc)
        prof = [(0.0, -margin), (rs + margin * math.tan(ang / 2), -margin), (r, h)]
        names = ["top", "csink"]
    elif kind != "simple":
        raise CadError(
            "INVALID_VALUE", "kind must be simple, counterbore or countersink", f"{loc}.kind"
        )
    prof.append((r, depth))
    names.append("wall")
    tip = f.get("tip_angle")
    if tip is not None and not through:
        ta = part.angle(tip, f"{loc}.tip_angle")
        if not 0 < ta < math.pi:
            raise CadError(
                "INVALID_VALUE", "tip_angle must be within (0, 180) degrees", f"{loc}.tip_angle"
            )
        prof.append((0.0, depth + r / math.tan(ta / 2)))
        names.append("tip")
    else:
        prof.append((0.0, depth))
        names.append("floor")
    names.append("axis")
    pieces = []
    for i in range(len(prof)):
        a, b = np.array(prof[i]), np.array(prof[(i + 1) % len(prof)])
        pieces.append(Piece(Segment(a, b), names[i], 0))
    region = Region(Loop(pieces))
    tool = None
    for k, p in enumerate(pts):
        prefix = f["id"] if len(pts) == 1 else f"{f['id']}.{k}"
        # Project onto the face plane; check the point is on the face.
        s = face.surface
        p0 = p - float((p - s.frame.origin) @ s.frame.z) * s.frame.z
        if not _point_on_face(body, face, p0):
            raise CadError(
                "HOLE_OFF_FACE",
                f"the hole center {(p0 / part.scale).round(6).tolist()} is not on face "
                f"{face.name!r}",
                f"{loc}.points",
            )
        down = -n
        x = Frame.from_axes(p0, down).x
        sk_frame = Frame(p0, x, down, np.cross(x, down))
        t = build.revolve(
            region, sk_frame, (0.0, 0.0), (0.0, 1.0), prefix=prefix, group=int(f.get("group", 0))
        )
        tool = t if tool is None else boolean(tool, t, "union")
    return FeatureOutput(
        _apply(state, target, tool, "cut", loc),
        {"holes": len(pts), "diameter_m": d, "depth_m": None if through else depth, "kind": kind},
        tool,
        "cut",
        target,
    )


def _hole_points(part: _Part, f, loc) -> list[np.ndarray]:
    if f.get("sketch") is not None:
        sk = part.sketch(f["sketch"], f"{loc}.sketch")
        ids = f.get("point_ids")
        if not isinstance(ids, list) or not ids:
            raise CadError("INVALID_VALUE", "point_ids must list sketch points", f"{loc}.point_ids")
        return [sk.point3(pid) for pid in ids]
    raw = f.get("points")
    if not isinstance(raw, list) or not raw or len(raw) > MAX_PATTERN:
        raise CadError(
            "INVALID_VALUE", f"points must list 1..{MAX_PATTERN} [x, y, z]", f"{loc}.points"
        )
    return [part.point(p, f"{loc}.points[{i}]") for i, p in enumerate(raw)]


def _point_on_face(body: Body, face: Face, p: np.ndarray) -> bool:
    fr = face.surface.frame
    q = fr.to_local(p)[:2]
    inside = False
    for li, loop in enumerate(face.loops):
        pts = []
        for ei, fwd in loop:
            e = body.edges[ei]
            seg = e.curve.point(np.linspace(e.t0, e.t1, 33))
            pts.append(seg if fwd else seg[::-1])
        poly = fr.to_local(np.concatenate(pts))[:, :2]
        x, y = poly[:, 0], poly[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cond = (y > q[1]) != (yn > q[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x + (q[1] - y) * (xn - x) / (yn - y)
        hit = bool(np.count_nonzero(cond & (q[0] < xc)) % 2)
        if li == 0:
            inside = hit
        elif hit:
            return False
    return inside


# --------------------------------------------------------------------------- #
# Fillets and chamfers
# --------------------------------------------------------------------------- #


def _edge_frame(body: Body, ei: int, loc: str):
    """For a straight edge between two planar faces: the edge's endpoints and the two
    faces' outward normals, and whether the edge is convex."""
    e = body.edges[ei]
    if not isinstance(e.curve, Line3):
        raise CadError("UNSUPPORTED_EDGE", "fillets and chamfers need straight edges", loc)
    uses = body.edge_faces()[ei]
    if len(uses) != 2:
        raise CadError("UNSUPPORTED_EDGE", "the edge is not between two faces", loc)
    (f1, fwd1), (f2, _) = uses
    faces = [body.faces[f1], body.faces[f2]]
    if not all(isinstance(fc.surface, Plane) for fc in faces):
        raise CadError(
            "UNSUPPORTED_EDGE",
            "fillets and chamfers are supported between two planar faces",
            loc,
            hint="fillet the profile in its sketch instead (tangent arcs)",
        )
    n1 = faces[0].surface.frame.z * (1 if faces[0].sense else -1)
    n2 = faces[1].surface.frame.z * (1 if faces[1].sense else -1)
    p0, p1 = body.vertices[e.v0], body.vertices[e.v1]
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    # Coedge of face 1 runs along d when fwd1: convex if (n1 x n2) . d1 > 0.
    d1 = d if fwd1 else -d
    convex = float(np.cross(n1, n2) @ d1) > 0
    if abs(float(n1 @ n2)) > 1 - 1e-9:
        raise CadError("UNSUPPORTED_EDGE", "the faces at the edge are tangent", loc)
    return p0, p1, d, n1, n2, convex, faces


def _corner_tool(
    p0, p1, d, n1, n2, convex: bool, section, name: str, prefix: str, group: int
) -> Body:
    """Extrude a 2D section (in the plane through p0 perpendicular to d) along the edge."""
    x = n1 - (n1 @ d) * d
    frame = Frame.from_axes(p0, d, x)
    length = float(np.linalg.norm(p1 - p0))
    return build.extrude(section, frame, 0.0, length, prefix=prefix, group=group)


def _local2(frame_x, frame_y, v):
    return np.array([float(v @ frame_x), float(v @ frame_y)])


def f_fillet(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    return _blend(part, f, state, loc, "fillet")


def f_chamfer(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    return _blend(part, f, state, loc, "chamfer")


def _blend(part: _Part, f, state, loc, mode: str) -> FeatureOutput:
    target = f.get("body", "main")
    if target not in state:
        raise CadError("NO_BODY", f"there is no body {target!r}", loc)
    body = state[target]
    refs = f.get("edges")
    if not isinstance(refs, list) or not refs:
        raise CadError("INVALID_VALUE", "edges must list edge references", f"{loc}.edges")
    if mode == "fillet":
        r = part.length(f.get("radius"), f"{loc}.radius")
        if r <= 0:
            raise CadError("INVALID_VALUE", "the radius must be positive", f"{loc}.radius")
    else:
        r = part.length(f.get("distance"), f"{loc}.distance")
        r2 = part.length(f.get("distance2", f.get("distance")), f"{loc}.distance2")
        if r <= 0 or r2 <= 0:
            raise CadError("INVALID_VALUE", "chamfer distances must be positive", f"{loc}.distance")
    edges = [resolve_edge(body, ref, f"{loc}.edges[{i}]", part.scale) for i, ref in enumerate(refs)]
    verts = [v for ei in edges for v in (body.edges[ei].v0, body.edges[ei].v1)]
    if len(set(verts)) != len(verts):
        raise CadError(
            "FILLET_CORNER_UNSUPPORTED",
            "selected edges meet at a vertex; blending corners is not supported",
            f"{loc}.edges",
            hint="blend non-adjacent edges, or round the profile in its sketch",
        )
    infos = []
    for i, ei in enumerate(edges):
        body = state[target]
        ei = resolve_edge(body, refs[i], f"{loc}.edges[{i}]", part.scale)
        p0, p1, d, n1, n2, convex, faces = _edge_frame(body, ei, f"{loc}.edges[{i}]")
        _check_end_faces(body, ei, d, f"{loc}.edges[{i}]")
        x = n1 - (n1 @ d) * d
        x /= np.linalg.norm(x)
        y = np.cross(d, x)
        a1 = _local2(x, y, n1)
        a2 = _local2(x, y, n2)
        sgn = -1.0 if convex else 1.0  # toward the material (convex) or the void (concave)
        # In-face directions away from the edge: w1 lies in face 1, perpendicular to d,
        # pointing into face 1 (away from face 2's side).
        w1 = _local2(x, y, np.cross(d, n1))
        w2 = _local2(x, y, np.cross(d, n2))
        if float(w1 @ a2) > 0:
            w1 = -w1
        if float(w2 @ a1) > 0:
            w2 = -w2
        if not convex:
            w1, w2 = -w1, -w2
        name = f"{f['id']}" if len(edges) == 1 else f"{f['id']}.{i}"
        if mode == "fillet":
            c2 = sgn * r * (a1 + a2) / (1.0 + float(a1 @ a2))
            t1 = c2 - sgn * r * a1
            t2 = c2 - sgn * r * a2
            corner = np.zeros(2)
            # Region: corner -> t1 -> arc (around c2, through the side near the corner) -> t2.
            a_t1 = math.atan2(*(t1 - c2)[::-1])
            a_t2 = math.atan2(*(t2 - c2)[::-1])
            mid_dir = corner - c2
            sweep = (a_t2 - a_t1) % (2 * math.pi)
            mid_angle = a_t1 + sweep / 2
            mid_pt = c2 + r * np.array([math.cos(mid_angle), math.sin(mid_angle)])
            if float((mid_pt - c2) @ mid_dir) < 0:
                sweep = sweep - 2 * math.pi
            arc = Arc(c2, r, a_t1, sweep)
            pieces = [
                Piece(Segment(corner, t1), "face1", 0),
                Piece(arc, "round", 0),
                Piece(Segment(t2, corner), "face2", 0),
            ]
        else:
            q1 = r * w1
            q2 = r2 * w2
            pieces = [
                Piece(Segment(np.zeros(2), q1), "face1", 0),
                Piece(Segment(q1, q2), "bevel", 0),
                Piece(Segment(q2, np.zeros(2)), "face2", 0),
            ]
        loop = Loop(pieces)
        if loop.area < 0:
            loop = loop.reversed()
        tool = _corner_tool(
            p0, p1, d, x, None, convex, Region(loop), "", name, int(f.get("group", 0))
        )
        op = "cut" if convex else "add"
        new = _apply(state, target, tool, op, loc)
        removed = abs(_volume(new[target]) - _volume(state[target]))
        expected = abs(loop.area) * float(np.linalg.norm(p1 - p0))
        if abs(removed - expected) > 1e-6 * max(expected, 1e-12) + 1e-12:
            raise CadError(
                "FILLET_UNSUPPORTED",
                f"the {mode} of edge {i} would also change other faces (the material near the "
                "edge's ends is not bounded by faces perpendicular to it)",
                f"{loc}.edges[{i}]",
                details={"expected_volume_change": expected, "actual": removed},
            )
        state = new
        infos.append({"edge": i, "convex": convex, "volume_change_m3": removed})
    return FeatureOutput(state, {"edges": infos, "mode": mode}, None, None, target)


def _check_end_faces(body: Body, ei: int, d: np.ndarray, loc: str) -> None:
    """The faces at the edge's end vertices (other than its two faces) must be
    perpendicular to the edge: then the straight tool ends exactly there."""
    e = body.edges[ei]
    own = {f for f, _ in body.edge_faces()[ei]}
    for v in (e.v0, e.v1):
        for fi, face in enumerate(body.faces):
            if fi in own:
                continue
            touches = any(
                v in (body.edges[x].v0, body.edges[x].v1) for loop in face.loops for x, _ in loop
            )
            if not touches:
                continue
            if (
                not isinstance(face.surface, Plane)
                or abs(abs(float(face.surface.frame.z @ d)) - 1) > 1e-9
            ):
                raise CadError(
                    "FILLET_UNSUPPORTED",
                    f"face {face.name!r} at the end of the edge is not perpendicular to it",
                    loc,
                    hint="blend edges whose ends meet perpendicular planar faces",
                )


def _volume(body: Body) -> float:
    from .brep.properties import mass_properties

    return mass_properties(body).volume


# --------------------------------------------------------------------------- #
# Shell, patterns, mirror, transform, boolean
# --------------------------------------------------------------------------- #


def f_shell(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    from .offset import offset_loop
    from .tolerance import Tolerance

    src = outputs.get(f.get("feature"))
    if src is None or src.extra.get("kind") != "extrude":
        raise CadError(
            "SHELL_UNSUPPORTED",
            "shell needs the id of an earlier extrude feature (prismatic bodies)",
            f"{loc}.feature",
        )
    t = part.length(f.get("thickness"), f"{loc}.thickness")
    if t <= 0:
        raise CadError("INVALID_VALUE", "the thickness must be positive", f"{loc}.thickness")
    open_caps = f.get("open", ["end"])
    if not isinstance(open_caps, list) or not set(open_caps) <= {"start", "end"}:
        raise CadError("INVALID_VALUE", 'open lists "start" and/or "end"', f"{loc}.open")
    lo, hi = src.extra["start"], src.extra["end"]
    margin = 0.5 * (hi - lo) + t
    a = lo - margin if "start" in open_caps else lo + t
    b = hi + margin if "end" in open_caps else hi - t
    if b <= a:
        raise CadError("SHELL_TOO_THICK", "the thickness leaves no cavity", f"{loc}.thickness")
    tol = Tolerance.for_scale(_state_size(state)).resolution
    tool = None
    for reg in src.extra["regions"]:
        try:
            inner = Region(
                offset_loop(reg.outer, t, tol), [offset_loop(h, t, tol) for h in reg.holes]
            )
        except CadError as exc:
            raise CadError(exc.code, exc.message, f"{loc}.thickness", hint=exc.hint) from None
        piece = build.extrude(
            inner, src.extra["frame"], a, b, prefix=f["id"], group=int(f.get("group", 0))
        )
        tool = piece if tool is None else boolean(tool, piece, "union")
    target = f.get("body", src.target or "main")
    return FeatureOutput(
        _apply(state, target, tool, "cut", loc),
        {"thickness_m": t, "open": open_caps},
        tool,
        "cut",
        target,
    )


def _rot(axis: np.ndarray, angle: float, point: np.ndarray) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    r = np.eye(3) + s * k + (1 - c) * (k @ k)
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = point - r @ point
    return m


def f_pattern(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    refs = f.get("features") or []
    srcs = [outputs[r] for r in refs if r in outputs]
    if not srcs or any(s.tool is None or s.op not in ("add", "cut", "new") for s in srcs):
        raise CadError(
            "PATTERN_UNSUPPORTED",
            "pattern repeats features that add or cut a tool (extrude, revolve, hole, primitives)",
            f"{loc}.features",
        )
    kind = f.get("kind", "linear")
    count = part.count(f.get("count", 2), f"{loc}.count", 2, MAX_PATTERN)
    mats = []
    if kind == "linear":
        d = part.direction(f.get("direction", [1, 0, 0]), f"{loc}.direction")
        step = part.length(f.get("spacing"), f"{loc}.spacing")
        for k in range(1, count):
            m = np.eye(4)
            m[:3, 3] = d * step * k
            mats.append(m)
    elif kind == "circular":
        ax = f.get("axis", {})
        if not isinstance(ax, dict):
            raise CadError("INVALID_VALUE", "axis is {point, direction}", f"{loc}.axis")
        p = part.point(ax.get("point", [0, 0, 0]), f"{loc}.axis.point")
        d = part.direction(ax.get("direction", [0, 0, 1]), f"{loc}.axis.direction")
        total = part.angle(f.get("angle", "360 deg"), f"{loc}.angle")
        full = abs(abs(total) - 2 * math.pi) < 1e-12
        step = total / (count if full else count - 1)
        for k in range(1, count):
            mats.append(_rot(d, step * k, p))
    else:
        raise CadError("INVALID_VALUE", "kind is linear or circular", f"{loc}.kind")
    for s in srcs:
        target = s.target or "main"
        copies = [s.tool.transformed(m).renamed({}) for m in mats]
        tool = copies[0]
        for c in copies[1:]:
            tool = _merge_tools(tool, c)
        state = _apply(state, target, tool, "cut" if s.op == "cut" else "add", loc)
    return FeatureOutput(state, {"copies": count - 1, "kind": kind})


def _merge_tools(a: Body, b: Body) -> Body:
    return boolean(a, b, "union")


def f_mirror(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    pl = f.get("plane")
    if not isinstance(pl, dict):
        raise CadError("INVALID_VALUE", "plane is {point, normal}", f"{loc}.plane")
    p = part.point(pl.get("point", [0, 0, 0]), f"{loc}.plane.point")
    n = part.direction(pl.get("normal", [1, 0, 0]), f"{loc}.plane.normal")
    m = np.eye(4)
    m[:3, :3] = np.eye(3) - 2 * np.outer(n, n)
    m[:3, 3] = 2 * float(p @ n) * n
    if f.get("body") is not None:
        target = f["body"]
        if target not in state:
            raise CadError("NO_BODY", f"there is no body {target!r}", f"{loc}.body")
        mirrored = state[target].transformed(m)
        mirrored = mirrored.renamed({})
        return FeatureOutput(_apply(state, target, mirrored, "add", loc), {"mirrored": "body"})
    refs = f.get("features") or []
    srcs = [outputs[r] for r in refs if r in outputs]
    if not srcs or any(s.tool is None for s in srcs):
        raise CadError(
            "MIRROR_UNSUPPORTED", "mirror features that add or cut a tool", f"{loc}.features"
        )
    for s in srcs:
        state = _apply(
            state, s.target or "main", s.tool.transformed(m), "cut" if s.op == "cut" else "add", loc
        )
    return FeatureOutput(state, {"mirrored": refs})


def f_transform(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    target = f.get("body", "main")
    if target not in state:
        raise CadError("NO_BODY", f"there is no body {target!r}", f"{loc}.body")
    m = np.eye(4)
    if f.get("rotate") is not None:
        r = f["rotate"]
        m = _rot(
            part.direction(r.get("axis", [0, 0, 1]), f"{loc}.rotate.axis"),
            part.angle(r.get("angle", 0), f"{loc}.rotate.angle"),
            part.point(r.get("point", [0, 0, 0]), f"{loc}.rotate.point"),
        )
    if f.get("translate") is not None:
        t = np.eye(4)
        t[:3, 3] = part.point(f["translate"], f"{loc}.translate")
        m = t @ m
    out = dict(state)
    out[target] = state[target].transformed(m)
    return FeatureOutput(out, {"body": target})


def f_boolean(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    op = f.get("op")
    if op not in ("union", "cut", "intersect"):
        raise CadError("INVALID_VALUE", "op is union, cut or intersect", f"{loc}.op")
    a, b = f.get("target", "main"), f.get("tool")
    for name, key in ((a, "target"), (b, "tool")):
        if name not in state:
            raise CadError("NO_BODY", f"there is no body {name!r}", f"{loc}.{key}")
    out = _apply(
        state, a, state[b], {"union": "add", "cut": "cut", "intersect": "intersect"}[op], loc
    )
    if not f.get("keep_tool", False):
        out.pop(b, None)
    return FeatureOutput(out, {"op": op})


# --------------------------------------------------------------------------- #
# Sweep and loft
# --------------------------------------------------------------------------- #


def f_sweep(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    prof = part.sketch(f.get("profile"), f"{loc}.profile")
    path = part.sketch(f.get("path"), f"{loc}.path")
    regions = _regions(part, prof, f.get("regions"), f"{loc}.regions")
    chain = path.open_chain()
    if not chain:
        raise CadError("INVALID_PATH", "the path sketch has no lines or arcs", f"{loc}.path")
    pf = prof.frame
    start = path.frame.to_world(np.array([*chain[0].start, 0.0]))
    t0 = chain[0].tangent(0.0)
    tan0 = path.frame.direction_to_world(np.array([t0[0], t0[1], 0.0]))
    tan0 /= np.linalg.norm(tan0)
    if abs(abs(float(pf.z @ tan0)) - 1) > 1e-9 or abs(
        float((start - pf.origin) @ pf.z)
    ) > 1e-9 * max(1.0, float(np.linalg.norm(start))):
        raise CadError(
            "SWEEP_PROFILE_NOT_PERPENDICULAR",
            "the profile's plane must pass through the path's start, perpendicular to it",
            f"{loc}.profile",
        )
    flip = float(pf.z @ tan0) < 0
    tool = None
    frame = Frame(pf.origin, pf.x, pf.y, pf.z) if not flip else Frame(pf.origin, -pf.x, pf.y, -pf.z)
    if flip:
        regions = [_mirror_region_x(r) for r in regions]
    pn = path.frame.z
    for k, c in enumerate(chain):
        prefix = f"{f['id']}.{k}" if len(chain) > 1 else f["id"]
        if isinstance(c, Segment):
            length = float(np.linalg.norm(c.p1 - c.p0))
            pieces = [build.extrude(reg, frame, 0.0, length, prefix=prefix) for reg in regions]
            nxt = Frame(frame.origin + length * frame.z, frame.x, frame.y, frame.z)
        else:
            center = path.frame.to_world(np.array([*c.center, 0.0]))
            axis_dir = pn if c.sweep > 0 else -pn
            local_c = frame.to_local(center)[:2]
            local_d = frame.to_local(frame.origin + axis_dir)[:2]
            pieces = []
            for reg in regions:
                pieces.append(_revolve_about(reg, frame, local_c, local_d, abs(c.sweep), prefix))
            rot = _rot(axis_dir, abs(c.sweep), center)
            nxt = frame.transformed(rot)
        for p in pieces:
            tool = p if tool is None else boolean(tool, p, "union")
        frame = nxt
    op = f.get("operation", "new" if not state else "add")
    target = f.get("body", "main")
    return FeatureOutput(
        _apply(state, target, tool, op, loc), {"segments": len(chain)}, tool, op, target
    )


def _revolve_about(reg, frame, c2, d2, angle, prefix):
    d2 = d2 / np.linalg.norm(d2)
    body = build.revolve(reg, frame, c2, d2, angle, prefix=prefix)
    return body


def _mirror_region_x(reg: Region) -> Region:
    def mirror_loop(loop: Loop) -> Loop:
        pieces = []
        for p in loop.pieces:
            c = p.curve
            if isinstance(c, Segment):
                nc = Segment(c.p0 * [-1, 1], c.p1 * [-1, 1])
            elif isinstance(c, Arc):
                nc = Arc(c.center * [-1, 1], c.radius, math.pi - c.start_angle, -c.sweep)
            else:
                raise CadError("UNSUPPORTED_GEOMETRY", "spline profiles cannot be swept backwards")
            pieces.append(Piece(nc, p.source, p.index))
        return Loop(pieces).reversed()

    return Region(mirror_loop(reg.outer), [mirror_loop(h) for h in reg.holes])


def f_loft(part: _Part, f, state, outputs, loc) -> FeatureOutput:
    from .brep.loft import ruled_loft

    secs = f.get("sections")
    if not isinstance(secs, list) or len(secs) < 2:
        raise CadError(
            "INVALID_VALUE", "a loft needs at least two section sketches", f"{loc}.sections"
        )
    sections = []
    for i, s in enumerate(secs):
        sk = part.sketch(s, f"{loc}.sections[{i}]")
        regs = sk.regions(None)
        if len(regs) != 1 or regs[0].holes:
            raise CadError(
                "LOFT_UNSUPPORTED",
                "each loft section must be one region without holes",
                f"{loc}.sections[{i}]",
            )
        sections.append((regs[0].outer, sk.frame))
    tool = ruled_loft(sections, prefix=f["id"], group=int(f.get("group", 0)))
    op = f.get("operation", "new" if not state else "add")
    target = f.get("body", "main")
    return FeatureOutput(
        _apply(state, target, tool, op, loc), {"sections": len(sections)}, tool, op, target
    )


FEATURES = {
    "extrude": f_extrude,
    "revolve": f_revolve,
    "sweep": f_sweep,
    "loft": f_loft,
    "hole": f_hole,
    "fillet": f_fillet,
    "chamfer": f_chamfer,
    "shell": f_shell,
    "pattern": f_pattern,
    "mirror": f_mirror,
    "transform": f_transform,
    "boolean": f_boolean,
    "box": _primitive,
    "cylinder": _primitive,
    "sphere": _primitive,
    "cone": _primitive,
    "torus": _primitive,
}


def circle3(*args):  # pragma: no cover - re-exported convenience
    return Circle3(*args)


__all__ = ["FEATURES", "PartResult", "check", "references", "regenerate", "ANGLE", "LENGTH"]
