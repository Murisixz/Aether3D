"""Expectations: what a scene should satisfy, as data (``aether3d.expectations``, version 1).

An expectations document states intent in checkable terms -- which objects must exist,
what must rest on what, what must not intersect, what a view must show and how, how bright
an image should be -- so that diagnosis compares the scene and its renders against
explicit rules instead of guessing. Floating objects, for example, are only reported for
objects a ``rests_on`` check names: a lamp may hang and a bird may fly.

```json
{
  "format": "aether3d.expectations", "version": 1, "label": "Living room",
  "views": {"main": {"camera": "Camera"}, "top": {"preset": "top"}},
  "checks": [
    {"check": "exists", "objects": {"tags": ["chair"]}, "count": {"min": 4}},
    {"check": "rests_on", "objects": "Vase", "support": "Table"},
    {"check": "no_overlap", "objects": {"tags": ["furniture"]}},
    {"check": "visible", "objects": "Sofa", "view": "main", "min_visible_fraction": 0.5},
    {"check": "framing", "objects": "Sofa", "region": [0.2, 0.2, 0.8, 0.8]},
    {"check": "exposure", "view": "main", "mean_luminance": {"min": 0.1, "max": 0.6}}
  ],
  "acceptance": {"max_errors": 0}
}
```

The checker reports every problem at once (code, JSON pointer, hint), like scene plans;
nothing in a document is executed. :data:`CHECKS` lists the check kinds and their fields;
``docs/FEEDBACK.md`` describes their exact semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..errors import FeedbackError
from ..sculpt.validate import TIERS
from ._doc import BAD, Doc, join
from .views import parse_view_spec

FORMAT = "aether3d.expectations"
VERSION = 1
SEVERITIES = ("error", "warning")
MAX_CHECKS = 500
MAX_VIEWS = 16
MAX_DOCUMENT_BYTES = 4 * 2**20
DEFAULT_TOLERANCE = 1e-4
"""Meters: gaps, penetrations and contacts within this count as touching."""

COMMON = ("check", "id", "severity", "label", "fix")
CHECKS: dict[str, dict[str, Any]] = {
    "exists": {
        "fields": ("objects", "count"),
        "required": ("objects",),
        "summary": "the selected objects exist, as many as count says (default at least 1)",
    },
    "visible": {
        "fields": ("objects", "view", "min_pixels", "min_visible_fraction", "in_frame"),
        "required": ("objects",),
        "summary": "each object is seen in the view: rendered, in the view volume, at least "
        "min_pixels visible (default 1), and optionally at least min_visible_fraction of its "
        "unoccluded footprint, wholly inside the frame (in_frame)",
    },
    "framing": {
        "fields": ("objects", "view", "region", "fill", "height"),
        "required": ("objects",),
        "summary": "composition in the view: the visible pixels' center inside region "
        "(normalized [x0, y0, x1, y1]), their share of the image within fill {min, max}, "
        "their screen height fraction within height {min, max}",
    },
    "rests_on": {
        "fields": ("objects", "support", "tolerance"),
        "required": ("objects",),
        "summary": "each object rests on the support (or, without support, on whatever is "
        "directly below it): no vertical gap and no penetration beyond tolerance",
    },
    "no_overlap": {
        "fields": ("objects", "against", "tolerance", "allow_contact"),
        "required": ("objects",),
        "summary": "the objects do not intersect each other (or the against objects); "
        "touching within tolerance is allowed unless allow_contact is false",
    },
    "inside_region": {
        "fields": ("objects", "region"),
        "required": ("objects", "region"),
        "summary": 'each object\'s world box lies inside region {"box": {"min", "max"}}',
    },
    "distance": {
        "fields": ("a", "b", "min", "max"),
        "required": ("a", "b"),
        "summary": "the surface distance between a and b lies within [min, max]",
    },
    "relation": {
        "fields": ("a", "b", "predicate", "expect", "near_distance"),
        "required": ("a", "b", "predicate"),
        "summary": "a spatial relation of a to b (above, below, in_front_of, behind, "
        "inside, contains, near, intersecting) has the expected truth value",
    },
    "upright": {
        "fields": ("objects", "max_tilt", "axis"),
        "required": ("objects",),
        "summary": "each object's local axis (default +Z) points within max_tilt degrees "
        "(default 2) of world +Z",
    },
    "material": {
        "fields": ("objects", "material"),
        "required": ("objects", "material"),
        "summary": "each object uses the material (by name or id)",
    },
    "exposure": {
        "fields": (
            "view",
            "mean_luminance",
            "over",
            "max_clipped_fraction",
            "max_black_fraction",
            "min_coverage",
        ),
        "required": (),
        "summary": "the view's image: mean linear luminance (of the objects' pixels, or the "
        "whole image with over: image) within a range, clipped and black fractions at most "
        "the given ones, geometry covering at least min_coverage of the image",
    },
    "camera_clipping": {
        "fields": ("objects", "view"),
        "required": (),
        "summary": "the view's near and far planes cut none of the objects (default: every "
        "visible object with geometry)",
    },
    "mesh_validity": {
        "fields": ("objects", "tier"),
        "required": ("objects",),
        "summary": "each object's mesh (a group: its parts' meshes) reaches the validity "
        "tier: valid, manifold, closed (default: manifold and watertight) or solid (also "
        "no self-intersection and a positive volume, tested exactly)",
    },
    "uv_quality": {
        "fields": (
            "objects",
            "resolution",
            "max_flipped",
            "max_overlapping",
            "max_stretch",
            "max_density_spread",
            "min_gap",
        ),
        "required": ("objects",),
        "summary": "each object's mesh (a group: its parts' meshes) has UVs with at most "
        "max_flipped flipped and max_overlapping overlapping triangles (default 0; stacked "
        "islands are deliberate and not counted), optionally an L2 stretch of at most "
        "max_stretch, a texel-density spread of at most max_density_spread and gaps of at "
        "least min_gap texels at resolution (default 1024)",
    },
    "cad_valid": {
        "fields": ("items", "fully_constrained"),
        "required": (),
        "summary": "the CAD items (default: every item) regenerate without error, every part "
        "body is a valid solid and every sketch solves; with fully_constrained, no sketch "
        "has degrees of freedom left (exact, from the CAD document)",
    },
    "animation_continuity": {
        "fields": ("objects", "order"),
        "required": ("objects",),
        "summary": "the selected objects' animation curves (their actions) have no unintended "
        "jump up to order (0 value, 1 velocity -- the default, 2 acceleration) at any key, "
        "from the curves' exact one-sided limits; constant keys and jump pairs are intended",
    },
    "foot_contacts": {
        "fields": ("objects", "feet", "max_sliding", "max_penetration", "height", "speed"),
        "required": ("objects", "feet"),
        "summary": "the armatures' planted feet (joints named in feet) slide at most "
        "max_sliding (default 0.005 m) and go below the ground (z = 0) at most "
        "max_penetration (default 0.002 m) over the timeline, sampled at 120 per second "
        "(planted: within height, default 0.03 m, of the ground and within speed, default "
        "0.3 m/s, of the stance's steady velocity: a skating stance is measured)",
    },
    "rig": {
        "fields": ("objects", "poses", "timeline", "ignore"),
        "required": ("objects",),
        "summary": "the selected armatures' rigs pass the rig diagnostics (diagnose_rig): the "
        "skeleton, the weights and binding of every skinned mesh, and the deformation in "
        "poses (default: the diagnostic suite -- elbows and knees bent 120 degrees, hips "
        "flexed 90, shoulders raised, wrists and torso twisted, the head turned) and, with "
        "timeline (default true), over the animation (IK flips, unreachable targets, "
        "sliding feet); errors and warnings fail, codes in ignore are skipped (numeric "
        "measurements of the mesh, not judgments of anatomy)",
    },
    "cad_dimension": {
        "fields": ("reference", "other", "quantity", "value", "tolerance", "min", "max"),
        "required": ("reference", "quantity"),
        "summary": "an exact measurement of a CAD reference (or between reference and "
        "other: distance, parallel_distance, axis_distance, angle_deg; of one: length, "
        "area, radius) equals value within tolerance, or lies within min..max (document "
        "units, degrees)",
    },
}
MESH_TIERS = TIERS
RELATION_PREDICATES = (
    "above",
    "below",
    "in_front_of",
    "behind",
    "inside",
    "contains",
    "near",
    "intersecting",
)
CAD_QUANTITIES = (
    "distance",
    "parallel_distance",
    "axis_distance",
    "angle_deg",
    "length",
    "area",
    "radius",
)
FIX_KEYS = ("move", "camera", "allow")
MOVE_CHOICES = ("objects", "against", "support", "none")


@dataclass(frozen=True)
class Check:
    """One check of an expectations document, checked and normalized."""

    id: str
    kind: str
    severity: str
    label: str | None
    pointer: str
    params: dict[str, Any] = field(default_factory=dict)
    fix: dict[str, Any] = field(default_factory=dict)

    @property
    def view(self) -> str | None:
        return self.params.get("view")

    def to_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "check": self.kind, "severity": self.severity, **self.params}
        if self.label:
            out["label"] = self.label
        if self.fix:
            out["fix"] = self.fix
        return out


@dataclass(frozen=True)
class Expectations:
    """A checked expectations document."""

    label: str
    views: dict[str, dict[str, Any]]
    checks: tuple[Check, ...]
    default_view: str | None
    max_errors: int = 0
    max_warnings: int | None = None
    require_vision: bool = False
    defaults: bool = True
    source: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def fingerprint(self) -> str:
        import hashlib

        text = json.dumps(self.source, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def check(self, check_id: str) -> Check:
        for c in self.checks:
            if c.id == check_id:
                return c
        raise FeedbackError("UNKNOWN_CHECK", f"no check {check_id!r}", "check")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "label": self.label,
            "views": self.views,
            "default_view": self.default_view,
            "checks": [c.to_dict() for c in self.checks],
            "acceptance": {
                "max_errors": self.max_errors,
                "max_warnings": self.max_warnings,
                "require_vision": self.require_vision,
            },
            "defaults": self.defaults,
        }


EMPTY_DOCUMENT = {"format": FORMAT, "version": VERSION}


def parse_expectations(document: Any) -> Expectations:
    """Check an expectations document (a dict, or JSON text) as a whole.

    Raises ``FeedbackError`` with every problem in ``issues`` (codes ``INVALID_JSON``,
    ``INVALID_FORMAT``, ``UNSUPPORTED_VERSION``, ``INVALID_TYPE``, ``INVALID_VALUE``,
    ``OUT_OF_RANGE``, ``UNKNOWN_KEY``, ``MISSING_KEY``, ``DUPLICATE_ID``,
    ``UNKNOWN_VIEW``, ``LIMIT_EXCEEDED``, ...).
    """
    if isinstance(document, Expectations):
        return document
    if isinstance(document, (str, bytes)):
        if len(document) > MAX_DOCUMENT_BYTES:
            raise FeedbackError("LIMIT_EXCEEDED", "the document is larger than 4 MiB", "")
        try:
            document = json.loads(document)
        except ValueError as exc:
            raise FeedbackError("INVALID_JSON", f"not valid JSON: {exc}", "") from None
    doc = Doc()
    obj = doc.object(document, "", "an expectations document (a JSON object)")
    if obj is None:
        doc.issues.raise_if_any("invalid expectations")
    assert obj is not None
    if obj.get("format") != FORMAT:
        doc.issues.add(
            "INVALID_FORMAT",
            f'"format" must be {FORMAT!r}',
            "/format",
            hint=f'"format": "{FORMAT}"',
        )
    version = obj.get("version")
    if isinstance(version, bool) or version != VERSION:
        doc.issues.add("UNSUPPORTED_VERSION", f'"version" must be {VERSION}', "/version")
    doc.keys(
        obj,
        "",
        (
            "format",
            "version",
            "label",
            "description",
            "views",
            "default_view",
            "checks",
            "acceptance",
            "defaults",
        ),
    )
    label = obj.get("label", "expectations")
    if doc.string(label, "/label", max_length=200) is BAD:
        label = "expectations"
    if "description" in obj:
        doc.text(obj["description"], "/description")
    views: dict[str, dict[str, Any]] = {}
    raw_views = obj.get("views", {})
    vobj = doc.object(raw_views, "/views")
    if vobj is not None:
        if len(vobj) > MAX_VIEWS:
            doc.issues.add("LIMIT_EXCEEDED", f"at most {MAX_VIEWS} views", "/views")
        for name, spec in vobj.items():
            p = join("/views", name)
            if doc.string(name, p, max_length=64) is BAD:
                continue
            if spec is None:
                doc.issues.add("INVALID_TYPE", "a view is an object ({} is the active camera)", p)
                continue
            parsed = parse_view_spec(doc, spec, p)
            if parsed is not None:
                views[name] = parsed
    default_view = obj.get("default_view")
    if default_view is not None:
        if doc.string(default_view, "/default_view", max_length=64) is not BAD and (
            default_view not in views
        ):
            doc.issues.add("UNKNOWN_VIEW", f"no view {default_view!r} in views", "/default_view")
    elif "main" in views:
        default_view = "main"
    elif len(views) == 1:
        default_view = next(iter(views))
    defaults = True
    if "defaults" in obj:
        value = doc.boolean(obj["defaults"], "/defaults")
        defaults = True if value is BAD else value
    max_errors, max_warnings, require_vision = 0, None, False
    if "acceptance" in obj:
        acc = doc.object(obj["acceptance"], "/acceptance")
        if acc is not None:
            doc.keys(acc, "/acceptance", ("max_errors", "max_warnings", "require_vision"))
            if "max_errors" in acc:
                v = doc.integer(acc["max_errors"], "/acceptance/max_errors", lo=0)
                max_errors = 0 if v is BAD else v
            if acc.get("max_warnings") is not None:
                v = doc.integer(acc["max_warnings"], "/acceptance/max_warnings", lo=0)
                max_warnings = None if v is BAD else v
            if "require_vision" in acc:
                v = doc.boolean(acc["require_vision"], "/acceptance/require_vision")
                require_vision = False if v is BAD else v
    checks: list[Check] = []
    raw_checks = obj.get("checks", [])
    items = doc.array(raw_checks, "/checks", max_items=MAX_CHECKS)
    seen: set[str] = set()
    for k, item in enumerate(items or []):
        check = _parse_check(doc, item, join("/checks", k), k, views, default_view)
        if check is None:
            continue
        if check.id in seen:
            doc.issues.add("DUPLICATE_ID", f"check id {check.id!r} is used twice", check.pointer)
            continue
        seen.add(check.id)
        checks.append(check)
    doc.issues.raise_if_any("invalid expectations")
    return Expectations(
        label=label,
        views=views,
        checks=tuple(checks),
        default_view=default_view,
        max_errors=max_errors,
        max_warnings=max_warnings,
        require_vision=require_vision,
        defaults=defaults,
        source=json.loads(json.dumps(obj)),
    )


def _parse_check(
    doc: Doc, item: Any, ptr: str, k: int, views: dict[str, Any], default_view: str | None
) -> Check | None:
    obj = doc.object(item, ptr, 'a check {"check": kind, ...}')
    if obj is None:
        return None
    kind = doc.enum(obj.get("check"), join(ptr, "check"), tuple(CHECKS))
    if kind is BAD:
        return None
    spec = CHECKS[kind]
    before = doc.issues.total
    doc.keys(obj, ptr, (*COMMON, *spec["fields"]), tuple(spec["required"]))
    cid = obj.get("id", f"{kind}-{k + 1}")
    if doc.string(cid, join(ptr, "id"), max_length=64) is BAD:
        cid = f"{kind}-{k + 1}"
    severity = obj.get("severity", "error")
    if doc.enum(severity, join(ptr, "severity"), SEVERITIES) is BAD:
        severity = "error"
    label = obj.get("label")
    if "label" in obj and doc.string(label, join(ptr, "label"), max_length=300) is BAD:
        label = None
    params: dict[str, Any] = {}
    for key in spec["fields"]:
        if key not in obj:
            continue
        value = _parse_field(doc, kind, key, obj[key], join(ptr, key), views)
        if value is not BAD:
            params[key] = value
    if kind in ("visible", "framing", "exposure", "camera_clipping") and "view" not in params:
        if default_view is None and views:
            doc.issues.add(
                "MISSING_KEY",
                "this check needs a view (or a default_view)",
                join(ptr, "view"),
                hint=f"views: {', '.join(views)}",
            )
        params["view"] = default_view
    if kind == "distance" and "min" not in params and "max" not in params:
        doc.issues.add("MISSING_KEY", "give min, max or both", ptr)
    if kind == "cad_dimension" and not ({"value", "min", "max"} & set(params)):
        doc.issues.add("MISSING_KEY", "give value (and tolerance), min, max", ptr)
    fix: dict[str, Any] = {}
    if "fix" in obj:
        fix = _parse_fix(doc, obj["fix"], join(ptr, "fix"))
    if doc.issues.total > before:
        return None
    return Check(cid, kind, severity, label, ptr, params, fix)


def _parse_field(doc: Doc, kind: str, key: str, value: Any, ptr: str, views: dict[str, Any]) -> Any:
    if key in ("objects", "against", "support"):
        return doc.selector(value, ptr)
    if key in ("a", "b"):
        return doc.string(value, ptr)
    if key == "view":
        name = doc.string(value, ptr, max_length=64)
        if name is not BAD and name not in views:
            doc.issues.add(
                "UNKNOWN_VIEW",
                f"no view {name!r}",
                ptr,
                hint=f"define it in views ({', '.join(views) or 'none yet'})",
            )
            return BAD
        return name
    if key == "count":
        r = doc.range(value, ptr, lo=0)
        if r is BAD:
            return BAD
        return {
            "min": None if r[0] is None else int(r[0]),
            "max": None if r[1] is None else int(r[1]),
        }
    if key in ("min_pixels",):
        return doc.integer(value, ptr, lo=0, hi=1 << 31)
    if key in (
        "min_visible_fraction",
        "max_clipped_fraction",
        "max_black_fraction",
        "min_coverage",
    ):
        return doc.number(value, ptr, lo=0.0, hi=1.0)
    if key in ("in_frame", "allow_contact", "expect"):
        return doc.boolean(value, ptr)
    if key == "region" and kind == "framing":
        r = doc.rect(value, ptr, normalized=True)
        return BAD if r is BAD else list(r)
    if key == "region":
        return _parse_box_region(doc, value, ptr)
    if key in ("fill", "height", "mean_luminance") and kind != "foot_contacts":
        hi = None if key == "mean_luminance" else 1.0
        r = doc.range(value, ptr, lo=0.0, hi=hi)
        return BAD if r is BAD else {"min": r[0], "max": r[1]}
    if key in ("tolerance", "near_distance"):
        return doc.number(value, ptr, lo=0.0)
    if key in ("min", "max"):
        return doc.number(value, ptr, lo=0.0)
    if key == "max_tilt":
        return doc.number(value, ptr, lo=0.0, hi=180.0)
    if key == "tier":
        return doc.enum(value, ptr, MESH_TIERS)
    if key == "resolution":
        return doc.integer(value, ptr, lo=16, hi=16384)
    if key in ("max_flipped", "max_overlapping"):
        return doc.integer(value, ptr, lo=0, hi=1 << 31)
    if key == "max_stretch":
        return doc.number(value, ptr, lo=1.0)
    if key in ("max_density_spread", "min_gap"):
        return doc.number(value, ptr, lo=0.0)
    if key == "axis":
        v = doc.vector(value, ptr, nonzero=True)
        return BAD if v is BAD else list(v)
    if key == "predicate":
        return doc.enum(value, ptr, RELATION_PREDICATES)
    if key == "material":
        return doc.string(value, ptr)
    if key == "over":
        return doc.enum(value, ptr, ("objects", "image"))
    if key == "items":
        if not isinstance(value, list) or not value or len(value) > 1000:
            doc.issues.add("INVALID_VALUE", "items is a list of 1 to 1000 CAD item names", ptr)
            return BAD
        names = [doc.string(v, join(ptr, k), max_length=128) for k, v in enumerate(value)]
        return BAD if any(n is BAD for n in names) else names
    if key == "fully_constrained":
        return doc.boolean(value, ptr)
    if key in ("reference", "other"):
        obj = doc.object(value, ptr, 'a CAD reference {"part": ..., "face": ...}')
        return BAD if obj is None else dict(obj)
    if key == "quantity":
        return doc.enum(value, ptr, CAD_QUANTITIES)
    if key == "value":
        return doc.number(value, ptr)
    if key == "order":
        return doc.integer(value, ptr, lo=0, hi=2)
    if key == "feet":
        if not isinstance(value, list) or not 1 <= len(value) <= 16:
            doc.issues.add("INVALID_VALUE", "feet is a list of 1 to 16 joint names", ptr)
            return BAD
        names = [doc.string(v, join(ptr, k), max_length=128) for k, v in enumerate(value)]
        return BAD if any(n is BAD for n in names) else names
    if key in ("max_sliding", "max_penetration", "speed"):
        return doc.number(value, ptr, lo=0.0)
    if key == "height" and kind == "foot_contacts":
        return doc.number(value, ptr, lo=0.0, hi=10.0)
    if key == "poses":
        if value == "suite":
            return value
        if not isinstance(value, list) or not 1 <= len(value) <= 32:
            doc.issues.add("INVALID_VALUE", 'poses is "suite" or a list of 1 to 32 pose names',
                           ptr)  # fmt: skip
            return BAD
        names = [doc.string(v, join(ptr, k), max_length=128) for k, v in enumerate(value)]
        return BAD if any(n is BAD for n in names) else names
    if key == "timeline":
        return doc.boolean(value, ptr)
    if key == "ignore":
        if not isinstance(value, list) or not 1 <= len(value) <= 64:
            doc.issues.add("INVALID_VALUE", "ignore is a list of 1 to 64 finding codes", ptr)
            return BAD
        codes = [doc.string(v, join(ptr, k), max_length=64) for k, v in enumerate(value)]
        return BAD if any(c is BAD for c in codes) else codes
    doc.issues.add("UNKNOWN_KEY", f"unknown key {key!r}", ptr)  # pragma: no cover
    return BAD


def _parse_box_region(doc: Doc, value: Any, ptr: str) -> Any:
    obj = doc.object(value, ptr, 'a region {"box": {"min": [x, y, z], "max": [x, y, z]}}')
    if obj is None:
        return BAD
    if not doc.keys(obj, ptr, ("box",), ("box",)):
        return BAD
    box = doc.object(obj["box"], join(ptr, "box"))
    if box is None or not doc.keys(box, join(ptr, "box"), ("min", "max"), ("min", "max")):
        return BAD
    lo = doc.vector(box["min"], join(ptr, "box/min"))
    hi = doc.vector(box["max"], join(ptr, "box/max"))
    if lo is BAD or hi is BAD:
        return BAD
    if any(a > b for a, b in zip(lo, hi, strict=True)):
        doc.issues.add("INVALID_VALUE", "the box's min exceeds its max", join(ptr, "box"))
        return BAD
    return {"box": {"min": list(lo), "max": list(hi)}}


def _parse_fix(doc: Doc, value: Any, ptr: str) -> dict[str, Any]:
    """Correction preferences: which side may move (``move``), whether the camera may change
    (``camera``), and which correction actions are allowed (``allow``)."""
    from .corrections import ACTIONS

    obj = doc.object(value, ptr, 'correction preferences {"move", "camera", "allow"}')
    if obj is None:
        return {}
    doc.keys(obj, ptr, FIX_KEYS)
    out: dict[str, Any] = {}
    if "move" in obj:
        v = doc.enum(obj["move"], join(ptr, "move"), MOVE_CHOICES)
        if v is not BAD:
            out["move"] = v
    if "camera" in obj:
        v = doc.boolean(obj["camera"], join(ptr, "camera"))
        if v is not BAD:
            out["camera"] = v
    if "allow" in obj:
        items = doc.array(obj["allow"], join(ptr, "allow"), min_items=0, max_items=len(ACTIONS))
        if items is not None:
            allowed = []
            for i, item in enumerate(items):
                v = doc.enum(item, join(join(ptr, "allow"), i), tuple(ACTIONS))
                if v is not BAD:
                    allowed.append(v)
            out["allow"] = allowed
    return out
