"""Diagnosis: expectations checked against the scene and its observations, as findings.

Every finding says what was found (a stable ``code``), how sure that is and why:

* ``basis`` -- ``"scene"`` (the scene's own data: what exists, materials, flags),
  ``"geometry"`` (computed from the triangles), ``"render"`` (measured on a rendered
  image, one representative sample per pixel), ``"sampled"`` (estimated from samples),
  ``"heuristic"`` (a default threshold applied to measured facts, not a stated
  expectation), ``"vision"`` (an image-aware model's judgment, as declared by its
  provider) or ``"agent"`` (a judgment supplied without a declared image input);
* ``exact`` -- whether the measurement is exact for its definition;
* ``evidence`` -- the measured values, ``magnitude`` and ``unit`` -- how far off it is;
* ``subjects`` -- the objects concerned, by id and name, with their role, and ``region``
  -- where they appear in an observation's image.

Findings never claim more than their basis supports: an exact overlap is a fact; a
default exposure threshold is a heuristic; whether an image matches a request is only
ever a vision or agent judgment.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, FeedbackError, PlanError
from .expectations import DEFAULT_TOLERANCE, Check, Expectations

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject
    from ..spatial.index import Entry, SpatialIndex
    from .observe import Observation

SEVERITY_WEIGHT = {"error": 1.0, "warning": 0.25, "info": 0.0}
BASES = ("scene", "geometry", "render", "sampled", "heuristic", "vision", "agent")
MAX_OVERLAP_PAIRS = 200
"""Intersecting pairs one no_overlap check classifies (contact or penetration)."""
MAX_DEFAULT_OBJECTS = 500
"""Visible objects up to which the default checks scan for penetrations."""
MAX_CLIPPING_OBJECTS = 200
DEFAULT_DARK = 0.01
"""Default rule: mean linear luminance of the objects' pixels below this is underexposed."""
DEFAULT_CLIPPED = 0.5
"""Default rule: more than this fraction of the objects' pixels clipped is overexposed."""


@dataclass(frozen=True)
class Finding:
    """One problem found (see the module docstring)."""

    key: str
    check: str
    kind: str
    code: str
    severity: str
    basis: str
    exact: bool
    message: str
    subjects: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    magnitude: float | None = None
    unit: str | None = None
    view: str | None = None
    observation: str | None = None
    region: tuple[int, int, int, int] | None = None
    hint: str | None = None
    id: str = ""

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(s["id"] for s in self.subjects if s.get("id"))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "check": self.check,
            "kind": self.kind,
            "code": self.code,
            "severity": self.severity,
            "basis": self.basis,
            "exact": self.exact,
            "message": self.message,
            "subjects": list(self.subjects),
        }
        if self.magnitude is not None:
            out["magnitude"] = _clean(self.magnitude)
            out["unit"] = self.unit
        if self.view is not None:
            out["view"] = self.view
        if self.observation is not None:
            out["observation"] = self.observation
        if self.region is not None:
            out["region"] = list(self.region)
        if self.evidence:
            out["evidence"] = _clean(self.evidence)
        if self.hint:
            out["hint"] = self.hint
        out["key"] = self.key
        return out


def _clean(value: Any) -> Any:
    """JSON-safe: non-finite floats become strings ("inf"), tuples lists."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


@dataclass
class CheckResult:
    id: str
    kind: str
    severity: str
    status: str  # "pass" | "fail" | "unknown"
    label: str | None = None
    findings: list[str] = field(default_factory=list)
    measured: dict[str, Any] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "check": self.kind,
            "severity": self.severity,
            "status": self.status,
            "findings": self.findings,
        }
        if self.label:
            out["label"] = self.label
        if self.measured:
            out["measured"] = _clean(self.measured)
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class Diagnosis:
    """The result of checking a scene: checks, findings, acceptance."""

    id: str
    scene: Scene
    revision: int
    expectations: Expectations
    checks: list[CheckResult]
    findings: list[Finding]
    observations: dict[str, str]
    errors: int
    warnings: int
    score: float
    accepted: bool
    acceptance: dict[str, Any]
    vision: dict[str, Any] | None = None
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    observed: dict[str, Observation] = field(default_factory=dict, repr=False)
    evaluations: list[Any] = field(default_factory=list, repr=False)
    transient: bool = False

    @property
    def stale(self) -> bool:
        return self.transient or self.scene.revision != self.revision

    def finding(self, ref: str) -> Finding:
        for f in self.findings:
            if f.id == ref or f.key == ref:
                return f
        raise FeedbackError("UNKNOWN_FINDING", f"no finding {ref!r} in {self.id}", "finding")

    def failing(self) -> dict[str, Finding]:
        """Findings that count against acceptance (errors and warnings), by key."""
        return {f.key: f for f in self.findings if f.severity in ("error", "warning")}

    def to_dict(self, *, max_findings: int = 100) -> dict[str, Any]:
        shown = self.findings[:max_findings]
        out: dict[str, Any] = {
            "format": "aether3d.diagnosis",
            "version": 1,
            "id": self.id,
            "revision": self.revision,
            "stale": self.stale,
            "expectations": self.expectations.label,
            "accepted": self.accepted,
            "acceptance": self.acceptance,
            "summary": {
                "errors": self.errors,
                "warnings": self.warnings,
                "info": sum(1 for f in self.findings if f.severity == "info"),
                "score": self.score,
                "checks": {
                    s: sum(1 for c in self.checks if c.status == s)
                    for s in ("pass", "fail", "unknown")
                },
            },
            "checks": [c.to_dict() for c in self.checks],
            "findings": [f.to_dict() for f in shown],
            "observations": self.observations,
        }
        if len(self.findings) > len(shown):
            out["findings_omitted"] = len(self.findings) - len(shown)
        if self.vision is not None:
            out["vision"] = self.vision
        if self.notes:
            out["notes"] = self.notes
        out["timings"] = {k: round(v, 6) for k, v in self.timings.items()}
        return out


def _ref(obj: Any, role: str) -> dict[str, Any]:
    return {"id": obj.id, "name": obj.name, "role": role}


class Diagnoser:
    """Runs the checks of one expectations document on a scene at one revision."""

    def __init__(
        self,
        scene: Scene,
        expectations: Expectations,
        observation: Callable[[str], Observation],
    ) -> None:
        from ..spatial.index import get_index

        self.scene = scene
        self.expectations = expectations
        self.observation = observation
        self.index: SpatialIndex = get_index(scene)
        self.findings: list[Finding] = []
        self.results: list[CheckResult] = []
        self.notes: list[str] = []

    # -- helpers ---------------------------------------------------------------

    def select(self, selector: Any, pointer: str) -> tuple[list[SceneObject], list[str]]:
        """Objects of a selector, and the references it names that do not exist.

        Names and ids given as a string or a list of strings are looked up one by one, so a
        missing one is reported as missing (``MISSING_ELEMENT``) while the others are
        checked; selector objects go through :func:`aether3d.plans.select_objects`.
        """
        from ..plans import select_objects

        if isinstance(selector, str) or (
            isinstance(selector, list) and all(isinstance(i, str) for i in selector)
        ):
            found, unknown = [], []
            for item in [selector] if isinstance(selector, str) else selector:
                obj = self.scene._objects.get(item) or self.scene._by_name.get(item)
                if obj is None:
                    unknown.append(item)
                elif obj not in found:
                    found.append(obj)
            order = {o.id: k for k, o in enumerate(self.scene)} if len(found) > 1 else {}
            found.sort(key=lambda o: order.get(o.id, 0))
            return found, unknown
        try:
            return select_objects(self.scene, selector, pointer), []
        except PlanError as exc:
            if exc.code == "UNKNOWN_REFERENCE":
                return [], [exc.message]
            raise FeedbackError(
                exc.code, exc.message, exc.pointer, hint=exc.hint, details=exc.details
            ) from None

    def entry(self, obj: SceneObject) -> Entry | None:
        return self.index.by_id.get(obj.id)

    def add(self, result: CheckResult, finding: Finding) -> None:
        self.findings.append(finding)
        result.findings.append(finding.key)

    def finding(self, check: Check, code: str, message: str, subjects=(), **kw) -> Finding:
        ids = "+".join(s["id"] for s in subjects) or "-"
        severity = kw.pop("severity", check.severity)
        basis = kw.pop("basis", "geometry")
        exact = kw.pop("exact", True)
        return Finding(
            key=f"{check.id}|{code}|{ids}",
            check=check.id,
            kind=check.kind,
            code=code,
            severity=severity,
            basis=basis,
            exact=exact,
            message=message,
            subjects=tuple(subjects),
            **kw,
        )

    def missing(self, check: Check, result: CheckResult, names: list[str]) -> None:
        for name in names:
            self.add(
                result,
                Finding(
                    key=f"{check.id}|MISSING_ELEMENT|{name}",
                    check=check.id,
                    kind=check.kind,
                    code="MISSING_ELEMENT",
                    severity=check.severity,
                    basis="scene",
                    exact=True,
                    message=f"no object {name!r} exists",
                    evidence={"reference": name},
                    hint="create it (execute_scene_plan), or correct the expectation",
                ),
            )

    # -- the run ---------------------------------------------------------------

    def run(self) -> None:
        for check in self.expectations.checks:
            result = CheckResult(check.id, check.kind, check.severity, "pass", check.label)
            try:
                getattr(self, f"_check_{check.kind}")(check, result)
            except FeedbackError as exc:
                result.status = "unknown"
                result.note = f"[{exc.code}] {exc.message}"
            except AetherError as exc:
                result.status = "unknown"
                code = getattr(exc, "code", type(exc).__name__)
                result.note = f"[{code}] {getattr(exc, 'message', str(exc))}"
            if result.findings and result.status == "pass":
                result.status = "fail"
            self.results.append(result)

    # -- the checks ------------------------------------------------------------

    def _check_exists(self, check: Check, result: CheckResult) -> None:
        objs, unknown = self.select(check.params["objects"], f"{check.pointer}/objects")
        count = check.params.get("count", {"min": 1, "max": None})
        lo, hi = count.get("min"), count.get("max")
        if lo is None:
            lo = 0
        result.measured = {"count": len(objs)}
        if unknown and isinstance(check.params["objects"], (str, list)):
            self.missing(check, result, [str(u) for u in unknown])
            return
        if len(objs) < lo:
            self.add(
                result,
                self.finding(
                    check,
                    "MISSING_ELEMENT",
                    f"{len(objs)} object(s) match, at least {lo} expected",
                    basis="scene",
                    evidence={"count": len(objs), "min": lo, "selector": check.params["objects"]},
                    magnitude=float(lo - len(objs)),
                    unit="objects",
                    hint="create the missing objects (execute_scene_plan)",
                ),
            )
        if hi is not None and len(objs) > hi:
            self.add(
                result,
                self.finding(
                    check,
                    "TOO_MANY",
                    f"{len(objs)} object(s) match, at most {hi} expected",
                    [_ref(o, "subject") for o in objs[:20]],
                    basis="scene",
                    evidence={"count": len(objs), "max": hi},
                    magnitude=float(len(objs) - hi),
                    unit="objects",
                ),
            )

    def _geometry_objects(self, check: Check, result: CheckResult, key: str = "objects"):
        objs, unknown = self.select(check.params[key], f"{check.pointer}/{key}")
        if unknown:
            self.missing(check, result, [str(u) for u in unknown])
        out = []
        for obj in objs:
            if obj.camera is not None or obj.light is not None:
                continue
            out.append(obj)
        if not objs and not unknown:
            result.status = "unknown"
            result.note = "the selector matches no object"
        return out

    def _check_visible(self, check: Check, result: CheckResult) -> None:
        objs = self._geometry_objects(check, result)
        obs = self.observation(check.view)
        min_pixels = check.params.get("min_pixels", 1)
        min_fraction = check.params.get("min_visible_fraction")
        in_frame = check.params.get("in_frame", False)
        measured = {}
        for obj in objs:
            subject = [_ref(obj, "subject")]
            base = {"view": check.view, "observation": obs.id}
            if not obj.visible:
                self.add(
                    result,
                    self.finding(
                        check,
                        "HIDDEN",
                        f"{obj.name!r} is hidden (visible is false)",
                        subject,
                        basis="scene",
                        hint="set visible to true",
                        **base,
                    ),
                )
                continue
            instance = obs.instance_of(obj)
            pixels = 0 if instance is None else int(obs.stats.pixels[instance])
            geometry = obs.geometry(obj, occlusion=min_fraction is not None or pixels < min_pixels)
            if geometry is None:
                self.add(
                    result,
                    self.finding(
                        check,
                        "NO_GEOMETRY",
                        f"{obj.name!r} has no geometry to see",
                        subject,
                        basis="scene",
                        **base,
                    ),
                )
                continue
            if "error" in geometry:
                self.add(
                    result,
                    self.finding(
                        check,
                        "NOT_RENDERED",
                        f"{obj.name!r} cannot be drawn: {geometry['error'].get('message')}",
                        subject,
                        basis="scene",
                        **base,
                    ),
                )
                continue
            occlusion = geometry.get("occlusion") or {}
            footprint = occlusion.get("footprint_pixels")
            fraction = occlusion.get("visible_fraction")
            measured[obj.name] = {
                "pixels": pixels,
                "frustum": geometry["frustum"],
                "footprint_pixels": footprint,
                "visible_fraction": fraction,
                "in_frame_fraction": geometry.get("in_frame_fraction"),
            }
            region = _bounds(obs, instance)
            evidence = {
                "visible_pixels": pixels,
                "frustum": geometry["frustum"],
                "depth_extent": geometry["depth_extent"],
                "projected_bounds": geometry.get("projected_bounds"),
                "in_frame_fraction": geometry.get("in_frame_fraction"),
            }
            if occlusion.get("computed"):
                evidence["occlusion"] = occlusion
            if geometry["frustum"] == "outside":
                behind = geometry["depth_extent"][1] <= 0.0
                self.add(
                    result,
                    self.finding(
                        check,
                        "OUT_OF_VIEW",
                        f"{obj.name!r} is "
                        + ("behind the camera" if behind else "outside the view volume")
                        + f" of view {check.view or 'camera'!r}",
                        subject,
                        basis="geometry",
                        evidence=evidence,
                        hint="frame the camera on it (frame_camera) or move it into view",
                        **base,
                    ),
                )
                continue
            if pixels < min_pixels:
                if footprint is not None and footprint < min_pixels:
                    code, why = (
                        "TOO_SMALL",
                        (
                            f"{obj.name!r} would cover only {footprint} pixel(s) even unoccluded "
                            f"(at least {min_pixels} expected)"
                        ),
                    )
                    magnitude, unit = float(min_pixels - footprint), "pixels"
                elif fraction is not None and footprint:
                    blockers = ", ".join(o["name"] for o in occlusion.get("occluders", [])[:3])
                    code, why = (
                        "OCCLUDED",
                        (
                            f"{obj.name!r} shows {pixels} of {footprint} pixels"
                            + (f", hidden by {blockers}" if blockers else "")
                        ),
                    )
                    magnitude, unit = 1.0 - (fraction or 0.0), "fraction"
                else:
                    code, why = "NOT_VISIBLE", f"{obj.name!r} shows {pixels} pixel(s)"
                    magnitude, unit = float(min_pixels - pixels), "pixels"
                self.add(
                    result,
                    self.finding(
                        check,
                        code,
                        why,
                        subject,
                        basis="render",
                        evidence=evidence,
                        magnitude=magnitude,
                        unit=unit,
                        region=region,
                        **base,
                    ),
                )
                continue
            if min_fraction is not None and fraction is not None and fraction < min_fraction:
                blockers = ", ".join(o["name"] for o in occlusion.get("occluders", [])[:3])
                self.add(
                    result,
                    self.finding(
                        check,
                        "OCCLUDED",
                        f"{fraction:.0%} of {obj.name!r}'s footprint is visible "
                        f"({min_fraction:.0%} expected)"
                        + (f"; in front: {blockers}" if blockers else ""),
                        subject
                        + [
                            _ref(self.scene.get(o["id"]), "occluder")
                            for o in occlusion.get("occluders", [])[:3]
                        ],
                        basis="render",
                        evidence=evidence,
                        magnitude=min_fraction - fraction,
                        unit="fraction",
                        region=region,
                        hint="move the camera, or the objects in front, or hide them",
                        **base,
                    ),
                )
            if in_frame:
                inside = geometry.get("in_frame_fraction")
                touches = (
                    instance is not None
                    and pixels
                    and any(obs.pixel_record(instance).get("touches_border", {}).values())
                )
                if (inside is not None and inside < 1.0) or touches:
                    self.add(
                        result,
                        self.finding(
                            check,
                            "PARTIALLY_OUT_OF_FRAME",
                            f"{obj.name!r} extends past the image border",
                            subject,
                            basis="geometry",
                            evidence=evidence,
                            magnitude=1.0 - (inside if inside is not None else 1.0),
                            unit="fraction",
                            region=region,
                            hint="frame the camera on the objects (frame_camera)",
                            **base,
                        ),
                    )
        result.measured = measured

    def _check_framing(self, check: Check, result: CheckResult) -> None:
        objs = self._geometry_objects(check, result)
        if not objs:
            return
        obs = self.observation(check.view)
        w, h = obs.width, obs.height
        s = obs.stats
        instances = [obs.instance_of(o) for o in objs]
        rows = [k for k in instances if k is not None and s.pixels[k] > 0]
        subjects = [_ref(o, "subject") for o in objs[:10]]
        base = {"view": check.view, "observation": obs.id}
        if not rows:
            self.add(
                result,
                self.finding(
                    check,
                    "NOT_VISIBLE",
                    "none of the objects is visible in the view",
                    subjects,
                    basis="render",
                    hint="frame the camera on them",
                    **base,
                ),
            )
            return
        pixels = float(sum(int(s.pixels[k]) for k in rows))
        cx = sum(float(s.cx[k]) * int(s.pixels[k]) for k in rows) / pixels
        cy = sum(float(s.cy[k]) * int(s.pixels[k]) for k in rows) / pixels
        x0 = min(int(s.x0[k]) for k in rows)
        y0 = min(int(s.y0[k]) for k in rows)
        x1 = max(int(s.x1[k]) for k in rows)
        y1 = max(int(s.y1[k]) for k in rows)
        center = (cx / w, cy / h)
        fill = pixels / (w * h)
        height = (y1 - y0) / h
        result.measured = {
            "center": [round(center[0], 6), round(center[1], 6)],
            "fill": fill,
            "height": height,
            "bounds": [x0, y0, x1, y1],
        }
        evidence = dict(result.measured)
        region = (x0, y0, x1, y1)
        if "region" in check.params:
            rx0, ry0, rx1, ry1 = check.params["region"]
            dx = max(rx0 - center[0], 0.0, center[0] - rx1)
            dy = max(ry0 - center[1], 0.0, center[1] - ry1)
            if dx > 0.0 or dy > 0.0:
                self.add(
                    result,
                    self.finding(
                        check,
                        "OFF_CENTER",
                        f"the objects' center is at {center[0]:.2f}, {center[1]:.2f} of the "
                        f"image; expected within {check.params['region']}",
                        subjects,
                        basis="render",
                        evidence={**evidence, "target_region": check.params["region"]},
                        magnitude=math.hypot(dx, dy),
                        unit="image fraction",
                        region=region,
                        hint="aim the camera (aim_camera) or frame it on the objects",
                        **base,
                    ),
                )
        for key, value, low_code, high_code in (
            ("fill", fill, "TOO_SMALL_IN_FRAME", "TOO_LARGE_IN_FRAME"),
            ("height", height, "TOO_SMALL_IN_FRAME", "TOO_LARGE_IN_FRAME"),
        ):
            if key not in check.params:
                continue
            lo, hi = check.params[key]["min"], check.params[key]["max"]
            if lo is not None and value < lo:
                code, bound = low_code, lo
            elif hi is not None and value > hi:
                code, bound = high_code, hi
            else:
                continue
            self.add(
                result,
                self.finding(
                    check,
                    code,
                    f"the objects' {key} is {value:.3f} of the image; expected "
                    f"{'at least' if code == low_code else 'at most'} {bound:.3f}",
                    subjects,
                    basis="render",
                    evidence={**evidence, "measure": key, "expected": check.params[key]},
                    magnitude=abs(math.log(max(value, 1e-9) / max(bound, 1e-9))),
                    unit="log ratio",
                    region=region,
                    hint="move the camera closer or farther (dolly_camera), or frame it",
                    **base,
                ),
            )

    def _support_entries(self, obj: SceneObject, supports: list[SceneObject] | None):
        if supports is not None:
            return [
                e
                for o in supports
                if o is not obj and (e := self.index.by_id.get(o.id)) is not None
            ]
        own = {obj.id} | {d.id for d in obj.iter_descendants()}
        entry = self.index.by_id[obj.id]
        lo, hi = entry.box
        out = []
        for e in self.index.entries:
            if e.obj.id in own or not e.visible:
                continue
            elo, ehi = e.box
            if np.any(elo[:2] > hi[:2]) or np.any(ehi[:2] < lo[:2]) or elo[2] > hi[2]:
                continue
            out.append(e)
        return out

    def _check_rests_on(self, check: Check, result: CheckResult) -> None:
        from .geometry import vertical_contact

        objs = self._geometry_objects(check, result)
        tol = check.params.get("tolerance", DEFAULT_TOLERANCE)
        supports = None
        if "support" in check.params:
            supports, unknown = self.select(check.params["support"], f"{check.pointer}/support")
            if unknown:
                self.missing(check, result, [str(u) for u in unknown])
                return
            if not supports:
                result.status = "unknown"
                result.note = "the support selector matches no object"
                return
        measured = {}
        for obj in objs:
            entry = self.entry(obj)
            subject = [_ref(obj, "subject")]
            if entry is None:
                continue
            candidates = self._support_entries(obj, supports)
            if not candidates:
                self.add(
                    result,
                    self.finding(
                        check,
                        "NO_SUPPORT",
                        f"nothing is below {obj.name!r} to rest on",
                        subject,
                        evidence={"tolerance": tol},
                        hint="move it over its support, or name the support",
                    ),
                )
                continue
            r = vertical_contact(self.index, entry, candidates, tol)
            measured[obj.name] = {k: r[k] for k in ("status", "gap", "penetration")}
            evidence = {**r, "tolerance": tol}
            support = r.get("gap_support") or r.get("penetrated_support")
            subjects = subject + ([{**support, "role": "support"}] if support else [])
            if r["status"] == "floating":
                self.add(
                    result,
                    self.finding(
                        check,
                        "FLOATING",
                        f"{obj.name!r} floats {r['gap']:.4g} m above "
                        f"{support['name'] if support else 'its support'!r}",
                        subjects,
                        exact=r["exact"],
                        evidence=evidence,
                        magnitude=r["gap"],
                        unit="m",
                        hint="lower it onto its support",
                    ),
                )
            elif r["status"] == "sunken":
                self.add(
                    result,
                    self.finding(
                        check,
                        "SUNKEN",
                        f"{obj.name!r} sinks {r['penetration']:.4g} m into "
                        f"{support['name'] if support else 'its support'!r}",
                        subjects,
                        exact=r["exact"],
                        evidence=evidence,
                        magnitude=r["penetration"],
                        unit="m",
                        hint="raise it onto its support's surface",
                    ),
                )
            elif r["status"] == "not_above":
                names = ", ".join(e.obj.name for e in candidates[:3])
                self.add(
                    result,
                    self.finding(
                        check,
                        "NOT_ON_SUPPORT",
                        f"{obj.name!r} is not above its support ({names})",
                        subject + [_ref(e.obj, "support") for e in candidates[:3]],
                        exact=r["exact"],
                        evidence=evidence,
                        hint="move it over the support",
                    ),
                )
        result.measured = measured

    def _check_no_overlap(self, check: Check, result: CheckResult) -> None:
        objs = self._geometry_objects(check, result)
        tol = check.params.get("tolerance", DEFAULT_TOLERANCE)
        allow_contact = check.params.get("allow_contact", True)
        against = None
        if "against" in check.params:
            against = self._geometry_objects(check, result, "against")
        found = self.penetrations(objs, against, tol, allow_contact)
        pairs = found["pairs"]
        result.measured = {
            "intersecting_pairs": found["intersecting"],
            "contacts": found["contacts"],
            "penetrating": len(pairs),
        }
        if found["truncated"]:
            result.note = f"only the first {MAX_OVERLAP_PAIRS} intersecting pairs were classified"
        mine = {o.id for o in objs}
        theirs = mine if against is None else {o.id for o in against}

        def side(o: SceneObject) -> str:
            if o.id in mine and o.id in theirs:
                return "either"
            return "subject" if o.id in mine else "against"

        for pair in pairs:
            a, b = pair["a"], pair["b"]
            subjects = [_ref(a, side(a)), _ref(b, side(b))]
            if pair["kind"] == "contained":
                message = f"{pair['inner'].name!r} lies inside {pair['outer'].name!r}"
                code, magnitude = "CONTAINED", None
            else:
                depth = pair["separation"][0]["distance"]
                message = (
                    f"{a.name!r} intersects {b.name!r} (separated by moving {a.name!r} "
                    f"{depth:.4g} m along {pair['separation'][0]['direction']})"
                )
                code, magnitude = "INTERSECTION", depth
            self.add(
                result,
                self.finding(
                    check,
                    code,
                    message,
                    subjects,
                    evidence={
                        "intersecting": True,
                        "contained": pair.get("contained"),
                        "separations": pair.get("separation"),
                        "tolerance": tol,
                        "method": "exact triangle intersection; separating translations "
                        "along the six axis directions, each verified by the exact test "
                        "(the true minimum translation can be shorter in other directions)",
                    },
                    magnitude=magnitude,
                    unit=None if magnitude is None else "m",
                    hint="move one of them apart",
                ),
            )

    def penetrations(
        self,
        objs: list[SceneObject],
        against: list[SceneObject] | None,
        tol: float,
        allow_contact: bool,
    ) -> dict[str, Any]:
        """Intersecting pairs among ``objs`` (or against ``against``), classified."""
        from ..spatial.queries import overlaps
        from .geometry import classify_contact, separation

        if not objs or (against is not None and not against):
            return {"pairs": [], "intersecting": 0, "contacts": 0, "truncated": False}
        found = overlaps(
            self.scene,
            objects=[o.id for o in objs],
            against=None if against is None else [o.id for o in against],
        )
        out, contacts, truncated = [], 0, False
        for k, ov in enumerate(found):
            if k >= MAX_OVERLAP_PAIRS:
                truncated = True
                break
            a = self.scene._objects[ov.a.id]
            b = self.scene._objects[ov.b.id]
            ea, eb = self.index.by_id[a.id], self.index.by_id[b.id]
            if ov.contained:
                inner, outer = (a, b) if ov.contained == "a_in_b" else (b, a)
                out.append(
                    {
                        "kind": "contained",
                        "a": a,
                        "b": b,
                        "inner": inner,
                        "outer": outer,
                        "contained": ov.contained,
                    }
                )
                continue
            if allow_contact and classify_contact(ea, eb, tol)["contact"]:
                contacts += 1
                continue
            from .geometry import AXES

            sep = separation(ea, eb, AXES, tolerance=tol)
            out.append({"kind": "intersection", "a": a, "b": b, "separation": sep})
        return {
            "pairs": out,
            "intersecting": len(found),
            "contacts": contacts,
            "truncated": truncated,
        }

    def _check_inside_region(self, check: Check, result: CheckResult) -> None:
        objs = self._geometry_objects(check, result)
        box = check.params["region"]["box"]
        rlo, rhi = np.array(box["min"]), np.array(box["max"])
        for obj in objs:
            entry = self.entry(obj)
            if entry is None:
                continue
            lo, hi = entry.box
            below = np.maximum(rlo - lo, 0.0)
            above = np.maximum(hi - rhi, 0.0)
            excess = float(max(below.max(), above.max()))
            if excess > 0.0:
                self.add(
                    result,
                    self.finding(
                        check,
                        "OUT_OF_REGION",
                        f"{obj.name!r} extends {excess:.4g} m outside the region",
                        [_ref(obj, "subject")],
                        evidence={
                            "bounds": {"min": lo.tolist(), "max": hi.tolist()},
                            "region": box,
                            "below_min": below.tolist(),
                            "above_max": above.tolist(),
                            "fits": bool(np.all(hi - lo <= rhi - rlo)),
                        },
                        magnitude=excess,
                        unit="m",
                        hint="move it into the region",
                    ),
                )

    def _two(self, check: Check, result: CheckResult):
        out = []
        for key in ("a", "b"):
            ref = check.params[key]
            obj = self.scene._objects.get(ref) or self.scene._by_name.get(ref)
            if obj is None:
                self.missing(check, result, [ref])
                return None
            out.append(obj)
        return out

    def _check_distance(self, check: Check, result: CheckResult) -> None:
        pair = self._two(check, result)
        if pair is None:
            return
        a, b = pair
        d = self.scene.distance(a, b)
        lo, hi = check.params.get("min"), check.params.get("max")
        result.measured = {"distance": d.distance}
        subjects = [_ref(a, "subject"), _ref(b, "other")]
        evidence = {"distance": d.distance, "point_a": d.point_a, "point_b": d.point_b}
        if lo is not None and d.distance < lo:
            code, off = "TOO_CLOSE", lo - d.distance
        elif hi is not None and d.distance > hi:
            code, off = "TOO_FAR", d.distance - hi
        else:
            return
        self.add(
            result,
            self.finding(
                check,
                code,
                f"{a.name!r} is {d.distance:.4g} m from {b.name!r}; expected "
                + (f"at least {lo:g}" if code == "TOO_CLOSE" else f"at most {hi:g}"),
                subjects,
                evidence={**evidence, "min": lo, "max": hi},
                magnitude=off,
                unit="m",
            ),
        )

    def _check_relation(self, check: Check, result: CheckResult) -> None:
        pair = self._two(check, result)
        if pair is None:
            return
        a, b = pair
        predicate = check.params["predicate"]
        near = check.params.get("near_distance")
        if predicate == "near" and near is None:
            raise FeedbackError("INVALID_VALUE", "predicate near needs near_distance", "check")
        rel = self.scene.relation(a, b, near_distance=near)
        value = getattr(rel, predicate)
        expect = check.params.get("expect", True)
        result.measured = {predicate: value}
        if value is None:
            result.status = "unknown"
            result.note = f"{predicate} is undefined here (it needs a closed solid)"
            return
        if bool(value) != expect:
            self.add(
                result,
                self.finding(
                    check,
                    "RELATION_VIOLATED",
                    f"{a.name!r} is {'' if value else 'not '}{predicate.replace('_', ' ')} "
                    f"{b.name!r}",
                    [_ref(a, "subject"), _ref(b, "other")],
                    exact=predicate not in ("above", "below"),
                    evidence={"relation": rel.to_dict(), "predicate": predicate, "expect": expect},
                ),
            )

    def _check_upright(self, check: Check, result: CheckResult) -> None:
        from ..rendering.scene_data import world_matrices

        objs = self._geometry_objects(check, result)
        limit = check.params.get("max_tilt", 2.0)
        axis = np.asarray(check.params.get("axis", (0.0, 0.0, 1.0)), dtype=np.float64)
        worlds = world_matrices(self.scene)
        measured = {}
        for obj in objs:
            world = worlds[obj.id]
            up = world[:3, :3] @ axis
            norm = float(np.linalg.norm(up))
            if not norm or not math.isfinite(norm):
                continue
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2]) / norm))))
            measured[obj.name] = tilt
            if tilt > limit:
                self.add(
                    result,
                    self.finding(
                        check,
                        "TILTED",
                        f"{obj.name!r} is tilted {tilt:.2f} degrees from upright",
                        [_ref(obj, "subject")],
                        evidence={"tilt": tilt, "max_tilt": limit, "world_axis": up.tolist()},
                        magnitude=tilt - limit,
                        unit="degrees",
                        hint="turn it upright (keeps its heading)",
                    ),
                )
        result.measured = measured

    def _check_material(self, check: Check, result: CheckResult) -> None:
        objs = self._geometry_objects(check, result)
        want = check.params["material"]
        library = self.scene.materials
        material = next((m for m in library.materials if want in (m.id, m.name)), None)
        if material is None:
            self.add(
                result,
                self.finding(
                    check,
                    "MISSING_MATERIAL",
                    f"there is no material {want!r}",
                    basis="scene",
                    evidence={"material": want},
                    hint="create it (execute_scene_plan materials), or correct the expectation",
                ),
            )
            return
        for obj in objs:
            current = obj.material
            if current is None or current.id != material.id:
                self.add(
                    result,
                    self.finding(
                        check,
                        "WRONG_MATERIAL",
                        f"{obj.name!r} uses "
                        + (repr(current.name) if current is not None else "its own color")
                        + f", expected {material.name!r}",
                        [_ref(obj, "subject")],
                        basis="scene",
                        evidence={
                            "current": None if current is None else current.name,
                            "expected": material.name,
                            "expected_id": material.id,
                        },
                        hint="assign the material",
                    ),
                )

    def _check_exposure(self, check: Check, result: CheckResult) -> None:
        obs = self.observation(check.view)
        st = obs.image_stats
        over = check.params.get("over", "objects")
        summary = st["luminance_covered"] if over == "objects" else st["luminance"]
        base = {"view": check.view, "observation": obs.id}
        result.measured = {
            "mean_luminance": None if summary is None else summary["mean"],
            "clipped_fraction": st["clipped_fraction"],
            "black_fraction": st["black_fraction"],
            "coverage": st["coverage"],
            "luminance_source": st["luminance_source"],
        }
        evidence = {"stats": st, "over": over}
        if "min_coverage" in check.params and st["coverage"] < check.params["min_coverage"]:
            self.add(
                result,
                self.finding(
                    check,
                    "EMPTY_VIEW",
                    f"geometry covers {st['coverage']:.1%} of the image "
                    f"(at least {check.params['min_coverage']:.1%} expected)",
                    basis="render",
                    evidence=evidence,
                    magnitude=check.params["min_coverage"] - st["coverage"],
                    unit="fraction",
                    hint="frame the camera on the scene",
                    **base,
                ),
            )
        if summary is None:
            if "mean_luminance" in check.params:
                self.add(
                    result,
                    self.finding(
                        check,
                        "EMPTY_VIEW",
                        "no object is visible to measure",
                        basis="render",
                        evidence=evidence,
                        hint="frame the camera on the scene",
                        **base,
                    ),
                )
            return
        mean = summary["mean"]
        if "mean_luminance" in check.params:
            lo, hi = check.params["mean_luminance"]["min"], check.params["mean_luminance"]["max"]
            if lo is not None and mean < lo:
                stops = math.log2(max(lo, 1e-12) / max(mean, 1e-12))
                self.add(
                    result,
                    self.finding(
                        check,
                        "UNDEREXPOSED",
                        f"mean luminance {mean:.4g} is below {lo:g} ({stops:.2f} stops)",
                        basis="render",
                        evidence={**evidence, "target": check.params["mean_luminance"]},
                        magnitude=stops,
                        unit="stops",
                        hint="raise the exposure or the lights",
                        **base,
                    ),
                )
            elif hi is not None and mean > hi:
                stops = math.log2(max(mean, 1e-12) / max(hi, 1e-12))
                self.add(
                    result,
                    self.finding(
                        check,
                        "OVEREXPOSED",
                        f"mean luminance {mean:.4g} is above {hi:g} ({stops:.2f} stops)",
                        basis="render",
                        evidence={**evidence, "target": check.params["mean_luminance"]},
                        magnitude=stops,
                        unit="stops",
                        hint="lower the exposure or the lights",
                        **base,
                    ),
                )
        if "max_clipped_fraction" in check.params:
            limit = check.params["max_clipped_fraction"]
            value = st["clipped_covered_fraction"] if over == "objects" else st["clipped_fraction"]
            if value > limit:
                self.add(
                    result,
                    self.finding(
                        check,
                        "CLIPPED_HIGHLIGHTS",
                        f"{value:.1%} of the pixels are clipped (at most {limit:.1%} expected)",
                        basis="render",
                        evidence=evidence,
                        magnitude=value - limit,
                        unit="fraction",
                        hint="lower the exposure",
                        **base,
                    ),
                )
        if "max_black_fraction" in check.params:
            limit = check.params["max_black_fraction"]
            if st["black_fraction"] > limit:
                self.add(
                    result,
                    self.finding(
                        check,
                        "BLACK_AREAS",
                        f"{st['black_fraction']:.1%} of the pixels are black "
                        f"(at most {limit:.1%} expected)",
                        basis="render",
                        evidence=evidence,
                        magnitude=st["black_fraction"] - limit,
                        unit="fraction",
                        hint="add or strengthen lights, or raise the exposure",
                        **base,
                    ),
                )

    def _check_camera_clipping(self, check: Check, result: CheckResult) -> None:
        obs = self.observation(check.view)
        if "objects" in check.params:
            objs = self._geometry_objects(check, result)
        else:
            visible = np.flatnonzero(obs.stats.pixels > 0)[:MAX_CLIPPING_OBJECTS]
            objs = [self.scene._objects[obs.instances[int(k)]["id"]] for k in visible]
        base = {"view": check.view, "observation": obs.id}
        for obj in objs:
            geometry = obs.geometry(obj)
            if not geometry or "error" in geometry:
                continue
            for plane, code in (("near", "NEAR_CLIPPED"), ("far", "FAR_CLIPPED")):
                if not geometry.get(f"{plane}_clipped"):
                    continue
                d0, d1 = geometry["depth_extent"]
                cam = obs.camera
                magnitude = (cam["near"] - d0) if plane == "near" else (d1 - cam["far"])
                self.add(
                    result,
                    self.finding(
                        check,
                        code,
                        f"the camera's {plane} plane cuts {obj.name!r}",
                        [_ref(obj, "subject")],
                        evidence={
                            "depth_extent": geometry["depth_extent"],
                            "near": cam["near"],
                            "far": cam["far"],
                        },
                        magnitude=max(magnitude, 0.0),
                        unit="m",
                        region=_bounds(obs, obs.instance_of(obj)),
                        hint=f"move the camera's {plane} plane",
                        **base,
                    ),
                )

    # -- default checks ----------------------------------------------------------

    def defaults(self, observed: dict[str, Observation]) -> None:
        """Checks Aether3D runs without being asked (``"defaults": true``, the default):
        renders that show nothing, render warnings, exposure far off (heuristic), and
        penetrations among the visible objects (informational)."""
        explicit_exposure = {c.view for c in self.expectations.checks if c.kind == "exposure"}
        for name, obs in observed.items():
            check = Check(f"defaults/{name}", "render", "error", "default render checks", "")
            result = CheckResult(check.id, "render", "error", "pass", check.label)
            st = obs.image_stats
            base = {"view": name, "observation": obs.id}
            if st["all_black"]:
                self.add(
                    result,
                    self.finding(
                        check,
                        "BLACK_IMAGE",
                        f"the image of view {name!r} is entirely black",
                        basis="render",
                        evidence={"stats": st},
                        hint="check the lights (default_lights, intensities), the exposure and "
                        "the camera",
                        **base,
                    ),
                )
            elif st["coverage"] == 0.0:
                self.add(
                    result,
                    self.finding(
                        check,
                        "EMPTY_VIEW",
                        f"view {name!r} shows no object at all",
                        basis="render",
                        evidence={"stats": st},
                        hint="frame the camera on the scene (frame_camera)",
                        **base,
                    ),
                )
            for d in obs.diagnostics:
                if d.severity in ("warning", "error"):
                    self.add(
                        result,
                        self.finding(
                            check,
                            "RENDER_WARNING",
                            f"[{d.code}] {d.message}",
                            severity="warning",
                            basis="render",
                            evidence={"diagnostic": d.code},
                            **base,
                        ),
                    )
            covered = st["luminance_covered"]
            if name not in explicit_exposure and covered is not None and not st["all_black"]:
                if covered["mean"] < DEFAULT_DARK and obs.luminance_source == "linear":
                    self.add(
                        result,
                        self.finding(
                            check,
                            "UNDEREXPOSED",
                            f"the objects' mean luminance is {covered['mean']:.4g} (default "
                            f"rule: below {DEFAULT_DARK:g} is underexposed)",
                            severity="warning",
                            basis="heuristic",
                            evidence={"stats": st, "rule": DEFAULT_DARK},
                            magnitude=math.log2(DEFAULT_DARK / max(covered["mean"], 1e-12)),
                            unit="stops",
                            hint="add an exposure expectation to make this exact",
                            **base,
                        ),
                    )
                if st["clipped_covered_fraction"] > DEFAULT_CLIPPED:
                    self.add(
                        result,
                        self.finding(
                            check,
                            "OVEREXPOSED",
                            f"{st['clipped_covered_fraction']:.0%} of the objects' pixels are "
                            f"clipped (default rule: over {DEFAULT_CLIPPED:.0%} is overexposed)",
                            severity="warning",
                            basis="heuristic",
                            evidence={"stats": st, "rule": DEFAULT_CLIPPED},
                            magnitude=st["clipped_covered_fraction"] - DEFAULT_CLIPPED,
                            unit="fraction",
                            hint="lower the exposure or the lights",
                            **base,
                        ),
                    )
            if result.findings:
                result.status = "fail"
            self.results.append(result)
        # Penetrations among the objects one view shows (informational: intent unknown).
        default = self.expectations.default_view
        obs = observed.get(default) if default is not None else None
        if obs is None and observed:
            obs = next(iter(observed.values()))
        if obs is None:
            return
        visible = np.flatnonzero(obs.stats.pixels > 0)
        check = Check("defaults/penetrations", "no_overlap", "warning", "default scan", "")
        result = CheckResult(check.id, "no_overlap", "info", "pass", "penetrations in view")
        if visible.size > MAX_DEFAULT_OBJECTS:
            result.status = "unknown"
            result.note = (
                f"{visible.size} visible objects: the default scan stops at "
                f"{MAX_DEFAULT_OBJECTS}; state no_overlap checks instead"
            )
            self.results.append(result)
            return
        objs = [self.scene._objects[obs.instances[int(k)]["id"]] for k in visible]
        found = self.penetrations(objs, None, DEFAULT_TOLERANCE, True)
        for pair in found["pairs"]:
            a, b = pair["a"], pair["b"]
            depth = None if pair["kind"] == "contained" else pair["separation"][0]["distance"]
            self.add(
                result,
                self.finding(
                    check,
                    "POSSIBLE_PENETRATION" if depth is not None else "POSSIBLE_CONTAINMENT",
                    f"{a.name!r} and {b.name!r} intersect"
                    + (f" ({depth:.4g} m deep along an axis)" if depth is not None else ""),
                    [_ref(a, "subject"), _ref(b, "other")],
                    severity="info",
                    evidence={
                        "separations": pair.get("separation"),
                        "contained": pair.get("contained"),
                    },
                    magnitude=depth,
                    unit=None if depth is None else "m",
                    hint="if unintended, state a no_overlap check and correct it",
                ),
            )
        result.measured = {
            "intersecting_pairs": found["intersecting"],
            "contacts": found["contacts"],
        }
        result.status = "pass"  # informational
        self.results.append(result)


def _bounds(obs: Observation, instance: int | None) -> tuple[int, int, int, int] | None:
    if instance is None or not obs.stats.pixels[instance]:
        return None
    s = obs.stats
    return (int(s.x0[instance]), int(s.y0[instance]), int(s.x1[instance]), int(s.y1[instance]))


def annotate_regions(findings: list[Finding], obs: Observation | None, view: str | None):
    """Findings with subjects visible in ``obs`` but no region get their subject's pixel
    bounds there (so an image-aware agent sees where each finding is)."""
    if obs is None:
        return findings
    out = []
    for f in findings:
        if f.region is None and f.subjects and f.subjects[0].get("id"):
            instance = obs._by_id.get(f.subjects[0]["id"])
            region = _bounds(obs, instance)
            if region is not None:
                from dataclasses import replace

                f = replace(f, region=region, observation=obs.id, view=f.view or view)
        out.append(f)
    return out


def number_findings(findings: list[Finding]) -> list[Finding]:
    from dataclasses import replace

    return [replace(f, id=f"F{k + 1}") for k, f in enumerate(findings)]


def tally(findings: list[Finding]) -> tuple[int, int, float]:
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")
    score = sum(SEVERITY_WEIGHT.get(f.severity, 0.0) for f in findings)
    return errors, warnings, score
