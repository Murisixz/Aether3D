"""Vision evaluations: an image-aware model's judgment of an observation, as checked data.

Geometry and pixel statistics prove placement facts; they cannot tell whether a render
*looks* like what was asked for, whether an object is recognizable, or whether a scene
is pleasing. That needs a model (or a person) that actually looks at the image. This
module is the provider-neutral boundary for such judgments:

* **The document** (``aether3d.vision-evaluation``, version 1) carries findings with a
  category, severity, confidence, a region of the observation's image, the objects meant,
  an optional suggested correction (an action from :data:`aether3d.feedback.corrections.ACTIONS`),
  an overall verdict, and **provenance**: provider, model and whether the model received
  the image -- as *declared* by whoever supplies the document.
* **Checking** (:func:`check_evaluation`) refuses malformed documents with every problem
  listed, ties the evaluation to one observation (its id and pixel digest), and marks it
  stale when the scene changed since.
* **Grounding** maps each finding's region onto the observation's object-index buffer:
  which objects really occupy it, and whether the objects the finding names are visible
  there. A finding naming an object that is not in its region is *ungrounded*: its target
  is never used for a correction.
* **Epistemic labels.** Findings from evaluations whose provenance declares an image input
  get ``basis: "vision"``; others ``basis: "agent"``. Aether3D records what it can verify --
  whether it delivered that observation's image (to an MCP client or a critic) -- and never
  states that a model saw an image. Vision findings never count as facts, never apply a
  correction by themselves, and count toward acceptance only when the expectations ask for
  it (``acceptance.require_vision``).

The engine contains no model and makes no network request. A :class:`VisionCritic` is any
callable that takes a :class:`CritiqueRequest` (the PNG bytes and the observation's
summary) and returns an evaluation document; over MCP, the client's own model is the
critic (the agent looks at ``observe_scene``'s image and submits its evaluation).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..errors import FeedbackError
from ._doc import BAD, Doc, join
from .checks import Finding

if TYPE_CHECKING:
    from .observe import Observation

FORMAT = "aether3d.vision-evaluation"
VERSION = 1
CATEGORIES = (
    "placement",
    "intersection",
    "floating",
    "scale",
    "orientation",
    "composition",
    "framing",
    "lighting",
    "exposure",
    "material",
    "color",
    "missing",
    "extra",
    "style",
    "other",
)
SEVERITIES = ("error", "warning", "info")
MAX_FINDINGS = 50
MAX_OBJECTS = 10
MAX_DOCUMENT_BYTES = 1 * 2**20


@dataclass(frozen=True)
class VisionEvaluation:
    """A checked and grounded vision evaluation."""

    digest: str
    observation: str
    provenance: dict[str, Any]
    request: str | None
    overall: dict[str, Any]
    findings: tuple[Finding, ...]
    stale: bool
    delivered: bool
    warnings: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def basis(self) -> str:
        return "vision" if self.provenance.get("image_input") is True else "agent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "digest": self.digest,
            "observation": self.observation,
            "basis": self.basis,
            "provenance": {
                **self.provenance,
                "image_input": "declared" if self.provenance.get("image_input") else False,
                "image_delivered_by_aether3d": self.delivered,
            },
            "request": self.request,
            "overall": self.overall,
            "stale": self.stale,
            "findings": [f.to_dict() for f in self.findings],
            "warnings": list(self.warnings),
        }


def check_evaluation(document: Any, observation: Observation) -> VisionEvaluation:
    """Check an evaluation document against ``observation`` and ground its findings.

    Raises ``FeedbackError`` (``INVALID_DOCUMENT`` family codes with every problem in
    ``issues``; ``OBSERVATION_MISMATCH`` when it names another observation or image).
    """
    if isinstance(document, (str, bytes)):
        if len(document) > MAX_DOCUMENT_BYTES:
            raise FeedbackError("LIMIT_EXCEEDED", "the evaluation is larger than 1 MiB", "")
        try:
            document = json.loads(document)
        except ValueError as exc:
            raise FeedbackError("INVALID_JSON", f"not valid JSON: {exc}", "") from None
    doc = Doc()
    obj = doc.object(document, "", "a vision evaluation (a JSON object)")
    if obj is None:
        doc.issues.raise_if_any("invalid vision evaluation")
    assert obj is not None
    if obj.get("format") != FORMAT:
        doc.issues.add("INVALID_FORMAT", f'"format" must be {FORMAT!r}', "/format")
    version = obj.get("version")
    if isinstance(version, bool) or version != VERSION:
        doc.issues.add("UNSUPPORTED_VERSION", f'"version" must be {VERSION}', "/version")
    doc.keys(
        obj,
        "",
        (
            "format",
            "version",
            "observation",
            "image_sha256",
            "provenance",
            "request",
            "overall",
            "findings",
        ),
        ("observation", "provenance"),
    )
    oid = doc.string(obj.get("observation"), "/observation", max_length=64)
    if "image_sha256" in obj:
        doc.string(obj["image_sha256"], "/image_sha256", max_length=64)
    provenance = _provenance(doc, obj.get("provenance"), "/provenance")
    request = None
    if "request" in obj:
        request = doc.text(obj["request"], "/request")
        request = None if request is BAD else request
    overall = _overall(doc, obj.get("overall", {}), "/overall")
    raw = doc.array(obj.get("findings", []), "/findings", max_items=MAX_FINDINGS)
    parsed = []
    seen: set[str] = set()
    for k, item in enumerate(raw or []):
        f = _finding(doc, item, join("/findings", k), k, observation)
        if f is None:
            continue
        if f["id"] in seen:
            doc.issues.add("DUPLICATE_ID", f"finding id {f['id']!r} is used twice", f["ptr"])
            continue
        seen.add(f["id"])
        parsed.append(f)
    doc.issues.raise_if_any("invalid vision evaluation")
    if oid != observation.id:
        raise FeedbackError(
            "OBSERVATION_MISMATCH",
            f"the evaluation is of observation {oid!r}, not {observation.id!r}",
            "/observation",
        )
    sha = obj.get("image_sha256")
    if sha is not None and sha != observation.sha256:
        raise FeedbackError(
            "OBSERVATION_MISMATCH",
            "image_sha256 does not match the observation's pixels",
            "/image_sha256",
            hint="evaluate the image of this observation (its sha256 is in the observation)",
        )
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    basis = "vision" if provenance.get("image_input") is True else "agent"
    findings = []
    warnings: list[str] = []
    if basis == "agent":
        warnings.append(
            "the provenance does not declare an image input: findings are agent judgments, "
            "not visual observations"
        )
    for f in parsed:
        findings.append(_ground(f, observation, basis, digest, provenance))
    ungrounded = [
        f.evidence["vision_id"] for f in findings if f.evidence["grounding"] == "ungrounded"
    ]
    if ungrounded:
        warnings.append(
            f"finding(s) {', '.join(ungrounded)} name objects that are not visible in their "
            "region: their targets are not used"
        )
    return VisionEvaluation(
        digest=digest,
        observation=observation.id,
        provenance=provenance,
        request=request,
        overall=overall,
        findings=tuple(findings),
        stale=observation.stale,
        delivered=bool(observation.options.get("delivered")),
        warnings=tuple(warnings),
        source=json.loads(text),
    )


def _provenance(doc: Doc, value: Any, ptr: str) -> dict[str, Any]:
    obj = doc.object(value, ptr, '{"provider", "model", "image_input", "method"}')
    if obj is None:
        return {}
    doc.keys(obj, ptr, ("provider", "model", "image_input", "method"), ("provider",))
    out: dict[str, Any] = {}
    for key, size in (("provider", 64), ("model", 128)):
        if key in obj:
            v = doc.string(obj[key], join(ptr, key), max_length=size)
            if v is not BAD:
                out[key] = v
    if "image_input" in obj:
        v = doc.boolean(obj["image_input"], join(ptr, "image_input"))
        if v is not BAD:
            out["image_input"] = v
    if "method" in obj:
        v = doc.string(obj["method"], join(ptr, "method"), max_length=500, empty=True)
        if v is not BAD:
            out["method"] = v
    return out


def _overall(doc: Doc, value: Any, ptr: str) -> dict[str, Any]:
    obj = doc.object(value, ptr, '{"matches_request", "score", "summary"}')
    if obj is None:
        return {}
    doc.keys(obj, ptr, ("matches_request", "score", "summary", "confidence"))
    out: dict[str, Any] = {"matches_request": None, "score": None}
    if obj.get("matches_request") is not None:
        v = doc.boolean(obj["matches_request"], join(ptr, "matches_request"))
        out["matches_request"] = None if v is BAD else v
    for key in ("score", "confidence"):
        if obj.get(key) is not None:
            v = doc.number(obj[key], join(ptr, key), lo=0.0, hi=1.0)
            out[key] = None if v is BAD else v
    if "summary" in obj:
        v = doc.text(obj["summary"], join(ptr, "summary"))
        if v is not BAD:
            out["summary"] = v
    return out


def _finding(doc: Doc, value: Any, ptr: str, k: int, obs: Observation) -> dict | None:
    from .corrections import check_action

    obj = doc.object(value, ptr, "a finding")
    if obj is None:
        return None
    before = doc.issues.total
    doc.keys(
        obj,
        ptr,
        (
            "id",
            "category",
            "severity",
            "confidence",
            "summary",
            "region",
            "region_normalized",
            "objects",
            "suggestion",
        ),
        ("category", "summary", "confidence"),
    )
    out: dict[str, Any] = {"ptr": ptr, "id": f"v{k + 1}"}
    if "id" in obj:
        v = doc.string(obj["id"], join(ptr, "id"), max_length=64)
        if v is not BAD:
            out["id"] = v
    out["category"] = doc.enum(obj.get("category"), join(ptr, "category"), CATEGORIES)
    out["severity"] = obj.get("severity", "warning")
    if doc.enum(out["severity"], join(ptr, "severity"), SEVERITIES) is BAD:
        out["severity"] = "warning"
    out["confidence"] = doc.number(obj.get("confidence"), join(ptr, "confidence"), lo=0.0, hi=1.0)
    out["summary"] = doc.string(obj.get("summary"), join(ptr, "summary"), max_length=500)
    if "region" in obj and "region_normalized" in obj:
        doc.issues.add("INVALID_VALUE", "give region or region_normalized, not both", ptr)
    if "region" in obj:
        r = doc.rect(obj["region"], join(ptr, "region"), normalized=False)
        if r is not BAD:
            x0, y0, x1, y1 = r
            if x1 <= 0 or y1 <= 0 or x0 >= obs.width or y0 >= obs.height:
                doc.issues.add(
                    "OUT_OF_RANGE",
                    f"the region lies outside the {obs.width}x{obs.height} image",
                    join(ptr, "region"),
                )
            else:
                out["region"] = r
    if "region_normalized" in obj:
        r = doc.rect(obj["region_normalized"], join(ptr, "region_normalized"), normalized=True)
        if r is not BAD:
            out["region"] = (
                r[0] * obs.width,
                r[1] * obs.height,
                r[2] * obs.width,
                r[3] * obs.height,
            )
    if "objects" in obj:
        items = doc.array(obj["objects"], join(ptr, "objects"), max_items=MAX_OBJECTS)
        names = []
        for i, item in enumerate(items or []):
            v = doc.string(item, join(join(ptr, "objects"), i))
            if v is not BAD:
                names.append(v)
        out["objects"] = names
    if "suggestion" in obj:
        out["suggestion"] = check_action(doc, obj["suggestion"], join(ptr, "suggestion"))
    if doc.issues.total > before:
        return None
    return out


def _ground(
    f: dict[str, Any], obs: Observation, basis: str, digest: str, provenance: dict[str, Any]
) -> Finding:
    """A vision finding as a :class:`Finding`, grounded in the observation's object index."""
    scene = obs.scene
    claimed = []
    for ref in f.get("objects", []):
        o = scene._objects.get(ref) or scene._by_name.get(ref)
        claimed.append((ref, o))
    region = f.get("region")
    evidence: dict[str, Any] = {
        "vision_id": f["id"],
        "category": f["category"],
        "confidence": f["confidence"],
        "evaluation": digest,
        "provenance": provenance,
    }
    status = "no_region"
    in_region = []
    pixel_rect = None
    if region is not None:
        in_region = obs.objects_in_rect(region)
        evidence["region_objects"] = in_region[:5]
        from .observe import _pixel_rect

        pixel_rect = _pixel_rect(region, obs.width, obs.height)
    present = {r["id"] for r in in_region}
    targets = []
    checks = []
    for ref, o in claimed:
        if o is None:
            checks.append({"object": ref, "exists": False})
            continue
        visible = obs._by_id.get(o.id) is not None and obs.stats.pixels[obs._by_id[o.id]] > 0
        entry = {
            "object": ref,
            "id": o.id,
            "name": o.name,
            "exists": True,
            "visible": bool(visible),
        }
        if region is not None:
            entry["in_region"] = o.id in present
            if o.id in present:
                targets.append(o)
        elif visible:
            targets.append(o)
        checks.append(entry)
    if claimed:
        evidence["claimed_objects"] = checks
        if region is None:
            status = "named" if targets else "ungrounded"
        else:
            status = (
                "grounded"
                if len(targets) == len(claimed)
                else ("partly_grounded" if targets else "ungrounded")
            )
    elif region is not None:
        status = "region_only"
        if in_region and in_region[0]["fraction"] >= 0.5:
            evidence["dominant_object"] = in_region[0]
    evidence["grounding"] = status
    if obs.stale:
        evidence["stale"] = True
    subjects = tuple({"id": o.id, "name": o.name, "role": "named"} for o in targets)
    if f.get("suggestion") is not None:
        evidence["suggestion"] = f["suggestion"]
    severity = f["severity"]
    return Finding(
        key=f"vision|{digest[:8]}|{f['id']}",
        check=f"vision/{digest[:8]}",
        kind="vision",
        code=f"VISION_{f['category'].upper()}",
        severity=severity,
        basis=basis,
        exact=False,
        message=f["summary"],
        subjects=subjects,
        evidence=evidence,
        view=obs.view.name,
        observation=obs.id,
        region=pixel_rect,
        hint=None,
    )


# --------------------------------------------------------------------------- #
# Critics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CritiqueRequest:
    """What a critic is given: the observation's images and summary, and the intent."""

    observation: str
    image_png: bytes
    annotated_png: bytes
    width: int
    height: int
    image_sha256: str
    summary: dict[str, Any]
    markers: list[dict[str, Any]]
    intent: str | None
    instructions: str

    def evaluation_template(self) -> dict[str, Any]:
        """A skeleton of the document the critic returns."""
        return {
            "format": FORMAT,
            "version": VERSION,
            "observation": self.observation,
            "image_sha256": self.image_sha256,
            "provenance": {"provider": "...", "model": "...", "image_input": True},
            "overall": {"matches_request": None, "score": None, "summary": ""},
            "findings": [],
        }


class VisionCritic(Protocol):
    """A provider-neutral image critic: returns a vision-evaluation document (a dict).

    ``cost`` is what one call costs in the caller's units (checked against the loop's
    ``max_cost``); implementations that call paid services should set it.
    """

    name: str
    cost: float

    def __call__(self, request: CritiqueRequest) -> dict[str, Any]: ...


INSTRUCTIONS = """You are reviewing a render made by the Aether3D engine. Look at the image
(and the annotated copy, whose numbered markers are listed with object ids). Report
concrete visual problems as JSON in the format aether3d.vision-evaluation version 1:
each finding with a category, severity, confidence (0-1), a one-sentence summary, the
pixel region [x0, y0, x1, y1] of the image where it is, and the ids of the objects
concerned when you can tell them from the markers. Judge only what you see. If you were
not given the image, set provenance.image_input to false."""


def critique_request(
    obs: Observation, intent: str | None = None, *, instructions: str = INSTRUCTIONS
) -> CritiqueRequest:
    """The request a critic receives for ``obs`` (marks its image as delivered)."""
    request = CritiqueRequest(
        observation=obs.id,
        image_png=obs.png("image"),
        annotated_png=obs.png("annotated"),
        width=obs.width,
        height=obs.height,
        image_sha256=obs.sha256,
        summary=obs.to_dict(max_objects=20),
        markers=obs.markers(),
        intent=intent,
        instructions=instructions,
    )
    obs.options["delivered"] = True
    return request


def run_critic(
    critic: Callable[[CritiqueRequest], Any], obs: Observation, intent: str | None = None
) -> VisionEvaluation:
    """Ask ``critic`` about ``obs`` and check its answer (a malformed answer raises
    ``FeedbackError``; nothing is applied)."""
    request = critique_request(obs, intent)
    answer = critic(request)
    return check_evaluation(answer, obs)
