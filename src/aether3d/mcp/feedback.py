"""The visual-feedback tools of the MCP server (Stage 12).

Five tools, in three kinds:

* **inspect** -- ``observe_scene`` (render views and measure them; the images come back
  as image content, with numbered markers on request) and ``diagnose_scene`` (check
  expectations, ground the agent's own visual evaluations);
* **propose** -- ``propose_corrections`` (typed, explained proposals; ``preview`` applies
  them in a transaction, observes, and rolls back);
* **change** -- ``apply_corrections`` (one undo step, refused for a changed scene, verified
  and undone when it does not help) and ``feedback_loop`` (repeat until the expectations
  hold, a budget runs out, or only approvals remain).

The agent is the image-aware evaluator: it looks at the images ``observe_scene`` returns
and passes its findings to ``diagnose_scene``/``propose_corrections`` as an
``aether3d.vision-evaluation``. The server records whether it delivered an observation's
image, never whether a model looked at it. MCP sampling (deprecated in the 2026-07-28
protocol, SEP-2577) is not used.
"""

from __future__ import annotations

from typing import Any

from ..errors import FeedbackError
from .session import Session, ToolError

MAX_IMAGES = 8
MAX_IMAGE_PIXELS = 6 * 1920 * 1080
IMAGE_KINDS = ("image", "annotated", "ids", "depth")


def feedback(session: Session):
    """The session's :class:`~aether3d.feedback.FeedbackSession` for the open scene."""
    from ..feedback import FeedbackSession, ObservationLimits

    scene = session.require_scene()
    current = session.feedback
    if current is not None and current.scene is scene:
        return current
    config = session.config
    fs = FeedbackSession(
        scene,
        backend=config.feedback_backend,
        adapter=config.gpu_adapter,
        limits=ObservationLimits(
            max_pixels=config.max_preview_pixels,
            max_path_samples=config.max_path_samples if config.allow_path_tracing else 0,
        ),
        base_dir=session.base_dir(),
    )
    session.feedback = fs
    return fs


def _expectations(fs: Any, args: dict[str, Any]) -> None:
    from ..feedback import parse_expectations

    if args.get("clear_expectations"):
        fs.expectations = parse_expectations({"format": "aether3d.expectations", "version": 1})
    if args.get("expectations") is not None:
        fs.expectations = parse_expectations(args["expectations"])


def _view_arg(value: Any) -> Any:
    return value


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def observe_scene(session: Session, args: dict, ctx: Any) -> Any:
    from .tools import ToolOutput

    fs = feedback(session)
    _expectations(fs, args)
    config = session.config
    views = args.get("views") or [args.get("view")]
    backend = args.get("backend", config.feedback_backend)
    path = None
    if backend in ("path", "gpu_path"):
        if not config.allow_path_tracing:
            raise ToolError("NOT_ALLOWED", "this server does not path trace (--no-path-tracing)")
        path = {"samples": args.get("spp", 16), "bounces": args.get("bounces", 4)}
    kinds = args.get("images", ["image"])
    width, height = args.get("width", 640), args.get("height", 360)
    images: list[bytes] = []
    described: list[dict[str, Any]] = []
    observations = []
    pixels = 0

    def add(png: bytes, w: int, h: int, info: dict[str, Any]) -> None:
        nonlocal pixels
        if len(images) >= MAX_IMAGES or pixels + w * h > MAX_IMAGE_PIXELS:
            raise ToolError(
                "LIMIT_EXCEEDED",
                f"a reply carries at most {MAX_IMAGES} images ({MAX_IMAGE_PIXELS} pixels)",
                hint="ask for fewer views, image kinds or crops",
            )
        pixels += w * h
        info["index"] = len(images)
        images.append(png)
        described.append(info)

    for view in views:
        if ctx.cancel.is_set():
            raise ToolError("CANCELLED", "cancelled")
        obs = fs.observe(
            view,
            focus=args.get("focus"),
            occlusion=args.get("occlusion", False),
            linear=args.get("linear", False),
            width=width,
            height=height,
            backend=backend,
            path=path,
        )
        observations.append(obs.to_dict(max_objects=args.get("max_objects", 30)))
        for kind in kinds:
            add(obs.png(kind), obs.width, obs.height, {"observation": obs.id, "kind": kind})
            if kind in ("image", "annotated"):
                obs.options["delivered"] = True
        for crop in args.get("crops", []):
            if crop.get("observation") not in (None, obs.id):
                continue
            x0, y0, x1, y1 = crop["rect"]
            rect = (x0 * obs.width, y0 * obs.height, x1 * obs.width, y1 * obs.height)
            png, used = obs.crop_png(rect, scale=crop.get("scale", 2))
            scale = crop.get("scale", 2)
            add(
                png,
                (used[2] - used[0]) * scale,
                (used[3] - used[1]) * scale,
                {"observation": obs.id, "kind": "crop", "rect": used, "scale": scale},
            )
            obs.options["delivered"] = True
    fs.store.recount()
    session.last_preview = images[0] if images else session.last_preview
    data = {
        "observations": observations,
        "images": described,
        "revision": fs.scene.revision,
        "note": "images follow as image content, in the order of 'images'; pixel regions "
        "and markers refer to each observation's own image",
    }
    return ToolOutput(data, images=images)


def diagnose_scene(session: Session, args: dict, ctx: Any) -> Any:
    from .tools import ToolOutput

    fs = feedback(session)
    _expectations(fs, args)
    diagnosis = fs.diagnose(
        evaluations=args.get("evaluations") or (), defaults=args.get("defaults")
    )
    data = diagnosis.to_dict(max_findings=args.get("max_findings", 50))
    images = []
    if args.get("include_image"):
        default = diagnosis.expectations.default_view or "camera"
        obs = diagnosis.observed.get(default) or next(iter(diagnosis.observed.values()), None)
        if obs is not None:
            images.append(obs.png("annotated"))
            obs.options["delivered"] = True
            data["image"] = {"observation": obs.id, "kind": "annotated", "markers": obs.markers()}
    return ToolOutput(data, images=images)


def propose_corrections(session: Session, args: dict, ctx: Any) -> Any:
    from .tools import ToolOutput

    fs = feedback(session)
    _expectations(fs, args)
    diagnosis = fs.diagnose(evaluations=args.get("evaluations") or ())
    corrections = fs.propose(diagnosis, policy=args.get("policy"))
    data = corrections.to_dict(compiled=args.get("include_compiled", True))
    data["diagnosis_summary"] = diagnosis.to_dict(max_findings=20)["summary"]
    data["findings"] = [f.to_dict() for f in diagnosis.findings[:30]]
    images = []
    if args.get("preview"):
        preview = fs.preview(corrections, args.get("preview_ids"))
        obs = preview.pop("observation", None)
        data["preview"] = preview
        if obs is not None and args.get("include_image", True):
            images.append(obs.png("annotated"))
            data["preview"]["image"] = {"kind": "annotated", "markers": obs.markers()}
    return ToolOutput(data, images=images)


def apply_corrections(session: Session, args: dict, ctx: Any) -> Any:
    from .tools import ToolOutput, scene_status

    fs = feedback(session)
    result = fs.apply(
        None if args.get("actions") else fs.last_corrections,
        args.get("proposals"),
        actions=args.get("actions"),
        approve=args.get("approve") or (),
        policy=args.get("policy"),
        verify=args.get("verify"),
        expected_revision=args.get("expected_revision"),
        label=args.get("label"),
    )
    data = result.to_dict()
    data["scene"] = scene_status(session)
    images = []
    if args.get("include_image") and result.status == "applied":
        obs = fs.observe(None)
        images.append(obs.png("annotated"))
        obs.options["delivered"] = True
        data["image"] = {"observation": obs.id, "kind": "annotated", "markers": obs.markers()}
    return ToolOutput(data, images=images)


def feedback_loop(session: Session, args: dict, ctx: Any) -> Any:
    from ..feedback import FeedbackBudget, run_loop
    from .tools import ToolOutput, scene_status

    fs = feedback(session)
    _expectations(fs, args)
    budget = FeedbackBudget.coerce(args.get("budget"))
    remaining = ctx.remaining()
    if remaining is not None and remaining < budget.max_seconds:
        from dataclasses import replace

        budget = replace(budget, max_seconds=max(0.5, remaining - 1.0))

    def progress(done: int, total: int, message: str) -> None:
        if ctx.progress is not None:
            ctx.progress(float(done), float(total), message)

    report = run_loop(
        fs,
        budget=budget,
        policy=args.get("policy"),
        cancel=ctx.cancel,
        progress=progress,
        resume=args.get("resume"),
    )
    data = report.to_dict()
    data["scene"] = scene_status(session)
    images = []
    if args.get("include_image", True):
        try:
            obs = fs.observe(None)
        except FeedbackError as exc:
            data["image_error"] = exc.to_dict()
        else:
            images.append(obs.png("annotated"))
            obs.options["delivered"] = True
            data["image"] = {"observation": obs.id, "kind": "annotated", "markers": obs.markers()}
    return ToolOutput(data, images=images)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

_VIEW = {
    "description": "a view name of the expectations, or {camera: id|name} (a scene camera; "
    "{} the active one), {virtual: {position, look_at|direction, up, fov, ...}}, "
    "{focus: {objects: selector, direction, margin}}, {preset: top|bottom|front|back|left|"
    "right|iso, objects}",
    "anyOf": [{"type": "string", "minLength": 1}, {"type": "object"}],
}
_EXPECTATIONS = {
    "type": "object",
    "description": "an aether3d.expectations document (schema: aether3d://schema/expectations; "
    "guide: aether3d://feedback/guide). Kept for later calls; omit to reuse it",
}
_EVALUATIONS = {
    "type": "array",
    "maxItems": 4,
    "description": "your own visual evaluations of observations, as aether3d.vision-evaluation "
    "documents (schema: aether3d://schema/vision-evaluation): set provenance.image_input only "
    "if you looked at the image",
    "items": {"type": "object"},
}
_POLICY = {
    "type": "object",
    "description": "auto_apply (classes: safe, verify), allow_placement|camera|lighting|"
    "materials|visibility|procedural, max_actions, max_objects, max_translation (m), "
    "max_rotation, max_scale, max_exposure_change, verify, rollback_on_regression, "
    "rollback_if_ineffective, min_vision_confidence",
}
_ACTION = {
    "type": "object",
    "description": "a correction action: {action: translate|place|rotate|scale|set_visibility|"
    "assign_material|set_color|frame_camera|aim_camera|dolly_camera|set_camera|set_light|"
    "add_light|set_exposure|set_ambient|set_parameter, ...} (schema: "
    "aether3d://schema/corrections)",
    "properties": {"action": {"type": "string"}},
    "required": ["action"],
}
_IDS = {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 100}


def specs() -> tuple[Any, ...]:
    from .tools import SELECTOR, ToolSpec, _schema

    return (
        ToolSpec(
            "observe_scene",
            "Observe the scene (render and measure)",
            "Render one or more views and return the images (image content) with what they "
            "show, measured: per object visible pixels, pixel bounds, center, depth range, "
            "whether it touches the border, and for focus objects the exact geometry facts "
            "(in the view volume, projected bounds, near/far clipping; occlusion: its "
            "unoccluded footprint and what covers it); for the image coverage, luminance, "
            "black and clipped fractions. images: image (clean), annotated (numbered markers, "
            "listed in markers with object ids), ids (object colors), depth. crops: "
            "normalized rects, enlarged. Views can be virtual cameras or framed on objects: "
            "nothing is added to the scene. Never changes the scene.",
            _schema(
                {
                    "view": _VIEW,
                    "views": {"type": "array", "items": _VIEW, "minItems": 1, "maxItems": 4},
                    "width": {"type": "integer", "minimum": 16, "maximum": 1920, "default": 640},
                    "height": {"type": "integer", "minimum": 16, "maximum": 1920, "default": 360},
                    "backend": {
                        "enum": ["auto", "gpu", "cpu", "path", "gpu_path"],
                        "description": "default: the server's --feedback-backend (auto)",
                    },
                    "spp": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "bounces": {"type": "integer", "minimum": 0, "maximum": 16},
                    "focus": SELECTOR,
                    "occlusion": {"type": "boolean", "default": False},
                    "linear": {"type": "boolean", "default": False},
                    "images": {
                        "type": "array",
                        "items": {"enum": list(IMAGE_KINDS)},
                        "uniqueItems": True,
                        "maxItems": 4,
                    },
                    "crops": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "rect": {
                                    "type": "array",
                                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                                    "minItems": 4,
                                    "maxItems": 4,
                                },
                                "scale": {"type": "integer", "minimum": 1, "maximum": 4},
                                "observation": {"type": "string"},
                            },
                            "required": ["rect"],
                            "additionalProperties": False,
                        },
                    },
                    "max_objects": {"type": "integer", "minimum": 1, "maximum": 200},
                    "expectations": _EXPECTATIONS,
                }
            ),
            observe_scene,
            read_only=True,
            idempotent=True,
        ),
        ToolSpec(
            "diagnose_scene",
            "Diagnose the scene against expectations",
            "Check the scene against an expectations document (what must exist, rest on what, "
            "not intersect, be visible and how framed, how bright, which material, upright, in "
            "a region, at a distance, in a relation) and default render checks. Findings have "
            "stable codes, the objects concerned, measured evidence and magnitude, the image "
            "region, and their basis: geometry/render (measured), heuristic (default "
            "threshold), vision/agent (your evaluations, grounded in the object index; never "
            "facts). Never changes the scene.",
            _schema(
                {
                    "expectations": _EXPECTATIONS,
                    "clear_expectations": {"type": "boolean", "default": False},
                    "evaluations": _EVALUATIONS,
                    "defaults": {"type": "boolean"},
                    "include_image": {"type": "boolean", "default": False},
                    "max_findings": {"type": "integer", "minimum": 1, "maximum": 500},
                }
            ),
            diagnose_scene,
            read_only=True,
            idempotent=True,
        ),
        ToolSpec(
            "propose_corrections",
            "Propose corrections",
            "Diagnose, then propose typed corrections for the findings: each with its target, "
            "actions (allowlisted: move, place, rotate, scale, show/hide, material, color, frame/"
            "aim/dolly camera, lens, light, exposure, ambient, procedural parameter), the "
            "evidence, expected effect, what verification checks, alternatives, the exact "
            "scene plan/script it compiles to, and a safety class: safe (exact repair of a "
            "stated expectation), verify (applied only if it measurably helps), approval "
            "(subjective or ambiguous). preview applies them in a transaction, re-renders "
            "and rolls back. Never changes the scene.",
            _schema(
                {
                    "expectations": _EXPECTATIONS,
                    "evaluations": _EVALUATIONS,
                    "policy": _POLICY,
                    "preview": {"type": "boolean", "default": False},
                    "preview_ids": _IDS,
                    "include_image": {"type": "boolean", "default": True},
                    "include_compiled": {"type": "boolean", "default": True},
                }
            ),
            propose_corrections,
            read_only=True,
        ),
        ToolSpec(
            "apply_corrections",
            "Apply corrections",
            "Apply proposals from the last propose_corrections (proposals: ids; default: those "
            "the policy applies automatically; approval ones need approve), or your own typed "
            "actions, as ONE undo step, all or nothing. Refused if the scene changed since "
            "(expected_revision). Then verifies: observes and diagnoses again, and undoes the "
            "step when a passing check now fails or the findings it addressed did not improve. "
            "Reports changes, resolved, persisting and new findings.",
            _schema(
                {
                    "proposals": _IDS,
                    "approve": _IDS,
                    "actions": {"type": "array", "items": _ACTION, "minItems": 1, "maxItems": 100},
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "policy": _POLICY,
                    "verify": {"type": "boolean"},
                    "label": {"type": "string", "minLength": 1, "maxLength": 200},
                    "include_image": {"type": "boolean", "default": False},
                }
            ),
            apply_corrections,
        ),
        ToolSpec(
            "feedback_loop",
            "Run the feedback loop",
            "Repeat observe -> diagnose -> propose -> apply -> verify until the expectations "
            "hold (accepted), only corrections needing approval remain (awaiting_approval: "
            "approve them with apply_corrections, then resume), no safe progress is possible, "
            "it stagnates or cycles, or a budget runs out (iterations, renders, pixels, "
            "seconds, memory). One undo step per applied correction; a checkpoint "
            "feedback:<id>:before returns to the start (history goto). Reports every iteration.",
            _schema(
                {
                    "expectations": _EXPECTATIONS,
                    "budget": {
                        "type": "object",
                        "description": "max_iterations, max_renders, max_pixels, max_seconds, "
                        "max_memory, patience",
                    },
                    "policy": _POLICY,
                    "resume": {"type": "string", "minLength": 1},
                    "include_image": {"type": "boolean", "default": True},
                }
            ),
            feedback_loop,
        ),
    )
