"""``aether3d feedback``: observe, check, correct and loop from the command line (Stage 12).

* ``feedback observe PROJECT`` -- render views and measure them; ``--out DIR`` writes the
  observation (JSON) and its images (PNG).
* ``feedback check PROJECT --expect FILE`` -- diagnose against expectations (and vision
  evaluations of observations made in the same run); exit status 3 when not accepted.
* ``feedback correct PROJECT --expect FILE`` -- propose corrections; ``--apply`` applies
  the automatic ones (and ``--approve`` ids), verified, as one undo step, and saves.
* ``feedback loop PROJECT --expect FILE`` -- the automatic loop, then saves.
* ``feedback schema KIND`` -- the JSON Schema of expectations, vision evaluations or
  correction actions.

Projects with a history file record every applied correction (``aether3d history undo``
takes one back).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cli import (
    EXIT_GEOMETRY_ERRORS,
    EXIT_OK,
    CommandError,
    CommandOutput,
    _load_scene,
    _project_only,
    _read_script,
    _write_or_dry_run,
)
from .errors import FeedbackError

BACKENDS = ("cpu", "gpu", "auto", "path", "gpu_path")


def _json_arg(value: str | None, what: str) -> Any:
    """A JSON value given inline (``{...}``) or as a file path."""
    if value is None:
        return None
    text = value.strip()
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise CommandError("FeedbackError", f"{what} is not valid JSON: {exc}", what) from None
    return _read_script(value)


def _session(args: argparse.Namespace, scene: Any, path: Path):
    from .feedback import FeedbackSession

    expectations = _json_arg(getattr(args, "expect", None), "expect")
    try:
        return FeedbackSession(
            scene,
            expectations,
            backend=args.backend,
            adapter=getattr(args, "adapter", None),
            width=args.width,
            height=args.height,
            policy=_json_arg(getattr(args, "policy", None), "policy"),
            base_dir=path.absolute().parent,
        )
    except FeedbackError as exc:
        raise _error(exc) from None


def _error(exc: FeedbackError) -> CommandError:
    details: dict[str, Any] = {"code": exc.code}
    if exc.hint:
        details["hint"] = exc.hint
    if exc.issues:
        details["issues"] = exc.issues
    if exc.details:
        details["details"] = exc.details
    return CommandError("FeedbackError", exc.message, exc.location, details)


def _write_images(out: Path, obs: Any, kinds: list[str]) -> dict[str, str]:
    from ._fileio import write_atomic

    out.mkdir(parents=True, exist_ok=True)
    written = {}
    report = out / f"{obs.id}.json"
    write_atomic(report, json.dumps(obs.to_dict(max_objects=200), indent=2).encode("utf-8"))
    written["json"] = str(report)
    for kind in kinds:
        target = out / f"{obs.id}_{kind}.png"
        write_atomic(target, obs.png(kind, compression=6))
        written[kind] = str(target)
    return written


def cmd_feedback_observe(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, args.allow_lossy)
    session = _session(args, scene, path)
    kinds = [k for k in args.images.split(",") if k]
    for k in kinds:
        if k not in ("image", "annotated", "ids", "depth"):
            raise CommandError("FeedbackError", f"unknown image kind {k!r}", "images")
    view = (
        _json_arg(args.view, "view") if args.view and args.view.lstrip()[:1] in "{[" else args.view
    )
    try:
        obs = session.observe(
            view,
            focus=_json_arg(args.focus, "focus") if args.focus else None,
            occlusion=args.occlusion,
            linear=args.linear,
            path={"samples": args.spp} if args.backend in ("path", "gpu_path") else None,
        )
    except FeedbackError as exc:
        raise _error(exc) from None
    data = obs.to_dict(max_objects=args.max_objects)
    if args.out:
        data["files"] = _write_images(Path(args.out), obs, kinds)
    st = obs.image_stats
    lines = [
        f"observation {obs.id}: view {obs.view.name!r}, {obs.width}x{obs.height}, "
        f"{obs.renderer['backend']} ({obs.renderer['device']})",
        f"  coverage {st['coverage']:.1%}, mean luminance "
        f"{(st['luminance'] or {}).get('mean', 0):.4g} ({st['luminance_source']}), clipped "
        f"{st['clipped_fraction']:.1%}, black {st['black_fraction']:.1%}",
    ]
    for o in data["objects"][:15]:
        bounds = o.get("screen_bounds")
        lines.append(
            f"  {o['name']}: {o['visible_pixels']} px"
            + (f", bounds {bounds}" if bounds else f" ({o.get('reason', 'not visible')})")
        )
    if args.out:
        lines.append(f"wrote {len(data['files'])} file(s) to {args.out}")
    return data, "\n".join(lines), EXIT_OK


def _finding_lines(diagnosis: Any) -> list[str]:
    lines = []
    for f in diagnosis.findings:
        if f.severity == "info" and len(lines) > 30:
            continue
        mag = "" if f.magnitude is None else f" [{f.magnitude:.4g} {f.unit}]"
        lines.append(f"  {f.id} {f.severity:<7} {f.code} ({f.basis}){mag}: {f.message}")
    return lines


def cmd_feedback_check(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, args.allow_lossy)
    session = _session(args, scene, path)
    evaluations = [_json_arg(e, "evaluation") for e in args.evaluation or []]
    try:
        if evaluations:
            # Evaluations name observations; make the views they refer to again (renders
            # are deterministic, so the same view of the same project has the same id).
            session.diagnose()
        diagnosis = session.diagnose(evaluations=evaluations)
    except FeedbackError as exc:
        raise _error(exc) from None
    data = diagnosis.to_dict(max_findings=args.max_findings)
    if args.out:
        files = {}
        for name, obs in diagnosis.observed.items():
            files[name] = _write_images(Path(args.out), obs, ["image", "annotated"])
        data["files"] = files
    status = "accepted" if diagnosis.accepted else "NOT accepted"
    lines = [
        f"{diagnosis.expectations.label}: {status} -- {diagnosis.errors} error(s), "
        f"{diagnosis.warnings} warning(s) ({diagnosis.id})",
        *_finding_lines(diagnosis),
        *(f"  note: {n}" for n in diagnosis.notes),
    ]
    return data, "\n".join(lines), EXIT_OK if diagnosis.accepted else EXIT_GEOMETRY_ERRORS


def cmd_feedback_correct(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    if args.apply:
        _project_only(path, "feedback correct --apply")
    scene, _ = _load_scene(path, args.allow_lossy, history=args.apply)
    session = _session(args, scene, path)
    try:
        corrections = session.propose()
        data: dict[str, Any] = {"corrections": corrections.to_dict()}
        lines = [f"{len(corrections.proposals)} proposal(s) for {corrections.diagnosis.id}:"]
        for p in corrections.proposals:
            auto = (
                "auto"
                if p.policy.get("auto")
                else "needs approval"
                if p.safety == "approval"
                else "manual"
            )
            lines.append(f"  {p.id} [{p.safety}, {auto}] {p.title}")
        for u in corrections.unresolved:
            lines.append(f"  unresolved {u['finding']} {u['code']}: {u['reason']}")
        if args.preview:
            preview = session.preview(corrections)
            preview.pop("observation", None)
            data["preview"] = preview
            lines.append(
                f"preview: {len(preview.get('comparison', {}).get('resolved', []))} finding(s) "
                "resolved (nothing changed)"
            )
        if args.apply:
            approve = [a for a in (args.approve or "").split(",") if a]
            result = session.apply(corrections, approve=approve)
            data["applied"] = result.to_dict()
            lines.append(
                f"apply: {result.status}" + (f" ({result.reason})" if result.reason else "")
            )
            if result.status == "applied":
                written = _write_or_dry_run(scene, path, args)
                data["output"] = written
                lines.append(f"saved {written}" if written else "nothing written (dry run)")
    except FeedbackError as exc:
        raise _error(exc) from None
    return data, "\n".join(lines), EXIT_OK


def cmd_feedback_loop(args: argparse.Namespace) -> CommandOutput:
    from .feedback import run_loop

    path = Path(args.path)
    _project_only(path, "feedback loop")
    scene, _ = _load_scene(path, history=True)
    session = _session(args, scene, path)
    try:
        report = run_loop(session, budget=_json_arg(args.budget, "budget"))
    except FeedbackError as exc:
        raise _error(exc) from None
    data = report.to_dict()
    lines = [f"loop {report.id}: {report.stop_reason} after {len(report.iterations)} iteration(s)"]
    for it in report.iterations:
        applied = it.get("applied") or {}
        d = it["diagnosis"]
        lines.append(
            f"  {it['iteration']}: {d['errors']} error(s), {d['warnings']} warning(s)"
            + (f"; {applied.get('status')} {applied.get('applied')}" if applied else "")
        )
    changed = any((it.get("applied") or {}).get("status") == "applied" for it in report.iterations)
    if changed:
        written = _write_or_dry_run(scene, path, args)
        data["output"] = written
        lines.append(f"saved {written}" if written else "nothing written (dry run)")
    if args.out and report.diagnosis is not None:
        files = {}
        for name, obs in report.diagnosis.observed.items():
            files[name] = _write_images(Path(args.out), obs, ["image", "annotated"])
        data["files"] = files
    code = EXIT_OK if report.stop_reason == "accepted" else EXIT_GEOMETRY_ERRORS
    return data, "\n".join(lines), code


def cmd_feedback_schema(args: argparse.Namespace) -> CommandOutput:
    from .feedback import schema

    found = {
        "expectations": schema.expectations_schema,
        "vision-evaluation": schema.evaluation_schema,
        "corrections": schema.actions_schema,
    }[args.kind]()
    return {"schema": found}, json.dumps(found, indent=2), EXIT_OK


def add_feedback_parsers(sub, common, lossy, writing) -> None:
    p_fb = sub.add_parser(
        "feedback", help="visual feedback: observe renders, check expectations, correct"
    )
    fb = p_fb.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    rendering = argparse.ArgumentParser(add_help=False)
    rendering.add_argument("--backend", default="cpu", choices=BACKENDS, help="default cpu")
    rendering.add_argument("--adapter", help="GPU adapter (gpu, auto, gpu_path)")
    rendering.add_argument("--width", type=int, default=640, help="image width (default 640)")
    rendering.add_argument("--height", type=int, default=360, help="image height (default 360)")
    expect = argparse.ArgumentParser(add_help=False)
    expect.add_argument("--expect", metavar="FILE|JSON", help="an aether3d.expectations document")

    p = fb.add_parser("observe", parents=[common, lossy, rendering, expect], help="render, measure")
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument("--view", help='a view name of the expectations, or JSON ({"preset": "top"})')
    p.add_argument("--focus", metavar="JSON", help="objects whose geometry facts to compute")
    p.add_argument("--occlusion", action="store_true", help="measure the focus objects' occlusion")
    p.add_argument("--linear", action="store_true", help="read back linear light")
    p.add_argument("--spp", type=int, default=16, help="path tracer samples (default 16)")
    p.add_argument("--images", default="image,annotated", help="image,annotated,ids,depth")
    p.add_argument("--max-objects", type=int, default=50)
    p.add_argument("--out", metavar="DIR", help="write the observation and its images here")
    p.set_defaults(handler=cmd_feedback_observe)

    p = fb.add_parser(
        "check", parents=[common, lossy, rendering, expect], help="diagnose against expectations"
    )
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument(
        "--evaluation", action="append", metavar="FILE", help="a vision evaluation (repeatable)"
    )
    p.add_argument("--max-findings", type=int, default=100)
    p.add_argument("--out", metavar="DIR", help="write the observations and images here")
    p.set_defaults(handler=cmd_feedback_check)

    p = fb.add_parser(
        "correct",
        parents=[common, lossy, rendering, expect, writing],
        help="propose (and --apply) corrections",
    )
    p.add_argument("path", help=".aether project")
    p.add_argument("--policy", metavar="FILE|JSON", help="a correction policy")
    p.add_argument("--preview", action="store_true", help="preview the automatic proposals")
    p.add_argument("--apply", action="store_true", help="apply the automatic proposals and save")
    p.add_argument("--approve", metavar="IDS", help="also apply these proposals (P1,P3)")
    p.set_defaults(handler=cmd_feedback_correct)

    p = fb.add_parser(
        "loop", parents=[common, rendering, expect, writing], help="the automatic loop, then save"
    )
    p.add_argument("path", help=".aether project")
    p.add_argument("--policy", metavar="FILE|JSON", help="a correction policy")
    p.add_argument("--budget", metavar="FILE|JSON", help="the loop's budget")
    p.add_argument("--out", metavar="DIR", help="write the final observations here")
    p.set_defaults(handler=cmd_feedback_loop)

    p = fb.add_parser("schema", parents=[common], help="JSON Schemas of the feedback documents")
    p.add_argument("kind", choices=["expectations", "vision-evaluation", "corrections"])
    p.set_defaults(handler=cmd_feedback_schema)
