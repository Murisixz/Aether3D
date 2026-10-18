"""``aether3d rig``: character rigging from the command line (Stage 18).

* ``rig autorig PROJECT --mesh BODY [--mesh JACKET ...]`` -- detect landmarks, build a
  template rig with controls and bind every mesh (one undo step), then save.
* ``rig info PROJECT --armature A`` -- joints (roles, helpers), controls, constraints and
  skins.
* ``rig landmarks PROJECT --mesh BODY`` -- landmarks of an unrigged character (read-only).
* ``rig weights PROJECT --mesh BODY [--joint J]`` -- a skin's validity, coverage, leakage.
* ``rig diagnose PROJECT --armature A [--pose P ...]`` -- the rig and its deformation
  under the diagnostic pose suite and over the timeline.
* ``rig fix PROJECT --armature A [--safety safe|verify]`` -- apply proposals, verified
  one by one (undone when they do not help), then save.
* ``rig preview PROJECT OUT.png --armature A [--mode pose|weights|stretch|skeleton]`` --
  a picture; ``.svg`` writes the skeleton diagram.
* ``rig parts PROJECT --root R`` -- the parts of a rigid-part character (no bones):
  roles, pivots, sides (read-only).
* ``rig convert PROJECT --source S --to skeleton|rigid`` -- on request, a rigid
  hierarchy to an armature with attached copies, or an armature's attached parts to a
  rigid hierarchy; motion baked and verified, the source kept (hidden), then save.
* ``rig guide`` -- the agent guide (Markdown).

Rigging scripts are animation scripts: ``aether3d animation run PROJECT SCRIPT``
(``generate_rigid`` animates a rigid-part character without bones).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cli import EXIT_OK, CommandError, CommandOutput, _load_scene, _write_or_dry_run


def _engine(exc: Exception) -> CommandError:
    details = {"code": getattr(exc, "code", None)}
    if getattr(exc, "hint", None):
        details["hint"] = exc.hint  # type: ignore[attr-defined]
    message = getattr(exc, "message", None) or str(exc)
    return CommandError(type(exc).__name__, message, getattr(exc, "location", None), details)


def _run(fn, *a: Any, **kw: Any) -> Any:
    from .errors import AetherError

    try:
        return fn(*a, **kw)
    except AetherError as exc:
        raise _engine(exc) from None


def cmd_rig_autorig(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    options: dict[str, Any] = {"template": args.template}
    if args.name:
        options["name"] = args.name
    if args.forward:
        options["forward"] = args.forward
    if args.up:
        options["up"] = args.up
    if args.landmarks:
        options["landmarks"] = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    rep = _run(scene.animation.autorig, args.mesh, dry_run=args.dry_run, **options)
    written = None if args.dry_run else _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, **rep}
    lines = [
        f"{rep['template']} rig {rep.get('armature', '(dry run)')}: "
        f"{len(rep['joints'])} joints, confidence {rep['confidence']}",
        f"pose: {rep.get('pose', '?')}, facing {rep['frame']['forward']}",
    ]
    lines += [f"{d['severity']}: {d['code']}: {d['message']}" for d in rep["diagnostics"][:10]]
    for name, w in rep.get("weights", {}).items():
        lines.append(f"{name}: {w['mode']} weights, {w.get('joints_used', '?')} joints used")
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_rig_info(args: argparse.Namespace) -> CommandOutput:
    from .mcp.rigging import rig_summary

    scene, _ = _load_scene(Path(args.path))
    data = _run(rig_summary, scene, args.armature)
    lines = [f"{data['armature']}: {len(data['joints'])} joints ({data['template'] or 'built'})"]
    lines += [f"control {c['name']} ({c['kind']})" for c in data["controls"][:40]]
    lines += [f"skin {s['mesh']}: {s['vertices']} vertices, {s['method']}" for s in data["skins"]]
    return data, "\n".join(lines), EXIT_OK


def cmd_rig_landmarks(args: argparse.Namespace) -> CommandOutput:
    scene, _ = _load_scene(Path(args.path))
    rep = _run(scene.animation.detect_landmarks, args.mesh, template=args.template)
    lines = [f"confidence {rep['confidence']}, pose {rep.get('pose', '?')}"]
    lines += [
        f"{n}: {lm['position']} ({lm['confidence']}, {lm['method']})"
        for n, lm in rep["landmarks"].items()
    ]
    return rep, "\n".join(lines), EXIT_OK


def cmd_rig_weights(args: argparse.Namespace) -> CommandOutput:
    scene, _ = _load_scene(Path(args.path))
    rep = _run(scene.animation.inspect_weights, args.mesh, joint=args.joint)
    lines = [f"{rep['mesh']}: {rep['vertices']} vertices, up to {rep['max_influences']} "
             f"influences, valid {rep['valid']}"]  # fmt: skip
    lines += [f"{f['severity']}: {f['code']} ({f.get('count', '')})" for f in rep["findings"]]
    return rep, "\n".join(lines), EXIT_OK


def cmd_rig_diagnose(args: argparse.Namespace) -> CommandOutput:
    scene, _ = _load_scene(Path(args.path))
    poses: Any = args.pose or "suite"
    rep = _run(scene.animation.diagnose_rig, args.armature, poses=poses, time=args.time)
    s = rep["summary"]
    lines = [f"{rep['armature']}: {s['error']} errors, {s['warning']} warnings, {s['info']} info"]
    lines += [f"{f['severity']}: {f['code']}: {f['message']}" for f in rep["findings"][:30]]
    return rep, "\n".join(lines), EXIT_OK


def cmd_rig_fix(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    rep = _run(scene.animation.apply_rig_fix, args.armature, args.safety)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **rep}
    lines = [f"applied {len(rep['applied'])}, undone {len(rep['reverted'])}",
             f"before {rep['before']}, after {rep['after']}"]  # fmt: skip
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_rig_preview(args: argparse.Namespace) -> CommandOutput:
    out = Path(args.output)
    if out.exists() and not args.force:
        raise CommandError("FileExistsError", f"{out} exists (pass --force)", str(out), {})
    scene, _ = _load_scene(Path(args.path))
    svg = out.suffix.lower() == ".svg"
    mode = "skeleton" if svg else args.mode
    rep = _run(
        scene.animation.rig_preview, args.armature, mode=mode, joint=args.joint, pose=args.pose,
        time=args.time, view=args.view, width=args.width, height=args.height,
        backend=args.backend, path=None if svg else out,
    )  # fmt: skip
    if svg:
        out.write_text(rep["svg"], encoding="utf-8")
    data = {k: v for k, v in rep.items() if k not in ("image", "svg")}
    data["output"] = str(out)
    return data, f"wrote {out} ({mode}, {args.view})", EXIT_OK


def cmd_rig_parts(args: argparse.Namespace) -> CommandOutput:
    scene, _ = _load_scene(Path(args.path))
    rep = _run(scene.animation.rigid_parts, args.root, forward=args.forward)
    lines = [f"{rep['root']}: {len(rep['parts'])} parts, facing {rep['forward']}"]
    lines += [
        f"{role}: {p['object']} (pivot {p['pivot']}, {p['pivot_source']})"
        for role, p in rep["parts"].items()
    ]
    lines += [f"{d['severity']}: {d['code']}: {d['message']}" for d in rep["diagnostics"]]
    return rep, "\n".join(lines), EXIT_OK


def cmd_rig_convert(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    options: dict[str, Any] = {"hide_original": not args.keep_visible, "forward": args.forward}
    if args.name:
        options["name"] = args.name
    if args.to == "rigid":
        options.pop("forward")
    rep = _run(scene.animation.convert_character, args.source, args.to, **options)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **rep}
    v = rep["verification"]
    made = rep.get("armature") or rep.get("root")
    lines = [
        f"{rep['source']} -> {made} ({args.to})",
        f"verified at {v['samples']} times: within {v['position']:.3g} m, "
        f"{v['rotation_deg']:.3g} deg",
    ]
    lines += [f"{d['severity']}: {d['code']}: {d['message']}" for d in rep["diagnostics"]]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_rig_guide(args: argparse.Namespace) -> CommandOutput:
    from importlib import resources

    text = (
        resources.files("aether3d.animation")
        .joinpath("rigging_guide.md")
        .read_text(encoding="utf-8")
    )
    return {"guide": text}, text, EXIT_OK


def add_rig_parsers(sub, common, writing) -> None:
    p_r = sub.add_parser("rig", help="character rigging: auto-rig, weights, diagnostics")
    rs = p_r.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = rs.add_parser("autorig", parents=[common, writing], help="rig and bind meshes",
                      description="Detect landmarks, build a template rig with controls and "
                      "bind every mesh (docs/CHARACTER_RIGGING.md), as one undo step.")  # fmt: skip
    p.add_argument("path", help="the .aether project")
    p.add_argument("--mesh", action="append", required=True, help="a mesh object (repeatable)")
    p.add_argument("--template", default="humanoid",
                   choices=("humanoid", "stylized", "blocky", "quadruped"))  # fmt: skip
    p.add_argument("--name", help="the armature's name")
    p.add_argument("--forward", choices=("+x", "-x", "+y", "-y"), help="where it faces")
    p.add_argument("--up", choices=("+x", "-x", "+y", "-y", "+z", "-z"), help="its up axis")
    p.add_argument("--landmarks", help="a JSON file of landmark overrides {name: [x, y, z]}")
    p.set_defaults(handler=cmd_rig_autorig)
    p = rs.add_parser("info", parents=[common], help="an armature's rig")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.set_defaults(handler=cmd_rig_info)
    p = rs.add_parser("landmarks", parents=[common], help="landmarks of unrigged meshes")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--mesh", action="append", required=True)
    p.add_argument("--template", default="humanoid",
                   choices=("humanoid", "blocky", "quadruped"))  # fmt: skip
    p.set_defaults(handler=cmd_rig_landmarks)
    p = rs.add_parser("weights", parents=[common], help="a skin's weights measured")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--mesh", required=True)
    p.add_argument("--joint", help="also this joint's weights")
    p.set_defaults(handler=cmd_rig_weights)
    p = rs.add_parser("diagnose", parents=[common], help="measure a rig and its deformation")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.add_argument("--pose", action="append", help="a diagnostic pose (repeatable)")
    p.add_argument("--time", type=float)
    p.set_defaults(handler=cmd_rig_diagnose)
    p = rs.add_parser("fix", parents=[common, writing], help="apply verified fixes")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.add_argument("--safety", default="safe", choices=("safe", "verify"))
    p.set_defaults(handler=cmd_rig_fix)
    p = rs.add_parser("preview", parents=[common], help="a picture of the rigged character")
    p.add_argument("path", help="the .aether project")
    p.add_argument("output", help="a .png, or .svg (the skeleton)")
    p.add_argument("--armature", required=True)
    p.add_argument("--mode", default="pose", choices=("pose", "weights", "stretch", "skeleton"))
    p.add_argument("--joint")
    p.add_argument("--pose")
    p.add_argument("--time", type=float)
    p.add_argument(
        "--view", default="front", choices=("front", "side", "left", "back", "three_quarter")
    )
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--backend", default="cpu", choices=("cpu", "gpu", "auto"))
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(handler=cmd_rig_preview)
    p = rs.add_parser("parts", parents=[common], help="a rigid-part character's parts")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--root", required=True, help="the character's root object")
    p.add_argument("--forward", default="-y", choices=("+x", "-x", "+y", "-y"))
    p.set_defaults(handler=cmd_rig_parts)
    p = rs.add_parser("convert", parents=[common, writing], help="convert between workflows",
                      description="On request: a rigid-part hierarchy to an armature with "
                      "attached copies (--to skeleton), or an armature's attached parts to a "
                      "rigid hierarchy (--to rigid); motion baked and verified; the source "
                      "is kept (hidden unless --keep-visible).")  # fmt: skip
    p.add_argument("path", help="the .aether project")
    p.add_argument("--source", required=True, help="the rigid root, or the armature")
    p.add_argument("--to", required=True, choices=("skeleton", "rigid"))
    p.add_argument("--name", help="the new armature or root's name")
    p.add_argument("--forward", default="-y", choices=("+x", "-x", "+y", "-y"))
    p.add_argument("--keep-visible", action="store_true", help="do not hide the source")
    p.set_defaults(handler=cmd_rig_convert)
    p = rs.add_parser("guide", parents=[common], help="the agent guide to rigging")
    p.set_defaults(handler=cmd_rig_guide)
