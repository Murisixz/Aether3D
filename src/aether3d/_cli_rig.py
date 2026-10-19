"""``aether3d rig``: character rigging from the command line (Stage 18).

* ``rig autorig PROJECT --mesh BODY [--mesh JACKET ...]`` -- detect landmarks, build a
  template rig with controls and bind every mesh (one undo step), then save.
* ``rig info PROJECT --armature A`` -- joints (roles, helpers), controls, constraints and
  skins.
* ``rig landmarks PROJECT --mesh BODY`` -- landmarks of an unrigged character (read-only).
* ``rig weights PROJECT --mesh BODY [--joint J]`` -- a skin's validity, coverage, leakage.
* ``rig diagnose PROJECT --armature A [--pose P ...]`` -- the rig and its deformation
  under the diagnostic pose suite and over the timeline.
* ``rig test PROJECT --armature A [--suite stress|suite] [--sheet OUT.png]`` -- the
  deformation under a pose suite (the extremes by default): per pose the skin's kept size,
  new creases, folds and worst stretch; ``--sheet`` writes a stretch map per pose; exit
  code 3 when a finding is an error.
* ``rig fix PROJECT --armature A [--safety safe|verify]`` -- apply proposals, verified
  one by one (undone when they do not help), then save.
* ``rig create PROJECT --armature A --rig RIG.json`` -- a rig document on an object (an
  empty is made when it does not exist), then save; ``rig edit PROJECT --armature A --op
  JSON ...`` -- rig operations (one undo step, skins rebound), then save.
* ``rig bind PROJECT --mesh M --armature A [--method heat|geodesic|envelope|nearest]
  [--smooth N] [--normals surface|joints]`` -- generate weights and bind, then save;
  ``rig transfer PROJECT --mesh JACKET --source BODY`` -- weights from a bound mesh.
* ``rig pose PROJECT --armature A --command JSON ... [--time T]`` -- pose commands as keys
  (``raise_arm``, ``reach``, ``curl_fingers``, ``roll_foot``...), then save; ``rig ikfk
  PROJECT --armature A --limb L --mode ik|fk --time T`` -- switch a limb, matched.
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

from .cli import (
    EXIT_GEOMETRY_ERRORS,
    EXIT_OK,
    CommandError,
    CommandOutput,
    _load_scene,
    _write_or_dry_run,
)


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
    from .animation.rig_report import rig_summary

    scene, _ = _load_scene(Path(args.path))
    data = _run(rig_summary, scene.animation, args.armature)
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


def cmd_rig_test(args: argparse.Namespace) -> CommandOutput:
    scene, _ = _load_scene(Path(args.path))
    a = scene.animation
    rep = _run(a.diagnose_rig, args.armature, poses=args.suite, time=args.time,
               animation=False, max_findings=200)  # fmt: skip
    s = rep["summary"]
    lines = [f"{rep['armature']} under the {args.suite} poses: {s['error']} errors, "
             f"{s['warning']} warnings, {s['info']} info",
             f"{'pose':24s} {'mesh':12s} {'kept':>6s} {'creases':>8s} {'folds':>6s} "
             f"{'stretch':>8s}"]  # fmt: skip
    for pose, meshes in rep["poses"].items():
        for mesh, r in meshes.items():
            lines.append(f"{pose:24s} {mesh[:12]:12s} {r.get('joint_ring', float('nan')):6.3f} "
                         f"{r['new_creases']:8d} {r['folded_triangles']:6d} "
                         f"{r['worst_stretch']:8.2f}")  # fmt: skip
    lines += [f"{f['severity']}: {f['code']}: {f['message']}" for f in rep["findings"][:20]]
    if args.sheet:
        rep["sheet"] = _stress_sheet(a, args.armature, list(rep["poses"]), Path(args.sheet),
                                     args.force)  # fmt: skip
        lines.append(f"wrote {args.sheet} (a stretch map per pose)")
    return rep, "\n".join(lines), EXIT_GEOMETRY_ERRORS if s["error"] else EXIT_OK


def _stress_sheet(a, armature: str, poses: list[str], out: Path, force: bool) -> dict:
    """Stretch maps of the poses, four to a row, labelled (one PNG)."""
    from .animation.frames import contact_sheet
    from .rendering.png import encode_png

    if out.exists() and not force:
        raise CommandError("FileExistsError", f"{out} exists (pass --force)", str(out), {})
    tiles = [
        _run(a.rig_preview, armature, mode="stretch", pose=p, view="three_quarter", width=320,
             height=400)["image"][..., :3]
        for p in poses
    ]  # fmt: skip
    out.write_bytes(encode_png(contact_sheet(tiles, columns=4, labels=poses)))
    return {"path": str(out), "poses": poses}


def cmd_rig_create(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    raw = json.loads(Path(args.rig).read_text(encoding="utf-8"))
    from .errors import ObjectNotFoundError

    try:
        obj = scene.get_by_name(args.armature)
    except ObjectNotFoundError:
        obj = scene.add_object(args.armature)
    _run(scene.animation.set_rig, obj, raw)
    written = _write_or_dry_run(scene, path, args)
    rig = scene.animation.rig(obj)
    data = {"path": str(path), "output": written, "armature": obj.name,
            "joints": [j.name for j in rig.joints]}  # fmt: skip
    return data, f"{obj.name}: {len(rig.joints)} joints; saved {written}", EXIT_OK


def cmd_rig_edit(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    ops = [json.loads(o) for o in args.op]
    rep = _run(scene.animation.edit_rig, args.armature, ops)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, **rep}
    return data, f"{len(ops)} rig operations; saved {written}", EXIT_OK


def cmd_rig_bind(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    options: dict[str, Any] = {"method": args.method, "normals": args.normals}
    if args.smooth:
        options["corrective"] = {"iterations": args.smooth, "strength": 0.5}
    rep = _run(scene.animation.generate_weights, args.mesh, args.armature, **options)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, **rep}
    lines = [f"{args.mesh}: {rep.get('technique', args.method)}",
             f"{rep.get('joints_used', '?')} joints used; saved {written}"]  # fmt: skip
    return data, "\n".join(lines), EXIT_OK


def cmd_rig_transfer(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    rep = _run(scene.animation.transfer_weights, args.mesh, args.source)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, **rep}
    return data, f"{args.mesh}: weights from {args.source}; saved {written}", EXIT_OK


def cmd_rig_pose(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    commands = [json.loads(c) for c in args.command]
    rep = _run(scene.animation.pose_character, args.armature, commands, time=args.time)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, **rep}
    lines = [f"{c['do']}: {json.dumps({k: v for k, v in c.items() if k != 'do'})[:120]}"
             for c in rep["commands"]]  # fmt: skip
    lines.append(f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_rig_ikfk(args: argparse.Namespace) -> CommandOutput:
    path = Path(args.path)
    scene, _ = _load_scene(path, history=True)
    rep = _run(scene.animation.ik_fk, args.armature, args.limb, args.mode, args.time)
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, **rep}
    return (
        data,
        f"{args.limb} -> {args.mode} at {args.time}: mismatch {rep['mismatch']:.3g}; "
        f"saved {written}",
        EXIT_OK,
    )


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
    p = rs.add_parser("test", parents=[common], help="the deformation under a pose suite",
                      description="Measure the deformation under the stress suite (or the "
                      "everyday one): per pose the kept size, creases, folds and stretch; "
                      "exit code 3 when a finding is an error.")  # fmt: skip
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.add_argument("--suite", default="stress", choices=("stress", "suite"))
    p.add_argument("--time", type=float)
    p.add_argument("--sheet", help="also write a stretch map per pose (PNG)")
    p.add_argument("--force", action="store_true", help="overwrite an existing sheet")
    p.set_defaults(handler=cmd_rig_test)
    p = rs.add_parser("create", parents=[common, writing], help="a rig document on an object")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True, help="the object (made when missing)")
    p.add_argument("--rig", required=True, help="a JSON rig document (docs/ANIMATION.md)")
    p.set_defaults(handler=cmd_rig_create)
    p = rs.add_parser("edit", parents=[common, writing], help="rig operations")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.add_argument("--op", action="append", required=True, help="a JSON operation (repeatable)")
    p.set_defaults(handler=cmd_rig_edit)
    p = rs.add_parser("bind", parents=[common, writing], help="generate weights and bind")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--mesh", required=True)
    p.add_argument("--armature", required=True)
    p.add_argument("--method", default="heat", choices=("heat", "geodesic", "envelope", "nearest"))
    p.add_argument("--smooth", type=int, help="corrective smoothing iterations (delta mush)")
    p.add_argument("--normals", default="surface", choices=("surface", "joints"))
    p.set_defaults(handler=cmd_rig_bind)
    p = rs.add_parser("transfer", parents=[common, writing], help="weights from a bound mesh")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--mesh", required=True, help="the mesh to bind (clothing)")
    p.add_argument("--source", required=True, help="the bound mesh it takes weights from")
    p.set_defaults(handler=cmd_rig_transfer)
    p = rs.add_parser("pose", parents=[common, writing], help="pose commands as keys")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.add_argument("--command", action="append", required=True,
                   help='a JSON command, e.g. \'{"do": "raise_arm", "side": "L"}\'')  # fmt: skip
    p.add_argument("--time", type=float, default=0.0)
    p.set_defaults(handler=cmd_rig_pose)
    p = rs.add_parser("ikfk", parents=[common, writing], help="switch a limb between IK and FK")
    p.add_argument("path", help="the .aether project")
    p.add_argument("--armature", required=True)
    p.add_argument("--limb", required=True, help="an IK constraint or one of its controls")
    p.add_argument("--mode", required=True, choices=("ik", "fk"))
    p.add_argument("--time", type=float, required=True)
    p.set_defaults(handler=cmd_rig_ikfk)
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
