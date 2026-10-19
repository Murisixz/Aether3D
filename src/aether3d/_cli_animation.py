"""``aether3d animation``: animation from the command line (Stage 17).

* ``animation run PROJECT SCRIPT`` -- apply an ``aether3d.animation-script`` (keys,
  curves, clips, NLA, rigs, IK, constraints, generators, cameras) as one undo step, all
  or nothing, and save.
* ``animation info PROJECT`` -- the timeline, clips, bindings, rigs and skins.
* ``animation report PROJECT`` -- measurements: ``--kind curve`` (one channel's exact
  report), ``diagnose`` (findings over the whole animation), ``trajectory`` (an object
  or ``--joint``), ``contacts`` (``--feet``: planted intervals, sliding, penetration).
* ``animation graph PROJECT OUT.svg|OUT.png`` -- value, velocity and acceleration of
  channels (``--target`` and ``--path``, repeatable) or of a ``--clip``.
* ``animation render PROJECT OUTDIR`` -- PNG frames at exact frame times (``--start``,
  ``--end``, ``--fps``, ``--step``), motion blur, a contact sheet, an FFmpeg video.
* ``animation schema`` -- the JSON Schema of animation scripts.

glTF export with animation is ``aether3d export --animation MODE``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli import (
    EXIT_OK,
    CommandError,
    CommandOutput,
    _check_destination,
    _load_scene,
    _project_only,
    _read_script,
    _script_label,
    _write_or_dry_run,
)


def _engine(exc: Exception, location: str | None = None) -> CommandError:
    code = getattr(exc, "code", None)
    details = {"code": code} if code else {}
    hint = getattr(exc, "hint", None)
    if hint:
        details["hint"] = hint
    message = getattr(exc, "message", None) or str(exc)
    return CommandError(type(exc).__name__, message, getattr(exc, "location", location), details)


def cmd_animation_run(args: argparse.Namespace) -> CommandOutput:
    from .animation.script import AnimationScriptError, run_animation_script

    path = Path(args.path)
    _project_only(path, "animation run")
    scene, _ = _load_scene(path, history=True)
    script = _read_script(args.script)
    try:
        result = run_animation_script(
            scene, script, label=_script_label(script, "animation run", args)
        )
    except AnimationScriptError as exc:
        details = {k: v for k, v in exc.to_dict().items() if k not in ("message", "location")}
        raise CommandError("AnimationScriptError", exc.message, exc.location, details) from None
    written = _write_or_dry_run(scene, path, args)
    data = {"path": str(path), "output": written, "dry_run": args.dry_run, **result}
    summary = result["animation"]
    lines = [f"{len(result['operations'])} operations applied as one step"]
    lines += [
        f"clip {c['name']}: {c['channels']} channels, {c['keys']} keys" for c in summary["clips"]
    ]
    lines.append("dry run: nothing written" if args.dry_run else f"saved {written}")
    return data, "\n".join(lines), EXIT_OK


def cmd_animation_info(args: argparse.Namespace) -> CommandOutput:
    from .animation.script import summary

    path = Path(args.path)
    scene, _ = _load_scene(path)
    data = {"path": str(path), **summary(scene)}
    t = data.get("timeline", {})
    lines = [
        f"timeline {t.get('start', 0):g}..{t.get('end', 0):g} s at {t.get('fps', '?')} fps"
        if t
        else "no animation"
    ]
    for c in data["clips"]:
        lines.append(f"clip {c['name']}: {c['channels']} channels, {c['keys']} keys, "
                     f"{c['range'][0]:g}..{c['range'][1]:g} s")  # fmt: skip
    for b in data["bindings"]:
        lines.append(f"{b['target']}: action {b['action']}, layers {b['layers'] or '-'}")
    counts = ", ".join(f"{k} {data.get(k, 0)}" for k in ("rigs", "skins", "morphs", "constraints"))
    lines.append(counts)
    return data, "\n".join(lines), EXIT_OK


def cmd_animation_report(args: argparse.Namespace) -> CommandOutput:
    from .animation.script import _jsonable
    from .errors import AetherError

    path = Path(args.path)
    scene, _ = _load_scene(path)
    system = scene.animation
    try:
        if args.kind == "curve":
            if not args.path_ or (not args.target and not args.clip):
                raise CommandError("ValidationError", "curve needs --path and --target or --clip")
            data = system.curve_report(
                args.target, args.path_[0], clip=args.clip, samples=args.samples
            )
        elif args.kind == "diagnose":
            data = system.diagnose()
        elif args.kind == "trajectory":
            if not args.target:
                raise CommandError("ValidationError", "trajectory needs --target")
            data = system.trajectory(args.target, start=args.start, end=args.end, joint=args.joint)
        elif args.kind == "motion":
            targets = None
            if args.target:
                targets = [{"object": args.target, "joint": args.joint} if args.joint
                           else args.target]  # fmt: skip
            data = system.motion_report(targets, start=args.start, end=args.end,
                                        subject=args.subject, loop=args.loop)  # fmt: skip
        else:
            if not args.target or not args.feet:
                raise CommandError(
                    "ValidationError", "contacts needs --target (armature) and --feet"
                )
            data = system.foot_contacts(args.target, args.feet, start=args.start, end=args.end,
                                        height=args.height, speed=args.speed)  # fmt: skip
    except AetherError as exc:
        raise _engine(exc) from None
    data = _jsonable(data)
    return {"path": str(path), "kind": args.kind, **data}, json.dumps(data, indent=2), EXIT_OK


def cmd_animation_graph(args: argparse.Namespace) -> CommandOutput:
    from ._fileio import write_atomic
    from .errors import AetherError
    from .rendering.png import encode_png

    path, out = Path(args.path), Path(args.output)
    suffix = out.suffix.lower()
    if suffix not in (".svg", ".png"):
        raise CommandError("ValidationError", "the graph is written as .svg or .png")
    _check_destination(out, args.force)
    scene, _ = _load_scene(path)
    system = scene.animation
    options = {
        "start": args.start,
        "end": args.end,
        "width": args.width,
        "title": args.title,
        "plots": tuple(p.strip() for p in args.plots.split(",") if p.strip()),
        "annotate": not args.no_annotate,
    }
    try:
        if args.path_:
            if not args.target and not args.clip:
                raise CommandError("ValidationError", "--path needs --target (or --clip)")
            channels = [(args.target, p) for p in args.path_]
            result = system.graph(channels, clip=args.clip, format=suffix[1:], **options)
        elif args.clip:
            result = system.graph(clip=args.clip, format=suffix[1:], **options)
        else:
            raise CommandError("ValidationError", "graph needs --path (with --target) or --clip")
    except AetherError as exc:
        raise _engine(exc) from None
    data_bytes = result.encode("utf-8") if suffix == ".svg" else encode_png(result)
    written = write_atomic(out, data_bytes)
    data = {"path": str(path), "output": str(written), "bytes": len(data_bytes)}
    return data, f"wrote {written} ({len(data_bytes)} bytes)", EXIT_OK


def cmd_animation_render(args: argparse.Namespace) -> CommandOutput:
    from ._fileio import write_atomic
    from .animation.frames import assemble_video, contact_sheet, render_frames
    from .errors import AetherError
    from .materials.images import decode_png
    from .rendering.png import encode_png

    path, out_dir = Path(args.path), Path(args.output)
    scene, _ = _load_scene(path)
    blur = None
    if args.motion_blur:
        try:
            shutter, samples = args.motion_blur.split(":")
            blur = {"shutter": float(shutter), "samples": int(samples)}
        except ValueError:
            raise CommandError(
                "ValidationError", "--motion-blur is SHUTTER:SAMPLES (0.5:8)"
            ) from None
    options: dict = {"width": args.width, "height": args.height, "backend": args.backend}
    if args.camera:
        options["camera"] = args.camera
    if args.tonemap:
        options["tonemap"] = args.tonemap
    if args.exposure is not None:
        options["exposure"] = args.exposure
    if args.backend in ("path", "gpu_path"):
        path_opts = {k: v for k, v in (("samples", args.spp), ("bounces", args.bounces),
                                       ("sampler", args.sampler)) if v is not None}  # fmt: skip
        if path_opts:
            options["path"] = path_opts
        if args.denoise:
            options["denoise"] = True
    elif args.spp is not None or args.bounces is not None or args.sampler or args.denoise:
        raise CommandError(
            "UsageError", "--spp, --bounces, --sampler and --denoise apply to --backend path "
            "or gpu_path",
        )  # fmt: skip
    passes = [p for p in args.passes.split(",") if p] if args.passes else None
    try:
        report = render_frames(
            scene, out_dir, start=args.start, end=args.end, fps=args.fps, step=args.step,
            motion_blur=blur, format=args.format, passes=passes, resume=args.resume,
            **options,
        )  # fmt: skip
        data = {"path": str(path), **report}
        if args.contact_sheet:
            frames = report["frames"][:: max(1, len(report["frames"]) // args.sheet_count or 1)]
            images = [decode_png(Path(f["path"]).read_bytes()).pixels[..., :3] for f in frames]
            labels = [f"{f['time']:.2f}s" for f in frames]
            sheet = contact_sheet(images, columns=args.columns, labels=labels)
            data["contact_sheet"] = str(write_atomic(args.contact_sheet, encode_png(sheet)))
        if args.video or args.lossless:
            if args.format == "exr":
                raise CommandError(
                    "UsageError", "videos are made of PNG frames: render --format png or png16"
                )
            for target, lossless in ((args.video, False), (args.lossless, True)):
                if target:
                    data["video" if not lossless else "lossless"] = assemble_video(
                        out_dir, "frame_{frame:05d}.png", report["fps"], target,
                        crf=args.crf, preset=args.preset, lossless=lossless,
                        deep=args.format == "png16",
                    )  # fmt: skip
    except AetherError as exc:
        raise _engine(exc) from None
    text = f"{len(report['frames'])} frames in {out_dir} ({report['seconds']:.1f} s)"
    return data, text, EXIT_OK


def cmd_animation_schema(args: argparse.Namespace) -> CommandOutput:
    from .animation.schema import animation_script_schema

    schema = animation_script_schema()
    return schema, json.dumps(schema, indent=2), EXIT_OK


def add_animation_parsers(sub, common, writing) -> None:
    p_an = sub.add_parser("animation", help="animation scripts, reports, graphs and frames")
    an = p_an.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    p = an.add_parser(
        "run",
        parents=[common, writing],
        help="apply a JSON animation script to a project (atomically)",
        description="Apply an aether3d.animation-script document (docs/ANIMATION.md): keys, "
        "curves, clips, layers, rigs, skins, IK, constraints, motion generators and cameras, "
        "as one step of the project's history (all or nothing), then save.",
    )
    p.add_argument("path", help="the .aether project")
    p.add_argument("script", help="animation script (.json), or - to read it from stdin")
    p.set_defaults(handler=cmd_animation_run)
    p = an.add_parser("info", parents=[common], help="the timeline, clips and bindings")
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.set_defaults(handler=cmd_animation_info)
    p = an.add_parser(
        "report",
        parents=[common],
        help="measure curves and motion (read-only)",
        description="curve: one channel's exact report (continuity at every key, extrema, "
        "overshoot, velocity/acceleration/jerk); diagnose: findings over the whole animation; "
        "trajectory: an object's (or --joint's) world motion; motion: world-space quality "
        "(jumps, acceleration steps, jitter, snaps, pole flips, pauses, loop seams, framing "
        "of a --subject); contacts: feet sliding.",
    )
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument(
        "--kind",
        choices=("curve", "diagnose", "trajectory", "motion", "contacts"),
        default="curve",
    )
    p.add_argument("--subject", help="motion: the object a camera should frame")
    p.add_argument("--loop", action="store_true", help="motion: check the loop seam")
    p.add_argument("--target", help="object (or armature) id or name")
    p.add_argument("--path", dest="path_", action="append", help="channel path")
    p.add_argument("--clip", help="a clip name (instead of the target's action)")
    p.add_argument("--joint", help="a joint (trajectory)")
    p.add_argument("--feet", action="append", help="foot joint (contacts; repeatable)")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--height", type=float, default=0.03, help="contacts: planted height")
    p.add_argument("--speed", type=float, default=0.3, help="contacts: planted speed")
    p.add_argument("--samples", type=int, default=0, help="curve: also return samples")
    p.set_defaults(handler=cmd_animation_report)
    p = an.add_parser("graph", parents=[common], help="graph channels as SVG or PNG")
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument("output", help="the .svg or .png to write")
    p.add_argument("--target", help="object id or name")
    p.add_argument("--path", dest="path_", action="append", help="channel path (repeatable)")
    p.add_argument("--clip", help="graph a clip's curves")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--width", type=int, default=900)
    p.add_argument("--title")
    p.add_argument(
        "--plots",
        default="value,velocity,acceleration",
        help="plots, comma-separated: value, velocity, acceleration, jerk",
    )
    p.add_argument(
        "--no-annotate",
        action="store_true",
        help="leave out the discontinuity and extremum marks",
    )
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(handler=cmd_animation_graph)
    p = an.add_parser("render", parents=[common], help="render frames (PNG), a sheet, a video")
    p.add_argument("path", help=".aether project or .glb/.gltf file")
    p.add_argument("output", help="directory for the frames")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.add_argument("--fps", help="frame rate (default: the timeline's), e.g. 24 or 30000/1001")
    p.add_argument("--step", type=int, default=1, help="every Nth frame")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--camera", help="camera id or name (default: the shots', else active)")
    p.add_argument("--backend", choices=("auto", "gpu", "cpu", "path", "gpu_path"),
                   default="auto")  # fmt: skip
    p.add_argument("--spp", type=int, help="path tracers: samples per pixel")
    p.add_argument("--bounces", type=int, help="path tracers: indirect bounces")
    p.add_argument("--sampler", choices=("random", "sobol"), help="path tracers' numbers")
    p.add_argument("--denoise", action="store_true", help="path tracers: denoise each frame")
    p.add_argument("--tonemap", choices=("none", "neutral", "film"), help="view transform")
    p.add_argument("--exposure", type=float, help="exposure in stops")
    p.add_argument("--format", choices=("png", "png16", "exr"), default="png",
                   help="frames: 8-bit PNG, 16-bit PNG masters or linear OpenEXR")  # fmt: skip
    p.add_argument("--passes", help="exr: passes to add (depth,normal,albedo,direct,...)")
    p.add_argument("--resume", action="store_true", help="keep frames already rendered")
    p.add_argument("--motion-blur", metavar="SHUTTER:SAMPLES", help="e.g. 0.5:8")
    p.add_argument("--contact-sheet", metavar="PNG", help="also write a contact sheet")
    p.add_argument("--sheet-count", type=int, default=12, help="frames on the sheet")
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--video", metavar="MP4", help="assemble a video with FFmpeg (if installed)")
    p.add_argument("--crf", type=int, default=10, help="video quality, lower is better (10)")
    p.add_argument("--preset", default="slow", help="x264 preset (slow)")
    p.add_argument("--lossless", metavar="MKV", help="also assemble a lossless FFV1 master")
    p.set_defaults(handler=cmd_animation_render)
    p = an.add_parser("schema", parents=[common], help="the JSON Schema of animation scripts")
    p.set_defaults(handler=cmd_animation_schema)
