"""The rigging tool of the MCP server (Stage 18).

Rigging changes go through ``run_script`` with an ``aether3d.animation-script`` (one undo
step, all or nothing): ``autorig``, ``edit_rig``, ``generate_weights``, ``edit_weights``,
``transfer_weights``, ``add_control``, ``ik_fk``, ``pose_character``, ``apply_rig_fix``...
(guide ``aether3d://rigging/guide``). ``inspect_rig`` answers read-only questions:

* ``summary`` -- an armature's rig: joints (roles, helpers), controls, constraints, skins;
* ``landmarks`` -- the landmarks of unrigged ``meshes`` with confidences and diagnostics;
* ``weights`` -- a mesh's skin: validity, coverage, leakage (``joint``: its weights);
* ``diagnose`` -- the rig, its skins and the deformation under the diagnostic pose suite;
* ``fixes`` -- correction proposals for the findings (apply them with ``apply_rig_fix``);
* ``preview`` -- a picture (PNG): ``mode`` pose, weights (``joint``), stretch (``pose``)
  or skeleton (with its SVG), from ``view``;
* ``parts`` -- a rigid-part character (no bones): the parts found under ``target`` (its
  root), their pivots and sides (animate it with ``generate_rigid``).
"""

from __future__ import annotations

from typing import Any

from .session import Session, ToolError

WHAT = ("summary", "landmarks", "weights", "diagnose", "fixes", "preview", "parts")
MAX_FINDINGS = 30


def inspect_rig(session: Session, args: dict, ctx: Any):
    from ..animation.rig_report import rig_summary
    from ..animation.script import _jsonable
    from ..errors import AetherError
    from .tools import ToolOutput, scene_status

    scene = session.require_scene()
    system = scene.animation
    what = args.get("what", "summary")
    image = None
    try:
        if what == "landmarks":
            meshes = args.get("meshes") or ([args["target"]] if args.get("target") else None)
            if not meshes:
                raise ToolError("INVALID_ARGUMENT", "landmarks needs meshes (or target)")
            data = system.detect_landmarks(meshes)
        elif what == "weights":
            data = system.inspect_weights(_need(args, "target"), joint=args.get("joint"))
        elif what == "diagnose":
            poses = [args["pose"]] if args.get("pose") else "suite"
            data = system.diagnose_rig(_need(args, "target"), time=args.get("time"),
                                       max_findings=MAX_FINDINGS, poses=poses)  # fmt: skip
        elif what == "fixes":
            data = system.propose_rig_fix(_need(args, "target"), time=args.get("time"))
        elif what == "parts":
            data = system.rigid_parts(_need(args, "target"))
        elif what == "preview":
            from ..rendering.png import encode_png

            out = system.rig_preview(
                _need(args, "target"),
                mode=args.get("mode", "pose"),
                joint=args.get("joint"),
                pose=args.get("pose"),
                time=args.get("time"),
                view=args.get("view", "front"),
                width=480,
                height=640,
            )
            image = encode_png(out.pop("image"))
            data = {k: v for k, v in out.items() if k != "svg"}
            if "svg" in out:
                data["svg_bytes"] = len(out["svg"])
        else:
            data = rig_summary(system, _need(args, "target"))
    except AetherError:
        raise  # (reported by the server as a structured error)
    return ToolOutput({"what": what, **_jsonable(data), "scene": scene_status(session)},
                      image=image)  # fmt: skip


def _need(args: dict, key: str) -> Any:
    if key in args and args[key] is not None:
        return args[key]
    raise ToolError("INVALID_ARGUMENT", f"this needs {key}")


def specs() -> tuple[Any, ...]:
    from .tools import ToolSpec, _schema

    text = {"type": "string"}
    return (
        ToolSpec(
            "inspect_rig",
            "Inspect a rig",
            "Read-only rigging questions: summary (joints, controls, skins); landmarks of "
            "unrigged meshes; weights of a mesh; diagnose (pose suite); fixes (proposals); "
            "preview (PNG: mode pose|weights|stretch|skeleton); parts (a rigid-part "
            "character, no bones). Rig with run_script.",
            _schema(
                {
                    "what": {"enum": list(WHAT)},
                    "target": text,
                    "meshes": {"type": "array", "items": text},
                    "joint": text,
                    "pose": text,
                    "mode": {"enum": ["pose", "weights", "stretch", "skeleton"]},
                    "view": {"enum": ["front", "side", "left", "back", "three_quarter"]},
                    "time": {"type": "number"},
                }
            ),
            inspect_rig,
            read_only=True,
            idempotent=True,
        ),
    )
