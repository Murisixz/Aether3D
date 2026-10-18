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
  or skeleton (with its SVG), from ``view``.
"""

from __future__ import annotations

from typing import Any

from .session import Session, ToolError

WHAT = ("summary", "landmarks", "weights", "diagnose", "fixes", "preview")
MAX_FINDINGS = 30


def rig_summary(scene: Any, armature: Any) -> dict[str, Any]:
    system = scene.animation
    arm = system._object(armature, "target")
    rig = system.rig(arm)
    if rig is None:
        raise ToolError("RIG_MISSING", f"{arm.name!r} has no rig", hint="autorig a mesh first")
    item = system._owned[("rig", arm._id)]
    skins = []
    for (kind, oid), sk in system._owned.items():
        if kind == "skin" and sk._definition.rig == item.id and oid in scene._objects:
            d = sk._definition
            skins.append({"mesh": scene._objects[oid].name, "vertices": d.vertex_count,
                          "joints": len(d.joints), "influences": int(d.indices.shape[1]),
                          "method": d.method})  # fmt: skip
    return {
        "armature": arm.name,
        "template": rig.template,
        "info": rig.info_dict,
        "joints": [
            {
                "name": j.name,
                "parent": None if j.parent < 0 else rig.joints[j.parent].name,
                **({"role": j.role} if j.role else {}),
                **({"deform": False} if not j.deform else {}),
            }
            for j in rig.joints
        ],  # fmt: skip
        "controls": [
            {
                "name": c.name,
                "kind": c.kind,
                **({"object": scene._objects[c.object].name} if c.object in scene._objects else {}),
                **({"joint": c.joint} if c.joint else {}),
                **({"constraint": c.constraint} if c.constraint else {}),
            }
            for c in rig.controls
        ],  # fmt: skip
        "constraints": [
            {"name": c.name, "type": c.type, "influence": c.influence} for c in rig.constraints
        ],
        "correctives": [c.name for c in rig.correctives],
        "attachments": len(rig.attachments),
        "skins": skins,
        "landmarks": len(rig.landmarks),
        "enforce_limits": rig.enforce_limits,
    }


def inspect_rig(session: Session, args: dict, ctx: Any):
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
            data = rig_summary(scene, _need(args, "target"))
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
            "preview (PNG: mode pose|weights|stretch|skeleton). Rig with run_script.",
            _schema(
                {
                    "what": {"enum": list(WHAT)},
                    "target": text,
                    "meshes": {"type": "array", "items": text},
                    "joint": text,
                    "pose": text,
                    "mode": {"enum": ["pose", "weights", "stretch", "skeleton"]},
                    "view": {"enum": ["front", "side", "back", "three_quarter"]},
                    "time": {"type": "number"},
                }
            ),
            inspect_rig,
            read_only=True,
            idempotent=True,
        ),
    )
