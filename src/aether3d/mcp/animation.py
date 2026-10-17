"""The animation tools of the MCP server (Stage 17).

Authoring goes through ``run_script`` with an ``aether3d.animation-script`` (one undo
step, all or nothing; schema ``aether3d://schema/animation-script``, guide
``aether3d://animation/guide``); renders take a ``time``; ``export_scene`` takes
``animation`` (glTF). One read-only tool measures:

* ``inspect_animation`` -- ``summary`` (timeline, clips, bindings, rigs, skins), ``curve``
  (one channel's exact report: continuity at every key, extrema, overshoot, velocity,
  acceleration and jerk peaks), ``diagnose`` (findings over the whole animation),
  ``trajectory`` (an object's or a joint's world motion), ``contacts`` (feet: planted
  intervals, sliding, penetration) and ``graph`` (value, velocity and acceleration of
  channels as a PNG -- an image for agents that can see; the numbers are in ``curve``).
"""

from __future__ import annotations

from typing import Any

from .session import Session, ToolError

WHAT = ("summary", "curve", "diagnose", "trajectory", "contacts", "graph")


def inspect_animation(session: Session, args: dict, ctx: Any):
    from ..animation.script import _jsonable, summary
    from ..errors import AetherError
    from .tools import ToolOutput, scene_status

    scene = session.require_scene()
    what = args.get("what", "summary")
    system = scene.animation
    image = None
    try:
        if what == "summary":
            data = summary(scene)
        elif what == "curve":
            data = system.curve_report(_need(args, "target", None), _need(args, "path"),
                                       clip=args.get("clip"))  # fmt: skip
        elif what == "diagnose":
            data = system.diagnose()
        elif what == "trajectory":
            data = system.trajectory(_need(args, "target"), start=args.get("start"),
                                     end=args.get("end"), joint=args.get("joint"))  # fmt: skip
        elif what == "contacts":
            data = system.foot_contacts(_need(args, "target"), _need(args, "feet"),
                                        start=args.get("start"), end=args.get("end"))  # fmt: skip
        else:
            from ..rendering.png import encode_png

            paths = [args["path"]] if "path" in args else None
            if args.get("clip") is not None and paths is None:
                pixels = system.graph(clip=args["clip"], format="png", start=args.get("start"),
                                      end=args.get("end"), width=720)  # fmt: skip
            else:
                target = _need(args, "target")
                if not paths:
                    raise ToolError("INVALID_ARGUMENT", "graph needs path(s) or a clip")
                pixels = system.graph(
                    [(target, p) for p in paths],
                    format="png",
                    start=args.get("start"),
                    end=args.get("end"),
                    width=720,
                )
            image = encode_png(pixels)
            data = {"graph": {"width": int(pixels.shape[1]), "height": int(pixels.shape[0])}}
    except AetherError:
        raise  # (reported by the server as a structured error)
    return ToolOutput(
        {"what": what, **_jsonable(data), "scene": scene_status(session)}, image=image
    )


def _need(args: dict, key: str, default: Any = ...) -> Any:
    if key in args:
        return args[key]
    if default is not ...:
        return default
    raise ToolError("INVALID_ARGUMENT", f"this needs {key}")


def specs() -> tuple[Any, ...]:
    from .tools import ToolSpec, _schema

    text = {"type": "string"}
    return (
        ToolSpec(
            "inspect_animation",
            "Inspect the animation",
            "Measure the animation (read-only): summary; curve (exact: continuity, extrema, "
            "overshoot, velocity, jerk); diagnose; trajectory; contacts (foot sliding); graph "
            "(PNG). Author with run_script.",
            _schema(
                {
                    "what": {"enum": list(WHAT)},
                    "target": text,
                    "path": text,
                    "clip": text,
                    "joint": text,
                    "feet": {"type": "array", "items": text},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                }
            ),
            inspect_animation,
            read_only=True,
            idempotent=True,
        ),
    )
