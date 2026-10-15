"""Resources and prompts of the Aether3D MCP server.

Resources (read-only):

* ``aether3d://guide`` -- the scene plan guide for agents (Markdown).
* ``aether3d://schema/scene-plan`` -- the JSON Schema of scene plans.
* ``aether3d://examples/{name}`` -- example plans (``basics``, ``castle``).
* ``aether3d://scene/summary`` -- a compact summary of the open scene.
* ``aether3d://scene/objects/{ref}`` -- one object (by id or name), all fields.
* ``aether3d://render/last-preview`` -- the last preview image (PNG).

Prompts: ``build_scene``, ``inspect_and_fix`` and ``from_example``.
"""

from __future__ import annotations

import base64
import json
from importlib import resources as importlib_resources
from typing import Any
from urllib.parse import unquote

import mcp_types as types

from .session import Session, ToolError

EXAMPLES = ("basics", "castle")


def _package_text(name: str) -> str:
    return importlib_resources.files("aether3d.plans").joinpath(name).read_text(encoding="utf-8")


def guide() -> str:
    return _package_text("guide.md")


def example_plan(name: str) -> dict[str, Any]:
    if name not in EXAMPLES:
        raise ToolError("NOT_FOUND", f"no example plan {name!r}", hint=", ".join(EXAMPLES))
    return json.loads(_package_text(f"examples/{name}.json"))


def resources(session: Session) -> list[types.Resource]:
    out = [
        types.Resource(
            uri="aether3d://guide",
            name="guide",
            title="Scene plan guide",
            description="How to build and change scenes with one plan: format, patterns, "
            "prefabs, operations, queries, errors",
            mime_type="text/markdown",
        ),
        types.Resource(
            uri="aether3d://schema/scene-plan",
            name="scene-plan-schema",
            title="Scene plan JSON Schema",
            description="The JSON Schema (2020-12) of aether3d.scene-plan version 1",
            mime_type="application/schema+json",
        ),
        *(
            types.Resource(
                uri=f"aether3d://examples/{name}",
                name=f"example-{name}",
                title=f"Example plan: {name}",
                description={
                    "basics": "meshes, materials, patterns, a light, a camera, a query",
                    "castle": "a Minecraft-style castle of thousands of blocks, with prefabs",
                }[name],
                mime_type="application/json",
            )
            for name in EXAMPLES
        ),
    ]
    if session.scene is not None:
        out.append(
            types.Resource(
                uri="aether3d://scene/summary",
                name="scene-summary",
                title="Open scene",
                description="A compact summary of the open scene",
                mime_type="application/json",
            )
        )
    if session.last_preview is not None:
        out.append(
            types.Resource(
                uri="aether3d://render/last-preview",
                name="last-preview",
                title="Last preview",
                description="The last image render_preview made",
                mime_type="image/png",
                size=len(session.last_preview),
            )
        )
    return out


def resource_templates() -> list[types.ResourceTemplate]:
    return [
        types.ResourceTemplate(
            uri_template="aether3d://scene/objects/{ref}",
            name="scene-object",
            title="Object of the open scene",
            description="Every field of one object, by id or name",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            uri_template="aether3d://examples/{name}",
            name="example-plan",
            title="Example plan",
            description=f"An example scene plan: {', '.join(EXAMPLES)}",
            mime_type="application/json",
        ),
    ]


def read_resource(session: Session, uri: str) -> list[Any]:
    from ..plans import plan_schema
    from .tools import OBJECT_FIELDS, object_fields, summarize

    def text(value: str, mime: str) -> list[Any]:
        return [types.TextResourceContents(uri=uri, mime_type=mime, text=value)]

    def data(value: Any, mime: str = "application/json") -> list[Any]:
        return text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), mime)

    if uri == "aether3d://guide":
        return text(guide(), "text/markdown")
    if uri == "aether3d://schema/scene-plan":
        return data(plan_schema(), "application/schema+json")
    if uri.startswith("aether3d://examples/"):
        return data(example_plan(uri.removeprefix("aether3d://examples/")))
    if uri == "aether3d://scene/summary":
        return data(summarize(session))
    if uri.startswith("aether3d://scene/objects/"):
        scene = session.require_scene()
        ref = unquote(uri.removeprefix("aether3d://scene/objects/"))
        obj = scene._objects.get(ref) or scene._by_name.get(ref)
        if obj is None:
            raise ToolError("NOT_FOUND", f"no object with id or name {ref!r}")
        return data(object_fields(obj, OBJECT_FIELDS))
    if uri == "aether3d://render/last-preview":
        if session.last_preview is None:
            raise ToolError("NOT_FOUND", "no preview yet", hint="call render_preview")
        blob = base64.b64encode(session.last_preview).decode("ascii")
        return [types.BlobResourceContents(uri=uri, mime_type="image/png", blob=blob)]
    raise ToolError("NOT_FOUND", f"unknown resource {uri!r}")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

_PROMPTS = {
    "build_scene": types.Prompt(
        name="build_scene",
        title="Build a scene",
        description="Build a scene from a description with one scene plan, check it, fix "
        "problems, look at it and save it",
        arguments=[
            types.PromptArgument(name="description", description="what to build", required=True),
            types.PromptArgument(
                name="project",
                description="the .aether file to save to (default scene.aether)",
                required=False,
            ),
        ],
    ),
    "inspect_and_fix": types.Prompt(
        name="inspect_and_fix",
        title="Inspect and fix placement",
        description="Find objects that intersect, float or sink, or are out of view, and "
        "fix them with one plan of operations",
        arguments=[
            types.PromptArgument(
                name="focus", description="which objects or problem to look at", required=False
            )
        ],
    ),
    "from_example": types.Prompt(
        name="from_example",
        title="Start from an example plan",
        description="Adapt one of the example plans and build it",
        arguments=[
            types.PromptArgument(name="example", description=", ".join(EXAMPLES), required=True),
            types.PromptArgument(name="changes", description="what to change", required=False),
        ],
    ),
}


def prompts() -> list[types.Prompt]:
    return list(_PROMPTS.values())


def _user(text: str) -> types.PromptMessage:
    return types.PromptMessage(role="user", content=types.TextContent(text=text))


def get_prompt(name: str, arguments: dict[str, str]) -> types.GetPromptResult:
    if name not in _PROMPTS:
        raise KeyError(f"unknown prompt {name!r}")
    for argument in _PROMPTS[name].arguments or []:
        if argument.required and not arguments.get(argument.name):
            raise KeyError(f"prompt {name!r} needs the argument {argument.name!r}")
    if name == "build_scene":
        project = arguments.get("project") or "scene.aether"
        text = f"""Build this scene with Aether3D: {arguments["description"]}

Work efficiently -- a scene plan builds everything in one call:
1. Read the resource aether3d://guide (the scene plan format) if you have not.
2. Call scene_new with path "{project}" (or scene_open if it exists).
3. Write ONE scene plan for the whole scene: shared meshes and materials, prefabs for
   repeated structures, patterns (grid, array, circle, line, scatter) for repeated objects,
   a light and a camera, and queries that check the result (overlaps of things that must
   not intersect). Check it with check_scene_plan, then execute_scene_plan.
4. Read the query answers. Fix what is wrong with one more plan of operations (update,
   delete), not one call per object.
5. render_preview and look at the image; adjust the camera or the scene if needed.
6. scene_save. Report what you built (counts, problems fixed)."""
    elif name == "inspect_and_fix":
        focus = arguments.get("focus") or "the whole scene"
        text = f"""Inspect {focus} in the open Aether3D scene and fix placement problems.
1. scene_info for an overview; objects_list with a selector for details.
2. spatial_query: overlaps (objects that intersect but should not), relation (above,
   below, inside), raycast straight down (what each object stands on), in_view (what the
   camera sees).
3. Fix everything with ONE execute_scene_plan of operations (update with translate,
   position, rotation_euler or scale; delete with an expected count), then run the same
   queries again to confirm, and render_preview.
4. If a fix makes things worse, use history undo."""
    else:
        example = arguments["example"]
        plan = json.dumps(example_plan(example), indent=1)
        changes = arguments.get("changes") or "keep it as it is"
        text = f"""Here is the Aether3D example plan "{example}":

{plan}

Adapt it ({changes}), check it with check_scene_plan, build it with execute_scene_plan in
a new scene (scene_new), read the query answers, then render_preview."""
    return types.GetPromptResult(description=_PROMPTS[name].description, messages=[_user(text)])
