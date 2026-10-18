"""Resources and prompts of the Aether3D MCP server.

Resources (read-only):

* ``aether3d://guide`` -- the scene plan guide for agents (Markdown).
* ``aether3d://schema/scene-plan`` -- the JSON Schema of scene plans.
* ``aether3d://examples/{name}`` -- example plans (``basics``, ``castle``, ``village``).
* ``aether3d://scene/summary`` -- a compact summary of the open scene.
* ``aether3d://scene/objects/{ref}`` -- one object (by id or name), all fields.
* ``aether3d://render/last-preview`` -- the last preview image (PNG).
* ``aether3d://feedback/guide`` -- the visual feedback guide for agents (Markdown).
* ``aether3d://sculpt/guide`` -- the sculpting guide for agents (Markdown).
* ``aether3d://schema/sculpt-script`` -- the JSON Schema of sculpt scripts.
* ``aether3d://texturing/guide`` -- the texturing guide for agents (Markdown).
* ``aether3d://schema/texture-script`` -- the JSON Schema of texture scripts.
* ``aether3d://cad/guide`` -- the precision modeling guide for agents (Markdown).
* ``aether3d://schema/cad-script`` -- the JSON Schema of CAD scripts.
* ``aether3d://animation/guide`` -- the animation guide for agents (Markdown).
* ``aether3d://schema/animation-script`` -- the JSON Schema of animation scripts.
* ``aether3d://worlds/guide`` -- the procedural worlds guide for agents (Markdown).
* ``aether3d://schema/world-script`` -- the JSON Schema of world scripts.
* ``aether3d://schema/expectations``, ``.../vision-evaluation``, ``.../corrections`` -- the
  JSON Schemas of the feedback documents.
* ``aether3d://observations/{id}`` -- an observation's measured facts (JSON), and
  ``aether3d://observations/{id}/{kind}`` its images (``image``, ``annotated``, ``ids``,
  ``depth``; PNG).

Prompts: ``build_scene``, ``inspect_and_fix``, ``from_example`` and ``visual_feedback``.
"""

from __future__ import annotations

import base64
import json
from importlib import resources as importlib_resources
from typing import Any
from urllib.parse import unquote

import mcp_types as types

from .session import Session, ToolError

EXAMPLES = ("basics", "castle", "village")


def _package_text(name: str) -> str:
    return importlib_resources.files("aether3d.plans").joinpath(name).read_text(encoding="utf-8")


def guide() -> str:
    return _package_text("guide.md")


def sculpt_guide() -> str:
    return (
        importlib_resources.files("aether3d.sculpt")
        .joinpath("guide.md")
        .read_text(encoding="utf-8")
    )


def texturing_guide() -> str:
    return (
        importlib_resources.files("aether3d.texturing")
        .joinpath("guide.md")
        .read_text(encoding="utf-8")
    )


def cad_guide() -> str:
    return (
        importlib_resources.files("aether3d.cad").joinpath("guide.md").read_text(encoding="utf-8")
    )


def animation_guide() -> str:
    return (
        importlib_resources.files("aether3d.animation")
        .joinpath("guide.md")
        .read_text(encoding="utf-8")
    )


def worlds_guide() -> str:
    return (
        importlib_resources.files("aether3d.worlds")
        .joinpath("guide.md")
        .read_text(encoding="utf-8")
    )


def feedback_guide() -> str:
    return (
        importlib_resources.files("aether3d.feedback")
        .joinpath("guide.md")
        .read_text(encoding="utf-8")
    )


FEEDBACK_SCHEMAS = {
    "aether3d://schema/expectations": ("expectations", "Expectations JSON Schema"),
    "aether3d://schema/vision-evaluation": ("vision-evaluation", "Vision evaluation JSON Schema"),
    "aether3d://schema/corrections": ("corrections", "Correction actions JSON Schema"),
}


def feedback_schema(uri: str) -> dict[str, Any]:
    from ..feedback import schema

    return {
        "aether3d://schema/expectations": schema.expectations_schema,
        "aether3d://schema/vision-evaluation": schema.evaluation_schema,
        "aether3d://schema/corrections": schema.actions_schema,
    }[uri]()


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
                    "village": "a village of about 11,000 objects from reusable structures",
                }[name],
                mime_type="application/json",
            )
            for name in EXAMPLES
        ),
    ]
    out.append(
        types.Resource(
            uri="aether3d://feedback/guide",
            name="feedback-guide",
            title="Visual feedback guide",
            description="Observe, diagnose against expectations, evaluate images, propose, "
            "apply and verify corrections; the loop",
            mime_type="text/markdown",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://sculpt/guide",
            name="sculpt-guide",
            title="Sculpting guide",
            description="Brushes, strokes, masks, symmetry, dynamic topology, remeshing, "
            "validity tiers and sculpt scripts",
            mime_type="text/markdown",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://schema/sculpt-script",
            name="sculpt-script-schema",
            title="Sculpt script JSON Schema",
            description="The JSON Schema (2020-12) of aether3d.sculpt-script version 1",
            mime_type="application/schema+json",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://texturing/guide",
            name="texturing-guide",
            title="Texturing guide",
            description="UV unwrapping and checks, painted textures and layers, strokes in "
            "surface, view and UV space, fills, baking, texture transfer and texture scripts",
            mime_type="text/markdown",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://schema/texture-script",
            name="texture-script-schema",
            title="Texture script JSON Schema",
            description="The JSON Schema (2020-12) of aether3d.texture-script version 1",
            mime_type="application/schema+json",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://cad/guide",
            name="cad-guide",
            title="Precision modeling guide",
            description="Units, parametric sketches, part features, persistent names, "
            "buildings, drawings, measurements, exchange formats and CAD scripts",
            mime_type="text/markdown",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://schema/cad-script",
            name="cad-script-schema",
            title="CAD script JSON Schema",
            description="The JSON Schema (2020-12) of aether3d.cad-script version 1",
            mime_type="application/schema+json",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://animation/guide",
            name="animation-guide",
            title="Animation guide",
            description="Clips, curves and keys, NLA, rigs, skins, IK, constraints, motion "
            "generators, cameras, reports and animation scripts",
            mime_type="text/markdown",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://schema/animation-script",
            name="animation-script-schema",
            title="Animation script JSON Schema",
            description="The JSON Schema (2020-12) of aether3d.animation-script version 1",
            mime_type="application/schema+json",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://worlds/guide",
            name="worlds-guide",
            title="Procedural worlds guide",
            description="Worlds, presets, features, structures, block edits, queries and "
            "world scripts",
            mime_type="text/markdown",
        )
    )
    out.append(
        types.Resource(
            uri="aether3d://schema/world-script",
            name="world-script-schema",
            title="World script JSON Schema",
            description="The JSON Schema (2020-12) of aether3d.world-script version 1",
            mime_type="application/schema+json",
        )
    )
    for uri, (name, title) in FEEDBACK_SCHEMAS.items():
        out.append(
            types.Resource(
                uri=uri,
                name=f"{name}-schema",
                title=title,
                description=f"The JSON Schema (2020-12) of {name.replace('-', ' ')} documents",
                mime_type="application/schema+json",
            )
        )
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
        types.ResourceTemplate(
            uri_template="aether3d://observations/{id}",
            name="observation",
            title="An observation",
            description="What an observation measured (JSON), by its id",
            mime_type="application/json",
        ),
        types.ResourceTemplate(
            uri_template="aether3d://observations/{id}/{kind}",
            name="observation-image",
            title="An observation's image",
            description="image, annotated (numbered markers), ids (object colors) or depth (PNG)",
            mime_type="image/png",
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
    if uri == "aether3d://feedback/guide":
        return text(feedback_guide(), "text/markdown")
    if uri == "aether3d://sculpt/guide":
        return text(sculpt_guide(), "text/markdown")
    if uri == "aether3d://schema/sculpt-script":
        from ..sculpt.schema import sculpt_script_schema

        return data(sculpt_script_schema(), "application/schema+json")
    if uri == "aether3d://texturing/guide":
        return text(texturing_guide(), "text/markdown")
    if uri == "aether3d://schema/texture-script":
        from ..texturing.schema import texture_script_schema

        return data(texture_script_schema(), "application/schema+json")
    if uri == "aether3d://cad/guide":
        return text(cad_guide(), "text/markdown")
    if uri == "aether3d://schema/cad-script":
        from ..cad.schema import cad_script_schema

        return data(cad_script_schema(), "application/schema+json")
    if uri == "aether3d://animation/guide":
        return text(animation_guide(), "text/markdown")
    if uri == "aether3d://schema/animation-script":
        from ..animation.schema import animation_script_schema

        return data(animation_script_schema(), "application/schema+json")
    if uri == "aether3d://worlds/guide":
        return text(worlds_guide(), "text/markdown")
    if uri == "aether3d://schema/world-script":
        from ..worlds.script import schema as world_script_schema

        return data(world_script_schema(), "application/schema+json")
    if uri in FEEDBACK_SCHEMAS:
        return data(feedback_schema(uri), "application/schema+json")
    if uri.startswith("aether3d://observations/"):
        from .feedback import IMAGE_KINDS

        parts = uri.removeprefix("aether3d://observations/").split("/")
        fs = session.feedback
        if fs is None or not parts[0]:
            raise ToolError("NOT_FOUND", "no observation yet", hint="call observe_scene")
        obs = fs.store.get(parts[0])
        if len(parts) == 1:
            return data(obs.to_dict(max_objects=100))
        if len(parts) != 2 or parts[1] not in IMAGE_KINDS:
            raise ToolError(
                "NOT_FOUND", f"unknown resource {uri!r}", hint=f"kinds: {', '.join(IMAGE_KINDS)}"
            )
        png = obs.png(parts[1])
        if parts[1] in ("image", "annotated"):
            obs.options["delivered"] = True
        blob = base64.b64encode(png).decode("ascii")
        return [types.BlobResourceContents(uri=uri, mime_type="image/png", blob=blob)]
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
    "visual_feedback": types.Prompt(
        name="visual_feedback",
        title="Check the render and correct it",
        description="Observe the rendered scene, state expectations, evaluate the images, "
        "and correct what is wrong with verified, undoable corrections",
        arguments=[
            types.PromptArgument(
                name="goal", description="what the scene should show and how", required=True
            )
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
    elif name == "visual_feedback":
        text = f"""Make the open Aether3D scene match this goal: {arguments["goal"]}

Work in a closed loop (read aether3d://feedback/guide once):
1. observe_scene with images ["image", "annotated"] and look at the images yourself.
2. Write expectations for what is checkable: objects that must exist, what rests on what,
   what must not intersect, what the camera must show (visible, framing), the exposure.
   Call diagnose_scene with them, and with your own visual findings as an
   aether3d.vision-evaluation of that observation (provenance.image_input: true only
   because you looked; regions in pixels of that image; object ids from the markers).
3. propose_corrections; read each proposal's safety, rationale and compiled plan.
   preview: true shows the result without changing anything.
4. apply_corrections with expected_revision (safe and verify proposals; approve
   approval ones only if they match the goal). Each is one undo step, verified, and
   undone automatically if it does not help.
5. observe_scene again and look. Repeat, or run feedback_loop for the mechanical part.
Report what you changed, what the measurements say now, and what is still a judgment."""
    else:
        example = arguments["example"]
        plan = json.dumps(example_plan(example), indent=1)
        changes = arguments.get("changes") or "keep it as it is"
        text = f"""Here is the Aether3D example plan "{example}":

{plan}

Adapt it ({changes}), check it with check_scene_plan, build it with execute_scene_plan in
a new scene (scene_new), read the query answers, then render_preview."""
    return types.GetPromptResult(description=_PROMPTS[name].description, messages=[_user(text)])
