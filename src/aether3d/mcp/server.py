"""The MCP server: Aether3D's tools, resources and prompts over stdio.

Built on the official MCP Python SDK's low-level ``Server`` (``mcp`` 2.x), so every tool
has a handwritten JSON Schema and the server controls validation, cancellation and
progress:

* **One tool at a time.** Calls are serialized (an ``anyio.Lock``); a tool runs in a
  worker thread so the connection stays responsive.
* **Validation.** Arguments are checked against the tool's schema (``jsonschema``,
  draft 2020-12) before the tool runs; problems come back as a tool error with JSON
  pointers, so the model can correct them.
* **Cancellation.** When the client cancels a request, the tool's cancel event is set;
  plans roll back at their next safe point, path-traced renders stop. The server waits
  for that before the request ends, so the scene is never left half-changed. (A
  cancelled request gets no response, as the protocol says.)
* **Progress.** Long plans and path-traced renders send ``notifications/progress`` when
  the client asked for it (a progress token).
* **Clean stdout.** stdout carries protocol messages only: the SDK points the process's
  file descriptor 1 at stderr while serving, and logs go to stderr.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import anyio
import jsonschema
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp_types.jsonrpc import INVALID_PARAMS

from .. import __version__
from ..errors import AetherError
from . import content
from .session import ServerConfig, Session, ToolError
from .tools import TOOL_INDEX, TOOLS, ToolContext, ToolOutput, encode_image, error_payload

log = logging.getLogger("aether3d.mcp")

INSTRUCTIONS = (
    "Aether3D builds, inspects and renders 3D scenes. Work in one open scene (scene_new or "
    "scene_open; scene_save writes it). Create and change objects with execute_scene_plan: "
    "one declarative plan builds thousands of objects in a single call and a single undo "
    "step (read aether3d://guide once). Check placement with spatial_query (overlaps, "
    "raycast, relation, describe) instead of guessing, look with render_preview, undo with "
    "history. Conventions: +Z up, meters, objects face -Y, rotation_euler in degrees."
)


class Aether3DServer:
    """An MCP server bound to one :class:`Session` (one workspace, one open scene)."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.session = Session(config)
        self._lock = anyio.Lock()
        self._validators = {
            spec.name: jsonschema.Draft202012Validator(spec.input_schema) for spec in TOOLS
        }
        self.server: Server[Any] = Server(
            "aether3d",
            version=__version__,
            title="Aether3D",
            description="An AI-native, code-first 3D engine: scene plans, spatial queries, "
            "rendering",
            instructions=INSTRUCTIONS,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
            on_list_resources=self._list_resources,
            on_list_resource_templates=self._list_resource_templates,
            on_read_resource=self._read_resource,
            on_list_prompts=self._list_prompts,
            on_get_prompt=self._get_prompt,
        )
        if config.project is not None:
            from .tools import scene_open

            scene_open(self.session, {"path": config.project}, ToolContext())

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    async def _list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name=spec.name,
                    title=spec.title,
                    description=spec.description,
                    input_schema=spec.input_schema,
                    annotations=types.ToolAnnotations(
                        title=spec.title,
                        read_only_hint=spec.read_only,
                        destructive_hint=spec.destructive,
                        idempotent_hint=spec.idempotent,
                        open_world_hint=False,
                    ),
                )
                for spec in TOOLS
            ]
        )

    async def _call_tool(
        self, ctx: Any, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        spec = TOOL_INDEX.get(params.name)
        if spec is None:
            raise MCPError(INVALID_PARAMS, f"unknown tool {params.name!r}")
        arguments = params.arguments or {}
        problems = sorted(
            self._validators[spec.name].iter_errors(arguments),
            key=lambda e: (len(e.absolute_path), list(map(str, e.absolute_path))),
        )
        if problems:
            issues = [
                {
                    "pointer": "/" + "/".join(str(p) for p in e.absolute_path),
                    "message": e.message[:300],
                }
                for e in problems[:20]
            ]
            return _error_result(
                {
                    "error": {
                        "type": "InvalidArguments",
                        "code": "INVALID_ARGUMENTS",
                        "message": f"{spec.name}: {issues[0]['message']}",
                        "pointer": issues[0]["pointer"],
                        "issues": issues,
                        "hint": "see the tool's input schema",
                    }
                }
            )
        async with self._lock:
            started = time.perf_counter()
            try:
                output = await self._run(spec, arguments, ctx)
            except anyio.get_cancelled_exc_class():
                log.info("%s cancelled after %.3f s", spec.name, time.perf_counter() - started)
                raise
            except (AetherError, ToolError) as exc:
                log.info("%s refused: %s", spec.name, exc)
                return _error_result(error_payload(exc, self.session))
            except Exception as exc:  # a bug: report it, keep serving
                log.exception("%s failed", spec.name)
                return _error_result(error_payload(exc, self.session))
            log.info("%s done in %.3f s", spec.name, time.perf_counter() - started)
        return _ok_result(output, self.session)

    async def _run(self, spec: Any, arguments: dict[str, Any], ctx: Any) -> ToolOutput:
        cancel = threading.Event()
        context = ToolContext(
            cancel=cancel,
            progress=self._progress_bridge(ctx),
            deadline=time.monotonic() + self.config.tool_timeout,
        )
        session = self.session
        outcome: dict[str, Any] = {}
        done = anyio.Event()

        def work() -> ToolOutput:
            with session.lock:
                return spec.run(session, arguments, context)

        async def worker() -> None:
            try:
                outcome["value"] = await anyio.to_thread.run_sync(work)
            except BaseException as exc:  # noqa: BLE001 - handed to the caller below
                outcome["error"] = exc
            finally:
                done.set()

        async with anyio.create_task_group() as tg:
            tg.start_soon(worker)
            try:
                await done.wait()
            except anyio.get_cancelled_exc_class():
                # The client cancelled: stop the tool at its next safe point (plans roll
                # back), and wait for that so the scene is consistent, then end the request.
                cancel.set()
                with anyio.CancelScope(shield=True):
                    await done.wait()
                raise
        error = outcome.get("error")
        if error is not None:
            if isinstance(error, anyio.get_cancelled_exc_class()):  # pragma: no cover
                raise error
            raise error
        return outcome["value"]

    @staticmethod
    def _progress_bridge(ctx: Any):
        """A thread-side callback sending ``notifications/progress`` (at most 5 per second).

        ``report_progress`` sends nothing unless the request carried a progress token.
        """
        meta = getattr(ctx, "meta", None) or {}
        if not (meta.get("progress_token") or meta.get("progressToken")):
            return None
        last = [0.0]

        def report(progress: float, total: float | None, message: str) -> None:
            now = time.monotonic()
            if now - last[0] < 0.2 and (total is None or progress < total):
                return
            last[0] = now
            try:
                anyio.from_thread.run(ctx.session.report_progress, progress, total, message)
            except Exception:  # the request may be gone (cancelled); progress is best effort
                log.debug("progress notification not sent", exc_info=True)

        return report

    # ------------------------------------------------------------------ #
    # Resources and prompts
    # ------------------------------------------------------------------ #

    async def _list_resources(self, ctx: Any, params: Any) -> types.ListResourcesResult:
        return types.ListResourcesResult(resources=content.resources(self.session))

    async def _list_resource_templates(
        self, ctx: Any, params: Any
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(resource_templates=content.resource_templates())

    async def _read_resource(
        self, ctx: Any, params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        async with self._lock:
            try:
                contents = await anyio.to_thread.run_sync(
                    content.read_resource, self.session, str(params.uri)
                )
            except (AetherError, ToolError) as exc:
                raise MCPError(INVALID_PARAMS, self.session.workspace.redact(str(exc))) from None
        return types.ReadResourceResult(contents=contents)

    async def _list_prompts(self, ctx: Any, params: Any) -> types.ListPromptsResult:
        return types.ListPromptsResult(prompts=content.prompts())

    async def _get_prompt(
        self, ctx: Any, params: types.GetPromptRequestParams
    ) -> types.GetPromptResult:
        try:
            return content.get_prompt(params.name, params.arguments or {})
        except KeyError as exc:
            raise MCPError(INVALID_PARAMS, str(exc.args[0])) from None

    # ------------------------------------------------------------------ #

    async def serve_stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


def _ok_result(output: ToolOutput, session: Session) -> types.CallToolResult:
    data = output.data
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)
    blocks: list[Any] = []
    if output.image is not None:
        blocks.append(types.ImageContent(data=encode_image(output.image), mime_type="image/png"))
    blocks.append(types.TextContent(text=text))
    return types.CallToolResult(content=blocks, structured_content=json.loads(text))


def _error_result(payload: dict[str, Any]) -> types.CallToolResult:
    text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str)
    return types.CallToolResult(
        content=[types.TextContent(text=text)], structured_content=payload, is_error=True
    )


def serve(config: ServerConfig) -> None:
    """Serve one client over stdin/stdout until it disconnects."""
    server = Aether3DServer(config)
    anyio.run(server.serve_stdio)
