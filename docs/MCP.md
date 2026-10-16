# The MCP server: Aether3D for AI agents

`aether3d mcp serve` makes Aether3D available to AI agents -- Claude Code and any other
[Model Context Protocol](https://modelcontextprotocol.io) client -- as tools, resources and
prompts. An agent builds a scene with [scene plans](SCENE_PLANS.md) (thousands of objects
in one call), checks placement with exact spatial queries, fixes mistakes, undoes and
redoes, looks at a quick GPU preview, saves the project and exports glTF, without writing
Python.

It runs locally over **stdio** (the client starts it as a subprocess and talks to it
through stdin and stdout). It opens no network port, makes no network requests, and uses
no files outside the folder it is given.

## Install and run

```bash
pip install "aether3d[mcp]"         # the official MCP Python SDK (MIT), mcp 2.x
aether3d mcp serve --workspace ~/scenes/castle
```

The server waits for a client on stdin; run it from a client rather than by hand. On Arch
Linux, see [INSTALL.md](INSTALL.md) (a virtual environment).

| Option | Meaning |
| --- | --- |
| `--workspace DIR` | the only folder tools may read or write (default: the current folder) |
| `--project PATH` | open this project (inside the workspace) at start |
| `--read-only` | never write a file (saving, exporting and image files are refused) |
| `--max-objects N` | objects one plan may create (default 100,000) |
| `--timeout S` | seconds a tool may run before it rolls back or stops (default 300) |
| `--no-path-tracing` | refuse path-traced renders |
| `--gpu-adapter SPEC` | the GPU for GPU renders: an index, part of a name, `discrete`, `vulkan`, ... |
| `--max-reply-kib N` | the largest reply of a read-only tool (default 256 KiB) |
| `--log-level LEVEL` | what to log on stderr: `debug`, `info`, `warning` (default), `error` |

Without the SDK installed, `aether3d mcp serve` exits with status 1 and says how to
install it; `import aether3d` never needs it.

## Claude Code

Add the server to a project (run this in the project's folder; `--scope local`, the
default, keeps it private to you in that folder):

```bash
claude mcp add aether3d -- aether3d mcp serve --workspace "$PWD"
claude mcp get aether3d        # Status: ✔ Connected
```

Use the absolute path of the `aether3d` command if it is not on the `PATH` Claude Code
sees (for example `~/aether3d/.venv/bin/aether3d` with a virtual environment), and an
absolute `--workspace` (a relative one is resolved in the folder the client starts the
server in). Then start `claude` in that folder and ask, for example:

> Build a Minecraft-style castle with four towers and a keep, check that nothing
> overlaps, show me a preview and export it to castle.glb.

Unless you allow its tools in advance, Claude Code asks for permission before it uses
them. `claude mcp remove aether3d` removes the server again.

To share the configuration with a team, `claude mcp add --scope project ...` writes a
`.mcp.json` in the folder (each user approves it when Claude Code starts):

```json
{
  "mcpServers": {
    "aether3d": {
      "type": "stdio",
      "command": "aether3d",
      "args": ["mcp", "serve", "--workspace", "/home/me/scenes/castle"],
      "env": {}
    }
  }
}
```

Other MCP clients take the same command and arguments.

These steps were checked with Claude Code 2.1.300 on Arch Linux: `claude mcp add` with
local scope, then `claude mcp get aether3d` reporting `✔ Connected`.

## Tools

Every tool takes JSON arguments, checked against a strict schema (unknown keys are
errors), and returns structured JSON -- also as text, for clients that read only text. A
tool that cannot do what was asked changes nothing and returns an error result:

```json
{"error": {"type": "WorkspaceError", "code": "PATH_OUTSIDE_WORKSPACE",
           "message": "'../x.aether' is outside the workspace",
           "hint": "use a path relative to the workspace root, without '..'"}}
```

Arguments that do not fit the schema give `INVALID_ARGUMENTS` with a JSON pointer per
problem; a plan with problems gives every problem, as in [SCENE_PLANS.md](SCENE_PLANS.md#errors).

| Tool | Does |
| --- | --- |
| `scene_new` | start an empty scene (`name`; `path`: the project `scene_save` writes; refuses to drop unsaved changes unless `discard_unsaved`) |
| `scene_open` | open a `.aether` project (with its history file) or import a `.glb`/`.gltf`; returns a summary |
| `scene_save` | save the project atomically (`path`, `overwrite`, `history`) |
| `scene_info` | a compact summary: counts by kind, tags, roots, cameras, lights, materials, bounds, history, revision (`detail: "full"` adds the world, settings, procedural parameters, the GPU) |
| `objects_list` | objects (all, or a [selector](SCENE_PLANS.md#selectors)) with chosen `fields`, paginated (`limit`, `cursor`) |
| `execute_scene_plan` | build or change the scene with one [plan](SCENE_PLANS.md): one undo step, all or nothing (`mode`, `expected_revision`, `timeout`, `max_results`) |
| `check_scene_plan` | check a plan without changing anything (`dry_run: true` applies and rolls back, reporting exactly what it would make) |
| `spatial_query` | one `request` or a batch of `requests` ([SPATIAL.md](SPATIAL.md)); object lists may be selectors; answers keep at most `max_results` entries (default 100) |
| `render_preview` | a quick image, returned as image content: the GPU rasterizer when it works, the CPU rasterizer otherwise (`backend`: `auto`, `gpu`, `cpu`; `width`, `height`, `camera`, `frame`, `shadows`, `save_to`). Only the image is read back from the GPU, and it is encoded for speed (zlib level 1) |
| `render_image` | an image file in the workspace (`.png`, linear `.hdr`/`.pfm`); `backend: "path"` or `"gpu_path"` path traces, within the server's sample budget |
| `history` | `list`, `undo`, `redo` (`steps`), `checkpoint`, `goto`, `remove_checkpoint` (`name`) |
| `export_scene` | a `.glb` or `.gltf` file (`allow_lossy`, `bake_textures`, `overwrite`) |
| `run_script` | one of the engine's JSON scripts, atomically: edit scripts (polygon editing), procedural scripts, scene scripts |
| `diagnostics` | checks with stable codes: `geometry`, `procedural`, `materials`, `history` |
| `capabilities` | what the engine offers, by `topic`: `overview`, `plan`, `primitives`, `modifiers`, `materials`, `queries`, `limits`, `gpu` |

Notes for agents (the server also says this in its instructions):

- **Build with plans.** `execute_scene_plan` is the way to create and change objects:
  one call, one undo step, thousands of objects. Read `aether3d://guide` once.
- **Use the revision.** Every reply that reads the scene gives its `revision`; pass it as
  `expected_revision` to a plan that depends on what you read.
- **Retry safely.** A plan with an `idempotency_key` applies once, even when resent after
  a lost reply.
- **Check, don't guess.** `spatial_query` answers exactly (`overlaps`, `raycast`,
  `relation`, `bounds`, `line_of_sight`, ...) without changing anything.
- **Previews are cheap, path tracing is not.** `render_preview` never path traces; the
  reply says which renderer drew the image (`backend`, `device`, `adapter`, and
  `fallback` when the GPU could not be used): a CPU image is never reported as a GPU one.
- **Replies stay small.** A long query answer (thousands of overlapping pairs, a
  `describe` of a large scene) keeps its first `max_results` entries (default 100) and
  says so: `"truncated": true` and the full length as `total`. A read-only tool whose
  reply would exceed `--max-reply-kib` (default 256 KiB) returns `RESULT_TOO_LARGE` with
  a hint to ask for less; replies to changes are never refused.

### Long calls: progress and cancellation

Plans and path-traced renders report progress to clients that ask for it (a
`progressToken`). A client can cancel a call: a plan stops at its next safe point and
**rolls back** before the server answers, so the scene is never half-changed; a render
stops and writes nothing. One tool runs at a time; a call past `--timeout` stops the same
way (`TIMEOUT`).

## Resources and prompts

| Resource | Content |
| --- | --- |
| `aether3d://guide` | the scene plan guide for agents (Markdown) |
| `aether3d://schema/scene-plan` | the plan format's JSON Schema |
| `aether3d://examples/basics`, `.../castle`, `.../village` | example plans: a courtyard; a Minecraft-style castle (2,190 objects); a village of 11,270 objects from reusable structures |
| `aether3d://scene/summary` | the open scene's summary |
| `aether3d://scene/objects/{ref}` | one object, by id or name |
| `aether3d://render/last-preview` | the last preview (PNG) |

Prompts: `build_scene` (build from a description, check, fix, preview, save),
`inspect_and_fix` (find intersections, floating or sunken objects, objects out of view, and
fix them), `from_example` (adapt an example plan).

## Security

The server treats every argument, plan, path and imported file as untrusted.

- **The workspace is the boundary.** Every path is resolved inside `--workspace` both as
  text (`..` cannot leave it) and after following symbolic links (a link pointing outside is
  refused). Only regular files are read (no devices or pipes), up to 512 MiB, with the
  suffixes a tool expects (`.aether`; `.glb`/`.gltf`; `.png`/`.hdr`/`.pfm`). Writes never go
  through a symbolic link, need an existing parent folder, and replace an existing file
  only with `"overwrite": true`. `scene_save` without `overwrite` replaces only the
  project this session opened or saved, and only if no one changed it on disk since
  (`PROJECT_CHANGED_ON_DISK`). `--read-only` refuses every write.
- **No code execution.** Plans and scripts are declarative data. Expressions use the
  procedural engine's whitelisted evaluator; nothing is passed to `eval`, `exec`, a shell
  or a compiler; shaders are Aether3D's own, never compiled from arguments.
- **Bounded work.** Plans have [limits](SCENE_PLANS.md#limits) (`--max-objects`); images
  are bounded in pixels (previews at most 1920 x 1080 pixels' worth, images 3840 x 2160)
  and path tracing in samples (`width x height x spp` at most 400 million;
  `--no-path-tracing` refuses it); replies are bounded too (above); every tool has a
  timeout.
- **Private.** stdio only: no port is opened and no network request is made. Messages show
  workspace-relative paths, never the machine's absolute paths; errors say what was wrong
  without internal details.
- **One client.** A stdio server serves the client that started it.

## Examples and measurements

- [`examples/mcp_client.py`](../examples/mcp_client.py): a minimal client with the SDK.
- [`examples/mcp_castle_agent.py`](../examples/mcp_castle_agent.py): an agent's whole loop
  as tool calls -- build a castle with two deliberate mistakes, find them with spatial
  queries, fix them, undo and redo, preview, save, export, reopen.

On the machine of [the benchmark](SCENE_PLANS.md#performance) (i9-12900H, Python 3.14), a
trivial tool call takes about 1.2 ms over stdio, and a plan of 10,000 objects about 9 ms
more as a tool call than in the server itself (JSON-RPC and the reply, which stays under
1 KB: ids are listed only for small symbols). Creating 10,000 objects with one tool call
each takes 22 s; with one plan, 0.10 s.

## Protocol

The server uses the official MCP Python SDK's low-level server. It speaks both protocol
generations the SDK supports: the initialize handshake (up to `2025-11-25`) and the
per-request version of `2026-07-28`. stdout carries protocol messages only; logs go to
stderr.
