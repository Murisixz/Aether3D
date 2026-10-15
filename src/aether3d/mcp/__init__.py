"""Aether3D as a Model Context Protocol (MCP) server: ``aether3d mcp serve``.

Optional: it needs the official MCP Python SDK (``pip install "aether3d[mcp]"``). Plain
``import aether3d`` never imports this package. See ``docs/MCP.md``.
"""

from __future__ import annotations

try:
    import mcp  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise ImportError(
        'the Aether3D MCP server needs the MCP SDK: pip install "aether3d[mcp]"'
    ) from exc

from .server import INSTRUCTIONS, Aether3DServer, serve
from .session import ServerConfig, Session, ToolError
from .tools import TOOLS
from .workspace import Workspace, WorkspaceError

__all__ = [
    "INSTRUCTIONS",
    "TOOLS",
    "Aether3DServer",
    "ServerConfig",
    "Session",
    "ToolError",
    "Workspace",
    "WorkspaceError",
    "serve",
]
