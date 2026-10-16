"""One MCP session: the workspace, the open scene and its project file.

The lifecycle is explicit. A session starts with no scene (or with ``--project``);
``scene_new`` and ``scene_open`` replace it (refusing to drop unsaved changes unless
asked), ``scene_save`` writes it. The session remembers the project file it was opened
from or saved to, and that file's size, time and SHA-256 at that moment: saving over a
project that something else changed in the meantime needs an explicit ``overwrite``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import AetherError
from ..plans import PlanLimits
from .workspace import Workspace

if TYPE_CHECKING:
    from ..scene import Scene


class ToolError(AetherError, ValueError):
    """A tool call that cannot be done as asked; it changed nothing."""

    def __init__(
        self, code: str, message: str, hint: str | None = None, details: dict | None = None
    ) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        self.details = dict(details or {})
        super().__init__(f"[{code}] {message}" + (f". Hint: {hint}" if hint else ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "details": self.details,
        }


@dataclass(frozen=True)
class ServerConfig:
    """How an Aether3D MCP server may act (``aether3d mcp serve`` options)."""

    workspace: Path
    read_only: bool = False
    project: str | None = None
    plan_limits: PlanLimits = field(default_factory=PlanLimits)
    tool_timeout: float = 300.0
    """Seconds a tool may run before it rolls back (plans) or stops (renders)."""
    max_preview_pixels: int = 1920 * 1080
    max_render_pixels: int = 3840 * 2160
    max_path_samples: int = 400_000_000
    """Budget of path-traced samples (width x height x samples per pixel) of one render."""
    allow_path_tracing: bool = True
    gpu_adapter: str | None = None
    """GPU adapter selection for GPU renders (as ``render(adapter=...)``; default: the best)."""
    max_reply_bytes: int = 256 * 1024
    """Largest JSON reply of a read-only tool; a larger one is refused (RESULT_TOO_LARGE)
    with a hint to ask for less, rather than flooding the client's context."""
    feedback_backend: str = "auto"
    """How the visual-feedback tools render observations: ``auto`` (the GPU rasterizer when
    one works), ``gpu`` or ``cpu``."""


@dataclass
class _DiskState:
    size: int
    mtime_ns: int
    sha256: str


def _disk_state(path: Path) -> _DiskState | None:
    try:
        info = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    return _DiskState(info.st_size, info.st_mtime_ns, digest)


class Session:
    """The state of one connection (a stdio server serves one client)."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.workspace = Workspace(config.workspace, read_only=config.read_only)
        self.scene: Scene | None = None
        self.project: Path | None = None  # where scene_save writes by default
        self.source: str | None = None  # what the scene came from (shown to clients)
        self.saved_revision: int | None = None
        self.disk: _DiskState | None = None
        self.last_preview: bytes | None = None
        self.notes: list[str] = []
        self.started = time.time()
        self.lock = threading.Lock()  # one tool at a time touches the scene
        self._gpu: dict[str, Any] | None = None
        self.feedback: Any = None  # the open scene's FeedbackSession (Stage 12 tools)

    # -- the scene ------------------------------------------------------------

    def require_scene(self) -> Scene:
        if self.scene is None:
            raise ToolError(
                "NO_SCENE",
                "no scene is open",
                hint="call scene_new to start one or scene_open to load a project",
            )
        return self.scene

    @property
    def dirty(self) -> bool:
        """Whether the scene changed since it was opened or saved (by revision)."""
        if self.scene is None:
            return False
        return self.saved_revision is None or self.scene.revision != self.saved_revision

    def check_unsaved(self, discard: bool) -> None:
        if self.scene is not None and self.dirty and len(self.scene) and not discard:
            raise ToolError(
                "UNSAVED_CHANGES",
                f"the open scene {self.scene.name!r} has unsaved changes",
                hint='save it with scene_save first, or pass "discard_unsaved": true',
            )

    def bind(self, scene: Scene, project: Path | None, source: str | None, saved: bool) -> None:
        self.scene = scene
        self.project = project
        self.source = source
        self.saved_revision = scene.revision if saved else None
        self.disk = _disk_state(project) if project is not None and saved else None
        self.last_preview = None
        self.feedback = None

    def mark_saved(self, path: Path) -> None:
        self.project = path
        self.source = self.workspace.relative(path)
        self.saved_revision = self.scene.revision if self.scene is not None else None
        self.disk = _disk_state(path)

    def project_changed_on_disk(self) -> bool:
        """Whether the project file is not what this session last read or wrote."""
        if self.project is None or self.disk is None or not self.project.exists():
            return False
        current = _disk_state(self.project)
        return current is None or current.sha256 != self.disk.sha256

    def base_dir(self) -> Path:
        """Where relative texture images of plans and scripts resolve."""
        if self.project is not None:
            return self.project.parent
        return self.workspace.root

    # -- GPU status (probed once: creating a device takes about half a second) ----

    def gpu(self) -> dict[str, Any]:
        if self._gpu is None:
            try:
                from ..rendering.gpu import gpu_status

                status = gpu_status(adapter=self.config.gpu_adapter)
            except Exception as exc:  # wgpu missing or broken: report, never fail
                status = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
            selected = status.get("selected") or {}
            adapter = selected.get("adapter") or {}
            self._gpu = {
                "available": bool(status.get("available")),
                "reason": status.get("reason"),
                "adapter": adapter.get("name"),
                "backend": adapter.get("backend"),
                "software": bool(adapter.get("software")) if adapter else None,
            }
        return self._gpu
