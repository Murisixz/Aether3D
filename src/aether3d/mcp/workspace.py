"""The workspace: the only directory tree an MCP session may read from or write to.

Every path a tool receives is untrusted. :meth:`Workspace.path` accepts a path only if it
stays inside the workspace root both textually (after normalizing ``..``) and after
resolving symbolic links, refuses anything that is not a regular file (devices, pipes,
directories), checks the file's suffix against what the tool may read or write, and
refuses to write through a symbolic link. Writes need an explicit ``overwrite`` to replace
an existing file, and none happen in read-only mode. Paths in messages are shown relative
to the root, so absolute paths of the machine do not leak into a conversation.
"""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

from ..errors import AetherError

MAX_READ_BYTES = 512 * 2**20
"""Largest file a session reads (projects, glTF files, plans, images)."""

# An absolute path (POSIX or Windows) that is not part of a word, a relative path, a URI
# ("aether3d://scene/...") or a JSON pointer inside one.
_ABSOLUTE_PATH = re.compile(
    r"(?<![\w~<>.:/\\-])(?:[A-Za-z]:[\\/]|/)(?:[^\s'\"<>|*?\\/]+[\\/])+[^\s'\"<>|*?\\/,;)]*"
)


def redact_paths(text: str) -> str:
    """``text`` with every absolute path replaced by ``<path>`` (for messages that this
    package did not write, such as unexpected exceptions)."""
    return _ABSOLUTE_PATH.sub("<path>", text)


PROJECT_SUFFIXES = (".aether",)
SCENE_SUFFIXES = (".aether", ".glb", ".gltf")
EXPORT_SUFFIXES = (".glb", ".gltf")
IMAGE_SUFFIXES = (".png", ".hdr", ".pfm")


class WorkspaceError(AetherError, ValueError):
    """A path or file operation the workspace policy refuses; nothing was read or written."""

    def __init__(self, code: str, message: str, hint: str | None = None) -> None:
        self.code = code
        self.message = message
        self.hint = hint
        super().__init__(f"[{code}] {message}" + (f". Hint: {hint}" if hint else ""))

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "hint": self.hint}


class Workspace:
    """A root directory and the rules for using paths inside it."""

    def __init__(self, root: str | os.PathLike[str], *, read_only: bool = False) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise WorkspaceError("WORKSPACE_INVALID", f"workspace {root!s} is not a directory")
        self.root = resolved
        self.read_only = read_only

    # ------------------------------------------------------------------ #

    def relative(self, path: Path) -> str:
        """``path`` as shown to clients: relative to the root, with forward slashes."""
        try:
            return path.resolve().relative_to(self.root).as_posix() or "."
        except ValueError:
            return "<outside the workspace>"

    def redact(self, text: str) -> str:
        """``text`` with paths in the workspace made relative to it, the workspace itself
        shown as ``<workspace>`` and the user's home folder as ``~``."""
        root = str(self.root)
        text = text.replace(root + os.sep, "").replace(root, "<workspace>")
        if os.sep != "/":
            text = text.replace(self.root.as_posix() + "/", "")
        home = str(Path.home())
        if len(home) > 1:
            text = text.replace(home, "~")
        return text

    def path(
        self,
        value: object,
        *,
        suffixes: tuple[str, ...],
        write: bool = False,
        must_exist: bool = True,
        overwrite: bool = False,
    ) -> Path:
        """A checked absolute path for ``value`` (see the module docstring)."""
        if not isinstance(value, str) or not value.strip():
            raise WorkspaceError("PATH_INVALID", "a path is a non-empty string")
        if len(value) > 1024 or any(c in value for c in "\0\n\r"):
            raise WorkspaceError("PATH_INVALID", "the path is too long or has control characters")
        raw = Path(value).expanduser() if value.startswith("~") else Path(value)
        if raw.expanduser() != raw:
            raise WorkspaceError("PATH_INVALID", "paths may not use ~")
        candidate = raw if raw.is_absolute() else self.root / raw
        normalized = Path(os.path.normpath(candidate))
        if not _inside(normalized, self.root):
            raise WorkspaceError(
                "PATH_OUTSIDE_WORKSPACE",
                f"{value!r} is outside the workspace",
                hint="use a path relative to the workspace root, without '..'",
            )
        resolved = normalized.resolve()
        if not _inside(resolved, self.root):
            raise WorkspaceError(
                "PATH_OUTSIDE_WORKSPACE",
                f"{value!r} leads outside the workspace through a symbolic link",
            )
        suffix = normalized.suffix.lower()
        if suffix not in suffixes:
            raise WorkspaceError(
                "SUFFIX_NOT_ALLOWED",
                f"{self.relative(normalized)!r} must end with {', '.join(suffixes)}",
            )
        exists = os.path.lexists(normalized)
        if write:
            if self.read_only:
                raise WorkspaceError(
                    "READ_ONLY",
                    "this server is read-only: it writes no files",
                    hint="restart it without --read-only to save or export",
                )
            if exists and os.path.islink(normalized):
                raise WorkspaceError(
                    "PATH_INVALID", f"{self.relative(normalized)!r} is a symbolic link"
                )
            if exists and not stat.S_ISREG(os.stat(normalized).st_mode):
                raise WorkspaceError("PATH_INVALID", f"{self.relative(normalized)!r} is not a file")
            if exists and not overwrite:
                raise WorkspaceError(
                    "FILE_EXISTS",
                    f"{self.relative(normalized)!r} exists",
                    hint='pass "overwrite": true to replace it',
                )
            parent = normalized.parent
            if not parent.is_dir():
                raise WorkspaceError(
                    "PATH_INVALID",
                    f"the folder of {self.relative(normalized)!r} does not exist",
                    hint="write to an existing folder of the workspace",
                )
            return normalized
        if not exists:
            if must_exist:
                raise WorkspaceError("NOT_FOUND", f"{self.relative(normalized)!r} does not exist")
            return normalized
        info = os.stat(resolved)
        if not stat.S_ISREG(info.st_mode):
            raise WorkspaceError("PATH_INVALID", f"{self.relative(normalized)!r} is not a file")
        if info.st_size > MAX_READ_BYTES:
            raise WorkspaceError(
                "FILE_TOO_LARGE",
                f"{self.relative(normalized)!r} is larger than {MAX_READ_BYTES} bytes",
            )
        return resolved


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
