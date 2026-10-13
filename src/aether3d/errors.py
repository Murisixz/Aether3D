"""Exception hierarchy for Aether3D.

Every error raised deliberately by Aether3D derives from :class:`AetherError`,
so callers (human or AI agent) can catch engine errors with a single
``except AetherError`` while still distinguishing the specific cause.

Where it makes sense, errors also derive from the matching built-in exception
(``ValueError``, ``KeyError``) so that generic Python error handling keeps
working.
"""

from __future__ import annotations


class AetherError(Exception):
    """Base class for all Aether3D errors."""


class ValidationError(AetherError, ValueError):
    """A value passed to the API is malformed or out of range."""


class ObjectNotFoundError(AetherError, KeyError):
    """No object with the requested identifier or name exists in the scene."""

    def __str__(self) -> str:
        # KeyError quotes its message; keep it readable instead.
        return str(self.args[0]) if self.args else ""


class NameConflictError(AetherError, ValueError):
    """An object name is already used by another object in the same scene."""


class HierarchyError(AetherError, ValueError):
    """An operation would produce an invalid scene hierarchy (e.g. a cycle)."""


class DetachedObjectError(AetherError):
    """The object has been removed from its scene and can no longer be used."""


class ProjectFormatError(AetherError, ValueError):
    """A project file is malformed, unsupported, or internally inconsistent.

    ``location`` points at the offending element using a dotted path such as
    ``scene.objects[2].transform.scale`` (``None`` when not applicable).
    """

    def __init__(self, message: str, location: str | None = None) -> None:
        self.message = message
        self.location = location
        super().__init__(f"{location}: {message}" if location else message)


class AssetFormatError(AetherError, ValueError):
    """An external asset (e.g. a glTF/GLB file) is malformed or violates its specification.

    ``code`` is a stable machine-readable identifier (e.g. ``"ACCESSOR_OUT_OF_BOUNDS"``)
    and ``location`` a JSON pointer into the asset (e.g. ``/accessors/3``), or ``None``.
    """

    def __init__(self, message: str, location: str | None = None, code: str = "INVALID_ASSET"):
        self.message = message
        self.location = location
        self.code = code
        super().__init__(f"{location}: {message}" if location else message)


class UnsupportedFeatureError(AetherError, ValueError):
    """The requested conversion cannot represent some content faithfully.

    ``diagnostics`` lists every offending item (see :class:`aether3d.Diagnostic`).
    Conversions that may drop or approximate such content can be requested
    explicitly with ``allow_lossy=True``.
    """

    def __init__(self, message: str, diagnostics: list | None = None) -> None:
        self.message = message
        self.diagnostics = list(diagnostics or [])
        super().__init__(message)


class ResourceLimitError(AetherError, ValueError):
    """Input exceeds a configured safety limit (file size, vertex count, ...)."""


class EditError(AetherError, ValueError):
    """A mesh-editing operation cannot be applied; nothing was changed.

    ``code`` is a stable identifier (e.g. ``"NOT_MANIFOLD"``, ``"EMPTY_SELECTION"``),
    ``elements`` maps element kinds (``"vertices"``, ``"edges"``, ``"faces"``) to the
    offending ids, and ``hint`` suggests how to proceed.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        elements: dict[str, list[int]] | None = None,
        hint: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.elements = {k: sorted(v) for k, v in (elements or {}).items() if v}
        self.hint = hint
        text = f"[{code}] {message}"
        if self.elements:
            text += " (" + "; ".join(f"{k}: {v[:20]}" for k, v in self.elements.items()) + ")"
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "elements": {k: list(v) for k, v in self.elements.items()},
            "hint": self.hint,
        }
