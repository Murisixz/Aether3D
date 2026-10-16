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


class SculptError(EditError):
    """A sculpting operation (a stroke, a mask change, a remesh) cannot be applied;
    nothing was changed.

    Like :class:`EditError` (which it extends): a stable ``code``, the offending
    ``elements`` (``"vertices"``, ``"triangles"``) and a ``hint``.
    """


class RenderError(AetherError, ValueError):
    """A camera, light, appearance, render setting or render request is invalid, or rendering
    failed; nothing was changed and no file was written.

    ``code`` is a stable identifier (e.g. ``"NO_CAMERA"``, ``"INVALID_CAMERA"``,
    ``"RESOURCE_LIMIT"``), ``location`` names the offending element (e.g.
    ``objects[Camera].camera.fov``), ``hint`` suggests how to proceed and
    ``details`` carries structured context.
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = dict(details or {})
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            "details": self.details,
        }


class MaterialError(AetherError, ValueError):
    """A material, texture, procedural texture or image is invalid, missing or unsafe;
    nothing was changed and no file was written.

    ``code`` is a stable identifier (e.g. ``"INVALID_MATERIAL"``, ``"MATERIAL_NOT_FOUND"``,
    ``"INVALID_IMAGE"``, ``"IMAGE_TOO_LARGE"``, ``"UNSAFE_PATH"``), ``location`` names the
    offending element (e.g. ``materials[Stone].roughness``), ``hint`` suggests how to
    proceed and ``details`` carries structured context.
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = dict(details or {})
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            "details": self.details,
        }


class HistoryError(AetherError, ValueError):
    """An undo, redo, checkpoint or history operation cannot be performed; nothing changed.

    ``code`` is a stable identifier (e.g. ``"NOTHING_TO_UNDO"``, ``"TRANSACTION_OPEN"``,
    ``"UNKNOWN_CHECKPOINT"``, ``"HISTORY_MISMATCH"``), ``location`` names the offending
    element, ``hint`` suggests how to proceed and ``details`` carries structured context.
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = dict(details or {})
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            "details": self.details,
        }


class QueryError(AetherError, ValueError):
    """A spatial query is malformed or cannot be answered; nothing changed.

    ``code`` is a stable identifier (e.g. ``"INVALID_ARGUMENT"``, ``"NO_GEOMETRY"``,
    ``"NO_CAMERA"``, ``"GEOMETRY_EVALUATION_FAILED"``, ``"INVALID_REQUEST"``),
    ``location`` names the offending argument, ``hint`` suggests how to proceed and
    ``details`` carries structured context.
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = dict(details or {})
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            "details": self.details,
        }


class PlanError(AetherError, ValueError):
    """A scene plan is invalid, or executing it failed; the scene is unchanged.

    ``code`` is a stable identifier (e.g. ``"INVALID_VALUE"``, ``"UNKNOWN_REFERENCE"``,
    ``"LIMIT_EXCEEDED"``, ``"NAME_CONFLICT"``, ``"REVISION_MISMATCH"``, ``"CANCELLED"``,
    ``"OPERATION_FAILED"``; see ``docs/SCENE_PLANS.md``). ``pointer`` is the JSON pointer
    of the offending part of the plan (``/objects/towers/grid/count``), ``operation`` the
    index of the failing entry of ``operations`` (or ``None``), ``hint`` an actionable
    suggestion and ``details`` structured context. ``issues`` lists every problem found
    when a plan is checked as a whole (this error is the first of them).
    """

    def __init__(
        self,
        code: str,
        message: str,
        pointer: str | None = None,
        *,
        operation: int | None = None,
        hint: str | None = None,
        details: dict | None = None,
        issues: list[dict] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.pointer = pointer
        self.operation = operation
        self.hint = hint
        self.details = dict(details or {})
        self.issues = list(issues or [])
        text = f"[{code}] " + (f"{pointer}: " if pointer else "") + message
        if hint:
            text += f". Hint: {hint}"
        if len(self.issues) > 1:
            text += f" (and {len(self.issues) - 1} more problem(s))"
        super().__init__(text)

    @property
    def location(self) -> str | None:
        """The JSON pointer (the name other Aether3D errors use for where they happened)."""
        return self.pointer

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "pointer": self.pointer,
            "operation": self.operation,
            "hint": self.hint,
            "details": self.details,
            "issues": self.issues,
        }


class TexturingError(AetherError, ValueError):
    """A UV, texture-painting or baking operation cannot be carried out; nothing was changed
    and no file was written.

    ``code`` is a stable identifier (e.g. ``"NO_UVS"``, ``"CHART_NOT_A_DISK"``,
    ``"ATLAS_FULL"``, ``"INVALID_STROKE"``, ``"BAKE_TOO_LARGE"``), ``location`` names the
    offending element or argument, ``hint`` suggests how to proceed and ``details`` carries
    structured context (counts, limits).
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = dict(details or {})
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            "details": self.details,
        }


class FeedbackError(AetherError, ValueError):
    """A visual-feedback request cannot be carried out; the scene is unchanged.

    Raised for observations (views, budgets), expectations documents, vision evaluations,
    correction proposals and feedback loops (see ``docs/FEEDBACK.md``). ``code`` is a stable
    identifier (e.g. ``"INVALID_VIEW"``, ``"INVALID_DOCUMENT"``, ``"STALE_OBSERVATION"``,
    ``"REVISION_MISMATCH"``, ``"NOT_ALLOWED"``, ``"UNSUPPORTED_ACTION"``), ``location`` a
    JSON pointer into the offending document (``/checks/2/objects``) or the name of the
    offending argument, ``hint`` an actionable suggestion and ``details`` structured
    context. ``issues`` lists every problem found when a document is checked as a whole
    (this error is the first of them).
    """

    def __init__(
        self,
        code: str,
        message: str,
        location: str | None = None,
        *,
        hint: str | None = None,
        details: dict | None = None,
        issues: list[dict] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.location = location
        self.hint = hint
        self.details = dict(details or {})
        self.issues = list(issues or [])
        text = f"[{code}] " + (f"{location}: " if location else "") + message
        if hint:
            text += f". Hint: {hint}"
        if len(self.issues) > 1:
            text += f" (and {len(self.issues) - 1} more problem(s))"
        super().__init__(text)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "hint": self.hint,
            "details": self.details,
            "issues": self.issues,
        }
