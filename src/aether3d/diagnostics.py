"""Structured findings reported by inspection, import and export."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ERROR = "error"
WARNING = "warning"
INFO = "info"
SEVERITIES = (ERROR, WARNING, INFO)


@dataclass(frozen=True)
class Diagnostic:
    """One finding, designed to be read by people and parsed by programs.

    * ``code`` -- stable identifier such as ``"DEGENERATE_TRIANGLES"``; match on this.
    * ``severity`` -- ``"error"``, ``"warning"`` or ``"info"``.
    * ``message`` -- human-readable explanation.
    * ``location`` -- where the finding applies: a JSON pointer into an
      external asset (``/meshes/0/primitives/1``) or a path into a report
      (``meshes[0]``, ``objects[<id>]``); ``None`` if global.
    * ``count`` -- number of affected elements, when meaningful.
    * ``samples`` -- up to :data:`MAX_SAMPLES` example element indices.
    * ``lossy`` -- ``True`` if the finding means content was (or would be)
      dropped or approximated by a conversion.
    """

    code: str
    severity: str
    message: str
    location: str | None = None
    count: int | None = None
    samples: tuple[int, ...] = field(default_factory=tuple)
    lossy: bool = False

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "location": self.location,
            "count": self.count,
            "samples": list(self.samples),
            "lossy": self.lossy,
        }


MAX_SAMPLES = 10


def worst_severity(diagnostics: list[Diagnostic]) -> str:
    """``"error"``, ``"warning"`` or ``"ok"`` -- infos never change the status."""
    severities = {d.severity for d in diagnostics}
    if ERROR in severities:
        return "error"
    if WARNING in severities:
        return "warning"
    return "ok"
