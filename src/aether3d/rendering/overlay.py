"""Render overlays: evaluated values that stand in for authored ones in one render.

An animated scene renders at a time: its authored state is the rest pose, and the
animation's evaluation at that time replaces some values *for that render only* --
objects' local transforms, visibility, camera/light/appearance fields, material
parameters, the world's background, ambient light, environment and atmosphere (0.20.0),
deformed (skinned, morphed) meshes and the shot camera. A
:class:`RenderOverlay` carries them; ``extract_render_scene(..., overlay=...)`` (and the
spatial index, world matrices and framing) apply it without touching the scene.

Local matrices are computed with exactly the arithmetic of authored transforms
(``scene_data._compose_locals``), so rendering an overlay gives the same bits as
writing the pose into the scene and rendering that.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..mesh import Mesh


@dataclass(frozen=True, eq=False)
class RenderOverlay:
    """See the module docstring. ``local`` is ``(n, 4, 4)`` float64, read-only, one row
    per id of ``ids``; ``camera`` is the id of the camera object to render from when the
    render names none (a shot sequence's camera at that time), or ``None``."""

    ids: tuple[str, ...] = ()
    local: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))
    visible: Mapping[str, bool] = field(default_factory=dict)
    fields: Mapping[tuple[str, str], Mapping[str, Any]] = field(default_factory=dict)
    materials: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    meshes: Mapping[str, Mesh] = field(default_factory=dict)
    world: Mapping[str, Any] = field(default_factory=dict)  # World.configure arguments
    camera: str | None = None
    time: float | None = None
    index: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        local = self.local
        if local.flags.writeable:
            local = np.ascontiguousarray(local, dtype=np.float64).copy()
            local.flags.writeable = False
            object.__setattr__(self, "local", local)
        object.__setattr__(self, "index", {oid: k for k, oid in enumerate(self.ids)})

    @property
    def empty(self) -> bool:
        return not (
            self.ids or self.visible or self.fields or self.materials or self.meshes
            or self.world or self.camera is not None
        )  # fmt: skip

    def is_visible(self, obj: Any) -> bool:
        return self.visible.get(obj._id, obj._visible)

    def field_values(self, obj_id: str, group: str) -> Mapping[str, Any] | None:
        return self.fields.get((obj_id, group))


def animation_overlay(scene: Any, time: Any = None) -> RenderOverlay | None:
    """The overlay of ``scene``'s animation at ``time`` (default: its current time), or
    ``None`` when nothing is animated (the authored scene renders as it is)."""
    animation = getattr(scene, "_animation", None)
    if animation is None:
        if time is not None:
            from ..animation.time import check_time

            check_time(time, "time")
        return None
    return animation.overlay(time)
