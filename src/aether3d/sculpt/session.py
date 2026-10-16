"""Sculpt sessions: sculpt a scene object's mesh and commit it as one undo step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..diagnostics import Diagnostic
from ..errors import DetachedObjectError, SculptError, ValidationError
from ..mesh import Mesh
from .mesh import SculptMesh
from .validate import TIERS, MeshReport, check_mesh, tier_index

if TYPE_CHECKING:
    from ..procedural.graph import ProceduralGeometry
    from ..scene import SceneObject


@dataclass
class SculptCommit:
    """What a commit did: the committed mesh, the objects that received it, whether
    anything changed, the strokes and operations applied, and a validity report of the
    committed mesh (:func:`~aether3d.sculpt.validate.check_mesh`)."""

    mesh: Mesh
    objects: list[str]
    changed: bool
    report: MeshReport
    operations: list[str] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "objects": list(self.objects),
            "operations": list(self.operations),
            "mesh": {
                "vertex_count": self.mesh.vertex_count,
                "triangle_count": self.mesh.triangle_count,
            },
            "report": self.report.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


def _require(report: MeshReport, require: str | None) -> None:
    """Refuse a result below the required validity tier (before anything changes)."""
    if require is None:
        return
    tier = report.summary.get("tier")
    if tier_index(tier) >= tier_index(require):
        return
    problems = [
        f"{d.count} {d.code.lower().replace('_', ' ')}"
        for d in report.diagnostics
        if d.code != "BOUNDARY_EDGES" or require in ("closed", "solid")
    ]
    if require == "solid" and tier == "closed" and not report.summary.get("volume", 0) > 0:
        problems.append("a volume that is not positive (inside out)")
    raise SculptError(
        "BELOW_VALIDITY_TIER",
        f"the result is {tier or 'not even valid'}, below the required {require!r}"
        + (f": {', '.join(problems)}" if problems else ""),
        hint="fix the mesh (fill holes, remesh, undo the step that broke it) or require "
        "a lower tier",
    )


class _Session:
    """Shared machinery: delegation to the sculpt mesh, the context manager."""

    mesh: SculptMesh

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["mesh"], name)

    @property
    def closed(self) -> bool:
        return self.mesh._closed

    def discard(self) -> None:
        """Close the session without changing anything."""
        self.mesh._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.discard()
        elif not self.closed:
            self.commit()
        return False

    def _state(self) -> tuple:
        return (len(self.mesh._history), self.mesh._serial, len(self.mesh._redo))


class SculptSession(_Session):
    """Sculpt one object's mesh, then commit a new immutable :class:`Mesh`.

    The object's mesh is never modified: the session works on a
    :class:`~aether3d.sculpt.SculptMesh` and :meth:`commit` assigns the result as one
    step of ``scene.history``. Strokes and other operations are available on the
    session (``session.stroke(...)``) and each is one step of the session's own undo
    (``session.undo()``, ``redo()``). ``commit(close=False)`` assigns the current result
    and keeps sculpting (each such commit is its own scene step).
    """

    def __init__(self, obj: SceneObject) -> None:
        if obj.scene is None:
            raise DetachedObjectError(f"{obj!r} has been removed from its scene")
        if obj.mesh is None:
            raise SculptError("NO_MESH", f"object {obj.name!r} has no mesh to sculpt")
        self.object = obj
        self._source = obj.mesh
        self.mesh = SculptMesh.from_mesh(self._source)
        self.mesh.world_matrix = obj.world_matrix
        self._committed_state = self._state()
        self.result: SculptCommit | None = None

    def commit(
        self,
        *,
        scope: str = "object",
        close: bool = True,
        check: bool = True,
        require: str | None = None,
    ) -> SculptCommit:
        """Assign the sculpted mesh (``scope="shared"``: to every object using the
        original mesh). ``check`` adds a self-intersection test to the report.
        ``require`` names the validity tier the result must reach (``"manifold"``,
        ``"closed"`` or ``"solid"``, see :data:`~aether3d.sculpt.validate.TIERS`): a
        result below it is refused with ``BELOW_VALIDITY_TIER`` and nothing changes (the
        session stays open)."""
        if self.closed:
            raise SculptError("SESSION_CLOSED", "the session was already committed or discarded")
        if scope not in ("object", "shared"):
            raise ValidationError(f"scope must be 'object' or 'shared', got {scope!r}")
        if require is not None and require not in TIERS:
            raise ValidationError(f"require must be one of {', '.join(TIERS)}")
        obj = self.object
        if obj.scene is None:
            raise DetachedObjectError(f"{obj!r} has been removed from its scene")
        new_mesh = self.mesh.to_mesh()
        current = obj.mesh
        expected = self._source if self.result is None else self.result.mesh
        if current is not expected:
            raise SculptError(
                "STALE_SESSION",
                f"the mesh of {obj.name!r} changed after this session started",
                hint="discard this session and start a new one",
            )
        changed = new_mesh is not current and self._state() != self._committed_state
        if not changed:
            new_mesh = current
        report = check_mesh(
            new_mesh.positions, new_mesh.triangles, self_intersections=check or require == "solid"
        )
        _require(report, require)
        targets = [obj]
        if scope == "shared":
            targets = [o for o in obj.scene if o.mesh is current]
        if changed:
            with obj.scene.history._op(f"Sculpt mesh of {obj.name!r}"):
                for target in targets:
                    target.mesh = new_mesh
        self._committed_state = self._state()
        if close:
            self.mesh._closed = True
        self.result = SculptCommit(
            mesh=new_mesh,
            objects=[t.id for t in targets] if changed else [],
            changed=changed,
            report=report,
            operations=[h["operation"] for h in self.mesh.history],
            diagnostics=list(self.mesh.source_diagnostics),
        )
        return self.result

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        count = len(self.mesh.history)
        return f"SculptSession(object={self.object.name!r}, {state}, operations={count})"


class SourceSculptSession(_Session):
    """Sculpt the stored base mesh of a procedural geometry; its modifiers stay
    non-destructive. Committing replaces the base mesh, so every user regenerates
    (a procedural transaction: nothing changes if any of them fails to evaluate)."""

    def __init__(self, geometry: ProceduralGeometry) -> None:
        from ..procedural.params import ProceduralError

        if geometry.source.kind != "mesh":
            raise ProceduralError(
                "SOURCE_NOT_EDITABLE",
                f"geometry {geometry.name!r} has a {geometry.source.kind} source",
                hint="call geometry.convert_source_to_mesh() first to sculpt it",
            )
        self.geometry = geometry
        self._source = geometry.source.mesh
        self.mesh = SculptMesh.from_mesh(self._source)
        self._committed_state = self._state()
        self.result: SculptCommit | None = None

    def commit(self, *, check: bool = True, require: str | None = None) -> SculptCommit:
        if self.closed:
            raise SculptError("SESSION_CLOSED", "the session was already committed or discarded")
        if require is not None and require not in TIERS:
            raise ValidationError(f"require must be one of {', '.join(TIERS)}")
        g = self.geometry
        if g.source.kind != "mesh" or g.source.mesh is not self._source:
            raise SculptError(
                "STALE_SESSION", f"the source of geometry {g.name!r} changed meanwhile"
            )
        new_mesh = self.mesh.to_mesh()
        changed = new_mesh is not self._source
        report = check_mesh(
            new_mesh.positions, new_mesh.triangles, self_intersections=check or require == "solid"
        )
        _require(report, require)
        if changed:
            with g._system._transaction(f"Sculpt source mesh of geometry {g.name!r}"):
                g._system.set_source_mesh(g, new_mesh)
        self.mesh._closed = True
        self.result = SculptCommit(
            mesh=new_mesh,
            objects=g.users,
            changed=changed,
            report=report,
            operations=[h["operation"] for h in self.mesh.history],
            diagnostics=list(self.mesh.source_diagnostics),
        )
        return self.result
