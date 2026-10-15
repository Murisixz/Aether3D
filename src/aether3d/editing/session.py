"""Editing sessions: edit a scene object's mesh and commit the result atomically."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..diagnostics import INFO, Diagnostic
from ..errors import DetachedObjectError, EditError, ValidationError
from ..inspection import MeshGeometryReport, inspect_mesh
from ..mesh import Mesh
from .mesh import EditableMesh
from .validation import TopologyReport

if TYPE_CHECKING:
    from ..scene import SceneObject


@dataclass
class CommitResult:
    """What a commit did.

    ``mesh`` is the committed :class:`Mesh` (``None`` if all geometry was
    deleted, in which case the objects now have no mesh); ``objects`` are the
    ids of the objects that received it; ``changed`` is ``False`` when no
    operation was applied (the original mesh is kept as-is).
    """

    mesh: Mesh | None
    objects: list[str]
    changed: bool
    validation: TopologyReport
    inspection: MeshGeometryReport | None
    diagnostics: list[Diagnostic] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": self.changed,
            "objects": list(self.objects),
            "operations": list(self.operations),
            "mesh": None
            if self.mesh is None
            else {
                "vertex_count": self.mesh.vertex_count,
                "triangle_count": self.mesh.triangle_count,
            },
            "validation": self.validation.to_dict(),
            "inspection": None if self.inspection is None else self.inspection.to_dict(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }


class EditSession:
    """Edit one object's geometry, then commit a new immutable :class:`Mesh`.

    The object's current mesh is never modified (meshes are immutable): the
    session works on an :class:`EditableMesh` copy and only :meth:`commit`
    assigns the result. By default only this object receives the new mesh,
    so other objects that shared the old mesh are unaffected; pass
    ``scope="shared"`` to update every object in the scene that uses the same
    mesh. Operations and queries of the editable mesh are available directly
    on the session (``session.extrude_faces(...)``) or via ``session.mesh``.

    Used as a context manager, the session commits when the block finishes
    normally and discards everything if it raises::

        with cube.edit_mesh() as edit:
            top = edit.select_faces(normal=((0, 0, 1), 1), expect=1)
            edit.extrude_faces(top, 0.5)
    """

    def __init__(self, obj: SceneObject, *, weld: str = "exact", quads: bool = True) -> None:
        if obj.scene is None:
            raise DetachedObjectError(f"{obj!r} has been removed from its scene")
        self.object = obj
        self._source = obj.mesh
        if self._source is None:
            self.mesh = EditableMesh()
        else:
            self.mesh = EditableMesh.from_mesh(self._source, weld=weld, quads=quads)
        self.mesh.world_matrix = obj.world_matrix
        self.result: CommitResult | None = None

    def __getattr__(self, name: str) -> Any:
        # Delegate operations and queries to the editable mesh.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["mesh"], name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | {n for n in dir(self.mesh) if not n.startswith("_")})

    @property
    def closed(self) -> bool:
        return self.mesh._closed

    def commit(self, *, scope: str = "object", allow_invalid: bool = False) -> CommitResult:
        """Validate, triangulate and assign the edited mesh; closes the session.

        Raises ``VALIDATION_FAILED`` (session stays open, nothing assigned) if
        the mesh has error-severity topology findings, unless
        ``allow_invalid=True``. Raises ``STALE_SESSION`` if the object's mesh
        was replaced since the session started.
        """
        if self.closed:
            raise EditError("SESSION_CLOSED", "the session was already committed or discarded")
        if scope not in ("object", "shared"):
            raise ValidationError(f"scope must be 'object' or 'shared', got {scope!r}")
        obj = self.object
        if obj.scene is None:
            raise DetachedObjectError(f"{obj!r} has been removed from its scene")
        if obj.mesh is not self._source:
            raise EditError(
                "STALE_SESSION",
                f"the mesh of {obj.name!r} changed after this session started",
                hint="discard this session and start a new one",
            )
        report = self.mesh.validate()
        if not report.ok and not allow_invalid:
            codes = sorted(d.code for d in report.diagnostics if d.severity == "error")
            raise EditError(
                "VALIDATION_FAILED",
                "the edited mesh has errors: " + ", ".join(codes),
                hint="fix them (see mesh.validate()) or pass allow_invalid=True",
            )
        diagnostics = list(self.mesh.source_diagnostics)
        changed = self.mesh.has_changes
        if not changed:
            new_mesh = self._source
        elif self.mesh.face_count == 0:
            new_mesh = None
            diagnostics.append(
                Diagnostic(
                    "MESH_EMPTY", INFO, "all faces were deleted; the object no longer has a mesh"
                )
            )
        else:
            new_mesh, build = self.mesh.build_mesh()
            diagnostics += build
        targets = [obj]
        if scope == "shared" and self._source is not None:
            targets = [o for o in obj.scene if o.mesh is self._source]
        with obj.scene.history._op(f"Edit mesh of {obj.name!r}"):  # one undo step
            for target in targets:
                target.mesh = new_mesh
        self.mesh._closed = True
        self.result = CommitResult(
            mesh=new_mesh,
            objects=[t.id for t in targets],
            changed=changed,
            validation=report,
            inspection=None if new_mesh is None else inspect_mesh(new_mesh),
            diagnostics=diagnostics,
            operations=[h["operation"] for h in self.mesh.history],
        )
        return self.result

    def discard(self) -> None:
        """Close the session without changing the object."""
        self.mesh._closed = True

    def __enter__(self) -> EditSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.discard()
        elif not self.closed:
            self.commit()
        return False

    def __repr__(self) -> str:
        state = "closed" if self.closed else "open"
        count = len(self.mesh.history)
        return f"EditSession(object={self.object.name!r}, {state}, operations={count})"
