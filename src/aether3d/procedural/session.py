"""Editing the stored base mesh of a procedural geometry with the mesh editing engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..editing import CommitResult, EditableMesh
from ..errors import EditError
from ..inspection import inspect_mesh
from .params import ProceduralError

if TYPE_CHECKING:
    from .graph import ProceduralGeometry


class SourceEditSession:
    """Destructively edit a geometry's *base mesh*; its modifiers stay non-destructive.

    Created by :meth:`ProceduralGeometry.edit_source`. Works like an
    :class:`~aether3d.editing.EditSession` (operations are available on the
    session; ``commit()`` / ``discard()``; context manager), but committing
    replaces the geometry node's stored base mesh, so every object using the
    node -- and every node that references it -- regenerates with all modifiers
    re-applied. The commit is a procedural transaction: if any affected
    geometry fails to evaluate, nothing changes.
    """

    def __init__(self, geometry: ProceduralGeometry) -> None:
        if geometry.source.kind != "mesh":
            raise ProceduralError(
                "SOURCE_NOT_EDITABLE",
                f"geometry {geometry.name!r} has a {geometry.source.kind} source",
                hint="call geometry.convert_source_to_mesh() first to edit it as a mesh",
            )
        self.geometry = geometry
        self._source = geometry.source.mesh
        self.mesh = EditableMesh.from_mesh(self._source)
        self.result: CommitResult | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["mesh"], name)

    @property
    def closed(self) -> bool:
        return self.mesh._closed

    def commit(self, *, allow_invalid: bool = False) -> CommitResult:
        if self.closed:
            raise EditError("SESSION_CLOSED", "the session was already committed or discarded")
        g = self.geometry
        if g.source.kind != "mesh" or g.source.mesh is not self._source:
            raise EditError("STALE_SESSION", f"the source of geometry {g.name!r} changed meanwhile")
        report = self.mesh.validate()
        if not report.ok and not allow_invalid:
            raise EditError(
                "VALIDATION_FAILED",
                "the edited mesh has errors: "
                + ", ".join(sorted(d.code for d in report.diagnostics if d.severity == "error")),
            )
        changed = self.mesh.has_changes
        diagnostics = list(self.mesh.source_diagnostics)
        if changed:
            new_mesh, build = self.mesh.build_mesh()
            diagnostics += build
            with g._system._transaction(f"Edit source mesh of geometry {g.name!r}"):
                g._system.set_source_mesh(g, new_mesh)
        else:
            new_mesh = self._source
        self.mesh._closed = True
        self.result = CommitResult(
            mesh=new_mesh,
            objects=g.users,
            changed=changed,
            validation=report,
            inspection=inspect_mesh(new_mesh),
            diagnostics=diagnostics,
            operations=[h["operation"] for h in self.mesh.history],
        )
        return self.result

    def discard(self) -> None:
        self.mesh._closed = True

    def __enter__(self) -> SourceEditSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.discard()
        elif not self.closed:
            self.commit()
        return False
