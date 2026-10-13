"""The public :class:`EditableMesh` class."""

from __future__ import annotations

from .bevel import BevelMixin
from .core import MeshCore
from .extrude import ExtrudeMixin
from .selection import SelectionMixin
from .subdivide import SubdivideMixin
from .topology import TopologyMixin
from .transform import TransformMixin
from .validation import TopologyReport, validate


class EditableMesh(
    SelectionMixin,
    TransformMixin,
    TopologyMixin,
    ExtrudeMixin,
    BevelMixin,
    SubdivideMixin,
    MeshCore,
):
    """An editable polygon mesh with stable element ids and transactional operations.

    Create one with :meth:`from_mesh` (from an immutable :class:`~aether3d.Mesh`)
    or empty (``EditableMesh()``) and add geometry with :meth:`add_box`,
    :meth:`add_grid`, :meth:`create_vertex` and :meth:`create_face`. Convert
    back with :meth:`to_mesh`. In a scene, use ``obj.edit_mesh()`` to get an
    :class:`~aether3d.editing.EditSession` that commits the result to the object.

    **Element ids.** Vertices, edges and faces have non-negative integer ids,
    unique per kind within this mesh. Ids of elements an operation does not
    touch never change; an operation that reshapes a face keeps that face's
    id (reported in ``modified_faces``); new elements get ids no earlier
    element had. Every operation returns an :class:`OperationResult` listing
    created and removed ids. ``undo()`` restores earlier ids exactly; ids
    are not persistent across :meth:`to_mesh` (a fresh ``from_mesh`` assigns
    ids deterministically from the triangle mesh).

    **Atomicity.** Every operation either completes or raises and leaves the
    mesh exactly as it was (no partial edits). Errors are
    :class:`~aether3d.EditError` with a stable ``code``.

    **Conventions.** Local coordinates; faces are counter-clockwise seen from
    the front (right-hand rule gives the outward normal); UV origin is the
    bottom-left; see ``docs/EDITING.md``.
    """

    def validate(self) -> TopologyReport:
        """Check topology and geometry; see :mod:`aether3d.editing.validation`."""
        return validate(self)

    @classmethod
    def operations(cls) -> list[str]:
        """Names of all editing operations (undoable methods returning OperationResult)."""
        return sorted(
            name for name in dir(cls) if getattr(getattr(cls, name, None), "is_operation", False)
        )
