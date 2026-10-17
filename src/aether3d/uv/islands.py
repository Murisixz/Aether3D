"""UV islands of an existing map: detection and per-island data.

An **island** is a set of triangles connected through edges whose two sides agree on
their UVs (no UV seam). On a :class:`~aether3d.Mesh` that is: faces joined across mated
edges (see :mod:`aether3d.uv.topology`) whose corner UVs match exactly.
"""

from __future__ import annotations

import numpy as np

from ..mesh import Mesh
from .topology import Topology, dense_labels


def island_labels(topo: Topology) -> tuple[np.ndarray, int]:
    """Island of every face (``0 .. k - 1``, by lowest face) and the island count."""
    labels = topo.face_components(topo.uv_seam)
    return dense_labels(labels)


def corner_uvs(mesh: Mesh) -> np.ndarray:
    """``(T, 3, 2)`` float64 UVs of every corner."""
    if mesh.uvs is None:
        raise ValueError("the mesh has no UVs")
    return mesh.uvs.astype(np.float64)[mesh.triangles.astype(np.int64)]
