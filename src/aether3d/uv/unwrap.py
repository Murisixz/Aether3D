"""``unwrap``: automatic UV maps for a mesh (charts, parameterization, packing)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import TexturingError
from ..mesh import Mesh
from . import pack as packing
from .charts import Charts, check_options, compute_charts
from .islands import island_labels
from .topology import Topology

MAX_TRIANGLES = 20_000_000


@dataclass
class UnwrapResult:
    """What :func:`unwrap` produced."""

    mesh: Mesh
    face_chart: np.ndarray  # (T,) island of every triangle in the new map
    report: dict[str, Any]
    diagnostics: list[Diagnostic] = field(default_factory=list)


def _check_resolution(value: Any, name: str = "resolution") -> tuple[int, int]:
    if isinstance(value, bool):
        raise TexturingError("INVALID_ARGUMENT", f"{name} must be an integer or [w, h]", name)
    if isinstance(value, (int, np.integer)):
        value = (int(value), int(value))
    try:
        w, h = (int(v) if not isinstance(v, bool) and float(v) == int(v) else None for v in value)
    except (TypeError, ValueError):
        raise TexturingError(
            "INVALID_ARGUMENT", f"{name} must be an integer or [w, h]", name
        ) from None
    if (
        w is None
        or h is None
        or not (1 <= w <= packing.MAX_RESOLUTION and 1 <= h <= packing.MAX_RESOLUTION)
    ):
        raise TexturingError(
            "INVALID_ARGUMENT", f"{name} must be in [1, {packing.MAX_RESOLUTION}] texels", name
        )
    return w, h


def _seam_mask(topo: Topology, seams: Any) -> np.ndarray:
    """Half-edges cut by user seams given as ``(a, b)`` render-vertex pairs."""
    cut = np.zeros(3 * topo.triangle_count, dtype=bool)
    if seams is None:
        return cut
    arr = np.asarray(seams)
    if arr.size == 0:
        return cut
    if not np.issubdtype(arr.dtype, np.integer) or arr.ndim != 2 or arr.shape[1] != 2:
        raise TexturingError(
            "INVALID_SEAM", "seams must be a list of [vertex, vertex] pairs", "seams"
        )
    n = topo.positions.shape[0]
    if arr.min() < 0 or arr.max() >= n:
        raise TexturingError("INVALID_SEAM", f"seam vertices must be in [0, {n - 1}]", "seams")
    a = topo.weld[arr[:, 0]]
    b = topo.weld[arr[:, 1]]
    count = max(topo.welded_count, 1)
    want = np.minimum(a, b) * count + np.maximum(a, b)
    have = topo.edges[:, 0] * count + topo.edges[:, 1]
    found = np.isin(want, have)
    if not found.all():
        missing = arr[~found][:5].tolist()
        raise TexturingError(
            "INVALID_SEAM",
            f"{int((~found).sum())} seam pair(s) are not edges of the mesh (first: {missing})",
            "seams",
        )
    edge_ids = np.searchsorted(have, want)
    flag = np.zeros(topo.edges.shape[0], dtype=bool)
    flag[edge_ids] = True
    return flag[topo.edge]


def unwrap(
    mesh: Mesh,
    *,
    seams: Any = None,
    split_by: Any = ("hard",),
    angle_limit: float = 66.0,
    method: str = "isometric",
    max_stretch: float = 1.15,
    iterations: int = 12,
    faces: Any = None,
    pack: bool = True,
    resolution: Any = 1024,
    padding: int = 4,
    rotate: bool = True,
    texels_per_unit: float | None = None,
    atoms: Any = None,
    should_stop=None,
) -> UnwrapResult:
    """New UVs for ``mesh`` (or for its triangles in ``faces``), as a new :class:`Mesh`.

    See ``docs/TEXTURING.md``: charts are cut at seams, boundaries, hard edges (``split_by``)
    and edges sharper than ``angle_limit`` degrees, parameterized without inverted
    triangles (``method="isometric"``: SLIM; ``"conformal"``: LSCM), split until their
    stretch is at most ``max_stretch``, and packed into a ``resolution`` atlas with
    ``padding`` texels between islands. Positions, normals, triangles and face groups are
    kept; vertices are split where the new map has seams. ``atoms`` (one id per triangle)
    keeps triangles of one polygon in one chart (used by ``EditableMesh.unwrap``).
    """
    t_start = time.perf_counter()
    if not isinstance(mesh, Mesh):
        raise TexturingError("INVALID_ARGUMENT", "unwrap needs a Mesh", "mesh")
    if mesh.triangle_count == 0:
        raise TexturingError("EMPTY_MESH", "the mesh has no triangles", "mesh")
    if mesh.triangle_count > MAX_TRIANGLES:
        raise TexturingError(
            "RESOURCE_LIMIT", f"unwrap is limited to {MAX_TRIANGLES:,} triangles", "mesh"
        )
    options = check_options(angle_limit, split_by, method, max_stretch, iterations)
    res = _check_resolution(resolution)
    if (
        isinstance(padding, bool)
        or not isinstance(padding, (int, np.integer))
        or not 0 <= padding <= 256
    ):
        raise TexturingError(
            "INVALID_ARGUMENT", "padding must be an integer in [0, 256]", "padding"
        )
    if not isinstance(rotate, bool) or not isinstance(pack, bool):
        raise TexturingError("INVALID_ARGUMENT", "pack and rotate must be true or false")
    if texels_per_unit is not None and (
        isinstance(texels_per_unit, bool)
        or not isinstance(texels_per_unit, (int, float))
        or not (math.isfinite(texels_per_unit) and texels_per_unit > 0)
    ):
        raise TexturingError(
            "INVALID_ARGUMENT", "texels_per_unit must be a positive number", "texels_per_unit"
        )
    topo = Topology.from_mesh(mesh)
    T = topo.triangle_count
    active = np.ones(T, dtype=bool)
    if faces is not None:
        idx = np.asarray(faces)
        if idx.size and (
            not np.issubdtype(idx.dtype, np.integer) or idx.min() < 0 or idx.max() >= T
        ):
            raise TexturingError(
                "INVALID_ARGUMENT", f"faces must be triangle indices in [0, {T - 1}]", "faces"
            )
        active = np.zeros(T, dtype=bool)
        active[idx.astype(np.int64)] = True
        if not active.any():
            raise TexturingError("INVALID_ARGUMENT", "faces selects no triangle", "faces")
        if (~active).any() and mesh.uvs is None:
            raise TexturingError(
                "NO_UVS",
                "a partial unwrap keeps the other faces' UVs, but the mesh has none",
                "faces",
                hint="unwrap the whole mesh",
            )
    cut = _seam_mask(topo, seams)
    cut |= topo.dihedral > math.radians(options.angle_limit)
    if "hard" in options.split_by:
        cut |= topo.hard
    if "groups" in options.split_by:
        cut |= topo.group_border
    if "seams" in options.split_by:
        cut |= topo.uv_seam
    # Edges between the unwrapped faces and the others always cut.
    m = topo.mate
    other = (m >= 0) & (active[np.arange(3 * T) // 3] != active[np.maximum(m, 0) // 3])
    cut |= other
    atom_ids = None
    if atoms is not None:
        atom_ids = np.asarray(atoms, dtype=np.int64).reshape(-1)
        if atom_ids.shape[0] != T:
            raise TexturingError("INVALID_ARGUMENT", "atoms needs one id per triangle", "atoms")
        # Edges inside a polygon are never cut.
        mh = np.flatnonzero(m >= 0)
        inner = mh[atom_ids[mh // 3] == atom_ids[m[mh] // 3]]
        cut[inner] = False
    t0 = time.perf_counter()
    charts = compute_charts(
        topo, cut, options, faces=active, atoms=atom_ids, should_stop=should_stop
    )
    t_charts = time.perf_counter() - t0
    diagnostics: list[Diagnostic] = []
    # Corner UVs (world units) for unwrapped faces; kept UVs for the others.
    corner_uv = np.zeros((T, 3, 2))
    corner_uv[active] = charts.uv[charts.corner_vertex[active]]
    island = np.full(T, -1, dtype=np.int64)
    island[active] = charts.face_chart[active]
    corner_id = np.full((T, 3), -1, dtype=np.int64)
    corner_id[active] = charts.corner_vertex[active]
    kept_islands = 0
    if (~active).any():
        labels, count = island_labels(topo)
        keep = ~active
        old = np.unique(labels[keep])
        remap = np.full(count, -1, dtype=np.int64)
        remap[old] = charts.count + np.arange(old.shape[0])
        island[keep] = remap[labels[keep]]
        kept_islands = int(old.shape[0])
        corner_uv[keep] = topo.uvs[topo.triangles[keep]]
        # Kept corners: one id per (render vertex), offset past the chart vertices.
        base = int(charts.uv.shape[0])
        nv = topo.positions.shape[0]
        corner_id[keep] = base + labels[keep][:, None] * nv + topo.triangles[keep]
    n_islands = charts.count + kept_islands
    t1 = time.perf_counter()
    pack_report: dict[str, Any] | None = None
    if pack:
        groups = [np.flatnonzero(island == c) for c in range(n_islands)]
        tri_uv = [corner_uv[g] for g in groups]
        if kept_islands:
            # Kept islands are in UV units; bring them to world units by their own density.
            for c in range(charts.count, n_islands):
                g = groups[c]
                uv_area = _area(tri_uv[c])
                area3 = float(topo.face_area[g].sum())
                if uv_area > 0 and area3 > 0:
                    tri_uv[c] = tri_uv[c] * math.sqrt(area3 / uv_area)
        placed, result = packing.pack(
            tri_uv,
            res,
            padding=int(padding),
            rotate=rotate,
            texels_per_unit=texels_per_unit,
            should_stop=should_stop,
        )
        for c, g in enumerate(groups):
            corner_uv[g] = placed[c]
        uv_area = float(sum(_area(t) for t in placed))
        pack_report = {
            "resolution": list(result.resolution),
            "padding": int(padding),
            "texels_per_unit": result.texels_per_unit,
            "utilization": uv_area,
            "rows_used": result.peak,
        }
    else:
        diagnostics.append(
            Diagnostic(
                "UVS_NOT_PACKED",
                INFO,
                "pack=False: islands keep their world-unit scale and may overlap",
                None,
            )
        )
    t_pack = time.perf_counter() - t1
    new_mesh = _rebuild(mesh, topo, corner_id, corner_uv)
    stretch_l2 = charts.stretch_l2
    bad = int(np.count_nonzero(stretch_l2 > options.max_stretch))
    if bad:
        diagnostics.append(
            Diagnostic(
                "STRETCH_ABOVE_LIMIT",
                WARNING,
                f"{bad} chart(s) of a single triangle could not reach max_stretch",
                None,
            )
        )
    report = {
        "triangles": T,
        "unwrapped_triangles": int(active.sum()),
        "charts": int(charts.count),
        "kept_islands": kept_islands,
        "splits": int(charts.splits),
        "method": options.method,
        "stretch_l2": {
            "max": float(stretch_l2.max()) if stretch_l2.size else 1.0,
            "mean": float(np.average(stretch_l2, weights=_chart_areas(topo, charts)))
            if stretch_l2.size
            else 1.0,
        },
        "stretch_max": float(charts.stretch_max.max()) if charts.count else 1.0,
        "pack": pack_report,
        "seconds": {
            "charts": t_charts,
            "pack": t_pack,
            "total": time.perf_counter() - t_start,
        },
    }
    return UnwrapResult(new_mesh, island, report, diagnostics)


def _chart_areas(topo: Topology, charts: Charts) -> np.ndarray:
    sel = charts.face_chart >= 0
    return np.bincount(charts.face_chart[sel], topo.face_area[sel], charts.count) + 1e-300


def _area(t: np.ndarray) -> float:
    e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return float(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() * 0.5)


def _rebuild(mesh: Mesh, topo: Topology, corner_id: np.ndarray, corner_uv: np.ndarray) -> Mesh:
    """A mesh whose vertices are the distinct (render vertex, map vertex) pairs of the
    corners, numbered by first use; positions, normals and groups are kept exactly."""
    tris = topo.triangles
    rv = tris.reshape(-1)
    mv = corner_id.reshape(-1)
    key = rv * (int(mv.max()) + 1) + mv
    _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.shape[0], dtype=np.int64)
    rank[order] = np.arange(order.shape[0])
    new_index = rank[inverse.reshape(-1)]
    src_corner = first[order]
    positions = mesh.positions[rv[src_corner]]
    normals = None if mesh.normals is None else mesh.normals[rv[src_corner]]
    uvs = corner_uv.reshape(-1, 2)[src_corner].astype(np.float32)
    return Mesh(positions, new_index.reshape(-1, 3), normals=normals, uvs=uvs, groups=mesh.groups)
