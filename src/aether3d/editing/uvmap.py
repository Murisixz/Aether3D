"""UV mapping operations of the editable mesh: ``unwrap`` and ``project_uvs``.

Both triangulate the current polygons (as :meth:`~EditableMesh.to_mesh` does), compute
the map with :mod:`aether3d.uv` -- keeping the triangles of every polygon in one chart, so
each face corner gets one UV -- and write the result into the ``"uv"`` corner layer of the
selected faces (one undoable operation). Positions, normals, groups and the other faces'
UVs are untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import EditError, TexturingError
from .core import UV, Elements, operation


class UVMapMixin:
    def _uv_faces(self, faces: Elements | None) -> list[int]:
        if faces is None:
            return sorted(self._faces)
        return sorted(self._elements(faces, "faces"))

    def _apply_map(self, mesh, sources, selected: set[int], new_mesh) -> int:
        if UV not in self._layers:
            if len(selected) < len(self._faces):
                raise EditError(
                    "NO_UV_LAYER",
                    "this editable mesh has no UVs: map every face first",
                    hint="call unwrap() or project_uvs() without faces",
                )
            self._add_layer(UV, 2, (0.0, 0.0))
        tri_uv = new_mesh.uvs.astype(np.float64)[new_mesh.triangles]
        values: dict[int, dict[int, tuple[float, float]]] = {}
        for t, (fid, a, b, c) in enumerate(sources):
            if fid not in selected:
                continue
            corner = values.setdefault(fid, {})
            for k, local in enumerate((a, b, c)):
                uv = (float(tri_uv[t, k, 0]), float(tri_uv[t, k, 1]))
                old = corner.setdefault(local, uv)
                if old != uv:
                    raise EditError(
                        "FACE_SPLIT_BY_MAP",
                        f"face {fid} lies across a seam of the map",
                        elements={"faces": [fid]},
                        hint="triangulate the face, or map it with another projection",
                    )
        for fid, corner in values.items():
            n = len(self._faces[fid])
            self._set_corners(fid, UV, [corner[k] for k in range(n)])
        return len(values)

    @operation
    def unwrap(
        self,
        faces: Elements | None = None,
        *,
        seams: Elements | None = None,
        **options: Any,
    ):
        """Automatic UVs for ``faces`` (default: all) with :func:`aether3d.uv.unwrap`.

        ``seams`` are edge ids the map must cut (besides boundaries, sharp edges with
        ``split_by=["hard"]`` -- the default -- and the options of ``aether3d.uv.unwrap``);
        the other options are passed through (``resolution``, ``padding``, ``method``,
        ``max_stretch``, ``angle_limit``, ``split_by``, ``texels_per_unit``, ...).
        Unselected faces keep their UV islands' shapes; their islands are repacked with the
        new ones (``pack=True``, the default), so they move.
        """
        from .. import uv

        mesh, _, sources = self._build_mesh()
        selected = set(self._uv_faces(faces))
        tri_faces = np.array([s[0] for s in sources], dtype=np.int64)
        tri_sel = np.flatnonzero(np.isin(tri_faces, list(selected)))
        seam_pairs = None
        if seams is not None:
            pairs = []
            pos = mesh.positions
            for e in self._elements(seams, "edges"):
                a, b = self._edge_verts[e]
                ia = np.flatnonzero(np.all(pos == self._pos[a].astype(np.float32), axis=1))
                ib = np.flatnonzero(np.all(pos == self._pos[b].astype(np.float32), axis=1))
                if ia.size and ib.size:
                    pairs.append([int(ia[0]), int(ib[0])])
            seam_pairs = pairs
        split = options.pop("split_by", ("hard",))
        extra_cut = None
        if "hard" in (split if not isinstance(split, str) else (split,)):
            # Sharp edges are this mesh's hard edges (normals may not differ at them yet).
            extra_cut = [list(self._edge_verts[e]) for e in self._sharp]
        if extra_cut:
            pos = mesh.positions
            more = []
            for a, b in extra_cut:
                ia = np.flatnonzero(np.all(pos == self._pos[a].astype(np.float32), axis=1))
                ib = np.flatnonzero(np.all(pos == self._pos[b].astype(np.float32), axis=1))
                if ia.size and ib.size:
                    more.append([int(ia[0]), int(ib[0])])
            seam_pairs = (seam_pairs or []) + more
        partial = len(selected) < len(self._faces)
        if partial and UV not in self._layers:
            raise EditError(
                "NO_UV_LAYER",
                "this editable mesh has no UVs: map every face first",
                hint="call unwrap() without faces",
            )
        try:
            result = uv.unwrap(
                mesh,
                faces=tri_sel if partial else None,
                seams=seam_pairs or None,
                split_by=split,
                atoms=tri_faces,
                **options,
            )
        except TexturingError as exc:
            raise EditError(exc.code, exc.message, hint=exc.hint) from None
        # Kept islands were repacked (moved, not reshaped) with the new ones: write them too.
        count = self._apply_map(mesh, sources, set(self._faces), result.mesh)
        self._op_diagnostics.extend(result.diagnostics)
        return {"faces": count, "report": result.report}

    @operation
    def project_uvs(self, kind: str, faces: Elements | None = None, **options: Any):
        """UVs by projection (``"planar"``, ``"box"``, ``"cylindrical"``, ``"spherical"``;
        :func:`aether3d.uv.project`) for ``faces`` (default: all; the projection is computed
        on the whole mesh and written to the selected faces)."""
        from .. import uv

        mesh, _, sources = self._build_mesh()
        selected = set(self._uv_faces(faces))
        try:
            result = uv.project(mesh, kind, **options)
        except TexturingError as exc:
            raise EditError(exc.code, exc.message, hint=exc.hint) from None
        count = self._apply_map(mesh, sources, selected, result.mesh)
        return {"faces": count, "report": result.report}
