"""Sweeps: extruding boundary edges, spinning faces around an axis, lathe surfaces.

* ``extrude_edges(edges, offset)`` turns boundary edges into walls: each edge gets a quad
  to its moved copy, wound like the face it borders, and consecutive edges share their
  copied vertex. Interior edges cannot be extruded this way (``NOT_BOUNDARY``).
* ``spin(faces, angle, steps, axis, center)`` extrudes a region ``steps`` times, rotating
  it by ``angle / steps`` each time (a swept solid). A full turn (``angle = 360``) with
  ``merge=True`` welds the last ring to the first and removes the region at the end,
  closing the sweep (a torus from a disc).
* ``add_lathe(profile, segments, angle)`` builds a surface of revolution from a profile
  of ``(radius, height)`` points around the local Z axis: quads between rings, triangle
  fans where the profile touches the axis, UVs ``u`` around and ``v`` along the profile.
  The profile runs bottom to top for outward-facing faces.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..diagnostics import WARNING, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import int_at_least, vec3
from . import _polygon as poly
from .core import NORMAL, UV, Elements, Selection, operation


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


class SweepMixin:
    @operation
    def extrude_edges(self, edges: Elements, offset: Iterable[float], *, space: str = "local"):
        """Extrude boundary edges by ``offset`` into walls (see the module docs).

        ``data["faces"]`` are the walls, ``data["vertices"]`` maps each original vertex to
        its moved copy.
        """
        eids = self._elements(edges, "edges")
        inner = [e for e in eids if len(self._edge_faces[e]) != 1]
        if inner:
            raise EditError(
                "NOT_BOUNDARY",
                f"{len(inner)} edges are not boundary edges",
                elements={"edges": inner},
                hint="extrude faces instead (extrude_faces)",
            )
        vector = np.asarray(vec3(offset, "offset"), dtype=np.float64)
        if space == "world":
            vector = self._world_direction_to_local(vector)
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        if not np.any(vector):
            raise EditError("ZERO_EXTRUSION", "the offset is zero")
        verts = sorted({v for e in eids for v in self._edge_verts[e]})
        copies = {v: self._new_vertex(self._pos[v] + vector) for v in verts}
        walls = []
        for e in eids:
            (f,) = self._edge_faces[e]
            a, b = self._edge_verts[e]
            loop = self._faces[f]
            if loop[(loop.index(a) + 1) % len(loop)] == b:
                a, b = b, a  # the wall runs the edge opposite to its face: b -> a
            corners = self._corners_of(f)
            ca, cb = corners[loop.index(a)], corners[loop.index(b)]
            walls.append(
                self._new_face(
                    [a, b, copies[b], copies[a]],
                    self._split_corners([ca, cb, cb, ca]),
                    smooth=self._smooth[f],
                    group=self._group.get(f, 0),
                )
            )
        self._require_non_degenerate(walls, "edge extrusion")
        if UV in self._layers:
            self._diagnose(
                Diagnostic(
                    "UV_DEGENERATE_FACES",
                    WARNING,
                    "walls created by edge extrusion copy the edges' UVs and have zero UV area",
                    None,
                    len(walls),
                    tuple(walls[:10]),
                )
            )
        return {"faces": Selection.of(faces=walls), "vertices": copies}

    @operation
    def spin(
        self,
        faces: Elements,
        angle: float,
        steps: int,
        *,
        axis: Iterable[float] = (0.0, 0.0, 1.0),
        center: Iterable[float] = (0.0, 0.0, 0.0),
        merge: bool = True,
        space: str = "local",
    ):
        """Sweep a region of faces around an axis (see the module docs).

        ``data["faces"]`` is the region at the end of the sweep (empty after a merged
        full turn), ``data["side_faces"]`` the swept walls.
        """
        fids = self._elements(faces, "faces")
        steps = int_at_least(steps, 1, "steps")
        if steps > 1024:
            raise ValidationError("steps must be at most 1024")
        a = float(angle)
        if not (math.isfinite(a) and 0 < abs(a) <= 360):
            raise ValidationError("angle must be a non-zero number of degrees up to 360")
        ax = poly.unit(np.asarray(vec3(axis, "axis"), dtype=np.float64))
        if ax is None:
            raise ValidationError("axis must be non-zero")
        c = np.asarray(vec3(center, "center"), dtype=np.float64)
        if space == "world":
            c = self._world_to_local(c[None])[0]
            ax = poly.unit(self._world_direction_to_local(ax))
        elif space != "local":
            raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
        regions = self._components(fids)
        if len(regions) != 1:
            raise EditError(
                "SPIN_NEEDS_ONE_REGION",
                f"the faces form {len(regions)} separate regions; spin one at a time",
            )
        region = regions[0]
        rot = _rotation(ax, math.radians(a) / steps)
        start_boundary = sorted(
            {v for a_, b_, _f in self._region_boundary_directed(region) for v in (a_, b_)}
        )
        if not start_boundary:
            raise EditError("SPIN_WITHOUT_BOUNDARY", "a closed region cannot be swept")
        track = {v: v for v in start_boundary}
        side_faces: list[int] = []
        for _ in range(steps):
            mapping, sides = self._extrude_region(region)
            side_faces += sides
            for v0, cur in list(track.items()):
                track[v0] = mapping.get(cur, cur)
            moved = sorted({v for f in region for v in self._faces[f]})
            p = self._pos[moved] - c
            self._set_positions(moved, p @ rot.T + c)
        full = abs(abs(a) - 360.0) <= 1e-9
        end_faces = list(region)
        if full and merge:
            if steps < 3:
                raise ValidationError("a merged full turn needs at least 3 steps")
            # The region is back where it started: drop it and weld its outline onto
            # the first ring, closing the sweep.
            region_vertices = {v for f in region for v in self._faces[f]}
            self._delete_faces_and_loose(region, keep_loose=True)
            for v0, cur in sorted(track.items()):
                if cur != v0 and self.has_vertex(cur):
                    self._merge_group([v0, cur], v0, self._pos[v0].copy())
            self._delete_loose(region_vertices)
            end_faces = []
        self._require_non_degenerate(side_faces, "spin")
        if NORMAL in self._layers:
            self._mark_dirty(side_faces)
        return {
            "faces": Selection.of(faces=[f for f in end_faces if f in self._faces]),
            "side_faces": Selection.of(faces=[f for f in side_faces if f in self._faces]),
        }

    @operation
    def add_lathe(
        self,
        profile: Iterable[Iterable[float]],
        *,
        segments: int = 32,
        angle: float = 360.0,
        smooth: bool = True,
        center: Iterable[float] = (0.0, 0.0, 0.0),
    ):
        """A surface of revolution of ``profile`` (``(radius, height)`` pairs) around the
        Z axis through ``center`` (see the module docs). ``data["faces"]`` and
        ``data["vertices"]`` select the new geometry."""
        pts = [tuple(float(x) for x in p) for p in profile]
        if len(pts) < 2 or any(len(p) != 2 for p in pts):
            raise ValidationError("profile needs at least 2 (radius, height) points")
        prof = np.asarray(pts, dtype=np.float64)
        if not np.all(np.isfinite(prof)) or np.any(prof[:, 0] < 0):
            raise ValidationError("profile radii must be finite and >= 0")
        if np.any(np.all(np.diff(prof, axis=0) == 0, axis=1)):
            raise ValidationError("consecutive profile points must differ")
        n = int_at_least(segments, 3, "segments")
        if n > 4096:
            raise ValidationError("segments must be at most 4096")
        a = float(angle)
        if not (math.isfinite(a) and 0 < a <= 360):
            raise ValidationError("angle must be in (0, 360]")
        full = abs(a - 360.0) <= 1e-9
        c = np.asarray(vec3(center, "center"), dtype=np.float64)
        columns = n if full else n + 1
        on_axis = prof[:, 0] == 0.0
        rings: list[list[int]] = []
        for (r, h), axis_point in zip(prof, on_axis, strict=True):
            if axis_point:
                rings.append([self._new_vertex(c + (0.0, 0.0, h))] * columns)
                continue
            ids = []
            for j in range(columns):
                phi = math.radians(a) * j / n
                ids.append(self._new_vertex(c + (r * math.cos(phi), r * math.sin(phi), h)))
            rings.append(ids)
        lengths = np.r_[0.0, np.cumsum(np.sqrt((np.diff(prof, axis=0) ** 2).sum(axis=1)))]
        vcoord = lengths / lengths[-1]
        faces = []
        for i in range(len(prof) - 1):
            lo, hi = rings[i], rings[i + 1]
            for j in range(n):
                j1 = (j + 1) % columns if full else j + 1
                u0, u1 = j / n, (j + 1) / n
                loop = [lo[j], lo[j1], hi[j1], hi[j]]
                uv = [(u0, vcoord[i]), (u1, vcoord[i]), (u1, vcoord[i + 1]), (u0, vcoord[i + 1])]
                keep = [k for k in range(4) if loop[k] not in loop[:k]]
                loop = [loop[k] for k in keep]
                if len(loop) < 3:
                    continue
                corners = {UV: [uv[k] for k in keep]} if UV in self._layers else {}
                faces.append(self._new_face(loop, corners, smooth=smooth))
        self._require_non_degenerate(faces, "lathe")
        verts = sorted({v for ring in rings for v in ring})
        return {"faces": Selection.of(faces=faces), "vertices": Selection.of(verts)}
