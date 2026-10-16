"""Face extrusion (region and individual) and inset."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..diagnostics import INFO, WARNING, Diagnostic
from ..errors import EditError, ValidationError
from ..math3d import vec3
from . import _polygon as poly
from .core import UV, Elements, Selection, operation


class ExtrudeMixin:
    @operation
    def extrude_faces(
        self,
        faces: Elements,
        distance: float | None = None,
        *,
        offset: Iterable[float] | None = None,
        mode: str = "region",
        space: str = "local",
    ):
        """Extrude faces, creating side walls along the region boundary.

        Give either ``distance`` (along the region's area-weighted average
        normal; negative pushes inward) or an ``offset`` vector. ``mode``:

        * ``"region"`` -- each connected group of faces moves as one piece
          (faces inside the group stay connected);
        * ``"individual"`` -- every face is extruded on its own along its normal.

        The extruded faces keep their ids (``data["faces"]``); the new walls are
        ``data["side_faces"]``; ``data["vertices"]`` maps each original boundary
        vertex to the new vertex now used by the extruded faces. Side-wall UVs
        are copied from the extruded face's boundary and are therefore
        degenerate (reported as ``UV_DEGENERATE_FACES``).
        """
        fids = self._elements(faces, "faces")
        if (distance is None) == (offset is None):
            raise ValidationError("give exactly one of distance or offset")
        if mode not in ("region", "individual"):
            raise ValidationError(f"mode must be 'region' or 'individual', got {mode!r}")
        if offset is not None:
            vector = np.asarray(vec3(offset, "offset"))
            if space == "world":
                vector = self._world_direction_to_local(vector)
            elif space != "local":
                raise ValidationError(f"space must be 'local' or 'world', got {space!r}")
            if not np.any(vector):
                raise EditError(
                    "ZERO_EXTRUSION", "the offset is zero", hint="give a non-zero offset"
                )
        else:
            if float(distance) == 0.0:
                raise EditError(
                    "ZERO_EXTRUSION", "the distance is zero", hint="give a non-zero distance"
                )
        regions = [[f] for f in fids] if mode == "individual" else self._components(fids)
        mapping: dict[int, int] = {}
        side_faces: list[int] = []
        for region in regions:
            direction = (
                vector if offset is not None else float(distance) * self._region_normal(region)
            )
            region_map, sides = self._extrude_region(region)
            mapping.update(region_map)
            side_faces += sides
            moved = sorted({v for f in region for v in self._faces[f]})
            self._set_positions(moved, self._pos[moved] + direction)
            if not sides:
                self._diagnose(
                    Diagnostic(
                        "EXTRUDE_WITHOUT_BOUNDARY",
                        INFO,
                        "the region has no boundary (a closed shell); it was moved",
                        None,
                        len(region),
                        tuple(region[:10]),
                    )
                )
        self._require_non_degenerate(side_faces + fids, "extrusion")
        if UV in self._layers and side_faces:
            self._diagnose(
                Diagnostic(
                    "UV_DEGENERATE_FACES",
                    WARNING,
                    "side faces created by extrusion copy the UVs of the extruded boundary and "
                    "have zero UV area",
                    None,
                    len(side_faces),
                    tuple(side_faces[:10]),
                )
            )
        return {
            "faces": Selection.of(faces=fids),
            "side_faces": Selection.of(faces=side_faces),
            "vertices": mapping,
        }

    @operation
    def inset_faces(
        self,
        faces: Elements,
        thickness: float,
        *,
        depth: float = 0.0,
        mode: str = "individual",
    ):
        """Inset faces: shrink them inward by ``thickness`` and fill the border with quads.

        ``thickness`` is the distance between the original and the new edges,
        measured within each face (corners are mitred, so borders have uniform
        width). ``depth`` then moves the inset faces along their normal.
        ``mode="individual"`` insets each face separately; ``"region"`` insets
        each connected group of faces along its outer boundary only.

        The inset faces keep their ids (``data["faces"]``); the border quads are
        ``data["side_faces"]``. UVs of new corners are interpolated within the
        original face, so the texture layout is preserved. Raises
        ``INSET_TOO_LARGE`` (and changes nothing) if the inset would invert or
        self-intersect a face.
        """
        fids = self._elements(faces, "faces")
        t = float(thickness)
        if not t > 0:
            raise ValidationError("thickness must be > 0")
        if mode not in ("region", "individual"):
            raise ValidationError(f"mode must be 'region' or 'individual', got {mode!r}")
        regions = [[f] for f in fids] if mode == "individual" else self._components(fids)
        side_faces: list[int] = []
        for region in regions:
            side_faces += self._inset_region(region, t, float(depth))
        return {"faces": Selection.of(faces=fids), "side_faces": Selection.of(faces=side_faces)}

    # -- shared region machinery ---------------------------------------------------

    def _components(self, fids: list[int]) -> list[list[int]]:
        pool = set(fids)
        regions = []
        while pool:
            start = min(pool)
            pool.discard(start)
            group, stack = [start], [start]
            while stack:
                f = stack.pop()
                for key in self._loop_edges(self._faces[f]):
                    for g in self._edge_faces[self._edge_by_pair[key]]:
                        if g in pool:
                            pool.discard(g)
                            group.append(g)
                            stack.append(g)
            regions.append(sorted(group))
        return regions

    def _region_normal(self, region: list[int]) -> np.ndarray:
        total = sum((poly.newell_normal(self._pos[self._faces[f]]) for f in region), np.zeros(3))
        n = poly.unit(total)
        if n is None:
            raise EditError(
                "UNDEFINED_NORMAL",
                "the faces' normals cancel out; give an explicit offset",
                elements={"faces": region},
            )
        return n

    def _world_direction_to_local(self, vector: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.inv(self.world_matrix[:3, :3]) @ vector
        except np.linalg.LinAlgError:
            raise EditError(
                "SINGULAR_WORLD_TRANSFORM",
                "the object's world transform is singular",
                hint="use space='local'",
            ) from None

    def _region_boundary_directed(self, region: list[int]) -> list[tuple[int, int, int]]:
        """Boundary edges of a region as ``(a, b, face)`` with the face running a -> b."""
        members = set(region)
        boundary = []
        for f in region:
            loop = self._faces[f]
            n = len(loop)
            for k in range(n):
                a, b = loop[k], loop[(k + 1) % n]
                users = [g for g in self._edge_faces[self._edge_id(a, b)] if g in members]
                if len(users) == 1:
                    boundary.append((a, b, f))
                elif len(users) == 2:
                    g = users[0] if users[1] == f else users[1]
                    lg = self._faces[g]
                    if lg[(lg.index(a) + 1) % len(lg)] == b:
                        raise EditError(
                            "INCONSISTENT_WINDING",
                            f"faces {f} and {g} are wound inconsistently",
                            elements={"faces": [f, g]},
                            hint="flip one of them first",
                        )
                else:
                    raise EditError(
                        "NOT_MANIFOLD",
                        "the region has an edge shared by more than two of its faces",
                        elements={"faces": users},
                    )
        return boundary

    def _extrude_region(self, region: list[int]) -> tuple[dict[int, int], list[int]]:
        boundary = self._region_boundary_directed(region)
        corners = {f: self._corners_of(f) for f in region}
        # Region-boundary vertices need copies even where no other face uses them
        # (open surfaces), so that side walls can connect old and new positions.
        outline = {v for a, b, _ in boundary for v in (a, b)}
        mapping = self._detach_region(region, also=outline)
        sides = []
        for a, b, f in boundary:
            loop = self._faces[f]
            ca = corners[f][loop.index(mapping.get(a, a))]
            cb = corners[f][loop.index(mapping.get(b, b))]
            side_corners = self._split_corners([ca, cb, cb, ca])
            sides.append(
                self._new_face(
                    [a, b, mapping[b], mapping[a]],
                    side_corners,
                    smooth=self._smooth[f],
                    group=self._group.get(f, 0),
                )
            )
        return mapping, sides

    def _inset_region(self, region: list[int], thickness: float, depth: float) -> list[int]:
        boundary = self._region_boundary_directed(region)
        if not boundary:
            raise EditError(
                "UNSUPPORTED_TOPOLOGY",
                "the region has no boundary to inset (closed shell)",
                elements={"faces": region},
            )
        outgoing: dict[int, tuple[int, int]] = {}
        incoming: dict[int, tuple[int, int]] = {}
        for a, b, f in boundary:
            if a in outgoing or b in incoming:
                raise EditError(
                    "UNSUPPORTED_TOPOLOGY",
                    f"the region boundary passes through vertex "
                    f"{a if a in outgoing else b} more than once",
                    elements={"faces": region},
                    hint="inset the faces individually",
                )
            outgoing[a] = (b, f)
            incoming[b] = (a, f)
        members = set(region)
        boundary_vertices = set(outgoing)
        for f in region:
            for v in self._faces[f]:
                if v not in boundary_vertices and any(
                    g not in members for g in self._faces_of_vertex(v)
                ):
                    raise EditError(
                        "UNSUPPORTED_TOPOLOGY",
                        f"vertex {v} touches other faces without being on the region boundary",
                        elements={"vertices": [v]},
                    )
        original_normal = {f: self._face_normal_vec(f) for f in region}
        flat = [f for f, n in original_normal.items() if n is None]
        if flat:
            raise EditError(
                "DEGENERATE_INPUT",
                "faces with zero area cannot be inset",
                elements={"faces": flat},
            )
        fits = {
            f: {
                name: poly.affine_fit(self._pos[self._faces[f]], np.asarray(layer.data[f]))
                for name, layer in self._layers.items()
            }
            for f in region
        }
        targets: dict[int, np.ndarray] = {}
        for v in sorted(boundary_vertices):
            p, f1 = incoming[v]
            n, f2 = outgoing[v]
            n1, n2 = original_normal[f1], original_normal[f2]
            if n1 is None or n2 is None:
                raise EditError(
                    "DEGENERATE_INPUT",
                    "a face in the region has zero area",
                    elements={"faces": [f1, f2]},
                )
            d1 = poly.unit(poly.cross3(n1, self._pos[v] - self._pos[p]))
            d2 = poly.unit(poly.cross3(n2, self._pos[n] - self._pos[v]))
            m = poly.unit(d1 + d2) if d1 is not None and d2 is not None else None
            if m is None or float(np.dot(m, d1)) < 1e-3:
                raise EditError(
                    "INSET_TOO_LARGE",
                    f"the boundary folds back at vertex {v}; the inset is undefined there",
                    elements={"vertices": [v]},
                )
            targets[v] = self._pos[v] + thickness / float(np.dot(m, d1)) * m

        mapping, sides = self._extrude_region(region)
        new_ids = [mapping[v] for v in sorted(targets)]
        self._set_positions(new_ids, np.array([targets[v] for v in sorted(targets)]))
        # Interpolate attributes of the moved corners within each original face.
        for f in region:
            loop = self._faces[f]
            corners = self._corners_of(f)
            changed = False
            for k, v in enumerate(loop):
                if v in new_ids:
                    for name, fit in fits[f].items():
                        value = fit(self._pos[v][None])[0]
                        if self._layers[name].normalize:
                            u = poly.unit(value)
                            value = u if u is not None else np.asarray(corners[k][name])
                        corners[k][name] = tuple(value.tolist())
                    changed = True
            if changed:
                for name in self._layers:
                    self._set_corners(f, name, [c[name] for c in corners])
        # Border quads: bottom corners already carry the original face's values; the top
        # corners take the interpolated values of the inset face that owns the inner edge.
        for side in sides:
            owner = self._side_owner(side, region)
            side_loop, owner_loop = self._faces[side], self._faces[owner]
            for name, layer in self._layers.items():
                values = list(layer.data[side])
                for k in (2, 3):
                    values[k] = layer.data[owner][owner_loop.index(side_loop[k])]
                self._set_corners(side, name, values)

        for f in region:
            n = self._face_normal_vec(f)
            if (
                n is None
                or float(np.dot(n, original_normal[f])) <= 0.0
                or not poly.is_simple(poly.project(self._pos[self._faces[f]], original_normal[f]))
            ):
                raise EditError(
                    "INSET_TOO_LARGE",
                    f"thickness {thickness} inverts or self-intersects face {f}",
                    elements={"faces": [f]},
                    hint="use a smaller thickness",
                )
        # Every inset edge must keep the direction of the edge it was offset from; a
        # reversed edge means the inset passed through the face's opposite side (which
        # can otherwise look like a valid, merely rotated polygon).
        for a, b, _f in boundary:
            new_a, new_b = mapping[a], mapping[b]
            if float(np.dot(self._pos[new_b] - self._pos[new_a], self._pos[b] - self._pos[a])) <= 0:
                raise EditError(
                    "INSET_TOO_LARGE",
                    f"thickness {thickness} collapses the face across edge {a}-{b}",
                    elements={"faces": region},
                    hint="use a smaller thickness",
                )
        self._require_non_degenerate(sides, "inset", code="INSET_TOO_LARGE")
        if depth:
            direction = depth * self._region_normal(region)
            moved = sorted({v for f in region for v in self._faces[f]})
            self._set_positions(moved, self._pos[moved] + direction)
            self._require_non_degenerate(sides, "inset")
        return sides

    def _side_owner(self, side: int, region: list[int]) -> int:
        """The region face that shares the inner edge of a side quad."""
        _, _, b2, a2 = self._faces[side]
        e = self._edge_id(b2, a2)
        members = set(region)
        return next(g for g in self._edge_faces[e] if g in members)

    def _require_non_degenerate(
        self, fids: list[int], what: str, code: str = "DEGENERATE_RESULT"
    ) -> None:
        eps = self._area_eps()
        bad = sorted(
            f
            for f in fids
            if f in self._faces and poly.polygon_area(self._pos[self._faces[f]]) <= eps
        )
        if bad:
            raise EditError(
                code,
                f"the {what} would create {len(bad)} zero-area faces",
                elements={"faces": bad},
                hint="check the distance/offset direction",
            )
