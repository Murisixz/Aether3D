# Advanced modeling and sculpting

Aether3D 0.13 extends the polygon editing engine ([EDITING.md](EDITING.md)) with
advanced modeling tools. This guide explains how they work, what they accept and refuse,
and why they were built the way they were. The design record written before the
implementation, with its research and measurements, is
[research/2026-10-stage13/DESIGN.md](../research/2026-10-stage13/DESIGN.md).

## Contents

1. [Rounded bevels](#rounded-bevels)
2. [Planar cuts and knife cuts](#planar-cuts-and-knife-cuts)
3. [Filling and bridging](#filling-and-bridging)
4. [Slides](#slides)
5. [Dissolving, connecting and cleaning up](#dissolving-connecting-and-cleaning-up)
6. [Shading and orientation](#shading-and-orientation)
7. [Sweeps](#sweeps)
8. [Shrinkwrap and surface queries](#shrinkwrap-and-surface-queries)
9. [Procedural creases and mirror bisect](#procedural-creases-and-mirror-bisect)

## Rounded bevels

`bevel_edges(edges, width, segments=1, profile=0.5)` replaces each edge by a strip of
`segments` faces. With one segment it is the chamfer of earlier versions, computed by the
same code. In every face next to a bevelled edge the new boundary runs parallel to the
edge at perpendicular distance `width`. Where two bevelled edges meet inside a face the
corner moves to the intersection of their offset lines (a mitre); where a bevelled edge
meets an unbevelled one, the corner slides along it.

```python
from aether3d.editing import EditableMesh

m = EditableMesh()
m.add_box(2)
result = m.bevel_edges(m.select_all(), 0.3, segments=6)  # a rounded box
print(len(result.data["faces"].faces), len(result.data["corner_faces"].faces))  # 72 216
print(m.validate().ok)  # True
```

### Profiles

At each end of a bevelled edge, the strip's cross-section is a **profile**: `segments + 1`
points from the edge's point in one face (`start`) to its point in the other face (`end`).
The profile is a quadrant of the superellipse `x^r + y^r = 1`, mapped affinely onto the
parallelogram with corners `start`, `middle`, `end`:

`P(x, y) = O + x (middle - start) + y (middle - end)`, with `O = start + end - middle`,

so `(0, 1)` is `start`, `(1, 0)` is `end`, and the square's corner `(1, 1)` would be
`middle`. The exponent comes from `profile` as in Blender, `r = -2 ln 2 / ln(profile)`:

| `profile` | `r` | Cross-section |
| --- | --- | --- |
| 0.5 (default) | 2 | an elliptical arc tangent to both faces; a circular arc of radius `width` where the faces meet at a right angle |
| 0.25 | 1 | the chord: a chamfer split into `segments` faces |
| toward 1 | toward infinity | toward the original sharp corner |
| toward 0 | toward 0 | concave |

Points are spaced evenly in angle on the circle (`r = 2`) and by arc length otherwise.

**The middle point** is where the two faces' tangent lines meet: where the edge's line
crosses the profile's plane.

- **Slide ends.** If either end of the profile slides along an unbevelled edge, the plane
  contains the original vertex, and `middle` is that vertex. This covers a single bevelled
  edge, whose end faces get rounded corners, and the mitred ends of a bevelled path. There
  the profile is exactly where the fillet meets the mitre plane.
- **Mitre ends.** If both ends are mitre points (a vertex all of whose nearby edges are
  bevelled), the plane is the edge's cross-section, as perpendicular to the edge as the
  two points allow. The strip is then a true fillet all the way to the corner patch.

With `profile=0.5` and equal widths, the bevels of a box are exact:

- every strip is a quarter cylinder of radius `width`;
- every corner is an octant of a sphere of radius `width`;
- every vertex lies on the rounded box's implicit surface to within 1e-12;
- the volume converges to the analytic `s³ - 12 (s - 2w) w² (1 - π/4) - 8 w³ (1 - π/6)`
  as O(1/n²).

`tests/test_modeling_bevel.py` checks all of these.

### Corners

Around each bevelled vertex, the new points of its faces and the profiles' inner points
form the corner's outline. It closes in the first of these ways that applies:

1. **Nothing to close.** The outline folds back on itself, as when two strips share a
   profile along a bevelled path.
2. **Merged into a face.** The outline is one profile and its chord, and a face already
   runs along the chord. That face takes the profile, which lies in its plane: the end
   faces of a single bevelled edge get rounded corners.
3. **Flat corner.** The outline has zero area and all its points lie on one line, as at
   a vertex in a flat region. The edges on both sides of the line are subdivided by each
   other's points, so no T-junction remains. A straight profile that passes through the
   vertex reuses the vertex. This generalizes the chamfer's merge into the chamfer face.
4. **Polygon.** The outline is planar and simple (any simple outline for a chamfer): one
   polygon closes it.
5. **Grid patch.** Every edge of the vertex is bevelled and `segments >= 2`.
   - Rings of quads close it, each ring two segments shorter per side, around a center
     vertex (even `segments`) or a center polygon (odd).
   - Inner points interpolate the outline with mean value coordinates (Floater 2003),
     plus a bulge matching the profiles'.
   - When the outline lies on a sphere, the inner points are projected onto it. A box
     corner with circular profiles is therefore an exact sphere octant.
6. **Fan.** Otherwise, triangles around the outline's centroid close it. A triangle that
   would fold raises `BEVEL_UNSUPPORTED_VERTEX`.

**Open boundaries.** A vertex may lie on an open boundary if its faces form one manifold
chain (an open fan). Its outline is then an open chain. It is closed along the boundary
(by a polygon or a fan) only if it covers part of the original surface, that is, if it
contains a point that is not on a profile. A corner made only of a profile stays open, so
the boundary follows the arc. A vertex where two fans meet (a bowtie) is refused.

### Attributes, groups and shading

- **UVs of rebuilt faces** come from an affine fit over the original face. This is exact
  for affine UVs on planar faces.
- **Strip corners** interpolate between the two faces' values along the profile.
- **Patch and fan points** interpolate the outline's values with the same weights as
  their positions.
- **Groups.** Strips take the group of the edge's lower-id face. Corners take the group of
  the lowest-id face around the vertex (as for chamfers).
- **Shading.** Strips and corners of bevels with more than one segment are smooth-shaded.
  A chamfer keeps the earlier rule (smooth if both faces are).

### Refusals

Every refusal raises an `EditError` and leaves the mesh exactly as it was:

| Code | Cause |
| --- | --- |
| `NOT_MANIFOLD` | a bevelled edge does not have exactly two faces |
| `BEVEL_UNSUPPORTED_VERTEX` | a vertex on a non-manifold edge, with faces in several fans, with fewer than 3 edges, or whose corner would fold |
| `BEVEL_WIDTH_TOO_LARGE` | the width folds a face, reverses an edge, collapses faces or makes zero-area ones |
| `ValidationError` | `width` not finite and positive, `segments` not an integer in 1..64, `profile` not strictly between 0 and 1 |

The procedural `bevel` modifier takes the same `segments` and `profile`
([PROCEDURAL.md](PROCEDURAL.md#bevel)).

## Planar cuts and knife cuts

**Bisect.** `bisect(point, normal, faces=None, clear="none", fill=False, threshold=None)`
cuts faces with a plane.

1. **Classify each vertex once.** Each vertex of the cut faces gets its signed distance
   `s = n · (p - point)`. It is on the plane if `|s| <= threshold` (default `1e-6` times
   the mesh's size, so float32 positions read from a `Mesh` that lie on the plane
   count as on it). Deciding once per vertex keeps every later decision consistent: an
   edge is crossed exactly when its two ends are strictly on opposite sides.
2. **Split the crossed edges** at `t = s_a / (s_a - s_b)`, and project the new vertex
   onto the plane.
3. **Split each face by its chord.** The chord joins the end of the on-plane run before
   the face's front stretch to the start of the run after it. A face crossed more than
   twice (a concave polygon) is triangulated first (`BISECT_TRIANGULATED`).

Faces outside `faces` only gain the cut points on edges they share with cut faces, so
there are no T-junctions. `clear` deletes the cut faces on one side; faces lying in the
plane stay. `fill` closes each boundary loop the clearing leaves in the plane: a polygon
when it is planar and simple, otherwise a triangulation.

**Knife.** `knife_project(points, direction, closed=False, through=True)` cuts along a
polyline swept along `direction`.

- **Ribbons.** Each segment `P_i P_i+1` spans a planar ribbon
  `{P_i + s (P_i+1 - P_i) + t d}`, `0 <= s <= 1`.
- **Pieces and paths.** In each face, the ribbons leave segments. Their ends lie on the
  face's boundary, or on the polyline's corners projected into the face. Pieces sharing a
  corner join into paths.
- **Splitting.** A path that enters and leaves a face splits it, with the corners as new
  vertices. Several paths split it into several pieces. Points on edges are shared with
  the neighbouring face (no T-junctions). Concave faces the cut crosses are triangulated
  first.
- **Dangling cuts.** A path that ends inside a face is not applied, because a polygon
  mesh cannot hold a dangling edge (`KNIFE_PARTIAL_CUT`).
- **`through=False`** cuts only faces whose front faces the cut, a simple criterion that
  does not test occlusion.
- **Attributes.** UVs of new vertices on edges are interpolated along the edge. Inside a
  face they come from the face's affine fit.

## Filling and bridging

Both work on **boundary loops**, chains of edges with one face each. A new face must run
each boundary edge opposite to the face already there, which fixes the winding of every
fill and bridge face. Several loops touching at one vertex are refused
(`AMBIGUOUS_BOUNDARY`) rather than guessed.

`fill_holes(edges=None, mode="auto", max_sides=0)` has four modes:

| `mode` | Result |
| --- | --- |
| `"ngon"` | one polygon (the loop must be simple seen along its normal) |
| `"fan"` | triangles around the loop's centroid |
| `"triangulate"` | the minimum-weight triangulation of Barequet and Sharir, with Liepa's weight ("Filling Holes in Meshes", SGP 2003): the largest dihedral angle between adjacent triangles (including the faces around the hole) first, then the total area; dynamic programming, O(n³), up to 64 vertices |
| `"auto"` | a polygon if the loop is planar and simple, a triangulation up to 64 vertices, else a fan |

The new corners take the UVs of the boundary they close. Faces whose UVs then have zero
area are reported (`UV_DEGENERATE_FACES`), as for extrusion walls.

`bridge_edge_loops(edges, segments=1, twist=0)` joins two boundary loops, or two open
chains, with a strip:

- **Pairing.** Oriented by their new faces, the two loops run in opposite directions
  across the bridge, so loop A's `a_i` meets loop B's `b_(offset - i)`. The offset that
  minimizes the summed squared rung lengths is used, and `twist` adds to it.
- **Quads or triangles.** Equal vertex counts give `segments` rows of quads, with
  interpolated intermediate rows. Different counts give one row of triangles that
  advance along whichever loop has the smaller normalized arc length next.

## Slides

`slide_edges(edges, factor, even=False)` moves chains of edges (loops or open chains)
along their **rails**.

- **Rails.** A rail is the edge leaving the chain at a vertex on the chosen side.
- **Sides.** Side `+1` holds the faces that run the chain's direction, walking from its
  lowest-id end (or, for a closed loop, from its lowest-id vertex toward its lower-id
  neighbour).
- **Distance.** `factor` in `[-1, 1]` moves each vertex that fraction along its rail.
  With `even=True`, every vertex moves `|factor|` times the shortest rail.
- **Poles.** At a pole, the rail is the edge most nearly perpendicular to the chain.

`slide_vertex(vertex, toward, factor)` moves one vertex along one edge.

- **UVs.** With `correct_uvs` (the default), a vertex's corners in the faces containing
  its rail are interpolated along the rail, which is exact. Its other corners take the
  same value where the vertex has a single UV. Corners across a UV seam keep their
  values and stretch, which is reported (`UV_SEAM_STRETCHED`).
- **Folds.** A slide that would fold a face is refused (`SLIDE_FOLDS_FACE`).

## Dissolving, connecting and cleaning up

- **`dissolve_vertices`.** A vertex on two edges leaves its faces' loops. Any other vertex
  must have one manifold fan, and its faces merge into one polygon: the concatenation of
  each face's outline without the vertex, wound like the faces.
- **`dissolve_faces`.** Each connected region becomes its outline. A region with holes
  or a pinched outline is refused (`DISSOLVE_INVALID_FACE`).
- **`limited_dissolve(angle)`.** Dissolves edges between faces whose normals differ by
  at most `angle` degrees, smallest angle first, then vertices in the middle of straight
  runs. Edges between face groups, across UV or normal seams, and sharp edges are kept
  (`delimit`).
- **`connect_vertex_path` and `connect_vertices`.** Split faces between vertices that
  share a face.
- **`cleanup`.** Runs explicit repairs in a fixed order: merge by distance, collapse
  zero-length edges, delete duplicate faces, recalculate winding, fill holes, delete
  loose vertices. It returns a report with what each step changed (`fixed`, with ids).
  Zero-area faces and non-manifold edges are only counted (`remaining`), because
  repairing them needs a decision `cleanup` does not guess.

## Shading and orientation

- **`set_auto_smooth(angle=30)`.** Smooth-shades faces and marks an edge sharp exactly
  when its faces' normals differ by more than `angle`.
- **`recalculate_normals(inside=False)`.**
  - **Consistent winding.** Within each piece connected by manifold edges, a
    breadth-first walk flips every face that runs a shared edge the same way as its
    neighbour.
  - **Outward.** A closed piece is turned outward by its signed volume. An open piece
    has no inside, so its faces face away from its centroid by area-weighted majority,
    which is reported (`ORIENTATION_HEURISTIC`).
  - **Non-orientable pieces.** A Moebius strip cannot be oriented; it is left unchanged
    and reported (`NON_ORIENTABLE`).
- **`set_weighted_normals(mode="face_area", weight=50)`.** Writes custom corner normals.
  - Within each smooth fan, the normal is the sum of the fan's face normals weighted by
    `area ** (weight / 50)`, `corner angle ** (weight / 50)`, or their product.
  - `weight=50` is plain area weighting. Larger weights let big faces dominate, which
    keeps the large flat faces of a bevelled hard-surface model flat.
  - The normals are kept at commit, and a later geometric edit of a face recomputes
    them as usual.

## Sweeps

- **`extrude_edges(edges, offset)`.** Builds walls from boundary edges, wound like the
  face each edge borders.
- **`spin(faces, angle, steps, axis, center, merge=True)`.** Extrudes a region `steps`
  times, rotating it by `angle / steps` each time. A merged full turn welds the last
  ring to the first and removes the region at the end: spinning a square profile makes
  a closed torus. The volume equals the polygonal solid of revolution's,
  `(n / 2) sin(2π / n) ∫(r_out² - r_in²) dz`, to 1e-12.
- **`add_lathe(profile, segments=32, angle=360)`.** Builds a surface of revolution from
  `(radius, height)` points around the local Z axis.
  - Faces: quads between rings, and triangle fans where the profile touches the axis.
  - UVs: `u` around the axis, `v` along the profile's arc length.
  - A profile running from bottom to top faces outward.

## Shrinkwrap and surface queries

`shrinkwrap(vertices, target, mode="nearest", direction=None, offset=0, max_distance=None)`
moves vertices onto a target. The target is an immutable `Mesh` or, in an editing
session or edit script, another scene object (both objects' world transforms apply).

- **`"nearest"`** uses the exact closest point.
- **`"project"`** uses the first hit along `direction`, both ways unless `both_ways=False`.
- **`offset`** moves the vertex along the target face's normal, on the side the vertex
  came from.
- **Misses.** Vertices without a target point within `max_distance` stay where they are
  (`SHRINKWRAP_MISSED`).

The queries use `aether3d.spatial.surface.TriangleSurface`, a float64 copy of the target
with a linear BVH built from the Morton order of its triangles (`build_bvh_morton`).

- **Build.** Leaf boxes are reduced once and inner boxes merged bottom-up. For 4.2M
  triangles the build takes 0.88 s instead of 9.9 s for the renderers' binned-SAH tree.
- **Nearest.** One traversal stack per point, run in lockstep: nodes farther than the
  best distance so far are skipped, and the nearer child is visited first. The result
  is exact; it matches brute force bit for bit in the tests.
- **Rays.** Möller-Trumbore in float64, front to back, closest hit with `tmin < t < tmax`.

## Procedural creases and mirror bisect

- **Creases.** The `subdivision` modifier's `crease` (0-10, default 0, which keeps the
  previous results bit for bit) is the sharpness of the edges marked sharp (DeRose, Kass
  and Truong 1998).
  - At sharpness 1 or more, a creased edge follows the boundary rules. Its edge point is
    the midpoint; a vertex on two creases moves by `(e1 + 6 p + e2) / 8`; a vertex on
    three or more stays.
  - Each level lowers the sharpness by 1, and values below 1 blend the sharp and smooth
    rules.
  - Primitives with `smooth=True` mark their rims sharp, so
    `subdivision(levels=3, crease=4)` keeps a cylinder's caps flat and its rims crisp.
- **Mirror bisect.** The `mirror` modifier's `bisect` first cuts the geometry at each
  mirror plane with the editing engine's bisect and keeps the positive side (or, with
  `bisect_flip`, the negative one). Geometry crossing the plane no longer overlaps its
  mirror image.
