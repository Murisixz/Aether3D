# Advanced modeling and sculpting

Aether3D 0.13 extends the polygon editing engine ([EDITING.md](EDITING.md)) with
advanced modeling tools. This guide explains how they work, what they accept and refuse,
and why they were built the way they were. The design record written before the
implementation, with its research and measurements, is
[research/2026-10-stage13/DESIGN.md](../research/2026-10-stage13/DESIGN.md).

## Contents

1. [Rounded bevels](#rounded-bevels)

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
