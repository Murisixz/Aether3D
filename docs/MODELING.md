# Advanced modeling and sculpting

Aether3D 0.13 extends the polygon editing engine ([EDITING.md](EDITING.md)) with
advanced modeling tools, and adds a sculpting system: brushes with masks, symmetry and
pressure, dynamic topology, isotropic and voxel remeshing, symmetrize and measured
validity. This guide explains how they work, what they accept and refuse, and why they
were built the way they were. The design record written before the
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
10. [Sculpting: quick start](#sculpting-quick-start)
11. [The sculpt mesh](#the-sculpt-mesh)
12. [Brushes and strokes](#brushes-and-strokes)
13. [Dynamic topology](#dynamic-topology)
14. [Remeshing](#remeshing)
15. [Symmetrize](#symmetrize)
16. [Validity](#validity)
17. [Sculpt scripts](#sculpt-scripts)
18. [Command line, MCP and visual feedback](#command-line-mcp-and-visual-feedback)
19. [History](#history)
20. [Design and languages](#design-and-languages)
21. [Performance](#performance)
22. [Limitations](#limitations)
23. [References](#references)

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

## Sculpting: quick start

Sculpting works on an object's mesh in a **session**: brush strokes, masks, dynamic
topology, remeshing and symmetrize run on a sculpt mesh, each as one step of the
session's own undo, and committing assigns the result as one step of the scene's history.

```python
from aether3d import Scene, create_sphere

scene = Scene()
head = scene.add_object("Head", mesh=create_sphere(1.0, 64, 32))
with head.sculpt() as s:  # commits when the block ends (discards on an exception)
    s.voxel_remesh(0.04)  # an even base without poles: new topology
    s.remesh(0.045, iterations=3)  # isotropic: even triangles near 0.045
    s.set_mask(1.0, box=((-2, -2, -2), (2, 2, -0.6)))  # the neck stays (local coordinates)
    r = s.stroke(
        {
            "brush": {"kind": "draw", "radius": 0.25, "strength": 0.2, "symmetry": "x"},
            "path": [[0.3, -0.9, 0.3], [0.5, -0.8, 0.4]],
        }
    )
    print(r.data["dabs"], r.data["moved_vertices"] > 0)  # 11 True
    print([h["operation"] for h in s.history])  # ['voxel_remesh', 'remesh', 'set_mask', 'stroke']
print(s.result.report.summary["tier"])  # solid
scene.history.undo()  # the whole session was one step
print(head.mesh.triangle_count)  # 3968
```

The same operations run from JSON (a sculpt script, below), the command line (`aether3d
sculpt run`) and MCP (`run_script`). A procedural object is refused
(`PROCEDURAL_OBJECT`): sculpt its base mesh with `obj.geometry.sculpt_source()` (the
modifiers stay non-destructive) or `bake()` it first.

## The sculpt mesh

A `Mesh` splits a vertex wherever its UVs or normals differ (a seam). Sculpting must move
those copies together, so `SculptMesh` keeps geometry and attributes apart:

* **welded vertices** -- one per distinct position, in growable float64 arrays;
* **render vertices** -- the mesh's own vertices, each pointing at its welded vertex and
  carrying its UV, its stored normal and its *normal group* (render vertices of one
  welded vertex with identical normals share a group: smooth surfaces stay smooth across
  UV seams, hard edges stay hard);
* **triangles** -- welded and render corners, the face group, an alive flag.

Triangles are partitioned into **chunks** of at most 1024 by the Morton codes of their
centroids (each chunk an octree cell), each with a box around its vertices. A brush asks
the chunks its sphere reaches for candidates, so a dab costs what its neighbourhood
costs, not what the mesh costs. Per triangle the cross product `(b - a) x (c - a)` and
per vertex the sum of its triangles' cross products are kept up to date incrementally:
area-weighted normals at any time without a pass over the mesh.

**History.** Every change goes through journaled primitives (positions, masks, the
changed flags, the derived normal data, topology changes as sparse records, a whole-mesh
replacement as a reference to the previous arrays). If an operation raises -- an invalid
setting, a limit, `should_stop` -- its journal is replayed backwards: nothing is left
half done. `undo()` replays the last operation's journal backwards, bit for bit;
`redo()` runs it again with the same arguments (everything is deterministic).

**Commit.** `to_mesh()` writes the render vertices back. Positions come from the welded
vertices; only the normal groups whose geometry changed get new normals (angle weighted
over the group's triangles); UVs, triangles and groups are reused. A mesh nothing
changed commits as the very object it was made from. After topology changes the alive
elements are renumbered in order.

## Brushes and strokes

A stroke is a brush and a path:

```python
s.stroke(
    {
        "brush": {
            "kind": "clay",
            "radius": 0.2,
            "strength": 0.4,
            "falloff": "smooth",
            "symmetry": "x",
            "pressure": "both",
            "automask": ["topology"],
        },
        "path": [
            {"position": [0.3, -0.9, 0.0], "pressure": 0.3},
            {"position": [0.5, -0.7, 0.1], "pressure": 1.0},
        ],
    }
)
```

Every brush is a closed-form displacement of the vertices within `radius` of the dab
centre `c`, weighted per vertex by `w = falloff(d / radius) * pressure * (1 - mask) *
automask`. `N` and `P` are the area normal and area centre (the `w`-weighted mean of the
vertex normals and positions within `normal_radius * radius`), `s` the strength, `r` the
radius and `sign` the direction (`+1`, or `-1` to invert).

| Brush | What a dab does |
| --- | --- |
| `draw` | `sign * s * r * w * N`: raises or lowers along the area normal |
| `inflate` | `sign * s * r * w * n_v`: along each vertex's own normal (in a concave valley the walls meet: prefer `draw` there) |
| `clay` | toward a plane `s * r * 0.5` above the area plane: vertices below it close half the gap per dab |
| `flatten`, `fill`, `scrape` | `-s * w * h * N` with `h = (p - P) . N`: toward the area plane (`fill`: only vertices below it, `scrape`: only above) |
| `pinch` | toward the centre within the tangent plane |
| `crease` | pinch at half strength plus draw at half strength (`sign = -1` by default: a groove) |
| `layer` | raises to a fixed height `sign * s * r` above the stroke's starting surface, never beyond |
| `grab` | the vertices under the first dab follow the stroke (`p = p0 + w0 * (c - c0)`) |
| `elastic` | a regularized Kelvinlet (de Goes and James 2017): the elastic response to the stroke's displacement, Poisson ratio `poisson_ratio` |
| `snake_hook` | grab taken again at every dab: pulls geometry out along the path |
| `twist` | rotation about `N` through `c` by `sign * s * w * 90` degrees |
| `smooth` | `s * w * L(p)` with `L` the umbrella Laplacian (`laplacian`), a Taubin pair (`taubin`, no shrinkage) or its tangential part (`relax`: evens vertices out without changing the shape); open boundaries follow their boundary neighbours |
| `mask` | paints the mask: `mask += sign * s * w` |

Strokes and their settings:

* **Lengths are world units** (radius, detail, displacements): a brush acts the same way
  on any object transform, including non-uniform scales and mirrors; normals are mapped
  by the inverse transpose.
* **Dabs** are placed every `spacing * radius` along the path (default 10 %), pressure
  interpolated; `"resample": false` uses the path's points as dabs. A stroke has at most
  200,000 dabs and 100,000 path samples.
* **Projection.** `"project": "nearest"` (default) moves each dab onto the closest point
  of the current surface (exact, chunked); `"ray"` casts it along `"view"`; `"none"`
  uses the points as given. `grab`, `elastic` and `snake_hook` project only their first
  dab: the later ones follow the path's own motion from there, so they pull geometry off
  the surface.
* **Falloffs**: `smooth`, `smoother`, `sphere`, `root`, `sharp`, `linear`, `constant`, or
  a custom curve of up to 64 `[x, y]` points; `hardness` keeps full strength up to a
  fraction of the radius. `"falloff_shape": "projected"` measures distances across the
  view direction (a cylinder through the surface), `"geodesic": true` along the surface
  (the fast iterative method of Fu et al. 2011: nothing reaches a finger next to the one
  being sculpted), `"front_faces_only": true` skips surfaces facing away from the view.
* **Pressure is data**: each path point may carry one (0..1); `"pressure"` says whether
  it scales the `strength`, the `radius`, `both` or `none`.
* **Symmetry**: `"symmetry": "x"` (any of `xyz`) mirrors every dab across the object's
  local planes through its origin. All mirrored dabs of one sample are computed from the
  same positions and their displacements added, so the result does not depend on their
  order and a symmetric mesh stays exactly symmetric; vertices on a mirror plane stay on
  it. A dab on a mirror plane coincides with its mirror and acts twice, as in other
  sculpting tools.
* **Masks**: `set_mask(value, vertices=, sphere=, box=, groups=, mode=)`,
  `invert_mask()`, `clear_mask()` (local coordinates); the `mask` brush paints it.
  Masked vertices (1) do not move under brushes, smoothing, relaxation or symmetrize's
  `positions` mode, and dynamic topology leaves them alone (symmetrize's `mirror` mode
  rebuilds the other side whatever its mask). `"use_mask": false` ignores the mask.
* **Automasking**: `"topology"` (only the surface connected to the dab's centre),
  `"face_groups"` (only its face group), `"boundary"` (open edges held).

`stroke()` returns `data` (dabs made, applied and skipped, vertices moved, the largest
displacement; with dynamic topology the splits, collapses and flips and the triangle
count), `timings` per stage (`query`: finding the vertices, `deform`: the brush,
`apply`: moving them and updating the normal data, `topology`: dynamic topology) and
diagnostics (`DABS_SKIPPED` when ray-projected dabs miss the surface,
`STROKE_NO_EFFECT` when no vertex was within reach).

## Dynamic topology

`"dyntopo": {"detail": 0.25}` on a brush (not `grab`, `elastic`, `layer` or `mask`)
changes the topology before each dab, so the detail follows the brush:

1. **Splits** inside the brush: every edge longer than `4/3 * L` (`L` = `detail * radius`,
   or `detail` in world units with `"mode": "constant"`) is split at its midpoint;
   triangles with one, two or three split edges are refined red-green (Bank, Sherman and
   Weiser 1983), up to `max_splits` per dab. A new vertex's UVs are interpolated within
   each face along its edge, so the texture mapping does not change; its mask is
   interpolated.
2. **Collapses** (`"refine": "both"` or `"collapse"`) of edges shorter than `4/5 * L`, in
   and around the brush: an independent set chosen in Luby-style rounds (Luby 1986) with
   deterministic scrambled priorities, after every candidate passes, vectorized: the link
   condition (the two vertices share exactly their two opposite vertices), valences above
   3, no new edge reaching the split length, no surrounding triangle turning by 60
   degrees or more or collapsing to zero area.
3. **Delaunay flips** (`"flip": true`) of edges whose two opposite angles sum to more
   than 180 degrees, unless the new triangles would turn by 60 degrees or more or the
   new edge exists already, and
4. **Tangential relaxation** (`"relax"`, default 0.5): vertices move that fraction of the
   way toward the area-weighted centroid of their neighbours, within their tangent
   plane, their UVs re-sampled from the face they move into.

**Features are kept**: open boundaries, UV seams, hard edges (normal-group borders),
face-group borders and non-manifold edges are split along but never collapsed across,
flipped or relaxed off; masked vertices and degenerate triangles are not touched. Splits
cannot fold a surface; collapses and flips are checked before they happen (above);
relaxation moves each vertex part of the way within its tangent plane, unchecked. Every
pass is journaled, so a stroke with dynamic topology undoes exactly like any other, and
the tests check validity (closed, manifold, oriented, no self-intersection) after
dynamic-topology strokes and remeshes.

## Remeshing

**Isotropic remeshing** (Botsch and Kobbelt, "A Remeshing Approach to Multiresolution
Modeling", SGP 2004): `remesh(target_length, iterations=5, flips="valence", relax=0.5,
project=True)` repeats split, collapse, flip (toward valence 6, or Delaunay) and
tangential relaxation, then projects the free vertices back onto the surface the remesh
started from (exact nearest points). It keeps the same features as dynamic topology:
UV seams stay seams, hard edges and group borders stay, a cube keeps its eight corners
and its volume exactly. The result is measured: `data` holds the splits, collapses,
flips and relaxed vertices and the edge-length statistics. A target that would need more
triangles than the limit is refused before anything changes (`REMESH_TOO_DENSE`).

**Voxel remeshing**: `voxel_remesh(voxel_size, mode="solid", thickness=None,
max_voxels=50_000_000)` rebuilds the mesh from a signed distance field
(:mod:`aether3d.sculpt.voxel`):

1. *Narrow band.* The grid (spacing `voxel_size`, world axes) is split into 8 x 8 x 8
   blocks; only blocks within reach of a triangle get storage. Every triangle measures
   the exact distance to the grid points of its box grown by the band (1.75 voxels:
   every corner of a cell the surface crosses is within `sqrt(3)` voxels of it), and each
   point keeps its nearest triangle (the lowest id among equally near ones, so batching
   never changes the result). Memory follows the surface's area, not the grid's volume.
2. *Sign* from angle-weighted pseudo-normals (Baerentzen and Aanaes 2005): exact for
   closed, manifold, consistently oriented parts, so each connected part must be one
   (`VOXEL_NOT_WATERTIGHT` otherwise, with the counts of boundary edges, non-manifold
   edges and vertices, degenerate triangles). Parts combine as solids: **overlapping
   parts fuse** (a point inside one part is inside the result however near it is to
   another part's buried surface), inward-facing parts (negative volume) are cavities.
   A mesh whose parts all face inward is refused (`VOXEL_INSIDE_OUT`).
   `mode="shell"` wraps any mesh, open or not, in a closed shell of `thickness`.
3. *Marching cubes* (Lorensen and Cline 1987) with a case table generated at import:
   on each cube face the crossings are joined into segments, the ambiguous face (two
   diagonal corners inside) always separating the inside corners (Montani, Scateni and
   Scopigno 1994), so neighbouring cells agree on every face; each cycle of crossings is
   triangulated without diagonals in a cube face. All 256 cases and random fields give
   closed, edge- and vertex-manifold, consistently oriented surfaces (tested).
   Crossings stay at least 1 % of a voxel from grid points, so no triangle degenerates.

The result is a new mesh: groups and the mask come from the nearest source point, UVs
are dropped (`VOXEL_UVS_DROPPED`: unwrap again or map textures by object coordinates),
normals are smooth. Details thinner than about two voxels are lost. `data` reports the
validity of the input and the result (tier, closed, manifold, volume, Euler
characteristic, components, smallest angle) and the grid. Marching-cubes triangles are
uneven (angles down to a fraction of a degree): follow with `remesh()` before sculpting
fine detail -- the [remeshing example](../examples/remesh_validity.py) measures both.

Quad remeshing (a field-aligned quad layout) is **not** offered: Aether3D's meshes are
triangles, and an honest quad remesher is a project of its own; the isotropic and voxel
remeshers are what exist.

## Symmetrize

`symmetrize(axis="x", direction="+", mode="auto", tolerance=None)` copies one side of the
local plane `axis = 0` onto the other:

* `mode="positions"` pairs vertices by mirrored positions (within `tolerance`, default
  `1e-5` of the mesh's size; using each vertex's position when the session started, so
  sculpting one side does not break its pairs) and gives each the exact mirror of its
  partner; ids, topology, UVs and the mask stay. A vertex without a partner moves onto the
  mirrored surface (`SYMMETRIZE_PROJECTED`) -- but no vertex is added, so a feature the
  other side's vertices do not reach is not reproduced.
* `mode="mirror"` cuts the mesh at the plane (crossed edges get exact on-plane vertices,
  UVs per render edge), removes the other side and mirrors the source side onto it with
  reversed winding, sharing the on-plane vertices: exactly symmetric, any topology.
  Cutting leaves thin triangles near the plane.
* `mode="auto"` (default): `positions` when every vertex on either side has a partner,
  else `mirror`.

## Validity

`aether3d.sculpt.check_mesh(positions, triangles, self_intersections=False)` reports,
with stable codes and sample ids: `NON_FINITE_POSITIONS`, `DEGENERATE_TRIANGLES`,
`NON_MANIFOLD_EDGES`, `NON_MANIFOLD_VERTICES`, `INCONSISTENT_WINDING`, `BOUNDARY_EDGES`
and, with `self_intersections=True`, `SELF_INTERSECTIONS`. Its summary has the counts,
`closed`, the manifold flags, the Euler characteristic, the volume of closed meshes,
edge lengths, the smallest angle and the **validity tier**:

| Tier | Means |
| --- | --- |
| `valid` | every position is finite |
| `manifold` | every edge has one or two triangles, every vertex one fan, consistent winding, no degenerate triangle |
| `closed` | manifold and no boundary edge (watertight) |
| `solid` | closed, a positive volume and no self-intersection -- only with the self-intersection test, never assumed |

`commit(require="closed")` (and a script's `"require"`) refuses a result below the tier
with `BELOW_VALIDITY_TIER` before anything changes; the session stays open to fix it.
`aether3d sculpt check` and the MCP tool `inspect_mesh` report any object's tier in
world units, and the visual feedback check `mesh_validity` turns it into a finding.

**Robustness.** Decisions about which side of a plane or line a point lies on use exact
predicates (:mod:`aether3d.spatial.predicates`: Shewchuk's static filters on float64
determinants, exact rational arithmetic when the filter cannot decide); intersection
points are computed in float64 and checked afterwards. Tolerances are relative to the
mesh's size (`1e-12 * extent^2` for zero areas, `1e-5 * extent` for symmetric pairs).
Self-intersections: candidate pairs from a BVH over the triangles, decided exactly --
pairs without a shared vertex by the triangle-triangle test (its constructive step uses
the exact segment test: a float ray-triangle determinant misfires on nearly parallel
triangles, which made the test report intersections between marching-cubes triangles one
cell apart), pairs sharing a vertex or an edge by exact segment and coplanarity tests.

## Sculpt scripts

```json
{
  "format": "aether3d.sculpt-script",
  "version": 1,
  "label": "Shape the head",
  "limits": {"max_dabs": 20000, "max_triangles": 1000000, "max_seconds": 60},
  "sculpts": [
    {
      "object": "Head",
      "require": "closed",
      "operations": [
        {"op": "set_mask", "value": 1, "box": [[-2, -2, -2], [2, 2, -0.6]]},
        {"op": "stroke",
         "brush": {"kind": "draw", "radius": 0.25, "strength": 0.2, "symmetry": "x"},
         "path": [[0.3, -0.9, 0.3], [0.5, -0.8, 0.4]]},
        {"op": "clear_mask"},
        {"op": "check", "self_intersections": true}
      ]
    }
  ]
}
```

`op` is `stroke` (the rest of the step is the stroke), `set_mask`, `invert_mask`,
`clear_mask`, `remesh`, `voxel_remesh`, `symmetrize` (their keyword arguments), `undo`,
`redo` or `check` (a read-only validity report). `aether3d.run_sculpt_script(scene,
script)`, `aether3d sculpt run PROJECT script.json` and MCP `run_script` apply it **all or
nothing, as one undo step**: if a step fails, a limit is reached (`BUDGET_EXCEEDED`:
`max_dabs` counts dabs, `max_triangles` the mesh after each step, `max_seconds` is
polled between dabs, passes and voxel batches) or a result is below its required tier,
every object keeps its mesh. Errors carry a stable `code`, the step's `location`
(`sculpts[0].operations[2]`) and a hint. Scripts are data: nothing in them is executed,
operation names come from a fixed list, every value is type-checked (a fuzz test feeds
mutated scripts to the executor: it never crashes, and every script it accepts is valid
against the JSON Schema, `aether3d sculpt schema` or the MCP resource
`aether3d://schema/sculpt-script`).

## Command line, MCP and visual feedback

```console
$ aether3d sculpt run head.aether shape.json --json        # apply a sculpt script, save
$ aether3d sculpt run head.aether shape.json --dry-run     # run and report, write nothing
$ aether3d sculpt check head.aether --self-intersections --require solid --json
$ aether3d sculpt schema                                   # the JSON Schema of scripts
```

`sculpt check` reports every mesh object's tier, counts, components, volume, edge
lengths and smallest angle in world units; with `--require TIER` its exit status is 3 when
one falls below. Projects with a history file record a sculpt run as one step (`aether3d
history undo` takes it back).

Over MCP (`aether3d mcp serve`), `run_script` accepts sculpt scripts and the read-only
tool `inspect_mesh` reports one object's validity (`self_intersections: true` for the
`solid` tier); `capabilities {"topic": "sculpt"}` lists brushes, operations, tiers and
limits; the resources `aether3d://sculpt/guide` (the guide for agents) and
`aether3d://schema/sculpt-script` describe the format. Errors are structured, with the
step's location and a hint, and show no paths of the machine.

The visual feedback expectations (Stage 12, [FEEDBACK.md](FEEDBACK.md)) gain the check
`mesh_validity`: `{"check": "mesh_validity", "objects": "Head", "tier": "closed"}`
reports `MESH_BELOW_TIER` (basis `geometry`, exact) with the counts that keep a mesh
below the tier. It is never corrected automatically -- corrections do not edit meshes. An
agent sculpts, looks (`observe_scene`), checks (`diagnose_scene`) and fixes what the
diagnosis finds; [`examples/mcp_sculpt_agent.py`](../examples/mcp_sculpt_agent.py) does
exactly that.

## History

A sculpt session has its own undo (one step per stroke, mask change, remesh or
symmetrize) and commits as **one step of the scene's history** ("Sculpt mesh of
'Head'"); `commit(close=False)` keeps sculpting, each commit its own step. A sculpt script
is one step for all its objects. The scene's undo restores the object's previous mesh
object exactly (meshes are immutable; the history keeps references, so undo costs no copy).
A session whose object's mesh changed meanwhile refuses to commit (`STALE_SESSION`).

The session's journal is sparse: a stroke records the moved vertices' old positions and
derived normal data, a dynamic-topology change the removed rows and the appended counts,
and a whole-mesh rebuild -- an isotropic or voxel remesh, a mirror symmetrize -- a
reference to the previous arrays (no copy: nothing writes into them afterwards). An
isotropic remesh runs its passes on a compact copy that records nothing, compacted again
when its dead rows pile up, and replaces the mesh once at the end: stopping it leaves the
mesh as it was, and its memory does not grow with the number of passes. `max_history` (256 steps) and `max_history_bytes` (512 MiB)
bound it; the oldest steps are dropped first.

## Design and languages

The design record ([research/2026-10-stage13/DESIGN.md](../research/2026-10-stage13/DESIGN.md))
chose a language per subsystem after measuring prototypes, without parallel
implementations:

| Subsystem | Language | Why |
| --- | --- | --- |
| Modeling operators (bevel, cuts, bridges, slides, dissolves, sweeps, cleanup) | Python on the editing engine's topology | topological bookkeeping over few elements; the engine's journal, validation and scripts come for free |
| Sculpt mesh, brushes, strokes, masks, symmetry | NumPy, data-oriented (structure of arrays, chunked) | a dab touches thousands of vertices at once: vectorized arithmetic over the neighbourhood |
| Dynamic topology, isotropic remeshing | NumPy, batch-parallel passes (independent sets, red-green refinement) | each pass decides all candidates at once; no per-edge Python loop |
| Voxel remeshing | NumPy: exact distances scattered from the triangles into blocks, marching cubes on arrays | measured 3.7-4.5 times faster than a nearest-point BVH query per grid point, with identical output |
| Exact predicates | NumPy filters, `fractions.Fraction` for the rare undecided rows | exact and dependency-free |

No GPU, C++ or Rust code was added: the measured workloads (below) run in interactive
time in NumPy, and a native kernel needs a bitwise-identical NumPy twin and an ABI bump.
The heaviest workload -- the voxel band at a million output triangles -- is the first
candidate if it ever needs to be faster.

## Performance

Measured with `benchmarks/bench_sculpt.py` (saved as
`benchmarks/results/stage13-sculpt.json`, commit 5323c44) on an i9-12900H laptop with
Python 3.14 and NumPy 2.5, each group and size in its own process, on UV spheres of
radius 1 with about 10k, 100k and 1M vertices (brush radii are world units):

| | 10k vertices | 100k | 1M |
| --- | ---: | ---: | ---: |
| mesh (vertices / triangles) | 10,011 / 19,320 | 101,025 / 199,808 | 1,001,820 / 1,996,568 |
| `SculptMesh.from_mesh` | 0.01 s | 0.16 s | 1.89 s |
| commit after one dab (`to_mesh`) | 1.06 ms | 9.52 ms | 99.9 ms |
| commit after a dynamic-topology dab | 1.67 ms | 18.6 ms | 188 ms |
| one `draw` dab, radius 0.03 (vertices moved) | 0.74 ms (2) | 1.53 ms (26) | 2.68 ms (249) |
| one `draw` dab, radius 0.1 (vertices moved) | 1.39 ms (26) | 2.33 ms (271) | 7.30 ms (2,715) |
| one `draw` dab, radius 0.3 (vertices moved) | 1.86 ms (241) | 6.62 ms (2,562) | 44.9 ms (23,675) |
| 100-dab `draw` stroke, radius 0.1, per dab | 0.92 ms | 1.57 ms | 6.54 ms |
| 100-dab `clay` stroke, radius 0.1, per dab | 0.78 ms | 1.46 ms | 5.57 ms |
| 100-dab `smooth` stroke, radius 0.1, per dab | 2.80 ms | 6.00 ms | 21.1 ms |
| 100-dab `grab` stroke, radius 0.1, per dab | 0.26 ms | 0.95 ms | 6.77 ms |
| undo of a 100-dab stroke | 1.28 ms | 4.05 ms | 23.5 ms |
| commit after a 100-dab stroke | 1.46 ms | 12.5 ms | 121 ms |
| journal of four strokes | 0.4 MiB | 3.1 MiB | 27.4 MiB |
| symmetry none: per dab | 0.93 ms | 1.63 ms | 6.68 ms |
| symmetry x: per dab | 1.45 ms | 2.96 ms | 12.5 ms |
| symmetry xy: per dab | 2.04 ms | 4.66 ms | 19.3 ms |
| dynamic topology, brush scaled to the density: per dab (radius) | 13.3 ms (0.15) | 16.0 ms (0.0466) | 20.1 ms (0.0148) |
| dynamic topology, radius 0.15: per dab | 13.2 ms | 85.7 ms | 870 ms |
| ... triangles before -> after 30 dabs | 19,320 -> 20,136 | 199,808 -> 206,958 | 1,996,568 -> 2,064,030 |
| ... undo | 3.08 ms | 16.3 ms | 137 ms |
| isotropic `remesh` (iterations) | 1.77 s (3) | 27.39 s (3) | 199 s (1) |
| `check_mesh` | 0.04 s | 0.49 s | 6.11 s |
| `check_mesh` with the exact self-intersection test | 0.36 s | 3.78 s | 47.56 s |
| peak memory, MiB (dab / stroke / remesh / validity) | 61 / 61 / 166 / 89 | 139 / 138 / 544 / 239 | 886 / 885 / 3440 / 1702 |

Voxel remeshing of the 100k sphere:

| Voxel remesh of the 100k sphere | voxel size | triangles out | time | result | peak MiB |
| --- | ---: | ---: | ---: | --- | ---: |
| about 10k triangles | 0.06 | 10,436 | 3.92 s | closed | 416 |
| about 100k triangles | 0.019 | 104,348 | 5.58 s | closed | 460 |
| about 1M triangles | 0.006 | 1,047,020 | 14.39 s | closed | 1224 |

The modeling operators on the editing engine:

| Modeling operator (`EditableMesh`) | 10k faces | 100k faces |
| --- | ---: | ---: |
| 3-segment bevel of a latitude ring (edges) | 69.3 ms (140) | 411 ms (448) |
| `bisect` | 93.3 ms | 1.39 s |
| `bisect` with `clear` and `fill` | 165 ms | 2.12 s |
| `knife_project` across the top | 204 ms | 2.00 s |
| `cleanup` | 61.0 ms | 775 ms |
| `recalculate_normals` | 76.6 ms | 881 ms |
| peak memory, MiB | 104 | 656 |

Against what 0.12 offered for the same work -- the editing API: select the vertices in a
sphere, move them along their normals (no falloff), commit -- measured on `main`
35e5a23 on the same machine (`benchmarks/results/stage13-baseline.json`):

| Against 0.12 (same machine) | 0.12 (editing API) | 0.13 (sculpt mesh) | factor |
| --- | ---: | ---: | ---: |
| open a 100k-vertex mesh | 12.43 s | 0.16 s | 76x |
| a dab of radius 0.1 at the pole (3,137 / 3,137 vertices moved) | 66.9 ms | 7.33 ms | 9x |
| commit it | 3.56 s | 15.8 ms | 226x |

What the numbers say:

- **Strokes stay interactive at a million vertices**: 6.5 ms per dab of a 100-dab `draw`
  stroke at 1M, 1.6 ms at 100k. Mirroring in x costs 1.9 times a plain dab, in x and y
  (four dabs) 2.9 times. A one-dab stroke costs 2.7 ms at 1M: nothing per stroke scans
  the mesh.
- **Dynamic topology costs what its brush covers.** With a brush scaled to the mesh's
  density (about the same vertices under it at every size) a dab costs 13.3, 16.0 and
  20.1 ms at 10k, 100k and 1M, for the same topology work (584, 583 and 592 splits); a
  brush of the same world size covers a hundred times more triangles on the 1M mesh and
  costs 870 ms a dab.
- **Commits rewrite what changed**: 9.5 ms after a dab at 100k, 100 ms at 1M (positions
  written back, normals recomputed only for the touched groups); undo of a 100-dab
  stroke 4 ms at 100k.
- **Whole-mesh operations are the slow ones**: isotropic remeshing takes 27 s for three
  iterations at 100k and 199 s for one at 1M (3.4 GB peak); voxel remeshing makes a
  million triangles in 14 s; the exact self-intersection test takes 3.8 s at 100k and
  48 s at 1M. They are the candidates for native kernels if they must get faster.
- **The modeling operators** run on the editing engine's per-element object graph: 60 to
  200 ms at 10k faces, 0.4 to 2.1 s at 100k.

## Limitations

- Quad remeshing is not offered (see Remeshing).
- Voxel remeshing needs closed, manifold, consistently oriented parts for `solid` (shells
  work on anything), loses details thinner than about two voxels and drops UVs; parts
  that touch at a single vertex are non-manifold and refused.
- Marching-cubes triangles are uneven; `remesh()` evens them out.
- Symmetrize's `mirror` mode leaves thin triangles near the plane where the cut passes
  close to vertices; its `positions` mode cannot add vertices.
- Relaxation (dynamic topology, remesh) is not checked for folds, and brushes can fold a
  surface (an `inflate` along a concave valley pushes its walls together); check with
  `require` or `check_mesh(..., self_intersections=True)`.
- Multiresolution sculpting (subdivision levels with displacement) is not part of this
  stage.
- The editing engine's operators are Python over a per-element object graph: they suit
  meshes up to about 100k faces (see Performance); the sculpting system is built for
  millions of vertices.
- Sculpting runs on the CPU only.

## References

- Bank, Sherman and Weiser, "Refinement algorithms and data structures for regular local
  mesh refinement", 1983 (red-green refinement).
- Baerentzen and Aanaes, "Signed Distance Computation Using the Angle Weighted
  Pseudonormal", IEEE TVCG 11(3), 2005.
- Botsch and Kobbelt, "A Remeshing Approach to Multiresolution Modeling", SGP 2004.
- de Goes and James, "Regularized Kelvinlets: Sculpting Brushes based on Fundamental
  Solutions of Elasticity", SIGGRAPH 2017.
- DeRose, Kass and Truong, "Subdivision Surfaces in Character Animation", SIGGRAPH 1998.
- Fu, Jeong, Pan, Kirby and Whitaker, "A Fast Iterative Method for Solving the Eikonal
  Equation on Triangulated Surfaces", SIAM J. Sci. Comput. 33(5), 2011.
- Liepa, "Filling Holes in Meshes", SGP 2003.
- Lorensen and Cline, "Marching Cubes", SIGGRAPH 1987.
- Luby, "A Simple Parallel Algorithm for the Maximal Independent Set Problem", SIAM J.
  Comput. 15(4), 1986.
- Montani, Scateni and Scopigno, "A modified look-up table for implicit disambiguation of
  Marching Cubes", The Visual Computer 10, 1994.
- Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
  Predicates", Discrete & Computational Geometry 18, 1997.
- Taubin, "A Signal Processing Approach to Fair Surface Design", SIGGRAPH 1995.
