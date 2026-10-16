# Stage 13 design record: advanced modeling and sculpting

Written on 16 October 2026, before any Stage 13 implementation code, against `main` at
35e5a23 (v0.12.0, Stages 1-12). It records what was inspected, what was measured, the
research behind each algorithm, and the implementation-language decision for every
subsystem. The user guide that describes what was finally built is
[docs/MODELING.md](../../docs/MODELING.md); where the two differ, this file is the plan
as decided and the guide is the truth about the code.

## 1. Starting point (inspected)

| Representation | Where | What it is | Strengths | Limits for Stage 13 |
| --- | --- | --- | --- | --- |
| `Mesh` | `mesh.py` | immutable indexed triangles: float32 positions, uint32 triangles, per-vertex normals and UVs (split at seams), uint16 face groups | the persisted, rendered, exported form; shared by reference | no topology, no polygons |
| `EditableMesh` | `editing/` | BMesh semantics in id-keyed Python tables; n-gons, non-manifold, per-corner layers, sharp/smooth, groups; seven journaled primitives; per-session undo; atomic operations | exact ids and rollback; every operation reports created/removed ids | ~90 us per triangle to open with attributes (audit: 16 s for 200k triangles), ~200 MiB per 200k triangles; per-element Python loops |
| `PolyMesh` | `procedural/polymesh.py` | immutable CSR polygons with per-corner attributes; vectorized NumPy | fast whole-mesh modifiers (Catmull-Clark: 168 ms for ~100k quads) | immutable, procedural only |
| scene history | `history.py` | aspects with before/after states; meshes shared by reference; 256 MiB budget | one undo step per user action | a full `Mesh` per recorded change |
| native kernels | `native/` (Rust, PyO3, abi3-py311) | ray traversal, hierarchies, PNG filters; bitwise equal to NumPy | 10-36x on scalar loops | optional: not on PyPI, no Python 3.10; every kernel needs its NumPy twin |

Documented limitations that Stage 13 addresses or keeps (EDITING.md, PROCEDURAL.md):
single-segment bevel without open-boundary support; no edge slide, knife, bridge, spin,
fill or rounded bevels; mirror without bisect; subdivision without creases; no
self-intersection test; no repair operations; no sculpting, dynamic topology or
remeshing.

Baseline measured on this machine before any change (`bench_engine.py --only editing`,
i9-12900H, CPython 3.14.8, NumPy 2.5.4): `from_mesh` with attributes 0.575 s for 16,128
triangles; topology only 0.360 s for 199,808 triangles; bevel of one edge 1.16 ms; bevel
of every edge of 2,048 faces 487 ms; inset 500 faces 95 ms; subdivide quads 112 ms;
`to_mesh` 33 ms. `benchmarks/bench_sculpt.py --only baseline`
(`benchmarks/results/stage13-baseline.json`, git 35e5a23) adds the workloads Stage 13
targets, on UV spheres:

| workload (existing engine) | 10k vertices | 100k vertices |
| --- | ---: | ---: |
| `EditableMesh.from_mesh` (UVs, normals, quads) | 0.71 s | 12.4 s |
| `to_mesh` | 0.17 s | 1.90 s |
| a "dab" with the editing API: `select_vertices(sphere=)` + `move_along_normals` (3,137 vertices, no falloff) | | 66.9 ms |
| committing that dab (`to_mesh`, normals recomputed) | | 3.56 s |

Catmull-Clark on 9,800 quads: 8.3 ms (level 1), 35 ms (level 2). `closest_point` on a
2M-triangle object: 9.5 s for the first query (BVH build), 1.4 ms warm. Peak RSS 1.4 GiB.
So sculpting through the editing engine costs seconds per dab once committed, and a
million-vertex mesh cannot reasonably be opened as an `EditableMesh` at all.

## 2. Measured feasibility (prototype, not product code)

`proto/dab.py` (not kept: scratch code) timed one Draw-like dab with plain NumPy on
welded UV spheres: candidate chunks by box test, candidate vertices, distances, falloff,
displacement along vertex normals from the face normals of the incident faces (CSR
gather), and the face normals of every affected face recomputed afterwards.

| mesh | vertices moved | dab |
| --- | ---: | ---: |
| 130,562 vertices, 261,120 triangles | 4,097 | 2.6 ms |
| 2,095,106 vertices, 4,190,208 triangles | 65,537 | 61.7 ms |

About 1 us per affected vertex plus a fixed ~0.3 ms: interactive for meshes up to a few
hundred thousand vertices, and still usable (tens of dabs per second) at two million.
Preparation of the 4.2M-triangle mesh: exact welding 0.43 s, vertex-to-face CSR 0.72 s,
all face normals 0.54 s. Partitioning it with the existing binned-SAH `build_bvh` (leaves
of 1,024 triangles) took 9.9 s; a Morton (Z-order) sort of the triangle centroids took
0.86 s. Chunks are therefore built from Morton order (refined into octree cells so no
chunk spans a Z-order jump), not with the SAH builder.

## 3. Language and architecture decision per subsystem

The rule: choose per job; a native kernel only where a measured loop dominates and its
arithmetic can be reproduced exactly by the NumPy implementation that the package
contract requires anyway (aether3d-native is optional, unpublished and Python >= 3.11).

| Subsystem | Work shape | Decision | Why |
| --- | --- | --- | --- |
| Modeling operators (multi-segment bevel, bridge, fill, knife, bisect, slide, dissolve, spin, ...) | combinatorial, selection-sized (1-10k elements), heavy branching, must journal into `EditableMesh` tables | **Python, extending `EditableMesh`** | Ids, journal, rollback and corner layers live there; operators cost O(selection). A native port needs the array-topology redesign the audit (Phase 4) deferred as a deliberate behavior change; Stage 13 must not destabilize the editing engine. |
| Robust predicates (orientation, plane side, segment crossing) | few calls, decisions must be exact | **Python: float filter with Shewchuk's error bound, exact rational fallback (`fractions.Fraction`)** | Floats are exact rationals; the exact path is rare. No dependency. |
| Sculpt representation (`SculptMesh`) | arrays: positions, triangles, render-vertex layer, adjacency CSR | **NumPy, data-oriented** | Bulk construction vectorizes (audit: topology 22x in NumPy vs 36x in Rust); no per-element Python objects. |
| Brush engine | per dab: gather k vertices, closed-form math, scatter | **NumPy** | Data-parallel; measured ~1 us/vertex. Brush math stays in one language; generic hot loops (gather, normals) use fixed-order arithmetic so a bitwise native twin is possible if measurements justify it. |
| Spatial acceleration for brushes | sphere, ray and nearest queries; refit after deformation; membership changes under dynamic topology | **NumPy: flat array of chunk boxes (octree-refined Morton chunks of ~1k triangles), vertex ownership per chunk** | Thousands of boxes are tested in one vectorized call (no tree walk needed); the chunk is also the unit of refit, normal updates and topology updates (as Blender's PBVH leaves are). |
| Dynamic topology and isotropic remeshing | local split / collapse / flip / relax; whole-mesh passes | **NumPy, batch-parallel: each pass applies an independent set of operations** | A sequential algorithm would need a sequential NumPy fallback (minutes in Python). Independent sets are the established way to run mesh adaptation in parallel (Rokos et al. 2013, PRAgMaTIc); one NumPy implementation serves every platform, and a native kernel could run the same batches. |
| Voxel remeshing | narrow-band distances (triangle x voxel pairs), surface extraction | **NumPy: sparse 8^3 blocks, angle-weighted pseudonormal sign, marching cubes with a generated, consistent table** | Pairwise distance work vectorizes; memory scales with surface area; no C++ dependency (OpenVDB has no portable pip wheels). |
| Geodesic falloff | eikonal distance in a brush region | **NumPy: fast iterative method with triangle updates** (Fu et al. 2011) | Designed for data-parallel execution; no sparse solver needed (SciPy is not a dependency). |
| GPU compute (WGSL) | — | **not used** | Per-dab work is 1k-100k vertices: dispatch, upload and readback through wgpu-py cost about as much as the NumPy work, the GPU is optional (CI runs without one), and the CPU result is needed anyway for history and validation. |
| C/C++ libraries | — | **not used** | CGAL's mesh processing is GPL; libigl is MPL-2.0 with GPL parts and its wheels lag Python releases; OpenVDB (Apache-2.0 since 12.0) has no portable wheels; OpenMesh/pmp are C++ without maintained wheels. Each would add a build or packaging burden larger than the algorithms they would replace here. |
| Interfaces (API, scripts, CLI, MCP) | control layer | **Python** | Unchanged architecture: declarative JSON scripts, never code. |

Not done: no second implementation of any feature for comparison; benchmarks compare
the chosen implementation with the existing baseline only.

## 4. Algorithms (with references)

- **Isotropic remeshing**: Botsch and Kobbelt, "A Remeshing Approach to Multiresolution
  Modeling", SGP 2004, sec. 4: split edges longer than 4/3 L, collapse edges shorter than
  4/5 L, flip to bring valences to 6 (4 on boundaries), tangential relaxation
  `p += lambda (I - n n^T)(g - p)` toward the area-weighted centroid, projection onto the
  input surface. Batch-parallel with independent sets (Rokos, Gorman, Southern, Kelly,
  "A thread-parallel algorithm for anisotropic mesh adaptation", arXiv:1308.2480).
- **Dynamic topology**: the same operators restricted to the brush region (Blender's
  Dyntopo refine modes: subdivide, collapse, both; detail as constant, relative to the
  brush or manual). Blender's Dyntopo discards UVs and other custom data (its warning,
  2015); Aether3D interpolates corner attributes exactly on splits and only collapses or
  flips where the corner attributes allow it, so nothing is discarded.
- **Brushes**: closed-form displacements (draw, clay, inflate, flatten, fill, scrape,
  pinch, crease, layer, grab, snake hook, twist, smooth, relax, mask). Smoothing without
  shrinkage: Taubin, "A Signal Processing Approach to Fair Surface Design", SIGGRAPH 1995
  (lambda/mu); HC-Laplacian (Vollmer, Mencl, Mueller 1999). Elastic grab: de Goes and
  James, "Regularized Kelvinlets", SIGGRAPH 2017.
- **Spatial chunks**: Morton order (Z-order) refined into octree cells; Blender's PBVH
  is the precedent for leaf-chunked sculpting (leaves are the unit of update and undo).
- **Geodesic distance**: Fu, Jeong, Pan, Kirby, Whitaker, "A Fast Iterative Method for
  Solving the Eikonal Equation on Triangulated Surfaces", SIAM J. Sci. Comput. 33(5), 2011.
- **Signed distance**: Baerentzen and Aanaes, "Signed Distance Computation using the
  Angle Weighted Pseudo-normal", IEEE TVCG 11(3), 2005.
- **Surface extraction**: Lorensen and Cline 1987 (marching cubes) with face ambiguities
  resolved by one fixed rule in every cell (the implicit disambiguation of Montani,
  Scateni and Scopigno, The Visual Computer 10, 1994): crack-free by construction. The
  table is generated by code (boundary cycles of the face segments), not copied.
- **Hole filling**: Liepa, "Filling Holes in Meshes", SGP 2003 (minimum-weight
  triangulation by dynamic programming, then refinement and fairing).
- **Bevel profiles**: superellipses (Blender's bevel: r = 2 circle, 1 straight, toward
  infinity a square corner), mapped affinely onto each corner's profile plane; corners
  where three or more bevelled edges meet are filled with a grid patch per sector.
- **Creases**: DeRose, Kass, Truong, "Subdivision Surfaces in Character Animation",
  SIGGRAPH 1998 (semi-sharp creases: sharp rules for floor(s) levels, then blended).
- **Predicates**: Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast
  Robust Geometric Predicates", DCG 18, 1997 (static error bounds as filters).

## 5. Robustness policy

- Tolerances are relative to the mesh scale (`1e-12 * extent^2` for areas, as the
  editing engine already does); decisions about which side of a plane or line a point is
  on use exact predicates; constructions (intersection points) are computed in float64
  and checked afterwards.
- Validity tiers: any `EditableMesh` is valid input to selection and inspection; each
  operator states its precondition (manifold edges, closed fans, consistent winding,
  closed volume) and refuses others before mutating, with a stable code and the
  offending element ids.
- Sculpting and dynamic topology never change what they cannot handle: non-manifold
  edges and vertices are frozen (no topology change touches them, reported once per
  stroke); open boundaries, UV seams, hard edges and face-group borders are features
  that are split along but not collapsed across or flipped.
- Every topology-changing pass checks its result locally (no folded or zero-area
  triangles, no new non-manifold edge, no duplicate triangle) and rolls back the whole
  operation if a check fails.

## 6. Identity, attributes, history

- Sculpt vertices keep their index for the whole session; new vertices and triangles are
  appended (never reused within a session); removed ones are marked dead. A commit
  compacts them into a `Mesh`. Unchanged meshes commit as the same object.
- Corner attributes live on render vertices (a `Mesh` vertex); geometry lives on welded
  vertices. Sculpting moves welded vertices and recomputes normals only for the normal
  groups it touched; UVs and groups are untouched; topology changes interpolate them.
- One stroke (or one remesh, mask or symmetrize call) is one undo step of the session,
  journaled as sparse reversible deltas (indices and old values), atomic like editing
  operations. A commit is one step of `scene.history`. Procedural objects are refused as
  for editing (`PROCEDURAL_OBJECT`), with the same alternatives.

## 7. Test strategy (written first, applied per component)

Brute-force references (distances, neighborhoods, normals, Laplacians, signed distances,
self-intersections), analytic brush cases (a point under a draw dab moves by
strength x radius x falloff), radius and falloff boundaries, zero-strength no-ops,
determinism (same inputs, same bits), mirrored-stroke symmetry, masks, undo/redo exact
restoration, cancellation rollback, topology invariants after every dynamic-topology and
remeshing pass (manifoldness, orientation, no degenerate or duplicate triangles, Euler
characteristic where expected), attribute preservation on seams, exhaustive marching
cubes cases (256 configurations: watertight and manifold), randomized operator tests with
fixed seeds, scripts and MCP security cases, NATIVE=0, minimal dependencies,
persistence and glTF round trips.

## 8. Benchmark plan

`benchmarks/bench_sculpt.py`: the baseline group (existing conversion, editing,
procedural subdivision, spatial queries, and a dab done with the existing API:
`select_vertices(sphere=)` + `move_along_normals`); conversion to and from the sculpt
representation; single dabs and 100-dab strokes (query, deformation, normals, topology
and bookkeeping timed separately) on 10k, 100k and 1M vertices; symmetry; dynamic
topology; isotropic and voxel remeshing; multi-segment bevel; peak memory per group (in
its own process). Medians after warm-up, GC paused during timed runs, machine and
versions recorded, as `bench_engine.py` does.
