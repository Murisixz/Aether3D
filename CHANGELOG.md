# Changelog

All notable changes to Aether3D are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may add
features; breaking changes are called out explicitly).

## 0.15.0: precision modeling, CAD and architecture

Aether3D gains exact, parametric modeling beside its meshes: units and tolerances,
sketches solved under constraints and dimensions, solid parts built from features on an
exact B-rep with persistent names, assemblies, measurements, drawings, STEP and DXF
exchange ([docs/CAD.md](docs/CAD.md)); and buildings -- levels, walls with exact joins,
doors and windows, rooms, slabs, roofs, stairs, columns, beams, quantities, schedules,
IFC4 ([docs/BUILDINGS.md](docs/BUILDINGS.md)). Everything works from Python, CAD scripts,
the CLI, scene plans and MCP, one undo step per change. Each subsystem's language was
decided before implementation (the design record:
[research/2026-10-stage15/DESIGN.md](research/2026-10-stage15/DESIGN.md)): Python and
NumPy (LAPACK) for geometry, the solver, the B-rep kernel and exchange; Manifold
(optional) for robust boolean topology; one new Rust kernel (batched polygon
triangulation, `aether3d-native` ABI 3, with a bit-identical NumPy twin).

Measured on the development machine (i9-12900H laptop, Python 3.14, NumPy 2.5;
`benchmarks/bench_cad.py`, saved as `benchmarks/results/stage15-cad.json`; the tables are
in [docs/PERFORMANCE.md](docs/PERFORMANCE.md#precision-modeling-015)):

- sketches: a fully constrained sketch of 999 constraints solves (with its DOF analysis)
  in 41 ms, one of 4,999 in 178 ms; changing one dimension re-solves only what follows
  it (38 ms and 188 ms, most of it bookkeeping);
- parts: the bracket of `examples/cad_bracket.py` (28 faces) builds in 0.47 s; a plate
  with 100 holes in 1.1 s; tessellating a sphere to 1e-4 of its size takes 27 ms (27,544
  triangles); exact mass properties of the bracket 70 ms;
- buildings: 10 levels of 100 rooms (2,200 walls, 1,000 doors, 1,000 rooms) build in
  16.6 s and regenerate cold in 11.2 s (227 MiB peak); moving one door re-evaluates one
  floor in 134 ms at every size measured;
- a plan with hidden lines and hatched sections of a 2-level, 72-room building: 1.2 s;
  STEP export and import of the bracket: 28 and 22 ms; IFC4 export of 312 products 0.09 s,
  import 1.7 s; triangulating 600 polygons (66k points): 18.6 ms native, 770 ms NumPy.

### Added

- **The CAD document** (`scene.cad`, `aether3d.cad`): items (parameters, frames,
  sketches, parts, assemblies, building elements, drawings) with JSON definitions,
  regenerated incrementally by content keys, atomic and undoable, shown in the scene as
  objects that follow them (`doc.show`, `convert_to_mesh`); project format 8.
- **Units and tolerances**: lengths, angles, areas, volumes, masses and densities with
  exact rational conversions, feet-and-inches and DMS text, fractions; safe expressions
  over parameters; scale-derived tolerances and tessellation tolerances.
- **Sketches**: points, lines, circles, arcs, splines; 18 constraint and dimension types;
  a solver with analytic Jacobians, Dulmage-Mendelsohn decomposition, Gauss-Newton and
  Levenberg-Marquardt, exact DOF / redundancy / conflict analysis, recorded branches,
  incremental solving; snapping (`doc.snap`).
- **The B-rep kernel**: exact planes, quadrics, tori, extrusions, revolutions, NURBS;
  validation; watertight tessellation within a chord; mass properties by boundary
  quadrature on the exact surfaces; booleans through Manifold with exact B-rep
  reconstruction and analytic edge recovery; sewing.
- **Features**: extrude (with draft), revolve, sweep, loft, holes (simple, counterbore,
  countersink, drill point), fillet, chamfer, shell, pattern, mirror, transform,
  boolean, primitives, imported solids; persistent face and edge names.
- **Assemblies** with mates (coincident, concentric, parallel, perpendicular, distance,
  angle, fixed), DOF and conflict analysis, nesting and exact clash detection.
- **Measurements** (`doc.measure`): exact descriptions, angles, parallel and axis
  distances, minimum distances with bounds.
- **Drawings** (`drawing` items): plan, section, elevations and isometric views with exact
  edges, ray-cast hidden lines, exact section fills, door swings, room labels,
  dimensions; SVG, DXF R12 and PNG (own rasterizer and stroke font).
- **Exchange**: STEP AP214 export and import of exact solids; DXF import into sketches
  and export of sketches and drawings; IFC4 export (walls with layer sets and types,
  openings, doors, windows, spaces, slabs, roofs, stairs, columns, beams, base
  quantities, property sets, deterministic GlobalIds) and import of its subset.
- **Buildings** (`aether3d.arch`, `doc.building`): levels, wall types, families, walls
  with exact L/T/junction joins, openings checked against their walls, rooms from wall
  faces, slabs with holes, flat/shed/gable/hip roofs, straight stairs checked against the
  user's criteria, columns and beams with rectangular, circular and I profiles;
  quantities (IFC base quantity formulas), schedules and CSV.
- **For agents**: CAD scripts (`aether3d.cad-script`, `run_cad_script`) with a JSON
  Schema and an agent guide; CLI `aether3d cad run | info | export | import | schema`;
  MCP: CAD scripts in `run_script`, the `inspect_cad` and `import_cad` tools, CAD exports
  in `export_scene`, the `cad` capabilities topic, guide and schema resources; the
  scene-plan operation `cad`; the feedback checks `cad_valid` and `cad_dimension`.
- **Native**: batched ear-clipping triangulation (`aether3d-native` 0.15.0).
- **Examples** (rendered and drawn by Aether3D): `cad_bracket.py`, `cad_floor_plan.py`,
  `cad_modern_house.py`, `cad_grid_building.py`, `mcp_cad_agent.py`;
  `benchmarks/bench_cad.py`.
- **Independent validation** in the tests (optional, `AETHER3D_VALIDATOR_PYTHON`):
  Open CASCADE reads the STEP exports as valid solids of the same volume, IfcOpenShell's
  validation finds no issue in the IFC exports, ezdxf's audit no error in the DXF;
  fixtures written by Open CASCADE and ezdxf are read exactly.

### Changed

- **Project format 8** (the CAD document); version 7 files load unchanged.
- **MCP**: 25 tools (`inspect_cad`, `import_cad` added); `export_scene` also writes
  STEP, IFC, DXF, SVG and PNG of CAD items.
- `aether3d-native` 0.15.0, ABI 3 (new kernel `triangulate`).
- Scene objects that show a CAD item refuse direct mesh edits (`CAD_DRIVEN`) until
  converted to plain meshes.

### Changed tests

Existing assertions changed (each for a reason stated in the PR): the project
`format_version` 7 -> 8 (tests/test_annotations.py x2, test_material_persistence.py,
test_pathtrace_api.py, test_plans_execution.py); the MCP tool list and count 23 -> 25
(tests/test_mcp.py x2, tests/test_examples.py).

## 0.14.0: UV texture painting and baking

Aether3D gives meshes UVs, paints textures on them in 3D and bakes detail from
high-resolution sources, from Python, JSON scripts, the CLI and MCP -- every operation
atomic, undoable and measured ([docs/TEXTURING.md](docs/TEXTURING.md)). The language of
each subsystem was decided before implementation (the design record:
[research/2026-10-stage14/DESIGN.md](research/2026-10-stage14/DESIGN.md)): vectorized
NumPy for topology, charts, rasterization, MikkTSpace, painting and map evaluation; Rust
(`aether3d-native`, ABI 2, with bit-identical NumPy twins) for the sparse LDL^T solver and
the packing placement; WGSL compute for bake rays, on the path tracer's BVH.

Measured on the development machine (i9-12900H laptop, RTX 4060 Laptop GPU, Python 3.14,
NumPy 2.5; `benchmarks/bench_texturing.py`, saved as
`benchmarks/results/stage14-texturing.json`; the tables are in
[docs/TEXTURING.md](docs/TEXTURING.md#performance)):

- unwrapping UV spheres of 10k, 100k and 1M triangles (two hemisphere charts each, the
  solver's worst case): 1.0 s, 9.3 s and 271 s (9.2 GiB); the native LDL^T makes the 10k
  and 100k unwraps 5.8 and 9.3 times faster than NumPy alone; MikkTSpace tangents: 0.024,
  0.25 and 3.1 s; exact UV reports: 0.71, 3.4 and 46 s;
- painting a 100k-triangle sphere: a dab of radius 0.05 costs 1.3 ms on a 1K texture and
  8.4 ms on an 8K one; a stroke touches only its tiles, and undoing a fill of the whole 8K
  texture takes 1.1 ms;
- baking from a 1M-triangle source with 2 x 2 samples: a 1K normal map in 6.1 s (CPU) or
  3.5 s (GPU rays), 4K in 45 s; AO with 64 rays per sample at 1K in 27.3 s on the CPU and
  3.8 s on the GPU (5.3 against 38.5 M rays/s).

### Added

- **UV mapping** (`aether3d.uv`, `obj.unwrap`, `obj.project_uvs`, `obj.edit_uvs`, and
  `EditableMesh.unwrap`/`project_uvs` in edit sessions and scripts): segmentation at hard
  edges, face groups, seams and an angle limit; LSCM then SLIM per chart (flip-free),
  Tutte fallback; splitting by geodesic distance until the stretch is at most
  `max_stretch`, with straightened borders; a sparse LDL^T solver with nested dissection;
  packing on conservative texel footprints with padding, quarter-turn orientations and
  optional fixed texel density; partial unwraps that keep polygons whole; planar, box,
  cylindrical and spherical projections; island transform, align, normalize, texel
  density, pack and stitch.
- **UV diagnostics**: `uv_report` (flips, degenerate and non-finite triangles, mirrored
  islands, exact overlaps with stacked islands told apart, stretch, texel density and
  spread, utilization, gaps) and `layout_image` (islands, density, stretch, overlaps,
  texture).
- **MikkTSpace**: a bit-exact float32 port of the reference implementation; every
  renderer reads UV normal maps in it; glTF export writes it as `TANGENT`.
- **Painted textures** (`scene.add_painted_texture`, `PaintedTexture`): 16-bit tiled
  layers (paint and fill), nine blend modes, opacity, visibility, masks, background,
  encodings (color, data, normal); history by tiles; incremental mips and partial GPU
  uploads; project format 7 stores layers losslessly.
- **Painting** (`aether3d.texturing.stroke`, `obj.paint`, `fill`, `gradient`): brushes
  (paint, erase, blur, smudge, clone, mask) with falloffs, spacing, pressure, opacity
  ceilings and symmetry; strokes in world space continuous across UV seams, in a
  camera's view with occlusion, or in UV space; material strokes over several
  textures; seam padding; one stroke, one undo step; cancellable.
- **Baking** (`scene.bake`, `aether3d.texturing.bake`): normal (tangent space, the exact
  inverse of the renderers' transform), object and world normals, height, AO, thickness,
  curvature, position, base color, roughness, metallic, emissive, opacity, material /
  object / group IDs; cage and depth or an explicit cage; supersampling; padding; miss
  and UV-overlap diagnostics; CPU (all cores) or GPU; `apply=True` binds the maps.
- **Texture transfer** (`scene.transfer_textures`): carry a look onto a remeshed copy.
- **Texture scripts** (`aether3d.texture-script`, `run_texture_script`) with a JSON
  Schema; CLI `aether3d texture run | report | bake | schema`; MCP: texture scripts in
  `run_script`, `inspect_uvs` and `inspect_texture` tools, the `texturing` capabilities
  topic, guide and schema resources; the `uv_quality` feedback check.
- **Examples** (rendered by Aether3D): `texturing_hard_surface.py`,
  `texturing_organic_paint.py`, `texturing_bake.py`, `texturing_pbr.py`,
  `mcp_texture_agent.py`; `benchmarks/bench_texturing.py`.

### Changed

- **Project format 7** (painted textures); version 6 files load unchanged.
- **glTF export** repacks metallic/roughness and occlusion read from channels glTF does
  not read (as baked maps are bound) into the channels it reads
  (`TEXTURE_CHANNELS_PACKED`); they used to be left out (`TEXTURE_CHANNEL_NOT_EXPORTED`,
  which remains for bindings with different transforms).
- **Normal maps** are read in MikkTSpace instead of per-triangle UV-gradient frames for
  UV bindings that keep the UV basis; maps baked by MikkTSpace tools (Blender,
  Substance, Aether3D) render as baked. Renders of normal-mapped meshes can change where
  the two frames differ (mostly on curved low-polygon meshes).
- **MCP**: 23 tools (`inspect_uvs`, `inspect_texture` added).
- `aether3d-native` 0.14.0, ABI 2 (new kernels `ldl_*`, `pack_horizon`).

### Changed tests

Existing assertions changed (each for a reason stated in the PR): the project
`format_version` 6 -> 7 (tests/test_annotations.py x2, test_material_persistence.py,
test_pathtrace_api.py, test_plans_execution.py); the MCP tool list and count 21 -> 23
(tests/test_mcp.py x2, tests/test_examples.py); glTF channel diagnostics now
`TEXTURE_CHANNELS_PACKED` (tests/test_material_gltf.py).

## 0.13.0: advanced modeling and sculpting

Aether3D gains the modeling tools of a production polygon editor and a sculpting system
built for scripts and agents: brushes with masks, symmetry and pressure as data, dynamic
topology, isotropic and voxel remeshing, symmetrize, exact validity checks with tiers --
each operation atomic, undoable and refused with a structured error before it changes
anything ([docs/MODELING.md](docs/MODELING.md)). Every subsystem's language was chosen
after measuring prototypes (the design record:
[research/2026-10-stage13/DESIGN.md](research/2026-10-stage13/DESIGN.md)): Python for
the topological operators, data-oriented NumPy for sculpting, dynamic topology and
remeshing; no new native code was needed for the measured workloads.

Measured on the development machine (i9-12900H laptop, Python 3.14, NumPy 2.5;
`benchmarks/bench_sculpt.py`, saved as `benchmarks/results/stage13-sculpt.json`; the full
tables are in [docs/MODELING.md](docs/MODELING.md#performance) and
[docs/PERFORMANCE.md](docs/PERFORMANCE.md)), on spheres of about 10k, 100k and 1M
vertices:

- a 100-dab `draw` stroke: 0.9, 1.6 and 6.5 ms per dab; mirrored in x, up to 1.9 times that;
- a dynamic-topology dab with a brush covering the same number of vertices: 13.3, 16.0 and
  20.1 ms (the work follows the brush, not the mesh);
- committing a dab: 1.1, 9.5 and 100 ms; undoing a 100-dab stroke: 1.3, 4.1 and 24 ms;
- against the 0.12 editing API on the 100k sphere: opening it for sculpting 76 times
  faster (0.16 s), the same 3,137-vertex dab 9 times (7.3 ms), committing it 226 times
  (16 ms);
- isotropic remeshing: 1.8 s and 27 s for three iterations at 10k and 100k, 199 s for one
  at 1M; voxel remeshing of the 100k sphere into 10k, 100k and 1M triangles: 3.9, 5.6 and
  14 s; the exact self-intersection test: 0.4, 3.8 and 48 s.

### Added

- **Modeling** (`EditableMesh` operations, each also in JSON edit scripts):
  multi-segment `bevel_edges(segments=, profile=)` (superellipse profiles, corner
  patches where three or more bevelled edges meet, open fans); `bisect` (with
  `clear`/`fill`); `knife_project`; `fill_holes` (polygon, fan, or Liepa's minimum-weight
  triangulation); `bridge_edge_loops` (segments, twist); `slide_edges`, `slide_vertex`;
  `dissolve_vertices`, `dissolve_faces`, `limited_dissolve`; `connect_vertex_path`,
  `connect_vertices`; `extrude_edges`, `spin`, `add_lathe`; `shrinkwrap`;
  `set_auto_smooth`, `recalculate_normals`, `set_weighted_normals`; `cleanup` with an
  auditable report.
- **Procedural**: Catmull-Clark subdivision `crease` (DeRose et al. 1998 semi-sharp
  creases); mirror `bisect`.
- **Sculpting** (`aether3d.sculpt`, `obj.sculpt()`, `geometry.sculpt_source()`):
  `SculptMesh` (welded geometry and render attributes, Morton chunks, incremental
  normals, a sparse undo journal, commit that recomputes only touched normal groups);
  15 brushes (`draw`, `inflate`, `clay`, `flatten`, `fill`, `scrape`, `pinch`, `crease`,
  `layer`, `grab`, `elastic` -- regularized Kelvinlets --, `snake_hook`, `twist`,
  `smooth` with Laplacian, Taubin and relax methods, `mask`); falloff curves, projected
  and geodesic falloff, front faces only; masks, automasking (topology, face groups,
  boundaries); symmetry on any of x, y, z; pressure per path point; world-unit
  settings under any transform; one stroke is one undo step, rolled back on any failure.
- **Dynamic topology** per brush (`dyntopo`): red-green splits, collapses as
  independent sets with link and fold checks, Delaunay flips, tangential relaxation;
  UVs interpolated exactly on splits; seams, hard edges, group borders and boundaries
  kept.
- **Remeshing**: `remesh()` (isotropic, Botsch-Kobbelt, with projection onto the
  starting surface and feature preservation) and `voxel_remesh()` (exact narrow-band
  signed distances in 8^3 blocks, pseudo-normal signs, overlapping parts fused,
  cavities, shells for open meshes, marching cubes from a generated crack-free table).
  Quad remeshing is not offered.
- **Symmetrize**: exact copy of paired vertices, or cut-and-mirror for different
  topologies (`mode="auto"` picks).
- **Validity**: `check_mesh()` with tiers `valid` < `manifold` < `closed` < `solid`;
  `commit(require=...)` refuses results below a tier (`BELOW_VALIDITY_TIER`); exact
  orientation predicates (`aether3d.spatial.predicates`: Shewchuk's filters, rational
  fallback).
- **Sculpt scripts** (`aether3d.sculpt-script` v1, `run_sculpt_script`): atomic, one
  undo step, with limits (`max_dabs`, `max_triangles`, `max_seconds`), locations and
  hints in errors, a JSON Schema; **CLI** `aether3d sculpt run | check | schema`;
  **MCP** `run_script` accepts sculpt scripts, the new read-only tool `inspect_mesh`,
  resources `aether3d://sculpt/guide` and `aether3d://schema/sculpt-script`,
  capabilities topic `sculpt`; **visual feedback** check `mesh_validity`.
- `SculptMesh.should_stop`: cancellation polled between dabs, passes and voxel batches
  (rolled back).
- Examples, all rendered by Aether3D: `hard_surface.py`, `organic_sculpt.py`,
  `remesh_validity.py`, `mcp_sculpt_agent.py`.
- Spatial: `TriangleSurface` (exact nearest points and ray casts on one mesh),
  `build_bvh_morton` (linear BVH for large meshes), `component_labels`.

### Fixed

- `triangles_intersect` (spatial `overlaps`, `distance`, self-intersection checks): its
  constructive step used a float ray-triangle test whose determinant is all rounding
  for nearly parallel triangles, and reported intersections between disjoint triangles
  (found on marching-cubes triangles one cell apart). It now uses exact predicates.
- GPU shadows with three or more shadowed lights failed with a wgpu validation error
  (shadow passes between the first and the last carried timestamp writes without an
  index) on adapters with timestamp queries, lavapipe included.

### Changed

- `bevel_edges` accepts open fans (an edge ending on a boundary) and `segments > 1`;
  `segments=0` is a `ValidationError`.
- The MCP server has 21 tools (`inspect_mesh`).

## 0.12.0: visual feedback and automatic corrections

Aether3D now closes the loop between what a scene should be and what its renders show:
**observe -> diagnose -> propose -> apply -> verify**, step by step or in a budgeted
loop, from Python, the command line and MCP ([docs/FEEDBACK.md](docs/FEEDBACK.md)). What
the engine can measure -- geometry, pixels, light -- it reports as facts, each labeled
with how it was measured; what only a model or a person can judge comes in as a checked
vision evaluation, grounded in the render's object ids and never treated as a fact. The
engine contains no model, makes no network request and never claims to have seen an
image.

Measured on the development machine (i9-12900H, RTX 4060 Laptop GPU, Vulkan;
`benchmarks/bench_feedback.py`, saved as `benchmarks/results/stage12-feedback.json`),
observations at 640 x 360:

| | GPU rasterizer | CPU rasterizer |
| --- | ---: | ---: |
| One observation after one object moved: 34 / 2,190 / 11,270 objects | 24.0 / 26.8 / 46.2 ms | 1.14 / 2.60 / 3.10 s |
| An observation's measurements (objects and image) | 11-15 ms | 11-15 ms |
| The same view again, the scene unchanged | 0.01 ms | 0.01 ms |
| The checks of the furnished room's diagnosis (6 checks) | 240 ms | 240 ms |
| Applying a correction step and verifying it (the room) | 100 ms | 1.22 s |
| The loop, first draft to accepted: the room (1 iteration, 2 renders) / the garden (2 iterations, 3 renders) | 390 / 179 ms | 2.64 / 5.58 s |

### Added

- **Observations** (`scene.observe(view, ...)`, `aether3d.feedback.observe`): render a
  view with an existing backend and measure it, without changing the scene -- per object
  its pixels, screen bounds, center, border contact, depth range and luminance; the
  frustum test, exact projected bounds and near/far clipping of its geometry; with
  `occlusion=True` its unoccluded footprint (its triangles rasterized alone) and the
  objects covering the rest; for the image coverage, luminance, black and clipped
  fractions. Images: the render, an annotated copy with numbered markers, object colors,
  depth, masks and enlarged crops. Observations are reused at the same revision from a
  memory-bounded `ObservationStore`. A group (a prefab of blocks, an imported hierarchy)
  is observed, checked and moved as one object.
- **Views** and **virtual cameras**: `render(camera=...)` and the camera queries take a
  camera that is not part of the scene (`aether3d.feedback.virtual_camera`; a scene
  camera's `camera_view` twin renders bit for bit like it); view specifications for a
  scene camera, a virtual camera, a close-up framed on objects and presets (`top`,
  `front`, `iso`, ...).
- **Expectations** (`aether3d.expectations`, version 1): `exists`, `visible`, `framing`,
  `rests_on`, `no_overlap`, `inside_region`, `distance`, `relation`, `upright`,
  `material`, `exposure`, `camera_clipping`, with severities, which side a correction
  may move, and acceptance rules; checked as a whole (codes, JSON pointers, hints) and
  described by a JSON Schema.
- **Diagnosis** (`FeedbackSession.diagnose`): findings with a stable code, their
  **basis** (`geometry`, `render`, `scene`, `heuristic`, `vision`, `agent`), whether the
  measurement is exact, a magnitude and unit, the subjects with their roles and the region
  in the image. Placement is decided on triangles: gaps by vertical rays, penetrations and
  separations by bisection verified with the exact intersection test. Default checks
  (black or empty images, render warnings, far-off exposure, possible penetrations) are
  labeled as heuristics or information.
- **Vision evaluations** (`aether3d.vision-evaluation`, version 1): an image-aware
  evaluator's findings (over MCP, the agent's own), checked, tied to one observation's id
  and pixel digest, marked stale after changes, grounded in the object index, labeled by
  their declared provenance (the engine records only whether it delivered the image), and
  only ever proposed for approval. Critics plug into the loop as callables
  (`run_loop(critic=...)`); none ships, and none is required.
- **Corrections**: 16 allowlisted actions (placement, visibility, materials, cameras,
  lights, exposure, procedural parameters -- no deletion, mesh editing, files or code)
  compiled to scene plans and scene scripts; proposals with rationale, expected effect,
  alternatives and a safety class (`safe`, `verify`, `approval`); a `CorrectionPolicy`
  with limits; `preview` (apply in a transaction, observe, roll back); `apply` as one undo
  step, refused for a changed revision or stale proposals, verified on a new render and
  undone when a passing check fails or nothing improved.
- **The loop** (`run_loop`, `FeedbackSession.run`): budgets (iterations, renders, pixels,
  time, memory, critic calls and cost), cancellation, stagnation, no-progress and cycle
  detection, a stop reason, a checkpoint before the first change, and `resume` after
  approvals.
- **MCP**: five tools that separate inspecting, proposing and changing --
  `observe_scene` and `diagnose_scene` (inspect), `propose_corrections` (propose),
  `apply_corrections` and `feedback_loop` (change) -- with images as MCP image content;
  resources (the agent guide, the schemas, observations and their images), the
  `visual_feedback` prompt, `capabilities(topic="feedback")` and
  `--feedback-backend auto|gpu|cpu`.
- **CLI**: `aether3d feedback observe|check|correct|loop|schema` (`check` and `loop` exit
  with 3 when the expectations are not accepted).
- **`FeedbackError`** (`code`, `location`, `hint`, `details`, `issues`).
- **Examples**: `examples/feedback_room.py` (a furnished room with a floating vase, a
  sunken chair and a lamp inside a wall), `examples/feedback_composition.py` (a camera
  aimed past its subject, too far away and underexposed) and
  `examples/mcp_feedback_agent.py` (the castle corrected over MCP).
- **Benchmarks**: `benchmarks/bench_feedback.py`.

### Changed

- **The MCP server has 20 tools** (15 before) and replies may carry several images.
  Existing tests whose expected values changed with it: `tests/test_mcp.py` (the tool
  list gains five names, 20 tools over raw JSON-RPC, the prompt list gains
  `visual_feedback`) and `tests/test_examples.py` ("20 tools").
- **`RenderCamera`** values without an object id are accepted wherever a camera is: by
  `render` (checked: `INVALID_CAMERA` for a malformed frame) and by the camera queries.

### Compatibility

- **No format change** (`FORMAT_VERSION` unchanged), **no new dependency**, no new WGSL
  shader and no new Rust kernel: `aether3d-native` is unchanged (0.11.1, ABI 1). The CPU
  path works without a GPU and without the native package.
- **Unchanged results:** the render goldens, the path tracer's bit-for-bit images, the
  spatial references and the history catalog pass unchanged; no existing assertion was
  weakened.
- **MCP sampling is not used** (it is deprecated as of the 2026-07-28 protocol,
  SEP-2577): over MCP the client's model is the evaluator.
- **CI:** the 22 jobs are unchanged; the wheel check also requires the feedback guide,
  and the lavapipe job runs the feedback benchmark's smoke mode.

## 0.11.1: performance modernization

An architecture-first performance release, following the plan of the
[October 2026 audit](research/2026-10-architecture-audit/REPORT.md) and recorded phase by
phase, with every measurement and every target not met, in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md). Renders, queries and edits reuse what earlier
ones computed and update it from what changed; an optional native package runs the
hottest CPU kernels. Every faster path computes exactly what the path it replaces
computes: the same images, query answers, documents and history.

Measured on the development machine (i9-12900H, RTX 4060 Laptop GPU, Vulkan), 10,000
objects, GPU frames at 1280x720 (`benchmarks/bench_engine.py`: the Phase 0 baseline, made
on 0.11.0 with the spatial-query leak fixed, against the final run, `phase4.json`):

| | 0.11.0 | 0.11.1 |
| --- | ---: | ---: |
| GPU frame, nothing changed: every output / image only | 115.5 ms / -- | 15.2 ms / 7.3 ms |
| GPU frame, one object moved: every output / image only | 113.4 ms / -- | 17.1 ms / 8.8 ms |
| GPU frame, the camera moved | 117.3 ms | 14.5 ms |
| GPU frame, every object moved | 116.2 ms | 40.8 ms |
| The frame's GPU passes alone (timestamp queries) | 1.8 ms | 1.8 ms |
| Extracting an unchanged scene | 69.1 ms | 0.21 ms |
| First raycast after an edit / a warm raycast | 242.7 / 7.4 ms | 54.5 / 1.4 ms |
| CPU camera rays, one thread (100k triangles) | 0.31 M/s | 3.1 M/s with `aether3d-native` |
| PNG decoding, 1080p | 311.5 ms | 41.0 ms with `aether3d-native` |
| `EditableMesh.from_mesh`, 199,808 triangles | 0.70 s | 0.35 s |
| Bevelling every edge of a 2,048-face sphere | 0.66 s | 0.50 s |
| One plan creating 10,000 objects | 89.1 ms | 77.1 ms |

0.11.0 always read back every output (28 bytes per pixel); "image only" is the new
`render(outputs=["image"])` (4 bytes per pixel), which MCP previews use. The audit
estimated a frame with one moved object at about 12 ms (about 6 ms reading back only
the image): measured, 17.1 ms and 8.8 ms -- the estimates are not reached. There is no
render without readback (every render returns an image); the GPU passes alone take
1.8 ms.

### Added

- **`aether3d-native`** (optional; `pip install ./native`, needs Rust 1.85 or later;
  Apache-2.0, built with PyO3 and maturin): Rust kernels for ray traversal (the CPU path
  tracer and spatial queries), transform hierarchies and PNG row filters, with exactly
  the arithmetic of their NumPy implementations -- the same hits, matrices and bytes.
  Without it, or with `AETHER3D_NATIVE=0`, the NumPy code runs. One wheel per platform
  for CPython 3.11 and later (stable ABI); `AETHER3D_NATIVE_THREADS` traces with
  threads (results do not depend on it).
- **`render(outputs=...)`**: return only some per-pixel outputs (`"image"`, `"linear"`,
  `"depth"`, `"object_index"`, `"coverage"`); the GPU then reads back only those (4 bytes
  per pixel for the image alone, instead of 28). Saving `.hdr`/`.pfm` without linear
  light raises `OUTPUT_NOT_RENDERED`.
- **`RenderResult.png_bytes(compression=...)`**: the zlib level (default 6, as before).
- **Benchmarks:** `benchmarks/bench_engine.py`, the engine-wide suite (objects, queries,
  extraction, GPU frames, ray tracing, editing, images, I/O, memory, lifecycle), with the
  saved run of every phase in `benchmarks/results/`.
- **Implementation switches** (environment variables, `aether3d/_features.py`): every
  new implementation keeps the old one until it has proven itself --
  `AETHER3D_MATERIAL_CACHE`, `AETHER3D_GPU_PERSISTENT_BUFFERS`,
  `AETHER3D_RETAINED_RENDER_WORLD`, `AETHER3D_NATIVE`, `AETHER3D_EDITING_BULK`.
- **docs/PERFORMANCE.md**: how to measure, the correctness guards, the switches and the
  record of every phase.
- **Research:** `research/2026-10-architecture-audit/` (the audit: report, results and
  experiments) and `research/2026-10-modernization/` (the camera-rebase study and the
  hardware ray tracing prototype).

### Changed

- **Rendering is incremental.** A retained render world, updated from the history's new
  change journal, replaces `extract_render_scene`'s walk over every object; materials
  resolved for one render are reused by the next while unchanged (images are read again
  every render); GPU frames keep their per-frame buffers and write only what changed,
  and keep and update their instance records, draw order and material tables.
  `world_matrices` and framed renders reuse the retained world's matrices and framing
  points.
- **Spatial queries** reuse every unchanged object's entry from the previous revision's
  index and build in bulk (a warm raycast without `aether3d-native`: 2.8-3.0 ms).
- **MCP previews** read back only the image and encode it at zlib level 1; PNG encoding
  chooses its row filters 1.8 times faster (the same bytes).
- **Scene plans** draw new object ids in one batch and check transforms in bulk.
- **Mesh editing:** `EditableMesh.from_mesh` builds its tables in bulk (the same ids,
  tables and orders); quad joining and cross products are cheaper.
- **The CPU path tracer** traces with `aether3d-native` when installed: a one-process
  render (160x120, 8 samples, 30 spheres and a plane) took 0.45 s instead of 1.85, with
  the same image bit for bit.

### Fixed

- **A spatial query kept its scene alive forever** (a module-level cache keyed by the
  scene): every scene that answered a query stayed in memory with its index. Caches
  derived from a scene now live with the scene (`tests/test_scene_lifecycle.py`).
- **A change between `0.0` and `-0.0` is a change**: the history recorded nothing when a
  value changed only in the sign of a zero, so undo could not restore it and caches kept
  the other zero.

### Compatibility

- **Unchanged results:** the render goldens, the path tracer's bit-for-bit images, the
  spatial brute-force references and the history catalog pass unchanged, with and
  without `aether3d-native`; every switch's old and new paths are compared by tests.
- **No format change** (`FORMAT_VERSION` unchanged); public APIs only gain the additions
  above; MCP workspace confinement and every security limit are unchanged.
- **New optional package, no new dependency of `aether3d`**: `aether3d-native` (Rust, PyO3
  is MIT or Apache-2.0). Aether3D stays pure Python and installs everywhere.
- **CI:** the 17 jobs are unchanged; five new jobs build and test with `aether3d-native`
  (Linux on Python 3.11 and 3.14, macOS, Windows) and lint the crate (rustfmt, clippy).
- **Costs:** a rendered scene keeps 1,721 bytes per object for its retained world; setting
  a transform to the value it already has costs 3.8-4.0 us instead of 3.3-3.4.
- **Hardware ray tracing** is not used; the GPU path tracer keeps its software traversal.
  A prototype with ray queries in its kernel (`research/2026-10-modernization/hw_pathtrace/`)
  hit the same triangles for at least 99.995 % of rays but made the showcase renders only
  1.04-1.65 times faster end to end, short of the 2x the audit set: deferred (see
  docs/PERFORMANCE.md, Phase 5).
- **Not reached** (documented with the reasons in docs/PERFORMANCE.md): every object moved
  within 25 ms (40.8 ms), creating objects within 5 us each (7.7 us), a refit of the
  spatial index after edits (not done), and editing at the audit's sizes (a
  200,000-triangle mesh with attributes, or bevelling 51,000 faces, within 1 s).

## 0.11.0: AI tools and MCP

### Added

- **Scene plans: `scene.execute_plan(plan)`** ([docs/SCENE_PLANS.md](docs/SCENE_PLANS.md)).
  One declarative JSON document (`aether3d.scene-plan`, version 1) builds or changes a
  scene: checked as a whole first (every problem with a stable code, a JSON pointer, the
  operation index and a hint), then applied as one undo step, all or nothing.
  - **Resources by symbol:** shared meshes (primitives, data, `from_object`), materials
    and textures, procedural geometry nodes, and **prefabs** (reusable structures that
    can use other prefabs), referenced as `"@symbol"`, `"@towers[2]/roof"`.
  - **Patterns:** `grid`, `array`, `circle`, `line`, `scatter` (reproducible),
    `instances`, `repeat`; per-instance variation (`vary`: cycle, choose, uniform,
    linear, values, expressions) and filters (`where`: a doorway in a wall of blocks);
    name templates; expressions over plan parameters and instance variables through the
    procedural engine's whitelisted evaluator.
  - **Operations** on new and existing objects (`update`, `delete`, `set_parameters`,
    `frame_camera`, `set_active_camera`) with **selectors** (references, ids, names,
    tags, name globs, kind, ancestor, material, `expect` counts); the world, render
    settings and the active camera; **queries** answered after the plan (every spatial
    query, with plan references and selectors).
  - **Modes:** execute, dry run (apply, report, roll back) and validate (check and
    estimate); `validate_plan(plan)` checks without a scene.
  - **Safe retries and concurrency:** `expected_revision` (optimistic concurrency) and
    `idempotency_key` (ids derived from the key, so a resent plan is recognized even by
    another process).
  - **Large plans:** opt-in `chunk_size` applies in resumable chunks, one undo step each,
    after a named checkpoint.
  - **Bounds:** `PlanLimits` (objects, patterns, operations, triangles, mesh memory, ...),
    `timeout`, `cancel` (rolls back at the next safe point), `progress`.
  - `PlanResult` (status, revisions, the undo step, counts, ids per symbol, per-operation
    and per-query results, warnings, estimates, timings per stage), `PlanError`,
    `plans.plan_schema()` (the JSON Schema), `plans.select_objects`,
    `plans.with_selections`.
  - A guide for agents and example plans ship in the package: `basics`, a
    Minecraft-style **castle** (2,190 objects) and a **village** of 11,270 objects from
    reusable houses, trees, lamps and cars.
- **Command line:** `aether3d execute-plan PROJECT PLAN` (one atomic, undoable step;
  `--dry-run`, `--validate`, `--timeout`, `--max-objects`, `--progress`, `--history`),
  `aether3d plan validate PLAN [--project P]` (exit status 3 on problems) and
  `aether3d plan schema`.
- **The MCP server: `aether3d mcp serve`** ([docs/MCP.md](docs/MCP.md)), on the official
  MCP Python SDK, as the optional `mcp` extra. Fifteen tools (scenes and projects,
  objects, plans, checks, spatial queries, previews and images, history, glTF export,
  scripts, diagnostics, capabilities) with strict argument schemas and structured
  replies; resources (the guide, the schema, example plans, the scene, the last preview)
  and prompts. stdio only; every file inside the `--workspace` (checked after resolving
  symbolic links), no overwrites without `overwrite`, `--read-only`, no code execution,
  bounded renders, timeouts, cancellation with rollback, progress, errors that show no
  paths of the machine (workspace-relative, `~`, `<path>`). Previews use the GPU
  rasterizer when it works, the CPU rasterizer otherwise, and say which. Replies stay
  small: long query answers keep their first `max_results` entries (with `total` and
  `truncated`), and read-only replies above `--max-reply-kib` are refused with a hint.
  Verified with Claude Code 2.1.300 (`claude mcp add`, `claude mcp get`: connected).
- **Examples:** `plan_basics.py` (the Python API), `village.py` (11,270 objects, a
  one-door edit, a GPU preview), `mcp_client.py` (an SDK client) and
  `mcp_castle_agent.py` (an agent's whole loop over MCP: build with mistakes, find them
  with spatial queries, fix, undo, redo, preview, save, export, reopen).
- **Benchmark:** `benchmarks/bench_plans.py` compares one plan with repeated operations
  from Python, the CLI and MCP at 100, 1,000 and 10,000 objects (time, throughput, peak
  RSS, undo steps and bytes, plan stages, queries, GPU and CPU rendering, tool-call
  overhead), and runs against older versions for a baseline. Results are in
  [docs/SCENE_PLANS.md](docs/SCENE_PLANS.md#performance): on an i9-12900H, 10,000 blocks
  take 105 ms with one plan, 1,925 ms with `add_cube` per block and 379 ms with
  `add_object` calls in one transaction; one CLI plan 0.77 s against a projected 3.2
  hours for one CLI call per block; one MCP call 102 ms against 22 s for one call per
  block.
- **Tests:** 170 new tests (no existing test changed): the format (the checker fuzzed
  against the JSON Schema, the executor fuzzed for crashes and atomicity), patterns and
  references, idempotency and chunks, cancellation and timeouts, the CLI, projects of
  0.1.0, 0.4.0 and 0.7.0, parity with spatial queries, the examples, the docs' plans, the
  MCP server as a real subprocess (both protocol generations, raw JSON-RPC for a clean
  stdout, path escapes and symbolic links, overwrite and read-only rules, cancellation
  with rollback, bounded replies, Unicode, redaction), and performance regressions that
  compare times on the same machine.

### Changed

- **Faster bulk creation (`Scene._create_many`)** and **automatic names in O(1)**:
  finding a free automatic name scanned every taken name, so N creations were quadratic.
- **Procedural scripts create N objects in O(N):** every step scanned every object
  (10,000 objects: 5,977 ms before, 820 ms now).
- **Plans pause Python's cyclic garbage collector while they run** (restored after, and
  left off if the caller turned it off). In a process holding many objects, a full
  collection triggered by a plan walked all of them: an 8,000-object plan in a process
  holding a million objects took 256-278 ms, now 117-121 ms.
- **Edit plans no longer walk the whole scene:** a one-object edit costs about 0.12 ms
  in a scene of any size.
- **Renders reuse one material record for objects of the same color** (a warm GPU frame
  of 10,000 cubes: 178 ms before, 118 ms now). Images are unchanged.
- **The `dev` extra includes `mcp`**, and CI checks that the wheel ships the plan guide,
  the example plans and the shaders, and runs the new examples and a benchmark smoke run.
- No change to the project format (still version 6), to existing tests or to the render
  goldens.

## 0.10.0: Scene intelligence and spatial queries

### Added

- **Spatial queries: `aether3d.spatial` and `Scene` methods**
  ([docs/SPATIAL.md](docs/SPATIAL.md)). Read-only answers about where things
  are, computed exactly from the triangles in world space (hierarchies,
  rotations, non-uniform and mirrored scales, shared meshes and evaluated
  procedural geometry included):
  - **Bounds:** `scene.bounds(obj, space="world"|"local", children=...)` and
    `obj.bounds()`: exact axis-aligned boxes, oriented boxes, enclosing
    spheres, and aggregates over hierarchies.
  - **Discovery:** `scene.find_objects(...)` by name (glob), kind, tags,
    metadata, parent or ancestor, visibility, material and region.
  - **Rays:** `scene.raycast` and `scene.raycast_all`: position, distance,
    geometric and shading normals, front/back, triangle, face group,
    barycentrics, UVs, material and instance identity. They run on the Stage 8
    two-level BVH and its watertight test; a ray through a shared edge or
    vertex reports one hit; `surfaces="rendered"` lets rays through alpha-mask
    cut-outs and backs of single-sided materials, as the path tracer does.
  - **Nearest points and distances:** `scene.closest_point`,
    `scene.nearest_objects`, `scene.distance(a, b)` (exact, with closest
    points; zero when touching).
  - **Containment:** `scene.contains_point(obj, point)`: inside, outside or
    surface for closed meshes, by the winding number; `"undefined"` (with the
    fractional winding number and the reason) for open, non-manifold or
    inconsistently oriented meshes.
  - **Overlaps:** `scene.overlaps(...)`: a broad phase on world boxes, then exact
    triangle tests (separating axes, edge crossings) and containment of one
    closed solid in another. Box overlaps are only reported as candidates
    (`exact=False`), never as collisions.
  - **Regions:** `scene.query_region(Box(...) | Sphere(...), mode=...)`: inside,
    intersecting, or enclosing the region.
  - **Cameras:** `scene.camera_ray` (the path tracer's camera ray, bit for bit),
    `scene.pick` (agrees with the rasterizer's object ids away from
    silhouettes), `scene.project`, `scene.objects_in_view` (exact frustum
    tests).
  - **Sight:** `scene.line_of_sight`, `scene.visible_from` (deterministic
    area-weighted surface samples; a camera observer also limits the view to
    its frustum).
  - **Relations:** `scene.relation(a, b)`: distance, intersecting, near, above,
    below, inside, contains, in front of, behind, and nearer to a viewer, with
    documented definitions and tolerances.
  - **`scene.describe()`:** a compact JSON summary of the scene for tools and
    agents.
  - Results are frozen dataclasses with `to_dict()`; errors are `QueryError`
    (`code`, `location`, `hint`).
- **JSON requests and the command line:** `aether3d.run_query(scene,
  request)`, `aether3d query QUERY PATH KEY=VALUE ...` and `aether3d query run
  PATH --request FILE` (one request or a list), and a read-only `query`
  operation in scene scripts. Python, CLI and scripts give the same JSON.
- **Object tags and metadata:** `obj.tags` (with `add_tags`, `remove_tags`,
  `has_tag`) and `obj.metadata` (JSON, with `set_metadata`, `remove_metadata`).
  They are undo steps, saved in projects and history files, written to and read
  from glTF (`node.extras.aether3d`), and copied by `duplicate` and `merge`.
  Scene scripts (`set_object`) and `aether3d object set` (`--tags`, `--add-tag`,
  `--remove-tag`, `--metadata KEY=VALUE`) set them.
- **`history.in_progress`:** whether an operation or transaction is open.
- **Examples:** `examples/castle_spatial.py` inspects the Minecraft castle (a
  height map from rays, exact contacts, relations, sight lines through the
  gate, visibility, picking checked against the rasterizer);
  `examples/hero_picking.py` picks and probes the blocky hero (clicks, an id
  map, dropping a crate onto the ground, contacts). Both draw cheap diagnostic
  images (`examples/output/castle_spatial_*.png`, `hero_picking.png`).
- **Tests:** 84 new tests. Geometric kernels are compared with
  independent references (Moller's interval test for triangle pairs, polygon
  clipping for triangle-box tests, closest points and segment distances).
  Queries are compared with brute force and analytic answers through every
  kind of transform, instancing, procedural modifiers, hidden and degenerate
  geometry, shared edges and grazing rays. Picking is checked against the
  rasterizer's object ids, and camera rays against the path tracer's, bit for
  bit. Also tested: determinism, cache invalidation through edits, undo, redo
  and rollback, history neutrality, serialization, and Python/CLI/script parity.

### Changed

- **Project format 6** adds `tags` and `metadata` to every object. Version 5
  projects load unchanged (the migration adds nothing); 0.9.0 history files
  load too.
- **The ray tracing core's boxes are closed.** `SceneAccel` grows every node
  and instance box by one ulp when it is built. A ray with a zero direction
  component lying exactly in the plane of a box face used to miss the box on
  one side (the slab test computed `0 x 1/0` there); it now reaches the
  triangles, whose watertight test decides. Renders change only for such
  rays: no golden image or test changed. A new test covers it.
- **Changed old tests:**
  - `tests/test_material_persistence.py` and `tests/test_pathtrace_api.py`
    asserted that projects are written as format version 5; they now expect
    6.
- `aether3d.QueryError` and `aether3d.run_query` are in the public API.

## 0.9.0: Operations history (undo, redo and transactions)

### Added

- **`scene.history`: every change can be undone and redone exactly**
  ([docs/HISTORY.md](docs/HISTORY.md)). Changes through the Python API, scripts
  and the command line are recorded: one step per public call or transaction.
  - **Coverage:** objects (create, remove, duplicate, merge, rename), the
    hierarchy, transforms, visibility and appearance, cameras, lights and the
    active camera, the world and its environment, render settings, meshes and
    committed edit sessions, materials, textures and assignments, and
    procedural parameters, rules, nodes, sources, modifiers, assignments and
    bakes.
  - **Exact:** undo gives back the same project document and the same Python
    objects (removed objects, materials, textures and procedural handles come
    back as the same instances), and the same renders, pixel for pixel, on the
    CPU and GPU rasterizers and path tracers.
  - **API:** `undo(steps)`, `redo(steps)` (all or nothing), `can_undo`,
    `can_redo`, `entries`, `redo_entries` (`HistoryEntry`: index, label,
    changes, paths, bytes), `clear()`, `to_dict()`, and `scene.revision`.
  - **Transactions:** `with scene.transaction("label") as tx:` makes one
    step, and rolls everything back if the block raises. Transactions nest as
    savepoints; `tx.rollback()` restores the start.
  - **Checkpoints:** `checkpoint(name)`, `goto(name)`, `checkpoints`,
    `remove_checkpoint(name)`. They cost no memory.
  - **Limits:** `max_entries` (256 steps) and `max_bytes` (256 MiB), with
    memory accounting. Steps store the before/after states of the parts they
    changed, never a copy of the scene, and share meshes and images by
    reference.
  - `history.disabled()` and `history.enabled` switch recording off.
  - **Failed operations change nothing.** Every public call is rolled back
    whole if it raises.
- **History files:** `scene.save(path, history=True)` and `Scene.load(path,
  history=True)` keep the history next to the project (`<project>.history`),
  so a later session can undo.
  - **Tied to their project.** A history file records its project's
    fingerprint, and a mismatch is refused (`HISTORY_MISMATCH`).
  - **Checked by replay.** Loading replays the history on a copy before using
    it (`INVALID_HISTORY` if it is not exact).
  - **Verification on demand.** `history.verify()` replays the current history
    through the file format.
- **CLI:** `aether3d history start|show|undo|redo|checkpoint|goto|clear|verify
  PROJECT`.
  - **Recording on the command line.** Every command that changes a project
    records its change when the project has a history file, or with
    `--history`. Undo then works across commands.
  - **Merging from the command line:** `aether3d scene merge PROJECT SOURCE`.
- **`scene.duplicate(obj, name=None, parent=..., children=True)`:** copies
  objects in one step. Meshes and geometry nodes are shared.
- **`scene.merge(source, parent=None, allow_lossy=False)`:** copies another
  scene, project or glTF file into the scene, in one undoable step. Materials
  and textures come along, and procedural objects become static meshes.
- **Scripts:** scene, edit and procedural scripts take an optional `"label"`
  (and a `label=` argument) that names their history step.
- **`HistoryError`** with stable codes; `History`, `HistoryEntry` and
  `Transaction` in the public API.
- **Example:** `examples/history_undo.py` makes 13 changes (a transaction, a
  rolled-back transaction, a checkpoint, a glTF import), undoes them back to
  identical pixels, and saves and replays the history.
- **Tests:** `tests/test_history.py`, 91 tests.
  - **61 kinds of change**, each undone and redone twice to the same project
    document and the same objects.
  - **A random chain:** 150 random changes undone and redone through every
    intermediate state.
  - **Failures and grouping:** failed operations, transactions, savepoints,
    redo branches, checkpoints, limits and memory.
  - **Files and imports:** scripts, procedural undo, loading, importing and
    merging, history files (tampered ones too).
  - **Behavior around undo:** renders after undo, caches, the CLI.

  There are also GPU tests for renders after undo, and an example test. Every
  test of the suite now runs with `history.STRICT`, which re-checks the scene's
  indexes after every undo, redo and rollback.

### Changed

- **Atomic by construction.** Scene changes run as operations of the history.
  - The scene, edit and procedural script runners and
    `MaterialLibrary.transaction()` now roll back through it; their own
    rollback code is gone. `MaterialLibrary.transaction()` now also rolls back
    non-material changes made in its block.
  - `ProceduralSystem.batch()` is a history operation too, and keeps its own
    validation and restore.
  - Their results are unchanged.
- **The procedural system's own `undo()`/`redo()`** stay (procedural
  definitions only), and each call is now a step of the scene's history.
- **Loading or importing a scene starts an empty history.**
- **Removed objects keep a reference to the scene that removed them,** whose
  history records their changes.
- **The project format is unchanged** (version 5); history files are separate.
- **Cost:** recording adds about 3 µs per change (setting a position: 5 µs
  instead of 2 µs). Building the path tracing castle (108 steps) takes the
  same 0.31 s, with 0.9 MiB of history.

## 0.8.1: GPU path tracing

### Added

- **The path tracer on the GPU: `render(backend="gpu_path")`**
  ([docs/RENDER_QUALITY.md](docs/RENDER_QUALITY.md#on-the-gpu-backendgpu_path)).
  A wgpu compute kernel runs the CPU path tracer's estimator on a GPU
  adapter: on the development machine's RTX 4060, on its shader cores
  through Vulkan.
  - **The same paths as `backend="path"`.** The kernel draws exactly the
    CPU path tracer's random numbers (SplitMix64 in 32-bit integer pairs),
    so with the same settings both backends trace the same paths, and their
    images agree to float32 rounding (the Cornell box: median relative
    difference 7e-6, the same bounce-ray count).
  - **The same scene data.** It uses the CPU path tracer's two-level BVH
    (with its BLAS cache), emitter table and environment, and the GPU
    rasterizer's materials, textures and procedural graphs.
  - **Every feature of the CPU path tracer:**
    - global illumination and metallic reflections;
    - emissive area lights with MIS, sized lights, sky environments;
    - alpha cut-outs and single-sided surfaces;
    - AO and the AO preview;
    - every pass and statistic, `progress` and stopping.
  - **Reproducible:** bit for bit on one device, however the image is
    tiled.
  - **Fast:** on the Stage 8 showcase scenes at their CPU settings, the RTX
    4060 is 51 to 65 times faster than the CPU path tracer with 8 processes
    on the i9-12900H. The castle at 1920x1080, 64 samples, 4 bounces takes
    18 s against 15.6 minutes.
  - **Software traversal:** the BVH is traversed in the kernel; no hardware
    ray tracing (RT cores) is used.
- **CLI:** `aether3d render --backend gpu_path` with the path tracer's options
  and `--adapter`.
- **Graphics APIs:** tested on Vulkan (the RTX 4060 and lavapipe) and Metal
  (GitHub's macOS runners).
- **`AETHER3D_GPU_DXC`:** the path of `dxcompiler.dll` makes Direct3D 12
  compile shaders with DXC instead of FXC, and the GPU path tracer needs it
  there. Measured on GitHub's Windows runners (Microsoft WARP):
  - FXC needed about 13 minutes for the kernel, and its code crashed WARP;
  - DXC compiled it in 12 s, but WARP did not finish a 24x24 test render in
    36 minutes.

  So WARP is refused (`GPU_FEATURE_UNSUPPORTED`, with the reason). With
  `adapter="auto"` the path tracer uses only adapters that can run it: a
  Windows GPU through Vulkan. Direct3D 12 on hardware is untested. A compile
  over a minute is reported as `SLOW_SHADER_COMPILE`.
- **Examples:** `examples/gpu_path_showcase.py` renders the Stage 8 showcases on
  the GPU with many more samples, and compares the CPU and GPU tracing the
  same paths.
- **Benchmark:** `benchmarks/bench_gpu_pathtrace.py` renders the showcase scenes
  at their CPU settings on the GPU and compares with the recorded CPU times.
- **Tests:** `tests/test_gpu_pathtrace.py` (29 tests):
  - the GPU's random numbers equal the CPU's;
  - per-pixel agreement with the CPU path tracer on every feature;
  - an exact white furnace and the inverse-square law on the GPU;
  - determinism across tilings, `progress` and stopping;
  - honest reports, limits and errors, the CLI;
  - the slow-compile report, the DXC setting, and the adapters that refuse the
    path tracer.

  The GPU CI jobs run them: lavapipe on Linux, with a quick GPU showcase run,
  and Metal on macOS. On Windows, WARP refuses the path tracer, and the job
  checks that refusal. The rasterizer tests run there twice, with FXC (the
  default) and with the Windows SDK's DXC.

### Changed

- **Device limits:** GPU devices are created with the adapter's maximum
  `max-storage-buffers-per-shader-stage` (the path tracer's kernel binds 11
  storage buffers; WebGPU's default is 8). Rasterizer renders are unchanged.
- **`SceneAccel`** records how many of its nodes form the top level
  (`top_nodes`).
- **`examples/showcase.py`** records GPU path-traced renders (the adapter
  instead of the CPU processes).
- **`backends.BACKENDS`** includes `"gpu_path"`.

### Compatibility

- **Unchanged images:**
  - the CPU and GPU rasterizers and the CPU path tracer render as in 0.8.0;
  - the render goldens and the path-traced golden image pass;
  - `examples/gpu_render.py` reproduces its committed images pixel for
    pixel.
- **No format change and no new dependency** (wgpu is the existing optional
  `gpu` extra).
- **Changed test:** `test_gpu_selection::test_backend_names` pinned
  `BACKENDS`, which gained `"gpu_path"`.

## 0.8.0: Render quality foundation (path tracing)

### Added

- **A path tracer: `render(backend="path")`**
  ([docs/RENDER_QUALITY.md](docs/RENDER_QUALITY.md)). It is physically based
  light transport computed on the CPU with NumPy, and it never uses the GPU
  (reports say `"device": "cpu"`).
  - **Global illumination:** unidirectional path tracing with next-event
    estimation and Russian roulette. `bounces` sets the limit (default 4;
    0 = direct light only).
  - **Shadows** by visibility from every light. They soften with the new
    light sizes: point-light `radius` and directional `angle`, both
    uniform-radiance models that keep each light's irradiance. Sized lights
    are found by BSDF-sampled rays too (MIS with light sampling), so mirrors
    reflect a sun or a lamp cleanly.
  - **Emissive surfaces are area lights,** sampled by power and combined
    with BSDF sampling by multiple importance sampling (power heuristic).
  - **Reflections** of the scene and the environment: Stage 7's
    metallic-roughness BRDF (divided by pi), sampled with a cosine lobe and
    GGX visible normals (Heitz 2018) as one mixture density.
  - **Materials** are evaluated by the rasterizer's own code: textures, mip
    levels (ray differentials on camera rays, ray cones on bounces),
    procedural graphs, normal and bump maps, triplanar mapping, face groups,
    alpha cut-outs (for shadow rays too), single-sided materials and
    textured emission.
  - **Environment:** `scene.world.environment` is `"ambient"` (the ambient
    light as a uniform environment; the default), `"uniform"` or a
    procedural `"sky"` gradient.
  - **Ambient occlusion** (`ao=`): a separate pass, never multiplied into
    the image. `path={"integrator": "ao"}` is a fast AO-only preview.
  - **Passes:** `result.passes` holds `beauty`, `direct`, `indirect`,
    `albedo`, `normal`, `depth`, `ao`, `sample_count` and `variance`.
  - **Statistics:** rays by kind, rays per second, a noise estimate, BVH
    statistics, per-stage timings and the work estimate.
  - **Reproducible:** counter-based random numbers and per-pixel summation
    in sample order make renders bit-identical for the same seed, whatever
    the batch size or number of worker processes. Across platforms, rounding
    differences in math libraries do not change which directions are
    sampled; a fixed image is compared on Linux, macOS and Windows in CI.
    `progress` reports every sample pass and can stop the render.
  - **Parallel on Linux:** forked worker processes trace tiles of pixels
    into shared memory (`workers`). With 8 processes, renders on an
    i9-12900H are 2.6–4.2 times faster than with one; more processes did not
    help. Other platforms use one process.
- **A ray tracing core** (`aether3d.rendering.raytrace`):
  - binned-SAH BVHs built vectorized and in bounded memory (1M triangles:
    3.3 s, 189 MiB peak);
  - a two-level structure for instancing (rotations, non-uniform and
    negative scales) with bottom-level BVHs cached by mesh identity (weak
    references, LRU budget `AETHER3D_RAYTRACE_CACHE_MB`);
  - closest-hit and any-hit traversal;
  - watertight ray/triangle tests (Woop et al. 2013).
- **Output:**
  - `tonemap="neutral"` (Khronos PBR Neutral) for any backend;
  - `RenderResult.color` reports the output transform;
  - `result.save()` writes linear HDR `.hdr` (Radiance RGBE) and `.pfm`.
- **API:** `PathSettings`, `AOSettings` and `Environment` are exported from
  `aether3d`. `Light(radius=, angle=)` and `scene.add_light(radius=, angle=)`
  set light sizes. `RenderLimits` gains `max_path_rays` and
  `max_path_memory`.
- **CLI:**
  - `aether3d render --backend path` with `--spp`, `--bounces`, `--seed`,
    `--light-samples`, `--russian-roulette`, `--clamp`, `--workers`, `--ao`,
    `--ao-only`, `--ao-samples`, `--ao-distance`, `--ao-strength` and
    `--passes DIR`;
  - `--tonemap` and `.hdr`/`.pfm` outputs for every backend;
  - `light add/set --radius --angle`;
  - `scene set --environment ... --sky-*`;
  - scene scripts accept `radius`, `angle` and `environment`.
- **Examples** rendered by the path tracer, each with its own checks and a
  JSON record of settings, times and ray counts:
  - `examples/cornell_box.py`: color bleeding from bounce light;
  - `examples/metal_spheres.py`: chrome, gold, copper and plastic
    reflecting the scene;
  - `examples/ao_contact.py`: the AO preview beside an overcast path trace;
  - `examples/castle_path_render.py`: the textured castle at golden hour;
  - `examples/window_room.py`: a room lit by sunlight through its window,
    reaching most of the room only by bouncing;
  - `examples/hero_path_render.py`: the blocky hero with its materials under
    a sun and a sky.

  The Cornell box, the metals and the castle compare the rasterizer, direct
  light only and bounced light from one camera; the window room compares
  direct and bounced light, the hero the rasterizer and the path tracer.
- **Benchmarks:** `benchmarks/bench_raytrace.py` (BVH build, memory,
  traversal at 1k/100k/1M triangles) and `benchmarks/bench_pathtrace.py`
  (render costs by stage, processes, memory, building blocks, and the
  rasterizers on the same scenes).
- **Tests:** 7 new test modules. They cover:
  - BVH traversal against a brute-force reference, watertightness and edge
    cases;
  - closed-form radiometry: furnace, interreflection series, inverse-square
    law, form factor, sky irradiance, AO beside a wall;
  - shadows against a brute-force shadow-ray reference (a concave shape
    shadowing itself too), and no shadow acne;
  - sized lights reached by light and BSDF sampling alike: unbiased on a
    plane, and reflected by a mirror without fireflies;
  - rounding residues in normals (either sign, as math libraries leave
    them) do not change which directions are sampled;
  - a mirror against a mirrored camera, Cornell-box color bleeding, and a
    room lit through its window only by bounces;
  - the BSDF against quadrature;
  - material parity with the rasterizer;
  - convergence and bitwise reproducibility, in one process and in several;
  - API, CLI, HDR files and persistence;
  - a deterministic path-traced image compared in CI.

### Changed

- **`.aether` format version 5:** `scene.world.environment`, and `radius`
  and `angle` on lights. Version 4 files are migrated on load (the
  `"ambient"` environment, zero sizes). `tests/data/v0.7.0/studio.aether`,
  written by Aether3D 0.7.0, tests the migration.
- **`RenderResult`** gains `passes` (empty for the rasterizers) and `color`.
  `save()` also accepts `.hdr` and `.pfm`. Reports list `color` and
  `passes`.
- **`backends.BACKENDS`** includes `"path"`.

### Compatibility

- **Unchanged images:** the CPU and GPU rasterizers render bit-identically to
  0.7.0. The new scene data is ignored by them, and `tonemap` defaults to
  `"none"`. The render goldens (`tests/data/renders`) are unchanged.
- **Old projects load:** projects from 0.1.0–0.7.0 (format versions 1–4)
  load, and saving writes version 5. Aether3D 0.7.0 cannot read version 5
  files.
- **No new dependencies:** the path tracer needs only NumPy.
- **Changed tests:** every earlier test passes. Assertions that encoded the
  old format version or option lists changed:
  - `test_material_persistence::test_projects_round_trip_materials_textures_and_groups`
    pinned `format_version == 4`; it is 5.
  - `test_render_persistence::test_document_layout` compares the exact saved
    `world` and light dictionaries, which gained `environment` and
    `radius`/`angle`.
  - `test_gpu_selection::test_backend_names` pinned the backend list, which
    gained `"path"`.

## 0.7.0: Materials, textures and procedural shaders

### Added

- **Materials** ([docs/MATERIALS.md](docs/MATERIALS.md)): a scene's
  `MaterialLibrary` (`scene.materials`) holds materials and textures with
  stable ids and unique names. Objects refer to them, and so can each face
  group of a mesh (`obj.material`, `obj.group_materials`,
  `scene.assign_material(..., group=k)`). Changes are validated all or
  nothing, and `transaction()` makes a block atomic. `inspect()` and
  `validate()` report bindings, users and problems.
  - **Models:**
    - `"standard"`: glTF 2.0's metallic-roughness BRDF (GGX, Smith
      height-correlated visibility, Schlick Fresnel; ambient light through
      Karis' environment BRDF fit);
    - `"simple"`: the 0.6 model. An object's color is exactly a simple
      material.
  - **Parameters:** base color, opacity with alpha masks, metallic,
    roughness, specular, shininess, emission and its strength, normal and
    occlusion strengths, double-sidedness.
  - **Texture slots:** base color, opacity, metallic, roughness, normal
    (tangent-space), emissive and occlusion. Each binding chooses a channel,
    UV/object/world coordinates, a projection (triplanar, planar, volume) and
    a transform.
- **Image textures:**
  - **PNG** decoded by Aether3D's own decoder (every color type and bit
    depth, Adam7, palettes, `tRNS`), with CRC, zlib-bound and size checks
    before allocating. **JPEG** through Pillow (the new `images` extra).
  - **Sampling:** mip chains (sRGB-correct, renormalized for normal maps),
    bilinear and trilinear filtering, repeat/clamp/mirror wrapping and
    isotropic level selection, shared by both renderers.
  - **Files:** stored as paths relative to the project, or embedded. A
    content-addressed image cache re-reads edited files.
- **Procedural textures:** node graphs stored as data. 16 node types:
  constant, coordinates, checker, gradient, stripes, noise, fbm, voronoi,
  bricks, tiles, wood, mix, ramp, math, bump, image.
  - Evaluated per sample with NumPy on the CPU, and compiled to fixed
    instructions run by an interpreter in the GPU shader.
  - Reproducible integer hashing (PCG3D) gives the same patterns on both.
  - `preview()` renders a texture to an image; `bake()` (an explicit,
    optional output) writes a PNG.
- **Presets:** stone, brick, wood, metal, grass and fabric, with options.
- **Face groups:** `Mesh(groups=...)`, `mesh.with_groups(...)`.
  - Mesh editing: `set_group`, `select(group=)`; every operation
    propagates groups, and `FACE_GROUP_CHANGED` reports a merge across
    groups.
  - Procedural modifiers keep groups, booleans included (through Manifold's
    face ids; `FACE_GROUPS_DROPPED` otherwise).
- **Rendering:**
  - **CPU:** materials are evaluated per sample, with exact barycentric
    derivatives for mip levels and bump steps, normal maps, triplanar
    projections, alpha masks during rasterization, single-sided culling and
    emission.
  - **GPU:** the same materials, from two storage buffers (material data
    with texture programs, and a texel pool keyed by image content); a
    masked shadow pipeline gives cut-out materials holes in their shadows.
  - Missing or broken textures and missing UVs are reported, with a factor
    fallback; they never abort a render.
- **glTF:** materials import and export: pbrMetallicRoughness, alpha
  modes, double-sided, normal, occlusion and emissive textures, samplers,
  `KHR_texture_transform`, `KHR_materials_emissive_strength` and
  `KHR_materials_specular`.
  - Primitives with different materials become face groups.
  - Face groups export as primitives.
  - `TANGENT` attributes are written for normal-mapped meshes.
  - Procedural textures are baked into PNG only on request
    (`bake_textures=SIZE`, `aether3d export --bake-textures SIZE`).
  - Everything glTF cannot express is reported.
- **CLI and scripts:**
  - `aether3d material` (`list`, `presets`, `schema`, `create`, `set`,
    `assign`, `texture`, `inspect`, `validate`, `remove`, `convert-colors`,
    `preview`) and `aether3d texture` (`list`, `add`, `procedural`, `set`,
    `inspect`, `preview`, `remove`). All are atomic, with `--dry-run`,
    `--output` and `--json`; `material validate` exits with code 3 on
    errors.
  - Scene-script operations for materials and textures, rolled back with
    the rest of a failed script. Script image paths must stay inside the
    project.
- **Public API:** `Material`, `MaterialLibrary`, `ImageTexture`,
  `ProceduralTexture`, `TextureBinding`, `TextureGraph` and `MaterialError`
  are exported from `aether3d`. `result.stats["timings"]["textures"]`
  reports material and texture preparation; the CPU adds
  `stats["alpha_discarded"]`.
- **Examples:**
  - `examples/make_textures.py` generates original example textures,
    baked from procedural graphs: stone bricks (color and normal map), wood
    planks (JPEG) and a UV grid.
  - The castle and the hero render with materials, beside their flat
    versions (`castle_flat_vs_materials.png`, `hero_flat_vs_materials.png`).
  - `examples/gpu_render.py` checks the textured castle on the GPU against
    the CPU.
- **Benchmarks:** `benchmarks/bench_materials.py` measures image decoding,
  mip chains, procedural nodes, shading cost per kind of material on both
  backends, texture uploads and memory.
- **Tests:** 11 new test modules (`tests/test_material*.py`,
  `tests/test_gpu_materials.py`). They cover:
  - image decoding: every PNG variant, malformed and hostile files;
  - sampling, noise and graphs;
  - the library and its transactions;
  - CPU rendering against analytic BRDF values;
  - CPU/GPU parity for textures and every node type;
  - persistence and migration;
  - face groups through editing and modifiers;
  - glTF in both directions, with the Khronos validator;
  - the CLI and scripts.

### Changed

- **`.aether` format version 4:** `scene.materials`, `scene.textures`,
  `material` and `group_materials` on objects, and optional `groups` on
  meshes. Version 3 files are migrated on load (no materials).
- **Object colors and materials:** while an object has a material, setting
  its color raises `MaterialError("OBJECT_HAS_MATERIAL")`, so a color never
  silently does nothing.
- **glTF import:** materials, textures, images and samplers are now
  imported. The `MATERIALS_IGNORED`, `TEXTURES_IGNORED`, `IMAGES_IGNORED`
  and `SAMPLERS_IGNORED` diagnostics are gone. Images are read from buffer
  views, data URIs and files in the asset's directory.
- **glTF export:** objects with a material, or a color other than the
  default, export glTF materials.
- **Dependencies:** new optional extra `images` = `pillow>=10` (JPEG
  textures). NumPy is still the only required runtime dependency.

### Compatibility

- **Old projects load:** projects from 0.1.0–0.6.0 (format versions 1–3)
  load unchanged, and saving writes version 4. Aether3D 0.6.0 cannot read
  version 4 files.
- **Unchanged images:** scenes without materials render bit-identically to
  0.6.0, on the CPU and on the GPU. Verified on 11 scenes on the CPU, and
  on the castle, hero and primitives renders committed with 0.6.0 on an
  RTX 4060.
- **Changed tests:** every earlier test passes. Assertions that encoded
  0.6.0 behavior changed:
  - `tests/test_render_persistence.py`: three assertions hard-coded
    `format_version == 3` and now use `FORMAT_VERSION`. One checked the last
    four object keys and now checks the four before `material` and
    `group_materials`.
  - `tests/test_gltf_import.py`: the Box sample's import now reports no
    diagnostics (its material is imported, not `MATERIALS_IGNORED`) and
    checks the material. `test_unicode_percent_encoded_uris_and_images` (was
    `..._images_never_loaded`) now checks that the image becomes a texture,
    and that a missing image gives `TEXTURE_MISSING`.
  - `tests/test_cli.py`: `validate`/`inspect` of a glTF file now report no
    import diagnostics.
  - `tests/render_reference.py` (the independent ray caster) learned
    material factors and the glTF BRDF, so the unchanged
    `test_gltf_imported_meshes_render` still compares like with like.
- **Examples:** `build_scene()` of the castle and hero examples still
  builds the flat-colored scenes (the render goldens use them); materials are
  `build_scene(..., materials=True)`.

## 0.6.0: GPU rendering engine

### Added

- **A GPU rendering backend** (`scene.render(backend="gpu")`,
  `aether3d render --backend gpu`; guide: [docs/GPU.md](docs/GPU.md)): real,
  headless GPU rasterization through wgpu (WebGPU) on Vulkan, Metal or
  Direct3D 12, written for Aether3D.
  - **WGSL shaders:** vertex pulling from cached storage buffers (exact face
    normals, shared meshes uploaded once, instanced draws) and the Stage 5
    shading model.
  - **Depth and culling:** camera-relative float64-to-float32 instance
    matrices, reversed float32 depth with hardware depth testing, and
    hardware back/front-face culling that is correct for mirrored instances.
  - **The compute finalize pass:** linear-light resolve, exposure, 8-bit sRGB
    encoding identical to the CPU's, view depth, object index, coverage and
    per-object visible-sample counts.
  - **Synchronized readback** and per-pass GPU timestamps.
  - **The same `RenderResult`** as the CPU renderer, plus `backend` (adapter,
    API, antialiasing, shadows, resources). GPU PNGs name their adapter in a
    `Source` text chunk.
- **Backend selection:**
  - `backend="cpu"` (default, unchanged);
  - `"gpu"` (an actionable `GPU_*` error, never a silent CPU fallback);
  - `"auto"` (GPU when it works, otherwise CPU with a `BACKEND_FALLBACK`
    diagnostic).
- **Adapter discovery and selection** (`aether3d.list_adapters()`,
  `aether3d.gpu_status()`, `aether3d gpu info [--check]`, `adapter=` /
  `--adapter` / `AETHER3D_GPU_ADAPTER`):
  - real API enumeration with types, APIs, drivers, features, limits and
    MSAA counts;
  - ranking: discrete GPU, then integrated, then software;
  - selection by index, kind, API or name;
  - a verified self-test render (`--check`).
- **MSAA** with device-supported sample counts (1/2/4/8/16 where available;
  unsupported counts are refused) and GPU supersampling. `samples` keeps its
  meaning (`samples²` per pixel). `supersample=s, msaa=1` reproduces the CPU
  renderer's sample positions (parity mode).
- **Directional-light shadow maps** (`shadows=True`, `ShadowSettings`, `--shadows`).
  - Fitting: maps fitted to the view frustum ∩ the scene, with square
    texels.
  - Bias: a constant depth bias plus a footprint-aware normal offset that
    keeps flat receivers free of acne at any angle.
  - Smooth surfaces: lookups start from the smooth surface the vertex
    normals describe (Hanika's shadow-terminator offset).
  - Filtering: PCF.
  - Verified against analytic shadows and an independent ray tracer.
- **GPU resource management:**
  - a mesh cache keyed by mesh identity, with weak references and an LRU
    budget (`AETHER3D_GPU_CACHE_MB`);
  - reusable render targets and per-device pipeline caches;
  - recovery from lost devices;
  - `RenderLimits.max_gpu_memory`;
  - resource statistics in every GPU result.
- **Validation and diagnostics:** `AETHER3D_GPU_DEBUG=1` enables wgpu and
  Vulkan validation layers (messages become `GPU_VALIDATION_MESSAGE`
  diagnostics). Error codes: `GPU_UNAVAILABLE`, `GPU_ADAPTER_NOT_FOUND`,
  `GPU_DEVICE_FAILED`, `GPU_FEATURE_UNSUPPORTED`, `GPU_LIMIT_EXCEEDED`,
  `GPU_OUT_OF_MEMORY`, `GPU_DEVICE_LOST`, `GPU_SHADER_ERROR`, and others.
- **Examples and benchmarks:**
  - `examples/gpu_render.py`: the castle on the CPU, on the GPU, and on the
    GPU with shadows, side by side, plus the hero and the primitives,
    checked against the CPU renderer and the CLI;
  - `benchmarks/bench_gpu.py`: CPU vs GPU on identical scenes, with the cold
    start separate, synchronized timings and memory.
- **Tests:** the GPU and its selection logic are checked by:
  - parity with the CPU renderer and with the independent ray caster;
  - analytic shading and projection values;
  - every geometry source;
  - culling with mirrored instances, extreme coordinates and 2,500 objects;
  - shadows against ray-traced references;
  - faults: device loss, shader errors, limits, validation layers;
  - the CLI, and selection on simulated adapters.
- **CI:**
  - a required GPU job renders on Mesa lavapipe (software Vulkan) with
    `AETHER3D_REQUIRE_GPU=1`;
  - Windows and macOS GPU jobs use whatever adapter the runner offers;
  - jobs without wgpu check that everything else still works.
- Optional dependency `gpu` = `wgpu>=0.32,<0.33` (Python 3.11+; BSD-2-Clause,
  with wgpu-native MIT OR Apache-2.0). The `dev` extra includes it.

### Unchanged

- The `.aether` format stays at version 3: shadows, MSAA and adapters are
  render options, not scene content, and 0.5.0 can read 0.6.0 projects.
- The CPU renderer, its defaults, its output and its tests are unchanged.

## 0.5.0: Cameras, lighting and the first renderer

### Added

- **Aether3D's own CPU renderer** (`aether3d.rendering`, `scene.render()`,
  `aether3d render`; guide: [docs/RENDERING.md](docs/RENDERING.md)). It is a
  deterministic, headless NumPy rasterizer with no external renderer, GPU,
  OpenGL context or GUI.
  - **Pipeline:**
    1. Backend-independent scene extraction (`RenderScene`).
    2. World and clip transforms.
    3. Back- and front-face culling.
    4. Homogeneous clipping against all six frustum planes, with per-vertex
       outcodes.
    5. Fixed-point (8 sub-pixel bits) rasterization with exact integer edge
       functions and a watertight top-left fill rule.
    6. An order-independent visibility buffer (packed depth + triangle id).
    7. Deferred shading with perspective-correct barycentrics, from the
       original triangle in homogeneous 2D coordinates.
  - **Antialiasing:** s×s ordered-grid supersampling (1–4), averaged in
    linear light.
  - **Output:** exposure, then the exact sRGB transfer function, then 8 bits,
    then a PNG from Aether3D's own encoder (adaptive filters, `sRGB` chunk,
    atomic writes).
  - **`RenderResult`:** the image plus linear light, depth, coverage and
    object-index maps, per-object visibility, camera information, counts,
    per-stage timings and diagnostics.
- **Cameras:**
  - perspective and orthographic, with vertical, horizontal or automatic
    field of view, near and far planes, and an optional preferred aspect
    ratio;
  - documented, tested projection matrices;
  - cameras look along local −Y with +Z up, and the frame is
    Gram–Schmidt-orthonormalized from the world matrix, so scale, shear and
    mirroring never distort the image;
  - `obj.look_at()` / `look_along()`, exact under scaled and rotated parents,
    with a defined fallback when looking along `up`;
  - an active camera (`scene.active_camera`);
  - exact vertex-based framing (`camera.frame()`, `render(frame=True)`).
- **Lights:**
  - directional and point lights (inverse-square or constant falloff, an
    optional smooth range window), plus ambient light (`scene.world`);
  - documented unitless intensities (non-physically-based);
  - a built-in key light when a scene has no lights.
- **Appearance:**
  - per-object solid color (display sRGB) and an optional Blinn–Phong
    highlight (`obj.appearance`, `obj.color`), plus `obj.visible`;
  - normals transformed with the inverse transpose (correct under mirroring);
  - flat shading for meshes without normals, and two-sided lighting;
  - a background color and render settings (`scene.render_settings`).
- **CLI:**
  - `aether3d render`, which renders projects or glTF files and frames
    automatically with `--frame`;
  - `camera add|set|list|frame|activate`, `light add|set|list`,
    `object set|remove`, `scene set|run`;
  - declarative scene-setup scripts (`aether3d.scene-script` v1,
    `run_scene_script`), atomic with exact rollback.
- **Examples:**
  - [`examples/first_render.py`](examples/first_render.py);
  - [`examples/minecraft_castle_render.py`](examples/minecraft_castle_render.py),
    the procedural castle rendered at 1920×1080, with an optional Blender
    silhouette comparison;
  - [`examples/hero_render.py`](examples/hero_render.py).

  Their PNGs are committed in `examples/output`.
- **Verification:**
  - an independent brute-force ray-caster reference
    (`tests/render_reference.py`);
  - analytic pixel tests and regression images (`tests/data/renders`);
  - Khronos validation of scenes with cameras and lights;
  - CI renders on Linux, macOS and Windows, uploaded as artifacts.
- **Benchmarks:** `benchmarks/bench_render.py` (per-stage timings and peak
  memory at three resolutions).

### Changed

- **`.aether` format version 3:** `scene.active_camera`, `scene.world`,
  `scene.render`, and `camera`, `light`, `appearance` and `visible` on
  objects. Versions 1 and 2 are migrated on load. The four object keys are
  optional when reading.
- **Data kinds:** an object holds at most one of mesh/geometry, camera and
  light. Assigning a mesh or procedural geometry to a camera or light is an
  error.
- **CLI `inspect`:** reports each object's `kind`, `camera`, `light`,
  `appearance` and `visible`, and marks cameras and lights in text output.
- **Dev extra:** includes Pillow, used by the tests only, to decode PNGs
  independently.

### Compatibility

- **Old projects load:** projects from 0.1.0–0.4.0 (format versions 1 and 2)
  load unchanged, and saving writes version 3. Aether3D 0.4.0 cannot read
  version 3 files.
- **Changed tests:** every earlier test passes. Three assertions in
  `tests/test_procedural_serialization.py` that hard-coded
  `format_version == 2` now use `FORMAT_VERSION`.
- **glTF:** cameras and lights export as plain nodes (they pass the Khronos
  validator). glTF cameras and `KHR_lights_punctual` are not written yet.
- **Dependencies:** NumPy is still the only required runtime dependency.

## 0.4.0: Procedural geometry engine

### Added

- **Non-destructive procedural geometry** (`aether3d.procedural`,
  `scene.procedural`; guide: [docs/PROCEDURAL.md](docs/PROCEDURAL.md)).
  - **Geometry nodes:** a source (a parametric primitive, a stored base
    mesh, or another node with parameter overrides and an optional `after`
    modifier) plus an ordered **modifier stack**.
  - **Modifiers** have stable UUID ids, unique names, typed and validated
    parameters, and an `enabled` flag. They can be inserted, moved and
    removed.
  - **Objects** reference nodes (`scene.add_object(..., geometry=)`,
    `scene.add_procedural`, `obj.set_geometry`), and `obj.mesh` returns the
    evaluated mesh. Base definitions are never modified.
- **Primitives:** `box`, `grid`, `cylinder`, `cone` (and frustum), `torus`
  and `uv_sphere`. All are watertight where closed and outward-wound, with
  UVs and analytic normals.
- **Modifiers:**
  - `transform` (pivot; mirroring scales re-wind faces);
  - `mirror` (multi-axis, merge with a threshold, interior cap removal, UV
    mirroring);
  - `array` (relative and constant offsets, per-step rotation and scale,
    boundary welding);
  - `solidify` (offset, even thickness, rims);
  - `subdivision` (vectorized Catmull–Clark with boundary rules, up to
    level 6);
  - `bevel` (the editing engine's chamfer, limited to all, angle or sharp
    edges);
  - `boolean` (union, difference and intersection with another node, using
    Manifold).

  Invalid input raises a stable error code. Noteworthy results are reported
  as diagnostics.
- **Parameters:** `float`, `int`, `bool`, `vec3`, `enum` and `string`, with
  ranges, choices, units and descriptions, at scene and node scope.
  - **Expressions** (`Expr`, `{"expr": ...}` in JSON) are a whitelisted
    arithmetic and logic language with resource limits. It is never Python
    `eval`.
  - **Rules** reject invalid parameter combinations
    (`CONSTRAINT_VIOLATED`).
- **Dependency graph:** node references and boolean operands, with
  `dependencies()` and `dependents()`, and cycle detection for both
  geometry and parameters (`DEPENDENCY_CYCLE`).
- **Content-addressed caching:** SHA-256 keys over the resolved inputs of
  each stack step. Results can never be stale, unchanged prefixes and nodes
  are reused, evaluated meshes are shared between instances, and the LRU
  cache is byte-bounded and reports statistics.
- **Transactions:** every change is validated, including evaluation of the
  affected geometry, and rolled back exactly on failure, keeping handles
  valid. `batch()` groups changes; `undo()` and `redo()` restore history.
- **Introspection:** `ProceduralSystem.types()` (every schema), `info()`
  (raw and resolved values, dependencies, users, evaluated statistics),
  `validate()` (per-node topology and diagnostics), and
  `parameter_dependencies()`.
- **Baking and source editing:** `obj.bake()`;
  `geometry.convert_source_to_mesh()`; `geometry.edit_source()`, which
  edits a base mesh with the editing engine and regenerates every user.
- **JSON procedural scripts** (`run_procedural_script`, format
  `aether3d.procedural-script` v1). They are atomic, report the failing
  operation and location, and never execute code.
- **CLI:** `aether3d procedural types | inspect | validate | run | set | bake`
  (JSON output; `--dry-run` and `--output`; exit code `3` when validation
  fails). `inspect` shows which geometry node an object uses.
- **Demo:** [`examples/minecraft_castle.py`](examples/minecraft_castle.py)
  builds a parametric castle: walls with battlements and an arched gate;
  towers with arrow slits and crowns; a keep, a roofed watchtower and
  instanced bushes. It verifies the castle's structure, then saves,
  reloads, modifies, rejects invalid changes, exports, validates with
  Khronos, compares in Blender (optional) and undoes.
- **Benchmarks:** `benchmarks/bench_procedural.py` (time and peak memory per
  modifier at three sizes, plus castle cache behaviour).
- **Optional dependency:** `aether3d[boolean]` installs
  [manifold3d](https://github.com/elalish/manifold) (Apache-2.0, ≥ 3.4).
  It is included in `dev`.

### Changed

- **`.aether` format version 2:** `scene.parameters`, `scene.rules`,
  `scene.geometry`, and an object `geometry` reference. Procedural objects
  store definitions, never evaluated meshes. Version 1 files are migrated
  on load.
- **Editing engine bevel is faster:** edge-to-face lookups are indexed
  instead of scanning every vertex fan. Results are identical; bevelling
  every edge of an 8,192-face mesh dropped from about 62 s to 3 s.
- **CLI:** `inspect` shows procedural objects (`geometry` in JSON,
  `procedural=` in text). Commands that need an evaluated mesh report
  evaluation failures as a `ProceduralError` (exit code `1`) with a
  location.

### Compatibility

- **Old projects load:** version 1 projects (0.1.0–0.3.0) load unchanged,
  and saving writes version 2. Aether3D 0.3.0 cannot read version 2 files.
- **No API changes for static objects.** All 0.1–0.3 APIs behave as before
  for objects without procedural geometry.
- **Two test changes:** the earlier tests pass unchanged, except two CLI
  tests that hard-coded `format_version: 1`. They now use
  `FORMAT_VERSION`.
- **NumPy is still the only required dependency.** Booleans need the
  optional `boolean` extra.

## 0.3.0: Mesh editing engine

### Added

- **Editable polygon meshes** (`aether3d.editing.EditableMesh`). The
  topology is BMesh-style (vertices, edges, faces, per-corner attributes)
  stored in id-keyed tables. It supports n-gons, boundaries, non-manifold
  geometry, and UV/normal seams. Ids are stable, and every operation reports
  the ids it created, removed and modified. `from_mesh()` (welding, and
  lossless quad reconstruction) and `to_mesh()` (ear-clipping triangulation)
  convert to and from the immutable `Mesh`.
- **Editing sessions:** `SceneObject.edit_mesh()` returns an `EditSession`.
  - Operations are atomic; a failure never leaves a partial edit.
  - Undo and redo restore exact state and ids.
  - Commits are validated, and a session refuses to commit over a mesh that
    changed meanwhile (stale-session protection).
  - Used as a context manager, a session commits on success and discards on
    exception.
  - Instancing is preserved: by default only the edited object changes;
    `scope="shared"` updates every object sharing the mesh.
- **Selection** by ids, box, sphere, plane, normal, nearest (ambiguity is an
  error), boundary, polygon size, or a predicate, in local or world space,
  with `expect=` count guards. Also edge loops, edge rings, linked faces,
  region boundaries, growth, and kind conversions.
- **Transforms:** translate, rotate, scale, arbitrary matrices, explicit
  positions, and moves along normals.
  - World-space edits are exact under rotated, non-uniform and mirrored
    parents.
  - Mirrored pieces are re-wound so they stay outward-facing.
  - Stored normals are transformed exactly.
- **Modeling operations:**
  - extrusion (region and individual, along normals or by an offset);
  - inset (individual and region, mitred, with depth);
  - edge bevel (chamfer, including box corners);
  - subdivision (quads, triangles, n-gons);
  - loop cuts (closed and open quad rings, with factor);
  - vertex merge, merge by distance and edge collapse;
  - edge dissolve and triangle-to-quad joining;
  - edge, face and vertex splitting, and face separation;
  - triangulation, polygon creation, box and grid creation;
  - deletion, flipping, and smooth/sharp shading control.

  Unsupported or unsafe cases raise `EditError` with a stable code,
  offending ids and a hint.
- **Attribute handling:**
  - UVs and normals are preserved exactly where untouched;
  - they are interpolated (linear, bilinear, barycentric or affine) where
    geometry is refined;
  - normals are recomputed from a flat/smooth/sharp shading model where
    geometry changed;
  - unavoidable loss is reported (`UV_DEGENERATE_FACES`, `UVS_UNDEFINED`);
  - corner layers are generic, so future attributes need no operation
    changes.
- **Topology validation** (`EditableMesh.validate()`): adjacency consistency,
  non-finite data, self-intersecting, zero-area and non-planar polygons,
  duplicate faces, coincident vertices, boundary and non-manifold edges,
  non-manifold vertices, inconsistent winding, concave faces, and loose
  vertices. Cross-face self-intersection is explicitly not evaluated.
- **Declarative JSON edit scripts** (`run_edit_script`, `ScriptError`): named
  selections, inline queries, `save`, validate/summary/describe/undo/redo
  steps, exact error locations, and all-or-nothing application. No code is
  ever evaluated.
- **CLI:** `aether3d edit` (atomic project updates, `--output`, `--dry-run`,
  scripts from stdin) and `aether3d topology` (polygon topology, validation
  and element listings).
- `examples/minecraft_hero.py`: a blocky hero and sword modelled entirely with
  editing operations, exported to GLB, then checked with the Khronos
  validator, a re-import, and optionally Blender (headless).
- `benchmarks/bench_editing.py`, plus documentation in `docs/EDITING.md`.

### Changed

- `inspect_mesh` (and `aether3d geometry`) now evaluates vertex manifoldness
  (`vertex_manifold`, `non_manifold_vertex_count`, `NON_MANIFOLD_VERTICES`).
  This was documented as not evaluated in 0.2.0.
- `EditError` and `ScriptError` join the error hierarchy.

### Compatibility

- **No `.aether` format change** (still version 1). `Mesh`, glTF import and
  export, and every 0.1.0 and 0.2.0 API behave as before; all earlier tests
  pass unchanged.
- **The CLI is backward compatible:** two commands were added, and the
  `topology` command uses exit code `3` like `geometry`.

## 0.2.0: Geometry interoperability and inspection

### Added

- **glTF 2.0 export** to `.glb` and `.gltf` with an embedded buffer:
  `Scene.export(path)` and `aether3d.export_gltf()`. It writes indexed
  triangle meshes with positions, normals and UVs, node transforms and
  hierarchy, and instanced shared meshes. Object ids go in
  `node.extras.aether3d.id`. Output is deterministic and written atomically.
- **glTF 2.0 import** from `.gltf` and `.glb`: `Scene.import_file(path)` and
  `aether3d.import_gltf()`, which returns an `ImportResult` with a
  node-to-object map and diagnostics.
  - **Supported data:** external, embedded and GLB buffers; interleaved and
    sparse accessors; triangle strips and fans; decomposable node matrices;
    multiple primitives.
  - **Unsupported content is reported explicitly.** Geometry-changing content
    (skins, morph targets, point/line primitives, sheared matrices) raises
    `UnsupportedFeatureError` unless `allow_lossy=True`.
  - **Hardened for untrusted files:** path confinement, URI scheme
    restrictions, `ImportLimits`, and strict JSON.
- **Exact Z-up ↔ Y-up conversion** (`aether3d.gltf.axes`), the same
  convention as Blender. Positions, normals, transforms and hierarchies
  convert bit-exactly; UVs are flipped `v → 1 − v` (round trip within 2.5e-7).
- **Geometry inspection:** `inspect_mesh()`, `inspect_scene()` and
  `Scene.inspect_geometry()`. Reports cover counts, local and world bounds,
  surface area, attribute availability, non-finite data, and degenerate and
  duplicate triangles. They also check unreferenced vertices, normal validity,
  normals that oppose the winding, and edge topology (closed, edge-manifold,
  consistent orientation, signed volume). Transform checks cover mirrored,
  sheared and zero scale. See `docs/INSPECTION.md`.
- `Diagnostic`, a structured finding with a stable `code`, `severity`,
  `location`, `count`, `samples` and `lossy` flag, shared by inspection,
  import and export.
- New errors: `AssetFormatError` (with `code` and a JSON-pointer `location`),
  `UnsupportedFeatureError` (with `diagnostics`), and `ResourceLimitError`.
- `Scene.unique_name(base)`.
- **CLI:**
  - new commands `aether3d import`, `aether3d export` and
    `aether3d geometry`;
  - `validate` and `inspect` also accept `.glb`/`.gltf`;
  - `--allow-lossy`;
  - exit code `3` when `geometry` finds error-severity problems;
  - JSON errors include `code` and `diagnostics` when available.
- `examples/gltf_roundtrip.py`, an end-to-end demonstration: build, export,
  validate with Khronos, import, inspect, and verify.
- **Verification tooling:**
  - the official Khronos glTF Validator, pinned in `tools/gltf-validator`
    (a new CI job);
  - trimesh as an independent parser in tests (development dependency only);
  - pinned Khronos sample assets in `tests/data/khronos`.
- `docs/GLTF.md` and `docs/INSPECTION.md`.

### Changed

- **Quaternion normalization is now idempotent.** Already-unit quaternions
  are left untouched instead of being divided by a norm of 1 ± 1 ulp, so
  round trips are bit-exact.
- **New UV spheres close their seam exactly.** The last column of vertices is
  now bit-identical to the first, where previously there was a ~1e-16 gap. The
  vertex and triangle counts are unchanged. Spheres stored in existing projects
  keep their original data.
- **Coordinate documentation clarified.** It said "+Y forward"; it now states
  that an asset's front faces −Y (as in Blender), which maps to glTF's +Z
  front. No behaviour depended on the previous wording.
- `Scene.save()` now shares its atomic-write helper with the exporters; the
  behaviour is unchanged.

### Compatibility

- **The `.aether` project format is unchanged (still version 1).** Projects
  written by 0.1.0 load, save and export without migration; this is covered by
  `tests/data/v0.1.0/first_scene.aether`.
- **The CLI is backward compatible.** Existing commands keep their arguments,
  output fields and exit codes; JSON output may contain additional fields.
- **Runtime dependencies are unchanged:** NumPy only.

## 0.1.0: Core foundation

- Scene graph with stable UUID ids, unique names and an ordered hierarchy.
- Transforms (position, quaternion rotation, scale) with local and world
  matrices.
- Immutable indexed triangle `Mesh`, plus cube, sphere and plane generators.
- Versioned, strictly validated `.aether` JSON project format.
- `aether3d new | validate | inspect` CLI with `--json` output.
