# Architecture

This document describes how the Aether3D foundation is organized, the
principles behind it, and where future systems are expected to plug in.

## Principles

- **Code-first.** Every capability is a Python call. The CLI is a thin layer
  over the same API, with no logic of its own beyond formatting.
- **AI-native.** Behaviour is predictable and failures are explicit:
  - inputs are validated eagerly, with messages that name the bad argument;
  - operations either fully succeed or leave state unchanged;
  - nothing is silently corrected (an explicit duplicate name is an error, an
    unknown key in a project file is an error);
  - identifiers are stable, names are unique, and output is deterministic;
  - every CLI command can answer in JSON, including errors with locations.
- **Small and replaceable modules.** Each module has one job and depends only
  on modules below it (see the dependency graph).
- **Data, not behaviour, at the boundaries.** Meshes are plain arrays, transforms
  are plain tuples, project files are plain JSON. This keeps the door open for
  native code and other tools without an API rewrite.

## Project structure

```text
Aether3D/
├── src/aether3d/          The Python package
│   ├── __init__.py        Public API surface (__all__)
│   ├── _version.py        Package version (single source of truth)
│   ├── errors.py          Exception hierarchy
│   ├── diagnostics.py     Diagnostic: structured findings (code, severity, location, ...)
│   ├── _fileio.py         Atomic file writes shared by all writers
│   ├── _features.py       Implementation switches (environment variables, docs/PERFORMANCE.md)
│   ├── _native.py         The optional native kernels (aether3d-native): loading, ABI check
│   ├── math3d.py          Stateless vector/quaternion/matrix helpers (NumPy)
│   ├── transform.py       Transform: position / rotation / scale
│   ├── mesh.py            Mesh: immutable indexed triangle mesh
│   ├── primitives.py      Cube, sphere and plane generators
│   ├── scene.py           Scene and SceneObject: the scene graph
│   ├── history.py         Undo/redo history: operations, aspects, transactions, limits,
│   │                      the change journal for derived caches
│   ├── _history_file.py   History files (<project>.history): encoding, checks, replay
│   ├── serialization.py   .aether reader/writer, validation, migrations
│   ├── inspection.py      Geometry analysis of meshes and scenes
│   ├── editing/           Polygon mesh editing engine (see docs/EDITING.md)
│   │   ├── core.py        EditableMesh storage, journaled primitives, transactions
│   │   ├── selection.py   Selection queries, loops, rings, conversions
│   │   ├── transform.py   Local/world transforms, mirroring, normal transforms
│   │   ├── topology.py    Create/delete/merge/split/dissolve/triangulate
│   │   ├── extrude.py     Region/individual extrusion and inset
│   │   ├── bevel.py       Edge chamfer and bevel_edges (dispatch)
│   │   ├── bevel_round.py Multi-segment bevels: profiles, open fans, corner patches
│   │   ├── cut.py         bisect and knife_project
│   │   ├── bridge.py      Boundary loops, fill_holes (Liepa), bridge_edge_loops
│   │   ├── slide.py       Edge and vertex slides
│   │   ├── cleanup.py     Dissolves, connections, cleanup reports
│   │   ├── shading.py     Auto smooth, consistent outward normals, weighted normals
│   │   ├── sweep.py       Edge extrusion, spin, lathe
│   │   ├── conform.py     Shrinkwrap
│   │   ├── subdivide.py   Subdivision and loop cuts
│   │   ├── normals.py     Shading model and corner-normal recomputation
│   │   ├── validation.py  Polygon-level topology validation
│   │   ├── session.py     EditSession: commit/discard on a scene object
│   │   ├── script.py      Declarative JSON edit scripts (CLI, future MCP)
│   │   └── _polygon.py    Newell normals, ear clipping, simplicity, affine fits
│   ├── sculpt/            Sculpting (see docs/MODELING.md)
│   │   ├── mesh.py        SculptMesh: welded geometry, render attributes, chunks,
│   │   │                  incremental normals, the journal, sculpt operations, commit
│   │   ├── brushes.py     Brush settings and validation, falloffs, Kelvinlets
│   │   ├── stroke.py      Strokes: paths, dabs, projection, symmetry, each brush's dab
│   │   ├── geodesic.py    Surface distances (fast iterative method) for geodesic falloff
│   │   ├── topology.py    Journaled topology changes: splits, collapses, flips, relaxation
│   │   ├── remesh.py      Isotropic remeshing and dynamic topology per dab
│   │   ├── voxel.py       Voxel remeshing: narrow-band distances, signs, marching cubes
│   │   ├── symmetry.py    Symmetrize by cutting and mirroring
│   │   ├── validate.py    check_mesh, validity tiers, welds, self-intersections
│   │   ├── session.py     SculptSession and SourceSculptSession: commits to the scene
│   │   ├── script.py      Declarative JSON sculpt scripts
│   │   ├── schema.py      Their JSON Schema (fuzzed against the executor)
│   │   └── guide.md       The guide for agents (an MCP resource)
│   ├── uv/                UV mapping (see docs/TEXTURING.md)
│   │   ├── topology.py    Welded half-edge topology, seams, chart vertices, components
│   │   ├── charts.py      Segmentation, flattening per chart, geodesic splitting
│   │   ├── param.py       LSCM, Tutte, SLIM (batched over charts), Jacobians
│   │   ├── sparse.py      Nested dissection, symbolic and numeric LDL^T (native or NumPy)
│   │   ├── metrics.py     Sander stretch, boundary self-overlaps
│   │   ├── pack.py        Texel footprints, orientations, skyline placement (native or NumPy)
│   │   ├── unwrap.py      unwrap(): the pipeline, partial unwraps, packing
│   │   ├── projection.py  Planar, box, cylindrical, spherical projections
│   │   ├── islands.py, ops.py  Islands; transform, align, normalize, density, pack, stitch
│   │   ├── report.py      uv_report(): exact flips, overlaps, stacking, stretch, density
│   │   ├── layout.py      Layout images (islands, density, stretch, overlaps, texture)
│   │   └── mikktspace.py  Bit-exact float32 MikkTSpace port, cached mesh tangents
│   ├── texturing/         Painting and baking (see docs/TEXTURING.md)
│   │   ├── raster.py      Exact UV rasterization of texel samples (fixed point, top-left)
│   │   ├── paint.py       Texel maps, brushes, strokes (surface, view, UV), fills, gradients
│   │   ├── bake.py        Cage rays, map evaluation, accumulation, apply_bake
│   │   ├── gpu_bake.py    Bake rays on the GPU (wgsl/bake.wgsl)
│   │   ├── dilate.py      Padding by frontier dilation
│   │   ├── transfer.py    Texture transfer onto remeshed copies
│   │   ├── script.py, schema.py  Texture scripts and their JSON Schema
│   │   └── guide.md       The guide for agents (an MCP resource)
│   ├── procedural/        Non-destructive procedural geometry (see docs/PROCEDURAL.md)
│   │   ├── polymesh.py    PolyMesh: immutable CSR polygon mesh flowing through stacks
│   │   ├── params.py      Parameters, safe expressions (Expr), rules, schemas
│   │   ├── primitives.py  Parametric primitive types (box, grid, cylinder, cone, ...)
│   │   ├── modifiers.py   Modifier types (transform, mirror, array, ..., boolean)
│   │   ├── graph.py       ProceduralSystem: nodes, dependencies, evaluation, cache, undo
│   │   ├── session.py     Editing a node's stored base mesh with the editing engine
│   │   └── script.py      Declarative JSON procedural scripts
│   ├── materials/         Materials, textures, procedural shaders (see docs/MATERIALS.md)
│   │   ├── material.py    Material, parameters, slots, TextureBinding, schemas
│   │   ├── textures.py    ImageTexture, ProceduralTexture, the image cache, project paths
│   │   ├── images.py      Safe PNG decoder (own) and JPEG decoding (optional Pillow)
│   │   ├── sampling.py    Mip chains, wrap modes, bilinear/trilinear sampling, LOD
│   │   ├── noise.py       Integer hashes (PCG3D), value/gradient noise, fbm, voronoi
│   │   ├── graph.py       Procedural node graphs: validation, CPU evaluation, GPU programs
│   │   ├── library.py     MaterialLibrary: a scene's materials, textures and assignments
│   │   ├── painted.py     PaintedTexture: tiled 16-bit layers, blend modes, masks, revisions
│   │   └── presets.py     Stone, brick, wood, metal, grass and fabric presets
│   ├── rendering/         Cameras, lights, the rasterizers and the path tracer
│   │                      (docs/RENDERING.md, GPU.md, RENDER_QUALITY.md)
│   │   ├── camera.py      Camera parameters, projection matrices, camera frames
│   │   ├── light.py       Light parameters
│   │   ├── settings.py    Appearance, World/Environment, RenderSettings, ShadowSettings,
│   │   │                  PathSettings, AOSettings, RenderLimits
│   │   ├── backends.py    Backend names (cpu/gpu/auto/path/gpu_path), option checks, antialiasing plan
│   │   ├── shadows.py     Backend-neutral shadow planning (lights, map fitting, bias)
│   │   ├── color.py       sRGB <-> linear transfer functions, color parsing
│   │   ├── scene_data.py  RenderScene extraction (backend-independent)
│   │   ├── retained.py    The retained render world: extraction updated from the change journal
│   │   ├── materials.py   Materials resolved per render (RenderMaterial), shared by both backends
│   │   ├── surface.py     CPU material evaluation, projections, normal maps, lighting models
│   │   ├── raster.py      Clipping, fixed-point rasterization, visibility buffer
│   │   ├── cpu.py         CPU backend: geometry, raster, deferred shading, resolve
│   │   ├── renderer.py    render() and RenderResult
│   │   ├── tonemap.py     Output transform: exposure, the optional tone mapper, sRGB encoding
│   │   ├── hdr.py         Radiance .hdr and .pfm writers (linear high dynamic range)
│   │   ├── framing.py     Exact camera framing
│   │   ├── png.py         PNG encoder (zlib from the standard library)
│   │   ├── script.py      Declarative scene-setup scripts
│   │   ├── gpu/           The GPU backend (optional wgpu; see docs/GPU.md)
│   │   │   ├── device.py      Adapters, selection, devices, GPUContext, wgpu error mapping
│   │   │   ├── shaders.py     WGSL modules, bind group layouts, pipeline cache
│   │   │   ├── resources.py   Mesh cache, render targets, exact shader data layouts
│   │   │   ├── materials.py   Material data packing, texture programs, the texel pool
│   │   │   ├── renderer.py    Shadow, scene and finalize passes; synchronized readback
│   │   │   ├── pathtrace.py   The GPU path tracer (backend="gpu_path"): data, dispatch, readback
│   │   │   ├── check.py       The self-test behind `aether3d gpu info --check`
│   │   │   └── wgsl/          common, material, scene, shadow, finalize and pathtrace shaders
│   │   ├── raytrace/      Ray tracing core (CPU, NumPy; see docs/RENDER_QUALITY.md)
│   │   │   ├── bvh.py         Binned-SAH BVH construction
│   │   │   └── accel.py       Two-level structure, BLAS cache, traversal, watertight tests
│   │   └── pathtrace/     The path tracer (backend="path")
│   │       ├── integrator.py  Camera rays, the bounce loop, passes, accumulation
│   │       ├── scene.py       Shading data, hit points, materials, pass-through surfaces
│   │       ├── bsdf.py        The metallic-roughness BRDF: evaluation and sampling
│   │       ├── lights.py      Point/directional lights, emissive triangles, NEE
│   │       ├── environment.py Ambient, uniform and sky environments
│   │       ├── sampling.py    Counter-based random numbers, warps, MIS
│   │       └── parallel.py    Tiles, rounds and forked worker processes
│   ├── spatial/           Spatial queries (see docs/SPATIAL.md)
│   │   ├── types.py       Regions (Box/AABB, Sphere) and result types with to_dict()
│   │   ├── geometry.py    Exact kernels: closest points, triangle intersection and distance,
│   │   │                  separating axes, winding numbers, components, BVH box queries
│   │   ├── predicates.py  Exact orientation predicates (float filters, rational fallback)
│   │   ├── surface.py     TriangleSurface: exact nearest points and ray casts on one mesh
│   │   ├── index.py       SpatialIndex: world-space geometry per scene revision (cached on the scene)
│   │   ├── rays.py        All-hits traversal of the Stage 8 BVH, shared-edge merging,
│   │   │                  rendered surfaces (the path tracer's pass-through rules)
│   │   ├── queries.py     The queries (bounds, rays, nearest, containment, overlaps,
│   │   │                  regions, cameras, sight lines, relations, describe)
│   │   └── request.py     JSON requests: the CLI's `aether3d query` and the script op
│   ├── plans/             Scene plans (see docs/SCENE_PLANS.md)
│   │   ├── model.py       Parsing and checking: symbols, references, patterns, selectors, issues
│   │   ├── patterns.py    Pattern expansion: grid, array, circle, line, scatter, instances, repeat
│   │   ├── executor.py    Binding to a scene, bulk creation, operations, queries, idempotency,
│   │   │                  chunks, select_objects
│   │   ├── schema.py      The format's JSON Schema (hand-written, fuzzed against model.py)
│   │   ├── limits.py      PlanLimits
│   │   ├── guide.md       The plan guide for agents (an MCP resource)
│   │   └── examples/      Example plans: basics, castle (2,190 objects), village (11,270)
│   ├── feedback/          Visual feedback and automatic corrections (see docs/FEEDBACK.md)
│   │   ├── views.py       Views: scene cameras, virtual cameras, framed close-ups, presets
│   │   ├── observe.py     Observations: render, per-object pixel and geometry facts,
│   │   │                  footprints and occlusion, image statistics, the observation store
│   │   ├── overlay.py     Annotated images (numbered markers), object colors, depth, masks
│   │   ├── expectations.py  The aether3d.expectations format and its checker
│   │   ├── geometry.py    Support (gaps, penetrations), contact and separation measurements
│   │   ├── checks.py      Diagnosis: checks, default checks, findings and their basis
│   │   ├── vision.py      Vision evaluations: checking, grounding, provenance; critics
│   │   ├── corrections.py The action allowlist, binding, compilation to plans and scripts,
│   │   │                  policy, proposals
│   │   ├── planners.py    Corrections per finding code (safe, verify, approval)
│   │   ├── session.py     FeedbackSession: observe, diagnose, propose, preview, apply, verify
│   │   ├── loop.py        The automatic loop: budgets, stop reasons, resume
│   │   ├── schema.py      JSON Schemas of the documents (fuzzed against the checkers)
│   │   └── guide.md       The guide for agents (an MCP resource)
│   ├── mcp/               The MCP server, optional (see docs/MCP.md)
│   │   ├── server.py      The protocol (official SDK, low-level server): one tool at a time
│   │   │                  in a worker thread, cancellation, progress, logging to stderr
│   │   ├── tools.py       The tools: argument schemas, implementations, replies, errors
│   │   ├── feedback.py    The visual feedback tools (observe, diagnose, propose, apply, loop)
│   │   ├── content.py     Resources and prompts
│   │   ├── session.py     The session: open scene, project and its disk state, configuration
│   │   └── workspace.py   The path policy: containment, suffixes, overwrites, redaction
│   ├── gltf/              glTF 2.0 interchange (no third-party dependencies)
│   │   ├── axes.py        Exact Z-up <-> Y-up conversion
│   │   ├── materials.py   glTF materials, textures and samplers, in and out
│   │   ├── writer.py      Export: scene -> glTF document + buffer -> .glb/.gltf
│   │   ├── reader.py      Import: validated, resource-limited, path-confined
│   │   └── anim_export.py, anim_import.py   Skeletons, skins, morph targets, animations
│   ├── animation/         Animation, rigging and cinematics (see docs/ANIMATION.md)
│   │   ├── time.py, easing.py              Exact frame rates and timecode; Penner easings
│   │   ├── curves.py, modifiers.py         F-curves (keys, handles, exact derivatives), modifiers
│   │   ├── rotation.py, tracks.py          Quaternion arrays; rotation tracks
│   │   ├── batch.py                        Batched evaluation of many curves and tracks
│   │   ├── paths.py                        The typed registry of animatable properties
│   │   ├── clip.py, nla.py                 Clips and key operations; bindings, layers, strips
│   │   ├── rig.py, ik.py, constraints.py   Skeletons, FK, attachments, skins; IK; constraints
│   │   ├── evaluate.py, rigging.py         The compiled program and a frame; the rigging pass
│   │   ├── system.py                       scene.animation: items, history, the API
│   │   ├── report.py, fitting.py, graph.py Measuring; reduction, resampling, baking; graphs
│   │   ├── generators.py, motion.py        Motion generators; mirroring, root motion,
│   │   │                                   contacts and planting, retargeting
│   │   ├── cinematics.py, frames.py        Paths, shots, camera moves; frame rendering
│   │   └── script.py, schema.py, guide.md  Animation scripts, their schema, the agent guide
│   ├── worlds/            Procedural worlds (see docs/WORLDS.md)
│   │   ├── definition.py                   Validated definitions, defaults, presets, biomes
│   │   ├── noise.py                        Deterministic noise: integer hashes, fBm, ridged,
│   │   │                                   eroded and warped fractals, exact cos/sin, blends
│   │   ├── features.py, fields.py          Authored features; the definition compiled into
│   │   │                                   pure field functions (height, climate, biomes, water)
│   │   ├── hydrology.py                    The regional stage: erosion, rivers, lakes (native)
│   │   ├── tiles.py, terrain_mesh.py       Quadtree LOD selection and stitching; tile meshes
│   │   ├── surface.py, materials.py        Surface colours; the world's layer materials
│   │   ├── vegetation.py, assets.py        Plant placement and culling; procedural species
│   │   ├── volumes.py                      Caves and overhangs: SDF, marching cubes
│   │   ├── blocks.py, voxels.py, edits.py  Block registry and pixel art; block chunks, greedy
│   │   │                                   meshing, coarse levels; block edits
│   │   ├── structures.py                   Structures, sites, roads, bridges, settlements
│   │   ├── runtime.py, workers.py          Generated data, caches, budgets, accounting; threads
│   │   ├── render.py, export.py            A world's render contribution; glTF regions
│   │   ├── system.py                       scene.worlds: items, history, the API
│   │   └── script.py, guide.md             World scripts and their schema; the agent guide
│   ├── cad/               Precision modeling (see docs/CAD.md)
│   │   ├── units.py, tolerance.py, expr.py   Quantities, tolerances, safe expressions
│   │   ├── nurbs.py, geom2d.py, geom3d.py    Exact curves and surfaces
│   │   ├── sketch/        Sketch model, equations, structure (Dulmage-Mendelsohn), solver, profiles
│   │   ├── brep/          B-rep topology, builders, tessellation, mass properties,
│   │   │                  intersections, booleans (Manifold + exact reconstruction), lofts, sewing
│   │   ├── document.py    The CAD document: items, kinds, incremental regeneration, links
│   │   ├── kinds.py       Parameters, frames, sketches, parts (handles)
│   │   ├── features.py    Part features; naming.py persistent names
│   │   ├── assembly.py    Assemblies: mates, solving, clashes
│   │   ├── measure.py     Measurements and snapping
│   │   ├── drawing/       Views, hidden lines, sections, SVG/DXF/PNG output, stroke font
│   │   ├── interop/       STEP (step.py), DXF (dxf.py); document entry points
│   │   ├── script.py, schema.py, guide.md   CAD scripts, their schema, the agent guide
│   │   └── triangulate.py Ear clipping (with the native twin)
│   ├── arch/              Buildings (see docs/BUILDINGS.md): walls and joins, families,
│   │                      floors and rooms, slabs, roofs, stairs, columns, beams,
│   │                      quantities and schedules, IFC4 (ifc.py), handles (api.py)
│   ├── cli.py             `aether3d` command-line interface
│   ├── _cli_materials.py  `aether3d material` and `aether3d texture`
│   ├── _cli_feedback.py   `aether3d feedback`
│   ├── _cli_sculpt.py     `aether3d sculpt`
│   ├── _cli_cad.py        `aether3d cad`
│   ├── _cli_animation.py  `aether3d animation`
│   └── __main__.py        `python -m aether3d`
├── tests/                 pytest suite, one file per module
│   └── data/              Khronos samples (pinned), v0.1.0/v0.4.0/v0.7.0 projects, reference renders
├── examples/              Runnable examples, a sample project, example textures, renders (output/)
├── tools/gltf-validator/  Pinned Khronos glTF Validator runner (Node.js, dev only)
├── benchmarks/            Editing, procedural, rendering, materials, ray tracing, path
│                          tracing and scene plan benchmarks (bench_*.py); the engine-wide
│                          suite (bench_engine.py) and its saved runs (results/)
├── native/                aether3d-native: optional Rust kernels (PyO3, maturin), exact
│                          counterparts of NumPy code (ray traversal, hierarchies, PNG filters,
│                          LDL^T, packing, triangulation, linear blend skinning)
├── research/              The October 2026 architecture audit and the performance
│                          modernization's investigations (not part of the package)
├── docs/                  API, file formats, glTF, inspection, architecture, install
└── .github/workflows/     Continuous integration
```

### Module dependencies

| Module | Imports (internal) |
| --- | --- |
| `errors`, `diagnostics`, `_fileio` | none |
| `math3d` | `errors` |
| `mesh` | `errors` |
| `history` | `errors`; `materials` lazily (texture states), `serialization` lazily (files) |
| `transform` | `math3d`, `history` (it records changes of an object's transform) |
| `primitives` | `mesh`, `math3d` |
| `scene` | `history`, `transform`, `mesh`, `primitives`, `math3d`, `errors` |
| `serialization` | `scene`, `transform`, `mesh`, `errors`, `_fileio` |
| `inspection` | `mesh`, `math3d`, `diagnostics` |
| `gltf.axes` | `math3d` |
| `gltf.writer` | `gltf.axes`, `mesh`, `diagnostics`, `errors`, `_fileio` |
| `gltf.reader` | `gltf.axes`, `gltf.writer` (constants), `scene`, `mesh`, `transform`, `math3d`, `diagnostics`, `errors` |
| `editing.*` | `mesh`, `math3d`, `diagnostics`, `errors`, `primitives` (box layout), `inspection` (session commit reports); `scene` only for type hints and the session |
| `procedural.*` | `mesh`, `math3d`, `diagnostics`, `errors`, `inspection`, `editing` (bevel, base-mesh sessions, polygon helpers); `scene` only for type hints; `manifold3d` (optional, lazily, boolean modifier only) |
| `serialization` (v2) | additionally `procedural.params` (value encoding), and `procedural` lazily when reading |
| `rendering.camera`, `.light`, `.settings`, `.color` | `errors`, `diagnostics`, `history` (value types that record in-place changes; imported by `scene` and `serialization`) |
| `rendering.*` (others) | the above, `mesh`, `math3d`, `_fileio`; `scene` only for type hints and lazily for lookups |
| `rendering.gpu.*` | `rendering` (scene data, settings, shadows, color, backends), `errors`, `diagnostics`; `wgpu` (optional, imported lazily on the first GPU call; never by `import aether3d`) |
| `materials.*` | `errors`, `diagnostics`, `rendering.color`, `_fileio`; `scene` only for type hints; Pillow (optional, lazily, JPEG only) |
| `rendering.materials`, `rendering.surface` | `materials`, `rendering` (scene data, settings, color), `diagnostics`, `errors` |
| `rendering.raytrace.*` | `rendering.scene_data` (types only); NumPy only |
| `rendering.pathtrace.*` | `rendering.raytrace`, `rendering.surface` (material evaluation), `rendering` (scene data, settings, color), `diagnostics`, `errors`; the standard library's `multiprocessing` and `mmap` for worker processes |
| `gltf.materials` | `materials`, `rendering.color`, `diagnostics`, `errors` |
| `_history_file` | `history`, `serialization`, `scene`, `materials`, `procedural` (states and entities), `_fileio` |
| `spatial.*` | `rendering.raytrace` (two-level BVH, watertight test), `rendering.scene_data` (world matrices, render-scene extraction), `rendering.pathtrace.scene` (rendered surfaces, lazily), `rendering.camera`, `inspection` (topology), `errors`; `scene` only for type hints and lazily for lookups |
| `plans.*` | `scene` (bulk creation, history), `primitives`, `procedural` (expressions, primitives, modifiers), `materials`, `rendering` (cameras, lights, colors, settings), `mesh`, `spatial` (query requests, lazily), `errors` |
| `feedback.*` | `rendering` (render, virtual cameras, framing, the CPU raster for footprints, PNG), `spatial` (index, rays, overlaps, frustum), `plans` (selectors, executing compiled corrections), `rendering.script` (compiled corrections), `history` (transactions, undo), `errors`; NumPy only |
| `uv.*` | `mesh`, `errors`, `diagnostics`, `spatial.predicates` (exact overlap tests), `_native` (LDL^T, packing); `texturing.raster` lazily (layout images) |
| `texturing.*` | `uv`, `materials` (painted textures), `rendering.pathtrace` (sources as path-tracer scenes: BVH, shading), `rendering.raytrace`, `rendering.gpu` (lazily, GPU bake rays), `spatial.surface` (nearest points), `history` (transactions), `errors`; `scene` only for type hints |
| `cad.*` | `errors`, `_native` (triangulation), `history` (aspects), `rendering.raytrace` (hidden lines), `rendering.png`; `manifold3d` (optional, lazily, booleans); `scene` only for type hints and through `scene.cad` |
| `arch.*` | `cad` (document, kinds, B-rep, sketches, interop helpers), `errors` |
| `animation.*` | `errors`, `diagnostics`, `history` (aspects), `mesh`, `transform`, `materials`, `rendering.scene_data` (world matrices), `rendering.overlay`, `rendering.camera`, `rendering.png`, `rendering.tonemap` (motion blur), `cad.drawing` (graph rasterizer and stroke font), `_native` (skinning); `scene` only for type hints and through `scene.animation` |
| `worlds.*` | `errors`, `diagnostics`, `history` (aspects), `mesh`, `rendering.scene_data` (render-scene parts), `rendering.materials`, `.color`, `.settings`, `.overlay`, `.png`, `sculpt.voxel` (marching cubes), `gltf.writer` and `scene` (region export, lazily), `_native` (hydrology); otherwise `scene` only for type hints and through `scene.worlds` |
| `gltf.anim_export`, `gltf.anim_import` | `gltf.axes`, `animation` (clips, curves, tracks, rigs), `diagnostics`, `errors` |
| `mcp.*` | `plans`, `scene`, `serialization`, `gltf`, `rendering`, `spatial`, `history`, `editing` and `procedural` (scripts), `feedback`, `errors`; the `mcp` SDK (optional: never imported by `import aether3d`) |
| `cli` | `editing`, `procedural`, `rendering`, `materials`, `gltf`, `serialization`, `scene`, `history`, `spatial`, `plans`, `feedback`, `errors`; `mcp` lazily (`aether3d mcp serve`) |

Lower layers never import higher ones. `scene` provides convenience methods
(`save`/`load`, `export`/`import_file`, `inspect_geometry`, `edit_mesh`) by importing
`serialization`, `gltf`, `inspection` and `editing` lazily inside those methods, and
`inspection`/`gltf.writer` refer to `Scene` only for type hints.

Animation feeds the renderers through one overlay: `scene.animation.overlay(t)` gives the
posed local matrices, field overrides and deformed meshes that `extract_render_scene` and
the retained world apply; nothing in `rendering` imports `animation` except lazily, to
evaluate at a requested time.

Worlds feed the renderers the same way: `renderer.render` asks `worlds.render.contribute`
(imported lazily, only when the scene has worlds) for each world's meshes and instances
for the frame's camera and appends them to the extracted render scene; the backends do
not know worlds exist.

## Core concepts

### Scene graph

A `Scene` owns a set of `SceneObject`s and indexes them by id and by name. Each
object holds a parent pointer and an ordered list of children; the scene keeps
an ordered list of roots. Iteration is depth-first, which is also the order
objects are written to disk, so files and CLI output are stable.

Objects are created only by the scene (`add_object`, `add_cube`, …), which
guarantees every object has a valid id, a unique name, and a consistent place
in the hierarchy. Removed objects are detached (`obj.scene is None`) and refuse
further hierarchy operations.

`SceneObject` is deliberately a generic node: today it carries a transform and
an optional mesh. Future capabilities (lights, cameras, materials, physics
bodies, …) are expected to attach to objects as additional optional data, and
the decision of how (dedicated attributes vs. a component registry) is left
until the first such capability exists, so the abstraction matches real needs.

### Identity

Ids are UUID4 strings, generated at creation and preserved by save/load. They
are the durable way to refer to an object (from scripts, files, or future
systems such as animation tracks). Names are unique per scene and are meant for
humans and convenient lookups; they can change, ids cannot.

### Transforms

`Transform` stores position, a unit quaternion, and scale as immutable tuples
of floats. Matrices are computed on demand (`T @ R @ S`), and world matrices
are the product of local matrices up the parent chain. Computing on demand
avoids cache-invalidation bugs; caching can be added inside `SceneObject`
later without changing the API.

`math3d` holds all the math as stateless functions over tuples and NumPy
arrays. It is the natural first candidate for a native implementation.

### Geometry

`Mesh` is an immutable indexed triangle mesh: `float32` positions, optional
normals and UVs, `uint32` triangle indices, and optional per-triangle face
groups (`uint16`, since 0.7.0; they select materials). These contiguous NumPy arrays
match what GPUs and native libraries consume, and they expose the buffer
protocol for zero-copy handoff to Rust/C++ extensions. Immutability makes
sharing a mesh between objects (instancing) safe.

New vertex attributes (colors, tangents, …) can be added as further optional
arrays without breaking existing code. Primitive generators are plain
functions returning a `Mesh`; future procedural tools can follow the same shape.

### Persistence

`.aether` files are versioned JSON (see [FILE_FORMAT.md](FILE_FORMAT.md)). The
reader validates everything and reports errors with a precise location. A
`MIGRATIONS` table upgrades old files one version at a time, and files from
newer versions are refused rather than partially read. Saves are atomic
(write to a temporary file, then rename).

Meshes are stored in a table and referenced by index, so shared meshes are
stored once. JSON keeps files transparent for agents and code review. If very
large meshes later need binary storage, that becomes a new format version
(e.g. a buffer reference in place of inline arrays) with a migration.

## Extending Aether3D

| Future system | Expected integration point |
| --- | --- |
| Rendering | Implemented in v0.5.0 (`rendering/`): backends consume a `RenderScene` from `extract_render_scene` (see below). |
| Animation | Targets objects by `id`; writes transform components through the public setters. The history already records every animatable property per aspect, under id-based paths (`objects[<id>].transform`, `materials[<id>]`, ...). |
| Operations history | Implemented in v0.9.0 (`history.py`): new mutating methods run as operations and touch the aspects they write (see below). |
| Physics | Reads/writes world transforms; stores per-object data alongside the transform. |
| Procedural modeling | Implemented in v0.4.0 (`procedural/`): new primitive and modifier types are registry entries (see below). |
| Materials / shaders | Replace `Appearance` (the only place surface properties enter shading) with materials; extend the per-triangle shading table. Format version bump. |
| Asset management / interop | Importers/exporters that convert to/from `Scene` and `Mesh`, outside the core. |
| Native acceleration | Replace `math3d` functions or mesh generation with Rust/C++ (e.g. PyO3) behind the same Python signatures; a compiled or GPU ray traversal can answer `SceneAccel.closest_hit`/`occluded` from the same flat BVH arrays. |
| Path tracing on the GPU | A wgpu compute integrator consuming the same `RenderScene`, BVH layout and counter-based sampling, verified against the CPU path tracer (future stage). |

Ground rules for extensions:

1. The core (`scene`, `transform`, `mesh`, `math3d`, `serialization`) must
   not import optional subsystems.
2. Any change to the file format bumps `FORMAT_VERSION`, adds a migration, and
   updates `docs/FILE_FORMAT.md`.
3. Public API additions are exported from `aether3d/__init__.py` and documented
   in `docs/API.md`.

## Interchange and inspection (v0.2.0)

Interchange code is an adapter around the core, not part of it. `gltf`
converts between external documents and the existing `Scene`/`Mesh` types,
and the core never imports it except lazily for convenience methods. Future
formats (OBJ, USD, FBX, ...) can be added the same way, as sibling packages
with their own reader and writer.

Several design decisions keep conversions predictable:

- **No third-party glTF library.** The container format is small. A direct
  reader and writer over `json`, `struct` and NumPy keep full control over
  validation, error locations, resource limits and exactness. Correctness is
  established externally instead: the official Khronos validator, an
  independent parser (trimesh, test-only), and pinned Khronos sample assets.
- **Exact axis conversion.** The Z-up → Y-up map is a signed axis
  permutation applied to TRS components, mesh data and every node.
  Conversion is therefore lossless and hierarchy-consistent, and needs no
  extra root node (see [GLTF.md](GLTF.md)).
- **One `Diagnostic` type for every report.** Inspection, import and export
  all report findings the same way: a stable `code`, a `severity`, a
  `location` and a `lossy` flag.
- **Lossy conversions are opt-in.** Content that a conversion would have to
  drop or approximate makes it fail with `UnsupportedFeatureError` listing
  every item, unless the caller passes `allow_lossy=True`. Non-geometric
  content that is merely skipped is reported but does not fail.
- **No partial state.** Imports build a new `Scene` and return it only on
  success. Exports build the complete file in memory, then write it
  atomically.
- **Inspection is read-only and vectorized.** It never mutates scenes, and
  each distinct mesh is analysed once. Its topology claims are limited to
  what it actually computes on exactly-welded vertices (see
  [INSPECTION.md](INSPECTION.md)).

## Mesh editing (v0.3.0)

The editing engine is a layer on top of the core: it consumes and produces
immutable `Mesh` objects and never changes how they are stored. The core
design decisions are explained in [EDITING.md](EDITING.md). Most importantly:

- **BMesh-style topology in id-keyed tables.** Polygons, non-manifold
  geometry and per-corner attributes, with stable integer ids, stored as
  NumPy positions plus dictionaries. The alternatives that were considered
  are compared in EDITING.md.
- **All mutation goes through journaled primitives.** Operations are atomic
  (exact rollback on any exception) and undoable, without copying meshes.
  `OperationResult` id lists are derived from the journal, so they are always
  complete.
- **Operations are plain methods on mixins** (`TransformMixin`,
  `ExtrudeMixin`, ...) composed into `EditableMesh`. A new operation is one
  `@operation` method in the right module; it is then automatically
  undoable, listed by `EditableMesh.operations()`, and callable from JSON
  scripts.
- **Attributes are generic corner layers.** Interpolation code is
  layer-agnostic, so vertex colours, extra UV sets or a per-face material
  index can be added without touching the operations.
- **Sessions bind an editable mesh to a scene object.** Committing is the
  only way geometry reaches the scene: instancing stays intact and failed
  edits never leak.
- **Scripts are data, not code.** `run_edit_script` is the single entry point
  for declarative edits. The CLI uses it today; a future MCP server can expose
  it unchanged.

## Procedural geometry (v0.4.0)

The procedural engine sits beside the editing engine, above the core. Objects
reference **geometry nodes**, and `SceneObject.mesh` returns the node's
evaluated `Mesh`. Everything downstream (inspection, glTF, the CLI) is
therefore unchanged and always sees current geometry. The design is explained
in [PROCEDURAL.md](PROCEDURAL.md); the key decisions:

- **A value type for evaluation.** Modifiers map an immutable `PolyMesh` to a
  new one: CSR polygons with per-corner UV and normal layers, smooth flags and
  sharp edges, as read-only NumPy arrays. Evaluation can therefore share
  arrays freely, caching is safe, and modifiers are vectorized. The editing
  engine's `EditableMesh` (id-keyed, journaled) is built only where an
  operation needs it (bevel, editing a base mesh).
- **Definitions are data.** A node is a source plus a list of `(id, name,
  type, enabled, params)` entries, and parameter values are literals or `Expr`
  objects. Serialization, undo snapshots, `info()` and JSON scripts all
  operate on that data. Nothing is executed as code.
- **Content-addressed caching.** Cache keys are SHA-256 hashes of fully
  resolved inputs, chained through the stack. Invalidation is implicit: a
  changed input gives a different key, so results can never be stale. There
  is no dependency bookkeeping to get wrong, and identical sub-evaluations are
  shared across nodes. The cache is a byte-bounded LRU.
- **Transactions validate by evaluating.** After any change, rules, schemas
  and cycles are checked and every node is evaluated (mostly cache hits); on
  failure, the definitions are restored *into the existing handle objects*.
  Errors therefore appear at the change that caused them, and handles held by
  callers stay valid across rollback, undo and redo.
- **Registries for extension.** `PRIMITIVES` and `MODIFIER_TYPES` map names
  to small classes that declare a parameter schema (`ParamSpec`) and a pure
  function. A new primitive or modifier is one class: it is then validated,
  serialized, listed by `types()`, and usable from scripts and the CLI
  without further changes. Modifiers that read other geometry declare it via
  `references()`, which feeds the dependency graph and cycle detection.
- **Booleans through Manifold.** Solid booleans need robust geometric
  predicates that are impractical to reimplement correctly in pure Python.
  The boolean modifier is the only place that uses an optional third-party
  library: `manifold3d` (Apache-2.0), imported lazily, validated on input,
  and reported clearly when absent. The core and every other feature work
  without it.

## Rendering (v0.5.0)

The renderer is a layer above the core, beside editing and procedural
geometry. The scene gains only *data*: camera and light components on
objects, an appearance and a visibility flag, a world and render settings.
All rendering logic lives in `rendering/`. The design is explained in
[RENDERING.md](RENDERING.md); the key decisions:

- **Extraction separates scenes from backends.** `extract_render_scene` walks
  the scene once and produces an immutable `RenderScene`:
  - shared mesh arrays (procedural objects contribute their current cached
    evaluation);
  - per-instance world and normal matrices and linear colors;
  - world-space lights and the camera frame.

  The CPU backend consumes only this, and a GPU backend (Stage 6) can
  consume the same data, so `scene.render`, the CLI and the tests do not
  change. Rendering never modifies the scene.
- **The render world is retained between renders** (since 0.11.1,
  `rendering/retained.py`). Extraction keeps what it computed among the scene's
  derived caches -- world and normal matrices as arrays over the objects in
  depth-first order, the evaluated meshes, materials and render instances -- and
  updates it from the history's change journal: a transform recomputes its
  subtree's matrices, an edited object is classified again, a change of the
  hierarchy reorders the objects and keeps what each unchanged one had. The
  render scene is exactly the full walk's (tested along random edits); the full
  walk still runs inside open operations, on errors and with
  `AETHER3D_RETAINED_RENDER_WORLD=0`.
- **A visibility buffer, then deferred shading.**
  - Rasterization writes one packed `int64` per sample (quantized depth
    above, triangle index below), using `np.minimum.at`.
  - Visibility is therefore order-independent and deterministic, and every
    covered sample is shaded exactly once.
  - Batches bound memory to about 8 bytes per sample plus per-pixel buffers.
- **Exact integer rasterization.** Vertices are snapped to 1/256 of a sample.
  Covered spans come directly from integer edge functions with the top-left
  rule, so the result is watertight (each sample of a shared edge is covered
  once) and no candidate pixels are wasted.
- **Homogeneous everything.** Clipping happens in clip space. Shading
  barycentrics come from the original triangle in homogeneous 2D
  coordinates, so near-plane clipping never distorts attributes. The world
  position is reconstructed exactly on the view ray.
- **Conventions shared with the rest of Aether3D.**
  - Cameras and lights use the same "front faces −Y, +Z up" convention as
    assets, so one `look_at` aims everything.
  - Colors follow one policy: sRGB in, linear for lighting, sRGB out.
  - The clip and NDC conventions (depth [0, 1], pixel centers at +0.5)
    match modern GPU APIs, ready for a GPU backend.
- **Data-only scripts.** `run_scene_script` is the single entry point for
  declarative setup changes. CLI commands build one-operation scripts, and
  the planned MCP server can expose it unchanged. Rollback is exact and keeps
  callers' handles valid.
- **Independent verification.** The test suite's reference renderer is a
  brute-force ray caster that shares no projection, clipping, rasterization
  or shading code with the renderer. Analytic tests and a cross-check of
  silhouettes with Blender, an external tool, complement it.

## GPU rendering (v0.6.0)

The GPU backend (`rendering/gpu/`) is a second consumer of the same
`RenderScene`; [GPU.md](GPU.md) explains it in full. Key decisions:

- **One API on every platform.** wgpu (WebGPU) runs on Vulkan, Metal and
  Direct3D 12, rather than three native implementations. It is optional and
  imported lazily, so the core, the CPU renderer and every other feature
  work without it. `backend="cpu"` remains the default.
- **The scene format does not change.** Shadows, MSAA and adapters are
  render options, not scene content, so the `.aether` format stays at
  version 3.
- **Layering.**
  - `backends.py` (choices and validation) and `shadows.py` (shadow-map
    fitting) are backend-neutral and tested without a GPU.
  - Inside `gpu/`: device management, shaders and pipelines, resources and
    data layouts, and the passes are separate modules.
  - Only `gpu/` touches wgpu.
- **Everything the GPU draws comes from `RenderScene`.** Vertex pulling from
  cached per-mesh storage buffers, camera-relative instance matrices
  (float64 to float32), and reversed float32 depth keep the GPU's results
  within documented tolerances of the CPU renderer. The parity mode samples
  the CPU's exact positions.
- **Caches have explicit identities and lifetimes.**
  - **Meshes** are keyed by read-only array identity, released through weak
    references, and evicted beyond an LRU budget.
  - **Pipelines** are compiled once per device.
  - **Targets** are reused while the configuration repeats.
  - **Lost devices** are replaced on the next render.
- **Shading is split for materials.** `surface()`, `shade()` and
  `shadow_visibility()` were separate WGSL functions, so Stage 7 replaced
  `surface()` with materials without touching lighting, shadows or passes.

## Materials (v0.7.0)

Materials, textures and procedural shaders ([MATERIALS.md](MATERIALS.md))
are a separate package, `materials/`, used by the scene, both renderers,
persistence, glTF and the CLI. Key decisions:

- **Separate from objects, referenced by id.** A scene's `MaterialLibrary`
  owns materials and textures. Objects refer to a material, and optionally to
  one per face group; materials refer to textures through bindings. Removing
  something in use is refused unless forced. Library changes and scene
  scripts are transactions (snapshot and restore).
- **One source of truth for colors.** An object's `Appearance` is exactly a
  `"simple"` material. Objects without a material render bit-identically to
  0.6, and an object with a material refuses color changes.
- **Resolved once per render, shared by both backends.**
  `rendering/materials.py` turns the materials of visible objects into
  immutable `RenderMaterial` records: linear factors, bindings, decoded mip
  chains and graphs. Backends never touch the scene. Problems become
  diagnostics with a documented fallback, never a failed render.
- **CPU: per-sample, deferred.** The CPU renderer already shaded each covered
  sample once, from its visibility buffer. Each sample knows its triangle,
  barycentrics and their exact screen derivatives, so texture coordinates
  come with derivatives (mip levels, bump steps). `rendering/surface.py`
  evaluates materials over arrays of samples. Alpha masks are tested during
  rasterization, so masked samples never win the depth test.
- **GPU: data, not code.** Material records, bindings and compiled texture
  programs are one storage buffer; every image's mip levels share another
  (the texel pool). A fixed WGSL interpreter runs the programs, and images
  are sampled manually with the CPU's arithmetic. Scenes therefore never
  generate shaders, and pipelines do not depend on materials.
- **Reproducible procedural textures.** Noise uses integer hashes (PCG3D)
  computed identically in NumPy and WGSL, and parameters are rounded to
  float32 on both backends, so a graph renders the same everywhere.
- **Untrusted input is data.**
  - PNG is decoded by Aether3D's own validated decoder; JPEG goes through
    Pillow, only when installed.
  - Images are limited before allocation.
  - Texture paths are confined to the project's directory, symbolic links
    included.
  - Graphs are validated against a schema and size limits.
- **Face groups flow through geometry.** Meshes, the editing engine (journaled
  group changes, propagation by every operation) and procedural modifiers
  (including booleans, through Manifold's face ids) all carry face groups.
  glTF primitives map to groups.
- **The project format became version 4**, migrated from version 3 by adding
  empty libraries.

## Path tracing (v0.8.0)

The path tracer (`rendering/raytrace/`, `rendering/pathtrace/`) is a third
consumer of `RenderScene`; [RENDER_QUALITY.md](RENDER_QUALITY.md) explains
it in full. Key decisions:

- **A separate backend, `backend="path"`.** It never replaces or changes the
  rasterizers (`"cpu"`, `"gpu"` and `"auto"` render exactly as before), and
  it runs on the CPU only. Its report and PNG metadata say so.
- **Shared semantics, not shared shortcuts.** Materials are evaluated by the
  rasterizer's own `surface.evaluate_material`, and the lights' intensities
  are given physical units that reproduce the rasterizers' direct lighting.
  Everything the rasterizers approximate is computed instead: shadows,
  bounce light, reflections and emissive area lights.
- **A reusable ray tracing core.** `raytrace/` knows nothing about shading:
  it builds BVHs (binned SAH, vectorized, chunked) and answers closest-hit
  and any-hit queries for batches of rays. It uses a two-level structure (one
  cached BVH per mesh, one over instances) and watertight triangle tests.
  Its flat arrays are the layout a compiled or GPU traversal would use.
- **Vectorized wavefront integrator.** Paths advance in NumPy batches; random
  numbers are counter-based hashes of `(seed, pixel, sample, dimension)`. The
  image therefore does not depend on batching or on the number of processes:
  forked workers (Linux) trace tiles of pixels into shared-memory sums, in
  sample order, bitwise identically to one process.
- **The scene format is version 5.** Light sizes and the world environment
  are scene content, saved and migrated (version 4 files get the
  `"ambient"` environment and zero sizes, so they render unchanged).
  Sample counts, bounces, seeds and AO are render options, not saved.

## GPU path tracing (v0.8.1)

`backend="gpu_path"` (`rendering/gpu/pathtrace.py`, `gpu/wgsl/pathtrace.wgsl`)
runs the path tracer on a GPU adapter. Key decisions:

- **A port, not a second renderer.** The host side reuses the CPU path
  tracer's scene preparation (`PathScene`: the BVH with its BLAS cache, the
  emitter table, the environment) and the GPU rasterizer's material packing
  and texel pool. The compute kernel reproduces the CPU estimator step by
  step, with the same random numbers (SplitMix64 in 32-bit pairs), so the
  CPU path tracer is its reference, path for path.
- **One kernel, one material call.** Each invocation traces whole paths for
  one pixel as a small state machine whose loop evaluates materials at one
  place (`material.wgsl`'s `evaluate_surface`), as D3D12's shader compiler
  requires. Traversal is software, over the BVH's flat arrays in one storage
  buffer.
- **Deterministic sums.** Each pixel's samples are added by one invocation,
  in sample order, into float32 sums: no atomics on pixels, so images repeat
  bit for bit on a device, however the image is tiled.

## Operations history (v0.9.0)

`scene.history` (`history.py`) records every change of a scene, so that it can be
undone and redone exactly. See [HISTORY.md](HISTORY.md). Key decisions:

- **Operations and aspects, not commands with inverses.** Each mutating method
  runs as an *operation* (`history._op`), and nested operations join the outermost
  one: a step is a whole user-level action. The scene's state is divided into
  *aspects*, small independent parts: an object's transform, camera, parent,
  children or membership; a material; the procedural definitions; ...
  - **Touch before write.** Before an operation first writes an aspect, it
    *touches* it: the aspect's state is captured.
  - **Commit.** When the operation ends, the new states are captured, and the
    pairs that differ are the step.
  - **Undo and redo** write the before or after states back and rebuild the
    indexes.

  No method needs a hand-written inverse. Capture is per aspect, at the first
  write, so an operation cannot change a touched aspect without recording it. A
  missing touch is an internal error that the test suite's strict mode
  (`history.STRICT`) reports.
- **No copies of the scene.** States are small values: tuples of floats, ids and
  references. Meshes, image bytes and texture graphs are immutable, so states
  share them by reference.
  - The procedural definitions are copied once per revision and shared by
    consecutive steps.
  - `max_entries` and `max_bytes` bound the history. Memory counts the states,
    plus the meshes and images only the history still refers to.
- **Atomic operations.** The same captures roll back an operation that raises, so
  every public call is all or nothing. Nested transactions are savepoints: each
  keeps the states at its own first writes.
- **Identity.** States restore values into the existing Python objects: removed
  objects keep their instance, id and children, and come back the same.
  Value objects (cameras, lights, appearances, transforms, the world, its
  environment, render settings) know their owner, so in-place edits are
  recorded and restored in place. Removed objects keep a reference to the scene
  whose history records them (`SceneObject._home`).
- **Caches need no invalidation.** Renderer caches are keyed by immutable content
  (array identity, image digests, definition hashes), so undo reuses them. The
  version counters (`materials.version`, `procedural.revision`,
  `scene.revision`) still increase with undo and redo.
- **Linear history.** A new change discards the redo branch.
- **Subsystems keep their own undo, as steps.**
  - An edit session's undo stays local to the session; committing is one step.
  - The procedural system's own stack remains for compatibility, and each of
    its undos is a step of the scene's history.
  - Scripts and `materials.transaction()` are history transactions.
- **Persistence through the project format.** `_history_file.py` encodes each
  aspect state with the project's encodings, plus records of the entities the
  project no longer holds. The file's fingerprint ties it to its project, and
  loading replays it on a copy of the scene before using it. The CLI loads and
  saves the history around every command that changes a project with a history
  file, so command-line sessions can undo across processes.

## Scene intelligence and spatial queries (v0.10.0)

`aether3d.spatial` answers read-only questions about where things are; `Scene`
exposes every query as a method. See [SPATIAL.md](SPATIAL.md). Key decisions:

- **Exact, from the triangles, in world space.** World matrices are the
  renderers' (`world_matrices`), so queries and renders agree on placement, and
  camera rays are the path tracer's, bit for bit. Bounding volumes only select
  candidates; every reported fact is decided on triangles (or says it rests on
  boxes: `exact=False`).
- **Reuse the Stage 8 ray tracing core.** Ray queries run on the two-level
  `SceneAccel` and its watertight triangle test, with bottom levels from the
  renderers' shared BLAS cache. `all_hits` walks the same structure without
  shrinking the ray, with the same arithmetic, so the first of all hits is the
  closest hit; hits of one object at one distance (shared edges) are merged.
  Rendered-surface queries use the path tracer's `PathScene.trace`, so alpha
  cut-outs and single-sided backs are skipped exactly as the path tracer does.
  The accelerator's boxes are grown by one ulp so that slab tests treat them as
  closed sets.
- **Per-object world BVHs** (built on demand with `build_bvh`) serve nearest-point
  branch and bound, pair walks for distances and intersections, and region and
  frustum candidates.
- **Separating axes prove disjointness; constructive tests prove contact.**
  Triangle-triangle intersection does not rely on the completeness of an axis
  set: pairs not separated are decided by edge crossings and zero vertex-triangle
  or edge-edge distances.
- **Containment only where it is defined.** Inside/outside needs a closed,
  edge-manifold, consistently oriented mesh (inspection's welded topology); the
  winding number decides it (ray-crossing parity with retries for many points).
  Other meshes are reported as `"undefined"`, with the fractional winding number.
- **One index per scene revision.** The index is a cache kept on the scene
  (`Scene._derived()`, never scene state), so it dies with the scene; a
  module-level weak dictionary would keep every queried scene alive, because the
  index references the scene. `scene.revision` increases with every change, undo and redo included;
  inside an open operation (`history.in_progress`) a fresh index is built, so a
  query never sees stale geometry and caching needs no invalidation hooks.
  Queries never mutate the scene or record history.
- **One request format.** `run_query` maps JSON requests to the Python
  functions; the CLI (`aether3d query`) and the scene-script `query` operation
  use it, so Python, CLI and scripts give the same JSON.
- **Tags and metadata** are object state like any other: an `object.annotations`
  history aspect, project format 6, history files and glTF extras.

## AI tools: scene plans and the MCP server (v0.11.0)

`aether3d.plans` builds or changes a scene from one declarative document;
`aether3d.mcp` serves the engine to AI agents. See [SCENE_PLANS.md](SCENE_PLANS.md) and
[MCP.md](MCP.md). Key decisions:

- **Check everything, then change everything.** A plan is parsed and checked as a whole
  (`model.py`: types, symbols, references between symbols, expressions, limits), every
  problem collected with a JSON pointer. Binding (`executor.py`) expands patterns,
  allocates names and ids, resolves references to the scene and evaluates each
  procedural node once; only then does one history transaction apply it. Any failure,
  timeout or cancellation rolls the transaction back, so the scene and its history are
  as before; a dry run is the same path, rolled back on purpose.
- **Bulk creation in O(n).** `Scene._create_many` validates every new object first, then
  creates them inside one history operation, capturing each once. Automatic names come
  from per-base hints instead of scanning the taken names (which made N creations
  quadratic before 0.11). Layouts (positions, rotations, scales, variations) are computed
  per definition with NumPy and reused for every parent instance. Identical mesh
  definitions share one mesh, which the GPU rasterizer draws instanced.
- **The garbage collector waits.** A plan pauses Python's cyclic garbage collector while
  it runs (and restores the caller's setting): its allocations would otherwise trigger
  full collections that walk every object of the process, which in a long-running
  process holding a large scene cost more than the plan itself.
- **Small edits stay small.** Selections by reference are found through the scene's
  indexes and ordered by their paths in the hierarchy, and the plan's new meshes are
  told from borrowed ones without scanning the scene, so a one-object edit costs the same
  in any scene. (A per-call history step still captures the scene's root list: adding
  one object grows slightly with the number of root objects.)
- **Retries are deterministic.** With an idempotency key, the ids of the objects,
  materials and textures a plan makes are UUID 5s of the key and their place in the plan,
  so an applied plan is recognized from the scene alone, across processes; chunked plans
  resume by skipping chunks whose objects all exist.
- **Three descriptions of one format.** The checker, the hand-written JSON Schema and the
  guide for agents describe the same format; tests fuzz the checker against the schema
  (they must agree on every mutation) and the executor against crashes.
- **The server is a thin, careful layer.** It uses the official SDK's low-level server
  (both protocol generations), runs one tool at a time in a worker thread (the scene is
  never touched from two threads), and turns a cancellation into an event the plan
  checks at safe points, waiting for the rollback before it answers. Every path goes
  through `Workspace.path`; every reply is structured JSON (also as text); errors carry
  codes and hints and no paths of the machine. Nothing but protocol messages reaches
  stdout. The SDK is an optional extra, imported only by `aether3d.mcp`.
- **Measured.** `benchmarks/bench_plans.py` compares one plan with repeated operations
  (Python, CLI and MCP) at 100, 1,000 and 10,000 objects; the profiles behind it found
  the quadratic name allocation, two whole-scene passes per edit plan, a whole-scene scan
  per step of procedural scripts, a material record built per object in every render and
  full garbage collections inside plans, all fixed in 0.11. Scaling tests time with the
  collector paused (`tests/timing.py`), as `timeit` does.

## Performance modernization (v0.11.1)

The [October 2026 audit](../research/2026-10-architecture-audit/REPORT.md) found that
Aether3D's time went into recomputation and data layout, not into Python as such: a
10,000-object GPU frame spent 1.8 ms on the GPU and over 100 ms rebuilding on the CPU
what had not changed. The modernization followed its order -- the architecture first,
then a short list of native kernels, hardware ray tracing as a separate decision -- and
records every phase, with what was measured and what was not reached, in
[PERFORMANCE.md](PERFORMANCE.md).

- **Caches follow the history.** The history's change journal
  (`History._changes_since(revision)`) names the parts each revision changed -- commits,
  undo and redo alike. Caches derived from the whole scene (the retained render world,
  kept GPU frame data, world matrices and framing points) update only those parts and
  recompute everything when the journal cannot answer. They live with their scene
  (`Scene._derived()`), never in a module-level dictionary keyed by scenes: such an entry
  kept every scene that answered a spatial query alive (fixed in 0.11.1).
- **The same results, by construction and by test.** Every faster path computes what the
  path it replaces computes, bit for bit: `tests/parity.py`, the retained world's
  comparisons with the full extraction along random edits, `VERIFY_KEPT` for GPU frames,
  and NumPy references for the native kernels. The replaced path stays behind a switch
  (`_features.py`) until the new one has proven itself.
- **Authoring stays object-based; rendering works on arrays.** Scene objects, transforms
  and history aspects keep their identity and exact undo; the render side keeps columns
  (instance arrays, matrices, kept GPU records) and patches them per change. Columnar
  authoring was not adopted: it would trade the scene model's guarantees for costs the
  render side's columns already removed.
- **Native code is an optional, exact accelerator.** `aether3d-native` (Rust, PyO3)
  holds a few scalar kernels -- ray traversal, transform hierarchies, PNG filters -- each
  performing its NumPy twin's operations in order. Aether3D stays pure Python and uses
  them only when they are installed and speak its calling convention (`_native.py`).
- **Exactness bounds what may change.** Where results rest on platform libraries (mesh
  editing's geometric predicates use BLAS and libm), only exact optimizations were made;
  changing that arithmetic would be a deliberate behavior change for a release of its own.
- **Hardware ray tracing waits.** A prototype measured ray queries inside the GPU path
  tracer: the same triangles for at least 99.995 % of rays, but renders only 1.04-1.65
  times faster -- they remove 10-42 % of the kernel's GPU time, and shading stays as it
  is. wgpu-py cannot build acceleration structures, so integrating them would need a
  second GPU device owned by a Rust host. The software traversal stays.

## Visual feedback and automatic corrections (v0.12.0)

`aether3d.feedback` closes the loop between what a scene should be and what its renders
show: observe -> diagnose -> propose -> apply -> verify. See [FEEDBACK.md](FEEDBACK.md).
Key decisions:

- **Languages were chosen per component, before writing them.** Observations reuse the
  existing renderers (no new renderer, no new WGSL: the GPU rasterizer's finalize pass
  already produces the object index and depth, read back selectively); per-object pixel
  statistics are one vectorized NumPy pass over buffers the renderers already return as
  arrays (measured: 11-15 ms for all of a 640 x 360 observation's measurements, against
  24-46 ms for its GPU render and over a second on the CPU); geometry reuses the
  spatial queries' exact kernels (and their optional Rust traversal); documents, policy,
  planning, the loop, MCP and the CLI are Python, dominated by renders and queries. No
  native kernel was added: nothing measured hot enough to justify a second
  implementation and its parity tests, and no dependency was added.
- **Facts and judgments are kept apart.** Every finding states its basis -- geometry,
  render, scene (measured), heuristic (a default threshold), vision or agent (a declared
  judgment) -- and whether it is exact. The engine contains no model and makes no network
  request; image-aware evaluations come in as checked documents, grounded in the render's
  object index, labeled by their declared provenance, never counted as facts and never
  applied without approval. Only stated expectations are checked: whether an object floats
  matters only where a rule says it must rest.
- **Views are not scene content.** `render()` and the camera queries take virtual cameras
  (a `RenderCamera` without an object id), so presets and close-ups add nothing to the
  scene; a scene camera's virtual twin renders bit for bit like it.
- **Corrections are the engine's own operations.** A fixed allowlist of typed actions
  compiles to a scene plan and a scene script, shown in each proposal and run in one
  history transaction: one undo step, all or nothing, refused for a changed revision,
  preserving identity, hierarchy, procedural definitions and materials. Verification is a
  new diagnosis; a step that breaks a passing check, or does not help, is undone exactly.
- **Groups are objects.** An object with children (a prefab of blocks, an imported
  hierarchy) is checked, observed and moved as one: the union of its visible parts.
- **Caches follow revisions.** Observations are reused while the revision is unchanged
  (renders are deterministic), diagnoses too; after a correction the retained render world
  and the spatial index update only what changed. Work inside an open transaction (a
  preview) is never cached.

## Advanced modeling and sculpting (v0.13.0)

The editing engine gains production modeling operators, and `aether3d.sculpt` a
sculpting system on its own data structure. See [MODELING.md](MODELING.md) and the design
record [research/2026-10-stage13/DESIGN.md](../research/2026-10-stage13/DESIGN.md). Key
decisions:

- **One language per subsystem, measured first.** Topological operators are Python on
  the editing engine (few elements, rich bookkeeping, its journal and scripts for free);
  sculpting, dynamic topology and remeshing are data-oriented NumPy (a dab is a
  vectorized pass over its neighbourhood; topology passes decide every candidate at
  once with independent sets); predicates are float filters with an exact rational
  fallback. No native code, no GPU code and no dependency was added: the measured
  workloads run in interactive time.
- **A sculpt mesh separate from the editing mesh.** `SculptMesh` keeps welded geometry
  in growable arrays and render attributes (UVs, normals, normal groups) per corner, so a
  seam never splits a brush's effect and a commit rewrites only what changed. Triangles
  live in Morton-ordered chunks with boxes: a dab's cost follows its neighbourhood.
- **Journals, not snapshots.** Every write goes through journaled primitives: sparse
  position and mask records, topology records, or (for whole-mesh replacements) a
  reference to the previous arrays. Failures roll back, undo is exact, redo replays.
- **Exact where it decides.** In/out and crossing decisions use exact orientation
  predicates; the shared triangle-triangle test now decides with them too (a float
  determinant misfired on nearly parallel triangles). Validity is measured and tiered;
  `solid` requires the exact self-intersection test.
- **Features first.** Dynamic topology and remeshing split along but never collapse
  across boundaries, UV seams, hard edges, group borders or masked vertices, and
  interpolate UVs exactly on splits; a destructive voxel remesh says what it drops.
- **Same contract for agents.** Sculpt scripts are data, atomic and limited, with a
  schema fuzzed against the executor; the CLI, the MCP server and the visual feedback
  checks reuse the same validation (tiers, structured errors).

## UV texture painting and baking (v0.14.0)

`aether3d.uv` gives meshes UVs and measures them; `aether3d.texturing` paints and bakes
textures through them; `materials.painted` stores the painted textures. See
[TEXTURING.md](TEXTURING.md) and the design record
[research/2026-10-stage14/DESIGN.md](../research/2026-10-stage14/DESIGN.md). Key
decisions:

- **One language per subsystem, decided up front and measured.** Topology, charts,
  parameterization assembly, rasterization, MikkTSpace, painting and map evaluation are
  vectorized NumPy. The two kernels NumPy cannot vectorize -- the sparse LDL^T
  factorization (58 times faster in Rust) and the skyline placement of packing -- are in
  `aether3d-native` with bit-identical NumPy twins. Bake rays run on the GPU through the
  path tracer's BVH and traversal (shared `wgsl/traverse.wgsl`), with the multi-threaded
  CPU traversal as fallback. No second implementation of any feature was built to compare.
- **One tangent space everywhere.** A bit-exact MikkTSpace port gives every corner's
  tangent; the CPU and GPU rasterizers and path tracers read normal maps in it, the baker
  writes its exact inverse, and glTF export writes it as `TANGENT`.
- **Painted textures are image textures.** A `PaintedTexture` is an `ImageTexture` whose
  pixels come from 16-bit tiled layers: every renderer, the texture cache, mipmaps and
  glTF export see an image; strokes recomposite and re-upload only the tiles they
  changed (incremental mips equal a full rebuild bit for bit).
- **History by tiles.** A stroke is one operation that replaces tiles (before and after
  kept), so undo is exact and costs the touched tiles, not the image; history files store
  tiles compressed. The project format (version 7) stores layers losslessly.
- **Painting in 3D from texel positions.** Each tile's covered texels get their surface
  point from the exact UV rasterization (lazily, cached); a dab is a distance test in
  space, which makes seams invisible, and view strokes test visibility against a depth
  and object-index render.
- **Exact diagnostics.** UV overlaps and flips use the exact orientation predicates;
  stacked islands are distinguished from unintended overlaps; bakes count misses and UV
  overlaps.
- **Same contract for agents.** Texture scripts are data with per-operation argument
  allowlists, never touch files, are atomic and limited, and their schema is fuzzed
  against the executor; the CLI, MCP tools and the `uv_quality` feedback check reuse the
  same code.

## Precision modeling, CAD and architecture (v0.15.0)

`aether3d.cad` adds an exact, parametric document beside the mesh scene graph;
`aether3d.arch` builds buildings on it. See [CAD.md](CAD.md), [BUILDINGS.md](BUILDINGS.md)
and the design record
[research/2026-10-stage15/DESIGN.md](../research/2026-10-stage15/DESIGN.md). Key
decisions:

- **Exact model, derived meshes.** Parts are B-reps on exact surfaces and curves; the
  meshes the renderers see are tessellations within a stated chord, regenerated with the
  model and shown through linked objects that refuse direct mesh edits. Mass properties
  and measurements use the exact geometry. A tessellation is never called exact.
- **Robust booleans without false claims.** Manifold computes boolean topology on float64
  tessellations whose triangles remember their source faces; Aether3D reconstructs a
  B-rep on the operands' exact surfaces, recovering edges analytically where it can and
  tracing them (with a measured deviation) where it cannot.
- **Definitions as data, results by content.** Items are JSON definitions checked by
  their kinds; results are keyed by the definition and the keys of what they read, so
  only what a change affects regenerates and undo and redo reuse results. Two history
  aspects (`cad.item`, `cad.settings`) record the document; project format 8 stores
  definitions only.
- **An analysable solver.** Constraints compile to equations with analytic Jacobians;
  structural decomposition (matching and strongly connected components) splits the
  system into small blocks solved in order; numerical rank gives DOF, redundant and
  conflicting constraints exactly; branches are recorded so solutions never flip; edits
  re-solve only what follows them.
- **Persistent names.** Faces and edges are named by the feature and sketch entity that
  made them and survive regeneration; references never use indices.
- **One ray core.** Hidden lines in drawings are ray casts through the renderers'
  watertight ray tracer (`SceneAccel`), not a second visibility algorithm.
- **Languages decided up front.** Python and NumPy/LAPACK throughout; one Rust kernel
  (batched triangulation) with a bitwise NumPy twin; external CAD kernels (Open CASCADE,
  IfcOpenShell, ezdxf) are validators in the tests, never dependencies.
- **Untrusted exchange.** STEP, DXF and IFC readers are strict, limited and never
  evaluate anything; unsupported content is refused or reported.
- **Same contract for agents.** CAD scripts are data, atomic, limited, schema-described;
  the CLI, MCP tools, scene-plan operation and feedback checks reuse them.

## Animation, rigging and cinematics (v0.17.0)

`aether3d.animation` adds animation without a GUI; see [ANIMATION.md](ANIMATION.md) and
the design record [research/2026-10-stage17/DESIGN.md](../research/2026-10-stage17/DESIGN.md).
Stage 17 was built before Stage 16, which is postponed. Key decisions:

- **Authored versus evaluated.** The scene keeps rest values; the animation document
  holds immutable item definitions (clips, bindings, rigs, skins, morph targets,
  constraints, paths, the shot sequence), recorded by two history aspects
  (`animation.item`, `animation.settings`) and saved in project format 9. Evaluation at a
  time is a pure function returning an immutable pose; rendering at a time applies it
  through the render overlay without changing the scene. `apply_pose` is the only write.
- **A compiled program per revision.** Bindings and strips become contributions, channels
  queries into one batched curve evaluation and one batched rotation evaluation per frame
  (structure of arrays, the same bits as each curve alone); strips are timed together.
  Nothing carries over between frames, so seeking is free and results never drift.
- **Typed paths, never attributes.** Channels name properties through a strict grammar and
  a registry; there is no attribute traversal, so a script cannot reach Python objects.
- **One documented order.** Channels, world matrices, then rigs and object constraints in
  dependency order (cycles refused when made), then morph targets and skins, then the
  shot camera.
- **Exact or labelled.** Curve derivatives, continuity and extrema come from closed forms;
  trajectories, contacts and fitted errors are sampled and say so. Generated motion is
  measured, not assumed.
- **Caches that respect transactions.** Caches of authored state are keyed by the scene
  revision and bypassed while an operation is open, like the spatial index.
- **Languages decided up front.** NumPy for curves, quaternions, blending and strip
  timing; the existing native hierarchy kernels for FK; one new Rust kernel (linear blend
  skinning, ABI 4) bitwise equal to its NumPy twin; no new shaders, no GPU skinning, no
  external renderer (FFmpeg, optional, only encodes Aether3D's frames).
- **Same contract for agents.** Animation scripts are data, atomic, limited and
  schema-described; the CLI, the MCP tools, the scene-plan operation and the feedback
  checks reuse them.

## Procedural worlds and massive environments (v0.18.0)

`aether3d.worlds` adds generated landscapes; see [WORLDS.md](WORLDS.md) and the design
record [research/2026-10-stage16/DESIGN.md](../research/2026-10-stage16/DESIGN.md).
Stage 16 was built after Stage 17 and ships as 0.18.0. Key decisions:

- **A world is a definition, not geometry.** Each world is an item with an immutable,
  validated JSON definition and an immutable store of block edits, recorded by one history
  aspect (`world.item`) and saved in project format 10. Everything generated -- the
  regional stage, tiles, plants, chunks, volumes -- is a cache keyed by content, so undo,
  load and reload give the same bits, and nothing generated is ever saved.
- **Pure functions of coordinates.** The definition compiles into field functions
  (height, climate, biome weights, water) of world coordinates; tiles, chunks and plant
  tiles are pure functions of their addresses, so generation order, batching and threads
  never change results.
- **Determinism by construction.** Integer hashing (SplitMix64) for every random
  decision; only correctly rounded floating-point operations (`+ - * /`, `sqrt`, `floor`)
  with explicit, ordered sums instead of BLAS (`noise.blend`) and our own exact cosine
  and sine; the native hydrology kernels are bitwise twins of the NumPy code. Golden
  hashes of float64 bits guard it.
- **Authoritative versus drawn.** Queries, structures and paths use the exact fields
  (heights at any point, the cave field, every block); renders draw levels of detail whose
  error is measured and bounded in pixels. Each is documented with what it approximates.
- **Contributions, not objects.** A world appends meshes and instances to the render
  scene for each frame's camera (synthetic ids, shared immutable meshes the GPU caches by
  identity); it adds nothing to the scene, its history or its files. Frame selection
  depends only on the frame's camera, so animated renders are order independent.
- **Bounded memory.** Byte-budgeted LRU caches per kind; a frame's working set is pinned
  for the frame only; memory is measured (cached bytes per kind, the last frame, the
  process's resident memory, GPU mesh caches) rather than estimated.
- **Languages decided up front.** NumPy (float64) for fields, tiles, plants, blocks,
  greedy meshing and volumes, vectorized and batched across tiles; Rust kernels only for
  the sequential hydrology sweeps (priority flood, receivers, accumulation, stream power;
  ABI 5); WGSL for the atmosphere on the GPU; no GPU generation (GPU float arithmetic is
  not reproducible across adapters).
- **Same contract for agents.** World scripts are data, atomic, limited and
  schema-described; the CLI, the MCP tools (`run_script`, `inspect_world`) and the
  scene-plan operation `world` reuse them.
