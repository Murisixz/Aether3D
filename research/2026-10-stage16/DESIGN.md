# Stage 16 design record: procedural worlds and massive environments

Written on 18 October 2026 against `main` at 903f010 (v0.17.0: Stages 1-15 and Stage 17,
which was built first while this stage was postponed). The roadmap keeps the name
"Stage 16"; the software version moves on monotonically to **0.18.0**. This file records
what was inspected, the research behind each decision, the language chosen for every
subsystem, and the test, benchmark and showcase plans. The user guide
([docs/WORLDS.md](../../docs/WORLDS.md)) describes what was finally built; where they
differ, this file is the plan as decided and the guide is the truth about the code.
Section 9, written after the implementation, records the outcome.

## 1. Starting point (inspected)

| Area | Where | What exists | Consequence for Stage 16 |
| --- | --- | --- | --- |
| Scene components | `scene.py` (`scene.cad`, `scene.animation`), `history.py` aspects | Subsystems are documents of immutable item definitions, recorded by per-item history aspects, saved in the project | `scene.worlds` follows the same pattern: world items with immutable definitions and edit stores, one aspect per world (`scene.world` already names the render settings) |
| Render input | `rendering/scene_data.py`, `retained.py`, `renderer.render` | Every backend consumes one immutable `RenderScene` (meshes, instances with float64 world matrices, resolved materials); the retained world updates the object part from the change journal | A world is not made of scene objects: it *contributes* meshes and instances to the render scene for the camera of each render, after extraction. Chunk meshes are immutable `Mesh` arrays, so the GPU's identity-keyed mesh cache uploads each once |
| GPU raster | `rendering/gpu/` | Vertex pulling, camera-relative instance matrices (float64 → float32), instanced draws per shared mesh, one shadow map per directional light (no cascades), no frustum culling, no fog | The world selects tiles and instances itself (distance, frustum, shadow reach). Kilometre views need cascaded shadow maps; outdoor images need fog and a sky on the rasterizers |
| CPU raster | `rendering/cpu.py`, `surface.py` | Visibility buffer, deferred shading per sample, no shadows | Fog, sky background, vertex colours and hemisphere ambient are added to the shading pass with the GPU's semantics |
| Path tracers | `rendering/pathtrace/`, `gpu/pathtrace.py` | Environment `ambient|uniform|sky`; no volumes | The sky gains a sun glow; fog is applied along camera rays (aerial perspective) like the rasterizers, documented as such |
| Meshes | `mesh.py` | positions, normals, UVs, face groups; no vertex colours | Terrain and blocks need per-vertex colour (biome tint, ambient occlusion, water depth): `Mesh.colors` (glTF `COLOR_0` semantics: multiplies the base colour) |
| Materials | `materials/` | Library materials with image/procedural textures, `nearest`/`linear` filters, mipmaps, wrap modes, world-space triplanar projection, alpha masks; no blending | Pixel-art blocks: per-face-texture materials with `nearest` + mipmaps and repeat wrap on greedy quads; terrain: triplanar detail textures times vertex colour; water is opaque (no blended transparency) and coloured by depth |
| Procedural noise | `materials/noise.py` | PCG3D integer hashes, value/gradient noise, fbm, voronoi for textures | World generation needs float64 coordinates, 2D/3D noise, ridged/warped fractals and platform-exact arithmetic: a separate `worlds/noise.py` (same family of hashes) |
| Marching cubes | `sculpt/voxel.py` | Vectorized MC over sparse keys, generated case table with a consistent face rule (closed, manifold) | Reused for cave chunks (no second iso-surface extractor) |
| Spatial queries | `spatial/` | Exact ray/nearest queries over scene objects | World queries are answered by the world itself (authoritative height function, voxels), not by meshing the world into the index |
| Animation | `animation/` | Paths, follow-path, cameras, generators, frame rendering (pure per frame) | Ground-following paths and camera clearance come from world queries; frame rendering stays pure because tile selection is a function of each frame's camera |
| Scripts, plans, MCP, CLI | `*/script.py`, `plans/`, `mcp/`, `cli.py` | Atomic data-only scripts per subsystem; `run_script` dispatches formats; plan operations embed scripts; 26 MCP tools whose list is 39,936 of the 40,000 bytes its test allows | `aether3d.world-script`, plan op `world`, `aether3d world ...`, one read-only MCP tool `inspect_world` paid for by tightening existing descriptions (the budget is not raised) |
| Native | `native/` (Rust, PyO3 abi3, ABI 4) | Bitwise NumPy twins for every kernel | New kernels (hydrology, block meshing, noise if profiled hot) follow the twin policy; ABI 5 |
| Format | `serialization.py` | Project format 9 | Format 10: `scene.worlds`, `scene.world.atmosphere`, mesh `colors` |

## 2. Research (sources used)

**Noise and terrain synthesis.** Perlin, "Improving noise", SIGGRAPH 2002 (quintic fade,
gradient lattice); Musgrave, Kolb and Mace, "The synthesis and rendering of eroded fractal
terrains", SIGGRAPH 1989, and Ebert et al., *Texturing and Modeling* (3rd ed.), ch. 16
(fBm, ridged multifractal, heterogeneous terrain); Quilez, "Domain warping" (2002/2012
articles) for warped fractals; Galin et al., "A review of digital terrain modeling",
CGF 2019 (procedural vs simulation vs example-based trade-offs). Minecraft 1.18's noise
router (community documentation and the Yarn API docs of `NoiseRouter`): continentalness,
erosion, peaks-and-valleys and weirdness fields mapped through splines to height and
density, noise caves (cheese, spaghetti, noodle) and aquifers -- used as the model for the
*macro-field + spline* structure and the block mode's vertical scale, not copied.

**Erosion and hydrology.** Braun and Willett, "A very efficient O(n), implicit and parallel
method to solve the stream power equation", Geomorphology 180-181, 2013 (receivers, stack,
implicit `n = 1` solve); Yuan, Braun, Guerit et al., "A new efficient method to solve the
stream power law model taking into account sediment deposition", JGR Earth Surface 2019
(deposition coefficient; transport-limited behaviour); Cordonnier et al., "Large scale
terrain generation from tectonic uplift and fluvial erosion", CGF 2016 (uplift + stream
power for graphics); Tzathas, Gailleton, Steer and Cordonnier, "Physically-based analytical
erosion for fast terrain generation", CGF 43(2), 2024 (closed-form stream-power solutions;
considered, see section 3); Barnes, Lehman and Mulla, "Priority-Flood: an optimal
depression-filling and watershed-labeling algorithm for DEMs", Computers & Geosciences 62,
2014 (O(n log n) fill; with an epsilon the filled surface drains everywhere: lakes are the
filled depressions); O'Callaghan and Mark 1984 (D8 flow directions); Leopold and Maddock
1953 (hydraulic geometry: width ~ discharge^0.5). Thermal erosion: Musgrave et al. 1989
(talus angle relaxation).

**Climate and biomes.** Whittaker biome diagram (temperature x precipitation envelopes);
environmental lapse rate ~6.5 K/km; orographic precipitation and rain shadows (upwind
relief); Minecraft's per-column biome colour blending as the model for smooth tints.

**Voxels and meshing.** Lysenko, "Meshing in a Minecraft game" (2012: greedy meshing, face
culling); per-vertex ambient occlusion from three neighbours (Lysenko 2013, "Ambient
occlusion for Minecraft-like worlds"); Lorensen and Cline 1987 and the existing generated
MC table (Montani et al. 1994 face rule); Lengyel, "Transvoxel" (2010) and naive surface
nets (Gibson 1998) considered and not needed (section 4.6).

**Level of detail.** Ulrich, "Rendering massive terrains using chunked level of detail
control" (SIGGRAPH 2002 course: quadtree tiles, screen-space error, skirts); Strugar,
"Continuous distance-dependent level of detail for rendering heightmaps" (CDLOD, 2010:
morphing); Losasso and Hoppe, "Geometry clipmaps" (SIGGRAPH 2004); restricted quadtrees
(2:1 balance) and edge stitching from the adaptive-mesh literature.

**Placement.** Bridson, "Fast Poisson disk sampling in arbitrary dimensions" (2007);
jittered-grid sampling with rejection as its order-independent cousin; Lane and
Prusinkiewicz, "Generating spatial distributions for multilevel models of plant
communities" (2002: species priority and exclusion radii).

**Rendering.** Quilez, "Better fog" (2010) and the exponential height-fog integral used by
production engines (analytic optical depth `a e^{-b(z0-zf)} (1 - e^{-b dz t}) / (b dz)`);
Dimitrov, "Cascaded shadow maps" (NVIDIA 2007) and Zhang et al., "Parallel-split shadow
maps" (2006: split scheme `lambda * log + (1 - lambda) * uniform`); hemisphere ambient
lighting as the rasterizers' cheap stand-in for sky light.

## 3. Language and execution decisions (made before writing each subsystem)

| Subsystem | Language | Why |
| --- | --- | --- |
| Definitions, validation, presets, scripts, API, history, files, CLI, MCP | Python | Orchestration and rich validation; one coherent API |
| Noise, terrain layers, climate, biome weights, scatter, structure layout | NumPy (float64), only `+ - * /`, `sqrt`, `floor` and integer hashing | Vectorized over points; correctly rounded IEEE operations give the same bits on every platform and with a native twin. A Rust noise kernel is added only if profiling shows sampling dominates generation |
| Hydrology: priority-flood, receivers, stack, accumulation, implicit stream power, sediment | Rust kernels (ABI 5) with Python/NumPy twins | Inherently sequential graph sweeps (heap order, receiver chains): Python loops cost seconds per million cells per sweep. Areas and sediment are accumulated in integers so sums are order-independent and the twins can vectorize |
| Thermal erosion, diffusion, distance fields | NumPy stencils | Jacobi updates vectorize; no native needed |
| Heightfield tile meshes, stitching, skirts, colours | NumPy | Regular grids vectorize |
| Cave chunks | NumPy + `sculpt.voxel.marching_cubes` | Reuse; density evaluated only near cave features |
| Block generation (columns, surface rules, caves, features) | NumPy | Columns and 3D noise vectorize over a chunk |
| Greedy block meshing | Rust kernel with a NumPy/Python twin (decided after profiling the twin) | Merging is a sequential scan per slice |
| GPU world generation | Not used | GPU float arithmetic is not reproducible across adapters (no control of fused operations); generation is cached and amortized, so the CPU stays the reference |
| Rendering additions (vertex colours, fog, sky glow, hemisphere ambient, cascades) | NumPy (CPU raster, CPU path tracer) and WGSL (GPU raster, GPU path tracer) | Same semantics on every backend that supports them; cascades only where shadow maps exist |

## 4. Architecture

### 4.1 Package and API

`aether3d.worlds` (new package); `scene.worlds` is a `WorldSystem` (created on first use)
holding `World` items. A world item has an id, a unique name and two immutable parts:

* the **definition** -- a validated JSON document: `seed`, `generator` version, `mode`
  (`"terrain"` or `"blocks"`), `extent` (bounded rectangle or `null` for unbounded),
  `terrain` (macro fields, layers, warp, height curve), `climate`, `biomes`, `hydrology`,
  `caves`, `vegetation`, `structures` (rules), `blocks` (palette and vertical range),
  `materials` (layer -> scene material), `streaming` (tile sizes, LOD error, budgets),
  `features` (authored terrain features and biome overrides, applied in order);
* the **edits** -- sparse block edits per chunk (compressed arrays) and frozen terrain
  tiles: authored data too large for the definition.

`World` methods answer queries (`sample`, `height`, `biome_at`, `profile`, `find_sites`,
`route`, `ground_path`, `block_at`, `stats`) and change the world through history
operations (`configure`, `add_feature`, `set_blocks`, `fill_blocks`, `place_structure`,
`build_settlement`, ...). Generation is lazy: nothing is generated by creating or loading a
world.

### 4.2 Determinism

* Every random decision is an integer hash of `(seed, stream, integer coordinates)`
  (SplitMix64-style finalizer on uint64, wrapping arithmetic): no global RNG state, so
  results never depend on visit order, threads or which regions came first.
* Coordinates are float64; lattice indices are `floor` of exact products. The generator
  uses only correctly rounded operations (`+ - * /`, `sqrt`, `floor`, comparisons) and
  piecewise polynomials (quintic fades, smoothsteps, splines) -- no `exp`, `sin` or `pow`
  from libm -- so results are bitwise identical across platforms; CI compares golden
  hashes on Linux, macOS and Windows.
* The definition carries `generator` (the algorithm version); cache keys hash the
  generator version, the definition sections a result depends on, the features that
  intersect its dependency box, and its edits.

### 4.3 Stages and caches

Generation is a set of pure stages with explicit inputs:

1. **Fields** (per point): macro fields, terrain layers, climate, biome weights.
2. **Regional stage** (per hydrology domain; bounded and explicit): the base terrain on a
   coarse grid, erosion (stream power with deposition, thermal), priority-flood lakes, D8
   flow, drainage, the river network (polylines with widths and monotone water levels),
   distance-to-water and detail-attenuation fields. Its cost (cells, seconds, bytes) is
   reported and it is cached in memory (and optionally on disk, content-addressed).
   Unbounded worlds compute it inside an explicit `hydrology.extent`; outside, terrain
   is the procedural surface (or rivers come from the `contour` mode, which needs no
   regional stage).
3. **Tiles** (per chunk and LOD level): heights, normals, colours, meshes; block sections;
   cave chunks; vegetation instances; water surfaces.
4. **Settlements** (per region): site selection, layout, roads, bridges; deterministic
   from the definition, frozen into authored entries on request.

Caches are LRU with byte budgets (`streaming.memory_mb`), keyed by content hashes; they
never hold the only copy of anything authored, so eviction cannot lose edits.

### 4.4 Height function (authoritative)

`h(x, y) = remap(macro + layers) + biome modifiers + erosion delta (bicubic, regional)
+ attenuated detail + river channels + authored features`. Normals, slopes and curvature
come from the same function by central differences. Rendered meshes sample it exactly at
their vertices; the reported LOD error bounds the difference elsewhere.

### 4.5 Tiles, LOD and seams

A quadtree of square tiles (power-of-two sizes and resolutions, so every vertex
coordinate is an exact binary fraction). LOD selection is a pure function of the camera:
refine while the tile's measured geometric error projects above `lod_error` pixels;
restrict the tree to 2:1 neighbours; a finer tile next to a coarser one stitches that
border (skips odd border vertices: the shared edge is the same segments on both sides);
skirts hide the remaining float32 rounding on the GPU. No hysteresis (it would make frames
depend on earlier frames); popping is bounded by `lod_error` pixels.

### 4.6 Caves and overhangs

Density `d(x, y, z) = z - h(x, y)` with cave and overhang terms (`min`/`max` CSG of worm
tunnels, caverns, authored spheres/capsules/tunnels and overhang noise on steep slopes).
Chunks whose box meets a 3D feature are *volumetric* and meshed with marching cubes on
the same lattice as heightfield tiles; because the density is linear in `z` on vertical
edges and linearly interpolated on horizontal ones, the volumetric surface meets a
neighbouring heightfield tile exactly on its border polyline (no crack). Volumetric chunks
render within `volume_distance`; beyond it their heightfield surface is used (documented).

### 4.7 Block worlds

Same fields, biomes, hydrology and features, sampled per block column; 16 x 16 columns per
chunk, sections of 16 blocks; caves carved from 3D noise; structures and trees stamped
from deterministic plans of the 3 x 3 chunk neighbourhood; sparse persistent edits on top.
Meshing: hidden-face removal across chunk borders, greedy merging per material, per-vertex
ambient occlusion and biome tint in vertex colours, pixel-art textures sampled with
`nearest` filtering and mipmaps. Distant chunks use coarser block LODs (2^k blocks per
cell) built from column heights, keeping the blocky silhouette.

### 4.8 Rendering

`render()` asks each world for its contribution for the frame's camera: selected terrain
tiles, cave chunks, water surfaces, vegetation and structure instances (frustum- and
distance-culled with shadow reach), materials resolved from the scene library. World
items appear in `result.objects` under synthetic ids (`world:<name>/...`). New render
features, opt-in and saved with the scene: `scene.world.atmosphere` (sky glow, exponential
height fog with sun scattering, hemisphere ambient, sky background on rasterizers),
`ShadowSettings.cascades`, mesh vertex colours.

### 4.9 Authoring surfaces

`aether3d.world-script` (atomic, limited, data only; schema and guide as MCP resources),
plan operation `world`, `aether3d world ...` CLI, MCP `inspect_world` (summary, sample,
profile, find, route, chunks, map image). Exports of finite regions to glTF bake tiles
(vertex colours as `COLOR_0`) and report what glTF cannot keep (streaming, LOD,
procedural definitions).

### 4.10 Animation integration

`ground_path` samples the authoritative ground along a route (terrain or block tops) for
`follow_path`; `place_on_ground` grounds objects; camera paths get clearance checks;
`prepare(frames=...)` generates every tile any frame needs before rendering. Each frame's
tile set is a function of that frame's camera only.

## 5. Robustness, safety and transactions

Definitions and scripts are untrusted data: schema and range checks with JSON pointers,
size limits (layers, features, edits per call, extent, hydrology cells, tiles per request,
generated bytes, seconds), cancellation at safe points, atomic history transactions
(validation by evaluation before commit), no attribute traversal, no file access from
scripts (exports and caches go through the CLI's/MCP workspace path checks), no shader
input from data.

## 6. Test plan

Determinism (visit orders, threads, golden hashes across platforms); seams (shared edges
bitwise, stitched LOD borders closed, volumetric/heightfield borders collinear); hydrology
(monotone water levels, every river ends at sea/lake/border, lakes flat and enclosed,
native = twin bitwise); biomes (weights sum to 1, Lipschitz continuity, no checkerboards);
blocks (greedy = naive face coverage, hidden faces gone, chunk borders consistent, edits
persist through eviction/save/load/undo); vegetation (stable ids, exclusions, grounding);
structures (pads flat, no overlaps, bridges along roads); streaming (bounded memory,
reload equality); rendering (vertex colours, fog and sky parity across backends, cascades);
animation (character stays on ground, frames order-independent); security (malformed and
oversized inputs, schema fuzzing).

## 7. Benchmark plan

`benchmarks/bench_worlds.py`: samples per second (fields, height), tile generation cold and
warm per mode and level, meshing (heightfield, volumetric, greedy), hydrology per grid size
(native and twin), vegetation placement, streaming traversal (RAM, VRAM, tiles generated,
cached, rendered, draw calls), render frame times (GPU/CPU), save/reload, query latency;
`bench_engine.py` baseline before (main 903f010) and after, to show no regression.

## 8. Showcases

A. Cinematic alpine valley (terrain, erosion, river and lake, forests, sunset atmosphere).
B. Minecraft-inspired block world (original textures, blocky trees, mountain/canyon, a
   castle or village). C. Large-world technical demo (streaming traversal with measured
   memory and tiles). D. Biome world (climate gradients and transitions). E. Caves and
   overhangs (volumetric terrain). F. Stage 16 + 17: the blocky character runs through the
   block world with a tracking camera. Stills, contact sheets, 1080p H.264 (CRF 14, slow)
   and lossless masters; scripts, projects and reports committed, frames and masters not.

## 9. Outcome

Written after the implementation (branch `stage-16-procedural-worlds`, version 0.18.0),
against the plan above. [docs/WORLDS.md](../../docs/WORLDS.md) describes the code as it
is; this section records where it differs from the plan, what the showcases found and what
was measured.

### 9.1 What was built, against section 4

* **4.1 Package and API** -- as planned: `scene.worlds`, world items with immutable
  definitions and edits, one history aspect (`world.item`), project format 10. Edits are
  block edits only: frozen terrain tiles were not needed (features cover local terrain
  authoring) and were not built. The settlement call is `build_settlements(count, ...)`;
  statistics are `world.runtime.stats()` and the `stats` query.
* **4.2 Determinism** -- as planned, golden hashes checked on Linux, macOS and Windows by
  CI and without the native kernels. Two holes were found and closed: matrix products
  (BLAS) gave results that depended on the batch size (a plant's scale differed between
  batched and single-tile placement), so generation uses explicit ordered sums
  (`noise.blend`); and structure layouts and wetness colours still called the platform's
  `atan2`, `cos`, `sin`, `tan`, `log10` and `hypot`, now replaced by exact `atan2_deg`,
  `log10`, `cos_sin_deg` and square roots (found while checking the documentation's
  claim against the code). The template meshes of plant species and buildings still use
  the platform's trigonometry for their shapes -- documented as outside the guarantee.
* **4.3 Stages and caches** -- as planned, without the optional disk cache: the presets'
  regional stages take seconds and stay cached in memory.
* **4.4 Height function** -- as planned; the erosion delta is smoothed by a binomial
  filter before its bicubic interpolation (the raw delta showed cell-sized quilting).
* **4.5 Tiles** -- as planned: no hysteresis and no morphing, popping bounded by
  `lod_error` pixels.
* **4.6 Caves and overhangs** -- 32 m cubes of 1 m voxels near the camera, surface and
  closed cubes, holes cut in the tiles; see 9.4 for two seam problems the showcases found.
* **4.7 Block worlds** -- as planned; greedy meshing stayed in NumPy (9.2).
* **4.8 Rendering** -- as planned; synthetic ids are `world:<world id>:<part>`.
* **4.9 Authoring surfaces** -- world scripts, the plan operation `world`, `aether3d
  world`, MCP `inspect_world` (`summary`, `info`, `sample`, `profile`, `sites`, `route`,
  `block`, `stats`; no map image -- `render_preview` renders worlds) and region export
  (`COLOR_0`). `scene.export` reports worlds it leaves out (`WORLDS_NOT_EXPORTED`).
* **4.10 Animation integration** -- `ground_path(clearance=)`, `place_on_ground`,
  `worlds.prepare(times, camera=)`; frames render identically in any order (tested).

### 9.2 Languages, as decided and as measured

| Subsystem | Decided (section 3) | Built and measured |
| --- | --- | --- |
| Hydrology sweeps | Rust kernels with NumPy twins | `native/src/hydro.rs` (ABI 5): priority flood, accumulation in integers, stream power, sediment; bitwise equal to the twins. `alpine_valley`'s regional stage: 53.7 s NumPy, 6.5 s native (erosion 51.6 -> 5.4 s) |
| Thermal relaxation | NumPy stencils | the NumPy stencil stays the reference; a threaded kernel runs the same Jacobi sweeps natively (threads cannot change the result) |
| D8 receivers, distance and attenuation fields | NumPy | NumPy, vectorized over the grid |
| Noise, fields, climate, biomes, placement, structures | NumPy, Rust only if profiling demanded | NumPy: profiles showed many calls on mid-sized arrays; batching across tiles and chunks gave the large wins (plant placement 48 s -> 1.0-1.6 s for a 2.5 km view) |
| Greedy block meshing | Rust kernel after profiling the twin | NumPy with vectorized run merging: 8.6 ms to mesh a chunk (5.9 ms to generate one); no kernel was written |
| Cave volumes | NumPy + the existing marching cubes | as decided: 41 ms a 32 m cube |
| GPU generation | not used | not used |
| Rendering additions | NumPy and WGSL, same semantics | as decided; CPU/WGSL parity tests for the atmosphere and vertex colours |
| Threads | -- | `streaming.threads` (pure functions of addresses, identical results): 2 threads about 17 % faster than 1 on block chunks, more slower (the GIL), so the default is 1 |

### 9.3 Deviations and limitations

* Not built: frozen terrain tiles as edits, the disk cache of the regional stage, a map
  image in `inspect_world`, a native greedy mesher and native noise (9.2).
* The tree line is a global temperature; forests end where their biomes end. Its
  default became -7 C after the showcases (9.4); alpine presets set their own.
* Unbounded worlds have no regional stage (`contour` rivers); caves are meshed only within
  `volume_distance`; water is flat; level changes are not morphed; threads gain little
  under the GIL. All are documented in WORLDS.md's limitations.

### 9.4 Found by the showcases and fixed

* Vegetation placement for a 2.5 km view took 48 s; batching per plant class across
  tiles and an 8 m field lattice brought it to about 1 s, bit for bit the same plants as
  tile-by-tile placement.
* Erosion quilting (above), soil drawn as bands along contours (now patches just below
  the rock), smooth cone-like mountains (sharper ridged alpine presets).
* Snowy taiga had no trees: its whole temperature window lay above the default tree line.
* Desert could never dominate: its window lay inside badlands'; the windows were split.
* Swamps were drawn as beaches (any low flat ground was sand): beaches now need water
  nearby.
* Coasts ran straight along a bounded world's border: the falloff is bent by noise.
* A crack opened between neighbouring overhang cubes: each faded its warp by its own
  closed sides, so neighbours with different corner neighbours computed different fields
  on their shared face. The fade is now a function of the distance to every undrawn
  footprint around (tested by comparing shared-face vertices). Ground crossing into a cube
  above an entrance was not drawn; every cube the ground crosses is drawn now.
* Roads crossed narrow rivers without bridges and could leave the world: water is sampled
  every 2 m along routes and the search grid is clamped to the extent.
* Showcase F: parenting Stage 17's armature alone left its attached parts behind -- by
  design, attachments follow joints relative to where they were authored -- so the
  showcase parents the parts too; Stage 17 is unchanged.

### 9.5 Measurements

`benchmarks/bench_worlds.py` at 70e4738 (`benchmarks/results/stage16-worlds.json`; tables in
[docs/PERFORMANCE.md](../../docs/PERFORMANCE.md#procedural-worlds-018)): a terrain view's
contribution 11.2 s first and 0.19 s cached, a median 0.61 s per frame along a 1.5 km
flight; a block-world view 6.4 s first, 34 ms per frame of a moving camera; 1920x1080 GPU
frames along a path a median 1.1 s (`alpine_valley`, 2.35M triangles) and 0.54 s
(`blocky_adventure`); 64,000 heights per second. The engine's other benchmarks, run
interleaved with `main` (903f010) on the same machine state, agree within -5 % to +5 %
(`stage16-engine-interleaved.json`); the plain comparison with `stage16-baseline.json`
reads 5-10 % slower on core operations because the laptop was warmer, and `main`
measured then reads the same. The showcases' reports hold every render's timing (A-F in
[docs/WORLDS.md](../../docs/WORLDS.md#showcases)).

### 9.6 Verification

Run locally on the final code (the matrix of the CI workflow, plus the native kernels):

| Configuration | Result |
| --- | --- |
| whole suite, Python 3.14, native kernels | 3,346 passed, 5 skipped |
| whole suite without the native kernels (`AETHER3D_NATIVE=0`) | 3,342 passed, 9 skipped (four heavy showcase runs need the native kernels) |
| GPU test files on the RTX 4060 / on lavapipe (`AETHER3D_REQUIRE_GPU=1`) | 215 passed / 214 passed |
| Khronos glTF Validator required (exports, world regions, vertex colours) | passed: 0 errors, 0 warnings |
| portable CI (GPU tests off, adapter enumeration refused and logged) | 3,209 passed; only the three tests that probe the GPU on purpose reached it |
| Python 3.11 without manifold3d, wgpu, MCP or jsonschema / with manifold3d 3.4.1 | 2,945 / 3,090 passed |
| Python 3.10 without optional packages | 2,945 passed |

New tests cover definitions and their errors, determinism (golden hashes with and without
the native kernels, batches, orders, threads), seams of tiles, stitching and volume cubes,
hydrology (twins, drainage), blocks (greedy meshes equal the exposed faces, edits, borders),
vegetation (exclusions, ids, nesting), structures and roads, budgets and eviction,
rendering on every backend, atmosphere parity, vertex colours, files and migrations, world
scripts (fuzzed), MCP, the documentation's examples and the six showcases.
