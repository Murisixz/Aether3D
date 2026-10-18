# Procedural worlds

A scene can hold **worlds** (`scene.worlds`): landscapes generated from a small JSON
**definition** -- seed, extent, terrain recipe, climate, biomes, rivers and lakes, caves,
vegetation, structures -- plus sparse authored **edits**. A world is drawn by every
renderer, queried exactly (heights, slopes, biomes, water, blocks), edited with undo,
saved in a few kilobytes, scripted by agents and exported to glTF. Two modes share
everything below the surface: **terrain** worlds (continuous ground, water, plants,
volumetric caves) and **block** worlds (a Minecraft-style voxel landscape with original
textures).

Contents: [Quick start](#quick-start) - [Concepts](#concepts) - [The definition](#the-definition)
- [Terrain](#terrain) - [Hydrology](#hydrology-erosion-rivers-and-lakes) - [Climate and
biomes](#climate-and-biomes) - [Features](#features) - [Water](#water) - [Caves and
overhangs](#caves-and-overhangs) - [Block worlds](#block-worlds) - [Vegetation](#vegetation)
- [Structures and settlements](#structures-and-settlements) - [Streaming, level of detail
and memory](#streaming-level-of-detail-and-memory) - [Rendering](#rendering) - [Queries
and animation](#queries-and-animation) - [History, files and export](#history-files-and-export)
- [World scripts, MCP and the command line](#world-scripts-mcp-and-the-command-line) -
[Limits and security](#limits-and-security) - [Performance](#performance) -
[Showcases](#showcases) - [Limitations](#limitations) - [Errors](#errors)

## Quick start

```python
from aether3d import Scene

scene = Scene("Hills")
hills = scene.worlds.create("Hills", "temperate", seed=4, size=2048)  # nothing generated yet
print(hills.info()["mode"])  # terrain
print(round(hills.height(100.0, -50.0), 3))  # 333.989 (exact; the same on every platform)
print(hills.biome_at(100.0, -50.0))  # temperate_forest

# A flat, dry site; a house on it (its foundation follows the natural ground) and a yard.
site = hills.find_sites(1, radius=10.0, max_slope=12.0, spacing=80.0)[0]
x, y = site["at"]
hills.place_structure({"id": "inn", "kind": "house", "at": [x, y], "floors": 2})
hills.add_feature({"type": "clear", "id": "yard", "shape": {"circle": [x, y, 45.0]}})
print(len(hills.structures()), len(scene.history.entries))  # 1 3

g = hills.height(x, y)
scene.add_camera("Cam", position=(x - 24, y - 30, g + 9), look_at=(x, y, g + 3), far=4000)
scene.add_directional_light("Sun", direction=(0.5, 0.6, -0.4), intensity=1.6)
scene.world.atmosphere.configure(enabled=True, sun="Sun", ambient="hemisphere")
image = scene.render(width=320, height=180, backend="cpu")  # generated here, for this view
print(image.image.shape)  # (180, 320, 3)
scene.save("hills.aether")  # the definition and edits, never generated geometry
```

Nothing is generated when the world is created: the first height query runs the regional
stage (a fraction of a second for this 2 km world), and the render generates the tiles,
plants and the house it sees. For cinematic results, render on the GPU with cascaded
shadows and the atmosphere, as the [showcases](#showcases) do:

```python
scene.world.atmosphere.configure(fog_density=0.00015, fog_height=g, fog_falloff=0.0012)
scene.render(
    backend="gpu", width=1920, height=1080, shadows={"cascades": 4}, tonemap="neutral"
).save("hills.png")
```

The same from a JSON world script (see [World scripts](#world-scripts-mcp-and-the-command-line)):

```bash
aether3d new valley.aether
aether3d world run valley.aether valley.json
aether3d render valley.aether -o valley.png --frame --backend gpu
```

## Concepts

* **Definition** -- a validated JSON document (`world.definition`, a copy), the only
  input of generation. It is checked key by key (unknown keys, types, ranges, list
  sizes), completed with defaults, and never mutated: a change replaces it (one undo
  step), so results are cached by its content.
* **Edits** -- sparse authored data too large for a definition: block edits of block
  worlds (compressed sections of 16 x 16 x 16).
* **Lazy generation** -- creating or loading a world generates nothing. A render, a query
  or an export generates what it needs: the regional hydrology stage (once per world),
  terrain tiles, plants, block chunks, cave volumes. Caches keep results (LRU, byte
  budgets) and never hold the only copy of anything authored.
* **Determinism** -- every random decision is an integer hash of
  `(seed, stream, integer coordinates)` (a SplitMix64 finalizer): no global random state,
  so results never depend on the order tiles are generated, on threads or batches.
  Generation -- fields, the regional stage, tiles with their colours, plant placement,
  blocks, cave volumes and structure layouts -- uses only correctly rounded operations
  (`+ - * /`, `sqrt`, `floor`, comparisons) and integer hashing: no `exp`, `log`, `sin` or
  `pow` from the C library (`worlds/noise.py` has exact `cos_sin_deg`, `atan2_deg` and
  `log10`) and no BLAS (explicit ordered sums instead of matrix products), so the same
  definition gives the same float64 bits on every platform, with or without the native
  kernels. `tests/test_worlds_determinism.py` checks golden hashes and runs without the
  native kernels in a subprocess. Two things are outside this guarantee: the template
  meshes of plant species and buildings, whose shapes are drawn with the platform's
  trigonometry (not where they stand), and query outputs in degrees (`slope`).
* **Generator version** -- `definition["generator"]` (1). A later Aether3D that changes
  what a definition generates bumps it.
* **Coordinates** -- metres, +Z up, float64; worlds work anywhere within +-10,000 km
  (`MAX_COORDINATE`). Renders are camera-relative, so detail holds far from the origin.

## The definition

`worlds.create(name, preset=None, *, seed, size, extent, mode, overrides, materials=True)`
builds one from a preset (`"temperate"` by default) and overrides (a partial definition,
deep-merged); `world.configure(patch)` changes it (the mode is fixed). Sections:

| Section | What it holds |
| --- | --- |
| `seed`, `generator`, `mode`, `extent`, `sea_level` | identity: `extent` is `{"min": [x, y], "max": [x, y]}` (bounded, at most 2,000 km) or `null` (unbounded) |
| `terrain` | the height recipe: `continents` (land/ocean field and its height `curve`), `ranges` (where mountains are), `mountains` (ridged and eroded multifractals, `height`, `scale`, `octaves`, `ridged`, `roughness`), `hills`, `valleys`, `plateaus` (terraces), `canyons`, `dunes`, `detail`, `warp`, an optional final `curve`; `scale` and `height_scale` stretch everything |
| `climate` | `temperature` (sea-level mean, C), `variation`, `gradient` (per km), `lapse_rate` (C per km of height), `moisture`, `moisture_variation`, `moisture_gradient`, `scale`, `water_influence`, `water_distance`, `rain_shadow`, `wind` (degrees), `boundary_noise` |
| `biomes` | `preset` (`earth`), `table` (overrides and new biomes), `disabled`, `blend` |
| `surface` | `beach_height`, `cliff_slope`, `snow_temperature`, `treeline_temperature` |
| `hydrology` | `mode` (`drainage`, `contour`, `none`), `extent` (unbounded worlds), `cell`, `max_cells`, `erosion`, `rivers`, `lakes` |
| `caves` | `enabled`, `tunnels`, `caverns`, `overhangs`, `radius`, `depth`, `entrances`, `scale` |
| `vegetation` | `density` (trees and shrubs), `clutter` (ground cover), `rocks`, `distance`, `detail_distance`, `variants` |
| `structures` | settlement rules (`settlements`, `size`, `spacing`, `castles`, `towers`, `ruins`, `roads`, `bridges`, `biomes`, `max_slope`) and the authored `items` |
| `blocks` | block worlds: `size` (metres per block), `min_height`, `max_height`, `trees`, `ores`, `plants` |
| `materials` | layer -> scene material id (made by `create`) |
| `streaming` | `chunk` and `resolution` of level-0 tiles (powers of two), `lod_error` (pixels), `view_distance`, `volume_distance`, `memory_mb`, `max_tiles`, `threads` |
| `features` | authored terrain features, applied in order |

Errors name their location, e.g. `worlds[Valley].definition.terrain.mountains.height:
-5.0 is outside [0.0, 20000.0]`.

**Presets** (`scene.worlds.presets()`, `aether3d world presets`): `temperate` (the
defaults), `alpine_valley` (8 km of eroded high ranges, rivers and lakes),
`forested_mountains`, `archipelago`, `canyon_desert`, `snowy_wilderness`,
`rolling_plains`, `biome_continent` (a 24 km continent spanning climates),
`cavernous_highlands` (caves and overhangs), `infinite_highlands` (unbounded, contour
rivers) and `blocky_adventure` (a block world).

## Terrain

The authoritative height is

```
h(x, y) = remap(macro + layers) + biome modifiers + erosion delta (regional)
          + attenuated detail + river channels + authored features
```

* **Layers** are fractal noises: gradient (Perlin) noise, fBm, ridged multifractals,
  derivative-damped "eroded" fBm (Quilez: octaves fade where the surface is steep, as
  erosion would), billow; domain warping bends them; monotone (Fritsch-Carlson) curves
  remap heights; terraces shape plateaus. Each octave has its own derived seed.
* **Macro and detail**: octaves with a wavelength of at least four hydrology cells form
  the *macro* surface, which the regional stage erodes; finer octaves are added back over
  the eroded result (bicubic), attenuated near water.
* **Normals and slopes** come from the same function (central differences); meshes sample
  it exactly at their vertices.

## Hydrology: erosion, rivers and lakes

`hydrology.mode = "drainage"` runs a **regional stage** on a grid of `cell` metres over
the world (or `hydrology.extent` for unbounded worlds), once, cached:

1. depressions of the macro surface are filled, except the `lakes.max` largest (kept at
   40 % of their depth);
2. `erosion.iterations` rounds of: priority-flood filling with an epsilon slope (Barnes
   et al.), D8 receivers, drainage accumulation weighted by rain (from the moisture
   field), implicit stream-power incision (Braun and Willett; `n = 1`, `m = 0.5`),
   sediment transport and deposition (Yuan-like: deposition proportional to the carried
   flux over the catchment, never above the lowest donor), thermal talus relaxation
   (`talus` degrees);
3. lakes are labelled with one level each; rivers are traced from sources (drainage area
   above `rivers.threshold`) to their mouths as smoothed polylines with widths growing
   with the square root of their area and water levels that never rise downstream; a
   coarse distance-to-water field and a detail attenuation field are kept.

`world.info()["hydrology"]` estimates the stage's cost (nodes, bytes, native or not) before
it runs; `regional().summary()` reports what it made (rivers, km, lakes, eroded volume,
seconds). The sequential sweeps run in Rust kernels (`aether3d-native`, ABI 5) when
installed -- the same operations in the same order, bitwise equal to the NumPy twins
(`tests/test_native.py`). `mode = "contour"` makes rivers along noise valleys instead
(no regional stage: unbounded worlds without an extent); `"none"` has no rivers.

## Climate and biomes

Temperature falls with height (`lapse_rate`) from a sea-level field (mean, variation,
gradients); moisture comes from a noise field, nearness to water (`water_influence` over
`water_distance`) and a **rain shadow** (relief upwind, `wind`, dries the lee). Both are
computed on a 64 m global lattice and interpolated, so they are smooth and cheap.

**Biomes** are windows in (temperature, moisture) space -- the `earth` table has 16:
`ice_sheet`, `tundra`, `snowy_taiga`, `taiga`, `alpine_meadow`, `temperate_forest`,
`dense_woodland`, `birch_forest`, `plains`, `meadow`, `savanna`, `desert`, `badlands`,
`tropical_forest`, `swamp`, `rocky_highlands`. Weights blend smoothly across window edges
(`blend`, with `boundary_noise`). A biome declares its ground, colours (grass, soil, rock,
sand, foliage, water), terrain modifiers (dunes, flattening, roughness), vegetation
(species with densities per hectare, clustering, scale, steepest slope) and blocks (top,
filler, depth, underwater). Add or change biomes declaratively:

```python
valley.configure(
    biomes={
        "table": {
            "heath": {
                "temperature": [2, 9],
                "moisture": [0.2, 0.5],
                "ground": "grass",
                "colors": {
                    "grass": "#7c7a48",
                    "soil": "#5a4632",
                    "rock": "#7d776e",
                    "sand": "#c8bc96",
                    "foliage": "#5c6b3a",
                    "water": "#35677a",
                },
                "vegetation": [
                    {"species": "heather", "density": 800},
                    {"species": "rock", "density": 6},
                ],
            }
        }
    }
)
```

`biome` features force a biome locally. **Surface colours** blend the biomes' grounds
(grass dries toward straw where moisture is low, with tone noise), soil and rock by slope
(the cliff angle shifts colder and warmer), beaches, snow by temperature at the vertex's
own height, wet banks, river and lake beds, and roads. The terrain material multiplies
them by a world-space triplanar detail texture and bump.

## Features

Authored, ordered, with an `id` and a `shape` -- `{"circle": [x, y, r]}`,
`{"rect": [x0, y0, x1, y1]}`, `{"polygon": [[x, y], ...]}` or `{"line": [[x, y(, z)], ...],
"width": w}`:

| Type | Stage | Fields |
| --- | --- | --- |
| `raise`, `lower` | shape | `amount`, `falloff` |
| `scale` | shape | `factor`, `falloff`, `base` |
| `mountain` | shape | `height`, `radius`, `sharpness` (shape: `circle`) |
| `ridge`, `valley` | shape | `height` / `depth`, `falloff`, `sharpness` |
| `flatten` | final | `height` (absent: the ground at the centre when added), `falloff` |
| `pad` | final | `height` (`null`: the natural ground, resolved at generation), `falloff`, `structure` |
| `road` | final | `falloff`, `material`; `[x, y]` lines follow the natural ground, smoothed |
| `river` | final | `depth`, `level` (absent: the line's z) |
| `lake` | final | `level`, `depth`, `falloff` |
| `smooth` | final | `factor`, `falloff` |
| `biome` | final | `biome`, `strength`, `falloff` |
| `clear` | final | `what` (`trees`, `plants`, `rocks`, `all`), `falloff` |
| `cave` | volume | `kind` (`tunnel`, `arch`: lines with z; `cavern`, `hole`: circles), `radius`, `radius_end`, `noise` |

Shape features act before erosion (the land is eroded with them); final features after.
Each feature touches only the tiles within its reach, which are regenerated: the rest of
the world keeps its cached tiles and bits.

## Water

Seas fill below `sea_level`; lakes and rivers come from the regional stage or from
features. Water surfaces are their own meshes: flat at the water level (rivers follow
their sloping profile), coloured by depth -- clear shallows over the bed, the biome's
water colour in the deep -- with a smooth, specular material (`water` layer). Beds and
banks are coloured under and beside the water. **What it is not**: there are no waves,
no refraction and no screen-space reflections in the rasterizers; the path tracers
reflect the sky and the land in the water's specular surface. Rivers carve channels into
the ground along their polylines; channels narrower than the regional cell are
represented at the cell's resolution.

## Caves and overhangs

Near the camera (`streaming.volume_distance`, 400 m by default) a terrain world is also a
signed distance field, negative in rock:

```
phi(x, y, z) = max(z - h(x + wx, y + wy), -a(x, y, z))
```

`(wx, wy)` is a 3D noise warp acting only on cliffs (`caves.overhangs`: rock leans out
over its foot) and `a` the cave field, negative in air: noise **tunnels** (two noise
fields both near zero), **caverns** (thresholded fractal noise) -- sampled every 4 m on a
global lattice and interpolated -- kept between `caves.depth` metres below the ground
except around hashed **entrances** (never under water); and exact `cave` **features**.
The field is sampled every metre in 32 m cubes and meshed with **marching cubes** (the
case table of `sculpt.voxel`: closed, manifold surfaces). Cubes where cave air reaches the
ground, or with overhangs, are *surface* cubes: the terrain tiles over them refine to
level 0 and leave their footprint out, hanging skirts along the hole; on vertical cube
edges the field is linear in `z`, so the volume's ground vertices lie on the tiles'
vertices and the surfaces meet. Ground near the surface takes the terrain's colours;
deeper walls are the biomes' rock, darker and damper with depth. Closed caves are drawn
within 160 m of a camera below the ground, or under and beside an entrance; overhangs
within 160 m. Beyond, the ground is the closed heightfield (a documented level of
detail; `Volumes.solid` queries always use the exact field).

## Block worlds

`mode = "blocks"` (`blocky_adventure`): the same fields, sampled per block column, become
blocks (`aether3d world blocks`: 58 kinds -- stone, dirt, grass, sand, ores, logs, leaves,
planks, bricks, glass, water, lava, glowstone, plants...). **Textures** are original
16 x 16 pixel art drawn by `worlds/blocks.py` (no third-party game assets); grass, leaves
and water are painted grey and tinted by the biome. Materials sample them with nearest
filtering, mipmaps and repeat wrapping; plants are alpha-masked and double-sided; lava
and glowstone glow.

* **Generation** per 16 x 16 chunk: the biome's top block, filler and stone (dark stone
  deep down) with hashed ores, bedrock, banded terracotta in badlands, stone cliffs where
  columns step three blocks or more, beaches, snow on cold ground, caves (noise on a
  4-block lattice), water up to the water level (ice on cold water), block trees (one
  hashed candidate per 5 x 5 columns, decided on its own column, so a tree crossing a
  chunk border is the same on both sides), plants and flowers, and the structures.
* **Edits** (`set_blocks`, `fill_blocks`, `replace_blocks`, `revert_blocks`; block
  coordinates) override generation, persist in the project and undo.
* **Meshing**: a face is drawn where an opaque block meets a non-opaque one (hidden faces
  removed, across chunk borders); faces with the same texture, direction and per-corner
  ambient occlusion merge into rectangles (greedy meshing: runs along rows, then equal
  runs stacked). UVs are in blocks (textures repeat per block, oriented up on sides);
  vertex colours carry the ambient occlusion and biome tints. `tests/test_worlds_blocks.py`
  checks that the meshed area equals the exposed faces exactly.
* **Distance**: chunks within 8 chunks are meshed block by block; beyond, coarse levels of
  2, 4, 8... blocks per cell keep blocky silhouettes, with the same trees as boxes and
  walls hanging down at their edges.

## Vegetation

29 procedural species with three levels of detail each (`worlds/assets.py`, original
low-poly meshes with vertex colours): oak, beech, maple, birch, willow, pine, spruce, fir,
larch, jungle trees, palm, acacia, baobab, dead trees, cactus; bushes, shrubs, ferns,
grass, tall grass, flowers, reeds, mushrooms, dead bushes, heather; rocks, boulders,
pebbles and logs.

**Placement** (per 128 m tile and class, from integer hashes): trees share one candidate
grid (two trees never share a cell), other species have their own; each candidate grows
with the probability its biomes' densities give, modulated by grove noise and zeroed on
slopes too steep for the species, above the tree line, on snow (ground cover), in water
(reeds: shallows and marshes), on roads, pads, cleared ground and over cave openings.
Fields come from exact samples on a global 8 m lattice; survivors are checked against the
exact fields and stand on the authoritative ground. Every plant has a **stable id**
(`tree:<cell>`, `<species>:<cell>`).

**Drawing**: within each class's reach (trees `distance`, shrubs and rocks less, ground
cover `detail_distance`), faded out over its last fifth; beyond a class's thinning
distance a hashed fraction `(T / d)^3` remains, slightly enlarged -- nested sets, so
nothing pops in as the camera approaches; three levels of detail switch at hashed
distances. Plants are instances of shared meshes (instanced draws on the GPU). Beyond the
reach, forests are carried by the terrain's colours.

## Structures and settlements

Structures are **authored items** of the definition (`structures.items`): `house`,
`tower`, `castle`, `ruin` and `well` at `at` with `rotation`, `size`, `floors`, `style`
(`timber`, `stone`, `brick`) and `seed`; `bridge` along a two-point `line`. Each item but
a bridge implies a foundation **pad** whose height is the median of the *natural* ground
(the terrain before final features) at nine footprint points, resolved at generation:
a structure stays on the ground whatever regenerates around it. Meshes are procedural
and original -- plinths, walls, timber frames, gable roofs, chimneys, doors, windows;
towers with crenellations or conical roofs; castles with curtain walls, corner and gate
towers, a gate and a keep; bridges with railings and piers -- in the `stone`, `plaster`,
`timber`, `roof` and `wood` layers. Block worlds stamp block versions.

* `world.place_structure(item)`, `remove_structure(id)`, `structures(kind)`;
* `world.find_sites(count, radius=, max_slope=, biomes=, spacing=, region=, kind=)`:
  candidate sites, best first (flat, dry, allowed biomes; castles and towers prefer high
  ground);
* `world.route(a, b)` (read-only) and `build_road(a, b)`: A* over the natural ground with
  a grade penalty; water crossings (sampled every 2 m, so narrow rivers count) become
  bridges; roads stay inside the world;
* `world.build_settlements(count, houses=[min, max], roads=, seed=)`: villages on the best
  sites -- streets along the contour, houses facing them (never on water, too steep
  ground or each other), a well on the plaza, a tower in larger villages, a clearing --
  joined by roads; castles on high ground. Everything is **frozen** into items and road
  features in one undo step.

## Streaming, level of detail and memory

* **Tiles** form a quadtree over the world (`chunk` metres and `resolution` vertices at
  level 0; powers of two, so vertex coordinates are exact). A tile is refined while its
  measured geometric error projects above `lod_error` pixels and it lies in the view
  frustum widened by 600 m (shadows); the tree is restricted to 2:1 neighbours, finer
  edges next to coarser tiles are stitched (odd vertices moved onto the coarse segment)
  and every tile hangs a skirt -- no cracks. Selection depends only on the frame's camera.
* **Caches** are LRU with byte budgets from `streaming.memory_mb` (tiles; block chunks,
  meshes and coarse levels take 50/30/20 %); a frame's own working set is never evicted,
  so a frame larger than the budget still renders, and the budget bounds what stays
  cached between frames.
* **Threads** (`streaming.threads`): every tile and chunk is a pure function of its
  coordinates, so they can be computed in parallel with identical results. Generation is
  many NumPy operations on mid-sized arrays that hold the GIL between them; measured on an
  i9-12900H, 2 threads build block chunks about 17 % faster than 1 and more are slower, so
  the default is 1.
* **Prefetching**: `scene.worlds.prepare(times, camera=...)` generates what every world
  draws for an animated camera at those times, before rendering them.
* **Accounting** (`world.runtime.stats()["memory"]`, all measured): the world's logical
  size (area, level-0 tiles or blocks and chunks), cached bytes per kind (tiles,
  regional stage, vegetation, blocks, volumes), the last frame's parts, distinct meshes
  (draw batches) and triangles, and the process's resident memory. GPU memory is reported
  by GPU renders (`result.backend["resources"]["mesh_cache"]`).
* **Large coordinates**: float64 everywhere on the CPU; GPU instance matrices are made
  camera-relative before rounding to float32.

**Level of detail and authoritative data**:

| What | Authoritative (queries, structures, physics-like uses) | Drawn level of detail |
| --- | --- | --- |
| Ground height | `h(x, y)`, exact at any point | tiles: exact at vertices, linear between (error reported per tile, kept under `lod_error` pixels) |
| Caves, overhangs | the field `phi` (`Volumes.solid`) | meshed within `volume_distance` (closed caves nearer); heightfield beyond |
| Plants | every placed plant (stable ids, exact positions) | thinned beyond 400 m (trees), three mesh levels |
| Block worlds | every block (generation + edits) | blocks within 8 chunks, coarse cells beyond |
| Water | levels and kinds per point | surfaces per tile |

## Rendering

`render()` asks each world for its **contribution** for the frame's camera -- tiles and
water, cave volumes, plants, structures; or block chunks and coarse levels -- and appends
it to the render scene as meshes and instances with synthetic ids
(`world:<world id>:<part>`); nothing is added to the scene, its history or its files.
Materials come from the world's layers (`definition["materials"]`); a missing one draws
grey and reports `WORLD_MATERIAL_MISSING`. Every backend draws worlds: the CPU and GPU
rasterizers, the CPU and GPU path tracers.

For outdoor scenes, `scene.world.atmosphere` adds a **sky** (environment `sky`) with a sun
glow and a horizon glow, exponential **height fog** with its exact optical depth and sun
in-scattering, and **hemisphere ambient** light (sky above, ground below) -- identical
formulas on the CPU and in WGSL, also in the path tracers. **Cascaded shadow maps**
(`shadows={"cascades": 4, "distance": ...}`) keep sun shadows sharp near the camera over
wide views; shadows are a GPU rasterizer feature (the CPU rasterizer refuses them; the
path tracers trace them). `tonemap="neutral"` tames bright skies.

## Queries and animation

* `world.height(x, y)`, `ground(x, y)` (blocks: the top solid block), `sample(points,
  fields=)` (height, normal, slope, biome, biome_weights, temperature, moisture, water,
  water_kind), `biome_at`, `profile(points, samples)`, `block_at(x, y, z)`; points outside
  a bounded world raise `OUTSIDE_WORLD`;
* `world.ground_path(points, spacing=, clearance=, smooth=)`: polylines over the ground
  for characters (on it) and cameras (never lower than `clearance` above it, smoothed);
* `world.place_on_ground(objects, offset=)`: one undo step;
* animated cameras drive streaming: each frame selects what its camera needs, so frames
  render identically in any order (`tests/test_worlds_render.py`); `scene.worlds.prepare`
  generates a camera path ahead.

## History, files and export

Creating, configuring, editing and removing worlds are history steps (the `world.item`
aspect: name, definition and edits), saved in history files too. Projects store each
world's definition and edits only (project format 10; version 9 projects migrate with no
worlds and the default atmosphere); a terrain world with features and structures takes a
few kilobytes. `world.export_region(path, [x0, y0, x1, y1], level=, include=, plant_lod=)`
(`aether3d world export`) writes a region's generated geometry -- tiles at one level,
water, caves, plants (instanced), structures, or block chunks -- as glTF through the
regular writer, with vertex colours as `COLOR_0`.

## World scripts, MCP and the command line

An **`aether3d.world-script`** is applied atomically (one undo step; on any error nothing
changes) and may ask read-only queries afterwards:

```json
{"format": "aether3d.world-script", "version": 1, "label": "Valley with a village",
 "operations": [
   {"op": "create", "name": "Valley", "preset": "alpine_valley", "seed": 7, "size": 3072},
   {"op": "add_feature", "feature": {"type": "lake", "id": "tarn",
                                     "shape": {"circle": [600, -300, 120]}, "level": 1380}},
   {"op": "build_settlements", "count": 1, "houses": [6, 10]}
 ],
 "queries": [{"query": "structures"}, {"query": "sample", "points": [[0, 0]]}]}
```

Operations: `create`, `remove`, `rename`, `configure`, `add_feature`, `update_feature`,
`remove_feature`, `place_structure`, `remove_structure`, `build_settlements`,
`build_road`, `set_blocks`, `fill_blocks`, `replace_blocks`, `revert_blocks`. Queries:
`info`, `sample`, `profile`, `sites`, `route`, `block`, `stats`, `structures`,
`features`. Schema: `aether3d world schema` or `aether3d://schema/world-script`; agent
guide: `aether3d world guide` or `aether3d://worlds/guide`.

* **MCP**: `run_script` with a world script; `inspect_world` (read-only: `summary`,
  `info`, `sample`, `profile`, `sites`, `route`, `block`, `stats`); renders through
  `render_preview` and `render_image`.
* **CLI**: `aether3d world run PROJECT SCRIPT`, `info`, `sample --at X,Y`, `export OUT.glb
  --region X0,Y0,X1,Y1`, `presets`, `blocks`, `schema`, `guide`; `aether3d render --frame`
  frames worlds by their ground.

## Limits and security

World definitions and scripts are data: validated before anything changes, never
executed; no operation reads or writes files (exports are explicit calls), and scripts
have a time budget (`limits.max_seconds`) and cancellation. Ceilings: extents up to
2,000 km and coordinates within +-10,000 km; 4,096 features of up to 4,096 points; 4,096
structures; regional grids of `max_cells` nodes (4M by default); 1M points per query;
512 operations and 64 queries per script; 8 km export regions and 20M exported triangles;
block edits of at most 4M cells per call; caches within `streaming.memory_mb`. Inputs that
exceed them raise `LIMIT_EXCEEDED` or `INVALID_*` errors with locations.

## Performance

Measured on the reference laptop (Intel i9-12900H, RTX 4060 Laptop GPU) with the native
kernels (`benchmarks/bench_worlds.py`; figures and methods in docs/PERFORMANCE.md):
the regional stage of `alpine_valley` (263k nodes, 60 erosion iterations) takes about 6 s
with the native kernels; vegetation placement for a 2.5 km view about 1 s (it was 48 s
before placement was batched); a first block-world frame (357 chunks, 289 meshed, 472
coarse) about 6 s, later frames of a moving camera reuse what they can. Profiling shows
generation is dominated by NumPy noise evaluation and per-call overhead; batching is
where the big wins were.

## Limitations

* **Water** is flat at its level (rivers follow their profile) and coloured by depth: no
  waves, foam, refraction or screen-space reflections in the rasterizers; the path
  tracers reflect the sky and the land. Channels narrower than the regional cell are
  drawn at the cell's resolution.
* **Level of detail** is not morphed: a tile switches level when its projected error
  crosses `lod_error` pixels (3 by default), so a switch moves the silhouette by at most
  about that much. Plants thin with distance through nested sets (nothing pops in as the
  camera approaches) and switch their three mesh levels at hashed distances.
* **Caves and overhangs** are meshed within `volume_distance` (400 m) of the camera; beyond,
  the ground is the closed heightfield. Erosion and rivers do not run through caves.
* **Unbounded worlds** have no regional stage: rivers follow noise valleys (`contour`
  hydrology) and erosion is the derivative-damped noise only.
* **Block worlds** span `min_height` to `max_height` (-64 to 192 blocks by default; at
  most +-4,096); lighting is ambient occlusion plus the scene's lights (no block light
  propagation); fluids do not flow.
* **Shadows** come from the GPU rasterizer's (cascaded) shadow maps and the path tracers;
  the CPU rasterizer draws none. Fog is aerial perspective along camera rays, not a
  participating medium (no light shafts).
* **Threads**: generation is NumPy work that holds the GIL between operations, so threads
  help little (measured above); the heavy sequential sweeps are native.
* **Not simulated**: physics and collisions, weather, seasons, day-night cycles (the sun
  is the scene's light), wildlife, water flow over time.

## Errors

`WorldError` (`aether3d.errors`): `code`, `message`, `location`, `hint`, `details`.
Codes include `INVALID_DEFINITION`, `UNKNOWN_BIOME`, `UNKNOWN_PRESET`, `UNKNOWN_WORLD`,
`UNKNOWN_FEATURE`, `UNKNOWN_STRUCTURE`, `UNKNOWN_BLOCK`, `DUPLICATE_ID`,
`INVALID_ARGUMENT`, `INVALID_SCRIPT`, `OUTSIDE_WORLD`, `NOT_A_BLOCK_WORLD`, `NO_ROUTE`,
`TOO_LARGE`, `LIMIT_EXCEEDED`, `BUDGET_EXCEEDED`, `CANCELLED` and
`GENERATION_CANCELLED`.
