# Texturing: UV mapping, texture painting and baking

Aether3D gives a mesh UVs, paints textures on it in 3D, and bakes detail from
high-resolution sources into its textures -- all in the engine, all undoable, all
reachable from Python, the command line, JSON scripts and MCP (0.14.0).

* [Quick start](#quick-start)
* [Conventions](#conventions) -- UV orientation, texel centres, encodings, color spaces
* [UV mapping](#uv-mapping) -- unwrapping, projections, island tools, packing,
  diagnostics, layout images
* [Tangent space](#tangent-space) -- MikkTSpace in every renderer and in glTF
* [Painted textures](#painted-textures) -- layers, blend modes, masks, history, files
* [Painting](#painting) -- brushes, strokes in surface, view and UV space, fills
* [Baking](#baking) -- normal, AO, curvature, thickness, height, IDs, material channels
* [Sculpting and remeshing](#sculpting-and-remeshing) -- texture transfer
* [Texture scripts, CLI and MCP](#texture-scripts-cli-and-mcp)
* [Safety and limits](#safety-and-limits)
* [Performance](#performance)
* [Limitations](#limitations)
* [Examples](#examples) and [references](#references)

## Quick start

```python
from aether3d import Scene, create_sphere

scene = Scene()
ball = scene.add_object("Ball", mesh=create_sphere(1.0, 64, 32))
unwrap = ball.unwrap(resolution=1024, padding=8)  # one undo step
uvs = ball.uv_report(resolution=1024)
print(unwrap["charts"], uvs["flipped_triangles"], uvs["overlapping_triangles"])  # 2 0 0

paint = scene.add_painted_texture("Paint", 1024, 1024, background="#3d6fb5")
paint.add_layer("Ink")
material = scene.create_material("Painted", base_color="#ffffff")
material.set_texture("base_color", paint)
ball.material = material

# A stroke in world space across the ball (radius in world units): one undo step.
stroke = ball.paint(paint, {"radius": 0.15, "color": "#f2c230"}, [[0, -1, 0.3], [0.7, -0.7, 0.3]])
print(stroke["dabs"], stroke["tiles"])  # 34 12
print(scene.history.entries[-1].label)  # Paint stroke (paint) on 'Ball'

# Bake normals and ambient occlusion from a detailed source onto the ball's UVs, and
# bind them to its material (one undo step).
scene.add_object("Detail", mesh=create_sphere(1.0, 256, 128))
result = scene.bake(ball, ["Detail"], ["normal", "ao"], resolution=512, apply=True, backend="cpu")
maps = sorted(result.report["applied"]["textures"])
print(result.report["miss_fraction"], maps)  # 0.0 ['ao', 'normal']
```

## Conventions

* **UVs** are per vertex (`Mesh.uvs`, `(V, 2)`), `u` to the right and `v` **up**: the
  texel row 0 of an image is `v = 1` (as in OpenGL). glTF's `TEXCOORD_0` has `v` down;
  import and export flip it (`1 - v`), so a texture looks the same in both.
* A texel `(x, y)` of a `W x H` texture covers `u in [x/W, (x+1)/W]`,
  `v in [1 - (y+1)/H, 1 - y/H]`; its centre is at `((x + 0.5)/W, 1 - (y + 0.5)/H)`.
  Rasterization, painting and baking all use texel centres (with the CPU rasterizer's
  fixed-point edge functions and top-left rule: a centre on an edge shared by two
  triangles belongs to exactly one).
* **Encodings** of painted textures: `color` (display sRGB values, decoded to linear
  light before shading: base color, emission), `data` (linear values: roughness,
  metallic, occlusion, masks, AO, curvature, height) and `normal` (tangent-space unit
  vectors stored `n * 0.5 + 0.5`). A texture's encoding sets its color space for every
  renderer and for glTF export.
* **Texture slots multiply their factor**: `base_color x texture`, `roughness x
  texture`, ... (see [MATERIALS.md](MATERIALS.md)). Set the factor to 1 (`"#ffffff"`)
  to see painted values as they are; `apply_bake` does this for the slots it binds.
* Number slots read one **channel** (roughness `g`, metallic `b`, occlusion `r`, opacity
  `a` by default; `set_texture(..., channel="r")` reads another). Baked and painted
  single-value maps hold the value in every color channel.

## UV mapping

### Unwrapping

`obj.unwrap(**options)` (or `aether3d.uv.unwrap(mesh, ...)`, which returns a new mesh)
computes UVs in four stages:

1. **Charts.** The surface (welded by exact position) is cut at its boundaries,
   non-manifold and inconsistently oriented edges, the `seams` you give (`[vertex,
   vertex]` pairs of mesh edges), and by `split_by`: `"hard"` (edges whose stored normals
   differ on the two sides -- auto-smooth creases), `"groups"` (face-group borders),
   `"seams"` (the mesh's existing UV seams). Edges whose dihedral angle exceeds
   `angle_limit` (default 66 degrees) are cut too: each flat side of a hard-surface model
   becomes its own chart. Charts are the connected pieces between cuts.
2. **Flattening.** Every chart that is a topological disk is flattened with LSCM (least
   squares conformal maps, Levy et al. 2002) and, with `method="isometric"` (default),
   refined by SLIM (scalable locally injective maps, Rabinovich et al. 2017: the symmetric
   Dirichlet energy, each step flip-free). A chart LSCM would fold falls back to a Tutte
   embedding. The linear systems are solved by a sparse LDL^T factorization with nested
   dissection ordering (Rust, `aether3d-native`, with a bit-identical NumPy fallback).
3. **Splitting.** A chart is accepted when no triangle is flipped, its border does not
   cross itself, and its L2 stretch (Sander et al. 2001, scale-normalized) is at most
   `max_stretch` (default 1.15). Otherwise it is split in two (geodesic Voronoi of two far
   seeds; staircase teeth of the border removed) and both halves start again at 2.
   Charts that are not disks (a tube, a torus) are split the same way.
4. **Packing.** Charts are rotated to their smallest bounding rectangle and placed in the
   unit square with `padding` texels between them at `resolution` (`[w, h]` for
   non-square textures), each in its best quarter-turn orientation (`rotate=True`), on a
   skyline of conservative texel footprints -- so islands never share a texel. The scale
   is the largest that fits (bisection), or `texels_per_unit` fixes the texel density
   (`ATLAS_FULL` if the charts do not fit).

`faces=` unwraps only some triangles (the others keep their UVs); `atoms=` keeps groups of
triangles in one chart (an editing session passes its polygons, so quads are never split
across charts). `should_stop` cancels. Everything is deterministic (ties break by lowest
id). The report gives `charts`, `splits`, stretch, the packing scale and timings, with
`diagnostics`.

### Projections and island tools

`obj.project_uvs(kind, axis=..., center=..., pack=...)` -- `"planar"` (along an axis),
`"box"` (each face along its dominant axis; one island per side), `"cylindrical"`,
`"spherical"` (seams where the angle wraps, so no triangle spans the wrap). Use them for
simple shapes or deliberate projections; check the result (a planar projection folds the
back onto the front).

`obj.edit_uvs(operation, ...)` works on islands (UV-connected pieces):

| operation | what it does |
| --- | --- |
| `transform` | `mirror` (`"u"`/`"v"`), `scale` (one or two factors), `rotate` (degrees, counter-clockwise) and `translate` islands, in that order, each about its own box centre (`pivot="center"`) or a common `[u, v]` pivot |
| `align` | rotate islands about their centres: `mode="min_area"` (their minimum-area rectangle, wider than tall), `"u"` / `"v"` (their longest hull edge horizontal / vertical) |
| `normalize` | scale and move the selected islands together to fill the unit square (`keep_aspect`) |
| `texel_density` | scale islands to `texels_per_unit` at `resolution` |
| `pack` | repack (`resolution`, `padding`, `rotate`, `texels_per_unit`, `keep_density`) |
| `stitch` | weld UV seams along `edges` (`[vertex, vertex]` pairs): the higher-numbered island is moved rigidly (mirrored if the sides' orientations differ) onto the other and the edges are welded; refused (`STITCH_OVERLAP`) if the joined islands would overlap, unless `allow_overlap` |

### Diagnostics

`obj.uv_report(resolution=...)` (`aether3d.uv.uv_report`) measures the map exactly:

* `islands`, `flipped_triangles` (UV winding against its island's), `degenerate_triangles`
  (zero UV area, non-zero 3D area), `nonfinite_triangles`, `mirrored_islands` (every
  triangle reversed: deliberate on mirrored models, reported apart);
* **overlaps**, found with exact orientation predicates: `overlapping_triangles` counts
  triangles in *unintended* overlaps (folds inside an island, `self_overlapping_islands`,
  and `overlapping_pairs` of islands); `stacked_pairs` and `stacked_triangles` are island
  pairs with the same triangle count and UV area laid on each other -- copies sharing
  texels on purpose, not errors;
* `stretch_l2` (max, area-weighted mean, invalid islands) and `stretch_max` (Linf);
* `texel_density` at `resolution`: mean, min, max and `spread` (area-weighted
  coefficient of variation; 0 means every island has the same density);
* `utilization` (UV area in the unit square), `bounds`, `islands_outside_unit_square`,
  `min_gap_texels` (the smallest free texel gap between two islands' footprints) and
  `per_island` details.

`aether3d.uv.layout_image(mesh, size, mode=...)` draws the layout (v up, island borders
dark): `islands` (a color each), `density` (blue half the mean density, green equal, red
double), `stretch` (anisotropy), `overlaps` (unintended red, flipped orange, stacked
yellow) or `texture` (an image under the wireframe). `inspect_uvs` (MCP) returns it.

## Tangent space

Tangent-space normal maps need a tangent frame per corner that the baker and every
renderer agree on. Aether3D computes **MikkTSpace** tangents (Mikkelsen 2008, the
standard of Blender, Substance, Unity, Unreal and glTF) with a float32 port of the
reference implementation that reproduces it bit for bit (`tests/data/mikktspace`; the one
known difference -- the reference's unsorted last group on non-manifold edges -- is
documented there). The renderers (CPU rasterizer, GPU rasterizer, CPU and GPU path
tracers) use the per-corner tangent and sign interpolated *unnormalized* with the
interpolated normal, bitangent `sign x cross(N, T)`, then normalize the perturbed normal
-- the convention MikkTSpace defines. Mirrored instances (negative determinant) flip the
sign. glTF export writes `TANGENT` (vertices are split where corners disagree:
`VERTICES_SPLIT_FOR_TANGENTS`), so other engines see the same normals.

Tangents are cached per mesh (meshes are immutable). They are used for normal textures
bound through UVs with a transform that keeps the tangent basis (any offset and a positive
uniform scale); a binding that rotates, mirrors or scales the UVs non-uniformly falls back
to each triangle's UV-gradient frame, as in 0.13 (see [Limitations](#limitations)).

## Painted textures

`scene.add_painted_texture(name, width, height, encoding=, background=, image=)` creates
a texture document (`PaintedTexture`, a kind of image texture: bind it to any slot):

* a **background** RGBA under everything (a `"#rrggbb"` color or 1-4 numbers);
* up to 64 **layers**, bottom to top: `paint` layers hold texels, `fill` layers a single
  `value`; each has a `blend` mode (`normal`, `multiply`, `screen`, `overlay`, `darken`,
  `lighten`, `difference`, `add`, `subtract`; W3C compositing formulas), an `opacity`, a
  `visible` flag and an optional **mask** (0..1 per texel, `set_mask`);
* texels stored as 16-bit RGBA in 64 x 64 **tiles**: a layer keeps only the tiles that
  were painted, so a stroke on a 4K texture touches a few tiles, not 16 million texels.

Methods: `add_layer`, `remove_layer`, `move_layer`, `set_layer` (name, blend, opacity,
visible, value), `clear_layer`, `set_mask`, `flatten`, `set_background`, `write_pixels`;
reading: `read()` (the composite, float 0..1), `read_layer`, `read_mask`, `pixels16()`,
`pixels8()`, `export_png(path, bits=8|16)`. Every change is a step of the scene's history
(tile deltas, not whole images) and bumps the texture's `revision`.

**Rendering.** Renderers sample the composite through the usual image path (mipmaps,
filtering, wrap). After a stroke only the changed tiles are recomposited, re-encoded and
their mip regions rebuilt (bit-identical to a full rebuild); the GPU uploads only the
changed rows (`stats["partial_uploads"]`).

**Files.** Projects (format 7, [FILE_FORMAT.md](FILE_FORMAT.md)) store every layer and
mask losslessly (16-bit PNG, embedded); format 6 projects load unchanged. History files
store tile deltas compressed. glTF export writes the 8-bit composite PNG; glTF has no
layers.

## Painting

`obj.paint(targets, brush, points, **options)` (`aether3d.texturing.stroke`) paints one
stroke; `obj.fill_texture(texture, value, where=...)` fills. Each is **one undo step**
(all the textures it touches together), rolled back if anything fails, cancellable
(`should_stop`).

**Targets**: a painted texture (its top paint layer, with the brush color), or a list of
`{"texture", "layer", "color"}` -- a *material stroke* painting base color, roughness,
metallic, emission... at once, each with its own value.

**Spaces** (`space=`):

* `"surface"` (default) -- `points` are 3D, in **world** space (`coordinates="object"`
  for the object's own space), and the brush `radius` is in the same units. Every texel
  whose surface point lies within the radius of a dab is painted (texel positions come
  from the exact UV rasterization), so strokes continue across UV seams with no visible
  edge. Texels facing away from the dab's normal by more than `normal_limit` degrees are
  left alone (the far side of a thin wall). `symmetry` (`"x"`, `"xz"`, ...) mirrors
  each dab across the object's own planes.
* `"view"` -- `points` are pixels of a view: `PaintView.render(scene, camera, w, h)`
  renders the scene's depth and object buffers once; a texel is painted when its surface
  point projects within the radius (pixels) **and is visible there** (the nearest surface
  at that pixel, facing the camera) -- projection painting with occlusion. `backfaces`
  paints through.
* `"uv"` -- `points` are `[u, v]`, the radius in texels: painting the image itself.

**Brushes** (`Brush` fields; also plain dicts): `kind` -- `paint`, `erase` (lowers the
layer's alpha), `blur` (averages over `blur_radius` texels), `smudge` (drags colors
along the stroke), `clone` (copies from `clone_offset` texels away), `mask` (paints the
layer's mask); `radius`, `strength` (per dab), `opacity` (the most a stroke can lay:
dabs build up within a stroke but never beyond it), `hardness`, `falloff` (`smooth`,
`linear`, `constant`, `gaussian`), `spacing` (fraction of the radius between dabs),
`color`, `blend` (how the stroke composes onto the layer), `normal_limit`, `backfaces`,
`symmetry`. `pressure=` gives a value per point (interpolated along the path; it scales
the dab's strength).

**Padding.** Strokes also paint `padding` rings (default 2) of the texels outside every
island next to painted ones, so bilinear filtering and the first mip levels do not bleed
the old colors at seams.

`fill(..., where={"islands": [...]}` / `{"groups": [...]}` / `{"triangles": [...]}`)
fills a layer everywhere or on those parts; `gradient(scene, obj, target, start, end,
colors)` blends two colors along a 3D segment over the surface.

## Baking

`scene.bake(target, sources, maps, **options)` (`aether3d.texturing.bake`) fills texture
maps of `target` (an object with UVs) from the surfaces of `sources` (the target itself
by default: a self-bake of AO or curvature):

1. The target's UV triangles are rasterized exactly with `samples x samples` stratified
   samples per texel (default 2 x 2); each is a point of the target's surface with its
   normal and MikkTSpace frame.
2. A ray starts on the **cage** -- the point pushed out by `cage` along the target's
   averaged normal (normals averaged over equal positions, so hard edges do not open
   gaps), or the matching point of an explicit `cage_mesh` -- and runs back through the
   surface until `depth` past it. Its first hit on the sources is where the sample's
   values come from. `cage` and `depth` default to 2% of the bounding diagonal of the
   target and sources each.
3. Each map is computed at the hit; texels average their samples (normals renormalized,
   colors averaged in linear light); uncovered texels get `padding` rings of dilation
   (default 8).

| map | value | encoding |
| --- | --- | --- |
| `normal` | the source's normal in the target's MikkTSpace tangent space: the exact inverse of the renderers' transform | normal |
| `object_normal`, `world_normal` | the source's normal in object or world space | normal |
| `height` | signed distance from the target's surface to the hit, `0.5 + 0.5 d / height_range` | data |
| `ao` | ambient occlusion: the visible fraction of `ao_rays` cosine-weighted directions within `ao_distance` | data |
| `thickness` | mean distance to the next surface inside the source over `thickness_rays` directions, / `thickness_distance` | data |
| `curvature` | mean curvature (cotangent Laplacian, Meyer et al. 2003), convex > 0.5, `0.5 + 0.5 k curvature_radius` | data |
| `position` | the hit in the target's object space, normalized to its box | data |
| `base_color`, `roughness`, `metallic`, `emissive`, `opacity` | the source's material evaluated at the hit, textures included | color / data |
| `material_id`, `object_id`, `group_id` | a distinct color per source material, object or face group | color |

`apply=True` stores the maps as painted textures, binds them (`normal`, `ao` ->
occlusion, material channels) to the target's material and sets the factors they
multiply to neutral -- one undo step. `result.to_texture(scene, map)` and
`result.save_png(map, path, bits=8|16)` give single maps.

**Diagnostics.** The report counts `texel_samples`, `hits`, `miss_fraction`,
`texels_missed` (texels none of whose samples hit) and `uv_overlap_samples` (samples
covered by more than one target triangle: the lowest triangle wins). `BAKE_MISSES`
(misses: raise `cage` or `depth`, or check that the sources cover the target) and
`UV_OVERLAP` (overlapping target UVs share texels) are warnings.

**Backends.** Rays are traced by the GPU when one works (`backend="auto"`; `"gpu"`
requires one, `"cpu"` never uses it): the path tracer's BVH and watertight triangle test,
with AO and thickness hemispheres generated on the GPU. The CPU traces with every core.
Results agree to float32 rounding: normals within 1e-5, and the odd AO or thickness ray at
a grazing contact may differ (mean difference ~1e-6 measured); a self-bake on the GPU can
miss a few samples in a million where the CPU misses none. A bake is deterministic for a
given backend and device (`seed` sets the AO and thickness directions).

## Sculpting and remeshing

Sculpt strokes keep UVs: vertices move, UVs stay, so painted textures follow the
surface. Dynamic topology interpolates UVs on the edges it splits (each side of a UV seam
separately). Voxel remeshing and isotropic remeshing make new vertices: voxel remeshing
drops UVs (`VOXEL_UVS_DROPPED`). To keep the look, duplicate the object first and
**transfer**:

```python
before = scene.duplicate(head, "Head before")
with head.sculpt() as session:
    session.voxel_remesh(0.03)
    session.commit()
scene.transfer_textures(before, head, resolution=2048)  # unwrap + bake + bind, one undo step
scene.remove(before)
```

`transfer_textures` unwraps the target if it has no UVs, bakes the source's evaluated
material (base color, roughness, metallic and its shading normals as a normal map by
default) and binds the maps to a material of the target's own when its material is shared.

## Texture scripts, CLI and MCP

A **texture script** is a JSON document of ordered operations, applied as one undo step,
all or nothing; it never reads or writes a file and runs no code:

```json
{
  "format": "aether3d.texture-script",
  "version": 1,
  "label": "Paint the crate",
  "limits": {"max_seconds": 120, "max_texels": 16777216, "max_dabs": 50000},
  "operations": [
    {"op": "unwrap", "object": "Crate", "resolution": 1024, "padding": 8},
    {"op": "add_texture", "name": "Crate color", "width": 1024, "height": 1024,
     "background": "#8a6a48", "layers": [{"name": "Dirt"}]},
    {"op": "bind", "object": "Crate", "slot": "base_color", "texture": "Crate color",
     "factor": "#ffffff"},
    {"op": "fill", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
     "color": "#3b2a1a", "opacity": 0.3},
    {"op": "stroke", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
     "brush": {"radius": 0.08, "color": "#202020", "strength": 0.6},
     "points": [[1.0, -0.6, 0.2], [1.0, 0.6, 0.4]]},
    {"op": "uv_report", "object": "Crate"}
  ]
}
```

Operations: `unwrap`, `project_uvs`, `edit_uvs`, `uv_report`, `add_texture`, `add_layer`,
`set_layer`, `remove_layer`, `clear_layer`, `flatten`, `texture_info`, `bind`, `stroke`
(`space`, `coordinates`, `camera` and `view_size` for view strokes, `targets` for material
strokes), `fill`, `gradient`, `bake` (`apply`), `transfer`. Each operation accepts exactly
its own arguments (unknown keys are refused). Errors are `TexturingError`s with a stable
`code` and the step's `location` (`operations[4].brush.radius`). The JSON Schema is
`aether3d texture schema` / the MCP resource `aether3d://schema/texture-script`.

Python: `aether3d.texturing.run_texture_script(scene, script)`.

**CLI** (the `texture` command):

| command | does |
| --- | --- |
| `aether3d texture run PROJECT SCRIPT` | apply a texture script, save (`--dry-run`, `--output`) |
| `aether3d texture report PROJECT [--object NAME] [--require-clean]` | UV diagnostics; exit 3 with `--require-clean` if a mesh has flipped or overlapping UVs |
| `aether3d texture bake PROJECT --target OBJ --source OBJ --map normal --map ao --out DIR` | bake maps to PNG files (`--bits 16`), and/or bind them in the project (`--apply`) |
| `aether3d texture schema` | the JSON Schema of texture scripts |

**MCP** ([MCP.md](MCP.md)): `run_script` runs texture scripts; `inspect_uvs` returns the UV
report with a layout image (`islands`, `density`, `stretch`, `overlaps`, or a texture);
`inspect_texture` a texture's layers, users and a preview; `capabilities` topic
`texturing`; resources `aether3d://texturing/guide` (an agent guide) and the schema.
**Feedback** ([FEEDBACK.md](FEEDBACK.md)): the `uv_quality` expectation check finds
flipped and overlapping UVs (`UV_FLIPPED`, `UV_OVERLAP`, exact, basis `geometry`),
stretch, density spread and gaps beyond the limits you state, and missing UVs; it is never
corrected automatically.

## Safety and limits

* Scripts are data: per-operation argument allowlists (no callback or internal option is
  reachable from JSON), texture references are names or ids only (never a path, so no
  file is read), and nothing is written. MCP keeps its workspace confinement.
* Limits: textures up to 16384 per side and 8192 x 8192 texels; 64 layers; strokes of at
  most 100,000 points and 100,000 dabs; bakes of at most 2^26 texel samples and 2^32
  rays; unwraps of at most 20 million triangles; scripts of at most 10,000 operations,
  with `limits.max_seconds`, `max_texels` and `max_dabs`.
* Invalid input (NaN, ragged arrays, wrong types, unknown keys) is refused with a
  `TexturingError` before anything changes. The script fuzzer of
  `tests/test_texture_script.py` (60 mutated scripts per test run; 3,000 in a longer
  local run) found only structured errors or atomic successes.

## Performance

Measured on the development laptop (Intel Core i9-12900H, 20 threads; NVIDIA RTX 4060
Laptop GPU; Python 3.14, NumPy 2.5; `aether3d-native` installed) by
`benchmarks/bench_texturing.py` at commit d5c70b9 (`benchmarks/results/stage14-texturing.json`); compare
runs on one machine only.

UV spheres of about 10k, 100k and 1M triangles (each sphere unwraps into two hemisphere
charts: the largest charts a sphere can give, the worst case for the solver), unwrapped at
2048; textures painted on the 100k sphere; maps baked onto a 10k sphere from a bumpy
1M-triangle sphere. Each group runs in its own process (peak memory per process).

| | 10k triangles | 100k | 1M |
| --- | ---: | ---: | ---: |
| `unwrap` (native LDL^T / NumPy only) | 1.0 s / 5.9 s | 9.3 s / 87 s | 271 s / -- |
| `uv_report` (exact overlaps, stretch, density, gaps) | 0.71 s | 3.4 s | 46 s |
| `layout_image` at 1024 | 0.19 s | 0.52 s | 2.7 s |
| MikkTSpace tangents | 0.024 s | 0.25 s | 3.1 s |
| peak memory of the unwrap process | 313 MiB | 878 MiB | 9.2 GiB |

The unwrap's time is mostly the sparse factorization of the two big charts (four
factorizations; their fill grows faster than the charts), which also takes the memory at
1M; meshes that segment into more charts unwrap faster. Packing the 100k sphere's charts
again: 1.3 s at 1K, 2.3 s at 2K, 4.8 s at 4K, 12.6 s at 8K.

| Painting (100k-triangle sphere) | 1K texture | 2K | 4K | 8K |
| --- | ---: | ---: | ---: | ---: |
| first dab of a session (surface BVH, texel maps) | 130 ms | 137 ms | 146 ms | 193 ms |
| 100-dab stroke, radius 0.05: per dab | 1.3 ms | 2.0 ms | 3.6 ms | 8.4 ms |
| 67-dab stroke, radius 0.2: per dab (tiles touched) | 17 ms (54) | 22 ms (144) | 36 ms (437) | 85 ms (1,437) |
| fill of the whole surface | 0.22 s | 0.89 s | 5.5 s | 21.8 s |
| composite of the whole texture | 0.04 s | 0.17 s | 0.69 s | 2.7 s |
| 8-bit level 0 and mipmaps of the whole texture | 0.07 s | 0.32 s | 1.4 s | 5.9 s |
| undo of the fill | 0.09 ms | 0.19 ms | 0.37 ms | 1.1 ms |
| peak memory | 876 MiB | 876 MiB | 1.6 GiB | 4.5 GiB |

A stroke costs what it covers (tiles, texels), not the texture's size; renders after a
stroke recomposite and re-upload only its tiles.

| Baking from a 1M-triangle source, 2 x 2 samples per texel | 1K | 2K | 4K |
| --- | ---: | ---: | ---: |
| normal map, CPU rays (all cores) | 6.1 s | 13.8 s | 45.3 s |
| normal map, GPU rays | 3.5 s | 11.8 s | 52.4 s |
| AO, 64 rays per sample, CPU / GPU | 27.3 s / 3.8 s | | |
| ... ray rate, CPU / GPU | 5.3 / 38.5 M rays/s | | |

The shading of hits (tangent frames, interpolation, accumulation) runs on the CPU for
both backends and dominates normal maps (27 s of the 45 s at 4K); one projection ray per
sample is not worth the GPU's transfers at 4K, so `backend="auto"` uses the GPU for AO
and thickness only (about 7 times faster there). Setup (the source's BVH) is 3.4 s for
1M triangles on the CPU.

## Limitations

* No multi-object atlas: each object is unwrapped and packed into its own UV square (two
  objects can share a texture only through UVs you arrange with `edit_uvs`).
* No UDIM tiles: UVs outside `[0, 1]` wrap or clamp as the sampler says.
* Chart borders on meshes of long, thin, diagonally split quads zigzag along the
  triangles (the shortest edge path); seams are not straightened to feature lines.
* Packing is a skyline of texel footprints with quarter-turn rotations, not arbitrary
  angles or nesting of islands into each other's holes.
* MikkTSpace tangents are used when the normal texture's binding keeps the UV basis
  (offset and positive uniform scale); a rotated, mirrored or non-uniformly scaled
  binding uses the per-triangle UV-gradient frame of 0.13.
* glTF export writes the composite of painted textures at 8 bits (layers and 16-bit
  precision stay in the project file).
* Emission baked from a source is clamped to [0, 1] (an emission stronger than 1 is not
  carried by `transfer_textures` or `base_color`/`emissive` bakes).
* The baker's `cage` is the averaged-normal offset or an explicit cage of the *same
  topology*; there is no cage editing tool.
* Painting is per object: a stroke does not continue onto another object.

## Examples

All images are rendered by Aether3D (`examples/output/`):

* [`texturing_hard_surface.py`](../examples/texturing_hard_surface.py) -- the power cell
  unwrapped (hard edges and face groups), parts at one texel density, a checker through
  the UVs, AO and curvature self-bakes driving grime and edge wear on base color,
  roughness and metallic, path traced.
* [`texturing_organic_paint.py`](../examples/texturing_organic_paint.py) -- the sculpted
  creature painted in 3D: symmetric spots, a stripe across UV seams, a view-space stroke
  that paints only what the camera sees, before and after.
* [`texturing_bake.py`](../examples/texturing_bake.py) -- a 240k-triangle sculpt baked onto
  a 10k-triangle remesh (normal, AO, curvature); the RMS difference to the high render
  drops with the maps.
* [`texturing_pbr.py`](../examples/texturing_pbr.py) -- material strokes on base color,
  roughness, metallic and emission, rendered by all four renderers and compared.
* [`mcp_texture_agent.py`](../examples/mcp_texture_agent.py) -- an agent over MCP: a refused
  script, a planar projection found overlapping by `inspect_uvs` and `uv_quality`,
  corrected by an unwrap, painted, verified.

## References

The research notes and design decisions are in
[`research/2026-10-stage14/DESIGN.md`](../research/2026-10-stage14/DESIGN.md). Main
sources: Levy, Petitjean, Ray, Maillot, *Least Squares Conformal Maps* (2002); Rabinovich,
Poranne, Panozzo, Sorkine-Hornung, *Scalable Locally Injective Mappings* (2017); Sander,
Snyder, Gortler, Hoppe, *Texture Mapping Progressive Meshes* (2001); Zhou, Snyder, Guo,
Shum, *Iso-charts* (2004); Mikkelsen, *Simulation of Wrinkled Surfaces Revisited* (2008)
and the reference MikkTSpace code; Meyer, Desbrun, Schröder, Barr, *Discrete
Differential-Geometry Operators* (2003); Woop, Benthin, Wald, *Watertight Ray/Triangle
Intersection* (2013); the W3C Compositing and Blending Level 1 formulas; the glTF 2.0
specification; Blender's baking and texture painting manuals; xatlas, libigl and
OpenImageIO for comparison.
