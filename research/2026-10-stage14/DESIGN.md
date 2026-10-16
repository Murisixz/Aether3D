# Stage 14 design record: UV mapping, texture painting and baking

Written on 16-17 October 2026, before any Stage 14 implementation code, against `main` at
6606aa5 (v0.13.0, Stages 1-13). It records what was inspected, the research behind each
algorithm, and the implementation-language decision for every subsystem. The user guide
that describes what was finally built is [docs/TEXTURING.md](../../docs/TEXTURING.md);
where the two differ, this file is the plan as decided and the guide is the truth about
the code. Section 9 (written after the implementation) records what changed.

## 1. Starting point (inspected)

| Area | Where | What exists | Gap for Stage 14 |
| --- | --- | --- | --- |
| Geometry | `mesh.py` | immutable `Mesh`: float32 positions, per-vertex normals and UVs (seams are split vertices), uint16 face groups | no tangents, no seam marks, no unwrapping |
| Editing | `editing/` | per-corner `uv` / `normal` layers, sharp edges, face groups; affine-fit UV interpolation | no seam marks, no UV operators |
| Sculpting | `sculpt/` | UVs kept through strokes and dyntopo (seams are features), dropped by voxel remeshing (`VOXEL_UVS_DROPPED`) | no re-unwrap or texture transfer after topology changes |
| Images | `materials/images.py`, `rendering/png.py` | own PNG decoder (all depths, 16-bit rounded to 8), Pillow JPEG, 8-bit PNG encoder; strict limits (16384 px/side, 8192^2 px) | no 16-bit output, no authoring storage |
| Textures | `materials/textures.py`, `sampling.py` | `ImageTexture` (file/embedded, sampler settings), `ProceduralTexture`; process-wide cache keyed by SHA-256; 8-bit RGBA mip chains built per decoding mode (`srgb`, `linear`, `normal`) | a texture is a file: an edit means a new file, a new digest, a new decode and a new mip chain |
| CPU shading | `rendering/surface.py` | slots, channels, color spaces, projections; normal maps with a **per-triangle UV-gradient frame** | not MikkTSpace: maps baked by MikkTSpace tools (and by Aether3D) shade with faceting on smooth low-poly meshes |
| GPU raster | `rendering/gpu/` | texel pool: every mip of every image in one RGBA8 storage buffer, keyed by digest, re-uploaded whole when the image set changes | painting would re-upload the pool every stroke |
| Path tracers | `rendering/pathtrace`, `gpu/pathtrace.py` | the CPU shading code; WGSL two-level BVH traversal at ~228 M camera/AO rays/s on the RTX 4060 | no bake kernels |
| Ray queries | `rendering/raytrace/accel.py` | `SceneAccel` (two-level BVH, watertight test), NumPy 0.1-0.3 M rays/s, native 0.8-3 M rays/s per thread, 4-12 M with 8 threads | |
| glTF | `gltf/` | PBR materials, V flipped on import/export, TANGENT exported from averaged UV gradients, TANGENT ignored on import, procedural textures baked over the unit square on request | no MikkTSpace, no authored textures |
| History | `history.py` | aspects with before/after states; immutable values shared by reference; 256 MiB budget; history files | a texture state is a file reference |
| Project format | `serialization.py` | JSON, format 6, strict readers, migrations | nowhere to keep layers |
| Native | `native/` (Rust, PyO3, abi3-py311, ABI 1) | traversal, hierarchies, PNG filters, **bitwise equal to their NumPy twins** | optional: every kernel needs its NumPy twin |
| AI interfaces | `mcp/`, `feedback/`, CLI | 21 MCP tools; `run_script` dispatches script formats; expectations with geometry/render basis | no UV, paint or bake operations |

## 2. Research (sources read for this stage)

**Tangent space.** glTF 2.0 (§3.7.2.1, §5.19 `normalTextureInfo`): TANGENT is XYZW with
`bitangent = cross(normal, tangent.xyz) * w`; without TANGENT, clients SHOULD compute
MikkTSpace tangents; normal maps are +X right, +Y up. MikkTSpace (Mikkelsen 2008, thesis;
reference `mikktspace.c`, zlib license, read in full): corners are welded by exact
equality of position, normal and texcoord; per-triangle first-order derivatives
(eq. 18-19) normalized with the orientation sign; groups are fans around a welded vertex
connected through shared edges with the same orientation; group tangents are the
angle-weighted sum of the projected per-triangle tangents, accumulated in ascending
triangle order, in float32; degenerate triangles copy from good ones. The header states
the exact pixel-shader convention (`n = normalize(x T + y B + z N)` with interpolated,
unnormalized T and N) and the baker's exact inverse (rows `cross(B, N)`, `cross(N, T)`,
`cross(T, B)`). Because Aether3D's UVs are V-up like Blender's, MikkTSpace computed on
Aether3D UVs gives glTF's tangent and sign directly (three.js negates w only because it
computes on glTF's V-down UVs). Mirrored instances (det < 0) flip w.

**Charts and parameterization.** Lévy et al. 2002 (LSCM, with a "Tetris" packer);
Sheffer et al. 2005 (ABF++); Liu et al. 2008 (ARAP local/global); Rabinovich et al. 2017
(SLIM: reweighted local/global minimization of the symmetric Dirichlet energy, flip-free
line search); Smith & Schaefer 2015 (bijective parameterization with free boundaries:
max step before a triangle inverts); Sander et al. 2001 (L2 and L-infinity texture
stretch); Zhou et al. 2004 (Iso-charts: split charts until a stretch bound holds);
Julius et al. 2005 (D-charts); xatlas (MIT, read: region-growing segmentation with
normal-deviation, roundness, straightness and seam costs, LSCM with fallback to
piecewise charts, bitmap packing); Blender's unwrap (ABF++, LSCM and SLIM
"minimum stretch"); libigl's SLIM (MPL-2.0, read for the algorithm, not used).
Discrete Tutte embeddings (Floater 1997) are injective on convex boundaries for
3-connected disks.

**Sparse solvers.** Davis, *Direct Methods for Sparse Linear Systems* (2006) and
Algorithm 849 (LDL, 2005): elimination tree, row patterns by tree reach, up-looking
LDL^T. George 1973 / Lipton-Rose-Tarjan 1979: nested dissection gives O(n log n) fill
on planar graphs; geometric bisection is a simple separator for embedded meshes.

**Painting.** Blender's projection paint (screen-space projection into UVs) and the
texel-space approach of Mari/Substance (a per-texel surface position map, brushes
evaluated in 3D, so strokes cross UV seams continuously). Porter & Duff 1984
(compositing), the W3C Compositing and Blending Level 1 blend formulas.

**Baking.** Blender's baking manual (cage or extrusion, max ray distance, margin),
xNormal and Marmoset docs (front/back ray distances, averaged-normal cages),
ambient occlusion as the cosine-weighted visibility integral within a radius (Zhukov
et al. 1998; Landis 2002), Meyer et al. 2003 (cotangent mean curvature and Voronoi
areas), Pharr et al. *PBRT* 4th ed. (cosine hemisphere sampling, watertight
intersection, ray offsets).

**Formats.** PNG 3rd edition (16-bit samples, `sRGB` chunk), KHR_texture_transform,
glTF's ORM packing (occlusion R, roughness G, metallic B).

## 3. Language and architecture decision per subsystem

The rule, as in Stage 13: choose per job; native code where a measured or analysed loop
dominates, GPU where the work is massively parallel and transfers are small; and keep
the package contract: `aether3d-native` is optional, so every native kernel has a NumPy
twin with the same arithmetic in the same order (bitwise equal results, tested), and the
GPU is optional, so every GPU path has a CPU path whose results agree within a stated
tolerance (as the rasterizers and path tracers do).

| Subsystem | Work shape | Decision | Why |
| --- | --- | --- | --- |
| Seams, welded topology, chart segmentation | edge/face graphs of 10k-1M faces; union-find, BFS frontiers | **NumPy** (vectorized components and frontier BFS) | All passes are whole-graph array operations; no per-face Python loop. |
| Parameterization (LSCM, SLIM) | sparse SPD solves per chart; SLIM refactors every iteration | **Rust: sparse LDL^T (nested-dissection order, elimination tree, up-looking factorization, solves); NumPy twin with the same operations** | A direct solver is the established engine (Blender, libigl, CGAL use Eigen's): exact, iteration-free, and SLIM's constant sparsity reuses the symbolic analysis. Up-looking LDL^T is a sequential loop over nonzeros: ~100x faster native than in Python; NumPy (no SciPy) has no sparse solver at all. Local steps (per-triangle 2x2 SVDs, energies, line search) are NumPy. |
| Atlas packing | per chart, a horizon search over the atlas width | **Rust kernel for the placement search; NumPy twin** | Integer loops of O(width x chart width) per chart and orientation; decisions are integers, so the twin is exactly equal. |
| UV / texel rasterization | spans of every triangle over a texel grid | **NumPy**, reusing the CPU rasterizer's fixed-point, top-left span arithmetic | Already vectorized and exact; each texel is written once. |
| MikkTSpace | per-triangle float32 math, grouping | **NumPy, an exact port** (vectorized; only the order-dependent fans of degenerate-UV triangles are walked in Python) | Validated bit for bit against the reference C (built in the scratchpad, not shipped). |
| Texture documents (tiles, layers, compositing, mips) | per tile 64x64 texels | **NumPy** | Compositing a dirty tile is a few vector operations; the work is proportional to what changed. |
| Brush kernels (paint, blur, smudge, clone, mask) | per dab: texels of the tiles the brush overlaps | **NumPy** | Memory-bound gathers and blends over 10^3-10^6 texels; the document, history and serialization are on the CPU, so a GPU brush would need a GPU copy of the authoritative layers. |
| GPU texel updates while painting | dirty rows of each mip level | **wgpu `write_buffer` of the changed rows** | Avoids re-uploading the whole pool after every stroke. |
| Ray-based baking: closest hits (normals, height, IDs, colors) and occlusion (AO, thickness) | 10^6-10^9 rays | **GPU (WGSL compute, the path tracer's traversal) when available; CPU `SceneAccel` (native traversal with threads, or NumPy) otherwise** | AO is ~10^9 rays at 4K: ~228 M rays/s on the GPU against 4-12 M on 8 CPU threads. Shading of hits (tangent frames, attribute interpolation, material sampling) stays in NumPy, shared by both paths. |
| Orchestration, API, scripts, CLI, MCP | control | **Python** | Declarative JSON scripts, never code. |

No second implementation of any feature is written for comparison. External libraries
(xatlas, libigl, OpenImageIO, CGAL) are not added: xatlas and libigl have no maintained
wheels for every supported Python, CGAL's parameterization package is GPL, and
OpenImageIO is a large binary dependency for what PNG covers.

## 4. Algorithms

**Seams.** A seam is an edge the UV map may cut. The unwrapper cuts at: marked seams
(`EditableMesh` edge flag `seam`, or edge lists given to the API), existing UV
discontinuities when `keep_seams=True`, boundaries and non-manifold edges, hard edges and
face-group or material borders when asked (`split_by`). Seams are kept in the result.

**Charts.** Faces are joined across non-seam edges whose dihedral angle is below
`angle_limit` (union-find on the dual graph): planar hard-surface regions become one
chart each. A chart that is not a disk (Euler characteristic, boundary count) or whose
parameterization fails the quality bound is split in two by growing two regions from its
two farthest faces (BFS on the dual graph, ties by face id) -- recursive, deterministic,
Iso-charts style. Every chart that survives is a disk with bounded stretch.

**Parameterization.** LSCM with two pinned vertices (the farthest boundary pair) gives
a conformal initial map; if it has inverted triangles, a Tutte embedding on a circle
does (injective on a convex boundary). SLIM then minimizes the symmetric Dirichlet
energy (isometric distortion) with free boundaries: per iteration, closed-form 2x2 SVDs,
the reweighted quadratic proxy solved with LDL^T (same sparsity every iteration: one
symbolic analysis), and a line search capped at the largest flip-free step. The result
has no inverted triangle. Boundary self-overlaps are detected exactly (segment
intersection) and resolved by splitting. `method="conformal"` stops after LSCM.

**Projections.** Planar, box (per-face axis by normal), cylindrical and spherical, with
explicit axes and seam handling (a face crossing the wrap is unwrapped continuously).

**Packing.** Charts are scaled by one texel density (or individually normalized),
rotated to their minimum-area rectangle, rasterized at texel resolution with a padding
dilation, and placed one by one, largest first, by Lévy's horizon ("Tetris") rule: for
each orientation and column, the chart rests on the horizon; the placement that wastes
the least area below the chart wins (ties: lowest, then leftmost). A fixed-size atlas is
filled by a binary search on the density. Overlapping (stacked) islands are allowed only
when asked; mirrored islands are reported as such, not as collisions.

**Diagnostics.** Per chart and overall: Sander's L2 and L-infinity stretch, area and
angle distortion, inverted and degenerate triangles, overlaps (exact), texel density
spread, utilization, island count, padding check.

**Tangents.** The exact MikkTSpace port, per corner. Renderers (CPU, GPU rasterizer, both
path tracers) use them for UV normal maps: `B = sign * cross(N, T)` per pixel with
interpolated, unnormalized N and T, `n = normalize(x T + y B + z N)`; mirrored instances
negate the sign. The baker uses the exact inverse. Imported TANGENT attributes are used
as given.

**Texture documents.** An `ImageTexture` whose pixels are authored: width x height,
channels RGBA, stored as 64x64 tiles of uint16 (lossless for 8- and 16-bit PNG; enough
precision for low-opacity accumulation) per layer, sparse (missing tiles are
transparent). Layers: paint or fill, with opacity, blend mode (normal, multiply, add,
subtract, screen, overlay, darken, lighten, difference), visibility and an optional mask.
Tiles are immutable arrays: an edit replaces tiles, so a history state is a tuple of
tile references (memory = changed tiles only). The flattened result is cached per tile;
renders get 8-bit levels updated only where tiles changed, and the GPU pool receives
only the changed rows.

**Painting.** A texel position map (object-space position, normal and triangle of every
texel center covered by the UVs, from the exact UV rasterization) is built once per
(mesh, UV layout, resolution). A 3D dab affects the texels within its radius of the dab
center (tiles culled by their 3D bounds), so strokes continue across UV seams; a view
dab affects texels whose projection falls in the screen disc and that are visible in the
view's depth buffer (other objects occlude, as in Blender's projection paint). Brushes:
paint, erase, blur, smudge, clone, fill, gradient, mask; falloff, hardness, spacing,
flow, pressure, symmetry. A stroke is one history step, cancellable, rolled back on
failure. After a stroke, padding texels near the change are re-dilated.

**Baking.** Low-poly texels are rasterized exactly (supersampled at stratified
positions); each sample casts a ray from the cage (low surface pushed out along the
averaged normals by `cage_offset`, or an explicit cage mesh) back along the normal, with
front and back distances; the first high-poly hit (or the nearest of the two directions)
gives the transferred values. Targets: tangent-space normals (MikkTSpace, exact inverse
of the renderer), object/world normals, height (signed distance along the ray), AO
(cosine-weighted hits within `distance`, from the high-poly hit point), thickness
(inward hemisphere), curvature (Meyer's mean curvature interpolated from the high
mesh), material/object/group IDs, and material channels sampled from the high-poly's
own materials. Misses are reported (count, fraction) and filled by dilation only when
asked. Seam-aware padding: iterative dilation into uncovered texels. Results go into
texture documents (one undo step) or files.

## 5. Robustness, safety and transactions

Validation before mutation everywhere (structured `AetherError` subclasses with stable
codes): finite coordinates, sizes within `ImageLimits`, ray counts, sample counts, stroke
lengths, iteration and time budgets, memory estimates refused before allocation. UV
operations on a `SceneObject` are one history step; failures change nothing. Scripts are
data (JSON Schema, fuzzed against the executor); MCP keeps workspace confinement; files
are written atomically, layers in content-addressed files so a failed save never
references a missing layer.

## 6. Test strategy

Analytic references: a plane's LSCM is its own isometry up to similarity; a cylinder cut
along a line flattens to a rectangle with zero stretch; a sphere cap has the known
conformal map; Tutte maps are injective; packing never overlaps (rasterized check) and
honors padding; MikkTSpace against the reference C (fixtures); renders of baked normal
maps against the high-poly's own normals; AO of a plane under a known occluder and of a
cavity against closed-form values; curvature of spheres and cylinders; texture document
undo/redo bit for bit; strokes across a seam continuous in 3D; GPU vs CPU bakes within
float32 tolerance; NATIVE=0 vs native bitwise; hostile inputs (non-finite UVs, huge
sizes, corrupted layer files, unsafe paths).

## 7. Benchmark plan

`benchmarks/bench_texturing.py`: unwrap (segmentation, parameterization, packing) at
10k/100k/1M faces; texture documents 1K/2K/4K/8K (create, stroke latency, composite,
mips, GPU update); baking normal and AO at 1K-4K from 100k and 1M-face sources, CPU and
GPU; MikkTSpace at 10k-1M; peak memory per group in its own process. Baselines: the
existing image path (decode, mip chain, pool upload) and `SceneAccel` ray rates.

## 9. Outcome

(written after the implementation)
