# Changelog

All notable changes to Aether3D are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may add
features; breaking changes are called out explicitly).

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
