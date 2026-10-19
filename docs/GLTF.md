# glTF 2.0 interoperability

Aether3D imports and exports **static geometry and materials** in the
[Khronos glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
format, both as binary `.glb` and as JSON `.gltf`. Materials use glTF's
metallic-roughness model, with PNG/JPEG textures (since 0.7.0). The
implementation lives in `aether3d.gltf`. It needs nothing beyond NumPy (plus
Pillow for JPEG textures) and is checked against the official Khronos glTF
Validator and an independent parser (trimesh).

```python
from aether3d import Scene, import_gltf

result = scene.export("world.glb")  # or "world.gltf"
scene = Scene.import_file("world.glb")  # just the scene

result = import_gltf("world.glb")  # scene + report
result.scene, result.objects, result.diagnostics
```

```console
$ aether3d export project.aether world.glb --json
$ aether3d export project.aether world.glb --bake-textures 1024   # procedural -> PNG
$ aether3d import world.glb project.aether --json
$ aether3d inspect world.glb
$ aether3d geometry world.glb --json
```

## What is supported

| Feature | Export | Import |
| --- | --- | --- |
| Node hierarchy (parent/child order) | yes | yes |
| Node translation / rotation / scale, including non-uniform and negative scale | yes (TRS) | yes (TRS, or `matrix` when it is decomposable) |
| Indexed triangle meshes | yes | yes |
| Non-indexed triangles, triangle strips, triangle fans | n/a | yes (converted to indexed triangles) |
| `POSITION`, `NORMAL`, `TEXCOORD_0` | yes (float) | yes (float; normalized unsigned byte/short texcoords) |
| `COLOR_0` (0.18.0) | `Mesh.colors` as float `VEC4` (linear RGBA, multiplying the base colour, as in glTF) | float or normalized unsigned byte/short `VEC3`/`VEC4` (alpha 1 for `VEC3`); primitives without it are white when merged with ones that have it |
| Procedural worlds (0.18.0) | a finite region's generated geometry with `world.export_region` / `aether3d world export` ([WORLDS.md](WORLDS.md#history-files-and-export)); `scene.export` itself writes scene objects only | as plain meshes and instances |
| Mesh instancing (several nodes using one mesh) | yes | yes |
| Several primitives per mesh | one primitive per face group, each with its group's material | merged into one mesh; primitives with different materials become face groups |
| Materials (`pbrMetallicRoughness`, alpha modes, double-sided, normal, occlusion, emission) | yes | yes |
| PNG and JPEG textures, samplers | yes (original image bytes) | yes (decoded and embedded in the scene) |
| `KHR_texture_transform`, `KHR_materials_emissive_strength`, `KHR_materials_specular` | yes | yes |
| `TANGENT` | MikkTSpace, for meshes with UV-mapped normal maps (vertices split where corners differ) | ignored (MikkTSpace is recomputed) |
| Painted textures (0.14.0) | their composite, as an 8-bit PNG (layers stay in the project) | as image textures |
| CAD items shown in the scene (0.15.0) | their display tessellations, as meshes with their materials (exact geometry: export STEP or IFC, [CAD.md](CAD.md)) | as plain meshes |
| Object colors | as materials | restored exactly from `extras` |
| Skeletons (0.17.0) | rig joints as nodes under the armature's node; rigidly attached parts under their joints | skin joints (and nodes between them) become a rig on an armature object; nodes below joints become attached parts |
| Skins (0.17.0) | `JOINTS_n`/`WEIGHTS_n` (up to 8 influences), exact inverse bind matrices | bound with the file's inverse bind matrices, up to 8 influences |
| Morph targets (0.17.0) | `POSITION`/`NORMAL` targets, default `weights`, `extras.targetNames` | merged across primitives, named from `targetNames` |
| Animations (0.17.0) | clips, actions or baked motion: exact `STEP`/`LINEAR`/`CUBICSPLINE` where glTF can say the same thing, fitted within stated tolerances otherwise | clips with the same curves (the first animation plays as actions) |
| Interleaved buffer views (`byteStride`), sparse accessors, accessors without buffer views | n/a | yes |
| Embedded (`data:` URI), GLB-binary, and external `.bin` buffers | GLB binary; `.gltf` with an embedded buffer | yes |
| Object ids | stored in `node.extras.aether3d.id` | restored from `extras` when present and unique; generated otherwise |
| Object tags and metadata | `node.extras.aether3d.tags` / `.metadata` (when present) | restored from `extras`; invalid ones are reported (`STORED_ANNOTATIONS_REJECTED`) and skipped |
| Scene name | `scenes[0].name` | from the imported scene, else the file name |

Export writes:
- **Meshes.** One glTF mesh per distinct Aether3D mesh: `TRIANGLES`, 16-bit
  indices when there are at most 65 535 vertices, 32-bit otherwise. A mesh
  gets one primitive per face group. Objects that share a mesh but not its
  materials get separate glTF meshes over the same accessors.
- **Nodes.** One node per object, in depth-first order.
- **Materials.** A material for every object that has one, or a color other
  than the default gray, with its images and samplers.
- **Skeletons, skins, morph targets and animations** (0.17.0): see
  [below](#skeletons-skins-morph-targets-and-animations-0170).
- **One buffer.**

Output is deterministic and atomic (written to a temporary file and renamed).

## Coordinate systems

Both systems are right-handed. Aether3D is **+Z up**; glTF is **+Y up** with
the front of an asset facing +Z. Aether3D converts with a rotation of −90°
about X (the same convention as Blender), so a scene moves between Aether3D
and Blender through glTF without changing its coordinates:

```text
glTF (x, y, z)     = (x, z, -y)      for Aether3D (x, y, z)
Aether3D (x, y, z) = (x, -z, y)      for glTF (x, y, z)
```

The conversion matrix `C` is a proper rotation (determinant +1), so:

- **Positions and normals** are multiplied by `C`. Normals need no
  inverse-transpose because `C` is orthonormal.
- **Triangle winding is unchanged** because `C` does not mirror space.
  Counter-clockwise remains the front face.
- **Node transforms** become `C·M·Cᵀ`, which is again a TRS transform:
  - the translation is multiplied by `C`;
  - the quaternion's vector part is multiplied by `C`, and `w` is unchanged;
  - the scale axes are permuted, `(sx, sy, sz) → (sx, sz, sy)`.

  Non-uniform and negative scale therefore convert exactly.
- **Hierarchies stay consistent.** Because `C(M₁M₂)Cᵀ = (CM₁Cᵀ)(CM₂Cᵀ)`,
  converting each node also converts every world matrix.
- **Texture coordinates:** Aether3D uses a bottom-left UV origin (as Blender
  and OpenGL do); glTF uses top-left. `v` is flipped: `v' = 1 − v`.

`C` only permutes components and changes a sign, so positions, normals,
indices and transforms convert **bit-exactly**. Only UVs are affected by
rounding: `1 − v` is rounded to float32, so a round trip changes UVs by at
most 2.5 × 10⁻⁷ (two float32 roundings near 1.0). The functions live in
`aether3d.gltf.axes`.

## Unsupported content: reported, never hidden

glTF features that Aether3D does not model fall into two groups.

**Not geometry**: skipped, and listed in the import report:

| Content | Diagnostic code | Severity |
| --- | --- | --- |
| material features Aether3D does not draw or model (see [Materials and textures](#materials-and-textures)) | `ALPHA_BLEND_UNSUPPORTED`, `MATERIAL_EXTENSION_IGNORED`, `TEXCOORD_UNSUPPORTED`, ... | warning or info |
| materials no imported mesh uses | `MATERIALS_UNUSED` | info |
| cameras | `CAMERAS_IGNORED`, `NODE_CAMERA_IGNORED` | info |
| animation channels on content Aether3D does not import (`KHR_animation_pointer` paths, nodes outside the scene, skinned mesh nodes, weights without targets) | `ANIMATION_CHANNELS_IGNORED` | info |
| animated rotations that are not unit length (normalized) | `ROTATION_NOT_UNIT` | info |
| optional extensions (`extensionsUsed`) | `EXTENSION_IGNORED` | info |
| other vertex attributes (`TANGENT`, `TEXCOORD_1`, `COLOR_1`, ...) | `ATTRIBUTES_IGNORED` | info |
| scenes other than the default scene, and nodes outside it | `OTHER_SCENES_IGNORED`, `NODES_NOT_IN_SCENE` | warning |

Images are read only from buffer views, base64 `data:` URIs, or files in the
asset's own directory (with the same rules as external buffers). A missing or
broken image is reported (`TEXTURE_MISSING`, `TEXTURE_INVALID`); its materials
keep their factors.

**Content whose omission changes the geometry**: the import fails with
`UnsupportedFeatureError`, listing every such item, unless
`allow_lossy=True` (`--allow-lossy` on the CLI) is passed. Lossy diagnostics
have `"lossy": true`.

| Content | Diagnostic code | With `allow_lossy=True` |
| --- | --- | --- |
| a skin whose joints belong to more than one skeleton | `SKIN_SPANS_RIGS` | imported unskinned (bind pose) |
| a skinned mesh without `JOINTS_0`/`WEIGHTS_0` | `SKIN_WITHOUT_WEIGHTS` | imported unskinned |
| vertices with more than 8 influences | `INFLUENCES_TRIMMED` | the 8 strongest, renormalized |
| vertices with no weight | `UNWEIGHTED_VERTICES` | they follow the skin's first joint |
| `POINTS` / `LINES` / `LINE_LOOP` / `LINE_STRIP` primitives | `PRIMITIVE_MODE_UNSUPPORTED` | skipped |
| node `matrix` with shear (not representable as TRS) | `NODE_MATRIX_NOT_TRS` | closest TRS transform |

**Always refused**: assets that list anything in `extensionsRequired`
(for example Draco or meshopt compression), glTF versions other than 2.x, and
malformed skins and animations (`INVALID_SKIN`, `ANIMATION_INPUT_NON_INCREASING`,
`ANIMATION_OUTPUT_COUNT`, `ANIMATION_DUPLICATE_TARGET`, `INVALID_REFERENCE`).

Export has the same policy. Content glTF cannot store raises
`UnsupportedFeatureError` and writes nothing, unless `allow_lossy=True`:

| Content | Diagnostic code | With `allow_lossy=True` |
| --- | --- | --- |
| a mesh with no triangles | `EMPTY_MESH_NOT_EXPORTED` (lossy) | objects exported without geometry |
| zero-length normals (glTF requires unit normals) | `ZERO_NORMALS_NOT_EXPORTED` (lossy) | the mesh's normals are omitted |
| non-unit normals | `NORMALS_RENORMALIZED` (warning, not lossy) | always renormalized; direction is kept |
| animated properties glTF cannot carry (visibility, camera, light, material channels; constraint influences only when clips are exported alone -- baked, their effect is in the keys) | `ANIMATED_PROPERTY_NOT_EXPORTED` (lossy) | the channels are left out |
| skins deforming by dual quaternion or blended skinning (glTF skins are linear blend; 0.19.0) | `SKINNING_METHOD_LOSSY` (lossy) | exported with the same weights: they bend with less volume at twisting and bending joints |
| skins with corrective smoothing (delta mush; 0.19.1) | `SKIN_SMOOTHING_LOSSY` (lossy) | exported with the same weights: the folds and pinches it relaxes come back |
| skins shading with surface-following normals (0.19.1) | `SKIN_NORMALS_DIFFER` (info) | glTF viewers turn the rest normals with the joints; the geometry is the same |
| rig authoring data (0.19.0) | `RIG_AUTHORING_DATA` (info) | joint roles, helper flags, limits and (0.19.1) weighting spans ride as extras; control definitions, corrective drivers, landmarks and constraints are not exported (their effect is baked) |
| animation before time 0 (glTF times start at 0) | `NEGATIVE_TIMES_CLIPPED` (lossy) | the timeline before 0 is left out |
| layers, constraints, IK, paths or corrective shapes shape the motion but clips are exported alone | `ANIMATION_NOT_BAKED` (warning) | -- (`animation="bake"` or `"auto"` bakes them) |
| a channel needing more keys than the budget to meet its tolerance | `KEY_BUDGET_SPENT` (warning) | the reported error exceeds the tolerance |
| an instantaneous jump | `JUMP_APPROXIMATED` (info) | held as a 1e-4 s transition |
| procedural worlds (generated, not scene objects; 0.18.0) | `WORLDS_NOT_EXPORTED` (warning, not lossy) | -- (export a region with `world.export_region`) |

Material content glTF cannot express is reported as a warning and left out.
It is not lossy geometry, so the export does not fail. See [Materials and
textures](#materials-and-textures).

## Materials and textures

**Import.** Every glTF material used by imported nodes becomes a `"standard"`
Aether3D material ([MATERIALS.md](MATERIALS.md)):

| glTF | Aether3D |
| --- | --- |
| `baseColorFactor`, `baseColorTexture` | `base_color` (alpha → `opacity`), `base_color` texture (its alpha multiplies opacity) |
| `metallicFactor`, `roughnessFactor`, `metallicRoughnessTexture` | `metallic`, `roughness`; the texture binds to both slots (blue → metallic, green → roughness) |
| `normalTexture` (`scale`) | `normal` texture, `normal_strength` |
| `occlusionTexture` (`strength`) | `occlusion` texture (red), `occlusion_strength` |
| `emissiveFactor`, `emissiveTexture`, `KHR_materials_emissive_strength` | `emissive`, `emissive` texture, `emissive_strength` |
| `alphaMode` `OPAQUE` / `MASK`, `alphaCutoff` | `alpha_mode`, `alpha_cutoff` |
| `alphaMode` `BLEND` | opaque, with `ALPHA_BLEND_UNSUPPORTED` (Aether3D draws no blended transparency) |
| `doubleSided` | `double_sided` |
| `KHR_materials_specular` | `specular = 0.5 × specularFactor × mean(specularColorFactor)`, clamped to [0, 1] (`SPECULAR_COLOR_APPROXIMATED` for colored factors; specular textures: `SPECULAR_TEXTURE_IGNORED`) |
| sampler `wrapS`/`wrapT`, `magFilter`/`minFilter` | `wrap`, `filter`, `mipmaps` (`SAMPLER_APPROXIMATED` when the filters differ) |
| `KHR_texture_transform` | the binding's `offset`, `rotation`, `scale`, converted exactly to Aether3D's bottom-left uv origin |

- **Images.** PNG and JPEG images become embedded textures of the scene. They
  are saved inside the project; `texture.extract(path)` turns one into a file.
- **What is left out is reported, and the rest of the material is kept:**
  - textures that use `TEXCOORD_1` or another set (`TEXCOORD_UNSUPPORTED`);
  - images in other formats (`IMAGE_FORMAT_UNSUPPORTED`,
    `TEXTURE_FORMAT_UNSUPPORTED`);
  - other material extensions (`MATERIAL_EXTENSION_IGNORED`);
  - factors out of range, which are clamped (`FACTOR_CLAMPED`).
- **Exact parameters.** A material written by Aether3D carries its exact
  parameters in `extras.aether3d`. Simple materials and object colors come
  back exactly, and a damaged record is ignored (`STORED_MATERIAL_REJECTED`).
- **Primitives and face groups.** Primitives of one glTF mesh that use the
  same material are merged (`PRIMITIVES_MERGED`). With different materials
  they become face groups of one mesh, each with its material
  (`PRIMITIVES_AS_GROUPS`).

**Export.** Each object's material, per face group, becomes a glTF material;
an object without one exports its color.
- **Images** keep their original PNG/JPEG bytes and their sampler.
- **Bindings.** Binding transforms are written with `KHR_texture_transform`.
- **Emission.** Emission brighter than glTF's 0..1 `emissiveFactor`
  (`emissive × emissive_strength`) is written with
  `KHR_materials_emissive_strength`.
- **Specular.** Dielectric `specular` other than 0.5 is written with
  `KHR_materials_specular`.
- **Simple materials** are written as metallic-roughness, with roughness
  `(2 / (shininess + 2))^¼` from the Blinn-Phong exponent
  (`SIMPLE_MATERIAL_APPROXIMATED`, information).
- **Tangents.** Meshes drawn with a UV-mapped normal map get MikkTSpace
  `TANGENT` attributes -- the frame Aether3D's renderers and baker use
  ([TEXTURING.md](TEXTURING.md#tangent-space)). glTF stores one tangent per
  vertex: where the corners of one vertex have different tangents (UV seams,
  mirrored islands), the vertex is split (`VERTICES_SPLIT_FOR_TANGENTS`,
  information).
- **Channel packing.** glTF reads metallic from blue and roughness from green
  of one texture, occlusion from red. Image or painted textures bound through
  other channels (as baked maps are: `channel="r"`), or two separate metallic
  and roughness images, are repacked into one PNG with the channels where
  glTF reads them (`TEXTURE_CHANNELS_PACKED`, information; 0.14.0 -- they
  were left out before). The values survive exactly at 8 bits.

What glTF cannot express is left out with a warning:

| Content | Diagnostic code |
| --- | --- |
| procedural textures (unless baked, below) | `PROCEDURAL_TEXTURE_NOT_EXPORTED` |
| textures mapped by object or world coordinates | `TEXTURE_MAPPING_NOT_EXPORTED` |
| a separate opacity texture (glTF reads alpha from the base color texture) | `OPACITY_TEXTURE_NOT_EXPORTED` |
| metallic and roughness textures with different transforms (one packed texture cannot hold both), or an image and a procedural texture for them | `TEXTURE_CHANNEL_NOT_EXPORTED` |
| a texture whose color space is forced (glTF decodes each slot in its own space) | `TEXTURE_COLOR_SPACE_NOT_EXPORTED` |

`bake_textures=SIZE` (`scene.export`, `export_gltf`) or `--bake-textures SIZE`
(`aether3d export`), with SIZE from 16 to 8192, is an explicit, optional
output. It renders UV-mapped procedural textures into SIZE × SIZE PNG images
over the unit uv square. A metallic and a roughness texture are baked
together into glTF's packed texture. Each bake is reported (`TEXTURE_BAKED`).

## Skeletons, skins, morph targets and animations (0.17.0)

The full model is in [ANIMATION.md](ANIMATION.md#gltf-interchange). In short:

- **Export** (`scene.export(path, animation="auto", fps=None, position_tolerance=1e-4,
  rotation_tolerance=0.01, weight_tolerance=1e-4)`, `aether3d export --animation`): rig
  joints become nodes under the armature's node (rest TRS, converted like every node);
  attached parts become their joints' children; a skinned mesh rides its own root node
  with no transform (glTF ignores a skinned mesh node's transform, and the Khronos
  validator warns about one), marked `extras.aether3d.skinned_mesh_of`; inverse bind
  matrices are exact (`C M C^T`). Animation modes: `clips` (each clip with a target as
  its own glTF animation), `actions`, `bake` (what the scene shows on the timeline's
  frames as `LINEAR` keys reduced within the tolerances), `auto` (bake when layers,
  constraints, IK or paths shape the motion, else clips), `none`. `STEP`, `LINEAR`
  (rotations: shortest-path slerp; a deliberately long arc is split at its middle) and
  `CUBICSPLINE` (one-sided slopes as tangents of unweighted Bezier keys) are exact;
  everything else is fitted, with the largest error reported per animation
  (`ExportResult.animations`). Extras record each animation's actions and each channel's
  extrapolation, so a round trip keeps them.
- **Character rigs** (0.19.0, [CHARACTER_RIGGING.md](CHARACTER_RIGGING.md#gltf)): joint
  nodes carry `extras.aether3d` `role`, `deform` (`false` for helpers), `tail`, `limits`
  and (0.19.1) `span`, read back on import (each checked as the rig checks it; an invalid
  one is reported `JOINT_EXTRAS_IGNORED` and left out); every mesh of a character has its
  own skin, with up to eight influences per vertex (`JOINTS_0`/`WEIGHTS_0` and
  `JOINTS_1`/`WEIGHTS_1`); baking adds corrective shapes' weights to the animated morph
  weights.
- **Import**: skins, their joints and the nodes between them make a rig; `STEP` becomes
  constant keys, `LINEAR` linear keys or slerped rotations, `CUBICSPLINE` Bezier keys
  with the tangents as slopes (rotations: `cubic` tracks) -- the same curves. Morph
  weights animate `morph["NAME"]`.

## Identifiers and names

- Every exported node carries `"extras": {"aether3d": {"id": "<object id>"}}`.
  On import, that id is reused when it is a non-empty string not already
  used. Otherwise a new UUID is generated and `STORED_ID_REJECTED` is
  reported.
- Nodes without that metadata get new UUIDs (`"id_source": "generated"` in
  `ImportResult.objects`).
- Objects with tags or metadata (0.10.0) also write `"tags": [...]` and
  `"metadata": {...}` there; import restores them, and reports values that are
  not valid tags or metadata (`STORED_ANNOTATIONS_REJECTED`, a warning) instead
  of importing them.
- glTF names may repeat or be missing; Aether3D names must be unique.
  - An unnamed node takes its mesh's name, or else `"Node"`.
  - A duplicate name gets a suffix (`"Twin.001"`), reported as `NAME_CHANGED`.
- `ImportResult.objects` maps each glTF node index to the created object's id
  and name.

## Safety with untrusted files

- **Strict, located errors:** every structural check raises
  `AssetFormatError` with a stable `code` (e.g. `ACCESSOR_OUT_OF_BOUNDS`,
  `INDEX_OUT_OF_RANGE`, `UNSAFE_URI`) and a JSON-pointer `location`
  (e.g. `/accessors/3`).
- **External files are confined to the asset's own directory:**
  - only relative paths are accepted (percent-encoding is decoded first);
  - these are refused: absolute paths, Windows drives and UNC paths, `..`
    escapes, symlinks that resolve outside the directory, query strings,
    fragments, backslashes, NUL bytes, and every URI scheme except base64
    `data:`.
- **Size limits are checked before data is allocated.** `ImportLimits`
  bounds:
  - file and external-buffer size (512 MiB);
  - total buffer bytes (1 GiB);
  - node count (100 000);
  - total vertices and triangles (50 000 000 each).

  Exceeding a limit raises `ResourceLimitError`.
- **Malformed JSON is rejected:** invalid UTF-8, duplicate keys,
  `NaN`/`Infinity`, and excessive nesting.
- **Images are decoded safely:** with Aether3D's own PNG decoder (or Pillow for
  JPEG), within `ImageLimits` (16384 pixels per side, 8192² in total), and
  with every chunk, CRC and the compressed stream checked. A broken image is
  reported, never fatal (see
  [MATERIALS.md](MATERIALS.md#untrusted-textures-and-graphs)).
- **No partial results:** a failed import has no side effects, and a
  failed export leaves any existing file untouched.

## Verification

- `tests/test_khronos_validator.py` exports a corpus with the official
  **Khronos glTF Validator 2.0.0-dev.3.10**, and requires zero errors and
  zero warnings. The corpus covers:
  - nested, rotated, mirrored and non-uniformly scaled objects;
  - instancing, an empty scene and empty nodes;
  - meshes without normals or UVs;
  - every Khronos sample re-exported by Aether3D.

  CI runs it on Linux and Windows. Locally:
  `npm ci --prefix tools/gltf-validator`, then
  `node tools/gltf-validator/validate.mjs FILE...`.
- `tests/test_material_gltf_validator.py` runs the same validator on exports
  with materials, textures, face groups, texture transforms, emission,
  specular, normal maps (with `TANGENT`) and baked textures. They also need
  zero errors and zero warnings. `tests/test_material_gltf.py` checks the
  material mappings in both directions, round trips, texture orientation and
  the diagnostics.
- `tests/test_gltf_independent.py` reads Aether3D's exports with
  **trimesh**, an independent implementation, and compares world-space
  vertices, triangle winding, instancing and enclosed volumes.
- `tests/data/khronos/` holds a pinned selection of
  [Khronos sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets).
  They cover separate, embedded and binary buffers; interleaved, sparse and
  non-indexed data; every primitive mode; skins, morph targets and multiple
  scenes; percent-encoded Unicode URIs; and texture coordinates, samplers and
  texture settings (`TextureCoordinateTest`, `TextureSettingsTest`).
  - Imports are compared with trimesh, except for triangle fans and sparse
    accessors, which trimesh does not handle. Those are checked against
    independent decoding and analytic values.
  - Static samples re-export and re-import without geometric change.
- `examples/gltf_roundtrip.py` is an end-to-end demonstration: it builds a
  scene, exports it, validates it, imports it back, and verifies it.
- `tests/test_animation_gltf.py` (0.17.0) checks skeletons, skins, morph targets and
  animations: exports pass the Khronos glTF Validator with no errors and no warnings
  (required in CI); `STEP`, `LINEAR` and Hermite curves export exactly (to float32),
  slerped rotations as `LINEAR` with long arcs split, easings and Euler rotations
  fitted within their tolerances; a rigged, skinned and morphed scene round-trips with
  world matrices and skinned vertices within 2e-4 scene units at every sampled time,
  its loops kept; `SimpleSkin` deforms as an independent NumPy evaluation of the file
  says (within 2e-6) and `AnimatedMorphCube` blends its targets as the file says
  (within 1e-6); malformed skins and animations are refused with their codes.

## Limitations

- Animations carry node transforms and morph weights only: no
  `KHR_animation_pointer` (camera, light, material and visibility channels are lossy);
  values are float32; jumps become 1e-4 s transitions; easings, weighted handles, Euler
  rotations, squad and modifiers are fitted within the stated tolerances (see the
  tables above).
- **Materials** are glTF's core metallic-roughness model plus three
  extensions. Other material extensions (clearcoat, transmission, sheen,
  volume, IOR...) are not imported. Procedural textures exist in glTF only
  when baked. Only `TEXCOORD_0` is used.
- **Exported cameras and lights (since 0.5.0)** are written as plain nodes:
  - their transforms, names and ids are kept, and the files pass the Khronos
    validator;
  - glTF `camera` objects and `KHR_lights_punctual` lights are not written
    yet.

  An exported glTF camera would also need a 180° turn about the up axis.
  Aether3D cameras look along their local −Y; glTF cameras look along −Z in
  Y-up space, which is +Y in Aether3D's world (see
  [RENDERING.md](RENDERING.md)).
- Several primitives in one glTF mesh are merged into one Aether3D mesh.
  Their materials are kept as face groups (since 0.7.0).
- Normals or UVs present on only some primitives of a mesh are dropped for
  the whole mesh (`ATTRIBUTE_DROPPED`).
- Apart from `KHR_texture_transform`, `KHR_materials_emissive_strength` and
  `KHR_materials_specular`, no glTF extensions are supported, including
  `KHR_mesh_quantization`, `KHR_draco_mesh_compression` and
  `EXT_meshopt_compression`.
- Export writes only `.glb`, or `.gltf` with an embedded base64 buffer.
  Separate `.bin` files are not written.
- Only the default scene of a multi-scene asset is imported.
