# glTF 2.0 interoperability

Aether3D imports and exports **static geometry** in the
[Khronos glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
format, both as binary `.glb` and as JSON `.gltf`. The implementation lives in
`aether3d.gltf`, has no dependencies beyond NumPy, and is checked against the
official Khronos glTF Validator and an independent parser (trimesh).

```python
from aether3d import Scene, import_gltf

result = scene.export("world.glb")  # or "world.gltf"
scene = Scene.import_file("world.glb")  # just the scene

result = import_gltf("world.glb")  # scene + report
result.scene, result.objects, result.diagnostics
```

```console
$ aether3d export project.aether world.glb --json
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
| Mesh instancing (several nodes using one mesh) | yes | yes |
| Several primitives per mesh | n/a | merged into one mesh |
| Interleaved buffer views (`byteStride`), sparse accessors, accessors without buffer views | n/a | yes |
| Embedded (`data:` URI), GLB-binary, and external `.bin` buffers | GLB binary; `.gltf` with an embedded buffer | yes |
| Object ids | stored in `node.extras.aether3d.id` | restored from `extras` when present and unique; generated otherwise |
| Scene name | `scenes[0].name` | from the imported scene, else the file name |

Export writes one glTF mesh per distinct Aether3D mesh (one primitive,
`TRIANGLES`, 16-bit indices when there are at most 65 535 vertices, 32-bit
otherwise), one node per object in depth-first order, no materials, and a
single buffer. Output is deterministic and atomic (written to a temporary
file and renamed).

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
| materials, textures, images, samplers | `MATERIALS_IGNORED`, `TEXTURES_IGNORED`, `IMAGES_IGNORED`, `SAMPLERS_IGNORED` | info |
| cameras | `CAMERAS_IGNORED`, `NODE_CAMERA_IGNORED` | info |
| animations (nodes keep their rest transforms) | `ANIMATIONS_IGNORED` | warning |
| optional extensions (`extensionsUsed`) | `EXTENSION_IGNORED` | info |
| other vertex attributes (`COLOR_0`, `TANGENT`, `TEXCOORD_1`, ...) | `ATTRIBUTES_IGNORED` | info |
| scenes other than the default scene, and nodes outside it | `OTHER_SCENES_IGNORED`, `NODES_NOT_IN_SCENE` | warning |

Images are never read from disk.

**Content whose omission changes the geometry**: the import fails with
`UnsupportedFeatureError`, listing every such item, unless
`allow_lossy=True` (`--allow-lossy` on the CLI) is passed. Lossy diagnostics
have `"lossy": true`.

| Content | Diagnostic code | With `allow_lossy=True` |
| --- | --- | --- |
| skinned meshes | `SKIN_IGNORED` | imported unskinned (bind pose) |
| morph targets | `MORPH_TARGETS_IGNORED` | base geometry only |
| `POINTS` / `LINES` / `LINE_LOOP` / `LINE_STRIP` primitives | `PRIMITIVE_MODE_UNSUPPORTED` | skipped |
| node `matrix` with shear (not representable as TRS) | `NODE_MATRIX_NOT_TRS` | closest TRS transform |

**Always refused**: assets that list anything in `extensionsRequired`
(for example Draco or meshopt compression), and glTF versions other than 2.x.

Export has the same policy. Content glTF cannot store raises
`UnsupportedFeatureError` and writes nothing, unless `allow_lossy=True`:

| Content | Diagnostic code | With `allow_lossy=True` |
| --- | --- | --- |
| a mesh with no triangles | `EMPTY_MESH_NOT_EXPORTED` (lossy) | objects exported without geometry |
| zero-length normals (glTF requires unit normals) | `ZERO_NORMALS_NOT_EXPORTED` (lossy) | the mesh's normals are omitted |
| non-unit normals | `NORMALS_RENORMALIZED` (warning, not lossy) | always renormalized; direction is kept |

## Identifiers and names

- Every exported node carries `"extras": {"aether3d": {"id": "<object id>"}}`.
  On import, that id is reused when it is a non-empty string not already
  used. Otherwise a new UUID is generated and `STORED_ID_REJECTED` is
  reported.
- Nodes without that metadata get new UUIDs (`"id_source": "generated"` in
  `ImportResult.objects`).
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
- `tests/test_gltf_independent.py` reads Aether3D's exports with
  **trimesh**, an independent implementation, and compares world-space
  vertices, triangle winding, instancing and enclosed volumes.
- `tests/data/khronos/` holds a pinned selection of
  [Khronos sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets).
  They cover separate, embedded and binary buffers; interleaved, sparse and
  non-indexed data; every primitive mode; skins, morph targets and multiple
  scenes; and percent-encoded Unicode URIs.
  - Imports are compared with trimesh, except for triangle fans and sparse
    accessors, which trimesh does not handle. Those are checked against
    independent decoding and analytic values.
  - Static samples re-export and re-import without geometric change.
- `examples/gltf_roundtrip.py` is an end-to-end demonstration: it builds a
  scene, exports it, validates it, imports it back, and verifies it.

## Limitations

- Only static geometry: no materials, textures, cameras, lights, animation,
  skinning or morph targets (see the tables above).
- **Exported cameras and lights (since 0.5.0)** are written as plain nodes:
  - their transforms, names and ids are kept, and the files pass the Khronos
    validator;
  - glTF `camera` objects and `KHR_lights_punctual` lights are not written
    yet;
  - object colors are not exported, because there are no materials yet.

  An exported glTF camera would also need a 180° turn about the up axis.
  Aether3D cameras look along their local −Y; glTF cameras look along −Z in
  Y-up space, which is +Y in Aether3D's world (see
  [RENDERING.md](RENDERING.md)).
- Several primitives in one glTF mesh are merged into one Aether3D mesh.
  Per-primitive material assignment is not preserved.
- Normals or UVs present on only some primitives of a mesh are dropped for
  the whole mesh (`ATTRIBUTE_DROPPED`).
- No glTF extensions are supported, including `KHR_mesh_quantization`,
  `KHR_draco_mesh_compression`, `EXT_meshopt_compression`, and
  `KHR_texture_transform`.
- Export writes only `.glb`, or `.gltf` with an embedded base64 buffer.
  Separate `.bin` files are not written.
- Only the default scene of a multi-scene asset is imported.
