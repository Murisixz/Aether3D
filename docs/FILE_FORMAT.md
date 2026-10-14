# The `.aether` project format

**Current version: 4** (Aether3D 0.7.0; version 1–3 files are upgraded on load)

An `.aether` file stores one scene:

- its objects, with their hierarchy and transforms;
- their mesh geometry;
- its procedural geometry definitions: parameters, rules and geometry nodes
  with modifier stacks (see [PROCEDURAL.md](PROCEDURAL.md));
- its cameras, lights, object colors, background, ambient light and render
  settings (see [RENDERING.md](RENDERING.md));
- its materials and textures, and which objects and face groups use them
  (see [MATERIALS.md](MATERIALS.md)). Image textures are stored as paths
  relative to the project, or embedded.

It is a UTF-8 encoded JSON document (RFC 8259), designed to be readable,
diffable, and safely editable by people, scripts, and AI agents.

## Example

```json
{
  "format": "aether3d.project",
  "format_version": 4,
  "generator": "aether3d 0.7.0",
  "scene": {
    "name": "First Scene",
    "active_camera": "5b0f1c2e-…",
    "world": {"background": [0.2, 0.2, 0.2], "ambient_color": [1.0, 1.0, 1.0], "ambient_intensity": 0.1},
    "render": {"width": 1280, "height": 720, "samples": 2, "exposure": 0.0, "cull": "none", "default_lights": true},
    "parameters": [],
    "rules": [],
    "geometry": [],
    "materials": [
      {
        "id": "e56f2bec-…",
        "name": "Rock",
        "model": "standard",
        "base_color": [0.75, 0.73, 0.70], "opacity": 1.0, "alpha_mode": "opaque",
        "alpha_cutoff": 0.5, "metallic": 0.0, "roughness": 0.8, "specular": 0.5,
        "emissive": [0.0, 0.0, 0.0], "emissive_strength": 1.0, "normal_strength": 1.0,
        "occlusion_strength": 1.0, "double_sided": true,
        "textures": {
          "base_color": {"texture": "209d2d9b-…", "scale": [2.0, 2.0, 1.0]},
          "roughness": {"texture": "5370919d-…", "coordinates": "object"}
        }
      }
    ],
    "textures": [
      {"id": "209d2d9b-…", "name": "Rock color", "type": "image", "path": "textures/rock.png",
       "wrap": ["repeat", "repeat"], "filter": "linear", "mipmaps": true, "color_space": "auto"},
      {"id": "5370919d-…", "name": "Rock roughness", "type": "procedural",
       "graph": {"nodes": {"out": {"type": "fbm", "scale": 6.0, "octaves": 4}}, "output": "out"}}
    ],
    "objects": [
      {
        "id": "c1181286-0e18-46b7-991d-99bac53d479b",
        "name": "Pivot",
        "parent": null,
        "transform": {
          "position": [3.0, 0.0, 1.0],
          "rotation": [0.0, 0.0, 0.7071067811865475, 0.7071067811865476],
          "scale": [1.0, 1.0, 1.0]
        },
        "mesh": null,
        "geometry": null,
        "camera": null,
        "light": null,
        "appearance": null,
        "visible": true,
        "material": null,
        "group_materials": {}
      },
      {
        "id": "8122a0f9-511d-4611-88f4-b44f02b96274",
        "name": "Moon",
        "parent": "c1181286-0e18-46b7-991d-99bac53d479b",
        "transform": {
          "position": [2.0, 0.0, 0.0],
          "rotation": [0.0, 0.0, 0.0, 1.0],
          "scale": [1.0, 1.0, 1.0]
        },
        "mesh": 0,
        "geometry": null,
        "camera": null,
        "light": null,
        "appearance": {"color": [0.75, 0.22, 0.17], "specular": 0.4, "shininess": 64.0},
        "visible": true,
        "material": "e56f2bec-…",
        "group_materials": {}
      },
      {
        "id": "5b0f1c2e-…",
        "name": "Camera",
        "parent": null,
        "transform": {"position": [8.0, -12.0, 6.0], "rotation": […], "scale": [1.0, 1.0, 1.0]},
        "mesh": null,
        "geometry": null,
        "camera": {"projection": "perspective", "fov": 40.0, "fov_axis": "vertical",
                   "ortho_scale": 10.0, "near": 0.1, "far": 1000.0, "aspect_ratio": null},
        "light": null,
        "appearance": null,
        "visible": true,
        "material": null,
        "group_materials": {}
      }
    ]
  },
  "meshes": [
    {
      "positions": [[0.0, 0.0, 0.25], ...],
      "normals": [[0.0, 0.0, 1.0], ...],
      "uvs": [[0.0, 1.0], ...],
      "triangles": [[0, 33, 34], ...]
    }
  ]
}
```

## Top level

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `format` | string | yes | Always `"aether3d.project"`. Identifies the file type. |
| `format_version` | integer ≥ 1 | yes | Version of this specification the file follows. |
| `generator` | string | no | Free-form name/version of the writing software. Ignored on read. |
| `scene` | object | yes | The scene (see below). |
| `meshes` | array | yes | Mesh table, referenced by index from objects. May be empty. |

## `scene`

| Key | Type | Description |
| --- | --- | --- |
| `name` | non-empty string | Scene name. |
| `active_camera` | string or `null` | `id` of the camera used for rendering by default. It must be a camera object. Since version 3. |
| `world` | object | Background and ambient light (see [Rendering](#rendering)). Since version 3. |
| `render` | object | Saved render settings (see [Rendering](#rendering)). Since version 3. |
| `parameters` | array | Scene parameters (see [Procedural geometry](#procedural-geometry)). Since version 2. |
| `rules` | array | Scene rules. Since version 2. |
| `geometry` | array | Procedural geometry nodes. Since version 2. |
| `materials` | array | Materials (see [Materials and textures](#materials-and-textures)). Since version 4. |
| `textures` | array | Image and procedural textures. Since version 4. |
| `objects` | array of objects | All scene objects (see below). |

Writers emit objects in depth-first hierarchy order (each parent before its
children). Readers accept any order. The relative order of objects that share
a parent (or that are all roots) defines their child order.

## Scene object

| Key | Type | Description |
| --- | --- | --- |
| `id` | non-empty string | Stable unique identifier. Aether3D generates UUID4 strings, but any unique string is accepted. Unique within the file. |
| `name` | non-empty string | Human-readable name. Unique within the file. |
| `parent` | string or `null` | `id` of the parent object, or `null` for a root object. Must not form a cycle. |
| `transform` | object | Local transform relative to the parent (see below). |
| `mesh` | integer or `null` | Index into the top-level `meshes` array, or `null` for an object without a static mesh. Several objects may share one mesh. |
| `geometry` | string or `null` | `id` of a procedural geometry node, or `null`. An object has a `mesh` or a `geometry`, never both. Its mesh is the node's evaluated result, which is never stored. Since version 2. |
| `camera` | object or `null` | The object is a camera (see [Rendering](#rendering)). Since version 3. |
| `light` | object or `null` | The object is a light. Since version 3. |
| `appearance` | object or `null` | Color and highlight; `null` means the defaults. Since version 3. |
| `visible` | boolean | Whether the object renders (lights: whether they shine). Since version 3. |
| `material` | string or `null` | `id` of the material the object renders with; `null`: its `appearance`. Since version 4. |
| `group_materials` | object | Materials of individual face groups: `{"1": material id, ...}` (keys are group numbers 0–65535 written as strings). Groups without an entry use `material`. Since version 4. |

An object holds at most one of `mesh`, `geometry`, `camera` and `light`.
Cameras and lights cannot have materials. Writers always emit `camera`,
`light`, `appearance`, `visible`, `material` and `group_materials`. Readers
treat them as optional, with the defaults `null`, `null`, `null`, `true`,
`null` and `{}`. An object keeps its `appearance` while it uses a material;
it renders with it again when the material is unassigned.

### `transform`

| Key | Type | Description |
| --- | --- | --- |
| `position` | `[x, y, z]` | Translation. |
| `rotation` | `[x, y, z, w]` | Rotation quaternion (glTF component order). Normalized on load; must be non-zero. |
| `scale` | `[x, y, z]` | Per-axis scale. |

The local matrix is `T @ R @ S` (column vectors). An object's world matrix is
`parent.world @ local`. All numbers must be finite.

## Procedural geometry

Since version 2. Values anywhere below may be literals or expressions,
written `{"expr": "floor(width / 2)"}`. Vectors are arrays whose items may
themselves be expressions.

### Parameter

```json
{"name": "width", "type": "float", "value": 24.0, "min": 12, "max": 80,
 "unit": "m", "description": "outer size of the curtain wall"}
```

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | identifier | yes | Unique in its scope. |
| `type` | string | yes | `float`, `int`, `bool`, `vec3`, `enum` or `string`. |
| `value` | value or expression | yes | The resolved value must satisfy the type and constraints. |
| `min`, `max` | number | no | Bounds for numbers. |
| `choices` | array of strings | enums | Allowed values. |
| `unit`, `description` | string | no | Documentation for people and agents. |

### Rule

`{"name": "even_towers", "expr": "tower_count % 2 == 0", "message": "..."}`:
a boolean expression that must hold. `message` is optional.

### Geometry node

```json
{
  "id": "4c1d…", "name": "Tower",
  "parameters": [{"name": "height", "type": "float", "value": {"expr": "wall_height * 1.4"}}],
  "rules": [],
  "source": {"type": "primitive", "primitive": "cylinder",
             "params": {"radius": {"expr": "tower_radius"}, "depth": {"expr": "height"}, "segments": 8}},
  "modifiers": [
    {"id": "9b2e…", "name": "Arrow Slits", "type": "boolean", "enabled": true,
     "params": {"operation": "difference", "operand": {"geometry": "77aa…"}}}
  ]
}
```

| Key | Description |
| --- | --- |
| `id`, `name` | Unique within the file. |
| `parameters`, `rules` | Node-local parameters and rules (same layout as the scene's). |
| `source` | One of: `{"type": "primitive", "primitive", "params"}`; `{"type": "mesh", "mesh": index into meshes}`; `{"type": "geometry", "geometry": node id, "parameters": overrides, "after": modifier id (optional)}`. |
| `modifiers` | Ordered stack: `id` (unique), `name` (unique in the stack), `type`, `enabled`, `params`. |

Modifier and primitive parameters follow the schemas in
[PROCEDURAL.md](PROCEDURAL.md). Boolean operands reference geometry by id.

## Rendering

Since version 3. Colors are display (sRGB-encoded) values: arrays of three
numbers in [0, 1].

**`camera`**

| Key | Type | Description |
| --- | --- | --- |
| `projection` | `"perspective"` or `"orthographic"` | Projection type. |
| `fov` | number, 0.01–179 | Field of view in degrees, along `fov_axis`. |
| `fov_axis` | `"vertical"`, `"horizontal"` or `"auto"` | Which image extent `fov` (and `ortho_scale`) measures. |
| `ortho_scale` | number > 0 | Orthographic view size in world units, along `fov_axis`. |
| `near`, `far` | numbers | Perspective: 0 < near < far and far/near ≤ 10⁹. Orthographic: 0 ≤ near < far. |
| `aspect_ratio` | number in [10⁻³, 10³] or `null` | Preferred frame width / height. |

Cameras look along their object's local −Y axis with local +Z up.

**`light`**

| Key | Type | Description |
| --- | --- | --- |
| `type` | `"directional"` or `"point"` | Directional lights shine along the object's local −Y axis; point lights shine from its position. |
| `color` | color | Light color. |
| `intensity` | number in [0, 10⁹] | Unitless multiplier (see [RENDERING.md](RENDERING.md)). |
| `range` | number > 0 or `null` | Point lights: smooth fade-out distance. |
| `falloff` | `"inverse_square"` or `"none"` | Point lights: distance attenuation. |

**`appearance`:** `color` (color), `specular` (0–1) and `shininess` (1–10000).

**`world`:** `background` (color), `ambient_color` (color) and
`ambient_intensity` (≥ 0).

**`render`:** `width` and `height` (1–16384), `samples` (1–4; antialiasing
is samples × samples per pixel), `exposure` (−20 to 20 EV), `cull`
(`"none"`, `"back"` or `"front"`) and `default_lights` (boolean).

## Materials and textures

Since version 4. Materials and textures have stable `id`s (any unique
non-empty strings; Aether3D generates UUID4s) and `name`s unique among the
scene's materials (or textures). Objects and bindings refer to them by id.
The meaning of every value is in [MATERIALS.md](MATERIALS.md).

**Material**

| Key | Type | Description |
| --- | --- | --- |
| `id`, `name` | strings | Required. |
| `model` | `"standard"` or `"simple"` | Required. |
| parameters | | The model's parameters: `base_color`, `opacity`, `alpha_mode`, `alpha_cutoff`, `specular`, `emissive`, `emissive_strength`, `normal_strength`, `occlusion_strength`, `double_sided`, plus `metallic` and `roughness` (standard) or `shininess` (simple). Colors are display (sRGB) values. Missing parameters take their defaults; parameters of the other model are errors. |
| `textures` | object | `{slot: binding}` for the slots `base_color`, `opacity`, `metallic`, `roughness` (standard only), `normal`, `emissive` and `occlusion`. |

**Texture binding:** `texture` (a texture id, required) and the options
`channel` (`rgb`, `r`, `g`, `b`, `a`, `luminance`), `coordinates` (`uv`,
`object`, `world`), `projection` (`triplanar`, `xy`, `xz`, `yz`, `volume`),
`offset`, `scale` (`[x, y, z]`), `rotation` (degrees) and `blend` (1–64).
Writers omit options at their defaults (the slot's channel, `uv`,
`triplanar`, offset 0, scale 1, rotation 0, blend 4).

**Texture**

| Key | Type | Description |
| --- | --- | --- |
| `id`, `name` | strings | Required. |
| `type` | `"image"` or `"procedural"` | Required. |
| `path` | string | Image, exactly one of `path` and `embedded`: a relative POSIX path inside the project's directory (no absolute paths, `..`, drive letters or backslashes). |
| `embedded` | object | `{"mime_type": "image/png" or "image/jpeg", "data": base64}`, at most 64 MiB decoded. |
| `wrap` | `[u, v]` | Image: `repeat`, `clamp` or `mirror` per axis. |
| `filter` | `"linear"` or `"nearest"` | Image. |
| `mipmaps` | boolean | Image. |
| `color_space` | `"auto"`, `"srgb"` or `"linear"` | Image. |
| `graph` | object | Procedural: a node graph (see [MATERIALS.md](MATERIALS.md#procedural-textures)). Writers emit every node parameter. |

Image files are not part of the project. Writers store paths relative to the
project's directory and refuse to save a texture outside it
(`TEXTURE_OUTSIDE_PROJECT`; embed it or copy it there). A file that is
missing when a project is loaded is not an error: the texture reports its
status and renders fall back to the slot's factor, with a warning.

## Mesh

An indexed triangle mesh.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `positions` | array of `[x, y, z]` | yes | Vertex positions (stored as 32-bit floats). |
| `triangles` | array of `[i, j, k]` | yes | Vertex indices, counter-clockwise winding seen from the front face. Each index is in `[0, vertex_count)`. |
| `normals` | array of `[x, y, z]` | no | Per-vertex normals; same length as `positions`. |
| `uvs` | array of `[u, v]` | no | Per-vertex texture coordinates; same length as `positions`. |
| `groups` | array of integers | no | Face group (0–65535) of each triangle; same length as `triangles`. Absent: every triangle is in group 0. Since version 4. |

Vertex attribute values are 32-bit floats. Writers emit the shortest decimal
text that round-trips to the same 32-bit value (e.g. `0.1`), so geometry
survives save/load bit-for-bit.

## Conventions

- Right-handed coordinate system, **+Z up**, 1 unit = 1 meter.
- Rotation quaternions are `(x, y, z, w)`.

## Validation rules

Readers are strict, so mistakes are reported instead of silently ignored:

- Unknown keys are errors (a typo such as `"postion"` is reported, not ignored).
- Missing required keys, wrong types, non-finite numbers (including `NaN`/`Infinity` literals),
  and duplicate keys within a JSON object are errors.
- Duplicate object ids or names, dangling `parent` references, parent cycles, mesh indices
  out of range, and triangle indices out of range are errors.
- Procedural content is fully validated: unknown primitive or modifier types; parameters
  that are unknown, mistyped or out of range; invalid expressions; violated rules;
  missing geometry or modifier references; geometry dependency cycles; duplicate ids
  or names; and an object with both `mesh` and `geometry`. Geometry is evaluated
  lazily after loading.
- Rendering data is validated with the same rules as the Python API:
  - camera, light, appearance, world and render values (types, ranges,
    choices, colors);
  - unknown or missing keys;
  - objects holding more than one of mesh, geometry, camera and light;
  - an `active_camera` that is not a camera object.
- Materials and textures are validated with the same rules as the Python API
  (parameter types and ranges, slots, binding options, sampler settings,
  procedural graphs and their size limits, embedded image types and sizes).
  Duplicate ids or names, unknown keys, bindings to textures that do not
  exist, object or group references to materials that do not exist, unsafe
  texture paths, and materials on cameras or lights are errors.

Every error names the offending element with a dotted path, such as
`scene.objects[1].transform.scale` or `meshes[0].triangles[12]`.

## Versioning and evolution

- `format_version` is a single integer, incremented for every change to this
  specification, including additive ones.
- A reader **rejects** files whose version is newer than it supports, with an
  error asking the user to upgrade, so newer data is never silently dropped.
- A reader **upgrades** older files on load by applying migrations one version
  at a time (`MIGRATIONS[n]` converts version *n* to *n + 1*, in
  `aether3d/serialization.py`). Files are always written in the current version.
- Each version bump must update this document, the writer, the reader, and add
  a migration plus a test that loads a document of the previous version.

## Version history

| Version | Aether3D | Changes |
| --- | --- | --- |
| 1 | 0.1.0 | Initial format: one scene, object hierarchy, TRS transforms, indexed triangle meshes with optional normals and UVs. |
| 2 | 0.4.0 | Procedural geometry: `scene.parameters`, `scene.rules`, `scene.geometry`, and a `geometry` reference on every object. The migration from 1 adds empty lists and `"geometry": null`. |
| 3 | 0.5.0 | Rendering: `scene.active_camera`, `scene.world`, `scene.render`, and `camera`, `light`, `appearance` and `visible` on every object. The migration from 2 adds `null`, the default world and the default render settings, and `null`, `null`, `null` and `true` on objects. |
| 4 | 0.7.0 | Materials: `scene.materials`, `scene.textures`, `material` and `group_materials` on every object, and optional per-triangle `groups` on meshes. The migration from 3 adds empty lists, and `null` and `{}` on objects, so every object keeps rendering with its color. |

Aether3D 0.2.0 and 0.3.0 read and write version 1 unchanged, and Aether3D
0.6.0 reads and writes version 3 unchanged (GPU rendering added render
options, not scene content). glTF interchange is a separate format (see
[GLTF.md](GLTF.md)); it did not require a project format change.
