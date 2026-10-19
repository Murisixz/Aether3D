# The `.aether` project format

**Current version: 12** (Aether3D 0.19.1; version 1–11 files are upgraded on load)

An `.aether` file stores one scene:

- its objects, with their hierarchy and transforms;
- their mesh geometry;
- its procedural geometry definitions: parameters, rules and geometry nodes
  with modifier stacks (see [PROCEDURAL.md](PROCEDURAL.md));
- its cameras, lights, object colors, background, ambient light and render
  settings (see [RENDERING.md](RENDERING.md));
- its materials and textures, and which objects and face groups use them
  (see [MATERIALS.md](MATERIALS.md)). Image textures are stored as paths
  relative to the project, or embedded; painted textures are embedded with every
  layer and mask (16 bits per channel);
- its objects' tags and metadata (see [SPATIAL.md](SPATIAL.md#tags-and-metadata));
- its CAD document (see [CAD.md](CAD.md));
- its animation document: timeline, clips, bindings, rigs, skins, morph targets,
  constraints, paths and shot sequences (see [ANIMATION.md](ANIMATION.md));
- its procedural worlds: each world's definition and block edits, never generated
  geometry (see [WORLDS.md](WORLDS.md)), and the outdoor atmosphere.

It is a UTF-8 encoded JSON document (RFC 8259), designed to be readable,
diffable, and safely editable by people, scripts, and AI agents.

## Example

```json
{
  "format": "aether3d.project",
  "format_version": 10,
  "generator": "aether3d 0.18.0",
  "scene": {
    "name": "First Scene",
    "active_camera": "5b0f1c2e-…",
    "world": {
      "background": [0.2, 0.2, 0.2], "ambient_color": [1.0, 1.0, 1.0], "ambient_intensity": 0.1,
      "environment": {"type": "ambient", "color": [1.0, 1.0, 1.0], "intensity": 1.0,
                      "zenith_color": [0.22, 0.42, 0.78], "horizon_color": [0.78, 0.85, 0.93],
                      "ground_color": [0.32, 0.3, 0.27]}
    },
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
    "worlds": [
      {"id": "3c7e9a10-…", "name": "Valley",
       "definition": {"seed": 7, "generator": 1, "mode": "terrain",
                      "extent": {"min": [-4096.0, -4096.0], "max": [4096.0, 4096.0]}, "…": "…"},
       "edits": []}
    ],
    "objects": [
      {
        "id": "c1181286-0e18-46b7-991d-99bac53d479b",
        "name": "Pivot",
        "tags": [],
        "metadata": {},
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
        "tags": ["landmark", "orbiting"],
        "metadata": {"period_days": 27.3, "discovered_by": null},
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
        "tags": [],
        "metadata": {},
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
| `cad` | object | The CAD document (see [CAD document](#cad-document)). Since version 8. |
| `animation` | object | The animation document (see [Animation document](#animation-document)). Since version 9. |
| `worlds` | array | Procedural worlds (see [Worlds](#worlds)). Since version 10. |

Writers emit objects in depth-first hierarchy order (each parent before its
children). Readers accept any order. The relative order of objects that share
a parent (or that are all roots) defines their child order.

## Scene object

| Key | Type | Description |
| --- | --- | --- |
| `id` | non-empty string | Stable unique identifier. Aether3D generates UUID4 strings, but any unique string is accepted. Unique within the file. |
| `name` | non-empty string | Human-readable name. Unique within the file. |
| `tags` | array of strings | The object's tags: sorted, unique, each 1–64 characters without surrounding whitespace, at most 64. Since version 6. |
| `metadata` | object | Free-form JSON data (string keys; strings, finite numbers, booleans, `null`, arrays, objects), at most 16 KiB as compact JSON. Since version 6. |
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
Cameras and lights cannot have materials. Writers always emit `tags`,
`metadata` (right after `name`), `camera`, `light`, `appearance`, `visible`,
`material` and `group_materials`. Readers treat them as optional, with the
defaults `[]`, `{}`, `null`, `null`, `null`, `true`, `null` and `{}`. Invalid
tags or metadata are a `ProjectFormatError` with their location
(`scene.objects[2].tags`). An object keeps its `appearance` while it uses a material;
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
| `sensor` | number in [1, 1000] | Sensor size in mm along `fov_axis` (the focal length is `sensor / (2 tan(fov/2))`). Since version 13. |
| `shift_x`, `shift_y` | numbers in [−10, 10] | Lens shift in units of the image's larger side. Since version 13. |
| `fstop` | 0, or a number in [0.1, 10⁴] | The aperture's f-number; 0 is a pinhole. Since version 13. |
| `focus_distance`, `focus_object` | number in [10⁻⁶, 10¹²], an object id or `null` | The plane in focus. Since version 13. |
| `aperture_blades`, `aperture_rotation`, `aperture_ratio` | 0 or 3–16, [−360, 360] degrees, [0.01, 100] | The aperture's shape. Since version 13. |

Cameras look along their object's local −Y axis with local +Z up.

**`light`**

| Key | Type | Description |
| --- | --- | --- |
| `type` | `"directional"`, `"point"`, `"spot"` or `"area"` (the last two since version 13) | Directional, spot and area lights shine along the object's local −Y axis; point and spot lights from its position; area lights from a shape centred there. |
| `color` | color | Light color. |
| `intensity` | number in [0, 10⁹] | Unitless multiplier (see [RENDERING.md](RENDERING.md)). |
| `range` | number > 0 or `null` | Point lights: smooth fade-out distance. |
| `falloff` | `"inverse_square"` or `"none"` | Point lights: distance attenuation. |
| `radius` | number in [0, 10⁹] | Point lights: radius of the emitting sphere (path tracer soft shadows; 0: a point). Since version 5. |
| `angle` | number in [0, 180] | Directional lights: angular diameter in degrees (path tracer soft shadows; 0: parallel rays). Since version 5. |
| `spot_size`, `spot_blend` | 0.1–180, 0–1 | Spot lights' cone (full angle, degrees) and soft edge. Since version 13. |
| `shape`, `size`, `size_y`, `spread` | `"square"`, `"rectangle"`, `"disk"` or `"ellipse"`; numbers in [10⁻⁶, 10⁶]; 1–180 | Area lights' shape, sizes (local X, Z) and emission spread in degrees. Since version 13. |
| `units` | `"aether"` or `"watts"` | How `intensity` is read. Since version 13. |
| `temperature`, `use_temperature` | 100–100000, boolean | Colour temperature (kelvins; the colour is computed at 800–12000), and whether it tints the colour. Since version 13. |
| `exposure` | number in [−40, 40] | Stops scaling the intensity. Since version 13. |
| `shadow` | boolean | Whether the light is occluded. Since version 13. |
| `include`, `exclude` | arrays of object ids | Light linking. Since version 13. |

Since version 13 every light key is written; a version 12 light gets the defaults (a
spot of 45 degrees, a 1 m square, spread 180, `aether` units, 6500 K unused, exposure 0,
shadows, no links), which light exactly as before.

**`appearance`:** `color` (color), `specular` (0–1) and `shininess` (1–10000).

**`world`:** `background` (color), `ambient_color` (color),
`ambient_intensity` (≥ 0) and, since version 5, `environment`:

| Key | Type | Description |
| --- | --- | --- |
| `type` | `"ambient"`, `"uniform"`, `"sky"` or `"image"` (since version 13) | What the path tracer sees where rays leave the scene: the ambient light as a uniform environment (camera rays show `background`), a uniform `color`, a procedural sky, or an HDR image (which lights the rasterizers too). See [RENDER_QUALITY.md](RENDER_QUALITY.md). |
| `color` | color | Uniform environment color. |
| `intensity` | number in [0, 10⁶] | Radiance multiplier of `"uniform"` and `"sky"`. |
| `zenith_color`, `horizon_color`, `ground_color` | colors | The sky gradient: straight up (+Z), at the horizon, below it. |
| `image` | string or `null` | `"image"`: a relative POSIX path of an `.hdr`/`.pic`/`.pfm` file inside the project's directory, or `"builtin:studio"`, `"builtin:overcast"`, `"builtin:sunset"`, `"builtin:night"`. Since version 13. |
| `rotation` | number in [−10⁶, 10⁶] | Degrees about +Z. Since version 13. |
| `visible` | boolean | Whether camera rays see the image (else `background`). Since version 13. |

All keys are always written. The rasterizers ignore `environment` except an image's
lighting (version 13).

Since version 10, `world.atmosphere` (all keys always written; off by default, so scenes
render as before; see [RENDERING.md](RENDERING.md)):

| Key | Type | Description |
| --- | --- | --- |
| `enabled` | boolean | Turn the atmosphere on. |
| `sun` | string or `null` | Name or id of the directional light that places the sun (`null`: the brightest). |
| `sky` | boolean | Rasterizers draw `environment`'s sky behind the geometry instead of `background`. |
| `sun_glow`, `sun_glow_color`, `sun_glow_size` | 0–100, color, 1–65536 | The glow around the sun in the sky: strength, color, tightness (an exponent). |
| `horizon_glow` | 0–100 | A warm band along the horizon, strongest toward the sun. |
| `fog_density`, `fog_height`, `fog_falloff` | 0–10, ±10⁷, 0–10 | Exponential height fog: extinction per metre at `fog_height`, thinning by `e^(-fog_falloff (z - fog_height))`. |
| `fog_color`, `fog_sun`, `fog_sun_power` | color, 0–100, 1–4096 | Fog color and sunlight scattered toward the camera. |
| `ambient` | `"uniform"` or `"hemisphere"` | Rasterizer ambient light: uniform, or sky above and `ground_ambient` below. |
| `ground_ambient` | color | The ambient light from below with `"hemisphere"`. |

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
| parameters | | The model's parameters: `base_color`, `opacity`, `alpha_mode`, `alpha_cutoff`, `specular`, `emissive`, `emissive_strength`, `normal_strength`, `occlusion_strength`, `double_sided`, plus `metallic`, `roughness`, `transmission` and `ior` (standard; the last two since version 13) or `shininess` (simple). Colors are display (sRGB) values. Missing parameters take their defaults; parameters of the other model are errors. |
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
| `type` | `"image"`, `"procedural"` or `"painted"` (since version 7) | Required. |
| `path` | string | Image, exactly one of `path` and `embedded`: a relative POSIX path inside the project's directory (no absolute paths, `..`, drive letters or backslashes). |
| `embedded` | object | `{"mime_type": "image/png" or "image/jpeg", "data": base64}`, at most 64 MiB decoded. |
| `wrap` | `[u, v]` | Image: `repeat`, `clamp` or `mirror` per axis. |
| `filter` | `"linear"` or `"nearest"` | Image. |
| `mipmaps` | boolean | Image. |
| `color_space` | `"auto"`, `"srgb"` or `"linear"` | Image. |
| `graph` | object | Procedural: a node graph (see [MATERIALS.md](MATERIALS.md#procedural-textures)). Writers emit every node parameter. |
| `width`, `height` | integers | Painted: the size in texels (1–16384 each, at most 8192 x 8192 in all). |
| `encoding` | `"color"`, `"data"` or `"normal"` | Painted: what the texels hold (display sRGB colors, linear values, tangent-space normals). |
| `background` | `[r, g, b, a]` | Painted: the RGBA under every layer, 0–1, in the encoding. |
| `layers` | array | Painted, bottom to top, at most 64: `{"id", "name", "kind": "paint" or "fill", "blend", "opacity", "visible", "mask"}`; a paint layer adds `pixels`, a fill layer `value` (`[r, g, b, a]`). `pixels` is an embedded 16-bit RGBA PNG of the layer (`{"mime_type": "image/png", "data": base64}`, transparent where nothing is painted); `mask` is `null` or an embedded 16-bit grayscale PNG. `blend` is one of `normal`, `multiply`, `screen`, `overlay`, `darken`, `lighten`, `difference`, `add`, `subtract`. |

Painted textures also take the image keys `wrap`, `filter`, `mipmaps` and `color_space`.

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
| `colors` | array of `[r, g, b, a]` | no | Per-vertex linear colors in [0, 1] (glTF `COLOR_0`: they multiply the material's base color and opacity); same length as `positions`. Since version 10. |

Vertex attribute values are 32-bit floats. Writers emit the shortest decimal
text that round-trips to the same 32-bit value (e.g. `0.1`), so geometry
survives save/load bit-for-bit.

## CAD document

`scene.cad` holds the precision-modeling document ([CAD.md](CAD.md)):

```json
"cad": {
  "settings": {"length_unit": "mm", "angle_unit": "deg", "precision": 3,
               "display": {"chord": 0.0005, "angle_deg": 12.0, "relative": 0.002}},
  "items": [
    {"id": "2ab6dad9-…", "kind": "parameter", "name": "w",
     "definition": {"value": "80 mm", "dimension": "length"}, "links": []},
    {"id": "0c1600bf-…", "kind": "sketch", "name": "S",
     "definition": {"plane": "XY", "constraints": [],
                    "entities": [{"id": "p1", "type": "point", "at": [0.0, 0.0]},
                                 {"id": "p2", "type": "point", "at": [0.08, 0.0]},
                                 {"id": "l1", "type": "line", "start": "p1", "end": "p2"}]},
     "links": []},
    {"id": "4be3e556-…", "kind": "part", "name": "P",
     "definition": {"features": [{"type": "box", "id": "b", "size": [10, 10, 10]}]},
     "links": ["811bb703-…"]}
  ]
}
```

| Key | Description |
| --- | --- |
| `settings` | The document's length and angle units, display precision and display tessellation tolerance. |
| `items` | Every item: `id`, `kind` (`parameter`, `frame`, `sketch`, `part`, `assembly`, `building`, `floor`, `slab`, `roof`, `stair`, `columns`, `beams`, `drawing`), unique `name`, `definition` (plain JSON, in the document's units -- except sketch point positions and radii, the solver's state, in metres), `links` (ids of the scene objects that show it). |

Only definitions are stored; results (solved sketches, bodies, drawings) are regenerated
on load. Objects that show an item keep their current tessellation in `meshes`, so a
reader without CAD support still sees the geometry. A link to a missing object is an
error. Definitions are checked by their item kinds as on any change.

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

## Animation document

`scene.animation` holds the animation ([ANIMATION.md](ANIMATION.md)): the timeline
settings and the items, each with an immutable JSON definition.

```json
"animation": {
  "settings": {"fps": "30000/1001", "start": 0.0, "end": 2.0,
               "markers": [{"time": 1.0, "name": "hit"}], "sections": [], "enabled": true},
  "items": [
    {"id": "9f2ba36d-…", "kind": "binding", "name": "Box",
     "definition": {"object": "013105d3-…", "action": "8e9f5335-…"}},
    {"id": "8e9f5335-…", "kind": "clip", "name": "Box Action",
     "definition": {"target": "object", "channels": {
       "transform.position.z": {"keys": {"time": [0.0, 1.0], "value": [0.0, 1.0],
         "interpolation": "bezier", "left_type": "auto_clamped",
         "right_type": "auto_clamped"}}}}}
  ]
}
```

| Key | Description |
| --- | --- |
| `settings` | The timeline: `fps` (an integer or an exact rational `"num/den"`), `start` and `end` (seconds), `markers` (`{time, name, data?}`), `sections`, `enabled`. |
| `items` | Every item: `id`, `kind` (`clip`, `binding`, `rig`, `skin`, `constraints`, `morph`, `path`, `sequence`), `name`, `definition` (plain JSON: clips hold channels by property path as structure-of-arrays keys, rotation tracks, modifiers and events; bindings reference an object or material (or, since version 13, the world: `{"world": "world"}`) and clips by id; rigs, skins, morph targets and constraints reference scene objects by id). |

Since version 11 (0.19.0) rig and skin definitions may carry the character-rigging data
([CHARACTER_RIGGING.md](CHARACTER_RIGGING.md)), all optional: joints `role` and
`deform`, swing-twist and hinge `limits` (besides Euler bounds); rigs `controls`,
`correctives`, `landmarks`, `template`, `info` and `enforce_limits`; constraints of type
`twist` and IK constraints' `bend_axis`; skins `method` `"blended"` with a per-vertex
`dq` share, and `locked_joints` / `locked_vertices`. A rig or skin without them is the
Stage 17 one. Since version 12 (0.19.1), also optional: joints `span` (`[start, end]`,
the part of the bone, head to tail, whose segment heat weights use; default `[0, 1]`),
skins `normals` (`"joints"`, the default when absent, or `"surface"`) and `smooth`
(corrective smoothing: `{"iterations": 1..100, "strength": (0, 1], "region": "blend" |
"all", "weights": "cotangent" | "uniform"}`, absent for none).

Since version 13 (0.20.0), also optional: a curve's `smoothing` (`"local"` when absent,
`"blender"` or `"continuous"`: how its automatic handles are computed), the modifiers
`generator`, `function` and `cycles`, and a strip's `blend_curve` (`"linear"` when absent,
`"smooth"`, `"smoother"`). The `back` parameter of `back` easing with `in_out` is
Blender's (Penner's scale of 1.525 is applied when the curve is evaluated): the migration
from 12 divides the stored values of such keys (and of rotation timings `{"easing":
"back", "ease": "in_out"}`) by 1.525, so they evaluate as before.

Only authored data is stored: evaluated poses, deformed meshes and the current time are
not. Definitions are checked by their item kinds on load as on any change (finite
numbers, key order, handle shapes, quaternion norms, acyclic joint hierarchies and
constraint graphs, influence counts, references); a reference to a missing object or
clip is an error.

## Worlds

`scene.worlds` holds procedural worlds ([WORLDS.md](WORLDS.md)) -- authored data only:

| Key | Description |
| --- | --- |
| `id` | The world's UUID. |
| `name` | Unique among the scene's worlds. |
| `definition` | The complete, validated definition: every section with its defaults filled in (`seed`, `generator`, `mode`, `extent`, `sea_level`, `terrain`, `climate`, `biomes`, `surface`, `hydrology`, `caves`, `vegetation`, `structures` with its authored `items`, `blocks`, `materials`, `streaming`, `features`). It is checked on load as on any change; `generator` (1) names the generation rules it was made with. |
| `edits` | Block worlds' edited blocks: a list of `{"section": [sx, sy, sz], "data": base64}`, one per edited 16 x 16 x 16 section, the data being zlib-compressed little-endian `uint16` pairs -- the edited cells' local indices `(lz * 16 + ly) * 16 + lx` in ascending order, then their block ids (indices into the block list, `aether3d world blocks`; `0` is air, a removed block; new kinds are only ever appended). Empty for terrain worlds. |

Generated data -- tiles, plants, chunks, caches -- is never stored: a world regenerates the
same bits from its definition and edits. A terrain world with features and structures
takes a few kilobytes.

## History files

`scene.save(path, history=True)` (and the command line, for projects that keep a
history) writes the scene's undo history next to the project, in
`<project>.history` (`castle.aether.history`). The project format does not
change: a project loads the same with or without its history file, and other
programs can ignore it. See [HISTORY.md](HISTORY.md).

The history file is a JSON document, `"format": "aether3d.history"`,
`"format_version": 1`:

| Key | |
| --- | --- |
| `project` | SHA-256 of the project document it belongs to (canonical JSON, without `generator`). A history is applied only to the exact project it was saved with |
| `position`, `serial` | Index of the step that produced the current state (0: none), and the last index used |
| `undo`, `redo` | Steps: `{"index", "parent", "label", "changes"}`, the undone ones in `redo` (the next to redo last) |
| `checkpoints` | Names and the index of the step behind their state |
| `max_entries`, `max_bytes`, `dropped` | The limits and how many steps were dropped |
| `objects`, `materials`, `textures`, `cad_items`, `animation_items`, `world_items` | Full records of entities the steps refer to that are not in the project (removed ones, which undo can bring back; `cad_items` since 0.15.0, `animation_items` since 0.17.0, `world_items` -- ids, their states being in the steps -- since 0.18.0) |
| `meshes`, `blobs` | Meshes (as in the project's `meshes`) and embedded image bytes (base64) the states refer to by index; painted textures' states refer to their tiles (64 x 64 texels, 16-bit RGBA, zlib-compressed) as blobs, so a stroke's step stores only the tiles it changed |

A change is `{"aspect", "target", "before", "after"}`:
- `aspect` is the part of the scene it changed: `object.name`,
  `object.transform`, `object.visible`, `object.appearance`, `object.camera`,
  `object.light`, `object.geometry`, `object.material`, `object.annotations`
  (tags and metadata, since 0.10.0), `object.parent`,
  `object.children`, `object.alive`, `scene.name`, `scene.roots`,
  `scene.active_camera`, `scene.world`, `scene.render`, `library.materials`,
  `library.textures`, `material`, `texture`, `procedural`, since 0.15.0
  `cad.item` (one CAD item: kind, name, definition, links, alive) and `cad.settings`,
  since 0.17.0 `animation.item` (one animation item: kind, name, definition, alive)
  and `animation.settings` (the timeline), and since 0.18.0 `world.item` (one world:
  alive, name, definition, edits; `scene.world` then includes the atmosphere).
- `target` is the id of the object, material or texture (`null` for the others).
- The states use the project's encodings (transforms as position, rotation and
  scale arrays; cameras, lights, appearances, world and render settings as in
  `scene.objects[]` and `scene`; materials and procedural definitions as in
  `scene.materials` and `scene.geometry`), and object, material and texture ids.

Readers check a history file before using it. It must match the project's
fingerprint (`HISTORY_MISMATCH` otherwise), and it is replayed on a copy of the
scene (every step undone and redone) and refused (`INVALID_HISTORY`) unless the
copy comes back to the same document.

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
| 5 | 0.8.0 | Path tracing: `scene.world.environment` and the light sizes `radius` and `angle`. The migration from 4 adds the `"ambient"` environment (the path tracer then lights with the ambient light, as the rasterizers do) and sizes of 0 (point lights and parallel directional lights), so every project renders unchanged on the rasterizers. |
| 6 | 0.10.0 | Object tags and metadata: `tags` and `metadata` on every object. The migration from 5 changes nothing: their defaults are empty. |
| 7 | 0.14.0 | Painted textures: textures of `"type": "painted"` with their layers and masks. The migration from 6 changes nothing: a version 6 document has none. |
| 8 | 0.15.0 | The CAD document: `scene.cad` (settings and items). The migration from 7 adds an empty document with the default settings. |
| 9 | 0.17.0 | The animation document: `scene.animation` (timeline settings and items). The migration from 8 adds an empty document with the default timeline (24 fps, 0 to 10 s), so every project loads and renders unchanged. |
| 10 | 0.18.0 | Procedural worlds and outdoor rendering: `scene.worlds`, `scene.world.atmosphere` and optional mesh `colors`. The migration from 9 adds no worlds and the default atmosphere (off), so every project loads and renders unchanged. |
| 11 | 0.19.0 | Character rigging: optional joint `role` and `deform`, swing-twist and hinge limits, rig `controls`, `correctives`, `landmarks`, `template`, `info`, `enforce_limits`, `twist` constraints and IK `bend_axis`, blended skins (`method: "blended"`, `dq`) and weight locks. The migration from 10 changes nothing: a version 10 document has none of them. |
| 12 | 0.19.1 | The rigging revision: optional joint `span`, skin `normals` and `smooth`. The migration from 11 changes nothing: a version 11 document has none of them (its skins keep joint normals and no smoothing). |
| 13 | 0.20.0 | Animation, lighting and film: camera lens keys; spot and area lights, light `units`, `temperature`, `exposure`, `shadow`, `include`/`exclude`; image environments; material `transmission` and `ior`; curve `smoothing`, new modifiers, strip `blend_curve`, world bindings. The migration from 12 writes the lens and light defaults (every camera a pinhole, every light as before), the environment's image keys, and divides the `back` of `in_out` back-eased keys by 1.525 (their curves keep their shapes); curves without `smoothing` keep the local automatic handles. Every project loads and renders unchanged. |

Aether3D 0.2.0 and 0.3.0 read and write version 1 unchanged, and Aether3D
0.6.0 reads and writes version 3 unchanged (GPU rendering added render
options, not scene content). Aether3D 0.8.1 and 0.9.0 read and write version 5
unchanged: 0.9.0 keeps the undo history in a separate file
([History files](#history-files)). History files of 0.9.0 still load in
0.10.0: their removed objects simply have no tags or metadata. glTF interchange is a separate format (see
[GLTF.md](GLTF.md)); it did not require a project format change.
