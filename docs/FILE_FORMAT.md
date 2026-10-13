# The `.aether` project format

**Current version: 2** (Aether3D 0.4.0; version 1 files are upgraded on load)

An `.aether` file stores one scene: its objects, their hierarchy and transforms,
their mesh geometry, and its procedural geometry definitions (parameters,
rules and geometry nodes with modifier stacks; see [PROCEDURAL.md](PROCEDURAL.md)). It is a UTF-8 encoded JSON document (RFC 8259) designed
to be readable, diffable, and safely editable by people, scripts, and AI agents.

## Example

```json
{
  "format": "aether3d.project",
  "format_version": 2,
  "generator": "aether3d 0.4.0",
  "scene": {
    "name": "First Scene",
    "parameters": [],
    "rules": [],
    "geometry": [],
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
        "geometry": null
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
        "geometry": null
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
| `parameters` | array | Scene parameters (see [Procedural geometry](#procedural-geometry)). Since version 2. |
| `rules` | array | Scene rules. Since version 2. |
| `geometry` | array | Procedural geometry nodes. Since version 2. |
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

## Mesh

An indexed triangle mesh.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `positions` | array of `[x, y, z]` | yes | Vertex positions (stored as 32-bit floats). |
| `triangles` | array of `[i, j, k]` | yes | Vertex indices, counter-clockwise winding seen from the front face. Each index is in `[0, vertex_count)`. |
| `normals` | array of `[x, y, z]` | no | Per-vertex normals; same length as `positions`. |
| `uvs` | array of `[u, v]` | no | Per-vertex texture coordinates; same length as `positions`. |

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

Aether3D 0.2.0 and 0.3.0 read and write version 1 unchanged. glTF interchange
is a separate format (see [GLTF.md](GLTF.md)); it did not require a project
format change.
