# The `.aether` project format

**Current version: 1**

An `.aether` file stores one scene: its objects, their hierarchy and transforms,
and their mesh geometry. It is a UTF-8 encoded JSON document (RFC 8259) designed
to be readable, diffable, and safely editable by people, scripts, and AI agents.

## Example

```json
{
  "format": "aether3d.project",
  "format_version": 1,
  "generator": "aether3d 0.1.0",
  "scene": {
    "name": "First Scene",
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
        "mesh": null
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
        "mesh": 0
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
| `mesh` | integer or `null` | Index into the top-level `meshes` array, or `null` for an object without geometry. Several objects may share one mesh. |

### `transform`

| Key | Type | Description |
| --- | --- | --- |
| `position` | `[x, y, z]` | Translation. |
| `rotation` | `[x, y, z, w]` | Rotation quaternion (glTF component order). Normalized on load; must be non-zero. |
| `scale` | `[x, y, z]` | Per-axis scale. |

The local matrix is `T @ R @ S` (column vectors). An object's world matrix is
`parent.world @ local`. All numbers must be finite.

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

Aether3D 0.2.0 still reads and writes version 1 unchanged. glTF interchange
is a separate format (see [GLTF.md](GLTF.md)); it did not require a project
format change.
