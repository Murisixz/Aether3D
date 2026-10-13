# Geometry inspection

Aether3D can analyse meshes and whole scenes and return structured,
JSON-compatible reports.

```python
from aether3d import inspect_mesh, create_cube

report = scene.inspect_geometry()  # SceneGeometryReport
report.status  # "ok" | "warning" | "error"
data = report.to_dict()  # plain JSON-compatible data

inspect_mesh(create_cube()).to_dict()  # MeshGeometryReport for one mesh
```

```console
$ aether3d geometry world.aether --json      # also accepts .glb / .gltf
```

`aether3d geometry` exits with `0` when the report has no error-severity
findings, `3` when it does, and `1` if the file cannot be read.

## Status and diagnostics

Every finding is a **diagnostic**:

```json
{
  "code": "DEGENERATE_TRIANGLES_ZERO_AREA",
  "severity": "warning",
  "message": "triangles have (near-)zero area",
  "location": "meshes[2]",
  "count": 3,
  "samples": [17, 40, 41],
  "lossy": false
}
```

| Field | Meaning |
| --- | --- |
| `code` | Stable identifier. Programs should match on this, not on the message. |
| `severity` | `error`: invalid data. `warning`: probably unintended. `info`: worth knowing, often expected. |
| `location` | `meshes[i]` for mesh findings, `objects[<id>]` for object findings. |
| `count` | Number of affected elements (triangles, vertices or edges, depending on the check). |
| `samples` | Up to 10 example element indices (e.g. triangle indices). |
| `lossy` | Always `false` for inspection; used by import and export. |

`status` is the worst severity present: `"error"`, else `"warning"`, else
`"ok"`. Info findings never change the status.

## Mesh report

| Field | Description |
| --- | --- |
| `vertex_count`, `triangle_count` | Counts. |
| `attributes` | `{"positions": true, "normals": bool, "uvs": bool}` |
| `bounds` | Local-space `min`, `max`, `size`, `center` (or `null` for an empty mesh). |
| `surface_area` | Sum of triangle areas, in local units². |
| `uv_bounds` | `min`/`max` of the UVs, or `null`. |
| `topology` | Edge analysis; see below. |
| `diagnostics` | Findings for this mesh. |

### Checks

| Code | Severity | Condition |
| --- | --- | --- |
| `NON_FINITE_POSITIONS` / `_NORMALS` / `_UVS` | error | NaN or ±inf values. `Mesh` already rejects these; the check is a safeguard. |
| `INDEX_OUT_OF_RANGE` | error | A triangle references a missing vertex (also prevented by `Mesh`). |
| `EMPTY_MESH` | info | The mesh has no triangles. |
| `DEGENERATE_TRIANGLES_REPEATED_INDEX` | warning | A triangle uses the same vertex index twice. |
| `DEGENERATE_TRIANGLES_ZERO_AREA` | warning | Distinct indices, but area ≤ 10⁻¹² × (bounding-box diagonal)². |
| `DUPLICATE_TRIANGLES` | warning | A triangle uses the same three vertex indices as an earlier one, in any order. |
| `UNREFERENCED_VERTICES` | info | Vertices used by no triangle. |
| `ZERO_LENGTH_NORMALS` | warning | A normal has length < 10⁻¹². |
| `NON_UNIT_NORMALS` | warning | A normal's length differs from 1 by more than 10⁻³. |
| `NORMALS_OPPOSE_WINDING` | warning | A triangle's front face (counter-clockwise winding) points away from the sum of its vertex normals. |
| `OPEN_BOUNDARY` | info | The mesh has boundary edges. This is expected for open surfaces such as planes. |
| `NON_MANIFOLD_EDGES` | warning | An edge is shared by more than two triangles. |
| `INCONSISTENT_ORIENTATION` | warning | Two triangles traverse a shared edge in the same direction. |
| `INSIDE_OUT` | warning | The mesh is closed and consistently oriented, but its volume is negative: the triangles face inwards. |

### Topology: exactly what is evaluated

Topology is computed on **welded** vertices. Two vertices are merged only
when their positions are **bit-identical** (`-0.0` equals `0.0`); no distance
tolerance is applied. Degenerate triangles are excluded. On that graph:

| Field | Meaning |
| --- | --- |
| `vertex_welding` | Always `"exact_position"`. |
| `welded_vertex_count`, `edge_count` | Sizes of the welded graph. |
| `boundary_edge_count` | Edges used by exactly one triangle. |
| `non_manifold_edge_count` | Edges used by more than two triangles. |
| `inconsistent_edge_count` | Two-triangle edges traversed in the same direction by both. |
| `closed` | `boundary_edge_count == 0`. |
| `edge_manifold` | `non_manifold_edge_count == 0`. |
| `consistently_oriented` | `inconsistent_edge_count == 0`. |
| `volume` | Signed enclosed volume, given **only** when the mesh is closed, edge-manifold and consistently oriented; otherwise `null`. Positive means the triangles face outwards. |

**Not evaluated:** vertex manifoldness (for example two surfaces touching at a
single vertex) and self-intersections. The reports therefore never claim a
mesh is "manifold" or "watertight" in general. They report only the edge
properties above. Because welding requires bit-identical positions, surfaces
that are merely very close count as separate. For example, UV spheres created
by Aether3D 0.1.0 have a seam gap of about 10⁻¹⁶ and are reported as open.

## Scene report

| Field | Description |
| --- | --- |
| `scene` | Scene name. |
| `summary` | See the summary fields below. |
| `world_bounds` | World-space bounds of all mesh vertices, or `null`. |
| `objects` | One entry per object, in depth-first order (see below). |
| `meshes` | One mesh report per **distinct** mesh, in order of first use, each with `index` and `used_by` (object ids). |
| `diagnostics` | Object-level findings (below). Mesh findings are inside `meshes`. |

The `summary` fields:

- `object_count`, `mesh_object_count`, `unique_mesh_count`;
- `instanced_vertex_count`, `instanced_triangle_count`: counted per object
  that uses a mesh;
- `unique_vertex_count`, `unique_triangle_count`: counted once per distinct
  mesh.

Each `objects` entry has:

- `id`, `name`, `parent`;
- `mesh`: an index into `meshes`, or `null`;
- `world_matrix_determinant`, `mirrored`, `sheared`;
- `world_bounds`;
- `world_surface_area`: computed from the transformed vertices.

| Code | Severity | Condition |
| --- | --- | --- |
| `ZERO_WORLD_SCALE` | warning | The world matrix is singular; the geometry collapses. |
| `MIRRORED_WORLD_TRANSFORM` | info | The world matrix has a negative determinant; winding appears reversed in world space. |
| `SHEARED_WORLD_TRANSFORM` | info | Non-uniform scale under a rotated parent produces shear; `world_rotation` and `world_scale` are approximations. |

Inspection never modifies the scene.
