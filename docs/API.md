# Python API reference

Everything below is importable from the top-level `aether3d` package and is
part of the supported public API. Names not listed in `aether3d.__all__` (and
modules or attributes starting with `_`) are internal and may change.

```python
from aether3d import Scene, SceneObject, Transform, Mesh
from aether3d import create_cube, create_sphere, create_plane
from aether3d import save_scene, load_scene, FORMAT_VERSION
from aether3d import import_gltf, export_gltf, ImportResult, ExportResult, ImportLimits
from aether3d import inspect_mesh, inspect_scene, MeshGeometryReport, SceneGeometryReport
from aether3d import Diagnostic
from aether3d import AetherError  # and the specific error classes below
```

## Conventions

| Topic | Convention |
| --- | --- |
| Coordinates | Right-handed, **+Z up**, +X right; an asset's front faces -Y (as in Blender); 1 unit = 1 meter |
| Vectors | Returned as tuples of Python `float`; accept any 3-element sequence or NumPy array |
| Rotation | Unit quaternion `(x, y, z, w)`, normalized on assignment and canonicalized to `w >= 0` |
| Euler angles | Degrees, XYZ order about fixed world axes: `R = Rz @ Ry @ Rx` |
| Matrices | 4x4 `float64` NumPy arrays, column vectors (`p' = M @ p`), local = `T @ R @ S` |
| World space | `world_matrix = parent.world_matrix @ local_matrix` |
| Winding | Triangles are counter-clockwise when seen from the front (outside) |
| UVs | Origin at the bottom-left of the texture (as in Blender/OpenGL); `v` grows upwards |
| glTF axes | `glTF (x, y, z) = (x, z, -y)`; see [GLTF.md](GLTF.md) |

Read-only values come back as tuples or fresh arrays, so `obj.position[0] = 5`
cannot silently change nothing. Always assign a whole value: `obj.position = (5, 0, 0)`.

## `Scene`

```python
Scene(name: str = "Untitled Scene")
```

| Member | Description |
| --- | --- |
| `name` | Scene name (non-empty string, settable). |
| `add_object(name=None, *, mesh=None, parent=None, position=None, rotation=None, scale=None, object_id=None)` | Add an object, optionally with a `Mesh`. Returns the `SceneObject`. |
| `add_cube(name=None, *, size=1.0, parent=None, position=None, rotation=None, scale=None)` | Add a cube with edge length `size`. |
| `add_sphere(name=None, *, radius=1.0, segments=32, rings=16, parent=None, position=None, rotation=None, scale=None)` | Add a UV sphere. |
| `add_plane(name=None, *, size=1.0, parent=None, position=None, rotation=None, scale=None)` | Add a square plane in the XY plane facing +Z. |
| `get(object_id)` | Object by id; raises `ObjectNotFoundError`. |
| `get_by_name(name)` | Object by name; raises `ObjectNotFoundError`. |
| `remove(obj_or_id, *, recursive=True)` | Remove an object. With `recursive=False`, its children move up to its parent and keep their world placement. |
| `roots` | Tuple of root objects, in order. |
| `objects` | Tuple of all objects in depth-first order (parents before children). |
| `len(scene)`, `iter(scene)`, `x in scene` | Object count, depth-first iteration, membership (object or id). |
| `save(path)` | Write a `.aether` file (atomically). |
| `Scene.load(path)` | Read and validate a `.aether` file. |
| `to_dict()` / `Scene.from_dict(doc)` | Convert to/from the JSON-compatible document stored in `.aether` files. |
| `export(path, *, allow_lossy=False)` | Export to `.glb` or `.gltf` (by extension); returns an `ExportResult`. |
| `Scene.import_file(path, *, allow_lossy=False, limits=None)` | Import a `.glb` or `.gltf` file as a new scene. |
| `inspect_geometry()` | Analyse all geometry; returns a `SceneGeometryReport`. |
| `unique_name(base)` | `base` if unused, else the first free `base.001`, `base.002`, … |

**Names.** Object names are unique within a scene. If `name` is omitted, a
unique one is generated (`"Cube"`, `"Cube.001"`, …). An explicit name that is
already taken raises `NameConflictError`, so a request is never silently
altered.

**Parents.** `parent` arguments accept a `SceneObject` or an object id string.

**Atomicity.** Creation and hierarchy operations validate all their inputs
before changing anything; a call that raises leaves the scene unchanged.

## `SceneObject`

Created only through a `Scene`.

| Member | Description |
| --- | --- |
| `id` | Stable unique id (UUID4 string). Read-only, preserved by save/load. |
| `name` | Unique name, settable (raises `NameConflictError` if taken). |
| `scene` | The owning scene, or `None` after removal. |
| `mesh` | A `Mesh` or `None`, settable. Meshes may be shared between objects. |
| `transform` | The local `Transform`. Assigning a `Transform` copies it. |
| `position`, `rotation`, `rotation_euler`, `scale` | Shortcuts for the local transform's components. |
| `parent` | Parent object or `None`. |
| `children` | Tuple of direct children, in order. |
| `iter_descendants()` | Depth-first iterator over all descendants. |
| `iter_ancestors()` | Iterator: parent, grandparent, …, root. |
| `set_parent(parent, *, keep_world_transform=False)` | Re-parent (`None` = root). Raises `HierarchyError` on cycles. |
| `local_matrix` | 4x4 local-to-parent matrix. |
| `world_matrix` | 4x4 local-to-world matrix. |
| `world_position` | World-space position. |
| `world_rotation`, `world_scale` | World-space rotation and scale, decomposed from `world_matrix`. Exact unless a parent combines non-uniform scale with rotation (shear), where the closest rotation is returned. |

Using a removed object in a hierarchy operation raises `DetachedObjectError`.

## `Transform`

```python
Transform(position=(0, 0, 0), rotation=(0, 0, 0, 1), scale=(1, 1, 1))
```

| Member | Description |
| --- | --- |
| `position` | `(x, y, z)` |
| `rotation` | Quaternion `(x, y, z, w)` |
| `rotation_euler` | XYZ Euler angles in degrees (a view of `rotation`) |
| `scale` | `(x, y, z)`; assigning a single number sets a uniform scale |
| `matrix` | 4x4 matrix `T @ R @ S` |
| `Transform.from_matrix(m)` | Decompose an affine matrix |
| `copy()`, `is_close(other, tolerance=1e-6)`, `==` | Value helpers |

## `Mesh`

```python
Mesh(positions, triangles, normals=None, uvs=None)
```

An immutable indexed triangle mesh. Inputs may be nested sequences or NumPy
arrays, `(N, 3)` or flat. Data is validated, copied, and stored read-only:

| Member | Description |
| --- | --- |
| `positions` | `(N, 3) float32` |
| `triangles` | `(M, 3) uint32` |
| `normals` | `(N, 3) float32` or `None` |
| `uvs` | `(N, 2) float32` or `None` |
| `vertex_count`, `triangle_count` | Counts |
| `bounds()` | `((min_x, min_y, min_z), (max_x, max_y, max_z))` in local space |
| `==` | Exact data equality |

## Primitive generators

Each returns a new `Mesh` centered at the origin, with normals and UVs.

| Function | Result |
| --- | --- |
| `create_cube(size=1.0)` | Cube with edge length `size`: 24 vertices (flat faces), 12 triangles |
| `create_sphere(radius=1.0, segments=32, rings=16)` | UV sphere with poles on ±Z; `segments >= 3`, `rings >= 2` |
| `create_plane(size=1.0)` | Square of side `size` in the XY plane, facing +Z: 4 vertices, 2 triangles |

## Serialization

| Name | Description |
| --- | --- |
| `save_scene(scene, path)` | Same as `scene.save(path)`; returns the resolved `Path`. |
| `load_scene(path)` | Same as `Scene.load(path)`. |
| `FORMAT_VERSION` | Project format version written by this release. |

See [FILE_FORMAT.md](FILE_FORMAT.md) for the format itself.

## glTF import and export

Full details, including the axis conversion, the supported feature set and
safety rules, are in [GLTF.md](GLTF.md).

| Name | Description |
| --- | --- |
| `export_gltf(scene, path, *, allow_lossy=False)` | Write `.glb` or `.gltf` (embedded buffer). Returns `ExportResult`. Raises `UnsupportedFeatureError` (nothing written) for content glTF cannot store, unless `allow_lossy=True`. |
| `import_gltf(path, *, allow_lossy=False, limits=None)` | Read `.glb`/`.gltf`. Returns `ImportResult`. Raises `AssetFormatError`, `ResourceLimitError`, `UnsupportedFeatureError`, `FileNotFoundError`. |
| `ImportLimits(max_file_bytes, max_buffer_bytes, max_nodes, max_vertices, max_triangles)` | Safety limits for untrusted files (frozen dataclass with defaults). |

`ExportResult` has these fields:

- `path`, `format` (`"glb"`/`"gltf"`), `byte_length`;
- `node_count`, `mesh_count`;
- `nodes`: a list of `{"node", "id", "name", "mesh"}`;
- `diagnostics`.

`ImportResult` has these fields:

- `scene`, `source`, `format`, `generator`;
- `objects`: a list of `{"node", "id", "name", "id_source"}`, where
  `id_source` is `"extras"` or `"generated"`;
- `diagnostics`;
- `lossy`: true if anything was dropped or approximated.

Both results have `to_dict()`.

## Geometry inspection

Field and diagnostic reference: [INSPECTION.md](INSPECTION.md).

| Name | Description |
| --- | --- |
| `inspect_mesh(mesh, *, location=None)` | Returns a `MeshGeometryReport`. |
| `inspect_scene(scene)` | Returns a `SceneGeometryReport` (same as `scene.inspect_geometry()`). |
| `MeshGeometryReport` | `vertex_count`, `triangle_count`, `attributes`, `bounds`, `surface_area`, `uv_bounds`, `topology`, `diagnostics`, `status`, `to_dict()` |
| `SceneGeometryReport` | `scene_name`, `summary`, `world_bounds`, `objects`, `meshes`, `diagnostics`, `all_diagnostics`, `status`, `to_dict()` |
| `Diagnostic` | Frozen dataclass: `code`, `severity` (`"error"`/`"warning"`/`"info"`), `message`, `location`, `count`, `samples`, `lossy`; `to_dict()` |

## `aether3d.math3d`

Stateless helpers used by the engine, available for scripts:
`quat_from_euler`, `quat_to_euler`, `quat_from_axis_angle`, `quat_multiply`,
`quat_to_matrix3`, `matrix3_to_quat`, `compose_matrix`, `decompose_matrix`,
`transform_point`, plus the validators `vec3`, `scale3` and `quat`.

## Errors

All deliberate errors derive from `AetherError`.

| Error | Also a | Raised when |
| --- | --- | --- |
| `ValidationError` | `ValueError` | An argument is malformed or out of range |
| `ObjectNotFoundError` | `KeyError` | No object with that id/name in the scene |
| `NameConflictError` | `ValueError` | An explicit name is already used |
| `HierarchyError` | `ValueError` | An operation would create a cycle |
| `DetachedObjectError` | | The object was removed from its scene |
| `ProjectFormatError` | `ValueError` | A project file is invalid; `.location` names the offending element |
| `AssetFormatError` | `ValueError` | An external asset (glTF/GLB) is malformed; `.code` is a stable identifier and `.location` a JSON pointer such as `/accessors/3` |
| `UnsupportedFeatureError` | `ValueError` | A conversion cannot represent some content faithfully; `.diagnostics` lists every item |
| `ResourceLimitError` | `ValueError` | Input exceeds an `ImportLimits` limit |

Missing files raise the standard `FileNotFoundError`.

## Command line

```text
aether3d new PATH [--name NAME] [--force] [--json]
aether3d validate PATH [--allow-lossy] [--json]
aether3d inspect PATH [--allow-lossy] [--json]
aether3d geometry PATH [--allow-lossy] [--json]
aether3d import SOURCE.glb|.gltf DESTINATION.aether [--allow-lossy] [--force] [--json]
aether3d export SOURCE DESTINATION.glb|.gltf [--allow-lossy] [--force] [--json]
aether3d --version
```

`validate`, `inspect`, `geometry` and `export` read `.aether` projects as well
as `.glb`/`.gltf` files. For glTF input they add `source_format`, `generator`,
`lossy` and `import_diagnostics` to their JSON output (`.aether` input reports
`source_format: "aether"` and `format_version`).

With `--json`, each command prints exactly one JSON object to stdout:
`{"ok": true, "command": ..., ...}` on success, or
`{"ok": false, "command": ..., "error": {...}}` on failure. The `error`
object has:

- `type` and `message`;
- `location`, when known;
- `code`, for `AssetFormatError`;
- `diagnostics`, for `UnsupportedFeatureError`.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Failure |
| `2` | Usage error |
| `3` | `geometry` completed but found error-severity problems |

The `import`, `export` and `geometry` commands report:

- **`import --json`:** `ImportResult.to_dict()` plus `destination`.
- **`export --json`:** `ExportResult.to_dict()` plus the source fields.
- **`geometry --json`:** the scene geometry report under `geometry`.

`inspect --json` returns `scene` (`name`, `object_count`, `mesh_count`,
`roots`) and an `objects` array in depth-first order, where each entry has
`id`, `name`, `parent`, `children`, `depth`, `transform` (`position`,
`rotation`, `rotation_euler`, `scale`), `world_position` and `mesh`
(`vertex_count`, `triangle_count`, `bounds`, or `null`).
