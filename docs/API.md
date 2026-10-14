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
from aether3d import EditableMesh, EditSession, Selection, OperationResult, CommitResult
from aether3d import TopologyReport, run_edit_script
from aether3d import ProceduralSystem, ProceduralGeometry, Modifier, PolyMesh, Expr
from aether3d import ProceduralError, run_procedural_script
from aether3d import Camera, Light, Appearance, World, RenderSettings, RenderLimits
from aether3d import RenderResult, RenderError, render, run_scene_script
from aether3d import ShadowSettings, list_adapters, gpu_status, release_gpu
from aether3d import Material, MaterialLibrary, ImageTexture, ProceduralTexture
from aether3d import TextureBinding, TextureGraph, MaterialError
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
| `add_object(name=None, *, mesh=None, parent=None, position=None, rotation=None, scale=None, object_id=None, geometry=None)` | Add an object, optionally with a `Mesh` or a procedural `geometry` node (id, name or object), but not both. Returns the `SceneObject`. |
| `procedural` | The scene's `ProceduralSystem` (see [Procedural geometry](#procedural-geometry)). |
| `add_camera(name=None, *, projection, fov, fov_axis, ortho_scale, near, far, aspect_ratio, parent, position, rotation, look_at, up, active)` | Add a camera object (see [Rendering](#rendering)). It becomes active if none is, or if `active=True`. |
| `add_light(type="directional", name=None, *, color, intensity, range, falloff, parent, position, rotation, direction, look_at, up)`, `add_directional_light(...)`, `add_point_light(...)` | Add a light object. |
| `cameras`, `lights` | Tuples of camera and light objects. |
| `active_camera` | The default render camera, or `None` (settable with an object, id, name or `None`). |
| `world`, `render_settings` | `World` (background, ambient light) and `RenderSettings` (saved output settings). Edit them in place. |
| `materials` | The scene's `MaterialLibrary`: materials and textures (see [Materials and textures](#materials-and-textures)). |
| `create_material(name=None, *, model="standard", preset=None, **params)` | Create a material from parameters, or from a preset and its options. Returns the `Material`. |
| `assign_material(objects, material, *, group=None)` | Give one or more mesh objects (objects, ids or names) a material (object, id, name, or `None` to unassign), or only their face group `group`. |
| `add_image_texture(name=None, path=None, **options)`, `add_procedural_texture(name, graph)` | Add a PNG/JPEG texture (`data=`, `embed=`, `wrap`, `filter`, `mipmaps`, `color_space`) or a procedural texture. |
| `render(**options)` | Render with the CPU renderer; returns a `RenderResult` (see [Rendering](#rendering)). |
| `add_geometry(name, primitive=None, *, mesh=None, source=None, overrides=None, after=None, parameters=None, **params)` | Create a procedural geometry node: from a primitive (with `**params`), a stored `mesh`, or another node (`source`, with `overrides` and an optional `after` modifier). `parameters` declares node-local parameters as `{name: value}` or `{name: {"value": ..., "min": ...}}`. |
| `add_procedural(name=None, primitive="box", *, parent=None, position=None, rotation=None, scale=None, parameters=None, **params)` | Create a primitive node and an object that uses it, in one step. |
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
| `export(path, *, allow_lossy=False, bake_textures=None)` | Export to `.glb` or `.gltf` (by extension); returns an `ExportResult`. `bake_textures=SIZE` bakes UV-mapped procedural textures into PNG images. |
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
| `mesh` | A `Mesh` or `None`, settable. Meshes may be shared between objects. For a procedural object, this is the evaluated mesh, and assigning to it raises `ProceduralError("PROCEDURAL_OBJECT")`. |
| `geometry` | The `ProceduralGeometry` the object uses, or `None`. |
| `kind` | `"mesh"`, `"procedural"`, `"camera"`, `"light"` or `"empty"`. An object holds at most one of mesh, camera and light. |
| `camera`, `light` | The `Camera` or `Light` component, or `None`. Edit it in place. |
| `appearance`, `color` | `Appearance` (color, specular, shininess); `color` is a shortcut. While the object has a material, setting them raises `MaterialError("OBJECT_HAS_MATERIAL")`. |
| `material` | The object's `Material`, or `None` (it renders with its color). Settable with a material, id, name or `None`. |
| `group_materials`, `material_for_group(group)` | Materials of individual face groups (`{group: Material}`, a copy), and the material a group renders with. |
| `visible` | Whether the object renders (lights: whether they shine). Not inherited by children. |
| `look_at(target, *, up=(0, 0, 1))`, `look_along(direction, *, up)` | Turn the object's front (local −Y) toward a point or object, or along a direction. Exact under any parent. |
| `world_forward` | Unit world direction of the object's front (local −Y). |
| `set_world_position(position)` | Move the object to a world position, respecting parents. |
| `frame(targets=None, *, direction, up, margin, fit_clipping, aspect)` | Camera objects: place the camera so `targets` exactly fill the frame. |
| `is_procedural` | `True` if the object uses a geometry node. |
| `set_geometry(node_or_none)` | Use a geometry node; `None` detaches it, leaving no mesh. |
| `bake()` | Replace the procedural geometry with a static copy of the evaluated mesh, and return the mesh. |
| `transform` | The local `Transform`. Assigning a `Transform` copies it. |
| `position`, `rotation`, `rotation_euler`, `scale` | Shortcuts for the local transform's components. |
| `parent` | Parent object or `None`. |
| `children` | Tuple of direct children, in order. |
| `iter_descendants()` | Depth-first iterator over all descendants. |
| `edit_mesh(weld="exact", quads=True)` | Start an `EditSession` on this object's geometry (see [EDITING.md](EDITING.md)). Procedural objects raise `EditError("PROCEDURAL_OBJECT")`. |
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
Mesh(positions, triangles, normals=None, uvs=None, groups=None)
```

An immutable indexed triangle mesh. Inputs may be nested sequences or NumPy
arrays, `(N, 3)` or flat. Data is validated, copied, and stored read-only:

| Member | Description |
| --- | --- |
| `positions` | `(N, 3) float32` |
| `triangles` | `(M, 3) uint32` |
| `normals` | `(N, 3) float32` or `None` |
| `uvs` | `(N, 2) float32` or `None` |
| `groups` | `(M,) uint16` face group of every triangle (0–65535), or `None` (all in group 0) |
| `group_ids` | The face groups used, ascending (`(0,)` without groups) |
| `with_groups(groups)` | A copy sharing this mesh's arrays, with other face groups (`None` removes them) |
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
| `export_gltf(scene, path, *, allow_lossy=False, bake_textures=None)` | Write `.glb` or `.gltf` (embedded buffer), with materials and textures. Returns `ExportResult`. Raises `UnsupportedFeatureError` (nothing written) for content glTF cannot store, unless `allow_lossy=True`. `bake_textures` (16–8192) bakes UV-mapped procedural textures into PNG images. |
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

## Mesh editing

Full guide: [EDITING.md](EDITING.md).

| Name | Description |
| --- | --- |
| `EditSession` | Returned by `obj.edit_mesh()`. Delegates queries and operations to its `mesh`. Its methods are `commit(scope="object"\|"shared", allow_invalid=False) -> CommitResult` and `discard()`. Used as a context manager, it commits on success and discards on exception. |
| `CommitResult` | `mesh`, `objects` (updated object ids), `changed`, `validation` (`TopologyReport`), `inspection` (`MeshGeometryReport`), `diagnostics`, `operations`, `to_dict()` |
| `EditableMesh` | The polygon mesh. Construct with `EditableMesh(uvs=True, normals=True)` or `EditableMesh.from_mesh(mesh, weld="exact", quads=True)`; convert back with `to_mesh()` / `build_mesh(compute_normals=None)`. |
| `Selection` | Frozen sets `vertices`, `edges`, `faces`; `|`, `&`, `-`; `Selection.of(...)`; `to_dict()` |
| `OperationResult` | `operation`, `created`, `removed` (dicts of id lists by kind), `modified_faces`, `data`, `diagnostics`, `selection(key)`, `to_dict()` |
| `TopologyReport` | `summary`, `diagnostics`, `status`, `ok`, `codes()`, `to_dict()` |
| `run_edit_script(scene, script) -> ScriptResult` | Apply a declarative JSON edit script atomically |

**`EditableMesh` queries:**

- Counts and ids: `vertex_count`, `edge_count`, `face_count`, `vertex_ids()`,
  `edge_ids()`, `face_ids()`, `has_vertex`, `has_edge`, `has_face`.
- Positions: `position(v)`, `positions(ids)`, `world_positions(ids)`.
- Topology: `face_vertices(f)`, `face_edges(f)`, `edge_vertices(e)`,
  `edge_faces(e)`, `vertex_edges(v)`, `vertex_faces(v)`, `edge_between(a, b)`.
- Geometry: `face_normal(f)`, `face_center(f)`, `face_area(f)`,
  `edge_length(e)`, `edge_midpoint(e)`, `bounds(space=)`.
- Attributes and shading: `corner_values(f, layer)`, `is_smooth(f)`,
  `is_sharp(e)`, `corner_layers`, `has_uvs`, `has_normals`.
- Reports: `summary()`, `describe(kinds, limit)`, `validate()`.
- History: `history`, `undo()`, `redo()`, `can_undo`, `can_redo`.
- Discovery: `EditableMesh.operations()` lists every operation name.

**Selection:**

- Queries: `select(kind, ids=None, *, box, sphere, plane, normal, nearest,
  boundary, sides, predicate, test, space, expect)`, plus the shortcuts
  `select_vertices`, `select_edges`, `select_faces` and `select_all()`.
- Adjacency: `edge_loop(e)`, `edge_ring(e)`, `linked_faces(faces)`,
  `grow(sel)`, `region_boundary(faces)`, `boundary_edges()`,
  `non_manifold_edges()`.
- Conversions: `to_vertices`, `to_edges`, `to_faces`.

**Operations** (atomic and undoable; each returns an `OperationResult`):

- Creation: `add_box`, `add_grid`, `create_vertex`, `create_face`.
- Deletion: `delete_faces`, `delete_edges`, `delete_vertices`.
- Transforms: `translate`, `rotate`, `scale`, `transform`, `set_positions`,
  `move_along_normals`.
- Extrusion and bevel: `extrude_faces`, `inset_faces`, `bevel_edges`.
- Refinement: `subdivide_faces`, `loop_cut`.
- Merging: `merge_vertices`, `merge_by_distance`, `collapse_edges`.
- Dissolving: `dissolve_edges`, `join_triangles`.
- Splitting: `split_edges`, `split_edge`, `split_face`, `rip_vertices`,
  `separate_faces`.
- Triangulation: `triangulate_faces`.
- Orientation and shading: `flip_faces`, `set_smooth`, `set_sharp`.
- Face groups: `set_group(faces, group)`; query with `face_group(f)`,
  `has_groups` and `select(..., group=k)`. New faces inherit their source
  face's group.

## Procedural geometry

Full guide: [PROCEDURAL.md](PROCEDURAL.md).

**`ProceduralSystem`** (`scene.procedural`):

| Member | Description |
| --- | --- |
| `types()` (static) | Schemas of every primitive and modifier, the parameter types and the expression functions. |
| `parameters`, `rules`, `geometries` | Scene parameters (`{name: Parameter}`), rules, and geometry nodes. |
| `add_parameter(name, value, *, type=None, min=None, max=None, choices=None, description="", unit="", geometry=None)` | Add a parameter. `value` may be an `Expr`. |
| `set_parameter(name, value, *, geometry=None)`, `set_parameters(values, *, geometry=None)`, `remove_parameter(name, *, geometry=None)` | Change parameters. |
| `add_rule(name, expr, message="", *, geometry=None)`, `remove_rule(...)` | Boolean conditions that forbid invalid combinations. |
| `geometry(ref)` | A node by id or name. |
| `add_geometry(...)`, `remove_geometry(ref)`, `rename_geometry(ref, name)`, `set_source(ref, params)`, `set_source_mesh(ref, mesh)`, `convert_source_to_mesh(ref)` | Nodes. |
| `add_modifier(ref, type, *, name=None, enabled=True, index=None, params=None)`, `set_modifier(...)`, `remove_modifier(ref, m)`, `move_modifier(ref, m, index)` | Modifier stacks. A modifier is given by id, name, index or `Modifier`. |
| `assign(obj, ref_or_none)` | Set an object's geometry. |
| `evaluate(ref, *, upto=None, overrides=None) -> Evaluation` | `key`, `polygons` (`PolyMesh`), `diagnostics`, `mesh` (cached `Mesh`). |
| `mesh(ref)` | The evaluated `Mesh`. |
| `validate(ref=None)` | A per-node evaluation and topology report. Never raises for a geometry problem. |
| `info(ref=None, *, evaluate=False)` | A JSON-compatible description of the definitions, with resolved values. |
| `dependencies(ref)`, `dependents(ref)`, `parameter_dependencies(ref=None)` | The dependency graph. |
| `batch()` | Context manager: several changes as one validated, undoable transaction. |
| `undo()`, `redo()`, `can_undo`, `can_redo`, `revision` | History. |
| `cache_info`, `clear_cache()`, `cache_limit_bytes`, `stats` | The evaluation cache. |

**`ProceduralGeometry`** has `id`, `name`, `source`, `parameters`, `rules`,
`modifiers`, `users`, plus these methods:

- parameters and rules: `add_parameter`, `set_parameter`,
  `remove_parameter`, `add_rule`, `remove_rule`;
- the stack: `add_modifier(type, *, name, enabled, index, **params)`,
  `modifier(ref)`, `remove_modifier`, `move_modifier`;
- the source: `set_source(**params)`, `convert_source_to_mesh()`,
  `edit_source()` (an editing session on the stored base mesh);
- results: `evaluate(*, upto, overrides) -> Mesh`, `evaluate_polygons(...)`,
  `validate()`, `info(evaluate=False)`.

**`Modifier`** has `id`, `name`, `type`, `enabled` and `params` (raw values),
plus `set(*, enabled=None, name=None, **params)`, `resolved()` and `info()`.

**`Expr(source)`** is a safe expression over parameters (written
`{"expr": ...}` in JSON). **`PolyMesh`** is the immutable polygon mesh that
flows through modifier stacks; its methods include `validate()`,
`triangulate()`, `to_mesh()`, `from_mesh()`, `bounds()` and `content_hash()`.

**`run_procedural_script(scene, script)`** applies a declarative JSON
procedural script atomically, and raises `ScriptError` on failure.

## Rendering

Full guides: [RENDERING.md](RENDERING.md) (the CPU renderer, cameras, lights,
shading) and [GPU.md](GPU.md) (the GPU backend).

| Name | Description |
| --- | --- |
| `render(scene, *, camera=None, width=None, height=None, samples=None, exposure=None, cull=None, default_lights=None, frame=False, margin=0.05, limits=None, backend="cpu", adapter=None, msaa=None, supersample=None, shadows=None)` | Render the scene. Arguments override `scene.render_settings` for this render only. Same as `scene.render(...)`. `backend`: `"cpu"` (the NumPy rasterizer, the default), `"gpu"` (wgpu; `GPU_*` error if unavailable) or `"auto"` (GPU if it works, else CPU with a `BACKEND_FALLBACK` diagnostic). `adapter`: GPU adapter selection (index, `"discrete"`, `"vulkan"`, part of a name, ...). `msaa`, `supersample`: GPU antialiasing (default: `samples²` samples per pixel). `shadows`: GPU shadow maps (`True`, a dict or a `ShadowSettings`). |
| `RenderResult` | `image` (`(h, w, 3) uint8` sRGB); `linear`, `depth`, `coverage` and `object_index` maps; `objects`, `camera`, `settings`, `stats` (counts and timings; GPU: also `gpu_timings`, `draw_calls`), `diagnostics`, `backend` (`{"name": "cpu"}`, or the GPU adapter, API, antialiasing, shadow maps and resources), `output`, `status`, `width`, `height`, `pixel(x, y)`, `png_bytes()`, `save(path)` (atomic, `.png` only), `to_dict()`. |
| `ShadowSettings(lights=None, map_size=2048, bias=0.5, normal_offset=0.5, pcf_radius=1)` | GPU shadow-map settings for one render (not saved in projects). |
| `list_adapters(*, check=False)` | Graphics adapters wgpu can use, best first: `index`, `name`, `vendor`, `type`, `backend`, `driver`, `software`, `features`, `limits`, `recommended` (with `check=True`: `usable`, `msaa`). Raises `RenderError("GPU_UNAVAILABLE")` without wgpu. |
| `gpu_status(*, adapter=None, check=False)` | Never raises: `available`, `reason`, `hint`, `versions`, `adapters`, `selected` (adapter, features, limits, MSAA counts) and, with `check=True`, a verified test render (`check`). |
| `release_gpu(adapter=None)` | Free cached GPU devices and memory; returns how many. |
| `Camera(projection="perspective", fov=40, fov_axis="vertical", ortho_scale=10, near=0.1, far=1000, aspect_ratio=None)` | Validated camera parameters, with `configure(**changes)`, `half_extents(aspect)` and `projection_matrix(aspect)`. |
| `Light(type="directional", color=(1, 1, 1), intensity=1, range=None, falloff="inverse_square")` | Validated light parameters, with `configure(**changes)`. |
| `Appearance(color=(0.8, 0.8, 0.8), specular=0, shininess=32)` | Object color and highlight. |
| `World(background=(0.2, 0.2, 0.2), ambient_color=(1, 1, 1), ambient_intensity=0.1)` | Background and ambient light. |
| `RenderSettings(width=1280, height=720, samples=2, exposure=0, cull="none", default_lights=True)` | Saved output settings. |
| `RenderLimits(max_dimension=16384, max_samples=40_000_000, max_triangles=20_000_000, max_gpu_memory=4 GiB)` | Resource limits, checked before allocating. |
| `run_scene_script(scene, script)` | Apply a declarative `aether3d.scene-script` (cameras, lights, appearance, world, settings) atomically. |
| `aether3d.rendering.extract_render_scene(scene, camera)` | The backend-independent `RenderScene`. |
| `aether3d.rendering.encode_png(image)`, `write_png(path, image)` | Aether3D's PNG encoder. |
| `aether3d.rendering.srgb_to_linear`, `linear_to_srgb`, `parse_color`, `color_to_hex` | Color helpers. |
| `aether3d.math3d.look_rotation_matrix(direction, up)` | The world rotation used by `look_at`. |

Colors are display (sRGB) values: `"#rrggbb"`, `"#rgb"`, or three numbers in
[0, 1]. Every `Camera`, `Light`, `Appearance`, `World` and `RenderSettings`
value has `to_dict()`, `from_dict()` and `copy()`. Setting an invalid value
raises `RenderError`, and nothing changes.

## Materials and textures

Full guide: [MATERIALS.md](MATERIALS.md). Materials and textures belong to a
scene's `MaterialLibrary` (`scene.materials`); they are never constructed
directly.

| Name | Description |
| --- | --- |
| `MaterialLibrary` | `materials`, `textures` (tuples); `material(ref)`, `texture(ref)` (raise when missing), `find_material(ref)`, `find_texture(ref)` (or `None`); `library["name"]`; `create_material(...)`, `add_image_texture(...)`, `add_procedural_texture(...)`; `assign(objects, material, *, group=None)`; `users(material)` (`[(object, group or None)]`); `remove_material(ref, *, force=False)`, `remove_texture(ref, *, force=False)`; `convert_appearances(objects=None, *, dedupe=True)`; `transaction()` (all-or-nothing block); `validate()` (`Diagnostic` list); `inspect()`, `inspect_material(ref)`; `base_dir` (where relative texture paths resolve) |
| `Material` | `id`, `name` (settable), `model` (settable), `params` (a copy); `configure(**changes)` (all or nothing); `set_texture(slot, texture_or_path, **binding)`, `set_procedural_texture(slot, graph, *, name=None, **options)` (a node type name with its parameters as keywords, or a graph dict; keywords that are not node parameters are binding options), `set_texture_options(slot, **binding)`, `clear_texture(slot)`, `texture(slot)`, `textures` (bindings by slot); `linear(name)`; `inspect()`; `to_dict()` |
| `TextureBinding` | `texture` (id), `channel`, `coordinates`, `projection`, `offset`, `scale`, `rotation`, `blend`; `matrix` (the 3 × 4 transform), `channel_for(slot, kind)`, `to_dict()` |
| `ImageTexture` | `id`, `name`, `path`, `file`, `source` (`"file"`/`"embedded"`), `wrap`, `filter`, `mipmaps`, `color_space`; `configure(**settings)`; `embed()`, `extract(path)`, `embedded_bytes`, `mime_type`; `load()`, `status()`, `info()` |
| `ProceduralTexture` | `id`, `name`, `graph` (`TextureGraph`); `set_graph(graph)`, `set_params(node=None, **params)`; `evaluate(points)` (`(N, 4)` linear RGBA); `preview(width, height=None, *, region)` (8-bit sRGB array); `bake(path, width=1024, height=None)` (PNG); `info()` |
| `TextureGraph` | A validated node graph: `TextureGraph.from_dict(data)`, `to_dict()` |
| `MaterialError` | See [Errors](#errors) |

`aether3d.materials` also provides the schemas, `material_schema()` and
`node_types()`; the presets, `preset_names()` and `preset_info()`; the
decoder, `decode_image(data, *, limits=None)` with `ImageLimits` and
`jpeg_available()`; and the image cache, `IMAGE_CACHE` (`stats()`, `clear()`).

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
| `EditError` | `ValueError` | An editing operation cannot be applied; nothing changed. `.code` (stable), `.elements` (offending ids by kind), `.hint` |
| `ScriptError` | `ValueError` | An edit or procedural script is malformed, or a step failed. `.code`, `.location` (e.g. `edits[0].operations[3]`), `.details` |
| `RenderError` | `ValueError` | A camera, light, appearance, world or render setting is invalid, or rendering failed (including GPU problems: `GPU_UNAVAILABLE`, `GPU_ADAPTER_NOT_FOUND`, `GPU_DEVICE_LOST`, ...; see [GPU.md](GPU.md#errors-diagnostics-and-debugging)); nothing changed and no file was written. `.code` (stable), `.location`, `.hint`, `.details` |
| `ProceduralError` | `ValueError` | A procedural definition, parameter or evaluation is invalid; nothing changed. `.code` (stable), `.location` (e.g. `geometry[Tower].modifiers[Row].params.count`), `.hint`, `.details` |
| `MaterialError` | `ValueError` | A material, texture, graph or image is invalid or cannot be used (`INVALID_MATERIAL`, `TEXTURE_MISSING`, `IMAGE_TOO_LARGE`, `UNSAFE_PATH`, ...; see [MATERIALS.md](MATERIALS.md#errors-and-diagnostics)); nothing changed. `.code` (stable), `.location` (e.g. `materials[Stone].roughness`), `.hint`, `.details` |

Missing files raise the standard `FileNotFoundError`.

## Command line

```text
aether3d new PATH [--name NAME] [--force] [--json]
aether3d validate PATH [--allow-lossy] [--json]
aether3d inspect PATH [--allow-lossy] [--json]
aether3d geometry PATH [--allow-lossy] [--json]
aether3d import SOURCE.glb|.gltf DESTINATION.aether [--allow-lossy] [--force] [--json]
aether3d export SOURCE DESTINATION.glb|.gltf [--allow-lossy] [--bake-textures SIZE] [--force] [--json]
aether3d topology PATH --object ID_OR_NAME [--elements] [--limit N] [--triangles] [--json]
aether3d edit PROJECT.aether SCRIPT.json|- [--output PATH] [--dry-run] [--json]
aether3d procedural types [--json]
aether3d procedural inspect PROJECT.aether [--geometry REF] [--evaluate] [--json]
aether3d procedural validate PROJECT.aether [--geometry REF] [--json]
aether3d procedural run PROJECT.aether SCRIPT.json|- [--output PATH] [--dry-run] [--json]
aether3d procedural set PROJECT.aether NAME=VALUE... [--geometry REF] [--output PATH] [--dry-run] [--json]
aether3d procedural bake PROJECT.aether (--object REF... | --all) [--output PATH] [--dry-run] [--json]
aether3d render PATH [-o OUT.png] [--width W] [--height H] [--camera REF] [--samples N] [--exposure EV] [--cull none|back|front] [--background COLOR] [--no-default-lights] [--frame] [--margin M] [--force] [--json]
    [--backend cpu|gpu|auto] [--adapter SPEC] [--msaa N] [--supersample N]
    [--shadows [--shadow-lights LIGHT...] [--shadow-map-size N] [--shadow-bias B] [--shadow-normal-offset X] [--shadow-pcf-radius R]]
aether3d gpu info [--adapter SPEC] [--check] [--json]
aether3d camera add|set|frame|activate|list PROJECT ... [--output PATH] [--dry-run] [--json]
aether3d light add|set|list PROJECT ... [--output PATH] [--dry-run] [--json]
aether3d object set|remove PROJECT OBJECT ... [--output PATH] [--dry-run] [--json]
aether3d scene set PROJECT ... | scene run PROJECT SCRIPT.json|- [--output PATH] [--dry-run] [--json]
aether3d material list|presets|schema|create|set|assign|texture|inspect|validate|remove|convert-colors|preview ...
aether3d texture list|add|procedural|set|inspect|preview|remove ...
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
- `code`, for `AssetFormatError`, `ScriptError`, `ProceduralError`, `RenderError` and
  `MaterialError` (with `hint` and `details` when available);
- `diagnostics`, for `UnsupportedFeatureError`.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Failure |
| `2` | Usage error |
| `3` | `geometry`, `topology`, `procedural validate` or `material validate` completed but found error-severity problems, or `gpu info --check` found that GPU rendering does not work |

The `import`, `export` and `geometry` commands report:

- **`import --json`:** `ImportResult.to_dict()` plus `destination`.
- **`export --json`:** `ExportResult.to_dict()` plus the source fields.
- **`geometry --json`:** the scene geometry report under `geometry`.

`inspect --json` returns `scene` (`name`, `object_count`, `mesh_count`,
`roots`) and an `objects` array in depth-first order, where each entry has
`id`, `name`, `parent`, `children`, `depth`, `transform` (`position`,
`rotation`, `rotation_euler`, `scale`), `world_position`, `mesh`
(`vertex_count`, `triangle_count`, `bounds`, or `null`) and `geometry` (the
procedural node's `id` and `name`, or `null`).
