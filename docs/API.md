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
from aether3d import PathSettings, AOSettings, Environment, Atmosphere
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
| `world`, `render_settings` | `World` (background, ambient light, environment, outdoor atmosphere) and `RenderSettings` (saved output settings). Edit them in place. |
| `worlds` | The scene's `WorldSystem`: procedural worlds (see [Procedural worlds](#procedural-worlds); since 0.18.0). |
| `materials` | The scene's `MaterialLibrary`: materials and textures (see [Materials and textures](#materials-and-textures)). |
| `cad` | The scene's `CadDocument`: precision models, buildings and drawings (see [Precision modeling and buildings](#precision-modeling-and-buildings)). Since 0.15.0. |
| `create_material(name=None, *, model="standard", preset=None, **params)` | Create a material from parameters, or from a preset and its options. Returns the `Material`. |
| `assign_material(objects, material, *, group=None)` | Give one or more mesh objects (objects, ids or names) a material (object, id, name, or `None` to unassign), or only their face group `group`. |
| `add_image_texture(name=None, path=None, **options)`, `add_procedural_texture(name, graph)` | Add a PNG/JPEG texture (`data=`, `embed=`, `wrap`, `filter`, `mipmaps`, `color_space`) or a procedural texture. |
| `render(**options)` | Render (the CPU rasterizer by default; `backend="gpu"`, `"auto"`, `"path"` or `"gpu_path"`); returns a `RenderResult` (see [Rendering](#rendering)). |
| `add_geometry(name, primitive=None, *, mesh=None, source=None, overrides=None, after=None, parameters=None, **params)` | Create a procedural geometry node: from a primitive (with `**params`), a stored `mesh`, or another node (`source`, with `overrides` and an optional `after` modifier). `parameters` declares node-local parameters as `{name: value}` or `{name: {"value": ..., "min": ...}}`. |
| `add_procedural(name=None, primitive="box", *, parent=None, position=None, rotation=None, scale=None, parameters=None, **params)` | Create a primitive node and an object that uses it, in one step. |
| `add_cube(name=None, *, size=1.0, parent=None, position=None, rotation=None, scale=None)` | Add a cube with edge length `size`. |
| `add_sphere(name=None, *, radius=1.0, segments=32, rings=16, parent=None, position=None, rotation=None, scale=None)` | Add a UV sphere. |
| `add_plane(name=None, *, size=1.0, parent=None, position=None, rotation=None, scale=None)` | Add a square plane in the XY plane facing +Z. |
| `get(object_id)` | Object by id; raises `ObjectNotFoundError`. |
| `get_by_name(name)` | Object by name; raises `ObjectNotFoundError`. |
| `remove(obj_or_id, *, recursive=True)` | Remove an object. With `recursive=False`, its children move up to its parent and keep their world placement. |
| `duplicate(obj, name=None, *, parent="same", children=True)` | Copy an object (and its descendants) in one undoable step: new ids, free names, shared meshes and geometry nodes. Returns the top copy. |
| `merge(source, *, parent=None, allow_lossy=False)` | Copy another `Scene`, `.aether` project or `.glb`/`.gltf` file into this one, in one undoable step, with its materials and textures (procedural objects as static meshes). Returns the copies of its roots. |
| `history` | The scene's `History`: undo, redo, transactions, checkpoints (see [History](#history)). |
| `transaction(label="Transaction")` | `with scene.transaction("..."):` -- the block's changes are one undo step, all or nothing. |
| `revision` | Increases with every change, undo and redo included. |
| `roots` | Tuple of root objects, in order. |
| `objects` | Tuple of all objects in depth-first order (parents before children). |
| `len(scene)`, `iter(scene)`, `x in scene` | Object count, depth-first iteration, membership (object or id). |
| `save(path, *, history=False)` | Write a `.aether` file (atomically); with `history=True` its undo history too (`<path>.history`). |
| `Scene.load(path, *, history=False)` | Read and validate a `.aether` file; with `history=True` restore and check the history saved with it. |
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

**Atomicity.** A call that raises leaves the scene unchanged: operations validate
their inputs first, and a change that fails part-way is rolled back.

**History.** Every change is recorded in `scene.history` and can be undone (see
[History](#history) and [HISTORY.md](HISTORY.md)).

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
| `tags`, `add_tags(*tags)`, `remove_tags(*tags)`, `has_tag(tag)` | The object's tags: a sorted tuple of unique strings (1–64 characters, at most 64), settable with any iterable of strings. Each change is an undo step (see [SPATIAL.md](SPATIAL.md#tags-and-metadata)). |
| `metadata`, `set_metadata(key, value)`, `remove_metadata(key)` | Free-form JSON data (at most 16 KiB): `metadata` returns a copy and is settable with a dict. |
| `bounds(*, space="world", children=False, include_hidden=False)` | Bounding box, oriented box and sphere of the object's geometry (and its descendants'), or `None` (see [SPATIAL.md](SPATIAL.md#bounds)). |
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
Mesh(positions, triangles, normals=None, uvs=None, groups=None, colors=None)
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
| `colors` | `(N, 4) float32` linear RGBA vertex colours in [0, 1] (`(N, 3)` input gets alpha 1), or `None`; like glTF's `COLOR_0` they multiply the base colour and opacity (since 0.18.0) |
| `with_groups(groups)` | A copy sharing this mesh's arrays, with other face groups (`None` removes them) |
| `with_colors(colors)` | A copy sharing this mesh's arrays, with other vertex colours (`None` removes them) |
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
| `export_gltf(scene, path, *, allow_lossy=False, bake_textures=None, animation="auto", fps=None, position_tolerance=1e-4, rotation_tolerance=0.01, weight_tolerance=1e-4)` | Write `.glb` or `.gltf` (embedded buffer), with materials and textures, and (0.17.0) skeletons, skins, morph targets and animations (`animation`: `auto`, `clips`, `actions`, `bake`, `none`; tolerances for fitted channels; `fps` for baking). Returns `ExportResult`. Raises `UnsupportedFeatureError` (nothing written) for content glTF cannot store, unless `allow_lossy=True`. `bake_textures` (16–8192) bakes UV-mapped procedural textures into PNG images. `scene.export(path, **options)` takes the same options. |
| `import_gltf(path, *, allow_lossy=False, limits=None)` | Read `.glb`/`.gltf`. Returns `ImportResult`. Raises `AssetFormatError`, `ResourceLimitError`, `UnsupportedFeatureError`, `FileNotFoundError`. |
| `ImportLimits(max_file_bytes, max_buffer_bytes, max_nodes, max_vertices, max_triangles)` | Safety limits for untrusted files (frozen dataclass with defaults). |

`ExportResult` has these fields:

- `path`, `format` (`"glb"`/`"gltf"`), `byte_length`;
- `node_count`, `mesh_count`;
- `nodes`: a list of `{"node", "id", "name", "mesh"}`;
- `animations`: per glTF animation, its name, channels, fitted channels and largest
  errors (0.17.0); `skins`: how many skins were written;
- `diagnostics`.

`ImportResult` has these fields:

- `scene`, `source`, `format`, `generator`;
- `objects`: a list of `{"node", "id", "name", "id_source"}`, where
  `id_source` is `"extras"` or `"generated"`;
- `animation`: `{rigs: [{armature, joints}], skins, morph_targets, animations: [{name,
  clips}]}` (0.17.0);
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
- Advanced modeling (0.13, [MODELING.md](MODELING.md)): `bevel_edges(edges, width,
  segments=1, profile=0.5)`; `bisect(point, normal, faces=, clear=, fill=, threshold=,
  space=)`; `knife_project(points, direction, faces=, closed=, through=, space=)`;
  `fill_holes(edges=None, mode=, max_sides=)`; `bridge_edge_loops(edges, segments=,
  twist=)`; `slide_edges(edges, factor, even=, correct_uvs=)`, `slide_vertex(vertex,
  toward, factor)`; `dissolve_vertices`, `dissolve_faces`, `limited_dissolve(angle=,
  faces=, delimit=)`; `connect_vertex_path`, `connect_vertices`; `extrude_edges(edges,
  offset)`, `spin(faces, angle, steps, axis=, center=, merge=)`, `add_lathe(profile,
  segments=, angle=, smooth=)`; `shrinkwrap(vertices, target, mode=, direction=, offset=,
  max_distance=)`; `set_auto_smooth(angle=)`, `recalculate_normals(inside=)`,
  `set_weighted_normals(mode=, weight=)`; `cleanup(...)` with a report.

## Sculpting

Brush strokes, masks, dynamic topology, remeshing and symmetrize on a sculpt mesh, with
validity tiers and JSON scripts ([MODELING.md](MODELING.md)). Since 0.13.0.

| Name | Description |
| --- | --- |
| `obj.sculpt() -> SculptSession` | A session on an object's mesh (procedural objects: `PROCEDURAL_OBJECT`). Delegates to its `mesh` (`s.stroke(...)`); `commit(scope="object"\|"shared", close=True, check=True, require=None) -> SculptCommit`, `discard()`, `closed`, `result`. A context manager: commits on success, discards on exception. One commit is one step of `scene.history`. |
| `geometry.sculpt_source() -> SourceSculptSession` | The same on a procedural geometry's stored base mesh (its modifiers stay); `commit(check=True, require=None)` |
| `SculptCommit` | `mesh`, `objects`, `changed`, `report` (`MeshReport`), `operations`, `diagnostics`, `to_dict()` |
| `SculptMesh` | `SculptMesh.from_mesh(mesh)`, `to_mesh()`; operations (atomic, undoable, each returns a `SculptResult`): `stroke(stroke)`, `set_mask(value=1, *, vertices, sphere, box, groups, mode)`, `invert_mask()`, `clear_mask()`, `remesh(target_length, *, iterations=5, flips="valence", relax=0.5, project=True)`, `voxel_remesh(voxel_size, *, mode="solid", thickness=None, max_voxels=None)`, `symmetrize(axis="x", *, direction="+", mode="auto", tolerance=None)`; `undo()`, `redo()`, `history`, `history_nbytes`, `can_undo`, `can_redo`; queries `vertex_count`, `triangle_count`, `positions()`, `triangles()`, `mask()`, `bounds()`, `vertex_normals(ids)`, `raycast(origin, direction)`, `nearest_point(point)`, `vertices_in_sphere(center, radius)`; `world_matrix`; `should_stop` (a callable polled between dabs, passes and voxel batches: true stops with `CANCELLED`, rolled back); `max_history`, `max_history_bytes` |
| `SculptResult` | `operation`, `data`, `timings`, `diagnostics`, `to_dict()` |
| `Brush`, `Brush.from_dict(d)`, `KINDS`, `FALLOFFS` | brush settings (validated): `kind`, `radius` (world units), `strength`, `direction`, `falloff`, `hardness`, `spacing`, `normal`, `normal_radius`, `falloff_shape`, `geodesic`, `front_faces_only`, `automask`, `use_mask`, `symmetry`, `pressure`, `smooth_method`, `poisson_ratio`, `dyntopo` |
| `Stroke`, `Stroke.from_dict(d)` | a brush and its path (`path`, `space`, `project`, `view`, `resample`); `to_dict()` |
| `check_mesh(positions, triangles, *, self_intersections=False, welded=False) -> MeshReport` | validity: `summary` (counts, flags, `tier`, volume, edge lengths, smallest angle), `diagnostics` (stable codes), `not_evaluated`, `status`, `ok`, `to_dict()`; `TIERS` = `("valid", "manifold", "closed", "solid")` |
| `aether3d.run_sculpt_script(scene, script, *, label=None)` | apply an `aether3d.sculpt-script` atomically (one undo step); `aether3d.sculpt.check_script(script)`; `aether3d.sculpt.schema.sculpt_script_schema()` |

## UV mapping, texture painting and baking

UVs, painted textures, brushes and bakes ([TEXTURING.md](TEXTURING.md)). Since 0.14.0.

| Name | Description |
| --- | --- |
| `obj.unwrap(**options) -> dict` | new UVs (one undo step): `seams`, `split_by` (`"hard"`, `"groups"`, `"seams"`), `angle_limit=66`, `method="isometric"\|"conformal"`, `max_stretch=1.15`, `iterations=12`, `faces`, `pack=True`, `resolution=1024`, `padding=4`, `rotate=True`, `texels_per_unit`; returns the report (`charts`, `splits`, stretch, timings, `diagnostics`) |
| `obj.project_uvs(kind, **options)` | `"planar"`, `"box"`, `"cylindrical"`, `"spherical"` (`axis`, `center`, `pack`, `resolution`, `padding`, `rotate`, `texels_per_unit`) |
| `obj.edit_uvs(operation, **options)` | `"transform"`, `"align"`, `"normalize"`, `"texel_density"`, `"pack"`, `"stitch"` (island operations; one undo step) |
| `obj.uv_report(resolution=1024, islands_limit=64) -> dict` | UV diagnostics (read-only) |
| `aether3d.uv`: `unwrap(mesh, ...) -> UnwrapResult`, `project`, `transform_islands`, `align_islands`, `normalize_islands`, `set_texel_density`, `pack_islands`, `stitch`, `uv_report(mesh, ...)`, `layout_image(mesh, size=1024, mode="islands", texture=None, edges=True)`, `PROJECTIONS` | the same on meshes, returning new meshes (`UnwrapResult`: `mesh`, `face_chart` (the island of every triangle), `report`, `diagnostics`) |
| `aether3d.uv.mikktspace.mesh_tangents(mesh)` | MikkTSpace tangents `(T, 3, 4)` (xyz, sign) of every corner, cached on the mesh |
| `scene.add_painted_texture(name, width=1024, height=None, *, encoding="color", background=None, image=None, **sampler) -> PaintedTexture` | a texture you paint (one undo step) |
| `PaintedTexture` | `width`, `height`, `encoding`, `background`, `layers`, `revision`; `add_layer(name=None, *, kind="paint"\|"fill", blend, opacity, visible, value, index)`, `layer(ref)`, `remove_layer`, `move_layer`, `set_layer(ref, **changes)`, `clear_layer`, `set_mask(ref, mask)`, `write_pixels(ref, pixels, x=0, y=0)`, `replace_tiles`, `replace_masks`, `flatten()`, `set_background`; `read()`, `read_layer(ref)`, `read_mask(ref)`, `pixels16()`, `pixels8()`, `export_png(path, bits=8)`, `info()`; `BLEND_MODES`, `ENCODINGS`, `LAYER_KINDS` |
| `obj.paint(targets, brush, points, **options) -> dict` | one stroke (`aether3d.texturing.stroke`): `space="surface"\|"view"\|"uv"`, `coordinates="world"\|"object"`, `pressure`, `view` (`PaintView.render(scene, camera, width, height)`), `padding=2`, `label`, `should_stop`; one undo step; returns `dabs`, `texels`, `tiles`, `skipped_dabs`, `seconds` |
| `Brush` (`aether3d.texturing`) | `kind` (`BRUSH_KINDS`: paint, erase, blur, smudge, clone, mask), `radius`, `strength`, `opacity`, `hardness`, `falloff`, `spacing`, `color`, `blend`, `normal_limit`, `backfaces`, `symmetry`, `clone_offset`, `blur_radius`; `Brush.from_value(dict)` |
| `obj.fill_texture(texture, value, *, where=None, opacity=1, blend_mode="normal", padding=2)`; `aether3d.texturing.fill`, `gradient(scene, obj, target, start, end, colors, *, coordinates="world", opacity=1)` | fills and gradients (one undo step each) |
| `scene.bake(target, sources=None, maps=("normal",), *, apply=False, **options) -> BakeResult` | `resolution`, `samples=2`, `cage`, `depth`, `cage_mesh`, `padding=8`, `ao_rays=64`, `ao_distance`, `thickness_rays=32`, `thickness_distance`, `curvature_radius`, `height_range`, `seed=0`, `backend="auto"\|"cpu"\|"gpu"`, `adapter`, `should_stop`, `progress`; `MAPS` lists the maps |
| `BakeResult` | `maps` (`{name: (H, W, 4)}`), `covered`, `hit`, `report`, `diagnostics`; `to_texture(scene, map, name=None)`, `save_png(map, path, bits=8)`; `aether3d.texturing.apply_bake(scene, target, result, material=None)` |
| `scene.transfer_textures(source, target, maps=None, **options) -> dict` | unwrap the target if needed, bake the source's material and bind it (one undo step) |
| `aether3d.texturing.run_texture_script(scene, script, *, label=None, should_stop=None)` | apply an `aether3d.texture-script` atomically; `aether3d.texturing.script.check_script(script)`; `aether3d.texturing.schema.texture_script_schema()` |

## Precision modeling and buildings

Exact parametric models, buildings and drawings ([CAD.md](CAD.md),
[BUILDINGS.md](BUILDINGS.md)). Since 0.15.0. Every changing call is one undo step;
lengths are in the document's unit or quantities (`"80 mm"`).

| Name | Description |
| --- | --- |
| `doc = scene.cad` | the `CadDocument` |
| `doc.set_units(length=None, angle=None)`, `doc.length_unit`, `doc.settings`, `doc.set_display_tolerance(chord, angle_deg, relative)` | units and display tolerance |
| `doc.parameter(name, value, *, dimension="length") -> Item` | create or change a parameter (`"=expr"` allowed) |
| `doc.add(kind, name, definition) -> Item`, `doc.set_definition(item, definition)`, `doc.update(item, changes)`, `doc.rename(item, name)`, `doc.remove(item)` | items by JSON definition (`KINDS`: parameter, frame, sketch, part, assembly, building, floor, slab, roof, stair, columns, beams, drawing) |
| `doc.items(kind=None)`, `doc.item(ref, kind=None)`, `doc.result(item)`, `doc.info(item=None)`, `doc.stats` | items, results (regenerated on demand), reports, regeneration counters |
| `with doc.batch(label, *, deadline=None, should_stop=None)` | several changes as one step, regenerated once |
| `doc.sketch(name, plane="XY") -> SketchHandle` | `point`, `line`, `polyline`, `rectangle`, `circle`, `arc`, `spline`, `constrain(kind, *refs, **options)`, `dimension(kind, *refs, value, driving=True)`, `set_value(id, value)`, `remove(*ids)`, `measure(id)`, `result` |
| `doc.part(name, **options) -> PartHandle` | `extrude`, `revolve`, `hole`, `fillet`, `chamfer`, `shell`, `pattern`, `mirror`, `add_feature(dict)`, `set(id, **changes)`, `faces()`, `measure(body="main")`, `body(name)`, `result` |
| `doc.building(name, *, levels, **options) -> BuildingHandle` | `floor(level) -> FloorHandle` (`wall`, `walls`, `walls_from_sketch`, `door`, `window`, `opening`, `room`, `set`, `remove`, `quantities()`), `slab`, `roof`, `stair`, `columns`, `beams`, `set`, `set_level`, `schedule(kind)` |
| `doc.drawing(name, view, items=None, **options) -> Item` | a drawing item; `doc.result(name)` is a `Drawing` (`to_svg()`, `to_dxf(unit)`, `to_png(dpi)`, `summary()`) |
| `doc.show(item, *, name=None) -> SceneObject`, `doc.unlink(obj)`, `doc.convert_to_mesh(obj)` | scene objects that follow an item |
| `doc.measure(a, b=None) -> dict`, `doc.snap(sketch, x, y, *, radius, kinds, grid)` | exact measurements; sketch snapping |
| `doc.export_step(items=None, units=None) -> str`, `doc.import_step(text, name=None)` | STEP AP214 |
| `doc.export_ifc(building=None, name=...) -> (str, counts)`, `doc.import_ifc(text, create_parts=True)` | IFC4 |
| `doc.import_dxf(text, name, *, plane="XY", layers=None, unit=None)`, `doc.export_sketch_dxf(sketch, unit=None)` | DXF |
| `aether3d.cad.script.run_cad_script(scene, script, *, label=None, should_stop=None)` | apply an `aether3d.cad-script` atomically; `check_script(script)`; `aether3d.cad.schema.cad_script_schema()` |
| `aether3d.cad.brep`: `build` (`box`, `cylinder`, `sphere`, `cone`, `torus`, `extrude`, `revolve`), `topology` (`Body`, `validate`, `merge`), `tessellate` (`tessellate`, `max_deviation`), `properties` (`mass_properties`), `boolean` (`boolean`) | the kernel |
| `aether3d.cad.assembly.clashes(doc, assembly)`, `aether3d.arch.quantities.schedule(doc, kind)`, `to_csv(rows)` | clashes; schedules |

## Animation, rigging and cinematics

Keyframes, curves, clips, layers, rigs, skins, IK, constraints, generators and cameras
([ANIMATION.md](ANIMATION.md)). Since 0.17.0. Every changing call is one undo step;
times are seconds; targets are objects (or ids, names), materials as `{"material": ...}`.

| Name | Description |
| --- | --- |
| `anim = scene.animation` | the `AnimationSystem` |
| `anim.set_timeline(fps=, start=, end=)`, `anim.fps` (`FrameRate`: `frame_time`, `frame_at`, `timecode`), `anim.start`, `anim.end`, `anim.add_marker(name, time, data=None)`, `anim.set_markers`, `anim.set_sections`, `anim.time` (the playhead: runtime state), `anim.frame_time(n)` | the timeline |
| `anim.keyframe(target, path, time, value, **key)`, `anim.set_channel(target, path, keys_or_curve, **options)`, `anim.channel(target, path)`, `anim.edit_keys(target, path, op, *, clip=None, **args)`, `anim.set_modifiers(target, path, modifiers, *, clip=None)` | keys and curves (paths: [ANIMATION.md](ANIMATION.md#animatable-properties)) |
| `anim.add_clip(name, clip=None)`, `anim.clip(ref)`, `anim.clips()`, `anim.remove_clip(ref)`; a clip handle: `set_channel`, `keyframe`, `edit_keys`, `set_events`, `set_range`, `rename`, `data`, `to_json()` | clips |
| `anim.set_action(target, clip)`, `anim.action(target)`, `anim.binding(target)`, `anim.set_binding(target, data)`, `anim.add_layer(target, name, *, mode, weight, mask, index)`, `anim.update_layer`, `anim.remove_layer`, `anim.add_strip(target, layer, clip, **options)`, `anim.update_strip`, `anim.remove_strip` | bindings, layers and strips |
| `anim.evaluate(time=None) -> AnimationPose`, `anim.overlay(time=None)`, `anim.rigging(time=None)`, `anim.sample(target, path, times, *, order=0)`, `anim.apply_pose(time=None)` | evaluation (pure, cached per revision and time); `apply_pose` writes a pose into authored state |
| `anim.add_rig(armature, joints, **options)`, `anim.rig(armature)`, `anim.set_rig`, `anim.attach(obj, armature, joint)`, `anim.detach`, `anim.skin(mesh_object, armature, *, weights="envelope", joints=None, influences=4, method="linear")`, `anim.unbind`, `anim.set_morph_targets(obj, targets)`, `anim.morph_targets`, `anim.pose_joints(armature, time, space)` | rigs, attachments, skins, morph targets |
| `anim.add_constraint(owner, constraint)`, `anim.set_constraints`, `anim.constraints(owner)`, `anim.remove_constraint` | constraints and IK |
| `anim.add_path(name, points=None, **options)`, `anim.path(ref)`, `anim.follow_path(obj, path, *, start, end, offsets, easing, follow, axis, up, loop, name)`, `anim.camera_move(camera, move, amount, start, end)`, `anim.camera_orbit(camera, target, start, end, degrees=)`, `anim.follow_camera(camera, target, **options)`, `anim.set_shots(shots)`, `anim.add_shot`, `anim.shots()` | cinematics |
| `anim.generate(armature, motion, **parameters)` | motion generators: `idle`, `walk`, `run`, `jump`, `land`, `wave`, `head_turn` (returns a report: clip, keys, errors, speed, ...) |
| `anim.mirror_clip(clip, armature, name=)`, `anim.extract_root_motion(clip, joint, mode=, yaw=)`, `anim.retarget(source, target, clip, mapping=)`, `anim.foot_contacts(armature, feet, **options)`, `anim.plant_feet(armature, legs, **options)` | motion editing and measurement |
| `anim.curve_report(target, path, clip=None)`, `anim.trajectory(target, start=, end=, joint=)`, `anim.diagnose()`, `anim.simplify(target, path, tolerance)`, `anim.resample(target, path, fps=)`, `anim.bake(targets, ...)`, `anim.graph(channels, format="svg"\|"png", ...)` | measuring, fitting and graphs |
| `aether3d.animation.frames.render_frames(scene, directory, ...)`, `contact_sheet(images, columns=, labels=)`, `assemble_video(...)` | frames, sheets, video |
| `aether3d.animation.FCurve`, `.tracks.RotationTrack`, `.clip.Clip`, `.time.FrameRate`, `.rotation`, `.fitting.fit_samples` | the value types (immutable) |
| `aether3d.animation.script.run_animation_script(scene, script, ...)`, `check_script(script)`, `aether3d.animation.schema.animation_script_schema()` | animation scripts |

## Character rigging

Automatic rigs, weights, deformation, controls, diagnostics and rigid-part characters
([CHARACTER_RIGGING.md](CHARACTER_RIGGING.md)). Since 0.19.0, on `scene.animation`; every
changing call is one undo step.

| Name | Description |
| --- | --- |
| `anim.detect_landmarks(meshes, *, template, up, forward, resolution=128, symmetry, landmarks)` | landmarks of unrigged meshes: positions, confidences, methods, the facing, the pose, diagnostics (read-only) |
| `anim.autorig(meshes, *, armature, name, template="humanoid", landmarks, up, forward, resolution, symmetry, orientation="bone", twist=True, controls=True, bind=True, parts, weights, skinning="linear", enforce_limits="ik", dry_run=False)` | a template rig with limits, helpers and controls, every mesh bound as its part mode says (`geodesic`, `transfer`, `rigid`, `attach`, `skip`) |
| `anim.edit_rig(armature, operations, *, skins="rebind", animation="keep", channels="keep", tolerance, dry_run)`; `add_joint`, `remove_joint`, `move_joint`, `orient_joints`, `rename_joint`, `reparent_joint`, `mirror_joint_chain` | structural editing with dependent data kept consistent |
| `anim.generate_weights(mesh, armature, *, method="heat", policy="organic", influences=None, diffusion=None, visibility=True, normals=None, corrective="keep", ...)`, `anim.edit_weights(mesh, operations, *, dry_run)`, `anim.normalize_weights(mesh)`, `anim.transfer_weights(target, source, *, max_distance, influences=None, unmatched, smooth, skinning)`, `anim.inspect_weights(mesh, *, joint)` | skin weights: generated (bone heat with up to eight influences by default, 0.19.1; `geodesic`, `envelope`, `nearest`), edited, transferred (the source's influence count by default), measured; `aether3d.animation.heat.heat_weights(positions, triangles, heads, ends, ...)` |
| `anim.skin(mesh, armature, *, method="linear" \| "dual_quaternion" \| "blended", dq=..., normals="joints" \| "surface", smooth=None \| {iterations, strength, region, weights})`; `aether3d.animation.rig.skin_dual_quaternion(...)`, `dual_quaternion_palette(...)`, `mush_rest(...)`, `delta_mush(positions, rest)`, `surface_normals(...)` | skinning methods, surface-following normals and corrective smoothing (native kernels when installed) |
| `anim.add_control(armature, name, kind, ...)`, `anim.ik_fk(armature, limb, "ik" \| "fk", time, *, blend=0)`, `anim.pose_character(armature, commands, *, time=0)` | controls, matched IK/FK switches, pose commands (`reach`, `plant_feet`, `look_at`, `raise_arm` with the clavicle's share, `lean`, `crouch`, `aim_hips`, `rotate`, `curl_fingers`, `roll_foot`; `aether3d.animation.controls.shoulder_share(angle, shoulder)`) |
| `anim.diagnose_rig(armature, *, poses="suite" \| "stress" \| [...], time, animation=True, max_findings=50)`, `anim.propose_rig_fix(armature, ...)`, `anim.apply_rig_fix(armature, fixes, *, dry_run, verify, poses, animation)`, `anim.rig_preview(armature, *, mode, joint, pose, time, view, width, height, backend, path)` | measuring (kept size, creases, folds, stretch), verified correction (`changes` per pose), pictures; `aether3d.animation.rig_report.pose_suite(rig, level="suite" \| "stress")`, `resolve_pose(rig, spec)`, `edge_pairs(triangles)`, `dihedrals(positions, triangles, pairs)` |
| `anim.rigid_parts(root, *, parts, forward)`, `anim.generate_rigid(root, motion, *, parts, name, forward, start, travel, tolerance, bind, **parameters)`, `anim.convert_character(source, "skeleton" \| "rigid", *, parts, forward, name, hide_original, start, end, tolerance)` | rigid-part characters with no bones, and optional conversions |
| `aether3d.animation.limits` (`check_limits`, `angles`, `clamp`, `relative`, `twist_angle`), `aether3d.animation.rigging.constraint_order(rig)` | swing-twist limits; the order joint constraints run in |
| `aether3d.sculpt.implicit.implicit_surface(shapes, voxel_size, ...)`, `humanoid_shapes(**options)`, `quadruped_shapes(...)` | smooth implicit bodies (closed meshes) |
| `aether3d.sculpt.figure.figure_mesh(voxel_size=0.006, *, remesh=None, **options)`, `figure_shapes(**options)`, `figure_joints(**options)`, `figure_garment(parts, inflate, voxel_size=0.008, **options)`, `FIGURE_DEFAULTS`, `PARTS` | the anatomical test figure (0.19.1; the showcases' character): known joint centres, builds and rest poses, garments over its parts |

## Procedural worlds

Terrain and block worlds generated from validated JSON definitions ([WORLDS.md](WORLDS.md)).
Since 0.18.0. Every changing call is one undo step; coordinates are metres (+Z up);
nothing is generated until a render, query or export needs it.

| Name | Description |
| --- | --- |
| `worlds = scene.worlds` | the `WorldSystem`: iterable, `len`, `get(ref)` (id or name), `names()`, `presets()` (name: description), `remove(ref)`, `to_json()` |
| `worlds.create(name="World", preset=None, *, seed=None, size=None, extent="keep", mode=None, overrides=None, materials=True)` | a new world from a preset (default `"temperate"`) deep-merged with `overrides`; `size` makes a square extent, `extent=None` an unbounded world; `materials` creates the world's layer materials |
| `worlds.prepare(times, *, camera=None, width=1280, height=720, should_stop=None, progress=None)` | generate what every world draws for a camera at those times, before rendering them |
| `world.id`, `name`, `mode`, `seed`, `extent`, `definition` (a copy), `edits`, `alive`, `location`, `runtime` | identity and state (`runtime`: caches, `stats()`, `counters`, `last_frame`) |
| `world.configure(patch=None, *, label=None, **sections)`, `world.rename(name)`, `world.info()` | change the definition (validated; the mode is fixed); a JSON summary with cost estimates |
| `world.add_feature(feature)`, `update_feature(id, patch)`, `remove_feature(id)`, `features(kind=None)` | terrain features (types and shapes: [WORLDS.md](WORLDS.md#features)) |
| `world.place_structure(item)`, `remove_structure(id)`, `structures(kind=None)`, `find_sites(count=5, **options)`, `route(a, b, *, cell=16)`, `build_road(a, b, *, name="road", cell=16)`, `build_settlements(count=None, *, houses=None, roads=None, seed=0, **options)` | structures, sites, roads and settlements |
| `world.height(x, y)`, `ground(x, y)`, `sample(points, *, fields=None)`, `biome_at(x, y)`, `profile(points, samples=64)`, `block_at(x, y, z)` | exact queries (scalars or arrays; `OUTSIDE_WORLD` beyond a bounded extent) |
| `world.ground_path(points, *, spacing=1, clearance=0, smooth=0)`, `place_on_ground(objects, *, offset=0)` | paths over the ground for characters and cameras; standing objects on the ground |
| `world.set_blocks(cells, block)`, `fill_blocks(lo, hi, block)`, `replace_blocks(lo, hi, source, block)`, `revert_blocks(lo, hi)` | block edits (block worlds; block coordinates); return how many cells changed |
| `world.export_region(path, region, *, level=0, include=("terrain", "water", "caves", "vegetation", "structures"), plant_lod=1, allow_lossy=True, bake_textures=None, max_triangles=4_000_000)` | a finite region's generated geometry as glTF (`COLOR_0` vertex colours; regions up to 8,192 m, at most 20M triangles and 200,000 plants) |
| `aether3d.worlds.script.run_world_script(scene, script, *, label=None, should_stop=None)`, `check_script(script)`, `schema()` | world scripts |

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
shading), [GPU.md](GPU.md) (the GPU backend) and
[RENDER_QUALITY.md](RENDER_QUALITY.md) (the path tracer).

| Name | Description |
| --- | --- |
| `render(scene, *, camera=None, width=None, height=None, samples=None, exposure=None, cull=None, default_lights=None, frame=False, margin=0.05, limits=None, backend="cpu", adapter=None, msaa=None, supersample=None, shadows=None, path=None, ao=None, tonemap="none", progress=None, outputs=None)` | Render the scene. Arguments override `scene.render_settings` for this render only. Same as `scene.render(...)`. `camera`: a camera object, id or name, or a *virtual* camera (a `RenderCamera` without an object id: `aether3d.feedback.virtual_camera(...)`), which renders from a view that is not part of the scene (since 0.12.0). `backend`: `"cpu"` (the NumPy rasterizer, the default), `"gpu"` (wgpu; `GPU_*` error if unavailable), `"auto"` (GPU if it works, else CPU with a `BACKEND_FALLBACK` diagnostic), `"path"` (the CPU path tracer) or `"gpu_path"` (the same path tracer on a GPU adapter; `GPU_*` error if unavailable). `adapter`: GPU adapter selection (also for `"gpu_path"`) (index, `"discrete"`, `"vulkan"`, part of a name, ...). `msaa`, `supersample`: GPU antialiasing (default: `samples²` samples per pixel). `shadows`: GPU shadow maps (`True`, a dict or a `ShadowSettings`). `path`: path tracer settings (`PathSettings` or a dict). `ao`: the path tracer's ambient occlusion pass (`True`, a dict or `AOSettings`). `tonemap`: `"none"` or `"neutral"` (any backend). `progress`: path tracer callback `progress(done, total, seconds)`; return `False` to stop. `outputs`: the per-pixel outputs to return (`None`: all; or some of `"image"`, `"linear"`, `"depth"`, `"object_index"`, `"coverage"`; the image always comes); the GPU reads back only those and the others are `None`. |
| `RenderResult` | `image` (`(h, w, 3) uint8` sRGB); `linear`, `depth`, `coverage` and `object_index` maps; `objects`, `camera`, `settings`, `stats` (counts and timings; GPU: also `gpu_timings`, `draw_calls`; path tracer: rays, rays per second, noise), `diagnostics`, `backend` (`{"name": "cpu"}`, the GPU adapter, API, antialiasing, shadow maps and resources, or the path tracer's device (`"cpu"` or `"gpu"`), processes or adapter, settings, BVH and lights), `passes` (path tracer: `beauty`, `direct`, `indirect`, `albedo`, `normal`, `depth`, `ao`, `sample_count`, `variance`), `color` (exposure and tone mapper), `output`, `status`, `width`, `height`, `pixel(x, y)`, `png_bytes(compression=6)` (zlib level; 1 encodes several times faster, a few percent larger), `save(path)` (atomic: `.png`, or `.hdr`/`.pfm` linear: `OUTPUT_NOT_RENDERED` if `linear` was not returned), `to_dict()`. |
| `ShadowSettings(lights=None, map_size=2048, bias=0.5, normal_offset=0.5, pcf_radius=1, cascades=1, cascade_split=0.75, distance=0)` | GPU shadow-map settings for one render (not saved in projects). `cascades` (1–4, since 0.18.0): maps per light over slices of the view's depth; `distance`: the farthest shadowed depth (0: the far plane). |
| `PathSettings(samples=16, bounces=4, seed=0, light_samples=1, russian_roulette=3, clamp=0, integrator="path", workers=0)` | Path tracer settings for one render (not saved in projects); see [RENDER_QUALITY.md](RENDER_QUALITY.md#python-and-cli-reference). |
| `AOSettings(samples=4, distance=1, strength=1)` | Ambient occlusion of one path-traced render (a separate pass and the `integrator="ao"` preview). |
| `list_adapters(*, check=False)` | Graphics adapters wgpu can use, best first: `index`, `name`, `vendor`, `type`, `backend`, `driver`, `software`, `features`, `limits`, `recommended` (with `check=True`: `usable`, `msaa`). Raises `RenderError("GPU_UNAVAILABLE")` without wgpu. |
| `gpu_status(*, adapter=None, check=False)` | Never raises: `available`, `reason`, `hint`, `versions`, `adapters`, `selected` (adapter, features, limits, MSAA counts) and, with `check=True`, a verified test render (`check`). |
| `release_gpu(adapter=None)` | Free cached GPU devices and memory; returns how many. |
| `Camera(projection="perspective", fov=40, fov_axis="vertical", ortho_scale=10, near=0.1, far=1000, aspect_ratio=None)` | Validated camera parameters, with `configure(**changes)`, `half_extents(aspect)` and `projection_matrix(aspect)`. |
| `Light(type="directional", color=(1, 1, 1), intensity=1, range=None, falloff="inverse_square", radius=0, angle=0)` | Validated light parameters, with `configure(**changes)`. `radius` (point) and `angle` (directional, degrees) are the light's size in the path tracer. |
| `Appearance(color=(0.8, 0.8, 0.8), specular=0, shininess=32)` | Object color and highlight. |
| `World(background=(0.2, 0.2, 0.2), ambient_color=(1, 1, 1), ambient_intensity=0.1, environment=None, atmosphere=None)` | Background, ambient light, the path tracer's environment and the outdoor atmosphere. `configure(environment={...})` updates some environment fields and keeps the others. |
| `Atmosphere(enabled=False, sun=None, sky=True, sun_glow=1, sun_glow_color=(1, 0.86, 0.62), sun_glow_size=32, horizon_glow=0, fog_density=0, fog_height=0, fog_falloff=0, fog_color=(0.78, 0.84, 0.92), fog_sun=0, fog_sun_power=8, ambient="uniform", ground_ambient=(0.36, 0.33, 0.29))` | `scene.world.atmosphere` (since 0.18.0): sky glows, exponential height fog with sun in-scattering, hemisphere ambient; `configure(**changes)`. Off by default. See [RENDERING.md](RENDERING.md#outdoor-atmosphere-0180). |
| `Environment(type="ambient", color=(1, 1, 1), intensity=1, zenith_color, horizon_color, ground_color)` | What rays leaving the scene see in the path tracer: the ambient light, a uniform color or a procedural sky (`scene.world.environment`, saved). |
| `RenderSettings(width=1280, height=720, samples=2, exposure=0, cull="none", default_lights=True)` | Saved output settings. |
| `RenderLimits(max_dimension=16384, max_samples=40_000_000, max_triangles=20_000_000, max_gpu_memory=4 GiB, max_path_rays=10**11, max_path_memory=8 GiB)` | Resource limits, checked before allocating. |
| `run_scene_script(scene, script)` | Apply a declarative `aether3d.scene-script` (cameras, lights, appearance, world, settings) atomically. |
| `aether3d.rendering.extract_render_scene(scene, camera)` | The backend-independent `RenderScene`. |
| `aether3d.rendering.encode_png(image)`, `write_png(path, image)` | Aether3D's PNG encoder. |
| `aether3d.rendering.srgb_to_linear`, `linear_to_srgb`, `parse_color`, `color_to_hex` | Color helpers. |
| `aether3d.math3d.look_rotation_matrix(direction, up)` | The world rotation used by `look_at`. |

Colors are display (sRGB) values: `"#rrggbb"`, `"#rgb"`, or three numbers in
[0, 1]. Every `Camera`, `Light`, `Appearance`, `World`, `Environment`,
`RenderSettings`, `PathSettings` and `AOSettings` value has `to_dict()`,
`from_dict()` and `copy()`. Setting an invalid value
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

## History

`scene.history` is a `History` ([HISTORY.md](HISTORY.md)). Every public call that
changes the scene is one step; a transaction is one step.

| Member | Description |
| --- | --- |
| `undo(steps=1)`, `redo(steps=1)` | Reverse or re-apply steps (all or nothing); return their `HistoryEntry` objects. |
| `can_undo`, `can_redo` | Whether there is anything to undo or redo. |
| `entries`, `redo_entries` | `HistoryEntry` tuples: the steps to undo (oldest first) and the undone ones (next first). |
| `transaction(label="Transaction")` | Context manager: one step, all or nothing; nests as savepoints. Yields a `Transaction` (`entry`, `rollback()`). |
| `checkpoint(name)`, `checkpoints`, `goto(name)`, `remove_checkpoint(name)` | Name the current state, list the names, undo or redo to one, forget one. |
| `max_entries`, `max_bytes`, `configure(max_entries=None, max_bytes=None)` | Limits (default 256 steps, 256 MiB); the oldest steps are dropped first. |
| `nbytes`, `dropped`, `revision` | Memory held only by the history, steps dropped so far, changes so far. |
| `in_progress` | Whether an operation or transaction is open (its changes are not counted in `revision` until it ends). |
| `enabled`, `disabled()` | Recording on or off (disabling clears the history; operations stay atomic). |
| `clear()` | Forget every step and checkpoint. |
| `save(path, *, project_dir=None)`, `load(path, *, project_dir=None, verify=True)` | Write or read a history file (see `Scene.save`/`Scene.load` with `history=True`). |
| `verify()` | Replay the history on a copy of the scene and report whether it comes back exactly. |
| `to_dict()` | JSON summary. |

`HistoryEntry`: `index` (unique, increasing), `label`, `changes` (count), `paths`
(the parts of the scene it changed, such as `objects[<id>].transform`), `bytes`
(memory estimate), `to_dict()`.

## Spatial queries

Read-only questions about where things are, answered exactly from the scene's
triangles in world space ([SPATIAL.md](SPATIAL.md)). They change nothing (no
history step, same `revision`). Common arguments: `objects`/`exclude` (objects,
ids or names, or lists), `include_hidden=False`, and `surfaces="geometry"` or
`"rendered"` for ray queries.

| `Scene` method | Returns |
| --- | --- |
| `bounds(obj=None, *, space="world", children=False, include_hidden=False)` | `ObjectBounds` (`aabb`, `obb`, `sphere`, counts) or `None` |
| `find_objects(*, name, kind, tags, any_tags, metadata, metadata_keys, parent, ancestor, visible, material, region, region_mode)` | objects matching every filter, in scene order |
| `raycast(origin, direction, *, max_distance, min_distance, ...)` | the first `SurfaceHit`, or `None` |
| `raycast_all(origin, direction, *, ..., limit=None)` | every `SurfaceHit`, nearest first |
| `closest_point(point, *, max_distance, ...)` | the nearest `SurfaceHit` (point-to-surface distance), or `None` |
| `nearest_objects(point, *, count=None, max_distance, ...)` | one `SurfaceHit` per object, nearest first |
| `contains_point(obj, point, *, tolerance=1e-6)` | `Containment`: `"inside"`, `"outside"`, `"surface"` or `"undefined"` (open or non-manifold meshes), with the winding number |
| `distance(a, b)` | `ObjectDistance`: the exact surface distance, its points, `intersecting` |
| `overlaps(objects=None, *, against=None, exact=True, tolerance=0.0, ...)` | `Overlap` pairs: intersecting surfaces or contained solids (`exact=False`: box candidates) |
| `query_region(region, *, mode="intersects", ...)` | `RegionMatch`es in a `Box` or `Sphere`: `"inside"`, `"intersects"`, `"contains"` (or `"bounds"`) |
| `camera_ray(x, y, *, camera=None, width=None, height=None)` | `CameraRay` through an image point (the path tracer's ray) |
| `pick(x, y, *, camera, width, height, objects, exclude, surfaces="rendered", cull=None)` | the `SurfaceHit` a camera shows at an image point, or `None` |
| `project(points, *, camera, width, height)` | `ScreenPoint` (or a list): image position, depth, `in_view` |
| `objects_in_view(*, camera, width, height, ..., exact=True)` | `ViewMatch`es in the view volume: `"inside"` or `"partial"` |
| `line_of_sight(start, end, *, ..., margin=1e-6)` | `LineOfSight`: `clear` and the first `blocker` |
| `visible_from(target, observer, *, samples=64, ...)` | `Visibility`: visible samples, `fraction`, blockers |
| `relation(a, b, *, near_distance=None, tolerance=1e-6, viewer=None)` | `Relation`: distance, intersecting, near, above, below, inside, contains, in front of, behind, nearer to a viewer |
| `describe(*, include_hidden=True)` | a compact JSON summary of the scene |

Every result has `to_dict()`. The functions are also in `aether3d.spatial`
(taking the scene first), with the regions `Box` (`AABB`) and `Sphere`, the
result types, `surface_samples`, `get_index` (the cached `SpatialIndex`) and
`run_query(scene, request)` (also `aether3d.run_query`): a JSON request
`{"query": name, ...arguments}` answered with `{"query": name, "result": ...}`,
which the command line (`aether3d query`) and the scene-script operation `query`
use. Errors are `QueryError`s.

## Scene plans

One JSON-compatible document builds or changes a scene in one call: checked as a whole,
then applied as one undo step, all or nothing ([SCENE_PLANS.md](SCENE_PLANS.md)).

| Function | |
| --- | --- |
| `scene.execute_plan(plan, *, mode="execute", expected_revision=None, limits=None, base_dir=None, trusted_paths=False, progress=None, cancel=None, timeout=None)` | apply a plan (a dict or JSON text); `mode` `"dry_run"` applies and rolls back, `"validate"` only checks; returns a `PlanResult`, raises `PlanError` |
| `aether3d.execute_plan(scene, plan, **options)` | the same |
| `aether3d.validate_plan(plan, scene=None, *, limits=None, base_dir=None)` | check without changing anything (never raises for a problem in the plan): `PlanResult` with `ok`, `issues`, `estimate` |
| `aether3d.plan_schema()` | the format's JSON Schema (draft 2020-12) |
| `aether3d.plans.select_objects(scene, selector)` | the objects a plan selector names (`{"tags": [...]}`, `{"under": ...}`, ...), in scene order |
| `aether3d.plans.with_selections(scene, request)` | a spatial query request whose object lists may be selectors, resolved |

`PlanResult` has `ok`, `mode`, `status` (`applied`, `dry_run`, `validated`,
`already_applied`), `label`, `fingerprint`, `idempotency_key`, `revision` (`before`,
`after`), `step` (the undo step), `counts`, `symbols`, `operations`, `queries`,
`warnings`, `issues`, `estimate`, `chunks`, `timings` and `to_dict()`. `PlanLimits`
(frozen dataclass) bounds a plan: `max_objects`, `max_scene_objects`,
`max_pattern_count`, `max_operations`, `max_triangles`, ... ([limits](SCENE_PLANS.md#limits)).

## Visual feedback

Observe renders, check expectations, propose, apply and verify corrections
([FEEDBACK.md](FEEDBACK.md)). Since 0.12.0.

| Name | |
| --- | --- |
| `scene.observe(view=None, **options)`, `aether3d.feedback.observe(scene, view=None, *, width, height, backend="cpu", adapter, samples, shadows, path, linear=False, focus=None, occlusion=False, max_markers=12, limits=None, store=None)` | render one view and measure it: an `Observation` (`id`, `revision`, `view`, `image`, `index`, `depth`, `linear`, `stats`, `image_stats`, `renderer`, `camera`, `sha256`; `record(obj)`, `geometry(obj, occlusion=)`, `instances_of(obj)`, `objects_in_rect(rect)`, `markers()`, `png(kind)` with kinds `image`, `annotated`, `ids`, `depth`, `mask`, `crop_png(rect, scale=)`, `to_dict()`). Never changes the scene |
| `aether3d.feedback.virtual_camera(position, *, look_at=None, direction=None, up=(0, 0, 1), name=..., **lens)`, `camera_view(scene, camera=None)` | a camera that is not in the scene (`RenderCamera`); a scene camera's virtual twin |
| `resolve_view(scene, spec, width, height)`, `check_view_spec(spec)` | view specifications: `{}`, `{"camera": ...}`, `{"virtual": {...}}`, `{"focus": {"objects": ...}}`, `{"preset": "top"}` |
| `FeedbackSession(scene, expectations=None, *, backend="cpu", adapter=None, width=640, height=360, samples=None, shadows=False, policy=None, store=None, limits=None, base_dir=None)` | the workflow on one scene: `observe(view, ...)`, `diagnose(evaluations=(), defaults=None)` -> `Diagnosis`, `propose(diagnosis=None, policy=None)` -> `CorrectionSet`, `preview(corrections, ids=None)`, `apply(corrections=None, ids=None, *, actions=None, approve=(), policy=None, verify=None, expected_revision=None, label=None)` -> `ApplyResult`, `run(**options)` -> `LoopReport`; also `aether3d.FeedbackSession` |
| `aether3d.feedback.parse_expectations(document)` | check an `aether3d.expectations` document (dict or JSON text): `Expectations` |
| `aether3d.feedback.check_evaluation(document, observation)` | check and ground an `aether3d.vision-evaluation`: `VisionEvaluation` (`findings`, `basis`, `provenance`, `overall`, `stale`, `delivered`) |
| `aether3d.feedback.critique_request(observation, intent=None)` | what a critic receives: `CritiqueRequest` (PNG bytes, markers, summary, instructions) |
| `aether3d.feedback.check_actions(actions)` | check correction actions against the allowlist (`ACTIONS`) |
| `aether3d.feedback.run_loop(session, *, budget=None, policy=None, critic=None, intent=None, cancel=None, progress=None, resume=None)` | the automatic loop: `LoopReport` (`stop_reason`, `iterations`, `final`, `budget`, `pending`, `checkpoint`) |
| `CorrectionPolicy`, `FeedbackBudget`, `ObservationLimits`, `ObservationStore` | policy (`auto_apply`, `allow_*`, limits, `verify`, `rollback_*`), loop budget, observation bounds, the observation cache; `coerce(dict)` for the first two |
| `aether3d.feedback.schema.expectations_schema()`, `evaluation_schema()`, `actions_schema()` | the documents' JSON Schemas (draft 2020-12) |

A `Diagnosis` has `id`, `revision`, `checks` (`CheckResult`: `id`, `kind`, `status`,
`findings`, `measured`), `findings` (`Finding`: `id`, `key`, `code`, `severity`,
`basis`, `exact`, `message`, `subjects`, `evidence`, `magnitude`, `unit`, `view`,
`observation`, `region`, `hint`), `errors`, `warnings`, `score`, `accepted`,
`acceptance`, `vision`, `observed` (the observations), `stale` and `to_dict()`. A
`Proposal` has `id`, `title`, `findings`, `actions`, `safety`, `basis`, `rationale`,
`confidence`, `expected`, `validation`, `alternatives`, `changes`, `compiled` (the scene
plan and script), `policy` and `to_dict()`.

## The MCP server

`aether3d.mcp` (needs `pip install "aether3d[mcp]"`; `import aether3d` never imports it)
serves Aether3D to AI agents over the Model Context Protocol ([MCP.md](MCP.md)):
`serve(config)` runs a stdio server with a `ServerConfig` (`workspace`, `read_only`,
`project`, `plan_limits`, `tool_timeout`, `max_preview_pixels`, `max_render_pixels`,
`max_path_samples`, `allow_path_tracing`, `gpu_adapter`, `max_reply_bytes`,
`feedback_backend`); `TOOLS` describes its tools,
`Workspace` its path policy. Most uses go through `aether3d mcp serve`.

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
| `SculptError` | `EditError` | A sculpt operation or commit cannot be carried out (`VOXEL_NOT_WATERTIGHT`, `REMESH_TOO_DENSE`, `BELOW_VALIDITY_TIER`, `CANCELLED`, `STALE_SESSION`, ...; see [MODELING.md](MODELING.md)); nothing changed. `.code`, `.hint` |
| `SculptScriptError` | `ValueError` | A sculpt script is malformed or a step failed (`INVALID_SCRIPT`, `INVALID_ARGUMENT`, `BUDGET_EXCEEDED`, or the step's own code); nothing changed. `.code`, `.location` (e.g. `sculpts[0].operations[2]`), `.details`, `to_dict()` |
| `TexturingError` | `ValueError` | A UV, painting, baking or texture-script operation cannot be carried out (`NO_UVS`, `CHART_NOT_A_DISK`, `ATLAS_FULL`, `INVALID_ARGUMENT`, `BAKE_TOO_LARGE`, `GPU_UNAVAILABLE`, `TEXTURE_NOT_FOUND`, `BUDGET_EXCEEDED`, `CANCELLED`, ...; see [TEXTURING.md](TEXTURING.md)); nothing changed and no file was written. `.code`, `.location` (e.g. `operations[3].brush.radius`), `.hint`, `.details`, `to_dict()` |
| `ScriptError` | `ValueError` | An edit or procedural script is malformed, or a step failed. `.code`, `.location` (e.g. `edits[0].operations[3]`), `.details` |
| `RenderError` | `ValueError` | A camera, light, appearance, world or render setting is invalid, or rendering failed (including GPU problems: `GPU_UNAVAILABLE`, `GPU_ADAPTER_NOT_FOUND`, `GPU_DEVICE_LOST`, ...; see [GPU.md](GPU.md#errors-diagnostics-and-debugging)); nothing changed and no file was written. `.code` (stable), `.location`, `.hint`, `.details` |
| `ProceduralError` | `ValueError` | A procedural definition, parameter or evaluation is invalid; nothing changed. `.code` (stable), `.location` (e.g. `geometry[Tower].modifiers[Row].params.count`), `.hint`, `.details` |
| `MaterialError` | `ValueError` | A material, texture, graph or image is invalid or cannot be used (`INVALID_MATERIAL`, `TEXTURE_MISSING`, `IMAGE_TOO_LARGE`, `UNSAFE_PATH`, ...; see [MATERIALS.md](MATERIALS.md#errors-and-diagnostics)); nothing changed. `.code` (stable), `.location` (e.g. `materials[Stone].roughness`), `.hint`, `.details` |
| `QueryError` | `ValueError` | A spatial query is malformed or cannot be answered (`INVALID_ARGUMENT`, `INVALID_REQUEST`, `OBJECT_NOT_FOUND`, `NO_GEOMETRY`, `NO_CAMERA`, ...; see [SPATIAL.md](SPATIAL.md#errors)); nothing changed. `.code` (stable), `.location` (the argument), `.hint`, `.details`, `to_dict()` |
| `PlanError` | `ValueError` | A scene plan is invalid or cannot be applied; nothing changed (except completed chunks of an opt-in chunked plan). `.code` (stable: `UNKNOWN_REFERENCE`, `NAME_CONFLICT`, `REVISION_MISMATCH`, `LIMIT_EXCEEDED`, ...; see [SCENE_PLANS.md](SCENE_PLANS.md#errors)), `.pointer` (a JSON pointer into the plan), `.operation`, `.hint`, `.details`, `.issues` (every problem found), `to_dict()` |
| `FeedbackError` | `ValueError` | A visual-feedback request cannot be carried out: a view, an expectations document, a vision evaluation, a correction or a loop (`INVALID_VIEW`, `UNKNOWN_KEY`, `OBSERVATION_MISMATCH`, `STALE_PROPOSALS`, `REVISION_MISMATCH`, `NOT_ALLOWED`, ...; see [FEEDBACK.md](FEEDBACK.md)); nothing changed. `.code` (stable), `.location` (a JSON pointer or argument), `.hint`, `.details`, `.issues`, `to_dict()` |
| `CadError` | `ValueError` | A CAD definition, regeneration, measurement, script or exchange cannot be carried out (`SKETCH_CONFLICT`, `REFERENCE_LOST`, `REFERENCE_AMBIGUOUS`, `HOLE_OFF_FACE`, `FILLET_CORNER_UNSUPPORTED`, `BOOLEAN_FAILED`, `WALLS_CROSS`, `OPENING_OVERLAP`, `ROOM_NOT_ENCLOSED`, `UNSUPPORTED_STEP`, `INVALID_DXF`, `EXPECTATION_FAILED`, `LIMIT_EXCEEDED`, `CANCELLED`, ...; see [CAD.md](CAD.md)); nothing changed. `.code` (stable), `.location` (e.g. `part[Bracket].features[2](h)`), `.hint`, `.details`, `to_dict()` |
| `AnimationError` | `ValueError` | A time, frame rate, key, curve, rotation track, clip, layer, strip, rig, skin, morph target, constraint, path, shot or animation export/import request cannot be carried out (`INVALID_KEYS`, `UNKNOWN_PROPERTY`, `INVALID_RIG`, `CONSTRAINT_CYCLE`, `ROLE_MISSING`, `RIG_MISSING`, `JOINT_MISSING`, `INVALID_RANGE`, `LIMIT_EXCEEDED`, ...; see [ANIMATION.md](ANIMATION.md)); nothing changed and no file was written. `.code` (stable), `.location`, `.hint`, `.details` |
| `WorldError` | `ValueError` | A world definition, preset, feature, structure, block edit, query, world script, budget or export request cannot be carried out (`INVALID_DEFINITION`, `UNKNOWN_BIOME`, `OUTSIDE_WORLD`, `NO_ROUTE`, `LIMIT_EXCEEDED`, `BUDGET_EXCEEDED`, `CANCELLED`, ...; see [WORLDS.md](WORLDS.md#errors)); nothing changed and no file was written. `.code`, `.location`, `.hint`, `.details`, `to_dict()` (since 0.18.0) |
| `AnimationScriptError` | `ValueError` | An animation script is malformed or one of its operations failed; nothing changed. `.code` (the operation's own, or `INVALID_SCRIPT`), `.location` (e.g. `operations[1].keyframe.value`), `.details`, `to_dict()` |
| `HistoryError` | `ValueError` | An undo, redo, checkpoint or history file operation cannot be performed (`NOTHING_TO_UNDO`, `TRANSACTION_OPEN`, `HISTORY_MISMATCH`, `INVALID_HISTORY`, ...; see [HISTORY.md](HISTORY.md#errors)); nothing changed. `.code` (stable), `.location`, `.hint`, `.details` |

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
aether3d render PATH [-o OUT.png|OUT.hdr|OUT.pfm] [--width W] [--height H] [--camera REF] [--samples N] [--exposure EV] [--tonemap none|neutral] [--cull none|back|front] [--background COLOR] [--no-default-lights] [--frame] [--margin M] [--force] [--json]
    [--backend cpu|gpu|auto|path|gpu_path] [--adapter SPEC] [--msaa N] [--supersample N]
    [--shadows [--shadow-lights LIGHT...] [--shadow-map-size N] [--shadow-bias B] [--shadow-normal-offset X] [--shadow-pcf-radius R]]
    path tracer: [--spp N] [--bounces N] [--seed N] [--light-samples N] [--russian-roulette N] [--clamp X] [--workers N]
                 [--ao] [--ao-only] [--ao-samples N] [--ao-distance D] [--ao-strength S] [--passes DIR]
aether3d gpu info [--adapter SPEC] [--check] [--json]
aether3d camera add|set|frame|activate|list PROJECT ... [--output PATH] [--dry-run] [--json]
aether3d light add|set|list PROJECT ... [--output PATH] [--dry-run] [--json]
aether3d object set|remove PROJECT OBJECT ... [--output PATH] [--dry-run] [--json]
aether3d scene set PROJECT ... | scene run PROJECT SCRIPT.json|- [--output PATH] [--dry-run] [--json]
aether3d scene merge PROJECT SOURCE.aether|.glb|.gltf [--parent REF] [--allow-lossy] [--output PATH] [--dry-run] [--json]
aether3d material list|presets|schema|create|set|assign|texture|inspect|validate|remove|convert-colors|preview ...
aether3d feedback observe PATH [--view NAME|JSON] [--expect FILE] [--focus JSON] [--occlusion] [--linear] [--images image,annotated,ids,depth] [--out DIR] [--backend B] [--width W] [--height H] [--json]
aether3d feedback check PATH --expect FILE [--evaluation FILE]... [--out DIR] [--json]
aether3d feedback correct PROJECT --expect FILE [--policy FILE|JSON] [--preview] [--apply] [--approve IDS] [--output PATH] [--dry-run] [--history] [--json]
aether3d feedback loop PROJECT --expect FILE [--budget FILE|JSON] [--policy FILE|JSON] [--out DIR] [--output PATH] [--dry-run] [--history] [--json]
aether3d feedback schema expectations|vision-evaluation|corrections
aether3d texture list|add|procedural|set|inspect|preview|remove ...
aether3d history start|show|undo|redo|checkpoint|goto|clear|verify PROJECT ... [--json]
aether3d query QUERY PATH [KEY=VALUE ...] [--allow-lossy] [--json]
aether3d query run PATH --request FILE|- [--allow-lossy] [--json]
aether3d execute-plan PROJECT.aether PLAN.json|- [--dry-run | --validate] [--output PATH] [--history] [--timeout S] [--max-objects N] [--progress] [--json]
aether3d plan validate PLAN.json|- [--project PROJECT.aether] [--max-objects N] [--json]
aether3d plan schema
aether3d mcp serve [--workspace DIR] [--project PATH] [--read-only] [--max-objects N] [--timeout S] [--no-path-tracing] [--gpu-adapter SPEC] [--log-level LEVEL]
aether3d --version
```

Commands that change a project accept `--history`: the change is recorded in the
project's history file (`<project>.history`), which starts if there is none. A
project that has a history file records every change, and `aether3d history
undo PROJECT` undoes the last one (see [HISTORY.md](HISTORY.md#command-line)).

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
- `code`, for `AssetFormatError`, `ScriptError`, `ProceduralError`, `RenderError`,
  `MaterialError` and `QueryError` (with `hint` and `details` when available);
- `diagnostics`, for `UnsupportedFeatureError`.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Failure |
| `2` | Usage error |
| `3` | `geometry`, `topology`, `procedural validate`, `material validate` or `plan validate` completed but found error-severity problems, or `gpu info --check` found that GPU rendering does not work |

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
