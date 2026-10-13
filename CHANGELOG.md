# Changelog

All notable changes to Aether3D are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may add
features; breaking changes are called out explicitly).

## 0.3.0: Mesh editing engine

### Added

- **Editable polygon meshes** (`aether3d.editing.EditableMesh`). The
  topology is BMesh-style (vertices, edges, faces, per-corner attributes)
  stored in id-keyed tables. It supports n-gons, boundaries, non-manifold
  geometry, and UV/normal seams. Ids are stable, and every operation reports
  the ids it created, removed and modified. `from_mesh()` (welding, and
  lossless quad reconstruction) and `to_mesh()` (ear-clipping triangulation)
  convert to and from the immutable `Mesh`.
- **Editing sessions:** `SceneObject.edit_mesh()` returns an `EditSession`.
  - Operations are atomic; a failure never leaves a partial edit.
  - Undo and redo restore exact state and ids.
  - Commits are validated, and a session refuses to commit over a mesh that
    changed meanwhile (stale-session protection).
  - Used as a context manager, a session commits on success and discards on
    exception.
  - Instancing is preserved: by default only the edited object changes;
    `scope="shared"` updates every object sharing the mesh.
- **Selection** by ids, box, sphere, plane, normal, nearest (ambiguity is an
  error), boundary, polygon size, or a predicate, in local or world space,
  with `expect=` count guards. Also edge loops, edge rings, linked faces,
  region boundaries, growth, and kind conversions.
- **Transforms:** translate, rotate, scale, arbitrary matrices, explicit
  positions, and moves along normals.
  - World-space edits are exact under rotated, non-uniform and mirrored
    parents.
  - Mirrored pieces are re-wound so they stay outward-facing.
  - Stored normals are transformed exactly.
- **Modeling operations:**
  - extrusion (region and individual, along normals or by an offset);
  - inset (individual and region, mitred, with depth);
  - edge bevel (chamfer, including box corners);
  - subdivision (quads, triangles, n-gons);
  - loop cuts (closed and open quad rings, with factor);
  - vertex merge, merge by distance and edge collapse;
  - edge dissolve and triangle-to-quad joining;
  - edge, face and vertex splitting, and face separation;
  - triangulation, polygon creation, box and grid creation;
  - deletion, flipping, and smooth/sharp shading control.

  Unsupported or unsafe cases raise `EditError` with a stable code,
  offending ids and a hint.
- **Attribute handling:**
  - UVs and normals are preserved exactly where untouched;
  - they are interpolated (linear, bilinear, barycentric or affine) where
    geometry is refined;
  - normals are recomputed from a flat/smooth/sharp shading model where
    geometry changed;
  - unavoidable loss is reported (`UV_DEGENERATE_FACES`, `UVS_UNDEFINED`);
  - corner layers are generic, so future attributes need no operation
    changes.
- **Topology validation** (`EditableMesh.validate()`): adjacency consistency,
  non-finite data, self-intersecting, zero-area and non-planar polygons,
  duplicate faces, coincident vertices, boundary and non-manifold edges,
  non-manifold vertices, inconsistent winding, concave faces, and loose
  vertices. Cross-face self-intersection is explicitly not evaluated.
- **Declarative JSON edit scripts** (`run_edit_script`, `ScriptError`): named
  selections, inline queries, `save`, validate/summary/describe/undo/redo
  steps, exact error locations, and all-or-nothing application. No code is
  ever evaluated.
- **CLI:** `aether3d edit` (atomic project updates, `--output`, `--dry-run`,
  scripts from stdin) and `aether3d topology` (polygon topology, validation
  and element listings).
- `examples/minecraft_hero.py`: a blocky hero and sword modelled entirely with
  editing operations, exported to GLB, then checked with the Khronos
  validator, a re-import, and optionally Blender (headless).
- `benchmarks/bench_editing.py`, plus documentation in `docs/EDITING.md`.

### Changed

- `inspect_mesh` (and `aether3d geometry`) now evaluates vertex manifoldness
  (`vertex_manifold`, `non_manifold_vertex_count`, `NON_MANIFOLD_VERTICES`).
  This was documented as not evaluated in 0.2.0.
- `EditError` and `ScriptError` join the error hierarchy.

### Compatibility

- **No `.aether` format change** (still version 1). `Mesh`, glTF import and
  export, and every 0.1.0 and 0.2.0 API behave as before; all earlier tests
  pass unchanged.
- **The CLI is backward compatible:** two commands were added, and the
  `topology` command uses exit code `3` like `geometry`.

## 0.2.0: Geometry interoperability and inspection

### Added

- **glTF 2.0 export** to `.glb` and `.gltf` with an embedded buffer:
  `Scene.export(path)` and `aether3d.export_gltf()`. It writes indexed
  triangle meshes with positions, normals and UVs, node transforms and
  hierarchy, and instanced shared meshes. Object ids go in
  `node.extras.aether3d.id`. Output is deterministic and written atomically.
- **glTF 2.0 import** from `.gltf` and `.glb`: `Scene.import_file(path)` and
  `aether3d.import_gltf()`, which returns an `ImportResult` with a
  node-to-object map and diagnostics.
  - **Supported data:** external, embedded and GLB buffers; interleaved and
    sparse accessors; triangle strips and fans; decomposable node matrices;
    multiple primitives.
  - **Unsupported content is reported explicitly.** Geometry-changing content
    (skins, morph targets, point/line primitives, sheared matrices) raises
    `UnsupportedFeatureError` unless `allow_lossy=True`.
  - **Hardened for untrusted files:** path confinement, URI scheme
    restrictions, `ImportLimits`, and strict JSON.
- **Exact Z-up ↔ Y-up conversion** (`aether3d.gltf.axes`), the same
  convention as Blender. Positions, normals, transforms and hierarchies
  convert bit-exactly; UVs are flipped `v → 1 − v` (round trip within 2.5e-7).
- **Geometry inspection:** `inspect_mesh()`, `inspect_scene()` and
  `Scene.inspect_geometry()`. Reports cover counts, local and world bounds,
  surface area, attribute availability, non-finite data, and degenerate and
  duplicate triangles. They also check unreferenced vertices, normal validity,
  normals that oppose the winding, and edge topology (closed, edge-manifold,
  consistent orientation, signed volume). Transform checks cover mirrored,
  sheared and zero scale. See `docs/INSPECTION.md`.
- `Diagnostic`, a structured finding with a stable `code`, `severity`,
  `location`, `count`, `samples` and `lossy` flag, shared by inspection,
  import and export.
- New errors: `AssetFormatError` (with `code` and a JSON-pointer `location`),
  `UnsupportedFeatureError` (with `diagnostics`), and `ResourceLimitError`.
- `Scene.unique_name(base)`.
- **CLI:**
  - new commands `aether3d import`, `aether3d export` and
    `aether3d geometry`;
  - `validate` and `inspect` also accept `.glb`/`.gltf`;
  - `--allow-lossy`;
  - exit code `3` when `geometry` finds error-severity problems;
  - JSON errors include `code` and `diagnostics` when available.
- `examples/gltf_roundtrip.py`, an end-to-end demonstration: build, export,
  validate with Khronos, import, inspect, and verify.
- **Verification tooling:**
  - the official Khronos glTF Validator, pinned in `tools/gltf-validator`
    (a new CI job);
  - trimesh as an independent parser in tests (development dependency only);
  - pinned Khronos sample assets in `tests/data/khronos`.
- `docs/GLTF.md` and `docs/INSPECTION.md`.

### Changed

- **Quaternion normalization is now idempotent.** Already-unit quaternions
  are left untouched instead of being divided by a norm of 1 ± 1 ulp, so
  round trips are bit-exact.
- **New UV spheres close their seam exactly.** The last column of vertices is
  now bit-identical to the first, where previously there was a ~1e-16 gap. The
  vertex and triangle counts are unchanged. Spheres stored in existing projects
  keep their original data.
- **Coordinate documentation clarified.** It said "+Y forward"; it now states
  that an asset's front faces −Y (as in Blender), which maps to glTF's +Z
  front. No behaviour depended on the previous wording.
- `Scene.save()` now shares its atomic-write helper with the exporters; the
  behaviour is unchanged.

### Compatibility

- **The `.aether` project format is unchanged (still version 1).** Projects
  written by 0.1.0 load, save and export without migration; this is covered by
  `tests/data/v0.1.0/first_scene.aether`.
- **The CLI is backward compatible.** Existing commands keep their arguments,
  output fields and exit codes; JSON output may contain additional fields.
- **Runtime dependencies are unchanged:** NumPy only.

## 0.1.0: Core foundation

- Scene graph with stable UUID ids, unique names and an ordered hierarchy.
- Transforms (position, quaternion rotation, scale) with local and world
  matrices.
- Immutable indexed triangle `Mesh`, plus cube, sphere and plane generators.
- Versioned, strictly validated `.aether` JSON project format.
- `aether3d new | validate | inspect` CLI with `--json` output.
