# Changelog

All notable changes to Aether3D are documented here. Versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may add
features; breaking changes are called out explicitly).

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
