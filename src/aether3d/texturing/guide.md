# Aether3D texturing: a guide for agents

Give objects UVs, paint textures on them in 3D, and bake detail from high-resolution
sources, all as **data**: one `aether3d.texture-script` document, applied as one undo step
(everything or nothing). Nothing in a script is executed as code, and a script never
reads or writes a file.

## A script

```json
{"format": "aether3d.texture-script", "version": 1, "label": "Paint the crate",
 "limits": {"max_seconds": 120, "max_texels": 16777216, "max_dabs": 50000},
 "operations": [
  {"op": "unwrap", "object": "Crate", "resolution": 1024, "padding": 8},
  {"op": "uv_report", "object": "Crate"},
  {"op": "add_texture", "name": "Crate color", "width": 1024, "height": 1024,
   "background": "#8a6a48", "layers": [{"name": "Dirt"}, {"name": "Scratches"}]},
  {"op": "bind", "object": "Crate", "slot": "base_color", "texture": "Crate color",
   "factor": "#ffffff"},
  {"op": "fill", "object": "Crate", "texture": "Crate color", "layer": "Dirt",
   "color": "#3b2a1a", "where": {"groups": [2]}, "opacity": 0.4},
  {"op": "stroke", "object": "Crate", "texture": "Crate color", "layer": "Scratches",
   "brush": {"radius": 0.03, "color": "#d8c8a8", "strength": 0.8, "hardness": 0.7},
   "points": [[0.5, -0.3, 0.1], [0.5, -0.1, 0.25], [0.5, 0.2, 0.3]]},
  {"op": "bake", "target": "Crate", "sources": ["Crate high"],
   "maps": ["normal", "ao"], "resolution": 1024, "apply": true}
 ]}
```

Run it with `run_script` (MCP), `aether3d texture run PROJECT script.json` (CLI) or
`aether3d.texturing.run_texture_script(scene, script)` (Python). The full schema is the
resource `aether3d://schema/texture-script`; `capabilities {"topic": "texturing"}` lists
operations, brushes, blend modes and bake maps.

## UVs first

* `unwrap` cuts the mesh into charts (at hard edges, face-group borders and `seams`, then
  wherever the flattening would stretch more than `max_stretch`), flattens each without
  flips or overlaps and packs them into the unit square with `padding` texels between
  them at `resolution`. `texels_per_unit` fixes the texel density instead of filling the
  square.
* `project_uvs` (`planar`, `box`, `cylindrical`, `spherical`) is for simple shapes and
  deliberate projections. `edit_uvs` moves, scales, rotates, mirrors, aligns and
  normalizes islands, sets texel density, packs and stitches.
* Check with `inspect_uvs` (or `uv_report`): `flipped_triangles` and
  `overlapping_triangles` should be 0 (stacked islands -- deliberate copies sharing
  texels -- are counted apart), `texel_density.spread` small, `min_gap_texels` at least
  the padding. The layout image (`image: "overlaps"`, `"density"`, `"stretch"`) shows
  where.

## Painted textures

A painted texture has layers over a background: `paint` layers hold texels, `fill`
layers a single `value`; each has a `blend` mode (`normal`, `multiply`, `screen`,
`overlay`, `darken`, `lighten`, `difference`, `add`, `subtract`), an `opacity`, a
`visible` flag and an optional mask. `encoding`: `color` (display sRGB colors: base color,
emission), `data` (linear values: roughness, metallic, masks) or `normal` (tangent-space
normal maps). Texels are 16-bit; everything is saved in the project.

`bind` attaches a texture to a slot of the object's material (`base_color`,
`roughness`, `metallic`, `normal`, `emissive`, `opacity`, `occlusion`); textures
**multiply** the slot's factor, so set `factor` (e.g. `"#ffffff"`, `1`) to see the painted
values as they are. Number slots read one `channel` (roughness: `g` by default).

## Strokes

* `space: "surface"` (default): `points` are 3D, in **world** space like sculpt strokes
  (`"coordinates": "object"` for the object's own), and the brush `radius` is in the same
  units. The brush reaches every texel whose surface point is within the radius, so
  strokes run across UV seams without a visible edge. `symmetry` (`"x"`, `"xz"`, ...)
  mirrors across the object's own planes.
* `space: "view"`: `points` are pixel positions in a `camera`'s image of `view_size`; the
  brush paints what that camera sees -- other objects in front block it, the far side is
  not painted (`brush.backfaces: true` paints through). `radius` is in pixels.
* `space: "uv"`: `points` are `[u, v]`, `radius` in texels (2D painting; no seam help).
* Brushes (`brush.kind`): `paint`, `erase` (lowers the layer's alpha), `blur`, `smudge`,
  `clone` (from `clone_offset` texels away), `mask` (paints the layer's mask). Each dab:
  `strength x falloff x pressure`; a stroke builds up dab by dab but never beyond
  `opacity`. `spacing` is a fraction of the radius; `falloff`: `smooth`, `linear`,
  `constant`, `gaussian`; `hardness` keeps full strength to a fraction of the radius.
* `targets: [{"texture": "Color", "color": "#aa3311"}, {"texture": "Rough", "color":
  [0.8]}]` paints several textures at once (a material stroke).
* `fill` covers a layer, or the texels of some `islands`, face `groups` or `triangles`;
  `gradient` blends two colors along a 3D segment over the surface.
* Every stroke also paints `padding` rings of texels outside the islands, so filtering and
  mipmaps do not bleed old colors at seams.

## Baking

`bake` traces from the `target`'s surface (pushed out along its normals by `cage`, back
by `depth`) to the `sources` and stores, per texel: `normal` (tangent space, matching the
renderers' MikkTSpace tangents and glTF), `object_normal`, `world_normal`, `height`, `ao`,
`thickness`, `curvature`, `position`, material channels (`base_color`, `roughness`,
`metallic`, `emissive`, `opacity`) and ids (`material_id`, `object_id`, `group_id`).
`samples` is the supersampling per texel side; `padding` dilates the maps.
`apply: true` stores them as painted textures, binds them to the target's material and
sets the factors they multiply to neutral. The report gives `miss_fraction`; misses mean
the cage is too small or the sources do not cover the target (raise `cage`/`depth`).

`transfer` carries one object's look to another of the same shape (e.g. after a voxel
remesh, which drops UVs): it unwraps the target if needed and bakes base color,
roughness, metallic and normals into new textures on a material of its own.

## Verify

* `inspect_uvs` and `inspect_texture` (read-only) show UV layouts and textures;
  `render_preview` or `observe_scene` shows the painted object.
* Errors carry a stable `code` and the `location` of the step
  (`operations[5].brush.radius`); nothing is changed when a script fails.
