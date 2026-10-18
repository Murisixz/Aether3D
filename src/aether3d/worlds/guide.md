# Procedural worlds: a guide for agents

A scene can hold **worlds** (`scene.worlds`): terrain or block landscapes generated from a
small JSON *definition* -- seed, extent, terrain recipe, climate, biomes, rivers and lakes,
caves, vegetation, structures -- plus sparse authored edits. Nothing is generated until
something needs it (a render, a query); results are deterministic: the same definition
gives the same world on every machine, whatever the order of generation.

## Make a world

```json
{"format": "aether3d.world-script", "version": 1,
 "operations": [
   {"op": "create", "name": "Valley", "preset": "alpine_valley", "seed": 7}
 ],
 "queries": [{"query": "info"}]}
```

Run it with MCP `run_script` (or `aether3d world run PROJECT script.json`). One script is
one undo step; if any operation fails nothing changes, and the error names the operation
(`operations[2].feature.shape`) with a stable `code` and a `hint`.

**Presets**: `temperate`, `alpine_valley`, `forested_mountains`, `archipelago`,
`canyon_desert`, `snowy_wilderness`, `rolling_plains`, `biome_continent`,
`cavernous_highlands`, `infinite_highlands` (unbounded), `blocky_adventure` (a block
world). `create` takes `seed`, `size` (a square extent in metres), `extent`
(`{"min": [x, y], "max": [x, y]}` or `null` for unbounded), `mode` (`terrain` or `blocks`)
and `overrides` (a partial definition merged over the preset).

## Change a world

* `configure` with a `patch` (partial definition): e.g.
  `{"op": "configure", "patch": {"terrain": {"mountains": {"height": 1800}}}}`.
* `add_feature` / `update_feature` / `remove_feature`: authored features, applied in order.
  Shape features (`raise`, `lower`, `scale`, `mountain`, `ridge`, `valley`) shape the land
  before erosion; final features (`flatten`, `pad`, `road`, `river`, `lake`, `smooth`,
  `biome`, `clear`, `cave`) apply after. Every feature has an `id` and a `shape`:
  `{"circle": [x, y, r]}`, `{"rect": [x0, y0, x1, y1]}`, `{"polygon": [[x, y], ...]}` or
  `{"line": [[x, y(, z)], ...], "width": w}`.
  Example lake: `{"type": "lake", "id": "tarn", "shape": {"circle": [200, 100, 80]},
  "level": 412}`. A `flatten` without `height` takes the ground height at its centre.
* Structures: `place_structure` with an item (`house`, `tower`, `castle`, `ruin`, `well`
  at `at` [x, y] with `rotation`, `size`, `floors`, `style` timber/stone/brick; or a
  `bridge` along `line`). Its foundation follows the natural ground automatically.
* `build_settlements` (`count`, `houses` [min, max], `roads`, `seed`, `options`:
  `max_slope`, `biomes`, `spacing`, `region`): villages on good sites joined by roads,
  frozen into the definition. `build_road` from `from` to `to` follows the terrain and
  bridges water.
* Block worlds: `set_blocks` (`cells` [[x, y, z]...] in block coordinates, `block`),
  `fill_blocks` / `replace_blocks` / `revert_blocks` (`min`, `max`, `block`, `source`).
  Block names: `aether3d world blocks` or the `info` query.

## Look before and after

Queries (read-only; in a script's `queries` or MCP `inspect_world`):

* `info` -- mode, seed, extent, biomes, features, structures, the regional stage's cost
  estimate, caches;
* `sample` -- `points` [[x, y]...]: `height` (authoritative), `normal`, `slope` (degrees),
  `biome`, `biome_weights`, `temperature`, `moisture`, `water` (level or null),
  `water_kind`;
* `profile` -- heights, slopes, biomes and water along a polyline;
* `sites` -- candidate building sites, best first; `route` -- a road plan between points;
* `block` -- the block at a position (block worlds); `stats` -- generation and memory
  counters; `structures`, `features`.

Then render (`render_preview` / `render_image`): worlds draw through every backend. For
outdoor views give the scene a sun and turn on the atmosphere
(`scene.world.atmosphere`: sky, sun glow, height fog), use `shadows` with `cascades: 4`
on the GPU, and `tonemap: "neutral"` for bright skies. Look at the image if you can see
images; otherwise rely on queries (heights, biomes, water) to check placement.

## Good habits

* Query heights before placing cameras or objects: `sample` gives the ground; cameras a
  few metres above it see the landscape, not the inside of a hill.
* Keep `size` modest while iterating (2-4 km); the first render computes the regional
  stage (seconds) and then caches it.
* Prefer features to raw configuration changes for local edits: they keep the rest of
  the world identical (tiles elsewhere are reused).
* Structures and roads written by `build_settlements` are ordinary items and features:
  inspect them with the `structures` and `features` queries, remove what you do not want.
* Caves (`caves.enabled`, or `cave` features) and overhangs are volumetric near the
  camera (`streaming.volume_distance`); beyond, the ground is a heightfield.
