# Aether3D visual feedback: a guide for agents

Build, then **look and check**: observe the rendered scene, state what it must satisfy,
evaluate the images yourself, and correct what is wrong with small, verified, undoable
corrections. Everything here is data: nothing is executed as code.

## The loop

1. `observe_scene` -- render views and get the images plus measured facts. Ask for
   `"images": ["image", "annotated"]`: the annotated image numbers the main objects
   (`markers` maps numbers to object ids). Views: `{}` (the active camera),
   `{"camera": "Cam"}`, `{"preset": "top"}` (also `front`, `back`, `left`, `right`,
   `iso`, `bottom`), `{"focus": {"objects": "Lamp"}}` (a close-up), `{"virtual":
   {"position": [x, y, z], "look_at": [x, y, z]}}`. Virtual views add nothing to the scene.
   `crops` enlarge regions (normalized `[x0, y0, x1, y1]`).
2. `diagnose_scene` with **expectations** (below) and, if you looked at an image, your
   **vision evaluation** of it. Read each finding's `code`, `basis`, `magnitude` and
   `region`.
3. `propose_corrections` -- typed proposals with a safety class. `preview: true` shows
   the result without changing anything.
4. `apply_corrections` with `expected_revision` -- one undo step, verified: it is undone
   automatically if a passing check now fails or nothing improved.
5. Observe again and look. Or let `feedback_loop` repeat steps 1-4 for what is mechanical.

## Expectations

```json
{"format": "aether3d.expectations", "version": 1, "label": "Reading corner",
 "views": {"main": {"camera": "Camera"}},
 "checks": [
   {"check": "exists", "objects": {"tags": ["chair"]}, "count": {"min": 2}},
   {"check": "rests_on", "objects": "Vase", "support": "Table"},
   {"check": "rests_on", "objects": {"tags": ["furniture"]}, "support": "Floor"},
   {"check": "no_overlap", "objects": {"tags": ["furniture"]}, "against": "Wall",
    "fix": {"move": "objects"}},
   {"check": "visible", "objects": "Lamp", "min_visible_fraction": 0.6},
   {"check": "framing", "objects": "Sofa", "region": [0.25, 0.3, 0.75, 0.8],
    "fill": {"min": 0.1, "max": 0.5}},
   {"check": "exposure", "mean_luminance": {"min": 0.12, "max": 0.45},
    "max_clipped_fraction": 0.02},
   {"check": "material", "objects": "Rug", "material": "Wool"},
   {"check": "upright", "objects": {"tags": ["furniture"]}, "max_tilt": 2}
 ],
 "acceptance": {"max_errors": 0}}
```

Checks: `exists`, `visible`, `framing`, `rests_on`, `no_overlap`, `inside_region`,
`distance`, `relation`, `upright`, `material`, `exposure`, `camera_clipping`,
`mesh_validity` (a mesh's validity tier: `{"check": "mesh_validity", "objects": "Head",
"tier": "closed"}`; useful after sculpting or remeshing), `uv_quality` (no flipped or
overlapping UV triangles, optional stretch, texel-density and gap limits:
`{"check": "uv_quality", "objects": "Crate", "max_density_spread": 0.2}`),
`foot_contacts` (planted feet slide and sink at most `max_sliding` / `max_penetration`
over the timeline: `{"check": "foot_contacts", "objects": "Hero", "feet": ["Foot.L",
"Foot.R"]}`), `animation_continuity` (no unintended jump of value, velocity or, with
`order: 2`, acceleration at any key of the objects' actions), `lighting` (how a subject is
lit, measured from the scene's lights at its visible surface: `{"check": "lighting",
"objects": "Hero", "ratio": {"min": 2, "max": 8}, "max_merged_fraction": 0.25, "lights":
["Key", "Rim"]}` -- the key's lit side over its shadow side, the share of the outline
within `min_separation` (10) L* of the background, lights that must reach it) (schema:
`aether3d://schema/expectations`; `capabilities {"topic": "feedback"}` summarizes them).
`objects` takes ids, names or selectors (`{"tags": [...]}`, `{"name_glob": "Chair*"}`).
`severity: "warning"` makes a check advisory. `fix` states which side may move
(`"move": "objects" | "against" | "support" | "none"`) and whether the camera may change
(`"camera": false`): stating it turns ambiguous corrections into deterministic ones.

**Only stated rules are checked.** A floating lamp is a problem only if a `rests_on`
check says it must rest; an intersection is an error only under `no_overlap`. Default
checks always run: a black or empty image is an error; render warnings, heuristic
exposure and penetrations among visible objects are reported as warnings or info. A
group (an object with children: a chair of boards, a prefab of blocks) is one object for
the checks that select it, and the default scan takes each group of parts as one object,
so a chair's legs are not reported against its seat.

## Findings: what is fact and what is judgment

| `basis` | Means |
| --- | --- |
| `geometry` | computed from the triangles (exact unless `exact: false`) |
| `render` | measured on the rendered image (one sample per pixel) |
| `scene` | the scene's data (what exists, flags, materials) |
| `heuristic` | a default threshold on measured facts, not your expectation |
| `vision` | an image-aware evaluation (yours), as declared -- never a fact |
| `agent` | a judgment declared without image input |

## Your vision evaluation

Look at the image, then describe what is wrong:

```json
{"format": "aether3d.vision-evaluation", "version": 1,
 "observation": "obs-...", "image_sha256": "...",
 "provenance": {"provider": "anthropic", "model": "your model", "image_input": true},
 "overall": {"matches_request": false, "summary": "the lamp is cut off at the right"},
 "findings": [
   {"category": "framing", "severity": "warning", "confidence": 0.8,
    "summary": "the lamp is cut off by the right border",
    "region": [560, 40, 640, 300], "objects": ["<lamp id from the markers>"],
    "suggestion": {"action": "frame_camera", "targets": ["<lamp id>", "<sofa id>"]}}]}
```

Set `image_input: true` only if you looked at that image. Regions are pixels of that
observation's image. The engine checks each region against the object-index buffer:
a finding naming an object that is not in its region is `ungrounded`, and its target is
not used. Suggestions become proposals that need approval. Evaluations of an
observation made before the scene changed are marked stale.

## Corrections

Actions (the only changes possible): `translate` (world offset), `place`, `rotate`,
`scale`, `set_visibility`, `assign_material`, `set_color`, `frame_camera`, `aim_camera`,
`dolly_camera`, `set_camera` (fov, near, far), `set_light`, `add_light` (directional:
`direction`; point: `position`; spot or area: `position` and `look_at` or `direction`, an
area light's `size`), `set_exposure`, `set_ambient`, `set_parameter`, and for animation `plant_feet` (keyed IK pins each
contact; `verify`) and `smooth_keys` (automatic handles; `approval`) -- both only with the
policy's `allow_animation: true`. No deletion, no mesh editing, no files.

Safety: `safe` (exact repair of a stated expectation -- applied automatically),
`verify` (applied when the policy allows it, kept only if it measurably helps),
`approval` (subjective or ambiguous -- pass its id in `approve`). Policy limits bound
moves, turns, scales, exposure changes and the number of objects one step changes.
You can also apply your own `actions`: the call is the approval; verification still
undoes them if a passing check breaks.

## The loop tool

`feedback_loop` stops with `accepted`, `awaiting_approval` (decide the listed proposals,
apply them, then `resume` the loop by its id), `no_safe_progress`, `stagnated`,
`cycle_detected`, `cancelled`, or a budget (`max_iterations`, `max_renders`,
`max_pixels`, `max_seconds`, `max_memory`). Each applied correction is one undo step;
`history goto feedback:<loop id>:before` returns to where it started.
