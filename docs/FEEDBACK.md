# Visual feedback and automatic corrections

Aether3D can look at what it renders, check it against what was asked for, and correct
it: **observe -> diagnose -> propose -> apply -> verify**, by hand or in a loop.

```python
from aether3d import FeedbackSession, Scene

scene = Scene("Reading corner")
scene.add_plane("Floor", size=8)
scene.add_cube("Table", size=1, position=(0, 0, 0.5))
scene.add_cube("Vase", size=0.2, position=(0.1, 0.1, 1.4))  # floats 0.3 above the table
scene.add_camera("Camera", position=(3, -4, 2.5), look_at=(0, 0, 0.7))
scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1.0), intensity=1.5)

feedback = FeedbackSession(
    scene,
    {
        "format": "aether3d.expectations",
        "version": 1,
        "views": {"main": {"camera": "Camera"}},
        "checks": [
            {"check": "rests_on", "objects": "Vase", "support": "Table"},
            {"check": "visible", "objects": ["Table", "Vase"]},
        ],
    },
)
diagnosis = feedback.diagnose()  # renders the view once; checks the rules
print([(f.code, round(f.magnitude, 3)) for f in diagnosis.findings])  # [('FLOATING', 0.3)]
corrections = feedback.propose()  # "Lower 'Vase' 0.3 m down onto 'Table'" (safe)
result = feedback.apply()  # one undo step; diagnosed again: kept, the finding resolved
print(result.status, result.verification["resolved"] != [])  # applied True
scene.history.undo()  # the correction, in one step
report = feedback.run()  # or the whole loop: 'accepted'
```

The engine contains no model and makes no network request. What it measures -- geometry,
pixels, light -- it reports as facts, labeled with how they were measured. What only a
model or a person can judge -- whether a render looks right, whether an object is
recognizable -- comes in as a **vision evaluation**: a checked document from an
image-aware evaluator (over MCP: the agent itself, looking at the images), grounded in
the render's object index and never treated as a fact.

Contents:
- [Observations](#observations)
- [Views](#views)
- [Expectations](#expectations)
- [Findings and what they rest on](#findings-and-what-they-rest-on)
- [Vision evaluations](#vision-evaluations)
- [Corrections](#corrections)
- [Applying and verifying](#applying-and-verifying)
- [The loop](#the-loop)
- [Command line](#command-line)
- [MCP](#mcp)
- [Safety](#safety)
- [Design](#design)
- [Performance](#performance)
- [Limitations](#limitations)

## Observations

`scene.observe(view, ...)` (or `aether3d.feedback.observe(scene, view, ...)`) renders one
view with an existing backend -- the CPU rasterizer by default, `backend="gpu"`/`"auto"`
for the GPU rasterizer, `"path"`/`"gpu_path"` with `path={"samples": N}` for the path
tracers -- and measures what the image shows. It never changes the scene: no object, no
history step, no revision.

| Fact | How it is measured |
| --- | --- |
| `visible_pixels`, `screen_bounds` `[x0, y0, x1, y1)`, `center`, `touches_border` | the render's object-index buffer: one representative sample per pixel (the center sample on the CPU; on the GPU with MSAA the first sample of the center block; the path tracers' center rays) |
| `depth_range` | the render's depth buffer (distance along the view direction) |
| `luminance` (per object and for the image) | linear light when the render returned it (`linear=True`), else decoded from the 8-bit sRGB pixels (values above 1 are clipped there); Rec. 709 weights |
| `geometry.frustum` | the exact triangle-frustum test of the spatial queries (`objects_in_view`) |
| `geometry.projected_bounds`, `in_frame_fraction` | the geometry between the near and far planes, projected exactly (vertices in the slab and edge crossings of its planes); the fraction of that box inside the image |
| `geometry.near_clipped`, `far_clipped` | whether a clipping plane cuts the geometry inside the image |
| `geometry.occlusion` (`occlusion=True`) | the object's **unoccluded footprint** -- its triangles rasterized alone at pixel centers with the CPU rasterizer's exact rules (clipping, fixed-point edge functions, culling, single-sided materials) -- against the object index: the visible fraction and the objects covering the rest |
| image `coverage`, `black_fraction`, `clipped_fraction`, `luminance` statistics | the image's buffers |

Observations keep their buffers so later steps can use them: `obs.objects_in_rect(rect)`
(what a region shows), `obs.png(kind)` with kinds `image`, `annotated` (numbered markers
and outlines on the main or focus objects; `obs.markers()` maps numbers to ids), `ids`
(object colors), `depth` and `mask`, and `obs.crop_png(rect, scale=)` (enlarged by pixel
repetition, never resampled). `obs.to_dict()` is the JSON report; `obs.sha256` is the
digest of the pixels.

Observations of the same view with the same options at the same scene revision are reused
from an `ObservationStore` (renders are deterministic); the store keeps them within a
memory budget and says `OBSERVATION_EXPIRED` for one it dropped. An observation made inside
an open transaction (a preview) is never stored or reused: the revision does not count its
changes yet.

## Views

| View | Means |
| --- | --- |
| `{}` or `None` | the active camera (without one: an `iso` preset view) |
| `{"camera": "Cam"}` | a scene camera, by id or name |
| `{"virtual": {"position": [x, y, z], "look_at": [x, y, z] \| "direction": [x, y, z], "up": [0, 0, 1], "fov": 40, ...}}` | a camera that is not part of the scene |
| `{"focus": {"objects": SELECTOR, "direction": [...], "margin": 0.08, "from_camera": "Cam"}}` | a virtual camera framed exactly on objects (the closed-form framing of `rendering/framing.py`), with that camera's lens and direction by default |
| `{"preset": "top" \| "bottom" \| "front" \| "back" \| "left" \| "right" \| "iso", "objects": SELECTOR}` | a framed view along a fixed direction: `top` looks down with +Y at the top of the image; `front` looks along +Y at the objects' fronts (assets face -Y); `left`/`right` from the -X/+X sides |

`aether3d.render(scene, camera=...)` takes a virtual camera (a `RenderCamera` without an
object id; `aether3d.feedback.virtual_camera(position, look_at=...)` builds one), and so do
the camera queries (`project`, `pick`, `objects_in_view`, `camera_ray`): a virtual twin of
a scene camera (`feedback.camera_view(scene, "Cam")`) renders and projects bit for bit like
the camera itself (tested on the CPU and the GPU).

## Expectations

An expectations document states what the scene must satisfy, as checks:

```json
{
  "format": "aether3d.expectations",
  "version": 1,
  "label": "Living room",
  "views": {"main": {"camera": "Camera"}, "top": {"preset": "top"}},
  "checks": [
    {"check": "exists", "objects": {"tags": ["chair"]}, "count": {"min": 4}},
    {"check": "rests_on", "objects": {"tags": ["furniture"]}, "support": "Floor"},
    {"check": "rests_on", "objects": "Vase", "support": "Table"},
    {"check": "no_overlap", "objects": {"tags": ["furniture"]}, "against": "Walls",
     "fix": {"move": "objects"}},
    {"check": "visible", "objects": ["Sofa", "Lamp"], "min_visible_fraction": 0.5},
    {"check": "framing", "objects": "Sofa", "region": [0.2, 0.25, 0.8, 0.85],
     "fill": {"min": 0.08, "max": 0.5}},
    {"check": "exposure", "mean_luminance": {"min": 0.1, "max": 0.5},
     "max_clipped_fraction": 0.02},
    {"check": "material", "objects": "Rug", "material": "Wool"},
    {"check": "upright", "objects": {"tags": ["furniture"]}, "max_tilt": 2},
    {"check": "camera_clipping", "objects": "Sofa"}
  ],
  "acceptance": {"max_errors": 0}
}
```

| Check | Holds when |
| --- | --- |
| `exists` | the selected objects exist, `count` `{min, max}` of them (default at least one); a named object that does not exist is `MISSING_ELEMENT` |
| `rests_on` | each object rests on its `support` (or, without one, on whatever is directly below: the objects under its footprint whose tops lie within its lowest quarter -- not a wall it stands against or a table beside it): no vertical gap and no penetration beyond `tolerance` (default 0.1 mm) |
| `no_overlap` | the objects do not intersect each other (or the `against` objects); touching within `tolerance` is allowed unless `allow_contact: false` |
| `visible` | each object is rendered, in the view volume, shows at least `min_pixels` (default 1), at least `min_visible_fraction` of its unoccluded footprint, and lies wholly in the frame with `in_frame: true` |
| `framing` | the objects' visible center lies in `region` (normalized `[x0, y0, x1, y1]`), their pixels cover `fill` `{min, max}` of the image and their screen height `height` `{min, max}` |
| `exposure` | the mean linear luminance of the objects' pixels (`over: "image"`: of the whole image) lies in `mean_luminance`, at most `max_clipped_fraction` of the pixels are clipped and `max_black_fraction` black, geometry covers at least `min_coverage` |
| `inside_region` | each object's world box lies in `region` (`{"box": {"min", "max"}}`) |
| `distance` | the surface distance between `a` and `b` lies in `[min, max]` |
| `relation` | `a`'s `predicate` relation to `b` (`above`, `below`, `in_front_of`, `behind`, `inside`, `contains`, `near` with `near_distance`, `intersecting`) is `expect` (default true) |
| `upright` | each object's local `axis` (default +Z) is within `max_tilt` degrees (default 2) of world +Z |
| `material` | each object uses the material (by name or id) |
| `camera_clipping` | the view's near and far planes cut none of the objects (default: every visible object) |
| `mesh_validity` | each object's mesh (a group: its parts' meshes) reaches the validity `tier` (`valid`, `manifold`, `closed` -- the default --, or `solid`, which runs the exact self-intersection test; [MODELING.md](MODELING.md#validity)); below it: `MESH_BELOW_TIER` (basis `geometry`, exact) with the counts. Never corrected: corrections do not edit meshes (0.13.0) |
| `uv_quality` | each object's mesh (a group: its parts' meshes) has UVs (`NO_UVS`) with at most `max_flipped` flipped (`UV_FLIPPED`) and `max_overlapping` overlapping triangles (`UV_OVERLAP`; default 0 each; stacked islands -- deliberate copies -- are not counted), and optionally an L2 stretch of at most `max_stretch` (`UV_STRETCH`), a texel-density spread of at most `max_density_spread` (`UV_DENSITY_SPREAD`) and gaps of at least `min_gap` texels (`UV_GAP`) at `resolution` (default 1024); basis `geometry`, exact ([TEXTURING.md](TEXTURING.md#diagnostics)). Never corrected: corrections do not edit UVs (0.14.0) |
| `cad_valid` | the CAD `items` (default: every item) regenerate (`CAD_ERROR`, basis `scene`, with the item's error), every part body is a valid solid (`CAD_INVALID_SOLID`), and with `fully_constrained` no sketch has degrees of freedom left (`CAD_UNDERCONSTRAINED`, magnitude the DOF); exact, from the CAD document ([CAD.md](CAD.md)). Never corrected (0.15.0) |
| `cad_dimension` | an exact measurement of a CAD `reference` (and `other`): `quantity` (`distance`, `parallel_distance`, `axis_distance`, `angle_deg`, `length`, `area`, `radius`) equals `value` within `tolerance`, or lies within `min`..`max` (document units, degrees); `CAD_DIMENSION` with the difference as magnitude, `CAD_REFERENCE` for a reference that does not resolve, `CAD_NOT_MEASURABLE` for a quantity that does not apply. Never corrected: change the driving dimension (0.15.0) |
| `animation_continuity` | the objects' animation curves (their actions) have no unintended jump at any key up to `order` -- 0 value, 1 velocity (the default), 2 acceleration -- from the curves' exact one-sided limits; constant keys and jump pairs are intended. `MOTION_DISCONTINUITY` (basis `scene`, exact) with the jump as magnitude; corrected by `smooth_keys` (`approval`) (0.17.0) |
| `foot_contacts` | the armatures' planted `feet` (joint names) slide at most `max_sliding` (default 0.005 m) and go below the ground (z = 0) at most `max_penetration` (0.002 m) over the timeline, sampled at 120 per second (planted: within `height`, 0.03 m, of the ground and within `speed`, 0.3 m/s, of the stance's steady velocity, so a skating stance is measured); `FOOT_SLIDING`, `FOOT_PENETRATION` (basis `scene`, sampled); corrected by `plant_feet` (`verify`) ([ANIMATION.md](ANIMATION.md#foot-contacts-and-planting); 0.17.0) |
| `rig` | the selected armatures' rigs pass the rig diagnostics ([CHARACTER_RIGGING.md](CHARACTER_RIGGING.md#diagnostics-and-verified-correction)): the skeleton, every skinned mesh's weights and binding, the deformation in `poses` (default `"suite"`: elbows and knees bent 120 degrees, hips flexed 90, shoulders raised, wrists and torso twisted, the head turned; or a list of pose names) and, with `timeline` (default true), the animation (IK flips, unreachable targets, sliding feet); codes in `ignore` are skipped. Each finding keeps the rig's code (`JOINT_COLLAPSE` -- magnitude the fraction of size lost --, `FLIPPED_TRIANGLES`, `WEIGHT_LEAK`, `IK_FLIP`, ...), its severity (never above the check's) and the measurement as evidence (basis `geometry`, sampled); `RIG_MISSING` for an object with no rig; corrected by `rig_fix` (0.19.0) |

Every check takes `id`, `severity` (`"error"`, default, or `"warning"`), `label`, and
`fix`: `move` (`"objects"`, `"against"`, `"support"` or `"none"`: which side a correction
moves), `camera` (`false`: never move the camera for it) and `allow` (the actions it may
use). `objects` takes ids, names or the selectors of scene plans (`{"tags": [...]}`,
`{"name_glob": "Chair*"}`, `{"under": "Room"}`). View checks use `view` (a name of
`views`; default `default_view`, or `main`).

**Only stated rules are checked.** Whether an object floats matters only if a `rests_on`
check says it must rest: lamps hang and birds fly. **Default checks** run unless
`"defaults": false`: a black image or a view that shows nothing is an error; render
warnings, an exposure far off (mean luminance below 0.01 in linear light, or more than half
of the objects' pixels clipped) and penetrations among visible objects are warnings or
information -- labeled `heuristic` or `info`, so they never pass for stated expectations.
The penetration scan (`POSSIBLE_PENETRATION`, `POSSIBLE_CONTAINMENT`) decides
intersections on triangles but bounds their depth by boxes (`exact: false`); it takes a
group of parts -- an object whose children have no children (a chair of boards, a lamp, a
tower of blocks) -- as one object, does not repeat a pair a stated `rests_on` or
`no_overlap` check already reports, and covers at most 500 visible objects (`unknown`
beyond: state `no_overlap` checks).

Documents are checked as a whole, like scene plans: every problem with a stable code
(`INVALID_TYPE`, `INVALID_VALUE`, `OUT_OF_RANGE`, `UNKNOWN_KEY` with a "did you mean",
`MISSING_KEY`, `DUPLICATE_ID`, `UNKNOWN_VIEW`, `LIMIT_EXCEEDED`, ...) and a JSON pointer.
The JSON Schema is `aether3d feedback schema expectations`; the tests check that it refuses
nothing the checker accepts.

## Findings and what they rest on

A diagnosis (`feedback.diagnose()`) lists every check's status (`pass`, `fail`,
`unknown`) and its **findings**:

| Field | Meaning |
| --- | --- |
| `code` | what was found: `FLOATING`, `SUNKEN`, `NOT_ON_SUPPORT`, `NO_SUPPORT`, `INTERSECTION`, `CONTAINED`, `HIDDEN`, `OUT_OF_VIEW`, `OCCLUDED`, `TOO_SMALL`, `NOT_VISIBLE`, `NOT_RENDERED`, `NO_GEOMETRY`, `PARTIALLY_OUT_OF_FRAME`, `OFF_CENTER`, `TOO_SMALL_IN_FRAME`, `TOO_LARGE_IN_FRAME`, `UNDEREXPOSED`, `OVEREXPOSED`, `CLIPPED_HIGHLIGHTS`, `BLACK_AREAS`, `EMPTY_VIEW`, `BLACK_IMAGE`, `NEAR_CLIPPED`, `FAR_CLIPPED`, `OUT_OF_REGION`, `TOO_CLOSE`, `TOO_FAR`, `RELATION_VIOLATED`, `TILTED`, `WRONG_MATERIAL`, `MISSING_MATERIAL`, `MISSING_ELEMENT`, `TOO_MANY`, `RENDER_WARNING`, `POSSIBLE_PENETRATION`, `POSSIBLE_CONTAINMENT`, `VISION_<CATEGORY>` |
| `basis` | `geometry` (computed from the triangles), `render` (measured on a rendered image), `scene` (the scene's data), `heuristic` (a default threshold on measured facts), `vision` (an image-aware evaluation, as declared), `agent` (a judgment declared without image input) |
| `exact` | whether the measurement is exact for its definition |
| `magnitude`, `unit` | how far off: meters, pixels, fractions, stops, degrees |
| `subjects` | the objects, by id and name, with their role (`subject`, `support`, `against`, `occluder`, ...) |
| `region`, `observation`, `view` | where the finding's main object appears in the default view's image |
| `evidence` | the measured values behind it |

**How placement is measured.** `rests_on` first asks whether the object's surfaces cross
a support's (the exact triangle test of the spatial queries). If they do, the
**penetration** is the upward translation that separates them, found by bisection and
verified separated by the same exact test -- correct for any shape, including a leg
piercing a rug with no vertex inside it. If nothing is crossed, the **gap** comes from
vertical rays (the watertight ray test of the path tracer) from the object's vertices down
onto the supports' tops and from the supports' vertices under it up: exact where the
extreme is reached at a vertex over a face -- an object standing on a floor or a table; an
edge resting across an edge can be measured too large. `no_overlap` decides intersection
with the exact triangle test; two objects merely *touch* when moving one by `tolerance`
along one of the six axis directions separates them (the exact test again), and otherwise
they penetrate, with the shortest separating axis translation found by bisection (each one
verified separated; the true minimum translation can be shorter in other directions).

**Groups are objects.** An object with children -- a prefab of blocks, a table of boards,
an imported hierarchy -- is one object for every check, observation and correction: its
geometry is the union of its own mesh and its visible descendants', its pixels those of all
of them, and a correction moves the group (the children follow). A check that selects a
group never reports its parts against each other, and an implicit support is never one of
the object's own parts.

A diagnosis is **accepted** when the error findings of deterministic checks are at most
`max_errors` (and warnings at most `max_warnings`), no error check is `unknown`, and --
only with `"require_vision": true` -- a fresh vision evaluation says the image matches
the request and reports no error. Vision findings never count otherwise.

## Vision evaluations

```json
{
  "format": "aether3d.vision-evaluation",
  "version": 1,
  "observation": "obs-3f9a6c2e8b1d",
  "image_sha256": "c5d9...",
  "provenance": {"provider": "anthropic", "model": "claude-opus-5-5", "image_input": true,
                 "method": "looked at the annotated image of observe_scene"},
  "overall": {"matches_request": false, "score": 0.6, "summary": "the lamp is cut off"},
  "findings": [
    {"category": "framing", "severity": "warning", "confidence": 0.8,
     "summary": "the lamp is cut off by the right border",
     "region": [560, 40, 640, 300], "objects": ["Lamp"],
     "suggestion": {"action": "frame_camera", "targets": ["Lamp", "Sofa"]}}
  ]
}
```

`check_evaluation(document, observation)` (and `diagnose(evaluations=[...])`) checks the
document strictly (every problem with a pointer; at most 50 findings, bounded text, no
control characters, suggestions from the action allowlist only), ties it to its
observation (id and pixel digest: `OBSERVATION_MISMATCH` otherwise) and marks it stale when
the scene changed since. Then it **grounds** each finding in the observation's object
index: which objects occupy the region, and whether the objects it names are visible
there. A finding naming an object that is not in its region is `ungrounded`, and its
target is never used for a correction.

**Provenance is declared, delivery is recorded.** Findings get `basis: "vision"` only when
the provenance declares an image input, `"agent"` otherwise; the report shows
`image_input: "declared"` and `image_delivered_by_aether3d` -- whether the engine itself
handed that observation's image out (as MCP image content, a resource, or to a critic). It
never states that a model saw an image.

**Critics.** `run_loop(..., critic=...)` asks a critic about each iteration's main
observation: any callable taking a `CritiqueRequest` (the clean and annotated PNGs, the
markers, the observation's summary, the intent, instructions) and returning an evaluation
document. Calls are budgeted (`max_critic_calls`, `max_cost` with the critic's `cost`); a
failing or malformed critic is recorded and the loop goes on. No critic ships with
Aether3D: wire one to the model provider you use, in your code. Over MCP, the client's own
model is the evaluator; MCP sampling (deprecated in the 2026-07-28 protocol, SEP-2577) is
not used.

## Corrections

`feedback.propose()` turns findings into **proposals**. A proposal is a list of
**actions** from a fixed allowlist -- nothing else can change:

| Action | Changes |
| --- | --- |
| `translate` (`offset`, `frame`: `world` or `local`), `place` (`position`), `rotate` (`rotation_euler` or `rotation`), `scale` (`factor`) | an object's placement; world offsets and positions are converted exactly through the parent's world matrix |
| `set_visibility`, `assign_material`, `set_color` | an object's visibility, material (an existing one) or color |
| `frame_camera` (`targets`, `direction`, `margin`), `aim_camera` (`look_at`), `dolly_camera` (`distance`), `set_camera` (`fov`, `near`, `far`) | a camera |
| `set_light` (`intensity`, `color`), `add_light`, `set_exposure`, `set_ambient` | light |
| `set_parameter` | a procedural scene parameter (off by default: `allow_procedural`) |
| `plant_feet` (`armature`, `legs`: foot -> `[hip, knee, foot]`), `smooth_keys` (`object`, `path`, `keys`, `handles`) | animation (off by default: `allow_animation`; 0.17.0): keyed IK pins each foot contact where it touched down; the keys of a channel get automatic handles (`auto_clamped` by default: no corners, no overshoot) |
| `rig_fix` (`armature`, `fixes`: proposals from `propose_rig_fix` / `inspect_rig` `fixes`, or `"safe"` / `"verify"`) | animation (`allow_animation`; 0.19.0): the rig's own corrections -- weight edits, dual quaternion blended in around a collapsing joint, rebinding, a pole moved off the chain's line, planted feet -- each checked as data before anything changes and applied by `apply_rig_fix`, which measures again after each and undoes what did not help; the session then verifies the step as for any correction |

No deletion, no mesh editing, no file, no network, no code. Actions **compile** to the
engine's own documents -- a scene plan (`update`, `frame_camera`, `set_parameters`,
`world`, `render_settings`, and since 0.17.0 `animation`, whose animation script runs
`plant_feet` or `edit_keys`) and, for lenses and lights, a scene script -- shown in the
proposal (`compiled`), so what will run can be read and run by hand.

Each proposal carries its `findings`, `rationale` (why, from the evidence), `expected`
effect (the magnitude after), `validation` (what verification checks), `alternatives`,
`changes` (before and after values) and a **safety** class:

| Safety | When | Applied |
| --- | --- | --- |
| `safe` | a deterministic, exact repair of a stated expectation: lower a floating object by its measured gap, raise a sunken one, assign the stated material, show a hidden object that must be visible, move a clipping plane outward | automatically (default policy) |
| `verify` | deterministic but approximate or with side effects: separate intersecting objects (the side `fix.move` names), re-frame, aim or move a camera, change the exposure, stand an object upright, move it into its region | when the policy allows it (`auto_apply: ["safe", "verify"]`) or it is named; kept only if verification shows improvement |
| `approval` | subjective or ambiguous: which of two objects to move when nothing says, hiding an occluder, adding a light, anything a vision evaluation or a default heuristic suggests | only with `approve=[ids]` |

Findings no planner corrects deterministically -- a missing object, an object inside
another, an unmet relation -- are listed as `unresolved`, with the reason: they need a
decision (a plan of the agent's).

**Policy** (`CorrectionPolicy`, or its JSON): `auto_apply`; `allow_placement`,
`allow_camera`, `allow_lighting`, `allow_materials`, `allow_visibility`,
`allow_procedural`, `allow_animation` (0.17.0); limits per application `max_actions`, `max_objects`,
`max_translation` (default a quarter of the scene's diagonal, at least 1 m),
`max_rotation` (90 degrees), `max_scale` (4), `max_exposure_change` (4 stops),
`max_light_factor` (8); `verify`, `rollback_on_regression`, `rollback_if_ineffective`;
`min_vision_confidence` (0.6).

## Applying and verifying

`feedback.apply(corrections=None, ids=None, *, actions=None, approve=(), policy=None,
verify=None, expected_revision=None, label=None)`:

1. refuses proposals made at another revision (`STALE_PROPOSALS`) or a scene not at
   `expected_revision` (`REVISION_MISMATCH`), unknown ids, and actions beyond the policy
   (`NOT_ALLOWED`, with every violation);
2. applies at most one proposal per changed object (and one exposure change) -- the others
   wait for the next diagnosis, which sees the first one's effect;
3. runs the compiled plan and script as **one history transaction**: one undo step, all or
   nothing (`APPLY_FAILED` rolls back);
4. **verifies**: diagnoses again (observing again only what changed -- the retained render
   world and the spatial index update incrementally) and compares: findings `resolved`,
   `persisting` (with magnitudes before and after), `new_failures`. It undoes the step
   (`history.undo()`, exact) when a check that passed now fails, or when the findings it
   addressed did not improve by at least 1 % of their magnitude. An agent's own `actions`
   are kept unless they regress.

`feedback.preview(corrections, ids)` applies proposals inside a transaction, diagnoses and
observes, and rolls back: the scene, its revision and its history are unchanged.

## The loop

`feedback.run(budget=..., policy=..., critic=..., cancel=..., progress=..., resume=...)`
(`run_loop`) repeats: diagnose (reusing the last verification's diagnosis), optionally
critique, propose, apply the proposals the policy allows (one undo step each), verify --
until a stop reason:

| `stop_reason` | |
| --- | --- |
| `accepted` | the expectations hold |
| `awaiting_approval` | only proposals that need approval remain (`pending_approval` lists them): decide, `apply(..., approve=[...])`, then `run(resume=loop_id)` |
| `no_safe_progress` | nothing remaining can be corrected automatically |
| `stagnated` | `patience` iterations without improvement, or every remaining proposal was tried and undone |
| `cycle_detected` | a finding resolved earlier came back |
| `max_iterations`, `budget_renders`, `budget_pixels`, `budget_time`, `budget_memory`, `budget_cost` | a budget of `FeedbackBudget` ran out (defaults: 5 iterations, 60 renders, 60 M pixels, 120 s, 512 MiB of observations, 3 critic calls) |
| `cancelled` | the `cancel` event was set (between steps: a step in progress finishes or rolls back) |

The report lists every iteration (diagnosis summary, proposals, what was applied, the
verification and decision, timings) and the budget used. Before its first change the loop
sets the checkpoint `feedback:<loop id>:before`: `scene.history.goto(...)` returns to the
start in one call.

## Command line

```console
$ aether3d feedback observe room.aether --out obs/ --images image,annotated,ids --json
$ aether3d feedback observe room.aether --view '{"preset": "top"}' --backend gpu
$ aether3d feedback check room.aether --expect room.expect.json        # exit 3: not accepted
$ aether3d feedback check room.aether --expect room.expect.json --evaluation eval.json
$ aether3d feedback correct room.aether --expect room.expect.json --preview
$ aether3d feedback correct room.aether --expect room.expect.json --apply --history
$ aether3d feedback loop room.aether --expect room.expect.json --budget '{"max_iterations": 4}'
$ aether3d feedback schema expectations
```

`correct --apply` and `loop` save the project (`--output` elsewhere, `--dry-run` not at
all); a project with a history file records each correction, so `aether3d history undo`
takes it back. Observations default to the CPU rasterizer at 640 x 360 (`--backend`,
`--width`, `--height`). Evaluations name observations by id: the same view of the same
project renders the same image, so ids repeat across runs.

## MCP

`aether3d mcp serve` adds five tools (docs/MCP.md):

| Tool | Kind | Does |
| --- | --- | --- |
| `observe_scene` | inspect | render views; images as image content (`image`, `annotated`, `ids`, `depth`, crops), measured facts as JSON |
| `diagnose_scene` | inspect | check expectations (kept for later calls) and the agent's own evaluations |
| `propose_corrections` | propose | proposals with their compiled plans; `preview: true` applies, observes and rolls back |
| `apply_corrections` | change | proposals or the agent's own actions, one undo step, verified |
| `feedback_loop` | change | the loop, with progress and cancellation; `resume` after approvals |

Resources: `aether3d://feedback/guide` (the guide for agents), the schemas
`aether3d://schema/expectations`, `.../vision-evaluation`, `.../corrections`, and
`aether3d://observations/{id}` (JSON) and `.../{id}/{kind}` (PNG). The prompt
`visual_feedback` walks an agent through the loop. `--feedback-backend` (`auto`, default:
the GPU rasterizer when one works; `gpu`; `cpu`) chooses how observations render.
Replies carry at most 8 images (6 x 1920 x 1080 pixels).

## Safety

- **Data, not code.** Expectations, evaluations, policies, budgets and actions are JSON
  checked against strict rules; nothing in them is executed. Actions come from the
  allowlist; their targets must exist and be of the right kind; magnitudes are bounded.
- **Untrusted evaluations.** Evaluation documents are bounded (1 MiB, 50 findings, 500
  characters per summary), tied to one observation and its pixels, grounded in the object
  index, labeled by declared provenance, and can only make approval proposals.
- **No silent changes.** Observing, diagnosing, proposing and previewing change nothing.
  Applying changes the scene in one undo step and reports it; nothing writes files except
  the CLI's explicit saves and `--out` reports; the MCP tools write no files.
- **Bounded work.** Observation sizes (`ObservationLimits`: 1920 x 1080 pixels, path
  tracing at most 50 M samples, 16 occlusion footprints), loop budgets, at most 200
  intersecting pairs classified per check and 500 visible objects scanned by default.

## Design

**Languages, chosen per component before writing it:**

| Component | Choice | Why |
| --- | --- | --- |
| rendering the observations | the existing backends (CPU NumPy rasterizer, GPU wgpu rasterizer, path tracers) | Stage 12 adds no renderer: it reuses the retained render world, the kept GPU frame data and selective readback (image, object index, depth; linear light only for exposure checks) |
| per-pixel and per-object statistics | NumPy (`bincount`, `minimum.at`/`maximum.at`, `percentile`) | one vectorized pass over buffers the renderers already return as NumPy arrays: no copy across a language boundary. Measured: the per-object statistics of 3,000 objects take 5.9 ms at 640 x 360 and 16.2 ms at 1920 x 1080; all of an observation's measurements, 11-15 ms at 640 x 360 (below) |
| geometry (support, contact, separation, footprints) | the spatial queries' NumPy kernels and ray core (Rust traversal when `aether3d-native` is installed), and the CPU rasterizer's exact integer raster for footprints | exact kernels already exist and are tested against brute force |
| documents, grounding, planning, policy, loop, MCP, CLI | Python | control flow over small data, dominated by renders and queries; direct use of the history, plans and scripts |
| GPU | no new WGSL | the finalize pass already produces the object index, depth and coverage |
| Rust | no new kernel | nothing measured hot enough; AGENTS.md requires a NumPy twin and bitwise parity for every native kernel |

No dependency was added.

**Corrections reuse the engine's operations.** Actions compile to scene plans and scene
scripts and run in one history transaction, so they are undone like any change, keep
object identity, hierarchy, procedural definitions and materials, and save with the
project's history.

## Performance

Measured with `benchmarks/bench_feedback.py` on the development machine (Intel Core
i9-12900H, NVIDIA GeForce RTX 4060 Laptop GPU over Vulkan, CPython 3.14, with
`aether3d-native`; the run is `benchmarks/results/stage12-feedback.json`). Observations at
640 x 360; medians of repeated runs. Compare on the same machine only.

| One observation | the room (34 objects) | the castle (2,190) | the village (11,270) |
| --- | ---: | ---: | ---: |
| GPU rasterizer, after one object moved | 24.0 ms | 26.8 ms | 46.2 ms |
| the same with the full extraction instead of the retained world | 24.6 ms | 36.8 ms | 96.4 ms |
| CPU rasterizer, after one object moved | 1.14 s | 2.60 s | 3.10 s |
| of which the measurements (every object's pixels, depth, luminance; the image's statistics) | 11.0-11.7 ms | 11.1-11.7 ms | 14.8-15.0 ms |
| the same view again, the scene unchanged (reused from the store) | 0.013 ms | 0.012 ms | 0.012 ms |

| Diagnosis, correction, loop | the room | the castle / the garden |
| --- | ---: | ---: |
| the checks of a diagnosis, its renders made (room: 6 checks; castle: 4) | 240 ms | 312 ms |
| a first diagnosis, GPU / CPU rasterizer | 290 ms / 1.41 s | 559 ms / 3.06 s |
| the same diagnosis again, the scene unchanged | 0.014 ms | 0.012 ms |
| proposing (3 proposals) | 0.6 ms | |
| a preview (apply, observe, diagnose, roll back), GPU / CPU | 110 ms / 1.23 s | |
| applying and verifying (one undo step, a new render, a new diagnosis), GPU / CPU | 100 ms / 1.22 s | |
| the loop from the first draft to accepted, GPU / CPU (room: 1 iteration, 2 renders; garden: 2 iterations, 3 renders) | 390 ms / 2.64 s | garden: 179 ms / 5.58 s |

What this shows:

- **Renders are minimized, not free.** Each diagnosis renders each view once; the loop
  reuses a verification's diagnosis as the next iteration's, so `n` iterations cost
  `n + 1` renders of a view. Unchanged views are reused (0.01 ms) instead of rendered.
- **On the CPU rasterizer the renders are the cost** (about 99 % of an observation).
- **On the GPU the checks can dominate**: exact triangle tests decide contacts and
  separations. Most of the room's 240 ms is the separation search for the lamp inside the
  wall (six axis directions, each bisected, every step an exact intersection test).
- **The retained render world** makes an observation after one move at 11,270 objects
  2.1 times faster (46.2 ms instead of 96.4), 1.4 times at 2,190 (26.8 ms instead of
  36.8); at 34 objects, and on the CPU rasterizer, the difference is within the noise.
- The first GPU observation of a process also creates the device and compiles the
  pipelines (228 ms for the room).
- The statistics alone, on synthetic buffers with 3,000 objects: 5.9 ms for the objects
  and 6.7 ms for the image at 640 x 360 (plus 2.5 ms decoding sRGB to luminance when the
  render returned no linear light); 16.2, 45.4 and 22.7 ms at 1920 x 1080.

Every observation, diagnosis, preview and application reports its `timings` (seconds by
stage), each loop iteration its `seconds`, and a loop's report the budget it used
(renders, pixels, time, memory, critic calls).

## Limitations

- **Judgment needs an evaluator.** Whether a render matches a natural-language request is
  never decided by the engine. Without an evaluation, only what expectations state is
  checked.
- **One sample per pixel.** Pixel facts come from the render's representative sample:
  silhouettes can differ by a pixel between backends, and the footprint (pixel centers)
  from the GPU's MSAA sample positions.
- **Alpha masks** count as covered in footprints (cut-out texels included); the object
  index, which follows the renderers' alpha tests, then shows a lower visible fraction.
- **Vertical gaps** are exact for vertex-over-face contacts; an edge resting across an edge
  can be measured too large (the correction then falls short, and verification says so).
  Penetrations and separations are found by bisection to within a quarter of the
  check's tolerance, always rounded up to a translation verified to separate.
- **Separation** is searched along the six axis directions only.
- **Upright corrections** need parents without non-uniform scale.
- **No blended transparency** exists in Aether3D, so none is observed.
