# Animation guide for agents

Aether3D animates without a GUI: everything below works from Python
(`scene.animation`), from JSON animation scripts, from the CLI (`aether3d animation`)
and over MCP (`run_script` with an animation script, `inspect_animation`, renders at a
`time`). Nothing here needs a mouse; the "graph editor" is mathematics on curves, and
graph images are only a picture of them.

## The model in one paragraph

**Authored** state (transforms, fields) is the rest pose and is never touched by
playback. **Clips** hold channels (F-curves for numbers, rotation tracks for
orientations) keyed by **paths**. A **binding** plays clips on a target: an *action*
(the main clip) plus **layers** of **strips** (NLA: replace or additive, masks, blends,
speed, repeat). **Evaluation** at a time is pure: the same time gives the same pose,
seeking has no history. Rendering, picking and observing at a time use that pose; saving
writes clips, not poses. `apply_pose(time)` is the one explicit way to bake a pose into
authored transforms (one undo step).

## Paths

| Path | Value |
| --- | --- |
| `transform.position`, `.x`/`.y`/`.z` | metres |
| `transform.rotation` | quaternion `[x, y, z, w]` (a rotation track) |
| `transform.rotation_euler.x`/`.y`/`.z` | degrees (XYZ) — graph-editable curves |
| `transform.scale.x` ... | factors |
| `visible` | true/false (stepped) |
| `camera.fov`, `camera.focus_distance`, `light.intensity`, `light.color`, `appearance.color` | object fields |
| `material.roughness` ... (target `{"material": NAME}`) | material parameters |
| `joints["NAME"].rotation` / `.position.x` / `.rotation_euler.z` | rig joints (on the armature) |
| `morph["NAME"]` | morph target weight |
| `constraints["NAME"].influence`, `.offset` | constraint mix, path fraction |

## Keys and curves (the graph editor, as data)

A key is `[time, value]` or `{time, value, interpolation, ease, left, right, weighted}`.

* `interpolation`: `constant`, `linear`, `bezier`, or an easing (`sine`, `quad`,
  `cubic`, `quart`, `quint`, `expo`, `circ`, `back`, `elastic`, `bounce`) with `ease`
  `in`/`out`/`in_out` (`back`: overshoot `back`; `elastic`: `amplitude`, `period`).
* handles (`left`/`right`): `{type, dt, dv}` or `{type, slope}`; types `free`, `aligned`,
  `vector`, `auto` (Catmull-Rom, smooth), `auto_clamped` (smooth, **never overshoots**).
  `weighted` keys keep their handle lengths (longer handles hold the motion longer).
* `extrapolation`: `constant`, `linear`, `cycle`, `cycle_offset` (walk cycles that
  advance), `mirror`. Automatic handles are cycle-aware: loops are smooth at the wrap.
* `modifiers`: `noise`, `cycles`, `generator`, `envelope`, `stepped`, `limits` ...

Key operations (`edit_keys`): `set`, `delete`, `move` (dt, dv), `scale_time` (factor,
pivot), `scale_values`, `reverse`, `quantize` (fps), `distribute`, `retime` (time map),
`handle`, `handle_type`, `interpolation`; `keys` selects `"all"`, indices or
`{"from": t0, "to": t1}`.

**Measure, do not guess.** `report` (a curve's continuity at every key from exact
one-sided limits, extrema, overshoot, velocity/acceleration/jerk peaks), `trajectory`
(world-space speed, jerk, smoothness), `diagnose` (scene-wide findings: velocity
discontinuities, overshoot, corrected handles, IK out of reach), `contacts` (feet:
planted intervals, sliding, penetration). `simplify` reduces keys to a bounded error,
`resample` keys every frame, `bake` turns layers/rigs/constraints into plain keys.

## Rigs, skins, IK, constraints

* `add_rig` on an armature object: joints `{name, parent, position, rotation|euler,
  scale, tail, limits}`. Joint channels drive forward kinematics.
* `attach` an object to a joint (rigid parts: blocky characters); `skin` a mesh
  (`weights`: `"envelope"`, `"nearest"` or explicit; `method`: `linear` or
  `dual_quaternion`); `morph_targets` for blend shapes.
* Constraints (`add_constraint`): `ik` (two-bone with `pole`, `fabrik`, `ccd`; `stretch`,
  `soft`), `look_at` (`target_offset`), `copy_location`/`rotation`/`scale`/`transforms`,
  `child_of`, `limit_*`, `follow_path`. Influence animates. Cycles are refused.

## Motion generators (editable keys, not procedural playback)

`generate` with `motion` `idle`, `walk`, `run`, `jump`, `land` (`style` `superhero` or
`soft`), `wave`, `head_turn`: roles are found by joint names (Mixamo, Rigify, blocky
`Leg.L`); walks plant their feet (foot trajectories + two-bone IK; knee-less legs walk a
compass gait) and move the armature by root motion. The result is a clip whose metadata
records the parameters; edit its keys like any other. Measure with `contacts`.

## Cameras and cinematics

`camera_move` (dolly, truck, pedestal, crane, pan, tilt, roll, zoom), `camera_orbit`
(exact circle + look-at), `follow_camera` (critically damped lag), `add_path` +
`follow_path` (constant speed along the path), `shots` (cuts and blends between
cameras; renders without a named camera use the shot's camera).

## Rendering motion

`render_frames` (Python/CLI) writes exact frame times as PNGs atomically, with progress,
cancellation and motion blur (averaging sub-frame renders in linear light); contact
sheets and optional FFmpeg video. Over MCP, render a preview at a `time`.

## Scripts

```json
{"format": "aether3d.animation-script", "version": 1, "label": "Bounce",
 "operations": [
   {"op": "timeline", "fps": 24, "start": 0, "end": 2},
   {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
    "keys": [[0, 2], [0.5, 0], [1, 2]], "interpolation": "bezier",
    "extrapolation": "cycle"},
   {"op": "report", "target": "Ball", "path": "transform.position.z"}
 ]}
```

All or nothing: a failing operation leaves the scene exactly as it was and names the
argument that failed (`operations[1].keys[2]`). Limits: `max_seconds`, `max_keys`.
Read the schema at `aether3d://schema/animation-script`.

## Interchange

glTF export: rigs become joint nodes, skins and morph targets are written exactly,
clips export exactly where glTF can (STEP, LINEAR, slerp, Hermite CUBICSPLINE) and are
fitted within stated tolerances otherwise; layers, constraints and IK are baked. Import
turns skins, morph targets and animations into rigs, skins and clips.

## Honest limits

Squad rotation is C1 only for uniformly spaced keys. Dual-quaternion skinning runs on
NumPy only. Animated material parameters render through the full scene extraction.
Motion blur averages renders (it is not a velocity-buffer blur). Generators make
plausible, measured motion, not motion capture.
