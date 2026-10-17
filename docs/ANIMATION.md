# Animation, rigging, motion curves and cinematics (0.17.0)

Keyframes on exact F-curves, rotation tracks, clips, layers and strips (non-linear
animation), skeletons with forward and inverse kinematics, constraints, rigid
attachments, linear and dual-quaternion skinning, morph targets, editable motion
generators, foot contacts and planting, retargeting, cinematic cameras and shot
sequences, frame rendering with real motion blur, and glTF interchange of skeletons,
skins, morph targets and animations.

Aether3D is headless and AI-native: there is no graph editor window. Everything a graph
editor does is mathematics on curves, reachable from Python (`scene.animation`), JSON
animation scripts, the command line (`aether3d animation ...`), scene plans
(`{"op": "animation", "script": ...}`) and MCP (`run_script`, `inspect_animation`,
renders at a `time`). Graph images (SVG or PNG) are a picture of the curves for people
and for agents that can look at images; nothing is edited through a picture. The design
record -- what was inspected, the research behind each decision, the language chosen for
every subsystem, and the outcome -- is
[research/2026-10-stage17/DESIGN.md](../research/2026-10-stage17/DESIGN.md).

Stage 17 was built before Stage 16 (procedural worlds), which is postponed; there is no
0.16.0.

## Quick start

```python
from aether3d import Scene

scene = Scene()
ball = scene.add_sphere("Ball", radius=0.25, position=(0, 0, 1))
anim = scene.animation
anim.set_timeline(fps=24, start=0, end=2)

# A bounce that loops: keys in seconds; automatic handles are smooth and never overshoot.
anim.set_channel(ball, "transform.position.z", [(0, 1.0), (0.5, 0.25), (1, 1.0)],
                 extrapolation="cycle")
print(round(anim.evaluate(1.5).position[0][2], 6))  # 0.25
print(ball.position)  # (0.0, 0.0, 1.0): authored state stays the rest pose
report = anim.curve_report(ball, "transform.position.z")
print(len(report["discontinuities"]), len(report["overshoot"]))  # 0 0
print(round(report["velocity"]["max"], 2))  # 2.25 (m/s, the velocity's peak)

# A rig, and an editable generated walk whose feet are measured, not assumed.
hero = scene.add_object("Hero", position=(2, 0, 0))
anim.add_rig(hero, [
    {"name": "Hips", "position": [0, 0, 0.75]},
    {"name": "Body", "parent": "Hips", "tail": [0, 0, 0.75]},
    {"name": "Head", "parent": "Body", "position": [0, 0, 0.75], "tail": [0, 0, 0.5]},
    {"name": "Arm.L", "parent": "Body", "position": [0.375, 0, 0.625], "tail": [0, 0, -0.75]},
    {"name": "Arm.R", "parent": "Body", "position": [-0.375, 0, 0.625], "tail": [0, 0, -0.75]},
    {"name": "Leg.L", "parent": "Hips", "position": [0.125, 0, 0], "tail": [0, 0, -0.75]},
    {"name": "Leg.R", "parent": "Hips", "position": [-0.125, 0, 0], "tail": [0, 0, -0.75]},
])
walk = anim.generate(hero, "walk")  # a clip of ordinary keys, played as the action
feet = anim.foot_contacts(hero, ["Leg.L", "Leg.R"])
print(walk["clip"], max(f["sliding_max"] for f in feet["feet"].values()) < 0.005)  # Walk True

# Render any time: the evaluated pose, without changing the scene.
image = scene.render(time=0.4, width=320, height=180, backend="cpu", frame=True)
print(image.image.shape)  # (180, 320, 3)
```

## The model: authored and evaluated

**Authored** state is what the scene saves and the history records: object transforms
and fields are the *rest* values, and the animation document holds clips, bindings,
rigs, skins, morph targets, constraints, paths and the shot sequence as items with
immutable definitions (one history aspect per item; edits replace definitions).

**Evaluated** state is what the animation shows at a time. `anim.evaluate(t)` is a pure
function of the authored scene and `t`: it returns an immutable `AnimationPose` (local
transforms, visibility, camera/light/appearance/material fields, joint poses, morph
weights, constraint influences, diagnostics) and changes nothing -- no history step, no
revision, nothing saved, nothing kept between frames. Seeking anywhere costs the same as
playing forward, and evaluating `t` gives the same bits whatever was evaluated before.
`anim.overlay(t)` adds the rigging pass (rigs, constraints, attachments, skins) as the
render overlay, and `scene.render(time=t)`, picks, spatial queries, observations and
exports at a time use it. Rendering at a time is bit for bit rendering the scene after
`apply_pose(t)`, which is the one explicit operation that writes evaluated values into
authored state (one undo step).

`anim.time` is runtime state (not saved, not recorded, no revision). Caches of evaluated
state are keyed by (scene revision, time); caches of authored state by the revision --
and while an operation or transaction is open (its changes not yet counted in the
revision) they are computed afresh and not kept, so evaluation inside a transaction sees
the edits made in it.

## Time and the timeline

Seconds are canonical: keys, strips, markers and events store times in seconds
(float64), and frames are a view through a frame rate, so changing the rate never
changes a motion. **Frame rates are exact rationals**: `24`, `25`, `30`, `"30000/1001"`
(29.97), `"24000/1001"` (23.976); the decimal spellings `23.976`, `29.97`, `59.94`, ...
mean their `n * 1000/1001` rationals. The time of frame `n` is `n * den / num` rounded
once to float64, and the frame at a time is decided on the float's exact rational value,
so `frame_at(frame_time(n)) == n` for every frame at every rate, with no drift over hours
of timeline. Timecode is `HH:MM:SS:FF`, non-drop-frame (frames counted at the nominal
rate `ceil(fps)`).

The timeline (`set_timeline(fps=, start=, end=)`) has **markers** (`add_marker(name,
time, data=)`) and named **sections** (`set_sections`). Loops and ping-pong are
properties of motion, not of a player: curves extrapolate `cycle` (loop), `cycle_offset`
(loop that advances) or `mirror` (ping-pong); strips `repeat` (any count, fractional
counts end mid-cycle), play `reverse`, `hold` before or after. **Retiming**: `edit_keys`
`scale_time` (velocities scale by `1/factor` so the motion is the same, faster) and
`retime` (a piecewise-linear time map; `velocity: "scale"` or `"preserve"`), strip
`speed` and a strip `time_curve` (any monotone or non-monotone remapping), and the time
modifiers `time_offset`, `time_scale`, `stepped`.

## Animatable properties

A channel is named by a **path**, parsed by a strict grammar and resolved against a typed
registry; anything else is refused (`UNKNOWN_PROPERTY`) -- there is no attribute lookup
by name, ever (a path such as `__class__.__init__` is an unknown property, not code).

| Path (object) | Kind | Unit | Valid values |
| --- | --- | --- | --- |
| `transform.position.x\|y\|z` | scalar | scene units (m) | finite |
| `transform.rotation` | rotation track | quaternion | unit quaternions |
| `transform.rotation_euler.x\|y\|z` | scalar | degrees (XYZ) | finite |
| `transform.scale.x\|y\|z` | scalar | factor | finite (0 hides) |
| `visible` | bool | on above 0.5 | -- |
| `camera.fov`, `camera.ortho_scale`, `camera.near`, `camera.far` | scalar | degrees, units | `0.01..179.9`, `> 0`, `near < far` |
| `light.intensity`, `light.color.r\|g\|b`, `light.range`, `light.radius`, `light.angle` | scalar | light units, sRGB, units, degrees | `>= 0`, `0..1` |
| `appearance.color.r\|g\|b`, `appearance.specular`, `appearance.shininess` | scalar | sRGB, 0..1, exponent | `0..1`, `1..10000` |
| `joints["NAME"].position.x\|y\|z`, `.rotation`, `.rotation_euler.x\|y\|z`, `.scale.x\|y\|z` | joint channels | armature space | as above |
| `morph["NAME"]` | scalar | weight | finite |
| `constraints["NAME"].influence`, `constraints["NAME"].offset` | scalar | 0..1, path fraction | `0..1`, finite |

Material clips (bound to `{"material": NAME}`) animate `base_color.r|g|b`, `opacity`,
`metallic`, `roughness`, `specular`, `shininess`, `emissive.r|g|b`,
`emissive_strength`, `normal_strength`, `occlusion_strength`. Grouped paths
(`transform.position`, `transform.rotation_euler`, `transform.scale`, `light.color`,
`appearance.color`, `base_color`, `emissive` and their joint forms) name all their
components at once in authoring calls. An evaluated value outside its property's range
is clamped and reported (`VALUE_CLAMPED`), never written into the scene.

## F-curves: the graph editor as mathematics

An `FCurve` is an immutable structure of arrays: key times and values, per-key
interpolation, easing mode and parameters, handle types and handle offsets. Editing
returns a new curve.

### Keys and interpolation

A key is `(time, value)` or `{time, value, interpolation, ease, left, right, handles,
weighted, back, amplitude, period}`. Times are non-decreasing; two keys may share a time
(a **jump**: right-continuous, its left limit the first key's value), never three.
Segment `k` uses key `k`'s interpolation:

* `constant` -- `v_k` (a step at the next key);
* `linear`;
* `bezier` -- the cubic Bezier through `(t_k, v_k)`, the right handle, the next key's left
  handle and `(t_{k+1}, v_{k+1})`;
* an **easing** -- Robert Penner's set (BSD): `sine`, `quad`, `cubic`, `quart`, `quint`,
  `expo`, `circ`, `back` (overshoot `back`, default 1.70158: about 10 %), `elastic`
  (`amplitude >= 1`, `period` in segment units) and `bounce`, each `in`, `out` or `in_out`
  (`E(u) = f(u)`, `1 - f(1 - u)`, or the two halves). `expo` is normalized
  (`(2^(10u) - 1) / 1023`) and `elastic` corrected at `u = 0`, so every eased segment
  passes through both keys exactly.

### Handles

Handles are offsets `(dt, dv)` from their key (left `dt <= 0`, right `dt >= 0`), so they
move with it. Types per side:

* `free` -- as given; `aligned` -- the slope of the key's other handle;
* `vector` -- toward the neighbouring key (a third of the way);
* `auto` -- the slope of the parabola through the key and its neighbours (non-uniform
  Catmull-Rom): C1 motion;
* `auto_clamped` (the default) -- `auto`, flat at local extrema and the ends, and limited
  by the Fritsch-Carlson condition (SIAM J. Numer. Anal. 17(2), 1980), which guarantees
  each segment stays between its two keys' values: **no overshoot**.

A key is **unweighted** by default: its handles' time extents are a third of the adjacent
intervals, which makes the segment a cubic Hermite curve in time (no inversion). A
**weighted** key keeps its handles' time extents (longer handles hold the motion longer).
As in Blender's `BKE_fcurve_correct_bezpart`, a handle reaching past the segment's other
end is scaled back along its own direction at evaluation (the stored handle is unchanged
and reports say so), which keeps time monotone in the segment, so the Bezier parameter of
a time is unique.

### Evaluation and derivatives

The Bezier parameter of a time is found by Newton's method safeguarded by bisection on
`[0, 1]` (deterministic, only `+ - * /`), the value by Horner's rule on precomputed
power-basis coefficients. `evaluate(times, order=3)` returns the value and the first three
time derivatives -- velocity, acceleration, jerk -- exactly: for Bezier segments
`v' = y'/x'`, `v'' = (y''x' - y'x'')/x'^3` and the next derivative; for easings their
closed forms (`E'`, `E''`, `E'''`); through extrapolation and modifiers by the chain and
product rules. A vertical handle (`dt = 0`, `dv != 0`) gives an infinite derivative at its
key, which reports flag (`circ`'s steep end is a deliberate one).

### Extrapolation and cycle-aware handles

Before the first and after the last key: `constant`, `linear` (the end handle's slope for
Bezier ends, else the end segment's), `cycle`, `cycle_offset` (each cycle shifted by the
value change across the keys: walk cycles that advance), `mirror`. Automatic handles are
**cycle-aware**: an end key whose side repeats takes its missing neighbour from the
repetition, so a loop whose end keys agree is C1 across the wrap.

### Modifiers

A curve's modifiers form a chain (`[m1, m2]` evaluates `m2(m1(curve))`), at most 16, each
with an `influence` (0..1), an optional `range` with `blend_in`/`blend_out` ramps, and
`mute`. Value modifiers: `offset`, `scale` (`factor`, `pivot`), `limits` (`min`, `max`;
derivatives 0 where clamped), `noise` (C2 gradient noise: `strength`, `scale`, `offset`,
`octaves`, `roughness`, `seed`), `wave` (`amplitude`, `period`, `phase`, `decay`, `start`),
`envelope` (`reference`, `[time, gain, offset]` points), `smooth` (Gaussian, `sigma`: 15
fixed nodes within 3 sigma, seekable). Time modifiers: `stepped` (`step`, `offset`:
on-twos, sample and hold), `time_offset`, `time_scale`. A time modifier applies to
everything before it in the chain. Derivatives stay exact for each definition.

### Key operations

`edit_keys(target, path, op, keys=..., ...)` on a scalar channel (Python, scripts, MCP);
keys are selected by indices, a time range `{"from": t0, "to": t1}` or `"all"`:
`insert`, `delete`, `set` (value, time, interpolation, ease and easing parameters,
weighted), `handle` (one side: type, `dt`/`dv` or `slope`; an aligned pair turns
together), `handle_type`, `move` (`dt`, `dv`), `scale_time` / `scale_values` (`factor`,
`pivot`), `reverse` (handles swap sides, `in`/`out` easings swap), `quantize` (snap to a
frame rate), `distribute`, `retime`, `interpolation` / `easing`, `extrapolation`,
`modifiers`. `set_modifiers` replaces a curve's modifier chain in one undo step.

### Measuring motion

* `curve_report(target, path)` -- key count, interpolation and handle types, corrected
  handles, **discontinuities** at keys from the two segments' exact one-sided limits
  (value C0, velocity C1, acceleration C2; steps and jump pairs reported as intended),
  **extrema** from sign changes of the analytic velocity refined by bisection, per-segment
  **overshoot** beyond the range of the segment's two keys, the largest velocity,
  acceleration and jerk, and vertical tangents.
* `trajectory(target, start=, end=, joint=)` -- an object's or a joint's world motion
  through the evaluated scene (after layers, rigs and constraints), sampled: path length,
  displacement, speed, acceleration, jerk, angular speed, and smoothness as the log
  dimensionless jerk (Balasubramanian et al. 2012; closer to 0 is smoother).
* `diagnose()` -- findings over the whole animation `{code, severity, location, time,
  magnitude, message, basis}` with `basis` `"exact"` (closed forms) or `"sampled"`:
  velocity and acceleration discontinuities, overshoot, corrected handles, IK targets out
  of reach, clamped values. A finding is a measurement; whether a snap is intended is the
  animator's (or the agent's) decision.

### Fitting, simplification, resampling, baking

* `simplify(target, path, tolerance)` reduces keys to a bounded error: from the end keys,
  it repeatedly inserts the original key where the reduced curve deviates most (measured
  at every original key plus `density` points per segment) until the deviation is at most
  `tolerance`. Each kept key keeps the original's value and exact slope (Hermite), so the
  error falls with the fourth power of the spacing on smooth motion; steps stay steps.
* `resample(target, path, fps=)` keys every frame with the original's values and slopes.
* `aether3d.animation.fitting.fit_samples(times, values, tolerance)` turns dense samples
  (motion capture, simulation) into a curve within `tolerance` of every sample, returning
  the curve and a report (keys, largest error).
* `bake(targets, ...)` samples what the scene *shows* (layers, rigs, constraints, paths)
  into ordinary keys of a new clip, optionally simplified -- the way to give NLA, IK and
  constraints to formats that only carry keys.

Reported errors are measured on the curve that was built (`basis: "sampled"`).

### Graph images

`graph([(target, path), ...], format="svg"|"png")` or `graph(clip=...)` draws value
(keys as squares, effective Bezier handles as lines with circles), velocity and
acceleration -- the exact derivatives, sampled densely, vertical tangents clipped and
marked -- on a shared time axis. SVG is exact vectors; PNG is rasterized by Aether3D's own
anti-aliased stroke rasterizer and stroke font. CLI: `aether3d animation graph`.

## Rotations

`transform.rotation` and `joints["NAME"].rotation` channels are **rotation tracks**, not
four independent curves. Each key is a unit quaternion `(x, y, z, w)` whose **sign is
kept** (`q` and `-q` are the same orientation, not the same path). Segment interpolation:
`constant`, `slerp` (constant angular speed; normalized linear interpolation below 1e-6
rad), `squad` (Shoemake's spherical cubic: continuous angular velocity at uniformly spaced
keys) or `cubic` (glTF CUBICSPLINE: component Hermite, normalized). `path: "shortest"`
(default) goes through `-q_{k+1}` when the dot product is negative; `"stored"` keeps a
deliberate long arc (up to 360 degrees per segment). `timing` maps time through the arc:
`linear`, an easing, or a CSS-style `{"bezier": [x1, y1, x2, y2]}` (values outside 0..1
overshoot along the arc). Angular velocity and acceleration are exact for `constant` and
`slerp`, fourth-order central differences for `squad` and `cubic` (labelled). Extrapolation
`cycle_offset` composes each cycle with the rotation across the keys (a turning walk keeps
turning).

Euler channels (`rotation_euler.x|y|z`, degrees, XYZ about fixed axes) are the
graph-editable form; `rotation.euler_filter` unwraps by 360 degrees and picks between the
two equivalent XYZ solutions (Blender's *Discontinuity (Euler) Filter*). Aim and look-at
are constraints (`look_at` with `axis`, `up`, `world_up`, `target_offset`).

## Clips, bindings, layers and strips

A **clip** holds channels by path and **events** (named times with JSON data: contacts,
footsteps, cues); its target kind is `object` or `material`. A **binding** animates one
target with an **action** (a clip in scene time: what keyframing writes into) and
**layers** above it, bottom to top:

* `replace` layers average their active strips by weight (`sum w x / sum w`) and blend that
  over what is below with influence `min(sum w, 1) * layer weight * mask` -- two strips
  crossfading with weights `1 - s` and `s` give `(1 - s) A + s B` exactly, never the rest
  pose; rotations average by successive slerps;
* `additive` layers add `w (x - reference)` (the clip relative to its first frame);
  rotations compose `(reference^-1 x)^w`;
* a **mask** lists paths or joints with weights; a joint entry with `"descendants": true`
  covers the joint's subtree in its rig (an upper-body layer: `[{"joint": "Chest",
  "descendants": true}]`);
* a layer's weight may be an F-curve.

A **strip** places a clip: `start`, `clip_start`/`clip_end`, `speed`, `repeat`, `reverse`,
`blend_in`/`blend_out` (overlapping strips crossfade), `before`/`after` (`nothing` or
`hold`), `accumulate` (each repetition adds the clip's change: root motion that keeps
walking, turns that keep turning), a `weight_curve` and a `time_curve`. Clip time is
computed directly from scene time (`local = (t - start) * speed`, cycle `floor(local /
L)`), so seeking costs the same everywhere. Strips with neither a time nor a weight curve
are timed together in one vectorized pass (`nla.StripTable`), bit for bit the per-strip
mapping: 10,000 objects playing one shared clip through offset strips evaluate in 2.76 ms
per frame on the reference machine (163 ms when timed one by one).

## Rigs

### Skeletons and forward kinematics

`add_rig(armature, joints)` gives an *armature object* a skeleton: joints `{name, parent,
position, rotation | euler, scale, tail, limits}` stored parents first, with stable ids;
rest transforms are relative to the parent joint (the root's to the armature object), so
moving the armature (placement, root motion) moves the skeleton. `tail` gives a bone its
length and direction (default: toward its first child). `limits` are Euler XYZ rotation
bounds in degrees (`{"rotation": {"min": [...], "max": [...]}}`, `null` for no bound),
applied to the joints IK moves. Joint channels pose the rig; forward kinematics uses the
scene hierarchy's arithmetic (and its native kernel when installed). At most 4,096
joints per rig.

### Attachments

`attach(object, armature, joint)` moves a rigid part with its joint relative to the rest
pose (`W' = A' J' (A J)^-1 W`: armature and joint now, times their inverse at rest, times
the part's authored world matrix), whatever the scene hierarchy -- blocky characters
animate without remodelling.

### Skins

`skin(mesh_object, armature, weights=, influences=4, method=)` binds a mesh smoothly in
the current rest pose. Weights are given explicitly (`{"indices", "weights", "joints",
"inverse_bind"}`), or computed: `"nearest"` (each vertex follows its nearest bone fully)
or `"envelope"` (inverse-distance weights over the `influences` nearest bones -- an
automatic starting point, not production weight painting). Up to 8 influences per vertex,
weights normalized. Deformation: `v' = sum_k w_k S_k v` with `S_j = W_m'^-1 A' J'_j B_j`
(`B_j` the inverse bind matrix), normals by the cofactor of the blended linear part
(correct under non-uniform scale and mirroring) -- **linear blend skinning** (a native
Rust kernel, threaded by vertex range, bitwise equal to its NumPy twin, when
`aether3d-native` is installed) -- or **dual quaternion skinning** (`method:
"dual_quaternion"`, Kavan et al. 2008; NumPy), which keeps volume under twisting where
linear blending collapses (the "candy wrapper": in the skinned-character showcase a
160-degree wrist twist keeps 56 % of the wrist's radius with linear blending and 100 %
with dual quaternions).

### Morph targets

`set_morph_targets(mesh_object, [{name, positions, normals?, default?}])` adds blend
shapes (per-vertex offsets): `A' = A + sum_m w_m dA_m`, weights animated as
`morph["NAME"]`. Morphs apply before skinning.

### Constraints and inverse kinematics

Constraint lists belong to objects or to rigs (joint constraints) and apply in order,
each blended by its `influence` (animatable). Targets: `{"object": ...}`, `{"joint":
...}` (same rig, or `{"object": ARMATURE, "joint": ...}`), `{"point": [x, y, z]}`
(`"space": "world" | "armature"`).

* `ik` (joints): `chain` (root to end), `target`, `pole`, `solver` -- `two_bone` (the
  analytic law-of-cosines solution in the root-target-pole plane), `fabrik` (Aristidou
  and Lasenby 2011), `ccd` -- with `iterations`, `tolerance`, `stretch` (bones may lengthen
  up to a factor), `soft` (eases the last fraction of the reach so knees do not snap
  straight) and `end` (`keep` the end joint's local rotation, or take the `target`'s). The
  residual distance is reported (`rigging(t).residuals`); joint limits apply.
* `look_at` (`axis`, `up`, `world_up`, `target_offset`), `copy_location`,
  `copy_rotation`, `copy_scale`, `copy_transforms` (`axes`, `offset`), `child_of`
  (`inverse`), `limit_location`, `limit_rotation`, `limit_scale`, `limit_distance`,
  `follow_path`.

Units (objects with constraints, rigs) run in dependency order -- a unit after every unit
that moves one of its targets or a target's ancestor; a cycle is refused when made
(`CONSTRAINT_CYCLE`). A missing target makes its constraint do nothing and is reported.
**Order** (the documented contract): object channels, world matrices; units in dependency
order (a rig: joint channels, forward kinematics, joint constraints, attachments; an
object: its constraints); morph targets, then skins; the shot camera.

## Motion

### Generators

`generate(armature, motion, **parameters)` computes a motion and **writes keys**: the
result is an ordinary clip (its metadata records the generator and parameters) that is
edited like any other. Motions: `idle` (breathing, weight shift, glances), `walk`,
`run`, `jump` (anticipation, ballistic flight under `gravity`, landing absorption),
`land` (`style: "superhero"` -- a fall into one knee and a fist down, a pause, a look up,
a rise -- or `"soft"`), `wave` (the forearm raised and swung side to side), `head_turn`.
Parameters include `forward`, `travel`, `tolerance`, `bind`, `amplitude`, `period`,
`cadence`, `stride`, `duty`, `lift`, `gravity`, `height`, `distance`, `style`, `knee`,
`side`, `waves`, `rate`, `yaw`, `pitch`, `turn`, `hold`, `back`, `mapping`.

Roles (`hips`, `spine`, `chest`, `neck`, `head`, arms and legs with sides) are found by
joint name (Mixamo, Rigify, `Leg.L` styles; `mapping` overrides) and hierarchy; a blocky
character with one joint per limb walks a straight-legged compass gait. Legs follow
**foot trajectories** -- planted during stance with heel and toe rockers, lifted on an arc
in swing -- reached by analytic two-bone IK; the hips' height follows the stance legs'
reach; the armature travels by root motion at exactly the speed the planted feet move
back. Samples at 240 Hz are fitted to Euler curves (quaternion tracks near gimbal lock)
within `tolerance` degrees, each key with the motion's own slope. Generated motion is
plausible and measured, not motion capture.

### Mirroring, root motion, retargeting

* `mirror_clip(clip, armature)` swaps left and right joints by name pairs and reflects
  across the armature's YZ plane (positions `x -> -x`, quaternions `(x, -y, -z, w)`, Euler
  `(rx, -ry, -rz)`); symmetric rigs only.
* `extract_root_motion(clip, joint, mode=, yaw=)` moves a root joint's horizontal travel
  (and optionally its yaw) to the armature's own channels, or drops it (`"in_place"`).
* `retarget(source, target, clip, mapping=)` transfers rotations relative to each rig's
  rest pose (A-pose and T-pose rigs with the same intent transfer), scales the root's
  translation by the ratio of the rigs' hip heights, keeps the target's bone lengths, and
  reports unmapped joints and rest directions differing by more than 30 degrees.

### Foot contacts and planting

`foot_contacts(armature, feet)` samples the evaluated motion (120 Hz by default) at each
foot's ground points -- the toe (its bone's end) and the heel under the joint -- and finds
**stances**: runs of samples within `height` (0.03 m) of the ground and slower than the
body (the rig's root joint, or at least `speed`, 0.3 m/s); a swinging foot moves about
twice as fast as its body. A point is planted while it stays within `speed` of its
stance's steady (median) velocity -- zero for a planted foot, so this is the plain speed
test, and the skating velocity when a body outruns its steps, whose sliding is then
measured instead of missed. Each contact reports its **sliding** (how far a planted point
moves horizontally while planted; a heel-to-toe roll is not sliding) and the foot's
**penetration** below the ground. These defaults are the feedback check's, so Python,
scripts, MCP and the check measure alike.

`plant_feet(armature, legs)` corrects sliding (after Kovar, Schreiner and Gleicher 2002):
for each foot a target object is pinned during each contact in the pose the foot touched
down with -- its position and orientation at the contact's start, from one instant --
lifted if a heel or toe of that pose was below the ground, and keyed to follow the foot
between contacts; an IK constraint on the leg (`end: "target"`) has its influence keyed 1
during contacts with blends around them. Everything it makes is ordinary, editable data;
the report gives the sliding before and after.

## Cameras and cinematics

**Paths** (items of kind `path`): `catmull_rom` (centripetal: no cusps or
self-intersections within a segment), `bezier` (per-point handles), `linear`, `closed`
loops, and exact `circle`s (`center`, `radius`, `normal`, `start` in degrees). Positions
are found **by arc length** (16-point Gauss-Legendre segment lengths, safeguarded Newton
inversion), so a uniformly moving offset moves at constant speed. `follow_path(object,
path, start=, end=, offsets=, easing=, follow=, axis=, up=, loop=)` keys a `follow_path`
constraint's offset.

**Camera moves** write ordinary keys and constraints: `camera_move(camera, move, amount,
start, end)` with `dolly` (along the view), `truck` (sideways), `pedestal`/`crane`
(vertically), `pan` (about world Z), `tilt` (about the camera's X), `roll`, `zoom` (field
of view); `camera_orbit(camera, target, start, end, degrees=)` (an exact circle, a
follow-path constraint and a look-at); `follow_camera(camera, target, offset=, lag=,
aim=)` (a critically damped follow, fitted into keys, aimed by a look-at constraint).

**Shot sequences** (`set_shots([{camera, start, end, name, transition}])`): a shot starting
while another is on cuts to it, or blends with `{"type": "blend", "duration": d}`
(position linearly, orientation by slerp, field of view linearly) through a virtual
camera. Renders that name no camera use the sequence's camera at their time.

**Depth of field is not simulated.** No renderer in Aether3D models focus or aperture;
lenses animate their field of view only, and nothing claims a depth-of-field effect.
**Motion blur is real temporal sampling**: `render_frames(..., motion_blur={"shutter": s,
"samples": n})` averages `n` renders spread over the fraction `s` of the frame interval,
centred on the frame time, in linear light, before the renderer's display encoding -- at
the cost of one render per sample. The shutter stays inside the rendered range.

## Rendering frames

`render_frames(scene, directory, start=, end=, fps=, step=, frames=, camera=, backend=,
motion_blur=, progress=, should_stop=, ...)` renders the times of a range of frames (the
timeline's rate by default) with Aether3D's own renderers, one frame at a time (memory
bounded), and writes each PNG atomically (`frame_{frame:05d}.png`); `should_stop`
cancels between frames (the frames written so far are listed), `progress` is called after
each. `contact_sheet(images, columns=, labels=)` lays frames out with labels in
Aether3D's stroke font; `video_command`/`assemble_video` turn a numbered PNG sequence into
an H.264 MP4 with FFmpeg when it is installed -- an external encoder, never a renderer,
and never run from the MCP server. No other engine renders anything here.

## Evaluation pipeline and performance

Evaluation compiles the animation once per revision into a program: every action and
strip becomes a *contribution*, every channel a query into one `CurveBatch` (scalars) or
`RotationBatch` (quaternions) aimed at a property *slot*. A frame is: the strips' clip
times (vectorized), one batched curve evaluation and one batched rotation evaluation
(each query bit for bit what the curve gives alone), then blending layer by layer from
the authored rest values (cached per revision). The rigging pass then runs rigs,
constraints, attachments and skins; the render overlay carries the posed local matrices,
fields and deformed meshes to the renderers' extraction (the retained render world
updates only what changed).

Measured on the development machine (i9-12900H laptop, 20 logical CPUs, Python 3.14.8,
NumPy 2.5.4, `aether3d-native` 0.17.0, NVIDIA GeForce RTX 4060 Laptop GPU through wgpu
0.32 on Vulkan) with `benchmarks/bench_animation.py`, saved as
`benchmarks/results/stage17-animation.json` (medians after a warm-up, every group in its
own process; laptop timings drift by about 10 % between runs):

| Workload | Time |
| --- | --- |
| A 10,000-key Bezier curve at 100,000 times: values / with three derivatives / weighted keys | 6.0 / 8.7 / 10.6 ms |
| 1,000 curves at one time each (one batch) | 0.15 ms |
| 10,000 objects, each with its own clip: evaluation / render overlay / posed world matrices per frame | 9.9 / 14.2 / 24.0 ms |
| 10,000 objects playing one clip through offset strips | 2.8 / 7.2 / 15.2 ms |
| the same plus 48 rigs with 6 attachments each on looping paths: overlay / posed worlds | 31.6 / 39.9 ms |
| A 200-joint rig: rigging pass per frame | 0.58 ms |
| Two-bone IK solve / FABRIK on 12 joints / a rigging pass with IK | 12.4 us / 0.93 ms / 0.50 ms |
| 4 layers of 4 strips on one target | 0.79 ms |
| Linear blend skinning of 1,000,000 vertices (4 influences, 64 joints): NumPy / native, 1 thread / native, 20 threads | 171 / 21.2 / 6.9 ms (bitwise equal) |
| Dual quaternion skinning of 1,000,000 vertices (NumPy) | 678 ms |
| Generating a walk / a superhero landing / a blocky walk (240 Hz solve, fit) | 230 / 746 / 103 ms |
| Simplifying 10,000 keys / fitting 20,000 samples | 77 / 16.4 ms |
| 48 frames at 320x180 to PNG, per frame: CPU / GPU; a 4-sample motion-blurred CPU frame | 44.1 / 3.7 ms; 131 ms |
| glTF: export with clips / baked / import | 42 / 56 / 10 ms |

Measured improvements of this stage, before and after, on the same machine and harness:
building 10,000 animated objects 40.3 s -> 5.9 s, evaluating them 34.5 -> 9.9 ms per
frame (incremental indexes, vectorized pose assembly), keyframe reduction of 10,000 keys
3.9 s -> 77 ms and fitting 20,000 samples 1.57 s -> 16.4 ms (a heap of candidate
insertions on direct Hermite evaluation), generating a walk 740 -> 230 ms, 10,000
instanced strips 163 -> 2.8 ms (vectorized strip timing), and the crowd's overlay
110 -> 31.6 ms (the scene order, authored worlds and circle bases cached per revision).
The large-scene showcase -- 10,341 objects, 123,470 triangles, 48 walkers -- renders
960x540 frames with shadows on the GPU in a median 143 ms each (evaluation 5.4 ms,
overlay with rigging 34.1 ms, world matrices 8.3 ms; the rest is extraction, the render
and the PNG); without shadows the same frames take 87-127 ms on the GPU and
1,063-1,432 ms on the CPU rasterizer.

## History and files

Every authoring call is one undo step; scripts and plans are one transaction (all or
nothing: a failing operation leaves the scene exactly as it was, including the playhead).
Evaluation never records anything. Projects use file format **9** (`scene.animation`
with the timeline settings and the items); files of format 8 and older are upgraded on
load, and scenes without animation load and render unchanged.

## glTF interchange

**Export** (`scene.export(path, animation=, fps=, position_tolerance=1e-4,
rotation_tolerance=0.01, weight_tolerance=1e-4)`): rig joints become nodes under the
armature's node; attached parts become children of their joints; a skinned mesh rides a
root node without a transform (glTF ignores a skinned mesh node's own transform); inverse
bind matrices are exact (`C M C^T` axis conversion); `JOINTS_n`/`WEIGHTS_n` carry up to 8
influences; morph targets carry `POSITION`/`NORMAL` offsets, default weights and
`extras.targetNames`. Animations: `"clips"` (every clip with a target, each its own glTF
animation), `"actions"`, `"bake"` (what the scene shows -- layers, constraints, IK, paths
-- on the timeline's frames as `LINEAR` keys reduced within the tolerances), `"auto"`
(bake when layers, constraints, IK or paths shape the motion, else clips), `"none"`.
Channels map **exactly** where glTF can say the same thing: `STEP` for constant keys,
`LINEAR` for linear keys and shortest-path slerps (a deliberately long arc is split at
its middle), `CUBICSPLINE` with one-sided slopes as tangents for unweighted Bezier keys and
imported glTF cubics. Everything else (easings, weighted handles, Euler rotations, squad,
timing curves, modifiers) is **fitted** within the tolerances, with the largest error
reported per animation; an instantaneous jump becomes a 1e-4 s transition; values are
float32. Visibility, camera, light, material and constraint channels have no glTF
equivalent (no `KHR_animation_pointer`): lossy diagnostics.

**Import**: skin joints (and the nodes between them) become a rig on their common parent
node or a new `Armature`; nodes below joints become attached parts; skins bind with the
file's inverse bind matrices and up to 8 influences; morph targets merge across
primitives; every animation becomes clips (`STEP` -> constant, `LINEAR` -> linear or
slerp, `CUBICSPLINE` -> Bezier keys with the tangents as slopes, or `cubic` rotation
tracks): the same curves. The first animation plays as actions. Inputs are checked
(strictly increasing times, output counts, duplicate targets, limits). Exports validate
with the Khronos glTF Validator with no errors and no warnings (tested in CI).

## Animation scripts

An `aether3d.animation-script` (version 1) is plain JSON: ordered operations, each mapped
to one whitelisted method with checked arguments; no code is evaluated and no file is
touched.

```json
{"format": "aether3d.animation-script", "version": 1, "label": "Bounce",
 "limits": {"max_seconds": 60, "max_keys": 500000},
 "operations": [
   {"op": "timeline", "fps": 24, "start": 0, "end": 2},
   {"op": "set_channel", "target": "Ball", "path": "transform.position.z",
    "keys": [[0, 1], [0.5, 0], [1, 1]], "extrapolation": "cycle"},
   {"op": "report", "target": "Ball", "path": "transform.position.z"}
 ]}
```

Operations: `timeline`, `set_time`, `markers`, `add_marker`, `sections`, `add_clip`,
`remove_clip`, `rename_clip`, `clip_range`, `clip_events`, `keyframe`, `set_channel`,
`edit_keys`, `set_action`, `set_binding`, `add_layer`, `update_layer`, `remove_layer`,
`add_strip`, `update_strip`, `remove_strip`, `add_rig`, `set_rig`, `attach`, `detach`,
`skin`, `unbind`, `add_constraint`, `set_constraints`, `remove_constraint`,
`morph_targets`, `add_path`, `remove_path`, `follow_path`, `shots`, `add_shot`,
`camera_move`, `camera_orbit`, `follow_camera`, `generate`, `set_modifiers`, `simplify`,
`resample`, `bake`, `mirror_clip`, `extract_root_motion`, `plant_feet`, `retarget`,
`apply_pose`; read-only: `report`, `trajectory`, `diagnose`, `contacts`, `sample`.
Objects are named or `{"object": ID}`, materials `{"material": NAME}`. A script runs as
one transaction within `limits` (defaults: 300 s, 5,000,000 keys; cooperative
cancellation); a failure names the argument that failed (`operations[1].keyframe.value`)
and changes nothing. Schema: `aether3d animation schema` (or the MCP resource
`aether3d://schema/animation-script`); guide for agents: `aether3d://animation/guide`.

## Command line

```text
aether3d animation run PROJECT script.json [--dry-run] [--output OUT] [--history] [--json]
aether3d animation info PROJECT|GLB [--json]
aether3d animation report PROJECT --kind curve|diagnose|trajectory|contacts ...
aether3d animation graph PROJECT out.svg|out.png --target T --path P [--clip C]
aether3d animation render PROJECT FRAMES_DIR [--start --end --fps --step --width --height
    --camera --backend --motion-blur SHUTTER:SAMPLES --contact-sheet PNG --video MP4]
aether3d animation schema
aether3d export PROJECT out.glb [--animation auto|clips|actions|bake|none] [--fps]
    [--position-tolerance] [--rotation-tolerance]
```

## For agents (MCP) and the feedback loop

Over MCP, author with `run_script` (an animation script), measure with
`inspect_animation` (`what`: `summary`, `curve`, `diagnose`, `trajectory`, `contacts`,
`graph`), render with `render_preview`/`render_image` at a `time`, export with
`export_scene` (`animation` mode). The scripted MCP client of the showcases reads
structured results only; it never claims to have looked at an image. An agent that can
see images may look at previews and graphs; one that cannot must not claim it did.

The feedback loop has animation checks -- `foot_contacts` (`feet`, `max_sliding` 0.005 m,
`max_penetration` 0.002 m, `height`, `speed`) and `animation_continuity` -- with findings
`FOOT_SLIDING`, `FOOT_PENETRATION` and `MOTION_DISCONTINUITY`, and the corrections
`plant_feet` (safety class `verify`: kept only if the finding improves and no passing
check regresses, else undone) and `smooth_keys` (`approval`). Animation corrections are
opt-in: the policy switch `allow_animation` (default `false`).

## Security and limits

Every boundary treats its input as untrusted: scripts, MCP arguments, imported files and
property paths are validated before anything changes (finite numbers, time ranges, key
order and counts, handle shapes, quaternion norms, joint hierarchies (acyclic, parents
first), influence counts and weights, modifier counts, constraint graphs (acyclic), path
references, sizes and time budgets); failures raise `AnimationError` (or
`AnimationScriptError`) with a stable code and location. Limits: 1,000,000 keys per
curve, 100,000 channels per clip, 64 layers and 10,000 strips per layer, 4,096 joints per
rig, 8 influences per vertex, 10,000,000 skinned vertices, 256 constraints, IK chains of
64 joints, 16 modifiers per curve, 100,000 path points, 10,000 shots, 100,000 frames per
render call, 64 motion-blur samples. Frame rendering writes only where asked, atomically;
CLI and MCP file access follow the existing workspace confinement and symbolic-link
rules; video assembly never runs from MCP.

## Language per subsystem

| Subsystem | Language | Why |
| --- | --- | --- |
| API, items, scripts, history, persistence, validation, glTF | Python | One coherent API; orchestration, no hot loop |
| F-curve storage and evaluation (all channels at once) | NumPy (structure of arrays, batched) | Vectorized over every channel of every strip; the measured frame cost did not justify a native kernel |
| NLA strip timing | NumPy (vectorized `StripTable`) | Replaced a per-strip loop (163 ms -> 2.76 ms for 10,000 strips) |
| Easing, quaternions, blending | NumPy | Transcendentals vectorize well; libm calls would break bitwise native twins |
| Forward kinematics | the scene hierarchy kernels (Rust, NumPy twin) | Already native and parity-tested |
| Linear blend skinning | Rust kernel (`skin_linear`, ABI 4) + NumPy twin, bitwise | The hottest loop of a deforming character: 1M vertices in 6.5 ms on 20 threads vs 167 ms in NumPy |
| Dual quaternion skinning, morph targets | NumPy | Rarer; memory-bound or fast enough |
| IK, constraints, generators' per-sample solver | Python + NumPy | Small chains; plain-float quaternion helpers where NumPy's per-call overhead dominated |
| Frame rendering | Aether3D's existing CPU and GPU renderers | No new shader code; posed transforms are instance matrices, deformed meshes new geometry; no GPU skinning (the path tracers need deformed geometry on the CPU anyway) |
| Graphs | Python (SVG text; PNG through the own rasterizer) | No dependency |
| Video | external FFmpeg, optional | An encoder, not a renderer |

## Showcases

Six scripts in `examples/`, rendered by Aether3D's own renderers (GPU: the RTX 4060
above); their outputs are in `examples/output/animation/` (frame sequences are
regenerated by the scripts, not committed).

1. **A Minecraft-style character** ([`animation_minecraft.py`](../examples/animation_minecraft.py)):
   Nova, an original 64x64 pixel skin, with generated idle, walk, run, jump and superhero
   landing (61, 109, 150, 328 and 398 keys, within 0.05 degrees of the solved motion),
   follow and three-quarter cameras, and a three-camera landing cinematic (cuts, a blend,
   8-sample motion blur). Measured feet: sliding 2.1 mm (walk), 1.3 mm (run), 0.8 mm
   (jump), and 8.5 mm on the landing's step-off (a knee-less leg drags its toe: a known
   limitation). [Views](../examples/output/animation/minecraft/nova_views.png),
   [walk](../examples/output/animation/minecraft/walk_sheet.png),
   [run](../examples/output/animation/minecraft/run_sheet.png),
   [jump](../examples/output/animation/minecraft/jump_sheet.png),
   [landing cinematic](../examples/output/animation/minecraft/cinematic_sheet.png),
   [video](../examples/output/animation/minecraft/nova.mp4),
   [GLB](../examples/output/animation/minecraft/nova.glb) (validator: 0 errors, 0
   warnings).
2. **Curve repair** ([`animation_curve_repair.py`](../examples/animation_curve_repair.py)):
   a crane's crate overshoots below a shelf and swings with jitter; one repair script
   (auto-clamped handles, a Bezier key, smoothing, resampling, simplification) gives
   identical curves through Python, a JSON script and MCP. Overshoot 2 -> 0 (height) and
   1 -> 0 (travel), the crate no longer dips into the shelf, the linear key's velocity
   corner is gone; acceleration still steps at keys (cubic Hermite curves are C1) and the
   height stops more firmly (peak jerk 2.8 -> 23.5 m/s^3); the swing goes from 193 keys
   to 36 and its peak velocity from 172 to 28 degrees/s.
   [Graphs before and after](../examples/output/animation/curve_repair/graphs_side_by_side.png).
3. **A skinned character** ([`animation_skinned_character.py`](../examples/animation_skinned_character.py)):
   a smooth mannequin (one mesh, 13,152 vertices, weight profiles along each limb) walks
   (sliding 3.9 mm), steps onto a 0.24 m box with an FK-to-IK ramp (ankle exactly 0.32 m
   up, residual 0), crossfades a wave into its mirror image on an upper-body layer
   (the raised hands at 1.631 and 1.6295 m) under an additive look-around, and twists its wrist 160
   degrees: linear blending keeps 56 % of the wrist's radius (24 % at worst), dual
   quaternions 100 %. [Walk](../examples/output/animation/skinned/walk_sheet.png),
   [step](../examples/output/animation/skinned/step_sheet.png),
   [layers](../examples/output/animation/skinned/layers_sheet.png),
   [LBS vs DQS](../examples/output/animation/skinned/twist_lbs_vs_dqs.png).
4. **Cinematic cameras** ([`animation_cinematics.py`](../examples/animation_cinematics.py)):
   a crane on a path, an orbit (radius 5.5 m at every sample), a dolly zoom whose field of
   view is solved per instant and fitted into 9 keys (Nova's screen height stays 0.7015
   of the frame), a damped follow of a run along a path at the cycle's speed, cuts and
   blends, timecodes; 289 frames at 640x360 in 6.0 s on the GPU.
   [Sequence](../examples/output/animation/cinematics/sequence_sheet.png),
   [motion blur](../examples/output/animation/cinematics/motion_blur.png),
   [video](../examples/output/animation/cinematics/cinematics.mp4).
5. **A large animated scene** ([`animation_large_scene.py`](../examples/animation_large_scene.py)):
   10,000 blocks playing one stretch clip through offset strips (a stadium wave) and 48
   walkers on circles, with the timings above.
   [Sequence](../examples/output/animation/large/sequence_sheet.png),
   [the pool](../examples/output/animation/large/hero_pool.png),
   [video](../examples/output/animation/large/large.mp4),
   [report](../examples/output/animation/large/report.json).
6. **An MCP agent workflow** ([`animation_mcp_agent.py`](../examples/animation_mcp_agent.py)):
   a scripted client drives a real server: builds and rigs a robot, generates a walk,
   checkpoints, makes the naive "30 % faster" edit (feet skate 205 mm), finds
   `FOOT_SLIDING`, applies `plant_feet` (kept: 205.6 -> 11.7 mm, still over the 5 mm
   check), returns to the checkpoint and retimes the whole cycle instead (accepted:
   3.6-3.9 mm, 30 % more distance), and is refused three times (an attribute path, a
   wrong-typed key after a valid operation, a save outside the workspace) without any
   change. The client reads structured results only; the previews it receives are saved,
   never analysed. [Transcript](../examples/output/animation/mcp_agent/transcript.md).

## What is not supported

* No graph editor window (by design); graph images are pictures of the curves.
* No depth of field; motion blur averages renders (no velocity-buffer blur).
* No GPU skinning; dual quaternion skinning is NumPy only; FABRIK and CCD are Python
  loops.
* Squad is C1 at keys only for uniformly spaced keys.
* Timecode is non-drop-frame only.
* glTF: no `KHR_animation_pointer` (camera, light, material and visibility channels are
  lossy), jumps become 1e-4 s transitions, values are float32.
* Generators make plausible motion on common rigs, not motion capture; envelope weights
  are a starting point, not weight painting.
* Joint limits are Euler bounds applied to the joints IK moves (no swing-twist cones).
* Retargeting needs similar rest-pose intent (directions within 30 degrees are reported,
  not fixed).
