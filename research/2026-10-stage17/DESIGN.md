# Stage 17 design record: animation, keyframes, rigging, motion curves and cinematics

Written on 17 October 2026 against `main` at d2b711c (v0.15.0, Stages 1-15), before any
Stage 17 implementation code. Stage 17 is deliberately built **before** Stage 16
(procedural worlds), which is postponed; nothing here implements or presumes Stage 16.
This file records what was inspected, the research behind each decision, the
implementation language chosen for every subsystem, and the test and benchmark plans.
The user guide ([docs/ANIMATION.md](../../docs/ANIMATION.md)) describes what was finally
built; where they differ, this file is the plan as decided and the guide is the truth
about the code. Section 10, written after the implementation, records the outcome.

## 1. Starting point (inspected)

| Area | Where | What exists | Consequence for Stage 17 |
| --- | --- | --- | --- |
| Objects and transforms | `scene.py`, `transform.py` | `SceneObject` with a float64 TRS `Transform` (tuples), every setter one recorded history operation; world matrices computed on demand (`parent.world @ local`) | Writing animated values through the setters would record one undo step per object per frame and change the saved scene: animation must not write authored state while it plays |
| Rotations | `math3d.py` | unit quaternions `(x, y, z, w)`, canonicalized to `w >= 0` on assignment; Euler XYZ in degrees, extrinsic (`R = Rz Ry Rx`) | Keys must keep their own quaternion signs (multi-turn and deliberate long rotations); canonicalization applies only when a pose is applied to authored state |
| Identity | `scene.py` | UUID4 object ids, unique changeable names | Animation binds to ids, never names |
| History | `history.py`, `_history_file.py` | aspects captured by reference, atomic operations, change journal, history files | One aspect per animation item, like the CAD document; evaluating a time is not an operation |
| Format | `serialization.py` | project format 8, strict readers, migrations | Format 9 adds `scene.animation`, with a migration from 8 |
| Render input | `rendering/scene_data.py`, `rendering/retained.py` | every backend consumes a `RenderScene` from `extract_render_scene`; the retained render world updates it from the history journal (one moved object rebuilds one instance); `world_matrices(scene)` serves the renderer's camera, framing, spatial index and feedback | Two choke points take an evaluated pose: the full extraction and the retained world (which must handle posed rows incrementally, bit for bit equal to the full walk) |
| GPU | `rendering/gpu/` | meshes cached by array identity (weak), camera-relative float32 instance matrices, instanced draws | A deformed (skinned) mesh is a new `Mesh`: it uploads like any changed geometry; no shader change is needed for posed transforms |
| Path tracers | `rendering/pathtrace`, `gpu/pathtrace.py` | BLAS cache keyed by mesh arrays, two-level BVH | Rigid animation moves instances (top level only); deformation needs new bottom levels |
| Cameras | `rendering/camera.py` | `projection, fov, fov_axis, ortho_scale, near, far, aspect_ratio`; no aperture or focus distance | Field of view, ortho scale and clipping animate; depth of field does not exist in the renderers and is not claimed |
| Lights, appearance, materials | `rendering/light.py`, `settings.py`, `materials/material.py` | typed fields with validation; library materials resolved per render through a cache keyed by material id and library version | Animated light/appearance fields override copies; animated material parameters need render-time copies with their own cache keys |
| Spatial queries | `spatial/` | index keyed by `scene.revision`, rays through the render ray core | The index must key on the evaluated time as well, so queries agree with renders at that time |
| Feedback | `feedback/` | observations keyed by revision; checks and correction actions in registries with checkers, compilers, policies | Observations record their time; animation checks and actions follow the registry pattern |
| AI surfaces | `plans/`, `mcp/`, `cli.py`, `cad/script.py` | scene plans (op `cad` embeds CAD scripts), `run_script` formats, 25 MCP tools whose list is 39,992 of the 40,000 bytes its test allows | New MCP surface must be paid for by tightening existing descriptions, never by raising the budget |
| glTF | `gltf/` | own reader and writer, exact axis conversion; animations ignored on import (`ANIMATIONS_IGNORED`) | Import and export of animations, skins and morph weights is new; the axis conversion extends exactly to keys, tangents and inverse bind matrices |
| Native | `native/` (Rust, PyO3 abi3, ABI 3) | ray traversal, `local_matrices`/`world_matrices` hierarchy kernels, PNG filters, LDL^T, packing, triangulation; every kernel bitwise equal to a NumPy twin | Skeleton FK reuses the hierarchy kernels; new kernels follow the twin policy |
| Characters | `examples/minecraft_hero.py` | blocky hero built from separate part objects with pivots, flat colours | A rig must drive existing parts without remodelling; Example 1 adds an original pixel skin |
| Procedural / CAD | `procedural/`, `cad/` | content-addressed evaluations; CAD items regenerate on definition changes | Animating geometry parameters re-evaluates per distinct value (cached); CAD definitions are not animated |

## 2. Research (sources read for this stage)

**Curves.** Blender's F-curve model (keyframes with explicit left/right handle points,
handle types free/aligned/vector/auto/auto-clamped, per-key interpolation constant/linear/
Bezier plus Penner easings with in/out modes, extrapolation constant/linear, F-modifiers
with ordering, influence and frame ranges -- Blender manual, *Animation > F-Curves*), its
evaluation code (`source/blender/blenkernel/intern/fcurve.cc`, read for this stage):
`BKE_fcurve_correct_bezpart` scales each handle along its own direction so its time extent
does not exceed the segment ("handles cannot exceed the adjacent key-frames time
position"), and the segment parameter is found with Cardano's formula, taking the first
root in [0, 1]. A cubic Bezier whose two control abscissae stay within the segment's time
span is monotone in time (its derivative is a non-negative Bernstein quadratic,
`b >= -sqrt(a c)` holds when `a, c <= L`), so the correction guarantees a unique time
inversion. Maya's animation curves add *weighted* tangents (free handle length) versus
unweighted ones (length fixed at a third of the interval). Fritsch and Carlson, "Monotone
piecewise cubic interpolation", SIAM J. Numer. Anal. 17(2), 1980: slope limits that make a
cubic Hermite interpolant preserve monotonicity (no overshoot). Kochanek and Bartels,
"Interpolating splines with local tension, continuity and bias control", SIGGRAPH 1984.
Penner's easing equations (2001-2002, BSD licence), the de facto set (sine, quad, cubic,
quart, quint, expo, circ, back with overshoot, elastic with amplitude and period, bounce).
Schneider, "An algorithm for automatically fitting digitized curves", Graphics Gems, 1990
(least-squares cubic Bezier fitting with Newton reparameterization), for keyframe reduction.

**glTF 2.0** (Khronos specification, read for this stage): samplers STEP, LINEAR (spherical
linear interpolation for rotations, short path through the sign of the dot product) and
CUBICSPLINE (per key in-tangent, value, out-tangent, Hermite basis with tangents scaled by
the segment duration `t_d`; the first in-tangent and last out-tangent unused; rotations
normalized after interpolation); outputs clamped to the nearest end outside the input
range; input accessors with `min`/`max`; one target per node and path per animation;
skins with `joints` and `inverseBindMatrices` in the same order, the skinned mesh node's
own transform ignored; `JOINTS_n`/`WEIGHTS_n` with four influences per set, non-negative
weights, unused joints set to zero; morph targets `A' = A + sum_m w_m dA_m` with mesh
default weights. `KHR_animation_pointer` (animating material, camera and light properties)
exists with partial validator support; Aether3D's glTF export writes no cameras or lights,
so only node transforms, skins and morph weights are exported, and anything else is
reported as a lossy conversion.

**Rotations.** Shoemake, "Animating rotation with quaternion curves", SIGGRAPH 1985 (slerp,
squad from intermediate quaternions `s_i = q_i exp(-(log(q_i^-1 q_{i+1}) +
log(q_i^-1 q_{i-1}))/4)`); Kim, Kim and Shin, "A general construction scheme for unit
quaternion curves with simple high order derivatives", SIGGRAPH 1995 (cumulative-basis
quaternion splines with C2 continuity); Euler-angle filters (unwrapping by 360 degrees and
choosing between the two equivalent XYZ solutions) as in Blender's *Discontinuity (Euler)
Filter*.

**Skinning.** Linear blend skinning (Magnenat-Thalmann, Laperrière, Thalmann 1988; its
collapsing-elbow and candy-wrapper artefacts analysed by Lewis, Cordner and Fong, "Pose
space deformation", SIGGRAPH 2000). Kavan, Collins, Žára and O'Sullivan, "Geometric skinning
with approximate dual quaternion blending", ACM TOG 27(4), 2008: dual quaternion skinning
preserves volume for rigid joint transforms; scale and shear need a separate linear stage.
Normals transform with the inverse transpose (the cofactor matrix) of the blended matrix.

**Inverse kinematics.** Analytic two-bone IK by the law of cosines with a pole vector for
the bend plane (standard in production rigs: Maya's ikRPsolver, Unreal's Two Bone IK);
Aristidou and Lasenby, "FABRIK: A fast, iterative solver for the Inverse Kinematics
problem", Graphical Models 73(5), 2011; Wang and Chen, cyclic coordinate descent, IEEE
Trans. Robotics and Automation 7(4), 1991; Buss, "Introduction to inverse kinematics with
Jacobian transpose, pseudoinverse and damped least squares methods", 2004.

**Retargeting, contacts, quality.** Gleicher, "Retargetting motion to new characters",
SIGGRAPH 1998; Unity's humanoid Avatar and Unreal's IK Retargeter (rest-pose-relative
rotation transfer, translation scaled by a reference height, root motion handled apart);
Kovar, Schreiner and Gleicher, "Footskate cleanup for motion capture editing", SCA 2002
(contact detection, then IK to plant feet); Flash and Hogan, "The coordination of arm
movements", J. Neuroscience 5(7), 1985 (jerk as a smoothness measure). Continuity terms:
C0/C1/C2 for value, velocity and acceleration in time, as opposed to geometric G1 of a curve
drawing.

**Non-linear animation.** Blender's NLA (strips with action range, scale, repeat, reverse,
blend in/out, extrapolation hold/hold-forward/nothing, blend modes replace/add/combine,
tracks evaluated bottom to top); Unreal's additive animations (local-space deltas against a
reference pose); Godot's `Animation` resource (typed tracks, `AnimationLibrary`).

**Cameras and paths.** Guenter and Parent, "Computing the arc length of parametric curves",
IEEE CG&A 10(3), 1990 (Gauss-Legendre arc length, inverse by Newton); cinematography terms
pan, tilt, dolly, truck, pedestal/crane, orbit, zoom; SMPTE drop-frame rates 24000/1001 and
30000/1001 (exact rationals). FFmpeg's `image2` demuxer for numbered PNG sequences.

## 3. Language and execution decisions (made before writing each subsystem)

| Subsystem | Workload | Decision | Why |
| --- | --- | --- | --- |
| Public API, items, scripts, history, persistence, validation | orchestration, small data | Python | One coherent API; every other subsystem is Python; no hot loop |
| Curve storage | keys of thousands of channels | NumPy structure-of-arrays per curve (times, values, handles, modes), immutable | Contiguous, shareable by reference in the history, cheap to concatenate |
| Curve evaluation (all channels of all strips at one time; dense sampling for reports, bakes and graphs) | per element: binary search, polynomial, safeguarded Newton with data-dependent exits | NumPy vectorized evaluator over all channels at once (structure of arrays, vectorized bisection search, masked Newton). A Rust kernel with the identical arithmetic is added only if the benchmark shows curve evaluation dominating a frame of the large-scene workloads | Per-element branching makes masked NumPy iterations do some wasted work, but one call evaluates every channel; the polynomial and inversion paths use only `+ - * /` and comparisons (no `cbrt`/`acos`, unlike Blender's Cardano solve), so a bitwise native twin remains possible if it is needed |
| Easing functions, quaternion interpolation and blending | transcendentals (`sin`, `exp`, `acos`) over arrays | NumPy only | Vectorized well already; libm transcendentals would break bitwise twins |
| Skeleton forward kinematics | parent-first matrix products over all joints of all rigs | the existing `local_matrices`/`world_matrices` kernels and their NumPy twins | Already native and parity-tested; joints of every rig in one call |
| Skinning (linear blend, dual quaternion) | per vertex, 4-8 influences, matrix blend, cofactor normals | Rust kernel + NumPy twin, bitwise | The hottest loop of a deforming character; data-parallel, threads by vertex range |
| Morph targets | `base + sum_m w_m delta_m` | NumPy (one weighted sum per active target) | Memory-bound; already optimal in NumPy |
| IK (two-bone, FABRIK, CCD), constraints | small chains, few per rig | NumPy, vectorized across chains of equal length | Per-chain work is tiny; batching across characters removes the interpreter overhead |
| Rendering of animated frames | the existing CPU/GPU rasterizers and path tracers | no new shader code | Posed transforms enter as instance matrices; deformed meshes as new geometry. GPU skinning would only serve the GPU rasterizer (the path tracers need deformed geometry on the CPU for their BVHs), so it is deferred unless measurements show CPU skinning plus upload dominating frames |
| Graph images | SVG text, PNG | Python; PNG through the CAD drawing rasterizer and stroke font | Own code, no dependency |
| Video | numbered PNGs -> MP4 | optional external `ffmpeg` invoked only by the Python API or CLI when asked (never by MCP) | Not a renderer; no new dependency |

## 4. Architecture

```text
aether3d/animation/
  time.py         frame rates (exact rationals), time <-> frame, ranges, remapping
  easing.py       Penner easings with exact first/second derivatives
  curves.py       FCurve: keys, handles, interpolation, extrapolation, correction,
                  compiled segments, vectorized evaluation and derivatives
  modifiers.py    F-curve modifiers (cycles, noise, limits, envelope, stepped, ...)
  rotation.py     quaternion arrays: slerp, squad, log/exp, blending, Euler unwrap
  tracks.py       RotationTrack (quaternion keys, slerp/squad/cubic, timing easing)
  paths.py        typed property paths and the binding registry (no attribute traversal)
  clip.py         Clip: channels, events, markers; immutable, edited by replacement
  editing.py      key operations (insert, delete, move, scale, quantize, ...)
  nla.py          layers, strips, time mapping, blending (replace/additive, masks)
  skeleton.py     Skeleton (joints, parents, rest pose), FK, pose arrays
  skin.py         skins (influences, inverse bind), LBS/DQS, morph targets
  ik.py           two-bone, FABRIK, CCD
  constraints.py  object and joint constraints, evaluation order
  retarget.py, rootmotion.py, contacts.py, mirror.py
  cinematics.py   paths with arc length, shots, sequences, camera helpers
  evaluate.py     the evaluation pipeline -> AnimationPose
  system.py       AnimationSystem (scene.animation): items, history, API
  report.py       curve and motion diagnostics
  generators.py   editable motion generators (idle, walk, run, jump, land, wave, ...)
  fitting.py      simplification, resampling, baking
  graph.py        SVG/PNG graphs of curves and derivatives
  frames.py       frame-sequence rendering, contact sheets, video commands
  script.py, schema.py, guide.md   animation scripts
  gltf.py         glTF animations, skins, morph targets
```

**Authored versus evaluated.** Everything the scene saves stays authored state: object
transforms and fields are the *rest* values, and the animation document holds clips,
layers, rigs, skins and sequences as items with immutable definitions (CAD document
pattern: one history aspect per item, captured by reference; edits replace definitions).
Evaluation at a time `t` is a pure function of the authored scene at its revision and `t`;
it returns an immutable `AnimationPose` (object local-matrix overrides, visibility and
field overrides, material parameter overrides, deformed meshes, joint matrices, the shot
camera, diagnostics). It writes nothing: no history step, no revision, nothing saved, no
drift (nothing accumulates across frames). `apply_pose(t)` is the one explicit operation
that writes evaluated values into authored state (one undo step).

**Current time.** `scene.animation.time` is runtime state (not saved, not recorded, no
revision). Renders, spatial queries and observations of a scene with animation evaluate at
it unless given `time=`; caches that depend on it key on `(revision, time)`.

**Evaluation order** (documented contract):
1. clips -> strips -> layers for every animated object and rig (local values);
2. object local overrides -> evaluated world matrices (parents first);
3. rigs and object constraints, in dependency order (targets first; cycles refused when
   authored): joint locals -> FK -> joint constraints and IK -> attachments of rigid parts;
   object constraints (look-at, path follow, copy, limits) -> subtree world matrices again;
4. morph targets, then skinning (on demand, per deformed mesh);
5. the shot camera of the active sequence.

## 5. Algorithms (summary; formulas in docs/ANIMATION.md)

- **Bezier segments**: control points `(t_k, v_k)`, `(t_k + h^+_t, v_k + h^+_v)`,
  `(t_{k+1} - h^-_t, ...)`, `(t_{k+1}, v_{k+1})` with the handle correction above; power-basis
  coefficients precomputed per segment; the parameter `u` for a time is found by Newton's
  method safeguarded by bisection on `[0, 1]` (32 iterations at most, deterministic), the
  value by Horner. Derivatives in time: `v' = y'/x'`, `v'' = (y''x' - y'x'')/x'^3`,
  jerk from the next derivative; unweighted keys keep handle times at a third of the
  interval, making `x(u)` linear (no inversion).
- **Automatic handles**: slope of the parabola through the neighbouring keys (non-uniform
  Catmull-Rom); *auto-clamped* adds zero slope at local extrema and the Fritsch-Carlson
  limit, so a segment's values stay between its keys' values (no overshoot, guaranteed).
- **Rotation tracks**: per segment constant, slerp (short path, or as stored), squad
  (Shoemake), or glTF cubic (component Hermite, normalized); an optional timing easing
  maps segment time before interpolation. Angular velocity is analytic for slerp.
- **Blending**: replace layers interpolate (`lerp`, `slerp` with hemisphere alignment);
  additive layers add `w (value - reference)` and compose `delta^w` rotations; masks select
  channels or joints.
- **Skinning**: `v' = sum_j w_j S_j v`, `S_j = (W_m^-1 W_a)(t) J_j(t) B_j^-1`, normals by the
  cofactor of the blended 3x3; dual quaternion skinning for rigid palettes.
- **IK**: two-bone by the law of cosines in the pole plane (reach clamped or stretched);
  FABRIK with cone and hinge limits; CCD; residuals reported.

## 6. Robustness, safety and transactions

- Every input validated before any change: finite numbers, time ranges, key order, handle
  shapes, quaternion norms, joint hierarchies (acyclic, parents first), influence counts
  (at most 8 per vertex), weight sums, modifier depth, constraint graphs (acyclic), sizes
  (keys, channels, joints, vertices, frames) and time budgets with cooperative
  cancellation.
- Property paths come from a typed registry: no attribute traversal, no expression
  evaluation beyond the existing arithmetic whitelist.
- Scripts are data: no file access; frame rendering writes only through the existing
  workspace and atomic-write policies; video assembly never runs from MCP.
- A failed operation or script rolls back as one history transaction; a failed rig or
  constraint evaluation reports diagnostics and never changes the scene.

## 7. Test plan

Analytic fixtures for every interpolation and easing (values, first and second
derivatives against closed forms), property-based tests (random curves: monotone time
inversion, continuity at keys, clamped no-overshoot, evaluation order independence),
quaternion identities (unit norm, sign handling, slerp endpoints and constant angular
speed, squad continuity, Euler unwrap round trips), FK against independent matrix
products, skinning against a reference loop (rigid vertices exact, normalized weights,
bind pose identity, mirrored and non-uniform joints), IK reach and residuals, constraints
and cycles, NLA composition, retargeting with different proportions, root motion across
loops, contacts, persistence and migration, undo/redo identity, scripts (fuzzed against
crashes), CLI, MCP through a real server, glTF round trips checked by the Khronos
validator, render-at-time equal to rendering an applied pose, native kernels bitwise
against NumPy.

## 8. Benchmark plan

`benchmarks/bench_animation.py`: curve evaluation (1 to 100k channels, dense 10k-key
curves, hours-long timelines), quaternion tracks, FK for 15/60/250-joint rigs and 1,000
rigs, skinning 10k/100k/1M vertices (NumPy and native), IK chains, thousands of animated
objects per frame (evaluation + extraction + render), clip blending with hundreds of
strips, serialization, frame sequences on the CPU rasterizer, the GPU (RTX 4060 and
lavapipe labelled as software) and the path tracers.

## 9. Showcases

1. Minecraft-style character with an original 64x64 pixel skin: idle, walk, run, jump and
   a superhero landing, filmed by an animated three-quarter camera.
2. Graph-curve repair: a badly timed motion improved through handles and easing, measured
   and plotted (position, velocity, acceleration), the same edit through Python, a script
   and MCP.
3. Skinned character with IK arms and legs, foot planting, FK/IK blend, mirroring, layers.
4. Cinematic cameras: paths, orbit, dolly, follow, cuts between cameras, shot sequence.
5. Large animated scene with measured evaluation and render costs.
6. An agent workflow over MCP: rig, clip, diagnose, repair, verify, render, export.

## 10. Outcome

Written after the implementation (Aether3D 0.17.0, branch
`stage-17-animation-keyframes-rigging`). The guide ([docs/ANIMATION.md](../../docs/ANIMATION.md))
describes the code; this section records how it compares with the plan above.

### 10.1 What was built, against the architecture of section 4

| Planned | Built |
| --- | --- |
| `time`, `easing`, `curves`, `modifiers`, `rotation`, `tracks`, `paths`, `clip`, `nla`, `constraints`, `ik`, `cinematics`, `evaluate`, `system`, `report`, `generators`, `fitting`, `graph`, `frames`, `script`, `schema`, `guide.md` | as planned |
| `editing.py` (key operations) | in `clip.py` (pure functions on curves) |
| `skeleton.py`, `skin.py` | `rig.py` (skeletons, FK, attachments, skins, morph targets); the rigging pass in its own `rigging.py` |
| `retarget.py`, `rootmotion.py`, `contacts.py`, `mirror.py` | one `motion.py` |
| `animation/gltf.py` | `gltf/anim_export.py` and `gltf/anim_import.py`, beside the writer and reader they extend |
| -- | `batch.py`: batched evaluation of many curves and tracks (`CurveBatch`, `RotationBatch`), and `nla.StripTable` |

Everything in section 4's contract holds: authored state is never written by
evaluation; evaluation is pure and seekable; the documented order is the code's order;
two history aspects (`animation.item`, `animation.settings`) and project format 9 record
the document; spatial queries and observations are posed at the animation's time.

### 10.2 Languages, as decided and as measured

The decisions of section 3 held. The one conditional decision -- a Rust curve kernel
only if curve evaluation dominated a frame -- was measured and not taken: 10,000 objects
with their own clips evaluate in 9.9 ms per frame and a 10,000-key curve at 100,000
times in 6.0 ms, so there is no native curve kernel. Linear blend skinning got its Rust
kernel as planned (`aether3d-native` 0.17.0, ABI 4, bitwise equal to NumPy): 1,000,000
vertices in 21.2 ms on one thread and 6.9 ms on 20, against 171 ms in NumPy. GPU skinning
stayed deferred, as planned.

### 10.3 Deviations and limitations

- Dual quaternion skinning is NumPy only (678 ms for 1,000,000 vertices).
- FABRIK and CCD are Python loops (0.93 ms for 12 joints); joint limits are Euler bounds
  applied to the joints IK moves, not the cone and hinge limits inside FABRIK that
  section 5 planned.
- Squad is C1 at keys only for uniformly spaced keys.
- `camera_orbit` circles the target's position at the orbit's start; a moving target
  needs `follow_camera` (or a `look_at` with a path).
- Animated material parameters render through the full extraction, not the retained
  world's incremental path.
- glTF: no `KHR_animation_pointer` (camera, light, material, visibility channels are
  lossy), jumps become 1e-4 s transitions, values are float32.
- Timecode is non-drop-frame only.
- Generators give plausible, measured motion, not motion capture; the blocky superhero
  landing drags a toe 8.5 mm when it steps off (a knee-less leg).
- Motion blur averages renders; no renderer simulates depth of field.
- Measuring motion over time (`foot_contacts`, trajectories) evaluates the whole scene at
  every sample: seconds per character in a 10,000-object scene.
- The benchmark plan of section 8 was narrowed: rigs of 50 and 200 joints (not 1,000
  rigs; a crowd of 48 rigs instead), no serialization or path-tracer frame groups, and
  frames on the RTX 4060 and the CPU rasterizer only (lavapipe runs the GPU tests, not
  the benchmark).

### 10.4 Found by the showcases and fixed

Building the six showcases found real bugs, each fixed with a regression test:

- a layer mask's `descendants` flag was parsed but ignored at evaluation;
- the wave generator bent the elbow about its rest hinge axis, which the shoulder's
  abduction turns upward, so the forearm swung forward instead of rising;
- motion-blur sub-frames could sample before the rendered range;
- caches of authored state were keyed by the scene revision, which an open transaction
  does not advance: evaluation inside a transaction used values from before it;
- `foot_contacts` missed skating: a stance moving faster than the 0.3 m/s contact speed
  was no contact at all, so gross sliding read 0.0 (now a stance's sliding is measured
  against its steady velocity, which leaves correct motion's numbers unchanged);
- `plant_feet` pinned each contact at an anchor and an orientation from different
  instants, mid-roll, and sank toes 7.2 mm; it now pins one pose, lifted clear of the
  ground;
- the contact defaults differed between the API (5 cm) and the feedback check (3 cm).

And performance problems, each measured before and after: 10,000 instanced strips
timed one by one (163 -> 2.8 ms per frame), a crowd's rigging pass rebuilding the scene
order per rig (overlay 110 -> 31.6 ms), contacts sampling each joint in its own pass of
whole-scene evaluations (a third of the work now).

### 10.5 Measurements

On the development machine (i9-12900H, 20 logical CPUs, Python 3.14.8, NumPy 2.5.4,
RTX 4060 Laptop GPU), `benchmarks/results/stage17-animation.json`; full tables in
[docs/PERFORMANCE.md](../../docs/PERFORMANCE.md#animation-017):

| Workload | Time |
| --- | --- |
| 10,000 objects with their own clips: evaluation / overlay per frame | 9.9 / 14.2 ms |
| 10,000 objects playing one clip through offset strips | 2.8 / 7.2 ms |
| ... plus 48 walking rigs: overlay | 31.6 ms |
| 200-joint rig: rigging pass | 0.58 ms |
| two-bone IK / FABRIK, 12 joints | 12.4 us / 0.93 ms |
| skinning 1,000,000 vertices: NumPy / native 1 thread / 20 threads / DQS | 171 / 21.2 / 6.9 / 678 ms |
| generate a walk / a superhero landing | 230 / 746 ms |
| simplify 10,000 keys / fit 20,000 samples | 77 / 16.4 ms |
| 320x180 frames: CPU / GPU | 44.1 / 3.7 ms |
| glTF export (clips) / import | 42 / 10 ms |

### 10.6 Verification

The local matrix on the final tree (i9-12900H, Arch Linux; GitHub CI runs the PR on
Linux with Python 3.10 to 3.14, macOS and Windows, plus the optional-dependency, GPU,
Khronos-validator and native-kernel jobs):

| Configuration | Result |
| --- | --- |
| ruff check and format (Markdown code blocks included); `cargo fmt --check`, `cargo clippy --release -D warnings` | clean |
| Python 3.14.8, `aether3d-native` 0.17.0, the RTX 4060 (Vulkan), Khronos glTF Validator required | 3,227 passed, 5 skipped |
| the same without the native kernels (`AETHER3D_NATIVE=0`), GPU tests off | 3,095 passed, 137 skipped |
| the GPU test files on lavapipe (software Vulkan, as CI), GPU required | 190 passed, 2 skipped |
| portable CI: GPU tests off and GPU adapter enumeration refused by a hook | 3,095 passed, 137 skipped; only the three tests that check GPU-less behaviour on purpose reach the enumeration |
| Python 3.10.22 without manifold3d, MCP, jsonschema, wgpu or native kernels | 2,851 passed, 196 skipped |
| Python 3.10.22 with manifold3d 3.4.1 and MCP | 3,055 passed, 139 skipped |

The first run of this matrix (at 9147036) found two portability problems, fixed before
the PR in 251bc1c: a module-level `jsonschema` import that aborted the collection of a
test module where it is not installed, and three animation tests that reached GPU
adapter enumeration in the portable configuration (they now ask for the CPU explicitly).
The first, second, fifth and sixth rows were run again at 251bc1c; the other three are
from 9147036 (251bc1c changes only those tests, an example and Markdown formatting).

The five skips in the main configuration are environment-specific tests of earlier stages
(three CAD cross-checks that need an external validator Python with OCP, ifcopenshell or
ezdxf; a GPU path-tracer fallback test that needs an adapter unable to run it; a CLI test
for machines without the MCP SDK); no animation test skips.

The nine animation test files hold 262 tests (c1bec02 added the last one, a regression
test for the motion-blur range, after the matrix ran; it passes); `test_native.py`, `test_docs_examples.py` and
`test_examples.py` gained eight more test functions (the skinning kernel, the quick
starts and the guide's script, the six showcases and their committed outputs).
