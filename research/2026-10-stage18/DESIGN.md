# Stage 18 design record: advanced character rigging

Written on 18 October 2026 against `main` at 59b68a2 (v0.18.0: Stages 1-17). The roadmap
name stays "Stage 18"; the version moves on to **0.19.0**. This is the compact gap
analysis and the plan the implementation follows. The user guide
([docs/CHARACTER_RIGGING.md](../../docs/CHARACTER_RIGGING.md)) describes what was
built; section 9, written afterwards, records the outcome.

## 1. Starting point (inspected in the source and tests)

Stage 17 already has, and Stage 18 keeps and extends (no second skeleton, solver,
timeline or skin representation):

| Stage 17 (verified) | Where | Limitation found |
| --- | --- | --- |
| Immutable `Rig` of `Joint`s (stable ids, unique names, parents first, rest TRS, `tail`, Euler limits), mirror name pairs | `animation/rig.py` | No roles, no helper (non-deforming) joints, no controls, no landmarks; only whole-rig replacement (`set_rig`): renaming or moving a joint silently breaks clip paths, IK chains and skins |
| FK through the scene's hierarchy kernel (native when installed) | `rig.forward_kinematics` | -- |
| IK: analytic two-bone, FABRIK, CCD; joint limits as Euler XYZ bounds on IK-moved joints | `animation/ik.py`, `rigging.py` | Python loops; Euler bounds wrap badly near +-180 deg and cannot express cones, hinges or twist ranges; no IK/FK switching or matching |
| Constraints (look-at, copy, child-of, limits, distance, follow-path) with dependency order and cycle refusal | `animation/constraints.py` | `Constraint.p` re-parses JSON on every access during evaluation; no twist distribution |
| Attachments (`W' = A' J' (A J)^-1 W`) | `rigging.py` | -- (bone-attached rigid parts: keep) |
| Skins: up to 8 influences; LBS (native Rust, bitwise NumPy twin); DQS (NumPy, 658 ms per 1M vertices) | `rig.py`, `native/src/skin.rs` | Weights only explicit, `nearest` or `envelope` (Euclidean, leaks across limbs that touch); no weight editing, mirroring, smoothing or transfer; DQS slow; no LBS/DQS mix |
| Morph targets before skinning, animated weights | `rig.Morph` | No pose-driven correctives |
| Generators (idle, walk, run, jump, land, wave, head turn), roles by joint name, retargeting, mirroring, root motion, foot contacts, planting | `generators.py`, `motion.py` | Roles are guessed from names only |
| glTF skeleton/skin/morph/animation export (exact or fitted, bake) and import | `gltf/anim_*.py` | DQS exports as LBS silently |
| Scripts, CLI, MCP (`run_script`, `inspect_animation`) | `script.py`, `_cli_animation.py`, `mcp/animation.py` | No rig authoring operations beyond `add_rig`/`set_rig`/`skin` |
| Rigid-part characters animated by object transforms or by attachments | `examples/animation_common.py` | Must stay untouched (requirement 10) |

Baseline (v0.18.0, this machine, `benchmarks/results/stage18-baseline.json`): LBS 1M
vertices 164 ms NumPy / 20.5 ms native 1 thread / 6.5 ms 20 threads; DQS 658 ms per 1M;
two-bone IK 11.8 us; FABRIK 12 joints 0.90 ms; 48-rig crowd overlay 30.5 ms.

## 2. Research (sources used)

* Baran and Popovic, "Automatic Rigging and Animation of 3D Characters", SIGGRAPH 2007
  (template embedding, bone heat). Dionne and de Lasa, "Geodesic Voxel Binding for
  Production Character Meshes", SCA 2013: weights from geodesic distances inside a
  voxelized solid -- robust to open, non-manifold, disconnected and interpenetrating
  meshes, and no leakage between limbs that touch only across empty space. Chosen as the
  default automatic weights.
* Jacobson et al., "Bounded Biharmonic Weights", SIGGRAPH 2011: needs a QP solver over a
  volumetric mesh; not implemented (documented).
* Kavan et al., "Geometric Skinning with Approximate Dual Quaternion Blending", TOG 2008
  (DQS with the scale stage, already used); per-vertex LBS/DQS blending as in production
  "weight blended" skinning.
* Lewis, Cordner and Fong, "Pose Space Deformation", SIGGRAPH 2000: correctives as morph
  targets driven by joint rotations (here: one driver per corrective, an explicit curve).
* Swing-twist decomposition of quaternions (Dobrowolski, arXiv:1506.05481, 2015) and
  elliptic swing cones (Baerlocher and Boulic, "Parametrization and Range of Motion of
  the Ball-and-Socket Joint", 2001).
* Borgefors, "Distance transformations in digital images", 1986 (3-4-5 chamfer metric);
  Dial's bucket queue for integer shortest paths: exact integer results, so the native
  and NumPy implementations agree bit for bit by construction.
* Anthropometric segment proportions (Drillis and Contini 1966, as tabulated in Winter,
  *Biomechanics and Motor Control of Human Movement*, 4th ed.) only as **priors** for
  landmarks the geometry does not decide; measured geometry always wins and every prior
  used is reported.

## 3. Languages (decided per subsystem)

| Subsystem | Language | Why |
| --- | --- | --- |
| Rig document, editing, templates, controls, diagnostics, scripts, CLI, MCP | Python | Data validation and orchestration; small per-call work |
| Voxelization, slice analysis, landmark fitting | NumPy | Whole-grid array operations (run-length flood fill, label propagation); measured < 1 s at the default resolution |
| Geodesic voxel distances | Rust + NumPy twin | Priority-queue shortest paths are sequential per source (NumPy: bucket-vectorized Dial); integer metric makes the twin exact |
| Weight generation from distances, editing, smoothing, mirroring, transfer | NumPy | Vectorized per vertex / per edge |
| DQS (and the LBS/DQS blend) | Rust + NumPy twin | Per-vertex hot loop; the NumPy reference is rewritten in explicit operation order so the twin is bitwise |
| LBS | Rust (existing) | Unchanged |
| IK solvers | Python/NumPy | Per-chain work is microseconds; the cost is the rigging pass's per-constraint overhead, removed by compiling rigs once per revision |
| GPU skinning (WGSL) | not built | Every consumer (path tracers, queries, motion blur, export, CPU raster) needs CPU positions; uploading skinned positions measured cheaper than keeping two skinning paths coherent (section 9) |

## 4. Architecture

* **Rig document** (extended, backward compatible): joints gain `role` (semantic: `hips`,
  `upper_arm.L`, ...), `deform` (helper joints get no weights). Limits gain `swing`
  (cone or ellipse about the bone axis), `twist` and `hinge` forms (Euler bounds stay).
  The rig gains `controls` (named, typed controls backed by ordinary scene objects:
  IK targets, poles, gaze, ...), `correctives` (morph targets driven by a joint's swing,
  twist or hinge angle), `landmarks` (editable, armature space), `template`, `info`
  (forward/up axes, description) and `limits` (`"ik"` -- Stage 17's rule -- or
  `"always"`). Skins gain the method `blended` with a per-vertex DQS fraction, and
  persistent weight locks. Project format 11 (identity migration from 10).
* **Structural editing** (`animation/rig_edit.py`): add, remove, move, orient, rename,
  reparent, insert, split, mirror; each returns the new rig plus the dependent changes
  (clip channel renames, layer masks, constraint names, skin palettes and bind matrices,
  weight transfer to the parent) applied in one undo step with an explicit rest-change
  policy (`keep_world` children, `rebind` or `keep_bind` skins). Validation refuses
  cycles, duplicates, near-zero bones and non-finite values.
* **Auto-rigging** (`animation/autorig.py`): voxelized solid of all given meshes ->
  orientation and symmetry -> geodesic extremities and limb paths -> landmarks (each with
  confidence, method and explanation; overrides) -> a template rig (humanoid, stylized,
  blocky, quadruped) with oriented joints, limits, controls, IK and correctives -> weights.
* **Weights** (`animation/weights.py`): generation (`geodesic` voxel binding, `envelope`,
  `nearest`, `rigid` per component), editing operations on selections with numeric
  brushes, locks, smoothing that never crosses disconnected parts, mirroring with a
  vertex map, transfer by closest surface point with distance and confidence, and
  validation/inspection with heatmaps.
* **Evaluation order** (extends Stage 17's contract): channels; units in dependency order
  (rig: joint channels, FK, joint constraints incl. twist and swing-twist limits,
  attachments); **correctives** (from final joint rotations); morph targets; skins
  (LBS, DQS or blended); the shot camera.
* **Controls and posing** (`animation/controls.py`): control objects with IK
  constraints, IK/FK switching with matched poses (keys written both ways and verified),
  high-level pose commands that become keys on controls and joints, with residuals.
* **Diagnostics and fixes** (`animation/rig_report.py`): numeric checks with codes,
  severity, ids and vertex lists; a diagnostic pose suite; proposals with safety classes
  applied in one undo step, verified, and undone when they do not improve.

## 5. Tests, benchmarks, showcases (plan)

Tests: independent references (per-vertex loops for DQS, brute-force geodesics), property
tests (random rigs and weights), adversarial scripts, save/undo/redo, glTF validator,
native/NumPy parity, CPU-only and optional-dependency-free runs. Benchmarks: the Stage 17
groups compared with the baseline plus rig generation, weight generation, smoothing,
transfer, 1/10/50 characters, memory per rig. Showcases A-G as specified, rendered by
Aether3D only, with key, fill and rim lights.

## 9. Outcome

Written after the implementation (branch `stage-18-advanced-rigging`, version 0.19.0),
against the plan above. [docs/CHARACTER_RIGGING.md](../../docs/CHARACTER_RIGGING.md)
describes the code as it is; this section records where it differs from the plan, what
the showcases found and what was measured.

### 9.1 What was built, against section 4

* **Rig document** -- as planned (roles, helper joints, swing-twist and hinge limits
  relative to the rest pose, controls, correctives, landmarks, template, info,
  `enforce_limits`; blended skins and weight locks; format 11). Added on the way: twist
  constraints, and joint constraints evaluated in dependency order -- the IK/FK switch
  tests showed the forearm twist helpers reading the hand before the arm's IK moved it.
* **Structural editing** -- as planned (`rig_edit.py`).
* **Auto-rigging** -- as planned, all on the voxelized solid: the mirror plane by a sweep
  (decided on a sample of the voxels, confirmed on all), extremities as maxima of the
  geodesic distance from the core, limbs from its Reeb graph, an eroded solid to separate
  touching limbs, joints at measured sections and bends, priors only where the mesh shows
  nothing. On the recipe's six test bodies (adult A-pose, T-pose, arms down, child,
  stylized, wide stance) every landmark lies within 4 % of the height of the modeled
  joint (the worst: 62 mm, the adult's shoulder). The blocky template attaches separate
  blocks (no skinning cost); the quadruped template is basic (one spine chain, four
  three-joint legs, a tail).
* **Weights** -- geodesic voxel binding as planned (Dial buckets, native twin), policies
  and safeguards. Not in the plan and needed: the auto-rig's body-part **regions** (the
  detection's limbs and torso decide which joints may weigh where, with smooth bands
  across the junctions); the showcases drove five measured revisions of them (9.4).
* **Evaluation order** -- as planned.
* **Controls and posing** -- as planned; IK bend axes are aligned only as far as the
  middle joint's limits require (a relaxed limb keeps its roll, so switching it changes
  nothing visible).
* **Diagnostics and fixes** -- as planned, plus poses as data; verified fixes re-measure
  only what a fix can change (9.5).
* **Beyond section 4** (required by the assignment): the rigid-part workflow
  (`rigid.py`: `rigid_parts`, `generate_rigid`, `convert_character`), the feedback `rig`
  check and `rig_fix` corrections, glTF joint semantics and loss reports, the MCP tool
  `inspect_rig` and `aether3d rig`, and implicit surfaces for an original mannequin.

### 9.2 Languages, as decided and as measured

| Subsystem | Decided (section 3) | Built and measured |
| --- | --- | --- |
| Rig document, editing, templates, controls, diagnostics, scripts, CLI, MCP | Python | as decided; an IK/FK switch 8.4 ms, four pose commands 96 ms, `diagnose_rig` over 13 poses 1.0 s |
| Voxelization, slice analysis, landmark fitting | NumPy | as decided: landmarks on an 18,008-vertex body in 0.30 s at 128 cells; the symmetry sweep runs on a sample of the voxels and decides on all of them (0.59 -> 0.45 s for a whole auto-rig, identical landmarks) |
| Geodesic voxel distances | Rust + NumPy twin | `native/src/geodesic.rs` (ABI 6): Dial's buckets on the integer 3-4-5 chamfer metric, bitwise equal to the bucket-vectorized NumPy twin; all of an 18,008-vertex auto-rig's geodesic distances (landmarks, body-part regions, weights for 23 bones: 20 calls) take about 60 ms |
| Weight generation, editing, smoothing, mirroring, transfer | NumPy | as decided: generation 0.21 s for 18,008 vertices; smoothing by one bincount per changing column (the bits of `np.add.at`, 418 -> 51 ms); transfer to a 9,994-vertex jacket 1.4 s (closest points: the slowest step left) |
| DQS and the LBS/DQS blend | Rust + NumPy twin | `native/src/dqs.rs`, bitwise: 1M vertices 393 ms NumPy, 66 ms one thread, 14.7 ms 20 threads; the NumPy reference gives v0.18.0's results bit for bit |
| LBS | Rust (existing) | unchanged |
| IK solvers and the rigging pass | Python/NumPy | as decided, with the per-constraint overhead removed where profiling found it: twist constraints in scalar arithmetic, constraint parameters parsed once (50 walking characters 27.9 -> 16.7 ms a frame); no compiled rig was needed |
| GPU skinning (WGSL) | not built | not built: a posed frame of a 6,488-vertex character costs the GPU rasterizer 12.2 ms against 11.0 ms at rest (skinning and upload about 1.2 ms), and every other consumer needs CPU positions |

### 9.3 Deviations and limitations

* Not built: bounded biharmonic weights (a QP solver over a volume mesh), GPU skinning
  (9.2), a full-body IK solver, finger and face rigging, muscle or collision deformation.
* Weights are distances with rules, not paint: a knee bent 140 degrees keeps about 0.76
  of its size with linear blend skinning (Stage 17's envelopes keep 0.87 there, and lose
  more at the shoulder: 0.71 against 0.90), hips flexed past 90 degrees fold their groin.
  The verified blend adds dual quaternion skinning only where a collapse is measured.
* Skinned normals stay Stage 17's (rest normals through the blended matrices). Under a
  strong twist the facets turn further than the normals where weights change quickly; the
  path tracers then leave jagged dark patches at a light's terminator (Showcase B's
  twisted torso, every method alike). Recomputing normals from the deformed facets
  removed most of them but showed the marching-cubes facets everywhere else; the
  renderers' facet test was left alone (not Stage 18's subsystem).
* The full list is in CHARACTER_RIGGING.md's limitations.

### 9.4 Found by the showcases and fixed

* **Crouch (A):** `plant_feet` measured each foot with its leg IK off; in a held crouch
  the straight FK leg put the foot 0.37 m below the floor and the next keys planted it
  there. Feet are measured as the pose evaluates (a test holds a crouch).
* **IK/FK (C):** switching IK to FK turned the forearm twist helpers up to 10 degrees
  (constraint order, above); switching FK to IK rolled a relaxed arm's bones (5.7 mm of
  skin), and a fixed fade of the alignment left a 5 cm reach residual -- the limit-aware
  alignment fixed both (switches move no joint; residuals 0).
* **Deformation (B), five weight revisions, each measured over the pose suite on the
  18,004-vertex body:**
  1. seams where the nearest bone changes: the skeleton-neighbourhood rule was applied
     again after smoothing; now only regions and sides are (a torso twisted 45 degrees:
     worst stretch 11.0 -> 3.5; no edge with a weight jump above 0.5, 119 before);
  2. a crease at the hip: the thigh reached 1.5 radii into the torso; now 3, and the
     legs' joints fade out at the mid-plane where the legs meet (a hip flexed 90: worst
     stretch 14.6 -> 7.3);
  3. collapsing knees and elbows: inverse-square weights from a limb's surface reach far
     along it (10 cm above the knee the shin still weighed 0.23); the organic policy
     falls off with the cube (a knee bent 140: 0.69 -> 0.76 of its size);
  4. armpits crumpling when the torso twists: armpit skin carried a third of its weight
     on the hips and the lower spine; near an arm's or the neck's junction the lower torso
     joints now fade out over three junction bands, and the spine's joints are smoothed
     among themselves (a torso twisted 60: flipped triangles 124 -> 15);
  5. clothing drawn by whole triangles made staircase seams: the showcases cut the mesh
     along the clothes' edges before rigging (`rigging_common.tailor`).
* **Stretch maps (B)** showed faint pink on white: the ramp now saturates at 1.41x and
  0.71x on a grey body, and previews gain a `left` view.
* **Outfit (D):** the walk rendered without the helmet, the shoulder plates and the
  backpack -- the showcase placed and turned the armature after auto-rigging, and an
  attachment follows its joint relative to the authored armature, so the pieces stayed
  4 m away. The auto-rig's `attach` mode now parents each piece to the armature where it
  is (Stage 17's attachments unchanged; tested with a placed, turned, walking character).
* **IK/FK (C):** fixed world targets lay beyond the full-resolution rig's arm (0.536 m;
  the coarser quick rig's is longer): the right hand stopped 2.6 and 8.2 cm short. The
  goals are now placed along the same directions at 85-90 % of the measured arm length.
  After the switch back to FK the idle's leg keys walked the stepping foot off its block;
  the idle now moves the upper body only, and the legs hold (0.0 m).
* **World (F):** the crane flew into a tree crown 14 m beside the road: trees are cleared
  along its path and its clearance above the ground is measured (at least 1.41 m).
* **The mannequin (A-G):** its nose, a small ellipsoid, faced its underside away from
  every light and read as a dark pit in close-ups; it is a soft ridge now.
* Weight smoothing used `np.add.at` on every column: one bincount per changing column
  adds in the same order (the same bits, tested) -- 20 iterations of three joints on
  18,004 vertices 418 -> 51 ms.

### 9.5 Measurements

`benchmarks/bench_rigging.py` at 662f319 (`benchmarks/results/stage18-rigging.json`;
tables in [docs/PERFORMANCE.md](../../docs/PERFORMANCE.md#character-rigging-019)):
auto-rigging 6,488 / 18,008 vertices in 0.47 / 0.67 s; 50 auto-rigged characters walking
with IK, twist helpers and gaze in 16.9 ms per frame; 50 Minecraft-style characters in
1.45 ms as rigid parts, 6.8 ms as bone-attached blocks, 3.6 ms as rigidly skinned meshes;
dual quaternion skinning of a million vertices in 14.7 ms on 20 threads. Stage 17's
`bench_animation.py`, run interleaved with `main` (59b68a2, with its own native 0.18.0
build) on one machine state, agrees within -5 % to +6.5 % in every group
(`stage18-animation-interleaved.json`); its dual quaternion skinning is 9-11 times faster.
A full run right after the interleaved one read the 18,008-vertex auto-rig at 0.86 s; the
group alone, and the full run repeated on the idle machine, read 0.66-0.67 s, which is
what in-process timing shows -- the laptop's state, not the code (the saved file is the
repeated run). The showcases' reports hold every render's timing (A-G in
[docs/CHARACTER_RIGGING.md](../../docs/CHARACTER_RIGGING.md#showcases)).

### 9.6 Verification

The local matrix on the final code (i9-12900H, Arch Linux; GitHub CI runs the PR on
Linux with Python 3.10 to 3.14, macOS and Windows, plus the optional-dependency, GPU,
Khronos-validator and native-kernel jobs):

| Configuration | Result |
| --- | --- |
| ruff check and format (Markdown code blocks included); `cargo fmt --check`, `cargo clippy --release -D warnings` | clean |
| Python 3.14.8, `aether3d-native` 0.19.0, the RTX 4060 (Vulkan), Khronos glTF Validator required | 3,474 passed, 5 skipped |
| the same without the native kernels (`AETHER3D_NATIVE=0`), GPU tests off | 3,332 passed, 147 skipped |
| the GPU test files on lavapipe (software Vulkan, as CI), GPU required | 191 passed, 2 skipped |
| portable CI: GPU tests off and GPU adapter enumeration refused by a hook | 3,337 passed, 142 skipped; only the three tests that check GPU-less behaviour on purpose reach the enumeration |
| Python 3.11.17 with NumPy, Pillow and trimesh only (no manifold3d, MCP, jsonschema, wgpu or native kernels) | 3,062 passed, 211 skipped |
| the same with manifold3d 3.4.1, the oldest supported | 3,207 passed, 178 skipped |
| Python 3.10.22 with NumPy, Pillow and trimesh only | 3,062 passed, 211 skipped |

The first run of this matrix (at bc5497d) found one portability problem, fixed in
116df53: `aether3d rig info` imported its summary from the MCP server package, which
needs the optional MCP SDK, so the command failed in the last three configurations
(`test_the_rig_command_line`, the only failure in each; 3,060, 3,205 and 3,060 passed).
The summary now lives in the engine (`rig_report.rig_summary`), an object without a rig
gets the command line's `RIG_MISSING` error instead of the server's exception, and a
subprocess test checks that the rig commands load no MCP module (it fails on bc5497d).
The last three rows were run again at 116df53, and so were the rigging, MCP, plans-CLI
and documentation test files, with the native kernels and the GPU (187 passed, 1 skipped) and
without the native kernels (185 passed, 3 skipped); the other rows are from bc5497d (116df53
changes only the rig command's import, the tool's summary, a test and a line of
docs/ARCHITECTURE.md).

The five skips in the main configuration are environment-specific tests of earlier stages
(three CAD cross-checks that need an external validator Python with OCP, ifcopenshell or
ezdxf; a GPU path-tracer fallback test that needs an adapter unable to run it; a CLI test
for machines without the MCP SDK); no rigging test skips. Without the native kernels, five
showcase runs whose worlds need a regional stage skip (four of Stage 16's and F: minutes
on NumPy alone; CI's native jobs run them).

Twelve new test files hold 83 test functions (109 tests with their parameters): the
rig document and its migration, limits relative to the rest, landmarks against the
modeled joints (placed anywhere, lying, overridden), auto-rigging humanoid, stylized,
bent, blocky and quadruped bodies, weights (exact geodesic distances, no leaks between
touching legs, no seams where bones change, brushes, locks, mirroring, transfer, the
scatter order of smoothing bit for bit, time limits), structural edits (renames into
clips, masks and constraints; orientation changes that convert the animation exactly),
IK/FK switches that move no joint, pose commands, diagnostics, verified fixes undone when
they do not help, the feedback `rig` check, rigid-part characters (no bones; Stage 17's
Nova against the v0.18.0 fixture), Stage 17's layers, strips, retargeting and mirroring on
auto-rigged characters, glTF round trips through the validator, scripts (a failing one
changes nothing), the command line and MCP through a real server process.
`test_native.py`, `test_docs_examples.py` and `test_examples.py` gained seven more test
functions (the DQS and geodesic twins, bit for bit; the guide's quick start, script and
pose examples and the rigid-parts example; the seven showcases and their committed
outputs). The suite grew from 3,351 tests (v0.18.0) to 3,480.
