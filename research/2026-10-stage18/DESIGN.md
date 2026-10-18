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

(Written after the implementation.)
