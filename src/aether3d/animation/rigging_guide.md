# Aether3D rigging: a guide for agents

Rig an unrigged character, bind and refine its weights, pose and animate it, verify the
deformation, and fix what the measurements show -- all as data. Rigging operations are
`aether3d.animation-script` operations (`run_script`; one undo step, all or nothing).
`inspect_rig` answers read-only questions and draws previews. On the command line:
`aether3d rig autorig | test | pose | ikfk | bind | transfer | edit | create | fix ...`.

## 1. Look before you rig

```json
{"what": "landmarks", "meshes": ["Body"]}
```

`inspect_rig` returns landmarks (pelvis, hips, knees, ankles, heels, balls, toe tips,
spine, chest, neck, head, clavicles, shoulders, elbows, wrists, hand tips, and the
fingers' knuckles, joints and tips when the mesh shows them) in world space, each with a
`confidence`, the `method` that placed it and a `note`; the facing it measured; the pose
(`A-pose`, `T-pose`, `arms down`); and diagnostics (`FORWARD_AMBIGUOUS`,
`HAND_NOT_FOUND`, `LEGS_FUSED`, `ARM_AGAINST_BODY`, `FINGERS_UNCERTAIN`...). Landmarks are
measured on the mesh surface (sections, ray probes, surface distances); a landmark from a
`proportion` prior was not measured: check it, and pass a corrected point in `landmarks`
when you rig. A symmetric body's hands must find the same fingers: when they do not (the
mesh is too coarse for fingers that thin), the hands get no finger landmarks and
`FINGERS_UNCERTAIN` says so -- remesh finer or give finger landmarks.

## 2. Rig and bind in one step

```json
{"format": "aether3d.animation-script", "version": 1, "operations": [
  {"op": "autorig", "meshes": ["Body", "Jacket", "Helmet"], "template": "humanoid",
   "parts": {"Jacket": "transfer", "Helmet": "attach"},
   "landmarks": {"knee.L": [0.1, -0.02, 0.48]}}
]}
```

Templates: `humanoid`, `stylized` (tighter weight blends), `blocky` (Minecraft-style
blocks bound rigidly), `quadruped`. Parts: `generate` (the default: weights by
`weights.method`, bone heat unless you ask otherwise; `heat` or `geodesic` choose the
method for one part), `transfer` (the first generated part's weights at the closest
surface point: clothing), `rigid` (one joint for the whole part), `attach` (parented to
its nearest joint), `skip`. The result names the armature, the joints, the control
objects and the weights of every part.

What you get (humanoid): Hips, Spine, Chest, UpperChest, Neck, Head; Pelvis.L/R (keep the
flank on the hips); Shoulder (clavicle), UpperArm, UpperArmTwist, ForeArm, ForeArmTwist,
Hand and three joints per finger and thumb when the hands show them; Thigh, ThighTwist,
Shin, Foot, Toe -- all `.L`/`.R`, with `role`s. Twist joints split their bone (its
`span`) and turn with part of the limb's twist, like Rigify's split deform bones. Bone
frames: +Y along the bone, +X the bend axis (a positive rotation about +X bends elbows
forward, knees back and fingers into the palm; a negative one raises a thigh or an arm
forward); swing-twist limits; IK targets for hands and feet, elbow and knee poles and a
gaze target (objects under the armature, IK influence 0: FK until you switch).

Weights (`generate_weights`, or `weights` in `autorig`): `method` `heat` (Baran and
Popovic's bone heat, as Blender's Automatic Weights: a sparse diffusion from each
vertex's nearest visible bone; on a mesh both were given, the dominant joint agrees with
Blender's on 99.8 % of the vertices, and the figure deforms as Blender's weights do),
`geodesic` (distances inside the voxelized solid), `envelope`, `nearest`; `policy`
organic, stylized, rigid; `diffusion`, `visibility`; up to eight `influences` per vertex
(capped at four, a twisted torso creases where a fifth joint takes over). Skins shade
with **surface normals** (`normals: "surface"`, turned as the deformed surface turns).
**Corrective smoothing** (`corrective: true` or `{iterations, strength, region,
weights}`; delta mush on cotangent weights) relaxes pinches, creases and folds where
weights blend -- a little on heat weights, a lot on skins capped at four influences
(imported ones) -- but glTF cannot carry it: exports report `SKIN_SMOOTHING_LOSSY` (pass
`allow_lossy` to export the plain skin).

## 3. Animate

* `{"op": "generate", "armature": "Body Rig", "motion": "walk", "forward": "-y"}` --
  walk, run, jump, land, idle, wave, head_turn: ordinary keys.
* `{"op": "pose_character", "armature": "Body Rig", "time": 2.0, "commands": [...]}`:
  * `{"do": "reach", "hand": "R", "target": [0.4, -0.5, 1.3]}` (IK; leans the chest
    when out of reach), `look_at`, `plant_feet`, `crouch` (`depth`), `lean`, `aim_hips`;
  * `{"do": "raise_arm", "side": "L", "angle": 150, "direction": "side"}` -- the arm
    points exactly where asked and the clavicle elevates with it (1 degree per 6 beyond
    30; `shoulder`: a fraction, or 0 for none);
  * `{"do": "curl_fingers", "side": "R", "shape": "fist"}` -- `curl` (`amount` 0..1),
    `fist`, `point`; `fingers` (some of thumb, index, middle, ring, pinky), `spread`;
  * `{"do": "roll_foot", "side": "L", "angle": 30}` -- about the heel (`angle` < 0),
    the ball (the toes stay flat) and, beyond `break` (35), the toe tip; absolute: the
    same roll gives the same pose whatever was keyed before;
  * `{"do": "rotate", "joint": "Head", "euler": [0, 0, 30]}` -- a joint, from its rest.
  Reports say whether a reach was possible and the IK residuals.
* `{"op": "ik_fk", "armature": "Body Rig", "limb": "leg_ik.L", "mode": "ik", "time":
  1.2}` -- switch at a time with a matched pose (the report's `mismatch` is the largest
  joint displacement, metres).
* Joint rotation channels (`joints["ForeArm.L"].rotation`) are **absolute** local
  rotations, rest included (as glTF): key `rest * delta`, or use `pose_character`
  `rotate`, which does that for you.

## 4. Verify, then fix

```json
{"what": "diagnose", "target": "Body Rig", "suite": "stress"}
```

Findings are measured (`basis: numeric`): `JOINT_COLLAPSE` (the skin around a bent joint
keeps less than 75 % of its size), `SKIN_CREASE` (edges smooth at rest bent past 60
degrees), `SKIN_FOLD` (triangles turned against their posed neighbourhood),
`FLIPPED_TRIANGLES`, `STRETCHED_EDGES`, `INTERPENETRATION`, `WEIGHT_LEAK`,
`CROSS_SIDE_WEIGHTS`, `BIND_MISMATCH`, `ASYMMETRIC_JOINTS`, `IK_FLIP`, `IK_UNREACHABLE`,
`FOOT_SLIDING`... Poses: `suite` (elbows and knees at 120 degrees, hips at 90, shoulders
raised 90, wrists twisted 90, torso 45, the head turned 60) or `stress` (elbows and knees
at 140, hips at 100, shoulders raised 150 with the clavicle, the torso twisted 60), or
your own `{name, turns, joint}`; every pose reports `joint_ring` (kept size),
`new_creases`, `sharpest_crease`, `folded_triangles` and `worst_stretch`. On the command
line: `aether3d rig test PROJECT --armature A --sheet stress.png` (exit code 3 on
errors). Look at `{"what": "preview", "target": "Body Rig", "mode": "stretch", "pose":
"hip_flex_100.L"}` (red stretched, blue compressed) or `"mode": "weights", "joint":
"UpperArm.L"`.

Then `{"what": "fixes", "target": "Body Rig"}` lists proposals with a safety class, and

```json
{"op": "apply_rig_fix", "armature": "Body Rig", "fixes": "verify"}
```

applies every `safe` and `verify` proposal, one at a time, re-measuring after each and
undoing any that does not improve what it targets or makes anything worse (the result
lists `applied`, `reverted` and why). `approval` proposals are choices: mirroring joints,
and corrective smoothing (`smooth_skin`: fewer creases and folds in most poses, sometimes
a few more at the hardest bends) -- run them with `dry_run: true` first and read
`changes` (per pose: kept size, creases, folds, flips, stretch before and after).

## 5. Edit by hand when you know better

* `edit_rig`: `add_joint`, `remove_joint` (weights and attachments go to the parent),
  `move_joint`, `orient_joint`, `rename_joint` (clips, layer masks, constraints and
  controls follow), `reparent_joint`, `insert_joint`/`split_bone` (weights split along the
  bone), `mirror_joints`, `align_bone`, `set_joint`; `skins: "rebind"` keeps the mesh
  where it is; `animation: "convert"` rewrites channels whose joint frames changed.
* `edit_weights`: `assign`, `add`, `subtract`, `scale`, `replace`, `normalize`, `prune`,
  `cap`, `smooth`, `mirror`, `rigid`, `lock`, `unlock`, on selections (`sphere`, `box`,
  `plane`, `joint`, `dominant`, `role`, `component`, `near_joint`, `and`/`or`/`not`) with
  numeric brushes (`falloff`: center or joint, radius, curve, euclidean or geodesic).
* `transfer_weights` for clothing; `generate_weights` to redo one mesh; `skin` to bind
  given weights (`method` linear, dual_quaternion or blended; `normals`; `smooth`).

## Rigid-part characters need no bones

A Minecraft-style character can stay a hierarchy of solid parts (head, torso, arms, legs)
moved by their own object transforms. Nothing here is required for that and nothing here
changes it: no armature, no skin, no weights.

* `{"what": "parts", "target": "Hero"}` (`inspect_rig`) shows the parts found under the
  root (by name: head, torso/body, arm, leg; sides measured), their pivots and any
  `SIDE_NAME_MISMATCH`. Name them yourself with `roles` when the names say nothing.
* `{"op": "generate_rigid", "root": "Hero", "motion": "walk"}` writes walk, run, jump,
  land, idle, wave or head_turn as ordinary `transform` clips on the parts (travel on the
  root); the report gives the fitting error, a verification against the motion and the
  feet's sliding. Edit the keys like any others.
* Bones only when you want them: `attach` (blocks follow joints), `autorig` with
  `template: "blocky"` (separate blocks are attached, not skinned), or
  `{"op": "convert_character", "source": "Hero", "to": "skeleton"}` -- an armature with
  attached copies of the parts and the motion baked onto its joints, verified; the
  original stays (hidden) and `to: "rigid"` converts back.

## Honest limits

Automatic rigging measures geometry; it does not know anatomy beyond proportion priors
and cannot be right for every mesh: read the confidences. Weights are diffusion (bone
heat) or distances, not a muscle or tissue simulation; deformation is linear, dual
quaternion or blended skinning, corrective smoothing and correctives (morph targets
driven by joint angles) -- not physically based, and nothing guarantees anatomical
correctness. Extreme poses still crease and stretch (the stress suite says how much).
All rigging and skinning runs on the CPU (native kernels when installed).
