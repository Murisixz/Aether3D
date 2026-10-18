# Aether3D rigging: a guide for agents

Rig an unrigged character, bind and refine its weights, pose and animate it, verify the
deformation, and fix what the measurements show -- all as data. Rigging operations are
`aether3d.animation-script` operations (`run_script`; one undo step, all or nothing).
`inspect_rig` answers read-only questions and draws previews.

## 1. Look before you rig

```json
{"what": "landmarks", "meshes": ["Body"]}
```

`inspect_rig` returns landmarks (hips, knees, ankles, heels, shoulders, elbows, wrists,
neck, head...) in world space, each with a `confidence`, the `method` that placed it and
a `note`; the facing it measured; the pose (`A-pose`, `T-pose`, `arms down`); and
diagnostics (`FORWARD_AMBIGUOUS`, `HAND_NOT_FOUND`, `LEGS_FUSED`, `EXTRA_EXTREMITY`...).
A landmark from a `proportion` prior was not measured: check it, and pass a corrected
point in `landmarks` when you rig.

## 2. Rig and bind in one step

```json
{"format": "aether3d.animation-script", "version": 1, "operations": [
  {"op": "autorig", "meshes": ["Body", "Jacket", "Helmet"], "template": "humanoid",
   "parts": {"Helmet": "rigid"}, "landmarks": {"knee.L": [0.1, -0.02, 0.48]}}
]}
```

Templates: `humanoid`, `stylized` (tighter weight blends), `blocky` (Minecraft-style
blocks bound rigidly), `quadruped`. Parts: `geodesic` (default: weights from distances
inside the whole character, so clothing over a body takes the body's), `transfer` (the
first geodesic part's weights at the closest surface point), `rigid` (one joint for the
whole part), `attach` (a rigid attachment), `skip`. The result names the armature, the
joints, the control objects and the weights of every part.

What you get: joints named for the motion generators (Hips, Spine, Chest, Neck, Head,
Shoulder/UpperArm/ForeArm/Hand, Thigh/Shin/Foot/Toe with `.L`/`.R`, forearm twist
helpers) with `role`s; bone frames (+Y along the bone, +X the bend axis: a positive
rotation about +X bends elbows forward and knees back; a negative one raises a thigh or
an arm forward); swing-twist limits; IK targets for hands and feet, elbow and knee poles
and a gaze target (objects under the armature, IK influence 0: FK until you switch).

## 3. Animate

* `{"op": "generate", "armature": "Body Rig", "motion": "walk", "forward": "-y"}` --
  walk, run, jump, land, idle, wave, head_turn: ordinary keys.
* `{"op": "pose_character", "armature": "Body Rig", "time": 2.0, "commands": [
  {"do": "reach", "hand": "R", "target": [0.4, -0.5, 1.3]}, {"do": "look_at", "target":
  [0.4, -0.5, 1.3]}, {"do": "crouch", "depth": 0.25}]}` -- also `plant_feet`,
  `raise_arm`, `lean`, `aim_hips`, `rotate` (a joint, relative to its rest). Reports say
  whether a reach was possible and the IK residuals.
* `{"op": "ik_fk", "armature": "Body Rig", "limb": "leg_ik.L", "mode": "ik", "time":
  1.2}` -- switch at a time with a matched pose (the report's `mismatch` is the largest
  joint displacement, metres).
* Joint rotation channels (`joints["ForeArm.L"].rotation`) are **absolute** local
  rotations, rest included (as glTF): key `rest * delta`, or use `pose_character`
  `rotate`, which does that for you.

## 4. Verify, then fix

```json
{"what": "diagnose", "target": "Body Rig"}
```

Findings are measured (`basis: numeric`): `JOINT_COLLAPSE` (the skin around a bent joint
keeps less than 75 % of its size), `INTERPENETRATION`, `FLIPPED_TRIANGLES`,
`STRETCHED_EDGES`, `WEIGHT_LEAK`, `CROSS_SIDE_WEIGHTS`, `BIND_MISMATCH`,
`ASYMMETRIC_JOINTS`, `IK_FLIP`, `IK_UNREACHABLE`, `FOOT_SLIDING`... over a pose suite
(elbows and knees at 120 degrees, hips at 90, shoulders raised, wrists and torso twisted,
the head turned) and the timeline. Look at `{"what": "preview", "target": "Body Rig",
"mode": "stretch", "pose": "hip_flex_90.L"}` (red stretched, blue compressed) or `"mode":
"weights", "joint": "UpperArm.L"`.

Then `{"what": "fixes", "target": "Body Rig"}` lists proposals with a safety class, and

```json
{"op": "apply_rig_fix", "armature": "Body Rig", "fixes": "verify"}
```

applies every `safe` and `verify` proposal, one at a time, re-measuring after each and
undoing any that does not improve what it targets or makes anything worse (the result
lists `applied`, `reverted` and why). `approval` proposals (mirroring joints, reorienting
frames) are choices: pass them explicitly.

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
* `transfer_weights` for clothing; `generate_weights` to redo one mesh (`method`
  geodesic, envelope or nearest; `policy` organic, stylized, rigid).

## Rigid-part characters need no bones

A Minecraft-style character can stay a hierarchy of block objects animated by their own
transforms (`keyframe` on `transform.rotation_euler` of pivot objects). Nothing here is
required for that, and nothing here changes it. Bones are optional: `attach` (blocks
follow joints) or `autorig` with `template: "blocky"` (blocks skinned rigidly) when you
want generators, retargeting or IK.

## Honest limits

Automatic rigging measures geometry; it does not know anatomy beyond proportion priors
and cannot be right for every mesh: read the confidences. Weights are geodesic distances
with falloffs and smoothing, not a muscle simulation; deformation is linear, dual
quaternion or blended skinning plus correctives (morph targets driven by joint angles).
All rigging and skinning runs on the CPU (native kernels when installed).
