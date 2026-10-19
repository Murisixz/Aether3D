# Character rigging (0.19.1)

Automatic rigs for unrigged meshes, skin weights computed and edited as data, deformation
measured and fixed with verification, IK/FK controls that switch without a jump, high-level
pose commands, and -- when no skeleton is wanted -- Minecraft-style characters animated by
their own object transforms. Everything runs headless and is reachable from Python
(`scene.animation`), animation scripts (JSON), the command line (`aether3d rig ...`), scene
plans and MCP (`run_script`, `inspect_rig`, the guide `aether3d://rigging/guide`).

Stage 18 extends Stage 17's animation system ([ANIMATION.md](ANIMATION.md)): the same
`Rig`, joints, skins, solvers, timelines, clips and glTF -- nothing was rebuilt next to
them. The design record (what was inspected, the research behind each method, the
language of every subsystem and the outcome) is
[research/2026-10-stage18/DESIGN.md](../research/2026-10-stage18/DESIGN.md). **0.19.1**
revises it after measuring it against Blender: bone heat weights, an anatomical test
figure, landmarks measured on the mesh, a Rigify-like humanoid with fingers and split
twist bones, surface normals and optional delta mush, natural pose commands, stress
diagnostics with creases and folds, and more of the workflow on the command line
([research/2026-10-stage18-revision/DESIGN.md](../research/2026-10-stage18-revision/DESIGN.md);
[measured against Blender](#measured-against-blender)).

**What it is not.** Automatic rigging measures geometry: it finds limbs, joints and
symmetry in the mesh and places joints by measured sections and proportion priors; it
does not know anatomy beyond those priors and cannot be right for every mesh (every
landmark says how it was placed and how confident it is). Weights are a diffusion over
the surface (bone heat), not a muscle or tissue simulation. Deformation is linear, dual
quaternion or blended skinning, optional corrective smoothing (delta mush) and corrective
shapes driven by joint angles -- not physically based. Corrections are verified by
measurement, which is not the same as being right. All rigging, weighting and skinning
runs on the CPU (with native kernels when `aether3d-native` is installed); nothing here
runs on the GPU.

## Quick start

```python
from aether3d import Scene
from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

scene = Scene("Rigging")
body = scene.add_object("Body", mesh=implicit_surface(humanoid_shapes(), 0.02))
anim = scene.animation

# Look first: landmarks measured on the bare mesh, each with a confidence.
found = anim.detect_landmarks(body)
print(found["pose"], found["frame"]["forward"])  # A-pose [-0.0, -1.0, -0.0]

# Rig and bind in one undo step: skeleton, limits, IK controls, bone heat weights.
rig = anim.autorig(body, name="Hero")
print(len(rig["joints"]), rig["weights"]["Body"]["method"])  # 30 heat

# Animate: a generated walk (in place), then a pose command keyed on the controls.
anim.generate("Hero", "walk", forward="-y", travel=False)
reach = anim.pose_character(
    "Hero", [{"do": "reach", "hand": "R", "target": [-0.3, -0.45, 1.35]}], time=1.0
)
print(reach["commands"][0]["reachable"], reach["residuals"]["arm_ik.R"])  # True 0.0

# Back to FK with a matched pose: no joint moves.
switch = anim.ik_fk("Hero", "arm_ik.R", "fk", 1.5)
print(switch["mismatch"] < 1e-9)  # True

# Natural poses: a raised arm lifts its clavicle; a foot rolls on its ball.
pose = anim.pose_character(
    "Hero",
    [
        {"do": "raise_arm", "side": "L", "angle": 150, "direction": "side"},
        {"do": "roll_foot", "side": "L", "angle": 30},
    ],
    time=2.0,
)
print(pose["commands"][0]["clavicle_angle"], pose["commands"][1]["pivot"])  # 20.0 ball

# The stress suite: elbows and knees at 140 degrees, hips at 100, arms raised 150...
stress = anim.diagnose_rig("Hero", poses="stress", animation=False)
print(len(stress["poses"]), "new_creases" in stress["poses"]["torso_twist_60"]["Body"])  # 13 True

# Measure the deformation (no errors), and see what Stage 18 proposes to fix.
report = anim.diagnose_rig("Hero", animation=False)
print(report["summary"]["error"])  # 0
proposals = anim.propose_rig_fix("Hero", animation=False)["proposals"]
print(all(p["safety"] in ("safe", "verify", "approval") for p in proposals))  # True
```

## Three workflows, chosen by the user

| Workflow | What moves the character | Use it for |
| --- | --- | --- |
| **Rigid transforms** (`generate_rigid`) | The part objects' own transforms; no armature, skin or weights | Minecraft-style segmented characters, props |
| **Bone-attached rigid parts** (Stage 17's `attach`, `autorig(template="blocky")`) | Joints; each part follows its joint rigidly | Blocky characters that want IK, retargeting, constraints |
| **Skinned rigs** (`autorig`, `skin`) | Joints; meshes deform by weights | Organic characters, clothing |

Nothing converts a character from one to another unless asked (`convert_character`).
Stage 17's blocky characters are unchanged: a fixture computed by the v0.18.0 sources
(`tests/data/v0.18.0/nova_motion.json`) pins their generated motions' part matrices.

## The rig document

A rig is Stage 17's immutable `Rig` with Stage 18 fields (file format 12 since 0.19.1;
older files load unchanged):

* joints -- as before (`name`, `parent`, rest `position`/`rotation`/`scale`, `tail`,
  `limits`) plus `role` (a semantic role: `hips`, `spine`, `chest`, `neck`, `head`,
  `shoulder.L`, `upper_arm.L`, `lower_arm.L`, `hand.L`, `thigh.L`, `shin.L`, `foot.L`,
  `toe.L`, ...; unique per rig; generators and pose commands use it first), `deform`
  (`false`: a helper joint -- IK, twist sources -- that never receives weights) and `span`
  (`[start, end]`, default `[0, 1]`: the part of the bone, head to tail, whose segment
  heat weights use -- a limb split in two deforming bones, as Rigify's, gives each its
  half);
* `controls` -- `{name, kind, object | joint, constraint, group, side, shape, size,
  mirror}`, kinds `ik_target`, `pole`, `gaze`, `fk`, `root`, `custom` (control objects
  are ordinary empties under the armature, keyed like any object);
* `correctives` -- corrective shapes: a morph target's weight driven by a joint's angle
  (`driver`: `angle`, `swing` (with an optional signed `direction`), `twist` or
  `rotation` about `axis`; `range` degrees mapped through `curve` `linear`/`smooth` to
  `weight`); evaluated after the joints, before the skin;
* `landmarks`, `template`, `info` (`forward`, `up`, `source`, `style`, `pose`) -- what
  the auto-rig measured, for diagnostics and for re-rigging;
* `enforce_limits` -- `"ik"` (limits apply to joints IK moves, Stage 17's rule; the
  default) or `"always"`.

**Limits.** Stage 17's Euler bounds (`{"rotation": {"min", "max"}}`, absolute local
angles) still work. Stage 18 adds forms relative to the rest pose, decomposed into swing
and twist about the bone axis (`axis`, default `+y`; quaternions canonicalized `w >= 0`, so
no wraparound or sign flips):

```json
{"swing": {"cone": 60}, "twist": [-40, 40]}
{"axis": "+y", "swing": {"x": [-5, 150], "z": [-5, 5]}, "twist": [-80, 80]}
{"hinge": {"axis": "+x", "min": -5, "max": 145, "play": 4}}
```

A cone or an ellipse (`x`, `z` ranges) bounds the swing, `twist` the rotation about the
bone; a hinge allows rotation about one axis (with `play` degrees of other swing).
Clamping keeps the closest allowed rotation; `limits.angles` reports a joint's swing,
twist and swing vector.

**Twist constraints** (`{"type": "twist", "joint", "target": {"joint"}, "axis",
"factor"}`) turn a helper joint by a fraction of another joint's twist (the auto-rig's
split upper arms, forearms and thighs: the half nearer the trunk turns back by half the
limb's twist).

**Skins** keep, besides their weights and inverse binds, `method` (`linear`,
`dual_quaternion`, `blended` with `dq`), `normals` (`joints`: the rest normals turned by
the blended joint matrices, glTF's rule; `surface`: turned as the deformed surface turns,
the default for new skins) and `smooth` (corrective smoothing: `null`, or `{iterations,
strength, region, weights}`; see [Deformation](#deformation)). Joint constraints are
evaluated in dependency order (a constraint after those that write what it reads).

## Automatic rigging

```python
rep = anim.autorig(
    [body, jacket, helmet],
    name="Hero",
    template="humanoid",
    parts={"Jacket": "transfer", "Helmet": "attach"},
    landmarks={"knee.L": [0.09, -0.02, 0.48]},
)
```

**Landmarks** (`detect_landmarks`, read-only). The meshes are voxelized together (the
auto-rig's: the body's, see **Parts** below; 128 cells over the height; open meshes
closed morphologically); the canonical frame comes
from the up axis, the best mirror plane (a sweep of plane angles and offsets scored by
the solid voxels it maps onto solid voxels) and the feet's facing. Extremities are local
maxima of the geodesic distance (inside the solid) from the body's core; limbs are the
branches of the Reeb graph of that distance; an eroded copy of the solid separates
touching limbs. Joints are then **measured on the exact mesh** (0.19.1): ray probes from
inside (how far the surface is; only a ray that leaves the body counts), sections across
the body (the midline's front and back, the spine set toward the back), limb axes and
their radius profiles; the hips from the crotch (measured by a ray up between the legs:
the voxels part nearly touching thighs lower), the buttock's crest and the trochanter,
the knees and elbows from the limb's bend and its narrowest section, the shoulders by
where the arm's section leaves the torso (an armpit rule calibrated on known joints),
the wrists at the narrowest section between the palm and the forearm, the neck from the
head's chain; **fingers** (knuckles, middle and last joints, tips) from the surface
distance along the hand: tips are its maxima, each finger's axis the centroids of rings
of equal distance from its tip. A symmetric body's two hands must find the same fingers
within 1.5 % of the height, or neither gets any (`FINGERS_UNCERTAIN`: the mesh is too
coarse for them; measured: good detections agree within 0.5 %, bad ones disagreed by 1.6
to 7.5 %); so neither does a hand whose pair was not found (a hand against a thigh), nor
one whose middle finger's tip lies more than 3 % of the height from the hand's tip
(fused fingers named one off: 4 %; resolved hands, at most 1.9 %). Where a mesh shows nothing, joints come from anthropometric fractions of the
height (Drillis & Contini). Every landmark reports `confidence`, the `method` that placed
it and a `note`; `diagnostics` name problems (`DETACHED_PARTS`, `FORWARD_AMBIGUOUS`,
`ASYMMETRIC_BODY`, `EXTRA_EXTREMITY`, `HAND_NOT_FOUND`, `FOOT_NOT_FOUND`,
`HEAD_NOT_FOUND`, `LEGS_FUSED`, `NECK_NOT_FOUND`, `HIP_NOT_MEASURED`, `ARM_AGAINST_BODY`,
`SHOULDER_NOT_MEASURED`, `FINGERS_UNCERTAIN`...). Pass corrected points as `landmarks`
(world space) to override any of them. On the anatomical figure (`aether3d.sculpt.figure`:
nine builds -- slim, heavy, child, stylized, mittens -- in A-, T- and arms-down poses, at
three mesh resolutions; its joint centres are known) the major joints fall within 2.6 %
of the height (at most 2.52 %) and the fingers within 2.3 % or are left out
(`tests/test_rig_quality.py`).

**Templates.** `humanoid` and `stylized`: Hips, Spine, Chest, UpperChest, Neck, Head
(the torso's joints along the measured back); Pelvis.L/R (from the root to the hip's
crest: the flank stays on the hips, Rigify's `DEF-pelvis`); Shoulder (the clavicle, from
the upper chest), UpperArm and UpperArmTwist, ForeArm and ForeArmTwist (each limb bone
split in two deforming halves, Rigify's split `DEF` bones), Hand, and when the hands show
them three joints per finger and thumb; Thigh and ThighTwist, Shin, Foot, Toe -- about 60
joints with fingers, 30 without; sides `.L`/`.R`; roles. `blocky` (Hips, Body, Head,
Arm.L/R, Leg.L/R; separate block objects are attached, a single mesh is skinned rigidly
block by block), `quadruped` (Pelvis, Spine, Chest, Neck, Head, Tail, four three-joint
legs). Bone frames (`orientation="bone"`): +Y along the bone, +X its bend axis -- a
positive rotation about +X bends elbows forward, knees back and fingers into the palm; a
negative one raises an arm or a thigh forward. Limits are swing-twist cones, ellipses and
hinges in those frames, the elbow's and knee's flexion range measured from their rest
bend, the fingers' knuckles and hinges set from anatomy. Controls: IK targets for hands
and feet, elbow and knee poles, a gaze target, two-bone IK constraints at influence 0
(the rig starts in FK) with `bend_axis`, twist constraints for the split bones.

**Parts** (`parts={mesh: mode}`): `generate` (the default: weights by `weights.method`,
bone heat unless asked otherwise; `heat` or `geodesic` choose the method for one part --
geodesic parts are voxelized together, so clothing over a body takes the body's
distances), `transfer` (each vertex takes the weights of the nearest point of the first
generated part's surface), `rigid` (the whole part follows its dominant joint), `attach`
(a rigid attachment to the bone nearest its centre -- of bones meeting there, the one
nearest the root, so mirrored pieces attach alike -- parented to the armature where it is
so the character can then be placed or turned as a whole: no skin, no deformation cost),
`skip`. The landmarks are measured on the body -- the generated parts (and a blocky
character's blocks, attached by default) -- so clothing and pieces bound any other way
leave the joints exactly where the bare body puts them; the report's `measured` names the
meshes. (Measured with the body, as up to 0.19.0, the outfit showcase's helmet raised
the skull's pivot 8 cm, its trousers lowered the hips 6 cm and its gloves hid the
fingers.)

## Skin weights

**Generated** (`generate_weights(mesh, armature, method=, policy=, ...)`):

* `heat` (the default since 0.19.1) -- **bone heat** (Baran and Popovic 2007; Blender's
  Automatic Weights): each deforming bone's segment (its `span`) heats the vertices it is
  nearest to and *visible* from (a ray from the vertex to the bone that does not enter
  the surface again, by Aether3D's watertight traversal), and the heat diffuses over the
  surface: one symmetric system `(L + D) w = D p` for all bones -- `L` the cotangent
  Laplacian, `D` each vertex's heat -- factorized once by the nested-dissection sparse
  LDL^T (with its native twin) and solved per bone. Up to **eight influences** per vertex
  (capped at four, a twisted torso's weights jump where a fifth joint takes over: 114 new
  creases and 23 folds measured, none with eight -- Blender gives 12 % of a figure's
  vertices more than four). Left and right joints fade out across the mirror plane (heat
  diffuses through the groin and the sternum). On a mesh and bones both were given, the
  dominant joint agrees with Blender's on 99.8 % of the vertices (mean difference 0.0013;
  `tests/test_rig_quality.py` checks it against a committed Blender result). `diffusion`
  scales `L` (the stylized policy blends tighter), `visibility` (on) can be turned off.
* `geodesic` -- Dionne and de Lasa's geodesic voxel binding (v0.19.0's default): the mesh
  and the bones voxelized, distances inside the solid by an integer 3-4-5 chamfer metric
  in Dial buckets (a native twin with the same results), weights `d^-p` over the bones
  within `ratio` times the nearest; the auto-rig adds body-part regions and spine
  smoothing to it.
* `envelope` (Stage 17's) or `nearest`.

Policies: `organic`, `stylized`, `rigid`. Safeguards, each reported: bones farther than
`hops` joints from the nearest bone take no weight (no leg on the other leg), sided bones
fade out across the mid-plane (`sides`), parts made of one block follow one joint
(`rigid_parts`). Smoothing never crosses mesh components. `max_seconds` bounds the work;
the report gives the technique, timings and coverage. New skins get surface-following
normals (`normals`) and, when asked, corrective smoothing (`corrective`).

**Edited** (`edit_weights(mesh, [operations])`, one undo step): `assign`, `add`,
`subtract`, `scale`, `replace` (from one joint to another), `normalize`, `prune`, `cap`
(at most N influences), `smooth`, `mirror`, `rigid`, `lock`, `unlock` -- on selections
(`all`, `vertices`, `sphere`, `box`, `plane`, `joint`, `dominant`, `role`, `component`,
`component_of`, `near_joint`, `locked`, and `and`/`or`/`not` of them) with brush falloffs
(`center` or `joint`, `radius`, `curve` `constant`/`linear`/`smooth`/`sharp`/`root`/
`sphere`, `distance` `euclidean` or `geodesic`). Locked joints and vertices keep their
weights; rows an operation does not touch stay bit for bit.

**Transferred** (`transfer_weights(target, source, max_distance=, influences=,
unmatched="rigid", smooth=2)`): nearest surface points of the source (barycentric
weights of its triangle; as many influences as the source's skin by default), per-part
distance statistics, and the parts too far to trust reported (and bound rigidly, or
left).

**Inspected** (`inspect_weights(mesh, joint=)`): validity, coverage, influence counts and
`WEIGHTS_NOT_NORMALIZED`, `WEIGHTS_INVALID`, `WEIGHT_LEAK`, `CROSS_SIDE_WEIGHTS`,
`HELPER_JOINT_WEIGHTED`, `UNUSED_JOINTS`.

## Deformation

`skin(mesh, armature, method=..., normals=..., smooth=...)`: `linear` (Stage 17),
`dual_quaternion` (Kavan et al.: per-joint palettes split into a rotation and a
translation by a polar decomposition, scale and shear kept in a separate matrix so
non-rigid joints still work) or `blended` (per vertex, `dq` gives the dual quaternion
share 0..1; 0 is linear blend skinning bit for bit, 1 dual quaternion skinning bit for
bit). A native kernel computes all three, bitwise equal to the NumPy reference
(`tests/test_native.py`). Split twist bones spread a limb's twist along it; corrective
shapes add volume or shape where a joint bends.

**Normals.** `joints` -- the rest normals transformed by the blended joint matrices (the
cofactor matrix: correct under non-uniform scale), as glTF viewers do; `surface` (the
default for new skins) -- each rest normal turned by the smallest rotation that takes its
rest face normal (area-weighted) to the deformed one: shading follows the deformed surface
(no dark patches where weights change quickly under a twist), exactly the rest normals at
rest. A native kernel computes them, bitwise equal to the NumPy reference (0.95 ms a frame
against 8.3 ms for 41,000 vertices).

**Corrective smoothing** (`smooth`, delta mush: Mancewicz et al. 2014; the principle of
Blender's Corrective Smooth). At bind, the rest surface is smoothed and each vertex's
offset from the smoothed surface (its detail) is stored in the frames of its triangle
corners; each frame, the skinned surface is smoothed the same way and the detail added
back in the deformed corners' frames, averaged by the corners' weights (the sine of their
angle), then blended by a mask: full where weights blend, none where one joint holds a
vertex (`region: "blend"`; `"all"` everywhere). Smoothing uses the rest surface's
cotangent weights (`weights: "cotangent"`; `"uniform"` drifts an irregular mesh's
vertices along the surface, and the large tangential detail it leaves scatters vertices
at bent joints -- measured), `iterations` (default 8) of strength 0.5. The modeled mesh at
rest; a native kernel, bitwise equal to the NumPy reference (9.6 ms a frame against 41 ms
for 41,000 vertices). On heat weights it lowers the stretch at bent joints a little and
leaves creases and folds about as they are; on skins capped at four influences (most
imported ones) it removes a twisted torso's creases. glTF cannot carry it
(`SKIN_SMOOTHING_LOSSY`), so it is opt-in: `corrective: true` (or a spec) when weights are
generated, `smooth=` on `skin`, or the `smooth_skin` fix.

## Structural editing

`edit_rig(armature, [operations], skins="rebind", animation="keep", channels="keep")` --
`add_joint`, `remove_joint`, `move_joint`, `orient_joint`, `rename_joint`,
`reparent_joint`, `insert_joint`, `split_bone`, `mirror_joints`, `align_bone`,
`set_joint` -- one undo step, all or nothing, `dry_run` to see first. Skins are rebound
so the mesh stays where it is (`J_new^-1 J_old B`); renamed joints are renamed in clips,
masks, constraints and controls; changed joint frames can rewrite their animation
(`animation="convert"`: quaternion tracks exactly, curves refitted within `tolerance`).
Single-operation shortcuts: `add_joint`, `remove_joint`, `move_joint`, `orient_joints`,
`rename_joint`, `reparent_joint`, `mirror_joint_chain`.

## Controls, IK/FK and posing

`add_control(armature, name, kind, ...)` gives any rig -- auto-rigged or hand-built,
Stage 17's included -- an IK limb (target, pole, constraint at influence 0), a gaze, an FK
control or the root.

`ik_fk(armature, limb, "ik" | "fk", time)` switches with **matched poses**: to IK, the
target and pole are keyed where the FK pose puts the hand or foot and its bend, then the
influence keyed to 1; to FK, the joints are keyed with the IK solution's rotations, then
the influence keyed to 0. The report measures the switch (`mismatch`: the largest joint
displacement; `rotation_mismatch`: the largest joint turn). IK bends a limb about its bend
axis only as far as the middle joint's limits require: a relaxed limb keeps its roll, so
switching it changes nothing visible (`tests/test_rig_controls.py` checks every joint,
twist helpers included, at both switches).

`pose_character(armature, [commands], time=)` turns intentions into keys: `reach` (a
hand to a point or object; leans the chest when out of reach and says so), `plant_feet`
(where the feet are now), `look_at`, `lean`, `crouch` (feet planted, hips lowered),
`aim_hips`, `rotate` (a joint, relative to its rest), and (0.19.1):

* `raise_arm` (`side`, `angle`, `direction` `forward` or `side`, `shoulder`) -- the arm
  points exactly where asked (its armature-space direction, whatever the clavicle does)
  while the clavicle elevates with it: nothing below 30 degrees, then 1 degree for every 6
  (25 at 180), a simple model of the shoulder girdle's share of an arm's elevation;
  `shoulder` overrides it (a fraction, 0 for none);
* `curl_fingers` (`side`, `shape` `curl` | `fist` | `point`, `amount` 0..1, `fingers`,
  `spread` degrees) -- the finger joints turned about their bend axes (into the palm), the
  thumb its own way; `ROLE_MISSING` on a hand without finger joints;
* `roll_foot` (`side`, `angle`, `break` 35, `ground`) -- Rigify's foot roll as keys: a
  negative angle turns the foot about its heel (the toes come up); up to `break` about its
  ball (the heel comes up, the toe joint turned back so the toes stay flat); beyond, about
  the toe tip (foot and toes together). The foot is first set flat where it stands (on
  the heel, ball or toe tip lifted least), so a roll is absolute: the same command gives
  the same pose whatever rolls were keyed before. The pivots are the heel, ball and toe
  tip landmarks the foot carries; the leg's IK is switched on.

Each reports reachability, IK residuals (measured after limits) and limits hit. Joint
rotation channels are absolute local rotations, rest included (as glTF and Stage 17):
commands compose `rest * delta`.

## Diagnostics and verified correction

`diagnose_rig(armature, poses="suite", animation=True)` measures the rig and its skins:
the skeleton (`ZERO_LENGTH_BONE`, `BONE_AXIS_MISALIGNED`, `ASYMMETRIC_JOINTS`,
`POLE_ON_LINE`), the weights (`inspect_weights`' codes, `BIND_MISMATCH`), the deformation
in a pose suite and the timeline (`IK_FLIP`, `IK_UNREACHABLE`, `FOOT_SLIDING`). Suites:
`"suite"` (the everyday range: elbows and knees bent 120 degrees, hips flexed 90,
shoulders raised 90, wrists twisted 90, the torso twisted 45, the spine bent 45, the head
turned 60) and `"stress"` (0.19.1, the extremes: elbows and knees 140, hips 100,
shoulders raised 150 with the clavicle's share, the torso twisted 60). Per pose and mesh:
`joint_ring` (the mean distance of the skin around the joint to its pivot, posed over
rest: below 0.75 is `JOINT_COLLAPSE`), `new_creases` and `sharpest_crease` (edges between
triangles of typical size, smooth at rest, bent past 60 degrees: `SKIN_CREASE` from five),
`folded_triangles` (triangles turned against the area-weighted normal of their posed
neighbourhood: `SKIN_FOLD`) -- both from the geometry alone, whatever normals the skin
carries -- `flipped_triangles` (turned against the skin's normals; creases of the modeled
surface are not counted), stretched and compressed edges and the worst stretch, the
volume change and `INTERPENETRATION`. Poses can be data:

```json
{"name": "deep_knee", "joint": "Shin.L",
 "turns": [{"joint": "Thigh.L", "axis": "+x", "angle": -60},
           {"joint": "Shin.L", "axis": "+x", "angle": 145}]}
```

(a string axis is in the joint's own frame, a vector in armature space). Every finding is
measured (`basis: numeric`); how a pose should look is not judged.

`propose_rig_fix` turns findings into proposals with a safety class -- `safe`
(normalizing, moving a helper's weights to its deforming parent, moving a pole off the
chain's line), `verify` (giving leaking vertices back to their dominant joint, smoothing,
rebinding, blending dual quaternion skinning around a collapsing joint -- the smallest
share that resolves it, measured), `approval` (mirroring joints: which side is right is
a choice; corrective smoothing, `smooth_skin`, for creases, folds and turned triangles:
fewer in most poses, sometimes a few more at the hardest bends -- a trade-off).
`apply_rig_fix(armature, fixes | "safe" | "verify", poses=, animation=, dry_run=)`
applies them one at a time, measures again what each can change, and undoes any `verify`
fix that did not improve its target or made anything worse; all that is kept is one undo
step, and the report lists `applied`, `reverted` and why, and `changes`: per pose and mesh,
each measure (kept size, creases, folds, flips, worst stretch) before and after -- run an
`approval` fix with `dry_run` first to see its trade-off.

`rig_preview(armature, mode=..., view=...)` draws `pose` (a time or a diagnostic pose),
`weights` (a joint's weights as a heat ramp), `stretch` (each vertex's mean edge-length
change: grey kept, red stretched, blue compressed, full colour at 1.41x and 0.71x) or
`skeleton` (bones, joints, IK targets and poles over the mesh, with an SVG), lit by key,
fill and rim lights in a separate preview scene, from the `front`, `back`, `side` (the
character's right), `left` or `three_quarter`.

**The feedback loop** ([FEEDBACK.md](FEEDBACK.md)): a `rig` expectation check
(`objects`, `poses`, `timeline`, `ignore`) reports the measured findings, planned into
`rig_fix` corrections (the rig's own proposals, checked as data and verified by the
feedback session as well; they need the policy's `allow_animation`).

## Rigid-part characters

```python
from aether3d import Scene

scene = Scene("Blocks")
hero = scene.add_object("Hero")


def block(name, size, center, parent, position):
    part = scene.add_object(name, parent=parent, position=position)
    with part.edit_mesh() as m:
        m.add_box(size, center=center)
    return part


torso = block("Torso", (0.5, 0.25, 0.75), (0, 0, 0), hero, (0, 0, 1.125))
block("Head", 0.5, (0, 0, 0), torso, (0, 0, 0.625))
block("Left Arm", (0.25, 0.25, 0.75), (0, 0, -0.3), torso, (0.375, 0, 0.3))
block("Right Arm", (0.25, 0.25, 0.75), (0, 0, -0.3), torso, (-0.375, 0, 0.3))
block("Left Leg", (0.25, 0.25, 0.75), (0, 0, -0.375), hero, (0.125, 0, 0.75))
block("Right Leg", (0.25, 0.25, 0.75), (0, 0, -0.375), hero, (-0.125, 0, 0.75))

walk = scene.animation.generate_rigid(hero, "walk")
print(walk["armature"], walk["verification"]["position"] < 0.002)  # None True
print(any(scene.animation.rig(o) is not None for o in scene.objects))  # False
```

`rigid_parts(root)` finds the parts (by name: head; torso, body or chest; arm; leg -- or
`parts={role: object}`), measures each side (left is `up x forward`; a contradicting name
is reported `SIDE_NAME_MISMATCH`) and each pivot (the shoulder half an arm's width below
the arm's top, the hip at the leg's top, the neck under the head, the waist under the
torso -- or the part's origin when it is already there). `generate_rigid(root, kind)`
runs the Stage 17 generators unchanged on a skeleton through those pivots in a private
scratch scene and writes each part's exact motion as ordinary `transform` clips (keys
chosen until every 240 Hz sample is within `tolerance`), with travel on the root; it
verifies the parts in the scene against the motion. An in-place motion stops a previous
generated travel; the user's own clips stay.

`convert_character(source, "skeleton" | "rigid")` converts on request: a rigid hierarchy
to an armature with attached copies of its parts (joints with roles, the motion baked onto
them), or an armature's attached parts to a hierarchy of pivots (the motion baked onto
them) -- verified sample by sample, the source untouched (hidden by default), one undo
step.

## Scripts, command line, MCP

Animation scripts (`aether3d.animation-script`) take every rigging operation as data:
`autorig`, `edit_rig` and its single-joint operations, `generate_weights`,
`edit_weights`, `normalize_weights`, `transfer_weights`, `add_control`, `ik_fk`,
`pose_character`, `apply_rig_fix`, `generate_rigid`, `convert_character`; read-only
`detect_landmarks`, `inspect_weights`, `diagnose_rig`, `propose_rig_fix`, `rigid_parts`.
Arguments are allowlisted and checked; a failing script changes nothing.

```json
{"format": "aether3d.animation-script", "version": 1, "label": "Rig the hero",
 "operations": [
  {"op": "autorig", "meshes": ["Body"], "name": "Hero"},
  {"op": "generate", "armature": "Hero", "motion": "walk", "forward": "-y"},
  {"op": "pose_character", "armature": "Hero", "time": 1.0,
   "commands": [{"do": "look_at", "target": [0.5, -3.0, 1.6]}]},
  {"op": "diagnose_rig", "armature": "Hero", "poses": ["knee_bend_120.L"], "animation": false}
 ]}
```

Command line: `aether3d rig autorig PROJECT --mesh Body`, `rig info`, `rig landmarks`,
`rig weights`, `rig diagnose`, `rig test --armature A [--suite stress|suite] [--sheet
stress.png]` (per pose: kept size, creases, folds, stretch; a labelled stretch map per
pose; exit code 3 when a finding is an error), `rig fix --safety verify`, `rig create
--armature A --rig RIG.json`, `rig edit --armature A --op JSON`, `rig bind --mesh M
--armature A [--method heat|geodesic|envelope|nearest] [--smooth N] [--normals
surface|joints]`, `rig transfer --mesh JACKET --source BODY`, `rig pose --armature A
--command JSON [--time T]`, `rig ikfk --armature A --limb arm_ik.L --mode ik --time T`,
`rig preview OUT.png --mode weights --joint UpperArm.L` (`.svg`: the skeleton), `rig parts
--root Hero`, `rig convert --source Hero --to skeleton`, `rig guide`; rigging scripts run
with `aether3d animation run`. Each command that changes the project is one undo step
and saves it (`--dry-run` to see first).

MCP: rigging changes go through `run_script`; `inspect_rig` answers read-only questions
(`what`: `summary`, `landmarks`, `weights`, `diagnose` -- `suite`: `suite` or `stress` --,
`fixes`, `preview` -- a PNG -- and `parts`); the guide is the resource
`aether3d://rigging/guide`.

## glTF

Exported: joints (with their roles, helper flags, tails and limits as Aether3D extras on
the joint nodes, read back on import), skins with their weights and inverse binds (every
mesh of a character, each its own skin), attachments as children of joint nodes,
animations (clips, or baked: what the scene shows -- constraints, IK and corrective
shapes' weights included); up to eight influences per vertex in two joint sets
(`JOINTS_0`/`JOINTS_1`). Reported: skins other than linear blend
(`SKINNING_METHOD_LOSSY`: glTF skins are linear blend; the same weights bend with less
volume -- a lossy export needs `allow_lossy`), corrective smoothing
(`SKIN_SMOOTHING_LOSSY`: exported with the same weights, its folds and pinches come back;
lossy), surface normals (`SKIN_NORMALS_DIFFER`, information: glTF viewers carry the rest
normals with the joints; the geometry is the same), and `RIG_AUTHORING_DATA` (what is
kept only as extras -- joint spans among them; control definitions, corrective drivers,
landmarks and constraints are not exported: their effect is baked). The round trip is
tested numerically on a body, a jacket and a helmet through a validated GLB
(`tests/test_rig_gltf.py`).

## Measured against Blender

Blender 5.2.2 LTS (headless `bpy`) was the reference of the 0.19.1 revision; it is a
development-time tool only (nothing in the package imports or runs it, no Blender code is
copied, every image is Aether3D's). The experiments, their scripts and their results are
in [research/2026-10-stage18-revision/blender/](../research/2026-10-stage18-revision/blender/).
On the anatomical figure (42,878 vertices) under the 13 stress poses, every binding
deformed and measured by the same geometry-only code -- totals over the suite, and the
share of their size the bent elbows' and knees' skin keeps:

| Binding | New creases | Folds | Worst stretch | Elbows and knees keep |
| --- | --- | --- | --- | --- |
| v0.19.0 (its own rig and weights for this mesh) | 280 | 16 | 5.88 | 0.73-0.74 |
| 0.19.1: heat weights, LBS (the default) | 158 | 10 | 4.24 | 0.87-0.89 |
| 0.19.1: heat weights + delta mush (8 iterations) | 159 | 12 | 3.78 | 0.87-0.89 |
| Blender 5.2 Automatic Weights, LBS | 136 | 6 | 4.24 | 0.87-0.89 |
| Blender, DQS (Preserve Volume) | 135 | 8 | 6.38 | 1.00 |
| Blender, LBS + Corrective Smooth (defaults) | 150 | 11 | 3.72 | 0.87-0.89 |
| Blender, LBS + Corrective Smooth (12 iterations) | 175 | 11 | 3.59 | 0.86-0.89 |

Blender's weights skinned by Aether3D equal Blender's own LBS within 3e-6 m; Aether3D's
heat weights and Blender's agree on the dominant joint of 99.97 % of the vertices. The
remaining difference is at the hips (a hip flexed 100 degrees: 43-44 new creases against
Blender's 30-37).
Showcase B renders these bindings side by side.

## Showcases

Seven scripts, `examples/rigging_*.py`, model, rig, animate and render everything with
Aether3D (FFmpeg only encodes the rendered frames); each writes a `report.json` of what it
measured. The character is the anatomical figure (`aether3d.sculpt.figure`), remeshed to
even 7 mm triangles (42,878 vertices; fingers rigged); B's v0.19.0 and Blender columns
are data computed for that exact mesh (`examples/data/rigging/`, its hash checked).
Previews, sheets, heatmaps, preview videos and reports are committed in
`examples/output/rigging/`; the 1080p videos, lossless masters, frame sequences,
full-resolution stills and the skinned characters' scenes and GLBs are regenerated by
running the scripts (`--quick --backend cpu` runs each in seconds; the tests do).

| | Script | Shows | Measured (full quality) |
| --- | --- | --- | --- |
| A | `rigging_autorig.py` | the anatomical figure as a bare mesh: landmarks (fingers included), the auto-rig (60 joints), bone heat weights, checks, clothes along lines cut into the mesh from its landmarks; walk, run, crouch, reach, and a rise onto tiptoe with the arms raised in a V (the clavicles lifting) and the hands closing into fists | auto-rig 2.5 s; no error-severity finding; feet slide at most 1.6 mm; the crouch's feet stay planted (residual 0.00 mm) and the reach's hand on its target (residual 0.00 mm); IK/FK switches move no joint (3e-16 m); the rise rolls the feet onto their toes (ball+toe) with the clavicles at 18.3 degrees and 30 finger joints keyed |
| B | `rigging_deformation.py` | the stress suite's hardest poses -- a shoulder raised 150 degrees, a hip flexed 100, a wrist twisted 90, a knee and an elbow bent 140, the torso twisted 60 -- for v0.19.0's rig and weights of the same mesh, the revision's heat weights, with delta mush, and Blender's Automatic Weights, side by side; stretch maps | over the 13 stress poses: v0.19.0 280 new creases, 16 folds, worst stretch 5.88; heat 158, 10, 4.24; Blender 136, 6, 4.24 (each tile's label gives its pose's numbers) |
| C | `rigging_ikfk.py` | both hands and both feet on targets (one foot steps onto a block), FK to IK and back, then FK keys | every hand and foot on its target at both keyframes and all along the IK interval (at most 4e-16 m); the switches move no joint (7e-16 m) and no skin beyond the idle's own motion (0.04 mm); the feet stay where they were after the switch (0 m) |
| D | `rigging_outfit.py` | the figure with a jacket, trousers, boots and gloves over its parts (weight transfer), a helmet, shoulder plates and a backpack (attached) walking and running | the skeleton measured on the body alone (60 joints, the fingers under the gloves; the helmet on `Head`, the plates on `Shoulder.L`/`Shoulder.R`); clothing keeps 0.1-0.6 mm from the skin at its closest; the feet slide at most 1.4 mm; the GLB passes the Khronos validator (0 errors, 0 warnings) |
| E | `rigging_minecraft_rigid.py` | Kit, an original pixel-skin character of six solid parts, animated by object transforms only: idle, walk, run, jump, wave (unchanged by the revision) | no rig and no skin in the scene; parts within 0.34 mm and 0.05 degrees of the generated motion; 0.68 ms per frame (0.91 ms as bone-attached blocks) |
| F | `rigging_world.py` | the auto-rigged figure and Kit on a road through a procedural world, a crane, a tracking and a ground-level shot | the figure's feet slide at most 1.4 mm; the crane stays at least 1.41 m above the ground |
| G | `rigging_roundtrip.py` | the figure with a jacket and a helmet through `.aether` and GLB | `.aether` exact (every joint and posed vertex bit for bit); GLB: same joints, parents, roles, limits and spans, eight influences, rest exact, posed skins within 0.06 mm, validator clean |

![Showcase A: the auto-rigged figure walking, running, crouching, reaching and rising onto tiptoe](../examples/output/rigging/a_autorig/motion_sheet.png)

![Showcase B: six stress poses -- v0.19.0, heat weights, heat with delta mush, Blender's Automatic Weights](../examples/output/rigging/b_deformation/deformation_grid.png)

![Showcase C: IK/FK controls](../examples/output/rigging/c_ikfk/ikfk_sheet.png)

![Showcase E: a rigid-part character with no bones](../examples/output/rigging/e_rigid/kit_sheet.png)

![Showcase F: both kinds of character in a procedural world](../examples/output/rigging/f_world/world_sheet.png)

The GLB copy in G is lit differently because glTF carries no studio world.

**Renders and encodes.** Video frames: the GPU rasterizer at 1920x1080 with cascaded
shadow maps and 4x MSAA, key, fill and rim lights; every frame rendered at a true 60 fps
(no frame repeated). Encodes: libx264 `-preset slow -crf 10`, yuv420p, BT.709,
faststart; FFV1 lossless masters; 960x540 previews (CRF 23); PSNR/SSIM measured against
the frames, frame by frame. Stills and B's tiles: the GPU path tracer (A 256, D and E 192,
B 96 samples per pixel). On the development machine (RTX 4060 Laptop GPU):

| | Frames (60 fps) | Duration | Frame rendering | Stills | 1080p MP4 | PSNR / SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| A | 900 | 15.0 s | 330.0 s (0.37 s a frame) | 5 path traced, 395 s | 5.3 MB | 42.8 dB / 0.996 |
| B | -- | -- | -- | 24 tiles 720x720, 209 s | -- | -- |
| C | 300 | 5.0 s | 118.7 s (0.40 s a frame) | 2 switch views | 1.3 MB | 42.7 dB / 0.996 |
| D | 324 | 5.4 s | 119.3 s (0.37 s a frame) | 2 path traced | 3.1 MB | 43.4 dB / 0.996 |
| E | 720 | 12.0 s | 239.8 s (0.33 s a frame) | 5 path traced, 152 s | 22.5 MB | 38.8 dB / 0.984 |
| F | 540 | 9.0 s | 311.8 s (0.58 s a frame) | 3 rasterized | 18.8 MB | 41.8 dB / 0.987 |
| G | -- | -- | -- | 9 views + a difference map | -- | -- |

## Performance

Measured with `benchmarks/bench_rigging.py` (results in
`benchmarks/results/stage18-revision-rigging.json`, a 12th Gen Intel Core i9-12900H
laptop; 0.19.0's in `stage18-rigging.json`): auto-rigging a 6,488-vertex body 0.42 s and
an 18,008-vertex one 0.77 s (0.19.0: 0.47 and 0.67); the anatomical figure (41,080
vertices, 60 joints) 1.92 s, of which bone heat 1.08 s (the sparse solve 0.77 s; geodesic
weights 0.77 s); per frame at 41,080 vertices, delta mush 9.6 ms with the native kernel
(NumPy 41 ms) and surface-following normals 0.95 ms (NumPy 8.3 ms); the stress suite (13
poses) 1.4 s; dual quaternion skinning of a million vertices 370 ms in NumPy, 61 ms with
the native kernel on one thread and 15 ms on 20; 50 auto-rigged walking characters
(6,488 vertices each, eight influences, surface normals, IK constraints, six twist
constraints, gaze) evaluated in 33.9 ms per frame (0.19.0's 23-joint rigs: 16.9 ms); 50
Minecraft-style characters per frame: 1.4 ms as rigid parts, 6.8 ms as bone-attached
blocks, 3.7 ms as one rigidly skinned mesh. See [PERFORMANCE.md](PERFORMANCE.md) (the
revision's section: what got slower, why, and what was optimized).

## Limitations

* **Automatic rigging** knows its four templates. Humanoids are found in A-poses, T-poses
  and arms-down poses, standing, roughly symmetric, one character per call; facial
  joints, tails on humanoids, wings and props are not detected (add joints with
  `edit_rig`). Landmarks come from measured sections, probes and proportion priors, each
  with its confidence -- not from anatomy -- and a mesh that hides a joint (a long coat
  over the knees) gets the prior there. Fingers are found only where the mesh resolves
  them (edges about as short as their radius) and the hands agree; knuckles are the least
  accurate landmarks (up to 2.3 % of the height on the test figure), and the thumb's base
  is placed from the wrist by a proportion of its measured length. Hands hanging against
  the thighs (an arms-down rest pose) can go unfound (`HAND_NOT_FOUND`: that arm is
  placed by proportions and mirrored, at low confidence, and there are no fingers): rig
  in an A-pose.
* **Weights** are bone heat: a smooth diffusion that matches Blender's Automatic Weights,
  not an artist's final paint. Extreme poses still crease and stretch -- a hip flexed 100
  degrees makes some 43-44 new creases and stretches groin edges 3.4-3.5 times (Blender's
  weights: 30-37 and 2.6-2.8; the hips are where the two differ most) -- and LBS loses
  volume at deep bends (a knee bent 140 degrees keeps about 0.89 of its size). Corrective shapes need targets someone makes; nothing simulates
  muscles, fat, skin sliding or collisions.
* **Delta mush** is opt-in and does not travel through glTF; on good weights it changes
  little, and it needs an even mesh (its smoothing weights come from the rest surface).
* **IK** is Stage 17's: two-bone, FABRIK and CCD per chain; there is no full-body solver,
  and `reach` leans the chest at most 30 degrees toward a goal out of reach, then reports
  what is left (`reachable: false`, the residual). Foot roll is a pose command (keys), not
  a live control with its own channel.
* **Diagnostics** are measurements of the mesh in chosen poses, not a judgment of how a
  pose should look; a verified fix improved what was measured, which is not the same as
  being right.
* **Everything runs on the CPU** (native kernels when installed); there is no GPU
  skinning. glTF keeps linear blend skinning only (dual quaternion and blended skins and
  corrective smoothing are reported losses), and controls, corrective drivers, landmarks
  and constraints are baked or kept as extras, not exported as rig logic.
* **Rigid-part characters** are found by part names (or given by role); pivots come from
  the parts' boxes or their origins; generated motions are the Stage 17 generators'.
## Stage 17, Stage 18 and its revision

| | Stage 17 (0.17.0) | Stage 18 (0.19.0) | Revision (0.19.1) |
| --- | --- | --- | --- |
| Getting a rig | Hand-written joints (`add_rig`) | `autorig` from a bare mesh, with measured landmarks | Landmarks measured on the mesh (probes, sections, armpits, crotch, fingers); a Rigify-like humanoid (~60 joints) |
| Weights | Envelope or nearest | Geodesic voxel binding, region-aware; programmatic edits; transfer | Bone heat (Blender's Automatic Weights), eight influences; the geodesic binding kept |
| Joint limits | Euler bounds | Swing-twist cones, ellipses, hinges relative to rest | The same; finger hinges |
| Skinning | Linear (native), dual quaternion (NumPy) | Linear, dual quaternion and blended (native), cofactor normals | Also surface normals and delta mush (native) |
| Deformation helpers | -- | Twist constraints and helper joints, corrective shapes | Split twist bones (joint `span`), pelvis bones |
| IK | Two-bone, FABRIK, CCD constraints | The same solvers, bend-axis alignment within limits, controls, matched IK/FK switches | The same |
| Posing | Keys | Pose commands (reach, plant feet, look at, crouch, ...) | The clavicle's share in arm raises; finger curls; foot roll |
| Rig changes | Replace the rig | `edit_rig`: structural edits with dependent data kept consistent | The same, also from the command line |
| Checking | Foot contacts, curve reports | `diagnose_rig`, custom poses, previews, verified fixes, feedback `rig` check | Creases and folds, the stress suite, `rig test`, trade-off fixes with measured dry runs |
| Rigid characters | Bone-attached blocks | Also rigid-part generation with no bones, and conversions | The same |
| glTF | Skeletons, skins, animation | Also joint semantics as extras, honest rig loss reports | Eight influences; smoothing and normals reported |
