# Stage 18 revision: rigging quality, measured against Blender

Written on 18 October 2026 against `main` at 51e6ba1 (v0.19.0, Stage 18 merged). This is
not a new stage: it revises Stage 18's character rigging because its results were poor in
practice although its tests passed. Section 1 is the gap analysis made **before** any
code changed; later sections record the plan and, afterwards, the outcome.

Blender 5.2.2 LTS (headless, `bpy` in background mode) was the reference. It is a
development-time tool only: no Blender code is copied into Aether3D (Blender is GPL, Aether3D
Apache-2.0), nothing in the package imports or runs it, and every final image and video is
rendered by Aether3D. The experiment scripts are in [`blender/`](blender/).

## 1. Gap analysis (measured before changing anything)

### 1.1 How the comparison was made

One case, identical in both systems: the showcases' mannequin (18,008 vertices, the
implicit humanoid meshed at 12 mm), the v0.19.0 auto-rig's 23 joints exported with their
exact frames (Blender bones built from the same heads, tails and roll axes: frames agree to
3e-7), and the same stress poses (the same joint rotations, sent as local quaternions).
Blender then computed **Automatic Weights** (bone heat) and deformed with its Armature
modifier -- linear (LBS), Preserve Volume (DQS) and with a Corrective Smooth modifier. All
deformed meshes came back into Aether3D and were measured with the same geometry-only
metrics (the same vertex sets for every binding; sliver edges and triangles of the
marching-cubes mesh excluded, as `diagnose_rig` does):

* **kept** -- the mean distance of the skin near the bent joint to its pivot, posed over
  rest (a collapsing elbow keeps less);
* **new creases** -- edges whose dihedral angle exceeds 60 degrees in the pose but not at
  rest; **p99.9 dihedral** -- how sharp the sharpest creases are;
* **folds** -- triangles facing against their neighbourhood;
* **stretch** -- edge length ratios.

The pipeline was validated first: Blender's LBS of Aether3D's own weights equals
Aether3D's `skin_linear` within 7e-7 m (float32) in all 18 poses.

### 1.2 What was wrong

| Area | v0.19.0 (measured) | Reference (measured) | Gap |
| --- | --- | --- | --- |
| **Weights** | geodesic voxel distances, `d^-3` falloff, region masks, smoothing: **40-425 new creases per pose**, p99.9 dihedral up to 151 deg (rest: 56), up to 35 folded triangles; elbows and knees bent 120 deg keep **0.81** of their size | Blender bone heat on the same skeleton: **0-40 new creases**, p99.9 dihedral 56-71 deg, 0-3 folds; elbows and knees bent 120 deg keep **0.94** | the weights, not the skinning, make the crumpled armpits, groin creases and dented backs seen in Showcase B. Dominant joints agree on only 83 % of vertices; mean weight difference 0.19 per vertex |
| **Joint placement** | on the same recipe body, depending only on the mesh's resolution: shoulder **0.9 or 6.2 cm** off its modeled joint, elbow 0.2 or 3.7 cm, knee **3.9-5.1 cm** at every resolution, hip 2.2-4.5 cm, neck 1.0-2.6 cm; the head bone **tilted 7-13 deg** (the head-top landmark is not kept on the mirror plane); the spine a straight line through the torso's centroid | Rigify's human metarig (as riggers place it): spine along the back with an S curve, clavicles from the sternum back to the shoulder, hip joints at about 0.54 of the stature | the tests accept errors up to 4 % of the height (7 cm) and use the one mesh resolution (14 mm) at which the shoulder happens to be found: **they pass badly** |
| **Twist** | one forearm twist helper (half the wrist's twist) | Rigify: every limb bone split in two deforming bones, each a 10-segment B-bone | upper arms and thighs twist as one piece |
| **Rig completeness** | 23 joints: 3 spine joints, no fingers, no heel, no foot roll | Rigify human: 160 deforming bones; 6-bone spine; 3 phalanges per finger and palm bones; heel, toe and ball pivots (foot roll); IK/FK switch, IK stretch, pole, parent-follow properties; finger curl masters | no fingers, no foot roll, a stiff spine |
| **Shoulders** | the diagnostic shoulder raise turns the upper arm alone, up to 150 deg | an animator lifts the clavicle above about 90 deg (the scapulohumeral rhythm) | the pose itself is unnatural, and the clavicle's weights never get used |
| **Normals** | rest normals carried by the blended joint matrices; documented jagged dark patches in path-traced twists | Blender recomputes normals from the deformed surface | visible artifacts in every twisted render |
| **Corrective smoothing** | none (DQS blend only where a collapse is measured) | Corrective Smooth modifier (delta mush) | no general cure for residual LBS pinching |
| **Showcase character** | capsules and ellipsoids, mitten hands, marching-cubes slivers | -- | weak test subject and weak pictures |

Measured detail (`blender/metrics_v019.json`; LBS unless noted):

| Pose | v0.19 kept | Blender heat kept | v0.19 new creases | Blender new creases | v0.19 folds | Blender folds |
| --- | --- | --- | --- | --- | --- | --- |
| elbow bent 120 | 0.817 | 0.936 | 84 | 14 | 5 | 0 |
| knee bent 120 | 0.811 | 0.937 | 47 | 16 | 0 | 2 |
| hip flexed 90 | 0.936 | 0.884 | 425 | 16 | 35 | 0 |
| shoulder raised 90 | 0.974 | 0.959 | 63 | -5 | 2 | 0 |
| shoulder raised 150 | 0.887 | 0.823 | 60 | 9 | 5 | 2 |
| torso twisted 60 | 0.982 | 0.989 | 269 | -5 | 6 | 0 |
| spine bent 45 | 0.994 | 0.997 | 191 | 0 | 0 | 0 |
| wrist twisted 90 | 0.969 | 0.955 | 60 | 13 | 5 | 0 |

Blender's Corrective Smooth with its default settings (simple smoothing, 5 iterations)
*added* creases on this marching-cubes mesh (its slivers fold when smoothed): delta mush
needs a clean mesh and length-weighted smoothing -- noted for the design below.

### 1.3 What Blender showed, and what Aether3D takes from it

* **Bone heat** (Baran and Popovic 2007, Blender's `heat_bone_weighting`): each bone is a
  heat source for the vertices it is nearest to *and visible from* (a ray from the vertex
  to the bone's nearest point that does not re-enter the surface); the distance is
  penalised when the vertex normal faces away from the bone; one sparse solve per bone of
  `(L + H) w = H p` with the cotangent Laplacian `L`, `H = 1/d^2`. The weights are smooth
  by construction (a diffusion), centred on the joints, and sum to one. Taken: the
  principle, re-derived as a symmetric system and solved with Aether3D's own nested-
  dissection LDL^T (`uv/sparse.py`, with its native twin), visibility by Aether3D's own
  watertight ray traversal. Added: safeguards Blender lacks (a vertex that sees no bone
  takes its neighbours' heat; disconnected parts are solved separately; the side and
  helper-joint rules; a voxel-geodesic fallback when the surface is not usable).
* **Rig structure** (Rigify): split limb bones for twist, a curved spine set toward the
  back, clavicles from the sternum, fingers, heel/toe/ball foot roll, IK/FK switching with
  snapping, poles, and parent-follow. Taken as the target structure of the humanoid
  template, built from Aether3D's own joints, constraints and control objects.
* **Deformation**: normals recomputed from the deformed surface; delta mush (Corrective
  Smooth) as an opt-in cure for LBS pinching.

## 2. Plan

1. **Characters to test on.** An original, more anatomical procedural character (hands
   with five fingers, deltoids, chest, glutes, calves, knees, feet, a face) whose joint
   centres are known exactly, in several builds and rest poses, meshed and remeshed to
   clean triangles; landmarks are measured against those centres at several mesh
   resolutions.
2. **Landmarks.** Midline landmarks on the mirror plane; shoulders, hips, knees and
   elbows located by limb sections measured perpendicular to the limb axis; the spine
   along the back; clavicles; fingers counted and placed when the hand shows them.
3. **Weights.** Bone heat as the default (`method="heat"`), the geodesic voxel binding
   kept as a method and as the fallback.
4. **The humanoid template.** Four torso joints along the measured back, neck, head;
   clavicles; twist segments for upper arms and forearms (thighs optional); fingers when
   detected; heel landmarks for foot roll.
5. **Deformation.** Skinned normals that follow the deformed surface; an optional delta
   mush pass (native kernel with a bitwise NumPy twin).
6. **Controls and poses.** Foot roll, finger curls, arm raises that share the elevation
   with the clavicle.
7. **Validation that cannot pass badly.** Landmark tolerances of 1.5 % of the height at
   every tested resolution; deformation-quality thresholds from the Blender reference
   measured on the same case (a committed fixture); before/after and Blender comparison
   sheets; a stress-test command.
8. **Interfaces.** `aether3d rig` verbs for the whole workflow; MCP `inspect_rig` gains
   the stress test; scripts gain every new option.
9. **Showcases and video**, rebuilt on the new character.

### Languages (decided per subsystem before writing it)

| Subsystem | Language | Why |
| --- | --- | --- |
| Heat system assembly (cotangent Laplacian, areas, distances, visibility rays) | NumPy + the existing native ray traversal | vectorized per triangle and per vertex-bone pair; rays already have a Rust kernel |
| Heat solve | the existing sparse LDL^T (Rust, bitwise NumPy twin) | one factorization, one solve per bone: exactly what it does for UV unwrapping |
| Landmarks, sections, template | NumPy / Python | geometry analysis on a few thousand voxels and sections; no hot loop |
| Delta mush (per frame) | Rust + a bitwise NumPy twin | iterations x edges every frame for every skinned mesh: the per-frame cost must stay small |
| Skinned normals from the surface (per frame) | NumPy (bincount scatter in a fixed order); native only if measurement demands it | one pass over the triangles |
| Pose commands, controls, diagnostics, CLI, MCP | Python | orchestration |

## 3. Outcome

### 3.1 What changed

| Subsystem | Before (v0.19.0) | After (the revision) |
| --- | --- | --- |
| Test character | capsules and ellipsoids, mitten hands | `aether3d.sculpt.figure`: an original anatomical figure (pelvis, glutes, ribcage, pectorals, deltoids, biceps and triceps, forearm muscles, knees, calves, feet with five toes, hands with knuckles, fingers and a thumb, a face) whose joint centres are known; nine builds and three rest poses, meshed and optionally remeshed |
| Landmarks | voxel chains and proportions; errors to 6 cm depending on the mesh resolution | measured on the exact mesh: ray probes, sections, limb axes, the crotch and buttock cues, the armpit rule for shoulders, the palm-to-forearm minimum for wrists, fingers from surface distances; all major joints within 2.5 % of the height over 27 figure cases and 3 resolutions; fingers within 2.3 % or, when the two hands disagree (a mesh too coarse for them), none and `FINGERS_UNCERTAIN` |
| Weights | geodesic voxel distances (default), 4 influences | **bone heat** (default): one symmetric system per mesh, Aether3D's sparse LDL^T, visibility by its watertight rays; **eight influences**; left and right faded across the mirror plane; the geodesic method kept |
| Template | 23 joints | about 60: Hips, Spine, Chest, UpperChest, Neck, Head; Pelvis.L/R; clavicles, upper arms and forearms split for twist (joint `span`, format 12), hands, three joints per finger and thumb; thighs split for twist, shins, feet, toes |
| Skinning | LBS, DQS, blended | the same, plus **surface normals** (default for new skins) and **delta mush** (opt-in; cotangent smoothing weights, Blender's per-corner frames; a Rust kernel bitwise equal to NumPy) |
| Poses | reach, plant_feet, look_at, raise_arm, lean, crouch, aim_hips, rotate | `raise_arm` with the clavicle's share (the arm exactly where asked), `curl_fingers` (curl, fist, point, spread), `roll_foot` (heel, ball, toe; absolute) |
| Diagnostics | joint ring, flips, stretch over an everyday suite | + **creases, folds** (geometry only) and a **stress suite** (elbows and knees 140, hips 100, shoulders 150 with the clavicle, torso 60); corrective smoothing proposed as an `approval` trade-off with a measured dry run |
| Interfaces | `rig autorig, info, landmarks, weights, diagnose, fix, preview, parts, convert, guide` | + `rig test` (with a stress sheet; exit code 3), `create`, `edit`, `bind`, `transfer`, `pose`, `ikfk`; MCP `inspect_rig` `suite`; every new option in the script schema and guides |

### 3.2 Measured results

The revision's figure (42,850 vertices), the 13 stress poses, geometry-only measures, all
deformations computed by Aether3D except Blender's own (`blender/metrics_figure.json`).
Totals over the suite and the bent joints' kept size:

| Binding | New creases | Folds | Worst stretch | Elbows and knees keep |
| --- | --- | --- | --- | --- |
| v0.19.0 (its own rig and weights) | 212 | 10 | 6.32 | 0.73-0.74 |
| revision: heat weights, LBS (the default) | 174 | 9 | 3.83 | 0.88-0.89 |
| revision: heat weights + delta mush (8 iterations) | 170 | 9 | 3.82 | 0.87-0.89 |
| Blender 5.2 Automatic Weights, LBS | 160 | 7 | 3.83 | 0.88-0.89 |
| Blender, DQS | 142 | 7 | 6.25 | 1.00 |
| Blender, LBS + Corrective Smooth (defaults) | 154 | 8 | 3.69 | 0.87-0.89 |
| Blender, LBS + Corrective Smooth (12 iterations) | 165 | 6 | 3.58 | 0.86-0.89 |

Weights against Blender's: the dominant joint agrees on 99.98 % of the figure's vertices
and on 99.8 % of case 1's (mean difference 0.0013 there; v0.19.0: 82.6 % and 0.194).
Blender's weights skinned by Aether3D equal Blender's own LBS within 1e-6 m.

### 3.3 What did not work, and what the measurements changed

* **Delta mush with uniform smoothing and one frame per vertex** (the first version):
  equivariant under rigid motion, yet it scattered vertices by about a millimetre at bent
  joints -- uniform Laplacian smoothing drifts an irregular mesh's vertices along the
  surface, the rest detail becomes large and tangential, and a frame built from one edge
  shears. On the figure it *added* creases (elbow 6 -> 70) and folds. Blender's Corrective
  Smooth averages a frame per triangle corner; that alone barely helped. Cotangent
  smoothing weights (no drift on flat regions: the detail is small and normal) fixed it.
* **The four-influence cap**, not the smoothing, was what creased twisted torsos: with
  four influences, heat weights gave a twisted torso 118 new creases and 24 folds where
  Blender's weights gave 2 and 0, although the dominant joints agreed on 99.98 % of the
  vertices; with eight (Blender gives 12 % of the vertices more than four), 2 and 0. Delta
  mush had been hiding it. With good weights, delta mush lowers the stretch at bent joints
  a little and changes creases and folds hardly at all, while glTF cannot carry it: it is
  **opt-in** (and proposed for skins capped at four influences, such as most glTF
  imports, where it removes a twisted torso's creases).
* **A midline rule** handing the crotch's weights to the hips made a hip flexion worse
  (99 -> 185 creases): the stretch it targeted came from the figure's thighs being fused
  into one surface down to mid-thigh. The figure now keeps a gap between the legs below
  the crotch (`LEG_GAP`, at least two voxels); the rule was dropped.
* **Fingers at coarse resolutions** came out 5-8 % of the height wrong with the same
  confidence as good ones. The two hands of a symmetric body must now agree within 1.5 %
  of the height (measured: good detections agree within 0.5 %, bad ones disagree by 1.6
  to 7.5 %), or neither gets fingers.

### 3.4 Languages, as decided and as measured

| Subsystem | Language | Measured |
| --- | --- | --- |
| Bone heat (assembly, visibility, solve) | NumPy + the native ray traversal and sparse LDL^T | 42,850 vertices, 60 joints: about 1 s inside a 2.2 s auto-rig |
| Delta mush (per frame) | Rust, bitwise equal to the NumPy reference | 11.7 ms per frame (NumPy: 53 ms) at 41k vertices |
| Surface normals (per frame) | NumPy | about 10 ms per frame at 41k vertices: not worth a kernel yet |
| Landmarks, template, poses, diagnostics, CLI, MCP | Python / NumPy | detection 1-2 s; the stress suite 1.5-2 s |

### 3.5 REA

Not used. The headless `bpy` experiments and Blender's source answered every question the
revision raised: the reference weights were reproduced to 99.98 % dominant agreement,
which says more about the shipped binary's behaviour than disassembling it would have.

### 3.6 Remaining limits

* Extreme poses still crease and stretch: a hip flexed 100 degrees makes 45-55 new
  creases and stretches some groin edges 3.5 times -- as much as Blender's Automatic
  Weights do. Real fixes there are corrective shapes or muscle-like deformation, which
  Aether3D does not simulate.
* Finger knuckles are the least accurate landmarks (up to 2.3 % of the height); the
  thumb's base is placed from the wrist by a proportion of its measured length.
* Delta mush and surface normals do not travel through glTF (reported:
  `SKIN_SMOOTHING_LOSSY`, `SKIN_NORMALS_DIFFER`).
* Foot roll is a pose command (keys), not a live control with its own channel.
