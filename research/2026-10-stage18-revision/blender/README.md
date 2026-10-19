# Blender reference experiments (Stage 18 revision)

Blender 5.2.2 LTS, headless (`bpy` in background mode), was the reference the revision was
measured against. These scripts are development-time tools: nothing in the `aether3d`
package imports or runs Blender, no Blender code was copied into Aether3D, and every image
in the repository is rendered by Aether3D. (On Arch Linux, run Blender with
`PATH=/usr/bin:/bin` so it does not pick up a virtual environment's Python.)

## Case 1: v0.19.0's rig (the gap analysis)

The showcases' v0.19 mannequin (the implicit humanoid at 12 mm, 18,008 vertices) with the
v0.19.0 auto-rig, exported with its exact bone frames and the stress poses as local
quaternions; Blender computes Automatic Weights (bone heat) and deforms with LBS, DQS and
Corrective Smooth; everything is measured with the same geometry-only metrics.

```sh
python research/2026-10-stage18-revision/blender/export_case.py /tmp/case_v019   # with the v0.19.0 sources on PYTHONPATH
blender -b --factory-startup --python research/2026-10-stage18-revision/blender/blender_ref.py -- /tmp/case_v019
blender -b --factory-startup --python research/2026-10-stage18-revision/blender/blender_ref.py -- /tmp/case_v019 --weights /tmp/case_v019/weights_aether_v019.npy
python research/2026-10-stage18-revision/blender/analyze.py /tmp/case_v019
```

Results: [`metrics_v019.json`](metrics_v019.json). The mesh and Blender's weights are also
the committed test fixture `tests/data/blender_heat/` (`test_rig_quality.py` checks that
Aether3D's bone heat reproduces them: dominant joint on 99.8 % of the vertices, mean
difference 0.0013).

## Case 2: the revision's figure

The anatomical figure (`aether3d.sculpt.figure.figure_mesh(0.007, remesh=0.007)`, 42,850
vertices) auto-rigged by the revision (60 joints, fingers included); every joint becomes a
Blender bone along its weighting segment with its exact frame (deviation 5e-6). Blender
computes Automatic Weights; each stress pose is set bone by bone to the rigid transform
Aether3D's pose gives the joint (`T @ rest`, parents first), so Blender's LBS of Blender's
weights equals Aether3D's LBS of the same weights (measured: within 1e-6 m). Variants:
Blender LBS, DQS (Preserve Volume), LBS + Corrective Smooth (defaults: 5 iterations) and
LBS + Corrective Smooth with 12 iterations; v0.19.0's own rig and weights on the same mesh
(made with the v0.19.0 sources); the revision's heat weights (LBS) and with delta mush.

```sh
python research/2026-10-stage18-revision/blender/export_figure_case.py /tmp/case_figure 0.007 0.007
blender -b --factory-startup --python research/2026-10-stage18-revision/blender/blender_figure.py -- /tmp/case_figure
# v0.19.0's rig of the same mesh (the v0.19.0 sources, e.g. a worktree of 51e6ba1, on PYTHONPATH):
#   load /tmp/case_figure/positions.npy and triangles.npy, autorig, save /tmp/case_figure/v019.aether
python research/2026-10-stage18-revision/blender/analyze_figure.py /tmp/case_figure
```

Results: [`metrics_figure.json`](metrics_figure.json) (per pose and variant: `joint_ring`,
the share of its size the skin around the bent joint keeps; `new_creases`;
`sharpest_crease`; `folded_triangles`; `worst_stretch`) and
[`blender_run_figure.json`](blender_run_figure.json).

`rigify_dump.py` summarizes Rigify's generated human rig (bones, deforming bones and their
B-bone segments, constraints, custom properties, controls): the structure the humanoid
template was compared with.
