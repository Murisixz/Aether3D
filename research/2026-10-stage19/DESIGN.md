# Stage 19: animation, cameras, lighting, path tracing and film output

The design record of Aether3D 0.20.0. It says where Stage 19 started (v0.19.1, measured),
what was learned from Blender and from the literature, what was decided for each part of
the assignment and why, how each piece is verified, and what was measured at the end.
The user documentation is in [ANIMATION.md](../../docs/ANIMATION.md),
[RENDERING.md](../../docs/RENDERING.md), [RENDER_QUALITY.md](../../docs/RENDER_QUALITY.md),
[GPU.md](../../docs/GPU.md), [MATERIALS.md](../../docs/MATERIALS.md) and
[FEEDBACK.md](../../docs/FEEDBACK.md); the Blender experiments are in
[blender/README.md](blender/README.md).

## Ground rules

- **Blender is a measured reference, never a dependency and never a source.** Headless
  `bpy` scripts measured what Blender does; the numbers became fixtures
  (`tests/data/blender_*`). No Blender or Cycles code was read for, copied into or
  translated into Aether3D (Blender is GPL; Cycles is Apache-2.0; nothing of either was
  reused, so neither licence's terms apply to the engine). Where matching Blender would
  have required reconstructing its code rather than measuring its outputs (elastic
  easing), Aether3D keeps its own documented behaviour.
- **Defaults keep old results.** Every new field defaults to v0.19.1's behaviour: the
  render goldens did not change, and the CPU path tracer gives v0.19.1's bits for the
  same scene (checked against the real release, Part eleven). One default changed on
  purpose: new F-curves use Continuous Acceleration (Blender 5.2's default); saved curves
  load as before.
- **Exact or labelled.** Closed forms where they exist (curve derivatives, light
  irradiance on the rasterizers, Fresnel), sampled measurements that say they are sampled
  (motion quality, lighting ratios), and approximations named as such (the rasterizers'
  image lighting and glass, the denoiser's bias).
- **Measure before and after.** v0.19.1 was benchmarked first
  (`benchmarks/results/stage19-baseline-*.json`); comparisons keep resolution, samples and
  scene content equal and say when a comparison is at equal time instead.
- **One branch, one pull request**, never merged without the owner's approval.

## Starting point: v0.19.1

| Area | v0.19.1 | Measured gap |
| --- | --- | --- |
| Automatic handles | Aether3D's local rule (C1) | far from Blender's in both smoothing modes (up to 5.8 on a range of 7.2; 14.7 on 26.6), E1 |
| Easings | Penner's | 26 of 30 cases equal to Blender's within 2.4e-6; back in-out off by 0.97 (Penner's 1.525 factor), elastic different (E1) |
| Modifiers | offset, scale, limits, noise, wave, envelope, smooth, stepped, time offset and scale | no generator, built-in function or cycles modifier |
| NLA | layers with masks and weights, strips with linear ramps | crossfade velocity steps at the ramps' ends (E2) |
| Motion checks | per-curve reports | nothing measured the evaluated world-space motion of objects, joints or cameras |
| Cameras | field of view, clipping | no lens model, no depth of field, no shot framing |
| Lights | directional and point (sized in the path tracer) | no spot or area lights, units, temperature, linking or per-light shadows |
| Environments | ambient, uniform, procedural sky | no images; nothing sampled the environment as a light |
| Path tracer | GGX metals and dielectrics, MIS for emitters, hashed random numbers, box filter | no glass, no low-discrepancy samples, no pixel filters, no adaptive sampling, no denoiser |
| Output | 8-bit PNG, `.hdr`, `.pfm`; neutral tone mapper; FFmpeg at CRF 18 | no filmic view transform, no 16-bit or EXR masters, no motion vectors, no BT.709 tags |
| Feedback | geometry, framing, exposure and other checks | nothing measured lighting |

## Part one: keyframes and graph curves

- **Smoothing modes** (`animation/smoothing.py`). `"continuous"` solves one tridiagonal
  system per run of automatic keys for C2 continuity (third-of-interval extents; free and
  aligned neighbours enter with their Bezier geometry, vector neighbours end runs
  naturally, cyclic curves are periodic systems, clamped keys become bounds enforced by
  locking and solving again). `"blender"` reproduces *None*: the mean of the secants,
  extents of the interval / 2.5614 with each interval limited to five times the other,
  Blender's clamping box. Both were written from E1/E1b measurements and the mathematics
  of cubic splines; an independent model (`blender/e1_curves/model.py`) was written first
  and checked against the data, then the engine's implementation against the fixture:
  every handle and value of 49 cases within 2e-6 of the case's scale.
- **Easing.** Back eased in and out now applies Penner's 1.525 to the overshoot, as
  Blender; stored keys are migrated (divided by 1.525) so their curves keep their shapes.
  Elastic stays Penner's (measured different from Blender's; see the ground rules).
- **Modifiers.** `generator` (polynomials, additive or replacing, ranged), `function`
  (sin, cos, tan, sqrt, ln, sinc, with amplitude, phase multiplier and offset) and
  `cycles` (before/after modes and counts) with exact derivatives; a curve with an endless
  `cycles` modifier gets cycle-aware automatic handles. 17 cases against Blender within
  4e-6.
- **Editing operations** (`edit_keys`): `insert` keeping the curve's shape, `ease`,
  `anticipate`, `overshoot`, `hold`, `smooth`, `smoothing`, `extrapolation`, beside the
  existing set, move, scale, reverse, quantize, distribute and retime. All validate first
  and change one curve atomically.
- **Graphs** (`animation/graph.py`): value, velocity, acceleration and jerk, labelled keys,
  handles, extrema and discontinuities, and reference samples drawn over them (showcase C
  draws Blender's), as SVG or PNG -- for agents that can see, with the numbers in
  `curve_report` for those that cannot. No mouse-driven editor was built.

## Part two: smoothness and expressive motion

- **World-space motion quality** (`animation/quality.py`, `anim.motion_report`): the
  evaluated scene (layers, strips, rigs, constraints, IK, paths, shots) sampled at a rate,
  with one-sided least-squares fits so smooth variation cancels and only breaks remain:
  position and velocity jumps, acceleration steps, rotation snaps, jitter above a
  frequency, pauses, IK pole flips, loop seams, ground penetration, and for cameras aim
  error, framing drift, the subject leaving the frame or being obstructed, and focus.
  Every number is labelled sampled and every threshold is in the report. It is a feedback
  check (`motion_quality`) with verified corrections (smoothing handles, crossfades,
  retiming), and MCP `inspect_animation` `motion`.
- **Blending.** Strip ramps may be `smooth` (C1) or `smoother` (C2), removing the velocity
  steps linear ramps make (E2). Quaternion blending stays slerp: a measured comparison
  showed it equals Blender's COMBINE, while Blender's REPLACE adds components (0.737 long
  midway); copying REPLACE would make rotations speed up and slow down through a blend.
- **Characters.** Stage 18's rigs are used unchanged (showcase B); rigid-part characters
  keep animating without bones (showcase A), with crossfaded gestures and jumps measured by
  `motion_report`.

## Part three: cameras

- **Lenses** (`rendering/camera.py`): sensor, focal length (derived from and setting the
  field of view), shift, f-stop, focus distance or focus object, aperture blades, rotation
  and ratio, validated and saved (format 13). E5 measured Blender's shift and circle of
  confusion; Aether3D's thin lens follows that model: a point at distance `d` blurs into a
  disk of `(f / N) |1/d − 1/focus|`.
- **Depth of field is real** on every backend: the path tracers sample a thin lens (its
  random numbers in the same Sobol' dimension group as the pixel position); the
  rasterizers average `dof_samples` pinhole views over the aperture. Nothing blurs an image
  in post.
- **Shots** (`animation/shots.py`): `frame_shot` places a camera for a named shot size
  measured on the posed subject; look-at constraints take a screen position (composition);
  focus is animatable for rack focus. Camera moves, paths and shot sequences existed.
- **Diagnostics**: camera motion is part of `motion_report` (jitter, drift, framing,
  focus error at holds).

## Part four: lighting

- **Lights** (`rendering/light.py`). Spot lights (cone and soft edge: a smoothstep of the
  cosine), area lights (square, rectangle, disk, ellipse; one-sided Lambertian; `spread`
  as a honeycomb's angular profile that keeps the power), Blender's watts as fixed
  factors (sun `S / pi`, point and spot `P / (4 pi^2)`, area `P / pi^2`), colour
  temperature (`blackbody.py`: Planck's law against the CIE 1931 functions, normalized to
  luminance 1), exposure in stops, per-light shadows and light linking. All measured
  against Cycles (E3, E3b) on every backend.
- **One model, four implementations.** The CPU rasterizer (`surface.py`) and the GPU's
  (`scene.wgsl`) integrate area lights exactly (Lambert's polygon formula clipped at the
  horizon; disks and ellipses as 16-gons of the same area) with a representative-point
  highlight, and spread by a 12 × 12 midpoint rule; the path tracers sample lights (MIS
  with BSDF sampling) and integrate spread exactly. Linking and shadow flags are honoured
  by all four.
- **Studio rigs** (`rendering/studio.py`): eight classical setups placed from the camera's
  side of the subject, intensities solved so the key and fill give the requested level in
  the preset's ratio, as ordinary tagged lights in one undo step.
- **Image environments** (`rendering/envimage.py`): Radiance and PFM files (untrusted:
  sizes checked first) or four built-ins drawn by Aether3D, with Blender's mapping (E6),
  importance sampled by the path tracers; the rasterizers light with nine spherical
  harmonics and one extracted sun (an approximation, documented).
- **Day and night are authored.** The world became an animation target (`{"world":
  true}`): background, ambient light, the environment's intensity, rotation and sky
  colours, the atmosphere's glows and fog, evaluated into the pose and applied by the
  render overlay in the one place both extraction paths share, so every backend shows
  them and rendering at a time equals applying the pose. With keys on the sun light, that
  is a day-night cycle (showcase D's timelapse); nothing simulates the sky, and glTF,
  which has no world, reports the loss.
- **Emission lights the scene** in the path tracers (emissive triangles are lights);
  showcase F proves it by rendering its street with every analytic light hidden and the
  environment off: the emissive windows and lamps alone light it, and turning the
  emission off leaves black.

## Part five: path tracing

- **Glass**: `transmission` and `ior` on standard materials; a rough dielectric
  (Walter et al. 2007) with the exact Fresnel term, total internal reflection, GGX visible
  normals, the side of the surface tracked along the path; verified by a white furnace
  (0.9988 smooth, 0.946 at roughness 0.3: single-scattering loss), a slab's
  `(1 − R) / (1 + R)` and lobe quadratures.
- **Sampling**: Owen-scrambled Sobol' (4D points per decision group, 24-bit integer
  arithmetic, the same bits on the GPU), filter importance sampling (box, Gaussian,
  Blackman-Harris through a shared 256-entry inverse CDF), adaptive sampling in doubling
  rounds whose pixels stop when they and their neighbours reach the threshold (a stopped
  pixel's samples are a prefix of its full sequence). Measured honestly at equal time,
  not only at equal samples (`measurements/adaptive.py`, 320 × 240, CPU): 5-45 % less
  error with the same mean samples, but the pixels that keep sampling are the expensive
  ones (35-60 % more time per sample), so at equal time it gained 20 % in one of four
  settings and lost 1-5 % in the others. It stays opt-in, documented with these numbers.
- **Denoiser**: an edge-avoiding à-trous filter guided by the variance (the spatial part of
  SVGF), on the beauty divided by the albedo; 20–35 % less error than the raw image on its
  test scenes; opt-in, biased and reported.
- **Environment and area lights sampled as lights**, with MIS.
- **Fireflies**: the existing `clamp` (opt-in bias) remains the tool; caustics are noisy
  and documented as such.
- **Energy**: furnace tests for glass; the GGX lobes' multiple-scattering loss is
  documented, not compensated.
- **The GPU kernel is a port**: every feature above runs in `pathtrace.wgsl` with the CPU's
  random numbers (Sobol' bits included), the same filter table and environment CDFs;
  adaptive rounds read the sums back to decide which pixels continue.
- **Hardware ray tracing** was re-examined on the RTX 4060: wgpu-native 29.0.1.1 (under
  wgpu-py 0.32.0) declares ray-query features but has no function to build or bind
  acceleration structures, so Phase 5's decision stands (deferred; software traversal on
  the shader cores, never reported as hardware ray tracing).

## Part six: film output

- **The `film` view transform** (`rendering/tonemap.py`): an inset matrix, a monotone
  cubic tone curve per channel on log2 exposure around middle grey 0.18 (Fritsch-Carlson),
  an outset matrix, sRGB. Its curve and matrices were fitted to Blender 5.2's AgX measured
  on grey and seven colour ramps (E3b; `film_fit/fit.py`, Nelder-Mead without SciPy): RMS
  display error 0.008, at most 0.038. Middle grey shows at 0.4613 (sRGB 118); +6.5 stops
  reach white. It is Aether3D's own construction, not Blender's configuration.
- **Masters**: 16-bit PNG (`bits=16`, `format="png16"`) and OpenEXR (half floats, ZIP,
  passes as channels; our own writer, read back exactly by FFmpeg and by our reader).
- **Motion vectors** (`rendering/motion.py`): exact per-pixel offsets to where each pixel's
  surface point was and will be, from the same triangle posed at other times.
- **Sequences** (`render_frames`): 60 fps, real motion blur by temporal sampling, depth of
  field, progress, cancellation, a manifest that resumes interrupted renders by keeping
  verified frames, reports.
- **Video**: FFmpeg only assembles finished frames: H.264 (libx264, preset slow, CRF 10,
  BT.709 matrix, primaries, transfer and range tags) and lossless FFV1 masters (8 or 16
  bits); the showcases verify each video by PSNR and SSIM against its frames.

## Part seven: AI-native controls

Every feature is reachable from Python, the typed animation paths, scene and animation
scripts (atomic, schema-described, fuzzed), the CLI and scene plans; MCP gains the render
options, `inspect_animation` `motion` and the lighting check while its tool list stays
under 40,000 bytes. The **lighting check** (`feedback/lighting.py`) measures, from the
scene's lights, each light's irradiance on the subject's visible points (the rasterizers'
diffuse term, ray-traced shadows, linking), the key-side lighting ratio, the subject's
separation from its surround in CIE L* of the displayed image, and lights that do not
reach it with the reason. Its planners predict the effect of scaling a light from the
measured contributions and keep a change only when a new render confirms it; adding a
fill or rim light needs approval. Showcase I's script runs the whole loop over MCP (in
the tests, at its quick size).

## Part eight: languages

| Subsystem | Language | Why |
| --- | --- | --- |
| Curves, smoothing solves, modifiers, motion quality | NumPy | vectorized over keys and samples; the solves are tridiagonal (linear time) |
| Lights, environments, studio rigs, lighting check | NumPy (raster), WGSL (GPU raster) | formulas evaluated per pixel or per sample point; the GPU already shades |
| Glass, Sobol', filters, adaptive sampling | NumPy (CPU path tracer), WGSL (GPU path tracer) | the CPU tracer is the reference; the GPU port is where speed comes from |
| Denoiser, view transforms, EXR, motion vectors | NumPy | once per image |
| Video | FFmpeg (subprocess, examples and `assemble_video` only) | encoding only, never rendering |

No native kernel was added: the measurements behind that choice are in
[Performance](#performance).

## Part nine: the Blender laboratory

See [blender/README.md](blender/README.md): eight experiments (E1, E1b, the modifier
cases, E2, E3, E3b, E5, E6) with what was measured, what came from published work instead,
and what was not verified. REA was available and not used: every question was answerable
by measuring Blender's outputs through `bpy`, and analysing its binaries would have
conflicted with the clean-room rule.

## Part ten: showcases

`examples/film_*.py`, each writing a `report.json` beside its images and videos under
`examples/output/film/`:

| | Showcase | What it shows |
| --- | --- | --- |
| A | `film_a_rigid.py` | a Minecraft-style rigid character: crossfaded gestures and jumps, measured; atmosphere; GPU raster frames |
| B | `film_b_character.py` | Stage 18's skinned character greeting: masks against foot sliding, eased nods, path-traced frames |
| C | `film_c_graphs.py` | curve graphs against Blender's fixture: cases, smoothing modes, quaternion blending |
| D | `film_d_world.py` | a crane-and-push through the Stage 16 alpine valley, clearance measured, lens zoom; a day-night timelapse authored with the animatable world |
| E | `film_e_studio.py` | a studio product shot: light rigs, glass, metals, an image environment, depth of field, an orbit with rack focus |
| F | `film_f_night.py` | a night street lit by emissive windows and lamps, with the proof that emission alone lights it |
| G | `film_g_renderer.py` | the path tracer of v0.19.1 (the real release) against 0.20.0 at equal samples and equal time |
| H | `film_h_trailer.py` | a 1080p60 trailer cut from the showcases' frames |
| I | `film_i_mcp_agent.py` | an agent lighting, checking, correcting, animating and rendering over MCP |

A, C, D, E and F were rendered for 0.20.0; their results are in
[ANIMATION.md](../../docs/ANIMATION.md#film-showcases-0200). B, G, H and I were not: their
full renders were stopped or cancelled so that the engine's validation could finish
first, and nothing in this record rests on their full-size results. The tests run all
nine at their quick sizes (`tests/test_examples.py`); at that size G skips its v0.19.1
run, so v0.19.1's bits were checked separately (Part eleven).

## Part eleven: verification and compatibility

- New test files: `test_curve_smoothing.py`, `test_curve_timing.py`,
  `test_motion_quality.py`, `test_camera_lens.py`, `test_lights.py`, `test_studio.py`,
  `test_environment_image.py`, `test_glass.py`, `test_sampler.py`, `test_denoise.py`,
  `test_film_output.py`, `test_motion_vectors.py`, `test_feedback_lighting.py`,
  `test_format_v13.py`, `test_world_animation.py`, and the film showcases in
  `test_examples.py`; the history catalog (`test_history.py`) covers every new kind of
  change, and the render-scene parity fingerprint covers the new light fields, linking,
  environments and the atmosphere.
- Parity: the GPU rasterizer against the CPU's (lights, linking, environments), the GPU
  path tracer drawing the CPU's random numbers and stopping the same adaptive pixels.
- Compatibility: format 13 with a migration and a test loading a version-12 document;
  goldens unchanged; v0.19.1's CPU path-traced bits reproduced. The last was checked by
  running the real v0.19.1 (a git worktree of `main` at 7752729) and 0.20.0 on one
  machine on two scenes, compared bit for bit: the Cornell room of
  `examples/film_g_scene.py` (`--cpu`, 96 x 54, 4 samples), and a scene with a sun with an
  angle, a sized point light, an emissive box, GGX metal and dielectric and the procedural
  sky (64 x 48, 8 samples, 5 bounces). It is not a committed test, because NumPy's
  transcendental functions may differ in their last bits between builds and CPUs.

## Performance

Measured at the end against the baseline, with the same commands on the same machine
([PERFORMANCE.md](../../docs/PERFORMANCE.md#film-lighting-and-path-tracing-020)). Median
end-to-end speed, 0.20.0 / v0.19.1:
- animation 0.84x (0.94x when the two releases run back to back);
- engine 0.95x;
- GPU rasterizer 0.95x;
- GPU path tracer 0.88x;
- CPU path tracer 0.88x;
- ray tracing 1.05x;
- CPU rasterizer 0.95x.

Unchanged code measured 3-10 % slower in the final run (the machine's state after hours
of rendering). Real regressions found by back-to-back runs:
- building curves with automatic handles costs about three times as much under the new
  default smoothing (0.50 against 0.16 ms for a four-key curve; `smoothing="local"` keeps
  the old cost);
- fifty rigs' rigging pass per frame went from 0.37 to 0.87 ms;
- the GPU rasterizer's castle scene supersampled at 1080p takes 30.6 against 10.8 ms of
  GPU time (cause not identified).

The GPU path tracer's larger kernel lowers its ray rate by 7-19 % against the baseline
(machine state included).

**Languages.** Profiling the new code at the sizes it is used at (`measurements/profile.py`,
`profile.json`) found no hot loop worth a native kernel:
- the solves are linear (7.8 ms for 10,000 keys);
- per frame at 1080p, the denoiser (4.9 s), the `film` transform (0.58 s), EXR (0.35 s)
  and the motion pass (3.6 s) are opt-in and small against a path-traced film frame
  (25.5-68 s on the GPU in showcases F and E);
- the lighting check's measurement takes 49 ms.

The measured costs worth a later kernel or GPU pass are the per-curve overhead of the
smoothing solve on small curves, the denoiser and the motion pass.

## Limitations

- **No hardware ray tracing.** wgpu-native 29.0.1.1 cannot build acceleration
  structures; both path tracers traverse their BVH in software.
- **The rasterizers approximate** image environments (nine spherical harmonics and one
  extracted sun) and glass (its reflection only); the path tracers are the reference.
- **Energy.** Rough glass and GGX lose what single scattering loses (5 % for glass at
  roughness 0.3); nothing compensates multiple scattering. Caustics are noisy; `clamp`
  (an opt-in bias) is the tool against fireflies.
- **Opt-in for measured reasons.** Adaptive sampling lowers error at equal samples but
  gained at equal time in one of four measured settings; the denoiser is biased and says
  so.
- **Deliberately not Blender's.** Elastic easing stays Penner's; sized point lights keep a
  uniform sphere's irradiance where Cycles' soft falloff gives `d^2 / (d^2 + r^2)` of it.
- **Not compared with Blender**: light linking's semantics, glass, Cycles' adaptive
  sampling and denoisers, shutter shapes, the Vector pass's conventions, AgX outside the
  seven measured ramps (see [blender/README.md](blender/README.md)).
- **Motion blur** averages renders at sampled times (no velocity-buffer blur); glTF has
  no world, so world animation is reported as not exported.
- **Showcases B, G, H and I** were not rendered at full size for this release (Part ten).
