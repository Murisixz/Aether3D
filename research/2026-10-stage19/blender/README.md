# Stage 19's Blender experiments

Blender was Stage 19's measured reference: headless `bpy` scripts asked it questions whose
answers are numbers -- handle positions, curve values, blended angles, pixel radiance,
centroids, blur widths, looked-up texture coordinates -- and those numbers became test
fixtures. Aether3D's implementations were written from the numbers and from published
mathematics, never from Blender's code: no Blender or Cycles source was read for, copied
into or translated into the engine. Blender is not a dependency of Aether3D, nothing in
the package imports `bpy` or runs Blender, and every image in the showcases is rendered by
Aether3D.

The lab lived outside the repository (`~/aether-reference-lab`, one folder per
experiment); the scripts and their outputs are copied here. Two Blender builds were
available: 5.2.0 LTS (Flatpak; E1, E1b, E2, E3, E3b, E5) and 5.2.2 LTS (Arch Linux
package; E6). Renders used Cycles on the CPU without denoising. No E4 experiment was run.

Run any experiment with

```text
blender -b --factory-startup --python SCRIPT.py -- OUT.json
```

and compare with the repository's Python (`.venv/bin/python compare_*.py ...`).

## The experiments

| | Question | Method | Result | Used by |
| --- | --- | --- | --- | --- |
| E1 `e1_curves/blender_curves.py` | How Blender computes automatic handles (both smoothing modes, clamped or not), easings, extrapolation and modifiers | 49 curve cases (uneven keys, extrema, plateaus, scaled values and times, cycles, free, vector and mixed handles): Blender recalculates the handles, `fcurve.evaluate` samples a dense grid | v0.19.1's local handles were far from Blender's (AUTO / None: up to 5.8 on a range of 7.2; AUTO / Continuous Acceleration with linear ends: 14.7 on 26.6, `compare_baseline.json`); 26 of its 30 eased cases already agreed within 2.4e-6, back eased in and out did not (0.97 on 13.5: Blender scales the overshoot by 1.525, as Penner), nor did elastic (below) | `aether3d.animation.smoothing`, `tests/test_curve_smoothing.py`, fixture `tests/data/blender_curves` |
| E1b `e1_curves/blender_curves_b.py` | Clamping under Continuous Acceleration, cyclic smoothing, mixed handle types, elastic defaults, the default smoothing preference | as E1 | Blender 5.2's preferences give new curves Continuous Acceleration and new keys automatic clamped handles (Aether3D's defaults: clamped automatic handles already, Continuous Acceleration since 0.20.0) | the same; `model.py` is the independent model written from E1 and E1b, `check_model.py` checks it |
| E1 modifiers `compare_modifiers.py` | Cycles, generator, built-in function, stepped, limits, ranged modifiers | keys at frames 1, 9, 17; Blender's default smoothing | Aether3D's new `cycles`, `generator` and `function` modifiers and its `stepped` and `limits` agree within 4e-6 (17 cases, one generator with a restricted frame range) | `test_modifiers_are_blenders` |
| E2 `e2_nla/blender_nla.py` | The shape of an NLA crossfade, blend types, how quaternion strips blend | a 170-degree turn crossfaded in over 20 frames, sampled every frame | the ramp is linear (its velocity steps at both ends); COMBINE reaches 42.5, 85 and 127.5 degrees at a quarter, a half and three quarters; REPLACE adds quaternion components (35.8, 85, 134.2 degrees; 0.737 long midway) | strip `blend_curve`; showcase C's quaternion chart |
| E3 `e3_lights/blender_lights.py` | Cycles' radiometry of every light type, units, spot cones, area shapes, spread, temperature | an orthographic top view of a large white Lambertian plane under one light; linear radiance read back from OpenEXR | watts map to Aether3D's units by fixed factors (sun `S / pi`, point and spot `P / (4 pi^2)`, area `P / pi^2`); spot blend is a smoothstep of the cosine; black bodies; sized point lights fall off as `d^2 / (d^2 + r^2)` | `rendering/light.py`, `blackbody.py`, `tests/test_lights.py`, fixture `tests/data/blender_lights` |
| E3b `e3_lights/blender_view.py` | The display transforms (Standard, AgX, Filmic) on exposure ramps; the angular profile of area-light spread | emissive patches from −8 to +10 stops, grey and seven colours, saved as 16-bit PNG; a spread light seen from many angles | AgX's grey ramp, its path to white and its colour behaviour; spread's profile `max(1 − tan a / tan h, 0)` normalized to keep the power | `research/2026-10-stage19/film_fit/fit.py` (the `film` view transform: RMS 0.008 of the display range, at most 0.038), `tests/test_film_output.py`; area-light spread |
| E5 `e5_camera/blender_camera.py` | Lens shift and depth of field | a tiny emissive sphere: its centroid gives the projection with shift, its blur disk the circle of confusion | shift moves the image by `shift` times its larger side; blur diameter `(f / N) \|1/d − 1/focus\|`, blades and ratio shape it | `rendering/camera.py`, `tests/test_camera_lens.py` |
| E6 `e6_environment/blender_env.py` | The orientation, seam and strength of equirectangular world images | a "UV image" (R = u, G = v) seen by tiny cameras along known directions; a white plane under a uniform image | `u = 0.5 − atan2(y, x) / 2 pi`, `v = 0.5 + asin(z) / pi`; the mapping node's rotation turns the lookup; strength 2 gives radiance 2 on the plane | `rendering/envimage.py`, `tests/test_environment_image.py`, fixture `tests/data/blender_env` |

## Measured, researched, unverified

- **Measured in Blender** (the table above): automatic handles in both smoothing modes,
  easings, extrapolation and five modifier types; NLA crossfades and quaternion
  blending; the radiometry and units of every light, spot cones, area shapes, spread and
  black bodies; AgX's response on grey and seven colour ramps; lens shift and depth of
  field; the equirectangular mapping and strength.
- **From published work, not measured in Blender**: the rough dielectric BSDF (Walter et
  al. 2007), GGX and Smith terms, the exact Fresnel equations; Sobol' sequences (Joe and
  Kuo's direction numbers) and Owen scrambling (Burley 2020); filter importance sampling;
  the à-trous and SVGF denoisers; spherical-harmonic irradiance; Lambert's polygon
  irradiance; representative-point highlights; Fritsch-Carlson monotone curves; Penner's
  easings; OpenEXR's file layout; H.264 and BT.709 through FFmpeg's documentation.
  Each is checked against closed forms or reference implementations of our own
  (furnace tests, quadratures, the brute-force ray caster), not against Blender.
- **Measured and deliberately different**: elastic easing. Blender's differs from
  Penner's equations, which Aether3D keeps (BSD): by up to 3.7 on a change of 10 with
  the default parameters, and its amplitude and period are in value units and frames
  where Aether3D's are relative to the segment. Matching it would mean reconstructing
  Blender's code rather than measuring it, so Aether3D's elastic stays Penner's and is
  documented as such. Sized point lights: Aether3D keeps a uniform sphere's irradiance
  where Blender's soft falloff gives `d^2 / (d^2 + r^2)` of it.
- **Not verified against Blender** (and documented as Aether3D's own behaviour):
  light linking's semantics; glass and its energy (Cycles was not compared; closed forms
  were); Cycles' adaptive sampling and its denoisers (Aether3D's are compared with
  converged renders of its own); motion blur's shutter shapes; the order and signs of
  Blender's Vector pass (Aether3D's `motion` pass is documented on its own terms); AgX on
  colours outside the seven measured ramps and on wide-gamut input.

## Licensing

Blender is distributed under the GNU GPL and its Cycles renderer under the Apache License
2.0. Stage 19 reused code from neither: the scripts here call Blender's public Python API
to produce measurements, and only those numbers entered the repository (as fixtures). The
`film` view transform is Aether3D's own construction fitted to measured outputs; no
OpenColorIO configuration, look-up table or curve data of Blender's was copied.

## REA

The reverse-engineering tool REA was available and was not used. Every question Stage 19
asked about Blender was observable from outside through `bpy` -- inputs in, numbers out --
so analysing Blender's binaries would have added nothing the measurements did not give,
and reconstructing its algorithms from its code would have gone against the clean-room
approach above. No REA evidence was produced.
