# GPU rendering

Aether3D 0.6 can render on graphics hardware. The GPU backend is a real
rasterization pipeline written for Aether3D:
- WGSL vertex, fragment and compute shaders;
- real draw calls, a hardware depth buffer, MSAA and shadow maps;
- framebuffer readback.

It runs through [wgpu](https://github.com/gfx-rs/wgpu), the WebGPU
implementation, on Vulkan, Metal or Direct3D 12. It is headless: no window,
desktop or display server is needed. Rendering a scene on the GPU takes the
same `scene.render()` call as on the CPU, with `backend="gpu"`. Aether3D's
NumPy CPU renderer (Stage 5) stays the default and the reference.

![The procedural castle: CPU | GPU | GPU with shadows](../examples/output/castle_comparison.png)

*The Stage 5 procedural castle, rendered by Aether3D three ways from one scene and
camera
([`examples/gpu_render.py`](../examples/gpu_render.py)):*
- *the CPU renderer, 2×2 samples per pixel;*
- *the GPU, 4× MSAA;*
- *the GPU, 4× MSAA with a sun shadow map.*

- [Quick start](#quick-start)
- [Installing GPU support](#installing-gpu-support)
- [Choosing the backend and the adapter](#choosing-the-backend-and-the-adapter)
- [How the GPU backend works](#how-the-gpu-backend-works)
- [Antialiasing](#antialiasing)
- [Shadows](#shadows)
- [GPU memory and caching](#gpu-memory-and-caching)
- [How the GPU output relates to the CPU renderer](#how-the-gpu-output-relates-to-the-cpu-renderer)
- [Performance](#performance)
- [Errors, diagnostics and debugging](#errors-diagnostics-and-debugging)
- [Testing and CI](#testing-and-ci)
- [Why wgpu: the backend decision](#why-wgpu-the-backend-decision)
- [Limitations](#limitations)
- [Toward later stages](#toward-later-stages)

## Quick start

```python
from aether3d import Scene, gpu_status

scene = Scene("Studio")
scene.add_plane("Ground", size=20).color = "#6f9b4f"
scene.add_cube("Crate", size=2, position=(0, 0, 1)).color = "#c0392b"
scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1.0), intensity=2.0)
scene.add_camera("Camera", fov=40).frame(direction=(-0.5, 1.0, -0.45))

print(gpu_status()["available"])  # can this machine render on the GPU?
result = scene.render(backend="gpu", width=1920, height=1080, samples=2, shadows=True)
result.save("studio.png")
print(result.backend["adapter"]["name"], result.backend["antialiasing"])
print(result.stats["timings"], result.stats["gpu_timings"])
```

The same steps from the terminal:

```bash
aether3d gpu info --check   # adapters, capabilities and a verified test render
aether3d render castle.aether --backend gpu --width 1920 --height 1080 \
    --samples 2 --shadows --output castle_gpu.png --json
```

- **What you get back.** `result` is the same `RenderResult` as for the CPU:
  - `image`;
  - the `linear`, `depth`, `coverage` and `object_index` maps;
  - per-object `visible_samples`, `stats` and `diagnostics`;
  - `backend`, which says what rendered it: the adapter, graphics API,
    antialiasing, shadow maps and resource statistics.
- **Provenance in the file.** GPU-rendered PNGs record the adapter in a
  standard `Source` text chunk.

## Installing GPU support

GPU support is optional. It needs Python 3.11 or newer and the `gpu` extra,
which installs [wgpu-py](https://github.com/pygfx/wgpu-py). Its wheels include
wgpu-native, so no compiler or SDK is required.

```bash
pip install "aether3d[gpu]"      # or, in a checkout: pip install -e ".[dev]"
aether3d gpu info --check
```

| Dependency (all permissive) | License | Notes |
| --- | --- | --- |
| `wgpu` (wgpu-py) 0.32.x | BSD-2-Clause | Python bindings (cffi). Wheels for Linux x86-64/aarch64, macOS x86-64/arm64, and Windows x86/x64/ARM64. Requires Python ≥ 3.11 |
| wgpu-native 29 (bundled in the wheel) | MIT OR Apache-2.0 | The Rust WebGPU implementation (wgpu-core, naga) |
| `cffi`, `pycparser` | MIT-0 / BSD-3-Clause | Installed by wgpu |
| `rendercanvas` | BSD-2-Clause | Installed by wgpu; Aether3D never opens a canvas |

- **Pinned minor series.** wgpu-py's API still changes between minor releases,
  so Aether3D pins the tested series (`>=0.32,<0.33`).
- **No GPU on Python 3.10.** On 3.10 the extra installs nothing, and GPU
  renders report `GPU_UNAVAILABLE` with the reason. Everything else works on
  3.10.
- **The core stays independent.** It never imports wgpu: `import aether3d`,
  the CPU renderer and the rest of the engine work without it.

The graphics driver is the part that differs by platform:

### Arch Linux

```bash
sudo pacman -S vulkan-icd-loader vulkan-tools                 # the Vulkan loader, vulkaninfo
sudo pacman -S nvidia-utils          # NVIDIA (with nvidia or nvidia-open)
sudo pacman -S vulkan-intel          # Intel integrated graphics
sudo pacman -S vulkan-radeon         # AMD
sudo pacman -S vulkan-swrast         # Mesa lavapipe: a software Vulkan adapter (CPU)
sudo pacman -S vulkan-validation-layers   # optional: for AETHER3D_GPU_DEBUG=1
vulkaninfo --summary                 # what the Vulkan loader sees
aether3d gpu info --check            # what Aether3D can use
```

**Hybrid laptops (NVIDIA + Intel or AMD).** Vulkan lists every GPU that has a
driver, so no PRIME offload variables are needed: Aether3D enumerates the
adapters through wgpu and picks the discrete GPU by default. Notes:
- **Intel needs its own Vulkan driver.** Without `vulkan-intel` installed, the
  Intel GPU is not listed at all (only through OpenGL).
- **Choosing explicitly:** `--adapter nvidia`, `--adapter integrated`,
  `--adapter llvmpipe` or `AETHER3D_GPU_ADAPTER=...`.
- **Power management.** The NVIDIA GPU may need to wake up (power state P8 to
  P0). The first device creation then takes a few hundred milliseconds.

The reference machine for this stage reports:
- `[0] NVIDIA GeForce RTX 4060 Laptop GPU (vulkan, discrete)` (driver 615.78.08), the auto choice;
- `[1] llvmpipe (vulkan, software)`;
- `[2] NVIDIA … (opengl)`, which cannot create a device (see Troubleshooting).

### Debian and Ubuntu

```bash
sudo apt install libvulkan1 vulkan-tools mesa-vulkan-drivers   # Mesa (Intel, AMD, lavapipe)
sudo apt install nvidia-driver-570                             # NVIDIA (or the recommended version)
```

### Windows

Install the GPU vendor's current driver; it provides Direct3D 12 and Vulkan.
wgpu prefers Direct3D 12 on Windows. `aether3d gpu info` lists both APIs for
the same GPU, and `--adapter vulkan` selects the other one. Microsoft's WARP
software rasterizer appears as a software adapter where Direct3D 12 offers it.

### macOS

Metal is built in (macOS 11 or newer on Apple silicon, 10.13 or newer on
Intel); nothing else is needed.

### Containers, virtual machines, cloud and remote sessions

A GPU is only usable where its driver can reach the device:
- **Docker:** use the NVIDIA Container Toolkit
  (`docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility ...`).
  Vulkan also needs the vendor ICD file inside the container.
- **Virtual machines:** need GPU passthrough or a virtual GPU with a Vulkan,
  Metal or Direct3D 12 driver.
- **SSH and headless servers:** work, because nothing needs a display.
- **Cloud and CI runners without a GPU:** usually have none, so they only get
  a software adapter (Mesa lavapipe, WARP) or no adapter at all.

`aether3d gpu info` always reports what was found and why a device could not
be created. Aether3D never reports a GPU as available because a card is
installed: an adapter counts as usable only after a device was created on it
and the shaders compiled. **Software adapters are labeled** (`"software": true`
in reports, and a `SOFTWARE_ADAPTER` diagnostic on every render): they run the
full GPU pipeline correctly but on the CPU, so they are not evidence of
hardware speed.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `GPU_UNAVAILABLE: ... 'wgpu' package, which is not installed` | `pip install "aether3d[gpu]"` (Python ≥ 3.11) |
| `GPU_UNAVAILABLE: wgpu found no graphics adapter` | No Vulkan, Metal or Direct3D 12 driver. Install one, or lavapipe for a software adapter |
| `GPU_DEVICE_FAILED` for an OpenGL adapter | wgpu's OpenGL backend may get a GL 3.3 context without compute shaders; use the Vulkan adapter (`--adapter vulkan`). `auto` already skips it |
| The NVIDIA GPU is missing on a hybrid laptop | `vulkaninfo --summary` must list it; install `nvidia-utils` and check that the NVIDIA kernel module is loaded |
| `GPU_DEVICE_LOST` | The driver reset the GPU (timeout, sleep/resume, GPU switch); render again, which creates a new device |
| `GPU_FEATURE_UNSUPPORTED` for `msaa` | The device cannot multisample that count; `aether3d gpu info` lists the supported counts (1 and 4 always work) |
| `GPU_LIMIT_EXCEEDED` | The image (times `supersample`) or shadow map is larger than the device's largest texture |

## Choosing the backend and the adapter

| Option | Python | CLI | Meaning |
| --- | --- | --- | --- |
| backend | `backend="cpu"` (default) | `--backend cpu` | The NumPy rasterizer: always available, no shadows |
| | `backend="gpu"` | `--backend gpu` | The GPU, or a `GPU_*` error. It never falls back silently |
| | `backend="auto"` | `--backend auto` | The GPU if it works, else the CPU with a `BACKEND_FALLBACK` diagnostic |
| adapter | `adapter=...` | `--adapter ...` | Default: `$AETHER3D_GPU_ADAPTER`, else the best adapter |

**What `auto` falls back on.** It uses the CPU only for GPU problems (codes
starting with `GPU_`). Invalid requests, such as a bad adapter selection or a
bad shadow setting, are reported as errors in every mode. When it does fall
back, options the CPU cannot honor are named in diagnostics:
- `GPU_OPTIONS_IGNORED` for `msaa` and `supersample`;
- `SHADOWS_NOT_RENDERED` for shadows.

The result's `backend["name"]` always says which backend actually ran.

**Adapters** are enumerated through wgpu (`aether3d.list_adapters()`,
`aether3d gpu info`). The CPU never guesses from device names or vendor
tools. `adapter=` accepts:

| Selection | Picks |
| --- | --- |
| `None`, `"auto"` | The best usable adapter: discrete GPUs first, then integrated, virtual, unknown and software ones. Within each, the platform's native API comes first (Vulkan on Linux, Metal on macOS, Direct3D 12 on Windows), and OpenGL last |
| `0`, `"2"` | The adapter with that index in `list_adapters()` |
| `"discrete"`, `"integrated"`, `"cpu"`, `"software"`, `"hardware"` | The best adapter of that kind |
| `"vulkan"`, `"metal"`, `"d3d12"` (`"dx12"`), `"opengl"` (`"gl"`) | The best adapter of that API |
| `"nvidia"`, `"rtx 4060"`, `"llvmpipe"`, `"intel"` | Adapters whose name contains the text |
| `"name=nvidia,backend=vulkan"`, `"hardware,vulkan"` | Every criterion must match (keys `index`, `name`, `type`, `backend`, `software`) |

If several adapters match, they are tried in that order until one creates a
working device. Adapters that failed are reported as `ADAPTER_FALLBACK`.

```python
from aether3d import gpu_status, list_adapters, release_gpu

for adapter in list_adapters():  # dicts: index, name, type, backend, driver,
    print(adapter["index"], adapter["name"], adapter["recommended"])  # features, limits
status = gpu_status(adapter="nvidia", check=True)  # never raises; renders a self-test
print(status["available"], status["selected"]["msaa"], status["check"]["ok"])
release_gpu()  # free cached devices and GPU memory
```

`aether3d gpu info --check` runs the self-test from
[`aether3d/rendering/gpu/check.py`](../src/aether3d/rendering/gpu/check.py):
- it renders a small scene on the GPU, in parity mode and with MSAA and
  shadows;
- it reads the images back;
- it verifies coverage, visible-sample counts, colors and shadows against the
  CPU renderer.

It exits with code 3 if GPU rendering does not work.

## How the GPU backend works

```text
Scene ─extract─▶ RenderScene ──────────────▶ GPU backend (aether3d/rendering/gpu/) ─▶ RenderResult
   (the same, backend-independent data the CPU renderer uses; nothing is evaluated twice)

per render, one command buffer:
  shadow pass(es)  depth of every instance seen from each shadowed directional light
                   (orthographic, depth32float array layer, depth-only pipeline)
  scene pass       every instance, msaa samples per pixel of a supersample x larger image:
                   linear radiance (rgba32float), object index (r32float), reversed depth
                   (depth32float, compare "greater"), hardware back/front-face culling
  finalize pass    compute, one invocation per output pixel: average the samples in linear
                   light (the resolve), exposure, exact 8-bit sRGB, view depth, object index,
                   coverage, visible-sample counts (atomics)
  copy             outputs -> mappable buffer; mapping it waits for the GPU to finish
```

| Module | Responsibility |
| --- | --- |
| `rendering/backends.py` | Backend names, option validation, the antialiasing plan |
| `rendering/shadows.py` | Backend-neutral shadow planning: which lights, and how each map is fitted |
| `rendering/gpu/device.py` | wgpu loading, adapter enumeration, ranking and selection, device creation, the per-device `GPUContext`, mapping wgpu errors to `RenderError` |
| `rendering/gpu/shaders.py` | WGSL sources, compilation, bind group layouts, the pipeline cache |
| `rendering/gpu/wgsl/*.wgsl` | The shaders: `common` (instances, vertex pulling), `scene`, `shadow`, `finalize` |
| `rendering/gpu/resources.py` | Mesh buffer cache, reusable targets, exact byte layouts of all shader data |
| `rendering/gpu/renderer.py` | The passes, synchronization, readback and statistics |
| `rendering/gpu/check.py` | The self-test |

Design decisions:

- **One scene description for both backends.** The GPU consumes the same
  `RenderScene` as the CPU renderer, so they render exactly the same content:
  - every geometry source (native, glTF-imported, edited, procedural);
  - the *current* procedural evaluation, never baked;
  - hidden objects skipped, shared meshes as instances;
  - the same world, normal matrices, lights and camera.
- **Vertex pulling.** Draws are non-indexed: three vertices per triangle. The
  vertex shader reads the triangle's indices, positions and normals from
  storage buffers, so every vertex knows its whole triangle. That gives:
  - exact face normals for flat-shaded meshes, from the object-space cross
    product through the normal matrix, which stays outward under mirroring;
  - one pipeline for meshes with and without normals;
  - no duplicated vertex buffers.
- **Camera-relative coordinates.** The CPU computes each instance's model
  matrix *minus the eye position* in float64 before rounding to float32.
  Objects a million units from the origin therefore render as precisely as
  objects at the origin (tested).
- **Reversed depth.** The depth row of the projection maps the near plane to
  1 and the far plane to 0, with a float32 depth buffer. Relative depth
  precision is then nearly constant over the whole range, including the
  `far / near` ratios up to 10⁹ that cameras allow.
- **Faces and mirroring.** The rasterizer decides which side of a triangle
  faces the camera. Shading flips normals toward the camera (two-sided
  lighting, as on the CPU), and mirrored instances, whose winding is
  reversed, are accounted for. With culling, mirrored instances use a
  pipeline whose front face is clockwise, so `cull="back"` removes the same
  faces as on the CPU.
- **Instancing.** Instances that share a mesh (and winding, when culling)
  become one instanced draw call: 10,000 instanced cubes are one draw.
- **Exact color encoding.** The finalize pass converts linear light to 8-bit
  sRGB by counting precomputed float32 thresholds, not with the GPU's `pow`.
  Each threshold is the smallest float32 whose sRGB value rounds up to the
  next code, computed in float64. For every float32 value, the GPU therefore
  produces exactly the CPU renderer's `round(255 * srgb(clamp(v)))` (tested
  on 250,000 values).
- **Synchronized timing.** Readback maps the output buffer, which waits for
  the GPU to finish. Times never stop at command submission. GPU timestamp
  queries additionally time each pass on the GPU itself when the device
  supports them.
- **Shader layout.** Shading is split so that materials (Stage 7) only have
  to replace one function:
  - `surface()` returns what an object looks like at a fragment: base color,
    highlight;
  - `shade()` applies the lights;
  - `shadow_visibility()` evaluates the shadow maps.

  Shaders are compiled from the bundled `.wgsl` files only; nothing from a
  scene, project or command line is ever compiled.

## Antialiasing

`samples` keeps its meaning on both backends: `samples × samples` samples per
pixel, averaged in linear light. The GPU reaches that count with hardware
multisampling where it can:

| `samples` | GPU default (`msaa`, `supersample`) | Samples per pixel |
| --- | --- | --- |
| 1 | 1, 1 | 1 |
| 2 (default) | 4, 1 (4× MSAA) | 4 |
| 3 | 1, 3 (3×3 supersampling) | 9 |
| 4 | 4, 2 (4× MSAA on 2×2 supersampling) | 16 |

**Choosing explicitly.** `msaa=` (1, 2, 4, 8 or 16) and `supersample=` (1–4)
override the defaults. Giving only one of them sets the other to 1.
- **MSAA counts.** These are real device capabilities:
  - every WebGPU device supports 1 and 4;
  - others depend on the adapter (the RTX 4060: 1, 2, 4, 8; lavapipe: 1, 4,
    8);
  - an unsupported count is an error (`GPU_FEATURE_UNSUPPORTED`), never
    silently changed.
- **Supersampling** renders the image `supersample` times larger per axis and
  averages it on the GPU.
- **Color precision.** MSAA color is float32 (`rgba32float`) where the device
  can multisample it (both tested devices). Otherwise it falls back to
  `rgba16float`, with a relative precision of 2⁻¹¹. The format used is
  reported as `backend["resources"]["color_format"]`.

**Parity mode.** `supersample=samples, msaa=1` (for example
`samples=2, supersample=2, msaa=1`) samples exactly the CPU renderer's pixel
positions. This is the fair CPU/GPU comparison mode (shadows are off by
default).

MSAA differs from supersampling in two documented ways:
- **Sample positions.** It samples the device's standard pattern, not the
  CPU's grid, so edges differ slightly.
- **Shading rate.** It shades once per pixel and triangle, at the centroid of
  the covered samples, instead of once per sample. So:
  - inside triangles the colors agree;
  - silhouettes agree up to sample positions;
  - sharp highlights and facet creases are averaged differently.

  Supersampling is exact for those, at a higher cost.

## Shadows

```python
result = scene.render(backend="gpu", shadows=True)  # all directional lights
result = scene.render(backend="gpu", shadows={"lights": ["Sun"], "map_size": 4096})
result = scene.render(backend="gpu", shadows=ShadowSettings(pcf_radius=2, bias=1.0))
print(result.backend["shadows"])  # settings, and per map: light, texel size, extent
```

```bash
aether3d render castle.aether --backend gpu --shadows --shadow-lights Sun --shadow-map-size 4096
```

| Setting | Default | Meaning |
| --- | --- | --- |
| `lights` | all directional lights | Light ids or names that cast shadows (at most 4) |
| `map_size` | 2048 | Texels per side of each shadow map (256–16384, within the device limit) |
| `bias` | 0.5 | Constant depth bias, in texels |
| `normal_offset` | 0.5 | Extra offset along the normal, in texels (see below) |
| `pcf_radius` | 1 | Percentage-closer filtering: (2r+1)² bilinear depth comparisons per lookup |

**What casts and receives.**
- **Lights:** directional lights cast shadows. Up to 4 per render, the
  brightest first when not named (`SHADOW_LIGHT_LIMIT` reports the others).
  Point lights do not yet (`POINT_LIGHT_SHADOWS_UNSUPPORTED`); naming one is
  an error.
- **Objects:** every visible object casts and receives shadows.
- **Ambient light** is not shadowed.
- **Persistence:** shadows are a render option, not saved in projects: the
  `.aether` format stays at version 3. **The CPU renderer has no shadows**:
  `shadows=` with `backend="cpu"` is an error (`BACKEND_UNSUPPORTED`).

**Fitting** (`rendering/shadows.py`, backend-neutral and unit-tested): one
orthographic map per light, recomputed every render.
1. **Light space** is measured from the camera's eye. Forward is the light's
   direction, `right = forward × up` (world +Z, or +Y for vertical lights),
   and `up = right × forward`.
2. **Casters:** the 8 world-space corners of every instance's bounding box,
   projected into light space.
3. **Receivers:** the camera frustum is cut off at the farthest geometry and
   projected the same way. Only receivers inside both can show a shadow. So:
   - the map covers the casters ∩ the view across the light, which is what
     the camera sees;
   - its depth range covers every caster, so casters outside the view still
     cast into it.
4. **Square texels:** the covered area is made square and widened by exactly
   `pcf_radius + 2` texels on each side.

The texel size in world units is reported per map. A huge ground plane seen up
close does not waste the map; a very wide view of a large scene does get
coarse shadows (no cascades yet).

**Bias, with a guarantee.** The lookup point moves along the shading normal,
toward the light, by `k = √2·(pcf_radius + 1) + normal_offset` texels times
`sin θ` (θ is the angle between the normal and the light).
- **Why this is enough.** On a flat receiver, that lifts the point `k·tan θ`
  texels above its own depth, everywhere within `k` texels. The filter reads
  at most `√2·(pcf_radius + 1)` texels away. So flat surfaces never shadow
  themselves, at any light angle.
- **The constant `bias`** covers depth-precision noise.
- **Peter-panning stays small.** Shadows start within a few texels of their
  casters.
- **Verified:** an analytic floating-box shadow test checks for acne-free lit
  ground (to the 8-bit code) and for the shadow's edges, for `pcf_radius` 0,
  1 and 2.

**Coarse smooth meshes (the shadow terminator).** A low-polygon mesh with
smooth normals is shaded as a smooth surface but casts the shadows of its
facets. Near the terminator, where the surface turns away from the light,
facets would shadow themselves in a jagged, faceted pattern. Exact ray tracing
of the faceted mesh produces that pattern too, but it does not belong on a
surface shaded as smooth.

Shadow lookups therefore start from the point on the smooth surface that the
vertex normals describe:
- each fragment is moved out of the tangent plane of every triangle corner it
  lies below, weighted by its barycentrics (J. Hanika, *Hacking the Shadow
  Terminator*, Ray Tracing Gems II, 2021);
- the normal offset is then applied along the shading normal;
- flat-shaded triangles are unaffected, because the lift is zero there.

What the tests check (`tests/test_gpu_shadows.py`):
- **A coarse 48×12 torus** does not shadow itself: its pixels are within 2
  codes of the unshadowed render on more than 99.5% of the ring, while it
  still shadows the ground.
- **A 96×48 torus under a low sun** does shadow itself: the near side of the
  ring shades the far side. Those self-shadows agree with exact shadow rays.

The cost of both offsets: self-shadows from occluders closer than the offset
(about three texels) at grazing light are not drawn. On the 48×12 torus at
320×180 that is a 34-pixel band along the terminator, the facet shadow that
exact ray tracing draws there; on a 96×48 torus, 1 pixel.

**Verified against ray tracing.** An independent ray tracer
(`shadow_mask` in `tests/render_reference.py`: shadow rays tested against every
triangle) is compared with the GPU on:
- the procedural castle;
- a rotated, non-uniformly scaled and mirrored hierarchy;
- the self-shadowing torus;
- a scene with two shadowed lights, where the GPU image matches
  `ambient + Σ unshadowed light × light's contribution` per pixel.

Disagreement is allowed only within the documented boundary band (the filter
width plus one pixel). On the castle at 240×135, 0 of 5,921 compared pixels
disagree with the default settings, on both the RTX 4060 and llvmpipe.

## GPU memory and caching

- **Meshes** are uploaded once per device and shared by every instance (and
  every render) that uses them:
  - **Identity:** the cache is keyed by the identity of the mesh's read-only
    NumPy arrays. A changed procedural parameter or a committed edit produces
    a new mesh object and therefore a new upload, so a stale mesh is never
    drawn (tested).
  - **Release:** entries hold weak references, so a mesh that is garbage
    collected frees its GPU buffers at the next render.
  - **Writable arrays:** arrays that were made writable again are never
    cached.
  - **Budget:** least-recently-used meshes beyond `AETHER3D_GPU_CACHE_MB`
    (default 1024 MB) are evicted, never those of the render in progress.
- **Targets**:
  - **What they are:** the multisampled color, object-index and depth images,
    the shadow-map array, and the output and staging buffers.
  - **Reuse:** they are kept while consecutive renders keep the same size and
    settings.
  - **Limit:** one render's estimated target memory must stay within
    `RenderLimits.max_gpu_memory` (default 4 GiB). This is checked before
    allocating, as are the device's texture and binding limits
    (`GPU_LIMIT_EXCEEDED`).
- **Pipelines and shader modules** are compiled once per device.
- **Statistics:** `result.backend["resources"]` reports:
  - this render's mesh uploads (count, bytes);
  - the mesh cache (meshes, bytes, budget, hits, evictions);
  - target memory and whether targets were reallocated;
  - frame-buffer bytes, color format and pipeline count.

  `result.stats` adds `draw_calls` and `shadow_maps`.
- **Devices** live as long as the process, and `release_gpu()` frees them.

## How the GPU output relates to the CPU renderer

| | Parity mode (`supersample=s, msaa=1`) | Default 4× MSAA |
| --- | --- | --- |
| Object seen at each pixel | Identical except samples exactly on triangle edges (hardware snaps vertices to 1/256 pixel): mismatches only on object boundaries, < 0.5% (castle 1080p: 100% identical on both test devices) | Same objects; ≥ 98% of pixels identical, the rest along edges |
| 8-bit colors away from edges | ≤ 2 codes (measured: 1) | Flat areas ≤ 2; highlights and creases within their local variation |
| Linear radiance | ≤ 2·10⁻³ (measured 2·10⁻⁴ RTX 4060, 3·10⁻⁶ llvmpipe) | — |
| Depth | Relative ≤ 10⁻³ (measured ≤ 2·10⁻⁴ on the RTX 4060, ≤ 5·10⁻⁶ on llvmpipe); orthographic depth has absolute resolution (far − near)/2²⁴ | — |
| Visible samples per object | Within 1% (measured: identical) | Within 1% of all samples |

- **Where the differences come from.** Hardware rasterizers snap vertices to
  their own sub-pixel grid and compute attributes in float32. The RTX 4060's
  interpolated attributes reflect that snapping (about 10⁻⁴); llvmpipe is
  closer.
- **Shading model.** Lighting is the Stage 5 shading model in float32. The GPU
  `pow` and `normalize` differ from float64 NumPy in the last bits.
- **Exact ties.** Exact depth ties between different objects (coplanar
  surfaces) are resolved by draw order on the GPU, but by triangle index on
  the CPU.
- **Determinism.** Renders on one device are bit-identical from run to run
  (tested), and the CLI reproduces the Python API's PNG exactly. Different
  GPUs or drivers can differ within these tolerances.
- **Same lighting rules.** Two-sided lighting, the default key light,
  exposure, the background and the sRGB policy are the same as on the CPU.

## Performance

Measured with `python benchmarks/bench_gpu.py` on the development laptop:
- **CPU:** Intel Core i9-12900H (20 logical CPUs);
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, Vulkan, driver 615.78.08;
- **Software:** Arch Linux (kernel 7.2.9), Python 3.14.7, NumPy 2.5.4,
  wgpu-py 0.32.0, wgpu-native 29.0.1.1.

Every row renders the same scene and camera at 4 samples per pixel: the CPU
with 2×2 supersampling, the GPU with 4× MSAA. Times are wall-clock seconds for
a complete `scene.render()`:
- the GPU: best of 5 after 3 warm-up renders at full size;
- the CPU: best of 3 (a single run at 4K).

Each time covers:
- including extraction, uploads, command preparation, GPU execution and
  readback into NumPy arrays (synchronized);
- excluding PNG encoding, which is listed separately (the same code for both).

**RTX 4060 Laptop GPU (Vulkan)**, warm renders, best of 5 after 3 warm-up renders:

| Scene (triangles) | Size | CPU | GPU (4× MSAA) | Speedup | GPU, CPU sample positions | GPU + shadows | GPU passes | Readback | PNG encode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| castle (procedural) (6,804) | 640×360 | 153 ms | 7 ms | **20.8×** | 8 ms | 9 ms | 0.1 ms | 1 ms | 22 ms |
| castle (procedural) (6,804) | 1920×1080 | 1.24 s | 44 ms | **28.0×** | 52 ms | 45 ms | 5.3 ms | 13 ms | 187 ms |
| castle (procedural) (6,804) | 3840×2160 | 4.52 s | 89 ms | **50.8×** | 99 ms | 95 ms | 2.8 ms | 55 ms | 906 ms |
| hero (edited meshes) (458) | 640×360 | 159 ms | 7 ms | **24.0×** | 7 ms | 7 ms | 0.4 ms | 2 ms | 20 ms |
| hero (edited meshes) (458) | 1920×1080 | 1.38 s | 45 ms | **30.8×** | 46 ms | 41 ms | 7.7 ms | 15 ms | 181 ms |
| hero (edited meshes) (458) | 3840×2160 | 5.61 s | 89 ms | **63.1×** | 99 ms | 95 ms | 2.4 ms | 59 ms | 874 ms |
| 10,000 instanced cubes (120,002) | 640×360 | 647 ms | 139 ms | **4.7×** | 141 ms | 159 ms | 2.0 ms | 1 ms | 26 ms |
| 10,000 instanced cubes (120,002) | 1920×1080 | 1.91 s | 214 ms | **8.9×** | 216 ms | 230 ms | 21.1 ms | 14 ms | 289 ms |
| 10,000 instanced cubes (120,002) | 3840×2160 | 6.24 s | 230 ms | **27.2×** | 226 ms | 255 ms | 11.6 ms | 57 ms | 952 ms |
| sphere ~10k triangles (9,802) | 640×360 | 75 ms | 6 ms | **13.5×** | 6 ms | 6 ms | 0.2 ms | 1 ms | 15 ms |
| sphere ~10k triangles (9,802) | 1920×1080 | 642 ms | 39 ms | **16.5×** | 41 ms | 39 ms | 3.2 ms | 15 ms | 194 ms |
| sphere ~10k triangles (9,802) | 3840×2160 | 2.61 s | 86 ms | **30.3×** | 88 ms | 92 ms | 1.8 ms | 57 ms | 853 ms |
| sphere ~100k triangles (101,762) | 640×360 | 121 ms | 6 ms | **20.1×** | 6 ms | 8 ms | 1.0 ms | 1 ms | 16 ms |
| sphere ~100k triangles (101,762) | 1920×1080 | 696 ms | 66 ms | **10.6×** | 41 ms | 42 ms | 8.5 ms | 18 ms | 200 ms |
| sphere ~100k triangles (101,762) | 3840×2160 | 2.72 s | 89 ms | **30.4×** | 89 ms | 88 ms | 1.9 ms | 59 ms | 867 ms |
| sphere ~1M triangles (998,002) | 640×360 | 471 ms | 14 ms | **34.1×** | 15 ms | 30 ms | 7.1 ms | 2 ms | 19 ms |
| sphere ~1M triangles (998,002) | 1920×1080 | 1.06 s | 47 ms | **22.4×** | 28 ms | 43 ms | 9.1 ms | 16 ms | 224 ms |
| sphere ~1M triangles (998,002) | 3840×2160 | 3.11 s | 90 ms | **34.5×** | 94 ms | 110 ms | 3.2 ms | 60 ms | 959 ms |

- **Start-up:** a new process needs about 0.8 s before its first GPU image: 0.09 s to import Aether3D and 0.63 s to load wgpu, enumerate the adapters, create the device and compile the pipelines (measured the same on llvmpipe). After that, a render takes milliseconds, so keep one process for interactive work. Re-creating a device inside a running process takes 0.06 s.
- **Memory, castle at 1080p:** peak host memory 221 MiB on the CPU renderer, 56 MiB on the GPU path. The GPU itself holds 301 MiB of images, buffers and meshes (1203 MiB at 4K).
- **Where a warm GPU frame's time goes (castle, 1080p, 44 ms):**
  - GPU passes: 5.3 ms;
  - waiting for the GPU, including the 58 MB output copy: 24 ms;
  - reading the outputs into NumPy: 13 ms;
  - extraction and uploads: 1.6 ms.

**Mesa llvmpipe (software Vulkan on the same CPU: not hardware speed)**, the adapter the CI uses:

| Scene (triangles) | Size | CPU | GPU (4× MSAA) | Speedup | GPU, CPU sample positions | GPU + shadows | GPU passes | Readback | PNG encode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| castle (procedural) (6,804) | 640×360 | 151 ms | 28 ms | **5.4×** | 27 ms | 39 ms | 18.6 ms | 2 ms | 22 ms |
| castle (procedural) (6,804) | 1920×1080 | 1.13 s | 109 ms | **10.4×** | 123 ms | 115 ms | 74.2 ms | 18 ms | 217 ms |
| hero (edited meshes) (458) | 640×360 | 165 ms | 14 ms | **11.7×** | 17 ms | 18 ms | 8.7 ms | 2 ms | 23 ms |
| hero (edited meshes) (458) | 1920×1080 | 1.22 s | 74 ms | **16.4×** | 90 ms | 86 ms | 51.6 ms | 14 ms | 168 ms |
| 10,000 instanced cubes (120,002) | 640×360 | 638 ms | 205 ms | **3.1×** | 164 ms | 262 ms | 71.4 ms | 1 ms | 24 ms |
| 10,000 instanced cubes (120,002) | 1920×1080 | 1.79 s | 252 ms | **7.1×** | 242 ms | 323 ms | 97.7 ms | 15 ms | 222 ms |
| sphere ~10k triangles (9,802) | 640×360 | 76 ms | 34 ms | **2.2×** | 25 ms | 42 ms | 29.4 ms | 1 ms | 15 ms |
| sphere ~10k triangles (9,802) | 1920×1080 | 588 ms | 81 ms | **7.3×** | 86 ms | 83 ms | 58.2 ms | 14 ms | 171 ms |
| sphere ~100k triangles (101,762) | 640×360 | 114 ms | 220 ms | **0.5×** | 148 ms | 262 ms | 215.8 ms | 1 ms | 14 ms |
| sphere ~100k triangles (101,762) | 1920×1080 | 641 ms | 237 ms | **2.7×** | 207 ms | 275 ms | 213.1 ms | 14 ms | 173 ms |
| sphere ~1M triangles (998,002) | 640×360 | 427 ms | 1.61 s | **0.3×** | 1.37 s | 1.94 s | 1604.3 ms | 2 ms | 14 ms |
| sphere ~1M triangles (998,002) | 1920×1080 | 972 ms | 1.67 s | **0.6×** | 1.45 s | 1.99 s | 1650.3 ms | 14 ms | 166 ms |

How to read these numbers:
- **Cold start.** The first GPU render in a process also creates the device
  and compiles the pipelines.
- **What costs most.** The GPU passes take a few milliseconds. Most of a warm
  GPU render is returning every per-pixel output to NumPy (28 bytes per
  pixel: the image, linear color, depth, object index and coverage), plus
  Python-side extraction for scenes with many objects. Those costs are the
  same however fast the GPU draws, so its advantage is smaller for small
  images and many-object scenes.
- **PNG encoding.** Aether3D's own PNG encoder runs on the CPU and costs more
  than a whole GPU render at 1080p and above; `result.image` is available
  without it.
- **Software adapters.** lavapipe runs the same pipeline on the CPU. It is
  correct, and faster than the NumPy renderer for large images, but not
  hardware speed.

## Errors, diagnostics and debugging

Errors (`RenderError.code`), in addition to those in
[RENDERING.md](RENDERING.md#error-and-diagnostic-codes):

| Code | Meaning |
| --- | --- |
| `GPU_UNAVAILABLE` | wgpu is not installed (or needs Python ≥ 3.11), cannot load, or finds no adapter |
| `GPU_ADAPTER_NOT_FOUND` | No adapter matches `adapter=`; `details["adapters"]` lists them |
| `GPU_DEVICE_FAILED` | No matching adapter could create a working device; `details["attempts"]` |
| `GPU_FEATURE_UNSUPPORTED` | The device cannot do what was asked (an MSAA count); `details["supported_msaa"]` |
| `GPU_LIMIT_EXCEEDED` | Larger than the device's limits (texture size, mesh buffer, object count) |
| `GPU_OUT_OF_MEMORY` | The GPU ran out of memory |
| `GPU_DEVICE_LOST` | The device was lost (driver reset, sleep, GPU switch); the next render creates a new one |
| `GPU_SHADER_ERROR`, `GPU_PIPELINE_ERROR`, `GPU_RESOURCE_ERROR`, `GPU_ERROR` | wgpu or driver errors, with the compiler or driver message in `details` |
| `BACKEND_UNSUPPORTED` | Shadows requested from the CPU renderer |
| `INVALID_SHADOWS`, `INVALID_SETTINGS` | Bad shadow, `backend`, `adapter`, `msaa` or `supersample` values |
| `RESOURCE_LIMIT` | `RenderLimits` exceeded (also `max_gpu_memory`) |

Diagnostics: `BACKEND_SELECTED`, `BACKEND_FALLBACK`, `GPU_OPTIONS_IGNORED`,
`SHADOWS_NOT_RENDERED`, `SOFTWARE_ADAPTER`, `ADAPTER_FALLBACK`,
`SHADOW_LIGHT_LIMIT`, `POINT_LIGHT_SHADOWS_UNSUPPORTED`, `SHADOW_FIT_FAILED`,
`GPU_VALIDATION_MESSAGE` and `NONFINITE_GEOMETRY` (objects beyond float32
range relative to the camera are skipped).

When something fails:
- **No partial files.** Renders that fail raise before anything is written:
  `save()` is atomic and the project is never modified.
- **Recovery.** A lost device is replaced at the next render.

**Validation.** `AETHER3D_GPU_DEBUG=1` turns on wgpu's debug and validation
instance flags. On Vulkan, that loads the Khronos validation layer when it is
installed (verified with `VK_LOADER_DEBUG=layer`).
- **Where messages go.** Validation and driver messages arrive through wgpu's
  log. Each one logged during a render becomes a `GPU_VALIDATION_MESSAGE`
  diagnostic.
- **The test suite** renders every pipeline variant under validation and
  requires none.

Other environment variables:
- `AETHER3D_GPU_BACKENDS=vulkan,metal,d3d12,opengl` limits which APIs wgpu
  tries;
- `AETHER3D_GPU_ADAPTER` sets the default adapter;
- `AETHER3D_GPU_CACHE_MB` sets the mesh cache budget.

## Testing and CI

| Tests | Needs | What |
| --- | --- | --- |
| `test_gpu_selection.py`, `test_gpu_data.py`, `test_shadow_planning.py` | nothing | Adapter ranking and selection (a simulated hybrid laptop), options, fallback without wgpu, WGSL struct layouts computed from the shader source vs the packed bytes, exact sRGB encoding, reversed depth, shadow fitting and light selection |
| `test_gpu_render.py` | a GPU adapter | Parity with the CPU renderer and with the independent ray caster, analytic Lambert/point-light/specular/exposure/projection values, every geometry source, procedural changes and caching, culling with mirrored instances, the camera inside geometry, 10⁷-unit and 10⁶-offset coordinates, degenerate triangles, 2,500 objects, MSAA |
| `test_gpu_shadows.py` | a GPU adapter | Analytic box shadows (acne, edges), filtering and map size, ray-traced shadows for the castle and a mirrored hierarchy, two lights and the shading equation |
| `test_gpu_faults.py` | a GPU adapter | Unknown adapters, unsupported MSAA, device and memory limits, shader errors, device loss and recovery, no files on failure, a validation-layer run |
| `test_gpu_cli.py` | a GPU adapter | `render --backend gpu` reproducing the Python API exactly, shadow and MSAA flags, `gpu info --check` |
| `test_gpu_performance.py` | a hardware GPU | The GPU at least 3× faster than the CPU renderer on a 200,000-triangle scene (skipped on software adapters) |

Without an adapter, GPU tests are skipped and the skip reason says why.

Environment variables for running them:
- `AETHER3D_REQUIRE_GPU=1` makes the GPU tests fail instead of skip;
- `AETHER3D_TEST_GPU_ADAPTER` chooses the adapter (for example `llvmpipe`
  or `nvidia`);
- `AETHER3D_GPU_TESTS=skip` turns the GPU tests off.

CI jobs ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)):
- **The portable matrix** (Linux Python 3.10–3.14, macOS, Windows) runs
  everything except GPU rendering.
- **The required `GPU (ubuntu-latest, Vulkan lavapipe)` jobs** (Python 3.11 and 3.14):
  - install Mesa's lavapipe, a conformant software Vulkan implementation;
  - set `AETHER3D_REQUIRE_GPU=1`;
  - run the GPU tests with real draw calls, framebuffer readback and pixel
    assertions;
  - run the self-test, the GPU example and a benchmark smoke run;
  - upload the renders.

  GitHub's Linux runners have no graphics hardware, so this validates the
  graphics API path, not hardware speed.
- **The `GPU (windows-latest / macos-latest, native adapter if any)` jobs** use
  whatever the hosted runner provides: Direct3D 12 / WARP, or Metal. They
  report the adapter or the reason there is none.
- **Jobs without wgpu** (Python 3.10 and 3.13) check that everything else
  works and that `aether3d gpu info` explains the absence.

## Why wgpu: the backend decision

Requirements:
- real GPU rasterization on Linux, Windows and macOS;
- headless and offscreen;
- a dependable, permissively licensed Python integration;
- one implementation to maintain;
- a path to compute shaders, materials and ray tracing.

| Option | Verdict |
| --- | --- |
| **wgpu-py + wgpu-native (chosen)** | One API (WebGPU) on **Vulkan, Metal and Direct3D 12**. Headless by design: no surface needed, offscreen textures and buffer readback. WGSL validated by naga at compile time with clear errors. Compute shaders, timestamp queries and MSAA; ray queries and other native features are exposed where the hardware has them. Prebuilt wheels for all major platforms, BSD-2 / MIT-or-Apache. Used by pygfx and fastplotlib. It enumerated and selected the RTX 4060 (Vulkan) and lavapipe here. Drawbacks: Python ≥ 3.11, and a pre-1.0 Python API (pinned). The OpenGL backend is limited (it may lack compute), so it is ranked last |
| Vulkan directly (`vulkan` cffi bindings) | Very verbose. Needs a SPIR-V compiler (glslang or shaderc) at build or run time. macOS only through MoltenVK. Direct3D 12 and Metal would each need another implementation |
| Three native APIs (Vulkan + Metal through PyObjC + Direct3D 12 through comtypes) | Three renderers to write, test and keep identical; rejected for this stage |
| ModernGL / PyOpenGL (OpenGL 3.3+, EGL headless) | Mature and simple, but OpenGL is deprecated on macOS (stuck at 4.1, no compute shaders). No path to modern features, and headless contexts differ by platform |
| pygfx (built on wgpu) | A complete rendering engine: using it would make Aether3D's renderer someone else's, with its own scene model and shading. Aether3D uses the same low-level layer (wgpu) directly |
| Taichi, NVIDIA Warp, CUDA | Compute-focused; CUDA-only or without a rasterization pipeline |
| A Rust or C++ extension (wgpu-rs, Vulkan) | Premature: it would add a native build to every platform. The WGSL shaders and the design carry over unchanged if hot paths move to Rust later |

Verified on the development machine:
- adapter enumeration found the NVIDIA GPU (Vulkan and OpenGL) and lavapipe;
- devices were created on the Vulkan adapters;
- every texture format and MSAA count this backend uses was probed on both
  (`texture-adapter-specific-format-features` is requested when available);
- shaders compiled;
- headless rendering and readback worked;
- timestamps were confirmed in nanoseconds against wall-clock time.

## Limitations

- **Shadows:**
  - directional lights only (no point-light cube maps yet), at most 4 per
    render;
  - one map per light, without cascades, so wide views of large scenes get
    coarse shadows;
  - filtering is PCF only.
- **The GPU uses float32.** Very large scenes keep precision near the camera
  (camera-relative coordinates), but objects beyond 10³⁰ units from the camera
  are skipped (`NONFINITE_GEOMETRY`).
- **Mesh size.** A single mesh must fit the device's storage-buffer binding
  size (128 MiB on lavapipe, 2 GiB on the RTX 4060), which is about 11
  million vertices or triangles on lavapipe. Larger meshes report
  `GPU_LIMIT_EXCEEDED`; split them or use the CPU renderer.
- **Readback cost.** Every GPU render reads back all auxiliary maps (depth,
  object index, coverage, linear color). That keeps results identical in
  shape to the CPU's, but costs readback time at very large sizes.
- **Not covered by testing:**
  - macOS (Metal), Windows (Direct3D 12) and AMD or Intel GPUs: only on
    whatever the CI runners provide (see the PR / CI reports); the
    development machine has an NVIDIA GPU and lavapipe;
  - the Intel integrated GPU of the hybrid laptop: it has no Vulkan driver
    installed there;
  - the OpenGL backend: it could not create a device with compute shaders on
    the test machine.
- **Nothing beyond Stage 5's appearance model.** There are no textures,
  materials, reflections, transparency, global illumination or
  post-processing (Stages 7, 8 and 26).

## Toward later stages

- **Stage 7 (materials and textures).** `surface()` in `scene.wgsl` is the only
  place an object's look enters shading. A material system replaces it (with
  textures bound in a new bind group, and UVs pulled like normals) without
  touching lighting, shadows, passes or readback. Instance records have room
  for a material index.
- **Stage 8 (quality).**
  - Cascaded and point-light shadow maps extend `rendering/shadows.py` (fitting
    is backend-neutral) and the shadow-map array.
  - Better filtering replaces the PCF loop in `shadow_visibility()`.
  - Post-processing slots in between the scene and finalize passes, where
    linear HDR color is available.
- **Stage 26 (cinematic rendering).** wgpu exposes compute shaders, ray
  queries (where hardware supports them) and timestamp queries. A GPU path
  tracer can reuse the device, caches and `RenderScene`, and the test
  reference ray caster gives it a ground truth.
