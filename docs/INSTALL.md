# Installing Aether3D

Aether3D requires **Python 3.10 or newer** and **NumPy**. It has no other
runtime dependencies and needs no GPU or graphical environment. Rendering on
a GPU is optional: it needs Python 3.11+, the `gpu` extra and a Vulkan,
Metal or Direct3D 12 driver (see [GPU rendering](#gpu-rendering-optional)).
JPEG textures are optional too: they need the `images` extra (PNG textures
need nothing). So is the MCP server for AI agents: it needs the `mcp` extra
(see [The MCP server](#the-mcp-server-optional)).

## Arch Linux

1. Install the system packages:

   ```bash
   sudo pacman -S --needed git python
   ```

2. Get the source:

   ```bash
   git clone https://github.com/Murisixz/Aether3D.git
   cd Aether3D
   ```

3. Create a virtual environment and install Aether3D in editable mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # fish: source .venv/bin/activate.fish
   pip install -e ".[dev]"
   ```

   Arch's system Python is marked *externally managed* (PEP 668), so
   `pip install` outside a virtual environment is refused. The virtual
   environment keeps Aether3D and its dependencies isolated from pacman-managed
   packages.

4. Verify the installation:

   ```bash
   pytest
   python examples/first_scene.py /tmp/first_scene.aether
   aether3d inspect /tmp/first_scene.aether
   ```

5. Optional: install the official Khronos glTF Validator, which the test
   suite uses to check exported files when it is available:

   ```bash
   sudo pacman -S --needed nodejs npm
   npm ci --prefix tools/gltf-validator
   pytest tests/test_khronos_validator.py
   python examples/gltf_roundtrip.py /tmp/gltf_demo --require-validator
   ```

   Without it, those tests are skipped (CI always runs them).

If you prefer pacman-managed dependencies, you can install
`python-numpy python-pytest python-trimesh python-manifold3d` with pacman and create the
environment with `python -m venv --system-site-packages .venv`, then run
`pip install -e . --no-deps`. (Arch's `python-manifold3d` 3.4.1 has been tested with
Aether3D; any version ≥ 3.4 is supported.)

## Other Linux distributions, macOS, Windows

```bash
git clone https://github.com/Murisixz/Aether3D.git
cd Aether3D
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Installing without the development tools

```bash
pip install .              # core: NumPy is the only dependency
pip install ".[boolean]"   # also solid booleans for procedural geometry
pip install ".[images]"    # also JPEG textures (Pillow)
pip install ".[gpu]"       # also GPU rendering (Python 3.11+)
pip install ".[mcp]"       # also the MCP server for AI agents (aether3d mcp serve)
```

The first command installs only the `aether3d` package, its `aether3d`
command, and NumPy. Everything else needs nothing more:
- glTF import and export;
- mesh editing;
- rendering, including PNG and HDR (`.hdr`, `.pfm`) output and the path
  tracer;
- materials, procedural textures and PNG textures (Aether3D decodes PNG
  itself);
- every procedural feature except the `boolean` modifier.

The `boolean` extra adds [manifold3d](https://github.com/elalish/manifold)
(Apache-2.0; prebuilt wheels for Linux, macOS and Windows on Python
3.10–3.14). Without it, a procedural boolean fails with
`BOOLEAN_BACKEND_UNAVAILABLE`, while projects that contain booleans still
load. The `dev` extra includes it, and so do the tests and the castle demo.
The `images` extra adds [Pillow](https://python-pillow.org) (MIT-CMU) to
decode JPEG textures. Without it, a JPEG texture fails with
`IMAGE_DECODER_MISSING` (renders keep the material's factors and report it).
PNG textures, and everything else, keep working. The `dev` extra includes it.
trimesh and the Khronos validator are used only by the tests, and so is
Pillow as an independent PNG decoder.

## The MCP server (optional)

The `mcp` extra installs the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(MIT), version 2.x, and its dependencies (pydantic, anyio, jsonschema, starlette, uvicorn,
an HTTP client, ...: MIT, BSD-3-Clause, Apache-2.0 or PSF licenses); the `dev` extra includes
it. `aether3d mcp serve` then serves
Aether3D to AI agents over stdio ([MCP.md](MCP.md)). Without the extra, the command says
how to install it, and everything else works.

To use it from Claude Code on Arch Linux, with Aether3D installed in `~/Aether3D/.venv`
as above:

```bash
mkdir -p ~/scenes/castle && cd ~/scenes/castle
claude mcp add aether3d -- ~/Aether3D/.venv/bin/aether3d mcp serve --workspace "$PWD"
claude mcp get aether3d          # Status: ✔ Connected
claude                           # then ask it to build something
```

## GPU rendering (optional)

The `gpu` extra installs [wgpu-py](https://github.com/pygfx/wgpu-py)
(BSD-2-Clause), whose wheels include wgpu-native (MIT OR Apache-2.0) for
Linux, macOS and Windows; it needs Python 3.11 or newer. The `dev` extra
includes it on Python 3.11+. Then check what the machine offers:

```bash
aether3d gpu info --check   # adapters, the one Aether3D would use, and a verified test render
```

A graphics driver is still required: on Arch Linux, `vulkan-icd-loader` plus
`nvidia-utils`, `vulkan-intel` or `vulkan-radeon` (and `vulkan-swrast` for a
software Vulkan adapter); on Windows the vendor driver; on macOS nothing
(Metal). [GPU.md](GPU.md#installing-gpu-support) has the per-platform
details, hybrid-laptop notes and troubleshooting. Without wgpu or a driver,
`backend="gpu"` reports `GPU_UNAVAILABLE` and everything else works.

## Path tracing

The CPU path tracer (`render(backend="path")`, see
[RENDER_QUALITY.md](RENDER_QUALITY.md)) needs nothing beyond the base
install: it is written in NumPy and runs on every platform Aether3D
supports, without a GPU. The GPU path tracer (`backend="gpu_path"`) needs
the `gpu` extra and a working adapter, like `backend="gpu"` (see GPU support
above).

On Linux it can trace with several CPU cores: it forks worker processes
(`path={"workers": 0}`, the default, picks one per CPU, up to 8, for renders
big enough to gain from it). On macOS and Windows renders use one process, and
`result.backend["parallel"]` says why. Worker processes need a little memory
each (about 100 MB per batch of 65,536 paths) and share the scene, its BVH
and its textures with the main process.
