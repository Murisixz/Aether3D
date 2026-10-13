# Installing Aether3D

Aether3D requires **Python 3.10 or newer** and **NumPy**. It has no other
runtime dependencies and needs no GPU or graphical environment.

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
`python-numpy python-pytest python-trimesh` with pacman and create the environment with
`python -m venv --system-site-packages .venv`, then run
`pip install -e . --no-deps`.

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
pip install .
```

This installs only the `aether3d` package, its `aether3d` command, and NumPy.
glTF import/export needs nothing else. trimesh and the Khronos validator are
used only by the tests.
