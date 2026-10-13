# Aether3D

[![Tests](https://github.com/Murisixz/Aether3D/actions/workflows/tests.yml/badge.svg)](https://github.com/Murisixz/Aether3D/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Aether3D** is an open-source, AI-native, code-first 3D engine. Every operation
is available through a small, predictable Python API and a JSON-speaking CLI,
so it can be driven reliably by people and AI coding agents alike, with no
graphical interface required.

> **Status: foundation (v0.1.0).** This release provides the core scene system,
> transforms and hierarchy, basic primitive geometry, the native `.aether`
> project format, and a minimal CLI. Rendering, animation, physics, and other
> systems are intentionally **not** part of this stage.

## Quick start

```python
from aether3d import Scene

scene = Scene("First Scene")

cube = scene.add_cube(name="Cube", size=2)
cube.position = (0, 0, 1)

sphere = scene.add_sphere(name="Sphere", radius=1)
sphere.position = (3, 0, 1)

scene.save("example.aether")

loaded_scene = Scene.load("example.aether")
```

Hierarchy and world-space queries:

```python
pivot = scene.add_object(name="Pivot", position=(3, 0, 1))
pivot.rotation_euler = (0, 0, 90)  # degrees, XYZ order
moon = scene.add_sphere(name="Moon", radius=0.25, parent=pivot, position=(2, 0, 0))

moon.world_position  # ≈ (3, 2, 1)
moon.set_parent(None, keep_world_transform=True)
scene.get(moon.id) is scene.get_by_name("Moon")  # True
```

A complete, runnable example lives in [`examples/first_scene.py`](examples/first_scene.py).

## Command-line interface

```console
$ aether3d new world.aether --name "My World"
$ aether3d validate world.aether
$ aether3d inspect examples/first_scene.aether
Scene "First Scene" (5 objects, format v1)
├── Cube  [d0da5f96-…]  pos=(0, 0, 1)  mesh=24v/12t
├── Sphere  [f9788aa5-…]  pos=(3, 0, 1)  mesh=561v/960t
├── Ground  [efddd1b2-…]  pos=(0, 0, 0)  mesh=4v/2t
└── Pivot  [c1181286-…]  pos=(3, 0, 1)  rot_euler=(0, 0, 90)
    └── Moon  [8122a0f9-…]  pos=(2, 0, 0)  mesh=561v/960t
```

Add `--json` to any command for machine-readable output. Errors point at the
exact element that is wrong:

```console
$ aether3d validate broken.aether --json
{
  "ok": false,
  "command": "validate",
  "error": {
    "type": "ProjectFormatError",
    "message": "expected an array of 3 numbers",
    "location": "scene.objects[1].transform.scale"
  }
}
```

Exit codes: `0` success, `1` failure (invalid project, missing file, …), `2` usage error.
`python -m aether3d` works the same as `aether3d`.

## Installation

### Arch Linux

```bash
sudo pacman -S --needed git python
git clone https://github.com/Murisixz/Aether3D.git
cd Aether3D
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"     # editable install + pytest and ruff
pytest                      # run the test suite
```

Arch marks the system Python as externally managed (PEP 668), so install
Aether3D into a virtual environment as shown rather than with a system-wide
`pip install`. See [docs/INSTALL.md](docs/INSTALL.md) for details and other
platforms.

### Other platforms

Aether3D is pure Python (3.10+) with NumPy as its only runtime dependency, and
is tested on Linux, macOS and Windows:

```bash
pip install -e ".[dev]"
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/API.md](docs/API.md) | Python API reference and conventions |
| [docs/FILE_FORMAT.md](docs/FILE_FORMAT.md) | The versioned `.aether` project format |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design principles, project structure, extension points |
| [docs/INSTALL.md](docs/INSTALL.md) | Installation details |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [AGENTS.md](AGENTS.md) | Orientation for AI coding agents working on this repository |

## Conventions at a glance

- Right-handed coordinates, **+Z up**, 1 unit = 1 meter.
- Rotations are unit quaternions `(x, y, z, w)`; `rotation_euler` offers XYZ Euler angles in degrees.
- Transforms compose as `T @ R @ S`; `world = parent.world @ local`.
- Object ids are UUID strings that never change, including across save/load.
- Object names are unique within a scene.

## License

Aether3D is licensed under the [Apache License 2.0](LICENSE).
