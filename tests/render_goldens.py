"""Small reference renders used to detect rendering regressions.

The PNGs in ``tests/data/renders`` were produced by Aether3D itself. They are
regression references (has anything changed?), not correctness references --
correctness is tested analytically and against the independent ray caster in
``render_reference.py``. Regenerate them deliberately after an intended
change with::

    python tests/render_goldens.py --update
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "renders"


def _example(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "examples"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(ROOT / "examples"))
    return module


def _hostile(projection: str):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from test_render_scene import hostile_scene
    finally:
        sys.path.pop(0)
    return hostile_scene(projection)


GOLDENS = {
    # name: (scene builder, render options, needs manifold3d)
    "primitives": (lambda: _example("first_render").build_scene(160, 90, 2), {}, False),
    "hero": (lambda: _example("hero_render").build_scene(128, 72, 2), {}, False),
    "hostile_perspective": (lambda: _hostile("perspective"), {"width": 128, "height": 80}, False),
    "hostile_orthographic": (lambda: _hostile("orthographic"), {"width": 128, "height": 80}, False),
    "castle": (lambda: _example("minecraft_castle_render").build_scene(192, 108, 2), {}, True),
}


def render_golden(name: str):
    build, options, _ = GOLDENS[name]
    return build().render(**options)


def main(argv: list[str]) -> int:
    if "--update" not in argv:
        print(__doc__)
        return 2
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in GOLDENS:
        result = render_golden(name)
        path = result.save(GOLDEN_DIR / f"{name}.png")
        print(f"{path} {result.width}x{result.height}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
