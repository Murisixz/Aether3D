"""The code examples in the documentation actually run and give the stated results."""

import json
import re
from pathlib import Path

import pytest

from aether3d import Scene, inspect_mesh, run_edit_script

ROOT = Path(__file__).resolve().parent.parent
# Inset 0.2 then sink 0.1 removes a 1.6 x 1.6 x 0.1 recess; the 0.05 chamfer of one
# 2-unit edge removes 0.05^2 / 2 * 2.
CRATE_VOLUME = 8 - 1.6 * 1.6 * 0.1 - 0.05**2 / 2 * 2


def python_block(path: Path, starts_with: str) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```python\n(" + re.escape(starts_with) + r".*?)```", text, re.S)
    assert match, f"no example starting with {starts_with!r} in {path.name}"
    return match.group(1)


@pytest.mark.parametrize("doc", ["docs/EDITING.md", "README.md"])
def test_crate_example(doc):
    source = python_block(ROOT / doc, "from aether3d import Scene\n\nscene = Scene()\ncrate")
    namespace: dict = {}
    exec(compile(source, doc, "exec"), namespace)
    volume = inspect_mesh(namespace["crate"].mesh).topology["volume"]
    assert volume == pytest.approx(CRATE_VOLUME, rel=1e-6)


def test_editing_json_script_example():
    text = (ROOT / "docs/EDITING.md").read_text(encoding="utf-8")
    script = json.loads(re.search(r"```json\n(\{.*?\})\n```", text, re.S).group(1))
    scene = Scene()
    scene.add_cube("Crate", size=2)
    result = run_edit_script(scene, script)
    assert result.edits[0]["commit"]["validation"]["status"] == "ok"
    volume = inspect_mesh(scene.get_by_name("Crate").mesh).topology["volume"]
    assert volume == pytest.approx(CRATE_VOLUME, rel=1e-6)
