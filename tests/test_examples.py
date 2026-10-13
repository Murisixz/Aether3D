import subprocess
import sys
from pathlib import Path

from aether3d import Scene

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_first_scene_example_runs(tmp_path):
    output = tmp_path / "first_scene.aether"
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "first_scene.py"), str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Saved 5 objects" in result.stdout
    scene = Scene.load(output)
    moon = scene.get_by_name("Moon")
    # Pivot at (3, 0, 1) rotated 90 degrees about Z puts the moon's (2, 0, 0) offset at +Y.
    x, y, z = moon.world_position
    assert abs(x - 3) < 1e-9 and abs(y - 2) < 1e-9 and abs(z - 1) < 1e-9


def test_committed_example_project_is_valid():
    scene = Scene.load(EXAMPLES / "first_scene.aether")
    assert [o.name for o in scene] == ["Cube", "Sphere", "Ground", "Pivot", "Moon"]
