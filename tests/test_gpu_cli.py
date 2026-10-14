"""The GPU from the command line: render options, `aether3d gpu info`, errors."""

import json
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene, load_scene
from aether3d.cli import main

PIL = pytest.importorskip("PIL.Image")


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("GPU CLI")
    scene.add_plane("Ground", size=20).color = "#6a9a4a"
    scene.add_cube("Crate", size=2, position=(0, 0, 1)).color = "#c0392b"
    scene.add_directional_light("Sun", direction=(0.4, 0.7, -1), intensity=1.5)
    scene.add_directional_light("Fill", direction=(-0.6, 0.3, -0.5), intensity=0.3)
    scene.add_camera("Main", position=(6, -8, 5), look_at=(0, 0, 1), fov=45)
    path = tmp_path / "scene.aether"
    scene.save(path)
    return path


def test_cli_gpu_render_reproduces_the_python_api_exactly(gpu, project, tmp_path, capsys):
    out = tmp_path / "gpu.png"
    argv = ["render", str(project), "-o", str(out), "--width", "160", "--height", "90"]
    argv += ["--backend", "gpu", "--samples", "2", "--shadows", "--shadow-lights", "Sun"]
    if gpu.adapter:
        argv += ["--adapter", gpu.adapter]
    code, data = run_json(capsys, *argv)
    assert code == 0 and data["ok"] and data["command"] == "render"
    backend = data["backend"]
    assert backend["name"] == "gpu" and backend["adapter"]["name"] == gpu.name
    assert backend["antialiasing"] == {"msaa": 4, "supersample": 1, "samples_per_pixel": 4}
    assert [m["light"] for m in backend["shadows"]["maps"]] == ["Sun"]
    expected = load_scene(project).render(
        width=160,
        height=90,
        samples=2,
        backend="gpu",
        adapter=gpu.adapter,
        shadows={"lights": ["Sun"]},
    )
    with PIL.open(out) as png:
        assert np.array_equal(np.asarray(png), expected.image)
    assert data["output"] == str(out.resolve())


def test_cli_shadow_and_antialiasing_options(gpu, project, tmp_path, capsys):
    out = tmp_path / "x.png"
    argv = ["render", str(project), "-o", str(out), "--width", "64", "--height", "36", "--force"]
    argv += ["--backend", "gpu", "--msaa", "1", "--supersample", "2", "--shadows"]
    argv += ["--shadow-map-size", "1024", "--shadow-bias", "1", "--shadow-pcf-radius", "2"]
    argv += ["--shadow-normal-offset", "0"]
    if gpu.adapter:
        argv += ["--adapter", gpu.adapter]
    code, data = run_json(capsys, *argv)
    assert code == 0
    shadows = data["backend"]["shadows"]
    assert shadows["map_size"] == 1024 and shadows["pcf_radius"] == 2
    assert shadows["bias"] == 1 and shadows["normal_offset"] == 0
    assert [m["light"] for m in shadows["maps"]] == ["Sun", "Fill"]
    assert data["backend"]["antialiasing"]["supersample"] == 2
    code = main(argv)  # text output
    text = capsys.readouterr().out
    assert code == 0 and "GPU:" in text and "shadows from Sun, Fill" in text


def test_cli_auto_backend_uses_the_gpu_when_it_works(gpu, project, tmp_path, capsys):
    argv = [
        "render",
        str(project),
        "-o",
        str(tmp_path / "a.png"),
        "--width",
        "32",
        "--height",
        "18",
    ]
    argv += ["--backend", "auto"] + (["--adapter", gpu.adapter] if gpu.adapter else [])
    code, data = run_json(capsys, *argv)
    assert code == 0 and data["backend"]["name"] == "gpu" and data["backend"]["requested"] == "auto"
    assert "BACKEND_SELECTED" in [d["code"] for d in data["diagnostics"]]


def test_gpu_info_lists_adapters_and_checks_rendering(gpu, capsys):
    argv = ["gpu", "info", "--check"] + (["--adapter", gpu.adapter] if gpu.adapter else [])
    code, data = run_json(capsys, *argv)
    assert code == 0 and data["ok"] and data["command"] == "gpu info"
    assert data["available"] is True and data["check"]["ok"] is True
    assert all(c["ok"] for c in data["check"]["checks"].values())
    assert data["selected"]["adapter"]["name"] == gpu.name
    assert data["versions"]["wgpu"].startswith("0.")
    assert any(a["recommended"] for a in data["adapters"])
    for adapter in data["adapters"]:
        assert {"index", "name", "type", "backend", "driver", "software", "features"} <= set(
            adapter
        )
    code = main(argv)
    text = capsys.readouterr().out
    assert code == 0 and "Adapters" in text and "Check passed" in text


def test_gpu_info_reports_unavailability_without_failing(capsys, monkeypatch):
    import sys as _sys

    from aether3d.rendering.gpu import device

    monkeypatch.setattr(device, "_wgpu", None)
    monkeypatch.setitem(_sys.modules, "wgpu", None)
    code, data = run_json(capsys, "gpu", "info")
    assert code == 0 and data["available"] is False and "wgpu" in data["reason"]
    code, data = run_json(capsys, "gpu", "info", "--check")
    assert code == 3 and data["available"] is False  # the check cannot pass
    code = main(["gpu", "info"])
    assert code == 0 and "not available" in capsys.readouterr().out


def test_cli_usage_errors(project, tmp_path, capsys):
    out = tmp_path / "bad.png"
    code, data = run_json(capsys, "render", str(project), "-o", str(out), "--shadow-bias", "2")
    assert (
        code == 1
        and data["error"]["type"] == "UsageError"
        and "--shadows" in data["error"]["message"]
    )
    code, data = run_json(capsys, "render", str(project), "-o", str(out), "--msaa", "4")
    assert code == 1 and data["error"]["code"] == "INVALID_SETTINGS"
    code, data = run_json(capsys, "render", str(project), "-o", str(out), "--shadows")
    assert code == 1 and data["error"]["code"] == "BACKEND_UNSUPPORTED"
    assert not out.exists()
    with pytest.raises(SystemExit):
        main(["render", str(project), "--backend", "vulkan"])
    capsys.readouterr()


def test_cpu_renders_from_the_cli_are_unchanged(project, tmp_path, capsys):
    """The default backend stays the CPU renderer, with the same text and JSON as before."""
    out = tmp_path / "cpu.png"
    code = main(["render", str(project), "-o", str(out), "--width", "64", "--height", "36"])
    text = capsys.readouterr().out
    assert code == 0 and "2x2 samples per pixel" in text and "GPU" not in text
    expected = load_scene(project).render(width=64, height=36)
    with PIL.open(out) as png:
        assert np.array_equal(np.asarray(png), expected.image)
    code, data = run_json(
        capsys, "render", str(project), "-o", str(out), "--width", "64", "--height", "36", "--force"
    )
    assert data["backend"] == {"name": "cpu", "requested": "cpu"}
    assert Path(data["output"]).exists()


def test_failed_gpu_renders_write_no_image_and_change_nothing(gpu, project, tmp_path):
    path = project
    document = path.read_text()
    out = tmp_path / "out.png"
    code = main(["render", str(path), "-o", str(out), "--backend", "gpu", "--adapter", "3dfx"])
    assert code == 1 and not out.exists()
    code = main(["render", str(path), "-o", str(out), "--backend", "gpu", "--msaa", "3"])
    assert code == 1 and not out.exists()
    assert path.read_text() == document and sorted(p.name for p in tmp_path.iterdir()) == [
        "scene.aether"
    ]
