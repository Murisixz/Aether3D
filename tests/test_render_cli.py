"""The rendering CLI: render, camera, light, object and scene commands."""

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aether3d import Scene
from aether3d.cli import main

DATA = Path(__file__).parent / "data"
PIL = pytest.importorskip("PIL.Image")


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("CLI Render")
    scene.add_cube("Crate", size=2, position=(0, 0, 1))
    scene.add_plane("Ground", size=20)
    path = tmp_path / "scene.aether"
    scene.save(path)
    return path


def setup_project(project, capsys):
    assert (
        main(
            [
                "camera",
                "add",
                str(project),
                "--name",
                "Main",
                "--position",
                "6",
                "-8",
                "5",
                "--look-at",
                "0",
                "0",
                "1",
                "--fov",
                "45",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "light",
                "add",
                str(project),
                "--name",
                "Sun",
                "--direction",
                "-0.3",
                "0.6",
                "-1",
                "--intensity",
                "2",
                "--color",
                "#fff2d6",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "light",
                "add",
                str(project),
                "--type",
                "point",
                "--name",
                "Bulb",
                "--position",
                "2",
                "-2",
                "3",
                "--intensity",
                "6",
                "--range",
                "12",
            ]
        )
        == 0
    )
    assert (
        main(["object", "set", str(project), "Crate", "--color", "#c0392b", "--specular", "0.3"])
        == 0
    )
    assert (
        main(
            [
                "scene",
                "set",
                str(project),
                "--background",
                "#8ec5f2",
                "--ambient-intensity",
                "0.25",
                "--width",
                "160",
                "--height",
                "90",
                "--samples",
                "2",
            ]
        )
        == 0
    )
    capsys.readouterr()


def test_full_terminal_workflow(project, capsys, tmp_path):
    setup_project(project, capsys)
    code, data = run_json(capsys, "camera", "list", str(project))
    assert (
        code == 0
        and [c["name"] for c in data["cameras"]] == ["Main"]
        and data["cameras"][0]["active"]
    )
    code, data = run_json(capsys, "light", "list", str(project))
    assert code == 0 and [light["name"] for light in data["lights"]] == ["Sun", "Bulb"]
    out = tmp_path / "render.png"
    code, data = run_json(capsys, "render", str(project), "-o", str(out))
    assert code == 0 and data["ok"] and data["command"] == "render"
    assert (data["width"], data["height"]) == (160, 90) and data["output"] == str(out.resolve())
    assert data["camera"]["name"] == "Main" and data["stats"]["triangles"] == 14
    assert set(data["stats"]["timings"]) >= {"total", "raster", "shade", "encode", "write"}
    image = np.asarray(PIL.open(out))
    assert image.shape == (90, 160, 3)
    # The same project rendered through the API gives the same pixels.
    api = Scene.load(project).render()
    assert np.array_equal(api.image, image)
    code, text = (
        main(["render", str(project), "-o", str(tmp_path / "text.png")]),
        capsys.readouterr(),
    )
    assert code == 0 and "Rendered" in text.out and "160x90" in text.out


def test_render_options_and_overwrite(project, capsys, tmp_path):
    setup_project(project, capsys)
    out = tmp_path / "small.png"
    code, data = run_json(
        capsys,
        "render",
        str(project),
        "-o",
        str(out),
        "--width",
        "64",
        "--height",
        "48",
        "--samples",
        "1",
        "--exposure",
        "1",
        "--cull",
        "back",
        "--background",
        "#000000",
        "--no-default-lights",
    )
    assert code == 0 and data["settings"]["exposure"] == 1.0 and data["settings"]["cull"] == "back"
    assert (data["width"], data["height"]) == (64, 48)
    code, data = run_json(capsys, "render", str(project), "-o", str(out))
    assert code == 1 and data["error"]["type"] == "FileExistsError"
    code, data = run_json(capsys, "render", str(project), "-o", str(out), "--force")
    assert code == 0


@pytest.mark.parametrize(
    ("argv", "error_type", "code"),
    [
        (["--camera", "Crate"], "RenderError", "NOT_A_CAMERA"),
        (["--camera", "Nope"], "RenderError", "OBJECT_NOT_FOUND"),
        (["--width", "0"], "RenderError", "INVALID_SETTINGS"),
        (["--samples", "9"], "RenderError", "INVALID_SETTINGS"),
        (
            ["--width", "8000", "--height", "8000", "--samples", "4"],
            "RenderError",
            "RESOURCE_LIMIT",
        ),
        (["--background", "blue"], "RenderError", "INVALID_WORLD"),
    ],
)
def test_render_errors_write_nothing(project, capsys, tmp_path, argv, error_type, code):
    setup_project(project, capsys)
    out = tmp_path / "never.png"
    status, data = run_json(capsys, "render", str(project), "-o", str(out), *argv)
    assert status == 1 and data["ok"] is False
    assert data["error"]["type"] == error_type and data["error"]["code"] == code
    assert not out.exists()


def test_projects_without_cameras_need_frame(project, capsys, tmp_path):
    code, data = run_json(capsys, "render", str(project), "-o", str(tmp_path / "a.png"))
    assert code == 1 and data["error"]["code"] == "NO_CAMERA" and "frame" in data["error"]["hint"]
    code, data = run_json(
        capsys,
        "render",
        str(project),
        "-o",
        str(tmp_path / "a.png"),
        "--frame",
        "--width",
        "80",
        "--height",
        "45",
    )
    assert (
        code == 0 and data["camera"]["temporary"] and data["camera"]["name"] == "temporary camera"
    )
    codes = [d["code"] for d in data["diagnostics"]]
    assert "CAMERA_FRAMED" in codes and "DEFAULT_LIGHTING" in codes
    assert Scene.load(project).cameras == ()  # the project was not modified


def test_render_gltf_files_directly(capsys, tmp_path):
    glb = DATA / "khronos" / "Box" / "glTF-Binary" / "Box.glb"
    out = tmp_path / "box.png"
    code, data = run_json(
        capsys, "render", str(glb), "-o", str(out), "--frame", "--width", "64", "--height", "64"
    )
    assert code == 0 and data["source_format"] == "glb" and data["stats"]["pixels_covered"] > 300
    assert np.asarray(PIL.open(out)).shape == (64, 64, 3)
    code, data = run_json(capsys, "render", str(glb), "-o", str(tmp_path / "x.jpg"))
    assert code == 1 and "png" in data["error"]["message"]


def test_camera_commands(project, capsys, tmp_path):
    code, data = run_json(
        capsys, "camera", "add", str(project), "--frame", "--frame-direction", "-1", "1", "-0.5"
    )
    assert code == 0 and data["operations"][0]["active"] and "framing" in data["operations"][0]
    code, data = run_json(
        capsys,
        "camera",
        "add",
        str(project),
        "--name",
        "Plan",
        "--projection",
        "orthographic",
        "--ortho-scale",
        "12",
        "--position",
        "0",
        "0",
        "20",
        "--look-at",
        "0",
        "0",
        "0",
        "--near",
        "0",
        "--far",
        "50",
        "--inactive",
    )
    assert code == 0 and not data["operations"][0]["active"]
    code, data = run_json(
        capsys,
        "camera",
        "set",
        str(project),
        "Plan",
        "--fov-axis",
        "horizontal",
        "--aspect-ratio",
        "1.5",
        "--activate",
    )
    assert code == 0
    scene = Scene.load(project)
    assert scene.active_camera.name == "Plan"
    assert scene.get_by_name("Plan").camera.aspect_ratio == 1.5
    code, data = run_json(
        capsys, "camera", "frame", str(project), "Plan", "--object", "Crate", "--margin", "0.2"
    )
    assert code == 0 and data["operations"][0]["object"]["name"] == "Plan"
    code, data = run_json(capsys, "camera", "activate", str(project), "Camera")
    assert code == 0 and data["operations"][0]["active_camera"]["name"] == "Camera"
    before = project.read_bytes()
    code, data = run_json(capsys, "camera", "set", str(project), "Plan", "--fov", "500")
    assert code == 1 and data["error"]["location"] == "operations[0]: objects[Plan].camera.fov"
    assert project.read_bytes() == before
    code, data = run_json(capsys, "camera", "set", str(project), "Plan", "--near", "1", "--dry-run")
    assert code == 0 and data["dry_run"] and project.read_bytes() == before
    out = tmp_path / "copy.aether"
    code, data = run_json(
        capsys, "camera", "set", str(project), "Plan", "--near", "1", "--output", str(out)
    )
    assert code == 0 and Scene.load(out).get_by_name("Plan").camera.near == 1.0
    assert project.read_bytes() == before
    code, text = main(["camera", "list", str(project)]), capsys.readouterr().out
    assert code == 0 and "Camera [active]" in text and "orthographic" in text


def test_light_object_and_scene_commands(project, capsys, tmp_path):
    setup_project(project, capsys)
    code, data = run_json(
        capsys,
        "light",
        "set",
        str(project),
        "Bulb",
        "--intensity",
        "9",
        "--range",
        "none",
        "--falloff",
        "none",
        "--position",
        "1",
        "1",
        "4",
    )
    assert code == 0
    code, data = run_json(
        capsys, "light", "set", str(project), "Sun", "--direction", "0", "0", "-1"
    )
    assert code == 0
    scene = Scene.load(project)
    bulb, sun = scene.get_by_name("Bulb"), scene.get_by_name("Sun")
    assert bulb.light.range is None and bulb.light.falloff == "none" and bulb.position == (1, 1, 4)
    assert np.allclose(sun.world_forward, (0, 0, -1))
    code, data = run_json(
        capsys,
        "object",
        "set",
        str(project),
        "Ground",
        "--visible",
        "false",
        "--scale",
        "2",
        "2",
        "1",
    )
    assert code == 0 and not Scene.load(project).get_by_name("Ground").visible
    code, data = run_json(
        capsys,
        "scene",
        "set",
        str(project),
        "--exposure",
        "0.5",
        "--cull",
        "back",
        "--default-lights",
        "off",
        "--active-camera",
        "none",
    )
    assert code == 0
    scene = Scene.load(project)
    assert scene.render_settings.exposure == 0.5 and scene.active_camera is None
    assert not scene.render_settings.default_lights
    code, data = run_json(capsys, "object", "remove", str(project), "Bulb")
    assert code == 0 and "Bulb" not in [o.name for o in Scene.load(project)]
    code, data = run_json(capsys, "scene", "set", str(project))
    assert code == 1 and data["error"]["type"] == "UsageError"


def test_scene_run_script_and_stdin(project, capsys, tmp_path, monkeypatch):
    script = {
        "format": "aether3d.scene-script",
        "version": 1,
        "operations": [
            {"op": "add_camera", "name": "Main", "frame": True, "frame_direction": [-1, 1, -0.6]},
            {"op": "add_light", "type": "directional", "direction": [0, 0.5, -1]},
            {"op": "set_object", "object": "Crate", "color": "#2e86c1"},
        ],
    }
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(script))
    code, data = run_json(capsys, "scene", "run", str(project), str(path))
    assert code == 0 and len(data["operations"]) == 3
    bad = dict(script, operations=[{"op": "set_camera", "camera": "Main", "fov": -1}])
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(json.dumps(bad).encode())))
    before = project.read_bytes()
    code, data = run_json(capsys, "scene", "run", str(project), "-")
    assert (
        code == 1 and data["error"]["code"] == "INVALID_CAMERA" and project.read_bytes() == before
    )


def test_inspect_reports_rendering_data(project, capsys):
    setup_project(project, capsys)
    code, data = run_json(capsys, "inspect", str(project))
    objects = {o["name"]: o for o in data["objects"]}
    assert objects["Main"]["kind"] == "camera" and objects["Main"]["camera"]["fov"] == 45
    assert objects["Sun"]["kind"] == "light" and objects["Sun"]["light"]["type"] == "directional"
    assert objects["Crate"]["appearance"]["specular"] == 0.3 and objects["Crate"]["visible"]
    main(["inspect", str(project)])
    text = capsys.readouterr().out
    assert "camera=perspective active" in text and "light=point" in text


def test_module_entry_point(project, tmp_path):
    out = tmp_path / "m.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aether3d",
            "render",
            str(project),
            "-o",
            str(out),
            "--frame",
            "--width",
            "40",
            "--height",
            "30",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["width"] == 40 and out.exists()
    usage = subprocess.run(
        [sys.executable, "-m", "aether3d", "render"], capture_output=True, text=True, check=False
    )
    assert usage.returncode == 2
