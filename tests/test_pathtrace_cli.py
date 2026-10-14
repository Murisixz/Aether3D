"""The path tracer from the command line: render options, passes, light sizes, environments."""

from __future__ import annotations

import json

import numpy as np
import pytest

from aether3d import Scene
from aether3d.cli import main


def run_json(capsys, *argv):
    code = main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.fixture
def project(tmp_path):
    scene = Scene("Path CLI")
    scene.add_cube("Crate", size=1, position=(0, 0, 0.5))
    scene.add_plane("Ground", size=8)
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2.0)
    scene.add_camera("Cam", position=(3, -4, 2.5), look_at=(0, 0, 0.5))
    path = tmp_path / "scene.aether"
    scene.save(path)
    return path


def test_render_with_the_path_tracer_matches_the_python_api(project, tmp_path, capsys):
    out = tmp_path / "path.png"
    code, data = run_json(
        capsys,
        "render",
        str(project),
        "--backend",
        "path",
        "--width",
        "40",
        "--height",
        "24",
        "--spp",
        "4",
        "--bounces",
        "2",
        "--seed",
        "3",
        "-o",
        str(out),
    )
    assert code == 0
    assert data["backend"]["name"] == "path" and data["backend"]["device"] == "cpu"
    assert data["backend"]["samples"] == 4 and data["backend"]["seed"] == 3
    assert data["stats"]["rays"] > 0 and data["output"] == str(out)
    from PIL import Image

    image = np.asarray(Image.open(out).convert("RGB"))
    direct = Scene.load(project).render(
        width=40, height=24, backend="path", path={"samples": 4, "bounces": 2, "seed": 3}
    )
    assert np.array_equal(image, direct.image)


def test_passes_ao_preview_and_hdr_output(project, tmp_path, capsys):
    passes = tmp_path / "passes"
    code, data = run_json(
        capsys,
        "render",
        str(project),
        "--backend",
        "path",
        "--width",
        "16",
        "--height",
        "12",
        "--spp",
        "2",
        "--ao",
        "--ao-samples",
        "2",
        "--ao-distance",
        "0.5",
        "--passes",
        str(passes),
        "-o",
        str(tmp_path / "image.hdr"),
        "--tonemap",
        "neutral",
    )
    assert code == 0
    names = sorted(p.name for p in passes.iterdir())
    assert names == sorted(f"{n}.pfm" for n in data["passes"])
    assert "ao.pfm" in names and data["backend"]["ao"]["distance"] == 0.5
    depth = (passes / "depth.pfm").read_bytes()
    assert depth.startswith(b"Pf\n16 12\n")  # single-channel float image
    assert (tmp_path / "image.hdr").read_bytes().startswith(b"#?RADIANCE")
    assert data["color"]["tonemap"] == "neutral"
    code, data = run_json(
        capsys,
        "render",
        str(project),
        "--backend",
        "path",
        "--width",
        "16",
        "--height",
        "12",
        "--spp",
        "2",
        "--ao-only",
        "-o",
        str(tmp_path / "ao.png"),
    )
    assert code == 0 and data["backend"]["integrator"] == "ao"
    assert data["stats"]["bounce_rays"] == 0


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--spp", "4"], "--spp applies to the path tracer"),
        (["--ao", "--bounces", "2"], "apply to the path tracer"),
        (["--backend", "path", "--samples", "2"], "samples per pixel with --spp"),
        (["--backend", "path", "--spp", "0"], "samples"),
        (["--backend", "path", "-o", "x.exr"], ".hdr or .pfm"),
    ],
)
def test_render_rejects_misplaced_path_options(project, tmp_path, capsys, flags, message):
    code = main(["render", str(project), "--width", "8", "--height", "8", *flags])
    assert code != 0
    assert message in capsys.readouterr().err


def test_light_sizes_and_the_environment_from_the_command_line(project, capsys):
    assert main(["light", "set", str(project), "Sun", "--angle", "1.5"]) == 0
    assert (
        main(
            [
                "light",
                "add",
                str(project),
                "--type",
                "point",
                "--name",
                "Orb",
                "--position",
                "0",
                "0",
                "3",
                "--radius",
                "0.4",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "scene",
                "set",
                str(project),
                "--environment",
                "sky",
                "--environment-intensity",
                "0.7",
                "--sky-zenith",
                "#204080",
            ]
        )
        == 0
    )
    capsys.readouterr()
    scene = Scene.load(project)
    assert scene.get_by_name("Sun").light.angle == 1.5
    assert scene.get_by_name("Orb").light.radius == 0.4
    env = scene.world.environment
    assert (
        env.type == "sky"
        and env.intensity == 0.7
        and env.zenith_color[2] == pytest.approx(0x80 / 255)
    )
    code, data = run_json(capsys, "light", "list", str(project))
    assert code == 0 and data["world"]["environment"]["type"] == "sky"
    assert {light["name"]: light["radius"] for light in data["lights"]}["Orb"] == 0.4
