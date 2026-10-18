"""Worlds in renders and files: every backend draws them, frames do not depend on order,
renders change nothing, projects and histories keep worlds, old projects migrate."""

import json

import numpy as np
import pytest

from aether3d import FORMAT_VERSION, Scene
from aether3d.serialization import scene_from_dict, scene_to_dict

SMALL = {"hydrology": {"cell": 32.0}, "vegetation": {"distance": 600.0}}


def world_scene(preset="temperate", size=1024.0, **kw):
    scene = Scene("Render")
    w = scene.worlds.create("W", preset, seed=5, size=size, overrides=SMALL, **kw)
    g = w.ground(0.0, 0.0)
    scene.add_camera(
        "Cam", position=(-160.0, -220.0, g + 90.0), look_at=(0.0, 0.0, g), fov=55, far=6000
    )
    scene.add_directional_light("Sun", direction=(0.4, 0.5, -0.6), intensity=1.8)
    scene.world.atmosphere.configure(
        enabled=True,
        sun="Sun",
        fog_density=0.0004,
        fog_height=g,
        fog_falloff=0.002,
        ambient="hemisphere",
    )
    scene.world.environment.configure(type="sky")
    return scene, w


def test_cpu_renders_of_worlds_are_deterministic_and_change_nothing():
    scene, w = world_scene()
    revision, steps = scene.revision, len(scene.history.entries)
    a = scene.render(width=96, height=54)
    b = scene.render(width=96, height=54)
    assert np.array_equal(a.image, b.image)
    assert a.image.std() > 3  # a landscape, not a flat colour
    assert scene.revision == revision and len(scene.history.entries) == steps
    assert "NOTHING_TO_RENDER" not in [d.code for d in a.diagnostics]
    assert w.runtime.last_frame["parts"] > 0


def test_frames_do_not_depend_on_render_order():
    scene, w = world_scene()
    cam = scene.get_by_name("Cam")
    a = scene.animation
    a.set_timeline(fps=4, start=0.0, end=2.0)
    end = (200.0, -100.0, w.ground(0.0, 0.0) + 60.0)
    a.keyframe(cam, "transform.position", 0.0, tuple(cam.transform.position))
    a.keyframe(cam, "transform.position", 2.0, end)
    first = scene.render(width=64, height=36, time=1.5).image
    scene.render(width=64, height=36, time=0.0)
    scene.render(width=64, height=36, time=2.0)
    again = scene.render(width=64, height=36, time=1.5).image
    fresh, _ = world_scene()
    fresh.animation.set_timeline(fps=4, start=0.0, end=2.0)
    fresh_cam = fresh.get_by_name("Cam")
    fresh.animation.keyframe(
        fresh_cam, "transform.position", 0.0, tuple(fresh_cam.transform.position)
    )
    fresh.animation.keyframe(fresh_cam, "transform.position", 2.0, end)
    cold = fresh.render(width=64, height=36, time=1.5).image
    assert np.array_equal(first, again) and np.array_equal(first, cold)


def test_block_worlds_render_with_textures():
    scene = Scene("Blocks")
    w = scene.worlds.create("B", "blocky_adventure", seed=3, size=256)
    g = w.ground(0.0, 0.0)
    scene.add_camera(
        "Cam", position=(-30.0, -40.0, g + 25.0), look_at=(0.0, 0.0, g), fov=60, far=3000
    )
    scene.add_directional_light("Sun", direction=(0.4, 0.5, -0.6), intensity=1.8)
    r = scene.render(width=96, height=54)
    assert r.image.std() > 5
    assert w.runtime.last_frame["parts"] > 0


def test_the_gpu_draws_worlds_like_the_cpu(gpu):
    scene, w = world_scene()
    cpu = scene.render(width=160, height=90, samples=2)
    on_gpu = gpu.render(scene, width=160, height=90, samples=2, supersample=2, msaa=1)
    diff = np.abs(cpu.image.astype(int) - on_gpu.image.astype(int))
    # Rasterization rules differ at silhouettes (thin plants); shading agrees elsewhere.
    assert np.median(diff) <= 1
    assert np.mean(diff > 24) < 0.03


def test_the_path_tracer_draws_worlds():
    scene, w = world_scene(size=512.0)
    r = scene.render(backend="path", width=48, height=27, path={"samples": 2, "bounces": 1})
    assert r.image.std() > 2
    assert r.backend["device"] == "cpu"


def test_a_missing_world_material_draws_grey_and_says_so():
    scene, w = world_scene(size=512.0)
    ref = w.definition["materials"]["terrain"]
    scene.materials.remove_material(ref, force=True)
    r = scene.render(width=48, height=27)
    assert "WORLD_MATERIAL_MISSING" in [d.code for d in r.diagnostics]


def test_projects_keep_worlds_and_stay_small(tmp_path):
    scene, w = world_scene()
    w.add_feature(
        {"type": "lake", "id": "tarn", "shape": {"circle": [100, 100, 40]}, "level": 300.0}
    )
    w.place_structure({"id": "hut", "kind": "house", "at": [-50.0, 20.0]})
    path = tmp_path / "w.aether"
    scene.save(path, history=True)
    assert path.stat().st_size < 400_000  # generated data is never saved
    loaded = Scene.load(path, history=True)
    w2 = loaded.worlds.get("W")
    assert w2.definition == w.definition and w2.id == w.id
    assert w2.height(10.0, 20.0) == w.height(10.0, 20.0)
    loaded.history.undo()
    assert w2.structures() == []
    loaded.history.redo()
    assert w2.structures()[0]["id"] == "hut"
    doc = json.loads(path.read_text())
    assert doc["format_version"] == FORMAT_VERSION and len(doc["scene"]["worlds"]) == 1


def test_version_9_projects_migrate_to_10():
    scene = Scene("Old")
    scene.add_cube("Cube")
    doc = scene_to_dict(scene)
    doc["format_version"] = 9
    del doc["scene"]["worlds"]
    del doc["scene"]["world"]["atmosphere"]
    loaded = scene_from_dict(doc)
    assert len(loaded.worlds) == 0
    assert loaded.world.atmosphere.enabled is False
    assert scene_to_dict(loaded)["format_version"] == FORMAT_VERSION


def test_world_regions_export_to_gltf_with_vertex_colours(tmp_path):
    from aether3d import import_gltf

    scene, w = world_scene(size=512.0)
    w.place_structure({"id": "tower", "kind": "tower", "at": [10.0, 10.0]})
    out = w.export_region(tmp_path / "region.glb", [-40, -40, 40, 40], level=1, plant_lod=2)
    assert out["objects"] > 1 and out["triangles"] > 0
    back = import_gltf(tmp_path / "region.glb")
    assert not back.lossy
    colored = [o for o in back.scene if o.mesh is not None and o.mesh.colors is not None]
    assert colored  # the ground's colours survive as COLOR_0
    blocks = Scene("B").worlds.create("B", "blocky_adventure", seed=3, size=256)
    out = blocks.export_region(tmp_path / "blocks.glb", [0, 0, 32, 32])
    assert out["objects"] >= 4 and out["bounds"] == [0.0, 0.0, 32.0, 32.0]
    with pytest.raises(Exception) as info:
        w.export_region(tmp_path / "huge.glb", [0, 0, 1e6, 1e6])
    assert getattr(info.value, "code", "") == "LIMIT_EXCEEDED"
