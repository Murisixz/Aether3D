"""Cameras, lights, appearance and render settings in .aether projects (format version 3)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from aether3d import FORMAT_VERSION, ProjectFormatError, Scene
from aether3d.serialization import loads, migrate, scene_from_dict, scene_to_dict

DATA = Path(__file__).parent / "data"
V010 = DATA / "v0.1.0" / "first_scene.aether"
V040 = DATA / "v0.4.0" / "castle.aether"  # written by Aether3D 0.4.0 (format version 2)

try:
    import manifold3d  # noqa: F401

    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


def build_scene() -> Scene:
    scene = Scene("Studio")
    crate = scene.add_cube("Crate", size=2, position=(0, 0, 1))
    crate.appearance.configure(color="#c0392b", specular=0.4, shininess=64)
    scene.add_sphere("Ghost", radius=0.5, position=(3, 0, 0.5)).visible = False
    rig = scene.add_object("Rig", position=(1, -1, 0))
    rig.rotation_euler = (0, 0, 20)
    scene.add_camera(
        "Wide",
        parent=rig,
        position=(6, -8, 5),
        look_at=(0, 0, 1),
        fov=55,
        aspect_ratio=2.0,
        far=200,
    )
    scene.add_camera(
        "Plan",
        projection="orthographic",
        ortho_scale=12,
        near=0.0,
        far=50,
        fov_axis="horizontal",
        position=(0, 0, 20),
        active=True,
    )
    scene.get_by_name("Plan").look_along((0, 0, -1))
    scene.add_directional_light("Sun", direction=(-0.3, 0.6, -1), color="#fff2d6", intensity=2.5)
    scene.add_point_light("Bulb", position=(2, -2, 3), intensity=7, range=12, falloff="none")
    scene.world.configure(background="#8ec5f2", ambient_color="#d5e6ff", ambient_intensity=0.3)
    scene.render_settings.configure(
        width=320, height=180, samples=3, exposure=0.5, cull="back", default_lights=False
    )
    return scene


def test_round_trip_and_rerender(tmp_path):
    scene = build_scene()
    path = tmp_path / "studio.aether"
    scene.save(path)
    loaded = Scene.load(path)
    assert scene_to_dict(loaded) == scene_to_dict(scene)
    assert loaded.active_camera.name == "Plan"
    assert loaded.get_by_name("Crate").appearance == scene.get_by_name("Crate").appearance
    assert loaded.get_by_name("Wide").camera == scene.get_by_name("Wide").camera
    assert loaded.get_by_name("Bulb").light == scene.get_by_name("Bulb").light
    assert not loaded.get_by_name("Ghost").visible
    assert loaded.world == scene.world and loaded.render_settings == scene.render_settings
    for camera in ("Plan", "Wide"):
        a = scene.render(camera=camera)
        b = loaded.render(camera=camera)
        assert np.array_equal(a.image, b.image) and (a.width, a.height) == (320, 180)


def test_document_layout():
    doc = scene_to_dict(build_scene())
    assert doc["format_version"] == FORMAT_VERSION
    s = doc["scene"]
    assert s["world"] == {
        "background": [0x8E / 255, 0xC5 / 255, 0xF2 / 255],
        "ambient_color": [0xD5 / 255, 0xE6 / 255, 1.0],
        "ambient_intensity": 0.3,
    }
    assert s["render"] == {
        "width": 320,
        "height": 180,
        "samples": 3,
        "exposure": 0.5,
        "cull": "back",
        "default_lights": False,
    }
    objects = {o["name"]: o for o in s["objects"]}
    assert s["active_camera"] == objects["Plan"]["id"]
    assert objects["Wide"]["camera"]["aspect_ratio"] == 2.0 and objects["Wide"]["mesh"] is None
    assert objects["Bulb"]["light"] == {
        "type": "point",
        "color": [1.0, 1.0, 1.0],
        "intensity": 7.0,
        "range": 12.0,
        "falloff": "none",
    }
    assert objects["Crate"]["appearance"]["specular"] == 0.4
    assert objects["Rig"]["appearance"] is None  # defaults are not written
    assert objects["Ghost"]["visible"] is False and objects["Crate"]["visible"] is True
    for o in s["objects"]:
        assert list(o)[-6:-2] == ["camera", "light", "appearance", "visible"]


def test_version_1_and_2_projects_load_unchanged(tmp_path):
    old = Scene.load(V010)
    assert [o.name for o in old] == ["Cube", "Sphere", "Ground", "Pivot", "Moon"]
    assert old.active_camera is None and old.cameras == () and old.lights == ()
    assert all(o.visible and o.appearance.is_default for o in old)
    r = old.render(width=64, height=36, frame=True)  # needs a camera: a temporary one
    assert r.camera["temporary"] and r.stats["pixels_covered"] > 0
    doc = scene_to_dict(old)
    assert doc["format_version"] == FORMAT_VERSION and doc["scene"]["active_camera"] is None
    v2 = loads(V040.read_text(encoding="utf-8"))
    assert v2["format_version"] == 2
    migrated = migrate(copy.deepcopy(v2))
    assert migrated["format_version"] == FORMAT_VERSION
    assert all(o["camera"] is None and o["visible"] is True for o in migrated["scene"]["objects"])
    assert migrated["scene"]["render"]["width"] == 1280


@pytest.mark.skipif(not HAS_MANIFOLD, reason="the castle uses booleans (manifold3d)")
def test_procedural_castle_stays_editable_after_reload_and_render(tmp_path):
    castle = Scene.load(V040)
    geometry_before = {g.name: [m.name for m in g.modifiers] for g in castle.procedural.geometries}
    camera = castle.add_camera("View", fov=30)
    camera.frame(direction=(-0.55, 1.0, -0.6), aspect=16 / 9)
    first = castle.render(width=160, height=90)
    path = tmp_path / "castle.aether"
    castle.save(path)
    reloaded = Scene.load(path)
    again = reloaded.render(width=160, height=90)
    assert np.array_equal(first.image, again.image)
    # Rendering baked nothing: every object is still procedural and every stack intact.
    assert all(o.is_procedural for o in reloaded if o.kind != "camera")
    assert {g.name: [m.name for m in g.modifiers] for g in reloaded.procedural.geometries} == (
        geometry_before
    )
    reloaded.procedural.set_parameter("tower_count", 10)  # still parametric
    changed = reloaded.render(width=160, height=90)
    assert not np.array_equal(changed.image, again.image)
    towers = next(o for o in changed.objects if o["name"] == "Towers")
    assert (
        towers["triangles"] > next(o for o in first.objects if o["name"] == "Towers")["triangles"]
    )


def corrupt(mutate):
    doc = scene_to_dict(build_scene())
    mutate(doc)
    return doc


def obj(doc, name):
    return next(o for o in doc["scene"]["objects"] if o["name"] == name)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda d: obj(d, "Wide")["camera"].update(fov=0), "fov"),
        (lambda d: obj(d, "Wide")["camera"].update(projection="fisheye"), "projection"),
        (lambda d: obj(d, "Wide")["camera"].pop("near"), "missing near"),
        (lambda d: obj(d, "Wide")["camera"].update(zoom=2), "unknown zoom"),
        (lambda d: obj(d, "Bulb")["light"].update(type="spot"), "type"),
        (lambda d: obj(d, "Bulb")["light"].update(intensity=-1), "intensity"),
        (lambda d: obj(d, "Bulb")["light"].update(color="red"), "hex color"),
        (lambda d: obj(d, "Crate")["appearance"].update(specular=2), "specular"),
        (lambda d: obj(d, "Crate").update(visible="yes"), "true or false"),
        (lambda d: obj(d, "Crate").update(camera=obj(d, "Wide")["camera"]), "only one of"),
        (lambda d: obj(d, "Bulb").update(camera=obj(d, "Wide")["camera"]), "only one of"),
        (lambda d: d["scene"].update(active_camera=obj(d, "Crate")["id"]), "camera object"),
        (lambda d: d["scene"].update(active_camera="missing"), "camera object"),
        (lambda d: d["scene"]["world"].update(background=[2, 0, 0]), "background"),
        (lambda d: d["scene"]["render"].update(samples=9), "samples"),
        (lambda d: d["scene"]["render"].update(width=0), "width"),
        (lambda d: d["scene"]["render"].update(cull="maybe"), "cull"),
        (lambda d: d["scene"].pop("world"), "world"),
    ],
)
def test_invalid_rendering_data_is_rejected(mutate, fragment):
    with pytest.raises(ProjectFormatError) as info:
        scene_from_dict(corrupt(mutate))
    assert fragment.lower() in str(info.value).lower()


def test_optional_object_keys_default_on_read():
    doc = scene_to_dict(build_scene())
    for o in doc["scene"]["objects"]:
        if o["camera"] is None and o["light"] is None and o["appearance"] is None:
            for key in ("camera", "light", "appearance", "visible"):
                del o[key]
    scene = scene_from_dict(json.loads(json.dumps(doc)))
    assert scene.get_by_name("Rig").visible and scene.get_by_name("Rig").appearance.is_default
