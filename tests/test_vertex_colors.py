"""Vertex colours (``Mesh.colors``, glTF ``COLOR_0``): validation, operations that keep or
refuse them, files, and every renderer multiplying the material's base colour by them."""

import numpy as np
import pytest

from aether3d import Mesh, Scene, create_cube, create_plane, create_sphere, import_gltf
from aether3d.errors import AetherError


def colored_plane(color=(1.0, 0.0, 0.0, 1.0)) -> Mesh:
    plane = create_plane(4.0)
    colors = np.tile(np.asarray(color, dtype=np.float32), (plane.vertex_count, 1))
    return plane.with_colors(colors)


def test_colors_are_validated_and_compared():
    cube = create_cube(1.0)
    rgb = np.full((cube.vertex_count, 3), 0.5, dtype=np.float32)
    m = cube.with_colors(rgb)
    assert m.colors.shape == (cube.vertex_count, 4) and np.all(m.colors[:, 3] == 1.0)
    assert m != cube and m == cube.with_colors(rgb)
    with pytest.raises(AetherError):
        cube.with_colors(np.zeros((cube.vertex_count + 1, 4)))
    with pytest.raises(AetherError):
        cube.with_colors(np.full((cube.vertex_count, 4), 1.5))
    with pytest.raises(AetherError):
        cube.with_colors(np.full((cube.vertex_count, 4), np.nan))
    assert not m.colors.flags.writeable  # immutable like every mesh array


def test_files_and_gltf_keep_vertex_colors(tmp_path):
    scene = Scene("Colors")
    rng = np.random.default_rng(2)
    cube = create_cube(1.0)
    colors = rng.uniform(0, 1, (cube.vertex_count, 4)).astype(np.float32)
    scene.add_object("Cube", mesh=cube.with_colors(colors))
    path = tmp_path / "c.aether"
    scene.save(path)
    np.testing.assert_array_equal(Scene.load(path).get_by_name("Cube").mesh.colors, colors)
    from aether3d import export_gltf

    export_gltf(scene, tmp_path / "c.glb")
    back = import_gltf(tmp_path / "c.glb")
    (obj,) = back.scene.objects
    assert not back.lossy
    np.testing.assert_array_equal(obj.mesh.colors, colors)


def test_operations_keep_or_refuse_vertex_colors():
    scene = Scene("Ops")
    cube = create_cube(1.0)
    colors = np.tile(np.array([0.2, 0.4, 0.6, 1.0], np.float32), (cube.vertex_count, 1))
    obj = scene.add_object("Cube", mesh=cube.with_colors(colors))
    with obj.edit_mesh() as e:  # editing keeps them (per face corner)
        e.translate(e.select_faces(), (0.0, 0.0, 0.1))
    assert obj.mesh.colors is not None
    np.testing.assert_allclose(np.unique(obj.mesh.colors, axis=0), colors[:1])
    with pytest.raises(AetherError) as info, obj.sculpt():  # sculpting would drop them
        pass
    assert getattr(info.value, "code", "") == "VERTEX_COLORS_UNSUPPORTED"


def test_renderers_multiply_the_base_colour_by_vertex_colours():
    scene = Scene("Shade")
    obj = scene.add_object("Plane", mesh=colored_plane((1.0, 0.0, 0.0, 1.0)))
    obj.appearance.configure(color="#ffffff")
    scene.add_camera("Cam", position=(0.0, 0.0, 6.0), look_at=(0.0, 0.0, 0.0))
    scene.add_directional_light("Sun", direction=(0.0, 0.0, -1.0), intensity=1.0)
    img = scene.render(width=32, height=32, default_lights=False).image.astype(int)
    centre = img[16, 16]
    assert centre[0] > 100 and centre[1] < 10 and centre[2] < 10  # red, whatever the colour


def test_gpu_vertex_colours_match_the_cpu(gpu):
    scene = Scene("Shade")
    plane = create_sphere(1.5, 24, 12)
    rng = np.random.default_rng(5)
    colors = rng.uniform(0.2, 1.0, (plane.vertex_count, 4)).astype(np.float32)
    colors[:, 3] = 1.0
    obj = scene.add_object("Plane", mesh=plane.with_colors(colors))
    obj.appearance.configure(color="#c0c0c0")
    scene.add_camera("Cam", position=(1.0, -3.0, 5.0), look_at=(0.0, 0.0, 0.0))
    scene.add_directional_light("Sun", direction=(0.2, 0.3, -1.0), intensity=1.0)
    cpu = scene.render(width=96, height=64, samples=2)
    on_gpu = gpu.render(scene, width=96, height=64, samples=2, supersample=2, msaa=1)
    assert np.abs(cpu.image.astype(int) - on_gpu.image.astype(int)).max() <= 1
