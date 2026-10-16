"""MikkTSpace tangents: the port against the reference implementation, and the renderers'
normal mapping in that basis (CPU, GPU rasterizer, path tracers) against closed forms."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aether3d import Mesh, Scene
from aether3d.primitives import create_sphere
from aether3d.rendering.png import encode_png
from aether3d.uv.mikktspace import mesh_tangents, mikktspace

DATA = Path(__file__).parent / "data" / "mikktspace" / "reference.npz"
CASES = (
    "sphere",
    "cube",
    "noisy",
    "mirrored",
    "degenerate_uvs",
    "degenerate_positions",
    "non_manifold",
)


@pytest.mark.parametrize("case", CASES)
def test_port_matches_the_reference_bit_for_bit(case):
    data = np.load(DATA)
    got = mikktspace(data[f"{case}_positions"], data[f"{case}_normals"], data[f"{case}_uvs"])
    want = data[f"{case}_tangents"]
    assert np.array_equal(got.view(np.int32), want.view(np.int32))


def test_the_references_unsorted_last_edge_group_is_the_only_difference():
    # The reference leaves the last group of its edge sort unsorted (see
    # tests/data/mikktspace/README.md); on this mesh that group is a non-manifold edge shared
    # by a triangle and its duplicate, where the reference pairs triangle 77 with the
    # duplicate (108) and the port with triangle 90. Only the corner groups around the
    # edge's two vertices differ.
    data = np.load(DATA)
    p = data["quirk_positions"]
    got = mikktspace(p, data["quirk_normals"], data["quirk_uvs"])
    want = data["quirk_tangents"]
    corners = np.flatnonzero(np.any(got != want, axis=2).reshape(-1))
    at = p.reshape(-1, 3)[corners][:, :2]
    assert corners.size and np.all(at[:, 1] == 6) and set(at[:, 0].tolist()) == {3.0, 4.0}
    elsewhere = np.ones(got.shape[:2], bool).reshape(-1)
    elsewhere[corners] = False
    assert np.array_equal(got.reshape(-1, 4)[elsewhere], want.reshape(-1, 4)[elsewhere])


def test_corner_order_and_triangle_order_do_not_matter():
    data = np.load(DATA)
    p, n, uv = data["noisy_positions"], data["noisy_normals"], data["noisy_uvs"]
    base = mikktspace(p, n, uv)
    perm = np.random.default_rng(1).permutation(p.shape[0])
    shuffled = mikktspace(p[perm], n[perm], uv[perm])
    assert np.allclose(shuffled, base[perm], atol=2e-6)


def test_tangents_are_cached_on_the_mesh():
    mesh = create_sphere(1.0, 16, 8)
    a = mesh_tangents(mesh)
    assert mesh_tangents(mesh) is a
    assert not a.flags.writeable
    assert mesh_tangents(Mesh(mesh.positions, mesh.triangles, normals=mesh.normals)) is None


def test_plane_tangent_follows_u_and_v():
    # u along +x, v along +y on a +z plane: T = +x, sign +1 (B = cross(N, T) = +y).
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    mesh = Mesh(p, [[0, 1, 2], [0, 2, 3]], normals=[[0, 0, 1]] * 4, uvs=p[:, :2])
    t = mesh_tangents(mesh)
    assert np.allclose(t[..., :3], [1, 0, 0]) and np.all(t[..., 3] == 1)
    # Mirrored UVs (u along -x) keep T = dP/du = -x and flip the sign.
    mirrored = Mesh(p, [[0, 1, 2], [0, 2, 3]], normals=[[0, 0, 1]] * 4, uvs=p[:, :2] * [-1, 1])
    t = mesh_tangents(mirrored)
    assert np.allclose(t[..., :3], [-1, 0, 0]) and np.all(t[..., 3] == -1)


# --------------------------------------------------------------------------- #
# Rendering with UV normal maps
# --------------------------------------------------------------------------- #


def _tilt_map(angle: float) -> bytes:
    """A 4x4 tangent-space normal map tilted by ``angle`` toward +u."""
    v = np.array([math.sin(angle), 0.0, math.cos(angle)]) * 0.5 + 0.5
    rgb = np.round(v * 255).astype(np.uint8)
    return encode_png(np.broadcast_to(rgb, (4, 4, 3)).copy())


def _decoded(angle: float) -> np.ndarray:
    v = np.round((np.array([math.sin(angle), 0.0, math.cos(angle)]) * 0.5 + 0.5) * 255) / 255
    n = v * 2 - 1
    return n / np.linalg.norm(n)


def _plane_scene(angle: float, mirrored: bool = False) -> Scene:
    scene = Scene()
    scene.world.ambient_intensity = 0.0
    p = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32)
    uv = (p[:, :2] + 1) / 2
    mesh = Mesh(p, [[0, 1, 2], [0, 2, 3]], normals=[[0, 0, 1]] * 4, uvs=uv)
    obj = scene.add_object("Plane", mesh=mesh)
    if mirrored:
        obj.scale = (-1, 1, 1)
    m = scene.create_material("Bumps", model="simple", base_color="#ffffff", specular=0.0)
    tex = scene.add_image_texture("Tilt", data=_tilt_map(angle), filter="nearest", mipmaps=False)
    m.set_texture("normal", tex)
    obj.material = m
    # Light arriving from +x, 60 degrees above the plane.
    elevation = math.radians(60)
    scene.add_directional_light(
        "Sun", direction=(-math.cos(elevation), 0.0, -math.sin(elevation)), intensity=1.0
    )
    scene.add_camera("Cam", position=(0, 0, 5), look_at=(0, 0, 0), fov=20)
    return scene


@pytest.mark.parametrize("mirrored", [False, True])
def test_cpu_normal_map_tilts_toward_u_in_world(mirrored):
    angle = math.radians(25)
    scene = _plane_scene(angle, mirrored)
    result = scene.render(width=32, height=32, samples=1)
    n = _decoded(angle)
    # World normal: T = +x (mirrored instance: -x), B = +y, N = +z.
    world = np.array([n[0] * (-1 if mirrored else 1), n[1], n[2]])
    elevation = math.radians(60)
    to_light = np.array([math.cos(elevation), 0.0, math.sin(elevation)])
    expected = max(float(world @ to_light), 0.0)
    center = result.linear[14:18, 14:18, 0]
    assert np.allclose(center, expected, atol=1e-6)


@pytest.mark.parametrize("mirrored", [False, True])
def test_gpu_rasterizer_and_path_tracer_agree_on_the_tilt(gpu, mirrored):
    angle = math.radians(25)
    scene = _plane_scene(angle, mirrored)
    cpu = scene.render(width=32, height=32, samples=1)
    raster = gpu.render(scene, width=32, height=32, samples=1, supersample=1, msaa=1)
    assert np.allclose(raster.linear[14:18, 14:18], cpu.linear[14:18, 14:18], atol=2e-5)
    scene.render_settings.default_lights = False
    traced = scene.render(
        backend="gpu_path",
        adapter=gpu.adapter,
        width=16,
        height=16,
        path={"samples": 1, "bounces": 1},
    )
    cpu_traced = scene.render(
        backend="path", width=16, height=16, path={"samples": 1, "bounces": 1}
    )
    assert np.allclose(traced.linear[7:9, 7:9], cpu_traced.linear[7:9, 7:9], atol=2e-5)


def test_smooth_sphere_cpu_and_gpu_agree(gpu):
    scene = Scene()
    sphere = scene.add_object("Ball", mesh=create_sphere(1.0, 48, 24))
    rng = np.random.default_rng(3)
    n = rng.normal(size=(64, 64, 3)) * [0.4, 0.4, 0.0] + [0, 0, 1]
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    img = np.round((n * 0.5 + 0.5) * 255).astype(np.uint8)
    tex = scene.add_image_texture("Noise", data=encode_png(img))
    m = scene.create_material("Bumpy", roughness=0.6)
    m.set_texture("normal", tex)
    sphere.material = m
    scene.add_directional_light("Sun", direction=(-0.3, 0.5, -1.0), intensity=2.0)
    scene.add_camera("Cam", position=(0, -4, 1), look_at=(0, 0, 0), fov=35)
    cpu = scene.render(width=96, height=96, samples=1)
    out = gpu.render(scene, width=96, height=96, samples=1, supersample=1, msaa=1)
    covered = (cpu.object_index == out.object_index) & (cpu.object_index >= 0)
    diff = np.abs(cpu.linear - out.linear)[covered]
    assert np.mean(diff) < 2e-3


def test_cpu_path_tracer_uses_the_same_basis():
    angle = math.radians(25)
    scene = _plane_scene(angle)
    scene.render_settings.default_lights = False
    path = scene.render(backend="path", width=16, height=16, path={"samples": 1, "bounces": 1})
    # A directional light with no bounce light on a single plane: direct light only, the
    # same Lambert term (the path tracer's BRDF is the rasterizer's divided by pi, the
    # light intensity carries the pi).
    raster = scene.render(width=16, height=16, samples=1)
    assert np.allclose(path.linear[7:9, 7:9, 0], raster.linear[7:9, 7:9, 0], rtol=1e-6, atol=1e-7)


def test_gltf_exports_mikktspace_tangents(tmp_path):
    scene = Scene()
    sphere = scene.add_object("Ball", mesh=create_sphere(1.0, 16, 8))
    m = scene.create_material("Bumpy")
    m.set_texture("normal", scene.add_image_texture("Flat", data=_tilt_map(0.0)))
    sphere.material = m
    path = tmp_path / "ball.glb"
    scene.export(path)
    import json
    import struct

    raw = path.read_bytes()
    jlen = struct.unpack("<I", raw[12:16])[0]
    gltf = json.loads(raw[20 : 20 + jlen])
    prim = gltf["meshes"][0]["primitives"][0]
    assert "TANGENT" in prim["attributes"]
