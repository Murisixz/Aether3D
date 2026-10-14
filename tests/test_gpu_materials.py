"""Materials on the GPU: data layout (no GPU needed), and CPU/GPU parity on a real adapter.

Parity renders use the CPU renderer's sample positions on the GPU
(``supersample=samples, msaa=1``). Procedural patterns have hard edges, so a
handful of samples that land within float32 rounding of an edge may differ;
tests therefore bound the share of differing pixels instead of requiring
every pixel to match.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from aether3d import Mesh, Scene
from aether3d.errors import RenderError
from aether3d.materials.graph import NODE_TYPES, EvalContext, Program, evaluate, run_program
from aether3d.materials.sampling import sample
from aether3d.rendering.gpu import materials as gm
from aether3d.rendering.gpu import shaders
from aether3d.rendering.png import write_png
from aether3d.rendering.scene_data import extract_render_scene, resolve_camera

MATERIAL_WGSL = shaders.shader_source("material")


def test_shader_constants_match_the_packer():
    found = dict(re.findall(r"const (\w+): u32 = (\w+)u;", MATERIAL_WGSL))
    for name, value in gm.CONSTANTS.items():
        assert name in found, name
        assert int(found[name], 0) == value, name


def test_both_shader_modules_assemble_the_material_code():
    for kind in ("scene", "shadow"):
        source = shaders.module_source(kind)
        assert "fn run_program(" in source and "fn build_vertex(" in source
        assert source.count("fn camera_eye()") == 1


def png(path: Path, pixels) -> Path:
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:2] + (1,), 255, np.uint8)], -1)
    write_png(path, arr)
    return path


def material_scene(tmp_path: Path) -> Scene:
    rng = np.random.default_rng(0)
    scene = Scene("GPU materials")
    ground = scene.add_plane("Ground", size=10)
    ground.material = scene.create_material("Grass", preset="grass")
    photo = png(tmp_path / "photo.png", rng.integers(0, 256, (16, 16, 3)))
    crate = scene.create_material("Crate", roughness=0.6)
    crate.set_texture("base_color", photo)
    crate.set_procedural_texture("roughness", "fbm", scale=5, octaves=3)
    scene.add_cube("Crate", size=1.2, position=(-1.5, 0, 0.6)).material = crate
    scene.add_sphere(
        "Ball", radius=0.6, position=(0.3, -0.5, 0.6)
    ).material = scene.create_material(preset="metal")
    scene.add_cube("Bricks", size=1.0, position=(1.8, 0.6, 0.5)).material = scene.create_material(
        preset="brick"
    )
    leaf = scene.add_plane(
        "Leaf", size=1.2, position=(0.2, 1.4, 1.0), rotation=(0.7071068, 0, 0, 0.7071068)
    )
    cut = scene.create_material("Leaf", base_color="#3c8a2a", alpha_mode="mask")
    cut.set_procedural_texture(  # leaf-like cells: opaque inside, cut out along their borders
        "opacity",
        {
            "nodes": {
                "v": {"type": "voronoi", "scale": 4},
                "o": {
                    "type": "math",
                    "op": "smoothstep",
                    "a": {"node": "v", "output": "edge"},
                    "b": 0.02,
                    "c": 0.06,
                },
            },
            "output": "o",
        },
    )
    leaf.material = cut
    scene.add_directional_light("Sun", direction=(0.4, 0.6, -0.8), intensity=2.5)
    scene.add_point_light("Lamp", position=(-2, -2, 2.5), intensity=6, range=10)
    scene.world.configure(background="#8ec5f2", ambient_intensity=0.3)
    camera = scene.add_camera("Cam", fov=40)
    camera.frame(direction=(-0.3, 1.0, -0.6), aspect=1.5)
    return scene


def packed_program_values(words: np.ndarray, binding: int, ctx: EvalContext, images) -> np.ndarray:
    """Run a binding's program straight out of the packed buffer (CPU interpreter)."""
    b = words[gm.HEADER_BINDINGS] + binding * gm.BINDING_WORDS
    first, count = int(words[b + 1]), int(words[b + 2])
    start = words[gm.HEADER_PROGRAMS] + first * 8
    program_words = words[start : start + count * 8]
    params_start, params_end = words[gm.HEADER_PARAMS], words[gm.HEADER_IMAGES]
    params = words[params_start:params_end].view(np.float32)
    image_refs = [(str(i), "srgb") for i in range(len(images))]
    program = Program(program_words.copy(), params, image_refs, 16)

    def sampler(ref, space, p, dx, dy):
        image = images[int(ref)]
        return sample(image.chain, p[:, 0], p[:, 1], wrap=image.wrap, filter=image.filter)

    return run_program(program, EvalContext(ctx.p, ctx.dpdx, ctx.dpdy, sampler))


def test_packed_programs_evaluate_like_their_graphs(tmp_path):
    scene = material_scene(tmp_path)
    rs = extract_render_scene(scene, resolve_camera(scene, None))
    order = np.arange(len(rs.instances))
    data = gm.pack_material_data(rs, order)
    words = data.words
    p = np.random.default_rng(1).uniform(-2, 2, (500, 3))
    ctx = EvalContext(p, np.zeros_like(p), np.zeros_like(p))
    checked = 0
    for m_index, material in enumerate(rs.materials):
        record = words[
            gm.HEADER_WORDS and words[gm.HEADER_MATERIALS] + m_index * gm.MATERIAL_WORDS :
        ]
        assert record[0] & gm.MATERIAL_TEXTURED == (
            gm.MATERIAL_TEXTURED if material.textures else 0
        )
        assert np.allclose(record[1:4].view(np.float32), material.base_color, atol=1e-7)
        for slot, rt in material.textures.items():
            binding = int(record[16 + gm.SLOT_ORDER.index(slot)])
            got = packed_program_values(words, binding, ctx, data.images)
            if rt.kind == "image":
                expected = sample(
                    rt.image.chain, p[:, 0], p[:, 1], wrap=rt.image.wrap, filter=rt.image.filter
                )
            else:
                expected = evaluate(rt.graph, ctx)
            assert np.allclose(got, expected, atol=2e-5), (material.name, slot)
            checked += 1
    assert checked >= 8
    # Per-instance tables follow the draw order.
    table = words[words[gm.HEADER_INSTANCES] :].reshape(-1)[: len(order) * gm.INSTANCE_WORDS]
    assert [int(v) for v in table[:: gm.INSTANCE_WORDS]] == [
        rs.instances[i].material for i in order
    ]


def compare(cpu, gpu, *, mean=0.5, share=0.002, worst=None):
    same = cpu.object_index == gpu.object_index
    assert same.mean() >= 0.998, same.mean()
    diff = np.abs(cpu.image.astype(int) - gpu.image.astype(int)).max(-1)[same]
    assert diff.mean() <= mean, diff.mean()
    assert (diff > 8).mean() <= share, (diff > 8).mean()
    if worst is not None:
        assert diff.max() <= worst, diff.max()
    return diff


def test_textured_scenes_render_alike_on_cpu_and_gpu(gpu, tmp_path):
    scene = material_scene(tmp_path)
    cpu = scene.render(width=180, height=120, samples=2)
    image = gpu.render(scene, width=180, height=120, samples=2, supersample=2, msaa=1)
    assert [d.code for d in image.diagnostics if d.severity != "info"] == []
    compare(cpu, image)
    resources = image.backend["resources"]
    assert resources["textured_materials"] >= 5 and resources["images"] == 1


def test_textured_msaa_renders_repeat_exactly(gpu, tmp_path):
    """Image textures under MSAA render bit-identically run to run. Mip levels come from
    derivatives at pixel centers, never of centroid-sampled values, whose helper-lane values
    some GPUs leave undefined (Apple's: a few edge pixels changed from render to render)."""
    scene = material_scene(tmp_path)
    noise = png(tmp_path / "noise.png", np.random.default_rng(1).integers(0, 256, (64, 64, 3)))
    stone = scene.create_material("Stone")
    stone.set_texture("base_color", noise, coordinates="object", scale=3)  # triplanar
    globe = scene.add_sphere("Globe", radius=0.5, segments=48, rings=24, position=(-0.4, 0.9, 1.4))
    globe.material = stone  # many triangle edges inside one textured object
    first = gpu.render(scene, width=240, height=160, samples=2)
    assert first.backend["antialiasing"]["msaa"] > 1
    for _ in range(3):
        again = gpu.render(scene, width=240, height=160, samples=2)
        assert np.array_equal(again.image, first.image)


@pytest.mark.parametrize("node", sorted(set(NODE_TYPES) - {"image", "bump"}))
def test_every_procedural_node_matches_the_cpu(gpu, node):
    scene = Scene(node)
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    params = {"type": node}
    if NODE_TYPES[node].pattern:
        params["scale"] = 3.7
    m.set_procedural_texture("base_color", params)
    plane.material = m
    scene.world.configure(background="#000000", ambient_color="#ffffff", ambient_intensity=1.0)
    scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=2.0,
        position=(0, 0, 5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )
    cpu = scene.render(width=96, height=96, samples=1, default_lights=False)
    image = gpu.render(scene, width=96, height=96, samples=1, msaa=1, default_lights=False)
    compare(cpu, image, mean=0.3, share=0.01)


def test_normal_maps_bump_nodes_and_triplanar_projections_match(gpu, tmp_path):
    scene = Scene("Relief")
    cube = scene.add_cube("Cube", size=1.6)
    m = scene.create_material("M", base_color="#c0a080", roughness=0.4)
    m.set_procedural_texture(
        "normal",
        {
            "nodes": {
                "b": {"type": "bricks", "scale": 3},
                "n": {"type": "bump", "height": {"node": "b", "output": "mortar"}, "depth": -0.01},
            },
            "output": "n",
        },
        coordinates="object",
    )
    m.set_procedural_texture(
        "base_color", "checker", scale=2, color1="#806040", color2="#e0c080", coordinates="world"
    )
    cube.material = m
    ball = scene.add_sphere("Ball", radius=0.6, position=(1.6, 0.4, 0))
    bumpy = scene.create_material("Bumpy", base_color="#3a7bd5", roughness=0.3)
    stripes = np.zeros((32, 32, 3), np.uint8)
    stripes[...] = (128, 128, 255)
    stripes[:, ::4] = (200, 128, 220)
    bumpy.set_texture("normal", png(tmp_path / "n.png", stripes), scale=(4, 2))
    ball.material = bumpy
    scene.add_directional_light("Sun", direction=(0.5, 0.7, -0.6), intensity=2.5)
    scene.add_camera("Cam", position=(3.0, -4.0, 2.6), look_at=(0.5, 0, 0))
    cpu = scene.render(width=160, height=120, samples=2)
    image = gpu.render(scene, width=160, height=120, samples=2, supersample=2, msaa=1)
    compare(cpu, image, mean=0.6, share=0.005)


def test_alpha_masks_and_single_sided_faces_cover_the_same_samples(gpu):
    scene = Scene("Cut")
    scene.add_plane("Back", size=6, position=(0, 0, -1)).color = "#ff0000"
    front = scene.add_plane("Front", size=2)
    m = scene.create_material("Cut", base_color="#00ff00", alpha_mode="mask")
    m.set_procedural_texture("opacity", "checker", scale=4, color1="#000000", color2="#ffffff")
    front.material = m
    flipped = scene.add_plane("Flipped", size=1, position=(2, 0, 0.5), rotation=(0, 1, 0, 0))
    flipped.material = scene.create_material("One side", double_sided=False)
    scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=6.0,
        position=(0, 0, 5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )
    cpu = scene.render(width=120, height=120, samples=2, default_lights=False)
    image = gpu.render(
        scene, width=120, height=120, samples=2, supersample=2, msaa=1, default_lights=False
    )
    assert np.mean(cpu.object_index == image.object_index) >= 0.999
    names = [o["name"] for o in image.objects]
    assert image.objects[names.index("Flipped")]["visible_samples"] == 0
    cut = image.objects[names.index("Front")]["visible_samples"]
    full = 4 * (120 * 2 // 6) ** 2  # the plane's samples without holes
    assert 0.4 < cut / full < 0.6


def test_cut_out_materials_cast_shadows_with_holes(gpu):
    scene = Scene("Fence")
    scene.add_plane("Ground", size=8).color = "#c8c8c8"
    fence = scene.add_plane("Fence", size=2, position=(0, 0, 1.5))
    m = scene.create_material("Holes", base_color="#a05020", alpha_mode="mask")
    m.set_procedural_texture("opacity", "checker", scale=2, color1="#000000", color2="#ffffff")
    fence.material = m
    scene.add_directional_light("Sun", direction=(0.0, 0.0, -1.0), intensity=2.0)
    scene.world.configure(ambient_intensity=0.2)
    scene.add_camera("Cam", position=(0, -6, 6), look_at=(0, 0, 0.5))
    masked = gpu.render(scene, width=160, height=120, shadows=True)
    m.alpha_mode = "opaque"
    solid = gpu.render(scene, width=160, height=120, shadows=True)

    def dark_ground(result):
        ground = result.object_index == 0
        return int(np.count_nonzero(ground & (result.linear.sum(-1) < 1.0)))

    # The solid fence shades a 2 x 2 square of ground; the cut-out one about half of it.
    assert dark_ground(solid) > 500
    assert 0.3 < dark_ground(masked) / dark_ground(solid) < 0.7
    assert masked.backend["shadows"] is not None


def test_textures_upload_once_and_reupload_when_the_file_changes(gpu, tmp_path):
    from aether3d.rendering.gpu import device

    path = png(tmp_path / "t.png", np.full((4, 4, 3), (255, 0, 0), np.uint8))
    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M", model="simple", base_color="#ffffff")
    m.set_texture("base_color", path)
    plane.material = m
    scene.world.configure(ambient_color="#ffffff", ambient_intensity=1.0)
    scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=2.0,
        position=(0, 0, 5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )
    first = gpu.render(scene, width=8, height=8, default_lights=False)
    second = gpu.render(scene, width=8, height=8, default_lights=False)
    assert second.backend["resources"]["texel_uploads"] == 0
    assert first.pixel(4, 4) == second.pixel(4, 4) == (255, 0, 0)
    for result in (first, second):  # the time spent preparing textures is reported
        assert 0 <= result.stats["timings"]["textures"] <= result.stats["timings"]["total"]
    png(path, np.full((5, 4, 3), (0, 0, 255), np.uint8))  # edited: new content, new size
    third = gpu.render(scene, width=8, height=8, default_lights=False)
    assert third.backend["resources"]["texel_uploads"] == 1 and third.pixel(4, 4) == (0, 0, 255)
    context, _ = device.get_context(gpu.adapter)
    assert context.textures.stats()["images"] >= 1


def test_too_many_texels_fail_cleanly_and_auto_falls_back(gpu, tmp_path):
    from aether3d.rendering.gpu import device

    scene = Scene()
    plane = scene.add_plane("P", size=2)
    m = scene.create_material("M")
    m.set_texture("base_color", png(tmp_path / "big.png", np.zeros((256, 256, 3), np.uint8)))
    plane.material = m
    scene.add_camera("Cam", position=(0, -3, 3), look_at=(0, 0, 0))
    context, _ = device.get_context(gpu.adapter)
    saved = context.limits.get("max-storage-buffer-binding-size")
    context.textures.clear()
    context.limits["max-storage-buffer-binding-size"] = 64 * 1024
    try:
        with pytest.raises(RenderError) as info:
            gpu.render(scene, width=32, height=32)
        assert info.value.code == "GPU_LIMIT_EXCEEDED" and info.value.location == "textures"
        fallback = scene.render(width=32, height=32, backend="auto", adapter=gpu.adapter)
        assert fallback.backend["name"] == "cpu"
        assert "BACKEND_FALLBACK" in [d.code for d in fallback.diagnostics]
    finally:
        context.limits["max-storage-buffer-binding-size"] = saved


def test_face_groups_and_missing_uvs_on_the_gpu(gpu):
    mesh = Mesh(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]], groups=[0, 1]
    )
    scene = Scene()
    obj = scene.add_object("Q", mesh=mesh)
    blue = scene.create_material("Blue", model="simple", base_color="#0000ff")
    blue.set_procedural_texture(
        "base_color", "checker", color1="#000000", color2="#000000"
    )  # no UVs: ignored
    obj.material = blue
    scene.assign_material(
        obj, scene.create_material("Red", model="simple", base_color="#ff0000"), group=1
    )
    scene.world.configure(background="#000000", ambient_color="#ffffff", ambient_intensity=1.0)
    scene.add_camera(
        "Top",
        projection="orthographic",
        ortho_scale=2.0,
        position=(0, 0, 5),
        look_at=(0, 0, 0),
        up=(0, 1, 0),
    )
    image = gpu.render(scene, width=16, height=16, default_lights=False)
    assert image.pixel(12, 12) == (0, 0, 255) and image.pixel(3, 3) == (255, 0, 0)
    assert "MISSING_UVS" in [d.code for d in image.diagnostics]
