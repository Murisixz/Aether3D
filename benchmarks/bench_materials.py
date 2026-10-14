"""Benchmark materials and textures: decoding, mipmaps, procedural nodes, shading, uploads.

Run from the repository root (``aether3d[images]`` for the JPEG rows,
``aether3d[gpu]`` for the GPU columns, ``aether3d[boolean]`` for the castle):

    python benchmarks/bench_materials.py [--quick] [--adapter SPEC] [--cpu-only]
        [--json results.json]

1. **Decoding and mipmaps** -- Aether3D's own PNG decoder and Pillow's JPEG
   decoder on photograph-like images (the example stone texture tiled, with
   per-pixel noise so that nothing repeats) of 512^2, 2048^2 and 4096^2
   pixels, and building the mip chain in each decoding mode (sRGB color,
   linear data, unit normals). Megapixels per second, and the bytes a chain
   holds (4/3 of its base level).
2. **Procedural nodes** -- the CPU evaluation of each node type over 2^20
   points (nanoseconds per point) and the size of the GPU program it compiles
   to.
3. **Shading cost** -- one scene (a grid of 24 UV spheres on a ground plane)
   rendered with flat colors, then with each kind of material on every
   object: the simple and standard models with constant factors, a UV-mapped
   image, a triplanar image, an image with a normal map, procedural noise,
   fbm, voronoi, bricks and wood, and the stone preset (color, roughness and
   normal textures). CPU and GPU, 4 samples per pixel, at 640x360 and
   1920x1080 (``--quick``: 640x360 on the CPU). Seconds and the cost over
   flat colors.
4. **Texture uploads** -- the first GPU render after a 4096^2 texture changes
   (decode, mip chain, texel upload) versus the next ones (everything cached),
   with the texel pool's size.
5. **The castle** (``examples/minecraft_castle_render.py``) in flat colors and
   with its materials, at 1920x1080 on both backends, with peak host memory.

Times are wall-clock seconds of a complete ``scene.render()``, best of
several runs after a warm-up render (decoded images and mip chains are cached
by content, GPU texels by image). Peak host memory is measured with
``tracemalloc``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

import aether3d
from aether3d import Scene, create_sphere, gpu_status, release_gpu
from aether3d.materials import IMAGE_CACHE, NODE_TYPES, TextureGraph, decode_image, jpeg_available
from aether3d.materials.graph import EvalContext, evaluate
from aether3d.materials.sampling import build_mip_chain
from aether3d.rendering.png import encode_png

ROOT = Path(__file__).resolve().parent.parent
TEXTURES = ROOT / "examples" / "textures"
SIZES = (512, 2048, 4096)
NODE_POINTS = 2**20


def hardware(adapter, gpu: bool) -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    selected = (gpu_status(adapter=adapter)["selected"] or {}) if gpu else {}
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aether3d": aether3d.__version__,
        "gpu": selected.get("adapter"),
        "cpu_renderer": "single-threaded NumPy",
    }


def warm_up(scene: Scene, seconds: float = 0.5, **options) -> None:
    """Render for ``seconds`` (at least 3 times): pipelines, uploads, and the clocks of a
    laptop GPU, which drop while the CPU works."""
    end = time.perf_counter() + seconds
    count = 0
    while count < 3 or time.perf_counter() < end:
        scene.render(**options)
        count += 1


def best_of(repeat: int, fn, *args, **kwargs):
    """``(fastest wall-clock seconds, last value)`` of ``repeat`` calls."""
    best, value = None, None
    for _ in range(repeat):
        t0 = time.perf_counter()
        value = fn(*args, **kwargs)
        seconds = time.perf_counter() - t0
        best = seconds if best is None else min(best, seconds)
    return best, value


# --------------------------------------------------------------------------- #
# 1. Decoding and mipmaps
# --------------------------------------------------------------------------- #


def photo_like(size: int, seed: int = 0) -> np.ndarray:
    """``(size, size, 4)`` RGBA: the stone texture tiled, plus noise so nothing repeats."""
    base = decode_image((TEXTURES / "stone_bricks_color.png").read_bytes()).pixels
    reps = -(-size // base.shape[0])
    tiled = np.tile(base, (reps, reps, 1))[:size, :size].astype(np.int16)
    noise = np.random.default_rng(seed).integers(-6, 7, (size, size, 3), dtype=np.int16)
    tiled[..., :3] += noise
    return np.clip(tiled, 0, 255).astype(np.uint8)


def decoding(quick: bool) -> list[dict]:
    rows = []
    sizes = SIZES[:2] if quick else SIZES
    for size in sizes:
        pixels = photo_like(size)
        encoded = {"png": encode_png(pixels)}
        if jpeg_available():
            import io

            from PIL import Image

            buffer = io.BytesIO()
            Image.fromarray(pixels[..., :3]).save(buffer, format="JPEG", quality=90)
            encoded["jpeg"] = buffer.getvalue()
        repeat = 2 if size >= 4096 else 3
        for kind, data in encoded.items():
            seconds, image = best_of(repeat, decode_image, data)
            row = {
                "format": kind,
                "size": size,
                "file_bytes": len(data),
                "decode_seconds": seconds,
                "megapixels_per_second": size * size / 1e6 / seconds,
            }
            for mode in ("srgb", "linear", "normal"):
                mip, chain = best_of(repeat, build_mip_chain, image.pixels, mode)
                row[f"mips_{mode}_seconds"] = mip
                row["chain_bytes"] = chain.nbytes
            rows.append(row)
            print(
                f"  {kind:4s} {size:>4d}^2 decode {seconds:.3f}s mips (srgb) "
                f"{row['mips_srgb_seconds']:.3f}s",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------- #
# 2. Procedural nodes
# --------------------------------------------------------------------------- #


def nodes(quick: bool) -> list[dict]:
    rng = np.random.default_rng(1)
    n = NODE_POINTS // 4 if quick else NODE_POINTS
    p = rng.uniform(0.0, 4.0, (n, 3))
    d = np.full((n, 3), 1e-3)
    ctx = EvalContext(p, d, d)
    rows = []
    for node_type in NODE_TYPES:
        if node_type == "image":  # needs a texture; measured by the shading rows
            continue
        graph = TextureGraph.from_dict(
            {"type": node_type, "octaves": 5} if node_type == "fbm" else {"type": node_type}
        )
        seconds, _ = best_of(2 if quick else 3, evaluate, graph, ctx)
        rows.append({"node": node_type, "ns_per_point": seconds / n * 1e9})
        print(f"  {node_type:12s} {rows[-1]['ns_per_point']:8.1f} ns/point", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# 3. Shading cost per kind of material
# --------------------------------------------------------------------------- #


def spheres_scene() -> Scene:
    scene = Scene("Material spheres")
    scene.add_plane("Ground", size=16).color = "#7a8a6a"
    ball = create_sphere(0.6, 48, 24)
    for i in range(6):
        for j in range(4):
            obj = scene.add_object(
                f"Ball {i} {j}", mesh=ball, position=(1.5 * i - 3.75, 1.5 * j - 2.25, 0.6)
            )
            obj.color = (0.3 + 0.1 * i, 0.4, 0.3 + 0.15 * j)
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1), intensity=2.2)
    scene.world.configure(ambient_intensity=0.3)
    scene.add_camera("Camera", fov=40).frame(direction=(-0.3, 1, -0.6), aspect=16 / 9)
    return scene


def dress(scene: Scene, kind: str, texture_dir: Path) -> None:
    """Give every object (ground and spheres) one material of ``kind``."""
    if kind == "flat colors":
        return
    if kind == "simple":
        material = scene.create_material("M", model="simple", base_color="#a08060")
    elif kind == "standard":
        material = scene.create_material("M", base_color="#a08060", roughness=0.5, metallic=0.2)
    elif kind == "stone preset":
        material = scene.create_material("M", preset="stone")
    else:
        material = scene.create_material("M", roughness=0.6)
        color = texture_dir / "stone_bricks_color.png"
        if kind == "image (uv)":
            material.set_texture("base_color", color, scale=4)
        elif kind == "image (triplanar)":
            material.set_texture("base_color", color, coordinates="object", projection="triplanar")
        elif kind == "image + normal map":
            material.set_texture("base_color", color, scale=4)
            material.set_texture("normal", texture_dir / "stone_bricks_normal.png", scale=4)
        else:
            node = kind.split()[0]
            spec = {"type": node, "scale": 3.0}
            if node == "fbm":
                spec["octaves"] = 5
            material.set_procedural_texture(
                "base_color", spec, coordinates="object", projection="volume"
            )
    scene.assign_material(meshes(scene), material)


def meshes(scene: Scene) -> list:
    return [obj for obj in scene if obj.kind == "mesh"]


KINDS = (
    "flat colors",
    "simple",
    "standard",
    "image (uv)",
    "image (triplanar)",
    "image + normal map",
    "noise",
    "fbm (5 octaves)",
    "voronoi",
    "bricks",
    "wood",
    "stone preset",
)


def shading(quick: bool, gpu: bool, adapter) -> list[dict]:
    rows = []
    cpu_sizes = [(640, 360)] if quick else [(640, 360), (1920, 1080)]
    gpu_sizes = [(640, 360), (1920, 1080)]
    for kind in KINDS:
        scene = spheres_scene()
        dress(scene, kind, TEXTURES)
        warm = scene.render(width=64, height=36, samples=1)  # caches, and a sanity check
        problems = [d.code for d in warm.diagnostics if d.severity != "info"]
        if problems:
            raise SystemExit(f"{kind}: unexpected diagnostics {problems}")
        row = {"kind": kind, "triangles": warm.stats["triangles"]}
        for w, h in cpu_sizes:
            seconds, _ = best_of(1 if w > 640 else 3, scene.render, width=w, height=h)
            row[f"cpu_{w}x{h}"] = seconds
        if gpu:
            release_gpu()  # so that the texel pool holds this scene's images only
            options = {"backend": "gpu", "adapter": adapter, "samples": 2}
            for w, h in gpu_sizes:
                warm_up(scene, width=w, height=h, **options)
                seconds, result = best_of(5, scene.render, width=w, height=h, **options)
                row[f"gpu_{w}x{h}"] = seconds
                row[f"gpu_pass_{w}x{h}"] = sum(result.stats["gpu_timings"].values())
            resources = result.backend["resources"]
            row["texture_program_instructions"] = resources["texture_program_instructions"]
            row["texel_mib"] = resources["texel_pool"]["bytes"] / 2**20
        rows.append(row)
        print(
            f"  {kind:20s} "
            + " ".join(f"{k} {v:.3f}s" for k, v in row.items() if k.startswith(("cpu_", "gpu_")))
            + "",
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------- #
# 4. Texture uploads
# --------------------------------------------------------------------------- #


def uploads(adapter, work: Path) -> dict:
    """First render after a 4096^2 texture changes vs the next one, on the GPU."""
    path = work / "big.png"
    path.write_bytes(encode_png(photo_like(4096, 1)))
    scene = spheres_scene()
    material = scene.create_material("M", roughness=0.6)
    material.set_texture("base_color", path, scale=4)
    scene.assign_material(meshes(scene), material)
    options = {"backend": "gpu", "adapter": adapter, "width": 640, "height": 360}
    scene.render(**options)  # warm: device, pipelines, meshes
    path.write_bytes(encode_png(photo_like(4096, 2)))  # a new image: nothing of it is cached
    rows: dict = {}
    t0 = time.perf_counter()
    first = scene.render(**options)
    rows["first_render_seconds"] = time.perf_counter() - t0
    rows["host_texture_seconds"] = first.stats["timings"]["textures"]
    res = first.backend["resources"]
    rows["texel_upload_bytes"] = res["texel_upload_bytes"]
    rows["texel_upload_seconds"] = res["texel_upload_seconds"]
    rows["texel_pool_bytes"] = res["texel_pool"]["bytes"]
    seconds, nxt = best_of(5, scene.render, **options)
    rows["next_render_seconds"] = seconds
    rows["next_texel_upload_bytes"] = nxt.backend["resources"]["texel_upload_bytes"]
    rows["image_cache"] = IMAGE_CACHE.stats()
    print(
        f"  4096^2 texture: first render {rows['first_render_seconds']:.3f}s (host decode and "
        f"mips {rows['host_texture_seconds']:.3f}s, upload {rows['texel_upload_seconds']:.3f}s), "
        f"next {seconds:.3f}s",
        flush=True,
    )
    return rows


# --------------------------------------------------------------------------- #
# 5. The castle
# --------------------------------------------------------------------------- #


def castle(gpu: bool, adapter) -> list[dict] | None:
    if importlib.util.find_spec("manifold3d") is None:
        return None
    spec = importlib.util.spec_from_file_location(
        "minecraft_castle_render", ROOT / "examples" / "minecraft_castle_render.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "examples"))
    spec.loader.exec_module(module)
    rows = []
    w, h = 1920, 1080
    for materials in (False, True):
        scene = module.build_scene(w, h, 2, materials=materials)
        scene.render(width=64, height=36)
        seconds, result = best_of(1, scene.render, width=w, height=h, samples=2)
        tracemalloc.start()
        scene.render(width=w, height=h, samples=2)
        peak = tracemalloc.get_traced_memory()[1] / 2**20
        tracemalloc.stop()
        row = {
            "materials": materials,
            "cpu_seconds": seconds,
            "cpu_shade_seconds": result.stats["timings"]["shade"],
            "cpu_peak_host_mib": peak,
        }
        if gpu:
            release_gpu()  # so that the texel pool holds this scene's images only
            options = {"backend": "gpu", "adapter": adapter, "width": w, "height": h, "samples": 2}
            warm_up(scene, **options)
            seconds, result = best_of(5, scene.render, **options)
            row["gpu_seconds"] = seconds
            row["gpu_pass_seconds"] = sum(result.stats["gpu_timings"].values())
            row["texel_bytes"] = result.backend["resources"]["texel_pool"]["bytes"]
        rows.append(row)
        print(
            f"  castle {'materials' if materials else 'flat colors'}: CPU {row['cpu_seconds']:.2f}s"
            + (f", GPU {row['gpu_seconds']:.3f}s" if gpu else ""),
            flush=True,
        )
    return rows


# --------------------------------------------------------------------------- #


def print_tables(data: dict) -> None:
    hw = data["hardware"]
    print(f"\nCPU: {hw['cpu']} ({hw['logical_cpus']} logical CPUs), {hw['platform']}")
    if hw["gpu"]:
        g = hw["gpu"]
        print(f"GPU: {g['name']} ({g['backend']}, {g['type']}, driver {g.get('driver')})")
    print(f"Python {hw['python']}, NumPy {hw['numpy']}, Aether3D {hw['aether3d']}")

    print("\n1. Decoding and mip chains (MP/s = megapixels per second)")
    print(
        f"  {'image':>12s} {'file':>9s} {'decode':>8s} {'MP/s':>6s} {'mips srgb':>10s} "
        f"{'linear':>8s} {'normal':>8s} {'chain':>9s}"
    )
    for r in data["decoding"]:
        print(
            f"  {r['format']:>4s} {r['size']:>4d}^2 {r['file_bytes'] / 2**20:7.2f}MB "
            f"{r['decode_seconds']:7.3f}s {r['megapixels_per_second']:6.1f} "
            f"{r['mips_srgb_seconds']:9.3f}s {r['mips_linear_seconds']:7.3f}s "
            f"{r['mips_normal_seconds']:7.3f}s {r['chain_bytes'] / 2**20:7.1f}MB"
        )

    print("\n2. Procedural nodes evaluated on the CPU (nanoseconds per point)")
    for r in data["nodes"]:
        print(f"  {r['node']:>12s} {r['ns_per_point']:9.1f}")

    rows = data["shading"]
    print(f"\n3. Shading cost ({rows[0]['triangles']:,} triangles, 4 samples per pixel)")
    keys = [k for k in rows[0] if k.startswith(("cpu_", "gpu_")) and "pass" not in k]
    texels = "texel_mib" in rows[0]
    print(
        "  "
        + f"{'material':>20s} "
        + " ".join(f"{k:>16s}" for k in keys)
        + (f" {'GPU texels':>11s}" if texels else "")
    )
    flat = rows[0]
    for r in rows:
        cells = []
        for k in keys:
            extra = f"({r[k] / flat[k]:.1f}x)" if r is not flat else ""
            cells.append(f"{r[k]:7.3f}s {extra:>7s}")
        line = f"  {r['kind']:>20s} " + " ".join(f"{c:>16s}" for c in cells)
        print(line + (f" {r['texel_mib']:7.1f} MiB" if texels else ""))

    if data.get("uploads"):
        u = data["uploads"]
        print(
            f"\n4. A new 4096^2 texture on the GPU: first render {u['first_render_seconds']:.3f} s "
            f"(host decode and mips {u['host_texture_seconds']:.3f} s; texel upload "
            f"{u['texel_upload_bytes'] / 2**20:.1f} MiB in {u['texel_upload_seconds']:.3f} s), "
            f"next renders {u['next_render_seconds']:.3f} s "
            f"({u['next_texel_upload_bytes']} bytes uploaded)"
        )
    if data.get("castle"):
        print("\n5. The castle at 1920x1080, 4 samples per pixel")
        for r in data["castle"]:
            label = "materials" if r["materials"] else "flat colors"
            line = (
                f"  {label:>12s}: CPU {r['cpu_seconds']:6.2f} s (shading "
                f"{r['cpu_shade_seconds']:.2f} s, peak host memory "
                f"{r['cpu_peak_host_mib']:.0f} MiB)"
            )
            if "gpu_seconds" in r:
                line += (
                    f"; GPU {r['gpu_seconds']:.3f} s (GPU passes "
                    f"{r['gpu_pass_seconds'] * 1e3:.1f} ms, texels {r['texel_bytes'] / 2**20:.1f}"
                    " MiB)"
                )
            print(line)


def main(argv: list[str]) -> int:
    quick = "--quick" in argv
    adapter = argv[argv.index("--adapter") + 1] if "--adapter" in argv else None
    gpu = "--cpu-only" not in argv and gpu_status(adapter=adapter)["available"]
    if not gpu and "--cpu-only" not in argv:
        print("GPU rendering is not available: CPU measurements only")
    data = {"hardware": hardware(adapter, gpu)}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        print("1. Decoding and mip chains")
        data["decoding"] = decoding(quick)
        print("2. Procedural nodes")
        data["nodes"] = nodes(quick)
        print("3. Shading cost per kind of material")
        data["shading"] = shading(quick, gpu, adapter)
        if gpu:
            print("4. Texture uploads")
            data["uploads"] = uploads(adapter, work)
        print("5. The castle")
        data["castle"] = castle(gpu, adapter)
    print_tables(data)
    if "--json" in argv:
        path = argv[argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
