"""Glass (0.20.0): transmission and ior on standard materials -- the rough dielectric lobe
against brute-force integrals and closed forms, white furnaces, a slab's transmittance,
both path tracers, the rasterizers' reflection-only drawing and glTF."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from aether3d import Scene, export_gltf, import_gltf
from aether3d.rendering.gpu import gpu_status
from aether3d.rendering.pathtrace.bsdf import fresnel_dielectric, glass_eval, glass_sample

GPU = gpu_status().get("available", False)
TRACERS = ["path", *(["gpu_path"] if GPU else [])]


def test_fresnel_matches_the_closed_forms():
    for eta in (1.33, 1.5, 2.4):
        r0 = ((eta - 1) / (eta + 1)) ** 2
        assert fresnel_dielectric(np.array([1.0]), np.array([eta]))[0] == pytest.approx(r0)
        inside = 1.0 / eta  # leaving the denser medium: total reflection past asin(1 / eta)
        critical = math.asin(inside)
        beyond = np.array([math.cos(critical + 0.01)])
        assert fresnel_dielectric(beyond, np.array([inside]))[0] == 1.0
        assert fresnel_dielectric(np.array([1.0]), np.array([inside]))[0] == pytest.approx(r0)


CASES = [(0.3, 1.5, 30), (0.3, 1 / 1.5, 30), (0.6, 1 / 1.33, 50), (0.2, 1.5, 65)]


@pytest.mark.parametrize(("alpha", "eta", "theta"), CASES)
def test_sampling_estimates_the_integral_of_the_lobe(alpha, eta, theta):
    rng = np.random.default_rng(11)
    n = 1_000_000
    th = math.radians(theta)
    wo = np.tile([math.sin(th), 0.0, math.cos(th)], (n, 1))
    a, e, tint = np.full(n, alpha), np.full(n, eta), np.ones((n, 3))
    z = 2 * rng.random(n) - 1
    phi = 2 * math.pi * rng.random(n)
    r = np.sqrt(1 - z * z)
    wi = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)  # uniform on the sphere
    f, pdf = glass_eval(wo, wi, a, e, tint)
    brute = (f[:, 0] * np.abs(wi[:, 2])).mean() * 4 * math.pi
    assert pdf.mean() * 4 * math.pi <= 1.0 + 0.02  # a density (some samples are discarded)
    u = rng.random((n, 3))
    ws, good = glass_sample(wo, a, e, u[:, 0], u[:, 1], u[:, 2])
    fs, ps = glass_eval(wo, ws, a, e, tint)
    ok = good & (ps > 0)
    est = np.where(ok, fs[:, 0] * np.abs(ws[:, 2]) / np.where(ok, ps, 1.0), 0.0).mean()
    assert est == pytest.approx(brute, rel=0.02)  # (the brute-force estimate's noise)


def glass_ball(roughness: float) -> Scene:
    scene = Scene()
    glass = scene.create_material("Glass", model="standard", base_color="#ffffff",
                                  transmission=1.0, ior=1.5, roughness=roughness)  # fmt: skip
    scene.add_sphere("Ball", radius=1.0, segments=96, rings=48).material = glass
    scene.add_camera("Cam", position=(0, -5, 0), look_at=(0, 0, 0), fov=30)
    scene.world.environment.configure(type="uniform", color="#ffffff", intensity=1.0)
    return scene


@pytest.mark.parametrize("backend", TRACERS)
def test_a_glass_ball_in_a_white_furnace_neither_gains_nor_loses_light(backend):
    path = {"samples": 32, "bounces": 32, "russian_roulette": 64}
    r = glass_ball(0.0).render(width=40, height=40, backend=backend, default_lights=False,
                               path=path)  # fmt: skip
    ball = r.linear[r.object_index >= 0].astype(np.float64)
    assert ball.mean() == pytest.approx(1.0, abs=0.002) and ball.max() <= 1.0 + 1e-4
    # Rough glass loses what single-scattering microfacets lose (measured: 5 % at 0.3).
    r = glass_ball(0.3).render(width=40, height=40, backend=backend, default_lights=False,
                               path=path)  # fmt: skip
    assert 0.93 < r.linear[r.object_index >= 0].mean() < 0.97


@pytest.mark.parametrize("backend", TRACERS)
def test_a_glass_slab_transmits_what_its_two_interfaces_let_through(backend):
    # At normal incidence each interface reflects R = 0.04; with every internal reflection
    # a slab transmits (1 - R)^2 / (1 - R^2) = (1 - R) / (1 + R) of what is behind it.
    scene = Scene()
    glass = scene.create_material("Glass", preset="glass")
    slab = scene.add_cube("Slab", size=1.0, position=(0, 0, 0))
    slab.scale = (3.0, 0.05, 3.0)
    slab.material = glass
    lamp = scene.create_material("Lamp", model="standard", base_color="#000000",
                                 emissive="#ffffff", emissive_strength=1.0)  # fmt: skip
    panel = scene.add_cube("Panel", size=1.0, position=(0, 2.0, 0))
    panel.scale = (20.0, 0.1, 20.0)
    panel.material = lamp
    scene.add_camera("Cam", position=(0, -4, 0), look_at=(0, 0, 0), fov=4)
    scene.world.ambient_intensity = 0.0
    r = scene.render(width=8, height=8, backend=backend, default_lights=False,
                     path={"samples": 64, "bounces": 16, "russian_roulette": 32})  # fmt: skip
    rr = 0.04
    assert r.linear[..., 1].mean() == pytest.approx((1 - rr) / (1 + rr), rel=0.004)


@pytest.mark.skipif(not GPU, reason="needs a GPU")
def test_the_gpu_path_tracer_traces_the_cpu_paths_through_glass():
    scene = glass_ball(0.15)
    scene.world.environment.configure(type="image", image="builtin:studio")
    floor = scene.add_cube("Floor", size=1.0, position=(0, 0, -1.5))
    floor.scale = (10, 10, 1)
    opts = {"width": 40, "height": 40, "default_lights": False,
            "path": {"samples": 16, "bounces": 6}}  # fmt: skip
    a = scene.render(backend="path", **opts).linear
    b = scene.render(backend="gpu_path", **opts).linear
    d = np.abs(a - b) / a.max()
    assert np.median(d) < 1e-5 and np.percentile(d, 95) < 1e-3


@pytest.mark.parametrize("backend", ["cpu", *(["gpu"] if GPU else [])])
def test_the_rasterizers_draw_only_the_reflection_of_glass(backend):
    def ball(transmission: float):
        scene = Scene()
        m = scene.create_material("M", model="standard", base_color="#ffffff",
                                  transmission=transmission, roughness=0.5)  # fmt: skip
        scene.add_sphere("Ball", radius=1.0).material = m
        scene.add_camera("Cam", position=(0, -5, 0), look_at=(0, 0, 0))
        scene.add_directional_light("Sun", direction=(0.3, 1, -0.5))
        scene.world.ambient_intensity = 0.0
        opts = {"supersample": 1, "msaa": 1} if backend == "gpu" else {"samples": 1}
        return scene.render(width=32, height=32, backend=backend, **opts).linear

    opaque, half, glass = ball(0.0), ball(0.5), ball(1.0)
    assert glass.sum() < half.sum() < opaque.sum()
    assert glass.max() > 0.0  # the highlight stays


def test_glass_goes_through_gltf(tmp_path):
    scene = Scene()
    scene.add_cube("Pool").material = scene.create_material("Water", preset="water")
    path = tmp_path / "glass.gltf"
    export_gltf(scene, path)
    doc = json.loads(path.read_text())
    record = doc["materials"][0]["extensions"]
    assert record["KHR_materials_transmission"] == {"transmissionFactor": 1.0}
    assert record["KHR_materials_ior"] == {"ior": pytest.approx(1.33)}
    assert {"KHR_materials_transmission", "KHR_materials_ior"} <= set(doc["extensionsUsed"])
    for o in doc["materials"]:  # without Aether3D's extras, the extensions carry it
        o.pop("extras", None)
    path.write_text(json.dumps(doc))
    got = import_gltf(path).scene.get_by_name("Pool").material
    assert got.params["transmission"] == 1.0 and got.params["ior"] == pytest.approx(1.33)


def test_glass_exports_pass_the_khronos_validator(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    tool = Path(__file__).resolve().parent.parent / "tools" / "gltf-validator"
    if not (shutil.which("node") and (tool / "node_modules" / "gltf-validator").is_dir()):
        pytest.skip("Khronos glTF Validator not installed")
    scene = Scene()
    scene.add_cube("Pane").material = scene.create_material("Glass", preset="frosted_glass")
    scene.add_sphere("Drop", position=(2, 0, 0)).material = scene.create_material(
        "Water", preset="water"
    )
    result = scene.export(tmp_path / "glass.glb")
    assert result.diagnostics == []
    out = subprocess.run(["node", str(tool / "validate.mjs"), str(result.path)],
                         capture_output=True, text=True, encoding="utf-8", check=False)  # fmt: skip
    report = json.loads(out.stdout.splitlines()[0])
    assert report["ok"] and report["errors"] == 0 and report["warnings"] == 0, report
