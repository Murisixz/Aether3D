"""GPU renders that read back only some outputs (``render(outputs=...)``).

The selected outputs, the image, per-object visible samples, statistics and
diagnostics are exactly those of a render that reads everything back; the others
are ``None``. The checks that used the full readback -- non-finite linear values,
covered pixels -- are counted on the GPU and decide the same way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from parity import assert_bitwise

from aether3d import RenderError, Scene, create_cube, create_sphere

ALL = ("image", "linear", "depth", "object_index", "coverage")


def busy_scene() -> Scene:
    s = Scene("Outputs")
    for k in range(30):
        s.add_object(
            f"o{k}",
            mesh=create_cube(0.7) if k % 2 else create_sphere(0.4, 10, 5),
            position=(float(k % 6) - 2.5, float(k // 6) - 2.0, 0.4),
        )
    s.add_directional_light("Sun", direction=(-0.3, 0.5, -1.0))
    s.add_camera("Cam", position=(0, -9, 6), look_at=(0, 0, 0))
    return s


def array(result, name):
    return getattr(result, name)


@pytest.mark.parametrize(
    "outputs",
    [("image",), ("image", "depth"), ("linear",), ("object_index", "coverage"), "depth"],
)
def test_selected_outputs_are_those_of_a_full_render(gpu, outputs):
    s = busy_scene()
    full = gpu.render(s, width=80, height=60, shadows=True)
    some = gpu.render(s, width=80, height=60, shadows=True, outputs=outputs)
    chosen = {outputs} if isinstance(outputs, str) else set(outputs)
    for name in ALL:
        if name == "image" or name in chosen:
            assert_bitwise(array(some, name), array(full, name), name)
        else:
            assert array(some, name) is None, name
    assert some.objects == full.objects
    for key in ("pixels_covered", "samples_covered", "samples", "draw_calls"):
        assert some.stats[key] == full.stats[key], key
    assert full.stats["pixels_covered"] == int(np.count_nonzero(full.coverage > 0))
    assert [d.code for d in some.diagnostics] == [d.code for d in full.diagnostics]


def test_tone_mapping_reads_linear_light_back(gpu):
    s = busy_scene()
    full = gpu.render(s, width=40, height=30, tonemap="neutral")
    image_only = gpu.render(s, width=40, height=30, tonemap="neutral", outputs=("image",))
    assert_bitwise(image_only.image, full.image)


def _scale_exposure(monkeypatch, scale: float):
    from aether3d.rendering.gpu import renderer

    original = renderer.pack_finalize

    def pack(*args, **kwargs):
        data = original(*args, **kwargs)
        data[7] = scale  # the exposure multiplier
        return data

    monkeypatch.setattr(renderer, "pack_finalize", pack)


@pytest.mark.parametrize("outputs", [None, ("image",)])
def test_non_finite_values_are_found_without_reading_linear_light(gpu, monkeypatch, outputs):
    s = busy_scene()
    _scale_exposure(monkeypatch, math.inf)  # +inf and NaN (0 * inf) in linear light
    with pytest.raises(RenderError) as info:
        gpu.render(s, width=32, height=24, outputs=outputs)
    assert info.value.code == "RENDER_FAILED"


def test_negative_infinity_alone_is_accepted_as_before(gpu, monkeypatch):
    """The CPU-side check was ``isfinite(max(linear, initial=0))``: -inf never failed it.
    A view entirely covered by a lit plane, scaled by -inf, has only -inf values."""
    s = Scene("Covered")
    s.add_plane("Floor", size=200)
    s.world.configure(background="#ffffff", ambient_color="#ffffff", ambient_intensity=1.0)
    s.add_camera("Cam", position=(0, 0, 5), look_at=(0, 0, 0), up=(0, 1, 0))
    _scale_exposure(monkeypatch, -math.inf)
    full = gpu.render(s, width=16, height=12)
    assert np.all(full.linear == -np.inf)
    assert math.isfinite(float(np.max(full.linear, initial=0.0)))  # the old check passed
    gpu.render(s, width=16, height=12, outputs=("image",))  # and so does the new one


def test_hdr_files_need_linear_light(gpu, tmp_path):
    s = busy_scene()
    result = gpu.render(s, width=24, height=18, outputs=("image",))
    result.save(tmp_path / "a.png")
    with pytest.raises(RenderError) as info:
        result.save(tmp_path / "a.hdr")
    assert info.value.code == "OUTPUT_NOT_RENDERED"
    gpu.render(s, width=24, height=18, outputs=("linear",)).save(tmp_path / "a.pfm")


def test_outputs_are_checked_and_the_cpu_returns_every_output():
    s = busy_scene()
    with pytest.raises(RenderError) as info:
        s.render(width=8, height=6, outputs=("image", "normals"))
    assert info.value.code == "INVALID_SETTINGS" and "normals" in info.value.message
    result = s.render(width=8, height=6, outputs=("image",))
    assert all(array(result, name) is not None for name in ALL)
