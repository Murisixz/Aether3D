"""Texture sampling: mip chains, wrap modes, filtering, level of detail and color decoding."""

import numpy as np
import pytest

from aether3d.materials.sampling import (
    SRGB_TABLE,
    build_mip_chain,
    level_count,
    lod,
    mip_bytes,
    sample,
    wrap_index,
)
from aether3d.rendering.color import srgb_to_linear


def rgba(pixels) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.uint8)
    if arr.shape[-1] == 3:
        arr = np.concatenate([arr, np.full(arr.shape[:-1] + (1,), 255, np.uint8)], axis=-1)
    return arr


def test_level_counts_and_sizes_follow_floor_halving():
    assert level_count(1, 1) == 1 and level_count(2, 1) == 2 and level_count(256, 256) == 9
    assert level_count(300, 7) == 9
    chain = build_mip_chain(np.zeros((7, 300, 4), np.uint8), "linear")
    sizes = [lv.shape[:2] for lv in chain.levels]
    assert sizes == [(7, 300), (3, 150), (1, 75), (1, 37), (1, 18), (1, 9), (1, 4), (1, 2), (1, 1)]
    assert chain.nbytes == mip_bytes(300, 7) == sum(lv.nbytes for lv in chain.levels)
    assert all(not lv.flags.writeable for lv in chain.levels)
    single = build_mip_chain(np.zeros((8, 8, 4), np.uint8), "srgb", mipmaps=False)
    assert len(single.levels) == 1 and single.nbytes == mip_bytes(8, 8, False) == 256


def test_srgb_levels_average_in_linear_light():
    # Black and white texels: the linear average is 0.5, which is sRGB code 188, not 128.
    level = rgba([[[0, 0, 0], [255, 255, 255]], [[0, 0, 0], [255, 255, 255]]])
    srgb = build_mip_chain(level, "srgb").levels[1]
    linear = build_mip_chain(level, "linear").levels[1]
    assert srgb[0, 0, :3].tolist() == [188, 188, 188]
    assert linear[0, 0, :3].tolist() == [128, 128, 128]  # round(127.5) to even
    assert srgb[0, 0, 3] == linear[0, 0, 3] == 255  # alpha is always linear


def test_normal_levels_renormalize():
    up = np.array([128, 128, 255])
    tilted = np.round((np.array([0.6, 0.0, 0.8]) * 0.5 + 0.5) * 255).astype(int)
    level = rgba(np.array([[up, tilted], [up, tilted]]))
    n = build_mip_chain(level, "normal").levels[1][0, 0, :3] / 255.0 * 2.0 - 1.0
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=0.01)
    plain = build_mip_chain(level, "linear").levels[1][0, 0, :3] / 255.0 * 2.0 - 1.0
    assert np.linalg.norm(plain) < 0.96  # plain averaging shortens it


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("repeat", [1, 2, 0, 1, 2, 0, 1]),
        ("clamp", [0, 0, 0, 1, 2, 2, 2]),
        ("mirror", [1, 0, 0, 1, 2, 2, 1]),
    ],
)
def test_wrap_modes(mode, expected):
    assert wrap_index(np.arange(-2, 5), 3, mode).tolist() == expected


def test_nearest_sampling_picks_the_texel_containing_the_point():
    # 4 x 2 image, row 0 is the top (v = 1 side).
    codes = np.arange(8, dtype=np.uint8).reshape(2, 4) * 30
    chain = build_mip_chain(rgba(np.repeat(codes[..., None], 3, -1)), "linear", mipmaps=False)
    u = np.array([0.1, 0.3, 0.6, 0.9, 0.1, 0.9])
    v = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
    got = sample(chain, u, v, filter="nearest")[:, 0] * 255
    assert np.round(got).tolist() == [0, 30, 60, 90, 120, 210]


def test_bilinear_sampling_interpolates_between_texel_centers():
    chain = build_mip_chain(rgba([[[0, 0, 0], [255, 255, 255]]]), "linear", mipmaps=False)
    u = np.array([0.25, 0.5, 0.75, 0.375])
    got = sample(chain, u, np.full(4, 0.5), wrap=("clamp", "clamp"))[:, 0]
    assert got == pytest.approx([0.0, 0.5, 1.0, 0.25])
    # With repeat, sampling at the left edge blends the two edge texels.
    edge = sample(chain, np.array([0.0]), np.array([0.5]), wrap=("repeat", "repeat"))[0, 0]
    assert edge == pytest.approx(0.5)


def test_srgb_texels_are_decoded_before_filtering():
    chain = build_mip_chain(rgba([[[0, 0, 0], [255, 255, 255]]]), "srgb", mipmaps=False)
    mid = sample(chain, np.array([0.5]), np.array([0.5]), wrap=("clamp", "clamp"))[0]
    assert mid[0] == pytest.approx(0.5)  # half of linear white, not srgb_to_linear(0.5)
    one = build_mip_chain(rgba([[[128, 64, 32]]]), "srgb", mipmaps=False)
    got = sample(one, np.array([0.5]), np.array([0.5]))[0, :3]
    assert got == pytest.approx(srgb_to_linear(np.array([128, 64, 32]) / 255.0))
    assert SRGB_TABLE[255] == 1.0 and SRGB_TABLE[0] == 0.0


def test_lod_selects_levels_by_footprint():
    chain = build_mip_chain(np.zeros((64, 64, 4), np.uint8), "linear")
    texel = 1.0 / 64
    lam = lod(
        chain,
        np.array([texel, 4 * texel, 0.0]),
        np.zeros(3),
        np.zeros(3),
        np.array([texel, 0.0, 16 * texel]),
    )
    assert lam == pytest.approx([0.0, 2.0, 4.0])


def test_trilinear_blends_levels_and_nearest_picks_one():
    base = np.zeros((4, 4, 4), np.uint8)
    base[..., 3] = 255
    base[::2, ::2, :3] = 255  # level 0 has contrast, level 1+ average to gray
    chain = build_mip_chain(base, "linear")
    u = np.full(3, 0.125)
    v = np.full(3, 1.0 - 0.125)
    lam = np.array([0.0, 0.5, 1.0])
    got = sample(chain, u, v, filter="linear", lam=lam)[:, 0]
    level1 = chain.levels[1][0, 0, 0] / 255.0
    assert got[0] == pytest.approx(1.0)
    assert got[2] == pytest.approx(level1)
    assert got[1] == pytest.approx(0.5 * (1.0 + level1))
    nearest = sample(chain, u, v, filter="nearest", lam=np.array([0.4, 0.6, 3.9]))[:, 0]
    assert nearest[0] == pytest.approx(1.0) and nearest[1] == pytest.approx(level1)
    # Levels beyond the chain clamp to the last one; negative LOD magnifies level 0.
    far = sample(chain, u, v, lam=np.array([99.0, -5.0, np.nan]))[:, 0]
    assert far[0] == pytest.approx(chain.levels[-1][0, 0, 0] / 255.0) and far[1] == pytest.approx(
        1.0
    )
    assert far[2] == pytest.approx(1.0)  # NaN LOD (degenerate derivatives) -> level 0
