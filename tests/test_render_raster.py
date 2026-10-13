"""The rasterizer core: clipping, watertight coverage, fill rule, depth and robustness."""

import numpy as np
import pytest

from aether3d.rendering import raster as R


def rasterize_ndc(triangles, width, height, depth=0.5, ids=None):
    """Rasterize NDC triangles (w = 1) into a fresh visibility buffer."""
    tris = np.asarray(triangles, dtype=float)
    pos = np.zeros((len(tris), 3, 4))
    pos[:, :, :2] = tris
    pos[:, :, 2] = depth
    pos[:, :, 3] = 1.0
    ids = np.arange(len(tris)) if ids is None else np.asarray(ids)
    cpos, cids = R.clip_triangles(pos, ids)
    X, Y, z, A2 = R.screen_setup(cpos, width, height)
    keys = np.full(width * height, R.EMPTY, np.int64)
    stats = R.RasterStats()
    R.rasterize(X, Y, z, A2, cids, width, height, keys, 20, stats)
    return keys, stats


def coverage_counts(triangles, width, height):
    counts = np.zeros(width * height, int)
    for tri in triangles:
        keys, _ = rasterize_ndc([tri], width, height)
        counts += keys != R.EMPTY
    return counts


def jittered_grid(n, rng, flip_some=True):
    g = np.linspace(-1, 1, n)
    pts = np.array([[x, y] for y in g for x in g])
    inner = (np.abs(pts[:, 0]) < 1) & (np.abs(pts[:, 1]) < 1)
    pts[inner] += rng.uniform(-0.6 / n, 0.6 / n, (inner.sum(), 2))
    tris = []
    for r in range(n - 1):
        for c in range(n - 1):
            a, b, d, e = r * n + c, r * n + c + 1, (r + 1) * n + c, (r + 1) * n + c + 1
            for t in ([a, b, e], [a, e, d]) if (r + c) % 2 else ([a, b, d], [b, e, d]):
                tris.append(pts[t[::-1]] if flip_some and rng.random() < 0.5 else pts[t])
    return tris


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("size", [(37, 23), (64, 64), (5, 3)])
def test_shared_edges_cover_every_sample_exactly_once(seed, size):
    rng = np.random.default_rng(seed)
    counts = coverage_counts(jittered_grid(6, rng), *size)
    assert counts.min() == 1 and counts.max() == 1


@pytest.mark.parametrize(
    "center",
    [(0.0, 0.0), (2 * 10.5 / 37 - 1, 1 - 2 * 5.5 / 23), (2 * 10 / 37 - 1, 1 - 2 * 5 / 23)],
)
def test_fans_around_sample_centers_and_pixel_corners_are_watertight(center):
    # A vertex exactly on a sample center, and one exactly on a pixel corner; the
    # fan reaches far outside the screen, so its triangles are clipped first.
    angles = np.linspace(0, 2 * np.pi, 17)[:-1]
    ring = np.c_[np.cos(angles), np.sin(angles)] * 4.0
    fan = [[center, ring[k], ring[(k + 1) % 16]] for k in range(16)]
    counts = coverage_counts(fan, 37, 23)
    assert counts.min() == 1 and counts.max() == 1


def test_coverage_matches_the_exact_area():
    tri = np.array([[-0.71, -0.53], [0.83, -0.2], [-0.1, 0.77]])
    keys, _ = rasterize_ndc([tri], 400, 400)
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    area = abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2 * 200 * 200
    perimeter = sum(np.linalg.norm(tri[i] - tri[i - 1]) for i in range(3)) * 200
    # Sample counts differ from the area by less than the samples along the edges.
    assert abs(np.count_nonzero(keys != R.EMPTY) - area) < perimeter / 10


def test_top_left_fill_rule():
    # A square whose edges pass exactly through sample centers (8 x 8 samples):
    # samples on its left and top edges are inside, on its right and bottom edges not.
    w = h = 8

    def ndc(x, y):  # sample coordinates -> NDC
        return (x / w * 2 - 1, 1 - y / h * 2)

    a, b, c, d = ndc(1.5, 1.5), ndc(4.5, 1.5), ndc(4.5, 4.5), ndc(1.5, 4.5)
    keys, _ = rasterize_ndc([[a, b, c], [a, c, d]], w, h)
    covered = (keys != R.EMPTY).reshape(h, w)
    expected = np.zeros((h, w), bool)
    expected[1:4, 1:4] = True  # sample centers 1.5, 2.5, 3.5 on both axes
    assert np.array_equal(covered, expected)


def test_clipping_keeps_winding_and_stays_inside_the_frustum():
    rng = np.random.default_rng(3)
    pos = rng.uniform(-3, 3, (400, 3, 4))
    pos[:, :, 3] = rng.uniform(-1, 3, (400, 3))  # vertices behind the camera too
    ids = np.arange(400)
    clipped, cids = R.clip_triangles(pos, ids)
    for _, (a, b, c, d) in R.CLIP_PLANES:
        dist = clipped[..., 0] * a + clipped[..., 1] * b + clipped[..., 2] * c + clipped[..., 3] * d
        assert np.all(dist >= -1e-9)
    assert np.all(clipped[..., 3] > 0)  # nothing at or behind the eye survives

    # Clipped pieces keep the orientation of the triangle they came from (in NDC).
    def orient(t):
        p = t[..., :2] / t[..., 3:4]
        return np.sign(
            (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
            - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
        )

    inside = R.inside_mask(pos)
    kept = np.isin(cids, ids[inside])
    assert np.array_equal(orient(clipped[kept]), orient(pos[inside][np.argsort(ids[inside])]))
    assert set(cids) <= set(ids[~R.outside_mask(pos)])


def test_depth_test_is_order_independent_and_deterministic():
    near = [[(-0.8, -0.8), (0.6, -0.8), (-0.8, 0.6)]]
    far = [[(-0.6, -0.6), (0.8, -0.6), (0.8, 0.8)]]
    w = h = 32

    def run(order):
        pos = np.zeros((2, 3, 4))
        pos[0, :, :2], pos[0, :, 2] = near[0], 0.3
        pos[1, :, :2], pos[1, :, 2] = far[0], 0.7
        pos[:, :, 3] = 1
        idx = np.array(order)
        X, Y, z, A2 = R.screen_setup(pos[idx], w, h)
        keys = np.full(w * h, R.EMPTY, np.int64)
        R.rasterize(X, Y, z, A2, idx, w, h, keys, 4, R.RasterStats())
        return keys & 0xF

    a, b = run([0, 1]), run([1, 0])
    assert np.array_equal(a, b)
    assert np.any(a == 0) and np.any(a == 1)
    # Exact ties go to the lowest id, whatever the order.
    pos = np.zeros((2, 3, 4))
    pos[:, :, :2] = near[0]
    pos[:, :, 2], pos[:, :, 3] = 0.5, 1
    for order in ([0, 1], [1, 0]):
        idx = np.array(order)
        X, Y, z, A2 = R.screen_setup(pos[idx], w, h)
        keys = np.full(w * h, R.EMPTY, np.int64)
        R.rasterize(X, Y, z, A2, idx, w, h, keys, 4, R.RasterStats())
        assert set(np.unique(keys[keys != R.EMPTY] & 0xF)) == {0}


def test_degenerate_and_offscreen_triangles():
    # Zero-area (collinear or repeated vertices) and fully off-screen triangles.
    keys, stats = rasterize_ndc(
        [
            [(0, 0), (0.5, 0.5), (1, 1)],
            [(0.2, 0.2), (0.2, 0.2), (0.4, 0.1)],
            [(3, 3), (4, 3), (3, 4)],
        ],
        16,
        16,
    )
    assert np.all(keys == R.EMPTY) and stats.degenerate == 2
    assert R.outside_mask(np.array([[[3, 3, 0.5, 1], [4, 3, 0.5, 1], [3, 4, 0.5, 1]]])).all()


def test_tiny_triangles_between_sample_centers_cover_nothing():
    keys, _ = rasterize_ndc([[(0.01, 0.01), (0.02, 0.01), (0.01, 0.02)]], 16, 16)
    assert np.all(keys == R.EMPTY)


def test_huge_coordinates_are_clipped_safely():
    # Vertices 1e12 away (but in front of the camera): clipping brings them into range.
    pos = np.array([[[-1e12, -1e12, 0.5, 1], [1e12, -1e12, 0.5, 1], [0, 1e12, 0.5, 1]]])
    cpos, cids = R.clip_triangles(pos, np.array([0]))
    X, Y, z, A2 = R.screen_setup(cpos, 64, 48)
    keys = np.full(64 * 48, R.EMPTY, np.int64)
    R.rasterize(X, Y, z, A2, cids, 64, 48, keys, 4, R.RasterStats())
    assert np.all(keys != R.EMPTY)  # the whole screen is inside that triangle
