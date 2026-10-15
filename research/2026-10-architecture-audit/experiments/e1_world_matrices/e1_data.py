import numpy as np
def make(n, seed=1, max_depth=8):
    rng = np.random.default_rng(seed)
    parent = np.full(n, -1, dtype=np.int64); depth = np.zeros(n, dtype=np.int64)
    for k in range(1, n):
        if rng.random() < 0.9:
            p = int(rng.integers(0, k))
            if depth[p] < max_depth:
                parent[k] = p; depth[k] = depth[p] + 1
    pos = rng.uniform(-10, 10, (n, 3))
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    scale = rng.uniform(0.5, 2.0, (n, 3))
    return parent, np.ascontiguousarray(pos), np.ascontiguousarray(q), np.ascontiguousarray(scale), depth

def numpy_level(parent, pos, quat, scale, depth):
    """Aether3D's algorithm (rendering/scene_data.py) on flat arrays."""
    n = len(parent)
    x, y, z, w = quat.T
    r = np.empty((n, 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z); r[:, 0, 1] = 2 * (x * y - z * w); r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w); r[:, 1, 1] = 1 - 2 * (x * x + z * z); r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w); r[:, 2, 1] = 2 * (y * z + x * w); r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    m = np.zeros((n, 4, 4)); m[:, 3, 3] = 1.0
    m[:, :3, :3] = r * scale[:, None, :]; m[:, :3, 3] = pos
    for level in range(1, int(depth.max()) + 1):
        rows = np.flatnonzero(depth == level)
        a, b = m[parent[rows]], m[rows]
        out = np.empty_like(b)
        for i in range(4):
            for j in range(4):
                out[:, i, j] = a[:, i, 0] * b[:, 0, j] + a[:, i, 1] * b[:, 1, j] + a[:, i, 2] * b[:, 2, j] + a[:, i, 3] * b[:, 3, j]
        m[rows] = out
    return m

