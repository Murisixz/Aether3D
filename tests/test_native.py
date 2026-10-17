"""The native kernels (``aether3d-native``) against Aether3D's NumPy implementations.

Every kernel must give bitwise the NumPy results (and, for the traversal, the same
statistics), whatever the number of threads; reject bad arrays with ``ValueError``
(never crash or read out of bounds); release the GIL; and keep no references or memory.
These tests need the package (``pip install ./native``); without it they are skipped --
CI jobs that build it set ``AETHER3D_REQUIRE_NATIVE=1``, which turns a missing module into
a failure (``conftest.py``).
"""

from __future__ import annotations

import gc
import sys
import threading
import time

import numpy as np
import pytest
from parity import assert_bitwise

from aether3d import Mesh, Scene, _native, create_cube, create_sphere
from aether3d.rendering.raytrace.accel import BATCH, SceneAccel
from aether3d.rendering.scene_data import extract_render_scene

native = pytest.importorskip("aether3d_native")

# Rays with an infinite direction (they hit nothing) divide infinities in NumPy's setup.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")


@pytest.fixture(autouse=True)
def native_on(monkeypatch):
    """These tests compare the native kernels with NumPy: on, whatever the environment says."""
    monkeypatch.setenv("AETHER3D_NATIVE", "1")
    monkeypatch.delenv("AETHER3D_NATIVE_THREADS", raising=False)


def without_native(monkeypatch, fn, *args, **kwargs):
    with monkeypatch.context() as m:
        m.setenv("AETHER3D_NATIVE", "0")
        assert _native.module() is None
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------------------
# Ray traversal


def tricky_scene(seed: int) -> Scene:
    """Instances in a hierarchy: rotated, non-uniform, mirrored, nested, shared meshes,
    a singular one, grouped triangles, thin and large objects."""
    rng = np.random.default_rng(seed)
    s = Scene(f"Rays{seed}")
    meshes = [create_cube(1.0), create_sphere(0.6, 12, 6), create_cube(0.2)]
    thin = create_cube(1.0)
    meshes.append(Mesh(thin.positions * np.array([5.0, 0.01, 1.0], np.float32), thin.triangles))
    parents = [None]
    for k in range(60):
        obj = s.add_object(
            f"o{k}",
            mesh=meshes[k % len(meshes)],
            parent=parents[int(rng.integers(len(parents)))],
            position=tuple(float(v) for v in rng.uniform(-6, 6, 3)),
            scale=tuple(float(v) for v in rng.choice([-1.5, -0.5, 0.7, 1.0, 2.0], 3)),
        )
        obj.rotation_euler = tuple(float(v) for v in rng.uniform(-180, 180, 3))
        if k % 7 == 0:
            parents.append(obj)
    s.get_by_name("o13").scale = (1.0, 0.0, 1.0)  # singular: never hit
    s.add_camera("Cam", position=(0, -20, 5), look_at=(0, 0, 0))
    return s


def rays(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    o = rng.uniform(-12, 12, (n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[: n // 10, 1:] = 0.0  # along an axis: zero direction components
    d[n // 10 : n // 5, 0] = -0.0
    o[: n // 50] = 0.0  # from inside objects
    o[-3] = np.nan  # non-finite rays hit nothing
    d[-2] = np.inf
    d[-1] = 0.0
    return o, d


def hits_and_stats(accel_scene, o, d, tmin, tmax, skip, any_hit):
    accel = SceneAccel.build(accel_scene)
    if any_hit:
        result = accel.occluded(o, d, tmin, tmax, *skip)
    else:
        result = accel.closest_hit(o, d, tmin, tmax, *skip)
    stats = accel.traced.to_dict()
    stats.pop("seconds")
    return result, stats


def same_hits(a, b, any_hit):
    if any_hit:
        assert_bitwise(a, b, "occluded")
    else:
        for field in ("t", "inst", "prim", "b"):
            assert_bitwise(getattr(a, field), getattr(b, field), field)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("any_hit", [False, True])
def test_traversal_hits_and_statistics_are_numpys(seed, any_hit, monkeypatch):
    s = tricky_scene(seed)
    rs = extract_render_scene(s, s.active_camera)
    o, d = rays(seed, 3000)
    tmin = np.where(np.arange(3000) % 3 == 0, 0.5, 1e-6)
    tmax = np.where(np.arange(3000) % 5 == 0, 4.0, np.inf)
    for skip_case in range(2):
        if skip_case:
            first, _ = hits_and_stats(rs, o, d, tmin, tmax, (None, None), False)
            skip = (np.where(first.inst >= 0, first.inst, -1), first.prim.copy())
        else:
            skip = (None, None)
        a, stats_a = without_native(
            monkeypatch, hits_and_stats, rs, o, d, tmin, tmax, skip, any_hit
        )
        b, stats_b = hits_and_stats(rs, o, d, tmin, tmax, skip, any_hit)
        same_hits(a, b, any_hit)
        assert stats_a == stats_b


def accel_arrays(accel: SceneAccel) -> list:
    return [
        accel.child, accel.child_lo, accel.child_hi, accel.kind, accel.first, accel.count,
        accel.v0, accel.v1, accel.v2, accel.tri_prim, accel.slot_inst, accel.inst_lo,
        accel.inst_hi, accel.inst_root, accel.to_object, accel.root_lo, accel.root_hi,
        int(accel.stack_size),
    ]  # fmt: skip


def native_trace(accel, o, d, threads=1, any_hit=False, batch=BATCH, tmin=None, tmax=None):
    n = len(o)
    out = (np.full(n, np.inf), np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.zeros((n, 3)))
    stats = native.trace(
        *accel_arrays(accel),
        np.ascontiguousarray(o),
        np.ascontiguousarray(d),
        np.full(n, 1e-6) if tmin is None else tmin,
        np.full(n, np.inf) if tmax is None else tmax,
        None,
        None,
        *out,
        any_hit,
        batch,
        threads,
    )
    return out, stats


def test_threads_change_nothing():
    s = tricky_scene(4)
    accel = SceneAccel.build(extract_render_scene(s, s.active_camera))
    o, d = rays(4, 5000)
    one, stats_one = native_trace(accel, o, d, threads=1, batch=700)
    for threads in (2, 3, 8):
        many, stats_many = native_trace(accel, o, d, threads=threads, batch=700)
        for a, b in zip(one, many, strict=True):
            assert_bitwise(a, b, f"{threads} threads")
        assert stats_many == stats_one


def test_an_empty_scene_and_no_rays():
    s = Scene("Empty")
    s.add_camera("Cam", position=(0, -5, 0), look_at=(0, 0, 0))
    accel = SceneAccel.build(extract_render_scene(s, s.active_camera))
    out, stats = native_trace(accel, np.zeros((4, 3)), np.tile([0.0, 1.0, 0.0], (4, 1)))
    assert (out[1] == -1).all() and stats == (0, 0, 0)
    out, stats = native_trace(accel, np.zeros((0, 3)), np.zeros((0, 3)))
    assert stats == (0, 0, 0)


# ---------------------------------------------------------------------------------------
# Transform hierarchies


def test_local_and_world_matrices_are_numpys(monkeypatch):
    from aether3d.rendering.scene_data import _world_stack

    rng = np.random.default_rng(5)
    s = Scene("Hierarchy")
    objects = []
    for k in range(400):
        parent = objects[int(rng.integers(len(objects)))] if objects and k % 3 else None
        q = rng.normal(size=4)
        obj = s.add_cube(name=f"c{k}", parent=parent)
        obj.position = tuple(float(v) for v in rng.choice([0.0, -0.0, 1e-300, 3.5, -1e12], 3))
        obj.rotation = tuple(float(v) for v in q / np.linalg.norm(q))
        obj.scale = tuple(float(v) for v in rng.choice([1.0, -2.0, 1e-8, 7.25], 3))
        objects.append(obj)
    order_a, world_a = without_native(monkeypatch, _world_stack, s)
    order_b, world_b = _world_stack(s)
    assert [o.id for o in order_a] == [o.id for o in order_b]
    assert_bitwise(world_b, world_a, "world matrices")


def test_world_matrices_reject_parents_after_children():
    local = np.tile(np.eye(4), (3, 1, 1))
    with pytest.raises(ValueError, match="earlier row"):
        native.world_matrices(local, np.array([-1, 2, 0], dtype=np.int64), np.empty_like(local))


# ---------------------------------------------------------------------------------------
# PNG filters


def images():
    rng = np.random.default_rng(6)
    out = []
    for channels in (1, 2, 3, 4):
        out.append(rng.integers(0, 256, (37, 53, channels), dtype=np.uint8))
        smooth = (np.cumsum(rng.integers(-3, 4, (300, 70, channels)), axis=1) % 256).astype(
            np.uint8
        )
        out.append(smooth)
        out.append(np.full((5, 9, channels), 255, np.uint8))
    return out


def test_png_encoding_is_numpys(monkeypatch):
    from aether3d.rendering.png import _filter_rows, encode_png

    for image in images():
        rows = image.reshape(image.shape[0], -1)
        bpp = image.shape[2]
        a = without_native(monkeypatch, _filter_rows, rows, bpp)
        assert_bitwise(_filter_rows(rows, bpp), a, "filtered rows")
        if image.shape[2] in (3, 4):
            assert encode_png(image) == without_native(monkeypatch, encode_png, image)


def test_png_decoding_is_numpys(monkeypatch, tmp_path):
    from aether3d.materials.images import _unfilter, decode_png
    from aether3d.rendering.png import write_png

    rng = np.random.default_rng(7)
    for bpp in (1, 2, 3, 4, 6, 8):
        for kinds in (np.arange(40) % 3, np.arange(40) % 5, rng.integers(0, 5, 40)):
            block = rng.integers(0, 256, (40, 1 + 11 * bpp), dtype=np.uint8)
            block[:, 0] = kinds
            a = without_native(monkeypatch, _unfilter, block, bpp, None)
            assert_bitwise(_unfilter(block, bpp, None), a, f"unfiltered bpp={bpp}")
    for k, image in enumerate(images()):
        if image.shape[2] in (3, 4):
            path = write_png(tmp_path / f"i{k}.png", image)
            data = path.read_bytes()
            fresh = without_native(monkeypatch, decode_png, data)
            assert_bitwise(decode_png(data).pixels, fresh.pixels, "decoded pixels")


def test_invalid_png_filter_types_are_errors():
    block = np.zeros((2, 7), dtype=np.uint8)
    block[1, 0] = 5
    with pytest.raises(ValueError, match="filter type 5"):
        native.png_unfilter(block, 2, 6, 3, np.empty((2, 6), np.uint8))


# ---------------------------------------------------------------------------------------
# Bad arrays are errors, never crashes


def test_bad_arrays_are_value_errors():
    s = tricky_scene(8)
    accel = SceneAccel.build(extract_render_scene(s, s.active_camera))
    args = accel_arrays(accel)
    o, d = rays(8, 10)
    n = 10
    outs = [
        np.full(n, np.inf),
        np.full(n, -1, np.int64),
        np.full(n, -1, np.int64),
        np.zeros((n, 3)),
    ]

    def call(a=args, oo=o, dd=d, out=outs):
        return native.trace(
            *a, oo, dd, np.zeros(n), np.full(n, np.inf), None, None, *out, False, 64, 1
        )

    call()
    with pytest.raises(ValueError, match="expected type"):
        call(oo=o.astype(np.float32))
    with pytest.raises(ValueError, match="contiguous"):
        call(dd=np.asfortranarray(np.repeat(d, 2, axis=1))[:, ::2])
    with pytest.raises(ValueError, match="elements"):
        call(dd=d[:5])
    broken = list(args)
    child = accel.child.copy()
    inner = int(np.flatnonzero(accel.kind == 0)[0])
    child[inner, 0] = 10**9
    broken[0] = child
    with pytest.raises(ValueError, match="out of range"):
        call(a=broken)
    broken = list(args)
    broken[10] = np.full_like(accel.slot_inst, 10**6)
    with pytest.raises(ValueError, match="out of range"):
        call(a=broken)
    big = np.zeros(4 * n)
    with pytest.raises(ValueError, match="share memory"):  # t inside b
        call(out=[big[:n], outs[1], outs[2], big[: 3 * n]])
    readonly = np.zeros(n)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        call(out=[readonly, outs[1], outs[2], outs[3]])
    with pytest.raises(ValueError, match="positive"):
        native.png_filter(np.zeros((2, 6), np.uint8), 2, 6, 0, np.empty((2, 7), np.uint8))


# ---------------------------------------------------------------------------------------
# The GIL, references and memory


def test_the_gil_is_released_while_tracing():
    s = tricky_scene(9)
    accel = SceneAccel.build(extract_render_scene(s, s.active_camera))
    o, d = rays(9, 400_000)
    ticks = [0]
    done = threading.Event()

    def count():
        while not done.is_set():
            ticks[0] += 1

    t = threading.Thread(target=count)
    t.start()
    start = time.perf_counter()
    try:
        before = ticks[0]
        native_trace(accel, o, d)
        during = ticks[0] - before
    finally:
        done.set()
        t.join()
    elapsed = time.perf_counter() - start
    assert elapsed > 0.02  # (the call took long enough to tell)
    assert during > 1000  # Python ran meanwhile: the call did not hold the GIL


def peak_memory_kib() -> int:
    """The process's peak memory (resident set; on Windows, working set), in KiB."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        assert ok
        return counters.PeakWorkingSetSize // 1024
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak // 1024 if sys.platform == "darwin" else peak  # (bytes on macOS)


def test_no_references_or_memory_are_kept():
    s = tricky_scene(10)
    accel = SceneAccel.build(extract_render_scene(s, s.active_camera))
    o, d = rays(10, 2000)
    arrays = [a for a in accel_arrays(accel) if isinstance(a, np.ndarray)] + [o, d]
    counts = [sys.getrefcount(a) for a in arrays]
    native_trace(accel, o, d)
    assert [sys.getrefcount(a) for a in arrays] == counts
    gc.collect()
    native_trace(accel, o, d)
    before = peak_memory_kib()
    for _ in range(300):
        native_trace(accel, o, d)
        native.png_filter(np.zeros((64, 192), np.uint8), 64, 192, 3, np.empty((64, 193), np.uint8))
    after = peak_memory_kib()
    assert after - before < 20_000  # KiB: no growth per call


# ---------------------------------------------------------------------------------------
# The switch


def test_the_switch_and_status(monkeypatch):
    assert _native.status()["available"] and _native.module() is native
    monkeypatch.setenv("AETHER3D_NATIVE", "0")
    assert _native.module() is None
    assert _native.status() == {
        "available": True,
        "enabled": False,
        "version": native.__version__,
        "reason": "loaded",
    }


def test_a_module_of_another_abi_is_not_used(monkeypatch):
    class Other:
        ABI = _native.ABI + 1
        __version__ = "9.9.9"

    monkeypatch.setitem(sys.modules, "aether3d_native", Other)
    monkeypatch.setattr(_native, "_loaded", [])
    assert _native.module() is None
    assert "ABI" in _native.status()["reason"]


def test_the_thread_setting(monkeypatch):
    import os

    assert _native.threads() == 1
    monkeypatch.setenv("AETHER3D_NATIVE_THREADS", "3")
    assert _native.threads() == 3
    monkeypatch.setenv("AETHER3D_NATIVE_THREADS", "0")
    assert _native.threads() == max(1, os.cpu_count() or 1)
    for bad in ("-1", "many"):
        monkeypatch.setenv("AETHER3D_NATIVE_THREADS", bad)
        with pytest.raises(ValueError, match="AETHER3D_NATIVE_THREADS"):
            _native.threads()


def test_traversal_with_threads_through_the_accel(monkeypatch):
    s = tricky_scene(11)
    rs = extract_render_scene(s, s.active_camera)
    o, d = rays(11, 4000)
    one, stats_one = hits_and_stats(rs, o, d, 1e-6, np.inf, (None, None), False)
    monkeypatch.setenv("AETHER3D_NATIVE_THREADS", "4")
    many, stats_many = hits_and_stats(rs, o, d, 1e-6, np.inf, (None, None), False)
    same_hits(one, many, False)
    assert stats_one == stats_many


# ---------------------------------------------------------------------------------------
# Sparse LDL^T and atlas placement (Stage 14)


def _spd(n: int, seed: int):
    from aether3d.uv import sparse

    rng = np.random.default_rng(seed)
    pts = rng.random((n, 3))
    rows, cols, vals = [], [], []
    for i in range(n):
        d = ((pts - pts[i]) ** 2).sum(axis=1)
        for j in np.argsort(d)[1:7]:
            w = -float(rng.random()) * (10.0 ** rng.integers(-3, 3))
            rows += [i, j]
            cols += [j, i]
            vals += [w, w]
    off = sparse.assemble(n, rows, cols, vals)
    col = np.repeat(np.arange(n), np.diff(off.indptr))
    diag = -np.bincount(col, off.data, n) + 1e-3
    m = sparse.assemble(n, np.r_[rows, np.arange(n)], np.r_[cols, np.arange(n)], np.r_[vals, diag])
    return m, pts


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ldl_is_numpys(seed, monkeypatch):
    from aether3d.uv import sparse

    m, pts = _spd(700, seed)
    perm = sparse.nested_dissection(pts, m.indptr, m.indices)
    b = np.random.default_rng(seed + 10).normal(size=(700, 3))

    def run():
        s = sparse.analyze(m, perm)
        f = sparse.factor(s, m)
        return s, f, f.solve(b), f.solve(b[:, 0])

    s1, f1, x1, y1 = run()
    s0, f0, x0, y0 = without_native(monkeypatch, run)
    for name in ("parent", "lp", "li", "rp", "rj", "rpos"):
        assert np.array_equal(getattr(s1, name), getattr(s0, name)), name
    assert_bitwise(f1.lx, f0.lx)
    assert_bitwise(f1.d, f0.d)
    assert_bitwise(x1, x0)
    assert_bitwise(y1, y0)


def test_ldl_reports_the_same_failing_pivot(monkeypatch):
    from aether3d.errors import TexturingError
    from aether3d.uv import sparse

    m, pts = _spd(200, 3)
    bad = sparse.CSC(m.n, m.indptr, m.indices, m.data.copy())
    diag = np.flatnonzero(m.indices == np.repeat(np.arange(m.n), np.diff(m.indptr)))
    bad.data[diag[150]] = -50.0
    perm = np.arange(200)
    codes = []
    for mode in ("native", "numpy"):
        call = (lambda: sparse.factor(sparse.analyze(bad, perm), bad)) if mode == "native" else None
        with pytest.raises(TexturingError) as exc:
            if call is not None:
                call()
            else:
                without_native(monkeypatch, lambda: sparse.factor(sparse.analyze(bad, perm), bad))
        codes.append(exc.value.details["pivot"])
    assert codes[0] == codes[1]


def test_ldl_rejects_inconsistent_structures():
    from aether3d.uv import sparse

    m, pts = _spd(50, 4)
    s = sparse.analyze(m, np.arange(50))
    ax = m.data[s.amap]
    lx = np.empty(s.li.size)
    d = np.empty(50)
    rpos = s.rpos.copy()
    rpos[0] = rpos[1]
    with pytest.raises(ValueError):
        native.ldl_factor(s.ap, s.ai, ax, s.lp, s.li, s.rp, s.rj, rpos, lx, d)
    li = s.li.copy()
    li[0] = 999
    with pytest.raises(ValueError):
        native.ldl_solve(s.lp, li, lx, s.rp, s.rj, s.rpos, d, np.zeros((50, 1)))
    ai = s.ai.copy()
    ai[-1] = 60
    with pytest.raises(ValueError):
        native.ldl_etree(s.ap, ai, np.empty(50, np.int64), np.empty(50, np.int64))


def test_atlas_placement_is_numpys(monkeypatch):
    from aether3d.uv import pack

    rng = np.random.default_rng(5)
    masks = [rng.random((rng.integers(2, 40), rng.integers(2, 50))) < 0.6 for _ in range(120)]
    masks = [m if m.any() else np.ones((2, 2), bool) for m in masks]
    bottoms, tops = [], []
    for m in masks:
        for o in range(4):
            b, t = pack.profiles(pack.oriented(m, o))
            bottoms.append(b)
            tops.append(t)
    order = np.lexsort((np.arange(120), -np.array([m.sum() for m in masks])))
    a = pack.place(300, bottoms, tops, order)
    b = without_native(monkeypatch, pack.place, 300, bottoms, tops, order)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]
    with pytest.raises(ValueError):
        native.pack_horizon(
            300,
            np.array([0]),
            np.array([0]),
            np.array([0, 1, 1, 1, 1]),
            np.array([0]),
            np.zeros(1, np.int64),
            np.zeros(2, np.int64),
        )


def test_unwraps_are_numpys(monkeypatch):
    from aether3d import uv

    mesh = create_sphere(1.0, 24, 12)
    a = uv.unwrap(mesh, resolution=256)
    b = without_native(monkeypatch, uv.unwrap, mesh, resolution=256)
    assert a.mesh == b.mesh


def _polygons(rng, count):
    """Random polygons with holes, and degenerate ones: collinear runs, duplicate points,
    self-intersections, spikes, rings of more than 80 points (z-order hashing)."""
    pts, rings, polys = [], [0], [0]

    def ring(p):
        pts.extend(p.tolist())
        rings.append(rings[-1] + len(p))

    for k in range(count):
        n = int(rng.integers(3, 200))
        t = np.sort(rng.random(n)) * 2 * np.pi
        r = 1.0 + 0.6 * rng.random(n)
        c = rng.normal(size=2) * 10
        outer = c + np.c_[r * np.cos(t), r * np.sin(t)]
        kind = k % 6
        if kind == 1:  # snapped to a grid: collinear runs and duplicates
            outer = np.round(outer * 4) / 4
        elif kind == 2:  # a spike and a repeated point
            outer = np.insert(outer, n // 2, [outer[n // 2], outer[n // 2] + 5], axis=0)
        elif kind == 3:  # self-intersecting: two points swapped
            outer[[1, n // 2]] = outer[[n // 2, 1]]
        ring(outer)
        holes = int(rng.integers(0, 4)) if kind in (0, 4, 5) else 0
        for h in range(holes):
            m = int(rng.integers(3, 30))
            th = np.sort(rng.random(m)) * 2 * np.pi
            hc = c + np.array([np.cos(h * 2.1), np.sin(h * 2.1)]) * 0.5
            hole = hc + 0.15 * np.c_[np.cos(th), np.sin(th)]
            ring(hole if kind != 5 else hole[::-1])
        polys.append(len(rings) - 1)
    return (
        np.array(pts, dtype=np.float64),
        np.array(rings, dtype=np.int64),
        np.array(polys, dtype=np.int64),
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_triangulation_is_numpys(seed):
    from aether3d.cad.triangulate import triangulate, triangulate_numpy

    pts, rings, polys = _polygons(np.random.default_rng(seed), 150)
    a = triangulate(pts, rings, polys)
    b = triangulate_numpy(pts, rings, polys)
    assert len(a[0]) > 1000
    assert_bitwise(a[0], b[0], "triangles")
    assert_bitwise(a[1], b[1], "owners")


def test_triangulation_of_real_faces_is_numpys(monkeypatch):
    # The faces of a part with holes, fillets and countersinks: the tessellations agree.
    pytest.importorskip("manifold3d")
    from aether3d.cad.brep.tessellate import tessellate

    scene = Scene("native")
    doc = scene.cad
    sk = doc.sketch("P")
    sk.rectangle((0, 0), (80, 40))
    p = doc.part("Bracket")
    p.extrude(sk, 10, id="base")
    p.hole("base:end", [[20, 20, 10], [60, 20, 10]], 8, kind="countersink", csink_diameter=16)
    p.fillet([{"edge": ["base:end", "base:side/r2"]}], 3)
    body = p.body()
    a = tessellate(body)
    b = without_native(monkeypatch, tessellate, body)
    assert_bitwise(a.triangles, b.triangles, "triangles")
    assert_bitwise(a.positions, b.positions, "positions")


def test_triangulation_rejects_bad_tables():
    pts = np.zeros((4, 2))
    out, owner = np.zeros((12, 3), np.int64), np.zeros(12, np.int64)
    good = (np.array([0, 4]), np.array([0, 1]))
    native.triangulate(pts, *good, out, owner)
    for rings, polys in (
        (np.array([0, 5]), np.array([0, 1])),  # past the points
        (np.array([0, 3, 2, 4]), np.array([0, 3])),  # decreasing
        (np.array([0, 4]), np.array([0, 2])),  # more rings than there are
        (np.array([0, 2, 4]), np.array([0, 0, 2])),  # a polygon without rings
    ):
        with pytest.raises(ValueError):
            native.triangulate(pts, rings, polys, out, owner)
    with pytest.raises(ValueError):
        native.triangulate(pts, *good, out[:1], owner)  # no room
    with pytest.raises(ValueError):
        native.triangulate(np.array([[0.0, np.nan]] * 4), *good, out, owner)
