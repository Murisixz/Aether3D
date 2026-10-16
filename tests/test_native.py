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
import resource
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
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(300):
        native_trace(accel, o, d)
        native.png_filter(np.zeros((64, 192), np.uint8), 64, 192, 3, np.empty((64, 193), np.uint8))
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
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
