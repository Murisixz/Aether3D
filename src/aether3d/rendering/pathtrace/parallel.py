"""Running the path tracer on several CPU cores.

The image is split into tiles (contiguous runs of pixels) and the samples
into rounds (all of them, or one pass per round when a progress callback
wants to hear about every pass). Every task traces one tile for one round
and adds its samples, in sample order, to that tile's running sums. Tiles
never share pixels, so the order in which tasks run does not matter: the
result is bitwise the same for any number of workers, including one.

Workers are processes forked from the rendering process (Linux only): they
inherit the scene, its BVH and the shading data without copying or
pickling, and add into accumulation buffers in shared memory (an anonymous
``mmap`` made before forking). On other platforms -- and for renders too
small to be worth starting processes for -- the same tasks run in the
calling process. Python threads would not help: most of the work holds the
interpreter lock.
"""

from __future__ import annotations

import mmap
import multiprocessing
import os
import sys
import time
import warnings
from typing import Any

import numpy as np

CHANNELS = {
    "beauty": 3,
    "squares": 3,
    "direct": 3,
    "indirect": 3,
    "albedo": 3,
    "normal": 3,
    "ao": 1,
    "hits": 1,
}
"""Running sums kept per pixel (``squares``: of the beauty, for the variance)."""
AUTO_MIN_RAYS = 3_000_000
"""Renders estimated to trace fewer rays run in one process when ``workers = 0`` (auto)."""
MAX_AUTO_WORKERS = 8
"""Most processes ``workers = 0`` starts (memory bandwidth limits scaling beyond)."""
TILES_PER_WORKER = 4
"""Tiles per worker process (smaller tiles balance uneven work better)."""


def fork_available() -> bool:
    return sys.platform.startswith("linux") and "fork" in multiprocessing.get_all_start_methods()


def cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)  # pragma: no cover - non-Linux


def plan_workers(requested: int, rays: int) -> tuple[int, str]:
    """Worker processes for a render, and why (``requested``: 0 = automatic)."""
    if requested == 1:
        return 1, "one process (workers=1)"
    if not fork_available():
        return 1, "one process: parallel path tracing uses fork() and runs on Linux only"
    if requested == 0:
        if rays < AUTO_MIN_RAYS:
            return 1, "one process: the render is small (workers=0 picks processes for big ones)"
        n = min(cpu_count(), MAX_AUTO_WORKERS)
        return n, f"{n} processes (workers=0: one per CPU, at most {MAX_AUTO_WORKERS})"
    return requested, f"{requested} processes (workers={requested})"


class Accumulators:
    """Per-pixel running sums, optionally in memory shared with forked workers."""

    def __init__(self, n_pix: int, shared: bool) -> None:
        width = sum(CHANNELS.values())
        nbytes = max(n_pix * width * 8, 8)
        self._map = mmap.mmap(-1, nbytes) if shared else None
        if self._map is not None:
            data = np.frombuffer(self._map, dtype=np.float64, count=n_pix * width)
        else:
            data = np.zeros(n_pix * width)
        data = data.reshape(n_pix, width)
        self.arrays: dict[str, np.ndarray] = {}
        column = 0
        for name, count in CHANNELS.items():
            view = data[:, column : column + count]
            self.arrays[name] = view if count > 1 else view[:, 0]
            column += count

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]

    def copy(self) -> dict[str, np.ndarray]:
        """Private copies of the sums (the shared buffer is released afterwards)."""
        return {name: np.array(a) for name, a in self.arrays.items()}

    def close(self) -> None:
        self.arrays = {}
        if self._map is not None:
            self._map.close()
            self._map = None


_STATE: tuple | None = None
"""``(tracer, accumulators, batch)`` of the render in progress: forked workers inherit it."""


def run_task(task: tuple[int, int, int, int]) -> dict[str, Any]:
    """Trace pixels ``[start, stop)`` for samples ``[s0, s1)``; add them to the sums."""
    tracer, acc, batch = _STATE
    start, stop, s0, s1 = task
    before_counts = dict(tracer.counts)
    before_times = dict(tracer.timings)
    before_ps = (tracer.ps.pass_through_rays, tracer.ps.layer_limited)
    n_tile = stop - start
    total = n_tile * (s1 - s0)
    visible = np.zeros(len(tracer.ps.rs.instances), dtype=np.int64)
    # Equal batches of at most `batch` paths (no small remainder batch: NumPy work per
    # path grows quickly in small batches).
    count = max(1, -(-total // batch))
    size = -(-total // count)
    t0 = time.perf_counter()
    for b in range(0, total, size):
        idx = np.arange(b, min(total, b + size), dtype=np.int64)
        offset, local = np.divmod(idx, n_tile)
        sample = s0 + offset
        pixel = start + local
        out = tracer.trace(pixel, sample)
        for s in np.unique(sample):  # per pixel, add samples in sample order
            sel = np.flatnonzero(sample == s)
            px = pixel[sel]
            beauty = out["direct"][sel] + out["indirect"][sel]
            acc["beauty"][px] += beauty
            acc["squares"][px] += beauty * beauty
            for name in ("direct", "indirect", "albedo", "normal"):
                acc[name][px] += out[name][sel]
            acc["ao"][px] += out["ao"][sel]
            hit = out["hit"][sel]
            acc["hits"][px] += hit
            visible += np.bincount(out["inst"][sel][hit], minlength=visible.size)
    return {
        "counts": {k: v - before_counts.get(k, 0) for k, v in tracer.counts.items()},
        "timings": {k: v - before_times.get(k, 0.0) for k, v in tracer.timings.items()},
        "pass_through_rays": tracer.ps.pass_through_rays - before_ps[0],
        "layer_limited": tracer.ps.layer_limited - before_ps[1],
        "visible": visible,
        "paths": total,
        "seconds": time.perf_counter() - t0,
        "pid": os.getpid(),
    }


def tile_count(n_pix: int, samples: int, workers: int, batch: int) -> int:
    """Tiles for a round of ``samples`` passes: one batch of paths each (at least
    :data:`TILES_PER_WORKER` per worker, so uneven tiles balance out)."""
    if workers <= 1:
        return 1
    return max(workers * TILES_PER_WORKER, -(-n_pix * samples // batch))


def tiles(n_pix: int, count: int) -> list[tuple[int, int]]:
    """``count`` contiguous pixel ranges of about the same size (fewer for tiny images)."""
    count = max(1, min(count, n_pix))
    edges = np.linspace(0, n_pix, count + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


class Runner:
    """Runs tasks in this process or in a pool of forked workers (a context manager)."""

    def __init__(self, tracer, acc: Accumulators, workers: int, batch: int) -> None:
        self.workers = workers
        self.pool = None
        global _STATE
        _STATE = (tracer, acc, batch)
        if workers > 1:
            context = multiprocessing.get_context("fork")
            with warnings.catch_warnings():
                # Python warns when a process with threads forks; the workers only run
                # NumPy code on inherited data and leave with os._exit().
                warnings.simplefilter("ignore", DeprecationWarning)
                self.pool = context.Pool(workers)

    def run(self, tasks: list[tuple[int, int, int, int]]) -> list[dict[str, Any]]:
        if self.pool is None:
            return [run_task(task) for task in tasks]
        return self.pool.map(run_task, tasks, chunksize=1)

    def __enter__(self) -> Runner:
        return self

    def __exit__(self, *exc) -> None:
        global _STATE
        if self.pool is not None:
            if exc[0] is None:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
            self.pool = None
        _STATE = None
