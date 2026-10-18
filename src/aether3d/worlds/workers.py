"""Threads for generation work: terrain tiles, block chunks and their meshes.

Every tile or chunk is a pure function of the definition and its coordinates, so they can
be computed in parallel and give the same results with any number of threads
(``streaming.threads``; 0 uses one per CPU up to :data:`MAX_THREADS`). Generation is many
NumPy operations on mid-sized arrays, which hold the GIL between them: measured on an
i9-12900H, 2 threads build block chunks about 17% faster than 1 and more threads are
slower, so the default is 1 (everything in the calling thread).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

MAX_THREADS = 16

_lock = threading.Lock()
_pools: dict[int, ThreadPoolExecutor] = {}


def thread_count(definition: dict) -> int:
    n = int(definition["streaming"].get("threads", 0))
    return n if n > 0 else max(1, min(MAX_THREADS, os.cpu_count() or 1))


def parallel_map(fn: Callable[[Any], Any], items: Sequence[Any], threads: int) -> list[Any]:
    """``[fn(x) for x in items]``, computed by up to ``threads`` threads (in order)."""
    if threads <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with _lock:
        pool = _pools.get(threads)
        if pool is None:
            pool = _pools[threads] = ThreadPoolExecutor(
                threads, thread_name_prefix="aether3d-world"
            )
    return list(pool.map(fn, items))
