"""Timing for the tests that check how costs grow (scaling, not speed).

``seconds(fn)`` is the wall time of one call with Python's cyclic garbage collector
paused, as ``timeit`` measures. A full collection walks every object of the test
process: how many there are depends on the tests that ran before, not on the code
measured, and on a slow machine one collection inside a timed call can cost more than
the call itself.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any


def seconds(fn: Callable[[], Any]) -> float:
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        fn()
        return time.perf_counter() - start
    finally:
        if enabled:
            gc.enable()
