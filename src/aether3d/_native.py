"""The native kernels of ``aether3d-native`` (Rust), used when installed and switched on.

``aether3d-native`` (``native/`` in the repository) implements a few hot kernels -- ray
traversal, transform hierarchies, PNG row filters -- with exactly the arithmetic of their
NumPy implementations in Aether3D, so the results are bitwise the same
(``tests/test_native.py``). Aether3D calls :func:`module` where a kernel can run natively;
``None`` means the NumPy implementation runs: the package is not installed, cannot be
loaded, speaks another calling convention (:data:`ABI`), or ``AETHER3D_NATIVE=0``.
"""

from __future__ import annotations

import os
from typing import Any

from . import _features

ABI = 2
"""The calling convention this version of Aether3D expects (``aether3d_native.ABI``)."""

_loaded: list[Any] = []  # [module or None, reason], once loading was tried


def _load() -> Any:
    if not _loaded:
        try:
            import aether3d_native as native
        except ImportError as exc:
            _loaded[:] = [None, f"not installed ({exc})"]
        else:
            abi = getattr(native, "ABI", None)
            if abi != ABI:
                version = getattr(native, "__version__", "?")
                _loaded[:] = [None, f"aether3d_native {version} has ABI {abi}; this needs {ABI}"]
            else:
                _loaded[:] = [native, "loaded"]
    return _loaded[0]


def module() -> Any:
    """The native module, or ``None`` (the NumPy implementations run)."""
    if not _features.enabled("AETHER3D_NATIVE"):
        return None
    return _load()


def status() -> dict[str, Any]:
    """Whether the native kernels are available and used, and why not (for reports)."""
    native = _load()
    return {
        "available": native is not None,
        "enabled": native is not None and _features.enabled("AETHER3D_NATIVE"),
        "version": None if native is None else getattr(native, "__version__", None),
        "reason": _loaded[1],
    }


def threads() -> int:
    """Threads the native ray traversal uses (``AETHER3D_NATIVE_THREADS``; default 1, and
    ``0`` one per CPU). Results do not depend on it: every ray is traced on its own. The
    CPU path tracer already runs several processes on Linux; elsewhere, or for spatial
    queries with many rays, more threads trace faster."""
    raw = os.environ.get("AETHER3D_NATIVE_THREADS", "").strip()
    if not raw:
        return 1
    try:
        count = int(raw)
    except ValueError:
        raise ValueError(f"AETHER3D_NATIVE_THREADS={raw!r} is not a number of threads") from None
    if count < 0:
        raise ValueError(f"AETHER3D_NATIVE_THREADS={raw!r}: use 0 (one per CPU) or more")
    return count if count else max(1, os.cpu_count() or 1)
