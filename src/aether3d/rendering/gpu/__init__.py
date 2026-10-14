"""Aether3D's GPU rendering backend (optional): real GPU rasterization through wgpu.

Use it with ``scene.render(backend="gpu")`` (or ``"auto"``); see ``docs/GPU.md``.
This package can always be imported: the optional ``wgpu`` dependency is
loaded only when a GPU function is called, so the CPU renderer and the rest
of Aether3D work without it.

* :func:`list_adapters` -- the graphics adapters wgpu can use, as dicts.
* :func:`gpu_status` -- availability, versions, adapters, the adapter a render
  would use and its capabilities (``check=True`` also renders a test image
  and verifies its pixels).
* :func:`release_gpu` -- free cached devices and GPU memory.

Modules: :mod:`.device` (adapters, devices, errors), :mod:`.shaders` (WGSL,
pipelines), :mod:`.resources` (buffers, caches, data layouts),
:mod:`.renderer` (the passes), :mod:`.check` (the self-test).
"""

from __future__ import annotations

import os
import platform
from typing import Any

from ...errors import RenderError


def list_adapters(*, check: bool = False) -> list[dict[str, Any]]:
    """Every adapter wgpu reports, best first for ``adapter="auto"``.

    Each dict has ``index`` (for ``adapter=``), ``name``, ``vendor``, ``type``
    (discrete, integrated, virtual, cpu, unknown), ``backend`` (vulkan, metal,
    d3d12, opengl), ``driver``, ``software``, ``features``, ``limits`` and
    ``recommended`` (the one ``"auto"`` tries first). With ``check=True``,
    a device is created on each adapter: ``usable``, ``msaa`` and
    ``error`` report the result. Raises ``RenderError("GPU_UNAVAILABLE")``
    if wgpu is not installed.
    """
    from . import device as dev

    adapters = dev.enumerate_adapters()
    best = dev.recommended(adapters)
    order = sorted(adapters, key=lambda a: dev._rank(a[0]))
    result = []
    for info, _ in order:
        item = info.to_dict()
        item["recommended"] = best is not None and info.key == best.key
        if check:
            item.update(_check_adapter(info.index))
        result.append(item)
    return result


def _check_adapter(index: int) -> dict[str, Any]:
    from . import device as dev

    try:
        context, _ = dev.get_context(index)
    except RenderError as exc:
        return {"usable": False, "error": {"code": exc.code, "message": exc.message}}
    return {
        "usable": True,
        "msaa": list(context.msaa_counts()),
        "timestamps": context.has_timestamps,
        "device_init_seconds": context.init_seconds,
    }


def gpu_status(*, adapter: Any = None, check: bool = False) -> dict[str, Any]:
    """Can this machine render on the GPU, and with what? (never raises for unavailability)

    Returns ``available`` (a device could be created for ``adapter``),
    ``reason``/``hint`` when not, ``versions``, ``environment`` (the
    ``AETHER3D_GPU_*`` variables), ``adapters`` and ``selected`` (the
    adapter and capabilities a render would use). ``check=True`` also
    renders a small test scene on the selected adapter, compares it with the
    CPU renderer and reports the result as ``check``.
    """
    from . import device as dev

    report: dict[str, Any] = {
        "available": False,
        "reason": None,
        "hint": None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": None,
        "environment": {
            name: os.environ.get(name)
            for name in (dev.ADAPTER_ENV, dev.DEBUG_ENV, dev.BACKENDS_ENV, "AETHER3D_GPU_CACHE_MB")
            if os.environ.get(name) is not None
        },
        "notes": [],
        "adapters": [],
        "selected": None,
        "attempts": [],
    }
    try:
        dev.load_wgpu()
        report["versions"] = dev.wgpu_versions()
        report["adapters"] = list_adapters()
    except RenderError as exc:
        report["reason"], report["hint"] = exc.message, exc.hint
        return report
    report["notes"] = dev.instance_notes()
    try:
        context, failures = dev.get_context(adapter)
    except RenderError as exc:
        report["reason"], report["hint"] = exc.message, exc.hint
        report["attempts"] = exc.details.get("attempts", [])
        return report
    report["attempts"] = failures
    report["available"] = True
    report["selected"] = context.report()
    if check:
        from .check import self_test

        report["check"] = self_test(adapter)
        report["available"] = bool(report["check"]["ok"])
        if not report["available"]:
            report["reason"] = report["check"].get("error", {}).get("message")
    return report


def release_gpu(adapter: Any = None) -> int:
    """Destroy cached GPU devices (all, or one adapter selection's) and free their memory.

    Returns how many were released; the next GPU render creates a device again.
    Does nothing (returns 0) when wgpu is not installed.
    """
    from . import device as dev

    if dev._wgpu is None:
        return 0
    return dev.release(adapter)


__all__ = ["gpu_status", "list_adapters", "release_gpu"]
