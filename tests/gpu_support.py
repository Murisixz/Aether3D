"""Which GPU backends a test module may parametrize over: the module-level twin of the
``gpu`` fixture in conftest.py, for tests that run one check on every backend.

* ``AETHER3D_GPU_TESTS=skip`` (CI's CPU jobs) turns the GPU off for these tests too;
* ``AETHER3D_REQUIRE_GPU=1`` makes an unusable GPU an error instead of a skip;
* ``AETHER3D_TEST_GPU_ADAPTER`` names the adapter that is probed (renders select theirs as
  the engine does: ``AETHER3D_GPU_ADAPTER``, else the best one);
* ``GPU_PATH`` is false where the adapter cannot run the GPU path tracer (Microsoft WARP,
  Direct3D 12 without DXC: ``device.pathtrace_support``), as in test_gpu_pathtrace.py.
"""

from __future__ import annotations

import os

from aether3d.rendering.gpu import gpu_status

ADAPTER = os.environ.get("AETHER3D_TEST_GPU_ADAPTER") or None


def _probe() -> tuple[bool, bool, str]:
    if os.environ.get("AETHER3D_GPU_TESTS") == "skip":
        return False, False, "GPU tests are switched off here (AETHER3D_GPU_TESTS=skip)"
    status = gpu_status(adapter=ADAPTER)
    if not status.get("available"):
        reason = f"GPU rendering is unavailable here: {status.get('reason')}"
        if os.environ.get("AETHER3D_REQUIRE_GPU") == "1":
            raise RuntimeError(f"AETHER3D_REQUIRE_GPU=1, but {reason}")
        return False, False, reason
    from aether3d.rendering.gpu import device

    raster, _ = device.get_context(ADAPTER)
    why = device.pathtrace_support(raster.info)
    return True, why is None, why or ""


GPU, GPU_PATH, REASON = _probe()
"""``GPU``: the GPU rasterizer can be tested; ``GPU_PATH``: the GPU path tracer too."""
