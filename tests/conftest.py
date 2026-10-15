"""Shared test fixtures.

GPU tests take the ``gpu`` fixture. Without a usable GPU adapter (no wgpu, no
driver, no device) they are skipped, and the skip reason says why. With
``AETHER3D_REQUIRE_GPU=1`` -- set in the CI job that must render on a real
graphics adapter -- they fail instead, so a broken GPU setup can never pass
silently. ``AETHER3D_TEST_GPU_ADAPTER`` selects the adapter (default: the
best one, as ``adapter="auto"`` would). ``AETHER3D_GPU_TESTS=skip`` skips them
on purpose (the portable CI matrix does; dedicated GPU jobs run them).
"""

from __future__ import annotations

import os
from typing import Any

import pytest


class GPU:
    """The adapter GPU tests render on."""

    def __init__(self, adapter: str | None, selected: dict[str, Any]) -> None:
        self.adapter = adapter
        self.info = selected
        self.name = selected["adapter"]["name"]
        self.msaa = tuple(selected["msaa"])

    def render(self, scene, **options):
        return scene.render(backend="gpu", adapter=self.adapter, **options)


@pytest.fixture(scope="session")
def gpu() -> GPU:
    from aether3d.rendering.gpu import gpu_status

    if os.environ.get("AETHER3D_GPU_TESTS") == "skip":
        pytest.skip(
            "GPU tests are switched off here (AETHER3D_GPU_TESTS=skip); CI runs them in "
            "the dedicated GPU jobs"
        )
    adapter = os.environ.get("AETHER3D_TEST_GPU_ADAPTER") or None
    status = gpu_status(adapter=adapter)
    if not status["available"]:
        message = f"GPU rendering is unavailable here: {status['reason']}"
        if os.environ.get("AETHER3D_REQUIRE_GPU") == "1":
            pytest.fail(message)
        pytest.skip(message)
    return GPU(adapter, status["selected"])


@pytest.fixture(autouse=True)
def strict_history(monkeypatch):
    """Every test runs with the history's strict checks: a change recorded outside an
    operation fails, and every undo, redo or rollback re-checks the scene's indexes."""
    from aether3d import history

    monkeypatch.setattr(history, "STRICT", True)


@pytest.fixture(autouse=True)
def strict_render_world(monkeypatch):
    """An unexpected error of the retained render world fails the test, instead of being
    covered by the full extraction it falls back to."""
    from aether3d.rendering import retained

    monkeypatch.setattr(retained, "STRICT", True)
