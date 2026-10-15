"""GPU adapters and devices: discovery, selection, creation and the per-device context.

Adapters are enumerated through wgpu (``wgpu.gpu.enumerate_adapters_sync()``):
every graphics API implementation wgpu can reach -- Vulkan, Metal, Direct3D
12, OpenGL -- reports its adapters with their real names, types, drivers,
features and limits. Nothing is guessed from device names or from tools such
as ``nvidia-smi``; an adapter counts as usable only once a device has
actually been created on it.

Selection (``adapter=`` / ``--adapter`` / ``AETHER3D_GPU_ADAPTER``):

* ``None`` or ``"auto"`` -- the best usable adapter: discrete GPUs first, then
  integrated, virtual, unknown and finally software (CPU) adapters; for each
  type the platform's native API first (Vulkan on Linux, Metal on macOS,
  Direct3D 12 on Windows), OpenGL last.
* an integer (or digits) -- the adapter with that index in :func:`list_adapters`.
* ``"discrete"``, ``"integrated"``, ``"virtual"``, ``"cpu"``, ``"software"``,
  ``"hardware"`` -- the best adapter of that kind.
* ``"vulkan"``, ``"metal"``, ``"d3d12"`` (``"dx12"``), ``"opengl"`` (``"gl"``) --
  the best adapter of that API.
* any other word -- adapters whose name contains it, case-insensitively
  (``"nvidia"``, ``"rtx 4060"``, ``"intel"``, ``"llvmpipe"``).
* ``"key=value,key=value"`` with keys ``index``, ``name``, ``type``,
  ``backend``, ``software`` -- all must match (``"name=nvidia,backend=vulkan"``);
  comma-separated keywords combine the same way (``"hardware,vulkan"``).

If several adapters match, they are tried in rank order until one creates a
device. Devices are created once per adapter and reused (see
:class:`GPUContext`); :func:`release` frees them.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import RenderError

WGPU_REQUIREMENT = "wgpu>=0.32,<0.33"
TESTED_WGPU = (0, 32)
INSTALL_HINT = (
    'install GPU support with: pip install "aether3d[gpu]" (needs Python 3.11 or newer); '
    "see docs/GPU.md"
)
ADAPTER_ENV = "AETHER3D_GPU_ADAPTER"
DEBUG_ENV = "AETHER3D_GPU_DEBUG"
BACKENDS_ENV = "AETHER3D_GPU_BACKENDS"
DXC_ENV = "AETHER3D_GPU_DXC"
"""Path of ``dxcompiler.dll``: Direct3D 12 then compiles shaders with DXC instead of FXC."""

WANTED_FEATURES = ("texture-adapter-specific-format-features", "timestamp-query")
"""Optional features requested when the adapter has them (more MSAA counts; GPU timings)."""
RAISED_LIMITS = (
    "max-texture-dimension-2d",
    "max-storage-buffer-binding-size",
    "max-buffer-size",
    "max-storage-buffers-per-shader-stage",
)
"""Limits requested at the adapter's maximum (larger images, meshes and buffers; the path
tracer's kernel reads 11 storage buffers, WebGPU's default allows 8)."""
REPORTED_LIMITS = (
    "max-texture-dimension-2d",
    "max-texture-array-layers",
    "max-buffer-size",
    "max-storage-buffer-binding-size",
    "max-uniform-buffer-binding-size",
    "max-storage-buffers-per-shader-stage",
    "max-bind-groups",
    "max-color-attachments",
    "max-compute-invocations-per-workgroup",
)

ADAPTER_TYPES = ("discrete", "integrated", "virtual", "unknown", "cpu")
_TYPE_NAMES = {
    "DiscreteGPU": "discrete",
    "IntegratedGPU": "integrated",
    "VirtualGPU": "virtual",
    "CPU": "cpu",
}
_API_NAMES = {
    "Vulkan": "vulkan",
    "Metal": "metal",
    "D3D12": "d3d12",
    "D3D11": "d3d11",
    "OpenGL": "opengl",
    "OpenGLES": "opengl",
}
_API_WORDS = {
    "vulkan": "vulkan",
    "metal": "metal",
    "d3d12": "d3d12",
    "dx12": "d3d12",
    "directx": "d3d12",
    "opengl": "opengl",
    "gl": "opengl",
    "gles": "opengl",
}
_SOFTWARE_NAMES = (
    "llvmpipe",
    "lavapipe",
    "softpipe",
    "swiftshader",
    "microsoft basic render",
    "warp",
)
_NATIVE_API = {"linux": "vulkan", "darwin": "metal", "win32": "d3d12"}

_lock = threading.RLock()
_wgpu: Any = None
_instance_notes: list[str] = []
_adapters: list[tuple[AdapterInfo, Any]] | None = None
_contexts: dict[tuple, GPUContext] = {}


# --------------------------------------------------------------------------- #
# wgpu loading
# --------------------------------------------------------------------------- #


class _WgpuLog(logging.Handler):
    """Watches wgpu's log for lost devices and (with validation on) validation messages."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.lost_events = 0
        self.messages: deque[tuple[int, str, str]] = deque(maxlen=200)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        self.count += 1
        self.messages.append((self.count, record.levelname, message))
        low = message.lower()
        if "device lost" in low or "device was lost" in low:
            self.lost_events += 1


_log = _WgpuLog()


def load_wgpu() -> Any:
    """Import wgpu (once), configuring its instance from the environment first."""
    global _wgpu
    if _wgpu is not None:
        return _wgpu
    with _lock:
        if _wgpu is not None:
            return _wgpu
        if sys.version_info < (3, 11):
            raise RenderError(
                "GPU_UNAVAILABLE",
                "GPU rendering needs Python 3.11 or newer (its wgpu dependency requires it); "
                f"this is Python {platform.python_version()}",
                hint="use backend='cpu', or install Python 3.11+ and aether3d[gpu]",
            )
        try:
            import wgpu
        except ImportError as exc:
            raise RenderError(
                "GPU_UNAVAILABLE",
                f"GPU rendering needs the optional 'wgpu' package, which is not installed ({exc})",
                hint=INSTALL_HINT,
            ) from None
        except Exception as exc:  # e.g. the native library fails to load
            raise RenderError(
                "GPU_UNAVAILABLE",
                f"the wgpu package could not be loaded: {exc}",
                hint=INSTALL_HINT,
            ) from None
        _configure_instance()
        logging.getLogger("wgpu").addHandler(_log)
        _wgpu = wgpu
        return wgpu


def _configure_instance() -> None:
    """Apply AETHER3D_GPU_DEBUG / _BACKENDS / _DXC (must run before the first enumeration)."""
    flags: list[str] = []
    backends: list[str] = []
    if os.environ.get(DEBUG_ENV, "").strip().lower() not in ("", "0", "false", "no", "off"):
        flags = ["Debug", "Validation"]
    raw = os.environ.get(BACKENDS_ENV, "").strip()
    if raw:
        names = {"vulkan": "Vulkan", "metal": "Metal", "d3d12": "DX12", "opengl": "GL"}
        for part in raw.split(","):
            api = _API_WORDS.get(part.strip().lower())
            if api not in names:
                raise RenderError(
                    "GPU_UNAVAILABLE",
                    f"{BACKENDS_ENV} lists an unknown graphics API {part.strip()!r}",
                    BACKENDS_ENV,
                    hint="use a comma-separated list of vulkan, metal, d3d12, opengl",
                )
            backends.append(names[api])
    compiler: dict[str, Any] = {}
    dxc = os.environ.get(DXC_ENV, "").strip()
    if dxc:
        path = Path(dxc)
        if path.is_dir():
            path = path / "dxcompiler.dll"
        if not path.is_file():
            raise RenderError(
                "GPU_UNAVAILABLE",
                f"{DXC_ENV} names no file: {str(path)!r}",
                DXC_ENV,
                hint="give the path of dxcompiler.dll (Windows SDK, or a DirectX Shader "
                "Compiler release), or unset it to compile with FXC",
            )
        compiler = {"dx12_compiler": "dxc", "dxc_path": str(path.absolute())}
    if not flags and not backends and not compiler:
        return
    try:
        from wgpu.backends.wgpu_native.extras import set_instance_extras

        set_instance_extras(backends=backends or ["All"], flags=flags or ["Default"], **compiler)
    except Exception as exc:  # the instance already exists (wgpu used before Aether3D)
        _instance_notes.append(f"{DEBUG_ENV}/{BACKENDS_ENV}/{DXC_ENV} not applied: {exc}")
        return
    if compiler:
        _instance_notes.append(f"Direct3D 12 compiles shaders with DXC ({compiler['dxc_path']})")
    if flags:
        _instance_notes.append("GPU validation enabled (wgpu validation and debug layers)")
    if backends:
        _instance_notes.append(f"graphics APIs limited to {', '.join(backends)}")


def wgpu_versions() -> dict[str, str]:
    wgpu = load_wgpu()
    from wgpu.backends import wgpu_native

    return {"wgpu": str(wgpu.__version__), "wgpu_native": str(wgpu_native.__version__)}


def validation_messages_since(mark: int) -> list[tuple[str, str]]:
    """wgpu log messages (level, text) logged after ``mark`` (see :func:`log_mark`)."""
    return [(level, text) for count, level, text in list(_log.messages) if count > mark]


def log_mark() -> int:
    return _log.count


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AdapterInfo:
    """What a graphics API reports about one adapter."""

    index: int
    name: str
    vendor: str
    vendor_id: int
    device_id: int
    type: str  # discrete, integrated, virtual, cpu, unknown
    backend: str  # vulkan, metal, d3d12, opengl
    driver: str
    software: bool
    features: tuple[str, ...]
    limits: tuple[tuple[str, int], ...]

    @property
    def key(self) -> tuple:
        return (self.index, self.name, self.backend, self.type, self.vendor_id, self.device_id)

    @property
    def label(self) -> str:
        kind = "software" if self.software else self.type
        return f"{self.name} ({self.backend}, {kind})"

    def to_dict(self, *, features: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "type": self.type,
            "backend": self.backend,
            "driver": self.driver,
            "software": self.software,
        }
        if features:
            data["features"] = list(self.features)
            data["limits"] = dict(self.limits)
        return data


def _rank(info: AdapterInfo) -> tuple[int, int, int]:
    native = _NATIVE_API.get(sys.platform)
    if info.backend == native:
        api = 0
    elif info.backend in ("vulkan", "metal", "d3d12"):
        api = 1
    elif info.backend == "opengl":
        api = 2
    else:
        api = 3
    kind = 4 if info.software else ADAPTER_TYPES.index(info.type)
    return (kind, api, info.index)


def _adapter_info(index: int, adapter: Any) -> AdapterInfo:
    raw = dict(adapter.info)
    name = str(raw.get("device") or "unknown adapter")
    kind = _TYPE_NAMES.get(str(raw.get("adapter_type", "")), "unknown")
    backend_type = str(raw.get("backend_type", ""))
    software = kind == "cpu" or any(s in name.lower() for s in _SOFTWARE_NAMES)
    limits = adapter.limits
    return AdapterInfo(
        index=index,
        name=name,
        vendor=str(raw.get("vendor", "")),
        vendor_id=int(raw.get("vendor_id", 0) or 0),
        device_id=int(raw.get("device_id", 0) or 0),
        type=kind,
        backend=_API_NAMES.get(backend_type, backend_type.lower() or "unknown"),
        driver=str(raw.get("description", "")),
        software=software,
        features=tuple(sorted(str(f) for f in adapter.features)),
        limits=tuple((k, int(limits[k])) for k in REPORTED_LIMITS if k in limits),
    )


def enumerate_adapters(refresh: bool = False) -> list[tuple[AdapterInfo, Any]]:
    """Every adapter wgpu reports, as ``(AdapterInfo, GPUAdapter)`` in wgpu's order."""
    global _adapters
    wgpu = load_wgpu()
    with _lock:
        if _adapters is None or refresh:
            try:
                found = wgpu.gpu.enumerate_adapters_sync()
            except Exception as exc:
                raise RenderError(
                    "GPU_UNAVAILABLE",
                    f"wgpu could not enumerate graphics adapters: {exc}",
                    hint="check the graphics driver installation (docs/GPU.md, Troubleshooting)",
                ) from None
            _adapters = [(_adapter_info(i, a), a) for i, a in enumerate(found)]
        return list(_adapters)


@dataclass(frozen=True)
class AdapterQuery:
    """A parsed adapter selection (see the module docstring)."""

    text: str = "auto"
    index: int | None = None
    name: str | None = None
    type: str | None = None
    backend: str | None = None
    software: bool | None = None

    @property
    def is_auto(self) -> bool:
        return self == AdapterQuery(self.text)

    def matches(self, info: AdapterInfo) -> bool:
        return (
            (self.index is None or info.index == self.index)
            and (self.name is None or self.name.lower() in info.name.lower())
            and (self.type is None or info.type == self.type)
            and (self.backend is None or info.backend == self.backend)
            and (self.software is None or info.software == self.software)
        )


def _bad_query(text: str, why: str) -> RenderError:
    return RenderError(
        "INVALID_SETTINGS",
        f"invalid adapter selection {text!r}: {why}",
        "adapter",
        hint="use an index, discrete/integrated/cpu/software/hardware, vulkan/metal/d3d12/"
        "opengl, part of an adapter name, or key=value pairs (index, name, type, backend, "
        "software); `aether3d gpu info` lists the adapters",
    )


def _is_keyword(word: str) -> bool:
    low = word.lower()
    return (
        low.isdigit()
        or low in ADAPTER_TYPES
        or low in _API_WORDS
        or low in ("software", "hardware")
    )


def parse_adapter_query(spec: Any) -> AdapterQuery:
    """Parse ``adapter=`` (``None`` reads ``AETHER3D_GPU_ADAPTER``, then means auto)."""
    if spec is None:
        spec = os.environ.get(ADAPTER_ENV, "").strip() or "auto"
    if isinstance(spec, bool) or not isinstance(spec, (int, str)):
        raise _bad_query(repr(spec), "expected an integer or a string")
    if isinstance(spec, int):
        if spec < 0:
            raise _bad_query(str(spec), "the index must be >= 0")
        return AdapterQuery(str(spec), index=spec)
    text = spec.strip()
    if text.lower() in ("", "auto", "default"):
        return AdapterQuery("auto")
    fields: dict[str, Any] = {}
    parts = [p.strip() for p in text.split(",")]
    if "=" not in text and not all(_is_keyword(p) for p in parts):
        parts = [text]  # a whole adapter name (names may contain commas)
    for part in parts:
        if not part:
            raise _bad_query(text, "empty item")
        if "=" in part:
            key, _, value = (s.strip() for s in part.partition("="))
            key = key.lower()
            if key == "api":
                key = "backend"
            if key not in ("index", "name", "type", "backend", "software"):
                raise _bad_query(text, f"unknown key {key!r}")
            if key in fields:
                raise _bad_query(text, f"{key!r} given twice")
            if not value:
                raise _bad_query(text, f"{key} needs a value")
            fields[key] = value
            continue
        low = part.lower()
        if low.isdigit():
            key, value = "index", part
        elif low in ADAPTER_TYPES:
            key, value = "type", low
        elif low in ("software", "hardware"):
            key, value = "software", "true" if low == "software" else "false"
        elif low in _API_WORDS:
            key, value = "backend", low
        else:
            key, value = "name", part
        if key in fields:
            raise _bad_query(text, f"{key!r} given twice")
        fields[key] = value
    query: dict[str, Any] = {"text": text}
    if "index" in fields:
        if not str(fields["index"]).isdigit():
            raise _bad_query(text, "index must be a non-negative integer")
        query["index"] = int(fields["index"])
    if "name" in fields:
        query["name"] = fields["name"]
    if "type" in fields:
        if fields["type"].lower() not in ADAPTER_TYPES:
            raise _bad_query(text, f"type must be one of {', '.join(ADAPTER_TYPES)}")
        query["type"] = fields["type"].lower()
    if "backend" in fields:
        if fields["backend"].lower() not in _API_WORDS:
            raise _bad_query(text, "backend must be vulkan, metal, d3d12 or opengl")
        query["backend"] = _API_WORDS[fields["backend"].lower()]
    if "software" in fields:
        value = str(fields["software"]).lower()
        if value not in ("true", "false", "1", "0", "yes", "no"):
            raise _bad_query(text, "software must be true or false")
        query["software"] = value in ("true", "1", "yes")
    return AdapterQuery(**query)


def ranked(adapters: list[tuple[AdapterInfo, Any]], query: AdapterQuery) -> list:
    """The adapters matching ``query``, best first."""
    return sorted((a for a in adapters if query.matches(a[0])), key=lambda a: _rank(a[0]))


def recommended(adapters: list[tuple[AdapterInfo, Any]]) -> AdapterInfo | None:
    """The adapter ``"auto"`` tries first."""
    order = ranked(adapters, AdapterQuery())
    return order[0][0] if order else None


# --------------------------------------------------------------------------- #
# Devices
# --------------------------------------------------------------------------- #


class GPUContext:
    """One device on one adapter, with everything cached for it.

    Holds the wgpu device and queue, the compiled shaders and pipelines
    (:class:`~.shaders.ShaderLibrary`), the mesh buffer cache
    (:class:`~.resources.MeshCache`), the reusable render targets
    (:class:`~.resources.TargetCache`) and the supported MSAA sample counts.
    Contexts are created by :func:`get_context` and reused by every render on
    the same adapter. A lock serializes renders on one context.
    """

    def __init__(self, wgpu: Any, info: AdapterInfo, adapter: Any, device: Any) -> None:
        from .materials import TexturePool
        from .resources import MeshCache, TargetCache, UtilityBuffers
        from .shaders import ShaderLibrary

        self.wgpu = wgpu
        self.info = info
        self.adapter = adapter
        self.device = device
        self.queue = device.queue
        self.features = frozenset(str(f) for f in device.features)
        self.limits = {k: int(v) for k, v in dict(device.limits).items()}
        self.lock = threading.RLock()
        self.lost = False
        self._lost_seen = _log.lost_events
        self.created = time.time()
        self.init_seconds = 0.0
        self.renders = 0
        self._sample_counts: dict[str, tuple[int, ...]] = {}
        self.shaders = ShaderLibrary(self)
        self.meshes = MeshCache(self)
        self.targets = TargetCache(self)
        self.utility = UtilityBuffers(self)
        self.textures = TexturePool(self)

    # -- state --------------------------------------------------------------------------

    def mark_lost(self) -> None:
        self.lost = True

    def usable(self) -> bool:
        """``False`` once the device is known (or found) to be lost."""
        if self.lost:
            return False
        if _log.lost_events != self._lost_seen:  # some device was lost: is it this one?
            self._lost_seen = _log.lost_events
            try:
                self.device.create_buffer(size=16, usage=self.wgpu.BufferUsage.COPY_DST).destroy()
            except Exception:
                self.lost = True
        return not self.lost

    def release(self) -> None:
        """Free every GPU resource of this context and destroy the device."""
        for part in (self.meshes, self.textures, self.targets, self.utility, self.shaders):
            with suppress(Exception):  # best effort while tearing down
                part.clear()
        with suppress(Exception):
            self.device.destroy()
        self.lost = True

    # -- capabilities ------------------------------------------------------------------

    @property
    def has_timestamps(self) -> bool:
        return "timestamp-query" in self.features

    def sample_counts(self, texture_format: str) -> tuple[int, ...]:
        """MSAA sample counts this device supports for a render-and-sample texture format."""
        counts = self._sample_counts.get(texture_format)
        if counts is None:
            wgpu = self.wgpu
            usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING
            found = [1]
            for n in (2, 4, 8, 16):
                try:
                    texture = self.device.create_texture(
                        size=(4, 4, 1), format=texture_format, usage=usage, sample_count=n
                    )
                except wgpu.GPUError:
                    continue
                texture.destroy()
                found.append(n)
            counts = tuple(found)
            self._sample_counts[texture_format] = counts
        return counts

    def msaa_counts(self) -> tuple[int, ...]:
        """Sample counts usable for the whole scene pass (color, object index and depth)."""
        from .shaders import COLOR_TARGET, COLOR_TARGET_HALF, DEPTH_FORMAT, OBJECT_TARGET

        color = set(self.sample_counts(COLOR_TARGET)) | set(self.sample_counts(COLOR_TARGET_HALF))
        common = (
            color & set(self.sample_counts(OBJECT_TARGET)) & set(self.sample_counts(DEPTH_FORMAT))
        )
        return tuple(sorted(common))

    def color_format(self, msaa: int) -> str:
        """Full float precision where the device can multisample it, else half floats."""
        from .shaders import COLOR_TARGET, COLOR_TARGET_HALF

        if msaa in self.sample_counts(COLOR_TARGET):
            return COLOR_TARGET
        return COLOR_TARGET_HALF

    def report(self) -> dict[str, Any]:
        return {
            "adapter": self.info.to_dict(features=False),
            "features": sorted(self.features),
            "limits": {k: self.limits[k] for k in REPORTED_LIMITS if k in self.limits},
            "msaa": list(self.msaa_counts()),
            "timestamps": self.has_timestamps,
            "device_init_seconds": self.init_seconds,
            "renders": self.renders,
            "cache": self.meshes.stats(),
        }


def _create_context(wgpu: Any, info: AdapterInfo, adapter: Any) -> GPUContext:
    t0 = time.perf_counter()
    features = [f for f in WANTED_FEATURES if f in adapter.features]
    limits = {k: adapter.limits[k] for k in RAISED_LIMITS if k in adapter.limits}
    try:
        device = adapter.request_device_sync(
            label="aether3d", required_features=features, required_limits=limits
        )
    except Exception as exc:
        first = " ".join(str(exc).split())[:300]
        raise RenderError(
            "GPU_DEVICE_FAILED",
            f"{info.label}: the driver could not create a device ({first})",
            "adapter",
            hint="try another adapter (`aether3d gpu info` lists them) or update the driver",
            details={"adapter": info.to_dict(features=False), "error": str(exc)},
        ) from None
    context = GPUContext(wgpu, info, adapter, device)
    # Compile everything now, so a broken driver fails here (and "auto" can move on).
    try:
        with translate_errors(context, "pipeline creation"):
            context.shaders.finalize_pipeline(False)
            context.shaders.shadow_pipeline()
            context.shaders.scene_pipeline(1, context.color_format(1), "none", False)
    except RenderError as exc:
        context.release()
        raise RenderError(
            exc.code if exc.code != "GPU_ERROR" else "GPU_DEVICE_FAILED",
            f"{info.label}: {exc.message}",
            "adapter",
            hint=exc.hint,
            details={**exc.details, "adapter": info.to_dict(features=False)},
        ) from None
    context.init_seconds = time.perf_counter() - t0
    return context


def get_context(adapter: Any = None) -> tuple[GPUContext, list[dict[str, Any]]]:
    """The context for an adapter selection, creating its device on first use.

    Returns the context and the adapters that were tried and failed before it
    (as dicts with ``adapter``, ``code`` and ``message``).
    """
    wgpu = load_wgpu()
    query = parse_adapter_query(adapter)
    with _lock:
        adapters = enumerate_adapters()
        candidates = ranked(adapters, query)
        if not candidates:
            names = [i.label for i, _ in adapters]
            if not adapters:
                raise RenderError(
                    "GPU_UNAVAILABLE",
                    "wgpu found no graphics adapter (no Vulkan, Metal, Direct3D 12 or OpenGL "
                    "driver it can use)",
                    hint="install a GPU driver, or Mesa's lavapipe for a software Vulkan "
                    "adapter (docs/GPU.md); or use backend='cpu'",
                )
            raise RenderError(
                "GPU_ADAPTER_NOT_FOUND",
                f"no adapter matches {query.text!r}; available: {'; '.join(names)}",
                "adapter",
                hint="`aether3d gpu info` lists the adapters and how to select them",
                details={"adapters": [i.to_dict(features=False) for i, _ in adapters]},
            )
        failures: list[dict[str, Any]] = []
        for info, raw in candidates:
            context = _contexts.get(info.key)
            if context is not None:
                if context.usable():
                    return context, failures
                context.release()
                del _contexts[info.key]
                # A lost device's adapter may be stale: enumerate again before retrying.
                fresh = {i.key: a for i, a in enumerate_adapters(refresh=True)}
                raw = fresh.get(info.key, raw)
            try:
                context = _create_context(wgpu, info, raw)
            except RenderError as exc:
                failures.append({"adapter": info.label, "code": exc.code, "message": exc.message})
                continue
            _contexts[info.key] = context
            return context, failures
        raise RenderError(
            "GPU_DEVICE_FAILED",
            "no matching adapter could create a working device: "
            + "; ".join(f"{f['adapter']}: {f['message']}" for f in failures),
            "adapter",
            hint="update the graphics driver, select another adapter, or use backend='cpu'",
            details={"attempts": failures},
        )


def release(adapter: Any = None) -> int:
    """Destroy cached GPU devices (all, or those of one adapter selection); returns how many."""
    with _lock:
        if adapter is None:
            keys = list(_contexts)
        else:
            query = parse_adapter_query(adapter)
            keys = [k for k, c in _contexts.items() if query.matches(c.info)]
        for key in keys:
            _contexts.pop(key).release()
        return len(keys)


def instance_notes() -> list[str]:
    return list(_instance_notes)


@contextmanager
def translate_errors(context: GPUContext, what: str) -> Iterator[None]:
    """Turn wgpu exceptions raised inside the block into :class:`RenderError` (``GPU_*``)."""
    wgpu = context.wgpu
    try:
        yield
    except RenderError:
        raise
    except wgpu.GPUOutOfMemoryError as exc:
        raise RenderError(
            "GPU_OUT_OF_MEMORY",
            f"{what}: the GPU ran out of memory on {context.info.name}",
            hint="lower the resolution, msaa/supersample or the shadow map size",
            details={"error": str(exc)},
        ) from None
    except (wgpu.GPUError, wgpu.GPUPipelineError) as exc:
        message = str(exc)
        low = message.lower()
        if "lost" in low and "device" in low:
            context.mark_lost()
            raise RenderError(
                "GPU_DEVICE_LOST",
                f"{what}: the GPU device was lost ({context.info.name})",
                hint="render again: the next render creates a new device (driver resets, "
                "sleep/resume and GPU switching can cause this)",
                details={"error": message},
            ) from None
        if "shadermodule" in low or "shader '" in low:
            code = "GPU_SHADER_ERROR"
        elif "pipeline" in low:
            code = "GPU_PIPELINE_ERROR"
        elif "createbuffer" in low or "createtexture" in low:
            code = "GPU_RESOURCE_ERROR"
        else:
            code = "GPU_ERROR"
        raise RenderError(
            code,
            f"{what} failed on {context.info.name}: " + " ".join(message.split())[:400],
            hint="run `aether3d gpu info --check` and see docs/GPU.md (Troubleshooting); "
            f"set {DEBUG_ENV}=1 for validation messages",
            details={"error": message, "adapter": context.info.name},
        ) from None
