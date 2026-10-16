"""Observations: what a view of the scene actually shows, measured.

:func:`observe` renders one view with an existing backend (the CPU or GPU rasterizer, or
a path tracer) and turns the render's buffers into machine-readable facts:

* **Per object** (from the object-index and depth buffers, one representative sample per
  pixel): visible pixels, pixel bounds, centroid, depth range, mean luminance, whether it
  touches the image border; and, from the geometry (exact, in the same view): whether it
  is in the view volume, its projected bounds (clipped to the near and far planes),
  whether the near or far plane cuts it, and on request its *unoccluded footprint* --
  the pixels it would cover if nothing were in front of it, rasterized at pixel centers
  with the CPU rasterizer's exact rules -- with the objects that cover the rest.
* **For the image**: coverage, luminance statistics (linear light when the render returned
  it, else decoded from the 8-bit sRGB pixels), black and clipped fractions.

An :class:`Observation` keeps its buffers (bounded by :class:`ObservationStore`), the exact
camera it saw through, the scene revision, the renderer that drew it (a CPU image is never
reported as a GPU one) and an SHA-256 of its pixels, so later steps -- a vision model's
findings about regions of the image, a correction, a verification -- refer to exactly
this image. Observing never changes the scene: no object, no history step, no revision.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..diagnostics import Diagnostic
from ..errors import AetherError, FeedbackError, QueryError, RenderError
from .views import ResolvedView, resolve_view

if TYPE_CHECKING:
    from ..rendering.renderer import RenderResult
    from ..scene import Scene, SceneObject

FORMAT = "aether3d.observation"
VERSION = 1
DEFAULT_WIDTH, DEFAULT_HEIGHT = 640, 360
BACKENDS = ("cpu", "gpu", "auto", "path", "gpu_path")
LUMA = np.array([0.2126, 0.7152, 0.0722])  # Rec. 709 luminance of linear sRGB primaries
BLACK_LEVEL = 3  # 8-bit values at or below this count as black (about 1 % of the range)


@dataclass(frozen=True)
class ObservationLimits:
    """Bounds on what one observation may cost."""

    max_pixels: int = 1920 * 1080
    """Largest image (width x height) of one observation."""
    max_path_samples: int = 50_000_000
    """Largest number of path-traced samples (width x height x spp) of one observation."""
    max_occlusion_objects: int = 16
    """Objects whose unoccluded footprint one observation rasterizes."""
    max_footprint_triangles: int = 2_000_000
    """Largest triangle count of one object whose footprint is rasterized."""


# --------------------------------------------------------------------------- #
# Buffer statistics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PixelStats:
    """Per-instance statistics of an object-index buffer (arrays indexed by instance)."""

    pixels: np.ndarray  # int64
    x0: np.ndarray  # int64, first column (width when absent)
    y0: np.ndarray
    x1: np.ndarray  # exclusive end column (-1 + 1 = 0 when absent)
    y1: np.ndarray
    cx: np.ndarray  # float64 centroid of the pixel centers (nan when absent)
    cy: np.ndarray
    dmin: np.ndarray  # float64 view depth range (nan when absent or no depth buffer)
    dmax: np.ndarray
    luminance: np.ndarray  # float64 mean linear luminance (nan when absent)


def pixel_statistics(
    index: np.ndarray, depth: np.ndarray | None, luminance: np.ndarray, count: int
) -> PixelStats:
    """Statistics of every instance ``0 .. count-1`` in the ``(h, w)`` index buffer."""
    h, w = index.shape
    flat = index.ravel()
    pix = np.flatnonzero(flat >= 0)
    ids = flat[pix].astype(np.int64)
    keep = ids < count
    if not keep.all():  # defensive: an index outside the instance table is ignored
        pix, ids = pix[keep], ids[keep]
    ys, xs = np.divmod(pix, w)
    pixels = np.bincount(ids, minlength=count).astype(np.int64)
    present = pixels > 0
    x0 = np.full(count, w, dtype=np.int64)
    y0 = np.full(count, h, dtype=np.int64)
    x1 = np.full(count, -1, dtype=np.int64)
    y1 = np.full(count, -1, dtype=np.int64)
    np.minimum.at(x0, ids, xs)
    np.minimum.at(y0, ids, ys)
    np.maximum.at(x1, ids, xs)
    np.maximum.at(y1, ids, ys)
    x1 += 1
    y1 += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.bincount(ids, weights=xs + 0.5, minlength=count) / pixels
        cy = np.bincount(ids, weights=ys + 0.5, minlength=count) / pixels
        lum = np.bincount(ids, weights=luminance.ravel()[pix], minlength=count) / pixels
    dmin = np.full(count, np.nan)
    dmax = np.full(count, np.nan)
    if depth is not None and pix.size:
        d = depth.ravel()[pix].astype(np.float64)
        finite = np.isfinite(d)
        lo = np.full(count, np.inf)
        hi = np.full(count, -np.inf)
        np.minimum.at(lo, ids[finite], d[finite])
        np.maximum.at(hi, ids[finite], d[finite])
        ok = np.isfinite(lo)
        dmin[ok], dmax[ok] = lo[ok], hi[ok]
    cx[~present] = np.nan
    cy[~present] = np.nan
    lum[~present] = np.nan
    return PixelStats(pixels, x0, y0, x1, y1, cx, cy, dmin, dmax, lum)


_SRGB_TO_LINEAR = None


def _decode_table() -> np.ndarray:
    global _SRGB_TO_LINEAR
    if _SRGB_TO_LINEAR is None:
        from ..rendering.color import srgb_to_linear

        _SRGB_TO_LINEAR = np.asarray(srgb_to_linear(np.arange(256) / 255.0), dtype=np.float64)
    return _SRGB_TO_LINEAR


def luminance_of(image: np.ndarray, linear: np.ndarray | None) -> tuple[np.ndarray, str]:
    """Per-pixel linear luminance: from ``linear`` when given, else decoded from the 8-bit
    sRGB ``image`` (values above 1 were clipped there)."""
    if linear is not None:
        lin = linear.astype(np.float64, copy=False)
        return lin[..., 0] * LUMA[0] + lin[..., 1] * LUMA[1] + lin[..., 2] * LUMA[2], "linear"
    table = _decode_table()
    lum = (
        table[image[..., 0]] * LUMA[0]
        + table[image[..., 1]] * LUMA[1]
        + table[image[..., 2]] * LUMA[2]
    )
    return lum, "display"


def _summary(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    p05, p50, p95 = np.percentile(values, (5, 50, 95))
    return {
        "mean": float(values.mean()),
        "p05": float(p05),
        "median": float(p50),
        "p95": float(p95),
        "max": float(values.max()),
    }


def image_statistics(
    image: np.ndarray, index: np.ndarray, luminance: np.ndarray, source: str
) -> dict[str, Any]:
    """Coverage, luminance and clipping statistics of a rendered image."""
    n = index.size
    covered = index >= 0
    peak = image.max(axis=-1)
    black = peak <= BLACK_LEVEL
    clipped = (image == 255).any(axis=-1)
    stats: dict[str, Any] = {
        "pixels": int(n),
        "coverage": float(np.count_nonzero(covered)) / n,
        "luminance_source": source,
        "luminance": _summary(luminance.ravel()),
        "luminance_covered": _summary(luminance[covered]),
        "black_fraction": float(np.count_nonzero(black)) / n,
        "clipped_fraction": float(np.count_nonzero(clipped)) / n,
        "clipped_covered_fraction": (
            float(np.count_nonzero(clipped & covered)) / max(int(np.count_nonzero(covered)), 1)
        ),
        "all_black": bool(black.all()),
        "uniform": bool((image == image.reshape(-1, 3)[0]).all()),
    }
    return stats


# --------------------------------------------------------------------------- #
# The observation
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class Observation:
    """One rendered view of a scene, with its buffers and measured facts (module docstring)."""

    id: str
    scene: Scene
    revision: int
    view: ResolvedView
    width: int
    height: int
    image: np.ndarray  # (h, w, 3) uint8 sRGB
    index: np.ndarray  # (h, w) int32 instance, -1 background
    depth: np.ndarray | None  # (h, w) float32 view depth, inf background
    linear: np.ndarray | None  # (h, w, 3) float32 or None (not requested)
    luminance: np.ndarray  # (h, w) float64 linear luminance
    luminance_source: str
    instances: list[dict[str, Any]]  # the render's objects: index, id, name, triangles, ...
    stats: PixelStats
    image_stats: dict[str, Any]
    renderer: dict[str, Any]
    diagnostics: list[Diagnostic]
    timings: dict[str, float]
    sha256: str
    options: dict[str, Any]
    camera: dict[str, Any]  # the render's camera report (lens, frame, fovs)
    transient: bool = False  # made inside an open operation (a preview): never reused
    _geometry: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _png: dict[str, bytes] = field(default_factory=dict, repr=False)
    _by_id: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {row["id"]: row["index"] for row in self.instances}

    # -- identity and memory -------------------------------------------------

    @property
    def nbytes(self) -> int:
        total = self.image.nbytes + self.index.nbytes + self.luminance.nbytes
        for arr in (self.depth, self.linear):
            if arr is not None:
                total += arr.nbytes
        return total + sum(len(b) for b in self._png.values())

    @property
    def stale(self) -> bool:
        """Whether the scene changed since this observation was made (an observation made
        inside an open operation, such as a preview, is always stale: it shows changes the
        revision does not count)."""
        return self.transient or self.scene.revision != self.revision

    def instance_of(self, ref: Any) -> int | None:
        """The render's instance index of an object (id, name or object), or ``None`` when
        it was not rendered (hidden, no geometry, failed to evaluate)."""
        obj = _resolve(self.scene, ref)
        return self._by_id.get(obj.id)

    # -- per-object facts ----------------------------------------------------

    def pixel_record(self, instance: int) -> dict[str, Any]:
        """What the image shows of one rendered instance (no geometry needed)."""
        s = self.stats
        row = self.instances[instance]
        n = int(s.pixels[instance])
        out: dict[str, Any] = {
            "id": row["id"],
            "name": row["name"],
            "rendered": True,
            "visible_pixels": n,
            "visible_samples": int(row.get("visible_samples", 0)),
            "pixel_fraction": n / (self.width * self.height),
        }
        if n:
            x0, y0, x1, y1 = (int(a[instance]) for a in (s.x0, s.y0, s.x1, s.y1))
            out["screen_bounds"] = [x0, y0, x1, y1]
            out["center"] = [round(float(s.cx[instance]), 3), round(float(s.cy[instance]), 3)]
            out["touches_border"] = {
                "left": x0 == 0,
                "top": y0 == 0,
                "right": x1 == self.width,
                "bottom": y1 == self.height,
            }
            if not math.isnan(float(s.dmin[instance])):
                out["depth_range"] = [float(s.dmin[instance]), float(s.dmax[instance])]
            out["luminance"] = round(float(s.luminance[instance]), 6)
        return out

    def record(self, ref: Any, *, geometry: bool = True, occlusion: bool = False) -> dict[str, Any]:
        """Everything this observation knows about one object: its pixels and, while the
        scene is still at this observation's revision, its geometry in this view."""
        obj = _resolve(self.scene, ref)
        instance = self._by_id.get(obj.id)
        if instance is not None:
            out = self.pixel_record(instance)
        else:
            out = {
                "id": obj.id,
                "name": obj.name,
                "rendered": False,
                "visible_pixels": 0,
                "reason": _not_rendered_reason(obj),
            }
        if geometry and obj.camera is None and obj.light is None:
            facts = self.geometry(obj, occlusion=occlusion)
            if facts is not None:
                out["geometry"] = facts
        return out

    def geometry(self, ref: Any, *, occlusion: bool = False) -> dict[str, Any] | None:
        """Exact facts about one object's geometry in this view (``None`` without geometry).

        Raises ``FeedbackError("STALE_OBSERVATION")`` once the scene has changed: the facts
        would describe a different scene than the image.
        """
        obj = _resolve(self.scene, ref)
        cached = self._geometry.get(obj.id)
        if cached is not None and (not occlusion or "occlusion" in cached):
            return cached
        if self.stale:
            raise FeedbackError(
                "STALE_OBSERVATION",
                f"the scene changed since observation {self.id} (revision {self.revision}, "
                f"now {self.scene.revision})",
                "observation",
                hint="observe again",
            )
        facts = geometry_in_view(self, obj, occlusion=occlusion)
        if facts is not None:
            self._geometry[obj.id] = facts
        return facts

    def objects_in_rect(self, rect: tuple[float, float, float, float]) -> list[dict[str, Any]]:
        """The objects visible inside a pixel rectangle ``(x0, y0, x1, y1)`` (clipped to the
        image), with their pixel counts there, most pixels first."""
        x0, y0, x1, y1 = _pixel_rect(rect, self.width, self.height)
        if x0 >= x1 or y0 >= y1:
            return []
        region = self.index[y0:y1, x0:x1].ravel()
        total = region.size
        found = region[region >= 0]
        counts = np.bincount(found, minlength=len(self.instances)) if found.size else []
        out = []
        for k in np.flatnonzero(counts):
            row = self.instances[int(k)]
            out.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "pixels": int(counts[k]),
                    "fraction": int(counts[k]) / total,
                }
            )
        out.sort(key=lambda r: (-r["pixels"], r["name"]))
        return out

    # -- images ---------------------------------------------------------------

    def png(self, kind: str = "image", *, ref: Any = None, compression: int = 1) -> bytes:
        """PNG bytes of the clean image (``"image"``), the numbered markers (``"annotated"``,
        :meth:`markers`), the object-index colors (``"ids"``), the depth (``"depth"``) or one
        object's mask (``"mask"`` with ``ref``)."""
        from ..rendering.png import encode_png
        from . import overlay

        key = kind if ref is None else f"{kind}:{_resolve(self.scene, ref).id}"
        found = self._png.get(key)
        if found is not None:
            return found
        text = {"Software": "Aether3D", "Comment": f"observation {self.id} ({kind})"}
        if kind == "image":
            pixels = self.image
        elif kind == "annotated":
            pixels = overlay.annotate(self.image, self.index, self._marker_boxes())
        elif kind == "ids":
            pixels = overlay.id_colors(self.index)
        elif kind == "depth":
            pixels = overlay.depth_gray(self.depth)[..., 0] if self.depth is not None else None
            if pixels is None:
                raise FeedbackError("OUTPUT_NOT_CAPTURED", "this observation has no depth buffer")
        elif kind == "mask":
            instance = self.instance_of(ref)
            if instance is None:
                raise FeedbackError(
                    "NOT_RENDERED", "the object was not rendered in this observation", "object"
                )
            pixels = overlay.mask_image(self.index, instance)[..., 0]
        else:
            raise FeedbackError(
                "INVALID_ARGUMENT",
                f"unknown image kind {kind!r}",
                "kind",
                hint="image, annotated, ids, depth, mask",
            )
        data = encode_png(pixels, text=text, compression=compression)
        self._png[key] = data
        return data

    def crop_png(
        self, rect: tuple[float, float, float, float], *, scale: int = 1, compression: int = 1
    ) -> tuple[bytes, list[int]]:
        """PNG bytes of a pixel rectangle of the clean image (enlarged ``scale`` times by
        pixel repetition, never resampled), and the rectangle actually used."""
        from ..rendering.png import encode_png

        x0, y0, x1, y1 = _pixel_rect(rect, self.width, self.height)
        if x0 >= x1 or y0 >= y1:
            raise FeedbackError("INVALID_ARGUMENT", "the crop lies outside the image", "crop")
        pixels = self.image[y0:y1, x0:x1]
        if scale > 1:
            pixels = np.repeat(np.repeat(pixels, scale, axis=0), scale, axis=1)
        text = {"Software": "Aether3D", "Comment": f"observation {self.id} crop"}
        return encode_png(pixels, text=text, compression=compression), [x0, y0, x1, y1]

    def markers(self) -> list[dict[str, Any]]:
        """The numbered markers of the annotated image: number, object id and name."""
        out = []
        for number, instance, _box in self._marker_boxes():
            row = self.instances[instance]
            out.append({"marker": number, "id": row["id"], "name": row["name"]})
        return out

    def _marker_boxes(self) -> list[tuple[int, int, tuple[int, int, int, int]]]:
        focus = self.options.get("focus_instances") or []
        order = list(focus) if focus else self._largest(self.options.get("max_markers", 12))
        s = self.stats
        out = []
        for number, k in enumerate((k for k in order if s.pixels[k] > 0), start=1):
            out.append((number, k, (int(s.x0[k]), int(s.y0[k]), int(s.x1[k]), int(s.y1[k]))))
        return out

    def _largest(self, count: int) -> list[int]:
        pixels = self.stats.pixels
        visible = np.flatnonzero(pixels > 0)
        ranked = sorted(visible.tolist(), key=lambda k: (-int(pixels[k]), k))
        return ranked[:count]

    # -- reports ----------------------------------------------------------------

    def to_dict(self, *, max_objects: int = 50, focus: Any = None) -> dict[str, Any]:
        """The observation as JSON (without pixels): the view, the renderer, statistics,
        the markers and the objects -- the focus objects first, then the most visible ones,
        at most ``max_objects`` -- with their geometry while the scene is unchanged."""
        focus_ids: list[str] = []
        if focus is not None:
            focus_ids = [_resolve(self.scene, f).id for f in focus]
        else:
            focus_ids = list(self.options.get("focus", []))
        s = self.stats
        visible = int(np.count_nonzero(s.pixels))
        chosen: list[str] = list(dict.fromkeys(focus_ids))
        for k in self._largest(len(self.instances)):
            if len(chosen) >= max_objects:
                break
            oid = self.instances[k]["id"]
            if oid not in chosen:
                chosen.append(oid)
        geometry = not self.stale
        objects = []
        for oid in chosen[: max(max_objects, len(focus_ids))]:
            if oid not in self.scene._objects:
                continue
            try:
                objects.append(
                    self.record(oid, geometry=geometry and oid in focus_ids, occlusion=False)
                    if oid in focus_ids
                    else self.record(oid, geometry=False)
                )
            except FeedbackError:
                objects.append(self.record(oid, geometry=False))
        return {
            "format": FORMAT,
            "version": VERSION,
            "id": self.id,
            "scene": {"name": self.scene.name, "revision": self.revision, "stale": self.stale},
            "view": self.view.to_dict(),
            "camera": self.camera,
            "conventions": {
                "world": "right-handed, +Z up, meters; objects face -Y",
                "image": "pixels from the top-left corner; pixel (i, j) covers [i, i+1) x "
                "[j, j+1); rectangles are [x0, y0, x1, y1) with exclusive ends",
                "depth": "distance along the camera's view direction",
                "per_pixel_facts": "one representative sample per pixel (object index, depth)",
            },
            "renderer": self.renderer,
            "image": {
                "width": self.width,
                "height": self.height,
                "sha256": self.sha256,
                "outputs": sorted(self.options.get("outputs", [])),
            },
            "stats": self.image_stats,
            "objects_rendered": len(self.instances),
            "objects_visible": visible,
            "objects": objects,
            "objects_omitted": max(visible - len(objects), 0),
            "markers": self.markers(),
            "diagnostics": [
                {"code": d.code, "severity": d.severity, "message": d.message}
                for d in self.diagnostics[:20]
            ],
            "timings": {k: round(v, 6) for k, v in self.timings.items()},
        }


def _pixel_rect(rect: Any, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in rect)
    return (
        max(0, min(width, math.floor(x0))),
        max(0, min(height, math.floor(y0))),
        max(0, min(width, math.ceil(x1))),
        max(0, min(height, math.ceil(y1))),
    )


def _resolve(scene: Scene, ref: Any) -> SceneObject:
    from ..scene import SceneObject

    if isinstance(ref, SceneObject):
        if ref.scene is not scene:
            raise FeedbackError("OBJECT_NOT_FOUND", f"{ref!r} is not in this scene", "object")
        return ref
    if isinstance(ref, str):
        found = scene._objects.get(ref) or scene._by_name.get(ref)
        if found is not None:
            return found
    raise FeedbackError("OBJECT_NOT_FOUND", f"no object with id or name {ref!r}", "object")


def _not_rendered_reason(obj: SceneObject) -> str:
    if obj.camera is not None or obj.light is not None:
        return "a camera or light has no surface"
    if not obj.visible:
        return "hidden (visible is false)"
    if obj._mesh is None and obj._geometry is None:
        return "no geometry (an empty)"
    try:
        mesh = obj.mesh
    except AetherError:
        return "its procedural geometry fails to evaluate"
    if mesh is None or mesh.triangle_count == 0:
        return "no triangles"
    return "not drawn (a zero or non-finite scale, or a failure reported in diagnostics)"


# --------------------------------------------------------------------------- #
# Geometry in a view
# --------------------------------------------------------------------------- #


def _view(obs: Observation):
    from ..spatial.queries import View

    return View(obs.scene, obs.view.target, obs.width, obs.height)


def geometry_in_view(obs: Observation, obj: SceneObject, *, occlusion: bool) -> dict | None:
    """Exact facts about ``obj``'s geometry seen through ``obs``'s camera (the scene must be
    at the observation's revision)."""
    from ..spatial.index import get_index
    from ..spatial.queries import objects_in_view

    index = get_index(obs.scene)
    if obj.id in index.failed:
        return {"error": index.failed[obj.id].to_dict()}
    entry = index.by_id.get(obj.id)
    if entry is None:
        return None
    view = _view(obs)
    try:
        matches = objects_in_view(
            obs.scene,
            camera=obs.view.target,
            width=obs.width,
            height=obs.height,
            objects=[obj.id],
            include_hidden=True,
        )
    except QueryError as exc:
        return {"error": exc.to_dict()}
    frustum = matches[0].visibility if matches else "outside"
    pts = entry.points
    rel = pts - view.eye
    depth = rel @ view.forward
    out: dict[str, Any] = {
        "frustum": frustum,
        "depth_extent": [float(depth.min()), float(depth.max())],
        "world_bounds": {
            "min": [float(v) for v in entry.box[0]],
            "max": [float(v) for v in entry.box[1]],
        },
    }
    extent = _projected_extent(view, entry)
    out.update(extent)
    if occlusion and obj.visible:
        out["occlusion"] = footprint_occlusion(obs, obj)
    return out


def _projected_extent(view: Any, entry: Any) -> dict[str, Any]:
    """The pixel bounds of the geometry between the near and far planes (exact: vertices
    inside the slab and edge crossings of its planes), and whether those planes cut it in
    view."""
    a, b, c = entry.corners()
    eye, forward = view.eye, view.forward
    near, far = view.near, view.far
    pts = entry.points
    d = (pts - eye) @ forward
    inside = (d >= near) & (d <= far)
    candidates = [pts[inside]]
    cuts: dict[str, bool] = {}
    for name, plane in (("near", near), ("far", far)):
        crossing_points = []
        for p, q in ((a, b), (b, c), (c, a)):
            dp, dq = (p - eye) @ forward, (q - eye) @ forward
            cross = (dp - plane) * (dq - plane) < 0.0
            if cross.any():
                t = (plane - dp[cross]) / (dq[cross] - dp[cross])
                crossing_points.append(p[cross] + t[:, None] * (q[cross] - p[cross]))
        if crossing_points:
            cp = np.concatenate(crossing_points)
            candidates.append(cp)
            x, y, _, _ = view.project(cp)
            ok = np.isfinite(x) & np.isfinite(y)
            cuts[name] = bool(
                np.any(ok & (x >= 0) & (x <= view.width) & (y >= 0) & (y <= view.height))
            )
        else:
            cuts[name] = False
    pts_all = np.concatenate(candidates) if candidates else np.zeros((0, 3))
    out: dict[str, Any] = {"near_clipped": cuts["near"], "far_clipped": cuts["far"]}
    if not pts_all.shape[0]:
        out["projected_bounds"] = None
        out["in_frame_fraction"] = 0.0
        return out
    x, y, _, _ = view.project(pts_all)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        out["projected_bounds"] = None
        out["in_frame_fraction"] = 0.0
        return out
    x0, x1 = float(x[ok].min()), float(x[ok].max())
    y0, y1 = float(y[ok].min()), float(y[ok].max())
    out["projected_bounds"] = [round(v, 3) for v in (x0, y0, x1, y1)]
    area = max(x1 - x0, 1e-9) * max(y1 - y0, 1e-9)
    ix = max(0.0, min(x1, view.width) - max(x0, 0.0))
    iy = max(0.0, min(y1, view.height) - max(y0, 0.0))
    out["in_frame_fraction"] = round(min(1.0, ix * iy / area), 6)
    out["projected_center"] = [round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3)]
    return out


def footprint(obs: Observation, obj: SceneObject) -> np.ndarray:
    """The pixels ``obj`` would cover in ``obs``'s image if nothing were in front of it: its
    triangles rasterized alone at pixel centers with the CPU rasterizer's exact rules
    (clipping, fixed-point edge functions, the render settings' culling and single-sided
    materials). Alpha masks are not applied: cut-out texels count as covered."""
    from ..rendering import raster
    from ..rendering.cpu import _clip_coords, _front_facing, _transform_points
    from ..rendering.scene_data import _mat4_mul
    from ..spatial.index import get_index

    entry = get_index(obs.scene).by_id.get(obj.id)
    w, h = obs.width, obs.height
    covered = np.zeros(h * w, dtype=bool)
    if entry is None or entry.singular:
        return covered.reshape(h, w)
    rc = obs.view.render_camera(obs.scene)
    vp = _mat4_mul(rc.camera.projection_matrix(w / h), rc.frame.view_matrix())
    mesh = entry.mesh
    P = _transform_points(mesh.positions, entry.world)
    clip = _clip_coords(P, vp)
    tv = mesh.triangles.astype(np.int64)
    finite = np.all(np.isfinite(clip), axis=1)[tv].all(axis=1)
    tv = tv[finite]
    groups = None if mesh.groups is None else mesh.groups[finite]
    cull = obs.options.get("cull", "none")
    single = _single_sided(obj, groups, tv.shape[0])
    if tv.size and (cull != "none" or single.any()):
        # The renderer's rules: the settings' culling, then single-sided materials' backs.
        mirrored = np.full(tv.shape[0], entry.det < 0)
        eye, forward = rc.frame.position, rc.frame.forward
        front = _front_facing(P, tv, mirrored, eye, forward, rc.camera.is_perspective)
        keep = np.ones(tv.shape[0], dtype=bool)
        if cull == "back":
            keep &= front
        elif cull == "front":
            keep &= ~front
        keep &= front | ~single
        tv = tv[keep]
    if not tv.size:
        return covered.reshape(h, w)
    ids = np.arange(tv.shape[0], dtype=np.int64)
    codes = raster.outcodes(clip)
    oc = codes[tv]
    outside = (oc[:, 0] & oc[:, 1] & oc[:, 2]) != 0
    ids, tv, oc = ids[~outside], tv[~outside], oc[~outside]
    inside = (oc[:, 0] | oc[:, 1] | oc[:, 2]) == 0
    VX, VY, VZ = raster.vertex_screen(clip, w, h)
    tin = tv[inside]
    X, Y, z = VX[tin], VY[tin], VZ[tin]
    ids_in = ids[inside]
    if not inside.all():
        cpos, cids = raster.clip_triangles(clip[tv[~inside]], ids[~inside])
        cX, cY, cz, _ = raster.screen_setup(cpos, w, h)
        X, Y, z = np.concatenate([X, cX]), np.concatenate([Y, cY]), np.concatenate([z, cz])
        ids_in = np.concatenate([ids_in, cids])
    keys = np.full(w * h, raster.EMPTY, dtype=np.int64)
    id_bits = max(1, int(max(tv.shape[0], 1)).bit_length())
    raster.rasterize(
        X, Y, z, raster.signed_area(X, Y), ids_in, w, h, keys, id_bits, raster.RasterStats()
    )
    return (keys != raster.EMPTY).reshape(h, w)


def _single_sided(obj: SceneObject, groups: np.ndarray | None, count: int) -> np.ndarray:
    """Per triangle: whether its material (the object's, or its face group's) is
    single-sided."""
    base = obj.material
    out = np.full(count, base is not None and not base.double_sided, dtype=bool)
    if groups is not None:
        for group, material in obj.group_materials.items():
            out[groups == group] = not material.double_sided
    return out


def footprint_occlusion(obs: Observation, obj: SceneObject) -> dict[str, Any]:
    """How much of ``obj``'s unoccluded footprint the image shows, and what covers the rest
    (exact at pixel centers; see :func:`footprint`)."""
    limits: ObservationLimits = obs.options.get("limits_object") or ObservationLimits()
    entry_triangles = obj.mesh.triangle_count if obj.mesh is not None else 0
    if entry_triangles > limits.max_footprint_triangles:
        return {
            "computed": False,
            "reason": f"{entry_triangles} triangles exceed max_footprint_triangles",
        }
    mask = footprint(obs, obj)
    total = int(np.count_nonzero(mask))
    instance = obs._by_id.get(obj.id)
    seen = obs.index[mask]
    visible = 0 if instance is None else int(np.count_nonzero(seen == instance))
    occluders = []
    others = seen[(seen >= 0) & (seen != (-2 if instance is None else instance))]
    if others.size:
        counts = np.bincount(others, minlength=len(obs.instances))
        for k in np.flatnonzero(counts):
            row = obs.instances[int(k)]
            occluders.append({"id": row["id"], "name": row["name"], "pixels": int(counts[k])})
        occluders.sort(key=lambda r: (-r["pixels"], r["name"]))
    return {
        "computed": True,
        "basis": "footprint rasterized at pixel centers (CPU rasterizer rules); the object "
        "index is the renderer's representative sample, which can differ at silhouettes",
        "footprint_pixels": total,
        "visible_pixels_in_footprint": visible,
        "visible_fraction": (visible / total) if total else None,
        "occluders": occluders[:10],
        "uncovered_pixels": int(np.count_nonzero(seen < 0)),
    }


# --------------------------------------------------------------------------- #
# Observing
# --------------------------------------------------------------------------- #


def _image_digest(image: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{image.shape[1]}x{image.shape[0]}x3:".encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def _renderer_report(result: RenderResult) -> dict[str, Any]:
    backend = result.backend
    name = backend.get("name")
    out: dict[str, Any] = {
        "backend": name,
        "requested": backend.get("requested"),
        "device": "gpu" if name in ("gpu", "gpu_path") else "cpu",
    }
    if name in ("gpu", "gpu_path"):
        adapter = backend.get("adapter", {})
        out["adapter"] = {
            "name": adapter.get("name"),
            "backend": adapter.get("backend"),
            "software": adapter.get("software"),
        }
    if name in ("path", "gpu_path"):
        out["samples_per_pixel"] = backend.get("samples_rendered")
        out["bounces"] = backend.get("bounces")
        out["seed"] = backend.get("seed")
        noise = result.stats.get("noise")
        if isinstance(noise, dict):
            out["noise"] = noise
    else:
        out["antialiasing_samples"] = result.settings.get("samples")
    fallback = [d for d in result.diagnostics if d.code == "BACKEND_FALLBACK"]
    if fallback:
        out["fallback"] = fallback[0].message
    return out


def observe(
    scene: Scene,
    view: Any = None,
    *,
    width: int | None = None,
    height: int | None = None,
    backend: str = "cpu",
    adapter: Any = None,
    samples: int | None = None,
    shadows: bool = False,
    path: dict[str, Any] | None = None,
    linear: bool = False,
    focus: Any = None,
    occlusion: bool = False,
    max_markers: int = 12,
    limits: ObservationLimits | None = None,
    store: ObservationStore | None = None,
) -> Observation:
    """Render ``view`` of ``scene`` and measure what it shows (see the module docstring).

    * ``view`` -- a view specification (:mod:`.views`): ``None`` is the active camera.
    * ``width``, ``height`` -- the image size (default 640 x 360; one of them keeps the
      aspect of the other).
    * ``backend`` -- ``"cpu"`` (default), ``"gpu"``, ``"auto"``, ``"path"`` or ``"gpu_path"``
      (with ``path`` settings, e.g. ``{"samples": 16}``); ``adapter``, ``samples`` and
      ``shadows`` as :func:`aether3d.render` takes them.
    * ``linear`` -- also read back linear light (exact exposure statistics above 1).
    * ``focus`` -- a selector of objects whose geometry facts are computed now (and listed
      first); ``occlusion`` also rasterizes their unoccluded footprints.
    * ``store`` -- an :class:`ObservationStore`: an observation of the same view with the
      same options at the scene's current revision is reused (renders are deterministic),
      and a new one is kept there.

    Never changes the scene. Raises ``FeedbackError`` (views, limits) or ``RenderError``.
    """
    from ..plans import select_objects

    t0 = time.perf_counter()
    limits = limits or ObservationLimits()
    if backend not in BACKENDS:
        raise FeedbackError(
            "INVALID_ARGUMENT", f"unknown backend {backend!r}", "backend", hint=", ".join(BACKENDS)
        )
    w, h = _size(width, height)
    if w * h > limits.max_pixels:
        raise FeedbackError(
            "LIMIT_EXCEEDED",
            f"{w}x{h} is {w * h} pixels; an observation has at most {limits.max_pixels}",
            "width",
            hint="observe smaller, or crop a region of interest",
        )
    resolved = resolve_view(scene, view, w, h)
    options: dict[str, Any] = {}
    outputs = ["image", "object_index", "depth"] + (["linear"] if linear else [])
    render_options: dict[str, Any] = {"outputs": tuple(outputs)}
    if backend in ("path", "gpu_path"):
        settings = dict(path or {})
        spp = int(settings.get("samples", 16))
        if w * h * spp > limits.max_path_samples:
            raise FeedbackError(
                "LIMIT_EXCEEDED",
                f"{w}x{h} at {spp} samples per pixel exceeds the observation budget of "
                f"{limits.max_path_samples} path-traced samples",
                "path",
                hint="lower the size or the samples, or observe with a rasterizer",
            )
        render_options["path"] = settings
    else:
        if path is not None:
            raise FeedbackError(
                "INVALID_ARGUMENT", "path settings need backend 'path' or 'gpu_path'", "path"
            )
        if samples is not None:
            render_options["samples"] = samples
        if shadows:
            render_options["shadows"] = True
    if adapter is not None and backend not in ("cpu", "path"):
        render_options["adapter"] = adapter
    focus_objects: list[SceneObject] = []
    if focus is not None:
        try:
            focus_objects = select_objects(scene, focus, "/focus")
        except AetherError as exc:
            raise FeedbackError(
                getattr(exc, "code", "INVALID_SELECTOR"),
                getattr(exc, "message", str(exc)),
                "focus",
                hint=getattr(exc, "hint", None),
            ) from None
    revision = scene.revision
    transient = scene.history.in_progress
    if transient:  # inside an operation the revision does not count the changes yet
        store = None
    key = json.dumps(
        {
            "view": resolved.key,
            "backend": backend,
            "options": {k: v for k, v in render_options.items() if k != "outputs"},
            "linear": linear,
        },
        sort_keys=True,
        default=str,
    )
    if store is not None:
        found = store.find(scene, key)
        if found is not None:
            for k, obj in enumerate(focus_objects):
                if obj.camera is None and obj.light is None:
                    found.geometry(obj, occlusion=occlusion and k < limits.max_occlusion_objects)
            store.recount()
            return found
    t_render = time.perf_counter()
    result = scene.render(
        camera=resolved.target, width=w, height=h, backend=backend, **render_options
    )
    t_render = time.perf_counter() - t_render
    t_metrics = time.perf_counter()
    if result.object_index is None:  # pragma: no cover - the outputs were requested
        raise RenderError("OUTPUT_NOT_RENDERED", "the render returned no object index")
    lum, source = luminance_of(result.image, result.linear if linear else None)
    stats = pixel_statistics(result.object_index, result.depth, lum, len(result.objects))
    image_stats = image_statistics(result.image, result.object_index, lum, source)
    digest = _image_digest(result.image)
    t_metrics = time.perf_counter() - t_metrics
    oid = "obs-" + hashlib.sha256(f"{revision}|{key}|{digest}".encode()).hexdigest()[:12]
    options.update(
        {
            "outputs": outputs,
            "cull": result.settings.get("cull", "none"),
            "focus": [o.id for o in focus_objects],
            "max_markers": max_markers,
            "limits_object": limits,
            "backend_requested": backend,
            "key": key,
        }
    )
    obs = Observation(
        id=oid,
        scene=scene,
        revision=revision,
        view=resolved,
        width=w,
        height=h,
        image=result.image,
        index=result.object_index,
        depth=result.depth,
        linear=result.linear if linear else None,
        luminance=lum,
        luminance_source=source,
        instances=result.objects,
        stats=stats,
        image_stats=image_stats,
        renderer=_renderer_report(result),
        diagnostics=list(result.diagnostics),
        timings={"render": t_render, "metrics": t_metrics},
        sha256=digest,
        options=options,
        camera=_camera_report(result.camera),
        transient=transient,
    )
    obs.options["focus_instances"] = [obs._by_id[o.id] for o in focus_objects if o.id in obs._by_id]
    t_geometry = time.perf_counter()
    budget = limits.max_occlusion_objects
    for k, obj in enumerate(focus_objects):
        if obj.camera is None and obj.light is None:
            obs.geometry(obj, occlusion=occlusion and k < budget)
    obs.timings["geometry"] = time.perf_counter() - t_geometry
    obs.timings["total"] = time.perf_counter() - t0
    if store is not None:
        store.put(obs)
    return obs


def _camera_report(camera: dict[str, Any]) -> dict[str, Any]:
    out = dict(camera)
    for key in ("position", "forward", "up", "right"):
        if key in out:
            out[key] = [round(float(v), 9) for v in out[key]]
    return out


def _size(width: Any, height: Any) -> tuple[int, int]:
    for name, value in (("width", width), ("height", height)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or not 16 <= value <= 8192
        ):
            raise FeedbackError(
                "INVALID_ARGUMENT", f"{name} must be an integer from 16 to 8192", name
            )
    if width is None and height is None:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    if width is None:
        return max(16, round(height * DEFAULT_WIDTH / DEFAULT_HEIGHT)), height
    if height is None:
        return width, max(16, round(width * DEFAULT_HEIGHT / DEFAULT_WIDTH))
    return width, height


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #


class ObservationStore:
    """Observations by id, least recently used first out, within a memory budget.

    Observations are kept with their buffers (a 640 x 360 observation holds about 3 MB),
    so later steps can ground a vision model's regions in the object index, crop, and
    compare. ``max_bytes`` bounds the total; the oldest are dropped first, and asking for
    a dropped one says so (``OBSERVATION_EXPIRED``) rather than pretending it never
    existed.
    """

    def __init__(self, max_bytes: int = 256 * 2**20, max_count: int = 64) -> None:
        self.max_bytes = max_bytes
        self.max_count = max_count
        self._items: OrderedDict[str, Observation] = OrderedDict()
        self._expired: OrderedDict[str, None] = OrderedDict()
        self.nbytes = 0

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, oid: object) -> bool:
        return oid in self._items

    def put(self, obs: Observation) -> Observation:
        old = self._items.pop(obs.id, None)
        if old is not None:
            self.nbytes -= old.nbytes
        self._items[obs.id] = obs
        self._expired.pop(obs.id, None)
        self.nbytes += obs.nbytes
        while self._items and (self.nbytes > self.max_bytes or len(self._items) > self.max_count):
            if len(self._items) == 1:
                break
            oid, dropped = self._items.popitem(last=False)
            self.nbytes -= dropped.nbytes
            self._expired[oid] = None
            while len(self._expired) > 4096:
                self._expired.popitem(last=False)
        return obs

    def get(self, oid: Any) -> Observation:
        if not isinstance(oid, str):
            raise FeedbackError("INVALID_ARGUMENT", "an observation id is a string", "observation")
        found = self._items.get(oid)
        if found is None:
            if oid in self._expired:
                raise FeedbackError(
                    "OBSERVATION_EXPIRED",
                    f"observation {oid} was dropped to stay within the memory budget",
                    "observation",
                    hint="observe again",
                )
            raise FeedbackError(
                "OBSERVATION_NOT_FOUND", f"no observation {oid!r}", "observation", hint="observe"
            )
        self._items.move_to_end(oid)
        found_png = found._png  # PNGs count toward memory once made
        del found_png
        return found

    def find(self, scene: Scene, key: str) -> Observation | None:
        """An observation of ``scene`` at its current revision with this content key."""
        for obs in reversed(self._items.values()):
            if obs.scene is scene and obs.revision == scene.revision and obs.options["key"] == key:
                self._items.move_to_end(obs.id)
                return obs
        return None

    def recount(self) -> None:
        """Recompute the memory total (after PNGs were encoded on demand)."""
        self.nbytes = sum(o.nbytes for o in self._items.values())

    def clear(self) -> None:
        self._items.clear()
        self._expired.clear()
        self.nbytes = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": len(self._items),
            "bytes": self.nbytes,
            "max_bytes": self.max_bytes,
            "ids": list(self._items),
        }
