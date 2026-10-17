"""Painting: brushes and strokes on painted textures, in 3D on a surface, through a view, or
in the texture's own UV space.

**Texel maps.** Every texel center of a texture is a point of the object's surface (or of
no surface: padding). A :class:`TexelMap` computes, per 64 x 64 tile and only when a brush
first reaches it, each covered texel's triangle, barycentrics, object-space position and
normal, from the exact UV rasterization (:mod:`.raster`). Tiles are culled by the 3D boxes
of the triangles their UVs cover, so a dab only looks at the tiles it can reach.

**Brushes** (``Brush``): ``paint`` (lay a color), ``erase`` (lower the layer's alpha),
``blur`` (average the layer over a neighbourhood), ``smudge`` (drag the colors along the
stroke), ``clone`` (copy from a source texture, at a texel offset: ``restore`` when the
source is the texture as it was), and ``mask`` (paint the layer's mask). Each dab has a
footprint and a falloff (``smooth``, ``linear``, ``constant``, ``gaussian``) shaped by
``hardness``; its alpha is ``strength x falloff x pressure``.

**Footprints.**

* ``"surface"``: texels whose surface point lies within ``radius`` (world units, as the
  points; ``coordinates="object"`` for the object's own) of the
  dab center -- continuous across UV seams, since neighbouring points on both sides of a
  seam are near in 3D. Texels whose normal faces away from the dab's surface normal by
  more than ``normal_limit`` degrees are left alone (the other side of a thin sheet).
* ``"view"``: texels whose surface point projects within ``radius`` pixels of the dab on
  the screen *and is visible there*: the view's depth and object buffers (a render of the
  whole scene) decide, so other objects in front block the brush and the hidden side of
  the object is not painted (projection painting). ``backfaces=True`` paints through.
* ``"uv"``: texels within ``radius`` texels of the dab in the texture itself (2D painting;
  this one does not cross seams).

**Strokes.** Dabs are placed along the path every ``spacing x radius`` (interpolating
pressure), mirrored with ``symmetry`` (object-space axes). A paint stroke accumulates its
coverage ``S`` dab by dab (``S += a (1 - S)``: build-up, ``flow``) and is laid once over
the layer as it was, with ``opacity`` as the ceiling -- so a stroke never paints over
itself beyond its opacity, as in image editors. Padding texels (outside every island) next
to painted texels receive the stroke too (``padding`` rings), so filtering does not bleed
the old colors at seams. A stroke is one step of the scene's history (all the textures
it paints together), rolled back if anything fails, and cancellable.
"""

from __future__ import annotations

import dataclasses
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import TexturingError
from ..materials.painted import FULL, TILE, PaintedTexture, composite_over
from .dilate import SHIFTS
from .raster import rasterize_uv

BRUSH_KINDS = ("paint", "erase", "blur", "smudge", "clone", "mask")
FALLOFFS = ("smooth", "linear", "constant", "gaussian")
SPACES = ("surface", "view", "uv")
MAX_DABS = 100_000
MAX_POINTS = 100_000
TILE_CACHE_BYTES = 512 * 2**20


def _bad(message: str, location: str) -> TexturingError:
    return TexturingError("INVALID_ARGUMENT", message, location)


# --------------------------------------------------------------------------- #
# Texel maps
# --------------------------------------------------------------------------- #


@dataclass
class TileTexels:
    """The covered texels of one tile and their surface points (object space)."""

    local: np.ndarray  # (k,) index within the tile (row * TILE + column)
    triangle: np.ndarray  # (k,)
    bary: np.ndarray  # (k, 3)
    position: np.ndarray  # (k, 3)
    normal: np.ndarray  # (k, 3) unit

    @property
    def nbytes(self) -> int:
        return int(
            sum(
                a.nbytes for a in (self.local, self.triangle, self.bary, self.position, self.normal)
            )
        )


class TexelMap:
    """Surface points of the texel centers of a ``width x height`` texture mapped by a mesh's
    UVs (see the module docstring)."""

    def __init__(self, mesh: Any, width: int, height: int) -> None:
        if mesh.uvs is None:
            raise TexturingError("NO_UVS", "the mesh has no UVs to paint through", "mesh")
        self.mesh = mesh
        self.width, self.height = int(width), int(height)
        self.tiles_x = -(-self.width // TILE)
        self.tiles_y = -(-self.height // TILE)
        tri = mesh.triangles.astype(np.int64)
        self.tri = tri
        self.uv = mesh.uvs.astype(np.float64)[tri]
        p = mesh.positions.astype(np.float64)
        self.corners = p[tri]
        g = np.cross(
            self.corners[:, 1] - self.corners[:, 0], self.corners[:, 2] - self.corners[:, 0]
        )
        length = np.linalg.norm(g, axis=1, keepdims=True)
        face = np.divide(g, length, out=np.zeros_like(g), where=length > 0)
        if mesh.normals is not None:
            self.cn = mesh.normals.astype(np.float64)[tri]
        else:
            self.cn = np.repeat(face[:, None, :], 3, axis=1)
        # Bin triangles into the tiles their UV boxes touch; tile boxes in 3D.
        tx = self.uv[..., 0] * self.width
        ty = (1.0 - self.uv[..., 1]) * self.height
        finite = np.isfinite(tx).all(axis=1) & np.isfinite(ty).all(axis=1)
        x0 = np.clip(np.floor(np.where(finite, tx.min(1), 0) / TILE), 0, self.tiles_x - 1).astype(
            np.int64
        )
        x1 = np.clip(np.floor(np.where(finite, tx.max(1), -1) / TILE), -1, self.tiles_x - 1).astype(
            np.int64
        )
        y0 = np.clip(np.floor(np.where(finite, ty.min(1), 0) / TILE), 0, self.tiles_y - 1).astype(
            np.int64
        )
        y1 = np.clip(np.floor(np.where(finite, ty.max(1), -1) / TILE), -1, self.tiles_y - 1).astype(
            np.int64
        )
        nx = np.maximum(x1 - x0 + 1, 0)
        ny = np.maximum(y1 - y0 + 1, 0)
        counts = nx * ny
        t_id = np.repeat(np.arange(tri.shape[0]), counts)
        k = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
        cx = x0[t_id] + k % np.maximum(nx[t_id], 1)
        cy = y0[t_id] + k // np.maximum(nx[t_id], 1)
        tile = cy * self.tiles_x + cx
        order = np.lexsort((t_id, tile))
        self.bin_tri = t_id[order]
        tile_sorted = tile[order]
        n_tiles = self.tiles_x * self.tiles_y
        self.bin_start = np.searchsorted(tile_sorted, np.arange(n_tiles + 1))
        lo = self.corners.min(axis=1)
        hi = self.corners.max(axis=1)
        self.tile_lo = np.full((n_tiles, 3), np.inf)
        self.tile_hi = np.full((n_tiles, 3), -np.inf)
        np.minimum.at(self.tile_lo, tile_sorted, lo[self.bin_tri])
        np.maximum.at(self.tile_hi, tile_sorted, hi[self.bin_tri])
        self._cache: OrderedDict[int, TileTexels] = OrderedDict()
        self._bytes = 0

    def tile_rect(self, t: int) -> tuple[int, int, int, int]:
        ty, tx = divmod(t, self.tiles_x)
        x0, y0 = tx * TILE, ty * TILE
        return x0, y0, min(x0 + TILE, self.width), min(y0 + TILE, self.height)

    def tiles_near(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Tiles whose 3D box comes within ``radius`` of ``center``."""
        d = np.maximum(self.tile_lo - center, 0.0) + np.maximum(center - self.tile_hi, 0.0)
        return np.flatnonzero((d * d).sum(axis=1) <= radius * radius)

    def texels(self, t: int) -> TileTexels:
        if not 0 <= t < self.tiles_x * self.tiles_y:
            raise _bad(f"tile {t} is outside the texture", "tile")
        found = self._cache.get(t)
        if found is not None:
            self._cache.move_to_end(t)
            return found
        tris = self.bin_tri[self.bin_start[t] : self.bin_start[t + 1]]
        rect = self.tile_rect(t)
        if tris.size:
            cov = rasterize_uv(self.uv[tris], self.width, self.height, 1, tris, region=rect)
            row, col = np.divmod(cov.sample, self.width)
            local = (row - rect[1]) * TILE + (col - rect[0])
            b = cov.bary
            position = np.einsum("nk,nkc->nc", b, self.corners[cov.triangle])
            normal = np.einsum("nk,nkc->nc", b, self.cn[cov.triangle])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
            out = TileTexels(local, cov.triangle, b, position, normal)
        else:
            out = TileTexels(
                np.zeros(0, np.int64),
                np.zeros(0, np.int64),
                np.zeros((0, 3)),
                np.zeros((0, 3)),
                np.zeros((0, 3)),
            )
        self._cache[t] = out
        self._bytes += out.nbytes
        while self._bytes > TILE_CACHE_BYTES and len(self._cache) > 1:
            _, old = self._cache.popitem(last=False)
            self._bytes -= old.nbytes
        return out

    def covered_mask(self, tiles: list[int]) -> dict[int, np.ndarray]:
        out = {}
        for t in tiles:
            m = np.zeros(TILE * TILE, dtype=bool)
            m[self.texels(t).local] = True
            out[t] = m.reshape(TILE, TILE)
        return out


_MAPS: dict[tuple, TexelMap] = {}


def texel_map(mesh: Any, width: int, height: int) -> TexelMap:
    """The (cached) texel map of a mesh's UVs at a texture size."""
    key = (id(mesh), width, height)
    found = _MAPS.get(key)
    if found is not None and found.mesh is mesh:
        return found
    if len(_MAPS) > 16:
        _MAPS.clear()
    tm = TexelMap(mesh, width, height)
    _MAPS[key] = tm
    return tm


# --------------------------------------------------------------------------- #
# Brushes
# --------------------------------------------------------------------------- #


@dataclass
class Brush:
    """A brush (see the module docstring); every field is plain data."""

    kind: str = "paint"
    radius: float = 0.1
    strength: float = 1.0
    opacity: float = 1.0
    hardness: float = 0.5
    falloff: str = "smooth"
    spacing: float = 0.15
    color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    blend: str = "normal"
    normal_limit: float = 90.0
    backfaces: bool = False
    symmetry: str = ""
    clone_offset: tuple[float, float] = (0.0, 0.0)
    blur_radius: int = 2

    @classmethod
    def from_value(cls, value: Any, location: str = "brush") -> Brush:
        if isinstance(value, Brush):
            return check_brush(value, location)
        if not isinstance(value, dict):
            raise _bad("a brush is a dict of its settings", location)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(value) - known)
        if unknown:
            raise _bad(f"unknown brush setting(s): {', '.join(unknown)}", location)
        data = dict(value)
        if "color" in data:
            data["color"] = _color(data["color"], f"{location}.color")
        if "clone_offset" in data:
            data["clone_offset"] = _pair(data["clone_offset"], f"{location}.clone_offset")
        return check_brush(cls(**data), location)


def _color(value: Any, location: str) -> tuple[float, float, float, float]:
    from ..materials.painted import _rgba
    from ..rendering.color import parse_color

    if isinstance(value, str):
        try:
            rgb = parse_color(value)
        except Exception:
            raise _bad(f"invalid color {value!r}", location) from None
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    try:
        return _rgba(value, location)
    except Exception:
        raise _bad("a color is '#rrggbb' or 1-4 numbers in [0, 1]", location) from None


def _pair(value: Any, location: str) -> tuple[float, float]:
    try:
        a, b = (float(v) for v in value)
    except (TypeError, ValueError):
        raise _bad("expected two numbers", location) from None
    if not (math.isfinite(a) and math.isfinite(b)):
        raise _bad("expected finite numbers", location)
    return (a, b)


def _array(value: Any, dtype: Any, location: str, what: str) -> np.ndarray:
    """``value`` as an array of ``dtype``; malformed input (ragged, non-numeric, ...) is an
    ``INVALID_ARGUMENT`` error rather than a crash."""
    if isinstance(value, (str, bytes, dict)) or value is None:
        raise _bad(f"{what} must be numbers", location)
    try:
        with np.errstate(all="ignore"):
            out = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError):
        raise _bad(f"{what} must be numbers", location) from None
    if out.dtype == object:
        raise _bad(f"{what} must be numbers", location)
    return out


def _number(value: Any, name: str, lo: float, hi: float, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _bad(f"{name} must be a number", f"{location}.{name}")
    if not lo <= value <= hi:
        raise _bad(f"{name} must be in [{lo}, {hi}]", f"{location}.{name}")
    return float(value)


def check_brush(b: Brush, location: str = "brush") -> Brush:
    from ..materials.painted import BLEND_MODES

    if b.kind not in BRUSH_KINDS:
        raise _bad(f"kind must be one of {', '.join(BRUSH_KINDS)}", f"{location}.kind")
    if b.falloff not in FALLOFFS:
        raise _bad(f"falloff must be one of {', '.join(FALLOFFS)}", f"{location}.falloff")
    if b.blend not in BLEND_MODES:
        raise _bad(f"blend must be one of {', '.join(BLEND_MODES)}", f"{location}.blend")
    if not isinstance(b.backfaces, bool):
        raise _bad("backfaces must be true or false", f"{location}.backfaces")
    if (
        not isinstance(b.symmetry, str)
        or any(c not in "xyz" for c in b.symmetry)
        or len(set(b.symmetry)) != len(b.symmetry)
    ):
        raise _bad("symmetry is a string of distinct axes among x, y, z", f"{location}.symmetry")
    if (
        isinstance(b.blur_radius, bool)
        or not isinstance(b.blur_radius, int)
        or not 1 <= b.blur_radius <= 16
    ):
        raise _bad("blur_radius must be an integer in [1, 16]", f"{location}.blur_radius")
    return Brush(
        b.kind,
        _number(b.radius, "radius", 1e-9, 1e9, location),
        _number(b.strength, "strength", 0.0, 1.0, location),
        _number(b.opacity, "opacity", 0.0, 1.0, location),
        _number(b.hardness, "hardness", 0.0, 1.0, location),
        b.falloff,
        _number(b.spacing, "spacing", 0.01, 10.0, location),
        _color(b.color, f"{location}.color") if not isinstance(b.color, tuple) else b.color,
        b.blend,
        _number(b.normal_limit, "normal_limit", 0.0, 180.0, location),
        b.backfaces,
        b.symmetry,
        tuple(b.clone_offset),
        b.blur_radius,
    )


def falloff(brush: Brush, d: np.ndarray) -> np.ndarray:
    """Weight in [0, 1] at normalized distance ``d`` (0 center, 1 rim)."""
    h = brush.hardness
    if brush.falloff == "constant":
        return (d <= 1.0).astype(np.float64)
    if brush.falloff == "linear":
        inner = np.clip((1.0 - d) / max(1.0 - h, 1e-9), 0.0, 1.0)
        return np.where(d <= 1.0, inner, 0.0)
    if brush.falloff == "gaussian":
        sigma = 0.25 + 0.5 * (1.0 - h) * 0.5
        return np.where(d <= 1.0, np.exp(-0.5 * (d / sigma) ** 2), 0.0)
    # smooth: 1 inside the hard core, smoothstep down to 0 at the rim.
    x = np.clip((d - h) / max(1.0 - h, 1e-9), 0.0, 1.0)
    return np.where(d <= 1.0, 1.0 - x * x * (3.0 - 2.0 * x), 0.0)


# --------------------------------------------------------------------------- #
# Views (projection painting)
# --------------------------------------------------------------------------- #


@dataclass
class PaintView:
    """A camera view for projection painting: the matrices and the depth and object buffers
    of a render of the whole scene from it."""

    view_proj: np.ndarray  # (4, 4) world -> clip
    width: int
    height: int
    depth: np.ndarray  # (H, W) view-space distance along the view direction (inf: none)
    object_index: np.ndarray  # (H, W) render object index (-1: none)
    objects: list  # render object ids by index
    eye: np.ndarray
    forward: np.ndarray
    perspective: bool

    @classmethod
    def render(
        cls, scene: Any, camera: Any = None, width: int = 512, height: int = 512
    ) -> PaintView:
        """Render the depth and object buffers seen by ``camera`` (an object or a name;
        ``None``: the active camera)."""
        from ..rendering.camera import frame_from_matrix

        cam = scene.active_camera if camera is None else _scene_object(scene, camera)
        if cam is None or cam.camera is None:
            raise TexturingError("NO_CAMERA", "projection painting needs a camera", "camera")
        result = scene.render(
            camera=cam, width=width, height=height, samples=1, outputs=["depth", "object_index"]
        )
        frame, _ = frame_from_matrix(np.asarray(cam.world_matrix, dtype=np.float64))
        view = frame.view_matrix()
        proj = cam.camera.projection_matrix(width / height)
        return cls(
            proj @ view,
            width,
            height,
            np.where(np.isfinite(result.depth), result.depth, np.inf),
            result.object_index,
            [o.get("id") for o in result.objects],
            np.asarray(frame.position, dtype=np.float64),
            np.asarray(frame.forward, dtype=np.float64),
            cam.camera.is_perspective,
        )

    def project(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel coordinates (x right, y down, pixel centers at +0.5), view depth, in front."""
        h = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1) @ self.view_proj.T
        w = h[:, 3]
        front = w > 1e-12
        safe = np.where(front, w, 1.0)
        x = (h[:, 0] / safe * 0.5 + 0.5) * self.width
        y = (0.5 - h[:, 1] / safe * 0.5) * self.height
        depth = (world - self.eye) @ self.forward
        return np.stack([x, y], axis=1), depth, front


# --------------------------------------------------------------------------- #
# Strokes
# --------------------------------------------------------------------------- #


@dataclass
class Target:
    """One texture a stroke paints: the painted texture, the layer, and the brush color for
    it (a stroke can lay base color, roughness and metallic at once)."""

    texture: PaintedTexture
    layer: str
    color: tuple[float, float, float, float] | None = None


@dataclass
class StrokeReport:
    dabs: int = 0
    texels: int = 0
    tiles: int = 0
    skipped_dabs: int = 0
    seconds: float = 0.0
    diagnostics: list = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dabs": self.dabs,
            "texels": self.texels,
            "tiles": self.tiles,
            "skipped_dabs": self.skipped_dabs,
            "seconds": self.seconds,
        }


def _to_object(obj: Any, points: np.ndarray) -> tuple[np.ndarray, float]:
    """World points in ``obj``'s own space, and the factor dividing world lengths there
    (the cube root of the transform's volume scale)."""
    m = np.asarray(obj.world_matrix, dtype=np.float64)
    det = abs(float(np.linalg.det(m[:3, :3])))
    if not det > 1e-300:
        raise TexturingError("DEGENERATE_TRANSFORM", f"{obj.name!r} has a degenerate transform")
    inv = np.linalg.inv(m)
    return points @ inv[:3, :3].T + inv[:3, 3], det ** (1.0 / 3.0)


def _dab_positions(
    points: np.ndarray, pressure: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dabs every ``step`` along the polyline (the first point always; pressure interpolated)."""
    if points.shape[0] == 1:
        return points.copy(), pressure.copy()
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return points[:1].copy(), pressure[:1].copy()
    count = int(math.floor(total / step)) + 1
    if count > MAX_DABS:
        raise TexturingError(
            "STROKE_TOO_LONG",
            f"the stroke needs {count:,} dabs (limit {MAX_DABS:,})",
            "points",
            hint="raise spacing or radius, or split the stroke",
        )
    s = np.arange(count) * step
    cum = np.r_[0.0, np.cumsum(seg)]
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, seg.shape[0] - 1)
    f = np.where(seg[idx] > 0, (s - cum[idx]) / np.where(seg[idx] > 0, seg[idx], 1.0), 0.0)
    pos = points[idx] + f[:, None] * (points[idx + 1] - points[idx])
    pr = pressure[idx] + f * (pressure[idx + 1] - pressure[idx])
    return pos, pr


def _mirrors(symmetry: str) -> list[np.ndarray]:
    out = [np.ones(3)]
    for axis in symmetry:
        k = "xyz".index(axis)
        more = []
        for m in out:
            n = m.copy()
            n[k] = -n[k]
            more.append(n)
        out += more
    return out


class _Accumulator:
    """Per-tile float32 working data of one target during a stroke."""

    def __init__(self, target: Target, tm: TexelMap) -> None:
        self.target = target
        self.tm = tm
        self.coverage: dict[int, np.ndarray] = {}  # S per texel (TILE, TILE)
        self.work: dict[int, np.ndarray] = {}  # working layer values (blur/smudge/clone)

    def cover(self, t: int) -> np.ndarray:
        found = self.coverage.get(t)
        if found is None:
            found = self.coverage[t] = np.zeros((TILE, TILE), dtype=np.float32)
        return found


def stroke(
    scene: Any,
    obj: Any,
    targets: Any,
    brush: Any,
    points: Any,
    *,
    space: str = "surface",
    coordinates: str = "world",
    pressure: Any = None,
    view: PaintView | None = None,
    padding: int = 2,
    label: str | None = None,
    should_stop=None,
) -> StrokeReport:
    """Paint one stroke on ``obj``'s painted textures (see the module docstring).

    ``targets``: a painted texture, a :class:`Target`, or a list of them (a texture alone
    paints its top paint layer with the brush color). ``points``: 3D points (``"surface"``;
    world space, or the object's own space with ``coordinates="object"``, and the brush
    radius in the same units, as sculpt strokes), pixel coordinates in ``view``
    (``"view"``) or UV coordinates (``"uv"``). ``pressure``: one value in [0, 1] per point
    (default 1). Symmetry mirrors across the object's own planes.
    """
    import time

    t0 = time.perf_counter()
    b = Brush.from_value(brush)
    if space not in SPACES:
        raise _bad(f"space must be one of {', '.join(SPACES)}", "space")
    mesh = obj.mesh
    if mesh is None or mesh.uvs is None:
        raise TexturingError("NO_UVS", f"{obj.name!r} has no UVs to paint through", "object")
    tlist = _targets(targets, b)
    pts = _array(points, np.float64, "points", "points")
    dims = 2 if space in ("view", "uv") else 3
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] != dims or pts.shape[0] == 0 or not np.all(np.isfinite(pts)):
        raise _bad(f"points must be a finite (n, {dims}) array", "points")
    if pts.shape[0] > MAX_POINTS:
        raise _bad(f"at most {MAX_POINTS:,} points per stroke", "points")
    pr = (
        np.ones(pts.shape[0])
        if pressure is None
        else _array(pressure, np.float64, "pressure", "pressure").reshape(-1)
    )
    if pr.shape[0] != pts.shape[0] or not np.all(np.isfinite(pr)) or pr.min() < 0 or pr.max() > 1:
        raise _bad("pressure needs one value in [0, 1] per point", "pressure")
    if isinstance(padding, bool) or not isinstance(padding, int) or not 0 <= padding <= 16:
        raise _bad("padding must be an integer in [0, 16]", "padding")
    if space == "view" and view is None:
        raise _bad("a view stroke needs a view (PaintView.render(scene, camera))", "view")
    if coordinates not in ("world", "object"):
        raise _bad("coordinates must be 'world' or 'object'", "coordinates")
    if space == "surface" and coordinates == "world":
        pts, scale = _to_object(obj, pts)
        b = dataclasses.replace(b, radius=b.radius / scale)
    report = StrokeReport()
    accs = []
    for target in tlist:
        doc = target.texture
        layer = doc.layer(target.layer)
        if b.kind != "mask" and layer.kind != "paint":
            raise _bad(f"layer {layer.name!r} is a fill layer; paint on a paint layer", "targets")
        accs.append(_Accumulator(target, texel_map(mesh, doc.width, doc.height)))
    step = b.spacing * b.radius
    dabs, dab_pressure = _dab_positions(pts, pr, step)
    report.dabs = int(dabs.shape[0])
    surface = None
    if space == "surface":
        from ..spatial.surface import TriangleSurface

        surface = TriangleSurface(
            mesh.positions.astype(np.float64), mesh.triangles.astype(np.int64)
        )
    world = None
    if space == "view":
        world = np.asarray(obj.world_matrix, dtype=np.float64)
    smudge_load: dict[int, np.ndarray] = {}
    for i in range(dabs.shape[0]):
        if should_stop is not None and should_stop():
            raise TexturingError("CANCELLED", "the stroke was cancelled")
        for m_i, mirror in enumerate(_mirrors(b.symmetry) if space == "surface" else [np.ones(3)]):
            center = dabs[i] * mirror if space == "surface" else dabs[i]
            alpha_scale = b.strength * float(dab_pressure[i])
            for k, acc in enumerate(accs):
                used = _dab(
                    acc,
                    b,
                    space,
                    center,
                    alpha_scale,
                    surface,
                    view,
                    world,
                    obj,
                    smudge_load,
                    (k, m_i),
                )
                if used is None:
                    report.skipped_dabs += 1
    # Lay the strokes and commit (one history step).
    edits: dict[int, tuple[PaintedTexture, dict]] = {}
    for acc in accs:
        _finish(acc, b, padding, edits)
    touched = 0
    texels = 0
    with scene.transaction(label or f"Paint stroke ({b.kind}) on {obj.name!r}"):
        for doc, by_layer in edits.values():
            tiles = by_layer.get("tiles", {})
            masks = by_layer.get("masks", {})
            if tiles:
                doc.replace_tiles(tiles, f"Paint {doc.name!r}")
            if masks:
                doc.replace_masks(masks, f"Paint mask of {doc.name!r}")
            touched += sum(len(v) for v in tiles.values()) + sum(len(v) for v in masks.values())
    for acc in accs:
        texels += int(sum(int(np.count_nonzero(c)) for c in acc.coverage.values()))
    report.tiles = touched
    report.texels = texels
    report.seconds = time.perf_counter() - t0
    return report


def _targets(targets: Any, b: Brush) -> list[Target]:
    items = targets if isinstance(targets, (list, tuple)) else [targets]
    out = []
    for item in items:
        if isinstance(item, Target):
            t = item
        elif isinstance(item, PaintedTexture):
            paint = [layer for layer in item.layers if layer.kind == "paint"]
            if not paint and b.kind != "mask":
                raise _bad(f"texture {item.name!r} has no paint layer", "targets")
            layer = (paint[-1] if paint else item.layers[-1]).id
            t = Target(item, layer, None)
        elif isinstance(item, dict):
            tex = item.get("texture")
            if not isinstance(tex, PaintedTexture):
                raise _bad("a target names a painted texture", "targets")
            color = item.get("color")
            layer = item.get("layer")
            if layer is None:
                if not tex.layers:
                    raise _bad(f"texture {tex.name!r} has no layer", "targets")
                layer = tex.layers[-1].id
            t = Target(tex, layer, None if color is None else _color(color, "targets.color"))
        else:
            raise _bad("targets are painted textures (or {texture, layer, color})", "targets")
        if not t.texture.layers:
            raise _bad(f"texture {t.texture.name!r} has no layer", "targets")
        t.texture.layer(t.layer)  # exists
        out.append(t)
    if not out:
        raise _bad("a stroke needs a target texture", "targets")
    return out


def _dab(acc, b, space, center, alpha_scale, surface, view, world, obj, smudge_load, key):
    """Accumulate one dab into ``acc``; returns the number of texels reached or None."""
    tm = acc.tm
    if space == "surface":
        near = surface.nearest(center[None, :], np.inf)
        if near.triangles[0] < 0:
            return None
        hit = near.points[0]
        tri = int(near.triangles[0])
        n_brush = np.einsum("k,kc->c", near.bary[0], tm.cn[tri])
        n_brush /= max(np.linalg.norm(n_brush), 1e-300)
        tiles = tm.tiles_near(hit, b.radius)
        limit = math.cos(math.radians(b.normal_limit)) if b.normal_limit < 180 else -2.0
        reached = 0
        for t in tiles.tolist():
            tx = tm.texels(t)
            if tx.local.size == 0:
                continue
            d = np.linalg.norm(tx.position - hit, axis=1) / b.radius
            w = falloff(b, d)
            w = np.where((tx.normal @ n_brush) >= limit, w, 0.0)
            if not np.any(w > 0):
                continue
            reached += int(np.count_nonzero(w > 0))
            _apply_weights(acc, b, t, tx, w * alpha_scale, smudge_load, key)
        return reached
    if space == "uv":
        u, v = center
        cx, cy = u * tm.width, (1.0 - v) * tm.height
        r = b.radius
        x0, x1 = int(math.floor(cx - r)), int(math.ceil(cx + r))
        y0, y1 = int(math.floor(cy - r)), int(math.ceil(cy + r))
        reached = 0
        for t in _tiles_in(tm, x0, y0, x1 + 1, y1 + 1):
            tx = tm.texels(t)
            rect = tm.tile_rect(t)
            # 2D painting reaches every texel (covered or not) of the tile within the disc.
            ly, lx = np.divmod(np.arange(TILE * TILE), TILE)
            gx, gy = rect[0] + lx + 0.5, rect[1] + ly + 0.5
            inside = (gx < tm.width) & (gy < tm.height)
            d = np.hypot(gx - cx, gy - cy) / r
            w = np.where(inside, falloff(b, d), 0.0)
            if not np.any(w > 0):
                continue
            reached += int(np.count_nonzero(w > 0))
            full = _AllTexels(np.flatnonzero(w > 0))
            _apply_weights(acc, b, t, full, (w * alpha_scale)[full.local], smudge_load, key)
            del tx
        return reached
    # View: screen-space disc, visibility from the view's buffers.
    px, py = center
    r = b.radius
    reached = 0
    target_index = _object_index(view, obj)
    lin = world[:3, :3]
    normal_matrix = np.linalg.inv(lin).T
    # Candidate tiles: those whose 3D box projects near the dab (all tiles; cheap test).
    for t in range(tm.tiles_x * tm.tiles_y):
        if not np.all(np.isfinite(tm.tile_lo[t])):
            continue
        lo, hi = tm.tile_lo[t], tm.tile_hi[t]
        corners = np.array(
            [
                [(lo, hi)[i & 1][0], (lo, hi)[(i >> 1) & 1][1], (lo, hi)[(i >> 2) & 1][2]]
                for i in range(8)
            ]
        )
        xy, _, front = view.project(corners @ lin.T + world[:3, 3])
        if not front.any():
            continue  # wholly behind the camera
        if front.all() and (
            xy[:, 0].max() < px - r
            or xy[:, 0].min() > px + r
            or xy[:, 1].max() < py - r
            or xy[:, 1].min() > py + r
        ):
            continue  # its box projects away from the dab
        tx = tm.texels(t)
        if tx.local.size == 0:
            continue
        pw = tx.position @ lin.T + world[:3, 3]
        xy, depth, front = view.project(pw)
        d = np.hypot(xy[:, 0] - px, xy[:, 1] - py) / r
        w = falloff(b, d) * front
        if not np.any(w > 0):
            continue
        ix = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, view.width - 1)
        iy = np.clip(np.floor(xy[:, 1]).astype(np.int64), 0, view.height - 1)
        onscreen = (
            (xy[:, 0] >= 0) & (xy[:, 0] < view.width) & (xy[:, 1] >= 0) & (xy[:, 1] < view.height)
        )
        if not b.backfaces:
            nw = tx.normal @ normal_matrix.T
            to_eye = (view.eye - pw) if view.perspective else -view.forward[None, :]
            facing = (nw * to_eye).sum(axis=1) > 0
            scene_depth = view.depth[iy, ix]
            tol = 1e-3 * np.maximum(np.abs(depth), 1e-3) + 1e-6
            visible = (
                onscreen
                & facing
                & (view.object_index[iy, ix] == target_index)
                & (depth <= scene_depth + tol)
            )
            w = np.where(visible, w, 0.0)
        else:
            w = np.where(onscreen, w, 0.0)
        if not np.any(w > 0):
            continue
        reached += int(np.count_nonzero(w > 0))
        _apply_weights(acc, b, t, tx, w * alpha_scale, smudge_load, key)
    return reached if reached else None


@dataclass
class _AllTexels:
    local: np.ndarray


def _tiles_in(tm: TexelMap, x0: int, y0: int, x1: int, y1: int) -> list[int]:
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(tm.width, x1), min(tm.height, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    return [
        j * tm.tiles_x + i
        for j in range(y0 // TILE, (y1 - 1) // TILE + 1)
        for i in range(x0 // TILE, (x1 - 1) // TILE + 1)
    ]


def _scene_object(scene: Any, ref: Any):
    from ..scene import SceneObject

    if isinstance(ref, SceneObject):
        return ref
    from ..errors import ObjectNotFoundError

    try:
        return scene.get(ref)
    except (ObjectNotFoundError, TypeError):
        try:
            return scene.get_by_name(ref)
        except (ObjectNotFoundError, TypeError):
            raise TexturingError("OBJECT_NOT_FOUND", f"no object {ref!r}", "object") from None


def _object_index(view: PaintView, obj: Any) -> int:
    try:
        return view.objects.index(obj.id)
    except ValueError:
        return -2


def _layer_tile(doc: PaintedTexture, layer_ref: str, t: int) -> np.ndarray:
    layer = doc.layer(layer_ref)
    tile = (layer.tiles or {}).get(t)
    if tile is None:
        return np.zeros((TILE, TILE, 4), dtype=np.float32)
    return tile.astype(np.float32) * np.float32(1.0 / FULL)


def _apply_weights(acc, b, t, tx, w, smudge_load, key) -> None:
    """Add dab weights ``w`` (per texel of ``tx``) to the accumulator of tile ``t``."""
    w = np.clip(w, 0.0, 1.0).astype(np.float32)
    doc = acc.target.texture
    if b.kind in ("paint", "erase", "mask"):
        cov = acc.cover(t).reshape(-1)
        cur = cov[tx.local]
        cov[tx.local] = cur + w * (1.0 - cur)  # build-up within the stroke
        return
    # Kinds that read the layer as they go: blur, smudge, clone.
    work = acc.work.get(t)
    if work is None:
        work = acc.work[t] = _layer_tile(doc, acc.target.layer, t)
    flat = work.reshape(-1, 4)
    cov = acc.cover(t).reshape(-1)
    cov[tx.local] = np.maximum(cov[tx.local], w)
    if b.kind == "blur":
        src = _neighbourhood_mean(acc, t, b.blur_radius)
        flat[tx.local] += (src[tx.local] - flat[tx.local]) * w[:, None]
    elif b.kind == "smudge":
        load = smudge_load.get(key)
        here = (flat[tx.local] * w[:, None]).sum(axis=0) / max(float(w.sum()), 1e-12)
        if load is None:
            smudge_load[key] = here
            return
        flat[tx.local] += (load[None, :] - flat[tx.local]) * w[:, None]
        smudge_load[key] = load * 0.7 + here * 0.3
    else:  # clone: copy from the layer itself at a texel offset (or from a source texture)
        dx, dy = b.clone_offset
        rect = acc.tm.tile_rect(t)
        ly, lx = np.divmod(tx.local, TILE)
        sx = np.clip(np.round(rect[0] + lx + dx).astype(np.int64), 0, doc.width - 1)
        sy = np.clip(np.round(rect[1] + ly + dy).astype(np.int64), 0, doc.height - 1)
        source = getattr(acc.target, "source", None) or doc
        values = _read_texels(source, acc.target.layer if source is doc else None, sx, sy)
        flat[tx.local] += (values - flat[tx.local]) * w[:, None]


def _read_texels(
    doc: PaintedTexture, layer: str | None, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    out = np.zeros((x.shape[0], 4), dtype=np.float32)
    tiles = (y // TILE) * doc.tiles_x + (x // TILE)
    for t in np.unique(tiles).tolist():
        sel = tiles == t
        tile = (
            _layer_tile(doc, layer, t)
            if layer is not None
            else doc.composite_tile(t).astype(np.float32) / FULL
        )
        x0, y0 = (t % doc.tiles_x) * TILE, (t // doc.tiles_x) * TILE
        out[sel] = tile[y[sel] - y0, x[sel] - x0]
    return out


def _neighbourhood_mean(acc, t: int, radius: int) -> np.ndarray:
    """The layer's box average over ``(2 r + 1)^2`` texels, for the texels of tile ``t``
    (neighbouring tiles read as the layer currently is)."""
    doc = acc.target.texture
    x0, y0, x1, y1 = acc.tm.tile_rect(t)
    gx0, gy0 = max(0, x0 - radius), max(0, y0 - radius)
    gx1, gy1 = min(doc.width, x0 + TILE + radius), min(doc.height, y0 + TILE + radius)
    region = np.zeros((gy1 - gy0, gx1 - gx0, 4), dtype=np.float32)
    for n in _tiles_in(acc.tm, gx0, gy0, gx1, gy1):
        rect = acc.tm.tile_rect(n)
        tile = acc.work.get(n)
        if tile is None:
            tile = _layer_tile(doc, acc.target.layer, n)
        ax0, ay0 = max(rect[0], gx0), max(rect[1], gy0)
        ax1, ay1 = min(rect[2], gx1), min(rect[3], gy1)
        region[ay0 - gy0 : ay1 - gy0, ax0 - gx0 : ax1 - gx0] = tile[
            ay0 - rect[1] : ay1 - rect[1], ax0 - rect[0] : ax1 - rect[0]
        ]
    k = 2 * radius + 1
    pad = np.pad(region, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    csum = pad.cumsum(axis=0).cumsum(axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0), (0, 0)))
    box = (csum[k:, k:] - csum[:-k, k:] - csum[k:, :-k] + csum[:-k, :-k]) / (k * k)
    out = np.zeros((TILE, TILE, 4), dtype=np.float32)
    oy, ox = y0 - gy0, x0 - gx0
    h, w = y1 - y0, x1 - x0
    out[:h, :w] = box[oy : oy + h, ox : ox + w]
    return out.reshape(-1, 4)


def _finish(acc: _Accumulator, b: Brush, padding: int, edits: dict) -> None:
    """Grow the coverage into padding texels, lay the stroke over the layer as it was, and
    record the new tiles in ``edits``."""
    doc = acc.target.texture
    if not acc.coverage:
        return
    tiles = sorted(acc.coverage)
    if padding:
        _pad_coverage(acc, tiles, padding)
        tiles = sorted(acc.coverage)
    color = acc.target.color if acc.target.color is not None else b.color
    entry = edits.setdefault(id(doc), (doc, {}))[1]
    layer = doc.layer(acc.target.layer)
    for t in tiles:
        s = acc.coverage[t] * np.float32(b.opacity)
        if not np.any(s > 0):
            continue
        if b.kind == "mask":
            old = (layer.mask or {}).get(t)
            m = (
                np.full((TILE, TILE), 1.0, np.float32)
                if old is None
                else old.astype(np.float32) / FULL
            )
            # Paint the mask toward the brush value (its first channel).
            m = m + (np.float32(color[0]) - m) * s
            entry.setdefault("masks", {}).setdefault(layer.id, {})[t] = np.round(
                np.clip(m, 0, 1) * FULL
            ).astype(np.uint16)
            continue
        base = _layer_tile(doc, acc.target.layer, t)
        if b.kind == "paint":
            rgb = np.broadcast_to(np.asarray(color[:3], np.float32), (TILE, TILE, 3))
            out = composite_over(base, rgb, s * np.float32(color[3]), b.blend)
        elif b.kind == "erase":
            out = base.copy()
            out[..., 3] = base[..., 3] * (1.0 - s)
        else:
            work = acc.work.get(t, base)
            out = base + (work - base) * s[..., None]
        new = np.round(np.clip(out, 0.0, 1.0) * FULL).astype(np.uint16)
        entry.setdefault("tiles", {}).setdefault(layer.id, {})[t] = new


def _pad_coverage(acc: _Accumulator, tiles: list[int], padding: int) -> None:
    """Grow each tile's stroke coverage into the uncovered (padding) texels around the
    covered ones, ``padding`` rings, within the tile and its neighbours."""
    tm = acc.tm
    # Work on the bounding region of the touched tiles plus a margin of one tile.
    xs = [t % tm.tiles_x for t in tiles]
    ys = [t // tm.tiles_x for t in tiles]
    tx0, tx1 = max(0, min(xs) - 1), min(tm.tiles_x - 1, max(xs) + 1)
    ty0, ty1 = max(0, min(ys) - 1), min(tm.tiles_y - 1, max(ys) + 1)
    h = (ty1 - ty0 + 1) * TILE
    w = (tx1 - tx0 + 1) * TILE
    cov = np.zeros((h, w), np.float32)
    covered = np.zeros((h, w), bool)
    work = {}
    for j in range(ty0, ty1 + 1):
        for i in range(tx0, tx1 + 1):
            t = j * tm.tiles_x + i
            sl = (
                slice((j - ty0) * TILE, (j - ty0 + 1) * TILE),
                slice((i - tx0) * TILE, (i - tx0 + 1) * TILE),
            )
            if t in acc.coverage:
                cov[sl] = acc.coverage[t]
            m = np.zeros(TILE * TILE, bool)
            m[tm.texels(t).local] = True
            covered[sl] = m.reshape(TILE, TILE)
            work[t] = sl
    grown = covered.copy()
    for _ in range(padding):
        total = np.zeros_like(cov)
        count = np.zeros_like(cov)
        for dy, dx in SHIFTS:
            ys_ = slice(max(dy, 0), h + min(dy, 0))
            yd = slice(max(-dy, 0), h + min(-dy, 0))
            xs_ = slice(max(dx, 0), w + min(dx, 0))
            xd = slice(max(-dx, 0), w + min(-dx, 0))
            src = grown[ys_, xs_]
            total[yd, xd] += np.where(src, cov[ys_, xs_], 0.0)
            count[yd, xd] += src
        new = ~grown & (count > 0)
        if not new.any():
            break
        cov[new] = total[new] / count[new]
        grown |= new
    for t, sl in work.items():
        block = cov[sl]
        if np.any(block > 0):
            acc.coverage[t] = block.copy()


# --------------------------------------------------------------------------- #
# Fills and gradients
# --------------------------------------------------------------------------- #


def fill(
    scene: Any,
    obj: Any,
    target: Any,
    value: Any,
    *,
    where: Any = None,
    opacity: float = 1.0,
    blend_mode: str = "normal",
    padding: int = 2,
    label: str | None = None,
) -> dict[str, Any]:
    """Fill a paint layer with ``value``: everywhere, or where ``where`` says --
    ``{"islands": [...]}`` (UV islands of the object's mesh), ``{"groups": [...]}`` (face
    groups), or ``{"triangles": [...]}`` -- with ``padding`` rings around the filled
    texels. One history step."""
    from ..uv.islands import island_labels
    from ..uv.topology import Topology

    t = _targets(target, Brush())[0]
    doc = t.texture
    color = _color(value, "value")
    opacity = _number(opacity, "opacity", 0.0, 1.0, "fill")
    mesh = obj.mesh
    tm = texel_map(mesh, doc.width, doc.height)
    tri_sel = None
    if where is not None:
        if not isinstance(where, dict) or len(where) != 1:
            raise _bad("where is {islands|groups|triangles: [...]}", "where")
        ((key, ids),) = where.items()
        if not isinstance(ids, list):
            raise _bad(f"where.{key} is a list of integers", f"where.{key}")
        if any(isinstance(i, bool) or not isinstance(i, int) or abs(i) >= 2**62 for i in ids):
            raise _bad(f"where.{key} is a list of integers", f"where.{key}")
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if key == "islands":
            labels, _ = island_labels(Topology.from_mesh(mesh))
            tri_sel = np.isin(labels, ids)
        elif key == "groups":
            groups = (
                mesh.groups if mesh.groups is not None else np.zeros(mesh.triangle_count, np.int64)
            )
            tri_sel = np.isin(groups.astype(np.int64), ids)
        elif key == "triangles":
            tri_sel = np.zeros(mesh.triangle_count, bool)
            tri_sel[ids[(ids >= 0) & (ids < mesh.triangle_count)]] = True
        else:
            raise _bad("where is {islands|groups|triangles: [...]}", "where")
    acc = _Accumulator(t, tm)
    for tile in range(tm.tiles_x * tm.tiles_y):
        tx = tm.texels(tile)
        if tx.local.size == 0:
            continue
        w = (
            np.ones(tx.local.size, np.float32)
            if tri_sel is None
            else tri_sel[tx.triangle].astype(np.float32)
        )
        if np.any(w > 0):
            cov = acc.cover(tile).reshape(-1)
            cov[tx.local] = np.maximum(cov[tx.local], w)
    b = Brush(kind="paint", opacity=opacity, color=color, blend=blend_mode)
    edits: dict = {}
    _finish(acc, b, padding, edits)
    with scene.transaction(label or f"Fill {doc.name!r}"):
        for d, by_layer in edits.values():
            if by_layer.get("tiles"):
                d.replace_tiles(by_layer["tiles"], f"Fill {d.name!r}")
    return {"tiles": sum(len(v) for _, e in edits.values() for v in e.get("tiles", {}).values())}


def gradient(
    scene: Any,
    obj: Any,
    target: Any,
    start: Any,
    end: Any,
    colors: Any,
    *,
    coordinates: str = "world",
    opacity: float = 1.0,
    padding: int = 2,
    label: str | None = None,
) -> dict[str, Any]:
    """A linear gradient over the object's surface from point ``start`` to ``end`` (world
    space, or the object's own with ``coordinates="object"``): texels get
    ``mix(colors[0], colors[1], t)`` with ``t`` the clamped projection of their surface
    point on the segment. One history step."""
    t = _targets(target, Brush())[0]
    doc = t.texture
    a = _array(start, np.float64, "start", "start").reshape(-1)
    e = _array(end, np.float64, "end", "end").reshape(-1)
    if a.shape != (3,) or e.shape != (3,):
        raise _bad("start and end are 3D points", "start")
    opacity = _number(opacity, "opacity", 0.0, 1.0, "gradient")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(e))) or np.allclose(a, e):
        raise _bad("start and end must be distinct finite points", "start")
    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise _bad("colors must be two colors", "colors")
    c0 = np.asarray(_color(colors[0], "colors[0]"), np.float32)
    c1 = np.asarray(_color(colors[1], "colors[1]"), np.float32)
    if coordinates not in ("world", "object"):
        raise _bad("coordinates must be 'world' or 'object'", "coordinates")
    if coordinates == "world":
        (a, e), _ = _to_object(obj, np.stack([a, e]))
    tm = texel_map(obj.mesh, doc.width, doc.height)
    d = e - a
    edits: dict = {}
    layer = doc.layer(t.layer)
    entry = edits.setdefault(id(doc), (doc, {}))[1]
    for tile in range(tm.tiles_x * tm.tiles_y):
        tx = tm.texels(tile)
        if tx.local.size == 0:
            continue
        f = np.clip(((tx.position - a) @ d) / float(d @ d), 0.0, 1.0).astype(np.float32)
        col = c0[None, :] + (c1 - c0)[None, :] * f[:, None]
        base = _layer_tile(doc, layer.id, tile).reshape(-1, 4)
        over = composite_over(
            base[tx.local][None],
            col[None, :, :3],
            (col[:, 3] * np.float32(opacity))[None],
            "normal",
        )[0]
        out = base.copy()
        out[tx.local] = over
        entry.setdefault("tiles", {}).setdefault(layer.id, {})[tile] = np.round(
            np.clip(out.reshape(TILE, TILE, 4), 0, 1) * FULL
        ).astype(np.uint16)
    with scene.transaction(label or f"Gradient on {doc.name!r}"):
        if entry.get("tiles"):
            doc.replace_tiles(entry["tiles"], f"Gradient on {doc.name!r}")
    del padding
    return {"tiles": len(entry.get("tiles", {}).get(layer.id, {}))}
