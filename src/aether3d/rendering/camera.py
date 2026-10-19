"""Cameras: validated projection parameters and the exact camera mathematics.

Conventions (see ``docs/RENDERING.md``):

* A camera object **looks along its local -Y axis** with its local **+Z axis
  up** -- the same "front faces -Y, +Z up" convention as every Aether3D asset,
  so ``obj.look_at(target)`` aims cameras, lights and models alike. The image's
  right-hand side is therefore the camera's local -X axis.
* The camera frame is derived from the object's world matrix: forward is the
  world direction of local -Y, up is the world direction of local +Z made
  orthogonal to forward (Gram-Schmidt), and right = forward x up. Scale,
  shear and mirroring in the world matrix therefore never distort or mirror
  the image (they are reported as ``CAMERA_SCALE_IGNORED``).
* View space is the usual right-handed one: x right, y up, the camera looking
  down -z. Clip space uses NDC x, y in ``[-1, 1]`` (y up) and depth z in
  ``[0, 1]`` (near plane at 0, far plane at 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..diagnostics import INFO, Diagnostic
from ..errors import RenderError
from ..history import recording

PROJECTIONS = ("perspective", "orthographic")
FOV_AXES = ("vertical", "horizontal", "auto")

MIN_FOV, MAX_FOV = 0.01, 179.0
MAX_DEPTH_RATIO = 1e9
"""Largest allowed ``far / near`` for perspective cameras (depth precision)."""
MIN_ASPECT, MAX_ASPECT = 1e-3, 1e3

_FIELDS = (
    "projection",
    "fov",
    "fov_axis",
    "ortho_scale",
    "near",
    "far",
    "aspect_ratio",
    "sensor",
    "shift_x",
    "shift_y",
    "fstop",
    "focus_distance",
    "focus_object",
    "aperture_blades",
    "aperture_rotation",
    "aperture_ratio",
)
LENS_DEFAULTS: dict[str, Any] = {
    "sensor": 36.0,
    "shift_x": 0.0,
    "shift_y": 0.0,
    "fstop": 0.0,
    "focus_distance": 10.0,
    "focus_object": None,
    "aperture_blades": 0,
    "aperture_rotation": 0.0,
    "aperture_ratio": 1.0,
}
"""The lens fields of 0.20.0 and their defaults (a pinhole: every render as before)."""
MAX_SHIFT = 10.0
MAX_BLADES = 16


def _number(value: Any, field: str, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise RenderError("INVALID_CAMERA", f"{field} must be a number, got {value!r}", location)
    result = float(value)
    if not math.isfinite(result):
        raise RenderError("INVALID_CAMERA", f"{field} must be finite, got {value!r}", location)
    return result


class Camera:
    """The camera component of a scene object (``obj.camera``).

    * ``projection`` -- ``"perspective"`` or ``"orthographic"``.
    * ``fov`` -- perspective field of view in degrees, measured along ``fov_axis``.
    * ``fov_axis`` -- ``"vertical"`` (default), ``"horizontal"``, or ``"auto"``
      (the larger image dimension).
    * ``ortho_scale`` -- orthographic view size in world units along ``fov_axis``.
    * ``near``, ``far`` -- clipping distances along the view direction
      (perspective: ``0 < near < far``, ``far / near <= 1e9``; orthographic:
      ``0 <= near < far``).
    * ``aspect_ratio`` -- the frame shape (width / height) this camera is set up
      for, or ``None``. It provides the default image height and is used when
      framing; renders always use square pixels, so an image of another shape
      shows more or less of the scene (reported as ``ASPECT_MISMATCH``).

    The lens (0.20.0; defaults: a pinhole, as before):

    * ``sensor`` -- the sensor's size in millimetres along ``fov_axis`` (36: a full-frame
      sensor's width); with ``fov`` it gives the focal length, ``focal_length =
      sensor / (2 tan(fov / 2))`` (``configure(focal_length=50)`` sets ``fov`` from it).
    * ``shift_x``, ``shift_y`` -- lens shift, in units of the image's larger dimension
      (as Blender): the view moves right/up without turning, so verticals stay vertical.
    * ``fstop`` -- the f-number of the aperture, ``0`` for a pinhole (no depth of field).
      The aperture's diameter is ``focal_length / fstop``.
    * ``focus_distance`` -- metres along the view axis to the plane in focus;
      ``focus_object`` -- an object id whose origin sets it instead, every frame.
    * ``aperture_blades`` (``0``: round, 3..16: a polygon inscribed in the round
      aperture, a corner up), ``aperture_rotation`` (degrees), ``aperture_ratio``
      (anamorphic: the aperture's width is divided by it).

    Depth of field is a thin lens at the eye with the field of view unchanged (Blender's
    model, measured): a point at distance ``d`` blurs into a disk of
    ``(focal_length / fstop) |1/d - 1/focus|`` (in units of the image plane at distance 1).

    Setters validate immediately; change several related values at once with
    :meth:`configure` (e.g. ``camera.configure(near=500, far=5000)``).
    """

    __slots__ = (*(f"_{name}" for name in _FIELDS), "_owner")

    def __init__(
        self,
        projection: str = "perspective",
        fov: float = 40.0,
        fov_axis: str = "vertical",
        ortho_scale: float = 10.0,
        near: float = 0.1,
        far: float = 1000.0,
        aspect_ratio: float | None = None,
        **lens: Any,
    ) -> None:
        unknown = sorted(set(lens) - set(LENS_DEFAULTS) - {"focal_length"})
        if unknown:
            raise RenderError(
                "INVALID_CAMERA",
                f"unknown camera parameter(s): {', '.join(unknown)}",
                "camera",
                hint=f"parameters: {', '.join(_FIELDS)}, focal_length",
            )
        values = {
            "projection": projection,
            "fov": fov,
            "fov_axis": fov_axis,
            "ortho_scale": ortho_scale,
            "near": near,
            "far": far,
            "aspect_ratio": aspect_ratio,
            **LENS_DEFAULTS,
            **lens,
        }
        values = _validate(_focal(values, "camera"), "camera")
        for name in _FIELDS:
            setattr(self, f"_{name}", values[name])
        self._owner = None  # the object this belongs to (records its changes)

    # -- parameters --------------------------------------------------------

    def configure(self, **changes: Any) -> Camera:
        """Change several parameters at once; validated together (all or nothing)."""
        unknown = sorted(set(changes) - set(_FIELDS) - {"focal_length"})
        if unknown:
            raise RenderError(
                "INVALID_CAMERA",
                f"unknown camera parameter(s): {', '.join(unknown)}",
                "camera",
                hint=f"parameters: {', '.join(_FIELDS)}, focal_length",
            )
        values = _validate(_focal({**self.to_dict(), **changes}, "camera"), "camera")
        with recording(self, changes):
            for name in _FIELDS:
                setattr(self, f"_{name}", values[name])
        return self

    projection = property(
        lambda self: self._projection, lambda self, v: self.configure(projection=v)
    )
    fov = property(lambda self: self._fov, lambda self, v: self.configure(fov=v))
    fov_axis = property(lambda self: self._fov_axis, lambda self, v: self.configure(fov_axis=v))
    ortho_scale = property(
        lambda self: self._ortho_scale, lambda self, v: self.configure(ortho_scale=v)
    )
    near = property(lambda self: self._near, lambda self, v: self.configure(near=v))
    far = property(lambda self: self._far, lambda self, v: self.configure(far=v))
    aspect_ratio = property(
        lambda self: self._aspect_ratio, lambda self, v: self.configure(aspect_ratio=v)
    )
    sensor = property(lambda self: self._sensor, lambda self, v: self.configure(sensor=v))
    shift_x = property(lambda self: self._shift_x, lambda self, v: self.configure(shift_x=v))
    shift_y = property(lambda self: self._shift_y, lambda self, v: self.configure(shift_y=v))
    fstop = property(lambda self: self._fstop, lambda self, v: self.configure(fstop=v))
    focus_distance = property(
        lambda self: self._focus_distance, lambda self, v: self.configure(focus_distance=v)
    )
    focus_object = property(
        lambda self: self._focus_object, lambda self, v: self.configure(focus_object=v)
    )
    aperture_blades = property(
        lambda self: self._aperture_blades, lambda self, v: self.configure(aperture_blades=v)
    )
    aperture_rotation = property(
        lambda self: self._aperture_rotation,
        lambda self, v: self.configure(aperture_rotation=v),
    )
    aperture_ratio = property(
        lambda self: self._aperture_ratio, lambda self, v: self.configure(aperture_ratio=v)
    )

    @property
    def focal_length(self) -> float | None:
        """Millimetres: ``sensor / (2 tan(fov / 2))`` (``None`` for orthographic cameras)."""
        if not self.is_perspective:
            return None
        return self._sensor / (2.0 * math.tan(math.radians(self._fov) / 2.0))

    @focal_length.setter
    def focal_length(self, value: float) -> None:
        self.configure(focal_length=value)

    @property
    def has_depth_of_field(self) -> bool:
        return self.is_perspective and self._fstop > 0.0

    def aperture_radius(self) -> float:
        """The aperture's radius in metres (0: a pinhole)."""
        if not self.has_depth_of_field:
            return 0.0
        return float(self.focal_length) / 1000.0 / (2.0 * self._fstop)

    def ndc_offset(self, aspect: float) -> tuple[float, float]:
        """Where the lens shift moves the view centre, in NDC (``x_ndc = x / hx - ox``)."""
        aspect = _aspect(aspect)
        return (
            2.0 * self._shift_x * max(1.0, 1.0 / aspect),
            2.0 * self._shift_y * max(1.0, aspect),
        )

    @property
    def is_perspective(self) -> bool:
        return self._projection == "perspective"

    def copy(self) -> Camera:
        return Camera(**self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, f"_{name}") for name in _FIELDS}

    @classmethod
    def from_dict(cls, data: Any, location: str = "camera") -> Camera:
        """Strict construction from a JSON object (unknown or missing keys are errors)."""
        if not isinstance(data, dict):
            raise RenderError("INVALID_CAMERA", "expected an object", location)
        # (the lens fields of 0.20.0 default when missing: documents written before them)
        missing = [f for f in _FIELDS if f not in data and f not in LENS_DEFAULTS]
        unknown = sorted(set(data) - set(_FIELDS))
        if missing or unknown:
            what = f"missing {', '.join(missing)}" if missing else f"unknown {', '.join(unknown)}"
            raise RenderError("INVALID_CAMERA", f"camera keys: {what}", location)
        camera = cls.__new__(cls)
        values = _validate({**LENS_DEFAULTS, **data}, location)
        for name in _FIELDS:
            setattr(camera, f"_{name}", values[name])
        return camera

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Camera):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        if self.is_perspective:
            shape = f"fov={self._fov:g} ({self._fov_axis})"
        else:
            shape = f"ortho_scale={self._ortho_scale:g} ({self._fov_axis})"
        return f"Camera({self._projection}, {shape}, near={self._near:g}, far={self._far:g})"

    # -- mathematics ---------------------------------------------------------

    def fov_applies_to_vertical(self, aspect: float) -> bool:
        if self._fov_axis == "auto":
            return aspect < 1.0
        return self._fov_axis == "vertical"

    def half_extents(self, aspect: float) -> tuple[float, float]:
        """``(horizontal, vertical)`` half extents of the view for an image aspect (w / h).

        Perspective: tangents of the half angles. Orthographic: half sizes in
        world units.
        """
        aspect = _aspect(aspect)
        if self.is_perspective:
            half = math.tan(math.radians(self._fov) / 2.0)
        else:
            half = self._ortho_scale / 2.0
        if self.fov_applies_to_vertical(aspect):
            return half * aspect, half
        return half, half / aspect

    def projection_matrix(self, aspect: float) -> np.ndarray:
        """4x4 view-to-clip matrix for an image aspect ratio (width / height).

        Perspective (``tx, ty`` = tangents of the half angles)::

            [1/tx, 0,    0,          0          ]
            [0,    1/ty, 0,          0          ]
            [0,    0,    f/(n - f),  n f/(n - f)]
            [0,    0,    -1,         0          ]

        Orthographic (``hx, hy`` = half extents)::

            [1/hx, 0,    0,          0          ]
            [0,    1/hy, 0,          0          ]
            [0,    0,    -1/(f - n), -n/(f - n) ]
            [0,    0,    0,          1          ]

        Both map view depth ``-n`` to NDC z 0 and ``-f`` to 1.
        """
        hx, hy = self.half_extents(aspect)
        ox, oy = self.ndc_offset(aspect)
        n, f = self._near, self._far
        m = np.zeros((4, 4), dtype=np.float64)
        m[0, 0] = 1.0 / hx
        m[1, 1] = 1.0 / hy
        if self.is_perspective:
            m[2, 2] = f / (n - f)
            m[2, 3] = n * f / (n - f)
            m[3, 2] = -1.0
            if ox or oy:  # a shifted lens: an off-centre frustum (x_ndc = x / (-z hx) - ox)
                m[0, 2] = ox
                m[1, 2] = oy
        else:
            m[2, 2] = -1.0 / (f - n)
            m[2, 3] = -n / (f - n)
            m[3, 3] = 1.0
            if ox or oy:
                m[0, 3] = -ox
                m[1, 3] = -oy
        return m


def aperture_points(
    u1: np.ndarray, u2: np.ndarray, blades: int, rotation: float, ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points of the unit aperture (radius 1, before the camera's radius) for uniform
    numbers ``u1, u2``: round (Shirley and Chiu's concentric mapping) or a regular polygon
    of ``blades`` corners inscribed in it, a corner up (rotation 0), uniform by area; then
    turned by ``rotation`` degrees and narrowed by ``ratio`` (x / ratio)."""
    if blades >= 3:
        k = np.minimum(np.floor(u1 * blades), blades - 1.0)
        a = u1 * blades - k  # uniform within the blade's triangle
        step = 2.0 * math.pi / blades
        a0 = math.pi / 2.0 + k * step
        r = np.sqrt(a)
        b1 = u2 * r  # triangle (centre, corner k, corner k + 1): weights 1 - r, r - b1, b1
        b0 = r - b1
        x = b0 * np.cos(a0) + b1 * np.cos(a0 + step)
        y = b0 * np.sin(a0) + b1 * np.sin(a0 + step)
    else:
        sx = 2.0 * u1 - 1.0
        sy = 2.0 * u2 - 1.0
        horizontal = np.abs(sx) > np.abs(sy)
        big = np.where(horizontal, sx, sy)
        small = np.where(horizontal, sy, sx)
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.where(
                horizontal,
                (math.pi / 4.0) * small / np.where(big != 0.0, big, 1.0),
                math.pi / 2.0 - (math.pi / 4.0) * small / np.where(big != 0.0, big, 1.0),
            )
        x = np.where(big == 0.0, 0.0, big * np.cos(phi))
        y = np.where(big == 0.0, 0.0, big * np.sin(phi))
    if rotation:
        c, s_ = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
        x, y = c * x - s_ * y, s_ * x + c * y
    if ratio != 1.0:
        x = x / ratio
    return x, y


def _focal(values: dict[str, Any], location: str) -> dict[str, Any]:
    """``focal_length`` (millimetres) given among camera values becomes their ``fov``."""
    if "focal_length" not in values:
        return values
    values = dict(values)
    focal = _number(values.pop("focal_length"), "focal_length", f"{location}.focal_length")
    sensor = _number(values.get("sensor", 36.0), "sensor", f"{location}.sensor")
    if not 0.1 <= focal <= 100_000.0:
        raise RenderError(
            "INVALID_CAMERA",
            f"focal_length must be within 0.1..100000 mm, got {focal:g}",
            f"{location}.focal_length",
        )
    values["fov"] = math.degrees(2.0 * math.atan(sensor / (2.0 * focal)))
    return values


def _aspect(aspect: Any) -> float:
    if isinstance(aspect, bool) or not isinstance(aspect, (int, float, np.integer, np.floating)):
        raise RenderError("INVALID_CAMERA", f"aspect must be a number, got {aspect!r}")
    value = float(aspect)
    if not (math.isfinite(value) and MIN_ASPECT <= value <= MAX_ASPECT):
        raise RenderError(
            "INVALID_CAMERA", f"aspect must be in [{MIN_ASPECT}, {MAX_ASPECT}], got {aspect!r}"
        )
    return value


def _validate(values: dict[str, Any], location: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    projection = values["projection"]
    if projection not in PROJECTIONS:
        raise RenderError(
            "INVALID_CAMERA",
            f"projection must be one of {', '.join(PROJECTIONS)}, got {projection!r}",
            f"{location}.projection",
        )
    out["projection"] = projection
    fov = _number(values["fov"], "fov", f"{location}.fov")
    if not MIN_FOV <= fov <= MAX_FOV:
        raise RenderError(
            "INVALID_CAMERA",
            f"fov must be between {MIN_FOV} and {MAX_FOV} degrees, got {fov:g}",
            f"{location}.fov",
        )
    out["fov"] = fov
    if values["fov_axis"] not in FOV_AXES:
        raise RenderError(
            "INVALID_CAMERA",
            f"fov_axis must be one of {', '.join(FOV_AXES)}, got {values['fov_axis']!r}",
            f"{location}.fov_axis",
        )
    out["fov_axis"] = values["fov_axis"]
    scale = _number(values["ortho_scale"], "ortho_scale", f"{location}.ortho_scale")
    if scale <= 0.0:
        raise RenderError(
            "INVALID_CAMERA", f"ortho_scale must be > 0, got {scale:g}", f"{location}.ortho_scale"
        )
    out["ortho_scale"] = scale
    near = _number(values["near"], "near", f"{location}.near")
    far = _number(values["far"], "far", f"{location}.far")
    if projection == "perspective" and near <= 0.0:
        raise RenderError(
            "INVALID_CAMERA",
            f"a perspective camera needs near > 0, got {near:g}",
            f"{location}.near",
        )
    if near < 0.0:
        raise RenderError("INVALID_CAMERA", f"near must be >= 0, got {near:g}", f"{location}.near")
    if far <= near:
        raise RenderError(
            "INVALID_CAMERA",
            f"far ({far:g}) must be greater than near ({near:g})",
            f"{location}.far",
            hint="set both at once with camera.configure(near=..., far=...)",
        )
    if projection == "perspective" and far / near > MAX_DEPTH_RATIO:
        raise RenderError(
            "INVALID_CAMERA",
            f"far / near is {far / near:.3g}; at most {MAX_DEPTH_RATIO:.0e} keeps depth precise",
            f"{location}.far",
            hint="increase near or decrease far",
        )
    out["near"], out["far"] = near, far
    aspect = values["aspect_ratio"]
    if aspect is not None:
        aspect = _number(aspect, "aspect_ratio", f"{location}.aspect_ratio")
        if not MIN_ASPECT <= aspect <= MAX_ASPECT:
            raise RenderError(
                "INVALID_CAMERA",
                f"aspect_ratio must be in [{MIN_ASPECT}, {MAX_ASPECT}], got {aspect:g}",
                f"{location}.aspect_ratio",
            )
    out["aspect_ratio"] = aspect
    for name, lo, hi in (
        ("sensor", 1.0, 1000.0),
        ("shift_x", -MAX_SHIFT, MAX_SHIFT),
        ("shift_y", -MAX_SHIFT, MAX_SHIFT),
        ("fstop", 0.0, 1e4),
        ("aperture_rotation", -360.0, 360.0),
        ("aperture_ratio", 0.01, 100.0),
    ):
        v = _number(values[name], name, f"{location}.{name}")
        if not lo <= v <= hi:
            raise RenderError(
                "INVALID_CAMERA", f"{name} must be within {lo:g}..{hi:g}, got {v:g}",
                f"{location}.{name}",
            )  # fmt: skip
        out[name] = v
    if 0.0 < out["fstop"] < 0.1:
        raise RenderError(
            "INVALID_CAMERA",
            f"fstop is 0 (a pinhole) or at least 0.1, got {out['fstop']:g}",
            f"{location}.fstop",
        )
    fd = _number(values["focus_distance"], "focus_distance", f"{location}.focus_distance")
    if not 1e-6 <= fd <= 1e12:
        raise RenderError(
            "INVALID_CAMERA",
            f"focus_distance must be within 1e-6..1e12 m, got {fd:g}",
            f"{location}.focus_distance",
        )
    out["focus_distance"] = fd
    fo = values["focus_object"]
    if fo is not None and (not isinstance(fo, str) or not 1 <= len(fo) <= 256):
        raise RenderError(
            "INVALID_CAMERA",
            f"focus_object is an object id or null, got {fo!r}",
            f"{location}.focus_object",
        )
    out["focus_object"] = fo
    blades = values["aperture_blades"]
    if (
        isinstance(blades, bool)
        or not isinstance(blades, (int, np.integer))
        or not (blades == 0 or 3 <= blades <= MAX_BLADES)
    ):
        raise RenderError(
            "INVALID_CAMERA",
            f"aperture_blades is 0 (round) or 3..{MAX_BLADES}, got {blades!r}",
            f"{location}.aperture_blades",
        )
    out["aperture_blades"] = int(blades)
    return out


# --------------------------------------------------------------------------- #
# Camera frame (world placement)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraFrame:
    """Orthonormal world-space placement of a camera.

    ``forward``, ``up`` and ``right`` are unit vectors with
    ``right = forward x up``; ``position`` is the eye point.
    """

    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    right: np.ndarray

    def view_matrix(self) -> np.ndarray:
        """4x4 world-to-view matrix (rows: right, up, -forward)."""
        v = np.eye(4, dtype=np.float64)
        v[0, :3], v[1, :3], v[2, :3] = self.right, self.up, -self.forward
        v[:3, 3] = -(v[:3, :3] @ self.position)
        return v

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "position": self.position.tolist(),
            "forward": self.forward.tolist(),
            "up": self.up.tolist(),
            "right": self.right.tolist(),
        }


def frame_from_matrix(world: np.ndarray, location: str = "camera") -> tuple[CameraFrame, list]:
    """The camera frame of a world matrix, plus diagnostics.

    Raises ``RenderError("INVALID_CAMERA_TRANSFORM")`` if the matrix is not
    finite or collapses the forward/up axes (zero scale or extreme shear).
    """
    m = np.asarray(world, dtype=np.float64)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise RenderError(
            "INVALID_CAMERA_TRANSFORM", "the camera's world matrix is not finite", location
        )
    forward_raw = -m[:3, 1]
    up_raw = m[:3, 2]
    f_len = float(np.linalg.norm(forward_raw))
    u_len = float(np.linalg.norm(up_raw))
    if not (f_len > 1e-12 and u_len > 1e-12):
        raise RenderError(
            "INVALID_CAMERA_TRANSFORM",
            "the camera's world transform has zero scale along its view or up axis",
            location,
        )
    forward = forward_raw / f_len
    up = up_raw - float(up_raw @ forward) * forward
    up_len = float(np.linalg.norm(up))
    if up_len <= 1e-9 * u_len:
        raise RenderError(
            "INVALID_CAMERA_TRANSFORM",
            "the camera's world transform makes its up axis parallel to its view axis",
            location,
        )
    up = up / up_len
    right = np.cross(forward, up)
    diagnostics = []
    linear = m[:3, :3]
    gram = linear.T @ linear
    if (
        float(np.linalg.det(linear)) < 0.0
        or not np.allclose(gram, np.eye(3) * gram[0, 0], rtol=1e-9, atol=1e-12)
        or not math.isclose(gram[0, 0], 1.0, rel_tol=1e-9)
    ):
        diagnostics.append(
            Diagnostic(
                "CAMERA_SCALE_IGNORED",
                INFO,
                "the camera's world transform contains scale, shear or mirroring; only its "
                "position, view direction and up direction are used",
                location,
            )
        )
    return CameraFrame(m[:3, 3].copy(), forward, up, right), diagnostics
