"""Views: where an observation looks from.

A view is either one of the scene's cameras or a **virtual camera** -- a lens and a
placement that exist only for the observation (:func:`aether3d.render` renders from it,
and spatial projections use it) and never become part of the scene. Inspection views
(a top view, a close-up framed on some objects) therefore change nothing: no object, no
history step, no revision.

View specifications are JSON (from Python, the CLI, expectations and MCP):

* ``{"camera": "Cam"}`` -- a scene camera by id or name; ``{}`` (or ``"camera": null``) is
  the active camera.
* ``{"virtual": {"position": [x, y, z], "look_at": [x, y, z] | "direction": [x, y, z],
  "up": [0, 0, 1], "projection", "fov", "fov_axis", "ortho_scale", "near", "far"}}`` -- a
  virtual camera placed explicitly.
* ``{"focus": {"objects": SELECTOR, "direction": [x, y, z], "up": [...], "margin": 0.1,
  "from_camera": REF}}`` -- a virtual camera framed exactly on the selected objects
  (``rendering/framing.py``): it looks along ``direction`` (default: the ``from_camera``'s
  or the active camera's view direction, else from the front-right and above) with that
  camera's lens (default: a 40-degree perspective lens).
* ``{"preset": "top" | "bottom" | "front" | "back" | "left" | "right" | "iso", "objects":
  SELECTOR, "margin": 0.08}`` -- a focus view along a fixed direction, on the selected
  objects (default: every visible object with geometry). ``top`` looks down (-Z, +Y at the
  top of the image), ``front`` looks along +Y at the objects' fronts (assets face -Y),
  ``back`` along -Y, ``left`` and ``right`` from the -X and +X sides (the viewer's left and
  right when facing the fronts), ``iso`` from the front-right and above.

Every form also takes ``"name"`` (a label for reports). A resolved view records the exact
camera it used (lens, eye, forward, up, right), so later projections -- grounding a vision
model's image regions, composition corrections -- use the same view.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError, FeedbackError, PlanError, RenderError
from ..rendering.camera import Camera, CameraFrame, frame_from_matrix
from ..rendering.framing import AUTO_VIEW_DIRECTION, compute_framing, orientation_frame
from ..rendering.scene_data import RenderCamera
from ._doc import BAD, Doc, join

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "top": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
    "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "iso": (AUTO_VIEW_DIRECTION, (0.0, 0.0, 1.0)),
}
"""View direction and image up of each preset view."""

LENS_KEYS = ("projection", "fov", "fov_axis", "ortho_scale", "near", "far")
DEFAULT_MARGIN = 0.08


def virtual_camera(
    position: Any,
    *,
    look_at: Any = None,
    direction: Any = None,
    up: Any = (0.0, 0.0, 1.0),
    name: str = "virtual camera",
    **lens: Any,
) -> RenderCamera:
    """A camera that is not part of the scene, for :func:`aether3d.render` and observations.

    It stands at ``position`` and looks at the point ``look_at`` or along ``direction``,
    with local +Z as close to ``up`` as possible (the convention of ``obj.look_at``).
    ``lens`` takes :class:`~aether3d.Camera` parameters (``projection``, ``fov``,
    ``fov_axis``, ``ortho_scale``, ``near``, ``far``). Raises ``FeedbackError``
    (``INVALID_VIEW``) for an invalid placement or lens.
    """
    from ..math3d import vec3

    try:
        eye = np.asarray(vec3(position, "position"), dtype=np.float64)
        if (look_at is None) == (direction is None):
            raise FeedbackError(
                "INVALID_VIEW", "give look_at or direction (one of them)", "virtual"
            )
        if look_at is not None:
            aim = np.asarray(vec3(look_at, "look_at"), dtype=np.float64) - eye
        else:
            aim = np.asarray(vec3(direction, "direction"), dtype=np.float64)
        if not np.all(np.isfinite(eye)) or not np.all(np.isfinite(aim)):
            raise FeedbackError("INVALID_VIEW", "the view's placement is not finite", "virtual")
        if float(np.linalg.norm(aim)) < 1e-12:
            raise FeedbackError(
                "INVALID_VIEW", "the camera looks at its own position", "virtual/look_at"
            )
        orientation = orientation_frame(tuple(aim.tolist()), up)
        camera = Camera(**lens)
    except FeedbackError:
        raise
    except (AetherError, TypeError, ValueError) as exc:
        message = getattr(exc, "message", None) or (str(exc.args[0]) if exc.args else str(exc))
        raise FeedbackError("INVALID_VIEW", message, "virtual") from None
    frame = CameraFrame(eye, orientation.forward, orientation.up, orientation.right)
    return RenderCamera(None, name, camera, frame)


def camera_view(scene: Scene, camera: Any = None) -> RenderCamera:
    """The virtual twin of a scene camera: its lens and its current world frame (renders from
    it are bit for bit renders from the camera)."""
    from ..rendering.scene_data import resolve_camera, world_matrices

    obj = resolve_camera(scene, camera)
    frame, _ = frame_from_matrix(world_matrices(scene)[obj.id], f"objects[{obj.name}]")
    return RenderCamera(None, obj.name, obj.camera.copy(), frame)


@dataclass(frozen=True)
class ResolvedView:
    """A view specification resolved against a scene at one revision."""

    name: str
    kind: str  # "camera" | "virtual" | "focus" | "preset"
    spec: dict[str, Any]
    camera: SceneObject | None  # the scene camera (kind "camera")
    virtual: RenderCamera | None  # the virtual camera (other kinds)
    width: int
    height: int
    notes: tuple[str, ...] = ()

    @property
    def target(self) -> Any:
        """What :func:`aether3d.render` and :class:`~aether3d.spatial.queries.View` take."""
        return self.camera if self.camera is not None else self.virtual

    @property
    def key(self) -> str:
        """Identifies the view's content (for observation caches)."""
        if self.camera is not None:
            ident: Any = {"camera": self.camera.id}
        else:
            assert self.virtual is not None
            cam = self.virtual.camera
            f = self.virtual.frame
            ident = {
                "lens": cam.to_dict(),
                "frame": [list(map(float, v)) for v in (f.position, f.forward, f.up, f.right)],
            }
        return json.dumps(
            {**ident, "size": [self.width, self.height]}, sort_keys=True, separators=(",", ":")
        )

    def render_camera(self, scene: Scene) -> RenderCamera:
        """The camera as rendered now (for a scene camera: its current world frame)."""
        if self.virtual is not None:
            return self.virtual
        return camera_view(scene, self.camera)

    def to_dict(self, scene: Scene | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "kind": self.kind, "spec": self.spec}
        out["camera"] = (
            None if self.camera is None else {"id": self.camera.id, "name": self.camera.name}
        )
        rc = self.virtual
        if rc is None and scene is not None and self.camera is not None:
            try:
                rc = camera_view(scene, self.camera)
            except AetherError:
                rc = None
        if rc is not None:
            info = rc.to_dict(self.width / self.height)
            info.pop("id", None)
            info.pop("name", None)
            out["lens_and_frame"] = _rounded(info)
        out["width"], out["height"] = self.width, self.height
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    return value


def check_view_spec(spec: Any, ptr: str = "") -> dict[str, Any]:
    """``spec`` checked for shape (every problem at once); a normalized copy."""
    doc = Doc()
    out = parse_view_spec(doc, spec, ptr)
    doc.issues.raise_if_any("invalid view")
    assert out is not None
    return out


def parse_view_spec(doc: Doc, spec: Any, ptr: str) -> dict[str, Any] | None:
    """Check a view specification into ``doc``'s issues; ``None`` if it is unusable."""
    if spec is None:
        return {}
    obj = doc.object(spec, ptr, "a view: {camera}, {virtual}, {focus} or {preset}")
    if obj is None:
        return None
    kinds = [k for k in ("camera", "virtual", "focus", "preset") if k in obj]
    doc.keys(obj, ptr, ("camera", "virtual", "focus", "preset", "objects", "margin", "name"))
    if len(kinds) > 1:
        doc.issues.add(
            "INVALID_VALUE", f"a view has one of camera, virtual, focus, preset (got {kinds})", ptr
        )
        return None
    out: dict[str, Any] = {}
    if "name" in obj:
        name = doc.string(obj["name"], join(ptr, "name"), max_length=128)
        if name is not BAD:
            out["name"] = name
    kind = kinds[0] if kinds else "camera"
    if kind != "preset":
        for key in ("objects", "margin"):
            if key in obj:
                doc.issues.add("INVALID_VALUE", f"{key} goes with a preset view", join(ptr, key))
    if kind == "camera":
        value = obj.get("camera")
        if value is not None and doc.string(value, join(ptr, "camera")) is BAD:
            return None
        out["camera"] = value
    elif kind == "virtual":
        v = doc.object(obj["virtual"], join(ptr, "virtual"))
        if v is None:
            return None
        p = join(ptr, "virtual")
        doc.keys(v, p, ("position", "look_at", "direction", "up", *LENS_KEYS), ("position",))
        clean: dict[str, Any] = {}
        for key in ("position", "look_at", "direction", "up"):
            if key in v:
                vec = doc.vector(v[key], join(p, key), nonzero=key in ("direction", "up"))
                if vec is not BAD:
                    clean[key] = list(vec)
        if ("look_at" in v) == ("direction" in v):
            doc.issues.add("INVALID_VALUE", "give look_at or direction (one of them)", p)
        clean.update(_lens(doc, v, p))
        out["virtual"] = clean
    elif kind == "focus":
        f = doc.object(obj["focus"], join(ptr, "focus"))
        if f is None:
            return None
        p = join(ptr, "focus")
        doc.keys(
            f, p, ("objects", "direction", "up", "margin", "from_camera", *LENS_KEYS), ("objects",)
        )
        clean = {}
        if "objects" in f and doc.selector(f["objects"], join(p, "objects")) is not BAD:
            clean["objects"] = f["objects"]
        for key in ("direction", "up"):
            if key in f:
                vec = doc.vector(f[key], join(p, key), nonzero=True)
                if vec is not BAD:
                    clean[key] = list(vec)
        if "margin" in f:
            m = doc.number(f["margin"], join(p, "margin"), lo=0.0, hi=0.9)
            if m is not BAD:
                clean["margin"] = m
        if "from_camera" in f:
            c = doc.string(f["from_camera"], join(p, "from_camera"))
            if c is not BAD:
                clean["from_camera"] = c
        clean.update(_lens(doc, f, p))
        out["focus"] = clean
    else:
        preset = doc.enum(obj["preset"], join(ptr, "preset"), tuple(PRESETS))
        if preset is BAD:
            return None
        out["preset"] = preset
        if "objects" in obj and doc.selector(obj["objects"], join(ptr, "objects")) is not BAD:
            out["objects"] = obj["objects"]
        if "margin" in obj:
            m = doc.number(obj["margin"], join(ptr, "margin"), lo=0.0, hi=0.9)
            if m is not BAD:
                out["margin"] = m
    return out


def _lens(doc: Doc, obj: dict[str, Any], ptr: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in LENS_KEYS:
        if key not in obj:
            continue
        p = join(ptr, key)
        if key == "projection":
            value = doc.enum(obj[key], p, ("perspective", "orthographic"))
        elif key == "fov_axis":
            value = doc.enum(obj[key], p, ("vertical", "horizontal", "auto"))
        elif key == "fov":
            value = doc.number(obj[key], p, lo=0.01, hi=179.0)
        else:
            value = doc.number(obj[key], p, lo=0.0)
        if value is not BAD:
            out[key] = value
    return out


def resolve_view(scene: Scene, spec: Any, width: int, height: int) -> ResolvedView:
    """Resolve a view specification (see the module docstring) against ``scene`` now.

    Raises ``FeedbackError`` (``INVALID_VIEW``, ``NO_CAMERA``, ``NOTHING_TO_FRAME``,
    ``OBJECT_NOT_FOUND``).
    """
    spec = check_view_spec(spec)
    aspect = width / height
    name = spec.get("name")
    try:
        if "virtual" in spec:
            v = dict(spec["virtual"])
            lens = {k: v.pop(k) for k in LENS_KEYS if k in v}
            rc = virtual_camera(
                v["position"],
                look_at=v.get("look_at"),
                direction=v.get("direction"),
                up=v.get("up", (0.0, 0.0, 1.0)),
                name=name or "virtual",
                **lens,
            )
            return ResolvedView(name or "virtual", "virtual", spec, None, rc, width, height)
        if "focus" in spec or "preset" in spec:
            return _focus_view(scene, spec, width, height, aspect)
        from ..rendering.scene_data import resolve_camera

        camera = resolve_camera(scene, spec.get("camera"))
        return ResolvedView(name or camera.name, "camera", spec, camera, None, width, height)
    except FeedbackError:
        raise
    except PlanError as exc:
        raise FeedbackError(exc.code, exc.message, exc.pointer or "view", hint=exc.hint) from None
    except RenderError as exc:
        code = exc.code if exc.code in ("NO_CAMERA", "NOTHING_TO_FRAME") else "INVALID_VIEW"
        if exc.code == "OBJECT_NOT_FOUND":
            code = "OBJECT_NOT_FOUND"
        raise FeedbackError(code, exc.message, "view", hint=exc.hint) from None


def _focus_view(
    scene: Scene, spec: dict[str, Any], width: int, height: int, aspect: float
) -> ResolvedView:
    from ..plans import select_objects
    from ..rendering.framing import target_points
    from ..rendering.scene_data import resolve_camera, world_matrices

    notes: list[str] = []
    if "preset" in spec:
        kind = "preset"
        direction, up = PRESETS[spec["preset"]]
        selector = spec.get("objects")
        margin = spec.get("margin", DEFAULT_MARGIN)
        lens_source = scene.active_camera
        lens: dict[str, Any] = {}
        label = spec["preset"]
    else:
        kind = "focus"
        focus = spec["focus"]
        selector = focus["objects"]
        margin = focus.get("margin", DEFAULT_MARGIN)
        up = tuple(focus.get("up", (0.0, 0.0, 1.0)))
        lens_source = (
            resolve_camera(scene, focus["from_camera"])
            if "from_camera" in focus
            else scene.active_camera
        )
        lens = {k: focus[k] for k in LENS_KEYS if k in focus}
        direction = focus.get("direction")
        if direction is None:
            if lens_source is not None:
                world = world_matrices(scene)[lens_source.id]
                frame, _ = frame_from_matrix(world, f"objects[{lens_source.name}]")
                direction = tuple(float(c) for c in frame.forward)
            else:
                direction = AUTO_VIEW_DIRECTION
        label = "focus"
    if selector is None:
        targets = None
        names = "everything visible"
    else:
        objects = select_objects(scene, selector, "/objects")
        if not objects:
            raise FeedbackError(
                "NOTHING_TO_FRAME", "the view's objects select nothing", "view/objects"
            )
        targets = [o for o in objects if o.camera is None and o.light is None]
        hidden = [o.name for o in targets if not o.visible]
        if hidden:
            notes.append(f"hidden objects do not render: {', '.join(hidden[:5])}")
        names = ", ".join(o.name for o in objects[:3]) + ("..." if len(objects) > 3 else "")
    camera = lens_source.camera.copy() if lens_source is not None else Camera()
    if lens:
        camera.configure(**lens)
    points = target_points(scene, targets)
    base = orientation_frame(direction, up)
    framing = compute_framing(camera, base, points, aspect, margin=margin, fit_clipping=True)
    camera.configure(**framing.camera_changes)
    frame = CameraFrame(framing.position, base.forward, base.up, base.right)
    name = spec.get("name") or f"{label} on {names}"
    rc = RenderCamera(None, name, camera, frame)
    if not all(math.isfinite(float(c)) for c in framing.position):  # pragma: no cover
        raise FeedbackError("INVALID_VIEW", "framing produced non-finite values", "view")
    return ResolvedView(name, kind, spec, None, rc, width, height, tuple(notes))
