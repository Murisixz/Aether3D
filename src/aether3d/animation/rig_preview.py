"""Pictures of rigs: posed previews, weight and stretch heatmaps, skeleton overlays.

:func:`rig_preview` renders a character with Aether3D's own renderers into an image (a
NumPy RGBA array; PNG with ``path``) and, for the skeleton, an SVG diagram:

* ``mode="pose"`` -- the meshes in a pose (``time``, or a diagnostic pose by name), lit
  by a key, a fill and a rim light;
* ``mode="weights"`` -- one joint's weights as a heat ramp (``joint``), or every vertex
  coloured by its dominant joint (no ``joint``);
* ``mode="stretch"`` -- a diagnostic pose with each vertex coloured by how much its edges
  stretch or compress against the rest (blue compressed, white kept, red stretched);
* ``mode="skeleton"`` -- the pose with bones, joints and controls drawn over it
  (projected with the camera; ``svg`` returns the same drawing as SVG text).

The previews are built in a separate scene (copies of the posed meshes): the user's scene
is not changed. They are pictures for people and vision-capable agents; the numbers are
in :func:`.rig_report.diagnose_rig`.
"""

from __future__ import annotations

import colorsys
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError

if TYPE_CHECKING:
    from .system import AnimationSystem

MODES = ("pose", "weights", "stretch", "skeleton")


def _err(code: str, message: str, location: str) -> AnimationError:
    return AnimationError(code, message, location)


def _palette(n: int) -> np.ndarray:
    """``n`` distinct colours (golden-angle hues)."""
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        out.append(colorsys.hsv_to_rgb(h, 0.65, 0.95))
    return np.array(out)


def _posed_meshes(system, arm, time: float | None, pose: Any):
    """``(rigging result, {mesh id: (posed Mesh, world matrix)})`` for a time or a
    diagnostic pose (a suite name, or a custom pose: :func:`.rig_report.resolve_pose`)."""
    from ..rendering.overlay import RenderOverlay
    from ..rendering.scene_data import world_matrices
    from .rig_report import _posed, resolve_pose

    rig = system.rig(arm)
    t = system.time if time is None else float(time)
    if pose is not None:
        result = _posed(system, arm, rig, resolve_pose(rig, pose, "pose")["turns"], t)
    else:
        result = system.rigging(t)
    ids = tuple(result.local)
    overlay = RenderOverlay(
        ids=ids, local=np.array([result.local[i] for i in ids]).reshape(-1, 4, 4)
    )
    worlds = world_matrices(system.scene, overlay)
    rig_item = system._owned[("rig", arm._id)]
    meshes = {}
    attached = {oid for oid, _ in rig.attachments}
    for (kind, oid), item in system._owned.items():
        if kind == "skin" and item._definition.rig == rig_item.id and oid in system.scene._objects:
            obj = system.scene._objects[oid]
            if obj.visible:  # (what the scene shows: hidden meshes stay out)
                meshes[oid] = (result.meshes.get(oid, obj.mesh), worlds[oid])
    for oid in attached:
        obj = system.scene._objects.get(oid)
        if obj is not None and obj.mesh is not None and obj.visible:
            meshes[oid] = (obj.mesh, worlds[oid])
    return result, meshes, worlds


def rig_preview(
    system: AnimationSystem,
    armature: Any,
    *,
    mode: str = "pose",
    joint: str | None = None,
    time: float | None = None,
    pose: Any = None,
    width: int = 480,
    height: int = 640,
    view: str = "front",
    backend: str = "cpu",
    path: Any = None,
) -> dict[str, Any]:
    """A picture of ``armature``'s character (see the module docstring); returns
    ``{"image": RGBA array, "svg": text (skeleton), ...}`` and writes ``path`` (PNG)."""
    from ..mesh import Mesh
    from ..scene import Scene
    from .weights import weight_colors

    if mode not in MODES:
        raise _err("INVALID_ARGUMENT", f"mode is one of {', '.join(MODES)}", "mode")
    if view not in ("front", "side", "back", "three_quarter"):
        raise _err("INVALID_ARGUMENT", "view is front, side, back or three_quarter", "view")
    for name, v, hi in (("width", width, 4096), ("height", height, 4096)):
        if isinstance(v, bool) or not isinstance(v, int) or not 16 <= v <= hi:
            raise _err("INVALID_ARGUMENT", f"{name} is 16..{hi} pixels", name)
    arm = system._object(armature, "armature")
    rig = system.rig(arm)
    if rig is None:
        raise _err("RIG_MISSING", "the armature has no rig", "armature")
    if mode == "stretch" and pose is None:
        raise _err("INVALID_ARGUMENT", "a stretch map needs a diagnostic pose", "pose")
    result, meshes, worlds = _posed_meshes(system, arm, time, pose)
    preview = Scene(f"{arm.name} preview")
    preview.world.background = (0.16, 0.17, 0.2)
    preview.world.ambient_intensity = 0.35
    pts = []
    names = list(rig.names)
    colors_by_joint = _palette(len(names))
    for oid, (mesh, world) in meshes.items():
        src = system.scene._objects[oid]
        colors = None
        skin_item = system._owned.get(("skin", oid))
        if mode in ("weights", "stretch") and skin_item is not None:
            sk = skin_item._definition
            dense = sk.dense()
            pal_names = [rig.joints[rig.index_of_id(j)].name for j in sk.joints]
            if mode == "weights" and joint is not None:
                if joint not in names:
                    raise _err("JOINT_MISSING", f"no joint {joint!r}", "joint")
                w = dense[:, pal_names.index(joint)] if joint in pal_names else np.zeros(len(dense))
                colors = weight_colors(w)
            elif mode == "weights":
                dom = np.argmax(dense, axis=1)
                rgb = colors_by_joint[[names.index(pal_names[c]) for c in dom]]
                colors = np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)
            else:
                colors = _stretch_colors(src.mesh, mesh)
        elif mode in ("weights", "stretch"):
            colors = np.tile([0.5, 0.5, 0.55, 1.0], (mesh.vertex_count, 1))
        m = Mesh._from_arrays(
            mesh.positions,
            mesh.triangles,
            mesh.normals,
            mesh.uvs,
            mesh.groups,
            None if colors is None else colors.astype(np.float32),
        )
        obj = preview.add_object(src.name, mesh=m)
        _place(obj, world)
        if colors is None:
            obj.color = src.color if hasattr(src, "color") else (0.8, 0.7, 0.6)
        else:
            obj.color = (1.0, 1.0, 1.0)
        p = mesh.positions.astype(np.float64) @ world[:3, :3].T + world[:3, 3]
        pts.append(p)
    if not pts:
        raise _err("INVALID_ARGUMENT", "the rig binds no meshes to draw", "armature")
    allp = np.concatenate(pts)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    center = 0.5 * (lo + hi)
    size = float(np.max(hi - lo))
    a_world = worlds[arm._id]
    fwd = a_world[:3, :3] @ np.array(
        {"-y": (0, -1, 0), "+y": (0, 1, 0), "-x": (-1, 0, 0), "+x": (1, 0, 0)}[
            rig.forward_axis("-y")
        ],
        float,
    )
    fwd /= np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(fwd, up)
    direction = {
        "front": fwd,
        "back": -fwd,
        "side": side,
        "three_quarter": (fwd + 0.8 * side) / np.linalg.norm(fwd + 0.8 * side),
    }[view]
    fov = 30.0
    dist = 0.62 * size / math.tan(math.radians(fov / 2)) * max(1.0, width / height * 0.75)
    eye = center + direction * dist + up * 0.08 * size
    cam = preview.add_camera(
        "Camera", position=tuple(eye.tolist()), look_at=tuple(center.tolist()), fov=fov
    )
    key_dir = -(direction * 0.6 - side * 0.7 + up * 0.0) - up * 0.7
    preview.add_directional_light("Key", direction=tuple(key_dir.tolist()), intensity=2.2)
    preview.add_directional_light(
        "Fill",
        direction=tuple((-(direction * 0.7 + side * 0.7) - up * 0.3).tolist()),
        intensity=0.7,
    )
    preview.add_directional_light(
        "Rim", direction=tuple((direction * 0.9 - up * 0.4).tolist()), intensity=1.1
    )
    image = preview.render(camera=cam, width=width, height=height, backend=backend).image
    out: dict[str, Any] = {
        "mode": mode,
        "view": view,
        "width": width,
        "height": height,
        "backend": backend,
        "lights": ["key", "fill", "rim"],
    }
    if mode == "skeleton":
        image, svg = _skeleton_overlay(
            system, preview, cam, rig, result, worlds, arm, image, width, height
        )
        out["svg"] = svg
    out["image"] = image
    if path is not None:
        from ..rendering.png import write_png

        write_png(path, image)
        out["path"] = str(path)
    return out


def _place(obj, world: np.ndarray) -> None:
    from .constraints import decompose

    p, q, s = decompose(world)
    obj.position = tuple(p.tolist())
    obj.rotation = tuple((q if q[3] >= 0 else -q).tolist())
    obj.scale = tuple(s.tolist())


def _stretch_colors(rest_mesh, posed_mesh) -> np.ndarray:
    """Per vertex: the log ratio of its edges' lengths, posed over rest, as a diverging
    ramp (blue compressed to half, white kept, red stretched to double)."""
    t = np.asarray(rest_mesh.triangles, np.int64)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    p0 = rest_mesh.positions.astype(np.float64)
    p1 = posed_mesh.positions.astype(np.float64)
    l0 = np.linalg.norm(p0[e[:, 0]] - p0[e[:, 1]], axis=1)
    l1 = np.linalg.norm(p1[e[:, 0]] - p1[e[:, 1]], axis=1)
    ok = l0 > 0.35 * float(np.median(l0))
    r = np.zeros(len(e))
    r[ok] = np.log2(np.maximum(l1[ok], 1e-12) / l0[ok])
    n = len(p0)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for col in (0, 1):
        np.add.at(acc, e[ok, col], r[ok])
        np.add.at(cnt, e[ok, col], 1)
    v = np.clip(np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0), -1.0, 1.0)
    rgb = np.ones((n, 3))
    pos, neg = v > 0, v < 0
    rgb[pos] = np.stack([np.ones(pos.sum()), 1 - v[pos], 1 - v[pos]], axis=1)
    rgb[neg] = np.stack([1 + v[neg], 1 + v[neg], np.ones(neg.sum())], axis=1)
    return np.concatenate([rgb, np.ones((n, 1))], axis=1)


def _skeleton_overlay(system, preview, cam, rig, result, worlds, arm, image, width, height):
    _, model = result.joints[arm._id]
    a_world = worlds[arm._id]
    tails = rig.tails()
    heads = [(a_world @ model[k])[:3, 3] for k in range(len(rig.joints))]
    ends = [(a_world @ model[k] @ np.append(tails[k], 1.0))[:3] for k in range(len(rig.joints))]
    proj_h = preview.project([tuple(p) for p in heads], camera=cam, width=width, height=height)
    proj_e = preview.project([tuple(p) for p in ends], camera=cam, width=width, height=height)
    controls = []
    for c in rig.controls:
        if c.object and c.object in worlds and c.kind != "root":
            controls.append((c, worlds[c.object][:3, 3]))
    proj_c = (
        preview.project([tuple(p) for _, p in controls], camera=cam, width=width, height=height)
        if controls
        else []
    )
    img = np.array(image, copy=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    for k, j in enumerate(rig.joints):
        a, b = proj_h[k], proj_e[k]
        if a.x is None or b.x is None:
            continue
        color = (1.0, 0.85, 0.2) if j.deform else (0.6, 0.8, 1.0)
        _line(img, a.x, a.y, b.x, b.y, color, 2)
        _disc(img, a.x, a.y, 3.5, (0.95, 0.3, 0.2))
        hexc = _hex(color)
        svg.append(
            f'<line x1="{a.x:.1f}" y1="{a.y:.1f}" x2="{b.x:.1f}" y2="{b.y:.1f}" '
            f'stroke="{hexc}" stroke-width="2"><title>{j.name}</title></line>'
        )
        svg.append(
            f'<circle cx="{a.x:.1f}" cy="{a.y:.1f}" r="3.5" fill="#f24d33">'
            f"<title>{j.name}</title></circle>"
        )
    for (c, _), sp in zip(controls, proj_c, strict=True):
        if sp.x is None:
            continue
        color = {
            "ik_target": (0.2, 1.0, 0.4),
            "pole": (0.9, 0.3, 1.0),
            "gaze": (0.3, 0.8, 1.0),
        }.get(c.kind, (1, 1, 1))
        _ring(img, sp.x, sp.y, 7.0, color)
        hexc = _hex(color)
        svg.append(
            f'<circle cx="{sp.x:.1f}" cy="{sp.y:.1f}" r="7" fill="none" stroke="{hexc}" '
            f'stroke-width="2"><title>{c.name}</title></circle>'
        )
    svg.append("</svg>")
    return img, "\n".join(svg)


def _hex(color) -> str:
    r, g, b = (int(255 * c) for c in color)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend(img, ys, xs, color, alpha=1.0) -> None:
    h, w = img.shape[:2]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    ys, xs = ys[ok], xs[ok]
    scale = 255.0 if img.dtype == np.uint8 else 1.0
    c = np.array(color, np.float64) * scale
    cur = img[ys, xs, :3].astype(np.float64)
    img[ys, xs, :3] = (cur * (1 - alpha) + c * alpha).astype(img.dtype)


def _line(img, x0, y0, x1, y1, color, width=1) -> None:
    n = int(max(abs(x1 - x0), abs(y1 - y0)) * 2) + 2
    t = np.linspace(0, 1, n)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    for dx in range(-(width // 2), width // 2 + 1):
        for dy in range(-(width // 2), width // 2 + 1):
            _blend(img, (ys + dy).astype(int), (xs + dx).astype(int), color)


def _disc(img, x, y, r, color) -> None:
    rr = int(math.ceil(r))
    yy, xx = np.mgrid[-rr : rr + 1, -rr : rr + 1]
    m = xx * xx + yy * yy <= r * r
    _blend(img, (int(y) + yy[m]).ravel(), (int(x) + xx[m]).ravel(), color)


def _ring(img, x, y, r, color) -> None:
    rr = int(math.ceil(r)) + 1
    yy, xx = np.mgrid[-rr : rr + 1, -rr : rr + 1]
    d = np.sqrt(xx * xx + yy * yy)
    m = np.abs(d - r) < 1.0
    _blend(img, (int(y) + yy[m]).ravel(), (int(x) + xx[m]).ravel(), color)
