"""Motion vectors (0.20.0): ``render(..., motion=dt)`` adds ``passes["motion"]``.

For every pixel, the surface its centre sees at the render's time is found with one ray
(the ray core of the path tracers: instance, triangle and barycentrics), and the same
point of the same triangle -- the triangle's corners posed at ``time - dt`` and ``time +
dt`` (transforms, skins and morphs as the animation evaluates them) -- is projected with
the camera at those times. ``passes["motion"]`` is ``(H, W, 4) float32``: the pixel
offsets ``(x, y)`` to where the point was at ``time - dt``, then ``(z, w)`` to where it
will be at ``time + dt`` (``+x`` right, ``+y`` down, in pixels; Blender's Vector pass
holds the same two motions). Pixels that see the background, or a point that is missing
or behind the camera at the other time, have no motion (0). It works whatever backend
drew the image (the rays use the renderers' world matrices), so compositors can blur or
retime any render; Aether3D's own motion blur samples time for real
(:func:`aether3d.animation.frames.render_frames`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import RenderError


def _scene_at(scene: Any, camera_ref: Any, time: float, aspect: float, default_lights: bool):
    from .overlay import animation_overlay
    from .scene_data import extract_render_scene, resolve_camera

    overlay = animation_overlay(scene, time)
    camera = camera_ref
    if camera is None and overlay is not None and overlay.camera is not None:
        camera = overlay.camera
    obj = resolve_camera(scene, camera)
    return extract_render_scene(
        scene, obj, default_lights=default_lights, aspect=aspect, overlay=overlay
    )


def _project(rs: Any, points: np.ndarray, width: int, height: int):
    """Pixel coordinates (centres at ``k + 0.5``) of world points, and where they are in
    front of the camera."""
    cam = rs.camera.camera
    frame = rs.camera.frame
    aspect = width / height
    hx, hy = cam.half_extents(aspect)
    ox, oy = cam.ndc_offset(aspect)
    v = points - np.asarray(frame.position, np.float64)
    x = v @ np.asarray(frame.right, np.float64)
    y = v @ np.asarray(frame.up, np.float64)
    z = v @ np.asarray(frame.forward, np.float64)
    if cam.is_perspective:
        ok = z > 1e-9
        zs = np.where(ok, z, 1.0)
        xn, yn = x / (zs * hx) - ox, y / (zs * hy) - oy
    else:
        ok = np.ones(points.shape[0], dtype=bool)
        xn, yn = x / hx - ox, y / hy - oy
    return (xn + 1.0) * (width / 2.0), (1.0 - yn) * (height / 2.0), ok


def motion_vectors(
    scene: Any,
    rs: Any,
    width: int,
    height: int,
    time: float,
    dt: float,
    camera_ref: Any = None,
    default_lights: bool = True,
) -> np.ndarray:
    """``(height, width, 4) float32`` motion to ``time - dt`` and ``time + dt`` (see the
    module docstring)."""
    from .pathtrace.scene import PathScene

    if rs.camera.object_id is None:
        raise RenderError(
            "INVALID_SETTINGS",
            "motion vectors follow a scene camera through time (not a temporary framed one)",
            "motion",
        )
    n = width * height
    cam = rs.camera.camera
    frame = rs.camera.frame
    aspect = width / height
    hx, hy = cam.half_extents(aspect)
    ox, oy = cam.ndc_offset(aspect)
    row, col = np.divmod(np.arange(n, dtype=np.int64), width)
    xn = (col + 0.5) * (2.0 / width) - 1.0 + ox
    yn = 1.0 - (row + 0.5) * (2.0 / height) + oy
    right = np.asarray(frame.right, np.float64)
    up = np.asarray(frame.up, np.float64)
    forward = np.asarray(frame.forward, np.float64)
    eye = np.asarray(frame.position, np.float64)
    if cam.is_perspective:
        d = forward[None] + (xn * hx)[:, None] * right + (yn * hy)[:, None] * up
        length = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / length
        o = np.broadcast_to(eye, d.shape).copy()
        tmin, tmax = cam.near * length[:, 0], cam.far * length[:, 0]
    else:
        o = eye + (xn * hx)[:, None] * right + (yn * hy)[:, None] * up
        d = np.broadcast_to(forward, o.shape).copy()
        tmin, tmax = np.full(n, cam.near), np.full(n, cam.far)
    out = np.zeros((n, 4), dtype=np.float32)
    if not rs.instances:
        return out.reshape(height, width, 4)
    ps = PathScene(rs)
    hits, _ = ps.trace(o, d, tmin, tmax)
    hit = np.flatnonzero(hits.inst >= 0)
    if not hit.size:
        return out.reshape(height, width, 4)
    inst, prim, bary = hits.inst[hit], hits.prim[hit], hits.b[hit]
    here_x, here_y = col[hit] + 0.5, row[hit] + 0.5
    for k, t2 in enumerate((time - dt, time + dt)):
        rs2 = _scene_at(scene, rs.camera.object_id, t2, aspect, default_lights)
        index = {i.object_id: j for j, i in enumerate(rs2.instances)}
        for i in np.unique(inst):
            rows = np.flatnonzero(inst == i)
            a = rs.instances[int(i)]
            j = index.get(a.object_id)
            if j is None:
                continue
            b = rs2.instances[j]
            mesh = rs2.meshes[b.mesh]
            if mesh.triangles.shape[0] != rs.meshes[a.mesh].triangles.shape[0]:
                continue  # (the topology changed: no correspondence)
            corners = np.asarray(mesh.positions, np.float64)[mesh.triangles[prim[rows]]]
            local = (bary[rows, :, None] * corners).sum(axis=1)
            world = local @ b.world[:3, :3].T + b.world[:3, 3]
            px, py, ok = _project(rs2, world, width, height)
            sel = hit[rows]
            out[sel, 2 * k] = np.where(ok, px - here_x[rows], 0.0)
            out[sel, 2 * k + 1] = np.where(ok, py - here_y[rows], 0.0)
    return out.reshape(height, width, 4)
