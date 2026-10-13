"""An independent reference renderer for tests: brute-force ray casting.

It shares no code with Aether3D's rasterizer -- no projection matrices,
clipping, fixed-point rasterization, visibility buffer or shading code. For
every pixel center it builds the camera ray from the documented camera
conventions, intersects it with every triangle (Moller-Trumbore), keeps the
nearest hit within the near/far range, and shades it with the documented
lighting equations (docs/RENDERING.md), written out here independently.

Use it only for small images: the cost is pixels x triangles.
"""

from __future__ import annotations

import math

import numpy as np

from aether3d import Scene


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _world(obj) -> np.ndarray:
    m = obj.local_matrix
    node = obj.parent
    while node is not None:
        m = node.local_matrix @ m
        node = node.parent
    return m


def gather(scene: Scene):
    """World-space triangles, vertex normals, object index and appearance per triangle."""
    tris, normals, has_normals, owners, objects = [], [], [], [], []
    for obj in scene:
        if obj.camera is not None or obj.light is not None or not obj.visible:
            continue
        mesh = obj.mesh
        if mesh is None or mesh.triangle_count == 0:
            continue
        m = _world(obj)
        lin = m[:3, :3]
        p = mesh.positions.astype(np.float64) @ lin.T + m[:3, 3]
        t = mesh.triangles.astype(np.int64)
        tris.append(p[t])
        if mesh.normals is not None:
            n = mesh.normals.astype(np.float64) @ np.linalg.inv(lin)  # (M^-T n)^T = n^T M^-1
            n /= np.linalg.norm(n, axis=1, keepdims=True)
            normals.append(n[t])
            has_normals.append(np.ones(len(t), bool))
        else:
            normals.append(np.zeros((len(t), 3, 3)))
            has_normals.append(np.zeros(len(t), bool))
        owners.append(np.full(len(t), len(objects)))
        objects.append(
            {
                "name": obj.name,
                "mirrored": np.linalg.det(lin) < 0,
                "base": srgb_to_linear(obj.appearance.color),
                "specular": obj.appearance.specular,
                "shininess": obj.appearance.shininess,
            }
        )
    return (
        np.concatenate(tris),
        np.concatenate(normals),
        np.concatenate(has_normals),
        np.concatenate(owners),
        objects,
    )


def camera_rays(scene: Scene, camera, width: int, height: int):
    """Ray origins and directions for pixel centers (row 0 at the top)."""
    m = _world(camera)
    forward = -m[:3, 1] / np.linalg.norm(m[:3, 1])
    up = m[:3, 2] - (m[:3, 2] @ forward) * forward
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    eye = m[:3, 3]
    cam = camera.camera
    aspect = width / height
    vertical = cam.fov_axis == "vertical" or (cam.fov_axis == "auto" and aspect < 1)
    if cam.projection == "perspective":
        half = math.tan(math.radians(cam.fov) / 2)
    else:
        half = cam.ortho_scale / 2
    hx, hy = (half * aspect, half) if vertical else (half, half / aspect)
    cols, rows = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    xn = cols / width * 2 - 1
    yn = 1 - rows / height * 2
    offset = xn[..., None] * hx * right + yn[..., None] * hy * up
    if cam.projection == "perspective":
        directions = forward + offset
        origins = np.broadcast_to(eye, directions.shape)
    else:
        directions = np.broadcast_to(forward, offset.shape)
        origins = eye + offset
    return origins.reshape(-1, 3), directions.reshape(-1, 3), eye, forward


def raycast(scene: Scene, camera, width: int, height: int):
    """Nearest hit per pixel: (triangle index or -1, ray parameter t = view depth, u, v)."""
    tris, *_ = gather(scene)
    origins, dirs, _, _ = camera_rays(scene, camera, width, height)
    near, far = camera.camera.near, camera.camera.far
    best_t = np.full(len(dirs), np.inf)
    best_i = np.full(len(dirs), -1)
    best_u = np.zeros(len(dirs))
    best_v = np.zeros(len(dirs))
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    for start in range(0, len(dirs), 512):
        o = origins[start : start + 512, None, :]
        d = dirs[start : start + 512, None, :]
        pvec = np.cross(d, e2[None])
        det = np.einsum("rtk,tk->rt", pvec, e1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o - v0[None]
        u = np.einsum("rtk,rtk->rt", tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None])
        v = np.einsum("rtk,rtk->rt", np.broadcast_to(d, qvec.shape), qvec) * inv
        t = np.einsum("tk,rtk->rt", e2, qvec) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= near) & (t <= far)
        t = np.where(hit, t, np.inf)
        i = np.argmin(t, axis=1)
        rows = np.arange(len(i))
        best_t[start : start + 512] = t[rows, i]
        best_i[start : start + 512] = np.where(np.isfinite(t[rows, i]), i, -1)
        best_u[start : start + 512] = u[rows, i]
        best_v[start : start + 512] = v[rows, i]
    return best_i, best_t, best_u, best_v


def reference_image(scene: Scene, camera, width: int, height: int, *, default_lights=False):
    """Linear radiance (H, W, 3), object index map (H, W) and depth map (H, W)."""
    tris, normals, has_normals, owners, objects = gather(scene)
    tri, t, u, v = raycast(scene, camera, width, height)
    origins, dirs, eye, forward = camera_rays(scene, camera, width, height)
    hit = tri >= 0
    k = tri[hit]
    b = np.stack([1 - u[hit] - v[hit], u[hit], v[hit]], axis=1)
    pos = origins[hit] + t[hit, None] * dirs[hit]
    geo = np.cross(tris[k, 1] - tris[k, 0], tris[k, 2] - tris[k, 0])
    mirrored = np.array([objects[o]["mirrored"] for o in owners[k]])
    geo[mirrored] *= -1
    perspective = camera.camera.projection == "perspective"
    to_eye = (eye - tris[k, 0]) if perspective else -forward
    front = np.einsum("ij,ij->i", geo, np.broadcast_to(to_eye, geo.shape)) > 0
    geo /= np.linalg.norm(geo, axis=1, keepdims=True)
    n = np.einsum("ij,ijk->ik", b, normals[k])
    n_len = np.linalg.norm(n, axis=1)
    use = has_normals[k] & (n_len > 1e-6)
    n = np.where(use[:, None], n / np.where(n_len > 0, n_len, 1)[:, None], geo)
    n[~front] *= -1
    view = (
        (eye - pos) / np.linalg.norm(eye - pos, axis=1, keepdims=True)
        if perspective
        else (np.broadcast_to(-forward, pos.shape))
    )
    base = np.array([objects[o]["base"] for o in owners[k]])
    spec = np.array([objects[o]["specular"] for o in owners[k]])
    shin = np.array([objects[o]["shininess"] for o in owners[k]])
    world = scene.world
    ambient = srgb_to_linear(world.ambient_color) * world.ambient_intensity
    color = base * ambient
    for light_obj in scene.lights:
        if not light_obj.visible:
            continue
        light = light_obj.light
        radiance = srgb_to_linear(light.color) * light.intensity
        m = _world(light_obj)
        if light.type == "directional":
            direction = -m[:3, 1] / np.linalg.norm(m[:3, 1])
            L = np.broadcast_to(-direction, pos.shape)
            att = np.ones(len(pos))
        else:
            delta = m[:3, 3] - pos
            dist = np.linalg.norm(delta, axis=1)
            L = delta / dist[:, None]
            dist = np.maximum(dist, 1e-4)
            att = 1 / dist**2 if light.falloff == "inverse_square" else np.ones(len(pos))
            if light.range is not None:
                att = att * np.clip(1 - (dist / light.range) ** 4, 0, 1) ** 2
        ndl = np.maximum(np.einsum("ij,ij->i", n, L), 0)
        color = color + base * radiance * (ndl * att)[:, None]
        h = L + view
        h = h / np.linalg.norm(h, axis=1, keepdims=True)
        ndh = np.maximum(np.einsum("ij,ij->i", n, h), 0)
        color = color + radiance * (spec * np.where(ndl > 0, ndh**shin, 0) * att)[:, None]
    image = np.broadcast_to(srgb_to_linear(world.background), (len(tri), 3)).copy()
    image[hit] = color
    owner_map = np.full(len(tri), -1)
    owner_map[hit] = owners[k]
    depth = np.full(len(tri), np.inf)
    if perspective:
        depth[hit] = t[hit]  # the ray direction has a unit forward component
    else:
        depth[hit] = t[hit]
    shape = (height, width)
    normal_map = np.zeros((len(tri), 3))
    normal_map[hit] = geo
    return (
        image.reshape(*shape, 3),
        owner_map.reshape(shape),
        depth.reshape(shape),
        objects,
        normal_map.reshape(*shape, 3),
    )
