"""Shared pieces of the Stage 17 animation showcases: an original pixel-art character
skin, Minecraft-proportioned box meshes UV-mapped to the standard 64x64 skin layout, a
blocky rigged character, pixel-textured ground and props, rendering helpers and the
Khronos validator. Everything is generated here: no third-party assets.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from aether3d import Mesh, Scene
from aether3d.rendering.png import encode_png, write_png

ROOT = Path(__file__).resolve().parent.parent
VALIDATOR = ROOT / "tools" / "gltf-validator"
PX = 1.0 / 16.0  # one skin pixel = 1/16 m: the character is 2 m tall, a block 1 m

# --------------------------------------------------------------------------- #
# An original skin ("Nova, the courier"): 64x64 RGBA in the standard layout
# --------------------------------------------------------------------------- #

PALETTE = {
    "skin": (226, 172, 132),
    "skin_dark": (196, 140, 104),
    "hair": (74, 46, 34),
    "hair_light": (102, 66, 46),
    "eye_white": (240, 240, 236),
    "iris": (40, 92, 160),
    "mouth": (150, 70, 66),
    "jacket": (34, 136, 132),
    "jacket_dark": (24, 104, 102),
    "zip": (220, 224, 214),
    "badge": (242, 140, 40),
    "belt": (88, 58, 38),
    "cuff": (242, 140, 40),
    "pants": (44, 54, 92),
    "pants_dark": (34, 42, 74),
    "boot": (96, 62, 36),
    "sole": (52, 36, 24),
    "bag": (190, 70, 50),
}

# (u0, v0, w, h, d) of each part's net in the 64x64 layout
LAYOUT = {
    "head": (0, 0, 8, 8, 8),
    "body": (16, 16, 8, 12, 4),
    "arm_r": (40, 16, 4, 12, 4),
    "leg_r": (0, 16, 4, 12, 4),
    "arm_l": (32, 48, 4, 12, 4),
    "leg_l": (16, 48, 4, 12, 4),
}


def _faces(u0: int, v0: int, w: int, h: int, d: int) -> dict[str, tuple[int, int, int, int]]:
    """Pixel rectangles ``(x, y, width, height)`` of a part's six faces."""
    return {
        "top": (u0 + d, v0, w, d),
        "bottom": (u0 + d + w, v0, w, d),
        "right": (u0, v0 + d, d, h),
        "front": (u0 + d, v0 + d, w, h),
        "left": (u0 + d + w, v0 + d, d, h),
        "back": (u0 + 2 * d + w, v0 + d, w, h),
    }


def pixel_skin(seed: int = 7, **colors) -> np.ndarray:
    """The 64x64 RGBA skin, with a little per-pixel shading noise (seeded); ``colors``
    override :data:`PALETTE` entries (a crowd's variants: ``jacket=(r, g, b)``, ...)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((64, 64, 4), np.uint8)

    def fill(rect, color, noise=6):
        x, y, w, h = rect
        base = np.array(color, np.float64)
        jitter = rng.integers(-noise, noise + 1, size=(h, w, 1))
        img[y : y + h, x : x + w, :3] = np.clip(base + jitter, 0, 255).astype(np.uint8)
        img[y : y + h, x : x + w, 3] = 255

    def px(x, y, color):
        img[y, x, :3] = color
        img[y, x, 3] = 255

    unknown = set(colors) - set(PALETTE)
    if unknown:
        raise ValueError(f"unknown palette entries {sorted(unknown)}")
    p = {**PALETTE, **colors}
    head = _faces(*LAYOUT["head"])
    for name in ("right", "front", "left", "back", "bottom"):
        fill(head[name], p["skin"])
    fill(head["top"], p["hair"], 10)
    for name in ("right", "left", "back"):  # hair down the back and sides
        x, y, w, h = head[name]
        fill((x, y, w, 3 if name != "back" else 6), p["hair"], 10)
    fx, fy, _, _ = head["front"]
    fill((fx, fy, 8, 2), p["hair"], 10)  # fringe
    px(fx + 0, fy + 2, p["hair_light"])
    px(fx + 7, fy + 2, p["hair_light"])
    for ex in (1, 5):  # eyes: white then iris toward the middle
        px(fx + ex, fy + 4, p["eye_white"])
        px(fx + ex + 1, fy + 4, p["iris"])
    px(fx + 6, fy + 4, p["eye_white"])
    px(fx + 5, fy + 4, p["iris"])
    px(fx + 2, fy + 4, p["iris"])
    for x in range(1, 3):
        px(fx + x, fy + 3, p["hair"])  # brows
        px(fx + x + 4, fy + 3, p["hair"])
    for x in (3, 4):
        px(fx + x, fy + 6, p["mouth"])
    px(fx + 2, fy + 6, p["skin_dark"])
    px(fx + 5, fy + 6, p["skin_dark"])
    body = _faces(*LAYOUT["body"])
    for name, rect in body.items():
        fill(rect, p["jacket"] if name != "bottom" else p["pants"])
    bx, by, _, _ = body["front"]
    fill((bx + 3, by, 2, 10), p["zip"], 4)
    fill((bx + 5, by + 2, 2, 2), p["badge"], 3)
    fill((bx, by + 10, 8, 2), p["belt"], 4)
    px(bx + 3, by + 10, (210, 190, 90))  # buckle
    px(bx + 4, by + 10, (210, 190, 90))
    kx, ky, _, _ = body["back"]
    fill((kx + 1, ky + 1, 6, 7), p["bag"], 6)  # a courier's bag
    fill((kx + 1, ky + 1, 6, 1), p["belt"], 3)
    for side in ("left", "right"):
        x, y, w, h = body[side]
        fill((x, y + 10, w, 2), p["belt"], 4)
    for part in ("arm_r", "arm_l"):
        faces = _faces(*LAYOUT[part])
        for name, rect in faces.items():
            x, y, w, h = rect
            if name in ("top",):
                fill(rect, p["jacket"])
            elif name == "bottom":
                fill(rect, p["skin"])
            else:
                fill((x, y, w, 8), p["jacket"])
                fill((x, y + 8, w, 1), p["cuff"], 3)
                fill((x, y + 9, w, 3), p["skin"])
    for part in ("leg_r", "leg_l"):
        faces = _faces(*LAYOUT[part])
        for name, rect in faces.items():
            x, y, w, h = rect
            if name == "top":
                fill(rect, p["pants"])
            elif name == "bottom":
                fill(rect, p["sole"])
            else:
                fill((x, y, w, 9), p["pants"])
                fill((x, y + 9, w, 3), p["boot"], 5)
                fill((x, y + 11, w, 1), p["sole"], 3)
                if name == "front":
                    fill((x + 1, y + 4, 2, 1), p["pants_dark"], 2)  # a knee seam
    return img


# --------------------------------------------------------------------------- #
# Box meshes UV-mapped to the skin
# --------------------------------------------------------------------------- #


def part_box(part: str, center: tuple[float, float, float]) -> Mesh:
    """The box of a skin part (Minecraft size, ``PX`` per pixel) centered at ``center``
    in its object's space, facing -Y (front), its right side at -X, UVs on the net."""
    u0, v0, w, h, d = LAYOUT[part]
    faces = _faces(u0, v0, w, h, d)
    hx, hy, hz = w * PX / 2, d * PX / 2, h * PX / 2
    cx, cy, cz = center
    x0, x1, y0, y1, z0, z1 = cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz
    inset = 0.01  # (a hair inside each texel row/column: crisp nearest sampling)

    def uv(rect, corners):
        x, y, rw, rh = rect
        out = []
        for a, b in corners:  # a, b in 0..1 across the rect (b downward in the image)
            px_x = x + inset + a * (rw - 2 * inset)
            px_y = y + inset + b * (rh - 2 * inset)
            out.append((px_x / 64.0, 1.0 - px_y / 64.0))
        return out

    quads = {
        # face: 4 corners (top-left, top-right, bottom-right, bottom-left as seen), normal
        "front": ([(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0)], (0, -1, 0)),
        "back": ([(x1, y1, z1), (x0, y1, z1), (x0, y1, z0), (x1, y1, z0)], (0, 1, 0)),
        "right": ([(x0, y1, z1), (x0, y0, z1), (x0, y0, z0), (x0, y1, z0)], (-1, 0, 0)),
        "left": ([(x1, y0, z1), (x1, y1, z1), (x1, y1, z0), (x1, y0, z0)], (1, 0, 0)),
        "top": ([(x0, y1, z1), (x1, y1, z1), (x1, y0, z1), (x0, y0, z1)], (0, 0, 1)),
        "bottom": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1)),
    }
    pos, nrm, uvs, tris = [], [], [], []
    for name, (corners, normal) in quads.items():
        base = len(pos)
        pos += corners
        nrm += [normal] * 4
        uvs += uv(faces[name], [(0, 0), (1, 0), (1, 1), (0, 1)])
        tris += [(base, base + 3, base + 2), (base, base + 2, base + 1)]
    mesh = Mesh(np.array(pos, np.float32), np.array(tris), normals=np.array(nrm, np.float32),
                uvs=np.array(uvs, np.float32))  # fmt: skip
    return mesh


def _fix_winding(mesh: Mesh) -> Mesh:
    """Triangles wound counter-clockwise around their normals (outward)."""
    p = mesh.positions.astype(np.float64)
    t = mesh.triangles.copy()
    n = mesh.normals.astype(np.float64)
    face_n = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    flip = np.sum(face_n * n[t[:, 0]], axis=1) < 0
    t[flip] = t[flip][:, [0, 2, 1]]
    return Mesh(mesh.positions, t, normals=mesh.normals, uvs=mesh.uvs)


_PART_MESHES: dict[str, Mesh] = {}


def skin_material(scene: Scene, name: str, skin_png: bytes):
    """A material showing a 64x64 skin with crisp pixels (characters can share one)."""
    tex = scene.materials.add_image_texture(f"{name} skin", data=skin_png, embed=True,
                                            filter="nearest", mipmaps=False)  # fmt: skip
    mat = scene.create_material(f"{name} skin", roughness=0.85)
    mat.set_texture("base_color", tex)
    mat.base_color = (1.0, 1.0, 1.0)
    return mat


def blocky_character(scene: Scene, name: str, skin_png: bytes | None = None,
                     position=(0.0, 0.0, 0.0), material=None):  # fmt: skip
    """An armature with Minecraft-proportioned parts (head, body, arms, legs), each a
    textured box attached to its joint, facing -Y. Returns the armature object. The part
    meshes are shared by every character; pass ``material`` (:func:`skin_material`) to
    share a skin too."""
    arm = scene.add_object(name, position=position)
    a = scene.animation
    joints = [
        {"name": "Hips", "position": [0, 0, 12 * PX]},
        {"name": "Body", "parent": "Hips", "position": [0, 0, 0], "tail": [0, 0, 12 * PX]},
        {"name": "Head", "parent": "Body", "position": [0, 0, 12 * PX], "tail": [0, 0, 8 * PX]},
        {"name": "Arm.R", "parent": "Body", "position": [-6 * PX, 0, 10 * PX],
         "tail": [0, 0, -12 * PX]},
        {"name": "Arm.L", "parent": "Body", "position": [6 * PX, 0, 10 * PX],
         "tail": [0, 0, -12 * PX]},
        {"name": "Leg.R", "parent": "Hips", "position": [-2 * PX, 0, 0], "tail": [0, 0, -12 * PX]},
        {"name": "Leg.L", "parent": "Hips", "position": [2 * PX, 0, 0], "tail": [0, 0, -12 * PX]},
    ]  # fmt: skip
    a.add_rig(arm, joints)
    mat = material if material is not None else skin_material(scene, name, skin_png)
    px0 = np.asarray(position, np.float64)
    parts = {
        # part: (joint, box center in the armature's space)
        "head": ("Head", (0, 0, 28 * PX)),
        "body": ("Body", (0, 0, 18 * PX)),
        "arm_r": ("Arm.R", (-6 * PX, 0, 18 * PX)),
        "arm_l": ("Arm.L", (6 * PX, 0, 18 * PX)),
        "leg_r": ("Leg.R", (-2 * PX, 0, 6 * PX)),
        "leg_l": ("Leg.L", (2 * PX, 0, 6 * PX)),
    }
    for part, (joint, center) in parts.items():
        mesh = _PART_MESHES.get(part)
        if mesh is None:
            mesh = _PART_MESHES[part] = _fix_winding(part_box(part, (0.0, 0.0, 0.0)))
        obj = scene.add_object(f"{name} {part}", mesh=mesh, position=tuple(px0 + center))
        obj.material = mat
        a.attach(obj, arm, joint)
    return arm


# --------------------------------------------------------------------------- #
# A pixel world: grass ground, trees, flowers
# --------------------------------------------------------------------------- #


def _tile(colors: list[tuple[int, int, int]], weights, seed: int, size: int = 16) -> bytes:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(colors), size=(size, size), p=np.asarray(weights) / np.sum(weights))
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = np.array(colors, np.uint8)[idx]
    img[..., 3] = 255
    return encode_png(img)


def pixel_world(scene: Scene, size: float = 24.0, seed: int = 3) -> None:
    """Grass ground (one block = 1 m, 16 pixels), a few blocky trees and flowers."""
    grass = scene.materials.add_image_texture(
        "grass", data=_tile([(96, 152, 64), (84, 138, 56), (112, 170, 74), (76, 124, 50)],
                            [5, 4, 2, 1], seed), filter="nearest", mipmaps=True)  # fmt: skip
    ground = scene.add_plane("Ground", size=size)
    g = scene.create_material("Grass", roughness=0.95)
    g.set_texture("base_color", grass, coordinates="world", projection="xy", scale=1.0)
    g.base_color = (1.0, 1.0, 1.0)
    ground.material = g
    bark = scene.materials.add_image_texture(
        "bark", data=_tile([(102, 76, 46), (86, 62, 38), (120, 92, 56)], [4, 3, 1], seed + 1),
        filter="nearest", mipmaps=False)  # fmt: skip
    leaves = scene.materials.add_image_texture(
        "leaves", data=_tile([(52, 112, 40), (40, 94, 32), (70, 132, 52), (30, 76, 26)],
                             [4, 3, 2, 1], seed + 2), filter="nearest", mipmaps=False)  # fmt: skip
    wood = scene.create_material("Bark", roughness=0.9)
    wood.set_texture("base_color", bark, coordinates="object", projection="triplanar", scale=1.0)
    wood.base_color = (1.0, 1.0, 1.0)
    leaf = scene.create_material("Leaves", roughness=0.8)
    leaf.set_texture("base_color", leaves, coordinates="object", projection="triplanar",
                     scale=1.0)  # fmt: skip
    leaf.base_color = (1.0, 1.0, 1.0)
    rng = np.random.default_rng(seed)
    for k, (x, y) in enumerate([(-6.5, 5.5), (7.5, 6.5), (-8.5, -2.5), (9.5, -4.5), (2.5, 9.5)]):
        h = int(rng.integers(3, 5))
        trunk = scene.add_cube(f"Trunk {k}", size=1.0, position=(x, y, h / 2), scale=(1, 1, h))
        trunk.material = wood
        crown = scene.add_cube(f"Leaves {k}", size=1.0, position=(x, y, h + 1.0), scale=(3, 3, 2))
        crown.material = leaf
    colors = [(230, 60, 60), (250, 220, 70), (240, 240, 250), (150, 90, 210)]
    for k in range(18):
        x, y = rng.uniform(-10, 10, size=2)
        if abs(x) < 3 and abs(y) < 7:
            continue
        stem = scene.add_cube(f"Flower {k}", size=0.18, position=(x, y, 0.09))
        stem.color = tuple(c / 255 for c in colors[k % len(colors)])


def lights(scene: Scene) -> None:
    scene.world.background = (0.56, 0.74, 0.95)
    scene.world.ambient_color = (0.78, 0.86, 1.0)
    scene.world.ambient_intensity = 0.32
    scene.add_directional_light("Sun", direction=(-0.45, 0.55, -0.7), color=(1.0, 0.95, 0.86),
                                intensity=2.4, angle=1.5)  # fmt: skip


# --------------------------------------------------------------------------- #
# Checks and helpers
# --------------------------------------------------------------------------- #


def validate_glb(path: Path) -> dict[str, Any]:
    """The Khronos glTF Validator's verdict, or why it did not run."""
    if shutil.which("node") is None or not (VALIDATOR / "node_modules" / "gltf-validator").is_dir():
        return {"ran": False, "reason": "the Khronos glTF Validator is not installed"}
    proc = subprocess.run(["node", str(VALIDATOR / "validate.mjs"), str(path)],
                          capture_output=True, text=True, check=False)  # fmt: skip
    report = json.loads(proc.stdout.splitlines()[-1])
    return {"ran": True, "errors": report["errors"], "warnings": report["warnings"],
            "infos": report["infos"], "version": report["validatorVersion"]}  # fmt: skip


REPOSITORY = Path(__file__).resolve().parent.parent


def _portable(value: Any) -> Any:
    """Paths inside the repository made relative to it, programs named without their
    directory: reports read the same on every machine."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    if isinstance(value, str) and Path(value).is_absolute():
        path = Path(value)
        if path.is_relative_to(REPOSITORY):
            return path.relative_to(REPOSITORY).as_posix()
        if shutil.which(path.name) == value:  # (an executable found on the PATH)
            return path.name
    return value


def save_png(path: Path, image: Any, description: str) -> None:
    """Write an image composed from Aether3D's own output (a contact sheet, a graph, a
    comparison), tagged as its renders are: ``Software`` names Aether3D and
    ``Description`` says what the image is."""
    from aether3d import __version__

    text = {"Software": f"Aether3D {__version__}", "Description": description}
    write_png(path, image, text=text)


def write_json(path: Path, data: Any) -> None:
    text = json.dumps(_portable(data), indent=2, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


_CHOSEN: list[str] = []  # (set by use_backend)


def use_backend(name: str) -> None:
    """Render on ``gpu`` or ``cpu``, or (``auto``) on the GPU when one works here."""
    if name not in ("auto", "gpu", "cpu"):
        raise ValueError("backend is auto, gpu or cpu")
    _CHOSEN[:] = [] if name == "auto" else [name]


def backend() -> str:
    """The renderer the showcases use (the reports say which): the one chosen with
    :func:`use_backend`, else ``gpu`` when a GPU renders here, else ``cpu``."""
    return _CHOSEN[0] if _CHOSEN else _detected()


@functools.cache
def _detected() -> str:
    try:
        from aether3d.rendering.gpu import gpu_status

        return "gpu" if gpu_status().get("available") else "cpu"
    except Exception:
        return "cpu"


def shadows() -> bool:
    """Shadows where they render: the CPU renderer draws none (and refuses to pretend)."""
    return backend() == "gpu"


# --------------------------------------------------------------------------- #
# A smooth mannequin: one mesh of tubes and spheres around a humanoid rig
# --------------------------------------------------------------------------- #


def _tube(a, b, r0, r1, segments=20, rings=20, caps=True):
    """Positions, normals and triangles of a tube from ``a`` to ``b`` (radii ``r0``,
    ``r1``) with hemispherical caps."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    axis = b - a
    length = float(np.linalg.norm(axis))
    w = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(w, helper)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring_dirs = np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v
    profile = []  # (center offset along w, radius, normal tilt) from cap to cap
    if caps:
        for k in range(rings // 2, 0, -1):
            phi = np.pi / 2 * k / (rings // 2)
            profile.append((-r0 * np.sin(phi), r0 * np.cos(phi), -np.sin(phi)))
    for k in range(rings + 1):
        f = k / rings
        profile.append((f * length, r0 + (r1 - r0) * f, 0.0))
    if caps:
        for k in range(1, rings // 2 + 1):
            phi = np.pi / 2 * k / (rings // 2)
            profile.append((length + r1 * np.sin(phi), r1 * np.cos(phi), np.sin(phi)))
    pos, nrm = [], []
    for off, rad, tilt in profile:
        c = a + w * off
        n = ring_dirs * np.sqrt(max(1.0 - tilt * tilt, 0.0)) + w * tilt
        pos.append(c + ring_dirs * max(rad, 1e-4))
        nrm.append(n / np.linalg.norm(n, axis=1, keepdims=True))
    pos = np.concatenate(pos)
    nrm = np.concatenate(nrm)
    tris = []
    nring = len(profile)
    for r in range(nring - 1):
        for s_ in range(segments):
            p0 = r * segments + s_
            p1 = r * segments + (s_ + 1) % segments
            q0, q1 = p0 + segments, p1 + segments
            tris += [(p0, p1, q1), (p0, q1, q0)]
    tip_a = len(pos)
    pos = np.vstack([pos, a - w * r0, b + w * r1])
    nrm = np.vstack([nrm, -w, w])
    for s_ in range(segments):
        tris.append((tip_a, (s_ + 1) % segments, s_))
        base = (nring - 1) * segments
        tris.append((tip_a + 1, base + s_, base + (s_ + 1) % segments))
    return pos, nrm, np.array(tris)


def humanoid_rig(scene: Scene, name: str, position=(0.0, 0.0, 0.0)):
    """A humanoid armature (1.75 m, arms hanging, facing -Y) for the smooth mannequin."""
    arm = scene.add_object(name, position=position)
    joints = [
        {"name": "Hips", "position": [0, 0, 0.95]},
        {"name": "Spine", "parent": "Hips", "position": [0, 0, 0.12]},
        {"name": "Chest", "parent": "Spine", "position": [0, 0, 0.2]},
        {"name": "Neck", "parent": "Chest", "position": [0, 0, 0.22]},
        {"name": "Head", "parent": "Neck", "position": [0, 0, 0.1], "tail": [0, 0, 0.24]},
    ]
    for side, sx in (("L", 1), ("R", -1)):  # (facing -Y: the character's left is +X)
        joints += [
            {"name": f"UpperArm.{side}", "parent": "Chest", "position": [0.21 * sx, 0, 0.17],
             "tail": [0, 0, -0.29]},
            {"name": f"ForeArm.{side}", "parent": f"UpperArm.{side}", "position": [0, 0, -0.29],
             "tail": [0, 0, -0.25]},
            {"name": f"Hand.{side}", "parent": f"ForeArm.{side}", "position": [0, 0, -0.25],
             "tail": [0, 0, -0.09]},
            {"name": f"Thigh.{side}", "parent": "Hips", "position": [0.1 * sx, 0, -0.02],
             "tail": [0, 0, -0.44]},
            {"name": f"Shin.{side}", "parent": f"Thigh.{side}", "position": [0, 0, -0.44],
             "tail": [0, 0, -0.41]},
            {"name": f"Foot.{side}", "parent": f"Shin.{side}", "position": [0, 0, -0.41],
             "tail": [0, -0.15, -0.08]},
        ]  # fmt: skip
    scene.animation.add_rig(arm, joints)
    return arm


def mannequin(scene: Scene, armature, name: str = "Body", method: str = "linear"):
    """A smooth-skinned mannequin for a :func:`humanoid_rig`: one mesh of tubes, each
    weighted along its length to its bone and, near its ends, to the neighbouring bones
    (piecewise-linear profiles: smooth joints, no bone pulling on another's flesh)."""
    rig = scene.animation.rig(armature)
    mesh, profiles = mannequin_mesh(rig)
    obj = scene.add_object(name, mesh=mesh, position=armature.position)
    names = list(rig.names)
    pos = mesh.positions.astype(np.float64)
    idx = np.zeros((mesh.vertex_count, 2), np.int64)
    w = np.zeros((mesh.vertex_count, 2))
    for start, stop, a, b, profile in profiles:
        axis = b - a
        f = np.clip(((pos[start:stop] - a) @ axis) / float(axis @ axis), 0.0, 1.0)
        knots = np.array([k for k, _ in profile])
        joints = [names.index(j) for _, j in profile]
        seg = np.clip(np.searchsorted(knots, f, side="right") - 1, 0, len(knots) - 1)
        nxt = np.minimum(seg + 1, len(knots) - 1)
        span = np.where(nxt > seg, knots[nxt] - knots[seg], 1.0)
        u = np.clip((f - knots[seg]) / span, 0.0, 1.0)
        u = u * u * (3 - 2 * u)  # (smoothstep: no crease at the knots)
        idx[start:stop, 0] = np.array(joints)[seg]
        idx[start:stop, 1] = np.array(joints)[nxt]
        w[start:stop, 0] = 1.0 - u
        w[start:stop, 1] = u
    same = idx[:, 0] == idx[:, 1]
    w[same, 0] += w[same, 1]
    w[same, 1] = 0.0
    scene.animation.skin(obj, armature, weights={"indices": idx, "weights": w, "joints": names},
                         method=method)  # fmt: skip
    return obj


def mannequin_mesh(rig):
    """A smooth mannequin around a :func:`humanoid_rig`'s rest pose (armature space):
    tubes along the limbs, a tapered torso, a neck and a head, merged into one mesh; and
    each tube's vertex range with its weight profile (see :func:`mannequin`)."""
    model = rig.rest_model()
    tails = rig.tails()

    def p(name):
        return model[rig.index(name)][:3, 3]

    def tip(name):
        k = rig.index(name)
        return (model[k] @ np.array([*tails[k], 1.0]))[:3]

    parts = [  # (start, end, radius at start, at end, weight profile along the tube)
        (
            p("Hips") + [0, 0, -0.06],
            p("Chest") + [0, 0, 0.17],
            0.135,
            0.15,
            [(0.0, "Hips"), (0.3, "Spine"), (0.65, "Chest")],
        ),
        (
            p("Neck") + [0, 0, -0.03],
            p("Head") + [0, 0, 0.02],
            0.052,
            0.048,
            [(0.0, "Chest"), (0.35, "Neck"), (1.0, "Head")],
        ),
        (p("Head") + [0, 0, 0.1], tip("Head") + [0, 0, -0.01], 0.1, 0.098, [(0.0, "Head")]),
        # a nose (the mannequin faces -Y): head turns read at a glance
        (
            p("Head") + [0, -0.07, 0.13],
            p("Head") + [0, -0.115, 0.115],
            0.022,
            0.018,
            [(0.0, "Head")],
        ),
    ]
    for side in ("L", "R"):
        parts += [
            (p(f"UpperArm.{side}"), p(f"ForeArm.{side}"), 0.056, 0.047,
             [(0.0, "Chest"), (0.12, f"UpperArm.{side}"), (0.85, f"UpperArm.{side}"),
              (1.0, f"ForeArm.{side}")]),
            (p(f"ForeArm.{side}"), p(f"Hand.{side}"), 0.045, 0.037,
             [(0.0, f"UpperArm.{side}"), (0.15, f"ForeArm.{side}"), (0.55, f"ForeArm.{side}"),
              (1.0, f"Hand.{side}")]),  # (the wrist's twist spreads along the forearm)
            (p(f"Hand.{side}"), tip(f"Hand.{side}"), 0.04, 0.034,
             [(0.0, f"ForeArm.{side}"), (0.2, f"Hand.{side}")]),
            (p(f"Thigh.{side}") + [0, 0, 0.05], p(f"Shin.{side}"), 0.082, 0.06,
             [(0.0, "Hips"), (0.15, f"Thigh.{side}"), (0.85, f"Thigh.{side}"),
              (1.0, f"Shin.{side}")]),
            (p(f"Shin.{side}"), p(f"Foot.{side}"), 0.058, 0.045,
             [(0.0, f"Thigh.{side}"), (0.15, f"Shin.{side}"), (0.9, f"Shin.{side}"),
              (1.0, f"Foot.{side}")]),
            (p(f"Foot.{side}") + [0, 0.03, -0.035], tip(f"Foot.{side}") + [0, 0, 0.035],
             0.045, 0.04, [(0.0, f"Shin.{side}"), (0.2, f"Foot.{side}")]),
        ]  # fmt: skip
    pos, nrm, tris, profiles = [], [], [], []
    for a, b, r0, r1, profile in parts:
        pp, nn, tt = _tube(a, b, r0, r1)
        start = sum(len(x) for x in pos)
        tris.append(tt + start)
        pos.append(pp)
        nrm.append(nn)
        profiles.append((start, start + len(pp), np.asarray(a, np.float64),
                         np.asarray(b, np.float64), profile))  # fmt: skip
    mesh = Mesh(np.concatenate(pos).astype(np.float32), np.concatenate(tris),
                normals=np.concatenate(nrm).astype(np.float32))  # fmt: skip
    return mesh, profiles
