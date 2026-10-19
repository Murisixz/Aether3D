"""Shared pieces of the Stage 18 rigging showcases (``examples/rigging_*.py``).

Everything is modeled, rigged, posed, animated and rendered by Aether3D: the character is
the anatomical figure of ``aether3d.sculpt.figure`` (an original procedural body -- pelvis,
ribcage, muscles, knees, feet with toes, hands with fingers and a thumb, a face; no
third-party asset), meshed by marching cubes and remeshed to even triangles, rigged by
``autorig``; clothing is its shapes inflated (``figure_garment``). FFmpeg only encodes the
PNG frames Aether3D rendered.

Lighting (``studio``): a warm key light from the front left with a soft disc, a cool fill
from the right, a strong rim from behind, cool ambient light, and a matte studio floor;
the path tracer adds the soft shadows of the lights' discs and the environment.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from worlds_common import (  # noqa: E402
    downscale,
    preview,
    progress_printer,
    software,
    verify,
)

from aether3d import Scene  # noqa: E402
from aether3d.rendering.png import encode_png  # noqa: E402

OUT = HERE / "output" / "rigging"
BACKEND = "auto"

# Clothing colors (display sRGB) and the joint roles each dresses.
PALETTE = {
    "skin": ((0.80, 0.62, 0.50), 0.55),
    "top": ((0.16, 0.36, 0.62), 0.62),
    "trousers": ((0.20, 0.21, 0.25), 0.7),
    "shoes": ((0.55, 0.30, 0.16), 0.45),
    "hair": ((0.17, 0.11, 0.07), 0.6),
}


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def set_backend(name: str) -> str:
    """``gpu`` (rasterizer with shadows; path tracer for hero stills), ``cpu`` or ``auto``
    (the GPU when one works)."""
    global BACKEND
    if name == "auto":
        try:
            from aether3d.rendering.gpu import gpu_status

            name = "gpu" if gpu_status().get("available") else "cpu"
        except Exception:  # noqa: BLE001 - no GPU stack: the CPU renders
            name = "cpu"
    if name not in ("gpu", "cpu"):
        raise ValueError(f"backend is auto, gpu or cpu, not {name!r}")
    BACKEND = name
    return name


def render_options(*, width: int, height: int, path_samples: int = 0) -> dict[str, Any]:
    if path_samples and BACKEND == "gpu":
        return {
            "backend": "gpu_path",
            "path": {"samples": path_samples, "bounces": 4},
            "tonemap": "neutral",
            "exposure": 0.0,
            "width": width,
            "height": height,
        }
    if BACKEND == "gpu":
        return {
            "backend": "gpu",
            "shadows": {"cascades": 2, "distance": 30.0, "map_size": 4096},
            "tonemap": "neutral",
            "msaa": 4,
            "width": width,
            "height": height,
        }
    return {"backend": "cpu", "tonemap": "neutral", "width": width, "height": height}


# --------------------------------------------------------------------------- #
# The character and its clothes
# --------------------------------------------------------------------------- #


CHARACTER_VOXEL = 0.007  # the figure's mesh at full quality: 7 mm, remeshed to 7 mm edges
QUICK_VOXEL = 0.02  # ... and for quick runs (marching cubes only: no fingers resolved)


def character_mesh(quick: bool = False, **options: Any):
    """The showcases' character: the anatomical figure (``figure_mesh``), remeshed to even
    7 mm triangles at full quality (cleaner shading and deformation; about 43,000
    vertices, fingers rigged), marching cubes at 20 mm for quick runs."""
    from aether3d.sculpt.figure import figure_mesh

    if quick:
        return figure_mesh(QUICK_VOXEL, **options)
    return figure_mesh(CHARACTER_VOXEL, remesh=CHARACTER_VOXEL, **options)


def waist() -> float:
    """The figure's waistline (metres): where a jacket overlaps the trousers."""
    from aether3d.sculpt.figure import FIGURE_DEFAULTS, figure_joints

    return float(figure_joints()["pelvis"][2] + 0.04 * FIGURE_DEFAULTS["height"])


def garment(parts: tuple[str, ...], by: float, quick: bool = False, **options: Any):
    """Clothing or armour over the figure's ``parts`` (``aether3d.sculpt.figure.PARTS``):
    their shapes inflated by ``by`` metres (``top``/``bottom``: cut flat there)."""
    from aether3d.sculpt.figure import figure_garment

    return figure_garment(parts, by, 0.024 if quick else 0.009, **options)


def armour(body_mesh: Any, quick: bool = False) -> dict[str, Any]:
    """A helmet (a shell over the figure's cranium, open at the face), shoulder plates over
    the shoulders and a backpack on the measured back -- meshes placed from the figure's
    joints and its surface."""
    from aether3d.sculpt.figure import FIGURE_DEFAULTS, figure_joints
    from aether3d.sculpt.implicit import implicit_surface

    voxel = 0.024 if quick else 0.009
    J = figure_joints()
    H = FIGURE_DEFAULTS["height"]
    hr = 0.13 * H * FIGURE_DEFAULTS["head"]  # (the figure's head height, chin to crown)
    top = J["head_top"][2]
    out = {
        "Helmet": implicit_surface([
            {"type": "ellipsoid", "center": [0.0, 0.012 * hr, top - 0.40 * hr],
             "radii": [0.40 * hr, 0.47 * hr, 0.47 * hr]},
            {"type": "box", "center": [0.0, -0.30 * hr, top - 0.72 * hr],
             "size": [0.62 * hr, 0.5 * hr, 0.42 * hr], "subtract": True, "blend": 0.01},
            {"type": "box", "center": [0.0, 0.0, top - 0.98 * hr],
             "size": [hr, hr, 0.3 * hr], "subtract": True, "blend": 0.01},
        ], voxel),
    }  # fmt: skip
    for side, x in (("L", 1.0), ("R", -1.0)):
        sh = J[f"shoulder.{side}"]
        out[f"Shoulder plate.{side}"] = implicit_surface(
            [{"type": "ellipsoid", "center": [float(sh[0] + 0.012 * x), float(sh[1]),
                                              float(sh[2] + 0.035)],
              "radii": [0.075, 0.085, 0.045]}], voxel)  # fmt: skip
    p = np.asarray(body_mesh.positions, np.float64)
    z = float(J["upper_chest"][2]) - 0.08
    band = (np.abs(p[:, 2] - z) < 0.06) & (np.abs(p[:, 0]) < 0.1)
    back = float(p[band, 1].max())  # (the back's surface behind the upper chest)
    out["Backpack"] = implicit_surface(
        [{"type": "box", "center": [0.0, back + 0.075, z], "size": [0.3, 0.13, 0.38],
          "blend": 0.04}], voxel)  # fmt: skip
    return out


def humanoid_mesh(voxel: float = 0.012, **shape: Any):
    """The implicit humanoid recipe meshed at ``voxel`` metres (no rig; v0.19's showcase
    character, kept for the revision's Blender experiment)."""
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    return implicit_surface(humanoid_shapes(**shape), voxel)


def garment_mesh(indices: tuple[int, ...], by: float, voxel: float, **shape: Any):
    """Clothing: shapes of the humanoid recipe (by index) inflated by ``by`` metres."""
    from aether3d.sculpt.implicit import humanoid_shapes, implicit_surface

    shapes = humanoid_shapes(**shape)
    out = []
    for i in indices:
        sh = copy.deepcopy(shapes[i])
        for key in ("radius", "radius_b"):
            if key in sh:
                sh[key] += by
        if "radii" in sh:
            sh["radii"] = [r + by for r in sh["radii"]]
        out.append(sh)
    return implicit_surface(out, voxel)


def materials(scene: Scene) -> dict[str, Any]:
    out = {}
    for name, (color, rough) in PALETTE.items():
        existing = scene.materials.find_material(name)
        out[name] = existing or scene.create_material(name, base_color=color, roughness=rough)
    return out


def tailor(scene: Scene, body: Any) -> dict[str, Any]:
    """Cut the unrigged body's surface along its clothes' edges -- the waistline, the
    cuffs, the collar, the shoes' tops and the hairline, placed from the detected
    landmarks -- so the clothes meet along clean lines instead of a staircase of whole
    triangles: vertices within a quarter of an edge of a line move onto it (along the
    surface: the lines cross it), triangles crossing it split. Call it before rigging
    (the rig binds the cut mesh); returns the edges for :func:`dress`."""
    from aether3d.mesh import Mesh

    lm = scene.animation.detect_landmarks(body)["landmarks"]
    inv = np.linalg.inv(body.world_matrix)

    def at(name: str) -> np.ndarray:
        return (inv @ np.append(np.asarray(lm[name]["position"], float), 1.0))[:3]

    up = np.array([0.0, 0.0, 1.0])
    edges: dict[str, Any] = {"waist": float(at("pelvis")[2] + 0.07)}
    cuts = [(up, np.array([0.0, 0.0, edges["waist"]]), at("pelvis") * [1, 1, 0]
             + [0, 0, edges["waist"]], 0.26)]  # fmt: skip
    for side in ("L", "R"):
        ankle, wrist, elbow = at(f"ankle.{side}"), at(f"wrist.{side}"), at(f"elbow.{side}")
        axis = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        edges[f"ankle.{side}"], edges[f"wrist.{side}"], edges[f"axis.{side}"] = ankle, wrist, axis
        top = ankle + [0.0, 0.0, 0.025]
        cuts += [(up, top, top, 0.2), (axis, wrist - 0.01 * axis, wrist, 0.25)]
    neck, head = at("neck"), at("head")
    edges["neck"], edges["crown"] = neck, head + np.array([0.0, 0.0, 0.07])
    # (on the neck itself, above where the shoulders' gentle slope would graze the line)
    collar = neck + [0.0, 0.0, 0.04]
    hairline = np.array([0.0, 0.75, 1.0]) / 1.25
    cuts += [(up, collar, collar, 0.16), (hairline, edges["crown"], edges["crown"], 0.2)]
    m = body.mesh
    attrs = {"p": m.positions.astype(np.float64), "n": m.normals.astype(np.float64)}
    if m.uvs is not None:
        attrs["uv"] = m.uvs.astype(np.float64)
    tris = np.asarray(m.triangles, np.int64)
    e = attrs["p"][tris] - attrs["p"][np.roll(tris, -1, axis=1)]
    snap = 0.25 * float(np.median(np.linalg.norm(e, axis=2)))
    for normal, point, centre, radius in cuts:
        attrs, tris = _cut(attrs, tris, normal, point, centre, radius, snap)
    body.mesh = Mesh(attrs["p"], tris, attrs["n"], attrs.get("uv"))
    edges["vertices_added"] = int(len(attrs["p"]) - len(m.positions))
    return edges


def _cut(attrs: dict, tris: np.ndarray, normal, point, centre, radius: float, snap: float):
    """Split the triangles near ``centre`` that cross the plane (``normal``, ``point``)."""
    p = attrs["p"]
    f = (p - point) @ normal
    near_v = np.linalg.norm(p - centre, axis=1) < radius
    move = near_v & (np.abs(f) < snap) & (f != 0.0)
    attrs["p"] = p = p.copy()
    p[move] -= f[move, None] * normal
    f = np.where(move, 0.0, f)
    sign = np.sign(f).astype(np.int64)
    s = sign[tris]
    near_t = np.linalg.norm(p[tris].mean(axis=1) - centre, axis=1) < radius
    crossing = near_t & (s.max(axis=1) > 0) & (s.min(axis=1) < 0)
    new: dict[tuple[int, int], int] = {}
    extra: dict[str, list] = {k: [] for k in attrs}
    count = len(p)

    def split(a: int, b: int) -> int:
        nonlocal count
        key = (min(a, b), max(a, b))
        if key not in new:
            t = f[a] / (f[a] - f[b])
            for k, v in attrs.items():
                x = v[a] + t * (v[b] - v[a])
                extra[k].append(x / np.linalg.norm(x) if k == "n" else x)
            new[key] = count
            count += 1
        return new[key]

    out = []
    for tri in tris[crossing]:
        sg = sign[tri]
        if (sg == 0).any():  # one corner on the plane: split the opposite edge
            r = int(np.flatnonzero(sg == 0)[0])
            z, a, b = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
            x = split(a, b)
            out += [(z, a, x), (z, x, b)]
        else:  # one corner alone on its side: a triangle and a quad
            r = int(np.flatnonzero(sg != np.sign(sg.sum()))[0])
            lone, a, b = tri[r], tri[(r + 1) % 3], tri[(r + 2) % 3]
            x, y = split(lone, a), split(lone, b)
            out += [(lone, x, y), (x, a, b), (x, b, y)]
    for k in attrs:
        if extra[k]:
            attrs[k] = np.concatenate([attrs[k], np.asarray(extra[k])])
    tris = np.concatenate([tris[~crossing], np.asarray(out, np.int64).reshape(-1, 3)])
    return attrs, tris


def dress(scene: Scene, body: Any, armature: Any, edges: dict[str, Any]) -> dict[str, int]:
    """Clothes by face group along the edges :func:`tailor` cut: trousers below the
    waistline, a top with sleeves to the cuffs and a collar at the neck, shoes below the
    ankles, a hair cap on the head -- so bends and twists read clearly. Each triangle
    takes the clothing its centre lies in (the arms, by their skin weights, are sleeves
    wherever they hang)."""
    from aether3d.mesh import Mesh

    a = scene.animation
    rig = a.rig(armature)
    m = body.mesh
    tris = np.asarray(m.triangles)
    q = m.positions.astype(np.float64)[tris].mean(axis=1)
    skin = a._owned[("skin", body._id)]._definition
    dominant = np.asarray(skin.indices)[np.arange(skin.vertex_count),
                                        np.argmax(skin.weights, axis=1)]  # fmt: skip
    arm_roles = ("shoulder", "upper_arm", "lower_arm", "hand")
    on_arm = np.array([
        (rig.joints[rig.index_of_id(j)].role or rig.joints[rig.index_of_id(j)].name.lower())
        .startswith(arm_roles) or "arm" in rig.joints[rig.index_of_id(j)].name.lower()
        for j in skin.joints
    ])[dominant][tris].sum(axis=1) >= 2  # fmt: skip
    kinds = list(PALETTE)
    groups = np.full(len(tris), kinds.index("top"))
    groups[(q[:, 2] < edges["waist"]) & ~on_arm] = kinds.index("trousers")
    for side in ("L", "R"):
        ankle, wrist, axis = edges[f"ankle.{side}"], edges[f"wrist.{side}"], edges[f"axis.{side}"]
        foot = (np.abs(q[:, 0] - ankle[0]) < 0.12) & (q[:, 2] < ankle[2] + 0.025)
        groups[foot] = kinds.index("shoes")
        hand = ((q - wrist) @ axis > -0.01) & (np.linalg.norm(q - wrist, axis=1) < 0.25)
        groups[hand] = kinds.index("skin")
    neck, crown = edges["neck"], edges["crown"]
    groups[(q[:, 2] > neck[2] + 0.04) & (np.abs(q[:, 0] - neck[0]) < 0.14)] = kinds.index("skin")
    hair = ((q[:, 2] - crown[2]) + 0.75 * (q[:, 1] - crown[1]) > 0.0) & (
        np.linalg.norm(q - crown, axis=1) < 0.2
    )
    groups[hair] = kinds.index("hair")
    body.mesh = Mesh(m.positions, m.triangles, m.normals, m.uvs, groups=groups.astype(np.uint16))
    mats = materials(scene)
    for k, name in enumerate(kinds):
        scene.assign_material(body, mats[name], group=k)
    return {name: int((groups == k).sum()) for k, name in enumerate(kinds)}


def cyclorama(
    scene: Scene,
    *,
    width: float = 120.0,
    depth: float = 40.0,
    height: float = 30.0,
    radius: float = 5.0,
    back: float = 8.0,
) -> Any:
    """A seamless studio backdrop: a floor curving up into a wall behind the subject (no
    horizon line), modeled as a profile swept along X."""
    from aether3d.mesh import Mesh

    prof = [(y, 0.0) for y in np.linspace(-depth, back - radius, 24)]
    for t in np.linspace(0.0, np.pi / 2, 24)[1:]:
        prof.append((back - radius + radius * np.sin(t), radius - radius * np.cos(t)))
    prof += [(back, z) for z in np.linspace(radius, height, 12)[1:]]
    xs = np.linspace(-width / 2, width / 2, 49)
    pos = np.array([(x, y, z) for (y, z) in prof for x in xs], np.float32)
    # Smooth normals across the curve (the profile's own normal at each point).
    yz = np.array(prof)
    tangent = np.gradient(yz, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    nrm_yz = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)  # (toward the subject)
    nrm = np.array([(0.0, ny, nz) for (ny, nz) in nrm_yz for _ in xs], np.float32)
    nx = len(xs)
    tris = []
    for i in range(len(prof) - 1):
        for j in range(nx - 1):
            a0, a1 = i * nx + j, i * nx + j + 1
            b0, b1 = (i + 1) * nx + j, (i + 1) * nx + j + 1
            tris += [(a0, a1, b1), (a0, b1, b0)]
    obj = scene.add_object("Cyclorama", mesh=Mesh(pos, np.array(tris, np.int64), nrm))
    obj.material = scene.create_material("Backdrop", base_color=(0.36, 0.36, 0.38),
                                         roughness=0.85)  # fmt: skip
    return obj


# --------------------------------------------------------------------------- #
# Lighting, cameras, renders
# --------------------------------------------------------------------------- #


def studio(scene: Scene, *, floor: bool = True, key: float = 2.6) -> dict[str, Any]:
    """Key, fill and rim lights, cool ambient light, a soft sky for the path tracer and a
    matte floor. Returns the setup for the report."""
    scene.world.background = (0.11, 0.12, 0.14)
    scene.world.ambient_color = (0.62, 0.7, 0.85)
    scene.world.ambient_intensity = 0.22
    scene.world.environment.configure(
        type="sky",
        intensity=0.35,
        zenith_color=(0.42, 0.5, 0.66),
        horizon_color=(0.3, 0.32, 0.36),
        ground_color=(0.12, 0.11, 0.1),
    )
    lights = {
        "key": ((0.45, 0.5, -0.74), (1.0, 0.92, 0.82), key, 4.0),
        "fill": ((-0.78, 0.36, -0.32), (0.72, 0.82, 1.0), 0.5 * key / 2.6, 8.0),
        "rim": ((-0.22, -0.72, -0.66), (0.86, 0.9, 1.0), 1.6 * key / 2.6, 2.0),
    }
    for name, (direction, color, intensity, angle) in lights.items():
        light = scene.add_directional_light(name.title(), direction=direction, intensity=intensity)
        light.light.configure(color=color, angle=angle)
    if floor:
        cyclorama(scene)
    return {
        "lights": {
            k: {"direction": v[0], "color": v[1], "intensity": round(v[2], 3), "angle_deg": v[3]}
            for k, v in lights.items()
        },
        "ambient": {"color": (0.62, 0.7, 0.85), "intensity": 0.22},
        "environment": "sky (path tracer)",
        "set": "cyclorama (a floor curving into a back wall)",
    }


def still(
    scene: Scene,
    path: Path,
    *,
    camera: Any,
    time: float | None = None,
    width: int = 1920,
    height: int = 1080,
    path_samples: int = 0,
) -> dict[str, Any]:
    """One rendered still (the GPU path tracer with ``path_samples`` when it works, else
    the rasterizer); returns the timing and the backend used."""
    options = render_options(width=width, height=height, path_samples=path_samples)
    if time is not None:
        options["time"] = time
    t0 = time_now()
    r = scene.render(camera=camera, **options)
    seconds = time_now() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    r.save(path)
    return {
        "path": str(path),
        "seconds": round(seconds, 2),
        "backend": r.backend.get("name"),
        "size": [width, height],
    }


def render_frames(
    scene: Scene,
    frames_dir: Path,
    *,
    camera: Any,
    fps: int,
    start: float,
    end: float,
    width: int,
    height: int,
    first: int = 0,
    progress=None,
) -> dict[str, Any]:
    """Every frame of ``[start, end)`` at ``fps`` as PNG, numbered from ``first`` (so
    segments append to one sequence); per-frame seconds."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = int(round((end - start) * fps))
    options = render_options(width=width, height=height)
    secs = []
    for k in range(count):
        t0 = time_now()
        r = scene.render(camera=camera, time=start + k / fps, **options)
        secs.append(time_now() - t0)
        r.save(frames_dir / f"frame_{first + k:05d}.png")
        if progress is not None:
            progress(k + 1, count)
    secs.sort()
    return {
        "frames": count,
        "first": first,
        "fps": fps,
        "start": start,
        "end": end,
        "seconds": round(sum(secs), 1),
        "seconds_per_frame_median": round(secs[len(secs) // 2], 3) if secs else None,
        "size": [width, height],
    }


def encode(
    frames_dir: Path, out_dir: Path, name: str, fps: int, *, size=(1920, 1080), master: bool = True
) -> dict[str, Any]:
    """``<name>_1080p.mp4`` (libx264, preset slow, CRF 10, yuv420p, BT.709, faststart: the
    frames' own rate, no frame repeated), ``<name>_master.mkv`` (FFV1, lossless RGB),
    ``<name>_preview.mp4`` (960x540, CRF 23); PSNR/SSIM of the encodes against the frames."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        return {"skipped": "ffmpeg is not installed"}
    pattern = str(frames_dir / "frame_%05d.png")
    rate = str(fps)
    color = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]

    def x264(crf: int, w: int, h: int, out: Path) -> list[str]:
        return [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            rate,
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            f"scale={w}:{h}:flags=lanczos:out_color_matrix=bt709:out_range=tv",
            *color,
            "-movflags",
            "+faststart",
            str(out),
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{name}_1080p.mp4"
    commands = {
        "mp4": x264(10, size[0], size[1], mp4),
        "preview": x264(23, 960, 540, out_dir / f"{name}_preview.mp4"),
    }
    if master:
        commands["master"] = [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            rate,
            "-i",
            pattern,
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            str(out_dir / f"{name}_master.mkv"),
        ]
    out: dict[str, Any] = {"commands": {k: " ".join(v[1:]) for k, v in commands.items()}}
    for key, cmd in commands.items():
        t0 = time_now()
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, check=False)
        if run.returncode != 0:
            raise RuntimeError(f"ffmpeg {key} failed: {run.stderr[-600:]}")
        target = Path(cmd[-1])
        out[key] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "seconds": round(time_now() - t0, 1),
        }
    out["verify"] = verify(mp4, pattern, rate)
    if master:
        out["verify_master"] = verify(out_dir / f"{name}_master.mkv", pattern, rate)
    return out


def sheet(
    images: list[Any],
    out: Path,
    *,
    labels: list[str] | None = None,
    columns: int = 4,
    width: int = 480,
) -> str:
    """A contact sheet of rendered images (arrays or PNG paths), drawn by Aether3D."""
    from aether3d.animation.frames import contact_sheet
    from aether3d.materials.images import decode_png

    tiles = []
    for im in images:
        if isinstance(im, (str, Path)):
            im = decode_png(Path(im).read_bytes()).pixels[..., :3]
        im = np.asarray(im)[..., :3]
        tiles.append(downscale(im, width) if im.shape[1] > width else im)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        encode_png(contact_sheet(tiles, columns=columns, labels=labels), text=software())
    )
    return str(out)


def save(path: Path, pixels: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(np.asarray(pixels)[..., :3], text=software()))
    return str(path)


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def machine() -> dict[str, Any]:
    import aether3d
    from aether3d import _native

    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu": cpu,
        "aether3d": aether3d.__version__,
        "python": platform.python_version(),
        "native": _native.status().get("enabled"),
        "backend": BACKEND,
    }


def write_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_portable(data), indent=2) + "\n", encoding="utf-8")


def _portable(value: Any) -> Any:
    root = str(HERE.parent)
    if isinstance(value, dict):
        return {str(k): _portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(v) for v in value]
    if isinstance(value, str) and root in value:
        return value.replace(root + os.sep, "").replace(root, ".")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def time_now() -> float:
    return time.perf_counter()


__all__ = [
    "OUT",
    "PALETTE",
    "dress",
    "encode",
    "garment_mesh",
    "humanoid_mesh",
    "machine",
    "materials",
    "preview",
    "progress_printer",
    "render_frames",
    "render_options",
    "save",
    "set_backend",
    "sheet",
    "still",
    "studio",
    "tailor",
    "write_report",
]
