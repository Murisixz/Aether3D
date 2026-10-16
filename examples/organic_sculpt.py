"""Organic sculpting with symmetry: a creature bust, from primitives to a sculpt, by Aether3D.

Run from the repository root:

    python examples/organic_sculpt.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

1. Block-out: a head, a snout and a neck (closed primitives, overlapping) in one mesh.
2. ``voxel_remesh`` fuses them into one closed surface (the buried parts disappear), and
   ``remesh`` (isotropic) evens out its triangles -- a base without poles or slivers.
3. Sculpt in one session (one undo step), with ``symmetry: "x"`` on the strokes: a brow
   ridge (``draw``), eye sockets (``draw`` inverted), cheekbones (``clay``), a nose
   bridge (``draw`` along the mirror plane: ``inflate`` would push the walls of the
   valley between head and snout into each other), a mouth line (``crease``), the chin
   pulled down (``grab``), pressure tapering each stroke. A mask keeps the neck's base still.
4. A horn sculpted on one side only (``snake_hook`` with dynamic topology, which adds
   triangles as the horn stretches), then ``symmetrize`` copies it to the other side.
5. ``smooth`` (``relax``: even spacing without shrinking) over the face.
6. Checks: the result is ``solid`` (closed, manifold, outward, no self-intersection:
   the exact test), mirror-symmetric (its mirror image lies on itself, measured by exact
   nearest points), and one undo restores the block-out.

Writes ``organic_sculpt.png`` (block-out, base, sculpt) and ``organic_sculpt.json``
(every operation with its dabs, time and result; symmetry and validity measured) to
OUTPUT_DIR. Exits with status 1 if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from aether3d import Mesh, Scene, __version__, create_sphere
from aether3d.rendering.png import encode_png
from aether3d.sculpt.validate import check_mesh
from aether3d.spatial.surface import TriangleSurface

HERE = Path(__file__).resolve().parent


def render(scene, camera, width: int, height: int, backend: str):
    """A render with sun shadows where the backend draws them (the GPU rasterizer); the
    CPU rasterizer has none."""
    from aether3d import RenderError

    options = {"width": width, "height": height, "camera": camera, "backend": backend}
    if backend != "cpu":
        try:
            return scene.render(**options, shadows=True)
        except RenderError as exc:
            if exc.code != "BACKEND_UNSUPPORTED":
                raise
    return scene.render(**options)


def blockout() -> Mesh:
    """A head, a snout and a neck: three closed parts that overlap."""

    def part(mesh: Mesh, scale, offset):
        scale = np.float32(scale)
        normals = mesh.normals / scale  # an ellipsoid's normals: by the inverse scale
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return mesh.positions * scale + np.float32(offset), normals, mesh.triangles

    sphere = create_sphere(1.0, 48, 24)
    parts = [
        part(sphere, (1.0, 1.0, 1.05), (0.0, 0.0, 0.0)),  # head
        part(sphere, (0.55, 0.6, 0.45), (0.0, -0.75, -0.35)),  # snout
        part(create_sphere(1.0, 32, 16), (0.45, 0.45, 0.8), (0.0, 0.15, -1.0)),  # neck
    ]
    positions, normals, triangles, base = [], [], [], 0
    for p, n, t in parts:
        positions.append(p)
        normals.append(n)
        triangles.append(t + base)
        base += len(p)
    return Mesh(
        np.concatenate(positions), np.concatenate(triangles), normals=np.concatenate(normals)
    )


def path(points, pressure=None):
    pressure = pressure or [1.0] * len(points)
    return [{"position": list(p), "pressure": q} for p, q in zip(points, pressure, strict=True)]


TAPER = [0.3, 0.8, 1.0, 0.8, 0.3]


def strokes() -> list[tuple[str, dict]]:
    """The symmetric strokes of step 3, as data (``(label, stroke)``)."""
    sym = {"symmetry": "x"}
    return [
        (
            "brow ridge",
            {
                "brush": {"kind": "draw", "radius": 0.22, "strength": 0.12, **sym},
                "path": path(
                    [
                        (0.05, -0.9, 0.42),
                        (0.2, -0.86, 0.47),
                        (0.35, -0.8, 0.46),
                        (0.5, -0.7, 0.42),
                        (0.62, -0.6, 0.36),
                    ],
                    TAPER,
                ),
            },
        ),
        (
            "eye sockets",
            {
                "brush": {"kind": "draw", "radius": 0.16, "strength": 0.35, "direction": -1, **sym},
                "path": [[0.32, -0.88, 0.22]],
            },
        ),
        (
            "cheekbones",
            {
                "brush": {"kind": "clay", "radius": 0.2, "strength": 0.25, **sym},
                "path": path(
                    [
                        (0.3, -0.85, 0.0),
                        (0.45, -0.75, 0.05),
                        (0.6, -0.62, 0.1),
                        (0.7, -0.5, 0.12),
                        (0.78, -0.38, 0.12),
                    ],
                    TAPER,
                ),
            },
        ),
        (
            "nose bridge",
            {
                "brush": {"kind": "draw", "radius": 0.14, "strength": 0.1, **sym},
                "path": path(
                    [(0.0, -1.0, 0.3), (0.0, -1.1, 0.0), (0.0, -1.25, -0.2)], [1, 0.8, 0.5]
                ),
            },
        ),
        (
            "mouth line",
            {
                "brush": {"kind": "crease", "radius": 0.08, "strength": 0.3, **sym},
                "path": path(
                    [
                        (0.0, -1.27, -0.47),
                        (0.15, -1.2, -0.47),
                        (0.3, -1.05, -0.45),
                        (0.42, -0.9, -0.4),
                        (0.5, -0.78, -0.36),
                    ],
                    TAPER,
                ),
            },
        ),
        (
            "chin",
            {
                "brush": {"kind": "grab", "radius": 0.3, "strength": 1.0, **sym},
                "path": [[0.0, -1.0, -0.72], [0.0, -1.08, -0.8]],
            },
        ),
    ]


HORN = {
    "brush": {
        "kind": "snake_hook",
        "radius": 0.2,
        "strength": 1.0,
        "dyntopo": {"detail": 0.05, "mode": "constant"},
    },
    "path": [[0.55, 0.0, 0.85], [0.68, 0.05, 1.05], [0.85, 0.12, 1.2], [1.05, 0.2, 1.28]],
}

SMOOTH = {
    "brush": {"kind": "smooth", "radius": 0.5, "strength": 0.5, "smooth_method": "relax",
              "symmetry": "x"},
    "path": [[0.0, -1.05, 0.2], [0.35, -0.9, 0.1], [0.6, -0.65, 0.0]],
}  # fmt: skip


def symmetry_error(mesh: Mesh) -> dict:
    """How far the mirror image of every vertex lies from the surface (exact nearest
    points): 0 for a mirror-symmetric shape."""
    p = mesh.positions.astype(np.float64)
    surface = TriangleSurface(p, mesh.triangles)
    mirrored = p * np.array([-1.0, 1.0, 1.0])
    d = surface.nearest(mirrored, batch=1 << 15).distance
    return {"max": float(d.max()), "mean": float(d.mean())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="coarser meshes, smaller images")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (320, 320) if args.quick else (640, 640)
    voxel, target = (0.06, 0.07) if args.quick else (0.035, 0.04)
    failures: list[str] = []
    log: list[dict] = []

    scene = Scene("Organic sculpt")
    head = scene.add_object("Creature", mesh=blockout())
    scene.assign_material(
        head, scene.create_material("Clay", base_color="#c49a74", roughness=0.75, specular=0.3)
    )
    ground = scene.add_plane("Ground", size=12, position=(0, 0, -1.8))
    scene.assign_material(ground, scene.create_material("Ground", base_color="#6f7378"))
    scene.add_directional_light("Key", direction=(0.5, 0.8, -0.7), intensity=2.4)
    scene.add_directional_light("Rim", direction=(-0.6, -0.4, -0.3), intensity=0.7)
    camera = scene.add_camera("Cam", position=(2.6, -4.2, 1.2), look_at=(0, 0, -0.1))
    camera.frame(["Creature"], margin=0.3, aspect=width / height)  # room for the horns

    def shot():
        return render(scene, camera, width, height, args.backend)

    first = shot()
    images = [first.image]
    blocked = head.mesh

    def record(label: str, result) -> None:
        entry = {"step": label, "operation": result.operation, "data": result.to_dict()["data"]}
        entry["ms"] = round(1000 * sum(result.timings.values()), 2) if result.timings else None
        entry["diagnostics"] = [d.code for d in result.diagnostics]
        log.append(entry)

    # 2. Fuse and even out the base (its own undo step).
    with head.sculpt() as session:
        t0 = time.perf_counter()
        r = session.voxel_remesh(voxel)
        record("fuse the block-out", r)
        log[-1]["seconds"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        r = session.remesh(target, iterations=4)
        record("even out the triangles", r)
        log[-1]["seconds"] = round(time.perf_counter() - t0, 3)
    base = head.mesh
    base_metrics = check_mesh(base.positions, base.triangles, self_intersections=True).summary
    images.append(shot().image)
    fused = log[0]["data"]
    if fused["input"]["components"] != 3 or fused["output"]["components"] != 1:
        failures.append(f"the voxel remesh did not fuse 3 parts into 1: {fused['output']}")

    # 3-5. Sculpt (one undo step).
    t_sculpt = time.perf_counter()
    with head.sculpt() as session:
        session.set_mask(1.0, box=((-2, -2, -3), (2, 2, -1.45)))
        for label, stroke in strokes():
            t0 = time.perf_counter()
            r = session.stroke(stroke)
            record(label, r)
            log[-1]["seconds"] = round(time.perf_counter() - t0, 3)
        r = session.stroke(HORN)
        record("horn, one side, dynamic topology", r)
        r = session.symmetrize("x", direction="+")
        record("symmetrize +x onto -x", r)
        r = session.stroke(SMOOTH)
        record("relax the face", r)
        session.clear_mask()
    sculpt_seconds = time.perf_counter() - t_sculpt
    sculpted = head.mesh
    images.append(shot().image)

    # 6. Checks.
    metrics = check_mesh(sculpted.positions, sculpted.triangles, self_intersections=True).summary
    if metrics["tier"] != "solid":
        failures.append(f"the sculpt is {metrics['tier']}, not solid")
    sym = symmetry_error(sculpted)
    size = float(np.ptp(sculpted.positions, axis=0).max())
    if sym["max"] > 0.02 * size:
        failures.append(f"the sculpt is not symmetric: {sym}")
    scene.history.undo()
    scene.history.undo()
    if head.mesh is not blocked:
        failures.append("two undos do not restore the block-out")
    scene.history.redo()
    scene.history.redo()
    if head.mesh is not sculpted:
        failures.append("two redos do not restore the sculpt")

    png = encode_png(np.concatenate(images, axis=1), text={"Software": f"Aether3D {__version__}"})
    (out / "organic_sculpt.png").write_bytes(png)
    keep = ("tier", "triangle_count", "vertex_count", "volume", "min_angle_degrees",
            "euler_characteristic", "self_intersecting_pairs")  # fmt: skip
    report = {
        "aether3d": __version__,
        "quick": args.quick,
        "operations": log,
        "base": {k: base_metrics.get(k) for k in keep},
        "sculpt": {k: metrics.get(k) for k in keep},
        "sculpt_seconds": round(sculpt_seconds, 3),
        "symmetry_error": sym,
        "render": {"backend": first.backend, "size": [width, height]},
        "failures": failures,
    }
    (out / "organic_sculpt.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    for entry in log:
        data = entry["data"]
        what = (
            f"{data.get('dabs')} dabs, {data.get('moved_vertices')} vertices moved"
            if "dabs" in data
            else ", ".join(f"{k} {v}" for k, v in data.items() if isinstance(v, (int, float)))
        )
        print(f"  {entry['step']:<36} {what}")
    print(
        f"base: {base_metrics['triangle_count']} triangles, sculpt: {metrics['triangle_count']} "
        f"triangles, {metrics['tier']}; symmetry error max {sym['max']:.2e}"
    )
    print(f"wrote {out / 'organic_sculpt.png'} and {out / 'organic_sculpt.json'}")
    for failure in failures:
        print("FAILED:", failure, file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
