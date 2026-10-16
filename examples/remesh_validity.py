"""Remeshing with validity metrics: from a broken, uneven model to a clean surface, by Aether3D.

Run from the repository root:

    python examples/remesh_validity.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

The input is what a quick block-out often is: a ring (a torus), a ball and a peg pushed
into each other -- three closed parts whose surfaces intersect -- triangulated unevenly
(a dense ball, long thin triangles on the ring). Each step is measured, not assumed,
with :func:`aether3d.sculpt.check_mesh` (the exact self-intersection test included):

1. Dynamic topology: a few strokes with ``dyntopo`` refine the ball locally.
2. ``voxel_remesh``: one closed surface -- the parts fuse, the buried surfaces are gone,
   no self-intersection remains (tier ``solid``); the volume is compared with the
   union's volume estimated independently (Monte Carlo with an exact inside test).
3. ``remesh`` (isotropic): even edges near the target length, angles near 60 degrees,
   most vertices of valence 6; the surface stays on the voxel surface.

``remesh_validity.png`` shows the three stages with each triangle coloured by its
smallest angle (green >= 30, yellow 20-30, red < 20 degrees); ``remesh_validity.json``
holds every metric. Exits with status 1 if a check fails.
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
from aether3d.sculpt import SculptMesh, check_mesh
from aether3d.spatial.geometry import component_labels
from aether3d.spatial.surface import TriangleSurface

HERE = Path(__file__).resolve().parent


def render(scene, camera, width: int, height: int, backend: str):
    """A render with sun shadows where the backend draws them (the GPU rasterizer)."""
    from aether3d import RenderError

    options = {"width": width, "height": height, "camera": camera, "backend": backend}
    if backend != "cpu":
        try:
            return scene.render(**options, shadows=True)
        except RenderError as exc:
            if exc.code != "BACKEND_UNSUPPORTED":
                raise
    return scene.render(**options)


def torus(major: float, minor: float, nu: int, nv: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.arange(nu) * 2 * np.pi / nu
    v = np.arange(nv) * 2 * np.pi / nv
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ring = major + minor * np.cos(vv)
    p = np.stack([ring * np.cos(uu), ring * np.sin(uu), minor * np.sin(vv)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a, b = i * nv + j, ((i + 1) % nu) * nv + j
    c, d = ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv
    t = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)]).reshape(-1, 3)
    return p, t


def blockout() -> Mesh:
    """A ring (coarse and stretched), a dense ball and a peg, overlapping."""
    ring_p, ring_t = torus(1.0, 0.32, 20, 10)  # long thin triangles
    ball = create_sphere(0.62, 96, 48)  # dense
    peg = create_sphere(1.0, 24, 12)
    peg_p = peg.positions.astype(np.float64) * [0.25, 0.25, 0.7] + [0.0, -0.85, 0.45]
    parts = [
        (ring_p, ring_t),
        (ball.positions.astype(np.float64) + [0.75, 0.25, 0.15], ball.triangles),
        (peg_p, peg.triangles),
    ]
    positions, triangles, base = [], [], 0
    for p, t in parts:
        positions.append(p)
        triangles.append(t + base)
        base += len(p)
    return Mesh(np.concatenate(positions).astype(np.float32), np.concatenate(triangles))


def metrics(mesh: Mesh) -> dict:
    """Validity and quality, measured on the mesh as it is."""
    report = check_mesh(mesh.positions, mesh.triangles, self_intersections=True)
    s = report.summary
    p = mesh.positions.astype(np.float64)
    t = mesh.triangles
    angles = triangle_min_angles(p, t)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    length = np.linalg.norm(p[e[:, 0]] - p[e[:, 1]], axis=1)
    valence = np.bincount(e.ravel(), minlength=len(p))
    labels = component_labels(t, len(p))
    return {
        "tier": s["tier"],
        "triangles": s["triangle_count"],
        "components": int(labels.max()) + 1,
        "self_intersecting_pairs": s["self_intersecting_pairs"],
        "volume": s.get("volume"),
        "min_angle_degrees": round(float(angles.min()), 2),
        "triangles_below_20_degrees": round(float((angles < 20).mean()), 4),
        "triangles_below_30_degrees": round(float((angles < 30).mean()), 4),
        "edge_length_mean": round(float(length.mean()), 4),
        "edge_length_cv": round(float(length.std() / length.mean()), 4),
        "valence_6": round(float((valence[valence > 0] == 6).mean()), 4),
    }


def triangle_min_angles(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full(len(t), 180.0)
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u, v = p[t[:, j]] - p[t[:, i]], p[t[:, k]] - p[t[:, i]]
        cos = (u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
        out = np.minimum(out, np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out


def union_volume(parts: list[TriangleSurface], lo, hi, samples: int, seed: int = 7) -> float:
    """Monte Carlo volume of the union of closed parts. A point is inside a part when the
    angle-weighted pseudo-normal of the feature holding its nearest point (face, edge or
    vertex) points away from it (Baerentzen and Aanaes 2005: exact for closed manifold
    parts). Independent of the voxel grid, the narrow band and marching cubes."""
    from aether3d.sculpt.voxel import pseudonormals

    rng = np.random.default_rng(seed)
    q = lo + rng.random((samples, 3)) * (hi - lo)
    inside = np.zeros(samples, dtype=bool)
    for surface in parts:
        t = surface.triangles
        face, edge, vertex = pseudonormals(surface.positions, t)
        near = surface.nearest(q, batch=1 << 14)
        tri, bary = near.triangles, near.bary
        nonzero = bary > 1e-12
        kind = nonzero.sum(axis=1)
        normal = face[tri]
        on_edge = kind == 2
        side = (np.argmin(nonzero[on_edge], axis=1) + 1) % 3
        normal[on_edge] = edge[tri[on_edge], side]
        at_vertex = kind == 1
        normal[at_vertex] = vertex[t[tri[at_vertex], np.argmax(nonzero[at_vertex], axis=1)]]
        inside |= ((q - near.points) * normal).sum(axis=1) < 0
    return float(inside.mean() * np.prod(hi - lo))


def quality_mesh(mesh: Mesh) -> Mesh:
    """The mesh with face groups by smallest angle: 0 good, 1 fair, 2 poor (flat shaded,
    so every triangle shows)."""
    angles = triangle_min_angles(mesh.positions.astype(np.float64), mesh.triangles)
    groups = np.where(angles >= 30, 0, np.where(angles >= 20, 1, 2)).astype(np.int64)
    p = mesh.positions[mesh.triangles].reshape(-1, 3)  # unshared corners: flat shading
    t = np.arange(len(p)).reshape(-1, 3)
    return Mesh(p, t, groups=groups)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=str(HERE / "output"))
    parser.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    parser.add_argument("--quick", action="store_true", help="coarser grids, smaller images")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    width, height = (360, 300) if args.quick else (720, 600)
    voxel, target, samples = (0.05, 0.06, 100_000) if args.quick else (0.025, 0.03, 400_000)
    failures: list[str] = []
    stages: dict[str, dict] = {}
    timings: dict[str, float] = {}

    source = blockout()
    stages["input"] = metrics(source)
    sm = SculptMesh.from_mesh(source)
    # 1. Dynamic topology on the ball.
    t0 = time.perf_counter()
    for path in ([[0.75, 0.25, 0.77], [1.2, 0.2, 0.5]], [[1.37, 0.25, 0.15], [1.2, -0.2, -0.2]]):
        sm.stroke(
            {
                "brush": {
                    "kind": "clay",
                    "radius": 0.25,
                    "strength": 0.3,
                    "dyntopo": {"detail": 0.03, "mode": "constant"},
                },
                "path": path,
            }
        )
    timings["dyntopo"] = time.perf_counter() - t0
    stroked = sm.to_mesh()
    stages["dyntopo"] = metrics(stroked)
    # 2. Voxel remesh: fuse.
    t0 = time.perf_counter()
    r = sm.voxel_remesh(voxel)
    timings["voxel_remesh"] = time.perf_counter() - t0
    fused = sm.to_mesh()
    stages["voxel"] = metrics(fused)
    stages["voxel"]["grid"] = r.data["grid"]
    # 3. Isotropic remesh.
    t0 = time.perf_counter()
    r = sm.remesh(target, iterations=5)
    timings["isotropic_remesh"] = time.perf_counter() - t0
    final = sm.to_mesh()
    stages["isotropic"] = metrics(final)
    stages["isotropic"]["operations"] = {
        k: r.data[k] for k in ("splits", "collapses", "flips", "relaxed")
    }

    # Independent checks.
    p0 = stroked.positions.astype(np.float64)
    labels = component_labels(stroked.triangles, len(p0))[stroked.triangles[:, 0]]
    parts = [TriangleSurface(p0, stroked.triangles[labels == k]) for k in range(3)]
    lo, hi = p0.min(axis=0) - 0.05, p0.max(axis=0) + 0.05
    estimate = union_volume(parts, lo, hi, samples)
    stderr = np.prod(hi - lo) * np.sqrt(estimate / np.prod(hi - lo) / samples)
    stages["union_volume_monte_carlo"] = {"volume": estimate, "standard_error": float(stderr)}
    if stages["input"]["self_intersecting_pairs"] == 0 or stages["input"]["components"] != 3:
        failures.append("the input should be three intersecting parts")
    for name in ("voxel", "isotropic"):
        st = stages[name]
        if st["tier"] != "solid" or st["components"] != 1:
            failures.append(f"{name}: {st['tier']} with {st['components']} component(s)")
        if abs(st["volume"] - estimate) > 4 * stderr + 0.02 * estimate:
            failures.append(f"{name}: volume {st['volume']:.4f}, union {estimate:.4f}")
    iso = stages["isotropic"]
    if iso["min_angle_degrees"] < 15 or iso["triangles_below_30_degrees"] > 0.02:
        failures.append(f"isotropic remesh quality: {iso}")
    near = TriangleSurface(fused.positions, fused.triangles).nearest(final.positions)
    stages["isotropic"]["distance_to_voxel_surface_max"] = float(near.distance.max())
    if near.distance.max() > voxel:
        failures.append(f"the remesh left the surface by {near.distance.max():.4f}")

    # Pictures: the stages side by side, coloured by triangle quality.
    scene = Scene("Remesh validity")
    colours = {0: "#3fae5a", 1: "#e3c341", 2: "#d9483b"}
    materials = {
        k: scene.create_material(f"q{k}", base_color=c, roughness=0.6) for k, c in colours.items()
    }
    images = []
    for name, mesh in (("input", stroked), ("voxel", fused), ("isotropic", final)):
        obj = scene.add_object(name, mesh=quality_mesh(mesh))
        for k, m in materials.items():
            scene.assign_material(obj, m, group=k)
    scene.add_directional_light("Key", direction=(0.4, 0.6, -1.0), intensity=2.2)
    scene.add_directional_light("Fill", direction=(-0.7, -0.2, -0.4), intensity=0.7)
    for name in ("input", "voxel", "isotropic"):
        for other in ("input", "voxel", "isotropic"):
            scene.get_by_name(other).visible = other == name
        camera = scene.add_camera(f"Cam {name}", position=(1.6, -3.2, 2.4), look_at=(0.3, 0, 0))
        camera.frame([name], margin=0.06, aspect=width / height)
        images.append(render(scene, camera, width, height, args.backend))
    png = encode_png(
        np.concatenate([im.image for im in images], axis=1),
        text={"Software": f"Aether3D {__version__}"},
    )
    (out / "remesh_validity.png").write_bytes(png)
    report = {
        "aether3d": __version__,
        "quick": args.quick,
        "voxel_size": voxel,
        "target_length": target,
        "stages": stages,
        "seconds": {k: round(v, 3) for k, v in timings.items()},
        "render": {"backend": images[0].backend, "size": [width, height]},
        "failures": failures,
    }
    (out / "remesh_validity.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    header = f"{'stage':<10} {'tier':<9} {'tris':>7} {'parts':>5} {'SI':>5} {'min angle':>9} "
    print(header + f"{'<30 deg':>8} {'edge cv':>8} {'val 6':>6}")
    for name in ("input", "dyntopo", "voxel", "isotropic"):
        st = stages[name]
        print(
            f"{name:<10} {st['tier']:<9} {st['triangles']:>7} {st['components']:>5} "
            f"{st['self_intersecting_pairs']:>5} {st['min_angle_degrees']:>9.2f} "
            f"{st['triangles_below_30_degrees']:>8.2%} {st['edge_length_cv']:>8.3f} "
            f"{st['valence_6']:>6.1%}"
        )
    print(
        f"union volume (Monte Carlo) {estimate:.4f} +- {stderr:.4f}; voxel "
        f"{stages['voxel']['volume']:.4f}, isotropic {stages['isotropic']['volume']:.4f}"
    )
    print(", ".join(f"{k} {v:.2f} s" for k, v in timings.items()))
    print(f"wrote {out / 'remesh_validity.png'} and {out / 'remesh_validity.json'}")
    for failure in failures:
        print("FAILED:", failure, file=sys.stderr)
    if failures:
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
