"""Advanced modeling and sculpting benchmarks (Stage 13).

Run from the repository root:

    python benchmarks/bench_sculpt.py [--json OUT.json] [--label NAME] [--quick]
        [--only baseline,...] [--compare SAVED.json]

Every group runs in its own Python process, so memory peaks and caches of one group never
influence another; each process reports its peak RSS (``VmHWM`` on Linux). Timings are
medians of repeated runs after a warm-up, with Python's cyclic garbage collector paused
during each timed run (``bench_engine.median_time``).

Groups:

* ``baseline`` -- what the engine offered before Stage 13, on UV spheres of about 10k,
  100k and 1M vertices: opening a mesh for editing and committing it, a sculpt-like
  dab done with the editing API (``select_vertices(sphere=)`` + ``move_along_normals``,
  no falloff), Catmull-Clark subdivision, and closest-point queries on a 1M-triangle
  object.

Meshes are generated (``create_sphere``), so every run measures the same inputs.
Results depend on the machine: compare runs on the same machine only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from bench_engine import compare, machine, median_time, ms, peak_rss_mib  # noqa: E402

SIZES = {"10k": (70, 140), "100k": (224, 448), "1m": (707, 1414)}
"""Sphere ``(rings, segments)`` per size: (rings + 1) x (segments + 1) vertices."""


def sphere(size: str):
    from aether3d import create_sphere

    rings, segments = SIZES[size]
    return create_sphere(1.0, segments, rings)


# ---------------------------------------------------------------------------------------
# Groups


def group_baseline(quick: bool) -> dict[str, Any]:
    import numpy as np

    from aether3d import Mesh, Scene
    from aether3d.editing import EditableMesh
    from aether3d.procedural.modifiers import catmull_clark
    from aether3d.procedural.polymesh import PolyMesh

    out: dict[str, Any] = {}
    sizes = ("10k", "100k") if quick else ("10k", "100k")
    for size in sizes:
        mesh = sphere(size)
        out[f"vertices_{size}"] = mesh.vertex_count
        out[f"triangles_{size}"] = mesh.triangle_count
        repeat = 3 if size == "10k" else 1
        out[f"editable_from_mesh_{size}_s"] = round(
            median_time(lambda mesh=mesh: EditableMesh.from_mesh(mesh), repeat, warmup=0), 3
        )
        em = EditableMesh.from_mesh(mesh)
        out[f"editable_to_mesh_{size}_s"] = round(median_time(em.to_mesh, repeat, warmup=0), 3)

    # A "dab" with the editing API: the vertices in a sphere move along their normals
    # (uniformly: the API has no falloff), then the mesh is committed (to_mesh).
    em = EditableMesh.from_mesh(sphere("100k"))
    center, radius = (0.0, 0.0, 1.0), 0.1

    def dab():
        sel = em.select_vertices(sphere=(center, radius))
        em.move_along_normals(sel, 0.001)

    out["api_dab_100k_vertices_moved"] = len(em.select_vertices(sphere=(center, radius)))
    out["api_dab_100k_ms"] = ms(median_time(dab, 3))
    out["api_dab_100k_commit_ms"] = ms(median_time(em.to_mesh, 1, warmup=0))

    # Catmull-Clark (procedural modifier) on a sphere of ~25k quads.
    poly = PolyMesh.from_mesh(sphere("10k"))
    out["subdivision_input_faces"] = poly.face_count
    out["subdivision_l1_ms"] = ms(median_time(lambda: catmull_clark(poly), 3))
    level1 = catmull_clark(poly)
    out["subdivision_l2_ms"] = ms(median_time(lambda: catmull_clark(level1), 3))

    # Closest-point queries on a 1M-vertex (2M-triangle) object.
    big = sphere("100k" if quick else "1m")
    scene = Scene()
    scene.add_object("Big", mesh=Mesh(big.positions, big.triangles))
    out["query_triangles"] = big.triangle_count
    t0 = time.perf_counter()
    scene.closest_point((0.0, 0.0, 2.0))
    out["closest_point_first_ms"] = ms(time.perf_counter() - t0)
    rng = np.random.default_rng(1)
    points = rng.normal(size=(20, 3)) * 1.5

    def queries():
        for p in points:
            scene.closest_point(p)

    out["closest_point_warm_ms"] = ms(median_time(queries, 3) / len(points))
    return out


GROUPS = {"baseline": group_baseline}


# ---------------------------------------------------------------------------------------
# Driver


def run_group(name: str, quick: bool) -> dict[str, Any]:
    result = GROUPS[name](quick)
    result["peak_rss_mib"] = peak_rss_mib()
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join(GROUPS))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print("BENCH-RESULT " + json.dumps(run_group(args.worker, args.quick)))
        return 0
    groups = [g for g in args.only.split(",") if g]
    unknown = sorted(set(groups) - set(GROUPS))
    if unknown:
        parser.error(f"unknown groups: {', '.join(unknown)}")
    data: dict[str, Any] = {"label": args.label, "machine": machine(), "groups": {}}
    for group in groups:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", group]
        if args.quick:
            command.append("--quick")
        t0 = time.perf_counter()
        proc = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BENCH-RESULT ")]
        if proc.returncode != 0 or not lines:
            data["groups"][group] = {"error": (proc.stderr or proc.stdout)[-2000:]}
            print(f"{group}: FAILED\n{proc.stderr[-2000:]}", file=sys.stderr)
            continue
        data["groups"][group] = json.loads(lines[-1][len("BENCH-RESULT ") :])
        print(f"{group} ({time.perf_counter() - t0:.0f} s): {json.dumps(data['groups'][group])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    if args.compare:
        compare(data, json.loads(args.compare.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
