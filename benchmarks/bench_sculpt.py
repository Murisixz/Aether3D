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

Sized groups (each size in its own process, so ``peak_rss_mib`` is per size), on spheres
of about 10k, 100k and 1M vertices (``--quick``: 10k and 100k):

* ``convert`` -- ``SculptMesh.from_mesh``; committing an unchanged mesh, after one dab
  (normals of the touched groups recomputed) and after a dynamic-topology dab (compaction).
* ``dab`` -- one ``draw`` dab of radius 0.03, 0.1 and 0.3 (the sphere's radius is 1).
* ``stroke`` -- 100-dab strokes of ``draw``, ``clay``, ``smooth`` and ``grab`` (radius
  0.1, dabs 10 % of the radius apart): per dab, per stage, undo, commit, journal size.
* ``symmetry`` -- the same 50-dab stroke without symmetry, mirrored in x, in x and y.
* ``dyntopo`` -- a 30-dab ``clay`` stroke refining to half the edge length: per dab,
  topology changes, commit, undo.
* ``remesh`` -- isotropic remeshing at the current mean edge length (3 iterations, 1 at
  1M).
* ``voxel`` -- voxel remeshing of the 100k sphere at voxel sizes giving about 10k, 100k
  and 1M triangles.
* ``validity`` -- ``check_mesh`` without and with the exact self-intersection test.
* ``modeling`` -- Stage 13 editing operators (3-segment bevel of a latitude ring, bisect,
  bisect with fill, knife, cleanup, recalculate normals) at 10k and 100k (the editing
  engine's own conversion takes seconds at 100k; see ``baseline``).

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


def _sculpt(size: str):
    from aether3d.sculpt import SculptMesh

    return SculptMesh.from_mesh(sphere(size))


def _stroke(kind: str, radius: float, dabs: int, **brush) -> dict:
    """A stroke of ``dabs`` dabs (spacing 10% of the radius) along an arc on the sphere's
    upper half: every dab hits the surface."""
    import numpy as np

    theta = 0.6  # a circle of latitude, its dabs 10% of the radius apart
    angle = np.arange(dabs) * 0.1 * radius / np.sin(theta)
    path = np.stack(
        [
            np.sin(theta) * np.cos(angle),
            np.sin(theta) * np.sin(angle),
            np.full(dabs, np.cos(theta)),
        ],
        axis=1,
    )
    return {
        "brush": {"kind": kind, "radius": radius, "strength": 0.1, **brush},
        "path": path.tolist(),
        "resample": False,
    }


def group_convert(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.sculpt import SculptMesh

    mesh = sphere(size)
    repeat = 3 if size != "1m" else 1
    out: dict[str, Any] = {"vertices": mesh.vertex_count, "triangles": mesh.triangle_count}
    out["from_mesh_s"] = round(median_time(lambda: SculptMesh.from_mesh(mesh), repeat, 0), 4)
    sm = SculptMesh.from_mesh(mesh)
    out["to_mesh_unchanged_ms"] = ms(median_time(sm.to_mesh, repeat))
    sm.stroke(_stroke("draw", 0.1, 1))
    out["to_mesh_after_dab_ms"] = ms(median_time(sm.to_mesh, repeat, 0))
    from aether3d.sculpt.remesh import edge_statistics

    detail = edge_statistics(sm)["mean"] / 2
    sm.stroke(_stroke("clay", 0.1, 1, dyntopo={"detail": detail, "mode": "constant"}))
    out["to_mesh_after_topology_ms"] = ms(median_time(sm.to_mesh, repeat, 0))
    return out


def group_dab(size: str, quick: bool) -> dict[str, Any]:
    sm = _sculpt(size)
    out: dict[str, Any] = {"vertices": sm.vertex_count}
    for radius in (0.03, 0.1, 0.3):
        stroke = _stroke("draw", radius, 1)
        r = sm.stroke(stroke)
        out[f"r{radius}_moved"] = r.data["moved_vertices"]
        out[f"r{radius}_ms"] = ms(median_time(lambda stroke=stroke: sm.stroke(stroke), 5))
    # The same dab with the editing API (the baseline), for the same vertices.
    return out


def group_stroke(size: str, quick: bool) -> dict[str, Any]:
    sm = _sculpt(size)
    out: dict[str, Any] = {"vertices": sm.vertex_count}
    for kind in ("draw", "clay", "smooth", "grab"):
        stroke = _stroke(kind, 0.1, 100)
        t0 = time.perf_counter()
        r = sm.stroke(stroke)
        seconds = time.perf_counter() - t0
        out[f"{kind}_100_dabs_ms"] = ms(seconds)
        out[f"{kind}_ms_per_dab"] = round(1000 * seconds / 100, 3)
        out[f"{kind}_stages_ms"] = {k: ms(v) for k, v in r.timings.items()}
        t0 = time.perf_counter()
        sm.undo()
        out[f"{kind}_undo_ms"] = ms(time.perf_counter() - t0)
    sm.stroke(_stroke("draw", 0.1, 100))
    out["commit_after_stroke_ms"] = ms(median_time(sm.to_mesh, 1, warmup=0))
    out["journal_bytes"] = sm.history_nbytes
    return out


def group_symmetry(size: str, quick: bool) -> dict[str, Any]:
    sm = _sculpt(size)
    out: dict[str, Any] = {"vertices": sm.vertex_count}
    for sym in ("", "x", "xy"):
        stroke = _stroke("draw", 0.1, 50, symmetry=sym)
        t0 = time.perf_counter()
        sm.stroke(stroke)
        out[f"sym_{sym or 'none'}_ms_per_dab"] = round(1000 * (time.perf_counter() - t0) / 50, 3)
    return out


def group_dyntopo(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.sculpt.remesh import edge_statistics

    sm = _sculpt(size)
    detail = edge_statistics(sm)["mean"] / 2  # refine to half the current edge length
    out: dict[str, Any] = {"triangles_before": sm.triangle_count, "detail": detail}
    stroke = _stroke("clay", 0.15, 30, dyntopo={"detail": detail, "mode": "constant"})
    t0 = time.perf_counter()
    r = sm.stroke(stroke)
    seconds = time.perf_counter() - t0
    out["ms_per_dab"] = round(1000 * seconds / 30, 2)
    out["topology"] = r.data["topology"]
    out["triangles_after"] = r.data["triangles"]
    out["stages_ms"] = {k: ms(v) for k, v in r.timings.items()}
    t0 = time.perf_counter()
    sm.to_mesh()
    out["commit_ms"] = ms(time.perf_counter() - t0)
    t0 = time.perf_counter()
    sm.undo()
    out["undo_ms"] = ms(time.perf_counter() - t0)
    return out


def group_remesh(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.sculpt.remesh import edge_statistics

    sm = _sculpt(size)
    mean = edge_statistics(sm)["mean"]
    out: dict[str, Any] = {"triangles_before": sm.triangle_count, "edge_mean_before": mean}
    iterations = 1 if size == "1m" else 3
    t0 = time.perf_counter()
    r = sm.remesh(mean, iterations=iterations)
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["iterations"] = iterations
    out["triangles_after"] = r.data["triangles"]
    out["edge_cv_after"] = round(r.data["edge_length"]["std"] / r.data["edge_length"]["mean"], 4)
    out["operations"] = {k: r.data[k] for k in ("splits", "collapses", "flips", "relaxed")}
    return out


VOXEL_SIZES = {"10k": 0.06, "100k": 0.019, "1m": 0.006}
"""Voxel sizes giving outputs of about 10k, 100k and 1M triangles from a unit sphere."""


def group_voxel(size: str, quick: bool) -> dict[str, Any]:
    sm = _sculpt("100k")
    out: dict[str, Any] = {"input_triangles": sm.triangle_count, "voxel_size": VOXEL_SIZES[size]}
    t0 = time.perf_counter()
    r = sm.voxel_remesh(VOXEL_SIZES[size])
    out["seconds"] = round(time.perf_counter() - t0, 3)
    out["output_triangles"] = r.data["triangles"]
    out["band_points"] = r.data["band_points"]
    out["tier"] = r.data["output"]["tier"]
    out["closed"] = r.data["output"]["closed"]
    return out


def group_validity(size: str, quick: bool) -> dict[str, Any]:
    from aether3d.sculpt import check_mesh

    mesh = sphere(size)
    out: dict[str, Any] = {"triangles": mesh.triangle_count}
    out["check_s"] = round(median_time(lambda: check_mesh(mesh.positions, mesh.triangles), 1, 0), 3)
    t0 = time.perf_counter()
    report = check_mesh(mesh.positions, mesh.triangles, self_intersections=True)
    out["check_with_self_intersections_s"] = round(time.perf_counter() - t0, 3)
    out["tier"] = report.summary["tier"]
    return out


def group_modeling(size: str, quick: bool) -> dict[str, Any]:
    """Stage 13 modeling operators on an editable sphere of the given size (the editing
    engine's own conversions are measured by ``baseline``)."""
    from aether3d.editing import EditableMesh

    em = EditableMesh.from_mesh(sphere(size))
    out: dict[str, Any] = {"faces": em.summary()["face_count"]}

    def timed(name, fn):
        t0 = time.perf_counter()
        result = fn()
        out[f"{name}_ms"] = ms(time.perf_counter() - t0)
        return result

    import math

    rings = SIZES[size][0]
    z = math.cos(round(math.acos(0.5) / (math.pi / rings)) * math.pi / rings)
    ring = em.select("edges", predicate=lambda _i, p: abs(p[2] - z) < 1e-6)  # one latitude
    out["bevel_edges"] = len(ring.edges)
    timed("bevel_3_segments", lambda: em.bevel_edges(ring, 0.002, segments=3))
    em.undo()
    timed("bisect", lambda: em.bisect((0, 0, 0.1234), (0.3, 0.2, 1)))
    em.undo()
    timed("bisect_fill", lambda: em.bisect((0, 0, 0.1234), (0.3, 0.2, 1), clear="front", fill=True))
    em.undo()
    path = [(-1.5, -1.5, 0.05), (1.5, 1.5, 0.05)]
    timed("knife_project", lambda: em.knife_project(path, (0, 0, -1), through=False))
    em.undo()
    timed("cleanup", lambda: em.cleanup())
    timed("recalculate_normals", lambda: em.recalculate_normals())
    return out


SIZED = {
    "convert": group_convert,
    "dab": group_dab,
    "stroke": group_stroke,
    "symmetry": group_symmetry,
    "dyntopo": group_dyntopo,
    "remesh": group_remesh,
    "voxel": group_voxel,
    "validity": group_validity,
    "modeling": group_modeling,
}


GROUPS = {"baseline": group_baseline}
ALL_SIZES = ("10k", "100k", "1m")
MODELING_SIZES = ("10k", "100k")
"""The editing engine (a Python object graph per element) is measured up to 100k faces:
its own conversion already takes seconds there (``baseline``)."""


# ---------------------------------------------------------------------------------------
# Driver


def run_group(name: str, quick: bool) -> dict[str, Any]:
    group, _, size = name.partition(":")
    result = SIZED[group](size, quick) if size else GROUPS[group](quick)
    result["peak_rss_mib"] = peak_rss_mib()
    return result


def jobs(only: list[str], quick: bool) -> list[str]:
    """Worker names: ``baseline`` and ``group:size`` for every size of a sized group."""
    out = []
    for name in only:
        group, _, size = name.partition(":")
        if group in GROUPS and not size:
            out.append(group)
        elif group in SIZED:
            sizes = (size,) if size else (MODELING_SIZES if group == "modeling" else ALL_SIZES)
            out += [f"{group}:{s}" for s in sizes if not (quick and s == "1m")]
        else:
            raise SystemExit(f"unknown group {name!r}")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", default=",".join([*GROUPS, *SIZED]))
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        print("BENCH-RESULT " + json.dumps(run_group(args.worker, args.quick)))
        return 0
    groups = jobs([g for g in args.only.split(",") if g], args.quick)
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
