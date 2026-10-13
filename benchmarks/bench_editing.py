"""Benchmark representative editing operations on small, medium and large meshes.

Run from the repository root:

    python benchmarks/bench_editing.py [--json results.json] [--quick]

Each timing is the best of several runs (``--quick``: one run, smaller "large"
mesh). Operations run on a fresh copy of the mesh so they do not influence
each other.
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np

import aether3d
from aether3d import Mesh, create_cube, create_sphere
from aether3d.editing import EditableMesh


def grid_mesh(n: int) -> Mesh:
    em = EditableMesh()
    em.add_grid(n, n, n)
    em.extrude_faces(em.select_all(), 1.0)  # a closed-bottom "slab" with walls
    return em.to_mesh()


def best(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def bench_case(mesh: Mesh, repeat: int) -> dict:
    """Time every benchmarked operation on fresh copies of one mesh."""
    base = EditableMesh.from_mesh(mesh)
    faces = base.face_ids()
    rng = np.random.default_rng(0)
    some = [int(f) for f in rng.choice(faces, size=min(1000, len(faces)), replace=False)]
    quads = [f for f in some if len(base.face_vertices(f)) == 4]
    edges = base.edge_ids()
    top = base.select_faces(normal=((0, 0, 1), 30))  # top cap (sphere: polar region)
    ring_edge = edges[len(edges) // 2]
    # Inset thickness relative to the smallest edge, so it is valid on every face.
    thickness = 0.05 * min(base.edge_length(e) for e in edges)
    row = {
        "triangles": mesh.triangle_count,
        "faces": base.face_count,
        "vertices": base.vertex_count,
    }

    def timed(name, fn, setup=base.copy):
        def go():
            m = setup()
            t0 = time.perf_counter()
            fn(m)
            return time.perf_counter() - t0

        row[name] = min(go() for _ in range(repeat))

    def after_extrude():
        m = base.copy()
        m.extrude_faces(top, 0.5)
        return m

    row["from_mesh"] = best(lambda: EditableMesh.from_mesh(mesh), repeat)
    row["from_mesh (no quads)"] = best(lambda: EditableMesh.from_mesh(mesh, quads=False), repeat)
    timed("select_faces(normal)", lambda m: m.select_faces(normal=((0, 0, 1), 1)))
    timed("translate all", lambda m: m.translate(m.select_all(), (0, 0, 1)))
    timed("extrude region (top)", lambda m: m.extrude_faces(top, 0.5))
    timed("inset 1000 faces", lambda m: m.inset_faces(some, thickness))
    timed("subdivide 1000 quads", lambda m: m.subdivide_faces(quads or some[:1], 1))
    timed("loop_cut", lambda m: m.loop_cut(ring_edge))
    timed("triangulate all", lambda m: m.triangulate_faces())
    timed("validate", lambda m: m.validate())
    timed("to_mesh", lambda m: m.to_mesh())
    timed("undo (extrude)", lambda m: m.undo(), setup=after_extrude)
    return row


def run(quick: bool = False) -> dict:
    repeat = 1 if quick else 3
    cases = {
        "small (box)": create_cube(2),
        "medium (sphere 64x32)": create_sphere(1, 64, 32),
        "medium (grid 100x100)": grid_mesh(100),
        "large (grid 200x200)" if quick else "large (grid 316x316)": grid_mesh(
            200 if quick else 316
        ),
    }
    results = {label: bench_case(mesh, repeat) for label, mesh in cases.items()}
    return {
        "aether3d": aether3d.__version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "results": results,
    }


def print_table(data: dict) -> None:
    rows = data["results"]
    labels = list(rows)
    keys = [k for k in next(iter(rows.values())) if k not in ("triangles", "faces", "vertices")]
    print(f"Aether3D {data['aether3d']}, Python {data['python']}, NumPy {data['numpy']}")
    print(f"{'':28}" + "".join(f"{lab:>24}" for lab in labels))
    for k in ("triangles", "faces", "vertices"):
        print(f"{k:28}" + "".join(f"{rows[lab][k]:>24}" for lab in labels))
    for k in keys:
        print(f"{k:28}" + "".join(f"{rows[lab][k] * 1000:>21.1f} ms" for lab in labels))


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    data = run(quick=quick)
    print_table(data)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
