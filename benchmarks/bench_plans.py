"""Benchmark building scenes of many objects: one scene plan versus repeated operations.

Run from the repository root:

    python benchmarks/bench_plans.py [--quick] [--sizes 100,1000,10000] [--repeat R]
        [--no-cli] [--no-cli-per-object] [--no-mcp] [--no-gpu] [--json results.json]

**Workload.** N blocks -- 0.9 m cubes on a grid 100 wide, each a mesh object with its own
name and position -- for N = 100, 1,000 and 10,000. Every method makes the same N objects:

| method | how | undo steps |
| --- | --- | --- |
| ``add_cube`` | ``scene.add_cube(...)`` per block: a new cube mesh each time | N |
| ``add_object`` | ``scene.add_object(mesh=shared_cube, ...)`` per block | N |
| ``transaction`` | the same inside ``with scene.transaction(...)`` | 1 |
| ``script`` | one procedural script: one box node, N ``create_object`` | 1 (2 at 10,000) |
| ``plan`` | ``scene.execute_plan(plan)``: one grid pattern | 1 |
| ``cli_per_object`` | ``aether3d procedural run`` once per block | N (in the file) |
| ``cli_script``, ``cli_plan`` | one ``aether3d procedural run`` / ``execute-plan`` | 1 |
| ``mcp_per_object`` | one ``execute_scene_plan`` tool call per block | N |
| ``mcp_plan`` | one ``execute_scene_plan`` tool call | 1 |

(Procedural scripts hold at most 10,000 operations, so ``script`` uses two at 10,000.)

**Measurements.** Each method runs in a fresh Python process: one warm-up build, then R
measured builds, each into a new scene. Reported: the median, minimum and maximum wall
time, the throughput (blocks per second), the peak RSS of the process (for MCP and the
CLI: of the server or CLI process), and the undo steps and bytes the build leaves.
``cli_per_object`` runs fully at the smallest size; at larger sizes it times 10 calls on a
project that already holds N blocks and projects N calls from that (labelled).

Then, on a scene of N blocks built by a plan (in its own process):

* the plan's stages (``PlanResult.timings``): parse and check, bind (patterns, names,
  references), resources, objects (creation), apply, total;
* geometry generation alone: N ``create_cube`` calls (what ``add_cube`` pays per block);
* undo and redo of the whole build;
* spatial queries: the first ``overlaps`` query (it builds the index), then repeated
  ``overlaps``, ``raycast`` and ``nearest`` queries;
* rendering: the GPU rasterizer's first render of the scene (pipelines, uploads; the
  device is created before) and the median of 5 warm renders at 1280x720, with its draw
  calls, and one CPU rasterizer render at 640x360. Path tracing is a separate stage that
  does not depend on how the scene was built: see ``bench_pathtrace.py`` and
  ``bench_gpu_pathtrace.py``.

And the example plans (castle, village): build time, stages, undo bytes, GPU draw calls.

MCP numbers include the JSON-RPC round trip over stdio; the tool-call overhead is the
median round trip of a trivial call, and a plan call's wall time minus the server's own
time for it.

Run it against an older Aether3D (``PYTHONPATH=old/src``) to measure a baseline: the
methods that need scene plans or the MCP server are skipped when they do not exist.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 100  # blocks per row
SIZE = 0.9


def position(i: int) -> list[float]:
    return [float(i % WIDTH), float(i // WIDTH), SIZE / 2]


# --------------------------------------------------------------------------- #
# Measurements of one process
# --------------------------------------------------------------------------- #


def rss_now_mib() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def peak_rss_mib() -> float | None:
    """This process's peak RSS (``None`` on Windows). On Linux, ``VmHWM``: ``ru_maxrss``
    would include the memory of the process that started this one (Linux carries it over
    an exec)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 2**20 if sys.platform == "darwin" else peak / 1024, 1)  # bytes on macOS


# Runs ``python -m aether3d ...`` and appends its peak RSS (MiB) to $AETHER3D_BENCH_RSS.
AETHER3D_WITH_RSS = (
    "import atexit, os, sys\n"
    "sys.path.insert(0, os.environ['AETHER3D_BENCH_DIR'])\n"
    "from bench_plans import peak_rss_mib\n"
    "def report():\n"
    "    with open(os.environ['AETHER3D_BENCH_RSS'], 'a') as f:\n"
    "        f.write(f'{peak_rss_mib()}\\n')\n"
    "atexit.register(report)\n"
    "import runpy\n"
    "sys.argv = ['aether3d', *sys.argv[1:]]\n"
    "runpy.run_module('aether3d', run_name='__main__', alter_sys=True)\n"
)


def child_env(rss_file: Path) -> dict:
    here = str(Path(__file__).parent)
    return dict(os.environ, AETHER3D_BENCH_RSS=str(rss_file), AETHER3D_BENCH_DIR=here)


def child_peak(rss_file: Path) -> float | None:
    """The largest peak RSS the wrapped processes reported."""
    try:
        return max(float(v) for v in rss_file.read_text().split())
    except (OSError, ValueError):
        return None


def summary(times: list[float], n: int) -> dict:
    median = statistics.median(times)
    return {
        "runs": len(times),
        "median_s": round(median, 6),
        "min_s": round(min(times), 6),
        "max_s": round(max(times), 6),
        "spread": round((max(times) - min(times)) / median, 3) if median else 0.0,
        "blocks_per_s": round(n / median) if median else None,
    }


def plan_document(n: int) -> dict:
    """N blocks in one plan: a grid pattern (the same positions and names as the other
    methods), or per-instance expressions when N is not a whole number of rows."""
    blocks: dict = {"mesh": "@block", "name": "block {i}"}
    if n <= WIDTH or n % WIDTH == 0:
        blocks["grid"] = {"count": [min(n, WIDTH), max(1, n // WIDTH)]}
        blocks["position"] = [0, 0, SIZE / 2]
    else:
        blocks["repeat"] = n
        blocks["position"] = [
            {"expr": f"i % {WIDTH}"},
            {"expr": f"floor(i / {WIDTH})"},
            SIZE / 2,
        ]
    return {
        "format": "aether3d.scene-plan",
        "version": 1,
        "label": f"{n} blocks",
        "meshes": {"block": {"primitive": "cube", "size": SIZE}},
        "objects": {"blocks": blocks},
    }


def script_documents(n: int, first: bool = True) -> list[dict]:
    """Procedural scripts that make blocks 0..n-1 (several: at most 10,000 operations);
    the first also makes the node, if ``first``."""
    from aether3d.procedural.script import MAX_OPERATIONS

    node = {"op": "add_geometry", "name": "Block", "primitive": "box"}
    node["params"] = {"width": SIZE, "depth": SIZE, "height": SIZE}
    creates = [
        {"op": "create_object", "name": f"block {i}", "geometry": "Block", "position": position(i)}
        for i in range(n)
    ]
    docs, room = [], MAX_OPERATIONS - 1
    for k in range(0, max(n, 1), room):
        ops = ([node] if first and k == 0 else []) + creates[k : k + room]
        docs.append({"format": "aether3d.procedural-script", "version": 1, "operations": ops})
    return docs


# --------------------------------------------------------------------------- #
# Builders (in-process methods)
# --------------------------------------------------------------------------- #


def build(method: str, n: int):
    from aether3d import Scene, create_cube

    scene = Scene("Blocks")
    if method == "add_cube":
        for i in range(n):
            scene.add_cube(f"block {i}", size=SIZE, position=position(i))
    elif method == "add_object":
        mesh = create_cube(SIZE)
        for i in range(n):
            scene.add_object(f"block {i}", mesh=mesh, position=position(i))
    elif method == "transaction":
        mesh = create_cube(SIZE)
        with scene.transaction(f"{n} blocks"):
            for i in range(n):
                scene.add_object(f"block {i}", mesh=mesh, position=position(i))
    elif method == "script":
        from aether3d import run_procedural_script

        for doc in script_documents(n):
            run_procedural_script(scene, doc)
    elif method == "plan":
        scene.execute_plan(plan_document(n))
    else:
        raise ValueError(method)
    assert len(scene) == n, (method, len(scene))
    return scene


def check_blocks(scene, n: int) -> None:
    """Every method must make the same objects: names and positions, in order."""
    made = [(o.name, tuple(o.position)) for o in scene]
    expected = [(f"block {i}", tuple(position(i))) for i in range(n)]
    if made != expected:
        raise AssertionError("the method did not make the benchmark's blocks")


def worker_build(method: str, n: int, repeat: int) -> dict:
    import gc

    import aether3d  # noqa: F401 - the starting RSS includes the engine and NumPy

    rss_start = rss_now_mib()
    check_blocks(build(method, n), n)  # warm-up, and the same objects as every method
    gc.collect()
    times, scene = [], None
    for _ in range(repeat):
        scene = None
        gc.collect()
        start = time.perf_counter()
        scene = build(method, n)
        times.append(time.perf_counter() - start)
    history = scene.history
    return {
        **summary(times, n),
        "undo_steps": len(history),
        "steps_dropped": history.dropped,
        "history_bytes": history.nbytes,
        "rss_start_mib": None if rss_start is None else round(rss_start, 1),
        "peak_rss_mib": peak_rss_mib(),
    }


# --------------------------------------------------------------------------- #
# The plan-built scene: stages, history, queries, rendering
# --------------------------------------------------------------------------- #


def median_time(fn, repeat: int) -> tuple[float, object]:
    times, value = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times), value


def worker_scene(n: int, repeat: int, gpu: bool) -> dict:
    from aether3d import Scene, create_cube, validate_plan
    from aether3d.spatial.request import run_query

    out: dict = {}
    doc = plan_document(n)
    seconds, _ = median_time(lambda: validate_plan(doc), repeat)
    out["static_check_ms"] = round(seconds * 1e3, 3)
    stages = []
    for _ in range(repeat + 1):
        scene = Scene("Blocks")
        result = scene.execute_plan(doc)
        stages.append(result.timings)
    stages = stages[1:]  # the first is the warm-up
    out["plan_stages_ms"] = {
        k: round(statistics.median(s[k] for s in stages) * 1e3, 3) for k in stages[0]
    }
    seconds, _ = median_time(lambda: [create_cube(SIZE) for _ in range(n)], repeat)
    out["geometry_generation_ms"] = round(seconds * 1e3, 3)
    out["history_bytes"] = scene.history.nbytes
    undo, redo = [], []
    for _ in range(repeat):
        start = time.perf_counter()
        scene.history.undo()
        undo.append(time.perf_counter() - start)
        start = time.perf_counter()
        scene.history.redo()
        redo.append(time.perf_counter() - start)
    out["undo_ms"] = round(statistics.median(undo) * 1e3, 3)
    out["redo_ms"] = round(statistics.median(redo) * 1e3, 3)

    # spatial queries (the first one builds the index)
    overlaps = {"query": "overlaps"}
    start = time.perf_counter()
    found = run_query(scene, overlaps)["result"]
    out["first_overlaps_ms"] = round((time.perf_counter() - start) * 1e3, 3)
    out["overlapping_pairs"] = len(found)
    queries = {
        "overlaps": overlaps,
        "raycast": {"query": "raycast", "origin": [50.3, -10, 0.4], "direction": [0, 1, 0]},
        "nearest": {"query": "nearest", "point": [50, 50, 3], "count": 5},
    }
    out["queries_ms"] = {
        name: round(median_time(lambda q=q: run_query(scene, q), repeat)[0] * 1e3, 3)
        for name, q in queries.items()
    }

    # rendering
    scene.add_directional_light("Sun", direction=(-0.4, 0.6, -1))
    camera = scene.add_camera("Camera", fov=50)
    camera.frame(direction=(-0.5, 0.8, -0.6), aspect=16 / 9)
    start = time.perf_counter()
    scene.render(width=640, height=360, backend="cpu")
    out["cpu_raster_640x360_s"] = round(time.perf_counter() - start, 3)
    if gpu:
        from aether3d.rendering.gpu import gpu_status

        if gpu_status()["available"]:
            warm = Scene("Warm-up")
            warm.add_cube("Cube")
            warm.add_camera("Camera").frame()
            warm.render(width=64, height=64, backend="gpu")  # creates the device
            start = time.perf_counter()
            result = scene.render(width=1280, height=720, backend="gpu")
            out["gpu_first_render_s"] = round(time.perf_counter() - start, 4)
            seconds, result = median_time(
                lambda: scene.render(width=1280, height=720, backend="gpu"), 5
            )
            out["gpu_render_1280x720_ms"] = round(seconds * 1e3, 2)
            out["gpu_draw_calls"] = result.stats.get("draw_calls")
            out["gpu_adapter"] = result.backend["adapter"]["name"]
    out["peak_rss_mib"] = peak_rss_mib()
    return out


def worker_examples(repeat: int, gpu: bool) -> dict:
    from importlib import resources

    from aether3d import Scene

    out = {}
    for name in ("castle", "village"):
        text = (resources.files("aether3d.plans") / "examples" / f"{name}.json").read_text()
        doc = json.loads(text)
        times, result, scene = [], None, None
        for k in range(repeat + 1):
            scene = Scene(name)
            start = time.perf_counter()
            result = scene.execute_plan(doc)
            if k:
                times.append(time.perf_counter() - start)
        row = {
            "objects": result.counts["objects"],
            "meshes": result.counts["meshes"],
            **summary(times, result.counts["objects"]),
            "stages_ms": {k: round(v * 1e3, 2) for k, v in result.timings.items()},
            "history_bytes": scene.history.nbytes,
        }
        if gpu:
            from aether3d.rendering.gpu import gpu_status

            if gpu_status()["available"]:
                scene.render(width=1280, height=720, backend="gpu")
                seconds, render = median_time(
                    lambda s=scene: s.render(width=1280, height=720, backend="gpu"), 5
                )
                row["gpu_render_1280x720_ms"] = round(seconds * 1e3, 2)
                row["gpu_draw_calls"] = render.stats.get("draw_calls")
        out[name] = row
    out["peak_rss_mib"] = peak_rss_mib()
    return out


# --------------------------------------------------------------------------- #
# The CLI
# --------------------------------------------------------------------------- #


def worker_cli(method: str, n: int, repeat: int, full: bool) -> dict:
    work = Path(tempfile.mkdtemp(prefix="aether3d-bench-"))
    rss = work / "rss.txt"

    def cli(*args: str) -> None:
        command = [sys.executable, "-c", AETHER3D_WITH_RSS, *args]
        env = child_env(rss)
        done = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(args)}: {done.stdout}{done.stderr}")

    try:
        empty = work / "empty.aether"
        cli("new", str(empty))
        if method in ("cli_plan", "cli_script"):
            files = []
            if method == "cli_plan":
                files.append(work / "plan.json")
                files[0].write_text(json.dumps(plan_document(n)))
            else:
                for k, doc in enumerate(script_documents(n)):
                    files.append(work / f"script{k}.json")
                    files[-1].write_text(json.dumps(doc))
            command = "execute-plan" if method == "cli_plan" else "procedural"
            times = []
            for k in range(repeat + 1):
                project = work / "blocks.aether"
                shutil.copyfile(empty, project)
                start = time.perf_counter()
                for f in files:
                    if command == "execute-plan":
                        cli("execute-plan", str(project), str(f))
                    else:
                        cli("procedural", "run", str(project), str(f))
                if k:
                    times.append(time.perf_counter() - start)
            return {
                **summary(times, n),
                "peak_rss_mib": child_peak(rss),
            }
        # cli_per_object: one call per block on a project that already has the node
        project = work / "blocks.aether"
        shutil.copyfile(empty, project)
        setup = script_documents(0)[0]
        if not full:  # start from N - 10 blocks and time the last 10 calls
            setup = script_documents(n - 10)[0]
        (work / "setup.json").write_text(json.dumps(setup))
        cli("procedural", "run", str(project), str(work / "setup.json"))
        first = 0 if full else n - 10
        times = []
        for i in range(first, n):
            one = {"format": "aether3d.procedural-script", "version": 1}
            one["operations"] = [
                {
                    "op": "create_object",
                    "name": f"block {i}",
                    "geometry": "Block",
                    "position": position(i),
                }
            ]
            (work / "one.json").write_text(json.dumps(one))
            start = time.perf_counter()
            cli("procedural", "run", str(project), str(work / "one.json"))
            times.append(time.perf_counter() - start)
        per_call = statistics.median(times)
        row = {
            "calls_timed": len(times),
            "per_call_median_s": round(per_call, 4),
            "per_call_max_s": round(max(times), 4),
            "peak_rss_mib": child_peak(rss),
        }
        if full:
            row.update(summary([sum(times)], n))
            row["runs"] = 1
        else:
            row["projected_total_s"] = round(per_call * n, 1)  # a lower bound: calls grow
        return row
    finally:
        shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------- #
# MCP
# --------------------------------------------------------------------------- #


def worker_mcp(method: str, n: int, repeat: int) -> dict:
    import anyio
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters

    work = Path(tempfile.mkdtemp(prefix="aether3d-bench-mcp-"))
    rss = Path(tempfile.mkdtemp(prefix="aether3d-bench-rss-")) / "rss.txt"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-c", AETHER3D_WITH_RSS, "mcp", "serve", "--workspace", str(work)],
        env=child_env(rss),
    )

    async def call(client, tool, arguments):
        result = await client.call_tool(tool, arguments)
        if result.is_error:
            raise RuntimeError(f"{tool}: {result.structured_content}")
        return result

    async def main() -> dict:
        async with Client(server) as client:
            row: dict = {}
            trivial = []
            for _ in range(60):
                start = time.perf_counter()
                await call(client, "capabilities", {"topic": "limits"})
                trivial.append(time.perf_counter() - start)
            row["trivial_call_ms"] = round(statistics.median(trivial[10:]) * 1e3, 3)
            if method == "mcp_plan":
                doc = plan_document(n)
                times, server_times, sizes = [], [], []
                for k in range(repeat + 1):
                    await call(client, "scene_new", {"discard_unsaved": True})
                    start = time.perf_counter()
                    result = await call(client, "execute_scene_plan", {"plan": doc})
                    seconds = time.perf_counter() - start
                    if k:
                        times.append(seconds)
                        server_times.append(result.structured_content["timings"]["total"])
                        sizes.append(len(result.content[-1].text))
                row.update(summary(times, n))
                row["server_median_s"] = round(statistics.median(server_times), 6)
                row["overhead_ms"] = round(
                    (statistics.median(times) - statistics.median(server_times)) * 1e3, 2
                )
                row["reply_bytes"] = sizes[-1]
            else:
                await call(client, "scene_new", {"discard_unsaved": True})
                times = []
                for i in range(n):
                    one = {
                        "format": "aether3d.scene-plan",
                        "version": 1,
                        "objects": {
                            "block": {
                                "mesh": {"primitive": "cube", "size": SIZE},
                                "name": f"block {i}",
                                "position": position(i),
                            }
                        },
                    }
                    start = time.perf_counter()
                    await call(client, "execute_scene_plan", {"plan": one})
                    times.append(time.perf_counter() - start)
                row.update(summary([sum(times)], n))
                row["per_call_median_ms"] = round(statistics.median(times) * 1e3, 3)
                row["per_call_last_100_ms"] = round(statistics.median(times[-100:]) * 1e3, 3)
            return row

    try:
        row = anyio.run(main)
        time.sleep(0.2)  # the server writes its peak RSS as it exits
        row["server_peak_rss_mib"] = child_peak(rss)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(rss.parent, ignore_errors=True)
    return row


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #


def hardware() -> dict:
    import numpy as np

    import aether3d

    cpu = platform.processor() or platform.machine()
    memory = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory = round(int(line.split()[1]) / 2**20, 1)
    except OSError:
        pass
    info = {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_gib": memory,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "aether3d": aether3d.__version__,
    }
    try:
        from aether3d.rendering.gpu import gpu_status

        status = gpu_status()
        if status["available"]:
            info["gpu"] = status["selected"]["adapter"]["name"]
    except Exception:  # noqa: BLE001 - wgpu missing: no GPU column
        pass
    return info


def run_worker(*args: str) -> dict:
    command = [sys.executable, __file__, "--worker", *args]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return {"error": (done.stderr or done.stdout).strip().splitlines()[-1]}
    return json.loads(done.stdout.strip().splitlines()[-1])


def available(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="100 and 1,000 blocks, 2 runs")
    parser.add_argument("--sizes", help="comma-separated block counts")
    parser.add_argument("--repeat", type=int, help="measured runs per method")
    parser.add_argument("--no-cli", action="store_true")
    parser.add_argument(
        "--no-cli-per-object", action="store_true", help="skip the slowest method (CLI per block)"
    )
    parser.add_argument("--no-mcp", action="store_true")
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--json", help="write the results here")
    parser.add_argument("--worker", nargs="+", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:  # one measurement, in its own process
        kind, *rest = args.worker
        if kind == "build":
            data = worker_build(rest[0], int(rest[1]), int(rest[2]))
        elif kind == "scene":
            data = worker_scene(int(rest[0]), int(rest[1]), rest[2] == "gpu")
        elif kind == "examples":
            data = worker_examples(int(rest[0]), rest[1] == "gpu")
        elif kind == "cli":
            data = worker_cli(rest[0], int(rest[1]), int(rest[2]), rest[3] == "full")
        else:
            data = worker_mcp(rest[0], int(rest[1]), int(rest[2]))
        print(json.dumps(data))
        return 0

    sizes = [
        int(s) for s in (args.sizes or ("100,1000" if args.quick else "100,1000,10000")).split(",")
    ]
    repeat = args.repeat or (2 if args.quick else 5)
    plans = available("aether3d.plans")
    mcp = plans and available("mcp") and not args.no_mcp
    gpu = "nogpu" if args.no_gpu or not available("wgpu") else "gpu"
    results: dict = {"hardware": hardware(), "sizes": sizes, "repeat": repeat, "builds": {}}
    print(json.dumps(results["hardware"]))
    methods = ["add_cube", "add_object", "transaction", "script"] + (["plan"] if plans else [])
    for n in sizes:
        rows = results["builds"][str(n)] = {}
        reps = max(2, repeat // 2) if n >= 10000 else repeat
        for method in methods:
            rows[method] = run_worker("build", method, str(n), str(reps))
            print(f"{n:>6} {method:<16} {rows[method]}", flush=True)
        if not args.no_cli:
            for method in ("cli_script", *(["cli_plan"] if plans else [])):
                rows[method] = run_worker("cli", method, str(n), str(max(1, reps // 2)), "full")
                print(f"{n:>6} {method:<16} {rows[method]}", flush=True)
            if not args.no_cli_per_object and (not args.quick or n == sizes[0]):
                mode = "full" if n == sizes[0] else "sample"
                rows["cli_per_object"] = run_worker("cli", "cli_per_object", str(n), "1", mode)
                print(f"{n:>6} {'cli_per_object':<16} {rows['cli_per_object']}", flush=True)
        if mcp:
            rows["mcp_plan"] = run_worker("mcp", "mcp_plan", str(n), str(reps))
            print(f"{n:>6} {'mcp_plan':<16} {rows['mcp_plan']}", flush=True)
            if not args.quick or n == sizes[0]:
                rows["mcp_per_object"] = run_worker("mcp", "mcp_per_object", str(n), "1")
                print(f"{n:>6} {'mcp_per_object':<16} {rows['mcp_per_object']}", flush=True)
        if plans:
            results.setdefault("scene", {})[str(n)] = run_worker("scene", str(n), str(repeat), gpu)
            print(f"{n:>6} {'scene':<16} {results['scene'][str(n)]}", flush=True)
    if plans:
        results["examples"] = run_worker("examples", str(max(2, repeat // 2)), gpu)
        print(f"examples {results['examples']}", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
