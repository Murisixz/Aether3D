"""Every measurement of the v0.19.1 baseline next to the same measurement of 0.20.0.

    .venv/bin/python research/2026-10-stage19/measurements/compare_benchmarks.py \\
        [--json OUT.json] [--markdown OUT.md]

Pairs ``benchmarks/results/stage19-baseline-<name>.json`` with ``stage19-final-<name>.json``
(the same commands, one machine, one at a time: research/2026-10-stage19/DESIGN.md) by
the path of every number in them, so nothing is chosen or left out: times (``_ms``,
``_s``, ``seconds``) and rates (``per_s``, ``mrays``) get a ratio (above 1: 0.20.0 is
faster), everything else (counts, sizes) must be equal or is listed as changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "benchmarks" / "results"
NAMES = ("animation", "engine", "gpu", "gpu-pathtrace", "pathtrace", "raytrace", "render")
SKIP = ("machine", "hardware", "label", "aether3d", "adapter", "platform", "python", "numpy")
LABELS = ("scene", "name", "case", "mode", "label")


def flatten(value, path=(), out=None) -> dict[tuple, object]:
    out = {} if out is None else out
    if isinstance(value, dict):
        for key, v in value.items():
            if key not in SKIP:
                flatten(v, (*path, str(key)), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            tag = str(i)
            if isinstance(v, dict):
                named = [f"{k}={v[k]}" for k in LABELS if k in v]
                size = [f"{v[k]}" for k in ("width", "height") if k in v]
                tag = ",".join(named + (["x".join(size)] if size else []) + [str(i)])
            flatten(v, (*path, tag), out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[path] = float(value)
    return out


def kind(path: tuple) -> str:
    leaf = path[-1]
    if "per_s" in leaf or "mrays" in leaf or leaf.endswith("_s") and "mpix" in leaf:
        return "rate"
    if leaf.endswith("_ms") or leaf.endswith("_s") or "seconds" in leaf or leaf in (
        "total", "trace", "shade", "raster", "geometry", "output", "extract", "bvh", "setup",
        "center", "evaluate", "textures") or path[-2] in ("timings", "stages"):  # fmt: skip
        return "time"
    if "rss" in leaf or "mib" in leaf:
        return "memory"
    return "value"


def compare() -> dict:
    report = {}
    for name in NAMES:
        base = RESULTS / f"stage19-baseline-{name}.json"
        final = RESULTS / f"stage19-final-{name}.json"
        if not (base.exists() and final.exists()):
            report[name] = {"missing": [str(p.name) for p in (base, final) if not p.exists()]}
            continue
        a, b = flatten(json.loads(base.read_text())), flatten(json.loads(final.read_text()))
        rows, changed, only = [], [], sorted(set(a) ^ set(b))
        for path in sorted(set(a) & set(b)):
            k = kind(path)
            x, y = a[path], b[path]
            if k == "time" and x > 0 and y > 0:
                rows.append({"path": "/".join(path), "kind": k, "baseline": x, "final": y,
                             "speedup": x / y})  # fmt: skip
            elif k == "rate" and x > 0 and y > 0:
                rows.append({"path": "/".join(path), "kind": k, "baseline": x, "final": y,
                             "speedup": y / x})  # fmt: skip
            elif k == "memory" and x > 0:
                rows.append({"path": "/".join(path), "kind": k, "baseline": x, "final": y,
                             "ratio": y / x})  # fmt: skip
            elif k == "value" and x != y:
                changed.append({"path": "/".join(path), "baseline": x, "final": y})
        report[name] = {"compared": rows, "changed_values": changed,
                        "only_in_one": ["/".join(p) for p in only]}  # fmt: skip
    return report


def markdown(report: dict) -> str:
    lines = [
        "| Benchmark | Measurement | v0.19.1 | 0.20.0 | Speed-up |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, data in report.items():
        for row in data.get("compared", []):
            if row["kind"] not in ("time", "rate") or row["path"].count("/") > 3:
                continue
            lines.append(f"| {name} | {row['path']} | {row['baseline']:.4g} | {row['final']:.4g} "
                         f"| {row['speedup']:.2f}x |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    report = compare()
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    if args.markdown:
        args.markdown.write_text(markdown(report))
    for name, data in report.items():
        rows = [r for r in data.get("compared", []) if r["kind"] in ("time", "rate")]
        if not rows:
            print(name, data.get("missing", "no comparable measurements"))
            continue
        ups = sorted(r["speedup"] for r in rows)
        print(f"{name}: {len(rows)} timings, speed-up min {ups[0]:.2f}x, median "
              f"{ups[len(ups) // 2]:.2f}x, max {ups[-1]:.2f}x; values changed: "
              f"{len(data['changed_values'])}, unpaired: {len(data['only_in_one'])}")  # fmt: skip


if __name__ == "__main__":
    main()
