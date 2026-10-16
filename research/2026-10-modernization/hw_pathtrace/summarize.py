"""Phase 5 prototype: the Markdown tables of README.md from results/*.json.

    python summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["cornell", "metals", "window-room", "castle", "hero", "castle-ao"]


def pct(x: float) -> str:
    return f"{100 * x:.4f} %"


def main() -> None:
    rows = {}
    for f in sorted((HERE / "results").glob("*.json")):
        rows[f.stem] = json.loads(f.read_text())
    names = [n for n in ORDER if n in rows]
    print("| Scene | Size, samples | Triangles (instances) | Software trace (GPU) | Ray queries (GPU) | Trace | End to end |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for n in names:
        d = rows[n]
        t = d["timings"]
        sw, hw, sp, e2e = t["sw"], t["hw"], t["speedup"], t["end_to_end_ms"]
        w, h = d["size"]
        print(
            f"| {n} | {w}x{h}, {d['samples']} | {d['triangles']:,} ({d['instances']}) "
            f"| {sw['trace_ms'] / 1e3:.2f} s ({sw['trace_gpu_ms'] / 1e3:.2f} s) "
            f"| {hw['trace_ms'] / 1e3:.2f} s ({hw['trace_gpu_ms'] / 1e3:.2f} s) "
            f"| {sp['trace_wall']:.2f}x | {e2e['sw_composed'] / 1e3:.2f} -> "
            f"{e2e['hw_composed'] / 1e3:.2f} s: {sp['end_to_end']:.2f}x |"
        )
    print()
    print("| Scene | Camera rays: same triangle | Bounce rays: same triangle | Any-hit: same verdict | Rays that differ (same point / one hit / elsewhere) | Image: difference / sample noise | Mean bias | Host = render_scene |")
    print("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for n in names:
        d = rows[n]
        r, im = d["rays"], d["image"]
        c, b, a = r["camera"], r["bounce"], r["any_hit"]
        differ = [
            f"{x['differ_same_point']} / {x['differ_hit_sw_only'] + x['differ_hit_hw_only']} / {x['differ_elsewhere']}"
            for x in (c, b)
        ]
        host = "bitwise" if d["host_vs_python"]["values_bitwise_equal"] == 1.0 else "differs"
        print(
            f"| {n} | {pct(c['same_triangle'])} of {c['rays']:,} | {pct(b['same_triangle'])} of {b['rays']:,} "
            f"| {pct(a['same_verdict'])} of {a['rays']:,} | camera {differ[0]}; bounce {differ[1]} "
            f"| {im['difference_over_noise']:.1e} | {im['mean_bias']:.1e} | {host} |"
        )
    print()
    print("| Scene | Fallback rays | Pass-through | AS build | Software traversal share (lower bound) |")
    print("| --- | ---: | --- | ---: | ---: |")
    for n in names:
        d = rows[n]
        t = d["timings"]
        share = 1 - t["hw"]["trace_gpu_ms"] / t["sw"]["trace_gpu_ms"]
        print(
            f"| {n} | {t['hw']['fallbacks']:,} | {'yes' if d['pass_through'] else 'no'} "
            f"| {t['hw']['as_build_ms']:.2f} ms | {100 * share:.0f} % |"
        )


if __name__ == "__main__":
    main()
