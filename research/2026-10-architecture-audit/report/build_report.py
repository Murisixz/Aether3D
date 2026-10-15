"""Build the Aether3D architecture & language audit page from the saved measurement files.

Every table of measured numbers is generated from the JSON files written by the experiment
scripts under ../lab; the few values that were only printed by a run are listed in HARD below,
with the script that printed them.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
L = HERE.parent / "results"  # the raw results, under the names used below
OUT = HERE / "aether3d-architecture-audit.html"


def J(p: str):
    return json.loads((L / p).read_text())


e1 = J("e1_world_matrices.json")
e2c = J("e2_instance_data_cpu.json")
e2g = J("e2_instance_data_gpu.json")
e3 = J("e3_traversal.json")
e3mt = J("e3_traversal_20_threads_interleaved.json")
emb = J("e3_embree.json")
ffi = J("e4_boundary.json")
e5 = J("e5_topology.json")
e6 = J("e6_png.json")
sub = J("subsystems.json")
imp = J("imports.json")
agree = J("e7_agreement.json")
e7 = [json.loads(x) for x in (L / "e7_runs.jsonl").read_text().splitlines() if x.strip()]

# Values printed (not saved as JSON) by runs in this session.
HARD = {
    # lab/e1/aether_world.py: Aether3D's own world_matrices(scene) on the same hierarchy
    "aether_world_ms": {"10000": 14.52, "100000": 279.17},
    # lab/e3/bench_cpp.py: the traversal kernel compiled as C++ (g++), 1 thread, Mrays/s
    "cpp_st": {
        "sphere 100k tris | camera 640x480": 6.02,
        "sphere 100k tris | bounce (diffuse)": 1.30,
        "400 cube instances | camera 640x480": 8.24,
        "400 cube instances | bounce (diffuse)": 7.06,
        "sphere 1M tris | camera 640x480": 3.89,
        "sphere 1M tris | bounce (diffuse)": 0.72,
    },
    # lab/e4/prof_create.py
    "plan_10k_ms": 89.1,
    "floor_us": 0.78,
    "uuid_us": 0.65,
    # lab/e3/numba_latency.py
    "numba_import_ms": 153,
    "numba_compile_s": 1.59,
    "numba_parallel_compile_s": 0.96,
    # ls of lab/e3/*.so
    "sizes_kb": {"C": 19, "C++": 19, "Zig": 53, "Rust": 369, "C# Native AOT": 1400, "Go": 2046},
    # PNG encode split on examples/output/castle_path.png (1920x1080 RGB)
    "png_filter_ms": 195.1,
    "zlib_ms": {1: 56.3, 3: 117.7, 6: 371.2, 9: 405.3},
    "zlib_mb": {1: 3.50, 3: 3.41, 6: 3.33, 9: 3.33},
    # lab/e5/prof_from_mesh.py (25,280 triangles, cProfile, cumulative seconds)
    "from_mesh_profile": {"total": 2.752, "_join_pairs": 1.953, "_new_face": 0.344, "_infer_shading": 0.287},
    # pre-compaction measurements of this audit (lab/m/subsystems.py and editing probes)
    "bevel_51k_s": 16.8,
    "subdiv_51k_ms": 168,
    "undo_redo_10k_ms": (2.5, 2.8),
    "raycast_traversal_ms": 1.7,
    "overlaps_first_ms": 175,
}

WORKLOADS = [
    "sphere 100k tris | camera 640x480",
    "sphere 100k tris | bounce (diffuse)",
    "400 cube instances | camera 640x480",
    "400 cube instances | bounce (diffuse)",
    "sphere 1M tris | camera 640x480",
    "sphere 1M tris | bounce (diffuse)",
]


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def num(x, nd=1) -> str:
    if isinstance(x, str):
        return x
    if x is None:
        return "–"
    if nd == 0:
        return f"{x:,.0f}"
    return f"{x:,.{nd}f}"


def sig(x) -> str:
    """Compact significant formatting for rates and ratios."""
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def tag(kind: str) -> str:
    label = {"m": "measured", "r": "repo docs", "e": "estimate", "c": "cited"}[kind]
    return f'<span class="tag {kind}" title="{label}">{kind.upper()}</span>'


VERDICT_CLASS = {
    "KEEP": "keep",
    "OPTIMIZE": "opt",
    "MOVE TO NATIVE": "native",
    "REARCHITECT": "rearch",
    "DEFER": "defer",
}


def chip(v: str) -> str:
    return f'<span class="chip {VERDICT_CLASS[v]}">{esc(v)}</span>'


def table(headers, rows, cls="", caption=None, numeric_cols=()) -> str:
    out = [f'<div class="table-wrap"><table class="{cls}">']
    if caption:
        out.append(f"<caption>{caption}</caption>")
    out.append("<thead><tr>")
    for i, h in enumerate(headers):
        c = ' class="n"' if i in numeric_cols else ""
        out.append(f"<th{c}>{h}</th>")
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>")
        for i, cell in enumerate(r):
            c = ' class="n"' if i in numeric_cols else ""
            out.append(f"<td{c}>{cell}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def code(s: str) -> str:
    return f"<code>{esc(s)}</code>"


# ----------------------------------------------------------------------------------------------
# SVG charts (drawn to scale; text and marks colored from theme tokens)


def hbar_log(items, vmin, vmax, unit, width=680, label_w=190, row_h=24, title=""):
    """Horizontal bars on a log axis. items: (label, value, css_class)."""
    plot_w = width - label_w - 70
    top = 26
    h = top + row_h * len(items) + 30
    lo, hi = math.log10(vmin), math.log10(vmax)

    def x(v):
        return label_w + (math.log10(max(v, vmin)) - lo) / (hi - lo) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    parts.append(f'<text x="{label_w}" y="14" class="ax-title">{esc(title)}</text>')
    d = math.floor(lo)
    while d <= math.ceil(hi):
        for m in (1, 2, 5):
            v = m * 10**d
            if vmin <= v <= vmax:
                xx = x(v)
                major = m == 1
                parts.append(
                    f'<line x1="{xx:.1f}" y1="{top - 4}" x2="{xx:.1f}" y2="{top + row_h * len(items)}" class="{"grid-major" if major else "grid"}"/>'
                )
                if major or (hi - lo) < 2.5:
                    lab = f"{v:g}"
                    parts.append(
                        f'<text x="{xx:.1f}" y="{top + row_h * len(items) + 16}" class="tick" text-anchor="middle">{lab}</text>'
                    )
        d += 1
    parts.append(
        f'<text x="{label_w + plot_w}" y="{top + row_h * len(items) + 28}" class="tick" text-anchor="end">{esc(unit)} (log scale)</text>'
    )
    for i, (label, v, cls) in enumerate(items):
        y = top + i * row_h
        parts.append(f'<text x="{label_w - 8}" y="{y + row_h * 0.66:.1f}" class="lab" text-anchor="end">{esc(label)}</text>')
        x1 = x(v)
        parts.append(f'<rect x="{label_w}" y="{y + 4}" width="{max(x1 - label_w, 1):.1f}" height="{row_h - 8}" class="{cls}"/>')
        parts.append(f'<text x="{x1 + 6:.1f}" y="{y + row_h * 0.66:.1f}" class="val">{sig(v)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def stacked(rows, segs, vmax, unit, width=680, label_w=230, row_h=34, title=""):
    """rows: (label, [values per seg]); segs: (name, css_class)."""
    plot_w = width - label_w - 60
    top = 30
    legend_h = 22 * math.ceil(len(segs) / 4)
    h = top + row_h * len(rows) + 26 + legend_h + 6

    def sx(v):
        return v / vmax * plot_w

    parts = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    parts.append(f'<text x="{label_w}" y="16" class="ax-title">{esc(title)}</text>')
    step = 10 if vmax <= 120 else 20
    t = 0
    while t <= vmax + 1e-9:
        xx = label_w + sx(t)
        parts.append(f'<line x1="{xx:.1f}" y1="{top - 4}" x2="{xx:.1f}" y2="{top + row_h * len(rows)}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top + row_h * len(rows) + 15}" class="tick" text-anchor="middle">{t:g}</text>')
        t += step
    parts.append(
        f'<text x="{label_w + plot_w}" y="{top + row_h * len(rows) + 28}" class="tick" text-anchor="end">{esc(unit)}</text>'
    )
    for i, (label, vals) in enumerate(rows):
        y = top + i * row_h
        parts.append(f'<text x="{label_w - 8}" y="{y + row_h * 0.6:.1f}" class="lab" text-anchor="end">{esc(label)}</text>')
        acc = 0.0
        for (name, cls), v in zip(segs, vals):
            if v <= 0:
                continue
            parts.append(
                f'<rect x="{label_w + sx(acc):.2f}" y="{y + 6}" width="{max(sx(v), 0.8):.2f}" height="{row_h - 12}" class="{cls}"><title>{esc(name)}: {v:g} ms</title></rect>'
            )
            acc += v
        parts.append(f'<text x="{label_w + sx(acc) + 6:.1f}" y="{y + row_h * 0.6:.1f}" class="val">{acc:.1f}</text>')
    ly = top + row_h * len(rows) + 40
    for k, (name, cls) in enumerate(segs):
        col, rr = k % 4, k // 4
        lx = label_w + col * (plot_w / 4)
        parts.append(f'<rect x="{lx:.1f}" y="{ly + rr * 22 - 10}" width="12" height="12" class="{cls}"/>')
        parts.append(f'<text x="{lx + 17:.1f}" y="{ly + rr * 22}" class="tick">{esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped(groups, series, vmax, unit, width=680, label_w=190, title=""):
    """groups: (label, [values per series]); series: (name, css_class). Linear axis."""
    row_h = 16 * len(series) + 12
    plot_w = width - label_w - 70
    top = 30
    h = top + row_h * len(groups) + 48

    def sx(v):
        return v / vmax * plot_w

    parts = [f'<svg viewBox="0 0 {width} {h}" role="img" aria-label="{esc(title)}" class="chart">']
    parts.append(f'<text x="{label_w}" y="16" class="ax-title">{esc(title)}</text>')
    t = 0
    while t <= vmax + 1e-9:
        xx = label_w + sx(t)
        parts.append(f'<line x1="{xx:.1f}" y1="{top - 4}" x2="{xx:.1f}" y2="{top + row_h * len(groups)}" class="grid"/>')
        parts.append(f'<text x="{xx:.1f}" y="{top + row_h * len(groups) + 15}" class="tick" text-anchor="middle">{t:g}</text>')
        t += 1
    parts.append(f'<text x="{label_w + plot_w}" y="{top + row_h * len(groups) + 30}" class="tick" text-anchor="end">{esc(unit)}</text>')
    for i, (label, vals) in enumerate(groups):
        y = top + i * row_h
        parts.append(f'<text x="{label_w - 8}" y="{y + row_h / 2 + 4:.1f}" class="lab" text-anchor="end">{esc(label)}</text>')
        for j, ((name, cls), v) in enumerate(zip(series, vals)):
            yy = y + 6 + j * 16
            parts.append(f'<rect x="{label_w}" y="{yy}" width="{max(sx(v), 1):.1f}" height="12" class="{cls}"><title>{esc(name)}</title></rect>')
            parts.append(f'<text x="{label_w + sx(v) + 6:.1f}" y="{yy + 10}" class="val">{v:.2f}</text>')
    ly = top + row_h * len(groups) + 36
    for k, (name, cls) in enumerate(series):
        lx = label_w + k * 220
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" class="{cls}"/>')
        parts.append(f'<text x="{lx + 17}" y="{ly}" class="tick">{esc(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------------------------------
# Derived numbers used in the prose

st = sub["gpu_stage_ms"]
frame_same, frame_moved = sub["gpu_frame_unchanged_ms"], sub["gpu_frame_one_object_moved_ms"]
e1_10k = e1["10000"]
cur_w10, cur_w100 = HARD["aether_world_ms"]["10000"], HARD["aether_world_ms"]["100000"]
np_w10 = e1_10k["numpy (Aether3D algorithm, flat arrays)"]["median_ms"]
native_w10 = [v["median_ms"] for k, v in e1_10k.items() if k not in ("pure Python", "numpy (Aether3D algorithm, flat arrays)") and "FMA" not in k]
st_np = [e3[w]["NumPy (current)"]["1"] for w in WORKLOADS]
st_c = [e3[w]["C (gcc -O3)"]["1"] for w in WORKLOADS]
st_ratio = [c / n for c, n in zip(st_c, st_np)]
mt_c = [e3mt[w]["C (gcc)"] for w in WORKLOADS]
mt_ratio = [c / n for c, n in zip(mt_c, st_np)]
emb_st = [emb[w.split(" | ")[0]][w.split(" | ")[1]]["x1"] for w in WORKLOADS]
emb_over_c = [a / b for a, b in zip(emb_st, st_c)]
llvm_vs_gcc = [e3[w][k]["1"] / e3[w]["C (gcc -O3)"]["1"] for w in WORKLOADS for k in ("C (clang -O3)", "Rust 1.99", "Zig 0.17")]
llvm_spread = max(max(e3[w][k]["1"] for k in ("C (clang -O3)", "Rust 1.99", "Zig 0.17")) / min(e3[w][k]["1"] for k in ("C (clang -O3)", "Rust 1.99", "Zig 0.17")) for w in WORKLOADS) - 1
llvm_ratio = [e3[w][k]["1"] / e3[w]["NumPy (current)"]["1"] for w in WORKLOADS for k in ("C (clang -O3)", "Rust 1.99", "Zig 0.17")]
slow_ratio = [e3[w][k]["1"] / e3[w]["NumPy (current)"]["1"] for w in WORKLOADS for k in ("Go 1.27", "C# .NET 10 AOT")]
numba_vs_c = [e3[w]["Numba 0.68"]["1"] / e3[w]["C (gcc -O3)"]["1"] for w in WORKLOADS]
scale20 = [e3mt[w][k] / e3[w][k1]["1"] for w in WORKLOADS for k, k1 in (("C (gcc)", "C (gcc -O3)"), ("Rust", "Rust 1.99"), ("Zig", "Zig 0.17"))]
go_vs_c = [e3[w]["Go 1.27"]["1"] / e3[w]["C (gcc -O3)"]["1"] for w in WORKLOADS]
cs_vs_c = [e3[w]["C# .NET 10 AOT"]["1"] / e3[w]["C (gcc -O3)"]["1"] for w in WORKLOADS]
gpu_rows = {(r["scene"], r["rays"], r["kernel"]): r for r in e7 if "kernel" in r}
gpu_build = {r["scene"]: r for r in e7 if "as_build_ms" in r}
hw_ratio = [gpu_rows[(s, r, "hw")]["mrays_per_s"] / gpu_rows[(s, r, "sw")]["mrays_per_s"] for s in ("sphere100k", "cubes400", "sphere1m") for r in ("camera", "bounce")]
hw_rates = [gpu_rows[(s, r, "hw")]["mrays_per_s"] for s in ("sphere100k", "cubes400", "sphere1m") for r in ("camera", "bounce")]
sw_rates = [gpu_rows[(s, r, "sw")]["mrays_per_s"] for s in ("sphere100k", "cubes400", "sphere1m") for r in ("camera", "bounce")]
min_same_tri = min(min(v["sw"]["same triangle as ref (both hit)"], v["hw"]["same triangle as ref (both hit)"]) for v in agree.values())
topo_cur_ms = e5["EditableMesh.from_mesh(quads=False) s"] * 1e3
topo_np = e5["NumPy (vectorized) ms"] + e5["weld (NumPy, shared) ms"]
topo_rs = e5["Rust (std HashMap, fast hasher) ms"] + e5["weld (NumPy, shared) ms"]
png_a = [v["Aether3D decode MP/s"] for v in e6.values()]
png_p = [v["Pillow decode MP/s"] for v in e6.values()]
png_c = [v["zlib + C unfilter MP/s"] for v in e6.values()]

# Composed frame estimates (ms), from measured components; see section 8.
est_one = {
    "evaluate": st["evaluate"],
    "materials (cached)": 0.5,
    "extract (1 dirty object)": 0.1,
    "upload (1 record)": round((e2c["dirty_1_numpy_us"] + e2g["write_buffer_1_records_one_range_us"]) / 1e3, 3),
    "prepare": st["prepare"],
    "gpu": st["gpu"],
    "readback (all outputs)": st["readback"],
}
est_one_total = sum(est_one.values())
img_only_readback = st["readback"] * 4 / 28
est_one_img = est_one_total - st["readback"] + img_only_readback
est_one_none = est_one_total - st["readback"]
est_all = {
    "evaluate": st["evaluate"],
    "materials (cached)": 0.5,
    "world matrices (NumPy, flat)": np_w10,
    "pack all (NumPy, flat)": e2c["full_pack_numpy_ms"],
    "write 1.28 MB": e2g["write_buffer_full_1.28MB_us"] / 1e3,
    "prepare": st["prepare"],
    "gpu": st["gpu"],
    "readback (all outputs)": st["readback"],
}
est_all_total = sum(est_all.values())


def rng(vals, fmt=sig, unit=""):
    lo, hi = min(vals), max(vals)
    return f"{fmt(lo)}–{fmt(hi)}{unit}"


# ----------------------------------------------------------------------------------------------
# Page

CSS = r"""
<title>Aether3D Language Audit</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@85,600;85,700;100,700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
/* Layout: engineering report. A sticky contents rail on wide screens, a 46rem reading column,
   and wide tables/diagrams that scroll inside their own frames. */
:root {
  --bg: #f5f6f4; --surface: #ffffff; --fg: #1b1f22; --muted: #5b6468; --line: #d8dcd9;
  --accent: #23577e;      /* steel blue: links, headings, CPU bars */
  --accent-2: #12806f;    /* teal: GPU and hardware */
  --warm: #a4561a;        /* copper: today's code, estimates */
  --code-bg: #eceeea;
  --keep: #2c7a4b; --opt: #23577e; --native: #6b3f9e; --rearch: #9a5a0c; --defer: #6a7276;
  --tag-m: #23577e; --tag-r: #12806f; --tag-e: #a4561a; --tag-c: #6a7276;
  --font-display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --font-body: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121516; --surface: #191d1f; --fg: #e5e8e6; --muted: #9aa3a6; --line: #2c3335;
    --accent: #84b6dd; --accent-2: #4fc4af; --warm: #e39b5e; --code-bg: #20262a;
    --keep: #6ccb93; --opt: #84b6dd; --native: #b996ea; --rearch: #e3a65a; --defer: #a3abaf;
    --tag-m: #84b6dd; --tag-r: #4fc4af; --tag-e: #e39b5e; --tag-c: #a3abaf;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #121516; --surface: #191d1f; --fg: #e5e8e6; --muted: #9aa3a6; --line: #2c3335;
  --accent: #84b6dd; --accent-2: #4fc4af; --warm: #e39b5e; --code-bg: #20262a;
  --keep: #6ccb93; --opt: #84b6dd; --native: #b996ea; --rearch: #e3a65a; --defer: #a3abaf;
  --tag-m: #84b6dd; --tag-r: #4fc4af; --tag-e: #e39b5e; --tag-c: #a3abaf;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 16px; line-height: 1.6; }
.page { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2rem; max-width: 76rem; margin: 0 auto;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 2.5rem 4rem; }
@media (min-width: 1100px) { .page { grid-template-columns: 14.5rem minmax(0, 1fr); } }
nav.toc { display: none; }
@media (min-width: 1100px) {
  nav.toc { display: block; position: sticky; top: calc(env(safe-area-inset-top, 0px) + 1.5rem); align-self: start;
    max-height: calc(100vh - 3rem); overflow-y: auto; font-size: 0.84rem; line-height: 1.45; }
}
nav.toc .toc-title { font-family: var(--font-display); font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.6rem; }
nav.toc ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.3rem; }
nav.toc a { color: var(--fg); text-decoration: none; display: block; padding: 0.1rem 0 0.1rem 0.6rem; border-left: 2px solid var(--line); }
nav.toc a:hover, nav.toc a:focus-visible { border-left-color: var(--accent); color: var(--accent); }
main { min-width: 0; }
.col { max-width: 46rem; }
header.doc { display: grid; gap: 0.9rem; margin-bottom: 2.5rem; }
.eyebrow { font-family: var(--font-mono); font-size: 0.78rem; color: var(--muted); letter-spacing: 0.02em; }
h1 { font-family: var(--font-display); font-stretch: 85%; font-weight: 700; font-size: clamp(2rem, 4.4vw, 3rem); line-height: 1.08; margin: 0; text-wrap: balance; letter-spacing: -0.01em; }
h2 { font-family: var(--font-display); font-stretch: 85%; font-weight: 700; font-size: 1.65rem; line-height: 1.2; margin: 3.2rem 0 0.9rem; text-wrap: balance; padding-top: 0.9rem; border-top: 2px solid var(--fg); }
h3 { font-family: var(--font-display); font-weight: 700; font-size: 1.12rem; margin: 2rem 0 0.5rem; text-wrap: balance; }
h4 { font-size: 0.95rem; margin: 1.4rem 0 0.4rem; }
p, li { max-width: 46rem; }
p { margin: 0.6rem 0; }
ul, ol { padding-left: 1.3rem; }
li { margin: 0.25rem 0; }
a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible, button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
code { font-family: var(--font-mono); font-size: 0.84em; background: var(--code-bg); padding: 0.08em 0.32em; border-radius: 3px; overflow-wrap: anywhere; }
.meta { display: flex; flex-wrap: wrap; gap: 0.4rem 1.4rem; font-size: 0.86rem; color: var(--muted); }
.meta b { color: var(--fg); font-weight: 600; }
.status { border: 1px solid var(--line); background: var(--surface); padding: 0.8rem 1rem; font-size: 0.9rem; max-width: 46rem; }
.verdict { border-left: 4px solid var(--accent); background: var(--surface); padding: 1.1rem 1.3rem; margin: 1.2rem 0 1.4rem; max-width: 46rem; }
.verdict p.big { font-family: var(--font-display); font-weight: 700; font-size: 1.28rem; line-height: 1.3; margin: 0 0 0.5rem; text-wrap: balance; }
.findings { counter-reset: f; list-style: none; padding: 0; display: grid; gap: 0.7rem; }
.findings li { padding-left: 2.1rem; position: relative; }
.findings li::before { counter-increment: f; content: counter(f); position: absolute; left: 0; top: 0.1rem; font-family: var(--font-mono); font-size: 0.8rem; font-weight: 600; color: var(--accent); border: 1px solid var(--accent); width: 1.45rem; height: 1.45rem; display: grid; place-items: center; border-radius: 50%; }
.table-wrap { overflow-x: auto; margin: 0.9rem 0 1.3rem; border: 1px solid var(--line); background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 0.84rem; line-height: 1.4; font-variant-numeric: tabular-nums; }
caption { caption-side: top; text-align: left; padding: 0.6rem 0.75rem 0.2rem; font-size: 0.8rem; color: var(--muted); }
th, td { padding: 0.42rem 0.65rem; border-bottom: 1px solid var(--line); vertical-align: top; text-align: left; }
th { font-weight: 600; font-size: 0.76rem; letter-spacing: 0.02em; color: var(--muted); background: var(--bg); position: sticky; top: 0; }
td.n, th.n { text-align: right; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td .sub { display: block; color: var(--muted); font-size: 0.74rem; }
.wide table { min-width: 56rem; }
.xwide table { min-width: 72rem; }
.tag { display: inline-block; font-family: var(--font-mono); font-size: 0.62rem; font-weight: 600; letter-spacing: 0.04em; padding: 0 0.28rem; border-radius: 2px; vertical-align: 0.12em; margin-left: 0.15rem; border: 1px solid; }
.tag.m { color: var(--tag-m); border-color: var(--tag-m); }
.tag.r { color: var(--tag-r); border-color: var(--tag-r); }
.tag.e { color: var(--tag-e); border-color: var(--tag-e); }
.tag.c { color: var(--tag-c); border-color: var(--tag-c); }
.chip { display: inline-block; font-family: var(--font-mono); font-size: 0.66rem; font-weight: 600; letter-spacing: 0.03em; padding: 0.12rem 0.4rem; border-radius: 2px; white-space: nowrap; border: 1px solid currentColor; }
.chip.keep { color: var(--keep); } .chip.opt { color: var(--opt); } .chip.native { color: var(--native); }
.chip.rearch { color: var(--rearch); } .chip.defer { color: var(--defer); }
.legend { display: flex; flex-wrap: wrap; gap: 0.5rem 1.2rem; font-size: 0.84rem; color: var(--muted); margin: 0.6rem 0 1rem; }
.figure { margin: 1rem 0 1.4rem; overflow-x: auto; border: 1px solid var(--line); background: var(--surface); padding: 0.8rem; }
.figure .cap { font-size: 0.8rem; color: var(--muted); margin-top: 0.4rem; max-width: 46rem; }
svg.chart { width: 100%; min-width: 34rem; height: auto; display: block; font-family: var(--font-body); }
svg.chart .ax-title { font-size: 12px; font-weight: 600; fill: var(--fg); }
svg.chart .lab { font-size: 11.5px; fill: var(--fg); }
svg.chart .val { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
svg.chart .tick { font-size: 10.5px; fill: var(--muted); }
svg.chart .grid { stroke: var(--line); stroke-width: 1; }
svg.chart .grid-major { stroke: var(--muted); stroke-opacity: 0.45; stroke-width: 1; }
.b-cur { fill: var(--warm); } .b-cpu { fill: var(--accent); } .b-cpu2 { fill: var(--accent); fill-opacity: 0.55; }
.b-gpu { fill: var(--accent-2); } .b-gpu2 { fill: var(--accent-2); fill-opacity: 0.5; } .b-mut { fill: var(--muted); fill-opacity: 0.55; }
.s1 { fill: var(--muted); fill-opacity: 0.35; } .s2 { fill: var(--warm); fill-opacity: 0.55; } .s3 { fill: var(--warm); }
.s4 { fill: var(--accent); fill-opacity: 0.75; } .s5 { fill: var(--muted); fill-opacity: 0.6; } .s6 { fill: var(--accent-2); } .s7 { fill: var(--native); fill-opacity: 0.8; }
pre.mermaid { background: var(--surface); border: 1px solid var(--line); padding: 1rem; overflow-x: auto; margin: 1rem 0 1.4rem; font-size: 0.85rem; }
.two { display: grid; grid-template-columns: minmax(0, 1fr); gap: 1rem; }
@media (min-width: 760px) { .two { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); } }
.card { border: 1px solid var(--line); background: var(--surface); padding: 0.9rem 1rem; min-width: 0; }
.card h4 { margin-top: 0; }
.note { font-size: 0.86rem; color: var(--muted); max-width: 46rem; }
.myth { display: grid; grid-template-columns: minmax(0, 1fr); gap: 0.2rem 1rem; padding: 0.8rem 0; border-bottom: 1px solid var(--line); }
@media (min-width: 760px) { .myth { grid-template-columns: minmax(0, 16rem) minmax(0, 1fr); } }
.myth .claim { font-weight: 600; }
.myth .claim::before { content: "Claim  "; font-family: var(--font-mono); font-size: 0.68rem; color: var(--warm); letter-spacing: 0.04em; }
.refs { font-size: 0.84rem; }
.refs li { margin: 0.35rem 0; overflow-wrap: anywhere; }
.kbd { font-family: var(--font-mono); font-size: 0.8rem; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
html { scroll-behavior: smooth; }
</style>
"""

sections: list[tuple[str, str, str]] = []  # (id, toc title, html)


def section(sid: str, title: str, body: str):
    sections.append((sid, title, f'<section id="{sid}"><h2>{title}</h2>{body}</section>'))


# ---------------------------------------------------------------- 1. Answer
best_frame = f"{est_one_img:.0f}"
summary = f"""
<div class="verdict">
<p class="big">Keep Python as the engine's language. Fix the architecture first, then add one native language, Rust through PyO3, for a short list of scalar kernels. Treat hardware ray tracing as a separate, later decision.</p>
<p>The slow parts of Aether3D are slow because of how data is laid out and when it is recomputed. A warm 10,000-object GPU frame takes {frame_same:.0f} ms, and the GPU works for {st['gpu']:.1f} ms of it. Moving one object costs the same {frame_moved:.0f} ms because every frame rebuilds everything. Rewriting the same per-object, per-frame design in C++ or Rust would leave that O(N) work in place.</p>
</div>

<h3>Key findings</h3>
<ol class="findings">
<li><b>Architecture beats language on the hot paths.</b> In NumPy on flat arrays, world matrices drop from {cur_w10:.1f} ms to {np_w10:.1f} ms {tag('m')} and instance packing from {e2c['pack_instances_ms']:.1f} ms to {e2c['full_pack_numpy_ms']:.2f} ms {tag('m')}. On a retained GPU buffer, updating one changed instance costs about {e2c['dirty_1_numpy_us'] + e2g['write_buffer_1_records_one_range_us']:.0f} µs. Today the whole frame is repacked and a new 1.28 MB buffer is created ({e2c['pack_instances_ms']:.1f} ms + {e2g['create_buffer_with_data_1.28MB_us'] / 1e3:.2f} ms) {tag('m')}. Combined from measured parts, a frame with one changed object would take about {est_one_total:.0f} ms, or about {est_one_img:.0f} ms if only the image is read back, against {frame_same:.0f} ms today {tag('e')}.</li>
<li><b>Where the work is truly scalar, compiled code wins by an order of magnitude, with bitwise determinism.</b> One-thread BVH traversal runs {rng(st_ratio, lambda x: f'{x:.0f}')}× faster than today's NumPy wavefront tracer; on 20 threads it is {rng(mt_ratio, lambda x: f'{x:.0f}')}× faster. Results are bit-identical in C, C++, Rust, Zig, Go, C# and Numba {tag('m')}. Mesh-topology construction is {topo_cur_ms / topo_np:.0f}× faster in vectorized NumPy and {topo_cur_ms / topo_rs:.0f}× faster in Rust, with identical edge ids {tag('m')}. The PNG unfilter step is about 50× faster in C {tag('m')}.</li>
<li><b>The compiler backend and data layout matter more than the language.</b> clang, Rust and Zig share the LLVM backend and land within {llvm_spread * 100:.0f}% of each other. GCC-compiled C/C++ is {rng([1 / x for x in llvm_vs_gcc], lambda x: f'{x:.1f}')}× faster than them on the traversal kernel. Go reaches {rng(go_vs_c, lambda x: f'{x:.2f}')}× of C, and C# Native AOT {rng(cs_vs_c, lambda x: f'{x:.2f}')}×. Embree's SIMD BVH8 is a further {rng(emb_over_c, lambda x: f'{x:.1f}')}× faster than the best scalar port {tag('m')}.</li>
<li><b>Binding cost is no reason to avoid native code.</b> A call through nanobind, PyO3, Cython or the CPython C API costs 11–13 ns with no arguments, the same as a Python function call. ctypes costs 80 ns with no arguments and 464 ns with three floats {tag('m')}.</li>
<li><b>Hardware ray tracing is fast but out of reach from Python today.</b> On the RTX 4060, wgpu ray queries trace {rng([x / 1e3 for x in hw_rates], lambda x: f'{x:.2f}')} G rays/s. Aether3D's own WGSL traversal manages {rng([x / 1e3 for x in sw_rates], lambda x: f'{x:.2f}')} G rays/s, so hardware is {rng(hw_ratio, lambda x: f'{x:.1f}')}× faster, with ≥{min_same_tri * 100:.3f}% of rays hitting the same triangle {tag('m')}. But wgpu-native's C API, which wgpu-py binds, has no acceleration-structure calls {tag('m')}. Hardware RT therefore needs a native Rust wgpu host. The current GPU path tracer also spends most of its time shading, not traversing {tag('r')}.</li>
<li><b>Timing: phase it in.</b> After PR #13 merges, start with a performance-foundation phase that needs no new toolchain: retained render state, dirty tracking, caches, persistent GPU buffers and selective readback. Add the Rust crate only once parity tests and performance gates exist, starting with the CPU BVH. Defer the native GPU host until a product need justifies it.</li>
</ol>

<h3>The numbers behind the recommendation</h3>
""" + table(
    ["What", "Today", "Best measured alternative", "Factor", "Basis"],
    [
        ["GPU frame, 10k objects, 1280×720 (one object moved)", f"{frame_moved:.0f} ms", f"~{est_one_img:.0f} ms retained + image-only readback", f"~{frame_moved / est_one_img:.0f}×", f"E2 components {tag('e')}"],
        ["World matrices, 10k-object hierarchy", f"{cur_w10:.1f} ms", f"{np_w10:.2f} ms NumPy flat · {min(native_w10):.2f} ms native", f"{cur_w10 / np_w10:.0f}× · {cur_w10 / min(native_w10):.0f}×", f"E1 {tag('m')}"],
        ["Instance packing, 10k", f"{e2c['pack_instances_ms']:.1f} ms", f"{e2c['full_pack_numpy_ms']:.2f} ms NumPy flat · {e2c['full_pack_c_ms']:.3f} ms C", f"{e2c['pack_instances_ms'] / e2c['full_pack_numpy_ms']:.0f}× · {e2c['pack_instances_ms'] / e2c['full_pack_c_ms']:.0f}×", f"E2 {tag('m')}"],
        ["CPU closest-hit, sphere 1M tris, diffuse rays", f"{st_np[5]:.2f} M rays/s", f"{e3mt[WORKLOADS[5]]['C (gcc)']:.1f} M rays/s C ×20 · {emb[WORKLOADS[5].split(' | ')[0]][WORKLOADS[5].split(' | ')[1]]['x20']:.1f} Embree ×20", f"{e3mt[WORKLOADS[5]]['C (gcc)'] / st_np[5]:.0f}× · {emb['sphere 1M tris']['bounce (diffuse)']['x20'] / st_np[5]:.0f}×", f"E3 {tag('m')}"],
        ["GPU traversal, sphere 1M, diffuse rays", f"{gpu_rows[('sphere1m', 'bounce', 'sw')]['mrays_per_s']:.0f} M rays/s (WGSL)", f"{gpu_rows[('sphere1m', 'bounce', 'hw')]['mrays_per_s']:,.0f} M rays/s (ray query)", f"{gpu_rows[('sphere1m', 'bounce', 'hw')]['mrays_per_s'] / gpu_rows[('sphere1m', 'bounce', 'sw')]['mrays_per_s']:.0f}×", f"E7 {tag('m')}"],
        ["Mesh topology, 200k tris (same edge ids)", f"{topo_cur_ms:.0f} ms", f"{topo_np:.0f} ms NumPy · {topo_rs:.0f} ms Rust", f"{topo_cur_ms / topo_np:.0f}× · {topo_cur_ms / topo_rs:.0f}×", f"E5 {tag('m')}"],
        ["PNG decode, 1080p render", f"{min(png_a):.1f}–{max(png_a):.1f} MP/s", f"{min(png_c):.0f}–{max(png_c):.0f} MP/s zlib + C unfilter", f"~{min(png_c) / max(png_a):.0f}–{max(png_c) / min(png_a):.0f}×", f"E6 {tag('m')}"],
        ["Warm raycast, 10k objects", f"{sub['spatial_warm_raycast_ms']:.1f} ms", f"~{HARD['raycast_traversal_ms']} ms after a cache fix (traversal only)", f"~{sub['spatial_warm_raycast_ms'] / HARD['raycast_traversal_ms']:.1f}×", f"profile {tag('m')} {tag('e')}"],
    ],
    cls="wide",
)
section("summary", "1. The answer", summary)

# ---------------------------------------------------------------- 2. Method
method = f"""
<p>Scope: <code>main</code> at 38697d5 (Stages 1–10, v0.10.0) plus the Stage 11 branch at 403581f (PR #13, open, not merged). Paths below are relative to <code>src/aether3d/</code>.</p>
""" + table(
    ["Item", "Value"],
    [
        ["CPU", "Intel Core i9-12900H, 6 performance + 8 efficiency cores, 20 threads, L3 24 MiB"],
        ["GPU", "NVIDIA GeForce RTX 4060 Laptop GPU, Vulkan, driver 615.78.08 (Mesa lavapipe also present)"],
        ["OS, memory", "Arch Linux, kernel 7.2.9, 15 GiB"],
        ["Python stack", "CPython 3.14.8, NumPy 2.5.4, wgpu-py 0.32.0 on wgpu-native 29, Pillow 12.3.0 (lab venv only)"],
        ["Toolchains (scratchpad only)", "gcc 16.2.1, clang 23.1.1, Rust 1.99.1, Zig 0.17.0, Go 1.27.2, .NET 10.0.402 (Native AOT), Numba 0.68, Cython 3.3, nanobind 3.1, PyO3 0.30.1, cffi 2.1, Embree 4.4.1 release binaries, Rust wgpu 30.0.1"],
    ],
) + f"""
<h3>Rules the experiments follow</h3>
<ul>
<li><b>Same algorithm, same data.</b> Every port reproduces Aether3D's own algorithm: the same arithmetic order, the same box and watertight triangle tests, and the same tie rules. Each one reads arrays exported from a real Aether3D scene.</li>
<li><b>Checked outputs.</b> Every port's output was compared with Aether3D's own output, and every comparison passed. Most are bitwise (world matrices, traversal hits and barycentrics, instance records, edge ids). Embree and the GPU kernels run in float32, so they are checked by agreement rate instead.</li>
<li><b>Timing.</b> One warm-up, then the median of 3–15 runs. Multi-threaded traversal was rerun with implementations interleaved in random order over 7 rounds, because laptop heat penalizes whatever runs later. GPU kernels were timed with timestamp queries.</li>
<li><b>Floating point.</b> FMA contraction is off everywhere, as it would have to be in production; E1 shows what changes when it is allowed.</li>
<li><b>Nothing touched the system.</b> Toolchains and packages went into the session scratchpad. No repository file changed, and no long CPU renders ran.</li>
</ul>
<div class="legend"><span>{tag('m')} measured in this audit</span><span>{tag('r')} measured by the repository's own benchmarks or docs</span><span>{tag('e')} estimate built from measured parts</span><span>{tag('c')} external source</span></div>
<h3>What was not tested</h3>
<ul>
<li>Swift, Julia, Java/Kotlin, D, Nim, Odin and Mojo: not installed, and none has a mainstream path to a small CPython extension wheel. They are assessed on architecture and ecosystem only (section 6), not on speed.</li>
<li>macOS (Metal, Apple Silicon) and Windows (D3D12, MSVC). All numbers are Linux on one laptop, so cross-platform speed and the bitwise-parity claims still have to be confirmed on the other CI operating systems.</li>
<li>The proposed retained frame was not built. Its time is assembled from measured components and labeled as an estimate.</li>
<li>Hardware ray tracing inside the real path tracer. E7 measures traversal alone.</li>
</ul>
"""
section("method", "2. How this was measured", method)

# ---------------------------------------------------------------- 3. Map
lines = imp["lines"]
mp = table(
    ["Subsystem", "Lines", "Language", "Runs on", "Data layout", "Cost profile (10k objects unless noted)"],
    [
        [code("scene.py · transform.py · history.py"), num(lines["scene"] + lines["transform"] + lines["history"] + lines["_history_file"], 0), "Python", "CPU, caller's thread", "Array of structures: one SceneObject (17 slots) + Transform (float tuples) + cloned Appearance per object; dicts by id and name", f"{sub['bytes_per_object_incl_history']} B/object; read position {sub['read_position_ns']} ns; world_matrix {sub['read_world_matrix_us']:.1f} µs (walks parents); set position {sub['set_position_us']:.1f} µs {tag('m')}"],
        [code("plans/") + " (Stage 11)", num(lines["plans"], 0), "Python", "CPU", "Plan records → one batch create", f"10k blocks in {HARD['plan_10k_ms']:.0f} ms ≈ {HARD['plan_10k_ms'] / 10:.1f} µs/object {tag('m')}"],
        [code("mcp/ · cli.py") + " (MCP: Stage 11)", num(lines["mcp"] + lines["cli"] + lines["_cli_materials"], 0), "Python", "CPU; MCP in its own process", "JSON requests", f"One MCP plan: 102 ms vs 21.9 s for per-object calls {tag('r')}"],
        [code("rendering/") + " (extract, CPU raster, materials resolve, settings)", num(lines["rendering"], 0), "Python + NumPy", "CPU", "Per frame: a list of RenderInstance objects built from SceneObjects", f"extract {st['extract']:.1f} ms + materials {st['textures']:.1f} ms per frame {tag('m')}"],
        [code("rendering/gpu/") + " + WGSL", f"{lines['rendering.gpu']:,} + {sum(imp['wgsl'].values()):,} WGSL", "Python host, WGSL shaders", "GPU through wgpu-py", "New storage buffers every frame; cached meshes and textures; 28 B/pixel readback", f"upload {st['upload']:.1f} ms, GPU {st['gpu']:.1f} ms, readback {st['readback']:.1f} ms {tag('m')}"],
        [code("rendering/raytrace/"), num(lines["rendering.raytrace"], 0), "NumPy", "CPU, 1 thread", "Structure of arrays: binary BVH (child, float32 child boxes, first/count), float32 triangles in leaf order", f"{min(st_np):.2f}–{max(st_np):.2f} M rays/s; 1M-triangle build {emb['sphere 1M tris']['aether3d_blas_build_seconds (NumPy, 1 thread)']:.1f} s {tag('m')}"],
        [code("rendering/pathtrace/") + " (CPU integrator)", num(lines["rendering.pathtrace"], 0), "NumPy", "CPU, 1–8 processes", "Wavefront arrays", f"0.7–3.2 M rays/s with 8 processes {tag('r')}"],
        [code("rendering/gpu/wgsl/pathtrace.wgsl"), num(imp["wgsl"]["pathtrace.wgsl"], 0), "WGSL", "GPU", "One packed u32 buffer (header + node, box, triangle, instance regions)", f"33–67 M rays/s full paths; 228 M camera + AO {tag('r')}"],
        [code("materials/"), num(lines["materials"], 0), "Python + NumPy (+ WGSL material.wgsl)", "CPU and GPU", "Node graphs; own PNG decoder", f"PNG decode {min(png_a):.1f}–{max(png_a):.1f} MP/s {tag('m')}"],
        [code("editing/"), num(lines["editing"], 0), "Python (positions in NumPy)", "CPU", "Hybrid: positions in arrays, topology in dicts, lists and sets per element", f"from_mesh {e5['EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s']:.1f} s for 200k triangles with attributes {tag('m')}"],
        [code("procedural/"), num(lines["procedural"], 0), "NumPy", "CPU", "Vectorized modifiers; SHA-256 content cache", f"Subdivision level 1: {HARD['subdiv_51k_ms']} ms for 51k faces {tag('m')}"],
        [code("spatial/"), num(lines["spatial"], 0), "Python + NumPy (uses raytrace)", "CPU", "Entry list + per-revision caches", f"warm raycast {sub['spatial_warm_raycast_ms']:.1f} ms {tag('m')}"],
        [code("gltf/ · serialization.py"), num(lines["gltf"] + lines["serialization"], 0), "Python + NumPy", "CPU", "JSON + binary buffers", f"save {sub['save_s']:.2f} s, load {sub['load_s']:.2f} s ({sub['file_mb']:.1f} MB); GLB export {sub['export_glb_s'] * 1e3:.0f} ms {tag('m')}"],
    ],
    cls="xwide",
    numeric_cols=(1,),
)
dep = """
<pre class="mermaid">
flowchart TB
  subgraph entry["Entry points"]
    cli["cli + _cli_materials · 3,250 lines"]
    mcp["mcp · Stage 11 · 2,304 lines"]
  end
  plans["plans · Stage 11 · 5,966 lines"]
  subgraph core["Core data model"]
    scene["scene.py facade · 2,300 lines · 17 eager imports, 50 lazy"]
    hist["history + history file · 2,200 lines"]
    base["transform · mesh · math3d · primitives · errors"]
  end
  subgraph feat["Feature services, imported lazily by scene.py"]
    spatial["spatial · 3,777"]
    rendering["rendering: extract, CPU raster, · material resolve · 6,379"]
    gpu["rendering.gpu: wgpu-py host · 3,790 + 3,150 WGSL"]
    pt["rendering.pathtrace: CPU · 2,175"]
    rt["rendering.raytrace: NumPy BVH · 998"]
    materials["materials · 5,593"]
    editing["editing · 5,352"]
    procedural["procedural · 4,661"]
    io["gltf 2,632 · serialization 898"]
  end
  cli --> plans
  cli --> scene
  mcp --> plans
  mcp --> scene
  plans --> scene
  plans --> procedural
  plans --> spatial
  scene --> hist
  scene --> base
  scene -.->|lazy x20| spatial
  scene -.->|lazy| rendering
  scene -.->|lazy| io
  scene -.->|lazy| procedural
  scene -.->|lazy| editing
  rendering --> materials
  rendering --> gpu
  gpu --> pt
  gpu --> rt
  pt --> rt
  spatial --> rt
  procedural --> editing
</pre>
"""
mapsec = f"""
<p>Aether3D is about {sum(lines.values()):,} lines of Python and {sum(imp['wgsl'].values()):,} lines of WGSL, with no native code of its own. NumPy, wgpu-native and optional libraries do the native work. The layering is sound. <code>scene.py</code> eagerly imports only the core (errors, history, math3d, mesh, camera, light, settings, transform), and reaches the 50 feature imports lazily, inside methods. That keeps cycles out of import time, and it means a new store can sit under the facade without touching the feature packages' imports.</p>
{dep}
<p class="note">Edges summarize a static import scan (<span class="kbd">lab/m/imports.py</span>). Line counts include docstrings and comments.</p>
<h3>Component map</h3>
{mp}
"""
section("map", "3. System map", mapsec)

# ---------------------------------------------------------------- 4. Bottlenecks
frame_chart = stacked(
    [
        ("Today, unchanged scene", [st["evaluate"], st["textures"], st["extract"], st["upload"], st["prepare"], st["gpu"], st["readback"]]),
        ("Retained, 1 object moved", [est_one["evaluate"], est_one["materials (cached)"], est_one["extract (1 dirty object)"], est_one["upload (1 record)"], est_one["prepare"], est_one["gpu"], est_one["readback (all outputs)"]]),
        ("Retained, all 10k moved", [est_all["evaluate"], est_all["materials (cached)"], est_all["world matrices (NumPy, flat)"], est_all["pack all (NumPy, flat)"] + est_all["write 1.28 MB"], est_all["prepare"], est_all["gpu"], est_all["readback (all outputs)"]]),
        ("Retained, 1 moved, image-only readback", [est_one["evaluate"], est_one["materials (cached)"], est_one["extract (1 dirty object)"], est_one["upload (1 record)"], est_one["prepare"], est_one["gpu"], img_only_readback]),
    ],
    [("evaluate", "s1"), ("materials / textures", "s2"), ("extract / world", "s3"), ("upload / pack", "s4"), ("prepare", "s5"), ("GPU passes", "s6"), ("readback", "s7")],
    vmax=110,
    unit="milliseconds",
    title="GPU frame at 10,000 objects, 1280×720 (top row measured, others estimated)",
)
bott = f"""
<h3 id="frame">4.1 The GPU frame is an O(N) rebuild</h3>
<p>The 10,000-cube scene (one plan, one light, one camera) renders at 1280×720 on the RTX 4060. Stage times come from the renderer's own <code>stats["timings"]</code> {tag('m')}:</p>
""" + table(
    ["Stage", "ms", "What runs", "Where"],
    [
        ["evaluate", num(st["evaluate"]), "Procedural cache check", code("rendering/scene_data.py")],
        ["textures", num(st["textures"]), "Material and texture resolution for every instance", code("rendering/materials.py:174 resolve_materials")],
        ["extract", num(st["extract"]), "A RenderInstance per object, world matrices via per-level gathers from Python objects, a finiteness check and an f-string location per object", code("rendering/scene_data.py:294, :190")],
        ["upload", num(st["upload"]), f"pack_instances {e2c['pack_instances_ms']:.1f} ms, draw_order {e2c['draw_order_ms']:.1f} ms (Python sort with key lambdas), new storage buffers", code("rendering/gpu/resources.py:621, :561; renderer.py:233")],
        ["prepare", num(st["prepare"]), "Command encoding", code("rendering/gpu/renderer.py")],
        ["gpu", num(st["gpu"]), "Shadow, scene and resolve passes", "WGSL"],
        ["readback", num(st["readback"]), "All outputs copied to a staging buffer and mapped: image, linear color, depth, object index, coverage (28 B/pixel)", code("rendering/gpu/renderer.py:434")],
        ["<b>total</b>", f"<b>{st['total']:.1f}</b>", f"Wall clock {frame_same:.1f} ms unchanged; {frame_moved:.1f} ms with one object moved", ""],
    ],
    numeric_cols=(1,),
) + f"""
<div class="figure">{frame_chart}<div class="cap">Bars to scale. The top bar is measured: the renderer's stage times add up to {sum(st[k] for k in ("evaluate", "textures", "extract", "upload", "prepare", "gpu", "readback")):.0f} ms, a little under its reported total ({st['total']:.0f} ms) and the wall-clock frame ({frame_same:.0f} ms). The other three bars are composed from measured parts in E1 and E2 with the assumptions in section 8; the materials stage is assumed cached at 0.5 ms.</div></div>
<p>A profile of <code>extract_render_scene</code> shows per-object Python work, not one hot function. The biggest self-time items are the loop body itself, 60k appearance getter calls, 10k <code>np.all(np.isfinite)</code> calls and 10k <code>repr</code> calls for location strings {tag('m')}. Making those statements faster would help, but the real fix is to stop doing per-object work for objects that did not change. The GPU's {st['gpu']:.1f} ms shows that the passes themselves are not the problem.</p>

<h3 id="objects">4.2 The object model</h3>
""" + table(
    ["Measure (10k objects)", "Value", "Note"],
    [
        ["Memory per object, history included", f"{sub['bytes_per_object_incl_history']} B {tag('m')}", "SceneObject + Transform + Appearance + dict entries"],
        ["Read position", f"{sub['read_position_ns']} ns {tag('m')}", "Python attribute access"],
        ["Read world_matrix", f"{sub['read_world_matrix_us']:.1f} µs {tag('m')}", "Recomputed through the parent chain on every read"],
        ["Set position (alone / in a transaction)", f"{sub['set_position_us']:.1f} / {sub['set_position_in_tx_us']:.1f} µs {tag('m')}", "Validation + history record"],
        ["Create via plan", f"{HARD['plan_10k_ms'] / 10:.1f} µs/object {tag('m')}", f"10k in {HARD['plan_10k_ms']:.0f} ms"],
        ["Floor for any Python object design", f"{HARD['floor_us']:.2f} µs/object {tag('m')}", f"Slotted object + transform + id + dict insert; <code>uuid.uuid4()</code> alone is {HARD['uuid_us']:.2f} µs (one <code>getrandom</code> syscall per id, <code>scene.py:77</code>)"],
        ["Undo / redo of the 10k plan", f"{HARD['undo_redo_10k_ms'][0]} / {HARD['undo_redo_10k_ms'][1]} ms {tag('m')}", "Already batched; fine"],
    ],
) + f"""
<p>The plan path is already lean: one validation pass, then one recorded operation (<code>scene.py:1065 _create_many</code>). A profile spreads the remaining ~{HARD['plan_10k_ms'] / 10 - HARD['floor_us']:.0f} µs/object above the floor over plan expansion, per-record validation (<code>_finite3</code>, unit-quaternion checks), history touches, appearance cloning and grid binding {tag('m')}. Each of these is a Python-level fix: batch ids, vectorized validation for expanded grids, a shared default appearance with copy-on-write. None needs a new language.</p>

<h3 id="spatial">4.3 Spatial queries</h3>
<p>A warm raycast at 10k objects takes {sub['spatial_warm_raycast_ms']:.1f} ms {tag('m')}. Every query, <code>SpatialIndex.accel</code> (<code>spatial/index.py:323</code>) rebuilds the list of usable entries and an N-element tuple key in Python. That is 200,020 <code>singular</code> calls for 20 queries, while the traversal itself is about {HARD['raycast_traversal_ms']} ms {tag('m')}. Caching the filtered list and key per index revision removes most of the cost, a small Python change. The first overlaps query after a change builds the index in about {HARD['overlaps_first_ms']} ms {tag('m')}.</p>

<h3 id="rt">4.4 CPU ray tracing</h3>
<p><code>SceneAccel._trace</code> (<code>rendering/raytrace/accel.py:452</code>) is a vectorized wavefront state machine. Each NumPy iteration advances every active ray by one node. That is a clever way to use NumPy for an inherently scalar algorithm, but it pays for array temporaries and gathers at every step. It reaches {min(st_np):.2f}–{max(st_np):.2f} M rays/s on one thread with 9–52 node visits per ray {tag('m')}. The binned SAH build (<code>rendering/raytrace/bvh.py:172</code>) takes {emb['sphere 1M tris']['aether3d_blas_build_seconds (NumPy, 1 thread)']:.1f} s for 1M triangles {tag('m')}. The repository's own benchmarks show 8 processes give only 2.6–4.2× because the work is memory-bandwidth bound, and traversal is 58–67% of path-tracer time {tag('r')}.</p>

<h3 id="edit">4.5 Editing</h3>
<p>Opening a 200k-triangle sphere with normals and UVs as an <code>EditableMesh</code> takes {e5['EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s']:.1f} s. With topology only it takes {e5['EditableMesh.from_mesh(quads=False) s']:.2f} s {tag('m')}. A profile on 25k triangles attributes {HARD['from_mesh_profile']['_join_pairs'] / HARD['from_mesh_profile']['total'] * 100:.0f}% to quad reconstruction (<code>editing/topology.py:573 _join_pairs</code>). It computes 117k face normals one at a time, with NumPy on 3-vectors (<code>newell_normal</code> and <code>unit</code>, about 234k calls), each paying about a microsecond of call overhead. Face creation (<code>editing/core.py:996 _new_face</code>) takes {HARD['from_mesh_profile']['_new_face'] / HARD['from_mesh_profile']['total'] * 100:.0f}% and shading inference {HARD['from_mesh_profile']['_infer_shading'] / HARD['from_mesh_profile']['total'] * 100:.0f}% {tag('m')}. Bevel on 51k faces took {HARD['bevel_51k_s']} s {tag('m')}. These are per-element Python loops over dict-based topology, the pattern Blender keeps in C (BMesh).</p>

<h3 id="img">4.6 Images</h3>
<p>Decoding a 1080p render takes {min(v['Aether3D decode_png ms'] for v in e6.values()):.0f}–{max(v['Aether3D decode_png ms'] for v in e6.values()):.0f} ms in Aether3D and {min(v['Pillow decode ms'] for v in e6.values()):.0f}–{max(v['Pillow decode ms'] for v in e6.values()):.0f} ms in Pillow {tag('m')}. Both inflate with zlib, which takes {min(v['zlib inflate ms'] for v in e6.values()):.0f}–{max(v['zlib inflate ms'] for v in e6.values()):.0f} ms. The gap is the Paeth/Average unfilter, a strict left-to-right dependency that Aether3D vectorizes along anti-diagonals (<code>materials/images.py:360</code>). That takes about 270 ms, against {min(v['C unfilter ms'] for v in e6.values()):.0f}–{max(v['C unfilter ms'] for v in e6.values()):.0f} ms for a plain C loop {tag('m')}. Encoding a 1080p render takes {HARD['png_filter_ms']:.0f} ms of NumPy filter selection plus {HARD['zlib_ms'][6]:.0f} ms of zlib at level 6. zlib level 1 takes {HARD['zlib_ms'][1]:.0f} ms for a {(HARD['zlib_mb'][1] / HARD['zlib_mb'][6] - 1) * 100:.0f}% larger file {tag('m')}.</p>

<h3 id="fine">4.7 What is already fine</h3>
<ul>
<li>Save and load: {sub['save_s']:.2f} s and {sub['load_s']:.2f} s at 10k objects ({sub['file_mb']:.1f} MB). GLB export: {sub['export_glb_s'] * 1e3:.0f} ms {tag('m')}.</li>
<li>Undo and redo of a 10k-object plan: {HARD['undo_redo_10k_ms'][0]} and {HARD['undo_redo_10k_ms'][1]} ms {tag('m')}.</li>
<li>Procedural modifiers are vectorized NumPy with a content-addressed cache {tag('m')}.</li>
<li>The GPU passes ({st['gpu']:.1f} ms) and the WGSL path tracer, which is 51–65× faster than 8 CPU processes {tag('r')}.</li>
<li>Python start-up: importing Aether3D takes 135 ms {tag('m')}.</li>
</ul>
"""
section("bottlenecks", "4. Where the time goes", bott)

# ---------------------------------------------------------------- 5. Experiments
# E1
e1_rows = []
order10 = sorted(e1_10k.items(), key=lambda kv: kv[1]["median_ms"])
e1_rows.append([f"<b>Aether3D today</b> {code('world_matrices(scene)')}", f"{cur_w10:.2f}", "1×", f"{cur_w100:.1f}", "1×", "reference"])
for k, v in order10:
    v100 = e1["100000"].get(k)
    e1_rows.append([
        esc(k),
        f"{v['median_ms']:.3f}" if v["median_ms"] < 10 else f"{v['median_ms']:.1f}",
        f"{cur_w10 / v['median_ms']:.0f}×",
        (f"{v100['median_ms']:.2f}" if v100 else "not run"),
        (f"{cur_w100 / v100['median_ms']:.0f}×" if v100 else ""),
        "yes" if v["bitwise_equal"] else f"no (max diff {v['max_abs_diff']:.1e})",
    ])
e1html = f"""
<h3 id="e1">E1 · World matrices of a hierarchy</h3>
<p><b>Question.</b> How much of the {cur_w10:.1f} ms comes from the language, and how much from gathering data out of Python objects? <b>Setup.</b> The test uses a random forest with 10k and 100k nodes, depth ≤ 8 and parents before children. Positions, quaternions and scales come from the same arrays a columnar store would hold, and every port uses the same 4×4 arithmetic as <code>rendering/scene_data.py</code>. Aether3D's own function ran on a Scene built from identical records and matched the flat-array reference exactly.</p>
""" + table(["Implementation", "10k: ms", "× today", "100k: ms", "× today", "Bitwise equal"], e1_rows, numeric_cols=(1, 2, 3, 4)) + f"""
<p><b>Reading.</b> Moving the same NumPy algorithm onto flat arrays gives {cur_w10 / np_w10:.0f}× at 10k and {cur_w100 / e1['100000']['numpy (Aether3D algorithm, flat arrays)']['median_ms']:.0f}× at 100k with no new language. Compiled code adds another {np_w10 / min(native_w10):.0f}×. Cython, C, Zig, Rust and Numba are within 1.6× of each other, and Go and C# within 2.5×. Allowing FMA contraction, which GCC does by default for GNU C when the target has FMA, changes results in the 13th digit. Every native kernel must be built with contraction off to keep CPU results reproducible. A ctypes call with no arguments cost {min(e1['ctypes_noop_call_ns'].values())}–{max(e1['ctypes_noop_call_ns'].values())} ns across the libraries {tag('m')}.</p>
"""
# E2
e2html = f"""
<h3 id="e2">E2 · Per-frame instance data: pack, upload, rebase</h3>
<p><b>Question.</b> What does it cost to keep the GPU instance buffer current with a full rebuild, with flat arrays, and with dirty-only updates? The record is <code>struct Instance</code> (32 floats, <code>rendering/gpu/resources.py:43</code>). Every flat-array packer's output was compared byte for byte with <code>pack_instances</code> and matched.</p>
<div class="two">
<div class="card"><h4>CPU side, 10k instances {tag('m')}</h4>""" + table(
    ["Step", "Time"],
    [
        ["world_matrices (today)", f"{e2c['world_matrices_ms']:.1f} ms"],
        ["draw_order (today)", f"{e2c['draw_order_ms']:.1f} ms"],
        ["pack_instances (today)", f"{e2c['pack_instances_ms']:.1f} ms"],
        ["pack all, NumPy on flat arrays", f"{e2c['full_pack_numpy_ms']:.3f} ms"],
        ["pack all, Numba", f"{e2c['full_pack_numba_ms']:.3f} ms"],
        ["pack all, C", f"{e2c['full_pack_c_ms']:.3f} ms"],
        ["rewrite 1 / 10 / 100 / 1000 dirty records, NumPy", f"{e2c['dirty_1_numpy_us']:.1f} / {e2c['dirty_10_numpy_us']:.1f} / {e2c['dirty_100_numpy_us']:.1f} / {e2c['dirty_1000_numpy_us']:.0f} µs"],
        ["same, C", f"{e2c['dirty_1_c_us']:.1f} / {e2c['dirty_10_c_us']:.1f} / {e2c['dirty_100_c_us']:.1f} / {e2c['dirty_1000_c_us']:.0f} µs"],
    ],
    numeric_cols=(1,),
) + f"""</div>
<div class="card"><h4>GPU side, RTX 4060, 1.28 MB buffer {tag('m')}</h4>""" + table(
    ["Operation", "Time"],
    [
        ["create_buffer_with_data (today, per frame)", f"{e2g['create_buffer_with_data_1.28MB_us']:.0f} µs"],
        ["write_buffer, whole buffer", f"{e2g['write_buffer_full_1.28MB_us']:.0f} µs"],
        ["write_buffer, 1 record", f"{e2g['write_buffer_1_records_one_range_us']:.1f} µs"],
        ["10 records: 10 writes vs 1 range", f"{e2g['write_buffer_10_records_separate_us']:.0f} vs {e2g['write_buffer_10_records_one_range_us']:.1f} µs"],
        ["100 records: 100 writes vs 1 range", f"{e2g['write_buffer_100_records_separate_us']:.0f} vs {e2g['write_buffer_100_records_one_range_us']:.1f} µs"],
        ["Empty submit + wait", f"{e2g['submit_and_wait_roundtrip_us']:.0f} µs"],
        ["Camera rebase on GPU, 10k instances (kernel)", f"{e2g['gpu_rebase_10k_kernel_us (timestamps)']:.2f} µs"],
    ],
    numeric_cols=(1,),
) + f"""</div></div>
<p><b>Precision of moving the camera rebase to the GPU.</b> Today each frame subtracts the eye from every model matrix in float64 on the CPU (<code>pack_instances</code>), so any camera move forces a full repack. Storing world translations as float32 high/low pairs and subtracting on the GPU was tested with objects 100 km from the origin and the camera among them. Max error was {e2g['gpu hi/lo rebase: max abs error vs float64 (m)']:.1e} m, the same as today's CPU path ({e2g['current CPU path (f64 subtract, then round to f32): max abs error (m)']:.1e} m). Naive float32 had {e2g['naive float32 world - eye: max abs error (m)'] * 1e3:.1f} mm of error {tag('m')}.</p>
<p><b>Reading.</b> With flat arrays, NumPy is within 10× of C for full packs and within 1–5× for dirty updates. The GPU API rewards coalescing: 100 single-record writes cost 50× more than one range write. Persistent buffers, coalesced dirty ranges and GPU-side rebase remove nearly all upload cost for frames where little changed.</p>
"""
# E3
e3_cols = ["NumPy (current)", "C (gcc -O3)", "C (clang -O3)", "C++ (g++)", "Rust 1.99", "Zig 0.17", "Numba 0.68", "C# .NET 10 AOT", "Go 1.27", "Embree 4.4.1"]
e3_rows = []
for w in WORKLOADS:
    sname, bname = w.split(" | ")
    r = e3[w]
    vals = []
    for c in e3_cols:
        if c == "C++ (g++)":
            v = HARD["cpp_st"][w]
        elif c == "Embree 4.4.1":
            v = emb[sname][bname]["x1"]
        else:
            v = r[c]["1"]
        ratio = v / r["NumPy (current)"]["1"]
        vals.append(f"{sig(v)}<span class='sub'>{'' if c == 'NumPy (current)' else f'{ratio:.0f}×'}</span>")
    e3_rows.append([f"{esc(sname)}<span class='sub'>{esc(bname)} · {r['rays']:,} rays · {r['node_visits_per_ray']} visits/ray</span>"] + vals)
mt_cols = ["C (gcc)", "Rust", "Zig", "Numba", "C# AOT", "Go"]
mt_rows = []
for w in WORKLOADS:
    sname, bname = w.split(" | ")
    r = e3mt[w]
    base = e3[w]["NumPy (current)"]["1"]
    row = [f"{esc(sname)}<span class='sub'>{esc(bname)}</span>"]
    for c in mt_cols:
        row.append(f"{sig(r[c])}<span class='sub'>{r[c] / base:.0f}×</span>")
    ev = emb[sname][bname]["x20"]
    row.append(f"{sig(ev)}<span class='sub'>{ev / base:.0f}×</span>")
    row.append("yes" if r["parity"] else "no")
    mt_rows.append(row)
agree_rows = []
for w in WORKLOADS:
    sname, bname = w.split(" | ")
    a = emb[sname][bname]["agreement"]
    agree_rows.append([f"{esc(sname)} · {esc(bname)}", f"{a['hit/miss same'] * 100:.4f}%", f"{a['same triangle (of rays both hit)'] * 100:.4f}%", f"{a['t rel. diff median']:.1e}"])
w_chart = WORKLOADS[0]
chart_items = []
for c, cls in (("NumPy (current)", "b-cur"), ("Go 1.27", "b-cpu2"), ("C# .NET 10 AOT", "b-cpu2"), ("Rust 1.99", "b-cpu"), ("C (clang -O3)", "b-cpu"), ("Zig 0.17", "b-cpu"), ("Numba 0.68", "b-cpu"), ("C (gcc -O3)", "b-cpu")):
    chart_items.append((c + " · 1 thread", e3[w_chart][c]["1"], cls))
chart_items.append(("Embree 4.4.1 · 1 thread", emb["sphere 100k tris"]["camera 640x480"]["x1"], "b-mut"))
for c, cls in (("Rust", "b-gpu2"), ("C (gcc)", "b-gpu2")):
    chart_items.append((c + " · 20 threads", e3mt[w_chart][c], cls))
chart_items.append(("Embree 4.4.1 · 20 threads", emb["sphere 100k tris"]["camera 640x480"]["x20"], "b-gpu"))
e3chart = hbar_log(chart_items, 0.1, 300, "M rays/s", title="Closest hit, sphere 100k triangles, 307,200 camera rays")
e3html = f"""
<h3 id="e3">E3 · BVH closest-hit traversal on the CPU</h3>
<p><b>Question.</b> What does each language gain on Aether3D's most important scalar kernel, with the same algorithm and data? <b>Setup.</b> Every port runs the per-ray state machine of <code>SceneAccel._trace</code>: the same two-level table, ulp-grown float32 boxes, float64 arithmetic, the Woop–Benthin–Wald watertight test, the same visit order and tie rules, and the same skip-triangle handling for bounce rays. The scenes come from <code>benchmarks/bench_raytrace.py</code>, and the arrays come straight from <code>SceneAccel.build</code>. The C++ row is the C file compiled by g++.</p>
<div class="figure">{e3chart}<div class="cap">Log scale. Copper is today's NumPy tracer, blue is compiled ports on one thread, and teal is 20 threads. Embree uses float32 and SIMD BVH8 nodes, so it is not bit-identical.</div></div>
""" + table(["Workload"] + e3_cols, e3_rows, cls="xwide", caption="One thread, M rays/s (× today's NumPy). Every compiled port produced the same t, instance, triangle and barycentrics as Aether3D, bit for bit.", numeric_cols=tuple(range(1, 11))) + table(
    ["Workload"] + mt_cols + ["Embree", "Bitwise"],
    mt_rows,
    cls="wide",
    caption="20 threads, M rays/s (× one-thread NumPy). Ports interleaved in random order, median of 7 rounds; Embree measured separately.",
    numeric_cols=tuple(range(1, 8)),
) + table(["Embree vs Aether3D float64", "Same hit/miss", "Same triangle", "Median relative t difference"], agree_rows, caption="Embree 4.4.1 (robust mode, same float32 vertices, rays rounded to float32, skip-triangle via a filter callback)", numeric_cols=(1, 2, 3)) + f"""
<p><b>Reading.</b></p>
<ul>
<li>Every compiled port beats the NumPy wavefront by about an order of magnitude on one thread: GCC C/C++ {rng(st_ratio, lambda x: f'{x:.0f}')}×, clang, Rust and Zig {rng(llvm_ratio, lambda x: f'{x:.0f}')}×. Go and C# trail at {rng(slow_ratio, lambda x: f'{x:.0f}')}× {tag('m')}.</li>
<li>clang, Rust and Zig are within {llvm_spread * 100:.0f}% of each other, and Numba is close (all LLVM). GCC is {rng([1 / x for x in llvm_vs_gcc], lambda x: f'{x:.1f}')}× faster than them, so on this kernel the compiler backend matters more than the language. Layout work of the kind Embree does (below) is worth more than that backend gap.</li>
<li>Threads scale {rng(scale20, lambda x: f'{x:.0f}')}× from 1 to 20 (6 hyper-threaded P-cores and 8 E-cores sharing one memory system). The repository's NumPy path tracer gets 2.6–4.2× from 8 processes, which its docs attribute to memory bandwidth {tag('r')}. Native threads share one BVH and avoid NumPy's large temporaries, so they scale further.</li>
<li>Embree is {rng(emb_over_c, lambda x: f'{x:.1f}')}× faster than the best scalar port on one thread. That gain comes from data layout and SIMD (BVH8 nodes, float32, packet-friendly code), which any native language can adopt and NumPy cannot. Embree agrees on 99.99% of triangles but is not bit-identical. Replacing the float64 reference with it would change goldens, so it belongs behind an opt-in.</li>
<li>Embree builds the 1M-triangle BVH in {emb['sphere 1M tris']['build_seconds (all threads, TBB)'] * 1e3:.0f} ms (TBB, all threads). Aether3D's NumPy build takes {emb['sphere 1M tris']['aether3d_blas_build_seconds (NumPy, 1 thread)']:.2f} s {tag('m')}.</li>
</ul>
"""
# E4
ffi_rows = []
for k, v in ffi.items():
    if k == "empty loop ns":
        continue
    ffi_rows.append([esc(k), f"{v['noop ns']:.1f}", f"{v['add3(float, float, float) ns']:.1f}"])
e4html = f"""
<h3 id="e4">E4 · The Python-to-native boundary</h3>
<p><b>Question.</b> Does calling native code from Python cost too much for a fine-grained API? Each module exposes <code>noop()</code> and <code>add3(a, b, c)</code>. Timings are the median of 7 × 1M calls from CPython 3.14; the empty loop costs {ffi['empty loop ns']} ns.</p>
""" + table(["Binding", "noop() ns", "add3(1.0, 2.0, 3.0) ns"], ffi_rows, numeric_cols=(1, 2)) + f"""
<p><b>Reading.</b> Compiled bindings cost the same as a Python call. ctypes and cffi's ABI mode are 5–15× slower because they convert arguments at run time. They are fine for one call per batch, but wrong for per-object calls. Numba's dispatcher costs {ffi['Numba 0.68 njit dispatch']['add3(float, float, float) ns']:.0f} ns per call. Its first call of the traversal kernel spent {HARD['numba_compile_s']} s compiling (another {HARD['numba_parallel_compile_s']} s for the parallel variant), on top of {HARD['numba_import_ms']} ms to import Numba, in every new process. That is acceptable in a lab and a poor fit for a library that AI agents start as short-lived processes {tag('m')}.</p>
<p><b>Binary size and runtimes.</b> The traversal library is {HARD['sizes_kb']['C']} KB in C/C++, {HARD['sizes_kb']['Zig']} KB in Zig and {HARD['sizes_kb']['Rust']} KB in Rust. In Go ({HARD['sizes_kb']['Go']:,} KB) and C# Native AOT ({HARD['sizes_kb']['C# Native AOT']:,} KB) it carries a garbage-collected runtime into the Python process. The first C# build also aborted on entry, from <code>stackalloc</code> in an <code>[UnmanagedCallersOnly]</code> export, and needed a workaround {tag('m')}.</p>
"""
# E5
e5html = f"""
<h3 id="e5">E5 · Mesh topology construction</h3>
<p><b>Question.</b> Is editing slow because of Python or because of per-element data structures? The test builds the edge table of a {e5['triangles']:,}-triangle sphere. Edge ids are numbered in first-use order, and each edge gets its list of faces in order, exactly what <code>EditableMesh._add_edge_use</code> produces. Every implementation was checked against Aether3D's own <code>_edge_verts</code> and <code>_edge_faces</code>.</p>
""" + table(
    ["Implementation", "Time", "× today", "Same edge ids and faces"],
    [
        ["EditableMesh.from_mesh, topology only (today)", f"{topo_cur_ms:.0f} ms", "1×", "reference"],
        ["NumPy: weld (shared, already in Aether3D) + vectorized edges", f"{e5['weld (NumPy, shared) ms']:.1f} + {e5['NumPy (vectorized) ms']:.1f} ms", f"{topo_cur_ms / topo_np:.0f}×", "yes" if e5["NumPy (vectorized) matches Aether3D edge ids/faces"] else "no"],
        ["C, simple open-addressing table", f"{e5['weld (NumPy, shared) ms']:.1f} + {e5['C (open addressing) ms']:.1f} ms", f"{topo_cur_ms / (e5['weld (NumPy, shared) ms'] + e5['C (open addressing) ms']):.0f}×", "yes" if e5["C (open addressing) matches Aether3D edge ids/faces"] else "no"],
        ["Rust, std HashMap (SipHash)", f"{e5['weld (NumPy, shared) ms']:.1f} + {e5['Rust (std HashMap, SipHash) ms']:.1f} ms", f"{topo_cur_ms / (e5['weld (NumPy, shared) ms'] + e5['Rust (std HashMap, SipHash) ms']):.0f}×", "yes" if e5["Rust (std HashMap, SipHash) matches Aether3D edge ids/faces"] else "no"],
        ["Rust, std HashMap (multiplicative hasher)", f"{e5['weld (NumPy, shared) ms']:.1f} + {e5['Rust (std HashMap, fast hasher) ms']:.1f} ms", f"{topo_cur_ms / topo_rs:.0f}×", "yes" if e5["Rust (std HashMap, fast hasher) matches Aether3D edge ids/faces"] else "no"],
        ["EditableMesh.from_mesh with normals, UVs and quads (today)", f"{e5['EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s']:.1f} s", "", "the default path"],
    ],
    numeric_cols=(1, 2),
) + f"""
<p><b>Reading.</b> The ports build the edge table and each edge's face list; <code>EditableMesh</code> also fills per-vertex edge sets and per-face dictionaries, which a data-oriented design would hold as arrays too, so these factors compare the core work rather than every byte of today's structure. Topology vectorizes well in NumPy ({topo_cur_ms / topo_np:.0f}× end to end, {topo_cur_ms / e5['NumPy (vectorized) ms']:.0f}× for the edge step alone). The welding step, already NumPy in Aether3D, then dominates. A good hash table counts for more than the language: my simple C table lost to Rust's standard SwissTable. The default path is {e5['EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s'] / e5['EditableMesh.from_mesh(quads=False) s']:.0f}× slower than topology alone because of per-face corner tuples, per-face normals and quad joining. Batched NumPy normals fix most of that; local operators such as bevel are the part that wants native code.</p>
"""
# E6
e6_rows = []
for name, v in e6.items():
    e6_rows.append([esc(name) + f"<span class='sub'>{v['size']}, filters N/S/U/A/P = {'/'.join(str(x) for x in v['filter types used (None,Sub,Up,Avg,Paeth)'])}</span>", f"{v['Aether3D decode_png ms']:.0f} ms<span class='sub'>{v['Aether3D decode MP/s']:.1f} MP/s</span>", f"{v['Pillow decode ms']:.0f} ms<span class='sub'>{v['Pillow decode MP/s']:.0f} MP/s</span>", f"{v['zlib inflate ms']:.0f} + {v['C unfilter ms']:.1f} ms<span class='sub'>{v['zlib + C unfilter MP/s']:.0f} MP/s</span>", f"{v['Aether3D encode_png ms']:.0f} ms<span class='sub'>{v['Aether3D encode bytes'] / 1e6:.2f} MB</span>", f"{v['Pillow encode ms']:.0f} ms<span class='sub'>{v['Pillow encode bytes'] / 1e6:.2f} MB</span>"])
e6html = f"""
<h3 id="e6">E6 · PNG decode and encode</h3>
<p>The test uses two showcase renders from <code>examples/output/</code>. Aether3D's decoder output equals Pillow's pixel for pixel (RGB equal, alpha 255), and the C unfilter equals Pillow's too.</p>
""" + table(["Image", "Aether3D decode", "Pillow decode", "zlib + C unfilter", "Aether3D encode (level 6)", "Pillow encode (level 6)"], e6_rows, cls="wide", numeric_cols=(1, 2, 3, 4, 5)) + f"""
<p><b>Reading.</b> Decoding is limited by one sequential kernel; about 50 lines of native code (or an optional Pillow fast path) fix it. Encoding is mostly zlib at level 6 in both libraries; a lower compression level for previews and a native filter step would cut it by more than half.</p>
"""
# E7
e7_rows = []
for s, label in (("sphere100k", "sphere 99k tris"), ("cubes400", "400 cube instances"), ("sphere1m", "sphere 998k tris")):
    for r in ("camera", "bounce"):
        sw, hw = gpu_rows[(s, r, "sw")], gpu_rows[(s, r, "hw")]
        a = agree[f"{s} {r}"]
        e7_rows.append([
            f"{label}<span class='sub'>{r} · {sw['n']:,} rays</span>",
            f"{sw['median_ms']:.2f} ms<span class='sub'>{sw['mrays_per_s']:,.0f} M rays/s</span>",
            f"{hw['median_ms']:.3f} ms<span class='sub'>{hw['mrays_per_s']:,.0f} M rays/s</span>",
            f"{hw['mrays_per_s'] / sw['mrays_per_s']:.1f}×",
            f"{a['sw']['same triangle as ref (both hit)'] * 100:.4f}% / {a['hw']['same triangle as ref (both hit)'] * 100:.4f}%",
            f"{gpu_build[s]['as_build_ms']:.1f} ms",
        ])
e7chart = grouped(
    [(f"{lab} · {r}", [gpu_rows[(s, r, 'sw')]['mrays_per_s'] / 1e3, gpu_rows[(s, r, 'hw')]['mrays_per_s'] / 1e3]) for s, lab in (("sphere100k", "99k tris"), ("cubes400", "400 cubes"), ("sphere1m", "998k tris")) for r in ("camera", "bounce")],
    [("Aether3D WGSL trace_ray (software)", "b-cpu"), ("wgpu ray query (RT cores)", "b-gpu")],
    vmax=5,
    unit="G rays/s",
    title="Closest-hit traversal on the RTX 4060, same rays, same process",
)
e7html = f"""
<h3 id="e7">E7 · GPU traversal: Aether3D's WGSL vs hardware ray queries</h3>
<p><b>Question.</b> What does hardware ray tracing buy over the traversal Aether3D runs today, and can Aether3D reach it? <b>Setup.</b> One Rust program (wgpu 30.0.1, Vulkan) runs two compute kernels on the same rays. The software kernel contains Aether3D's own <code>trace_ray</code> and helpers, copied verbatim from <code>rendering/gpu/wgsl/pathtrace.wgsl</code>. It reads a word buffer built exactly as <code>pack_scene_words</code> builds it (<code>rendering/gpu/pathtrace.py:169</code>), from the same SceneAccel. The hardware kernel uses <code>rayQueryInitialize</code> and <code>rayQueryProceed</code> on a BLAS/TLAS built by wgpu from the same float32 vertices. Rays are 1920×1080 camera rays plus diffuse bounce rays leaving the camera hits; GPU time comes from timestamp queries, median of 10.</p>
<div class="figure">{e7chart}<div class="cap">Linear scale, to scale.</div></div>
""" + table(["Workload", "WGSL software (today's code)", "Hardware ray query", "Speed-up", "Same triangle as float64 ref (SW / HW)", "BLAS + TLAS build"], e7_rows, cls="wide", numeric_cols=(1, 2, 3, 4, 5)) + f"""
<p><b>Reading.</b></p>
<ul>
<li>RT cores trace {rng(hw_ratio, lambda x: f'{x:.1f}')}× faster than the WGSL traversal. The gain is largest for incoherent rays in large meshes, which is where path tracing spends its rays. Both kernels agree with the float64 CPU reference on ≥{min_same_tri * 100:.3f}% of rays, and with each other on ≥99.999% {tag('m')}.</li>
<li>wgpu-py cannot use this. The wgpu-native 29 library it ships exports 228 C functions and none of them builds acceleration structures {tag('m')}. In Rust wgpu the API exists but is gated as experimental (<code>Features::EXPERIMENTAL_RAY_QUERY</code> plus an <code>ExperimentalFeatures</code> opt-in, and WGSL needs <code>enable wgpu_ray_query;</code>) {tag('m')} {tag('c')}.</li>
<li>Expect less end-to-end. The production path tracer reaches 228 M rays/s on camera and AO rays but 33–67 M rays/s on full paths, because material evaluation dominates {tag('r')}. Traversal alone in E7 runs at {rng(sw_rates, lambda x: f'{x:.0f}')} M rays/s in software. Hardware RT matters most after shading is restructured (wavefront or material sorting), which the repository's own notes already identify.</li>
</ul>
"""
section("experiments", "5. Experiments", e1html + e2html + e3html + e4html + e5html + e6html + e7html)

# ---------------------------------------------------------------- 6. Languages
lang = table(
    ["Option", "Kernel speed (E1/E3)", "Python boundary", "Packaging for 3 OSes × CPython 3.10–3.14", "Safety and determinism", "3D / GPU ecosystem", "Verdict"],
    [
        ["<b>Python + NumPy</b> (today)", f"Baseline. Vectorizable work is good; scalar work is 10–150× behind {tag('m')}", "none", "Pure wheels", "Memory-safe; deterministic (no FMA contraction)", "Rich (NumPy, trimesh, Pillow, wgpu-py)", chip("KEEP") + " for API, orchestration and vectorizable math"],
        ["<b>Rust</b> (PyO3 + maturin)", f"LLVM tier: {rng([e3[w]['Rust 1.99']['1'] / e3[w]['C (gcc -O3)']['1'] for w in WORKLOADS], lambda x: f'{x:.2f}')}× of GCC C on traversal; fastest on hash-heavy topology {tag('m')}", f"PyO3 13 / 27 ns {tag('m')}", "One toolchain; abi3 wheels (one per OS/arch for all supported CPythons); maturin builds them", "Memory-safe by default (decoders, MCP inputs); never contracts FMA unless asked; threads without data races", "wgpu (same WGSL, ray queries), naga, rayon, Embree via C API, meshoptimizer bindings", "<b>Primary native language</b>"],
        ["<b>C++</b> (nanobind + scikit-build-core)", f"GCC tier: fastest scalar port {tag('m')}; MSVC on Windows untested", f"nanobind 11 / 31 ns {tag('m')}", "CMake + 3 compilers (GCC, clang, MSVC); more CI work", "Manual memory safety; GCC's default for GNU C/C++ contracts FMA, so builds must pin -ffp-contract=off", "Deepest: Embree, OptiX, OpenVDB, OpenSubdiv, USD", "Alternative if Embree/USD/OptiX integration becomes central"],
        ["<b>C</b> (C API / cffi)", f"Same as C++ {tag('m')}", f"C API 12 / 28 ns; cffi 58–80 ns {tag('m')}", "Simple for tiny kernels", "Manual; same FMA caveat", "Embree's native API", "No advantage over Rust or C++"],
        ["<b>Cython</b>", f"Top tier on the simple E1 loop {tag('m')}; not ported for E3", f"11 / 26 ns {tag('m')}", "C compiler matrix", "Manual (C semantics)", "Good glue next to Python objects", "Fine for small loops; weaker than Rust for big kernels"],
        ["<b>Numba</b>", f"LLVM tier, {rng(numba_vs_c, lambda x: f'{x:.2f}')}× of GCC on traversal {tag('m')}", f"Dispatch 25 / 116 ns; {HARD['numba_compile_s']} s first-call compile, {HARD['numba_import_ms']} ms import {tag('m')}", "Runtime JIT; llvmlite tracks CPython releases with a lag (3.14 support arrived in 0.63) " + tag("c"), "Deterministic with fastmath off", "Numerical; no GPU via wgpu", "Lab tool, not a dependency"],
        ["<b>Zig</b>", f"LLVM tier {tag('m')}", "C ABI via a hand-written CPython module or ctypes", "Excellent cross-compilation", "Safer than C in Debug/ReleaseSafe; not 1.0, breaking releases " + tag("c"), "Small", "Not now"],
        ["<b>C# Native AOT</b>", f"{rng(cs_vs_c, lambda x: f'{x:.2f}')}× of C {tag('m')}", "C ABI exports; GC runtime in-process", f"1.4 MB per library; we hit an AOT abort {tag('m')}", "Managed", "Small for 3D kernels", "No"],
        ["<b>Go</b> (cgo c-shared)", f"{rng(go_vs_c, lambda x: f'{x:.2f}')}× of C {tag('m')}", "cgo export, 40–60 ns per call on top " + tag("c"), "2 MB runtime with GC and signal handlers inside Python", "Spec allows FMA fusion on some targets", "Small", "No"],
        ["<b>Swift, Julia, Java/Kotlin, D, Nim, Odin, Mojo</b>", "Not measured. Each compiles through LLVM (Swift, Julia, D/LDC, Odin, Mojo) or C (Nim), so E1/E3 suggest kernel speed would track that backend " + tag("e"), "No mainstream way to ship a small CPython extension wheel; Julia and the JVM bring a whole runtime; Julia's trimmed static compiler is experimental " + tag("c"), "Varies; most are heavy or immature for this use", "Varies", "Thin for this use", "No: deployment, not speed, rules them out"],
    ],
    cls="xwide",
)
gputech = table(
    ["Technology", "What it gives Aether3D", "Python integration", "Cost and risk", "Verdict"],
    [
        ["<b>WGSL on wgpu-py</b> (today)", "Vulkan, Metal and D3D12 from one shader language; headless; validated by naga", "Native Python package", "No ray-tracing API in wgpu-native " + tag("m"), chip("KEEP")],
        ["<b>Rust wgpu host</b>", f"The same WGSL and backends, plus ray queries: {rng(hw_ratio, lambda x: f'{x:.1f}')}× faster traversal {tag('m')}", "Through the same PyO3 crate as the kernels", "Experimental API; would take over device ownership from wgpu-py (two wgpu instances cannot share buffers)", chip("DEFER")],
        ["<b>Vulkan directly</b> (ash or C++)", "Full RT pipelines, every extension", "Native module", "Large effort; loses Metal/D3D12 unless rewritten per API", "No"],
        ["<b>CUDA + OptiX</b>", "Best-in-class NVIDIA path tracing (as in Cycles)", "Native module", "NVIDIA only; proprietary SDK terms " + tag("c"), "No for core; possible optional plugin"],
        ["<b>DX12 / DXR</b>", "Windows hardware RT", "Native module", "Windows only; wgpu already covers D3D12", "No"],
        ["<b>Metal / Metal RT</b>", "Apple hardware RT", "Native module", "Apple only; better reached through wgpu if and when its ray queries cover Metal (untested here)", "Through wgpu only"],
        ["<b>HIP / HIP RT</b>, <b>SYCL</b>", "AMD and Intel compute", "Native module", "Vendor-specific", "No"],
        ["<b>Slang</b> (shader language)", "Modules, generics, autodiff; many targets", "Offline compiler", "WGSL and Metal targets are experimental " + tag("c"), chip("DEFER")],
        ["<b>NVIDIA Warp</b> (Python GPU kernels, Apache-2.0)", "CUDA kernels written in Python", "Python package", "CUDA GPUs only; second GPU stack beside wgpu " + tag("c"), "No"],
    ],
    cls="wide",
)
langsec = f"""
<h3>6.1 Language families</h3>
<p>Scores come from the measurements where they exist; otherwise the row says what was not measured. Performance alone does not separate C, C++, Rust and Zig: they share two compiler backends, and the gap between backends (up to {max(1 / x for x in llvm_vs_gcc):.1f}× on one kernel) is smaller than the gap a better data layout opens (Embree, up to {max(emb_over_c):.1f}×). The deciding factors are memory safety for untrusted inputs, one predictable toolchain for wheels on three operating systems, and the GPU path. Rust leads on all three: wgpu is a Rust library, so a future hardware-RT host would share the kernel crate's language.</p>
{lang}
<h3>6.2 GPU technologies</h3>
{gputech}
<h3>6.3 Why Rust over C++, and when that would flip</h3>
<ul>
<li><b>For Rust.</b> The native code would parse untrusted data (PNG files, glTF, plans from MCP clients) and run multithreaded kernels; Rust's defaults prevent the memory and data-race bugs C++ leaves to review. maturin with abi3 gives one wheel per OS and architecture for every supported CPython. Floating-point contraction is never implicit. wgpu, the stack behind wgpu-py, is native Rust, so a future GPU host shares the kernel crate's language.</li>
<li><b>For C++.</b> GCC produced the fastest scalar traversal here, and the deepest 3D libraries (Embree, OptiX, OpenVDB, OpenSubdiv, USD) are C++. If Aether3D decides to build on several of them, nanobind + scikit-build-core is the better home.</li>
<li><b>Either way, pick one.</b> Two native languages would double CI, packaging and review cost for kernels that total a few thousand lines.</li>
</ul>
"""
section("languages", "6. Languages and GPU technologies", langsec)

# ---------------------------------------------------------------- 7. Other engines
eng = table(
    ["Engine", "Relevant design", "Lesson for Aether3D"],
    [
        ["<b>Blender</b>", f"C/C++ data (DNA) behind a Python API generated from RNA descriptions. The dependency graph evaluates changed data into copies (copy-on-evaluation) [1]. Meshes moved from arrays of structs to per-attribute arrays [2]. Bulk <code>foreach_get/foreach_set</code> took an image-pixel access from 2,900 ms to 29 ms [3]. Cycles runs one kernel source on CPU, CUDA, OptiX, HIP, oneAPI and Metal [4]. {tag('c')}", "Keep Python handles as views over columnar data; add bulk array accessors; tag changes and evaluate only what depends on them."],
        ["<b>Unreal Engine</b>", f"The renderer mirrors game objects as scene proxies and primitive scene info [5]; per-primitive GPU data lives in buffers updated for changed primitives. Hot CPU loops in Chaos physics use ISPC [6]. {tag('c')}", "A render-side mirror with dirty uploads; SIMD-friendly kernels for the few hot loops."],
        ["<b>Godot</b>", f"Scene nodes are front ends for servers addressed by opaque RIDs; heavy users call the RenderingServer directly, and MultiMesh draws many instances from one buffer [7][8]. {tag('c')}", "Stable ids into a flat service; instance buffers for repeated meshes, as Aether3D already batches draws."],
        ["<b>Bevy</b>", f"ECS columns; the renderer extracts components into a separate render world each frame (0.6) [9] and later moved to GPU-driven rendering (0.16) [10]. {tag('c')}", "Extract only changed components; keep render data resident on the GPU."],
        ["<b>Unity</b>", f"DOTS jobs written in HPC#, a C# subset compiled by LLVM through Burst [11]. {tag('c')}", "A small, restricted kernel layer, which for Aether3D means Rust kernels with NumPy fallbacks."],
        ["<b>OpenUSD Hydra</b>", f"A scene delegate reports per-prim dirty bits; render delegates pull only what changed [12]. {tag('c')}", "A dirty-bit protocol between the store and each consumer (renderer, spatial index, exporters)."],
    ],
    cls="wide",
)
engsec = f"""<p>None of these engines got fast by swapping a scripting language for a compiled one at the API. They all separate the authoring model from a render-side, data-oriented mirror, track changes, and keep hot loops in a small compiled layer.</p>{eng}"""
section("engines", "7. What other engines do", engsec)

# ---------------------------------------------------------------- 8. Architecture vs language
avl = table(
    ["Change", "Measured basis", "Effect", "Needs native code?"],
    [
        ["Persistent GPU buffers, coalesced range writes", f"create {e2g['create_buffer_with_data_1.28MB_us']:.0f} µs vs write 1 record {e2g['write_buffer_1_records_one_range_us']:.1f} µs {tag('m')}", "Upload cost proportional to change", "No"],
        ["Retained render world with dirty tracking", f"extract {st['extract']:.1f} ms rebuilt every frame {tag('m')}", f"Unchanged objects cost nothing; one dirty object ~0.1 ms {tag('e')}", "No"],
        ["Columnar transforms + cached world matrices", f"{cur_w10:.1f} → {np_w10:.1f} ms flat NumPy; {min(native_w10):.2f} ms native {tag('m')}", f"Full recompute {cur_w10 / np_w10:.0f}×; dirty subtrees only µs", f"Optional (saves another {np_w10 - min(native_w10):.1f} ms at 10k)"],
        ["Material resolution cache keyed by revision", f"{st['textures']:.1f} ms per frame {tag('m')}", f"~0 when unchanged {tag('e')}", "No"],
        ["Camera rebase on the GPU (hi/lo float32)", f"kernel {e2g['gpu_rebase_10k_kernel_us (timestamps)']:.1f} µs, same precision {tag('m')}", "Camera moves stop forcing a repack", "No (WGSL)"],
        ["Selective readback; preview without full readback", f"{st['readback']:.1f} ms at 720p; 13–60 ms at 1080p–4K {tag('r')}", f"Image-only: {img_only_readback:.1f} ms at 720p {tag('e')}", "No"],
        ["Draw order by NumPy argsort, kept between frames", f"{e2c['draw_order_ms']:.1f} ms Python sort {tag('m')}", "Sub-millisecond, or zero when unchanged", "No"],
        ["Spatial query cache per revision", f"{sub['spatial_warm_raycast_ms']:.1f} ms query, ~{HARD['raycast_traversal_ms']} ms traversal {tag('m')}", "~3.5× now", "No"],
        ["Native BVH traversal and build", f"{rng(st_ratio, lambda x: f'{x:.0f}')}×/thread; {rng(mt_ratio, lambda x: f'{x:.0f}')}× at 20 threads {tag('m')}", "CPU path tracer, spatial queries, CPU fallback", "<b>Yes</b>"],
        ["Native topology and edit operators", f"{topo_cur_ms / topo_rs:.0f}× topology {tag('m')}", "Interactive editing of large meshes", f"Partly (NumPy alone gets {topo_cur_ms / topo_np:.0f}×)"],
        ["Native PNG unfilter", f"~270 → ~5 ms {tag('m')}", "10× faster texture loading", "<b>Yes</b> (or Pillow)"],
    ],
    cls="wide",
)
est_tbl = table(
    ["Frame (10k objects, 1280×720)", "Composition", "Total"],
    [
        ["Today, unchanged or one object moved", "measured", f"{frame_same:.0f}–{frame_moved:.0f} ms {tag('m')}"],
        ["Retained, one object moved", " + ".join(f"{k} {v:.2f}" for k, v in est_one.items()), f"{est_one_total:.1f} ms {tag('e')}"],
        ["Retained, one moved, image-only readback", "as above, readback × 4/28 bytes", f"{est_one_img:.1f} ms {tag('e')}"],
        ["Retained, one moved, preview without readback", "as above, no readback", f"{est_one_none:.1f} ms {tag('e')}"],
        ["Columnar, all 10k objects moved (NumPy only)", " + ".join(f"{k} {v:.2f}" for k, v in est_all.items()), f"{est_all_total:.1f} ms {tag('e')}"],
    ],
    cls="wide",
)
avlsec = f"""
<p>The table lists each candidate change with the measurement behind it. Most of the frame-time win needs no native code.</p>
{avl}
<h3>Composed frame estimates</h3>
<p>These estimates add measured component times. The materials cache is assumed to cost 0.5 ms; command encoding and GPU passes are assumed unchanged. Python overhead for the dirty-update bookkeeping is not measured, so expect the real numbers to be somewhat higher. Even when every object moves, the columnar NumPy frame ({est_all_total:.0f} ms) is {frame_same / est_all_total:.0f}× faster than today. Native code would shave about another {np_w10 - min(native_w10) + e2c['full_pack_numpy_ms'] - e2c['full_pack_c_ms']:.1f} ms off that.</p>
{est_tbl}
<h3>Does SoA or ECS fit Aether3D?</h3>
<p>A full ECS (archetypes, systems scheduling) would turn the object-centric Python API, plans and MCP tools inside out. A columnar store behind the existing handles fits better, the way Blender's RNA sits over its data and Godot's nodes sit over RIDs:</p>
<ul>
<li>Every object keeps its UUID. A map from id to row index resolves it, and <code>SceneObject</code> stays the same Python object per id (cached) but reads and writes columns.</li>
<li>Columns: position, rotation and scale (float64, as today), parent row, visibility and flags, mesh and material references. Rarely used components (camera, light, metadata) can stay as Python objects.</li>
<li>Each column has a dirty bitset and a revision counter. The render world, world-matrix cache, spatial index and exporters consume dirty sets and clear their own copies.</li>
<li>History records column deltas per operation; today's per-object touches map onto that. Undo restores rows, so ids and handle identity survive.</li>
</ul>
"""
section("arch-vs-lang", "8. Architecture or language? Quantified", avlsec)

# ---------------------------------------------------------------- 9. Misconceptions
myths = [
    ("Python is why the renderer is slow.", f"The GPU passes take {st['gpu']:.1f} ms of {frame_same:.0f} ms. The rest is per-frame, per-object data handling that a retained design removes in Python ({tag('m')} section 4.1, {tag('e')} section 8)."),
    ("Rewriting in Rust or C++ would make Aether3D N times faster.", f"Only where the algorithm is scalar (BVH, topology, PNG filters). For the frame, a native port of the same O(N)-per-frame design still does O(N) work per frame. Flat NumPy arrays already get world matrices {cur_w10 / np_w10:.0f}× and packing {e2c['pack_instances_ms'] / e2c['full_pack_numpy_ms']:.0f}× faster (E1, E2)."),
    ("Rust is faster than C (or the reverse).", f"Same algorithm, same backend, nearly the same speed: clang, Rust and Zig land within {llvm_spread * 100:.0f}%. Backends differ: GCC beat LLVM by up to {max(1 / x for x in llvm_vs_gcc):.1f}× on traversal, and Rust's SwissTable beat a simple C hash table on topology (E3, E5)."),
    ("Native code breaks determinism.", "Seven languages reproduced Aether3D's float64 traversal bit for bit, on 1 to 20 threads. What breaks it is FMA contraction or fast-math, which GCC enables by default for GNU C when the target has FMA (E1, E3)."),
    ("Crossing into native code is too expensive for fine-grained APIs.", "nanobind, PyO3, Cython and the C API cost 11–13 ns per call, the same as a Python call. ctypes is the slow path, at 80–464 ns (E4)."),
    ("Numba gives native speed without a native build.", f"It does give LLVM-tier speed (E3), but each new process pays {HARD['numba_import_ms']} ms of import and {HARD['numba_compile_s']} s of compile for one kernel, and llvmlite has to catch up with each CPython release."),
    ("The GPU path tracer already uses the RT cores.", "It runs a software BVH in WGSL. Hardware ray queries would trace 6–15× faster, but wgpu-py exposes no acceleration-structure API (E7)."),
    ("Hardware RT will make path tracing 10× faster.", "Traversal, yes. Full paths are shading-bound (33–67 M rays/s against 228 M for camera and AO rays), so the end-to-end gain waits on restructuring shading."),
    ("Embree is a drop-in replacement for the CPU BVH.", f"It is {rng(emb_over_c, lambda x: f'{x:.1f}')}× faster than the best scalar port, but it uses float32 and agrees on 99.99% of triangles, not 100%. It would change CPU goldens, so it belongs behind an opt-in."),
    ("Moving to SoA means breaking the public API.", "Handles can stay the same Python objects with the same UUIDs; only their storage moves. Blender has kept its Python API stable through exactly this kind of storage change."),
    ("Threads in Python need free-threaded CPython.", f"Native kernels release the GIL and scale {rng(scale20, lambda x: f'{x:.0f}')}× on 20 threads today. The free-threaded 3.14 build is supported, but it does not make per-object Python work cheaper " + tag("c") + "."),
]
mythsec = "".join(f'<div class="myth"><div class="claim">{esc(a)}</div><div>{b}</div></div>' for a, b in myths)
section("myths", "9. Misconceptions to retire", mythsec)

# ---------------------------------------------------------------- 10. Target architecture
target = """
<pre class="mermaid">
flowchart TB
  subgraph py["Python: unchanged public surface"]
    api["Scene · SceneObject handles · plans · MCP · CLI"]
    hist["History: column deltas per operation (undo/redo)"]
  end
  subgraph store["Scene store: columnar, NumPy first"]
    cols["Columns: position · rotation · scale (float64) · parent row · flags · mesh and material refs"]
    ids["id to row map · stable UUIDs · cached handles"]
    dirty["Dirty bitsets + revision per column"]
  end
  subgraph derived["Derived state, rebuilt only where dirty"]
    wm["World matrices: dirty subtrees"]
    rw["Retained render world: · instance slots · draw batches"]
    mat["Material resolution cache"]
    si["Spatial index: refit or rebuild per revision"]
    blas["BLAS cache (exists)"]
  end
  subgraph native["Native kernels: one Rust crate via PyO3, NumPy fallback for each"]
    k1["BVH build + traversal"]
    k2["Hierarchy update · packing"]
    k3["Mesh topology · edit operators"]
    k4["PNG filter and unfilter"]
  end
  subgraph gpu["GPU: wgpu-py + WGSL"]
    buf["Persistent buffers · coalesced range writes"]
    reb["Camera rebase in WGSL"]
    out["Selective readback · preview path"]
    ptg["WGSL path tracer"]
  end
  hw["Later, optional: Rust wgpu host · for hardware ray queries"]
  api --> cols
  api --> hist
  hist --> dirty
  cols --> dirty
  dirty --> wm
  dirty --> rw
  dirty --> mat
  dirty --> si
  wm --> rw
  rw --> buf
  buf --> reb
  buf --> out
  si --> k1
  blas --> k1
  wm -.-> k2
  ptg -.-> hw
</pre>
"""
targetsec = f"""
<p>The target keeps every public contract and changes what sits under it. It is reached in phases (section 13), and each phase leaves the engine shippable.</p>
{target}
<div class="two">
<div class="card"><h4>Contracts between layers</h4>
<ul>
<li>The store owns truth. Derived state never writes back.</li>
<li>Every consumer reads dirty sets since its last revision and is correct if it rebuilds everything, so "rebuild all" is always the fallback.</li>
<li>Every native kernel has a NumPy twin. <code>AETHER3D_NATIVE=0</code> selects the twin, and CI checks both bit for bit.</li>
<li>File format, ids and history semantics do not change.</li>
</ul></div>
<div class="card"><h4>Languages by layer</h4>
<ul>
<li><b>Python:</b> API, validation, plans, MCP, CLI, history, serialization, orchestration of derived state.</li>
<li><b>NumPy:</b> columns, vectorized updates, fallbacks and references.</li>
<li><b>Rust (one crate, abi3 wheels):</b> BVH, hierarchy, topology and edit operators, image filters.</li>
<li><b>WGSL:</b> all shaders, as today.</li>
<li><b>Later:</b> a Rust wgpu host if hardware RT is adopted.</li>
</ul></div>
</div>
"""
section("target", "10. Target architecture", targetsec)

# ---------------------------------------------------------------- 11. Modules
mods = [
    ("rendering/scene_data.py", "REARCHITECT", "Python + NumPy", f"Rebuilds a RenderInstance per object every frame ({st['extract']:.1f} ms); one moved object costs a full frame {tag('m')}", f"Frame {frame_same:.0f} → ~{est_one_total:.0f} ms (~{est_one_img:.0f} ms image-only) {tag('e')} · medium", "High", "Medium (render parity)", "P1", "Dirty tracking (store or revisions)"),
    ("rendering/gpu/renderer.py, resources.py", "REARCHITECT", "Python + WGSL", f"New buffers per frame (renderer.py:233); pack {e2c['pack_instances_ms']:.1f} ms; draw order {e2c['draw_order_ms']:.1f} ms; 28 B/pixel readback {tag('m')}", f"Upload {st['upload']:.0f} ms → &lt;0.1 ms for small changes {tag('e')} · high", "Medium", "Low–medium", "P1", "None; can start first"),
    ("scene.py, transform.py", "REARCHITECT", "Python handles over NumPy columns", f"{sub['bytes_per_object_incl_history']} B/object; world_matrix {sub['read_world_matrix_us']:.1f} µs per read via parents {tag('m')}", f"Enables O(changes) frames; world matrices {cur_w10 / np_w10:.0f}× {tag('m')} · medium", "High", "High (identity, undo)", "P2", "History adapter"),
    ("history.py, _history_file.py", "KEEP", "Python", f"Undo/redo of 10k plan {HARD['undo_redo_10k_ms'][0]}/{HARD['undo_redo_10k_ms'][1]} ms {tag('m')}", "Unchanged semantics; records become column deltas", "Medium", "Medium", "with P2", "Store"),
    ("plans/", "OPTIMIZE", "Python", f"{HARD['plan_10k_ms'] / 10:.1f} µs/object; floor {HARD['floor_us']} µs; uuid4 {HARD['uuid_us']} µs/id {tag('m')}", f"~3–4 µs/object (batched ids, vectorized validation, shared default appearance) {tag('e')} · low", "Low", "Low", "P2", "None"),
    ("mcp/, cli.py", "KEEP", "Python", f"Protocol and orchestration; batch plans already 200× faster than per-object calls {tag('r')}", "None needed", "–", "–", "–", "–"),
    ("spatial/index.py", "OPTIMIZE", "Python, then the native traversal", f"O(N) filter + tuple key per query (index.py:323) is most of a {sub['spatial_warm_raycast_ms']:.1f} ms raycast {tag('m')}", f"~{HARD['raycast_traversal_ms']} ms now, ~0.1 ms with native traversal {tag('e')} · medium", "Low", "Low", "P1", "None"),
    ("rendering/raytrace/", "MOVE TO NATIVE", "Rust (NumPy kept as reference)", f"{min(st_np):.2f}–{max(st_np):.2f} M rays/s; native {rng(st_ratio, lambda x: f'{x:.0f}')}×/thread, bit-identical {tag('m')}", f"{min(st_ratio):.0f}–{max(mt_ratio):.0f}× for CPU tracing and queries {tag('m')} · high", "Medium", "Medium (cross-OS parity)", "P2", "Native crate and CI"),
    ("rendering/pathtrace/ (CPU)", "DEFER", "Rust, after raytrace", f"GPU path tracer is 51–65× faster {tag('r')}; CPU path serves CI, no-GPU machines, references", f"Large on CPU-only machines {tag('e')} · medium", "High", "Medium", "P3", "Native raytrace"),
    ("rendering/gpu/wgsl/pathtrace.wgsl", "KEEP", "WGSL", f"Shading-bound: 33–67 M rays/s full paths {tag('r')}", "Wavefront or material sorting before HW RT", "–", "–", "P3", "–"),
    ("Hardware ray tracing (new)", "DEFER", "Rust wgpu host, or an upstream wgpu-native C API", f"{rng(hw_ratio, lambda x: f'{x:.1f}')}× traversal; unreachable from wgpu-py {tag('m')}", f"Traversal 6–15×; end-to-end less (shading) {tag('e')} · medium", "High", "High (experimental API)", "P4", "Native crate; shading rework"),
    ("CPU rasterizer (rendering/)", "KEEP", "NumPy", f"Reference and fallback; GPU is 5–63× faster {tag('r')}", "–", "–", "–", "–", "–"),
    ("rendering/materials.py", "OPTIMIZE", "Python", f"resolve_materials {st['textures']:.1f} ms/frame for unchanged materials {tag('m')}", f"~0 when unchanged {tag('e')} · high", "Low", "Low", "P1", "Revisions"),
    ("materials/images.py (PNG decode)", "MOVE TO NATIVE", "Rust kernel (Pillow fast path when installed)", f"Unfilter ~270 ms vs ~5 ms in C {tag('m')}", f"~10× faster decode {tag('m')} · high", "Low", "Medium (fuzz the decoder)", "P2", "Native crate"),
    ("rendering/png.py (encode)", "OPTIMIZE", "Python (+ native filter)", f"Filter {HARD['png_filter_ms']:.0f} ms + zlib-6 {HARD['zlib_ms'][6]:.0f} ms at 1080p {tag('m')}", f"Preview level 1: ~{HARD['png_filter_ms'] + HARD['zlib_ms'][1]:.0f} ms; with native filter ~{HARD['zlib_ms'][1] + 5:.0f} ms {tag('e')} · high", "Low", "Low", "P2", "–"),
    ("materials/ node evaluation (CPU)", "KEEP", "NumPy", "53 ns–1.9 µs per point; GPU path in material.wgsl " + tag("m"), "–", "–", "–", "–", "–"),
    ("editing/ (core, topology, bevel)", "REARCHITECT", "NumPy arrays, then Rust operators", f"from_mesh {e5['EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s']:.0f} s at 200k tris; topology {topo_cur_ms / topo_np:.0f}× in NumPy, {topo_cur_ms / topo_rs:.0f}× in Rust; bevel {HARD['bevel_51k_s']} s for 51k faces {tag('m')}", f"Open 200k-tri meshes in &lt;1 s; operators 10–100× {tag('e')} · medium", "High", "Medium–high (journal, diagnostics)", "P3", "Native crate"),
    ("procedural/", "KEEP", "NumPy", f"Vectorized, cached; subdivision {HARD['subdiv_51k_ms']} ms for 51k faces {tag('m')}", "–", "–", "–", "–", "–"),
    ("gltf/", "KEEP", "Python + NumPy", f"GLB export {sub['export_glb_s'] * 1e3:.0f} ms at 10k {tag('m')}", "–", "–", "–", "–", "–"),
    ("serialization.py (+ file format)", "KEEP", "Python", f"Save {sub['save_s']:.2f} s, load {sub['load_s']:.2f} s at 10k {tag('m')}", "Format unchanged", "–", "–", "–", "–"),
    ("WGSL shaders (scene, material, shadow)", "KEEP", "WGSL", f"GPU passes {st['gpu']:.1f} ms of {frame_same:.0f} ms {tag('m')}", "–", "–", "–", "–", "–"),
    ("inspection, diagnostics, errors, math3d, mesh, primitives", "KEEP", "Python", "Small, not on hot paths", "–", "–", "–", "–", "–"),
]
modtbl = table(
    ["Module", "Verdict", "Language", "Evidence", "Expected benefit · confidence", "Complexity", "Risk", "Priority", "Depends on"],
    [[code(m), chip(v), esc(lng), ev, ben, cx, rk, pr, dp] for (m, v, lng, ev, ben, cx, rk, pr, dp) in mods],
    cls="xwide",
)
modsec = f"""<p>P1 is the first phase after Stage 11, P4 the last. Confidence describes how sure I am of the expected benefit: high means the benefit is measured directly, medium means it is composed from measurements, low means it is a judgment.</p>{modtbl}"""
section("modules", "11. Module-by-module recommendations", modsec)

# ---------------------------------------------------------------- 12. Invariants
inv = table(
    ["Must not change", "Where the risk is", "How the plan preserves it", "Gate"],
    [
        ["Public API: Scene, SceneObject, plans, MCP tools, CLI", "Store rework under the facade", "Handles keep their methods and properties; the store is internal", "Existing 2,006 tests unchanged; never weakened"],
        ["Object identity", "Rows move on delete and undo", "UUID ids stay the key; a handle cache returns the same object per id; tombstones or a row map instead of renumbering ids", "Property-based tests: identity across delete, undo, redo, save, load"],
        ["Undo / redo", "Column deltas replace per-object records", "Same operation boundaries and labels; deltas restore exact float64 values", "History replay tests on random edit scripts; bitwise state comparison"],
        ["Determinism and the CPU reference", "Native kernels, threads", "No fast-math and no FMA contraction; fixed summation order; per-ray independence; NumPy twins", "Bitwise parity of native vs NumPy on random scenes, on all three CI operating systems"],
        ["File format (.aether, history file)", "Columnar internals", "Serializer reads and writes the same JSON from columns", "Round-trip tests on files from v0.10/v0.11; byte-identical saves"],
        ["Portability (Linux, macOS, Windows; CPython 3.10–3.14; no-wgpu and no-manifold installs)", "A compiled extension", "abi3 wheels per OS and architecture; pure-Python fallback when the extension is missing", "A CI job with the extension absent; the existing optional-dependency jobs"],
        ["Security (MCP redaction and sandbox, decoder limits)", "Native parsing of untrusted data", "Rust for decoders; ImageLimits checked before native calls; sizes validated at the boundary", "cargo-fuzz targets for decoders; existing MCP security tests"],
        ["Golden images", "Kernel changes", "CPU goldens stay bit-identical (float64 reference unchanged); GPU goldens keep their tolerances; Embree only as an opt-in", "Golden suite on every PR"],
        ["The 17 CI jobs", "New build steps", "Keep all 17. Add: a wheel build per OS, a native-off job, a parity job, and ratio-based performance gates (as in tests/timing.py)", "CI green before any merge"],
        ["Licenses (Apache-2.0, no GPL)", "New dependencies", "PyO3 and maturin (MIT/Apache-2.0), Embree (Apache-2.0), wgpu (MIT/Apache-2.0); cargo-deny license check", "License check in CI"],
    ],
    cls="xwide",
)
section("invariants", "12. What must not break", inv)

# ---------------------------------------------------------------- 13. Migration
phases = table(
    ["Phase", "Scope", "Parity tests", "Performance gate (RTX 4060 laptop, 10k objects)", "Rollback", "Effort", "Risk"],
    [
        ["0 · Guard rails", "Benchmark suite in the repo (frame, query, creation, editing); parity harness; feature flags", "Harness compares old and new paths on generated scenes", "Baselines recorded; gates are ratios, not absolute times", "Nothing to roll back", "~1 week", "Low"],
        ["1 · Python quick wins", "Spatial query cache; batched ids; NumPy draw order; material resolution cache; persistent GPU buffers with range writes; selective readback; PNG preview level", "Pixel and result parity on goldens; query results identical", f"Raycast ≤ 2.5 ms (from {sub['spatial_warm_raycast_ms']:.1f}); unchanged frame ≤ 80 ms (from {frame_same:.0f})", "Each change behind a flag for one release", "1–2 weeks", "Low"],
        ["2 · Retained render world + columnar transforms", "Store columns with dirty bitsets; world-matrix cache; render world with instance slots; GPU camera rebase; history column deltas", "Bitwise world matrices; byte-identical instance records; identity and undo property tests", "One moved object ≤ 15 ms with image-only readback; all moved ≤ 25 ms; creation ≤ 5 µs/object", "AETHER3D_RETAINED=0 restores per-frame rebuild", "3–5 weeks", "Medium–high"],
        ["3 · Native kernel crate", "Rust + PyO3 + maturin, abi3 wheels for 3 OSes; BVH traversal and build, hierarchy, PNG filters, topology; NumPy twins kept", "Bitwise vs NumPy twin on random scenes on every CI OS", "Traversal ≥ 8× NumPy per thread on all E3 workloads; PNG decode ≥ 50 MP/s at 1080p", "AETHER3D_NATIVE=0; wheel can be yanked; pure-Python path always tested", "4–6 weeks", "Medium"],
        ["4 · Data-oriented EditableMesh", "Array half-edge and corner layers; batched normals; Rust operators (bevel, extrude)", "Topology, diagnostics and journal identical on the editing test corpus", "from_mesh 200k tris with attributes ≤ 1 s; bevel 51k faces ≤ 1 s", "Old EditableMesh kept for one release behind a flag", "4–8 weeks", "High"],
        ["5 · Optional hardware RT", "Rust wgpu host (or contribute BLAS/TLAS to wgpu-native's C API); ray queries in the WGSL path tracer after shading rework", "Agreement with the software kernel (≥ 99.99% same triangle), image-level tolerances", "Showcase path renders ≥ 2× faster end to end", "Software traversal remains the default", "6–10 weeks", "High"],
    ],
    cls="xwide",
)
migsec = f"""
<p>Each phase lands as normal staged PRs into <code>main</code>, never stacked, each with tests, benchmarks and docs, and each mergeable on its own. Every new path ships behind a flag with the old path still tested, and the old path is removed only after a release in which the new one was the default with no regressions. Effort assumes one experienced developer working with AI assistance {tag('e')}.</p>
{phases}
<h3>Modernize right after Stage 11, or phase it in?</h3>
<p><b>Phase it in, starting right after Stage 11 merges.</b> Phases 0 and 1 are small, need no new toolchain, and fix problems AI users of Aether3D feel directly: query latency, frame time and creation speed. Phase 2 is the real architectural change, and it is easier before more features are built on the per-frame extraction. Phase 3 should follow once the guard rails exist, because it adds a compiled extension to a 17-job matrix that is all pure Python today. Phases 4 and 5 can interleave with feature stages when there is a need. A big-bang migration would stall feature work for months and put identity, undo and goldens at risk all at once.</p>
"""
section("migration", "13. Migration plan", migsec)

# ---------------------------------------------------------------- 14. Top 10
top = table(
    ["#", "Change", "Where", "Evidence", "Expected gain", "Effort", "Native?"],
    [
        ["1", "Retained render world with dirty tracking", code("rendering/scene_data.py"), f"Full rebuild each frame; one move = {frame_moved:.0f} ms {tag('m')}", f"{frame_same:.0f} → ~{est_one_total:.0f} ms {tag('e')}", "Weeks", "No"],
        ["2", "Persistent GPU buffers, coalesced writes, GPU camera rebase", code("rendering/gpu/renderer.py:233, resources.py:621"), f"{e2g['create_buffer_with_data_1.28MB_us']:.0f} µs vs {e2g['write_buffer_1_records_one_range_us']:.1f} µs; rebase {e2g['gpu_rebase_10k_kernel_us (timestamps)']:.1f} µs {tag('m')}", f"Upload {st['upload']:.0f} ms → ~0 {tag('e')}", "Days", "No"],
        ["3", "Selective readback and a preview path", code("rendering/gpu/renderer.py:434"), f"28 B/pixel read back every frame {tag('r')}", f"{st['readback']:.1f} → {img_only_readback:.1f} ms at 720p; 55–60 ms saved at 4K {tag('e')}", "Days", "No"],
        ["4", "Columnar transforms and cached world matrices", code("scene.py, transform.py"), f"{cur_w10:.1f} → {np_w10:.1f} ms (E1) {tag('m')}", f"Frame and queries; enables 1–2 {tag('m')}", "Weeks", "No"],
        ["5", "Spatial query cache", code("spatial/index.py:323"), f"O(N) per query {tag('m')}", f"{sub['spatial_warm_raycast_ms']:.1f} → ~{HARD['raycast_traversal_ms']} ms {tag('e')}", "Hours", "No"],
        ["6", "Native CPU BVH (Rust), NumPy kept as reference", code("rendering/raytrace/"), f"Bit-identical, {rng(st_ratio, lambda x: f'{x:.0f}')}×/thread {tag('m')}", f"{rng(mt_ratio, lambda x: f'{x:.0f}')}× at 20 threads {tag('m')}", "Weeks", "Yes"],
        ["7", "Material resolution cache", code("rendering/materials.py:174"), f"{st['textures']:.1f} ms per frame {tag('m')}", f"~{st['textures']:.0f} ms saved per frame {tag('e')}", "Days", "No"],
        ["8", "Vectorized from_mesh, then array half-edge", code("editing/core.py:275, topology.py:573"), f"{e5['EditableMesh.from_mesh(sphere with normals/uvs, quads=True) s']:.0f} s at 200k tris; topology {topo_cur_ms / topo_np:.0f}× in NumPy {tag('m')}", f"10–40× to open meshes {tag('e')}", "Weeks", "Later"],
        ["9", "Native PNG unfilter (or Pillow when installed)", code("materials/images.py:331"), f"~270 ms vs ~5 ms {tag('m')}", f"~10× decode {tag('m')}", "Days", "Yes"],
        ["10", "Batched ids and vectorized plan validation", code("scene.py:77, :1065"), f"uuid4 {HARD['uuid_us']} µs/id; {HARD['plan_10k_ms'] / 10:.1f} µs/object {tag('m')}", f"~2× creation {tag('e')}", "Days", "No"],
    ],
    cls="xwide",
)
section("top10", "14. Top ten changes", top + "<p class='note'>Ranked by expected benefit per unit of effort and risk. Seven of the ten need no native code.</p>")

# ---------------------------------------------------------------- 15. Next benchmarks
nextsec = """
<ol>
<li>A retained-frame prototype, to replace the composed estimates in section 8 with measurements, including Python bookkeeping for dirty sets.</li>
<li>Memory and access cost of a columnar store with cached handles (target: under 300 B/object, handle reads within 2× of today).</li>
<li>Native traversal inside the real CPU path tracer: showcase renders with native threads against today's 8 processes.</li>
<li>Embree as an opt-in: golden-image differences, and speed on Windows and macOS (including Apple Silicon).</li>
<li>Every native kernel on macOS and Windows: bitwise parity, speed, and MSVC vs clang-cl if C++ is ever chosen.</li>
<li>Hardware ray queries inside the WGSL path tracer (traversal swapped, shading unchanged) to measure the end-to-end gain before committing to a native GPU host.</li>
<li>Persistent buffers and selective readback on lavapipe, which is what CI runs.</li>
<li>Free-threaded CPython 3.14 with native kernels, against processes, for batch rendering.</li>
<li>Wheel build time and size in CI for the abi3 crate on all target platforms.</li>
<li>Untested languages only if a non-performance reason appears (for example a Swift or Kotlin host application).</li>
</ol>
"""
section("next", "15. What to benchmark next", nextsec)

# ---------------------------------------------------------------- 16. Risks
risks = table(
    ["Risk", "Likelihood", "Impact", "Mitigation"],
    [
        ["Identity or undo regressions in the columnar store", "Medium", "High", "Handle cache; property-based identity and history replay tests; flag-guarded rollout"],
        ["A compiled extension breaks the 3-OS × 5-Python matrix", "Medium", "Medium", "abi3 wheels; pure-Python fallback job; build in CI before merge"],
        ["Floating-point divergence across compilers or OSes", "Medium", "High", "No fast-math or contraction; avoid libm transcendentals in parity paths; bitwise tests on every OS"],
        ["Embree's float32 results change goldens", "High if adopted", "Medium", "Opt-in only; float64 reference stays default"],
        ["wgpu's experimental ray-query API changes", "High", "Medium", "Defer; isolate behind an interface; consider upstreaming a C API"],
        ["Native decoders become an attack surface", "Medium", "High", "Rust; limits checked first; fuzzing in CI"],
        ["Performance gates flake on shared CI runners", "High", "Low", "Ratio gates with generous margins, GC paused, as tests/timing.py already does"],
        ["Feature work stalls behind infrastructure", "Medium", "Medium", "Phases 0–1 first and small; each phase shippable"],
        ["Rust raises the contributor bar", "Medium", "Low", "Small kernels with NumPy twins as executable specifications"],
    ],
    cls="wide",
)
section("risks", "16. Risks", risks)

# ---------------------------------------------------------------- 17. Decision
decision = """
<p>Nothing has been changed. No source file was edited, no PR was opened or merged, Stage 12 has not started, and no migration implementation prompt has been written. To move forward I need your decisions on:</p>
<ol>
<li><b>Direction.</b> Keep Python as the engine's language, with a phased, architecture-first plan.</li>
<li><b>Native language.</b> Rust through PyO3 and maturin as the single native language, or C++ through nanobind if you expect to build on Embree, OptiX or USD.</li>
<li><b>First step after PR #13 merges.</b> Phases 0 and 1 (guard rails and Python-only quick wins), or a different order.</li>
<li><b>Hardware ray tracing.</b> Deferred until shading is restructured, or prototyped earlier (benchmark 6 in section 15).</li>
</ol>
<p>PR #13 (Stage 11) is still open and unmerged, waiting for your approval.</p>
"""
section("decision", "17. Decisions needed", decision)

# ---------------------------------------------------------------- References and appendix
refs = [
    ("Blender developer docs: Depsgraph, copy-on-evaluation", "https://developer.blender.org/docs/features/core/depsgraph/"),
    ("Blender task T95965: mesh struct-of-arrays refactor", "https://developer.blender.org/T95965"),
    ("Blender commit: foreach_get/foreach_set for image pixels (2,900 ms → 29 ms)", "https://developer.blender.org/rB9075ec8269e7cb029f4fab6c1289eb2f1ae2858a"),
    ("Blender manual: Cycles render devices", "https://docs.blender.org/manual/en/dev/editors/preferences/system.html"),
    ("Unreal Engine API: FPrimitiveSceneInfo", "https://dev.epicgames.com/documentation/unreal-engine/API/Runtime/Renderer/FPrimitiveSceneInfo"),
    ("Intel: Unreal Engine's Chaos physics and ISPC", "https://www.intel.com/content/www/us/en/developer/articles/technical/unreal-engines-new-chaos-physics-system-screams-with-in-depth-intel-cpu-optimizations.html"),
    ("Godot docs: Optimization using Servers", "https://docs.godotengine.org/en/stable/tutorials/performance/using_servers.html"),
    ("Godot docs: Optimization using MultiMeshes", "https://docs.godotengine.org/en/latest/tutorials/performance/using_multimesh.html"),
    ("Bevy 0.6 release notes: pipelined rendering", "https://bevy.org/news/bevy-0-6/"),
    ("Bevy 0.16 release notes: GPU-driven rendering", "https://bevy.org/news/bevy-0-16/"),
    ("Unity manual: Burst compiler and HPC#", "https://docs.unity.com/en-us/engine/6000.3/manual/programming-environment/script-compilation/burst"),
    ("NVIDIA: Learn OpenUSD, Hydra", "https://docs.nvidia.com/learn-openusd/latest/beyond-basics/hydra.html"),
    ("wgpu documentation: ray tracing extensions", "https://wgpu.rs/doc/wgpu/documentation/extensions/ray_tracing/index.html"),
    ("Wald et al., Embree: A Kernel Framework for Efficient CPU Ray Tracing (2014)", "https://www.sci.utah.edu/~wald/Publications/2014/embree/embree.pdf"),
    ("nanobind benchmarks", "https://nanobind.readthedocs.io/en/latest/benchmark.html"),
    ("PyO3 user guide", "https://pyo3.rs"),
    ("PEP 779: criteria for supported status of free-threaded Python", "https://peps.python.org/pep-0779/"),
    ("Numba 0.63 release notes (Python 3.14)", "https://numba.readthedocs.io/en/stable/release/0.63.0-notes.html"),
    ("Go issue 42469: cgo call overhead", "https://go.dev/issue/42469"),
    ("Microsoft Learn: Native AOT interop and native libraries", "https://learn.microsoft.com/dotnet/core/deploying/native-aot/interop"),
    ("Julia 1.12 highlights (juliac --trim, experimental)", "https://julialang.org/blog/2025/10/julia-1.12-highlights/"),
    ("Zig still has no 1.0 release (byteiota)", "https://byteiota.com/zig-still-wont-ship-1-0-thats-the-point/"),
    ("Phoronix: Modular open-sources Mojo (secondary source)", "https://www.phoronix.com/news/Modular-Mojo-Open-Source"),
    ("Khronos: Slang initiative (WGSL and Metal targets experimental)", "https://www.khronos.org/news/press/khronos-group-launches-slang-initiative-hosting-open-source-compiler-contributed-by-nvidia"),
    ("NVIDIA developer forum: OptiX in commercial software", "https://forums.developer.nvidia.com/t/optix-use-in-commercial-software/222703"),
    ("heise: NVIDIA Warp under Apache 2.0", "https://heise.de/-10381872"),
]
refhtml = "<ol class='refs'>" + "".join(f'<li>{esc(t)} · <a href="{esc(u)}" target="_blank" rel="noopener">{esc(u)}</a></li>' for t, u in refs) + "</ol>"
appendix = f"""
<h3>References</h3>
<p class="note">Numbers in brackets in section 7 refer to this list; {tag('c')} marks other claims from it.</p>
{refhtml}
<h3>Reproducing the experiments</h3>
<p>All scripts and raw results are in the session scratchpad under <span class="kbd">{esc(str(L))}</span>. Run Python scripts from the repository root with <code>PYTHONPATH=src</code> and the lab interpreter <span class="kbd">lab/venv/bin/python</span>.</p>
""" + table(
    ["Experiment", "Scripts", "Results"],
    [
        ["Subsystems", "m/subsystems.py, m/prof_extract.py, m/imports.py", "m/subsystems.json, m/imports.json"],
        ["E1 world matrices", "e1/bench_e1.py, e1/aether_world.py; kernels world.c, world.zig, rust/, go/, cs4/", "e1/e1.json"],
        ["E2 packing and upload", "e2/bench_e2_cpu.py, e2/bench_e2_gpu_precision.py, e2/pack.c", "e2/e2_cpu.json, e2/e2_gpu_precision.json"],
        ["E3 CPU traversal", "e3/e3common.py, bench_e3.py, bench_mt_interleaved.py, bench_cpp.py, bench_embree.py; trace.c, rust/, trace.zig, go/, cs/, trace_numba.py, embree_bench.c", "e3/e3_results.json, e3_mt_interleaved.json, embree_results.json"],
        ["E4 boundary", "e4/bench_ffi.py, e4/prof_create.py; capi.c, cy_mod.pyx, nb_mod.cpp, pyo3/", "e4/ffi.json"],
        ["E5 topology", "e5/bench_e5.py, e5/prof_from_mesh.py; topo.c, rust/", "e5/e5.json"],
        ["E6 PNG", "e6/bench_e6.py; unfilter.c", "e6/e6.json"],
        ["E7 GPU traversal", "e7/export.py, e7/rtbench (Rust), e7/check.py", "e7/e7_runs.jsonl, e7/data/agreement.json"],
    ],
    cls="wide",
)
section("appendix", "References and reproduction", appendix)

# ---------------------------------------------------------------- assemble
toc = "".join(f'<li><a href="#{sid}">{esc(t)}</a></li>' for sid, t, _ in sections)
headhtml = f"""
<header class="doc">
<div class="eyebrow">AETHER3D · ARCHITECTURE AND LANGUAGE AUDIT · 15 OCTOBER 2026</div>
<h1>Should Aether3D change languages?</h1>
<div class="meta"><span><b>Scope</b> main 38697d5 (Stages 1–10) + PR #13 at 403581f (Stage 11, unmerged)</span><span><b>Machine</b> i9-12900H · RTX 4060 Laptop · Arch Linux</span><span><b>Evidence</b> 7 experiments, 11 implementations, every port checked against Aether3D's output</span></div>
<div class="status"><b>Research only.</b> No source code was changed, no PR was opened or merged, and Stage 12 has not started. Every toolchain and experiment lives in a session scratchpad.</div>
</header>
"""
body = (
    CSS
    + '<div class="page"><nav class="toc" aria-label="Contents"><div class="toc-title">Contents</div><ol>'
    + toc
    + "</ol></nav><main>"
    + headhtml
    + "".join(h for _, _, h in sections)
    + '<p class="note" style="margin-top:3rem">Generated from the measurement files by report/build_report.py in the session scratchpad.</p></main></div>'
)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(body, encoding="utf-8")
print(f"wrote {OUT} ({len(body) / 1024:.0f} KB, {len(sections)} sections)")
