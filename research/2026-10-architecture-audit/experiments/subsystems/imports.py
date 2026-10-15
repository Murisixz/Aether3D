"""Static import graph of src/aether3d grouped by subsystem, with line counts."""
import ast, json
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path("/home/murisi/Aether3D/src/aether3d")
def group(mod: str) -> str:
    parts = mod.split(".")
    if parts[0] != "aether3d" or len(parts) == 1:
        return "aether3d (root)"
    if parts[1] == "rendering" and len(parts) > 2 and parts[2] in ("gpu", "raytrace", "pathtrace"):
        return f"rendering.{parts[2]}"
    if (ROOT / parts[1]).is_dir():
        return parts[1]
    return parts[1]  # top-level module (scene, mesh, history, ...)
lines = Counter(); edges = Counter(); wgsl = Counter()
for path in ROOT.rglob("*.py"):
    rel = path.relative_to(ROOT.parent).with_suffix("")
    mod = ".".join(rel.parts)
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    src = path.read_text(encoding="utf-8")
    g = group(mod); lines[g] += src.count("\n")
    pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
    for node in ast.walk(ast.parse(src)):
        targets = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                full = ".".join(base + ([node.module] if node.module else []))
            else:
                full = node.module or ""
            targets = [full + "." + a.name for a in node.names] if not node.module and node.level else [full]
        elif isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        for t in targets:
            if t.startswith("aether3d"):
                h = group(t)
                if h != g:
                    edges[(g, h)] += 1
for path in ROOT.rglob("*.wgsl"):
    wgsl[path.name] = path.read_text().count("\n")
print(json.dumps({"lines": dict(lines.most_common()), "wgsl": dict(wgsl), "edges": [[a, b, n] for (a, b), n in sorted(edges.items())]}, indent=0))
