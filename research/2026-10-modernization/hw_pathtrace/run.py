"""Phase 5 prototype: export, render, agree and compare showcase scenes.

    python run.py OUT HOST [SCENE ...] [--quick]

HOST is the host binary (``cargo build --release`` in ``host/``). For each scene:
``export.py`` (the Python renders and the capture), one host process rendering
``ORDER`` (the first render of each variant is a warm-up; the timed ones alternate),
one software render of the next ``samples`` samples (the noise reference), the
agreement kernel, and ``compare.py``. Without ``--quick`` the summaries go to
``results/SCENE.json`` next to this file. A scene's binary files (inputs and planes:
about 1 GB at 1920x1080) are deleted once its summary is written, unless ``--keep``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = "sw,hw,sw,hw,hw,sw"
SCENES = ["cornell", "metals", "window-room", "castle", "hero", "castle-ao"]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quick = "--quick" in sys.argv
    keep = "--keep" in sys.argv
    out, host, scenes = Path(args[0]), Path(args[1]), args[2:] or SCENES
    for scene in scenes:
        d = out / scene
        export = [sys.executable, str(HERE / "export.py"), str(out), scene]
        subprocess.run(export + (["--quick"] if quick else []), check=True)
        plan = dict(line.split(maxsplit=1) for line in (d / "plan.txt").read_text().splitlines())
        samples = plan["samples"]
        lines = []
        for cmd in (
            ["render", ORDER],
            ["render", "sw", "--sample-offset", samples],
            ["agree"],
        ):
            r = subprocess.run([str(host), str(d), *cmd], capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"{scene}: the host failed ({r.returncode}):\n{r.stderr[-4000:]}")
            lines += [line for line in r.stdout.splitlines() if line.startswith("{")]
        (d / "runs.jsonl").write_text("\n".join(lines) + "\n")
        r = subprocess.run(
            [sys.executable, str(HERE / "compare.py"), str(d)],
            check=True,
            capture_output=True,
            text=True,
        )
        summary = json.loads(r.stdout)
        if not quick:
            (HERE / "results").mkdir(exist_ok=True)
            (HERE / "results" / f"{scene}.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(scene, json.dumps(summary["timings"].get("speedup")), flush=True)
        if not keep:
            for f in d.glob("*.bin"):
                f.unlink()


if __name__ == "__main__":
    main()
