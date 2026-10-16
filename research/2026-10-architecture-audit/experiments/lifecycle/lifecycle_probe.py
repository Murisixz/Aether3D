"""Does a Scene die after each kind of use? weakref + gc.collect, one fresh scene per case."""
import gc, os, sys, weakref
import numpy as np
import aether3d
from aether3d import Scene, create_cube, create_sphere

def build():
    s = Scene("probe")
    a = s.add_object("A", mesh=create_cube(1.0))
    b = s.add_object("B", mesh=create_sphere(0.5, 16, 8), position=(2.0, 0.0, 0.0))
    s.add_directional_light("Sun", direction=(-0.3, 0.5, -1))
    s.add_camera("Camera", position=(0, -8, 2)).frame(aspect=1.0)
    return s

def case_nothing(s): pass
def case_raycast(s): s.raycast((0, -5, 0), (0, 1, 0))
def case_overlaps(s): s.overlaps(s.get_by_name("A"))
def case_render_cpu(s): s.render(width=32, height=32, backend="cpu")
def case_render_gpu(s): s.render(width=32, height=32, backend="gpu")
def case_path_cpu(s): s.render(width=16, height=16, backend="path", path={"samples": 1, "bounces": 1})
def case_path_gpu(s): s.render(width=16, height=16, backend="gpu_path", path={"samples": 1, "bounces": 1})
def case_plan(s):
    s.execute_plan({"format": "aether3d.scene-plan", "version": 1, "meshes": {"m": {"primitive": "cube", "size": 0.5}},
                    "objects": {"g": {"mesh": "@m", "grid": {"count": [3, 3]}}}})
def case_procedural(s):
    s.procedural.add_object("P", "box", size=(1.0, 1.0, 1.0)) if hasattr(s.procedural, "add_object") else None
def case_edit(s):
    with s.get_by_name("A").edit_mesh() as em:
        pass
def case_undo(s):
    s.get_by_name("A").position = (1.0, 1.0, 1.0); s.history.undo(); s.history.redo()
def case_save_load(s):
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "x.aether"); s.save(p); Scene.load(p)
def case_export(s):
    import tempfile
    s.export(os.path.join(tempfile.mkdtemp(), "x.glb"))
def case_describe(s):
    from aether3d.spatial.request import run_query
    run_query(s, {"query": "describe"})

cases = [c for n, c in sorted(globals().items()) if n.startswith("case_")]
only = sys.argv[1:] or None
for case in cases:
    name = case.__name__[5:]
    if only and name not in only:
        continue
    s = build()
    try:
        case(s)
        err = ""
    except Exception as exc:  # report, keep probing
        err = f" ({type(exc).__name__}: {str(exc)[:80]})"
    ref = weakref.ref(s)
    del s
    gc.collect()
    alive = ref() is not None
    holders = ""
    if alive:
        holders = "; referrers: " + ", ".join(sorted({type(r).__name__ for r in gc.get_referrers(ref())}))[:200]
    print(f"{name:14s} {'LEAKED' if alive else 'freed '}{err}{holders}")
