import cProfile, pstats, io
from aether3d import create_sphere
from aether3d.editing import EditableMesh
m = create_sphere(1.0, 160, 80)
print("triangles", m.triangles.shape[0])
pr = cProfile.Profile(); pr.enable(); EditableMesh.from_mesh(m); pr.disable()
st = pstats.Stats(pr); buf = io.StringIO(); st.stream = buf
st.sort_stats("cumulative").print_stats("editing", 12)
print("\n".join(l for l in buf.getvalue().splitlines() if "editing" in l or "ncalls" in l)[:4000])
