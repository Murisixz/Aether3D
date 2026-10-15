import cProfile, pstats, io
from aether3d import Scene
from aether3d.rendering.scene_data import extract_render_scene
s = Scene("s")
s.execute_plan({"format": "aether3d.scene-plan", "version": 1, "meshes": {"block": {"primitive": "cube", "size": 0.9}},
                "objects": {"blocks": {"mesh": "@block", "grid": {"count": [100, 100]}, "position": [0, 0, 0.45]}}})
s.add_directional_light("Sun", direction=(-0.4, 0.6, -1))
cam = s.add_camera("Camera", fov=50); cam.frame(direction=(-0.5, 0.8, -0.6), aspect=16 / 9)
extract_render_scene(s, cam)
pr = cProfile.Profile(); pr.enable(); extract_render_scene(s, cam); pr.disable()
st = pstats.Stats(pr); buf = io.StringIO(); st.stream = buf; st.sort_stats("tottime").print_stats(16)
print("\n".join(buf.getvalue().splitlines()[2:30]))
