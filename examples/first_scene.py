"""Build a small scene, save it as an .aether project, and load it back.

Run from anywhere:

    python examples/first_scene.py [output.aether]

The default output is ``first_scene.aether`` in the current directory.
"""

import sys

from aether3d import Scene

output = sys.argv[1] if len(sys.argv) > 1 else "first_scene.aether"

# The core API, exactly as in the README.
scene = Scene("First Scene")

cube = scene.add_cube(name="Cube", size=2)
cube.position = (0, 0, 1)

sphere = scene.add_sphere(name="Sphere", radius=1)
sphere.position = (3, 0, 1)

# A ground plane and a small hierarchy: a pivot object with an orbiting moon.
scene.add_plane(name="Ground", size=10)

pivot = scene.add_object(name="Pivot", position=(3, 0, 1))
pivot.rotation_euler = (0, 0, 90)  # degrees, XYZ order
moon = scene.add_sphere(name="Moon", radius=0.25, parent=pivot, position=(2, 0, 0))

scene.save(output)

loaded_scene = Scene.load(output)

# Everything survives the round trip: ids, names, hierarchy and transforms.
loaded_moon = loaded_scene.get(moon.id)
assert loaded_moon.name == "Moon"
assert loaded_moon.parent.name == "Pivot"
assert loaded_moon.mesh == moon.mesh

print(f"Saved {len(scene)} objects to {output} and loaded them back.")
for obj in loaded_scene:
    depth = len(list(obj.iter_ancestors()))
    x, y, z = obj.world_position
    print(f"{'  ' * depth}- {obj.name}: world position ({x:.3g}, {y:.3g}, {z:.3g})")
