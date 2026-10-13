"""Scene builders shared by the interchange tests."""

import numpy as np

from aether3d import Mesh, Scene


def challenging_scene() -> Scene:
    """Nested, rotated, non-uniformly scaled, mirrored and instanced objects."""
    scene = Scene("Challenging")
    cube = scene.add_cube(name="Cube", size=2, position=(0, 0, 1))
    cube.rotation_euler = (10, 20, 30)

    pivot = scene.add_object(name="Pivot", position=(3, 1, -2), scale=(1, 2, 3))
    pivot.rotation_euler = (0, 45, 90)
    moon = scene.add_sphere(name="Moon", radius=0.5, segments=12, rings=6, parent=pivot)
    moon.position = (1, 0, 0)
    mirrored = scene.add_object(
        name="Mirrored Moon", mesh=moon.mesh, parent=pivot, position=(-1, 0, 0), scale=(-1, 1, 1)
    )
    mirrored.rotation_euler = (90, 0, 0)

    arm = scene.add_object(name="Arm", parent=cube, position=(0, 0, 2), scale=(0.5, 1, 2))
    arm.rotation_euler = (-30, 60, 15)
    hand = scene.add_cube(name="Hand", size=0.25, parent=arm, position=(1, 2, 3))
    hand.rotation_euler = (5, -170, 45)
    scene.add_object(name="Hand Twin", mesh=hand.mesh, parent=arm, position=(-1, -2, -3))

    scene.add_plane(name="Ground", size=10)
    scene.add_object(name="Empty Group")
    custom = Mesh(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.75, 0.5)],
        [(0, 1, 2), (1, 3, 2)],
        normals=[(0, 0, 1)] * 4,
        uvs=[(0.1, 0.2), (0.9, 0.2), (0.1, 0.8), (0.33, 0.66)],
    )
    scene.add_object(name="Custom", mesh=custom, position=(-4, 2, 0.5), scale=(2, 1, 0.5))
    return scene


def world_vertices(obj) -> np.ndarray:
    """World-space vertex positions (float64) of an object's mesh."""
    m = obj.world_matrix
    return obj.mesh.positions.astype(np.float64) @ m[:3, :3].T + m[:3, 3]
