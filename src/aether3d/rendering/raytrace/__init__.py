"""Ray tracing core: BVH construction and vectorized ray queries (CPU, NumPy).

* :mod:`.bvh` -- binned-SAH bounding volume hierarchies over primitive boxes.
* :mod:`.accel` -- the two-level structure of a render scene (one BVH per
  mesh, cached and shared by instances, plus a BVH over instances) and the
  closest-hit / any-hit traversal with watertight triangle tests.

The path tracer (:mod:`aether3d.rendering.pathtrace`) is built on these.
"""

from .accel import BLAS_CACHE, BLASCache, Hits, SceneAccel, build_mesh_blas
from .bvh import BVH, build_bvh

__all__ = ["BLAS_CACHE", "BLASCache", "BVH", "Hits", "SceneAccel", "build_bvh", "build_mesh_blas"]
