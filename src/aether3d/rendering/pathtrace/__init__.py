"""The path tracer: physically based light transport on the CPU (``render(backend="path")``).

* :mod:`.sampling` -- counter-based random numbers and sample warps.
* :mod:`.bsdf` -- the metallic-roughness BRDF, evaluated and importance sampled.
* :mod:`.lights` -- analytic lights, emissive triangles, next-event estimation.
* :mod:`.environment` -- radiance of rays that leave the scene.
* :mod:`.scene` -- geometry, materials and ray queries (on :mod:`..raytrace`).
* :mod:`.integrator` -- the path tracing loop, passes and accumulation.

See ``docs/RENDER_QUALITY.md`` for the algorithms and their conventions.
"""

from .integrator import PASSES, PathFrame, estimate_work, render_scene

__all__ = ["PASSES", "PathFrame", "estimate_work", "render_scene"]
