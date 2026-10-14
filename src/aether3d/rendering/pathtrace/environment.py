"""The environment: radiance arriving along rays that leave the scene.

Built from ``scene.world`` (:class:`~aether3d.rendering.settings.Environment`):

* ``"ambient"`` -- uniform radiance equal to the world's ambient light
  (``ambient_color x ambient_intensity``, linear). A white Lambertian
  surface open to the whole sky then reflects exactly the ambient term the
  rasterizers add. Camera rays that miss show ``world.background``, as on
  the raster backends.
* ``"uniform"`` -- uniform radiance ``color x intensity``.
* ``"sky"`` -- with ``z`` the up (+Z) component of the unit direction:
  above the horizon ``mix(horizon, zenith, sqrt(z))``, below it
  ``mix(horizon, ground, min(1, 4 |z|))`` (so the ground color is reached
  about 14 degrees below the horizon), times ``intensity``.

With ``"uniform"`` and ``"sky"`` camera rays that miss see the environment
too. The environment is reached by BSDF-sampled rays (it is not sampled as
a light), which suits these smooth, low-contrast environments; a sun is a
directional light with an ``angle``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..color import to_linear


@dataclass(frozen=True)
class EnvironmentModel:
    kind: str
    uniform: np.ndarray  # (3,) linear radiance ("ambient", "uniform")
    background: np.ndarray  # (3,) linear: what camera rays see with "ambient"
    zenith: np.ndarray  # (3,) linear x intensity ("sky")
    horizon: np.ndarray
    ground: np.ndarray

    @classmethod
    def build(cls, env, background: np.ndarray, ambient: np.ndarray) -> EnvironmentModel:
        """From an :class:`~aether3d.rendering.settings.Environment` (``None``: ambient)."""
        if env is None:
            from ..settings import Environment

            env = Environment()
        scale = env.intensity
        return cls(
            env.type,
            ambient.copy() if env.type == "ambient" else to_linear(env.color) * scale,
            background.copy(),
            to_linear(env.zenith_color) * scale,
            to_linear(env.horizon_color) * scale,
            to_linear(env.ground_color) * scale,
        )

    @property
    def black(self) -> bool:
        if self.kind == "sky":
            return not (self.zenith.any() or self.horizon.any() or self.ground.any())
        return not self.uniform.any()

    def radiance(self, d: np.ndarray) -> np.ndarray:
        """``(n, 3)`` radiance arriving from unit directions ``d`` (toward the sky)."""
        if self.kind != "sky":
            return np.broadcast_to(self.uniform, d.shape).copy()
        z = np.clip(d[:, 2], -1.0, 1.0)
        up = np.sqrt(np.maximum(z, 0.0))[:, None]
        down = np.minimum(1.0, 4.0 * np.maximum(-z, 0.0))[:, None]
        sky = self.horizon + (self.zenith - self.horizon) * up
        ground = self.horizon + (self.ground - self.horizon) * down
        return np.where((z >= 0.0)[:, None], sky, ground)

    def camera(self, d: np.ndarray) -> np.ndarray:
        """What camera rays that miss show."""
        if self.kind == "ambient":
            return np.broadcast_to(self.background, d.shape).copy()
        return self.radiance(d)

    def to_dict(self) -> dict:
        data = {"type": self.kind}
        if self.kind == "sky":
            data.update(
                zenith=self.zenith.tolist(),
                horizon=self.horizon.tolist(),
                ground=self.ground.tolist(),
            )
        else:
            data["radiance"] = self.uniform.tolist()
        return data
