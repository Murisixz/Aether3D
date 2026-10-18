"""Animation: keyframes and F-curves, rotations, clips and layers, skeletons, skinning,
IK and constraints, cameras and cinematics (see ``docs/ANIMATION.md``)."""

from .curves import EXTRAPOLATIONS, HANDLE_TYPES, INTERPOLATIONS, FCurve
from .time import FrameRate

__all__ = ["EXTRAPOLATIONS", "HANDLE_TYPES", "INTERPOLATIONS", "FCurve", "FrameRate"]
