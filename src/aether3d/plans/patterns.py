"""Placement patterns and per-instance variations of scene plans.

A pattern turns one object definition into ``n`` instances, each with a position (and
for some patterns a rotation and a scale) in its parent's space:

* ``grid`` -- ``count`` ``[nx, ny(, nz)]`` at ``spacing``; x varies fastest. With
  ``center`` the grid is centered on the origin.
* ``array`` -- instance ``i`` at ``i * offset``, rotated by ``i * rotation_step``
  (degrees) and scaled by ``scale_step ** i``.
* ``circle`` -- ``count`` points of ``radius`` around +Z at height ``z``, from
  ``start_angle`` (degrees, counter-clockwise from +X) all the way round, or to
  ``end_angle`` inclusive; ``orient`` turns each instance's front (local -Y)
  ``outward``, ``inward`` or along the ``tangent``.
* ``line`` -- ``count`` points from ``from`` to ``to``, both ends included.
* ``scatter`` -- ``count`` random points in a ``box`` (``min``/``max`` in XY) or a
  ``circle`` (``center``, ``radius``) at height ``z``, at least ``min_distance``
  apart; reproducible for a ``seed``.
* ``instances`` -- an explicit list (or columns) of positions, rotations and scales.
* ``repeat`` -- ``n`` instances at the origin, placed by per-instance expressions.

Randomness is counter based (SplitMix64, as the path tracer's): value ``k`` of a
stream depends only on the seed, the stream's name and ``k``, so plans give the same
scene everywhere and whatever else they contain. Positions use plain IEEE float64
arithmetic; angles go through Python's ``math`` (the platform's libm).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import PlanError
from .model import Generator, Pattern, evaluate

_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_INDEX = 0xD1B54A32D192ED03


def _mix(x: int) -> int:
    """SplitMix64's finalizer of ``x + golden ratio`` (64-bit wrapping arithmetic)."""
    x = (x + _GOLDEN) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


def stream_id(name: str) -> int:
    """A stable 64-bit id for a random stream (not Python's per-process ``hash``)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


class Random:
    """Counter-based uniform numbers in ``[0, 1)`` for one seed and stream."""

    def __init__(self, seed: int, stream: str) -> None:
        self.key = _mix(_mix(seed & _M64) ^ stream_id(stream))

    def uniform(self, k: int) -> float:
        x = _mix(self.key ^ ((k * _INDEX) & _M64))
        return (x >> 11) * 2.0**-53


@dataclass
class Instances:
    """What a pattern makes: per-instance placement and expression variables."""

    count: int
    positions: np.ndarray  # (n, 3) float64, in the parent's space
    rotations: np.ndarray | None = None  # (n, 3) Euler degrees added by the pattern
    scales: np.ndarray | None = None  # (n, 3) factors applied by the pattern
    variables: dict[str, list] = field(default_factory=dict)  # x, y, z, angle
    rows: list[dict[str, Any]] | None = None  # "instances" given as a list
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _vec(value: Any, env: dict, ptr: str) -> tuple[float, float, float]:
    v = evaluate(value, env, ptr)
    if not isinstance(v, tuple) or len(v) != 3:
        raise PlanError("INVALID_VALUE", "expected [x, y, z]", ptr)
    out = tuple(float(c) for c in v)
    if not all(math.isfinite(c) for c in out):
        raise PlanError("INVALID_VALUE", "expected finite numbers", ptr)
    return out  # type: ignore[return-value]


def _num(value: Any, env: dict, ptr: str) -> float:
    v = evaluate(value, env, ptr)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise PlanError("INVALID_VALUE", f"expected a finite number, got {v!r}", ptr)
    return float(v)


def expand(pattern: Pattern | None, env: dict[str, Any], stream: str) -> Instances:
    """The instances of ``pattern`` (one at the origin without a pattern)."""
    if pattern is None:
        return Instances(1, np.zeros((1, 3)))
    kind, n, p, ptr = pattern.kind, pattern.count, pattern.params, pattern.pointer
    if kind == "repeat":
        return Instances(n, np.zeros((n, 3)))
    if kind == "grid":
        nx, ny, nz = p["counts"]
        sx, sy, sz = _vec(p["spacing"], env, f"{ptr}/spacing")
        k = np.arange(n)
        ix, iy, iz = k % nx, (k // nx) % ny, k // (nx * ny)
        positions = np.stack([ix * sx, iy * sy, iz * sz], axis=1).astype(np.float64)
        if p["center"]:
            positions -= np.array([(nx - 1) * sx / 2, (ny - 1) * sy / 2, (nz - 1) * sz / 2])
        return Instances(
            n, positions, variables={"x": ix.tolist(), "y": iy.tolist(), "z": iz.tolist()}
        )
    if kind == "array":
        offset = np.array(_vec(p["offset"], env, f"{ptr}/offset"))
        k = np.arange(n, dtype=np.float64)[:, None]
        out = Instances(n, k * offset)
        if "rotation_step" in p:
            out.rotations = k * np.array(_vec(p["rotation_step"], env, f"{ptr}/rotation_step"))
        if "scale_step" in p:
            step = np.array(_vec(p["scale_step"], env, f"{ptr}/scale_step"))
            if np.any(step == 0):
                raise PlanError("INVALID_VALUE", "scale_step must not be 0", f"{ptr}/scale_step")
            out.scales = np.stack([[float(s) ** i for s in step] for i in range(n)])
            if not np.all(np.isfinite(out.scales)):
                raise PlanError("INVALID_VALUE", "scale_step ** count overflows", ptr)
        return out
    if kind == "circle":
        radius = _num(p["radius"], env, f"{ptr}/radius")
        start = _num(p["start_angle"], env, f"{ptr}/start_angle")
        z = _num(p["z"], env, f"{ptr}/z")
        if p["end_angle"] is None:
            step = 360.0 / n
        else:
            end = _num(p["end_angle"], env, f"{ptr}/end_angle")
            step = (end - start) / (n - 1) if n > 1 else 0.0
        angles = [start + i * step for i in range(n)]
        positions = np.array(
            [
                (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), z)
                for a in angles
            ],
            dtype=np.float64,
        ).reshape(n, 3)
        out = Instances(n, positions, variables={"angle": angles})
        turn = {"outward": 90.0, "inward": -90.0, "tangent": 180.0}.get(p["orient"])
        if turn is not None:
            out.rotations = np.array([(0.0, 0.0, a + turn) for a in angles], dtype=np.float64)
        return out
    if kind == "line":
        a = np.array(_vec(p["from"], env, f"{ptr}/from"))
        b = np.array(_vec(p["to"], env, f"{ptr}/to"))
        t = np.arange(n, dtype=np.float64)[:, None] / (n - 1) if n > 1 else np.zeros((1, 1))
        return Instances(n, a + (b - a) * t)
    if kind == "scatter":
        return _scatter(pattern, env, stream)
    if kind == "instances":
        if "columns" in p:
            cols = p["columns"]
            positions = np.array(
                [_vec(v, env, f"{ptr}/positions/{k}") for k, v in enumerate(cols["positions"])],
                dtype=np.float64,
            ).reshape(n, 3)
            out = Instances(n, positions)
            if "rotations_euler" in cols:
                out.rotations = np.array(cols["rotations_euler"], dtype=np.float64).reshape(n, 3)
            if "scales" in cols:
                out.scales = np.array(cols["scales"], dtype=np.float64).reshape(n, 3)
            return out
        rows = p["rows"]
        positions = np.array(
            [row.get("position", (0.0, 0.0, 0.0)) for row in rows], dtype=np.float64
        ).reshape(n, 3)
        return Instances(n, positions, rows=rows)
    raise AssertionError(kind)  # pragma: no cover


def _scatter(pattern: Pattern, env: dict, stream: str) -> Instances:
    p, n, ptr = pattern.params, pattern.count, pattern.pointer
    rng = Random(p["seed"], f"{stream}#scatter")
    z = _num(p["z"], env, f"{ptr}/z")
    d = float(p["min_distance"])
    kind, a, b = p["region"]
    cell = d / math.sqrt(2.0) if d > 0 else 0.0
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    placed: list[tuple[float, float]] = []
    budget = n * int(p["max_attempts"])
    k = 0
    while len(placed) < n and k < budget:
        u1, u2 = rng.uniform(2 * k), rng.uniform(2 * k + 1)
        k += 1
        if kind == "box":
            x = a[0] + (b[0] - a[0]) * u1
            y = a[1] + (b[1] - a[1]) * u2
        else:
            r = b * math.sqrt(u1)
            t = 2.0 * math.pi * u2
            x, y = a[0] + r * math.cos(t), a[1] + r * math.sin(t)
        if d > 0:
            cx, cy = int(math.floor(x / cell)), int(math.floor(y / cell))
            near = False
            for gx in range(cx - 2, cx + 3):
                for gy in range(cy - 2, cy + 3):
                    for qx, qy in grid.get((gx, gy), ()):
                        if (qx - x) ** 2 + (qy - y) ** 2 < d * d:
                            near = True
                            break
                    if near:
                        break
                if near:
                    break
            if near:
                continue
            grid.setdefault((cx, cy), []).append((x, y))
        placed.append((x, y))
    warnings = []
    if len(placed) < n:
        if not p["allow_fewer"]:
            raise PlanError(
                "PATTERN_UNSATISFIABLE",
                f"scatter placed {len(placed)} of {n} points {d:g} apart in its region "
                f"({budget} attempts)",
                ptr,
                hint="enlarge the region, lower count or min_distance, or set allow_fewer",
                details={"placed": len(placed), "count": n},
            )
        warnings.append(
            {
                "code": "SCATTER_FEWER",
                "message": f"scatter placed {len(placed)} of {n} points",
                "pointer": ptr,
            }
        )
    positions = np.array([(x, y, z) for x, y in placed], dtype=np.float64).reshape(-1, 3)
    return Instances(len(placed), positions, warnings=warnings)


# --------------------------------------------------------------------------- #
# Variations
# --------------------------------------------------------------------------- #


def vary(gen: Generator, field_name: str, n: int, envs: list[dict], stream: str) -> list[Any]:
    """The value of a varied field for each of ``n`` instances."""
    kind = gen.kind
    if kind == "cycle":
        values = gen.values
        return [values[i % len(values)] for i in range(n)]
    if kind == "values":
        return list(gen.values)
    if kind == "choose":
        rng = Random(gen.seed, f"{stream}#{field_name}")
        values = gen.values
        if gen.weights is None:
            return [
                values[min(int(rng.uniform(i) * len(values)), len(values) - 1)] for i in range(n)
            ]
        total = sum(gen.weights)
        cumulative = []
        acc = 0.0
        for w in gen.weights:
            acc += w / total
            cumulative.append(acc)
        out = []
        for i in range(n):
            u = rng.uniform(i)
            index = next((k for k, c in enumerate(cumulative) if u < c), len(values) - 1)
            out.append(values[index])
        return out
    if kind in ("uniform", "linear"):
        lo, hi = gen.low, gen.high
        vector = isinstance(lo, tuple)
        if kind == "uniform":
            rngs = [
                Random(gen.seed, f"{stream}#{field_name}#{c}") for c in range(3 if vector else 1)
            ]

            def t_of(i: int, c: int) -> float:
                return rngs[c].uniform(i)

        else:

            def t_of(i: int, c: int) -> float:
                return i / (n - 1) if n > 1 else 0.0

        if vector:
            return [tuple(lo[c] + (hi[c] - lo[c]) * t_of(i, c) for c in range(3)) for i in range(n)]
        return [lo + (hi - lo) * t_of(i, 0) for i in range(n)]
    if kind == "expr":
        out = []
        for i in range(n):
            value = evaluate(gen.expr, envs[i], gen.pointer)
            if field_name == "visible":
                if not isinstance(value, bool):
                    raise PlanError(
                        "INVALID_VALUE",
                        f"visible must be true or false, got {value!r}",
                        gen.pointer,
                    )
            elif field_name in ("offset", "rotation_euler") or (
                field_name == "scale" and isinstance(value, tuple)
            ):
                if not isinstance(value, tuple) or len(value) != 3:
                    raise PlanError(
                        "INVALID_VALUE",
                        f"{field_name} must be (x, y, z), got {value!r}",
                        gen.pointer,
                    )
                value = tuple(float(c) for c in value)
            elif field_name == "scale":
                value = (float(value),) * 3
            else:
                value = float(value)
            out.append(value)
        return out
    raise AssertionError(kind)  # pragma: no cover
