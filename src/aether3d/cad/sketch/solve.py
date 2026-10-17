"""Solving a sketch definition: values, the solver, branch stability and the report.

``solve_sketch(defn, values)`` returns a :class:`SketchSolution`: the solved geometry (a
new definition with updated coordinates and radii), the status, degrees of freedom,
free entities, redundant and conflicting constraints, measured reference dimensions and
timings. The input definition is never changed.

**Branch stability.** After a solve, the orientation signature of the sketch (the turning
direction at every point shared by two curves, every arc's sweep) is compared with the
one before. When a changed dimension flipped part of the sketch into its mirror image,
the solve is repeated by continuation -- the changed values moved from their previous
values in 2, 4, 8 and 16 steps, each step starting from the last -- which follows the
solution branch continuously. If the flip remains, the result carries a
``BRANCH_CHANGED`` warning naming the corners that turned over.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...errors import CadError
from ..tolerance import Tolerance
from .model import VALUE_TYPES, Index, compile_problem, measure, validate
from .solver import Context, Structure, analyze_structure, solve

TAU = 2.0 * math.pi


@dataclass
class SketchSolution:
    definition: dict
    status: str
    dof: int
    converged: bool
    max_residual: float
    condition: float | None
    free_points: list[str]
    free_radii: list[str]
    redundant: list[dict]
    conflicts: list[dict]
    reference: dict[str, float]
    warnings: list[dict] = field(default_factory=list)
    iterations: int = 0
    seconds: float = 0.0
    structure: dict[str, Any] = field(default_factory=dict)
    tolerance: dict[str, float] = field(default_factory=dict)
    analysis: str = "svd"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dof": self.dof,
            "converged": self.converged,
            "max_residual": self.max_residual,
            "condition": self.condition,
            "free": {"points": self.free_points, "radii": self.free_radii},
            "redundant": self.redundant,
            "conflicts": self.conflicts,
            "reference": self.reference,
            "warnings": self.warnings,
            "iterations": self.iterations,
            "seconds": self.seconds,
            "structure": self.structure,
            "tolerance": self.tolerance,
            "analysis": self.analysis,
        }


def check_values(index: Index, values: Mapping[str, float]) -> None:
    """Driving values must be finite and in range for their constraint type."""
    for k, c in enumerate(index.constraints):
        if c["type"] not in VALUE_TYPES or c.get("driving", True) is False:
            continue
        v = values.get(c["id"])
        loc = f"constraints[{k}]({c['id']}).value"
        if v is None or not math.isfinite(v):
            raise CadError("INVALID_VALUE", f"constraint {c['id']!r} has no finite value", loc)
        t = c["type"]
        if t == "angle":
            if not -TAU < v < TAU:
                raise CadError("INVALID_VALUE", "an angle must be within (-360, 360) degrees", loc)
        elif t in ("radius", "diameter"):
            if v <= 0:
                raise CadError("INVALID_VALUE", f"a {t} must be positive", loc)
        elif t == "distance" and v < 0:
            raise CadError("INVALID_VALUE", "a distance must not be negative", loc)
        elif t == "length" and v <= 0:
            raise CadError(
                "INVALID_VALUE",
                "a length must be positive",
                loc,
                hint="to make two points meet, use a coincident constraint",
            )
        if abs(v) > 1e8:
            raise CadError("VALUE_TOO_LARGE", f"{v:g} exceeds 1e8", loc)


def _points(index: Index, x: np.ndarray, problem) -> tuple[dict, dict]:
    pts = {pid: x[[i, j]] for pid, (i, j) in problem.point_vars.items()}
    radii = {cid: float(x[i]) for cid, i in problem.radius_vars.items()}
    return pts, radii


def signature(index: Index, pts: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Orientation features: turning at shared points, arc sweeps (radians)."""
    out: dict[str, float] = {}
    at_point: dict[str, list[tuple[str, np.ndarray]]] = {}
    for eid in index.order:
        e = index.entities[eid]
        t = e["type"]
        if t == "line":
            s, f = pts[e["start"]], pts[e["end"]]
            at_point.setdefault(e["start"], []).append((eid, f - s))
            at_point.setdefault(e["end"], []).append((eid, s - f))
        elif t == "arc":
            c, s, f = pts[e["center"]], pts[e["start"]], pts[e["end"]]
            a0 = math.atan2(*(s - c)[::-1])
            a1 = math.atan2(*(f - c)[::-1])
            out[f"sweep:{eid}"] = (a1 - a0) % TAU
            # Tangents leaving the endpoints along the arc.
            ds = np.array([-(s - c)[1], (s - c)[0]])
            df = np.array([(f - c)[1], -(f - c)[0]])
            at_point.setdefault(e["start"], []).append((eid, ds))
            at_point.setdefault(e["end"], []).append((eid, df))
    for pid, items in at_point.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ea, da), (eb, db) = items[i], items[j]
                na, nb = float(np.hypot(*da)), float(np.hypot(*db))
                if na > 0 and nb > 0:
                    out[f"turn:{pid}:{ea}:{eb}"] = float(da[0] * db[1] - da[1] * db[0]) / (na * nb)
    return out


def _flips(before: Mapping[str, float], after: Mapping[str, float]) -> list[str]:
    flipped = []
    for key, b in before.items():
        a = after.get(key)
        if a is None:
            continue
        if key.startswith("sweep:"):
            d = abs(a - b)
            if min(d, TAU - d) > math.pi * 0.75 and min(b, TAU - b) > 1e-3:
                flipped.append(key)
        elif abs(b) > 1e-6 and abs(a) > 1e-6 and (a > 0) != (b > 0):
            flipped.append(key)
    return flipped


def solve_sketch(
    defn: Mapping,
    values: Mapping[str, float],
    *,
    previous_values: Mapping[str, float] | None = None,
    deadline: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    structure_cache: dict | None = None,
    location: str = "sketch",
) -> SketchSolution:
    """Solve ``defn`` with evaluated constraint ``values`` (SI). Never raises for an
    unsatisfiable sketch: the solution says ``status="conflicting"`` and lists why."""
    t0 = time.perf_counter()
    index = validate(defn, location)
    check_values(index, values)
    problem = compile_problem(index, values)
    scale = max(problem.scale, float(np.max(np.abs(problem.x0), initial=0.0)))
    tol = Tolerance.for_scale(scale)
    ctx = Context(tol.solve, deadline, should_stop)
    key = _structure_key(index, problem)
    st: Structure | None = None
    if structure_cache is not None:
        st = structure_cache.get(key)
    if st is None:
        st = analyze_structure(problem)
        if structure_cache is not None:
            if len(structure_cache) > 64:
                structure_cache.clear()
            structure_cache[key] = st
    before = signature(index, _points(index, problem.x0, problem)[0])
    result = solve(problem, ctx, st)
    warnings: list[dict] = []
    flipped = _flips(before, signature(index, _points(index, result.x, problem)[0]))
    if flipped and previous_values:
        changed = {
            k: (previous_values[k], v)
            for k, v in values.items()
            if k in previous_values and previous_values[k] != v
        }
        if changed:
            for steps in (2, 4, 8, 16):
                follow = _continuation(index, values, changed, steps, ctx, st, problem)
                if follow is not None and not _flips(
                    before, signature(index, _points(index, follow.x, problem)[0])
                ):
                    result = follow
                    flipped = []
                    break
    if flipped:
        warnings.append(
            {
                "code": "BRANCH_CHANGED",
                "message": "part of the sketch turned over into its mirror image",
                "features": flipped[:20],
            }
        )
    return _report(defn, index, problem, result, values, tol, warnings, time.perf_counter() - t0)


def _structure_key(index: Index, problem) -> tuple:
    return (
        tuple(problem.var_names),
        tuple(tuple(a) for a in problem.adj),
        tuple(problem.eq_constraint.tolist()),
    )


def _continuation(index, values, changed, steps, ctx, st, problem):
    x = problem.x0.copy()
    result = None
    for i in range(1, steps + 1):
        frac = i / steps
        stepped = dict(values)
        for k, (old, new) in changed.items():
            stepped[k] = old + (new - old) * frac
        sub = compile_problem(index, stepped)
        sub.x0 = x
        result = solve(sub, ctx, st, analyze=(i == steps))
        if not result.converged:
            return None
        x = result.x
    return result


def _report(defn, index, problem, result, values, tol, warnings, seconds) -> SketchSolution:
    pts, radii = _points(index, result.x, problem)
    # New definition: solved coordinates and radii.
    defn_entities = []
    for eid in index.order:
        e = dict(index.entities[eid])
        if e["type"] == "point":
            e["at"] = [float(pts[eid][0]), float(pts[eid][1])]
        elif e["type"] in ("circle", "arc"):
            e["radius"] = float(radii[eid])
        defn_entities.append(e)
    names = problem.var_names
    free_vars = sorted({c for comp in result.components for c in comp.free_cols})
    free_points, free_radii = [], []
    for c in free_vars:
        ent, attr = names[c].rsplit(".", 1)
        if attr == "radius":
            free_radii.append(ent)
        elif ent not in free_points:
            free_points.append(ent)
    cids = [c["id"] for c in index.constraints]
    redundant: list[dict] = []
    conflicts: list[dict] = []
    res = result.residuals
    seen_conf: set[tuple] = set()
    for comp in result.components:
        for row, deps in comp.dependent:
            k = int(problem.eq_constraint[row])
            group = [row, *deps]
            group_err = float(np.max(np.abs(res[group]))) if len(res) else 0.0
            cid = cids[k] if k >= 0 else None
            dep_ids = sorted(
                {cids[int(problem.eq_constraint[d])] for d in deps if problem.eq_constraint[d] >= 0}
            )
            if group_err > tol.solve * 10 or not result.converged and group_err > tol.solve:
                key = tuple(sorted({cid, *dep_ids} - {None}))
                if key not in seen_conf:
                    seen_conf.add(key)
                    conflicts.append(
                        {"constraint": cid, "conflicts_with": dep_ids, "residual": group_err}
                    )
            elif cid is not None and all(r["constraint"] != cid for r in redundant):
                redundant.append({"constraint": cid, "depends_on": dep_ids})
    if not result.converged and not conflicts:
        worst = int(np.argmax(np.abs(res))) if len(res) else -1
        k = int(problem.eq_constraint[worst]) if worst >= 0 else -1
        conflicts.append(
            {
                "constraint": cids[k] if k >= 0 else None,
                "conflicts_with": [],
                "residual": result.max_residual,
                "note": "the solver did not converge; no dependency was identified",
            }
        )
    dof = result.dof
    cond = max((c.condition for c in result.components if c.condition), default=None)
    if not problem.m and not problem.n:
        status = "empty"
    elif conflicts:
        status = "conflicting"
    elif redundant:
        status = "redundant"
    elif dof == 0:
        status = "fully_constrained"
    else:
        status = "under_constrained"
    if cond is not None and cond > 1e10:
        warnings.append(
            {
                "code": "ILL_CONDITIONED",
                "message": f"the constraint system is ill-conditioned (condition {cond:.3g}); "
                "small changes of values move the geometry a lot",
            }
        )
    reference = {}
    for c in index.constraints:
        if c["type"] in VALUE_TYPES:
            m = measure(index, pts, radii, c)
            if m is not None:
                reference[c["id"]] = m
    definition = {**dict(defn), "entities": defn_entities}
    return SketchSolution(
        definition=definition,
        status=status,
        dof=dof,
        converged=result.converged,
        max_residual=result.max_residual,
        condition=cond,
        free_points=free_points,
        free_radii=free_radii,
        redundant=redundant,
        conflicts=conflicts,
        reference=reference,
        warnings=warnings,
        iterations=result.iterations,
        seconds=seconds,
        structure=result.structure,
        tolerance=tol.to_dict(),
        analysis="blocks" if any(c.method == "blocks" for c in result.components) else "svd",
    )
