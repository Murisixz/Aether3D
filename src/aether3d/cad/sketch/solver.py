"""The sketch solver: decomposition, damped Newton steps, and rank analysis.

``solve(problem)`` returns new variable values and an analysis; it never changes its
input. The steps (see ``research/2026-10-stage15/DESIGN.md`` section 5):

1. **Structure.** Connected components of the equation/variable graph, each split by the
   Dulmage-Mendelsohn decomposition (``structure.py``): the over-determined part, the
   well-determined blocks in solving order, the under-determined part.
2. **Numerics per piece**, in that order, each piece seeing earlier pieces' variables as
   known: Gauss-Newton with *minimum-norm* least-squares steps (``numpy.linalg.lstsq``,
   SVD) and a backtracking line search; Levenberg-Marquardt with Nielsen's damping
   update when a Newton step does not decrease the residual. Minimum-norm steps move an
   under-constrained sketch as little as possible, which keeps it on the branch it was
   on. A piece that does not converge makes the whole component be solved at once from
   its starting values.
3. **Analysis at the result**, per component: numerical rank of the Jacobian by SVD
   (``tolerance.rank_threshold``), degrees of freedom ``n - rank``, the free variables
   (the null space), the condition number, and -- when there are more equations than the
   rank -- which equations depend on earlier ones (rows orthogonalized in constraint
   creation order, so the newest constraint is the one reported), each with the
   constraints it depends on. A dependent equation whose group cannot be satisfied is a
   *conflict*; otherwise it is *redundant*.

Everything is deterministic for the same input on the same platform (fixed orders, no
randomness); results agree across platforms to rounding.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...errors import CadError
from ..tolerance import UNIT_ROUNDOFF, rank_threshold
from .equations import KERNELS
from .model import Problem
from .structure import Decomposition, components, decompose

MAX_ITERATIONS = 60
DENSE_ANALYSIS_LIMIT = 3000
"""Components with more variables are analysed per decomposition block (labeled)."""


@dataclass
class Context:
    """Budgets and cancellation for one solve."""

    tolerance: float
    deadline: float | None = None
    should_stop: Callable[[], bool] | None = None
    iterations: int = 0

    def check(self) -> None:
        if self.should_stop is not None and self.should_stop():
            raise CadError("CANCELLED", "the solve was cancelled; nothing was changed")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise CadError("TIMEOUT", "the solve exceeded its time budget; nothing was changed")


class Evaluator:
    """Residuals and Jacobians of a problem, for all rows or a subset."""

    def __init__(self, problem: Problem) -> None:
        self.p = problem
        m = problem.m
        self.kernel_names = list(problem.kernels)
        self.row_kernel = np.zeros(m, dtype=np.int64)
        self.row_local = np.zeros(m, dtype=np.int64)
        for kid, name in enumerate(self.kernel_names):
            rows = problem.kernels[name][2]
            self.row_kernel[rows] = kid
            self.row_local[rows] = np.arange(len(rows))

    def _groups(self, rows: np.ndarray):
        kids = self.row_kernel[rows]
        for kid in np.unique(kids):
            mask = kids == kid
            yield self.kernel_names[kid], np.flatnonzero(mask), self.row_local[rows[mask]]

    def residual(self, x: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        if rows is None:
            rows = np.arange(self.p.m)
        out = np.empty(len(rows))
        for name, pos, local in self._groups(rows):
            idx, cs, _ = self.p.kernels[name]
            r, _ = KERNELS[name](x[idx[local]], cs[local])
            out[pos] = r
        return out

    def jacobian(self, x: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> tuple:
        """(residuals, dense Jacobian over ``cols``) for ``rows``."""
        col_pos = np.full(self.p.n, -1, dtype=np.int64)
        col_pos[cols] = np.arange(len(cols))
        jac = np.zeros((len(rows), len(cols)))
        res = np.empty(len(rows))
        for name, pos, local in self._groups(rows):
            idx, cs, _ = self.p.kernels[name]
            sub = idx[local]
            r, j = KERNELS[name](x[sub], cs[local])
            res[pos] = r
            cp = col_pos[sub]
            ok = cp >= 0
            rr = np.broadcast_to(pos[:, None], cp.shape)
            np.add.at(jac, (rr[ok], cp[ok]), j[ok])
        return res, jac


# --------------------------------------------------------------------------- #
# Newton iterations on one piece
# --------------------------------------------------------------------------- #


def _norm2(r: np.ndarray) -> float:
    return float(r @ r)


def solve_piece(
    ev: Evaluator, x: np.ndarray, rows: np.ndarray, cols: np.ndarray, ctx: Context
) -> bool:
    """Drive ``rows`` to zero by changing ``cols`` of ``x`` in place; True if converged."""
    if len(rows) == 0 or len(cols) == 0:
        return len(rows) == 0 or float(np.max(np.abs(ev.residual(x, rows)))) <= ctx.tolerance
    tol = ctx.tolerance
    mu = None
    nu = 2.0
    stagnant = 0
    for _ in range(MAX_ITERATIONS):
        ctx.check()
        ctx.iterations += 1
        r, jac = ev.jacobian(x, rows, cols)
        err = float(np.max(np.abs(r)))
        if err <= tol:
            return True
        f0 = _norm2(r)
        accepted = False
        if mu is None:
            step = np.linalg.lstsq(jac, r, rcond=None)[0]
            alpha = 1.0
            while alpha >= 1.0 / 64:
                trial = x.copy()
                trial[cols] -= alpha * step
                rn = ev.residual(trial, rows)
                if _norm2(rn) < f0 * (1.0 - 1e-4 * alpha) or float(np.max(np.abs(rn))) <= tol:
                    x[cols] = trial[cols]
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                jtj = jac.T @ jac
                mu = 1e-3 * max(float(np.max(np.diag(jtj))), 1e-300)
        if not accepted:
            # Levenberg-Marquardt step with Nielsen's damping update.
            jtj = jac.T @ jac
            g = jac.T @ r
            for _ in range(30):
                try:
                    step = np.linalg.solve(jtj + mu * np.eye(len(cols)), g)
                except np.linalg.LinAlgError:
                    mu *= 10
                    continue
                trial = x.copy()
                trial[cols] -= step
                rn = ev.residual(trial, rows)
                f1 = _norm2(rn)
                predicted = float(step @ (mu * step + g))
                rho = (f0 - f1) / predicted if predicted > 0 else -1.0
                if rho > 0:
                    x[cols] = trial[cols]
                    mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                    nu = 2.0
                    accepted = True
                    if f1 > 0.5 * f0:
                        stagnant += 1
                    else:
                        stagnant = 0
                    # Return to Newton steps once LM made good progress.
                    if rho > 0.75:
                        mu = None
                    break
                mu *= nu
                nu *= 2.0
            if not accepted:
                break
            if stagnant >= 6:
                break
    r = ev.residual(x, rows)
    return float(np.max(np.abs(r))) <= tol


# --------------------------------------------------------------------------- #
# Whole solve
# --------------------------------------------------------------------------- #


@dataclass
class ComponentAnalysis:
    rows: list[int]
    cols: list[int]
    rank: int
    dof: int
    condition: float | None
    free_cols: list[int]
    dependent: list[tuple[int, list[int]]]  # (row, rows it depends on)
    method: str  # "svd" or "blocks"


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    residuals: np.ndarray
    max_residual: float
    iterations: int
    components: list[ComponentAnalysis] = field(default_factory=list)
    structure: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def dof(self) -> int:
        return sum(c.dof for c in self.components)


@dataclass
class Structure:
    comps: list[tuple[list[int], list[int]]]
    decomps: list[Decomposition]
    pieces: list[list[tuple[np.ndarray, np.ndarray]]]


def analyze_structure(problem: Problem) -> Structure:
    """Components and their Dulmage-Mendelsohn pieces in solving order."""
    comps = components(problem.adj, problem.n)
    decomps: list[Decomposition] = []
    pieces: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for rows, cols in comps:
        if not rows:
            decomps.append(Decomposition([], [], [], [], list(cols), 0))
            pieces.append([])
            continue
        col_pos = {c: i for i, c in enumerate(cols)}
        local_adj = [[col_pos[c] for c in problem.adj[r]] for r in rows]
        d = decompose(local_adj, len(cols))
        decomps.append(d)
        seq: list[tuple[np.ndarray, np.ndarray]] = []
        if d.over_rows:
            seq.append(
                (
                    np.array([rows[i] for i in d.over_rows]),
                    np.array([cols[i] for i in d.over_cols]),
                )
            )
        for brows, bcols in d.blocks:
            seq.append((np.array([rows[i] for i in brows]), np.array([cols[i] for i in bcols])))
        if d.under_rows or d.under_cols:
            urows = [rows[i] for i in d.under_rows]
            ucols = [cols[i] for i in d.under_cols]
            # The under-determined part splits further once the rest is known.
            upos = {c: i for i, c in enumerate(ucols)}
            uadj = [[upos[c] for c in problem.adj[r] if c in upos] for r in urows]
            for srows, scols in components(uadj, len(ucols)):
                if srows:
                    seq.append(
                        (np.array([urows[i] for i in srows]), np.array([ucols[i] for i in scols]))
                    )
        pieces.append(seq)
    return Structure(comps, decomps, pieces)


def solve(
    problem: Problem,
    ctx: Context,
    structure: Structure | None = None,
    *,
    analyze: bool = True,
) -> SolveResult:
    t0 = time.perf_counter()
    ev = Evaluator(problem)
    st = structure or analyze_structure(problem)
    x = problem.x0.copy()
    converged = True
    for (rows, cols), seq in zip(st.comps, st.pieces, strict=True):
        if not rows:
            continue
        ok = True
        for prow, pcol in seq:
            if not solve_piece(ev, x, prow, pcol, ctx):
                ok = False
                break
        if not ok:
            # Solve the component as a whole from its starting values.
            ca = np.array(cols)
            x[ca] = problem.x0[ca]
            ok = solve_piece(ev, x, np.array(rows), ca, ctx)
        converged &= ok
    res = ev.residual(x) if problem.m else np.zeros(0)
    result = SolveResult(
        x=x,
        converged=bool(converged),
        residuals=res,
        max_residual=float(np.max(np.abs(res))) if len(res) else 0.0,
        iterations=ctx.iterations,
        structure={
            "components": len(st.comps),
            "blocks": sum(len(d.blocks) for d in st.decomps),
            "largest_block": max((len(b[0]) for d in st.decomps for b in d.blocks), default=0),
        },
    )
    if analyze:
        for (rows, cols), d in zip(st.comps, st.decomps, strict=True):
            result.components.append(_analyze(ev, x, rows, cols, d, ctx, problem))
    result.seconds = time.perf_counter() - t0
    return result


def _analyze(
    ev: Evaluator,
    x: np.ndarray,
    rows: list[int],
    cols: list[int],
    d: Decomposition,
    ctx: Context,
    problem: Problem,
) -> ComponentAnalysis:
    n = len(cols)
    if not rows:
        return ComponentAnalysis([], list(cols), 0, n, None, list(cols), [], "svd")
    ctx.check()
    rows_sorted = sorted(rows, key=lambda r: (int(problem.eq_constraint[r]), r))
    if n > DENSE_ANALYSIS_LIMIT:
        return _analyze_blocks(ev, x, rows, cols, d, problem)
    _, jac = ev.jacobian(x, np.array(rows_sorted), np.array(cols))
    u, s, vt = np.linalg.svd(jac, full_matrices=True)
    thresh = rank_threshold(float(s[0]) if len(s) else 0.0, len(rows), n)
    rank = int(np.sum(s > thresh)) if len(s) and s[0] > 0 else 0
    cond = float(s[0] / s[rank - 1]) if rank > 0 else None
    free = []
    if rank < n:
        null = vt[rank:]  # (dof, n)
        weight = np.linalg.norm(null, axis=0)
        free = [cols[i] for i in np.flatnonzero(weight > 1e-6)]
    dependent: list[tuple[int, list[int]]] = []
    if len(rows) > rank:
        dependent = _dependent_rows(jac, rows_sorted, thresh)
    return ComponentAnalysis(rows_sorted, list(cols), rank, n - rank, cond, free, dependent, "svd")


def _dependent_rows(jac: np.ndarray, rows: list[int], thresh: float) -> list[tuple[int, list[int]]]:
    """Rows that lie in the span of earlier rows (modified Gram-Schmidt, twice)."""
    basis: list[np.ndarray] = []
    kept: list[int] = []
    out: list[tuple[int, list[int]]] = []
    scale_tol = max(thresh, 1e3 * UNIT_ROUNDOFF)
    for i, r in enumerate(rows):
        v = jac[i].copy()
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            out.append((r, []))
            continue
        w = v.copy()
        for _ in range(2):
            for q in basis:
                w -= (q @ w) * q
        if float(np.linalg.norm(w)) <= scale_tol * max(1.0, norm):
            a = jac[kept]
            coef = np.linalg.lstsq(a.T, v, rcond=None)[0]
            contrib = np.abs(coef) * np.linalg.norm(a, axis=1)
            deps = [
                rows[kept_i] for kept_i, c in zip(kept, contrib, strict=True) if c > 1e-6 * norm
            ]
            out.append((r, deps))
        else:
            basis.append(w / np.linalg.norm(w))
            kept.append(i)
    return out


def _analyze_blocks(ev, x, rows, cols, d: Decomposition, problem) -> ComponentAnalysis:
    """Large components: structural degrees of freedom plus per-block numerical checks."""
    rank = d.structural_rank
    under = {cols[i] for i in d.under_cols}
    dependent = []
    if d.over_rows:
        local_rows = sorted(
            (rows[i] for i in d.over_rows), key=lambda r: (int(problem.eq_constraint[r]), r)
        )
        local_cols = [cols[i] for i in d.over_cols]
        _, jac = ev.jacobian(x, np.array(local_rows), np.array(local_cols))
        s = np.linalg.svd(jac, compute_uv=False)
        thresh = rank_threshold(float(s[0]) if len(s) else 0.0, *jac.shape)
        dependent = _dependent_rows(jac, local_rows, thresh)
    rows_sorted = sorted(rows, key=lambda r: (int(problem.eq_constraint[r]), r))
    return ComponentAnalysis(
        rows_sorted, list(cols), rank, len(cols) - rank, None, sorted(under), dependent, "blocks"
    )
