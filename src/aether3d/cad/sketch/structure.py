"""Structural analysis of equation systems: matching and the Dulmage-Mendelsohn split.

The incidence of equations (rows) and variables (columns) is a bipartite graph. A
maximum matching (Hopcroft-Karp) gives the canonical Dulmage-Mendelsohn decomposition:

* the **over-determined** part -- rows left unmatched by some maximum matching, and
  everything reachable from them by alternating paths: more equations than unknowns;
* the **under-determined** part -- columns left unmatched, and what alternating paths
  reach from them: more unknowns than equations (degrees of freedom);
* the **well-determined** part -- a perfect matching, split into strongly connected
  blocks (Tarjan) of the "row uses the column matched to row'" digraph. Blocks come out
  in an order in which every block uses only columns of earlier blocks (or of the
  over-determined part), so they can be solved one after another.

The split depends only on the structure (which equation reads which variable), never on
values: it is computed once per sketch structure and reused while dimensions change.
Structure cannot see numerical dependence (two constraints that say the same thing in
different ways); the solver's rank analysis does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


def hopcroft_karp(adj: list[list[int]], ncols: int) -> tuple[list[int], list[int]]:
    """Maximum matching of rows to columns. Returns ``(row_match, col_match)`` with -1 for
    unmatched. ``adj[r]`` lists the columns of row ``r``."""
    nrows = len(adj)
    row_match = [-1] * nrows
    col_match = [-1] * ncols
    inf = 1 << 30
    # Greedy start (cheap and usually most of the matching).
    for r in range(nrows):
        for c in adj[r]:
            if col_match[c] == -1:
                row_match[r] = c
                col_match[c] = r
                break
    while True:
        dist = [inf] * nrows
        queue: deque[int] = deque()
        for r in range(nrows):
            if row_match[r] == -1:
                dist[r] = 0
                queue.append(r)
        found = False
        while queue:
            r = queue.popleft()
            for c in adj[r]:
                r2 = col_match[c]
                if r2 == -1:
                    found = True
                elif dist[r2] == inf:
                    dist[r2] = dist[r] + 1
                    queue.append(r2)
        if not found:
            break
        # Iterative DFS along layered alternating paths.
        it = [0] * nrows
        for root in range(nrows):
            if row_match[root] != -1:
                continue
            stack = [root]
            path_cols: list[int] = []
            while stack:
                r = stack[-1]
                advanced = False
                while it[r] < len(adj[r]):
                    c = adj[r][it[r]]
                    it[r] += 1
                    r2 = col_match[c]
                    if r2 == -1:
                        # Augment along the stack.
                        path_cols.append(c)
                        for rr, cc in zip(stack, path_cols, strict=True):
                            row_match[rr] = cc
                            col_match[cc] = rr
                        stack = []
                        advanced = True
                        break
                    if dist[r2] == dist[r] + 1:
                        path_cols.append(c)
                        stack.append(r2)
                        advanced = True
                        break
                if not advanced:
                    dist[r] = inf
                    stack.pop()
                    if path_cols:
                        path_cols.pop()
    return row_match, col_match


def tarjan_order(succ: list[list[int]]) -> list[list[int]]:
    """Strongly connected components of a digraph, sinks first (iterative Tarjan)."""
    n = len(succ)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    out: list[list[int]] = []
    counter = 0
    for start in range(n):
        if index[start] != -1:
            continue
        work = [(start, 0)]
        index[start] = low[start] = counter
        counter += 1
        stack.append(start)
        on_stack[start] = True
        while work:
            v, i = work[-1]
            if i < len(succ[v]):
                work[-1] = (v, i + 1)
                w = succ[v][i]
                if index[w] == -1:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack[w] = True
                    work.append((w, 0))
                elif on_stack[w]:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    comp.append(w)
                    if w == v:
                        break
                out.append(sorted(comp))
    return out


@dataclass
class Decomposition:
    """The Dulmage-Mendelsohn split of one system (row and column indices)."""

    over_rows: list[int]
    over_cols: list[int]
    blocks: list[tuple[list[int], list[int]]]  # well-determined, in solving order
    under_rows: list[int]
    under_cols: list[int]
    structural_rank: int

    @property
    def structurally_well_constrained(self) -> bool:
        return not self.over_rows and not self.under_cols


def decompose(adj: list[list[int]], ncols: int) -> Decomposition:
    """The Dulmage-Mendelsohn decomposition of rows ``adj`` over ``ncols`` columns."""
    nrows = len(adj)
    row_match, col_match = hopcroft_karp(adj, ncols)
    col_rows: list[list[int]] = [[] for _ in range(ncols)]
    for r, cols in enumerate(adj):
        for c in cols:
            col_rows[c].append(r)
    # Over-determined: alternating paths from unmatched rows.
    over_r = [False] * nrows
    over_c = [False] * ncols
    queue = deque(r for r in range(nrows) if row_match[r] == -1)
    for r in queue:
        over_r[r] = True
    while queue:
        r = queue.popleft()
        for c in adj[r]:
            if not over_c[c]:
                over_c[c] = True
                r2 = col_match[c]
                if r2 != -1 and not over_r[r2]:
                    over_r[r2] = True
                    queue.append(r2)
    # Under-determined: alternating paths from unmatched columns.
    under_r = [False] * nrows
    under_c = [False] * ncols
    queue = deque(c for c in range(ncols) if col_match[c] == -1)
    for c in queue:
        under_c[c] = True
    while queue:
        c = queue.popleft()
        for r in col_rows[c]:
            if not under_r[r] and not over_r[r]:
                under_r[r] = True
                c2 = row_match[r]
                if c2 != -1 and not under_c[c2]:
                    under_c[c2] = True
                    queue.append(c2)
    # Well-determined rest: perfectly matched; blocks of its digraph.
    rest = [r for r in range(nrows) if not over_r[r] and not under_r[r]]
    pos = {r: i for i, r in enumerate(rest)}
    succ: list[list[int]] = []
    for r in rest:
        targets = []
        for c in adj[r]:
            r2 = col_match[c]
            if r2 != -1 and r2 != r and r2 in pos:
                targets.append(pos[r2])
        succ.append(targets)
    blocks = []
    for comp in tarjan_order(succ):
        rows = [rest[i] for i in comp]
        blocks.append((rows, [row_match[r] for r in rows]))
    rank = sum(1 for r in range(nrows) if row_match[r] != -1)
    return Decomposition(
        over_rows=[r for r in range(nrows) if over_r[r]],
        over_cols=[c for c in range(ncols) if over_c[c]],
        blocks=blocks,
        under_rows=[r for r in range(nrows) if under_r[r]],
        under_cols=[c for c in range(ncols) if under_c[c]],
        structural_rank=rank,
    )


def components(adj: list[list[int]], ncols: int) -> list[tuple[list[int], list[int]]]:
    """Connected components of the incidence graph: (rows, cols), each sorted.

    Columns that no row reads form components of their own (free variables)."""
    parent = list(range(ncols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for cols in adj:
        if cols:
            root = find(cols[0])
            for c in cols[1:]:
                rc = find(c)
                if rc != root:
                    parent[rc] = root
    groups: dict[int, tuple[list[int], list[int]]] = {}
    order: list[int] = []
    for c in range(ncols):
        root = find(c)
        if root not in groups:
            groups[root] = ([], [])
            order.append(root)
        groups[root][1].append(c)
    for r, cols in enumerate(adj):
        if cols:
            groups[find(cols[0])][0].append(r)
    return [groups[root] for root in order]
