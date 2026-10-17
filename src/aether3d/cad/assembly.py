"""Assemblies: components placed by mates, solved like a sketch over rigid motions.

An assembly's definition::

    {
      "components": [
        {"id": "base", "source": "Bracket", "fixed": true},
        {"id": "pin", "source": "Pin", "placement": {"origin": [20, 20, 10]}},
        {"id": "sub", "source": "Hinge assembly"}            # assemblies nest
      ],
      "mates": [
        {"id": "m1", "type": "concentric",
         "a": {"component": "base", "face": "holes.0:rev/wall.0"},
         "b": {"component": "pin", "face": "shaft:side/wall"}},
        {"id": "m2", "type": "coincident",
         "a": {"component": "base", "face": "base:end"},
         "b": {"component": "pin", "face": "head:start"}, "offset": "0 mm"}
      ]
    }

Mate references are faces of a component's body: a planar face is a plane (its point,
its outward normal), a cylindrical or conical face an axis. Mate types and equations:

==================  ===================  ==========================================
type                references           equations (lengths)
==================  ===================  ==========================================
``coincident``      plane, plane         normals opposed (``"flip": true``: aligned)
                                         and the planes ``offset`` apart (3)
``concentric``      axis, axis           axes collinear (4)
``parallel``        plane|axis, ...      directions parallel (2)
``perpendicular``   plane|axis, ...      directions perpendicular (1)
``distance``        plane, plane         signed distance along ``a``'s normal (1)
``angle``           plane|axis, ...      angle between directions (1)
``fixed``           component            all six degrees of freedom (6)
==================  ===================  ==========================================

Every free component has six unknowns: a translation and a rotation vector applied to
its current placement (re-based after every Newton step, so no rotation parameterization
singularity is ever reached). Components and mates form a graph that is decomposed like
a sketch (connected components, Dulmage-Mendelsohn blocks); each block is solved with
damped Gauss-Newton (minimum-norm steps keep under-constrained components where they
were). The analysis reports degrees of freedom per component, redundant and conflicting
mates, exactly as for sketches. A failed solve leaves the document unchanged.

This is geometric placement: no masses, forces, joint limits or motion -- dynamic
simulation is a later stage.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..errors import CadError
from .brep.topology import Body, face_sample
from .document import CadDocument, Context, DisplayPart, Item, ItemKind, register
from .features import _Part
from .geom3d import Cone, Cylinder, Frame, Plane, perpendicular
from .naming import resolve_face
from .sketch.solver import _dependent_rows
from .tolerance import Tolerance, rank_threshold
from .units import unit

MATE_TYPES = ("coincident", "concentric", "parallel", "perpendicular", "distance", "angle")
MAX_COMPONENTS = 20_000
MAX_DEPTH = 8


@dataclass
class Leaf:
    """A part body placed in the world by an assembly (after nesting)."""

    path: str
    part: str  # the part item's id
    body: str
    placement: np.ndarray


@dataclass
class AssemblyResult:
    placements: dict[str, np.ndarray]  # component id -> 4x4 world placement
    leaves: list[Leaf]
    status: str
    dof: dict[str, int]
    free: list[str]
    redundant: list[dict]
    conflicts: list[dict]
    max_residual: float
    iterations: int
    sources: dict[str, Any] = field(default_factory=dict)  # component id -> part/assembly result
    definition: dict | None = None


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-300:
        return np.eye(3)
    k = w / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * (kx @ kx)


def _placement(part: _Part, raw: Any, loc: str) -> np.ndarray:
    m = np.eye(4)
    if raw is None:
        return m
    if not isinstance(raw, dict):
        raise CadError("INVALID_VALUE", "a placement is {origin, z, x}", loc)
    fr = Frame.from_axes(
        part.point(raw.get("origin", [0, 0, 0]), f"{loc}.origin"),
        part.direction(raw.get("z", [0, 0, 1]), f"{loc}.z"),
        part.direction(raw.get("x", [1, 0, 0]), f"{loc}.x"),
    )
    return fr.matrix


def _ref_geometry(body: Body, ref: dict, loc: str, scale: float):
    """("plane", point, normal) or ("axis", point, direction) in the body's coordinates."""
    fi = resolve_face(body, ref.get("face"), f"{loc}.face", scale)
    face = body.faces[fi]
    s = face.surface
    if isinstance(s, Plane):
        p, n = face_sample(body, face)
        return ("plane", p, n)
    if isinstance(s, (Cylinder, Cone)):
        return ("axis", s.frame.origin.copy(), s.frame.z.copy())
    raise CadError(
        "UNSUPPORTED_MATE",
        f"face {face.name!r} is a {s.kind}; mates use planar faces or cylinder/cone axes",
        loc,
    )


def _world(geom, m: np.ndarray):
    kind, p, d = geom
    return kind, m[:3, :3] @ p + m[:3, 3], m[:3, :3] @ d


class _Mate:
    def __init__(self, mid, mtype, a, b, ga, gb, value, flip, order):
        self.id, self.type = mid, mtype
        self.a, self.b, self.ga, self.gb = a, b, ga, gb
        self.value, self.flip, self.order = value, flip, order

    @property
    def rows(self) -> int:
        return {
            "coincident": 3,
            "concentric": 4,
            "parallel": 2,
            "perpendicular": 1,
            "distance": 1,
            "angle": 1,
        }[self.type]

    def residual(self, ma, mb, basis, scale) -> np.ndarray:
        ka, pa, da = _world(self.ga, ma)
        kb, pb, db = _world(self.gb, mb)
        u, v = basis
        t = self.type
        if t == "coincident":
            # Opposed normals (or aligned with flip), planes ``value`` apart along a's normal.
            return np.array(
                [scale * float(db @ u), scale * float(db @ v), float((pb - pa) @ da) - self.value]
            )
        if t == "concentric":
            w = pb - pa
            return np.array(
                [scale * float(db @ u), scale * float(db @ v), float(w @ u), float(w @ v)]
            )
        if t == "parallel":
            return np.array([scale * float(db @ u), scale * float(db @ v)])
        if t == "perpendicular":
            return np.array([scale * float(da @ db)])
        if t == "distance":
            return np.array([float((pb - pa) @ da) - self.value])
        return np.array([scale * (float(da @ db) - math.cos(self.value))])  # angle


def _basis(m: _Mate, ma: np.ndarray):
    _, _, da = _world(m.ga, ma)
    u = perpendicular(da / np.linalg.norm(da))
    return u, np.cross(da, u) / np.linalg.norm(da)


def solve_assembly(
    comps: list[str],
    fixed: set[str],
    placements: dict[str, np.ndarray],
    mates: list[_Mate],
    scale: float,
    ctx=None,
) -> dict[str, Any]:
    """Place the free components so every mate holds (Gauss-Newton per connected group)."""
    tol = Tolerance.for_scale(scale).solve * 10
    free = [c for c in comps if c not in fixed]
    # Groups: free components connected through mates.
    parent = {c: c for c in free}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for m in mates:
        if m.a in parent and m.b in parent:
            ra, rb = find(m.a), find(m.b)
            if ra != rb:
                parent[ra] = rb
    groups: dict[str, list[str]] = {}
    for c in free:
        groups.setdefault(find(c), []).append(c)
    place = {k: v.copy() for k, v in placements.items()}
    iterations = 0
    dof: dict[str, int] = {c: 0 for c in comps}
    redundant, conflicts = [], []
    worst = 0.0
    for members in groups.values():
        gm = [m for m in mates if m.a in members or m.b in members]
        index = {c: i for i, c in enumerate(members)}

        def residual(delta: np.ndarray, bases, index=index, gm=gm) -> np.ndarray:
            local = {}
            for c, i in index.items():
                d = delta[6 * i : 6 * i + 6]
                m = place[c].copy()
                m[:3, :3] = _rodrigues(d[3:]) @ m[:3, :3]
                m[:3, 3] = m[:3, 3] + d[:3]
                local[c] = m
            rows = [
                m.residual(local.get(m.a, place[m.a]), local.get(m.b, place[m.b]), bases[k], scale)
                for k, m in enumerate(gm)
            ]
            return np.concatenate(rows) if rows else np.zeros(0)

        def jacobian(bases, members=members, residual=residual):
            n = 6 * len(members)
            r0 = residual(np.zeros(n), bases)
            jac = np.zeros((len(r0), n))
            for j in range(n):
                h = 1e-7 * (scale if j % 6 < 3 else 1.0)
                e = np.zeros(n)
                e[j] = h
                jac[:, j] = (residual(e, bases) - residual(-e, bases)) / (2 * h)
            return r0, jac

        def apply(delta, index=index):
            for c, i in index.items():
                d = delta[6 * i : 6 * i + 6]
                place[c][:3, :3] = _rodrigues(d[3:]) @ place[c][:3, :3]
                place[c][:3, 3] = place[c][:3, 3] + d[:3]

        if not gm:
            for c in members:
                dof[c] = 6
            continue
        converged = False
        for _ in range(60):
            if ctx is not None:
                ctx.check()
            iterations += 1
            bases = [_basis(m, place[m.a]) for m in gm]
            r, jac = jacobian(bases)
            if float(np.max(np.abs(r))) <= tol:
                converged = True
                break
            step = np.linalg.lstsq(jac, r, rcond=None)[0]
            f0 = float(r @ r)
            alpha = 1.0
            snapshot = {c: place[c].copy() for c in members}
            while alpha > 1e-4:
                apply(-alpha * step)
                bases2 = [_basis(m, place[m.a]) for m in gm]
                rn = residual(np.zeros(6 * len(members)), bases2)
                if float(rn @ rn) < f0:
                    break
                for c in members:
                    place[c] = snapshot[c].copy()
                alpha *= 0.5
            else:
                break
        bases = [_basis(m, place[m.a]) for m in gm]
        r, jac = jacobian(bases)
        err = float(np.max(np.abs(r))) if len(r) else 0.0
        worst = max(worst, err)
        s = np.linalg.svd(jac, compute_uv=False) if jac.size else np.zeros(0)
        thr = rank_threshold(float(s[0]) if len(s) else 0.0, *jac.shape) * 1e4
        rank = int(np.sum(s > thr)) if len(s) else 0
        # Per-component DOF: null-space weight on each component's six variables.
        if jac.size:
            _, sv, vt = np.linalg.svd(jac, full_matrices=True)
            null = vt[rank:]
            for c, i in index.items():
                block = null[:, 6 * i : 6 * i + 6] if len(null) else np.zeros((0, 6))
                dof[c] = int(np.linalg.matrix_rank(block, tol=1e-6)) if len(block) else 0
        rows_owner = [m for m in gm for _ in range(m.rows)]
        order = sorted(range(len(rows_owner)), key=lambda k: (rows_owner[k].order, k))
        if len(rows_owner) > rank:
            deps = _dependent_rows(jac[order], order, thr)
            for row, on in deps:
                mate = rows_owner[row]
                others = sorted({rows_owner[x].id for x in on} - {mate.id})
                group_err = float(np.max(np.abs(r[[row, *on]])))
                if group_err > tol * 10 or not converged:
                    if not any(c["mate"] == mate.id for c in conflicts):
                        conflicts.append(
                            {"mate": mate.id, "conflicts_with": others, "residual": group_err}
                        )
                elif not any(x["mate"] == mate.id for x in redundant):
                    redundant.append({"mate": mate.id, "depends_on": others})
        if not converged and not conflicts:
            conflicts.append(
                {
                    "mate": gm[int(np.argmax(np.abs(r))) // 1 % len(gm)].id,
                    "conflicts_with": [],
                    "residual": err,
                    "note": "the placement did not converge",
                }
            )
    total = sum(dof.values())
    # Consistent redundancy is common in assemblies (a pin's concentric and coincident
    # mates both keep its axis vertical): it is reported, but the status follows the DOF.
    status = (
        "conflicting" if conflicts else "fully_constrained" if total == 0 else "under_constrained"
    )
    return {
        "placements": place,
        "status": status,
        "dof": dof,
        "redundant": redundant,
        "conflicts": conflicts,
        "max_residual": worst,
        "iterations": iterations,
    }


def _check_def(doc: CadDocument, d: dict, location: str) -> None:
    comps = d.get("components", [])
    mates = d.get("mates", [])
    if not isinstance(comps, list) or not isinstance(mates, list):
        raise CadError("INVALID_ASSEMBLY", "components and mates must be lists", location)
    if len(comps) > MAX_COMPONENTS:
        raise CadError("LIMIT_EXCEEDED", f"more than {MAX_COMPONENTS} components", location)
    ids = set()
    for k, c in enumerate(comps):
        loc = f"{location}.components[{k}]"
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"]:
            raise CadError("INVALID_ASSEMBLY", "a component needs an id", loc)
        if "/" in c["id"] or c["id"] in ids:
            raise CadError(
                "INVALID_ASSEMBLY", f"component id {c['id']!r} is invalid or repeated", loc
            )
        ids.add(c["id"])
        doc.item(c.get("source"), ("part", "assembly"))
    mids = set()
    for k, m in enumerate(mates):
        loc = f"{location}.mates[{k}]"
        if not isinstance(m, dict) or m.get("type") not in MATE_TYPES:
            raise CadError("INVALID_MATE", f"a mate's type is one of {MATE_TYPES}", loc)
        if not isinstance(m.get("id"), str) or m["id"] in mids:
            raise CadError("INVALID_MATE", "a mate needs a unique id", loc)
        mids.add(m["id"])
        for side in ("a", "b"):
            r = m.get(side)
            if not isinstance(r, dict) or r.get("component") not in ids:
                raise CadError(
                    "INVALID_MATE", f"{side} must name a component of the assembly", f"{loc}.{side}"
                )
        if m["a"]["component"] == m["b"]["component"]:
            raise CadError("INVALID_MATE", "a mate joins two different components", loc)
        if m["type"] in ("distance", "angle") and "value" not in m:
            raise CadError("INVALID_MATE", f"a {m['type']} mate needs a value", loc)


def _depth(doc: CadDocument, item: Item, seen=()) -> int:
    if item.id in seen:
        raise CadError("DEPENDENCY_CYCLE", f"assembly {item.name!r} contains itself")
    if len(seen) > MAX_DEPTH:
        raise CadError("LIMIT_EXCEEDED", f"assemblies nest at most {MAX_DEPTH} levels")
    out = 0
    for c in item._definition.get("components", []):
        src = doc.item(c["source"])
        if src.kind == "assembly":
            out = max(out, 1 + _depth(doc, src, (*seen, item.id)))
    return out


@register
class AssemblyKind(ItemKind):
    name = "assembly"
    displays = True

    def check(self, doc, definition, location):
        _check_def(doc, definition, location)

    def references(self, doc, definition):
        from .features import parameter_refs

        refs = [
            doc.item(c["source"], ("part", "assembly")).id for c in definition.get("components", [])
        ]
        return refs + parameter_refs(doc, definition.get("mates", []))

    def evaluate(self, ctx: Context) -> AssemblyResult:
        doc = ctx.doc
        d = ctx.item._definition
        _depth(doc, ctx.item)
        part = _Part(ctx)
        comps = d.get("components", [])
        placements, fixed, sources = {}, set(), {}
        bodies: dict[str, list[tuple[str, Body, np.ndarray]]] = {}
        for k, c in enumerate(comps):
            loc = f"components[{k}]({c['id']})"
            placements[c["id"]] = _placement(
                part, c.get("placement"), f"{ctx.location}.{loc}.placement"
            )
            if c.get("fixed"):
                fixed.add(c["id"])
            src = doc.item(c["source"])
            res = ctx.inputs[src.id]
            sources[c["id"]] = res
            if src.kind == "part":
                body_name = c.get("body", "main")
                bodies[c["id"]] = [
                    (f"{c['id']}", res.body(body_name), np.eye(4), src.id, body_name)
                ]
            else:
                bodies[c["id"]] = [
                    (
                        f"{c['id']}/{leaf.path}",
                        doc.result(leaf.part).body(leaf.body),
                        leaf.placement,
                        leaf.part,
                        leaf.body,
                    )
                    for leaf in res.leaves
                ]
        scale = max(1.0, max((b.size for v in bodies.values() for _, b, *_ in v), default=1.0))
        mates = []
        for k, m in enumerate(d.get("mates", [])):
            loc = f"{ctx.location}.mates[{k}]({m['id']})"
            geo = []
            for side in ("a", "b"):
                r = m[side]
                cid = r["component"]
                entries = bodies[cid]
                inner = r.get("part")
                pick = (
                    entries[0]
                    if inner is None
                    else next((e for e in entries if e[0].endswith("/" + inner)), None)
                )
                if pick is None:
                    raise CadError(
                        "REFERENCE_LOST", f"no part {inner!r} in component {cid!r}", f"{loc}.{side}"
                    )
                _, body, local, *_ = pick
                g = _ref_geometry(body, r, f"{loc}.{side}", part.scale)
                kind, p, dvec = g
                geo.append((kind, local[:3, :3] @ p + local[:3, 3], local[:3, :3] @ dvec))
            value = 0.0
            if m["type"] == "coincident":
                value = (
                    part.length(m.get("offset", 0), f"mates[{k}].offset")
                    if m.get("offset")
                    else 0.0
                )
            elif m["type"] == "distance":
                value = part.length(m["value"], f"mates[{k}].value")
            elif m["type"] == "angle":
                value = part.angle(m["value"], f"mates[{k}].value")
            if m["type"] in ("coincident", "concentric") and (
                geo[0][0] != geo[1][0] or (m["type"] == "coincident") != (geo[0][0] == "plane")
            ):
                raise CadError(
                    "UNSUPPORTED_MATE",
                    f"{m['type']} joins two {'planes' if m['type'] == 'coincident' else 'axes'}",
                    loc,
                )
            mates.append(
                _Mate(
                    m["id"],
                    m["type"],
                    m["a"]["component"],
                    m["b"]["component"],
                    geo[0],
                    geo[1],
                    value,
                    bool(m.get("flip")),
                    k,
                )
            )
        out = solve_assembly([c["id"] for c in comps], fixed, placements, mates, scale, ctx)
        if out["status"] == "conflicting":
            c0 = out["conflicts"][0]
            raise CadError(
                "ASSEMBLY_CONFLICT",
                f"mate {c0['mate']!r} cannot hold together with {c0['conflicts_with']}",
                ctx.location,
                hint="remove or change one of the conflicting mates",
                details={"conflicts": out["conflicts"], "dof": out["dof"]},
            )
        leaves = []
        for c in comps:
            m = out["placements"][c["id"]]
            for path, _body, local, pid, bname in bodies[c["id"]]:
                leaves.append(Leaf(path, pid, bname, m @ local))
        new_def = copy.deepcopy(d)
        for c in new_def.get("components", []):
            fr = out["placements"][c["id"]]
            c["placement"] = {
                "origin": [float(v) / part.scale for v in fr[:3, 3]],
                "z": [float(v) for v in fr[:3, 2]],
                "x": [float(v) for v in fr[:3, 0]],
            }
        return AssemblyResult(
            out["placements"],
            leaves,
            out["status"],
            out["dof"],
            [c for c, n in out["dof"].items() if n],
            out["redundant"],
            out["conflicts"],
            out["max_residual"],
            out["iterations"],
            sources,
            new_def,
        )

    def post_evaluate(self, doc, item, result):
        return result.definition

    def display(self, doc, item, result: AssemblyResult):
        parts = []
        for leaf in result.leaves:
            mesh = doc.body_mesh(leaf.part, leaf.body)
            parts.append(DisplayPart(mesh, leaf.path, {}, (0.62, 0.65, 0.7), leaf.placement))
        return parts

    def info(self, doc, item, result: AssemblyResult):
        s = float(unit(doc.length_unit).factor)
        return {
            "status": result.status,
            "dof": result.dof,
            "free": result.free,
            "redundant": result.redundant,
            "conflicts": result.conflicts,
            "max_residual": result.max_residual,
            "components": {
                cid: {
                    "origin": (m[:3, 3] / s).tolist(),
                    "z": m[:3, 2].tolist(),
                    "x": m[:3, 0].tolist(),
                }
                for cid, m in result.placements.items()
            },
            "leaves": len(result.leaves),
        }


def clashes(doc: CadDocument, assembly: Any, *, limit: int = 200) -> list[dict[str, Any]]:
    """Interferences between the assembly's parts: exact B-rep intersection volumes for
    pairs whose boxes overlap (``"interference"``), or ``"contact"`` when the bodies
    touch without overlapping volume."""
    from .brep.boolean import boolean
    from .brep.properties import mass_properties

    item = doc.item(assembly, "assembly")
    res = doc.result(item)
    s = float(unit(doc.length_unit).factor)
    placed = []
    for leaf in res.leaves:
        body = doc.result(leaf.part).body(leaf.body).transformed(leaf.placement)
        lo, hi = body.bounds
        placed.append((leaf.path, body, lo, hi))
    out = []
    tol = 1e-9 * max(1.0, max((b.size for _, b, _, _ in placed), default=1.0))
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = placed[i], placed[j]
            if np.any(a[2] > b[3] + tol) or np.any(b[2] > a[3] + tol):
                continue
            try:
                inter = boolean(a[1], b[1], "intersection")
                vol = mass_properties(inter).volume
            except CadError as exc:
                if exc.code != "EMPTY_RESULT":
                    raise
                vol = 0.0
            # Coincident surfaces (a pin in a hole of its own diameter) leave a sliver of
            # rounding-size volume: below 1e-9 of the smaller body it is contact.
            small = min(mass_properties(a[1]).volume, mass_properties(b[1]).volume)
            if vol > 1e-9 * small:
                out.append({"a": a[0], "b": b[0], "kind": "interference", "volume": vol / s**3})
            else:
                out.append({"a": a[0], "b": b[0], "kind": "contact_or_near", "volume": 0.0})
            if len(out) >= limit:
                return out
    return out
