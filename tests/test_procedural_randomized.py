"""Randomized checks: modifier stacks, transaction sequences and castle parameter sweeps.

Seeds are fixed so failures reproduce; each case reports its seed.
"""

import json
import random

import numpy as np
import pytest
from procedural_helpers import is_solid, same_mesh

from aether3d import Expr, ProceduralError, Scene

try:
    import manifold3d  # noqa: F401

    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False

PRIMITIVE_CHOICES = [
    ("box", {"width": 1.3, "depth": 0.7, "height": 1.1}),
    ("cylinder", {"segments": 7, "radius": 0.6, "depth": 1.5}),
    ("cone", {"segments": 9, "radius_bottom": 0.8, "radius_top": 0.3, "depth": 1.2}),
    ("torus", {"major_segments": 10, "minor_segments": 6}),
    ("uv_sphere", {"segments": 10, "rings": 6, "radius": 0.7}),
]


def poly_volume(poly) -> float:
    """Exact signed volume of the polygons, in float64 (no float32 rounding)."""
    # Fan from each face's first corner: affine maps transform these triangles exactly
    # (a "best diagonal" triangulation may flip on non-planar quads).
    p = poly.positions - poly.positions.mean(axis=0)
    fv = poly.face_vertices
    first = np.repeat(poly.face_offsets[:-1], poly.face_sizes)
    corner = np.arange(poly.corner_count)
    fan = (corner != first) & (poly.next_corner != first)
    a, b, c = p[fv[first[fan]]], p[fv[corner[fan]]], p[fv[poly.next_corner[fan]]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


def random_modifier(rng: random.Random, poly):
    """A modifier that keeps a closed solid closed, with its exact volume factor (or None)."""
    kind = rng.choice(
        ["transform", "array", "mirror", "subdivision", "bevel"]
        + (["boolean"] if HAS_MANIFOLD else [])
    )
    if kind == "transform":
        scale = [rng.choice([-1, 1]) * rng.uniform(0.5, 2) for _ in range(3)]
        rotation = [rng.uniform(-180, 180) for _ in range(3)]
        return (
            kind,
            {"scale": scale, "rotation": rotation, "translation": [rng.uniform(-3, 3)] * 3},
            abs(np.prod(scale)),
        )
    if kind == "array":
        count = rng.randint(1, 4)
        return kind, {"count": count, "relative_offset": [rng.choice([1.25, -1.5]), 0, 0]}, count
    if kind == "mirror":
        axes = rng.choice(["x", "y", "z", "xy", "xyz"])
        # Planes beyond the current bounds: copies never overlap the original.
        origin = (poly.bounds()[1] + rng.uniform(0.5, 3)).tolist()
        return kind, {"axes": axes, "origin": origin, "merge": False}, 2 ** len(axes)
    if kind == "subdivision":
        return kind, {"levels": rng.randint(0, 2)}, None
    if kind == "bevel":
        return kind, {"width": 0.01, "limit": "angle", "angle": 50}, None
    return (
        kind,
        {
            "operation": rng.choice(["union", "difference", "intersection"]),
            "operand": {"geometry": "Cutter"},
            "operand_translation": [rng.uniform(-0.5, 0.5) for _ in range(3)],
        },
        None,
    )


ALLOWED_FAILURES = {
    "BEVEL_WIDTH_TOO_LARGE",
    "BEVEL_UNSUPPORTED_VERTEX",
    "BEVEL_UNSUPPORTED_TOPOLOGY",
    "BEVEL_UNEVEN_WIDTH",
    "PROCEDURAL_LIMIT",
}


@pytest.mark.parametrize("seed", range(40))
def test_random_stacks_keep_solids_valid(seed):
    rng = random.Random(seed)
    scene = Scene()
    name, params = rng.choice(PRIMITIVE_CHOICES)
    geo = scene.add_geometry("Shape", name, **params)
    scene.add_geometry("Cutter", "box", width=0.6, depth=0.6, height=0.6)
    expected = poly_volume(geo.evaluate_polygons())
    for _ in range(rng.randint(1, 4)):
        kind, values, factor = random_modifier(rng, geo.evaluate_polygons())
        before = json.dumps(scene.procedural.info(), default=str)
        try:
            geo.add_modifier(kind, **values)
        except ProceduralError as exc:
            assert exc.code in ALLOWED_FAILURES, (seed, kind, exc)
            info = scene.procedural.info()
            info.pop("cache")
            ref = json.loads(before)
            ref.pop("cache")
            assert json.dumps(info, default=str) == json.dumps(ref, default=str)  # nothing changed
            continue
        poly = geo.evaluate_polygons()
        report = poly.validate()
        if poly.face_count == 0:  # an intersection may legitimately be empty
            assert kind == "boolean"
            break
        assert report.summary["closed"] and report.summary["consistently_oriented"], (seed, kind)
        assert report.summary["edge_manifold"], (seed, kind)
        assert is_solid(poly) or kind == "array" or kind == "boolean", (seed, kind)
        volume = poly_volume(poly)
        assert volume > 0, (seed, kind)
        if factor is not None:
            assert volume == pytest.approx(expected * factor, rel=1e-9), (seed, kind)
        expected = volume
    # Determinism: a fresh evaluation without cache reproduces the same mesh.
    mesh = geo.evaluate()
    scene.procedural.clear_cache()
    assert same_mesh(mesh, geo.evaluate())


OPS = ["set_param", "add", "remove", "move", "toggle", "set_mod", "undo", "redo", "bad"]


@pytest.mark.parametrize("seed", range(25))
def test_random_transaction_sequences(seed, tmp_path):
    rng = random.Random(1000 + seed)
    scene = Scene()
    proc = scene.procedural
    proc.add_parameter("n", 2, min=1, max=4)
    proc.add_parameter("w", 1.0, min=0.2, max=3)
    proc.add_rule("not_too_big", "n * w <= 8")
    geo = scene.add_geometry("G", "box", width=Expr("w"))
    scene.add_object("O", geometry=geo)
    initial = definitions(scene)
    initial_mesh = geo.evaluate()
    depth = 0  # successful changes since the start (for undo bookkeeping)
    for _ in range(30):
        op = rng.choice(OPS)
        before = definitions(scene)
        try:
            if op == "set_param":
                proc.set_parameter(
                    rng.choice(["n", "w"]), rng.choice([0, 1, 2, 3, 4, 5, 0.5, 2.5, -1])
                )
            elif op == "add":
                geo.add_modifier(
                    rng.choice(["array", "mirror", "subdivision", "transform"]),
                    index=rng.randint(0, len(geo.modifiers)) if geo.modifiers else None,
                )
            elif op == "remove" and geo.modifiers:
                geo.remove_modifier(rng.randrange(len(geo.modifiers)))
            elif op == "move" and geo.modifiers:
                geo.move_modifier(
                    rng.randrange(len(geo.modifiers)), rng.randint(-1, len(geo.modifiers))
                )
            elif op == "toggle" and geo.modifiers:
                m = rng.choice(geo.modifiers)
                m.set(enabled=not m.enabled)
            elif op == "set_mod" and geo.modifiers:
                m = rng.choice(geo.modifiers)
                if m.type == "array":
                    m.set(count=rng.choice([Expr("n"), 0, 3, Expr("n * 100000")]))
                elif m.type == "subdivision":
                    m.set(levels=rng.choice([0, 1, 2, 9]))
                else:
                    m.set(enabled=m.enabled)
            elif op == "undo" and depth:
                proc.undo()
                depth -= 1
                continue
            elif op == "redo" and proc.can_redo:
                proc.redo()
                depth += 1
                continue
            elif op == "bad":
                proc.set_parameter("missing", 1)
            else:
                continue
            depth += 1
        except ProceduralError:
            assert definitions(scene) == before, (seed, op)
            continue
        # Every accepted state evaluates; a non-manifold result is always explained by a
        # warning (mirroring geometry that spans the plane is the only way to get one here).
        evaluation = proc.evaluate(geo)
        if not evaluation.polygons.validate().summary["edge_manifold"]:
            assert "MIRROR_OVERLAP" in [d.code for d in evaluation.diagnostics], (seed, op)
    # Persisted state reproduces the same evaluation.
    path = tmp_path / "r.aether"
    scene.save(path)
    loaded = Scene.load(path)
    assert definitions(loaded) == definitions(scene)
    assert same_mesh(loaded.get_by_name("O").mesh, scene.get_by_name("O").mesh)
    for _ in range(depth):  # undo exactly the changes made above (not the setup)
        proc.undo()
    assert definitions(scene) == initial and same_mesh(geo.evaluate(), initial_mesh)


def definitions(scene) -> str:
    info = scene.procedural.info()
    info.pop("cache")
    info.pop("revision")
    return json.dumps(info, sort_keys=True, default=str)


@pytest.mark.skipif(not HAS_MANIFOLD, reason="the castle uses booleans (manifold3d)")
@pytest.mark.parametrize("seed", range(4))
def test_random_castle_parameters(seed):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "examples" / "minecraft_castle.py"
    spec = importlib.util.spec_from_file_location("minecraft_castle", path)
    castle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(castle)
    rng = random.Random(2000 + seed)
    scene = Scene()
    castle.build_castle(scene)
    proc = scene.procedural
    accepted = 0
    for _ in range(6):
        values = {
            "castle_width": round(rng.uniform(14, 60), 2),
            "wall_height": round(rng.uniform(4, 14), 2),
            "wall_thickness": round(rng.uniform(0.6, 2.5), 2),
            "tower_count": rng.choice([4, 6, 8, 10]),
            "tower_radius": round(rng.uniform(1.2, 4), 2),
            "gate_width": round(rng.uniform(1.5, 5), 2),
            "gate_height": round(rng.uniform(2.5, 8), 2),
            "detail_density": rng.randint(0, 3),
        }
        try:
            proc.set_parameters(values)
        except ProceduralError as exc:
            assert exc.code in ("CONSTRAINT_VIOLATED", "PARAMETER_OUT_OF_RANGE"), exc
            continue
        accepted += 1
        castle.place_watchtower(scene)
        assert castle.check_solids(scene) == [], (seed, values)
        assert castle.check_structure(scene) == [], (seed, values)
    assert accepted, "no parameter sample was accepted"
