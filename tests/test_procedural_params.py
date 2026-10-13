"""Parameters, safe expressions and rules."""

import math

import pytest

from aether3d import Expr, ProceduralError, Scene
from aether3d.procedural.params import (
    check_type,
    evaluate_expression,
    make_parameter,
    value_from_json,
    value_to_json,
)


def ev(source, **env):
    return evaluate_expression(Expr(source), env)


@pytest.mark.parametrize(
    ("source", "env", "expected"),
    [
        ("1 + 2 * 3", {}, 7),
        ("width / 2", {"width": 5.0}, 2.5),
        ("floor(7 / 2)", {}, 3),
        ("ceil(3.2)", {}, 4),
        ("round(2.5)", {}, 2),
        ("clamp(n, 1, 4)", {"n": 9}, 4),
        ("max(3, min(5, x))", {"x": 4}, 4),
        ("sqrt(16)", {}, 4.0),
        ("cos(60)", {}, 0.5),  # degrees
        ("2 ** 10", {}, 1024),
        ("-x", {"x": 2}, -2),
        ("x % 2 == 0", {"x": 4}, True),
        ("1 < x <= 3", {"x": 3}, True),
        ("x > 1 and not flag", {"x": 2, "flag": False}, True),
        ("a if flag else b", {"a": 1, "b": 2, "flag": False}, 2),
        ("(x, x * 2, 0)", {"x": 1.5}, (1.5, 3.0, 0)),
        ("pi", {}, math.pi),
        ("int(2.9) + float(1)", {}, 3.0),
    ],
)
def test_expressions_evaluate(source, env, expected):
    result = ev(source, **env)
    assert result == pytest.approx(expected) if isinstance(expected, float) else result == expected


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os')",
        "open('x')",
        "x.real",
        "[1, 2]",
        "{'a': 1}",
        "'text'",
        "lambda: 1",
        "(y := 2)",
        "x[0]",
        "f'{x}'",
        "print(1)",
        "max(x, key=abs)",
        "1 @ 2",
        "1 << 2",
        "x is None",
        "1 in (1, 2)",
        "1 +",
        "a" * 2000,
        "+".join(["1"] * 300),
    ],
)
def test_unsafe_or_invalid_expressions_are_rejected(source):
    with pytest.raises(ProceduralError) as info:
        ev(source, x=1)
    assert info.value.code in ("INVALID_EXPRESSION", "UNKNOWN_PARAMETER")


@pytest.mark.parametrize(
    "source", ["(1,) * 1000000000", "((10 ** 64) ** 64) ** 64", "(1, 2) + (3,)", "2 ** 65"]
)
def test_expressions_cannot_exhaust_resources(source):
    # Arithmetic is on numbers only, exponents are bounded and so is integer size.
    with pytest.raises(ProceduralError) as info:
        ev(source)
    assert info.value.code in ("INVALID_EXPRESSION", "EXPRESSION_ERROR")
    assert ev("2 ** 64") == 2**64


def test_expression_runtime_errors_are_reported():
    cases = [
        ("1 / 0", "EXPRESSION_ERROR"),
        ("sqrt(-1)", "EXPRESSION_ERROR"),
        ("undefined + 1", "UNKNOWN_PARAMETER"),
        ("1e308 * 10", "EXPRESSION_ERROR"),
    ]
    for source, code in cases:
        with pytest.raises(ProceduralError) as info:
            ev(source)
        assert info.value.code == code, source


def test_expr_names_and_json():
    e = Expr("a + max(b, 2) * pi")
    assert e.names == frozenset({"a", "b"})
    assert value_to_json({"v": (1, e)}) == {"v": [1, {"expr": "a + max(b, 2) * pi"}]}
    assert value_from_json({"v": [1, {"expr": "a"}]}) == {"v": (1, Expr("a"))}


def test_parameter_types_and_ranges():
    assert make_parameter("w", 2.0).type == "float"
    assert make_parameter("n", 3).type == "int"
    assert make_parameter("b", True).type == "bool"
    assert make_parameter("v", [1, 2, 3]).type == "vec3"
    assert make_parameter("e", "a", choices=["a", "b"]).type == "enum"
    assert make_parameter("s", "text").type == "string"
    assert check_type("float", 2) == 2.0 and isinstance(check_type("float", 2), float)
    assert check_type("vec3", [1, 2, 3]) == (1.0, 2.0, 3.0)
    bad = [
        (dict(name="1x", value=1), "INVALID_PARAMETER"),
        (dict(name="pi", value=1), "INVALID_PARAMETER"),  # reserved
        (dict(name="x", value=5, min=0, max=3), "PARAMETER_OUT_OF_RANGE"),
        (dict(name="x", value=1, min=3, max=0), "INVALID_PARAMETER"),
        (dict(name="x", value="c", choices=["a", "b"]), "INVALID_PARAMETER"),
        (dict(name="x", value=Expr("y")), "INVALID_PARAMETER"),  # expression needs a type
        (dict(name="x", value=1.5, type="int"), "INVALID_PARAMETER"),
        (dict(name="x", value=True, type="float"), "INVALID_PARAMETER"),
        (dict(name="x", value=float("nan")), "INVALID_PARAMETER"),
        (dict(name="x", value=(1, 2), type="vec3"), "INVALID_PARAMETER"),
        (dict(name="x", value=1, type="matrix"), "INVALID_PARAMETER"),
    ]
    for kwargs, code in bad:
        with pytest.raises(ProceduralError) as info:
            make_parameter(**kwargs)
        assert info.value.code == code, kwargs


def test_scene_parameters_resolve_with_dependencies():
    scene = Scene()
    proc = scene.procedural
    proc.add_parameter("width", 10.0, min=1, max=100)
    proc.add_parameter("half", Expr("width / 2"), type="float", max=20)
    proc.add_parameter("count", Expr("floor(half / 2)"), type="int")
    info = proc.info()["parameters"]
    assert (info["half"]["resolved"], info["count"]["resolved"]) == (5.0, 2)
    assert proc.parameter_dependencies() == {"width": [], "half": ["width"], "count": ["half"]}
    # Derived values are range-checked too: half would be 30 > 20.
    with pytest.raises(ProceduralError) as info:
        proc.set_parameter("width", 60.0)
    assert info.value.code == "PARAMETER_OUT_OF_RANGE" and info.value.location == "parameters.half"
    assert proc.parameters["width"].value == 10.0


def test_parameter_cycles_are_rejected():
    proc = Scene().procedural
    proc.add_parameter("a", 1.0)
    proc.add_parameter("b", Expr("a * 2"), type="float")
    with pytest.raises(ProceduralError) as info:
        proc.set_parameter("a", Expr("b + 1"))
    assert info.value.code == "DEPENDENCY_CYCLE"
    assert info.value.details["cycle"][0] == info.value.details["cycle"][-1]
    assert proc.parameters["a"].value == 1.0


def test_rules_reject_invalid_combinations_atomically():
    proc = Scene().procedural
    proc.add_parameter("count", 4, min=2, max=10)
    proc.add_parameter("radius", 1.0)
    proc.add_rule("even", "count % 2 == 0", "count must be even")
    proc.add_rule("fits", "radius * count <= 12")
    with pytest.raises(ProceduralError) as info:
        proc.set_parameter("count", 5)
    assert info.value.code == "CONSTRAINT_VIOLATED"
    assert info.value.message == "count must be even"
    assert info.value.location == "parameters.rules[even]"
    assert info.value.details["values"] == {"count": 5}
    with pytest.raises(ProceduralError) as info, proc.batch():
        proc.set_parameter("count", 8)
        proc.set_parameter("radius", 2.0)
    assert "fits" in info.value.message
    assert (proc.parameters["count"].value, proc.parameters["radius"].value) == (4, 1.0)
    # A rule that does not hold for the current values cannot be added.
    with pytest.raises(ProceduralError):
        proc.add_rule("small", "count < 3")
    assert set(proc.rules) == {"even", "fits"}
    with pytest.raises(ProceduralError) as info:
        proc.add_rule("not_bool", "count + 1")
    assert info.value.code == "INVALID_RULE"
    proc.remove_rule("even")
    proc.set_parameter("count", 5)


def test_geometry_rules_apply_to_overrides():
    scene = Scene()
    base = scene.add_geometry("Base", "box", parameters={"size": 1.0})
    base.set_source(width=Expr("size"))
    base.add_rule("positive", "size > 0.5")
    with pytest.raises(ProceduralError) as info:
        scene.add_geometry("Small", source=base, overrides={"size": 0.1})
    assert info.value.code == "CONSTRAINT_VIOLATED"
    assert [g.name for g in scene.procedural.geometries] == ["Base"]
