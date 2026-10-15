"""Scene plans: the format, its checks, and what each part of a plan builds.

Execution semantics (atomicity, history, idempotency, chunks, cancellation, parity with
the per-object API) are in ``test_plans_execution.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aether3d import PlanError, PlanLimits, Scene, create_cube, plan_schema, validate_plan
from aether3d.plans import parse_plan


def plan(**sections) -> dict:
    return {"format": "aether3d.scene-plan", "version": 1, **sections}


def issues_of(doc: dict, scene: Scene | None = None) -> list[dict]:
    result = validate_plan(doc, scene)
    assert not result.ok
    return result.issues


def codes(doc: dict, scene: Scene | None = None) -> set[str]:
    return {i["code"] for i in issues_of(doc, scene)}


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


def test_a_minimal_plan_does_nothing_and_succeeds():
    scene = Scene("Empty")
    result = scene.execute_plan(plan())
    assert result.ok and result.status == "applied" and result.counts["objects"] == 0
    assert len(scene) == 0 and scene.revision == 0 and not scene.history.can_undo


@pytest.mark.parametrize(
    "doc, code, pointer",
    [
        ({"version": 1}, "MISSING_KEY", "/format"),
        ({"format": "aether3d.scene-script", "version": 1}, "INVALID_FORMAT", "/format"),
        ({"format": "aether3d.scene-plan", "version": 2}, "UNSUPPORTED_VERSION", "/version"),
        (plan(objekts={}), "UNKNOWN_KEY", "/objekts"),
        (plan(objects=[]), "INVALID_TYPE", "/objects"),
        (plan(objects={"bad name": {}}), "INVALID_SYMBOL", "/objects/bad name"),
        (plan(objects={"a": {"mesh": "@nope"}}), "UNKNOWN_REFERENCE", "/objects/a/mesh"),
        (plan(objects={"a": {"position": [0, 0]}}), "INVALID_VALUE", "/objects/a/position"),
        (plan(objects={"a": {"position": [0, "x", 0]}}), "INVALID_VALUE", "/objects/a/position/1"),
        (
            plan(objects={"a": {"scale": {"expr": "i +"}}}),
            "INVALID_EXPRESSION",
            "/objects/a/scale/expr",
        ),
        (
            plan(objects={"a": {"scale": {"expr": "big * 2"}}}),
            "UNKNOWN_VARIABLE",
            "/objects/a/scale/expr",
        ),
        (plan(objects={"a": {"color": "#12"}}), "INVALID_VALUE", "/objects/a/color"),
        (plan(objects={"a": {"tags": ["ok", " no"]}}), "INVALID_VALUE", "/objects/a/tags"),
        (plan(objects={"a": {"repeat": 0}}), "OUT_OF_RANGE", "/objects/a/repeat"),
        (
            plan(objects={"a": {"mesh": {"primitive": "cube"}, "light": {}}}),
            "INVALID_VALUE",
            "/objects/a",
        ),
        (
            plan(objects={"a": {"grid": {"count": [2, 2]}, "array": {"count": 2}}}),
            "INVALID_VALUE",
            "/objects/a",
        ),
        (plan(meshes={"m": {"primitive": "teapot"}}), "INVALID_VALUE", "/meshes/m/primitive"),
        (
            plan(meshes={"m": {"primitive": "sphere", "segments": 5000}}),
            "OUT_OF_RANGE",
            "/meshes/m/segments",
        ),
        (plan(operations=[{"op": "explode"}]), "UNKNOWN_OPERATION", "/operations/0/op"),
        (plan(queries=[{"query": "teleport"}]), "INVALID_VALUE", "/queries/0/query"),
        (plan(options={"chunk_size": 1000}), "INVALID_VALUE", "/options/chunk_size"),
    ],
)
def test_problems_have_a_code_and_a_pointer(doc, code, pointer):
    found = issues_of(doc)
    assert any(i["code"] == code and i["pointer"] == pointer for i in found), found
    with pytest.raises(PlanError) as info:
        Scene("Untouched").execute_plan(doc)
    assert info.value.code == found[0]["code"] and info.value.pointer == found[0]["pointer"]


def test_every_problem_is_reported_at_once_with_suggestions():
    doc = plan(
        meshes={"block": {"primitive": "cube", "size": -1}},
        objects={
            "wall": {"mesh": "@blok", "positon": [0, 0, 0]},
            "tower": {"mesh": "@block", "scale": "big"},
        },
        operations=[{"op": "update", "select": "@wall", "set": {}}],
    )
    found = issues_of(doc)
    by_pointer = {i["pointer"]: i for i in found}
    assert by_pointer["/meshes/block/size"]["code"] == "OUT_OF_RANGE"
    assert by_pointer["/objects/wall/positon"]["hint"] == "did you mean 'position'?"
    assert by_pointer["/objects/tower/scale"]["code"] == "INVALID_VALUE"
    assert by_pointer["/operations/0/set"]["operation"] == 0
    assert "did you mean '@block'" in by_pointer["/objects/wall/mesh"]["hint"]
    error = PlanError(**{k: v for k, v in found[0].items() if k != "pointer"}, pointer="/x")
    assert error.to_dict()["pointer"] == "/x"


def test_symbols_are_unique_across_sections_and_references_check_their_section():
    doc = plan(
        materials={"stone": {"base_color": "#888888"}},
        objects={"stone": {}, "wall": {"mesh": "@other"}, "other": {"material": "@wall"}},
    )
    found = issues_of(doc)
    assert any(i["code"] == "DUPLICATE_SYMBOL" and i["pointer"] == "/objects/stone" for i in found)
    assert any(
        i["code"] == "UNKNOWN_REFERENCE" or "names a object, not a mesh" in i["message"]
        for i in found
    )
    wrong = [i for i in found if i["code"] == "INVALID_REFERENCE"]
    assert {i["message"] for i in wrong} >= {
        "'@other' names a object, not a mesh",
        "'@wall' names a object, not a material",
    }


def test_plans_are_plain_json_tuples_are_arrays_and_other_values_are_refused():
    scene = Scene("Tuples")
    scene.execute_plan(plan(objects={"a": {"position": (1, 2, 3)}}))
    assert scene.get_by_name("a").position == (1.0, 2.0, 3.0)
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"b": {"position": np.zeros(3)}}))
    assert info.value.code == "INVALID_JSON"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"b": {"position": [0, float("nan"), 0]}}))
    assert info.value.code == "INVALID_JSON"
    assert scene.execute_plan('{"format": "aether3d.scene-plan", "version": 1}').ok
    with pytest.raises(PlanError) as info:
        scene.execute_plan("{not json")
    assert info.value.code == "INVALID_JSON"


def test_limits_are_checked_before_anything_is_built():
    small = PlanLimits(max_objects=50, max_pattern_count=40, max_plan_bytes=4000)
    scene = Scene("Limits")
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"a": {"repeat": 41}}), limits=small)
    assert info.value.code == "OUT_OF_RANGE"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"a": {"repeat": 30}, "b": {"repeat": 30}}), limits=small)
    assert info.value.code == "LIMIT_EXCEEDED" and "50" in info.value.message
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(description="x" * 3990), limits=small)
    assert info.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(
            plan(
                meshes={"m": {"primitive": "sphere", "segments": 64, "rings": 32}},
                objects={"a": {"mesh": "@m", "repeat": 10}},
            ),
            limits=PlanLimits(max_triangles=10_000),
        )
    assert info.value.code == "LIMIT_EXCEEDED" and "triangles" in info.value.message
    assert len(scene) == 0
    with pytest.raises(PlanError):
        PlanLimits(max_objects=-1)


def test_validation_without_a_scene_counts_objects():
    doc = plan(
        prefabs={"house": {"children": {"walls": {}, "roof": {}}}},
        objects={
            "houses": {"prefab": "@house", "grid": {"count": [3, 4]}},
            "lamp": {"children": {"bulb": {"light": {"type": "point"}}}},
        },
    )
    result = validate_plan(doc)
    assert result.ok and result.status == "validated"
    assert result.estimate["objects"] == 12 * 3 + 2
    assert result.estimate["by_symbol"] == {"houses": 36, "lamp": 2}
    assert result.warnings[0]["code"] == "NOT_BOUND"


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #


def test_identical_mesh_definitions_share_one_mesh():
    scene = Scene("Meshes")
    scene.execute_plan(
        plan(
            meshes={"a": {"primitive": "cube", "size": 2}, "b": {"primitive": "cube", "size": 2}},
            objects={
                "x": {"mesh": "@a"},
                "y": {"mesh": "@b"},
                "z": {"mesh": {"primitive": "cube", "size": 2}, "repeat": 3},
                "w": {"mesh": {"primitive": "cube", "size": 3}},
            },
        )
    )
    meshes = {id(o.mesh) for o in scene if o.name != "w"}
    assert len(meshes) == 1
    assert scene.get_by_name("x").mesh == create_cube(2)  # the same mesh as add_cube makes
    assert scene.get_by_name("w").mesh is not scene.get_by_name("x").mesh


def test_primitive_meshes():
    scene = Scene("Primitives")
    scene.execute_plan(
        plan(
            objects={
                "box": {"mesh": {"primitive": "box", "size": [1, 2, 3]}},
                "box2": {"mesh": {"primitive": "box", "width": 1, "depth": 2, "height": 3}},
                "cyl": {
                    "mesh": {"primitive": "cylinder", "radius": 0.5, "depth": 2, "segments": 8}
                },
                "cone": {"mesh": {"primitive": "cone", "radius_bottom": 1, "segments": 6}},
                "torus": {"mesh": {"primitive": "torus", "major_segments": 8, "minor_segments": 4}},
                "grid": {"mesh": {"primitive": "grid", "x_segments": 2, "y_segments": 3}},
                "ball": {"mesh": {"primitive": "uv_sphere", "segments": 8, "rings": 4}},
                "sphere": {"mesh": {"primitive": "sphere", "radius": 2, "segments": 8, "rings": 4}},
                "plane": {"mesh": {"primitive": "plane", "size": 3}},
            }
        )
    )
    lo, hi = scene.get_by_name("box").mesh.bounds()
    assert np.allclose(np.subtract(hi, lo), (1, 2, 3))
    assert scene.get_by_name("box").mesh == scene.get_by_name("box2").mesh
    assert scene.get_by_name("cyl").mesh.triangle_count == 8 * 2 + 2 * 6
    assert scene.get_by_name("grid").mesh.triangle_count == 12


def test_inline_mesh_data_and_meshes_from_existing_objects():
    scene = Scene("Data")
    original = scene.add_sphere(name="Ball", segments=6, rings=3)
    scene.execute_plan(
        plan(
            meshes={
                "tri": {
                    "data": {
                        "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                        "triangles": [[0, 1, 2]],
                    }
                },
                "ball": {"from_object": "Ball"},
            },
            objects={"t": {"mesh": "@tri"}, "copy": {"mesh": "@ball"}},
        )
    )
    assert scene.get_by_name("t").mesh.triangle_count == 1
    assert scene.get_by_name("copy").mesh is original.mesh
    with pytest.raises(PlanError) as info:
        scene.execute_plan(
            plan(meshes={"bad": {"data": {"positions": [[0, 0, 0]], "triangles": [[0, 1, 2]]}}})
        )
    assert info.value.code == "INVALID_VALUE" and info.value.pointer == "/meshes/bad/data"


def test_materials_presets_and_textures():
    scene = Scene("Materials")
    result = scene.execute_plan(
        plan(
            textures={"dots": {"procedural": {"type": "checker", "scale": 4}}},
            materials={
                "stone": {"preset": "stone", "scale": 2},
                "red": {"base_color": "#c0392b", "roughness": 0.7},
                "flat": {"model": "simple", "base_color": "#3060a0", "shininess": 64},
                "dotted": {"textures": {"base_color": {"texture": "@dots"}}},
            },
            objects={
                "a": {"mesh": {"primitive": "cube"}, "material": "@red"},
                "b": {"mesh": {"primitive": "cube"}, "material": "stone", "position": [2, 0, 0]},
                "c": {"mesh": {"primitive": "cube"}, "material": "@dotted"},
            },
        )
    )
    assert result.counts["materials"] == 4 and result.counts["textures"] == 1
    library = scene.materials
    assert library["red"].params["roughness"] == 0.7
    assert scene.get_by_name("a").material is library["red"]
    assert scene.get_by_name("b").material is library["stone"]  # by name, no "@"
    assert library["flat"].model == "simple"
    assert library["dotted"].texture("base_color").name == "dots"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(materials={"red": {"base_color": "#ffffff"}}))
    assert info.value.code == "NAME_CONFLICT"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(materials={"m2": {"model": "simple", "metallic": 0.5}}))
    assert info.value.code == "OPERATION_FAILED"
    assert info.value.details["error"]["code"] == "PARAMETER_NOT_USED"
    assert info.value.pointer == "/materials/m2"


def test_image_textures_stay_inside_the_base_directory(tmp_path):
    from aether3d.rendering.png import write_png

    write_png(tmp_path / "wood.png", np.full((4, 4, 3), 128, np.uint8))
    outside = tmp_path.parent / "outside.png"
    write_png(outside, np.zeros((2, 2, 3), np.uint8))
    scene = Scene("Images")
    scene.execute_plan(
        plan(
            textures={"wood": {"image": "wood.png"}},
            materials={"m": {"textures": {"base_color": {"texture": "@wood"}}}},
        ),
        base_dir=tmp_path,
    )
    assert scene.materials.texture("wood").file == (tmp_path / "wood.png").absolute()
    for path in ("../outside.png", str(outside), "sub/../../outside.png"):
        with pytest.raises(PlanError) as info:
            scene.execute_plan(plan(textures={"t2": {"image": path}}), base_dir=tmp_path)
        assert info.value.code == "UNSAFE_PATH", path
    link = tmp_path / "link.png"
    link.symlink_to(outside)
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(textures={"t3": {"image": "link.png"}}), base_dir=tmp_path)
    assert info.value.code == "UNSAFE_PATH"


def test_procedural_geometries_are_shared_by_their_objects():
    scene = Scene("Procedural")
    result = scene.execute_plan(
        plan(
            procedural={"parameters": {"height": {"value": 6, "min": 1}}},
            geometries={
                "tower": {
                    "primitive": "cylinder",
                    "params": {"radius": 1, "depth": {"expr": "height"}, "segments": 8},
                    "modifiers": [
                        {
                            "type": "transform",
                            "params": {"translation": {"expr": "(0, 0, height / 2)"}},
                        }
                    ],
                },
                "tall": {"source": "@tower", "overrides": {}},
            },
            objects={"towers": {"geometry": "@tower", "array": {"count": 4, "offset": [5, 0, 0]}}},
        )
    )
    towers = [o for o in scene if o.name.startswith("towers")]
    assert len(towers) == 4 and result.counts["by_kind"] == {"procedural": 4}
    assert len({id(t.mesh) for t in towers}) == 1
    assert towers[0].mesh.bounds()[1][2] == pytest.approx(6)
    scene.procedural.set_parameter("height", 9)  # a live definition
    assert towers[3].mesh.bounds()[1][2] == pytest.approx(9)
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(geometries={"a": {"source": "@b"}, "b": {"source": "@a"}}))
    assert info.value.code == "REFERENCE_CYCLE" and info.value.details["cycle"][0] in "ab"


# --------------------------------------------------------------------------- #
# Objects, hierarchy and prefabs
# --------------------------------------------------------------------------- #


def test_objects_parents_and_children():
    scene = Scene("Hierarchy")
    ground = scene.add_cube(name="Ground")
    scene.execute_plan(
        plan(
            objects={
                "castle": {"parent": "Ground", "position": [0, 0, 1], "tags": ["castle"]},
                "keep": {
                    "parent": "@castle",
                    "mesh": {"primitive": "cube", "size": 4},
                    "children": {"flag": {"mesh": {"primitive": "plane"}, "position": [0, 0, 3]}},
                },
            }
        )
    )
    castle, keep = scene.get_by_name("castle"), scene.get_by_name("keep")
    assert castle.parent is ground and keep.parent is castle
    flag = scene.get_by_name("keep/flag")
    assert flag.parent is keep and flag.world_position == pytest.approx((0, 0, 4))
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"a": {"parent": "@b"}, "b": {"parent": "@a"}}))
    assert info.value.code == "REFERENCE_CYCLE"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"rows": {"repeat": 3}, "x": {"parent": "@rows"}}))
    assert info.value.code == "AMBIGUOUS_REFERENCE"
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"x": {"parent": "Nowhere"}}))
    assert info.value.code == "UNKNOWN_REFERENCE" and info.value.pointer == "/objects/x/parent"


def test_children_are_created_for_every_instance_and_referenced_by_path():
    scene = Scene("Towers")
    result = scene.execute_plan(
        plan(
            objects={
                "towers": {
                    "mesh": {"primitive": "cylinder", "segments": 8},
                    "array": {"count": 3, "offset": [10, 0, 0]},
                    "children": {
                        "crown": {
                            "mesh": {"primitive": "cube", "size": 0.5},
                            "circle": {"count": 6, "radius": 1.2, "z": 1.2},
                        },
                        "flag": {"position": [0, 0, 3]},
                    },
                },
            },
            operations=[
                {"op": "update", "select": "@towers[1]/flag", "set": {"tags": ["middle"]}},
                {
                    "op": "update",
                    "select": {"refs": ["@towers/crown"]},
                    "set": {"color": "#aa0000"},
                },
            ],
        )
    )
    assert result.counts["objects"] == 3 * (1 + 6 + 1)
    assert result.symbols["towers/crown"]["count"] == 18
    assert result.operations[1]["matched"] == 18
    assert scene.get_by_name("towers 1/flag").tags == ("middle",)
    crown = scene.get_by_name("towers 2/crown 0")
    assert crown.parent.name == "towers 2"
    assert crown.world_position == pytest.approx((21.2, 0, 1.2))


def test_prefabs_are_reusable_structures():
    scene = Scene("Village")
    result = scene.execute_plan(
        plan(
            meshes={"block": {"primitive": "cube", "size": 2}},
            prefabs={
                "house": {
                    "tags": ["house"],
                    "metadata": {"kind": "house"},
                    "children": {
                        "walls": {"mesh": "@block", "position": [0, 0, 1]},
                        "roof": {
                            "mesh": {"primitive": "cone", "segments": 4},
                            "position": [0, 0, 3],
                        },
                    },
                }
            },
            objects={
                "houses": {
                    "prefab": "@house",
                    "grid": {"count": [3, 2], "spacing": 5},
                    "name": "House {x}{y}",
                    "metadata": {"street": "Main"},
                },
                "chapel": {"prefab": "@house", "position": [20, 0, 0], "tags": ["chapel"]},
            },
        )
    )
    assert result.counts["objects"] == 7 * 3
    house = scene.get_by_name("House 21")
    assert house.position == (10.0, 5.0, 0.0) and house.tags == ("house",)
    assert house.metadata == {"kind": "house", "street": "Main"}
    assert [c.name for c in house.children] == ["House 21/walls", "House 21/roof"]
    chapel = scene.get_by_name("chapel")
    assert chapel.tags == ("chapel", "house")
    walls = [o for o in scene if o.name.endswith("/walls")]
    assert len({id(w.mesh) for w in walls}) == 1
    with pytest.raises(PlanError) as info:
        scene.execute_plan(
            plan(
                prefabs={
                    "a": {"children": {"x": {"prefab": "@b"}}},
                    "b": {"children": {"y": {"prefab": "@a"}}},
                },
                objects={"thing": {"prefab": "@a"}},
            )
        )
    assert info.value.code == "REFERENCE_CYCLE"
    found = codes(plan(prefabs={"p": {"position": [1, 2, 3]}}))
    assert "INVALID_VALUE" in found


def test_cameras_and_lights():
    scene = Scene("Lights")
    scene.execute_plan(
        plan(
            objects={
                "box": {"mesh": {"primitive": "cube"}},
                "sun": {"light": {"type": "directional", "intensity": 2}, "direction": [0, 0, -1]},
                "lamps": {
                    "light": {"type": "point", "range": 10},
                    "line": {"from": [-5, 0, 3], "to": [5, 0, 3], "count": 3},
                },
                "cam": {
                    "camera": {"fov": 50},
                    "position": [0, -10, 2],
                    "look_at": "@box",
                    "active": True,
                },
                "overview": {
                    "camera": {"projection": "orthographic"},
                    "frame": {"targets": "@box"},
                },
            }
        )
    )
    sun = scene.get_by_name("sun")
    assert sun.light.intensity == 2 and sun.world_forward == pytest.approx((0, 0, -1), abs=1e-12)
    assert [o.position for o in scene if o.name.startswith("lamps")] == [
        (-5.0, 0.0, 3.0),
        (0.0, 0.0, 3.0),
        (5.0, 0.0, 3.0),
    ]
    cam = scene.get_by_name("cam")
    assert scene.active_camera is cam
    assert cam.world_forward == pytest.approx((0, 1, -0.2) / np.linalg.norm((0, 1, -0.2)))
    assert scene.get_by_name("overview").camera.projection == "orthographic"


# --------------------------------------------------------------------------- #
# Patterns and variations
# --------------------------------------------------------------------------- #


def _positions(scene: Scene, prefix: str) -> list[tuple]:
    return [o.position for o in scene if o.name.startswith(prefix)]


def test_grid_array_line_and_repeat_positions():
    scene = Scene("Patterns")
    scene.execute_plan(
        plan(
            parameters={"gap": 2.5},
            objects={
                "g": {
                    "grid": {"count": [3, 2], "spacing": [1, {"expr": "gap"}]},
                    "position": [0, 0, 7],
                },
                "c": {"grid": {"count": [2, 2, 2], "spacing": 2, "center": True}},
                "a": {
                    "array": {
                        "count": 4,
                        "offset": [0, 1, 0],
                        "rotation_step": [0, 0, 30],
                        "scale_step": 2,
                    }
                },
                "l": {"line": {"from": [0, 0, 0], "to": [1, 2, 4], "count": 5}},
                "r": {"repeat": 4, "position": {"expr": "(i * i, n, u)"}},
            },
        )
    )
    assert _positions(scene, "g ") == [
        (0.0, 0.0, 7.0), (1.0, 0.0, 7.0), (2.0, 0.0, 7.0),
        (0.0, 2.5, 7.0), (1.0, 2.5, 7.0), (2.0, 2.5, 7.0),
    ]  # fmt: skip
    assert sorted(_positions(scene, "c ")) == sorted(
        (x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)
    )
    arrays = [o for o in scene if o.name.startswith("a ")]
    assert [o.position for o in arrays] == [(0.0, k * 1.0, 0.0) for k in range(4)]
    assert arrays[3].rotation_euler == pytest.approx((0, 0, 90))
    assert arrays[3].scale == (8.0, 8.0, 8.0)
    assert _positions(scene, "l ")[2] == (0.5, 1.0, 2.0)
    assert _positions(scene, "r ") == [
        (0.0, 4.0, 0.0),
        (1.0, 4.0, 1 / 3),
        (4.0, 4.0, 2 / 3),
        (9.0, 4.0, 1.0),
    ]


@pytest.mark.parametrize("orient, extra", [("outward", 0.0), ("inward", 180.0), ("tangent", 90.0)])
def test_circle_patterns_turn_fronts_outward_inward_or_along(orient, extra):
    scene = Scene("Circle")
    scene.execute_plan(plan(objects={"c": {"circle": {"count": 8, "radius": 3, "orient": orient}}}))
    for k, obj in enumerate(o for o in scene if o.name.startswith("c ")):
        angle = math.radians(45 * k)
        assert obj.position == pytest.approx(
            (3 * math.cos(angle), 3 * math.sin(angle), 0), abs=1e-12
        )
        out = math.radians(45 * k + extra)
        assert obj.world_forward == pytest.approx((math.cos(out), math.sin(out), 0), abs=1e-12)
    scene.execute_plan(
        plan(
            objects={
                "arc": {"circle": {"count": 3, "radius": 1, "start_angle": 0, "end_angle": 90}}
            }
        )
    )
    assert _positions(scene, "arc ")[2] == pytest.approx((0, 1, 0), abs=1e-12)


def test_scatter_is_reproducible_and_keeps_its_distance():
    def scatter(seed: int, **extra) -> list[tuple]:
        scene = Scene("Scatter")
        spec = {
            "count": 60,
            "seed": seed,
            "box": {"min": [0, 0], "max": [30, 20]},
            "min_distance": 1.5,
        }
        spec.update(extra)
        scene.execute_plan(plan(objects={"s": {"scatter": spec}}))
        return _positions(scene, "s ")

    first, again, other = scatter(1), scatter(1), scatter(2)
    assert first == again and first != other and len(first) == 60
    points = np.array(first)[:, :2]
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1) + np.eye(60) * 1e9
    assert distances.min() >= 1.5
    assert np.all((points >= 0) & (points <= [30, 20]))
    with pytest.raises(PlanError) as info:
        scatter(1, count=500, min_distance=3)
    assert info.value.code == "PATTERN_UNSATISFIABLE" and info.value.details["placed"] < 500
    scene = Scene("Fewer")
    result = scene.execute_plan(
        plan(
            objects={
                "s": {
                    "scatter": {
                        "count": 500,
                        "box": {"min": [0, 0], "max": [10, 10]},
                        "min_distance": 3,
                        "allow_fewer": True,
                    }
                }
            }
        )
    )
    assert result.warnings[0]["code"] == "SCATTER_FEWER" and 0 < result.counts["objects"] < 500
    disc = Scene("Disc")
    disc.execute_plan(
        plan(objects={"d": {"scatter": {"count": 200, "circle": {"center": [5, 5], "radius": 2}}}})
    )
    assert all(math.hypot(x - 5, y - 5) <= 2 for x, y, _ in _positions(disc, "d "))


def test_explicit_instances_as_rows_and_as_columns():
    scene = Scene("Instances")
    scene.execute_plan(
        plan(
            materials={"red": {"base_color": "#ff0000"}},
            objects={
                "rows": {
                    "mesh": {"primitive": "cube"},
                    "instances": [
                        {"position": [0, 0, 0], "name": "Left"},
                        {"position": [2, 0, 0], "rotation_euler": [0, 0, 90], "material": "@red"},
                        {"position": [4, 0, 0], "scale": 2, "tags": ["big"], "visible": False},
                    ],
                },
                "cols": {
                    "instances": {"positions": [[0, 5, 0], [1, 5, 0]], "scales": [1, [1, 2, 3]]}
                },
            },
        )
    )
    assert scene.get_by_name("Left").position == (0.0, 0.0, 0.0)
    second = scene.get_by_name("rows 1")
    assert second.material.name == "red" and second.rotation_euler == pytest.approx((0, 0, 90))
    third = scene.get_by_name("rows 2")
    assert third.scale == (2.0, 2.0, 2.0) and third.tags == ("big",) and not third.visible
    assert scene.get_by_name("cols 1").scale == (1.0, 2.0, 3.0)


def test_variations():
    scene = Scene("Vary")
    scene.execute_plan(
        plan(
            meshes={
                "a": {"primitive": "cube"},
                "b": {"primitive": "sphere", "segments": 6, "rings": 3},
            },
            objects={
                "v": {
                    "mesh": "@a",
                    "repeat": 6,
                    "vary": {
                        "color": {"cycle": ["#ff0000", "#00ff00", "#0000ff"]},
                        "scale": {"linear": [1, 2]},
                        "offset": {"expr": "(i, 0, 0)"},
                        "rotation_euler": {"uniform": [[0, 0, 0], [0, 0, 360]], "seed": 4},
                        "mesh": {"choose": ["@a", "@b"], "weights": [1, 0]},
                        "visible": {"expr": "i % 2 == 0"},
                        "tags": {"values": [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]},
                    },
                }
            },
        )
    )
    objs = [o for o in scene if o.name.startswith("v ")]
    assert [o.color for o in objs[:4]] == [(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0), (1.0, 0, 0)]
    assert [o.scale[0] for o in objs] == pytest.approx([1, 1.2, 1.4, 1.6, 1.8, 2])
    assert [o.position[0] for o in objs] == [0, 1, 2, 3, 4, 5]
    angles = [o.rotation_euler[2] % 360 for o in objs]
    assert len(set(np.round(angles, 6))) == 6 and all(0 <= a < 360 for a in angles)
    assert all(o.mesh.triangle_count == 12 for o in objs)  # weight 0 never chooses "b"
    assert [o.visible for o in objs] == [True, False] * 3
    assert objs[5].tags == ("f",)
    again = Scene("Vary again")
    again.execute_plan(
        plan(
            objects={
                "v": {
                    "repeat": 6,
                    "vary": {"rotation_euler": {"uniform": [[0, 0, 0], [0, 0, 360]], "seed": 4}},
                }
            }
        )
    )
    assert [o.rotation for o in again] == [o.rotation for o in objs]  # same seed, same values
    found = issues_of(plan(objects={"v": {"vary": {"color": {"cycle": ["#fff"]}}}}))
    assert found[0]["message"].startswith("vary needs a pattern")


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #


def test_name_templates_and_defaults():
    scene = Scene("Names")
    scene.execute_plan(
        plan(
            objects={
                "wall": {"repeat": 3, "name": "Brick {i:3} of {n}"},
                "row": {"repeat": 2, "name": "Row"},
                "lit": {"repeat": 2, "name": "{{literal}} {i}"},
                "grid": {"grid": {"count": [2, 2]}, "name": "Cell ({x},{y})"},
                "chateau": {"name": "Château ★"},
            }
        )
    )
    names = [o.name for o in scene]
    assert names[:3] == ["Brick 000 of 3", "Brick 001 of 3", "Brick 002 of 3"]
    assert "Row 0" in names and "Row 1" in names
    assert "{literal} 1" in names and "Cell (1,0)" in names and "Château ★" in names
    found = issues_of(plan(objects={"x": {"name": "Bad {i.__class__}"}}))
    assert found[0]["code"] == "INVALID_VALUE" and "unknown placeholder" in found[0]["message"]
    # Symbols are ASCII identifiers (names may be any text).
    assert codes(plan(objects={"Château": {}})) == {"INVALID_SYMBOL"}


def test_name_conflicts_are_errors_or_renames():
    scene = Scene("Conflicts")
    scene.add_cube(name="crate")
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"crate": {}}))
    assert info.value.code == "NAME_CONFLICT" and "already used in the scene" in info.value.message
    with pytest.raises(PlanError) as info:
        scene.execute_plan(plan(objects={"a": {"repeat": 2, "name": "Same {n}"}}))
    assert info.value.code == "NAME_CONFLICT" and "in this plan" in info.value.message
    result = scene.execute_plan(
        plan(
            options={"on_name_conflict": "rename"},
            objects={"crate": {}, "a": {"repeat": 2, "name": "Same {n}"}},
        )
    )
    assert {"crate.001", "Same 2", "Same 2.001"} <= {o.name for o in scene}
    assert result.warnings[0]["code"] == "NAMES_CHANGED"


def test_the_schema_describes_the_format():
    schema = plan_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["format"] == {"const": "aether3d.scene-plan"}
    assert {"objects", "meshes", "materials", "prefabs", "operations", "queries"} <= set(
        schema["properties"]
    )
    schema["properties"].clear()
    assert plan_schema()["properties"]  # a fresh copy every time


def test_parse_plan_exposes_typed_specs():
    parsed = parse_plan(plan(objects={"a": {"repeat": 3}}, label="Typed"))
    assert parsed.label == "Typed" and parsed.objects[0].pattern.count == 3
    assert len(parsed.fingerprint) == 64
