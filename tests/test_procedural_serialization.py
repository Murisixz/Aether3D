"""Persistence of procedural content (.aether format version 2) and v1 migration."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from procedural_helpers import same_mesh

from aether3d import (
    FORMAT_VERSION,
    Expr,
    ProceduralError,
    ProjectFormatError,
    Scene,
    __version__,
    create_cube,
    import_gltf,
)
from aether3d.serialization import loads, migrate, scene_from_dict, scene_to_dict

V010 = Path(__file__).resolve().parent / "data" / "v0.1.0" / "first_scene.aether"
try:
    import manifold3d  # noqa: F401

    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False


def build_scene() -> Scene:
    scene = Scene("Persisted")
    proc = scene.procedural
    proc.add_parameter("width", 4.0, min=1, max=20, unit="m", description="block width")
    proc.add_parameter("count", Expr("floor(width / 2)"), type="int", min=1)
    proc.add_parameter("style", "round", choices=["round", "square"])
    proc.add_rule("wide_enough", "width >= 2", "too narrow")
    block = scene.add_geometry("Block", "box", width=Expr("width"), depth=1, height=1)
    block.add_modifier("array", name="Row", count=Expr("count"), relative_offset=(1.5, 0, 0))
    block.add_modifier("bevel", width=0.05, limit="all", enabled=False)
    pillar = scene.add_geometry(
        "Pillar", "cylinder", segments=8, parameters={"h": {"value": 2.0, "min": 0.5}}
    )
    pillar.set_source(depth=Expr("h"))
    pillar.add_rule("short", "h <= 10")
    pillar.add_modifier("transform", translation=Expr("(0, 0, h / 2)"))
    pillar.add_modifier("mirror", axes="x", origin=(3, 0, 0))
    scene.add_geometry("Tall Pillar", source=pillar, overrides={"h": 6.0}, after="Transform")
    cube = create_cube(1)
    imported = scene.add_geometry("Imported", mesh=cube)
    imported.add_modifier("subdivision", levels=1)
    if HAS_MANIFOLD:
        block.add_modifier(
            "boolean",
            operation="difference",
            operand={"geometry": "Pillar", "parameters": {"h": 3.0}},
            operand_translation=(0, 0, -1),
        )
    scene.add_object("Block A", geometry=block)
    scene.add_object("Block B", geometry=block, position=(0, 4, 0))
    scene.add_object("Tall", geometry="Tall Pillar")
    scene.add_object("Imported", geometry=imported)
    scene.add_object("Static", mesh=cube)  # shares the base mesh entry
    return scene


def meshes(scene):
    return {o.name: o.mesh for o in scene}


def test_round_trip_is_exact(tmp_path):
    scene = build_scene()
    path = tmp_path / "p.aether"
    scene.save(path)
    loaded = Scene.load(path)
    assert scene_to_dict(loaded) == scene_to_dict(scene)
    before, after = meshes(scene), meshes(loaded)
    for name in before:
        assert same_mesh(before[name], after[name]), name
    assert loaded.get_by_name("Block A").mesh is loaded.get_by_name("Block B").mesh
    # Definitions keep their ids, so references and handles stay stable.
    assert [g.id for g in loaded.procedural.geometries] == [
        g.id for g in scene.procedural.geometries
    ]
    tall = loaded.procedural.geometry("Tall Pillar")
    assert tall.source.after == scene.procedural.geometry("Pillar").modifier("Transform").id


def test_document_layout(tmp_path):
    scene = build_scene()
    doc = scene_to_dict(scene)
    assert doc["format_version"] == FORMAT_VERSION == 2
    s = doc["scene"]
    assert [p["name"] for p in s["parameters"]] == ["width", "count", "style"]
    assert s["parameters"][1]["value"] == {"expr": "floor(width / 2)"}
    assert s["rules"] == [{"name": "wide_enough", "expr": "width >= 2", "message": "too narrow"}]
    by_name = {g["name"]: g for g in s["geometry"]}
    assert by_name["Block"]["source"]["params"]["width"] == {"expr": "width"}
    assert by_name["Pillar"]["rules"] == [{"name": "short", "expr": "h <= 10"}]
    row = by_name["Block"]["modifiers"][0]
    assert set(row) == {"id", "name", "type", "enabled", "params"}
    assert by_name["Block"]["modifiers"][1]["enabled"] is False
    # Procedural objects store a reference, never an evaluated (possibly stale) mesh.
    objs = {o["name"]: o for o in s["objects"]}
    assert objs["Block A"]["mesh"] is None and objs["Block A"]["geometry"] == by_name["Block"]["id"]
    assert objs["Static"]["geometry"] is None
    # Only the stored base mesh is in the mesh table, shared with the static object.
    assert len(doc["meshes"]) == 1
    assert by_name["Imported"]["source"] == {"type": "mesh", "mesh": 0}
    assert objs["Static"]["mesh"] == 0


def test_exports_use_the_current_evaluation(tmp_path):
    build_scene().save(tmp_path / "p.aether")
    scene = Scene.load(tmp_path / "p.aether")
    scene.procedural.set_parameter("width", 8.0)
    scene.export(tmp_path / "p.glb")
    back = import_gltf(tmp_path / "p.glb").scene
    for obj in scene:
        assert np.array_equal(back.get(obj.id).mesh.positions, obj.mesh.positions), obj.name
    block = back.get_by_name("Block A").mesh
    assert block.positions[:, 0].max() - block.positions[:, 0].min() > 8 * 4  # 4 copies of width 8


def test_version_1_projects_load_and_upgrade(tmp_path):
    scene = Scene.load(V010)
    assert [o.name for o in scene] == ["Cube", "Sphere", "Ground", "Pivot", "Moon"]
    assert scene.procedural.geometries == [] and not any(o.is_procedural for o in scene)
    doc = scene_to_dict(scene)
    assert doc["format_version"] == 2
    assert doc["scene"]["parameters"] == [] and doc["scene"]["geometry"] == []
    assert all(o["geometry"] is None for o in doc["scene"]["objects"])
    v1 = loads(V010.read_text(encoding="utf-8"))
    migrated = migrate(copy.deepcopy(v1))
    assert migrated["format_version"] == 2 and migrated["scene"]["rules"] == []
    scene.save(tmp_path / "up.aether")
    resaved = Scene.load(tmp_path / "up.aether")
    for a, b in zip(scene, resaved, strict=True):
        assert (a.mesh is None and b.mesh is None) or same_mesh(a.mesh, b.mesh)


def corrupt(mutate):
    doc = scene_to_dict(build_scene())
    mutate(doc)
    return doc


def geometry(doc, name):
    return next(g for g in doc["scene"]["geometry"] if g["name"] == name)


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (
            lambda d: geometry(d, "Block")["modifiers"][0].update(type="melt"),
            "unknown modifier type",
        ),
        (lambda d: geometry(d, "Block")["source"].update(primitive="pyramid"), "unknown primitive"),
        (lambda d: geometry(d, "Block")["modifiers"][0]["params"].update(count=0), "minimum"),
        (
            lambda d: geometry(d, "Block")["modifiers"][0]["params"].update(bogus=1),
            "unknown parameter",
        ),
        (lambda d: geometry(d, "Block")["modifiers"][0].update(enabled="yes"), "enabled"),
        (
            lambda d: geometry(d, "Block")["source"]["params"].update(width={"expr": "width +"}),
            "invalid expression",
        ),
        (
            lambda d: geometry(d, "Block")["source"]["params"].update(
                width={"expr": "__import__('os')"}
            ),
            "function",
        ),
        (
            lambda d: geometry(d, "Block")["modifiers"].append(
                dict(geometry(d, "Block")["modifiers"][0])
            ),
            "duplicate modifier",
        ),
        (lambda d: geometry(d, "Pillar").update(name="Block"), "duplicate geometry"),
        (lambda d: geometry(d, "Tall Pillar")["source"].update(geometry="missing"), "missing"),
        (lambda d: geometry(d, "Tall Pillar")["source"].update(after="missing"), "after"),
        (
            lambda d: geometry(d, "Pillar").update(
                source={
                    "type": "geometry",
                    "geometry": geometry(d, "Tall Pillar")["id"],
                    "parameters": {},
                }
            ),
            "cycle",
        ),
        (lambda d: geometry(d, "Imported")["source"].update(mesh=7), "index into meshes"),
        (lambda d: geometry(d, "Block").update(extra=1), "extra"),
        (lambda d: d["scene"]["objects"][0].update(geometry="nope"), "missing geometry"),
        (lambda d: d["scene"]["objects"][0].update(mesh=0), "both a mesh and a geometry"),
        (lambda d: d["scene"]["parameters"][0].update(value=99.0), "maximum"),
        (
            lambda d: d["scene"]["parameters"].append(dict(d["scene"]["parameters"][0])),
            "duplicate parameter",
        ),
        (lambda d: d["scene"]["rules"][0].update(expr="width >= 5"), "too narrow"),
        (lambda d: d["scene"]["rules"][0].update(expr="width >"), "invalid expression"),
        (lambda d: d["scene"].pop("rules"), "rules"),
    ],
)
def test_invalid_procedural_documents_are_rejected(mutate, fragment):
    with pytest.raises(ProjectFormatError) as info:
        scene_from_dict(corrupt(mutate))
    assert fragment.lower() in str(info.value).lower()


def test_loading_does_not_evaluate_and_missing_backend_fails_cleanly(tmp_path, monkeypatch):
    if not HAS_MANIFOLD:
        pytest.skip("needs manifold3d to build the project")
    path = tmp_path / "p.aether"
    build_scene().save(path)
    import builtins

    real_import = builtins.__import__

    def no_manifold(name, *args, **kwargs):
        if name == "manifold3d":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_manifold)
    scene = Scene.load(path)  # definitions load; nothing is evaluated yet
    assert scene.get_by_name("Tall").mesh is not None  # no boolean involved
    with pytest.raises(ProceduralError) as info:
        _ = scene.get_by_name("Block A").mesh
    assert info.value.code == "BOOLEAN_BACKEND_UNAVAILABLE"
    assert info.value.location == "geometry[Block].modifiers[Boolean]"


def test_project_text_is_deterministic(tmp_path):
    build_scene().save(tmp_path / "a.aether")
    a = (tmp_path / "a.aether").read_text(encoding="utf-8")
    doc = json.loads(a)
    Scene.load(tmp_path / "a.aether").save(tmp_path / "b.aether")
    again = (tmp_path / "b.aether").read_text(encoding="utf-8")
    assert a == again and doc["generator"] == f"aether3d {__version__}"
