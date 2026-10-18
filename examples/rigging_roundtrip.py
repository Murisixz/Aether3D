"""Showcase G (Stage 18): a rigged, clothed, animated character through ``.aether`` and
GLB -- skeleton, weights, rest poses and animated poses compared before and after.

The character: the auto-rigged humanoid with a jacket bound by weight transfer and a
helmet attached to the head, a generated walk and a keyed reach (IK) with a look-at. It
is saved as ``.aether`` and loaded back, and exported as GLB (baked animation: what the
scene shows, sampled at the frames) and imported back; each copy is compared with the
original:

* **skeleton** -- joint names, parents, rest matrices, roles, limits;
* **weights** -- every vertex's weights per joint (the GLB's vertices matched to the
  original's by position: a file may reorder or split them);
* **rest pose** -- the meshes' bind (rest) positions and the skins' inverse bind matrices;
* **animated poses** -- joint world matrices and skinned vertex positions at frame times.

``.aether`` keeps everything exactly (it is Aether3D's own format); GLB keeps what glTF can
say (linear blend skinning, node animation, float32) and says what it drops
(``RIG_AUTHORING_DATA``); its differences are measured, not assumed. The GLB is checked
by the Khronos glTF Validator when installed.

Run from the repository root:

    python examples/rigging_roundtrip.py [OUTPUT_DIR] [--quick] [--backend auto|gpu|cpu]

Writes to ``OUTPUT_DIR`` (default ``examples/output/rigging/g_roundtrip``):
``roundtrip.png`` (original | .aether | GLB at three times), ``difference.png`` (the GLB
copy's skin coloured by its distance to the original's, worst frame), ``character.aether``,
``character.glb`` and ``report.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rigging_common as C  # noqa: E402
from animation_common import validate_glb  # noqa: E402

from aether3d import Scene, import_gltf  # noqa: E402
from aether3d.rendering.overlay import animation_overlay  # noqa: E402
from aether3d.rendering.scene_data import world_matrices  # noqa: E402
from aether3d.sculpt.implicit import implicit_surface  # noqa: E402

MESHES = ("Body", "Jacket")


def build(voxel: float) -> Scene:
    scene = Scene("Showcase G")
    C.studio(scene)
    body = scene.add_object("Body", mesh=C.humanoid_mesh(voxel))
    jacket = scene.add_object("Jacket", mesh=C.garment_mesh((0, 2, 3, 4, 9, 14), 0.016, voxel))
    helmet = scene.add_object(
        "Helmet",
        mesh=implicit_surface(
            [
                {"type": "ellipsoid", "center": [0.0, 0.012, 1.665], "radii": [0.118, 0.13, 0.12]},
                {
                    "type": "box",
                    "center": [0.0, -0.06, 1.56],
                    "size": [0.4, 0.3, 0.16],
                    "subtract": True,
                },
            ],
            voxel,
        ),
    )
    body.material = scene.create_material("Skin", base_color=(0.8, 0.62, 0.5), roughness=0.55)
    jacket.material = scene.create_material("Jacket", base_color=(0.16, 0.36, 0.62), roughness=0.6)
    helmet.material = scene.create_material(
        "Steel", base_color=(0.72, 0.74, 0.78), metallic=0.9, roughness=0.28
    )
    a = scene.animation
    a.autorig([body, jacket, helmet], name="Rig", parts={"Jacket": "transfer", "Helmet": "attach"})
    a.set_timeline(start=0.0, end=2.0)
    arm = scene.get_by_name("Rig")
    a.generate(arm, "walk", forward="-y", travel=False)
    a.pose_character(arm, [{"do": "look_at", "target": [0.6, -2.0, 1.7]}], time=1.0)
    return scene


def skeleton(scene: Scene) -> dict:
    a = scene.animation
    arm = next(o for o in scene.objects if a.rig(o) is not None)
    rig = a.rig(arm)
    model = rig.rest_model()
    return {
        j.name: {
            "parent": None if j.parent < 0 else rig.joints[j.parent].name,
            "rest": model[k],
            "role": j.role,
            "limits": j.limits,
            "deform": j.deform,
        }
        for k, j in enumerate(rig.joints)
    } | {"_armature": arm}


def weights(scene: Scene, name: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    a = scene.animation
    obj = scene.get_by_name(name)
    sk = a._owned[("skin", obj._id)]._definition
    rig = a._items[sk.rig]._definition
    names = [rig.joints[rig.index_of_id(j)].name for j in sk.joints]
    return sk.dense(), names, np.asarray(sk.inverse_bind)


def posed(scene: Scene, name: str, t: float) -> np.ndarray:
    obj = scene.get_by_name(name)
    ov = animation_overlay(scene, t)
    mesh = ov.meshes.get(obj.id, obj.mesh) if ov is not None else obj.mesh
    w = world_matrices(scene, ov)[obj.id]
    return mesh.positions.astype(np.float64) @ w[:3, :3].T + w[:3, 3]


def compare(src: Scene, dst: Scene, times, *, exact: bool) -> dict:
    out: dict = {}
    sa, sb = skeleton(src), skeleton(dst)
    names = sorted(k for k in sa if not k.startswith("_"))
    out["joints"] = len(names)
    out["same_joints"] = names == sorted(k for k in sb if not k.startswith("_"))
    out["same_parents"] = all(sa[n]["parent"] == sb[n]["parent"] for n in names)
    out["rest_max_m"] = float(max(np.abs(sa[n]["rest"] - sb[n]["rest"]).max() for n in names))
    out["same_semantics"] = all(
        (sa[n]["role"], sa[n]["limits"], sa[n]["deform"])
        == (sb[n]["role"], sb[n]["limits"], sb[n]["deform"])
        for n in names
    )
    a, b = src.animation, dst.animation
    pj = {}
    for t in times:
        ja = a.pose_joints(sa["_armature"], t, space="world")
        jb = b.pose_joints(sb["_armature"], t, space="world")
        pj[round(t, 4)] = float(max(np.abs(ja[n][:3, 3] - jb[n][:3, 3]).max() for n in names))
    out["posed_joints_max_m"] = pj
    out["meshes"] = {}
    for name in MESHES:
        wa, na, ib_a = weights(src, name)
        wb, nb, ib_b = weights(dst, name)
        pa = src.get_by_name(name).mesh.positions.astype(np.float32)
        pb = dst.get_by_name(name).mesh.positions.astype(np.float32)
        at = {p.tobytes(): i for i, p in enumerate(pa)}
        idx = np.array([at.get(p.tobytes(), -1) for p in pb])
        rec: dict = {"vertices": [len(pa), len(pb)], "unmatched": int((idx < 0).sum())}
        idx = np.where(idx < 0, 0, idx)
        wbb = np.zeros((len(pb), len(na)))
        for col, n in enumerate(nb):
            wbb[:, na.index(n)] += wb[:, col]
        rec["weights_max"] = float(np.abs(wbb - wa[idx]).max())
        ibm = {n: ib_a[i] for i, n in enumerate(na)}
        rec["inverse_bind_max"] = float(
            max(np.abs(ib_b[i] - ibm[n]).max() for i, n in enumerate(nb))
        )
        rec["rest_positions_max_m"] = float(
            np.abs(pb.astype(np.float64) - pa[idx].astype(np.float64)).max()
        )
        diffs = {}
        for t in times:
            d = np.linalg.norm(posed(dst, name, t) - posed(src, name, t)[idx], axis=1)
            diffs[round(t, 4)] = float(d.max())
        rec["posed_max_m"] = diffs
        out["meshes"][name] = rec
    worst = max(max(r["posed_max_m"].values()) for r in out["meshes"].values())
    out["worst_posed_m"] = worst
    if exact:
        out["exact"] = worst == 0.0 and out["rest_max_m"] == 0.0
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", nargs="?", default=str(C.OUT / "g_roundtrip"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--backend", default="auto", choices=("auto", "gpu", "cpu"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = C.set_backend(args.backend)
    t_all = time.perf_counter()
    scene = build(0.02 if args.quick else 0.012)
    report: dict = {"backend": backend, "machine": C.machine()}
    fps = scene.animation.fps.fps
    times = [k / fps for k in (0, 7, 12, 19, 24, 31, 36)]
    print("1. .aether")
    t0 = time.perf_counter()
    scene.save(out / "character.aether")
    loaded = Scene.load(out / "character.aether")
    report["aether"] = {
        "seconds": round(time.perf_counter() - t0, 2),
        "bytes": (out / "character.aether").stat().st_size,
        **compare(scene, loaded, times, exact=True),
    }
    print(f"   exact: {report['aether'].get('exact')}")
    print("2. GLB")
    t0 = time.perf_counter()
    exp = scene.export(out / "character.glb", animation="bake").to_dict()
    imported = import_gltf(out / "character.glb").scene
    report["glb"] = {
        "seconds": round(time.perf_counter() - t0, 2),
        "bytes": exp["byte_length"],
        "export_diagnostics": [
            {"code": d["code"], "message": d["message"]} for d in exp["diagnostics"]
        ],
        "validator": validate_glb(out / "character.glb"),
        **compare(scene, imported, times, exact=False),
    }
    print(
        f"   worst posed difference {report['glb']['worst_posed_m']:.2e} m, "
        f"validator {report['glb']['validator']}"
    )
    print("3. pictures")
    size = (240, 320) if args.quick else (480, 640)
    tiles, labels = [], []
    for t in (times[1], times[3], times[5]):
        for label, s in (("original", scene), (".aether", loaded), ("GLB", imported)):
            cam = next((o for o in s.objects if o.name == "Roundtrip camera"), None)
            if cam is None:
                cam = s.add_camera(
                    "Roundtrip camera", position=(1.4, -3.4, 1.25), look_at=(0.0, 0.0, 0.92), fov=32
                )
            r = s.render(camera=cam, time=t, **C.render_options(width=size[0], height=size[1]))
            tiles.append(r.image)
            labels.append(f"{label} {t:.3f}s")
    C.sheet(tiles, out / "roundtrip.png", labels=labels, columns=3, width=size[0])
    # The GLB copy's skin, coloured by distance to the original (worst time).
    rec = report["glb"]["meshes"]["Body"]
    t_worst = max(rec["posed_max_m"], key=rec["posed_max_m"].get)
    pa = posed(scene, "Body", t_worst)
    pb = posed(imported, "Body", t_worst)
    at = {p.tobytes(): i for i, p in enumerate(scene.get_by_name("Body").mesh.positions)}
    idx = np.array([at.get(p.tobytes(), 0) for p in imported.get_by_name("Body").mesh.positions])
    d = np.linalg.norm(pb - pa[idx], axis=1)
    scale = max(float(d.max()), 1e-9)
    from aether3d.animation.weights import weight_colors

    body = imported.get_by_name("Body")
    from aether3d.mesh import Mesh

    m = body.mesh
    body.mesh = Mesh(m.positions, m.triangles, m.normals, m.uvs, colors=weight_colors(d / scale))
    body.material = imported.create_material("Difference", base_color=(1, 1, 1), roughness=0.7)
    cam = imported.get_by_name("Roundtrip camera")
    r = imported.render(camera=cam, time=t_worst, **C.render_options(width=size[0], height=size[1]))
    C.save(out / "difference.png", r.image)
    report["difference_image"] = {"time": t_worst, "max_m": scale, "colors": "blue 0 .. red max"}
    report["seconds"] = round(time.perf_counter() - t_all, 1)
    C.write_report(out / "report.json", report)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
