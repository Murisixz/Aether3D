"""Observations: per-object pixel facts, geometry in a view, occlusion, image statistics,
images, the store (Stage 12).

Every measured value is checked against an independent computation: brute force over the
index buffer, analytic projections and depths of known scenes, renders of the object alone.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from aether3d import FeedbackError, Scene
from aether3d.feedback import ObservationLimits, ObservationStore, observe
from aether3d.feedback.observe import pixel_statistics

Image = pytest.importorskip("PIL.Image")


def _png(data: bytes):
    return Image.open(io.BytesIO(data))


def test_pixel_statistics_match_brute_force():
    rng = np.random.default_rng(7)
    for trial in range(5):
        h, w, n = 23 + trial, 41, 6
        index = rng.integers(-1, n, size=(h, w)).astype(np.int32)
        index[:, : 3 * trial] = -1
        depth = rng.random((h, w)).astype(np.float32) * 10
        depth[index < 0] = np.inf
        lum = rng.random((h, w))
        s = pixel_statistics(index, depth, lum, n + 1)  # one instance never appears
        for k in range(n + 1):
            ys, xs = np.nonzero(index == k)
            assert s.pixels[k] == ys.size
            if not ys.size:
                assert math.isnan(s.cx[k]) and s.x0[k] == w and s.x1[k] == 0
                continue
            assert (s.x0[k], s.y0[k], s.x1[k], s.y1[k]) == (
                xs.min(),
                ys.min(),
                xs.max() + 1,
                ys.max() + 1,
            )
            assert s.cx[k] == pytest.approx(xs.mean() + 0.5)
            assert s.cy[k] == pytest.approx(ys.mean() + 0.5)
            assert s.dmin[k] == pytest.approx(depth[index == k].min())
            assert s.dmax[k] == pytest.approx(depth[index == k].max())
            assert s.luminance[k] == pytest.approx(lum[index == k].mean())


def _wall_scene(distance: float = 5.0) -> Scene:
    """A camera on the -Y axis looking at a unit cube face-on (+Y): known projections."""
    scene = Scene("Wall")
    scene.add_cube("Box", size=1, position=(0, 0, 0))
    scene.add_camera("Cam", position=(0, -distance, 0), look_at=(0, 0, 0), fov=40)
    scene.add_directional_light("Sun", direction=(0, 1, -0.2))
    return scene


def test_projected_bounds_and_depth_of_a_cube_face_on():
    scene = _wall_scene(5.0)
    w = h = 200
    obs = observe(scene, width=w, height=h, focus="Box")
    rec = obs.record("Box")
    # The front face (y = -0.5) is 4.5 m from the eye; its half size 0.5 projects to
    # 0.5 / (4.5 tan 20deg) of the half image.
    half = 0.5 / (4.5 * math.tan(math.radians(20))) * (w / 2)
    x0, y0, x1, y1 = rec["geometry"]["projected_bounds"]
    assert x0 == pytest.approx(w / 2 - half, abs=1e-3) and x1 == pytest.approx(
        w / 2 + half, abs=1e-3
    )
    assert y0 == pytest.approx(h / 2 - half, abs=1e-3) and y1 == pytest.approx(
        h / 2 + half, abs=1e-3
    )
    assert rec["depth_range"][0] == pytest.approx(4.5, abs=1e-5)
    assert rec["geometry"]["depth_extent"] == pytest.approx([4.5, 5.5])
    assert rec["geometry"]["frustum"] == "inside"
    sx0, sy0, sx1, sy1 = rec["screen_bounds"]
    assert abs(sx0 - (w / 2 - half)) <= 1 and abs(sx1 - (w / 2 + half)) <= 1
    assert rec["center"] == pytest.approx([w / 2, h / 2], abs=0.5)
    assert not any(rec["touches_border"].values())


def test_the_footprint_equals_a_render_of_the_object_alone():
    scene = _wall_scene(4.0)
    blocker = scene.add_cube("Blocker", size=0.6, position=(0.3, -1.5, 0.2))
    obs = observe(scene, width=120, height=90, samples=1, focus="Box", occlusion=True)
    from aether3d.feedback.observe import footprint

    mask = footprint(obs, scene.get_by_name("Box"))
    blocker.visible = False
    alone = scene.render(width=120, height=90, samples=1)
    k = next(o["index"] for o in alone.objects if o["name"] == "Box")
    assert np.array_equal(mask, alone.object_index == k)
    occ = obs.geometry("Box", occlusion=True)["occlusion"]
    assert occ["footprint_pixels"] == int(mask.sum())
    assert occ["occluders"][0]["name"] == "Blocker"
    seen = int(np.count_nonzero(obs.index == obs.instance_of("Box")))
    assert occ["visible_pixels_in_footprint"] == seen
    assert occ["visible_fraction"] == pytest.approx(seen / mask.sum())
    assert 0.0 < occ["visible_fraction"] < 1.0


def test_an_object_behind_the_camera_or_outside_the_view():
    scene = _wall_scene(5.0)
    scene.add_cube("Behind", size=0.5, position=(0, -8, 0))
    scene.add_cube("Aside", size=0.5, position=(30, 0, 0))
    scene.add_cube("Edge", size=1.0, position=(2.6, 0, 0))  # past the right border
    obs = observe(scene, width=160, height=90, focus=["Behind", "Aside", "Edge"])
    assert obs.record("Behind")["geometry"]["frustum"] == "outside"
    assert obs.record("Behind")["geometry"]["depth_extent"][1] < 0
    assert obs.record("Aside")["geometry"]["frustum"] == "outside"
    edge = obs.record("Edge")
    assert edge["geometry"]["frustum"] == "partial"
    assert edge["geometry"]["in_frame_fraction"] < 1.0
    assert edge["touches_border"]["right"]


def test_near_and_far_planes_cutting_an_object():
    scene = _wall_scene(5.0)
    scene.get_by_name("Cam").camera.configure(near=4.8, far=5.2)
    obs = observe(scene, width=64, height=64, focus="Box")
    geometry = obs.geometry("Box")
    assert geometry["near_clipped"] and geometry["far_clipped"]
    scene.get_by_name("Cam").camera.configure(near=0.1, far=100)
    obs = observe(scene, width=64, height=64, focus="Box")
    geometry = obs.geometry("Box")
    assert not geometry["near_clipped"] and not geometry["far_clipped"]


def test_image_statistics_black_clipped_and_linear():
    scene = _wall_scene()
    scene.render_settings.default_lights = False
    scene.remove(scene.get_by_name("Sun"))
    scene.world.ambient_intensity = 0.0
    scene.world.background = "#000000"
    obs = observe(scene, width=64, height=36, linear=True)
    assert obs.image_stats["all_black"] and obs.image_stats["black_fraction"] == 1.0
    assert obs.image_stats["luminance_source"] == "linear"
    scene.add_directional_light("Hot", direction=(0, 1, 0), intensity=50)
    obs = observe(scene, width=64, height=36, linear=True)
    st = obs.image_stats
    assert st["clipped_covered_fraction"] > 0.9
    assert st["luminance_covered"]["mean"] > 1.0  # linear light above 1 is kept
    display = observe(scene, width=64, height=36)
    assert display.image_stats["luminance_source"] == "display"
    assert display.image_stats["luminance_covered"]["mean"] <= 1.0  # clipped at 1


def test_objects_in_a_rectangle_come_from_the_object_index():
    scene = _wall_scene()
    obs = observe(scene, width=100, height=100)
    inside = obs.objects_in_rect((40, 40, 60, 60))
    assert [o["name"] for o in inside] == ["Box"] and inside[0]["fraction"] == 1.0
    assert obs.objects_in_rect((0, 0, 5, 5)) == []
    assert obs.objects_in_rect((200, 200, 300, 300)) == []


def test_images_annotated_ids_depth_masks_and_crops():
    scene = _wall_scene()
    scene.add_sphere("Ball", radius=0.3, position=(1.2, -0.5, 0))
    obs = observe(scene, width=200, height=120)
    clean = np.asarray(_png(obs.png("image")))
    assert np.array_equal(clean, obs.image)
    annotated = np.asarray(_png(obs.png("annotated")))
    changed = np.any(annotated != clean, axis=-1)
    markers = obs.markers()
    assert [m["name"] for m in markers] == ["Box", "Ball"]  # most visible first
    assert changed.any() and changed.mean() < 0.2
    ids = np.asarray(_png(obs.png("ids")))
    assert np.all(ids[obs.index < 0] == 0) and np.all(ids[obs.index >= 0].max(axis=-1) > 0)
    depth = np.asarray(_png(obs.png("depth")))
    assert depth.ndim == 2 and depth[obs.index < 0].max() == 0
    mask = np.asarray(_png(obs.png("mask", ref="Ball")))
    assert np.array_equal(mask > 0, obs.index == obs.instance_of("Ball"))
    png, rect = obs.crop_png((50, 30, 100, 90), scale=3)
    crop = np.asarray(_png(png))
    assert rect == [50, 30, 100, 90] and crop.shape == (180, 150, 3)
    assert np.array_equal(crop[::3, ::3], obs.image[30:90, 50:100])
    with pytest.raises(FeedbackError):
        obs.crop_png((500, 500, 600, 600))
    with pytest.raises(FeedbackError):
        obs.png("nope")


def test_the_store_reuses_unchanged_views_and_drops_old_ones():
    scene = _wall_scene()
    store = ObservationStore()
    a = observe(scene, width=64, height=36, store=store)
    b = observe(scene, width=64, height=36, store=store)
    assert a is b and len(store) == 1
    c = observe(scene, width=64, height=48, store=store)
    assert c is not a and len(store) == 2
    scene.get_by_name("Box").position = (0.1, 0, 0)
    assert a.stale
    d = observe(scene, width=64, height=36, store=store)
    assert d is not a and d.revision == scene.revision
    with pytest.raises(FeedbackError) as exc:
        a.geometry("Box")
    assert exc.value.code == "STALE_OBSERVATION"
    small = ObservationStore(max_bytes=a.nbytes + 10)
    observe(scene, width=64, height=36, store=small)
    first = next(iter(small._items))
    observe(scene, width=64, height=48, store=small)
    with pytest.raises(FeedbackError) as exc:
        small.get(first)
    assert exc.value.code == "OBSERVATION_EXPIRED"
    with pytest.raises(FeedbackError) as exc:
        small.get("obs-nope")
    assert exc.value.code == "OBSERVATION_NOT_FOUND"


def test_observations_inside_a_transaction_are_transient():
    scene = _wall_scene()
    store = ObservationStore()
    before = observe(scene, width=64, height=36, store=store)
    with scene.transaction("try") as tx:
        scene.get_by_name("Box").position = (1, 0, 0)
        inside = observe(scene, width=64, height=36, store=store)
        assert inside is not before and inside.stale and inside.id not in store
        tx.rollback()
    again = observe(scene, width=64, height=36, store=store)
    assert again is before  # the rolled-back transaction left no cached image behind


def test_limits_and_arguments():
    scene = _wall_scene()
    with pytest.raises(FeedbackError) as exc:
        observe(scene, width=10_000, height=100)
    assert exc.value.code == "INVALID_ARGUMENT"
    with pytest.raises(FeedbackError) as exc:
        observe(scene, width=4000, height=4000)  # within 8192 per side, beyond the pixel limit
    assert exc.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(FeedbackError) as exc:
        observe(scene, width=1000, height=1000, limits=ObservationLimits(max_pixels=10_000))
    assert exc.value.code == "LIMIT_EXCEEDED"
    with pytest.raises(FeedbackError) as exc:
        observe(scene, backend="raytrace")
    assert exc.value.code == "INVALID_ARGUMENT"
    with pytest.raises(FeedbackError) as exc:
        observe(
            scene,
            width=64,
            height=64,
            backend="path",
            path={"samples": 4},
            limits=ObservationLimits(max_path_samples=100),
        )
    assert exc.value.code == "LIMIT_EXCEEDED"


def test_a_path_traced_observation_reports_the_cpu():
    scene = _wall_scene()
    obs = observe(scene, width=32, height=18, backend="path", path={"samples": 2, "bounces": 1})
    assert obs.renderer["backend"] == "path" and obs.renderer["device"] == "cpu"
    assert obs.renderer["samples_per_pixel"] == 2
    assert obs.record("Box")["visible_pixels"] > 0


def test_gpu_observations_agree_with_the_cpu(gpu):
    scene = _wall_scene()
    scene.add_sphere("Ball", radius=0.3, position=(1.2, -0.5, 0))
    cpu = observe(scene, width=160, height=90, samples=1)
    gpu_obs = observe(scene, width=160, height=90, samples=1, backend="gpu", adapter=gpu.adapter)
    assert gpu_obs.renderer["device"] == "gpu" and gpu_obs.renderer["adapter"]["name"]
    for name in ("Box", "Ball"):
        a, b = cpu.record(name, geometry=False), gpu_obs.record(name, geometry=False)
        # The same objects within a pixel of silhouette (edge snapping differs).
        assert abs(a["visible_pixels"] - b["visible_pixels"]) <= 0.05 * a["visible_pixels"] + 4
        assert all(
            abs(p - q) <= 1 for p, q in zip(a["screen_bounds"], b["screen_bounds"], strict=True)
        )


def test_a_reused_observation_marks_the_focus_of_each_call():
    scene = _wall_scene()
    scene.add_sphere("Ball", radius=0.3, position=(1.2, -0.5, 0))
    store = ObservationStore()
    plain = observe(scene, width=120, height=90, store=store)
    assert [m["name"] for m in plain.markers()] == ["Box", "Ball"]
    focused = observe(scene, width=120, height=90, store=store, focus="Ball")
    assert focused.id == plain.id and focused.image is plain.image  # one render
    assert [m["name"] for m in focused.markers()] == ["Ball"]
    assert focused.png("annotated") != plain.png("annotated")
    assert store.get(plain.id) is focused  # the store keeps the latest marking
