"""Shadow settings, shadowed-light selection and shadow-map fitting (no GPU needed)."""

import numpy as np
import pytest

from aether3d import RenderError, Scene, ShadowSettings
from aether3d.rendering.scene_data import extract_render_scene, resolve_camera
from aether3d.rendering.settings import MAX_SHADOW_MAPS
from aether3d.rendering.shadows import instance_corners, plan_shadows, select_shadow_lights


def scene_with(*lights, camera=((8, -12, 7), (0, 0, 0))):
    scene = Scene("Shadows")
    scene.add_plane("Ground", size=20).color = "#6a9a4a"
    scene.add_cube("Tower", size=2, position=(0, 0, 3), scale=(1, 1, 3))
    for kind, name, value, intensity in lights:
        if kind == "directional":
            scene.add_directional_light(name, direction=value, intensity=intensity)
        else:
            scene.add_point_light(name, position=value, intensity=intensity)
    position, target = camera
    scene.add_camera("Cam", position=position, look_at=target, fov=40, near=0.1, far=500)
    return scene


def extract(scene, aspect=16 / 9):
    return extract_render_scene(scene, resolve_camera(scene, None), aspect=aspect)


def test_shadow_settings_validate_and_coerce():
    assert ShadowSettings.coerce(None) is None and ShadowSettings.coerce(False) is None
    assert ShadowSettings.coerce(True) == ShadowSettings()
    s = ShadowSettings.coerce({"lights": ["Sun"], "map_size": 4096, "pcf_radius": 2})
    assert s.lights == ("Sun",) and s.to_dict()["lights"] == ["Sun"]
    assert ShadowSettings.from_dict(s.to_dict(), "shadows") == s
    for bad, location in (
        ({"map_size": 100}, "shadows.map_size"),
        ({"map_size": 2048.0}, "shadows.map_size"),
        ({"bias": -1}, "shadows.bias"),
        ({"normal_offset": float("nan")}, "shadows.normal_offset"),
        ({"pcf_radius": 5}, "shadows.pcf_radius"),
        ({"lights": "Sun"}, "shadows.lights"),
        ({"lights": ["Sun", ""]}, "shadows.lights[1]"),
        ({"lights": ["a", "b", "c", "d", "e"]}, "shadows.lights"),
    ):
        with pytest.raises(RenderError) as info:
            ShadowSettings.coerce(bad)
        assert info.value.code == "INVALID_SHADOWS" and info.value.location == location, bad
    with pytest.raises(RenderError):
        ShadowSettings.coerce("on")


def test_every_directional_light_casts_by_default_brightest_first():
    lights = [("directional", f"D{i}", (0.1 * i, 0.5, -1), 1 + i) for i in range(6)]
    rs = extract(scene_with(*lights, ("point", "Bulb", (0, -3, 4), 50)))
    chosen, diagnostics = select_shadow_lights(rs, ShadowSettings())
    assert [rs.lights[i].name for i in chosen] == ["D2", "D3", "D4", "D5"]
    assert len(chosen) == MAX_SHADOW_MAPS
    codes = {d.code: d for d in diagnostics}
    limit = codes["SHADOW_LIGHT_LIMIT"]
    assert limit.severity == "warning" and "D0" in limit.message and "D1" in limit.message
    assert codes["POINT_LIGHT_SHADOWS_UNSUPPORTED"].severity == "info"


def test_named_lights_cast_and_mistakes_are_reported():
    scene = scene_with(
        ("directional", "Sun", (0.3, 0.4, -1), 2),
        ("directional", "Fill", (-0.5, 0.2, -1), 0.5),
        ("point", "Bulb", (0, -3, 4), 5),
    )
    rs = extract(scene)
    sun_id = scene.get_by_name("Sun").id
    chosen, _ = select_shadow_lights(rs, ShadowSettings(lights=[sun_id]))
    assert [rs.lights[i].name for i in chosen] == ["Sun"]
    chosen, _ = select_shadow_lights(rs, ShadowSettings(lights=["Fill", "Sun", "Sun"]))
    assert sorted(rs.lights[i].name for i in chosen) == ["Fill", "Sun"]
    for refs, needle in ((["Moon"], "no visible light"), (["Bulb"], "point light")):
        with pytest.raises(RenderError) as info:
            select_shadow_lights(rs, ShadowSettings(lights=refs))
        assert info.value.code == "INVALID_SHADOWS" and needle in info.value.message
        assert info.value.location == "shadows.lights[0]"
    scene.get_by_name("Fill").visible = False  # hidden lights do not shine, or cast
    with pytest.raises(RenderError):
        select_shadow_lights(extract(scene), ShadowSettings(lights=["Fill"]))


def light_space(shadow, points, eye):
    h = np.c_[points - eye, np.ones(len(points))]
    return (shadow.matrix @ h.T).T


@pytest.mark.parametrize(
    "direction", [(0.3, 0.4, -1.0), (0.0, 0.0, -1.0), (1.0, 0.0, -0.05), (-0.2, -0.9, -0.4)]
)
def test_the_map_covers_every_caster_in_depth_and_every_visible_receiver(direction):
    scene = scene_with(("directional", "Sun", direction, 2))
    rs = extract(scene)
    plan = plan_shadows(rs, ShadowSettings(map_size=1024), 16 / 9)
    (shadow,) = plan.maps
    eye = rs.camera.frame.position
    corners = instance_corners(rs).reshape(-1, 3)
    p = light_space(shadow, corners, eye)
    assert np.all((p[:, 2] > 0) & (p[:, 2] < 1))  # all casters inside the depth range
    assert np.allclose(p[:, 3], 1.0)
    # Every visible surface point is inside the map: sample the camera's view.
    result = scene.render(width=64, height=36, samples=1)
    f, r, u = rs.camera.frame.forward, rs.camera.frame.right, rs.camera.frame.up
    hx, hy = rs.camera.camera.half_extents(16 / 9)
    rows, cols = np.nonzero(result.object_index >= 0)
    xn = (cols + 0.5) / 64 * 2 - 1
    yn = 1 - (rows + 0.5) / 36 * 2
    depth = result.depth[rows, cols][:, None]
    seen = eye + depth * (f + xn[:, None] * hx * r + yn[:, None] * hy * u)
    q = light_space(shadow, seen, eye)
    assert np.all(np.abs(q[:, :2]) < 1.0)
    # Square texels of the documented size, and the padding for the filter.
    assert shadow.texel_size == pytest.approx(shadow.extent[0] / 1024)
    assert shadow.extent[0] == shadow.extent[1]


def test_fitting_follows_the_view_not_the_whole_ground():
    # A huge ground plane, but the camera looks at a small part of it.
    scene = scene_with(("directional", "Sun", (0.3, 0.4, -1), 2), camera=((4, -6, 3), (0, 0, 1)))
    scene.get_by_name("Ground").scale = (100, 100, 1)  # 2 km wide
    rs = extract(scene)
    (shadow,) = plan_shadows(rs, ShadowSettings(map_size=2048), 16 / 9).maps
    caster_width = np.ptp(instance_corners(rs).reshape(-1, 3)[:, 0])
    assert caster_width > 1900
    # The view frustum (ending at the farthest geometry) bounds the map across the light ...
    assert shadow.extent[0] < caster_width
    # ... and the bias is expressed in that map's texels.
    settings = ShadowSettings(map_size=2048)
    assert shadow.bias == pytest.approx(settings.bias * shadow.texel_size / shadow.depth_range)
    footprint = np.sqrt(2) * (settings.pcf_radius + 1) + settings.normal_offset
    assert shadow.normal_offset == pytest.approx(footprint * shadow.texel_size)


def test_no_shadow_maps_without_directional_lights_or_geometry():
    rs = extract(scene_with(("point", "Bulb", (0, -3, 4), 5)))
    plan = plan_shadows(rs, ShadowSettings(), 16 / 9)
    assert plan.maps == () and plan.layer_of(0) == -1
    empty = Scene()
    empty.add_directional_light("Sun")
    empty.add_camera("Cam")
    assert plan_shadows(extract(empty), ShadowSettings(), 16 / 9).maps == ()


def test_plan_reports_are_plain_json():
    import json

    rs = extract(scene_with(("directional", "Sun", (0.3, 0.4, -1), 2)))
    report = plan_shadows(rs, ShadowSettings(), 16 / 9).to_dict()
    assert json.loads(json.dumps(report))["maps"][0]["light"] == "Sun"
