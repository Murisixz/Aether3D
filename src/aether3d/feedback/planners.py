"""Planners: from one finding to correction drafts (see :mod:`.corrections`).

Each planner computes its correction from the finding's measured evidence -- a gap, a
penetration, a separating translation verified by the exact intersection test, the exact
upright rotation, a camera framing -- and says how safe it is. A finding no planner can
correct deterministically (a missing object, an object inside another, an unmet spatial
relation) is returned as *unresolved* with the reason: those need a decision by the agent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AetherError
from .corrections import CorrectionPolicy, Draft

if TYPE_CHECKING:
    from .checks import Diagnosis, Finding


def _check_of(diagnosis: Diagnosis, finding: Finding):
    try:
        return diagnosis.expectations.check(finding.check)
    except AetherError:
        return None


def _expected(finding: Finding, after: float | None = 0.0) -> list[dict[str, Any]]:
    return [
        {
            "finding": finding.key,
            "code": finding.code,
            "before": finding.magnitude,
            "after": after,
            "unit": finding.unit,
        }
    ]


def _subject(finding: Finding, *roles: str) -> dict[str, Any] | None:
    for s in finding.subjects:
        if not roles or s.get("role") in roles:
            return s
    return None


def plan_finding(
    diagnosis: Diagnosis, finding: Finding, policy: CorrectionPolicy, context: dict[str, Any]
) -> tuple[list[Draft], str | None]:
    """Drafts for ``finding``, or ``([], reason)``."""
    if finding.basis in ("vision", "agent"):
        return _vision(diagnosis, finding, policy)
    planner = PLANNERS.get(finding.code)
    if planner is None:
        return [], REASONS.get(finding.code, "no automatic correction for this finding")
    check = _check_of(diagnosis, finding)
    fix = {} if check is None else dict(check.fix)
    if fix.get("move") == "none":
        return [], "the expectation asks not to move anything (fix.move: none)"
    return planner(diagnosis, finding, fix, context)


REASONS = {
    "MISSING_ELEMENT": "an object is missing: create it with a plan (the engine does not "
    "invent content)",
    "TOO_MANY": "more objects than expected: decide which to remove (corrections never delete)",
    "MISSING_MATERIAL": "the expected material does not exist: create it first",
    "CONTAINED": "one object lies inside another: decide where it belongs",
    "NO_SUPPORT": "nothing is below the object: decide where it should stand",
    "RELATION_VIOLATED": "a spatial relation does not hold: decide which object to move",
    "NO_GEOMETRY": "the object has no geometry",
    "NOT_RENDERED": "the object cannot be drawn (see its error)",
    "RENDER_WARNING": "a render diagnostic: see its message",
    "POSSIBLE_PENETRATION": "informational: state a no_overlap check if it is unintended",
    "POSSIBLE_CONTAINMENT": "informational: state a no_overlap check if it is unintended",
    "BLACK_AREAS": "black areas can be shadows or background: decide on lights or exposure",
}


# --------------------------------------------------------------------------- #
# Placement
# --------------------------------------------------------------------------- #


def _rests_on(diagnosis, finding, fix, context):
    subject = _subject(finding, "subject")
    support = _subject(finding, "support")
    ev = finding.evidence
    exact = bool(ev.get("exact", True))
    if finding.code == "FLOATING":
        dz, verb = -float(ev["gap"]), "Lower"
    else:
        dz, verb = float(ev["penetration"]), "Raise"
    move_support = fix.get("move") == "support" and support is not None
    target = support if move_support else subject
    offset = [0.0, 0.0, -dz if move_support else dz]
    where = f"onto {support['name']!r}" if support else "onto its support"
    title = (
        f"{'Move' if move_support else verb} {target['name']!r} "
        f"{abs(dz):.4g} m {'up' if offset[2] > 0 else 'down'}"
        + ("" if move_support else f" {where}")
    )
    measured = "gap" if finding.code == "FLOATING" else "penetration"
    rationale = (
        f"check {finding.check!r} states that {subject['name']!r} rests on "
        f"{support['name'] if support else 'what is below it'!r}; its measured vertical "
        f"{measured} is {abs(dz):.6g} m ({ev.get('method')}). Moving "
        f"{target['name']!r} by exactly that closes it."
    )
    return [
        Draft(
            title,
            [finding],
            [{"action": "translate", "object": target["id"], "offset": offset, "frame": "world"}],
            "safe" if exact and not move_support else "verify",
            rationale,
            expected=_expected(finding),
            validation=[
                f"re-check {finding.check!r}: rests within tolerance",
                "every check that passed still passes (no new intersection)",
            ],
        )
    ], None


def _not_on_support(diagnosis, finding, fix, context):
    subject = _subject(finding, "subject")
    supports = [s for s in finding.subjects if s.get("role") == "support"]
    if len(supports) != 1:
        return [], "several objects could be the support: name it in the check"
    from ..spatial.index import get_index
    from .geometry import union_box
    from .observe import parts_of

    index = get_index(diagnosis.scene)
    scene = diagnosis.scene

    def entries(oid: str) -> list:
        return [index.by_id[o.id] for o in parts_of(scene._objects[oid]) if o.id in index.by_id]

    es, eb = entries(subject["id"]), entries(supports[0]["id"])
    if not es or not eb:
        return [], "no geometry"
    slo, shi = union_box(es)
    blo, bhi = union_box(eb)
    target_xy = (blo[:2] + bhi[:2]) / 2.0
    center_xy = (slo[:2] + shi[:2]) / 2.0
    offset = [
        float(target_xy[0] - center_xy[0]),
        float(target_xy[1] - center_xy[1]),
        float(bhi[2] - slo[2]),
    ]
    return [
        Draft(
            f"Put {subject['name']!r} on {supports[0]['name']!r}",
            [finding],
            [{"action": "translate", "object": subject["id"], "offset": offset, "frame": "world"}],
            "verify",
            f"{subject['name']!r} is not above its support; this centers its world box on "
            f"the support's and sets its lowest point on the support's highest (box-based: "
            "verification measures the result exactly)",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
        )
    ], None


def _intersection(diagnosis, finding, fix, context):
    ev = finding.evidence
    seps = ev.get("separations") or []
    if not seps:
        return [], "no separating translation was found"
    a, b = finding.subjects[0], finding.subjects[1]
    tol = float(ev.get("tolerance", 1e-4))
    move = fix.get("move", "objects")
    want = "subject" if move == "objects" else "against"
    movers = [s for s in (a, b) if s.get("role") == want]
    ambiguous = len(movers) != 1
    mover = a if ambiguous else movers[0]
    sign = 1.0 if mover is a else -1.0  # separations move a; moving b goes the other way
    horizontal = [s for s in seps if abs(s["direction"][2]) < 0.5]
    choices = horizontal or seps
    best = choices[0]

    def action(who: dict[str, Any], sep: dict[str, Any], sgn: float) -> dict[str, Any]:
        d = np.asarray(sep["direction"]) * sgn * (sep["distance"] + 0.5 * tol)
        return {
            "action": "translate",
            "object": who["id"],
            "offset": [float(v) for v in d],
            "frame": "world",
        }

    other = b if mover is a else a
    alternatives = [
        {
            "title": f"move {other['name']!r} instead",
            "actions": [action(other, best, -sign)],
            "safety": "approval",
        }
    ]
    for sep in choices[1:3]:
        alternatives.append(
            {
                "title": f"move {mover['name']!r} {sep['distance']:.4g} m along "
                f"{[round(v * sign, 3) for v in sep['direction']]}",
                "actions": [action(mover, sep, sign)],
                "safety": "verify",
            }
        )
    rationale = (
        f"{a['name']!r} and {b['name']!r} intersect (exact triangle test). The shortest "
        f"{'horizontal ' if horizontal else ''}axis translation found that separates them -- "
        f"verified by the exact test -- moves {mover['name']!r} {best['distance']:.6g} m"
    )
    notes = []
    if ambiguous:
        notes.append(
            "which object should move is not stated (fix.move, or an against selector): "
            "this needs approval"
        )
    return [
        Draft(
            f"Separate {mover['name']!r} from {other['name']!r}",
            [finding],
            [action(mover, best, sign)],
            "approval" if ambiguous else "verify",
            rationale,
            expected=_expected(finding),
            validation=[
                f"re-check {finding.check!r}: the pair no longer intersects",
                f"every check involving {mover['name']!r} that passed still passes",
            ],
            alternatives=alternatives,
            notes=notes,
        )
    ], None


def _out_of_region(diagnosis, finding, fix, context):
    ev = finding.evidence
    if not ev.get("fits", False):
        return [], "the object is larger than the region"
    below = np.asarray(ev["below_min"])
    above = np.asarray(ev["above_max"])
    offset = [float(v) for v in below - above]
    subject = _subject(finding, "subject")
    return [
        Draft(
            f"Move {subject['name']!r} into its region",
            [finding],
            [{"action": "translate", "object": subject["id"], "offset": offset, "frame": "world"}],
            "verify",
            "the smallest move along each axis that brings its world box inside the region",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
        )
    ], None


def _distance(diagnosis, finding, fix, context):
    ev = finding.evidence
    pa, pb = np.asarray(ev["point_a"]), np.asarray(ev["point_b"])
    span = pa - pb
    norm = float(np.linalg.norm(span))
    if norm < 1e-12:
        return [], "the objects touch: the direction to move is undefined"
    a = _subject(finding, "subject")
    off = float(finding.magnitude or 0.0) * (1.0 if finding.code == "TOO_CLOSE" else -1.0)
    offset = [float(v) for v in span / norm * off]
    return [
        Draft(
            f"Move {a['name']!r} {'away from' if off > 0 else 'toward'} "
            f"{finding.subjects[1]['name']!r}",
            [finding],
            [{"action": "translate", "object": a["id"], "offset": offset, "frame": "world"}],
            "verify",
            "along the line between the closest points, by the distance missing",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
        )
    ], None


def _tilted(diagnosis, finding, fix, context):
    from ..math3d import matrix3_to_quat
    from ..rendering.scene_data import world_matrices

    subject = _subject(finding, "subject")
    scene = diagnosis.scene
    obj = scene._objects[subject["id"]]
    check = _check_of(diagnosis, finding)
    axis = np.asarray(
        (check.params.get("axis") if check is not None else None) or (0.0, 0.0, 1.0),
        dtype=np.float64,
    )
    worlds = world_matrices(scene)
    m = worlds[obj.id][:3, :3]
    norms = np.linalg.norm(m, axis=0)
    if np.any(norms < 1e-12):
        return [], "the object has a zero scale"
    rw = m / norms
    if not np.allclose(rw.T @ rw, np.eye(3), atol=1e-9):
        return [], "the object's world transform has shear (a non-uniformly scaled parent)"
    up = m @ axis
    up = up / np.linalg.norm(up)
    zhat = np.array([0.0, 0.0, 1.0])
    rot_axis = np.cross(up, zhat)
    s, c = float(np.linalg.norm(rot_axis)), float(up @ zhat)
    if s < 1e-15:
        return [], "the axis is vertical already" if c > 0 else "the object is upside down"
    k = rot_axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    fix_rot = np.eye(3) + s * kx + (1 - c) * (kx @ kx)  # Rodrigues: up -> +Z
    new_world = fix_rot @ rw
    if obj.parent is not None:
        pm = worlds[obj.parent.id][:3, :3]
        pn = np.linalg.norm(pm, axis=0)
        if np.any(pn < 1e-12) or not np.allclose(pn, pn[0], rtol=1e-9):
            return [], "the parent is not uniformly scaled: an upright local rotation is not exact"
        pr = pm / pn
        local = pr.T @ new_world
    else:
        local = new_world
    q = matrix3_to_quat(local)
    tilt = math.degrees(math.atan2(s, c))
    return [
        Draft(
            f"Stand {subject['name']!r} upright",
            [finding],
            [{"action": "rotate", "object": subject["id"], "rotation": [float(v) for v in q]}],
            "verify",
            f"turns the object by {tilt:.4g} degrees about a horizontal axis so its local "
            f"axis {axis.tolist()} points up, keeping its heading (the smallest such turn, "
            "about its origin)",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "it still rests where it did"],
        )
    ], None


def _wrong_material(diagnosis, finding, fix, context):
    subject = _subject(finding, "subject")
    ev = finding.evidence
    return [
        Draft(
            f"Assign {ev['expected']!r} to {subject['name']!r}",
            [finding],
            [{"action": "assign_material", "object": subject["id"], "material": ev["expected_id"]}],
            "safe",
            f"check {finding.check!r} states the material; it exists",
            expected=_expected(finding, None),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


def _hidden(diagnosis, finding, fix, context):
    subject = _subject(finding, "subject")
    return [
        Draft(
            f"Show {subject['name']!r}",
            [finding],
            [{"action": "set_visibility", "object": subject["id"], "visible": True}],
            "safe",
            f"check {finding.check!r} states it must be visible; it is hidden",
            expected=_expected(finding, None),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #


def _view_camera(diagnosis: Diagnosis, finding: Finding):
    """The scene camera behind a finding's view, or ``(None, reason)``."""
    obs = diagnosis.observed.get(finding.view) if finding.view is not None else None
    if obs is None and finding.observation is not None:
        obs = next((o for o in diagnosis.observed.values() if o.id == finding.observation), None)
    if obs is None:
        return None, None, "the finding has no observed view"
    if obs.view.camera is None:
        return None, obs, "the view is a virtual camera (not part of the scene)"
    return obs.view.camera, obs, None


def _frame(diagnosis, finding, fix, context):
    if fix.get("camera") is False:
        return [], "the expectation does not allow camera changes (fix.camera: false)"
    camera, obs, reason = _view_camera(diagnosis, finding)
    if camera is None:
        return [], reason
    targets = context.setdefault("frame_targets", {}).get(camera.id)
    if targets is None:
        # Everything the view's visibility and framing checks expect to see.
        ids: list[str] = []
        for check in diagnosis.expectations.checks:
            if check.kind in ("visible", "framing") and check.view == finding.view:
                from ..plans import select_objects

                try:
                    objs = select_objects(diagnosis.scene, check.params["objects"])
                except AetherError:
                    continue
                ids.extend(o.id for o in objs if o.visible and o.camera is None and o.light is None)
        targets = list(dict.fromkeys(ids))
        context["frame_targets"][camera.id] = targets
    action: dict[str, Any] = {"action": "frame_camera", "camera": camera.id, "margin": 0.08}
    if targets and finding.code != "EMPTY_VIEW":
        action["targets"] = targets
    names = (
        "everything visible"
        if "targets" not in action
        else ", ".join(diagnosis.scene._objects[t].name for t in targets[:4])
        + ("..." if len(targets) > 4 else "")
    )
    return [
        Draft(
            f"Frame {camera.name!r} on {names}",
            [finding],
            [action],
            "verify",
            "moves the camera along its view direction (exact framing on the targets' "
            "vertices, rendering/framing.py) so every object the view must show is in frame; "
            "the view direction and lens are kept",
            expected=_expected(finding),
            validation=["re-check the view's visibility and framing checks", "no new failure"],
            group=f"frame:{camera.id}",
        )
    ], None


def _clipping(diagnosis, finding, fix, context):
    camera, obs, reason = _view_camera(diagnosis, finding)
    if camera is None:
        return [], reason
    d0, d1 = finding.evidence["depth_extent"]
    if finding.code == "NEAR_CLIPPED":
        if d0 <= 0.0:
            return [], "the object reaches behind the camera's eye: move the camera instead"
        far = camera.camera.far
        near = max(d0 * 0.5, far / 1e8)
        if near >= camera.camera.near:
            return [], "the near plane is already in front of the object"
        action = {"action": "set_camera", "camera": camera.id, "near": near}
        what = f"near plane to {near:.4g} m"
    else:
        far = d1 * 1.25 + 1e-6
        action = {"action": "set_camera", "camera": camera.id, "far": far}
        what = f"far plane to {far:.4g} m"
    return [
        Draft(
            f"Move {camera.name!r}'s {what}",
            [finding],
            [action],
            "safe",
            "a clipping plane cuts a visible object; moving it outward only reveals geometry",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


def _just_inside(value: float, lo: float | None, hi: float | None, margin: float) -> float | None:
    """The smallest change that brings ``value`` into ``[lo, hi]``: the violated bound,
    ``margin`` times inside it (never past the middle of the range)."""
    if lo is not None and value < lo:
        target = lo * margin
        return min(target, math.sqrt(lo * hi) if hi is not None and lo > 0 else target)
    if hi is not None and value > hi:
        target = hi / margin
        return max(target, math.sqrt(lo * hi) if lo is not None and lo > 0 else target)
    return None


def _frame_limit(obs: Any, rows: list[int], margin: float = 0.03) -> float:
    """The largest size ratio a move along the view direction can apply before the objects'
    pixel bounds leave the frame (bounds scale about the image center; points at other
    depths scale differently, so verification has the last word)."""
    s = obs.stats
    w, h = obs.width, obs.height
    x0 = min(int(s.x0[k]) for k in rows)
    y0 = min(int(s.y0[k]) for k in rows)
    x1 = max(int(s.x1[k]) for k in rows)
    y1 = max(int(s.y1[k]) for k in rows)
    limit = math.inf
    for lo, hi, size in ((x0, x1, w), (y0, y1, h)):
        half, room = size / 2, size / 2 - margin * size
        for edge in (lo, hi):
            reach = abs(edge - half)
            if reach > 1e-9:
                limit = min(limit, room / reach)
    return limit


def _scale_to(finding: Finding, check: Any) -> float | None:
    """The size ratio (new / current) that brings a framing measure into its range."""
    ev = finding.evidence
    key = ev.get("measure")
    if key is None or check is None:
        return None
    value = float(ev[key])
    lo, hi = check.params[key]["min"], check.params[key]["max"]
    target = _just_inside(value, lo, hi, 1.15)
    if value <= 0 or target is None or target <= 0:
        return None
    ratio = target / value
    return math.sqrt(ratio) if key == "fill" else ratio


def _dolly(diagnosis, finding, fix, context):
    if fix.get("camera") is False:
        return [], "the expectation does not allow camera changes (fix.camera: false)"
    camera, obs, reason = _view_camera(diagnosis, finding)
    if camera is None:
        return [], reason
    if not camera.camera.is_perspective:
        return [], "moving an orthographic camera does not change sizes"
    check = _check_of(diagnosis, finding)
    ratio = 2.0 if finding.code == "TOO_SMALL" else _scale_to(finding, check)
    if ratio is None:
        return [], "the size cannot be measured"
    # Distance from the eye to the objects, along the view: their pixels' mean depth.
    ids = [s["id"] for s in finding.subjects]
    depths = []
    for oid in ids:
        if oid not in obs.scene._objects:
            continue
        for k in obs.instances_of(oid):
            if obs.stats.pixels[k] and not math.isnan(float(obs.stats.dmin[k])):
                depths.append((float(obs.stats.dmin[k]) + float(obs.stats.dmax[k])) / 2)
    if not depths:
        return [], "the objects are not visible: frame the camera on them first"
    rows = [k for oid in ids if oid in obs.scene._objects for k in obs.instances_of(oid)]
    rows = [k for k in rows if obs.stats.pixels[k]]
    notes = []
    if ratio > 1.0:
        cap = _frame_limit(obs, rows)
        if cap < ratio:
            notes.append(
                f"limited to {cap:.3g} times (of {ratio:.3g}) to keep the objects in the frame"
            )
            ratio = cap
        if ratio <= 1.0 + 1e-6:
            return [], "moving closer would push the objects out of the frame"
    d = float(np.mean(depths))
    new_d = d / ratio
    move = d - new_d
    if new_d <= camera.camera.near:
        return [], "the camera would have to move past the objects"
    return [
        Draft(
            f"Move {camera.name!r} {abs(move):.4g} m {'closer' if move > 0 else 'back'}",
            [finding],
            [{"action": "dolly_camera", "camera": camera.id, "distance": move}],
            "verify",
            f"projected size scales with the inverse of the distance (perspective): from "
            f"{d:.4g} m to {new_d:.4g} m changes it {ratio:.3g} times, just into the "
            "expected range",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
            notes=notes,
            group=f"dolly:{camera.id}",
        )
    ], None


def _aim(diagnosis, finding, fix, context):
    if fix.get("camera") is False:
        return [], "the expectation does not allow camera changes (fix.camera: false)"
    camera, obs, reason = _view_camera(diagnosis, finding)
    if camera is None:
        return [], reason
    check = _check_of(diagnosis, finding)
    region = check.params.get("region") if check is not None else None
    if region is None:
        return [], "no target region"
    ev = finding.evidence
    cx, cy = ev["center"]
    tx, ty = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    point = _aim_point(obs, (cx, cy), (tx, ty))
    if point is None:
        return [], "the aim could not be solved"
    look_at, residual = point
    return [
        Draft(
            f"Aim {camera.name!r} to put the objects at ({tx:.2f}, {ty:.2f}) of the image",
            [finding],
            [{"action": "aim_camera", "camera": camera.id, "look_at": look_at}],
            "verify",
            "turns the camera in place so the objects' visible center lands at the center of "
            f"the expected region (solved on the camera's exact projection; residual "
            f"{residual:.3g} of the image)",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
            group=f"aim:{camera.id}",
        )
    ], None


def _aim_point(obs: Any, center: tuple[float, float], target: tuple[float, float]):
    """A point for the camera to look at so the world point now seen at ``center``
    (normalized image coordinates) appears at ``target``: Newton steps on the exact
    projection, turning the camera about its eye with +Z up."""
    from ..math3d import look_rotation_matrix
    from ..spatial.queries import View

    view = View(obs.scene, obs.view.target, obs.width, obs.height)
    # The world point at the objects' visible center (its depth along the view).
    xn = center[0] * 2 - 1
    yn = 1 - center[1] * 2
    depth = _center_depth(obs, center)
    if depth is None:
        return None
    if view.perspective:
        p = view.eye + depth * (view.forward + xn * view.hx * view.right + yn * view.hy * view.up)
    else:
        p = view.eye + depth * view.forward + xn * view.hx * view.right + yn * view.hy * view.up
    eye = view.eye

    def project(direction: np.ndarray) -> np.ndarray:
        rot = look_rotation_matrix(tuple(direction.tolist()), (0.0, 0.0, 1.0))
        forward, up = -rot[:, 1], rot[:, 2]
        right = np.cross(forward, up)
        rel = p - eye
        z = rel @ forward
        if view.perspective:
            x = (rel @ right) / (z * view.hx)
            y = (rel @ up) / (z * view.hy)
        else:
            x, y = (rel @ right) / view.hx, (rel @ up) / view.hy
        return np.array([(x + 1) / 2, (1 - y) / 2])

    goal = np.array(target)
    yaw = math.atan2(view.forward[1], view.forward[0])
    pitch = math.asin(max(-1.0, min(1.0, float(view.forward[2]))))

    def direction(a: float, b: float) -> np.ndarray:
        return np.array([math.cos(b) * math.cos(a), math.cos(b) * math.sin(a), math.sin(b)])

    for _ in range(20):
        cur = project(direction(yaw, pitch))
        err = cur - goal
        if float(np.max(np.abs(err))) < 1e-6:
            break
        eps = 1e-6
        jac = np.column_stack(
            [
                (project(direction(yaw + eps, pitch)) - cur) / eps,
                (project(direction(yaw, pitch + eps)) - cur) / eps,
            ]
        )
        try:
            step = np.linalg.solve(jac, err)
        except np.linalg.LinAlgError:
            return None
        yaw -= float(step[0])
        pitch = max(-1.55, min(1.55, pitch - float(step[1])))
    residual = float(np.max(np.abs(project(direction(yaw, pitch)) - goal)))
    if not math.isfinite(residual) or residual > 1e-3:
        return None
    distance = float(np.linalg.norm(p - eye))
    look = eye + direction(yaw, pitch) * max(distance, 1e-3)
    return [float(v) for v in look], residual


def _center_depth(obs: Any, center: tuple[float, float]) -> float | None:
    if obs.depth is None:
        return None
    x = min(obs.width - 1, max(0, int(center[0] * obs.width)))
    y = min(obs.height - 1, max(0, int(center[1] * obs.height)))
    d = float(obs.depth[y, x])
    if math.isfinite(d):
        return d
    finite = np.isfinite(obs.depth) & (obs.index >= 0)
    if not finite.any():
        return None
    return float(np.median(obs.depth[finite]))


# --------------------------------------------------------------------------- #
# Light and exposure
# --------------------------------------------------------------------------- #


def _exposure(diagnosis, finding, fix, context):
    scene = diagnosis.scene
    check = _check_of(diagnosis, finding)
    current = scene.render_settings.exposure
    # A default (heuristic) threshold is not a stated expectation: its fix needs approval.
    safety = "approval" if finding.basis == "heuristic" else "verify"
    if finding.code == "CLIPPED_HIGHLIGHTS":
        obs = diagnosis.observed.get(finding.view)
        if obs is None or obs.linear is None:
            return [], "needs linear light (observe with linear=True)"
        limit = check.params["max_clipped_fraction"] if check is not None else 0.0
        covered = obs.index >= 0
        peak = obs.linear.max(axis=-1)[covered]
        if not peak.size:
            return [], "nothing visible"
        q = float(np.quantile(peak, max(0.0, 1.0 - limit)))
        if q <= 1.0:
            return [], "the clipped pixels are not light above 1 (already at the limit)"
        delta = math.log2(0.98 / q)
    else:
        ev = finding.evidence
        stats = ev["stats"]
        over = ev.get("over", "objects")
        summary = stats["luminance_covered"] if over == "objects" else stats["luminance"]
        mean = summary["mean"] if summary else 0.0
        if mean <= 0.0:
            return _add_light(diagnosis, finding)
        if check is not None and "mean_luminance" in check.params:
            lo, hi = check.params["mean_luminance"]["min"], check.params["mean_luminance"]["max"]
        elif finding.code == "UNDEREXPOSED":
            lo, hi = ev.get("rule", 0.01) * 4, None
        else:
            lo, hi = None, 0.5
        target = _just_inside(mean, lo, hi, 1.25)
        if target is None:
            return [], "the luminance is already in range"
        delta = math.log2(target / mean)
        if stats.get("luminance_source") == "display" and finding.code == "OVEREXPOSED":
            delta -= 0.5  # clipped 8-bit values understate the light: aim lower
    new = current + delta
    return [
        Draft(
            f"Set the exposure to {new:+.3g} stops",
            [finding],
            [{"action": "set_exposure", "exposure": round(new, 6)}],
            safety,
            f"exposure scales linear light by 2^exposure: {delta:+.3g} stops brings the "
            "measured luminance to the target (verification re-measures; clipped pixels "
            "can make one step fall short)",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
            alternatives=[
                {
                    "title": "change the lights instead (keeps the exposure)",
                    "actions": [],
                    "safety": "approval",
                }
            ],
            group="exposure",
        )
    ], None


def _add_light(diagnosis, finding):
    from ..rendering.camera import frame_from_matrix
    from ..rendering.scene_data import world_matrices

    scene = diagnosis.scene
    camera = scene.active_camera
    direction = [-0.4, 0.6, -1.0]
    if camera is not None:
        frame, _ = frame_from_matrix(world_matrices(scene)[camera.id])
        d = frame.forward - 0.6 * frame.right + -0.8 * np.array([0.0, 0.0, 1.0])
        direction = [float(v) for v in d / np.linalg.norm(d)]
    name = "Feedback key light"
    k = 1
    while name in scene._by_name:
        k += 1
        name = f"Feedback key light {k}"
    return [
        Draft(
            "Add a key light",
            [finding],
            [
                {
                    "action": "add_light",
                    "type": "directional",
                    "name": name,
                    "direction": direction,
                    "intensity": 2.0,
                }
            ],
            "approval",
            "the image has no light to scale; a directional light from the camera's "
            "upper left (a choice: needs approval)",
            expected=_expected(finding, None),
            validation=["re-check the view's exposure"],
        )
    ], None


def _black(diagnosis, finding, fix, context):
    scene = diagnosis.scene
    if scene.render_settings.exposure < -6:
        return [
            Draft(
                "Reset the exposure to 0",
                [finding],
                [{"action": "set_exposure", "exposure": 0.0}],
                "approval",
                f"the exposure is {scene.render_settings.exposure:g} stops",
                expected=_expected(finding, None),
                validation=["re-check the view"],
            )
        ], None
    return _add_light(diagnosis, finding)


def _occluded(diagnosis, finding, fix, context):
    occluders = [s for s in finding.subjects if s.get("role") == "occluder"]
    occ = finding.evidence.get("occlusion", {}).get("occluders", [])
    if not occluders and occ:
        occluders = [{"id": o["id"], "name": o["name"]} for o in occ[:1]]
    subject = _subject(finding, "subject")
    if not occluders:
        return [], "no single object covers it"
    first = occluders[0]
    return [
        Draft(
            f"Hide {first['name']!r}, which covers {subject['name']!r}",
            [finding],
            [{"action": "set_visibility", "object": first["id"], "visible": False}],
            "approval",
            f"{first['name']!r} covers most of the pixels {subject['name']!r} would show "
            "(its unoccluded footprint against the object index); hiding it is one choice -- "
            "moving it or the camera are others",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}", "no new failure"],
            alternatives=[
                {
                    "title": "move the camera to another viewpoint",
                    "actions": [],
                    "safety": "approval",
                },
                {"title": f"move {first['name']!r} aside", "actions": [], "safety": "approval"},
            ],
        )
    ], None


# --------------------------------------------------------------------------- #
# Vision and agent findings
# --------------------------------------------------------------------------- #


def _vision(diagnosis, finding, policy):
    ev = finding.evidence
    suggestion = ev.get("suggestion")
    if suggestion is None:
        return [], "a vision finding without a suggested action: decide on a correction"
    confidence = float(ev.get("confidence", 0.0))
    if confidence < policy.min_vision_confidence:
        return [], (
            f"confidence {confidence:g} is below min_vision_confidence "
            f"{policy.min_vision_confidence:g}"
        )
    target = suggestion.get("object") or suggestion.get("light") or suggestion.get("camera")
    grounded_ids = {s["id"] for s in finding.subjects}
    notes = []
    if suggestion.get("object") is not None:
        obj = diagnosis.scene._objects.get(target) or diagnosis.scene._by_name.get(target)
        if obj is None:
            return [], f"the suggested target {target!r} does not exist"
        if obj.id not in grounded_ids:
            notes.append(
                f"{obj.name!r} is not grounded in the finding's region by the object index: "
                "check the target before approving"
            )
    provenance = ev.get("provenance", {})
    who = provenance.get("model") or provenance.get("provider") or "an evaluator"
    return [
        Draft(
            f"{suggestion['action'].replace('_', ' ').capitalize()} (suggested by {who})",
            [finding],
            [dict(suggestion)],
            "approval",
            f"{finding.basis} finding: {finding.message}",
            confidence=confidence,
            basis=finding.basis,
            expected=[],
            validation=["re-observe; a vision judgment is checked again only by a new evaluation"],
            notes=notes,
        )
    ], None


PLANNERS = {
    "FLOATING": _rests_on,
    "SUNKEN": _rests_on,
    "NOT_ON_SUPPORT": _not_on_support,
    "INTERSECTION": _intersection,
    "OUT_OF_REGION": _out_of_region,
    "TOO_CLOSE": _distance,
    "TOO_FAR": _distance,
    "TILTED": _tilted,
    "WRONG_MATERIAL": _wrong_material,
    "HIDDEN": _hidden,
    "OUT_OF_VIEW": _frame,
    "PARTIALLY_OUT_OF_FRAME": _frame,
    "NOT_VISIBLE": _frame,
    "EMPTY_VIEW": _frame,
    "NEAR_CLIPPED": _clipping,
    "FAR_CLIPPED": _clipping,
    "TOO_SMALL": _dolly,
    "TOO_SMALL_IN_FRAME": _dolly,
    "TOO_LARGE_IN_FRAME": _dolly,
    "OFF_CENTER": _aim,
    "UNDEREXPOSED": _exposure,
    "OVEREXPOSED": _exposure,
    "CLIPPED_HIGHLIGHTS": _exposure,
    "BLACK_IMAGE": _black,
    "OCCLUDED": _occluded,
}
