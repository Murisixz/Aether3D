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
    if finding.kind == "rig":
        return _rig_fix(diagnosis, finding, policy, context)
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
    "LIGHT_NO_EFFECT": "a light does not reach the subject: decide whether to aim it, move "
    "it or remove what blocks it",
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


# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #


def _plant_feet(diagnosis, finding, fix, context):
    subject = _subject(finding, "subject")
    foot = finding.evidence.get("foot")
    scene = diagnosis.scene
    arm = scene._objects.get(subject["id"]) if subject else None
    rig = scene.animation.rig(arm) if arm is not None else None
    k = rig.index(foot) if rig is not None and foot else None
    if k is None or rig.joints[k].parent < 0 or rig.joints[rig.joints[k].parent].parent < 0:
        return [], "the foot needs two parent joints (hip and knee) for a leg IK chain"
    knee = rig.joints[k].parent
    hip = rig.joints[knee].parent
    legs = {foot: [rig.joints[hip].name, rig.joints[knee].name, foot]}
    return [
        Draft(
            f"Plant {subject['name']!r} {foot}",
            [finding],
            [{"action": "plant_feet", "armature": subject["id"], "legs": legs}],
            "verify",
            f"check {finding.check!r}: {foot} "
            + ("slides" if finding.code == "FOOT_SLIDING" else "sinks")
            + f" {finding.magnitude:.4g} m (sampled). Keyed IK pins each contact where it "
            "touched down; the rest of the motion is unchanged.",
            expected=_expected(finding, None),
            validation=[f"re-check {finding.check!r}: sliding and sinking within the limits"],
            group=f"plant:{subject['id']}",
        )
    ], None


RIG_REASONS = {
    "RIG_MISSING": "the armature has no rig: rig it first (autorig)",
    "IK_UNREACHABLE": "an IK target is beyond its chain's reach: move the target or the "
    "character (a choice of staging)",
    "IK_FLIP": "an IK chain flips between frames: move its pole or key it (a choice)",
    "ZERO_LENGTH_BONE": "a joint sits on its children: move it (edit_rig move_joint)",
    "JOINT_COLLAPSE": "the skin is already dual quaternion: edit the weights or add a "
    "corrective (a choice of shape)",
}


def _rig_fix(diagnosis, finding, policy, context):
    """A rig finding's correction: the rig's own proposal (propose_rig_fix), applied by
    apply_rig_fix, which re-measures after each change and undoes what does not help."""
    from ..animation.rig_report import propose_rig_fix

    subject = _subject(finding, "subject")
    scene = diagnosis.scene
    arm = scene._objects.get(subject["id"]) if subject else None
    measured = finding.evidence.get("rig")
    if arm is None or measured is None or scene.animation.rig(arm) is None:
        return [], RIG_REASONS["RIG_MISSING"]
    proposals = propose_rig_fix(scene.animation, arm, diagnosis={"findings": [measured]})
    drafts = []
    for p in proposals["proposals"]:
        drafts.append(
            Draft(
                f"{subject['name']!r}: {p['title']}",
                [finding],
                [{"action": "rig_fix", "armature": subject["id"], "fixes": [p]}],
                p["safety"],
                f"check {finding.check!r}: {measured['message']} (measured on the mesh). "
                "apply_rig_fix makes the change, measures the rig again and undoes it if "
                "the finding did not improve or anything got worse.",
                expected=_expected(finding),
                validation=[
                    f"re-check {finding.check!r}: the finding resolves or shrinks and no rig "
                    "finding appears"
                ],
                basis="geometry",
                group=f"rig:{subject['id']}:{p['safety']}",
            )
        )
    if not drafts:
        return [], RIG_REASONS.get(finding.code, "no automatic rig correction for this finding")
    return drafts, None


def _smooth_keys(diagnosis, finding, fix, context):
    subject = _subject(finding, "subject")
    ev = finding.evidence
    return [
        Draft(
            f"Smooth {subject['name']!r} {ev['path']} at {ev['time']:.3g} s",
            [finding],
            [
                {
                    "action": "smooth_keys",
                    "object": subject["id"],
                    "path": ev["path"],
                    "keys": [ev["key"]],
                }
            ],  # fmt: skip
            "approval",
            f"the curve's {('value', 'velocity', 'acceleration')[ev['order']]} jumps at key "
            f"{ev['key']} (exact one-sided limits); automatic clamped handles make it smooth "
            "without overshoot. A corner may be intended (an impact): approve it if not.",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


def _motion_subject(finding) -> dict[str, Any] | None:
    """The object a motion finding is about (none for joints and the shot camera)."""
    if ".joints[" in str(finding.evidence.get("location", "")):
        return None
    return _subject(finding, "subject")


def _curve_smoothing(diagnosis, finding, fix, context):
    """Acceleration steps and velocity jumps of an object's own keys: continuous handles."""
    from ..animation.curves import FCurve

    subject = _motion_subject(finding)
    if subject is None:
        return [], "the motion belongs to joints or shots: edit their keys or constraints"
    scene = diagnosis.scene
    obj = scene._objects.get(subject["id"])
    action = None if obj is None else scene.animation.action(obj)
    curves = [] if action is None else [
        ch for _p, ch in action.data.channels if isinstance(ch, FCurve)
    ]  # fmt: skip
    if not curves or all(c.smoothing == "continuous" for c in curves):
        return [], "its curves already use continuous smoothing (or it has none)"
    return [
        Draft(
            f"Continuous acceleration for {subject['name']!r}",
            [finding],
            [{"action": "curve_smoothing", "object": subject["id"], "smoothing": "continuous"}],
            "verify",
            "its automatic handles are computed so the acceleration is continuous at every "
            "automatic key (Blender's Continuous Acceleration); keys and values do not change, "
            "only the handles between them. Kept only if the steps shrink and nothing else "
            "regresses.",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


def _smooth_motion(diagnosis, finding, fix, context):
    subject = _motion_subject(finding)
    if subject is None:
        return [], "the jitter belongs to joints or shots: smooth their channels"
    return [
        Draft(
            f"Smooth {subject['name']!r}'s keys",
            [finding],
            [{"action": "smooth_motion", "object": subject["id"], "sigma": 0.05}],
            "approval",
            "a Gaussian average of its key values over 0.05 s removes motion above about 3 "
            "Hz; a shake or a vibration may be wanted: approve it if not.",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


def _make_cyclic(diagnosis, finding, fix, context):
    subject = _motion_subject(finding)
    if subject is None:
        return [], "the seam belongs to joints: make their clip's channels cyclic"
    return [
        Draft(
            f"Loop {subject['name']!r}'s channels",
            [finding],
            [{"action": "make_cyclic", "object": subject["id"], "mode": "cycle"}],
            "verify",
            "cyclic extrapolation makes the automatic handles of the end keys see across the "
            "seam, so velocity continues into the next loop. Kept only if the seam shrinks.",
            expected=_expected(finding),
            validation=[f"re-check {finding.check!r}"],
        )
    ], None


# --------------------------------------------------------------------------- #
# Lighting design (the lighting check)
# --------------------------------------------------------------------------- #

LIGHT_STEPS = 2.0 ** (np.arange(-12, 13) / 4.0)
"""Intensity factors tried for a light: 1/8 to 8 in quarter stops (max_light_factor)."""


def _lighting_of(diagnosis, finding):
    """The check and the lighting measurement a lighting finding came from (cached on its
    observation), or ``(None, None)``."""
    from .lighting import DEFAULT_SEPARATION, measure

    check = _check_of(diagnosis, finding)
    obs = diagnosis.observed.get(finding.view)
    if check is None or obs is None:
        return None, None
    scene = diagnosis.scene
    objs = [
        scene._objects[s["id"]]
        for s in finding.subjects
        if s.get("role") == "subject" and s.get("id") in scene._objects
    ]
    if not objs:
        return None, None
    sep = check.params.get("min_separation", DEFAULT_SEPARATION)
    return check, measure(obs, objs, min_separation=sep)


def _scene_lights(diagnosis, lighting):
    """``(column, light object)`` of the measured lights that are scene objects."""
    out = []
    for k, info in enumerate(lighting.lights):
        obj = diagnosis.scene._objects.get(info.get("id") or "")
        if obj is not None and obj.light is not None and obj.light.intensity > 0.0:
            out.append((k, obj))
    return out


def _side_ratio(total: np.ndarray, lit: np.ndarray, shadow: np.ndarray) -> float:
    hi, lo = float(np.median(total[lit])), float(np.median(total[shadow]))
    return math.inf if lo <= 1e-9 * hi else hi / lo


def _subject_frame(diagnosis, finding, lighting):
    """The subject's world-box centre and size (its parts included), the ids of its
    parts, and unit vectors toward the camera (horizontal) and to the camera's left."""
    from ..spatial.index import get_index
    from .observe import parts_of

    scene = diagnosis.scene
    index = get_index(scene)
    lo, hi, ids = np.full(3, np.inf), np.full(3, -np.inf), []
    for s in finding.subjects:
        obj = scene._objects.get(s.get("id") or "") if s.get("role") == "subject" else None
        for part in [] if obj is None else parts_of(obj):
            entry = index.by_id.get(part.id)
            if entry is not None:
                lo, hi = np.minimum(lo, entry.box[0]), np.maximum(hi, entry.box[1])
                ids.append(part.id)
    if not ids:  # (no geometry indexed: the measured surface)
        lo, hi = lighting.points.min(axis=0), lighting.points.max(axis=0)
    centre = 0.5 * (lo + hi)
    size = max(float(np.max(hi - lo)), 1e-3)
    toward = lighting.eye - centre
    toward[2] = 0.0
    if np.linalg.norm(toward) < 1e-9:
        toward = np.array([0.0, -1.0, 0.0])
    toward /= np.linalg.norm(toward)
    left = np.cross(toward, np.array([0.0, 0.0, 1.0]))
    return centre, size, ids, toward, left


def _unique_light_name(scene, base: str) -> str:
    name, k = base, 1
    while name in scene._by_name:
        k += 1
        name = f"{base} {k}"
    return name


def _lighting_ratio(diagnosis, finding, fix, context):
    check, lighting = _lighting_of(diagnosis, finding)
    if lighting is None or not lighting.count:
        return [], "the lighting measurement is not available"
    lo, hi = check.params["ratio"]["min"], check.params["ratio"]["max"]
    high = finding.code == "LIGHTING_RATIO_HIGH"
    target = hi / 1.15 if high else lo * 1.15
    if lo is not None and hi is not None and not lo <= target <= hi:
        target = math.sqrt(lo * hi)
    lit, shadow, _method = lighting.sides()
    total = lighting.total
    current = _side_ratio(total, lit, shadow)
    key = lighting.key_index()

    def score(r: float) -> float:
        return abs(math.log(r / target)) if math.isfinite(r) and r > 0.0 else math.inf

    def inside(r: float, margin: float = 1.0) -> bool:
        return (lo is None or r >= lo * margin) and (hi is None or r <= hi / margin)

    def rank(r: float, is_key: bool, f: float) -> tuple:
        # A ratio safely inside the range first; then the fill rather than the key (the
        # subject's exposure moves less), then the smallest change; else the closest.
        if inside(r, 1.05):
            return (0, is_key, abs(math.log(f)), 0.0)
        return (1, False, score(r), abs(math.log(f)))

    # Each light's (and the ambient's) direct contribution scales with its intensity: the
    # ratio after a change is predicted from the measured contributions.
    best = None
    for k, obj in _scene_lights(diagnosis, lighting):
        c = lighting.contribution[:, k]
        if not (c > 0.0).any():
            continue
        for f in LIGHT_STEPS:
            r = _side_ratio(total + (f - 1.0) * c, lit, shadow)
            here = rank(r, k == key, float(f))
            if best is None or here < best[0]:
                best = (here, obj, float(f), r)
    scene = diagnosis.scene
    ambient_ok = scene.world.environment.type != "image" and scene.world.ambient_intensity > 0
    ambient = None
    if ambient_ok and (lighting.ambient > 0.0).any():
        for f in LIGHT_STEPS:
            r = _side_ratio(total + (f - 1.0) * lighting.ambient, lit, shadow)
            here = rank(r, False, float(f))
            if ambient is None or here < ambient[0]:
                ambient = (here, None, float(f), r)
    key_name = None if key is None else lighting.lights[key]["name"]
    drafts = []
    for choice in (best, ambient):
        if choice is None or not inside(choice[3]):
            continue
        _rank, obj, f, r = choice
        if obj is not None:
            new = obj.light.intensity * f
            title = f"Set {obj.name!r}'s intensity to {new:.4g} (x{f:.3g})"
            actions = [{"action": "set_light", "light": obj.id, "intensity": round(new, 6)}]
        else:
            new = scene.world.ambient_intensity * f
            title = f"Set the ambient intensity to {new:.4g} (x{f:.3g})"
            actions = [{"action": "set_ambient", "intensity": round(new, 6)}]
        drafts.append(
            Draft(
                title,
                [finding],
                actions,
                "verify",
                f"the measured direct light predicts a lighting ratio of {r:.3g}:1 after "
                f"this change ({current:.3g}:1 now, target {target:.3g}:1; the key, the light "
                f"giving the subject the most, is {key_name!r}); verification re-renders and "
                "re-measures",
                expected=_expected(finding),
                validation=[f"re-check {finding.check!r}", "no new failure"],
                group="lighting_ratio",
            )
        )
        break  # (the light first; the ambient, which changes every object, only without)
    if drafts:
        return drafts, None
    if not high:
        return [], (
            "no light or ambient change reaches the ratio: the key lights the subject from "
            "the camera's side (move it, e.g. scene.light_rig with another preset)"
        )
    return _add_fill(diagnosis, finding, lighting, lit, shadow, target, key)


def _add_fill(diagnosis, finding, lighting, lit, shadow, target, key):
    """A soft fill on the side opposite the key, its intensity solved for the target ratio
    (unshadowed: needs approval and verification)."""
    from ..rendering.scene_data import RenderLight
    from .lighting import _luma, unshadowed

    scene = diagnosis.scene
    centre, size, _ids, toward, left = _subject_frame(diagnosis, finding, lighting)
    side = -1.0
    if key is not None:
        obj = scene._objects.get(lighting.lights[key].get("id") or "")
        if obj is not None and obj.light is not None:
            if obj.light.type == "directional":
                towards_key = -np.asarray(obj.world_matrix[:3, :3] @ [0.0, -1.0, 0.0])
            else:
                towards_key = np.asarray(obj.world_matrix[:3, 3]) - centre
            side = -1.0 if float(towards_key @ left) >= 0.0 else 1.0
    d = 1.25 * size
    a, e = math.radians(50.0 * side), math.radians(15.0)
    direction = math.cos(e) * (math.cos(a) * toward + math.sin(a) * left)
    pos = centre + d * (direction + np.array([0.0, 0.0, math.sin(e)]))
    w = 0.75 * d
    normal = (centre - pos) / np.linalg.norm(centre - pos)
    ax = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    ax = ax / max(np.linalg.norm(ax), 1e-12)
    ay = np.cross(ax, normal)
    probe = RenderLight(None, "", "area", np.ones(3), pos, normal, None, "inverse_square",
                        axis_x=ax * w / 2, axis_y=ay * w / 2, shape="square",
                        area=w * w)  # fmt: skip
    u = _luma(unshadowed(probe, lighting.points, lighting.normals))
    if not (u[shadow] > 0.0).any():
        return [], "no fill position reaches the shadow side"
    guess = max(float(np.median(lighting.total[lit])) / target, 1e-6) / max(
        float(np.median(u[shadow])), 1e-12
    )
    best = None
    for f in 2.0 ** (np.arange(-16, 17) / 4.0):
        r = _side_ratio(lighting.total + guess * f * u, lit, shadow)
        rank = abs(math.log(r / target)) if math.isfinite(r) else math.inf
        if best is None or rank < best[0]:
            best = (rank, guess * f, r)
    name = _unique_light_name(scene, "Feedback fill light")
    return [
        Draft(
            "Add a soft fill light opposite the key",
            [finding],
            [
                {
                    "action": "add_light",
                    "type": "area",
                    "name": name,
                    "position": [round(float(v), 6) for v in pos],
                    "look_at": [round(float(v), 6) for v in centre],
                    "size": round(float(w), 6),
                    "intensity": round(float(best[1]), 6),
                }
            ],
            "approval",
            f"no existing light fills the shadow side; a square area light ({w:.3g} m) "
            f"on the side opposite the key, intensity solved for {target:.3g}:1 "
            f"(predicted {best[2]:.3g}:1 without its shadows; a choice: needs approval)",
            expected=_expected(finding, None),
            validation=[f"re-check {finding.check!r}", "no new failure"],
        )
    ], None


def _sector_means(lighting, xy, values) -> np.ndarray:
    """Per outline sector: the mean of ``values`` over the points there (``(S, ...)``)."""
    from .lighting import SECTORS, sector_of

    sec = sector_of(xy, lighting.separation["centroid"])
    count = np.maximum(np.bincount(sec, minlength=SECTORS), 1)
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        return np.bincount(sec, weights=values, minlength=SECTORS) / count
    return np.stack(
        [np.bincount(sec, weights=values[:, k], minlength=SECTORS) / count
         for k in range(values.shape[1])],
        axis=1,
    ) if values.shape[1] else np.zeros((SECTORS, 0))  # fmt: skip


def _predicted_merged(rows, add_in, add_out, min_sep: float) -> float:
    """The merged fraction after each sector's inside and outside display luminance grow
    by ``add_in`` and ``add_out`` (``(S,)``; light adds to what the image shows)."""
    from .lighting import SECTORS, lstar_to_y, y_to_lstar

    merged = valid = 0
    for row in rows:
        s = min(int(row["angle"] // (360.0 / SECTORS)), SECTORS - 1)
        y_in = float(lstar_to_y(row["inside"])) + add_in[s]
        y_out = float(lstar_to_y(row["outside"])) + add_out[s]
        valid += row["pixels"]
        if abs(float(y_to_lstar(y_in)) - float(y_to_lstar(y_out))) < min_sep:
            merged += row["pixels"]
    return merged / valid if valid else 0.0


def _separation(diagnosis, finding, fix, context):
    check, lighting = _lighting_of(diagnosis, finding)
    sep = None if lighting is None else lighting.separation
    if sep is None or not lighting.count:
        return [], "the subject's outline was not measured"
    limit = check.params.get("max_merged_fraction", 0.25)
    min_sep = sep["min_separation"]
    rows = sep["sectors"]
    current = sep["merged_fraction"]
    # A diffuse surface shows albedo x irradiance x 2^exposure: what a light change adds to
    # each sector's inside and outside is predicted from the measured contributions (the
    # diffuse part: metals' and glass's highlights are not predicted).
    gain = 2.0**diagnosis.scene.render_settings.exposure
    ring = lighting.on_outline
    lit_in = (
        _sector_means(
            lighting, lighting.xy[ring], lighting.albedo[ring, None] * lighting.contribution[ring]
        )
        * gain
    )
    sur = lighting.surround
    if sur is not None and sur["xy"].shape[0]:
        lit_out = (
            _sector_means(lighting, sur["xy"], sur["albedo"][:, None] * sur["contribution"]) * gain
        )
    else:
        lit_out = np.zeros_like(lit_in)  # fmt: skip
    best = None
    for k, obj in _scene_lights(diagnosis, lighting):
        for f in LIGHT_STEPS:
            m = _predicted_merged(rows, (f - 1.0) * lit_in[:, k], (f - 1.0) * lit_out[:, k],
                                  min_sep)  # fmt: skip
            rank = (m, abs(math.log(f)))
            if best is None or rank < best[0]:
                best = (rank, obj, float(f), m)
    if best is not None and best[3] <= limit and best[3] < current - 0.02:
        _rank, obj, f, m = best
        new = obj.light.intensity * f
        return [
            Draft(
                f"Set {obj.name!r}'s intensity to {new:.4g} (x{f:.3g})",
                [finding],
                [{"action": "set_light", "light": obj.id, "intensity": round(new, 6)}],
                "verify",
                f"the light's measured contribution to the outline and to what surrounds it "
                f"predicts {m:.0%} of the outline merged after this change ({current:.0%} "
                "now); verification re-renders and re-measures",
                expected=_expected(finding),
                validation=[f"re-check {finding.check!r}", "no new failure"],
                group="separation",
            )
        ], None
    return _add_rim(diagnosis, finding, lighting, rows, gain, min_sep, current)


RIM_PLACES = [(a, e) for a in (150.0, -150.0, 120.0, -120.0, 180.0) for e in (10.0, 35.0, 60.0)]
"""Where a rim light is tried: (azimuth from the camera's side, positive to its left;
elevation), in degrees."""


def _add_rim(diagnosis, finding, lighting, rows, gain, min_sep, current):
    """A spot light behind the subject aimed at it (a rim): the place and intensity with
    the least predicted merging (unshadowed: needs approval and verification)."""
    from ..rendering.scene_data import RenderLight
    from .lighting import _luma, unshadowed

    scene = diagnosis.scene
    centre, size, ids, toward, left = _subject_frame(diagnosis, finding, lighting)
    ring = lighting.on_outline
    sur = lighting.surround
    has_sur = sur is not None and sur["xy"].shape[0] > 0
    key = lighting.key_index()
    level = float(lighting.contribution[:, key].mean()) if key is not None else 1.0
    best = None
    for az, el in RIM_PLACES:
        a, e = math.radians(az), math.radians(el)
        direction = math.cos(e) * (math.cos(a) * toward + math.sin(a) * left)
        direction = direction + np.array([0.0, 0.0, math.sin(e)])
        d = 1.5 * size
        hit = scene.raycast(centre.tolist(), direction.tolist(), max_distance=d,
                            exclude=ids or None)  # fmt: skip
        if hit is not None:  # keep the light in front of what stands behind the subject
            d = 0.8 * hit.distance
        if d < 0.6 * size:
            continue
        pos = centre + d * direction
        half = min(60.0, max(10.0, math.degrees(math.atan(0.75 * size / d))))
        cos_outer = math.cos(math.radians(half))
        aim = (centre - pos) / np.linalg.norm(centre - pos)
        probe = RenderLight(None, "", "spot", np.ones(3), pos, aim, None, "inverse_square",
                            cos_outer=cos_outer, spot_smooth=(1.0 - cos_outer) * 0.15)  # fmt: skip
        u_in = _luma(unshadowed(probe, lighting.points[ring], lighting.normals[ring]))
        if not (u_in > 0.0).any():
            continue
        p_in = _sector_means(lighting, lighting.xy[ring], lighting.albedo[ring] * u_in) * gain
        p_out = np.zeros_like(p_in)
        if has_sur:
            u_out = _luma(unshadowed(probe, sur["points"], sur["normals"]))
            p_out = _sector_means(lighting, sur["xy"], sur["albedo"] * u_out) * gain
        guess = max(level, 1e-6) / max(float(u_in[u_in > 0.0].mean()), 1e-12)
        for f in 2.0 ** (np.arange(-12, 13) / 4.0):
            intensity = guess * f
            m = _predicted_merged(rows, intensity * p_in, intensity * p_out, min_sep)
            rank = (m, abs(math.log(f)))
            if best is None or rank < best[0]:
                best = (rank, intensity, m, pos, d, az, el)
    if best is None or best[2] >= current - 0.02:
        return [], (
            "no light change predicts a better separation: change the background's tone "
            "or the subject's material, or light what surrounds it (a decision)"
        )
    _rank, intensity, m, pos, d, az, el = best
    name = _unique_light_name(scene, "Feedback rim light")
    side = "left" if az > 0 else "right" if az < 0 else "back"
    return [
        Draft(
            "Add a rim light behind the subject",
            [finding],
            [
                {
                    "action": "add_light",
                    "type": "spot",
                    "name": name,
                    "position": [round(float(v), 6) for v in pos],
                    "look_at": [round(float(v), 6) for v in centre],
                    "intensity": round(float(intensity), 6),
                }
            ],
            "approval",
            f"a spot light behind the subject on the camera's {side} ({el:.0f} degrees up, "
            f"{d:.3g} m away) predicts {m:.0%} of the outline merged ({current:.0%} now), "
            "without its shadows; a choice: needs approval",
            expected=_expected(finding, None),
            validation=[f"re-check {finding.check!r}", "no new failure"],
        )
    ], None


def _unlit(diagnosis, finding, fix, context):
    return _add_light(diagnosis, finding)


PLANNERS = {
    "ACCELERATION_STEP": _curve_smoothing,
    "VELOCITY_JUMP": _curve_smoothing,
    "JITTER": _smooth_motion,
    "CAMERA_JITTER": _smooth_motion,
    "LOOP_SEAM": _make_cyclic,
    "FOOT_SLIDING": _plant_feet,
    "FOOT_PENETRATION": _plant_feet,
    "MOTION_DISCONTINUITY": _smooth_keys,
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
    "LIGHTING_RATIO_HIGH": _lighting_ratio,
    "LIGHTING_RATIO_LOW": _lighting_ratio,
    "POOR_SEPARATION": _separation,
    "UNLIT": _unlit,
}
