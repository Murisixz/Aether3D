"""The scene's animation document (``scene.animation``).

It holds **items** -- clips (reusable animation data) and bindings (what animates each
object or material: an action and layers of strips) -- and the **timeline** settings
(frame rate, start and end, markers, sections). Items have immutable definitions:
every edit replaces one, as one step of the scene's history (one aspect per item, like
the CAD document), so undo and redo restore them exactly and caches keyed by them stay
valid. ``with scene.animation.batch("label"):`` makes several edits one step.

The **current time** (``scene.animation.time``) is the playhead: renders, spatial queries
and observations of an animated scene evaluate there (unless given ``time=``). It is
runtime state -- not saved, not recorded, no new revision -- because moving it changes
nothing authored. Evaluation (:meth:`AnimationSystem.evaluate`) is a pure function of
the scene and the time, cached per (scene revision, time).

Animation never writes the scene while it plays: the authored transforms and properties
are the rest pose that bindings animate. :meth:`AnimationSystem.apply_pose` is the one
explicit operation that writes an evaluated pose into the scene (one undo step).
"""

from __future__ import annotations

import copy
import json
import math
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import AnimationError
from . import paths as P
from . import rotation as R
from .clip import Clip, Event, edit_curve, insert_key, parse_event
from .curves import FCurve
from .evaluate import AnimationPose, Program, compile_program, evaluate
from .nla import Binding, Layer, Strip, parse_binding, parse_layer, parse_strip
from .time import FrameRate, check_time
from .tracks import RotationTrack

if TYPE_CHECKING:
    from ..scene import Scene, SceneObject

MAX_ITEMS = 100_000
MAX_MARKERS = 10_000
EVALUATION_CACHE = 64

DEFAULT_SETTINGS: dict[str, Any] = {
    "fps": 24,
    "start": 0.0,
    "end": 10.0,
    "markers": [],
    "sections": [],
    "enabled": True,
}

KINDS = ("clip", "binding", "rig", "skin", "constraints", "morph", "path", "sequence")


def _err(code: str, message: str, location: str | None = None, **kw: Any) -> AnimationError:
    return AnimationError(code, message, location, **kw)


class AnimItem:
    """A handle on an animation item (a clip or a binding); stays valid across undo."""

    __slots__ = ("_system", "id", "kind", "_name", "_definition", "_alive", "_serial")

    def __init__(
        self, system: AnimationSystem, iid: str, kind: str, name: str, definition: Any
    ) -> None:
        self._system = system
        self.id = iid
        self.kind = kind
        self._name = name
        self._definition = definition
        self._alive = True
        self._serial = system._next_serial()

    @property
    def name(self) -> str:
        return self._name

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def definition(self) -> Any:
        """The immutable definition (a :class:`~.clip.Clip` or a
        :class:`~.nla.Binding`)."""
        return self._definition

    def __repr__(self) -> str:
        return f"<animation {self.kind} {self._name!r}>"


class AnimationSystem:
    """A scene's animation document; see the module docstring."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._items: dict[str, AnimItem] = {}
        self._clip_names: dict[str, AnimItem] = {}
        self._bindings: dict[tuple[str, str], AnimItem] = {}
        self._owned: dict[tuple[str, str], AnimItem] = {}  # (kind, object id) -> item
        self._path_names: dict[str, AnimItem] = {}
        self._settings: dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
        self._serial = 0
        self._time = 0.0
        self.revision = 0
        self._compiled: Program | None = None
        self._evaluations: OrderedDict[tuple[int, str], AnimationPose] = OrderedDict()
        self._depth = 0

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial

    @contextmanager
    def batch(self, label: str = "Animation change") -> Iterator[AnimationSystem]:
        """Several changes as one atomic step of the scene's history."""
        with self.scene._history._op(label):
            yield self

    def _op(self, label: str):
        return self.scene._history._op(label)

    def _touch(self, item: AnimItem | None = None) -> None:
        from ..history import ANIMATION_ITEM, ANIMATION_SETTINGS

        if item is None:
            self.scene._history._touch(ANIMATION_SETTINGS, self)
        else:
            self.scene._history._touch(ANIMATION_ITEM, item)

    def _changed(self) -> None:
        self.revision += 1
        self._compiled = None
        self._evaluations.clear()

    def _set(
        self,
        item: AnimItem,
        *,
        definition: Any = None,
        name: str | None = None,
        alive: bool | None = None,
    ) -> None:
        self._touch(item)
        if definition is not None:
            item._definition = definition
        if name is not None:
            item._name = name
        if alive is not None:
            item._alive = alive
        self._reindex()
        self._changed()

    def _reindex(self) -> None:
        names: dict[str, AnimItem] = {}
        path_names: dict[str, AnimItem] = {}
        bindings: dict[tuple[str, str], AnimItem] = {}
        owned: dict[tuple[str, str], AnimItem] = {}
        for item in self._items.values():
            if not item._alive:
                continue
            if item.kind == "clip":
                if item._name in names:
                    raise _err("DUPLICATE_NAME", f"two clips would be named {item._name!r}")
                names[item._name] = item
            elif item.kind == "path":
                if item._name in path_names:
                    raise _err("DUPLICATE_NAME", f"two paths would be named {item._name!r}")
                path_names[item._name] = item
            elif item.kind == "binding":
                key = item._definition.target
                if key in bindings:
                    raise _err("DUPLICATE_BINDING", f"two bindings would animate {key[0]} {key[1]}")
                bindings[key] = item
            else:  # rig, skin, constraints: one per object
                key = (item.kind, _owner_of(item))
                if key in owned:
                    raise _err("DUPLICATE_BINDING", f"two {item.kind} items for object {key[1]}")
                owned[key] = item
        self._clip_names = names
        self._path_names = path_names
        self._bindings = bindings
        self._owned = owned

    def _restored(self) -> None:
        """After undo, redo or a rollback (history fixup)."""
        self._reindex()
        self._changed()

    def _new_item(self, kind: str, name: str, definition: Any) -> AnimItem:
        if sum(1 for i in self._items.values() if i._alive) >= MAX_ITEMS:
            raise _err("LIMIT_EXCEEDED", f"more than {MAX_ITEMS} animation items")
        item = AnimItem(self, str(uuid.uuid4()), kind, name, definition)
        item._alive = False
        self._items[item.id] = item
        self._set(item, alive=True)
        return item

    def _binding_items(self) -> list[AnimItem]:
        return sorted(self._bindings.values(), key=lambda i: i._serial)

    # ------------------------------------------------------------------ #
    # Timeline settings
    # ------------------------------------------------------------------ #

    @property
    def fps(self) -> FrameRate:
        return FrameRate(self._settings["fps"])

    @property
    def start(self) -> float:
        return float(self._settings["start"])

    @property
    def end(self) -> float:
        return float(self._settings["end"])

    @property
    def enabled(self) -> bool:
        return bool(self._settings["enabled"])

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise _err("INVALID_ARGUMENT", "enabled is true or false", "enabled")
        self._set_settings({"enabled": value}, "Enable animation" if value else "Mute animation")

    def set_timeline(self, *, fps: Any = None, start: Any = None, end: Any = None) -> None:
        """Change the frame rate and/or the range (seconds); one undo step."""
        changes: dict[str, Any] = {}
        if fps is not None:
            changes["fps"] = FrameRate(fps).to_json()
        s = check_time(start, "start") if start is not None else self.start
        e = check_time(end, "end") if end is not None else self.end
        if not s < e:
            raise _err("INVALID_RANGE", "the timeline's start must be before its end", "start")
        if start is not None:
            changes["start"] = s
        if end is not None:
            changes["end"] = e
        if changes:
            self._set_settings(changes, "Set timeline")

    def _set_settings(self, changes: dict[str, Any], label: str) -> None:
        new = {**self._settings, **changes}
        with self._op(label):
            self._touch()
            self._settings = new
            self._changed()

    @property
    def markers(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self._settings["markers"]]

    def set_markers(self, markers: list[Any]) -> None:
        """Replace the timeline markers (``[{time, name, data}]``)."""
        if not isinstance(markers, list) or len(markers) > MAX_MARKERS:
            raise _err("INVALID_ARGUMENT", f"markers is a list of at most {MAX_MARKERS}", "markers")
        evs = sorted(
            (parse_event(m, f"markers[{i}]") for i, m in enumerate(markers)),
            key=lambda e: (e.time, e.name),
        )
        self._set_settings({"markers": [e.to_json() for e in evs]}, "Set markers")

    def add_marker(self, name: str, time: float, data: Any = None) -> None:
        self.set_markers([*self._settings["markers"], {"name": name, "time": time, "data": data}])

    @property
    def sections(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self._settings["sections"]]

    def set_sections(self, sections: list[Any]) -> None:
        """Named time ranges (``[{name, start, end}]``): shots, beats, acts."""
        if not isinstance(sections, list) or len(sections) > MAX_MARKERS:
            raise _err(
                "INVALID_ARGUMENT", f"sections is a list of at most {MAX_MARKERS}", "sections"
            )
        out = []
        for i, s in enumerate(sections):
            loc = f"sections[{i}]"
            if not isinstance(s, dict) or set(s) != {"name", "start", "end"}:
                raise _err("INVALID_ARGUMENT", "a section is {name, start, end}", loc)
            a, b = check_time(s["start"], f"{loc}.start"), check_time(s["end"], f"{loc}.end")
            if not a < b:
                raise _err("INVALID_RANGE", "a section's start must be before its end", loc)
            out.append(
                {
                    "name": P.check_name(s["name"], "a section name", f"{loc}.name"),
                    "start": a,
                    "end": b,
                }
            )
        self._set_settings(
            {"sections": sorted(out, key=lambda s: (s["start"], s["name"]))}, "Set sections"
        )

    # ------------------------------------------------------------------ #
    # The current time (runtime only)
    # ------------------------------------------------------------------ #

    @property
    def time(self) -> float:
        """The playhead in seconds (not saved, not recorded; see the module docstring)."""
        return self._time

    @time.setter
    def time(self, value: float) -> None:
        self._time = check_time(value, "time")

    @property
    def frame(self) -> int:
        return self.fps.frame_at(self._time, "floor")

    @frame.setter
    def frame(self, value: int) -> None:
        self._time = self.fps.frame_time(value)

    def frame_time(self, frame: int) -> float:
        return self.fps.frame_time(frame)

    # ------------------------------------------------------------------ #
    # References
    # ------------------------------------------------------------------ #

    def _target(self, ref: Any, location: str = "target") -> tuple[str, str]:
        """``("object" | "material", id)`` of a scene object, material, id or name; a
        ``{"material": ...}`` dict names a material explicitly."""
        from ..materials import Material
        from ..scene import SceneObject

        scene = self.scene
        if isinstance(ref, SceneObject):
            if ref.scene is not scene:
                raise _err("TARGET_MISSING", f"object {ref.name!r} is not in this scene", location)
            return (P.OBJECT, ref.id)
        if isinstance(ref, Material):
            if scene.materials.find_material(ref.id) is not ref:
                raise _err(
                    "TARGET_MISSING", f"material {ref.name!r} is not in this scene", location
                )
            return (P.MATERIAL, ref.id)
        if isinstance(ref, dict) and set(ref) in ({"object"}, {"material"}):
            kind = next(iter(ref))
            return (
                self._target(ref[kind], location)
                if kind == "object"
                else self._material(ref[kind], location)
            )
        if isinstance(ref, (list, tuple)) and len(ref) == 2 and ref[0] in (P.OBJECT, P.MATERIAL):
            return self._target({ref[0]: ref[1]}, location)
        if isinstance(ref, str):
            obj = scene._objects.get(ref) or scene._by_name.get(ref)
            if obj is not None:
                return (P.OBJECT, obj.id)
            raise _err(
                "TARGET_MISSING",
                f"no object {ref!r} (by id or name)",
                location,
                hint='name a material as {"material": NAME}',
            )
        raise _err("TARGET_MISSING", f"cannot animate {ref!r}", location)

    def _material(self, ref: Any, location: str) -> tuple[str, str]:
        m = self.scene.materials.find_material(ref) if isinstance(ref, str) else None
        if m is None:
            raise _err("TARGET_MISSING", f"no material {ref!r}", location)
        return (P.MATERIAL, m.id)

    def _clip_item(self, ref: Any, location: str = "clip") -> AnimItem:
        if isinstance(ref, AnimItem) and ref.kind == "clip" and ref._alive:
            return ref
        if isinstance(ref, ClipHandle):
            return ref._item
        if isinstance(ref, str):
            item = self._items.get(ref)
            if item is not None and item.kind == "clip" and item._alive:
                return item
            item = self._clip_names.get(ref)
            if item is not None:
                return item
        raise _err("CLIP_MISSING", f"no clip {ref!r} (by id or name)", location)

    def _clip_id(self, ref: Any, location: str) -> str:
        return self._clip_item(ref, location).id

    # ------------------------------------------------------------------ #
    # Clips
    # ------------------------------------------------------------------ #

    def clips(self) -> list[ClipHandle]:
        return [
            ClipHandle(self, i) for i in sorted(self._clip_names.values(), key=lambda i: i._serial)
        ]

    def clip(self, ref: Any) -> ClipHandle:
        return ClipHandle(self, self._clip_item(ref))

    def add_clip(self, name: str, data: Any = None, *, target: str = P.OBJECT) -> ClipHandle:
        """A new clip (``data``: a :class:`~.clip.Clip` or its JSON; empty by default)."""
        P.check_name(name, "a clip name", "name")
        if name in self._clip_names:
            raise _err("DUPLICATE_NAME", f"a clip is already named {name!r}", "name")
        clip = _clip_from(data, target, "clip")
        with self._op(f"Add clip {name!r}"):
            item = self._new_item("clip", name, clip)
        return ClipHandle(self, item)

    def _unique_clip_name(self, base: str) -> str:
        name = base[: P.MAX_NAME]
        k = 1
        while name in self._clip_names:
            k += 1
            suffix = f".{k:03d}"
            name = base[: P.MAX_NAME - len(suffix)] + suffix
        return name

    def remove_clip(self, ref: Any) -> None:
        """Remove a clip; strips and actions that play it are removed with it."""
        item = self._clip_item(ref)
        with self._op(f"Remove clip {item._name!r}"):
            for b in list(self._bindings.values()):
                binding = b._definition
                if item.id not in binding.clips():
                    continue
                layers = tuple(
                    Layer(layer.name, layer.mode, layer.weight, layer.weight_curve, layer.mask,
                          tuple(s for s in layer.strips if s.clip != item.id), layer.mute)
                    for layer in binding.layers
                )  # fmt: skip
                action = None if binding.action == item.id else binding.action
                self._set(b, definition=Binding(binding.target, action, layers, binding.mute))
            self._set(item, alive=False)

    def _set_clip(self, item: AnimItem, clip: Clip, label: str) -> None:
        with self._op(label):
            self._set(item, definition=clip)

    # ------------------------------------------------------------------ #
    # Bindings, actions, layers, strips
    # ------------------------------------------------------------------ #

    def binding(self, target: Any) -> Binding | None:
        item = self._bindings.get(self._target(target))
        return None if item is None else item._definition

    def bindings(self) -> list[tuple[tuple[str, str], Binding]]:
        return [(i._definition.target, i._definition) for i in self._binding_items()]

    def _binding_item(self, key: tuple[str, str], create: bool) -> AnimItem | None:
        item = self._bindings.get(key)
        if item is None and create:
            name = self._target_name(key)
            item = self._new_item("binding", name, Binding(key))
        return item

    def _target_name(self, key: tuple[str, str]) -> str:
        kind, tid = key
        if kind == P.OBJECT:
            return self.scene._objects[tid].name
        m = self.scene.materials.find_material(tid)
        return m.name if m is not None else tid

    def set_binding(self, target: Any, data: Any) -> None:
        """Replace what animates ``target`` (``{"action", "layers", "mute"}``; ``None``
        removes the binding)."""
        key = self._target(target)
        with self._op(f"Animate {self._target_name(key)!r}"):
            item = self._binding_item(key, create=data is not None)
            if data is None:
                if item is not None:
                    self._set(item, alive=False)
                return
            raw = dict(data) if isinstance(data, dict) else data
            if isinstance(raw, dict):
                raw = {**raw, key[0]: key[1]}
            binding = parse_binding(raw, "binding", self._clip_id)
            if binding.target != key:
                raise _err(
                    "INVALID_BINDING", "a binding's target is the one it is set on", "binding"
                )
            self._check_binding_clips(binding)
            self._set(item, definition=binding)

    def _check_binding_clips(self, binding: Binding) -> None:
        for cid in binding.clips():
            clip = self._items[cid]._definition
            if clip.target != binding.target[0]:
                raise _err(
                    "CLIP_TARGET_MISMATCH",
                    f"clip {self._items[cid]._name!r} animates {clip.target}s, "
                    f"not {binding.target[0]}s",
                    "binding",
                )

    def action(self, target: Any) -> ClipHandle | None:
        b = self.binding(target)
        if b is None or b.action is None:
            return None
        return ClipHandle(self, self._items[b.action])

    def set_action(self, target: Any, clip: Any) -> None:
        """Play ``clip`` as ``target``'s action (``None`` clears it)."""
        key = self._target(target)
        cid = None if clip is None else self._clip_id(clip, "clip")
        with self._op(f"Set action of {self._target_name(key)!r}"):
            item = self._binding_item(key, create=True)
            assert item is not None
            b = item._definition
            new = Binding(b.target, cid, b.layers, b.mute)
            self._check_binding_clips(new)
            self._set(item, definition=new)

    def add_layer(
        self,
        target: Any,
        name: str,
        *,
        mode: str = "replace",
        weight: float = 1.0,
        mask: Any = None,
        index: int | None = None,
    ) -> None:
        """A new (empty) layer above the others (or at ``index``)."""
        key = self._target(target)
        layer = parse_layer(
            {"name": name, "mode": mode, "weight": weight, "mask": mask or []}, "layer"
        )
        with self._op(f"Add layer {name!r}"):
            item = self._binding_item(key, create=True)
            assert item is not None
            b = item._definition
            if any(layer.name == name for layer in b.layers):
                raise _err("DUPLICATE_NAME", f"a layer is already named {name!r}", "name")
            layers = list(b.layers)
            layers.insert(len(layers) if index is None else int(index), layer)
            if len(layers) > 64:
                raise _err("LIMIT_EXCEEDED", "at most 64 layers", "layers")
            self._set(item, definition=Binding(b.target, b.action, tuple(layers), b.mute))

    def update_layer(self, target: Any, name: str, **changes: Any) -> None:
        """Change a layer's ``mode``, ``weight``, ``weight_curve``, ``mask``, ``mute`` or
        ``name``."""
        key = self._target(target)
        item = self._bindings.get(key)
        if item is None:
            raise _err("LAYER_MISSING", f"{self._target_name(key)!r} has no layers", "layer")
        b = item._definition
        layers = list(b.layers)
        idx = next((i for i, layer in enumerate(layers) if layer.name == name), None)
        if idx is None:
            raise _err("LAYER_MISSING", f"no layer {name!r}", "layer")
        raw = layers[idx].to_json()
        unknown = set(changes) - {"mode", "weight", "weight_curve", "mask", "mute", "name"}
        if unknown:
            raise _err("INVALID_ARGUMENT", f"unknown layer changes {sorted(unknown)}", "layer")
        raw.update(changes)
        raw["strips"] = list(layers[idx].strips)
        layers[idx] = parse_layer(raw, "layer", self._clip_id)
        with self._op(f"Change layer {name!r}"):
            self._set(item, definition=Binding(b.target, b.action, tuple(layers), b.mute))

    def remove_layer(self, target: Any, name: str) -> None:
        key = self._target(target)
        item = self._bindings.get(key)
        b = None if item is None else item._definition
        if b is None or not any(layer.name == name for layer in b.layers):
            raise _err("LAYER_MISSING", f"no layer {name!r}", "layer")
        with self._op(f"Remove layer {name!r}"):
            self._set(
                item,
                definition=Binding(
                    b.target, b.action, tuple(x for x in b.layers if x.name != name), b.mute
                ),
            )

    def add_strip(self, target: Any, layer: str, clip: Any, **options: Any) -> None:
        """Place ``clip`` in ``layer`` (see :class:`~.nla.Strip` for the options)."""
        key = self._target(target)
        item = self._bindings.get(key)
        b = None if item is None else item._definition
        if b is None or not any(x.name == layer for x in b.layers):
            raise _err("LAYER_MISSING", f"no layer {layer!r} (add_layer first)", "layer")
        strip = parse_strip(
            {"clip": clip if isinstance(clip, str) else self._clip_id(clip, "clip"), **options},
            "strip",
            self._clip_id,
        )
        new_layers = []
        for x in b.layers:
            if x.name == layer:
                if len(x.strips) >= 10_000:
                    raise _err("LIMIT_EXCEEDED", "at most 10000 strips per layer", "strips")
                x = Layer(
                    x.name, x.mode, x.weight, x.weight_curve, x.mask, (*x.strips, strip), x.mute
                )
            new_layers.append(x)
        new = Binding(b.target, b.action, tuple(new_layers), b.mute)
        self._check_binding_clips(new)
        with self._op(f"Add strip {strip.name!r}"):
            self._set(item, definition=new)

    def update_strip(self, target: Any, layer: str, index: int, **changes: Any) -> None:
        key = self._target(target)
        item = self._bindings.get(key)
        b = None if item is None else item._definition
        lay = None if b is None else next((x for x in b.layers if x.name == layer), None)
        if lay is None:
            raise _err("LAYER_MISSING", f"no layer {layer!r}", "layer")
        if not isinstance(index, int) or not -len(lay.strips) <= index < len(lay.strips):
            raise _err("STRIP_MISSING", f"no strip {index!r} in layer {layer!r}", "strip")
        raw = lay.strips[index].to_json()
        raw.update(changes)
        strip = parse_strip(raw, f"layers[{layer}].strips[{index}]", self._clip_id)
        strips = list(lay.strips)
        strips[index] = strip
        new_layers = tuple(
            Layer(x.name, x.mode, x.weight, x.weight_curve, x.mask, tuple(strips), x.mute)
            if x is lay
            else x
            for x in b.layers
        )
        new = Binding(b.target, b.action, new_layers, b.mute)
        self._check_binding_clips(new)
        with self._op(f"Change strip {strip.name!r}"):
            self._set(item, definition=new)

    def remove_strip(self, target: Any, layer: str, index: int) -> None:
        key = self._target(target)
        item = self._bindings.get(key)
        b = None if item is None else item._definition
        lay = None if b is None else next((x for x in b.layers if x.name == layer), None)
        if (
            lay is None
            or not isinstance(index, int)
            or not -len(lay.strips) <= index < len(lay.strips)
        ):
            raise _err("STRIP_MISSING", f"no strip {index!r} in layer {layer!r}", "strip")
        strips = list(lay.strips)
        del strips[index]
        new_layers = tuple(
            Layer(x.name, x.mode, x.weight, x.weight_curve, x.mask, tuple(strips), x.mute)
            if x is lay
            else x
            for x in b.layers
        )
        with self._op("Remove strip"):
            self._set(item, definition=Binding(b.target, b.action, new_layers, b.mute))

    # ------------------------------------------------------------------ #
    # Keyframing (the target's action)
    # ------------------------------------------------------------------ #

    def _action_item(self, key: tuple[str, str], create: bool) -> AnimItem | None:
        item = self._binding_item(key, create=create)
        if item is None:
            return None
        b = item._definition
        if b.action is not None:
            return self._items[b.action]
        if not create:
            return None
        name = self._unique_clip_name(f"{self._target_name(key)} Action")
        clip_item = self._new_item("clip", name, Clip(key[0]))
        self._set(item, definition=Binding(b.target, clip_item.id, b.layers, b.mute))
        return clip_item

    def keyframe(self, target: Any, path: str, time: float, value: Any, **options: Any) -> None:
        """Insert (or update) keys at ``time`` in ``target``'s action: one value for a
        property, a vector/colour for a grouped path (``transform.position``), a
        quaternion ``[x, y, z, w]`` for ``transform.rotation`` (or a joint's). Key
        options: ``interpolation``, ``ease``, ``left``, ``right``, ``weighted``, ... (for
        rotations: ``interpolation``, ``path``, ``timing``). One undo step."""
        key = self._target(target)
        specs = P.parse(path, key[0], grouped=True, location="path")
        time = check_time(time, "time")
        with self._op(f"Keyframe {path} of {self._target_name(key)!r}"):
            clip_item = self._action_item(key, create=True)
            assert clip_item is not None
            clip = clip_item._definition
            clip = _keyframe_into(clip, specs, time, value, options, "keyframe")
            self._set(clip_item, definition=clip)

    def set_channel(
        self, target: Any, path: str, keys: Any, *, clip: Any = None, **curve_options: Any
    ) -> None:
        """Replace a channel of ``target``'s action (or of ``clip``) with ``keys`` (key
        dicts or pairs; a :class:`FCurve`/:class:`RotationTrack`; ``None`` removes it)."""
        if clip is not None:
            item = self._clip_item(clip)
            target_kind = item._definition.target
        else:
            key = self._target(target)
            target_kind = key[0]
            item = None
        spec = P.parse(path, target_kind, location="path")
        assert not isinstance(spec, list)
        channel = _channel_from(spec, keys, curve_options, "keys")
        with self._op(f"Set channel {spec.path}"):
            if item is None:
                item = self._action_item(key, create=channel is not None)
                if item is None:
                    return
            self._set(item, definition=item._definition.with_channel(spec.path, channel))

    def channel(self, target: Any, path: str, *, clip: Any = None) -> FCurve | RotationTrack | None:
        if clip is not None:
            c = self._clip_item(clip)._definition
            spec = P.parse(path, c.target, location="path")
        else:
            key = self._target(target)
            item = self._action_item(key, create=False)
            if item is None:
                return None
            c = item._definition
            spec = P.parse(path, key[0], location="path")
        assert not isinstance(spec, list)
        return c.channel(spec.path)

    def edit_keys(self, target: Any, path: str, op: str, *, clip: Any = None, **args: Any) -> None:
        """A key operation (``set``, ``delete``, ``handle``, ``move``, ``scale_time``,
        ``reverse``, ``quantize``, ...; see :mod:`.clip`) on a scalar channel."""
        if clip is not None:
            item = self._clip_item(clip)
            kind = item._definition.target
        else:
            key = self._target(target)
            item = self._action_item(key, create=False)
            kind = key[0]
            if item is None:
                raise _err("CHANNEL_MISSING", f"{self._target_name(key)!r} has no action", "target")
        spec = P.parse(path, kind, location="path")
        assert not isinstance(spec, list)
        curve = item._definition.channel(spec.path)
        if curve is None:
            raise _err("CHANNEL_MISSING", f"no channel {spec.path}", "path")
        if not isinstance(curve, FCurve):
            raise _err(
                "INVALID_CHANNEL", f"{spec.path} is a rotation track: use set_channel", "path"
            )
        new = edit_curve(curve, op, args, f"operations[{op}]")
        with self._op(f"{op.replace('_', ' ').capitalize()} keys of {spec.path}"):
            self._set(item, definition=item._definition.with_channel(spec.path, new))

    # ------------------------------------------------------------------ #
    # Rigs, attachments, constraints, skins
    # ------------------------------------------------------------------ #

    def _object(self, ref: Any, location: str) -> Any:
        kind, oid = self._target(ref, location)
        if kind != P.OBJECT:
            raise _err("INVALID_TARGET", "a scene object is needed here", location)
        return self.scene._objects[oid]

    def rig(self, armature: Any) -> Any:
        """The :class:`~.rig.Rig` on ``armature`` (an object), or ``None``."""
        obj = self._object(armature, "armature")
        item = self._owned.get(("rig", obj._id))
        return None if item is None else item._definition

    def add_rig(self, armature: Any, joints: list[Any], **options: Any) -> Any:
        """Give ``armature`` (an object, usually an empty) a skeleton: ``joints`` are
        ``{name, parent, position, rotation | euler, scale, tail, limits}`` (rest
        transforms relative to the parent joint; roots relative to the armature).
        Options: ``attachments`` (``[{object, joint}]``), ``constraints``, ``mirror``."""
        obj = self._object(armature, "armature")
        if ("rig", obj._id) in self._owned:
            raise _err(
                "DUPLICATE_BINDING",
                f"{obj.name!r} already has a rig (set_rig replaces it)",
                "armature",
            )
        self.set_rig(obj, {"joints": joints, **options})
        return self.rig(obj)

    def set_rig(self, armature: Any, data: Any) -> None:
        """Replace ``armature``'s rig (a :class:`~.rig.Rig`, or its JSON form; ``None``
        removes it, with the skins bound to it). One undo step."""
        from .rig import Rig, parse_rig

        obj = self._object(armature, "armature")
        item = self._owned.get(("rig", obj._id))
        if data is None:
            if item is None:
                return
            with self._op(f"Remove rig of {obj.name!r}"):
                for (kind, _oid), other in list(self._owned.items()):
                    if kind == "skin" and other._definition.rig == item.id:
                        self._set(other, alive=False)
                self._set(item, alive=False)
            return
        raw = data.to_json() if isinstance(data, Rig) else dict(data)
        raw["object"] = obj._id
        raw["attachments"] = [
            {
                "object": self._object(a.get("object"), f"attachments[{i}].object")._id,
                "joint": a.get("joint"),
            }
            if isinstance(a, dict)
            else a
            for i, a in enumerate(raw.get("attachments", []))
        ]
        raw["constraints"] = [
            self._canonical_constraint(c, f"constraints[{i}]")
            for i, c in enumerate(raw.get("constraints", []))
        ]
        rig = parse_rig(raw, "rig")
        ancestors = {a._id for a in obj.iter_ancestors()}
        for oid, _ in rig.attachments:
            if oid in ancestors:
                raise _err(
                    "ATTACHMENT_CYCLE",
                    "an object above the armature cannot follow its joints",
                    "rig.attachments",
                )
        with self._op(f"Rig {obj.name!r}"):
            if item is None:
                self._new_item("rig", f"{obj.name} Rig", rig)
            else:
                self._set(item, definition=rig)
            self._check_cycles()

    def _canonical_constraint(self, raw: Any, location: str) -> Any:
        """Object references of a constraint's targets as ids (renames keep them)."""
        if not isinstance(raw, dict):
            return raw
        out = dict(raw)
        for key in ("target", "pole"):
            t = out.get(key)
            if isinstance(t, dict) and isinstance(t.get("object"), str):
                out[key] = {
                    **t,
                    "object": self._object(t["object"], f"{location}.{key}.object")._id,
                }
        if isinstance(out.get("path"), str):
            out["path"] = self._path_id(out["path"], f"{location}.path")
        return out

    def _path_id(self, ref: str, location: str) -> str:
        item = self._items.get(ref)
        if item is not None and item.kind == "path" and item._alive:
            return item.id
        item = self._path_names.get(ref)
        if item is None:
            raise _err("PATH_MISSING", f"no path {ref!r} (by id or name)", location)
        return item.id

    def attach(self, obj: Any, armature: Any, joint: str) -> None:
        """Make ``obj`` follow ``joint`` of ``armature``'s rig rigidly (from where it is
        now, relative to the rest pose)."""
        rig = self.rig(armature)
        if rig is None:
            raise _err("RIG_MISSING", "the armature has no rig (add_rig first)", "armature")
        target = self._object(obj, "object")
        raw = rig.to_json()
        raw["attachments"] = [a for a in raw.get("attachments", []) if a["object"] != target._id]
        raw["attachments"].append({"object": target._id, "joint": joint})
        self.set_rig(armature, raw)

    def detach(self, obj: Any) -> None:
        target = self._object(obj, "object")
        for (kind, arm), item in list(self._owned.items()):
            if kind == "rig" and any(a[0] == target._id for a in item._definition.attachments):
                raw = item._definition.to_json()
                raw["attachments"] = [a for a in raw["attachments"] if a["object"] != target._id]
                self.set_rig(arm, raw)

    def constraints(self, owner: Any) -> list[dict[str, Any]]:
        """``owner``'s constraints as JSON (an object's own; a rig's joint constraints are
        in ``rig(owner).constraints``)."""
        obj = self._object(owner, "owner")
        item = self._owned.get(("constraints", obj._id))
        return [] if item is None else [c.to_json() for c in item._definition.constraints]

    def set_constraints(self, owner: Any, constraints: list[Any]) -> None:
        """Replace an object's constraints (applied in order); one undo step."""
        from .constraints import parse_constraints

        obj = self._object(owner, "owner")
        canon = [
            self._canonical_constraint(c, f"constraints[{i}]") for i, c in enumerate(constraints)
        ]
        parsed = parse_constraints(canon, "constraints", joints=False)
        item = self._owned.get(("constraints", obj._id))
        with self._op(f"Constrain {obj.name!r}"):
            if not parsed:
                if item is not None:
                    self._set(item, alive=False)
                return
            definition = ObjectConstraints(obj._id, parsed)
            if item is None:
                self._new_item("constraints", f"{obj.name} Constraints", definition)
            else:
                self._set(item, definition=definition)
            self._check_cycles()

    def add_constraint(self, owner: Any, constraint: dict[str, Any]) -> None:
        """Add a constraint: to ``owner``'s rig when it names a joint (or is IK), else to
        the object itself."""
        obj = self._object(owner, "owner")
        if constraint.get("type") == "ik" or "joint" in constraint:
            rig = self.rig(obj)
            if rig is None:
                raise _err("RIG_MISSING", "joint constraints need a rig on the owner", "owner")
            raw = rig.to_json()
            raw["constraints"] = [*raw.get("constraints", []), constraint]
            self.set_rig(obj, raw)
        else:
            self.set_constraints(obj, [*self.constraints(obj), constraint])

    def remove_constraint(self, owner: Any, name: str) -> None:
        obj = self._object(owner, "owner")
        rig = self.rig(obj)
        if rig is not None and any(c.name == name for c in rig.constraints):
            raw = rig.to_json()
            raw["constraints"] = [c for c in raw.get("constraints", []) if c["name"] != name]
            self.set_rig(obj, raw)
            return
        current = self.constraints(obj)
        if not any(c["name"] == name for c in current):
            raise _err("CONSTRAINT_MISSING", f"no constraint {name!r}", "name")
        self.set_constraints(obj, [c for c in current if c["name"] != name])

    def _check_cycles(self) -> None:
        from .rigging import _target_objects, _unit_order

        moves: dict = {}
        reads: dict = {}
        for (kind, oid), item in self._owned.items():
            if kind == "rig":
                moves[("rig", oid)] = {a for a, _ in item._definition.attachments}
                reads[("rig", oid)] = _target_objects(item._definition.constraints, self.scene) | {
                    oid
                }
            elif kind == "constraints":
                moves[("object", oid)] = {oid}
                reads[("object", oid)] = _target_objects(item._definition.constraints, self.scene)
        _, cyclic = _unit_order(self.scene, moves, reads)
        if cyclic:
            names = sorted(
                self.scene._objects[u[1]].name for u in cyclic if u[1] in self.scene._objects
            )
            raise _err(
                "CONSTRAINT_CYCLE",
                f"constraints and attachments would make {names[:6]} depend on themselves",
                "constraints",
            )

    def skin(
        self,
        mesh_object: Any,
        armature: Any,
        *,
        weights: Any = "envelope",
        joints: list[str] | None = None,
        influences: int = 4,
        method: str = "linear",
    ) -> dict[str, Any]:
        """Bind a mesh object to a rig smoothly, in the current (rest) pose. ``weights``:
        ``{"indices": (n, k), "weights": (n, k), "joints": [names]}`` given explicitly,
        ``"nearest"`` (each vertex follows its nearest bone fully: rigid parts) or
        ``"envelope"`` (inverse-distance weights over the ``influences`` nearest bones: an
        automatic starting point, not production weight painting). Returns a report."""
        from .rig import MAX_INFLUENCES, make_skin

        obj = self._object(mesh_object, "mesh_object")
        arm = self._object(armature, "armature")
        rig = self.rig(arm)
        if rig is None:
            raise _err("RIG_MISSING", "the armature has no rig", "armature")
        mesh = obj.mesh
        if mesh is None or mesh.vertex_count == 0:
            raise _err("INVALID_SKIN", f"{obj.name!r} has no mesh to bind", "mesh_object")
        if obj._id == arm._id:
            raise _err("INVALID_SKIN", "the armature cannot skin itself", "mesh_object")
        if (
            isinstance(influences, bool)
            or not isinstance(influences, int)
            or not 1 <= influences <= MAX_INFLUENCES
        ):
            raise _err("INVALID_SKIN", f"influences is 1..{MAX_INFLUENCES}", "influences")
        from ..rendering.scene_data import world_matrices

        worlds = world_matrices(self.scene)
        a_rest = worlds[arm._id]
        m_rest = worlds[obj._id]
        rest_model = rig.rest_model()
        names = list(rig.names) if joints is None else list(joints)
        idx = []
        for i, n in enumerate(names):
            k = rig.index(n)
            if k is None:
                raise _err("INVALID_SKIN", f"no joint {n!r}", f"joints[{i}]")
            idx.append(k)
        bind_inv = np.linalg.inv(rest_model[idx]) @ np.linalg.inv(a_rest) @ m_rest
        if isinstance(weights, dict):
            if set(weights) - {"indices", "weights", "joints"}:
                raise _err(
                    "INVALID_SKIN", "explicit weights are {indices, weights, joints}", "weights"
                )
            if "joints" in weights:
                names = list(weights["joints"])
                idx = []
                for i, n in enumerate(names):
                    k = rig.index(n)
                    if k is None:
                        raise _err("INVALID_SKIN", f"no joint {n!r}", f"weights.joints[{i}]")
                    idx.append(k)
                bind_inv = np.linalg.inv(rest_model[idx]) @ np.linalg.inv(a_rest) @ m_rest
            vi, vw = weights.get("indices"), weights.get("weights")
            how = "explicit"
        elif weights in ("nearest", "envelope"):
            vi, vw = _automatic_weights(
                rig,
                idx,
                mesh.positions,
                np.linalg.inv(m_rest) @ a_rest,
                1 if weights == "nearest" else influences,
            )
            how = weights
        else:
            raise _err(
                "INVALID_SKIN", "weights is 'nearest', 'envelope' or {indices, weights}", "weights"
            )
        skin = make_skin(
            obj._id, "", [rig.joints[k].id for k in idx], bind_inv, vi, vw, method, "skin"
        )
        if skin.vertex_count != mesh.vertex_count:
            raise _err(
                "INVALID_SKIN",
                f"weights for {skin.vertex_count} vertices; the mesh has {mesh.vertex_count}",
                "weights",
            )
        rig_item = self._owned[("rig", arm._id)]
        skin = type(skin)(
            obj._id,
            rig_item.id,
            skin.joints,
            skin.inverse_bind,
            skin.indices,
            skin.weights,
            skin.method,
        )
        item = self._owned.get(("skin", obj._id))
        with self._op(f"Skin {obj.name!r} to {arm.name!r}"):
            if item is None:
                self._new_item("skin", f"{obj.name} Skin", skin)
            else:
                self._set(item, definition=skin)
        used = np.unique(skin.indices[skin.weights > 0])
        return {
            "vertices": skin.vertex_count,
            "joints": len(skin.joints),
            "influences": int(skin.indices.shape[1]),
            "joints_used": int(used.size),
            "weights": how,
            "method": method,
        }

    def unbind(self, mesh_object: Any) -> None:
        obj = self._object(mesh_object, "mesh_object")
        item = self._owned.get(("skin", obj._id))
        if item is not None:
            with self._op(f"Unbind {obj.name!r}"):
                self._set(item, alive=False)

    def pose_joints(
        self, armature: Any, time: float | None = None, space: str = "world"
    ) -> dict[str, np.ndarray]:
        """Joint matrices of ``armature``'s rig at ``time`` (``"world"`` or ``"armature"``
        space), after constraints -- for inspection."""
        obj = self._object(armature, "armature")
        if self.rig(obj) is None:
            raise _err("RIG_MISSING", "the armature has no rig", "armature")
        if space not in ("world", "armature"):
            raise _err("INVALID_ARGUMENT", "space is 'world' or 'armature'", "space")
        rigging = self.rigging(time)
        rig, model = rigging.joints[obj._id]
        if space == "armature":
            return {j.name: model[k].copy() for k, j in enumerate(rig.joints)}
        from ..rendering.scene_data import world_matrices

        overlay = self.overlay(time)
        a = world_matrices(self.scene, overlay)[obj._id]
        return {j.name: a @ model[k] for k, j in enumerate(rig.joints)}

    def rigging(self, time: float | None = None) -> Any:
        """The rigging pass at ``time`` (:class:`~.rigging.RiggingResult`: local
        matrices, deformed meshes, joint matrices, IK residuals, diagnostics)."""
        from .rigging import evaluate_rigging

        pose = self.evaluate(time)
        found = pose.extra.get("rigging")
        if found is None:
            found = pose.extra["rigging"] = evaluate_rigging(self, pose)
        return found

    # ------------------------------------------------------------------ #
    # Cinematics: paths, path following, shots, camera moves
    # ------------------------------------------------------------------ #

    def add_path(self, name: str, points: Any = None, **options: Any) -> str:
        """A path item (``kind``: ``catmull_rom`` (default), ``bezier`` with ``handles``,
        ``linear``, or ``circle`` with ``center``, ``radius``, ``normal``, ``start``);
        returns its id."""
        from .cinematics import parse_path

        P.check_name(name, "a path name", "name")
        if name in self._path_names:
            raise _err("DUPLICATE_NAME", f"a path is already named {name!r}", "name")
        raw = dict(options)
        if points is not None:
            raw["points"] = points.tolist() if isinstance(points, np.ndarray) else points
        path = parse_path(raw, "path")
        with self._op(f"Add path {name!r}"):
            item = self._new_item("path", name, path)
        return item.id

    def path(self, ref: str) -> Any:
        return self._items[self._path_id(ref, "path")]._definition

    def remove_path(self, ref: str) -> None:
        item = self._items[self._path_id(ref, "path")]
        with self._op(f"Remove path {item._name!r}"):
            self._set(item, alive=False)

    def follow_path(
        self,
        obj: Any,
        path: str,
        *,
        start: float,
        end: float,
        offsets: tuple[float, float] = (0.0, 1.0),
        easing: str | None = "sine",
        ease: str = "in_out",
        follow: bool = True,
        axis: str = "+y",
        up: str = "+z",
        loop: bool = False,
        name: str = "follow",
    ) -> None:
        """Move ``obj`` along ``path`` from ``offsets[0]`` to ``offsets[1]`` (length
        fractions) between times ``start`` and ``end``, eased (``easing=None``: constant
        speed): a follow_path constraint and keys on its offset. One undo step."""
        target = self._object(obj, "object")
        pid = self._path_id(path, "path")
        s, e = check_time(start, "start"), check_time(end, "end")
        if not s < e:
            raise _err("INVALID_RANGE", "start must be before end", "start")
        constraint = {
            "name": name,
            "type": "follow_path",
            "path": pid,
            "follow": follow,
            "axis": axis,
            "up": up,
            "loop": loop,
            "offset": float(offsets[0]),
        }
        with self._op(f"Follow path with {target.name!r}"):
            others = [c for c in self.constraints(target) if c["name"] != name]
            self.set_constraints(target, [*others, constraint])
            opts = {"interpolation": easing or "linear", "ease": ease}
            self.keyframe(target, f'constraints["{name}"].offset', s, float(offsets[0]), **opts)
            self.keyframe(target, f'constraints["{name}"].offset', e, float(offsets[1]), **opts)

    def shots(self) -> list[dict[str, Any]]:
        item = self._owned.get(("sequence", "sequence"))
        return [] if item is None else [s.to_json() for s in item._definition.shots]

    def set_shots(self, shots: list[Any] | None) -> None:
        """The shot sequence: ``[{camera, start, end, name, transition}]`` (cameras by
        id or name; ``transition``: ``{"type": "blend", "duration": d}``); ``None`` or
        ``[]`` removes it. One undo step."""
        from .cinematics import parse_sequence

        item = self._owned.get(("sequence", "sequence"))
        with self._op("Set shots"):
            if not shots:
                if item is not None:
                    self._set(item, alive=False)
                return
            canon = []
            for i, s in enumerate(shots):
                if not isinstance(s, dict):
                    raise _err("INVALID_SEQUENCE", "a shot is an object", f"shots[{i}]")
                cam = self._object(s.get("camera"), f"shots[{i}].camera")
                if cam._camera is None:
                    raise _err(
                        "INVALID_SEQUENCE", f"{cam.name!r} is not a camera", f"shots[{i}].camera"
                    )
                canon.append({**s, "camera": cam._id})
            seq = parse_sequence({"shots": canon}, "sequence")
            if item is None:
                self._new_item("sequence", "Sequence", seq)
            else:
                self._set(item, definition=seq)

    def add_shot(
        self, camera: Any, start: float, end: float, *, name: str | None = None, blend: float = 0.0
    ) -> None:
        shot: dict[str, Any] = {"camera": camera, "start": start, "end": end}
        if name is not None:
            shot["name"] = name
        if blend:
            shot["transition"] = {"type": "blend", "duration": blend}
        self.set_shots([*self.shots(), shot])

    def camera_move(
        self,
        camera: Any,
        move: str,
        amount: float,
        start: float,
        end: float,
        *,
        easing: str | None = "sine",
        ease: str = "in_out",
    ) -> None:
        """A camera move as keys (start: the camera's pose at ``start``; editable like any
        keys): ``dolly`` (forward, along the view), ``truck`` (right), ``pedestal`` /
        ``crane`` (up, world Z), ``pan`` (degrees about world Z), ``tilt`` (degrees about
        the camera's X), ``roll`` (degrees about its view axis), ``zoom`` (a factor: 2
        halves the view's width). One undo step."""
        cam = self._object(camera, "camera")
        if cam._camera is None:
            raise _err("INVALID_TARGET", f"{cam.name!r} is not a camera", "camera")
        moves = ("dolly", "truck", "pedestal", "crane", "pan", "tilt", "roll", "zoom")
        if move not in moves:
            raise _err("INVALID_ARGUMENT", f"move is one of {', '.join(moves)}", "move")
        if (
            isinstance(amount, bool)
            or not isinstance(amount, (int, float))
            or not math.isfinite(float(amount))
        ):
            raise _err("INVALID_ARGUMENT", "amount is a finite number", "amount")
        s, e = check_time(start, "start"), check_time(end, "end")
        if not s < e:
            raise _err("INVALID_RANGE", "start must be before end", "start")
        from ..rendering.scene_data import world_matrices

        overlay = self.overlay(s)
        worlds = world_matrices(self.scene, overlay)
        world = worlds[cam._id]
        parent = cam._parent
        p_world = np.eye(4) if parent is None else worlds[parent._id]
        local = np.linalg.solve(p_world, world)
        from .constraints import decompose

        pos, rot, _ = decompose(local)
        opts: dict[str, Any] = {"interpolation": easing or "linear", "ease": ease}
        timing = None if easing is None else {"easing": easing, "ease": ease}
        with self._op(f"{move.capitalize()} {cam.name!r}"):
            if move == "zoom":
                if amount <= 0:
                    raise _err("INVALID_ARGUMENT", "a zoom factor is positive", "amount")
                lens = cam._camera
                values = None if overlay is None else overlay.field_values(cam._id, "camera")
                fov0 = float(values["fov"]) if values and "fov" in values else float(lens.fov)
                fov1 = math.degrees(2 * math.atan(math.tan(math.radians(fov0) / 2) / amount))
                self.keyframe(cam, "camera.fov", s, fov0, **opts)
                self.keyframe(cam, "camera.fov", e, fov1, **opts)
                return
            if move in ("pan", "tilt", "roll"):
                world_rot = world[:3, :3] / np.linalg.norm(world[:3, :3], axis=0)
                axis = {
                    "pan": np.array([0.0, 0.0, 1.0]),
                    "tilt": world_rot[:, 0],
                    "roll": world_rot[:, 1],
                }[move]
                turn = R.axis_angle(axis, np.radians(amount))[0]
                parent_rot = (
                    np.eye(3)
                    if parent is None
                    else p_world[:3, :3] / np.linalg.norm(p_world[:3, :3], axis=0)
                )
                q_parent = R.from_matrices(parent_rot)[0]
                local_turn = R.multiply(
                    R.conjugate(q_parent)[None], R.multiply(turn[None], q_parent[None])
                )[0]
                q1 = R.multiply(local_turn[None], rot[None])[0]
                key0: dict[str, Any] = {"timing": timing} if timing else {}
                self.keyframe(cam, "transform.rotation", s, rot.tolist(), **key0)
                self.keyframe(cam, "transform.rotation", e, R.align(rot, q1).tolist())
                return
            forward = -world[:3, 1] / np.linalg.norm(world[:3, 1])
            right = world[:3, 0] / np.linalg.norm(world[:3, 0])
            direction = {
                "dolly": forward,
                "truck": right,
                "pedestal": np.array([0.0, 0.0, 1.0]),
                "crane": np.array([0.0, 0.0, 1.0]),
            }[move]
            delta_world = direction * float(amount)
            delta_local = np.linalg.solve(p_world[:3, :3], delta_world)
            self.keyframe(cam, "transform.position", s, pos.tolist(), **opts)
            self.keyframe(cam, "transform.position", e, (pos + delta_local).tolist(), **opts)

    def camera_orbit(
        self,
        camera: Any,
        target: Any,
        start: float,
        end: float,
        *,
        degrees: float = 360.0,
        easing: str | None = "sine",
        ease: str = "in_out",
        name: str | None = None,
    ) -> str:
        """Orbit ``camera`` around ``target`` (an object or a point) by ``degrees`` from
        where it is at ``start``, keeping its distance and height, looking at the target:
        an exact circle path, a follow_path constraint with eased offset keys and a look_at
        constraint -- all editable. Returns the path's id."""
        cam = self._object(camera, "camera")
        from ..rendering.scene_data import world_matrices

        worlds = world_matrices(self.scene, self.overlay(start))
        if isinstance(target, (list, tuple)) and len(target) == 3:
            center = np.asarray(target, np.float64)
            look = {"point": [float(v) for v in center]}
        else:
            tobj = self._object(target, "target")
            center = worlds[tobj._id][:3, 3]
            look = {"object": tobj._id}
        here = worlds[cam._id][:3, 3]
        offset = here - center
        radius = float(np.hypot(offset[0], offset[1]))
        if radius < 1e-9:
            raise _err(
                "INVALID_ARGUMENT", "the camera is above its target: no circle to orbit", "camera"
            )
        start_angle = math.degrees(math.atan2(offset[1], offset[0]))
        pname = name or self._unique_path_name(f"{cam.name} Orbit")
        with self._op(f"Orbit {cam.name!r}"):
            pid = self.add_path(
                pname,
                kind="circle",
                center=[float(center[0]), float(center[1]), float(here[2])],
                radius=radius,
                start=start_angle,
            )
            self.follow_path(
                cam,
                pid,
                start=start,
                end=end,
                offsets=(0.0, float(degrees) / 360.0),
                easing=easing,
                ease=ease,
                follow=False,
                loop=True,
                name="orbit",
            )
            others = [c for c in self.constraints(cam) if c["name"] != "orbit_aim"]
            self.set_constraints(
                cam, [*others, {"name": "orbit_aim", "type": "look_at", "target": look}]
            )
        return pid

    def _unique_path_name(self, base: str) -> str:
        name, k = base, 1
        while name in self._path_names:
            k += 1
            name = f"{base}.{k:03d}"
        return name

    # ------------------------------------------------------------------ #
    # Scene hooks
    # ------------------------------------------------------------------ #

    def _objects_removed(self, objects: list[SceneObject]) -> None:
        """Bindings of removed objects go with them (inside the removal's operation)."""
        gone = {o.id for o in objects}
        for key, item in list(self._bindings.items()):
            if key[0] == P.OBJECT and key[1] in gone:
                self._set(item, alive=False)
        for (kind, oid), item in list(self._owned.items()):
            if oid in gone or (kind == "skin" and item._definition.rig in self._dead_rigs(gone)):
                self._set(item, alive=False)
            elif kind == "rig":
                rig = item._definition
                kept = tuple(a for a in rig.attachments if a[0] not in gone)
                if len(kept) != len(rig.attachments):
                    self._set(item, definition=_replace_rig(rig, attachments=kept))

    def _dead_rigs(self, gone: set[str]) -> set[str]:
        return {i.id for (k, oid), i in self._owned.items() if k == "rig" and oid in gone}

    def _material_removed(self, material_id: str) -> None:
        item = self._bindings.get((P.MATERIAL, material_id))
        if item is not None:
            self._set(item, alive=False)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _program(self) -> Program:
        prog = self._compiled
        if prog is None or prog.revision != self.revision:
            prog = self._compiled = compile_program(self)
        return prog

    def _rig_rest(self) -> Any:
        return _RigRest(self)

    def _rigs(self) -> dict[str, tuple[AnimItem, Any]]:
        return {oid: (i, i._definition) for (k, oid), i in self._owned.items() if k == "rig"}

    def _skins(self) -> dict[str, Any]:
        return {oid: i._definition for (k, oid), i in self._owned.items() if k == "skin"}

    def _morphs(self) -> dict[str, Any]:
        return {oid: i._definition for (k, oid), i in self._owned.items() if k == "morph"}

    def set_morph_targets(self, mesh_object: Any, targets: list[Any] | None) -> None:
        """Give a mesh object morph targets (blend shapes): ``[{name, positions,
        normals?, default?}]`` with per-vertex offsets; weights animate as
        ``morph["NAME"]``. ``None`` removes them. One undo step."""
        from .rig import make_morph

        obj = self._object(mesh_object, "mesh_object")
        item = self._owned.get(("morph", obj._id))
        with self._op(f"Morph targets of {obj.name!r}"):
            if targets is None:
                if item is not None:
                    self._set(item, alive=False)
                return
            mesh = obj.mesh
            if mesh is None:
                raise _err("INVALID_MORPH", f"{obj.name!r} has no mesh", "mesh_object")
            morph = make_morph(obj._id, targets, mesh.vertex_count, "morph")
            if item is None:
                self._new_item("morph", f"{obj.name} Morph", morph)
            else:
                self._set(item, definition=morph)

    def morph_targets(self, mesh_object: Any) -> Any:
        obj = self._object(mesh_object, "mesh_object")
        item = self._owned.get(("morph", obj._id))
        return None if item is None else item._definition

    def _object_constraints(self) -> dict[str, tuple[AnimItem, tuple]]:
        return {
            oid: (i, i._definition.constraints)
            for (k, oid), i in self._owned.items()
            if k == "constraints"
        }

    def _follow_path(self, constraint: Any, world: np.ndarray, offset: float) -> Any:
        from .cinematics import follow_path

        return follow_path(self, constraint, world, offset)

    @property
    def active(self) -> bool:
        """Whether evaluation can change anything (enabled, with bindings)."""
        return self.enabled and (bool(self._bindings) or bool(self._owned))

    def evaluate(self, time: float | None = None) -> AnimationPose:
        """The animation at ``time`` (default: the current time); see
        :class:`~.evaluate.AnimationPose`. Cached per (scene revision, time)."""
        t = self._time if time is None else check_time(time, "time")
        key = (self.scene.revision, float(t).hex())
        history = self.scene._history
        cached = None if history.in_progress else self._evaluations.get(key)
        if cached is not None:
            self._evaluations.move_to_end(key)
            return cached
        pose = evaluate(self, t)
        if not history.in_progress:
            self._evaluations[key] = pose
            while len(self._evaluations) > EVALUATION_CACHE:
                self._evaluations.popitem(last=False)
        return pose

    def overlay(self, time: float | None = None) -> Any:
        """The :class:`~aether3d.rendering.overlay.RenderOverlay` of the pose at ``time``
        (default: the current time), or ``None`` when nothing is animated."""
        if not self.active:
            if time is not None:
                check_time(time, "time")
            return None
        pose = self.evaluate(time)
        found = pose.extra.get("overlay", _UNSET)
        if found is _UNSET:
            found = _overlay_of(pose, self.scene)
            pose.extra["overlay"] = found
        return found

    def sample(self, target: Any, path: str, times: Any, *, order: int = 0) -> Any:
        """A channel of ``target``'s action at ``times`` (value and derivatives up to
        ``order``) -- the clip's curve, before layers and blending."""
        ch = self.channel(target, path)
        if ch is None:
            raise _err("CHANNEL_MISSING", f"no channel {path}", "path")
        return ch.evaluate(np.asarray(times, np.float64), order=order)

    # ------------------------------------------------------------------ #
    # Motion tools: reports, curve fitting, generators, retargeting, contacts
    # ------------------------------------------------------------------ #

    def _curve_ref(self, target: Any, path: str, clip: Any) -> tuple[Any, str, Any]:
        """``(clip item, canonical path, channel)`` of a channel in ``target``'s action or
        in ``clip``."""
        if clip is not None:
            item = self._clip_item(clip)
            kind = item._definition.target
        else:
            key = self._target(target)
            item = self._action_item(key, create=False)
            kind = key[0]
            if item is None:
                raise _err("CHANNEL_MISSING", f"{self._target_name(key)!r} has no action", "target")
        spec = P.parse(path, kind, location="path")
        assert not isinstance(spec, list)
        ch = item._definition.channel(spec.path)
        if ch is None:
            raise _err("CHANNEL_MISSING", f"no channel {spec.path}", "path")
        return item, spec.path, ch

    def curve_report(
        self, target: Any, path: str, *, clip: Any = None, samples: int = 0
    ) -> dict[str, Any]:
        """Everything measurable about one channel (see :mod:`.report`)."""
        from .report import curve_report

        _, canonical, ch = self._curve_ref(target, path, clip)
        if not isinstance(ch, FCurve):
            raise _err("INVALID_CHANNEL", f"{canonical} is a rotation track", "path")
        out = curve_report(ch, samples=samples)
        out["path"] = canonical
        return out

    def trajectory(self, target: Any, **options: Any) -> dict[str, Any]:
        """The world-space motion of an object or (``joint=``) a joint (see
        :func:`.report.trajectory_report`)."""
        from .report import trajectory_report

        return trajectory_report(self, target, **options)

    def diagnose(self, **options: Any) -> dict[str, Any]:
        """Findings over the whole animation (see :func:`.report.diagnose`)."""
        from .report import diagnose

        return diagnose(self, **options)

    def simplify(
        self, target: Any, path: str, tolerance: float, *, clip: Any = None, **options: Any
    ) -> dict[str, Any]:
        """Replace a channel with fewer keys within ``tolerance`` (one undo step); returns
        ``{keys_before, keys_after, max_error, basis}``."""
        from .fitting import simplify

        item, canonical, ch = self._curve_ref(target, path, clip)
        if not isinstance(ch, FCurve):
            raise _err("INVALID_CHANNEL", f"{canonical} is a rotation track", "path")
        new, report = simplify(ch, tolerance, **options)
        with self._op(f"Simplify {canonical}"):
            self._set(item, definition=item._definition.with_channel(canonical, new))
        return report

    def resample(
        self, target: Any, path: str, *, fps: Any = None, clip: Any = None, **options: Any
    ) -> int:
        """Replace a channel with a key on every frame (of ``fps``, default the
        timeline's); returns the key count. One undo step."""
        from .fitting import resample

        item, canonical, ch = self._curve_ref(target, path, clip)
        if not isinstance(ch, FCurve):
            raise _err("INVALID_CHANNEL", f"{canonical} is a rotation track", "path")
        new = resample(ch, self.fps if fps is None else fps, **options)
        with self._op(f"Resample {canonical}"):
            self._set(item, definition=item._definition.with_channel(canonical, new))
        return len(new)

    def bake(self, targets: list[Any], **options: Any) -> dict[str, Any]:
        """What the scene shows (layers, rigs, constraints) as plain keys in new clips
        (see :func:`.fitting.bake`)."""
        from .fitting import bake

        return bake(self, list(targets), **options)

    def generate(self, armature: Any, kind: str, **options: Any) -> dict[str, Any]:
        """A motion (``idle``, ``walk``, ``run``, ``jump``, ``land``, ``wave``,
        ``head_turn``) written as an editable clip (see :mod:`.generators`)."""
        from .generators import generate

        return generate(self, armature, kind, **options)

    def follow_camera(self, camera: Any, target: Any, **options: Any) -> dict[str, Any]:
        """Key a camera to follow a target with lag, aimed at it (see
        :func:`.generators.follow_camera`)."""
        from .generators import follow_camera

        return follow_camera(self, camera, target, **options)

    def mirror_clip(self, clip: Any, armature: Any, *, name: str | None = None) -> ClipHandle:
        """A new clip: ``clip`` with left and right swapped by ``armature``'s rig (see
        :func:`.motion.mirror_clip`)."""
        from .motion import mirror_clip

        source = self._clip_item(clip)
        rig = self.rig(armature)
        if rig is None:
            raise _err("RIG_MISSING", "the armature has no rig", "armature")
        data = mirror_clip(source._definition, rig)
        new_name = name or self._unique_clip_name(f"{source._name} Mirrored")
        return self.add_clip(new_name, data)

    def extract_root_motion(
        self, clip: Any, joint: str, *, mode: str = "extract", yaw: bool = False
    ) -> dict[str, Any]:
        """Move (or drop) a root joint's travel out of its channels (see
        :func:`.motion.extract_root_motion`). One undo step."""
        from .motion import extract_root_motion

        item = self._clip_item(clip)
        new, report = extract_root_motion(item._definition, joint, mode=mode, yaw=yaw)
        self._set_clip(item, new, f"Extract root motion of {item._name!r}")
        return report

    def foot_contacts(self, armature: Any, feet: list[str], **options: Any) -> dict[str, Any]:
        """Planted intervals of feet, their sliding and penetration (see
        :func:`.motion.foot_contacts`)."""
        from .motion import foot_contacts

        return foot_contacts(self, armature, list(feet), **options)

    def plant_feet(
        self, armature: Any, legs: dict[str, list[str]], **options: Any
    ) -> dict[str, Any]:
        """Correct foot sliding with keyed IK (see :func:`.motion.plant_feet`)."""
        from .motion import plant_feet

        return plant_feet(self, armature, dict(legs), **options)

    def retarget(self, source: Any, target: Any, clip: Any, **options: Any) -> dict[str, Any]:
        """A clip for ``target``'s rig from ``source``'s (see :func:`.motion.retarget`)."""
        from .motion import retarget

        return retarget(self, source, target, clip, **options)

    def graph(
        self,
        channels: list[tuple[Any, str]] | None = None,
        *,
        clip: Any = None,
        format: str = "svg",
        **options: Any,
    ) -> Any:
        """A graph of channels -- ``[(target, path), ...]`` from actions, or every
        scalar channel of ``clip`` (up to 32) -- as SVG text (``format="svg"``) or an RGB
        image (``"png"``); see :mod:`.graph`."""
        from .graph import graph_png, graph_svg

        curves: dict[str, FCurve] = {}
        if clip is not None and channels is None:
            item = self._clip_item(clip)
            for path, ch in item._definition.channels:
                if isinstance(ch, FCurve):
                    curves[path] = ch
        else:
            for i, entry in enumerate(channels or []):
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise _err(
                        "INVALID_ARGUMENT", "channels are [target, path] pairs", f"channels[{i}]"
                    )
                _, canonical, ch = self._curve_ref(entry[0], entry[1], clip)
                if not isinstance(ch, FCurve):
                    raise _err(
                        "INVALID_CHANNEL", f"{canonical} is a rotation track", f"channels[{i}]"
                    )
                label = (
                    canonical
                    if clip is not None
                    else f"{self._target_name(self._target(entry[0]))}: {canonical}"
                )
                curves[label] = ch
        if format not in ("svg", "png"):
            raise _err("INVALID_ARGUMENT", "format is 'svg' or 'png'", "format")
        if not curves:
            raise _err("CHANNEL_MISSING", "no curves to graph", "channels")
        return (graph_svg if format == "svg" else graph_png)(curves, **options)

    # ------------------------------------------------------------------ #
    # Applying a pose (explicit)
    # ------------------------------------------------------------------ #

    def apply_pose(self, time: float | None = None) -> int:
        """Write the pose at ``time`` into the scene's authored transforms and fields
        (one undo step); returns how many objects changed. The animation is kept."""
        pose = self.evaluate(time)
        scene = self.scene
        from ..transform import Transform

        count = 0
        with self._op(f"Apply animation pose at {pose.time:g} s"):
            for k, oid in enumerate(pose.ids):
                obj = scene._objects.get(oid)
                if obj is None:
                    continue
                obj.transform = Transform(pose.position[k], pose.rotation[k], pose.scale[k])
                count += 1
            for oid, vis in pose.visible.items():
                if oid in scene._objects:
                    scene._objects[oid].visible = vis
            for (oid, group), values in pose.fields.items():
                obj = scene._objects.get(oid)
                if obj is None:
                    continue
                holder = (
                    obj.camera
                    if group == "camera"
                    else obj.light
                    if group == "light"
                    else obj.appearance
                )
                holder.configure(
                    **{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()}
                )
            for mid, values in pose.materials.items():
                m = scene.materials.find_material(mid)
                if m is not None:
                    m.configure(
                        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()}
                    )
        return count

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        return (
            not any(i._alive for i in self._items.values()) and self._settings == DEFAULT_SETTINGS
        )

    def to_json(self) -> dict[str, Any]:
        items = []
        for item in sorted(self._items.values(), key=lambda i: i._serial):
            if not item._alive:
                continue
            items.append(
                {
                    "id": item.id,
                    "kind": item.kind,
                    "name": item._name,
                    "definition": _definition_json(item),
                }
            )
        return {"settings": copy.deepcopy(self._settings), "items": items}

    def _load(self, data: Any, location: str = "animation") -> None:
        """Read a document (``to_json``'s form) into this empty system (no history)."""
        if not isinstance(data, dict) or set(data) - {"settings", "items"}:
            raise _err("INVALID_DOCUMENT", "the animation section is {settings, items}", location)
        settings = _check_settings(data.get("settings", DEFAULT_SETTINGS), f"{location}.settings")
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list) or len(raw_items) > MAX_ITEMS:
            raise _err(
                "INVALID_DOCUMENT", f"items is a list of at most {MAX_ITEMS}", f"{location}.items"
            )
        self._settings = settings
        # Items keep the file's order (their serials decide the order they are saved in).
        records: list[tuple[dict, str, AnimItem]] = []
        for i, raw in enumerate(raw_items):
            loc = f"{location}.items[{i}]"
            if not isinstance(raw, dict) or set(raw) != {"id", "kind", "name", "definition"}:
                raise _err("INVALID_DOCUMENT", "an item is {id, kind, name, definition}", loc)
            if raw["kind"] not in KINDS:
                raise _err(
                    "INVALID_DOCUMENT",
                    f"unknown animation item kind {raw['kind']!r}",
                    f"{loc}.kind",
                )
            iid = raw["id"]
            if not isinstance(iid, str) or not iid or iid in self._items:
                raise _err("INVALID_DOCUMENT", "item ids must be unique strings", f"{loc}.id")
            P.check_name(raw["name"], "an item name", f"{loc}.name")
            item = AnimItem(self, iid, raw["kind"], raw["name"], None)
            self._items[iid] = item
            records.append((raw, loc, item))
        for raw, loc, item in records:
            if item.kind == "clip":
                item._definition = Clip.from_json(raw["definition"], f"{loc}.definition")
        for raw, loc, item in records:
            if item.kind in ("path", "sequence"):
                item._definition = definition_from_json(
                    item.kind, raw["definition"], self, f"{loc}.definition"
                )
                if item.kind == "sequence":
                    for shot in item._definition.shots:
                        if shot.camera not in self.scene._objects:
                            raise _err(
                                "INVALID_DOCUMENT",
                                f"a shot names a missing camera {shot.camera!r}",
                                loc,
                            )
        for kind in ("rig", "skin", "constraints", "morph"):
            for raw, loc, item in records:
                if item.kind == kind:
                    item._definition = definition_from_json(
                        kind, raw["definition"], self, f"{loc}.definition"
                    )
                    owner = item._definition.object
                    if owner not in self.scene._objects:
                        raise _err(
                            "INVALID_DOCUMENT",
                            f"{kind} {raw['name']!r} names a missing object {owner!r}",
                            loc,
                        )
                    if kind == "skin":
                        rig_item = self._items.get(item._definition.rig)
                        if rig_item is None or rig_item.kind != "rig":
                            raise _err(
                                "INVALID_DOCUMENT", f"skin {raw['name']!r} names a missing rig", loc
                            )
                    if kind == "rig":
                        for oid, _ in item._definition.attachments:
                            if oid not in self.scene._objects:
                                raise _err(
                                    "INVALID_DOCUMENT",
                                    f"rig {raw['name']!r} attaches a missing object {oid!r}",
                                    loc,
                                )
        for raw, loc, item in records:
            if item.kind != "binding":
                continue
            binding = parse_binding(raw["definition"], f"{loc}.definition", self._clip_id)
            kind, tid = binding.target
            if kind == P.OBJECT and tid not in self.scene._objects:
                raise _err(
                    "INVALID_DOCUMENT",
                    f"binding {raw['name']!r} animates a missing object {tid!r}",
                    loc,
                )
            if kind == P.MATERIAL and self.scene.materials.find_material(tid) is None:
                raise _err(
                    "INVALID_DOCUMENT",
                    f"binding {raw['name']!r} animates a missing material {tid!r}",
                    loc,
                )
            self._check_binding_clips(binding)
            item._definition = binding
        self._reindex()
        self._changed()


class ClipHandle:
    """A clip of the animation document: read it, edit its channels and events."""

    def __init__(self, system: AnimationSystem, item: AnimItem) -> None:
        self._system = system
        self._item = item

    @property
    def id(self) -> str:
        return self._item.id

    @property
    def name(self) -> str:
        return self._item._name

    @property
    def data(self) -> Clip:
        return self._item._definition

    @property
    def paths(self) -> tuple[str, ...]:
        return self._item._definition.paths

    def channel(self, path: str) -> FCurve | RotationTrack | None:
        spec = P.parse(path, self.data.target, location="path")
        assert not isinstance(spec, list)
        return self.data.channel(spec.path)

    def set_channel(self, path: str, keys: Any, **curve_options: Any) -> None:
        self._system.set_channel(None, path, keys, clip=self._item, **curve_options)

    def keyframe(self, path: str, time: float, value: Any, **options: Any) -> None:
        specs = P.parse(path, self.data.target, grouped=True, location="path")
        clip = _keyframe_into(
            self.data, specs, check_time(time, "time"), value, options, "keyframe"
        )
        self._system._set_clip(self._item, clip, f"Keyframe {path} in {self.name!r}")

    def edit_keys(self, path: str, op: str, **args: Any) -> None:
        self._system.edit_keys(None, path, op, clip=self._item, **args)

    def set_events(self, events: list[Any]) -> None:
        self._system._set_clip(
            self._item, self.data.with_events(events), f"Set events of {self.name!r}"
        )

    def set_range(self, rng: Any) -> None:
        self._system._set_clip(self._item, self.data.with_range(rng), f"Set range of {self.name!r}")

    def rename(self, name: str) -> None:
        P.check_name(name, "a clip name", "name")
        system = self._system
        if name in system._clip_names and system._clip_names[name] is not self._item:
            raise _err("DUPLICATE_NAME", f"a clip is already named {name!r}", "name")
        with system._op(f"Rename clip to {name!r}"):
            system._set(self._item, name=name)

    def to_json(self) -> dict[str, Any]:
        return self.data.to_json()

    def __repr__(self) -> str:
        return f"<clip {self.name!r}: {len(self.paths)} channels>"


# --------------------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------------------- #

_UNSET = object()


@dataclass(frozen=True)
class ObjectConstraints:
    """An object's constraint list (an item of kind ``constraints``)."""

    object: str
    constraints: tuple

    def to_json(self) -> dict[str, Any]:
        return {"object": self.object, "constraints": [c.to_json() for c in self.constraints]}


def _owner_of(item: AnimItem) -> str:
    return item._definition.object


def _replace_rig(rig: Any, **changes: Any) -> Any:
    from .rig import Rig

    fields = {
        f: getattr(rig, f) for f in ("object", "joints", "attachments", "constraints", "mirror")
    }
    fields.update(changes)
    return Rig(**fields)


class _RigRest:
    """Rest values of joint, morph and constraint channels (for evaluation)."""

    def __init__(self, system: AnimationSystem) -> None:
        self.system = system

    def _rig(self, arm: str) -> Any:
        item = self.system._owned.get(("rig", arm))
        return None if item is None else item._definition

    def rest_trs(self, arm: str, name: str) -> tuple:
        rig = self._rig(arm)
        k = None if rig is None else rig.index(name)
        if k is None:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0))
        j = rig.joints[k]
        return (j.position, j.rotation, j.scale)

    def rest_component(self, arm: str, spec: Any) -> float | None:
        rig = self._rig(arm)
        k = None if rig is None else rig.index(spec.name)
        if k is None:
            return None
        j = rig.joints[k]
        if spec.field == "position":
            return float(j.position[spec.component])
        if spec.field == "scale":
            return float(j.scale[spec.component])
        if spec.field == "rotation_euler":
            return float(R.to_euler(np.array(j.rotation))[0][spec.component])
        return None

    def rest_rotation(self, arm: str, name: str) -> Any:
        rig = self._rig(arm)
        k = None if rig is None else rig.index(name)
        return None if k is None else rig.joints[k].rotation

    def rest_extra(self, oid: str, spec: Any) -> float:
        if spec.group == "morph":
            item = self.system._owned.get(("morph", oid))
            if item is not None and spec.name in item._definition.names:
                return float(item._definition.defaults[item._definition.names.index(spec.name)])
            return 0.0
        for kind in ("rig", "constraints"):
            item = self.system._owned.get((kind, oid))
            if item is not None:
                for c in item._definition.constraints:
                    if c.name == spec.name:
                        return float(c.influence)
        return 1.0


def _automatic_weights(
    rig: Any, joints: list[int], positions: np.ndarray, mesh_from_armature: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Weights from distances to bone segments (mesh space): each vertex takes its ``k``
    nearest bones with weights proportional to ``1 / (d + 1e-6 L)^4`` (``L`` the median
    bone length), normalized."""
    model = rig.rest_model()
    tails = rig.tails()
    heads = []
    ends = []
    for j in joints:
        m = mesh_from_armature @ model[j]
        heads.append(m[:3, 3])
        ends.append((m @ np.append(tails[j], 1.0))[:3])
    a = np.array(heads)
    b = np.array(ends)
    p = positions.astype(np.float64)
    seg = b - a
    length2 = np.maximum(np.sum(seg * seg, axis=1), 1e-30)
    median = float(np.median(np.sqrt(length2))) or 1.0
    n = p.shape[0]
    k = min(k, len(joints))
    best_i = np.zeros((n, k), np.int32)
    best_d = np.full((n, k), np.inf)
    for start in range(0, n, 32768):
        q = p[start : start + 32768]
        rel = q[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(rel * seg[None], axis=2) / length2[None], 0.0, 1.0)
        closest = a[None] + t[..., None] * seg[None]
        d = np.linalg.norm(q[:, None, :] - closest, axis=2)
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        best_i[start : start + q.shape[0]] = order
        best_d[start : start + q.shape[0]] = np.take_along_axis(d, order, axis=1)
    w = 1.0 / (best_d + 1e-6 * median) ** 4
    return best_i, w / w.sum(axis=1, keepdims=True)


def canonical_quats(q: np.ndarray) -> np.ndarray:
    """Rotations as authored transforms store them (``math3d.quat``): unit, ``w >= 0``,
    no negative zeros -- so posed matrices equal those of the pose written into the scene."""
    from .. import math3d

    q = np.array(q, np.float64).reshape(-1, 4)
    norm = np.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    off = np.abs(norm - 1.0) > 1e-15
    for k in np.flatnonzero(off).tolist():  # (rare: poses are normalized already)
        q[k] = math3d.quat(q[k])
    flip = ~off & (q[:, 3] < 0.0)
    q[flip] = -q[flip]
    return q + 0.0


def _overlay_of(pose: AnimationPose, scene: Any) -> Any:
    """The render overlay of ``pose`` (``None`` when it changes nothing)."""
    from ..diagnostics import WARNING, Diagnostic
    from ..rendering.overlay import RenderOverlay

    system = scene._animation
    rigging = system.rigging(pose.time)
    ids = tuple(rigging.local)
    local = np.array([rigging.local[i] for i in ids]).reshape(-1, 4, 4)
    if rigging.diagnostics:
        pose.extra.setdefault("overlay_diagnostics", []).extend(rigging.diagnostics)
    fields = {}
    dropped = []
    for (oid, group), values in pose.fields.items():
        obj = scene._objects.get(oid)
        holder = (
            None
            if obj is None
            else (
                obj._camera
                if group == "camera"
                else obj._light
                if group == "light"
                else obj._appearance
            )
        )
        if holder is None:
            continue
        try:  # the combination must be valid (e.g. near < far)
            trial = holder.copy()
            trial._owner = None
            trial.configure(
                **{k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()}
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised: renders go on
            dropped.append((oid, group, str(exc)))
            continue
        fields[(oid, group)] = values
    if dropped:
        extra = pose.extra.setdefault("overlay_diagnostics", [])
        for oid, group, why in dropped:
            extra.append(
                Diagnostic(
                    "ANIMATED_VALUE_INVALID",
                    WARNING,
                    f"the animated {group} values of {scene._objects[oid].name!r} are invalid "
                    f"together ({why}); this evaluation uses the authored ones",
                    f"objects[{scene._objects[oid].name}].{group}",
                )
            )
    overlay = RenderOverlay(
        ids=ids,
        local=local,
        visible=dict(pose.visible),
        fields=fields,
        materials=dict(pose.materials),
        meshes=dict(rigging.meshes),
        time=pose.time,
    )
    if ("sequence", "sequence") in system._owned:
        from ..rendering.scene_data import world_matrices
        from .cinematics import shot_camera

        camera = shot_camera(system, pose.time, world_matrices(scene, overlay), fields)
        if camera is not None:
            overlay = RenderOverlay(
                ids=ids,
                local=local,
                visible=overlay.visible,
                fields=fields,
                materials=overlay.materials,
                meshes=overlay.meshes,
                camera=camera,
                time=pose.time,
            )
    return None if overlay.empty else overlay


def _clip_from(data: Any, target: str, location: str) -> Clip:
    if data is None:
        if target not in (P.OBJECT, P.MATERIAL):
            raise _err("INVALID_CLIP", "target is 'object' or 'material'", f"{location}.target")
        return Clip(target)
    if isinstance(data, Clip):
        return data
    if isinstance(data, dict):
        data = {"target": target, **data}
        return Clip.from_json(data, location)
    raise _err("INVALID_CLIP", "clip data is a Clip or its JSON", location)


def _keyframe_into(
    clip: Clip, specs: list[P.PathSpec], time: float, value: Any, options: dict, location: str
) -> Clip:
    if specs[0].kind == P.BOOL:  # on/off: stepped, never interpolated
        if options.get("interpolation", "constant") != "constant":
            raise _err(
                "INVALID_INTERPOLATION",
                f"{specs[0].path} is on/off: its keys are constant",
                location,
            )
        options = {**options, "interpolation": "constant"}
    if specs[0].kind == P.ROTATION:
        q = _quat_value(value, f"{location}.value")
        spec = specs[0]
        track = clip.channel(spec.path)
        row = {"time": time, "rotation": q, **options}
        if track is None:
            new = RotationTrack.from_keys([row], location=location)
        else:
            keys = [k for k in track.keys() if k["time"] != time]  # noqa: SIM118 (a method)
            keys.append(row)
            new = RotationTrack.from_keys(
                keys, extrapolation=list(track.extrapolation), location=location
            )
        return clip.with_channel(spec.path, new, location)
    values = _vector_value(value, len(specs), specs[0], f"{location}.value")
    for spec, v in zip(specs, values, strict=True):
        clip = clip.with_channel(
            spec.path, insert_key(clip.channel(spec.path), time, v, location, **options), location
        )
    return clip


def _quat_value(value: Any, location: str) -> list[float]:
    try:
        q = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err("INVALID_VALUE", "a rotation is [x, y, z, w]", location) from None
    if q.shape != (4,) or not np.all(np.isfinite(q)) or float(np.linalg.norm(q)) < 1e-9:
        raise _err("INVALID_VALUE", "a rotation is four finite numbers, not all zero", location)
    return q.tolist()


def _vector_value(value: Any, n: int, spec: P.PathSpec, location: str) -> list[float]:
    if spec.kind == P.BOOL and isinstance(value, bool):
        return [1.0 if value else 0.0]
    if n == 1:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise _err("INVALID_VALUE", f"{spec.path} takes a number, got {value!r}", location)
        v = float(value)
        if not math.isfinite(v):
            raise _err("INVALID_VALUE", "values must be finite", location)
        return [v]
    try:
        arr = np.asarray(value, np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise _err(
            "INVALID_VALUE", f"{spec.path.rsplit('.', 1)[0]} takes {n} numbers", location
        ) from None
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise _err(
            "INVALID_VALUE", f"{spec.path.rsplit('.', 1)[0]} takes {n} finite numbers", location
        )
    return arr.tolist()


def _channel_from(
    spec: P.PathSpec, keys: Any, options: dict, location: str
) -> FCurve | RotationTrack | None:
    if keys is None:
        return None
    if spec.kind == P.ROTATION:
        if isinstance(keys, RotationTrack):
            return keys
        if isinstance(keys, dict):
            return RotationTrack.from_json(keys, location)
        return RotationTrack.from_keys(keys, location=location, **options)
    if isinstance(keys, FCurve):
        return keys
    if isinstance(keys, dict):
        return FCurve.from_json(keys, location)
    if spec.kind == P.BOOL:
        keys = [
            (
                {**k, "value": 1.0 if k["value"] else 0.0}
                if isinstance(k, dict) and isinstance(k.get("value"), bool)
                else k
            )
            for k in keys
        ]
        if options.get("interpolation", "constant") != "constant":
            raise _err(
                "INVALID_INTERPOLATION", f"{spec.path} is on/off: its keys are constant", location
            )
        options = {**options, "interpolation": "constant"}
    return FCurve.from_keys(keys, location=location, **options)


def _definition_json(item: AnimItem) -> Any:
    if item.kind == "skin":
        rig_item = item._system._items.get(item._definition.rig)
        return item._definition.to_json(None if rig_item is None else rig_item._definition)
    return item._definition.to_json()


def _check_settings(raw: Any, location: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_SETTINGS):
        raise _err("INVALID_DOCUMENT", f"settings are {sorted(DEFAULT_SETTINGS)}", location)
    out = copy.deepcopy(DEFAULT_SETTINGS)
    if "fps" in raw:
        out["fps"] = FrameRate(raw["fps"], f"{location}.fps").to_json()
    for k in ("start", "end"):
        if k in raw:
            out[k] = check_time(raw[k], f"{location}.{k}")
    if not out["start"] < out["end"]:
        raise _err("INVALID_RANGE", "the timeline's start must be before its end", location)
    if "enabled" in raw:
        if not isinstance(raw["enabled"], bool):
            raise _err("INVALID_DOCUMENT", "enabled is true or false", f"{location}.enabled")
        out["enabled"] = raw["enabled"]
    if "markers" in raw:
        if not isinstance(raw["markers"], list) or len(raw["markers"]) > MAX_MARKERS:
            raise _err("INVALID_DOCUMENT", "markers is a list", f"{location}.markers")
        out["markers"] = [
            parse_event(m, f"{location}.markers[{i}]").to_json()
            for i, m in enumerate(raw["markers"])
        ]
    if "sections" in raw:
        secs = raw["sections"]
        if not isinstance(secs, list) or len(secs) > MAX_MARKERS:
            raise _err("INVALID_DOCUMENT", "sections is a list", f"{location}.sections")
        clean = []
        for i, s in enumerate(secs):
            loc = f"{location}.sections[{i}]"
            if not isinstance(s, dict) or set(s) != {"name", "start", "end"}:
                raise _err("INVALID_DOCUMENT", "a section is {name, start, end}", loc)
            a, b = check_time(s["start"], f"{loc}.start"), check_time(s["end"], f"{loc}.end")
            if not a < b:
                raise _err("INVALID_RANGE", "a section's start must be before its end", loc)
            clean.append(
                {
                    "name": P.check_name(s["name"], "a section name", f"{loc}.name"),
                    "start": a,
                    "end": b,
                }
            )
        out["sections"] = clean
    return out


def definition_from_json(
    kind: str, data: Any, system: AnimationSystem, location: str, clips: Any = None
) -> Any:
    """Parse an item definition (history files; ``clips`` resolves clip references)."""
    if kind == "clip":
        return Clip.from_json(data, location)
    if kind == "binding":
        return parse_binding(data, location, clips or system._clip_id)
    if kind == "rig":
        from .rig import parse_rig

        return parse_rig(data, location)
    if kind == "skin":
        from .rig import make_skin

        if not isinstance(data, dict) or set(data) - {
            "object",
            "rig",
            "joints",
            "joint_names",
            "inverse_bind",
            "indices",
            "weights",
            "method",
        }:
            raise _err(
                "INVALID_DOCUMENT",
                "a skin is {object, rig, joints, inverse_bind, indices, weights, method}",
                location,
            )
        try:
            return make_skin(
                data["object"],
                data["rig"],
                list(data["joints"]),
                data["inverse_bind"],
                np.asarray(data["indices"]),
                data["weights"],
                data.get("method", "linear"),
                location,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _err(
                "INVALID_DOCUMENT", f"a skin's arrays are malformed ({exc})", location
            ) from None
    if kind == "path":
        from .cinematics import parse_path

        return parse_path(data, location)
    if kind == "sequence":
        from .cinematics import parse_sequence

        return parse_sequence(data, location)
    if kind == "morph":
        from .rig import make_morph

        if (
            not isinstance(data, dict)
            or set(data) != {"object", "targets"}
            or not isinstance(data["object"], str)
        ):
            raise _err("INVALID_DOCUMENT", "a morph item is {object, targets}", location)
        targets = data["targets"]
        count = (
            len(targets[0]["positions"])
            if isinstance(targets, list)
            and targets
            and isinstance(targets[0], dict)
            and isinstance(targets[0].get("positions"), list)
            else 0
        )
        return make_morph(data["object"], targets, count, location)
    if kind == "constraints":
        from .constraints import parse_constraints

        if (
            not isinstance(data, dict)
            or set(data) != {"object", "constraints"}
            or not isinstance(data["object"], str)
        ):
            raise _err("INVALID_DOCUMENT", "a constraints item is {object, constraints}", location)
        return ObjectConstraints(
            data["object"],
            parse_constraints(data["constraints"], f"{location}.constraints", joints=False),
        )
    raise _err("INVALID_DOCUMENT", f"unknown animation item kind {kind!r}", location)


__all__ = [
    "AnimItem",
    "AnimationSystem",
    "ClipHandle",
    "DEFAULT_SETTINGS",
    "Event",
    "Strip",
    "json",
]
