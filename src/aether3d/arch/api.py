"""Python handles for buildings: ``doc.building(...)`` and its floors and elements.

Lengths are numbers in the document's length unit or quantity text (``"3.2 m"``,
``"12' 6\\""``); every call is one undoable step (wrap several in ``doc.batch()``)."""

from __future__ import annotations

import copy
from typing import Any

from ..cad.document import CadDocument, Item
from ..errors import CadError


class FloorHandle:
    def __init__(self, doc: CadDocument, item: Item) -> None:
        self.doc = doc
        self.item = item

    @property
    def name(self) -> str:
        return self.item.name

    def _new_id(self, defn: dict, base: str) -> str:
        used = {x["id"] for key in ("walls", "openings", "rooms") for x in defn.get(key, [])}
        k = 1
        while f"{base}{k}" in used:
            k += 1
        return f"{base}{k}"

    def _change(self, label: str, fn) -> Any:
        defn = copy.deepcopy(self.item._definition)
        out = fn(defn)
        self.doc.set_definition(self.item, defn, label=f"{label} ({self.item.name})")
        return out

    def wall(
        self,
        start=None,
        end=None,
        *,
        type: str | None = None,
        id: str | None = None,
        line=None,
        **options: Any,
    ) -> str:  # noqa: A002
        """A wall from ``start`` to ``end`` ([x, y]) or along a sketch line
        (``line={"sketch": name, "id": line_id}``)."""

        def fn(defn):
            wid = id or self._new_id(defn, "w")
            w: dict[str, Any] = {"id": wid}
            if line is not None:
                w["line"] = line
            else:
                w["start"], w["end"] = list(start), list(end)
            if type is not None:
                w["type"] = type
            w.update(options)
            defn.setdefault("walls", []).append(w)
            return wid

        return self._change("Add wall", fn)

    def walls(
        self, points, *, closed: bool = True, type: str | None = None, prefix: str = "w"
    ) -> list[str]:  # noqa: A002
        """A chain of walls through ``points`` (closed by default)."""

        def fn(defn):
            ids = []
            n = len(points)
            for i in range(n if closed else n - 1):
                wid = self._new_id(defn, prefix)
                w = {"id": wid, "start": list(points[i]), "end": list(points[(i + 1) % n])}
                if type is not None:
                    w["type"] = type
                defn.setdefault("walls", []).append(w)
                ids.append(wid)
            return ids

        return self._change("Add walls", fn)

    def walls_from_sketch(
        self, sketch: Any, lines: list[str] | None = None, *, type: str | None = None
    ) -> list[str]:
        """One wall per (non-construction) line of a sketch: moving the sketch's lines
        (its dimensions) moves the walls."""
        sk = self.doc.item(sketch.item if hasattr(sketch, "item") else sketch, "sketch")
        ids = lines or [
            e["id"]
            for e in sk._definition["entities"]
            if e["type"] == "line" and not e.get("construction")
        ]

        def fn(defn):
            out = []
            for lid in ids:
                wid = (
                    f"{sk.name}.{lid}"
                    if not any(w["id"] == f"{sk.name}.{lid}" for w in defn.get("walls", []))
                    else self._new_id(defn, "w")
                )
                w = {"id": wid, "line": {"sketch": sk.name, "id": lid}}
                if type is not None:
                    w["type"] = type
                defn.setdefault("walls", []).append(w)
                out.append(wid)
            return out

        return self._change("Add walls from sketch", fn)

    def _opening(
        self, kind_label: str, wall: str, family: str | None, id: str | None, options: dict
    ) -> str:  # noqa: A002
        def fn(defn):
            oid = id or self._new_id(defn, kind_label[0])
            o = {"id": oid, "wall": wall, **options}
            if family is not None:
                o["family"] = family
            defn.setdefault("openings", []).append(o)
            return oid

        return self._change(f"Add {kind_label}", fn)

    def door(
        self, wall: str, family: str, *, position=None, id: str | None = None, **options: Any
    ) -> str:  # noqa: A002
        if position is not None:
            options["position"] = position
        return self._opening("door", wall, family, id, options)

    def window(
        self, wall: str, family: str, *, position=None, id: str | None = None, **options: Any
    ) -> str:  # noqa: A002
        if position is not None:
            options["position"] = position
        return self._opening("window", wall, family, id, options)

    def opening(
        self, wall: str, *, width, height, sill=0, position=None, id: str | None = None
    ) -> str:  # noqa: A002
        opts = {"width": width, "height": height, "sill": sill}
        if position is not None:
            opts["position"] = position
        return self._opening("opening", wall, None, id, opts)

    def room(self, name: str, point, *, number: str = "", id: str | None = None) -> str:  # noqa: A002
        def fn(defn):
            rid = id or self._new_id(defn, "r")
            defn.setdefault("rooms", []).append(
                {"id": rid, "name": name, "number": number, "point": list(point)}
            )
            return rid

        return self._change("Add room", fn)

    def set(self, element_id: str, **changes: Any) -> None:
        """Change a wall, opening or room (``None`` removes a setting)."""

        def fn(defn):
            for key in ("walls", "openings", "rooms"):
                for x in defn.get(key, []):
                    if x["id"] == element_id:
                        for k, v in changes.items():
                            if v is None:
                                x.pop(k, None)
                            else:
                                x[k] = v
                        return
            raise CadError(
                "UNKNOWN_ELEMENT", f"the floor has no wall, opening or room {element_id!r}"
            )

        self._change(f"Change {element_id}", fn)

    def remove(self, *element_ids: str) -> None:
        """Remove walls (with their openings), openings or rooms."""
        doomed = set(element_ids)

        def fn(defn):
            defn["walls"] = [w for w in defn.get("walls", []) if w["id"] not in doomed]
            alive = {w["id"] for w in defn["walls"]}
            defn["openings"] = [
                o for o in defn.get("openings", []) if o["id"] not in doomed and o["wall"] in alive
            ]
            defn["rooms"] = [r for r in defn.get("rooms", []) if r["id"] not in doomed]

        self._change("Remove elements", fn)

    @property
    def result(self):
        return self.doc.result(self.item)

    def quantities(self) -> dict[str, Any]:
        from .quantities import floor_quantities

        return floor_quantities(self.doc, self.result)


class BuildingHandle:
    def __init__(self, doc: CadDocument, item: Item) -> None:
        self.doc = doc
        self.item = item

    @property
    def name(self) -> str:
        return self.item.name

    def floor(self, level: str, name: str | None = None) -> FloorHandle:
        """The floor (walls, openings, rooms) of a level, created on first use."""
        name = name or f"{self.item.name} {level}"
        if name in self.doc._by_name:
            return FloorHandle(self.doc, self.doc.item(name, "floor"))
        item = self.doc.add(
            "floor",
            name,
            {"building": self.item.name, "level": level, "walls": [], "openings": [], "rooms": []},
        )
        return FloorHandle(self.doc, item)

    def _element(self, kind: str, name: str | None, definition: dict) -> Item:
        return self.doc.add(
            kind,
            name or self.doc._unique_name(f"{self.item.name} {kind}"),
            {"building": self.item.name, **definition},
        )

    def slab(
        self, level: str, *, outline=None, boundary=None, name: str | None = None, **options: Any
    ) -> Item:
        d: dict[str, Any] = {"level": level, **options}
        if outline is not None:
            d["outline"] = {"floor": outline.name if hasattr(outline, "name") else outline}
        else:
            d["boundary"] = boundary
        return self._element("slab", name, d)

    def roof(
        self, level: str, *, outline=None, boundary=None, name: str | None = None, **options: Any
    ) -> Item:
        d: dict[str, Any] = {"level": level, **options}
        if outline is not None:
            d["outline"] = {"floor": outline.name if hasattr(outline, "name") else outline}
        else:
            d["boundary"] = boundary
        return self._element("roof", name, d)

    def stair(
        self, from_level: str, to_level: str, *, name: str | None = None, **options: Any
    ) -> Item:
        return self._element("stair", name, {"from": from_level, "to": to_level, **options})

    def columns(
        self, from_level: str, *, to: str | None = None, name: str | None = None, **options: Any
    ) -> Item:
        d = {"from": from_level, **options}
        if to is not None:
            d["to"] = to
        return self._element("columns", name, d)

    def beams(self, level: str, lines, *, name: str | None = None, **options: Any) -> Item:
        return self._element("beams", name, {"level": level, "lines": lines, **options})

    def set(self, **changes: Any) -> None:
        self.doc.update(self.item, changes)

    def set_level(self, level: str, **changes: Any) -> None:
        defn = copy.deepcopy(self.item._definition)
        for lv in defn["levels"]:
            if lv["name"] == level:
                lv.update(changes)
                self.doc.set_definition(self.item, defn, label=f"Change level {level!r}")
                return
        raise CadError("UNKNOWN_LEVEL", f"the building has no level {level!r}")

    def schedule(self, kind: str = "all") -> list[dict[str, Any]]:
        from .quantities import schedule

        return schedule(self.doc, kind)


def _install() -> None:
    def building(self: CadDocument, name: str, *, levels=None, **options: Any) -> BuildingHandle:
        """Create (or get) a building. ``levels``: [(name, elevation, height), ...] or
        [{"name", "elevation", "height"}, ...]."""
        if name in self._by_name:
            return BuildingHandle(self, self.item(name, "building"))
        lv = []
        for x in levels or [("Ground", 0, "3 m")]:
            if isinstance(x, dict):
                lv.append(x)
            else:
                lv.append({"name": x[0], "elevation": x[1], "height": x[2]})
        return BuildingHandle(self, self.add("building", name, {"levels": lv, **options}))

    CadDocument.building = building


_install()
