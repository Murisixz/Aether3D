"""``scene.worlds``: the procedural worlds of a scene.

A world is an *item* of the scene, like an animation clip or a CAD part: an id, a unique
name, an immutable **definition** (:mod:`.definition`) and immutable **edits**
(:mod:`.edits`, block worlds). Every change replaces them in one history operation
(``world.item`` aspect), so undo, redo, rollback and history files treat worlds like the
rest of the scene, and saving writes the definition and edits -- never generated data.

Generated data (the regional stage, tiles, meshes, plants) lives in each world's
:class:`~.runtime.WorldRuntime`, a cache keyed by content: a change regenerates only what
depends on it, and evicting or losing the cache never loses anything authored.
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import WorldError
from . import definition as D
from .edits import Edits

if TYPE_CHECKING:
    from ..scene import Scene

MAX_WORLDS = 64
NAME = re.compile(r"^[^\x00-\x1f]{1,64}$")


def _err(code: str, message: str, location: str | None = None, **kw: Any) -> WorldError:
    return WorldError(code, message, location, **kw)


class World:
    """A handle on a world; it stays valid across undo and redo (see the module docstring).

    Reading: :attr:`definition` (a copy), :attr:`edits`, :meth:`info`. Querying (exact,
    read-only, generating what they need): :meth:`sample`, :meth:`height`, :meth:`ground`,
    :meth:`biome_at`, :meth:`profile`, :meth:`block_at`. Changing (one undo step each):
    :meth:`configure`, :meth:`add_feature`, :meth:`update_feature`, :meth:`remove_feature`,
    :meth:`set_blocks`, :meth:`fill_blocks`, :meth:`replace_blocks`, :meth:`revert_blocks`.
    """

    __slots__ = ("_system", "id", "_name", "_definition", "_edits", "_alive", "_serial")

    def __init__(
        self, system: WorldSystem, wid: str, name: str, definition: dict, edits: Edits
    ) -> None:
        self._system = system
        self.id = wid
        self._name = name
        self._definition = definition
        self._edits = edits
        self._alive = True
        self._serial = system._next_serial()

    # -- reading -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def definition(self) -> dict:
        """A copy of the definition (change it with :meth:`configure`)."""
        return copy.deepcopy(self._definition)

    @property
    def edits(self) -> Edits:
        return self._edits

    @property
    def mode(self) -> str:
        return self._definition["mode"]

    @property
    def seed(self) -> int:
        return int(self._definition["seed"])

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        e = self._definition["extent"]
        return None if e is None else (e["min"][0], e["min"][1], e["max"][0], e["max"][1])

    @property
    def location(self) -> str:
        return f"worlds[{self._name}]"

    def _check(self) -> None:
        if not self._alive:
            raise _err("WORLD_REMOVED", f"world {self._name!r} was removed", self.location)

    @property
    def runtime(self) -> Any:
        """The world's generated-data cache (:class:`~.runtime.WorldRuntime`)."""
        self._check()
        return self._system._runtime(self)

    def info(self) -> dict[str, Any]:
        """A JSON summary: mode, seed, extent, biomes, features, edits, materials, the
        regional stage's estimate, and what is generated and cached now."""
        self._check()
        d = self._definition
        e = self.extent
        rt = self.runtime
        out: dict[str, Any] = {
            "id": self.id,
            "name": self._name,
            "mode": d["mode"],
            "seed": d["seed"],
            "generator": d["generator"],
            "extent": None if e is None else list(e),
            "area_km2": None if e is None else round((e[2] - e[0]) * (e[3] - e[1]) / 1e6, 6),
            "sea_level": d["sea_level"],
            "biomes": list(D.biome_table(d)),
            "hydrology": rt.regional_estimate(),
            "features": [{"id": f["id"], "type": f["type"]} for f in d["features"]],
            "edits": {"sections": len(self._edits.sections()), "blocks": self._edits.count},
            "materials": dict(d["materials"]),
            "cache": rt.stats(),
        }
        if d["mode"] == "blocks":
            b = d["blocks"]
            out["blocks"] = {"size": b["size"], "height": [b["min_height"], b["max_height"]]}
            if e is not None:
                cols = (e[2] - e[0]) * (e[3] - e[1]) / (b["size"] * b["size"])
                out["blocks"]["logical_blocks"] = int(cols * (b["max_height"] - b["min_height"]))
        return out

    # -- queries ---------------------------------------------------------------------------

    def sample(self, points: Any, *, fields: Any = None) -> dict[str, Any]:
        """Fields at points ``[[x, y], ...]``: ``height`` (authoritative), ``normal``,
        ``slope`` (degrees), ``biome`` (dominant), ``biome_weights`` (top 3),
        ``temperature``, ``moisture``, ``water`` (level or ``None``), ``water_kind``.
        ``fields`` limits the answer (default: all)."""
        self._check()
        return self.runtime.sample(points, fields)

    def height(self, x: Any, y: Any) -> Any:
        """The authoritative ground height (float, or an array for array inputs)."""
        self._check()
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        h = self.runtime.heights(xa.reshape(-1), ya.reshape(-1))
        return float(h[0]) if xa.ndim == 0 else h.reshape(xa.shape)

    def ground(self, x: Any, y: Any) -> Any:
        """The ground under points: for terrain worlds, :meth:`height`; for block worlds,
        the top of the highest solid block (water surfaces are not ground)."""
        self._check()
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        g = self.runtime.ground(xa.reshape(-1), ya.reshape(-1))
        return float(g[0]) if xa.ndim == 0 else g.reshape(xa.shape)

    def biome_at(self, x: float, y: float) -> str:
        self._check()
        return self.runtime.sample([[x, y]], ["biome"])["biome"][0]

    def profile(self, points: Any, samples: int = 64) -> dict[str, Any]:
        """Heights, slopes, biomes and water along a polyline (``samples`` evenly spaced
        points): distances, x, y, height, slope, biome, water."""
        self._check()
        return self.runtime.profile(points, samples)

    def block_at(self, x: float, y: float, z: float) -> str:
        """The block at a world position (block worlds): edits first, else generation."""
        self._check()
        if self.mode != "blocks":
            raise _err(
                "NOT_A_BLOCK_WORLD", f"world {self._name!r} is a terrain world", self.location
            )
        return self.runtime.block_at(x, y, z)

    # -- changes ---------------------------------------------------------------------------

    def configure(
        self, patch: dict | None = None, *, label: str | None = None, **sections: Any
    ) -> World:
        """Deep-merge ``patch`` (and keyword sections, e.g. ``terrain={...}``) into the
        definition, validated as a whole; one undo step. Lists (features, curves) replace."""
        self._check()
        p = dict(patch or {})
        p.update(sections)
        if not p:
            return self
        if "mode" in p and p["mode"] != self._definition["mode"]:
            raise _err(
                "INVALID_ARGUMENT",
                "a world's mode cannot change (its materials and edits depend on it)",
                f"{self.location}.mode",
                hint="create another world",
            )
        merged = D.merge(self._definition, p)
        merged["materials"] = {**self._definition["materials"], **(p.get("materials") or {})}
        new = D.check_definition(merged, f"{self.location}.definition")
        self._system._set(self, label or f"Configure world {self._name!r}", definition=new)
        return self

    def rename(self, name: str) -> World:
        self._check()
        self._system._set(self, f"Rename world {self._name!r}", name=_check_name(name))
        return self

    def add_feature(self, feature: dict) -> dict:
        """Add an authored feature (see ``docs/WORLDS.md``); returns it as stored. A
        ``flatten`` feature without a height takes the ground height at its shape's
        centre now (so it stays where it was, whatever changes later)."""
        self._check()
        f = D.check_feature(dict(feature), f"{self.location}.features[new]")
        if any(x["id"] == f["id"] for x in self._definition["features"]):
            raise _err(
                "DUPLICATE_ID", f"a feature with id {f['id']!r} exists", f"{self.location}.features"
            )
        if f["type"] == "flatten" and f["height"] is None:
            cx, cy = _centre(f["shape"])
            f["height"] = round(float(self.height(cx, cy)), 4)
        features = [*self._definition["features"], f]
        self._replace_features(features, f"Add {f['type']} feature {f['id']!r}")
        return copy.deepcopy(f)

    def update_feature(self, fid: str, patch: dict) -> dict:
        self._check()
        features = list(self._definition["features"])
        for k, f in enumerate(features):
            if f["id"] == fid:
                new = D.check_feature(D.merge(f, patch), f"{self.location}.features[{fid}]")
                if new["id"] != fid:
                    raise _err(
                        "INVALID_ARGUMENT",
                        "a feature's id cannot change",
                        f"{self.location}.features[{fid}].id",
                    )
                features[k] = new
                self._replace_features(features, f"Update feature {fid!r}")
                return copy.deepcopy(new)
        raise _err(
            "UNKNOWN_FEATURE",
            f"no feature {fid!r}",
            f"{self.location}.features",
            hint="features: " + ", ".join(f["id"] for f in features[:30]),
        )

    def remove_feature(self, fid: str) -> None:
        self._check()
        features = [f for f in self._definition["features"] if f["id"] != fid]
        if len(features) == len(self._definition["features"]):
            raise _err("UNKNOWN_FEATURE", f"no feature {fid!r}", f"{self.location}.features")
        self._replace_features(features, f"Remove feature {fid!r}")

    def features(self, kind: str | None = None) -> list[dict]:
        return [
            copy.deepcopy(f)
            for f in self._definition["features"]
            if kind is None or f["type"] == kind
        ]

    def _replace_features(self, features: list[dict], label: str) -> None:
        new = dict(self._definition)
        new["features"] = features
        new = D.check_definition(new, f"{self.location}.definition")
        self._system._set(self, label, definition=new)

    # -- block edits -----------------------------------------------------------------------

    def _block_id(self, block: Any, loc: str) -> int:
        from .blocks import BLOCKS

        if isinstance(block, str) and block in BLOCKS:
            return BLOCKS[block]
        raise _err(
            "UNKNOWN_BLOCK",
            f"unknown block {block!r}",
            loc,
            hint="blocks: " + ", ".join(sorted(BLOCKS)),
        )

    def _cells(self, cells: Any, loc: str) -> np.ndarray:
        arr = np.asarray(cells)
        if arr.ndim != 2 or arr.shape[1] != 3 or not np.issubdtype(arr.dtype, np.number):
            raise _err("INVALID_ARGUMENT", "cells is a list of [x, y, z] block coordinates", loc)
        if not np.all(np.isfinite(arr)):
            raise _err("INVALID_ARGUMENT", "cells must be finite", loc)
        out = np.floor(arr.astype(np.float64)).astype(np.int64)
        b = self._definition["blocks"]
        if np.any(out[:, 2] < b["min_height"]) or np.any(out[:, 2] >= b["max_height"]):
            raise _err(
                "OUTSIDE_WORLD",
                f"z must be within [{b['min_height']}, {b['max_height']}) blocks",
                loc,
            )
        return out

    def _block_mode(self) -> None:
        if self.mode != "blocks":
            raise _err(
                "NOT_A_BLOCK_WORLD",
                f"world {self._name!r} is a terrain world",
                self.location,
                hint="block edits apply to block worlds",
            )

    def set_blocks(self, cells: Any, block: str) -> int:
        """Set block cells (``[[x, y, z], ...]`` in block coordinates) to ``block``
        (``"air"`` removes); one undo step. Returns the number of cells."""
        self._check()
        self._block_mode()
        c = self._cells(cells, f"{self.location}.cells")
        bid = self._block_id(block, f"{self.location}.block")
        self._edit_cells(c, np.full(c.shape[0], bid), f"Set {c.shape[0]} blocks to {block}")
        return int(c.shape[0])

    def _box(self, lo: Any, hi: Any, loc: str) -> np.ndarray:
        a = np.floor(np.asarray(lo, dtype=np.float64)).astype(np.int64)
        b = np.floor(np.asarray(hi, dtype=np.float64)).astype(np.int64)
        if a.shape != (3,) or b.shape != (3,) or np.any(b < a):
            raise _err("INVALID_ARGUMENT", "a box is [x0, y0, z0] .. [x1, y1, z1], inclusive", loc)
        n = int(np.prod(b - a + 1))
        if n > MAX_BOX_CELLS:
            raise _err(
                "LIMIT_EXCEEDED", f"the box has {n:,} cells (at most {MAX_BOX_CELLS:,})", loc
            )
        g = np.stack(
            np.meshgrid(*[np.arange(a[k], b[k] + 1) for k in range(3)], indexing="ij"), axis=-1
        ).reshape(-1, 3)
        return self._cells(g, loc)

    def fill_blocks(self, lo: Any, hi: Any, block: str) -> int:
        """Fill the inclusive box ``lo .. hi`` (block coordinates) with ``block``."""
        self._check()
        self._block_mode()
        c = self._box(lo, hi, f"{self.location}.box")
        bid = self._block_id(block, f"{self.location}.block")
        self._edit_cells(c, np.full(c.shape[0], bid), f"Fill {c.shape[0]} blocks with {block}")
        return int(c.shape[0])

    def replace_blocks(self, lo: Any, hi: Any, source: str, block: str) -> int:
        """Replace ``source`` blocks (as they are now) by ``block`` within a box."""
        self._check()
        self._block_mode()
        c = self._box(lo, hi, f"{self.location}.box")
        src = self._block_id(source, f"{self.location}.source")
        bid = self._block_id(block, f"{self.location}.block")
        current = self.runtime.blocks_at(c)
        sel = c[current == src]
        if sel.shape[0]:
            self._edit_cells(
                sel, np.full(sel.shape[0], bid), f"Replace {sel.shape[0]} {source} with {block}"
            )
        return int(sel.shape[0])

    def revert_blocks(self, lo: Any, hi: Any) -> int:
        """Drop the edits in a box (generation decides those blocks again)."""
        self._check()
        self._block_mode()
        c = self._box(lo, hi, f"{self.location}.box")
        new = self._edits.with_cells(c[:, 0], c[:, 1], c[:, 2], None, revert=True)
        if new != self._edits:
            self._system._set(self, f"Revert blocks of world {self._name!r}", edits=new)
        return int(c.shape[0])

    def _edit_cells(self, cells: np.ndarray, ids: np.ndarray, label: str) -> None:
        new = self._edits.with_cells(cells[:, 0], cells[:, 1], cells[:, 2], ids)
        self._system._set(self, label, edits=new)

    def __repr__(self) -> str:
        return f"<world {self._name!r} ({self.mode})>"


MAX_BOX_CELLS = 4_000_000


def _centre(shape: dict) -> tuple[float, float]:
    if "circle" in shape:
        return shape["circle"][0], shape["circle"][1]
    if "rect" in shape:
        r = shape["rect"]
        return (r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0
    pts = np.asarray(shape.get("polygon") or shape["line"], dtype=np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _check_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip() or not NAME.match(name):
        raise _err(
            "INVALID_NAME", f"a world name is 1..64 printable characters, got {name!r}", "name"
        )
    return name


class WorldSystem:
    """The worlds of a scene (``scene.worlds``); see the module docstring."""

    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self._items: dict[str, World] = {}  # every world ever created here (alive or not)
        self._names: dict[str, World] = {}
        self._serial = 0
        self.revision = 0
        self._runtimes: dict[str, Any] = {}

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial

    # -- discovery -------------------------------------------------------------------------

    def __iter__(self) -> Iterator[World]:
        return iter(sorted((w for w in self._items.values() if w._alive), key=lambda w: w._serial))

    def __len__(self) -> int:
        return sum(1 for w in self._items.values() if w._alive)

    def __bool__(self) -> bool:
        return len(self) > 0

    def names(self) -> list[str]:
        return [w._name for w in self]

    def get(self, ref: Any) -> World:
        """A world by handle, id or name."""
        if isinstance(ref, World):
            if self._items.get(ref.id) is ref and ref._alive:
                return ref
        elif isinstance(ref, str):
            found = self._names.get(ref)
            if found is None:
                cand = self._items.get(ref)
                found = cand if cand is not None and cand._alive else None
            if found is not None:
                return found
        raise _err(
            "UNKNOWN_WORLD",
            f"no world {ref!r}",
            "worlds",
            hint="worlds: " + (", ".join(self.names()) or "none"),
        )

    __getitem__ = get

    @staticmethod
    def presets() -> dict[str, str]:
        """Preset names and one-line descriptions."""
        return D.describe_presets()

    # -- changes ---------------------------------------------------------------------------

    def _op(self, label: str):
        return self.scene._history._op(label)

    def _touch(self, world: World) -> None:
        from ..history import WORLD_ITEM

        self.scene._history._touch(WORLD_ITEM, world)

    def _set(
        self,
        world: World,
        label: str,
        *,
        definition: dict | None = None,
        name: str | None = None,
        edits: Edits | None = None,
        alive: bool | None = None,
    ) -> None:
        new_name = world._name if name is None else name
        new_alive = world._alive if alive is None else alive
        if new_alive:
            other = self._names.get(new_name)
            if other is not None and other is not world:
                raise _err(
                    "NAME_CONFLICT", f"a world named {new_name!r} exists", f"worlds[{new_name}]"
                )
        with self._op(label):
            self._touch(world)
            if definition is not None:
                world._definition = definition
            if edits is not None:
                world._edits = edits
            world._name = new_name
            world._alive = new_alive
            self._reindex()
            self.revision += 1

    def _reindex(self) -> None:
        names: dict[str, World] = {}
        for w in self._items.values():
            if w._alive:
                if w._name in names:
                    raise _err("NAME_CONFLICT", f"two worlds would be named {w._name!r}")
                names[w._name] = w
        self._names = names

    def _restored(self) -> None:
        """After undo, redo or a rollback (history fixup)."""
        self._reindex()
        self.revision += 1

    def create(
        self,
        name: str = "World",
        preset: str | None = None,
        *,
        seed: int | None = None,
        size: float | None = None,
        extent: Any = "keep",
        mode: str | None = None,
        overrides: dict | None = None,
        materials: bool = True,
    ) -> World:
        """Create a world from a preset (see :meth:`presets`) with optional ``seed``,
        ``size`` (a square extent of that side, metres), ``extent`` (``{"min", "max"}`` or
        ``None`` for an unbounded world), ``mode`` and ``overrides`` (a deep patch of the
        definition). Its materials are created in the scene's library (``materials=False``:
        layers keep the ids given in ``overrides["materials"]``). One undo step; nothing is
        generated until it is rendered or queried."""
        name = _check_name(name)
        if name in self._names:
            raise _err(
                "NAME_CONFLICT",
                f"a world named {name!r} exists",
                f"worlds[{name}]",
                hint="choose another name or remove the world first",
            )
        if len(self) >= MAX_WORLDS:
            raise _err("LIMIT_EXCEEDED", f"a scene has at most {MAX_WORLDS} worlds", "worlds")
        defn = D.build_definition(
            preset,
            seed=seed,
            mode=mode,
            size=size,
            extent=extent,
            overrides=overrides,
            location=f"worlds[{name}].definition",
        )
        with self._op(f"Create world {name!r}"):
            if materials:
                from .materials import create_materials

                made = create_materials(self.scene, name, defn["mode"], seed=1)
                defn = {**defn, "materials": {**made, **defn["materials"]}}
            world = World(self, str(uuid.uuid4()), name, defn, Edits())
            world._alive = False
            self._items[world.id] = world
            self._set(world, f"Create world {name!r}", alive=True)
        return world

    def remove(self, ref: Any) -> None:
        """Remove a world (one undo step); its materials stay in the library."""
        world = self.get(ref)
        self._set(world, f"Remove world {world._name!r}", alive=False)
        self._runtimes.pop(world.id, None)

    # -- generated data --------------------------------------------------------------------

    def _runtime(self, world: World) -> Any:
        rt = self._runtimes.get(world.id)
        if rt is None:
            from .runtime import WorldRuntime

            rt = self._runtimes[world.id] = WorldRuntime(world)
        return rt

    # -- files -----------------------------------------------------------------------------

    def to_json(self) -> list[dict[str, Any]]:
        out = []
        for w in self:
            out.append(
                {
                    "id": w.id,
                    "name": w._name,
                    "definition": copy.deepcopy(w._definition),
                    "edits": w._edits.to_json(),
                }
            )
        return out

    def _load(self, data: Any, location: str = "worlds") -> None:
        """Read ``to_json``'s form into this empty system (no history)."""
        from .blocks import BLOCK_LIST

        if not isinstance(data, list) or len(data) > MAX_WORLDS:
            raise _err("INVALID_DOCUMENT", f"worlds is a list of at most {MAX_WORLDS}", location)
        for i, raw in enumerate(data):
            loc = f"{location}[{i}]"
            if not isinstance(raw, dict) or set(raw) != {"id", "name", "definition", "edits"}:
                raise _err("INVALID_DOCUMENT", "a world is {id, name, definition, edits}", loc)
            wid = raw["id"]
            if not isinstance(wid, str) or not wid or wid in self._items:
                raise _err("INVALID_DOCUMENT", "world ids must be unique strings", f"{loc}.id")
            name = _check_name(raw["name"])
            if name in self._names:
                raise _err("INVALID_DOCUMENT", f"two worlds are named {name!r}", f"{loc}.name")
            defn = D.check_definition(raw["definition"], f"{loc}.definition")
            for layer, ref in defn["materials"].items():
                if ref not in self.scene.materials._materials:
                    raise _err(
                        "INVALID_DOCUMENT",
                        f"material {ref!r} of layer {layer!r} does not exist",
                        f"{loc}.definition.materials.{layer}",
                    )
            edits = Edits.from_json(raw["edits"], f"{loc}.edits", len(BLOCK_LIST))
            w = World(self, wid, name, defn, edits)
            self._items[wid] = w
            self._names[name] = w
