"""Block edits: the sparse, immutable store of a block world's authored blocks.

Generation decides every block of a block world; edits record the blocks an author placed,
replaced or removed, and win over generation. They are kept per **section** -- 16 x 16 x 16
blocks, keyed by ``(sx, sy, sz) = (bx // 16, by // 16, bz // 16)`` -- as two read-only
arrays: the local indices of the edited cells (``(lz * 16 + ly) * 16 + lx``, ascending) and
their block ids (:data:`aether3d.worlds.blocks.BLOCK_LIST` indices; ``0`` is air, i.e. a
removed block). A change makes a new :class:`Edits` sharing every untouched section, so the
history keeps edits by reference and undo costs only the sections a step changed.

In project files each section is ``{"section": [sx, sy, sz], "data": base64}``, the data
being zlib-compressed little-endian uint16 pairs (indices, then ids).
"""

from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np

from ..errors import WorldError

SECTION = 16
CELLS = SECTION**3
MAX_EDITED_SECTIONS = 1_000_000


def _frozen(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


class Edits:
    """Immutable sparse block edits (see the module docstring)."""

    __slots__ = ("_sections", "_count")

    def __init__(
        self, sections: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] | None = None
    ) -> None:
        self._sections = dict(sections or {})
        self._count = sum(int(v[0].shape[0]) for v in self._sections.values())

    # -- reading -----------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Edited cells."""
        return self._count

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes + b.nbytes) for a, b in self._sections.values())

    def sections(self) -> list[tuple[int, int, int]]:
        return sorted(self._sections)

    def section(self, key: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
        return self._sections.get(key)

    def payloads(self) -> tuple:
        return tuple(a for pair in self._sections.values() for a in pair)

    def digest_of(self, keys: list[tuple[int, int, int]]) -> str:
        """A content key of the edits in some sections (``""`` without any)."""
        import hashlib

        h = hashlib.sha256()
        found = False
        for key in sorted(keys):
            pair = self._sections.get(key)
            if pair is None:
                continue
            found = True
            h.update(np.asarray(key, dtype=np.int64).tobytes())
            h.update(pair[0].tobytes())
            h.update(pair[1].tobytes())
        return h.hexdigest()[:24] if found else ""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edits):
            return NotImplemented
        if self._sections.keys() != other._sections.keys():
            return False
        for key, (a, b) in self._sections.items():
            c, d = other._sections[key]
            if not (np.array_equal(a, c) and np.array_equal(b, d)):
                return False
        return True

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Edits(sections={len(self._sections)}, cells={self._count})"

    # -- changing ----------------------------------------------------------------------

    def with_cells(
        self,
        bx: np.ndarray,
        by: np.ndarray,
        bz: np.ndarray,
        ids: np.ndarray | None,
        revert: bool = False,
    ) -> Edits:
        """New edits with the cells ``(bx, by, bz)`` (int64 arrays) set to ``ids`` -- or,
        with ``revert``, returned to what generation makes (their edits dropped). Later
        duplicates of a cell win."""
        bx = np.asarray(bx, dtype=np.int64).reshape(-1)
        by = np.asarray(by, dtype=np.int64).reshape(-1)
        bz = np.asarray(bz, dtype=np.int64).reshape(-1)
        if not revert:
            ids = np.asarray(ids, dtype=np.int64).reshape(-1)
            if ids.shape != bx.shape:
                raise WorldError("INVALID_ARGUMENT", "one block id per cell", "ids")
        sx, sy, sz = bx // SECTION, by // SECTION, bz // SECTION
        local = ((bz - sz * SECTION) * SECTION + (by - sy * SECTION)) * SECTION + (
            bx - sx * SECTION
        )
        keys = np.stack([sx, sy, sz], axis=1)
        sections = dict(self._sections)
        if keys.shape[0]:
            uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            for u in range(uniq.shape[0]):
                key = (int(uniq[u, 0]), int(uniq[u, 1]), int(uniq[u, 2]))
                sel = np.flatnonzero(inverse == u)
                cells = local[sel]
                old = sections.get(key)
                table = np.full(CELLS, -1, dtype=np.int64)
                if old is not None:
                    table[old[0].astype(np.int64)] = old[1].astype(np.int64)
                if revert:
                    table[cells] = -1
                else:
                    table[cells] = ids[sel]  # (fancy assignment: the last duplicate wins)
                keep = np.flatnonzero(table >= 0)
                if keep.size:
                    sections[key] = (
                        _frozen(keep.astype(np.uint16)),
                        _frozen(table[keep].astype(np.uint16)),
                    )
                else:
                    sections.pop(key, None)
        if len(sections) > MAX_EDITED_SECTIONS:
            raise WorldError(
                "LIMIT_EXCEEDED", f"more than {MAX_EDITED_SECTIONS:,} edited sections", "edits"
            )
        return Edits(sections)

    # -- files -------------------------------------------------------------------------

    def to_json(self) -> list[dict[str, Any]]:
        out = []
        for key in sorted(self._sections):
            idx, ids = self._sections[key]
            raw = idx.astype("<u2").tobytes() + ids.astype("<u2").tobytes()
            out.append(
                {
                    "section": list(key),
                    "data": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
                }
            )
        return out

    @classmethod
    def from_json(cls, data: Any, location: str, block_count: int) -> Edits:
        if not isinstance(data, list) or len(data) > MAX_EDITED_SECTIONS:
            raise WorldError("INVALID_DOCUMENT", "edits is a list of sections", location)
        sections = {}
        for i, raw in enumerate(data):
            loc = f"{location}[{i}]"
            if not isinstance(raw, dict) or set(raw) != {"section", "data"}:
                raise WorldError("INVALID_DOCUMENT", "a section is {section, data}", loc)
            key = raw["section"]
            if (
                not isinstance(key, list)
                or len(key) != 3
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in key)
            ):
                raise WorldError("INVALID_DOCUMENT", "section is [sx, sy, sz]", f"{loc}.section")
            k = (key[0], key[1], key[2])
            if k in sections:
                raise WorldError("INVALID_DOCUMENT", "duplicate section", f"{loc}.section")
            try:
                raw_bytes = zlib.decompress(
                    base64.b64decode(raw["data"], validate=True), bufsize=4 * CELLS + 16
                )
            except (ValueError, TypeError, zlib.error):
                raise WorldError("INVALID_DOCUMENT", "data is not valid base64 zlib", loc) from None
            if len(raw_bytes) % 4 or len(raw_bytes) > 4 * CELLS:
                raise WorldError("INVALID_DOCUMENT", "data has a wrong length", loc)
            n = len(raw_bytes) // 4
            idx = np.frombuffer(raw_bytes[: 2 * n], dtype="<u2").astype(np.uint16)
            ids = np.frombuffer(raw_bytes[2 * n :], dtype="<u2").astype(np.uint16)
            if n == 0 or np.any(np.diff(idx.astype(np.int64)) <= 0) or int(idx.max()) >= CELLS:
                raise WorldError("INVALID_DOCUMENT", "indices must ascend within a section", loc)
            if int(ids.max()) >= block_count:
                raise WorldError("INVALID_DOCUMENT", "an unknown block id", loc)
            sections[k] = (_frozen(idx), _frozen(ids))
        return cls(sections)
