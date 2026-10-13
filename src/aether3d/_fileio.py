"""Small file-system helpers shared by the writers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: str | os.PathLike[str], data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically and return the resolved path.

    The bytes go to a temporary sibling file that is then renamed over the
    target, so an interrupted or failed write never leaves a partial file and
    never damages an existing one.
    """
    target = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return target.resolve()
