"""``python -m aether3d.mcp`` is ``aether3d mcp serve``."""

import sys

from ..cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(["mcp", "serve", *sys.argv[1:]]))
