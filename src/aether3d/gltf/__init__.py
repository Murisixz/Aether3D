"""glTF 2.0 interchange: import and export of static geometry.

See ``docs/GLTF.md`` for the supported feature set, the coordinate-system
conversion, and limitations.
"""

from .reader import ImportLimits, ImportResult, import_gltf
from .writer import EXTRAS_KEY, ExportResult, export_gltf

__all__ = [
    "EXTRAS_KEY",
    "ExportResult",
    "ImportLimits",
    "ImportResult",
    "export_gltf",
    "import_gltf",
]
