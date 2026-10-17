"""Drawings of CAD documents: plans, sections, elevations and isometric views with
hidden lines, section fills, dimensions and text; output as SVG, DXF and PNG.

See :mod:`.views` for what is drawn and how visibility is decided, :mod:`.model` for
the drawing itself and its line styles, :mod:`.kind` for the ``drawing`` item kind.
"""

from .model import Drawing
from .views import VIEWS, Cut, Solid, ViewFrame, gather, project

__all__ = ["VIEWS", "Cut", "Drawing", "Solid", "ViewFrame", "gather", "project"]
