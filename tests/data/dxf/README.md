# DXF fixture written by ezdxf

`ezdxf_plate.dxf` was written by [ezdxf](https://ezdxf.mozman.at/) 1.4 (an independent
DXF library, MIT licensed; not a dependency of Aether3D) to test Aether3D's DXF reader
on another writer's output: DXF R2018, `$INSUNITS` millimetres, a 100 x 60 plate whose
corners are rounded (radius 10) by bulged `LWPOLYLINE` segments, a circular hole
(radius 8), a slot of two lines and two arcs -- one of them written in a mirrored
object coordinate system (extrusion -Z) -- and a text and a block reference, which the
reader skips and reports. It was generated once with the script below.

```python
import math
import sys

import ezdxf

doc = ezdxf.new("R2018", setup=True)
doc.header["$INSUNITS"] = 4  # millimetres
msp = doc.modelspace()
b = math.tan(math.pi / 8)  # a quarter circle
msp.add_lwpolyline(
    [
        (10, 0, 0, 0, 0),
        (90, 0, 0, 0, b),
        (100, 10, 0, 0, 0),
        (100, 50, 0, 0, b),
        (90, 60, 0, 0, 0),
        (10, 60, 0, 0, b),
        (0, 50, 0, 0, 0),
        (0, 10, 0, 0, b),
    ],
    format="xyseb",
    close=True,
    dxfattribs={"layer": "OUTLINE"},
)
msp.add_circle((30, 30), 8, dxfattribs={"layer": "HOLES"})
msp.add_line((60, 25), (80, 25), dxfattribs={"layer": "HOLES"})
msp.add_line((80, 35), (60, 35), dxfattribs={"layer": "HOLES"})
msp.add_arc((80, 30), 5, -90, 90, dxfattribs={"layer": "HOLES"})
msp.add_arc((-60, 30), 5, -90, 90, dxfattribs={"layer": "HOLES", "extrusion": (0, 0, -1)})
msp.add_text("PLATE", dxfattribs={"layer": "NOTES", "height": 5}).set_placement((5, 65))
blk = doc.blocks.new("MARK")
blk.add_circle((0, 0), 1)
msp.add_blockref("MARK", (50, 70))
doc.saveas(sys.argv[1])
```
