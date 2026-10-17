# Buildings: architecture and BIM (0.15.0)

Parametric building elements on Aether3D's exact B-rep kernel ([CAD.md](CAD.md)): levels,
walls with exact joins, doors and windows from families, rooms found from the walls,
slabs, roofs, stairs, columns and beams -- with quantities from the IFC base quantity
definitions, schedules, plans and sections, and IFC4 exchange. Every element is an item
of the scene's CAD document: one undo step per change, regenerated incrementally, shown
in the scene as objects that follow it.

## Quick start

```python
from aether3d import Scene

scene = Scene()
doc = scene.cad
doc.set_units(length="m")
house = doc.building(
    "House",
    levels=[("Ground", 0, 3.0), ("First", 3.0, 3.0)],
    wall_types={
        "Ext": {"thickness": "300 mm", "function": "exterior"},
        "Int": {"thickness": "120 mm"},
    },
    families={
        "D90": {"kind": "door", "width": "900 mm", "height": "2100 mm"},
        "W12": {"kind": "window", "width": "1200 mm", "height": "1400 mm", "sill": "900 mm"},
    },
)
ground = house.floor("Ground")
outer = ground.walls([(0, 0), (10, 0), (10, 8), (0, 8)], type="Ext")  # one closed chain
ground.wall((6, 0), (6, 8), type="Int", id="mid")
ground.door(outer[0], "D90", position=3.0)
ground.window(outer[1], "W12", position=4.0)
ground.door("mid", "D90", position=4.0)
ground.room("Living", (3, 4), number="001")
ground.room("Bedroom", (8, 4), number="002")
house.slab("Ground", outline=ground, thickness=0.25)
house.roof(
    "First", outline=ground, type="gable", slope="30 deg", overhang=0.5, ridge="x", thickness=0.2
)

q = ground.quantities()
print([round(r["quantities"]["NetFloorArea"], 4) for r in q["rooms"]])  # [44.583, 29.183]
print(q["totals"]["walls"], q["totals"]["doors"], q["totals"]["windows"])  # 5 2 1
print(round(q["gross_floor_area"], 4))  # 85.49

doc.drawing("Plan", "plan", level="Ground", scale="1:50")
ifc, products = doc.export_ifc(name="House")
print(products["IfcWall"], products["IfcSpace"])  # 5 2
```

`doc.show(item)` puts any element in the scene (walls, doors and windows of a floor as
child objects with their materials); `scene.render(...)` renders them like any mesh.

## Items

| kind | definition (lengths in the document's unit, or quantities) |
| --- | --- |
| `building` | `levels` (name, elevation, height), `wall_types` (`thickness` or `layers` of `{name, thickness, material}`, `function: exterior`, `material`), `families`, `materials` (role -> scene material), `criteria` (design criteria for stairs), `grids`, `site` (`origin`, `rotation`) |
| `floor` | `building`, `level`, `walls`, `openings`, `rooms` |
| `slab` | `building`, `level`, `outline: {"floor": name}` or `boundary` (points), `holes`, `thickness`, `top_offset` |
| `roof` | `building`, `level`, `type` (`flat`, `shed`, `gable`, `hip`), `slope`, `overhang`, `ridge` (`x`, `y`), `direction` (shed), `thickness`, outline or boundary |
| `stair` | `building`, `from`, `to`, `start`, `direction`, `width`, `tread`, `riser` (target), `waist`, `enforce` |
| `columns` | `building`, `from`, `to` or `height`, `points` or `grid` (`x`, `y` lists), `profile` (`rect` width/depth, `circle` diameter, `I` width/depth/web/flange) |
| `beams` | `building`, `level`, `lines` (pairs of points), `profile`, `top_offset` |

`doc.building(...)` returns a `BuildingHandle` (`floor`, `slab`, `roof`, `stair`,
`columns`, `beams`, `set`, `set_level`, `schedule`); `handle.floor(level)` a
`FloorHandle` (`wall`, `walls`, `walls_from_sketch`, `door`, `window`, `opening`,
`room`, `set`, `remove`, `quantities`). Any value may be `"=expression"` over document
parameters: a structural grid whose coordinates are `"=2*bay_x"` follows `bay_x`
(`examples/cad_grid_building.py`).

## Walls

A wall is a straight **location line** (`start`, `end`, or a sketch line: `line:
{"sketch": ..., "id": ...}` -- dimension the sketch and the walls follow), a type (or a
`thickness`), where the thickness lies (`location`: `center`, `left`, `right`), a `height`
(default: to the level above) and a `base_offset`.

**Joins are exact, in plan.** Wall ends meeting at a point (L corners, Y joints,
junctions of three or more walls) are mitred so their footprints tile the junction with
no gap and no overlap, for any thicknesses and angles; a wall ending on another wall's
line stops at its face (T join); free ends are square. Walls crossing away from their
ends are refused (`WALLS_CROSS`: split one of them at the crossing) rather than
overlapped. Each wall's footprint, joins and solid are in its result; its solid is the
footprint extruded, minus its openings.

## Openings, doors and windows

An opening is hosted by a wall at a `position` along it (its centre; or `offset` from
the start), sized by a **family** (`door`, `window`, or `opening`: width, height, sill,
frame, leaf, mullions, glass) with per-instance overrides; `flip` swings a door the
other way. Openings are checked against their wall (inside its straight, unjoined
part: `OPENING_OUTSIDE_WALL`; under its top: `OPENING_TOO_TALL`; not overlapping each
other: `OPENING_OVERLAP`). The door or window body (frame, leaf, glass and mullions as
exact boxes, with material groups) is built in the opening's frame, so it follows any
move of the wall or the opening; identical instances share one display mesh.

## Rooms

A room is a name, a number and a **point**: its region is the area bounded by wall
faces around the point (an exact arrangement of the walls' footprints), so rooms follow
the walls. A point outside closed walls is `ROOM_NOT_ENCLOSED`; inside a wall,
`ROOM_IN_WALL`. Net floor area, perimeter, height and volume are in its quantities.

## Slabs, roofs, stairs, columns, beams

* **Slabs**: a floor's outline (the outside of its exterior walls) or a polygon, with
  holes (stair openings), a thickness downward from the level plus `top_offset`.
* **Roofs**: `flat` (a slab), `shed` (one plane), `gable` (two planes meeting at a ridge
  along `x` or `y`), `hip` (every edge rises at the slope: exact for convex footprints);
  a shell of the given thickness under the roof surface, built with exact planar
  booleans; `overhang` grows the footprint.
* **Stairs**: straight flights between two levels; the number of risers follows from the
  storey height and the target riser, all risers equal. The riser, tread, Blondel's
  2R + T and the width are checked against the building's `criteria` -- values *you*
  give (`riser_min`, `riser_max`, `tread_min`, `tread_max`, `blondel_min`,
  `blondel_max`, `stair_width_min`, `stair_width_max`); the result lists every check with its value and limits; `"enforce":
  true` makes a failed check an error. Aether3D assumes no building code and claims no
  code compliance.
* **Columns** at points or grid intersections, **beams** along lines, with rectangular,
  circular or I profiles; members of one size share one mesh.

## Quantities and schedules

Quantities come from the parameters, with the formulas of the IFC4 base quantity sets
(`Qto_WallBaseQuantities`: Length, Width, Height, GrossFootprintArea, NetFootprintArea,
GrossSideArea, NetSideArea, GrossVolume, NetVolume; door, window, space, slab, roof,
column and beam sets likewise), and are checked in the tests against the measured
volumes of the B-reps. `floor.quantities()` reports walls, openings, rooms and totals;
`schedule(doc, kind)` (`aether3d.arch.quantities`; kinds `all`, `walls`, `doors`,
`windows`, `rooms`, `slab`, `roof`, `stair`, `columns`, `beams`) gives one row per
element, and `to_csv(rows)` writes them.

## Drawings

`doc.drawing(name, "plan", level=..., cut_height="1.2 m")` draws a level cut above its
floor: cut walls and elements hatched (exact cut faces), what lies below in thin lines,
door swings, room names, numbers and areas, and overall dimensions. Sections
(`"section"`, any plane) and elevations (`"front"`, `"back"`, `"left"`, `"right"`) work on
any items; see [CAD.md](CAD.md#drawings) for views, sheets, scales and output (SVG, DXF,
PNG).

## IFC

`doc.export_ifc(building=None, name=...)` writes IFC4 (ISO 10303-21): the project, site,
building and storeys; `IfcWall` (swept solids of their footprints, axis
representations, wall types, material layer sets), `IfcOpeningElement` voiding them,
`IfcDoor` and `IfcWindow` filling them, `IfcSpace` per room, `IfcSlab`, `IfcRoof`,
`IfcStair`, `IfcColumn`, `IfcBeam` -- with their base quantities, `Pset_WallCommon`,
`Pset_StairCommon` and an `Aether3D_Identity` set naming the CAD item and element.
GlobalIds are deterministic (UUID 5 of the item and element ids): exporting the same
document again, also after saving and loading it, gives the same ids, so external tools
can track elements across exports. Items get new ids when they are created, so a script
that builds a new document gets new GlobalIds.

`doc.import_ifc(text, create_parts=True)` reads the products of an IFC4 file whose
geometry is extruded area solids (rectangle, circle, I, arbitrary profiles with voids) or
faceted B-reps, with openings subtracted from their hosts, their storeys, property and
quantity sets; each becomes a part (`create_parts=False` only reports them). Other
geometry is reported per product (`unsupported`), IFC2X3 files are refused.

**Validated.** IfcOpenShell's validation (schema and EXPRESS rules) reports no issue on
the exports of the tests and examples, and the volumes IfcOpenShell computes from them
match Aether3D's (test `test_ifcopenshell_validates_exported_ifc`, with
`AETHER3D_VALIDATOR_PYTHON`; [CAD.md](CAD.md#exchange-step-dxf-ifc)).

## Performance

Elements regenerate only when they or what they read change; within a floor, room sets
and door and window bodies are reused by content (moving a door on a floor of 220 walls
takes about 0.13 s on the development laptop, 1.25 s before these caches). Figures for buildings of up to 2,200
walls, 1,000 doors and 1,000 rooms are in
[PERFORMANCE.md](PERFORMANCE.md#precision-modeling-015).

## What is not supported

Curved walls; walls crossing without a junction; sloped or stepped walls; curtain walls;
spiral, winding or L-shaped stairs and landings; ramps and railings; roof shapes other
than flat, shed, gable and hip (and hip only on convex footprints); structural analysis;
MEP; building-code checks; IFC property and quantity sets beyond the base sets, IFC
classifications, and IFC geometry other than extrusions and faceted B-reps.
