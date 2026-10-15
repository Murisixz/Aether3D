# Operations history: undo, redo and transactions

Every change to a scene is recorded in its history, `scene.history`: changes made
through the Python API, by scripts or from the command line. `undo()` and `redo()`
reverse and re-apply them exactly. The scene then has the same project document,
the same Python objects and the same renders, pixel for pixel.

```python
from aether3d import Scene

scene = Scene("Workshop")
crate = scene.add_cube(name="Crate")  # step 1
crate.position = (2, 0, 0.5)  # step 2
crate.color = "#c0392b"  # step 3
with scene.transaction("Build the gate"):  # step 4: several changes, one step
    scene.add_cube(name="Post A", position=(-1, 3, 1))
    scene.add_cube(name="Post B", position=(1, 3, 1))

scene.history.undo()  # the gate is gone
scene.history.undo(2)  # the crate is back at the origin, in its old color
scene.history.redo(3)  # ... and everything returns
print([e.label for e in scene.history.entries])
# ["Add object 'Crate'", "Set position of 'Crate'",
#  "Set appearance color of 'Crate'", 'Build the gate']
```

Contents:
- [What is recorded](#what-is-recorded)
- [How Python changes become steps](#how-python-changes-become-steps)
- [Undo and redo](#undo-and-redo)
- [Transactions](#transactions)
- [Checkpoints](#checkpoints)
- [Limits and memory](#limits-and-memory)
- [Identity, references and caches](#identity-references-and-caches)
- [Saving the history with the project](#saving-the-history-with-the-project)
- [Command line](#command-line)
- [Scripts](#scripts)
- [Merging scenes and duplicating objects](#merging-scenes-and-duplicating-objects)
- [Other undo systems](#other-undo-systems)
- [Errors](#errors)
- [Design](#design)
- [Limitations](#limitations)

## What is recorded

| Area | Changes that are steps |
| --- | --- |
| Objects | create (`add_object`, `add_cube`, ..., `add_camera`, `add_light`, `add_procedural`), remove (with or without children), `duplicate`, `merge`, rename |
| Hierarchy | `set_parent` (with `keep_world_transform`), the order of children and roots |
| Transforms | position, rotation, Euler rotation, scale, `obj.transform = ...`, `look_at`, `look_along`, `set_world_position`, `frame` |
| Appearance | visibility, color, specular, shininess, `obj.appearance = ...` |
| Cameras and lights | every parameter (`obj.camera.fov = 35`, `obj.light.configure(...)`), the active camera |
| World and settings | background, ambient light, the environment and its fields, render settings, the scene's name |
| Meshes | `obj.mesh = ...`, committed edit sessions (`obj.edit_mesh()`, with `scope="shared"` too) |
| Materials | create (presets too), every parameter, model, name, texture bindings and their options, remove, assign (whole objects and face groups), `convert_appearances` |
| Textures | add (images, procedural graphs), name, sampler settings, graph and node parameters, embed and extract, remove |
| Procedural geometry | parameters and rules (scene and geometry), geometry nodes, sources (and `edit_source()` commits), modifiers (add, set, move, remove), assigning nodes to objects, `bake()`, `procedural.undo()` |
| Scripts | a scene, edit or procedural script is one step |
| Command line | every command that changes a project, when the project keeps a history (see [Command line](#command-line)) |

These are the properties a future animation system would animate (transforms,
visibility, colors, camera and light parameters, material parameters, procedural
parameters). The history records them under stable, id-based paths:
`objects[<id>].transform`, `objects[<id>].camera`, `materials[<id>]`, `world`,
`procedural`, ... (`HistoryEntry.paths`).

**Not recorded:**
- Effects outside the scene: saved projects, exported files, an image written by
  `texture.extract(path)` (undo restores the texture, not the file system), renders.
- Caches and counters: evaluation caches, `materials.version`,
  `procedural.revision`, `scene.revision` (they increase with undo and redo as
  with any change).
- The edits inside an open edit session. They have their own undo (see
  [Other undo systems](#other-undo-systems)); committing the session is one step.
- Changes made while recording is disabled (see [Limits and memory](#limits-and-memory)).

## How Python changes become steps

- **Every public call that changes the scene is one operation**, and one operation is
  one step. Calls it makes are part of it: `add_camera` creates an object, aims it
  and may activate it, all in one step, "Add camera 'Cam'".
- **In-place edits of value objects** that belong to the scene are recorded:
  `obj.camera.fov = 35`, `obj.light.intensity = 2`, `obj.transform.position = ...`,
  `scene.world.environment.intensity = 3`, `scene.render_settings.width = 640`.
  A value object that belongs to no scene (`a3.Camera()`, `a3.Transform()`)
  records nothing. Assigning it (`obj.transform = t`) copies it into the scene: one
  step.
- **Removed objects keep recording.** An edit to a removed object (which undo may
  bring back) is a step of its scene's history. This keeps undoing past the
  removal exact.
- **Nothing changes, nothing is recorded:** setting a value to what it already is
  makes no step.
- **Failed operations make no step and change nothing.** An operation that raises
  is rolled back, as a whole, before the exception propagates: the scene, its
  revision and its history are as they were.
- **Private attributes (`obj._mesh = ...`) are not part of the API**, and changes
  made through them are not recorded.

## Undo and redo

| | |
| --- | --- |
| `history.undo(steps=1)` | Reverse the last `steps` steps; returns their entries, the most recent first |
| `history.redo(steps=1)` | Re-apply the last `steps` undone steps; returns their entries |
| `history.can_undo`, `history.can_redo` | Whether there is anything to undo or redo |
| `history.entries` | The steps `undo` reverses, oldest first (`HistoryEntry`: `index`, `label`, `changes`, `paths`, `bytes`) |
| `history.redo_entries` | The undone steps, the next to redo first |
| `history.to_dict()` | A JSON summary: steps, undone steps, checkpoints, limits, memory |
| `scene.revision` | Increases with every change, undo and redo included (to know when to re-render) |

- **All or nothing.** `undo(3)` either undoes three steps or, if it cannot (fewer are
  recorded: `NOTHING_TO_UNDO`), changes nothing.
- **Undo and redo never record themselves.**
- **A new change discards the redo branch** (the undone steps), and the checkpoints
  on it. The history is linear: there is no undo tree.

## Transactions

```python
with scene.transaction("Build the tower") as tx:
    base = scene.add_cube(name="Base", size=4)
    top = scene.add_cube(name="Top", size=2, parent=base, position=(0, 0, 3))
    top.material = scene.create_material("Slate", preset="stone")
print(tx.entry.label, tx.entry.changes)  # one step
```

- **One step.** Everything changed in the block is one step (`tx.entry`; `None` if
  nothing changed).
- **Atomic.** If the block raises, everything it changed is restored and the
  exception propagates.
- **Savepoints.** Transactions nest. An inner one that fails is rolled back alone if
  the outer block handles its exception; the outer one goes on.
- **`tx.rollback()`** restores the scene as it was when the transaction started; the
  block may go on making changes.
- `scene.transaction(label)` is `scene.history.transaction(label)`.
  `scene.materials.transaction()` and `scene.procedural.batch()` are transactions of
  the scene's history too.
- `undo()` and `redo()` inside an open transaction raise `TRANSACTION_OPEN`.

## Checkpoints

```python
scene.history.checkpoint("walls done")
# ... more changes ...
scene.history.goto("walls done")  # undo (or redo) until that state
```

- **Free.** A checkpoint is a name for a position in the history; it costs no memory.
- **Both directions.** `goto` undoes or redoes (all or nothing) and returns the
  steps it moved.
- **Inspection and removal.** `checkpoints` maps names to the index of the step
  behind their state (0 means no step), and `remove_checkpoint(name)` forgets one.
- **Unreachable checkpoints are forgotten:** those whose steps were dropped
  (limits), or discarded (a new change after undoing past them).

## Limits and memory

| Setting | Default | |
| --- | --- | --- |
| `history.max_entries` | 256 | Undo steps kept; the oldest are dropped first |
| `history.max_bytes` | 256 MiB | Memory budget; the oldest steps are dropped to respect it |

Change them with `history.configure(max_entries=..., max_bytes=...)`; steps beyond
the new limits are dropped at once. `history.dropped` counts the steps dropped so
far, and `history.nbytes` estimates what the history holds.

**No copies of the scene.**
- **Small states.** A step stores, for each part of the scene it changed, that
  part's state before and after: an object's transform, its camera, its place in
  the hierarchy, a material, the procedural definitions.
- **Shared data.** Meshes, image bytes and texture graphs are immutable and shared
  by reference, never copied. Consecutive steps that change the procedural
  definitions share one copy of them.
- **What `nbytes` counts:** the states, plus the meshes and images that only the
  history still refers to. A mesh the scene still uses costs the history nothing.

**When a step exceeds the budget:** the history is cleared rather than holding
more. For example, replacing an object's large mesh makes a step that holds the
old mesh. If that mesh alone is over `max_bytes`, the step cannot be kept, and
nothing older can be undone either.

**Disabling recording.**
- `with scene.history.disabled():` (or `history.enabled = False`) records nothing,
  for example to build a large scene.
- Disabling clears the history: changes made while it is disabled cannot be undone,
  so nothing before them can be either.
- Operations stay atomic: a failing call or transaction is still rolled back.
- Loading a project or importing a glTF file starts an empty history.

**Cost.** Measured on an i9-12900H (Python 3.14, one run of 20,000 changes):
- Setting a position takes about 5 µs with recording, against 2 µs before 0.9.0
  (4 µs with recording disabled: operations stay atomic).
- Adding a cube takes 184 µs, against 163 µs before 0.9.0.
- Building the textured castle of the path tracing examples (108 steps) takes the
  same 0.31 s, and its history holds 0.9 MiB.
- On the command line, changing that castle saved as a project (229 KiB, with
  a history file of 19 steps, 9 KiB) takes 0.21 s instead of 0.18 s. Loading
  and checking the history is the difference.

## Identity, references and caches

- **The same Python objects come back.** Undoing a removal re-inserts the same
  `SceneObject` (same id, name, children and handle) at the same place among its
  siblings. Materials, textures, procedural geometry nodes and modifier handles you
  hold, and an object's camera, light, appearance and transform, come back the same
  way. Values change in place, so a handle like `cam = obj.camera` stays live.
- **References are restored with what they point to.** Material assignments,
  texture bindings, procedural assignments, the active camera, and the order of
  children, roots, materials and textures.
- **Caches stay valid.** Renderer caches are keyed by immutable content: GPU mesh
  buffers, ray tracing BVHs, decoded images, GPU texels and procedural evaluations.
  So undo reuses them: a render after undo uploads nothing new and finds the old
  procedural evaluations cached. `materials.version`, `procedural.revision` and
  `scene.revision` increase with undo and redo like any change, for tools that
  cache by version.
- **Renders after undo are identical.** The tests render with the CPU rasterizer,
  the CPU path tracer, the GPU rasterizer and the GPU path tracer: after undo the
  images equal the originals pixel for pixel, and after redo the edited ones.

## Saving the history with the project

```python
scene.save("castle.aether", history=True)  # also writes castle.aether.history
later = Scene.load("castle.aether", history=True)  # a later session can undo
later.history.undo()
```

- **The history file.** `<project>.history` holds the steps (undone ones too), the
  checkpoints and the limits. It also holds what the steps refer to that the
  project does not: removed objects, materials and textures, old meshes, embedded
  images. The format is described in
  [FILE_FORMAT.md](FILE_FORMAT.md#history-files).
- **It must match its project.** The file records a fingerprint (SHA-256) of the
  project document. `Scene.load(path, history=True)` refuses a history that does
  not match the project (`HISTORY_MISMATCH`, for example after the project was
  saved without it) or a missing one (`NO_HISTORY`). The project itself always
  loads.
- **It is checked when loaded.** The history is first replayed on a copy of the
  scene: every step is undone and redone, with the scene's invariants checked after
  each. It is refused (`INVALID_HISTORY`) unless the copy comes back to the same
  document. A damaged or edited file therefore cannot corrupt a scene.
- `history.save(path)` and `history.load(path)` work with any path.
- `history.verify()` replays the current history in the same way, on a copy,
  through the file format. It returns `{"steps", "undone_and_redone",
  "redone_and_undone", "base", "ok"}`, and leaves the scene untouched.

## Command line

A project keeps a history once it has a history file. Then every command that
changes the project records its change there; the step's label is the command.

```console
$ aether3d history start castle.aether        # castle.aether.history
$ aether3d object set castle.aether Crate --position 2 0 0.5
$ aether3d material assign castle.aether Stone Crate
$ aether3d history checkpoint castle.aether "crate placed"
$ aether3d scene merge castle.aether props.glb --allow-lossy
$ aether3d history show castle.aether
History of castle.aether: 3 step(s) to undo, 0 to redo, 0.1 MiB
     1  aether3d object set
     2  aether3d material assign  [checkpoint 'crate placed']
     3  aether3d scene merge props.glb
  ----  (current state)
$ aether3d history undo castle.aether          # -n N for several steps
$ aether3d history goto castle.aether "crate placed"
$ aether3d history redo castle.aether
$ aether3d history verify castle.aether        # replay on a copy: exact?
```

| Command | |
| --- | --- |
| `history start PROJECT` | Start recording (an empty history file; `--force` replaces one) |
| `history show PROJECT` | The steps, undone steps, checkpoints and memory (`--json`) |
| `history undo PROJECT [-n N]` | Undo and save the project and its history (`--dry-run`) |
| `history redo PROJECT [-n N]` | Redo |
| `history checkpoint PROJECT NAME` | Name the current state |
| `history goto PROJECT NAME` | Undo or redo to a checkpoint |
| `history clear PROJECT` | Forget every step; recording goes on |
| `history verify PROJECT` | Replay the history on a copy; exit status 1 unless exact |

- **Starting with a change.** `--history` on any command that changes a project
  records that change and starts the history file if there is none.
- **A stale history is set aside.** If a project was changed without its history
  (by another program, say), its old history no longer matches. The next command
  that changes the project reports that and starts a new history from the
  project's current state; the old one is not applied.

## Scripts

- **One step per script.** A scene script, an edit script or a procedural script is
  one transaction. Applied whole or not at all, it is one undo step.
- **Labels.** All three formats accept an optional `"label"` that names the step,
  as do the `label=` arguments of `run_scene_script`, `run_edit_script` and
  `run_procedural_script`. Without one, the label describes the script ("Scene
  script (3 operations)"); on the command line, it is the command and the script's
  file name.

## Merging scenes and duplicating objects

`scene.merge(source, parent=None, allow_lossy=False)` copies another scene into this
one, in one undoable step. `source` is a `Scene`, a `.aether` project or a
`.glb`/`.gltf` file, so it is also how a file is **imported into an existing scene**.
- **New ids, free names.** Copies get new ids and names free in the scene
  (`"Cube"` becomes `"Cube.001"`), and keep their hierarchy under `parent`.
- **Materials and textures come along.** They are copied, renamed the same way,
  and every reference is followed: assignments, texture bindings, image nodes of
  procedural textures.
- **Procedural objects become static meshes** of their current evaluation; their
  geometry nodes stay in the source.
- **This scene's settings stay.** The active camera, the world and the render
  settings are not taken from the source.
- On the command line: `aether3d scene merge PROJECT SOURCE [--parent OBJECT]`.

`scene.duplicate(obj, name=None, parent="same", children=True)` copies an object and
its descendants in one step.
- **New ids, free names**, as with `merge`; the copy keeps the original's
  transform, visibility, color, camera or light settings and materials.
- **Shared geometry.** Meshes are immutable and shared with the originals, and a
  procedural copy uses the same geometry node (`bake()` either object to separate
  them).
- A copied camera does not become active.

## Other undo systems

- **Edit sessions** (`obj.edit_mesh()`) keep their own undo of the operations made
  in the session. Committing is one step of the scene's history. A session opened
  before an undo that replaced its object's mesh refuses to commit
  (`STALE_SESSION`).
- **The procedural system's own `undo()`/`redo()`** remain, for compatibility:
  they restore procedural definitions only, step by step. Each call is itself a
  step of the scene's history ("Undo procedural change"), so the two stay
  consistent: `scene.history.undo()` undoes a `procedural.undo()`.

## Errors

History errors are `HistoryError` (a `ValueError`), with a stable `code`, a
`location`, a `hint` and `details`:

| Code | When |
| --- | --- |
| `NOTHING_TO_UNDO`, `NOTHING_TO_REDO` | Fewer steps than asked for (nothing changes) |
| `TRANSACTION_OPEN` | Undo, redo, `goto`, `load` or `verify` while a transaction is open |
| `TRANSACTION_CLOSED` | `tx.rollback()` after the transaction ended |
| `UNKNOWN_CHECKPOINT` | `goto` or `remove_checkpoint` with an unknown name |
| `INVALID_ARGUMENT` | A step count, limit, label or name that is not valid |
| `NO_HISTORY` | No history file where one was asked for |
| `HISTORY_MISMATCH` | The history file belongs to another version of the project |
| `INVALID_HISTORY` | The history file is malformed, or does not replay exactly |
| `UNSUPPORTED_HISTORY` | A history file of an unknown format version |
| `HISTORY_CONFLICT` | Undo would give two objects the same name: the scene was changed through private attributes (the undo is not applied) |

## Design

`aether3d/history.py` divides the scene's state into **aspects**: an object's
name, transform, visibility, appearance, camera, light, geometry (mesh and
procedural node), materials, parent, children and membership; the scene's name,
roots, active camera, world and render settings; the material library's lists of
materials and textures; each material; each texture; the procedural definitions.

1. **Operations.** Every mutating method opens an operation, and nested operations
   join the outermost one.
2. **Touch before write.** Before an operation first writes an aspect, it
   captures the aspect's state. Capturing per aspect, at the first write, means a
   write can never be missed for an aspect that was touched.
3. **Commit.** When the outermost operation ends, the new states are captured, and
   aspects that did not change are dropped. The pairs left are the step.
4. **Applying a step.** Undo writes the before-states in reverse order and redo the
   after-states. Then the indexes are rebuilt: the scene's name index, the
   material library's names, its version.
5. **Rollback.** A failing operation does the same with the states it captured.
6. **Savepoints.** Nested transactions keep their own captures, taken at their
   first write.

Every test of the test suite runs with `history.STRICT`. Every undo, redo and
rollback then checks the scene's invariants (hierarchy, indexes, references), and
a change recorded outside an operation fails.

The central test makes every kind of change (more than 60) in a scene with a bit of
everything. It undoes and redoes each one twice, and requires the same project
document (JSON) and the same Python objects each time. Other tests cover:
- a seeded chain of 150 random changes, undone and redone through every
  intermediate document;
- failed operations, transactions, savepoints and the redo branch;
- checkpoints, limits and memory;
- loading, importing and merging, persistence, tampered history files and the CLI;
- renders after undo on all four renderers.

## Limitations

- **Linear history.** A new change discards the undone steps; there is no undo tree.
- **Not thread-safe:** change a scene from one thread at a time.
- **Recording is not free.** About 3 µs per change in Python (see
  [Limits and memory](#limits-and-memory)). Changes that copy the procedural
  definitions cost what that copy costs, about the size of the definitions.
- **History files store meshes inline,** as JSON like the project, so a history of
  many large mesh edits makes a large file.
- **Merging copies procedural objects as static meshes;** their definitions are not
  merged.
