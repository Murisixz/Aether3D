# MCP agent transcript (Stage 17 showcase 6)

A scripted client (deterministic code standing in for an agent) drove a real `aether3d mcp serve` process. It decided only from structured JSON results; the 3 images it received were saved, never analysed.

| # | tool | ok | ms | what came back |
| --- | --- | --- | --- | --- |
| 1 | `capabilities` | yes | 7.0 | {"aether3d": "0.17.0", "tools": 26} |
| 2 | `scene_new` | yes | 2.2 | {} |
| 3 | `execute_scene_plan` | yes | 12.5 | {"revision": {"before": 0, "after": 1}, "status": "applied"} |
| 4 | `run_script` | yes | 848.0 | {"operations": 17} |
| 5 | `inspect_animation` | yes | 1546.4 | {"Foot.L": {"contacts": 4, "sliding_max_m": 0.0036, "penetration_m": 0.0006}, "Foot.R": {"contacts": 4, "sliding_max_m": 0.0043, "penetration_m": 0.0004}} |
| 6 | `inspect_animation` | yes | 304.2 | {"path_length_m": 3.978, "displacement_m": 3.978} |
| 7 | `history` | yes | 1.8 | {"action": "checkpoint", "moved": 0, "steps": ["Robot parts", "Rig and walk"], "checkpoints": {"walk": 2}} |
| 8 | `run_script` | yes | 2.4 | {"operations": 1} |
| 9 | `inspect_animation` | yes | 1540.5 | {"Foot.L": {"contacts": 4, "sliding_max_m": 0.2048, "penetration_m": 0.0006}, "Foot.R": {"contacts": 4, "sliding_max_m": 0.2056, "penetration_m": 0.0004}} |
| 10 | `diagnose_scene` | yes | 1783.8 | {"findings": ["FOOT_SLIDING (0.20479453272050038)", "FOOT_SLIDING (0.20560070500492267)"]} |
| 11 | `propose_corrections` | yes | 4.4 | {"findings": ["FOOT_SLIDING (0.20479453272050038)", "FOOT_SLIDING (0.20560070500492267)"], "proposals": [{"id": "P1", "action": "plant_feet", "safety": "verify"}]} |
| 12 | `propose_corrections` | yes | 2.5 | {"findings": ["FOOT_SLIDING (0.20479453272050038)", "FOOT_SLIDING (0.20560070500492267)"], "proposals": [{"id": "P1", "action": "plant_feet", "safety": "verify"}]} |
| 13 | `apply_corrections` | yes | 21421.4 | {"status": "applied", "applied": ["P1"], "decision": "kept", "reason": "the findings it addressed improved and nothing regressed", "findings": [{"code": "FOOT_SLIDING", "before": 0.2056, "after": 0.0117}], "new_failures" |
| 14 | `inspect_animation` | yes | 2564.7 | {"Foot.L": {"contacts": 4, "sliding_max_m": 0.0116, "penetration_m": 0.0}, "Foot.R": {"contacts": 4, "sliding_max_m": 0.0117, "penetration_m": 0.0}} |
| 15 | `diagnose_scene` | yes | 2.0 | {"findings": ["FOOT_SLIDING (0.011560861314838194)", "FOOT_SLIDING (0.01174214016653902)"]} |
| 16 | `history` | yes | 1.9 | {"action": "goto", "moved": 2, "steps": ["Robot parts", "Rig and walk"], "checkpoints": {"walk": 2}} |
| 17 | `run_script` | yes | 4458.8 | {"operations": 2} |
| 18 | `inspect_animation` | yes | 1572.5 | {"Foot.L": {"contacts": 4, "sliding_max_m": 0.0036, "penetration_m": 0.0007}, "Foot.R": {"contacts": 5, "sliding_max_m": 0.0039, "penetration_m": 0.0004}} |
| 19 | `inspect_animation` | yes | 302.0 | {"path_length_m": 5.1714, "displacement_m": 5.1714} |
| 20 | `diagnose_scene` | yes | 1811.2 | {"findings": []} |
| 21 | `scene_info` | yes | 3.6 | {"revision": 7, "objects": 17} |
| 22 | `run_script` | refused | 1.6 | {"refused": "UNKNOWN_PROPERTY", "location": "operations[0].path", "message": "'__class__.__init__' is not an animatable property: unknown property __class__.__init__"} |
| 23 | `run_script` | refused | 1.9 | {"refused": "INVALID_VALUE", "location": "operations[1].keyframe.value", "message": "transform.position.z takes a number, got 'high'"} |
| 24 | `scene_save` | refused | 1.7 | {"refused": "PATH_OUTSIDE_WORKSPACE", "location": null, "message": "'../outside.aether' is outside the workspace"} |
| 25 | `scene_info` | yes | 11.1 | {"revision": 7, "objects": 17} |
| 26 | `render_preview` | yes | 920.6 | {"revision": 7, "width": 640, "height": 360, "saved": "preview_0.6s.png", "status": "ok", "images": "1 (not inspected)"} |
| 27 | `render_preview` | yes | 16.6 | {"revision": 7, "width": 640, "height": 360, "saved": "preview_1.8s.png", "status": "ok", "images": "1 (not inspected)"} |
| 28 | `render_preview` | yes | 15.5 | {"revision": 7, "width": 640, "height": 360, "saved": "preview_3.2s.png", "status": "ok", "images": "1 (not inspected)"} |
| 29 | `export_scene` | yes | 38.5 | {"path": "robot.glb", "bytes": 51736} |
| 30 | `scene_save` | yes | 22.1 | {"saved": "robot.aether"} |
| 31 | `history` | yes | 1.5 | {"action": "list", "moved": 0, "steps": ["Robot parts", "Rig and walk", "Walk 30% faster"], "checkpoints": {"walk": 2}} |

## Outcome

```json
{
  "server": {
    "tools": {
      "scene_new": "New scene",
      "scene_open": "Open a project",
      "scene_save": "Save the project",
      "scene_info": "Scene summary",
      "objects_list": "List objects",
      "execute_scene_plan": "Build or change the scene (scene plan)",
      "check_scene_plan": "Check a scene plan",
      "spatial_query": "Spatial query",
      "render_preview": "Render a preview",
      "render_image": "Render an image file",
      "history": "Undo, redo, checkpoints",
      "export_scene": "Export glTF or CAD data",
      "run_script": "Run a JSON script",
      "inspect_mesh": "Inspect a mesh's validity",
      "inspect_uvs": "Inspect an object's UVs",
      "inspect_texture": "Inspect a texture",
      "diagnostics": "Diagnostics",
      "capabilities": "Capabilities",
      "observe_scene": "Observe the scene (render and measure)",
      "diagnose_scene": "Diagnose the scene against expectations",
      "propose_corrections": "Propose corrections",
      "apply_corrections": "Apply corrections",
      "feedback_loop": "Run the feedback loop",
      "inspect_cad": "Inspect the CAD document",
      "import_cad": "Import a CAD file",
      "inspect_animation": "Inspect the animation"
    }
  },
  "walk": {
    "Foot.L": {
      "contacts": 4,
      "sliding_max_m": 0.0036,
      "penetration_m": 0.0006
    },
    "Foot.R": {
      "contacts": 4,
      "sliding_max_m": 0.0043,
      "penetration_m": 0.0004
    }
  },
  "walk_distance_m": 3.978,
  "rushed": {
    "Foot.L": {
      "contacts": 4,
      "sliding_max_m": 0.2048,
      "penetration_m": 0.0006
    },
    "Foot.R": {
      "contacts": 4,
      "sliding_max_m": 0.2056,
      "penetration_m": 0.0004
    }
  },
  "diagnosis": {
    "findings": [
      "FOOT_SLIDING (0.20479453272050038)",
      "FOOT_SLIDING (0.20560070500492267)"
    ]
  },
  "proposals_default_policy": {
    "findings": [
      "FOOT_SLIDING (0.20479453272050038)",
      "FOOT_SLIDING (0.20560070500492267)"
    ],
    "proposals": [
      {
        "id": "P1",
        "action": "plant_feet",
        "safety": "verify"
      }
    ]
  },
  "proposals": {
    "findings": [
      "FOOT_SLIDING (0.20479453272050038)",
      "FOOT_SLIDING (0.20560070500492267)"
    ],
    "proposals": [
      {
        "id": "P1",
        "action": "plant_feet",
        "safety": "verify"
      }
    ]
  },
  "applied": {
    "status": "applied",
    "applied": [
      "P1"
    ],
    "decision": "kept",
    "reason": "the findings it addressed improved and nothing regressed",
    "findings": [
      {
        "code": "FOOT_SLIDING",
        "before": 0.2056,
        "after": 0.0117
      }
    ],
    "new_failures": []
  },
  "planted": {
    "Foot.L": {
      "contacts": 4,
      "sliding_max_m": 0.0116,
      "penetration_m": 0.0
    },
    "Foot.R": {
      "contacts": 4,
      "sliding_max_m": 0.0117,
      "penetration_m": 0.0
    }
  },
  "diagnosis_after_planting": {
    "findings": [
      "FOOT_SLIDING (0.011560861314838194)",
      "FOOT_SLIDING (0.01174214016653902)"
    ]
  },
  "retimed": {
    "Foot.L": {
      "contacts": 4,
      "sliding_max_m": 0.0036,
      "penetration_m": 0.0007
    },
    "Foot.R": {
      "contacts": 5,
      "sliding_max_m": 0.0039,
      "penetration_m": 0.0004
    }
  },
  "retimed_distance_m": 5.1714,
  "final_diagnosis": {
    "findings": []
  },
  "accepted": true,
  "refusals": {
    "attribute_path": {
      "type": "AnimationScriptError",
      "code": "UNKNOWN_PROPERTY",
      "message": "'__class__.__init__' is not an animatable property: unknown property __class__.__init__",
      "location": "operations[0].path",
      "hint": "see aether3d.animation.paths (or the 'properties' of an animation script's capabilities) for the supported paths"
    },
    "wrong_type_after_a_valid_operation": {
      "type": "AnimationScriptError",
      "code": "INVALID_VALUE",
      "message": "transform.position.z takes a number, got 'high'",
      "location": "operations[1].keyframe.value"
    },
    "path_outside_workspace": {
      "type": "WorkspaceError",
      "code": "PATH_OUTSIDE_WORKSPACE",
      "message": "'../outside.aether' is outside the workspace",
      "hint": "use a path relative to the workspace root, without '..'"
    },
    "revision_unchanged": true
  },
  "export": {
    "path": "robot.glb",
    "bytes": 51736,
    "animations": [
      {
        "name": "Walk",
        "channels": 17,
        "clip": "Walk",
        "actions": [],
        "fitted_channels": 15,
        "max_error": {
          "translation": 2.859934511434403e-13,
          "rotation": 0.005944765510940228
        }
      },
      {
        "name": "Cam Action",
        "channels": 1,
        "clip": "Cam Action",
        "actions": [
          "d50fa8dc-91bc-401a-8269-ab578bd1fe5f"
        ],
        "fitted_channels": 0,
        "max_error": {
          "translation": 3.9968028886505635e-15
        }
      }
    ],
    "skins": null
  },
  "history": {
    "action": "list",
    "moved": [],
    "steps": [
      {
        "index": 1,
        "label": "Robot parts",
        "changes": 21
      },
      {
        "index": 2,
        "label": "Rig and walk",
        "changes": 7
      },
      {
        "index": 5,
        "label": "Walk 30% faster",
        "changes": 4
      }
    ],
    "undone": [],
    "checkpoints": {
      "walk": 2
    },
    "bytes": 83608
  },
  "calls": 31,
  "images_received": 3,
  "images_inspected_by_the_client": 0,
  "gltf_validator": {
    "ran": true,
    "errors": 0,
    "warnings": 0,
    "infos": 9,
    "version": "2.0.0-dev.3.10"
  }
}
```
