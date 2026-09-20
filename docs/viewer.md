# Viewer

Tianji and YAM Viewer use the offline `RobotModel` and named runtime groups. They
do not construct robot drivers or load policy implementations. The old YAM adapter
remains for historical collection replay.

```bash
.venv/bin/manimux-viewer --robot tianji --host 127.0.0.1 --port 8086
# Offline synthetic display; no hardware connections:
.venv/bin/manimux-viewer --robot tianji --demo --host 127.0.0.1
# Select another station's display configuration:
.venv/bin/manimux-viewer --config path/to/viewer.yaml
```

## Files

- `viewer/robots/{tianji,yam}/viewer.yaml`: model reference, group display placement,
  table, stand, view, colors, initial display pose and camera panels. Initial/demo
  poses only affect display.
- `viewer/robot_view.py`: derives display geometry and coordinate mappings from
  `RobotModel`; model FK stays in each arm's base and scene mounting is applied once.
- `viewer/dashboard.py`: YAML reading, existing page/control layout and demo.
- `viewer/camera_panel.py`, `chunk_timeline.py`, `top_overlay.py`: existing panels.
- `viewer/publisher.py`: runtime bridge and external observer publishing API.
- `viewer/communication.py`: messages, ZeroMQ transport and existing control channel.

The shared YAML reader is `manimux.cli.read_yaml`; there is no Viewer config class.
`model` is resolved relative to the Viewer YAML. The packaged default references
the same embodiment configs included in the wheel.

## Cameras and scene

`cameras` is an ordered list. `source` is a runtime camera key, `label` is its display
name, and `slot` selects the existing top/left/right positions. Additional slots
appear in the existing additional-camera folder. Omit a view if the station does
not have it. Tianji defaults to two wrist cameras without an empty top preview.

```yaml
camera_mode: policy
cameras:
  - {source: agent_view, label: External, slot: top}
  - {source: left_wrist, label: wrist, slot: left}
  - {source: right_wrist, label: wrist, slot: right}
```

`camera_mode: policy` follows the runtime's policy camera map; `manual` selects the
configured sources without changing model inputs. `camera_aliases` identifies
spatial roles for sources. Cached image/state timestamps retain their original values.

Table dimensions and position live under `scene.boxes`. Initial camera/grid settings
live under `scene.view`; static stand meshes live under `scene.meshes`. Mesh URDFs
resolve relative to this Viewer YAML, or use `package://package.name/resource`.
Arm/tool geometry and joint layouts still come from `RobotModel`.

Each group's `viewer_display_frame` places its model in the scene (metres and xyz
Euler angles in radians). The Viewer supplies its own scene origin; there is no
`root_frame` in the robot configuration and no shared-base transform in FK/IK:

```yaml
groups:
  left_arm:
    viewer_display_frame:
      xyz: [0.0, 0.32, 0.0]
      rpy: [0.0, 0.0, 0.0]
```

Omitting this display field places a group at the scene origin. Only Viewer reads
it; changing it cannot change robot commands or local FK/IK results. Stand meshes
accept the same display field. The robot's end-effector `mount` remains the physical
flange-to-tool transform and continues to participate in TCP computation.

## Runtime interface

The new wire format uses `groups` in both state and plan messages. For Tianji these
are `left_arm` and `right_arm`, each containing seven arm coordinates and `gripper`.
There is no concatenation, dimension-based left/right inference or old TCP adapter.
Predictions and measured states remain separate messages. State and camera metadata
preserve original timestamps and sequence numbers. Update publisher and Viewer together.

The experiment setting is `viewer.robot: tianji`, selecting the body display configuration.
Experiments select the display model using `viewer.robot`; all bundled recipes use this field.

Tianji recovery/home remain capabilities of Session/runtime; this Viewer refactor
adds no hardware recovery or motion implementation. Existing UI controls remain.

The old MolmoAct observer launcher and `manimux-molmoact-yam` entry point have been
retired with the legacy YAM hardware stack. YAM uses the common runtime and Viewer;
see [the YAM component migration](yam-integrated-component.md). This hardware
migration does not validate or replace a learned-model deployment.
