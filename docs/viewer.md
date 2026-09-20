# Viewer

Tianji Viewer uses the offline `RobotModel` and named runtime groups. It does not
construct robot drivers or load policy implementations. The current migration is
limited to Tianji; YAM's old adapter remains pending its own RobotModel integration.
The new YAML entry currently supports Tianji only.

```bash
.venv/bin/manimux-viewer --robot tianji --host 127.0.0.1 --port 8086
# Offline synthetic display; no hardware connections:
.venv/bin/manimux-viewer --robot tianji --demo --host 127.0.0.1
# Select another station's display configuration:
.venv/bin/manimux-viewer --config path/to/viewer.yaml
```

## Files

- `viewer/robots/tianji/viewer.yaml`: model reference, table, view, group colors,
  initial pose selection and camera panels. Initial/demo poses only affect display.
- `viewer/robot_view.py`: derives display geometry and coordinate mappings from
  `RobotModel`; model FK stays in each arm's base and scene mounting is applied once.
- `viewer/dashboard.py`: YAML reading, existing page/control layout and demo.
- `viewer/camera_panel.py`, `chunk_timeline.py`, `top_overlay.py`: existing panels.
- `viewer/publisher.py`: runtime bridge and external observer publishing API.
- `viewer/communication.py`: messages, ZeroMQ transport and existing control channel.

The shared YAML reader is `manimux.cli.read_yaml`; there is no Viewer config class.
`model` is resolved relative to the Viewer YAML. The packaged default references
the same embodiment configs included in the wheel.

Tianji uses `initial_pose: home`, reading arm targets from `home.joints_deg` in
`configs/embodiment/robot/tianji_taccap.yaml`. These are the active joint values
from `teleop/data/dp_start_20260910.yaml` (A = left, B = right), stored in degrees
and exposed as radians through `RobotModel.home_joints`. No external teleop checkout
is needed. Edit this assembly config to change the shared Home/initial target.
The recorded pose contains no gripper target; each Viewer's `initial_end_effector`
sets only its displayed aperture. Explicit group `initial` vectors remain available
when `initial_pose` is omitted, but cannot override a selected Home pose.

The assembly config supplies the shared target. `TianjiTaccapRobot.home()` remains
unsupported, while the idle `manimux serve` recovery path executes Viewer **Return Home**
as a 6 deg/s cosine joint trajectory. It preserves measured gripper apertures, confirms
arrival within 0.5 degrees, then disables and releases the robot. Loading or displaying
Home alone never moves hardware.

Viewer **Start drag** is also owned by the idle session rather than the robot embodiment.
It reproduces teleop's joint-drag sequence through the Marvin SDK: register the measured
per-arm UMI tool parameters, enter torque/joint-impedance mode with K=1 and D=0.3, enable
joint drag, and track measured joints at 250 Hz with a 15 deg/s reference limit. **Exit
drag**, loss of the Viewer lease, or session shutdown first leaves drag space, confirms
servo-off, and then releases the controller connection. A/B/AB share one connection.

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
Temporal `_prev` inputs are listed in the bottom **Images** diagnostics but reuse their
physical camera preview instead of creating duplicate black views. The Top reference
overlay is created only when the Viewer configuration actually declares a `top` slot.

Table dimensions and position live under `scene.boxes`. Initial camera/grid settings
live under `scene.view`. Arm, tool and stand geometry remain in the referenced
`RobotModel`; do not copy installation transforms or joint layouts into display code.

## Runtime interface

The new wire format uses `groups` in both state and plan messages. For Tianji these
are `left_arm` and `right_arm`, each containing seven arm coordinates and `gripper`.
There is no concatenation, dimension-based left/right inference or old TCP adapter.
Predictions and measured states remain separate messages. State and camera metadata
preserve original timestamps and sequence numbers. Update publisher and Viewer together.

The experiment setting is `viewer.robot: tianji-taccap`, matching the model name.
The config reader still normalizes the old `robot_adapter` spelling for existing
experiment files; it does not load a Viewer adapter.

Tianji recovery, drag and Home remain capabilities of Session/runtime; the Viewer only
sends recovery requests and keeps the drag lease alive. The `tianji-taccap` model uses
**Finish rollout**, which saves and closes without homing; explicit **Return Home** and
**Start drag** become available while the executing runtime service is idle.

The old MolmoAct observer launcher is now
`manimux.integrations.molmoact_yam.viewer_launch`; `manimux-molmoact-yam` retains its
entry point name. Its old vector conversion stays at that experiment boundary.
