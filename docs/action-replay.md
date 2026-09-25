# Offline action replay

The Viewer can play a sequence of absolute robot joint configurations without a
runtime, policy server, camera service or robot connection. It displays targets,
not measured execution. Launch it on a separate port when a live Viewer is running.

Save one `(steps, coordinates)` NumPy array per robot group:

```python
import numpy as np

# Existing YAM trajectories: each array is (T, 7).
# Columns: six joint angles in radians, then normalized gripper opening (0 closed, 1 open).
np.savez("actions.npz", left_arm=left_actions, right_arm=right_actions)
```

Group names and dimensions must match the selected robot model. Arrays must have
finite values and the same nonzero number of steps. A single group is allowed;
other groups stay at their configured initial display pose. Raw normalized model
outputs, delta actions and end-effector poses must first be decoded to absolute
joint configurations with the matching policy adapter.

From the repository root:

```bash
envs/yam/.venv/bin/python -m manimux.viewer.dashboard \
  --robot yam --replay-actions actions.npz --action-dt-s 0.03333333333333333 \
  --host 127.0.0.1 --port 8087
```

Open the printed URL. Playback starts paused. Use **Play**, **Frame** and **Speed**
to play, pause, seek or change speed. Playback stops at the last frame; pressing Play
again restarts it. Samples are displayed at the supplied action interval, without
executor smoothing or changes to the input trajectory. The browser display can skip
frames when playback is faster than rendering.

The Python interface accepts the same named arrays directly:

```python
from manimux.viewer.action_replay import ActionReplayViewer
from manimux.viewer.dashboard import load_robot_view, load_viewer_config

robot = load_robot_view(load_viewer_config(robot="yam"))
viewer = ActionReplayViewer(
    {"left_arm": left_actions, "right_arm": right_actions},
    robot, action_dt_s=1 / 30, port=8087,
)
# Call viewer.tick() regularly from your event loop; call viewer.close() on exit.
```

Without `--replay-actions`, the existing live Viewer startup and runtime control
protocol are unchanged. Replay does not construct `PolicyViewer`, bind its control
socket, connect a runtime subscriber, write rollout data, or send robot commands.
Only the robot's offline model dependencies are required. The NPZ playback path
has no video or PyAV requirement.
