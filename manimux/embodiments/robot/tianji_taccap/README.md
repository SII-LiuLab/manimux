# Tianji–TacCap robot

The robot name is `tianji-taccap`, its Python class is `TianjiTaccapRobot`, and experiments select
`robot.type: tianji_taccap`. The arm component is still `TianjiArm`.

## Call chain

```text
Experiment YAML → config.load_config()
          → EdgeRuntime
          → embodiments.robot.build_robot()
          → TianjiTaccapRobot.from_config()
                → RobotBase manages arms, end effectors and sensors
```

The registry selects the robot class using `robot.type`; `robot.config` resolves relative to the experiment YAML.
`RobotBase` owns `RobotModel` and reuses its `kinematics`; subclasses do not provide independent duplicate kinematics.
`RobotModel` stores component models and mounting data together, enabling offline action decoding without control connections.
The old `robots/` import directory was removed. Factories live in `embodiments.robot`; Tianji and YAM use component assembly.

## File responsibilities

| File | Responsibility |
| --- | --- |
| `manimux/configs/embodiment/robot/tianji_taccap.yaml` | Components, mounts, control groups and visual assets |
| `manimux/configs/embodiment/robot/tianji_control.yaml + manimux/configs/executor/tianji_smooth_control_profile.yaml` | Shared motion limits, command guards and execution smoothing |
| `.local/tianji_taccap.yaml` | Private device, service and path bindings |
| `manimux/configs/embodiment/arm/tianji_left.yaml`, `tianji_right.yaml` | Left/right arm models and control parameters |
| `manimux/configs/embodiment/end_effector/taccap.yaml` | End-effector parameters |
| `manimux/configs/embodiment/sensor/taccap.yaml` | Camera capture parameters |
| `embodiments/robot/base.py` | Shared lifecycle, component coordination and offline assembly data |
| `embodiments/robot/tianji_taccap/robot.py` | Shared Tianji controller and component construction |
| `embodiments/arm/tianji/arm.py` | Official control SDK connections, feedback and batched commands |
| `embodiments/arm/tianji/kinematics.py` | Existing Tianji numerics and official flange interface |
| `kinematics/composed.py` | Shared arm + end-effector TCP transform composition |
| `policy_adapter/umi_dp/tianji.py`, `policy_adapter/umi_dp/history.py` | UMI observations/actions, history windows, timing semantics and FK/IK calls |

SDKs and assets live with their owning components. Legacy `kinematics/tianji.py` wraps the old parameters and TCP API;
the arm directory contains the single numerical solver implementation. `TianjiSDKKinematics` and the legacy entry point share
the original `TianjiArmKinematics` solving rules. Arms do not load tool configuration or store TCP offsets;
shared `ComposedManipulatorKinematics` assembles tools for the new robot.

## Frames and solver behavior

Each group is `[joint_1, ..., joint_7, gripper]`: joints in rad, gripper 0 closed and 1 open.
**FK/IK TCP poses always use the corresponding arm base frame, with translation in meters.**

```text
T_arm_tcp = FK_official(q_arm) × T_flange_tool × T_tool_tcp
T_arm_flange_target = T_arm_tcp_target × inverse(T_flange_tool × T_tool_tcp)
```

Embodiment configuration no longer declares `root_frame`, arm-base display transforms or an additional body frame.
Display positions and orientations live in `robogui/robots/tianji/robogui.yaml` under
`groups.<name>.robogui_display_frame`; static mounts live in `scene.meshes` in the same file.
Only RoboGUI reads these settings; changing them does not affect FK/IK.
The end-effector `mount` and its own TCP offset participate in control calculations exactly once.

IK retains official `ik + ik_nsp`, FK substitution checks, joint limits and margins, measured J6/J7 interference constraints and
seed-relative branch-jump handling. The original right-arm J6 override lives in the arm configuration.
Assembly adds no second, stricter TCP acceptance threshold. Failure does not return executable substitute joint targets.

The UMI service already restores predictions relative to observed TCP poses to absolute targets in each arm base frame.
The ManiMux adapter preserves this convention without adding a common-frame conversion. In-process decoding reuses the robot model;
an independent action-decoding subprocess loads the offline model from the same configuration without hardware connections.
Optional differential IK retains the original QP; assembly removes tool offsets before sending flange targets to that solver.

## Offline construction and hardware lifecycle

```python
from manimux.embodiments.robot.tianji_taccap import TianjiTaccapRobot

robot = TianjiTaccapRobot.from_config("manimux/configs/embodiment/robot/tianji_taccap.yaml")
# robot.fk(configuration) / robot.ik(targets, seed, fixed_coordinates=...)
# Arms, grippers and cameras are not connected; complete hardware bindings are not required yet.
```

`from_config()` defaults to `execute=False` and `end_effector_control=False`, preserving the original read-only defaults.
Direct Python `ip/arms` construction retains its prior API defaults. Device parameters such as IPs, serial ports and gains are used in
`connect()` / `start()` rather than scanning the full configuration for all missing fields beforehand.

`connect()` reads arm and gripper feedback; the component interface requires arms to be disabled and enables them at measured positions
on the first execution command. Both arms share one target batch; grippers are commanded separately afterward. Cross-device execution is not atomic.
Configured joint limits, tracking-error, stale-feedback and controller-fault checks remain in place.
`stop()` requests owned components to stop; failed `close()` calls can be retried.

Marvin submits command batches asynchronously. Its `clear_set()` can return false
while the previous batch is queued or being sent; this is a busy send buffer, not
the robot's fault-clear operation. The controller waits up to 20 ms (or the shorter
configured readiness timeout), polling every 1 ms, before writing any new batch
fields. The same wait applies to position commands, disable commands and drag
transitions. A buffer that stays busy raises a timeout; `send_cmd()` failures are
still surfaced without replaying a possibly submitted command. Feedback, fault
and state-transition checks remain required.

Recovery lives on the assembly, not in the session. `clear_errors()` clears latched
controller faults over a short SDK session before `connect()`. `home()` moves the
connected arms to `home.joints_deg` on a 9 deg/s cosine profile, settles within 0.5
degrees and then fully opens the grippers when end-effector control is enabled.
`drag(groups, stop)` opens only the arm controller, enters Marvin joint-space drag with
each arm's `drag_tool` load from the assembly YAML, and disables the arms on exit.
With execution enabled and a bound controller IP, `recovery_actions` offers
`clear_error`, `home`, `drag`, `release` and `release_5s`; `drag_selections` maps A, B and AB
to the left, right and both arm groups. The idle recovery entry points
`recover_home(stop)` and `recover_drag()` clear latched faults first, because `connect()`
rejects faulted arms. Idle Home restores grippers within the assembly, without changing
the caller's configuration. The stop event cancels trajectory, settling and gripper waits;
the owned components are stopped and closed before recovery returns. Task experiments
select `tianji_taccap_pass_ball.yaml` or `tianji_taccap_pack_plate.yaml`; the shared
`tianji_taccap.yaml` preserves the original pass-ball Home.

RoboGUI's **Release** button opens a choice of **Continuous release** or **Release 5s**,
with an A / B / AB arm selector. Both call the assembly's `recover_release()` through
`hand_guiding_actions`. It clears selected-arm faults, sends the selected arms' calibrated
UMI `drag_tool` parameters, confirms their SDK readback, and enters native cooperative
release (`state=4`). This is separate from joint-space drag (`state=3`) and does not
force the brakes open. Cameras and grippers remain closed.

Continuous release lasts until **Exit release**, recovery lease loss or service shutdown.
The five-second duration starts after confirmed mode entry and can also be stopped early.
The assembly monitors controller feedback and stops on faults, stale feedback, an
unexpected mode change, or a joint entering the five-degree margin of its configured
limits. Every exit closes the owned arm controller, disabling enabled arms first;
cleanup failures are reported in RoboGUI. Mode selection, duration and cleanup live in
the Tianji assembly/controller; the session only dispatches cancellable handlers. During
a rollout, **Stop rollout & release** finishes it without homing before starting recovery.

Camera startup is independent of `connect()`. New UMI experiments still subscribe to the existing camera server without automatically calling
`robot.start_sensors()`. Subscription maps `left_wrist/right_wrist` to
`left_wrist_camera/right_wrist_camera`, preserving RGB, timestamps and frame sequences.
Null optical extrinsics mean unknown. TacCap TCP remains the fixed CAD approximation, without new calibration or opening compensation.

## Experiment entry point and validation

The experiment entry point is `manimux/configs/experiments/pass_ball/umi_dp/tianji_taccap_umi_dp.yaml`.
Select the station with `--local`; policy configuration lives in `manimux/configs/policy/umi_dp/`.
The existing XPolicyLab UMI_DP model and `xpolicylab_ws` remain in use. Bind checkpoint identity with the original launcher
before running. See `.agents/skills/manimux-inference/references/deployment/umi-dp-tianji-taccap.md`.

Tests cover registry-to-adapter wiring, numerical regression, display/control frame separation, subprocess decoding parity,
existing timing/motion constraints and read-only/execution dispatch with fake SDKs. No real hardware or model service was started.
RoboGUI loads visuals and independent display transforms from the same `RobotModel`.
Clear error, Return Home drag and release recovery are implemented by `TianjiTaccapRobot`; the
idle RoboGUI session only schedules them.
