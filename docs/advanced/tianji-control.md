# Tianji control responsibilities and execution choices

2026-09-13. This audit follows the user's clarification: first distinguish
hardware requirements from runtime choices, then decide whether to use the
existing ManiMux execution or add a general interpolation option. It does not
introduce another Tianji runtime or change execution defaults.

## Vendor evidence

The vendor repository is `cynthia-you/TJ_FX_ROBOT_CONTRL_SDK`. The installed
source checkout is at `a285e6cb1df92adf9da1827152f152f73582a511`; its
`SDK_PYTHON/libMarvinSDK.so` exactly matches ManiMux's vendored binary, SHA256
`47ab2ab6e35cf899eae3978dfb50ee219e05b10f72aedd6bc9ba6a338a8b4011`.
The public `master` head was also checked at
`02440e886fb59095711eb9ec6dcbedd8be08922a`. Its README retains the alternatives
below. The public default `main` branch is not the SDK implementation branch.

The vendor separates its control SDK from its kinematics/planning SDK. Its
[README](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/blob/02440e886fb59095711eb9ec6dcbedd8be08922a/README.md)
describes several supported paths:

1. Solve individual Cartesian targets with IK, then send joint targets.
2. Generate a Cartesian path on the host, solve its points with IK, and send
   the resulting joint targets at the selected rate.
3. Use SDK joint/Cartesian planners and the corresponding buffered execution
   interfaces, including `setPln_joint` and `setPln_Cart`.
4. Use the separate PVT trajectory reproduction mode.

The [position-following example](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/blob/a285e6cb1df92adf9da1827152f152f73582a511/DEMO_PYTHON/showcase_position.py)
sets velocity/acceleration percentages, enters position mode, submits a joint
target and waits for motion. It does not require a host TCP interpolator or
OSQP diff IK. Controller-side joint following therefore exists independently
of a model chunk's host-side interpolation. The exact firmware interpolation
law and inner servo frequency were not established by this audit.

Current ManiMux uses **position mode plus streaming joint targets**:
`MarvinSession.prepare_position_mode` selects state 1, and `send_joints` calls
`clear_set → set_joint_cmd_pose(A/B) → send_cmd`. It does not invoke the vendor
buffered planner or PVT path. Changing to those interfaces would also change
trajectory ownership, cancellation and completion semantics; it is not a
transparent replacement for the current driver call.

## Distinguish the clocks

The [control API documentation](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/blob/02440e886fb59095711eb9ec6dcbedd8be08922a/python_doc_contrl.md)
describes a 1 kHz communication/buffer cycle. The installed source's
[`contrlSDK100343/Robot.cpp`](https://github.com/cynthia-you/TJ_FX_ROBOT_CONTRL_SDK/blob/a285e6cb1df92adf9da1827152f152f73582a511/contrlSDK100343/Robot.cpp)
creates a 1 ms timer, calls `DoRecv/DoSend` from `OnTimerTick`, and marks pending
data in `OnSetSend`. `DoSend` sends a pending command and clears the flag.
Read-only disassembly of the actual vendored binary also shows the pending
flag and conditional-send design. This is not an automatic interpolator that
turns a model's sparse trajectory into 1,000 fresh target positions per second.

The independent quantities are:

- model inference/chunk arrival timing;
- model action-row interval, determined by the checkpoint;
- ManiMux control sampling and target-update rate (`robot.control_hz`);
- SDK communication servicing;
- controller firmware's motion following and inner servo loops.

Thus the control rate is a **deployment/runtime choice**. The pass-ball templates
use 100 Hz (250 Hz before 2026-09-15); the local teleop stack, the driver's home
move and `set-state drag` still use 250 Hz. It is not a vendor-mandated
interpolation rate. The vendor README
mentions up to 200 Hz in one streamed-planning section and commands below 1 kHz
in its general FAQ, while other modes have their own rates. These statements
must not be combined into a claim that every interface is validated at 100 or
250 Hz.
No physical frequency or tracking test was performed here.

## Responsibility boundary

| Behavior | Owner in ManiMux |
| --- | --- |
| Connection, mode transition, SDK encoding, feedback, stop and hardware errors | Tianji RobotBase / vendor SDK |
| Controller velocity/acceleration percentages and control-mode parameters | Body profile and driver / controller |
| DH geometry, TCP transform, joint limits, J6/J7 constraints, FK/IK solver | Body kinematics capability |
| Choice to interpolate joint values or TCP poses, and when/how often to call IK | General trajectory preparation / execution configuration |
| Analytic vs differential IK | Selectable kinematics backend, independent of interpolation space |
| Chunk timing, expiry, splicing, RTC conditions, preparation scheduling | Shared runtime |
| Joint smoothing, software limiting and command validation | Shared executor / SafetyGuard, using configured limits |
| Gripper protocol, aperture conversion and impedance control | Gripper driver |
| Predicted-aperture thresholds and close/reopen latch | Configurable action/execution policy; not a Tianji hardware requirement |

An algorithm being shipped in a vendor SDK does not make it mandatory hardware
behavior. Conversely, runtime selection of a path must respect the capabilities
and timing contract of the selected hardware interface.

## Recommended direction

Keep the current **joint-knot → shared timeline → executor → guard → driver**
path as the default. Preserve the original model's input/output semantics and
the body's calibrated geometry/limits. Matching every CalibWrist execution
detail is a separate reproduction objective, not a prerequisite for Tianji
body support.

UMI differential IK can opt into `execute_diff_ik_substeps`, which passes the
joint samples already computed along each Cartesian segment to the same runtime
and driver. The option is adapter-local, defaults off and is rejected for the
analytic backend; it is not available to policies that already predict joints.
It remains separate from `motion_limits.arm.mode`, which selects velocity
shaping, and does not change the model action horizon or interval.

Existing `direct/smooth/mpc` choices do not select interpolation space. With the
option disabled, the timeline retains its prior joint-knot interpolation. With
it enabled, only the executor-facing runtime trajectory uses the dense diff-IK
samples; Viewer, RTC and other source-horizon consumers keep the model knots.

A separate runtime trajectory carries its own start timestamp, interval and
samples. Source indices, chunk caps and RTC masks continue to use the original
`ActionChunk` fields. In particular, the checkpoint `action_dt_s` remains 30 Hz;
the dense execution interval does not change model semantics.

Preparation should use the existing process decoder. UMI currently depends on
parent-side request anchors; it must first support immutable decode context.
The runtime's current strategy-name checks must become explicit capability
checks, with measured history, RTC deadlines, seed time and takeover continuity
tested together. Merely removing the checks would not establish compatibility.
This scheduling work is useful even while retaining joint-knot interpolation.

No production source/configuration changes, hardware operations, commits or
pushes were made for this audit. Previous test counts describe the earlier
implementation, not tests of an unimplemented Cartesian mode.
