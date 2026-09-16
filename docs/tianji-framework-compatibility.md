# Tianji integration: framework compatibility audit

Audit date: 2026-09-13. Comparison baseline: original remote `main` commit
`9e313ebbe3ffea0b85de44e282ddfdc2a0f74765`, including both the existing Tianji
commits and the current working-tree changes. This is an offline code and
regression audit; no robot or physical camera was connected.

The subsequent vendor-source and layer-ownership audit is in
[`tianji-control-layering.md`](tianji-control-layering.md). It distinguishes
the SDK's nominal 1 kHz communication servicing from application target
updates, and recommends retaining the shared ManiMux execution by default.

## Viewer control

Tianji uses the existing `ViewerBridge` and `EdgeRuntime` control path. The
body adapter changes robot geometry, mounting transforms, scene defaults and
the gripper index; it does not implement another button/control protocol.

- Start/resume selects the existing RUNNING state and executor path.
- Pause selects the existing measured-position hold and executor reset path.
- Home calls the Tianji driver's home implementation, refreshes measured state,
  resets the timeline/executor/strategy, discards old requests and stays paused.
- Finish exits the episode through the existing recorder and robot cleanup.

`test_viewer_controls_reach_tianji_and_smooth_ticks_send_commands` in
`tests/unit/test_tianji_driver.py` exercises the real bridge, runtime, executor
and Tianji driver with fake transport/Marvin SDK, at both 100 and 250 Hz. It
covers pause, start, resume, home, restart, finish, SDK sends and cleanup.
This verifies software routing, not physical arm motion or a browser session.
The UMI example configs have `viewer.enabled: false`; Viewer control requires
enabling it and running the normal Viewer/session service.

## What changed in shared code

| Area | Change and compatibility boundary |
| --- | --- |
| Runtime loop, inference scheduling, timeline, session service, process decoder | Unchanged from the baseline. The UMI history strategy is an integration plugin delegating to existing strategies. |
| Viewer bridge and transport | Unchanged from the baseline. |
| YAM kinematics | Unchanged from the baseline; original pose policies still call YAM IK. |
| Robot, kinematics and Viewer registries | Add Tianji as a selectable body, with lazy backend loading. |
| Viewer scene and body interface | Add optional base orientation, scene parameters and static meshes. Defaults retain the original YAM scene. Shared gripper-close detection accepts the body's gripper index. |
| YAM driver | Extract the blocking-move Ctrl-C helper for reuse. Its executable function body is AST-identical to the original YAM implementation. |
| Configuration and Direct/Smooth executors | Add optional `per_joint` / `isotropic` limiting and `max_step_dt_s`. The default remains `per_joint` without a timestep cap. Tianji opts into isotropic limiting in its body profile. |
| Command safety | Allow an omitted acceleration limit; explicitly supplied limits continue to be checked. Tianji disables this software acceleration gate in its profile. |
| Camera server and RealSense | Add selectable camera backends and atomic frame/timestamp reads. Existing `read()` two-item results remain compatible. Camera timestamps are host receipt times, not sensor exposure timestamps. |
| Differential IK dependency | OSQP is an optional `tianji-diff-ik` extra. It is loaded when selecting the diff backend; it is not a new mandatory dependency for other robots. |

There are shared-code changes, so “Tianji only adds files and has zero impact”
would be inaccurate. The intended compatibility claim is that existing
default behavior is preserved, with the following evidence:

- Viewer, session, config and executor unit suites: **148 passed**.
- Mock runtime integration suite: **14 passed**.
- New Tianji Viewer/control-frequency cases: **2 passed**.
- Loaded the baseline's actual `limits.py`, `direct.py` and `smooth.py` in
  isolated Python modules, then compared them against the current executors:
  **3,840 arm commands exactly equal**, across 100/250 Hz, absent/finite arm
  velocity and acceleration limits, and direct/legacy-smooth/braking-smooth
  execution. Both arms included the independently controlled gripper.
- Final combined regression, including the above suites plus the complete
  Tianji driver, camera, UMI adapter and differential IK suites: **218 passed**
  in 13.02 s. These totals overlap; they are not additional independent cases.

```bash
.venv/bin/pytest -o addopts='' -q -p no:cacheprovider \
  tests/unit/test_viewer.py tests/unit/test_session_service.py \
  tests/unit/test_config.py tests/unit/test_executors.py \
  tests/unit/test_tianji_driver.py tests/unit/test_taccap_camera.py \
  tests/unit/test_umi_dp_tianji.py tests/unit/test_tianji_diff_ik.py \
  tests/integration/test_mock_run.py
```

The YAM interrupt suite could not be collected in the current environment
because its driver import requires the absent `i2rt` package. The helper's AST
comparison passed, but this does not replace YAM hardware testing. These checks
also do not certify every original model/backend or physical Viewer action.

## What the control rate means

The model returns a chunk asynchronously. Its prediction interval and the
timestamps of rows within a chunk are separate from the runtime control tick.
The normal running path is:

```text
model chunk → body decode/IK → joint timeline
                             ↓ sample at robot.control_hz
                         executor smoothing/limiting
                             ↓ once per control tick
                         driver.send_command → SDK target update
```

The pass-ball templates' `robot.control_hz: 100` gives a nominal 10 ms runtime
tick (250 Hz and 4 ms before 2026-09-15). This is the frequency
at which the runtime samples and smooths an accepted trajectory. The same loop
calls the Tianji driver's `send_command`, which calls the Marvin session's
`send_joints`, then `send_cmd` / `OnSetSend`. Thus it is also the intended SDK
target-update call frequency. It is **not** 100 model/chunk submissions per
second, nor proof of the controller's internal servo or network packet rate.
Overruns can make the actual application call rate lower.

A whole chunk's IK preparation does not inherently have to finish within one tick.
It may run ahead while the previous trajectory is being smoothed. The current
UMI adapter decodes inline on the control thread, however, so its whole-chunk
work interrupts those control ticks. The previously measured analytic decode
times (roughly 19–23 ms for H16 and 77–79 ms for H64, synthetic reachable dual-arm
paths) identify that scheduling limitation; they are not individual IK-step
times. Supporting asynchronous preparation needs to preserve observation
anchors, measured-state seeding, request identity and RTC timeline semantics.

## Why IK paths differ

Original ManiMux already has IK: `kinematics/yam.py` uses Mink, and pose-output
integrations such as OpenWAM and SAPolicy call it. Models emitting joint
positions can directly construct joint chunks instead.

CalibWrist's `sync_runtime.prepare_chunk` interpolates TCP poses at the control
rate, solves IK at every sampled point from the preceding command, and retains
the dense joint-command sequence. This can be prepared ahead of execution;
`async_runtime` also has a separate sender thread. It does not require an IK
solve from measured feedback inside every real-time sender tick.

The current ManiMux UMI adapter samples an SE(3) path while decoding and solving
each model knot, retains only its final joint configuration, then lets the
existing executor interpolate/smooth the joint timeline. In general,
interpolating joints does not reproduce interpolation in TCP space. The
difference is where IK and interpolation happen and which samples survive,
not the existence of IK.

The Tianji diff backend is selected in the UMI integration and implemented as
reusable Tianji kinematics. It does not require changing the shared runtime
loop. Selecting it does not, by itself, change the knot-based execution order
into local per-servo TCP-space execution. See
[`tianji-diff-ik.md`](tianji-diff-ik.md) for its algorithm, configuration and
validation, and [`umi_dp-tianji-validation.md`](umi_dp-tianji-validation.md)
for model-level evidence and remaining integration limits.
