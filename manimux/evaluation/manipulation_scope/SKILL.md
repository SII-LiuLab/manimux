---
name: manimux-manipulation-scope
description: Draw recorded gripper close/open locations from demonstrations or finalized inference rollouts onto a calibrated camera frame. Use for training-layout checks, inference coverage, camera calibration, or scope figures in a task report. Always distinguish the data source.
---

# ManiMux Manipulation Scope

## Robot authorization — required for every task

Without explicit user permission, do not start or restart a robot, connect its
controller, enable actuators, or cause robot motion. This includes Prepare, Warmup
on connected hardware, Start, Home, gripper calibration, replay and recovery/release.
Invoking this skill or requesting integration, configuration, testing, evaluation or
video capture does not grant that permission. Reuse explicit authorization already
given for the same robot and operation; do not ask for it repeatedly.

Complete configuration, code and disconnected validation first. If permission is
missing, show the concrete robot, resolved configuration and intended operations,
then ask before the first hardware-affecting action. Configuration review is not
robot-start authorization. Leave physical controls with the operator by default.
`execute: false`, “dry run” and “warmup” are not evidence of isolation: constructors,
connect and cleanup can move hardware. Use disconnected models/fake devices offline.
A separately authorized GUI/model/camera service launch may proceed only after
verifying it will not connect or actuate the robot. Preserve existing service and
hardware ownership; do not stop, restart or release an active robot to test a change.

Work from the ManiMux checkout. Code: this package, `manimux/evaluation/manipulation_scope/`
(`python -m manimux.evaluation.manipulation_scope` with subcommands `points`,
`calibrate`, `silhouette`, `render`). The experiments skill calls this workflow. Profiles live in
`data/evaluation_tasks/<task>/manipulation_scope/` (local data, not in Git).
Label every path as local or remote when reporting.

## Scope and reusable inputs

This is an offline recorded-data workflow. Read the source cohort's robot/tool model,
coordinate conventions and existing camera calibration before requesting new facts.
Reuse a calibration only when its camera pose and frame convention match. If geometry
exists but the model API does not expose it, route that gap to development rather than
asking the user to remeasure. Never connect a robot to obtain a figure's missing inputs
without explicit authorization. If model integration is required, follow the shared
[protocol and exception gate](../../../.agents/skills/manimux-development/references/policies.md#mandatory-shared-contract-for-every-model-and-backend).

The teleop/ABC readers and YAM examples below describe current supported data formats,
not universal conventions for other embodiments or datasets. Verify the event threshold,
height filters, FK/control point and camera/frame mapping against the actual source;
unsupported formats need an explicit offline reader, not silent reinterpretation.

## What the figure is

Each dot is a recorded gripper event. Demonstrations describe demonstration coverage;
inference rollouts describe the policy's observed coverage. A threshold crossing does
not establish that an object was successfully grasped or released.

1. **Events**: per arm, the gripper (1 = open) crossing 0.6 closing is a grasp,
   opening is a release. Teleop keeps all events; ABC keeps grasps below 12 cm and
   releases above it (ABC throws into the bin).
2. **3D**: FK of the joint row gives the `grasp_site` in that arm's base frame.
   For a dataset carrying `<side>-ee_pos.npy`, compare it with FK first;
   establish the recorded control point and frame instead of assuming equivalence.
3. **Pixels**: a calibrated pinhole camera (left base frame; right base at
   `right_base_offset_xy`, yawed by `right_base_yaw`) projects the points.
4. **Draw**: per-arm grasp dots, central-80% hull, white median cross; releases per
   arm (teleop, at their height) or one table-plane bin zone (ABC); thin lines are
   teleop grasp-site paths; profile `regions` add dashed pixel polygons.

## Inputs

- **Dataset**: teleop task dir (`<episode>/<side>-joint_pos.npy`,
  `<side>-gripper_pos.npy`, `top-images-rgb.mp4`, `write_complete.flag`) or ABC
  preview (`<split>/<episode>/states_actions.bin`). Robot config
  `manimux/configs/embodiment/robot/yam_dual.yaml`.
- **Image to draw on**: an image, an `.mp4` (first frame) or a ManiMux episode dir
  (`videos/<camera_name>.mp4`). It must come from the calibrated camera pose.
- **Inference session**: `rollout-*/result.json` plus `data.zarr/ticks/state/<arm>`
  and `ticks/monotonic_ns`. Only finalized attempts are loaded, including failures.
  Command and scheduled targets are not substituted for measured state. The points
  artifact records `source_kind`, threshold, and every event's episode/tick/time/XYZ.
  For `silhouette`, pass `--video d405_front.mp4`; frame timestamps in
  `videos/index.json` are matched to the nearest measured tick (at most 50 ms).
  Frames outside the recorded state interval are excluded. Never pair frame number
  with control tick number. Use one explicit session per calibration/source cohort.

## Procedure

1. **Find camera poses.** Tile first frames across the dataset's dates. Do not
   assume the camera stayed fixed between collection or inference sessions.
   Use one calibration per pose; if a
   dataset mixes poses, point `--dataset` at a directory of symlinks to one pose's
   episodes. Collection may be writing into the same task dir; episodes without
   `write_complete.flag` are skipped.
2. **Points.** `python -m manimux.evaluation.manipulation_scope points --dataset D
   --robot-config R --out <profile dir>/teleop_points.json` (seconds).
3. **Camera.**
   - Teleop video: `silhouette --dataset D --robot-config R --camera
     <profile dir>/top_camera.json`. It matches projected FK mesh points of both
     arms to the motion foreground (frame minus episode median) using up to 9
     evenly spaced episodes, and writes `top_camera_reference.png`. About 15 min:
     run it in the background. Inspect residuals and image overlays; a historical
     residual from another dataset is not an acceptance criterion.
   - Rollouts or stills without usable motion: `calibrate --camera camera.json`
     from `correspondences` (`arm`, `xyz` FK grasp site, `px` clicked pixel).
     Clicked fingertips are unreliable in oblique views; prefer silhouette.
   - Focal length is weakly determined when the events are all near the table
     plane; that does not move table-height points, so do not chase it.
4. **Profile** `manipulation_scope.yaml` (`schema: manipulation-scope-v1`):
   `camera_name`, `camera`, `points`, `points_offset_m` (derive from the source's
   geometry and calibrated frame; do not copy another station's plate offset),
   `align_mask_rows`, `labels`, optional
   `regions`. Labels: `left_arm`, `right_arm`, and either `release` (one bin zone)
   or `left_release` + `right_release`; `{n}` is replaced by the event count.
   Write all figure text (labels, region labels, `--title`) in English and keep each
   legend line within the image width.
   For inference, label legends `Left close`, `Right close`, `Left open`, `Right open`;
   set `title` and `caption` to identify the inference cohort and state that events
   are not verified grasps. Do not describe inference coverage as training support.
   `align_mask_rows`: `[]` for frames from the calibration session; otherwise
   static rows only (robot bases, base plate) such as `[[300, 480]]`. A bucket in
   the band drags the ECC fit; real camera moves are refused, not drawn.
5. **Render** `render --profile P --input A [B ...] --title ... --out out.jpg`
   (several inputs are placed side by side).
6. **Verify before delivering.** At several event frames, project the event's
   grasp site onto the video frame and confirm it lands on the gripper (within
   ~10 px). Also look at the full figure: grasps on the objects, releases at the
   target.

## Task reports

In `data/evaluation_tasks/<task>/task.yaml` set
`manipulation_scope: manipulation_scope/manipulation_scope.yaml`; then
`python -m manimux.evaluation --task-dir data/evaluation_tasks/<task>` renders each
finalized attempt's first frame with the scope. Missing videos or a moved camera
become per-attempt errors with the reason.

## Reuse local profiles

Inspect `data/evaluation_tasks/<task>/manipulation_scope/` for existing profiles and
verify source cohort, robot geometry, camera pose and event conventions before reuse.
A profile name or a previous user's task does not establish compatibility.
