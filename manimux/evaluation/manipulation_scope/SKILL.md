---
name: manimux-manipulation-scope
description: Draw where demonstrations grasped and released objects (teleop recordings or ABC training data) onto a camera frame, to check a layout or a rollout against the training distribution. Use when asked to visualize teleop/training data on the image, to check whether a table layout is in distribution, to calibrate a camera to the arms, or to add the scope figure to a task report.
---

# ManiMux Manipulation Scope

Work from the ManiMux checkout. Code: this package, `manimux/evaluation/manipulation_scope/`
(`python -m manimux.evaluation.manipulation_scope` with subcommands `points`,
`calibrate`, `silhouette`, `render`). The experiments skill calls this workflow. Profiles live in
`data/evaluation_tasks/<task>/manipulation_scope/` (local data, not in Git).
Label every path as local or remote when reporting.

## What the figure is

Each dot is a demonstration event, not a policy trajectory:

1. **Events**: per arm, the gripper (1 = open) crossing 0.6 closing is a grasp,
   opening is a release. Teleop keeps all events; ABC keeps grasps below 12 cm and
   releases above it (ABC throws into the bin).
2. **3D**: FK of the joint row gives the `grasp_site` in that arm's base frame.
   The recorded teleop `<side>-ee_pos.npy` equals our FK exactly; check this first
   on a new dataset.
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

## Procedure

1. **Find camera poses.** Tile first frames across the dataset's dates. The top
   camera moves between days (2026-10-10 and 10-11 conveyor differ; the 10-07
   ABC-DiT `d405_front` differs from all teleop). One calibration per pose; if a
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
     run it in the background. Expect a chamfer of ~15 px.
   - Rollouts or stills without usable motion: `calibrate --camera camera.json`
     from `correspondences` (`arm`, `xyz` FK grasp site, `px` clicked pixel).
     Clicked fingertips are unreliable in oblique views; prefer silhouette.
   - Focal length is weakly determined when the events are all near the table
     plane; that does not move table-height points, so do not chase it.
4. **Profile** `manipulation_scope.yaml` (`schema: manipulation-scope-v1`):
   `camera_name`, `camera`, `points`, `points_offset_m` (ABC on our 2.2 cm plate:
   `[0, 0, -0.022]`; teleop: zeros), `align_mask_rows`, `labels`, optional
   `regions`. Labels: `left_arm`, `right_arm`, and either `release` (one bin zone)
   or `left_release` + `right_release`; `{n}` is replaced by the event count.
   Write all figure text (labels, region labels, `--title`) in English and keep each
   legend line within the image width.
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

## Existing profiles (local)

- `put_bottles`: ABC real put/throw-bottles points on the 10-07 `d405_front`
  pose, with the ABC-DiT start-sensitive yellow box.
- `pick_the_moving_cube`: 2026-10-10 conveyor teleop (9 episodes).
- `put_the_coin_into_the_bin`: 2026-09-27 coin teleop (50 episodes).
