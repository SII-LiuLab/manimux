# Xiaomi Robotics 1 / Tianji–TacCap pass ball

This entry loads a local step-50000 DeepSpeed model file through
`third_party/XPolicyLab/policy/Xiaomi_Robotics_1` and serves ManiMux through the shared
`xpolicylab_ws` worker. The model receives only the two real wrist RGB streams. The third
ego view was pure black during training, so XPolicyLab generates an all-zero `uint8 RGB`
image during preprocessing; no fake camera is registered and no live image is reused.

## Fixed contract

- Checkpoint: `mp_rank_00_model_states.pt`; the eight optimizer states are not used for inference.
- Normalization: `training_metadata/normalize.json` in the same directory, with
  `mean/std=(30,60)` and `q01/q99=(1,60)`.
- Observation: seven joints and one `[0,1]` TacCap opening per Tianji arm; two wrist RGB
  streams; a black third view at the left-wrist resolution, then XR-1's common resize.
- Action: the checkpoint natively outputs 30×60 anchor-relative EE deltas. Every row is
  relative to the TCP pose at request time, not accumulated row by row. As with UMI-DP,
  XPolicyLab owns the relative encoding: the adapter sends the observation-time
  `left/right_ee_pose` from Tianji TCP FK (per-arm base frame, `xyz + wxyz`), the server
  returns absolute TCP poses and openings with `output_format: xpolicylab`, and the adapter
  converts them with IK into two 30×8 joint-position groups. Tianji FK is the checkpoint's
  EE frame, so `eef_reframe_matrix` is the identity.
- TacCap keeps the checkpoint's continuous `[0,1]` opening; close-latch semantics are not
  substituted. A restored gripper value outside the physical range by at most
  `gripper_clip_tolerance: 0.1` saturates to that range; a larger excursion rejects the
  whole action chunk.
- The waist/base slots `[16:20]` have no Tianji actuator and are dropped by XPolicyLab's
  standard EE output.
- Diff-IK decodes the left and right arms in parallel in two spawned processes and commits
  atomically only when both succeed; the control thread no longer runs the whole chunk's IK.
  Diff-IK velocity and step limits match the shared motion profile.
- RTC conditions use the same absolute layout: the adapter converts joint waypoints with FK
  into 30×16 dual-arm rows `[xyz, wxyz, opening]` (rows with weight 0 skip FK and stay 0),
  and XPolicyLab encodes them relative to the observation pose as 60-D native deltas. The
  current experiment template still enables only ordinary ManiMux single-inflight.

## Files

- Model service: `manimux/configs/policy/xiaomi-xr1/tianji/pass_ball/step50000.yaml`
- Experiment template with execution off (station-local; build it like the
  [example study](../../../../manimux/configs/experiments/example-tianji-pi05/README.md)):
  `manimux/configs/experiments/{xr1-tianji-study}/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml`
- Server launcher: `manimux/servers/xiaomi_xr1_tianji_server.py`

The template keeps `robot.options.execute: false` and RoboGUI off. Loading the
configuration, checking the checkpoint or starting the service never moves the robot.

## 1. Install the isolated model environment

Model dependencies must not be installed into the Tianji hardware runtime environment:

```bash
cd third_party/XPolicyLab/policy/Xiaomi_Robotics_1
MIBOT_CONDA_ENV=mibot bash install.sh
```

The Qwen processor defaults to `Qwen/Qwen3-VL-4B-Instruct`; the first start needs
HuggingFace access or an existing cache. On an offline machine, pass a local directory with
`--processor /absolute/local/processor`.

## 2. Check artifacts and configuration only

Run from the ManiMux root:

```bash
conda run -n mibot --no-capture-output python \
  manimux/servers/xiaomi_xr1_tianji_server.py \
  --experiment manimux/configs/experiments/{xr1-tianji-study}/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /path/to/posttrain_pass_ball_50k_8gpu_epoch0_step50000_checkpoint \
  --check
```

Given a checkpoint directory, the launcher resolves exactly one
`mp_rank_00_model_states.pt` and uses the adjacent
`training_metadata/normalize.json`. `--check` validates files, shapes, protocol and the
black-view contract without loading the 5B model.

## 3. Start the model service

```bash
conda run -n mibot --no-capture-output python \
  manimux/servers/xiaomi_xr1_tianji_server.py \
  --experiment manimux/configs/experiments/{xr1-tianji-study}/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /path/to/posttrain_pass_ball_50k_8gpu_epoch0_step50000_checkpoint
```

The service publishes its checkpoint, normalization, `xpolicylab` output format, black-ego
profile and action-semantics identity; ManiMux compares each before a rollout starts, so it
cannot connect to UMI-DP or the wrong XR-1 checkpoint.

## 4. Cameras and a dry run

In another terminal, start the TacCap camera server with only the two wrist cameras:

```bash
conda run -n xense-taccap --no-capture-output python \
  -m manimux.servers.camera.server \
  --config manimux/configs/experiments/{xr1-tianji-study}/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml
```

Keep `execute: false` from the template and run the runtime once to observe the handshake,
inference latency, IK and saved records:

```bash
conda run -n xense-taccap --no-capture-output python -m manimux run --once \
  --config manimux/configs/experiments/{xr1-tianji-study}/pass_ball/tianji_taccap_xiaomi_xr1_step50000.yaml \
  --local .local/tianji_taccap.yaml
```

Execute on hardware only after the dry run, start pose, both wrist-camera orientations, TCP,
the gripper `[0,1]` convention and the full 30-step IK have all passed. Then copy the
template to the Git-ignored `.local/`, review it by hand, and set `execute: true` only in that
copy. This integration started no hardware and claims no real pass-ball success.
