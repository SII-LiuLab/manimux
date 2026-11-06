# Pack-plate Pi05 checkpoint comparison

This independent catalog contains nine checkpoints: seven TCP-action checkpoints
with Serial 32 and RTC 16 in both chunk and step DiffIK, plus two native-joint
checkpoints with Serial 32 and RTC 16: 32 choices across 16 policy groups.
The initial selection is **Joint-FK + TacCap mix / 29999,
Serial 32**. It uses the existing managed-policy lifecycle: one model is resident,
and changing Serial/RTC within one policy group does not reload its weights.
Step groups append `/ Step DiffIK` to each checkpoint label; switching between
chunk and step groups uses the existing managed model restart lifecycle.

## Launch

From the repository root:

```bash
study=manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261019
bash "$study/launch.sh" check
bash "$study/launch.sh" runtime
```

If the compatible camera and RoboGUI services are not already running, start each
in a separate terminal with `bash "$study/launch.sh" camera` and
`bash "$study/launch.sh" gui`. Reuse existing matching services instead of starting
duplicate owners. The launcher uses the project's TacCap 0.4.2 SDK for hardware
processes and removes that Python path from managed model processes.

Do not start a second Pi05 server for this catalog. Runtime manages the selected
model at `ws://127.0.0.1:8574`; model startup has a 600-second timeout. An unrelated
GPU model is not stopped by this launcher. Runtime shutdown stops only its owned
model service.

Every study entry sets `policy_server.checkpoint_low_memory: true` to reduce host
RAM peaks. The loader reads parameters sequentially, uses one TensorStore file I/O
and decompression worker, and waits for each GPU transfer. It keeps the original
checkpoint values and bfloat16 inference dtype. This is a loading strategy, not
a total process RAM cap; model semantics and Serial/RTC settings are unchanged.
If startup was killed by the host OOM killer, Retry policy reloads the updated
experiment and model code; freeing GPU memory alone does not address host RAM.

In RoboGUI: **Policy → Switch policy → Task: Pack plate → Method / Config:
Serial 32 or RTC 16 → Prepare Run → Activate Warmup → Start**. For step execution,
select the checkpoint group ending in **/ Step DiffIK**, then choose
**Serial 32 / Step DiffIK** or **RTC 16 / Step DiffIK**. Selection alone
does not move the robot. Execution is enabled for explicitly started rollouts;
Tianji does not accept the YAM-specific `move_to_start_on_connect` or
`home_on_close` constructor options. Home is an explicit robot recovery action
using this study's `robot.yaml` snapshot. Finish and complete recovery/evaluation before switching.
The 2026-10-20 integration adds runtime sensor code: finish the current rollout
and evaluation, stop runtime with Ctrl-C in its terminal, then run the commands
above once. Existing compatible wrist-camera and RoboGUI services can remain.
Later YAML-only edits can use **Reload config** while idle.

## Available policies

- Joint-FK + TacCap mix / 29999: the requested 2026-10-19 full checkpoint.
- Joint-FK TCP / 29999: full zero-pose model using FK next-state TCP targets.
- Joint / Wrist / 29999: full model using measured absolute joints and two wrists.
- Joint / 3 cameras / 29999: the same joint contract plus the real main RGB image.
- Full zero-state / 59999 and / 20000: the existing full-data checkpoints.
- Inter-hand relative / 59999 and / 20000: the existing LoRA checkpoints.
- Wrist-only legacy / 59999: the original wrist-only checkpoint.

`run.task_label: Pack plate` is presentation only. The new and full-data models
receive `place each plate into a bubble wrap sleeve`; legacy wrist-only and
inter-hand models retain `plate`. These prompts, architectures and state encodings
are part of each model's deployment, so the catalog compares complete policies;
it is not an ablation holding all training and prompting choices fixed.

The three additional exports were integrated on 2026-10-20 after checking their
download-complete markers, file sizes and checkpoint-local normalization hashes.
All use full Gemma 2B / 300M weights, 32 steps at 30 Hz and the full plate instruction.

The two native-joint models use `policy-runtime-joint.yaml` and the existing
`JointAdapter`: left seven radians + left opening, then right seven radians +
right opening. Measured state is retained and quantile-normalized; actions and
RTC conditions remain absolute joints. They perform no TCP anchoring, axis
conversion or IK. They reuse the smooth executor, safety limits and force-position
grippers. The aperture curve and inverse RTC mapping are shared with the TCP
experiments. Inline joint decoding sets the decode forecast to zero. There are
no redundant Step DiffIK choices for these models.

The 3-camera choice additionally starts `main_camera` as a runtime-owned UVC
sensor. Its explicit `device_path` is in the private station's
`robot.components.main_camera` binding; on this station it is the SN0002 USB camera
confirmed as the non-TacCap main view. Capture is 1280x720 RGB at 30 Hz, matching
the collection application's main-camera settings, before OpenPI's 224x224 resize.
The sensor reuses the project's callback MJPEG backend, publishes host receipt
timestamps, and releases the video device when the rollout closes. The existing
wrist camera server remains unchanged; other policies do not open the main camera.
A missing main image fails explicitly instead of substituting a black/wrist image.

## Frozen execution settings

The study owns `robot.yaml`, `policy-runtime.yaml`, `executor.yaml`, `serial32.yaml`
and `rtc16.yaml`. These are snapshots of the existing Pi05 chunk-Diff-IK deployment;
they do not load another experiment or a mutable shared inference/executor preset.
Robot component implementations, camera recipe and existing model recipes are reused.

As of 2026-10-20, every choice uses two independent position envelopes:

- Command/IK: base joint range inset by **3 degrees**. Effective degree ranges
  are J1 ±167, J2 ±117, J3 ±167, J4 [-142, 57], J5 ±167,
  left J6 ±57 / right J6 ±55, and J7 ±87.
- Measured feedback: base range inset by **0.5 degrees**, with checks enabled in
  both driver and runtime. Ranges are J1 ±169.5, J2 ±119.5, J3 ±169.5,
  J4 [-144.5, 59.5], J5 ±169.5, left J6 ±59.5 / right J6 ±57.5, and J7 ±89.5.

These are absolute margins from the base range, not increments to the previous
2 degree profile. The 2.5 degree separation allows feedback to pass the command
boundary without immediately triggering the measured-range fault. Commands
must still remain inside their own bounds, including hold commands. Native joint
outputs still have no IK feasibility projection and may be rejected if outside
command bounds. Chunk and step DiffIK position and J6/J7 margins inherit 3 degrees.

`robot.yaml` owns arm `options.limit_margin_deg`, `hardware.joint_limits`, and
independent `hardware.measured_joint_limits`. `executor.yaml` mirrors the two boxes
in `command_safety.position_lower/upper` and `measured_position_lower/upper`.
Both `check_measured_position_limits` switches are explicitly `true`.
`launch.sh check` verifies alignment for all 32 choices through the real loaders.
Other deployments and the sibling teleop repository are unchanged. Earlier
recordings retain their original settings and provenance.

Executor and runtime arm-velocity limits remain pi rad/s (180 deg/s); chunk and
step DiffIK caps remain pi/2 rad/s (90 deg/s). Isotropic execution, the null arm
acceleration bound, gripper settings, hardware ratios and the 10 degree tracking
error threshold are unchanged. Finite/fresh feedback and hardware faults still
stop execution. Measured-bound failures now report the joint, angle and bounds.

Restart the serving runtime once after finishing the rollout to install the
independent measured-bound code. Later YAML-only edits can use idle **Reload config**.

- 100 Hz control, 30 Hz policy actions, 32-row model horizon, 10 denoising steps.
- Process-based chunk Diff-IK with dense substeps; IK seed uses execution reference.
- Smooth execution at 8 Hz, arm velocity bound pi rad/s, with pi/2 rad/s chunk-QP cap.
- Force-position grippers, grasp torque 1.0 Nm, close speed 3.0 rad/s.
- No additional handoff skip or blend; Diff-IK lag remains `report` at 5 mm.
- Serial executes 32 rows before requesting again, with `first_step_when_ready`.
- RTC requests at 16-step progress adjusted for forecast latency, beta 5,
  initial delay 8, delay buffer 10, and `drop_infer_latency`. It retains the full
  32-row prediction: RTC 16 is not a fixed execution-length cap.

## Step DiffIK counterparts

Each TCP experiment has a separate `*_step_ik.yaml` counterpart. It retains
checkpoint identity, task, station, camera mapping, hardware settings, model action
spacing/horizon, Serial/RTC scheduling, joint position and velocity guards, and
arm/gripper motion limits. Original chunk files and their default selection remain
unchanged. Step policy groups use `policy-runtime-step.yaml`; the executor references
`executor.yaml` with `type: direct`, keeping its shared limits.

Required execution differences:

- Inline absolute-TCP decoding replaces process joint decoding and dense substeps.
- `execution.ik.mode: step`, `backend: diff`, and a shared dual-arm budget of 5 ms
  select the existing 100 Hz execution pipeline. The adapter's `ik_backend: analytic`
  only configures pose-only decoding; it does not select the execution solver.
- Direct execution replaces the 8 Hz smooth filter. Executor arm velocity is pi rad/s,
  step QP velocity is pi/2 rad/s, with `max_acceleration: null`; the runtime
  arm-velocity guard is pi rad/s.
- QP tuning is copied from the chunk decoder into `execution.ik.diff_ik`, retaining
  the 5 mm reporting threshold and the existing joint margin/nullspace settings.
- No additional per-tick Cartesian limiter is enabled (`pose_speed_limit: null`).
  Decode-time waypoint handoff is removed; it is not a step-mode operation.
- The subprocess decode forecast is disabled (`expected_decode_s: 0`,
  `decode_forecast_size: 0`). All other inference settings are preserved.
- The direct profile explicitly supplies the same aperture curve (deadzone 0.1,
  exponent 1.05) to the adapter, including its inverse for RTC conditions.

Differential step IK seeds from the last dispatched command, then direct execution
applies the existing motion limits. Failed or late IK holds both arms. Pose plans
remain in TCP space for RTC; scheduled and sent commands are joint-space rows.
The step variants use distinct `_step_ik` recording directories and conditions.
They are not claimed to produce the same physical path as chunk IK plus smoothing.

Each setting records under
`data/experiments/pi05-pack-plate-checkpoint-comparison-20261019/<checkpoint>/<method>/`.
No results, scoring weights, layouts or repetition budget are invented here.

The private station is
`.local/pack_plate/pi05_checkpoint_comparison_20261019/station.yaml`; it copies the
known controller, wrist-device and camera bindings and owns the separate policy
endpoint. Its checkpoint root is `/home/jw/Desktop/policy`.

## New checkpoint provenance

Recipe: `manimux/configs/policy/pi05/tianji/pack_plate/joint-fk-taccap-mix-zero-state-step29999.yaml`.
The export's `training_run.json` declares full `gemma_2b` / `gemma_300m`, zero-pose
state with live grippers, right/left 20-D TCP-relative actions, wrist RGB inputs,
masked base image, first-two-row Rot6D, tool-axis conversion, 30 Hz and horizon 32.
Joint-FK describes the source of training TCP poses, not joint action outputs.

The export normalization SHA-256 is
`1d0ce205cd6e2aacdb694a41bf4f328e28bfd771a38c9a3c86dc4e88de2e593d`.
It matches `SHA256SUMS`. The training metadata's `019c9964...` hash is exactly the
same bytes plus one final newline; numerical contents are unchanged. The exported
download-complete marker reports verification of all 71 files; this integration
rechecked the normalization hash, not every weight shard. The training tool-YAML
hash matches the local teleop `configs/tool/umi.yaml`.

Training metadata records source hashes but does not bundle the original training
transforms. Deployment reuses the established zero-pose transform chain, including
`ZeroNormalizedPose`; metadata agreement alone is not byte-for-byte verification
against the original training preprocessing.

## Validation and extension

`launch.sh check` loads every configuration, validates shared station/service
contracts, constructs the disconnected robot assembly, adapter and strategy, and verifies each model's
artifact identity and normalization. For every step variant it also checks parity
with its chunk counterpart, decodes a synthetic TCP chunk, runs dual-arm step QP
and direct execution near Home, checks the joint-rate bound, and exercises the RTC
pose condition/aperture round-trip. It starts no service or hardware. These CPU
checks do not establish sustained timing or real-robot step-rollout success. A separate
`manimux.servers.pi05 --experiment ... --check` also passed in the model interpreter.
On 2026-10-19, the new checkpoint loaded in 18.4 seconds with a 7.02 GiB peak
process RSS and passed both default and RTC synthetic-image forwards (32 finite
action rows each). The offline process ran with a 10 GiB cgroup memory limit and
no cgroup swap. This verifies model loading/inference, not physical task success.
Evidence: `data/analysis/pi05-startup-memory-20261019/offline-forward.json`.
Four small-checkpoint regression cases also verified identical values/dtypes and
that the process-global Orbax handler registry stays unchanged. Detailed static
check results are saved beside the private station.

For the three exports added on 2026-10-20, all download-manifest file sizes were
rechecked (73 TCP, 66 joint-wrist, 76 joint-3cam files). Their download-complete
markers report full SHA256 verification; this integration independently rehashed
the normalization files. The actual TCP normalization hash is `b760a1ea...`,
and the two joint exports share `e20129a3...`. Training metadata hashes the same
numerical JSON with different formatting/final newline; recipe hashes pin the
actual exported bytes.

The expanded check also exercises native wire joint decoding, aperture inversion
for RTC and inert main-camera construction. CPU tests run the real OpenPI
normalization, tokenization, padding, `Policy.infer` and output transforms with
only the neural sampler substituted, for both default and RTC. Fake-camera tests
cover explicit device selection, RGB, timestamps and release. Two legacy camera
server tests still reference a removed standalone recipe/old loader format and
are outside these focused checks. These three new weights have not been loaded
for inference, and the main camera has not been opened during integration:
the active model was preserved and available host RAM was insufficient for a
second safe model load. First Activate Warmup remains the real-weight check.

To add another compatible TCP checkpoint: add its recipe and identity registration,
copy this study's four experiment files with distinct conditions/output directories,
then add chunk and step policy service groups and choices to `catalog.yaml`. Keep its own state
encoding, normalization and training prompt. Joint-action or three-camera models
need their own adapter/observation contract; all entries must still resolve to a
compatible station and camera service. Update the validation counts in `check.py`,
validate the entire catalog, then use Reload config while idle to load its choices.
