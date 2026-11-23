# Pi05 + Flexiv OmniGripper joint deployment

The opt-in `flexiv_joint` profile is implemented in
`third_party/XPolicyLab/policy/Pi_05/flexiv_joint.py`, selected by the existing Pi_05 model
adapter and shared WebSocket server. It uses ManiMux's `JointAdapter`; no new
robot driver, transport or inference scheduler is needed.

## Contract

- State and actions are 16 values: left seven joint radians and normalized
  opening, then right seven joint radians and normalized opening. Opening is
  0 closed / 1 open. The profile performs no ALOHA sign or aperture conversion.
- Three HWC uint8 RGB inputs map to base, left-wrist and right-wrist model slots.
  The profile reproduces the training dataset's PIL bilinear resize, rounding
  the scaled dimensions before centered black padding to 224 square. A 640×480
  image becomes 224×168 with 28 black rows above and below. OpenPI then
  quantile-normalizes state/actions,
  tokenizes the prompt/state and pads actions to 32 coordinates. Output removes
  only this padding, preserving all 16 native coordinates.
- The recipe selects full Pi05 Gemma 2B/300M, horizon 50, ten denoising steps and
  seed 0. Native actions are absolute; no delta reconstruction is applied.
- The runtime uses a 30 Hz action grid, serial 16-step prefixes and 50 Hz control.
  Smooth execution is separate from the model. Arm velocity/acceleration limits
  are 0.15 rad/s and 0.3 rad/s². Continuous gripper shaping saturates to [0,1],
  with opening velocity/acceleration limits of 0.4/s and 1.0/s².
- The JAX profile supports default and RTC sampling through the shared Pi05
  sampler. PyTorch remains default-only; PAINT/AAC/AutoHorizon are not enabled
  for Flexiv. Artifact checks describe the JAX profile contract; the loaded
  backend's HELLO remains authoritative.

Recipe: `manimux/configs/policy/pi05/flexiv/stack_cube/joint-step29999.yaml`.
Experiment: `manimux/configs/experiments/stack_cube/pi05/flexiv_pi05_serial_joint_step29999.yaml`.
RTC variant: `manimux/configs/experiments/example-flexiv-pi05/stack_cube/rtc.yaml`.
Camera recipe: `manimux/configs/embodiment/sensor/cameras/gemini_3_views.yaml`.
Station `paths.checkpoints` names the parent of the downloaded run directory.
The recipe's normalization SHA256 is checked before loading weights; the client
also checks paired backend identity, including checkpoint, profile, horizon and
`preprocessing_contract: stack_cube_joint_3cam_v1`.

## Optional model-side compression (2026-11-07)

Joint and EE recipes accept `policy_server.image_processing` in the experiment,
or top-level `image_processing` in the referenced model recipe. The default is
`null`. Enable `type: jpeg_h264_single_frame` to simulate the collection's JPEG80
then H.264 CRF20 path before model resize; codec parameters are documented in
the XPolicyLab Pi05 README. Install PyAV with libx264/H.264 in the model environment.

This replaces the earlier `run.sensor_reading.image_processing` setting. Remove
that reader option and its `workers` setting; sensor reading now always preserves
the original pixels. The shared `SingleFrameCompression` class belongs to the
Flexiv model inputs, not a camera driver. The model processes each supplied image
on each input-transform invocation. Source-frame caching and cross-camera runtime
workers are no longer involved. GUI images and episode videos remain raw.

Model metadata reports the selected and effective settings; model-service INFO
logs report per-image timings, source RGB hashes and codec versions. New episodes
do not emit the old `ticks/image_processing` datasets or
`inference_submitted.camera_processing` fields. Existing recordings are untouched.
Processing now contributes to model-request latency, so previous latency results
below predate this migration. Use a new variant ID for the changed conditions;
the two prepared private studies use the `_model_compression` suffix.

## RTC

The portable Pi05 Flexiv experiment selects the collection-aligned joint safety
profile for both arms: joint impedance with nominal stiffness and damping 0.7,
zero extra joint margin, a 10-degree outward velocity damper, a 5-degree
measured/reference tracking envelope, 0.016-second maximum integrated timestep,
and a 0.5-second persistent tracking timeout. A final clip keeps submitted
targets at SDK bounds without an inward offset (`joint_clip_margin: 0.0`).
`joint_limit_tolerance: 0.0005235987755982988` classifies controller-input excursions
up to 0.03 degree as mild; larger excursions warn but still clip. This diagnostic
threshold does not expand hardware bounds. Serial and RTC inherit this same controller
path; private Async variants must select it through their shared baseline too.
Keep existing operator velocity/acceleration limits. See the
[Flexiv component](../../../../manimux/embodiments/arm/flexiv/README.md#collection-aligned-joint-safety)
for configuration, evidence and limitations. Give changed experiment conditions
a new variant ID instead of mixing them into existing recorded conditions.

The RTC variant inherits the Serial experiment's checkpoint, cameras, 30 Hz
action grid, 50 Hz control, Smooth executor and motion limits. It selects the
shared `RtcInferenceStrategy`, single-inflight algorithm admission, latency
trimming and the full 50-row prediction. Its execution-window floor is 16 policy
steps, guidance beta is 5, and the ten-sample delay buffer starts from warmup
latency (`initial_delay_policy_steps: null`). These are initial settings, not
measured optimal values. Keep the inherited read-only setting until station
qualification; loading an RTC YAML does not authorize robot execution.

The existing `JointAdapter` and XPolicyLab client carry canonical `(50,16)`
absolute targets and `(50,)` weights. `Model.get_action_rtc` validates them;
OpenPI's ordinary input transforms quantile-normalize and pad targets to 32
coordinates before the same JAX Pi0 guidance used by YAM. Output transforms
restore physical units and retain the 16 native coordinates. Unlike Tianji's
pose policy, this joint profile needs no FK or observation-relative conversion.
Scheduling stays in ManiMux runtime; sampling stays in XPolicyLab/OpenPI.

When switching a private station, apply only the RTC variant's inference
overrides and a separate output directory to its existing resolved configuration.
Preserve its calibrated robot model, bindings and operator-selected limits.
Reload the model service to advertise RTC and the runtime to select the variant;
the existing Serial deployment is not changed automatically.

CPU regression coverage uses the real loader, HELLO/INFER dispatch, RTC strategy,
JointAdapter, checkpoint transforms and Pi0 Euler/VJP sampler with a small fake
network. It checks condition order/normalization/padding, zero-weight equivalence,
guidance, reset and invalid requests. This is not a real-weight RTC latency or
physical-task qualification; those checks remain required before deployment.

## Environment and operation

Initialize the pinned XPolicyLab submodule and use its Pi_05 installation script.
Install ManiMux into that model interpreter separately:

```bash
git submodule update --init third_party/XPolicyLab
bash third_party/XPolicyLab/policy/Pi_05/install.sh
uv pip install --python third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -e '.[xpolicylab]' 'numpy<2'
```

Use the Flexiv environment for runtime, cameras and RoboGUI. Camera capture needs
`opencv-python` and `pyudev`; the RDK must match firmware and Python ABI. Bind
`left_wrist_camera`, `head_camera`, `right_wrist_camera`, both arms and both
grippers in the same private station. Physical device assignments come from the
collection station, not USB enumeration order. Keep calibrated arm geometry in
the selected assembly. See `guideline.html#pi05-flexiv` for startup commands.

The experiment enables `run.warmup_before_start`. After **Prepare Run**, the
operator selects **Activate Warmup** to exercise the shared planner and Smooth
executor with predicted command dispatch suppressed, then **Start** to drain
warmup and await backend reset before formal execution. Prepare still owns the
ordinary hardware connection; warmup is not a hardware-free connection mode.

Use the `flexiv_omnigripper` RoboGUI preset, or copy its `cameras` and
`camera_aliases` into a private preset. `gemini_main` occupies `top`, and
`gemini_left_wrist` / `gemini_right_wrist` occupy `left` / `right` in the shared
left overlay. Keep `camera_mode: policy`; these display slots do not change the
model input map. An empty camera list places inputs in the additional-input
folder instead of the overlay.

After a main update, finish the active episode/recovery before restarting the
runtime and RoboGUI with the updated configuration, then refresh the browser.
Existing Python processes retain their loaded UI/runtime code. The current
interactive entry point is `manimux run --config`, not the retired `serve` CLI.
Camera and model services do not need restarting for these display/warmup settings.

## Validation boundary (2026-11-02)

For the supplied step 29999 export, the normalization has 16 finite coordinates
and SHA256 `5dcc19b1f4a1dc0dd48930cff4005e3c22ec5b88643fae5955ee76b1eed60a67`.
Reconstructing the local 38,449-frame collection exactly matches the preparation
report's `state.npy` and `action.npy` SHA256 values. Recomputing all state and
absolute H50 action statistics exactly matches every mean, standard deviation,
q01 and q99 value in the checkpoint. Action windows are t..t+49, repeated at
episode ends; targets are recorded controller q_d, without a one-frame shift.
The collection records 30 Hz and the instruction `stack cube`.

A real-weight forward on RTX 5080 produced finite left/right (50,8) arrays;
the first call including compilation took approximately 8.2 seconds. This is
not steady-state latency. Synthetic images were used, so it does not measure
task quality. Model aperture outputs can slightly exceed [0,1]; the configured
continuous executor handles this without altering the raw model prediction.
An additional real-weight temporary loopback WebSocket check passed the paired
backend identity, capability advertisement, RESET, observation encoding, action
decoding and executor aperture bounds. The test closes its server on completion;
no deployment service is left running.

The user supplied the original training configuration and preprocessing source
in the private training-source snapshot. The source confirms full
Pi05, `wrist_main`, absolute state and recorded joint actions. The training seed
is 42; deployment sampling seed 0 is independent. The data profile uses
`local/stack-cube-joint-20261102`, with `joint-h50-all-train` normalization assets.

The source's `reports/source_versions.json` pins upstream OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`. Its Pi0Config, PaliGemma tokenizer,
ModelTransformFactory, DataConfigFactory and trained-policy factory match this
vendored runtime at the Python AST level. Training inputs and deployment inputs
were compared for 640×480, 641×479, 500×333 and 224×224 RGB frames: image slots,
pixels, masks, quantile normalization, tokens, padding and unnormalized outputs
are array-equal. Pixel normalization is float32 [-1,1]; training augmentation
remains disabled during inference. The local TokenizePrompt adds a batch branch;
the single-observation branch used here was checked against the original source.

Source SHA256: `src/pass_ball/dataset.py` =
`13db9083da25eacef2fc6d9e28bddc3ca09de0b6ea1ee78fb10f73de84ef0e57`;
`src/pass_ball/openpi_adapter.py` =
`c45d264c1b588747b4ddc18f54b54fc214943f93beb8576962d3ab0c249b695a`.
The source snapshot is provenance, not a runtime dependency. Reproducible checks
and results are in `.local/flexiv/pi05_source_check.py`, `pi05_source_result.json`,
`pi05_full_stats_check.py` and `pi05_full_stats_result.json`.

The collection-matched gripper profile is selected in experiment
`robot.options.component_hardware`: force-position, 100 Hz, kp/kd 8/0.2,
reference speed/acceleration 1.2 rad/s and 6 rad/s², encoded PD torque budget
0.3 Nm, independent measured-torque trip 3 Nm. This inner controller is separate
from the executor's normalized opening slew limits. Bind motor SN 0/1 and raw
calibration bounds in this station; never infer sides from device enumeration.

With the user's approval the teleoperation preview was stopped, releasing all
three cameras. Real UVC capture through the shared camera server and runtime
sensor passed: 151 distinct RGB 640×480 frames per view over five seconds,
29.97–30.06 FPS, maximum age 38.2 ms. Both grippers passed simultaneous five-second
100 Hz read-only polling (501 samples each), SN 0/1, CAN-rate register 4,
MIT mode, status 0 throughout, temperature 27–30 C. The temporary validation
services closed on completion. Evidence: `.local/flexiv/pi05_camera_result.json`,
`pi05_gripper_readonly_result.json`, plus camera script and the reusable
`scripts/validation/omnigripper_readonly.py` probe.

Execution and automatic robot enable remain disabled. No arm SDK connection,
force-position motion or physical task trial was performed by these checks.


## Local manual-recovery alignment (2026-11-03)

The workstation's Pi05 deployment now selects `backend: flexivpy` and the existing
Python 3.12 SDK in `.local/flexiv-sdk`, matching `omnigripper_control`. Its assembly
enables `hardware.hand_guiding`. This exposes Clear error, Home, Drag, Release,
Release 5s and A/B/AB selection through the shared session/RoboGUI protocol.
The grouped `flexivrdk` path does not implement manual guiding; changing only the
hand-guiding flag is insufficient on that backend.

Keep the task-specific collection Home and calibrated arm/tool geometry. Preserve
the checkpoint, observation/action contract and current operator-selected motion
limits when changing the SDK binding. The current local execution/automatic-enable
flags are true; the portable example remains read-only. The dated November 2
read-only checks above describe their original test configuration.

Configuration/ABI checks and recovery lifecycle tests passed without connecting
hardware. Restart the runtime to load these bindings; the model, camera and GUI
services need no restart for this configuration change. Physical operation of the
aligned Pi05 recovery controls still requires station acceptance.


## Main integration (2026-11-03)

External model sources now live in `third_party/XPolicyLab`; Flexiv deployment
examples live in `manimux/configs/experiments/example-flexiv`. Update private
experiment `run.task` to `run.task_prompt` while retaining the same instruction.
The local Flexiv profile and upstream DROID/ChunkTrust changes are merged in the
pinned XPolicyLab revision; do not replace it with the upstream revision alone.

The pose adapter retains bare-arm support and follows the shared
`execution.ik.failure_policy` for both bare and gripper-equipped arms. Background
camera polling is paused after the fresh-frame preflight and restarted after
blocking robot initialization, preserving the upstream failure attribution and
latched pause-hold behavior. Main-merge checks use fake devices and offline
artifact inspection; they do not operate or restart physical hardware.

## Flexiv flange EE / UMI checkpoint (2026-11-05)

The separate `flexiv_ee_umi` profile in `Pi_05/flexiv_ee.py` selects the full
Gemma 2B/300M checkpoint `pi05-stackcube-flexiv-ee-umi-b64-h50-30k-p2-20261105-060841/29999`.
Its recipe is `policy/pi05/flexiv/stack_cube/ee-umi-step29999.yaml`; runnable
examples are `experiments/example-flexiv-pi05/stack_cube/ee_umi_serial.yaml` and
`ee_umi_async.yaml`, relative to `manimux/configs/`. Both keep actuation disabled.
Bind `paths.checkpoints` to the parent of the downloaded run directory.

The submitted training source establishes these differences from the joint policy:

- Stored targets are the next measured **flange** pose, clamped at episode ends;
  H50 therefore predicts t+1 through t+50. No gripper-tip offset is trained.
- Native rows are right then left, each xyz metres, first-two-rows Rot6D and
  normalized opening. Tool-axis conversion is `R_train = R_flange @ C.T`, with
  the existing HiFi-UMI optical-from-FLU matrix `C`.
- All targets in a chunk share this request's measured observation anchor.
  Relative translation and rotation are expressed in the anchor tool frame.
  Pose state is zeroed before normalization and again before tokenization;
  openings retain quantile normalization. Full three-camera RGB preprocessing
  matches the joint recipe's PIL resize/padding, without color swaps.
- The provider reconstructs absolute flange targets from the saved request
  anchor. `PoseAdapter` supplies measured FK to the provider and converts the
  returned left/right xyz+wxyz+opening targets with the independently selected
  execution-reference IK seed. Reference joints never replace model observations.
- The checkpoint's normalization SHA256 is
  `e3e065af9199dab8cfbaf55e4c1912fdd8b7e4e020b30a92d8436b962be289aa`.
  Its values exactly equal the preparation report's normalization; the export's
  JSON formatting accounts for the different file hash.

`embodiment/robot/flexiv_omnigripper_ee.yaml` selects the existing offline Flexiv
solver with `ik_max_joint_step_rad: 0.15` and `ik_joint_motion_cost: 0.001` on
each arm. Like YAM's nearby IK, it bounds each solve around the last accepted
joint vector and regularizes joint motion; unlike YAM's approximate DLS acceptance,
it retains the original 0.1 mm / 0.0001 rad acceptance criteria. The box is a
branch-continuity constraint, not a physical velocity limit. Existing Smooth and
controller limits still apply. Seeds outside model bounds fail explicitly.
Keep calibrated URDF paths in a private assembly, copying these two model options.

`policy.adapter.pose_frame: flange` checks that mount and tool geometry cancel
to the flange at both gripper limits. A fingertip TCP is rejected. The opt-in
`gripper_output: clip` saturates finite regression overshoot with diagnostic
metadata before IK; raw poses/openings remain in evidence. Other pose recipes
retain their strict default. `execution.ik.failure_policy: hold_last_valid` holds
the complete previous arm-and-tool command at an unsuccessful solve and retries
later targets. `reject` rejects the chunk. Chunk decoding runs in independent
processes, one per arm, so IK does not block control ticks. Pre-Start warmup uses
these same processes and drains them before acknowledging backend reset.

The JAX integration supports **default and RTC sampling**. Serial and ordinary
Async remain available. RTC uses absolute pose conditions through step IK; joint
RTC conditions must not be sent to this EE profile. PAINT/AAC are not enabled. The portable Serial recipe
inherits the joint recipe's 30 Hz action grid, 50 Hz control and 16-step prefix;
Async retains all 50 rows. Private station variants may retain the operator's
existing control rate, image processing and motion limits.

Validation used recorded `stack_cube_1` observations at frames 0, 120 and 240,
real checkpoint weights and a temporary loopback WebSocket server. Training and
deployment inputs/outputs were array-equal across four image shapes, including
zero-state normalization, tokens, action padding and UMI reconstruction. All 300
arm targets across the three requests solved. Cold inference was about 8 seconds,
steady inference about 117 ms, and sequential full dual-arm IK about 0.9 seconds.
The recording's per-arm reference differs from offline FK by approximately +/-90
degrees about Y; after a fixed reference alignment, the three samples differed
by at most 0.37 mm and 0.00053 rad. Zero-pose input and tool-relative actions are
invariant to a fixed per-arm reference transform; this does not certify all
workspace calibration.

The actual Async runtime with process decoding completed 1200 synthetic control
ticks in about 12.7 seconds, admitted 15 plans and rejected none. Process warmup
completed its reset fence with zero hardware sends; its maximum measured tick
was about 1.2 ms. Failure/recovery tests changed gripper targets across an
unreachable pose and verified both hold and reject behavior. Request-associated
raw wire responses, image hashes and would-be commands were retained locally.
Malformed-pose and timeout injection also verified rejection and persisted
request/error records; malformed wire data was saved before shape validation.
These are offline software results with recorded pixels, not live sensor or
physical task qualification. No motor driver was constructed or connected.

For a one-off EE trial, use the same selected config and station for all services.
The following commands are an operator handoff, not permission to connect hardware:

```bash
MM_EE_CONFIG=manimux/configs/experiments/example-flexiv-pi05/stack_cube/ee_umi_serial.yaml
MM_EE_LOCAL=manimux/configs/local/station.yaml

envs/flexiv/.venv/bin/python -m manimux inspect --config "$MM_EE_CONFIG" --local "$MM_EE_LOCAL"
envs/flexiv/.venv/bin/python -m manimux.servers.camera.server --config "$MM_EE_CONFIG" --local "$MM_EE_LOCAL" --check
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 --experiment "$MM_EE_CONFIG" --local "$MM_EE_LOCAL" --check

# Each service runs in its own terminal after offline checks and operator authorization.
envs/flexiv/.venv/bin/python -m manimux.servers.camera.server --config "$MM_EE_CONFIG" --local "$MM_EE_LOCAL"
XLA_PYTHON_CLIENT_PREALLOCATE=false third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 --experiment "$MM_EE_CONFIG" --local "$MM_EE_LOCAL"
envs/flexiv/.venv/bin/python -m manimux.robogui.dashboard --robot flexiv_omnigripper
envs/flexiv/.venv/bin/python -m manimux run --config "$MM_EE_CONFIG" --local "$MM_EE_LOCAL"
```

RoboGUI has no `--check` option; validate its YAML with `load_robogui_config` and
`load_robot_view` offline. For a calibrated station, select its private EE assembly
and matching RoboGUI YAML instead of the nominal preset. The operator owns
Prepare, Activate Warmup, Start and Finish. Keep the joint deployment and its
recordings separate; these configs do not switch running services.

### Optional Flexiv teleop diffIK

Use `experiments/example-flexiv-pi05/stack_cube/ee_umi_diffik_serial.yaml` or
`ee_umi_diffik_async.yaml` for execution-time differential IK. Set `MM_EE_CONFIG`
to that variant in the complete service/check sequence above. Keep the same EE
checkpoint, calibrated private assembly, station bindings and RoboGUI geometry.
The local variant should inherit the station's current motion limits. These
examples preserve the joint baseline's executor limits; they do not automatically
raise them to the teleop solver's velocity ceiling.

The configuration selects `execution.ik.mode: step`, `solver: differential`,
inline pose decoding, Direct execution at 100 Hz and a shared 5 ms solve budget.
All solver constants are explicit in `execution.ik.differential`. The solver
ports `tianji-control/algos/flexiv_diff_ik.py` at revision
`4d20258bdf8d53d66798da07691d7c5bbcb4c023` (source SHA256
`ed9df3e3848cecf386eeab582436b0be0942cc577cf005ed5e5783df18c9c328`).
It uses offline calibrated URDF FK/Jacobians, not SDK IK. Previous command,
measured joints and nominal dt remain independent. Teleop's bounded velocity
solve, Cartesian speed caps, soft joint-limit damping, 5-degree tracking envelope
and nullspace avoidance are preserved. Commands are bounded increments, not
claims of exact waypoint convergence. See the
[execution contract](../../../manimux-development/references/runtime-config.md#flexiv-execution-time-differential-ik)
for reset, failure, final-command bounds and recording semantics.

Offline validation on 2026-11-05:

- 600 comparisons against the source algorithm, including joint boundaries,
  measurement lag, capped dt and nullspace bias: 60 matching tracking failures;
  maximum accepted joint difference `4.44e-16 rad`. Calibrated-URDF Jacobians
  matched central differences to `1.52e-10` maximum element error.
- Real step-29999 weights, recorded three-camera frames 0/120/240, WebSocket,
  pose decoding, timeline and Direct commands: 492 control samples. With each
  recorded measured state held fixed, six post-executor envelope violations
  triggered complete holds; 486 steps were accepted. No observation was replaced
  by a predicted command. Dual-arm solve p50 was about 1.35 ms.
- The actual Async runtime with a disconnected synthetic plant and recorded
  camera images ran 1200 ticks in 12.48 s, admitted 49 plans and rejected none.
  Host control frequency was 99.47 Hz; dual-arm IK p50/p95/p99 was
  2.09/3.37/5.48 ms. Seventeen 5 ms deadline misses produced recorded atomic
  holds; 861 accepted steps reported active joint bounds. These counts are
  command-generation evidence, not motor timing or real tracking performance.
- Warmup used the same differential pipeline and reset fence with no hardware
  sends. Failure/recovery, reject, shared deadline, tool transform and final
  executor-envelope tests passed. Raw responses and generated commands were
  retained privately. Camera/model `--check` and config inspection passed.

No controller was constructed or connected. These results qualify offline
integration only; physical stack-cube performance remains untested.

### Joint / EE and Serial / RTC config list

`lists/example-flexiv-pi05-models.yaml` provides four portable choices with
actuation disabled. The reference station's private list is
`lists/pi05-flexiv-stack-cube-models.yaml`; paths here are relative to
`manimux/configs/`. It retains the installed station's calibrated EE assembly,
100 Hz control, image processing, controller and executor limits. The two joint
choices use Smooth; the two EE choices use Direct plus execution-time diffIK.

| Choice | Model | Inference |
| --- | --- | --- |
| `joint_serial` | Joint step 29999 | Serial, full H50 |
| `joint_rtc` | Joint step 29999 | RTC, minimum window 16, beta 10 |
| `ee_serial` | EE UMI step 29999 | Serial, full H50, diffIK |
| `ee_rtc` | EE UMI step 29999 | RTC, minimum window 16, beta 10, diffIK |

These windows preserve the installed joint comparison's H50 Serial and K16 RTC
choices; they are not the separate YAM K32 study. Both RTC variants condition on
the executing timeline at the new observation timestamps. For EE, absolute flange
poses are converted to right/left training Rot6D/tool axes, then re-expressed
against the new measured anchor before zero-pose state encoding, normalization
and action padding. No future IK is solved to construct the condition. Joint
conditions are never treated as pose rows. Entirely zero-weight RTC requests use
the default sampler graph to avoid bf16 graph-dependent drift after RNG reset.

The list opts into `policy_switching: on_demand`. Run the generic policy launcher
in the Pi05 environment; do not use the single-recipe Pi05 launcher or pass a
model-service `--experiment-id`. The router snapshots the resolved recipes,
passes XPolicyLab messages unchanged, and owns one model child at a time. It
refuses replacement while a client is connected. Same-model Serial/RTC switching
reuses weights; cross-model switching terminates only its owned child and loads
the other checkpoint. Normal HELLO identity, capability validation and RESET fences
remain mandatory. A failed load leaves no selected model and surfaces an error;
retrying does not connect hardware. Graceful shutdown and Linux parent-death
signalling release the owned model child.

In RoboGUI, finish the active rollout, evaluation and recovery, then select the
Method label for the desired model/algorithm. The next Prepare creates a fresh
worker, adapter, strategy, timeline and warmup state. Activate Warmup and wait
before Start. Adding a catalog entry does not change existing variants' scientific
fingerprints. Recordings remain separated by collection/task/variant; no layout
or repeat budget is invented for this station list.

Use the same list/station for all services. Finish the existing session before
replacing its runtime or occupied model service; existing camera/GUI services
can be reused when their configuration matches. The private station's list has
execution and enable-on-connect enabled; the operator owns physical controls.

```bash
cd <manimux-checkout>
MM_LIST=manimux/configs/lists/pi05-flexiv-stack-cube-models.yaml
MM_LOCAL=.local/flexiv/pi05_ee/station_run.yaml

envs/flexiv/.venv/bin/python -m manimux inspect --config-list "$MM_LIST" --local "$MM_LOCAL"
envs/flexiv/.venv/bin/python -m manimux.servers.camera.server --config-list "$MM_LIST" --local "$MM_LOCAL" --check
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.policy --config-list "$MM_LIST" --local "$MM_LOCAL" --check

# Separate terminals; define MM_LIST/MM_LOCAL in each.
XLA_PYTHON_CLIENT_PREALLOCATE=false third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.policy --config-list "$MM_LIST" --local "$MM_LOCAL"
envs/flexiv/.venv/bin/python -m manimux.servers.camera.server --config-list "$MM_LIST" --local "$MM_LOCAL"
envs/flexiv/.venv/bin/python -m manimux.robogui.dashboard --config .local/flexiv/pi05_ee/robogui.yaml --host 127.0.0.1 --port 8086
envs/flexiv/.venv/bin/python -m manimux run --config-list "$MM_LIST" --local "$MM_LOCAL" --experiment-id ee_serial
```

Recorded-image/real-weight validation on 2026-11-05 passed all four choices,
including joint -> EE -> joint child replacement, same-model process reuse,
backend identity and zero-weight/default equivalence. Model loading took about
7–8 s; first RTC compilation about 7 s. EE steady conditioned inference was about
164 ms. The actual RTC runtime completed 1800 disconnected-sink ticks, accepted
33 plans and rejected none. Host frequency was 99.74 Hz; 21 dual-arm 5 ms budget
misses produced complete holds. Warmup completed one ordinary and two conditioned
calls with zero hardware sends. These timings do not establish motor timing or
physical task success. Provider responses and generated commands remain in the
private verification artifacts; the router tests also cover active leases,
unknown routes, failed loading and idle-only selection.
