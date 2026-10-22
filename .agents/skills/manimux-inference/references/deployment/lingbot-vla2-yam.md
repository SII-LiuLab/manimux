# LingBot-VLA2 + YAM runbook

## Current local experiment entry (2026-10-20)

Use `manimux/configs/lists/lingbot-vla2-latest300ep-step30000.yaml` for the
five-task study: Serial, zero-blend progress-triggered Async, compiled PAINT and
serial-scheduled adaptive AutoHorizon. The list uses the existing
`paint32_compiled_sampler` and `autohorizon_serial_compile` variants and preserves
their recording identities. The fixed-window methods use 32/50 source steps;
AutoHorizon selects its own window. All tasks currently stop at 6000 control ticks.
Moving cube defaults to `raised_ready`; Home remains selectable for every task.

The model interpreter is `envs/lingbot-vla2/.venv/bin/python`; runtime and cameras
use `envs/yam/.venv/bin/python`. Pass the same `--config-list` to the generic
policy server, camera server and `manimux run`; no experiment ID is needed.
RTC is withdrawn, and BID is excluded from the approved experiment menu.
Historical configs and recordings remain available. The compiled-samplers list
below is a PAINT/AutoHorizon subset, not the full study entry.

> 2026-10-09: The old screwdriver, red-object, pass-ball and offline experiment entries below were retired. These commands are historical; select a current entry from the [experiment index](../../../../../manimux/configs/experiments/README.md). Diagnostic tools now require an explicit experiment argument.


> 2026-10-18: RTC has been withdrawn from LingBot capabilities and experiment lists. Historical RTC measurements below are failure evidence, not launch recommendations.

## Overview

LingBot-VLA2 has released full source and inference entry points. ManiMux now has a dedicated
`third_party/XPolicyLab/policy/LingBot_VLA2/` adapter instead of reusing the old Qwen2.5
`LingBot_VLA`.

The local `checkpoints/pretrained/lingbot-vla-v2-6b` is a foundation checkpoint,
not a YAM post-trained model. Official deployment requires `path_to_posttraining_ckpt`, so foundation
zero-shot operation is not an officially promised use case. To test whether it has learned cross-embodiment priors, this project provides a separate
`server/base.yaml`; execution reuses standard `infra/manimux.yaml` without adding another runtime.
Passing structural checks must not be reported as passing task-capability tests.

## Upstream baseline

- Official source: <https://github.com/Robbyant/lingbot-vla-v2>
- Bundled source baseline: official `951475a`, plus XPolicyLab YAM extensions
- Official foundation weights: `robbyant/lingbot-vla-v2-6b`
- Official RoboTwin post-training weights: `robbyant/lingbot-vla-v2-6b-robotwin`
- Source license: Apache-2.0

Official V2 uses a 55D canonical state/action: 14 arm joints, 14 EE pose, 2
gripper, 12 hand, 4 waist, 2 head, 3 mobile base, and 4 reserved dimensions. Active dimensions actually sent into the model
are jointly determined by training configuration, robot configuration, and norm stats; YAM's 14D
data cannot be placed arbitrarily into a 55D zero vector.

The paper ablates Joint and EEF as separate action spaces, with mean success rates of 55.0% and
56.0%, and notes task-dependent preferences. It does not claim joint+EEF supervision is superior. The official
55D feature/mask mechanism permits both blocks in one robot configuration, so this project adds
joint+EEF supervision as an extension experiment, not as the paper's default setting.

With joint supervision, model inputs remain only joint + gripper 14D. Absolute EEF state anchors relative
EEF targets but does not enter model state. YAM actions have 28 active dimensions: relative joint
12D, local-frame relative EEF 14D (`XYZ + quaternion(xyzw)` per arm), and absolute gripper 2D. Official `joint_mask`
averages the same flow-matching L2 loss over these active dimensions; there is no separately weighted
position loss. Deployment still selects a joint-only robot configuration and executes joint+gripper; EEF is an auxiliary training target.

## ManiMux / XPolicy responsibilities

```text
YAM + cameras
      |
      v
ManiMux canonical observation
      |
      v
XPolicyLab WebSocket protocol
      |
      v
LingBot_VLA2 adapter
  - 3 cameras -> official camera feature names
  - left/right 6 joints -> arm.position[12]
  - left/right gripper -> effector.position[2]
      |
      v
official LingbotVLAv2Server
  - FeatureTransform
  - checkpoint-specific norm stats
  - official flow sampling
      |
      v
model-native arm action[12] + absolute gripper[2]
      |
      v
XPolicy per-step dictionaries
      |
      v
ManiMux PolicyAdapter
  - absolute checkpoint: direct decode
  - relative checkpoint: request-time joints + arm deltas
      |
      v
canonical absolute joint ActionChunk
```

Upstream source owns the model, FeatureTransform, normalization, and flow sampling. The XPolicy
adapter owns protocol and YAM field mapping. ManiMux owns task lifecycle, scheduling, execution, and
independent safety boundaries.

## Current files

```text
finetune server: manimux/configs/policy/lingbot-vla2/yam/finetune.yaml
base server:     manimux/configs/policy/lingbot-vla2/yam/base.yaml
infra:   manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml
adapter: third_party/XPolicyLab/policy/LingBot_VLA2/model.py
sampler: third_party/XPolicyLab/policy/LingBot_VLA2/sampling.py  # PAINT / AutoHorizon
server:  third_party/XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh
profile: third_party/XPolicyLab/policy/LingBot_VLA2/robot_configs/yam_dual_absolute.yaml
source:  third_party/XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/  # vendored upstream source
check:   scripts/validation/check_lingbot_vla2_yam.py
audit:   scripts/validation/lingbot_vla2_yam_audit.py
prepare: scripts/datasets/prepare_lingbot_vla2_base_assets.py
joint+EEF train: training/scripts/train_lingbot_vla2_yam_joint_ee_cluster.sh
joint+EEF profile: training/configs/lingbot-vla2/yam_dual_joint_ee_relative.yaml
stats:   manimux/configs/policy/lingbot-vla2/yam/assets/base_projection_norm_stats.json
```

The `training/` entries above belong to the optional private workspace, which is
ignored by Git and absent from a fresh clone. Deployment uses the policy recipes
and checkpoint artifacts independently of those local training launchers.

There is no second native server entry point. The standard launcher calls
`third_party/XPolicyLab/setup_policy_server.py`, which loads `model.py` solely through `policy_name: LingBot_VLA2`.
ManiMux uses it only through the shared WebSocket bridge.

The shared flow bridge prepares normalized inputs and cached visual prefixes for
PAINT and AutoHorizon. RTC is no longer exposed or implemented by this adapter.

## Explicit deployment configuration

For the server check to report `ready`, provide all of:

1. The official `lingbot-vla-v2` source checkout pinned by this repository.
2. `hf_ckpt/` post-trained on YAM data.
3. The `lingbotvla_cli.yaml` required by the official loader.
4. `norm_stats.json` matching that YAM training data.
5. The exact YAM robot configuration used for training.
6. The native control frequency and action horizon used for training.

As with other models, these values belong directly in the server configuration:

```yaml
lingbot_vla2_root: lingbot_vla_v2
model_root: /path/to/checkpoints/global_step_15000/hf_ckpt
training_config_path: /path/to/lingbotvla_cli.yaml
qwen3vl_path: /path/to/qwen3_vl_4b_processor
robot_config_path: /path/to/robot_config.yaml
norm_stats_path: /path/to/norm_stats.json
action_horizon: 50
native_hz: 30.0
```

`robot_config.yaml` determines whether outputs are absolute joints or arm-relative actions with absolute
grippers. XPolicy preserves native output semantics; ManiMux selects the corresponding `PolicyAdapter`. The server
does not silently convert relative actions into absolute ones.

Official source is included directly under
`third_party/XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/`. Its baseline is official revision
`951475ae1b1d87553e7dc47c97b53a3d695c0d13`, with YAM extensions tracked by XPolicyLab itself.
LingBot no longer requires a nested submodule or personal fork.

The official loader reads the training configuration from `checkpoint_path.parent.parent.parent / "lingbotvla_cli.yaml"`.
Keep the checkpoint in the following recommended layout:

```text
checkpoints/pretrained/lingbot-vla-v2-yam/
├── lingbotvla_cli.yaml
├── norm_stats.json
├── robot_config.yaml
└── runs/yam/hf_ckpt/
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

Three parent levels above `hf_ckpt` must be exactly the directory containing `lingbotvla_cli.yaml`; this is an official loader
path constraint. Server `native_hz` and `action_horizon` must use the actual collection/training
values.
`lingbotvla_cli.yaml` must retain `model.post_training: true`,
`model.config_key: LingbotVLAV2Config`, the 55D canonical configuration, the three-camera order, and
`train.chunk_size` identical to the server configuration.

## Offline checks

These commands check files and YAML/JSON contracts only; they neither import the official Torch model nor start a service:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/check_lingbot_vla2_yam.py
```

The checker reads explicit paths from the server configuration. Missing weight shards, training configuration, robot
configuration, or norm stats produce `status: blocked` without loading a GPU model.

The checker also reads `manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml` and requires:

- `policy.action_dt_s == 1 / native_hz`;
- `policy.horizon_policy_steps == action_horizon`;
- Relative checkpoints must use
  `policy.adapter.type: manimux.policy_adapter.lingbot_vla2.yam:LingBotVLA2YamAdapter`;
- baseline `inference.algorithm == async`.

Thus mismatched training artifacts and execution timing fail before model loading instead of silently
stretching actions in the hardware loop.

RTC configurations are rejected by the LingBot validation tool.

Audit the existing foundation checkpoint's 55D projection:

```bash
envs/yam/.venv/bin/python scripts/validation/lingbot_vla2_yam_audit.py
```

## Base-checkpoint capability experiment

### Norm stats

LingBot cannot reuse XR-1 stats. XR-1 uses `30 x 60` anchor-relative EE deltas; LingBot
in this experiment uses `12` absolute arm joints plus `2` normalized grippers. The repository's
`base_projection_norm_stats.json` was computed from `60` complete YAM episodes and `25,743` transitions,
containing:

- `observation.state.arm.position[12]`;
- `observation.state.effector.position[2]`;
- `action.arm.position[12]`;
- `action.effector.position[2]`.

All four feature groups use the official real-robot configuration's `meanstd`. This gives state inputs and action outputs
correct YAM units, but these are **not the foundation checkpoint's paired post-training stats**,
so they are only for base-capability diagnostics. For a new dataset or formal fine-tuning, use
`third_party/XPolicyLab/policy/LingBot_VLA2/lingbot_vla_v2/scripts/compute_norm_stats.py`
to recompute statistics from the training data configuration, and store the generated `norm_stats.json` beside the checkpoint.

### Prepare base assets

These commands do not copy the 6B weights. They create relative symlinks in the ignored checkpoint directory and generate
the loader-required `lingbotvla_cli.yaml` from the pinned official `real_robot.yaml`:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/datasets/prepare_lingbot_vla2_base_assets.py

envs/yam/.venv/bin/python scripts/validation/check_lingbot_vla2_yam.py \
  --config manimux/configs/policy/lingbot-vla2/yam/base.yaml \
  --infra-config manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml
```

The second command must report `status: ready`,
`checkpoint_variant: lingbot_vla2_6b_base_with_yam_stats`,
and `inference_status: not_verified`. `ready` only means source, 55D architecture, weight shards,
YAM feature mapping, stats shapes, 50-step horizon, and the 30 Hz deployment assumption agree.

### Install the model environment

Like other models, LingBot uses an isolated `uv venv`, fixed at
`envs/lingbot-vla2/.venv`. This ordinary venv is not managed by the root `uv.lock`; do not run
`uv sync` on it. Installation is performed by the operator:

```bash
cd /home/ubuntu/manimux/third_party/XPolicyLab/policy/LingBot_VLA2
bash install.sh
```

The script uses Python 3.12, PyTorch 2.8 CUDA 12.8, and FlashAttention 2.8.3, and installs official
depth dependencies, LingBot-VLA2, and XPolicyLab. After installation, run a GPU/WS forward test before starting
hardware:

```bash
# terminal 1: XPolicy foundation base server
cd /home/ubuntu/manimux
bash third_party/XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh \
  manimux/configs/policy/lingbot-vla2/yam/base.yaml

# terminal 2: no-CAN forward probe
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml
```

Proceed to camera and dual-arm YAM experiments only after the probe returns finite `native_shape: [50, 14]` and
`canonical_shape: [50, 14]` outputs:

This check passed on RTX 4090 on 2026-08-20: Python `3.12.13`, PyTorch
`2.8.0+cu128`, Transformers `4.57.3`, FlashAttention `2.8.3`; all six official shards
loaded successfully. In the second independent service process, the first 10-step denoise took `1.253 s`, with a WS round trip of
`1.392 s`; the next two steady-state denoise passes took `365 / 369 ms`, with WS round trips of `392.3 / 396.1 ms`.
Fifty steps at 30 Hz cover `1.667 s`, so steady-state latency permits continuous supply to default ManiMux.
All three returned finite `native_shape: [50, 14]` and `canonical_shape: [50, 14]` outputs. This evidence
confirms the GPU model, XPolicy WS, YAM field mapping, and action decoding are connected. A default ManiMux
hardware loop subsequently completed; base-checkpoint task capability remains a separate question.

## Hardware operation: Base + ManiMux

Use default Async only initially, without RTC. Clear both arms' workspace, prepare the emergency stop, and open four
terminals in order; do not start a second LingBot server or RTC runtime concurrently.

### Terminal 1: LingBot model service

```bash
cd /home/ubuntu/manimux
bash third_party/XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh \
  manimux/configs/policy/lingbot-vla2/yam/base.yaml
```

After `Model initialized ...`, wait for the service to listen on `127.0.0.1:8501`. To check outputs again,
run the no-CAN probe in another terminal:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml
```

### Terminal 2: Three-camera service

Do not start a duplicate if a camera service already uses `5555`:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux-camera-server --camera-config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml
```

Confirm all three RealSense cameras are open and both `REP bound` and `PUB bound` appear.

<a id="terminal-3robogui"></a>

### Terminal 3: RoboGUI

```bash
cd /home/ubuntu/manimux
.venv/bin/manimux-robogui --robot yam --host 0.0.0.0 --port 8086
```

Open `http://localhost:8086` in the browser.

### Terminal 4: CAN checks and ManiMux

First confirm both CAN interfaces report `ERROR-ACTIVE`:

```bash
for c in can_left can_right; do
  printf '%s: ' "$c"
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

Run only when both report `ERROR-ACTIVE`:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --once \
  --config manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml
```

After connection, the arms take the configured `3.5 s` to reach the initial pose, and return Home in `3.5 s` at completion. For normal shutdown,
press `Ctrl-C` once in the runtime terminal, wait for homing and Recorder finalization, then stop cameras, model service, and
RoboGUI in order. New rollouts are stored under the configured `data/experiments/.../session-*/rollout-*`; unfinished
recordings have a `.partial` suffix. Exploratory recordings before 2026-08-24 are archived together under
`data/archive/pre-campaign-20260824/root-runs/`.

30 Hz is a YAM comparison assumption, not a training frequency recovered from the foundation checkpoint. Record task failure
first as a base-policy result; it does not establish an adapter or ManiMux connection failure.

### Completed hardware record

The 2026-08-20 run `run-20260820T122849Z-b474d665` used
`LingBot-VLA2 YAM via XPolicyLab`, default `manimux` runtime, three real cameras, and dual-arm YAM:

- Executed `1600` control ticks over `61.9 s`.
- Accepted `42` action chunks of `50 x 14`.
- Recorded `42` plan boundaries without plan rejection.
- Runtime ended normally and Recorder finalized.

Thus the default LingBot GPU -> XPolicy WS -> ManiMux -> cameras/CAN -> dual-arm YAM -> Recorder
integration passed. Runtime success in the record means only a complete rollout lifecycle, not
successful picking, and does not change the foundation checkpoint's lack of YAM post-training.

## Commands after fine-tuned weights are ready

Set absolute checkpoint, training configuration, robot configuration, and norm stats paths in the
server configuration, and align ManiMux infrastructure dt/horizon with training. No adapter or
server code changes are needed.
Start the following in order only after the check reports `ready`:

```bash
# terminal 1: model server
cd /home/ubuntu/manimux
bash third_party/XPolicyLab/policy/LingBot_VLA2/setup_eval_policy_server.sh \
  manimux/configs/policy/lingbot-vla2/yam/finetune.yaml

# terminal 2: cameras
# Use the camera-server command already validated at the station.

# terminal 3: ManiMux
envs/yam/.venv/bin/manimux run --once \
  --config manimux/configs/experiments/pick_red_object/lingbot-vla2/yam_lingbot_vla2_manimux.yaml
```

These fine-tuning commands still have no recorded GPU forward, server handshake, camera, CAN, or hardware evidence.

## RTC withdrawal (2026-10-18)

The removed implementation produced conditioned actions but failed the study's
real-time budget: H=50, K=32, 30 Hz leaves 0.60 s after the replan threshold.
Matched synthetic-input sampling measurements (12 warmed repetitions per mode):
native ordinary 120.33 ms; custom ordinary loop 230.67 ms; gradient-enabled forward
without backward 538.13 ms; full RTC 961.32 ms. These exclude preprocessing and transport.
Recorded conditioned service requests averaged 1166.22 ms. The runtime also issued
unconditioned requests when conditions were infeasible, so historical RTC-labelled
records cannot establish the quality of continuously conditioned RTC.

The no-gradient MoE kernel is unavailable with gradients enabled, and the custom
loop bypassed parts of native compilation. This is evidence about this implementation,
not proof that MoE can never support RTC. No optimized replacement is validated.
Keep historical data; do not resume RTC collection or regenerate its configs.

## Latest300ep compilation entry

`manimux/configs/lists/lingbot-vla2-latest300ep-step30000.yaml` selects
`use_compile: true` for Serial, Async, PAINT and AutoHorizon across five tasks.
BID settings described below are retained historical configurations, not active study entries.

The anchor-relative YAM adapter supports repeated candidate decoding for BID:
all candidates from one request share its captured observation joints. The
bounded anchor cache must not consume the anchor after the first decode.
Latest and bottles all-task BID variants request 16 sequential samples and use
a 10-second runtime request deadline (ordinary single-chunk variants retain
their own deadline). This is a timeout budget, not a measured inference latency.
The fixed source window remains 32/50. The official CLI defaults compilation on;
its server constructor defaults it off, so the wrapper must pass an explicit choice.
Historical real-checkpoint CUDA requests returned finite ordinary, RTC and PAINT outputs on synthetic observations; RTC has since been withdrawn. This is not an eager/compiled equivalence or speedup benchmark.

The 2026-10-18 moving-cube recordings exposed a PAINT stall: a 50-step chunk
with a 32-step trigger leaves 0.60 s, but conditioned latency was 0.66–0.79 s.
The shared strategy now pads an insufficient old suffix with its terminal target
and submits normally; the runtime holds that target until the replacement arrives.
New-chunk latency trimming and real request deadlines remain active. This allows
waiting without changing the 32-step trigger or silently switching algorithms.
Offline tests cover terminal hold and resumption; no new hardware run has validated
this fix. An entire new horizon consumed by latency still fails explicitly.

PAINT and AutoHorizon now support `use_compile: true` through cached compiled
prefix and native velocity-prediction regions; sampling control remains in Python. AutoHorizon returns the third-step action attention tensor from
model forward, then applies the unchanged selector outside compilation. It does
not attach a mutable Python hook to the compiled network.

Use `manimux/configs/lists/lingbot-vla2-latest300ep-step30000-compiled-samplers.yaml`
for the five tasks, each with PAINT and serial-scheduled adaptive AutoHorizon.
The same list selects a single compiled model recipe for policy, camera and runtime;
no experiment ID is needed. The new `paint32_compiled_sampler` and
`autohorizon_serial_compile` recording variants preserve previous results.
The old `-autohorizon-eager.yaml` list remains available for historical reproduction.
`use_compile` remains part of backend identity. Restart services only after the
operator normally ends the current experiment. Warm up each selected sampler;
a successful ordinary warmup does not establish that PAINT/attention graphs are ready.

The dated offline report is
`data/analysis/lingbot-paint-autohorizon-compile-20261018/`; it records actual-weight
latency, graph counters and BF16 output differences. These are model-side probes,
not a physical task-success or bitwise-equivalence guarantee.


### Compile handshake and cold-start verification

The model must report `use_compile` from its loaded server object in
`runtime_metadata()`. Adding only the runtime expected field fails HELLO even when
the recipe enables compilation. Offline regression covers true, false, conflicting
config metadata and missing-field rejection; a live HELLO from the loaded latest300ep
checkpoint also passed.

The 2026-10-18 synthetic observation probe (no robot) returned finite `[50,14]`
actions for ordinary, RTC and PAINT. Ordinary first call took 86.52 s; subsequent
ordinary calls took 150.90–225.11 ms. RTC subsequent calls took 1016.23–1106.34 ms;
PAINT's second call took 666.18 ms. These websocket round-trip samples are neither
steady-state statistical estimates nor a matched eager comparison. Do not claim
compiled RTC is faster. Evidence: `data/analysis/lingbot-compile-debug/results.json`.

`policy.options.warmup_request_timeout_s: 300` applies only to the negative-sequence,
display-only requests issued by PolicyWarmup. Ordinary request timeout remains 20 s
and formal runtime timing remains unchanged. The default for all other deployments
is their existing request timeout. Manifests retain the warmup wait setting, but it
is excluded from scientific condition fingerprints; changing the ordinary request
timeout still changes the fingerprint. Runtime must restart to load this option.
