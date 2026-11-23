# OpenWAM through XPolicyLab

## Scope and layout

OpenWAM follows the XR1/Pi05 split: the policy and trainer live under
`third_party/XPolicyLab/policy/OpenWAM`; ManiMux owns robot observations, FK/IK, scheduling,
execution constraints, recording and the robogui. There is no ManiMux OpenWAM
JSON client or standalone OpenWAM service. The vendored upstream deployment
modules remain because the XPolicy adapter uses their loader and preprocessor
in-process, without starting their WebSocket listener.

For an upstream-only environment, install OpenWAM and its XPolicy extras into
an already compatible Python environment:

```bash
bash third_party/XPolicyLab/policy/OpenWAM/install.sh
python -m pip install -e '.[xpolicylab]'
```

For this YAM workstation, use the isolated `envs/openwam/.venv` setup described
under inference below. CUDA/PyTorch compatibility remains the operator's
responsibility; XPolicy integration does not make Pi05's JAX dependencies and
every policy's PyTorch dependencies interchangeable.

## Representation

`yam_base` accepts XPolicy three-camera RGB observations and per-arm
`[x, y, z, qw, qx, qy, qz]` poses in metres in each arm's base frame.
ManiMux computes these with YAM FK and sends them through the shared XPolicy
state extension. No ARX-X5 simulation calibration is applied. Grippers are
absolute `0=closed, 1=open` values. The policy converts poses to EEF20 using
the upstream math, and returns absolute pose dictionaries with explicit
`absolute_per_arm_base_xyz_wxyz` action semantics. ManiMux runs YAM IK;
malformed poses, invalid grippers and wrong horizons reject the whole chunk.
Use the shared `execution.ik.failure_policy`: `hold_last_valid` (default) holds
both arm and gripper at an unsuccessful point and attempts later targets with the
last valid seed; `reject` rejects the chunk. An all-failed chunk remains a recorded
hold under the default, with zero solved rows. Malformed successful solutions remain
errors. The same setting reaches inline/process decoders and step IK, where failure
holds all groups atomically. This is ManiMux execution behavior, not a claim about
the upstream GR1 example. See the [shared IK contract](../../../manimux-development/references/runtime-config.md#chunk-ik-and-execution-time-ik).
The default action rate is 30 Hz and horizon is 32.

ARX-X5 simulation remains supported through `arx_x5_sim`; its calibration and
standard XPolicy batch evaluation are separate from the YAM profile.

## Training source

ManiMux runtime branches do not include OpenWAM dataset conversion, cluster
launchers, or QZ training recipes. Those files are maintained on the
`experiment` branch. This runbook covers deployment of an already prepared
checkpoint and its matching `normalization_stats.npy`.

## Inference and evaluation

Create the isolated local policy-server environment once. It reuses the
CUDA/Torch installation already validated for the YAM machine, while keeping
OpenWAM's Transformers and deployment dependencies outside `envs/yam`:

```bash
bash scripts/setup/setup_openwam_local_env.sh
```

[`manimux/configs/policy/openwam/yam/finetune.yaml`](../../../../manimux/configs/policy/openwam/yam/finetune.yaml)
is an unbound template. Copy it to a private recipe, set `ckpt_dir` to your export
below `${paths.checkpoints}`, `task_name` and `default_instruction`, then bind it with
`--bind-runtime-config` as described below; `{recipe}` is the resulting `-server.yaml`,
which pins the checkpoint and normalization hashes. In the 30k put-bottles study the
policy trajectory remains 30 Hz, while ManiMux interpolates and commands YAM at
100 Hz. Arm and gripper limits are
separate: arms use 0.25 rad/s and 0.5 rad/s^2; normalized grippers use 1.0/s and
12.0/s^2. Its inference schedule is serial: ManiMux executes the committed tail,
then requests the next chunk instead of sampling concurrently.

Start the policy server with the bound recipe:

```bash
envs/openwam/.venv/bin/python manimux/servers/openwam.py \
  --config manimux/configs/policy/openwam/yam/{recipe}.yaml
```

Then validate one real model forward plus the YAM IK boundary without touching
the robot:

```bash
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config manimux/configs/experiments/{openwam-study}/put_bottles/manimux_step30000.yaml \
  --instruction "Put the bottles into the bin."
```

After the probe succeeds, start the camera service and ManiMux runtime:

```bash
envs/yam/.venv/bin/manimux-camera-server --camera-config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml

envs/yam/.venv/bin/manimux run \
  --config-list manimux/configs/lists/{openwam-study}.yaml --experiment-id put_bottles_manimux_step30000
```

```bash
python manimux/servers/openwam.py --checkpoint /path/to/yam_run \
  --runtime-template manimux/configs/experiments/{openwam-study}/put_bottles/manimux_step30000.yaml \
  --bind-runtime-config /path/to/deployment/openwam.yaml
python manimux/servers/openwam.py \
  --config /path/to/deployment/openwam-server.yaml
```

The first command validates artifacts and writes a runtime config plus a paired
server config, without starting a server. These bind the checkpoint directory,
highest-step filename, SHA-256 of weights/config/stats, stats path and action
horizon. Hashing streams 1 MiB at a time but reads the full weight file; perform
this explicitly on the deployment machine, not as a casual laptop check.
Never bind a directory while training is modifying it. The server validates
the bound identity again; ManiMux checks the reported identity at handshake.
Replacing weights/stats or adding a newer checkpoint requires rebinding.
Existing config files are never overwritten by the binding command.

The second command starts only the standard XPolicy server on port 8500.
The unbound repository runtime template is deliberately blocked for the YAM
driver. The deployment launcher requires a real model and rejects dummy policies.
Use disconnected test fixtures in tests/validation scripts for protocol checks.

In another terminal, test the complete inference/IK boundary without hardware:

```bash
python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config /path/to/deployment/openwam.yaml
```

Only after real-checkpoint offline validation and robot setup, run the ordinary
ManiMux runtime for task evaluation, recording and robogui output:

```bash
manimux run --once --config /path/to/deployment/openwam.yaml
```

For RoboDojo simulation, use the policy's standard `eval.sh` and
`setup_eval_*` scripts. YAM real-robot task evaluation uses ManiMux, not an
ARX simulator. The YAM full-horizon adapter also supports AAC and local RTC/PAINT/AutoHorizon
sampler hooks. Specialized modes require a single-stream, full-horizon backend and
CFG scale 1. RTC permits compilation; PAINT/AutoHorizon retain the current eager-only
support boundary. Compiled RTC has synthetic CPU regression coverage, not a validated
real-checkpoint CUDA performance claim. Existing eager experiment records retain
their settings; use a distinct variant when comparing compilation.
RTC can use native DiT caching via `dit_cache_modes: [default, rtc]`; PAINT and
AutoHorizon remain uncached. See the [cache settings and commands](../../../../guideline.html#openwam-rtc-cache). CPU/dummy tests do not establish GPU inference,
training convergence, task success, or real-robot safety.
The trained-continuation copies `OpenWAM_TrainRTC` and `OpenWAM_Legato` opt into
compilation and caching with `dit_cache_modes: [default, train_time_rtc]`; both
default off. Legato's cache tests stability on free rows only and always runs the
final step, because its trained prefix velocity shrinks to zero rather than staying
constant (see the `OpenWAM_Legato` README). Its eager and cache-compile variants
use distinct recipes and recording identities.

## Trained continuation prefix repair

For `OpenWAM_TrainRTC`, `inference.train_time_rtc.repair_prefix: true` enables
exact restoration of the physical EEF20 condition after output transforms and
before pose decoding/latency trimming. It defaults to `false`; use a distinct
experiment variant for repaired trials. The suffix and scheduling are unchanged.
The setting is supported for eager and cached sampling, but not `OpenWAM_Legato`.
Reload the model service and runtime configuration to apply it; leave the operator
in control of Finish/Home and the next Start.
