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
Chunk IK defaults to `policy.adapter.ik_failure_policy: reject`. The explicit
`hold_last_valid` policy holds the arm and gripper at the last accepted target on
unconverged rows and continues with the clean seed. Invalid solutions or an arm
with no converged rows still reject the chunk. Metadata records per-arm failed
counts and indices. Step IK retains its separate atomic-hold contract. The default action rate is 30 Hz and horizon is 32.

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
bash scripts/setup_openwam_local_env.sh
```

The checked-in 30k put-bottles deployment is already bound to the local
checkpoint and its artifact hashes. The policy trajectory remains 30 Hz, while
ManiMux interpolates and commands YAM at 100 Hz. Arm and gripper limits are
separate: arms use 0.25 rad/s and 0.5 rad/s^2; normalized grippers use 1.0/s and
12.0/s^2. Its inference schedule is serial: ManiMux executes the committed tail,
then requests the next chunk instead of sampling concurrently.

Start the policy server:

```bash
envs/openwam/.venv/bin/python manimux/servers/openwam.py \
  --config manimux/configs/policy/openwam/yam/finetune-put-bottles-step30000.yaml
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
driver. Dummy mode (`--dummy`) is for non-hardware protocol debugging only;
it cannot generate a bound deployment config.

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

## Local latest300 study (2026-10-20)

`manimux/configs/lists/{openwam-latest300-study}.yaml` (station-local; build it like the [example study](../../../../manimux/configs/lists/example-yam-pi05.yaml)) contains five
tasks with Serial/Async only, 20/32 source steps, compilation and native DiT
cache enabled for default sampling. The policy recipe is
`manimux/configs/policy/openwam/yam/latest300ep-step60000-cache-compile.yaml`;
use `envs/openwam/.venv/bin/python -m manimux.servers.policy --config-list ...`.
Runtime and camera services use the same list with the YAM environment.
Current chunk IK variants explicitly select `hold_last_valid` and use distinct
`_ik_hold` recording identities. The shared YAM arm configuration currently sets `options.pos_threshold: 0.001` (1 mm) for position convergence and acceptance; orientation acceptance and joint limits remain unchanged.
An offline replay accepted a previously rejected prediction with held rows; this
is not physical validation or evidence of task success. Restart the model and
runtime after normal Finish/Home to load these changes; never hot-swap hardware
control to apply a configuration.
