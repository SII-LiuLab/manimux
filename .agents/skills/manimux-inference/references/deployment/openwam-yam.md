# OpenWAM through XPolicyLab

## Scope and layout

OpenWAM follows the XR1/Pi05 split: the policy and trainer live under
`XPolicyLab/policy/OpenWAM`; ManiMux owns robot observations, FK/IK, scheduling,
execution constraints, recording and the robogui. There is no ManiMux OpenWAM
JSON client or standalone OpenWAM service. The vendored upstream deployment
modules remain because the XPolicy adapter uses their loader and preprocessor
in-process, without starting their WebSocket listener.

For an upstream-only environment, install OpenWAM and its XPolicy extras into
an already compatible Python environment:

```bash
bash XPolicyLab/policy/OpenWAM/install.sh
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
malformed poses, invalid grippers, wrong horizons and IK failures reject the
whole chunk. The default action rate is 30 Hz and horizon is 32.

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
  --config manimux/configs/experiments/put_bottles/openwam/yam_openwam_manimux_step30000.yaml \
  --instruction "Put the bottles into the bin."
```

After the probe succeeds, start the camera service and ManiMux runtime:

```bash
envs/yam/.venv/bin/manimux-camera-server --config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml

envs/yam/.venv/bin/manimux serve \
  --config manimux/configs/experiments/put_bottles/openwam/yam_openwam_manimux_step30000.yaml
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
manimux run --config /path/to/deployment/openwam.yaml
```

For RoboDojo simulation, use the policy's standard `eval.sh` and
`setup_eval_*` scripts. YAM real-robot task evaluation uses ManiMux, not an
ARX simulator. Ordinary chunk inference is supported; RTC/AAC/PAINT are not
implemented for OpenWAM. CPU/dummy tests do not establish GPU inference,
training convergence, task success, or real-robot safety.
