# OpenWAM through XPolicyLab

## Scope and layout

OpenWAM follows the XR1/Pi05 split: the policy and trainer live under
`XPolicyLab/policy/OpenWAM`; ManiMux owns robot observations, FK/IK, scheduling,
execution constraints, recording and the viewer. There is no ManiMux OpenWAM
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

## Data and training

Only pass **training episodes** to data preparation; hold out evaluation
episodes in another root. The upstream reader computes statistics over all
tasks in that root. It uses future achieved states as action labels, not the
recorded controller-command arrays. Do not compare these as interchangeable
supervision targets.

Convert complete YAM recordings with recorded EE transforms and RGB videos:

```bash
python scripts/datasets/prepare_openwam_yam_dataset.py \
  --episodes /path/to/training_episodes --output /path/to/openwam_train \
  --task assemble_the_screwdriver --instruction 'Assemble the screwdriver.' \
  --frequency 30
export OPENWAM_DATASET_DIR=/path/to/openwam_train
bash scripts/training/train_openwam_yam_cluster.sh prepare screwdriver-v1
```

This writes native `<root>/<task>/yam_dual/data/episode_*.hdf5`, validates
the reader, builds its real-YAM EEF20 statistics, and inspects a training
sample. Conversion does not resample: confirm the recording really is 30 Hz.
Do not reuse ARX statistics or place validation episodes in the training root.

```bash
export OPENWAM_FINETUNE_CKPT_PATH=/path/to/openwam_foundation_checkpoint
export OPENWAM_GPU_IDS=0,1
bash scripts/training/train_openwam_yam_cluster.sh gate-train screwdriver-v1
```

`prepare`, `smoke`, `train`, `gate-train` mirror the other cluster launchers.
The XPolicy `train.sh` accepts the standard six arguments followed by Hydra
overrides, and invokes the vendored trainer using the selected Python:

```bash
bash XPolicyLab/policy/OpenWAM/train.sh \
  RoboDojo_real screwdriver-v1 yam_dual ee 0 0,1 \
  training.max_steps=3000 training.num_epochs=null
```

Set `OPENWAM_PYTHON` for an existing interpreter, `OPENWAM_OUTPUT_ROOT` for
cluster outputs, and `OPENWAM_RESUME_CKPT_PATH` for full-state resume (unset
`OPENWAM_FINETUNE_CKPT_PATH`). Fresh runs refuse nonempty outputs. Resume needs
upstream optimizer/scheduler/RNG state, not just safetensors. A standalone
deployment bundle contains `config.yaml`, `checkpoint_step_*.safetensors`,
`normalization_stats.npy` and any upstream-required tokenizer/model assets.

Logs append to `data/training/openwam/logs/<run>-<mode>.log`; override the
directory with `OPENWAM_LOG_DIR`. Resume is permitted only in `train` mode:
both training and the post-training artifact check use the resume directory.
`smoke` and `gate-train` reject inherited resume parameters before any work.
No environment installation is performed. Select an existing interpreter with
`OPENWAM_PYTHON`; a missing executable fails before data preparation.

### Put-bottles training recipe

The local task entry (ignored by Git) is `scripts/training/put_bottles_into_the_bin/openwam.sh`.
Set `YAM_TRAIN_ROOT` (or `OPENWAM_TRAIN_ROOT`) to the prepared storage root;
`OPENWAM_WORKSPACE` defaults to this checkout. The recipe selects the 50-episode,
35,118-frame, 30 Hz dataset. It defaults to four GPUs, micro batch 1 and
accumulation 16, giving effective global batch 64. The native schedule remains
30,000 micro steps with saves every 5,000 micro steps; this contains 1,875 full
accumulation windows, not 30,000 optimizer updates.

```bash
bash scripts/training/put_bottles_into_the_bin/openwam.sh plan my-new-run
```

`plan` checks arithmetic and prints paths without model imports, installation
or artifact writes. Data conversion and the native statistics stage are
listed in [the training task guide](../scripts/training/README.md#data-preparation).
For the existing training image, `scripts/training/setup_openwam_env.sh` builds
an isolated environment reusing its CUDA/Torch stack. It ends with `pip check`
and an import check; it does not submit work.

On a prepared machine, `ready` checks the dataset manifest and foundation
hash, prepares native stats, and prints the native command. After the training
resources are allocated, use:

```bash
bash scripts/training/put_bottles_into_the_bin/openwam.sh gate-train my-new-run
```

The smoke covers one accumulation window and checks its checkpoint before the
formal run. This is separate from deployment and task-success validation.

The September 10 historical job used micro batch 1, accumulation 8 and effective
batch 32. Its unchanged request is now local-only at
`.local/training/archive/20260914/put-bottles-4xh200.json`; it records historical
resources and is not a current submission template. New machine profiles and
scheduler requests live under `.local/training/` and call the ignored task script. Include that task folder when synchronizing code.

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
envs/openwam/.venv/bin/python scripts/servers/openwam_yam_server.py \
  --config configs/openwam/yam/server/finetune-put-bottles-step30000.yaml
```

Then validate one real model forward plus the YAM IK boundary without touching
the robot:

```bash
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/openwam/yam/infra/manimux-put-bottles-step30000.yaml \
  --instruction "Put the bottles into the bin."
```

After the probe succeeds, start the camera service and ManiMux runtime:

```bash
envs/yam/.venv/bin/manimux-camera-server --config configs/cameras.yaml

envs/yam/.venv/bin/manimux serve \
  --config configs/openwam/yam/infra/manimux-put-bottles-step30000.yaml
```

```bash
python scripts/servers/openwam_yam_server.py --checkpoint /path/to/yam_run \
  --bind-runtime-config /path/to/deployment/openwam.yaml
python scripts/servers/openwam_yam_server.py \
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
ManiMux runtime for task evaluation, recording and viewer output:

```bash
manimux serve --config /path/to/deployment/openwam.yaml
```

For RoboDojo simulation, use the policy's standard `eval.sh` and
`setup_eval_*` scripts. YAM real-robot task evaluation uses ManiMux, not an
ARX simulator. Ordinary chunk inference is supported; RTC/AAC/PAINT are not
implemented for OpenWAM. CPU/dummy tests do not establish GPU inference,
training convergence, task success, or real-robot safety.
