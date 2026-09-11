# Pi05 bottles joint-only / joint + EEF preparation

This profile uses all 50 episodes of `put_bottles_into_the_bin` (35,118 frames,
30 Hz). It does not create a QZ job or select a compute project. The `hdd3`
preparation Notebook is not evidence that the original `intern-ziyang` training
workers can mount these paths; confirm the actual training node's mounts before
submission.

## Artifacts

```bash
export YAM_TRAIN_ROOT=/inspire/hdd3/project/agent-driven-world-model/ky26300/manimux_training
export PI05_WORKSPACE="$YAM_TRAIN_ROOT/operate/manimux"
export OPENPI_LEROBOT_REPO_ID=yam_put_bottles_into_the_bin_20260905_joint_ee_v1
export OPENPI_ASSETS_BASE_DIR="$YAM_TRAIN_ROOT/assets"
```

Raw source (unchanged):
`/inspire/hdd3/project/agent-driven-world-model/public/sa/teleop/put_bottles_into_the_bin`.
The prepared dataset lives under
`$YAM_TRAIN_ROOT/datasets/lerobot/$OPENPI_LEROBOT_REPO_ID`.
Statistics live under
`$OPENPI_ASSETS_BASE_DIR/pi05_yam_joint_ee/$OPENPI_LEROBOT_REPO_ID/norm_stats.json`.

## Conversion

Use the dedicated data environment, not the SAPolicy environment. The output
must not already exist; do not overwrite an earlier conversion.

```bash
"$YAM_TRAIN_ROOT/envs/data/bin/python" \
  "$PI05_WORKSPACE/scripts/datasets/convert_yam_to_lerobot.py" \
  /inspire/hdd3/project/agent-driven-world-model/public/sa/teleop/put_bottles_into_the_bin \
  --repo-id "$OPENPI_LEROBOT_REPO_ID" \
  --output-root "$YAM_TRAIN_ROOT/datasets/lerobot/$OPENPI_LEROBOT_REPO_ID" \
  --include-ee-pose --video-codec h264 --streaming-encoding --encoder-threads 2
```

The three cameras remain separate RGB streams. Joint/gripper state and action
columns are 14D. Observation and action EE poses are 24D (per arm: xyz in meters
and a row-major 3x3 rotation matrix). Each arm's poses use its own base frame;
they are not a calibrated shared world frame.

## Normalization

```bash
"$YAM_TRAIN_ROOT/envs/data/bin/python" \
  "$PI05_WORKSPACE/scripts/datasets/compute_pi05_joint_ee_norm_stats.py" \
  "$YAM_TRAIN_ROOT/datasets/lerobot/$OPENPI_LEROBOT_REPO_ID" \
  --openpi-root "$PI05_WORKSPACE/XPolicyLab/policy/Pi_05/openpi" \
  --output-dir "$OPENPI_ASSETS_BASE_DIR/pi05_yam_joint_ee/$OPENPI_LEROBOT_REPO_ID" \
  --horizon 50
```

This reads numeric parquet columns without decoding images, applies the same
joint deltas and current-EE-frame translation/rotation-vector conversion as
Pi05, and uses OpenPI's `RunningStats` for mean, std, q01 and q99. Unlike the
default normalization loader's incomplete-batch dropping, every frame anchor
is included. Future targets repeat the final action at episode boundaries.
There are 35,118 state samples and 1,755,900 action-step samples. The result has
14D state and 26D actions: 14 joint/gripper dimensions plus 12 EEF dimensions.
Grippers remain absolute in [0,1]. Inference still exposes only 14D joint/gripper
actions; EEF dimensions are auxiliary supervision.

For the joint-only comparison, use the **same dataset** and compute separate
14D action statistics. No additional conversion or removal of EE columns is needed:

```bash
"$YAM_TRAIN_ROOT/envs/data/bin/python" \
  "$PI05_WORKSPACE/scripts/datasets/compute_pi05_joint_ee_norm_stats.py" \
  "$YAM_TRAIN_ROOT/datasets/lerobot/$OPENPI_LEROBOT_REPO_ID" \
  --openpi-root "$PI05_WORKSPACE/XPolicyLab/policy/Pi_05/openpi" \
  --output-dir "$OPENPI_ASSETS_BASE_DIR/pi05_yam/$OPENPI_LEROBOT_REPO_ID" \
  --horizon 50 --joint-only
```

The joint-only config is `pi05_yam`; the auxiliary config is `pi05_yam_joint_ee`.
Both use the same 14D state, three cameras, horizon 50, base weights, and seed 0.
The model's 32D action input/output is unchanged: joint-only targets are padded
from 14D to 32D, whereas joint+EEF targets are padded from 26D to 32D.

## Training entry point (not submitted by preparation)

The conversion environment is **not** the Pi05 GPU training environment.
The dedicated training environment and official Pi05 base checkpoint are staged
separately below. On the approved GPU node, verify these paths are mounted:

```bash
export OPENPI_ENV_DIR="$YAM_TRAIN_ROOT/envs/pi05/.venv"
export OPENPI_BASE_PARAMS="$YAM_TRAIN_ROOT/weights/base/pi05_base/params"
export OPENPI_GPU_IDS=0,1,2,3
export OPENPI_FSDP_DEVICES=4
export OPENPI_BATCH_SIZE=32
export OPENPI_NUM_TRAIN_STEPS=15000
export OPENPI_SAVE_INTERVAL=500
export OPENPI_MAX_TO_KEEP=10

bash "$PI05_WORKSPACE/scripts/training/train_pi05_yam_bottles_joint_ee_cluster.sh" \
  prepare put-bottles-joint-ee-v1-s0-15k

# Run only after the compute project, GPU resources and job payload are approved:
bash "$PI05_WORKSPACE/scripts/training/train_pi05_yam_bottles_joint_ee_cluster.sh" \
  train put-bottles-joint-ee-v1-s0-15k

# Separate job, with the same resource and hyperparameter settings:
bash "$PI05_WORKSPACE/scripts/training/train_pi05_yam_bottles_joint_cluster.sh" \
  prepare put-bottles-joint-v1-s0-15k
bash "$PI05_WORKSPACE/scripts/training/train_pi05_yam_bottles_joint_cluster.sh" \
  train put-bottles-joint-v1-s0-15k
```

The four-GPU/batch-32 values above are a configurable launch example, not an
allocated resource request. The wrapper checks the new dataset's exact frame
and episode counts. Checkpoints go to
`$YAM_TRAIN_ROOT/weights/finetuned/pi05/<run_name>`; existing run directories are
not overwritten. Preparation logs are under `$YAM_TRAIN_ROOT/runs/preparation`.

The training source is pinned to XPolicyLab
`e78d1bfccdf00fd8bda219a8613632d7f954866b`. Training dependencies come from its
OpenPI lockfile; when the official wheel host is unreachable, a mirror is used
with the same pinned versions and mandatory lockfile hashes. Base parameters are
staged independently. Minimal dependency installs must also include the locked
`pytest`, `iniconfig`, and `pluggy` packages: OpenPI's model import chain imports
`pytest` even for JAX training. Base parameters are
downloaded from `gs://openpi-assets/checkpoints/pi05_base/params`, checked against
each object's generation, size and official CRC32C (plus MD5 where available), and published only after all
files pass. `download_manifest.json` also records SHA-256 per file.

Preparation validation covers all numeric rows against the raw source, all
35,118 video frames per camera, and LeRobot reads at both ends of each episode.
The separate OpenPI training-batch check runs on **CPU**, not GPU, and does not
perform model inference, backpropagation or an optimizer step. It is not evidence
of GPU training throughput or task success.
