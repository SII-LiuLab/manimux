# Shared launchers and local training tasks

The top-level scripts are reusable model launchers. Task directories contain
our own dataset paths, experiment settings and thin model-specific commands.
All task subdirectories are ignored by Git except the portable `example/`.
Model implementations remain in `XPolicyLab/policy/<POLICY>/`; local scheduler
commands remain under `.local/training/`.

```text
scripts/training/
├── train_pi05_yam_cluster.sh
├── train_lingbot_vla2_yam_cluster.sh
├── train_xr1_yam_cluster.sh
├── train_gr00t_n17_yam_cluster.sh
├── train_openwam_yam_cluster.sh
├── _batch.sh / setup_openwam_env.sh
├── example/                         # tracked template
├── assemble_screwdriver/             # ignored task settings
│   ├── paths.env
│   ├── pi05.sh / pi05_joint_ee.sh
│   ├── lingbot_vla2.sh / lingbot_vla2_joint_ee.sh
│   └── xr1.sh / gr00t_n17.sh
├── put_bottles_into_the_bin/          # ignored task settings
│   ├── paths.env
│   ├── pi05_joint.sh / pi05_joint_ee.sh
│   └── lingbot_vla2.sh / xr1.sh / openwam.sh
└── pick_red_ball_box/                # ignored task settings
    └── paths.env / gr00t_n17.sh
```

Each task has a `paths.env` for storage roots and a script per model/variant
for dataset IDs, native batch knobs, steps and output names. Task scripts
source their sibling `paths.env`, then call the shared model wrapper. The
shared wrappers require task/dataset parameters instead of selecting an
unrelated task implicitly. The optional `train.py` planner is also retained.

## Task workflow

```bash
# Existing local task: inspect paths/batch only, without installing or running models.
bash scripts/training/put_bottles_into_the_bin/pi05_joint.sh plan my-new-run
bash scripts/training/put_bottles_into_the_bin/pi05_joint_ee.sh plan my-new-run

# On the training machine after preparing source data, assets and environment:
bash scripts/training/put_bottles_into_the_bin/pi05_joint.sh prepare my-new-run
bash scripts/training/put_bottles_into_the_bin/pi05_joint.sh train my-new-run

# New local task: copy the tracked example, then edit paths.env and model settings.
cp -r scripts/training/example scripts/training/my_task
bash scripts/training/my_task/pi05.sh plan my-new-run
```

The ignored task folders are available in this working tree, but a colleague's
fresh Git checkout only includes the shared scripts and `example/`. Share/copy
the needed task folder explicitly. `git pull` or a sync using Git-ignore rules
will not transfer those folders. Remote training needs the shared scripts,
the selected task folder and matching XPolicyLab source. It does not need local
scheduler submitters.

Set `YAM_TRAIN_ROOT` to relocate a task's storage (or edit its `paths.env`).
Bottles preserves the historical Pi05/OpenWAM hdd3 profiles and LingBot/XR1 hdd2
profiles through per-model root overrides. Those paths are local task data,
not public launcher defaults. See each task's README for dataset IDs and stages.

Choose a fresh run name. Pi05/OpenWAM append `-smoke` in smoke mode; the other
models use the supplied name. `gate-train`, where supported, runs the native
smoke/checkpoint gate before formal training. Model-specific smoke checks do
not establish deployment or real-robot task success.

## Native batch parameters

| Model | Parameters | Default calculation |
| --- | --- | --- |
| Pi05 | `OPENPI_BATCH_SIZE` | Global 64; do not multiply by GPU count or FSDP shards again |
| LingBot | `LINGBOT_VLA2_MICRO_BATCH_SIZE`, `LINGBOT_VLA2_GRAD_ACCUM_STEPS`; optional `LINGBOT_VLA2_GLOBAL_BATCH_SIZE` assertion | 1 × 8 GPUs × 8 accumulation = 64 |
| XR1 | `XR1_MICRO_BATCH_SIZE`, `XR1_GRAD_ACCUM_STEPS` | 1 × 8 GPUs × 8 accumulation = 64 |
| GR00T | `GR00T_GLOBAL_BATCH_SIZE` → native `GLOBAL_BATCH_SIZE` | Global 64, 16 per GPU on 4 GPUs |
| OpenWAM | `OPENWAM_BATCH_SIZE`, `OPENWAM_GRADIENT_ACCUMULATION_STEPS` | 1 × 4 GPUs × 16 accumulation = 64 |

GPU lists use each model's `*_GPU_IDS` variable. If accumulation is omitted,
the shared launcher derives it from the native micro batch and GPU count; an
explicit inconsistent value fails before preparation or training. XR1 forwards
the micro batch to Hydra even when reusing an older converted data config.
These recipes are single-node data-parallel/FSDP jobs. Model-parallel or
multi-node experiments need their own native topology calculation.

`plan` and formal runs check against `YAM_EXPECTED_GLOBAL_BATCH_SIZE=64` by
default. To reproduce a historical batch-32 run, explicitly set both that
expectation and the model's native parameters. A smoke may explicitly use a
smaller batch. The shared `_batch.sh` performs arithmetic, not model training.

OpenWAM's step counter counts micro steps. At accumulation 16, 30000 micro
steps contain 1875 complete accumulation windows; this is not 30000 optimizer
updates. Its smoke now spans one complete accumulation window. The September
2026 bottles checkpoints used batch 32 (Pi05 and OpenWAM); changing defaults
does not change those historical experiments or their saved configurations.

## Data preparation

Task `prepare` modes do **not** all start from raw recordings. Follow the
model-specific chain below. Select a Python environment with the required data
dependencies. Existing conversion tools remain reusable; no conversion is
implicitly moved into the hardware runtime.

| Model | Raw recordings to native dataset | What the task's `prepare` then does |
| --- | --- | --- |
| Pi05 | `scripts/datasets/convert_yam_to_lerobot.py` → LeRobot v3 | Computes OpenPI normalization stats matching `pi05_yam` or `pi05_yam_joint_ee`; validates dimensions/counts |
| LingBot | Same LeRobot converter | Resolves native depth paths/config; computes native LingBot stats using the selected relative/absolute robot transform |
| XR1 | `scripts/datasets/prepare_xr1_yam_dataset.py` → JSON + `norm_stats.json` + Hydra YAML | Runs this conversion if missing, using `XR1_YAM_EPISODES`; otherwise binds the YAML recorded in the manifest |
| GR00T | Same LeRobot converter | Calls `XPolicyLab/policy/GR00T_N17/process_data.sh` for v2.1 conversion, modality and stats |
| OpenWAM | `scripts/datasets/prepare_openwam_yam_dataset.py` → native HDF5 | Calls `XPolicyLab/policy/OpenWAM/process_data.sh` for native stats/validation |

For example, prepare the common bottles LeRobot dataset used by Pi05 and
LingBot. The joint/EEF recipes require the recorded EE columns:

```bash
"$YAM_TRAIN_ROOT/envs/pi05/.venv/bin/python" scripts/datasets/convert_yam_to_lerobot.py \
  "$YAM_TRAIN_ROOT/datasets/raw/put_bottles_into_the_bin" \
  --repo-id yam_put_bottles_into_the_bin_20260905_joint_ee_v1 \
  --output-root "$YAM_TRAIN_ROOT/datasets/lerobot/yam_put_bottles_into_the_bin_20260905_joint_ee_v1" \
  --include-ee-pose --streaming-encoding --encoder-threads 1
bash scripts/training/put_bottles_into_the_bin/pi05_joint.sh prepare
bash scripts/training/put_bottles_into_the_bin/pi05_joint_ee.sh prepare
```

For ordinary screwdriver joint training, use source `assemble_the_screwdriver`
and dataset ID `yam_assemble_screwdriver_20260825_v1`; omit `--include-ee-pose`.
For the separate LingBot joint/EEF variant, retain EE columns and use its
`yam_assemble_screwdriver_20260825_v1_joint_ee` dataset ID.

OpenWAM has a different native dataset and action target:

```bash
"$YAM_TRAIN_ROOT/envs/openwam/.venv/bin/python" scripts/datasets/prepare_openwam_yam_dataset.py \
  --episodes "$YAM_TRAIN_ROOT/datasets/raw/put_bottles_into_the_bin" \
  --output "$YAM_TRAIN_ROOT/datasets/openwam/yam_put_bottles_into_the_bin_20260905_v1" \
  --task put_bottles_into_the_bin --instruction "Put the bottles into the bin." \
  --frequency 30 --expected-episodes 50 --expected-frames 35118
bash scripts/training/put_bottles_into_the_bin/openwam.sh prepare
```

Pi05 and LingBot preparation may use their model environments and native
distributed statistics utilities; they are not all guaranteed CPU-only.
XR1 preserves recorded commanded/achieved EEF transforms. OpenWAM uses future
achieved EEF states. Do not interchange their converted datasets or statistics.

## Optional planner and local submission

`train.py` / `XPolicyLab/training/` remain an optional development planner.
Its built-in generic recipes retain their own development defaults; they do
not automatically use the settings in your ignored task folder. A task does not have to use that planner.
See [planner documentation](../../docs/training-entrypoints.md).

Edit locally, validate the scripts, synchronize shared scripts, the selected ignored task directory and
XPolicyLab and verify versions/hashes, prepare data/environments on the remote
storage, then use the local private submitter. The CPU notebook provides file
access/preparation; the scheduler allocates GPU jobs. Remote job commands call
the selected task script. Machine identity, resources and scheduler requests stay
in `.local/training/`; they are not required to understand the model recipe.
