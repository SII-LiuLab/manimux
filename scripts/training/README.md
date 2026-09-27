# YAM training launchers

These launchers orchestrate the pinned implementations under `XPolicyLab/policy/`.
Keep private task recipes and machine configuration in the ignored root `training/`
directory. Set `YAM_TRAIN_ROOT` to the shared storage root containing `datasets/`,
`envs/`, `weights/`, `cache/` and `runs/`. Source code can live separately under
`operate/`. No launcher requires an HDD2 path.

Initialize the model source before use:

```bash
git submodule update --init XPolicyLab
git -C XPolicyLab submodule update --init policy/LingBot_VLA2/lingbot_vla_v2
python3 scripts/training/train.py --list
```

Each shell launcher accepts `plan RUN_NAME` to print its configuration without
installing dependencies, converting data or starting training. Required variables:

- Pi05 (`train_pi05_yam_cluster.sh`): `OPENPI_LEROBOT_REPO_ID`, `OPENPI_TASK_NAME`.
- XR1 (`train_xr1_yam_cluster.sh`): `XR1_DATASET_PATH`, `XR1_DATA_CONFIG_NAME`,
  `XR1_TASK_NAME`, `XR1_INSTRUCTION`. Set `XR1_YAM_EPISODES` when conversion is needed.
- LingBot (`train_lingbot_vla2_yam_cluster.sh`): `LINGBOT_VLA2_DATASET_PATH`.
- GR00T N1.7 (`train_gr00t_n17_yam_cluster.sh`): `GR00T_TASK_NAME`,
  `GR00T_SRC_DATASET` (a dataset name beneath `$YAM_TRAIN_ROOT/datasets/lerobot`).

For example, after sourcing a private task recipe:

```bash
bash scripts/training/train_xr1_yam_cluster.sh plan my-run
```

The default global batch is 64. XR1 and LingBot derive gradient accumulation from
the selected GPU count and micro batch. Use the model's GPU and batch environment
variables to configure a reviewed run. A successful plan checks configuration
arithmetic only; it does not prove that dependencies, checkpoints, converted data
or GPU execution are ready. Modes other than `plan` can prepare artifacts or run
training; inspect the selected launcher's mode handling before executing them.

Pi05, LingBot and GR00T require their corresponding converted LeRobot datasets.
XR1 converts recorded YAM end-effector poses into JSON and computes normalization
statistics. Its native dataset sampler retains the existing end-effector-relative
delta action convention. These launchers do not change action axes or semantics.

The unified `train.py` entry point delegates to `XPolicyLab.training.launch` and
defaults to a plan; `--execute` is required to execute its selected recipe. Model
environments and base weights are separate artifacts and are not included in Git.

For recorded YAM XR1 data, the standard preparation entry now lives in
`XPolicyLab/policy/Xiaomi_Robotics_1/process_data.sh` with `XR1_SOURCE_FORMAT=yam`.
The `xr1-yam` recipe calls it directly. The old shell launcher and dataset CLI
remain available for compatibility; conversion implementation belongs to the
XPolicyLab policy. See its README for the source/output/config environment fields.
