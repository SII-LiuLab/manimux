# Unified training entry

The launcher composes existing data scripts and native XPolicy trainers. It
does not implement model losses, start inference services, allocate cluster
resources, or command a robot. Existing cluster/task scripts remain available.

## Safety and prerequisites

- Run training on a suitably provisioned training machine, not by testing these
  commands on a desktop. Model assets, datasets and compatible environments
  must already exist. No weights are bundled or downloaded by the launcher.
- Without `--execute`, only the command plan is printed. Planning uses Python
  standard-library code: no torch/JAX imports, CUDA probes, hashing of weights,
  dataset conversion, directory creation, installation or subprocess execution.
- `--execute` explicitly opts into costly data processing and GPU training.
  Stages run sequentially; errors stop the pipeline. Logs append under
  `data/training/<policy>/<run>/`. Training output must be new/empty. A lock
  prevents two launcher instances using the same output concurrently.
- Default YAM recipes use one GPU, batch one, zero data workers and one encoder
  thread. These are conservative launch defaults, not a promise that a large
  model fits a laptop or a particular GPU.
- Use a new `run` for new data. Preparation is not silently skipped based on a
  directory's existence. To train data already prepared by the same recipe,
  explicitly select `--phase train`; required artifacts are checked first.
- The unified launcher deliberately does not offer automatic resume. Use the
  policy-specific resume workflow to retain its optimizer/state conventions.

## YAM one-command recipes

All paths passed with `--set` should be absolute. `source` contains complete
native YAM recordings. `pretrained` is a local base-model path. `python` selects
an existing compatible model interpreter; `source` preparation may additionally
require LeRobot/video dependencies in that interpreter.

```bash
python scripts/training/train.py --list

python scripts/training/train.py --recipe pi05-yam \
  --set run=my-task-v1 --set source=/data/raw/my-task \
  --set pretrained=/models/pi05_base/params \
  --set python=/envs/pi05/bin/python --execute
```

This single command converts recordings, computes normalization statistics,
checks their presence, then calls `XPolicyLab/policy/Pi_05/train.sh`. Omit
`--execute` to inspect the same plan without starting anything.

| Recipe | Preparation chain | Native trainer | Extra parameters |
| --- | --- | --- | --- |
| `pi05-yam` | YAM -> LeRobot -> OpenPI statistics | Pi_05/train.sh | None |
| `pi05-yam-joint-ee` | YAM -> LeRobot with recorded EE -> joint+EE statistics | Pi_05/train.sh | Requires recorded observation/command EE arrays |
| `xr1-yam` | YAM -> XR1 JSON/statistics -> bind Hydra data config | Xiaomi_Robotics_1/train.sh | `processor`, `instruction` |
| `lingbot-vla2-yam` | YAM -> LeRobot -> LingBot statistics | LingBot_VLA2/train.sh | `processor` |
| `gr00t-n17-yam` | YAM -> LeRobot -> GR00T v2.1/modality/statistics | GR00T_N17/train.sh | `cosmos`, `converter_env` |
| `openwam-yam` | YAM -> native HDF5 -> OpenWAM statistics | OpenWAM/train.sh | `instruction`; 30 Hz recordings with achieved EE poses |

For example:

```bash
python scripts/training/train.py --recipe xr1-yam \
  --set run=my-task-v1 --set source=/data/raw/my-task \
  --set pretrained=/models/xr1/model_states.pt --set processor=/models/qwen3-processor \
  --set instruction='Assemble the screwdriver.' --set python=/envs/xr1/bin/python --execute

python scripts/training/train.py --recipe openwam-yam \
  --set run=my-task-v1 --set source=/data/raw/my-task \
  --set pretrained=/models/openwam-foundation --set instruction='Assemble the screwdriver.' \
  --set python=/envs/openwam/bin/python --execute
```

Shared parameters: `gpus=0,1`, `steps=3000`, `save_steps=500`, `batch=2`,
`workers=0`, `data_root=/data/prepared`, `output=/data/checkpoints/run`.
Default dataset/checkpoint directories are separated by recipe and run, so
different models cannot accidentally reuse each other's prepared dataset.
Pi05 and GR00T use global `batch` (divisible by GPU count); XR1, LingBot and
OpenWAM use per-GPU batch. Native model defaults still control accumulation.
GR00T needs a preinstalled conversion environment containing LeRobot,
imageio-ffmpeg and a `bin/ffmpeg` executable; this recipe disables automatic
conversion-environment installation. YAM joint+EE LingBot task-specific teacher
recipes remain in the existing cluster scripts; they are not silently mapped
to joint-only training.

## Other XPolicy models

`--list` inventories each top-level policy's `train.sh`, `process_data.sh` and
argument convention. `native_entry` means a script exists, not that all assets
are installed, all robot types work, or training was validated. No YAM support
is inferred from a generic RoboDojo training script.

Use `configs/training/native-policy.example.json` as the schema example:

```bash
python scripts/training/train.py --config /path/to/my-policy-job.json --execute
```

For six-argument adapters, `train` can be omitted: the launcher uses
`bench run robot action seed gpus`. Omit `prepare` to call the policy's
`process_data.sh` with the first four arguments. Configure raw-data roots,
backbones and model-specific options in `env`, following that policy's README.
Adapters that require additional arguments (Mem_0, RISE, etc.) use explicit
`train.args`; EventVLA and Hy_Embodied_05_VLA must always provide their native
arguments. Do not pass the common six arguments to those nonstandard launchers.

Models without `process_data.sh` can use upstream-native prepared data:
declare `prepare: []` and a nonempty `prepared` list of absolute artifact
paths. Alternatively, declare one or more explicit upstream preparation
stages. Every stage supports `entry`, string-array `args`, `cwd`, `requires`
and `produces`. Relative entry/cwd paths are resolved against the policy
directory. `${policy}`, `${workspace}`, `${python}`, `${python_env}` and params
are substituted without shell evaluation. Stages can invoke existing `.py`
or `.sh` files; they cannot contain a shell command string.

| Current category | Policies |
| --- | --- |
| Standard native training entry | A1, ACT, AHA_WAM, Abot_M0, Being_H05, DP, Dexbotic_DM0, DreamZero, FastWAM, G05, GO1, GR00T_N17, GalaxeaVLA, GigaWorldPolicy, H_RDT, InternVLA_A1, LDA_1B, LingBot_VA, LingBot_VLA, LingBot_VLA2, Mem_0, OpenVLA_OFT, OpenWAM, Pi_0, Pi_05, Pi_0_Fast, RDT_1B, RISE, SmolVLA, Spirit_v15, TinyVLA, X_VLA, X_WAM, Xiaomi_Robotics_0, Xiaomi_Robotics_1, starVLA |
| Nonstandard native arguments | EventVLA, Hy_Embodied_05_VLA |
| Entry exists but default training source is absent in this checkout | MolmoACT2 (`molmoact2/lerobot` missing) |
| Template only, rejected | demo_policy |
| No top-level training entry | Other policies listed as `no_entry`; not automatically trainable |

The native route forwards existing scripts faithfully. Some upstream scripts
activate their own environment or invoke package managers; inspect the policy
README/scripts before `--execute`. The runner never calls `install.sh` itself.
GUI legacy entries pointing at absent `third_party` sources (including ABC)
are not a substitute for a complete XPolicy training adapter, and are not
exposed as working YAM recipes here.

## Validation status

The planner unit tests and `--list` inventory were executed locally on
September 13, 2026. No data conversion or training subprocess was started.
Successful GPU training, convergence and checkpoint deployment remain separate
validation steps on a training machine.
