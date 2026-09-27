# StarVLA integration

StarVLA runs in XPolicyLab's shared policy server. ManiMux connects through
`xpolicylab_ws` and uses its existing adapters, schedulers, executors and recorder.
The checked-in examples use synthetic cameras and an offline test robot.

## Choose a checkpoint and experiment

Paths below are relative to
[`manimux/configs/experiments/offline/starvla/`](../manimux/configs/experiments/offline/starvla/).
Use a checkpoint that matches the complete contract, including camera order,
state input, normalization, action meaning and embodiment.

| Experiment | Required checkpoint contract |
| --- | --- |
| `aloha_oft_serial.yaml` | RoboTwin few-shot QwenOFT; three RGB views, no state, 50 × 14 absolute joints |
| `arx_pi_v3_serial.yaml` | RoboDojo PI-v3; three RGB views, 14D joint state, 50 × 14 absolute joints |
| `arx_pi_v3_manimux.yaml` | Same PI-v3 checkpoint, asynchronous scheduling |
| `arx_pi_v3_act_temporal_ensemble.yaml` | Same PI-v3 checkpoint, temporal ensembling |
| `arx_pi_v3_{rtc,paint,aac,autohorizon,dvac}.yaml` | Same PI-v3 checkpoint, with the corresponding flow sampler enabled |
| `libero_groot_serial.yaml` | LIBERO GR00T; one RGB view, no state, 8 × 7 native feedback EEF deltas |
| `libero_fast_serial.yaml` | LIBERO FAST; same EEF contract, with its action-token VLM and FAST processor |

`yam_eef_contract.yaml` is a **synthetic interface test template**, not a validated
checkpoint recipe. It covers dual-arm EEF state, 50 predicted poses, a 20-step
IK prefix and real offline YAM geometry. A matching real checkpoint is still
needed. Do not substitute LIBERO deltas or a joint checkpoint for this contract.

## Install dependencies and prepare assets

Use separate environments:

- **ManiMux:** Python 3.11 or 3.12, the repository's runtime/test dependencies,
  and `python -m pip install -e '.[xpolicylab,replay]'`.
- **StarVLA:** follow [the XPolicyLab adapter installation](../XPolicyLab/policy/starVLA/README.md#installation).
  Model dependencies stay in this environment; the tested version is Python 3.10.

Download weights and assets under the ignored `checkpoints/` directory. Public
checkpoint sources and revisions are in the [validation report](starvla-validation.md#assets).
Retain the original run layout:

```text
run/
├── config.yaml
├── config.full.yaml
├── dataset_statistics.json
└── checkpoints/steps_30000_pytorch_model.pt
```

The base VLM must match the checkpoint. FAST also requires its matching action
processor. Missing assets or ambiguous state-input metadata cause an error;
the adapter does not select a substitute model or silently disable state input.

## Configure and start the shared server

Run commands from the ManiMux repository root. In the **ManiMux environment**:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p .local/starvla
cp manimux/configs/experiments/offline/starvla/station.example.yaml .local/starvla/station.yaml
```

Edit `paths.checkpoint` in that private file to point to the weight file, and
set `services.policy.endpoint` to the server address. Resolve the paired server
configuration without loading weights:

```bash
python -m manimux.servers.starvla \
  --experiment manimux/configs/experiments/offline/starvla/arx_pi_v3_serial.yaml \
  --local .local/starvla/station.yaml --check > .local/starvla/server.json
```

In the **StarVLA environment**, from the same repository root:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
export STARVLA_BASE_VLM=/path/to/Qwen3-VL-4B-Instruct
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m manimux.servers.starvla --config .local/starvla/server.json
```

Select a device with enough free GPU memory. For FAST, use its
`Qwen2.5-VL-3B-Instruct-Action` VLM and set
`STARVLA_FAST_TOKENIZER=/path/to/fast_tokenizer`. Resolve a new server JSON when
changing experiment, checkpoint or sampling mode. The separate resolution step
keeps Python 3.11 runtime imports out of the Python 3.10 model process.

## Validate the deployment

Wait for the server to report that it is listening. In the **ManiMux environment**:

```bash
python scripts/validation/starvla_offline_probe.py \
  --experiment manimux/configs/experiments/offline/starvla/arx_pi_v3_serial.yaml \
  --local .local/starvla/station.yaml \
  --runtime --output .local/starvla/pi-serial.json
```

The probe checks server identity, four requests including reset and changed RGB,
finite actions, action decoding, and optionally the complete runtime. It accepts
only test robot and sensor factories. JSON reports, NumPy arrays and recordings
are written beside the selected output path.

| Option | Purpose |
| --- | --- |
| Omit `--runtime` | Check transport and adapter behavior only |
| `--runtime --executor smooth` or `mpc` | Exercise another existing executor |
| `--deterministic` | Require repeated input and reset to reproduce the same actions; use for OFT, not stochastic flow heads |
| `--sampling-matrix` | Exercise all five flow request types against an advanced PI-v3 server; AAC uses synthetic selector calibration |

Run probes sequentially against a server because reset affects the served model
instance. A successful offline rollout proves completion of the configured test,
not manipulation task success. Stop the temporary server after validation.

## Action and sampling contracts

Joint recipes return standard per-arm joint groups. The OFT RoboTwin few-shot
recipe explicitly permutes native `[left joints, right joints, left gripper,
right gripper]` into the standard interleaved arm/gripper order. It preserves
min/max normalization, the binary gripper threshold `> 0.49`, and a 30 Hz action
interval. Equal action dimensions do not establish compatibility between robots.

`PoseAdapter` consumes canonical rows `[x, y, z, qw, qx, qy, qz, gripper]`:

| `action_semantics` | Interpretation |
| --- | --- |
| `absolute_per_arm_base_xyz_wxyz` | Absolute target in each arm's base frame |
| `delta_observation_base_xyz_wxyz` | Base-frame translation and left-multiplied rotation, anchored to the request's measured pose |
| `delta_observation_tool_xyz_wxyz` | Transform right-multiplied onto the request's measured pose |
| `delta_step_base_xyz_wxyz` | One feedback command from measured state, decoded inline before replanning |

LIBERO recipes clip the native controller input, apply translation scale 0.05
and rotation-vector scale 0.5, and threshold the gripper. They retain eight
predictions on the wire but decode only the first feedback delta, held for one
50 ms action interval. This follows the
[LIBERO wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py)
and robosuite 1.4.1's [OSC configuration](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/controllers/config/osc_pose.json).
The analytic test plant does not reproduce LIBERO physics or Franka dynamics.

For absolute and observation-anchored EEF chunks:

- `decode_policy_steps` selects the prefix before IK.
- `policy.action_decoding: process` enables the existing decoder process;
  `policy.adapter.parallel_ik: true` partitions work by arm.
- Optional `ik_warmup_joints` initializes offline FK/IK before control starts.
  It does not command a robot or replace measured state.
- An invalid pose or failed IK partition rejects the whole chunk.

OFT and FAST expose default inference. GR00T and PI-v3 flow heads can explicitly
enable RTC, PAINT, AAC, AutoHorizon and DVAC for **joint chunks**. EEF conditioning
is not implemented; the existing Pi05 YAM EEF adapter also rejects RTC conditions.
Backend identity and capability checks must remain enabled.

## Development and contribution

| Location | Responsibility |
| --- | --- |
| `XPolicyLab/policy/starVLA/` | Checkpoint loading, observation conversion, normalization, model sampling and standard action output |
| `manimux/policy_adapter/pose.py` | Robot action semantics and FK/IK |
| `manimux/configs/policy/starvla/` | Checkpoint contracts and inference settings |
| `manimux/configs/experiments/offline/starvla/` | Paired runtime examples |
| `scripts/validation/starvla_offline_probe.py` | Real-server validation using synthetic observations |

See [tests and validation scope](starvla-validation.md) before making support
claims. Publish XPolicyLab changes first; update the parent submodule reference
only to a reachable revision. Keep weights, private station files and generated
evidence out of commits.
