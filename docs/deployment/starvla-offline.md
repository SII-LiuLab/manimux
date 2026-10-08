# StarVLA offline integration

StarVLA is a separately versioned policy framework alongside XPolicyLab. Its native
WebSocket service loads the checkpoint and runs the model; ManiMux selects
`worker: starvla_ws` and reuses its adapters, runtime, executors and recorder.
These examples use synthetic images and an offline plant. They never open a robot
or camera connection. See [validation and limits](starvla-offline.md).

```text
StarVLA/ native model service
    ↕ WebSocket / StarVLA MessagePack
manimux/policies/starvla/     PolicyModel client and wire conversion
    → JointAdapter / PoseAdapter
    → existing inference strategy → executor → configured robot
```

Use the steps below to select a recipe, launch its service and validate the paired
runtime. For code review, see [Extension points](#extension-points). The native
[server guide](../../StarVLA/deployment/model_server/README.md#configured-native-policy-serving)
defines the wire protocol.

## Select a matching recipe

Experiments are under `manimux/configs/experiments/offline/starvla/`.

| Experiment | Checkpoint contract |
| --- | --- |
| `aloha_oft_serial.yaml` | RoboTwin QwenOFT, three RGB views, no state, 50 × 14 absolute joints |
| `arx_pi_v3_serial.yaml` | RoboDojo QwenPI_v3, three RGB views, 14D joint state, 50 × 14 absolute joints |
| `arx_pi_v3_manimux.yaml` | Same PI-v3 recipe, asynchronous scheduling |
| `arx_pi_v3_act_temporal_ensemble.yaml` | Same PI-v3 recipe, temporal ensembling |
| `arx_pi_v3_{rtc,paint,aac,autohorizon,dvac}.yaml` | Same PI-v3 recipe, the selected flow sampling mode |
| `libero_groot_serial.yaml` | LIBERO QwenGR00T, one RGB view, no state, 8 × 7 native EEF feedback deltas |
| `libero_fast_serial.yaml` | LIBERO QwenFast, same EEF contract, matching action-token VLM and FAST processor |

`yam_eef_contract.yaml` describes a synthetic dual-arm EEF contract: 16D pose state,
50 predicted poses, a 20-step IK prefix and real offline YAM geometry. It needs a
matching checkpoint to serve real predictions. It is not interchangeable with the
LIBERO or joint recipes.

## Install and prepare assets

From a source checkout:

```bash
git submodule update --init StarVLA
```

Use separate Python environments:

- **ManiMux, Python 3.11–3.12:** `python -m pip install -e '.[starvla,replay]'`.
  The client requires `websockets>=15` for synchronous connection keepalive options.
  YAM FK/IK examples also need the YAM geometry dependencies described in
  [Python environments](../usage/environments.md).
- **StarVLA, Python 3.10:** follow its [installation guide](../../StarVLA/docs/starVLA_guideline.md#0-installation).
  The model process additionally needs `websockets>=14`, `msgpack`, `PyYAML`,
  `opencv-python` and the checkpoint's model dependencies. It does not import ManiMux.

Keep downloaded weights under the ignored `checkpoints/pretrained/starvla/` directory.
Preserve the
checkpoint's original `config.yaml` and `dataset_statistics.json`:

```text
run/
├── config.yaml
├── dataset_statistics.json
└── checkpoints/steps_30000_pytorch_model.pt
```

Use the matching base VLM; FAST needs its action-token VLM and FAST processor.
The server restores the training transforms from the registered data schema.
Missing assets, incompatible dimensions and unsupported sampling modes fail explicitly.

Configuration has three owners:

| File | What belongs there |
| --- | --- |
| `configs/policy/starvla/<embodiment>/<task>/*.yaml` | Checkpoint observation/action contract, training normalization key and enabled samplers |
| `configs/experiments/offline/starvla/*.yaml` | Client, robot groups, adapter, scheduling, executor and recording |
| `.local/starvla/station.yaml` | This installation's checkpoint path and service endpoint |

The first two paths are relative to `manimux/`. A recipe's `backend_identity`
is resolved into the client's expected metadata; keep these paired through the
existing config loader.

## Resolve and launch

In the **ManiMux environment**, from the repository root:

```bash
mkdir -p .local/starvla
cp -n manimux/configs/experiments/offline/starvla/station_example.yaml .local/starvla/station.yaml
```

Set `paths.checkpoint` and `services.policy.endpoint` in that private file. Then:

```bash
python -m manimux.servers.starvla \
  --experiment manimux/configs/experiments/offline/starvla/arx_pi_v3_serial.yaml \
  --local .local/starvla/station.yaml --check > .local/starvla/server.json
```

If assets are in another directory, add explicit `config_overrides` to the resolved
JSON, for example:

```json
{
  "config_overrides": [
    "framework.qwenvl.base_vlm=/path/to/Qwen3-VL-4B-Instruct"
  ]
}
```

Merge this field into `server.json`, retaining its resolved deployment fields.

For FAST, also set `framework.action_model.fast_tokenizer_path=/path/to/fast_tokenizer`.
Use its matching `Qwen2.5-VL-3B-Instruct-Action` base VLM. Overrides must preserve the
checkpoint architecture and training contract; do not change action dimensions to
make an incompatible checkpoint load.

In the **StarVLA environment**, from the same repository root:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONPATH="$PWD/StarVLA" python -m deployment.model_server.serve \
  --config .local/starvla/server.json
```

The selected GPU must have enough free memory. Resolve a new server configuration
when changing the checkpoint or deployment contract. Resolution and model loading
use separate processes so Python 3.11 runtime dependencies do not enter Python 3.10.
The native [serving contract](../../StarVLA/deployment/model_server/README.md) describes
metadata, sessions and sampler requests.

## Validate without hardware

After the server reports that it is listening, run in the **ManiMux environment**:

```bash
python -m scripts.validation.starvla_offline_probe \
  --experiment manimux/configs/experiments/offline/starvla/arx_pi_v3_serial.yaml \
  --local .local/starvla/station.yaml \
  --runtime --output .local/starvla/pi-serial.json
```

The probe checks identity, dimensions, reset, changed RGB, finite actions and
adapter conversion. `--runtime` also runs the worker, scheduler, executor and
recorder for 500 control steps. Its robot and sensor factories are restricted to
the included offline fixtures.

| Option | Effect |
| --- | --- |
| Omit `--runtime` | Check inference and action conversion only |
| `--runtime --executor smooth` or `mpc` | Exercise another existing executor |
| `--deterministic` | Require identical repeated/reset predictions; use for OFT or deterministic FAST |
| `--sampling-matrix` | Exercise all five specialized joint sampling requests on the PI-v3 server |

Outputs include a JSON report, decoded NumPy arrays, and optional rollout recordings.
The synthetic AAC example uses offline YAM FK and fixture statistics to test the
selector interface with 14D outputs. It does **not** validate ARX geometry or a
calibrated physical AAC metric. Use the actual robot geometry and dataset statistics
for a deployment. Run probes sequentially to avoid competing for GPU resources;
reset state is isolated per WebSocket connection.

Saved canonical joint trajectories use the same [action replay](../usage/replay.md)
and [recording workflow](../usage/records.md) as other policies. No model service is
needed to replay a recording. Stop the temporary server after validation.

## Action semantics and capability limits

Camera names and order are explicit. Images are HWC `uint8` RGB; server resizing
uses the configured `[width, height]`. State stays in raw environment units until
the server applies the checkpoint's training transform. Action and state permutations
are independent; normalized sampler conditions also pass through the training transform.

The OFT recipe converts native `[left joints, right joints, left gripper, right
gripper]` into interleaved arm/gripper groups. Its training transform retains
min/max joint normalization and the 0.49 binary gripper threshold. RoboDojo PI-v3
uses its interleaved q99 transform. Both retain a 30 Hz action interval.

`PoseAdapter` consumes `[x, y, z, qw, qx, qy, qz, gripper]` per arm:

| Semantics | Interpretation |
| --- | --- |
| `absolute_per_arm_base_xyz_wxyz` | Absolute target in each arm's base frame |
| `delta_observation_base_xyz_wxyz` | Translation and left-multiplied rotation relative to the request observation |
| `delta_observation_tool_xyz_wxyz` | Transform right-multiplied onto the request observation pose |
| `delta_step_base_xyz_wxyz` | One feedback delta from measured state, decoded inline before replanning |

LIBERO recipes preserve eight predictions on the wire, clip native controller
inputs, scale translation by 0.05 and rotation vectors by 0.5, and threshold the
gripper. Only the first feedback delta is decoded and held for a 50 ms action
interval. The analytic plant checks this contract, not LIBERO physics or Franka dynamics.

OFT/FAST expose default inference only. QwenGR00T and QwenPI_v3 have actual flow
sampler hooks; enabled modes are checked against the loaded head. The PI-v3 recipe
requires canonical DiT forwarding. AutoHorizon requires self-attention and at
least three denoising steps; PI-v3 legacy forwarding does not advertise it. DVAC
requires at least two. The tested GR00T EEF recipe exposes default inference only. Conditioned EEF sampling is unsupported, as in the
existing Pi05 YAM EEF path. Framework integration does not imply every architecture,
checkpoint, embodiment and sampler combination is supported.

## Troubleshooting

An inference error returned in a valid RPC response rejects that observation and
preserves the connection and episode for the next request. Transport failures or
invalid response envelopes close the connection and require an explicit reset
before inference can resume. The client never silently reconnects or resets
sampler history after a failure.

| Failure | Check |
| --- | --- |
| Backend identity mismatch | Resolve the experiment again and start the service with that recipe; confirm framework, normalization key, dimensions and semantics |
| Missing checkpoint or VLM assets | Preserve run metadata and set explicit asset overrides in the resolved config |
| Unsupported sampler | Inspect `serving_contract.sampling_modes`; both the loaded head and recipe must enable the requested mode |
| Invalid FAST output | The generated tokens could not be decoded; do not substitute zero actions or another processor |
| Pose IK rejection | Check coordinate frames, pose convention, gripper range and offline geometry against the chosen contract |

## Extension points

Follow the data path in this order when reviewing the integration:

| Component | Responsibility |
| --- | --- |
| `manimux/servers/starvla.py` | Resolve the existing configuration and launch the independent native service |
| `StarVLA/deployment/model_server/serve.py` | Load the checkpoint and start native serving |
| `StarVLA/deployment/model_server/serving_contract.py` | Validate deployment identity and inspect loaded sampler capabilities |
| `StarVLA/deployment/model_server/sampling_session.py` | Validate sampler inputs; own reset and per-connection DVAC calibration |
| `manimux/policies/starvla/client.py` | Own the connection, verify identity, exchange requests and select AAC candidates |
| `manimux/policies/starvla/codec.py` and `eef.py` | Convert native arrays to canonical robot groups and pose conventions |
| `manimux/policies/capabilities.py` | Shared backend identity comparison for clients and runtime |
| `manimux/policies/aac.py` | Geometry-based candidate features and selection on canonical joint chunks |
| `manimux/policy_adapter/pose.py` | Apply explicit pose anchors and convert targets through configured offline FK/IK |
| `scripts/validation/starvla_offline_probe.py` | Exercise native serving, action conversion and optional offline runtime |

For another checkpoint using an existing schema, add a matching recipe and experiment.
For a new training schema or sampler, implement it in StarVLA first and verify its
normalization and capabilities there. Model loading and transforms stay in StarVLA;
robot geometry stays in ManiMux. Neither side should guess an embodiment from a
checkpoint name.

`group_prefixes` is explicit for pose adapters. Native chunks contain the full
configured horizon; AAC selection and IK-prefix decoding shorten them at their own
boundaries. Identity, session, dimensions and unsupported-mode checks remain active.

Production source, recipes and documentation live in the repository directories
listed above. `.local/` holds private station files, probe outputs and review material;
`tests/` holds ignored local regression checks. Neither is imported by the production
client or model service. Additional integration tests stay outside `StarVLA/`.
