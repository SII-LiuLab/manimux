# MolmoAct2 + YAM deployment

Start with the [serial example](../../../../manimux/configs/examples/yam-xpolicylab/molmoact2-serial.yaml).
Its recipe uses checkpoint and normalization placeholders for offline configuration
checks. In a private copy, select an export matching the robot layout, normalization
assets and action horizon. Keep study presets, recorded inputs and benchmark results local.

## Model and action contract

- Model implementation: `third_party/XPolicyLab/policy/MolmoAct2/model.py`.
- Canonical source: `third_party/XPolicyLab/policy/MolmoAct2/molmoact2/`, a pinned
  upstream submodule. The YAM recipe selects `implementation: local`; model and
  processor classes come from its `experiments/olmo/hf_model/`, while weights and
  preprocessing/normalization assets remain in the selected HF checkpoint.
- Native training/export entrypoints: `policy/MolmoAct2/train.sh` and `export.sh`
  inside XPolicyLab. Training run directories hold data, configuration and results,
  not the maintained model implementation. See the policy README for dry-run usage.
- Set `model_path` and `norm_tag` from the selected checkpoint in a private policy
  recipe. Bind checkpoint paths through `${paths.checkpoints}` in the station file.
  Use the export's own tokenizer and normalization assets.
- Three RGB views, in order: front/high, left wrist, right wrist. The HF processor
  owns image preprocessing and discrete state tokens.
- Measured state and absolute output: left 6 joints + gripper, right 6 joints +
  gripper. HF `predict_action` owns state normalization
  and action denormalization. The shared JointAdapter must not normalize again or use IK.
- Set action horizon, action rate, solver steps, dtype and attention backend to
  match the checkpoint contract. Measure GPU latency before choosing Async timing.
- HELLO verifies checkpoint, norm tag, horizon, steps, dtype and seed.
- Serial/Async use `default`; BID uses `aac` independent candidates with shared
  observation context and batched action-expert generation. Local-source inference also implements `rtc`, `paint`
  and `autohorizon`. The complete model horizon and `n_obs_steps=1` are required;
  AutoHorizon needs at least three solver steps. Check actual HELLO capabilities.
- RTC/PAINT conditions are physical absolute joint rows on the wire and are normalized
  only in XPolicyLab. Padding is excluded from guidance. AutoHorizon reads action
  self-attention; it does not interpret visual cross-attention as an action horizon.
- Specialized sampling reuses the shared flow samplers and explicit model hooks.
  `enable_cuda_graph: true` supports ordinary, RTC, PAINT, AutoHorizon and candidates
  through native CUDA Graph capture/replay, not `torch.compile`. Runtime delay constraints still apply. A successful
  sampler call does not imply all conditioned requests fit the 30-row horizon.
- AHS uses `return_denoise_trace: true` and the shared ChunkTrust selector with a
  MolmoAct2 H30 profile. Candidate horizons `[6,12,18,24,30]` are an explicit local
  adaptation of the PI-family selector, not a claim of upstream MolmoAct2 results.
  The model returns raw normalized velocities at every actual denoising step;
  CPU selection and episode history stay outside capture. Trace-enabled services
  advertise default/RTC only; select a separate trace recipe for AHS.
- Graph/trace flags participate in backend identity checks. Reload an incompatible
  service only with operator authorization before selecting a different configuration.

## Environments and offline checks

Use a separate MolmoAct2 environment. For the official YAM HF export:

```bash
MOLMOACT2_PROFILE=official_yam MOLMOACT2_PYTHON=3.12 bash third_party/XPolicyLab/policy/MolmoAct2/install.sh
```

The default installation profile remains RoboDojo. Do not replace packages inside
another running model's environment. The example uses `services.policy.endpoint`.
Bind the selected endpoint, checkpoints, camera serials and YAM buses in the station file.

From the repository root, set `STATION` to that file and run:

```bash
CONFIG=/path/to/private/deployment.yaml
STATION=/path/to/private/station.yaml
envs/yam/.venv/bin/python -m manimux inspect --config "$CONFIG" --local "$STATION"
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config "$CONFIG" --local "$STATION" --check
third_party/XPolicyLab/policy/MolmoAct2/.venv/bin/python -m manimux.servers.policy --config "$CONFIG" --local "$STATION" --check
```

These commands resolve configuration only. A real-weight disconnected probe must
also verify the HELLO identity, recorded RGB/state inputs, finite output with the declared horizon and 14 joint dimensions,
reset behavior, JointAdapter, ActionTimeline, executor and final command validation.
Keep raw predictions before decoding. CPU checks establish wiring; they do not
qualify CUDA latency or physical task success.

## Operator launch

Launch only when the user requests the services and the GPU/robot ownership is clear.
Reuse an already-compatible camera or GUI service; do not create a second owner.
Run these in separate terminals with the same `CONFIG` and `STATION` above:

```bash
# Camera process; omit if the matching service is already running.
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config "$CONFIG" --local "$STATION"
# Policy process; loads weights without connecting the robot.
third_party/XPolicyLab/policy/MolmoAct2/.venv/bin/python -m manimux.servers.policy --config "$CONFIG" --local "$STATION"
# GUI process; omit if already running.
envs/yam/.venv/bin/python -m manimux.robogui.dashboard --robot yam --port 8086
# Interactive runtime. Do not add --once for an operator-managed test.
envs/yam/.venv/bin/python -m manimux run --config "$CONFIG" --local "$STATION"
```

The operator performs Prepare Run, Activate Warmup, Start rollout and Finish & Home.
Prepare may connect and move the robot when the private recipe enables start/Home motion.
Do not perform those actions during disconnected qualification. Reuse the same model
service for Serial and Async only when their checkpoint identity is unchanged.

## Disconnected sampler and runtime qualification

Use recorded RGB/state inputs and fake devices to check fixed-noise parity, finite
outputs, conditioned sampling, candidate independence, timeline acceptance and final
command validation. Keep checkpoint-specific probe scripts and results local.
Measure model loading, warmup, steady inference and transport separately; record
source and configuration identities. Solver-step changes require separate timing
and physical-quality evaluation. Software checks do not establish task success.
