# MolmoAct2 + YAM deployment

This deployment uses the official OLMo-to-Hugging-Face export of a YAM checkpoint,
not the RoboDojo model's normalization or robot layout. Select
`manimux/configs/experiments/example-yam-molmoact2/coin/serial.yaml` first;
`coin/async.yaml` adds progress-triggered replanning and latency trimming.

## Model and action contract

- Model implementation: `third_party/XPolicyLab/policy/MolmoAct2/model.py`.
- Canonical source: `third_party/XPolicyLab/policy/MolmoAct2/molmoact2/`, a pinned
  upstream submodule. The YAM recipe selects `implementation: local`; model and
  processor classes come from its `experiments/olmo/hf_model/`, while weights and
  preprocessing/normalization assets remain in the selected HF checkpoint.
- Native training/export entrypoints: `policy/MolmoAct2/train.sh` and `export.sh`
  inside XPolicyLab. Training run directories hold data, configuration and results,
  not the maintained model implementation. See the policy README for dry-run usage.
- Recipe: `manimux/configs/policy/molmoact2/yam/latest300-step30000.yaml`.
- Checkpoint: `${paths.checkpoints}/finetuned/MolmoAct2/molmoact2_latest300_step30000/hf`.
  Set `paths.checkpoints` in the private station file. Use the validated export's
  tokenizer, not an unrelated tokenizer folder beside the original native checkpoint.
- Three RGB views, in order: front/high, left wrist, right wrist. The HF processor
  owns image preprocessing and discrete state tokens.
- Measured state and absolute output: left 6 joints + gripper, right 6 joints +
  gripper. Norm tag `yam_latest300_joint`; HF `predict_action` owns state normalization
  and action denormalization. The shared JointAdapter must not normalize again or use IK.
- 30 action rows at 30 Hz, ten flow steps, BF16 CUDA, SDPA, CUDA graphs disabled.
  The runtime interpolates the reference at 100 Hz. Actual GPU latency must be
  measured before claiming Async timing is sufficient for this one-second horizon.
- HELLO verifies checkpoint, norm tag, horizon, steps, dtype and seed.
- Serial/Async use `default`; BID uses `aac` independent candidates with shared
  observation context and batched action-expert generation. Local-source inference also implements `rtc`, `paint`
  and `autohorizon`. The complete model horizon and `n_obs_steps=1` are required;
  AutoHorizon needs at least three solver steps. Check actual HELLO capabilities.
- Example variants live beside `coin/async.yaml`: `rtc.yaml`, `paint.yaml`,
  `autohorizon.yaml` and `bid.yaml`. BID uses 16 batched candidates and synchronous execution/selection;
  AutoHorizon uses its selected serial cadence. These presets are software integration
  examples, not tuned real-robot experiment arms or additions to the formal study.
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
- FULL checkpoint Coin examples are `coin/rtc_graph.yaml`, `paint_graph.yaml`,
  `autohorizon_graph.yaml`, `ahs_graph.yaml` and `bid_graph.yaml` under
  `manimux/configs/experiments/molmoact2-full-latest300-step30000/`.
  They do not alter the existing Serial18 formal study list. Graph/trace flags
  participate in backend identity checks; a compatible service reload is necessary
  when selecting these configurations. No automatic service restart is performed.

## Environments and offline checks

Use a separate MolmoAct2 environment. For the official YAM HF export:

```bash
MOLMOACT2_PROFILE=official_yam MOLMOACT2_PYTHON=3.12 bash third_party/XPolicyLab/policy/MolmoAct2/install.sh
```

The default installation profile remains RoboDojo. Do not replace packages inside
another running model's environment. Bind `services.policy_molmoact2.endpoint`,
checkpoints, camera serials and YAM buses in one private station file.

From the repository root, set `STATION` to that file and run:

```bash
CONFIG=manimux/configs/experiments/example-yam-molmoact2/coin/serial.yaml
STATION=manimux/configs/local/station_molmoact2.yaml
envs/yam/.venv/bin/python -m manimux inspect --config "$CONFIG" --local "$STATION"
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config "$CONFIG" --local "$STATION" --check
third_party/XPolicyLab/policy/MolmoAct2/.venv/bin/python -m manimux.servers.policy --config "$CONFIG" --local "$STATION" --check
```

These commands resolve configuration only. A real-weight disconnected probe must
also verify the HELLO identity, recorded RGB/state inputs, finite 30-by-14 output,
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
Prepare can connect and move the robot because this recipe enables start/Home motion.
Do not perform those actions during disconnected qualification. Reuse the same model
service for Serial and Async only when their checkpoint identity is unchanged.

## Disconnected sampler and runtime qualification

Use `scripts/validation/molmoact2_sampling_probe.py` in the model environment with
`--checkpoint`, `--observation` (recorded three-camera/state NPZ) and `--output`.
It checks exact fixed-noise baseline/bridge/AutoHorizon/zero-guidance RTC parity,
nonzero guidance, independent BID candidates and synchronized adapter wall latency.
Two warmup calls and ten measured calls per mode are the defaults. Runtime scheduling
and transport are excluded from that timing; model loading is also excluded.
Use `--steps 5` for the separate lower-step comparison, not a silent baseline change.

`molmoact2_runtime_probe.py` exercises the real WS server, process worker, shared
JointAdapter, strategy, timeline, executor and command validation with recorded pixels
and an explicitly constructed fake plant. Model and runtime dependencies stay in
separate processes/environments. Required arguments are `--local`, `--checkpoint`,
`--observation` and `--output`; `--seconds 10` runs 1,000 control ticks per variant.
It records raw responses, rejection reasons, conditional request counts and final
would-be commands. It never constructs physical camera or motor drivers.

The explicit `coin/rtc_5steps.yaml` / `coin/paint_5steps.yaml` variants use five solver
steps and a 14-row initial delay/window. They require a five-step policy server;
backend identity rejects an accidental ten-step service. Fewer solver steps change
the numerical sampler and require a separate physical-quality evaluation.

A November 13, 2026 shared-RTX-4090 probe of the latest300 30k checkpoint found
adapter medians of 262/646/630/267/2916 ms for default/RTC/PAINT/AutoHorizon/BID-16
at ten steps. That BID measurement used serial candidate generation, before the
batched action-expert path; it is not a latency estimate for the current BID path.
The ten-step RTC runtime reported an infeasible delay and issued
mostly unconditioned requests; PAINT rejected two underestimated-prefix responses.
Do not label that run sustained conditioned RTC or uninterrupted PAINT. These are
hardware/configuration-specific measurements on saved, nonsimultaneous image/state
inputs, not general speed claims or physical qualification.

The separate five-step run measured RTC/PAINT medians of 359/354 ms. Their
14-row-window variants each completed a ten-second disconnected runtime with 20
conditioned requests, no timing-infeasibility event and no rejected plans. This is
short-run software evidence on that shared GPU; it does not establish task quality
or a worst-case real-time guarantee. Fixed-noise AutoHorizon left the actions exactly
unchanged and selected the full 30-row horizon on the benchmark input; horizon
adaptation benefits remain unqualified.

A subsequent same-day repeat explicitly stopped the InternW0 model service with user
authorization. No other model process occupied the GPU; desktop/remote-display
processes remained. With identical source hashes, input and settings, ten-step
medians were 259/631/618/259/2780 ms in the same algorithm order, approximately
0.8-4.7% lower than the earlier run. Five-step medians were 173/350/346/178/1453 ms.
Each algorithm used two warmup and ten measured calls. Baseline actions were
bitwise identical to the earlier run at each step count. The ten-step runtime
still reported RTC delay infeasibility and two PAINT prefix-delay rejections.
Five-step RTC/PAINT each completed 1,000 disconnected control ticks, with 20
conditioned requests and zero rejected plans or timing-infeasibility events.
The model service was left stopped as requested; this benchmark did not restart it.
