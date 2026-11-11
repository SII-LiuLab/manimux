# MolmoAct2 + YAM deployment

This deployment uses the official OLMo-to-Hugging-Face export of a YAM checkpoint,
not the RoboDojo model's normalization or robot layout. Select
`manimux/configs/experiments/example-yam-molmoact2/coin/serial.yaml` first;
`coin/async.yaml` adds progress-triggered replanning and latency trimming.

## Model and action contract

- Model implementation: `third_party/XPolicyLab/policy/MolmoAct2/model.py`.
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
- Serial/Async use `default`. The optional `aac` hook returns independently sampled
  candidates and supplies the common AAC/BID transport. That hook alone does not
  qualify candidate selection or its latency. RTC, PAINT and AutoHorizon are not
  implemented for this backend and must not be advertised or selected.

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
