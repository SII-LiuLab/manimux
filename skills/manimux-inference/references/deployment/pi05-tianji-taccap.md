# Pi05 zero-pose / Tianji–TacCap

This deployment reuses XPolicyLab `Pi_05` and the Xiaomi Tianji absolute EE decoder.
Existing YAM/ALOHA Pi05 recipes retain their original defaults and transforms.
No camera, policy or robot service is started by configuration loading or `--check`.

The published entry is the pack-plate wrist-only checkpoint in the
[`example-tianji-pi05`](../../../../manimux/configs/experiments/example-tianji-pi05/README.md)
study. The zero-pose profile was first built for a pass-ball checkpoint, which shares the
contract below; its recipe and export are private.

## Contract

- Profile: `tianji_taccap_pi05_pack_plate` (pass ball: `tianji_taccap_pi05_zero_pose`);
  `action_type: ee`, `env_cfg_type: tianji_dual`.
- Model: Pi05 LoRA (`gemma_2b_lora` / `gemma_300m_lora`), 32 model dimensions,
  32-step horizon, 30 Hz action targets. Hardware control remains 100 Hz.
- Native state/action order: right then left, each `xyz(m), Rot6D(first two rows), opening`.
  Both grippers use continuous normalized `[0,1]` opening.
- Model pose slots are zeroed before normalization and again after normalization,
  before state tokenization; gripper normalization remains active. Zero pose is an
  input ablation, not a robot Home command.
- Two real wrist RGB images; synthetic black base image has `image_mask=false`.
  Do not substitute Xiaomi's active black ego view or the YAM camera pipeline.
- Training converted tool axes with
  `C=[[0,-1,0],[0,0,-1],[1,0,0]]`: `R_train=R_TCP @ C.T`.
  Positions preserve each arm's base reference; do not apply a base-frame translation.
- Predictions use one current-TCP anchor for the whole chunk. XPolicyLab retains the
  real anchor outside zero-pose inputs, restores absolute poses and converts rotations
  back with `R_TCP=R_train @ C`. Output is standard `xyz + quaternion wxyz`.
- ManiMux performs FK/IK only. Shared differential IK, motion limits, gripper tolerance,
  process decoding and atomic dual-arm chunk submission match the Xiaomi configuration.
- The pack-plate profile advertises `default` and `rtc` sampling; the pass-ball profile
  only `default`. AAC/PAINT/AutoHorizon/DVAC are unsupported until their native action
  transforms and sampler hooks are validated.
- Task instruction is the training task ID: `plate` for pack plate
  ([task](../../../../manimux/configs/tasks/pack_plate.yaml)), `pass_ball` for pass ball.
- Pack-plate experiments select `tianji_taccap_pack_plate.yaml`. The shared
  `tianji_taccap.yaml` retains the original pass-ball Home. Loading any assembly never
  moves hardware.

## Artifacts and station binding

- Recipe: [`manimux/configs/policy/pi05/tianji/pack_plate/wrist-only-step59999.yaml`](../../../../manimux/configs/policy/pi05/tianji/pack_plate/wrist-only-step59999.yaml)
- Study: `manimux/configs/experiments/example-tianji-pi05/{_base_.yaml,pack_plate/serial.yaml,pack_plate/rtc.yaml}`
- List: [`manimux/configs/lists/example-tianji-pi05.yaml`](../../../../manimux/configs/lists/example-tianji-pi05.yaml)

The base keeps `execute: false` and `end_effector_control: false`. The inter-hand-relative
recipes in the same folder have their [own runbook](pi05-tianji-pack-plate-inter-hand-relative.md).

Keep device addresses and serials in a private Tianji station file copied from
`manimux/configs/local/tianji_taccap_example.yaml`, following
`manimux/configs/local/README.md`. Do not infer left/right from enumeration order.
The default station is `manimux/configs/local/station.yaml`; `--local` selects another.

Set `paths.checkpoints` in the private station to the directory holding the export. The
recipe selects its run and checkpoint beneath that root. Check existing YAM artifact
bindings before changing a shared storage-root setting. Ensure the station's
`services.policy.endpoint` points to the intended Pi05 service; its listener and
runtime client will use that endpoint instead of recipe port 8571.

The export must retain:

```text
pi05-pack-plate-wrist-only-final-59999/
  checkpoint-59999/params/...
  checkpoint-59999/assets/pack-plate-taccap-h32-zero-pose/norm_stats.json
```

XPolicyLab's `policy/Pi_05/pack_plate_contract.py` validates the step, export identity
and the normalization SHA256. Do not use another state variant's statistics. The source
export directory name is part of this recipe's identity; when relocating the export,
preserve that directory name.

## Read-only checks

Run from the ManiMux root. These load no model and open no device:

```bash
CONFIG=manimux/configs/experiments/example-tianji-pi05/pack_plate/serial.yaml
LOCAL=manimux/configs/local/station.yaml
envs/tianji/.venv/bin/manimux inspect --config $CONFIG --local $LOCAL
envs/tianji/.venv/bin/python -m manimux.servers.camera.server --config $CONFIG --local $LOCAL --check
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.policy \
  --config $CONFIG --local $LOCAL --check
```

To also validate the artifact paths and normalization digest without loading the model:

```bash
.venv/bin/python -m manimux.servers.pi05 --experiment $CONFIG --local $LOCAL --check
```

## Offline model forward

Use the existing isolated OpenPI environment (`Pi_05/install.sh` if it is not prepared).
This command loads the real model and uses synthetic constant RGB, without services:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  -m XPolicyLab.policy.Pi_05.offline_pack_plate \
  --config manimux/configs/policy/pi05/tianji/pack_plate/wrist-only-step59999.yaml \
  --checkpoint <paths.checkpoints>/pi05-pack-plate-wrist-only-final-59999/checkpoint-59999
```

A synthetic forward checks loading and output contracts, not perception, control
tracking or task success.

## Deployment commands

The commands below are instructions, not an instruction to start hardware automatically.
First bind the station as above. Use four separate terminals, each with the same
`CONFIG` and `LOCAL`; select `pack_plate/rtc.yaml` for RTC.

The TacCap SDK must be importable from `envs/tianji/.venv` (station setup). If it is
only built in-tree, prefix the camera and runtime commands with
`PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python"`.
The model server does not need this hardware SDK path.

Camera server (opens physical TacCap cameras; reuse an existing matching service):

```bash
envs/tianji/.venv/bin/python -m manimux.servers.camera.server --config $CONFIG --local $LOCAL
```

Model server (isolated OpenPI environment; no hardware ownership):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.policy \
  --config $CONFIG --local $LOCAL
```

RoboGUI (display and experiment controls; no hardware ownership):

```bash
envs/tianji/.venv/bin/python -m manimux.robogui.dashboard --robot tianji \
  --host 127.0.0.1 --port 8086
```

Open `http://127.0.0.1:8086`. The study enables RoboGUI and keeps execution false.

Runtime service (`execute: false`; reads hardware observations during the experiment):

```bash
envs/tianji/.venv/bin/python -m manimux run --config $CONFIG --local $LOCAL
```

For counted trials of all methods, pass
`--config-list manimux/configs/lists/example-tianji-pi05.yaml --local $LOCAL` to the
camera, model and runtime services instead of `--config`.

Use RoboGUI Prepare → Start rollout → Finish rollout. In an executing Tianji service,
Return Home is available during a paused rollout and through idle recovery after Finish.
Home/Recovery operations can move the robot; zero pose does not mean a physical zero pose.

Verify handshake, camera orientation, TCP axes, continuous opening, full 32-step IK,
tracking/decode latency and saved rollouts before enabling physical execution. For an
explicitly authorized robot trial, copy the experiment into an ignored private directory,
retain resolved config references and set `execute: true` and
`end_effector_control: true` only in that copy.

## Validation performed during implementation

- Static Python/shell checks and focused ManiMux / XPolicyLab regression tests.
- Real step-20000 GPU forward of the private pass-ball checkpoint with synthetic wrist RGB:
  32 finite standard dual-arm EE actions. Pack-plate checks are in its
  [runbook](pi05-tianji-pack-plate.md#checks-on-this-installation).
- No camera or robot service was started; no real-robot rollout or task success was measured.
- The training submission transforms are preserved under `third_party/XPolicyLab/policy/Pi_05`.
  See that policy README for source provenance and unsupported training stages.
