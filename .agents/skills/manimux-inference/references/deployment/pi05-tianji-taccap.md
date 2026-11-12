# Pi05 zero-pose / Tianji–TacCap pass ball

This deployment reuses XPolicyLab `Pi_05` and the Xiaomi Tianji absolute EE decoder.
Existing YAM/ALOHA Pi05 recipes retain their original defaults and transforms.
No camera, policy or robot service is started by configuration loading or `--check`.

## Contract

- Profile: `tianji_taccap_pi05_zero_pose`; `action_type: ee`, `env_cfg_type: tianji_dual`.
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
- Only `default` sampling is advertised. RTC/AAC/PAINT/AutoHorizon/DVAC are unsupported
  for this profile until their native action transforms and sampler hooks are validated.
- Task instruction is `pass_ball`, matching one of the training manifest's task strings.
- Pass-ball experiments select `tianji_taccap_pass_ball.yaml`; pack-plate experiments
  select `tianji_taccap_pack_plate.yaml`. The shared `tianji_taccap.yaml` retains the
  original pass-ball Home. Loading any assembly never moves hardware.

## Artifacts and station binding

Default recipe/experiment:

- `manimux/configs/policy/pi05/tianji/pass_ball/zero-pose-step20000.yaml`
- `manimux/configs/experiments/pass_ball/pi05/tianji_taccap_pi05_zero_pose_step20000.yaml`

The fixed held-out validation selected step 20000; this is not a robot success result.
Matching step-59999 recipe and experiment are also available for the final checkpoint.
Use the matching pair when switching steps; `--checkpoint` alone cannot relabel a recipe.

Keep device addresses and serials in the existing private Tianji station file, following
`manimux/configs/local/README.md`. Do not infer left/right from enumeration order.
The default station is `manimux/configs/local/station.yaml`; `--local` selects another.

To bind the checked-in artifact paths, use `paths.checkpoints: /home/jw/Desktop/policy`
in the private station. These recipes select their own run and checkpoint beneath that
root. Check existing YAM artifact bindings before changing a shared storage-root setting.
Alternatively this profile accepts `paths.checkpoint` as the explicit step directory and
`paths.norm_stats` as the normalization directory. Other models' bindings are unchanged.
Ensure the station's `services.policy.endpoint` points to the intended Pi05 service;
its listener and runtime client will use that endpoint instead of recipe port 8570.

The export must retain:

```text
pi05-passball-zero-pose-h32-b16-60k-20260923-192017/
  checkpoints/20000/params/...
  checkpoints/59999/params/...
  normalization/norm_stats.json
```

The launcher/model validate step, export identity and the normalization SHA256. Do not
use another state variant's statistics. The source export directory name is part of
this recipe's identity; when relocating the export, preserve that directory name.

## Read-only artifact check

Run from the ManiMux root. This command uses the actual local step-20000 export and
loads no model or hardware:

```bash
.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/pass_ball/pi05/tianji_taccap_pi05_zero_pose_step20000.yaml \
  --checkpoint /home/jw/Desktop/policy/pi05-passball-zero-pose-h32-b16-60k-20260923-192017/checkpoints/20000 \
  --check
```

Add `--local <private-tianji-station.yaml>` to inspect an existing station binding.

## Offline model forward

Use the existing isolated OpenPI environment (`Pi_05/install.sh` if it is not prepared).
This command loads the real model and uses synthetic constant RGB, without services:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  -m XPolicyLab.policy.Pi_05.offline_pass_ball \
  --config manimux/configs/policy/pi05/tianji/pass_ball/zero-pose-step20000.yaml \
  --checkpoint /home/jw/Desktop/policy/pi05-passball-zero-pose-h32-b16-60k-20260923-192017/checkpoints/20000
```

For step 59999, replace both the recipe and checkpoint step. A synthetic forward checks
loading and output contracts, not perception, control tracking or pass-ball success.

## Deployment commands

The commands below are instructions, not an instruction to start hardware automatically.
First bind the selected experiment and station as above. Use four separate terminals;
replace the station path with the already mapped private Tianji station.

Model server (isolated OpenPI environment; no hardware ownership):

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/pass_ball/pi05/tianji_taccap_pi05_zero_pose_step20000.yaml \
  --local .local/tianji_taccap.yaml \
  --checkpoint /home/jw/Desktop/policy/pi05-passball-zero-pose-h32-b16-60k-20260923-192017/checkpoints/20000
```

The local Python environments are managed by uv; no `conda run` is required.
The root `.venv` currently lacks an installed `xense.taccap`; use the built SDK
kept with this ManiMux checkout at
`manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python`. This exact
import was checked without connecting a device. The documented
`envs/tianji/.venv` does not currently exist on this station. On another installation,
install the TacCap SDK into the intended hardware venv and use its interpreter.

Camera server (opens physical TacCap cameras; reuse an existing matching service):

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/pass_ball/pi05/tianji_taccap_pi05_zero_pose_step20000.yaml \
  --local .local/tianji_taccap.yaml
```

RoboGUI (display and experiment controls; no hardware ownership):

```bash
.venv/bin/python -m manimux.robogui.dashboard \
  --config manimux/configs/robogui/tianji-pass-ball.yaml \
  --host 127.0.0.1 --port 8086
```

Open `http://127.0.0.1:8086`. The templates enable RoboGUI and keep execution false.

Runtime service (`execute: false`; reads hardware observations during the experiment):

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/pass_ball/pi05/tianji_taccap_pi05_zero_pose_step20000.yaml \
  --local .local/tianji_taccap.yaml
```

Use RoboGUI Prepare → Start rollout → Finish rollout. In an executing Tianji service,
Return Home is available during a paused rollout and through idle recovery after Finish.
Home/Recovery operations can move the robot; zero pose does not mean a physical zero pose.

Verify handshake, camera orientation, TCP axes, continuous opening, full 32-step IK,
tracking/decode latency and saved rollouts before enabling physical execution. For an
explicitly authorized robot trial, copy the experiment into an ignored private directory,
retain resolved config references and set `execute: true` only in that copy.

## Validation performed during implementation

- Static Python/shell checks and focused ManiMux / XPolicyLab regression tests.
- Real step-20000 GPU forward with synthetic wrist RGB: 32 finite standard dual-arm EE actions.
- No camera or robot service was started; no real-robot rollout or task success was measured.
- The training submission transforms are preserved under `XPolicyLab/policy/Pi_05`.
  See that policy README for source provenance and unsupported training stages.
