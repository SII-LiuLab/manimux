# GR00T N1.7 + YAM runbook

> 2026-10-09: The old screwdriver, red-object, pass-ball and offline experiment entries below were retired. These commands are historical; select a current entry from the [experiment index](../../../../../manimux/configs/experiments/README.md). Diagnostic tools now require an explicit experiment argument.


## Red-ball step10000 with Pi-style RTC

The paired experiment is
`manimux/configs/experiments/pick_red_object/groot/yam_groot_rtc_joint_step10000.yaml`.
It selects `ziyang/gr00t-n17-yam-pick-red-ball-box-step10000`, not the robocurve
baseline described in the historical section below. The recipe uses the
checkpoint's three RGB views, absolute joint/gripper actions, 16 output steps,
four denoising iterations and a 1/30-second action interval.

The model implementation lives in `third_party/XPolicyLab/policy/GR00T_N17/`; ManiMux reuses
its existing RTC strategy, WebSocket client, JointAdapter and smooth executor.
`inference.rtc.beta` controls guided sampling, and
`inference.rtc.min_execute_policy_steps` controls the execution window. Initial
settings are beta 9.1, an 8-step window and a 4-step delay estimate. The runtime
updates its delay estimate from measured request/commit latency. A 16-step
horizon requires the conservative delay to fit within 8 steps (about 267 ms)
for the standard RTC overlap condition; inspect runtime delay events if it does
not. Changing the YAML horizon cannot extend the checkpoint's trained horizon.

Run each service in its own terminal from the repository root. Reuse an already
running camera service only if it uses the same three-camera configuration.

```bash
cd {repo}
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --config manimux/configs/experiments/pick_red_object/groot/yam_groot_rtc_joint_step10000.yaml
```

```bash
cd {repo}
third_party/XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv/bin/python -m manimux.servers.groot \
  --experiment manimux/configs/experiments/pick_red_object/groot/yam_groot_rtc_joint_step10000.yaml
```

```bash
cd {repo}
envs/yam/.venv/bin/python -m manimux run \
  --config manimux/configs/experiments/pick_red_object/groot/yam_groot_rtc_joint_step10000.yaml
```

The experiment enables real YAM execution when a rollout is started in the RoboGUI.
All three commands resolve the private station's camera and policy endpoints.
The GR00T model environment uses Python 3.10; its launcher delegates experiment
resolution to `envs/yam/.venv/bin/python`, then loads the model in its own environment.
Use `--local /path/to/station.yaml` consistently to select another station.
Adding `--check` to the model command validates configuration and local assets
without loading the GPU model or connecting hardware; it does not prove inference.

Validation on 2026-09-28: 14 existing adapter/WS tests and 7 focused offline RTC
checks passed. A reduced real DiT matched the pre-change sampler exactly for
ordinary inference, AAC and upstream prefix freezing with fixed seeds. Full
red-ball step10000 weights ran on an RTX 4090 using synthetic RGB and checkpoint
mean state: finite `(16, 14)` output, identical zero-mask/default results and
post-reset results, no accumulated parameter gradients, and changed output under
nonzero guidance. Four warm RTC calls took 98.7–126.2 ms; these timings exclude
camera capture, network transport, runtime decoding and execution. The default
recipe's Cosmos repository ID also loaded successfully. No camera, robot or
real-task rollout was exercised in these checks.

## Historical robocurve baseline (2026-08-20)

The following records an earlier checkpoint and deployment. Its old experiment
paths and hardware results are not acceptance evidence for the red-ball RTC recipe.

This guide covers only `robocurve/gr00t-n1.7-yam-molmoact2`, a YAM fine-tuned checkpoint based on NVIDIA
[Isaac-GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T). It does not run the
`nvidia/GR00T-N1.7-3B` base model zero-shot on YAM.

Recorded status: **The XPolicy adapter, checkpoint contract, model environment, Cosmos, GPU forward pass,
XPolicy WebSocket, default ManiMux runtime, three real cameras, dual-arm YAM, and Recorder have passed end-to-end validation. The
checkpoint did not complete the tested tasks; this is a policy-quality result, not an infrastructure connection failure.**

## Contract

```text
3 RGB cameras + 14D current joint state + instruction
  -> XPolicyLab GR00T_N17 adapter
  -> GR00T N1.7 YAM checkpoint + checkpoint statistics
  -> 16 x 14 absolute joint positions at 30 Hz
  -> ManiMux ActionTimeline -> SmoothExecutor -> YAM
```

| Item | Value |
|---|---|
| Official model code | `third_party/XPolicyLab/policy/GR00T_N17/gr00t_n17/` (vendored Isaac-GR00T N1.7) |
| XPolicy adapter | `third_party/XPolicyLab/policy/GR00T_N17/model.py` |
| Weights | `{repo}/checkpoints/finetuned/robocurve/gr00t-n1.7-yam-molmoact2` |
| Release source | `robocurve/gr00t-n1.7-yam-molmoact2` |
| Normalization | `new_embodiment` in the checkpoint's `statistics.json` |
| cameras | `base_view`, `left_wrist_view`, `right_wrist_view` |
| state/action | `left_arm[6] + left_gripper[1] + right_arm[6] + right_gripper[1]` |
| Action | 16 steps, 14D, absolute joint position |
| Timing semantics | Training data at 30 Hz, `action_dt_s = 1/30` |
| denoise | checkpoint `num_inference_timesteps = 4` |

The checkpoint's `processor_config.json` and `statistics.json` are part of its model contract. Do not substitute
Pi05 stats, the original XPolicy ARX modality configuration, or the base model's default embodiment.

## 1. Offline checks

This step does not import torch or load weights onto a GPU:

```bash
cd {repo}
envs/yam/.venv/bin/python manimux/servers/groot.py \
  --config manimux/configs/policy/groot/yam/finetune.yaml \
  --check
```

The check verifies both weight shards, three matching camera keys, 14D state/action, absolute actions,
a horizon of 16, consistent 30 Hz declarations, and complete YAM q01/q99 stats dimensions. In the output,
`contract_status: ready` **means only that these static contracts pass**. Once both the model environment and Cosmos are ready,
`runtime_status` becomes `environment_and_cosmos_present_gpu_forward_not_verified`. This still
means only that local files are complete. Until actual inference completes,
`inference_status` remains `not_verified`.

## 2. Install the model environment

The documented local `gr00t_n17/.venv` has its dependencies installed. Use the same installer for a new checkout:

```bash
cd {repo}/third_party/XPolicyLab/policy/GR00T_N17
bash install.sh
```

The GR00T N1.7 processor also requires gated `nvidia/Cosmos-Reason2-2B`. Obtain Hugging Face
access first. To use a local model, set `cosmos_model_path` in
`manimux/configs/policy/groot/yam/finetune.yaml` to its local directory.

After installation, repeat step 1. Do not start the model service while `runtime_status` remains blocked or requires operator
action.

The ManiMux WebSocket worker also requires the shared XPolicy dependencies. In a new environment, install
`.[xpolicylab]` using [`xpolicylab-runbook.md`](xpolicylab.md). There is no need to reinstall
when the existing `envs/yam/.venv` already satisfies them.

## 3. Start the model service

Terminal 1:

```bash
cd {repo}
third_party/XPolicyLab/policy/GR00T_N17/gr00t_n17/.venv/bin/python \
  manimux/servers/groot.py \
  --config manimux/configs/policy/groot/yam/finetune.yaml
```

In terminal 2, first run a single forward probe without cameras or CAN. It sends three deterministic synthetic RGB
images and the configured 14D YAM initial state, checking through the production XPolicy WebSocket and ManiMux adapter
`16 x 14` absolute joint chunk:

```bash
cd {repo}
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config manimux/configs/experiments/pick_box/yam_groot_manimux.yaml
```

A real GPU forward pass and WS round trip are complete only when the output reports `"status": "ok"`, `"action_space": "joint_position"`,
`"canonical_shape": [16, 14]`, and finite actions. A listening port alone
does not establish model readiness.

The first request may include CUDA/kernel/cache startup, so it cannot establish steady-state latency. Keep the model service running
and execute three times:

```bash
cd {repo}

for i in 1 2 3; do
  echo "===== Probe $i ====="
  envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
    --config manimux/configs/experiments/pick_box/yam_groot_manimux.yaml
done
```

Acceptance: all three requests return finite values with `native_shape` / `canonical_shape = [16, 14]`, and steady-state
`round_trip_ms` is substantially below the `533.3 ms` duration of a 16-step / 30 Hz chunk.

Measured on 2026-08-20: the first cold request took `632.2 ms`; the next three took `101.3 / 91.7 / 89.4 ms`.
Thus the GPU, model, WS, adapter, and steady-state latency checks passed; this alone does not validate the ManiMux runtime,
Recorder, or hardware loop.

## 4. Cameras, preflight, and hardware

In terminal 2, start the shared camera service. Do not start another instance if a service already uses `5555`:

```bash
cd {repo}
envs/yam/.venv/bin/manimux-camera-server --camera-config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml
```

First check both CAN interfaces:

```bash
for c in can_left can_right; do
  ip -details link show "$c" | grep -o 'ERROR-ACTIVE\|ERROR-PASSIVE\|BUS-OFF'
done
```

Then run the read-only preflight. It connects to both physical arms and three cameras and performs two GR00T inference requests, but explicitly
disables `move_to_start_on_connect` and `home_on_close`, and sends no model actions:

```bash
cd {repo}
envs/yam/.venv/bin/python scripts/validation/pi05_base_yam_preflight.py \
  --config manimux/configs/experiments/pick_box/yam_groot_manimux.yaml
```

Verify every `contract_checks` value is `true`, then manually inspect the measured state, first action,
`max_first_joint_delta_rad`, gripper range, camera shapes, and steady-state latency. The script retains its early
Pi05 name, but constructs the generic YAM policy/adapter from the supplied configuration and does not load Pi05 here.

After clearing the workspace, keeping the emergency stop ready, and confirming preflight output, start default ManiMux:

```bash
cd {repo}
envs/yam/.venv/bin/manimux run --once --config manimux/configs/experiments/pick_box/yam_groot_manimux.yaml
```

After connecting, the hardware runtime takes `3.5 s` to move to the configured initial pose; normal `Ctrl-C` exit takes `3.5 s`
to return Home. Do not stop model or camera services while the arms are moving; use the physical emergency stop first in an emergency.

The two controlled episodes on 2026-08-20 recorded 91/74 accepted chunks with no plan rejection or
invalid actions, validating default ManiMux, three cameras, dual-arm command delivery, and Recorder. Neither completed
the pick task; these are checkpoint task failures, not evidence that the infrastructure did not run.

At the time of this baseline, the adapter exposed only ordinary inference and did
not connect the upstream overlap/frozen-step branch to ManiMux. The new Pi-style
RTC integration is described above; the old hardware runs did not exercise it.

## Not yet validated

- Task success rate and cross-task generalization of this checkpoint.
- Real-robot RTC task success with the red-ball checkpoint.
