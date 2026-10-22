# SAPolicy + YAM

## Current deployment

SAPolicy checkpoints connect through `third_party/XPolicyLab/policy/SAPolicy` and shared `xpolicylab_ws`.
Model source, dependency installation, data, and training entry points reside in that policy directory. ManiMux owns cameras,
observation mapping, YAM FK/IK, scheduling, execution, and recording. For installation and resources, see
[SAPolicy README](../../../../../third_party/XPolicyLab/policy/SAPolicy/README.md).

Supports `yam_dual / ee`. The model returns absolute end-effector poses for both arms (`[x,y,z,qw,qx,qy,qz]`) and a
continuous gripper opening from 0 closed to 1 open. The YAM adapter solves IK for the whole chunk and hands the runtime
two 7D joint trajectories, so RoboGUI reports `joint_position`; the model's own EE targets are recorded per plan as
`raw_model_eef`.

| Runtime configuration (`manimux/configs/experiments/put_bottles/sapolicy/`) | Policy server (`manimux/configs/policy/sapolicy/yam/`) | Checkpoint |
| --- | --- | --- |
| `yam_sapolicy_mv51_async25_braking_capped.yaml` | `teleopMV51/raw.yaml` | MV51 RAW, trained on this station's teleop data |
| `yam_sapolicy_tcp_rot20_async48_z22.yaml` | `tcp_rot20/raw-50k.yaml` | tcp_rot20 RAW 50k, ABC bottles, zero-shot (below) |
| `yam_sapolicy_caps1010_3task_async48_z22.yaml` | `caps1010_3task_tcp/ema-160k.yaml` | caps1010 3-task EMA 160k, zero-shot (below) |

All three use ManiMux `async` scheduling (no prefix conditioning: each chunk is generated from the current observation
only), four-step seam blending, process-based IK and the shared `SmoothExecutor` in `braking` mode with arm limits
0.6 rad/s and 1.5 rad/s². Without those limits chunk seams reach the motors and the arms shake visibly.
Restart `manimux run` after changing a runtime configuration; restart the policy server only to change checkpoints or sampling.

### MV51 settings

| Parameter | Setting |
| --- | --- |
| Checkpoint | `ft_teleopMV51_from_abc_notcpkv_rot5_bs1024_raw.ckpt`, RAW (`use_ema: false`) |
| Observation | One frame, `d405_front` as top + both wrists |
| Action horizon / sampler steps | 50 / 10 |
| Execution | Up to 25 source steps per chunk |
| Smoother | Braking, 8 Hz low-pass, arm 0.6 rad/s / 1.5 rad/s², gripper 1.0 / 12.0, no grasp/release guards |
| Model action interval / control frequency | 1/30 s / 100 Hz |
| TCP reference | Training MJCF grasp_site, no mount offset, Sep-13 start pose |
| Images | RGB; 640×480 → 224×168 |

## Installation and model service

Run from the ManiMux root. First follow the policy README to build the checkpoint bundle under
`checkpoints/finetuned/sapolicy/` from the original model weights, DINOv2 weights, and matching normalizer.
Model weights and local recordings are excluded from Git.

```bash
bash third_party/XPolicyLab/policy/SAPolicy/install.sh
third_party/XPolicyLab/policy/SAPolicy/.venv/bin/python manimux/servers/sapolicy.py \
  --config manimux/configs/policy/sapolicy/yam/teleopMV51/raw.yaml --check
third_party/XPolicyLab/policy/SAPolicy/.venv/bin/python manimux/servers/sapolicy.py \
  --config manimux/configs/policy/sapolicy/yam/teleopMV51/raw.yaml
```

`--check` validates configuration and resources only. The shared model service defaults to `ws://127.0.0.1:8510`
and does not connect to hardware; only one SAPolicy server can hold that port at a time.

## Cameras and GUI

Reuse an existing camera service; the commands below apply when none is running.

```bash
envs/yam/.venv/bin/manimux-camera-server --camera-config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml
envs/yam/.venv/bin/manimux-robogui --robot yam --host 0.0.0.0 --port 8086
```

GUI defaults to `camera_mode: policy`, following `policy.adapter.camera_map` reported by runtime
(`top: d405_front`, `left: d405_left`, `right: d405_right`). New runs, runtime changes and disconnection clear stale images.

## Rollout

```bash
envs/yam/.venv/bin/manimux run \
  --config-list manimux/configs/lists/sapolicy-custom-standalone.yaml --experiment-id put_bottles_mv51_async25_braking_capped
```

Open `http://localhost:8086`, click **Prepare normal rollout**, arrange the scene, then
click **Start rollout**. **Pause / Hold** pauses; **Finish & Home** saves the run and homes both arms.
Recordings go to the configuration's `run.output_dir` (MV51: `data/experiments/sapolicy-put-bottles-teleopMV51/`).
`result.run_completed` means only that the run completed; bottle-deposit success requires observation or manual evaluation.

## tcp_rot20: ABC bottles zero-shot

`tcp_rot20` is a training-provider checkpoint for dual-arm, three-camera TCP bottle placement (RAW, step 50k),
without fine-tuning on this station's data.

### Resource bundle

`checkpoints/finetuned/sapolicy/tcp_rot20/` is excluded from Git. Copy it between machines with `rsync -aL`
(`-L` dereferences local symlinks):

| File | Contents |
| --- | --- |
| `model.ckpt` | Training checkpoint, `sha256:d84e94f2…b032825a3f` (full value in configuration, verified at service startup) |
| `backbone.pth` | DINOv2 initialization weights (same as teleopMV51), subsequently overwritten by strict checkpoint loading |
| `action_normalizer.pt` | Action normalization matching the checkpoint |
| `resolved_config.yaml` | Resolved training configuration |
| `resolved_config.compiled.yaml` | Same, with `compile_backbone` / `compile_action_head` enabled |
| `source/` | SpatialAlignVLA rollsweep0917 source snapshot with an inference-time RTC patch; ordinary sampling is bitwise identical to the original |

```bash
rsync -aL --info=progress2 <source-host>:~/manimux/checkpoints/finetuned/sapolicy/tcp_rot20/ \
  checkpoints/finetuned/sapolicy/tcp_rot20/
```

### Startup

```bash
third_party/XPolicyLab/policy/SAPolicy/.venv/bin/python manimux/servers/sapolicy.py \
  --config manimux/configs/policy/sapolicy/yam/tcp_rot20/raw-50k.yaml
envs/yam/.venv/bin/manimux run \
  --config-list manimux/configs/lists/sapolicy-custom-standalone.yaml --experiment-id put_bottles_tcp_rot20_async48_z22
```

Server inference takes about 60–90 ms on a 4090 without compilation. For faster inference, change `cfg_file` to
`resolved_config.compiled.yaml` and set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` (approximately 36 ms).
Initial requests compile for several minutes; warm up with requests before starting a rollout.

### Settings and rationale

| Item | Setting | Reason |
| --- | --- | --- |
| Scheduling | ManiMux async, execute 48/50 steps per chunk, blend 4, refill 0.2 s, max plan age 2.5 s | obs_hist=1 and no prefix conditioning; SA plans its descent in rows ~36–49, so replanning after 25–40 steps keeps postponing it (hovering, up-down oscillation) |
| IK decoding | `action_decoding: process` | Control-thread IK stalls each handover for 30–100 ms, causing arm shaking |
| Executor | Smooth braking, 0.6 rad/s, 1.5 rad/s² | Without caps, chunk seams reach the motors (measured 5–10× more high-frequency motion than with caps) |
| Gripper | Continuous, no rate limit or guard | — |
| Initial pose | ABC-130k bottles ready pose shifted 2.2 cm down (same joints as ABC-DiT mount22), 4 s | With the z +2.2 cm transform the model then observes the exact ABC ready pose |
| Mounting height | `model_from_kinematics` z +2.2 cm | Local base plate is 2.2 cm, verified from J2 axis height and zero-pose fingertip height |
| TCP reference | Stock `grasp_site` | Training EE labels are the official station MJCF `{side}_grasp_site` for every episode, whatever gripper was fitted; the stock chain matches that FK to < 0.1 mm. The former crank-site shift moved the chain ~6 cm off |
| Target z bias | None in the configuration. Optional `policy.adapter.target_z_bias` (e.g. `{offset_m: -0.03, table_z_m: 0.75, full_below_m: 0.9, zero_above_m: 1.0, min_height_m: 0.015}`); a biased target IK cannot reach falls back to the model's own target | tcp_rot20 closed at model z 0.837 (executed within 2 mm of the plan) vs 0.804 in ABC real demos; with −3 cm it grasped |

### Known behavior

- Test-time RTC is ineffective: denoiser input sensitivity ∂x̂/∂x is only approximately 0.01–0.06.
- Repeated requests on the same observation near the ABC ready pose differ by 0.5–3.5 cm over the first 16 rows and 3–10 cm at the chunk end.
  The backend has no `inference_seed`.
- The closing height follows the model frame: removing the 2.2 cm mount compensation raised caps1010's closing height by
  about 2 cm, so frame corrections are not cancelled by the delta actions. A bias on the target only (`target_z_bias`)
  also moves it; with −3 cm tcp_rot20 grasps. Right-to-left handovers still fail often (only 3 of 16 handovers in local
  ABC demos go that way).
- On identical observations ABC-DiT 200k descends within 1 s where tcp_rot20 delays the descent to the chunk tail.

## caps1010 3-task: ABC bottles + pen caps + unscrew caps

EMA weights at step 160k of a training-provider checkpoint (bottles real/sim, insert pen caps, unscrew bottle caps, about one
third each; unfrozen DINOv2-L; CLIP v2 task vector in the DiT AdaLN). Bundle `checkpoints/finetuned/sapolicy/caps1010_3task_tcp/`
(Git-ignored; symlinks to the downloaded `last.ckpt`, `resolved_config.yaml`, `action_normalizer.pt`, `task_clip_table_v2.pt`,
DINOv2 init `backbone.pth`, and the extracted `source/SpatialAlignVLA_dinov3abc1003` snapshot).

The checkpoint stores the **sim** bottles task vector as its default; the server configuration sets `task_cond_table` (local
copy of the training table) and `task_cond_name: put_the_plastic_bottles_in_the_bin`, which the XPolicyLab adapter copies
into the model after loading. The action head compiles on the first request (minutes, longer than the runtime's 10 s
request timeout), so warm the server up with a request before starting a rollout. Inference-time RTC was ported into
the snapshot from tcp_rot20 (`action_head/rtc.py`, DiT `rtc_condition` and guided sampling; `dit.py.orig` is the original);
`validate_checkpoint --rtc` passed (zero-mask RTC identical to ordinary sampling, conditioned error 2.6–10× lower,
RTC forward about 200 ms because the guided branch runs uncompiled). RTC runtime configuration:
`yam_sapolicy_caps1010_3task_rtc_z22.yaml` (tcp_rot20: `yam_sapolicy_tcp_rot20_rtc_z22.yaml`).

```bash
envs/sapolicy/.venv/bin/python -m manimux.servers.sapolicy \
  --config manimux/configs/policy/sapolicy/yam/caps1010_3task_tcp/ema-160k.yaml
envs/yam/.venv/bin/manimux run \
  --config-list manimux/configs/lists/sapolicy-custom-standalone.yaml --experiment-id put_bottles_caps1010_3task_async48_z22
```

Same deployment as tcp_rot20 above, without a z bias. On 2026-10-15/16 grippers closed at 0.800–0.818 (ABC demos 0.804),
both arms grasped, and inference took about 55 ms. Removing the 2.2 cm mount compensation raised the closing height to
0.83–0.84 and no grasp held, so the model does use the absolute height: keep `model_from_kinematics` z +2.2 cm.

## Offline validation

### Using other SAPolicy training checkpoints

For each checkpoint, configure `model_path`, its resolved training `cfg_file`, matching
`normalizer_path`, `backbone_path`, and `checkpoint_source`. To match a training source snapshot,
point `sapolicy_root` to it. The adapter retains its normalization and TCP pose wrappers,
importing only the model core from that snapshot. Changing source snapshots requires a new model process.

Joint TCP + action training may use nested `dataset_opts` lists. The adapter uses
`combined_loader_opts` to select the final action branch, checks that all its datasets agree on action semantics,
image preprocessing, and camera order, then uses it for inference. Strict weight loading rejects missing or extra parameters.
Training snapshots without RTC declare only ordinary sampling; the runtime refuses an `rtc` configuration for them.

The following commands still use MV51 as an example; substitute the new checkpoint's own configuration and weight paths.

```bash
envs/yam/.venv/bin/python -m pytest -o addopts='' -q \
  third_party/XPolicyLab/policy/SAPolicy/tests tests/unit/test_sapolicy_xpl_model.py \
  tests/unit/test_sapolicy_xpl_transport.py tests/unit/test_sapolicy_decode_timing.py
third_party/XPolicyLab/policy/SAPolicy/.venv/bin/python -m pytest -o addopts='' -q \
  third_party/XPolicyLab/policy/SAPolicy/tests
third_party/XPolicyLab/policy/SAPolicy/.venv/bin/python -m XPolicyLab.policy.SAPolicy.validate_checkpoint \
  --model-config manimux/configs/policy/sapolicy/yam/teleopMV51/raw.yaml \
  --checkpoint "$PWD/checkpoints/finetuned/sapolicy/teleopMV51" --rtc \
  --output data/sapolicy/checkpoint-validation.json
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config manimux/configs/experiments/sapolicy-custom/put_bottles/mv51_async25_braking_capped.yaml
```

The final command requires the model service and sends only synthetic images and configured joint states; it opens no cameras and controls
no robot. See the policy README for standard debug, encoded images, batching, native data processing, and training entry points.

Validation on 2026-09-13 covered a fresh environment installation, actual GPU forward inference, numerical equivalence between standard and legacy
action formats, the shared server, offline YAM IK, batch/reset, native ABC reading, and
one GPU training step with checkpoint saving on synthetic data. After the standard-interface migration, one Top hardware rollout was
ended and saved by the operator; it still used old speed limits and had no task-success annotation.

Subsequent offline smoother/RTC validation reduced overlap error for three fixed seeds with real weights.
Zero-weight RTC was value-identical to ordinary sampling; reset and condition clearing passed. Shared debug with ordinary images,
encoded-image batches, and RTC → YAM IK for all three views passed. With synthetic observations, RTC forward inference took
approximately 172–183 ms and warm requests plus dual-arm IK about 310 ms. The new smoother/RTC had not yet run a hardware rollout in this record.
