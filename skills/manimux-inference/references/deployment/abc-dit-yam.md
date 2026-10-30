# ABC-DiT + YAM runbook

## Current deployment

Official [ABC-DiT](https://abc.bot) (XL, 2.02B parameters) connects through `third_party/XPolicyLab/policy/ABC_DiT` and shared
`xpolicylab_ws`. Like Pi05, it uses the generic `JointAdapter` without a dedicated ManiMux adapter.
Model loading, image preprocessing, normalization, CLIP instruction encoding, and sampling reside in that policy directory. ManiMux owns cameras,
observation mapping, scheduling, execution, and recording. For installation and `deploy.yml` keys, see
[ABC_DiT README](../../../../third_party/XPolicyLab/policy/ABC_DiT/README.md).

| Parameter | Setting |
| --- | --- |
| Checkpoint | `checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt` (official multitask pretraining for 200k steps, no local fine-tuning) |
| SHA-256 | `79754ece…2ba175ddce41d018c9f51ba7`; identical to `{sapolicy_workspace}/checkpoints/sapolicy/abc_dit_xl_200k_model.pt` |
| Normalization | XDOF `norm_stats` embedded in the checkpoint (state / actions z-score) |
| Input | Three RGB views: `top` ← `d405_front`, `left` ← `d405_left`, `right` ← `d405_right`; 14D joint state |
| Images | 640×480 → aspect-preserving bicubic resize with antialiasing to 224×168, 28 zero rows padded above and below, then ImageNet normalization |
| Output | `30 × 14` absolute joint positions in radians, `[6 joints, gripper]` per arm; gripper 0 closed, 1 open, clipped server-side to `[0,1]` |
| Instruction | CLIP ViT-B/32 text embedding from `run.task_prompt` (overridden by a nonempty recipe `prompt`); official bottles instruction: `throw plastic bottles in bin` |
| Action interval / horizon / sampling steps | 1/30 s / 30 / 10 |
| Noise seed | `inference_seed: 0`, reset on each RESET and advanced across requests |
| Precision | fp32 (`fast_inference: false`, matching official deployment; bf16 fast-mode outputs have quantization increments of approximately 0.008 rad) |
| RTC | Official ABC RTC: training-time action-prefix conditioning (`get_action_paint` → upstream `sample_actions_rtc`, four prefix steps). PiGDM inference-time guidance (`get_action_rtc`) remains available but is unused in the experiment |
| Port / station service | 8530 / `policy_abc_dit` |

Camera bindings are name-based: the checkpoint's attention-pool queries are indexed by `top/left/right`. Swapping wrist views
raised joint error on ABC's validation set from 0.058 rad to 0.122 rad; incorrect wiring silently invalidates model behavior.

## Experiment configurations

Both configurations align execution with Pi05 Joint 30k (Direct executor, unrestricted speed profile, blend 0, 100 Hz,
12000 control steps, 10 layouts × 3 repeats). RTC follows the official ABC bottles deployment command in the abc repository's deploy README:

```text
--diffusion-steps=10 --rtc --rtc-prefix-length=4 --rtc-inference-lead-steps=7
--execute-chunk-dim=16 --prompt='throw plastic bottles in bin'
```

- Serial: `manimux/configs/experiments/{abc-dit-study}/put_bottles/serial_200k.yaml` (station-local; build the study like the [example study](../../../../manimux/configs/lists/example-yam-pi05.yaml))
- RTC: `manimux/configs/experiments/{abc-dit-study}/put_bottles/rtc_200k.yaml`
- Model recipe: `manimux/configs/policy/abc_dit/yam/put-bottles/pretrained-200k.yaml`

| Item | Pi05 Joint 30k | ABC-DiT 200k | Reason |
| --- | --- | --- | --- |
| Horizon | 50 | 30 | Fixed model chunk length |
| Serial K / RTC execution per chunk | 16 / s_min 16 | 16 / 16 | Official `--execute-chunk-dim=16` |
| RTC method | PiGDM inference-time guidance, β 5 | ABC training-time prefix conditioning: prefix 4, lead 7 | Official deployment; see below |
| Initial pose | Zero pose, grippers open | ABC-130k bottles ready pose (same as SAPolicy) | Zero pose is about 3σ from the ABC state mean and never occurs in ABC data |
| `run.task_prompt` | Put the bottles into the bin. | throw plastic bottles in bin | Official bottles deployment instruction |
| Service / identity | `policy` / pi05 | `policy_abc_dit` / abc_dit | — |

ABC hardware data uses two task names: `put_the_plastic_bottles_in_the_bin` and
`throw_plastic_bottles_in_bin`; official deployment uses the latter.

### Mapping official RTC to ManiMux

The upstream `_RTCManager` executes E=16 steps per chunk, observes and starts inference lead=7 steps before the execution window ends,
and uses the last p=4 steps of that window as a hard prefix in rows 0–3 of the new chunk. This conditioning was learned during training; checkpoint
`max_action_prefix: 8` means prefix lengths 0–7. Execution resumes at row 4 of the new chunk.

ManiMux's existing `paint` scheduler (`manimux/runtime/paint.py`) needs no changes: it observes at row 16,
sends the d=7 rows to be executed during inference (rows 16–22), aligns new row 0 to observation time, and trims for actual latency.
`delay_buffer_size: 1000` retains the initial value 7 as the maximum delay estimate, fixing lead at 7. If actual latency exceeds seven steps,
the chunk is rejected (`PAINT response advanced beyond its anchored prefix`); runtime continues the old plan and increases the prediction.
`ABC_DiT.get_action_paint` takes the last four of these seven rows as a hard prefix for upstream `sample_actions_rtc`,
then returns the first three old rows plus the new chunk, truncated to 30 rows. Rows 0–6 exactly match the old plan; new actions begin at row 7,
matching upstream switching time and prefix. Only the first chunk differs: upstream observes at row 9, while this scheduler observes at row 16.
`paint` metadata records `method: abc_action_prefix` and `prefix_length: 4`; this is unrelated to the PAINT paper's inversion sampling.

## Startup

The station file needs `services.policy_abc_dit.endpoint: ws://127.0.0.1:8530` (see
`manimux/configs/local/yam_example.yaml`). Use four terminals, all at the repository root:

```bash
# 1: Cameras
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --config-list manimux/configs/lists/{abc-dit-study}.yaml --experiment-id put_bottles_rtc_200k

# 2: RoboGUI
envs/yam/.venv/bin/python -m manimux.robogui.dashboard --robot yam --host 127.0.0.1 --port 8086

# 3: ABC-DiT model service (fp32 loading takes about 10 s; wait for the port before starting runtime)
envs/abc/.venv/bin/python -m manimux.servers.abc_dit \
  --experiment manimux/configs/experiments/{abc-dit-study}/put_bottles/rtc_200k.yaml

# 4: Runtime (for Serial, use yam_abc_dit_serial_200k.yaml; no model-service restart needed)
envs/yam/.venv/bin/python -m manimux run \
  --config-list manimux/configs/lists/{abc-dit-study}.yaml --experiment-id put_bottles_rtc_200k
```

Adding `--check` to the model command resolves only paths and contracts without loading the model. Serial and RTC share the same model service and identity.

The 8 GB checkpoint is excluded from Git. On a new machine, copy it to `checkpoints/pretrained/abc/abc_dit_xl_200k_model.pt`;
the service verifies its SHA-256 at startup.

`envs/abc` is ABC's isolated environment (Python 3.12, torch 2.11 cu128). On a new machine, create it with
`bash third_party/XPolicyLab/policy/ABC_DiT/install.sh envs/abc/.venv`. CLIP assets live in
`~/.cache/clip/` (`ViT-B-32.pt`, `bpe_simple_vocab_16e6.txt.gz`) and download on first startup if missing.

## Validation record (2026-10-07, RTX 4090)

All tests below used the real `xpolicylab_ws` client → WebSocket → model service → `JointAdapter` decoding:

- Ten validation episodes from the official bottles preview set (eight hardware, two simulation), three frames each, 30 samples total.
  Mean joint error over 30 predicted steps: 0.058 rad versus 0.101 rad for holding still; gripper error: 0.025 / 0.064.
  On the same validation frame, bf16 fast mode and fp32 had essentially identical ground-truth errors (0.0876 / 0.0874 rad).
- RTC: conditioning the first eight steps on ground truth +0.05 rad reduced prefix error from 0.087 to 0.038 rad.
- Latency including transport: default p50 68 ms, RTC 175 ms. In fp32: 166 / 266 ms respectively (in-process).
- Seed: identical observations after RESET produced identical values; runtime `expected_backend` validation passed for both experiments.
- Three local teleop bottle recordings (640×480): zero-shot predictions were no closer to human demonstrations than holding still
  (0.063 versus 0.052 rad, better on only 20% of samples). Maximum first-step deviation from current state was 0.045 rad, without a jump.

These results validate the integration and preprocessing, not physical task success at this station.

## Known limitations

- This is a multitask model pretrained on XDOF workstations, without local fine-tuning. Station differences identified for SAPolicy also apply:
  wrist camera angles (ours nearly horizontal, ABC looking down at the table), gripper appearance, table/background,
  and our additional 2.2 cm YAM base mounting plate (see mounting compensation below).
- Observe initial hardware trials at low speed with the emergency stop ready. For speed limits, use the existing `executor.type: smooth`
  (for example, SAPolicy braking with 0.6 rad/s and 1.5 rad/s²), without altering model outputs.
- fp32 prefix inference takes approximately 174 ms including transport, within the seven-step lead (233 ms). Other GPU processes can push it beyond that window,
  causing chunk rejection.
- ABC human-operation data has p99 per-step joint changes of 0.07–0.24 rad (several rad/s); model actions under the official instruction
  can be equally fast. The Direct executor imposes no speed limit.

## Hardware records

**First trial, 2026-10-07 (PiGDM inference-time RTC, s_min 15, bf16, instruction: put the plastic bottles in the bin)**:
The model mostly hovered. Plan 61 suddenly descended at 2.7 rad/s; plan 62 returned to hovering despite RTC conditioning.
The command jumped 0.353 rad in one handover tick, executed unchanged by Direct, producing a stationary–violent-shake–stationary pattern.
Offline replay of recorded observations showed that PiGDM adherence depended on the noise seed; even β 100 could not restore it.
Official prefix conditioning connected exactly and continued smoothly for all three seeds. bf16 quantization also caused small handover jumps, median 0.0165 rad.
The configuration was therefore changed to official ABC RTC + fp32. Replaying 15 observations from the same rollout through the real client yielded:
rows 0–6 matching the old plan; row-7 seam median 0.0019 rad, maximum 0.014 rad; latency p50 174 ms.

## Mounting-height compensation (experimental)

Our YAM bases sit on 2.2 cm plates; ABC's sit directly on the table. The joint-space model reproduces ABC joint poses:
in the official RTC rollout on 2026-10-07, median base-frame grasp_site height during closing was 0.044 m,
matching 0.045 m in official ABC hardware data (152 tabletop grasps). Our gripper was consequently 2.2 cm higher above the table,
closing above the bottle (opening 0.32–0.45 versus approximately 0.52–0.57 when ABC holds a bottle).

`yam_abc_dit_rtc_200k_mount22.yaml` uses `manimux.policy_adapter.joint_mount:MountOffsetJointAdapter`,
`base_offset_m: {left_arm: [0, 0, 0.022], right_arm: [0, 0, 0.022]}`:

- Request: measured joints and PAINT prefixes (likewise RTC conditions) pass through FK → base-frame z +2.2 cm → IK into model coordinates.
- Action: each model output row passes through FK → z −2.2 cm → IK into robot joints; orientation and gripper remain unchanged on successful solves. Action IK uses the shared `execution.ik.failure_policy`: default `hold_last_valid` holds both arm and gripper for failed points and retries later targets; `reject` rejects the chunk. Observation coordinate conversion still fails explicitly.
- Initial pose: the ABC ready pose is also shifted down 2.2 cm, so the model still sees its ABC ready pose.

Offline checks: each row moves down exactly 22.0 mm, horizontal deviation < 0.03 mm; prefix round-trip error < 2e-3 rad;
seam median 0.0020 rad; p50 latency including conversion 183 ms. Conversion runs on the control thread,
with approximately 15 ms decoding + 6.5 ms request preparation per chunk (`paint` does not support `action_decoding: process`).

**2026-10-07 mounting compensation + official fast inference + official RTC (lead 7)** (`yam_abc_dit_rtc_200k_mount22.yaml`,
bottles lying flat): the operator confirmed all were grasped. Most of the left arm's nine closures occurred at robot base-frame z 0.024–0.035 m (model coordinates approximately
0.046–0.057 m, compared with ABC median 0.045 m). Opening 1.5 s after closing was 0.35–0.53, constrained by the bottle. All 121 chunks were accepted,
with no rejections; obs→commit median 133 ms, maximum 166 ms (3–4 steps of latency), including approximately 30 ms for compensation decoding.
Three preceding issues and fixes: PiGDM RTC handover jumps → official prefix RTC; fp32 latency plus compensation exceeding lead 7
and stalling after PAINT rejection → official fast inference (`FastRTCInferenceGraph`); gripper 2.2 cm too high → mounting compensation.

Still unresolved in this record: after rejection, `manimux/runtime/paint.py` stops requesting permanently if executed steps + delay > horizon.
