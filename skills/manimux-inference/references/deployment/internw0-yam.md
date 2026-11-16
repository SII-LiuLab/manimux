# InternW0-Delta + YAM

## Scope

InternW0-Delta (Wan2.2-TI2V-5B video DiT + action DiT through masked attention, RynnBrain
1.1-2B as a frozen VLM) through the standard XPolicyLab server. It returns absolute 14-D joint
targets (6 joints + gripper per arm) at 30 Hz with horizon 32; ManiMux owns scheduling,
execution and recording. Only Serial and ordinary Async are wired here. RTC, PAINT, BID and
AutoHorizon need sampler hooks the vendored policy does not have; none is claimed.

## What the model sees

Each request carries three frames, not a stream: the **anchor** (first observation after
Start), the **recent** frame 32 policy steps (about 1.07 s) before the request, and the
**current** frame. The anchor and recent frames travel with their joint states and the episode
step. Before 32 steps have elapsed the recent slot is padding (`recent_is_real: false`).
Other inputs: three raw camera images to the frozen VLM, the instruction, current joints.
Past actions are not a condition for this checkpoint.

Time-addressed history lives in the runtime (`manimux/policy_adapter/internw0/memory.py`,
selected through `inference.strategy`), so the server is stateless and any scheduler works.
History starts at the first running tick after Start and is cleared by Pause/Home/Reset.
A request is withheld, never filled with a substitute frame, while the 32-step-old frame is
missing. The XPolicyLab session path (`memory_source: session`) needs one `update_obs` per
executed step and is unusable here.

## Recipe

[`manimux/configs/policy/internw0/yam/latest300ep-step30000.yaml`](../../../../manimux/configs/policy/internw0/yam/latest300ep-step30000.yaml).
It must use the checkpoint's own `dataset_stats.json`; the file shipped under
`third_party/XPolicyLab/policy/InternW0_delta/config/` is the RoboDojo release's and differs.
The model config stays the shipped `eval_model.yaml`: the checkpoint's own `config.yaml` has a
`training_time_rtc` key (disabled) that the vendored `wam_runtime` rejects at construction.
Checkpoint and assets resolve below `paths.checkpoints` (`finetuned/InternW0/...`,
`internw0-assets/{Wan-AI,Alibaba-DAMO-Academy}`; symlinks to the external drive are fine).
`replan_steps: 32` follows the official real-robot deployment (it fixes the recent frame's
position id); the shipped RoboDojo default of 10 would not match.

## Environment

`envs/internw0` (conda, Python 3.11, torch 2.10+cu128, transformers 5.13). Do not install the
prebuilt `causal-conv1d` wheel: it is ABI-incompatible with this torch and breaks the import;
transformers falls back to its torch implementation. Assets: `download_assets.sh` fetches
Wan2.2 from ModelScope, but RynnBrain1.1-2B is only on HuggingFace
(`Alibaba-DAMO-Academy/RynnBrain1.1-2B`); with a SOCKS `ALL_PROXY` set, unset it for the HF
client (httpx needs `socksio` otherwise).

## Measured (RTX 4090, 10 denoising steps, synthetic input, 2026-11-11)

Load 120 s; first call 1.1-20 s (VLM warm-up), so Warmup is required. Steady call 300-360 ms:
VAE 18, VLM 27, 10 action steps 180-240, prefill and glue about 75. Peak 27.9 GB. One
unexplained 1.1-4.2 s call at the fifth replan, after which steady time rises by about 55 ms.
The video K/V cache is exact reuse of the fixed conditions; there is no OpenWAM-style DiT-cache
step skipping, so every denoising step runs the action DiT.

## Launch and validation

[`manimux/configs/lists/internw0-latest300ep-step30000.yaml`](../../../../manimux/configs/lists/internw0-latest300ep-step30000.yaml)
holds Serial and Async for the five tasks (3 repeats per layout). Validate first, without
opening devices:

```bash
L=manimux/configs/lists/internw0-latest300ep-step30000.yaml
envs/yam/.venv/bin/python -m manimux inspect --config-list $L
envs/internw0/bin/python -m manimux.servers.policy --config-list $L --check
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config-list $L --check
```

Then one terminal each:

```bash
envs/internw0/bin/python -m manimux.servers.policy --config-list $L
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config-list $L
envs/yam/.venv/bin/manimux-robogui --robot yam --host 0.0.0.0 --port 8086
envs/yam/.venv/bin/python -m manimux run --config-list $L
```

Entries per task: `{task}_serial` (Serial, 32 steps) and `{task}_async` (single in flight, request
when 0.5 s of plan remains, cap 32). The progress trigger and multi-inflight are refused for
custom strategies, which is why Async here is refill-triggered. Other algorithms are refused at
construction: the policy has no sampler hooks for them.

## Verification

`tests/test_internw0_memory.py` covers time-addressed lookup, request-cadence independence and
the wire fields. With the real checkpoint and fixed seed, the request path matched the stateful
per-step-ack session path bit-exactly (max abs difference 0.0) at episode steps 0, 32 and 64.
