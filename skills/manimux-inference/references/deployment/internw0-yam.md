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

## Configuration and environment

Start with the [serial example](../../../../manimux/configs/examples/yam-xpolicylab/internw0-serial.yaml)
for the request-memory adapter, strategy and camera mapping. In a private copy of
its recipe, select the checkpoint's own weights and dataset statistics. Keep
training configuration, asset paths and study lists local. Bind
`${paths.checkpoints}` in the station file. Select a model configuration compatible
with the vendored runtime and verify its accepted fields before construction. Set
`replan_steps: 32` for the history contract described above.

Use an isolated model environment compatible with the vendored InternW0 source and
its VLM/video assets. Do not replace packages in an active model environment.

## Launch and validation

Set the private deployment and station paths, then validate without opening devices:

```bash
CONFIG=/path/to/private/deployment.yaml
STATION=/path/to/private/station.yaml
envs/yam/.venv/bin/python -m manimux inspect --config "$CONFIG" --local "$STATION"
envs/internw0/bin/python -m manimux.servers.policy --config "$CONFIG" --local "$STATION" --check
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config "$CONFIG" --local "$STATION" --check
```

After authorization, run the services in separate terminals using the same variables:

```bash
envs/internw0/bin/python -m manimux.servers.policy --config "$CONFIG" --local "$STATION"
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config "$CONFIG" --local "$STATION"
envs/yam/.venv/bin/manimux-robogui --robot yam --host 127.0.0.1 --port 8086
envs/yam/.venv/bin/python -m manimux run --config "$CONFIG" --local "$STATION"
```

The operator performs Prepare, Warmup, Start and Finish. Reuse compatible running
services. Custom history strategies use refill-triggered, single-inflight Async;
progress-triggered and multi-inflight scheduling are rejected. Unsupported sampler
algorithms are rejected at construction.

## Verification

Offline validation should cover time-addressed lookup, request-cadence independence,
reset boundaries and wire fields. Verify fixed-seed request/session parity against
the selected checkpoint without constructing physical drivers. Keep benchmark
measurements and study results local; these checks do not establish physical success.
