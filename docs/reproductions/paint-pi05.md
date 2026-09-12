# PAINT on Pi05 Reproduction Record

## 1. Scope and Claim Boundary

- **Method:** PAINT, *Start Right, Arrive Right: Asynchronous Execution via Initial Noise Selection*.
- **Primary specification:** [arXiv:2606.19774](https://arxiv.org/abs/2606.19774), Algorithm 1 and Appendices C–F.
- **Official repository:** [htrbao/paint-action-chunking](https://github.com/htrbao/paint-action-chunking).
- **Target backend:** the existing XPolicy Pi05 JAX/OpenPI adapter.
- **Current upstream state:** the official repository exposes documentation and the project page but
  no implementation files. This integration is therefore a **paper reproduction**, not an official
  code port.
- **No-retraining claim:** PAINT uses the existing frozen checkpoint and requires neither PAINT-specific
  training nor a learned correction module.
- **Validation gate:** formula, protocol, adapter, configuration, runtime scheduling, real Pi05 GPU
  execution and one YAM hardware rollout are complete. Steady latency/memory and statistical task
  benefit remain open measurements.

No ManiMux test, forward probe or finite chunk is evidence of task success.

## 2. Paper Algorithm

For an observation `o`, old action chunk `A_old`, execution index `s`, inference delay `d`, horizon
`H` and `N` flow steps, PAINT performs:

1. Sample `x_free` from the normal prior.
2. Run the unmodified policy from `x_free` to obtain `x_naive`.
3. Construct `x_target = concat(A_old[s:s+d], x_naive[d:])`.
4. Starting at the target endpoint, evaluate the same velocity network for `N` backward-Euler steps.
5. Retain the inverted initial-noise prefix and restore `x_free[d:]` as the suffix.
6. Run the unmodified forward sampler once more from the repainted noise.

The total is `3N` velocity evaluations and no VJP or gradient. Repainting changes the **prefix of
initial noise**; it does not replace the suffix with new noise and does not overwrite the final
action chunk after generation.

## 3. Pi05 Time Convention

The paper writes noise at flow time `0` and actions at `1`. The vendored OpenPI Pi05 sampler explicitly
uses the opposite convention:

```text
Pi05 time 1: noise
Pi05 time 0: clean action
```

The equivalent discrete updates are therefore:

```text
forward: x <- x - (1/N) * v(x, o, t),  t: 1 -> 0
inverse: x <- x + (1/N) * v(x, o, t),  t: 0 -> 1
```

This is a coordinate reversal, not a changed algorithm. Both loops execute exactly `N` velocity
calls and use the same `predict_velocity` closure, observation tokens, KV cache and official Pi05
action head.

## 4. Normalization and Action Semantics

The runtime prefix is in YAM robot units:

```text
left 6 joint radians + left gripper
+ right 6 joint radians + right gripper
```

`XPolicyLab/policy/Pi_05/model.py` pads the `d` prefix rows to the model horizon only so the unchanged
OpenPI input transform can process a full action tensor. `Policy.infer` places that tensor under the
official `actions` transform key. The resulting normalized/padded model tensor becomes
`paint_action_condition`; only the first `d` rows participate in Algorithm 1.

The sampled result then passes through the unchanged official output transform. PAINT does not use
AAC EE statistics, does not change checkpoint norm stats, and does not convert the generated joint
trajectory to EE space.

## 5. Ownership Boundaries

### XPolicy / Pi05 sampler

`XPolicyLab/policy/Pi_05/openpi/src/openpi/models/pi0.py` owns:

- fresh Gaussian noise;
- the naive forward pass;
- target endpoint construction;
- backward-Euler inversion;
- prefix-only noise repainting;
- the final ordinary forward pass.

`XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/policy.py` owns the official input/output
transforms and passes the normalized condition into the sampler.

### XPolicy adapter and wire

`XPolicyLab/policy/Pi_05/model.py` exposes `get_action_paint` and validates `d x native_action_dim`.
The WebSocket server advertises `paint` only when that method exists. The request payload is:

```text
mode: paint
action_prefix: A_old[s:s+d]
delay_steps: d
```

### ManiMux runtime

`src/manimux/runtime/paint.py` owns only deployment timing:

- wait until the old chunk has reached `s`;
- forecast `d` from completed request latency;
- copy exactly `A_old[s:s+d]`;
- submit asynchronously while the old chunk continues;
- let `ActionTimeline.commit` trim the generated prefix that elapsed during inference;
- reject a response when actual trimming exceeds the prefix length used during generation.

PAINT configs require `blend_steps: 0`. Otherwise Timeline would rewrite the old chunk at commit,
while the next PAINT request still conditions on its pre-blend values. Smooth/MPC execution limits and
Safety remain explicit outer hardware layers; their effect must be reported separately from PAINT's
chunk-space prefix consistency.

RobotDriver, SensorDriver, Smooth/MPC Executor, Safety, Recorder and Viewer remain unchanged.

## 6. Delay Forecast Adaptation

The paper defines `d = floor(delta / action_dt)` from inference wall time. A live request cannot know
its own final `delta` before it finishes. ManiMux therefore supplies the method's required `d` using
the maximum of a bounded history, initialized from config. This forecast is infrastructure around
Algorithm 1, not a modification of its inverse-flow computation.

The runtime enforces the paper feasibility region:

```text
d <= s <= H - d
```

If the measured response age would trim more than conditioned `d`, the response is rejected. This is
deliberately stricter than switching to an unanchored suffix.

## 7. Experiment Configuration

The initial Pi05 experiment is:

```text
server: configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
infra:  configs/pi05/yam/infra/pick-red-ball-box/paint-step1000.yaml
H:      50
N:      10
s:      12
d0:     10
point rate: 30 Hz
robot control: 100 Hz
```

The server config, checkpoint and checkpoint norm stats are identical to the ordinary ManiMux, ACT
and AAC comparisons for this step-1000 policy. Only the inference strategy changes.

No PAINT config is provided for the 16-step Robocurve Pi05 checkpoint: a roughly 300 ms `3N` sampler
at 30 Hz would consume about nine policy steps, violating `2d <= H` for `H=16`.

### 7.1 Operator commands

Start the same step-1000 Pi05 model server used by the ordinary ManiMux baseline:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
```

After every server restart, warm the PAINT-specific JAX shape without camera, CAN or robot commands:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/pi05/yam/infra/pick-red-ball-box/paint-step1000.yaml
```

Record at least three warmed `round_trip_ms` values. Then complete the normal camera, CAN,
achieved-state, start-pose and emergency-stop checks. Only the operator starts the real robot runtime:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/pick-red-ball-box/paint-step1000.yaml
```

Stop with one `Ctrl-C` and wait for the configured Home return and partial episode save. Do not stop
the model or camera service until runtime shutdown completes.

## 8. Validation Ladder

1. **Formula unit test:** constant velocity field recovers the requested prefix and preserves the
   naive suffix under repainting.
2. **Policy test:** the robot-unit condition passes through the official input transform and reaches
   the JAX sampler with the correct `d`.
3. **Adapter test:** Pi05 validates/pads the prefix and reports `3N` model evaluations.
4. **Protocol test:** WebSocket HELLO advertises `paint`; INFER dispatches only to
   `get_action_paint`.
5. **Runtime test:** ManiMux sends the exact old-chunk slice `A[s:s+d]` and rejects prefix shortfall.
6. **Hardware-free GPU probe:** operator starts the existing Pi05 server and runs the documented
   forward probe. Record first-compile and at least three steady latencies.
7. **Hardware gate:** operator separately verifies cameras, CAN, achieved state, start pose and
   emergency stop before any YAM command.
8. **A/B rollout:** compare the same checkpoint and task under ordinary ManiMux, RTC and PAINT.

## 9. Non-Regression Contract

- Requests without `paint_action_prefix` use the existing single-sample callable.
- RTC uses its existing action condition, weights and VJP branch.
- AAC uses its existing multi-sample branch.
- ACT temporal ensembling remains runtime-only.
- PAINT and RTC/AAC are mutually exclusive in one request.
- The base checkpoint, norm stats, camera contract, joint order, gripper semantics, horizon and action
  period are not changed.

## 10. Open Gates

- Compare this paper reproduction against upstream source if the authors publish it.
- Measure first-call compilation, steady latency and memory on the local Pi05 checkpoint.
- Confirm the rolling `d` remains feasible for `H=50` under real network/camera load.
- Inspect recorded prefix consistency and rejection counts before claiming smoothness or task benefit.

## 11. First GPU Evidence

The operator ran the hardware-free PAINT forward probe against the local step-1000 Pi05 server. The
first request returned finite native and canonical shapes `[50, 14]`, `delay_steps=10`,
`num_steps=10`, `model_evaluations=30`, and `inversion=backward_euler`. Round trip was `6087.2 ms`.
This request included the first JAX compilation, so it verifies the real GPU execution path and wire
contract but does not establish steady latency. At least three warmed requests are still required.

## 12. First YAM Hardware Evidence

On 2026-08-21 the operator ran the step-1000 Pi05 PAINT configuration on dual YAM after the real GPU
probe. The runtime completed physical execution, and the operator reported that PAINT's effect was
very good, with markedly better visible continuity than the preceding baselines. This upgrades the
method to hardware exercised. It does not yet establish task success rate, steady latency, memory
stability or statistical superiority; those require saved-episode analysis and repeated controlled
A/B trials.
