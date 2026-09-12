# AutoHorizon on Pi05/YAM Reproduction Record

## 1. Scope and Claim Boundary

- **Method:** AutoHorizon, *VLA Knows Its Limits: Adaptive Execution Horizons for Robot Policies*.
- **Paper:** [arXiv:2602.21445](https://arxiv.org/abs/2602.21445).
- **Official repository:** [hatchetProject/AutoHorizon](https://github.com/hatchetProject/AutoHorizon).
- **Pinned upstream commit:** `c7504f1756109103f2cfcc2e23f1b1a23841c885`.
- **Official backend:** converted PyTorch Pi0.5 over LIBERO.
- **ManiMux backend:** the existing JAX Pi05/YAM checkpoint and official OpenPI transforms.
- **No-retraining claim:** the checkpoint, norm stats and model parameters are unchanged.
- **Current gate:** implementation, targeted contract tests, Pi05 GPU forward, WebSocket transport and
  YAM hardware execution are verified. The faithful synchronous cadence produced visible inference
  holds on YAM; framework parity and task benefit remain validation gates.

This is a **JAX attention-path port of the official AutoHorizon implementation**, not a claim that the
official repository publishes a JAX implementation. The official bidirectional soft-pointer logic and
defaults are preserved. JAX replaces only the framework-specific mechanism used to expose the same
post-softmax action self-attention tensor.

## 2. Official Algorithm and Source Audit

The official code was read at the pinned commit from:

- `src/openpi/models_pytorch/pi0_pytorch.py`: third-denoising-step attention capture, layer/head
  reduction and `bidir_soft_pointer`;
- `src/openpi/models_pytorch/gemma_pytorch.py`: expert attention transport;
- `src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py`: post-softmax
  attention return;
- `src/openpi/policies/policy.py`: `actions` plus `e_steps` wire output;
- `examples/libero/main.py`: synchronous `action_chunk[:e_steps]` execution.

For a prediction horizon `p`, AutoHorizon uses the action-to-action post-softmax attention matrix
`S in R^(p x p)`. The official implementation:

1. reads all action-expert layers and heads at the third Euler denoising step;
2. averages layer, batch and head axes into one `p x p` matrix;
3. row-normalizes the matrix;
4. computes a monotone forward soft pointer and its first reliable plateau;
5. repeats the same operation after reversing both attention axes;
6. returns the full horizon when forward and backward coverage meet, otherwise the forward horizon;
7. executes exactly the returned prefix and synchronously queries again when it is exhausted.

Official defaults are fixed in this integration:

| Parameter | Value | Meaning |
|---|---:|---|
| attention sampling step | `3` | Third denoising step, one-based |
| `hold_thr` | `0.3` | Pointer-increment plateau threshold |
| `max_entropy_q` | `0.9` | Reliable-row entropy quantile |
| `run_len` | `1` | Required consecutive plateau rows |
| pointer | bidirectional | Official active code path |

The server does not accept config overrides for these values. A future parameter sweep must use a
separate explicitly non-default experiment profile rather than silently changing this reproduction.

## 3. JAX Attention Equivalence

The current Pi05/YAM deployment uses the original JAX checkpoint. Converting it to PyTorch would add a
derived checkpoint and a separate Transformers-patched execution path. Instead, this integration
exposes the corresponding tensor directly from the JAX Gemma action expert:

```text
official PyTorch: [layer, batch, head, action_query, prefix+action_key]
JAX port:         [layer, batch, head, action_query, prefix+action_key]
```

Both paths use:

- the action expert, not PaliGemma vision-language attention;
- post-mask, post-softmax attention probabilities;
- all expert layers and all query heads;
- only the final `p` action-key columns;
- the third denoising step;
- a mean over layer, batch and head axes;
- the official soft-pointer logic with the official defaults.

The expected framework difference is numerical only: JAX and PyTorch may produce slightly different
floating-point values. Because AutoHorizon contains a hard plateau threshold, a matrix close to the
threshold may yield a different horizon by one or more steps. Until a converted PyTorch checkpoint is
run on identical observations and noise, framework parity remains unverified and must not be claimed.

## 4. Ownership Boundaries

### XPolicy Pi05 sampler

- `openpi/models/gemma.py` optionally returns post-softmax attention without changing its ordinary
  output path.
- `openpi/models/pi0.py` captures the third-step action-to-action matrix only when
  `return_attention=True`.
- `openpi/models_pytorch/autohorizon_official.py` holds the pinned official soft-pointer logic.
- `openpi/policies/policy.py` calculates `execution_steps` before applying ordinary output transforms.
- `XPolicyLab/policy/Pi_05/model.py` exposes `get_action_autohorizon` and returns the full decoded
  `50 x 14` YAM joint chunk plus method metadata.

Default, RTC, AAC and PAINT continue calling the existing sampler paths and do not request attention.

### XPolicy WebSocket

The server advertises `autohorizon` only when the loaded model provides the method. One request is:

```yaml
sampling:
  mode: autohorizon
```

The response contains the full action chunk and:

```yaml
autohorizon:
  execution_steps: 1..50
  attention_step: 3
  forward_horizon: 1..50
  backward_horizon: 1..50
  method: bidir_soft_pointer
  framework: jax_attention_port
  upstream_commit: c7504f1756109103f2cfcc2e23f1b1a23841c885
```

### ManiMux Runtime

`src/manimux/runtime/autohorizon.py` owns only official execution cadence:

- wait until the previous selected prefix is exhausted;
- request one new full chunk from the latest observation;
- reject missing or out-of-range `execution_steps`;
- retain exactly `chunk[:execution_steps]`;
- execute that prefix through the configured Executor and RobotDriver;
- query again only after the prefix ends.

There is no asynchronous prefetch, temporal ensemble or seam blend. `blend_steps` must be zero.
Smooth/MPC limits and Safety remain explicit outer real-robot layers and are not part of the paper.

## 5. Configuration

```text
server: configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
infra:  configs/pi05/yam/infra/pick-red-ball-box/autohorizon-step1000.yaml
```

The server config is unchanged because AutoHorizon reuses the same JAX checkpoint and matching norm
stats. The infra config selects only `execution.runtime: autohorizon` and disables seam blending.

## 6. User-Run Validation

Start the unchanged Pi05 server:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
```

Before any robot process, run the forward probe:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/pi05/yam/infra/pick-red-ball-box/autohorizon-step1000.yaml
```

The probe must report a finite `50 x 14` native/canonical chunk and an `autohorizon` metadata block
whose `execution_steps` lies in `[1, 50]`. First-call JAX compilation is not steady-state latency.

On 2026-08-24, the corrected current-checkout server produced:

- ordinary Pi05: finite `50 x 14`, first compiled round trip `5846.9 ms`;
- AutoHorizon first call: finite `50 x 14`, `execution_steps=21`, `3859.5 ms`;
- AutoHorizon warmed call: finite `50 x 14`, `execution_steps=50`, `117.0 ms`;
- ManiMux AutoHorizon runtime tests: `2 passed`;
- XPolicy adapter, official selector and WS sampling tests: `12 passed` when the current checkout was
  placed first on `PYTHONPATH`.

The two probes use different sampled noise, so different selected horizons are expected and are not a
framework-parity measurement.

### Real-robot cadence evidence

On 2026-08-24, the operator ran this configuration on dual YAM and observed clearly periodic
stop-and-go motion. This matches the official execution loop rather than indicating a missing action
chunk: AutoHorizon requests a new prediction only after the selected prefix is exhausted. With
`execution_steps=21`, `dt=33.3 ms` and a warmed `116.5 ms` request, the robot receives about `0.67 s`
of references and then holds its last command during the next request.

The official LIBERO client also calls `infer()` only when its action deque is empty. Simulator time
does not advance while that blocking request runs; a physical robot control clock does, so the same
cadence becomes a visible hold. AutoHorizon selects how much of a chunk to trust. It does not overlap
inference, condition a new denoising trajectory on already executed actions, or guarantee a continuous
chunk boundary. Those are separate RTC/PAINT concerns.

Only after reviewing the forward output and completing the normal YAM physical preflight may the user
run:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/pick-red-ball-box/autohorizon-step1000.yaml
```

## 7. Validation Matrix

| Gate | Status | Evidence required |
|---|---|---|
| Source audit | complete | pinned repository, paper and source list above |
| Official selector unit contract | passed | targeted XPolicy unit suite |
| WebSocket capability/dispatch | passed | targeted XPolicy unit suite |
| ManiMux truncate/synchronous cadence | passed | `2 passed` |
| Pi05 JAX GPU forward | passed | finite `50 x 14`, metadata, warmed `117.0 ms` |
| PyTorch/JAX parity | pending | identical checkpoint, observation and noise comparison |
| YAM hardware | passed for execution | operator observed functional but visibly stop-and-go motion |
| Task benefit | unverified | repeated controlled trials, not one finite chunk |

## 8. Reviewer Checklist

- [ ] Official commit remains pinned and upstream changes are reviewed explicitly.
- [ ] Ordinary Pi05 output is unchanged when attention is not requested.
- [ ] Attention comes from the action expert and retains all layers and heads.
- [ ] Only action-key columns are passed to the selector.
- [ ] The selected sampling step is the official third denoising step.
- [ ] `hold_thr=0.3`, `max_entropy_q=0.9`, `run_len=1` and the bidirectional pointer remain unchanged.
- [ ] XPolicy returns a full chunk; ManiMux alone truncates it for execution.
- [ ] Runtime remains synchronous and does not prefetch from a stale observation.
- [ ] `blend_steps=0` and no other inference method is composed implicitly.
- [ ] GPU and hardware claims remain separate from unit-contract claims.
