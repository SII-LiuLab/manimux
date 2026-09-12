# AAC on Pi05 Reproduction Record

## 0. Status

- **Recorded:** 2026-08-21.
- **Target:** Pi05 JAX checkpoints through XPolicy + ManiMux AAC.
- **Configs:** `configs/pi05/yam/infra/aac.yaml` and
  `configs/pi05/yam/infra/pick-red-ball-box/aac-step1000.yaml`.
- **Current gate:** source, contract, shape, regression and real `N=20` GPU forward passed. YAM
  hardware rollout has not been run for Pi05 AAC.
- **Safety boundary:** Codex did not start the Pi05 server, cameras, CAN, preflight or robot.

This document supplements [`aac.md`](aac.md), which is authoritative for the official AAC equations,
candidate selectors and the YAM joint-to-incremental-EE adaptation.

## 1. Upstream Boundary

The Pi05 model is the vendored Physical Intelligence OpenPI implementation under
`XPolicyLab/policy/Pi_05/openpi/`, captured by XPolicyLab revision
`9fff42bf2681379e2b6673c79893db14f13fec0b`. The official OpenPI entry points retained here are:

| Responsibility | File |
|---|---|
| policy transforms and output unnormalization | `openpi/src/openpi/policies/policy.py` |
| Pi0/Pi05 prefix encoding and flow denoising | `openpi/src/openpi/models/pi0.py` |
| checkpoint/data/action transforms | `openpi/src/openpi/policies/policy_config.py` and training config |

The model input transforms, RGB handling, state normalization, action unnormalization, absolute/delta
repacking, flow velocity prediction, Euler update and `num_steps` are unchanged. AAC adds one explicit
multi-sample hook; it does not replace the official inference algorithm.

## 2. Pi05 Multi-Sample Hook

For one transformed observation and `N > 1`:

```text
official preprocess_observation
  -> official embed_prefix once
  -> official prefix KV cache once
  -> repeat transformed observation batch N times
  -> repeat prefix mask on batch axis
  -> repeat KV cache on its batch axis
  -> draw N independent Gaussian action-noise tensors
  -> unchanged official flow denoise loop
  -> official output transforms/unnormalization for all N samples
```

The expensive image/language prefix is therefore evaluated once. The action suffix and denoising
batch are evaluated with size `N`, matching AAC's shared-backbone multi-sample requirement. The JAX
call treats `num_samples` as a static argument because it changes array shapes; changing `N` causes a
new JIT compilation.

The hook refuses:

- `num_samples <= 0`;
- multi-sample inference with more than one input observation;
- multi-sample inference combined with RTC conditioning;
- PyTorch Pi0/Pi05 checkpoints, which do not yet implement this hook;
- returned candidate tensors whose shape or finiteness does not match the checkpoint contract.

Normal `Policy.infer(...)`, Pi-guided RTC and ACT temporal ensembling do not pass `num_samples`; they
continue through the original single-sample callable.

## 3. XPolicy Contract

`XPolicyLab/policy/Pi_05/model.py::get_action_aac` accepts only:

```json
{"mode": "aac", "num_samples": 20}
```

It invokes the OpenPI policy once and returns `N` native candidate chunks. Each candidate is decoded
with the same `unpack_robot_state` path used by normal Pi05 inference. Entropy, FK, motion threshold,
candidate selection and truncation stay outside the model in ManiMux.

The WebSocket capability is advertised only because `get_action_aac` exists. This makes an AAC config
fail during capability negotiation rather than silently fall back to ordinary inference.

## 4. Checkpoint Contracts

### Robocurve 16-step checkpoint

```text
server: configs/pi05/yam/server/finetune.yaml
infra:  configs/pi05/yam/infra/aac.yaml
output: N x 16 x 14 absolute joint positions
dt:     1/30 s
```

### Local red-ball 50-step checkpoint

```text
server: configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
infra:  configs/pi05/yam/infra/pick-red-ball-box/aac-step1000.yaml
output: N x 50 x 14 absolute joint positions after official output transforms
dt:     1/30 s
```

The local model was initialized from official `pi05_base` and uses its own checkpoint-matched
`yam_pick_red_ball_box_v1` norm stats. Those model stats remain separate from AAC's scoring-only EE
stats at `src/manimux/integrations/xpolicylab/norm_stats/yam_60ep_ee_increment.json`.

## 5. YAM Scoring Adaptation

Both Pi05 checkpoints output absolute joints, while official AAC scores 7D incremental EE actions.
ManiMux therefore applies the same adaptation documented for GR00T:

```text
N native absolute-joint chunks
  -> YAM FK for left/right arms
  -> measured-to-first and action-to-next EE increments
  -> fixed YAM min-max for entropy/selectors
  -> official entropy elbow + motion floor + candidate selector
  -> execute the selected original joint prefix
```

No IK result replaces model output. The stats file records separate 16-step and 50-step motion
distributions. Both initial configs deliberately use the same physical threshold `0.2`; it is a
controlled starting value, not a hardware-validated optimum.

## 6. Isolation Matrix

| Pi05 experiment | Server | Runtime request | Changed by this hook |
|---|---|---|---|
| ManiMux | same matching checkpoint | `mode=default` | no |
| ACT temporal ensemble | same matching checkpoint | repeated `mode=default` | no |
| Pi-guided RTC | same matching checkpoint | `mode=rtc` with condition | no |
| AAC | same matching checkpoint | `mode=aac`, `num_samples=N` | yes, isolated branch |

Switching algorithms requires only the matching infra config. It does not rename or duplicate the
Pi05 server config.

## 7. Validation Commands

Offline tests only:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m pytest -q \
  XPolicyLab/policy/Pi_05/openpi/src/openpi/models/pi0_test.py \
  XPolicyLab/policy/Pi_05/openpi/src/openpi/policies/policy_test.py

XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m pytest -q \
  XPolicyLab/tests/unit/test_pi05_aac_adapter.py \
  XPolicyLab/tests/unit/test_ws_infer_sampling.py
```

Operator-run real GPU gate, after starting the matching Pi05 server:

```bash
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/pi05/yam/infra/pick-red-ball-box/aac-step1000.yaml
```

Pass criteria are `aac` capability, `N=20` finite native candidates, a finite selected `[K,14]`
chunk, `2 <= K <= H`, AAC metadata, stable repeated memory use and recorded first-call/steady latency.
The first request includes JAX compilation and must not be reported as steady inference latency.

### Real-robot command

AAC reuses the matching Pi05 model server; it does not use a separate checkpoint or server process.
After the hardware-free warm-up above and the normal camera, CAN, achieved-state, start-pose and
emergency-stop checks, only the operator runs:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/pick-red-ball-box/aac-step1000.yaml
```

Stop with one `Ctrl-C` and wait for the configured Home return and episode save. Do not stop the
model or camera service before the runtime finishes shutdown.

## 8. Known Gaps

1. The official AAC release did not evaluate Pi05 or YAM; this is a model/backend and embodiment
   adaptation.
2. One observed 4090 server run compiled the `N=20`, `H=50` JAX shape in `7036.7 ms`; three warm
   requests then completed in `530.1`, `509.9` and `515.4 ms`. Long-run memory stability is not yet
   characterized.
3. `N=20` multiplies action-head/denoising batch work even though the image/language prefix is shared.
4. Synchronous AAC holds after a selected chunk ends while the next candidate batch is generated.
5. The PyTorch OpenPI backend deliberately does not advertise this capability yet.
6. Real-robot behavior must be recorded separately for the 16-step and 50-step checkpoints.

## 9. Real GPU Evidence

On 2026-08-21 the operator ran the local 50-step red-ball checkpoint through the real XPolicy server
and ManiMux AAC bridge. The first response selected a finite `[31,14]` joint chunk with
`entropy_elbow=23`, `motion_floor=31` and `chunk_id=0`; total round trip was `7036.7 ms` because JAX
compiled the new static `N=20` shape.

Without restarting the server, three subsequent requests measured:

| Request | Round trip | Selected K | Entropy elbow | Motion floor |
|---:|---:|---:|---:|---:|
| 1 | `530.1 ms` | 50 | 49 | 50 |
| 2 | `509.9 ms` | 40 | 18 | 40 |
| 3 | `515.4 ms` | 50 | 38 | 50 |

This verifies the real Pi05 `N=20` GPU path and shows that batching/reusing the prefix reduces steady
cost far below 20 independent model calls. The operator subsequently ran this 50-step configuration
on YAM and observed severe visible pauses: the synchronous runtime holds after the selected prefix
ends while the next roughly `0.51 s` candidate batch is produced. This is hardware execution evidence,
not task-success evidence, and it confirms that the current AAC path is too laggy for responsive
control. After every server restart, the operator must run one hardware-free AAC probe before starting
the robot so the first JIT compile cannot consume the runtime's five-second inference deadline.

## 10. Reviewer Checklist

- [ ] Prefix embedding and KV cache are computed before batch expansion.
- [ ] KV cache repeats on batch axis `1`, not layer axis `0`.
- [ ] Each sample receives independent Gaussian initial noise.
- [ ] Official denoising and output transforms remain unchanged.
- [ ] Default, RTC and ACT paths never pass `num_samples > 1`.
- [ ] XPolicy returns all native candidates and ManiMux owns selection.
- [ ] Checkpoint norm stats and AAC EE stats remain separate.
- [ ] The active server config matches the selected 16/50-step infra config.
- [ ] Offline, real GPU and hardware evidence are labeled separately.
