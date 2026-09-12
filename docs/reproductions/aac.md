# Adaptive Action Chunking Reproduction Record

## 0. Status Snapshot

- **Recorded:** 2026-08-21.
- **Target:** GR00T N1.7 and Pi05 YAM checkpoints through XPolicy + ManiMux.
- **Current gate:** source/contract/formula/config/unit tests passed; real `N=20` GPU forward and YAM hardware are not yet verified for AAC.
- **Execution safety boundary:** this record contains commands, but Codex did not start a server, camera, CAN process, preflight or robot.
- **Claim boundary:** this is an embodiment adaptation of official AAC, not evidence that the AAC authors evaluated GR00T N1.7 or YAM.

## 1. Upstream Pin

Two upstream repositories define the executable method:

| Role | Repository | Pinned commit | Relevant files |
|---|---|---|---|
| Multi-sample policy | `Adaptive-Action-Chunking/gr00t-multi-sample` | `11e926b0f34cf6acfcb92c0fe6127a1bdc7b856a` | GR00T action head and policy inference |
| Client selection | `Adaptive-Action-Chunking/robocasa` | `fed3e6b5eb348160dd0570f326f726758fee9056` | `robocasa/demos/action_optimization/action_entropy_v2.py`, `action_sampler.py`, `inference_client_action_entropy.py`, `help_function.py` |

The upstream executable setup is GR00T N1.5, one observation, `N=20` stochastic samples, a
16-step 7D action and RoboCasa/LIBERO end-effector control. The action fields are:

```text
action.end_effector_position[3]
action.end_effector_rotation[3]
action.gripper_close[1]
```

The official client sends each 7D action directly to the RoboCasa robot action vector. The official
method therefore scores the same end-effector action representation that the simulator executes.

## 2. Official Multi-Sample Generation

Official AAC does not run the visual backbone 20 independent times. For one observation it:

1. runs the backbone and state encoder once;
2. expands the resulting features from batch `1` to batch `N`;
3. samples independent Gaussian initial action noise for each expanded batch item;
4. runs the unchanged denoising loop in one batched action-head call;
5. decodes all `N` normalized chunks to their environment action fields.

The required invariant is that feature expansion happens **after** the expensive observation backbone
and **before** initial action noise. Expanding images before the backbone would reproduce the output
distribution but not the intended compute path.

## 3. Official Chunk-Length Formula

For candidate `n`, time `t`, normalized position `p`, normalized rotation `r` and decoded binary
gripper `g`, official code computes separate sample entropies:

```text
H_pos(t)  = 0.5 * (3 * log(2*pi*e) + logdet(cov_n(p[n,t]) + eps*I))
H_rot(t)  = 0.5 * (3 * log(2*pi*e) + logdet(cov_n(r[n,t]) + eps*I))
H_grip(t) = -q*log(q) - (1-q)*log(1-q), q = mean_n(g[n,t])
H(t)      = H_pos(t) + H_rot(t) + H_grip(t)
C(k)      = mean(H(0:k))
```

The executable elbow indexing is:

```text
entropy_elbow = max(argmax(diff(C)) + 1, 2)
```

For candidate `0`, official motion uses **decoded, unnormalized** actions:

```text
M(k) = ||sum(delta_position[0:k])||
     + ||compose_left(delta_rotation[0:k])||
     + 0.2 * any(gripper toggles in 0:k)

motion_floor = first k in [2, H] where M(k) > move_th, else H
chunk_size   = max(entropy_elbow, motion_floor)
```

`compose_left` initializes identity and updates `R_total = R_delta * R_total`. The official default is
`move_th=3.0` in its RoboCasa controller-action scale.

## 4. Official Candidate Selection

The default selector is candidate `0`. Two optional selectors are present upstream:

- `mean`: flatten every candidate's complete normalized `H x 7` chunk, compute the population mean,
  and select the candidate with minimum whole-vector L2 distance;
- `backward`: retain the **entire previous candidate batch**, compare each current candidate with the
  unexecuted suffix of the same previous candidate index, calculate per-step L2 distances and apply
  normalized weights `beta**t`, with `beta=0.99`.

The chunk-size computation always uses candidate `0` for motion, independent of the final selector.

## 5. ManiMux Target Contract

The current model contract differs from the official experiment:

```text
GR00T N1.7 YAM checkpoint
  input: 3 RGB + current 14D dual-arm joint state + instruction
  output: N x 16 x 14 absolute joint positions
  order: left 6 joints + gripper, right 6 joints + gripper
  policy dt: 1/30 s
```

AAC is split across two ownership boundaries:

```text
XPolicy
  official GR00T N1.7 observation processing
  -> one backbone forward
  -> N-way feature expansion
  -> unchanged official denoise
  -> N native absolute-joint chunks

ManiMux XPolicy bridge
  native chunks + measured YAM state
  -> YAM FK
  -> incremental dual-arm EE actions
  -> fixed min-max normalization
  -> official AAC selection equations
  -> truncate selected native joint chunk only

ManiMux runtime
  -> ActionChunk -> Timeline -> SmoothExecutor -> Safety -> YAM driver
```

No EE value is sent to IK. FK and EE normalization are scoring-only; the exact selected joint values
remain the commands decoded by the normal XPolicy adapter.

## 6. Joint-to-Incremental-EE Adaptation

For each candidate and each arm, YAM FK produces absolute grasp-site poses `T[t]`. Let `T[-1]` be the
measured pose at inference time. The per-step feature is:

```text
delta_p[t] = p[t] - p[t-1]                         # arm-base frame
delta_R[t] = R[t] @ R[t-1].T                       # official left action convention
delta_r[t] = RotationLog(delta_R[t])                # rotation vector
feature[t] = concat(delta_p[t], delta_r[t], grip[t])
```

This is intentionally not “every future pose relative to the observation pose.” Using one fixed anchor
would make later entries cumulative poses instead of the official per-control-step action sequence.

Each arm is scored independently with the official single-arm formula. The final scalar entropy,
motion and selector distance are the arithmetic mean of the two arm scalars. Dual-arm averaging is an
explicit embodiment adaptation requested for YAM; it is not in the single-arm official client.

## 7. Fixed EE Normalization

The model checkpoint contains joint normalization statistics, not EE-action statistics. Reusing joint
stats or normalizing each candidate batch would change the entropy meaning. ManiMux therefore requires
a fixed external stats file for AAC:

```text
src/manimux/integrations/xpolicylab/norm_stats/yam_60ep_ee_increment.json
```

Generation command:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/datasets/compute_yam_aac_ee_stats.py \
  --episodes /home/ubuntu/yam-abc-reproduce/data/episodes \
  --out src/manimux/integrations/xpolicylab/norm_stats/yam_60ep_ee_increment.json
```

The file records 60 complete YAM episodes and two kinds of action increments:

1. measured pose at `t` to recorded absolute action at `t`;
2. recorded absolute action at `t-1` to recorded absolute action at `t`.

For each arm and each of the six pose dimensions, entropy and selectors use the official min-max formula:

```text
x_norm = 2 * (x - min) / (max - min) - 1
```

Constant dimensions map to zero. Gripper remains continuous for candidate distance and becomes
`gripper >= 0.5` only for Bernoulli entropy/toggle detection.

This stats file is invalidated by any change to robot embodiment, joint order, FK model, grasp site,
translation frame, rotation convention, gripper semantics or materially different training data. It is
separate from the model checkpoint's own norm stats.

## 8. Motion-Threshold Calibration

Official `3.0` is not portable from RoboCasa controller-action units to YAM meters/radians. The stats
generation also measures the exact official motion expression on recorded dual-arm windows. The
16-step distribution contains 24,856 samples:

| Quantile | Motion magnitude |
|---:|---:|
| 10% | `0.03369` |
| 25% | `0.08140` |
| 50% | `0.15646` |
| 75% | `0.25937` |
| 90% | `0.43900` |
| 99% | `0.81824` |

The initial YAM config uses `motion_threshold: 0.2`, between the median and 75th percentile. This is a
reproducible first hardware-test value, not a claim of optimality or official YAM validation. Every
hardware run must retain `chunk_size`, `entropy_elbow`, `motion_floor` and motion traces so later changes
are evidence-led.

The 50-step Pi05 distribution contains 22,911 windows with median `0.34782` and 75th percentile
`0.55810`. The initial 50-step config deliberately keeps the same physical `0.2` crossing threshold:
the selector still asks for the first prefix reaching the same motion, while the longer horizon only
provides more possible prefixes. This value remains GPU/hardware-unverified.

Pi05's sampler-specific audit is recorded separately in [`aac-pi05.md`](aac-pi05.md); the scoring,
selection and runtime below are shared.

## 9. Implementation Walkthrough

### 9.1 XPolicy sampler hook

| File | Change | Invariant |
|---|---|---|
| `XPolicyLab/policy/GR00T_N17/gr00t_n17/gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | Accept `options.n_samples`, expand post-backbone features and alternate-model masks | Denoising equations and step count unchanged |
| `XPolicyLab/policy/GR00T_N17/gr00t_n17/gr00t/policy/gr00t_policy.py` | Pass options and repeat state batch only for decoding | One observation only |
| `XPolicyLab/policy/GR00T_N17/model.py` | Add `get_action_aac`, return `N` native action-step sequences | No entropy or robot FK inside model package |
| `XPolicyLab/client_server/ws/model_server.py` | Advertise and dispatch `aac` capability | Non-AAC models cannot claim support |
| `XPolicyLab/tests/unit/test_ws_infer_sampling.py` | Cover capability and dispatch | Wire behavior remains explicit |

Only `mode=aac` and `num_samples=N` cross the WebSocket boundary. Motion threshold, normalization and
selectors are ManiMux concerns and are not sent to the model server.

### 9.2 ManiMux selection and runtime

| File | Responsibility |
|---|---|
| `src/manimux/integrations/xpolicylab/aac.py` | FK, incremental EE features, min-max, entropy, motion, selectors and truncation |
| `src/manimux/integrations/xpolicylab/policy_plugin.py` | Capability request, stats loading/cache and candidate selection |
| `src/manimux/integrations/xpolicylab/ws_client.py` | Preserve the structured candidate response |
| `src/manimux/runtime/aac.py` | Official synchronous query cadence and selected-chunk metadata |
| `src/manimux/config.py` | Typed AAC config and required stats validation |
| `scripts/datasets/compute_yam_aac_ee_stats.py` | Reproducible embodiment stats and motion calibration |
| `scripts/validation/xpolicylab_yam_forward_probe.py` | Hardware-free AAC request and selected-horizon report |
| `configs/groot/yam/infra/aac.yaml` | Complete GR00T/YAM experiment composition |
| `configs/pi05/yam/infra/aac.yaml` | Robocurve 16-step Pi05/YAM composition |
| `configs/pi05/yam/infra/pick-red-ball-box/aac-step1000.yaml` | Local 50-step Pi05/YAM composition |

AAC waits until the selected chunk ends before submitting the next observation, matching the official
synchronous rollout cadence. During model latency, the robot holds; AAC is not RTC.

## 10. Configuration Audit

Current experiment-critical values:

```yaml
policy:
  action_dt_s: 0.03333333333333333
  horizon_steps: 16
  options:
    allow_short_horizon: true
    aac_kinematics: yam

execution:
  runtime: aac
  blend_steps: 0
  aac:
    num_samples: 20
    motion_threshold: 0.2
    ee_stats_path: src/manimux/integrations/xpolicylab/norm_stats/yam_60ep_ee_increment.json
    chunk_id_selector: "0"
    backward_beta: 0.99
```

- `blend_steps=0` prevents Timeline seam blending from rewriting the selected official prefix.
- `allow_short_horizon=true` permits the bridge to return `2..16` actions.
- selector `"0"` is the official default and removes backward-history behavior from the first test.
- SmoothExecutor and Safety remain active outer hardware layers; they are not claimed as AAC logic.

## 11. Fidelity Matrix

| Item | Official | Current GR00T/YAM | Status |
|---|---|---|---|
| Candidate count | 20 | 20 | exact |
| Backbone reuse | expand post-backbone features | same placement | exact by code inspection; GPU pending |
| Denoise loop | unchanged GR00T N1.5 loop | unchanged N1.7 loop | equivalent adaptation |
| Native policy action | 7D EE delta | 14D absolute joint | different |
| Scoring action | native EE delta | joint → FK → incremental EE | equivalent geometric adaptation |
| Entropy normalization | checkpoint EE min-max | fixed matched-data EE min-max | equivalent adaptation |
| Gaussian/Bernoulli equations | official source | same equations | exact |
| Elbow indexing | `max(argmax(diff)+1,2)` | same | exact |
| Motion composition | sum translation, left-compose rotation | same | exact |
| Motion threshold | 3.0 RoboCasa units | 0.2 YAM physical units | different, calibrated |
| Default selector | candidate 0 | candidate 0 | exact |
| Mean selector | whole normalized chunk L2 | per-arm whole-chunk L2 then mean | dual-arm adaptation |
| Backward history | full candidate batch | full candidate batch | exact structure; dual-arm distance adaptation |
| Arms | single | dual, scalar mean | different by design |
| Model | GR00T N1.5 | GR00T N1.7 YAM finetune | different |
| Runtime cadence | synchronous | synchronous | exact |
| Final execution | EE simulator action | selected original joint chunk | embodiment adaptation |

## 12. Validation Ladder

### 12.1 Static and unit evidence

Commands:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python -m pytest -q \
  tests/unit/test_aac.py tests/unit/test_xpolicylab_plugins.py
```

Evidence on 2026-08-21: `45 passed`. Covered items include per-step translation, left-multiplied
rotation, min-max, official indexing, motion composition, dual-arm mean, candidate selectors, required
stats, WebSocket request mapping, variable horizon and synchronous runtime cadence.

Full local evidence from the same revision:

- ManiMux unit suite: `169 passed`;
- XPolicy WebSocket sampling suite: `6 passed`;
- touched ManiMux Ruff checks: passed;
- ManiMux and GR00T/XPolicy Python compilation: passed;
- root and XPolicy `git diff --check`: passed.

The GR00T model virtualenv does not install `pytest`; the six XPolicy tests were run from the ManiMux
test environment with working directory `XPolicyLab`. This does not constitute a GR00T GPU forward.

Before handoff also run:

```bash
envs/yam/.venv/bin/python -m pytest -q tests/unit
envs/yam/.venv/bin/ruff check \
  src/manimux/integrations/xpolicylab/aac.py \
  src/manimux/integrations/xpolicylab/policy_plugin.py \
  src/manimux/runtime/aac.py \
  scripts/datasets/compute_yam_aac_ee_stats.py \
  scripts/validation/xpolicylab_yam_forward_probe.py \
  tests/unit/test_aac.py tests/unit/test_xpolicylab_plugins.py
git diff --check
```

### 12.2 Real-model GPU forward gate

Operator starts the existing GR00T server using the GR00T environment, then runs:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/groot/yam/infra/aac.yaml
```

Pass criteria:

- server handshake advertises `aac`;
- one response contains finite native joint actions;
- selected `canonical_shape` is `[K, 14]`, `2 <= K <= 16`;
- metadata contains `chunk_size`, `chunk_id`, `entropy_elbow`, `motion_floor`, entropy and motion arrays;
- repeated calls do not show uncontrolled GPU memory growth;
- latency and peak memory are recorded, not inferred from unit tests.

### 12.3 Real-robot gate

Only the operator runs hardware after the normal GR00T/YAM camera, achieved-state and emergency-stop
checks in `docs/gr00t-yam-runbook.md`:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run --config configs/groot/yam/infra/aac.yaml
```

First run is an infrastructure characterization, not a task-success claim. Record:

- initial achieved joints and first selected joint target;
- server latency, total round trip and hold duration;
- every selected `K`, entropy elbow and motion floor;
- accepted/rejected chunk counts and reasons;
- commanded velocity/acceleration after SmoothExecutor;
- visible discontinuity, pauses, oscillation or unexpected gripper transitions;
- recorder output directory and stop reason.

Stop immediately for unexpected direction, joint-limit approach, stale camera/state, non-finite action,
repeated large oscillation or any mismatch between viewer and physical achieved state.

## 13. Known Differences and Open Questions

1. GR00T N1.7 is not the official GR00T N1.5 policy and may produce a different candidate distribution.
2. Joint→FK reproduces EE geometry but cannot turn a joint-trained checkpoint into an EE-trained policy.
3. Dual-arm averaging may hide high uncertainty on only one arm; max or weighted fusion would be a new
   method and is deliberately not introduced before baseline testing.
4. `motion_threshold=0.2` is data-calibrated but not task-validated.
5. Synchronous AAC may hold during the larger `N=20` action-head latency.
6. SmoothExecutor/Safety can alter the final issued command after selection; recorder evidence must
   distinguish selected reference from issued command.
7. Real GPU memory, real latency and hardware behavior remain unverified until the gates above run.

## 14. Rollback

- Stop the hardware runtime before stopping model/camera services.
- Return to the previously verified default path with
  `configs/groot/yam/infra/manimux.yaml`; it does not request AAC capability.
- No checkpoint, checkpoint norm stats or robot driver files need to change when switching runtimes.
- Do not delete AAC logs: they are required to explain why a selected horizon behaved differently.

## 15. Reviewer Checklist

- [ ] Official commits and inspected files still exist and match this record.
- [ ] XPolicy expands post-backbone features, not raw images.
- [ ] Denoising step count/equations are unchanged.
- [ ] WebSocket receives only sampling mode and sample count.
- [ ] Config stats path belongs to the active embodiment/data domain.
- [ ] First EE increment starts from measured state; later increments are step-to-step.
- [ ] Translation and rotation frames match the documented convention.
- [ ] Entropy/selector use fixed min-max; motion uses unnormalized increments.
- [ ] Backward selector retains the full candidate batch.
- [ ] Selected native joint chunk is not replaced by IK output.
- [ ] `blend_steps=0` and short horizon are both enforced.
- [ ] Offline, GPU, simulator and hardware evidence are reported separately.
- [ ] Hardware conclusions include recorder paths and observed failures, not only “ran successfully.”
