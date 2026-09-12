# DVAC on Pi05/YAM Reproduction Record

## 1. Scope and Claim Boundary

- **Method:** DVAC, *Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for
  Flow-Based Robot Policies*.
- **Paper:** [arXiv:2606.03847v1](https://arxiv.org/abs/2606.03847).
- **Paper source:** arXiv source archive retrieved on 2026-08-24.
- **Official repository:** no author implementation was located on 2026-08-24.
- **Official primary backbone:** a Pi0.5-based flow policy.
- **ManiMux backbone:** the existing JAX Pi05/YAM step-1000 checkpoint and its matching norm stats.
- **No-retraining claim:** model parameters, checkpoint, inputs, output transforms and ten-step Euler
  flow integration are unchanged.
- **Current gate:** targeted ManiMux/XPolicy unit tests, real Pi05 GPU forward, WebSocket metadata,
  two stateful rolling-window probes and one dual-YAM hardware rollout passed.

This integration is a **paper reproduction**, not an official-code port. The equations, execution
order and published defaults are preserved. Two details omitted by the paper are implemented as
explicit, reviewable conventions rather than being presented as author code:

1. an empty first-call rolling buffer is bootstrapped from that call's current variance sequence;
2. variance is summed over YAM's 14 valid normalized action dimensions, excluding OpenPI's padded
   dimensions.

## 2. Paper Algorithm

Pi05 integrates a velocity field from noise time `t=1` to clean time `t=0`. At Euler step `i`, DVAC
reuses the already computed velocity and forms the clean-action estimate

```text
z_i = x_i - t_i * v_theta(x_i, t_i, s)
```

For future action index `k`, action dimension `d`, and the final `L` denoising estimates, the paper
defines

```text
mean_z[k,d] = (1/L) * sum_i z_i[k,d]
V_s(k) = sum_d (1/L) * sum_i (z_i[k,d] - mean_z[k,d])^2
```

The rolling calibration window contains every per-step variance from up to `m` recent policy calls:

```text
W_s = union_q {V_q(0), ..., V_q(H-1)}
tau_s = mean(W_s) + alpha * std(W_s)
```

The first index whose variance is strictly greater than `tau_s` terminates the stable prefix. If no
index crosses the threshold, `N_max` actions execute. ManiMux uses the bounded operational form

```text
N_exec = N_max                              if no crossing
N_exec = max(N_min, first_crossing)         otherwise
```

This intentionally follows Equation 7 exactly: despite its name, `N_max` appears only in the
no-crossing branch. The distinction is inert in the current experiment because `N_max = H = 50`.

Published defaults used here:

| Parameter | Value | Meaning |
|---|---:|---|
| `L` | `5` | final denoising clean estimates |
| `alpha` | `2.0` | rolling standard-deviation tolerance |
| `m` | `5` | recent policy calls in the rolling window |
| `N_min` | `1` | minimum executed prefix |
| `N_max` | `50` | current YAM action horizon |

The method performs the same ten velocity evaluations as ordinary Pi05. It does not add VJP,
candidate sampling, inverse flow or an auxiliary network.

## 3. Paper Omissions and Declared Conventions

### 3.1 Empty rolling buffer

Algorithm 1 computes the threshold from the prior rolling window and appends the current variance
after selection, but it does not define the first request when the prior window is empty. This
reproduction uses the current variance sequence for the first threshold only, then appends it to the
ordinary history. Metadata reports:

```yaml
cold_start: true
cold_start_policy: current_variance_bootstrap
```

Every later request calibrates from prior calls only, matching the order in Algorithm 1. XPolicy
clears this history on the protocol `reset`, so statistics never leak across episodes.

### 3.2 Valid action dimensions

The paper writes `D` as the action dimension. The current OpenPI model tensor is padded to `32D`,
while the YAM dataset, norm stats and returned action contain exactly 14 values:

```text
[left 6 joints, left gripper, right 6 joints, right gripper]
```

DVAC therefore computes the equation over the first 14 **normalized model-action** dimensions. It
does not include the 18 padded dimensions and does not mix physical joint units before summation.
The ordinary output transform still unnormalizes, restores absolute joint actions and trims padding
exactly as before.

## 4. Ownership Boundaries

### OpenPI sampler

`openpi/models/pi0.py` adds an isolated `return_denoising_variance` branch. During each existing Euler
step it stores `z_i` in a fixed five-entry rolling tail. Ordinary, RTC, AAC, PAINT and AutoHorizon
branches do not request or allocate this output.

`openpi/policies/policy.py` slices the clean estimates to the declared valid action dimension and
computes Equation 4 before the normal action output transform. It returns the full action chunk plus
the variance sequence; it does not select or truncate robot commands.

### XPolicy Pi05 adapter

`XPolicyLab/policy/Pi_05/model.py` owns the episode-local rolling variance history, threshold and
prefix selector. `get_action_dvac` returns the full decoded `50 x 14` YAM chunk plus metadata.

### XPolicy WebSocket

The server advertises `dvac` only when the loaded model exposes `get_action_dvac`. One request is:

```yaml
sampling:
  mode: dvac
  tail_steps: 5
  alpha: 2.0
  rolling_window_size: 5
  min_execution_steps: 1
  max_execution_steps: 50
```

### ManiMux Runtime

`src/manimux/runtime/dvac.py` follows the paper's synchronous cadence:

1. wait until the selected prefix is exhausted;
2. request a new full Pi05 chunk from the latest observation;
3. validate `dvac.execution_steps`;
4. retain exactly that prefix;
5. execute it through the unchanged Timeline, Executor, Safety and RobotDriver;
6. request again after exhaustion.

This is not RTC or PAINT. Inference is not overlapped with execution, so a physical robot can hold
after a short selected prefix while the next request runs. `blend_steps` is fixed to zero so Timeline
does not rewrite the paper-selected prefix.

## 5. Configuration

```text
server: configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
infra:  configs/pi05/yam/infra/pick-red-ball-box/dvac-step1000.yaml
```

The server, checkpoint and norm stats are identical to ordinary Pi05. Only the infra runtime and
DVAC method parameters change.

## 6. User-Run Validation

Run the hardware-free contract tests first:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python -m pytest -q \
  tests/unit/test_dvac_runtime.py \
  tests/unit/test_xpolicylab_plugins.py

PYTHONPATH=/home/ubuntu/manimux/XPolicyLab/policy/Pi_05/openpi/src:/home/ubuntu/manimux \
  XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m pytest -q \
  XPolicyLab/tests/unit/test_pi05_dvac_variance.py \
  XPolicyLab/tests/unit/test_pi05_dvac_adapter.py \
  XPolicyLab/tests/unit/test_ws_infer_sampling.py
```

Start the unchanged Pi05 server:

```bash
cd /home/ubuntu/manimux
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/finetune-pick-red-ball-box-step1000.yaml
```

Before any camera, CAN or robot process, run one hardware-free forward probe:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_forward_probe.py \
  --config configs/pi05/yam/infra/pick-red-ball-box/dvac-step1000.yaml
```

Required output:

- finite native and canonical shape `50 x 14`;
- `dvac.variance` contains 50 finite non-negative scalars;
- `1 <= dvac.execution_steps <= 50`;
- `tail_steps=5`, `action_dim=14`, `variance_space=normalized_valid_action`;
- first request reports `cold_start=true`.

Then exercise the rolling window with three requests in one protocol session:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/python scripts/validation/xpolicylab_yam_dvac_probe.py \
  --config configs/pi05/yam/infra/pick-red-ball-box/dvac-step1000.yaml \
  --requests 3
```

Required output:

- `cold_start` is `[true, false, false]`;
- `rolling_states` is `[1, 2, 3]`;
- every request returns a finite full `50 x 14` chunk before Runtime truncation;
- every selected `execution_steps` is in `[1, 50]`.

After the normal YAM camera, CAN, achieved-state, start-pose and emergency-stop checks, only the
operator may run:

```bash
cd /home/ubuntu/manimux
envs/yam/.venv/bin/manimux run \
  --config configs/pi05/yam/infra/pick-red-ball-box/dvac-step1000.yaml
```

DVAC may select a very short prefix. The unchanged synchronous paper cadence then holds the last
command during the next model request; this is expected behavior, not a missing chunk.

## 7. Validation Matrix

| Gate | Status | Evidence required |
|---|---|---|
| Paper/source audit | complete | arXiv v1 equations, algorithm and defaults |
| Official code parity | unavailable | no author repository located |
| Targeted unit contracts | passed | ManiMux `39 passed`; XPolicy current checkout `14 passed` |
| OpenPI sampler contract | GPU exercised | finite variance from `L=5`, valid `action_dim=14` |
| Pi05 adapter/rolling state | passed by probe | cold start then prior-history states `1,2,3` |
| WebSocket capability/dispatch | passed by probe | three DVAC replies in one session |
| ManiMux synchronous truncation | unit passed | full decode then prefix selection |
| Pi05 JAX GPU forward | passed | finite `50 x 14`, warmed `96.4/88.9 ms` |
| YAM hardware | exercised | operator completed one dual-YAM rollout |
| Task benefit | unverified | controlled repeated trials |

## 8. Reviewer Checklist

- [ ] Ordinary Pi05 output is unchanged when DVAC is not requested.
- [ ] Exactly the final five clean estimates are used.
- [ ] `z_i = x_i - t_i v_i` uses OpenPI's noise-to-clean time convention.
- [ ] Variance uses divisor `L`, sums valid action dimensions and excludes padding.
- [ ] Threshold uses prior calls, except the disclosed first-call bootstrap.
- [ ] Current variance is appended only after selection.
- [ ] Strict crossing is `V_s(k) > tau_s`, not `>=`.
- [ ] XPolicy returns a full chunk; ManiMux alone truncates it.
- [ ] Protocol reset clears the rolling buffer.
- [ ] `blend_steps=0`; no RTC, AAC, PAINT or AutoHorizon mode is composed implicitly.
- [ ] Synchronous inference holds are reported rather than mislabeled as missing chunks.
- [ ] Unit, GPU, WebSocket and hardware evidence remain separate.

## 9. First Stateful GPU Evidence

On 2026-08-24 the operator ran three DVAC requests against the real local Pi05 step-1000 checkpoint
without cameras, CAN or robot motion. All requests returned finite full `50 x 14` actions and
normalized 14D variance metadata. The first request compiled JAX in `5292.6 ms`; warmed requests took
`96.4 ms` and `88.9 ms`.

The rolling state advanced exactly as required:

```text
request       1      2      3
cold_start  true   false  false
history        1      2      3
N_exec        50     50     46
crossing    none   none     46
```

This validates the real sampler, protocol envelope and episode-local rolling selector. It does not
validate ManiMux physical execution, task success or a performance improvement over another method.

An independent current-checkout probe then returned warmed latencies `107.0/94.4/93.1 ms` and
selected horizons `47/50/3`. This confirms that the adaptive selector is active rather than silently
falling back to a fixed horizon. It also predicts that stable phases may execute long continuous
prefixes, while a three-step uncertain prefix can expose an approximately one-inference-duration
hold before the next chunk arrives.

## 10. First YAM Hardware Evidence

On 2026-08-24 the operator ran the DVAC configuration on dual YAM with the local Pi05 step-1000
checkpoint. The robot executed model output successfully. Motion remained visibly stop-and-go, like
a serial inference loop, but the operator judged the physical trajectory substantially more accurate
than the immediately preceding adaptive-horizon method.

This observation is consistent with DVAC discarding a future suffix after denoising variance crosses
the rolling threshold and replanning from a newer observation. It is one qualitative rollout, not a
controlled success-rate result, and does not prove that DVAC is generally more accurate. The pauses
remain expected because the faithful Runtime waits for the selected prefix to finish before issuing
the next synchronous model request.
