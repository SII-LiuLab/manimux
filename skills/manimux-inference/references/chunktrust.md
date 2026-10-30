# Built-in ChunkTrust AHS and RTC+AHS

`chunktrust_ahs` and `chunktrust_ahs_rtc` are separate runtime algorithms. They share
the embedded PI-family numerical core, complete validated options and transactional
selector state. No ChunkTrust package or model framework is installed in the runtime
environment by this integration. Model sampling stays in XPolicyLab.

The source is [ChunkTrust a9fd574](https://github.com/hf618/ChunkTrust/tree/a9fd57412b3a19a0728cd726b0558dd89395fb84).
The source/hash/license identity is recorded by `runtime/chunktrust/provenance.py`.
The effective selector identifies `pi_family_frozen`; timing identifies the ManiMux
policy grid rather than claiming reproduction of variable-duration simulator TOPP.

## Selecting recipes and actual backend support

Use the existing experiment loader and presets:

- `manimux/configs/experiments/example-yam-pi05/put_bottles/chunktrust_ahs.yaml`
  selects `manimux/configs/inference/yam_chunktrust_ahs.yaml`: serial, ordinary
  sampling and adaptive execution of K policy intervals.
- `manimux/configs/experiments/example-yam-pi05/put_bottles/chunktrust_ahs_rtc.yaml`
  selects `manimux/configs/inference/yam_chunktrust_ahs_rtc.yaml`: single-inflight,
  consume at most the selected old K while preparing a conditioned replacement.

Both require Pi05 JAX, H=50, native joint output with two 6+1 groups, Direct executor
and inline decoding. Extra blend/skip, ACT, process decode, Smooth/MPC, other profiles
and multi-inflight algorithm composition are rejected initially. The transport probe
also checks streaming payload preservation; that does not enable a streaming strategy.

The canonical XPolicyLab submodule does not expose the required velocity trace.
The optional public fork
[`hf618/XPolicyLab@53ac927`](https://github.com/hf618/XPolicyLab/tree/53ac927bd485c97d068586b0b9116b14b5e07126)
exposes **default-only** trace. It can supply `chunktrust_ahs`; it cannot supply
`chunktrust_ahs_rtc`. Its Model, Policy and sampler reject RTC+trace. Startup validates
the real HELLO before hardware connection; separate RTC and default trace capabilities
are insufficient evidence of combined support. No verified RTC+trace service is shipped.

## Parameters and their consumers

All 19 `inference.chunktrust` fields are read locally by the shared selector. They
are not sent to the model. The schema/defaults live once in `runtime/chunktrust/config.py`.
Both presets list the complete values; experiment overrides take precedence over the
preset, including inherited experiment values. Clear inherited fixed chunk limits.

| Field | Default | Actual consumer / restriction |
| --- | --- | --- |
| candidates | 10,20,30,40,50; RTC preset ends at 40 | Candidate scoring basis; sorted unique positive integers |
| exec_mode | expected_round | expected_round, thompson or qmix_expected_round |
| temperature | 0.8 | Expected-mode probability power transform; positive |
| epsilon | 0.05 | Posterior-mode random candidate exploration; [0,1] |
| seed | 0 | Selector RNG/reset, independent of model noise |
| update_mode | kernel_forget | kernel_forget or independent posterior update |
| kernel_bandwidth | 10 | Gaussian posterior kernel; positive |
| forget_rho | 0.99 | Kernel posterior forgetting; (0,1] |
| update_eta | 1 | Kernel posterior update strength; nonnegative |
| intra_cut_t | 0.25 | FFT low/high frequency split; [0,1] |
| intra_z_mode | window_rms | window_rms, tau0_rms or trend |
| intra_tau_window | 3 | window_rms baseline, positive integer |
| intra_alpha | 4 | Intra score sensitivity; nonnegative |
| inter_use_speed_uniformity | true | Whether to compute inter continuity evidence |
| mix_mode | both | intra, inter or both |
| history_window | 60 | Policy-row continuity window, integer at least 4 |
| inter_weight | 0.5 | Both-mode inter weight; [0,1] |
| inter_beta | 4 | Inter continuity score sensitivity; nonnegative; not RTC beta |
| inter_fallback | ehh_only | Missing-history fallback: ehh_only or neutral |

Certainly inactive parameters must keep their defaults; a nondefault inactive value
is rejected instead of appearing to tune an unused branch. Effective metadata reports
active, conditional, diagnostic-only and inactive controls. `qmix_expected_round` has
no selector randomness or posterior update; `independent` does not read the kernel
controls. Inter-only can still use intra evidence when its fallback requires it.
Expected-round K can lie between candidate values; K need not be a candidate.

`policy.horizon_policy_steps` and the validated backend profile determine H and the
14-dimensional scoring layout. They are not additional AHS knobs. FFT uses the original
H50 grid even after RTC removes expired rows. Beta(1,1) prior and numerical conventions
belong to the fixed source identity, not additional configuration switches.

`inference.rtc.beta` passes through `RtcInferenceRequest.rtc_beta`, the shared MM
client encoder, wire `sampling.beta`, XPolicyLab `validate_rtc`, Pi05 `get_action_rtc`,
Policy and the sampler's guidance cap. The RTC preset uses 10 to match the reference
cap; ordinary ManiMux RTC defaults to 5. `initial_delay_policy_steps: null` seeds the
online policy-grid estimate from warmup; `delay_buffer_size: 10` retains recent delays.
These are ManiMux deployment semantics. The original ChunkTrust RTC panel instead
froze B1 P95 delay and waypoint P05 duration, with no online buffer. Its per-request
identity-hashed model noise is also distinct from the backend's rolling seeded RNG.

`policy_server.num_steps` and `inference_seed` are read by the model deployment,
reported in HELLO, and checked against the returned trace. Changing selector seed
does not change model noise. No AHS field is copied into `policy.options` to imply
server consumption.

## Wire, evidence and the future RTC contract

Trace is a deployment output option, `policy_server.return_denoise_trace: true`,
not an INFER sampling field. Requests retain the existing strict wire:

- Ordinary/cold-start: `sampling: {mode: default}`.
- Conditioned RTC: `mode: rtc`, physical `action_condition[H,14]`,
  `condition_weights[H]` in [0,1], and positive `beta`. The server transforms the
  condition using this request's observation and its loaded normalization.

The same result contains complete actions and `denoise_trace.v: float32[T,H,Dmodel]`.
V is normalized model-space **raw network velocity before each Euler update**, in
noise-to-clean order. It is not joint speed. Actions go through normal output transforms;
V does not. Both blocking and streaming MM client paths retain the full mapping.

The live `model.denoise_trace` schema declares enabled=true, integer schema_version=1,
field=v, axes=[denoising_step,action_step,model_action], coordinate_space=normalized_model,
velocity_kind=raw_network_velocity, model_action_dim, valid_action_indices=0..13,
dtype=float32 and supported_modes. T must match live model.num_steps, H must match the
runtime, and model_action_dim must match live model.action_dim. All values must be finite.
Missing, wrong-dtype or misaligned evidence rejects the response; original provider
wire is journaled before parsing or rejection.

RTC requires both `default` and `rtc` in sampling and trace modes, plus this **future**
model metadata (it is not implemented by the published trace fork):

```yaml
rtc_guidance:
  padding_mask: valid_action_indices
  valid_action_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
```

This declares that padded model dimensions receive zero RTC guidance weight, matching
the source RTC panel. Existing Pi05 RTC broadcasts row weights into padding; a trace
flag cannot fix that. Future trace must retain raw v along the guided trajectory,
not replace it with the corrected `guided_update`. No metadata is fabricated by MM.

RTC state retains full H, selected K, reference-plan identity, actual submitted-row
progress, remaining budget and the request snapshot separately. The forecast triggers
the request; actual consumption determines the common actions/V suffix at handoff.
Early results may replace unstarted old actions at a runtime control-tick boundary. Late results do
not extend old K: the terminal target is held without creating extra history rows.
Insufficient available suffix for the frozen candidate basis rejects explicitly.

Preparation computes on a clone. Acceptance advances posterior/RNG once; rejection,
duplicate, timeout and reset discard pending state. Successful command submission
admits the corresponding original model row to the scoring history, so control
100 Hz/policy 30 Hz does not triple-count rows. Values come from the accepted
absolute-joint prediction before control interpolation. Preview uses an isolated
ledger and the same strategy; Start drains outputs and acknowledges backend reset
before formal execution.
The preview strategy excludes the first ordinary and the first conditioned cold call
from delay calibration. Steady end-to-end samples seed formal execution after reset.
Formal execution does not exclude its first latency sample or clamp infeasible delays.

`robot.options.execute: false` produces validation-only feedback and does not advance
the formal RTC consumption ledger. It is not a closed-loop substitute for successful
command submission; the formal RTC request cadence cannot be qualified by simply
disabling dispatch. Use the isolated preview lifecycle and public runtime/lifecycle
probes below for a complete disconnected loop. Preview advances its own validated
target ledger and never promotes those rows into formal execution history.

The first successful command submission in a policy interval marks its source row
as consumed; history stores that original model action, not the interpolated command.
RTC source offsets select the matching row of the full prediction. Expired prefixes,
unexecuted suffixes, duplicate ticks and terminal holds after K add no history rows.
The ledger retains the latest `history_window` admitted model rows across as many
chunks as needed; it never fills history with complete unexecuted predictions.
This counts submitted policy intervals, not waypoint completion or measured arrival.
Handoff does not wait for the source simulator's variable-duration waypoint to finish.
This is the ManiMux control-grid adaptation, not a reproduction of the source TOPP
execution protocol. Experiment recording of provider responses, control commands
and measured feedback is separate and retains its existing values and cadence.

## External model service without changing the default pin

The managed launcher explicitly imports `third_party/XPolicyLab`; changing only its
Python interpreter does not select another source tree. Keep that default unchanged.
Prepare the fixed public fork above in its own checkout/model environment, then export
the resolved standalone model recipe using the station's real bindings:

```bash
python scripts/setup/export_chunktrust_backend.py \
  --config manimux/configs/experiments/example-yam-pi05/put_bottles/chunktrust_ahs.yaml \
  --local /path/to/station.yaml --output /tmp/chunktrust-backend.yaml
```

This only exports configuration and a SHA/provenance sidecar. It does not clone, patch,
load a checkpoint, create an environment or start a service. RTC export is refused
because no supported combined-trace revision is known. The native XPL entry point uses
plain YAML; do not pass the whole ManiMux experiment or unresolved `${paths.*}` to it.

In the model environment, using a checkout directory named `XPolicyLab`:

```bash
CHUNKTRUST_XPL_ROOT=/path/to/XPolicyLab
git -C "$CHUNKTRUST_XPL_ROOT" rev-parse HEAD  # must equal 53ac927bd485c97d068586b0b9116b14b5e07126
PYTHONPATH="$CHUNKTRUST_XPL_ROOT/..:$CHUNKTRUST_XPL_ROOT:$CHUNKTRUST_XPL_ROOT/policy/Pi_05/openpi/src" \
  "$CHUNKTRUST_XPL_ROOT/policy/Pi_05/openpi/.venv/bin/python" \
  "$CHUNKTRUST_XPL_ROOT/setup_policy_server.py" --config_path /tmp/chunktrust-backend.yaml --protocol ws
```

After separately qualifying the model, `manimux up --config <same-experiment>
--local <same-station> --external-policy` connects to that service and does not own
its process. Do not also pass `--policy-python`. This document does not assert that
model-environment dependencies, a checkpoint or a physical station were validated.

## Public disconnected checks

```bash
git submodule update --init third_party/XPolicyLab
uv sync --locked --extra xpolicylab --group dev
uv run --locked --extra xpolicylab python scripts/validation/chunktrust_config_probe.py
uv run --locked --extra xpolicylab python scripts/validation/chunktrust_core_probe.py \
  --output /tmp/manimux-chunktrust-core.json
uv run --locked --extra xpolicylab python scripts/validation/chunktrust_runtime_probe.py \
  --output /tmp/manimux-chunktrust-runtime-probe
uv run --locked --extra xpolicylab python scripts/validation/chunktrust_lifecycle_probe.py \
  --output /tmp/manimux-chunktrust-lifecycle.json
uv run --locked --extra xpolicylab python scripts/validation/chunktrust_wire_probe.py \
  --output /tmp/manimux-chunktrust-wire-probe
```

The config probe exercises actual preset/experiment loading and matrix generation.
The standalone core probe checks checked-in independent reference fixtures, all 19
parameter consumers, JSON replay, nondefault reset and transactional acceptance.
The runtime probe runs the production planner, timeline, Direct executor and safety
guard across AHS/RTC early/late handoffs and submitted-history cases. The lifecycle
probe runs production warmup including both cold branches, recovery after a rejected
conditioned trace, the reset acknowledgement fence, dispatch failure and capability
validation. It also proves the first formal accepted delay is retained. Startup order
is checked statically
alongside the executed capability hook. This is not a complete physical Edge.run.

Install the runtime's `xpolicylab` transport extra and initialize its pinned submodule
for this codec probe. It uses synthetic arrays, real XPL message parsing/serialization,
the strict RTC validator and an unchanged Pi05 forwarding method extracted from source.
A recording downstream policy proves nondefault beta reaches that boundary. It checks
both MM clients, strict live identity, full trace, and production ResponseLog recovery
of missing/dtype/shape/NaN evidence. Network socket construction is forbidden. Outputs
include source hashes, dependency versions and request/response bytes outside the repo.
It does not load OpenPI or test its JAX sampler. Real weights, guided raw-v semantics,
numeric equivalence, latency/memory and physical task success need separate evidence.
