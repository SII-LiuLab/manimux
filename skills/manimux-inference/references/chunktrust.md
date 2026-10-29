# ChunkTrust serial integration contract

The user-facing configuration and install commands live in
[the guide](../../../guideline.html#chunktrust). This integration is an optional
strategy plugin, with `algorithm: serial`, not a new model or transport.

## Ownership and evidence

- XPolicyLab owns Pi05 JAX sampling and the default-off raw-velocity trace.
  [PR #4](https://github.com/Cuzyoung/XPolicyLab/pull/4), initially
  `53ac927bd485c97d068586b0b9116b14b5e07126`, supplies the trace contract.
  This parent change deliberately leaves its existing gitlink unchanged.
- The separately installed, commit-pinned ChunkTrust package owns the frozen
  PI-family AHS algorithm. Do not substitute the StarVLA or GR00T selectors or
  assume that attention outputs provide the same evidence as raw velocity.
  [Package readiness PR #2](https://github.com/hf618/ChunkTrust/pull/2), commit
  `6dacd61688e6579ee82a11fe60ef3810f06cd034`, is pinned by the optional install file.
- ManiMux owns payload preservation, identity validation, decoding, execution
  prefix selection, state transitions, submission feedback and recording.

The paired recipe requires `model.denoise_trace` schema 1, axes
`[denoising_step, action_step, model_action]`, normalized model coordinates,
raw network velocity, float32 dtype and `supported_modes: [default]`.
The model action width and valid indices are explicit. The first recipe uses
32 model coordinates with indices 0 through 13 valid. Complete actions are
decoded before slicing the execution prefix. Actions and command history have
matching canonical physical joint coordinates; velocity stays normalized.

The first profile is native dual-arm joint control with Direct execution, inline
decode, zero blending/skip and no ensemble or additional motion/gripper mapping.
Backend identity and each response are checked; unsupported input must not fall
back to fixed-length Serial or generate placeholder evidence.

## State and clocks

Selection draws and posterior changes are pending until plan acceptance. Rejecting
a plan discards that pending state. Repeated preparation of the same response
reuses the selection. Accepted K does not imply K executed targets.

The command feedback hook records the first successful canonical target in each
accepted plan's policy interval, retaining only the configured history window.
Higher control rates do not duplicate rows. Unsent/skipped rows are not filled in.
Gap/Pause holds and failed submissions do not advance this history.
The history is controller-submitted targets, not measured robot motion or a
lossless hardware bus log. Interpolated targets can differ from predicted rows.

Preview owns a separate selector, accepted plan and history. It exercises at least
two ordinary requests through the same planner/executor. `preview` feedback feeds
that private instance only. `validated_only` never becomes submitted history.
Start uses the existing backend-reset acknowledgment; Pause/Home reset strategy
state and fence old replies. This adds no robot lifecycle or recovery logic.

## Reproduction and remaining gates

The public disconnected probe exercises the actual configuration, message codec,
clients, adapter, selector, timeline, executor and feedback using synthetic inputs.
It saves input arrays, selected prefixes, state and a report with dependency
versions. No motor driver or model backend is constructed. Repeat from a clean
source snapshot and the pinned install file before submitting changes.

Passing the probe is CPU contract evidence. Real JAX compilation, trace-on/off
action equivalence, independent velocity checks, representative camera/state
inputs, steady latency/memory and physical task outcomes remain separate gates.
The station recipe is not itself a claim that these gates passed.

RTC+AHS additionally needs guided-path raw velocity, consumed-source offsets,
aligned action/trace suffixes and late-result handling. It is outside this serial
integration and cannot be enabled by adding an RTC parameter block.
