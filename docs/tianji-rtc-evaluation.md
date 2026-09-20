# Tianji RTC evaluation — 2026-09-14

The RTC integration now runs the real UMI_DP sampler, measured-history adapter,
parallel Tianji IK decoders and shared executor together. This evaluation used
simulated robot feedback and fixed synthetic RGB fixtures. It did not connect
to cameras or the robot, and establishes no pass-ball task success rate.

## Configuration and method

- Parent branch starts at `99dc1e2`; XPolicyLab remains at
  `2077534393039cf844a744f52eb6a44cbdd1016c`.
- Real H64 EMA checkpoint SHA256:
  `640c2ad99f939e5d1a39f32880266a17fd7cefdb960b5e1da9a2af46d14be548`.
- UMI_DP shared WebSocket server, real 16-step DDIM with PiGDM, beta 5.
  Checkpoint/backend identity and sampling capability negotiation remained enabled.
- Both runs use `configs/experiments/runtime/tianji_taccap.yaml`, bound differential IK,
  two decoder processes, smooth execution, continuous grippers, 250 Hz control,
  30 Hz action knots, H64, and zero commit lead.
- RTC starts with a 4-action-step delay estimate and the default half-horizon
  execution window. Ordinary ManiMux uses single-inflight scheduling with a
  250 ms refill threshold.
- Each run has 5,000 control ticks (20 seconds of scheduled control). The
  simulated plant starts at the same reachable pose and follows commands with
  the existing mock driver's tracking gain. Synthetic RGB is fixed; synthetic
  capture timestamps advance at 30 Hz and pass through the real history wrapper.
- Runs were sequential, after model warmup. The checkpoint's inference
  augmentation and DDIM randomness remained active; this is not a matched-seed
  experiment or a statistical task comparison.

## Results

| Metric | RTC | Ordinary ManiMux async |
|---|---:|---:|
| Requests submitted | 19 | 11 |
| Submitted with RTC conditions | 18 | 0 |
| Plans accepted | 18 | 11 |
| Accepted with RTC conditions | 17 | 0 |
| Plans rejected | 0 | 0 |
| RTC delay-infeasible events | 0 | N/A |
| Control tick interval, median | 4.00 ms | 4.00 ms |
| Control tick interval, P99 | 4.41 ms | 4.28 ms |
| Control tick interval, maximum | 8.93 ms | 5.97 ms |
| Process decode stage, median | 75.52 ms | 79.33 ms |
| Process decode stage, P99 | 79.62 ms | 79.72 ms |
| Measured observation to commit, median | 181.24 ms | 156.24 ms |
| Measured observation to commit, P99 | 203.46 ms | 171.84 ms |
| Maximum joint command increment per tick, P99 | 0.00273 rad | 0.00358 rad |
| Maximum joint command/state difference per tick, P99 | 0.00597 rad | 0.00949 rad |
| Maximum joint reference change at a plan boundary, P99 | 0.04703 rad | 0.05756 rad |

The last RTC request was still in flight when the configured run ended; it was
not counted as an accepted plan. The process decoder takes roughly 76–80 ms
while control ticks continue near 4 ms. RTC replans more often in this fixture
and has higher inference-to-commit latency because its guided sampler performs
extra work. These numbers do not establish superiority on a manipulation task.
The observed maximum tick interval also does not constitute a hard real-time
bound.

Latency is measured from the **measured observation**, including UMI's first
action offset. The legacy event field `observation_to_commit_ms` is based on the
canonical chunk origin, which already includes that offset; the evaluation
summary adds `first_action_offset_ns` back. With zero commit lead this agrees
with RTC's new `rtc_delay_ms`. In this run the largest RTC delay was 204.87 ms,
so a 7-step initial forecast (about 233 ms at 30 Hz) would cover the observed
range. Cold startup, hardware and load variation require separate measurement.

## Interface and regression evidence

The real H64 shared-server debug client completed ordinary inference, guided
RTC, batch indices `[3, 7]` and reset. Static UMI_DP Python parsing and all policy
shell syntax checks passed.

The final regression set passed 240 tests across the two environments: 238 in
the isolated runtime environment, plus two existing XR-1 codec tests rerun in
the model environment because they import torch. One unrelated YAM IK test was
skipped because i2rt was unavailable. Torch was not added to the runtime
environment. Ruff and `git diff --check` passed for the implementation.

New regression tests cover:

- Real analytic and differential IK, H16 and H64: two decoder-process outputs
  match serial decoding, including a pre-trimmed source suffix. Invalid aperture
  in one arm rejects the whole chunk, and children are cleaned up.
- RTC retains the source horizon across adapter and commit trimming, aligns
  weighted conditions to the actual committed trajectory, rejects missing tails
  and partial-arm holds, and restores blending when no valid overlap remains.
- End-to-end delay includes observation age, decoder work and commit lead, with
  fractional action steps rounded up.
- CPU-bound decoder processes run alongside the 250 Hz mock loop; pause and
  homing during decoding reject stale results, resume without old conditions,
  and subsequently resume conditioned RTC. Decoder timeout closes the runtime.
- The runtime factory preserves Tianji's history wrapper and checks real model
  identity and RTC capabilities before robot connection.

Commands and the reusable evaluation harness are documented in
[the Tianji–TacCap runbook](umi-dp-tianji-taccap-runbook.md). Raw logs, Zarr recordings,
paired checkpoint configurations and JSON summaries are local artifacts under
`data/rtc-evaluation/` and are excluded from Git. Model source, checkpoints,
control limits and hardware drivers were not changed by this integration.

Real camera timing, hardware tracking during process decoding, motion/task
success and statistical comparisons remain unverified. In particular, the IK
seed is measured when decoding starts; the real arm continues moving before
takeover. Evaluate that tracking error with the configured motion limits and
executor before claiming hardware readiness.
