# UMI DP on Tianji

The model is implemented in `XPolicyLab/policy/UMI_DP`, with vendored model and
training source. ManiMux handles camera history, FK/IK and execution through the
existing `xpolicylab_ws` worker. The unified runtime loop is unchanged.

## Prepare and bind a checkpoint

Run from the ManiMux root. The model uses its own Python 3.11 venv; the hardware
runtime needs the normal Tianji dependencies plus ManiMux's `xpolicylab` extra.

```bash
bash XPolicyLab/policy/UMI_DP/install.sh envs/umi_dp/.venv
# Check the actual EMA/model, SHA, H, dt, first offset, and preprocessing:
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --checkpoint /path/to/trusted/pass_ball.ckpt --check
# Write a paired model-server and runtime config, bound to those artifacts:
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --checkpoint /path/to/trusted/pass_ball.ckpt \
  --bind-runtime-config data/experiments/pass-ball-bound.yaml
# Select RTC with --runtime-template configs/umi_dp/tianji/infra/pass_ball/rtc.yaml.
```

The binder updates H16/H64 and observation/action timing from the checkpoint,
checks the shared control profile, and refuses to overwrite outputs. The model
checks the paired `expected_artifacts`; ManiMux verifies the server's identity
and sampling capabilities before execution. The checked-in templates are
intentionally unbound and rejected by the actual Tianji adapter until bound.

`configs/robots/tianji/common.yaml` owns all device/command limits and has 30Hz
action points. A legacy 100ms-action checkpoint cannot bind to that profile.
To use one, explicitly create/select a matching control profile and runtime
template with its action interval, retaining reviewed hardware and motion limits;
the binder will never silently change those settings. Its first action offset
is still 1/source_fps (33.3ms at 30fps).

## Start the model service

```bash
envs/umi_dp/.venv/bin/python scripts/servers/umi_dp_tianji_server.py \
  --config data/experiments/pass-ball-bound-server.yaml
```

Use the model README's `EVAL_ENV_TYPE=debug` commands to exercise real weights
without devices. The dedicated debug client checks plain and encoded RGB,
absolute EE action shapes, real RTC conditioning, batches and reset. It does not
provide evidence of robot motion or task success.

The hardware-side configuration uses the existing camera server's PUB endpoint
(default `tcp://127.0.0.1:5556`) and configured wrist names `left_wrist` and
`right_wrist`. Start camera/runtime services only as part of the intended hardware
session, using the existing Tianji camera config and the paired runtime config.
The common Tianji profile remains `execute: false`, `gripper_control: false` by
default. No devices are accessed by model validation or binding.

## Measured observation history

`TimestampedCameraSensor` consumes `CameraSubscriber.try_recv_bundle()`, retains
server capture timestamps and does not count repeated polls as new frames. It
maps wall-clock capture time to monotonic time with an offset sampled at startup,
and rejects missing, backward, stale/future timestamps or a local wall-clock jump
above 20ms. The camera server must be on the same machine, or its clock must be
synchronized within the configured state/camera tolerances. These timestamps identify the backend's host receive/callback time, not sensor
exposure time. The camera server obtains each image and timestamp atomically.
This is not a hardware synchronization guarantee.

`HistoryStrategy` uses the existing per-tick `build_submission` plugin hook to
cache measured states. Each new pair of camera frames is matched to the closest
buffered robot state, within 20ms per camera; camera skew is bounded at 40ms. It
selects two distinct measured snapshots around the checkpoint interval, within
40ms, then calls the unchanged standard `manimux` or `rtc` strategy. Warmup and
missing history defer submissions. There is no inference-request-based history,
extra hardware polling thread, or change to control_hz/the main loop.

Current runtime configuration rules restrict serial scheduling, process action
decoding and max_chunk_steps to the built-in `manimux` name. This history plugin
therefore supports single-inflight scheduling and RTC, with inline decoding;
it does not bypass those restrictions. The wrapper revalidates the delegated
strategy's full configuration, including RTC delay/horizon constraints.

## Action conversion and execution differences

The model freezes each arm's latest measured TCP reference, processes two RGB
frames, predicts relative row-rot6d poses, and returns absolute per-arm base TCP
poses and raw apertures. ManiMux converts through the configured `umi_follower`
tool and configured Tianji IK. The default `ik_backend: analytic` uses the SDK;
optional `diff` uses the ported OSQP velocity solver. See
[differential IK configuration and validation](tianji-diff-ik.md) for binding it
to the shared motion profile. Arm A/robot0 is left; arm B/robot1 is right.

The first action origin includes the checkpoint's first offset. Already expired
source rows are removed before IK and recorded in `source_offset_steps`. Each
remaining knot's SE(3) segment is checked through the selected IK in at-most-4ms
substeps. Analytic retains the original branch/limits/FK checks; diff uses its
rate, position/interference and tracking-lag checks. The first knot uses
its remaining time to the target; subsequent knots use action_dt. Any invalid
pose, aperture or IK rejects the entire chunk. Intermediate IK samples are
validation points; the shared executor interpolates final **joint** knots.
CalibWrist retains all control-rate TCP/IK samples as dense joint commands and
can precompute them; its async runtime sends them from a separate thread. This
adapter retains model knots, so that dense-command behavior is not reproduced.

For RTC, the history wrapper resamples the actually committed joint timeline at
the new observation's `first_offset + j*dt` targets, masks rows beyond the valid
committed plan, and the embodiment converts weighted conditions through FK. The
model rebases these absolute poses into the new TCP reference and guides the real
DDIM sampler with PiGDM or soft inpainting. No joint-space tensor is passed to
UMI's 20D normalizer. Guidance cost and inline IK must fit the deployed timing
budget; a finite offline result alone does not prove that budget is met.

The provided profile uses continuous gripper targets. CalibWrist's current
adapter latches closed below .6, commands .2 while latched, reopens at .75, and
otherwise retains raw aperture. That differs from this continuous profile and
from ManiMux's built-in fixed-open-value hysteresis. The model preserves raw
aperture predictions; gripper latch parity is not claimed. The shared motion
profile controls rate shaping and command guards independently of this choice.

## Validation

```bash
.venv/bin/python -m pytest tests/unit/test_umi_dp_tianji.py -q
bash -n XPolicyLab/policy/UMI_DP/*.sh
python -m compileall -q XPolicyLab/policy/UMI_DP
```

A real libKine decode probe is also available:

```bash
.venv/bin/python scripts/validation/umi_dp_tianji_decode_probe.py
```

On the integration machine, reachable synthetic two-arm chunks took about
19–23ms for H16 and 77–79ms for H64 (three trials, no hardware connection).
These are analytic-backend timings. The diff backend took 65–66ms for H16 and
260–262ms for H64; its individual QP steps had medians of 0.22–0.26ms.
Whole-chunk work exceeds the 4ms budget of a 250Hz tick because decoding is
currently inline and blocks executor smoothing until it finishes.
The model/interface integration is validated offline; the whole 250Hz control
chain is **not ready for a real-motion timing claim**. Reusing the generic process
decoder with measured history and RTC needs a separate design review. These
measurements do not justify relaxing IK checks or lowering the unified control
frequency.

See the UMI_DP README for real recorded-window forward/parity and shared-server
commands. Runtime tests cover capture identity, wrong-time history rejection,
clock jumps, RTC first offsets and tail masks, FK mapping and whole-chunk
rejection. Model training and camera/CAN motion have not been exercised by this
integration. Preserve real checkpoint and hardware timing results separately
from these hardware-free contract tests.

Recorded evidence: [2026-09-13 validation report](umi_dp-tianji-validation.md).
