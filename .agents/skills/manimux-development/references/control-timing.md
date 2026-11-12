# Control-loop timing

Measure the existing control path before changing scheduling or moving work to
another thread. This diagnostic does not change command targets, chunk handoff,
control frequency, sleep policy, or model sampling settings.

## Record a timing run

The default experiment baseline is a 100 Hz host control loop and a 30 Hz RoboGUI
display cap. Higher-rate control runs are explicit diagnostics, not a new default.

Finish the current rollout and restart only the ManiMux runtime with the diagnostic
flag. Existing camera, RoboGUI and policy services can stay running:

```bash
envs/yam/.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml \
  --control-timing
```

Use the usual Prepare, Start and Finish controls. A useful first sample contains
steady warmup followed by 30–60 seconds of normal execution. Do not change the
experiment's inference or execution settings for this baseline. This is a real
robot runtime command: the operator remains responsible for starting execution.

The flag sets `run.control_timing: true` in the recorded resolved configuration.
The default is false. `run.timing_max_cycles` defaults to 20000; the buffer retains
the latest cycles and reports how many earlier rows were dropped. Set a larger
bound in the experiment only when a longer capture is needed. Sampling consumes
CPU and memory; it is not a zero-overhead measurement.

Each episode saves these files when Finish or orderly abort finalizes recording:

- `control_timing.jsonl`: cycle starts, phase, scheduling targets, sleep timestamps,
  and per-stage wall/CPU spans.
- `control_timing_summary.json`: phase-separated rates, latency distributions,
  stage distributions and capture/truncation metadata.

There is no per-cycle diagnostic printing or file writing. An abrupt process kill
cannot preserve the in-memory buffer.

## Read the result

```bash
envs/yam/.venv/bin/python -m scripts.validation.analyze_control_timing \
  --episode /absolute/path/to/session-.../rollout-001
```

`--json` prints the full report; `--output /path/report.json` saves a recomputed
report. The analyzer only reads saved timing evidence and never connects to a robot.

Interpret the measurements separately:

- Loop Hz uses adjacent cycle-start timestamps within a continuous phase.
  Warmup, running, pause and lifecycle transitions are separate segments.
- Command submission Hz uses `robot_send_command` call starts. It does not measure
  CAN packet arrival or physical motion.
- Work duration excludes the deliberate sleep. Sleep overshoot and deadline lag
  use the original schedule only when its clock is comparable to host monotonic time.
- Camera reads, RoboGUI control/publishing, worker queues, adapter decoding,
  timeline/executor, robot I/O and recording have separate spans.
- Robot spans include the assembly-lock acquisition and per-controller SDK calls.
  SDK call duration is not a direct measurement of an internal SDK lock.
- Nested spans overlap: do not add parent and child times. Thread CPU excludes
  other threads/processes; wall time minus CPU does not identify a specific lock.
- Incomplete or truncated captures are explicit. Synthetic/offline tests validate
  instrumentation, not the real robot's achieved control rate.

For comparison, retain the session manifest, this timing output and the experiment
configuration. Measure instrumentation overhead separately and confirm any eventual
scheduling change with an equivalent real-robot run.

## Camera reader comparison

The Pi05 RTC step-30000 experiment now selects
`run.sensor_reading.mode: background` with a 100 Hz polling cap. This is the generic
`SensorBase` reader, not a Pi05/YAM-specific path. The control-loop `camera_read`
span now measures snapshot access and freshness checks; its `mode` attribute
identifies the path. It excludes background RPC, copying and waiting. Compare the
whole-loop period, overrun count and command submission rate as well as this span:
background work still consumes CPU and can contend for interpreter scheduling.

For a paired comparison, change only `run.sensor_reading.mode` to `inline`, run the
same command and finish the rollout. The session manifest retains resolved reader
settings and source hashes. RoboGUI control polling, robot reads/submission, executor,
inference settings are unchanged by the camera reader change. Keep the same
scheduling implementation in both runs when comparing reader modes.

The REQ/REP driver now preserves source capture timestamps instead of stamping each
reply with receipt time. This applies to both reader modes. For frame-age or video
sampling comparisons, collect a fresh inline baseline with this same driver;
historical recordings with receipt-time camera fields cannot establish capture age.
Camera capture timestamps are host capture/receipt timestamps from the sensor
service, not a claim about hardware exposure time. On different hosts, the existing
Unix wire protocol requires synchronized clocks.

## Deadline handling

Warmup and normal control use the same deadline rule. A cycle that finishes early
sleeps only until the next scheduled start. An overdue cycle starts the next
iteration immediately and rebases the schedule to its finish time; it does not add
another full control period or replay missed ticks. At 100 Hz, 6 ms of work leaves
4 ms to sleep, while 12 ms of work leaves no sleep. The following cycle gets a new
10 ms budget. Timeline sampling still uses current monotonic time.

<a id="runtime-robogui-controls"></a>

## Runtime RoboGUI controls

The runtime's `RoboGUIBridge` uses one background control client. Its thread owns
the REQ socket throughout creation, polling, reconnect and close. The main-loop
`robogui_control` span now measures a local mailbox read, not a network round trip.
Session preparation retains its synchronous client; it does not poll concurrently
with a running rollout.

The resolved `robogui.control` configuration records these defaults:

```yaml
robogui:
  control:
    poll_hz: 100.0
    timeout_s: 0.02
    max_age_s: 0.05
```

The polling rate is a cap, not a camera or motor rate. `timeout_s` bounds each
background wait, not the age of the last valid control state. A missing reply does
not manufacture a Pause or refresh the cached sample. The loop keeps using a valid
sample until `max_age_s`; missing initial state, expired state or a stopped control
thread pauses execution. Sample age starts at request time, so a late reply cannot
make an old state appear fresh. A pending request is retained for bounded late
replies, then reconnected after the reply deadline to recover from service loss.
The preparation service retains its synchronous fail-closed `poll()` API.

Received Pause pulses and Home/Finish events remain latched until the loop consumes them. Finish
takes priority over Home, retains its `finish_home` value and stops the polling
thread. The mailbox cannot recover an event lost before a transport reply arrives;
the existing RoboGUI protocol does not acknowledge individual button events.
The reply retention bound is the larger of `timeout_s` and `max_age_s` and is checked
after each wait. The freshness check runs independently in the control loop.

`events.jsonl` records `robogui_control_state` when control reason, transport status
or error/timeout counts change. It includes sample age, last reply round-trip time,
and RoboGUI lock/read timing. `stale_control`, `robogui_pause`, `home`, `finish` and
`control_thread_stopped` are distinct. Rejected model responses also distinguish
pause/Home invalidation, session mismatch, supersession, deadline expiry and missing
actions. These records diagnose failure origin without relaxing inference deadlines.

The RoboGUI still shares its dashboard lock with rendering. Moving requests off
the control loop removes the synchronous wait from that loop, but does not prove
a bound on button-to-robot response time. Measure this separately from loop Hz.

Warmup previews use the same A/B chunk renderer and sampling animation as formal
rollouts. A display-only cursor plays at the action interval; it is not measured
execution. Predicted TCP paths remain blue, and Start/Finish clear previews and
reject late warmup messages. Warmup never commits these preview actions to the
robot timeline.


<a id="robogui-display-cadence"></a>

## RoboGUI display cadence

Live display reception and rendering run in separate threads. The receiver merges
adjacent robot-state messages into the newest sample; it retains the latest images
per camera because image updates are less frequent than joint updates. It does not
merge across robot/episode identity, plan messages or lifecycle events. Received
plans and events retain their order. This does not add reliable delivery to the
existing best-effort transport.

The renderer consumes these batches at a default cap of 30 Hz. Set the RoboGUI CLI
`--render-hz 60` to try 60 Hz; this never changes runtime, camera or motor rates.
Rendering over budget skips missed display ticks rather than replaying them.
Prediction FK is cached per chunk; the remaining path is redrawn when its cursor
changes. Each message's scene changes are batched with RoboGUI's atomic update.

The English `RoboGUI queue / draw` field reports receiver-to-callback age (including
waiting for the dashboard lock) and Python callback work, in milliseconds. These
numbers use the RoboGUI process's monotonic clock. They exclude upstream network
queueing and browser/WebSocket presentation latency; they are not end-to-end age.
RoboGUI controls still share the dashboard lock with a display callback, so this
change reduces work and callback frequency but does not eliminate that lock.
