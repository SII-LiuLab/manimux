# YAM SDK lock-wait diagnostic

This is a separate, opt-in launch mode. It does not change the ordinary collection
launcher, installed SDK files, lock objects, getter return values, control periods,
or target generation. It instruments the installed SDK's existing lock contexts
inside the diagnostic process and checks that removing those contexts gives the
original function AST. The audited SDK hashes are pinned; a different SDK fails
before collection starts and requires a source audit.

Run a source-only check (does not construct robots or start services):

```bash
envs/yam/.venv/bin/python -m manimux.robots.yam.sdk_lock_probe \
  --check --output /tmp/yam-lock-audit-$(date +%Y%m%d-%H%M%S)
```

After stopping teleop and exiting the existing collection server, start the
same collection GUI through the probe. The example uses the existing synchronous
station config; its `collection_hz` is the input and execution target frequency.
The probe itself does not select or change a frequency.

```bash
envs/yam/.venv/bin/python -u -m manimux.robots.yam.sdk_lock_probe \
  --seconds 30 --output /tmp/yam-sdk-locks-$(date +%Y%m%d-%H%M%S) -- \
  --config configs/collection/yam/station.yaml \
  --cameras configs/collection/yam/cameras.yaml \
  --host 127.0.0.1 --port 8043
```

Open the same page and Start Teleop normally. Capture starts at the first SDK
getter inside an active normal teleop cycle, so alignment and initial gripper
calibration do not consume the 30-second window. Keep teleop active for 30 seconds;
episode recording is unnecessary. After the terminal reports `[sdk-lock-probe]
saved`, Stop Teleop saves the existing application timing diagnostics as usual.
The probe does not start, stop, align, pause, or disconnect hardware on its own.
One process captures one window; restarting the diagnostic launcher arms another.

The output directory contains:

- `manifest.json`: SDK paths and hashes, application source hashes, launch command,
  probe hash, and instrumented method coverage.
- `instrumentation.patch`: exact function-level in-memory instrumentation diff.
- `sdk-locks.jsonl`: per-call monotonic timestamps, thread ID, arm/CAN channel,
  original lock ID, parent span ID, wall and thread CPU timings. `cycle_ns` joins
  the existing `control-timing.jsonl` on `cycle_monotonic_ns`.
- `summary.json`: per-arm and per-thread distributions; paired getter totals,
  acquisition intervals, protected bodies, and time outside the lock; the 20
  slowest getters and observed holders of their exact lock during acquisition.
- `write_complete.flag`: the trace and summary finished writing.

The probe covers each arm's robot `_state_lock` and `_command_lock`, motor-chain
`command_lock`, `state_lock`, and `same_bus_device_lock`. It separately times
gravity compensation and the all-motor communication call. Startup constructor
locks are excluded. SDK recovery behavior is unchanged.

Interpretation:

- `wait` is elapsed time across the original lock's `__enter__`, including any GIL
  or OS scheduling delay before Python resumes. It is **not pure kernel-blocked
  lock time**. Low thread CPU distinguishes calculation from non-running time;
  it does not distinguish all causes of non-running time.
- `hold` ends just before the original `__exit__`. Nested spans overlap: do not
  add an owner's entire hold time to waits already included in that hold.
- Matching holder intervals establish observed ownership during a getter's
  acquisition interval. Uncovered intervals are not automatically scheduling
  delay; timestamp boundary overhead and incomplete boundary spans also exist.
- Buffers are bounded per thread, with drop counts and in-flight span IDs in the
  summary. There is no trace file I/O during capture, nor a new statistics mutex
  in SDK critical sections. Serialization runs on a separate thread after capture
  and can affect teleop **after** the measurement window.
- Instrumentation still adds CPU work and slightly extends protected sections.
  Compare the same mode/cameras/motion with an ordinary run if the overall rate
  changes materially. An offline overhead benchmark is not a hardware guarantee.

Return to the ordinary `python -m manimux.collection ...` command to run without
the probe. No SDK-file rollback is needed.
