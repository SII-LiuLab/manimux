# Pi05 pack-plate / Tianji–TacCap

This integration is isolated from the existing UMI-DP pack-plate and Pi05
pass-ball experiments. The default and RTC ManiMux experiments share one Pi05
recipe; both checked-in experiments have `execute: false`.

## Local Serial launch (verified 2026-10-13)

This workstation also has the private full-zero-state, pack-instruction step59999
Serial experiment at
`.local/pack_plate/tianji-pi05-full-zero-state-pack-instruction-step59999-serial.yaml`.
It selects `inference_schedule: serial`, 32 policy steps, `execute: true`, station
`.local/tianji_pi05_pack_plate.yaml`, and the shared
`manimux/configs/embodiment/robot/tianji_taccap_pack_plate.yaml` Home values.
The commands below apply to that local experiment, not the wrist-only RTC recipe
documented in the remaining sections.

The hardware runtime uses CPython 3.12 in `.venv`. A compiled TacCap **0.4.2** already
exists under the vendored SDK's `python/xense/taccap/` directory, including the
native extension and `libtaccap_core.so.0.4.2`. Its `Motor.get_model`, `Motor.get_spec`,
`FollowerGripper.audit_envelope` and force-position controller interfaces were
verified by import without opening hardware. Reinstallation is not needed for this
path correction. The old external path
`/home/jw/Desktop/project/TacCap-Gripper/python` loads **0.1.9**, which lacks both
motor methods and causes `AttributeError` during gripper connection.

From the repository root, verify SDK resolution before starting the runtime:

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python - <<'PY'
import xense.taccap as sdk
from xense.taccap import _taccap_native as native
print(sdk.__version__, native.__file__)
assert hasattr(native.Motor, "get_model")
assert hasattr(native.Motor, "get_spec")
assert hasattr(native.FollowerGripper, "audit_envelope")
assert hasattr(native, "ForcePositionController")
PY
```

Model server (separate CPython 3.11 environment; do not inherit the hardware SDK path):

```bash
env -u PYTHONPATH XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment .local/pack_plate/tianji-pi05-full-zero-state-pack-instruction-step59999-serial.yaml
```

Hardware runtime, after the previous runtime has released ownership:

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python -m manimux serve \
  --config .local/pack_plate/tianji-pi05-full-zero-state-pack-instruction-step59999-serial.yaml
```

Reuse the existing compatible wrist-camera service and Tianji RoboGUI. When
starting a new camera service, use the same command-scoped hardware `PYTHONPATH`
with `.venv/bin/python -m manimux.servers.camera.server --experiment` and the same
Serial experiment. Correcting `PYTHONPATH` does not replace an SDK already loaded
by a running process; restart the affected hardware process after correcting it.
The Pi05 model server does not need restarting for this SDK path correction.

The Serial experiment passed `--check`; this verifies configuration and artifacts,
not successful model loading or physical execution. On 2026-10-12, model startup
was killed by the host RAM OOM killer with the 8 GiB swap exhausted. Disabling XLA
GPU preallocation does not resolve host RAM exhaustion.

## Serial step IK (offline validated 2026-10-14)

The private counterpart to the local Serial checkpoint is
`.local/pack_plate/tianji-pi05-full-zero-state-pack-instruction-step59999-serial-step-ik.yaml`.
It retains the checkpoint, station, task, 32-step Serial schedule, zero blend and
joint command limits. Its differences are inline pose decoding, direct execution,
native analytic IK per 100 Hz tick, a shared 5 ms budget, and TCP target limits of
0.4 m/s and 1.2 rad/s. Process decoding, decode-time diffIK, waypoint preparation
and smooth filtering are absent. The original Serial YAML is unchanged.

Use the same hardware interpreter and SDK environment shown above, substituting
the new `-serial-step-ik.yaml` path for `--config`. Its execution flags are inherited
from the live Serial configuration; starting a rollout enables motion. Configuration
loading and offline FK/IK checks do not start services or move hardware.

A checked-in offline-first example is
`manimux/configs/experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999_serial_step_ik.yaml`;
it retains the wrist-only checkpoint and disabled execution switches.
Offline verification covers native Tianji FK, Pi05 pose decoding and dual-arm
step IK near the configured Home pose. It does not establish real-model rollout
quality, sustained worst-case timing or physical task success. Native IK cannot
be interrupted; late results are held instead of dispatched.

## Contract and artifact

- Recipe: `manimux/configs/policy/pi05/tianji/pack_plate/wrist-only-step59999.yaml`
- Experiment: `manimux/configs/experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999.yaml`
- RTC experiment: `manimux/configs/experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999_rtc.yaml`
- Checkpoint: `/home/jw/Desktop/policy/pi05-pack-plate-wrist-only-final-59999/checkpoint-59999`
- Normalization: that checkpoint's `assets/pack-plate-taccap-h32-zero-pose/norm_stats.json`
- Dataset task prompt: `plate` (from `pack-plate/meta/tasks.parquet`)
- Input: left and right wrist RGB, zero TCP pose slots, live normalized gripper openings
- Output: 32 current-TCP-relative model actions at 30 Hz, restored to absolute per-arm-base TCP targets; grippers remain absolute
- Sampling: default and RTC; PAINT is not enabled for this profile

The local `train_config_name` reconstructs inference architecture and transforms;
it is not claimed to be the original cluster training config name. The export
does not include the training source. The current deployment uses the pass-ball
Tianji tool-axis transform. Verify that transform against the pack-plate
training source before enabling robot execution.

## RTC handoff

The RTC variant keeps the 32-step, 30 Hz checkpoint contract and uses
`min_execute_steps: 12`, `initial_delay_steps: 8`, and `beta: 5.0`.
The initial delay is based on roughly 160 ms observation-to-commit time plus
the configured 80 ms commit lead, rounded up to 30 Hz action rows; the runtime
then measures and updates its delay forecast.
The first request is unconditioned. Later requests condition the model on the
unexecuted tail of the previous joint chunk. ManiMux converts that tail to
absolute TCP targets; the Pi05 adapter restores the raw gripper opening before
the model converts the condition to its right/left, current-TCP-relative 20-D
training representation. RTC uses the last issued command as its blend anchor;
conditioned chunks disable the extra joint blend.

The ignored `.local/tianji_pi05_pack_plate_guarded_live.yaml` now selects RTC
while retaining its existing hardware, motion and gripper values. Its
`lag_policy: report` still accepts chunks with excessive Diff-IK lag. The
last pre-RTC rollout recorded severe right-arm IK lag and a controller fault;
static RTC checks do not establish that live execution is safe or that pullback
is solved. Keep the checked-in `execute: false` variant for the first handshake
check and inspect RTC conditioning and handoffs before another physical rollout.

## Private station

This installation has an isolated, ignored station file at
`.local/tianji_pi05_pack_plate.yaml`. It reuses the recorded Tianji device mapping,
points `paths.checkpoints` at the desktop policy directory, and reserves policy
endpoint `ws://127.0.0.1:8571`. Do not edit the shared UMI-DP station to select
this checkpoint. The active UMI-DP service uses a separate endpoint.

## Read-only checks

From the ManiMux root:

```bash
.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999_rtc.yaml \
  --local .local/tianji_pi05_pack_plate.yaml --check
```

When the GPU is free, run the standalone synthetic-image forward shown in
`XPolicyLab/policy/Pi_05/README.md`. That verifies model loading and output
shapes; it does not establish camera alignment, robot safety or task success.

## Service selection

Only after the existing robot and camera owners are stopped or deliberately
handed over, start the policy server with the same experiment and station:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment manimux/configs/experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999_rtc.yaml \
  --local .local/tianji_pi05_pack_plate.yaml
```

The existing camera service can be reused if its two wrist names, serials and
timestamps match the station. A robot runtime must have a single owner; this
integration does not start one automatically. After handing over the existing
runtime, use the paired experiment for the observation-only ManiMux process:

```bash
.venv/bin/python -m manimux serve \
  --config manimux/configs/experiments/pack_plate/pi05/tianji_taccap_pi05_wrist_only_step59999_rtc.yaml \
  --local .local/tianji_pi05_pack_plate.yaml
```

Keep `execute: false` for initial observation and handshake checks. Physical
execution additionally requires the training axis convention to be confirmed
and an explicitly selected rollout.

## Checks on this installation

The 30 exported checkpoint files passed SHA256 verification. The Pi05 `--check`
entry passed under the model interpreter, the paired ManiMux configuration and
adapter loaded without hardware, and the real checkpoint produced 32 finite
dual-arm EE steps from synthetic wrist images. The pack-plate and pass-ball
unit tests passed together. Camera serial and endpoint bindings were inspected
without opening devices. No live camera, robot or task rollout was tested.
