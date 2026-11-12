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

## Managed checkpoint comparison (2026-10-19)

`manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261019/catalog.yaml`
contains nine checkpoints after the 2026-10-20 integration: seven TCP checkpoints
with chunk/step DiffIK and two native-joint checkpoints without IK. Each offers
Serial 32 and RTC 16 (32 choices, 16 policy groups). Step variants have separate policy
groups ending in `/ Step DiffIK`, inline pose decoding and direct execution at
pi rad/s executor/runtime limits and a pi/2 rad/s QP cap with a 5 ms shared IK budget. The independent QP-cap code change requires one runtime restart. The original chunk Serial configuration is first. The adjacent README
documents the independent station, frozen control snapshots, training metadata,
normalization checksum and launch commands. `launch.sh check` is offline;
`launch.sh runtime` starts the selected model automatically. The Task display label
is Pack plate, while each model retains its original instruction.

`Joint-FK TCP / 29999` uses the existing relative-TCP adapter.
`Joint / Wrist / 29999` and `Joint / 3 cameras / 29999` use measured absolute
16-D left/right joint state and absolute joint targets through `JointAdapter`.
The 3-camera choice starts a runtime-owned main UVC sensor with its explicit
`device_path` in the private station. It shares the unchanged wrist camera server.
Restart runtime once for these code changes; subsequent YAML-only edits can use
Reload config while idle. No real-weight forward or camera capture was run for
these three exports during integration; existing services were preserved.

The new recipe remains a 20-D TCP-relative, zero-pose, two-wrist-camera profile;
Joint-FK describes training labels. Its training metadata confirms full Gemma
variants, 32 steps at 30 Hz and the existing axis/Rot6D convention. The original
training transforms are not bundled. Real-weight inference and hardware validation
are still separate from contract/configuration checks.

## UMI half-speed H50 checkpoint (2026-11-06)

The existing `checkpoint_comparison_20261028/catalog.yaml` now also contains
`Pi05 / UMI Half Speed H50 / 29999` and its `/ Step DiffIK` group. Existing
UMI Only, UMI DAgger and DP entries retain their configuration and order.
The new checkpoint adds nine choices: chunk Serial 50; RTC 16 with None,
Blend 2 or Waypoint; step Serial 50; step RTC 16 with None or Blend 2;
and step RTC 12 / Beta 10 / Skip 1 with Blend 1 or Blend 2.

The recipe is
`manimux/configs/policy/pi05/tianji/pack_plate/umi-half-speed-h50-step29999.yaml`.
It selects the requested
`pi05-plate-umi-half-b64-h50-30k-dist-p4-20261105-152316-final-29999/checkpoint-29999`
under the station's `/home/jw/Desktop/policy` root. Normalization comes only from
that checkpoint's `assets/umi-half-speed-h50/norm_stats.json`, SHA256
`6dc735554aef5d2d50449a2188bab49eeabdcdfdf7747b97f1886ace5297d39c`.
The Orbax metadata has no LoRA leaves. The recipe retains full Gemma 2B/300M,
20 native coordinates padded to 32 model dimensions and a 50-step horizon.
The operator confirmed half-speed training resampling, 30 Hz action timing,
two wrist cameras, zero-pose state and current-TCP-relative actions on 2026-11-06.
Do not halve the runtime rate. The training source is not bundled;
`train_config_name` reconstructs deployment transforms rather than claiming an
original training job name. The task instruction remains
`place each plate into a bubble wrap sleeve`.

Use the same isolated station at
`.local/pack_plate/pi05_checkpoint_comparison_20261019/station.yaml`:
robot `192.168.1.190`, left/right TacCap `0041s`/`0042s`, camera ports 5556/5555
and managed policy port 8574. No station edits are needed. New rollouts go under
`data/experiments/pack-plate-20261106/` with a separate directory per method.
The existing Home, gripper, motion limits and `execute: true` settings are retained.
Prepare can connect and move hardware; Start rollout executes the model.

From `/home/jw/Desktop/project/manimux`, run this read-only check first:

```bash
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261028/launch.sh check
```

Then run each command in a separate terminal, from the same directory:

```bash
# Terminal 1: wrist camera service (reuse an already compatible camera service).
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261028/launch.sh camera

# Terminal 2: RoboGUI at http://127.0.0.1:8086.
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261028/launch.sh gui

# Terminal 3: catalog runtime; starts and switches policy servers automatically.
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261028/launch.sh runtime
```

Finish any existing rollout and stop its runtime before starting the new runtime.
This integration changes Python code, so an existing runtime needs one restart;
Reload config alone is insufficient. The initial selection remains UMI DAgger.
While idle, choose `Pi05 / UMI Half Speed H50 / 29999` (or its Step DiffIK group),
select the method and apply the selection. Wait for the managed policy to be ready,
then Prepare, Activate Warmup, Start rollout, and Finish. Complete or skip evaluation
before switching again. Do not start a second manual policy server on port 8574.

Validation: the 29-choice/eight-service catalog, adapters and strategies load offline;
the new recipe passes the model interpreter's `--check`; TacCap 0.4.2 imports and
required SDK methods were checked without opening hardware. CPU tests cover H32/H50
native/absolute round trips, default and RTC hooks, metadata, reset and wrong-horizon
rejection. Real-weight forward inference, service readiness and physical trials have
not been run. The broader local suite also has unrelated missing XR1 configuration
and inter-hand checkpoint assets, plus an outdated RTC-default assertion.

## Processed2 native-world H50 with half-speed playback (2026-11-08)

The 1028 catalog now starts with `Pi05 / Processed2 Native World H50 / 29999`.
Its three methods are `Serial 32 / 15 Hz`, `RTC 16 / 15 Hz`, and
`Async 16 / 15 Hz`. All use one managed Pi05 service and the existing DP robot,
camera, executor, Home, gripper and recording settings. Existing entries remain
available. The previous initial UMI DAgger export currently lacks
`checkpoint-40000/params/_METADATA`; restoring that artifact is separate from
this integration. No running service was stopped or switched during integration.

The selecting recipe is
`manimux/configs/policy/pi05/tianji/pack_plate/processed2-native-world-h50-step29999.yaml`.
It points to
`pi05-processed2-native-world-p2-retry-20261108-035036-final-29999/checkpoint-29999`
under the existing station checkpoint root. Its checkpoint-local normalization
asset is `processed2-h50-zero-pose`, SHA256
`4ef9bd5ad46798da7d5f9774e8daa6b09175099cf139874a972a72fb5a64615c`.

The original training configuration was read from QZ job
`705c527b-105b-4fd4-9ecd-ec4d7ab56961`, project directory
`/inspire/hdd3/project/agent-driven-world-model/public/ky26302/pi05-pack-plate-processed2-20261106`.
`configs/train_half.json`, `configs/data_half.json`, the prepared manifest,
`scripts/prepare_pack_plate.py` and `src/pass_ball/openpi_adapter.py` confirm:
full Gemma 2B/300M, H50 at native 30 Hz, two wrist RGB views with a masked base view,
zero-pose model state, normalized absolute grippers, right/left 20-D native order,
first-two-row Rot6D, `axis_change: tool`, and current-TCP-relative action chunks.
The task prompt remains `place each plate into a bubble wrap sleeve`.
The run name does not imply a different deployment world-coordinate transform.
The prepared statistics have different JSON serialization but identical numeric
contents; the remote checkpoint's statistics match the local file byte-for-byte.
Private source copies and hashes are preserved under
`.local/pack_plate/pi05_processed2_20261108/training_source/`, with numerical
comparison results in the adjacent `verification.json`.

Slowdown uses `policy.playback_hz: 15.0`; native `action_dt_s: 1/30` and the
100 Hz robot servo remain unchanged. Each model output retains all 50 action rows.
Serial executes a 32-row prefix (about 2.13 s including the final-row hold), then
requests again. RTC uses a 16-row query window, beta 5 and an initial eight-row
delay estimate; the runtime subsequently updates the estimate. Ordinary Async uses
`request_trigger: progress`, single inflight admission and a 16-row progress window,
retaining the full horizon until replacement. RTC sends remaining-action conditions;
Async sends default sampling requests. Both use latency trimming and no extra blend,
skip or waypoint handoff. A 16-row window is about 1.07 s at 15 Hz; inference and
admission timing also determine actual handoffs. This Pi05 RTC is inference-time
guidance, not DP's trained-prefix sampler.

Read-only check from the ManiMux root:

```bash
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261028/launch.sh \
  check --policy 'Pi05 / Processed2 Native World H50 / 29999'
```

This loads all 36 catalog choices and 13 policy service definitions, then checks
the selected three adapters, strategies and artifacts. Omitting `--policy` still
checks every model and reports missing old artifacts instead of silently skipping them.

After finishing the current rollout and stopping its runtime, restart it once for
the new Python identity gate:

```bash
bash manimux/configs/experiments/pack_plate/pi05/checkpoint_comparison_20261028/launch.sh runtime
```

Reuse the running camera service and RoboGUI at `http://127.0.0.1:8086`.
For a fresh launch, run `launch.sh camera` and `launch.sh gui` in separate terminals
using the same full script path. Runtime starts the selected model automatically
on station port 8574. Select the new policy and method, wait for readiness, then
Prepare, Activate Warmup, Start rollout and Finish. Prepare can move to Home.
The Action playback field defaults to 15 Hz and is fixed per rollout; changing it
before Prepare changes playback without retraining or restarting the model.

Validation includes model `--check`, catalog and adapter loading, synthetic H50
decode at 15 Hz with native 30 Hz wire metadata, old H32/H50 regression tests,
and a numerical comparison against the downloaded QZ source with 50 randomized
dual-arm targets. Observation encoding, relative labels, decoded absolute poses
and RTC conditions agree within 1e-6. No real-weight forward, live service
handshake or physical rollout was run for this checkpoint.

## Original wrist-only contract and artifact

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
