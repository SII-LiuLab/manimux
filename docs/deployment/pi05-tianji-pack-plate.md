# Pi05 pack-plate / Tianji–TacCap

This integration is isolated from the existing UMI-DP pack-plate and Pi05
pass-ball experiments. The default and RTC ManiMux experiments share one Pi05
recipe; both checked-in experiments have `execute: false`.

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
