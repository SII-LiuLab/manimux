# Pi05 pack-plate inter-hand-relative experiment

This deployment selects the tracker-fixed, wrist-only **LoRA** export
`pi05-pack-plate-inter-hand-relative-tracker-fixed-best-20000-final-59999`.
The primary experiment uses step **20000**, the export's best held-out validation
checkpoint (loss 0.0268518791). A separate experiment selects final step **59999**.
Neither experiment inherits the other experiment's settings.

## Independent configuration

The entry points are:

- step 20000 experiment: `manimux/configs/experiments/{interhand-step20000-study}/pack_plate/rtc.yaml` (station-local; build both experiments like the [example study](../../../../manimux/configs/lists/example-yam-pi05.yaml))
- step 59999 experiment: `manimux/configs/experiments/{interhand-step59999-study}/pack_plate/rtc.yaml`
- [step 20000 model recipe](../../../../manimux/configs/policy/pi05/tianji/pack_plate/inter-hand-relative-step20000.yaml)
- [step 59999 model recipe](../../../../manimux/configs/policy/pi05/tianji/pack_plate/inter-hand-relative-step59999.yaml)
- [frozen robot assembly and Home](../../../../manimux/configs/embodiment/robot/tianji_taccap_inter_hand_relative.yaml)
- Private station: `.local/tianji_pi05_pack_plate_inter_hand_relative.yaml`.

The local station is already configured on this installation. It supplies device
bindings, the checkpoint storage root `{tianji_workspace}/policy`, and separate
model endpoints: `8572` for best and `8573` for final. The experiments reference
this station explicitly and do not fall back to the shared station. On another
installation, provide its replacement with `--local`.

Each experiment freezes its own RTC, adapter, executor and command safety
settings. There are no references to an existing experiment, inference preset,
control profile or executor preset. Both model recipes independently declare
their base transforms and normalization identity. Device implementations and
component geometry are shared. Home and assembly hardware settings are copied
into the new assembly so edits to the older robot assemblies do not change this
experiment's start pose or controller ratios.

Initial values follow the requested guarded-live configuration's intent, using
current field names: task `plate`, control 100 Hz, actions 30 Hz, horizon 32,
10 denoising steps, RTC minimum execution 18 policy steps, initial delay 8,
beta 10, `handoff: blend`, skip 2 and blend 2. The executor keeps an 8 Hz filter,
arm velocity 0.5 rad/s, 16 ms maximum step interval, and gripper curve deadzone
0.005 / exponent 1.1. Differential IK uses 4 ms validation steps and reports lag
above 5 mm. `execute_diff_ik_substeps: false` retains the guarded-live output
resolution. Both grippers use `force_position` with `grasp_torque_nm: 1.0` and
`close_speed_radps: 1.1`, as requested after initial deployment. These values were
copied from `.local/pack_plate/tianji-pi05-full-zero-state-pack-instruction-step59999.yaml`
and are declared independently in both new experiments; the reference file is
not loaded or modified. Restart the runtime after changing the control mode.

The camera stream names are `taccap_left` and `taccap_right`, matching the current
station service. They map to the same left/right wrist inputs as the old
`left_wrist` / `right_wrist` names. The declared capture configuration uses the
same physical wrist cameras; do not start a second capture owner for them.

The live experiment files declare `execute: true` and
`end_effector_control: true`. Starting the interactive service creates a session;
**Prepare can connect hardware and move to the configured Home**. Only one
runtime may own this robot. No existing process is stopped or switched by
creating these files.

## State, action anchor and IK seed

The model's 20-D state remains right hand first, then left: xyz (metres),
Rot6D (first two rows), and absolute normalized gripper opening for each hand.
For common-frame TCP transforms `T_R` and `T_L`, the model receives
`inv(T_L) @ T_R` in the right block and `inv(T_R) @ T_L` in the left block.
Training tool axes are applied before forming these relative transforms.

The data path is:

1. The existing ManiMux adapter computes both TCPs from the measured observation
   joints in their respective arm bases, including the configured TacCap TCP.
2. The existing model encoder preserves those original per-arm poses as the
   request's action anchors. It converts any RTC condition to actions relative
   to these anchors before changing the model state.
3. `InterHandRelativeState` applies `state_base_to_common`, computes the two
   inter-hand poses, and preserves both gripper openings. This is a copied state
   array, so the original anchors are not mutated.
4. The checkpoint's state normalization and Pi05 state tokenization run normally.
   The zero-pose post-normalization override is omitted for this variant.
5. Predictions are decoded relative to each hand's original observation TCP.
   Absolute targets sent to differential IK remain in their individual arm bases.
   The adapter still declares `decode_seed_source = execution_reference`; runtime
   reference selection and its missing-reference fallback are unchanged.

This is the existing Pi05 inference-time RTC sampler, not the UMI-DP train-time
RTC deployment. `handoff_skip_steps` still acts on the decoded trajectory.

## Geometry and artifact provenance

The recipe matrices mean `common <- arm base`. The common frame is the CAD
`Link_Base`; the mounting chain is `Link_Base -> Link_Stand -> Base_L/Base_R`.
The left mount has y = +0.037 m and Rx(-pi/2); the right has y = -0.037 m and
Rx(+pi/2); both have z = 1.121 m. Equivalently, right-base coordinates transformed
into the left base use rotation diag(1,-1,-1) and translation [0,0,-0.074] m.
These constants belong to the model recipe and backend identity, not RoboGUI.

The source whole-body URDF is under the supplied teleop export's
`generated_urdf/body_source/Marvin M6-S-CCS-696-V4.0_Base_and_Stand_Asm urdf/urdf/`
directory. Its 1.5708 rad CAD angles are represented here by exact pi/2. The
previous CAD/SDK FK cross-check supports this geometry; it is not a new physical
calibration. A joint-independent common-world transform cancels from the
inter-hand state, so alignment with the recording world's origin is unnecessary.

Both recipes load normalization from their own checkpoint's
`assets/pack-plate-taccap-h32-inter-hand-relative/norm_stats.json` and pin its
SHA256 to `a1e7da1b33b9399c3fb64a32e069409e09138e8c02ffb0aae7a87f206d433d5c`.
The actual exported asset and `SHA256SUMS` determine this digest; the prepared-data
report contains a different earlier normalization digest. The inter-hand manifest
and report determine the state contract; the copied zero-pose coordinate document
does not override them. Action statistics and the current-TCP-relative action
representation are unchanged by the state variant.

## Launch

From the ManiMux repository root, select one experiment:

```bash
export PI05_INTERHAND_EXPERIMENT=manimux/configs/experiments/{interhand-step20000-study}/pack_plate/rtc.yaml

.venv/bin/python -m manimux.servers.pi05 \
  --experiment "$PI05_INTERHAND_EXPERIMENT" --check
```

When the model resources and robot ownership have been handed over, start the
policy server and the interactive runtime in separate terminals. Set the same
experiment variable in each terminal:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --experiment "$PI05_INTERHAND_EXPERIMENT"
```

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python -m manimux run --config "$PI05_INTERHAND_EXPERIMENT"
```

This installation uses the project's already-built TacCap 0.4.2 SDK. The explicit
`PYTHONPATH` above replaces inherited SDK paths for this process. The failed
runtime had inherited `{tianji_workspace}/project/TacCap-Gripper/python`, loading
the external 0.1.9 native extension, whose `Motor` has no `get_model()` method.
The root `.venv` does not have an independently installed TacCap package, so
simply unsetting `PYTHONPATH` is insufficient here. The model-server command does
not need this hardware SDK path.

The local SDK's Python module, native extension and required `Motor.get_model`,
`Motor.get_spec`, `FollowerGripper.audit_envelope` and controller methods were
checked without opening devices. Restart an already-running runtime with the
corrected command: changing the shell environment cannot replace a native library
already loaded in that process. The SDK import can be checked independently:

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python -c 'import xense.taccap as t; print(t.__version__, t.__file__); assert hasattr(t.Motor, "get_model")'
```

Reuse the current timestamped camera service when it publishes the declared
streams; the current camera process already uses the local 0.4.2 SDK. If no capture
owner is running, use the same SDK path for its startup:

```bash
PYTHONPATH="$PWD/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python" \
.venv/bin/python -m manimux.servers.camera.server --config "$PI05_INTERHAND_EXPERIMENT"
```

For the final checkpoint, select the step59999 experiment in both terminals.
Do not change the best experiment's checkpoint field: artifact identity, port,
condition and output directory are already paired in the final experiment.

Recordings go to separate `data/experiments/pi05-tianji-pack-plate-inter-hand-relative-step{20000,59999}-rtc`
directories. Research metadata identifies experiment
`pi05-pack-plate-inter-hand-relative`, condition `step20000-rtc` or `step59999-rtc`.
The common recorder saves resolved configuration and backend identity, including
the model state encoding and base transforms. No task score or success claim has
been assigned to this deployment.

## Validation completed on 2026-10-11

Both real experiment loads, station bindings, artifact checks, normalization
digests and differential-IK adapter construction passed without connecting devices.
CPU-only abstract model checks matched all 71 parameter leaves in each Orbax
checkpoint, including 20 LoRA leaves. This checks names and shapes, not weight
restoration or neural inference.

The focused local suites passed 34 tests: independent homogeneous-transform
calculations, reciprocal poses, common-frame invariance, anchor preservation,
RTC condition round trips, actual OpenPI normalization/tokenization for both
checkpoints, old zero-pose behavior, backend identity and existing IK handoffs.
All pre-existing experiment/station YAML hashes remained unchanged.

The existing Pi05 server and robot runtime were active during this work. No new
GPU forward, server readiness check or physical rollout was run. The existing
offline forward tool can verify weight restoration and neural output when model
resources are available:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  -m XPolicyLab.policy.Pi_05.offline_pack_plate \
  --config manimux/configs/policy/pi05/tianji/pack_plate/inter-hand-relative-step20000.yaml \
  --checkpoint {tianji_workspace}/policy/pi05-pack-plate-inter-hand-relative-tracker-fixed-best-20000-final-59999/checkpoints/20000
```
