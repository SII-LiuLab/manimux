# UMI Diffusion Policy on Tianji–TacCap

Start with the [local station guide](../manimux/configs/local/README.md) when connecting
another Tianji–TacCap installation. One private station file supplies the controller IP,
gripper and camera serials, service addresses and local artifact paths. The experiment
selects the robot assembly, policy adapter, inference algorithm and execution settings.

## Prepare the environments

Run the commands below from the repository root. Hardware and model processes use
separate Python environments; see [Python environments](../envs/README.md).

- Install this ManiMux checkout and its `xpolicylab` extra in the Tianji hardware
  environment. The commands below use `envs/tianji/.venv/bin/python`.
- Obtain the private Marvin SDK from the Tianji installation owner. The current arm
  implementation imports `manimux.embodiments.arm.tianji.sdk.marvin.fx_robot` for control
  and `fx_kine` for kinematics. Place the supplied wrappers, required native libraries
  and `ccs_m6_40.MvKDCfg` under that component's `sdk/marvin/` layout. The SDK is not
  installed by filling a station file and is absent from a clean checkout.
- Install the TacCap native package `xense.taccap` into the hardware environment,
  following the SDK's installation instructions. Both the gripper and wrist camera
  use this dependency; see the [TacCap component](../manimux/embodiments/end_effector/taccap/README.md).
- Initialize the repository's XPolicyLab submodule and use its UMI_DP installation
  entry point in a separate model environment:

```bash
git submodule update --init --recursive XPolicyLab
bash XPolicyLab/policy/UMI_DP/install.sh envs/umi_dp/.venv
```

These paths name environments to prepare; cloning the repository does not create them.
The station and configuration commands below do not install SDKs or test hardware.

## Create the station file

```bash
cp -n manimux/configs/local/tianji_taccap.example.yaml manimux/configs/local/station.yaml
```

Read an existing `station.yaml` before changing it. Fill these fields with confirmed
bindings for the local installation:

| Field | Meaning |
| --- | --- |
| `robot.hardware.ip` | Shared Tianji controller IP for both arms |
| `robot.components.left_end_effector.serial` / `right_end_effector.serial` | TacCap gripper firmware serial; the driver also matches follower role and side |
| `robot.components.left_wrist_camera.camera_serial` / `right_wrist_camera.camera_serial` | UVC camera serial used to select its IMX385 capture node under `/dev/v4l/by-id` |
| `services.policy.endpoint` | Runtime-accessible UMI_DP WebSocket address, normally `ws://127.0.0.1:8560` |
| `services.camera.endpoint` | Timestamped camera subscription address, normally PUB `tcp://127.0.0.1:5556` |
| `services.camera.request_endpoint` | Camera request address, normally REP `tcp://127.0.0.1:5555` |
| `paths.checkpoint` | Trusted UMI artifact path to bind |
| `paths.output_dir` | Optional run-output override |

Gripper firmware serials and camera UVC serials are separate identifiers. Establish
physical left/right placement before assigning them; enumeration order is not a mapping.
Paths under `paths` resolve relative to the station file, or may be absolute.

For remote services, client endpoints contain reachable host addresses. Set
`services.policy.bind_host` and camera `bind_endpoint` / `bind_request_endpoint`
when their listen addresses differ. Tianji consumes PUB frames on port 5556; port 5555
is the separate request socket. A remote camera server also needs clock alignment within
the experiment's state/camera tolerances, because timestamps are backend host receipt times.

Runtime and camera/UMI `--experiment` entry points select the station in this order:
`--local <path>`, the experiment's `local:` reference, then
`manimux/configs/local/station.yaml`. A missing selected file produces the normal file-read
error. Use `--local` only to select another station. Pure `read_experiment()` and
`load_config()` calls still allow offline inspection without an implicit station.

The station does not select TCP geometry, the adapter, control frequency or execution
switches. These remain in the assembly and experiment.

## Select and bind the experiment

The following recipes all support the shared station file:

| Experiment under `manimux/configs/experiments/pass_ball/umi_dp/` | Scheduling |
| --- | --- |
| `tianji_taccap_umi_dp.yaml` | Component-based experiment with `manimux` scheduling |
| `tianji_taccap_umi_dp_diff.yaml` | Component-based `manimux` experiment using differential IK |
| `tianji_taccap_umi_dp_diff_live.yaml` | Execution-enabled DiffIK experiment with Viewer-controlled rollouts |
| `tianji_umi_dp_default.yaml` | Existing `manimux` recipe and shared control profile |
| `tianji_umi_dp_rtc.yaml` | RTC recipe with process action decoding |

Each recipe maps camera-server streams `left_wrist` / `right_wrist` to assembly components
`left_wrist_camera` / `right_wrist_camera`. The component-based recipe also renames the
runtime images to the component names; the other two retain their existing stream names.
Their `policy.adapter.camera_map` matches the corresponding image names.

Bind checkpoint identity in the model environment before launching a runtime:

```bash
envs/umi_dp/.venv/bin/python -m manimux.servers.umi_dp \
  --experiment manimux/configs/experiments/pass_ball/umi_dp/tianji_taccap_umi_dp.yaml \
  --bind-runtime-config .local/pass_ball/run.yaml
```

Select `tianji_taccap_umi_dp_diff.yaml` for differential IK or
`tianji_umi_dp_rtc.yaml` for RTC. Append `--local <station.yaml>` when selecting another
station. Add `--check` without `--bind-runtime-config` to inspect artifact identity without
exporting a pair.

For a reviewed real-robot DiffIK deployment, select the explicit live recipe and keep the
bound pair beside the private station file:

```bash
envs/umi_dp/.venv/bin/python -m manimux.servers.umi_dp \
  --experiment manimux/configs/experiments/pass_ball/umi_dp/tianji_taccap_umi_dp_diff_live.yaml \
  --local manimux/configs/local/station.yaml \
  --bind-runtime-config manimux/configs/local/deployments/tianji_taccap_umi_dp_diff_live.yaml
```

The live recipe selects `tianji_control_live.yaml`, enables arm and end-effector commands,
and enables Viewer-controlled rollouts. It does not copy controller addresses, serials or
checkpoint paths out of the private station. The non-live recipes remain read-only defaults.

Binding reads the actual artifacts and records checkpoint identity, horizon, observation
period, first-action offset and preprocessing conventions. The checkpoint action interval
must match the experiment; binding does not silently change it. It starts no model,
camera or robot service. Existing output files are not overwritten.

Binding writes the requested runtime path and a sibling whose name ends in
`-server.yaml`. In the live example these are
`manimux/configs/local/deployments/tianji_taccap_umi_dp_diff_live.yaml` and
`manimux/configs/local/deployments/tianji_taccap_umi_dp_diff_live-server.yaml`:

- The runtime file retains an absolute reference to the selected station. Runtime and
  `--experiment` service launches reread its current bindings; hardware identifiers
  are not copied into the exported runtime.
- The `-server.yaml` file is a standalone resolved snapshot. Launching it with `--config`
  uses the saved addresses and artifact path, without consulting the station.

After changing service addresses, use `--experiment` to read the updated station or
regenerate the standalone snapshot. After changing the checkpoint, bind a new pair so
its expected identity matches the selected artifacts. Do not bypass identity checks.

## Start the services

Use the bound experiment for the model, camera and runtime roles. The following commands open services or
hardware and belong to an intended deployment session.

Start the model in its environment:

```bash
envs/umi_dp/.venv/bin/python -m manimux.servers.umi_dp \
  --config manimux/configs/local/deployments/tianji_taccap_umi_dp_diff_live-server.yaml
```

Start the camera service on the computer with the wrist cameras:

```bash
envs/tianji/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/local/deployments/tianji_taccap_umi_dp_diff_live.yaml
```

Start the hardware runtime:

```bash
envs/tianji/.venv/bin/python -m manimux serve \
  --config manimux/configs/local/deployments/tianji_taccap_umi_dp_diff_live.yaml
```

Start Viewer after the runtime is listening:

```bash
envs/tianji/.venv/bin/python -m manimux.viewer.dashboard \
  --robot tianji --host 127.0.0.1 --port 8086
```

The camera and runtime read the station referenced by the bound experiment. The model
command intentionally uses the standalone server snapshot whose checkpoint identity was
verified while binding. For another station, regenerate the pair with that station before
starting the services. `manimux serve` keeps the runtime available for Viewer-controlled
rollouts; use `manimux run` only for an immediate single session.
Open `http://127.0.0.1:8086`, then use **Prepare normal rollout → Start rollout →
Finish rollout**. `Start rollout` begins real command execution; Tianji Home remains a
separate recovery action.

The non-live experiments default to `robot.options.execute: false` and
`robot.options.end_effector_control: false`; runtime still connects and reads feedback.
The explicit `tianji_taccap_umi_dp_diff_live.yaml` recipe sets both fields and
`viewer.enabled` to true. Execution settings belong to the experiment, not the station.
Tianji connection does not Home; the existing controller enables on the first executed
command. See [Viewer](viewer.md) for its separate display and control interface.

## Preserved action and timing conventions

- Groups are `left_arm` and `right_arm`: seven arm joints in radians followed by one
  normalized gripper opening, zero closed and one open. Arm A is left; arm B is right.
- UMI outputs absolute TCP poses in each arm's own base frame, with translation in
  metres and quaternion order WXYZ. ManiMux applies the configured tool transform
  and IK. Viewer placement does not enter FK/IK.
- These experiments command at `robot.control_hz: 100` with model action spacing
  `policy.action_dt_s: 1/30` seconds. Model action spacing and command frequency are
  independent. The first-action offset is bound from the matching checkpoint.
- Observation history, camera mapping, IK selection, smoothing and motion limits
  retain each recipe's existing values. Local binding changes device and service
  connections without selecting different control behavior.

This configuration update was checked with pure configuration tests and a fake artifact
provider. No Tianji numerical, SDK, hardware or physical-task validation was rerun.
For the existing model/action contract details and historical evidence, see the
[UMI deployment reference](umi_dp-tianji-runbook.md) and
[validation report](umi_dp-tianji-validation.md).
