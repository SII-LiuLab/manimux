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

| Experiment under `manimux/configs/experiments/pass_ball/` | Scheduling |
| --- | --- |
| `tianji_taccap_umi_dp.yaml` | Component-based experiment with `manimux` scheduling |
| `tianji_umi_dp_default.yaml` | Existing `manimux` recipe and shared control profile |
| `tianji_umi_dp_rtc.yaml` | RTC recipe with process action decoding |

Each recipe maps camera-server streams `left_wrist` / `right_wrist` to assembly components
`left_wrist_camera` / `right_wrist_camera`. The component-based recipe also renames the
runtime images to the component names; the other two retain their existing stream names.
Their `policy.adapter.camera_map` matches the corresponding image names.

Bind checkpoint identity in the model environment before launching a runtime:

```bash
envs/umi_dp/.venv/bin/python -m manimux.servers.umi_dp \
  --experiment manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml \
  --bind-runtime-config .local/pass_ball/run.yaml
```

Select `tianji_umi_dp_rtc.yaml` in that command for RTC. Append `--local <station.yaml>`
when selecting another station. Add `--check` without `--bind-runtime-config` to inspect
artifact identity without exporting a pair.

Binding reads the actual artifacts and records checkpoint identity, horizon, observation
period, first-action offset and preprocessing conventions. The checkpoint action interval
must match the experiment; binding does not silently change it. It starts no model,
camera or robot service. Existing output files are not overwritten.

The command writes `.local/pass_ball/run.yaml` and `run-server.yaml`:

- `run.yaml` retains an absolute reference to the selected station. Runtime and
  `--experiment` service launches reread its current bindings; hardware identifiers
  are not copied into the exported runtime.
- `run-server.yaml` is a standalone resolved snapshot. Launching it with `--config`
  uses the saved addresses and artifact path, without consulting the station.

After changing service addresses, use `--experiment` to read the updated station or
regenerate the standalone snapshot. After changing the checkpoint, bind a new pair so
its expected identity matches the selected artifacts. Do not bypass identity checks.

## Start the services

Use the bound experiment for all three roles. The following commands open services or
hardware and belong to an intended deployment session.

Start the model in its environment:

```bash
envs/umi_dp/.venv/bin/python -m manimux.servers.umi_dp \
  --experiment .local/pass_ball/run.yaml
```

Start the camera service on the computer with the wrist cameras:

```bash
envs/tianji/.venv/bin/python -m manimux.servers.camera.server \
  --experiment .local/pass_ball/run.yaml
```

Start the hardware runtime:

```bash
envs/tianji/.venv/bin/python -m manimux run \
  --config .local/pass_ball/run.yaml
```

All three read the station referenced by the bound experiment. For another station,
append the same `--local <path>` to each command. To intentionally launch the standalone
model-server snapshot instead, use:

```bash
envs/umi_dp/.venv/bin/python -m manimux.servers.umi_dp \
  --config .local/pass_ball/run-server.yaml
```

The experiment defaults to `robot.options.execute: false` and
`robot.options.end_effector_control: false`. Runtime still connects and reads feedback.
Execution settings belong to the experiment, not the station. Tianji connection does
not Home; the existing controller enables on the first executed command. The component
assembly does not implement Home or manual drag recovery. See [Viewer](viewer.md) for
its separate display and control interface.

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
