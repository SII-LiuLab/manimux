# Connect your local robot

For your own installation of a supported robot, start here. **One private station file
binds the existing components to your CAN interfaces, controller IPs, USB serials and
service addresses.** Keep task, action format, inference algorithm and control settings
in the experiment. You do not need a new driver for another robot of the same model.

The default file is `manimux/configs/local/station.yaml`. It is ignored by Git and excluded
from packages. Runtime startup and the camera, Pi05 and UMI_DP `--experiment` entry points
read it automatically. Pi05's standalone `--config` entry also resolves local path/service bindings. `--local <path>` selects another station.

## 1. Select the robot and prepare its environment

All paths and commands below are relative to the repository root. Use the Python
environment with the selected hardware dependencies installed.

| Robot | Station template | Dependencies and deployment |
| --- | --- | --- |
| YAM with integrated grippers | [yam.example.yaml](yam.example.yaml) | [YAM component](../../embodiments/arm/yam/README.md), [RealSense](../../embodiments/sensor/realsense/README.md), [Pi05 runbook](../../../docs/pi05-yam-runbook.md) |
| Tianji–TacCap | [tianji_taccap.example.yaml](tianji_taccap.example.yaml) | [UMI-DP runbook](../../../docs/umi-dp-tianji-taccap-runbook.md) · [Xiaomi XR-1 runbook](../../../docs/xiaomi-xr1-tianji-taccap-runbook.md) |

Hardware and model environments are separate; see [Python environments](../../../envs/README.md).
A clone does not install private SDKs, create CAN interfaces or download checkpoints.

## 2. Create your station file

Choose **one** template. For YAM:

```bash
cp -n manimux/configs/local/yam.example.yaml manimux/configs/local/station.yaml
```

For Tianji–TacCap:

```bash
cp -n manimux/configs/local/tianji_taccap.example.yaml manimux/configs/local/station.yaml
```

`cp -n` preserves an existing file. Read and reuse existing bindings before changing them.
Fill in your actual devices and paths; example addresses and placeholder serials do not
identify your robot. For multiple stations, keep additional files under the root's ignored `.local/` directory
and select one with `--local`; only the default `station.yaml` is ignored inside this
configuration directory.

## 3. Fill in the hardware and service bindings

| Connection | Station field | Value |
| --- | --- | --- |
| YAM CAN | `robot.components.left_yam.channel` / `right_yam.channel` | Actual Linux CAN interface, such as `can0` / `can1` |
| Tianji shared controller | `robot.hardware.ip` | Controller IP; the two arms share the connection |
| USB camera | `robot.components.<component>.camera_serial` | Camera serial number |
| TacCap gripper | `robot.components.<component>.serial` | Gripper serial number |
| Component using a serial port | `robot.components.<component>.port` | The component's `/dev/tty…` or stable device path |
| Policy client | `services.policy.endpoint` | Reachable server address, such as `ws://127.0.0.1:8500` |
| Camera clients/server | `services.camera` | Request, subscription and bind addresses, as below |
| Pi05 checkpoint root | `paths.checkpoints` | Local root containing the checkpoint/stat subpaths selected by the experiment |
| UMI_DP artifact | `paths.checkpoint` | Local checkpoint directory used by the UMI_DP artifact binding |
| XR-1 artifact | `paths.checkpoint` | Local checkpoint file or directory containing `mp_rank_00_model_states.pt` |
| XR-1 normalization | `paths.norm_stats` | Optional explicit `training_metadata/normalize.json` path |
| XR-1 processor | `paths.vlm_processor` | Optional local Qwen processor directory; otherwise use the recipe's repository ID |
| Run output override | `paths.output_dir` | Optional local directory for experiment output |

Only fill in fields the selected component actually uses. YAM's integrated gripper shares
its arm's CAN connection: configure `left_yam` and `right_yam`, without a separate gripper.
Tianji's controller IP and the TacCap gripper/camera serials are independent bindings.

`left_yam` is a component name in the robot assembly; `can_left` is an operating-system
interface name. The loader passes the latter to `YamController(channel=...)`, then to
`i2rt.get_yam_robot(channel=...)`. Python does not create or rename that interface.
Our development station uses udev rules to name USB-CAN adapters `can_left` / `can_right`;
if yours uses `can0` / `can1`, enter those names instead.

Read the existing CAN interface names without changing them:

```bash
ip -details link show type can
```

With the RealSense dependency installed, list the visible camera serials:

```bash
envs/yam/.venv/bin/python -c 'from manimux.embodiments.sensor.realsense import get_device_ids; print(get_device_ids())'
```

Component keys must match the assembly referenced by the experiment's `robot.config`.
Enumeration order does not establish physical left/right placement. Use the station's
known mapping, or confirm placement before assigning a device. Binding another installation
requires configuration changes, not edits to SDK code or `self.channel`.

### Service addresses

Use `127.0.0.1` when the client and server run on the same computer. If Pi05 runs on another
GPU computer, the policy endpoint must use that computer's reachable IP. A server can use
`bind_host: 0.0.0.0` to listen on its network interfaces; a client uses the actual destination IP.

The current camera templates use different transport modes:

- YAM requests frames over REP: `endpoint` and `request_endpoint` both use port `5555`.
- Tianji's timestamped client subscribes to PUB: `endpoint` uses port `5556`, while
  `request_endpoint` uses port `5555`.
- `bind_endpoint` and `bind_request_endpoint` are the server's PUB and REP listen addresses.
  Match each client port to the corresponding server port when changing them.

The cameras connect over USB to the camera server's computer; the runtime connects to that
service over the network. Cross-machine cameras also need the clock alignment required by
the timestamped client. Viewer network options remain separate, as listed below.

### Local paths and model identity

Paths under `paths` resolve relative to the station file; absolute paths work too.
`paths.checkpoints` is the root for Pi05 checkpoints. The selected experiment/server recipe
keeps the relative `model_path` and `norm_stats_path`; the loader resolves both against this
root and uses the resolved paths in the runtime's expected backend identity. Switching
experiments still selects that experiment's checkpoint, instead of reusing one globally
overridden artifact. UMI_DP retains `paths.checkpoint` for its explicit artifact binding.
The XR-1 Tianji recipe uses the same key for its checkpoint and additionally accepts
`paths.norm_stats` and `paths.vlm_processor`. These fields do not change training
configuration, normalization convention or action semantics.

## 4. Inspect the resolved configuration without connecting devices

This uses the same station selection as startup, then only reads configuration:

```bash
envs/yam/.venv/bin/python - <<'PYCODE'
from pprint import pprint
from manimux.cli import load_config, resolve_local_path
from manimux.servers.camera.server import camera_config

experiment = "manimux/configs/experiments/put_bottles/yam_pi05_rtc_joint_step30000.yaml"
config = load_config(experiment, local=resolve_local_path(experiment))
pprint({
    "local": str(config["local"]),
    "shared_hardware": config["robot"]["options"].get("hardware", {}),
    "components": config["robot"]["options"]["component_hardware"],
    "runtime_camera": config["sensors"][0]["options"]["endpoint"],
    "policy": config["policy"]["options"]["server"],
    "camera_server": camera_config(config)["sensors"]["cameras"],
})
PYCODE
```

For Tianji, use its Python environment and replace the experiment path with
`manimux/configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml`.
The shared controller IP appears under `shared_hardware.ip`. This inspection does not
construct the robot, open devices or test Tianji FK/IK.

The loading chain is:

```text
experiment YAML + selected station file
    -> read_experiment(): resolve references and apply bindings
    -> load_config(): add runtime defaults
    -> robot / camera / policy constructor parameters
    -> SDK or network client when the relevant process connects
```

Station selection is `--local` first, then an experiment's explicit `local:` reference,
then `manimux/configs/local/station.yaml`. A CLI path is relative to the working directory;
an experiment reference is relative to that experiment. A missing station file produces
the normal file-read error at startup. Pure `read_experiment()` / `load_config()` calls do
not implicitly require a station, so offline tools can still inspect shared recipes.

## 5. Start the selected experiment

Runtime, camera and model services must use the same experiment and station:

```text
runtime:        python -m manimux serve --config <experiment.yaml>
camera server:  python -m manimux.servers.camera.server --experiment <experiment.yaml>
Pi05 server:    python -m manimux.servers.pi05 --experiment <experiment.yaml>
UMI_DP server:  python -m manimux.servers.umi_dp --experiment <experiment.yaml>
XR-1 Tianji:    python manimux/servers/xiaomi_xr1_tianji_server.py --experiment <experiment.yaml>
```

Select the appropriate model server, using its own Python environment. Append
`--local <path>` to each command when choosing a different station. For complete Pi05
commands, see the [startup guide](../../../docs/guideline.md#pi05-30k-on-yam).

The current `yam_pi05_rtc_joint_step30000.yaml` has `execute: true`,
`move_to_start_on_connect: true` and `control_hz: 30`. The separate assembly example
`yam_pi05_joint.yaml` uses `execute: false` and `control_hz: 100`.
Binding a station does not switch these experiment settings.

### Scope and remaining independent entry points

| Entry point | Station-file behavior |
| --- | --- |
| Runtime `run` / `serve` | Reads the selected station automatically for CAN, controller IP, components and client addresses |
| Camera `--experiment` | Uses the experiment's named camera components and the same station serials/listen addresses |
| Pi05 `--experiment` or `--config` | Uses the selected station's policy service and checkpoint root |
| UMI_DP `--experiment` | Uses the same policy service and checkpoint binding |
| XR-1 Tianji launcher | Uses the selected station's policy service, checkpoint, normalization and optional processor bindings |
| Camera / UMI_DP standalone `--config` | Reads that standalone server configuration; use `--experiment` for shared station bindings |
| Viewer process | Still uses its own launch options for network addresses and the web port |
| YAM collection | Still uses its own collection station file, including leader-device settings |
| Other model launchers | Follow their model runbooks; this change does not add shared station loading to every launcher |

UMI_DP's `--bind-runtime-config` generates a paired runtime and server configuration.
The generated runtime retains its station reference; the standalone server YAML contains
the resolved values from that generation. After changing addresses or checkpoints, regenerate
the pair or launch through `--experiment` to read the current station bindings.

The current YAM RTC joint-step30000 and generic assembly examples, plus the Tianji–TacCap
experiments, declare named camera components for shared station loading. Other historical
YAM recipes may still contain direct addresses or lack a `camera_server` mapping. Use a
recipe with these bindings when launching all services from one experiment; the loader
does not infer missing camera assembly from old addresses.

SDK installation, model readiness and physical task validation are separate from configuration
loading. A successful configuration inspection establishes the resolved bindings only.

## For coding agents

1. Read this guide, the component README, the chosen experiment and any existing station file.
2. Identify the actual CAN/USB devices and service hosts; preserve confirmed physical mappings.
3. Fill the same private station file, preserving unrelated bindings and experiment semantics.
4. Inspect the resolved device/service configuration without opening hardware.
5. Provide or run startup commands according to the user's requested scope. Configuration
   work alone does not instruct you to start services or control the robot.
