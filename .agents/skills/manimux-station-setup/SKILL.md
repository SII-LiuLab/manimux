---
name: manimux-station-setup
description: Connect a ManiMux installation to a developer's local YAM, Tianji or newly integrated robot and cameras. Use for SDK environments, CAN or controller addresses, device serials and local configuration binding, rather than experiment scheduling.
---

# ManiMux Station Setup

Paths are relative to the repository root. Read the
[repository guide](../../../docs/development/agent-guide.md) before editing. Start with
[`docs/usage/station.md`](../../../docs/usage/station.md).
Installing ManiMux does not discover, name or connect a robot automatically. Bind existing
components to the actual devices; do not copy a developer's addresses, serials or paths as
universal defaults.

## One private station file

The default is `manimux/configs/local/station.yaml`, ignored by Git and excluded from
packages. Copy either `yam_example.yaml` or `tianji_taccap_example.yaml` from the same
directory if no station file exists. Read an existing file before changing it and preserve
unrelated bindings. Device keys must match the selected assembly's component names.

Physical runtime startup and camera/Pi05/UMI_DP/XR-1 Tianji `--experiment` entry points
resolve the station in this order: CLI `--local`, experiment `local:`, default station path. CLI paths
are relative to the working directory, experiment references to that YAML, and `paths`
values to the station file. Use the same experiment and station for the three processes.
Pi05 and camera-server `--config` also read the selected station. UMI_DP standalone
`--config`, RoboGUI sockets, other model launchers retain separate
configuration paths; do not claim they all use the station file. Some historical YAM
experiments still lack named camera/service mappings; check the selected recipe instead
of assuming every existing experiment supports camera `--experiment` startup.

## Hardware connection paths

| Robot | Connection | Station bindings |
| --- | --- | --- |
| YAM | `YamRobot` → `YamController` → `i2rt.get_yam_robot(channel=...)` | `robot.components.left_yam.channel` and `right_yam.channel` |
| Tianji | `TianjiTaccapRobot` → `TianjiController` → Marvin SDK | Shared `robot.hardware.ip` |
| TacCap grippers | End-effector components → TacCap SDK | Each component's `serial` |
| RealSense/TacCap cameras | Camera server → camera components → USB devices | Each component's `camera_serial` |

CAN values must name actual Linux interfaces. `can_left` is a station-specific name,
not an interface created by installation. Determine physical left/right mapping; enumeration
order is insufficient. YAM's integrated grippers use their arm CAN connections and require
no separate end-effector binding. Tianji's arm bindings belong under
`manimux/embodiments/arm/tianji/sdk/`; TacCap needs its native dependency installed.
Neither robot requires a separate original control/reproduction repository at runtime.

Follow physical camera → assembly component → camera-server stream name → runtime sensor
name → `policy.adapter.camera_map`. A camera serial and a gripper serial identify different
devices. Follow the chosen template's REP/PUB addressing: current YAM uses requests on
port 5555, while Tianji's timestamped client subscribes on 5556. A manual RoboGUI preview
does not alter model inputs. Confirm unknown placement with the user or an authorized preview.

RoboGUI preset names and runtime robot identities are distinct: `--robot tianji` selects
the RoboGUI preset, while the Tianji assembly is named `tianji-taccap`. Experiments must
use `robogui.robot: tianji-taccap` so RoboGUI message filtering and runtime ownership agree.

## Environments and model paths

Read `docs/usage/environments.md` and the component/model runbook. Existing `envs/*/.venv` paths are
local assets, not installed by cloning. Target hardware/model interpreters explicitly with
`uv pip install --python`; do not point root `uv sync` or `uv run` at those environments.
Do not invent a single all-hardware dependency extra.

`paths.checkpoints` supplies Pi05's checkpoint root; the selected experiment/server recipe
chooses relative checkpoint and normalization paths within it. UMI_DP retains the explicit
`paths.checkpoint` artifact binding. XR-1 Tianji also uses `paths.checkpoint`, with optional
`paths.norm_stats` and `paths.vlm_processor`; see its deployment runbook linked from the
station guide. These relocate selected artifacts without replacing the experiment's
checkpoint identity, transforms or action conventions.
The SDK/model dependency environment is separate from these configuration bindings.

## Inspect without connecting

Use `resolve_local_path(experiment)` with `load_config(experiment, local=...)` and inspect
`robot.options.hardware`, `component_hardware`, client endpoints and `camera_config(config)`.
The station guide contains a complete example. Pure configuration readers do not implicitly
require a local station; real startup selects it explicitly through the shared resolver.
Do not construct or connect hardware just to inspect addresses. Tianji configuration
inspection does not require revalidating its FK/IK or its team's hardware validation.

Keep action semantics, TCP/FK/IK, timing, limits and execution switches unchanged when
binding the same robot model. The current YAM Pi05 RTC 30k experiment enables execution,
uses a 100 Hz command loop and can move during Prepare; the generic YAM assembly example
has different execution settings. Do not substitute one for the other without saying so.

Hand off the chosen experiment, interpreter, station file and resolved device/service
bindings. Address inspection and requests for commands do not authorize starting services
or moving hardware; avoid an unsolicited home/reconnect workflow. Offline inspection is
not proof of device connection.
