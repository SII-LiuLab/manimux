# Experiment configuration

To connect your own installation of a supported robot, start with
[local station setup](../../docs/usage/station.md). Put devices, service addresses and local paths in
one private `manimux/configs/local/station.yaml`. The guide explains which entry points
read it automatically and which still use separate configuration.

An experiment starts at `manimux/configs/experiments/<task>/<model>/<experiment>.yaml`.
`manimux.cli.read_experiment()` resolves references and selected station bindings;
`load_config()` supplies runtime defaults. These readers do not connect hardware or services.

```text
manimux/configs/
├── experiments/<task>/<model>/  # Robot, policy, adapter, inference and execution choices
├── examples/                   # Annotated complete configurations and launch walkthrough
├── embodiment/                 # arm / end_effector / sensor / robot assembly
├── policy/<model>/<embodiment>/  # Deployment recipes consumed by XPolicyLab
├── inference/                  # Reusable inference scheduling parameters
├── executor/                   # Reusable smoothing and motion limits
├── local/                      # Templates and the private local station file
└── robogui/                     # Display and camera-preview layouts
```

Camera combinations live in `embodiment/sensor/cameras/`; IK settings belong to their
embodiment. Policy recipes select a checkpoint and its inference settings; XPolicyLab
owns model loading, preprocessing, normalization and the shared policy server. See
[policy recipes](policy/README.md) for this boundary. Complete runtime choices belong
in experiments, so policy recipes have no separate `infra/` or `server/` directory.

Private training configurations, launchers and notes live in the repository root's
ignored `training/` directory, outside the package. Deployment still retains checkpoint
metadata and normalization references needed for inference. Private cluster-job payloads
may also live under `.local/`. Old directory paths have no forwarding aliases.

[Python environments](../../docs/usage/environments.md) describes local dependency environments.
Model-side action dimensions and batch size belong in [policy recipes](policy/README.md#model-layout-passed-to-xpolicylab),
resolved through the experiment's `policy_server` section. No root `env_cfg/` registry
is required by the shipped ManiMux deployment recipes.

Start with the [annotated Pi05 RTC example](examples/README.md) for component references,
`run` versus `serve`, and the RoboGUI experiment workflow.

## Action spacing and command frequency

Each experiment explicitly declares `policy.action_dt_s`, `policy.horizon_policy_steps`,
`robot.control_hz`, `inference.algorithm` and `executor.type`.

- `policy.action_dt_s: 0.03333333333333333`: predicted action points are 1/30 second apart.
- `robot.control_hz: 30.0`: the runtime targets one command tick every 1/30 second.
- Setting `robot.control_hz: 100.0` samples the same timeline at 10 ms intervals using
  interpolation. It does not change the model's action spacing or request 100 inferences
  per second. This is the current Pi05 RTC 30k recipe's choice, not a requirement for other experiments.
- `executor.smooth.cutoff_hz` is a filter cutoff, not an interpolation or command rate.

Shared YAML is packaged with the code. References resolve relative to the referring YAML.
Run output paths remain relative to the process's working directory unless the station
supplies an output override; records are not implicitly written into the package.
Private station files are ignored by Git and excluded from packages.

## Choose each pipeline stage in the experiment

```yaml
robot:
  type: yam
  config: ../../../embodiment/robot/yam_dual.yaml
  control_hz: 30.0
policy:
  worker: xpolicylab_ws
  adapter:
    type: manimux.policy_adapter.joint:JointAdapter
    camera_map:
      cam_head: d405_front
      cam_left_wrist: d405_left
      cam_right_wrist: d405_right
  action_dt_s: 0.03333333333333333
  horizon_policy_steps: 50
policy_server:
  config: ../../../policy/pi05/yam/put-bottles/joint-step30000.yaml
inference:
  algorithm: rtc
  config: ../../../inference/aligned/yam_rtc.yaml
executor:
  type: smooth
  config: ../../../executor/yam_smooth.yaml
```

This illustrates the fields; see the complete
[Pi05 RTC experiment](experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml).
The policy service address comes from the station. `policy.adapter` selects a Python
implementation and its mappings directly, without a separate YAML for every adapter.
See [policy adapters](../policy_adapter/README.md).

One model can use different adapters when training changes its I/O contract; one adapter
can serve different models with the same contract. XPolicyLab owns model normalization
and internal encoding. ManiMux owns robot groups, observation mapping and necessary FK/IK.
The adapter base class does not guess a default joint-action format.

Each of `policy`, `policy_server`, `camera_server`, `inference` and `executor` can reference one base file.
Experiment values override that file; lists are replaced as a whole. References are expanded
one level. `robot.config` refers to the assembled robot configuration.

For example, `camera_server.config: ../../../embodiment/sensor/cameras/realsense_3_views.yaml`
selects named camera components for a YAM experiment. Device serials still come from the
station. The same recipe also works with camera-server `--config`; see
[the camera options and naming guide](embodiment/sensor/cameras/README.md).

Physical runtime startup selects `--local`, then an explicit experiment `local:` reference,
then the default station file. A CLI path is relative to the working directory; an experiment
reference is relative to the experiment; values under station `paths` are relative to the
station file. Pure config readers remain usable without implicitly selecting a station.

Local bindings do not choose the algorithm, action semantics or execution switches. Pi05's
`paths.checkpoints` provides the local checkpoint root; the experiment's selected server
recipe still chooses its checkpoint and normalization subpaths. UMI_DP retains its explicit
`paths.checkpoint` artifact binding. Sessions record the resolved configuration. Backend
identity and sampling-capability checks remain in effect.

Shared control profiles own robot layout and arm/gripper motion limits. Model action spacing
is declared in the experiment, while command frequency is `robot.control_hz`. Collection and
deployment choose their executors and filters separately. The YAM profile is
`embodiment/robot/yam_control.yaml`.

Model services run in separate environments. The former native and external-HTTP
ABC/MolmoAct compatibility paths have been removed; learned-policy integrations use
XPolicyLab and ManiMux's shared `xpolicylab_ws` worker.

The Xiaomi Robotics 1 pass-ball checkpoint on Tianji-TacCap uses
`experiments/pass_ball/xiaomi-xr1/tianji_taccap_xiaomi_xr1_step50000.yaml` and the policy recipe at
`policy/xiaomi-xr1/tianji/pass_ball/step50000.yaml`. Its Cartesian action adapter is selected
by the experiment and performs inline FK/IK using the assembled Tianji robot kinematics.
See the [XR-1 Tianji-TacCap runbook](../../docs/deployment/xiaomi-xr1-tianji-taccap.md).

## Arm motion limiting

The shared profile's `motion_limits.arm.mode` selects Direct/Smooth command limiting:

| Mode | Behavior |
| --- | --- |
| `per_joint` (default) | Clip each joint independently. |
| `isotropic` | Scale each arm's joint-velocity vector by one factor, excluding configured gripper dimensions. |

For example, the shared Tianji profile includes:

```yaml
motion_limits:
  arm:
    mode: isotropic
    max_velocity: 0.9047786842338605  # rad/s = 51.84 deg/s
    max_step_dt_s: 0.016
    max_acceleration: null
  gripper:
    group_indices: {left_arm: 7, right_arm: 7}
    max_velocity: 3.0
    max_acceleration: 12.0
    max_closing_velocity: 1.0
```

`max_step_dt_s` defaults to null. When set, the single-step velocity budget uses
`max_velocity × min(1/control_hz, max_step_dt_s)`. It does not change the loop frequency
or replace command-timeout handling.

With isotropic acceleration limiting, the velocity vector is scaled first, then its change
relative to the preceding tick is scaled. The second operation preserves the direction of
the velocity change, so the final position increment need not point toward the original
target. Set `max_acceleration: null` for teleop-style proportional step limiting without
that acceleration stage. Smooth's filtering, braking and position bounds remain active.

`command_safety` independently rejects commands outside its configured bounds. Position
bounds and velocity limits are configured together; acceleration can be omitted or null.
Omitting it disables the software acceleration check without changing the controller's
acceleration setting; position, velocity and finite-value checks still apply.

Without a shared profile, Direct takes the mode from `executor.motion_limits.arm`, while
Smooth takes `mode` / `max_step_dt_s` under `executor.smooth`. Local and profile values must
not conflict. See [Tianji motion-limit provenance](../../docs/advanced/tianji-control.md).

## Common experiment fields

### Run and robot

| Field | Meaning |
| --- | --- |
| `run.task` | Task instruction sent to the policy and recorded for the RoboGUI/Recorder. |
| `run.output_dir` | Parent directory for `session-<timestamp>-<id>/` and rollout records. |
| `run.max_control_steps` | Maximum control ticks; for example, 120 ticks at 100 Hz are about 1.2 s. |
| `run.experiment_mode` | Default RoboGUI experiment mode; normally false and selectable before Prepare. |
| `run.layout_id` | Optional initial-layout or experimental-condition identifier. |
| `robot.type` | Assembly implementation, such as `yam` or `tianji_taccap`. |
| `robot.config` | Robot assembly YAML. |
| `robot.control_hz` | Runtime command-loop target frequency, independent of inference frequency. |
| `robot.group_dims` | Canonical group names/dimensions, shared by adapter, executor and robot. |
| `robot.options` | Body-specific options; station hardware bindings are merged here. |

### Sensors and policy

`sensors` is a list of named data sources. Network sources consume frames from the camera
service; the runtime does not set their physical acquisition rate.

| Field | Meaning |
| --- | --- |
| `sensors[].name` | Observation key used to identify the sensor. |
| `sensors[].driver` | Source name or `module:factory`; camera factories live under `embodiments.sensor`. |
| `sensors[].service` | Named station service whose endpoint is applied to this sensor. |
| `sensors[].options` | Source-specific endpoint, stream names and timeouts. |
| `sensors[].width` / `height` / `fps` | Declared dimensions/rate; a network source uses the service's actual frames. |
| `policy.worker` | PolicyModel plugin; model work runs in a separate spawned worker. |
| `policy.adapter` | Python `type` plus observation/action mapping parameters. |
| `policy.device` | Device hint for the selected model plugin. |
| `policy.action_dt_s` | Time between adjacent model action points. |
| `policy.horizon_policy_steps` | Points per action chunk; H points span `(H-1) × action_dt_s` between first and last. |
| `policy.timeout_s` | Inference request deadline; late responses are discarded. |
| `policy.startup_timeout_s` | Worker initialization timeout, default 30 s. |
| `policy.trajectory_duration_s` | If set, overrides spacing with `duration / (horizon_policy_steps-1)`. |
| `policy.inference_delay_s` | Simulated latency for the fake model used in tests. |
| `policy.options` | Client connection settings; observation/action mappings belong in the adapter. |
| `policy.expected_backend` | Expected server/model metadata, matched before the robot connects. |

Do not pin a per-process `server_instance_id` in `expected_backend`. Declare stable fields
such as policy family, task, checkpoint variant/source, normalization source and resolved
artifact paths. Every declared field must be present and equal in the server handshake;
additional server fields are allowed.

### Inference scheduling

| Field | Meaning |
| --- | --- |
| `inference.algorithm` | Strategy, including `manimux`, `act_temporal_ensemble`, `rtc`, `aac`, `paint`, `autohorizon`, an entry point or `module:factory`. |
| `inference.refill_threshold_s` | Default strategy: request another chunk when the timeline has less remaining duration. |
| `inference.inference_schedule` | Request admission: `deadline`, `single_inflight`, `multi_inflight` or `serial`; valid combinations depend on the algorithm. |
| `inference.request_trigger` | Ordinary async: `refill` (default) or `continuous`; specialized/serial: `algorithm`. Independent of in-flight capacity. |
| `inference.observation_hz` | Optional request frequency cap for single or multi-inflight; explicitly required for multi-inflight. Does not change camera or control frequency. |
| `inference.handoff_skip_steps` | Additional leading source rows skipped at a chunk handoff without delaying its start; defaults to `0` and does not affect the first chunk. Skipped rows shorten the remaining horizon and may increase the handoff jump. A waypoint handoff skips them before decoding and requires `algorithm: manimux`. |
| `inference.max_plan_age_s` | Maximum age measured from the chunk's observation time. |
| `inference.blend_policy_steps` | Number of leading accepted policy points blended from the measured command; zero disables it. |
| `inference.handoff` | `blend` (default) joins chunks at commit; `waypoint` lets an adapter that declares `handoff_waypoint` join them in EE space before dense IK. Waypoint requires process decoding, `manimux` or `rtc`, `blend_policy_steps: 0` and `action_start_mode: drop_infer_latency`. |
| `inference.handoff_margin_s` | Extra time added to the expected decode finish before a waypoint handoff is planned. |

Strategies have different scheduling contracts. RTC uses its horizon/execution/delay
contract; ACT temporal ensembling uses query intervals; AAC waits for its selected short
chunk; PAINT uses an asynchronous prefix. Unsupported scheduler/algorithm combinations are rejected
where they do not apply. Strategies share the robot, timeline and executor infrastructure.
An accepted, decoded chunk takes effect at commit time without an additional switch delay.

| Algorithm section | Main fields |
| --- | --- |
| `rtc` | Initial delay, delay history, guidance and chunk execution threshold; see [RTC](../../docs/deployment/xpolicylab.md#rtc-规则). |
| `temporal_ensemble` | `coefficient: 0.01`; `query_interval_policy_steps: 1`. |
| `aac` | `num_samples: 20`, `motion_threshold`, required `ee_stats_path`, `chunk_id_selector`, `backward_beta: 0.99`. |
| `paint` | `execution_policy_steps: 10`, `initial_delay_policy_steps: 4`, `delay_buffer_size: 10`. |

ACT uses official exponential weights `w_i ∝ exp(-coefficient × i)` from commit `742c753`.
Its queries are asynchronous; `blend_policy_steps: 0` prevents an extra seam blend after aggregation.
See [ACT temporal ensembling](../../docs/advanced/inference.md).

AAC requires short-horizon support and `blend_policy_steps: 0`. The YAM recipes adapt its scoring
to 14D absolute joints: shared FK produces per-arm EE increments, matched fixed statistics
normalize those increments, and the arm scores are averaged. The selected joint chunk
remains the executed representation. These are YAM adaptations, not official Pi05/YAM
recipes. See [AAC](../../docs/advanced/reproductions/aac.md) and [Pi05 AAC](../../docs/advanced/reproductions/aac-pi05.md).

PAINT requires `d <= s <= H-d` and `blend_policy_steps: 0`. ManiMux submits the old chunk's
`A[s:s+d]` prefix; the model sampler implements the repaint sequence. Responses are rejected
when delay would discard more than the anchored prefix. See [PAINT](../../docs/advanced/reproductions/paint-pi05.md).

AutoHorizon has no configurable method parameters: the Pi05 sampler selects an execution
prefix from the action expert's third denoising-step self-attention. It requires
`blend_policy_steps: 0` and synchronous prefix execution. The JAX port uses upstream commit
`c7504f1`; numerical parity with the upstream PyTorch implementation remains a separate
validation boundary. See [AutoHorizon](../../docs/advanced/reproductions/autohorizon-pi05.md).

### Executors

| Field | Meaning |
| --- | --- |
| `executor.type` | `direct`, `smooth` or `mpc`, independent of inference scheduling. |
| `executor.smooth.cutoff_hz` | First-order low-pass cutoff; lower values trade responsiveness for smoothing. |
| `executor.smooth.max_velocity` | Per-scalar velocity limit, typically rad/s for joint positions. |
| `executor.smooth.max_acceleration` | Per-scalar acceleration limit, typically rad/s² for joints. |
| `executor.smooth.position_limit_abs` | Generic symmetric position envelope, not the robot's precise per-joint limits. |
| `executor.mpc.horizon_control_steps` | Optimization horizon in control ticks. |
| `executor.mpc.dynamics_a` | Simplified state retention coefficient in `(0,1)`. |
| `executor.mpc.tracking_weight` | Cost of deviation from the policy reference. |
| `executor.mpc.command_delta_weight` | Cost of command changes between ticks. |
| `executor.mpc.max_velocity` / `max_acceleration` / `position_limit_abs` | Limits applied after optimization. |

<a id="robogui-and-recording"></a>

### RoboGUI and recording

| Field | Meaning |
| --- | --- |
| `robogui.enabled` | Publish state, cameras, plans and events for the RoboGUI. |
| `robogui.robot` | Robot geometry/joint mapping used for display. |
| `robogui.policy_label` | Label displayed in the RoboGUI and stored by the Recorder. |
| `robogui.camera_hz` | Maximum camera publication rate to the RoboGUI, default 5 Hz. |
| `recording.enabled` | Real runs require recording of episodes, events and command lineage. |
| `recording.video_fps` | Video encoding target rate; zero disables video without changing policy/camera rates. |
| `recording.video_codec` | OpenCV four-character codec, default `mp4v`. |
| `recording.video_queue_size` | Asynchronous queue capacity; a full queue drops video bundles instead of blocking control. |

## Research metadata

`run.experiment_name`, `run.condition` and `run.notes` are editable in RoboGUI and
frozen at Prepare. `run.experiment_template` optionally constrains study layouts,
repeat counts and reference images. See the [workflow and schema example](../../docs/usage/research.md).
Keep outputs under the repository-level `data/`; `run.output_dir` or the private
station's `paths.output_dir` chooses the actual destination.
