# Policies: model, client and robot adapter are separate

Source paths below are relative to the repository root. Read
`.agents/skills/manimux-development/references/components.md` for payload examples and migration limits.

## Classify the integration first

Do not infer the integration layer from an upstream repository name or from the fact
that it contains model code. Establish which of these changes is being introduced:

1. A task, checkpoint or embodiment for an existing model: change its recipe and
   experiment; do not add another Python backend.
2. A learned model supported by an existing framework: implement the model adapter in
   that framework and reuse its ManiMux client.
3. A reusable algorithm or serving framework: keep the framework peer to XPolicyLab
   and add a `PolicyModel` client under `manimux/policies/<framework>/`.
4. Backend-independent scheduling or chunk handoff: implement it in the ManiMux runtime.

A framework owns a reusable runtime boundary, such as its dependency environment,
model registry, lifecycle, sampler interface or wire protocol. A repository containing
one model, a checkpoint loader or a thin server wrapper remains a model integration.
Do not create a peer backend for a task, checkpoint, embodiment or convenience launcher.

## A new checkpoint or learned model

A checkpoint/task change normally selects a recipe and experiment, not another
Python integration. When XPolicyLab is the model's owning framework, follow
`XPolicyLab/AGENTS.md`,
`XPolicyLab/CONTRIBUTING.md` and the reference adapter under
`XPolicyLab/policy/demo_policy/`.
Reuse an existing `XPolicyLab/policy/<POLICY>/` implementation when available.
Model loading, preprocessing, normalization, sampling and training live there.
Use the shared server and ManiMux's existing `xpolicylab_ws` client.
Select the checkpoint's input/output transforms in its existing model implementation
and reuse the matching ManiMux action adapter. A checkpoint's image keys or gripper
convention do not justify a new ManiMux adapter. Loaded
backend capabilities own sampling support; artifact inspection must not infer
RTC/PAINT support from a checkpoint name or joint/pose representation.

Do not put torch/JAX, model weights or another model server into ManiMux. A launcher
that loads deployment config is different from a second inference implementation.
Preserve checkpoint action/observation semantics and advertise RTC/PAINT only when
the sampler implements the required hooks.

## A backend framework client

Read `manimux/policies/base.py` (`PolicyModel`), `policies/__init__.py`
(`build_policy_model`) and `policies/xpolicylab/client.py`. The factory accepts the
policy dictionary and returns `reset(session_id)`, `infer(request)`,
`capabilities()` and `close()`. Discovery uses `policy.worker` and the
`manimux.policies.models` entry-point group or `module:factory`.

The client owns transport, backend wire encoding/decoding and capability/identity
exchange. It does not own cameras, robot sessions, IK, motion smoothing or chunk
handoff. Preserve reset behavior and response metadata across the wire boundary.

When inspection shows that the requested integration owns a reusable runtime and
serving contract, propose it as a peer framework and ask the user to confirm that
scope. Keep a confirmed peer framework in an independently versioned top-level
repository or submodule, alongside `XPolicyLab/`. Keep its ManiMux client under
`manimux/policies/<framework>/`, reusing the adapter and runtime interfaces.

Do not copy an entire framework into `XPolicyLab/policy/<POLICY>/` merely to satisfy
the XPolicyLab model template. Conversely, do not label a single model wrapper as a
framework just because upstream ships a server. Record the ownership decision in the
integration documentation: framework source, model adapter, serving process, transport,
ManiMux client, action adapter and runtime strategy must each have one clear owner.

Do not claim a real framework integration exists because a synthetic client passes
the interface test. Verify its native lifecycle, identity/capability exchange, reset
behavior and response decoding through the public client boundary.

The StarVLA peer implementation uses `policy.worker: starvla_ws` and
`manimux/policies/starvla/`. Its independently versioned `StarVLA/` service owns
loading, normalization and sampling; the client owns native MessagePack, reset and
identity checks. See `.agents/skills/manimux-inference/references/deployment/starvla-offline.md` for explicit joint/EEF
contracts and offline examples. AAC selection operates on grouped joint candidates
in `manimux/policies/aac.py`; each backend converts its own wire format before selection.

The grouped payload reader is `manimux/policies/actions.py`; the XPolicyLab codec
in `manimux/policies/xpolicylab/codec.py` selects grouped or native output:

- `joint`: per-group NumPy matrices `(horizon, group_dim)` in declared coordinate
  order. Format describes representation, not absolute versus delta semantics.
- `pose`: per-group rows `[x, y, z, qw, qx, qy, qz, tool...]`, translation in metres.
  The selected adapter must define the frame and absolute/delta interpretation.
- `native`: explicit specialized/unmigrated representation. Document its decoder
  pairing; do not silently guess a layout from vector width.

The XPolicyLab client selects these through `policy.options.action_format`; its
default is currently `native`. Migrated joint/pose adapters therefore need the
matching explicit experiment option. Wire field names belong in the client/codec;
do not import an XPolicyLab codec into a new backend-independent action adapter.
Retain action semantics and sampler metadata such as AAC/PAINT/AutoHorizon
fields. Translation must not silently renormalize or reinterpret the action.

## Observation and action adapter

### Hosted API policies

`policy.worker: dream_api` uses the official `dreamlabs-client==1.0.3` package
(`pip install -e '.[dream]'`) directly from `PolicyWorkerClient`. No local model
server or GPU is required. Other API providers need their own explicit
`PolicyModel` transport and observation/action contract; this is not a generic
JSON-to-robot passthrough.

Dream-1 beta supports YAM native joint output and one current RGB frame from each
of three cameras. The public baseline is
`manimux/configs/examples/dream1-yam/serial30-example.yaml` (30 Hz serial, full 48
rows, Direct). Tuned station-specific trial configurations are not distributed. Keys are read from
`DREAM_API_KEY`, never YAML. SDK episode recording is disabled. Endpoint selection
follows the SDK's `DREAM_GATEWAY_URL` / `DREAM_API_URL` settings. Inference sends
camera images, task text and robot state to the provider. Compare policies with the
same playback clock and executor; full-trajectory playback at receipt retains
observation lag and is not latency compensation.

`DreamYamJointAdapter` computes input hand poses using the pinned provider
training-reference URDF, whose axes/geometry differ from the local grasp site.
The configured robot still owns execution geometry. `per_key` joint/grip
outputs retain model joint rows; do not use the SDK-smoothed `q_*_chunk` and then
apply a second handoff. Timing is retained in `sampling_stats.dream_api`.
Finite normalized gripper outputs saturate to `[0, 1]`, matching the official
`dreamlabs==1.0.5` `robots.adapter._arm_command` conversion. Raw outputs remain
in response logs, and `gripper_clipping` reports affected rows/ranges. Non-finite
values and incorrect shapes still fail; arm joint predictions are never clipped
in this provider codec.
Reset closes the prior provider session and clears observations; failures
propagate without a caller retry loop. Only default sampling is advertised;
RTC, PAINT, AAC, BID, AutoHorizon and multi-inflight are refused.

Every API response is saved before validation under
`policy.options.response_log_dir` (default `data/diagnostics/dream-api`), grouped
by a hash of the session ID. JSON includes the untouched SDK response values,
request sequence, observation timestamp, prompt/state, camera hashes and timing.
Non-finite numbers are represented by explicit `NaN`/`Infinity` strings, and API
credentials are redacted. Transport failures also produce records. Successful
sampling statistics and decode errors reference the saved path. A write failure
fails inference rather than silently losing diagnostic evidence. These local
records are separate from the disabled provider-side episode recorder.

The recipe uses 48 rows at 30 Hz (the research page's 1.6-second horizon).
The SDK pacing default is separately 20 Hz; it is not authoritative model
timing and ManiMux does not use its pacing. Single-frame acceptance and finite
actions are interface evidence, not proof of task quality or continuous motion.
Latency trimming can exhaust the horizon, even with Smooth.
Use `scripts/validation/dream_api_probe.py` for a hardware-free worker/adapter/
executor check; it sends synthetic images and writes local evidence only.
For recorded representative inputs, use `--recorded-observation <observation.npz>`
with `image__<camera-name>` RGB arrays and `state__<group-name>` joint/gripper
vectors, plus `--task-prompt "<instruction>" --calls 3 --runtime-seconds 12
--output <report.json>`. Preserve capture provenance alongside the NPZ, including
any mismatch between image and state capture times. The actual Async runtime uses
the real API and Smooth with a disconnected synthetic robot and saved pixels;
timestamps are playback timestamps, not fresh camera feedback. The report includes
accepted/rejected plans, raw response paths and a `.commands.npz` artifact for every
would-be command. This validates the software path, not live observation freshness,
physical tracking or task success.

Read `manimux/policy_adapter/base.py` and its factory in `__init__.py`.
`policy.adapter.type: module:Class` selects the class. Construction receives
`robot`, `policy`, and injected `kinematics`; `uses_motion_limits` opt-in supplies
motion limits to implementations that require them. Reuse a compatible adapter
before adding `policy_adapter/<model-or-representation>/` code.

`prepare_request(InferenceRequest)` can add semantic model inputs such as
`model_state` / `model_info`. `decode_action(raw, ActionContext)` returns an
`ActionChunk` with robot groups, action interval, observation anchor and metadata.
`build_observation` is an optional mapping hook; it is not a place to implement
inference-algorithm chunk handoff by changing measured observations.

For a new adapter, explicitly establish:

- Input images/state and camera mapping; joint order and tool coordinates.
- Absolute or relative action meaning, the delta anchor, frame and units.
- Chunk interval and retained horizon; source offsets when dropping leading rows.
- IK seed, fixed tool coordinates, solver constraints and failure behavior.

Use the injected assembled kinematics. If action decoding runs in another process,
rebuild from the same resolved `robot["config"]` using the offline `RobotModel`;
do not instantiate a second default body or pass a hardware connection. See
`policy_adapter/kinematics.py` for current DP/OpenWAM/XR1 handling. Specialized
adapters may have different solver requirements; preserve whole-chunk rejection
versus per-step hold behavior. SAPolicy's bounded IK must not be replaced as a
side effect of a different model's integration.

### Observation state and IK seeds

The XPolicyLab transport uses its standard single-arm keys `joint_state` and
`ee_joint_state` when one group is selected. Dual-arm layouts retain side-prefixed
keys such as `left_arm_joint_state`. The robot group's name is not a wire prefix
for the standard single-arm schema. Checkpoint-specific gripper conventions and
observation-relative action decoding belong to the selected model's transforms.

`ActionContext.observation_state` is the real state captured with the request.
`ActionContext.ik_seed` is the independently selected embodiment-conversion starting
state. Adapters must use the former for observation-relative pose anchors and the
latter for IK. The old overloaded `measured_state` field is removed.

The runtime selects the seed once using `adapter.decode_seed_source`:
`observation_state` selects the request measurement; `execution_reference` prefers
the outgoing trajectory. `decode_seed_time_ns(context)` optionally specifies the
exact required time. Tianji full-chunk decoders request observation time, regardless
of blend/waypoint configuration; absent reference coverage falls back to that same
request's measurement. Other adapters keep the default execution-start/hold policy.
`ik_seed_source` records the actual selection, including fallback.
The runtime owns seed selection and timing; absolute EE decoders consume that seed
without revalidating the unused observation state or resampling the reference.

### Tianji waypoint handoff

Pi05 and XR-1 own their Tianji decoding in `policy_adapter/pi05/tianji.py` and
`policy_adapter/xr1/tianji.py`, respectively. Pi05's pass-ball and pack-plate
adapters share a private base in the Pi05 module; XR-1 does not depend on Pi05.
Both reuse the robot kinematics, differential IK solver and the same
`WaypointHandoff` planner as UMI-DP. Opt in with `policy.adapter.handoff_waypoint`
(`max_pos_speed_m_s`, `max_rot_speed_rad_s`), `ik_backend: diff` and
`execute_diff_ik_substeps: true`; select `inference.handoff: waypoint` with
`blend_policy_steps: 0`. Both ManiMux and RTC support `handoff_skip_steps`; see the
[runtime contract](runtime-config.md#rtc-waypoint-handoff-skip) for post-decode
cropping and effective-horizon accounting. When a valid waypoint window exists,
the adapter requests the aligned handoff time for its seed and decodes only the
future source suffix plus its dense lead-in. The planner consumes `ik_seed`; it
does not select or override it. Waypoint mode requires `execution_reference`.
Without a waypoint window, the full-chunk seed rule above still applies.
This does not change measured model inputs, pose anchors or first-action offsets.
Plan metadata records the actual `ik_seed_source`, `ik_seed_time_ns` and, for a
waypoint reference seed, `ik_seed_plan_id`. The timeline rejects a missed handoff or a changed
decode reference before applying any requested execution-only crop. A nonzero crop
can remove the continuous seed without changing the decoder's lead-in.

## Model layout configuration

For XPolicyLab models, declare `robot_action_dim_info` (`arm_dim`, `ee_dim`) and
`num_envs` in the policy recipe and consume the complete config in the shared
framework dimension helpers. `policy_server.config` and its inline overrides are
the deployment path; do not add a root `env_cfg/` directory or derive model dimensions
from a robot SDK. `env_cfg_type` may still identify checkpoints/profiles, but explicit
layouts do not use it for a filesystem lookup. See the
[recipe contract](../../../../manimux/configs/policy/README.md#model-layout-passed-to-xpolicylab).
Test experiment overrides, standalone recipes and unchanged pack/unpack results.
Model representations and physical joints can differ; the action adapter owns that
conversion. Existing benchmark-only framework adapters retain their legacy registry
until explicitly migrated; they are not templates for new ManiMux deployments.

## Verify the connection between layers

Use a real client decoder with a synthetic wire response, then the actual adapter.
Check groups/shapes, action semantics, metadata and reset/capability behavior.
For FK/IK changes compare default results and failure handling, test injected
geometry and independent-process reconstruction where used. Exercise the selected
recipe's client-format/adapter pairing. Offline geometry or synthetic transport
tests do not prove checkpoint inference or physical task success.


## Optional pose decoding for step IK

IK adapters declare `supports_step_ik` and implement `decode_pose_only`, selected
by EdgeRuntime after validation. Preserve each model's original action meaning,
request-relative anchors, frame/TCP conversion and gripper mapping. Use
`policy_adapter/pose_output.py` to publish `absolute_tcp_pose` chunks, with finite
`(H,8)` xyz/wxyz/gripper rows in the physical group's base frame. Pure joint
adapters keep `joint_position`. Mount-offset joint conversion is an IK adapter:
its step output is the corrected TCP target, while request-side frame mapping
remains unchanged.

Condition codecs declare `pose_condition_modes`; test the explicit
`request.action_condition_space` before interpreting condition arrays. Never
infer joint versus pose space from width alone. The runtime and restrictions
are documented in [Runtime](runtime-config.md#chunk-ik-and-execution-time-ik).

## Configurable AutoHorizon

`XPolicyLab.utils.autohorizon_selection.ConfiguredAutoHorizon` delegates the model
contract and scopes selector options to one AutoHorizon invocation. Native call
sites use `select_horizon`; Pi05 passes its pinned selector explicitly, while shared
flow samplers use XPolicyLab's pinned implementation. ManiMux passes validated
recipe options through this API. Do not monkey-patch model methods or replace
module-level selector functions. A model that does not call the configurable
selector must fail explicitly rather than claim the options were applied.
