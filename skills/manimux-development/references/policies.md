# Policies: model, client and robot adapter are separate

Source paths below are relative to the repository root. Read
`skills/manimux-development/references/components.md` for payload examples and migration limits.

## Mandatory shared contract for every model and backend

These requirements apply to every model, checkpoint, embodiment, local framework and
hosted API. Backend-specific examples below describe implementation details; they do
not exempt an integration from this contract. An older implementation is evidence to
inspect, not permission to reproduce its protocol gaps.

The shared path is:

```text
robot observation -> PolicyAdapter.prepare_request -> PolicyModel client/wire codec
backend response -> client/wire codec -> grouped joint/pose payload
                 -> PolicyAdapter.decode_action -> ActionChunk
                 -> shared scheduling/executor -> robot command boundary
```

`PolicyModel` supplies reset, inference, capabilities and close. The client translates
the backend's transport and field names. The adapter maps observations and actions to
the configured embodiment, using its geometry. Runtime owns timing, admission and
execution. Different HTTP/WebSocket protocols do not justify different runtime action
contracts. Reuse an existing compatible adapter before creating another one.

For every integration, establish and validate both observation and action contracts:

- Images: physical sensor mapping, image keys, ordering, color space and preprocessing.
- State/actions: group and coordinate order, representation, units, absolute/delta
  meaning and delta anchor; pose reference frame, controlled point (flange or TCP),
  quaternion order, and tool/gripper meaning and range.
- Timing: observation identity/time, action interval, horizon, retained source offsets,
  request/reset association and metadata used by scheduling or sampling.
- Capabilities: supported sampling/conditioning modes, condition coordinate space,
  reset/failure behavior and explicit refusal of unsupported combinations.

Do not infer semantics from vector width, a model name or an example's defaults.
Convert wire field names, array layout, units and quaternion order in the codec using
documented semantics. Use the adapter and configured geometry for physical frame,
control-point and embodiment conversion. Preserve the meaning through every boundary;
do not relabel flange coordinates as TCP coordinates to satisfy a shape check.

The current grouped reader is `manimux/policies/actions.py`; it validates formats,
groups, finite matrices and horizons. `PolicyModel.infer()` currently returns `object`,
and not every semantic item above has a typed, centrally validated field. These are
implementation gaps, not exemptions. Inspect the actual recipe, metadata and adapter
contract; add a shared declaration/validation where needed instead of inventing an
unsupported YAML field or claiming the method signature guarantees compatibility.

### Native payloads require a justified, user-approved exception

Here `native` means preserving backend-specific action data for a designated adapter
to interpret, instead of converting it into the common grouped `joint` or `pose`
representation at the client boundary. It does not mean permission to send API output
directly to motors. It couples that client to that decoder and makes cross-backend
validation and adapter reuse harder. Backend-native HTTP/MessagePack transport and
model-internal tensors are different concepts; they can remain behind a common client.

New or changed integrations MUST use the shared representation when it can express
the required semantics. Different JSON keys, flat arrays, quaternion order, units,
single/dual-arm layouts, gripper conventions, additional mapping work or the existence
of a legacy native adapter are not reasons to add another native action path.

Before proposing an exception:

1. Inspect the shared action reader, compatible adapters, model schema and resolved
   embodiment. Attempt a concrete mapping and identify the exact semantic requirement
   the shared contract cannot express; show an example input/output and its meaning.
2. Determine whether a reusable extension of the shared contract solves the gap.
   Prefer that extension when appropriate. Missing conversion code or missing exposed
   geometry is work to implement in its owning layer, not evidence of impossibility.
3. Only if conformance is genuinely infeasible, present the specific limitation,
   alternatives considered, affected capabilities and smallest proposed exception.
   Ask for the user's explicit confirmation before implementing or selecting a new
   native fallback. Silence, general integration permission and prior legacy usage
   do not approve this exception. Continue independent conforming work while pending.
4. If approved, document its scope, schema, frames/units/timing, paired decoder,
   validation, failure behavior, tests and the user's decision in the integration
   record. Keep ordinary decoding, command validation and runtime ownership intact.

Never silently fall back to native after canonical decoding fails. Preserve working
legacy deployments outside the requested scope, but do not present them as conforming
new templates. A shared-contract migration should preserve model/control semantics
and be verified before switching an existing deployment.

### Resolve geometry and station facts before asking the user

Read the resolved experiment and recipe, selected robot assembly/groups, component and
tool models, private station bindings/calibration and model input/output schema. Trace
the real loaders and record the source of each value. Use an offline `RobotModel` and
the same injected kinematics used by runtime; do not connect hardware for discovery.

Ownership is independent of which model happens to need the information:

| Information | Owning source |
| --- | --- |
| Joint order/units, link geometry, flange-to-TCP/tool geometry | Embodiment/component/tool model |
| Component mounts and physical control groups | Robot assembly |
| Device addresses, physical camera/arm mapping, measured station extrinsics | Reusable private station bindings/calibration |
| Training/model frame, controlled point, normalization, output convention | Model contract and model transforms |
| Mapping selected model inputs/actions to the installed assembly | Adapter composed from those sources |

Automatically reuse and validate available data. Do not ask the user to re-enter known
geometry, select an arm already selected by the recipe, or remeasure an existing
calibration just because a new model is being integrated. If geometry exists but the
public model API does not expose it, implement the reusable access in that layer.
Do not copy intrinsic tool geometry into every policy's calibration fields.

Only genuinely missing physical facts or conflicting/unresolved user choices require
a question. Route missing measurements to station setup or new-embodiment setup;
identify the missing quantity, inspected sources, required measurement and why it is
needed. Continue client/conversion/offline work independently. Never invent identity
transforms, infer left/right from enumeration, or treat GUI display placement as
calibration. Clearly mark synthetic transforms as test fixtures.

After writing the integration, show the selected groups/cameras, observation/action
meaning, transform chain, source files and resolved values, validation results and
remaining unknowns for user review. Do the discovery, mapping and writing first;
ask the user to check the concrete result, not design the adapter or fill an empty
template. That review does not authorize robot startup.

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
`third_party/XPolicyLab/AGENTS.md`,
`third_party/XPolicyLab/CONTRIBUTING.md` and the reference adapter under
`third_party/XPolicyLab/policy/demo_policy/`.
Reuse an existing `third_party/XPolicyLab/policy/<POLICY>/` implementation when available.
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
scope. Keep a confirmed peer framework in an independently versioned
repository or submodule under `third_party/`, alongside `third_party/XPolicyLab/`. Keep its ManiMux client under
`manimux/policies/<framework>/`, reusing the adapter and runtime interfaces.

Do not copy an entire framework into `third_party/XPolicyLab/policy/<POLICY>/` merely to satisfy
the XPolicyLab model template. Conversely, do not label a single model wrapper as a
framework just because upstream ships a server. Record the ownership decision in the
integration documentation: framework source, model adapter, serving process, transport,
ManiMux client, action adapter and runtime strategy must each have one clear owner.

Do not claim a real framework integration exists because a synthetic client passes
the interface test. Verify its native lifecycle, identity/capability exchange, reset
behavior and response decoding through the public client boundary.

The StarVLA peer implementation uses `policy.worker: starvla_ws` and
`manimux/policies/starvla/`. Its independently versioned `third_party/StarVLA/` service owns
loading, normalization and sampling; the client owns native MessagePack, reset and
identity checks. See `skills/manimux-inference/references/deployment/starvla-offline.md` for explicit joint/EEF
contracts and offline examples. AAC selection operates on grouped joint candidates
in `manimux/policies/aac.py`; each backend converts its own wire format before selection.

The grouped payload reader is `manimux/policies/actions.py`; the XPolicyLab codec
in `manimux/policies/xpolicylab/codec.py` selects grouped or native output:

- `joint`: per-group NumPy matrices `(horizon, group_dim)` in declared coordinate
  order. Format describes representation, not absolute versus delta semantics.
- `pose`: per-group rows `[x, y, z, qw, qx, qy, qz, tool...]`, translation in metres.
  The selected adapter must define the frame and absolute/delta interpretation.
- `native`: existing specialized/unmigrated representation, subject to the
  [exception gate](#native-payloads-require-a-justified-user-approved-exception).
  It is not an equal default choice for new integrations.

The XPolicyLab client selects these through `policy.options.action_format`; its
default is currently `native`; this is a legacy compatibility default, not guidance
for new integrations. Migrated joint/pose adapters therefore need the
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
`manimux/configs/experiments/example-yam-dream1/_base_.yaml` (30 Hz serial, full 48
rows, Direct), with `put_bottles/serial.yaml` and `put_bottles/rpent.yaml` variants. Tuned station-specific trial configurations are not distributed. Keys are read from
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

Experiment response recording is owned by the shared worker's `ResponseLog`, enabled
by `policy.options.response_log_dir` (the recipe selects `data/diagnostics/dream-api`).
The client contributes the SDK response before decoding; it does not open files.
Request-scoped `request.json`, `response-*.json` and `outcome.json` preserve inputs,
raw/decoded outputs, timing and failures. Non-finite values are encoded explicitly
and credentials are redacted. Worker/runtime rejection events link the evidence
folder. Standalone client calls without an active recorder perform no archive I/O.
This experiment record is separate from the disabled provider-side episode recorder.

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

### Poke-Aurora API

Implementation status: the initial Poke codec returns a native action payload and
its adapter takes explicit transform fields. This does not yet satisfy the shared
contract above and is not a template for another API. Review/migrate the payload to
the common pose contract and resolve geometry from its owning sources; existing code
does not constitute user approval of a native exception. The details below describe
the current software path, not a completed protocol migration or calibrated station.

`policy.worker: poke_api` uses Poke's HTTPS OpenAI-compatible endpoint through
Python's standard library. It is separate from `dream_api`: Dream-1 beta returns
dual YAM joints, while Poke-Aurora uses one wrist RGB image and absolute flange
targets `[x,y,z,qx,qy,qz,qw,gripper]` in a calibrated model frame. Source contract:
[cloud API](https://developers.pokebot.com/zh/docs/roboapi/latest/cloud-api),
[coordinates](https://developers.pokebot.com/zh/docs/roboapi/latest/inputs), and
[calibration](https://developers.pokebot.com/zh/docs/roboapi/latest/calibration).

The client checks `/models` and the selected model's `/schema` on reset, sends
one stateless `/chat/completions` request per observation, and decodes exactly
one `execute_actions` tool call. The initial recipe selects `poke-aurora-preview`,
40 rows at 30 Hz. Delivered row count is validated against schema and optional
`metadata.action_shape`; request IDs come from `X-Request-ID`. PNG resizing
preserves aspect ratio, without cropping, rotation or mirroring. Credentials
come from `PLATFORM_API_KEY`; the base URL is explicit YAML. There are no retries,
redirects, SDK episode recordings, or fabricated cloud execution receipts.
Cloud `/sessions` and execution-feedback history are not implemented: the
PolicyModel contract does not receive actual dispatched-step receipts. Local
reset fences request identities only. Default sampling and single-inflight are
the only advertised capabilities.

`PokePoseAdapter` uses the configured robot's FK/IK, selected `controlled_group`
and `camera`. Other robot groups hold the runtime-selected reference. Three
explicit xyz/wxyz poses describe `model_from_base`, `flange_from_tcp`, and the
rotation-only `flange_from_model_tool` orientation alignment:

```
model_T_tool = model_T_base @ base_T_tcp @ inv(flange_T_tcp) @ flange_T_model_tool
```

The inverse maps provider targets back to configured TCP targets. This includes
the orientation alignment used by RoboAPI 0.4.5's Home/plane calibration; a
change of quaternion order alone is insufficient. Missing transforms fail at
adapter construction. Do not replace missing station calibration with identities
or claim a synthetic transform is physically calibrated. IK rejects the entire
chunk on failure; the configured executor still owns smoothing and limits.

The client supplies the raw HTTP response before JSON/action decoding to the shared
experiment recorder (`policies/response_log.py`). The recipe selects
`data/diagnostics/poke-api`; RPent selects a session-owned directory. Request, response
and failure records share the same format as other backends. Direct client calls do
not write files; validation scripts can activate the recorder explicitly. Raw actions
remain intact in experiment evidence. The documented gripper tolerance is 0.01: small
overshoot is clipped with affected rows recorded, larger overshoot fails.
Non-finite values and non-unit quaternions fail with row diagnostics.

The template is `manimux/configs/examples/poke-yam/serial-example.yaml`, with
unfilled station transforms and hardware execution disabled. It is not a
qualified real-station rollout. `scripts/validation/poke_api_probe.py` uses
recorded observations and the process worker, without creating a device driver.
With configured transforms it checks FK, inference, IK, timeline admission and
Smooth would-be commands. `--model-state <json>` explicitly selects an
interface-only test and skips physical coordinate conversion; report this limit.
Inputs must carry capture provenance, especially if image/state are not simultaneous.

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
[recipe contract](../../../manimux/configs/policy/README.md#model-layout-passed-to-xpolicylab).
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


## Shared IK failure behavior

All model and backend adapters use the [shared execution IK policy](runtime-config.md#chunk-ik-and-execution-time-ik). Configure `execution.ik.failure_policy`, not an adapter-specific default. `hold_last_valid` holds both arm and gripper at a failed target and attempts later targets; `reject` fails the decode. Use the common `IKSequence` helper for chunk decoding and preserve the resolved policy when constructing child decoders. Raw solver failures must never produce independently advancing gripper commands.

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

## ChunkTrust denoising evidence

The built-in `chunktrust_ahs` and `chunktrust_ahs_rtc` selectors consume model evidence
locally; their 19 algorithm options are not server sampling parameters. Both XPolicyLab
client paths preserve an `INFER_RESULT` mapping containing `denoise_trace`, including
the original actions. The adapter transforms actions only. Runtime trace validation
checks live HELLO identity, raw normalized coordinates, dtype, T/H/D and the explicit
valid-action mapping before scoring. No normalized velocity is interpreted as a robot
command. Raw wire responses are persisted before decoding or evidence rejection.

`return_denoise_trace` is the pinned XPolicyLab deployment flag, not a wire
sampling override. Keep strict RTC validators intact: send only mode, physical action
condition, row weights and beta. The canonical Pi05 pin includes opt-in default trace;
the AHS experiment enables it through the existing managed launcher. It does not support
RTC+trace or the required
zero-guidance mask for padded model dimensions. A dedicated strategy capability hook
checks the actual loaded model before hardware connection and preview; expected
metadata cannot manufacture support. See the [ChunkTrust contract and deployment
workflow](../../manimux-inference/references/chunktrust.md) for capability fields,
source restrictions, presets and disconnected validation. Keep the submodule gitlink,
declared backend provenance and documentation aligned. Trace stays off by default for
other recipes. An external model service remains optional. CPU integration evidence
does not qualify native JAX, checkpoints or physical deployment.
