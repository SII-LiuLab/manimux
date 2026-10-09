# Policies: model, client and robot adapter are separate

Source paths below are relative to the repository root. Read
`docs/development/components.md` for payload examples and migration limits.

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
identity checks. See `docs/deployment/starvla-offline.md` for explicit joint/EEF
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
Retain action semantics and sampler metadata such as AAC/PAINT/DVAC/AutoHorizon
fields. Translation must not silently renormalize or reinterpret the action.

## Observation and action adapter

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

### Tianji waypoint handoff and IK seeds

The shared `TianjiAbsoluteEEAdapter` (including Pi05) supports the same
`WaypointHandoff` planner as UMI-DP. Opt in with `policy.adapter.handoff_waypoint`
(`max_pos_speed_m_s`, `max_rot_speed_rad_s`), `ik_backend: diff` and
`execute_diff_ik_substeps: true`; select `inference.handoff: waypoint` with
`blend_policy_steps: 0`. Both ManiMux and RTC support `handoff_skip_steps`; see the
[runtime contract](runtime-config.md#rtc-waypoint-handoff-skip) for RTC guidance
alignment and effective-horizon accounting. The first chunk,
or a chunk without a valid outgoing reference window, keeps the request-observation
IK seed. Subsequent handoffs seed from the outgoing runtime reference at the aligned
handoff time and decode only the future source suffix plus its dense lead-in.
This does not change measured model inputs, pose anchors or first-action offsets.
Plan metadata records the actual `ik_seed_source`, `ik_seed_time_ns` and, for a
reference seed, `ik_seed_plan_id`. The timeline rejects a missed handoff or a changed
reference instead of committing a discontinuous replacement.

## Model layout configuration

For XPolicyLab models, declare `robot_action_dim_info` (`arm_dim`, `ee_dim`) and
`num_envs` in the policy recipe and consume the complete config in the shared
framework dimension helpers. `policy_server.config` and its inline overrides are
the deployment path; do not add a root `env_cfg/` directory or derive model dimensions
from a robot SDK. `env_cfg_type` may still identify checkpoints/profiles, but explicit
layouts do not use it for a filesystem lookup. See the
[recipe contract](../../manimux/configs/policy/README.md#model-layout-passed-to-xpolicylab).
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
