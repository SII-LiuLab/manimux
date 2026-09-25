# Policies: model, client and robot adapter are separate

Paths are relative to the repository root. Read
`docs/component-policy-development.md` for payload examples and migration limits.

## A new checkpoint or learned model

A checkpoint/task change normally selects a recipe and experiment, not another
Python integration. For a new learned model, follow `XPolicyLab/AGENTS.md`,
`XPolicyLab/CONTRIBUTING.md` and
`XPolicyLab/.agents/skills/xpolicylab-model-integration/SKILL.md`. For a requested
model-adapter review, use `XPolicyLab/.agents/skills/xpolicylab-adapter-check/SKILL.md`.
Reuse an existing
`XPolicyLab/policy/<POLICY>/` implementation when available. Model loading,
preprocessing, normalization, sampling and training live there. Use the shared
server and ManiMux's existing `xpolicylab_ws` client.

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

RLinf/StarVLA as peer serving frameworks are a different scope from adding one
model. Current repository policy requires learned-model integration in XPolicyLab;
the technical plugin mechanism is not permission to bypass that rule. If the user
explicitly requests a peer framework, make that architectural scope clear and
keep its client under `policies/<framework>/`, reusing the adapter/runtime interfaces.
Do not claim a real framework integration exists because a synthetic client passes
the interface test.

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

## Verify the connection between layers

Use a real client decoder with a synthetic wire response, then the actual adapter.
Check groups/shapes, action semantics, metadata and reset/capability behavior.
For FK/IK changes compare default results and failure handling, test injected
geometry and independent-process reconstruction where used. Exercise the selected
recipe's client-format/adapter pairing. Offline geometry or synthetic transport
tests do not prove checkpoint inference or physical task success.
