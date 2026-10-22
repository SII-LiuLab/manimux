# RPent / HarnessVLA integration

RPent owns planning, memory and tool selection. ManiMux owns the selected policy
client, action adapter, motion planning, robot, execution, RoboGUI and recording.
`third_party/RPent/` remains the pinned peer submodule at `efc392adc98a30045d8fb83c4097917f28e20a94`.
No model implementation or hardware controller is copied into the agent package.

## One entry, one toolkit, one runtime

### RPent experiments in RoboGUI

An experiment becomes an RPent experiment by declaring `agent: {config: /path/to/agent.yaml}`.
`config` resolves beside the declaring YAML, including inherited bases. The planner
interpreter `python` is optional: it defaults to `third_party/RPent/.venv/bin/python` when
that environment exists, otherwise the runtime interpreter. Set it only for another env.

Such an experiment runs only RPent; there is no Direct VLA choice. Running the VLA on its
own is what an ordinary recipe without `agent` does (XPolicyLab, Dream-1 and the other
policy integrations). `run.execution_mode` is not a YAML setting; recorded configs keep the
mode of each rollout. The planner calls the experiment's own model, its `policy` section
(and `policy_server` for a local model service), through `execute_policy` with a subtask
prompt and `use_length`. To change the VLA, `_base_` another runtime recipe or override
`policy`; there is no separate RPent model field.

The shared `manimux run` / `--config-list` service shows the RPent agent view (bottom
message bar). Prepare validates the planner environment; Start launches the native
planner with the current task. The profile's experiment/task/output/endpoint do not replace
the experiment's: output and loopback endpoint are unique per rollout. The profile supplies
planner/model/reasoning, memory and the calibrated motion file.

The experiment must satisfy the Serial bridge contract checked by
`PolicyDispatch.validate_config`: `algorithm`/`inference_schedule: serial`,
`action_start_mode: first_step_when_ready`, no blending, handoff skip, strategy or
temporal ensemble, inline decoding and chunk IK. `chunk_policy_steps` implies
`max_chunk_policy_steps`, so a base recipe with a chunk cap needs both set to `null`
(the planner chooses rows per call). Blending stays off: one request at a time, and the
robot holds between chunks while the next inference runs.

Policy chunks still use the main loop's timeline and executor. Rows are interpolated at
`robot.control_hz`, so a 10 Hz policy (`action_dt_s: 0.1`) at 100 Hz control works, and a
Smooth executor filters both VLA chunks and motion primitives. Smooth limits add tracking
lag, so `settle_timeout_s` in the motion profile must cover it (the default primitive
speeds, 0.03 rad/s and 4 mm/s, are far below the YAM Smooth limits). Example on the hosted
Dream-1 YAM baseline (Serial + Smooth, 10 Hz rows at 100 Hz control):

```yaml
_base_: ../examples/dream1-yam/serial-slow10.yaml
agent: {config: rpent_dream1_agent.yaml}   # private recipe: motion, planner, model
inference: {chunk_policy_steps: null, max_chunk_policy_steps: null}
```

Hosted workers (`dream_api`) need no model service: launch with `manimux up` without
`--policy-python`, after exporting the key named by `policy.options.api_key_env`.
The Dream gateway evicts sessions that sit idle, which happens while the planner thinks
between calls. The Dream client therefore starts a new episode (`reset_episode()`) whenever
the instruction changes, and on `DreamSessionExpired` asks once more on the replacement
session the gateway already adopted. That retry costs one extra round trip (~3 s), so RPent
Dream configs should allow `policy.timeout_s` and `inference.max_plan_age_s` of about 10 s;
the arms hold while a call waits.

`agents/rpent/session.py` wraps the ordinary EdgeRuntime, validates the independent
planner environment before device startup, and owns only the planner process group
and dispatch server. It does not implement a second hardware loop or cleanup path.
Pause holds/cancels the active bounded call; managed tools wait for operator Resume,
then return fresh observations without resubmitting a cancelled motion. Finish or
runtime faults fence the mailbox and terminate the owned planner and its children.
A valid planner finish ends the rollout through normal cleanup and offers the same
human evaluation controls; it never assigns a human score. Planner failure surfaces
as a session error. Every Prepare creates new planner artifacts and a new dispatch
session, preventing an old planner from reaching a subsequent rollout.

RPent trials append ` / RPent` to the configured study condition (or use `RPent`
when empty), keeping their repeat counts and human evaluations apart from VLA-only
recipes in the same study. The episode records execution mode and agent artifact
directory; that directory retains effective configuration, motion profile, native
recipe, provider evidence, tool records and planner result. Existing Serial bridge
limits still apply; incompatible methods fail before devices connect. Do not silently
rewrite a VLA inference algorithm to enable the option.

With policy calls enabled, optional `run.warmup_before_start: true` uses **Prepare → Activate
Warmup → Start rollout** flow. RPent warmup runs the
selected VLA configuration through the shared isolated preview, including the
same gripper-range and calibrated path validation as policy calls. It uses the
configured task prompt and horizon without asking the planner to choose actions.
Predictions never reach actuators or formal rollout counters; the runtime retains
its measured-position hold. Start drains preview work, waits for backend reset
acknowledgement and acquires a new observation before enabling native Astra.
Reset failure leaves the planner blocked. Warmup does not relax plan-age, deadline
or motion checks; cold calls can still be rejected before later requests warm up.

For an explicit debugging condition, the experiment's optional
`agent.debug_max_plan_age_s` accepts a finite positive number of seconds. Only
managed GUI RPent applies it to its copied runtime configuration, including the
warmup preview. The provider request timeout, control freshness and
motion checks are unchanged. This field is included in collection fingerprints
only when set; use a distinct debug variant instead of changing a recorded condition.
The session manifest retains the base limit and opt-in, while the agent's
`runtime-config.json` records the effective limit. See
`configs/examples/rpent_gui_debug.yaml` for a disconnected 2-second example.

To test observation and motion primitives without VLA, set the experiment's
`agent.policy_enabled: false`. It accepts only a boolean and defaults to enabled.
The managed RPent runtime omits its policy worker and VLA warmup; native Toolkit
omits `execute_policy` and `observe.policy_contract`, and its prompt describes
observation and motion without VLA instructions. Runtime status advertises the
flag and `/execute` rejects requests independently of the tool schema. The same
mailbox, motion planner, executor, safety checks and GUI lifecycle remain in use;
Start still belongs to the operator. Explicit `false` participates in condition
fingerprints; omission and `true` retain the baseline. Use a separate
**RPent · Primitives only** variant.
`configs/examples/rpent_gui_primitives.yaml` provides a disconnected example.

### Separate service/client commands

Use `manimux-rpent serve --config <agent.yaml>` for the runtime and
`manimux-rpent run --config <agent.yaml>` for the planner. These are service/client
roles using the same recipe, not alternative execution modes. Model and camera
services still use their normal ManiMux launchers and the selected runtime YAML.
The public commands are maintained in [the user guide](../../../../guideline.html#agent-integrations).

`manimux/agents/rpent/cli.py` resolves the shared recipe. It references an ordinary
runtime `experiment` and a `motion` YAML, alongside endpoint, planner, task, memory
and output settings. Recipe paths resolve beside the recipe; command-line paths
resolve from cwd. The motion file, agent file and calibration source are hashed
into the runtime manifest. Machine addresses stay in the normal private station.
For separate `serve` / `run` commands, the shared agent recipe also accepts the
boolean `policy_enabled` field, defaulting to true. Both roles must use the same
setting; the runtime remains authoritative about the tools it permits.

Provider selection remains native to RPent: `planner` selects `api`, `codex` or
`claude_code`. API mode requires an explicit provider-prefixed `model`; `model: null`
is a template awaiting configuration, not an API default. The ManiMux launcher
passes `base_url` (API only), `max_tokens`, `planner_timeout_s` and
`claude_code_max_budget_usd` through to RPent's planner factory. Keys remain in the
provider's environment. Codex reads its native `CODEX_BASE_URL` / `CODEX_API_KEY`
settings; Claude Code retains its SDK authentication. Managed runs inherit the
launching runtime's environment. `endpoint` addresses the ManiMux dispatch server,
not the model provider. Do not place credentials in URLs or recipes saved as evidence.

For autonomous Astra decisions, select `planner: codex`, `model: gpt-6-astra`
and `reasoning_effort: high` (or another level supported by both RPent and the
selected model). The planner uses the account already authenticated on its host;
an API key is not required for the Codex account-login path. The `run` command
invokes RPent's native planner loop and records its transcript and tool calls.
Calling Toolkit methods from a chat or a scripted probe does not validate
autonomous planning. Keep the planner output directory unique for each run.
The launcher initializes RPent's native output context before resolving tool
schemas; MCP template substitution depends on it. A missing finish verdict is
a failed planner run, even if the provider turn completed without an SDK error.

By default, `ManiMuxToolkit` exposes observe, move_to, move_delta, rotate, set_gripper and
execute_policy. A planner can preposition, invoke a VLA, then move again in one
rollout. It never chooses a native-versus-dispatch mode or a separate VLA endpoint.
`execute_policy` uses the experiment's existing PolicyModel worker and action
adapter, including backend identity checks. RPent is not a policy.worker value.
A compatible Franka model now uses the ordinary joint8 adapter; the retired
RLinf state19/native delta7 bridge is not an implicit conversion.

When policy calls are enabled, the Toolkit reads the active runtime's `horizon_policy_steps` at startup.
It binds that limit into the native `execute_policy` JSON schema and argument
validator: `use_length` is an integer from 1 through the selected horizon. Its
omitted default is the smaller of 5 rows and that horizon. `observe.policy_contract`
exposes the same bound; the planner manifest saves both the contract and native
schema. Missing or invalid horizon metadata fails before planner invocation.
Changing experiments creates a new Toolkit and capability bound. The HTTP runtime
still validates every request independently; oversized requests are rejected,
never clipped or silently replaced.

The public examples are:

- `configs/agents/rpent.yaml`: entirely synthetic robot, geometry and policy.
- `configs/agents/rpent_franka.yaml`: standard Franka Pi05 joint-position policy recipe.
- `configs/agents/rpent_yam.yaml`: standard YAM Pi05 joint-position policy recipe.

The physical examples default to execute=false. Copy the corresponding
`rpent_*_motion.example.yaml` to the private local directory and supply measured
workspace bounds and calibration. Null bounds intentionally fail before connection.
Do not copy the synthetic demo's workspace into a physical deployment.

## Ownership and action contracts

`runtime/dispatch.py` is the single bounded-call mailbox and strategy, and
`dispatch/server.py` is its HTTP transport. Only EdgeRuntime writes hardware.
Both motion and policy calls share admission, session/request IDs, cancellation,
deadlines, operator pause and Finish. Policy calls and motion calls are mutually
exclusive within one rollout, including while IK is still planning. Cancellation
waits for planning to drain and an acknowledged hold plus a subsequent observation;
a late planning result cannot revive a cancelled command.
Local primitive planning uses that call's deadline, not the VLA observation-age
limit. Original observation timestamps remain in the recording. VLA admission
continues to enforce the experiment's max_plan_age_s. The Franka motion example
uses upstream RPent completion defaults (5 mm position, 0.04 rad orientation).

`runtime/motion.py` maps group-based tool arguments onto the existing
`kinematics/motion.py` bounded IK and joint-path planner. It owns no robot session,
control loop or model inference. All groups and tool coordinates come from the
configured kinematics and motion profile, without robot-name branches in callers.
Prepared motion chunks enter the same timeline, executor, command validation and
ordinary plan recording as VLA chunks. They are recorded as motion, not fabricated
model inference. Policy and primitive action intervals retain their own meanings:
VLA timing comes from the experiment; primitives sample at the control period.
RoboGUI reuses the same trajectory and action timeline for both. Primitive lanes
show the tool name without model inference latency. The Agent action field reports
measured arrival for primitives and completed chunk budgets for VLA calls; neither
is a physical task-success verdict. Pause and Finish remain the shared controls.

Coordinates are explicit in observe.motion_contract. move_to takes XYZ metres
and optional unit wxyz quaternion in the selected group's named base_frame.
Omitted orientation preserves measured orientation. move_delta is a translation
from measured TCP; rotate applies XYZ Euler increments in radians about base axes.
Both use measured goals and continue the previous submitted joint reference.
Paths are bounded joint interpolation, not Cartesian straight-line motion.
set_gripper takes normalized opening (0 closed, 1 open), never a hidden extra axis.
Pose tools retain the commanded gripper target while transporting an object.

Every group requires a TCP workspace and a declared normalized gripper coordinate.
The existing planner enforces primitive size, speed and waypoint budgets without
dropping rows. VLA prefixes also pass workspace checks; policy speed/timing remains
owned by the existing adapter/executor. Invalid normalized grippers are rejected
with row indices and values, never silently clipped by the RPent toolkit.
YAM's configured path validator reuses the calibrated geometry and collision
checker in integrations/rpent/geometry.py. It requires both base transforms, wrist
calibration, a measured table and the supported robot meshes. Both primitive and
VLA paths use that validator. These sampled checks do not model unconfigured
objects, payload geometry or tracking errors. A Franka workspace is not a collision
model or camera grounding calibration.

Motion targets remain on the shared timeline through measured arrival, so a late
control tick cannot lose the final waypoint. Cancellation and completion clear
that target through the normal hold fence. Ordinary policy horizons still expire.
Motion completion additionally checks measured TCP orientation/position and
requested gripper opening. Closing contact counts only for a device-reported closed
target with sustained fresh feedback for at least 0.1 seconds. It is exposed as
closing_contact_confirmed, not task success. Arrival timeout returns failed after
an acknowledged hold. Policy budget completion counts references, not arrival or
task success. Operator/evaluation verdicts remain authoritative.

The current bridge supports ordinary Serial, first_step_when_ready, inline decode,
complete joint-position chunks and optional pre-Start warmup. Blending, configured
chunk caps and specialized sampling/ensemble modes remain unsupported and fail
explicitly. Adding tool motion does not advertise broader sampler capabilities.
Subtask calls share policy history within a rollout. A new rollout resets it;
enabled warmup also resets the backend before formal Start. Standard RoboGUI
Start/Pause/Finish and the assembly's existing Home/release policy apply.
YAM retains required measured Home before release;
Franka does not acquire a Home capability. Agent tools cannot bypass these controls.

## Receipts and evidence

HTTP v1 exposes /status, /observation, /requests/<id>, /execute, /motion, /cancel and
/finish. Default bind is loopback; other interfaces require MANIMUX_DISPATCH_TOKEN.
Mutations require session_id. Motion submission also supplies request_id, tool,
arguments and timeout_s. The same request ID cannot switch between motion/policy
or change arguments. Identical retries return the saved receipt. Retention is 128
IDs per rollout; the server rejects further calls rather than forgetting IDs.

Receipts include call_type, plan_ids, completed_policy_steps,
completed_motion_steps, dispatched_control_steps and measured arrival diagnostics.
Tool steps and plan records join through dispatch_request_id and session ID.
The Toolkit reports selected/completed/failed tool decisions through the
session-scoped `/agent-decision` telemetry endpoint. It never changes runtime
admission or controls. Full `agent_decision` events (arguments, model label and
outcome) are retained in the episode recording; RoboGUI no longer has a separate
decision panel.

The planner's own transcript is RoboGUI's agent view: a bottom message bar, shown while
RPent is selected, with answers, reasoning summaries, tool lines, operator steering and
errors in chronological order and the time since Start ("planner is thinking" before the
first message). Messages fill preallocated Viser slots: adding a Viser element re-creates
its siblings' DOM, so appending would reset scrolling and text selection. The bar follows
new lines only while scrolled to the bottom, keeps text selectable for copying, and its
top grip resizes it.
`agents/rpent/transcript.py` follows the backend's live `planner/*_planner.txt` (Codex and
Claude Code write it as they run; the API planner returns its transcript only at the end,
so only its tool decisions appear live). Each block is recorded as an `agent_message`
event (text up to 16 KiB); heartbeats carry the latest 12, truncated for display.
Completed motion budgets are not reported as policy rows. Toolkit close cancels
its outstanding calls but does not finalize the rollout; RoboGUI Finish does.

The unified launcher enables request-scoped provider evidence under the session's
provider-responses directory. The worker records prompt, input hashes/state,
model identity, request sequence and timing. XPolicyLab and StarVLA wire replies
are captured before decoding, including malformed responses; decoded model outputs
and provider errors are also retained. Adapter rejection events/receipts identify
the evidence directory. Provider outcome is distinct from runtime acceptance.
Image payloads are not dumped into console logs; credentials are redacted from
structured evidence. Other backend clients must supply the same pre-decode capture
hook before claiming malformed-wire qualification.

## Acceptance and migration

Run the actual Toolkit through move_to -> execute_policy -> set_gripper ->
move_delta using the public offline probe. It accepts only the demonstrably
synthetic robot and no physical sensors:

```bash
python -m scripts.validation.rpent_offline_probe \
  --config manimux/configs/agents/rpent.yaml \
  --output data/acceptance/rpent-unified-<new-run>
python -m scripts.validation.rpent_finish_probe \
  --output data/acceptance/rpent-finish-<new-run>
python -m pytest tests/unit/test_rpent_unified.py tests/unit/test_policy_dispatch.py \
  tests/unit/test_rpent_agent.py tests/unit/test_rpent_native.py \
  tests/unit/test_franka_bridge.py
```

The finish probe injects RoboGUI Finish and planner `/finish` during in-flight motion
and VLA calls and requires zero robot commands afterwards. Local tests remain ignored. Cover cross-tool contention/idempotency, pause, deadline,
cancel during planning, no measured arrival, malformed-response persistence,
out-of-range grippers, recording and ordinary non-agent runtime regressions.
Software checks are not physical qualification. Before operator handoff, use
representative live/recorded observations, the real provider and a disconnected
command sink through the actual adapter/executor, including failure cases. Never
connect a motor driver merely to call the run a dry run.

The former manimux-rpent-env launcher and standalone manimux-dispatch console
entry are retired, including direct module launches. Legacy launch attempts return migration guidance
before constructing hardware. Old Env RPC classes and external-action fixtures
remain for protocol regression/other callers; they are not another RPent product
mode. Migrate native pi05_act/vla_grasp to execute_policy with an explicitly matched
ManiMux model recipe. Migrate arm aliases to configured group names and use each
group's base frame (the former YAM common left_base coordinates are not silently
reinterpreted). Native ready-receipt reset is replaced by the shared RoboGUI lifecycle.

Historical evidence from 2026-10-16 is retained under the private
`data/acceptance/rpent-real-vla-20261016/` bundle: four real DROID predictions through
the previous policy-only bridge used synthetic RGB and robot state. It did not
exercise autonomous planning or physical tasks and does not qualify this unified
implementation. Re-run real-provider and station checks for the selected deployment.
