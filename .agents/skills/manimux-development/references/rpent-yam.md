# RPent YAM VLA execution

RPent owns task planning, observation-to-policy transforms and Pi0.5 inference.
ManiMux owns YAM connections, camera access, action dispatch and recording. The
bridge implements RPent's existing YAM environment RPC; no RPent source changes
or second policy server are required. This is an environment integration, not a
new `PolicyModel` backend.

## Installation and configuration

Use Python 3.11 or 3.12. Install ManiMux and RPent's base dependencies in the
control environment; install the YAM SDK using the existing robot runbook.
Keep the existing RPent Pi0.5 server in its own model environment.

```bash
# From ManiMux, with your control environment activated:
uv pip install -e . -e '/path/to/RPent[yam]'
python -m manimux.integrations.rpent.server --help
```

Copy `manimux/configs/experiments/rpent/yam_vla.yaml` for your experiment. Use
`--local /path/to/station.yaml` for device bindings. The station's
`services.camera.endpoint` retains its existing PUB meaning;
`services.camera.request_endpoint` supplies the bridge's REQ connection.
The example enables aligned depth for all three streams; unaligned depth,
flipped RGB-D profiles and missing depth are rejected. RGB-only camera requests
and existing RGB clients remain unchanged.

The example defaults to `robot.options.execute: false`. This prevents physical
command dispatch but still connects to configured hardware; it is not a
hardware-free simulator. Enable execution in your private experiment only after
checking the station, SDK limits and scene. Neither connection nor reset commands
an initial pose, and shutdown does not Home. Configure the existing
`executor.command_safety` position/velocity/acceleration limits for your deployment;
the bridge does not invent station-specific limits or suppress safety failures.

Match `rpent.task_name`, `seed` and `max_episode_steps` to the RPent invocation.
The sample uses the existing `tabletop_cleanup_a` task. Camera mapping is explicit:
`top: d405_front`, `left: d405_left`, `right: d405_right`.

## Launch and run

These commands start hardware services; perform them in the appropriate prepared
environments when you intend to connect the station.

```bash
python -m manimux.servers.camera.server \
  --experiment /path/to/yam_vla.yaml --local /path/to/station.yaml
python -m manimux.integrations.rpent.server \
  --experiment /path/to/yam_vla.yaml --local /path/to/station.yaml \
  --host 127.0.0.1 --port 8110
```

Start the existing RPent YAM VLA server using its YAM guide and trained qpos14
checkpoint. The bridge does not launch or reconfigure inference. In an operator
terminal, use an RPent site YAML containing the same `operator_receipt_path`
as the ManiMux experiment:

```yaml
operator_receipt_path: /tmp/manimux-rpent-yam-receipt.json
```

```bash
# Read the current episode ID, then confirm scene readiness locally.
python -m robots.yam.operator_control --robot-config /path/to/operator.yaml \
  --endpoint http://127.0.0.1:8110 --event status
python -m robots.yam.operator_control --robot-config /path/to/operator.yaml \
  --endpoint http://127.0.0.1:8110 --event start \
  --episode-id <pending-id> --note 'Scene checked and ready'

# In the existing RPent environment (retain your normal planner options):
rpent --robot yam --task-name tabletop_cleanup_a --seed 0 \
  --max-episode-steps 1000 --env-endpoint http://127.0.0.1:8110 \
  --vla-endpoint http://127.0.0.1:8120
```

Use the new episode ID returned by `start` for subsequent operator events.
RPent's `pi05_act` predicts 30 rows and submits its selected five-row prefix.
Each row is `[left joints x6, left opening, right joints x6, right opening]`:
absolute radians and normalized openings (`0=closed`, `1=open`). Commands run at
30 Hz with DirectExecutor, without seam blending, extra normalization or action
prefix expansion. One batch (at most 30 rows) may be outstanding; another is
rejected rather than replacing it. Completion counts dispatched source rows,
not physical arrival, and is acknowledged after the last control period.

`env.request_stop` signals the control thread without waiting for a batch lock.
A timeout also latches stop. A fresh local ready receipt and reset are needed to
resume. Reset rejects rearming until the control thread has acknowledged the
hardware hold and cleared execution history; a rejected reset leaves the ready
receipt unconsumed. `read_control_state` returns joint/control feedback without
copying RGB-D frames, including `ready_to_reset` for a matching unused receipt. `success`, `failure` and `abort` receipts use RPent's existing operator
commands and are matched to the current episode; only an operator success receipt
marks task success. Executing all five rows alone is not task success.

## Supported scope and limitations

The bridge supports observation, status, single/chunk qpos actions, RGB-D camera
reads, episode reset and stop. It advertises RPent's fixed YAM metadata for client
compatibility; that upstream metadata includes extension names and is not an
independent guarantee that every high-level operation is implemented.
`plan_arm_path`, `reset_to_configured_qpos`, `recover_up` and `release_gripper`
explicitly report unsupported. RPent still exposes its normal YAM tool list;
restrict the task to `pi05_act`, observation/status and finish. High-level
manipulation tools and exploration/recovery workflows are outside this version.

RGB and metric depth retain the same capture. Calibrated world projection is
explicitly unavailable: camera metadata marks `projection_valid=false`, so RPent
records unavailable world maps instead of fabricated coordinates. Left TCP comes
from ManiMux's assembly model; right TCP in the left-base frame is unavailable
without inter-arm calibration. These limitations do not change VLA RGB/qpos inputs.

The runtime retains ManiMux's exclusive robot lock and recording lifecycle.
Both startup failures and hardware faults terminate pending actions and require
inspection before restarting. Offline synthetic validation is separate from real
checkpoint inference and physical task success.

## Offline checks

Where the repository's local test workspace and RPent dependencies are installed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/unit/test_rpent_external.py tests/unit/test_robogui_recovery.py
python -m ruff check manimux/integrations/rpent manimux/runtime/external.py
python -m mypy manimux/integrations/rpent manimux/runtime/external.py \
  --follow-imports=silent --ignore-missing-imports
```

Tests use synthetic YAM/cameras/VLA with the actual execution loop and RPent
client/primitive modules. Tests follow ManiMux's ignored local test-workspace
policy. A restricted environment may prevent the HTTP loopback test or importing
RPent's complete planner/tool discovery. Protocol-module tests alone do not
verify full Toolkit startup, trained inference, USB/CAN access or task success.
