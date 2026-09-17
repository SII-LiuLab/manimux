# viewer_bridge.py — optional universal_viewer mirror

Design rationale for the `--viewer` flag on `goto_joints.py` and
`run_teleop.py`. There is no real incident history yet (this is the first
pass) -- this file records the design decisions instead, so the next person
touching it knows which constraints were deliberate.

## Why this isn't a policy integration

`universal_viewer` (see `/home/jw/Desktop/project/universal_viewer`) is
built around two messages: `PolicyPlan` (a predicted future action chunk)
and `RobotSnapshot` (achieved state). Tianji has no real-time policy yet --
teleop itself is the source of truth driving the robot -- so this bridge
only ever publishes `RobotSnapshot`. This is a real, supported mode on the
viewer side ("observe mode" in its own terms), not a stripped-down
placeholder: no `PolicyPlan` ghost trajectory, and the dashboard's
pause/home/step buttons are cosmetic here since nothing polls them back.
Wiring those up would mean giving the viewer a say over the control loop,
which is a bigger decision than "can I see the arm move" -- deferred until
there's an actual reason to pause teleop from the browser.

## Why a separate module instead of inside drivers/arm_driver.py

`arm_driver.py` is the safety-critical layer (mode switching, command
dispatch, tracking-error faults -- see docs/drivers.md). Folding a network
publish call into it would couple that module to an unrelated concern and
make it harder to reason about "can this call block the control loop."
`viewer_bridge.py` is a standalone best-effort side-channel that the
*callers* (`goto_joints.py`, `run_teleop.py`) opt into at their own loop
points; `arm_driver.py` has no idea it exists. One consequence: since
`arm_driver.move_to_joints()` has no per-frame hook, `goto_joints.py` only
publishes before the move, after it lands, and during the post-arrival hold
loop -- not the transit itself. Good enough to confirm "did it get there
and is it holding," not a live scrub of the move. Extending
`move_to_joints()` with an optional per-frame callback is the natural next
step if that turns out to matter, but it touches shared safety code, so it
wasn't done speculatively here.

## Why RobotSnapshot always carries both arms

`RobotConnection.subscribe()` reports both arms' feedback unconditionally,
regardless of which one a script is actively driving (`ArmDriver.state()`
already relies on this). `viewer_bridge.publish_state()` takes that same
dict and always concatenates A then B. The alternative -- publishing only
the arm a script happens to be driving -- would make `TianjiAdapter`'s
single-arm width (7/8 values) ambiguous about which physical side it is:
`goto_joints.py --arm B` would otherwise report B's motion under the
viewer's "left" slot, since a bare 7-value vector always means "left" (same
simplification `YamAdapter` already makes). Always publishing both sides
sidesteps that: the driven arm moves, the idle one sits at its real
resting pose, and both consistently line up with `TianjiAdapter`'s
`left = A, right = B` grouping.

## Why it's not a pip dependency

`universal_viewer` declares `mujoco`/`viser`/`yourdfpy` as base dependencies
(for the dashboard itself, not this bridge) -- installing it as a normal
teleop dependency would pull all of that into a venv that also runs a
250Hz hardware control loop, for no benefit to teleop. `ViewerBridge`
imports `universal_policy_viewer` lazily, only inside `__init__`, and only
when `enabled=True` -- exactly the pattern its own reference architecture
(`manimux`, ziyang branch, `viewer/bridge.py`'s `ViewerBridge`) uses for the
same reason. Point `UNIVERSAL_VIEWER_SRC` at a sibling checkout's `src/`
(defaults to `../universal_viewer/src` next to this repo) if it isn't
there; same convention as `MARVIN_SDK` in `drivers/arm_driver.py`. Running
without `--viewer` needs neither `pyzmq` nor a universal_viewer checkout at
all.

`pyzmq` itself (the actual wire transport, not `universal_policy_viewer`'s
Python code) is a real runtime dependency the moment `--viewer` is used, so
it's declared as an optional extra (`[project.optional-dependencies]
viewer`) rather than smuggled in unlisted: `uv sync --extra viewer`
installs it. Forgetting this step fails loudly inside
`ViewerBridge.__init__`'s lazy import (`ModuleNotFoundError: zmq`), not
silently.

## Publish rate

Throttled to 20Hz (`ViewerBridge.publish_hz`) independent of the 250Hz
control loop -- the dashboard doesn't need control-rate updates, and
`ViewerPublisher`'s ZMQ socket only buffers 2 messages
(`universal_viewer/bridge.py`'s `SNDHWM`), so publishing faster than the
receiver drains just means more dropped frames, not smoother rendering.
`ViewerBridge.due()` lets a caller skip the state fetch itself on ticks
that would be throttled away (see `goto_joints.py`'s hold loop) rather than
paying for an SDK read it can't use.
