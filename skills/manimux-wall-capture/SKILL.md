---
name: manimux-wall-capture
description: Collects real ManiMux RoboGUI execution footage from a local robot station and packages clips with provenance for a multi-embodiment product video wall. Use when asked to contribute station footage, capture an embodiment, or prepare a ManiMux video submission.
disable-model-invocation: true
---

# Contribute ManiMux video-wall footage

## Robot authorization — required for every task

Without explicit user permission, do not start or restart a robot, connect its
controller, enable actuators, or cause robot motion. This includes Prepare, Warmup
on connected hardware, Start, Home, gripper calibration, replay and recovery/release.
Invoking this skill or requesting integration, configuration, testing, evaluation or
video capture does not grant that permission. Reuse explicit authorization already
given for the same robot and operation; do not ask for it repeatedly.

Complete configuration, code and disconnected validation first. If permission is
missing, show the concrete robot, resolved configuration and intended operations,
then ask before the first hardware-affecting action. Configuration review is not
robot-start authorization. Leave physical controls with the operator by default.
`execute: false`, “dry run” and “warmup” are not evidence of isolation: constructors,
connect and cleanup can move hardware. Use disconnected models/fake devices offline.
A separately authorized GUI/model/camera service launch may proceed only after
verifying it will not connect or actuate the robot. Preserve existing service and
hardware ownership; do not stop, restart or release an active robot to test a change.

For every model/backend, retain the shared observation/action adapter and execution
protocol. Do not add a backend-specific `native` action path to make a capture work.
Reuse resolved embodiment/tool geometry, station calibration and model metadata;
do not ask the user to repeat facts already recorded. If development is necessary,
inspect this checkout's shared contracts first. Only a demonstrated inability to
conform justifies requesting explicit user approval for an exception; neither an
existing legacy path nor a footage request grants it.

You are the contributor's local agent. Deliver a validated ZIP of this station's
footage to the contributor, who will forward it to the video editor. Do not upload
or message anyone automatically. This package is self-contained: no originating
workstation paths, ports, installed agent plugins, or other skills are required.

## What the editor needs

**The subject is real model-generated execution trajectories on physical robots.**
For each primary take, an actual trained policy/checkpoint must run inference on
the task observations, and its actions must drive the robot through ManiMux's
normal adaptation, scheduling and execution path. A connected robot or moving GUI
is not enough. Capture the resulting physical motion and its corresponding GUI
trajectory/action view together in the same continuous shot.

**All three must remain visible in the nominated interval:**
1. The physical robot and task through a live camera.
2. RoboGUI's robot and action/trajectory visualization from this run.
3. The active model/checkpoint identity and genuine live inference activity.

Use real completed-inference counts, inference latency/results or action-chunk IDs
from the running model/runtime. A static model name, loaded/connected badge,
generic spinner, camera FPS or browser animation is not live inference evidence.
If the GUI lacks model telemetry, arrange an existing read-only live model-status
panel beside it within the same recording, sourced from the actual active service.
Do not invent telemetry or add a decorative model badge in post-production.
If one view is hidden, unavailable or only supplied in a separate clip, the primary
shot is incomplete. Resolve the framing/telemetry gap before declaring it ready.

Verify the model → inference output → runtime execution → physical motion chain
from the active run. Record the public model/checkpoint identity and how that
chain was established. Displayed future chunks are predictions, not measured or
executed paths; identify which kind of trajectory the GUI shows. Do not fabricate
an executed-path overlay if this local version only displays predicted chunks.
Scripted paths, demos, teleoperation, replay and unexecuted model predictions do
not meet the primary footage requirement. Successful task completion is optional;
authentic model execution is required.

The product film moves from full-screen RoboGUI into the upper-left cell of a
6 × 6 wall, then blurs to a white title card. The editor supplies the transitions,
labels, music and title. Submit clean continuous footage, not an edited trailer.
Do not build your own wall, add titles, animate the camera, or accelerate execution.

**Required per station:** one actual RoboGUI screen recording, preferably 30–60
seconds, with a nominated continuous 12–20 second interval of visible execution
of a real model's actions by the physical robot. Two to four genuinely different takes are welcome if readily
available; do not repeat the same clip to inflate the count. Different embodiments
and tasks are more useful than many identical runs. Do not perform new training or
invent a model/algorithm combination just for this capture.

- Landscape 16:9. Native 1920×1080 at 30 fps is the default. Native 2560×1440 at
  30 fps is welcome if smooth; native 1280×720 at 30 fps is an accepted fallback.
- H.264 MP4, yuv420p, original speed, no audio in the submission. Keep the source
  recording locally. Never upscale a smaller source just to satisfy resolution.
- Capture RoboGUI's robot/trajectory view, live physical camera panel(s), active
  model identity, live inference state and relevant run state/task together.
  An adjacent genuine model-status panel is acceptable; exclude unrelated terminal
  content, address bars, desktop, notifications and private information.
  Keep the layout and 3D viewpoint fixed during the take.
- Make at least one live physical camera view large enough to recognize the robot,
  object and task. A moving 3D model alone is not evidence of physical execution.
- Start capture about 3 seconds before the operator starts; retain about 3 seconds
  after motion. Record the whole task when feasible. Long tasks may exceed 60 seconds;
  nominate a continuous interval rather than speeding up the recording.
- English UI/text where supported. Hide passwords, access tokens, private paths,
  names or faces that the contributor does not intend to distribute.

**Optional:** a synchronized external robot camera clip of the same take, and a
separate 10–30 second clip showing actual human review → score → save. Only record
an evaluation save if this local version supports it and the operator supplies an
honest score. Do not fabricate a success or alter experiment records for the video.
See [shot framing and acceptance](references/shots.md).

## Discover the local station first

1. Locate the actual ManiMux checkout using the current workspace, project files
   and operator context. Read its AGENTS.md, README and local deployment/runbooks.
   Inspect git revision/dirty state and CLI `--help` using its existing interpreter.
   Never assume `manimux up`, `serve`, `run`, a port, an SDK path, or a YAML layout
   exists in this version. Prefer existing station-specific instructions.
2. Inspect running services and existing GUI URL before starting anything. Resolve
   embodiment, policy/checkpoint public identity, inference strategy, task, cameras,
   and source of those facts from the running config/session where possible.
   A config filename alone is not proof of the active model or strategy.
3. Ask only for unresolved facts that affect recording: which robot/task is intended,
   who operates it, whether an existing successful recording may be used, and the
   public station/checkpoint labels. Mark unavailable provenance unknown, never guess.
4. Reuse a healthy local GUI and supported recipe. If setup is needed, use the local
   version's documented launch path. Preflight does not establish hardware readiness.
   Leave preparation, warmup, Start, reset/Home and physical movement with the local
   operator unless they explicitly authorize those exact actions. Requesting this
   skill authorizes recording and packaging, not autonomous robot movement.
5. If no safe working hardware run is available, report the concrete blocker. An
   existing real recording is acceptable with truthful provenance. Replay, teleop,
   and synthetic clips must have their own evidence mode; they are not substitutes
   for the requested real policy-execution contribution.

## Capture without disturbing execution

Read [capture methods and resource limits](references/capture.md). Prefer an
existing native window recorder, OBS/PipeWire on Wayland, or the included X11 helper
when applicable. Do not install a new browser, change GPU settings, kill existing
services, or start an unbounded screen recorder as an assumed prerequisite.

Do one five-second stationary trial recording first. Inspect its image and resource
usage, then coordinate the actual take with the operator. If recording affects
control/UI responsiveness, stop the recorder, lower to 720p30, or use an independent
capture machine. The robot control loop has priority over filming.

Do not apply the editor's cgroup-freeze duty cycle to a live browser, robot runtime,
or capture process: it would introduce timing stalls and dropped frames. Capture is
real-time work. Limit encoder threads and resolution instead. Heavy normalization
and full decode validation happen after the run, sequentially.

## Prepare the submission

All helper commands use Python 3 standard library plus installed FFmpeg/ffprobe.
Replace `<skill>` with this skill directory and use actual local paths. Helpers do
not connect robots or launch ManiMux services.

```bash
python3 <skill>/scripts/submission.py init --out <submission-dir> --station <public-station-id>
# Optional codec/fps normalization after capture; no cropping or speed change:
python3 <skill>/scripts/submission.py normalize --input <original.mp4> \
  --output <submission-dir>/clips/take-001/gui.mp4
# Fill manifest.json with actual observations and suggested interval; see schema.
python3 <skill>/scripts/submission.py validate <submission-dir> --decode
python3 <skill>/scripts/submission.py pack <submission-dir> --output <delivery.zip>
```

Use a new output directory; do not overwrite other runs. Copy an already conforming
source to the named clip path instead of re-encoding. Normalize only if necessary;
its `.source.json` sidecar records source SHA-256 and original media properties.
Transcoding cannot repair dropped capture frames. Inspect actual motion playback.

Fill [manifest fields](references/manifest.md), using evidence from the active run.
For primary takes, set `trajectory_source` to `model_inference`, identify
`trajectory_display`, and supply a known public policy/checkpoint identity.
Explain in `execution_evidence` how actual inference and robot execution were
verified; an open port, model config filename or moving 3D scene is insufficient.
Set `model_identity_visible`, `live_inference_visible`, and `trajectory_visible`
only after reviewing their simultaneous visibility with the physical camera.
Record the actual signal source in `inference_visibility_evidence`.
The GUI path must be a relative file under `clips/`. Mark task outcome separately
from evidence mode; a service exit or animation does not establish task success.
Set `visual_reviewed: true` only after actually watching the nominated interval.
Set `content_reviewed_for_sharing: true` only after inspecting the shareable files.

Validation checks technical media properties, intervals, metadata and checksums.
It cannot verify that a physical task happened or that a claimed score is honest.
It reports low-resolution/source-cadence and non-policy modes separately; only
`real_robot_policy` takes with no blockers count as target wall contributions.

## Finish

Return the ZIP path and a concise inventory: station, embodiment, policy, inference,
number of real-policy takes, each suggested interval, task outcome, and limitations.
Include optional evaluation footage only with actual operator-entered scores.
Report any capture-induced latency/dropouts. Leave existing services as found;
stop only a recorder you started. Keep the original recordings locally until the
editor confirms receipt. The contributor forwards the ZIP; no automatic upload.
