# Shot contract for the editor

| Shot | Requirement | Composition |
| --- | --- | --- |
| Primary GUI | Required; 30–60 s preferred, ≥12 s useful execution | Full clean RoboGUI with fixed view, visible robot/object camera, task/run state |
| External robot | Optional; same take | Fixed landscape view, both gripper/work area visible, no fast pan |
| Evaluation | Optional; 10–30 s preferred | Human reviews result, enters honest score and saves; actual visible acknowledgement |

Keep 3 seconds of handles where possible. Nominate in/out in seconds from the
submitted file's first frame, not wall-clock time. The main interval should show
meaningful continuous model-driven manipulation (approach, grasp, move, place), not only idle
waiting or a rotating 3D model. Never trim around a failure to claim success. A
failure can be useful footage when labelled accurately.

Show the real model's action/trajectory visualization alongside its physical
execution. If unavailable, resolve that gap before submitting a primary take.
Keep the visualization type explicit:
predicted future chunks, executed controller commands, and measured robot motion
are different evidence. Do not play back a prerecorded path, use a scripted
trajectory or manually teleoperate while labelling it as model-generated execution.

Keep composition consistent across takes on the same station. Do not force every
embodiment into the YAM layout: fit the relevant arms/mobile base into view and
retain the camera panel that best shows the physical task. Preserve readable
operator controls while giving the scene most of the available space. Avoid
pointer motion over the action, tooltip popups, menus, scrolling and viewpoint
orbit during the nominated interval.

1080p30 native is enough for small wall cells and a full-screen lead shot. 1440p30
native is preferable when already available. Do not record 60 fps unless the
station already sustains it without affecting control. 720p30 is valid for smaller
tiles; identify it as such. Do not pad or upscale screenshots into a fake higher
resolution. Capture an appropriate viewport rather than stretching the image.

For an optional external clip, record `offset_from_gui_seconds`: the external
clip's first frame time minus the GUI clip's first frame time. Example: if the
external recording starts 2 seconds later, offset is +2; an event at GUI time 8 s
appears at external time 6 s. Use a shared visible event/timebase and record the
method; use null when unknown. Never invent alignment or move a robot just to
create a synchronization cue.

The nominated interval must show the physical camera, robot/trajectory view, model
identity and live inference activity simultaneously. Use a fixed layout; do not
switch tabs or alternate between the model and robot. An adjacent genuine live
model-status panel is acceptable if RoboGUI lacks that information. A separate
external recording supplements the main shot and does not satisfy a missing
physical camera panel. An absent required view is a blocker for primary footage.

Show a real model inference signal: completed-inference counter, measured inference
latency/results, or action-chunk identifiers tied to that service. The indicator
must change with actual inference, not with browser/camera refresh or mock timers.
Document its source and confirm it matches the model that drives this robot.

Before marking visual review complete, watch the full nominated interval and check
for freezes, repeated stale camera frames, jump cuts, unrelated motion, clipped
text, hidden task objects, private content and recorder overlays. File metadata
alone cannot establish smooth motion or real execution.
