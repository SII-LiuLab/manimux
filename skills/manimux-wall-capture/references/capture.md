# Portable capture paths

Use the local agent's available tools and the station's existing recorder. Discover
the display system and window dimensions; never assume Linux/X11, DISPLAY=:0,
port 18086, or a particular desktop. Keep capture at original speed.

## Existing native recorder / OBS

On Wayland use an existing PipeWire/desktop portal window capture or OBS source.
On Windows/macOS use an existing app/window recorder. Select only RoboGUI and
capture 1920×1080 at 30 fps, or 1280×720 when needed. Disable audio. Review a short
sample. Export or normalize to H.264 MP4 yuv420p after the run. If an OS screen-
recording permission is missing, ask the operator to grant it; do not bypass it.

## Included X11 helper

Only on an actual X11 desktop with FFmpeg's x11grab input. Arrange RoboGUI's content
area to the desired 16:9 size. Obtain the region from the actual visible window
using a screenshot or xwininfo; do not guess coordinates. The rectangle is absolute
screen pixels and must exclude decorations. X11 screen capture records whatever
covers that rectangle, so keep the GUI unobscured.

```bash
python3 <skill>/scripts/capture_x11.py --display "$DISPLAY" \
  --region 1920x1080+100+80 --duration 60 --output <raw-gui.mp4>
```

The coordinates above are examples, not defaults. The helper refuses a missing
display, Wayland session, non-16:9/odd-sized region, or existing output. It captures
30 fps, disables audio, limits x264 and filter threads to one, uses veryfast CRF20,
and lower scheduling priority. Ctrl-C asks FFmpeg to finalize. It never opens the
browser, launches services, selects a robot, or triggers motion. One encoder thread
is a limit, not a promise that the station can sustain 1080p30.

If capture overruns or interferes with the task, stop the recorder first. Reduce the
actual browser region to 1280×720 or use another machine. Do not cap the robot
runtime's CPU or GPU, freeze live processes, force unlimited browser frame rates,
or silently change inference/controller timing. Do not run final video rendering,
multiple capture processes, optical-flow interpolation or thumbnail generation
during a hardware run.

## After the run

Keep raw media locally. `submission.py normalize` scales down only when above the
chosen maximum width, converts cadence to 30 fps and encodes one thread at low
priority. It writes a source sidecar with original properties and SHA-256. It does
not crop, speed up, interpolate motion, or prove every encoded frame is distinct.
A source below 24 fps needs a better capture or a clearly disclosed limitation.
Full decode validation is also single-threaded. Run all offline operations one at
a time, away from active control when they could contend for station resources.
