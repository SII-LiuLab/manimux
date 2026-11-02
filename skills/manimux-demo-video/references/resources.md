# Resource limits and recording pitfalls

The user's desktop froze repeatedly during earlier work. The cause was not proven.
An uncapped GPU browser capture ran at a reported ~1485 requestAnimationFrame calls
per second without improving useful captured cadence. Never restore
--disable-frame-rate-limit or --disable-gpu-vsync. Prefer the bundled recording.

The launcher binds the renderer and encoder to one permitted CPU core at nice 15.
The systemd scope has a 1536 MiB memory ceiling, zero swap allowance, and requests
60% CPU quota. This host does not delegate the CPU controller to the user manager;
its accepted CPUQuota property alone therefore does not enforce the quota.
throttle.py enforces an additional 60 ms run / 40 ms freeze duty cycle for that
scope. It unfreezes on ordinary exceptions and handled interruption. Do not kill
the supervisor with SIGKILL. If interrupted externally, unfreeze only its exact
scope with `systemctl --user thaw <unit>` before stopping the owned render.

Do not change global CPU/GPU settings, other services, or unrelated processes.
Check the actual scope's memory.max, memory.swap.max, affinity, and cgroup state;
measure rather than inferring all limits from systemd property strings. The V4
render used roughly 170–230 MB at sampled points, not a measured peak guarantee.

The decoder, encoder, OpenCV, and numerical libraries use one thread. Frames
stream directly to FFmpeg. Text caching is bounded. One decoded video frame and
one thumbnail are shared across the 36 wall cells. Blur a small intermediate image
then upscale it. Avoid retaining a full movie as uncompressed images in RAM.

If recording is necessary, first inspect source cadence. Software WebGL + CDP
screencast previously produced choppy motion despite a 30 fps container. A capped
GPU capture can help, but do not run it alongside rendering and desktop-heavy work.
Use isolated browser profiles and demo ports. Never connect hardware or trigger
Prepare/Warmup/Start just to obtain B-roll without explicit authorization.

Select source frames using round(time * source_fps), not int(): floating-point
subtraction can cause duplicated and skipped frame indices. Downsample clips to
the intended output cadence once before repeated composition. No optical-flow
interpolation or capture is needed for the default 30 fps film.
