# Sensor components

All camera components implement `SensorBase`: construction is inert, `start()` opens
the device, `read()` returns a `SensorFrame`, and `close()` releases it, including
after partial startup. A physical camera returns one frame; network sensors may
return a named bundle. Images are RGB `uint8` arrays. Capture time uses the injected
host monotonic clock; cached reads retain the original time and sequence.

Physical camera constructors accept keyword arguments `name`, `camera_serial`,
`clock`, and their device options. Component YAML selects the class through
`implementation: package.module:ClassName`. Both robot assembly and camera service
use that implementation and the component's options/hardware settings.

The camera service uses `build_camera()` and the same lifecycle for every device.
Standalone service YAML can use the built-in `type` aliases `realsense`, `orbbec`,
and `taccap`, or specify `implementation` directly. `device_id` remains accepted
as the old spelling of `camera_serial`; do not specify both. No server branch is
needed for a custom camera class. Runtime network data sources continue to use
`build_sensor(config, clock)` and the `manimux.embodiments.sensor` factory plugins.

The service preserves its existing RGB and Unix-seconds network format. It converts
host monotonic capture times using a fixed wall-clock offset for the service
lifetime; it does not timestamp a cached image again when serving a request.
Sensors in the service must use the host monotonic clock domain.

RealSense defaults are RGB-only, 640x480, with no warmup. Service YAMLs explicitly
set their existing 15-frame warmup; RGBD YAML also enables depth and alignment.
External standalone configurations relying on the former server-only defaults
must explicitly set `height: 360`, `enable_depth: true`, `align_depth: true`, and
`warmup_frames: 15` to retain that behavior.

For a new camera, add the implementation and component YAML, then test deferred
startup, RGB conversion, cached metadata, failure cleanup, and restart with a
fake SDK. Do not open hardware during imports, construction, or model loading.

## Background runtime reading

`SensorReader` in `reader.py` operates on the existing `SensorBase` interface,
independently of camera vendor, robot or policy. Configure it in the experiment:

```yaml
run:
  sensor_reading:
    mode: background         # inline remains the default and comparison mode
    poll_hz: 100.0           # source polling, not camera acquisition or control Hz
    max_frame_age_s: 0.5
    startup_timeout_s: 5.0
    shutdown_timeout_s: 2.0
```

One runtime thread owns `start/read/close` for all configured sources. A network
bundle remains one request for all selected cameras. The control loop only reads
the latest complete snapshot; it never waits for a new frame after startup. The
reader waits for the first complete snapshot before robot connection, publishes
owned read-only RGB arrays, retains capture times/frame identities on cached reads,
and reports background exceptions or expired frames to the control loop. An empty
source during startup is retried; losing frame names after readiness is an error.

Polling and image copies occur outside the snapshot lock. No unbounded frame queue
is introduced. Sources must bound their blocking I/O below the configured shutdown
timeout. A shutdown timeout reports failure rather than closing a socket/device
concurrently with a blocked read; its owning thread performs cleanup when I/O ends.
The reader does not modify robot states, timeline timestamps, action intervals or
the control loop's sleep policy.

The REQ/REP camera driver retains the server's existing Unix capture timestamps,
converts them with a fixed local Unix-to-monotonic offset, and uses capture time as
frame identity because this wire format does not include a hardware frame counter.
Repeated replies for the same capture keep the same timestamp and identity. Camera
server and runtime hosts must have synchronized Unix clocks; both clocks must be
stable. A local offset jump (default tolerance `options.clock_jump_tolerance_s: 0.02`)
or backward source time fails visibly. Transport receipt time is not capture time.

`max_frame_age_s` checks the source capture age on every cached read, not the last
successful fetch age. Retaining source timestamps also lets recordings distinguish
new frames from repeated reads. It does not make image capture simultaneous with
robot feedback or change the existing inference observation-time anchor.
