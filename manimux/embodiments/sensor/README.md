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
