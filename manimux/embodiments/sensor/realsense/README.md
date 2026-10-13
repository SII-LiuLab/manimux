# RealSense component

`RealSenseSensor(SensorBase)` provides pyrealsense2 capture for the camera service
and runtime sensors. Demonstration collection is outside this repository.

## Installation

Install into the existing hardware environment from the repository root:

```bash
uv pip install --python envs/yam/.venv/bin/python -e '.[realsense]'
```

The reference environment uses Python 3.12 and pyrealsense2 2.58.3.10794. Install the SDK in the environment instead of copying
it into the component directory. Bind devices by serial number; `get_device_ids()` enumerates without resetting them.
Construction neither imports pyrealsense2 nor opens USB; `start()` opens the pipeline and `close()` releases it.

## Frames and parameters

- `read()` returns an RGB uint8 `SensorFrame`; sequence comes from the RGB frame and time is host reception time.
- `capture()` also returns optional uint16 depth, meters-per-unit scale and Unix/monotonic timestamps for the same frame.
- `read_with_timestamp()` exposes RGB/depth/Unix seconds for direct diagnostic use.
  The camera service uses the shared `read() -> SensorFrame` interface.
- `camera_serial` binds the device; `width`, `height` and `fps` explicitly select the stream.
- `enable_depth`, `align_depth`, `flip`, `exposure_us` and `white_balance` follow configuration.
- The component defaults to 640×480, 30 Hz and RGB only; component YAML explicitly
  selects capture settings.

With the default `background: true`, the component captures in the background and network requests read the latest cache,
without sequentially waiting for each camera's next frame. Repeated cache reads retain identical timestamps and sequences.
Set `background: false` to call `capture()` synchronously without a capture worker.

First-frame waiting and `max_frame_age_sec` stall handling remain. SDK exceptions propagate to readers;
unlike the old network implementation, the pipeline is not silently restarted and devices/resolutions are not changed automatically.

Configuration lives in `manimux/configs/embodiment/sensor/realsense.yaml`. The YAM example binds camera serials in
`manimux/configs/local/yam_example.yaml`; the service entry point also converts the legacy `device_id` field.
Both configuration routes call the same component; there is no second RealSense pipeline implementation.

For camera services launched with `--experiment` or `--config`, each entry in the camera preset
can supply `options`, for example `cameras.d405_front.options.exposure_us` and
`white_balance`. These override component capture settings; explicit
station component bindings take precedence. Editing one entry affects only that
stream. Every experiment referencing the preset inherits its overrides.

`exposure_us` selects manual exposure and disables auto exposure; `white_balance`
selects manual white balance and disables auto white balance. Omitting a setting or
using `null` leaves that device control untouched; it does not restore a factory
default. Overrides apply when the camera service starts, not while it is running.

See [camera recipes and the full supported option list](../../../configs/embodiment/sensor/cameras/README.md).

## RGB exposure and QR timing diagnostic

From the repository root, run these in separate terminals on the same computer:

```bash
envs/yam/.venv/bin/python -m scripts.validation.realsense_qr_latency --show-qr
envs/yam/.venv/bin/python -m scripts.validation.realsense_qr_latency --camera front --csv /tmp/realsense-front.csv
```

The camera must be released by other applications first. The diagnostic reuses the
camera recipe and private station, opens one camera, and explicitly disables depth.
Use `--camera left` or `--camera right` for another station-bound device.
`--print-config` only resolves settings and does not open a device.

The diagnostic enables `collect_metadata` (off by default for normal capture).
`read_capture()` returns image, host receipt times, frame number and metadata in one
snapshot. Actual exposure is reported in microseconds directly from
`actual_exposure`, without an extra conversion factor. Missing metadata displays
`N/A` and is blank in CSV; the configured exposure is never used as a substitute.
Librealsense `rs_frame.h` defines raw `sensor_timestamp` as the exposure midpoint
and raw `frame_timestamp` as readout/transmission start, both in device microseconds.
`get_timestamp()` is separately recorded in milliseconds alongside its clock domain.
Device timestamps are not automatically comparable to Unix host time.

The diagnostic also records `sdk_arrival_unix_ms` from `time_of_arrival`.
In librealsense 2.58.3, this is host Unix time at entry to the UVC backend
callback, before subsequent SDK frame allocation/copy and pipeline delivery.
It is integer milliseconds, with less than 1 ms truncation uncertainty; it is
not a microsecond field. See the upstream
[UVC callback](https://github.com/realsenseai/librealsense/blob/v2.58.3/src/uvc-sensor.cpp),
[metadata registration](https://github.com/realsenseai/librealsense/blob/v2.58.3/src/sensor.cpp),
[integer metadata parser](https://github.com/realsenseai/librealsense/blob/v2.58.3/src/metadata-parser.h)
and [host clock](https://github.com/realsenseai/librealsense/blob/v2.58.3/src/core/time-service.h).

For each sampled frame, `sdk_to_python_ms` subtracts SDK arrival from Python
receipt. When the frame timestamp domain is `global_time`, `global_to_sdk_ms`
and `global_to_python_ms` also compare the mapped device timestamp against these
host timestamps. The two segments sum to the total for that same frame. Negative
values are retained to expose clock mapping errors. Unsupported comparisons stay
blank in CSV and display as `N/A`. These measurements work without a decoded QR.
Neither the SDK nor Python receipt measurement includes subsequent preview display.

The host receipt timestamp is taken after the RGB buffer copy and before metadata
reads. QR-to-host is this host time minus the decoded same-host QR timestamp. It
includes source display delay and camera delivery, and excludes later QR decoding
and preview display. The CSV contains one row per decoder-sampled frame, including
frames without a decoded QR. Capture, preview and decoder use latest snapshots;
slow decoding can skip frames. Metadata on the preview belongs to its displayed
frame; the last QR result has its own frame number. Each second the console reports
a preview-frame metadata snapshot and cumulative QR statistics. Exposure duration
is not exposure-to-host latency, and no fixed monitor-delay correction is applied.
