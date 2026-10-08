# RealSense 组件

`RealSenseSensor(SensorBase)` provides pyrealsense2 capture for the camera service
and runtime sensors. Demonstration collection is outside this repository.

## 安装

从仓库根目录安装到已有硬件环境：

```bash
uv pip install --python envs/yam/.venv/bin/python -e '.[realsense]'
```

当前参考环境为 Python 3.12、pyrealsense2 2.58.3.10794。SDK 通过环境安装，不复制
到组件目录。设备使用序列号绑定；`get_device_ids()` 只枚举、不重置设备。
构造组件不导入 pyrealsense2，不打开 USB；`start()` 打开 pipeline，`close()` 释放它。

## 帧与参数

- `read()` 返回 RGB uint8 的 `SensorFrame`；序号来自 RGB 帧，时间为主机接收时间。
- `capture()` 额外返回可选 uint16 深度、米/计数比例、同帧 Unix 与单调时间戳。
- `read_with_timestamp()` exposes RGB/depth/Unix seconds for direct diagnostic use.
  The camera service uses the shared `read() -> SensorFrame` interface.
- `camera_serial` 绑定设备；`width`、`height`、`fps` 显式选择流。
- `enable_depth`、`align_depth`、`flip`、`exposure_us`、`white_balance` 按配置执行。
- The component defaults to 640×480, 30 Hz and RGB only; component YAML explicitly
  selects capture settings.

默认 `background: true`，组件在后台采集，网络服务的请求直接读取最新缓存，
不会为每台相机依次等待下一帧。重复读取同一缓存保持相同的时间戳和序号。
Set `background: false` to call `capture()` synchronously without a capture worker.

保留首帧等待和 `max_frame_age_sec` 停帧处理；SDK 异常直接交给读取方，
不再像旧网络实现一样在后台自动重启 pipeline，不自动更换设备或分辨率。

配置位于 `manimux/configs/embodiment/sensor/realsense.yaml`。YAM 整机示例中相机序列号在
`manimux/configs/local/yam_example.yaml`；旧相机服务的 `device_id` 字段也在服务入口转换。
这两个配置入口调用同一个组件，不保留第二套 RealSense pipeline 实现。

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
