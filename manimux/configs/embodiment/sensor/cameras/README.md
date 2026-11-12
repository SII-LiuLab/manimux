# Camera service recipes

Use one schema for both `camera_server.config` in an experiment and camera-server
`--config`. Keep exactly these four combinations:

- `realsense_3_views.yaml`: `d405_left`, `d405_front`, `d405_right`.
- `gemini_2_views.yaml`: `gemini305`, `gemini335`; uses `services.external_camera`.
- `taccap_2_views.yaml`: `taccap_left`, `taccap_right`.
- `realsense_gemini_5_views.yaml`: all three D405 and both Gemini streams, in one service.

The RealSense recipes describe D405 devices. Each stream references its driver
component YAML and lists all public capture options supported by that ManiMux
wrapper. These are not the complete vendor SDK option lists.

## Names and device binding

```yaml
cameras:
  d405_left:                    # Image name used by clients.
    config: ../realsense.yaml   # Driver and shared capture defaults.
    component: left_camera     # Key in station.robot.components.
    options:
      exposure_us: 10000
```

The station binds `left_camera.camera_serial` to the physical left device. Its
existing component name also appears in the robot assembly; changing a stream name
does not change the device, its mounting or its serial. Gemini components are named
`gemini305` and `gemini335`; TacCap components remain `left_wrist_camera` and
`right_wrist_camera`. Confirm serial-to-position bindings physically, not by USB order.

Capture precedence is component `options`, component `hardware`, stream `options`,
then station component overrides. Keep station files focused on serials and service
addresses; place reusable capture settings here. Experiment camera-server overrides
merge into the selected recipe. Relative component paths resolve beside the YAML
that declares them.

Runtime `sensors[].options.camera_names` selects stream names.
`output_names`, when present, renames selected streams for the runtime.
`policy.adapter.camera_map` maps model input keys to those runtime names. Model keys
such as `cam_head` and `cam_left_wrist` do not change with camera hardware.

## Supported settings

All drivers accept `camera_serial` through station bindings. `name` and `clock` are
provided by the factory, not user capture settings.

Common settings:

- `width`, `height`: requested image dimensions in pixels; default 640 by 480.
- `fps`: requested frames per second; default 30. The device must support the mode.
- `max_frame_age_sec`: maximum cached-frame age; driver default 0.30 seconds.

RealSense additionally supports:

- `enable_depth`: capture depth; default false.
- `align_depth`: align depth to RGB; default false, relevant when depth is enabled.
- `flip`: horizontal flip; default false.
- `exposure_us`: manual exposure in microseconds; driver default null.
- `white_balance`: manual white balance in kelvin; driver default null.
- `warmup_frames`: frames discarded at startup; driver default 0, shared component
  and recipes use 15.
- `read_timeout_ms`: SDK frame wait timeout; default 1200 ms.
- `background`: use background capture; default true.

An explicit exposure or white balance disables its corresponding automatic mode.
`null` leaves that device setting unchanged; it does not restore factory defaults
or enable automatic mode. The supplied D405 recipes explicitly select 10000 us and
4000 K, matching the requested left-camera settings. Change each stream independently.
Do not assume another RealSense model supports the same values or sensor layout.

Orbbec additionally supports `flip` (default false) and `enable_depth` (default
false). The current Gemini UVC wrapper rejects depth capture; exposure and white
balance are not exposed through this wrapper.

TacCap additionally supports `startup_timeout_sec` (default 3 seconds) and
`by_id_root` (default `/dev/v4l/by-id`). `autostart` is internal: the wrapper owns
start/close and supplies it itself. Flip, depth, exposure and white balance are not
supported TacCap configuration options.

Normally adjust resolution, FPS and, for D405, exposure/white balance. Leave frame-age,
warmup, read timeout, background capture and device-discovery paths at the listed
values unless diagnosing acquisition. These are ManiMux capture settings, not
vendor recommendations. Device/firmware defaults for exposure and white balance
remain distinct from current device values and from explicit YAML overrides.

## Launch

Both entry points use `--local`, or the default private
`manimux/configs/local/station.yaml`:

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml

envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --experiment manimux/configs/experiments/put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml
```

Use one of these commands, not both on the same ports/devices. The station template
includes optional Gemini bindings; fill in their physical serials before using a
Gemini recipe. For historical SAPolicy configurations that select a separate
`external_camera` service, start RealSense and Gemini recipes as separate services.
The five-view recipe instead serves all five on `services.camera`; clients must
select that service explicitly. Selecting the recipe alone does not change existing
experiment topology.

The old RGBD standalone recipe was removed. Workflows requiring depth must explicitly
set `enable_depth: true` and `align_depth: true` on the required D405 streams.
Restart the camera service and its consuming runtime together after renaming streams;
existing recordings retain their original names.
