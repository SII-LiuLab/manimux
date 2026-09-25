# Camera service recipes

Name camera combinations by device family and total view count. Robot names belong
to the assembly that references the cameras, not to the camera recipe filename.

## Component recipes

Reference these through an experiment's `camera_server.config`. Stream names map to
robot components; device serials and service addresses come from the local station.

- `realsense_3_views.yaml`: left wrist, front and right wrist RealSense streams.
- `taccap_2_views.yaml`: left and right TacCap wrist streams.

For example, an experiment under `experiments/put_bottles/pi05/` uses:

```yaml
camera_server:
  config: ../../../embodiment/sensor/cameras/realsense_3_views.yaml
```

Start the camera service with `--experiment <experiment.yaml>` so the launcher can
resolve the robot components and station. `sensors.options.camera_names` selects
the runtime's input streams; `policy.adapter.camera_map` maps those streams to model
inputs. These remain explicit in the experiment.

## Standalone recipes

Files ending in `_standalone.yaml` retain the existing direct device settings and
are launched with `--config <camera-recipe.yaml>`. Their `sensors.cameras` structure
is different from the component recipes above; do not use them as
`camera_server.config` references.

- `realsense_3_views_standalone.yaml`: three RealSense RGB cameras.
- `realsense_3_views_rgb_standalone.yaml`: the same views with explicit frame-age bounds.
- `realsense_3_views_rgbd_standalone.yaml`: the historical recipe with depth defaults preserved.
- `realsense_gemini305_3_views_standalone.yaml`: two RealSense wrists and one Gemini 305.
- `realsense_gemini335_3_views_standalone.yaml`: two RealSense wrists and one Gemini 335.
- `gemini305_gemini335_2_views_standalone.yaml`: the two external Gemini views.
- `taccap_2_views_standalone.yaml`: two TacCap wrist cameras.

The SAPolicy five-camera setup still uses two services: the three RealSense views
plus the two external Gemini views. Renaming recipes does not combine those services
or change their endpoints, serials, resolution, frame rate or depth settings.
