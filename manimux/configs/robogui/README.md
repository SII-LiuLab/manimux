# RoboGUI display overrides

RoboGUI has two ways to choose what it draws:

| Use | How | Where the file lives |
| --- | --- | --- |
| Normal runs and studies | `--robot <preset>`; experiments name the same preset in `robogui.robot` | [`manimux/robogui/robots/<preset>/robogui.yaml`](../../robogui/robots/README.md) |
| A station-specific view of a preset | `--config <file>` | this folder |

Start with the preset: `python -m manimux.robogui.dashboard --list-robots` lists `yam`,
`tianji`, `franka_fr3_robotiq`, `aloha` and `piper`. Write an override only when this
station needs something the preset does not show, for example other camera panels, a
different table or another Home for display. Copy the preset, edit it here and start the
dashboard with `--config`. Loading a display file never connects hardware.

[`example-yam-manual-cameras.yaml`](example-yam-manual-cameras.yaml) is the published
example: a YAM view with `camera_mode: manual`, which shows the three RealSense
streams as fixed panels instead of following the images the policy consumes.

```bash
python -m manimux.robogui.dashboard --config manimux/configs/robogui/example-yam-manual-cameras.yaml \
  --host 127.0.0.1 --port 8086
```

Only `example-*.yaml` and this README are published; other files in this folder stay local.
Model references (`model:`) resolve beside the file, so keep overrides in this folder or
adjust the relative path.
