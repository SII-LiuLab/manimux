# Reference images and Top overlay

Reference images help restore a scene between trials. They are optional unless
your study template requires them. They do not replace the model's task prompt.
See the [research workflow](research.md) for free/study modes.

## Save references

With your camera service already running, start the camera-only reference tool:

```bash
envs/yam/.venv/bin/python -m manimux.robogui.reference_capture \
  --task put_bottles_into_the_bin --camera d405_front --port 8087
```

Open `http://127.0.0.1:8087`. Create or select a task, enter a **Reference ID** such
as `01`, `near` or `cluttered`, arrange the scene and save. The same ID replaces
its existing PNG. Names accept letters, digits, underscores and hyphens. There is
no fixed number of slots. Only images already saved appear in the RoboGUI selector.

This tool subscribes to the existing camera service; it does not create a robot
or policy runtime. `--camera-endpoint` selects the PUB address, `--camera` selects
the stream, and `--root` selects the gallery. Use an unused port if offline replay
is already using 8087. The tool requires a fresh frame before saving.

## Use an overlay

In RoboGUI, expand **Reference layout · Top**, select the task and image, and use
**Show reference overlay**, **Reference opacity** and **White tint** to compare it
with the current Top image. Refresh the reference library after saving new images.
The RoboGUI needs a camera configured with the `top` slot to display this overlay.
Its live image follows the runtime's camera mapping.

For a study rollout, enable **Attach selected reference image** in Research details.
The image ID supplies an empty layout ID, or must match the selected template ID.
Prepare freezes task/layout/repeat metadata and the SHA-256 of the selected PNG.
Selection stays locked for the active attempt; display opacity remains adjustable.
A free rollout can still use the overlay visually without attaching a study identity.

A changed reference file is reported on restoration rather than silently displayed
as the old image. The hash identifies bytes; it is not an image backup. Keep images
used in formal studies unchanged. Different image sizes are not silently resized
for scene alignment; restore the capture setup or capture a matching reference.

## Gallery layout

```text
data/evaluation_layouts/
└── put_bottles_into_the_bin/
    ├── 01.png
    ├── near.png
    └── cluttered.png
```

Files live under the repository-level `data/` when launched from its root.
Use the same custom directory with capture `--root` and RoboGUI `--reference-root`.
Legacy `01.png` through `10.png` galleries remain usable. The selected gallery
name is independent of the policy instruction and optional evaluation rubric.
