# Reference images and Top overlay

Reference images help restore a scene between trials. They are optional unless
your study template requires them. They do not replace the model's task prompt.
See the [research workflow](../../../guideline.html#experiments) for free/study modes.

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

In RoboGUI, select **Layout** under **New rollout**, then **Prepare Experiment**.
Click **Prepare Run** for a free trial without layout or scoring requirements.
The experiment's layout/repeat rules come from `run.experiment_template`.
The task prompt selects its gallery by case-insensitive matching, treating spaces,
punctuation and underscores equivalently. Store images under the matching task name;
the UI has no second task/reference selector or attach checkbox.

The same selector drives the Top preview and the recorded layout ID. **Reference
appearance** contains optional opacity/tint controls. A camera with the `top` slot
is required for the overlay, and its live view follows the runtime camera mapping.
After adding new gallery IDs, reload the GUI to refresh the available choices.

Prepare records provisional layout/repeat metadata. Layout stays editable through
Warmup until first Start, which freezes the final layout, suggested repeat and PNG
SHA-256. Pause/Resume keeps the selection locked; opacity remains adjustable.

The next repeat is the first unused number among finalized human-scored trials in
this method's output directory, matching task, experiment name, condition, layout
and image hash. Both task success and failure count. Invalid, skipped, unfinished
and unstarted attempts remain recorded without consuming a repeat. Use separate
output directories for changed methods/checkpoints/rubrics. Notes and failure
annotations belong to the post-rollout evaluation, not initial setup.

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
name matches the task instruction as described above; the evaluation rubric remains separate.
