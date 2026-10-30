# Pi05 through XPolicyLab on YAM

[Back to ManiMux](../../../../README.md#quick-start) · [Deployment guide](https://sii-liulab.github.io/manimux/guideline.html#configuration)

Run Pi05 (put-bottles, joint positions) on a configured dual-arm YAM station with three
cameras. The model is served locally by [XPolicyLab](../../../../third_party/XPolicyLab);
ManiMux connects to it over WebSocket. This example study shows how a comparison is
organized: one shared base, one task, one file per method, and a list that selects them.

```text
example-yam-pi05/
├── _base_.yaml               # station, Pi05 client + model recipe, Serial 30 Hz, RoboGUI
└── put_bottles/
    ├── serial.yaml           # base + task, nothing else
    ├── rtc.yaml              # RTC, 100 Hz control
    ├── async10.yaml          # Async, 10 Hz playback, Smooth executor
    └── rpent.yaml            # same model driven by the RPent agent
../../lists/example-yam-pi05.yaml   # selects the four entries, trials and recording root
```

The [shared base](_base_.yaml) is a simple **30 Hz serial baseline** that executes all 50
returned actions before requesting the next chunk. It is a starting example, not a claim of
validated task performance. [`put_bottles/rpent.yaml`](put_bottles/rpent.yaml) ships with
`robot.options.execute: false`; enable actuation in a private variant only after the RPent
offline and Finish checks.

First complete the [station setup](../../../../skills/manimux-station-setup/SKILL.md),
including the YAM SDK, camera bindings and your private
`manimux/configs/local/station.yaml`. Its `paths.checkpoints` is the root below which the
[model recipe](../../policy/pi05/yam/put-bottles/joint-step30000.yaml) selects the
checkpoint and normalization statistics. Install the Pi05 model environment as described in
the XPolicyLab Pi05 policy README (`third_party/XPolicyLab/policy/Pi_05/`).

Start camera, model, RoboGUI and runtime together; Ctrl-C stops them all:

```bash
manimux up --config manimux/configs/experiments/example-yam-pi05/put_bottles/serial.yaml \
  --runtime-python envs/yam/.venv/bin/python \
  --policy-python third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python --open
```

Or start each service in its own terminal:

```bash
CONFIG=manimux/configs/experiments/example-yam-pi05/put_bottles/serial.yaml
LOCAL=manimux/configs/local/station.yaml
envs/yam/.venv/bin/python -m manimux.servers.camera.server --config $CONFIG --local $LOCAL
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.policy \
  --config $CONFIG --local $LOCAL
envs/yam/.venv/bin/python -m manimux.robogui.dashboard --robot yam --host 127.0.0.1 --port 8086
envs/yam/.venv/bin/python -m manimux run --config $CONFIG --local $LOCAL
```

Open **http://127.0.0.1:8086/**, then use **Prepare Run → Activate Warmup → Start
rollout**. Prepare moves the robot to its selected initial pose; warmup previews predictions
without dispatching them. Finish with **Finish & Home**.

Any other entry runs the same way with its own `--config`. For counted, scored trials of all
four, start the list with `--config-list manimux/configs/lists/example-yam-pi05.yaml`
instead of `--config`.
