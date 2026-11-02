# Pi05 through XPolicyLab on Tianji Marvin + TacCap

[Back to ManiMux](../../../../README.md#quick-start) · [Deployment guide](https://sii-liulab.github.io/manimux/guideline.html#configuration)

Run the Pi05 pack-plate checkpoint (wrist cameras only, end-effector poses) on a dual-arm
Tianji Marvin with two TacCap grippers. The model is served locally by
[XPolicyLab](../../../../third_party/XPolicyLab); the adapter converts its per-arm TCP
poses to joints with differential IK. The study has the same shape as
[example-yam-pi05](../example-yam-pi05/README.md): one base, one task, one file per method.

```text
example-tianji-pi05/
├── _base_.yaml        # station, Pi05 client + model recipe, Serial, RoboGUI; actuation off
└── pack_plate/
    ├── serial.yaml    # base + task, nothing else
    ├── async.yaml     # Async, request 0.25 s before the chunk runs out, 2-row blend
    └── rtc.yaml       # RTC, conditioned on the rows still executing
../../tasks/pack_plate.yaml              # prompt "plate" and the success/failure score
../../lists/example-tianji-pi05.yaml     # selects the three entries, trials and recording root
```

**Actuation is off.** The base sets `robot.options.execute: false` and
`end_effector_control: false`: the runtime reads state, runs the policy and IK, shows the
plan in RoboGUI and records it, but sends no command. To move the robot, create a private
variant that sets both to `true`, only after the read-only checks below.

## Before the first run

1. Complete the [station setup](../../../../skills/manimux-station-setup/SKILL.md): the
   Tianji SDK and robot assets (not redistributed), and a private station file copied from
   [`local/tianji_taccap_example.yaml`](../../local/tianji_taccap_example.yaml) with the
   controller IP, gripper and camera serials, and `paths.checkpoints`.
2. Place the checkpoint below `paths.checkpoints`. The
   [model recipe](../../policy/pi05/tianji/pack_plate/wrist-only-step59999.yaml) pins its
   directory name, step and normalization hash; XPolicyLab refuses any other export.
3. Validate without opening devices:

```bash
CONFIG=manimux/configs/experiments/example-tianji-pi05/pack_plate/serial.yaml
LOCAL=manimux/configs/local/station.yaml
envs/tianji/.venv/bin/manimux inspect --config $CONFIG --local $LOCAL
```

4. Stream the real joint and gripper state with nothing enabled, and compare it with the
   digital twin (`--robogui` publishes to a running RoboGUI):

```bash
envs/tianji/.venv/bin/python scripts/validation/tianji_dry_run.py --config $CONFIG --local $LOCAL --seconds 10
```

## Run one method

Start each service in its own terminal:

```bash
envs/tianji/.venv/bin/python -m manimux.servers.camera.server --config $CONFIG --local $LOCAL
third_party/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.policy \
  --config $CONFIG --local $LOCAL
envs/tianji/.venv/bin/python -m manimux.robogui.dashboard --robot tianji --host 127.0.0.1 --port 8086
envs/tianji/.venv/bin/python -m manimux run --config $CONFIG --local $LOCAL
```

Open **http://127.0.0.1:8086/**, then use **Prepare Run → Activate Warmup → Start
rollout** and **Finish & Home**. With actuation off, Prepare and Home do not move the arms.

For counted, scored trials of all three methods, pass
`--config-list manimux/configs/lists/example-tianji-pi05.yaml --local $LOCAL` instead of
`--config $CONFIG --local $LOCAL` to the camera, model and runtime services.

## Make it yours

- Another Home: copy `embodiment/robot/tianji_taccap_pack_plate.yaml`, change `home`, and
  point `robot.config` at the copy. Home is part of the assembly because Tianji's recovery
  and Finish & Home drive to it.
- Another checkpoint: write a recipe next to the published ones in
  `policy/pi05/tianji/pack_plate/` and add its artifact to XPolicyLab's pack-plate contract,
  or serve a different model family (see the
  [deployment runbooks](../../../../skills/manimux-inference/references/deployment/)).
- Keep your own variants and lists private; only `example-*` studies are published.
