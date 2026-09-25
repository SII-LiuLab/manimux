# Experiment entry points

Each YAML is one experiment, grouped as `<task>/<model>/<experiment>.yaml`.
Task folders contain model folders such as `pi05/`, `sapolicy/`, `dp/` and `umi_dp/`.
Filenames identify the robot, policy
and algorithm/checkpoint variant. Start with `put_bottles/pi05/yam_pi05_rtc_joint_step30000.yaml` or
`pass_ball/umi_dp/tianji_taccap_umi_dp.yaml`.

For a commented complete configuration and launch commands, see
[the Pi05 RTC example](../examples/README.md).

`pick_red_object/` groups the red-ball, red-block and red-object-to-box recipes.
`put_bottles/` groups bottle-placement recipes, including SAPolicy MV51 camera-view
comparisons. MV51 is a model-training version label, retained in those filenames;
it is not a separate robot task. Each family's recipes retain their exact `run.task`
prompt, checkpoint identity and action settings. A shared directory does not imply
that every checkpoint was trained with the same instruction.

Keep sections in this order when present: `control_profile`, `run`, `robot`, `sensors`,
`policy`, `viewer`, `recording`, `inference`, `executor`, `policy_server`, `camera_server`.
Adapters and camera input mappings are selected and parameterized inline. Inference,
executor and camera-server presets are expanded once, relative to the experiment.
Model-server recipes are under `manimux/configs/policy/<model>/`; local bindings select
device/service addresses. Do not copy one experiment's rates, limits or execution flags
into another experiment merely to match its formatting.

Camera recipes use device families and view counts, such as `realsense_3_views.yaml`
and `taccap_2_views.yaml`. Experiments with separate external camera services retain
their existing sensor/service mappings. The Pi05 Gemini 335 experiment is named
`put_bottles/pi05/yam_pi05_rtc_joint_step30000_gemini335.yaml` to match its actual input stream.

For the same task, policy, action representation and algorithm, retain the step-30000
entry when it supersedes a step-15000 entry. The put-bottles Pi05 joint `manimux`, `rtc`
and `serial` experiments now use the step-30000 entries. The step-15000 `paint` entry
remains because it has no step-30000 counterpart. Keep distinct algorithms, camera
combinations and experiments whose only available checkpoint is step 15000.

Migrated recipes retain their previous execute flags, timing and checkpoint identity.
The local-binding examples can intentionally differ from older checkpoint recipes.
Directory migration is not hardware or task validation.

The runtime uses `manimux.cli.load_config(path, local=...)`. Service launchers can
read `read_experiment(path)["policy_server"]`. The session manifest stores the fully
resolved configuration for reproducibility.
