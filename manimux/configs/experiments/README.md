# Experiment entry points

Each YAML is one experiment, grouped by task family. Filenames identify the robot, policy
and algorithm/checkpoint variant. Start with `put_bottles/yam_pi05_joint.yaml` or
`pass_ball/tianji_taccap_umi_dp.yaml`.

`pick_red_object/` groups the red-ball, red-block and red-object-to-box recipes.
`put_bottles/` groups bottle-placement recipes, including SAPolicy MV51 camera-view
comparisons. MV51 is a model-training version label, retained in those filenames;
it is not a separate robot task. Each family's recipes retain their exact `run.task`
prompt, checkpoint identity and action settings. A shared directory does not imply
that every checkpoint was trained with the same instruction.

Adapters are selected and parameterized inline. Optional inference/executor presets
are expanded once, relative to the experiment. Model-server recipes are under
`manimux/configs/policy/<model>/`; local bindings select device/service addresses.

Migrated recipes retain their previous execute flags, timing and checkpoint identity.
The local-binding examples can intentionally differ from older checkpoint recipes.
Directory migration is not hardware or task validation.

The runtime uses `manimux.cli.load_config(path, local=...)`. Service launchers can
read `read_experiment(path)["policy_server"]`. The session manifest stores the fully
resolved configuration for reproducibility.
