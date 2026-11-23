# SmolVLA offline integration

The [synthetic experiment](../../../../manimux/configs/experiments/example-smolvla-offline/serial.yaml)
runs a six-coordinate, three-camera [SmolVLA base checkpoint](https://huggingface.co/lerobot/smolvla_base)
through the ManiMux serial runtime, JointAdapter and Direct executor. An in-memory
robot and three generated RGB streams provide the inputs and command sink. The
policy worker uses a supervisor for the isolated model process. The example uses
in-process synthetic cameras and disables RoboGUI, so the runtime command owns the
full run.

## Prepare the model environment

Use a ManiMux Python 3.11–3.12 environment for the runtime. From the repository
root, install the same checkout and LeRobot in a separate model environment:

```bash
uv venv --python 3.11 envs/smolvla
uv pip install --python envs/smolvla/bin/python -e . 'lerobot[smolvla]==0.4.4'
mkdir -p .local/smolvla
cp -n manimux/configs/experiments/example-smolvla-offline/station_example.yaml .local/smolvla/station.yaml
```

Set the checkpoint directory, model environment root and response directory in the
copied station file. Supply the full local checkpoint with the saved preprocessor
and postprocessor. For network-disconnected use, cache the checkpoint's VLM,
processor and tokenizer assets in the model environment before enabling
`HF_HUB_OFFLINE=1`. The example selects CPU. Choose an available accelerator in
`policy.device` as needed, or adjust the inference timeout and run duration for
slower CPU inference.

To pin a specific checkpoint, set `policy.expected_backend.model.checkpoint_path`
to its resolved directory, typically `${paths.smolvla_checkpoint}`. To pin the
weight contents, use `policy.expected_backend.model.checkpoint_sha256` with the
SHA-256 of `model.safetensors`. The worker hashes weights when this field is
configured. LeRobot saves different training steps under directories named
`pretrained_model`, so checking that name requires a path or hash alongside it.

## Check and run the synthetic experiment

From the repository root in the runtime environment:

```bash
config=manimux/configs/experiments/example-smolvla-offline/serial.yaml
station=.local/smolvla/station.yaml
manimux inspect --config "$config" --local "$station"
envs/smolvla/bin/python -I -m manimux.policies.lerobot.process --check
manimux run --once --config "$config" --local "$station"
```

If you selected a different model environment in the station file, use that
environment's Python for the import check. A successful inference probe has
`accepted` greater than zero in the runtime summary. The rollout records decoded
plans and synthetic commands. Each prediction is written under
`policy.options.response_log_dir` before validation, including rejected chunks.

The example validates software on synthetic inputs. Physical use requires a
checkpoint and calibration for the chosen embodiment and cameras. The standalone
runtime starts the configured interpreter automatically. Configurations that
omit `policy.options.model_python` use a policy worker in the selected runtime
environment, where LeRobot must already be installed.
