# Isaac 0.5 offline integration guide

This integration uses the [official PerceptronAI Isaac repository](https://github.com/perceptron-ai-inc/isaac)
and [official Isaac-0.5 weights](https://huggingface.co/PerceptronAI/Isaac-0.5). Internally it retains
the official `PerceptronIsaacPolicy`, observation preparation, preprocessor, Flow expert,
`predict_action_chunk`, and postprocessor; XPolicyLab provides only the standard WebSocket contract.

## Current boundaries

The public checkpoint is a **LIBERO Spatial** policy, not a YAM policy:

```text
2 x RGB (image, wrist_image) + 8D LIBERO state + language
  -> official Isaac 0.5 Flow policy
  -> 50 x 7 checkpoint-native absolute EE actions
  -> expose 8 x 7 actions per synchronous XPolicy request at 20 Hz
```

Do not send these `7D` actions directly to YAM. YAM requires a separately fine-tuned checkpoint, matching stats,
and a validated Isaac-to-YAM action adapter. Until these requirements are met, this integration supports only model serving and
offline forward validation, with no `manimux run/serve` hardware configuration.

## Pinned versions

- Isaac repository: `be6507b4aed7472f2029606c22684d4ebc9d73e6`
- Perceptron LeRobot: `e12389c1f8f591ad05dced4e284d4e92e48c5df4`
- XPolicy source: `XPolicyLab/policy/Isaac_05/lerobot`
- Server config: `manimux/configs/policy/isaac05/libero/base.yaml`

The model card requires Perceptron's LeRobot fork for this checkpoint; stock Transformers and
stock LeRobot are incompatible entry points. The official policy also explicitly rejects LeRobot RTC, so this integration declares
only `default` sampling and does not claim RTC support.

## 1. Download the weights

The complete repository is approximately `143 GB`. `hf download` supports resuming:

```bash
cd /home/ubuntu/manimux
mkdir -p checkpoints/pretrained/perceptron-ai/isaac-0.5
hf download PerceptronAI/Isaac-0.5 \
  --local-dir checkpoints/pretrained/perceptron-ai/isaac-0.5
```

On the documented machine, a download was started in the `isaac05-download` tmux session. Check its progress:

```bash
tmux attach -t isaac05-download
tail -f data/download-logs/isaac-0.5.log
du -sh checkpoints/pretrained/perceptron-ai/isaac-0.5
```

Do not start a second download process concurrently.

## 2. Initialize the source and environment

After cloning the parent repository, fetch XPolicyLab and its Isaac source:

```bash
git submodule update --init XPolicyLab
git -C XPolicyLab submodule update --init policy/Isaac_05/lerobot
bash XPolicyLab/policy/Isaac_05/install.sh
```

The installer uses the official lockfile and CUDA extra. The official source also requires production
`trained_policy` to use PyTorch `2.10.0+cu128`, Transformers `5.5.4`,
`torch_sdpa_v1`, and an NVIDIA H100. The lockfile's default PyTorch `2.11` supports dependency installation and
checks without allocating the full model. If the official environment cannot import or render, do not replace model internals with an approximation.

## 3. Static contract check

This command does not load Torch weights, allocate GPU resources, or open a port:

```bash
envs/yam/.venv/bin/python manimux/servers/isaac05.py --check \
  --config manimux/configs/policy/isaac05/libero/base.yaml
```

Until the download completes, it returns `status: blocked` with a nonzero exit code. It returns `status: ready` only when the source revision, configuration, adapter,
stats, weight index, and every shard referenced by the index are present.

## 4. Model service and offline forward pass

This model has approximately `36B` parameters and its public repository is approximately `143 GB`. The official code also rejects production
CUDA runtimes on non-H100 GPUs, so the current RTX 4090 cannot run an official forward pass with this checkpoint. The following
commands should be run by an operator only on a machine meeting the official ABI and hardware requirements above.

Terminal 1:

```bash
bash XPolicyLab/policy/Isaac_05/setup_eval_policy_server.sh \
  manimux/configs/policy/isaac05/libero/base.yaml
```

Terminal 2:

```bash
envs/isaac-0.5/.venv/bin/python scripts/validation/isaac05_forward_probe.py \
  --server ws://127.0.0.1:8504 \
  --instruction "Pick up the object."
```

Acceptance requires a finite `8 x 7` action chunk. The probe uses synthetic LIBERO-shaped inputs,
so it proves only one forward pass through the official model and XPolicy wire protocol, not simulation success or
YAM hardware capability.

## Validation evidence

- Official source revision, checkpoint manifest, deployment adapter, and native stats align in static checks.
- XPolicy observation/action adapter tests for `2 camera + 8D state -> 8 x 7 action` passed.
- Full weight download, GPU model loading, actual forward inference, simulation, and hardware validation remain incomplete.
