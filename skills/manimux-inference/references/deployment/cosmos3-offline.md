# Cosmos3 offline inference guide

This guide covers **offline inference with the Cosmos3 DROID policy**. It does not start cameras or CAN, connect to YAM,
or provide a YAM configuration. The public
`nvidia/Cosmos3-{Edge,Nano}-Policy-DROID` checkpoints target single-arm DROID: inputs are three RGB
images, language, and a `7+1`-dimensional state; outputs are `32×8` absolute joint-position chunks.

## Implementation boundaries

```text
XPolicy observation
    -> policy/Cosmos3/model.py        Field renaming only
    -> official RobolabPolicyService
       Official model / transforms / normalization / denoise / postprocess throughout
    -> official H×8 action
    -> policy/Cosmos3/model.py        Split into a 7D arm and a 1D gripper only
```

The official project entry point is `NVIDIA/cosmos`; its DROID inference implementation resides in
`NVIDIA/cosmos-framework`, pinned as a nested submodule within XPolicy to
`c7e8d76b5da8aeae38cdac91c6cfd57185b2f6bc`. We do not copy or modify the official model source.

## 1. Initialize the source

```bash
cd {repo}
git submodule update --init --recursive
```

Verify the revision:

```bash
git -C third_party/XPolicyLab/policy/Cosmos3/cosmos-framework rev-parse HEAD
```

## 2. Install the official environment

```bash
cd {repo}/third_party/XPolicyLab/policy/Cosmos3
bash install.sh
```

The default Python is `{repo}/envs/cosmos3/.venv/bin/python`. The script runs the official
`uv sync --all-extras --group cu130-torch213-train --group policy-server`, then additionally installs
XPolicyLab in editable mode so the adapter and official service can load in the same process.

The pinned CUDA wheels use the CPython 3.13 ABI, so the installer defaults to `COSMOS3_PYTHON=3.13`.
To use the framework's CUDA 12.8 dependency group locally, run:

```bash
COSMOS3_CUDA_GROUP=cu128-train bash install.sh
```

## 3. Run one offline forward pass

This command loads the official Edge DROID checkpoint and runs a real forward pass with zero RGB/state inputs. It checks
only whether the model loads, the output shape is `H×8`, and all values are finite:

```bash
cd {repo}
envs/cosmos3/.venv/bin/python \
  third_party/XPolicyLab/policy/Cosmos3/offline_infer.py \
  --config third_party/XPolicyLab/policy/Cosmos3/deploy.yml \
  --prompt "Pick up the object."
```

Example fields from a successful result:

```json
{
  "status": "ok",
  "checkpoint": "nvidia/Cosmos3-Edge-Policy-DROID",
  "action_shape": [32, 8],
  "finite": true
}
```

This proves one forward pass through the official weights and XPolicy adapter, not task success.

## 4. XPolicy WebSocket offline round trip

After caching the weights, run the XPolicy debug client. It does not enter a simulator or real-world client:

```bash
cd {repo}/third_party/XPolicyLab/policy/Cosmos3
EVAL_ENV_TYPE=debug bash eval.sh \
  Cosmos3 offline edge droid_single joint 0 0 0 uv envs/yam
```

## Configuration

The default `deploy.yml` reproduces NVIDIA's Edge serving recipe in full:

- `checkpoint_path: nvidia/Cosmos3-Edge-Policy-DROID`;
- `format_prompt_as_json: true`;
- `guidance_interval: [960, 1001]`;
- `num_steps: 4`, `action_chunk_size: 32`, `action_dim: 8`;
- `conditioning_fps: 15.0`, `action_space: joint_pos`.

For Nano, use `nvidia/Cosmos3-Nano-Policy-DROID` and follow the official Nano command by setting
`format_prompt_as_json` and `guidance_interval` to `null`.

## Recorded validation status

- ✅ Official source revision, API, and DROID input/output contract checked.
- ✅ XPolicy adapter static checks and tests verifying unchanged numerical behavior passed.
- ⚠️ This integration has not yet installed the full Cosmos environment, downloaded the 4B weights, or run a GPU forward pass.
- ⛔ No cameras, CAN, YAM, or other hardware were connected.
