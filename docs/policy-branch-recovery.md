# Policy regression recovery after the kaifeng update

Recovery baseline: XPolicyLab `e78d1bf`, current parent `4f3f9bf` and
XPolicyLab `3fb19ef`. The OpenWAM XPolicy implementation and its YAM
data/training/deployment refactor are retained, not replaced by the old
standalone OpenWAM WebSocket path.

## Restored functionality

- Pi05: restore `pi05_yam_joint_ee`, its 26-D joint/EEF training target and
  14-D joint deployment output. Joint-only `pi05_yam` is unchanged. Training
  and conversion launchers explicitly prioritize the current checkout over
  an editable installation left pointing at another repository.
- LingBot-VLA2: initialize the pinned nested source `187f840`; restore RTC
  for YAM packed observations and relative-action checkpoints. Absolute RTC
  conditions subtract the observation anchor before normalization; native
  relative outputs are re-anchored only by ManiMux, not twice.
- XR1: correct the relocated `scripts/datasets/prepare_xr1_yam_dataset.py`
  entry. Accept nonempty datasets of different sizes; optional
  `XR1_EXPECTED_EPISODES` / `XR1_EXPECTED_FRAMES` assert a specific manifest.
- LingBot/XR1 `gate-train`: preserve the requested formal step/save count
  after the small smoke run instead of forcing 3000/500.
- Restore SAPolicy `image_native_hw` forwarding and the existing Isaac 0.5
  adapter/source registration removed by the same branch synchronization.
- Keep control-executor imports out of request deserialization. Otherwise
  the first RTC request includes cold SciPy/executor imports in the delay
  forecast, potentially disabling guidance for the whole short test run.
  This does not change RTC masks, delay prediction, or executor algorithms.

## Repeatable offline regression

Initialize the recorded submodule revisions on a fresh checkout:

```bash
git submodule update --init --recursive
bash scripts/validation/run_policy_branch_regressions.sh
```

The runner uses the existing `envs/yam` and Pi05 environments, never installs
packages or starts hardware. The OpenWAM data tests additionally need
`pandas>=2,<4` and `pyarrow>=14,<26` in the test environment; they do not need
the complete CUDA model environment. Full OpenWAM training/inference has its
own dependencies; follow [the runbook](openwam-yam-runbook.md), without
upgrading Pi05 or XR1 dependencies in place.

## Validation boundaries

On 2026-09-09, actual local GPU forwards using synthetic RGB/state and the
existing screwdriver step15000 checkpoints produced finite outputs:

| Policy | Executed GPU paths | Output |
| --- | --- | --- |
| Pi05 | ordinary inference | 50 × 14 absolute joints/grippers |
| LingBot-VLA2 | ordinary and guided RTC | 50 × 14 native relative arms / absolute grippers |
| Xiaomi XR1 | ordinary inference | 30 × 60 native EE/gripper deltas |
| OpenWAM | no full-weight GPU forward in this recovery | native recording → training sample, artifact checks and dummy WebSocket/IK boundary tests |

Pi05 joint/EEF transforms and checkpoint-compatible output slicing are tested
with the actual OpenPI code. Launcher tests verify the requested formal steps
survive smoke gating without launching training. These are not full training,
convergence, task success, or real-robot evaluations.

OpenWAM's downloaded foundation weights are not a YAM fine-tuned checkpoint.
Keep `configs/openwam/yam/server/finetune.yaml` as an unbound template until a
YAM checkpoint with matching configuration/statistics exists. Do not relabel
the foundation weights or disable artifact checks to make this template pass.
The refactored preparation, training and checkpoint-binding commands remain in
[openwam-yam-runbook.md](openwam-yam-runbook.md).

No QZ training jobs are started or changed by these fixes. Remote job status
requires a valid QZ login; the local branch update does not update the remote
training checkout automatically. Existing user-staged changes are preserved.
