# Model sampling matrix

This integration targets the existing **YAM** deployments: Pi05 JAX joint,
XR1 anchor-relative EEF, LingBot-VLA2 relative joint, DP absolute XYZ/Euler,
and OpenWAM absolute EEF20. A hook and a passing CPU test establish an
implementation path; they do not establish checkpoint latency, GPU memory
capacity, closed-loop stability, or task success.

| Deployment | Serial / Async / ACT ensemble | RTC | PAINT | AAC | AutoHorizon |
|---|---|---|---|---|---|
| Pi05 JAX joint | Existing | Existing | Existing | Existing; full chunk decoding fixed | Existing |
| Xiaomi XR1 | Existing | Existing | Added flow hook | Added independent batched candidates | Added action attention hook |
| LingBot-VLA2 | Existing | Existing | Added flow hook | Added repeated independent inference | Added action attention hook |
| DP U-Net DDPM | Existing runtime paths | Added VP guidance adaptation | Not applicable | Added batched candidates | Not applicable |
| OpenWAM joint flow | Existing runtime paths | Added joint-flow guidance | Added joint-flow inversion adaptation | Added independent seeds | Added MoT attention hook |

ACT here means the existing temporal-ensemble execution algorithm, not the ACT
policy architecture. Pi05 PyTorch and Pi05 EEF are outside this matrix. Other
embodiments need their own action conversion validation.

The loaded server's `HELLO.capabilities.sampling_modes` is authoritative.
OpenWAM specialized flow modes require `compile_enabled: false`,
`dit_cache_enabled: false`, and CFG scale 1; AutoHorizon additionally requires
the joint self-attention MoT driver. LingBot requires a complete native horizon
for RTC/PAINT/AutoHorizon and eager execution for attention capture. AutoHorizon
needs at least three denoising evaluations and one action token per waypoint.
Unsupported combinations must fail rather than execute the default sampler.

## Selecting configurations

Use `scripts/experiments/build_inference_matrix.py` from the repository root.
It reads an existing task, retains its checkpoint, cameras, embodiment,
action interval and executor, then generates selecting YAMLs validated through
`manimux.cli.load_config`. It starts neither services nor robots. It refuses to
overwrite existing outputs. DP's measured-history strategy is preserved.
OpenWAM's entire generated comparison disables compilation and DiT caching.

```bash
PYTHONPATH=. envs/yam/.venv/bin/python scripts/experiments/build_inference_matrix.py \
  --base manimux/configs/experiments/put_bottles/pi05/yam_pi05_serial_joint_step30000.yaml \
  --output-dir data/configs/pi05-inference \
  --execution-steps 16 --paint-delay-steps 4 \
  --methods serial async act rtc paint autohorizon
```

For AAC, include `aac` in `--methods` and pass `--aac-stats` explicitly. The
statistics must describe EE increments for the selected embodiment and data;
the existence of `yam_60ep_ee_increment.json` is not evidence that it matches a
new task. AAC requests 20 samples by default in the generated smoke presets;
review memory and latency before fixing the experiment budget.

Existing inputs for the remaining families:

- XR1: `manimux/configs/experiments/put_bottles/xiaomi-xr1/yam_xiaomi_xr1_manimux_step30000.yaml`.
  Supply `--policy-server manimux/configs/policy/xiaomi-xr1/yam/finetune-put-bottles-step30000.yaml`
  when the base only describes the client. H=30; example K=12, d=4.
- LingBot: `manimux/configs/examples/lingbot_vla2_yam_sampling.yaml`.
  This example binds an existing screwdriver checkpoint and disables robot
  execution. It does not restore the retired task into the experiment register.
  H=50; example K=16, d=4.
- DP: `manimux/configs/experiments/put_bottles/dp/yam_dp_serial_eef_step100000.yaml`.
  H=6; example K=3. PAINT and AutoHorizon are reported as not generated.
- OpenWAM: `manimux/configs/experiments/put_bottles/openwam/yam_openwam_serial_step30000.yaml`.
  Supply `--policy-server manimux/configs/policy/openwam/yam/finetune-put-bottles-step30000.yaml`.
  H=32; example K=12, d=4.

These K/d values are smoke-test examples, not matched study settings. Keep
checkpoint, seed policy, denoising evaluations, action interval, executor,
timeout and scene fixed when estimating an algorithm's effect. PAINT uses three
flow passes, AAC uses multiple samples, and AutoHorizon varies execution length;
equal denoising-step settings do not imply equal compute or equal wall time.
PAINT needs `0 < d <= K` and `K+d <= H`; measured delays can still make a run
infeasible. Do not truncate a checkpoint's action horizon to fit a preset.

## Code ownership and action conversion

`manimux/runtime/` owns request times, overlap masks and execution prefixes.
`policy_adapter/<model>/yam.py` maps canonical absolute joint targets into each
model's physical action representation. `policies/xpolicylab/client.py` transports
the transformed condition with the observation. `XPolicyLab/policy/<model>/`
owns normalization, noise, sampling and output metadata.

- XR1: old absolute joint targets go through FK and are rebased to the new
  observation's EE frame before normalization. PAINT now uses the RTC codec.
- LingBot: the server subtracts the current observed arm joints, then applies
  training normalization and padding; grippers remain absolute.
- OpenWAM: FK produces absolute per-arm EEF20, then the **action** index map is
  applied. Unified proprio and action indices need not be identical.
- DP: FK produces absolute per-arm XYZ/Euler plus gripper; normalized conditions
  are inserted at `n_obs_steps - 1`, the start of the returned action window.
  Its camera/state input uses measured history, not repeated current frames.

AAC scores all candidates in the same physical EE-increment space. Relative
joint/EEF outputs are anchored before scoring; absolute EEF candidates are
scored without IK. It returns the full selected native chunk and
`aac.execution_steps`. The embodiment adapter decodes the full horizon before
the runtime selects the execution prefix. This also preserves fixed-horizon
LingBot, DP and OpenWAM adapters. StarVLA returns the same execution-length key.

## Algorithm provenance and validation limits

- [PAINT Algorithm 1](https://arxiv.org/html/2606.19774v1) supplies the naive
  forward, backward Euler inversion, prefix-noise replacement and final forward
  sequence. `XPolicyLab/utils/flow_sampling.py` implements the flow sequence for
  XR1/LingBot. `policy/OpenWAM/sampling.py` extends it to the checkpoint's joint
  video/action grid; this is a local WAM adaptation, not an official WAM port.
- [RTC](https://arxiv.org/html/2506.07339v2) uses guided denoising and runtime
  overlap weighting. DP's `policy/DP/rtc.py` uses a variance-preserving PiGDM
  approximation, `r_t² = sigma_t²`, with a capped epsilon correction and the
  native diffusion scheduler. Metadata identifies it as `vp_pigdm_capped_v1`.
  It is **not** Pi05's flow-Euler recurrence or a validated official DP port.
  OpenWAM identifies its conditioned sampler as `joint_flow_v1`.
- [AutoHorizon](https://arxiv.org/html/2602.21445v1) captures third-step action
  self-attention. The selector is copied from the previously audited official
  implementation at `c7504f1756109103f2cfcc2e23f1b1a23841c885`; its reverse-pointer
  convention is preserved. New model-specific attention hooks retain the full
  visible-key softmax before extracting the action block. Their usefulness for
  XR1/LingBot/OpenWAM still needs measurement.

Offline checks cover flow invariants, independent candidate seeds, attention
selection, joint/EEF score equivalence, action-reference conversion, preservation
of sampling fields, execution-prefix timing and real config-loader dispatch.
Tests live locally under the repository's test ignore policy. Run from the
correct checkout; the parent has a legacy `client_server` package that can
shadow XPolicyLab tests.

```bash
PYTHONPATH=. envs/yam/.venv/bin/python -m pytest -o addopts='' -q \
  tests/unit/test_inference_expansion.py

cd XPolicyLab
PYTHONPATH=..:policy/Pi_05/openpi/src JAX_PLATFORMS=cpu \
  policy/Pi_05/openpi/.venv/bin/python -m pytest -o addopts='' -q \
  tests/unit/test_flow_sampling_expansion.py tests/unit/test_ws_infer_sampling.py \
  tests/unit/test_lingbot_vla2_xpolicy_adapter.py tests/unit/test_openwam_artifact_identity.py \
  tests/unit/test_pi05_aac_adapter.py tests/unit/test_pi05_paint_adapter.py
```

Before counting a new pair as model-validated, load the intended checkpoint and
verify HELLO, a default request, a specialized request, finite/full-horizon output,
metadata and reset behavior. For guidance, test zero weights against default with
identical noise and nonzero weights against a known prefix. For AAC, check actual
candidate diversity; for AutoHorizon, compare identical-noise actions with default.
Record latency and peak memory. These real-weight checks and robot trials remain
outstanding for the newly added combinations.
