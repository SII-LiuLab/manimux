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
| LingBot-VLA2 | Existing | Withdrawn (2026-10-18; latency budget failure) | Added flow hook | Added repeated independent inference | Added action attention hook |
| DP U-Net DDPM | Existing runtime paths | Added VP guidance adaptation | Not applicable | Added batched candidates | Not applicable |
| OpenWAM joint flow | Existing runtime paths | Added joint-flow guidance | Added joint-flow inversion adaptation | Added independent seeds | Added MoT attention hook |

ACT here means the existing temporal-ensemble execution algorithm, not the ACT
policy architecture. Pi05 PyTorch and Pi05 EEF are outside this matrix. Other
embodiments need their own action conversion validation.

The loaded server's `HELLO.capabilities.sampling_modes` is authoritative.
OpenWAM specialized flow modes require CFG scale 1 and a single-stream, full-horizon
backend. RTC allows either compile setting and can opt into `dit_cache_enabled: true`
with `dit_cache_modes: [default, rtc]`. Compiled RTC has synthetic CPU regression
coverage; real-checkpoint CUDA correctness and latency still require a separate probe.
PAINT/AutoHorizon retain the current `compile_enabled: false` support boundary.
PAINT/AutoHorizon requests always disable reuse; unsupported cache modes fail. AutoHorizon additionally requires
the joint self-attention MoT driver. LingBot RTC is withdrawn. LingBot requires a complete native horizon
for PAINT/AutoHorizon. LingBot now supports compiled functional attention capture;
its pinned vendor must expose `get_predict_velocity_fn` and the explicit attention return. AutoHorizon
needs at least three denoising evaluations and one action token per waypoint.
Unsupported combinations must fail rather than execute the default sampler.

ABC-DiT's existing `yam_abc_dit_rtc_200k*.yaml` deployments use the generic
`train_time_rtc` strategy and model hook. Fixed runtime prefix 7 / request progress
16 retain the official model prefix 4: only the final four fixed rows condition
the model, while all seven rows are preserved in the returned 30-row trajectory.
ABC no longer advertises `paint`; its separate `rtc` hook remains inference-time
guidance. The mount-compensation adapter transforms the generic action condition
through the same joint/frame mapping as observations and decoded actions.

ManiMux's websocket server wrapper accepts `policy_server.autohorizon` with
`hold_thr`, `run_len` and `max_entropy_q`; standalone server recipes place that
block at their top level. `manimux/policies/xpolicylab/autohorizon.py` binds the existing
selector through a call-local context for Pi05 JAX and the shared XR1/LingBot/OpenWAM
path. XPolicyLab files, third-step attention capture and runtime alignment stay
unchanged. Omitted configuration leaves the original model method in place.
Response metadata reports the effective thresholds. Validate `run_len < H` against
the attention matrix: the pinned selector's full-horizon convolution can collapse
to a scalar. Direct XPolicyLab launchers do not apply the wrapper options. See
[AutoHorizon usage](../../../guideline.html#autohorizon) for defaults and YAML.

## Selecting configurations

Use `scripts/experiments/build_inference_matrix.py` from the repository root.
It reads an existing task, retains its checkpoint, cameras, embodiment,
action interval and executor, then generates selecting YAMLs validated through
`manimux.cli.load_config`. It starts neither services nor robots. It refuses to
overwrite existing outputs. DP's measured-history strategy is preserved.
OpenWAM's entire generated comparison disables compilation and DiT caching.

```bash
PYTHONPATH=. envs/yam/.venv/bin/python scripts/experiments/build_inference_matrix.py \
  --base manimux/configs/experiments/example-yam-pi05/put_bottles/serial.yaml \
  --output-dir data/configs/pi05-inference \
  --execution-steps 16 --paint-delay-steps 4 --async-trigger progress \
  --methods serial async act rtc paint autohorizon
```

For AAC, include `aac` in `--methods` and pass `--aac-stats` explicitly. The
statistics must describe EE increments for the selected embodiment and data;
the existence of `yam_60ep_ee_increment.json` is not evidence that it matches a
new task. AAC requests 20 samples by default in the generated smoke presets;
review memory and latency before fixing the experiment budget.

`--async-trigger progress` retains the complete prediction and uses
`chunk_policy_steps` as the source-progress threshold, including latency-trimmed rows.
`--async-trigger refill` (the compatibility default) caps the source prefix to K and
sets the refill threshold to K action intervals. Choose explicitly; these are different
experiments even when K is equal. Progress currently requires built-in Async with
single-inflight scheduling, not an observation-history strategy plugin.
The generator covers the listed standalone methods; configure BID and ensemble
compositions through the [composition guide](../../../guideline.html#optional-algorithm-composition).

Inputs for the remaining families (station-local experiments; build them like the [example study](../../../manimux/configs/lists/example-yam-pi05.yaml)):

- XR1: `manimux/configs/experiments/{xr1-study}/put_bottles/manimux_step30000.yaml`.
  Supply `--policy-server manimux/configs/policy/xiaomi-xr1/yam/{recipe}.yaml`
  when the base only describes the client. H=30; example K=12, d=4.
- LingBot: `manimux/configs/experiments/{lingbot-study}/<task>/<sampling-variant>.yaml`.
  The former sampling example binds an existing screwdriver checkpoint and disables robot
  execution. It does not restore the retired task into the experiment register.
  H=50; example K=16, d=4.
- DP: `manimux/configs/experiments/{dp-study}/put_bottles/serial_eef_step100000.yaml`.
  H=6; example K=3. PAINT and AutoHorizon are reported as not generated.
- OpenWAM: `manimux/configs/experiments/{openwam-study}/put_bottles/serial_step30000.yaml`.
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
the transformed condition with the observation. `third_party/XPolicyLab/policy/<model>/`
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
  sequence. `third_party/XPolicyLab/utils/flow_sampling.py` implements the flow sequence for
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

cd third_party/XPolicyLab
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


## Shared sampler structure and cache validation

`third_party/XPolicyLab/utils/sampling_contracts.py` owns CPU request checks (shapes, finite
values, weights in [0,1], positive beta). It imports neither Torch nor JAX.
Pi05 asks for writable copies; each model retains its normalization and schedule.
`flow_sampling.rtc_correction` shares only the clean-estimate Torch VJP between
OpenWAM, FastWAM and the generic RTC sampler. Pi05 JAX, Xiaomi and LingBot native
loops remain separate. Do not force them through a common scheduler.

OpenWAM's `dit_cache_modes` is a startup-validated allowlist, default `[default]`.
The engine selects cache per request without swapping its global cache object.
RTC stores corrected action predictions and native video predictions; reuse
skips forward and VJP together. Preserve native similarity and skip rules.
For an all-task example and commands see [RTC cache](../../../guideline.html#openwam-rtc-cache).
Keep uncached parity tests distinct from approximate cached quality/latency tests.

## Compilation audit (2026-10-18)

Scope: all eleven policy-family directories currently present under
`manimux/configs/policy`, plus FastWAM requested by the operator. This is a source
and configuration audit, not a GPU benchmark or a claim about every XPolicyLab
adapter. Missing source or runtime evidence is explicitly retained below.

| Family | Effective configured/source path | Official comparison and limit |
|---|---|---|
| Pi05 | Current JAX deployment uses `nnx_utils.module_jit` in `policy/Pi_05/openpi/src/openpi/policies/policy.py`; not disabled. PyTorch is a separate path. | [Official policy](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/policy.py) also JITs JAX sampling. Do not infer Torch compilation from JAX status. |
| LingBot-VLA2 | Latest300ep main recipe now explicitly sets `use_compile: true`; legacy recipes remain false. Main list has 20 compiled variants; the separate eager AutoHorizon list has five. | [Official deploy](https://github.com/Robbyant/lingbot-vla-v2/blob/main/deploy/lingbot_vla_v2_policy.py): CLI default true, constructor default false. PAINT and AutoHorizon now have cached compiled prefix/predictor entries and explicit third-step attention return; the old eager list is historical. The separate `lingbot-vla2-latest300ep-step30000-compiled-samplers.yaml` selects the new variants. Real-weight timing and numerical diagnostics are required alongside the compile flag. RTC was withdrawn after measured conditioned latency exceeded the real-time budget; finite historical outputs did not establish real-time support. |
| OpenWAM | Latest300ep recipe and experiment base still false; ordinary finetune recipes true. RTC compilation restriction was removed with CPU AOT/Inductor tests; existing eager records/configs were not silently changed. | [Official deploy.yaml](https://github.com/OpenWAM-Official/OpenWAM/blob/main/configs/deploy.yaml) enables compilation and DiT cache. PAINT/AutoHorizon integration still requires eager. Real-checkpoint compiled RTC CUDA remains unverified. |
| Xiaomi Robotics-1 | No compile switch or wrapping in the selected checkpoint loader. | [Official XR1 loader](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/main/xr1/mibot/server/deploy.py) builds, casts and loads the model without `torch.compile`; the adapter follows that path. HF compile capability flags do not enable it. |
| FastWAM | Independent `policy/FastWAM` model, not DP. No bound ManiMux recipe/list. RoboTwin wrapper does not pass `compile_action_infer`; native default false. | [LIBERO config](https://github.com/yuantianyuan01/FastWAM/blob/main/configs/sim_libero.yaml) enables it; [RoboTwin deploy](https://github.com/yuantianyuan01/FastWAM/blob/main/experiments/robotwin/fastwam_policy/deploy_policy.py) does not. Local custom-denoise hooks currently reject compile; needs dedicated integration validation. |
| ABC-DiT | `policy/abc_dit/yam/put-bottles/pretrained-200k.yaml` selects `fast_inference: true`; adapter compiles velocity and captures sampler CUDA graphs. | Local `policy/ABC_DiT/model.py::_enable_fast_inference` is the evidence for this recipe; no new online upstream/default or GPU validation in this audit. |
| SAPolicy | Recipe-specific: tcp_rot20 and teleopMV51 source configs disable backbone/action compilation; caps1010_3task_tcp enables action compilation and disables backbone compilation. tcp_rot20 also has a separate compiled config. | Adapter delegates to checkpoint-bound SpatialAlign source/config; no single official default is established across these source snapshots. Do not infer current flags from a comment about another variant. |
| GR00T-N1.7 | Adapter constructs `Gr00tPolicy`; no inference compile wrapping in that loaded policy path. | Official [benchmark](https://github.com/NVIDIA/Isaac-GR00T/blob/main/scripts/deployment/benchmark_inference.py) explicitly adds action-head compilation as a separate benchmark backend; [policy API](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/policy/gr00t_policy.py) does not automatically enable it. |
| DP | Adapter loads the saved workspace/policy and calls prediction without adding compile. Exact checkpoint policy remains the owner. | The [official U-Net policy](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py) has no automatic compile wrapper; this is not proof that arbitrary checkpoint plugins cannot compile internally. |
| UMI-DP | Adapter instantiates the checkpoint policy with Hydra and calls `predict_action`, without a compile override. | No fresh checkpoint-specific upstream compile-default verification; do not claim an optimization was disabled or that adding a YAML flag would enable it. |
| Isaac-0.5 | Adapter loads `PerceptronIsaacPolicy.from_pretrained`; no explicit torch.compile call in the inspected local integration. | The official [project](https://github.com/perceptron-ai-inc/isaac) delegates inference to a pinned LeRobot/runtime stack. This audit does not establish compilation inside all checkpoint-owned/runtime implementations. |
| StarVLA | Recipes expose no compile setting; launcher delegates to `third_party/StarVLA/deployment/model_server/serve.py`. That checkout is absent on this station. | Cannot verify effective backend compilation here. A configured framework label or HF capability is not runtime evidence. |

Keep the live service separate from the files on disk: edits take effect only
when the operator restarts the model service and runtime. First-call compilation
can exceed normal inference timeouts; warm up each selected sampling branch before
measuring steady state. Do not remove timeout/safety checks to conceal a cold start.
