# StarVLA validation

This page defines the integration's evidence and limits. Follow the
[runbook](starvla-offline-runbook.md) to reproduce the serving path. All ManiMux
checks described here used synthetic RGB and test robot state, without physical
cameras, CAN or robot motion. A completed runtime test is not task success.

## Verified scope

The following representative checkpoints were exercised on 2026-09-27:

| Checkpoint | Real model output | Verified ManiMux path |
| --- | --- | --- |
| Qwen3-VL-2B OFT, RoboTwin few-shot step 40000 | 50 × 14 absolute joints | Shared server, reset, action permutation, joint adapter and serial runtime |
| Qwen3-VL-4B PI-v3, RoboDojo step 100000 | 50 × 14 absolute joints | Shared server, joint adapter, all eight runtime modes, direct/smooth/MPC |
| Qwen2.5-VL GR00T, LIBERO-4in1 step 30000 | 8 × 7 feedback EEF deltas | Shared server, canonical pose conversion, analytic FK/IK, direct/smooth/MPC |
| Qwen2.5-VL FAST, LIBERO-4in1 step 30000 | 8 × 7 feedback EEF deltas | Same EEF path, plus strict action-token validation |

Transport probes send four requests: initial input, repeated input, input after
reset, and changed RGB. They check backend identity, dimensions and finite
values. OFT repeated/reset predictions were identical. Flow matching samples
fresh noise; reset clears episode state rather than reseeding the model.

PI-v3 passed serial, asynchronous ManiMux, ACT temporal ensemble, RTC, PAINT,
AAC, AutoHorizon and DVAC runtime checks. Each completed 500 control cycles with
accepted plans and zero rejections in the final runs. GR00T/FAST smooth and MPC
runs each completed 500 cycles: GR00T accepted 21 plans, FAST 11, with zero
rejections. Real GR00T and PI-v3 heads also passed native flow sampler probes.
The tested GR00T EEF deployment still exposes default inference only.

## Multi-step EEF contract tests

`tests/integration/test_starvla_eef_runtime.py` uses the actual StarVLA adapter,
shared server, ManiMux worker, YAM geometry, decoders, scheduler, executors and
recorder. Only model computation is replaced with explicitly labeled synthetic
output. No checkpoint or GPU is required.

The fixture has distinct arm states, three cameras, 16D EEF state, 50 predicted
poses and a 20-step decode prefix. It reuses Pi05's YAM timing and control
profiles. Nine tests cover:

- Inline, one-process and parallel-arm decoding.
- Asynchronous, serial and temporal ensemble scheduling; smooth and MPC execution.
- Absolute targets and observation-base/tool delta anchors.
- Pose preservation through FK/IK, recorded values and process cleanup.
- Numerical agreement with `Pi05YamEefAdapter` on identical absolute targets.
- Whole-chunk rejection when one arm's IK fails.

Each of the eight runtime cases completes 300 cycles and requires zero rejected
plans. **No matching real dual-arm EEF checkpoint has been validated.** This
matrix establishes the software contract, not a trained YAM policy.

## Reproduce the checks

From the ManiMux repository root, set
`PYTHONPATH="$PWD:$PWD/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"`.
Use the environment that owns each dependency set.

```bash
# ManiMux environment, including the documented offline YAM geometry dependencies.
python -m pytest tests/unit/test_pose_adapter.py tests/unit/test_starvla_offline.py \
  tests/unit/test_xpolicylab_plugins.py tests/unit/test_pi05_yam_eef_runtime.py \
  XPolicyLab/tests/unit/test_starvla_chunks.py
python -m pytest tests/integration/test_starvla_eef_runtime.py

# StarVLA environment: real small flow heads and adapter reproducibility checks.
python -m pytest XPolicyLab/policy/starVLA/scripts/test_flow_sampling.py \
  XPolicyLab/policy/starVLA/scripts/test_fast_runtime.py \
  XPolicyLab/policy/starVLA/scripts/test_qwen3_attention_backend.py \
  XPolicyLab/policy/starVLA/scripts/test_runtime_reproducibility.py
```

The code-cleanup regression set passed 104 adapter/configuration/transport tests
and 46 model-side tests. The model-side tests ran on CPU; CUDA/autocast warnings
were expected in that run. They are distinct from real-weight GPU validation.
The nine EEF integration tests also passed after cleanup. Separate real-weight
checks passed for OFT with both raw and encoded RGB through `eval.sh`, and for
FAST through the unified probe: four requests plus 500 control cycles, ten
accepted plans and zero rejected plans.

For real weights, use the runbook's unified probe. To inspect native model output
without ManiMux action conversion, run in the StarVLA environment:

```bash
python XPolicyLab/policy/starVLA/scripts/probe_checkpoint.py \
  --checkpoint /path/to/run/checkpoints/steps_30000_pytorch_model.pt \
  --base-vlm /path/to/Qwen2.5-VL-3B-Instruct \
  --unnorm-key franka --camera-count 1 --include-state false \
  --output .local/starvla/groot-native.json
```

Add `--sampling-modes rtc paint aac autohorizon dvac` for a compatible flow head.
For FAST, instead select its action-token VLM and add `--fast-tokenizer`.
Native EEF sampler checks do not enable joint-conditioned EEF sampling in ManiMux.

XPolicyLab's standard OFT debug loop was also checked with raw and encoded RGB
(`DEBUG_OBS_ENCODED=0` and `1`). Existing offline Viewer/replay tests passed except
for an unavailable Tianji SDK scene, which was excluded and is not counted as a
pass. Recorded arrays and RGB videos were reopened to check evidence integrity.
These focused checks are not a full repository test run.

## Known limits and failure behavior

- FAST can generate an invalid number of DCT coefficients for some synthetic
  observations. The strict decoder rejects such output instead of returning zero
  actions. Fixed-gradient inputs exercised the valid path; arbitrary-input
  robustness is not established.
- A response that outlives its usable horizon is rejected. PAINT's initial delay
  must cover its sampler latency. EEF decoder workers can pre-initialize lazy IK
  dependencies with `ik_warmup_joints` before the control clock starts.
- OFT/FAST do not implement the specialized flow modes. EEF sampling conditions
  require a separate model-space conversion and are not exposed.
- LIBERO's analytic test plant does not simulate Franka dynamics or OSC physics.
  Single-step feedback deltas are not accumulated into a predicted multi-step pose path.
- Training reproduction, clean-environment installation, simulator task evaluation
  and physical robot deployment were not validated by the ManiMux checks.

Generated reports, arrays and recordings belong under ignored local directories.
Keep failing reports as well as successful results, and state whether a result
uses real weights or synthetic model output when attaching evidence to a PR.

## Assets

Public assets used for validation are pinned below. Preserve their original
configurations and statistics. OFT few-shot and GR00T LIBERO weights were supplied
locally and are not distributed by this contribution.

| Asset | Source | Revision |
| --- | --- | --- |
| PI-v3 RoboDojo | [StarVLA PI-v3](https://huggingface.co/StarVLA/StarVLA-Qwen3vl4b-PIv3-RoboDojo) | `c119685777cf17d27940b9f36fbc7a83663361e0` |
| FAST LIBERO | [StarVLA FAST](https://huggingface.co/StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1) | `2a30a316ab65ab1a3172082fa7ef4b7121beb447` |
| FAST VLM | [Qwen2.5-VL-3B-Instruct-Action](https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action) | `97163e6190ca87d6abae7a4dd15840cadde2da1d` |
| FAST processor | [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) | `ec4d7aa71691cac0b8bed6942be45684db2110f4` |

Download PI-v3 with the provider's manifest-based tool:

```bash
python XPolicyLab/policy/starVLA/scripts/prepare_hf_checkpoint.py \
  --variant pi_v3 --output-dir checkpoints/pretrained/starvla/pi_v3_robodojo
```

For the other public assets, use `hf download <source> --revision <revision>
--local-dir <destination>` with the source/revision pair above and a destination
under `checkpoints/pretrained/starvla/`. The `hf` CLI is supplied by the model
environment's `huggingface_hub` package.
