# PRM-as-a-Judge evaluation

ManiMux tracks the official evaluator as the `PRM-as-a-Judge/` submodule and
keeps model weights outside Git. Initialize it after cloning ManiMux with:

```bash
git submodule update --init PRM-as-a-Judge
```

Use `scripts/evaluation/prm_as_a_judge.py` as the entry point. It forwards the
official CLI unchanged and additionally supports
`PRM_GPU_MEMORY_UTILIZATION`, because upstream currently fixes vLLM's budget
at `0.9`.

```bash
PRM_GPU_MEMORY_UTILIZATION=0.68 \
/home/ubuntu/miniconda3/envs/prm-judge/bin/python \
  scripts/evaluation/prm_as_a_judge.py eval \
  --manifest /path/to/manifest.jsonl \
  --prm dopamine \
  --prm-path checkpoints/pretrained/Robo-Dopamine-GRM-2.0-8B-Preview \
  --gpus 0 \
  --eval-mode forward \
  --frame-interval 72 \
  --batch-size 10 \
  --outlier-method none \
  --smoothing none \
  --visualize
```

The local judge weights are stored at
`checkpoints/pretrained/Robo-Dopamine-GRM-2.0-8B-Preview/` (ignored by Git).
Run the command from the ManiMux repository root. The previous location under
`/home/ubuntu/workspace/Project/PRM-as-a-Judge/PRM/` remains a compatibility
symlink so historical evaluation paths continue to resolve.

For a top-only evaluation, provide only the manifest's required `video` field.
The upstream Dopamine adapter fills all three model camera slots from that one
video. This measures judgment from the requested top view; it is not a
three-view evaluation.

The generated `visualization_report.md` currently calls every successfully
processed record a "Successful case". Task success is instead the `SR` value
in `run_summary.json`/`per_case.jsonl`, or the manifest's human `label` when
metrics are run with `--success-source label`.
