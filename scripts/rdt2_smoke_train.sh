#!/usr/bin/env bash
# Single-GPU smoke train for RDT2-FM action expert on manimux UMI data.
#
# Goal is NOT a usable policy: it is to prove the pipeline runs end to end on
# this box and that the loss actually descends. Real training goes remote.
#
# Notes on why the settings differ from upstream scripts/finetune_rdt.sh:
#   * upstream --train_batch_size=64 is PER DEVICE (rdt/main.py:50) and assumes
#     a multi-GPU DeepSpeed node. Here it is one RTX 5090.
#   * upstream never passes --pretrained_model_name_or_path, so it trains the
#     action expert FROM SCRATCH. We point it at ckpt/RDT2-FM to actually
#     fine-tune (rdt/train.py:210 -> RDTRunner.from_pretrained).
#   * DeepSpeed is dropped: ZeRO-1 shards optimizer state across ranks and there
#     is only one rank, so it buys nothing and adds a failure surface.
#   * --sample_period=-1: log_sample_res runs full flow-matching sampling and
#     costs extra VRAM we do not need for a descent check.
#   * dataloader workers kept low: rdt/dataset.py:33 buffers 8192 raw samples
#     per worker (~57 KB each => ~470 MB/worker) and this box has 30 GB RAM.
#   * AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct") is hardcoded
#     at rdt/train.py:127, hence HF_HUB_OFFLINE + the local hub cache.
set -euo pipefail

RDT2_DIR="${RDT2_DIR:-/home/jw/Desktop/project/RDT2}"
VENV="${VENV:-$RDT2_DIR/.venv}"
WDS_CONFIG="${WDS_CONFIG:-/home/jw/Desktop/dataset/exchange_ball_v0_rdt2/dataset.yaml}"
RUN_ROOT="${RUN_ROOT:-/home/jw/Desktop/dataset/rdt2_smoke}"

BATCH="${BATCH:-2}"
STEPS="${STEPS:-200}"
WORKERS="${WORKERS:-2}"
LR="${LR:-1e-4}"
SEED="${SEED:-0}"

RUN_NAME="${RUN_NAME:-bs${BATCH}_steps${STEPS}}"
OUT_DIR="$RUN_ROOT/$RUN_NAME"
LOG_DIR="$OUT_DIR/logs"  # rdt/train.py:67 joins this onto --output_dir
mkdir -p "$OUT_DIR" "$LOG_DIR"

[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV -- build the env first" >&2; exit 1; }
[ -f "$WDS_CONFIG" ]      || { echo "no dataset config at $WDS_CONFIG" >&2; exit 1; }

# Offline: every weight is already on disk. Never let this reach the network.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$RDT2_DIR"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
# 30 GB box: keep the allocator from hoarding, and make OOM messages useful.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$RDT2_DIR"

echo "=== rdt2 smoke train ==="
echo "  venv    : $VENV"
echo "  data    : $WDS_CONFIG"
echo "  out     : $OUT_DIR"
echo "  batch=$BATCH steps=$STEPS workers=$WORKERS lr=$LR seed=$SEED"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
free -g | sed -n 2p

"$VENV/bin/accelerate" launch \
    --num_processes=1 \
    --mixed_precision=bf16 \
    rdt/main.py \
    --config_path="./configs/rdt/post_train.yaml" \
    --pretrained_model_name_or_path="$RDT2_DIR/ckpt/RDT2-FM" \
    --pretrained_vision_language_model_name_or_path="$RDT2_DIR/ckpt/RDT2-VQ" \
    --webdataset_config="$WDS_CONFIG" \
    --output_dir="$OUT_DIR" \
    --logging_dir="logs" \
    --report_to=tensorboard \
    --train_batch_size="$BATCH" \
    --sample_batch_size="$BATCH" \
    --max_train_steps="$STEPS" \
    --checkpointing_period=1000000 \
    --sample_period=-1 \
    --lr_scheduler=constant \
    --learning_rate="$LR" \
    --mixed_precision=bf16 \
    --dataloader_num_workers="$WORKERS" \
    --seed="$SEED" \
    2>&1 | tee "$OUT_DIR/train.log"

echo
echo "=== peak VRAM / loss trend ==="
"$VENV/bin/python" /home/jw/Desktop/project/manimux/scripts/rdt2_read_loss.py "$LOG_DIR"
