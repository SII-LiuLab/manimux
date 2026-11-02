#!/usr/bin/env bash
set -euo pipefail
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(git -C "$SOURCE_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${MANIMUX_VIDEO_PYTHON:-$REPO_DIR/.venv/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENCV_FFMPEG_CAPTURE_OPTIONS='threads;1'
CPU_ID="$(python3 -c 'import os; print(max(os.sched_getaffinity(0)))')"
UNIT="manimux-video-render-$$.scope"
exec "$PYTHON_BIN" "$SOURCE_DIR/throttle.py" "$UNIT" \
  systemd-run --user --scope --unit="$UNIT" \
  -p CPUQuota=60% -p MemoryMax=1536M -p MemorySwapMax=0 \
  nice -n 15 taskset -c "$CPU_ID" "$PYTHON_BIN" "$SOURCE_DIR/render.py" "$@"
