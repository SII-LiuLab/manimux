#!/usr/bin/env bash
# Shared arithmetic only; native trainers retain their own parameter names.

training_positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
    echo "$1 must be a positive integer, got: $2" >&2; return 2;
  }
}

training_gpu_count() {
  local ids=$1 id seen=, count=0
  [[ "${ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    echo "GPU IDs must be comma-separated nonnegative integers" >&2; return 2;
  }
  local -a devices
  IFS=, read -r -a devices <<< "${ids}"
  for id in "${devices[@]}"; do
    id=$((10#${id}))
    [[ "${seen}" != *",${id},"* ]] || {
      echo "Duplicate GPU ID: ${id}" >&2; return 2;
    }
    seen+="${id},"
    count=$((count + 1))
  done
  # These task recipes describe one training node, without tensor/pipeline parallelism.
  if [[ "${NNODES:-1}" != 1 || "${WORLD_SIZE:-1}" != 1 || "${MLP_WORKER_NUM:-1}" != 1 ]]; then
    echo "These task recipes require a single-node launcher; use the native trainer for multi-node runs" >&2
    return 2
  fi
  printf '%s\n' "${count}"
}

training_default_accum() {
  local micro=$1 count=$2 target=${YAM_EXPECTED_GLOBAL_BATCH_SIZE:-64}
  training_positive_int micro_batch "${micro}" || return
  training_positive_int gpu_count "${count}" || return
  training_positive_int expected_global_batch "${target}" || return
  if (( target % (micro * count) != 0 )); then
    echo "Global batch ${target} is not divisible by micro batch ${micro} × ${count} GPUs" >&2
    return 2
  fi
  printf '%s\n' "$((target / (micro * count)))"
}

training_check_batch() {
  local model=$1 mode=$2 global=$3 count=$4 detail=$5
  local expected=${YAM_EXPECTED_GLOBAL_BATCH_SIZE:-64}
  training_positive_int global_batch "${global}" || return
  training_positive_int expected_global_batch "${expected}" || return
  if (( global % count != 0 )); then
    echo "${model}: global batch ${global} is not divisible by ${count} GPUs" >&2
    return 2
  fi
  printf '[%s] GPUs=%s %s effective_global_batch=%s\n' "${model}" "${count}" "${detail}" "${global}"
  # Small smoke runs may explicitly choose a smaller batch; formal runs are checked.
  if [[ "${mode}" != smoke && "${global}" != "${expected}" ]]; then
    echo "${model}: expected global batch ${expected}, got ${global}; check native batch/accumulation settings" >&2
    return 2
  fi
}

training_plan() {
  local mode=$1
  shift
  if [[ "${mode}" == plan ]]; then
    printf '%s\n' "$@"
    exit 0
  fi
}
