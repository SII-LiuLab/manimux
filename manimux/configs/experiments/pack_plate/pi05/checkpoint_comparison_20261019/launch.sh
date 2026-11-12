#!/usr/bin/env bash
# One component per terminal. check does not start services or open hardware.
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$study_dir/../../../../../.." && pwd)"
cd -- "$repo_dir"
hardware_pythonpath="$repo_dir/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python"
default_config="$study_dir/joint_fk_taccap_mix_29999_serial32.yaml"

case "${1:-}" in
  camera)
    exec env PYTHONPATH="$hardware_pythonpath" .venv/bin/python \
      -m manimux.servers.camera.server --experiment "$default_config"
    ;;
  gui)
    exec env PYTHONPATH="$hardware_pythonpath" .venv/bin/python \
      -m manimux.robogui.dashboard --robot tianji
    ;;
  runtime)
    exec env PYTHONPATH="$hardware_pythonpath" .venv/bin/python \
      -m manimux serve --config-list "$study_dir/catalog.yaml"
    ;;
  check)
    env PYTHONPATH="$repo_dir:$hardware_pythonpath" .venv/bin/python "$study_dir/check.py"
    ;;
  *)
    printf 'Usage: bash %s {check|camera|gui|runtime}\n' "$0" >&2
    exit 2
    ;;
esac
