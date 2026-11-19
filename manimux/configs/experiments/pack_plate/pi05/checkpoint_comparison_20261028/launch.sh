#!/usr/bin/env bash
# One component per terminal. check does not start services or open hardware.
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$study_dir/../../../../../.." && pwd)"
cd -- "$repo_dir"
hardware_pythonpath="$repo_dir/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python"
default_config="$study_dir/pi05_processed2_native_world_h50_29999_serial32.yaml"

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
    printf '%s\n' 'Choose Policy and Method / Config in RoboGUI; inference and executor settings follow the selected configuration.'
    exec env PYTHONPATH="$hardware_pythonpath" .venv/bin/python \
      -m manimux serve --config-list "$study_dir/catalog.yaml"
    ;;
  check)
    env PYTHONPATH="$repo_dir:$hardware_pythonpath" .venv/bin/python "$study_dir/check.py" "${@:2}"
    ;;
  *)
    printf 'Usage: bash %s {check [--policy LABEL]|camera|gui|runtime}\n' "$0" >&2
    exit 2
    ;;
esac
