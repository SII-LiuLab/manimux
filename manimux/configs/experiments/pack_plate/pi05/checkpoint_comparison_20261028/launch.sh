#!/usr/bin/env bash
# One component per terminal. check does not start services or open hardware.
set -euo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$study_dir/../../../../../.." && pwd)"
cd -- "$repo_dir"
hardware_pythonpath="$repo_dir/manimux/embodiments/end_effector/taccap/sdk/TacCap-Gripper/python"
default_config="$study_dir/pi05_umi_dagger_40000_serial32.yaml"

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
    printf '%s\n' 'Step DiffIK: QP pi/2 rad/s, 80 rad/s^2, elbow guard 1 deg; executor pi rad/s, 160 rad/s^2.'
    printf '%s\n' 'RoboGUI chunk DiffIK methods: RTC 16 / None | Blend 2 | Waypoint (0.4 m/s, 1.2 rad/s).'
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
