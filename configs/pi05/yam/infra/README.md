# Pi05 YAM runtime configs

Task-specific runtime configurations are grouped by task:

- `assemble-screwdriver/`: screwdriver assembly checkpoints and inference methods.
- `pick-red-ball-box/`: red-ball checkpoint and inference methods.
- `put-bottles/`: bottle-placement joint-only and joint+EE checkpoints.

Generic reusable baselines such as `manimux.yaml`, `rtc.yaml`, `aac.yaml`, and
`act-temporal-ensemble.yaml` remain in this directory.

The `joint-ee` Pi05 model uses EE motion as an auxiliary training target. Its
deployed robot action remains the first 14 joint/gripper dimensions.

Step-30000 joint+EE server:

```bash
XPolicyLab/policy/Pi_05/openpi/.venv/bin/python \
  scripts/servers/pi05_yam_server.py \
  --config configs/pi05/yam/server/put-bottles/joint-ee-step30000.yaml
```

Choose exactly one matching runtime configuration:

```text
put-bottles/manimux-joint-ee-step30000.yaml
put-bottles/serial-joint-ee-step30000.yaml
put-bottles/rtc-joint-ee-step30000.yaml
```
