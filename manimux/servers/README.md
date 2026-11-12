# Service entry points

Launch from the repository root with the interpreter from the matching hardware or model environment.

| Module | Responsibility |
| --- | --- |
| `manimux.servers.camera.server` | Open configured cameras and serve images over ZMQ |
| `manimux.servers.pi05` | Read Pi05 configuration and start the XPolicyLab service |
| `manimux.servers.groot` / `xr1` / `sapolicy` / `abc_dit` / `isaac05` | Existing XPolicyLab launch/configuration entry points for each model |
| `manimux.servers.openwam` / `umi_dp` | Start XPolicyLab, including existing checkpoint-binding workflows |

Model networks, weight loading, normalization and sampling remain in `XPolicyLab/policy/`; these launchers do not duplicate model implementations.
Camera clients and runtime sensors live in `manimux/embodiments/sensor/camera_server/`;
reading network images does not create physical cameras.

```bash
envs/yam/.venv/bin/python -m manimux.servers.camera.server \
  --config manimux/configs/embodiment/sensor/cameras/realsense_3_views.yaml

XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -m manimux.servers.pi05 \
  --config manimux/configs/policy/pi05/yam/put-bottles/joint-step30000.yaml
```

Model entry points use the repository's XPolicyLab submodule and their own environments. Moving launch files does not install models,
download checkpoints or initialize submodules. `--help` only displays arguments and does not start services.
