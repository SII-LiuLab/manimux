# TacCap sensor server

This service owns cameras and publishes RGB frames with their host capture/receipt
timestamps. Device drivers live in `manimux.embodiments.sensor`; this package
contains the shared ZMQ transport and runtime client, not robot control.

The implementation lives under `server/sensor/taccap/` and accepts TacCap
cameras only. Other device families must use their own service packages.

| File | Responsibility |
| --- | --- |
| `server.py` | Resolve camera bindings, own cameras, serve REP requests and PUB frames |
| `client.py` | `CameraClient` for requests, `CameraSubscriber` for latest-frame subscription |
| `driver.py` | Generic runtime source for a selected camera bundle |
| `__main__.py` | `python -m manimux.server.sensor.taccap` entry point |

From the repository root, use the environment with the TacCap SDK installed. On
the current Tianji station that environment is `xense-taccap`:

```bash
/home/jw/miniforge3/envs/xense-taccap/bin/python -m manimux.server.sensor.taccap \
  --experiment configs/experiments/pass_ball/tianji_taccap_umi_dp.yaml \
  --local .local/tianji_taccap.yaml
```

The installed `manimux-camera-server` command points to the same implementation.
After changing the entry point, reinstall the project in an existing environment
before using that console command; the module command works directly with the
updated source. `--config` may point to a TacCap-only camera YAML.

The local binding selects the endpoints. Defaults are REP `127.0.0.1:5555`
(`ping` / `obs`) and PUB `127.0.0.1:5556` (30 Hz). The existing pickled dictionary
wire format remains `{ok, frames, timestamps}`; frames are RGB uint8 HWC and
timestamps are host wall-clock seconds, not hardware exposure timestamps.
Use this protocol only between trusted local/network peers.

Endpoints are bound before cameras are opened. A duplicate service exits without
opening or resetting cameras. Stop the old camera owner before changing services;
do not open the same device through `TacCapSensor` simultaneously.

UMI DP uses `TimestampedCameraSensor` to preserve these capture timestamps and
convert them to runtime monotonic time. The generic `CameraServerSensorDriver`
retains its existing receipt-time semantics. Neither path commands a gripper or arm.
