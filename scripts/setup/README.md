# Station setup helpers

| Script | Use |
| --- | --- |
| `check_can_arm_sdks.py` | Verify installed PiPER / X5 SDK bindings offline (below) |
| `dls_gate_calibrate.py` | Derive the YAM DLS `singularity_gate` from this station's recordings ([DLS gate](../../skills/manimux-development/references/runtime-config.md)) |
| `export_chunktrust_backend.py` | Write a standalone XPolicyLab recipe for a ChunkTrust backend ([ChunkTrust](../../skills/manimux-inference/references/chunktrust.md)) |
| `franka-linux-cp312.txt` | Pinned FR3 Franky/libfranka SDK artifact |

Each script prints its options with `--help`; none of them opens a device.

## Offline SDK installation checks

`franka-linux-cp312.txt` pins the tested FR3 Franky/libfranka SDK artifact and
pylibfranka version. It is specific to Linux x86_64 / CPython 3.12; the component
README and public user guide describe the firmware and control contract.

`check_can_arm_sdks.py` verifies the official PiPER and X5 bindings used by the
ManiMux controllers. Install the selected SDK following the
[installation guide](../../skills/manimux-station-setup/references/robot-sdks.md), then run from the repository root:

```bash
envs/aloha/.venv/bin/python scripts/setup/check_can_arm_sdks.py --sdk piper
envs/aloha/.venv/bin/python scripts/setup/check_can_arm_sdks.py --sdk arx
envs/aloha/.venv/bin/python scripts/setup/check_can_arm_sdks.py --sdk both
```

The default is `both`. Missing SDKs or API/FK mismatches fail explicitly.
PiPER checks inspect driver classes and pure MDH FK. X5 checks inspect
`InterfacesPy` methods and instantiate only the offline `KinematicSolver`.
All three packaged robot assemblies are loaded without hardware controllers.

No device session is created. Passing establishes imports, API availability
and offline geometry; it does not establish CAN readiness or hardware behavior.
Use `--output /path/to/report.json` to keep JSON separate from native SDK log
messages. The parent output directory must already exist. Keep reports and
temporary hardware mocks in the ignored local development workspace.
