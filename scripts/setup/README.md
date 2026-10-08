# Offline SDK installation checks

`check_can_arm_sdks.py` verifies the official PiPER and X5 bindings used by the
ManiMux controllers. Install the selected SDK following the
[installation guide](../../docs/usage/robot-sdks.md), then run from the repository root:

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
