# Station setup helpers

| Script | Use |
| --- | --- |
| `check_can_arm_sdks.py` | Verify installed PiPER / X5 SDK bindings offline (below) |
| `dls_gate_calibrate.py` | Derive the YAM DLS `singularity_gate` from this station's recordings ([DLS gate](../../skills/manimux-development/references/runtime-config.md)) |
| `export_chunktrust_backend.py` | Write a standalone XPolicyLab recipe for a ChunkTrust backend ([ChunkTrust](../../skills/manimux-inference/references/chunktrust.md)) |
| `setup_openwam_local_env.sh` | Create the OpenWAM model environment ([OpenWAM runbook](../../skills/manimux-inference/references/deployment/openwam-yam.md)) |
| `franka-linux-cp312.txt` | Pinned FR3 Franky/libfranka SDK artifact |
| `yam_backlash.py` / `yam_backlash.html` | Live joint feedback and paired manual alignment measurements ([user guide](../../guideline.html#yam-backlash)) |
| `yam_leader_backlash.py` / `yam_leader_backlash.html` | Independent motorized YAM leader readout on port 8769 ([user guide](../../guideline.html#yam-leader-backlash)) |
| `yam_backlash_can.py` | Passive SocketCAN feedback and explicit read-status queries for the joint measurement UI |
| `yam_set_home.py` | Preview or zero supported YAM joints at a saved reference or an explicitly selected current fixture pose ([user guide](../../guideline.html#yam-set-home)) |

The command-line helpers print their options with `--help`. The backlash UI requires
an explicit mode: `--mode demo` is disconnected, `--mode passive` subscribes to existing CAN
feedback, and `--mode read` requests status replies only on Refresh or Record. The real
modes do not initialize a controller or transmit motion/enable commands. The SDK
installation checks below do not open devices. `yam_set_home.py` previews an explicit
`--reference` without SDK/CAN access. With `--apply --fixture-supported --report`, every selected joint
must match that historical raw-angle reference before any motor is enabled; each
joint is checked again before zeroing. It does not move to the saved pose.
With `--current-fixture`, fresh stable readings of the installed fixture replace
historical targets for this run; the historical file remains unchanged for audit,
and the same retry ledger remains active. Both modes save all prechecks before
enabling and fsync each actual pre-write reading before issuing zero. The report
preserves actual before/after angles and old Home coordinates; a locked state file
beside the reference blocks automatic retries of already requested zero writes.
Physical fixture support is required: the official SDK sends zero gains/torque
during zeroing and this tool disables each motor afterward. It cannot zero an
unsupported arm while maintaining continuous motor holding. An apply call without
the support declaration is rejected before SDK/CAN or report access.

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
