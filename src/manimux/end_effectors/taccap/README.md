# TacCap compatibility imports

The implementation, geometry and official SDK source now live in
[`embodiments/end_effector/taccap`](../../embodiments/end_effector/taccap/README.md).
The implementation file is named `end_effector.py`; the old `gripper` module
forwards to that same module, preserving existing Python imports.

The camera implementation lives in `embodiments/sensor/taccap/`. Gripper and
camera both use the installed `xense.taccap` SDK; its source is stored once under
`embodiments/end_effector/taccap/sdk/TacCap-Gripper/`.
