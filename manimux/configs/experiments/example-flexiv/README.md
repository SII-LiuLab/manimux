# Flexiv deployment examples

These examples use the shared runtime with execution disabled. `flexiv_joint.yaml`
and `flexiv_pose.yaml` select bare seven-axis arms. The `flexiv_omnigripper_*`
variants add a normalized opening coordinate per arm; their pose target is the
flange reference, not a calibrated grasp TCP. The read-only variant uses fake
predictions and must never be enabled for execution.

Bind devices with the matching private station template. See the
[Flexiv assembly guide](../../../embodiments/robot/flexiv/README.md) and
[user guide](../../../../guideline.html#flexiv-omnigripper) for setup and recovery.
