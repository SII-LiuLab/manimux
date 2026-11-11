# MolmoAct2 YAM example

`coin/serial.yaml` is the initial real-robot test configuration; `coin/async.yaml`
uses the same weights and 30-step action horizon with a ten-step progress trigger.
These are standalone trial configs, not a scored study or a claim of task success.

The model predicts absolute dual-arm joints, not end-effector poses. Normalization
and denormalization belong to the model's official HF processor/model API. The
runtime uses the shared JointAdapter, timeline and executor without IK.

Read the [deployment runbook](../../../../skills/manimux-inference/references/deployment/molmoact2-yam.md)
for the checkpoint contract, private station bindings, environment and all four
services. Only Serial/Async are provided; specialized algorithms require their own
validated model hooks and should not be enabled by editing a method name alone.
