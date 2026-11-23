# MolmoAct2 YAM example

`coin/serial.yaml` is the initial real-robot test configuration; `coin/async.yaml`
uses the same weights and 30-step action horizon with a ten-step progress trigger.
These are standalone trial configs, not a scored study or a claim of task success.

The model predicts absolute dual-arm joints, not end-effector poses. Normalization
and denormalization belong to the model's official HF processor/model API. The
runtime uses the shared JointAdapter, timeline and executor without IK.

Read the [deployment runbook](../../../../skills/manimux-inference/references/deployment/molmoact2-yam.md)
for the checkpoint contract, private station bindings, environment and all four
services. `coin/rtc.yaml`, `coin/paint.yaml`, `coin/autohorizon.yaml` and
`coin/bid.yaml` select the local-source sampler hooks. BID requests 16 independent
candidates with serial selection; AutoHorizon selects a serial execution horizon.
RTC and PAINT use single-inflight conditioning. Keep the baseline checkpoint,
normalization, horizon and control rate fixed when measuring algorithm cost.

These are software integration presets. Qualify measured delay, stale-plan rejection
and terminal holding before physical execution; a model sampler passing does not
establish continuous real-time performance for the one-second prediction horizon.

`coin/rtc_5steps.yaml` and `coin/paint_5steps.yaml` explicitly reduce the solver to
five evaluations per forward pass and set a 14-row timing window/initial delay.
They are separate latency tradeoffs, not changes to the ten-step baseline. The
policy identity includes `num_steps`; launch a matching model service for each.
Action quality with fewer solver steps still requires task evaluation.
