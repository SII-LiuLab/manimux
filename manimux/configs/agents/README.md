# Agent recipes

An experiment becomes an RPent experiment by adding `agent: {config: <recipe>}`; RPent
then plans subtasks and calls the experiment's own policy as a tool. A recipe supplies the
planner, memory and the motion profile; the experiment keeps robot, cameras, policy and
recording.

| Recipe | Experiment | Motion profile |
| --- | --- | --- |
| [`rpent.yaml`](rpent.yaml) | `example-rpent-synthetic` (no hardware) | [`rp_motion_demo.yaml`](rp_motion_demo.yaml), synthetic bounds |
| [`rpent_pi05_yam.yaml`](rpent_pi05_yam.yaml) | `example-yam-pi05/put_bottles/rpent.yaml` | private `../local/rpent_yam_motion.yaml` |
| [`rpent_dream1_yam.yaml`](rpent_dream1_yam.yaml) | `example-yam-dream1/put_bottles/rpent.yaml` | private `../local/rpent_yam_motion.yaml` |
| [`rpent_franka.yaml`](rpent_franka.yaml) | `example-franka/pick_yellow_block/rpent.yaml`, via `manimux-rpent serve`/`run` | private `../local/rpent_franka_motion.yaml` |

A motion profile holds station-measured workspace bounds and speed limits for the motion
tools, so it is private. Start from the templates:
[`rpent_yam_motion.example.yaml`](rpent_yam_motion.example.yaml) and
[`rpent_franka_motion.example.yaml`](rpent_franka_motion.example.yaml); their null bounds
fail on purpose until you fill them in.

The RPent experiment must use the Serial bridge contract (Serial, first step when ready,
no blending, chunk IK); see the
[RPent integration reference](../../../skills/manimux-development/references/rpent.md)
and [agent integration](../../../guideline.html#agent-integrations) in the user guide.
