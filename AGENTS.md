# ManiMux agent entry point

Task workflows live in [`skills/`](skills/). Claude Code (`.claude/skills`) and
Codex (`.agents/skills`) discover them automatically through symlinks; other agents
should read the matching `SKILL.md` before starting.

| Skill | Use for |
|---|---|
| [manimux-deploy](skills/manimux-deploy/SKILL.md) | Deploy a supported model recipe on an installed station, start services, open RoboGUI, RPent execution, hardware-free demo |
| [manimux-inference](skills/manimux-inference/SKILL.md) | Select or verify inference algorithms, runtime timing, experiment YAMLs, per-model launch runbooks (`references/deployment/`) |
| [manimux-experiments](skills/manimux-experiments/SKILL.md) | Generate manual experiment launch commands; build a real-robot study: tasks, scoring, config lists, trials, evaluation |
| [manimux-station-setup](skills/manimux-station-setup/SKILL.md) | Bind a deployment to local robots, cameras, SDK environments and model paths |
| [manimux-development](skills/manimux-development/SKILL.md) | New components, adapters, strategies, execution and RoboGUI features |
| [manimux-demo-video](skills/manimux-demo-video/SKILL.md) | Product demo videos |
| [manimux-wall-capture](skills/manimux-wall-capture/SKILL.md) | Station footage for the video wall (invoke explicitly) |

Rules that apply to every task:

- Do not start or restart a robot, connect its controller or cause motion without the
  user's explicit permission. The operator performs Prepare, Warmup, Start and Finish
  in RoboGUI.
- Before giving launch commands, follow the model's runbook under
  `skills/manimux-inference/references/deployment/` and give every service it lists
  (model, camera, RoboGUI, runtime). Validate with `manimux inspect` and the applicable
  service `--check` modes first. For manual launches, give [one standalone command per
  terminal](skills/manimux-experiments/SKILL.md#manual-launch-commands) with verified
  target-host absolute paths, not a shared shell setup block.
- Keep machine-specific absolute paths out of committed files; use the repository's
  placeholders and `--local` station files.
