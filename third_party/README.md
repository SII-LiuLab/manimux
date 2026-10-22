# External frameworks

Independent upstream projects are pinned Git submodules here. Their algorithms,
dependencies, package names and licenses remain owned by those projects.

| Source | ManiMux integration |
| --- | --- |
| `XPolicyLab/` | Model serving clients in `manimux/policies/xpolicylab/` |
| `StarVLA/` | Model serving clients in `manimux/policies/starvla/` |
| `RPent/` | Harness tools and lifecycle bridge in `manimux/agents/rpent/` |
| `PRM-as-a-Judge/` | Offline evaluation launcher in `scripts/evaluation/` |

Initialize only the projects needed for your deployment, using their full path,
for example `git submodule update --init third_party/RPent`. Install each framework
in its own environment following its upstream instructions. Model and Harness
environments do not acquire robot ownership; ManiMux retains execution and records.

When upgrading an existing station from root-level submodules, preserve dirty and
untracked framework files first. Finish active runs before migrating that checkout.
Reinstall editable packages from the new paths and update private launcher/config
paths; existing virtual environments can retain absolute paths to the old checkout.
Saved experiment records and historical source paths must remain unchanged.

See the [development guide](../.agents/skills/manimux-development/references/agent-guide.md#harness-frameworks)
for ownership boundaries. Do not reorganize an upstream project's internals solely
to match another framework's implementation.
