# Security policy

## Supported versions

ManiMux has not tagged a release yet. Security fixes land on `main`; please reproduce
issues against the latest `main`.

## Reporting a vulnerability

Please do not open a public issue. Report privately through GitHub:
**Security → [Report a vulnerability](https://github.com/SII-LiuLab/manimux/security/advisories/new)**.
Include the affected component, the configuration, steps to reproduce and the impact.
We aim to acknowledge reports within seven days.

## Scope

ManiMux drives physical robots, so robot-safety defects are treated like security issues.
Report privately, for example:

- a path that sends motion commands while `robot.options.execute` is `false`, or bypasses
  command bounds, motion limits or the Pause/Finish holds;
- network services (camera, policy, RoboGUI, RPent dispatch) that accept commands from
  unauthenticated non-loopback clients against their documented defaults;
- credentials or API keys written to logs, recordings or published files.

Out of scope: vulnerabilities in third-party SDKs, model frameworks or submodules (report
them upstream), and unsafe behavior of a trained policy itself.
