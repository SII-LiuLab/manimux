# Changelog

Notable changes to ManiMux. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). No release has been tagged yet;
`pyproject.toml` carries the development version 0.1.0. Pull request numbers refer to
[SII-LiuLab/manimux](https://github.com/SII-LiuLab/manimux/pulls).

## Unreleased

### Added

- Hardware-free `example-synthetic` study covering the full RoboGUI trial loop, a Tianji
  Marvin + TacCap `example-tianji-pi05` study, folder guides for tasks, embodiment, agents,
  RoboGUI overrides, policy recipes and inference presets, and a command reference.
- Community files: contributing guide, code of conduct, security policy, issue and pull
  request templates, this changelog.
- RoboGUI shows the remaining trials for every method of a task (#41).
- Pause latches position holds; RoboGUI shows PAINT condition ranges (#34).
- Shared IK failure policy and DLS / DLS-gate chunk IK for YAM (#33).
- Skills exposed to coding agents through `.claude/skills` and `.agents/skills` (#32).
- Built-in ChunkTrust AHS and gated RTC+AHS inference (#25); Poke API support (#26).
- RPent agent execution and Franka FR3 support (#15).

### Changed

- Checkpoint-specific recipes, station presets and one-off analyses are no longer
  published; stepless example recipes replace them.
- Documentation commands that did not run on a clean clone were fixed; RoboGUI `--demo`
  binds free ports; a missing station file explains how to create one.
- Experiment records UI simplified (#27).

### Fixed

- ChunkTrust AHS handoff timing (#31).
- Device command bounds, silent gripper clips and dropped request fields (#36).

### Removed

- Unreachable RoboGUI collection replay and other dead code (#37); defensive
  self-attribute guards (#38).

## Earlier history

Integrations before October 2026 include StarVLA (#6), ALOHA/PiPER assets and ARX X5/PiPER
SDK adapters (#7), Tianji deployments with differential IK and recovery (#8, #12), request
scheduling with latest-observation streaming (#9) and runtime-driven RoboGUI recovery
controls (#10). See the merged pull requests for details.
