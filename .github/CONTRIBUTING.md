# Contributing to ManiMux

Thanks for helping. ManiMux welcomes robot and policy integrations, inference strategies,
RoboGUI work, examples and documentation.

## Before you start

- **Bugs and questions:** open an [issue](https://github.com/SII-LiuLab/manimux/issues/new/choose).
  Include the command, the resolved config (`manimux inspect`) and the log.
- **Larger changes** (a new robot, model framework or algorithm): open an issue first so the
  owning layer can be agreed on. The [development guide](../skills/manimux-development/SKILL.md)
  describes the component protocols, and the
  [contribution checklist](../skills/manimux-development/references/README.md#contributing)
  lists what a pull request should contain.

## Pull requests

1. Fork the repository and branch from `main`.
2. Keep the change focused: implementation, the configuration that selects it, a runnable
   example and evidence. Model dependencies stay in their framework (for example
   XPolicyLab); device bindings stay in the private station file; research settings stay
   in YAML.
3. Run the checks from the repository root:

   ```bash
   uv sync --dev
   uv run ruff format --check manimux
   uv run ruff check manimux
   uv run mypy manimux
   python3 website/build.py        # when guideline.html or website/ changed
   ```

4. Describe what you verified and how (commands and results), and state what still needs
   hardware or a checkpoint. Regression tests live in a local, Git-ignored `tests/`
   workspace; report the test commands you ran instead of adding them to the PR.
5. Do not commit recordings, weights, credentials, hardware addresses or machine-specific
   absolute paths. Use `${paths.checkpoints}`-style placeholders and `--local` station
   files.
6. Write code comments and documentation in English. When user-facing behavior changes,
   update both `README.md` and `README.zh-CN.md`, and the user guide `guideline.html`.

Never start a robot, connect a controller or cause motion as part of an automated check.
Hardware validation is performed by an operator through RoboGUI.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](../LICENSE). Third-party code or assets you add must keep their license and be
recorded in [`licenses/THIRD_PARTY_NOTICES.md`](../licenses/THIRD_PARTY_NOTICES.md).
