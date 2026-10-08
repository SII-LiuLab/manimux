# Integrate with ManiMux

Implement the part your research changes; reuse the runtime, robot assembly and
RoboGUI around it. Start your coding agent with
[the development skill](../../.agents/skills/manimux-development/SKILL.md), which loads
[the repository guide](agent-guide.md).
Humans and agents use the same protocol references below.

| I want to add… | Owning implementation | Guide |
| --- | --- | --- |
| An arm or shared hardware controller | `manimux/embodiments/arm/<vendor>/` | [Arm/controller](components.md#arm-and-shared-controller) |
| A gripper or tool | `manimux/embodiments/end_effector/<name>/` | [End effectors](components.md#gripper-and-other-tools) |
| An assembled robot and its geometry | `manimux/embodiments/robot/`, component YAMLs | [Assembly and FK/IK](components.md#assembly-coordinate-layout-and-geometry) |
| A camera or sensor | `manimux/embodiments/sensor/<name>/` | [Sensors](components.md#cameras-and-runtime-sensors) |
| A checkpoint or learned model | Its owning model framework; deployment recipe | [Model integration](policies.md#a-new-checkpoint-or-learned-model) |
| A peer framework such as RLinf | Independent framework + `manimux/policies/<framework>/` | [PolicyModel client](policies.md#a-backend-framework-client) |
| A different action representation | `manimux/policy_adapter/` | [Action semantics](policies.md#observation-and-action-adapter) |
| A chunk scheduling algorithm | `manimux/runtime/<algorithm>/` | [InferenceStrategy](runtime-config.md#inference-algorithm-versus-executor) |
| A command generator | `manimux/runtime/executors/` | [Executor](runtime-config.md#inference-algorithm-versus-executor) |
| Robot display, replay or experiment UI | `manimux/robogui/`, `manimux/recording/` | [RoboGUI and records](runtime-config.md#robogui-replay-and-recording) |

RLinf is an example of an integration request, not a claim of an existing client.

## What to supply with an integration

1. **Implementation:** the existing interface and its lifecycle, including resource ownership.
2. **Configuration:** component/deployment YAML and a complete experiment selecting it.
3. **Semantics:** input/output shapes, units, group order, frames, action spacing and reset behavior.
4. **Example:** one actual path through the public loader, with required assets/dependencies.
5. **Evidence:** focused offline checks and clearly stated model/hardware validation limits.

A robot-dependent policy adapter must reuse the assembled robot geometry. A camera
must work through the existing camera server. A new client must reuse action adapters,
inference scheduling and executors. If adding one forces vendor-name branches into
the main loop, revisit the ownership before adding those branches.

## A concrete example: replace a camera

Implement `SensorBase.start/read/close` in `manimux/embodiments/sensor/<vendor>/`.
Use `build_camera`'s `implementation: module:Class` selector and accept `name`,
`clock`, `camera_serial` plus documented device options. Return an RGB `SensorFrame`
with the original capture metadata. Declare a component YAML, reference it from
the camera set, and bind the serial in the user's private station file.

The experiment's `sensors[].options.camera_names` selects the stream;
`policy.adapter.camera_map` maps it to the checkpoint's input key. Neither the
policy server nor the main control loop needs a vendor-specific branch. Test the
real camera factory with a fake SDK before attempting hardware startup.

## A concrete example: replace a framework client

A peer framework keeps model loading and sampling in its own process/environment.
Its `build_model(policy_config)` factory returns a `PolicyModel` implementing
`reset(session_id)`, `infer(request)`, `capabilities()` and `close()`.

```yaml
# Fragment in a complete experiment; demo_framework is your installed package.
policy:
  worker: demo_framework.manimux_client:build_model
  action_dt_s: 0.03333333333333333
  horizon_policy_steps: 50
  adapter:
    type: manimux.policy_adapter.joint:JointAdapter
```

The client translates its native response into the action representation consumed
by that adapter. Document and test the client/adapter pair; changing a worker name
alone cannot make incompatible payloads interchangeable. Use StarVLA and XPolicyLab
as concrete transport examples, preserving their independent backend identities.

## A concrete example: add scheduling

Implement `InferenceStrategy` in `manimux/runtime/<name>/strategy.py`; inspect
`build_inference_strategy` for its current selector and constructor. Start from
`DefaultChunkStrategy` or the closest existing strategy, changing only relevant
hooks. Requests go through `build_submission`; decoded chunks go through
`prepare_chunk`, `commit_settings` and the shared `ActionTimeline`.

Use a deterministic clock to exercise delayed responses, overlap and reset. If the
algorithm needs a specialized sampler, implement that in the owning model framework
and declare its capability. Observation mapping and command generation retain their
existing owners. See [runtime details](runtime-config.md).

## Contributing

Submit a focused pull request with the implementation, selecting configuration,
runnable example and evidence described above. State remaining hardware or checkpoint
requirements. Keep model dependencies in their framework, local device bindings in the
private station file, and research settings in YAML.

Write new comments and general documentation in English; update both homepage languages
when changing user-facing behavior. Do not commit recordings, weights, credentials or
hardware addresses. Regression tests currently stay in the ignored local `tests/`
workspace; report the commands and results in the PR instead of force-adding tests.

### Local checks

Run checks on the owning code paths. These commands work from the repository root
without a Makefile:

```bash
uv run ruff format --check manimux
uv run ruff check manimux
uv run mypy manimux
```

To apply formatting and lint fixes, use `uv run ruff format manimux` and
`uv run ruff check --fix manimux`. If your local test workspace is installed, run
`uv run pytest tests/unit` or `uv run pytest tests/integration` as relevant; include
those test paths in Ruff checks when changing them. See the
[environment guide](../usage/environments.md#offline-regression-tests) for hardware-specific
interpreters. Documentation changes use `mkdocs build --strict -f docs/mkdocs.yml` in the
[documentation environment](https://github.com/SII-LiuLab/manimux/blob/main/docs/README.md#preview-locally).
