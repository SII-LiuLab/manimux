"""Real-time chunking: the mask, the scheduler, and the no-regression guarantee.

RTC only works if the action the robot executes is exactly the one the next
chunk was conditioned on, so these tests pin the alignment and the resume index
as hard as the mask formula itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from manimux.config import ManiMuxConfig, load_config
from manimux.policies.capabilities import PolicyCapabilities
from manimux.runtime import build_runtime
from manimux.runtime.edge import EdgeRuntime
from manimux.runtime.rtc import (
    RtcInferenceRequest,
    RtcRuntime,
    inpainting_condition,
    soft_mask,
)
from manimux.types import InferenceRequest, ObservationSnapshot, RobotState

# --------------------------------------------------------------------------- mask


def test_soft_mask_matches_equation_5() -> None:
    horizon, executed, delay = 30, 15, 4
    weights = soft_mask(horizon, executed, delay)

    assert weights.shape == (horizon,)
    # i < d: frozen, those steps will have elapsed before the new chunk lands.
    np.testing.assert_array_equal(weights[:delay], 1.0)
    # i >= H - s: beyond the old chunk, generated freely.
    np.testing.assert_array_equal(weights[horizon - executed :], 0.0)

    overlap = horizon - executed
    denominator = overlap - delay + 1
    for i in range(delay, overlap):
        c_i = (overlap - i) / denominator
        assert weights[i] == pytest.approx(c_i * np.expm1(c_i) / np.expm1(1.0))

    # Monotonically non-increasing: continuity matters most near the seam.
    assert np.all(np.diff(weights) <= 1e-9)


def test_soft_mask_rejects_infeasible_schedules() -> None:
    # The paper's feasibility constraint is d <= s <= H - d.
    with pytest.raises(ValueError, match="delay"):
        soft_mask(30, 15, 16)  # d > s
    with pytest.raises(ValueError, match="delay"):
        soft_mask(30, 27, 4)  # s > H - d


def test_inpainting_condition_left_shifts_the_unexecuted_tail() -> None:
    horizon, dim, executed, delay = 30, 14, 12, 3
    chunk = np.arange(horizon * dim, dtype=np.float32).reshape(horizon, dim)

    targets, weights = inpainting_condition(chunk, executed_steps=executed, delay_steps=delay)

    assert targets.shape == chunk.shape
    # targets[0] must denote the same controller step as the new chunk's index 0.
    np.testing.assert_array_equal(targets[: horizon - executed], chunk[executed:])
    np.testing.assert_array_equal(targets[horizon - executed :], 0.0)
    np.testing.assert_array_equal(weights, soft_mask(horizon, executed, delay))


def test_rtc_request_passes_the_core_worker_contract() -> None:
    """The condition rides a subclass so manimux.types stays untouched."""
    import pickle

    snapshot = ObservationSnapshot(
        state=RobotState(groups={"left_arm": np.zeros(7)}, monotonic_ns=1, sequence=1)
    )
    request = RtcInferenceRequest(
        session_id="s",
        request_seq=1,
        observation_time_ns=0,
        deadline_ns=1,
        observation=snapshot,
        instruction="task",
        action_condition=np.zeros((30, 14)),
        condition_weights=np.ones(30),
        rtc_beta=5.0,
    )

    # PolicyWorkerClient gates on isinstance and ships requests through a queue.
    assert isinstance(request, InferenceRequest)
    restored = pickle.loads(pickle.dumps(request))
    assert restored.action_condition.shape == (30, 14)
    assert restored.rtc_beta == 5.0


def test_policy_plugins_only_send_a_condition_when_one_is_present() -> None:
    import json_numpy

    from manimux.policies import build_policy_model

    config = load_config("configs/abc/yam/infra/manimux.yaml")
    model = build_policy_model(config.policy)
    model._session_id = "s"

    frame_shape = (4, 5, 3)
    snapshot = ObservationSnapshot(
        state=RobotState(
            groups={"left_arm": np.zeros(7), "right_arm": np.zeros(7)},
            monotonic_ns=1,
            sequence=1,
        ),
        frames={
            name: __import__("manimux.types", fromlist=["SensorFrame"]).SensorFrame(
                name=name,
                data=np.zeros(frame_shape, dtype=np.uint8),
                capture_monotonic_ns=1,
                sequence=1,
            )
            for name in ("left_camera", "front_camera", "right_camera")
        },
    )

    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        text = json_numpy.dumps({"actions": np.zeros((30, 14), dtype=np.float32)})

    def _post(url: str, **kwargs: object) -> _Response:
        captured["payload"] = json_numpy.loads(kwargs["data"])
        return _Response()

    import sys
    import types as pytypes

    stub = pytypes.ModuleType("requests")
    stub.post = _post  # type: ignore[attr-defined]
    original = sys.modules.get("requests")
    sys.modules["requests"] = stub
    try:
        plain = InferenceRequest(
            session_id="s",
            request_seq=1,
            observation_time_ns=0,
            deadline_ns=2**62,
            observation=snapshot,
            instruction="task",
        )
        model.infer(plain)
        assert "action_condition" not in captured["payload"]

        model.infer(
            RtcInferenceRequest(
                session_id="s",
                request_seq=2,
                observation_time_ns=0,
                deadline_ns=2**62,
                observation=snapshot,
                instruction="task",
                action_condition=np.zeros((30, 14)),
                condition_weights=soft_mask(30, 15, 4),
                rtc_beta=5.0,
            )
        )
        payload = captured["payload"]
        assert payload["action_condition"].shape == (30, 14)
        assert payload["action_condition_weights"].shape == (30,)
        assert payload["rtc_beta"] == 5.0
    finally:
        if original is None:
            del sys.modules["requests"]
        else:
            sys.modules["requests"] = original


# ----------------------------------------------------------------- runtime wiring


def test_default_runtime_is_unchanged() -> None:
    """execution.runtime defaults to the original runtime, for every config."""
    for path in sorted(Path("configs").glob("*-yam*.yaml")):
        if "-rtc" in path.name:
            continue  # these opt in on purpose
        config = load_config(path)
        assert config.execution.runtime == "manimux", path
        assert type(build_runtime(config, Path("/tmp"))) is EdgeRuntime, path


def test_rtc_runtime_is_selected_by_config(tmp_path: Path) -> None:
    config = load_config("configs/abc/yam/infra/manimux.yaml")
    config.execution.runtime = "rtc"
    runtime = build_runtime(config, tmp_path)

    assert isinstance(runtime, RtcRuntime)
    # Execution stays the default runtime's: same timeline, same executor.
    assert runtime._executor is not None
    assert type(runtime._timeline).__name__ == "ActionTimeline"
    assert RtcRuntime.run is EdgeRuntime.run


def test_rtc_capability_is_checked_before_robot_connection(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    config.execution.runtime = "rtc"
    runtime = build_runtime(config, tmp_path)

    class _DefaultOnlyWorker:
        capabilities = PolicyCapabilities()

    runtime._worker = _DefaultOnlyWorker()
    with pytest.raises(RuntimeError, match="does not advertise"):
        runtime._validate_policy_capabilities()


def test_policy_backend_identity_accepts_matching_metadata(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    payload = config.model_dump(mode="python")
    payload["policy"]["expected_backend"] = {
        "server": "xpolicylab_policy_server",
        "model": {
            "checkpoint_variant": "pi05-step-1000",
            "task_name": "pick_red_ball",
        },
    }
    runtime = build_runtime(ManiMuxConfig.model_validate(payload), tmp_path)

    class _MatchingWorker:
        capabilities = PolicyCapabilities(
            backend_metadata={
                "server": "xpolicylab_policy_server",
                "server_revision": "allowed-extra-field",
                "model": {
                    "checkpoint_variant": "pi05-step-1000",
                    "task_name": "pick_red_ball",
                    "model_root": "/checkpoints/pi05-step-1000",
                },
            }
        )

    runtime._worker = _MatchingWorker()
    runtime._validate_policy_capabilities()


def test_policy_backend_identity_rejects_a_different_checkpoint(tmp_path: Path) -> None:
    config = load_config("configs/mock.yaml")
    payload = config.model_dump(mode="python")
    payload["policy"]["expected_backend"] = {
        "model": {
            "checkpoint_variant": "pi05-step-1000",
            "task_name": "pick_red_ball",
        }
    }
    runtime = build_runtime(ManiMuxConfig.model_validate(payload), tmp_path)

    class _WrongWorker:
        capabilities = PolicyCapabilities(
            backend_metadata={
                "model": {
                    "checkpoint_variant": "pi05-base",
                    "task_name": "pick_place",
                }
            }
        )

    runtime._worker = _WrongWorker()
    with pytest.raises(
        RuntimeError,
        match=r"backend\.model\.checkpoint_variant expected 'pi05-step-1000'",
    ):
        runtime._validate_policy_capabilities()


def test_runtime_package_binds_to_factories_not_to_a_policy_or_a_body() -> None:
    """A runtime is a scheduling strategy: it must work for any policy and body.

    Depending on ``manimux.robots``/``manimux.policies`` is the point — those are
    the factories a runtime builds through. Naming a *specific* integration or a
    *specific* embodiment is the violation.
    """
    import manimux.runtime as runtime_pkg

    root = Path(runtime_pkg.__file__).resolve().parent
    # Factories and protocol types are the intended coupling; a concrete
    # implementation module is not.
    allowed = {
        "manimux.robots",
        "manimux.robots.base",
        "manimux.sensors",
        "manimux.sensors.base",
        "manimux.policies",
        "manimux.policies.base",
        "manimux.policies.worker",
        "manimux.kinematics",  # Factory used by the optional measured-pose release guard.
    }
    import ast

    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module is None or not module.startswith("manimux."):
                continue
            if module.startswith(
                (
                    "manimux.runtime",
                    "manimux.types",
                    "manimux.config",
                    "manimux.clock",
                    "manimux.plugins",
                    "manimux.recording",
                    "manimux.viewer",
                )
            ):
                continue
            assert module in allowed, f"{path.name} imports {module}, which is not a factory"


def test_execution_horizon_respects_the_feasibility_window(tmp_path: Path) -> None:
    config = load_config("configs/abc/yam/infra/manimux.yaml")
    config.execution.runtime = "rtc"
    config.execution.rtc.min_execute_steps = 15
    runtime = build_runtime(config, tmp_path)

    for delay in range(0, 13):
        s = runtime._execution_horizon(30, delay)
        assert delay <= s <= 30 - delay, (delay, s)
    # A large forecast squeezes s down rather than violating s <= H - d.
    assert runtime._execution_horizon(30, 14) == 15

    # Beyond d > H/2 the window d <= s <= H-d is empty; the runtime must refuse
    # to submit rather than build a mask that violates the paper's constraint.
    assert runtime._execution_horizon(30, 16) < 16


def _run_mock(runtime_kind: str, tmp_path: Path, max_steps: int = 120):
    """Drive a full session on mock robot/sensor/policy — no hardware."""
    import json

    config = load_config("configs/mock.yaml")
    config.run.max_steps = max_steps
    config.execution.runtime = runtime_kind  # type: ignore[assignment]
    config.viewer.enabled = False
    if runtime_kind == "rtc":
        config.execution.rtc.min_execute_steps = 8
        config.execution.rtc.initial_delay_steps = 2
    result = build_runtime(config, tmp_path).run()
    events = [
        json.loads(line)
        for line in (Path(result.episode_dir) / "events.jsonl").read_text().splitlines()
    ]
    return result, events


def _of(events: list[dict], kind: str) -> list[dict]:
    return [event for event in events if event.get("kind") == kind]


def test_rtc_executes_chunks_exactly_like_the_default_runtime(tmp_path: Path) -> None:
    """RTC changes when to infer and what to condition on -- nothing else.

    The paper is about generation. Execution (timeline, executor, limits) is the
    robot's business and must behave identically on both runtimes, or the two
    move the same arm differently for reasons invisible in the config.
    """
    import zarr

    def run(kind: str) -> tuple[object, np.ndarray]:
        config = load_config("configs/mock.yaml")
        config.run.max_steps = 200
        config.execution.runtime = kind  # type: ignore[assignment]
        config.viewer.enabled = False
        if kind == "rtc":
            config.execution.rtc.min_execute_steps = 8
            config.execution.rtc.initial_delay_steps = 2
        result = build_runtime(config, tmp_path / kind).run()
        ticks = zarr.open(str(Path(result.episode_dir) / "data.zarr"), mode="r")["ticks"]
        names = sorted(ticks["command"].array_keys())
        return result, np.concatenate([ticks[f"command/{n}"][:] for n in names], axis=1)

    _, default_cmd = run("manimux")
    _, rtc_cmd = run("rtc")

    dt_s = 1.0 / load_config("configs/mock.yaml").robot.control_hz

    def profile(commands: np.ndarray) -> tuple[float, float]:
        velocity = np.abs(np.diff(commands, axis=0)) / dt_s
        accel = np.abs(np.diff(velocity, axis=0)) / dt_s
        return float(np.percentile(velocity, 99)), float(np.percentile(accel, 99))

    default_v, default_a = profile(default_cmd)
    rtc_v, rtc_a = profile(rtc_cmd)
    # Same executor, same limits, so the profiles stay the same order of
    # magnitude. RTC re-plans more often and every commit blends, so its
    # acceleration tail is legitimately heavier -- but a runtime that shaped
    # chunks on its own would sit far outside this band.
    assert 0.4 <= rtc_v / max(default_v, 1e-9) <= 2.5, (default_v, rtc_v)
    assert 0.4 <= rtc_a / max(default_a, 1e-9) <= 3.0, (default_a, rtc_a)


def test_rtc_only_conditions_when_the_schedule_is_feasible(tmp_path: Path) -> None:
    """A mask needs d <= s <= H-d. Outside that window RTC still has to ask for
    a chunk -- unconditioned -- rather than stall or build an invalid mask."""
    _, events = _run_mock("rtc", tmp_path)

    submissions = _of(events, "inference_submitted")
    assert submissions
    assert not submissions[0]["conditioned"], "the first chunk has nothing to condition on"
    assert any(event["conditioned"] for event in submissions[1:]), "guidance never engaged"


def test_replanning_does_not_yank_the_command_back_to_the_measurement(
    tmp_path: Path,
) -> None:
    """``ActionTimeline.commit`` blends the new plan out of ``current_command``.

    Seeding that with the measured state pulls the plan back by the servo's
    tracking error on every commit. RTC re-plans several times more often than
    the default runtime, so what is a rare nudge there becomes a stutter here.
    The seed has to be the last command actually sent.
    """
    import zarr

    def profile(kind: str) -> tuple[int, float]:
        config = load_config("configs/mock.yaml")
        config.run.max_steps = 400
        config.execution.runtime = kind  # type: ignore[assignment]
        config.viewer.enabled = False
        if kind == "rtc":
            config.execution.rtc.min_execute_steps = 8
            config.execution.rtc.initial_delay_steps = 2
        result = build_runtime(config, tmp_path / kind).run()
        ticks = zarr.open(str(Path(result.episode_dir) / "data.zarr"), mode="r")["ticks"]
        names = sorted(ticks["command"].array_keys())
        commands = np.concatenate([ticks[f"command/{n}"][:] for n in names], axis=1)
        plan_ids = ticks["plan_id"][:]
        commits = sum(1 for i in range(1, len(plan_ids)) if plan_ids[i] != plan_ids[i - 1])
        return commits, float(np.abs(np.diff(commands, n=2, axis=0)).max())

    default_commits, default_worst = profile("manimux")
    rtc_commits, rtc_worst = profile("rtc")

    assert rtc_commits >= default_commits, "RTC is supposed to re-plan more often"
    # More commits must not mean rougher motion; that is the whole point of
    # conditioning the next chunk on the current one.
    assert rtc_worst <= default_worst * 1.5, (default_worst, rtc_worst)


# ------------------------------------------------- regression: the H that RTC uses


def _run_slow_policy(tmp_path: Path, inference_delay_s: float = 0.4) -> list[dict]:
    """A rollout where inference costs a large fraction of one chunk.

    ``configs/mock.yaml`` infers in 40 ms against a 20 x 50 ms = 1 s chunk, so
    ``timeline.commit`` trims almost nothing and every indexing mistake stays
    invisible. Real policies cost 170-600 ms, which trims a third of the chunk.
    """
    import json

    config = load_config("configs/mock.yaml")
    config.run.max_steps = 300
    config.execution.runtime = "rtc"
    config.viewer.enabled = False
    config.policy.inference_delay_s = inference_delay_s
    config.policy.timeout_s = 2.0
    config.execution.rtc.min_execute_steps = 8
    config.execution.rtc.initial_delay_steps = 2
    result = build_runtime(config, tmp_path).run()
    return [
        json.loads(line)
        for line in (Path(result.episode_dir) / "events.jsonl").read_text().splitlines()
    ]


def test_the_condition_indexes_the_model_chunk_not_the_trimmed_plan(tmp_path: Path) -> None:
    """Regression: a slow policy used to switch guidance off entirely.

    ``timeline.active_horizon()`` returns the chunk *after* commit-time trimming.
    Using its length as ``H`` shrinks the feasibility window ``d <= s <= H - d``
    until it is empty, so every request after the first went out unconditioned
    and RTC silently degraded into naive async. ``H`` has to be the horizon the
    model produced, and ``s`` has to index that same array.
    """
    events = _run_slow_policy(tmp_path)
    submissions = _of(events, "inference_submitted")

    assert not submissions[0]["conditioned"], "the first chunk has nothing to condition on"
    conditioned = [event for event in submissions[1:] if event["conditioned"]]
    assert conditioned, "guidance never engaged once inference cost a third of a chunk"
    assert not _of(events, "rtc_delay_infeasible"), "the window must not collapse here"

    horizon = load_config("configs/mock.yaml").policy.horizon_steps
    for event in conditioned:
        delay, executed = event["forecast_delay"], event["executed_steps"]
        assert delay <= executed <= horizon - delay, (delay, executed, horizon)
    # Rows skipped at commit still count: ``s`` regularly exceeds what the
    # cursor alone would report right after a chunk lands.
    assert max(event["executed_steps"] for event in conditioned) >= 8


def test_a_conditioned_chunk_commits_without_a_blend(tmp_path: Path) -> None:
    """The blend would overwrite exactly the prefix the guidance froze.

    An unconditioned chunk keeps it: nothing guarantees that one lines up.
    """
    config = load_config("configs/mock.yaml")
    config.run.max_steps = 300
    config.execution.runtime = "rtc"
    config.viewer.enabled = False
    config.policy.inference_delay_s = 0.4
    config.policy.timeout_s = 2.0
    config.execution.blend_steps = 6
    config.execution.rtc.min_execute_steps = 8
    config.execution.rtc.initial_delay_steps = 2
    runtime = build_runtime(config, tmp_path)

    seen: list[int] = []
    original = runtime._timeline.commit

    def spy(chunk, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(int(kwargs["blend_steps"]))
        return original(chunk, **kwargs)

    runtime._timeline.commit = spy  # type: ignore[method-assign]
    runtime.run()

    assert seen, "no chunk was ever committed"
    assert seen[0] == config.execution.blend_steps, "the first chunk is unconditioned"
    assert 0 in seen[1:], "a conditioned chunk still went through the blend"
