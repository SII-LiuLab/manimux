"""MolmoAct eval launcher.

Runs N rollouts, prompting for an instruction each time. Saves all three
cameras frame-by-frame (PNG) plus the joint trajectory (``episode.h5``) per
rollout, classifies rollouts via cv2 keypress (y/n/q) or a post-timeout
stdin prompt, and converts the session's labeled rollouts to a LeRobot v3.0
dataset on the way out.

CLI::

    uv run --extra molmoact-yam manimux-molmoact-yam \
        --left-config-path <left.yaml> \
        --right-config-path <right.yaml> \
        -n 10
"""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

from manimux.robots.yam import BimanualRobot
from manimux.server.sensor.taccap import CameraClient

from .eval_utils import (
    EvalRolloutSaver,
    LiveCameraView,
    RolloutOutcome,
    convert_session_to_lerobot,
    move_rollout,
    prompt_instruction,
    resolve_label,
)
from .gello_min.env import RobotEnv
from .gello_min.launch_utils import (
    instantiate_from_dict,
    move_to_start_position,
    move_to_zero_home,
)
from .gello_min.logging_utils import log_collect_demos
from .molmoact_client import MolmoAct, MolmoActLocal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"
SMOOTHING_RAD_PER_STEP = 0.02


def _runtime_call(runtime: Any, method: str, **kwargs: Any) -> None:
    """Call an optional rollout observer without affecting robot execution."""
    if runtime is None:
        return
    callback = getattr(runtime, method, None)
    if callback is None:
        return
    try:
        callback(**kwargs)
    except Exception:  # noqa: BLE001 — visualization is strictly best-effort
        logger.exception("Policy runtime hook failed: %s", method)


def _runtime_cameras(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract RGB frames without imposing MolmoAct observation keys on viewers."""
    return {
        key.removesuffix("_rgb"): np.asarray(value)
        for key, value in observation.items()
        if key.endswith("_rgb")
    }


# ---------------------------------------------------------------------------
# atexit parking
# ---------------------------------------------------------------------------

_env: RobotEnv | None = None
_bimanual: bool = False
_left_cfg: dict[str, Any] | None = None
_right_cfg: dict[str, Any] | None = None
_cleanup_done: bool = False


def _return_to_zero_home(env: RobotEnv, bimanual: bool) -> bool:
    """Move to zero home and report whether the safety transition succeeded."""
    print("Returning robot to zero home...")
    try:
        move_to_zero_home(env, bimanual=bimanual, time_interval_s=5.0)
        print("Robot returned to zero home.")
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning("Parking failed: %s", exc)
        return False


def _park_robot() -> None:
    """Return the follower arm(s) to zero-joint home before process exit."""
    global _cleanup_done
    if _cleanup_done or _env is None:
        return
    if _return_to_zero_home(_env, _bimanual):
        _cleanup_done = True


def _handle_termination(signum: int, _frame: Any) -> None:
    """Turn SIGTERM into the same orderly shutdown path as Ctrl-C."""
    print(f"\n[signal] Received {signal.Signals(signum).name}; parking before exit...")
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    left_config_path: str = str(Path(__file__).resolve().parent / "configs/molmoact_yam_left.yaml")
    """Path to the left arm configuration YAML file."""

    right_config_path: str | None = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    num_rollouts: Annotated[int, tyro.conf.arg(aliases=("-n",))] = 1
    """How many rollouts to run in this session."""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _build_env(
    args: Args,
) -> tuple[RobotEnv, dict[str, Any], dict[str, Any] | None, bool]:
    """Build cameras + robot(s) + RobotEnv from the launch configs.

    Cameras must be provided by an external service; this process does not own
    physical camera SDK objects.
    """
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.left_config_path), resolve=True)
    bimanual = args.right_config_path is not None
    right_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
        if bimanual
        else None
    )

    cam_server_cfg = (left_cfg.get("eval") or {}).get("camera_server") or {}
    use_server = bool(cam_server_cfg.get("enabled", False))

    if not use_server:
        raise ValueError("eval.camera_server.enabled must be true")
    camera_dict = None
    endpoint = str(cam_server_cfg.get("endpoint", "tcp://127.0.0.1:5555"))
    timeout_ms = int(cam_server_cfg.get("request_timeout_ms", 500))
    max_age = cam_server_cfg.get("max_frame_age_sec", 0.5)
    max_age = float(max_age) if max_age is not None else None
    print(f"[eval] Using camera server at {endpoint} (timeout={timeout_ms} ms)")
    camera_client = CameraClient(
        endpoint=endpoint,
        request_timeout_ms=timeout_ms,
        max_frame_age_sec=max_age,
    )
    if not camera_client.ping():
        raise RuntimeError(
            f"Camera server at {endpoint} did not respond to ping. "
            "Start the configured camera service first."
        )

    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )
    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )
        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)
    else:
        robot = left_robot

    env = RobotEnv(
        robot,
        control_rate_hz=left_cfg.get("hz", 30),
        camera_dict=camera_dict,
        camera_client=camera_client,
    )
    return env, left_cfg, right_cfg, bimanual


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


def dynamic_smoothing(env: RobotEnv, target_joints: np.ndarray) -> dict[str, Any]:
    """Apply ``target_joints`` via sub-tick linear interpolation. Returns final obs.

    The interpolation sub-steps issue command-only ticks (no camera reads). A
    single ``get_obs()`` at the end produces the obs the caller actually
    consumes — this is what makes the rollout loop run at robot rate instead
    of camera rate.
    """
    curr_joints = env.get_robot_state()["joint_positions"]
    max_delta = float(np.abs(curr_joints - target_joints).max())
    # The previous 0.01-rad spacing turned each policy action into many 30 Hz
    # sub-steps and made a nominal 30-action chunk take several seconds.  A
    # conservative 0.02-rad spacing keeps interpolation while reducing that
    # temporal distortion and allowing the policy to replan sooner.
    steps = max(1, int(np.ceil(max_delta / SMOOTHING_RAD_PER_STEP)))
    if steps <= 1:
        env.step_command_only(target_joints)
    else:
        for jnt in np.linspace(curr_joints, target_joints, steps):
            env.step_command_only(jnt)
            time.sleep(0.001)
    return env.get_obs()


def _infer_actions(
    policy: MolmoAct,
    input_dict: dict[str, Any],
) -> list[Any]:
    """Run one policy request and normalize its action-list contract."""
    t0 = time.perf_counter()
    actions = policy.inference(input_dict)["actions"]
    if len(actions) == 0:
        raise RuntimeError("Policy returned an empty action chunk.")
    log_collect_demos(
        f"Policy inference {time.perf_counter() - t0:.3f}s ({len(actions)} actions)",
        "data_info",
    )
    return actions


def run_one_rollout(
    env: RobotEnv,
    policy: MolmoAct,
    saver: EvalRolloutSaver,
    instruction: str,
    rollout_idx: int,
    num_rollouts: int,
    max_steps: int,
    live_view: LiveCameraView,
    async_inference: bool = False,
    async_lead_steps: int = 6,
    async_blend_steps: int = 6,
    control_rate_hz: float = 30.0,
    runtime: Any = None,
) -> RolloutOutcome:
    """Execute one rollout and buffer per-step observations into ``saver``.

    End conditions:

    * ``cv2`` keypress ``y`` -> success (labeled)
    * ``cv2`` keypress ``n`` -> failure (labeled)
    * ``cv2`` keypress ``q`` -> quit (no label; rollout stays in ``eval/``)
    * step >= ``max_steps`` -> timeout (stdin prompt afterwards)

    Does NOT flush the saver — the caller does that so the Ctrl-C path can
    also flush the partial buffer.
    """
    chunk_size = max(1, int(policy.get_action_horizon()))

    if async_inference:
        return _run_one_rollout_async(
            env=env,
            policy=policy,
            saver=saver,
            instruction=instruction,
            rollout_idx=rollout_idx,
            num_rollouts=num_rollouts,
            max_steps=max_steps,
            live_view=live_view,
            chunk_size=chunk_size,
            lead_steps=async_lead_steps,
            blend_steps=async_blend_steps,
            control_rate_hz=control_rate_hz,
            runtime=runtime,
        )

    action_chunk: list[Any] | None = None
    active_chunk_id = -1

    for step in range(max_steps):
        if action_chunk is None or (step % chunk_size) == 0:
            obs_for_policy = env.get_obs()
            input_dict = policy.prepare_input(obs_for_policy, instruction)
            active_chunk_id += 1
            _runtime_call(
                runtime,
                "inference_submitted",
                step=step,
                chunk_id=active_chunk_id,
                planned_switch_step=step,
            )
            inference_started_at = time.perf_counter()
            action_chunk = _infer_actions(policy, input_dict)
            _runtime_call(
                runtime,
                "plan_activated",
                actions=np.asarray(action_chunk),
                action_index=0,
                chunk_id=active_chunk_id,
                step=step,
                action_dt=1.0 / max(control_rate_hz, 1.0),
                inference_ms=(time.perf_counter() - inference_started_at) * 1000.0,
                instruction=instruction,
                metadata={"mode": "sync"},
            )

        action_index = step % chunk_size
        action = np.asarray(action_chunk[action_index])
        obs_pre = env.get_obs()
        obs_post = dynamic_smoothing(env, action) or obs_pre

        saver.add_step(obs_pre=obs_pre, obs_post=obs_post)
        if runtime is not None:
            _runtime_call(
                runtime,
                "step_executed",
                joint_positions=env.robot().get_joint_state(),
                cameras=_runtime_cameras(obs_pre),
                step=step + 1,
                max_steps=max_steps,
                action_index=action_index + 1,
                chunk_id=active_chunk_id,
                metadata={"mode": "sync"},
            )

        key = live_view.update(
            obs=obs_pre,
            rollout_idx=rollout_idx,
            num_rollouts=num_rollouts,
            step=step + 1,
            max_steps=max_steps,
            instruction=instruction,
        )
        if key == "y":
            return RolloutOutcome(end_reason="success", last_step=step + 1)
        if key == "n":
            return RolloutOutcome(end_reason="failure", last_step=step + 1)
        if key == "q":
            return RolloutOutcome(end_reason="quit", last_step=step + 1)

    return RolloutOutcome(end_reason="timeout", last_step=max_steps)


def _run_one_rollout_async(
    env: RobotEnv,
    policy: MolmoAct,
    saver: EvalRolloutSaver,
    instruction: str,
    rollout_idx: int,
    num_rollouts: int,
    max_steps: int,
    live_view: LiveCameraView,
    chunk_size: int,
    lead_steps: int,
    blend_steps: int,
    control_rate_hz: float,
    runtime: Any = None,
) -> RolloutOutcome:
    """Overlap the next HTTP inference with execution of the current chunk.

    The server returns 30 actions even when the receding-horizon window is
    shorter.  We submit the next observation ``lead_steps`` before the window
    boundary, keep executing unused actions from the current response, then
    switch to the new response.  Actions that became stale while inference was
    running are skipped according to elapsed control time.
    """
    if chunk_size < 2:
        raise ValueError("Async inference requires action_horizon >= 2.")
    lead_steps = max(1, min(int(lead_steps), chunk_size - 1))
    blend_steps = max(1, int(blend_steps))
    control_rate_hz = max(float(control_rate_hz), 1.0)

    first_obs = env.get_obs()
    first_input = policy.prepare_input(first_obs, instruction)
    active_chunk_id = 0
    next_chunk_id = 1
    _runtime_call(
        runtime,
        "inference_submitted",
        step=0,
        chunk_id=active_chunk_id,
        planned_switch_step=0,
    )
    first_inference_started_at = time.perf_counter()
    action_chunk = _infer_actions(policy, first_input)
    action_index = 0
    next_submit_step = chunk_size - lead_steps
    _runtime_call(
        runtime,
        "plan_activated",
        actions=np.asarray(action_chunk),
        action_index=action_index,
        chunk_id=active_chunk_id,
        step=0,
        action_dt=1.0 / control_rate_hz,
        inference_ms=(time.perf_counter() - first_inference_started_at) * 1000.0,
        instruction=instruction,
        metadata={"mode": "async", "initial": True},
    )

    pending: concurrent.futures.Future[list[Any]] | None = None
    pending_started_at = 0.0
    pending_submit_step = 0
    pending_switch_step = 0
    pending_chunk_id: int | None = None

    def activate_pending(
        step: int,
        current_state: np.ndarray,
        *,
        wait: bool,
    ) -> bool:
        nonlocal action_chunk, action_index, pending, next_submit_step
        nonlocal active_chunk_id, pending_chunk_id
        if pending is None or (not wait and not pending.done()):
            return False
        new_chunk = np.asarray(pending.result(), dtype=np.float64).copy()
        elapsed = max(0.0, time.perf_counter() - pending_started_at)
        wall_clock_actions = int(round(elapsed * control_rate_hz))
        # Align the new prediction with policy actions actually executed since
        # its observation was captured.  Wall-clock time also includes the
        # interpolation sub-steps in dynamic_smoothing; treating those as
        # consumed policy actions advances semantic events (such as gripper
        # release) too far into the new chunk.
        executed_actions = max(0, step - pending_submit_step)
        # Always retain at least one complete receding-horizon window.  Without
        # this cap, a slow interpolated move can skip almost the entire chunk
        # and force a synchronous refill a few steps later.
        max_stale_actions = max(0, len(new_chunk) - chunk_size)
        stale_actions = min(
            executed_actions,
            max_stale_actions,
        )

        # Absolute joint targets from independently predicted chunks need not
        # meet at the boundary.  Anchor the first retained action to the
        # measured state and decay that offset over several actions so position
        # and velocity do not jump when the background result becomes active.
        current_state = np.asarray(current_state, dtype=np.float64)
        stitch_offset = current_state - new_chunk[stale_actions]
        stitch_count = min(blend_steps, len(new_chunk) - stale_actions)
        for blend_idx in range(stitch_count):
            alpha = blend_idx / float(blend_steps)
            new_chunk[stale_actions + blend_idx] += (1.0 - alpha) * stitch_offset

        action_chunk = new_chunk
        action_index = stale_actions
        if pending_chunk_id is None:
            raise RuntimeError("pending inference is missing its chunk id")
        active_chunk_id = pending_chunk_id
        pending_chunk_id = None
        pending = None
        next_submit_step = step + chunk_size - lead_steps
        log_collect_demos(
            f"Async chunk switch at step {step + 1}: "
            f"skipped {stale_actions}/{executed_actions} executed actions "
            f"(wall-clock estimate={wall_clock_actions}); "
            f"stitched {stitch_count} actions "
            f"(offset_max={np.max(np.abs(stitch_offset)):.3f} rad)",
            "data_info",
        )
        _runtime_call(
            runtime,
            "plan_activated",
            actions=action_chunk,
            action_index=action_index,
            chunk_id=active_chunk_id,
            step=step,
            action_dt=1.0 / control_rate_hz,
            inference_ms=elapsed * 1000.0,
            instruction=instruction,
            metadata={
                "mode": "async",
                "executed_actions": executed_actions,
                "skipped_actions": stale_actions,
                "wall_clock_actions": wall_clock_actions,
                "stitched_actions": stitch_count,
                "stitch_offset_max": float(np.max(np.abs(stitch_offset))),
            },
        )
        return True

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="molmoact-inference",
    ) as executor:
        for step in range(max_steps):
            obs_pre = first_obs if step == 0 else env.get_obs()

            if pending is not None and step >= pending_switch_step:
                activate_pending(
                    step,
                    obs_pre["joint_positions"],
                    wait=False,
                )

            if pending is None and step >= next_submit_step:
                input_dict = policy.prepare_input(obs_pre, instruction)
                pending_started_at = time.perf_counter()
                pending_submit_step = step
                pending_chunk_id = next_chunk_id
                next_chunk_id += 1
                pending = executor.submit(_infer_actions, policy, input_dict)
                pending_switch_step = step + lead_steps
                _runtime_call(
                    runtime,
                    "inference_submitted",
                    step=step,
                    chunk_id=pending_chunk_id,
                    planned_switch_step=pending_switch_step,
                )
                log_collect_demos(
                    f"Async inference submitted at step {step + 1}; "
                    f"planned switch at step {pending_switch_step + 1}",
                    "data_info",
                )

            if action_index >= len(action_chunk):  # noqa: SIM102
                if not activate_pending(
                    step,
                    obs_pre["joint_positions"],
                    wait=True,
                ):
                    input_dict = policy.prepare_input(obs_pre, instruction)
                    active_chunk_id = next_chunk_id
                    next_chunk_id += 1
                    _runtime_call(
                        runtime,
                        "inference_submitted",
                        step=step,
                        chunk_id=active_chunk_id,
                        planned_switch_step=step,
                    )
                    inference_started_at = time.perf_counter()
                    action_chunk = _infer_actions(policy, input_dict)
                    action_index = 0
                    next_submit_step = step + chunk_size - lead_steps
                    _runtime_call(
                        runtime,
                        "plan_activated",
                        actions=np.asarray(action_chunk),
                        action_index=action_index,
                        chunk_id=active_chunk_id,
                        step=step,
                        action_dt=1.0 / control_rate_hz,
                        inference_ms=(time.perf_counter() - inference_started_at) * 1000.0,
                        instruction=instruction,
                        metadata={"mode": "async", "fallback": True},
                    )

            action = np.asarray(action_chunk[action_index])
            action_index += 1
            obs_post = dynamic_smoothing(env, action) or obs_pre
            saver.add_step(obs_pre=obs_pre, obs_post=obs_post)
            if runtime is not None:
                _runtime_call(
                    runtime,
                    "step_executed",
                    joint_positions=env.robot().get_joint_state(),
                    cameras=_runtime_cameras(obs_pre),
                    step=step + 1,
                    max_steps=max_steps,
                    action_index=action_index,
                    chunk_id=active_chunk_id,
                    metadata={"mode": "async"},
                )

            key = live_view.update(
                obs=obs_pre,
                rollout_idx=rollout_idx,
                num_rollouts=num_rollouts,
                step=step + 1,
                max_steps=max_steps,
                instruction=instruction,
            )
            if key == "y":
                return RolloutOutcome(end_reason="success", last_step=step + 1)
            if key == "n":
                return RolloutOutcome(end_reason="failure", last_step=step + 1)
            if key == "q":
                return RolloutOutcome(end_reason="quit", last_step=step + 1)

    return RolloutOutcome(end_reason="timeout", last_step=max_steps)


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------


def run_session(
    env: RobotEnv,
    policy: MolmoAct,
    left_cfg: dict[str, Any],
    right_cfg: dict[str, Any] | None,
    bimanual: bool,
    num_rollouts: int,
    runtime: Any = None,
) -> None:
    """Drive ``num_rollouts`` rollouts; convert the labeled set to LeRobot at the end.

    Ctrl-C flushes the partial rollout, waits for zero home, and exits the
    session. The same process never starts another rollout after interruption.
    """
    global _cleanup_done
    storage = left_cfg["storage"]
    base_save_dir = Path(storage["base_dir"]) / "data" / storage["task_directory"]
    max_steps = int(left_cfg.get("max_steps", 1000))
    last_prompt = storage.get("language_instruction") or ""

    eval_cfg = left_cfg.get("eval") or {}
    cam_srv_cfg = eval_cfg.get("camera_server") or {}
    pub_endpoint = cam_srv_cfg.get("pub_endpoint") if cam_srv_cfg.get("enabled") else None
    live_view = LiveCameraView(
        enabled=bool(eval_cfg.get("live_view_enabled", True)),
        pub_endpoint=pub_endpoint,
        recv_timeout_ms=int(cam_srv_cfg.get("recv_timeout_ms", 100)),
    )

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labeled_rollouts: list[Path] = []
    completed_rollouts = 0

    try:
        while completed_rollouts < num_rollouts:
            rollout_idx = completed_rollouts
            saver: EvalRolloutSaver | None = None
            outcome: RolloutOutcome | None = None
            rollout_dir: Path | None = None
            episode_active = False

            try:
                move_to_start_position(env, bimanual, left_cfg, right_cfg)
                instruction = prompt_instruction(rollout_idx, num_rollouts, last_prompt)
                last_prompt = instruction

                rollout_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rollout_dir = base_save_dir / "eval" / rollout_timestamp
                saver = EvalRolloutSaver(
                    rollout_dir=rollout_dir,
                    instruction=instruction,
                    max_workers=int(storage.get("saver_max_workers", 2)),
                    png_compress_level=int(storage.get("png_compress_level", 1)),
                )

                print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
                print(f"  instruction: {instruction}")
                print(f"  rollout_dir: {rollout_dir}")

                _runtime_call(
                    runtime,
                    "episode_started",
                    instruction=instruction,
                    max_steps=max_steps,
                    metadata={
                        "rollout_index": rollout_idx,
                        "num_rollouts": num_rollouts,
                        "async_inference": bool(eval_cfg.get("async_inference", False)),
                        "async_lead_steps": int(eval_cfg.get("async_lead_steps", 6)),
                        "async_blend_steps": int(eval_cfg.get("async_blend_steps", 6)),
                        "control_rate_hz": float(left_cfg.get("hz", 30)),
                    },
                )
                episode_active = True

                outcome = run_one_rollout(
                    env=env,
                    policy=policy,
                    saver=saver,
                    instruction=instruction,
                    rollout_idx=rollout_idx,
                    num_rollouts=num_rollouts,
                    max_steps=max_steps,
                    live_view=live_view,
                    async_inference=bool(eval_cfg.get("async_inference", False)),
                    async_lead_steps=int(eval_cfg.get("async_lead_steps", 6)),
                    async_blend_steps=int(eval_cfg.get("async_blend_steps", 6)),
                    control_rate_hz=float(left_cfg.get("hz", 30)),
                    runtime=runtime,
                )

                _runtime_call(
                    runtime,
                    "episode_finished",
                    reason=outcome.end_reason,
                    step=outcome.last_step,
                    metadata={"rollout_dir": str(rollout_dir)},
                )
                episode_active = False

                saver.flush()
                label = resolve_label(outcome)
                if label is not None:
                    new_path = move_rollout(rollout_dir, label, base_save_dir)
                    labeled_rollouts.append(new_path)
                    print(f"  -> labeled '{label}': {new_path}")
                else:
                    print(f"  -> kept in eval/: {rollout_dir}")

                completed_rollouts += 1
            except KeyboardInterrupt:
                interrupted_step = saver.num_steps if saver is not None else 0
                print(
                    "\n[interrupt] Ctrl-C received — saving current rollout and returning home..."
                )
                if episode_active:
                    _runtime_call(
                        runtime,
                        "episode_finished",
                        reason="interrupted",
                        step=interrupted_step,
                        metadata={"rollout_dir": str(rollout_dir) if rollout_dir else ""},
                    )
                if saver is not None:
                    try:
                        saver.flush()
                        saver.write_err(
                            reason="KeyboardInterrupt",
                            step=interrupted_step,
                        )
                        print(f"  -> incomplete rollout saved: {saver.rollout_dir}")
                    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                        logger.exception("Failed to flush incomplete rollout: %s", exc)

                if not _return_to_zero_home(env, bimanual):
                    print("Home failed; stopping the session instead of continuing.")
                    break
                _cleanup_done = True
                break
    finally:
        live_view.close()
        # Return to zero-joint home before conversion or process exit.
        _park_robot()
        _convert_if_any(labeled_rollouts, base_save_dir, session_timestamp, left_cfg)


def _convert_if_any(
    labeled_rollouts: list[Path],
    base_save_dir: Path,
    session_timestamp: str,
    left_cfg: dict[str, Any],
) -> None:
    """Best-effort LeRobot conversion of this session's labeled rollouts."""
    if not labeled_rollouts:
        print("\n[session] No labeled rollouts this session — nothing to convert.")
        return

    lerobot_cfg = left_cfg.get("lerobot", {}) or {}
    output_dir = base_save_dir / "eval_lerobot_v30" / session_timestamp
    print(
        f"\n[session] Converting {len(labeled_rollouts)} labeled rollouts "
        f"to LeRobot v3.0 at {output_dir} ..."
    )
    try:
        convert_session_to_lerobot(
            session_rollout_dirs=labeled_rollouts,
            output_dir=output_dir,
            fps=int(lerobot_cfg.get("fps", left_cfg.get("hz", 30))),
            robot_type=str(lerobot_cfg.get("robot_type", "molmoact_dual_arm")),
            repo_id=str(lerobot_cfg.get("hf_repo_id", "local/eval_session")),
            action_mode=str(lerobot_cfg.get("action_mode", "next_joint_fields")),
            vcodec=str(lerobot_cfg.get("vcodec", "libsvtav1")),
            sanitize_online_viz_meta=bool(lerobot_cfg.get("sanitize_online_viz_meta", True)),
            image_writer_processes=int(lerobot_cfg.get("image_writer_processes", 0)),
            image_writer_threads=int(lerobot_cfg.get("image_writer_threads", 0)),
            parallel_encoding=bool(lerobot_cfg.get("parallel_encoding", True)),
        )
    except Exception as exc:  # noqa: BLE001 — keep raw rollouts even if conversion fails
        logger.exception("LeRobot conversion failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(runtime: Any = None) -> None:
    atexit.register(_park_robot)
    signal.signal(signal.SIGTERM, _handle_termination)

    args = tyro.cli(Args)
    if args.num_rollouts < 1:
        raise SystemExit("--num_rollouts must be >= 1")

    env, left_cfg, right_cfg, bimanual = _build_env(args)

    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg

    if bimanual:
        move_to_start_position(env, True, left_cfg, right_cfg)
    else:
        move_to_start_position(env, False, left_cfg)

    print(f"Launching robot: {env.robot().__class__.__name__}")
    print(f"Control loop: {left_cfg.get('hz', 30)} Hz")
    print(
        f"Rollouts this session: {args.num_rollouts}, max_steps: {left_cfg.get('max_steps', 1000)}"
    )

    eval_cfg = left_cfg.get("eval") or {}
    mode = eval_cfg.get("mode", "server")
    if mode == "local":
        policy = MolmoActLocal(**(eval_cfg.get("local") or {}))
    elif mode == "server":
        policy = MolmoAct(server=eval_cfg.get("molmoact_server"))
    else:
        raise SystemExit(f"eval.mode must be 'server' or 'local', got {mode!r}")
    run_session(
        env=env,
        policy=policy,
        left_cfg=left_cfg,
        right_cfg=right_cfg,
        bimanual=bimanual,
        num_rollouts=args.num_rollouts,
        runtime=runtime,
    )


if __name__ == "__main__":
    main()
