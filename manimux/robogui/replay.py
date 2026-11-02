"""Offline replay; the module CLI opens finalized experiment trajectories."""

from __future__ import annotations

from pathlib import Path


def create_recorded_replay(episode, robot, *, source="state", host="127.0.0.1", port=8087):
    """Reuse recorded timestamps and the shared, hardware-free trajectory player."""
    from .action_replay import ActionReplayRoboGUI
    from .records import SOURCES, load_episode_trajectory

    actions, stamps = load_episode_trajectory(episode, source, robot)
    return ActionReplayRoboGUI(
        actions, robot, action_dt_s=1.0, timestamps_ns=stamps,
        source_label=SOURCES[source], host=host, port=port,
    )


def main() -> None:
    import argparse

    from .action_replay import run_replay
    from .records import SOURCES
    from .robot_view import load_robogui_config, load_robot_view

    parser = argparse.ArgumentParser(
        description="Replay a finalized experiment trajectory; no robot connection or RGB video playback."
    )
    parser.add_argument("--episode", type=Path, required=True, help="Finalized rollout directory")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--robot", help="RoboGUI robot preset, e.g. yam")
    model.add_argument("--config", type=Path, help="Custom RoboGUI YAML with offline RobotModel")
    parser.add_argument("--source", choices=tuple(SOURCES), default="state",
                        help="state: measured; scheduled: timeline reference; command: executor output")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8087)
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    episode = args.episode.expanduser().resolve()
    try:
        config = load_robogui_config(args.config, robot=args.robot)
        robot = load_robot_view(config)
        robogui = create_recorded_replay(
            episode, robot, source=args.source, host=args.host, port=args.port,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Recorded trajectory: {episode} · ticks/{args.source}")
    run_replay(robogui, host=args.host)


if __name__ == "__main__":
    main()
