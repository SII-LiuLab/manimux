"""Exercise the actual Viser scene and named runtime messages without devices."""

import time

import numpy as np
import zmq

from manimux.types import ActionChunk, RobotState, SensorFrame
from manimux.viewer.communication import ViewerPublisher
from manimux.viewer.dashboard import PolicyViewer, load_robot_view, load_viewer_config
from manimux.viewer.publisher import ViewerBridge


def test_tianji_offline_scene_receives_runtime_groups_and_frames(tmp_path):
    config = load_viewer_config()
    view = load_robot_view(config)
    app = PolicyViewer(
        "127.0.0.1",
        0,
        "tcp://127.0.0.1:*",
        "tcp://127.0.0.1:*",
        view,
        reference_root=tmp_path,
        viewer_config=config,
    )
    bridge = ViewerBridge(enabled=False, robot=view.name)
    publisher = ViewerPublisher(app.receiver._socket.getsockopt_string(zmq.LAST_ENDPOINT))
    try:
        assert set(app.robot_handles) == {"left_arm", "right_arm"}
        assert len(app.camera_view.images) == 2
        from manimux.viewer.communication import PolicyPlan, RobotSnapshot, RuntimeEvent

        bridge._enabled = True
        bridge._publisher = publisher
        bridge._policy_plan_type = PolicyPlan
        bridge._snapshot_type = RobotSnapshot
        bridge._runtime_event_type = RuntimeEvent
        assert publisher._socket.poll(1000, zmq.POLLOUT)
        bridge.publish_event(
            "episode_started",
            metadata={
                "control_mode": "observe",
                "camera_map": {
                    "left_wrist": "left_wrist",
                    "right_wrist": "right_wrist",
                },
            },
        )
        deadline = time.monotonic() + 3
        while not app.episode_active and time.monotonic() < deadline:
            time.sleep(0.01)
        assert app.episode_active
        groups = {name: view.initial_positions(name) for name in view.model.groups}
        chunk = ActionChunk(
            plan_id="offline",
            request_seq=3,
            observation_time_ns=1,
            created_time_ns=2,
            action_space="joint_position",
            dt_ns=33_333_333,
            groups={name: np.tile(q, (4, 1)) for name, q in groups.items()},
        )
        bridge.publish_plan(chunk, 1.0)
        deadline = time.monotonic() + 3
        while app.plan_chunk_id != 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert app.plan_chunk_id == 3
        frames = {
            name: SensorFrame(
                name=name,
                data=np.full((12, 16, 3), color, np.uint8),
                capture_monotonic_ns=123,
                sequence=5,
            )
            for name, color in [("left_wrist", 40), ("right_wrist", 190)]
        }
        bridge.publish_state(
            RobotState(groups, 123, 8), frames, step=1, max_steps=10, active_chunk_id=3
        )
        deadline = time.monotonic() + 3
        while app.camera_view._received != {0, 1} and time.monotonic() < deadline:
            time.sleep(0.01)
        assert app.camera_view._received == {0, 1}
        assert set(app.last_joint_positions) == set(groups)
        assert app.plan_chunk_id == 3
        assert "Rejected" not in app.status.content
        assert all(app.current_plan_handles.values())
        for name, q in groups.items():
            np.testing.assert_allclose(app.last_joint_positions[name], q)
    finally:
        bridge.close()
        app.close()
