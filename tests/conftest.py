"""Explicit hardware substitutes for collection tests; never used by launchers."""

import pytest

from tests.support.leader import SyntheticLeader
from tests.support.robot import build_robot


@pytest.fixture
def collection_hardware(monkeypatch, tmp_path):
    from manimux.collection.yam import backend, runtime
    from manimux.collection.yam.gui import session

    monkeypatch.setattr(backend, "build_robot", build_robot)
    monkeypatch.setattr(session, "check_can_up", lambda channels: [])
    initialize = backend.CollectionBackend.__init__

    def isolated_backend(self, *args, **kwargs):
        # Keep ownership checks active, while isolating tests from real station locks.
        kwargs.setdefault("lock_dir", tmp_path)
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(backend.CollectionBackend, "__init__", isolated_backend)

    def build_units(config, followers_only=False):
        control = backend.CollectionBackend(
            backend.load_backend_config(config), execution_mode=config.execution_mode
        )
        control.connect()
        units = []
        for robot in config.robot.robots:
            side = robot.type.removeprefix("yam_")
            follower = backend.FollowerView(control, f"{side}_arm")
            units.append(runtime.ArmUnit(side, follower, SyntheticLeader(follower)))
        return units

    monkeypatch.setattr(runtime, "build_arm_units", build_units)
    monkeypatch.setattr(session, "build_arm_units", build_units)
