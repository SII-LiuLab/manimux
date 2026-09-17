from __future__ import annotations

from pathlib import Path

import pytest

from manimux.runtime.lock import RuntimeInstanceLock, RuntimeLockError


def test_runtime_lock_rejects_a_second_owner_and_can_be_reacquired(tmp_path: Path) -> None:
    first = RuntimeInstanceLock(
        "yam",
        mode="serve",
        config_path=Path("configs/mock.yaml"),
        lock_dir=tmp_path,
    )
    second = RuntimeInstanceLock(
        "yam",
        mode="serve",
        config_path=Path("configs/mock.yaml"),
        lock_dir=tmp_path,
    )

    with first, pytest.raises(RuntimeLockError, match="already owns robot 'yam'"), second:
        pass

    with second:
        assert second.path.is_file()
