import numpy as np
import pytest
from manimux.runtime.timeline import ActionTimeline
from manimux.types import ActionChunk


@pytest.mark.parametrize('skip', [-20, -1, 0, 1])
def test_signed_skip_preserves_clock_and_never_has_negative_crop(skip):
    dt = 83333333
    timeline = ActionTimeline({'arm': 1})
    def chunk(seq, obs):
        return ActionChunk(plan_id=str(seq), request_seq=seq, observation_time_ns=obs,
                           created_time_ns=obs, action_space='joint_position',
                           dt_ns=dt, groups={'arm': np.arange(48.)[:, None]})
    kw = dict(max_plan_age_ns=5_000_000_000, current_command={'arm': np.zeros(1)},
              blend_steps=0, handoff_skip_steps=skip)
    first = timeline.commit(chunk(0, 0), now_ns=0, **kw)
    assert first.accepted and first.trimmed_steps == 0
    result = timeline.commit(chunk(1, dt), now_ns=14*dt, **kw)
    crop = max(0, 13+skip)
    assert result.accepted and result.trimmed_steps == crop
    assert timeline.active_horizon().horizon_steps == 48-crop
    assert timeline.sample(14*dt)['arm'][0] == crop
