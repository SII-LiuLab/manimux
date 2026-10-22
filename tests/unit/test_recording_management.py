import json
from pathlib import Path

import pytest

from manimux.collections import inventory
from manimux.recording.management import list_recordings, trash_recordings, restore_recordings
from manimux.robogui.study_setup import completed_repeats
from manimux.runtime.edge import _next_rollout_id


def make_episode(root, name='rollout-001', layout='05', repeat=1):
    episode = root / 'session-test' / name
    (episode/'evaluation').mkdir(parents=True)
    (episode/'meta.json').write_text(json.dumps(dict(task='coin', experiment_mode=True,
        identity_frozen=True, layout_id=layout, repeat_id=repeat)))
    (episode/'result.json').write_text('{"steps":10}')
    (episode/'evaluation/human-label.json').write_text('{"task_result":"failure","score":0}')
    (episode/'videos').mkdir()
    (episode/'videos/camera.mp4').write_bytes(b'video fixture')
    return episode


def selection(root):
    return {r['relative']:r['revision'] for r in list_recordings(root) if r['revision']}


def test_delete_restore_preserves_data_progress_and_id(tmp_path):
    episode = make_episode(tmp_path)
    make_episode(tmp_path, 'rollout-002', layout='06')
    before = {str(p.relative_to(episode)):p.read_bytes() for p in episode.rglob('*') if p.is_file()}
    selected = selection(tmp_path); selected.pop('session-test/rollout-002')
    receipt = trash_recordings(tmp_path, selected, reason='Retest')
    assert ('05','') not in completed_repeats(tmp_path,'coin',{})
    assert completed_repeats(tmp_path,'coin',{})[('06','')]=={1}
    assert [r['episode'] for r in list_recordings(tmp_path)]==['rollout-002']
    assert sum(len(s['rollouts']) for s in inventory(tmp_path))==1
    assert _next_rollout_id(episode.parent)=='rollout-003'
    restore_recordings(tmp_path,receipt)
    assert completed_repeats(tmp_path,'coin',{})[('05','')]=={1}
    assert before=={str(p.relative_to(episode)):p.read_bytes() for p in episode.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='already restored'):restore_recordings(tmp_path,receipt)


def test_deleted_highest_id_is_not_reused(tmp_path):
    episode=make_episode(tmp_path)
    trash_recordings(tmp_path,selection(tmp_path),reason='Retest')
    assert _next_rollout_id(episode.parent)=='rollout-002'


@pytest.mark.parametrize('relative', ['../outside','/tmp/rollout-001','session-test',
    'session-test/rollout-001.partial','session-test/rollout-001/videos','session-test/../rollout-001'])
def test_reject_unsafe_paths(tmp_path,relative):
    episode=make_episode(tmp_path)
    with pytest.raises(ValueError): trash_recordings(tmp_path,{relative:'x'},reason='Retest')
    assert episode.exists()


def test_reject_changed_label_and_symlink(tmp_path):
    episode=make_episode(tmp_path)
    selected=selection(tmp_path)
    (episode/'evaluation/human-label.json').write_text('{"task_result":"success"}')
    with pytest.raises(ValueError,match='changed'):trash_recordings(tmp_path,selected,reason='Retest')
    (episode.parent/'rollout-002').symlink_to(episode, target_is_directory=True)
    with pytest.raises(ValueError,match='Linked'):trash_recordings(tmp_path,{'session-test/rollout-002':'x'},reason='Retest')
    assert len(list_recordings(tmp_path))==1


def test_batch_preflight_and_rollback(tmp_path,monkeypatch):
    one=make_episode(tmp_path)
    two=make_episode(tmp_path,'rollout-002')
    selected=selection(tmp_path)
    invalid={**selected,'session-test/rollout-003':'x'}
    with pytest.raises(ValueError):trash_recordings(tmp_path,invalid,reason='Retest')
    assert one.exists() and two.exists()
    original=Path.rename
    def fail(path,target):
        if path==one:raise OSError('Injected move failure')
        return original(path,target)
    monkeypatch.setattr(Path,'rename',fail)
    with pytest.raises(OSError):trash_recordings(tmp_path,selected,reason='Retest')
    assert one.exists() and two.exists()
    assert not list(one.parent.glob('*.deleted'))


def test_restore_refuses_overwrite(tmp_path):
    episode=make_episode(tmp_path)
    receipt=trash_recordings(tmp_path,selection(tmp_path),reason='Retest')
    episode.mkdir()
    with pytest.raises(ValueError,match='overwrite'):restore_recordings(tmp_path,receipt)
    assert (receipt.parent/'session-test/rollout-001/meta.json').exists()


def test_partial_is_visible_but_not_deletable(tmp_path):
    make_episode(tmp_path,'rollout-001.partial')
    row=list_recordings(tmp_path)[0]
    assert row['revision'] is None
    with pytest.raises(ValueError):trash_recordings(tmp_path,{row['relative']:'x'},reason='Retest')
