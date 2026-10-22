from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import json

import numpy as np
import pytest
import zarr

from manimux.cli import load_config, run_parameters
from manimux.evaluation.identity import rollout_identity
from manimux.robogui.reference_layouts import ReferenceLayouts
from manimux.robogui.records import load_episode_trajectory
from manimux.robogui.action_replay import ActionReplayRoboGUI
from manimux.robogui import dashboard
from manimux.session import RuntimeSessionService


def test_identity_free_study_and_legacy():
    free = rollout_identity(dict(experiment_name='throwing', condition='A', notes='trial',
                                 experiment_template={'repeats': 2}))
    assert free['experiment_name'] == 'throwing' and free['repeat_id'] is None
    arbitrary = rollout_identity(dict(experiment_mode=True, layout_id='scene-12', repeat_id=15))
    assert arbitrary['reference_layout'] is None and arbitrary['repeat_id'] == 15
    old = dict(experiment_mode=True, layout_id='01', repeat_id=3,
               reference_layout=dict(task='bottles', path='/tmp/01.png', sha256='a'*64))
    assert rollout_identity(old)['reference_layout'] == old['reference_layout']


def test_template_limits_and_snapshot():
    template = dict(name='Custom', layout_ids=['near','far'], repeats=5, require_reference=False)
    identity = rollout_identity(dict(experiment_mode=True, layout_id='far', repeat_id=5,
                                     experiment_template=template))
    template['layout_ids'].append('new')
    assert identity['experiment_template']['layout_ids'] == ['near','far']
    for overrides in ({'layout_id':'wrong','repeat_id':1}, {'layout_id':'near','repeat_id':6}):
        with pytest.raises(ValueError):
            rollout_identity(dict(experiment_mode=True, experiment_template=template, **overrides))
    with pytest.raises(ValueError, match='reference'):
        rollout_identity(dict(experiment_mode=True, experiment_template={'require_reference':True}))


def test_real_config_loader():
    config = load_config(Path('manimux/configs/experiments/pi05-latest300ep-step30000/coin/rtc32_beta10.yaml'))
    assert config['experiment'] == {'task': 'coin', 'variant': 'rtc32_beta10'}
    assert config['inference']['rtc']['min_execute_policy_steps'] == 32
    assert config['evaluation']['kind'] == 'stages'
    assert config['run']['experiment_template'] is not None
    with pytest.raises(ValueError):
        run_parameters(experiment_template={'repeatz':4})


def test_reference_named_and_legacy(tmp_path):
    library = ReferenceLayouts(tmp_path)
    image=np.zeros((4,5,3),dtype=np.uint8)
    for name in ('01','10','11','near'):
        library.save('demo',name,image)
    assert library.slots('demo') == ('01','10','11','near')
    _, frozen = library.snapshot('demo','near')
    image[:] = 200
    library.save('demo','near',image)
    assert frozen != library.snapshot('demo','near')[1]
    with pytest.raises(ValueError):
        library.save('demo','../escape', image)


@pytest.fixture
def robot():
    return dashboard.load_robot_view(dashboard.load_robogui_config(robot='yam'))


@pytest.fixture
def robogui(monkeypatch, robot, tmp_path):
    # All message transports are replaced; the Viser UI itself is real.
    class Transport:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
    monkeypatch.setattr(dashboard, 'RoboGUIReceiver', Transport)
    monkeypatch.setattr(dashboard, 'ControlServer', Transport)
    instance=dashboard.PolicyRoboGUI('127.0.0.1',0,'unused','unused',robot,
                                    reference_root=tmp_path/'references')
    yield instance
    instance.close()


def ready(robogui, **fields):
    meta = dict(run_dir='/tmp/research-session', task='Move the object.',
                runtime='rtc', policy_label='Example')
    meta.update(fields)
    robogui._update_event(dict(event='runtime_service_ready', metadata=meta))
    import time
    deadline = time.monotonic() + 2
    while robogui._history_loading and time.monotonic() < deadline:
        time.sleep(.001)
    with robogui.lock:
        assert not robogui._history_loading


def test_gui_experiment_button_uses_yaml_defaults(robogui):
    ready(robogui, experiment_template={'name':'Study','layout_ids':['near','far'],
                                      'repeats':5,'require_reference':False},
          research_defaults={'experiment_name':'A','condition':'baseline','notes':'configured'})
    ready(robogui)
    robogui.layout.value='far'
    click(robogui.prepare_experiment_btn)
    request=deepcopy(robogui._rollout_request)
    assert request['layout_id']=='far' and request['repeat_id']==1
    assert request['reference_layout'] is None
    assert request['notes']=='configured' and request['condition']=='baseline'
    assert request['experiment_mode']
    assert not hasattr(robogui, 'template_layout')
    assert not hasattr(robogui, 'use_reference')
    assert not hasattr(robogui, 'rollout_mode')
    assert robogui.prepare_normal_btn.label == 'Prepare Run'


def test_gui_template_requires_reference(robogui):
    ready(robogui, experiment_template={'name':'Images','layout_ids':[],
                                      'repeats':None,'require_reference':True})
    robogui._prepare_rollout(experiment_mode=True)
    assert not robogui.new_rollout_requested
    assert 'not submitted' in robogui.rollout_setup_status.content
    robogui._prepare_rollout(experiment_mode=False)
    assert robogui.new_rollout_requested


def episode(tmp_path):
    folder=tmp_path/'rollout-001';folder.mkdir()
    (folder/'meta.json').write_text(json.dumps({'experiment_name':'Study','condition':'A'}))
    (folder/'result.json').write_text(json.dumps({'success':True,'steps':3}))
    ticks=zarr.open_group(str(folder/'data.zarr'),mode='w').create_group('ticks')
    stamps=np.array([1000000000,1100000000,1500000000],dtype=np.int64)
    ticks.create_dataset('monotonic_ns',data=stamps)
    for stream in ('state','command','scheduled'):
        group=ticks.create_group(stream)
        group.create_dataset('left_arm',data=np.zeros((3,7)))
        group.create_dataset('right_arm',data=np.ones((3,7))*0.2)
    return folder


def test_record_reader_and_nonuniform_playback(tmp_path,robot):
    folder=episode(tmp_path)
    actions, stamps=load_episode_trajectory(folder,'state',robot)
    replay=ActionReplayRoboGUI(actions,robot,action_dt_s=1,timestamps_ns=stamps,port=0,
                              source_label='Measured state')
    try:
        assert replay.times.tolist()==[0,0.1,0.5]
        replay.seek(1)
        assert replay.frame.value==1
        replay.play.value=True
        anchor=replay._last_time
        replay.tick(now=anchor+0.2)
        assert replay.frame.value==1  # No invented frame during a recording gap.
        replay.tick(now=anchor+0.45)
        assert replay.frame.value==2 and not replay.play.value
        assert 'Measured state' in replay.status.content
    finally:
        replay.close()
    with pytest.raises(ValueError):
        load_episode_trajectory(folder,'raw_policy',robot)


def click(handle):
    for callback in handle._impl.update_cb:
        callback(None)


def test_recorded_replay_cli_is_separate_and_paused(robogui, tmp_path, monkeypatch):
    from manimux.robogui import replay, action_replay
    folder = episode(tmp_path)
    assert not hasattr(robogui, 'records')
    opened = []
    def inspect(player, *, host):
        try:
            assert host == '127.0.0.1'
            assert not player.play.value
            assert player.source_label == 'Executor command'
            assert player.server is not robogui.server
            assert player.times.tolist() == [0, 0.1, 0.5]
            player.seek(2)
            assert player.frame.value == 2
            opened.append(True)
        finally:
            player.close()
    monkeypatch.setattr(action_replay, 'run_replay', inspect)
    monkeypatch.setattr('sys.argv', ['replay', '--episode', str(folder), '--robot', 'yam',
                                     '--source', 'command', '--port', '0'])
    replay.main()
    assert opened == [True]


@pytest.mark.parametrize('source', ['state', 'scheduled', 'command'])
def test_recorded_sources_keep_timestamps_and_values(tmp_path, robot, source):
    folder = episode(tmp_path)
    actions, stamps = load_episode_trajectory(folder, source, robot)
    assert stamps.tolist() == [1000000000, 1100000000, 1500000000]
    assert actions['left_arm'].shape == (3, 7)
    np.testing.assert_allclose(actions['right_arm'], 0.2)


def test_replay_rejects_partial_missing_stream_and_nonmonotonic_time(tmp_path, robot):
    folder = episode(tmp_path)
    partial = folder.with_suffix('.partial')
    folder.rename(partial)
    with pytest.raises(ValueError, match='finalized'):
        load_episode_trajectory(partial, 'state', robot)
    partial.rename(folder)
    ticks = zarr.open_group(str(folder / 'data.zarr'), mode='a')['ticks']
    del ticks['command']
    with pytest.raises(ValueError, match='no ticks/command'):
        load_episode_trajectory(folder, 'command', robot)
    ticks['monotonic_ns'][:] = [100, 100, 200]
    with pytest.raises(ValueError, match='strictly increase'):
        load_episode_trajectory(folder, 'state', robot)


def test_session_identity_does_not_leak_between_prepares(tmp_path, monkeypatch):
    config=load_config(Path('manimux/configs/experiments/pi05-latest300ep-step30000/coin/rtc32_beta10.yaml'))
    original_name=config['run']['experiment_name']
    config['run']['experiment_template']={'name':'Study','layout_ids':[],
                                        'repeats':None,'require_reference':False}
    requests=iter([dict(experiment_mode=True,experiment_name='A',notes='first',repeat_id=4),
                   dict(experiment_mode=False,experiment_name='B')])
    records=[]
    class Runtime:
        def __init__(self,cfg,path): records.append(deepcopy(cfg['run']))
        def run(self): return SimpleNamespace(episode_dir=tmp_path/'episode',terminal_reason='done')
    service=RuntimeSessionService(config,tmp_path,runtime_factory=Runtime)
    monkeypatch.setattr(service,'_wait_for_rollout_request',lambda:next(requests))
    service.serve(max_rollout_attempts=2)
    assert records[0]['notes']=='first'
    assert records[1]['notes']=='' and records[1]['repeat_id'] is None
    assert records[1]['experiment_name']=='B'
    assert config['run']['experiment_name']==original_name

@pytest.mark.parametrize('recipe,method', [
    ('pi05-yam-put-bottles-joint-step30000/put_bottles/serial_joint_step30000.yaml', 'serial'),
    ('pi05-yam-put-bottles-joint-step30000/put_bottles/rtc_joint_step30000.yaml', 'rtc'),
    ('pi05-yam-put-bottles-pretrained/put_bottles/rtc_pretrained.yaml', 'rtc'),
])
def test_aligned_bottle_study_survives_integration(recipe, method):
    from manimux.collections import load_collection
    group = recipe.split('/')[0]
    source = Path('manimux/configs/lists') / (group + '-standalone.yaml')
    if not source.exists():
        pytest.skip('Private historical study configuration is not distributed')
    catalog = load_collection(source)
    target = (Path('manimux/configs/experiments') / recipe).resolve()
    cfg = next(c.config for c in catalog.choices.values() if c.path == target)
    assert cfg['robot']['control_hz'] == 100
    assert cfg['executor']['type'] == 'direct'
    assert cfg['policy']['action_dt_s'] == pytest.approx(1 / 30)
    assert cfg['inference']['blend_policy_steps'] == 0
    if method == 'rtc':
        assert cfg['inference']['rtc']['min_execute_policy_steps'] == 32
        assert cfg['inference']['rtc']['beta'] == 10
    else:
        assert cfg['inference']['chunk_policy_steps'] == 16
    assert cfg['policy_server']['inference_seed'] == 0
    assert cfg['policy_server']['robot_action_dim_info'] == {'arm_dim': [6, 6], 'ee_dim': [1, 1]}
    assert cfg['policy_server']['num_envs'] == 1
    template = cfg['run']['experiment_template']
    assert template['layout_ids'] == [f'{i:02}' for i in range(1, 11)]
    assert template['repeats'] == 3 and template['require_reference'] is True
    attempt = dict(cfg['run'], experiment_mode=True, layout_id='10', repeat_id=3,
                   reference_layout={'task': 'put_bottles_into_the_bin',
                                     'path': '/tmp/reference-10.png', 'sha256': 'a' * 64})
    assert rollout_identity(attempt)['repeat_id'] == 3
    for override in ({'layout_id': '11'}, {'repeat_id': 4}, {'reference_layout': None}):
        with pytest.raises(ValueError):
            rollout_identity({**attempt, **override})
    assert rollout_identity({**attempt, 'experiment_mode': False})['reference_layout'] is None


def test_prepared_reference_can_change_until_first_start(robogui):
    template = dict(name="Study", layout_ids=["01", "02"], repeats=3, require_reference=True)
    overlay = robogui.top_overlay
    overlay.layouts.save("demo", "01", np.zeros((20, 20, 3), dtype=np.uint8))
    overlay.layouts.save("demo", "02", np.ones((20, 20, 3), dtype=np.uint8))
    ready(robogui, task="demo", experiment_template=template)
    robogui._prepare_rollout(experiment_mode=True)
    prepared = deepcopy(robogui._rollout_request)
    assert prepared["layout_id"] == "01"
    robogui._update_event(dict(event="episode_started", metadata={
        **prepared, "episode_id": "rollout-001", "run_dir": "/tmp/research-session",
        "control_mode": "managed", "warmup": {"phase": "idle"},
    }))
    assert not robogui.layout.disabled
    robogui.layout.value = "02"
    robogui._refresh_layout()
    robogui._activate_warmup()
    robogui._update_warmup({"phase": "warming"})
    assert not robogui.layout.disabled
    robogui._start_rollout()
    assert robogui.rollout_started and not robogui.paused
    control = robogui.control_state()
    assert control["start_identity"]["layout_id"] == "02"
    assert control["start_identity"]["reference_layout"] == overlay.layouts.snapshot("demo", "02")[1]["reference_layout"]
    assert robogui.layout.disabled
    robogui.paused = True
    robogui._set_policy_controls_enabled(True)
    assert robogui.layout.disabled
    assert prepared["layout_id"] == "01"


def test_start_revalidates_reference_against_prepared_template(robogui):
    ready(robogui, experiment_template=dict(name="Study", layout_ids=["01"], repeats=3, require_reference=False))
    robogui._prepare_rollout(experiment_mode=True)
    prepared = deepcopy(robogui._rollout_request)
    robogui._update_event(dict(event="episode_started", metadata={
        **prepared, "episode_id": "rollout-001", "control_mode": "managed",
    }))
    overlay = robogui.top_overlay
    overlay.layouts.save("demo", "02", np.zeros((20, 20, 3), dtype=np.uint8))
    robogui._reference_task = "demo"
    robogui.layout.options = ("01", "02")
    robogui.layout.value = "02"
    robogui._start_rollout()
    assert robogui.paused and not robogui.rollout_started
    assert "not submitted" in robogui.status.content
    assert not robogui.layout.disabled


def test_start_identity_keeps_rules_and_prepared_fields():
    from manimux.evaluation.identity import start_identity
    prepared = rollout_identity(dict(experiment_mode=True, layout_id="01", repeat_id=2,
        experiment_template=dict(layout_ids=["01", "02"], repeats=3), condition="rtc"))
    result = start_identity(prepared, dict(layout_id="02", reference_layout=None))
    assert result["layout_id"] == "02" and result["repeat_id"] == 2 and result["condition"] == "rtc"
    with pytest.raises(ValueError):
        start_identity(prepared, dict(layout_id="03"))
    with pytest.raises(ValueError):
        start_identity(prepared, dict(experiment_mode=False))


def test_reconnect_after_start_keeps_layout_locked(robogui):
    ready(robogui)
    robogui._prepare_rollout(experiment_mode=False)
    robogui._update_event(dict(event="episode_started", metadata={
        **robogui._rollout_request, "episode_id": "rollout-001", "control_mode": "managed",
        "identity_frozen": True, "warmup": {"phase": "complete", "latency_calibrated": True},
    }))
    assert robogui.rollout_started and robogui.paused
    assert robogui.layout.disabled
    assert robogui.start_btn.label == "Resume rollout"
    robogui._start_rollout()
    assert not robogui.paused
    assert robogui._start_identity == {
        "layout_id": "", "repeat_id": None, "reference_layout": None,
    }


def test_layout07_is_the_only_selection_and_reference_is_attached(robogui):
    library = robogui.top_overlay.layouts
    library.save('put_the_coin_into_the_bin','07',np.zeros((20,20,3),dtype=np.uint8))
    ready(robogui, task='put_the_coin_into_the_bin',
          experiment_template=dict(layout_ids=['01','07'],repeats=3,require_reference=True))
    robogui.layout.value='07'
    click(robogui.prepare_experiment_btn)
    assert robogui.new_rollout_requested
    request=robogui._rollout_request
    assert request['layout_id']=='07' and request['repeat_id']==1
    assert request['reference_layout']==library.snapshot('put_the_coin_into_the_bin','07')[1]['reference_layout']


def test_repeat_restored_from_scored_records_and_recomputed_on_layout_change(robogui,tmp_path):
    root=tmp_path/'method'
    library=robogui.top_overlay.layouts
    for slot in ['01','07']:
        library.save('demo',slot,np.full((20,20,3),int(slot),dtype=np.uint8))
    ref=library.snapshot('demo','07')[1]['reference_layout']
    ep=root/'session-old'/'rollout-001';(ep/'evaluation').mkdir(parents=True)
    (ep/'meta.json').write_text(json.dumps(dict(task='demo',experiment_mode=True,identity_frozen=True,layout_id='07',repeat_id=1,reference_layout=ref)))
    (ep/'result.json').write_text('{}')
    (ep/'evaluation/human-label.json').write_text(json.dumps({'task_result':'failure'}))
    ready(robogui,run_dir=str(root/'session-new'),task='demo',
          experiment_template=dict(layout_ids=['01','07'],repeats=3,require_reference=True))
    click(robogui.prepare_experiment_btn)
    prepared=deepcopy(robogui._rollout_request)
    assert prepared['repeat_id']==1
    robogui._update_event(dict(event='episode_started',metadata={**prepared,'instruction':'demo',
        'episode_id':'rollout-001','run_dir':str(root/'session-new'),'control_mode':'managed'}))
    robogui.layout.value='07'
    robogui._start_rollout()
    assert robogui.rollout_started
    assert robogui._start_identity['repeat_id']==2
    assert robogui._start_identity['layout_id']=='07'


def test_repeat_history_excludes_skipped_invalid_partial_and_other_images(tmp_path):
    from manimux.robogui.study_setup import completed_repeats,next_repeat
    for i,result in enumerate(['failure','success','invalid',None],1):
        ep=tmp_path/'session-a'/f'rollout-{i:03d}';(ep/'evaluation').mkdir(parents=True)
        (ep/'meta.json').write_text(json.dumps(dict(task='demo',experiment_mode=True,identity_frozen=True,
            layout_id='07',repeat_id=i,reference_layout={'sha256':'old' if i==2 else 'new'})))
        (ep/'result.json').write_text('{}')
        if result:(ep/'evaluation/human-label.json').write_text(json.dumps({'task_result':result}))
    history=completed_repeats(tmp_path,'demo',{})
    assert history=={('07','new'):{1},('07','old'):{2}}
    assert next_repeat(history['07','new'],3)==2
    assert next_repeat({1,3},3)==2
    with pytest.raises(ValueError,match='Choose another'):next_repeat({1,2,3},3)


def test_run_button_bypasses_template_and_experiment_remains_available(robogui):
    template = dict(layout_ids=['01', '07'], repeats=3, require_reference=True)
    ready(robogui, experiment_template=template)
    assert not hasattr(robogui, 'rollout_mode')
    assert not robogui.prepare_normal_btn.disabled
    assert not robogui.prepare_experiment_btn.disabled
    click(robogui.prepare_normal_btn)
    request = deepcopy(robogui._rollout_request)
    assert robogui.new_rollout_requested and not request['experiment_mode']
    assert not request['layout_id'] and request['repeat_id'] is None
    assert request['reference_layout'] is None
    assert not robogui.layout.visible
    assert robogui.prepare_normal_btn.disabled and robogui.prepare_experiment_btn.disabled
    click(robogui.prepare_experiment_btn)
    assert robogui._rollout_request == request
    robogui._update_event(dict(event='episode_started', metadata={
        **request, 'episode_id': 'rollout-001', 'run_dir': '/tmp/research-session',
        'control_mode': 'managed', 'launch_mode': 'serve',
    }))
    assert robogui.evaluation_complete
    assert not robogui.layout.visible
    robogui._update_event(dict(event='episode_finished', metadata={}))
    ready(robogui, experiment_template=template)
    assert not robogui.prepare_experiment_btn.disabled
    assert robogui.layout.visible
    assert robogui.experiment_template == template
    click(robogui.prepare_experiment_btn)
    assert 'not submitted' in robogui.rollout_setup_status.content


def test_experiment_button_attaches_layout_and_blocks_prepare_until_evaluation(robogui):
    library = robogui.top_overlay.layouts
    library.save('demo', '07', np.zeros((20, 20, 3), dtype=np.uint8))
    ready(robogui, task='demo')
    robogui.layout.value = '07'
    click(robogui.prepare_experiment_btn)
    request = deepcopy(robogui._rollout_request)
    assert request['experiment_mode'] and request['layout_id'] == '07'
    assert request['reference_layout'] == library.snapshot('demo', '07')[1]['reference_layout']
    robogui._update_event(dict(event='episode_started', metadata={
        **request, 'episode_id': 'rollout-001', 'run_dir': '/tmp/research-session',
        'control_mode': 'managed', 'launch_mode': 'serve',
    }))
    assert robogui.prepare_normal_btn.disabled and robogui.prepare_experiment_btn.disabled
    assert not robogui.layout.disabled
    robogui._update_event(dict(event='episode_finished', metadata={}))
    ready(robogui)
    assert not robogui.evaluation_complete
    assert robogui.prepare_normal_btn.disabled and robogui.prepare_experiment_btn.disabled


def test_reconnect_during_free_run_recovers_experiment_rules_when_idle(robogui):
    robogui._update_event(dict(event='episode_started', metadata={
        'experiment_mode': False, 'experiment_template': None,
        'instruction': 'demo', 'episode_id': 'rollout-001',
        'run_dir': '/tmp/research-session', 'control_mode': 'managed',
        'launch_mode': 'serve', 'identity_frozen': True,
    }))
    assert robogui.prepare_normal_btn.disabled and robogui.prepare_experiment_btn.disabled
    robogui._update_event(dict(event='episode_finished', metadata={}))
    template = dict(layout_ids=['01', '07'], repeats=3, require_reference=True)
    ready(robogui, task='demo', experiment_template=template)
    assert robogui.experiment_template == template
    assert robogui.layout.options == ('01', '07')
    click(robogui.prepare_experiment_btn)
    assert not robogui.new_rollout_requested
    assert 'not submitted' in robogui.rollout_setup_status.content


@pytest.mark.parametrize('kind', ['binary', 'count'])
def test_reopen_unscored_experiment_after_free_run_uses_recorded_rubric(robogui, tmp_path, kind):
    root = tmp_path / 'method'
    ep = root / 'session-old' / 'rollout-005'
    ep.mkdir(parents=True)
    profile = {'kind': 'binary'} if kind == 'binary' else dict(
        kind='count', task_id='demo', rubric_id='old-count-v1', label='Demo',
        count_label='Blocks stacked', target_count=2)
    meta = dict(task='demo', experiment_mode=True, identity_frozen=True,
                layout_id='09', repeat_id=1, evaluation=profile)
    raw = json.dumps(meta)
    (ep / 'meta.json').write_text(raw)
    (ep / 'result.json').write_text(json.dumps({'steps': 30, 'run_completed': True}))
    ready(robogui, run_dir=str(root / 'session-new'), task='demo',
          evaluation={'kind': 'binary'}, last_episode_dir=str(root / 'session-new' / 'rollout-006'))
    assert not robogui.evaluation_folder.visible
    assert robogui.score_previous_btn.visible and not robogui.score_previous_btn.disabled
    before = robogui.control_state()
    click(robogui.score_previous_btn)
    assert robogui.evaluation_folder.visible and not robogui.save_evaluation_btn.disabled
    assert robogui.current_episode_dir == ep and not robogui.evaluation_complete
    assert robogui.evaluation_profile == profile
    ready(robogui, run_dir=str(root / 'session-new'), task='demo', evaluation={'kind':'binary'})
    assert robogui.current_episode_dir == ep and robogui.evaluation_profile == profile
    after = robogui.control_state()
    for key in ['episode_id', 'control_run_dir', 'new_rollout_requested', 'warmup_requested',
                'home_requested', 'finish_requested', 'start_identity']:
        assert before[key] == after[key]
    if kind == 'count':
        robogui.completed_count.value = '1'
    else:
        robogui.task_result.value = 'Not completed'
    robogui.operator_note.value = 'Block ejected during grasp'
    click(robogui.save_evaluation_btn)
    import time
    deadline = time.monotonic() + 2
    while (robogui._saving_evaluation or robogui._history_loading) and time.monotonic() < deadline:
        time.sleep(.001)
    with robogui.lock:
        assert not robogui._saving_evaluation and not robogui._history_loading
    saved = json.loads((ep / 'evaluation/human-label.json').read_text())
    assert saved['task_result'] == 'failure'
    assert saved['evaluation'] == profile
    if kind == 'count': assert saved['score'] == 0.5
    assert (ep / 'meta.json').read_text() == raw
    assert robogui.evaluation_complete and not robogui.score_previous_btn.visible
    assert not robogui.prepare_normal_btn.disabled


def test_pending_scoring_excludes_free_partial_unstarted_and_labelled_trials(tmp_path):
    from manimux.robogui.study_setup import unscored_experiments
    for name, changes, steps, labeled in [
        ('rollout-001', {}, 30, False),
        ('rollout-002', {'experiment_mode': False}, 30, False),
        ('rollout-003.partial', {}, 30, False),
        ('rollout-004', {'identity_frozen': False}, 0, False),
        ('rollout-005', {}, 30, True),
        ('rollout-006', {'task': 'other'}, 30, False),
    ]:
        ep = tmp_path / 'session-a' / name
        ep.mkdir(parents=True)
        (ep/'meta.json').write_text(json.dumps(dict(task='demo', experiment_mode=True,
                                                   identity_frozen=True, **{}) | changes))
        (ep/'result.json').write_text(json.dumps({'steps':steps}))
        if labeled:
            (ep/'evaluation').mkdir()
            (ep/'evaluation/human-label.json').write_text('{}')
    assert list(unscored_experiments(tmp_path, 'demo').values()) == [tmp_path/'session-a'/'rollout-001']


@pytest.mark.parametrize('active,preparing', [(True, False), (False, True)])
def test_cannot_open_old_evaluation_during_rollout(robogui, active, preparing):
    ready(robogui)
    robogui.episode_active = active
    robogui.preparing_rollout = preparing
    original = robogui.current_episode_dir
    robogui._open_unscored_experiment()
    assert robogui.current_episode_dir == original and robogui.evaluation_complete


@pytest.mark.parametrize('rubric_file, selected, score, label_text', [
    ('stack_blocks_small', 'Blocks stacked: 1/2 · 0.5', 0.5, '0.50 · Blocks stacked: 1/2'),
    ('coin', 'Left-right handover complete · 0.3', 0.3, '0.30 · Left-right handover complete'),
    (None, 'Not completed', None, '0 · Not completed'),
])
def test_expanded_score_choices_load_template_and_save_one_outcome(
    robogui, tmp_path, rubric_file, selected, score, label_text,
):
    import yaml
    import viser
    profile = (yaml.safe_load(Path(f'manimux/configs/tasks/{rubric_file}.yaml').read_text())['evaluation']
               if rubric_file else None)
    ep = tmp_path / 'rollout-001'
    ep.mkdir()
    (ep / 'meta.json').write_text(json.dumps({'evaluation': profile}))
    (ep / 'result.json').write_text('{}')
    robogui._configure_evaluation(profile)
    robogui._set_experiment_mode(True)
    robogui._reset_evaluation(str(ep))
    robogui.episode_finalized = True
    robogui._set_evaluation_enabled(True)
    choices = robogui.task_result
    assert choices.value == 'unlabeled'
    assert 'unlabeled' not in choices._buttons
    assert choices._buttons[selected].label == label_text
    click(robogui.save_evaluation_btn)
    import time
    deadline = time.monotonic() + 2
    while (robogui._saving_evaluation or robogui._history_loading) and time.monotonic() < deadline:
        time.sleep(.001)
    with robogui.lock:
        assert not robogui._saving_evaluation and not robogui._history_loading
    assert not (ep / 'evaluation/human-label.json').exists()
    buttons = list(choices._buttons.values())
    click(buttons[-1])
    click(choices._buttons[selected])
    assert choices.value == selected
    assert sum(button.icon == viser.Icon.CIRCLE_DOT for button in buttons) == 1
    assert all(button.visible and not button.disabled for button in buttons)
    assert all(choices._heading.order < button.order < choices._heading.order + 1
               for button in buttons)
    robogui.invalid_trial.value = True
    robogui._set_evaluation_enabled(True)
    click(buttons[-1])
    assert choices.value == selected
    robogui.invalid_trial.value = False
    robogui._set_evaluation_enabled(True)
    click(robogui.save_evaluation_btn)
    import time
    deadline = time.monotonic() + 2
    while (robogui._saving_evaluation or robogui._history_loading) and time.monotonic() < deadline:
        time.sleep(.001)
    with robogui.lock:
        assert not robogui._saving_evaluation and not robogui._history_loading
    saved = json.loads((ep / 'evaluation/human-label.json').read_text())
    assert saved['task_result'] == 'failure'
    if score is not None:
        assert saved['score'] == score
    assert all(button.disabled for button in buttons)


def test_radio_rubric_change_clears_selection_and_rejects_stale_clicks(robogui):
    ready(robogui)
    robogui._configure_evaluation(None)
    choices = robogui.task_result
    choices.disabled = False
    old_button = choices._buttons['Completed']
    click(old_button)
    assert choices.value == 'Completed'
    robogui._configure_evaluation(None)
    click(old_button)
    assert choices.value == 'unlabeled'
