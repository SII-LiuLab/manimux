"""Finish button to EdgeRuntime shutdown; fake devices, no physical motion."""
import json
from types import SimpleNamespace

import pytest

from test_research_workflow import robot, robogui, ready, click
from test_runtime_shutdown import runtime_case


@pytest.mark.parametrize('actions', [[], ['home'], ['home', 'drag']])
def test_finish_button_homes_even_when_config_disables_home(robogui, runtime_case, actions):
    ready(robogui, recovery={'available': bool(actions), 'actions': actions})
    robogui._update_event({'event': 'episode_started', 'metadata': {
        'episode_id': 'rollout-001', 'run_dir': robogui.service_id,
        'experiment_mode': False, 'launch_mode': 'serve', 'control_mode': 'managed'}})
    assert robogui.finish_btn.label == 'Finish & Home'
    assert not robogui.finish_btn.disabled
    click(robogui.finish_btn)
    controls = robogui.control_state()
    assert controls['finish_requested'] is True
    assert controls['finish_home'] is True
    case = runtime_case
    case.runtime._config['robot']['options']['home_on_close'] = False
    controls["control_run_dir"] = str(case.root.resolve())
    case.gui.poll_control.return_value = SimpleNamespace(**controls)
    result = case.runtime.run()
    assert case.events.index('robot.home') < case.events.index('robot.stop')
    assert case.events.index('robot.home') < case.events.index('robot.close')
    events = [json.loads(line) for line in (result.episode_dir / 'events.jsonl').read_text().splitlines()]
    assert next(e for e in events if e['kind'] == 'robogui_finish_requested')['home'] is True


def test_explicit_hand_guiding_finish_is_separate(robogui):
    ready(robogui)
    robogui._update_event({'event': 'episode_started', 'metadata': {
        'episode_id': 'rollout-001', 'run_dir': robogui.service_id,
        'experiment_mode': False, 'launch_mode': 'serve', 'control_mode': 'managed'}})
    robogui._finish_rollout(home=False)
    assert robogui.finish_home is False
