"""Disconnected Poke contract checks; all API requests use a fake transport."""

import base64
import io
import json
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation

from manimux.cli import load_config
from manimux.kinematics.poses import matrix_pose, pose_matrix
from manimux.policies.poke.client import PokePolicyModel
from manimux.policies.poke.codec import PokeCodec
from manimux.policy_adapter import build_policy_adapter
from manimux.types import ActionContext, InferenceRequest, ObservationSnapshot, RobotState, SensorFrame


@pytest.fixture
def config(tmp_path):
    c = load_config('manimux/configs/experiments/example-yam-poke/serial.yaml')
    c['policy']['options']['response_log_dir'] = str(tmp_path / 'responses')
    a = c['policy']['adapter']
    a['model_from_base'] = [.2, -.1, .3, *Rotation.from_euler('z', 35, degrees=True).as_quat()[[3,0,1,2]]]
    a['flange_from_tcp'] = [0, 0, .13, 1, 0, 0, 0]
    a['flange_from_model_tool'] = [0,0,0, *Rotation.from_euler('x', 25, degrees=True).as_quat()[[3,0,1,2]]]
    return c


@pytest.fixture
def schema():
    return {'model':'test-schema-model','revision':'v1',
      'inputs':{'properties':{'state':{'minItems':8,'maxItems':8}}},
      'output':{'properties':{'output':{'minItems':1,'maxItems':40,'items':{'minItems':8,'maxItems':8}}}},
      'image':{'mode':'RGB','formats':['PNG'],'max_pixels':16000000,'max_aspect_ratio':20},
      'text':{'max_chars':4096}}


@pytest.fixture
def sample_request(config):
    state = RobotState({g:np.array([.0,.5,.5,-.35,0,0,1.]) for g in config['robot']['group_dims']},100,0)
    obs = ObservationSnapshot(state, {'d405_right':SensorFrame('d405_right', np.zeros((720,1280,3),np.uint8),100,0)})
    return InferenceRequest('s1',2,100,10**12,obs,'Put the bottles into the bin.')


def completion(rows):
    return {'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','content':None,'tool_calls':[
      {'id':'call_test','type':'function','function':{'name':'execute_actions','arguments':json.dumps({'output':rows})}}]}}],
      'metadata':{'action_shape':json.dumps([len(rows),8])}}


@pytest.fixture
def model(config, schema, monkeypatch):
    monkeypatch.setenv('PLATFORM_API_KEY','private-test-key')
    model = PokePolicyModel(config['policy'])
    def transport(path, payload=None):
        if path == '/models': return json.dumps({'data':[{'id':'poke-aurora-preview'}]}), None
        if path.endswith('/schema'): return json.dumps(schema), None
        state = json.loads(payload['messages'][0]['content'][0]['text'])['state']
        return json.dumps(completion([state]*40)), 'provider-sample_request-id'
    model._request = transport
    model.reset('s1')
    yield model
    model.close()


def test_calibrated_roundtrip_and_passive_arm(config, sample_request, model):
    adapter = build_policy_adapter(config['robot'], config['policy'])
    encoded = adapter.prepare_request(sample_request)
    raw = model.infer(encoded)
    adapter.decode_pose_only = True
    chunk = adapter.decode_action(raw,ActionContext(2,100,200,ik_seed=sample_request.observation.state))
    for group in sample_request.observation.state.groups:
        expected = adapter.kinematics.models[group].fk(sample_request.observation.state.groups[group])
        np.testing.assert_allclose(pose_matrix(chunk.groups[group][0,:7]), expected, atol=1e-10)
    assert chunk.horizon_steps == 40 and chunk.dt_ns == 33333333
    assert chunk.metadata['held_groups'] == ['left_arm']
    assert raw['sampling_stats']['poke_api']['request_id'] == 'provider-sample_request-id'


def test_actual_yam_ik_recovers_observation_and_rejects_unreachable(config, sample_request, model):
    adapter = build_policy_adapter(config['robot'], config['policy'])
    raw = model.infer(adapter.prepare_request(sample_request))
    context = ActionContext(2,100,200,ik_seed=sample_request.observation.state)
    chunk = adapter.decode_action(raw,context)
    for group, rows in chunk.groups.items():
        np.testing.assert_allclose(rows, np.tile(sample_request.observation.state.groups[group],(40,1)), atol=1e-3)
    raw['actions'][0,0] += 100
    with pytest.raises(ValueError,match='IK failed.*row=0'):
        adapter.decode_action(raw,context)


def test_image_preserves_aspect_ratio_and_only_selected_camera(config, sample_request, model):
    r = build_policy_adapter(config['robot'],config['policy']).prepare_request(sample_request)
    messages, obs = model.codec.encode(r)
    content = messages[0]['content']
    image = Image.open(io.BytesIO(base64.b64decode(content[1]['image_url']['url'].split(',')[1])))
    assert image.size == (640,360) and len(content) == 2
    assert obs['state'] == r.model_state


@pytest.mark.parametrize('field',['model_from_base','flange_from_tcp','flange_from_model_tool'])
def test_missing_transform_fails(config,field):
    config['policy']['adapter'][field] = None
    with pytest.raises(ValueError,match=field): build_policy_adapter(config['robot'],config['policy'])


@pytest.mark.parametrize('value,col',[(float('nan'),0),(float('inf'),2),(0,6),(1.2,7)])
def test_invalid_response_preserved_before_decode(config, sample_request, model, value, col):
    rows = deepcopy([[0,0,0,0,0,0,1,1]] * 40)
    rows[3][col] = value
    wire = json.dumps(completion(rows))
    model._request = lambda *args: (wire, 'bad-sample_request')
    request = build_policy_adapter(config['robot'], config['policy']).prepare_request(sample_request)
    with pytest.raises(ValueError):
        model.infer(request)
    saved = json.loads(next(Path(model.response_log_dir).rglob("*.json")).read_text())
    assert saved["stage"] == "received_before_validation"
    assert saved["raw_response"]["choices"] == json.loads(wire)["choices"]

def test_error_is_single_attempt_and_secret_redacted(config,sample_request,model):
    calls=[]
    def fail(*args):
        calls.append(1); raise TimeoutError('private-test-key timeout')
    model._request=fail
    r=build_policy_adapter(config['robot'],config['policy']).prepare_request(sample_request)
    with pytest.raises(RuntimeError) as e:model.infer(r)
    assert len(calls)==1 and 'private-test-key' not in str(e.value)
    assert '[REDACTED]' in str(e.value)


def test_malformed_json_is_saved(config, sample_request, model):
    model._request = lambda *args: ('broken JSON private-test-key', 'broken-sample_request')
    request = build_policy_adapter(config['robot'], config['policy']).prepare_request(sample_request)
    with pytest.raises(RuntimeError):
        model.infer(request)
    text = next(Path(model.response_log_dir).rglob("*.json")).read_text()
    assert 'broken JSON [REDACTED]' in text and 'private-test-key' not in text

def test_reset_fences_old_request(config,sample_request,model):
    r=build_policy_adapter(config['robot'],config['policy']).prepare_request(sample_request)
    model.reset('s2')
    with pytest.raises(ValueError,match='active session'):model.infer(r)
    model.infer(replace(r,session_id='s2'))


def test_sampler_rejected_before_transport(config,sample_request,model):
    r=build_policy_adapter(config['robot'],config['policy']).prepare_request(sample_request)
    r=SimpleNamespace(**{name:getattr(r,name) for name in ['session_id']},action_condition=np.zeros((2,8)))
    with pytest.raises(ValueError,match='does not support'):model.infer(r)


def test_delivered_shape_and_gripper_tolerance(config,schema):
    codec=PokeCodec(config['policy']);codec.bind_schema(schema)
    raw=completion([[0,0,0,0,0,0,1,1.005]])
    result=codec.decode(raw)
    assert result['actions'][0,7]==1 and result['sampling_stats']['poke_api']['gripper_clipped_rows']==[0]
    raw['metadata']['action_shape']='[40,8]'
    with pytest.raises(ValueError,match='action_shape'):codec.decode(raw)
