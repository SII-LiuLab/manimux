"""Persist selected counterfactual traces and verify their cached scores.

Never treats recorded state as counterfactual state. Original seeds stay frozen.
"""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import audit
from manimux.embodiments.robot.base import RobotModel

def export(row):
 d=audit.load(row);kin=RobotModel.from_config(d['c']['robot']['config']).kinematics
 locked=json.loads((audit.OUT/'locked_candidates.json').read_text())[row['task']]
 choice=next(c for c in locked if c['id']!='baseline')
 reference,target,plans,diag,failures=audit.regenerate(d,kin,choice);assert not failures
 command,flags=audit.replay(d,reference,choice)
 baseline_target=np.load(audit.OUT/'baseline'/f'{row["id"]}.npz')['target']
 assert np.array_equal(target,baseline_target,equal_nan=True)
 metrics=audit.score(d,kin,reference,command,target)
 cached=json.loads((audit.OUT/'candidates'/row['id']/f'{choice["id"]}.json').read_text())
 for key in ['target_reference_all_mm','target_reference_all_deg','target_command_all_mm','target_command_all_deg']:
  assert abs(metrics[key]['p95']-cached['metrics'][key]['p95'])<1e-10,(row['id'],key)
 path=audit.OUT/'selected_traces'/f'{row["id"]}.npz';path.parent.mkdir(exist_ok=True)
 np.savez_compressed(path,monotonic_ns=d['t'],plan_id=d['ids'],active=d['active'],reset=d['resets'],
   original_target_tcp=target,recorded_scheduled=d['a']['scheduled'],counterfactual_reference=reference,
   recorded_command=d['a']['command'],counterfactual_command=command,recorded_state=d['a']['state'],
   executor_flags_speed_acceleration=flags,pre_handoff=d['masks']['pre'],post_handoff=d['masks']['post'],stable=d['masks']['stable'])
 audit.write(path.with_suffix('.json'),dict(candidate=choice,episode=row['path'],npz_sha256=audit.digest(path.read_bytes()),
   score_reproduction_max_tolerance=1e-10,recorded_state_is_not_counterfactual=True,seed_mode='recorded per chunk',
   plans=plans,ik_executed_velocity_active_fraction=diag[:,:,:7].mean(axis=(0,1)).tolist(),
   ik_executed_position_active_fraction=diag[:,:,7:14].mean(axis=(0,1)).tolist(),ik_executed_j67_active_fraction=float(diag[:,:,14].mean())))
 return row['id']
if __name__=='__main__':
 rows=json.loads((audit.OUT/'cohort.json').read_text())['selected']
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(export,r) for r in rows]):print(f.result(),flush=True)
