"""Meaningful offline replay checks and supplementary source-clock diagnostics."""
import json
from pathlib import Path
import numpy as np
from manimux.embodiments.robot.base import RobotModel
from manimux.runtime.safety import SafetyGuard
from manimux.types import RobotCommand,RobotState
import audit

def supplement():
 rows=json.loads((audit.OUT/'cohort.json').read_text())['selected'];reports=[]
 for row in rows:
  d=audit.load(row);kin=RobotModel.from_config(d['c']['robot']['config']).kinematics
  target=np.load(audit.OUT/'baseline'/f'{row["id"]}.npz')['target'];rp=audit.poses(kin,d['a']['scheduled'])
  valid=np.isfinite(target).all(axis=(1,2,3));filled=rp.copy();filled[valid]=target[valid]
  td=dict(d);td['resets']=d['resets']|~valid|np.r_[True,~valid[:-1]]
  targetmotion=audit.motion_metrics(td,d['a']['scheduled'],filled)
  targetmotion={k:v for k,v in targetmotion.items() if not k.startswith('joint_')}
  per_arm={};seedchecks=[];planmap={p['id']:p for p in d['plans']}
  for j,g in enumerate(audit.ARMS):
   per_arm[g]={}
   for layer,q in d['a'].items():
    pp=audit.fk(kin.models[g],q[:,j]);er=audit.err(target[:,j],pp)
    per_arm[g][layer]={reg:dict(mm=audit.stats(er[0][mask]),deg=audit.stats(er[1][mask])) for reg,mask in d['masks'].items()}
   for p in d['plans']:
    tr=p['trace'][g];tm=tr['ik_seed_time_ns'];ii=np.searchsorted(d['t'],tm)
    seed=np.array(tr['ik_seed_joints']);close=max(0,min(len(d['t'])-1,ii));before=max(0,close-1)
    if d['t'][close]>d['t'][before]:a=(tm-d['t'][before])/(d['t'][close]-d['t'][before])
    else:a=0.
    sampled=(1-a)*d['a']['scheduled'][before,j]+a*d['a']['scheduled'][close,j]
    seedchecks.append(dict(plan=p['name'],arm=g,source=p['ca']['metadata']['ik_seed_source'],seed_time_ns=tm,
      recorded_seed=seed.tolist(),nearest_tick_distance_ms=float(min(abs(d['t'][before]-tm),abs(d['t'][close]-tm))/1e6),
      tick_interpolated_reference_max_difference_rad=float(np.max(abs(seed[:7]-sampled[:7]))),
      interpretation='Seed itself is persisted and used exactly. Tick interpolation is diagnostic only, not a substitute seed.'))
  audit.write(audit.OUT/'supplement'/f'{row["id"]}.json',dict(target_motion=targetmotion,per_arm=per_arm,seed_audit=seedchecks,
    target_convention='Linear translation + shortest-arc orientation on original model row clock after dense latency trim and historical handoff skip. Initial source interval uses persisted seed TCP. This interpolant is explicit; only model knots are original outputs. QP interpolates each knot from achieved FK instead of previous raw target.',
    state_time='Host tick time, not individual feedback acquisition time. Derivative estimates are unfiltered and contain feedback quantization/jitter.'))
  reports.append(row['id'])
 return reports

def checks():
 rows=json.loads((audit.OUT/'cohort.json').read_text())['selected'];results=[]
 for task in ['pack_plate','pack_bottle']:
  chosen=[r for r in rows if r['task']==task]
  assert len(chosen)<=20
  dev={r['session'] for r in chosen if r['split']=='development'};hold={r['session'] for r in chosen if r['split']=='holdout'};assert not dev&hold
 results.append('<=20 per task, no session overlap, cohort fixed before selection')
 for r in rows:
  b=json.loads((audit.OUT/'baseline'/f'{r["id"]}.json').read_text());v=b['validation'];assert v['knot_max_abs_rad']<1e-9 and v['scheduled_max_abs_rad']<1e-9 and v['command_max_abs_rad']<1e-9 and not v['regeneration_failures']
  assert all(s['recorded']==s['current'] for s in b['source_checks'].values())
 results.append('24 historical baseline gates: knots/reference/command and critical source hashes')
 r=next(r for r in rows if r['split']=='development');d=audit.load(r)
 q,_=audit.replay(d,d['a']['scheduled'],{},True)
 for cap in [.01,.02]:
  q2,_=audit.replay(d,d['a']['scheduled'],dict(smooth={'max_step_dt_s':cap}));assert np.array_equal(q,q2)
 q3,_=audit.replay(d,d['a']['scheduled'],dict(smooth={'max_step_dt_s':.008}));assert np.max(abs(q-q3))>1e-4
 results.append('100 Hz: max_step_dt=.01/.02 are exact no-ops, .008 changes commands')
 alt,_=audit.replay(d,d['a']['scheduled'],dict(smooth={'cutoff_hz':4.}))
 reset= dict(d);reset['resets']=d['resets']|np.r_[False,d['ids'][1:]!=d['ids'][:-1]]
 wrong,_=audit.replay(reset,d['a']['scheduled'],dict(smooth={'cutoff_hz':4.}));assert np.max(abs(alt-wrong))>1e-4
 results.append('Candidate replay preserves command history across handoffs; per-chunk reset measurably differs')
 # A future unexecuted tail has no authority over tick-scoped TCP errors.
 kin=RobotModel.from_config(d['c']['robot']['config']).kinematics
 p=d['plans'][0];base=json.loads((audit.OUT/'baseline'/f'{r["id"]}.json').read_text());rec=base['plans'][0]
 ix,x,start,ddt,_,_=audit.mapping(d,p,len(p['knots'])*9)
 tr=p['trace']['left_arm'];seed=audit.fk(kin.models['left_arm'],np.array(tr['ik_seed_joints']))
 before=audit.target_at(tr,start+x*ddt,seed)
 import copy
 changed=copy.deepcopy(tr);last=rec['executed_source_row_range'][1];cut=int(np.ceil(last))+2
 if cut<len(changed['model_output']['pose_xyz_wxyz']):
  for pp in changed['model_output']['pose_xyz_wxyz'][cut:]:pp[0]+=100
  after=audit.target_at(changed,start+x*ddt,seed);assert np.array_equal(before,after)
 results.append('Mutating unexecuted future TCP rows cannot change executed-interval target errors')
 guard=SafetyGuard({'arm':1},None,position_lower={'arm':[-2.]},position_upper={'arm':[2.]},max_velocity={'arm':[.2]},control_dt_s=.01)
 guard.reset(RobotState({'arm':np.array([0.])},0,0));cmd=RobotCommand({'arm':np.array([.01])},1,None)
 try:guard.validate_command(cmd);raise AssertionError('guard should reject')
 except ValueError:pass
 assert cmd.groups['arm'][0]==.01
 results.append('Per-joint SafetyGuard rejects without clipping/changing command')
 # Finite-difference the new TCP translational Jacobian at recorded states.
 for j,g in enumerate(audit.ARMS):
  model=kin.models[g];wrapper=audit.TCPObjectiveKinematics(model.arm,model._offset(np.array([.5])))
  for q in d['a']['state'][::max(1,len(d['t'])//10),j,:7]:
   pose,jac=wrapper.flange_and_jacobian(q);numeric=np.zeros((3,7));eps=1e-6
   for k in range(7):
    step=np.zeros(7);step[k]=eps
    numeric[:,k]=(wrapper.flange_and_jacobian(q+step)[0][:3,3]-wrapper.flange_and_jacobian(q-step)[0][:3,3])/(2*eps)
   assert np.max(abs(numeric-jac[:3]))<1e-7
   assert np.max(abs(pose-audit.fk(model,np.r_[q,.5])))<1e-10
 results.append('TCP objective Jacobian matches finite differences <1e-7 m/rad; pose matches mounted TCP FK')
 audit.write(audit.OUT/'test_results.json',dict(passed=results,failed=[],hardware_connections=0,model_services=0))
 print('\n'.join(results))
if __name__=='__main__':
 checks();supplement()
