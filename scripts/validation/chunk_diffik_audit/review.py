"""Aggregate episode metrics without pooling sessions, configurations or outcomes.

Selection reads development candidate files only. Holdout is evaluated once after
locked_candidates.json has been written; baseline holdout diagnostics do not tune.
"""
import argparse, json, math, hashlib, platform, subprocess, sys
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import audit
O=audit.OUT
MOTION_KEYS=['joint_rad_velocity_all','joint_rad_acceleration_all','joint_rad_jerk_all','tcp_m_velocity_all','tcp_m_acceleration_all','tcp_m_jerk_all','tcp_rad_velocity_all','tcp_rad_acceleration_all','tcp_rad_jerk_all',
 'tcp_m_jerk_pre','tcp_m_jerk_post','tcp_m_jerk_stable','tcp_rad_jerk_pre','tcp_rad_jerk_post','tcp_rad_jerk_stable']
KEYS=['target_reference_all_mm','target_reference_all_deg','reference_command_all_mm','reference_command_all_deg','target_command_all_mm','target_command_all_deg']
def flat(r):
 m=r['metrics'];v={k:m[k]['p95'] for k in KEYS}
 for k in MOTION_KEYS:v[k]=m['motion']['command'][k].get('p95',float('nan'))
 for reg in ['pre','post','stable']:
  for layer in ['target_reference','reference_command','target_command']:
   for unit in ['mm','deg']:
    k=f'{layer}_{reg}_{unit}';v[k]=m[k].get('p95',float('nan'))
 return v

def records(split):
 c=json.loads((O/'cohort.json').read_text())['selected'];out=[]
 for r in c:
  if r['split']!=split:continue
  for f in (O/'candidates'/r['id']).glob('*.json'):
   v=json.loads(f.read_text());out.append(dict(row=r,result=v))
 return out

def aggregate(items,group_fields):
 groups=defaultdict(list)
 for x in items:
  if 'metrics' not in x['result']:continue
  k=tuple(x['row'][f] for f in group_fields)+(x['result']['id'],);groups[k].append(x)
 out=[]
 for key,rows in groups.items():
  sessions=defaultdict(list)
  for x in rows:sessions[x['row']['session']].append(flat(x['result']))
  def finite_mean(v):
   a=np.asarray([x for x in v if x is not None and np.isfinite(x)])
   return float(a.mean()) if len(a) else None
  values={k:finite_mean([finite_mean([v[k] for v in vv]) for vv in sessions.values()]) for k in KEYS+MOTION_KEYS}
  out.append(dict(zip(group_fields,key[:-1]),candidate=key[-1],episodes=len(rows),sessions=len(sessions),eligible=all(x['result']['eligible'] for x in rows),metrics=values))
 return out

def joint():
 items=records('development');agg=aggregate(items,['task']);audit.write(O/'single_development_aggregate.json',agg)
 # Conservative, prospectively stated joint design: all available speed interventions
 # paired with the tested acceleration/filter values. No holdout read here.
 choices={}
 for task in ['pack_plate','pack_bottle']:
  taskrows=[x for x in agg if x['task']==task];base=next(x for x in taskrows if x['candidate']=='baseline')
  speed=[x for x in taskrows if x['candidate'].startswith('shared_v_') and x['metrics']['target_command_all_mm']<base['metrics']['target_command_all_mm'] and x['metrics']['target_command_all_deg']<=base['metrics']['target_command_all_deg']]
  # Pick the lowest speed that improves both TCP metrics. Retain the next speed
  # as a tradeoff if available. This is based on development, never holdout.
  speed=sorted(speed,key=lambda x:float(x['candidate'].split('_')[-1]))[:2]
  vals=[float(x['candidate'].split('_')[-1]) for x in speed]
  choices[task]=[dict(id=f'joint_v{v}_a{a}_f{f}',velocity=v,smooth={'max_acceleration':a,'cutoff_hz':f}) for v in vals for a in (8.,20.) for f in (4.,8.)]
 audit.write(O/'joint_candidates.json',choices)
 audit.write(O/'joint_rationale.json',dict(rule='Development only: lowest two tested shared velocities improving both translation and rotation; cross with acceleration 8/20 and filter 4/8. Physical bounds, scheduling and seeds unchanged.',selected=choices))
 print(choices)

def lock():
 items=records('development');agg=aggregate(items,['task']);strata=aggregate(items,['task','stratum','status'])
 chosen={};review={}
 for task in ['pack_plate','pack_bottle']:
  rs=[x for x in agg if x['task']==task];base=next(x for x in rs if x['candidate']=='baseline');b=base['metrics'];eligible=[]
  count=base['episodes']
  for r in rs:
   m=r['metrics']
   if r['candidate']=='baseline' or not r['eligible'] or r['episodes']!=count:continue
   if all(abs(m[k]/b[k]-1)<1e-8 for k in ['target_command_all_mm','target_command_all_deg','tcp_m_jerk_all','tcp_rad_jerk_all']):continue
   if m['target_command_all_mm']>b['target_command_all_mm'] or m['target_command_all_deg']>b['target_command_all_deg']:continue
   if m['tcp_m_jerk_all']>b['tcp_m_jerk_all']*1.05 or m['tcp_rad_jerk_all']>b['tcp_rad_jerk_all']*1.05:continue
   eligible.append(r)
  # Explicit multiobjective score, using TCP errors first, then jerk. It is an
  # offline review rank, not a task-success prediction.
  def rank(r):
   m=r['metrics'];return .4*m['target_command_all_mm']/b['target_command_all_mm']+.4*m['target_command_all_deg']/b['target_command_all_deg']+.1*m['tcp_m_jerk_all']/b['tcp_m_jerk_all']+.1*m['tcp_rad_jerk_all']/b['tcp_rad_jerk_all']
  eligible.sort(key=rank)
  best=eligible[0] if eligible else base
  frozen=[r for r in eligible if not r['candidate'].startswith(('joint_','shared_','lambda_','nullspace_','position_','j67_','tcp'))]
  frozen_best=frozen[0] if frozen else base
  choices={x['result']['id']:x['result']['choice'] for x in items if x['row']['task']==task}
  ids=list(dict.fromkeys(['baseline',frozen_best['candidate'],best['candidate']]))
  chosen[task]=[choices[c] for c in ids]
  review[task]=dict(selected=best['candidate'],frozen_reference_selected=frozen_best['candidate'],gate='Both TCP error p95 no worse than baseline, linear/angular TCP jerk p95 <=105% baseline, all development episodes present and all physical guard checks pass.',
    aggregation='Average episode p95 within session, then equal session mean. Completed and aborted, config/checkpoint/schedule strata remain separately reported.',
    score='0.4 translation ratio + 0.4 rotation ratio + 0.1 linear jerk ratio + 0.1 angular jerk ratio',ranked=eligible,baseline=base,
    warning='Selected for offline review only. Original targets, seeds, acceptance times and active intervals frozen. No plant or new measured state. Singleton strata and outcome distributions confound transfer.')
 audit.write(O/'locked_candidates.json',chosen);audit.write(O/'selection_development_only.json',review);audit.write(O/'development_stratified.json',strata)
 print({k:[c['id'] for c in v] for k,v in chosen.items()})

def rates():
 cohort=json.loads((O/'cohort.json').read_text())['selected'];out=[];seen=set()
 for r in cohort:
  if r['session'] in seen:continue
  seen.add(r['session']);d=audit.load(r);c=d['c'];s=c['executor']['smooth'];ratio=c['robot']['options']['hardware'].get('velocity_ratio',32);rated=math.pi
  # Profiles are already resolved and control_profile=None. Do not invent an
  # authored command_margin for manually overridden historical motion rates.
  examples=[]
  for rr in (32,40,50):
   for margin in (.5,.7,.9):
    examples.append(dict(velocity_ratio=rr,command_margin=margin,controller_cap_rad_s=rated*rr/100,
      linked_shared_max_velocity_rad_s=rated*rr/100*margin,requires_new_ik=True,
      exceeds_frozen_command_guard=rated*rr/100*margin>min(c['executor']['command_safety']['max_velocity']['left_arm'][:7])))
  q=d['a']['command'];guard=[]
  for scale in (.5,.8,1.,1.2):
   for wrist_only in (False,True):
    caps=np.stack([c['executor']['command_safety']['max_velocity'][g][:7] for g in audit.ARMS]);caps[:,5:]*=scale
    if not wrist_only:caps[:,:5]*=scale
    v=np.abs(np.diff(q[:,:,:7],axis=0)/d['dt']);valid=~d['resets'][1:]
    guard.append(dict(scale=scale,wrist_only=wrist_only,rejected_tick_count=int(np.any(v>caps[None]+1e-9,axis=(1,2))[valid].sum()),command_change_if_accepted=0.,effect='reject only, never clips or accelerates'))
  out.append(dict(session=r['session'],task=r['task'],shared_max_velocity=s['max_velocity'],configured_hardware_velocity_ratio=ratio,
    authored_command_margin=None,effective_shared_to_hardware_cap_ratio=s['max_velocity']/(rated*ratio/100),
    frozen_guard_velocity=c['executor']['command_safety']['max_velocity'],frozen_control_profile=c['control_profile'],
    nominal_dt_s=d['dt'],max_step_dt_s=s['max_step_dt_s'],step_budget_factor=min(d['dt'],s['max_step_dt_s'])/d['dt'],
    profile_formula_scenarios=examples,guard_only_tests=guard))
 audit.write(O/'rate_contract_review.json',dict(sessions=out,notes=[
 'The recorded robot_backend reports controller_metadata_not_supported; velocity_ratio is configured identity, not a saved hardware readback. Hardware velocity_ratio and acceleration_ratio change the physical controller. With the resolved shared rate held fixed they do not alter this offline IK/executor replay.',
 'Fresh resolve_rate_contract uses rated velocity * velocity_ratio/100 for command_safety; shared IK/executor rate additionally multiplies command_margin. Historical snapshots retain manual shared limits and must not be overwritten with today\'s profile formula.',
 'No authored per-joint QP velocity option: QP has seven independent box inequalities using shared scalar vmax. SafetyGuard per-joint rates reject; SmoothExecutor mode changes arm-wide scaling versus independent clipping.',
 'max_step_dt_s >=0.01 is inactive at frozen 100 Hz; 0.008 reduces executor budget only, not IK, hence adds mismatch.',
 'DiffIK max_lag_mm=5 under lag_policy=report is diagnostic. Under abort it rejects a solved chunk. Neither setting improves the solution.',
 'Physical joint bounds and J6/J7 interference check unchanged. Only tighter margin probes were permitted.',
 'Tianji hardware.max_tracking_error_deg is a refusal threshold for command/measured discrepancy, not a controller gain. The analytic IK max_step_deg branch detector is not the chunk DiffIK velocity budget. Neither was relaxed.',
 'Hardware acceleration_ratio=100 is not the software max_acceleration in rad/s^2 and has no calibrated offline plant model.']))

if __name__=='__main__':
 mode=sys.argv[1]
 {'joint':joint,'lock':lock,'rates':rates}[mode]()
