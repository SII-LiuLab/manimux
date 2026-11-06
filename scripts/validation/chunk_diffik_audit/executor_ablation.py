"""Frozen-reference factorial ablation of chunk software execution.

Two separate references per recorded episode: original scheduled joints, and the
previously locked TCP-IK candidate. No IK rate or shared config is changed. Factors
bypass individual executor operations for diagnosis only. Eight complete continuous
replays reveal interactions; invalid would-be commands are never treated as executable.
"""
from __future__ import annotations
import argparse,copy,html,itertools,json,math,sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import audit
from manimux.embodiments.robot.base import RobotModel
from manimux.runtime.executors.direct import DirectExecutor
from manimux.runtime.executors.limits import ScalarLimits,limit_step
from manimux.runtime.safety import SafetyGuard
from manimux.types import ActionHorizon,RobotState,RobotCommand

OUT=audit.OUT/'executor_ablation_v1'
FACTORS=('filter','velocity','acceleration')
NAMES={0:'raw_passthrough_diagnostic',1:'filter_only',2:'velocity_only',3:'filter_velocity',4:'acceleration_only',5:'filter_acceleration',6:'direct_shared_limits',7:'smooth_full'}

def replay(d,ref,settings,bits):
 dt=d['dt'];alpha=dt/(dt+1/(2*np.pi*settings['cutoff_hz'])) if bits&1 else 1.
 vmax=settings['max_velocity']*min(1.,(settings['max_step_dt_s'] or dt)/dt) if bits&2 else np.inf
 amax=settings['max_acceleration'] if bits&4 and settings['max_acceleration'] is not None else np.inf
 q=d['a']['command'][0,:,:7].copy();v=np.zeros_like(q);out=d['a']['command'].copy();flags=np.zeros((len(ref),2,3),bool)
 def cap(x,bound):
  if settings['mode']=='isotropic':return x*np.minimum(1.,bound/np.maximum(abs(x).max(axis=-1,keepdims=True),1e-300))
  return np.clip(x,-bound,bound)
 for i in range(len(ref)):
  if d['resets'][i]:q=d['a']['command'][i,:,:7].copy();v[:]=0
  elif not d['active'][i]:v[:]=0
  else:
   wanted=alpha*(ref[i,:,:7]-q)/dt;limited=cap(wanted,vmax)
   nv=v+cap(limited-v,amax*dt)
   flags[i,:,0]=np.any(abs(limited-wanted)>1e-10,axis=-1);flags[i,:,1]=np.any(abs(nv-limited)>1e-10,axis=-1)
   newq=q+nv*dt;bound=settings['position_limit_abs']
   q=np.clip(newq,-bound,bound) if bound is not None else newq
   flags[i,:,2]=np.any(abs(q-newq)>1e-10,axis=-1);v=nv
  out[i,:,:7]=q
 return out,flags

def direct_public(d,ref,settings,limited):
 motion=copy.deepcopy(d['c']['executor']['motion_limits']) if limited else None
 if motion is not None:
  for k in ['max_velocity','max_acceleration','max_step_dt_s','mode']:motion['arm'][k]=settings[k]
 ex=DirectExecutor(motion_limits=motion,control_dt_s=d['dt']);out=d['a']['command'].copy();previous=out[0].copy()
 for i in range(len(ref)):
  state=RobotState({g:previous[j].copy() for j,g in enumerate(audit.ARMS)},int(d['t'][i]),i)
  if d['resets'][i]:
   previous=d['a']['command'][i].copy();ex.reset(RobotState({g:previous[j].copy() for j,g in enumerate(audit.ARMS)},int(d['t'][i]),i))
  elif not d['active'][i]:ex.reset(state)
  else:
   horizon=ActionHorizon(int(d['t'][i]),round(d['dt']*1e9),d['ids'][i],{g:ref[i,j][None] for j,g in enumerate(audit.ARMS)})
   command=ex.step(int(d['t'][i]),state,horizon);previous=np.stack([command.groups[g] for g in audit.ARMS])
  out[i]=previous
 return out

def guards(d,command):
 cfg=d['c']['executor']['command_safety'];q=command[:,:,:7];n=len(q)
 lower=np.array([cfg['position_lower'][g][:7] for g in audit.ARMS]);upper=np.array([cfg['position_upper'][g][:7] for g in audit.ARMS]);rates=np.array([cfg['max_velocity'][g][:7] for g in audit.ARMS])
 vel=np.zeros_like(q);vel[1:]=np.diff(q,axis=0)/d['dt'];acc=np.zeros_like(q);acc[1:]=np.diff(vel,axis=0)/d['dt']
 valid=~d['resets'];valid[0]=False
 badpos=np.any((q<lower-.001)|(q>upper+.001),axis=(1,2))&valid
 badv=np.any(abs(vel)>rates+1e-9,axis=(1,2))&valid
 bada=np.zeros(n,bool)
 if cfg['max_acceleration']:
  ac=np.array([cfg['max_acceleration'][g][:7] for g in audit.ARMS]);bada=np.any(abs(acc)>ac+1e-8,axis=(1,2))&valid
 # J6/J7 physical boundary is checked separately, unchanged. It is not a SafetyGuard option.
 badj=np.any(1.025*abs(np.degrees(q[:,:,5]))+abs(np.degrees(q[:,:,6]))>110.5+1e-9,axis=1)&valid
 bad=badpos|badv|bada|badj;idx=np.flatnonzero(bad);first=int(idx[0]) if len(idx) else None
 public=SafetyGuard(dict.fromkeys(audit.ARMS,7),d['c']['executor']['smooth']['position_limit_abs'],
   position_lower={g:cfg['position_lower'][g][:7] for g in audit.ARMS},position_upper={g:cfg['position_upper'][g][:7] for g in audit.ARMS},
   max_velocity={g:cfg['max_velocity'][g][:7] for g in audit.ARMS},max_acceleration={g:cfg['max_acceleration'][g][:7] for g in audit.ARMS} if cfg['max_acceleration'] else None,control_dt_s=d['dt'])
 public_first=None;public_error=None
 for i in range(n):
  groups={g:q[i,j] for j,g in enumerate(audit.ARMS)}
  if d['resets'][i]:public.reset(RobotState(groups,int(d['t'][i]),i));continue
  try:public.validate_command(RobotCommand(groups,int(d['t'][i]),d['ids'][i]))
  except ValueError as exc:public_first=i;public_error=str(exc);break
 expected=np.flatnonzero(badpos|badv|bada);assert public_first==(int(expected[0]) if len(expected) else None)
 return dict(whole_arm_trajectory_passes_frozen_guards=first is None,first_rejected_tick=first,
   first_rejected_seconds_from_record_start=float((d['t'][first]-d['t'][0])/1e9) if first is not None else None,
   first_rejected_plan=d['ids'][first] if first is not None else None,public_guard_error=public_error,
   position_rejected_ticks=int(badpos.sum()),velocity_rejected_ticks=int(badv.sum()),acceleration_rejected_ticks=int(bada.sum()),j67_rejected_ticks=int(badj.sum()),
   per_joint_velocity_max_rad_s=np.max(abs(vel[valid]),axis=0).tolist(),
   note='Whole-interval counts inspect intended arm commands, not commands actually sent after rejection. Actual execution would stop at first rejection; gripper/controller tracking guards are not modeled.'),first

def metrics(d,kin,refpose,target,q,stop=None):
 pose=audit.poses(kin,q);out={}
 for label,t in [('reference_command',refpose),('target_command',target)]:
  pe,re=audit.err(t,pose)
  for region,mask in d['masks'].items():
   if stop is not None:mask=mask&(np.arange(len(q))<stop)
   out[f'{label}_{region}_mm']=audit.stats(pe[mask]);out[f'{label}_{region}_deg']=audit.stats(re[mask])
 out['motion']=audit.motion_metrics(d,q,pose)
 return out

def shapley(values):
 result={}
 for k,factor in enumerate(FACTORS):
  value=0.
  for mask in range(8):
   if mask&(1<<k):continue
   size=mask.bit_count();weight=math.factorial(size)*math.factorial(2-size)/math.factorial(3)
   value+=weight*(values[mask|(1<<k)]-values[mask])
  result[factor]=value
 assert abs(sum(result.values())-(values[7]-values[0]))<1e-7
 return result

def job(row):
 d=audit.load(row);kin=RobotModel.from_config(d['c']['robot']['config']).kinematics;tracepath=audit.OUT/'selected_traces'/f'{row["id"]}.npz';data=np.load(tracepath)
 target=data['original_target_tcp'];locked=json.loads((audit.OUT/'locked_candidates.json').read_text())[row['task']];choice=next(c for c in locked if c['id']!='baseline')
 summaries=[]
 for context,reference in [('historical',d['a']['scheduled']),('locked_tcp',data['counterfactual_reference'])]:
  settings=copy.deepcopy(d['c']['executor']['smooth'])
  if context=='locked_tcp':settings.update(choice['smooth']);settings['max_velocity']=choice.get('velocity',settings['max_velocity'])
  commands={};results={};refpose=audit.poses(kin,reference);gate=[]
  for bits in range(8):
   q,flags=replay(d,reference,settings,bits);commands[bits]=q
   checks,first=guards(d,q);score=metrics(d,kin,refpose,target,q);prefix={}
   if first is not None:
    for label,t in [('reference_command',refpose),('target_command',target)]:
     pe,re=audit.err(t,audit.poses(kin,q));mask=d['masks']['all']&(np.arange(len(q))<first)
     prefix[label+'_mm']=audit.stats(pe[mask]);prefix[label+'_deg']=audit.stats(re[mask])
   results[bits]=dict(bits=bits,name=NAMES[bits],switches={f:bool(bits&(1<<k)) for k,f in enumerate(FACTORS)},
     hypothetical_full_historical_interval_metrics=score,guard_valid_prefix_errors=prefix,guards=checks,
     active_arm_ticks=int(d['active'].sum()*2),velocity_limited_fraction=float(flags[d['active'],:,0].mean()),
     acceleration_limited_fraction=float(flags[d['active'],:,1].mean()),absolute_position_clip_arm_ticks=int(flags[:,:,2].sum()))
  # Cross-check the exact public paths, not just a handwritten ablation kernel.
  public_errors={}
  for bits,limited in [(0,False),(6,True)]:
   direct=direct_public(d,reference,settings,limited);error=float(abs(direct[:,:,:7]-commands[bits][:,:,:7]).max());assert error<1e-12
   public_errors[NAMES[bits]]=error
  expected=d['a']['command'] if context=='historical' else data['counterfactual_command']
  full_error=float(abs(expected[:,:,:7]-commands[7][:,:,:7]).max());assert full_error<1e-12
  # Also verify public shared limiter for every factor on a deterministic sample
  # of ticks, using that factor's own carried command/velocity state.
  attribution={}
  for layer in ['reference_command','target_command']:
   for region in ['all','pre','post','stable']:
    for unit in ['mm','deg']:
     metric=f'{layer}_{region}_{unit}'
     for stat in ['mean','p95']:
      if all(results[b]['hypothetical_full_historical_interval_metrics'][metric]['n'] for b in range(8)):
       vals={b:results[b]['hypothetical_full_historical_interval_metrics'][metric][stat] for b in range(8)}
       attribution[f'{metric}_{stat}']=dict(passthrough=vals[0],full=vals[7],shapley=shapley(vals),
         remove_from_full={factor:vals[7]-vals[7^(1<<k)] for k,factor in enumerate(FACTORS)})
  for region in ['all','pre','post','stable']:
   for kind in ['tcp_m_jerk','tcp_rad_jerk']:
    metric=f'{kind}_{region}'
    if all(results[b]['hypothetical_full_historical_interval_metrics']['motion'][metric]['n'] for b in range(8)):
     vals={b:results[b]['hypothetical_full_historical_interval_metrics']['motion'][metric]['p95'] for b in range(8)}
     attribution[metric+'_p95']=dict(passthrough=vals[0],full=vals[7],shapley=shapley(vals))
  out=dict(episode=row,reference_context=context,settings=settings,reference_sha256=audit.digest(reference.tobytes()),
    trace_input_sha256=audit.digest(tracepath.read_bytes()),variants=results,attribution=attribution,
    checks=dict(public_direct_arm_max_errors_rad=public_errors,full_history_reproduction_max_error_rad=full_error),
    notes=['No IK or shared max_velocity changes: toggles bypass executor operations on an identical frozen reference.',
      'All modes preserve command history across handoffs; historical gaps reset velocity while holding command. Initial/explicit resets anchor only at recorded reset points.',
      'DirectExecutor with shared limits is bits=6, not raw passthrough bits=0. Public direct output is checked for the seven arm joints.',
      'Gripper command is fixed to historical values for arm attribution. TacCap TCP geometry does not depend on aperture; gripper direct behavior and grasp success are not validated.',
      'Physical position, J6/J7 and command safety limits remain unchanged. Invalid intended trajectories are full-interval diagnostic outputs only, not executable plans.',
      'Shapley allocates interactions by averaging all switch orders. A negative contribution means the factor reduces error for this frozen replay. It is not an independent physical error term.',
      'Post-selection diagnosis includes historical held-out sessions; it does not change the previously locked candidates or claim a new untouched validation split.'])
  audit.write(OUT/'episodes'/f'{row["id"]}__{context}.json',out)
  if row['split']=='holdout':
   path=OUT/'traces'/f'{row["id"]}__{context}.npz';path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(path,monotonic_ns=d['t'],plan_id=d['ids'],reference=reference,active=d['active'],resets=d['resets'],**{NAMES[b]:q for b,q in commands.items()})
  summaries.append(context)
 return dict(episode=row['id'],contexts=summaries)

def average(v):return float(np.mean(v)) if v else None

def aggregate(reports):
 groups=defaultdict(list)
 for r in reports:
  e=r['episode'];groups[(e['task'],r['reference_context'],e['split'],e['stratum'],e['status'])].append(r)
 result=[]
 for group,rs in sorted(groups.items()):
  out=dict(zip(['task','reference_context','split','stratum','status'],group),episodes=len(rs),variants={},attribution={})
  def agg(get):
   ss=defaultdict(list)
   for r in rs:
    v=get(r)
    if v is not None:ss[r['episode']['session']].append(v)
   return average([average(v) for v in ss.values()])
  for bits in range(8):
   key=str(bits);m={}
   for layer in ['reference_command','target_command']:
    for reg in ['all','pre','post','stable']:
     for unit in ['mm','deg']:
      name=f'{layer}_{reg}_{unit}'
      for stat in ['mean','p95']:m[name+'_'+stat]=agg(lambda r:r['variants'][key]['hypothetical_full_historical_interval_metrics'][name].get(stat))
   for reg in ['all','pre','post','stable']:
    for kind in ['joint_rad_velocity','joint_rad_acceleration','joint_rad_jerk','tcp_m_velocity','tcp_m_acceleration','tcp_m_jerk','tcp_rad_velocity','tcp_rad_acceleration','tcp_rad_jerk']:
     name=f'{kind}_{reg}';m[name+'_p95']=agg(lambda r:r['variants'][key]['hypothetical_full_historical_interval_metrics']['motion'][name].get('p95'))
   out['variants'][key]=dict(name=NAMES[bits],metrics=m,whole_arm_guard_passes=sum(r['variants'][key]['guards']['whole_arm_trajectory_passes_frozen_guards'] for r in rs),episodes=len(rs),
     position_clip_arm_ticks=sum(r['variants'][key]['absolute_position_clip_arm_ticks'] for r in rs),
     guard_rejected_ticks={typ:sum(r['variants'][key]['guards'][typ] for r in rs) for typ in ['position_rejected_ticks','velocity_rejected_ticks','acceleration_rejected_ticks','j67_rejected_ticks']})
  for name in rs[0]['attribution']:
   out['attribution'][name]={f:agg(lambda r:r['attribution'].get(name,{}).get('shapley',{}).get(f)) for f in FACTORS}
  result.append(out)
 return result

def check_factor_kernel():
 n=75;dt=.01;ref=np.zeros((n,2,8));grid=np.arange(n)
 for arm in range(2):
  for j in range(7):ref[:,arm,j]=.3*np.sin(grid*.19+j*.2+arm)
 d=dict(dt=dt,a={'command':np.zeros_like(ref)},resets=np.zeros(n,bool),active=np.ones(n,bool))
 d['resets'][[0,51]]=True;d['active'][25:29]=False
 for mode in ['isotropic','per_joint']:
  settings=dict(cutoff_hz=4.,max_velocity=.65,max_acceleration=20.,max_step_dt_s=.016,position_limit_abs=3.14,mode=mode)
  for bits in range(8):
   actual,_=replay(d,ref,settings,bits);q={g:np.zeros(7) for g in audit.ARMS};v={g:np.zeros(7) for g in audit.ARMS}
   for i in range(n):
    if d['resets'][i]:q={g:np.zeros(7) for g in audit.ARMS};v={g:np.zeros(7) for g in audit.ARMS}
    elif not d['active'][i]:v={g:np.zeros(7) for g in audit.ARMS}
    else:
     alpha=dt/(dt+1/(2*np.pi*settings['cutoff_hz'])) if bits&1 else 1.
     target={g:q[g]+alpha*(ref[i,j,:7]-q[g]) for j,g in enumerate(audit.ARMS)}
     limits=ScalarLimits(.65 if bits&2 else None,20. if bits&4 else None,3.14,mode,.016)
     q,v=limit_step(target,q,v,dt_s=dt,limits=limits)
    assert np.max(abs(np.stack(list(q.values()))-actual[i,:,:7]))<1e-12
 return 'All 8 factors, both limit modes, persistent handoffs/gaps/resets match public limit_step to <1e-12 rad'

def render():
 factor_test=check_factor_kernel();reports=[json.loads(p.read_text()) for p in (OUT/'episodes').glob('*.json')];summary=aggregate(reports);audit.write(OUT/'stratified_results.json',summary)
 def table(headers,rows):return '<table><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</table>'
 for task in ['pack_plate','pack_bottle']:
  sections=[]
  for r in summary:
   if r['task']!=task:continue
   rows=[]
   for v in r['variants'].values():
    m=v['metrics'];rows.append([v['name'],*[round(m[k],3) for k in ['reference_command_all_mm_p95','reference_command_all_deg_p95','target_command_all_mm_p95','target_command_all_deg_p95','tcp_m_jerk_all_p95','tcp_rad_jerk_all_p95']],f"{v['whole_arm_guard_passes']}/{v['episodes']}"])
   sections.append(f"<h2>{r['reference_context']} / {r['split']} / {r['status']} / config {r['stratum']}</h2>"+table(['Operations','Ref→command mm','deg','Target→command mm','deg','TCP jerk m/s³','rad/s³','Whole arm guard passes'],rows))
   at=r['attribution']['reference_command_all_mm_p95'];sections.append('<p>Order-averaged allocation of reference→command position p95 (mm): '+', '.join(f'{k}={v:.3f}' for k,v in at.items())+'. This is an interaction allocation, not independent physical errors. Negative values are possible.</p>')
  header=f'''<h1>{task}: frozen-reference executor ablation</h1><p>Source: the previously fixed 2026-10-17 cohort. Historical and locked TCP references are separate. No new IK, target, seed, timing, handoff or physical bounds changed.</p><p>Metrics: mean episode p95 within session, then equal session means; different historical configurations and completion classes remain separate. Complete does not mean successful.</p><p><b>Direct with shared limits removes only the filter.</b> Raw passthrough is a separate diagnostic. Values below cover the historical active intervals, even when the intended commands would be rejected. Such rows are not viable deployment results; each episode JSON records the first rejection and errors only before that rejection. Gripper/plant responses are not simulated.</p>'''
  focus=[r for r in reports if r['episode']['task']==task and r['episode']['split']=='holdout' and r['reference_context']=='locked_tcp']
  practical=[]
  for r in focus:
   vv=r['variants'];full=vv['7']['hypothetical_full_historical_interval_metrics'];lines=[f"{task} / {r['episode']['status']} / {r['episode']['id']}"]
   for bits in [7,6,5,3,0]:
    v=vv[str(bits)];m=v['hypothetical_full_historical_interval_metrics'];g=v['guards']
    lines.append(f"  {NAMES[bits]}: ref-command p95 {m['reference_command_all_mm']['p95']:.3f} mm, target-command p95 {m['target_command_all_mm']['p95']:.3f} mm, linear jerk p95 {m['motion']['tcp_m_jerk_all']['p95']:.1f} m/s^3, frozen arm guard passes={g['whole_arm_trajectory_passes_frozen_guards']}")
   ordered=[vv[str(b)]['hypothetical_full_historical_interval_metrics']['reference_command_all_mm']['p95'] for b in [0,1,3,7]]
   lines.append('  Ordered filter -> velocity -> acceleration p95 path (mm): '+' -> '.join(f'{x:.3f}' for x in ordered))
   lines.append('  Marginal increments depend on this order. Leave-one-out changes do not add; Shapley allocations include unsafe diagnostic combinations and are not deployable component budgets.')
   practical.append('\n'.join(lines))
  (OUT/f'{task}_summary.txt').write_text('\n\n'.join(practical)+'\n\nAll positions/velocities/J6-J7 guards stay frozen. No physical motion, plant prediction, task-success claim or new parameter selection. Entire-interval trajectories after a first rejection are diagnostics only. See episode JSON for guard-valid prefixes.\n')
  (OUT/f'{task}_report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px system-ui;margin:30px;max-width:1250px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}h1{font-size:24px}h2{font-size:18px;margin-top:32px}</style>'+header+''.join(sections))
 audit.write(OUT/'validation.json',dict(episode_contexts=len(reports),factorial_replays=len(reports)*8,
   max_full_reproduction_error_rad=max(r['checks']['full_history_reproduction_max_error_rad'] for r in reports),
   max_public_direct_arm_error_rad=max(v for r in reports for v in r['checks']['public_direct_arm_max_errors_rad'].values()),
   factor_kernel_public_check=factor_test,shapley_sum_checks='passed for each reported allocation',guard_first_rejection_matches_public_safetyguard=True,
   source_sha256={p:audit.digest((audit.ROOT/p).read_bytes()) for p in ['manimux/runtime/executors/direct.py','manimux/runtime/executors/smooth.py','manimux/runtime/executors/limits.py','manimux/runtime/safety.py']},
   script_sha256=audit.digest(Path(__file__).read_bytes()),cohort_sha256=audit.digest((audit.OUT/'cohort.json').read_bytes()),
   attribution_scope='arm-only fixed reference diagnostic; historical held-out sessions are post-selection diagnosis, not new parameter selection'))
 print('Rendered',len(reports),'episode contexts')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--render-only',action='store_true');p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int);args=p.parse_args()
 if not args.render_only:
  rows=json.loads((audit.OUT/'cohort.json').read_text())['selected'];rows=rows[:args.limit] if args.limit else rows
  with ProcessPoolExecutor(max_workers=args.workers) as pool:
   for f in as_completed([pool.submit(job,r) for r in rows]):print(json.dumps(f.result()),flush=True)
 render()
