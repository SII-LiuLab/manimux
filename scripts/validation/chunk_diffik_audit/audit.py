"""Read-only, CPU-only chunk DiffIK audit. Never constructs a Robot/controller.

Outputs are derived evidence. Commands are counterfactual open-loop commands, not
predicted measured states. Recorded decode seeds and commit clocks stay frozen.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, platform, subprocess
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import zarr
from scipy.spatial.transform import Rotation, Slerp
from manimux.embodiments.robot.base import RobotModel
from manimux.policy_adapter.umi_dp.tianji import UmiDpTianjiAdapter
from manimux.runtime.executors.smooth import SmoothExecutor
from manimux.types import ActionHorizon, RobotState

ROOT=Path(__file__).resolve().parents[3]
CAMPAIGN=ROOT/'data/experiments/umi-dp-bottle-plate-scratch-tf32-20261017'
OUT=Path(os.environ.get('CHUNK_DIFFIK_OUTPUT',str(ROOT/'data/analysis/chunk_diffik_audit_20261019'))).resolve()
ARMS=('left_arm','right_arm')
SOURCE=['manimux/policy_adapter/umi_dp/tianji.py','manimux/kinematics/tianji_diff.py',
'manimux/runtime/timeline.py','manimux/runtime/edge.py','manimux/runtime/executors/smooth.py',
'manimux/runtime/executors/limits.py','manimux/runtime/safety.py',
'manimux/embodiments/arm/tianji/kinematics.py','manimux/kinematics/composed.py',
'manimux/configs/embodiment/robot/tianji_taccap_pack_plate.yaml',
'manimux/configs/embodiment/arm/tianji_left.yaml','manimux/configs/embodiment/arm/tianji_right.yaml',
'manimux/configs/embodiment/end_effector/taccap.yaml']
def digest(b): return hashlib.sha256(b).hexdigest()
def js_sha(x): return digest(json.dumps(x,sort_keys=True).encode())
def write(path,x):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def stats(a):
 a=np.asarray(a).ravel();a=a[np.isfinite(a)]
 return dict(n=len(a),mean=float(a.mean()),p50=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else dict(n=0)
def key(e):return e.parent.parent.name+'__'+e.parent.name+'__'+e.name

def inventory():
 write(OUT/'tcp_candidates.json',[dict(id='tcp_objective',objective='tcp')])
 selected=[];excluded=[]
 for task in ('pack_plate','pack_bottle'):
  rows=[]
  paths=sorted(CAMPAIGN.glob('*chunk/session-*/rollout-*'),key=lambda p:(p.parent.name,p.name),reverse=True)
  for p in paths:
   meta=json.loads((p/'meta.json').read_text());m=json.loads((p.parent/'session-manifest.json').read_text());c=m['config']
   if meta.get('task')!=task:continue
   reason=None
   try:
    z=zarr.open_group(str(p/'data.zarr'),mode='r');n=len(z['ticks/monotonic_ns'])
    assert c['execution']['ik']['mode']=='chunk' and c['policy']['adapter']['ik_backend']=='diff','not_chunk_diff'
    assert n>=10 and len(list(z['plans'].group_keys()))>0,'no_executed_plan'
    for name in z['plans'].group_keys():
     a=dict(z['plans'][name]['canonical_raw'].attrs)
     assert 'ee_target_trace' in a['metadata'],'missing_original_target_or_seed'
    if len(rows)>=20:reason='beyond_recent_20'
   except Exception as exc: reason=str(exc)
   if reason:excluded.append(dict(path=str(p.relative_to(ROOT)),reason=reason));continue
   res=json.loads((p/'result.json').read_text());s=c['executor']['smooth'];model=c['policy']['expected_backend']['model']
   identity=dict(algorithm=c['inference']['algorithm'],checkpoint=model['checkpoint_sha256'],
     smooth=s,diff=c['policy']['adapter']['diff_ik'],skip=c['inference']['handoff_skip_steps'],
     policy_dt=c['policy']['action_dt_s'],robot=c['robot'],safety=c['executor']['command_safety'],
     gripper_mapping=c['policy']['adapter']['gripper_mapping'])
   rows.append(dict(path=str(p.relative_to(ROOT)),id=key(p),task=task,session=p.parent.name,
      status='complete' if res['run_completed'] and not p.name.endswith('.partial') else 'aborted',
      success='unlabelled',stratum=js_sha(identity)[:12],identity=identity,ticks=n))
  newest=max(r['session'] for r in rows)
  for r in rows:r['split']='holdout' if r['session']==newest else 'development'
  selected.extend(rows)
 write(OUT/'cohort.json',dict(selection='Latest <=20 valid recorded chunk DiffIK episodes per task, before metrics; no outcome filtering.',
  split_rule='Latest nonempty session per task held out, all other sessions development. Singleton/config-shift strata cannot support within-stratum generalization.',selected=selected,excluded=excluded))
 return selected

def load(r):
 p=ROOT/r['path'];m=json.loads((p.parent/'session-manifest.json').read_text());c=m['config']
 z=zarr.open_group(str(p/'data.zarr'),mode='r');t=z['ticks/monotonic_ns'][:];ids=z['ticks/plan_id'][:].astype(str)
 a={k:np.stack([z[f'ticks/{k}/{g}'][:] for g in ARMS],axis=1) for k in ('state','scheduled','command')}
 assert np.all(np.diff(t)>0) and all(np.isfinite(v).all() for v in a.values())
 events=[json.loads(x) for x in (p/'events.jsonl').read_text().splitlines()];bounds={e['plan_id']:e for e in events if e['kind']=='plan_boundary'}
 active=(ids!='') & np.any(abs(a['scheduled']-a['state'])>1e-10,axis=(1,2))
 resets=np.r_[True,np.diff(t)>250_000_000]
 for e in events:
  if e['kind']=='robogui_control_state' and e.get('paused'):
   i=np.searchsorted(t,e['monotonic_ns'])
   if i<len(t):resets[i]=True
 masks={'all':active & ~resets};seam=np.zeros(len(t),bool);pre=seam.copy();post=seam.copy()
 transitions=np.flatnonzero((ids[1:]!=ids[:-1]) & active[1:] & (ids[:-1]!=''))+1
 for i in transitions:
  pre|=(t>=t[i]-100_000_000)&(t<t[i]);post|=(t>=t[i])&(t<t[i]+100_000_000)
 masks.update(pre=pre&masks['all'],post=post&masks['all'],stable=~(pre|post)&masks['all'],hold=~active&~resets)
 plans=[]
 for name in sorted(z['plans'].group_keys()):
  pg=z['plans'][name];pid=pg.attrs['plan_id'];ca=dict(pg['canonical_raw'].attrs);co=dict(pg['committed'].attrs)
  assert not ca.get('hold_from_step') and not c['inference']['blend_policy_steps']
  trace=ca['metadata']['ee_target_trace'];assert all(len(trace[g]['ik_input']['pose_xyz_wxyz'])==len(trace[g]['model_output']['pose_xyz_wxyz']) for g in ARMS),'waypoint_unsupported'
  plans.append(dict(name=name,id=pid,ca=ca,co=co,trace=trace,boundary=bounds[pid],
    knots=np.stack([pg['canonical_raw'][g][:] for g in ARMS],axis=1)))
 return dict(row=r,path=p,manifest=m,c=c,z=z,t=t,ids=ids,a=a,events=events,active=active,resets=resets,masks=masks,plans=plans,dt=1/c['robot']['control_hz'])

def fk(model,q):
 """Vectorized exact modified-DH FK; independently checked against public SDK FK."""
 shape=q.shape[:-1];q=q.reshape(-1,8);v=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
 for j,(alpha,a,d,theta0) in enumerate(model.arm._dh):
  alpha=np.radians(alpha);theta=np.radians(theta0)+(q[:,j] if j<7 else 0);ct=np.cos(theta);st=np.sin(theta);ca=np.cos(alpha);sa=np.sin(alpha)
  l=np.zeros_like(v);l[:,0,0]=ct;l[:,0,1]=-st;l[:,0,3]=a*.001;l[:,1,0]=st*ca;l[:,1,1]=ct*ca;l[:,1,2]=-sa;l[:,1,3]=-sa*d*.001;l[:,2,0]=st*sa;l[:,2,1]=ct*sa;l[:,2,2]=ca;l[:,2,3]=ca*d*.001;l[:,3,3]=1;v=v@l
 offset=model._offset(np.array([.5]));assert np.allclose(offset,model._offset(np.array([0.]))) and np.allclose(offset,model._offset(np.array([1.])))
 return (v@offset).reshape(*shape,4,4)
def poses(kin,q):return np.stack([fk(kin.models[g],q[:,j]) for j,g in enumerate(ARMS)],axis=1)
def err(a,b):
 shape=a.shape[:-2];valid=np.isfinite(a).all(axis=(-2,-1))&np.isfinite(b).all(axis=(-2,-1));p=np.full(shape,np.nan);r=p.copy()
 p[valid]=np.linalg.norm(a[valid,:3,3]-b[valid,:3,3],axis=-1)*1000
 r[valid]=np.degrees(Rotation.from_matrix(np.swapaxes(a[valid,:3,:3],-1,-2)@b[valid,:3,:3]).magnitude())
 return p,r

def mapping(d,p,n):
 ca,co=p['ca'],p['co'];dt=ca['dt_ns'];ddt=ca['metadata']['runtime_trajectory_dt_ns'];start=ca['observation_time_ns']-dt+ddt
 now=p['boundary']['monotonic_ns'];serial=d['c']['inference']['action_start_mode']=='first_step_when_ready'
 if serial:i0=0;clock=now
 else:i0=max(0,(now-start+ddt-1)//ddt);clock=start+i0*ddt
 skip=round(co['handoff_skipped_steps']*dt/ddt);begin=i0+skip
 ix=np.flatnonzero((d['ids']==p['id'])&d['active']);x=(d['t'][ix]-clock)/ddt+begin
 # Dense rows were not persisted. Only ticks with an actually scheduled reference are mapped.
 assert np.all((x>=begin-1e-8)&(x<=(n if serial else n-1)+1e-8)),(p['name'],x.min(),x.max(),n,serial)
 return ix,np.minimum(x,n-1),start,ddt,begin,clock

def interp(values,x):
 j=np.minimum(np.floor(x).astype(int),len(values)-1);k=np.minimum(j+1,len(values)-1);w=x-j
 return values[j]*(1-w).reshape((-1,)+(1,)*(values.ndim-1))+values[k]*w.reshape((-1,)+(1,)*(values.ndim-1))
def target_at(trace,x_ns,seed_pose):
 tr=trace['model_output'];rows=np.asarray(tr['pose_xyz_wxyz']);ts=np.arange(len(rows))*tr['dt_ns'];xt=x_ns-tr['start_time_ns']
 # Seed-to-first interpolation is a declared continuous target convention, not a saved model row.
 rot=Rotation.from_quat(rows[:,[4,5,6,3]]);pp=rows[:,:3]
 if np.any(xt<0):
  ts=np.r_[-tr['dt_ns'],ts];pp=np.vstack([seed_pose[:3,3],pp]);rot=Rotation.from_matrix(np.concatenate([seed_pose[None,:3,:3],rot.as_matrix()]))
 xt=np.clip(xt,ts[0],ts[-1]);out=np.broadcast_to(np.eye(4),(len(xt),4,4)).copy()
 for j in range(3):out[:,j,3]=np.interp(xt,ts,pp[:,j])
 out[:,:3,:3]=Slerp(ts/1e9,rot)(xt/1e9).as_matrix();return out

class TCPObjectiveKinematics:
 """Offline QP geometry at the mounted TCP; all physical constraints delegated.

 For fixed flange-to-TCP offset r, Jv_tcp = Jv_flange + omega cross r.
 This changes the translational objective, not the target or joint bounds.
 """
 def __init__(self,arm,offset):self.arm=arm;self.offset=offset
 def __getattr__(self,name):return getattr(self.arm,name)
 def flange_and_jacobian(self,q):
  flange,j=self.arm.flange_and_jacobian(q);j=j.copy();r=flange[:3,:3]@self.offset[:3,3]
  j[:3]+=np.cross(j[3:].T,r).T
  return flange@self.offset,j

def adapter_for(d,kin,choice):
 c=d['c'];pol=copy.deepcopy(c['policy']);motion=copy.deepcopy(c['executor']['motion_limits'])
 if 'velocity' in choice:motion['arm']['max_velocity']=choice['velocity']
 pol['adapter']['diff_ik'].update(choice.get('diff',{}))
 adapter=UmiDpTianjiAdapter(c['robot'],pol,kinematics=kin,motion_limits=motion)
 if choice.get('objective')=='tcp':
  for side,solver in adapter.diff_solvers.items():
   model=kin.models[f'{side}_arm'];offset=model._offset(np.array([.5]))
   assert np.array_equal(offset,model._offset(np.array([0.]))) and np.array_equal(offset,model._offset(np.array([1.])))
   solver.kinematics=TCPObjectiveKinematics(model.arm,offset)
  adapter._diff_target=lambda model,target,aperture: target
 return adapter

def regenerate(d,kin,choice):
 adapter=adapter_for(d,kin,choice);ref=d['a']['scheduled'].copy();target=np.full((len(ref),2,4,4),np.nan);records=[];diagnostics=[];failures=[]
 for p in d['plans']:
  rows=[];dense=[];logs=[]
  for j,g in enumerate(ARMS):
   side=g.removesuffix('_arm');tr=p['trace'][g];steps=[{f'{side}_ee_pose':np.array(pose),f'{side}_ee_joint_state':np.array([grip])} for pose,grip in zip(tr['ik_input']['pose_xyz_wxyz'],tr['ik_input']['aperture'])]
   solver=adapter.diff_solvers[side];base_solve=solver.solve;log=[]
   def tracked(t,q,dt):
    result=base_solve(t,q,dt)
    if result.joints is not None:
     qd=np.degrees(result.joints);margin=np.minimum(solver.upper-qd,qd-solver.lower)
     log.append(np.r_[np.abs(result.qdot_rad_s)>=np.radians(solver.vmax)*.999,margin<=solver.margin+.02,
       110.5-1.025*abs(qd[5])-abs(qd[6])-solver.j67_margin<=.02,
       np.min(margin),110.5-1.025*abs(qd[5])-abs(qd[6]),np.linalg.norm(solver._nullspace_gradient(qd))>1e-12])
    return result
   solver.solve=tracked
   try:r,den,_,_=adapter._decode_side(side,steps,np.asarray(tr['ik_seed_joints']))
   except Exception as exc:
    failures.append(dict(plan=p['name'],arm=g,error=str(exc)));r=None;den=None
   finally:solver.solve=base_solve
   rows.append(r);dense.append(den);logs.append(np.asarray(log))
  if any(v is None for v in dense):continue
  dense=np.stack(dense,axis=1);rows=np.stack(rows,axis=1);ix,x,start,ddt,begin,clock=mapping(d,p,len(dense))
  ref[ix]=interp(dense,x)
  for j,g in enumerate(ARMS):target[ix,j]=target_at(p['trace'][g],start+x*ddt,fk(kin.models[g],np.asarray(p['trace'][g]['ik_seed_joints'])))
  touched=np.unique(np.r_[np.floor(x).astype(int),np.ceil(x).astype(int)])
  if len(touched):diagnostics.append(np.stack(logs,axis=1)[touched])
  records.append(dict(plan=p['name'],plan_id=p['id'],executed_ticks=len(ix),dense_start=start,dense_dt_ns=ddt,
    executed_dense_index_range=[float(x.min()),float(x.max())] if len(x) else None,
    executed_source_row_range=[float((start+x.min()*ddt-p['ca']['observation_time_ns'])/p['ca']['dt_ns']),float((start+x.max()*ddt-p['ca']['observation_time_ns'])/p['ca']['dt_ns'])] if len(x) else None,
    runtime_start_index=int(begin),runtime_start_clock=int(clock),time_trimmed_steps=p['co']['time_trimmed_steps'],handoff_skipped_steps=p['co']['handoff_skipped_steps'],
    knot_max_abs_rad=float(np.max(abs(rows[...,:7]-p['knots'][...,:7])))))
 return ref,target,records,np.concatenate(diagnostics) if diagnostics else np.empty((0,2,18)),failures

def replay(d,ref,choice,public=False):
 s=copy.deepcopy(d['c']['executor']['smooth']);s.update(choice.get('smooth',{}))
 if 'velocity' in choice:s['max_velocity']=choice['velocity']
 dt=d['dt'];alpha=dt/(dt+1/(2*np.pi*s['cutoff_hz']));vm=s['max_velocity']*min(1,(s['max_step_dt_s'] or dt)/dt);am=s['max_acceleration'] or np.inf
 q=d['a']['command'][0,:,:7].copy();v=np.zeros_like(q);out=d['a']['command'].copy();flags=np.zeros((len(ref),2,2),bool)
 if public:
  ex=SmoothExecutor(s,dt);ex.reset(RobotState({g:d['a']['command'][0,j].copy() for j,g in enumerate(ARMS)},int(d['t'][0]),0))
 def bound(x,limit):
  return x*np.minimum(1,limit/np.maximum(abs(x).max(axis=-1,keepdims=True),1e-300)) if s['mode']=='isotropic' else np.clip(x,-limit,limit)
 for i in range(len(ref)):
  if d['resets'][i]:q=d['a']['command'][i,:,:7].copy();v[:]=0
  elif not d['active'][i]:v[:]=0
  else:
   wanted=alpha*(ref[i,:,:7]-q)/dt;flags[i,:,0]=np.any(abs(wanted)>vm+1e-10,axis=-1);limited=bound(wanted,vm);nv=v+bound(limited-v,am*dt);flags[i,:,1]=np.any(abs(nv-limited)>1e-10,axis=-1);q=np.clip(q+nv*dt,-s['position_limit_abs'],s['position_limit_abs']);v=nv
  out[i,:,:7]=q
  if public:
   state=RobotState({g:out[i,j].copy() for j,g in enumerate(ARMS)},int(d['t'][i]),i)
   if d['resets'][i]:ex.reset(state);pc=out[i]
   elif not d['active'][i]:pc=np.stack([ex.hold(int(d['t'][i]),state).groups[g] for g in ARMS])
   else:
    h=ActionHorizon(int(d['t'][i]),round(dt*1e9),d['ids'][i],{g:ref[i,j][None] for j,g in enumerate(ARMS)})
    cmd=ex.step(int(d['t'][i]),state,h);pc=np.stack([cmd.groups[g] for g in ARMS])
   assert np.max(abs(pc[:,:7]-out[i,:,:7]))<1e-12,'kernel/public executor mismatch'
 return out,flags

def motion_metrics(d,q,pose):
 t=d['t']/1e9;n=len(t);valid=~d['resets'];valid[0]=False
 # Backward differences use recorded host tick spacing; no resampling/smoothing.
 dt=np.diff(t);jv=np.zeros((n,2,7));jv[1:]=np.diff(q[:,:,:7],axis=0)/dt[:,None,None]
 tv=np.zeros((n,2,3));tv[1:]=np.diff(pose[:,:,:3,3],axis=0)/dt[:,None,None]
 rv=np.zeros((n,2,3));r=pose[:,:,:3,:3];rv[1:]=Rotation.from_matrix((r[1:]@np.swapaxes(r[:-1],-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(n-1,2,3)/dt[:,None,None]
 metrics={};series={'joint_rad':jv,'tcp_m':tv,'tcp_rad':rv}
 for order in ('velocity','acceleration','jerk'):
  for name,v in series.items():
   magnitude=np.max(abs(v),axis=-1) if name=='joint_rad' else np.linalg.norm(v,axis=-1)
   for region,mask in d['masks'].items():metrics[f'{name}_{order}_{region}']=stats(magnitude[mask&valid])
  series={name:np.concatenate([np.zeros_like(v[:1]),np.diff(v,axis=0)/dt[:,None,None]]) for name,v in series.items()};valid &= np.r_[False,valid[:-1]]
 return metrics

def score(d,kin,reference,command,target,include_state=False):
 ps={'reference':poses(kin,reference),'command':poses(kin,command)}
 if include_state:ps['state']=poses(kin,d['a']['state'])
 out={};pairs=[('target_reference',target,ps['reference']),('reference_command',ps['reference'],ps['command']),('target_command',target,ps['command'])]
 if include_state:pairs.extend([('command_state',ps['command'],ps['state']),('target_state',target,ps['state'])])
 for name,a,b in pairs:
  pe,re=err(a,b)
  for region,mask in d['masks'].items():
   out[f'{name}_{region}_mm']=stats(pe[mask]);out[f'{name}_{region}_deg']=stats(re[mask])
 out['motion']={k:motion_metrics(d,reference if k=='reference' else command if k=='command' else d['a']['state'],p) for k,p in ps.items()}
 return out

def candidates():
 # These are single-factor interventions. No retained reference is reused when shared speed changes.
 return [dict(id='baseline'),*[dict(id=f'filter_{x}',smooth={'cutoff_hz':x}) for x in (4.,12.,20.)],
 *[dict(id=f'accel_{x}',smooth={'max_acceleration':x}) for x in (8.,20.,40.)],dict(id='per_joint',smooth={'mode':'per_joint'}),
 *[dict(id=f'step_dt_{x}',smooth={'max_step_dt_s':x}) for x in (.008,.01,.02)],
 *[dict(id=f'shared_v_{x}',velocity=x) for x in (.3,.5,.65,.8)],
 *[dict(id=f'lambda_{x}',diff={'lam':x}) for x in (.0001,.01)],
 *[dict(id=f'nullspace_{x}',diff={'mu_nullspace':x}) for x in (0.,3000.)],
 dict(id='position_margin_8',diff={'limit_margin_deg':8.}),dict(id='j67_margin_8',diff={'j67_margin_deg':8.})]

def baseline_job(r):
 d=load(r);kin=RobotModel.from_config(d['c']['robot']['config']).kinematics
 fkcheck=[]
 for j,g in enumerate(ARMS):
  q=d['a']['state'][::max(1,len(d['t'])//20),j];fkcheck.append(max(float(np.max(abs(a-kin.models[g].fk(b)))) for a,b in zip(fk(kin.models[g],q),q)))
 ref,target,plans,diag,fails=regenerate(d,kin,{});cmd,flags=replay(d,d['a']['scheduled'],{},public=True)
 active=d['active'];validation=dict(fk_public_max_abs=max(fkcheck),knot_max_abs_rad=max(p['knot_max_abs_rad'] for p in plans),
  scheduled_max_abs_rad=float(np.max(abs(ref[active,:,:7]-d['a']['scheduled'][active,:,:7]))),
  command_max_abs_rad=float(np.max(abs(cmd[active,:,:7]-d['a']['command'][active,:,:7]))),regeneration_failures=fails,public_executor_check=True)
 source=d['manifest']['source_provenance']['repositories']['manimux']['files'];checks={f:dict(recorded=source.get(f,{}).get('sha256'),current=digest((ROOT/f).read_bytes())) for f in SOURCE}
 metrics=score(d,kin,d['a']['scheduled'],d['a']['command'],target,True)
 lag={}
 for j,g in enumerate(ARMS):
  losses=[]
  for ms in range(0,201,5):
   times=d['t']-ms*1_000_000;past=np.stack([np.interp(times-d['t'][0],d['t']-d['t'][0],d['a']['command'][:,j,k]) for k in range(7)],axis=1)
   mask=d['masks']['stable']&(times>=d['t'][0]);losses.append(float(np.sqrt(np.mean((past[mask]-d['a']['state'][mask,j,:7])**2))))
  best=int(np.argmin(losses));lag[g]=dict(best_delay_ms=best*5,zero_delay_joint_rmse_rad=losses[0],best_joint_rmse_rad=losses[best],note='Descriptive alignment only; ticks precede dispatch and per-arm feedback acquisition timestamps are missing.')
 guard=d['c']['executor']['command_safety'];limits=[]
 for j,g in enumerate(ARMS):
  q=d['a']['scheduled'][:,j,:7];lower,upper=kin.models[g].arm.joint_position_limits();limits.append(dict(arm=g,physical_margin_deg=stats(np.degrees(np.minimum(q-lower,upper-q))[active]),j67_gap_deg=stats((110.5-1.025*abs(np.degrees(q[:,5]))-abs(np.degrees(q[:,6])))[active]),safety_velocity_rad_s=guard['max_velocity'][g][:7]))
 digest_input=hashlib.sha256()
 for file in [d['path'].parent/'session-manifest.json',d['path']/'meta.json',d['path']/'events.jsonl',d['path']/'result.json']:digest_input.update(file.read_bytes())
 for a in [d['t'],*d['a'].values()]:digest_input.update(a.tobytes())
 for p in d['plans']:digest_input.update(json.dumps(p['ca'],sort_keys=True).encode());digest_input.update(p['knots'].tobytes())
 report=dict(**r,validation=validation,metrics=metrics,plans=plans,source_checks=checks,git_sha=d['manifest']['git_sha'],input_sha256=digest_input.hexdigest(),
  raw_config_sha256=js_sha(d['c']),config_path=d['manifest']['config_path'],entry_yaml_sha256=d['manifest']['config_sha256'],
  tick_interval_ms=stats(np.diff(d['t'])/1e6),active_ticks=int(active.sum()),holds=int((~active).sum()),lag=lag,limits=limits,
  executor_speed_limited_fraction=float(flags[active,:,0].mean()),executor_acceleration_limited_fraction=float(flags[active,:,1].mean()),
  ik_executed_dense_samples=int(len(diag)),ik_velocity_active_fraction_per_joint=diag[:,:,:7].mean(axis=(0,1)).tolist(),
  ik_position_active_fraction_per_joint=diag[:,:,7:14].mean(axis=(0,1)).tolist(),ik_j67_active_fraction=float(diag[:,:,14].mean()),
  ik_nullspace_active_fraction=float(diag[:,:,17].mean()),
  abort_events=[e for e in d['events'] if e['kind']=='episode_aborted'],
  sent_command_valid={g:int(d['z'][f'ticks/sent_command/{g}/valid'][:].sum()) for g in ARMS},
  label=json.loads((d['path']/'evaluation/human-label.json').read_text()) if (d['path']/'evaluation/human-label.json').exists() else None)
 write(OUT/'baseline'/f'{r["id"]}.json',report)
 np.savez_compressed(OUT/'baseline'/f'{r["id"]}.npz',target=target,reference=ref)
 return dict(id=r['id'],validation=validation)

def candidate_job(args):
 r,choices=args;d=load(r);kin=RobotModel.from_config(d['c']['robot']['config']).kinematics
 baseline=json.loads((OUT/'baseline'/f'{r["id"]}.json').read_text());v=baseline['validation']
 if v['scheduled_max_abs_rad']>1e-5 or v['command_max_abs_rad']>1e-8 or v['regeneration_failures']:return dict(id=r['id'],error='baseline_gate_failed')
 target=np.load(OUT/'baseline'/f'{r["id"]}.npz')['target'];result=[];ik_cache={}
 for choice in choices:
  cid=choice['id'];out=OUT/'candidates'/r['id']/f'{cid}.json'
  if out.exists():continue
  if 'velocity' in choice or 'diff' in choice or choice.get('objective')=='tcp':
   cachekey=json.dumps({k:choice[k] for k in ('velocity','diff','objective') if k in choice},sort_keys=True)
   if cachekey not in ik_cache:
    regenerated=regenerate(d,kin,choice);ik_cache[cachekey]=(regenerated[0],regenerated[3],regenerated[4])
   ref,diag,fails=ik_cache[cachekey]
   if fails:
    write(out,dict(id=cid,choice=choice,failures=fails,eligible=False));continue
  else:ref=d['a']['scheduled'];diag=None
  cmd,flags=replay(d,ref,choice);metrics=score(d,kin,ref,cmd,target)
  safety=[]
  for j,g in enumerate(ARMS):
   lower=np.array(d['c']['executor']['command_safety']['position_lower'][g][:7]);upper=np.array(d['c']['executor']['command_safety']['position_upper'][g][:7]);vel=np.diff(cmd[:,j,:7],axis=0)/d['dt'];mask=~d['resets'][1:]
   safety.append(dict(arm=g,position_violations=int(np.any((cmd[:,j,:7]<lower-.001)|(cmd[:,j,:7]>upper+.001),axis=1).sum()),
    velocity_violations=int(np.any(abs(vel)>np.array(d['c']['executor']['command_safety']['max_velocity'][g][:7])+1e-9,axis=1)[mask].sum()),
    j67_violations=int(np.sum(1.025*abs(np.degrees(cmd[:,j,5]))+abs(np.degrees(cmd[:,j,6]))>110.5))))
  write(out,dict(id=cid,choice=choice,eligible=not any(s['position_violations'] or s['velocity_violations'] or s['j67_violations'] for s in safety),
    metrics=metrics,safety=safety,executor_speed_limited_fraction=float(flags[d['active'],:,0].mean()),executor_acceleration_limited_fraction=float(flags[d['active'],:,1].mean()),
    intervention='original TCP + recorded IK seeds + new DiffIK + persistent executor' if diag is not None else 'frozen recorded scheduled reference + persistent executor',
    measured_state_prediction=None))
  result.append(cid)
 return dict(id=r['id'],completed=result)

def main():
 global OUT
 pa=argparse.ArgumentParser();pa.add_argument('phase',choices=['inventory','baseline','single','joint','tcp','tcp_joint','holdout']);pa.add_argument('--workers',type=int,default=3);pa.add_argument('--limit',type=int);pa.add_argument('--output',type=Path,default=OUT);args=pa.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
 if args.phase=='inventory':print(json.dumps(inventory(),ensure_ascii=False));return
 rows=json.loads((OUT/'cohort.json').read_text())['selected']
 if args.limit:rows=rows[:args.limit]
 if args.phase=='baseline':fn=baseline_job;jobs=rows
 else:
  if args.phase=='single':choices=candidates();rows=[r for r in rows if r['split']=='development']
  else:
   filename={'joint':'joint_candidates.json','tcp':'tcp_candidates.json','tcp_joint':'tcp_joint_candidates.json','holdout':'locked_candidates.json'}[args.phase]
   choices=json.loads((OUT/filename).read_text());rows=[r for r in rows if r['split']==('holdout' if args.phase=='holdout' else 'development')]
  fn=candidate_job;jobs=[(r,choices[r['task']] if isinstance(choices,dict) else choices) for r in rows]
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  for f in as_completed([pool.submit(fn,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
if __name__=='__main__':main()
