"""Offline position-interface tracking audit; no robot drivers or service calls.

Run with the ManiMux Python environment. Outputs preserve cohort/provenance,
chronological splits, causal controller simulation and per-episode metrics.
Recorded command is a host reference, never assumed to be a hardware receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr
from scipy.signal import lfilter

DT = .01
RAD = 180 / np.pi
ARMS = ('left_arm', 'right_arm')
JOINTS = [f'{arm}/J{j+1}' for arm in ARMS for j in range(7)]


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                               else x.item() if isinstance(x, np.generic) else str(x)))


def rmse(x, axis=None):
    return np.sqrt(np.mean(np.asarray(x)**2, axis=axis))


def derivative(x, cutoff=8.):
    # Causal backward difference followed by one-pole filtering; no future samples.
    raw = np.diff(x, axis=0, prepend=x[:1]) / DT
    alpha = np.exp(-2*np.pi*cutoff*DT)
    return lfilter([1-alpha], [1, -alpha], raw, axis=0)


def load_cohort(root):
    manifests = sorted(root.glob('*/session-*/session-manifest.json'),
                       key=lambda p: p.parent.name)
    newest_day = max(p.parent.name[8:16] for p in manifests)
    rows, segments = [], []
    for manifest in manifests:
        if manifest.parent.name[8:16] != newest_day:
            continue
        m = json.loads(manifest.read_text())
        config = m['config']
        if not any(s in config['run']['task'].lower() for s in ('throw', 'toss')):
            continue
        for episode in sorted(manifest.parent.glob('rollout-*')):
            if not (episode/'result.json').exists():
                rows.append(dict(path=str(episode), eligible=False, reason='not_finalized'))
                continue
            result = json.loads((episode/'result.json').read_text())
            ticks = zarr.open_group(str(episode/'data.zarr'), mode='r')['ticks']
            stamp = np.asarray(ticks['monotonic_ns'])
            status = 'complete' if result.get('run_completed', False) else 'partial'
            entry = dict(path=str(episode), session=manifest.parent.name,
                         method=config['run']['condition'], status=status,
                         result_reason=result.get('detail', result.get('terminal_reason')),
                         tick_count=len(stamp), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                         config_sha256=m.get('config_sha256'), git_sha=m.get('git_sha'),
                         playback_hz=config['policy'].get('playback_hz'),
                         executor=config['executor']['type'], handoff=config['inference'].get('handoff'),
                         control_hz=config['robot']['control_hz'],
                         result_sha256=hashlib.sha256((episode/'result.json').read_bytes()).hexdigest())
            if len(stamp) < 3:
                entry.update(eligible=False, reason='empty_or_too_short')
                rows.append(entry)
                continue
            time = (stamp-stamp[0])*1e-9
            q, u, r = [np.concatenate([np.asarray(ticks[f'{stage}/{arm}'])[:, :7]
                                       for arm in ARMS], axis=1)
                       for stage in ('state', 'command', 'scheduled')]
            gripper = np.column_stack([ticks[f'state/{arm}'][:, 7] for arm in ARMS])
            valid = np.isfinite(q).all(1) & np.isfinite(u).all(1) & np.isfinite(r).all(1)
            if not np.all(np.diff(time)>0) or not valid.all():
                entry.update(eligible=False, reason='invalid_time_or_nonfinite')
                rows.append(entry)
                continue
            # Diagnose every finalized nonempty attempt, including failures.
            sent = [float(np.mean(ticks[f'sent_command/{arm}/valid'][:]))
                    if f'sent_command/{arm}/valid' in ticks else 0. for arm in ARMS]
            naive_velocity=np.diff(q,axis=0)/np.diff(time)[:,None]
            filtered_velocity=np.zeros_like(q)
            for k, dt in enumerate(np.diff(time),1):
                alpha=np.exp(-2*np.pi*8*dt)
                filtered_velocity[k]=alpha*filtered_velocity[k-1]+(1-alpha)*naive_velocity[k-1]
            entry.update(naive_velocity_p99_rad_s=float(np.quantile(np.abs(naive_velocity),.99)),
                         naive_velocity_peak_rad_s=float(np.max(np.abs(naive_velocity))),
                         filtered_velocity_p99_rad_s=float(np.quantile(np.abs(filtered_velocity),.99)),
                         filtered_velocity_peak_rad_s=float(np.max(np.abs(filtered_velocity))))
            digest=hashlib.sha256()
            for array in (stamp,q,u,r,gripper):digest.update(np.ascontiguousarray(array).tobytes())
            entry.update(eligible=True, duration_s=float(time[-1]),
                         analysis_input_arrays_sha256=digest.hexdigest(),
                         dt_ms_quantiles=np.quantile(np.diff(time)*1e3, [.01,.5,.95,.99]).tolist(),
                         gaps_over_50ms=int(np.sum(np.diff(time)>.05)),
                         sent_valid_fraction=sent,
                         command_error_deg=rmse(u-q, axis=0)*RAD,
                         shaping_error_deg=rmse(r-u, axis=0)*RAD,
                         reference_error_deg=rmse(r-q, axis=0)*RAD,
                         command_error_p95_deg=np.quantile(np.abs(u-q)*RAD,.95,axis=0),
                         repeated_state_fraction=[float(np.mean(np.all(np.diff(q[:,j:j+7],axis=0)==0,axis=1)))
                                                  for j in (0,7)])
            # Split by session (not by adjacent samples), never tune on latest test sessions.
            clock = manifest.parent.name[17:23]
            split = 'train' if clock < '133000' else 'validation' if clock < '134700' else 'test'
            entry['split'] = split if status == 'complete' else 'stress'
            bounds = config['executor']['command_safety']
            lower = np.concatenate([bounds['position_lower'][arm][:7] for arm in ARMS])
            upper = np.concatenate([bounds['position_upper'][arm][:7] for arm in ARMS])
            # No interpolation across pauses/stalls. Tiny intervals are collapsed by resampling.
            cuts = np.r_[0, np.flatnonzero(np.diff(time)>.05)+1, len(time)]
            segment_count_before = len(segments)
            for start, stop in zip(cuts[:-1],cuts[1:]):
                end = time[stop-1] - (.5 if status == 'partial' and stop == len(time) else 0)
                grid = np.arange(time[start],end,DT)
                if len(grid)<300:
                    continue
                sample = lambda values: np.column_stack([np.interp(grid,time[start:stop],values[start:stop,j])
                                                         for j in range(values.shape[1])])
                seg = dict(id=len(segments), episode=len(rows), time=grid, q=sample(q),
                           u=sample(u), reference=sample(r), gripper=sample(gripper),
                           split=entry['split'], lower=lower, upper=upper,
                           vmax=float(config['executor']['smooth']['max_velocity']),
                           amax=float(config['executor']['smooth']['max_acceleration']))
                segments.append(seg)
            entry["model_segments"] = len(segments)-segment_count_before
            rows.append(entry)
    return rows, segments


def plant_response(u, initial, order, tau, delay):
    """Open-loop free response of a stable delay + one/two-pole position model."""
    result = np.empty_like(u)
    for j in range(u.shape[1]):
        a = np.exp(-DT/max(tau[j],1e-9))
        # q[k+1] responds to u[k-delay]; initial history equals initial measured q.
        d = int(delay[j])+1
        shifted = np.r_[np.full(d, initial[j]), u[:-d,j]]
        for _ in range(order):
            shifted, _ = lfilter([1-a],[1,-a],shifted,zi=[a*initial[j]])
        result[:,j] = shifted
    return result


def episode_mse(errors, segments):
    by_episode = {}
    for error, seg in zip(errors,segments):
        by_episode.setdefault(seg['episode'],[]).append(np.mean(error[100:]**2,axis=0))
    return np.mean([np.mean(v,axis=0) for v in by_episode.values()],axis=0)


def identify(segments, order):
    taus = np.r_[.001, np.arange(.005,.101,.005)]
    delays = np.arange(0,13)
    best = np.full(14,np.inf); selected_tau=np.zeros(14);selected_delay=np.zeros(14,dtype=int)
    for delay in delays:
        for tau in taus:
            errors = [plant_response(s['u'],s['q'][0],order,np.full(14,tau),np.full(14,delay))-s['q']
                      for s in segments]
            score = episode_mse(errors,segments)
            improve=score<best
            best[improve]=score[improve];selected_tau[improve]=tau;selected_delay[improve]=delay
    return dict(order=order,tau=selected_tau,delay=selected_delay,train_rmse_deg=np.sqrt(best)*RAD)


def candidates():
    values=[dict(name='baseline',ff=0.,kv=0.,cutoff=8.)]
    for ff in [.02,.04,.06,.08,.10,.12]:
        values.append(dict(name=f'FF {ff*1000:g}ms',ff=ff,kv=0.,cutoff=8.))
    for kv in [.005,.01,.02,.04,.08]:
        values.append(dict(name=f'VFB {kv*1000:g}ms',ff=0.,kv=kv,cutoff=8.))
    for ff in [.02,.04,.06,.08,.10,.12]:
        for kv in [.005,.01,.02,.04]:
            values.append(dict(name=f'FF {ff*1000:g}+VFB {kv*1000:g}ms',ff=ff,kv=kv,cutoff=8.))
    return values


def simulate(seg, model, controllers, *, delay_extra=0, tau_scale=1., noise_deg=0.,
             sensor_delay=0, seed=17):
    """Counterfactual uses simulated feedback, NEVER frozen logged state feedback.

    All controllers act through position commands; grippers are excluded.
    One-second warmup is excluded from scores. Noise affects measurement only.
    """
    r=seg['u']; n,j=r.shape;c=len(controllers)
    tau=model['tau']*tau_scale; delay=np.maximum(0,model['delay']+delay_extra)
    a=np.exp(-DT/np.maximum(tau,1e-9));order=model['order']
    states=np.broadcast_to(seg['q'][0],(order,c,j)).copy()
    y=np.empty((n,c,j));cmd=np.empty_like(y)
    previous_measure=np.broadcast_to(seg['q'][0],(c,j)).copy()
    filtered_velocity=np.zeros((c,j));previous_cmd=np.broadcast_to(r[0],(c,j)).copy()
    previous_v=np.zeros((c,j)); rng=np.random.default_rng(seed)
    noise=rng.normal(0,noise_deg/RAD,(n,j))
    ref_vel=derivative(r)
    ff=np.array([p['ff'] for p in controllers])[:,None]
    kv=np.array([p['kv'] for p in controllers])[:,None]
    alpha=np.exp(-2*np.pi*np.array([p['cutoff'] for p in controllers])[:,None]*DT)
    saturation=np.zeros(c);rate_limited=np.zeros(c)
    for k in range(n):
        y[k]=states[-1]
        measured=(y[k-sensor_delay] if k>=sensor_delay else y[0])+noise[k]
        filtered_velocity=alpha*filtered_velocity+(1-alpha)*(measured-previous_measure)/DT
        previous_measure=measured
        correction=ff*ref_vel[k]+kv*(ref_vel[k]-filtered_velocity)
        saturation+=np.mean(np.abs(correction)>3/RAD,axis=1)
        target=np.clip(r[k]+np.clip(correction,-3/RAD,3/RAD),seg['lower'],seg['upper'])
        velocity=(target-previous_cmd)/DT
        limited=np.clip(velocity,np.maximum(-seg['vmax'],previous_v-seg['amax']*DT),
                        np.minimum(seg['vmax'],previous_v+seg['amax']*DT))
        rate_limited+=np.mean(np.abs(limited-velocity)>1e-9,axis=1)
        cmd[k]=previous_cmd+limited*DT
        # Baseline is the exact recorded command, not a second executor pass.
        for ci, controller in enumerate(controllers):
            if controller['name'] == 'baseline':
                cmd[k,ci] = r[k]
                limited[ci] = (r[k]-previous_cmd[ci])/DT
        previous_cmd=cmd[k];previous_v=limited
        source=np.empty((c,j))
        for joint in range(j):
            index=k-int(delay[joint])
            source[:,joint]=cmd[index,:,joint] if index>=0 else seg['q'][0,joint]
        for stage in range(order):
            states[stage]=a*states[stage]+(1-a)*source
            source=states[stage]
    for ci, controller in enumerate(controllers):
        if controller['name']=='baseline':rate_limited[ci]=0
    return y,cmd,saturation/n,rate_limited/n


def evaluate(segments, model, controllers, **scenario):
    records=[]
    for s in segments:
        y,u,sat,limited=simulate(s,model,controllers,**scenario)
        err=y[100:]-s['u'][100:,None,:]
        reference_err=y[100:]-s['reference'][100:,None,:]
        speed=derivative(s['u']); fast=np.abs(speed[100:])>.5
        for i,c in enumerate(controllers):
            velocity=np.diff(u[:,i],axis=0)/DT;acc=np.diff(velocity,axis=0)/DT
            records.append(dict(episode=s['episode'],segment=s['id'],controller=c['name'],
                rmse_deg=float(rmse(err[:,i])*RAD),p95_deg=float(np.quantile(np.abs(err[:,i]),.95)*RAD),
                fast_rmse_deg=float(rmse(err[:,i][fast])*RAD) if fast.any() else None,
                reference_rmse_deg=float(rmse(reference_err[:,i])*RAD),
                speed_p99=float(np.quantile(np.abs(velocity[100:]),.99)),
                accel_p99=float(np.quantile(np.abs(acc[100:]),.99)),
                correction_clip_fraction=float(sat[i]),rate_limit_fraction=float(limited[i]),
                max_correction_deg=float(np.max(np.abs(u[:,i]-s['u']))*RAD)))
    # Mean segment scores inside each episode; then equal episode weights.
    return records


def aggregate(records, key):
    values={}
    for r in records:
        if r[key] is not None:values.setdefault((r['controller'],r['episode']),[]).append(r[key])
    return {name:float(np.mean([np.mean(v) for (c,_),v in values.items() if c==name]))
            for name,_ in values}


def pole_radius(model, controller, *, delay_extra=0, tau_scale=1., sensor_delay=0, **unused):
    """Linear unsaturated closed-loop pole radius, including the causal D filter."""
    af=np.exp(-2*np.pi*controller['cutoff']*DT)
    radii=[]
    for tau, delay in zip(model['tau'],model['delay']):
        a=np.exp(-DT/max(tau*tau_scale,1e-9))
        plant_den=np.polynomial.polynomial.polypow([1,-a],model['order'])
        den=np.convolve(plant_den,[1,-af])
        shift=max(0,int(delay)+delay_extra)+1+sensor_delay
        loop=np.r_[np.zeros(shift),controller['kv']*(1-a)**model['order']*(1-af)/DT*np.array([1,-1])]
        poly=np.polynomial.polynomial.polyadd(den,loop)
        radii.append(float(np.max(np.abs(np.roots(np.trim_zeros(poly,'b'))))))
    return max(radii)


def main(root,out):
    out.mkdir(parents=True,exist_ok=True)
    rows,segments=load_cohort(root)
    dump(out/'cohort.json',rows)
    sets={split:[s for s in segments if s['split']==split] for split in ('train','validation','test','stress')}
    print('Cohort',len(rows),'attempts; usable segments', {k:len(v) for k,v in sets.items()},flush=True)
    # Raw tracking, lag and velocity-estimator sensitivity on every eligible segment.
    diagnostics=[]
    for s in segments:
        q,u,r=s['q'],s['u'],s['reference'];v=derivative(u)
        lag_errors=[]
        for d in range(16):
            sl=slice(max(100,d),None)
            shifted=np.r_[np.repeat(u[:1],d,axis=0),u[:len(u)-d]] if d else u
            lag_errors.append(np.mean((q[sl]-shifted[sl])**2,axis=0))
        lag_errors=np.array(lag_errors);best=lag_errors.argmin(0)
        raw_v=np.diff(q,axis=0)/DT;vf=derivative(q)
        slow=np.max(np.abs(v),axis=1)<.05
        noise=np.std((raw_v-vf[1:])[slow[1:]],axis=0) if slow[1:].sum()>10 else None
        diagnostics.append(dict(episode=s['episode'],segment=s['id'],split=s['split'],
            command_rmse_deg=rmse(q[100:]-u[100:],axis=0)*RAD,
            shaping_rmse_deg=rmse(r[100:]-u[100:],axis=0)*RAD,
            lag_ms=best*10,lag_aligned_rmse_deg=np.sqrt(lag_errors[best,np.arange(14)])*RAD,
            raw_velocity_noise_rad_s=noise,fast_fraction=np.mean(np.abs(v)>.5,axis=0)))
    dump(out/'diagnostics.json',diagnostics)
    models=[identify(sets['train'],order) for order in (1,2)]
    for m in models:
        for split in ('validation','test'):
            ss=sets[split]
            errs=[plant_response(s['u'],s['q'][0],m['order'],m['tau'],m['delay'])-s['q'] for s in ss]
            m[split+'_rmse_deg']=np.sqrt(episode_mse(errs,ss))*RAD
        print('Model',m['order'],'validation mean joint RMSE deg',m['validation_rmse_deg'].mean(),flush=True)
    selected=min(models,key=lambda m:np.mean(m['validation_rmse_deg']**2))
    dump(out/'models.json',models)
    options=candidates()
    validation=evaluate(sets['validation'],selected,options)
    score=aggregate(validation,'rmse_deg')
    groups={'FF':[c for c in options if c['ff']>0 and c['kv']==0],
            'VFB':[c for c in options if c['ff']==0 and c['kv']>0],
            'FF+VFB':[c for c in options if c['ff']>0 and c['kv']>0]}
    chosen=[options[0]]+[min(g,key=lambda c:score[c['name']]) for g in groups.values()]
    nominal_poles={c['name']:pole_radius(selected,c) for c in chosen}
    assert max(nominal_poles.values())<1, 'Selected controller is nominally unstable'
    test=evaluate(sets['test'],selected,chosen)
    stress=evaluate(sets['stress'],selected,chosen)
    dump(out/'validation-sweep.json',validation);dump(out/'heldout.json',test);dump(out/'partial-stress.json',stress)
    print('Selected on validation only',chosen,flush=True)
    # Sensitivity is not a probability distribution or a calibrated confidence interval.
    robustness=[]
    for order_model in models:
        scenarios=[(0,1,0,0),(2,1,0,0),(0,1.3,0,0),(0,1,.02,1),(2,1.3,.05,2)]
        if order_model['order']==selected['order']:
            scenarios += [(-age,1,0,age) for age in (1,2,3) if min(order_model['delay'])>=age]
        for extra,scale,noise,age in scenarios:
            data=evaluate(sets['test'],order_model,chosen,delay_extra=extra,tau_scale=scale,
                          noise_deg=noise,sensor_delay=age)
            scores=aggregate(data,'rmse_deg');baseline=scores['baseline']
            robustness.append(dict(order=order_model['order'],delay_extra_ms=extra*10,tau_scale=scale,
                scenario_type='same_observed_delay_sensor_allocation' if extra<0 else 'plant_and_measurement_stress',
                noise_deg=noise,sensor_age_ms=age*10,rmse_deg=scores,
                max_pole_radius={c['name']:pole_radius(order_model,c,delay_extra=extra,tau_scale=scale,sensor_delay=age) for c in chosen},
                gain_pct={k:100*(1-v/baseline) for k,v in scores.items()}))
    dump(out/'robustness.json',robustness)
    # Largest dynamic test segment, chosen by command speed, not controller benefit.
    example=max(sets['test'],key=lambda s:np.quantile(np.abs(derivative(s['u'])),.95))
    y,commands,_,_=simulate(example,selected,chosen)
    joint=int(np.argmax(rmse(example['q'][100:]-example['u'][100:],axis=0)))
    np.savez_compressed(out/'example.npz',time=example['time'],state=example['q'],command=example['u'],
        reference=example['reference'],simulated=y,modified_commands=commands,gripper=example['gripper'])
    raw_model=selected['test_rmse_deg']; measured=np.sqrt(episode_mse([s['q']-s['u'] for s in sets['test']],sets['test']))*RAD
    test_score=aggregate(test,'rmse_deg')
    # Bootstrap independent held-out episodes, never individual correlated ticks.
    episode_scores={}
    for record in test:episode_scores.setdefault((record['controller'],record['episode']),[]).append(record['rmse_deg'])
    eps=sorted({ep for _,ep in episode_scores});rng=np.random.default_rng(42);intervals={}
    for c in chosen[1:]:
        base=np.array([np.mean(episode_scores['baseline',ep]) for ep in eps]);new=np.array([np.mean(episode_scores[c['name'],ep]) for ep in eps])
        draws=rng.integers(0,len(eps),(2000,len(eps)))
        gains=100*(1-new[draws].mean(1)/base[draws].mean(1))
        intervals[c['name']]=np.quantile(gains,[.025,.975]).tolist()
    summary=dict(root=str(root.resolve()),joints=JOINTS,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cohort_counts={
        'attempts':len(rows),'nonempty':sum(r.get('eligible',False) for r in rows),
        'complete':sum(r.get('eligible',False) and r['status']=='complete' for r in rows),
        'partial':sum(r.get('eligible',False) and r['status']=='partial' for r in rows)},
        split_episode_counts={k:len({s['episode'] for s in v}) for k,v in sets.items()},
        selected_model_order=selected['order'],controllers=chosen,nominal_max_pole_radius=nominal_poles,
        measured_test_joint_rmse_deg=measured,model_test_joint_rmse_deg=raw_model,
        model_residual_to_measured_error=raw_model/measured,
        heldout_rmse_deg=test_score,heldout_fast_rmse_deg=aggregate(test,'fast_rmse_deg'),
        heldout_reference_rmse_deg=aggregate(test,'reference_rmse_deg'),
        heldout_gain_pct={k:100*(1-v/test_score['baseline']) for k,v in test_score.items()},
        episode_bootstrap_gain_intervals=intervals,
        robustness_gain_range={c['name']:[min(r['gain_pct'][c['name']] for r in robustness),
                                       max(r['gain_pct'][c['name']] for r in robustness)] for c in chosen},
        example_episode=example['episode'],example_segment=example['id'],example_joint=joint,
        latest_episode=max((r for r in rows if r.get('eligible')),key=lambda r:(r['session'],r['path']))['path'],
        model_equation='q[k+1] = cascade(order, exp(-dt/tau), u[k-delay]); unit DC gain',
        controller_equation='u = r + Tff*LPF(diff(r)/dt) + Tv*(LPF(diff(r)/dt)-LPF(diff(q_sim)/dt))',
        reference='recorded executor command; scheduled reference scored separately',
        constraints='Soft correction request cap 3 deg, followed by position/rate limiting; actual offset can exceed 3 deg during slew. Baseline is unchanged recorded command.',
        split_rule='UTC 2026-11-13: complete sessions before 13:30 fit; 13:30-13:47 tune; >=13:47 test. Partial attempts stress only.',
        processing='100 Hz linear resampling, split gaps >50ms, segments >=3s, exclude first 1s; partial stress omits final .5s',
        limitations=['Apparent plant includes unknown transport and measurement timing; no motor velocity/torque or valid arm send samples.',
        'Frozen reference counterfactual does not rerun the vision policy, contacts, grasp or ball dynamics.',
        'No inference of task success, true release time, motor bandwidth or root cause of fault 13.',
        'Stable SISO one/two-pole surrogates omit coupling, load changes and inner limits; simulated benefits are conditional.',
        'Bootstrap covers episode variation in the surrogate only, not model uncertainty or real deployment.',
        'Controller gains are offline candidates, not deployable settings.'])
    dump(out/'summary.json',summary)
    build_views(out, rows, segments)
    print(json.dumps({k:summary[k] for k in ('cohort_counts','split_episode_counts','heldout_gain_pct','robustness_gain_range')},indent=2),flush=True)


def build_views(p, rows, segs):
    """Package selected traces and an original-timestamp derivative diagnostic."""
    s=json.loads((p/'summary.json').read_text());models=json.loads((p/'models.json').read_text())
    m=next(x for x in models if x['order']==s['selected_model_order']);m['tau']=np.array(m['tau']);m['delay']=np.array(m['delay'])
    selected=[s['example_segment']]
    for predicate in [lambda r:r.get('split')=='test',lambda r:r.get('status')=='partial',lambda r:'gripper' in r.get('method','')]:
     choices=[(i,r) for i,r in enumerate(rows) if r.get('eligible') and r.get('model_segments') and predicate(r)]
     i,r=max(choices,key=lambda x:(x[1]['session'],x[1]['path']))
     selected.append(max((v for v in segs if v['episode']==i),key=lambda v:len(v['time']))['id'])
    cases=[]
    for i in dict.fromkeys(selected):
     seg=segs[i];row=rows[seg['episode']];y,u,_,_=simulate(seg,m,s['controllers']);idx=np.arange(0,len(seg['time']),5)
     compact=lambda a:np.round(a[idx],5).tolist()
     cases.append(dict(id=str(i),episode=seg['episode'],label=row['method']+' / '+row['session'][17:23]+' / '+Path(row['path']).name,
       path=str(Path(row['path']).resolve()),split=seg['split'],status=row['status'],time=np.round(seg['time'][idx],3).tolist(),
       state=compact(seg['q']*180/np.pi),command=compact(seg['u']*180/np.pi),reference=compact(seg['reference']*180/np.pi),
       simulated=[compact(y[:,ci]*180/np.pi) for ci in range(len(s['controllers']))],
       commands=[compact(u[:,ci]*180/np.pi) for ci in range(len(s['controllers']))]))
    # One unresampled recorded velocity-spike window for timing diagnosis.
    latest=max((r for r in rows if r.get('split')=='test' and r.get('eligible')),key=lambda r:r['session'])
    z=zarr.open_group(latest['path']+'/data.zarr',mode='r')['ticks'];stamp=np.asarray(z['monotonic_ns']);t=(stamp-stamp[0])*1e-9
    q=np.concatenate([np.asarray(z['state/'+arm])[:,:7] for arm in ['left_arm','right_arm']],1)
    v=np.diff(q,axis=0)/np.diff(t)[:,None];vf=np.zeros_like(q)
    for k,dt in enumerate(np.diff(t),1):
     a=np.exp(-2*np.pi*8*dt);vf[k]=a*vf[k-1]+(1-a)*v[k-1]
    k,j=np.unravel_index(np.argmax(np.abs(v)),v.shape);window=(t[1:]>t[k]-.7)&(t[1:]<t[k]+.7)
    noise=dict(path=latest['path'],joint=s['joints'][j],time=t[1:][window].tolist(),dt_ms=(np.diff(t)[window]*1000).tolist(),
               naive=v[window,j].tolist(),filtered=vf[1:][window,j].tolist())
    np.savez_compressed(p/'timing-spike.npz',**{k:np.array(v) for k,v in noise.items() if k not in ('path','joint')})
    diagnostics=json.loads((p/'diagnostics.json').read_text());test=[d for d in diagnostics if d['split']=='test']
    observed={}
    for key in ['command_rmse_deg','shaping_rmse_deg']:
     grouped={}
     for d in test:grouped.setdefault(d['episode'],[]).append(float(np.sqrt(np.mean(np.array(d[key])**2))))
     observed[key]=float(np.mean([np.mean(v) for v in grouped.values()]))
    records=json.loads((p/'heldout.json').read_text())
    extra=dict(observed=observed,median_lag_ms=np.median([d['lag_ms'] for d in test],axis=0).tolist(),
               controls={key:aggregate(records,key) for key in ['speed_p99','accel_p99','correction_clip_fraction','rate_limit_fraction','max_correction_deg']})
    payload=dict(summary=s,extra=extra,models=models,diagnostics=diagnostics,cohort=rows,cases=cases,noise=noise,
                 robustness=json.loads((p/'robustness.json').read_text()),validation=json.loads((p/'validation-sweep.json').read_text()),
                 heldout=records,stress=json.loads((p/'partial-stress.json').read_text()))
    (p/'view-data.json').write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    print('Prepared interactive views:',len(cases),'cases',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/experiments/throw-ball-pi05-native-h50-29999'))
    parser.add_argument('--output',type=Path,default=Path('data/analysis/throw-ball-velocity-loop-20261117'))
    args=parser.parse_args();main(args.root,args.output)
