"""Create task-specific offline review artifacts after development lock + holdout."""
import html,json,sys,platform,subprocess,hashlib,os
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np,yaml
if os.environ.get('CHUNK_DIFFIK_PLOT_DEPS'):sys.path.append(os.environ['CHUNK_DIFFIK_PLOT_DEPS'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import audit,review
O=audit.OUT

def mean(a):
 a=[x for x in a if x is not None and np.isfinite(x)]
 return float(np.mean(a)) if a else None

def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
def f(x):return 'missing' if x is None else f'{x:.3f}' if isinstance(x,(float,np.floating)) else str(x)
def sessionmean(rows,func):
 groups=defaultdict(list)
 for r in rows:groups[r['session']].append(func(r))
 return mean([mean(x) for x in groups.values()])

def main():
 cohort=json.loads((O/'cohort.json').read_text());base=[json.loads(p.read_text()) for p in (O/'baseline').glob('*.json')]
 selection=json.loads((O/'selection_development_only.json').read_text());locked=json.loads((O/'locked_candidates.json').read_text())
 dev=review.records('development');hold=review.records('holdout');agg=review.aggregate(dev,['task'])+review.aggregate(hold,['task']);strata=review.aggregate(dev+hold,['task','split','stratum','status'])
 audit.write(O/'all_candidate_stratified.json',strata)
 all_result_files=list((O/'candidates').glob('*/*.json'));failures=[]
 for p in all_result_files:
  r=json.loads(p.read_text())
  if 'failures' in r:failures.append(dict(episode=p.parent.name,candidate=p.stem,failures=r['failures']))
 audit.write(O/'rejected_counterfactual_chunks.json',failures)
 for task in ['pack_plate','pack_bottle']:
  rows=[r for r in base if r['task']==task];select=selection[task];cid=select['selected'];by=defaultdict(list)
  for r in rows:by[(r['session'],r['status'],r['stratum'])].append(r)
  baseline_rows=[];baseline_json=[]
  for (session,status,stratum),rr in sorted(by.items()):
   vals=[mean([r['metrics'][k]['p95'] for r in rr]) for k in ['target_reference_all_mm','target_reference_all_deg','reference_command_all_mm','reference_command_all_deg','command_state_all_mm','command_state_all_deg','target_state_all_mm','target_state_all_deg']]
   baseline_rows.append([session.split('T')[-1],rr[0]['split'],status,len(rr),stratum,rr[0]['identity']['smooth']['max_velocity'],rr[0]['identity']['smooth']['max_acceleration'],*[f(x) for x in vals]])
   baseline_json.append(dict(session=session,status=status,stratum=stratum,episodes=len(rr),split=rr[0]['split'],metrics=dict(zip(['target_reference_mm','target_reference_deg','reference_command_mm','reference_command_deg','command_state_mm','command_state_deg','target_state_mm','target_state_deg'],vals))))
  audit.write(O/f'{task}_baseline_stratified.json',baseline_json)
  comparisons=[];summary_rows=[];regressions=[]
  for split,records in [('development',dev),('holdout',hold)]:
   for status in ['complete','aborted']:
    rr=[x for x in records if x['row']['task']==task and x['row']['status']==status];aa=review.aggregate(rr,['task'])
    for cc in [c['id'] for c in locked[task]]:
     found=[x for x in aa if x['candidate']==cc]
     if not found:continue
     r=found[0];m=r['metrics'];summary_rows.append([split,status,cc,r['episodes'],*[f(m[k]) for k in ['target_reference_all_mm','target_reference_all_deg','reference_command_all_mm','reference_command_all_deg','target_command_all_mm','target_command_all_deg','tcp_m_jerk_all','tcp_rad_jerk_all']]])
     comparisons.append(dict(split=split,status=status,**r))
  audit.write(O/f'{task}_candidate_comparison.json',comparisons)
  phase_rows=[];phase_json=[]
  for split,records in [('development',dev),('holdout',hold)]:
   for status in ['complete','aborted']:
    for cc in [c['id'] for c in locked[task]]:
     rr=[x for x in records if x['row']['task']==task and x['row']['status']==status and x['result']['id']==cc and 'metrics' in x['result']]
     for region in ['pre','post','stable']:
      metric_names=[f'target_command_{region}_mm',f'target_command_{region}_deg']
      values=[];coverage={}
      for k in metric_names:
       vv=[dict(session=x['row']['session'],v=x['result']['metrics'][k].get('p95')) for x in rr]
       values.append(sessionmean(vv,lambda x:x['v']));coverage[k]=sum(x['v'] is not None for x in vv)
      for kind in ['tcp_m_velocity','tcp_m_acceleration','tcp_m_jerk','tcp_rad_velocity','tcp_rad_acceleration','tcp_rad_jerk']:
       k=f'{kind}_{region}';vv=[dict(session=x['row']['session'],v=x['result']['metrics']['motion']['command'][k].get('p95')) for x in rr]
       values.append(sessionmean(vv,lambda x:x['v']));metric_names.append(k);coverage[k]=sum(x['v'] is not None for x in vv)
      phase_rows.append([split,status,cc,region,*[f(v) for v in values]])
      phase_json.append(dict(split=split,status=status,candidate=cc,region=region,metrics=dict(zip(metric_names,values)),included_episode_counts=coverage))
  audit.write(O/f'{task}_regions.json',phase_json)
  baseline_motion=[]
  for (session,status,stratum),rr in sorted(by.items()):
   for stage in ['target','reference','command','state']:
    for region in ['pre','post','stable','hold']:
     metrics={};counts={}
     for kind in ['tcp_m_velocity','tcp_m_acceleration','tcp_m_jerk','tcp_rad_velocity','tcp_rad_acceleration','tcp_rad_jerk']:
      k=f'{kind}_{region}';vv=[]
      for r in rr:
       if stage=='target':entry=json.loads((O/'supplement'/f'{r["id"]}.json').read_text())['target_motion'][k]
       else:entry=r['metrics']['motion'][stage][k]
       if entry['n']:vv.append(entry['p95'])
      metrics[k]=mean(vv);counts[k]=len(vv)
     baseline_motion.append(dict(session=session,status=status,stratum=stratum,stage=stage,region=region,metrics=metrics,included_episode_counts=counts))
  audit.write(O/f'{task}_baseline_motion.json',baseline_motion)

  # Each configuration and completion category receives its own regression check.
  groups=defaultdict(dict)
  for r in strata:
   if r['task']==task:groups[(r['split'],r['stratum'],r['status'])][r['candidate']]=r
  detail=[]
  for (split,stratum,status),v in groups.items():
   if cid not in v or 'baseline' not in v:continue
   b=v['baseline']['metrics'];m=v[cid]['metrics'];delta={k:100*(m[k]/b[k]-1) for k in review.KEYS+review.MOTION_KEYS if m[k] is not None and b[k] not in (None,0)}
   detail.append(dict(split=split,stratum=stratum,status=status,episodes=v[cid]['episodes'],percent_change=delta))
   if any(delta[k]>0 for k in ['target_command_all_mm','target_command_all_deg','tcp_m_jerk_all','tcp_rad_jerk_all']):regressions.append(detail[-1])
  audit.write(O/f'{task}_selected_stratum_deltas.json',detail)
  info=dict(task=task,review_only=True,apply_to_production=False,selected=next(c for c in locked[task] if c['id']==cid),
    frozen_reference_alternative=select['frozen_reference_selected'],
    resolved_parameter_paths={
     'velocity':'executor.motion_limits.arm.max_velocity AND executor.smooth.max_velocity; recreate chunk DiffIK from original TCP targets',
     'smooth.max_acceleration':'executor.motion_limits.arm.max_acceleration AND executor.smooth.max_acceleration (rad/s^2)',
     'smooth.cutoff_hz':'executor.smooth.cutoff_hz',
     'objective':'Local TCPObjectiveKinematics in audit.py only; there is no production YAML option for this variant.'},
    fixed={'inference.handoff_skip_steps':1,'timing':'per-episode historical values','joint_position_bounds':'unchanged','j67_constraint':'unchanged','IK_seed':'recorded per chunk','acceptance_clock':'recorded','command_state_history':'persistent; only historical holds/resets clear velocity'},
    tested_ranges={'shared_max_velocity_rad_s':[.3,.4,.5,.65,.8],'cutoff_hz':[4,8,12,20],'max_acceleration_rad_s2':[None,8,20,40],'max_step_dt_s':[.008,.01,.016,.02],'mode':['isotropic','per_joint'],
      'lam':[.0001,.001,.01],'mu_nullspace':[0,1000,3000],'limit_margin_deg':[6,8],'j67_margin_deg':[5,8],
      'hardware_velocity_ratio_formula_only':[32,40,50],'command_margin_formula_only':[.5,.7,.9]},
    hardware_rate_note='Ratio/margin scenarios are formula/guard audits, not simulated physical controller responses; see rate_contract_review.json.',
    selection=select['gate'],selection_rule=select['score'],development_sessions=sorted({r['session'] for r in rows if r['split']=='development'}),holdout_sessions=sorted({r['session'] for r in rows if r['split']=='holdout'}),
    primary_objective_regressions=regressions,per_layer_deltas_file=f'{task}_selected_stratum_deltas.json',layer_regressions=[dict(split=x['split'],stratum=x['stratum'],status=x['status'],percent_change={k:v for k,v in x['percent_change'].items() if k in review.KEYS and v>0}) for x in detail],holdout_used_for_selection=False,task_success_inference=False,
    limitations=['Open-loop fixed-target intervention with recorded seeds, not exact counterfactual closed loop. Changed command does not change later original target, recorded seed or observation.',
      'No new measured-state trajectory, contact dynamics, collision geometry, controller acceleration/filter or feedback timestamps are available.',
      'Dense reference baseline is reconstructed and validated, not originally persisted.',
      'Acceptance/decode latency is frozen; new IK execution latency and rejection effects require runtime validation.',
      'Only accepted plans have original targets. Rejected/unrecorded predictions and the failed final dispatch cannot be reconstructed.'])
  (O/f'{task}_candidate_review.yaml').write_text(yaml.safe_dump(info,sort_keys=False,allow_unicode=True))
  # Plot development tradeoffs only, keeping holdout out of search.
  a=[r for r in review.aggregate(dev,['task']) if r['task']==task and r['episodes']==select['baseline']['episodes']]
  fig,ax=plt.subplots(figsize=(7,4.5))
  for prefix,marker,color in [('other','o','0.65'),('tcp','s','tab:blue')]:
   rr=[r for r in a if r['candidate'].startswith('tcp')==(prefix=='tcp')]
   ax.scatter([r['metrics']['target_command_all_mm'] for r in rr],[r['metrics']['tcp_m_jerk_all'] for r in rr],s=18,label='TCP objective' if prefix=='tcp' else 'Flange objective',marker=marker,color=color)
  for name in ['baseline',cid]:
   r=next(x for x in a if x['candidate']==name);ax.annotate(name,(r['metrics']['target_command_all_mm'],r['metrics']['tcp_m_jerk_all']),fontsize=7,xytext=(3,6),textcoords='offset points')
  ax.set(xlabel='Target-to-command TCP translation error (mm)',ylabel='Command TCP linear jerk (m/s³)',title=f'{task}: development accuracy / smoothness');ax.legend(fontsize=8);fig.text(.5,.015,'Session-balanced mean of episode p95 | Source: 2026-10-17 campaign, development sessions',ha='center',fontsize=7);fig.tight_layout(rect=(0,.055,1,1));fig.savefig(O/f'{task}_tradeoffs.png',dpi=160);plt.close(fig)
  allavg={k:sessionmean(rows,lambda r:r['metrics'][k]['p95']) for k in ['target_reference_all_mm','reference_command_all_mm','command_state_all_mm']}
  single=next(x for x in review.aggregate(dev,['task']) if x['task']==task and x['candidate']=='tcp_objective');devbase=select['baseline']
  dsel=next(x for x in review.aggregate(dev,['task']) if x['task']==task and x['candidate']==cid)
  holdagg=review.aggregate([x for x in hold if x['row']['task']==task],['task']);hbase=next(x for x in holdagg if x['candidate']=='baseline');hsel=next(x for x in holdagg if x['candidate']==cid)
  def compare_line(b,s):
   return '; '.join(f'{k}: {b["metrics"][k]:.3f} -> {s["metrics"][k]:.3f}' for k in ['target_command_all_mm','target_command_all_deg','tcp_m_jerk_all','tcp_rad_jerk_all'])
  text=f'''{task}: chunk DiffIK offline review

Cohort: {len(rows)} valid episodes; {sum(r['status']=='complete' for r in rows)} completed and {sum(r['status']=='aborted' for r in rows)} aborted. Completion is not success. Source: {audit.CAMPAIGN}. See {task}_baseline_stratified.json for each historical configuration/session/outcome; values below average episode p95 within session then sessions equally.

Confirmed: historical knots reproduced exactly; scheduled/command maximum differences <=1.34e-15 rad. Critical source/config hashes match the saved source manifest. The 628-plan cohort has original TCP targets and per-arm seeds. Dense trajectories are reconstructed. Historical handoff_skip_steps remains 1 (first chunk skips 0); asynchronous latency trim, serial retiming, outgoing plan ownership and gap holds are retained. Only actually executed tick intervals contribute errors.

TCP-objective single factor (development): target-reference translation {devbase['metrics']['target_reference_all_mm']:.3f} -> {single['metrics']['target_reference_all_mm']:.3f} mm; rotation {devbase['metrics']['target_reference_all_deg']:.3f} -> {single['metrics']['target_reference_all_deg']:.3f} degrees. This independently supports the user's verified observation. The existing TCP-to-flange transform is correct, but under rate/position constraints a flange residual and orientation residual do not minimize TCP position error. Direct TCP pose/Jacobian optimization changes that constrained objective; it does not relax joint limits. The historical max_lag_mm diagnostic measures flange lag.

Review candidate: {cid}. Frozen-reference-only alternative: {select['frozen_reference_selected']}.
Development: {compare_line(devbase,dsel)}.
Locked candidate holdout: {compare_line(hbase,hsel)}.
Downstream tradeoff: holdout reference-to-command translation {hbase['metrics']['reference_command_all_mm']:.3f} -> {hsel['metrics']['reference_command_all_mm']:.3f} mm and rotation {hbase['metrics']['reference_command_all_deg']:.3f} -> {hsel['metrics']['reference_command_all_deg']:.3f} degrees; this layer gets worse even though total target-to-command error improves.
Units: mm, degrees, m/s^3, rad/s^3. These are counterfactual commands, never predicted measured-state improvements. Per-stratum regressions: {len(regressions)}; see {task}_selected_stratum_deltas.json and candidate_review.yaml, including all positive/negative changes. A global average cannot establish transfer across historical configurations.

Confirmed bottlenecks: QP per-joint speed saturation and recorded positional constraints; downstream filter/rate shaping; observable command-to-state discrepancy. QP J4/J6 constraints occur in some episodes, while J6-J7 activation is less frequent. See per-episode fractions rather than inferring all residuals from a single bound. 100 Hz max_step_dt_s .01/.016/.02 are equivalent; .008 reduces only executor budget. Changing the shared rate requires new IK. Per-joint command_safety entries reject, not shape. Lam/nullspace probes are secondary and do not remove the main error. Tightened position/J67 margins can reject whole chunks and are not scored as successful motion.

Evidence only: command/state alignment typically prefers 65-75 ms delay; tick time is not per-arm acquisition time, so controller delay is not identified. Controller fault codes are recorded as raw evidence and not diagnosed as speed faults. All 24 valid records lack human outcome labels and have zero valid sent_command wire samples. Command means submitted software output, not proven hardware receipt. No task success is inferred from normal termination. Physical response to velocity_ratio/acceleration_ratio, contact, payload and missed deadline behavior requires hardware validation.

Limit: recorded original target, seed and acceptance clock remain frozen for all variants. Executor q/v history is retained across handoffs and gaps follow historical semantics. This is a reproducible fixed-target intervention, not a claim of exact closed-loop replay; altered commands would change later observations and possibly seeds/policy outputs on the robot.
'''
  short=f"""{task} — chunk DiffIK offline conclusion

Samples: {len(rows)} valid records ({sum(r['status']=='complete' for r in rows)} complete, {sum(r['status']=='aborted' for r in rows)} aborted), fixed by recency before analysis. No human success labels. Latest nonempty session held out; historical configs and completion classes are reported separately in {task}_report.html.

Baseline reproduced: joint knots exactly; scheduled/command maximum differences <=1.34e-15 rad across the 24-record, 628-plan cohort. Original targets and seeds exist; dense IK samples were regenerated and checked. handoff_skip_steps stays at its historical 1, first-chunk skip 0.

Confirmed TCP-objective single-factor gain on development: target→reference position {devbase['metrics']['target_reference_all_mm']:.2f}→{single['metrics']['target_reference_all_mm']:.2f} mm; rotation {devbase['metrics']['target_reference_all_deg']:.2f}→{single['metrics']['target_reference_all_deg']:.2f} degrees. QP rate/position saturation and downstream command shaping remain additional bottlenecks.

Locked review candidate: {cid} (TCP objective, shared 0.65 rad/s, acceleration 20 rad/s², cutoff 4 Hz). No physical bounds were relaxed. No frozen-reference-only candidate met the full accuracy/smoothness gate.
Held-out total target→command p95: translation {hbase['metrics']['target_command_all_mm']:.2f}→{hsel['metrics']['target_command_all_mm']:.2f} mm; rotation {hbase['metrics']['target_command_all_deg']:.2f}→{hsel['metrics']['target_command_all_deg']:.2f} degrees. Linear jerk {hbase['metrics']['tcp_m_jerk_all']:.1f}→{hsel['metrics']['tcp_m_jerk_all']:.1f} m/s³; angular jerk {hbase['metrics']['tcp_rad_jerk_all']:.1f}→{hsel['metrics']['tcp_rad_jerk_all']:.1f} rad/s³. These displayed summaries average the two episode p95s within the held-out session; completed/aborted values remain separate in the report.
Tradeoff: reference→command translation worsens {hbase['metrics']['reference_command_all_mm']:.2f}→{hsel['metrics']['reference_command_all_mm']:.2f} mm. Better IK does not imply proportional command or measured-state improvement.

Evidence only: observed command/state alignment prefers about 65–75 ms, but per-arm feedback timestamps and valid wire sends are missing. Raw fault codes do not identify the cause.
Hardware validation still needed: controller ratio/acceleration response, contact/payload, timing/deadlines and task success. Counterfactual commands retain recorded seeds/target/acceptance clocks; this is fixed-target open-loop replay, not an exact new closed loop. No formal config, hardware or service was changed.
"""
  (O/f'{task}_conclusion.txt').write_text(short)
  body=f'<h1>{task} · chunk DiffIK</h1><p>Offline review only. No hardware connection, model service or production configuration change.</p><p><b>Selected development candidate:</b> {html.escape(cid)}. See independent <a href="{task}_candidate_review.yaml">parameter review</a> and <a href="{task}_conclusion.txt">short conclusion</a>.</p><p>Values are episode p95, averaged within session and then across sessions. Completed ≠ successful. Translation uses TCP Euclidean distance; rotation uses SO(3) geodesic angle. Error layers are not additive.</p>'
  body+='<h2>Historical baseline, separated by session / configuration / completion</h2>'+table(['Session UTC','Split','Status','n','Config stratum','vmax rad/s','amax rad/s²','Target→ref mm','deg','Ref→command mm','deg','Command→state mm','deg','Target→state mm','deg'],baseline_rows)
  body+='<h2>Locked candidates: development and untouched session holdout</h2>'+table(['Split','Status','Candidate','n','Target→ref mm','deg','Ref→command mm','deg','Target→command mm','deg','Command jerk m/s³','rad/s³'],summary_rows)
  body+='<h2>Handoffs and stable execution</h2><p>Pre/post windows are 100 ms before/after the change of executing plan. Entries are mean episode p95, session-balanced; empty phase data are explicitly missing.</p>'+table(['Split','Status','Candidate','Region','Target→command mm','deg','TCP v m/s','a m/s²','jerk m/s³','omega rad/s','alpha rad/s²','angular jerk rad/s³'],phase_rows)
  body+=f'<img src="{task}_tradeoffs.png" width="850" alt="Development TCP accuracy and linear jerk tradeoffs"><p>Source: 2026-10-17 recorded campaign. Fixed-target/seed replay; original timing and handoffs. Command/state changes are not simulated.</p>'
  body+='<h2>Method and limitations</h2><pre>'+html.escape(text)+'</pre>'
  body+='<h2>Detailed evidence</h2><ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['cohort.json','test_results.json','rate_contract_review.json','selection_development_only.json','all_candidate_stratified.json',f'{task}_selected_stratum_deltas.json',f'{task}_baseline_motion.json',f'{task}_regions.json','rejected_counterfactual_chunks.json','provenance.json'])+'</ul>'
  (O/f'{task}_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>'+task+' chunk DiffIK</title><style>body{font:14px system-ui;margin:32px;color:#20252b;max-width:1500px}table{border-collapse:collapse;font-size:12px;margin:20px 0}td,th{padding:7px;border-bottom:1px solid #d6dade;text-align:left}th{background:#eef0f2}pre{white-space:pre-wrap;line-height:1.5}img{max-width:100%}h1{font-size:26px}h2{font-size:19px;margin-top:32px}</style>'+body+'</html>')
 source={str(p.relative_to(audit.ROOT)):audit.digest(p.read_bytes()) for p in (audit.ROOT/'scripts/validation/chunk_diffik_audit').glob('*.py')}
 import scipy,osqp,zarr
 provenance=dict(generated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),versions={'matplotlib':matplotlib.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'osqp':osqp.__version__,'zarr':zarr.__version__},scripts_sha256=source,
  git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=audit.ROOT,text=True).strip(),
  critical_source_sha256={p:audit.digest((audit.ROOT/p).read_bytes()) for p in audit.SOURCE},baseline_episodes=len(base),baseline_plans=sum(len(r['plans']) for r in base),
  candidate_episode_runs=len(all_result_files),failed_counterfactual_runs=len(failures),selected_trace_reproductions=len(list((O/'selected_traces').glob('*.npz'))),production_config_modified=False,hardware_connected=False,model_service_started=False)
 audit.write(O/'provenance.json',provenance)
 print(provenance)
if __name__=='__main__':main()
