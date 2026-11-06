"""Design joint TCP candidates from completed development single-factor evidence."""
import json
import audit,review
items=review.records('development');agg=review.aggregate(items,['task']);choices={};evidence={}
for task in ['pack_plate','pack_bottle']:
 rs=[x for x in agg if x['task']==task];b=next(x for x in rs if x['candidate']=='baseline');t=next(x for x in rs if x['candidate']=='tcp_objective')
 assert t['episodes']==b['episodes'],'Finish TCP single-factor development first'
 evidence[task]=dict(baseline=b,tcp=t)
 # Direct TCP objectives reduce raw-target residual. Test whether downstream
 # acceleration shaping can preserve that gain without increasing TCP jerk.
 assert t['metrics']['target_reference_all_mm']<b['metrics']['target_reference_all_mm']
 choices[task]=[dict(id=f'tcp_joint_v{v}_a{a}_f{f}',objective='tcp',**({'velocity':v} if v is not None else {}),smooth={'max_acceleration':a,'cutoff_hz':f})
   for v in (None,.5,.65) for a in (8.,20.) for f in (4.,8.)]
audit.write(audit.OUT/'tcp_joint_candidates.json',choices)
audit.write(audit.OUT/'tcp_joint_rationale.json',dict(evidence=evidence,rule='User supplied verified TCP-objective hypothesis. Confirm on development single-factor first; cross recorded/.5/.65 shared speed with 8/20 acceleration and 4/8 cutoff. Physical bounds remain unchanged. No holdout read.'))
print({k:len(v) for k,v in choices.items()})
