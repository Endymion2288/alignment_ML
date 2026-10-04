#!/usr/bin/env python3
"""Descriptive fixed-pair saved-data appendix; never overrides frozen WB122 UNKNOWN."""
import argparse,sys,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public,write_new
from wb100_contract import digest,verify

def first(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return None if len(a)==len(b) else min(len(a),len(b))
def build(out):
    verify(out);r=read_public(out/'event/response.json');c=read_public(out/'control.json');profiles=[];mapping={}
    for row in r['rows']:
        if row['input']['arm']!='observer_enabled':continue
        inp=row['input'];obs=row['observations'];steps=[x for x in obs if x['record']=='step_end'];trials=[x for x in obs if x['record']=='trial'];queries=[x for x in obs if x['record']=='field_query'];actions=[x for x in obs if x['record']=='action'];bb=[x for x in obs if x['record']=='bound_before'][0]
        frame=np.asarray(inp['frame']);p=np.asarray(bb['position_mm']).reshape(3);u=np.asarray(bb['direction']).reshape(3);local=frame[:3,:3].T@(p-frame[:3,3]);direction=frame[:3,:3].T@u;hf=np.r_[local[:2],direction[:2]/direction[2]];h=np.asarray(row['state']['h']).reshape(4)
        flat=[]
        for step in steps:
            i=step['step_index'];qs=[q for q in queries if q['step_index']==i];ts=[t for t in trials if t['step_index']==i]
            flat.append({'step_index':i,'rejections':step['rejections'],'accepted_h_mm':step['accepted_h_mm'],'path_mm':step['path_mm'],
              'errors':[t['error_estimate'] for t in ts],
              'cache_branches':[[q['query_phase'],q['trial_index'],q['cache_hit_before'],q['cache_refilled'],q['outside_map_fallback'],q['cache_after']['ranges_mm']] for q in qs]})
        profile={'sample_id':inp['sample_id'],'station':inp['station'],'accepted_steps':len(steps),'trials':len(trials),'field_queries':len(queries),'rejections':len(trials)-len(steps),'cache_refills':sum(q['cache_refilled'] for q in queries),'outside_fallbacks':sum(q['outside_map_fallback'] for q in queries),'path_mm':steps[-1]['path_mm'],
          'postprop_target_reached':actions[-1]['target_reached'],'postprop_navigation_break':actions[-1]['navigation_break'],'all_actions_target_reached_false':all(not a['target_reached'] for a in actions),
          'last_two_actions_same_free_state':all(actions[-1][k]==actions[-2][k] for k in ('position_mm','direction','time_acts','path_mm','constraints','rk_rejections')),
          'free_before_bound_local_z_mm':float(local[2]),'free_vs_bound_abs_h_difference':np.abs(hf-h).tolist(),
          'steps':flat}
        profiles.append(profile);mapping[(inp['station'],c['samples'][inp['sample_id']]['step_multiple'],c['samples'][inp['sample_id']]['sign'])]=profile
    order=[(s['step_multiple'],s['sign']) for s in c['samples']];pairs=[]
    for st in (1,2):
        choices=[('versus_nominal',order[0],x) for x in order[1:]]+[('plus_minus',(m,-1),(m,1)) for m in (.25,.5,1.,2.)]+[('versus_full_step',(1.,sign),(m,sign)) for sign in (1,-1) for m in (.25,.5,2.)]
        for role,a,b in choices:
            ra=mapping[(st,*a)];rb=mapping[(st,*b)];sa=ra['steps'];sb=rb['steps'];item={'station':st,'role':role,'reference_step_sign':list(a),'compared_step_sign':list(b),'alignment':'ORDINAL_STEP_NOT_SAME_ARC','accepted_step_count_difference':rb['accepted_steps']-ra['accepted_steps']}
            for key,label in (('rejections','first_rejection_difference_step'),('cache_branches','first_cache_branch_difference_step'),('errors','first_error_difference_step'),('accepted_h_mm','first_h_difference_step')):item[label]=first([s[key] for s in sa],[s[key] for s in sb])
            i=item['first_rejection_difference_step'];item['rejection_context']=None if i is None else {'reference':sa[i] if i<len(sa) else None,'compared':sb[i] if i<len(sb) else None};pairs.append(item)
    verify(out)
    return {'schema':'wb122_retrospective_saved_profile_appendix_v1','classification':'DIAGNOSTIC_ONLY_RETROSPECTIVE_RAW_DESCRIPTION','frozen_hypothesis_unchanged':read_public(out/'summary.json')['hypothesis'],'profiles':profiles,'comparisons':pairs,'causal_attribution':'UNVERIFIED','new_physical_calls':0,'qualification':'NOT_EVALUATED','source_sha256':digest(Path(__file__)),'runtime_sha256':digest(out/'event/response.json')}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('result',type=Path);a=p.parse_args();write_new(a.result,build(a.out))
