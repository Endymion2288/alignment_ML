#!/usr/bin/env python3
"""Reporting only: source-defined query/trial accounting and saved terminal states."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from alignment.wb93_transport_error import v,norm
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb96_'):p.error('WB96 output required')
r=read_public(out/'event/acts.json');rows=[];errors=[];official=[];all_calls=0;saved_traces=0
for setting in r['settings']:
    calls=[('entry',0,0,setting['entry_nominal'])]
    for target in setting['targets']:
        calls.extend(('direct',target['station'],i,q) for i,q in enumerate(target['samples']))
        calls.append(('fixed_start',target['station'],0,target['fixed_reference_start_nominal']))
    for path,station,index,q in calls:
        all_calls+=1;label={'tolerance':setting['tolerance'],'cap_m':setting['cap_m'],'path':path,'station':station,'sample_index':index}
        if q['status']!='PASS':errors.append({**label,'error':'failed call'});continue
        startup=int(q['options']['loopProtection'])
        if q['field_counts']['total']!=startup+3*q['accepted_steps']+2*q['rejected_trials']:errors.append({**label,'error':'query/trial count'})
        if 'official_default_state' in q:official.append(norm((v(q['state']['h'])-v(q['official_default_state']['h']))/[1,1,.001,.001]))
        if q['accepted_trace']:
            saved_traces+=1
            for step_index,step in enumerate(q['accepted_trace']):
                expected=3+2*step['rejected_trials']+(startup if step_index==0 else 0)
                if step['query_end']-step['query_begin']!=expected:errors.append({**label,'error':'step query/trial indexing'})
            if startup and q['field_queries'][0]!=q['field_queries'][1]:errors.append({**label,'error':'loop/start query identity'})
            if sum(s['rejected_trials'] for s in q['accepted_trace'])!=q['rejected_trials']:errors.append({**label,'error':'rejected sum'})
            h=sum(s['h_mm'] for s in q['accepted_trace'])
            if abs(h-q['path_length_mm'])>1e-8:errors.append({**label,'error':'accepted length sum'})
        delta=v(q['state']['position_mm'],3)-v(q['last_free_state']['position_mm'],3)
        row={**label,'accepted_steps':q['accepted_steps'],'rejected_trials':q['rejected_trials'],
          'field_counts':q['field_counts'],'free_to_bound_position_delta_mm':delta.tolist(),
          'navigator_target_reached_flag':q['last_free_state']['target_reached'],
          'saved_negative_steps':sum(s['h_mm']<0 for s in q['accepted_trace']),
          'max_saved_accepted_error_ratio':max((s['error_over_tolerance'] for s in q['accepted_trace']),default=None)}
        if path=='entry':
            row['in_map_field_queries']=[x for x in q['field_queries'] if x['zone_id']==1]
            row['last_accepted_step']=q['accepted_trace'][-1]
            begin=row['last_accepted_step']['query_begin']+(startup if q['accepted_steps']==1 else 0)
            row['accepted_last_trial_queries']=([q['field_queries'][begin]]+
              q['field_queries'][row['last_accepted_step']['query_end']-2:row['last_accepted_step']['query_end']])
        rows.append(row)
one_mm=[s for s in r['settings'] if s['cap_m']==.001]
identical=all(one_mm[0]['targets'][k]['samples'][i]['state']==s['targets'][k]['samples'][i]['state'] for s in one_mm[1:] for k in range(3) for i in range(25))
d={'schema':'wb96_posthoc_saved_trace_audit_v2','acts_sha256':digest(out/'event/acts.json'),
  'loop_protection_header':'/cvmfs/atlas.cern.ch/repo/sw/software/24.0/AthenaExternals/24.0.41/InstallArea/x86_64-el9-gcc13-opt/include/Acts/Propagator/detail/LoopProtection.hpp',
  'supersedes_reporting_only':'saved_trace_audit.json',
  'reporting_script_sha256':digest(Path(__file__)),'accounting_gate':'PASS' if not errors else 'FAIL','errors':errors,
  'all_calls':all_calls,'saved_traces':saved_traces,'official_comparisons':len(official),
  'max_current_official_difference':max(official),'one_mm_returned_states_identical_all_tolerances':identical,
  'rows':rows,'mechanism_regrading':False,'qualification':'NOT_EVALUATED'}
write_new(out/'saved_trace_audit_v2.json',d)
print({k:d[k] for k in ('accounting_gate','all_calls','saved_traces','official_comparisons','max_current_official_difference','one_mm_returned_states_identical_all_tolerances')})
