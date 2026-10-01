#!/usr/bin/env python3
"""Posthoc reporting: compare every copied step/query to immutable WB96 source."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from wb97_contract import RAW
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb97_'):p.error('WB97 output required')
r=read_public(RAW);calls=[]
for setting in r['settings']:
    calls.append(setting['entry_nominal'])
    for target in setting['targets']:
        calls.extend(target['samples'][i] for i in (0,23,24));calls.append(target['fixed_reference_start_nominal'])
protocol=read_public(out/'protocol.json');summary=read_public(out/'summary.json')
total=0;errors=[];ratios={};cells={};first=[];domain=[];max_arc=0.;resolved_count=0
for line in (out/'steps.ndjson').open():
    row=json.loads(line)
    if row['record']!='step':continue
    total+=1;call=calls[row['trace']];step=call['accepted_trace'][row['index']]
    if row['saved_step']!=step:errors.append([row['trace'],row['index'],'saved step changed'])
    begin=step['query_begin']+int(row['index']==0 and call['options']['loopProtection']);end=step['query_end']
    queries=[call['field_queries'][begin]]+call['field_queries'][end-2:end]
    if row.get('accepted_queries')!=queries:errors.append([row['trace'],row['index'],'accepted query changed'])
    if row['index']==0:first.append({'trace':row['trace'],'position_difference_mm':max(abs(x-y) for x,y in zip(row['start_position_mm'],queries[0]['position_mm']))})
    if 'position_L1_mm' not in row:continue
    category=row['classification']['class'];estimate=step['accepted_error_estimate'];ratio=row['position_L1_mm']/estimate
    resolved=(row['position_L1_mm']>max(protocol['significance_absolute_position_mm'],protocol['significance_factor']*row['reference_position_uncertainty_L1_mm'],protocol['significance_factor']*row['closure']['position_L1_mm']))
    ratios.setdefault(category,{'max_all':0.,'max_numerically_resolved':None,'resolved_steps':0})
    x=ratios[category];x['max_all']=max(x['max_all'],ratio)
    if resolved:x['resolved_steps']+=1;x['max_numerically_resolved']=max(x['max_numerically_resolved'] or 0,ratio);resolved_count+=1
    for ref in row['reference_ladder']:max_arc=max(max_arc,abs(ref['arc_residual_mm']))
    key=f"{row['tolerance']:g}/{row['cap_m']:g}"
    c=cells.setdefault(key,{'steps':0,'max_position_L1_mm':0.,'max_direction':0.,'reference_unknown':0,'significant':0,'max_estimate_mm':0.})
    c['steps']+=1;c['max_position_L1_mm']=max(c['max_position_L1_mm'],row['position_L1_mm']);c['max_direction']=max(c['max_direction'],row['direction_max'])
    c['reference_unknown']+=row['reference_gate']!='PASS';c['significant']+=row['significant_underestimate'];c['max_estimate_mm']=max(c['max_estimate_mm'],estimate)
    if category=='domain':domain.append({k:row[k] for k in ('trace','index','tolerance','cap_m','path','station','sample','position_L1_mm','direction_max','position_defect_mm','direction_defect','significant_underestimate')}
      | {'estimate_mm':estimate,'start_z_mm':row['start_position_mm'][2],'end_z_mm':step['position_mm'][2],
         'last_query_z_mm':queries[-1]['position_mm'][2]})
result={'schema':'wb97_posthoc_source_audit_v1','accounting_gate':'PASS' if not errors and total==summary['steps'] else 'FAIL',
        'source_sha256':digest(RAW),'steps_sha256':digest(out/'steps.ndjson'),'script_sha256':digest(Path(__file__)),
        'errors':errors,'steps':total,'source_traces':len(calls),'initial_position_max_difference_mm':max(x['position_difference_mm'] for x in first),
        'position_estimate_ratios':ratios,'resolved_steps':resolved_count,'max_arc_residual_mm':max_arc,'cells':cells,'domain_rows':domain,
        'inferential_note':'Ratios are descriptive only; roundoff floor ratios do not regrade frozen H. Repeated prefixes are not independent events.',
        'qualification':'NOT_EVALUATED','cumulative_attribution':'UNKNOWN'}
write_new(out/'saved_step_source_audit.json',result)
print({k:result[k] for k in ('accounting_gate','steps','source_traces','initial_position_max_difference_mm','position_estimate_ratios','max_arc_residual_mm')})
