#!/usr/bin/env python3
"""Reporting only: all source state identities and independent FD/global gate arithmetic."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from alignment.wb98_defect_transport import vector,norm,SCALE
from wb98_contract import previous,WB97
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb98_'):p.error('WB98 output required')
raw=read_public(previous.RAW);traces=read_public(out/'traces.json')['traces'];protocol=read_public(out/'protocol.json');calls=[]
for setting in raw['settings']:
    calls.append(setting['entry_nominal'])
    for target in setting['targets']:
        calls.extend(target['samples'][i] for i in (0,23,24));calls.append(target['fixed_reference_start_nominal'])
initial={}
for line in (WB97/'steps.ndjson').open():
    if '"index":0,' in line:
        row=json.loads(line)
        if row.get('record')=='step' and row['index']==0:initial[row['trace']]=row
def chart(p,u):return np.array([p[0],p[1],u[0]/u[2],u[1]/u[2]])
def jnorm(a):return float(np.max(np.abs(a*SCALE[None,:]/SCALE[:,None])))
def converged(coarse,fine):return fine<=protocol['reference_scaled_tolerance'] and (fine<=protocol['reference_contraction_ratio']*coarse or max(coarse,fine)<=protocol['reference_roundoff_scaled'])
errors=[];total=0;checkpoints=0;maxsource=0.;maxfd=0.;maxroots=0.;maxrkn=0.;expected_checkpoints=0
for call in calls:expected_checkpoints+=len({0,len(call['accepted_trace'])//2,len(call['accepted_trace'])-1})
for line in (out/'steps.ndjson').open():
    row=json.loads(line);t=row['trace'];i=row['index'];call=calls[t];step=call['accepted_trace'][i]
    prior=initial[t] if i==0 else call['accepted_trace'][i-1]
    sp=prior['start_position_mm'] if i==0 else prior['position_mm'];su=prior['start_direction'] if i==0 else prior['direction']
    source=max(norm(vector(row['start_X'])-chart(sp,su)),norm(vector(row['end_X'])-chart(step['position_mm'],step['direction'])))
    maxsource=max(maxsource,source)
    if source>1e-10 or row['from_z_mm']!=sp[2] or row['to_z_mm']!=step['position_mm'][2] or row['h_mm']!=step['h_mm']:errors.append([t,i,'source state or step identity'])
    maxrkn=max(maxrkn,row.get('source_rkn_closure_scaled',0))
    selected=i in {0,len(call['accepted_trace'])//2,len(call['accepted_trace'])-1}
    if selected:
        checkpoints+=1
        if 'fd' not in row:errors.append([t,i,'missing FD checkpoint'])
        elif len(row['ladders'])==3:
            a=np.asarray(row['ladders'][-1]['A']);fd=row['fd'];matrices=[np.asarray(r['matrix']) for r in fd['rows']]
            differences=[jnorm(m-a) for m in matrices];difference=jnorm(matrices[1]-matrices[0]);maxfd=max(maxfd,*differences,difference)
            expected=max(*differences,difference)<=protocol['jacobian_scaled_tolerance']
            if expected!=(fd['gate']=='PASS'):errors.append([t,i,'FD gate arithmetic'])
    elif row['fd_gate']!='NOT_SELECTED':errors.append([t,i,'posthoc FD selection'])
    total+=1
for t,row in enumerate(traces):
    if len(row['endpoint_ladders'])!=3:continue
    ls=row['endpoint_ladders'];fixed=[vector(l['global_same_z_X']) for l in ls]
    arc_positions=[np.asarray(l['global_same_arc_position_mm']) for l in ls];arc_directions=[np.asarray(l['global_same_arc_direction']) for l in ls]
    plane_positions=[np.asarray(l['plane_reference_position_mm']) for l in ls]
    plane_directions=[]
    for l in ls:
        tx,ty=l['plane_reference_X'][2:];u=np.array([tx,ty,1.]);plane_directions.append(u/np.linalg.norm(u))
        maxroots=max(maxroots,abs(l['global_arc_root_residual_mm']),abs(l['plane_root_residual_mm']))
    def differences(positions,directions):return [max(float(np.max(np.abs(positions[i+1]-positions[i]))),float(np.max(np.abs(directions[i+1]-directions[i])))/protocol['direction_scale']) for i in (0,1)]
    pairs=[[norm(fixed[i+1]-fixed[i]) for i in (0,1)],differences(arc_positions,arc_directions),differences(plane_positions,plane_directions)]
    if all(converged(*pair) for pair in pairs)!=(row['global_gate']=='PASS'):errors.append([t,'global representation gate arithmetic'])
result={'schema':'wb98_posthoc_source_fd_audit_v1','gate':'PASS' if not errors and total==protocol['expected_steps'] and checkpoints==expected_checkpoints and len(initial)==protocol['expected_traces'] else 'FAIL',
        'errors':errors,'steps':total,'source_initial_states':len(initial),'fd_checkpoints':checkpoints,'expected_fd_checkpoints':expected_checkpoints,
        'max_source_chart_difference_scaled':maxsource,'max_source_rkn_closure_scaled':maxrkn,'max_fd_scaled':maxfd,'max_arc_or_plane_root_residual_mm':maxroots,
        'steps_sha256':digest(out/'steps.ndjson'),'source_sha256':digest(previous.RAW),'wb97_steps_sha256':digest(WB97/'steps.ndjson'),
        'script_sha256':digest(Path(__file__)),'qualification':'NOT_EVALUATED'}
write_new(out/'saved_transport_source_audit.json',result)
print({k:v for k,v in result.items() if not k.endswith('sha256')})
