#!/usr/bin/env python3
"""Post-run source-aware identity correction; frozen per-state gates unchanged."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT as PROJECT,read_public,write_new,digest
from run_wb91_covariance_contract import verify,OUTPUT

def aggregate(out):
    verify(out);inputs=read_public(out/'array_inputs.json');selection=read_public(out/'selection.json')['events']
    pilot=Path(inputs['pilot_root'])
    if digest(pilot/'physical/00/validation.json')!=inputs['pilot_validation_sha256']:raise ValueError('pilot changed')
    paths=[pilot/'physical/00']+[out/'physical'/f'{i:02d}' for i in inputs['indices']]
    summaries=[];missing=[];input_identities=set();header_identities=set();header_groups={}
    for i,work in enumerate(paths):
        if not (work/'validation.json').is_file():
            missing.append({'index':i,'error':read_public(work/'execution_error.json') if (work/'execution_error.json').exists() else 'UNKNOWN'})
            continue
        s=read_public(work/'validation.json');manifest=read_public(work/'input_manifest.json')
        if manifest['row']!=selection[i]:raise ValueError('selected ordinal/source mismatch')
        row=selection[i];header=tuple(s['actual_event_header'])
        input_id=(row['input_xaod'],int(row['xaod_entry_index']))
        header_id=(row['input_xaod'],*header)
        if input_id in input_identities or header_id in header_identities:raise ValueError('duplicate event within same input source')
        input_identities.add(input_id);header_identities.add(header_id)
        header_groups.setdefault(header,[]).append({'index':i,'source_id':row['source_id'],'input_xaod':row['input_xaod']})
        for mode in ('legacy','repair'):
            if digest(work/f'{mode}.jsonl')!=s[f'{mode}_sha256']:raise ValueError('audit changed')
        s.update(index=i,validation_path=str(work/'validation.json'),source_id=row['source_id'],input_xaod=row['input_xaod'])
        summaries.append(s)
    results=[r for s in summaries for r in s['results']]
    good=len(summaries)==24 and all(s['gate']=='PASS' and s['input_event_header_verified'] for s in summaries)
    write_new(out/'physical_summary.json',{'gate':'PASS' if good else 'FAIL' if any(s['gate']=='FAIL' for s in summaries) else 'INCOMPLETE',
        'n_events':len(summaries),'n_states':len(results),'missing':missing,
        'event_summaries':[{k:v for k,v in s.items() if k!='results'} for s in summaries],
        'max_errors':{k:max(r[k] for r in results) for k in ('paired_error','state_error','normal_inverse_error','fd_error','jacobian_roundtrip_error','covariance_roundtrip_error','legacy_covariance_error')} if results else {},
        'min_legacy_covariance_error':min(r['legacy_covariance_error'] for r in results) if results else None,
        'min_native_correlation_eigenvalue':min(r['min_native_correlation_eigenvalue'] for r in results) if results else None,
        'identity_key':['input_xaod','xaod_entry_index','actual_run','actual_event'],
        'unique_run_event_pairs':len(header_groups),'cross_source_header_collisions':[
            {'header':list(key),'inputs':rows} for key,rows in header_groups.items() if len(rows)>1],
        'identity_correction_is_post_run':True,'aggregator_sha256':digest(Path(__file__)),
        'qualification':'NOT_EVALUATED','acts':'NOT_EVALUATED','coverage':'UNKNOWN',
        'pilot_is_reused':True,'pilot_validation_sha256':inputs['pilot_validation_sha256']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT);a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'):p.error('WB91 output required')
    aggregate(out)
