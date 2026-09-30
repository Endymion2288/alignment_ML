#!/usr/bin/env python3
"""Record completed WB91 evidence, hashes and limitations without new physics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from alignment.wb90_measurement_contract import ROOT as PROJECT,read_public,write_new,digest,ALLOWLIST
from run_wb91_covariance_contract import OUTPUT,verify

def finalize(out):
    freeze=verify(out);summary=read_public(out/'physical_summary.json')
    if summary['gate']!='PASS' or summary['n_events']!=24:raise ValueError('incomplete/nonpassing physical contract')
    sources={r['source_id']:r for r in read_public(ALLOWLIST)['sources']}
    selection=read_public(out/'selection.json')['events'];source_stats=[]
    for source in sorted({r['source_id'] for r in selection}):
        row=next(r for r in selection if r['source_id']==source);p=Path(row['input_xaod']);stat=p.stat();prior=sources[source]
        if str(p)!=prior['path'] or stat.st_size!=prior['size_bytes'] or stat.st_mtime_ns!=prior['mtime_ns']:
            raise ValueError('original allowlist source stat changed')
        source_stats.append({'source_id':source,'path':str(p),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                             'matches_original_allowlist':True,'full_root_sha256':'NOT_COMPUTED'})
    records=[];reference=[];artifacts={}
    for s in summary['event_summaries']:
        path=Path(s['validation_path']);work=path.parent
        v=read_public(path);artifacts[str(path.relative_to(PROJECT))]=digest(path)
        for mode in ('legacy','repair'):
            p=work/f'{mode}.jsonl';artifacts[str(p.relative_to(PROJECT))]=digest(p)
            if digest(p)!=v[f'{mode}_sha256']:raise ValueError('completed event changed')
        with (work/'repair.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
        manifest=read_public(work/'input_manifest.json');expected=selection[s['index']]
        if manifest['row']!=expected:raise ValueError('input selection mismatch')
        if any([r['run'],r['event']]!=v['actual_event_header'] for r in rows):raise ValueError('state header mismatch')
        for r in rows:
            raw=np.array(r['raw_fit']).ravel();c=np.array(r['raw_covariance']);native=np.array(r['native_covariance'])
            records.append({'index':s['index'],'station':r['station'],'state_index':r['state_index'],
                'r':float(np.hypot(*raw[2:])), 'sigma_tx':float(np.sqrt(c[2,2])),
                'sigma_ty':float(np.sqrt(c[3,3])),'sigma_phi_rad':float(np.sqrt(native[2,2])),
                'sigma_theta_rad':float(np.sqrt(native[3,3])),
                'dz_mm':r['z_state_mm']-r['z_center_mm'],'q_over_p':r['native_parameters'][4][0],
                'q_over_p_variance':r['native_covariance'][4][4]})
        reference += [r for r in records if r['index']==s['index'] and r['state_index']==0]
    if len(records)!=summary['n_states']:raise ValueError('state count mismatch')
    for r in records:
        if r['q_over_p']!=1.e-5 or abs(r['q_over_p_variance']-5.e-6)>1.e-18:raise ValueError('dummy q/p changed')
    for name in ('freeze.json','protocol.json','selection.json','kernel_summary.json','kernel_results.json','physical_summary.json','array_inputs.json'):
        artifacts[str((out/name).relative_to(PROJECT))]=digest(out/name)
    original_fitter=PROJECT.parent/'calypso/Tracker/TrackerRecAlgs/TrackerSegmentFit/src/SegmentFitAlg.cxx'
    if digest(original_fitter)!=freeze['source_hashes'][str(original_fitter.resolve())]:raise ValueError('original fitter changed')
    kernels=read_public(out/'kernel_results.json')['results']
    write_new(out/'result_integrity.json',{'schema':'wb91_result_integrity_v1','data_role':'historically seen development',
        'source_stat_verification':source_stats,'artifacts_sha256':artifacts,
        'source_allowlist_sha256':digest(ALLOWLIST),'original_fitter_unchanged':True,
        'n_reference_states':len(reference),'n_states':len(records),'stations':sorted({r['station'] for r in records}),
        'dz_mm_range':[min(r['dz_mm'] for r in records),max(r['dz_mm'] for r in records)],
        'native_angle_diagnostics':{'n_reference_sigma_phi_gt_pi':sum(r['sigma_phi_rad']>np.pi for r in reference),
            'sigma_phi_range_rad':[min(r['sigma_phi_rad'] for r in reference),max(r['sigma_phi_rad'] for r in reference)],
            'reference_states':reference,'purpose':'retrospective chart-risk diagnostic; no gate or coverage claim'},
        'negative_control_max_analytic_errors':{key:max(r[key] for r in kernels if r['kind']=='analytic') for key in ('legacy_error','sign_only_error','raw_position_export_error')},
        'input_header_equals_catalog_ordinal':all(s['actual_event_header'][1]==selection[s['index']]['xaod_entry_index'] for s in summary['event_summaries']),
        'q_over_p_dummy_preserved':True,'independent_qualification':False,'coverage':'UNKNOWN','acts':'NOT_EVALUATED',
        'heldout_or_sealed_opened':False,'finalizer_sha256':digest(Path(__file__))})
    print(json.dumps({k:v for k,v in read_public(out/'result_integrity.json').items() if k not in ('artifacts_sha256','source_stat_verification','native_angle_diagnostics')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,default=OUTPUT);a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'):p.error('WB91 output required')
    finalize(out)
