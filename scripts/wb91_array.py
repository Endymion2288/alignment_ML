#!/usr/bin/env python3
"""Immutable pilot reuse, 23-event Condor array, and explicit result aggregation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT as PROJECT,read_public,write_new,digest
from run_wb91_covariance_contract import verify,OUTPUT
from run_wb91_physical_validation import validate,run

def prepare(out,pilot):
    verify(out)
    if read_public(out/'kernel_summary.json')['gate']!='PASS':raise ValueError('kernel failed')
    summary=read_public(pilot/'physical/00/validation.json')
    if summary['gate']!='PASS' or not summary['input_event_header_verified']:raise ValueError('pilot failed')
    for mode in ('legacy','repair'):
        if digest(pilot/'physical/00'/f'{mode}.jsonl')!=summary[f'{mode}_sha256']:raise ValueError('pilot artifact changed')
    if read_public(out/'selection.json')!=read_public(pilot/'selection.json'):raise ValueError('array selection changed')
    current=read_public(out/'freeze.json');prior=read_public(pilot/'freeze.json')
    for name in ('research/wb91/CovarianceContract.h','research/wb91/Audit.h','scripts/prepare_wb91_calypso_extension.py','configs/research_review/wp91_covariance_repair.json'):
        p=str((PROJECT/name).resolve())
        if current['source_hashes'][p]!=prior['source_hashes'][p]:raise ValueError('pilot/build science changed')
    binary=read_public(pilot/'binary_manifest.json')
    if digest(Path(binary['binary']))!=binary['sha256']:raise ValueError('binary changed')
    payload=pilot/'identity_payload'
    payload_hashes={str(p.relative_to(payload)):digest(p) for p in payload.rglob('*') if p.is_file()}
    shutil.copytree(payload,out/'identity_payload')
    for name,h in payload_hashes.items():
        if digest(out/'identity_payload'/name)!=h:raise ValueError('payload copy mismatch')
    # XML catalog intentionally retains the immutable v6 POOL PFN and GUID.
    write_new(out/'array_inputs.json',{'pilot_root':str(pilot),'pilot_validation_sha256':digest(pilot/'physical/00/validation.json'),
        'binary':binary,'payload_source':str(payload),'payload_hashes':payload_hashes,
        'pool_pfn_is_original_immutable_v6':True,'indices':list(range(1,24))})
    worker=PROJECT/'scripts/run_wb91_array_condor.sh'
    sub='\n'.join(['universe = vanilla',f'executable = {worker}',f'arguments = {PROJECT} {out} $(event_index)',
        f'output = {out}/condor.$(ClusterId).$(ProcId).out',f'error = {out}/condor.$(ClusterId).$(ProcId).err',
        f'log = {out}/condor.$(ClusterId).log','request_cpus = 1','request_memory = 8000',
        'request_disk = 8000000','requirements = (Arch == "X86_64")','+JobFlavour = "workday"',
        'getenv = False','should_transfer_files = NO','queue event_index in ('+' '.join(str(x) for x in range(1,24))+')',''])
    with (out/'array.sub').open('x') as s:s.write(sub)

def event(out,index):
    verify(out);f=read_public(out/'array_inputs.json')
    if index not in f['indices']:raise ValueError('unfrozen event index')
    binary=f['binary']
    if digest(Path(binary['binary']))!=binary['sha256']:raise ValueError('binary changed')
    for name,h in f['payload_hashes'].items():
        if digest(out/'identity_payload'/name)!=h:raise ValueError('payload changed')
    work=out/'physical'/f'{index:02d}';work.mkdir(parents=True,exist_ok=False)
    try:
        platform_dir=Path(binary['binary']).parent.parent
        script='\n'.join(['source '+shlex.quote(str(PROJECT/'scripts/setup_environment.sh'))+' calypso',
            'source '+shlex.quote(str(platform_dir/'setup.sh')),
            'export LD_LIBRARY_PATH='+shlex.quote(str(platform_dir/'lib'))+':"$LD_LIBRARY_PATH"',
            'python '+shlex.quote(str(PROJECT/'scripts/wb91_reconstruct.py'))+' --output-root '+shlex.quote(str(out))+
            ' --index '+str(index)+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(out/'identity_payload/tracker_alignment.sqlite'))])
        write_new(work/'command.json',{'bash_script':script,'binary_sha256':binary['sha256'],
            'condor_job':os.environ.get('_CONDOR_JOB_AD')})
        run(['bash','-c',script],work,work/'reconstruction.log')
        validate(work,read_public(out/'protocol.json'))
    except Exception as e:
        write_new(work/'execution_error.json',{'error':repr(e),'qualification':'NOT_EVALUATED'})
        raise

def aggregate(out):
    verify(out);inputs=read_public(out/'array_inputs.json')
    pilot=Path(inputs['pilot_root'])
    if digest(pilot/'physical/00/validation.json')!=inputs['pilot_validation_sha256']:raise ValueError('pilot changed')
    paths=[pilot/'physical/00']+[out/'physical'/f'{i:02d}' for i in inputs['indices']]
    summaries=[];missing=[]
    for i,work in enumerate(paths):
        if not (work/'validation.json').is_file():
            missing.append({'index':i,'error':read_public(work/'execution_error.json') if (work/'execution_error.json').exists() else 'UNKNOWN'})
            continue
        s=read_public(work/'validation.json')
        for mode in ('legacy','repair'):
            if digest(work/f'{mode}.jsonl')!=s[f'{mode}_sha256']:raise ValueError('audit changed')
        s['index']=i;s['validation_path']=str(work/'validation.json');summaries.append(s)
    results=[r for s in summaries for r in s['results']]
    good=len(summaries)==24 and all(s['gate']=='PASS' and s['input_event_header_verified'] for s in summaries)
    identity={(s['actual_event_header'][0],s['actual_event_header'][1]) for s in summaries}
    if len(identity)!=len(summaries):raise ValueError('event header repeated across frozen inputs')
    write_new(out/'physical_summary.json',{'gate':'PASS' if good else 'FAIL' if any(s['gate']=='FAIL' for s in summaries) else 'INCOMPLETE',
        'n_events':len(summaries),'n_states':len(results),'missing':missing,
        'event_summaries':[{k:v for k,v in s.items() if k!='results'} for s in summaries],
        'max_errors':{k:max(r[k] for r in results) for k in ('paired_error','state_error','normal_inverse_error','fd_error','jacobian_roundtrip_error','covariance_roundtrip_error','legacy_covariance_error')} if results else {},
        'min_legacy_covariance_error':min(r['legacy_covariance_error'] for r in results) if results else None,
        'min_native_correlation_eigenvalue':min(r['min_native_correlation_eigenvalue'] for r in results) if results else None,
        'qualification':'NOT_EVALUATED','acts':'NOT_EVALUATED','coverage':'UNKNOWN',
        'pilot_is_reused':True,'pilot_validation_sha256':inputs['pilot_validation_sha256']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','event','aggregate'])
    p.add_argument('--output-root',type=Path,default=OUTPUT);p.add_argument('--pilot-root',type=Path)
    p.add_argument('--index',type=int);a=p.parse_args();out=a.output_root.resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'):p.error('WB91 output required')
    if a.action=='prepare':
        pilot=a.pilot_root.resolve() if a.pilot_root else None
        if not pilot or not pilot.is_relative_to(PROJECT/'outputs') or not pilot.name.startswith('mc24_four_station_wb91_'):p.error('WB91 pilot required')
        prepare(out,pilot)
    elif a.action=='event':event(out,a.index)
    else:aggregate(out)
