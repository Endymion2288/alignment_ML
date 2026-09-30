#!/usr/bin/env python3
"""Condor build and fail-fast paired SegmentFit development validation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from alignment.wb90_measurement_contract import ROOT as PROJECT,write_new,read_public,digest
from alignment.physical_common_track_execution import calypso_payload_command,identity_payload
from run_wb91_covariance_contract import verify,relative,covariance_error

def run(command,cwd,log):
    with log.open('x') as stream:
        rc=subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT).returncode
    if rc:raise RuntimeError(f'command failed ({rc}); see {log}')

def state_key(r):
    return (r['run'],r['event'],r['station'],tuple(sorted(r['clusters'])),r['state_index'])

def validate(work,protocol):
    pairs={}
    for mode in ('legacy','repair'):
        with (work/f'{mode}.jsonl').open() as stream:rows=[json.loads(line) for line in stream]
        if not rows:raise ValueError('no reconstructed states; no replacement')
        mapping={state_key(r):r for r in rows}
        if len(mapping)!=len(rows):raise ValueError('duplicate physical state identity')
        pairs[mode]=mapping
    if pairs['legacy'].keys()!=pairs['repair'].keys():raise ValueError('paired track/state selection changed')
    results=[]
    for key,r in pairs['repair'].items():
        old=pairs['legacy'][key]
        if not r.get('input_event_header_verified') or not old.get('input_event_header_verified'):
            raise ValueError('persisted ROOT event header not verified')
        names=('raw_fit','raw_covariance','raw_normal','native_covariance','native_parameters',
               'surface_transform','raw_to_native_jacobian','native_to_fixed_z_jacobian',
               'native_to_fixed_z_fd','fixed_z_state','fixed_z_covariance')
        values={n:np.array(r[n],dtype=float) for n in names}
        if not all(x.size and np.isfinite(x).all() for x in values.values()):raise ValueError('nonfinite audit')
        paired=max(float(np.max(np.abs(values[n]-np.array(old[n])))) for n in ('raw_fit','raw_covariance','raw_normal','fixed_z_state','native_parameters','surface_transform'))
        c=values['raw_covariance'];s=np.sqrt(np.diag(c))
        normal=values['raw_normal'];normal_scaled=normal*s[:,None]*s[None,:]
        inverse_error=relative(normal_scaled@(c/s[:,None]/s[None,:]),np.eye(4))
        dz=r['z_state_mm']-r['z_center_mm'];t=np.eye(4);t[0,2]=t[1,3]=dz
        expected=t@c@t.T;e=np.sqrt(np.diag(expected))
        nc=values['native_covariance'];ns=np.sqrt(np.diag(nc))
        eig=float(np.linalg.eigvalsh(nc/ns[:,None]/ns[None,:]).min())
        a=values['raw_to_native_jacobian'];b=values['native_to_fixed_z_jacobian'];fd=values['native_to_fixed_z_fd']
        fd_error=relative(b*ns[None,:]/e[:,None],fd*ns[None,:]/e[:,None])
        closure=relative((b@a)*s[None,:]/e[:,None],t*s[None,:]/e[:,None])
        ce=covariance_error(values['fixed_z_covariance'],expected)
        raw=values['raw_fit'].ravel();expected_state=t@raw
        state_error=float(np.max(np.abs(values['fixed_z_state'].ravel()-expected_state)))
        passed=paired<=protocol['paired_state_absolute_tolerance'] and state_error<=protocol['paired_state_absolute_tolerance'] and eig>0 and max(inverse_error,closure,ce)<=protocol['roundtrip_relative_tolerance'] and fd_error<=protocol['normalized_fd_relative_tolerance']
        results.append({'key':[key[0],key[1],key[2],list(key[3]),key[4]],'dz_mm':dz,
            'paired_error':paired,'state_error':state_error,'normal_inverse_error':inverse_error,
            'min_native_correlation_eigenvalue':eig,'fd_error':fd_error,'jacobian_roundtrip_error':closure,
            'covariance_roundtrip_error':ce,'legacy_covariance_error':covariance_error(np.array(old['fixed_z_covariance']),expected),
            'pass':bool(passed)})
    identity={(r['key'][0],r['key'][1]) for r in results}
    if len(identity)!=1:raise ValueError('multiple event headers for one input ordinal')
    logs=(work/'reconstruction.log').read_text(errors='replace')
    manifest=read_public(work/'input_manifest.json')
    if 'Reading folder /Tracker/Align from sqlite' not in logs:raise ValueError('sqlite consumption not demonstrated')
    if str(manifest['sqlite']) not in logs:raise ValueError('wrong sqlite identity in log')
    if 'WB91SegmentFit' not in logs:raise ValueError('isolated component not demonstrated')
    write_new(work/'validation.json',{'gate':'PASS' if all(r['pass'] for r in results) else 'FAIL',
        'results':results,'n_states':len(results),'actual_event_header':list(next(iter(identity))),
        'expected_input_ordinal':manifest['row']['xaod_entry_index'],
        'catalog_ids_are_ordinal_labels':True,'input_event_header_verified':True,
        'legacy_sha256':digest(work/'legacy.jsonl'),'repair_sha256':digest(work/'repair.jsonl')})
    return read_public(work/'validation.json')

def main(out):
    verify(out);protocol=read_public(out/'protocol.json')
    if read_public(out/'kernel_summary.json')['gate']!='PASS':raise ValueError('kernel failed')
    from prepare_wb91_calypso_extension import prepare
    prepare(out)
    write_new(out/'physical_environment.json',{'host':platform.node(),'platform':platform.platform(),
        'environment':{k:os.environ.get(k) for k in ('CMAKE_PREFIX_PATH','LD_LIBRARY_PATH','Athena_DIR','Calypso_DIR','AtlasVersion','AtlasProject','CMTCONFIG')},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()})
    build=out/'isolated_build'
    run(['cmake','-S',str(out/'isolated_source'),'-B',str(build),'-DCalypso_DIR='+str(PROJECT.parent/'calypso/run/cmake')],out,out/'configure.log')
    run(['cmake','--build',str(build),'-j','1'],out,out/'build.log')
    platform_dir=build/'x86_64-el9-gcc13-opt'
    if not (platform_dir/'setup.sh').is_file():raise ValueError('missing isolated runtime setup')
    binaries=list(platform_dir.rglob('*WB91SegmentFit*.so'))
    if len(binaries)!=1:raise ValueError('ambiguous isolated library')
    write_new(out/'binary_manifest.json',{'binary':str(binaries[0]),'sha256':digest(binaries[0]),
        'source_manifest_sha256':digest(out/'extension_manifest.json')})
    payload=out/'identity_payload'
    run(['bash','-c',calypso_payload_command(output_dir=payload,payload=identity_payload())],out,out/'payload.log')
    summaries=[]
    for index,row in enumerate(read_public(out/'selection.json')['events']):
        work=out/'physical'/f'{index:02d}';work.mkdir(parents=True,exist_ok=False)
        command='\n'.join(['source '+shlex.quote(str(PROJECT/'scripts/setup_environment.sh'))+' calypso',
            'source '+shlex.quote(str(platform_dir/'setup.sh')),
            'export LD_LIBRARY_PATH='+shlex.quote(str(platform_dir/'lib'))+':"$LD_LIBRARY_PATH"',
            'python '+shlex.quote(str(PROJECT/'scripts/wb91_reconstruct.py'))+' --output-root '+shlex.quote(str(out))+
            ' --index '+str(index)+' --work-dir '+shlex.quote(str(work))+' --sqlite '+shlex.quote(str(payload/'tracker_alignment.sqlite'))])
        write_new(work/'command.json',{'bash_script':command,'input':row,'binary_sha256':digest(binaries[0])})
        run(['bash','-c',command],work,work/'reconstruction.log')
        summary=validate(work,protocol);summaries.append(summary)
        if summary['gate']!='PASS':break
    write_new(out/'physical_summary.json',{'gate':'PASS' if len(summaries)==24 and all(s['gate']=='PASS' for s in summaries) else 'FAIL',
        'n_events':len(summaries),'n_states':sum(s['n_states'] for s in summaries),
        'event_summaries':[{k:v for k,v in s.items() if k!='results'} for s in summaries],
        'qualification':'NOT_EVALUATED','acts':'NOT_EVALUATED','coverage':'UNKNOWN'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True,type=Path);args=p.parse_args()
    out=args.output_root.resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'):p.error('new WB91 output required')
    try:main(out)
    except Exception as e:
        write_new(out/'physical_error.json',{'status':'EXECUTION_OR_VALIDATION_ERROR','error':repr(e),'qualification':'NOT_EVALUATED'})
        raise
