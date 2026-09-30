#!/usr/bin/env python3
"""Freeze and evaluate a versioned covariance/frame repair on seen development."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from alignment.wb90_measurement_contract import ROOT as PROJECT, digest, read_public, write_new, development_slopes

OUTPUT=PROJECT/'outputs/mc24_four_station_wb91_covariance_repair_v6'
PROTOCOL=PROJECT/'configs/research_review/wp91_covariance_repair.json'
SELECTION=PROJECT/'outputs/mc24_four_station_wb90_measurement_contract_v1/selection.json'
RELEASE=Path('/cvmfs/atlas.cern.ch/repo/sw/software/24.0/Athena/24.0.41/InstallArea/x86_64-el9-gcc13-opt')

def relative(actual,expected):
    return float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1.e-300))

def covariance_error(actual,expected):
    scale=np.sqrt(np.diag(expected))
    return relative(actual/scale[:,None]/scale[None,:],expected/scale[:,None]/scale[None,:])

def freeze(out):
    if out.exists(): raise FileExistsError(out)
    paths=[PROTOCOL,SELECTION,Path(__file__).resolve(),PROJECT/'alignment/wb90_measurement_contract.py',
           PROJECT/'research/wb91/CovarianceContract.h',PROJECT/'scripts/wb91_trk_probe.cpp.in',
           PROJECT/'scripts/setup_environment.sh',PROJECT/'research/wb91/Audit.h',
           PROJECT/'scripts/prepare_wb91_calypso_extension.py',PROJECT/'scripts/wb91_reconstruct.py',
           PROJECT/'scripts/run_wb91_physical_validation.py',PROJECT/'scripts/run_wb91_condor.sh',
           PROJECT/'scripts/submit_wb91_condor.py',PROJECT/'scripts/write_station_alignment_payload.py',
           PROJECT/'alignment/physical_common_track_execution.py']
    paths += [RELEASE/f'lib/{lib}.so' for lib in ('libTrkParameters','libTrkSurfaces')]
    paths += [RELEASE/f'include/{p}' for p in ('TrkParametersBase/CurvilinearParametersT.icc','TrkEventPrimitives/CurvilinearUVT.h','TrkSurfaces/Surface.h')]
    rows=[r for r in read_public(SELECTION)['events'] if r['role']=='development']
    if len(rows)!=24: raise ValueError('expected exact 24 WB90 development events')
    write_new(out/'protocol.json',read_public(PROTOCOL))
    write_new(out/'selection.json',{'events':rows,'role':'historically seen development','n_events':24})
    paths += [PROJECT/r['replica_path'] for r in rows]
    fitter=PROJECT.parent/'calypso/Tracker/TrackerRecAlgs/TrackerSegmentFit'
    paths += [fitter/'src/SegmentFitAlg.cxx',fitter/'src/SegmentFitAlg.h',fitter/'CMakeLists.txt']
    write_new(out/'freeze.json',{'source_hashes':{str(p.resolve()):digest(p) for p in paths},
        'protocol_sha256':digest(out/'protocol.json'),'selection_sha256':digest(out/'selection.json'),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],text=True),
        'check_numerical_cases_opened':False,'qualification':'NOT_EVALUATED'})

def verify(out):
    f=read_public(out/'freeze.json')
    for p,h in f['source_hashes'].items():
        if digest(Path(p))!=h: raise ValueError(f'frozen source changed: {p}')
    for p in ('protocol','selection'):
        if digest(out/f'{p}.json')!=f[f'{p}_sha256']: raise ValueError(f'{p} changed')
    return f

def probe(out):
    f=verify(out)
    if (out/'kernel_summary.json').exists(): raise FileExistsError('completed probe exists')
    import ROOT
    libraries={}
    for lib in ('libTrkParameters','libTrkSurfaces'):
        if ROOT.gSystem.Load(lib)<0: raise RuntimeError(f'cannot load {lib}')
        p=Path(str(ROOT.gSystem.DynamicPathName(lib+'.so',True))).resolve()
        if f['source_hashes'].get(str(p))!=digest(p): raise ValueError('wrong Trk binary')
        libraries[lib]={'path':str(p),'sha256':digest(p)}
    for prefix in os.environ.get('CMAKE_PREFIX_PATH','').split(':'):
        for suffix in ('include','include/eigen3'):
            p=Path(prefix)/suffix
            if p.is_dir(): ROOT.gInterpreter.AddIncludePath(str(p))
    ROOT.gInterpreter.AddIncludePath(str(PROJECT/'research/wb91'))
    code=(PROJECT/'scripts/wb91_trk_probe.cpp.in').read_text()
    with (out/'compiled_probe.cpp').open('x') as s:s.write(code)
    if not ROOT.gInterpreter.Declare(code): raise RuntimeError('Trk probe compilation failed')
    protocol=read_public(out/'protocol.json')
    lower=np.array(protocol['covariance_cholesky']);c=lower@lower.T
    cases=[{'kind':'analytic','tx':tx,'ty':ty} for tx,ty in protocol['analytic_slopes']]
    cases += [{'kind':'seen_development',**r} for r in development_slopes(read_public(out/'selection.json')['events'])]
    rows=[]
    for case in cases:
        for dz in protocol['lever_arms_mm']:
            v=np.array(list(ROOT.WB91Probe.evaluate(case['tx'],case['ty'],dz,list(c.ravel()))))
            if len(v)!=209 or not np.isfinite(v).all(): raise ValueError(f'invalid vector: {len(v)}')
            a,afd=v[:20].reshape(5,4),v[20:40].reshape(5,4)
            b,fd,half=(v[k:k+20].reshape(4,5) for k in (40,60,80))
            native=v[100:125].reshape(5,5);export=v[125:141].reshape(4,4)
            product=v[141:157].reshape(4,4)
            t=np.eye(4);t[0,2]=dz;t[1,3]=dz;expected=t@c@t.T
            scale=np.sqrt(np.diag(c));ns=np.sqrt(np.diag(native))
            ae=relative(a*scale[None,:]/ns[:,None],afd*scale[None,:]/ns[:,None])
            es=np.sqrt(np.diag(expected))
            be=relative(b*ns[None,:]/es[:,None],fd*ns[None,:]/es[:,None])
            he=relative(half*ns[None,:]/es[:,None],fd*ns[None,:]/es[:,None])
            ce=covariance_error(export,expected)
            pe=relative(product*scale[None,:]/es[:,None],t*scale[None,:]/es[:,None])
            neg=[covariance_error(v[k:k+16].reshape(4,4),expected) for k in (157,173,189)]
            spd=bool(np.linalg.eigvalsh(native/ns[:,None]/ns[None,:]).min()>0)
            passed=max(ae,be,he)<=protocol['normalized_fd_relative_tolerance'] and max(ce,pe)<=protocol['roundtrip_relative_tolerance'] and spd
            rows.append({**case,'dz_mm':dz,'raw_to_native_fd_error':ae,'native_to_fixed_z_fd_error':be,
                'fd_halfstep_error':he,'covariance_roundtrip_error':ce,'jacobian_roundtrip_error':pe,
                'native_spd':spd,'legacy_error':neg[0],'sign_only_error':neg[1],'raw_position_export_error':neg[2],
                'pass':bool(passed),'native_covariance':native.tolist(),'export_covariance':export.tolist(),
                'raw_to_native_jacobian':a.tolist(),'native_to_fixed_z_jacobian':b.tolist()})
    # A negative control must fail somewhere in the preregistered analytic set.
    negpass=all(any(r[key]>protocol['negative_control_min_relative_error'] for r in rows if r['kind']=='analytic')
                for key in ('legacy_error','sign_only_error','raw_position_export_error'))
    write_new(out/'kernel_results.json',{'results':rows,'libraries':libraries,'root':ROOT.gROOT.GetVersion(),
        'covariance_is_actual_fit':False,'probe_cpp_sha256':digest(out/'compiled_probe.cpp')})
    write_new(out/'kernel_summary.json',{'gate':'PASS' if all(r['pass'] for r in rows) and negpass else 'FAIL',
        'n_kernels':len(rows),'n_development_events':24,'negative_controls_detected':negpass,
        'n_failures':sum(not r['pass'] for r in rows),'qualification':'NOT_EVALUATED','acts':'NOT_EVALUATED',
        'max_errors':{k:max(r[k] for r in rows) for k in ('raw_to_native_fd_error','native_to_fixed_z_fd_error','fd_halfstep_error','covariance_roundtrip_error','jacobian_roundtrip_error')},
        'result_sha256':digest(out/'kernel_results.json'),'freeze_sha256':digest(out/'freeze.json')})
    print(json.dumps(read_public(out/'kernel_summary.json'),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze','probe']);parser.add_argument('--output-root',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output_root.resolve()
    if not out.is_relative_to(PROJECT/'outputs') or not out.name.startswith('mc24_four_station_wb91_'): parser.error('new WB91 output required')
    (freeze if args.action=='freeze' else probe)(out)
