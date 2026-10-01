#!/usr/bin/env python3
"""Reporting only: complete scheduler evidence, figures and immutable index."""
import argparse,sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from alignment.wb98_defect_transport import vector,norm
from wb98_contract import verify
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--cluster',type=int,required=True);p.add_argument('--action',choices=['report','index'],required=True)
a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb98_'):p.error('WB98 output required')
freeze=verify(out);s=read_public(out/'summary.json')
for name,key in (('steps.ndjson','steps_sha256'),('traces.json','traces_sha256'),('freeze.json','freeze_sha256')):
    if digest(out/name)!=s[key]:raise ValueError('raw/freeze changed')
if a.action=='report':
    command=['bash','-c',f'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && condor_history {a.cluster} -limit 1 -json']
    r=subprocess.run(command,capture_output=True,text=True,check=True);records=json.loads(r.stdout)
    if len(records)!=1:raise ValueError('missing unique history')
    history=records[0];write_new(out/'scheduler_history.json',records)
    audit={key:history.get(key) for key in ('ClusterId','ProcId','JobStatus','JobRunCount','NumJobStarts','NumRestarts','LastRemoteHost','RequestCpus','RequestMemory','ShouldTransferFiles','QDate','JobCurrentStartExecutingDate','CompletionDate','RemoteUserCpu','RemoteWallClockTime')}
    audit.update(schedd='bigbird24.cern.ch',exit_code=history.get('ExitCode'),normal_termination=history.get('JobStatus')==4 and history.get('ExitBySignal') is False and history.get('ExitCode')==0)
    write_new(out/'scheduler_terminal_audit.json',audit)
    if not audit['normal_termination'] or read_public(out/'worker_exit.json')['exit_code']!=0:raise ValueError('execution incomplete')
    rows=[]
    for t in s['traces_detail']:
        if len(t['endpoint_ladders'])!=3:continue
        last=t['endpoint_ladders'][-1];actual=vector(last['actual_same_z_defect']);prediction=vector(last['prediction']);position=vector(last['position_contribution']);slope=vector(last['slope_contribution']);field=vector(last['fixed_stage_arithmetic_contribution'])
        rows.append({key:t[key] for key in ('trace','tolerance','cap_m','path','station','sample','closure_gate','closure_budget_scaled','global_gate','reference_unknown','jacobian_unknown','fd_unknown')}
            | {'actual_defect_scaled':norm(actual),'prediction_scaled':norm(prediction),'position_contribution_scaled':norm(position),'slope_contribution_scaled':norm(slope),
               'slope_only_residual_scaled':norm(slope-actual),'position_only_residual_scaled':norm(position-actual),
               'fixed_stage_arithmetic_scaled':norm(field),'arithmetic_only_residual_scaled':norm(field-actual),
               'closure_scaled':norm(prediction-actual),'split_residual_scaled':norm(position+slope-prediction),
               'same_arc_minus_same_z_scaled':norm(t['same_arc_minus_same_z_X']),'plane_minus_same_z_scaled':norm(t['plane_minus_same_z_X']),
               'float_double_same_point_max_T':t['same_point_float_double_max_T'],
               'actual_defect':actual.tolist(),'position_contribution':position.tolist(),'slope_contribution':slope.tolist(),'fixed_stage_arithmetic_contribution':field.tolist()})
    write_new(out/'contribution_report.json',{'schema':'wb98_descriptive_contributions_v1','rows':rows,
       'interpretation':'Chart-dependent vector diagnostics; fixed-stage arithmetic is not a float-backend replacement. No new gates or thresholds.',
       'qualification':'NOT_EVALUATED','reporting_script_sha256':digest(Path(__file__))})
    figures=out/'figures';figures.mkdir(exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    good=[r for r in rows if r['closure_gate']!='UNKNOWN']
    x=np.array([r['actual_defect_scaled'] for r in good]);y=np.array([r['closure_scaled'] for r in good]);b=np.array([r['closure_budget_scaled'] for r in good])
    if len(x):
        axes[0].scatter(np.maximum(x,1e-16),np.maximum(y,1e-16),s=10,label='Closure residual');axes[0].scatter(np.maximum(x,1e-16),b,s=8,marker='_',label='Frozen budget')
    axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xlabel('Actual endpoint defect (scaled)');axes[0].set_ylabel('Residual / budget (scaled)');axes[0].legend(fontsize=8)
    direct=[r for r in rows if r['path']=='direct' and r['sample']==0]
    for key,label in (('position_contribution_scaled','Position injections'),('slope_contribution_scaled','Slope injections'),('fixed_stage_arithmetic_scaled','Fixed-stage field arithmetic')):
        axes[1].scatter([r['trace'] for r in direct],[max(r[key],1e-16) for r in direct],s=12,label=label)
    axes[1].set_yscale('log');axes[1].set_xlabel('Frozen nominal direct trace index');axes[1].set_ylabel('Transported contribution (scaled)');axes[1].legend(fontsize=8)
    fig.suptitle(f"WB98: all 156 traces; H={s['hypothesis']}")
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(figures/f'defect_transport_contract.{ext}',dpi=180)
    plt.close(fig)
    write_new(figures/'manifest.json',{'summary_sha256':digest(out/'summary.json'),'reporting_script_sha256':digest(Path(__file__)),
        'hashes':{p.name:digest(p) for p in figures.iterdir() if p.is_file()}})
    print({k:s[k] for k in ('hypothesis','closure_pass','closure_fail','closure_unknown','local_reference_unknown','local_jacobian_unknown','fd_unknown')})
else:
    history=read_public(out/'scheduler_terminal_audit.json')
    if not history['normal_termination'] or read_public(out/'worker_exit.json')['exit_code']!=0:raise ValueError('execution incomplete')
    if digest(out/'defect_transport')!=read_public(out/'binary_manifest.json')['sha256']:raise ValueError('binary changed')
    workbook=ROOT/'workbook/2026-10-01_98_四站局部缺陷到终态与混合响应传递闭合研究.md'
    source_audit=read_public(out/'saved_transport_source_audit.json')
    if source_audit['gate']!='PASS':raise ValueError('posthoc source/FD audit failed')
    paths=[p for p in out.rglob('*') if p.is_file()]+[workbook,Path(__file__),ROOT/'scripts/audit_wb98_saved_transport.py']
    control=ROOT/'outputs/mc24_four_station_wb98_control_test_v1';paths+=[control/'defect_transport',control/'controls.json']
    index={'schema':'wb98_result_integrity_v1','implementation_commit':freeze['commit'],'hypothesis':s['hypothesis'],'execution_contract':s['execution_contract'],
       'numerical_prerequisites':s['numerical_prerequisites'],'qualification':'NOT_EVALUATED','held_out_access':False,'frozen_identities_verified':len(freeze['hashes']),
       'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in paths},'workbook_sha256':digest(workbook),'reporting_script_sha256':digest(Path(__file__))}
    write_new(out/'result_integrity.json',index);write_new(ROOT/'docs/wb98_defect_transport_result_manifest.json',{**index,'result_integrity_sha256':digest(out/'result_integrity.json')})
    print({'artifacts':len(index['artifacts']),'identities':len(freeze['hashes'])})
