#!/usr/bin/env python3
"""Reporting and immutable result index; no new physical or calibration calls."""
import argparse,json,subprocess,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from wb99_contract import verify

parser=argparse.ArgumentParser();parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--cluster',type=int,required=True);parser.add_argument('--action',choices=['report','index'],required=True)
a=parser.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb99_'):parser.error('WB99 output')
freeze=verify(out);s=read_public(out/'summary.json')
for name,h in s['artifact_hashes'].items():
    if digest(out/name)!=h:raise ValueError('result changed '+name)
if a.action=='report':
    command=['bash','-c',f'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && condor_history {a.cluster} -limit 1 -json']
    r=subprocess.run(command,capture_output=True,text=True,check=True);records=json.loads(r.stdout)
    if len(records)!=1:raise ValueError('unique terminal history required')
    write_new(out/'scheduler_history.json',records);h=records[0]
    scheduler={k:h.get(k) for k in ('ClusterId','ProcId','JobStatus','ExitCode','ExitBySignal','NumJobStarts','JobRunCount','NumRestarts','LastRemoteHost','RequestCpus','RequestMemory','ShouldTransferFiles','QDate','JobCurrentStartExecutingDate','CompletionDate','RemoteUserCpu','RemoteWallClockTime')}
    scheduler.update(schedd='bigbird24.cern.ch',normal_termination=h.get('JobStatus')==4 and h.get('ExitBySignal') is False and h.get('ExitCode')==0)
    write_new(out/'scheduler_terminal_audit.json',scheduler)
    if not scheduler['normal_termination'] or read_public(out/'worker_exit.json')['exit_code']!=0 or h.get('NumJobStarts')!=1:raise ValueError('incomplete/repeated execution')
    groups={};strata={};scatter=[];actual_examples=[]
    for n,line in enumerate((out/'metrics.ndjson').open()):
        row=json.loads(line);key=(row['tolerance'],row['cap_m'],row['classification']);t=groups.setdefault(key,Counter());t['steps']+=1
        st=(row['tolerance'],row['cap_m'],row['path'],row['station'],row['sample']);v=strata.setdefault(st,Counter());v['steps']+=1
        valid=row['source_gate']=='PASS' and row['reference_gate']=='PASS'
        for label,value in (('resolved',row['direction']['resolved']),('false_negative',row['direction']['false_negative']),('near_zero_resolved',row['direction']['resolved'] and row['direction']['near_zero_estimate']),('slope_false_negative',row['slope']['false_negative'])):
            t[label]+=bool(valid and value);v[label]+=bool(valid and value)
        t['max_defect']=max(t['max_defect'],row['defect_max']);t['max_uncertainty']=max(t['max_uncertainty'],row['uncertainty'])
        t['max_budget_ratio']=max(t['max_budget_ratio'],row['defect_max']/row['direction']['budget'])
        if n%20==0 or row['direction']['false_negative']:scatter.append(row)
        if valid and (row['direction']['false_negative'] or row['slope']['false_negative']) and len(actual_examples)<100:actual_examples.append(row)
    controls=read_public(out/'controls.json');cg={};control_examples=[]
    for row in controls['rows']:
        t=cg.setdefault(row['kind'],Counter());t['rows']+=1;t['unknown']+=row['reference_gate']!='PASS';t['resolved']+=row['resolved'];t['false_negative']+=row['false_negative'];t['near_zero_resolved']+=row['resolved'] and row['near_zero_estimate']
        if row['false_negative'] and len(control_examples)<100:control_examples.append(row)
    write_new(out/'calibration_report.json',{'schema':'wb99_descriptive_report_v1','groups':[dict(tolerance=k[0],cap_m=k[1],classification=k[2],**dict(v)) for k,v in sorted(groups.items())],
      'strata':[dict(tolerance=k[0],cap_m=k[1],path=k[2],station=k[3],sample=k[4],**dict(v)) for k,v in sorted(strata.items())],
      'control_groups':{k:dict(v) for k,v in cg.items()},'control_examples':control_examples,'actual_examples':actual_examples,'qualification':'NOT_EVALUATED',
      'interpretation':'No thresholds retuned. All steps counted. Figures show every twentieth step plus all false negatives.'})
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for kind in ('domain','mesh','interior','outside'):
        rows=[x for x in scatter if x['classification']==kind]
        if rows:axes[0].scatter([max(x['direction']['budget'],1e-16) for x in rows],[max(x['defect_max'],1e-16) for x in rows],s=5,alpha=.5,label=kind)
    axes[0].plot([1e-12,1e-6],[1e-12,1e-6],'k--',lw=1);axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xlabel('Frozen direction budget');axes[0].set_ylabel('Direction defect');axes[0].legend(fontsize=8)
    for kind,marker in (('constant','o'),('domain','x'),('kink','+'),('triangle','s')):
        rows=[x for x in controls['rows'] if x['kind']==kind and x['resolved']]
        if rows:axes[1].scatter([max(x['difference_max'],1e-16) for x in rows],[max(x['defect_max'],1e-16) for x in rows],marker=marker,s=12,alpha=.6,label=kind)
    axes[1].set_xscale('log');axes[1].set_yscale('log');axes[1].set_xlabel('|u full - u two-half| max');axes[1].set_ylabel('Resolved control defect');axes[1].legend(fontsize=8)
    fig.suptitle(f"WB99: calibration={s['calibration']}; actual pilot={s['actual_pilot_hypothesis']}");fig.tight_layout();figures=out/'figures';figures.mkdir(exist_ok=False)
    for ext in ('png','pdf'):fig.savefig(figures/f'direction_calibration.{ext}',dpi=180)
    plt.close(fig)
    write_new(figures/'manifest.json',{'summary_sha256':digest(out/'summary.json'),'reporting_script_sha256':digest(Path(__file__)),
      'hashes':{x.name:digest(x) for x in figures.iterdir() if x.is_file()}})
    print({k:s[k] for k in ('execution_contract','hypothesis','actual_pilot_hypothesis','calibration','counts')})
else:
    if not read_public(out/'scheduler_terminal_audit.json')['normal_termination'] or read_public(out/'worker_exit.json')['exit_code']!=0:raise ValueError('execution incomplete')
    workbook=ROOT/'workbook/2026-10-01_99_四站完整步与双半步方向误差估计校准合同.md'
    bm=read_public(out/'binary_manifest.json')
    if digest(Path(bm['binary']))!=bm['sha256'] or digest(out/'controls')!=read_public(out/'control_binary_manifest.json')['sha256']:raise ValueError('binary changed')
    # Hash scientific evidence and logs, not transient CMake state or derivative object files.
    paths=[x for x in out.iterdir() if x.is_file()]+[x for folder in ('event','figures','identity_payload','isolated_source') for x in (out/folder).rglob('*') if x.is_file()]
    paths+=[Path(bm['binary']),workbook,Path(__file__)]+[x for x in (ROOT/'outputs/mc24_four_station_wb99_build_smoke_v1').iterdir() if x.is_file()]
    index={'schema':'wb99_result_integrity_v1','implementation_commit':freeze['commit'],'hypothesis':s['hypothesis'],'actual_pilot_hypothesis':s['actual_pilot_hypothesis'],
      'execution_contract':s['execution_contract'],'qualification':'NOT_EVALUATED','held_out_access':False,'frozen_identities_verified':len(freeze['hashes']),
      'artifacts':{str(x.relative_to(ROOT)):digest(x) for x in paths},'workbook_sha256':digest(workbook),'reporting_script_sha256':digest(Path(__file__))}
    write_new(out/'result_integrity.json',index);write_new(ROOT/'docs/wb99_direction_step_doubling_result_manifest.json',{**index,'result_integrity_sha256':digest(out/'result_integrity.json')})
    print({'artifacts':len(index['artifacts']),'identities':len(freeze['hashes'])})
