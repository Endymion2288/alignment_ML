#!/usr/bin/env python3
"""Report or seal an already-completed WB100; never reruns the diagnostic."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb100_contract import verify,digest,reference_source

WORKBOOK=ROOT/'workbook/2026-10-01_100_四站域与网格分段方向误差包络校准合同.md'

def report(out):
    verify(out);s=read_public(out/'summary.json');receipt=read_public(out/'worker_exit.json')
    if receipt['exit_code']!=0:raise ValueError('worker infrastructure failure')
    sub=read_public(out/'submission.json');cluster=re.search(r'cluster (\d+)',sub['stdout']).group(1)
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null && condor_history '+cluster+' -json'
    r=subprocess.run(['bash','-c',cmd],text=True,capture_output=True,check=True)
    history=json.loads(r.stdout)
    if len(history)!=1:raise ValueError('single scheduler history missing')
    ad=history[0]
    if ad.get('JobStatus')!=4 or ad.get('ExitCode')!=0 or ad.get('NumJobStarts')!=1 or ad.get('ExitBySignal',False):
        raise ValueError('scheduler completion contract')
    write_new(out/'scheduler_history.json',history)
    write_new(out/'scheduler_terminal_audit.json',{'gate':'PASS','cluster':cluster,'starts':ad['NumJobStarts'],'exit_code':ad['ExitCode'],
        'status':ad['JobStatus'],'RequestCpus':ad.get('RequestCpus'),'RequestMemory':ad.get('RequestMemory'),
        'ShouldTransferFiles':ad.get('ShouldTransferFiles'),'worker':ad.get('LastRemoteHost'),'RemoteWallClockTime':ad.get('RemoteWallClockTime'),
        'stdout_command':cmd,'stderr':r.stderr})
    freeze=read_public(out/'freeze.json')
    if 'parent' in freeze:
        parent=Path(freeze['parent']);parent_sub=read_public(parent/'submission.json')
        parent_cluster=re.search(r'cluster (\d+)',parent_sub['stdout']).group(1)
        cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null && condor_history '+parent_cluster+' -json'
        old=subprocess.run(['bash','-c',cmd],text=True,capture_output=True,check=True);oldhistory=json.loads(old.stdout)
        if len(oldhistory)!=1 or oldhistory[0].get('ExitCode')!=1 or oldhistory[0].get('NumJobStarts')!=1:raise ValueError('original failure history')
        write_new(out/'primary_scheduler_history.json',oldhistory)
        write_new(out/'primary_execution_audit.json',{'execution':'FAIL','exit_code':1,'cluster':parent_cluster,
             'C++_completed_steps':read_public(parent/'terminal.json')['steps'],'C++_completed_traces':len(read_public(parent/'terminal.json')['traces']),
             'failure':'NumPy int64 count JSON serialization','original_artifacts_preserved':True,'C++_rerun':False,'new_wrapper_queries':0})
    bm=read_public(out/'binary_manifest.json')
    if digest(out/'diagnostic')!=bm['sha256'] or digest(out/'ControlReference.inc')!=bm['reference_header_sha256'] or (out/'ControlReference.inc').read_text()!=reference_source():
        raise ValueError('binary or immutable reference source changed')
    if read_public(out/'numerical_audit.json')['gate']!='PASS':raise ValueError('all-row numerical audit')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    with (out/'metrics.ndjson').open() as f:
        for line in f:
            row=json.loads(line)
            if row['evaluation']['resolved'] and row['classification']!='domain':rows.append(row)
    ratio=np.array([r['envelope']['information_ratio'] for r in rows]);h=np.array([r['h_mm'] for r in rows])
    defects=np.array([r['evaluation']['defect'] for r in rows]);E=np.array([r['envelope']['E'] for r in rows])
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].scatter(h,ratio,s=3,alpha=.2);ax[0].set(xscale='log',yscale='log',xlabel='Saved accepted arc (mm)',ylabel='Envelope E / baseline C',title='Resolved seen pilot, non-domain')
    for value in (.1,.5):ax[0].axhline(value,color='red',linestyle='--',linewidth=1)
    ax[1].scatter(defects,E,s=3,alpha=.2);ax[1].set(xscale='log',yscale='log',xlabel='Independent direction defect',ylabel='Envelope E',title='Coverage is distinct from usefulness')
    limits=[min(defects.min(),E.min()),max(defects.max(),E.max())];ax[1].plot(limits,limits,'k--',linewidth=1)
    fig.tight_layout();directory=out/'figures';directory.mkdir(exist_ok=False)
    fig.savefig(directory/'cell_envelope.png',dpi=180);fig.savefig(directory/'cell_envelope.pdf');plt.close(fig)
    write_new(directory/'manifest.json',{x.name:digest(x) for x in directory.iterdir()})
    # No selected subsets: compare every old reliable miss descriptively.
    oldmisses=covered=0;maximum_ratio=0.
    with (out/'metrics.ndjson').open() as f,(out.parent/'mc24_four_station_wb99_aggregation_recovery_v1/metrics.ndjson').open() as old:
        for a,b in zip(f,old):
            current,prior=json.loads(a),json.loads(b)
            if (current['trace'],current['index'])!=(prior['trace'],prior['index']):raise ValueError('old miss identity')
            if prior['direction']['false_negative']:
                oldmisses+=1;covered+=not current['evaluation']['false_negative'];maximum_ratio=max(maximum_ratio,current['evaluation']['defect']/current['evaluation']['budget'])
    write_new(out/'calibration_report.json',{'gate':'PASS','WB99_seen_misses':oldmisses,'WB100_covered_old_misses':covered,'maximum_old_miss_defect_over_new_budget':maximum_ratio,
        'summary_sha256':digest(out/'summary.json'),'numerical_audit_sha256':digest(out/'numerical_audit.json'),
        'hypothesis':s['hypothesis'],'qualification':'NOT_EVALUATED'})

def seal(out):
    freeze=verify(out);s=read_public(out/'summary.json')
    if read_public(out/'scheduler_terminal_audit.json')['gate']!='PASS':raise ValueError('scheduler gate')
    paths=[x for x in out.rglob('*') if x.is_file()]
    # Preserve failed prequalification smoke evidence as well as the passing smoke.
    for name in ('mc24_four_station_wb100_build_smoke_v1','mc24_four_station_wb100_build_smoke_v2','mc24_four_station_wb100_build_smoke_v3'):
        paths.extend(x for x in (ROOT/'outputs'/name).rglob('*') if x.is_file())
    if 'parent' in freeze:paths.extend(x for x in Path(freeze['parent']).rglob('*') if x.is_file())
    paths.extend((WORKBOOK,Path(__file__).resolve()))
    value={'schema':'wb100_result_integrity_v1','implementation_commit':freeze['commit'],'frozen_identities_verified':len(freeze['hashes']),
        'artifacts':{str(x.relative_to(ROOT)):digest(x) for x in paths},'workbook_sha256':digest(WORKBOOK),
        'execution_contract':s['execution_contract'],'coverage':s['coverage'],'hypothesis':s['hypothesis'],
        'qualification':'NOT_EVALUATED','held_out_access':False,'new_wrapper_queries':0}
    if 'parent' in freeze:value.update(parent=freeze['parent'],parent_implementation_commit=freeze['parent_implementation_commit'],
                                      primary_execution='FAIL',aggregation_recovery_execution='PASS',CXX_rerun=False)
    write_new(out/'result_integrity.json',value)
    write_new(ROOT/'docs/wb100_cell_direction_envelope_result_manifest.json',value|{'result_integrity_sha256':digest(out/'result_integrity.json')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('report','seal'));parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();out=args.output_root.resolve()
    if out not in (ROOT/'outputs/mc24_four_station_wb100_cell_direction_envelope_v1',ROOT/'outputs/mc24_four_station_wb100_aggregation_recovery_v1'):raise ValueError('wrong WB100 output')
    globals()[args.action](out)
