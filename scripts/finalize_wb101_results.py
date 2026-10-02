#!/usr/bin/env python3
"""Report and seal completed WB101 artifacts; never propagate or retry."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new
from wb101_contract import verify
from wb100_contract import digest

WORKBOOK=ROOT/'workbook/2026-10-02_101_四站方向包络接受条件与完整微扰终态因果诊断.md'

def scalar_audit(out,s):
    # Scalar Python arithmetic is independent of the vectorized metric code.
    raw=read_public(out/'event/acts.json');f=read_public(out/'fixture.json');reference=read_public(Path(f['wb101_field_source']));p=read_public(out/'protocol.json')
    scales=p['output_scales'];refs={x['name']:x['ladders'] for x in reference['modes']}
    def flat(row):
        if len(row)!=4:raise ValueError('scalar endpoint shape')
        return [float(x[0]) if isinstance(x,list) and len(x)==1 else float(x) for x in row]
    def diff(a,b):return [x-y for x,y in zip(a,b)]
    def mag(a):return max(abs(x/z) for x,z in zip(a,scales))
    def effects(a):
        first=[[v/4 for v in diff(a[1+4*k],a[2+4*k])] for k in range(5)]
        return first+[[sum((-1)**k*first[k][j] for k in range(5)) for j in range(4)]]
    def taylor(a):
        e=effects(a)
        return [[x-y-z*f for x,y,z in zip(a[index],a[0],e[k])] for k in range(6) for index,f in ((3+4*k,1),(4+4*k,.5))]
    def quadratic(a):return [[x-2*y+z for x,y,z in zip(a[3+4*k],a[4+4*k],a[0])] for k in range(6)]
    transforms={'effect':effects,'taylor_excess':taylor,'full_minus_two_half_excess':quadratic}
    recomputed={};maxdiff=0.
    for cell in s['cells']:
        tau,station,mode=(cell[k] for k in ('threshold','station','reference_mode'))
        arrays=[[flat(r['state']['h']) for r in setting['targets'][station-1]['samples']] for setting in raw['settings'] if setting['direction_threshold']==tau]
        b=[flat(r['targets'][station-1]['h']) for r in refs[mode][-1]['samples']];prev=[flat(r['targets'][station-1]['h']) for r in refs[mode][-2]['samples']]
        result={'endpoint':max(mag(diff(a,r)) for cap in arrays for a,r in zip(cap,b)),
                'cap_spread':max(mag(diff(a,r)) for c in arrays for d in arrays for a,r in zip(c,d))}
        U={'endpoint':max(mag(diff(a,r)) for a,r in zip(b,prev))};U['cap_spread']=U['endpoint']
        for name,transform in transforms.items():
            result[name]=max(mag(diff(a,r)) for cap in arrays for a,r in zip(transform(cap),transform(b)))
            U[name]=max(mag(diff(a,r)) for a,r in zip(transform(b),transform(prev)))
        for name in result:
            discrepancy=max(abs(cell[name]-result[name]),abs(cell['uncertainty'][name]-U[name]));maxdiff=max(maxdiff,discrepancy)
            if discrepancy>1e-12*max(1,result[name]):raise ValueError('scalar metric '+name)
        recomputed[tau,station,mode]=result,U
    for check in s['decision_checks']:
        tau,station,mode,name=(check[k] for k in ('threshold','station','reference_mode','metric'))
        a,U=recomputed[tau,station,mode];base=recomputed[0,station,mode][0][name];candidate=a[name];allowance=p['uncertainty_factor']*U[name]
        floor=base<=allowance;passed=candidate<=allowance if floor else candidate<=p['reduction_required']*base and base-candidate>allowance
        if passed!=check['pass'] or floor!=check['floor_limited']:raise ValueError('scalar decision')
    return {'gate':'PASS','cells':len(recomputed),'decisions':len(s['decision_checks']),'max_difference':maxdiff,'qualification':'NOT_EVALUATED'}

def report(out):
    verify(out);s=read_public(out/'summary.json');receipt=read_public(out/'worker_exit.json')
    if receipt['exit_code']!=0:raise ValueError('worker infrastructure failure')
    sub=read_public(out/'submission.json');cluster=re.search(r'cluster (\d+)',sub['stdout']).group(1)
    cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null && condor_history '+cluster+' -limit 1 -json'
    r=subprocess.run(['bash','-c',cmd],capture_output=True,text=True,check=True);history=json.loads(r.stdout)
    if len(history)!=1:raise ValueError('single scheduler history missing')
    ad=history[0]
    if ad.get('JobStatus')!=4 or ad.get('ExitCode')!=0 or ad.get('NumJobStarts')!=1 or ad.get('ExitBySignal',False):raise ValueError('scheduler terminal')
    write_new(out/'scheduler_history.json',history)
    write_new(out/'scheduler_terminal_audit.json',{'gate':'PASS','cluster':cluster,'starts':ad['NumJobStarts'],'exit_code':ad['ExitCode'],
      'RequestCpus':ad.get('RequestCpus'),'RequestMemory':ad.get('RequestMemory'),'ShouldTransferFiles':ad.get('ShouldTransferFiles'),
      'worker':ad.get('LastRemoteHost'),'RemoteWallClockTime':ad.get('RemoteWallClockTime'),'command':cmd,'stderr':r.stderr})
    parent=read_public(out/'freeze.json').get('parent')
    if parent:
        primary=Path(parent);sub=read_public(primary/'submission.json');oldcluster=re.search(r'cluster (\d+)',sub['stdout']).group(1)
        cmd='source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null && condor_history '+oldcluster+' -limit 1 -json'
        old=subprocess.run(['bash','-c',cmd],capture_output=True,text=True,check=True);history=json.loads(old.stdout)
        if len(history)!=1 or history[0].get('ExitCode')!=1 or history[0].get('NumJobStarts')!=1:raise ValueError('primary failure scheduler')
        write_new(out/'primary_scheduler_history.json',history)
        write_new(out/'primary_execution_audit.json',{'execution':'FAIL','cluster':oldcluster,'exit_code':1,'physical_matrix_calls':0,
          'failure':'WB95 column vector decoded as scalar before trial stream creation','original_artifacts_preserved':True,'threshold_or_decision_change':False})
    write_new(out/'scalar_summary_audit.json',scalar_audit(out,s))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4));colors=['black','#0072b2','#009e73','#d55e00']
    keys=['endpoint','cap_spread','effect','taylor_excess','full_minus_two_half_excess']
    for station,ax in enumerate(axes,1):
        for tau,color in zip(read_public(out/'protocol.json')['arms'],colors):
            cells=[c for c in s['cells'] if c['station']==station and c['reference_mode']=='mesh_z_double' and c['threshold']==tau]
            if cells:ax.plot(range(5),[cells[0][k] for k in keys],'o-',label='disabled' if tau==0 else str(tau),color=color)
        ax.set_xticks(range(5),['E','C','D','T','Q']);ax.set_yscale('log');ax.set_title('Station '+str(station));ax.set_ylabel('Full four-cap scaled maximum');ax.legend()
    fig.tight_layout();directory=out/'figures';directory.mkdir(exist_ok=False);fig.savefig(directory/'endpoint_metrics.png',dpi=180);fig.savefig(directory/'endpoint_metrics.pdf');plt.close(fig)
    write_new(directory/'manifest.json',{x.name:digest(x) for x in directory.iterdir()})

def seal(out):
    freeze=verify(out);s=read_public(out/'summary.json')
    for name in ('scheduler_terminal_audit','scalar_summary_audit','trial_integrity_audit'):
        if read_public(out/(name+'.json'))['gate']!='PASS':raise ValueError('result gate '+name)
    paths=[x for x in out.rglob('*') if x.is_file()]
    for smoke in sorted((ROOT/'outputs').glob('mc24_four_station_wb101_build_smoke_v*')):paths.extend(x for x in smoke.rglob('*') if x.is_file())
    if freeze.get('parent'):
        for folder in (Path(freeze['parent']),ROOT/'outputs/mc24_four_station_wb101_recovery_preparation_v1'):paths.extend(x for x in folder.rglob('*') if x.is_file())
    paths.append(WORKBOOK)
    value={'schema':'wb101_result_integrity_v1','implementation_commit':freeze['commit'],'frozen_identities_verified':len(freeze['hashes']),
      'artifacts':{str(x.relative_to(ROOT)):digest(x) for x in paths},'workbook_sha256':digest(WORKBOOK),
      'execution_contract':s['execution_contract'],'hypothesis':s['hypothesis'],'qualification':'NOT_EVALUATED',
      'held_out_access':False,'population':1,'production_backend_changed':False}
    if freeze.get('parent'):value.update(parent=freeze['parent'],primary_execution='FAIL',recovery_execution='PASS',parent_physical_matrix_calls=0)
    write_new(out/'result_integrity.json',value)
    write_new(ROOT/'docs/wb101_direction_acceptance_result_manifest.json',value|{'result_integrity_sha256':digest(out/'result_integrity.json')})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('report','seal'));parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args();out=args.output_root.resolve()
    if out not in (ROOT/'outputs/mc24_four_station_wb101_direction_acceptance_v1',ROOT/'outputs/mc24_four_station_wb101_direction_acceptance_v2'):raise ValueError('exclusive WB101 output')
    globals()[args.action](out)
