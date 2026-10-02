#!/usr/bin/env python3
"""Record scheduler and seal completed WB102; no physical execution."""
import argparse, json, re, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new, digest
from wb102_contract import OUT, verify, P

WORKBOOK = ROOT/'workbook/2026-10-02_102_四站五维Jacobian扰动尺度平台诊断.md'


def report(out):
    verify(out)
    receipt = read_public(out/'worker_exit.json')
    if receipt['exit_code'] != 0: raise ValueError('worker execution failure')
    cluster = re.search(r'cluster (\d+)',read_public(out/'submission.json')['stdout']).group(1)
    cmd = 'source /usr/share/Modules/init/bash && module load lxbatch/eossubmit && myschedd out >/dev/null && condor_history '+cluster+' -limit 1 -json'
    run = subprocess.run(['bash','-c',cmd],capture_output=True,text=True,check=True,timeout=55)
    history = json.loads(run.stdout)
    if len(history) != 1: raise ValueError('scheduler history missing')
    ad = history[0]
    if ad.get('JobStatus') != 4 or ad.get('ExitCode') != 0 or ad.get('NumJobStarts') != 1 or ad.get('ExitBySignal',False):
        raise ValueError('scheduler terminal failure')
    write_new(out/'scheduler_history.json',history)
    write_new(out/'scheduler_terminal_audit.json',{'gate':'PASS','cluster':cluster,'starts':ad['NumJobStarts'],
        'exit_code':ad['ExitCode'],'RequestCpus':ad.get('RequestCpus'),'RequestMemory':ad.get('RequestMemory'),
        'ShouldTransferFiles':ad.get('ShouldTransferFiles'),'worker':ad.get('LastRemoteHost'),
        'RemoteWallClockTime':ad.get('RemoteWallClockTime'),'command':cmd,'stderr':run.stderr})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary = read_public(out/'summary.json')
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for station,ax in enumerate(axes,1):
        for tau in P['arms']:
            values=[]
            for lam in P['lambdas']:
                cells=[c for c in summary['cells'] if c['lambda']==lam and c['station']==station and c['threshold']==tau and c['mode']=='mesh_z_double']
                values.append(max(max(abs(a-b) for row,ref in zip(c['K'],c['reference_K']) for a,b in zip(row,ref)) for c in cells))
            ax.plot(P['lambdas'],values,'o-',label='disabled' if tau==0 else str(tau))
        ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_title('Station '+str(station)); ax.set_xlabel('Perturbation scale lambda'); ax.set_ylabel('All-cap maximum |K - Kref|'); ax.legend()
    fig.tight_layout(); directory=out/'figures'; directory.mkdir(exist_ok=False)
    fig.savefig(directory/'jacobian_scale.png',dpi=180); fig.savefig(directory/'jacobian_scale.pdf'); plt.close(fig)
    write_new(directory/'manifest.json',{x.name:digest(x) for x in directory.iterdir()})


def seal(out):
    frozen = verify(out); summary = read_public(out/'summary.json')
    if read_public(out/'worker_exit.json')['exit_code'] != 0: raise ValueError('worker failure')
    for name in ('trial_integrity_audit','scalar_summary_audit','scheduler_terminal_audit','lambda_1_preflight'):
        if read_public(out/(name+'.json'))['gate'] != 'PASS': raise ValueError('incomplete result gate '+name)
    paths = [x for x in out.rglob('*') if x.is_file()] + [WORKBOOK]
    for smoke in sorted((ROOT/'outputs').glob('mc24_four_station_wb102_build_smoke_v*')):
        paths.extend(x for x in smoke.rglob('*') if x.is_file())
    value = {'schema':'wb102_result_integrity_v1','implementation_commit':frozen['commit'],
        'frozen_identities_verified':len(frozen['hashes']),'artifacts':{str(x.relative_to(ROOT)):digest(x) for x in paths},
        'workbook_sha256':digest(WORKBOOK),'execution_contract':summary['execution_contract'],
        'hypothesis':summary['hypothesis'],'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED',
        'final_oracle':'NOT_EVALUATED','held_out_access':False,'population':1,'production_backend_changed':False}
    write_new(out/'result_integrity.json',value)
    write_new(ROOT/'docs/wb102_jacobian_scale_result_manifest.json',value|{'result_integrity_sha256':digest(out/'result_integrity.json')})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('report','seal')); parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args(); out=args.output_root.resolve()
    if out != OUT: raise ValueError('exclusive WB102 output')
    globals()[args.action](out)
