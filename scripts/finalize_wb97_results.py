#!/usr/bin/env python3
"""Reporting only: immutable result index and aggregate figures; no integrations."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from wb97_contract import verify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb97_'):p.error('WB97 output required')
freeze=verify(out);s=read_public(out/'summary.json')
if digest(out/'steps.ndjson')!=s['steps_sha256'] or digest(out/'freeze.json')!=s['freeze_sha256']:raise ValueError('raw/freeze identity changed')
receipt=read_public(out/'worker_exit.json');scheduler=read_public(out/'scheduler_terminal_audit.json')
if receipt['exit_code']!=0 or scheduler['exit_code']!=0 or not scheduler['normal_termination']:raise ValueError('execution incomplete')
binary=read_public(out/'binary_manifest.json')
if digest(out/'local_defect')!=binary['sha256']:raise ValueError('binary changed')
figures=out/'figures';figures.mkdir(exist_ok=False)
categories=['outside','interior','mesh','domain','unclassified']
counts=[s['classification'].get(k,{}).get('count',0) for k in categories]
significant=[s['classification'].get(k,{}).get('significant',0) for k in categories]
unknown=[s['classification'].get(k,{}).get('reference_unknown',0) for k in categories]
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(categories,counts,color='#4675a8');axes[0].set_yscale('symlog',linthresh=1);axes[0].set_ylabel('All accepted steps');axes[0].set_title('Observed domain/cell classification')
axes[1].bar(categories,significant,label='Verified significant underestimate',color='#cb6b35')
axes[1].bar(categories,unknown,bottom=significant,label='Reference UNKNOWN',color='#777777')
axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylabel('Steps');axes[1].legend(fontsize=8)
fig.suptitle(f"WB97: {s['traces']} traces / {s['steps']} steps; H={s['hypothesis']}")
for ax in axes:ax.tick_params(axis='x',rotation=20)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(figures/f'local_defect_contract.{ext}',dpi=180)
plt.close(fig)
write_new(figures/'manifest.json',{'reporting_script_sha256':digest(Path(__file__)),
    'summary_sha256':digest(out/'summary.json'),'hashes':{p.name:digest(p) for p in figures.iterdir() if p.is_file()}})
paths=[p for p in out.iterdir() if p.is_file() and p.name not in ('result_integrity.json',)]
paths+=list(figures.iterdir())
test=ROOT/'outputs/mc24_four_station_wb97_control_test_v1'
paths+=[test/'local_defect',test/'controls.json']
index={'schema':'wb97_result_integrity_v1','implementation_commit':freeze['commit'],'output_root':str(out),
       'hypothesis':s['hypothesis'],'execution_contract':s['execution_contract'],'reference_contract':s['reference_contract'],
       'qualification':'NOT_EVALUATED','held_out_access':False,'frozen_identities_verified':len(freeze['hashes']),
       'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in paths},'reporting_script_sha256':digest(Path(__file__))}
write_new(out/'result_integrity.json',index)
write_new(ROOT/'docs/wb97_rkn_local_defect_result_manifest.json',{**index,'result_integrity_sha256':digest(out/'result_integrity.json')})
print({k:index[k] for k in ('hypothesis','execution_contract','reference_contract','frozen_identities_verified')})
