#!/usr/bin/env python3
"""Append-only completion index for descriptive reports added after the first index."""
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
freeze=verify(out);s=read_public(out/'summary.json');index=read_public(out/'result_integrity.json')
for path,h in index['artifacts'].items():
    if digest(ROOT/path)!=h:raise ValueError('first-index artifact changed '+path)
audit=read_public(out/'saved_step_source_audit.json')
if audit['accounting_gate']!='PASS' or audit['steps_sha256']!=s['steps_sha256']:raise ValueError('source audit failed')
read_public(out/'position_estimate_examples.json')
figures=out/'figures';categories=['outside','interior','mesh','domain']
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,key,label in zip(axes,['max_position_L1_mm','max_direction'],['Position L1 defect (mm)','Maximum unit-direction component defect']):
    values=[s['classification'][k][key] for k in categories]
    ax.bar(categories,values,color=['#888888','#4675a8','#cb6b35','#8555a1']);ax.set_yscale('log');ax.set_ylabel(label)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    for k,y in zip(categories,values):ax.text(k,y*1.3,f'{y:.2e}',ha='center',fontsize=8)
    ax.set_ylim(min(values)*.3,max(values)*20)
fig.suptitle('WB97: measurable local defects; frozen position-underestimate gate NOT_SUPPORTED')
fig.tight_layout()
new=[]
for ext in ('png','pdf'):
    path=figures/f'local_defect_magnitudes.{ext}'
    if path.exists():raise FileExistsError(path)
    fig.savefig(path,dpi=180);new.append(path)
plt.close(fig)
write_new(figures/'magnitude_manifest.json',{'reporting_script_sha256':digest(Path(__file__)),
    'summary_sha256':digest(out/'summary.json'),'hashes':{p.name:digest(p) for p in new}})
workbook=ROOT/'workbook/2026-10-01_97_四站接受RKN步局部缺陷与误差估计合同研究.md'
paths=[p for p in out.rglob('*') if p.is_file()]+[workbook,ROOT/'docs/wb97_rkn_local_defect_result_manifest.json',
    ROOT/'scripts/finalize_wb97_results.py',ROOT/'scripts/audit_wb97_saved_steps.py',Path(__file__)]
paths+=[ROOT/'outputs/mc24_four_station_wb97_control_test_v1'/p for p in ('local_defect','controls.json')]
result={'schema':'wb97_result_completion_integrity_v1','supplements':'result_integrity.json',
    'first_index_sha256':digest(out/'result_integrity.json'),'implementation_commit':freeze['commit'],
    'frozen_identities_verified':len(freeze['hashes']),'hypothesis':s['hypothesis'],'qualification':'NOT_EVALUATED',
    'held_out_access':False,'artifacts':{str(p.relative_to(ROOT)):digest(p) for p in paths},
    'workbook_sha256':digest(workbook),'reporting_script_sha256':digest(Path(__file__))}
write_new(out/'result_completion_integrity.json',result)
write_new(ROOT/'docs/wb97_rkn_local_defect_completion_manifest.json',{**result,
    'completion_integrity_sha256':digest(out/'result_completion_integrity.json')})
print({'indexed_artifacts':len(result['artifacts']),'hypothesis':s['hypothesis'],'frozen_identities_verified':len(freeze['hashes'])})
