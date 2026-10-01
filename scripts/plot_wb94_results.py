#!/usr/bin/env python3
"""Reporting only: saved endpoint errors and actual boundary field probes."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb94_'):p.error('WB94 output required')
s=read_public(out/'summary.json');r=read_public(out/'event/acts.json');protocol=read_public(out/'protocol.json')
directory=out/'figures';directory.mkdir(exist_ok=False)
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
caps=protocol['max_step_sizes_m'];tiny=np.finfo(float).tiny
for ax,m in zip(axes.flat,s['metrics']):
    for path,values in m['per_cap_reference_errors'].items():ax.loglog(caps,[max(x,tiny) for x in values],marker='o',label=path)
    ax.axhline(protocol['reference_scaled_tolerance'],color='grey',ls=':',label='reference diagnostic budget')
    ax.set(title=f"Station {m['station']}: nominal endpoint error",xlabel='official maximum step cap (m)',ylabel='max scaled difference from finest RK4')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
ax=axes.flat[3]
for k,name in enumerate(('Bx','By','Bz')):
    ax.plot(protocol['boundary_probe_offsets_mm'],[np.asarray(q['field_T']).reshape(3)[k] for q in r['probes']],marker='o',label=name)
ax.set_xscale('symlog',linthresh=1e-6);ax.axvline(0,color='grey',ls=':')
ax.set(title='Actual field-cache probes at fixed pilot x/y',xlabel='z - actual map entrance (mm)',ylabel='field (T)')
ax.grid(alpha=.25);ax.legend()
fig.suptitle('WB94: one seen pilot; '+s['mechanism_hypothesis']+'; no physical qualification')
for ext in ('png','pdf'):fig.savefig(directory/('field_boundary_contract.'+ext),dpi=180)
plt.close(fig)
write_new(directory/'manifest.json',{'summary_sha256':digest(out/'summary.json'),'acts_sha256':digest(out/'event/acts.json'),
  'plot_script_sha256':digest(Path(__file__)),'hashes':{f.name:digest(f) for f in directory.iterdir() if f.is_file()}})
