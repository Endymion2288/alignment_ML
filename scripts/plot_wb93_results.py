#!/usr/bin/env python3
"""Reporting-only plots of saved diagnostic errors; no scientific decisions."""
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
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb93_'):p.error('WB93 output required')
s=read_public(out/'summary.json');protocol=read_public(out/'protocol.json');directory=out/'figures';directory.mkdir(exist_ok=False)
fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
for row,station in enumerate((1,2,3)):
    for cap in protocol['max_step_sizes_m']:
        cell=next(c for c in s['cells'] if c['station']==station and c['cap_m']==cap and c['direction']=='alternating_mixed')
        axes[row,0].loglog([r['multiplier'] for r in cell['rows']],
            [max(r['full_error'],r['half_error'],np.finfo(float).tiny) for r in cell['rows']],marker='o',label=f'{cap:g} m cap')
    axes[row,0].axhline(protocol['roundoff_scaled_floor'],color='grey',ls=':',label='diagnostic roundoff floor')
    axes[row,0].set(title=f'Station {station}: mixed Taylor error',xlabel='perturbation multiplier',ylabel='max scaled full/half error')
    axes[row,0].grid(alpha=.25);axes[row,0].legend(fontsize=8)
    x=np.arange(6)
    for cap in protocol['max_step_sizes_m']:
        cells=[next(c for c in s['cells'] if c['station']==station and c['cap_m']==cap and c['direction']==d) for d in protocol['directions']]
        axes[row,1].semilogy(x,[max(c['small_step_envelope'],np.finfo(float).tiny) for c in cells],marker='o',label=f'{cap:g} m cap')
    axes[row,1].set_xticks(x,protocol['directions'],rotation=25,ha='right')
    axes[row,1].set(title=f'Station {station}: small-step envelope (multipliers <= 1)',ylabel='max scaled full/half error')
    axes[row,1].grid(alpha=.25);axes[row,1].legend(fontsize=8)
fig.suptitle('WB93: one seen pilot; deterministic transport diagnostics, no qualification',fontsize=13)
for extension in ('png','pdf'):fig.savefig(directory/('transport_error_budget.'+extension),dpi=180)
plt.close(fig)
write_new(directory/'manifest.json',{'summary_sha256':digest(out/'summary.json'),'plot_script_sha256':digest(Path(__file__)),
    'hashes':{p.name:digest(p) for p in directory.iterdir() if p.is_file()}})
