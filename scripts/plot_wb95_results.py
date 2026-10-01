#!/usr/bin/env python3
"""Reporting only: all saved precision ladders, no rerun or decision selection."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from alignment.wb95_field_precision import point,POINTS
from alignment.wb93_transport_error import v,norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb95_'):p.error('WB95 output required')
s=read_public(out/'summary.json');r=read_public(out/'event/acts.json');protocol=read_public(out/'protocol.json');scale=v(protocol['output_scales'])
directory=out/'figures';directory.mkdir(exist_ok=False)
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');tiny=np.finfo(float).tiny
for mode in r['modes']:
    ladders=mode['ladders'];xs=[];ys=[];es=[]
    for previous,current in zip(ladders,ladders[1:]):
        xs.append(current['dz_mm']);ys.append(max(norm((v(point(b,k)['h'])-v(point(a,k)['h']))/scale)
          for a,b in zip(previous['samples'],current['samples']) for k in POINTS))
        effects=[]
        for station in range(3):
            per_ladder=[]
            for ladder in (previous,current):
                rows=ladder['samples'];q=[.25*(v(rows[1+4*i]['targets'][station]['h'])-v(rows[2+4*i]['targets'][station]['h'])) for i in range(5)]
                q.append(sum(((-1)**i)*x for i,x in enumerate(q)));per_ladder.append(q)
            effects.extend(norm((b-a)/scale) for a,b in zip(*per_ladder))
        es.append(max(effects))
    axes[0,0].loglog(xs,[max(y,tiny) for y in ys],marker='o',label=mode['name'])
    axes[0,1].loglog(xs,[max(y,tiny) for y in es],marker='o',label=mode['name'])
for ax,title in zip(axes[0],('All 25 samples / 5 checkpoints: adjacent reference difference','All 18 coordinate/mixed effects: adjacent difference')):
    ax.axhline(protocol['reference_scaled_tolerance'],color='grey',ls=':',label='frozen diagnostic budget')
    ax.set(title=title,xlabel='finer RK4 dz (mm)',ylabel='max scaled difference');ax.grid(alpha=.25);ax.legend(fontsize=8)
for mode,values in s['modes'].items():
    for station in (1,2,3):
        rows=[x for x in values['effects']['taylor'] if x['station']==station and x['direction']=='alternating_mixed']
        axes[1,0].loglog([x['dz_mm'] for x in rows],[max(x['full_error'],x['half_error'],tiny) for x in rows],marker='o',label=f'{mode}, station {station}')
axes[1,0].set(title='Mixed Taylor errors: numerical and true curvature combined',xlabel='RK4 dz (mm)',ylabel='max scaled full/half error')
axes[1,0].grid(alpha=.25);axes[1,0].legend(fontsize=6)
probes=r['probes'][:3*len(r['mesh_mm'][2])];z=[v(q['position_mm'],3)[2] for q in probes]
axes[1,1].semilogy(z,[max(norm(v(q['float_T'],3)-v(q['double_T'],3)),tiny) for q in probes],'.',ms=2)
axes[1,1].set(title='Same-node float/double field: probes at z-mesh planes ±1e-6 mm',xlabel='z (mm), fixed original seed x/y',ylabel='max field component difference (T)')
axes[1,1].grid(alpha=.25)
fig.suptitle('WB95: one seen pilot; '+s['mechanism_hypothesis']+'; double evaluator is a diagnostic')
for ext in ('png','pdf'):fig.savefig(directory/('field_precision_contract.'+ext),dpi=180)
plt.close(fig)
write_new(directory/'manifest.json',{'summary_sha256':digest(out/'summary.json'),'acts_sha256':digest(out/'event/acts.json'),
  'plot_script_sha256':digest(Path(__file__)),'hashes':{f.name:digest(f) for f in directory.iterdir() if f.is_file()}})
