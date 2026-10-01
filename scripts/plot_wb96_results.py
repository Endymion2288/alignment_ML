#!/usr/bin/env python3
"""Reporting only: complete saved option matrix, no new propagation or selection."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb96_'):p.error('WB96 output required')
s=read_public(out/'summary.json');protocol=read_public(out/'protocol.json');directory=out/'figures';directory.mkdir(exist_ok=False)
fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
colors=['tab:blue','tab:orange','tab:green'];tiny=np.finfo(float).tiny
for ax,key,title in zip(axes.flat,('endpoint','cap_spread','effect','taylor_excess'),
  ('All 25 samples and 4 caps: endpoint-reference difference','All 25 samples: spread across 4 caps',
   'All 6 coordinate/mixed effects: difference from reference','Taylor remainder minus reference remainder')):
    for station,color in zip((1,2,3),colors):
        for mode,style in (('mesh_z_double','-'),('mesh_z_float','--')):
            if key=='cap_spread' and mode=='mesh_z_float':continue
            rows=[c for c in s['cells'] if c['station']==station and c['reference_mode']==mode]
            ax.loglog([c['tolerance'] for c in rows],[max(c[key],tiny) for c in rows],marker='o',color=color,ls=style,
              label=f'station {station}, {mode}')
        if key!='taylor_excess':
            base=next(c for c in s['cells'] if c['station']==station and c['reference_mode']=='mesh_z_double' and c['tolerance']==protocol['step_tolerances'][0])
            ax.axhline(base[key]*protocol['effect_reduction_required'],color=color,ls=':',alpha=.45)
    ax.set(title=title,xlabel='ACTS stepTolerance (smaller to right)',ylabel='max scaled difference');ax.invert_xaxis();ax.grid(alpha=.25);ax.legend(fontsize=6)
fig.suptitle('WB96: one seen pilot; '+s['mechanism_hypothesis']+'\nDotted lines: 50% of default double-reference metric; no qualification')
for ext in ('png','pdf'):fig.savefig(directory/('acts_tolerance_contract.'+ext),dpi=180)
plt.close(fig)
write_new(directory/'manifest.json',{'summary_sha256':digest(out/'summary.json'),'acts_sha256':digest(out/'event/acts.json'),
  'plot_script_sha256':digest(Path(__file__)),'hashes':{f.name:digest(f) for f in directory.iterdir() if f.is_file()}})
