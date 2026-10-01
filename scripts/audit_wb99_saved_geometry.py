#!/usr/bin/env python3
"""Reporting-only independent batched stage-position audit of immutable exports."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT,read_public,write_new,digest
from wb99_contract import verify
p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);a=p.parse_args();out=a.output_root.resolve()
if not out.is_relative_to(ROOT/'outputs') or not out.name.startswith('mc24_four_station_wb99_'):p.error('WB99 output')
verify(out);protocol=read_public(out/'protocol.json');raw=out/'event/acts.json.doubling.ndjson';batch=[];count=0;worst=0.;branches={}
def audit(rows):
    global count,worst
    pp=np.array([r['start_position_mm'] for r in rows]);uu=np.array([r['start_direction'] for r in rows]);q=np.array([r['q_over_p_Acts'] for r in rows])[:,None];h=np.array([r['h_mm'] for r in rows])[:,None]
    for name in ('full','half1','half2','accepted_branch'):
        position=pp if name!='half2' else np.array([r['half1']['position_mm'] for r in rows])
        direction=uu if name!='half2' else np.array([r['half1']['direction'] for r in rows])
        hh=h/2 if name.startswith('half') else h
        queries=np.array([[x['position_mm'] for x in r[name]['queries']] for r in rows]);b=np.array([[x['field_native'] for x in r[name]['queries']] for r in rows])
        k1=q*np.cross(direction,b[:,0]);k2=q*np.cross(direction+hh*k1/2,b[:,1]);k3=q*np.cross(direction+hh*k2/2,b[:,1])
        expected=np.stack([position,position+hh*direction/2+hh*hh*k1/8,position+hh*direction+hh*hh*k3/2],axis=1)
        error=float(np.max(np.abs(queries-expected)));branches[name]=max(branches.get(name,0.),error);worst=max(worst,error)
    count+=len(rows)
with raw.open() as f:
    for line in f:
        row=json.loads(line)
        if row['record']!='step':continue
        batch.append(row)
        if len(batch)==1024:audit(batch);batch=[]
if batch:audit(batch)
gate='PASS' if count==protocol['expected_steps'] and worst<=protocol['closure_position_mm'] else 'FAIL'
write_new(out/'saved_stage_geometry_audit.json',{'gate':gate,'steps':count,'branches':branches,'max_position_difference_mm':worst,
  'budget_mm':protocol['closure_position_mm'],'raw_sha256':digest(raw),'reporting_script':str(Path(__file__).relative_to(ROOT)),
  'reporting_script_sha256':digest(Path(__file__)),'field_queries_executed':0,'threshold_retuned':False})
print({'gate':gate,'steps':count,'max_position_difference_mm':worst})
if gate!='PASS':raise SystemExit(2)
