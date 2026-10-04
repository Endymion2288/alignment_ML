#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mc24_four_station_wb130_sensor_bounds_v2'
REC=OUT/'recovery_v2'
AUDIT=REC/'independent_audit.json'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 events=[]
 for index in (12,20):
  name=f'{index:02d}'; d=json.loads((REC/'events'/name/'bounds_audit.json').read_text()); rows=d['rows']
  if not rows or any(not r['inside_bounds'] or not r['is_on_surface_with_bounds'] for r in rows): raise ValueError(f'outside surface {index}')
  if any(abs(r['frame_roundtrip_mm'])>1e-9 for r in rows): raise ValueError(f'frame closure {index}')
  by_station=[]
  for station in sorted({r['station'] for r in rows}):
   rr=[r for r in rows if r['station']==station]; y=np.asarray([r['endpoint_local_y_mm'] for r in rr]); x=np.asarray([r['curve_residual_mm'] for r in rr]); bounds=[r['bounds_values'] for r in rr]
   if len({tuple(v) for v in bounds})!=1: raise ValueError(f'bounds inconsistency {index}/{station}')
   by_station.append({'station':station,'rows':len(rr),'endpoint_local_y_min_mm':float(y.min()),'endpoint_local_y_max_mm':float(y.max()),'endpoint_local_y_rms_mm':float(np.sqrt(np.mean(y*y))),'curve_residual_rms_mm':float(np.sqrt(np.mean(x*x))),'bounds_values':bounds[0],'min_y_edge_margin_mm':float(min(63.045-abs(y)))})
  events.append({'index':index,'rows':len(rows),'surface_type_counts':sorted({r['surface_type'] for r in rows}),'bounds_type_counts':sorted({r['bounds_type'] for r in rows}),'inside_bounds_rows':sum(r['inside_bounds'] for r in rows),'is_on_surface_rows':sum(r['is_on_surface_with_bounds'] for r in rows),'max_frame_roundtrip_mm':max(r['frame_roundtrip_mm'] for r in rows),'stations':by_station})
 result={'schema':'wb130_independent_audit_v1','population':2,'rows':sum(e['rows'] for e in events),'new_propagation_calls':0,'new_reconstruction_calls':0,'runtime_surface_bounds':'PASS','all_endpoints_inside_bounds':True,'all_endpoints_on_surface_with_bounds':True,'measurement_frame_roundtrip':'PASS','event20_edge_proximity':'DIAGNOSTIC_ONLY','association':'INHERITED_CONDITIONAL','alignment':'NOT_ESTABLISHED','events':events,'input_digests':{str(p.relative_to(ROOT)):digest(p) for p in sorted(REC.glob('events/*/bounds_audit.json'))}}
 with AUDIT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print('AUDIT_COMPLETE',AUDIT)
if __name__=='__main__':main()
