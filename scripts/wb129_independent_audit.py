#!/usr/bin/env python3
"""Independent audit of WB129 actual sensor-surface endpoint responses."""
import hashlib, json, math, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/mc24_four_station_wb129_sensor_surface_curve_v2'
WB127=ROOT/'outputs/mc24_four_station_wb127_strip_measurement_v1/recovery_v5'
REC=OUT/'recovery_v2'
AUDIT=REC/'independent_audit.json'
INDICES=(1,4,8,12,16,20)

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def flat(x):return np.asarray(x,dtype=float).reshape(-1)

def main():
 events=[]; total=0
 for index in INDICES:
  name=f'{index:02d}'; export=json.loads((WB127/'events'/name/'export.json').read_text()); fixture=json.loads((WB127/'events'/name/'fixture.json').read_text())
  response=json.loads((REC/'events'/name/'curve_response.json').read_text()); log=(REC/'events'/name/'athena.log').read_text(errors='replace')
  rows=export['rows']; outrows=response['rows']; expected=[r['cluster_id'] for r in rows]; actual=[r['cluster_id'] for r in outrows]
  if actual!=expected: raise ValueError(f'cluster/order mismatch {index}')
  if response['identity']!=export['identity'] or fixture['actual_run']!=export['identity']['actual_run'] or fixture['actual_event']!=export['identity']['actual_event']: raise ValueError(f'identity {index}')
  if response['official_calls']!=len(rows) or len(outrows)!=len(rows): raise ValueError(f'call budget {index}')
  begins=[int(x) for x in re.findall(r'WB129_CALL_BEGIN id=(\d+)',log)]; ends=[int(x) for x in re.findall(r'WB129_CALL_END id=(\d+)',log)]
  if begins!=ends or begins!=list(range(1,len(rows)+1)): raise ValueError(f'call trace {index}')
  byid={r['cluster_id']:r for r in rows}; straight={};
  refs={r['station']:r for r in fixture['references']}; seed=flat(export['p_seed']); z0=float(refs[0]['z_center_mm'])
  for row in rows:
   T=np.asarray(row['sensor_transform'],dtype=float); gp=np.asarray(row['global_position'],dtype=float)
   prediction=np.asarray([seed[0]+seed[2]*(gp[2]-z0),seed[1]+seed[3]*(gp[2]-z0),gp[2],1.])
   straight[row['cluster_id']]=float(row['local_position'][0]-(np.linalg.inv(T)@prediction)[0])
  statuses={r['status'] for r in outrows};
  if statuses!={'SUCCESS'}: raise ValueError(f'endpoint status {index}: {statuses}')
  curve={}; z_errors=[]; finite=True
  for row in outrows:
   state=row['state']; local=flat(state['local_position_mm']); direction=flat(state['global_direction'])
   if local.size!=3 or direction.size!=3 or not np.all(np.isfinite(local)) or not np.all(np.isfinite(direction)): finite=False
   if abs(local[2])>1e-5 or abs(float(row['local_residual_mm'])-(byid[row['cluster_id']]['local_position'][0]-local[0]))>1e-12: raise ValueError(f'endpoint/residual {index}')
   z_errors.append(abs(local[2]));curve[row['cluster_id']]=float(row['local_residual_mm'])
  if not finite: raise ValueError(f'nonfinite endpoint {index}')
  station=[]
  for st in sorted({r['station'] for r in rows}):
   ids=[r['cluster_id'] for r in rows if r['station']==st]; s=np.asarray([straight[i] for i in ids]); c=np.asarray([curve[i] for i in ids])
   station.append({'station':st,'rows':len(ids),'straight_rms_mm':float(np.sqrt(np.mean(s*s))),'curve_rms_mm':float(np.sqrt(np.mean(c*c))),'curve_max_abs_mm':float(np.max(np.abs(c))),'rms_ratio_curve_over_straight':float(np.sqrt(np.mean(c*c))/np.sqrt(np.mean(s*s)))})
  total+=len(rows)
  events.append({'index':index,'actual_run':fixture['actual_run'],'actual_event':fixture['actual_event'],'rows':len(rows),'official_calls':response['official_calls'],'status_counts':{'SUCCESS':len(outrows)},'max_abs_local_surface_z_mm':max(z_errors),'station_metrics':station})
 result={'schema':'wb129_independent_audit_v1','population':len(events),'rows':total,'official_calls':total,'endpoint_execution':'PASS','exact_cluster_order':'PASS','event_identity':'PASS','surface_status':'PASS','finite_endpoints':'PASS','curve_vs_straight':'DIAGNOSTIC_ONLY','physical_common_track':'INHERITED_CONDITIONAL','alignment':'NOT_ESTABLISHED','events':events,'input_digests':{str(p.relative_to(ROOT)):digest(p) for p in sorted(REC.glob('events/*/curve_response.json'))}}
 with AUDIT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print('AUDIT_COMPLETE',AUDIT)
if __name__=='__main__':main()
