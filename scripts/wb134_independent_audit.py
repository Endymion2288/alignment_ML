#!/usr/bin/env python3
"""Saved-only independent material-class, graph and analytic-control audit."""
import math,sys
sys.dont_write_bytecode=True
from wb134_contract import ROOT,OUT,PARENT_STAGE,PROTOCOL,read,write,require,verify


def main():
 verify();events=[];total=0;summary=read(OUT/'summary.json')
 for index in read(PROTOCOL)['indices']:
  event=OUT/'events'/f'{index:02d}';st=read(event/'status.json');ex=read(event/'export.json');req=read(event/'request.json')
  require(req==read(PARENT_STAGE/'events'/event.name/'request.json'),'unchanged accepted source')
  if not (event/'audit.json').exists():
   require(st['interface']=='UNKNOWN' and 'failure' in st,'unrecorded failure');events.append({'index':index,'interface':'UNKNOWN','failure':st['failure']});continue
  r=read(event/'response.json');a=read(event/'audit.json');vs=r['volumes'];ss=r['surfaces']
  require(r['source']==req['source'] and r['identity']==ex['identity'],'identity')
  require([v['index'] for v in vs]==list(range(len(vs))) and [s['index'] for s in ss]==list(range(len(ss))),'deduplicated indices')
  reached=set();pending=[r['root_volume_index']]
  while pending:
   i=pending.pop()
   if i in reached:continue
   reached.add(i);pending.extend(vs[i]['children'])
  require(reached==set(range(len(vs))),'volume graph coverage')
  sc={};vc={};arrays=set();memberships=set()
  for v in vs:
   vc[v['material_kind']]=vc.get(v['material_kind'],0)+1
   for i in v['surface_indices']:require(0<=i<len(ss),'volume surface index')
  for s in ss:
   sc[s['material_kind']]=sc.get(s['material_kind'],0)+1
   for m in s['memberships']:
    require(s['index'] in vs[m['volume_index']]['surface_indices'],'membership identity');memberships.add(s['index'])
    if m['role']=='SENSITIVE_ARRAY':arrays.add(s['index'])
   if s['material_kind']=='PROTO_VACUUM':
    slab=s['center_sample'];require(slab=={'valid':False,'thickness_mm':0.,'thickness_in_X0':0.,'thickness_in_L0':0.},'Proto dummy slab')
   if s['material_kind']=='NONE':require(s['center_sample'] is None,'null material')
  require(len(memberships)==len(ss) and len(arrays)==r['sensitive_visit_count']==a['sensitive_surfaces'],'surface recursion')
  require(sc==a['surface_counts']==st['surface_counts'] and vc==a['volume_counts']==st['volume_counts'],'material counts')
  require(len(r['targets'])==len(ex['rows'])==a['targets'],'147 target coverage');total+=len(ex['rows'])
  for orig,t in zip(ex['rows'],r['targets']):
   require(all(orig[k]==t[k] for k in ('cluster_id','wafer_id','station')),'original target ID/order')
   s=ss[t['surface_index']];require(s['geometry_id']==t['geometry_id'],'geometry ID')
   error=max(abs(s['transform'][i][k]-orig['sensor_transform'][i][k]) for i in range(4) for k in range(4))
   require(error<=1e-9 and t['frame_max_error']<=1e-9,'original frame')
  c=r['controls'];momentum=c['momentum_GeV']*1000.;mass=c['mass_MeV'];beta=momentum/math.sqrt(momentum**2+mass**2)
  fraction=c['thin_slab']['thickness_mm']/93.7
  theta=13.6/math.sqrt(momentum**2/(1+mass**2/momentum**2))*math.sqrt(fraction)*(1+.038*math.log(fraction/beta**2))
  require(abs(c['theta0_rad']-theta)<=max(1e-10,abs(theta)*1e-4),'independent Highland')
  require(c['synthetic_only'] and not c['vacuum']['valid'] and c['vacuum']['thickness_in_X0']==0,'synthetic/vacuum')
  # Explicit scalar covariance for two independent transverse kicks, d=(0,0,1).
  q=[[0.]*6 for _ in range(6)];variance=c['theta0_rad']**2;length=c['lever_mm']
  for axis in (0,1):
   q[axis][axis]=length*length*variance;q[axis+3][axis+3]=variance
   q[axis][axis+3]=q[axis+3][axis]=length*variance
  require(max(abs(q[i][k]-a['toy_q'][i][k]) for i in range(6) for k in range(6))<1e-15,'scalar toy Q')
  for k in ('new_propagation_calls','new_reconstruction_calls','algorithm_field_queries'):require(r[k]==0,'unexpected work')
  require(r['path_integral'] is None and r['process_noise_covariance'] is None and not r['map_loaded'],'unqualified physics not computed')
  require(a['physical_budget']==summary['physical_budget']=='UNKNOWN_MATERIAL_INPUT_AND_PATH_NOT_QUALIFIED','UNKNOWN preserved')
  events.append({'index':index,'interface':'PASS','targets':len(ex['rows']),'surfaces':len(ss),'volumes':len(vs),
   'surface_counts':sc,'volume_counts':vc,'synthetic_theta0_rad':c['theta0_rad'],'physical_budget':a['physical_budget']})
 require(read(ROOT/'docs/wb133_accepted_anchor_result_manifest.json')['summary']['scientific_verdict']=='FAIL_GROSS_MODEL','WB133 FAIL preserved')
 if all(e['interface']=='PASS' for e in events):require(total==147,'population')
 verify();write(OUT/'independent_audit.json',{'schema':'wb134_independent_saved_audit_v1','artifact_consistency':'PASS','events':events,
  'inventory_targets':total,'physical_budget':'UNKNOWN_MATERIAL_INPUT_AND_PATH_NOT_QUALIFIED','WB133_FAIL_preserved':True,
  'new_input_event_executions':0,'new_propagation_calls':0,'new_reconstruction_calls':0,'algorithm_field_queries':0,'held_out_access':False,'truth_access':False})
 print('WB134 independent artifact audit PASS; physical budget UNKNOWN')

if __name__=='__main__':main()
