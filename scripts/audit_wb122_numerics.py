#!/usr/bin/env python3
"""Independent scalar checks of saved WB122 records; no calls to its analyzer."""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public,write_new
from wb100_contract import digest,verify

def cross(x,y):return [x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]]
def v(row,key):
    a=row[key]
    if len(a)!=3 or any(len(z)!=1 or not math.isfinite(z[0]) for z in a):raise ValueError('scalar vector shape/finite')
    return [z[0] for z in a]
def require(ok,label):
    if not ok:raise ValueError(label)
def audit(out):
    frozen=verify(out);r=read_public(out/'event/response.json');old=read_public(out/'historical_runtime.json')
    maxdiff=0.;components=0;fidelity_differences=[];exact={'arms':0,'historical_h':0,'public_snapshots':0,'acceptance_decisions':0,'cache_flags':0,'target_parameters':0};counts=[]
    def arithmetic(x,y):
        nonlocal maxdiff,components
        require(math.isfinite(x) and math.isfinite(y),'finite independent scalar')
        components+=1;maxdiff=max(maxdiff,abs(x-y)/max(1.,abs(x),abs(y)))
    def vec(a,b):
        for x,y in zip(a,b):arithmetic(x,y)
    for sid in range(9):
      for st in range(1,3):
        start=6*sid+3*(st-1);a,b,c=r['rows'][start:start+3]
        require(all(z['has_value'] for z in (a,b,c)),'physical optional present')
        faithful=a['state']==b['state']==c['state'] and a['bound_parameters']==b['bound_parameters']==c['bound_parameters']
        exact['arms']+=int(faithful)
        if not faithful:fidelity_differences.append({'sample_id':sid,'station':st,'reason':'ARMS'})
        previous=old['rows'][4*sid+2*(st-1)];snaps=old['rows'][4*sid+2*(st-1)+1]['snapshots']
        faithful=a['state']['h']==previous['state']['h'];exact['historical_h']+=int(faithful)
        if not faithful:fidelity_differences.append({'sample_id':sid,'station':st,'reason':'HISTORICAL_H'})
        obs=c['observations'];actions=[z for z in obs if z['record']=='action'];steps=[z for z in obs if z['record']=='step_end'];trials=[z for z in obs if z['record']=='trial'];queries=[z for z in obs if z['record']=='field_query'];begins=[z for z in obs if z['record']=='step_begin']
        logged=[z for z in actions if not z['target_reached']]
        if len(logged)!=len(snaps):fidelity_differences.append({'sample_id':sid,'station':st,'reason':'LOGGER_COUNT'})
        for idx,(x,s) in enumerate(zip(logged,snaps)):
            faithful=x['position_mm']==s['global_position_mm'] and x['logger_direction']==s['global_direction'] and x['momentum_MeV']==s['momentum_MeV'] and x['geometry_id']==s['geometry_id'] and x['surface_present']==s['surface_present'] and x['surface_geometry_id']==s['surface_geometry_id'] and x['constraints']==s['step_constraints'] and x['rk_rejections']==s['rk_rejections_before_previous_accepted_step'] and (x['rk_rejections'] is None)==s['rk_counter_unset'] and x['navigation_direction']==s['navigation_direction'] and s['snapshot_index']==idx;exact['public_snapshots']+=int(faithful)
            if not faithful:fidelity_differences.append({'sample_id':sid,'station':st,'reason':'LOGGER_VALUE','snapshot_index':idx})
        for t in trials:
            p=v(t,'start_position_mm');u=v(t,'start_direction');q=t['qop_acts'];h=t['h_mm'];k=[v(t,'k'+str(i)) for i in range(1,5)];fields=[v(t,'B_'+z+'_native') for z in ('first','middle','last')]
            expected=[q*x for x in cross(u,fields[0])];vec(k[0],expected)
            for i in range(1,4):
                scale=h if i==3 else h*.5;field=fields[2] if i==3 else fields[1]
                expected=[q*x for x in cross([u[j]+scale*k[i-1][j] for j in range(3)],field)];vec(k[i],expected)
            vec(v(t,'pos1_mm'),[p[j]+h*.5*u[j]+h*h*.125*k[0][j] for j in range(3)])
            vec(v(t,'pos2_mm'),[p[j]+h*u[j]+h*h*.5*k[2][j] for j in range(3)])
            err=max(h*h*(sum(abs(k[0][j]-k[1][j]-k[2][j]+k[3][j]) for j in range(3))+abs(t['kQoP'][0]-t['kQoP'][1]-t['kQoP'][2]+t['kQoP'][3])),1e-20)
            arithmetic(t['error_estimate'],err);require(t['accepted']==(t['error_estimate']<=t['step_tolerance']),'original acceptance');exact['acceptance_decisions']+=1
        path=0.
        for i,(s,begin) in enumerate(zip(steps,begins)):
            ts=[t for t in trials if t['step_index']==i];require(len(ts)==s['trial_count'] and s['rejections']==len(ts)-1 and all(not z['accepted'] for z in ts[:-1]) and ts[-1]['accepted'],'complete trial rejection counters')
            require(s['accepted_h_mm']==ts[-1]['h_mm'],'actual accepted arc');path+=s['accepted_h_mm'];arithmetic(path,s['path_mm'])
            t=ts[-1];p=v(t,'start_position_mm');u=v(t,'start_direction');h=t['h_mm'];k=[v(t,'k'+str(j)) for j in range(1,5)]
            vec(v(s,'position_mm'),[p[j]+h*u[j]+h*h/6.*(k[0][j]+k[1][j]+k[2][j]) for j in range(3)])
            direction=[u[j]+h/6.*(k[0][j]+2.*(k[1][j]+k[2][j])+k[3][j]) for j in range(3)];norm=math.sqrt(sum(z*z for z in direction));vec(v(s,'direction'),[z/norm for z in direction])
        for field in queries:
            p=v(field,'position_mm');inside=[]
            for key in ('cache_before','cache_after'):
                cache=field[key];bounds=cache['ranges_mm'];ins=all(bounds[j][0]<=p[j]<=bounds[j][1] for j in range(3));inside.append(ins);require(ins==cache['contains_query'] and cache['actual_condition_map_match'],'actual cache identities/containment')
            require(field['cache_hit_before']==inside[0] and field['cache_refilled']==(not inside[0] and inside[1]) and field['outside_map_fallback']==(not inside[1]),'cache branches');exact['cache_flags']+=1
            native=v(field,'field_native');tesla=v(field,'field_T');vec(tesla,[z/r['acts_T_unit'] for z in native])
            if inside[1]:
                cache=field['cache_after'];bounds=cache['ranges_mm'];inv=cache['float_inverse_widths'];fractions=[np.float32((p[j]-bounds[j][0])*inv[j]) for j in range(3)];fx,fy,fz=fractions;gx,gy,gz=[np.float32(1.-z) for z in fractions]
                for j in range(3):
                    f=[np.float32(z) for z in cache['float_corner_fields'][j]]
                    kt=np.float32(cache['float_bscale_kT'])*(gx*(gy*(gz*f[0]+fz*f[1])+fy*(gz*f[2]+fz*f[3]))+fx*(gy*(gz*f[4]+fz*f[5])+fy*(gz*f[6]+fz*f[7])))
                    arithmetic(native[j],float(kt)*1000.*r['acts_T_unit'])
            else:vec(tesla,[1e-5]*3)
        bb=[x for x in obs if x['record']=='bound_before'];ba=[x for x in obs if x['record']=='bound_after'];result=[x for x in obs if x['record']=='propagator_result'];require(len(bb)==len(ba)==len(result)==1,'complete target records')
        require(bb[0]['requested_target'] and bb[0]['frame']==c['input']['frame'] and ba[0]['parameters']==c['bound_parameters'],'target parameters/frame');exact['target_parameters']+=1
        require(result[0]['steps']==len(steps)-1 and result[0]['path_mm']==steps[-1]['path_mm'],'final zero based loop counter')
        frame=c['input']['frame'];pos=v(c['state'],'global_position_mm');direction=v(c['state'],'global_direction');origin=[frame[j][3] for j in range(3)]
        local=[sum(frame[j][i]*(pos[j]-origin[j]) for j in range(3)) for i in range(3)];udir=[sum(frame[j][i]*direction[j] for j in range(3)) for i in range(3)]
        require(abs(local[2])<=1e-6 and abs(udir[2])>=1e-12,'returned target surface')
        vec(local,v(c['state'],'local_position_mm'))
        for z,expected in zip(c['state']['h'],local[:2]+[udir[0]/udir[2],udir[1]/udir[2]]):arithmetic(z[0],expected)
        counts.append({'sample_id':sid,'station':st,'steps':len(steps),'trials':len(trials),'queries':len(queries),'rejections':len(trials)-len(steps),'refills':sum(q['cache_refilled'] for q in queries),'outside':sum(q['outside_map_fallback'] for q in queries),'public_snapshots':len(logged),'last_poststep_target_reached':[x for x in actions if x['stage']=='postStep'][-1]['target_reached'],'path_mm':path})
    require(maxdiff<=1e-8,'independent scalar arithmetic budget');verify(out)
    return {'schema':'wb122_independent_scalar_audit_v1','integrity':'PASS','identities':len(frozen['hashes']),'exact_checks':exact,'fidelity_differences':fidelity_differences,'numeric_components':components,'max_normalized_difference':maxdiff,'engineering_budget':1e-8,'profiles':counts,'runtime_sha256':digest(out/'event/response.json'),'source_sha256':digest(Path(__file__)),'scope':'saved-only scalar RKN/field cache/target/logger; no physical calls','qualification':'NOT_EVALUATED'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('result',type=Path);args=p.parse_args();write_new(args.result,audit(args.out))
