"""Direction calibration with source identities and numerical budgets kept separate."""
import json
from collections import Counter
import numpy as np

ACTS_T=0.000299792458
def vec(x):
    a=np.asarray(x,dtype=float).reshape(-1)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('invalid direction/position')
    return a
def maximum(x):return float(np.max(np.abs(x)))
def slope(u):
    u=vec(u)
    if u[2]<=1e-6:raise ValueError('nonforward slope')
    return u[:2]/u[2]
def slope_lipschitz(*directions):
    # Mean-value bound using max |ux,uy| and min uz along the convex segment.
    v=np.array([vec(x) for x in directions]);z=float(np.min(v[:,2]))
    if z<=1e-6:raise ValueError('nonforward Jacobian')
    return float(1/z+np.max(np.abs(v[:,:2]))/z**2)
def gate(defect,difference,uncertainty,p):
    if any(not np.isfinite(x) or x<0 for x in (defect,difference,uncertainty)):raise ValueError('invalid metric')
    resolved=defect>p['resolved_factor']*uncertainty
    budget=p['calibration_factor']*difference+p['uncertainty_factor']*uncertainty
    return {'resolved':resolved,'budget':budget,'false_negative':resolved and defect>budget,
            'near_zero_estimate':difference<=p['absolute_direction_floor']}
def verdict(complete,identity,numerical,misses):
    if not complete or not identity:return 'UNKNOWN'
    if misses:return 'NOT_SUPPORTED'
    return 'SUPPORTED_BUT_LIMITED' if numerical else 'UNKNOWN'
def algebra(p,u,h,q,queries):
    b0,bm,b1=(vec(x['field_native']) for x in queries)
    a=q*np.cross(u,b0);b=q*np.cross(u+h*a/2,bm);c=q*np.cross(u+h*b/2,bm);d=q*np.cross(u+h*c,b1)
    un=u+h*(a+2*b+2*c+d)/6
    return p+h*u+h*h*(a+b+c)/6,un/np.linalg.norm(un)
def node_field(positions,old,nodes):
    positions=np.asarray(positions);mesh=[np.asarray(x) for x in old['mesh_mm']]
    inside=np.all((positions>=np.array([a[0] for a in mesh]))&(positions<=np.array([a[-1] for a in mesh])),axis=1)
    result=np.full((len(positions),3),1e-5*ACTS_T);x=positions[inside]
    if len(x):
        ix=np.column_stack([np.clip(np.searchsorted(mesh[k],x[:,k],side='right')-1,0,len(mesh[k])-2) for k in range(3)])
        f=np.column_stack([(x[:,k]-mesh[k][ix[:,k]])/(mesh[k][ix[:,k]+1]-mesh[k][ix[:,k]]) for k in range(3)])
        b=np.zeros((len(x),3))
        for corner in range(8):
            high=np.array([(corner>>(2-k))&1 for k in range(3)]);idx=ix+high
            weights=np.prod(np.where(high,f,1-f),axis=1)
            b+=weights[:,None]*nodes[idx[:,0],idx[:,1],idx[:,2]]
        result[inside]=b*old['conditions']['bscale_kT']*old['conditions']['scale']*1000*ACTS_T
    return result

def aggregate(out,inventory,p,wb97,old,nodes):
    counts=Counter();classes={};traces={};examples=[];maxima=Counter();fidelity=None;terminal=None
    # Independent source audit; recompute new saved RKN algebra and each decision.
    with (out/'event/acts.json.doubling.ndjson').open() as f,(wb97/'steps.ndjson').open() as ref_file,(out/'metrics.ndjson').open('x') as metrics:
        first=json.loads(next(ref_file))
        if first['record']!='controls':raise ValueError('missing prior controls')
        for line in f:
            row=json.loads(line)
            if row['record']=='fidelity':
                if fidelity is not None:raise ValueError('duplicate fidelity')
                fidelity=row;continue
            if row['record']=='terminal':
                if terminal is not None:raise ValueError('duplicate terminal')
                terminal=row;continue
            if row['record']!='step' or terminal is not None:raise ValueError('record order')
            oldrow=json.loads(next(ref_file));t=row['trace'];i=row['index']
            if (t,i)!=(oldrow['trace'],oldrow['index']) or i!=traces.get(t,0):raise ValueError('source identity/order')
            meta=inventory['traces'][t]
            if any(row[k]!=meta[k] or row[k]!=oldrow[k] for k in ('tolerance','cap_m','path','station','sample')):raise ValueError('stratum identity')
            start=vec(row['start_position_mm']);u=vec(row['start_direction']);q=row['q_over_p_Acts'];h=row['h_mm']
            if h!=oldrow['saved_step']['h_mm'] or q!=oldrow['q_over_p_Acts'] or maximum(start-vec(oldrow['start_position_mm']))>p['closure_position_mm'] or maximum(u-vec(oldrow['start_direction']))>p['closure_direction']:raise ValueError('initial/arc identity')
            full=vec(row['full']['direction']);two=vec(row['half2']['direction']);ref=vec(oldrow['reference_ladder'][-1]['direction'])
            ref_prev=vec(oldrow['reference_ladder'][-2]['direction']);end=ref+vec(oldrow['direction_defect'])
            endp=vec(oldrow['reference_ladder'][-1]['position_mm'])+vec(oldrow['position_defect_mm'])
            source_ok=oldrow['closure_gate']=='PASS'
            for name,pp,uu,hh in (('full',start,u,h),('half1',start,u,h/2),('half2',vec(row['half1']['position_mm']),vec(row['half1']['direction']),h/2),('accepted_branch',start,u,h)):
                branch=row[name]
                ap,au=algebra(pp,uu,hh,q,branch['queries'])
                arithmetic=max(maximum(ap-vec(branch['position_mm'])),maximum(au-vec(branch['direction'])))
                maxima['RKN_arithmetic']=max(maxima['RKN_arithmetic'],arithmetic)
                if arithmetic>1e-9:raise ValueError('RKN export algebra')
                if len(branch['queries'])!=3:raise ValueError('query coverage')
            for name in ('full','accepted_branch'):
                cp=maximum(vec(row[name]['position_mm'])-endp);cu=maximum(vec(row[name]['direction'])-end)
                if abs(cp-row[name+'_position_closure_mm'])>1e-12 or abs(cu-row[name+'_direction_closure'])>1e-15:raise ValueError('source closure arithmetic')
                source_ok &= cp<=p['closure_position_mm'] and cu<=p['closure_direction']
                maxima[name+'_position_closure_mm']=max(maxima[name+'_position_closure_mm'],cp)
                maxima[name+'_direction_closure']=max(maxima[name+'_direction_closure'],cu)
            refs=oldrow['reference_ladder'];fine=maximum(ref-ref_prev)
            coarse=maximum(vec(refs[1]['direction'])-vec(refs[0]['direction']))
            reference_ok=oldrow['reference_gate']=='PASS'
            fields=[vec(x['field_native']) for x in row['full']['queries']]
            fields.append(node_field(np.array([vec(refs[-1]['position_mm'])]),old,nodes)[0])
            curvature=abs(q)*max(np.linalg.norm(x) for x in fields)
            arc_budget=curvature*max(abs(x['arc_residual_mm']) for x in refs)
            uncertainty=max(p['absolute_direction_floor'],fine,row['full_direction_closure'],arc_budget)
            defect=maximum(full-ref);difference=maximum(full-two);g=gate(defect,difference,uncertainty,p)
            lipschitz=slope_lipschitz(full,ref,ref_prev)
            ds=maximum(slope(full)-slope(ref));es=maximum(slope(full)-slope(two));gs=gate(ds,es,uncertainty*lipschitz,p)
            kind=oldrow['classification']['class'];stat=classes.setdefault(kind,Counter());stat['steps']+=1
            valid=source_ok and reference_ok;counts['source_unknown']+=not source_ok;counts['reference_unknown']+=not reference_ok
            for label,val in (('resolved',g['resolved']),('false_negative',g['false_negative']),('near_zero_resolved',g['resolved'] and g['near_zero_estimate']),('slope_false_negative',gs['false_negative'])):
                counts[label]+=bool(valid and val);stat[label]+=bool(valid and val)
            # Same-stage arithmetic sensitivity only. No altered-stage or global float oracle claim.
            queries=[x for name in ('full','half1','half2') for x in row[name]['queries']]
            positions=np.array([vec(x['position_mm']) for x in queries]);db=node_field(positions,old,nodes)
            field_diff=maximum(db-np.array([vec(x['field_native']) for x in queries]))/ACTS_T
            arithmetic_dirs=[]
            for name,pp,uu,hh,bb in (('full',start,u,h,db[:3]),('half1',start,u,h/2,db[3:6]),('half2',vec(row['half1']['position_mm']),vec(row['half1']['direction']),h/2,db[6:9])):
                _,du=algebra(pp,uu,hh,q,[{'field_native':b} for b in bb]);arithmetic_dirs.append(du-vec(row[name]['direction']))
            fd_effect=max(maximum(a) for a in arithmetic_dirs)
            maxima['same_stage_float_double_T']=max(maxima['same_stage_float_double_T'],field_diff)
            maxima['same_stage_direction_arithmetic']=max(maxima['same_stage_direction_arithmetic'],fd_effect)
            maxima['direction_defect']=max(maxima['direction_defect'],defect);maxima['direction_difference']=max(maxima['direction_difference'],difference)
            maxima['uncertainty']=max(maxima['uncertainty'],uncertainty);maxima['reference_fine']=max(maxima['reference_fine'],fine);maxima['arc_direction_budget']=max(maxima['arc_direction_budget'],arc_budget)
            result={k:row[k] for k in ('trace','index','tolerance','cap_m','path','station','sample','h_mm')}
            result.update(classification=kind,source_gate='PASS' if source_ok else 'UNKNOWN',reference_gate='PASS' if reference_ok else 'UNKNOWN',
              direction_defect=(full-ref).tolist(),direction_estimate=(p['calibration_factor']*(full-two)).tolist(),smooth_direction_estimate=(p['smooth_factor']*(full-two)).tolist(),defect_max=defect,difference_max=difference,
              uncertainty=uncertainty,arc_direction_budget=arc_budget,reference_fine=fine,reference_coarse=coarse,
              direction=g,slope_defect=(slope(full)-slope(ref)).tolist(),slope_estimate=(p['calibration_factor']*(slope(full)-slope(two))).tolist(),slope_uncertainty=uncertainty*lipschitz,slope=gs,
              same_stage_float_double_T=field_diff,same_stage_direction_arithmetic=fd_effect)
            if valid and (g['false_negative'] or gs['false_negative']) and len(examples)<30:examples.append(result)
            metrics.write(json.dumps(result,allow_nan=False)+'\n');traces[t]=i+1;counts['steps']+=1
        last=json.loads(next(ref_file))
        if last['record']!='terminal' or any(ref_file):raise ValueError('prior remaining records')
    complete=(len(traces)==p['expected_traces'] and counts['steps']==p['expected_steps'] and all(traces.get(i)==v['steps'] for i,v in enumerate(inventory['traces'])) and terminal is not None and terminal['steps']==p['expected_steps'])
    controls=json.loads((out/'controls.json').read_text());recomputed=0
    for c in controls['rows']:
        u1=vec(c['full']['direction']);u2=vec(c['half2']['direction']);rr=vec(c['reference_ladder'][-1]['direction']);rp=vec(c['reference_ladder'][-2]['direction'])
        gg=gate(maximum(u1-rr),maximum(u1-u2),max(p['absolute_direction_floor'],maximum(rr-rp)),p)
        if gg['false_negative']!=c['false_negative'] and c['reference_gate']=='PASS':raise ValueError('control gate mismatch')
        recomputed+=c['reference_gate']=='PASS' and gg['false_negative']
    if recomputed!=controls['false_negatives']:raise ValueError('control miss count')
    identity=fidelity is not None and terminal is not None and fidelity['gate']=='PASS' and terminal['replay_max_T']<=p['field_replay_T'] and counts['source_unknown']==0 and controls['execution_contract']=='PASS'
    numerical=counts['reference_unknown']==0 and controls['reference_unknown']==0
    actual=verdict(complete,identity,counts['reference_unknown']==0,counts['false_negative']+counts['slope_false_negative'])
    return {'schema':'wb99_summary_v1','execution_contract':'PASS' if complete and identity else 'FAIL','coverage_gate':'PASS' if complete else 'FAIL',
      'hypothesis':verdict(complete,identity,numerical,recomputed+counts['false_negative']+counts['slope_false_negative']),
      'actual_pilot_hypothesis':actual,'calibration':controls['calibration'],'qualification':'NOT_EVALUATED','held_out_access':False,
      'counts':dict(counts),'traces':len(traces),'classes':{k:dict(v) for k,v in classes.items()},'maxima':dict(maxima),
      'fidelity':fidelity,'terminal':terminal,'examples':examples,'controls':{k:v for k,v in controls.items() if k!='rows'},
      'WB86':'FAIL','WB92':'FAIL','WB94':'UNKNOWN','WB95':'SUPPORTED_BUT_LIMITED','WB96':'NOT_SUPPORTED','WB97':'NOT_SUPPORTED','WB98':'SUPPORTED_BUT_LIMITED'}
