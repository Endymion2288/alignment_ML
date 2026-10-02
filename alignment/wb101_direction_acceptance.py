"""Complete frozen endpoint criteria, separate scientific failures and identity."""
import numpy as np
from alignment.wb93_transport_error import v,norm
from alignment.wb94_field_boundary import check_state
from alignment.wb96_acts_tolerance import metrics,effect_matrix,remainders

def qmatrix(values):
    a=np.asarray(values,dtype=float).reshape(25,4)
    return np.asarray([a[3+4*k]-2*a[4+4*k]+a[0] for k in range(6)])

def complete_metrics(values,reference,scale):
    result=metrics(values,reference,scale)
    result['full_minus_two_half_excess']=max(norm((qmatrix(x)-qmatrix(reference))/scale) for x in values)
    result['per_cap_full_minus_two_half_excess']=[norm((qmatrix(x)-qmatrix(reference))/scale) for x in values]
    result['per_cap_direction_effect']=[[norm((a-b)/scale) for a,b in zip(effect_matrix(x),effect_matrix(reference))] for x in values]
    result['per_cap_direction_taylor_excess']=[[norm((a-b)/scale) for a,b in zip(remainders(x),remainders(reference))] for x in values]
    result['per_cap_direction_full_minus_two_half_excess']=[[norm((a-b)/scale) for a,b in zip(qmatrix(x),qmatrix(reference))] for x in values]
    return result

def criterion(base,candidate,U,p):
    if not np.isfinite([base,candidate,U]).all() or min(base,candidate,U)<0:raise ValueError('invalid criterion')
    allowance=p['uncertainty_factor']*U
    floor=base<=allowance
    passed=candidate<=allowance if floor else candidate<=p['reduction_required']*base and base-candidate>allowance
    return {'default':float(base),'enabled':float(candidate),'U':float(U),'floor_limited':bool(floor),'pass':bool(passed),
            'ratio':None if base==0 else float(candidate/base)}

def analyze(raw,fixture,old,reference,rs,p):
    for k in ('input_xaod','ordinal','actual_run','actual_event'):
        if raw[k]!=fixture[k]:raise ValueError('pilot identity')
    if raw['conditions']!=old['conditions'] or raw['navigator']!=old['navigator'] or raw['sensors']!=old['sensors']:raise ValueError('physical identity')
    if raw['math_control']!=old['math_control']:raise ValueError('analytic sign/unit controls changed')
    node=raw['wb101_node_evidence']
    if node['identity_negative_controls_rejected']!=5:raise ValueError('identity negative controls')
    if node['nodes_compared']!=reference['node_count'] or node['probe_count']!=len(reference['probes'])+len(reference['domain_controls']) or node['probe_max_T']>p['field_probe_T_tolerance'] or node['official_field_replaced'] or node['actual_collinear']:raise ValueError('node identity/control')
    order=[(tau,cap) for tau in p['arms'] for cap in p['max_step_sizes_m']]
    if [(r['direction_threshold'],r['cap_m']) for r in raw['settings']]!=order:raise ValueError('complete arm matrix')
    if raw['wb101_calls']!=p['expected_direct_calls']+p['expected_auxiliary_calls']:raise ValueError('call population')
    scale=np.asarray(p['output_scales']);values={};rows=[];failures=[];repro=[];pathchanges=[];calls={};free_distances=[]
    seed=v(raw['seed'],5)
    if not np.array_equal(seed,v(old['seed'],5)) or raw['seed_z_mm']!=old['seed_z_mm']:raise ValueError('seed identity')
    previous={}
    for setting in raw['settings']:
        tau,cap=setting['direction_threshold'],setting['cap_m'];oldsetting=next(x for x in old['settings'] if x['tolerance']==p['step_tolerance'] and x['cap_m']==cap)
        if [x['station'] for x in setting['targets']]!=[1,2,3]:raise ValueError('station multiplicity')
        allcalls=[('entry',0,0,setting['entry_nominal'],oldsetting['entry_nominal'])]
        for target,ot in zip(setting['targets'],oldsetting['targets']):
            station=target['station']
            if target['frame']!=ot['frame'] or target['y']!=ot['y'] or len(target['samples'])!=25:raise ValueError('target identity')
            current=[]
            for i,(r,o) in enumerate(zip(target['samples'],ot['samples'])):
                if r['label']!=o['label'] or r['seed']!=o['seed']:raise ValueError('perturbation identity')
                allcalls.append(('direct',station,i,r,o));current.append(v(r['state']['h']) if r['status']=='PASS' else None)
                if tau:
                    base=previous[(cap,station,i)]
                    if r['sensitive_sequence']!=base:pathchanges.append({'threshold':tau,'cap_m':cap,'station':station,'sample':i,'default':base,'enabled':r['sensitive_sequence']})
                else:previous[(cap,station,i)]=r['sensitive_sequence']
            values[tau,cap,station]=current
            if target['fixed_start']!=ot['fixed_start']:raise ValueError('fixed reference state')
            allcalls.append(('fixed_start',station,0,target['fixed_reference_start_nominal'],ot['fixed_reference_start_nominal']))
        for path,station,i,r,o in allcalls:
            identity={'threshold':tau,'cap_m':cap,'path':path,'station':station,'sample':i,'call_id':r['call_id']}
            if r['call_id'] in calls:raise ValueError('duplicate call')
            calls[r['call_id']]=r
            opt=r['options'];oo=o['options']
            if opt!=oo:raise ValueError('actual options changed')
            if r['start_state']['q_over_p_per_MeV']!=(seed[4] if path!='direct' else v(r['seed'],5)[4]):raise ValueError('start qop')
            cc=r['direction_control']
            if cc['threshold']!=tau or cc['allowance']!=p['direction_uncertainty_allowance']:raise ValueError('control constants')
            if tau:
                if cc['accepted']!=r['accepted_steps'] or cc['trials']!=cc['accepted']+cc['direction_rejected']+cc['position_rejected']:raise ValueError('trial count')
                if cc['max_accepted_budget']>tau:raise ValueError('accepted budget exceeded')
            elif any(cc[k] for k in ('trials','accepted','node_queries','direction_rejected','position_rejected')):raise ValueError('disabled intercepted original step')
            counts=r['field_counts']
            expected=3*r['accepted_steps']+2*r['rejected_trials']+int(opt['loopProtection'])
            # A failed final step can have queries/trials absent from postStep observations.
            if (r['status']=='PASS' and counts['total']!=expected) or counts['inside']+counts['outside']!=counts['total'] or counts['gradient_calls']:raise ValueError('actual query account')
            if r['status']!='PASS':failures.append(identity|{'error':r.get('error_message'),'error_code':r.get('error_code')})
            else:
                z=fixture['references'][station]['z_state_mm'] if station else raw['conditions']['min_mm'][2]
                check_state(r['state'],z,r['start_state']['q_over_p_per_MeV'])
                if r['accepted_steps']!=r['propagator_steps_counter']+1:raise ValueError('propagation count')
                pos=v(r['last_free_state']['position_mm'],3);frame=np.asarray(r['target_frame']);distance=float(abs(frame[:3,2]@(pos-frame[:3,3])))
                free_distances.append(identity|{'distance_mm':distance})
            if tau==0:
                if r['status']!='PASS' or 'official_default_state' not in r:raise ValueError('disabled official missing')
                saved=norm((v(r['state']['h'])-v(o['state']['h']))/scale);official=norm((v(r['state']['h'])-v(r['official_default_state']['h']))/scale)
                if max(saved,official)>p['baseline_scaled_tolerance'] or r['accepted_steps']!=o['accepted_steps'] or r['rejected_trials']!=o['rejected_trials'] or counts!=o['field_counts'] or r['sensitive_sequence']!=o['sensitive_sequence']:raise ValueError('disabled historical fidelity')
                repro.append(identity|{'saved_error':saved,'official_error':official})
            rows.append(identity|{'status':r['status'],'accepted_steps':r['accepted_steps'],'rejected_trials':r['rejected_trials'],'field_counts':counts,'direction_control':cc})
    if sorted(calls)!=list(range(1264)) or len(repro)!=p['expected_official_default_calls']:raise ValueError('complete call identity')
    ref_ok=rs['modes']['mesh_z_double']['reference']['gate']=='NUMERICAL_REFERENCE_SUPPORTED' and rs['modes']['mesh_z_double']['effects']['all_pass']
    refs={r['name']:r['ladders'] for r in reference['modes']};cells=[];checks=[];degradations=[]
    for tau in p['arms']:
        for station in (1,2,3):
            a=[values[tau,cap,station] for cap in p['max_step_sizes_m']]
            if any(x is None for row in a for x in row):continue
            for mode in p['reference_modes']:
                levels=refs[mode];b=np.array([v(x['targets'][station-1]['h']) for x in levels[-1]['samples']]);prev=np.array([v(x['targets'][station-1]['h']) for x in levels[-2]['samples']])
                U={'endpoint':norm((b-prev)/scale),'cap_spread':norm((b-prev)/scale),
                   'effect':norm((effect_matrix(b)-effect_matrix(prev))/scale),'taylor_excess':norm((remainders(b)-remainders(prev))/scale),
                   'full_minus_two_half_excess':norm((qmatrix(b)-qmatrix(prev))/scale)}
                cells.append({'threshold':tau,'station':station,'reference_mode':mode,'uncertainty':U,**complete_metrics(a,b,scale)})
                if tau==0:continue
                base=next(c for c in cells if c['threshold']==0 and c['station']==station and c['reference_mode']==mode)
                for k in U:
                    if mode=='mesh_z_float' and k=='cap_spread':continue
                    checks.append({'threshold':tau,'station':station,'reference_mode':mode,'metric':k,**criterion(base[k],cells[-1][k],U[k],p)})
                # Position/slopes each use their own component reference budget.
                for k in range(4):
                    err=float(np.max(np.abs(np.asarray(a)[:,:,k]-b[:,k]))/scale[k]);orig=float(np.max(np.abs(np.asarray([values[0,cap,station] for cap in p['max_step_sizes_m']])[:,:,k]-b[:,k]))/scale[k]);u=float(np.max(np.abs(b[:,k]-prev[:,k]))/scale[k])
                    if err>orig+p['uncertainty_factor']*u:degradations.append({'threshold':tau,'station':station,'reference_mode':mode,'component':k,'metric':'endpoint','default':orig,'enabled':err,'U':u})
                    ef=float(np.max(np.abs(np.asarray([effect_matrix(x) for x in a])[:,:,k]-effect_matrix(b)[:,k]))/scale[k]);of=float(np.max(np.abs(np.asarray([effect_matrix(values[0,cap,station]) for cap in p['max_step_sizes_m']])[:,:,k]-effect_matrix(b)[:,k]))/scale[k]);ue=float(np.max(np.abs(effect_matrix(b)[:,k]-effect_matrix(prev)[:,k]))/scale[k])
                    if ef>of+p['uncertainty_factor']*ue:degradations.append({'threshold':tau,'station':station,'reference_mode':mode,'component':k,'metric':'effect','default':of,'enabled':ef,'U':ue})
    if not failures and (len(cells)!=24 or len(checks)!=81):raise ValueError('incomplete decision matrix')
    reliable_fail=bool(failures or degradations or any(not x['pass'] for x in checks))
    hypothesis='UNKNOWN' if not ref_ok else ('NOT_SUPPORTED' if reliable_fail else ('JOINT_CONTROL_NAVIGATION_SUPPORTED_ONLY' if pathchanges else 'SUPPORTED_BUT_LIMITED'))
    return {'schema':'wb101_summary_v1','execution_contract':'PASS','reference_prerequisite':ref_ok,'hypothesis':hypothesis,
      'cells':cells,'decision_checks':checks,'degradations':degradations,'failures':failures,'sensitive_path_changes':pathchanges,
      'call_stats':rows,'baseline_reproduction':repro,'target_free_distances':free_distances,'calls':len(calls),
      'qualification':'NOT_EVALUATED','held_out_access':False,'production_backend_changed':False,'physical_field_accuracy':'UNKNOWN',
      'historical':{'WB86':'FAIL','WB92':'FAIL','WB94':'UNKNOWN','WB95':'SUPPORTED_BUT_LIMITED','WB96':'NOT_SUPPORTED','WB97':'NOT_SUPPORTED','WB98':'SUPPORTED_BUT_LIMITED','WB99':'NOT_SUPPORTED','WB100':'SUPPORTED_BUT_LIMITED'}}
