"""Frozen complete-matrix saved-value audit; no fitted thresholds or propagation."""
from __future__ import annotations
import numpy as np
from alignment.wb93_transport_error import v,norm
from alignment.wb94_field_boundary import check_state

NAMES=('central_plus','central_minus','full','half')

def effect_matrix(values):
    a=np.asarray(values,dtype=float).reshape(25,4)
    if not np.isfinite(a).all():raise ValueError('nonfinite response')
    e=np.array([(a[1+4*k]-a[2+4*k])/4 for k in range(5)])
    return np.vstack([e,sum((-1)**k*x for k,x in enumerate(e))])

def remainders(values):
    a=np.asarray(values,dtype=float).reshape(25,4);e=effect_matrix(a)
    return np.array([[a[3+4*k]-a[0]-e[k],a[4+4*k]-a[0]-.5*e[k]] for k in range(6)])

def metrics(values,reference,scale):
    a=np.asarray(values,dtype=float).reshape(4,25,4);b=np.asarray(reference,dtype=float).reshape(25,4)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite metric input')
    return {'endpoint':norm((a-b)/scale),'cap_spread':norm((a[:,None]-a[None,:])/scale),
      'effect':max(norm((effect_matrix(x)-effect_matrix(b))/scale) for x in a),
      'taylor_excess':max(norm((remainders(x)-remainders(b))/scale) for x in a),
      'per_cap_endpoint':[norm((x-b)/scale) for x in a],
      'per_cap_effect':[norm((effect_matrix(x)-effect_matrix(b))/scale) for x in a],
      'per_cap_taylor_excess':[norm((remainders(x)-remainders(b))/scale) for x in a]}

def decide(cells,p,reference_budget,effect_budget,path_changes,failures,execution):
    if execution!='PASS' or failures:return 'UNKNOWN',[]
    expected=[(t,s,m) for t in p['step_tolerances'] for s in (1,2,3) for m in p['reference_modes']]
    if [(c['tolerance'],c['station'],c['reference_mode']) for c in cells]!=expected:return 'UNKNOWN',[]
    flags=[]
    for c in cells:
        if c['tolerance']==p['step_tolerances'][0]:continue
        base=next(x for x in cells if x['tolerance']==p['step_tolerances'][0] and x['station']==c['station'] and x['reference_mode']==c['reference_mode'])
        keys=('endpoint','cap_spread','effect') if c['reference_mode']=='mesh_z_double' else ('endpoint','effect')
        for k in keys:
            budget=effect_budget if k=='effect' else reference_budget
            flags.append({'tolerance':c['tolerance'],'station':c['station'],'reference_mode':c['reference_mode'],'metric':k,
              'default':base[k],'tightened':c[k],'ratio':c[k]/base[k] if base[k] else None,
              'budget':budget,'pass':bool(c[k]<=p['effect_reduction_required']*base[k] and
               base[k]-c[k]>p['effect_to_reference_uncertainty_required']*budget)})
    if not all(x['pass'] for x in flags):return 'NOT_SUPPORTED',flags
    return ('JOINT_CONTROL_NAVIGATION_SUPPORTED_ONLY' if path_changes else 'SUPPORTED_BUT_LIMITED'),flags

def validate_call(row,p,tolerance,cap,z,qop,save):
    o=row['options']
    expected={'stepTolerance':tolerance,'surfaceTolerance':p['surface_tolerance_mm'],'maxSteps':p['max_steps'],
      'maxRungeKuttaStepTrials':p['max_runge_kutta_step_trials'],'stepSizeCutOff':p['step_size_cutoff'],
      'maxStepSize_mm':1000*cap,'loopFraction':.5,'forward':True}
    if any(o[k]!=x for k,x in expected.items()) or not np.isfinite(o['pathLimit']):raise ValueError('actual option mismatch')
    counts=row['field_counts']
    if counts['total']!=counts['inside']+counts['outside'] or counts['gradient_calls']!=0:raise ValueError('field observation mismatch')
    if save:
        if len(row['field_queries'])!=counts['total'] or len(row['accepted_trace'])!=row['accepted_steps']:raise ValueError('trace truncation')
        last=0
        for step in row['accepted_trace']:
            v(step['position_mm'],3);v(step['direction'],3)
            if step['query_begin']!=last or not last<=step['query_end']<=counts['total']:raise ValueError('query-step indexing')
            last=step['query_end']
            if not np.isfinite([step['h_mm'],step['accepted_error_estimate'],step['error_over_tolerance']]).all():raise ValueError('nonfinite accepted trace')
        for q in row['field_queries']:
            pos=v(q['position_mm'],3);b=v(q['field_T'],3)
            inside=all(lo<=x<=hi for lo,x,hi in zip((-200,-200,-1762.3),pos,(200,200,2537.7)))
            if q['zone_id']!=(1 if inside else -1):raise ValueError('observed domain mismatch')
            if not inside and norm(b-1e-5)>1e-12:raise ValueError('outside field fallback changed')
    elif row['field_queries'] or row['accepted_trace']:raise ValueError('unauthorized trace sample')
    if row['status']!='PASS':return None
    h=check_state(row['state'],z,qop)
    if row['accepted_steps']!=row['propagator_steps_counter']+1:raise ValueError('accepted versus zero-based step counter')
    if 'official_default_state' in row:check_state(row['official_default_state'],z,qop)
    return h

def analyze(r,f,prior,reference,reference_summary,p):
    for k in ('input_xaod','ordinal','actual_run','actual_event'):
        if r[k]!=f[k]:raise ValueError('event identity mismatch')
    seed=np.r_[v(f['references'][0]['fixed_z_state']),f['references'][0]['q_over_p_per_MeV']]
    if not np.array_equal(v(r['seed'],5),seed) or r['seed_z_mm']!=f['references'][0]['z_state_mm']:raise ValueError('seed identity mismatch')
    if r['conditions']!=reference['conditions']:raise ValueError('conditions identity mismatch')
    if r['navigator']!={'resolveSensitive':True,'resolveMaterial':True,'resolvePassive':False}:raise ValueError('navigator configuration changed')
    expected_sensors=[s for t in f['wb92_targets'] for s in t['sensors']]
    if r['sensors']!=expected_sensors:raise ValueError('sensitive identity mismatch')
    default=r['defaults']
    if default['stepTolerance']!=1e-4 or default['surfaceTolerance']!=1e-4 or default['stepSizeCutOff']!=0 or default['maxRungeKuttaStepTrials']!=10000:raise ValueError('library defaults changed')
    if [q['tolerance'] for q in r['math_control']]!=p['step_tolerances']:raise ValueError('math control matrix changed')
    math_ok=all(np.isfinite([q['max_scaled_error'],q['wrong_sign_error'],q['wrong_unit_error']]).all() and
      q['max_scaled_error']<=p['analytic_control_scaled_tolerance'] and min(q['wrong_sign_error'],q['wrong_unit_error'])>p['analytic_control_scaled_tolerance'] for q in r['math_control'])
    order=[(t,c) for t in p['step_tolerances'] for c in p['max_step_sizes_m']]
    if [(x['tolerance'],x['cap_m']) for x in r['settings']]!=order:raise ValueError('complete option matrix missing')
    refs={q['name']:q['ladders'][-1]['samples'] for q in reference['modes']}
    scale=v(p['output_scales']);values={};repro=[];failures=[];paths={};controls=[];call_stats=[]
    for setting in r['settings']:
        tolerance,cap=setting['tolerance'],setting['cap_m'];baseline=tolerance==p['step_tolerances'][0]
        if [q['station'] for q in setting['targets']]!=[1,2,3]:raise ValueError('station multiplicity changed')
        oldcap=next(q for q in prior['caps'] if q['cap_m']==cap)
        entry=setting['entry_nominal'];eh=validate_call(entry,p,tolerance,cap,-1762.3,seed[4],True)
        controls.append({'tolerance':tolerance,'cap_m':cap,'path':'entry','error':None if eh is None else norm((eh-v(refs['mesh_z_double'][0]['entry']['h']))/scale)})
        if eh is None:failures.append({'tolerance':tolerance,'cap_m':cap,'path':'entry','error':entry.get('error_message')})
        elif baseline:
            official=entry.get('official_default_state');repro.append({'cap_m':cap,'path':'entry','official_error':None if official is None else norm((eh-v(official['h']))/scale),
              'saved_error':norm((eh-v(oldcap['targets'][0]['nominal']['entry']['h']))/scale)})
        for target,old in zip(setting['targets'],oldcap['targets']):
            station=target['station'];z=f['references'][station]['z_state_mm']
            if target['frame']!=old['frame'] or target['y']!=f['references'][station]['fixed_z_state']:raise ValueError('measurement/frame changed')
            if len(target['samples'])!=25:raise ValueError('sample matrix missing')
            current=[]
            oldrows=[old['nominal']['direct']]+[next(a for a in d['samples'] if a['multiplier']==1)[name]['direct'] for d in old['directions'] for name in NAMES]
            for i,row in enumerate(target['samples']):
                expected_label={'name':'nominal'} if i==0 else {'direction':p['directions'][(i-1)//4],'name':NAMES[(i-1)%4]}
                if row['label']!=expected_label:raise ValueError('sample label changed')
                xs=seed.copy()
                if i:
                    k=(i-1)//4;direction=np.eye(5)[k] if k<5 else np.array([1,-1,1,-1,1]);xs+=np.array(p['seed_steps'])*direction*(.5,-.5,.25,.125)[(i-1)%4]
                if not np.array_equal(v(row['seed'],5),xs):raise ValueError('perturbation changed')
                h=validate_call(row,p,tolerance,cap,z,xs[4],i in p['trace_sample_indices'])
                if h is None:failures.append({'tolerance':tolerance,'cap_m':cap,'station':station,'sample':expected_label,'error':row.get('error_message')})
                current.append(h);paths[(tolerance,cap,station,i)]=row['sensitive_sequence']
                call_stats.append({'tolerance':tolerance,'cap_m':cap,'station':station,'sample':expected_label,'status':row['status'],
                  **{k:row[k] for k in ('accepted_steps','rejected_trials','max_accepted_error_estimate','field_counts')}})
                if baseline and h is not None:
                    official=row.get('official_default_state');repro.append({'cap_m':cap,'station':station,'sample':expected_label,
                      'saved_error':norm((h-v(oldrows[i]['h']))/scale),'official_error':None if official is None else norm((h-v(official['h']))/scale)})
            values[(tolerance,cap,station)]=current
            inside=target['fixed_reference_start_nominal'];ih=validate_call(inside,p,tolerance,cap,z,seed[4],True)
            fs=f['wb96_fixed_start'];check_state(target['fixed_start'],fs['z_mm'],seed[4])
            if norm(v(target['fixed_start']['h'])-v(fs['seed'][:4]))>1e-9 or target['fixed_start']['time_Acts']!=fs['time_Acts']:raise ValueError('fixed reference start changed')
            if ih is None:failures.append({'tolerance':tolerance,'cap_m':cap,'station':station,'path':'fixed_start','error':inside.get('error_message')})
            controls.append({'tolerance':tolerance,'cap_m':cap,'station':station,'path':'fixed_start','error':None if ih is None else norm((ih-v(refs['mesh_z_double'][0]['targets'][station-1]['h']))/scale)})
    cells=[]
    for tolerance in p['step_tolerances']:
        for station in (1,2,3):
            arrays=[values[tolerance,c,station] for c in p['max_step_sizes_m']]
            if any(h is None for a in arrays for h in a):continue
            for mode in p['reference_modes']:
                base=[v(a['targets'][station-1]['h']) for a in refs[mode]]
                cells.append({'tolerance':tolerance,'station':station,'reference_mode':mode,**metrics(arrays,base,scale)})
    changed=[]
    for (t,c,s,i),sequence in paths.items():
        if t!=p['step_tolerances'][0] and sequence!=paths[p['step_tolerances'][0],c,s,i]:changed.append({'tolerance':t,'cap_m':c,'station':s,'sample_index':i,
          'default':paths[p['step_tolerances'][0],c,s,i],'tightened':sequence})
    baseline_ok=len(repro)==304 and all(q['official_error'] is not None and max(q['saved_error'],q['official_error'])<=p['baseline_reproduction_scaled_tolerance'] for q in repro)
    ref=reference_summary['modes']['mesh_z_double'];ref_ok=ref['reference']['gate']=='NUMERICAL_REFERENCE_SUPPORTED' and ref['effects']['all_pass']
    execution='PASS' if math_ok and baseline_ok and ref_ok else 'FAIL'
    hypothesis,flags=decide(cells,p,ref['reference']['max_fine_difference'],ref['effects']['max_fine_effect_difference'],changed,failures,execution)
    return {'execution_contract':execution,'mechanism_hypothesis':hypothesis,'math_controls_pass':math_ok,'reference_prerequisite_pass':ref_ok,
      'baseline_reproduction':repro,'baseline_max_saved_difference':max((q['saved_error'] for q in repro),default=None),
      'baseline_max_official_difference':max((q['official_error'] for q in repro if q['official_error'] is not None),default=None),
      'cells':cells,'decision_checks':flags,'failures':failures,'sensitive_path_changes':changed,'controls':controls,'call_stats':call_stats,
      'qualification':'NOT_EVALUATED','WB86_gate':'FAIL','WB92_gate':'FAIL','WB94_mechanism':'UNKNOWN','physical_field_accuracy':'UNKNOWN'}
