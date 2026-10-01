"""Independent saved-value field-domain and split-transport mechanism analysis."""
from __future__ import annotations
import numpy as np
from alignment.wb93_transport_error import v,norm,errors

NAMES=('central_plus','central_minus','full','half')

def check_state(s,z,qop):
    h=v(s['h']);x=v(s['position_mm'],3);u=v(s['direction'],3);b=v(s['bound_parameters'],6)
    if abs(x[2]-z)>1e-6 or u[2]<=0 or abs(np.linalg.norm(u)-1)>1e-10:
        raise ValueError('state plane/direction contract')
    if norm(h-np.r_[x[:2],u[:2]/u[2]])>1e-9:raise ValueError('state chart mismatch')
    if abs(s['q_over_p_per_MeV']-qop)>1e-16 or abs(b[4]*.001-qop)>1e-16:
        raise ValueError('q/p unit or continuation mismatch')
    if not np.isfinite(s['time_Acts']) or abs(b[5]-s['time_Acts'])>1e-10:
        raise ValueError('state time mismatch')
    return h

def field_contract(r,p):
    c=r['conditions'];expected=p['expected_zone'];tol=p['field_comparison_T_tolerance']
    if c['map_key']!=p['map_key'] or c['cache_key']!=p['cache_key'] or not c['wrapper_cache_identity']:
        raise ValueError('conditions key/context mismatch')
    if c['min_mm']!=expected['min_mm'] or c['max_mm']!=expected['max_mm'] or c['zone_id']!=expected['id']:
        raise ValueError('conditions domain changed')
    if c['scale']!=p['expected_scale'] or not c['map_IOV'] or not c['cache_IOV']:
        raise ValueError('conditions scale/IOV mismatch')
    max_cache=0.;max_order=0.
    if len(r['probes'])!=len(p['boundary_probe_offsets_mm']) or len(r['reverse_probes'])!=len(r['probes']):
        raise ValueError('probe multiplicity')
    for a,b,offset in zip(r['probes'],reversed(r['reverse_probes']),p['boundary_probe_offsets_mm']):
        pos=v(a['position_mm'],3)
        if abs(pos[2]-(expected['min_mm'][2]+offset))>1e-10 or norm(pos-v(b['position_mm'],3))>1e-10:
            raise ValueError('probe coordinate mismatch')
        if a['zone_id']!=(-1 if offset<0 else expected['id']) or b['zone_id']!=a['zone_id']:
            raise ValueError('Cartesian zone membership mismatch')
        max_order=max(max_order,norm(v(a['field_T'],3)-v(b['field_T'],3)))
        if offset<0 and norm(v(a['field_T'],3)-v(p['field_default_signature_T'],3))>tol:
            raise ValueError('outside map field mismatch')
    if [q['zone_id'] for q in r['domain_controls']]!=[-1,expected['id'],-1,-1,-1]:
        raise ValueError('Cartesian domain negative controls failed')
    for a in r['probes']+r['reverse_probes']+r['domain_controls']:
        max_cache=max(max_cache,norm(v(a['field_T'],3)-v(a['fresh_field_T'],3)))
    if max_cache>tol or max_order>tol:raise ValueError('cache/order-dependent field')
    return {'gate':'PASS','max_cache_difference_T':max_cache,'max_order_difference_T':max_order,
            'boundary_jump_T':(v(r['probes'][4]['field_T'],3)-v(r['probes'][2]['field_T'],3)).tolist(),
            'physical_outside_coverage':'UNKNOWN'}

def reference_check(ladders,p):
    if [x['dz_mm'] for x in ladders]!=p['reference_rk4_dz_mm']:raise ValueError('reference ladder changed')
    if any(len(x['samples'])!=25 for x in ladders):raise ValueError('reference sample multiplicity')
    labels=[{'name':'nominal'}]+[{'direction':d,'name':name} for d in p['directions'] for name in NAMES]
    checks=[];scale=v(p['output_scales'])
    for i,label in enumerate(labels):
        rows=[x['samples'][i] for x in ladders]
        if any(x['label']!=label or x['outside_stages']<=0 or x['inside_stages']<=0 for x in rows):
            raise ValueError('reference sample identity/scope')
        for point in ('entry','interior',0,1,2):
            states=[x[point] if isinstance(point,str) else x['targets'][point] for x in rows]
            if len({x['z_mm'] for x in states})!=1:raise ValueError('reference plane changed')
            if len({x['q_over_p_per_MeV'] for x in states})!=1:raise ValueError('reference qop changed')
            e0=norm((v(states[1]['h'])-v(states[0]['h']))/scale)
            e1=norm((v(states[2]['h'])-v(states[1]['h']))/scale)
            small=max(e0,e1)<=p['roundoff_scaled_floor']
            ok=e1<=p['reference_scaled_tolerance'] and (small or e0>0 and e1/e0<=p['reference_contraction_ceiling'])
            checks.append({'sample':label,'point':point,'coarse_difference':e0,'fine_difference':e1,
                           'ratio':e1/e0 if e0 else None,'roundoff_limited':small,'pass':bool(ok)})
    return {'gate':'NUMERICAL_REFERENCE_SUPPORTED' if all(x['pass'] for x in checks) else 'UNKNOWN',
            'max_fine_difference':max(x['fine_difference'] for x in checks),'checks':checks,
            'rigorous_remainder_bound':'NOT_ESTABLISHED'}

def mechanism_decision(metrics,reference,p):
    if reference['gate']!='NUMERICAL_REFERENCE_SUPPORTED':return 'UNKNOWN'
    if any(m['direct_max_error']<=p['roundoff_scaled_floor'] or m['direct_mixed_max_taylor']<=p['roundoff_scaled_floor'] or
           m['direct_cap_spread']<=p['roundoff_scaled_floor'] for m in metrics):return 'UNKNOWN'
    return 'SUPPORTED_BUT_LIMITED' if all(m['all_criteria'] for m in metrics) else 'NOT_SUPPORTED'

def analyze(r,f,wb93,p):
    for name in ('input_xaod','ordinal','actual_run','actual_event'):
        if r[name]!=f[name]:raise ValueError('source/header identity mismatch')
    seed=np.r_[v(f['references'][0]['fixed_z_state']),f['references'][0]['q_over_p_per_MeV']]
    if not np.array_equal(v(r['seed'],5),seed) or r['seed_z_mm']!=f['references'][0]['z_state_mm']:
        raise ValueError('seed changed')
    field=field_contract(r,p);reference=reference_check(r['references'],p)
    math=r['math_control']
    if not np.isfinite(list(math.values())).all() or math['max_scaled_error']>1e-8 or min(math['wrong_sign_error'],math['wrong_unit_error'])<=1e-8:
        raise ValueError('independent analytic controls failed')
    if [x['cap_m'] for x in r['caps']]!=p['max_step_sizes_m']:raise ValueError('caps changed')
    scale=v(p['output_scales']);cells=[];reproduction=[];traces=[];nominals={}
    for cap,prior in zip(r['caps'],wb93['caps']):
        if [x['station'] for x in cap['targets']]!=[1,2,3]:raise ValueError('station multiplicity changed')
        for t,old in zip(cap['targets'],prior['targets']):
            s=t['station'];z=f['references'][s]['z_state_mm'];nominals[(cap['cap_m'],s)]={}
            if not np.array_equal(v(t['y']),v(old['y'])) or t['frame']!=old['frame'] or len(t['sensors'])!=len(old['sensors']):
                raise ValueError('measurement/frame/sensitive geometry changed')
            for sensor,previous in zip(t['sensors'],old['sensors']):
                if sensor['wafer']!=previous['wafer'] or norm(np.asarray(sensor['transform'])-np.asarray(previous['transform']))>1e-9:
                    raise ValueError('sensitive geometry changed')
            for name in ('direct','boundary_split'):
                nominals[(cap['cap_m'],s)][name]=check_state(t['nominal'][name],z,seed[4])
            for name in ('interior_split','fixed_reference_start'):
                nominals[(cap['cap_m'],s)][name]=check_state(t[name],z,seed[4])
            check_state(t['nominal']['entry'],p['expected_zone']['min_mm'][2],seed[4])
            check_state(t['interior_prefix'],p['expected_zone']['min_mm'][2]+p['interior_restart_offset_mm'],seed[4])
            check_state(t['fixed_start'],p['expected_zone']['min_mm'][2]+p['interior_restart_offset_mm'],seed[4])
            repro=norm((nominals[(cap['cap_m'],s)]['direct']-v(old['h']))/scale)
            if [x['name'] for x in t['directions']]!=p['directions']:raise ValueError('direction identity changed')
            for k,(d,od) in enumerate(zip(t['directions'],old['directions'])):
                if [x['multiplier'] for x in d['samples']]!=p['step_multipliers']:raise ValueError('perturbation ladder changed')
                for path in ('direct','boundary_split'):
                    rows=[]
                    for j,(sample,os) in enumerate(zip(d['samples'],od['samples'])):
                        sign=np.eye(5)[k] if k<5 else np.array([1,-1,1,-1,1]);delta=np.asarray(p['seed_steps'])*sign*sample['multiplier']
                        for name,a in zip(NAMES,(.5,-.5,.25,.125)):
                            check_state(sample[name][path],z,seed[4]+a*delta[4])
                            check_state(sample[name]['entry'],p['expected_zone']['min_mm'][2],seed[4]+a*delta[4])
                            if path=='direct':repro=max(repro,norm((v(sample[name][path]['h'])-v(os[name]))/scale))
                        values={name:sample[name][path]['h'] for name in NAMES};effect=None
                        if k==5:effect=sum(.25*((-1)**a)*(v(t['directions'][a]['samples'][j]['central_plus'][path]['h'])-v(t['directions'][a]['samples'][j]['central_minus'][path]['h'])) for a in range(5))
                        rows.append({'multiplier':sample['multiplier'],**errors(values,nominals[(cap['cap_m'],s)][path],scale,effect)})
                    cells.append({'cap_m':cap['cap_m'],'station':s,'direction':d['name'],'path':path,'rows':rows,
                                  'small_step_envelope':max(max(x['full_error'],x['half_error']) for x in rows if x['multiplier']<=1)})
            reproduction.append({'cap_m':cap['cap_m'],'station':s,'max_scaled_difference':repro})
            for trace in t['traces']:
                seq=[];outside=0;difference=0.
                for q in trace['steps']:
                    v(q['position_mm'],3);outside+=q['zone_id']==-1
                    difference=max(difference,norm(v(q['field_T'],3)-v(q['fresh_field_T'],3)))
                    gid=q['geometry_id']
                    if gid&0x000000000fffff00 and (not seq or seq[-1]!=gid):seq.append(gid)
                traces.append({'cap_m':cap['cap_m'],'station':s,'path':trace['path'],'count':len(trace['steps']),
                               'outside_points':outside,'max_cache_difference_T':difference,'sensitive_sequence':seq})
    fine=r['references'][-1]['samples'][0];metrics=[]
    for s in (1,2,3):
        truth=v(fine['targets'][s-1]['h']);raw={path:[norm((nominals[(c,s)][path]-truth)/scale) for c in p['max_step_sizes_m']] for path in ('direct','boundary_split','interior_split','fixed_reference_start')}
        spread={path:max(norm((nominals[(a,s)][path]-nominals[(b,s)][path])/scale) for a in p['max_step_sizes_m'] for b in p['max_step_sizes_m']) for path in raw}
        taylor={path:max(max(row['full_error'],row['half_error']) for cell in cells if cell['station']==s and cell['direction']=='alternating_mixed' and cell['path']==path for row in cell['rows'] if row['multiplier']==1) for path in ('direct','boundary_split')}
        ed,eb,es,ei=(max(raw[path]) for path in raw);td,tb=taylor['direct'],taylor['boundary_split'];unc=reference['max_fine_difference']
        criteria={'nominal_reduction':eb<=p['effect_reduction_required']*ed,'spread_reduction':spread['boundary_split']<=p['effect_reduction_required']*spread['direct'],
                  'sham_retention':es>=p['sham_retention_required']*ed,'inside_control':ei<=p['effect_reduction_required']*ed,
                  'mixed_taylor_reduction':tb<=p['effect_reduction_required']*td,
                  'effect_exceeds_reference_uncertainty':ed-eb>=p['effect_to_reference_uncertainty_required']*unc}
        metrics.append({'station':s,'per_cap_reference_errors':raw,'cap_spreads':spread,'direct_max_error':ed,'boundary_max_error':eb,
                        'sham_max_error':es,'inside_max_error':ei,'direct_cap_spread':spread['direct'],
                        'direct_mixed_max_taylor':td,'boundary_mixed_max_taylor':tb,'criteria':criteria,'all_criteria':bool(all(criteria.values()))})
    reproduction_ok=max(x['max_scaled_difference'] for x in reproduction)<=p['baseline_reproduction_scaled_tolerance']
    trace_ok=all(x['max_cache_difference_T']<=p['field_comparison_T_tolerance'] for x in traces)
    return {'execution_contract':'PASS' if reproduction_ok and trace_ok else 'FAIL','field_contract':field,'reference':reference,
            'mechanism_hypothesis':mechanism_decision(metrics,reference,p) if reproduction_ok and trace_ok else 'UNKNOWN',
            'baseline_reproduction':reproduction,'math_control':math,'metrics':metrics,'cells':cells,'traces':traces,
            'qualification':'NOT_EVALUATED','WB92_gate':'FAIL','physical_field_accuracy':'UNKNOWN'}
