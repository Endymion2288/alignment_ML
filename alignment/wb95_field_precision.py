"""Saved-value numerical precision audit; diagnostic double field is not an oracle."""
from __future__ import annotations
import numpy as np
from alignment.wb93_transport_error import v,norm,errors
from alignment.wb94_field_boundary import reference_check

NAMES=('central_plus','central_minus','full','half')
POINTS=('entry','interior',0,1,2)

def trilinear(nodes,fraction,scale):
    a=np.asarray(nodes,dtype=float).reshape(8,3);f=v(fraction,3)
    if not np.isfinite(a).all() or np.any(f< -1e-10) or np.any(f>1+1e-10) or not np.isfinite(scale):
        raise ValueError('invalid nodes/cell fraction/scale')
    return sum(a[c]*scale*np.prod([f[k] if c&(1<<(2-k)) else 1-f[k] for k in range(3)]) for c in range(8))

def point(row,name):return row[name] if isinstance(name,str) else row['targets'][name]

def field_contract(r,p,node_values):
    c=r['conditions'];z=p['expected_zone'];tol=p['double_trilinear_T_tolerance']
    if c['map_key']!=p['map_key'] or c['cache_key']!=p['cache_key'] or not c['wrapper_cache_identity'] or not c['map_IOV'] or not c['cache_IOV']:
        raise ValueError('conditions context/key/IOV mismatch')
    if c['min_mm']!=z['min_mm'] or c['max_mm']!=z['max_mm'] or c['zone_id']!=z['id'] or c['scale']!=p['expected_scale'] or c['bscale_kT']!=p['expected_bscale_kT']:
        raise ValueError('domain/scale/payload mismatch')
    mesh=[np.asarray(x,dtype=float) for x in r['mesh_mm']]
    if [len(x) for x in mesh]!=p['expected_mesh_dimensions'] or r['node_count']!=p['expected_nodes'] or node_values.shape!=(r['node_count'],3):
        raise ValueError('mesh/nodes dimension mismatch')
    for k,x in enumerate(mesh):
        if not np.array_equal(x,np.linspace(z['min_mm'][k],z['max_mm'][k],len(x))):
            # Source stores decimal mesh points; compare at sub-nanometre precision.
            if not np.allclose(x,np.linspace(z['min_mm'][k],z['max_mm'][k],len(x)),rtol=0,atol=1e-10):
                raise ValueError('unexpected mesh coordinates')
    expected_probes=3*len(mesh[2])+len(mesh[2])-1+12
    if len(r['probes'])!=expected_probes:raise ValueError('probe multiplicity changed')
    max_independent=0.;max_precision=0.;outside=0;inside=0
    for q in r['probes']+r['domain_controls']:
        pos=v(q['position_mm'],3);fb=v(q['float_T'],3);db=v(q['double_T'],3)
        expected_inside=all(a<=x<=b for a,x,b in zip(z['min_mm'],pos,z['max_mm']))
        if q['zone_id']!=(z['id'] if expected_inside else -1):raise ValueError('Cartesian domain mismatch')
        max_precision=max(max_precision,norm(fb-db))
        if not expected_inside:
            outside+=1
            if max(norm(fb-v(p['field_default_signature_T'],3)),norm(db-v(p['field_default_signature_T'],3)))>tol:
                raise ValueError('outside fallback changed or replaced with zero')
            continue
        inside+=1;ix=q['cell'];fraction=v(q['fractions'],3)
        exact_ix=[max(0,min(len(m)-2,int(np.searchsorted(m,pos[k],side='left'))-1)) for k,m in enumerate(mesh)]
        if ix!=exact_ix:raise ValueError('Cartesian mesh index mismatch')
        for k in range(3):
            if abs(fraction[k]-(pos[k]-mesh[k][ix[k]])/(mesh[k][ix[k]+1]-mesh[k][ix[k]]))>1e-12:
                raise ValueError('cell fraction changed')
        expected_nodes=[]
        for corner in range(8):
            x=ix[0]+((corner>>2)&1);y=ix[1]+((corner>>1)&1);zz=ix[2]+(corner&1)
            expected_nodes.append(node_values[(x*len(mesh[1])+y)*len(mesh[2])+zz])
        if not np.array_equal(np.asarray(q['nodes']),np.asarray(expected_nodes)):raise ValueError('different field nodes')
        independent=trilinear(q['nodes'],fraction,c['bscale_kT']*c['scale']*1000)
        max_independent=max(max_independent,norm(independent-db),norm(independent-v(q['independent_double_T'],3)))
    if max_independent>tol:raise ValueError('double implementation versus independent nodes disagreement')
    if [x['zone_id'] for x in r['domain_controls']]!=[-1,1,-1,-1,-1]:raise ValueError('domain negative controls failed')
    return {'gate':'PASS','max_double_vs_independent_T':max_independent,'max_float_vs_double_T':max_precision,
            'inside_probes':inside,'outside_probes':outside,'same_nodes':'VERIFIED','physical_field_accuracy':'UNKNOWN'}

def effects(ladders,p):
    scale=v(p['output_scales']);checks=[];taylor=[]
    for station in (1,2,3):
        all_effects=[]
        for ladder in ladders:
            rows=ladder['samples'];nominal=v(rows[0]['targets'][station-1]['h']);effect=[]
            for k in range(5):
                plus=v(rows[1+4*k]['targets'][station-1]['h']);minus=v(rows[2+4*k]['targets'][station-1]['h'])
                effect.append(.25*(plus-minus))
            effect.append(sum(((-1)**k)*x for k,x in enumerate(effect)))
            all_effects.append(effect)
            for k,name in enumerate(p['directions']):
                values={key:rows[1+4*k+j]['targets'][station-1]['h'] for j,key in enumerate(NAMES)}
                taylor.append({'dz_mm':ladder['dz_mm'],'station':station,'direction':name,**errors(values,nominal,scale,effect[k])})
        for k,name in enumerate(p['directions']):
            d=norm((all_effects[-1][k]-all_effects[-2][k])/scale)
            checks.append({'station':station,'direction':name,'fine_effect_difference':d,
                           'pass':bool(d<=p['reference_scaled_tolerance'])})
    return {'all_pass':bool(all(x['pass'] for x in checks)),'max_fine_effect_difference':max(x['fine_effect_difference'] for x in checks),
            'checks':checks,'taylor':taylor}

def mechanism_decision(refs,field,p):
    d=refs['mesh_z_double'];f=refs['mesh_z_float']
    if d['reference']['gate']!='NUMERICAL_REFERENCE_SUPPORTED' or not d['effects']['all_pass']:return 'UNKNOWN'
    if (f['reference']['gate']=='UNKNOWN' and d['reference']['max_fine_difference']<=p['precision_reduction_required']*f['reference']['max_fine_difference']
        and field['max_float_vs_double_T']>p['double_trilinear_T_tolerance']):return 'SUPPORTED_BUT_LIMITED'
    return 'NOT_SUPPORTED'

def analyze(r,f,prior,p,node_values):
    for k in ('input_xaod','ordinal','actual_run','actual_event'):
        if r[k]!=f[k]:raise ValueError('event identity changed')
    seed=np.r_[v(f['references'][0]['fixed_z_state']),f['references'][0]['q_over_p_per_MeV']]
    if not np.array_equal(seed,v(r['seed'],5)) or r['seed_z_mm']!=f['references'][0]['z_state_mm']:raise ValueError('seed changed')
    if [x['name'] for x in r['modes']]!=p['modes']:raise ValueError('mode identity changed')
    expected_sensors=[s for t in f['wb92_targets'] for s in t['sensors']]
    if len(r['sensors'])!=len(expected_sensors):raise ValueError('sensitive multiplicity changed')
    for a,b in zip(r['sensors'],expected_sensors):
        if a['wafer']!=b['wafer'] or norm(np.asarray(a['transform'])-np.asarray(b['transform']))>1e-9:raise ValueError('sensitive geometry changed')
    math=r['math_control']
    if not np.isfinite(list(math.values())).all() or math['max_scaled_error']>1e-8 or min(math['wrong_sign_error'],math['wrong_unit_error'])<=1e-8:
        raise ValueError('analytic mathematical controls failed')
    field=field_contract(r,p,node_values);scale=v(p['output_scales']);refs={};repro=0.;endpoint_differences=[]
    for mode in r['modes']:
        name=mode['name'];ladders=mode['ladders'];expected=p['original_rk4_dz_mm'] if name=='original_float' else p['mesh_rk4_dz_mm']
        if [x['dz_mm'] for x in ladders]!=expected:raise ValueError('precision ladder changed')
        for ladder in ladders:
            if len(ladder['samples'])!=25:raise ValueError('sample multiplicity changed')
            for i,row in enumerate(ladder['samples']):
                label={'name':'nominal'} if i==0 else {'direction':p['directions'][(i-1)//4],'name':NAMES[(i-1)%4]}
                if row['label']!=label:raise ValueError('perturbation identity changed')
                xs=seed.copy()
                if i:
                    k=(i-1)//4;sign=np.eye(5)[k] if k<5 else np.array([1,-1,1,-1,1]);xs+=np.array(p['seed_steps'])*sign*(.5,-.5,.25,.125)[(i-1)%4]
                if not np.array_equal(v(row['seed'],5),xs):raise ValueError('sample seed changed')
                if row['max_same_point_field_difference_T']<0 or not np.isfinite(row['max_same_point_field_difference_T']):raise ValueError('nonfinite field sensitivity')
                for key in POINTS:
                    a=point(row,key);h=v(a['h']);
                    zz=p['expected_zone']['min_mm'][2]+(p['interior_restart_offset_mm'] if key=='interior' else 0) if isinstance(key,str) else f['references'][key+1]['z_state_mm']
                    if a['z_mm']!=zz or a['q_over_p_per_MeV']!=xs[4] or not np.isfinite(a['time_Acts']):raise ValueError('reference coordinate/qop/time changed')
                    if name=='original_float':
                        old=next(x for x in prior['references'] if x['dz_mm']==ladder['dz_mm'])['samples'][i]
                        if old['label']!=row['label']:raise ValueError('prior sample identity mismatch')
                        repro=max(repro,norm((h-v(point(old,key)['h']))/scale))
        controls={**p,'reference_rk4_dz_mm':expected[-3:]}
        refs[name]={'reference':reference_check(ladders[-3:],controls),'effects':effects(ladders,p),
          'max_same_point_field_difference_T':max(x['max_same_point_field_difference_T'] for l in ladders for x in l['samples']),
          'transverse_stage_cell_transition_counts':[{'dz_mm':l['dz_mm'],'nominal':l['samples'][0]['transverse_stage_cell_transitions']} for l in ladders]}
    float_rows=r['modes'][1]['ladders'][-1]['samples'];double_rows=r['modes'][2]['ladders'][-1]['samples']
    for i,(a,b) in enumerate(zip(float_rows,double_rows)):
        for key in POINTS:endpoint_differences.append({'sample':a['label'],'point':key,'scaled_difference':norm((v(point(a,key)['h'])-v(point(b,key)['h']))/scale),
           'raw_difference':(v(point(a,key)['h'])-v(point(b,key)['h'])).tolist()})
    execution='PASS' if repro<=p['baseline_reproduction_scaled_tolerance'] else 'FAIL'
    return {'execution_contract':execution,'field_contract':field,'math_control':math,'baseline_reproduction_max_scaled_difference':repro,
            'modes':refs,'float_double_endpoint_differences':endpoint_differences,
            'mechanism_hypothesis':mechanism_decision(refs,field,p) if execution=='PASS' else 'UNKNOWN',
            'qualification':'NOT_EVALUATED','WB92_gate':'FAIL','WB94_mechanism':'UNKNOWN','physical_field_accuracy':'UNKNOWN'}
