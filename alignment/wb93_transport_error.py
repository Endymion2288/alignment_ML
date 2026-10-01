"""Saved-value analysis of prospective single-event transport refinement."""
from __future__ import annotations
import numpy as np


def v(value,n=4):
    a=np.asarray(value,dtype=float).reshape(n)
    if not np.isfinite(a).all():raise ValueError('nonfinite diagnostic value')
    return a


def norm(value):return float(np.max(np.abs(value)))


def errors(sample,nominal,scale,effect=None):
    # Derivative uses +/- delta/2; full Taylor displacement is delta/4.
    own_effect=.25*(v(sample['central_plus'])-v(sample['central_minus']))
    effect=own_effect if effect is None else effect
    full=norm((v(sample['full'])-nominal-effect)/scale)
    half=norm((v(sample['half'])-nominal-.5*effect)/scale)
    return {'full_error':full,'half_error':half,'contraction':half/full if full else None,
            'effect':effect.tolist(),'directional_vs_coordinate_effect':norm((own_effect-effect)/scale)}


def windows(rows,p):
    flags=[]
    floor=p['roundoff_scaled_floor']
    for row in rows:
        tiny=max(row['full_error'],row['half_error'])<=floor
        contracts=row['full_error']>floor and row['half_error']/row['full_error']<=p['contraction_ceiling']
        flags.append(tiny or contracts)
    n=p['scaling_window_adjacent_pairs']
    found=[i for i in range(len(flags)-n+1) if all(flags[i:i+n])]
    return {'found':bool(found),'start_multipliers':[rows[i]['multiplier'] for i in found],
            'roundoff_limited_count':sum(max(r['full_error'],r['half_error'])<=floor for r in rows)}


def analytic_control(fixture,p):
    seed=np.r_[v(fixture['references'][0]['fixed_z_state']),fixture['references'][0]['q_over_p_per_MeV']]
    z=fixture['references'][0]['z_state_mm'];steps=np.asarray(p['seed_steps']);scale=np.asarray(p['output_scales'])
    worst=0.;bad_sign=0.;frozen_response=0.
    for ref in fixture['references'][1:]:
        dz=ref['z_state_mm']-z
        # Independent exact affine state and matrix, no ACTS helper shared.
        J=np.array([[1,0,dz,0,0],[0,1,0,dz,0],[0,0,1,0,0],[0,0,0,1,0]],dtype=float)
        def h(x):return np.array([x[0]+dz*x[2],x[1]+dz*x[3],x[2],x[3]])
        nominal=h(seed)
        for k in range(6):
            d=np.zeros(5)
            if k<5:d[k]=steps[k]
            else:d=steps*np.array([1,-1,1,-1,1])
            for s in p['step_multipliers']:
                delta=d*s
                sample={name:h(seed+factor*delta) for name,factor in [('central_plus',.5),('central_minus',-.5),('full',.25),('half',.125)]}
                r=errors(sample,nominal,scale)
                exact=.25*J@delta
                worst=max(worst,r['full_error'],r['half_error'],norm((np.asarray(r['effect'])-exact)/scale))
                if k==0 and s==1:
                    bad_sign=max(bad_sign,errors(sample,nominal,scale,-exact)['full_error'])
                    frozen_response=max(frozen_response,norm(exact/scale))
    return {'gate':'PASS' if worst<=p['analytic_control_scaled_tolerance'] and bad_sign>p['analytic_control_scaled_tolerance'] and frozen_response>p['analytic_control_scaled_tolerance'] else 'FAIL',
            'max_scaled_error':worst,'wrong_sign_error':bad_sign,'fixed_prediction_missing_effect':frozen_response,
            'scope':'zero_field_affine_math_only'}


def analyze(result,fixture,wb92,p):
    for name in ('input_xaod','ordinal','actual_run','actual_event'):
        if result[name]!=fixture[name]:raise ValueError('source/header mismatch: '+name)
    seed=np.r_[v(fixture['references'][0]['fixed_z_state']),fixture['references'][0]['q_over_p_per_MeV']]
    if not np.array_equal(v(result['seed'],5),seed):raise ValueError('seed changed')
    if result['seed_z_mm']!=fixture['references'][0]['z_state_mm']:raise ValueError('seed plane changed')
    if [c['max_step_size_m'] for c in result['caps']]!=p['max_step_sizes_m']:raise ValueError('caps changed')
    scale=np.asarray(p['output_scales']);cells=[];traces=[];reproduction=[];determinism=[];nominals={}
    for cap in result['caps']:
        capsize=cap['max_step_size_m']
        if [t['station'] for t in cap['targets']]!=[1,2,3]:raise ValueError('target multiplicity changed')
        for target in cap['targets']:
            station=target['station'];old=wb92['targets'][station]
            if not np.array_equal(v(target['y']),v(fixture['references'][station]['fixed_z_state'])):raise ValueError('y changed')
            if not np.array_equal(np.asarray(target['frame']),np.asarray(old['frame'])):raise ValueError('frame changed')
            if len(target['sensors'])!=len(old['sensors']):raise ValueError('sensor multiplicity changed')
            for sensor,prior in zip(target['sensors'],old['sensors']):
                if sensor['wafer']!=prior['wafer'] or norm(np.asarray(sensor['transform'])-np.asarray(prior['transform']))>1e-9:raise ValueError('actual sensitive geometry changed')
            nominal=v(target['h']);nominals[(capsize,station)]=nominal
            determinism.append(norm((v(target['repeat_h'])-nominal)/scale))
            directions=target['directions']
            if [d['name'] for d in directions]!=p['directions']:raise ValueError('directions changed')
            for k,direction in enumerate(directions):
                if [s['multiplier'] for s in direction['samples']]!=p['step_multipliers']:raise ValueError('ladder changed')
                rows=[]
                for j,s in enumerate(direction['samples']):
                    effect=None
                    if k==5:
                        effect=sum(.25*((-1)**i)*(v(directions[i]['samples'][j]['central_plus'])-v(directions[i]['samples'][j]['central_minus'])) for i in range(5))
                    rows.append({'multiplier':s['multiplier'],**errors(s,nominal,scale,effect)})
                small=[r for r in rows if r['multiplier']<=1]
                cells.append({'cap_m':capsize,'station':station,'direction':direction['name'],'rows':rows,
                              'window':windows(rows,p),'small_step_envelope':max(max(r['full_error'],r['half_error']) for r in small),
                              'small_step_median':float(np.median([max(r['full_error'],r['half_error']) for r in small]))})
            mixed=next(s for s in directions[5]['samples'] if s['multiplier']==1.)
            determinism.append(norm((v(target['repeat_mixed_full'])-v(mixed['full']))/scale))
            if capsize==10.:
                reproduction.append({'station':station,'nominal_error':norm((nominal-v(old['h']))/scale),
                                     'mixed_full_error':norm((v(mixed['full'])-v(old['xi_direction']['full']))/scale),
                                     'mixed_half_error':norm((v(mixed['half'])-v(old['xi_direction']['half']))/scale)})
            if [t['name'] for t in target['traces']]!=p['trace_variants']:raise ValueError('trace multiplicity')
            for trace in target['traces']:
                points=trace['steps']+[trace['endpoint']]
                if not trace['steps']:raise ValueError('empty official trace')
                signature=[];ids=[];maxgap=0.;previous=None
                for point in points:
                    position=v(point['position_mm'],3);field=v(point['field_T'],3)
                    signature.append(bool(np.allclose(field,p['field_default_signature_T'],rtol=0,atol=1e-15)))
                    if previous is not None:maxgap=max(maxgap,float(np.linalg.norm(position-previous)))
                    previous=position
                    if 'geometry_id' in point:ids.append(point['geometry_id'])
                traces.append({'cap_m':capsize,'station':station,'name':trace['name'],'step_count':len(trace['steps']),
                               'default_signature_count':sum(signature),'default_signature_transitions':sum(a!=b for a,b in zip(signature,signature[1:])),
                               'default_signature_z_mm':[float(v(point['position_mm'],3)[2]) for point,flag in zip(points,signature) if flag],
                               'geometry_sequence':[x for i,x in enumerate(ids) if i==0 or x!=ids[i-1]],
                               'max_recorded_gap_mm':maxgap,'endpoint_z_error_mm':abs(v(trace['endpoint']['position_mm'],3)[2]-fixture['references'][station]['z_state_mm'])})
    reproduction_ok=all(max(r[k] for k in ('nominal_error','mixed_full_error','mixed_half_error'))<=p['baseline_reproduction_scaled_tolerance'] for r in reproduction)
    deterministic=max(determinism)<=p['roundoff_scaled_floor']
    refinement=[]
    for station in (1,2,3):
        for direction in p['directions']:
            c=[r for r in cells if r['station']==station and r['direction']==direction]
            initial=c[0]['small_step_envelope'];fine=c[-1]['small_step_envelope']
            roundoff=initial<=p['roundoff_scaled_floor']
            refinement.append({'station':station,'direction':direction,'envelopes':[r['small_step_envelope'] for r in c],
                               'reduction_ratio':fine/initial if initial else None,'baseline_roundoff':roundoff,
                               'pass':roundoff or fine<=p['refinement_reduction_required']*initial,
                               'finest_window':c[-1]['window']['found']})
    support=all(r['pass'] and r['finest_window'] for r in refinement)
    return {'execution_contract':'PASS' if reproduction_ok and deterministic else 'FAIL',
            'mechanism_hypothesis':'SUPPORTED_BUT_LIMITED' if support and reproduction_ok and deterministic else 'NOT_SUPPORTED',
            'baseline_reproduction':reproduction,'max_repeat_error':max(determinism),'cells':cells,'refinement':refinement,'traces':traces,
            'nominal_cap_differences':[{'station':s,'cap_m':cap,'scaled_difference_from_10m':norm((nominals[(cap,s)]-nominals[(10.,s)])/scale)} for cap in p['max_step_sizes_m'] for s in (1,2,3)],
            'qualification':'NOT_EVALUATED','WB92_gate':'FAIL','map_domain_membership':'UNKNOWN'}
