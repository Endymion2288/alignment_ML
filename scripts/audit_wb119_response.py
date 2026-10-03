#!/usr/bin/env python3
"""One-axis repeated FD response diagnostic, without a production step selection."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public
from audit_wb117_response import check,array


def analyze(runtime,control,fixture,historical,old_qop):
    check(runtime['schema']=='wb119_bounded_response_runtime_v1' and runtime['control_used']==control,'schema/control')
    check((fixture['index'],fixture['ordinal'],fixture['actual_run'],fixture['actual_event'])==(12,2268,100044,2268),'single seen event')
    check(runtime['identity']==historical['identity']=={k:fixture[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},'event identity')
    for key in ('world','field_conditions','sensors','acts_MeV_unit','acts_T_unit'):
        check(runtime[key]==historical[key],'cross-job '+key)
    joint_control=historical['control_used']
    check(control['seed']==joint_control['seed'] and control['seed_z_mm']==joint_control['seed_z_mm'] and
          control['targets']==joint_control['targets'] and control['output_scales']==joint_control['output_scales'],'nominal controls')
    seed=array(control['seed'],(5,1)).reshape(5);scales=array(control['output_scales'],(4,));step=control['qop_step_per_MeV']
    check(step==1e-8 and seed[4]==1e-5 and np.all(scales>0),'original units/steps')
    expected_order=[(0.,0,r) for r in (0,1)]+[(m,sg,r) for m in (.25,.5,1.,2.) for sg in (1,-1) for r in (0,1)]
    check(len(control['samples'])==18 and runtime['official_calls']==len(runtime['rows'])==36,'call matrix')
    by={};differences=[];missing=[];rows=[];historical_checks=[]
    oldmap={(r['station'],r['step_multiple'],r['sign']):r for r in old_qop}
    check(len(oldmap)==len(old_qop)==8,'historical qop matrix')
    for sample_id,(multiple,sign,repeat) in enumerate(expected_order):
        sample=control['samples'][sample_id];x=seed.copy();x[4]+=multiple*sign*step
        check((sample['sample_id'],sample['step_multiple'],sample['sign'],sample['repeat'])==(sample_id,multiple,sign,repeat) and
              sample['offset_multiple']==multiple*sign and np.array_equal(array(sample['seed'],(5,)),x) and x[4]>0,'sample ordering/seed')
        for ti,target in enumerate(control['targets']):
            station=ti+1;call=2*sample_id+ti+1;row=runtime['rows'][call-1];inp=row['input']
            check(target['station']==station and inp['call_id']==call and inp['sample_id']==sample_id and inp['direction']=='qop_scan' and
                  inp['factor']==multiple*sign and inp['repeat']==repeat and inp['step_multiple']==multiple and inp['sign']==sign and
                  inp['station']==station and inp['seed_z_mm']==control['seed_z_mm'] and inp['frame']==target['frame'],'call/input identity')
            check(np.array_equal(array(inp['seed'],(5,1)).reshape(5),x) and inp['role']=='POSITIVE_QOP_DIAGNOSTIC','positive qop-only state')
            start=inp['start_state'];q=x[4]/runtime['acts_MeV_unit'];u=np.r_[x[2:4],1.];u/=np.linalg.norm(u)
            check(start['covariance_present'] is False and start['time_acts']==0. and start['qop_acts']==q,'bound units/covariance/time')
            check(np.max(np.abs(array(start['h'],(4,1)).reshape(4)-x[:4])/scales)<=1e-9 and
                  np.allclose(array(start['global_position_mm'],(3,1)).reshape(3),np.r_[x[:2],control['seed_z_mm']],rtol=0,atol=1e-9) and
                  np.allclose(array(start['global_direction'],(3,1)).reshape(3),u,rtol=0,atol=1e-12),'start roundtrip')
            check(type(row['has_value']) is bool,'optional type')
            key=(station,multiple,sign,repeat);by[key]=row
            detail={'call_id':call,'sample_id':sample_id,'station':station,'step_multiple':multiple,'sign':sign,'repeat':repeat,'qop_per_MeV':x[4],'has_value':row['has_value']}
            if not row['has_value']:
                check(multiple!=0. and 'state' not in row,'nominal missing-state guard');missing.append(call)
            else:
                state=row['state'];h=array(state['h'],(4,1)).reshape(4)
                check(state['qop_acts']==q and state['covariance_present'] is False and np.isfinite(state['time_acts']),'returned state qop/covariance')
                frame=array(target['frame'],(4,4));local=np.linalg.solve(frame,np.r_[array(state['global_position_mm'],(3,1)).reshape(3),1.])[:3]
                u=frame[:3,:3].T@array(state['global_direction'],(3,1)).reshape(3)
                check(abs(local[2])<=1e-6 and np.allclose(local,array(state['local_position_mm'],(3,1)).reshape(3),rtol=0,atol=1e-8) and abs(u[2])>=1e-12 and
                      np.allclose(np.r_[local[:2],u[:2]/u[2]],h,rtol=0,atol=1e-8),'returned frame')
                if multiple==0.:check(state['h']==target['official_h'],'nominal exact')
                detail['h']=h.tolist()
                if multiple in (.5,1.):
                    old=oldmap[station,multiple,sign];check(old['seed']==x.tolist(),'historical seed guard')
                    delta=(h-array(old['h'],(4,1)).reshape(4))/scales;exact=state['h']==old['h']
                    historical_checks.append({'station':station,'step_multiple':multiple,'sign':sign,'repeat':repeat,
                        'current_call_id':call,'historical_call_id':old['historical_call_id'],'exact':exact,'scaled_difference':delta.tolist()})
            rows.append(detail)
    repeated=[];derivatives=[]
    for station in (1,2):
        h0=array(control['targets'][station-1]['official_h'],(4,1)).reshape(4)
        for multiple,sign in [(0.,0)]+[(m,sg) for m in (.25,.5,1.,2.) for sg in (1,-1)]:
            a,b=[by[station,multiple,sign,r] for r in (0,1)]
            if a['has_value'] and b['has_value']:
                delta=(array(b['state']['h'],(4,1))-array(a['state']['h'],(4,1))).reshape(4)/scales;exact=a['state']['h']==b['state']['h']
                repeated.append({'station':station,'step_multiple':multiple,'sign':sign,'exact':exact,'scaled_difference':delta.tolist()})
            else:repeated.append({'station':station,'step_multiple':multiple,'sign':sign,'exact':'UNKNOWN_MISSING_STATE'})
        for multiple in (.25,.5,1.,2.):
            for repeat in (0,1):
                p,m=[by[station,multiple,sg,repeat] for sg in (1,-1)]
                row={'station':station,'step_multiple':multiple,'repeat':repeat,'status':'UNKNOWN_MISSING_STATE'}
                if p['has_value'] and m['has_value']:
                    hp=array(p['state']['h'],(4,1)).reshape(4);hm=array(m['state']['h'],(4,1)).reshape(4)
                    j=(hp-hm)/(2*multiple*step)
                    row.update({'status':'CENTRAL_RESPONSE_MEASURED','Jq_per_MeV_inverse':j.tolist(),
                         'effect_in_original_step_scaled':(j*step/scales).tolist(),
                         'plus_effect_scaled':((hp-h0)/scales).tolist(),'minus_effect_scaled':((hm-h0)/scales).tolist(),
                         'central_midpoint_sum_scaled':((hp+hm-2*h0)/scales).tolist()})
                derivatives.append(row)
    comparisons=[]
    for station in (1,2):
        for repeat in (0,1):
            base=next(r for r in derivatives if (r['station'],r['step_multiple'],r['repeat'])==(station,1.,repeat))
            for multiple in (.25,.5,2.):
                row=next(r for r in derivatives if (r['station'],r['step_multiple'],r['repeat'])==(station,multiple,repeat))
                entry={'station':station,'repeat':repeat,'step_multiple':multiple,'comparison_reference_step':1.,'status':'UNKNOWN_MISSING_STATE'}
                if base['status']==row['status']=='CENTRAL_RESPONSE_MEASURED':
                    a=np.asarray(base['effect_in_original_step_scaled']);b=np.asarray(row['effect_in_original_step_scaled']);delta=b-a
                    entry.update({'status':'STEP_DIFFERENCE_MEASURED','scaled_effect_difference':delta.tolist(),
                        'max_abs_difference':float(np.max(np.abs(delta))),'difference_norm':float(np.linalg.norm(delta)),
                        'signed_component_ratios':[float(v/w) if w!=0. else 'UNKNOWN_ZERO_DENOMINATOR' for v,w in zip(b,a)]})
                comparisons.append(entry)
    nonexact=any(r['exact'] is False for r in repeated+historical_checks)
    pattern='INCOMPLETE_OFFICIAL_RESPONSE' if missing else ('RESPONSE_REPRODUCTION_DIFFERENCE' if nonexact else 'DETERMINISTIC_STEP_RESPONSE_MEASURED')
    return {'schema':'wb119_qop_derivative_summary_v1','integrity_gate':'PASS','classification':'DIAGNOSTIC_ONLY_QOP_DERIVATIVE_RESPONSE',
       'response_pattern':pattern,'identity':runtime['identity'],'official_calls':36,'successful_calls':36-len(missing),'nominal_exact_guards':4,
       'missing_calls':missing,'rows':rows,'repeat_checks':repeated,'historical_checks':historical_checks,'derivatives':derivatives,'step_comparisons':comparisons,
       'production_step_selected':False,'qop_all_positive':True,'truth_momentum_used':False,'dummy_variance_used':False,
       'association':'UNKNOWN_OR_AMBIGUOUS_WB114','covariance_calibration':'UNVERIFIED','historical_generation_conditions':'UNKNOWN',
       'held_out_access':False,'new_reconstruction_calls':0,'qualification':'NOT_EVALUATED','physics_screening':'NOT_EVALUATED','final_oracle':'NOT_EVALUATED'}


def audit(out):
    runtime=read_public(out/'event/response.json');stream=[json.loads(l) for l in (out/'event/response.json.calls.ndjson').read_text().splitlines()]
    check(len(stream)==73 and stream[-1]=={'record':'terminal','status':'COMPLETED','official_calls':36},'complete callstream')
    for i,row in enumerate(runtime['rows']):
        check(stream[2*i]=={'record':'before_official','input':row['input']} and stream[2*i+1]=={'record':'after_official','row':row},'callstream/runtime identity')
    return analyze(runtime,read_public(out/'control.json'),read_public(out/'fixture.json'),read_public(out/'historical_runtime.json'),read_public(out/'historical_qop_responses.json'))
