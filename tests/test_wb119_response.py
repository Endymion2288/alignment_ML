"""Synthetic repeated one-axis response and actual producer schema integration."""
import copy
import re
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb119_response import analyze
from wb119_sources import files
from test_wb117_response import synthetic as joint_synthetic,state


def synthetic(curved=False,noise=False):
    joint,oldcontrol,fixture,_=joint_synthetic();oldcontrol['original_FD_steps']=[.01,.01,1e-5,1e-5,1e-8]
    joint['control_used']=copy.deepcopy(oldcontrol)
    seed=np.asarray(oldcontrol['seed']).reshape(5);step=1e-8
    order=[(0.,0,r) for r in (0,1)]+[(m,sg,r) for m in (.25,.5,1.,2.) for sg in (1,-1) for r in (0,1)]
    samples=[]
    for i,(m,sg,r) in enumerate(order):
        x=seed.copy();x[4]+=m*sg*step
        samples.append({'sample_id':i,'step_multiple':m,'sign':sg,'repeat':r,'offset_multiple':m*sg,'seed':x.tolist()})
    c={'seed':oldcontrol['seed'],'seed_z_mm':0.,'targets':oldcontrol['targets'],'samples':samples,
       'qop_step_per_MeV':step,'output_scales':oldcontrol['output_scales']}
    runtime={k:copy.deepcopy(joint[k]) for k in ('identity','world','field_conditions','sensors','acts_MeV_unit','acts_T_unit')}
    runtime.update(schema='wb119_bounded_response_runtime_v1',control_used=copy.deepcopy(c),official_calls=36,rows=[])
    old=[]
    def response(x,station):
        dq=x[4]-seed[4];j=np.array([1e4,2e4,30.,40.])*station
        return j*dq+(np.array([1.,2.,.001,.002])*(dq/step)**2 if curved else 0.)
    for sample in samples:
        x=np.asarray(sample['seed'])
        for target in c['targets']:
            st=target['station'];h=response(x,st)
            if noise and sample['repeat']==1 and sample['step_multiple']!=0.:h+=np.array([.001,.002,1e-6,2e-6])
            runtime['rows'].append({'input':{'call_id':len(runtime['rows'])+1,'direction':'qop_scan','factor':sample['offset_multiple'],
                'sample_id':sample['sample_id'],'repeat':sample['repeat'],'step_multiple':sample['step_multiple'],'sign':sample['sign'],
                'station':st,'seed':x[:,None].tolist(),'seed_z_mm':0.,'frame':copy.deepcopy(target['frame']),
                'role':'POSITIVE_QOP_DIAGNOSTIC','start_state':state(x[:4],0.,x[4])},'has_value':True,'state':state(h,st*10.,x[4],time=1.)})
    for st in (1,2):
        for m in (1.,.5):
            for sg in (1,-1):
                x=seed.copy();x[4]+=sg*m*step
                old.append({'station':st,'step_multiple':m,'sign':sg,'historical_call_id':len(old)+100,
                            'seed':x.tolist(),'h':response(x,st)[:,None].tolist()})
    return runtime,c,fixture,joint,old


def test_linear_exact_repeated_derivatives_match_analytic_and_fixed_units():
    s=analyze(*synthetic())
    assert s['response_pattern']=='DETERMINISTIC_STEP_RESPONSE_MEASURED'
    assert s['official_calls']==s['successful_calls']==36
    assert len(s['historical_checks'])==16 and len(s['repeat_checks'])==18 and len(s['derivatives'])==16
    for d in s['derivatives']:
        assert d['Jq_per_MeV_inverse']==pytest.approx(np.array([1e4,2e4,30.,40.])*d['station'],rel=1e-12)
        assert d['central_midpoint_sum_scaled']==pytest.approx([0.]*4,abs=1e-11)
    assert not s['production_step_selected'] and s['qualification']=='NOT_EVALUATED'


def test_quadratic_midpoint_and_step_response_are_reported_without_acceptance_cut():
    s=analyze(*synthetic(curved=True));d=next(d for d in s['derivatives'] if d['step_multiple']==.5)
    assert d['central_midpoint_sum_scaled']==pytest.approx([.5,1.,.5,1.],rel=1e-11)
    assert s['classification']=='DIAGNOSTIC_ONLY_QOP_DERIVATIVE_RESPONSE'


def test_nonzero_repeat_difference_is_science_pattern_not_execution_failure():
    s=analyze(*synthetic(noise=True))
    assert s['integrity_gate']=='PASS' and s['response_pattern']=='RESPONSE_REPRODUCTION_DIFFERENCE'
    assert any(r['exact'] is False for r in s['repeat_checks'])


def test_historical_difference_is_reported_without_using_old_outputs_as_current():
    r,c,f,h,old=synthetic();old[0]['h'][0][0]+=.001
    s=analyze(r,c,f,h,old)
    assert s['response_pattern']=='RESPONSE_REPRODUCTION_DIFFERENCE'
    assert any(r['exact'] is False for r in s['historical_checks'])


def test_missing_nonzero_response_stays_unknown_without_extra_call():
    r,c,f,h,old=synthetic();r['rows'][4]['has_value']=False;r['rows'][4].pop('state')
    s=analyze(r,c,f,h,old)
    assert s['successful_calls']==35 and s['response_pattern']=='INCOMPLETE_OFFICIAL_RESPONSE'
    assert s['missing_calls']==[5]


def test_actual_producer_schema_and_call_budget_agree_with_consumer():
    source=files()['WB119Diagnostic/BoundedResponse.cxx']
    schema=re.search(r'Json out\{\{"schema","([^"]+)"\}',source).group(1)
    data=list(synthetic());data[0]['schema']=schema
    assert analyze(*data)['official_calls']==36 and 'if(count!=36)' in source
    assert 'seed.head<4>()-nominal.head<4>()' in source


@pytest.mark.parametrize('bad',['schema','source','units','sensor','world','field','callcount','seed_other','qop','offset','step','repeat',
                             'frame','role','qop_unit','missing_zero','nan','historical_seed','historical_duplicate','sample_order'])
def test_identity_and_single_axis_mutations_fail_closed(bad):
    r,c,f,h,old=synthetic();row=r['rows'][4]
    if bad=='schema':r['schema']='wb118_bounded_response_runtime_v1'
    elif bad=='source':r['identity']['input_xaod']='/wrong.root'
    elif bad=='units':r['acts_MeV_unit']*=1000.
    elif bad=='sensor':r['sensors'][0]['transform'][0][3]+=.1
    elif bad=='world':r['world']['name']='wrong'
    elif bad=='field':r['field_conditions']['scale']=-1.
    elif bad=='callcount':r['official_calls']=24
    elif bad=='seed_other':row['input']['seed'][0][0]+=1.
    elif bad=='qop':row['input']['seed'][4][0]*=1000.
    elif bad=='offset':row['input']['factor']*=2.
    elif bad=='step':c['qop_step_per_MeV']*=2.;r['control_used']=copy.deepcopy(c)
    elif bad=='repeat':row['input']['repeat']=1
    elif bad=='frame':row['input']['frame'][2][3]+=1.
    elif bad=='role':row['input']['role']='CHARGE_SIGN_NEGATIVE_CONTROL_ONLY'
    elif bad=='qop_unit':row['input']['start_state']['qop_acts']*=1000.
    elif bad=='missing_zero':r['rows'][0]['has_value']=False;r['rows'][0].pop('state')
    elif bad=='nan':row['state']['h'][0][0]=np.nan
    elif bad=='historical_seed':old[0]['seed'][4]*=1000.
    elif bad=='historical_duplicate':old[1]=copy.deepcopy(old[0])
    elif bad=='sample_order':c['samples'].reverse();r['control_used']=copy.deepcopy(c)
    with pytest.raises(ValueError):analyze(r,c,f,h,old)
