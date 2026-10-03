"""Synthetic split-response identities; no current event response execution."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb118_response import analyze, decompose, objective, FACTORS
from test_wb117_response import synthetic as joint_synthetic, state


def synthetic(nonlinear=False):
    joint, oldcontrol, fixture, _ = joint_synthetic()
    historical = copy.deepcopy(joint)
    for row in historical['rows']:
        if nonlinear:
            f = row['input']['factor']; h = np.asarray(row['state']['h']).reshape(4)+f*f*np.array([.6, -.2, .001, .003])
            row['state'] = state(h, row['input']['station']*10., row['input']['seed'][4][0], time=1.)
    control = copy.deepcopy(oldcontrol); control['directions'] = []
    for direction in oldcontrol['directions']:
        for split in ('four_d_only', 'qop_only'):
            d = copy.deepcopy(direction); d['name'] += '_'+split; d['parent_direction'] = direction['name']; d['split'] = split
            d['delta'] = direction['delta'][:4]+[0.] if split == 'four_d_only' else [0.]*4+[direction['delta'][4]]
            control['directions'].append(d)
    control['factors'] = list(FACTORS); control['original_FD_steps'] = [1.]*5
    historical['control_used']['original_FD_steps'] = [1.]*5
    runtime = {k: copy.deepcopy(joint[k]) for k in ('identity', 'world', 'field_conditions', 'acts_MeV_unit', 'acts_T_unit', 'sensors')}
    runtime.update(schema='wb118_bounded_response_runtime_v1',control_used=copy.deepcopy(control),official_calls=24,rows=[])
    seed = np.asarray(control['seed']).reshape(5)
    for d in control['directions']:
        delta = np.asarray(d['delta'])
        for factor in FACTORS:
            for target in control['targets']:
                station = target['station']; x = seed+factor*delta; h = factor*np.asarray(d['jacobians'][str(station)])@delta
                if nonlinear:
                    h += factor**2*(np.array([.3,.2,.005,.007]) if d['split']=='four_d_only' else np.array([.1,-.1,-.003,.002]))
                runtime['rows'].append({'input':{'call_id':len(runtime['rows'])+1,'direction':d['name'],'factor':factor,'station':station,
                   'seed':x[:,None].tolist(),'seed_z_mm':0.,'frame':copy.deepcopy(target['frame']),
                   'role':'CHARGE_SIGN_NEGATIVE_CONTROL_ONLY' if d['split']=='qop_only' and factor==1. else 'POSITIVE_QOP_DIAGNOSTIC',
                   'start_state':state(x[:4],0.,x[4])},'has_value':True,'state':state(h,station*10.,x[4],time=1.)})
    return runtime, control, fixture, historical


def result(data):
    r,c,f,h=data
    return decompose(analyze(r,c,f,h),c,f,h)


def test_linear_response_decomposition_has_zero_interaction_and_zero_individual_errors():
    s=result(synthetic())
    assert s['official_calls']==24 and s['nominal_repeat_controls']==8 and s['joint_new_calls']==0
    assert s['decomposition_pattern']=='SIGNED_DECOMPOSITION_MEASURED'
    for row in s['decomposition']:
        for key in ('E_joint_scaled','E_four_d_scaled','E_qop_scaled','I_actual_scaled'):
            assert row[key]==pytest.approx([0.]*4,abs=1e-11)
    assert s['classification']=='DIAGNOSTIC_ONLY_RESPONSE_DECOMPOSITION'


def test_nonlinear_signed_terms_and_cancellation_have_independent_analytic_expectation():
    s=result(synthetic(nonlinear=True)); r=next(r for r in s['decomposition'] if r['factor']==.2 and r['station']==1)
    assert r['E_four_d_scaled']==pytest.approx([.012,.008,.2,.28])
    assert r['E_qop_scaled']==pytest.approx([.004,-.004,-.12,.08])
    assert r['E_joint_scaled']==pytest.approx([.024,-.008,.04,.12])
    assert r['I_actual_scaled']==pytest.approx([.008,-.012,-.04,-.24])
    assert r['signed_sum_scaled']==pytest.approx(r['E_joint_scaled'])
    assert r['cause_classification']=='NOT_ASSIGNED_UNCALIBRATED_FINITE_CHANGE'


def test_four_d_factor_one_remains_positive_while_qop_factor_one_is_negative_control():
    s=result(synthetic())
    for row in s['rows']:
        if row['factor']==1.:
            if row['direction'].endswith('four_d_only'):
                assert row['seed'][4]>0 and row['role']=='POSITIVE_QOP_DIAGNOSTIC'
            else:
                assert row['seed'][4]<0 and row['role']=='CHARGE_SIGN_NEGATIVE_CONTROL_ONLY'


def test_missing_split_nonzero_preserves_unknown_and_does_not_fill_from_joint():
    r,c,f,h=synthetic(); row=r['rows'][2]; row['has_value']=False; row.pop('state')
    s=result((r,c,f,h))
    assert s['successful_calls']==23 and s['decomposition_pattern']=='UNKNOWN_INCOMPLETE_RESPONSE'
    assert next(row for row in s['decomposition'] if row['direction']=='full' and row['factor']==.2 and row['station']==1)['status']=='UNKNOWN_INCOMPLETE_RESPONSE'


def test_split_objective_uses_four_d_prior_only_and_never_native_dummy_variance():
    s=result(synthetic())
    for row in s['joint_objectives']:
        if row['direction'].endswith('qop_only'):
            assert row['actual_objective']['prior4']==0.
        assert row['actual_objective']['qop_prior']=='NONE'
    assert s['dummy_qop_variance_used'] is False and s['qualification']=='NOT_EVALUATED'


@pytest.mark.parametrize('bad',['sensor_transform','source','units','world','field','duplicate_call','wrong_frame','wrong_seed','wrong_qop_unit',
    'wrong_role','missing_zero','nan','swapped_direction','alter_delta','alter_J','alter_joint_target','alter_joint_steps','joint_duplicate','missing_joint_count'])
def test_split_and_cross_job_identity_mutations_fail_closed(bad):
    r,c,f,h=synthetic(); row=r['rows'][2]
    if bad=='sensor_transform':r['sensors'][0]['transform'][0][3]+=.1
    elif bad=='source':r['identity']['input_xaod']='/wrong.root'
    elif bad=='units':r['acts_MeV_unit']*=1000.
    elif bad=='world':r['world']['name']='wrong'
    elif bad=='field':r['field_conditions']['scale']=-1.
    elif bad=='duplicate_call':row['input']['call_id']=1
    elif bad=='wrong_frame':row['input']['frame'][2][3]+=1.
    elif bad=='wrong_seed':row['input']['seed'][0][0]+=1.
    elif bad=='wrong_qop_unit':row['input']['start_state']['qop_acts']*=1000.
    elif bad=='wrong_role':r['rows'][4]['input']['role']='CHARGE_SIGN_NEGATIVE_CONTROL_ONLY'
    elif bad=='missing_zero':r['rows'][0]['has_value']=False;r['rows'][0].pop('state')
    elif bad=='nan':row['state']['h'][0][0]=np.nan
    elif bad=='swapped_direction':c['directions'].reverse();r['control_used']=copy.deepcopy(c)
    elif bad=='alter_delta':c['directions'][0]['delta'][4]=1e-8;r['control_used']=copy.deepcopy(c)
    elif bad=='alter_J':c['directions'][0]['jacobians']['1'][0][0]+=1.;r['control_used']=copy.deepcopy(c)
    elif bad=='alter_joint_target':h['control_used']['targets'][0]['frame'][2][3]+=1.
    elif bad=='alter_joint_steps':h['control_used']['original_FD_steps'][0]*=2.
    elif bad=='joint_duplicate':h['rows'][1]=copy.deepcopy(h['rows'][0])
    elif bad=='missing_joint_count':h['official_calls']=15
    with pytest.raises((ValueError,np.linalg.LinAlgError)):result((r,c,f,h))


@pytest.mark.parametrize('bad',['negative','singular','nan','zero_scale'])
def test_invalid_covariance_is_rejected_without_floor(bad):
    c=np.eye(4);sc=np.ones(4)
    if bad=='negative':c[0,0]=-1.
    elif bad=='singular':c[0,0]=0.
    elif bad=='nan':c[0,0]=np.nan
    elif bad=='zero_scale':sc[0]=0.
    with pytest.raises((ValueError,np.linalg.LinAlgError)):objective(np.ones(8),np.ones(4),c,[c,c],sc)
