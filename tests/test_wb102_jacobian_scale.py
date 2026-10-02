import copy
import sys
from pathlib import Path
import numpy as np
import pytest
from alignment.wb102_jacobian_scale import response, decision, seeds, budget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from alignment.wb90_measurement_contract import read_public

P = read_public(ROOT/'configs/research_review/wp102_jacobian_scale_contract.json')


def analytic_values(lam, matrix, curvature=0.):
    x = np.zeros(5)
    rows = seeds(x, lam, P)
    return np.array([matrix@r + curvature*np.dot(r,r)*np.array([1,2,3,4]) for r in rows])


def test_jacobian_units_central_denominator_and_mixed_curvature():
    j = np.arange(20, dtype=float).reshape(4,5)+1
    j[:,4] *= 1e6
    expected = j*np.array(P['seed_steps'])[None,:]/np.array(P['output_scales'])[:,None]
    for lam in P['lambdas']:
        actual, k, mixed = response(analytic_values(lam,j), lam, P)
        np.testing.assert_allclose(actual,j,rtol=1e-14)
        np.testing.assert_allclose(k,expected,rtol=1e-14)
        np.testing.assert_allclose(mixed,np.zeros(4),atol=1e-13)
        _, curved_k, curved_m = response(analytic_values(lam,j,1e3), lam, P)
        np.testing.assert_allclose(curved_k,k,rtol=1e-13)
        assert np.max(abs(curved_m)) > 0
    _, wrong_k, _ = response(analytic_values(.5,j*np.array([1,1,1,1,1000])), .5, P)
    assert not decision(abs(wrong_k-expected),budget(expected,P))['pass']


def test_no_single_element_or_scale_exemption():
    b = np.ones((4,5))*1e-6
    a = np.zeros((4,5)); a[3,4] = 1.01e-6
    assert not decision(a,b)['pass']
    assert not decision(a,b)['pass_matrix'][3][4]
    for lam in (0,-1,.3):
        with pytest.raises(ValueError): response(np.zeros((25,4)),lam,P)
    a = np.zeros((25,4)); a[24,3] = np.nan
    with pytest.raises(ValueError): response(a,1,P)
    with pytest.raises(ValueError): response(np.zeros((24,4)),1,P)


def test_generation_keeps_original_acceptance_and_scope():
    from wb102_sources import files
    from wb101_sources import files as parent_files
    generated = files(); parent = parent_files()
    for name in ('Acceptance.h','NodeModel.h','DirectionStep.inc','CompensatedRK4.h','ReferenceRK4.h'):
        assert generated['WB102Diagnostic/'+name] == parent['WB101Diagnostic/'+name]
    actual = generated['WB102Diagnostic/JacobianScaleAudit.cxx']
    assert '*m_fixture.at("wb102_lambda").get<double>()' in actual
    assert 'if(historical && m_fixture.at("wb102_lambda")==1.)' in actual
    assert actual.count('DECLARE_COMPONENT(WB102::JacobianScaleAudit)') == 1
    reference = generated['WB102Diagnostic/ScaleReferenceAudit.cxx']
    assert 'namespace WB102Reference {' in reference
    assert '*m_fixture.at("wb102_lambda").get<double>()' in reference
    assert 'WB95Reference::integrate' in reference


def test_scalar_independence_complete_decisions_and_corruption():
    from audit_wb102_results import scalar_audit
    raw, refs, cells, checks, ref_checks = {}, {}, [], [], []
    matrix = np.arange(20,dtype=float).reshape(4,5)
    for lam in P['lambdas']:
        values = analytic_values(lam,matrix)
        rows = [{'state':{'h':[[x] for x in r]}} for r in values]
        refrows = [{'targets':[{'h':[[x] for x in r]} for _ in range(3)]} for r in values]
        raw[lam] = {'settings':[{'direction_threshold':tau,'cap_m':cap,'targets':[{'samples':copy.deepcopy(rows)} for _ in range(3)]}
                                for tau in P['arms'] for cap in P['max_step_sizes_m']]}
        refs[lam] = {'modes':[{'name':mode,'ladders':[{'samples':copy.deepcopy(refrows)} for _ in range(3)]}
                              for mode in P['reference_modes']]}
        j,k,m = response(values,lam,P); b = budget(k,P); u = np.zeros((4,5)); um = np.zeros(4)
        for mode in P['reference_modes']:
            for station in (1,2,3):
                ref_checks.append({'lambda':lam,'mode':mode,'station':station,'metric':'reference_U_K',
                                   **decision(u,.1*b)})
                for tau in P['arms']:
                    for cap in P['max_step_sizes_m']:
                        cell = {'lambda':lam,'mode':mode,'station':station,'threshold':tau,'cap_m':cap,
                                'J':j.tolist(),'K':k.tolist(),'reference_K':k.tolist(),'U_K':u.tolist(),'B':b.tolist(),
                                'mixed_closure':m.tolist(),'reference_mixed_closure':m.tolist(),'U_M':um.tolist()}
                        cells.append(cell)
                        if mode == 'mesh_z_double' and lam in P['plateau_lambdas'] and tau in P['candidate_arms']:
                            identity = {key:cell[key] for key in ('lambda','mode','station','threshold','cap_m')}
                            checks.append(identity|{'metric':'reference_agreement',**decision(u,b)})
                            checks.append(identity|{'metric':'mixed_agreement',**decision(um,b.sum(axis=1))})
    for a,b in zip(P['plateau_lambdas'][:-1],P['plateau_lambdas'][1:]):
        k = response(analytic_values(a,matrix),a,P)[1]; allow = budget(k,P)
        for station in (1,2,3):
            ref_checks.append({'lambdas':[a,b],'mode':'mesh_z_double','station':station,'metric':'reference_plateau',
                               **decision(np.zeros((4,5)),allow)})
            for tau in P['candidate_arms']:
                for cap in P['max_step_sizes_m']:
                    checks.append({'lambdas':[a,b],'mode':'mesh_z_double','station':station,'threshold':tau,'cap_m':cap,
                                   'metric':'scale_plateau',**decision(np.zeros((4,5)),allow)})
    summary = {'cells':cells,'decision_checks':checks,'reference_checks':ref_checks}
    audit = scalar_audit(summary,raw,refs,P)
    assert audit['cells'] == 384 and audit['decisions'] == 192 and audit['reference_checks'] == 30
    broken = copy.deepcopy(summary); broken['decision_checks'][-1]['pass'] = False
    with pytest.raises(ValueError,match='scalar frozen decision'): scalar_audit(broken,raw,refs,P)
    broken = copy.deepcopy(summary); broken['cells'][-1]['J'][3][4] *= 1000
    with pytest.raises(ValueError,match='scalar arithmetic'): scalar_audit(broken,raw,refs,P)


def test_lambda_and_cap_population_is_not_optional():
    from alignment.wb102_jacobian_scale import analyze
    with pytest.raises(ValueError,match='complete lambda ladder'):
        analyze({1:{}},{1:{}},{1:{}},{},{},P)


def synthetic_population():
    from alignment.wb102_jacobian_scale import calls
    matrix = np.array([[1,0,100,0,1e3],[0,1,0,100,0],[0,0,1,0,1],[0,0,0,1,0]],dtype=float)
    fixture = {'wb101_protocol':{'direction_uncertainty_allowance':1e-11},
               'wb95_protocol':{'directions':['x','y','tx','ty','q_over_p','alternating_mixed'],
                                'output_scales':P['output_scales'],'roundoff_scaled_floor':1e-11,
                                'reference_scaled_tolerance':1e-8,'reference_contraction_ceiling':.7}}
    identity = {'input_xaod':'synthetic','ordinal':2270,'actual_run':100043,'actual_event':2270,
                'seed':[0]*5,'seed_z_mm':0,'conditions':{},'navigator':{},'sensors':[],
                'defaults':{},'math_control':{},'wb101_node_evidence':{},'wb101_muon_mass_Acts':1}
    labels = [{'name':'nominal'}]+[{'direction':d,'name':name} for d in fixture['wb95_protocol']['directions']
                                 for name in ('central_plus','central_minus','full','half')]
    def state(xs,z):
        h=matrix@xs; u=np.r_[h[2:],1];u/=np.linalg.norm(u)
        return {'h':h.tolist(),'position_mm':[h[0],h[1],z],'direction':u.tolist(),
                'q_over_p_per_MeV':xs[4],'time_Acts':z,'bound_parameters':[0,0,0,0,xs[4]*1000,z]}
    raw,refs,fixtures = {},{},{}
    for lam in P['lambdas']:
        xs = seeds(np.zeros(5),lam,P); settings=[]; cid=0
        def row(x,z,tau):
            nonlocal cid
            frame=np.eye(4);frame[2,3]=z
            s=state(x,z)
            value={'call_id':cid,'status':'PASS','options':{'loopProtection':False},
                   'target_frame':frame.tolist(),'start_state':state(x,0),'state':s,
                   'accepted_steps':1,'rejected_trials':0,'propagator_steps_counter':0,
                   'field_counts':{'total':3,'inside':3,'outside':0,'gradient_calls':0},'sensitive_sequence':[],
                   'direction_control':{'threshold':tau,'allowance':1e-11,'accepted':int(tau>0),
                     'trials':int(tau>0),'position_rejected':0,'direction_rejected':0,'node_queries':int(tau>0)*2,
                     'max_accepted_budget':0},'official_default_state':s}
            cid+=1;return value
        for tau in P['arms']:
            for cap in P['max_step_sizes_m']:
                setting={'direction_threshold':tau,'cap_m':cap,'tolerance':1e-4,'entry_nominal':row(xs[0],1,tau),'targets':[]}
                for station in (1,2,3):
                    frame=np.eye(4);frame[2,3]=station
                    sample=[row(x,station,tau)|{'seed':x.tolist(),'label':label} for x,label in zip(xs,labels)]
                    setting['targets'].append({'station':station,'frame':frame.tolist(),'y':[0]*4,
                       'fixed_start':state(np.zeros(5),0),'samples':sample,'fixed_reference_start_nominal':row(xs[0],station,tau)})
                settings.append(setting)
        assert len([r for s in settings for r in calls(s)])==1264
        raw[lam]=copy.deepcopy(identity)|{'settings':settings,'wb102_lambda':lam,'wb101_calls':1264}
        reference_rows=[]
        for x,label in zip(xs,labels):
            def rs(z): return {'h':(matrix@x).tolist(),'z_mm':z,'q_over_p_per_MeV':x[4],'time_Acts':z}
            reference_rows.append({'label':label,'seed':x.tolist(),'entry':rs(1),'interior':rs(2),
                                  'targets':[rs(s) for s in (1,2,3)],'outside_stages':1,'inside_stages':1})
        refs[lam]=copy.deepcopy(identity)|{'wb102_lambda':lam,'mesh_mm':[],'node_count':0,'probes':[],
             'domain_controls':[],'modes':[{'name':mode,'ladders':[{'dz_mm':dz,'samples':copy.deepcopy(reference_rows)}
               for dz in P['reference_dz_mm']]} for mode in P['reference_modes']]}
        fixtures[lam]=copy.deepcopy(fixture)
    return raw,refs,fixtures,copy.deepcopy(raw[1]),copy.deepcopy(refs[1])


def test_full_conjunction_reference_failure_and_one_bad_cap():
    from alignment.wb102_jacobian_scale import analyze
    raw,refs,fixtures,historic,oldref=synthetic_population()
    result=analyze(raw,refs,fixtures,historic,oldref,P)
    assert result['hypothesis']=='SUPPORTED_BUT_LIMITED' and len(result['decision_checks'])==192
    bad=copy.deepcopy(raw)
    setting=next(s for s in bad[.125]['settings'] if s['direction_threshold']==1e-10 and s['cap_m']==.001)
    r=setting['targets'][2]['samples'][17]
    r['state']['h'][0]+=1e-3;r['state']['position_mm'][0]+=1e-3
    result=analyze(bad,refs,fixtures,historic,oldref,P)
    assert result['hypothesis']=='NOT_SUPPORTED'
    assert any(not c['pass'] and c['cap_m']==.001 for c in result['decision_checks'])
    broken=copy.deepcopy(refs)
    # The same finest-sample error defeats reference contraction/precision.
    broken[.125]['modes'][1]['ladders'][-1]['samples'][1]['targets'][2]['h'][0]+=1e-3
    assert analyze(raw,broken,fixtures,historic,oldref,P)['hypothesis']=='UNKNOWN'
    missing=copy.deepcopy(raw);missing[.5]['settings'].pop()
    with pytest.raises(ValueError,match='complete arm/cap matrix'): analyze(missing,refs,fixtures,historic,oldref,P)
