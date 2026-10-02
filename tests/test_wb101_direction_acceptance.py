import sys
from pathlib import Path
import numpy as np
import pytest
from alignment.wb101_direction_acceptance import criterion,qmatrix,complete_metrics

def test_prospective_reduction_and_floor_branch():
    p={'uncertainty_factor':10,'reduction_required':.5}
    assert criterion(100,50,.01,p)['pass']
    assert not criterion(100,51,.01,p)['pass']
    assert criterion(0,.01,.001,p)['floor_limited']
    assert criterion(0,.01,.001,p)['pass']
    assert not criterion(0,.011,.001,p)['pass']

def test_curvature_subtracted_and_all_caps_retained():
    reference=np.zeros((25,4));reference[3,0]=2;reference[4,0]=.6
    a=np.repeat(reference[None],4,axis=0)
    assert qmatrix(reference)[0,0]==.8
    assert complete_metrics(a,reference,np.ones(4))['full_minus_two_half_excess']==0
    a[3,3,0]+=10
    assert complete_metrics(a,reference,np.ones(4))['endpoint']==10

def test_generated_hook_before_finalize_and_only_frozen_markers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from wb101_sources import step_source,component_source
    s=step_source()
    assert s.index('m_control->judge')<s.index('extension.finalize')<s.index('pathAccumulated += h')
    assert 'return Acts::EigenStepper<>::step(state,navigator)' in s
    assert 'h *= directionReject ? 0.5 : stepSizeScaling;' in s
    c=component_source()
    assert 'directionThreshold==0' in c and 'NodeModel nodes' in c
    assert c.count('DECLARE_COMPONENT(WB101::DirectionAcceptanceAudit)')==1

def test_rejection_immutability_and_half_retry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_wb101_trials import retry
    a={'accepted':False,'direction_rejected':True,'h_mm':4,'start_position_mm':[1,2,3],
       'start_direction':[0,0,1],'start_time_Acts':3,'start_path_mm':2,'q_over_p_Acts':.01}
    b=a|{'h_mm':2};retry(a,b)
    for key,value in (('h_mm',1),('start_time_Acts',4),('start_path_mm',3),('q_over_p_Acts',.02)):
        with pytest.raises(ValueError):retry(a,b|{key:value})

def test_both_historical_manifest_schemas_fail_closed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from wb101_contract import artifact_hashes
    assert artifact_hashes({'artifacts':{'a':'h'}})=={'a':'h'}
    assert artifact_hashes({'attempts':[{'hashes':{'a':'h'}},{'hashes':{'b':'j'}}]})=={'a':'h','b':'j'}
    with pytest.raises(ValueError):artifact_hashes({'attempts':[{'hashes':{'a':'h'}},{'hashes':{'a':'j'}}]})
    with pytest.raises(ValueError):artifact_hashes({})

def test_scalar_metric_audit_and_json_serialization(tmp_path,monkeypatch):
    import json
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from finalize_wb101_results import scalar_audit
    from alignment.wb90_measurement_contract import write_new
    arrays=np.arange(25*4,dtype=float).reshape(25,4)/100
    scale=np.array([1,1,.001,.001]);p={'arms':[0,1e-8,1e-9,1e-10],'output_scales':scale.tolist(),'uncertainty_factor':10,'reduction_required':.5}
    ref=[{'targets':[{'h':r.tolist()} for _ in range(3)]} for r in arrays]
    reference={'modes':[{'name':mode,'ladders':[{'samples':ref},{'samples':ref}]} for mode in ('mesh_z_double','mesh_z_float')]}
    source=tmp_path/'reference.json';write_new(source,reference);write_new(tmp_path/'fixture.json',{'wb101_field_source':str(source)});write_new(tmp_path/'protocol.json',p)
    settings=[];cells=[];checks=[]
    for tau in p['arms']:
        for cap in (10,.1,.01,.001):
            settings.append({'direction_threshold':tau,'targets':[{'samples':[{'state':{'h':r.tolist()}} for r in arrays]} for _ in range(3)]})
        for station in (1,2,3):
            for mode in ('mesh_z_double','mesh_z_float'):
                metric=complete_metrics([arrays]*4,arrays,scale);U={k:0. for k in ('endpoint','cap_spread','effect','taylor_excess','full_minus_two_half_excess')}
                cells.append({'threshold':tau,'station':station,'reference_mode':mode,'uncertainty':U,**metric})
                if tau:
                    for key in U:
                        if mode=='mesh_z_float' and key=='cap_spread':continue
                        checks.append({'threshold':tau,'station':station,'reference_mode':mode,'metric':key,**criterion(0,0,0,p)})
    (tmp_path/'event').mkdir();write_new(tmp_path/'event/acts.json',{'settings':settings});summary={'cells':cells,'decision_checks':checks}
    json.dumps(summary,allow_nan=False)
    result=scalar_audit(tmp_path,summary)
    assert result['gate']=='PASS' and result['cells']==24 and result['decisions']==81
    checks[0]['pass']=False
    with pytest.raises(ValueError,match='scalar decision'):scalar_audit(tmp_path,summary)

def test_failed_final_trial_remains_scientific_failure():
    import copy,json
    from alignment.wb101_direction_acceptance import analyze
    seed=[0,0,0,0,1e-5]
    state={'h':[0]*4,'position_mm':[0,0,1],'direction':[0,0,1],
           'q_over_p_per_MeV':1e-5,'time_Acts':0,'bound_parameters':[0,0,0,0,.01,0]}
    base={'status':'PASS','state':state,'accepted_steps':1,'rejected_trials':0,
          'field_counts':{'total':3,'inside':3,'outside':0,'gradient_calls':0},
          'sensitive_sequence':[],'options':{'loopProtection':False},'last_free_state':{'position_mm':[0,0,1]},
          'target_frame':np.eye(4).tolist(),'propagator_steps_counter':0,'official_default_state':state}
    fixture={'input_xaod':'synthetic','ordinal':2270,'actual_run':100043,'actual_event':2270,
             'references':[{'z_state_mm':1}]*4}
    p={'arms':[0,1e-8,1e-9,1e-10],'max_step_sizes_m':[10,.1,.01,.001],'step_tolerance':1e-4,
       'expected_direct_calls':1200,'expected_auxiliary_calls':64,'expected_official_default_calls':316,
       'direction_uncertainty_allowance':1e-11,'field_probe_T_tolerance':1e-12,'baseline_scaled_tolerance':1e-9,
       'reference_modes':['mesh_z_double','mesh_z_float'],'output_scales':[1,1,.001,.001],
       'uncertainty_factor':10,'reduction_required':.5}
    old=fixture|{'conditions':{'min_mm':[0,0,1]},'navigator':{},'sensors':[],'seed':seed,'seed_z_mm':0,
                 'math_control':[],'settings':[]}
    for cap in p['max_step_sizes_m']:
        target=[{'station':s,'frame':[], 'y':[], 'samples':[copy.deepcopy(base)|{'seed':seed,'label':{'i':i}} for i in range(25)],
                 'fixed_start':state,'fixed_reference_start_nominal':copy.deepcopy(base)} for s in (1,2,3)]
        old['settings'].append({'tolerance':1e-4,'cap_m':cap,'entry_nominal':copy.deepcopy(base),'targets':target})
    raw=copy.deepcopy(old);raw['settings']=[];cid=0
    for tau in p['arms']:
        for setting in old['settings']:
            new=copy.deepcopy(setting);new['direction_threshold']=tau
            rows=[new['entry_nominal']]
            for t in new['targets']:rows.extend(t['samples']);rows.append(t['fixed_reference_start_nominal'])
            for row in rows:
                row.update(call_id=cid,start_state=state,direction_control={'threshold':tau,'allowance':1e-11,
                   'trials':int(tau>0),'accepted':int(tau>0),'node_queries':int(tau>0)*2,'position_rejected':0,'direction_rejected':0,'max_accepted_budget':0})
                cid+=1
            raw['settings'].append(new)
    raw['wb101_calls']=cid;raw['wb101_node_evidence']={'nodes_compared':0,'probe_count':0,'probe_max_T':0,
       'official_field_replaced':False,'actual_collinear':False,'identity_negative_controls_rejected':5}
    ref={'node_count':0,'probes':[],'domain_controls':[],'modes':[{'name':m,'ladders':[{'samples':[{'targets':[{'h':[0]*4}]*3}]*25}]*2} for m in p['reference_modes']]}
    rs={'modes':{'mesh_z_double':{'reference':{'gate':'NUMERICAL_REFERENCE_SUPPORTED'},'effects':{'all_pass':True}}}}
    assert analyze(raw,fixture,old,ref,rs,p)['hypothesis']=='SUPPORTED_BUT_LIMITED'
    failed=raw['settings'][4]['targets'][0]['samples'][0];failed['status']='FAIL';failed['error_message']='StepSizeAdjustmentFailed'
    failed['field_counts']['total']+=2;failed['field_counts']['inside']+=2
    failed['direction_control']['trials']+=1;failed['direction_control']['direction_rejected']+=1
    result=analyze(raw,fixture,old,ref,rs,p)
    assert result['hypothesis']=='NOT_SUPPORTED' and len(result['failures'])==1
    json.dumps(result,allow_nan=False)
