import copy
import json
from pathlib import Path
import numpy as np
import pytest
from alignment.wb94_field_boundary import check_state,field_contract,reference_check,mechanism_decision

P=json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp94_field_boundary_contract.json').read_text())

def test_state_rejects_wrong_units_and_inconsistent_continuation():
    s={'h':[1,2,0,0],'position_mm':[1,2,3],'direction':[0,0,1],
       'q_over_p_per_MeV':1e-5,'time_Acts':10,'bound_parameters':[1,2,0,0,.01,10]}
    check_state(s,3,1e-5)
    bad=copy.deepcopy(s);bad['bound_parameters'][4]=1e-5
    with pytest.raises(ValueError,match='q/p unit'):check_state(bad,3,1e-5)
    bad=copy.deepcopy(s);bad['h'][0]=0
    with pytest.raises(ValueError,match='chart'):check_state(bad,3,1e-5)
    bad=copy.deepcopy(s);bad['direction'][2]=float('nan')
    with pytest.raises(ValueError):check_state(bad,3,1e-5)

def field_fixture():
    out={'conditions':{'map_key':P['map_key'],'cache_key':P['cache_key'],'wrapper_cache_identity':True,
      'min_mm':P['expected_zone']['min_mm'],'max_mm':P['expected_zone']['max_mm'],'zone_id':1,'scale':1.,'map_IOV':'actual','cache_IOV':'actual'}}
    def point(offset):
        b=[1e-5]*3 if offset<0 else [0,1,0]
        return {'position_mm':[47,74,-1762.3+offset],'zone_id':-1 if offset<0 else 1,'field_T':b,'fresh_field_T':b.copy()}
    out['probes']=[point(x) for x in P['boundary_probe_offsets_mm']]
    out['reverse_probes']=copy.deepcopy(out['probes'][::-1])
    out['domain_controls']=[point(x) for x in [-1,1,-1,-1,-1]]
    return out

def test_domain_contract_rejects_axis_swap_and_stale_cache():
    r=field_fixture();assert field_contract(r,P)['gate']=='PASS'
    bad=copy.deepcopy(r);bad['domain_controls'][2]['zone_id']=1
    with pytest.raises(ValueError,match='negative controls'):field_contract(bad,P)
    bad=copy.deepcopy(r);bad['probes'][4]['fresh_field_T'][1]=.5
    with pytest.raises(ValueError,match='cache'):field_contract(bad,P)

def reference_fixture():
    labels=[{'name':'nominal'}]+[{'direction':d,'name':n} for d in P['directions'] for n in ('central_plus','central_minus','full','half')]
    out=[]
    for step,e in zip(P['reference_rk4_dz_mm'],[4e-9,1e-9,0.]):
        point={'z_mm':0.,'q_over_p_per_MeV':1e-5,'h':[e,0,0,0]}
        out.append({'dz_mm':step,'samples':[{'label':label,'entry':copy.deepcopy(point),'interior':copy.deepcopy(point),
            'targets':[copy.deepcopy(point) for _ in range(3)],'inside_stages':1,'outside_stages':1} for label in labels]})
    return out

def test_reference_requires_precision_and_mechanism_rejects_sham_success():
    r=reference_fixture();check=reference_check(r,P);assert check['gate']=='NUMERICAL_REFERENCE_SUPPORTED'
    bad=copy.deepcopy(r);bad[-1]['samples'][0]['targets'][0]['h'][0]=1e-5
    assert reference_check(bad,P)['gate']=='UNKNOWN'
    m=[{'direct_max_error':1e-4,'direct_mixed_max_taylor':1e-6,'direct_cap_spread':1e-4,'all_criteria':True} for _ in range(3)]
    assert mechanism_decision(m,check,P)=='SUPPORTED_BUT_LIMITED'
    m[1]['all_criteria']=False
    assert mechanism_decision(m,check,P)=='NOT_SUPPORTED'
    assert mechanism_decision(m,{'gate':'UNKNOWN'},P)=='UNKNOWN'
