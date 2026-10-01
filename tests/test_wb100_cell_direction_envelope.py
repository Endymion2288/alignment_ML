import pytest
from alignment.wb100_cell_direction_envelope import decision,information,audit_row

P={'informativeness_ratio_max':.5,'informativeness_median_ratio_max':.1,'informativeness_fraction_required':.9,
   'uncertainty_factor':10,'resolved_factor':10}

def test_coverage_does_not_override_information_failure():
    assert decision(True,0,0,False)=='NOT_SUPPORTED'
    assert decision(True,0,0,True)=='SUPPORTED_BUT_LIMITED'
    assert decision(True,1,0,True)=='UNKNOWN'
    assert decision(False,0,1,True)=='NOT_SUPPORTED'

def test_information_population_and_exact_boundary():
    assert information([.1]*9+[.5],P)['gate']=='PASS'
    assert information([.11]*10,P)['gate']=='FAIL'
    assert information([.01]*8+[.6]*2,P)['gate']=='FAIL'
    assert information([],P)['gate']=='UNKNOWN'
    assert information([None],P)['gate']=='UNKNOWN'

def test_strict_resolved_and_uncertainty_budget_audit():
    r={'h_mm':1,'envelope':{'E':1e-10,'cuts_mm':[0,1]},
       'evaluation':{'gate':'PASS','defect':1e-11,'uncertainty':1e-12,'budget':1.1e-10,'resolved':False,'false_negative':False,'slope':{'gate':'UNKNOWN'}}}
    audit_row(r,P)
    r['evaluation']['resolved']=True
    with pytest.raises(ValueError,match='direction decision'):audit_row(r,P)
