import numpy as np
import pytest
from alignment.wb98_defect_transport import decision,mixed_checks,norm

def test_independence_and_unknown_gate():
    assert decision(True,True,['PASS']*156,['PASS']*36)=='SUPPORTED_BUT_LIMITED'
    assert decision(True,True,['FAIL'],['PASS'])=='NOT_SUPPORTED'
    assert decision(True,True,['PASS'],['UNKNOWN'])=='UNKNOWN'
    assert decision(False,True,['PASS'],['PASS'])=='UNKNOWN'
    assert decision(True,False,['PASS'],['PASS'])=='UNKNOWN'
    assert decision(True,True,[],[])=='UNKNOWN'

def test_mixed_vector_budget_and_missing_identity():
    rows=[{'path':'direct','tolerance':1e-4,'cap_m':1.,'station':1,'sample':i,'closure_gate':'PASS',
           'closure_budget_scaled':1e-8,'endpoint_ladders':[{'closure_residual':[value,0,0,0]}]} for i,value in ((0,0),(23,1e-8),(24,1e-8))]
    check=mixed_checks(rows)[0]
    assert check['gate']=='PASS'
    assert check['checks'][-1]['budget_scaled']==4e-8
    assert mixed_checks(rows[:2])[0]['gate']=='UNKNOWN'
    rows[-1]['endpoint_ladders'][0]['closure_residual']=[1e-6,0,0,0]
    assert mixed_checks(rows)[0]['gate']=='FAIL'

def test_units_nan_and_position_slope_transport():
    assert norm([0,0,1e-8,0])==1e-5
    with pytest.raises(ValueError):norm([0,0,float('nan'),0])
    a=np.eye(4);a[0,2]=100;a[1,3]=100
    position=np.array([1e-6,0,0,0]);slope=np.array([0,0,0,1e-9])
    ep=np.zeros(4);eu=np.zeros(4)
    for i in range(10):ep=a@ep+position;eu=a@eu+slope
    assert np.allclose(ep+eu,[1e-5,4.5e-6,0,1e-8],rtol=0,atol=1e-18)
