import numpy as np
import pytest
from alignment.wb99_direction_step_doubling import gate,verdict,vec,slope,slope_lipschitz,algebra
P={'calibration_factor':2.,'resolved_factor':10.,'uncertainty_factor':10.,'absolute_direction_floor':1e-12}
def test_zero_estimate_cannot_hide_resolved_error():
    assert gate(1e-7,0,1e-12,P)['false_negative']
    assert not gate(1e-12,0,1e-12,P)['resolved']
    assert not gate(1e-7,5e-8,1e-12,P)['false_negative']
    assert gate(1e-7,0,1e-8,P)['resolved'] is False
def test_identity_and_unknown_do_not_become_support():
    assert verdict(True,True,True,1)=='NOT_SUPPORTED'
    assert verdict(True,True,False,1)=='NOT_SUPPORTED'
    assert verdict(True,True,False,0)=='UNKNOWN'
    assert verdict(False,True,True,1)=='UNKNOWN'
    assert verdict(True,False,True,0)=='UNKNOWN'
    assert verdict(True,True,True,0)=='SUPPORTED_BUT_LIMITED'
def test_finite_direction_and_slope_budget():
    with pytest.raises(ValueError):vec([0,0,float('nan')])
    with pytest.raises(ValueError):slope([1,0,0])
    with pytest.raises(ValueError):gate(0,float('inf'),0,P)
    a=np.array([.1,.01,1.]);a/=np.linalg.norm(a);b=a+np.array([1e-8,-1e-8,-1e-8])
    assert np.max(np.abs(slope(a)-slope(b)))<=slope_lipschitz(a,b)*np.max(np.abs(a-b))
def test_zero_field_source_algebra_and_wrong_sign():
    z=[{'field_native':[0,0,0]}]*3
    p,u=algebra(np.zeros(3),np.array([0,0,1.]),10,.01,z)
    assert np.array_equal(p,[0,0,10]) and np.array_equal(u,[0,0,1])
    f=[{'field_native':[0,.000299792458,0]}]*3
    _,u=algebra(np.zeros(3),np.array([0,0,1.]),10,.01,f)
    _,v=algebra(np.zeros(3),np.array([0,0,1.]),10,-.01,f)
    assert u[0]<0<v[0]
