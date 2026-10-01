import copy,json
from pathlib import Path
import numpy as np
import pytest
from alignment.wb96_acts_tolerance import effect_matrix,remainders,metrics,decide
P=json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp96_acts_tolerance_contract.json').read_text())

def test_coordinate_mixed_effect_and_curvature_are_separate():
    a=np.zeros((25,4))
    for k in range(6):
        for j,d in enumerate((.5,-.5,.25,.125)):a[1+4*k+j,0]=(1 if k<5 else 99)*d+d*d
    assert effect_matrix(a)[5,0]==.25
    assert remainders(a)[5,0,0]>20
    assert metrics(np.repeat(a[None],4,axis=0),a,np.array(P['output_scales']))['taylor_excess']==0
    b=a.copy();b[1,0]+=.01
    assert metrics(np.repeat(b[None],4,axis=0),a,np.array(P['output_scales']))['effect']>0
    b[1,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):effect_matrix(b)

def cells():
    return [{'tolerance':t,'station':s,'reference_mode':m,'endpoint':1 if t==1e-4 else .1,
      'cap_spread':1 if t==1e-4 else .1,'effect':1 if t==1e-4 else .1}
      for t in P['step_tolerances'] for s in (1,2,3) for m in P['reference_modes']]

def test_complete_matrix_required_and_path_changes_limit_attribution():
    c=cells();assert decide(c,P,1e-12,1e-13,[],[],'PASS')[0]=='SUPPORTED_BUT_LIMITED'
    assert decide(c,P,1e-12,1e-13,[{}],[],'PASS')[0]=='JOINT_CONTROL_NAVIGATION_SUPPORTED_ONLY'
    assert decide(c[:-1],P,1e-12,1e-13,[],[],'PASS')[0]=='UNKNOWN'
    assert decide(c,P,1e-12,1e-13,[],[{}],'PASS')[0]=='UNKNOWN'
    assert decide(c,P,1e-12,1e-13,[],[],'FAIL')[0]=='UNKNOWN'
    bad=copy.deepcopy(c);bad[-1]['effect']=.6
    assert decide(bad,P,1e-12,1e-13,[],[],'PASS')[0]=='NOT_SUPPORTED'

def test_both_levels_all_stations_and_field_sensitivity_required():
    c=cells();c[6]['endpoint']=.9
    assert decide(c,P,1e-12,1e-13,[],[],'PASS')[0]=='NOT_SUPPORTED'
    c=cells();c[7]['endpoint']=.9
    assert decide(c,P,1e-12,1e-13,[],[],'PASS')[0]=='NOT_SUPPORTED'
    assert decide(cells(),P,.1,.1,[],[],'PASS')[0]=='NOT_SUPPORTED'
