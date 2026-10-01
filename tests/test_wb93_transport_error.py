import json
from pathlib import Path
import numpy as np
import pytest
from alignment.wb93_transport_error import errors,windows,analytic_control

P=json.loads((Path(__file__).resolve().parents[1]/'configs/research_review/wp93_transport_error_budget.json').read_text())

def test_affine_oracle_and_negative_controls():
    f={'references':[{'fixed_z_state':[[47],[74],[0],[.002]],'q_over_p_per_MeV':1e-5,'z_state_mm':z} for z in (-1860.,47.,1237.,2427.)]}
    assert analytic_control(f,P)['gate']=='PASS'

def test_directional_contraction_detects_quadratic_and_wrong_sign():
    def h(d):return np.array([d+d*d,0,0,0])
    s={'central_plus':h(.5),'central_minus':h(-.5),'full':h(.25),'half':h(.125)}
    e=errors(s,np.zeros(4),np.ones(4))
    assert e['contraction']==.25
    assert errors(s,np.zeros(4),np.ones(4),-np.array(e['effect']))['full_error']>e['full_error']
    s['full'][0]=float('nan')
    with pytest.raises(ValueError):errors(s,np.zeros(4),np.ones(4))

def test_windows_require_adjacent_pairs_and_distinguish_floor():
    rows=[{'multiplier':x,'full_error':1.,'half_error':y} for x,y in ((1,.25),(2,1.),(4,.25))]
    assert not windows(rows,P)['found']
    rows[1]['half_error']=.25
    assert windows(rows,P)['found']
    for r in rows:r['full_error']=r['half_error']=1e-12
    assert windows(rows,P)['roundoff_limited_count']==3
