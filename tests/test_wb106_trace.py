import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb106_trace import validate_trace
from wb106_contract import generated_source


def sample():
    f={'input_xaod':'seen.root','ordinal':2268,'actual_run':100044,'actual_event':2268,
       'protocol':{'seed_steps':[.01,.01,1e-5,1e-5,1e-8],'geometry_steps':[.01,.01,.01,1e-6,1e-6,1e-6]},
       'references':[{'fixed_z_state':[[1.],[2.],[0.],[0.]],'q_over_p_per_MeV':1e-5,'z_state_mm':float(i*100),'clusters':[i+10]} for i in range(4)]}
    rows=[{'record':'event',**{k:f[k] for k in ('input_xaod','ordinal','actual_run','actual_event')}}]
    rows.append({'record':'sensors','station':0,'sensors':[{'strip':10,'geometry_id':100}]})
    seed=np.array([1.,2.,0.,0.,1e-5]);frame=np.eye(4)
    calls=[(-1,'nominal',seed),(-1,'repeat',seed)]
    for axis,h in enumerate(f['protocol']['seed_steps']):
        for label,fac in (('plus',1),('minus',-1),('half_plus',.5),('half_minus',-.5)):
            s=seed.copy();s[axis]+=fac*h;calls.append((axis,'xi_'+label,s))
    delta=np.asarray(f['protocol']['seed_steps'])*np.array([1,-1,1,-1,1])*.25
    calls.extend([(-1,'xi_mixed_full',seed+delta),(-1,'xi_mixed_half',seed+.5*delta)])
    for cid,(axis,label,s) in enumerate(calls):
        rows += [{'record':'input','call_id':cid,'station':0,'axis':axis,'label':label,
                  'seed':s[:,None].tolist(),'seed_z_mm':0.,'frame':frame.tolist()},
                 {'record':'boundary','call_id':cid}]
    rows.append({'record':'sensors','station':1,'sensors':[{'strip':11,'geometry_id':101}]})
    frame[2,3]=100.
    rows += [{'record':'input','call_id':24,'station':1,'axis':-1,'label':'nominal',
              'seed':seed[:,None].tolist(),'seed_z_mm':0.,'frame':frame.tolist()},
             {'record':'before_official','call_id':24,'start_position':[[1.],[2.],[0.]],
              'start_direction':[[0.],[0.],[1.]],'distance':100.,'forward':True,
              'target_geometry_id':0,'target_frame':frame.tolist()},
             {'record':'after_official','call_id':24,'has_value':False},
             {'record':'terminal','calls':25,'status':'FAIL_CLOSED'}]
    return rows,f


def test_partial_failure_validates_without_future_sensor_receipts():
    rows,f=sample()
    assert len(validate_trace(rows,f))==25


@pytest.mark.parametrize('mutation',['source','header','frame','qop','missing_call','geometry_id','swapped_station'])
def test_invalid_identity_or_trace_rejected(mutation):
    rows,f=sample();rows=copy.deepcopy(rows)
    if mutation=='source': rows[0]['input_xaod']='other.root'
    elif mutation=='header': rows[0]['actual_event']+=1
    elif mutation=='frame': rows[-4]['frame'][2][3]+=1.
    elif mutation=='qop': rows[-4]['seed'][4][0]*=1000.
    elif mutation=='missing_call': rows.pop(-3)
    elif mutation=='geometry_id': rows[-3]['target_geometry_id']=42
    elif mutation=='swapped_station': rows[-4]['station']=2
    with pytest.raises(ValueError): validate_trace(rows,f)


def test_source_keeps_official_propagation_and_steps():
    s=generated_source()
    assert s.count('m_tool->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward)')==1
    assert 'stepTolerance' not in s
    assert 'O_EXCL' in s and 'std::endl' in s
    assert 'SurfaceError' not in s
