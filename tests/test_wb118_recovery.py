"""Producer/consumer integration and schema adapter controls, synthetic responses."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from wb118_recovery import recover
from wb118_sources import files
from test_wb118_response import synthetic


def inputs():
    r,c,f,h = synthetic(nonlinear=True); r['schema']='wb117_bounded_response_runtime_v1'
    stream=[]
    for row in r['rows']:
        stream.extend([{'record':'before_official','input':copy.deepcopy(row['input'])},
                       {'record':'after_official','row':copy.deepcopy(row)}])
    stream.append({'record':'terminal','status':'COMPLETED','official_calls':24})
    return r,c,f,h,stream


def test_actual_generated_schema_is_adapted_without_mutating_inputs_or_old_code():
    data=inputs();before=copy.deepcopy(data)
    source=files()['WB118Diagnostic/BoundedResponse.cxx']
    assert '"wb117_bounded_response_runtime_v1"' in source and 'if(count!=24)' in source
    s=recover(*data)
    assert data==before
    assert s['official_calls']==24 and s['decomposition_pattern']=='SIGNED_DECOMPOSITION_MEASURED'
    assert s['qualification']=='NOT_EVALUATED' and s['joint_new_calls']==0


@pytest.mark.parametrize('bad',['other_schema','already_normalized','wrong_matrix','wrong_control','missing_line','changed_output','wrong_terminal','sensor_drift'])
def test_only_exact_known_producer_can_recover_without_weakening_guards(bad):
    r,c,f,h,stream=inputs()
    if bad=='other_schema':r['schema']='unknown'
    elif bad=='already_normalized':r['schema']='wb118_bounded_response_runtime_v1'
    elif bad=='wrong_matrix':r['official_calls']=16
    elif bad=='wrong_control':r['control_used']['factors'][1]=.3
    elif bad=='missing_line':stream.pop(2)
    elif bad=='changed_output':stream[1]['row']['state']['h'][0][0]+=1.
    elif bad=='wrong_terminal':stream[-1]['official_calls']=16
    elif bad=='sensor_drift':r['sensors'][0]['transform'][0][3]+=.1
    with pytest.raises(ValueError):recover(r,c,f,h,stream)
