import json
import pytest
from alignment.wb97_rkn_local_defect import aggregate
from alignment.wb97_rkn_local_defect import decision

P={'crossing_significant_fraction_required':.9,'crossing_rate_enrichment_required':2.}

def test_full_coverage_and_controls_are_mandatory():
    counts={'domain':{'count':10,'significant':10},'interior':{'count':100,'significant':0}}
    assert decision(True,0,0,True,counts,P)=='SUPPORTED_BUT_LIMITED'
    for complete,closure,reference,control in ((False,0,0,True),(True,1,0,True),(True,0,1,True),(True,0,0,False)):
        assert decision(complete,closure,reference,control,counts,P)=='UNKNOWN'

def test_concentration_and_negative_control():
    assert decision(True,0,0,True,{'mesh':{'count':10,'significant':1}},P)=='UNKNOWN'
    assert decision(True,0,0,True,{'mesh':{'count':10,'significant':0},'interior':{'count':100,'significant':0}},P)=='NOT_SUPPORTED'
    assert decision(True,0,0,True,{'mesh':{'count':10,'significant':8},'interior':{'count':10,'significant':2}},P)=='NOT_SUPPORTED'
    assert decision(True,0,0,True,{'mesh':{'count':1000,'significant':9},'interior':{'count':10,'significant':1}},P)=='NOT_SUPPORTED'

def test_streaming_missing_terminal_and_identity_fail_closed(tmp_path):
    p={**P,'expected_traces':1,'field_probe_T_tolerance':1e-12}
    source={'tolerance':1e-4,'cap_m':1.,'path':'entry','station':0,'sample':0,'steps':1,'last_position_mm':[0,0,1]}
    controls={'record':'controls','controls':{'gate':'PASS'},'probe_max_T':0}
    row={'record':'step','trace':0,'index':0,**{k:source[k] for k in ('tolerance','cap_m','path','station','sample')},
         'saved_step':{'position_mm':[0,0,1]},'closure_gate':'UNKNOWN','reference_gate':'UNKNOWN','unknown_reason':'missing state'}
    file=tmp_path/'steps.ndjson';file.write_text(json.dumps(controls)+'\n'+json.dumps(row)+'\n')
    s=aggregate(file,{'traces':[source],'steps':1},p)
    assert s['hypothesis']=='UNKNOWN' and s['coverage_gate']=='FAIL' and s['closure_failures']==1
    row['station']=1;file.write_text(json.dumps(controls)+'\n'+json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='identity'):aggregate(file,{'traces':[source],'steps':1},p)
