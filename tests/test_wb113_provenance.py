import copy
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb113_provenance import classify

def case():
    f={'input_xaod':'seen.root','ordinal':2268,'actual_run':100044,'actual_event':2268,
       'references':[{'station':s,'clusters':[s+100]} for s in range(4)]}
    link={'barcode':42,'event_collection':'a','event_index':0,'event_position':0,'valid':True,'weight_native':1.,
          'resolved_particle':{'pdg':-13,'momentum_native':[0.,0.,100000.],'parent_event_number':2268,'momentum_unit':'MEV'}}
    d={'identity':{k:f[k] for k in ('input_xaod','ordinal','actual_run','actual_event')},
       'clusters':[{'cluster_id':s+100,'station':s,'rdos':[{'rdo_id':s+200,'sdo_present':True,'deposits':[copy.deepcopy(link)]}]} for s in range(4)],
       'related_truth_particles':[{'barcode':42,'pdg':-13,'charge_e':1.,'momentum_native':[0.,0.,100000.]}],'persisted_tracks':[]}
    return f,d

def test_common_and_distinct_valid_particles():
    f,d=case();assert classify(f,d)['association']=='SUPPORTED_PERSISTED_COMMON_PARTICLE'
    d['clusters'][3]['rdos'][0]['deposits'][0]['barcode']=43
    assert classify(f,d)['association']=='CONTRADICTED_COMMON_PARTICLE'

@pytest.mark.parametrize('key,value',[('valid',False),('event_position',None),('event_position',4294967295),('event_index',4294967295),('barcode',0),('resolved_particle',None),('weight_native',0.)])
def test_incomplete_link_never_confirms(key,value):
    f,d=case();d['clusters'][0]['rdos'][0]['deposits'][0][key]=value
    assert classify(f,d)['association']=='UNKNOWN_OR_AMBIGUOUS'

@pytest.mark.parametrize('field,value',[('event_index',1),('event_collection','b'),('event_position',1),('barcode',43)])
def test_same_cluster_mixed_identity(field,value):
    f,d=case();x=copy.deepcopy(d['clusters'][0]['rdos'][0]['deposits'][0]);x[field]=value
    d['clusters'][0]['rdos'][0]['deposits'].append(x)
    assert classify(f,d)['association']=='UNKNOWN_OR_AMBIGUOUS'

def test_duplicates_not_confirmed_and_missing_sdo():
    f,d=case();r=d['clusters'][0]['rdos'][0];r['deposits']*=2
    assert classify(f,d)['association']=='UNKNOWN_OR_AMBIGUOUS'
    f,d=case();d['clusters'][0]['rdos'][0]['sdo_present']=False
    assert classify(f,d)['association']=='UNKNOWN_OR_AMBIGUOUS'

def test_ambiguous_xaod_join_not_charge_oracle():
    f,d=case();d['related_truth_particles']*=2
    s=classify(f,d);assert s['xaod_barcode_join_unique'][42] is False
    assert s['qualification']=='NOT_EVALUATED'

def test_inconsistent_resolved_particle_refused():
    f,d=case();d['clusters'][3]['rdos'][0]['deposits'][0]['resolved_particle']['pdg']=13
    with pytest.raises(ValueError,match='inconsistent resolved particle'):classify(f,d)

@pytest.mark.parametrize('field,value',[('input_xaod','other.root'),('ordinal',2269),('actual_run',100043),('actual_event',2269)])
def test_wrong_event_refused(field,value):
    f,d=case();d['identity'][field]=value
    with pytest.raises(ValueError):classify(f,d)

@pytest.mark.parametrize('kind',['cluster','station','coverage','duplicate','weight_nan','truth_nan'])
def test_corrupt_records_refused(kind):
    f,d=case()
    if kind=='cluster':d['clusters'][0]['cluster_id']=999
    if kind=='station':d['clusters'][0]['station']=3
    if kind=='coverage':d['clusters'].pop()
    if kind=='duplicate':d['clusters'][0]=d['clusters'][1]
    if kind=='weight_nan':d['clusters'][0]['rdos'][0]['deposits'][0]['weight_native']=float('nan')
    if kind=='truth_nan':d['related_truth_particles'][0]['charge_e']=float('nan')
    with pytest.raises(ValueError):classify(f,d)
