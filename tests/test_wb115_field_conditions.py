import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb115_field_conditions import compare,validate_rows

def case():
    return {'status':'READ','leaf_tag':'a','description':'<timeStamp>time</timeStamp>',
        'payload_specification':[{'name':'value','storage_type':'Float'}],'channel_names':{'1':''},
        'records':[{'channel':1,'since':0,'until':10,'payload':{'value':1.}},{'channel':1,'since':10,'until':20,'payload':{'value':1.}}]}

def test_leaf_identity_not_enough_or_required():
    a=case();b=copy.deepcopy(a);b['leaf_tag']='b'
    assert compare(a,b)['classification']=='EQUIVALENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT'
    b=copy.deepcopy(a);b['records'][0]['payload']['value']=2.
    assert compare(a,b)['classification']=='DIFFERENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT'

def test_sorting_and_channel_overlap():
    a=case();b=copy.deepcopy(a);b['records'].reverse()
    assert compare(a,b)['classification']=='EQUIVALENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT'
    a['records'].append({'channel':2,'since':0,'until':20,'payload':{'value':1.}})
    assert len(validate_rows(a))==3

@pytest.mark.parametrize('kind',['negative_scale','channel_name','iov_boundary','specification','description'])
def test_material_record_differences(kind):
    a=case();b=copy.deepcopy(a)
    if kind=='negative_scale':b['records'][0]['payload']['value']=-1.
    if kind=='channel_name':b['channel_names']['1']='Dipole_Scale'
    if kind=='iov_boundary':b['records'][0]['until']=9
    if kind=='specification':b['payload_specification'][0]['storage_type']='Double'
    if kind=='description':b['description']='<timeStamp>run-lumi</timeStamp>'
    assert compare(a,b)['classification']=='DIFFERENT_FIELD_FOLDER_RECORDS_IN_FROZEN_SNAPSHOT'

@pytest.mark.parametrize('kind',['nan','duplicate','overlap','empty','undefined','wrong_iov','unhandled','missing_field'])
def test_invalid_records_unknown(kind):
    a=case();b=copy.deepcopy(a)
    if kind=='nan':b['records'][0]['payload']['value']=float('nan')
    if kind=='duplicate':b['records'].append(copy.deepcopy(b['records'][0]))
    if kind=='overlap':b['records'][1]['since']=9
    if kind=='empty':b['records']=[]
    if kind=='undefined':b['status']='UNKNOWN'
    if kind=='wrong_iov':b['records'][0]['until']=0
    if kind=='unhandled':b['records'][0]['payload']['value']={'arbitrary':1}
    if kind=='missing_field':b['records'][0]['payload']={}
    assert compare(a,b)['classification']=='UNKNOWN'
