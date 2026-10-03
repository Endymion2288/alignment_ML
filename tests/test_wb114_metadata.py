import copy
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from wb114_metadata import control_from_metadata,TRACK_CLASS,TRACK_KEYS

def case():
    source={'path':'seen.root','bytes':1000,'mtime_ns':123,'sha256':'a'*64}
    b={'source':source,'tree':'CollectionTree','entries':500000,'root_uuid':'uuid',
       'event_entries_decoded':0,'branches':[{'name':k,'class':TRACK_CLASS,'title':k} for k in TRACK_KEYS]}
    md={'seen.root':{'/TagInfo':{'AtlasRelease':'Athena-24.0.41','GeoFaser':'FASERNU-04','IOVDbGlobalTag':'OFLCOND-FASER-05'}}}
    return b,copy.deepcopy(b),md

def test_missing_itemlist_supported_without_fabrication():
    b,e,m=case();c=control_from_metadata(b,e,m)
    assert c['track_keys']==sorted(TRACK_KEYS)
    assert c['event_stream_itemlist_present'] is False
    assert 'eventdata_items' not in c
    assert c['source_tag_info']['IOVDbGlobalTag']=='OFLCOND-FASER-05'
    assert c['executor_tag_info']['IOVDbGlobalTag']=='OFLCOND-FASER-06'
    assert c['physical_conditions_compatibility']=='UNKNOWN'

def test_branch_order_not_identity():
    b,e,m=case();b['branches'].reverse();assert control_from_metadata(b,e,m)['track_keys']==sorted(TRACK_KEYS)

def test_optional_real_itemlist_crosscheck():
    b,e,m=case();items=[['TrackCollection',k] for k in TRACK_KEYS]
    assert control_from_metadata(b,e,m,items)['event_stream_itemlist_present']
    with pytest.raises(ValueError):control_from_metadata(b,e,m,items[:-1])
    with pytest.raises(ValueError):control_from_metadata(b,e,m,items+[items[0]])

@pytest.mark.parametrize('kind',['source','entries','uuid','tree','decoded','duplicate','class','missing','extra','title','tag','metadata_source','unknown_extra_branch'])
def test_drift_refused(kind):
    b,e,m=case()
    if kind=='source':b['source']['sha256']='b'*64
    if kind=='entries':b['entries']=499999
    if kind=='uuid':b['root_uuid']='other'
    if kind=='tree':b['tree']='nt'
    if kind=='decoded':b['event_entries_decoded']=1
    if kind=='duplicate':b['branches'].append(copy.deepcopy(b['branches'][0]))
    if kind=='class':b['branches'][0]['class']='Trk::TrackCollection_tlp7'
    if kind=='missing':b['branches'].pop()
    if kind=='extra':b['branches'].append({'name':'OtherTracks','class':TRACK_CLASS,'title':'OtherTracks'})
    if kind=='title':b['branches'][0]['title']='wrong'
    if kind=='tag':m['seen.root']['/TagInfo']['IOVDbGlobalTag']='OFLCOND-FASER-06'
    if kind=='metadata_source':m['other.root']=m.pop('seen.root')
    if kind=='unknown_extra_branch':b['branches'].append({'name':'other','class':'Unknown','title':'other'})
    with pytest.raises(ValueError):control_from_metadata(b,e,m)

def test_unknown_class_not_inferred_from_tracklike_name():
    b,e,m=case();extra={'name':'ImaginaryTracks','class':'Unknown','title':'ImaginaryTracks'}
    b['branches'].append(extra);e['branches'].append(extra)
    assert control_from_metadata(b,e,m)['track_keys']==sorted(TRACK_KEYS)

def test_real_saved_file_metadata_without_event_values():
    base=ROOT/'outputs/mc24_four_station_wb113_persisted_provenance_v1'
    b=json.loads((base/'root_branch_metadata.json').read_text());m=json.loads((base/'source_metadata.json').read_text())
    c=control_from_metadata(b,b,m)
    assert c['track_keys']==sorted(TRACK_KEYS) and c['event_stream_itemlist_present'] is False
