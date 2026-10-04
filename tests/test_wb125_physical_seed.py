"""Fail-closed seed provenance and effect classification, with analytic controls."""
import copy
import math
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wb125_audit import choose_state,audit_event,expected_cells

def state(index,z,charge=1.):
    frame=np.eye(4);frame[2,3]=z
    return {'persistent_index':index,'global_position_native':[0.,0.,z],
        'global_momentum_native':[0.,0.,100000.],'charge_e':charge,
        'native_parameters':[0.,0.,0.,0.,charge/100000.],
        'surface_type':4,'surface_transform':frame.tolist(),
        'position_unit':'mm','momentum_unit':'MeV','qop_unit':'MeV^-1'}

def fixture(index=12):
    return {'input_xaod':'synthetic_only','ordinal':0,'actual_run':1,'actual_event':2,'index':index,
        'references':[{'station':s,'clusters':[s+1],'z_state_mm':10.+s,'fixed_z_state':[[0.],[0.],[0.],[0.]]} for s in range(4)]}

def provenance(f,states=None):
    identity={k:f[k] for k in ('input_xaod','ordinal','actual_run','actual_event')}
    link={'event_collection':'a','event_index':0,'event_position':0,'barcode':10001,'valid':True,
        'weight_native':1.,'resolved_particle':{'pdg':-13,'momentum_native':[0.,0.,100000.],'parent_event_number':1,'momentum_unit':'MEV'}}
    return {'identity':identity,'clusters':[{'station':s,'cluster_id':s+1,'rdos':[{'rdo_id':s+1,'sdo_present':True,'deposits':[copy.deepcopy(link)]}]} for s in range(4)],
        'related_truth_particles':[{'barcode':10001,'charge_e':1.,'momentum_native':[0.,0.,100000.]}],
        'persisted_tracks':[{'key':'CKFTrackCollection','container_size':1,'matching_tracks':[{'track_index':0,
            'parameters':states or [state(0,10.)],'cluster_membership':[{'cluster_id':s+1,'prd_link_resolved':True} for s in range(4)]}]}]}

def response(f,p):
    selected=choose_state(p,f);seed=[[0.],[0.],[0.],[0.],[1e-5]]
    zero={'h':[[0.],[0.],[0.],[0.]],'qop_per_MeV':1e-5,'on_surface':True,'covariance_present':False,'local':[[0.],[0.],[0.]]}
    bridge=copy.deepcopy(zero);bridge['status']='SUCCESS'
    rows=[];call=1
    for arm,kind,s,m in expected_cells(f['index']):
        failed=arm=='D' and s==3
        row={'arm':arm,'kind':kind,'station':s,'qop_multiplier':m,'seed':copy.deepcopy(seed),
            'status':'FAIL_OFFICIAL_NULL' if failed else 'SUCCESS','state':None if failed else copy.deepcopy(zero),'propagation_call':None}
        if kind=='fd':row['seed'][4][0]+=m*1e-8
        if s>0:call+=1;row['propagation_call']=call
        rows.append(row)
    return {'identity':p['identity'],'index':f['index'],'selection':{'status':'KNOWN',**selected},
        'bridge':bridge,'seeds':{'D':copy.deepcopy(seed),'M':copy.deepcopy(seed),'P':copy.deepcopy(seed)},
        'responses':rows,'propagation_calls':call}

def test_nearest_state_tie_uses_persistent_order():
    f=fixture();p=provenance(f,[state(0,9.875),state(1,10.125)])
    assert choose_state(p,f)['selected_index']==0

def test_selection_does_not_fall_back_after_nearest_is_invalid():
    f=fixture();states=[state(0,10.),state(1,10.5)];states[0]['charge_e']=0
    with pytest.raises(ValueError,match='charge'):choose_state(provenance(f,states),f)

def test_negative_charge_preserved_not_filename_inferred():
    f=fixture();assert choose_state(provenance(f,[state(0,10.,-1.)]),f)['source_qop_per_MeV']==-1e-5

@pytest.mark.parametrize('mutation',[
    'units','qop','surface_type','reflection','off_surface','angle','nonfinite','zero_pz','multiple_tracks','wrong_collection','missing_surface','wrong_order'])
def test_source_negative_controls(mutation):
    f=fixture();p=provenance(f);s=p['persisted_tracks'][0]['matching_tracks'][0]['parameters'][0]
    if mutation=='units':s['momentum_unit']='GeV'
    elif mutation=='qop':s['native_parameters'][4]*=-1
    elif mutation=='surface_type':s['surface_type']=6
    elif mutation=='reflection':s['surface_transform'][0][0]=-1
    elif mutation=='off_surface':s['surface_transform'][2][3]=20
    elif mutation=='angle':s['native_parameters'][2]=.1
    elif mutation=='nonfinite':s['global_momentum_native'][0]=math.nan
    elif mutation=='zero_pz':s['global_momentum_native'][2]=0
    elif mutation=='multiple_tracks':p['persisted_tracks'][0]['container_size']=2
    elif mutation=='wrong_collection':p['persisted_tracks'][0]['key']='CKFTrackCollectionBackward'
    elif mutation=='missing_surface':del s['surface_transform']
    elif mutation=='wrong_order':s['persistent_index']=1
    with pytest.raises((ValueError,KeyError)):choose_state(p,f)

def test_complete_provenance_supports_only_development_effect():
    f=fixture();p=provenance(f);r=response(f,p);s=audit_event(f,p,r)
    assert s['integrity']=='PASS' and s['physical_fixture_supported']
    assert s['hypothesis']=='SUPPORTED_BUT_LIMITED_SEED_INPUT_EFFECT'
    assert s['qualification']=='NOT_EVALUATED'
    assert all(q['exact'] for q in s['repeat_checks'])

def test_missing_sdo_does_not_block_conditional_intervention_or_become_pass():
    f=fixture();p=provenance(f);p['clusters'][0]['rdos'][0]['sdo_present']=False
    s=audit_event(f,p,response(f,p))
    assert s['hypothesis']=='SUPPORTED_CONDITIONAL_INPUT_EFFECT'
    assert s['conditional'] and not s['physical_fixture_supported']
    assert s['association']=='UNKNOWN_OR_AMBIGUOUS'

def test_mixed_particle_is_not_single_track_association():
    f=fixture();p=provenance(f);link=copy.deepcopy(p['clusters'][0]['rdos'][0]['deposits'][0]);link['barcode']=20002
    p['clusters'][0]['rdos'][0]['deposits'].append(link)
    s=audit_event(f,p,response(f,p))
    assert not s['physical_fixture_supported'] and s['conditional']

def test_native_charge_conflict_is_reported_without_truth_repair():
    f=fixture();p=provenance(f);p['related_truth_particles'][0]['charge_e']=-1
    s=audit_event(f,p,response(f,p))
    assert s['source_charge_consistency']=='CONTRADICTED_RELATED_TRUTH_CHARGE'
    assert not s['physical_fixture_supported'] and s['conditional']

def test_no_material_bridge_cannot_silently_change_momentum():
    f=fixture();p=provenance(f);r=response(f,p);r['bridge']['qop_per_MeV']*=2
    with pytest.raises(ValueError,match='conservation'):audit_event(f,p,r)

@pytest.mark.parametrize('mutation',['header','duplicate_cell','missing_cell','modified_repeat','qop_fd_step','call_cap','off_plane','covariance','mixed_changes_xy'])
def test_response_negative_controls(mutation):
    f=fixture();p=provenance(f);r=response(f,p)
    if mutation=='header':r['identity']=dict(r['identity'],actual_event=3)
    elif mutation=='duplicate_cell':r['responses'].append(copy.deepcopy(r['responses'][0]))
    elif mutation=='missing_cell':r['responses'].pop()
    elif mutation=='modified_repeat':next(x for x in r['responses'] if x['kind']=='repeat' and x['state'])['state']['h'][0][0]=1
    elif mutation=='qop_fd_step':next(x for x in r['responses'] if x['kind']=='fd')['seed'][4][0]+=1e-8
    elif mutation=='call_cap':r['propagation_calls']=25
    elif mutation=='off_plane':r['responses'][0]['state']['on_surface']=False
    elif mutation=='covariance':r['responses'][0]['state']['covariance_present']=True
    elif mutation=='mixed_changes_xy':r['seeds']['M'][0][0]=.1
    with pytest.raises(ValueError):audit_event(f,p,r)
