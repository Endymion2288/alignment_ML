"""Saved seen probe and synthetic topology receipts; no Athena or ROOT."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wb110_contract import probe,BASE
from audit_wb110_topology import validate,metadata
from wb110_sources import files,athena_source

def sample():
    p=probe();f=json.loads((BASE/'fixture.json').read_text())
    owner={'name':p['last_nonnull_volume_name'],'geometry_id':12345,
      'bounds_type':1,'bounds_values':[250.,250.,500.],'transform':np.eye(4).tolist()}
    owner['transform'][2][3]=1837.4
    world=copy.deepcopy(p['world']);world['boundaries']=[{k:v for k,v in b.items() if k!='current_surface_pointer_match'} for b in p['world_boundary_surfaces']]
    child=copy.deepcopy(owner);child['boundaries']=[copy.deepcopy(p['current_surface'])]
    target=copy.deepcopy(p['target_volume']);target['boundaries']=[]
    result={'probe_used':copy.deepcopy(p),'identity':{k:f[k] for k in ('actual_run','actual_event','input_xaod','ordinal')},
      'new_propagation_calls':0,'algorithm_field_queries':0,'world':copy.deepcopy(p['world']),
      'world_inside_probe':True,'volumes':[world,child,target],
      'edges':[{'parent':world['geometry_id'],'child':child['geometry_id'],'kind':'confined'},
         {'parent':world['geometry_id'],'child':target['geometry_id'],'kind':'confined'}],
      'matches':[{'owner_geometry_id':owner['geometry_id'],'surface':copy.deepcopy(p['current_surface']),
        'forward_attachment':None,'reverse_attachment':copy.deepcopy(owner)}],'attachment_queries':2}
    return result,p,f

def test_internal_child_boundary_chain_supported_without_physics_claim():
    r,p,f=sample();s=validate(r,p,f)
    assert s['classification']=='INTERNAL_CHILD_BOUNDARY_NULL_ATTACHMENT'
    assert s['owner']['name']=='ShortDipole_2'
    assert s['face_axis']==0 and s['face_sign']==1
    assert s['world_inside_probe'] is True
    assert s['new_propagation_calls']==s['algorithm_field_queries']==0
    assert s['deeper_acceptance_or_seed_mechanism']=='UNKNOWN'

@pytest.mark.parametrize('mutation',['header','source','probe','world','surface','volume_id','edge','cycle',
  'nonfinite','attachment_metadata','query_count','propagation','field','inside','target'])
def test_corrupt_or_changed_identity_rejected(mutation):
    r,p,f=sample()
    if mutation=='header':r['identity']['actual_event']+=1
    elif mutation=='source':r['identity']['input_xaod']='other.root'
    elif mutation=='probe':r['probe_used']['position'][0][0]+=1.
    elif mutation=='world':r['world']['bounds_values'][0]+=1.
    elif mutation=='surface':r['matches'][0]['surface']['frame'][0][3]+=1.
    elif mutation=='volume_id':r['volumes'][1]['geometry_id']=r['volumes'][0]['geometry_id']
    elif mutation=='edge':r['edges'][0]['parent']=999999
    elif mutation=='cycle':r['edges'].append({'parent':r['volumes'][1]['geometry_id'],'child':r['world']['geometry_id'],'kind':'confined'})
    elif mutation=='nonfinite':r['volumes'][1]['transform'][0][0]=float('nan')
    elif mutation=='attachment_metadata':r['matches'][0]['reverse_attachment']['name']='changed'
    elif mutation=='query_count':r['attachment_queries']=0
    elif mutation=='propagation':r['new_propagation_calls']=1
    elif mutation=='field':r['algorithm_field_queries']=1
    elif mutation=='inside':r['world_inside_probe']=False
    elif mutation=='target':r['volumes'][2]['bounds_values'][0]+=1.
    with pytest.raises(ValueError):validate(r,p,f)

@pytest.mark.parametrize('alternative',['forward_nonnull','reverse_other','multiple_owner','no_match'])
def test_plausible_alternative_topology_is_reported_without_forcing_hypothesis(alternative):
    r,p,f=sample()
    if alternative=='forward_nonnull':r['matches'][0]['forward_attachment']=copy.deepcopy(p['world'])
    elif alternative=='reverse_other':r['matches'][0]['reverse_attachment']=copy.deepcopy(p['world'])
    elif alternative=='multiple_owner':
        n=copy.deepcopy(r['volumes'][1]);n['geometry_id']=12346;n['name']='synthetic_alias'
        r['volumes'].append(n);r['edges'].append({'parent':r['world']['geometry_id'],'child':n['geometry_id'],'kind':'dense'})
        m=copy.deepcopy(r['matches'][0]);m['owner_geometry_id']=n['geometry_id'];m['reverse_attachment']=metadata(n)
        r['matches'].append(m);r['attachment_queries']=4
    elif alternative=='no_match':
        r['volumes'][1]['boundaries']=[];r['matches']=[];r['attachment_queries']=0
    assert validate(r,p,f)['classification']=='OTHER_OR_UNRESOLVED_TOPOLOGY'

def test_runner_and_source_do_not_instantiate_propagation_or_field_path():
    source=files()['WB110Diagnostic/BoundaryTopology.cxx'];runner=athena_source()
    assert '->propagate(' not in source and '.step(' not in source and 'getField(' not in source
    assert source.count('boundary->attachedVolume(')==2
    assert 'CompFactory.FaserActsExtrapolationTool(' not in runner
    assert 'acc.merge(MagneticFieldSvcCfg(flags))' not in runner
    assert "flags.Exec.SkipEvents = fixture['ordinal']; flags.Exec.MaxEvents = 1" in runner
    assert 'attachVolume(' not in source and 'glueTrackingVolume(' not in source
    compile(runner,'wb110_athena.py','exec')
