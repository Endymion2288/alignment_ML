"""Seen saved trace plus synthetic abort receipts; no ROOT or propagation."""
import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb109_abort import validate,boundary_geometry
from wb109_sources import files
from wb108_contract import OUT as BASE
from wb107_contract import BASE as BASE106

def cube():
    boundaries=[]
    for axis in range(3):
        for sign in (-1,1):
            frame=np.eye(4);frame[axis,3]=sign*[250.,250.,3000.][axis]
            if axis==0:frame[:3,:3]=[[0,0,1],[0,1,0],[-1,0,0]]
            elif axis==1:frame[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
            boundaries.append({'geometry_id':len(boundaries)+100,'type':0,'frame':frame.tolist(),
              'bounds_type':1,'bounds_values':[3000.,250.], 'current_surface_pointer_match':False})
    return boundaries

def sample():
    load=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
    baseline=load(BASE/'event/acts.json.calls.ndjson');baseline106=load(BASE106/'event/acts.json.calls.ndjson')
    fixture=json.loads((BASE/'fixture.json').read_text())
    rows=[]
    for r in baseline:
        rows.append(copy.deepcopy(r))
        if r['record']!='before_diagnostic':continue
        cid=r['call_id'];seq=[x for x in baseline if x.get('call_id')==cid]
        start=next(x for x in seq if x['record']=='before_official')
        bind=next(x for x in seq if x['record']=='bound_before')
        options={'maxSteps':10000,'maxStepSize_mm':10000.,'surfaceTolerance_mm':1e-4,'stepTolerance':1e-4,
          'pathLimit':float(np.finfo(float).max),'loopProtection':False,'forward':True}
        initial={'record':'end_world_check','call_id':cid,'stage':'prePropagation','step_index':0,
          'position':start['start_position'],'direction':start['start_direction'],'path_mm':0.,
          'current_volume_null_before':False,'current_volume_null_after':False,'current_volume_name':'seen-test-volume',
          'end_of_world_before':False,'returned':False,'target_reached_before':False,'target_reached_after':False,
          'navigation_break_before':False,'navigation_break_after':False,'current_surface_is_target':False,
          'navigator_target_matches_requested':True,'options':options}
        final=copy.deepcopy(initial);final.update(stage='postStep',step_index=40,position=bind['position'],direction=bind['direction'],path_mm=bind['path_mm'])
        if cid==124:
            final.update(current_volume_null_before=True,current_volume_null_after=True,current_volume_name=None,
              end_of_world_before=True,returned=True,target_reached_after=True,navigation_break_before=True,navigation_break_after=True)
            boundaries=cube();current={k:v for k,v in boundaries[1].items() if k!='current_surface_pointer_match'}
            final.update(world={'name':'synthetic-test-world','geometry_id':42,'transform':np.eye(4).tolist(),
              'bounds_type':1,'bounds_values':[250.,250.,3000.]},target_volume=None,
              current_surface=current,navigation_boundary_valid=True,navigation_boundary_surface=copy.deepcopy(current),
              target_surface={'geometry_id':0,'type':0,'frame':bind['surface_frame'],'bounds_type':0,'bounds_values':[]},
              world_boundary_surfaces=boundaries)
        rows.extend([copy.deepcopy(initial),copy.deepcopy(final)])
    return rows,baseline,baseline106,fixture

def test_observed_world_abort_localizes_without_claiming_seed_root_cause():
    r,b,c,f=sample();result=validate(r,b,c,f)
    assert result['classification']=='END_OF_WORLD_BEFORE_TARGET'
    assert result['world_boundary_geometry']['classification']=='WORLD_BOUNDARY_GEOMETRY_SUPPORTED'
    assert result['world_boundary_geometry']['face_axis']==0
    assert result['world_boundary_geometry']['face_sign']==1
    assert result['deeper_acceptance_or_seed_mechanism']=='UNKNOWN'

@pytest.mark.parametrize('mutation',['missing','foreign','flags','position','path','target','bounds',
  'world_transform','nonfinite','parameters','terminal','missing_bound','stage','options','numeric_flag','order'])
def test_changed_or_corrupted_evidence_rejected(mutation):
    r,b,c,f=sample();last=next(x for x in reversed(r) if x['record']=='end_world_check')
    if mutation=='missing':r=[x for x in r if not(x['record']=='end_world_check' and x['call_id']==24)]
    elif mutation=='foreign':last['call_id']=125
    elif mutation=='flags':last['target_reached_after']=False
    elif mutation=='position':last['position'][0][0]+=1.
    elif mutation=='path':last['path_mm']+=1.
    elif mutation=='target':last['target_surface']['frame'][2][3]+=1.
    elif mutation=='bounds':last['world']['bounds_values'][0]+=1.
    elif mutation=='world_transform':last['world']['transform'][0][3]+=1.
    elif mutation=='nonfinite':last['position'][0][0]=float('nan')
    elif mutation=='parameters':next(x for x in r if x['record']=='comparison')['diagnostic_parameters'][0][0]+=1.
    elif mutation=='terminal':r.pop()
    elif mutation=='missing_bound':r.remove(next(x for x in r if x['record']=='bound_before'))
    elif mutation=='stage':last['stage']='preStep'
    elif mutation=='options':last['options']['maxSteps']=4000
    elif mutation=='numeric_flag':last['returned']=1
    elif mutation=='order':r.remove(last);r.append(last)
    with pytest.raises(ValueError):validate(r,b,c,f)

def test_world_null_without_world_geometry_keeps_geometry_unknown():
    r,b,c,f=sample();last=next(x for x in reversed(r) if x['record']=='end_world_check');last['world']=None
    result=validate(r,b,c,f)
    assert result['classification']=='END_OF_WORLD_BEFORE_TARGET'
    assert result['world_boundary_geometry']['classification']=='UNKNOWN'

def test_position_far_from_world_face_does_not_pass_geometry_by_tuning():
    r,b,c,f=sample();last=next(x for x in reversed(r) if x['record']=='end_world_check')
    last=copy.deepcopy(last);last['position'][0][0]-=1.
    assert boundary_geometry(last)['classification']=='UNSUPPORTED'

def test_only_header_alias_and_include_change_from_sealed_source():
    generated=files()
    for name,text in generated.items():
        if name.endswith('AbortTrace.h'):continue
        old=(BASE/'isolated_source'/name).read_text()
        if name.endswith('WB107ExtrapolationTool.h'):
            text=text.replace('\n#include "AbortTrace.h"','').replace('using EndOfWorld = WB109Trace::EndOfWorld;',
              'using EndOfWorld = Acts::EndOfWorldReached;')
            assert 'IFaserActsExtrapolationTool' in text and 'IWB107ExtrapolationTool' not in text
        assert text==old
    wrapper=generated['WB107Diagnostic/AbortTrace.h']
    assert wrapper.count('Acts::EndOfWorldReached{}(s,stepper,navigator,logger)')==1
    assert 'return result;' in wrapper and 'getField(' not in wrapper and '.intersect(' not in wrapper
