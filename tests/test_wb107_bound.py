import copy,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb107_bound import validate
from wb107_sources import files,tool_sources,OFFICIAL

BASE=Path(__file__).resolve().parents[1]/'outputs/mc24_four_station_wb106_surface_trace_v1'

def sample():
    baseline=[json.loads(x) for x in (BASE/'event/acts.json.calls.ndjson').read_text().splitlines()]
    fixture=json.loads((BASE/'fixture.json').read_text())
    rows=[]
    for r in baseline:
        rows.append(copy.deepcopy(r))
        if r['record']!='after_official':continue
        cid=r['call_id'];success=r['has_value']
        original=[x for x in baseline if x.get('call_id')==cid]
        inp=next(x for x in original if x['record']=='input')
        before=next(x for x in original if x['record']=='before_official')
        frame=np.asarray(before['target_frame'])
        if success:
            o=next(x for x in original if x['record']=='output')
            pos=o['global_position'];direc=(frame[:3,:3]@np.asarray(o['direction']).reshape(3))[:,None].tolist()
        else:pos=(frame[:3,3]+np.array([0.,0.,2.]))[:,None].tolist();direc=[[0.],[0.],[1.]]
        b={'record':'bound_before','call_id':cid,'requested_target':True,'surface_geometry_id':0,
           'surface_frame':frame.tolist(),'position':pos,'direction':direc,'qop_acts':inp['seed'][4][0]/1e-3,
           'MeV_acts':1e-3,'time':0.,'path_mm':100.,'cov_transport':False,'transport_cov_argument':True,'on_surface_tolerance_mm':1e-4}
        err={} if success else {'error_category':'SurfaceError','error_value':1,'error_message':'synthetic test only'}
        c={'record':'comparison','call_id':cid,'official_has_value':success,'diagnostic_has_value':success}
        if success:
            for k,v in (('parameters',[[0.]]*6),('position',pos),('direction',direc)):
                c['official_'+k]=copy.deepcopy(v);c['diagnostic_'+k]=copy.deepcopy(v)
        rows.extend([{'record':'before_diagnostic','call_id':cid},b,
          {'record':'bound_after','call_id':cid,'ok':success,**err},
          {'record':'propagator_result','call_id':cid,'ok':success,**err},c])
    return rows,baseline,fixture

def test_final_binding_attribution_is_localization_only():
    r,b,f=sample();s=validate(r,b,f)
    assert s['classification']=='FINAL_TARGET_BOUND_CONVERSION'
    assert s['absolute_plane_distance_mm']==2.
    assert s['deeper_navigation_or_field_mechanism']=='UNKNOWN'

@pytest.mark.parametrize('mutation',['event','target','qop','missing_bound','parameters','error','requested',
  'nonfinite','terminal','foreign_call','presence','official_output'])
def test_corrupted_or_changed_diagnostic_rejected(mutation):
    r,b,f=sample()
    last=lambda kind:next(x for x in reversed(r) if x['record']==kind)
    if mutation=='event':r[0]['actual_event']+=1
    elif mutation=='target':last('bound_before')['surface_frame'][2][3]+=1.
    elif mutation=='qop':last('bound_before')['qop_acts']*=1000.
    elif mutation=='missing_bound':r.remove(next(x for x in r if x['record']=='bound_before'))
    elif mutation=='parameters':next(x for x in r if x['record']=='comparison')['diagnostic_parameters'][0][0]+=1.
    elif mutation=='error':last('bound_after')['error_value']=2
    elif mutation=='requested':last('bound_before')['requested_target']=False
    elif mutation=='nonfinite':last('bound_before')['position'][0][0]=float('nan')
    elif mutation=='terminal':r.pop()
    elif mutation=='foreign_call':last('comparison')['call_id']=900
    elif mutation=='presence':last('comparison')['diagnostic_has_value']=True
    elif mutation=='official_output':next(x for x in r if x['record']=='output')['h'][0][0]+=1.
    with pytest.raises(ValueError):validate(r,b,f)

def test_missing_failed_binding_gives_unresolved_stage():
    r,b,f=sample();r=[x for x in r if not (x.get('call_id')==124 and x['record'] in ('bound_before','bound_after'))]
    assert validate(r,b,f)['classification']=='OTHER_OR_UNRESOLVED_STAGE'

def test_tool_copy_is_exact_outside_passive_instrumentation():
    source=tool_sources()['WB107ExtrapolationTool.cxx']
    start=source.index('      WB107Trace::Json receipt');end=source.index('      if (!result.ok())',start)
    source=source[:start]+source[end:]
    source=source.replace('\n#include "BoundTrace.h"','').replace('\nDECLARE_COMPONENT(WB107ExtrapolationTool)\n','')
    source=source.replace('WB107Trace::Stepper','Acts::EigenStepper<>').replace('WB107ExtrapolationDetail','ActsExtrapolationDetail').replace('WB107ExtrapolationTool','FaserActsExtrapolationTool')
    assert source==(OFFICIAL/'FaserActsExtrapolationTool.cxx').read_text()
    h=tool_sources()['WB107ExtrapolationTool.h'].replace('WB107_EXTRAPOLATION_TOOL_H','FASERACTSGEOMETRY_ACTSEXTRAPOLATIONTOOL_H').replace('WB107ExtrapolationDetail','ActsExtrapolationDetail').replace('WB107ExtrapolationTool','FaserActsExtrapolationTool')
    assert h==(OFFICIAL/'FaserActsExtrapolationTool.h').read_text()
    generated=files()['WB107Diagnostic/CommonSeedAudit.cxx']
    assert generated.count('m_tool->propagate(ctx,start,*target,distance>=0?Acts::Direction::Forward:Acts::Direction::Backward)')==1
    assert 'stepTolerance' not in generated
