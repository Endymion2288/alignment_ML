#!/usr/bin/env python3
"""Reversible source-identified WB122 terminal semantics adapter, saved JSON only."""
import hashlib,sys,types
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT
from audit_wb117_response import check,array
from wb92_contract import ACTS
SOURCE=ROOT/'scripts/audit_wb122_response.py'
OLD="    check(actions[-1]['stage']=='postPropagation' and actions[-1]['target_reached'] is True,'target final action')"
NEW="    validate_terminal_actions(actions,n)"
SURFACE_TOLERANCE_MM=1e-4


def validate_terminal_actions(actions,n):
    check(len(actions)==n+2 and actions[-1]['stage']=='postPropagation' and actions[-1]['accepted_step_count']==n,'terminal stage/count')
    check(sum(a['stage']=='prePropagation' for a in actions)==1 and sum(a['stage']=='postPropagation' for a in actions)==1,'unique initial/final actions')
    for a in actions:
        check(type(a['target_reached']) is bool and type(a['navigation_break']) is bool and type(a['accepted_step_count']) is int,'actual action flag/counter types')
    post=actions[-2];final=actions[-1]
    check(post['stage']=='postStep' and post['accepted_step_count']==n,'last accepted Action')
    for key in ('position_mm','direction','qop_acts','time_acts','path_mm','rk_rejections'):
        check(final[key]==post[key],'postPropagation free-state preservation '+key)
    # Aborters may change flags/constraints; neither target flag value is a success oracle.


def generated_source(original=None):
    original=SOURCE.read_text() if original is None else original
    check(original.count(OLD)==1 and NEW not in original,'one exact original terminal predicate')
    generated=original.replace(OLD,NEW)
    check(generated.replace(NEW,OLD)==original,'exact source reverse restoration')
    compile(generated,'wb123_generated_reader.py','exec')
    return generated


def source_proof():
    original=SOURCE.read_text();generated=generated_source(original)
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    return {'original_sha256':sha(original),'generated_sha256':sha(generated),'restored_original_exact':True,
            'replacement_count':1,'original_predicate':OLD,'replacement':NEW,'scope':'one terminal interface predicate only; all original scientific arithmetic/exact gates unchanged'}


def load_reader(source=None,plane_rows=None):
    module=types.ModuleType('wb123_generated_reader');module.__file__='wb123_generated_reader.py'
    exec(compile(generated_source() if source is None else source,module.__file__,'exec'),module.__dict__)
    module.validate_terminal_actions=validate_terminal_actions
    if plane_rows is not None:
        original_check=module.trace_check
        def trace_check(obs,row,old_snaps,T):
            profile=original_check(obs,row,old_snaps,T)
            profile['requested_plane_check']=target_check(obs,row,plane_rows,module.near)
            profile['terminal_flags']={'target_reached':next(x for x in reversed(obs) if x['record']=='action')['target_reached'],
                                       'navigation_break':next(x for x in reversed(obs) if x['record']=='action')['navigation_break']}
            return profile
        module.trace_check=trace_check
    return module


def target_request(row):
    before=[x for x in row['observations'] if x['record']=='bound_before']
    check(len(before)==1,'unique target request state');b=before[0];i=row['input']
    return {'call_id':i['call_id'],'sample_id':i['sample_id'],'station':i['station'],'frame':b['frame'],
            'position_mm':b['position_mm'],'direction':b['direction'],'navigation_direction':i['navigation_direction']}


def target_check(obs,row,plane_rows,near):
    call=row['input']['call_id'];evaluation=plane_rows[call];check(evaluation['input']==target_request(row),'installed API request identity')
    check(type(evaluation['on_surface']) is bool and evaluation['on_surface'] is True and evaluation['surface_tolerance_mm']==SURFACE_TOLERANCE_MM,'installed PlaneSurface onSurface contract')
    p=next(x for x in obs if x['record']=='bound_before');after=next(x for x in obs if x['record']=='bound_after')
    frame=array(p['frame'],(4,4));position=array(p['position_mm'],(3,1)).reshape(3);direction=array(p['direction'],(3,1)).reshape(3)
    local=frame[:3,:3].T@(position-frame[:3,3]);near(local,array(evaluation['local_position_mm'],(3,1)).reshape(3),'installed target local coordinate')
    params=array(after['parameters'],(6,1)).reshape(6)
    check(params[4]==p['qop_acts']==row['state']['qop_acts'] and params[5]==p['time_acts']==row['state']['time_acts'],'target qop/time bound convention')
    near(params[:2],local[:2],'free-to-bound local position')
    bound_position=frame[:3,:3]@np.r_[params[:2],0.]+frame[:3,3];near(bound_position,array(after['position_mm'],(3,1)).reshape(3),'actual bound target position')
    phi,theta=params[2:4];reconstructed=np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta),np.cos(theta)])
    near(reconstructed,direction,'bound phi/theta global direction');near(reconstructed,array(after['direction'],(3,1)).reshape(3),'bound exported direction')
    local_direction=frame[:3,:3].T@array(after['direction'],(3,1)).reshape(3);check(abs(local_direction[2])>=1e-12,'target local slope denominator')
    near(np.r_[params[:2],local_direction[:2]/local_direction[2]],array(row['state']['h'],(4,1)).reshape(4),'bound scientific h')
    return {'status':'PASS_INSTALLED_REQUESTED_PLANE','call_id':call,'on_surface':True,
            'surface_tolerance_mm':evaluation['surface_tolerance_mm'],'path_to_plane_mm':evaluation['path_to_plane_mm'],
            'free_local_z_mm':float(local[2]),'terminal_flag_is_not_success_oracle':True,'aborter_branch':'UNKNOWN_NOT_RECORDED'}


def recover(out,source,plane_result):
    from alignment.wb90_measurement_contract import read_public
    check(plane_result['schema']=='wb123_installed_plane_contract_v1' and plane_result['propagation_calls']==0 and plane_result['event_access'] is False and plane_result['field_access'] is False,'saved geometric API scope/schema')
    rows={r['input']['call_id']:r for r in plane_result['rows']};check(len(rows)==len(plane_result['rows'])==18,'complete plane matrix')
    summary=load_reader(source,rows).audit(out)
    summary.update(schema='wb123_saved_complete_rk_semantics_summary_v1',classification='DIAGNOSTIC_ONLY_SAVED_COMPLETE_RK_SEMANTICS_RECOVERY',
        original_wb122_hypothesis='UNKNOWN_FIDELITY_OR_COVERAGE',
        recovery_execution='PASS_SAVED_INTERFACE_RECOVERY' if not summary['unknowns'] and len(summary['profiles'])==18 else 'UNKNOWN_INTERFACE_OR_COVERAGE',
        new_physical_calls=0,new_event_root_access=False,original_summary_unchanged=True,prospective_physical_qualification=False)
    return summary
