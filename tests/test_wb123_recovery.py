"""Source-backed terminal flags, complete stream and installed target API recovery."""
import copy,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wb123_recovery import generated_source,source_proof,load_reader,target_request,target_check,OLD,NEW,SOURCE
from test_wb122_response import synthetic,column
from audit_wb122_response import projection


def false_flag_data():
    data=list(synthetic());r,c,f,h=data
    for i,row in enumerate(r['rows']):
        if row['input']['arm']!='observer_enabled':continue
        final=[x for x in row['observations'] if x['record']=='action'][-1];final['target_reached']=False;final['navigation_break']=True
        sid=row['input']['sample_id'];station=row['input']['station'];snaps=h['rows'][4*sid+2*(station-1)+1]['snapshots']
        snaps.append(projection(final,len(snaps)))
    return data


def fake_plane_rows(r):
    # Synthetic fixture, not used by production recover(): actual API is separately tested.
    return {row['input']['call_id']:{'input':target_request(row),'on_surface':True,'surface_tolerance_mm':1e-4,
       'path_to_plane_mm':0.,'local_position_mm':column([0.,0.,0.])}
       for row in r['rows'] if row['input']['arm']=='observer_enabled'}


def test_reversible_one_predicate_change_and_no_arithmetic_changes():
    source=SOURCE.read_text();new=generated_source();proof=source_proof()
    assert new.replace(NEW,OLD)==source and proof['restored_original_exact'] and proof['replacement_count']==1
    assert len([1 for a,b in zip(source.splitlines(),new.splitlines()) if a!=b])==1
    assert load_reader().BUDGET==1e-8


@pytest.mark.parametrize('flag_false',[False,True])
def test_true_and_false_terminal_flags_are_preserved_with_complete_target_contract(flag_false):
    data=false_flag_data() if flag_false else list(synthetic());saved=copy.deepcopy(data)
    module=load_reader(plane_rows=fake_plane_rows(data[0]));s=module.analyze(*data)
    assert data==saved and s['hypothesis']=='SUPPORTED_BUT_LIMITED_OBSERVATION_FIDELITY'
    assert len(s['profiles'])==18 and len(s['comparisons'])==36
    assert all(p['requested_plane_check']['status']=='PASS_INSTALLED_REQUESTED_PLANE' for p in s['profiles'])
    assert s['profiles'][0]['public_snapshots']==(3 if flag_false else 2)
    assert all(p['terminal_flags']['target_reached'] is (not flag_false) for p in s['profiles'])
    assert s['qualification']=='NOT_EVALUATED'


def test_frozen_analyzer_keeps_original_false_flag_unknown():
    import audit_wb122_response as original
    data=false_flag_data();old=original.analyze(*data);new=load_reader().analyze(*data)
    assert old['hypothesis']=='UNKNOWN_FIDELITY_OR_COVERAGE' and len(old['profiles'])==0
    assert new['hypothesis']=='SUPPORTED_BUT_LIMITED_OBSERVATION_FIDELITY'


@pytest.mark.parametrize('bad',['terminal_stage','terminal_count','terminal_position','terminal_time','terminal_qop','terminal_counter','terminal_type','duplicate_action','missing_post','missing_query','duplicate_trial','step_index','field_unit','cache_map','nonfinite','wrong_bound','wrong_request'])
def test_no_relaxation_of_complete_coverage_or_fidelity(bad):
    r,c,f,h=false_flag_data();row=r['rows'][2];obs=row['observations'];snaps=h['rows'][1]['snapshots'];final=obs[8]
    if bad=='terminal_stage':final['stage']='prePropagation'
    elif bad=='terminal_count':final['accepted_step_count']=2
    elif bad=='terminal_position':final['position_mm'][0][0]=1.
    elif bad=='terminal_time':final['time_acts']=0.
    elif bad=='terminal_qop':final['qop_acts']=2.
    elif bad=='terminal_counter':final['rk_rejections']=3
    elif bad=='terminal_type':final['target_reached']=0
    elif bad=='duplicate_action':obs.insert(8,copy.deepcopy(final))
    elif bad=='missing_post':obs.pop(7)
    elif bad=='missing_query':obs.pop(3)
    elif bad=='duplicate_trial':obs.insert(6,copy.deepcopy(obs[5]))
    elif bad=='step_index':obs[1]['step_index']=1
    elif bad=='field_unit':obs[2]['field_T'][0][0]=1.
    elif bad=='cache_map':obs[2]['cache_after']['actual_condition_map_match']=False
    elif bad=='nonfinite':obs[5]['error_estimate']=float('nan')
    elif bad=='wrong_bound':obs[10]['parameters'][0][0]=1.
    elif bad=='wrong_request':obs[9]['requested_target']=False
    with pytest.raises(ValueError):load_reader().trace_check(obs,row,snaps,r['acts_T_unit'])


@pytest.mark.parametrize('bad',['off_surface','changed_frame','changed_input_position','wrong_tolerance','wrong_qop','wrong_time','wrong_phi','wrong_local_position','changed_exported_h'])
def test_actual_target_checks_fail_closed(bad):
    r,c,f,h=false_flag_data();row=r['rows'][2];obs=row['observations'];planes=fake_plane_rows(r);e=planes[row['input']['call_id']]
    if bad=='off_surface':e['on_surface']=False
    elif bad=='changed_frame':e['input']['frame'][0][3]=3.
    elif bad=='changed_input_position':e['input']['position_mm'][0][0]=3.
    elif bad=='wrong_tolerance':e['surface_tolerance_mm']=1e-3
    elif bad=='wrong_qop':obs[10]['parameters'][4][0]=.1
    elif bad=='wrong_time':obs[10]['parameters'][5][0]=.1
    elif bad=='wrong_phi':obs[10]['parameters'][3][0]=1.
    elif bad=='wrong_local_position':e['local_position_mm'][0][0]=1.
    elif bad=='changed_exported_h':row['state']['h'][0][0]=1.
    with pytest.raises(ValueError):target_check(obs,row,planes,load_reader().near)


def test_neither_boolean_flag_value_can_override_failed_requested_plane():
    for data in (list(synthetic()),false_flag_data()):
        planes=fake_plane_rows(data[0]);planes[3]['on_surface']=False
        summary=load_reader(plane_rows=planes).analyze(*data)
        assert summary['hypothesis']=='UNKNOWN_FIDELITY_OR_COVERAGE'
        assert summary['unknowns'][0]['detail']=='installed PlaneSurface onSurface contract'


@pytest.mark.parametrize('bad',['missing_original','duplicate_original','already_adapted'])
def test_source_adapter_rejects_unexpected_original(bad):
    s=SOURCE.read_text()
    if bad=='missing_original':s=s.replace(OLD,'')
    elif bad=='duplicate_original':s+='\n'+OLD+'\n'
    else:s=generated_source()
    with pytest.raises(ValueError):generated_source(s)
