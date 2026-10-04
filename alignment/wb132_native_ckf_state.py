"""Exact native-state evidence; no transport, fit, source replacement or prior."""
from __future__ import annotations
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parameter_identity(old, new):
    require(len(old)==len(new), 'flat parameter coverage')
    for before, after in zip(old,new):
        for key, value in before.items():
            require(key in after and after[key]==value, 'native parameter identity '+key)


def exact_state_map(states):
    """Match only explicit resolved measurement PRD links, never geometry proximity."""
    mapping={}
    for state in states:
        measurement=state['measurement']
        if measurement is None or not measurement['selected_prd']:
            continue
        require(measurement['prd_link_resolved'] and measurement['prd_link_valid'] and
                measurement['selected_prd_pointer_equal'], 'unresolved exact PRD link')
        identifier=measurement['cluster_id']
        require(identifier==measurement['rot_identifier'], 'ROT/PRD identifier mismatch')
        require(identifier not in mapping, 'duplicate selected PRD TSOS')
        mapping[identifier]=state
    return mapping


def plane_tolerance(position, frame):
    return max(1e-6,4*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(position))),float(np.max(np.abs(frame[:3,3])))))


def native_residual(original, parameter, state, protocol):
    m=state['measurement'];prediction=m['native_prediction']
    require(m['wafer_id']==original['wafer_id'] and m['station']==original['station'], 'wrong wafer/station')
    require(m['rdo_ids']==original['rdo_ids'], 'RDO association changed')
    require(m['prd_local_position']==original['local_position'], 'PRD measurement changed')
    require(m['prd_local_covariance'][0][0]==original['sigma_sq'], 'PRD covariance changed')
    frame=np.asarray(m['sensor_transform'],dtype=float)
    expected=np.asarray(original['sensor_transform'],dtype=float)
    require(frame.shape==(4,4) and np.all(np.isfinite(frame)), 'invalid sensor frame')
    require(np.max(np.abs(frame-expected))<=protocol['surface_transform_tolerance'], 'sensor surface mismatch')
    require(np.max(np.abs(frame[:3,:3].T@frame[:3,:3]-np.eye(3)))<=1e-9, 'sensor frame not rigid')
    require(m['measurement_surface_type']==4, 'native measurement is not a Trk plane')
    position=np.asarray(parameter['global_position_native'])
    local=frame[:3,:3].T@(position-frame[:3,3])
    require(np.all(np.isfinite(local)), 'nonfinite native prediction')
    tolerance=plane_tolerance(position,frame)
    require(abs(tolerance-prediction['persistence_plane_tolerance_mm'])<1e-15, 'plane tolerance definition')
    require(np.max(np.abs(local-prediction['sensor_local_position_mm']))<=1e-9, 'native frame computation')
    require(np.linalg.norm(frame[:3,:3]@local+frame[:3,3]-position)<=protocol['frame_roundtrip_tolerance_mm'], 'sensor frame roundtrip')
    require(prediction['sensor_frame_roundtrip_mm']<=protocol['frame_roundtrip_tolerance_mm'], 'exporter frame roundtrip')
    require(abs(local[0]-prediction['loc0_mm'])<=1e-9, 'prediction component')
    residual=original['local_position'][0]-local[0]
    require(abs(residual-prediction['prd_loc0_residual_mm'])<=1e-9, 'residual sign')
    require(m['rot_loc1'] is not None, 'ROT loc1 absent')
    rot_residual=m['rot_loc1']-local[0]
    require(abs(rot_residual-prediction['rot_loc1_residual_mm'])<=1e-9, 'ROT residual sign')
    flags=state['type_flags']
    reasons=[]
    if not flags[0] or flags[5] or flags[6]:reasons.append('NOT_MEASUREMENT_OR_OUTLIER_HOLE')
    if abs(local[2])>tolerance:reasons.append('OFF_NATIVE_MEASUREMENT_PLANE')
    if not prediction['inside_bounds']:reasons.append('OUTSIDE_NATIVE_SENSOR_BOUNDS')
    return {'cluster_id':original['cluster_id'],'wafer_id':original['wafer_id'],'station':original['station'],
            'tsos_index':state['tsos_index'],'parameter_index':state['parameter_index'],
            'eligible':not reasons,'unknown_reasons':reasons,'prd_residual_mm':float(residual),
            'rot_residual_mm':float(rot_residual),'rot_minus_prd_mm':float(m['rot_loc1']-original['local_position'][0]),
            'prd_sigma_sq':original['sigma_sq'],'rot_sigma_sq':m['rot_covariance'][0][0],
            'plane_distance_mm':float(local[2]),'plane_tolerance_mm':tolerance,
            'strict_is_on_surface':prediction['strict_is_on_surface'],'inside_bounds':prediction['inside_bounds'],
            'qop_per_MeV':parameter['native_parameters'][4],'type_description':state['type_description'],
            'fit_chi2':state['fit_chi2'],'fit_dof':state['fit_dof']}


def audit_event(export, response, parent, old_response, protocol, producer):
    require(response['identity']==export['identity']==parent['identity'], 'event identity')
    for key in ('new_reconstruction_calls','new_propagation_calls','algorithm_field_queries'):
        require(response[key]==0,'unexpected work '+key)
    old_container=parent['persisted_tracks'][0];old=old_container['matching_tracks'][0]
    require(response['track_key']==old_container['key']==protocol['selected_container'], 'track container')
    require(response['container_size']==old_container['container_size'] and
            response['track_index']==old['track_index']==protocol['selected_track_index'],'track selection')
    flat=response['flat_parameters'];parameter_identity(old['parameters'],flat)
    states=response['states']
    require([s['tsos_index'] for s in states]==list(range(len(states))), 'TSOS sequence')
    indices=[s['parameter_index'] for s in states if s['parameter_index'] is not None]
    require(sorted(indices)==list(range(len(flat))), 'exact TSOS/flat parameter mapping')
    for parameter in flat:
        require(parameter['parameter_frame_roundtrip_mm']<=1e-9,'parameter frame roundtrip')
        covariance=parameter['covariance']
        if covariance is not None:
            require(np.asarray(covariance).shape==(5,5) and np.all(np.isfinite(covariance)), 'native covariance representation')
    mapping=exact_state_map(states)
    require([r['cluster_id'] for r in response['selected_prds']]==[r['cluster_id'] for r in export['rows']], 'selected PRD order')
    for original,raw in zip(export['rows'],response['selected_prds']):
        for key in ('cluster_id','wafer_id','station','local_position','sigma_sq','rdo_ids'):
            require(raw[key]==original[key],'original PRD identity '+key)
        require(np.max(np.abs(np.asarray(raw['sensor_transform'])-original['sensor_transform']))<=protocol['surface_transform_tolerance'],'PRD transform')
    expected={r['cluster_id'] for r in old['cluster_membership'] if r['cluster_id'] in {e['cluster_id'] for e in export['rows']}}
    require(set(mapping)==expected,'old selected membership changed')
    index=old_response['index'];require(len(mapping)==protocol['expected_selected_prd_matches'][str(index)], 'match count')
    rows=[];missing=[]
    for original in export['rows']:
        state=mapping.get(original['cluster_id'])
        if state is None:
            require(original['station']==0 or (index==20 and original['station']==2 and original['cluster_id']==10469004361915695104),'unexpected missing PRD')
            missing.append({'cluster_id':original['cluster_id'],'station':original['station'],'status':'NOT_IN_SELECTED_CKF'})
            continue
        if state['parameter_index'] is None or state['measurement']['native_prediction'] is None:
            rows.append({'cluster_id':original['cluster_id'],'station':original['station'],'eligible':False,'unknown_reasons':['NATIVE_PARAMETER_PREDICTION_MISSING']})
        else:
            rows.append(native_residual(original,flat[state['parameter_index']],state,protocol))
    require(len(rows)+len(missing)==len(export['rows']),'original coverage')
    stations={};gross=False
    for station in (1,2,3):
        members=[r for r in rows if r['station']==station and r['eligible']]
        unknown=[r for r in rows if r['station']==station and not r['eligible']]
        metric={'matched_rows':len(members)+len(unknown),'eligible_rows':len(members),'unknown_rows':len(unknown)}
        if members:
            residual=np.array([r['prd_residual_mm'] for r in members]);rot=np.array([r['rot_residual_mm'] for r in members]);variance=np.array([r['prd_sigma_sq'] for r in members])
            rms=float(np.sqrt(np.mean(residual**2)));maximum=float(np.max(np.abs(residual)))
            metric.update(prd_rms_mm=rms,rot_rms_mm=float(np.sqrt(np.mean(rot**2))),max_abs_prd_mm=maximum,
                          descriptive_prd_cost=float(np.sum(residual**2/variance)),
                          qop_range_per_MeV=[min(r['qop_per_MeV'] for r in members),max(r['qop_per_MeV'] for r in members)])
            gross |= rms>protocol['gross_station_rms_mm'] or maximum>protocol['gross_max_abs_residual_mm']
        stations[str(station)]=metric
    unknown=any(not r['eligible'] for r in rows)
    hypothesis=('FAIL_GROSS_NATIVE_RESIDUAL' if gross else 'UNKNOWN' if unknown else 'PASS_GROSS_SCREEN_ONLY')
    selected=old_response['selection']['selected_index']
    source_states=[s for s in states if s['parameter_index']==selected]
    require(len(source_states)==1,'source index exact TSOS')
    source_state=source_states[0];source_parameter=flat[selected];flags=source_state['type_flags']
    current_rule='predicted' if flags[6] else 'filtered' if flags[5] else 'smoothed' if flags[0] else 'UNCLASSIFIED'
    source={'persistent_index':selected,'tsos_index':source_state['tsos_index'],'type_flags':flags,
            'type_description':source_state['type_description'],'measurement_present':source_state['measurement_present'],
            'cluster_id':source_state['measurement']['cluster_id'] if source_state['measurement'] else None,
            'measurement_anchor':bool(flags[0] and not flags[5] and not flags[6] and source_state['measurement'] is not None),
            'current_producer_rule':current_rule,'producer_rule_evidence':'CURRENT_RULE_CONSISTENT_HISTORICAL_UNKNOWN',
            'native_qop_per_MeV':source_parameter['native_parameters'][4],'p_seed_qop_per_MeV':old_response['seeds']['P'][4][0],
            'covariance':source_parameter['covariance'],'global_position_mm':source_parameter['global_position_native'],
            'qop_covariance_frame_prior':'NOT_ESTABLISHED_SAME_TRACK_CONDITIONAL'}
    type_counts={}
    for s in states:type_counts[s['type_description']]=type_counts.get(s['type_description'],0)+1
    return {'schema':'wb132_native_state_audit_v1','index':index,'identity':response['identity'],'interface':'PASS',
            'native_prediction_hypothesis':hypothesis,'matched_rows':len(rows),'unmatched_rows':len(missing),
            'eligible_rows':sum(r['eligible'] for r in rows),'rows':rows,'missing':missing,'source':source,
            'stations':stations,'type_counts':type_counts,'material_effects_state_count':sum(s['material_effects_present'] for s in states),
            'global_fit_quality':response['global_fit_quality'],'track_info':response['track_info'],
            'historical_producer_identity':producer['historical_mc24_producer_execution_identity'],
            'new_reconstruction_calls':0,'new_propagation_calls':0,'truth_access':False}
