"""Frozen conditional common-track screen; no fitting or covariance prior."""
from __future__ import annotations
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_request(export, native):
    require(native['identity'] == export['identity'], 'source event identity')
    ids = {r['cluster_id']: r for r in export['rows']}
    require(len(ids) == len(export['rows']), 'duplicate original PRD')
    candidates = []
    linked = set()
    for state in native['states']:
        m = state['measurement']
        if not m or not m['selected_prd']:
            continue
        identifier = m['cluster_id']
        require(identifier in ids and identifier not in linked, 'wrong/duplicate linked PRD')
        linked.add(identifier)
        require(m['prd_link_valid'] and m['prd_link_resolved'] and m['selected_prd_pointer_equal'] and
                m['rot_identifier'] == identifier and m['wafer_id'] == ids[identifier]['wafer_id'], 'exact source link')
        flags = state['type_flags']
        if flags[0] and not flags[5] and not flags[6]:
            require(state['parameter_index'] is not None, 'accepted parameter absent')
            p = native['flat_parameters'][state['parameter_index']]
            require(p['persistent_index'] == state['parameter_index'], 'parameter index')
            position = np.asarray(p['global_position_native'], float)
            require(position.shape == (3,) and np.all(np.isfinite(position)), 'candidate position')
            candidates.append((float(position[2]), state['tsos_index'], state, p))
    require(bool(candidates), 'no accepted source')
    _, _, state, p = min(candidates, key=lambda t: (t[0], t[1]))
    m = state['measurement']
    position = np.asarray(p['global_position_native'], float)
    momentum = np.asarray(p['global_momentum_native'], float)
    require(momentum.shape == (3,) and np.all(np.isfinite(momentum)) and np.linalg.norm(momentum) > 0, 'source momentum')
    direction = momentum / np.linalg.norm(momentum)
    phi, theta, qop = p['native_parameters'][2:5]
    require(np.isfinite(qop) and qop != 0 and qop * p['charge_e'] > 0, 'source qop/charge')
    angles = [np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]
    require(np.linalg.norm(direction-angles) < 1e-9, 'direction angles inconsistency')
    require(abs(np.linalg.norm(momentum)*abs(qop)-1) < 1e-6, 'source momentum/qop units')
    require(p['position_unit'] == 'mm' and p['momentum_unit'] == 'MeV' and p['qop_unit'] == 'MeV^-1', 'source units')
    frame = np.asarray(ids[m['cluster_id']]['sensor_transform'], float)
    local = frame[:3,:3].T @ (position-frame[:3,3])
    tol = max(1e-6, 4*2**-23*max(1., max(abs(position)), max(abs(frame[:3,3]))))
    prediction = m['native_prediction']
    require(prediction is not None and abs(local[2]) <= tol and prediction['inside_bounds'], 'invalid source measurement plane')
    return {'schema': 'wb133_accepted_anchor_request_v1', 'identity': export['identity'], 'source': {
        'tsos_index': state['tsos_index'], 'parameter_index': state['parameter_index'],
        'cluster_id': m['cluster_id'], 'wafer_id': m['wafer_id'], 'station': m['station'],
        'type_flags': state['type_flags'], 'type_description': state['type_description'],
        'native_parameters': p['native_parameters'], 'position_mm': position.tolist(),
        'direction': direction.tolist(), 'qop_per_MeV': qop, 'persistence_plane_tolerance_mm': tol,
        'covariance_prior': 'NONE_SAME_TRACK_POSTERIOR_NOT_USED'}}


def tangent_path(source, frame):
    normal = np.asarray(frame)[:3,2]
    denom = float(normal @ np.asarray(source['direction']))
    require(abs(denom) >= 1e-12, 'near-parallel target')
    return float(normal @ (np.asarray(frame)[:3,3]-source['position_mm']) / denom)


def station_metrics(records, residual_key):
    result = {}
    for station in (0,1,2,3):
        group = [r for r in records if r['station'] == station]
        valid = [r for r in group if r.get(residual_key) is not None and r.get('valid', True)]
        values = np.array([r[residual_key] for r in valid], float)
        metric = {'rows': len(group), 'valid_rows': len(valid), 'complete': len(valid) == len(group)}
        if len(values):
            metric.update(mean_mm=float(np.mean(values)), rms_mm=float(np.sqrt(np.mean(values**2))),
                          max_abs_mm=float(np.max(np.abs(values))),
                          descriptive_cost=float(sum(r[residual_key]**2/r['sigma_sq'] for r in valid)))
        result[str(station)] = metric
    return result


def audit_event(export, native, request, response, baseline, protocol):
    require(request == source_request(export, native), 'source definition changed')
    source = request['source'];qop = source['qop_per_MeV']
    require(response['identity'] == baseline['identity'] == export['identity'] and response['source'] == source, 'response identity')
    require(response['schema'] == 'wb133_accepted_anchor_response_v1', 'response schema')
    require(response['new_reconstruction_calls'] == 0 and response['material_source'] == 'None' and
            response['field_mode'] == 'FASER' and not response['covariance_transport'], 'model contract')
    require(np.linalg.norm(np.asarray(response['source_actual_position_mm']).reshape(3)-source['position_mm']) <= 1e-9 and
            np.linalg.norm(np.asarray(response['source_actual_direction']).reshape(3)-source['direction']) <= 1e-12 and
            response['source_position_roundtrip_mm'] <= 1e-9 and response['source_direction_roundtrip'] <= 1e-12, 'source roundtrip')
    require(abs(response['source_actual_qop_per_MeV']-qop) <= max(1e-15, abs(qop)*1e-10), 'source qop conversion')
    require(len(response['rows']) == len(baseline['rows']) == len(export['rows']), 'row coverage')
    links = {s['measurement']['cluster_id']: s for s in native['states'] if s['measurement'] and s['measurement']['selected_prd']}
    records = [];calls = [];controls = 0
    for original, row, old in zip(export['rows'], response['rows'], baseline['rows']):
        for key in ('cluster_id', 'wafer_id', 'station'):
            require(row[key] == old[key] == original[key], 'row identity')
        anchor = original['cluster_id'] == source['cluster_id']
        path = tangent_path(source, original['sensor_transform'])
        require(abs(path-row['tangent_path_mm']) <= max(1e-9, abs(path)*1e-12), 'direction path')
        if anchor:
            require(row['kind'] == 'SOURCE_IDENTITY_CONTROL' and row['call_id'] is None and row['propagation_direction'] == 'ZERO_PATH', 'anchor control')
            controls += 1
        else:
            require(row['kind'] == 'PROPAGATED' and row['propagation_direction'] == ('FORWARD' if path >= 0 else 'BACKWARD'), 'propagation direction')
            calls.append(row['call_id'])
        require(old['status'] == 'SUCCESS', 'baseline prediction absent')
        old_prediction = old['state']['local_position_mm'][0][0]
        linked = links.get(original['cluster_id'])
        native_prediction = (linked['measurement']['native_prediction']['loc0_mm'] if linked and linked['measurement']['native_prediction'] is not None else None)
        record = {'cluster_id': original['cluster_id'], 'wafer_id': original['wafer_id'], 'station': original['station'],
                  'kind': row['kind'], 'sigma_sq': original['sigma_sq'], 'status': row['status'],
                  'accepted_measurement': bool(linked and linked['type_flags'][0] and not linked['type_flags'][5] and not linked['type_flags'][6]),
                  'exact_native_link': linked is not None, 'native_prediction_mm': native_prediction,
                  'original_P_prediction_mm': old_prediction, 'original_P_residual_mm': original['local_position'][0]-old_prediction,
                  'valid': False, 'prediction_mm': None, 'residual_mm': None, 'new_minus_P_mm': None, 'new_minus_native_mm': None}
        require(original['sigma_sq'] > 0 and np.isfinite(original['sigma_sq']), 'strip covariance')
        if row['status'] == 'SUCCESS':
            position = np.asarray(row['global_position_mm'], float).reshape(3)
            frame = np.asarray(original['sensor_transform'], float)
            local = frame[:3,:3].T @ (position-frame[:3,3])
            require(np.all(np.isfinite(local)) and np.all(np.isfinite(row['global_direction'])), 'nonfinite endpoint')
            require(np.max(np.abs(local-np.asarray(row['local_position_mm']).reshape(3))) < 1e-9 and
                    np.linalg.norm(frame[:3,:3]@local+frame[:3,3]-position) <= 1e-9 and row['frame_roundtrip_mm'] <= 1e-9, 'frame roundtrip')
            require(abs(row['qop_per_MeV']-qop) <= max(1e-15, abs(qop)*1e-10) and not row['covariance_present'], 'qop/covariance model')
            require(abs(local[0]-row['predicted_loc0_mm']) < 1e-9 and
                    abs(original['local_position'][0]-local[0]-row['local_residual_mm']) < 1e-9, 'residual component/sign')
            tol = source['persistence_plane_tolerance_mm'] if anchor else protocol['endpoint_plane_tolerance_mm']
            require(row['plane_tolerance_mm'] == tol, 'endpoint tolerance changed')
            if anchor:
                require(np.linalg.norm(position-source['position_mm']) <= 1e-9, 'anchor position altered')
            valid = row['inside_bounds'] and abs(local[2]) <= tol and (anchor or row['strict_is_on_surface_with_bounds'])
            pred = float(local[0]);residual = float(original['local_position'][0]-pred)
            record.update(valid=bool(valid), prediction_mm=pred, residual_mm=residual,
                          new_minus_P_mm=pred-old_prediction, new_minus_native_mm=(pred-native_prediction if native_prediction is not None else None),
                          plane_distance_mm=float(local[2]), inside_bounds=row['inside_bounds'], strict_on_surface=row['strict_is_on_surface_with_bounds'])
        else:
            require(row['status'] == 'FAIL_OFFICIAL_NULL' and not anchor, 'unknown response status')
        records.append(record)
    require(calls == list(range(1,len(export['rows']))) and controls == response['zero_path_controls'] == 1 and
            response['official_calls'] == response['new_propagation_calls'] == len(calls), 'exact call coverage')
    stations = station_metrics(records, 'residual_mm')
    baseline_records = [{**r, 'valid': True} for r in records]
    baseline_stations = station_metrics(baseline_records, 'original_P_residual_mm')
    complete = all(r['valid'] for r in records)
    gross = any(m.get('rms_mm',0) > protocol['gross_station_rms_mm'] or m.get('max_abs_mm',0) > protocol['gross_max_abs_residual_mm'] for m in stations.values())
    verdict = 'UNKNOWN_INCOMPLETE_PREDICTION' if not complete else 'FAIL_GROSS_MODEL' if gross else 'PASS_GROSS_MODEL_ONLY'
    return {'schema': 'wb133_accepted_anchor_audit_v1', 'identity': export['identity'], 'source': source, 'interface': 'PASS',
            'scientific_verdict': verdict, 'complete_coverage': complete, 'valid_rows': sum(r['valid'] for r in records),
            'original_rows': len(records), 'official_calls': len(calls), 'zero_path_controls': controls,
            'rows': records, 'stations': stations, 'original_P_stations': baseline_stations,
            'gross_on_available_rows': gross, 'new_reconstruction_calls': 0, 'held_out_access': False, 'truth_access': False}
