#!/usr/bin/env python3
"""Frozen 24-call response matrix audit; all objectives are descriptive."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public

FACTORS = (0., .2, 1.)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def array(value, shape):
    v = np.asarray(value, dtype=float)
    check(v.shape == shape and np.isfinite(v).all(), 'finite/shape')
    return v


def objective(residual, delta4, c0, cs, scales):
    r = array(residual, (8,)); d4 = array(delta4, (4,)); scale = array(scales, (4,))
    check(np.all(scale > 0) and len(cs) == 2, 'positive scales/two targets')
    r = r/np.tile(scale, 2); d4 = d4/scale
    c0 = array(c0, (4, 4))/scale[:, None]/scale[None, :]
    d = np.zeros((8, 8))
    for i, c in enumerate(cs):
        d[4*i:4*i+4, 4*i:4*i+4] = array(c, (4, 4))/scale[:, None]/scale[None, :]
    check(np.allclose(c0, c0.T, rtol=1e-12, atol=1e-12) and np.allclose(d, d.T, rtol=1e-12, atol=1e-12), 'covariance symmetry')
    l0, ld = np.linalg.cholesky(c0), np.linalg.cholesky(d)
    prior = float(np.linalg.norm(np.linalg.solve(l0, d4))**2)
    data = float(np.linalg.norm(np.linalg.solve(ld, r))**2)
    independent = float(r@np.linalg.solve(d, r)+d4@np.linalg.solve(c0, d4))
    error = abs(prior+data-independent)/max(1., prior+data, abs(independent))
    check(error <= 1e-8, 'objective independent arithmetic')
    return {'total': prior+data, 'data': data, 'prior4': prior,
            'independent_solve': independent, 'relative_error': error,
            'qop_prior': 'NONE', 'p_value': 'NOT_EVALUATED'}


def analyze(runtime, control, fixture, historical):
    check(runtime['schema'] == 'wb118_bounded_response_runtime_v1' and runtime['control_used'] == control, 'runtime/control identity')
    expected_identity = {k: fixture[k] for k in ('actual_run', 'actual_event', 'input_xaod', 'ordinal')}
    check(runtime['identity'] == expected_identity and (fixture['index'], fixture['ordinal'], fixture['actual_run'], fixture['actual_event']) == (12, 2268, 100044, 2268), 'event identity')
    check(runtime['acts_MeV_unit'] == historical['acts_MeV_unit'] and runtime['acts_T_unit'] == historical['acts_T_unit'], 'ACTS unit identity')
    check(runtime['world'] == historical['world'] and runtime['field_conditions'] == historical['field_conditions'], 'World/field conditions identity')
    check([d['name'] for d in control['directions']] == ['full_four_d_only', 'full_qop_only', 'half_four_d_only', 'half_qop_only'] and control['factors'] == list(FACTORS), 'directions/factors')
    check([t['station'] for t in control['targets']] == [1, 2], 'target ordering')
    seed = array(control['seed'], (5, 1)).reshape(5)
    check(np.array_equal(seed, np.r_[array(fixture['references'][0]['fixed_z_state'], (4, 1)).reshape(4), fixture['references'][0]['q_over_p_per_MeV']]) and
          control['seed_z_mm'] == fixture['references'][0]['z_state_mm'], 'seed reference identity')
    scales = array(control['output_scales'], (4,)); check(np.all(scales > 0), 'output scales')
    rows = runtime['rows']; check(runtime['official_calls'] == len(rows) == 24, 'call count')
    sensors = runtime['sensors']; expected_sensors = [(r['station'], cid) for r in fixture['references'] for cid in r['clusters']]
    check(sensors == historical['sensors'], 'cross-job actual sensor transforms')
    check([(r['station'], r['strip']) for r in sensors] == expected_sensors, 'sensor allowlist/order')
    for r in sensors:
        check(type(r['wafer']) is int and r['wafer'] > 0 and type(r['geometry_id']) is int and r['geometry_id'] > 0, 'sensor identity types')
        t = array(r['transform'], (4, 4)); check(np.allclose(t[3], [0, 0, 0, 1], atol=1e-12, rtol=0) and
            np.allclose(t[:3, :3].T@t[:3, :3], np.eye(3), atol=1e-10, rtol=0), 'actual sensor transform')
    results = []; grouped = []; count = 0; successes = 0
    c0 = fixture['references'][0]['fixed_z_covariance']; cs = [fixture['references'][k]['fixed_z_covariance'] for k in (1, 2)]
    for direction in control['directions']:
        delta = array(direction['delta'], (5,))
        for factor in FACTORS:
            rs = []; linear_rs = []; missing = []
            for target in control['targets']:
                row = rows[count]; count += 1; inp = row['input']; station = target['station']
                expected = seed+factor*delta
                check(inp['call_id'] == count and inp['direction'] == direction['name'] and inp['factor'] == factor and inp['station'] == station, 'call ordering/identity')
                check(np.array_equal(array(inp['seed'], (5, 1)).reshape(5), expected) and inp['seed_z_mm'] == control['seed_z_mm'] and inp['frame'] == target['frame'], 'input seed/qop/frame/z')
                negative = direction['name'].endswith('qop_only') and factor == 1.
                role = 'CHARGE_SIGN_NEGATIVE_CONTROL_ONLY' if negative else 'POSITIVE_QOP_DIAGNOSTIC'
                check(inp['role'] == role and (expected[4] < 0 if negative else expected[4] > 0), 'charge role/sign')
                start = inp['start_state']; check(start['covariance_present'] is False and start['time_acts'] == 0., 'start covariance/time')
                check(start['qop_acts']*runtime['acts_MeV_unit'] == expected[4], 'qop roundtrip/unit')
                check(np.max(np.abs(array(start['h'], (4, 1)).reshape(4)-expected[:4])/np.r_[scales]) <= 1e-9, 'start bound roundtrip')
                check(np.allclose(array(start['global_position_mm'], (3, 1)).reshape(3), np.r_[expected[:2], control['seed_z_mm']], rtol=0, atol=1e-9), 'start global position')
                u = np.r_[expected[2:4], 1.]; u /= np.linalg.norm(u)
                check(np.allclose(array(start['global_direction'], (3, 1)).reshape(3), u, rtol=0, atol=1e-12), 'start direction')
                check(type(row['has_value']) is bool, 'optional type')
                h0 = array(target['official_h'], (4, 1)).reshape(4)
                jac = array(direction['jacobians'][str(station)], (4, 5))
                effect = factor*jac@delta; linear = h0+effect
                y = array(fixture['references'][station]['fixed_z_state'], (4, 1)).reshape(4)
                detail = {'direction': direction['name'], 'factor': factor, 'station': station,
                          'call_id': count, 'role': role, 'has_value': row['has_value'],
                          'seed': expected.tolist(), 'linear_h': linear.tolist(), 'linear_effect_scaled': (effect/scales).tolist()}
                linear_rs.append(linear-y)
                if not row['has_value']:
                    check(factor != 0. and 'state' not in row, 'zero guard/missing state')
                    missing.append(station); detail['physical_response'] = 'UNKNOWN_NO_OFFICIAL_STATE'
                else:
                    successes += 1; state = row['state']; h = array(state['h'], (4, 1)).reshape(4)
                    check(state['covariance_present'] is False and state['qop_acts'] == start['qop_acts'] and np.isfinite(state['time_acts']), 'returned qop/covariance/time')
                    frame = array(target['frame'], (4, 4)); pos = array(state['global_position_mm'], (3, 1)).reshape(3)
                    local = np.linalg.solve(frame, np.r_[pos, 1.])[:3]
                    check(np.allclose(local, array(state['local_position_mm'], (3, 1)).reshape(3), atol=1e-8, rtol=0) and abs(local[2]) <= 1e-6, 'returned frame/plane')
                    u = frame[:3, :3].T@array(state['global_direction'], (3, 1)).reshape(3)
                    check(abs(u[2]) >= 1e-12 and np.allclose(np.r_[local[:2], u[:2]/u[2]], h, atol=1e-8, rtol=0), 'returned h frame')
                    if factor == 0.: check(state['h'] == target['official_h'], 'nominal exact guard')
                    discrepancy = (h-linear)/scales
                    detail.update({'actual_h': h.tolist(), 'measurement_residual': (h-y).tolist(),
                                   'scaled_linearization_discrepancy': discrepancy.tolist(),
                                   'max_scaled_linearization_discrepancy': float(np.max(np.abs(discrepancy))),
                                   'relative_linearization_error': float(np.max(np.abs(discrepancy))/max(float(np.max(np.abs(effect/scales))), np.finfo(float).eps)) if factor else None})
                    rs.append(h-y)
                results.append(detail)
            group = {'direction': direction['name'], 'factor': factor, 'missing_stations': missing,
                     'role': role, 'linear_objective': objective(np.concatenate(linear_rs), factor*delta[:4], c0, cs, scales)}
            group['actual_objective'] = objective(np.concatenate(rs), factor*delta[:4], c0, cs, scales) if not missing else 'UNKNOWN_MISSING_OFFICIAL_STATE'
            grouped.append(group)
    return {'schema': 'wb118_bounded_response_summary_v1', 'integrity_gate': 'PASS',
            'classification': 'DIAGNOSTIC_ONLY_RESPONSE_DECOMPOSITION',
            'response_pattern': 'NONLINEAR_RESPONSE_MEASURED' if successes == 24 else 'INCOMPLETE_OFFICIAL_RESPONSE',
            'identity': expected_identity, 'official_calls': 24, 'successful_calls': successes,
            'nominal_repeat_controls': 8, 'nominal_exact_guards': 'PASS', 'rows': results, 'joint_objectives': grouped,
            'station3': 'NOT_CALLED_HISTORICAL_FAILURE_UNCHANGED', 'covariance_calibration': 'UNVERIFIED',
            'association': 'UNKNOWN_OR_AMBIGUOUS_WB114', 'historical_generation_conditions': 'UNKNOWN',
            'truth_momentum_used': False, 'dummy_qop_variance_used': False, 'held_out_access': False,
            'new_reconstruction_calls': 0, 'new_official_propagation_calls': 24,
            'qualification': 'NOT_EVALUATED', 'physics_screening': 'NOT_EVALUATED', 'final_oracle': 'NOT_EVALUATED'}


def audit_split(out):
    runtime = read_public(out/'event/response.json')
    summary = analyze(runtime, read_public(out/'control.json'), read_public(out/'fixture.json'), read_public(out/'historical_runtime.json'))
    lines = [json.loads(line) for line in (out/'event/response.json.calls.ndjson').read_text().splitlines()]
    check(len(lines) == 49 and lines[-1] == {'record': 'terminal', 'status': 'COMPLETED', 'official_calls': 24}, 'stream complete')
    for i, row in enumerate(runtime['rows']):
        check(lines[2*i] == {'record': 'before_official', 'input': row['input']} and
              lines[2*i+1] == {'record': 'after_official', 'row': row}, 'stream/runtime identity')
    return summary


def decompose(summary, control, fixture, historical):
    """Signed finite-change interaction; no thresholds assigning physical causes."""
    joint_control = historical['control_used']
    from audit_wb117_response import analyze as audit_joint_runtime
    audit_joint_runtime(historical, joint_control, fixture, historical)
    check(joint_control['factors'] == [0., .1, .2, 1.] and historical['official_calls'] == 16, 'joint frozen matrix')
    check(summary['identity'] == historical['identity'] and
          control['seed'] == joint_control['seed'] and control['seed_z_mm'] == joint_control['seed_z_mm'] and
          control['targets'] == joint_control['targets'] and control['output_scales'] == joint_control['output_scales'] and
          control['original_FD_steps'] == joint_control['original_FD_steps'], 'cross-job controls')
    for direction in control['directions']:
        parent, split = direction['parent_direction'], direction['split']
        check(split in ('four_d_only', 'qop_only') and direction['name'] == parent+'_'+split, 'split identity')
        previous = next(d for d in joint_control['directions'] if d['name'] == parent)
        expected_delta = previous['delta'][:4]+[0.] if split == 'four_d_only' else [0.]*4+[previous['delta'][4]]
        check(direction['delta'] == expected_delta and direction['jacobians'] == previous['jacobians'], 'split delta/J lineage')
    old = {(r['input']['direction'], r['input']['factor'], r['input']['station']): r for r in historical['rows']}
    check(len(old) == 16, 'joint duplicate matrix')
    new = {(r['direction'], r['factor'], r['station']): r for r in summary['rows']}
    check(len(new) == 24, 'split duplicate matrix')
    scales = array(control['output_scales'], (4,)); rows = []; objectives = []
    c0 = fixture['references'][0]['fixed_z_covariance']; cs = [fixture['references'][st]['fixed_z_covariance'] for st in (1, 2)]
    for direction in joint_control['directions']:
        name = direction['name']; delta = array(direction['delta'], (5,))
        for factor in FACTORS:
            residuals = []; absent = []
            for target in control['targets']:
                station = target['station']; key = (name, factor, station)
                joint = old[key]; four = new[(name+'_four_d_only', factor, station)]; qop = new[(name+'_qop_only', factor, station)]
                h0 = array(target['official_h'], (4, 1)).reshape(4)
                row = {'direction': name, 'factor': factor, 'station': station,
                       'joint_source_call_id': joint['input']['call_id'], 'four_d_call_id': four['call_id'],
                       'qop_call_id': qop['call_id'], 'status': 'UNKNOWN_INCOMPLETE_RESPONSE'}
                if not joint['has_value'] or not four['has_value'] or not qop['has_value']:
                    rows.append(row); absent.append(station); continue
                hj = array(joint['state']['h'], (4, 1)).reshape(4)
                h4 = array(four['actual_h'], (4,)); hq = array(qop['actual_h'], (4,))
                jac = array(direction['jacobians'][str(station)], (4, 5))
                lj4 = factor*(jac[:, :4]@delta[:4]); ljq = factor*jac[:, 4]*delta[4]
                e4 = h4-h0-lj4; eq = hq-h0-ljq; interaction = hj-h4-hq+h0
                ej = hj-h0-lj4-ljq; sum_e = e4+eq+interaction
                err = float(np.max(np.abs((ej-sum_e)/scales))/max(1.,float(np.max(np.abs(ej/scales))),float(np.max(np.abs(sum_e/scales)))))
                check(err <= 1e-8, 'signed decomposition arithmetic')
                row.update({'status': 'SIGNED_DECOMPOSITION_MEASURED', 'h0': h0.tolist(), 'h_joint': hj.tolist(),
                            'h_four_d_only': h4.tolist(), 'h_qop_only': hq.tolist(),
                            'E_joint_scaled': (ej/scales).tolist(), 'E_four_d_scaled': (e4/scales).tolist(),
                            'E_qop_scaled': (eq/scales).tolist(), 'I_actual_scaled': (interaction/scales).tolist(),
                            'signed_sum_scaled': (sum_e/scales).tolist(), 'identity_relative_error': err,
                            'component_max_abs_scaled': {k: float(np.max(np.abs(v/scales))) for k, v in
                                (('joint', ej), ('four_d', e4), ('qop', eq), ('interaction', interaction))},
                            'cause_classification': 'NOT_ASSIGNED_UNCALIBRATED_FINITE_CHANGE'})
                residuals.append(hj-array(fixture['references'][station]['fixed_z_state'], (4, 1)).reshape(4))
                rows.append(row)
            objectives.append({'direction': name, 'factor': factor, 'missing_stations': absent,
                'joint_saved_actual_objective': objective(np.concatenate(residuals),factor*delta[:4],c0,cs,scales) if not absent else 'UNKNOWN_INCOMPLETE_RESPONSE'})
    summary = dict(summary)
    summary.update({'decomposition': rows, 'saved_joint_objectives': objectives,
                    'decomposition_pattern': 'SIGNED_DECOMPOSITION_MEASURED' if all(r['status']=='SIGNED_DECOMPOSITION_MEASURED' for r in rows) else 'UNKNOWN_INCOMPLETE_RESPONSE',
                    'joint_new_calls': 0, 'signed_component_labels_are_not_causal_probabilities': True,
                    'cross_job_sensor_transforms': 'EXACT_RUNTIME_IDENTITY_REQUIRED'})
    return summary


def audit(out):
    summary = audit_split(out)
    return decompose(summary, read_public(out/'control.json'), read_public(out/'fixture.json'), read_public(out/'historical_runtime.json'))
