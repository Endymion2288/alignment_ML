"""Synthetic nonlinear responses and identity mutation controls; no event calls."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb117_response import analyze, objective, FACTORS


def state(h, z, qop, mev=.001, time=0.):
    h = np.asarray(h); u = np.r_[h[2:], 1.]; u /= np.linalg.norm(u)
    return {'h': h[:, None].tolist(), 'local_position_mm': np.r_[h[:2], 0.][:, None].tolist(),
            'global_position_mm': np.r_[h[:2], z][:, None].tolist(), 'global_direction': u[:, None].tolist(),
            'qop_acts': qop/mev, 'time_acts': time, 'covariance_present': False}


def synthetic(nonlinear=False):
    seed = np.r_[np.zeros(4), 1e-5]; factors = list(FACTORS)
    refs = [{'station': k, 'clusters': [100+k], 'fixed_z_state': np.zeros((4, 1)).tolist(),
             'fixed_z_covariance': np.eye(4).tolist(), 'z_state_mm': k*10., 'q_over_p_per_MeV': 1e-5} for k in range(4)]
    fixture = {'index': 12, 'ordinal': 2268, 'actual_run': 100044, 'actual_event': 2268,
               'input_xaod': '/allowed/seen.root', 'references': refs}
    targets = []
    for k in (1, 2):
        frame = np.eye(4); frame[2, 3] = k*10.
        targets.append({'station': k, 'frame': frame.tolist(), 'official_h': np.zeros((4, 1)).tolist()})
    jac = np.c_[np.eye(4), [1e4, 2e4, 3e2, 4e2]]
    directions = [{'name': n, 'delta': [d, .2, .01, .02, -3e-5],
                   'jacobians': {'1': jac.tolist(), '2': (2*jac).tolist()}} for n, d in (('full', 1.), ('half', .8))]
    control = {'seed': seed[:, None].tolist(), 'seed_z_mm': 0., 'targets': targets, 'directions': directions,
               'factors': factors, 'output_scales': [1., 1., .001, .001]}
    historical = {'acts_MeV_unit': .001, 'acts_T_unit': 1., 'world': {'name': 'actualSyntheticWorld'},
                  'field_conditions': {'scale': 1., 'official_context_is_actual_cache': True}}
    runtime = dict(copy.deepcopy(historical), schema='wb117_bounded_response_runtime_v1',
                   control_used=copy.deepcopy(control), identity={k: fixture[k] for k in ('actual_run', 'actual_event', 'input_xaod', 'ordinal')},
                   official_calls=16, rows=[], sensors=[{'station': k, 'strip': 100+k, 'wafer': 10+k,
                     'geometry_id': 20+k, 'transform': np.eye(4).tolist()} for k in range(4)])
    for direction in directions:
        delta = np.asarray(direction['delta'])
        for factor in factors:
            for target in targets:
                x = seed+factor*delta; station = target['station']
                h = factor*np.asarray(direction['jacobians'][str(station)])@delta
                if nonlinear: h += factor**2*np.array([.3, .2, .005, .007])
                runtime['rows'].append({'input': {'call_id': len(runtime['rows'])+1, 'direction': direction['name'],
                    'factor': factor, 'station': station, 'seed': x[:, None].tolist(), 'seed_z_mm': 0.,
                    'frame': copy.deepcopy(target['frame']),
                    'role': 'CHARGE_SIGN_NEGATIVE_CONTROL_ONLY' if factor == 1. else 'POSITIVE_QOP_DIAGNOSTIC',
                    'start_state': state(x[:4], 0., x[4])}, 'has_value': True,
                    'state': state(h, station*10., x[4], time=1.)})
    return runtime, control, fixture, historical


def test_linear_operator_has_zero_discrepancy_and_equal_actual_linear_objective():
    s = analyze(*synthetic())
    assert s['successful_calls'] == 16 and s['nominal_repeat_controls'] == 4
    assert s['response_pattern'] == 'NONLINEAR_RESPONSE_MEASURED'
    for row in s['rows']: assert row['max_scaled_linearization_discrepancy'] < 1e-12
    for row in s['joint_objectives']:
        assert row['actual_objective']['total'] == pytest.approx(row['linear_objective']['total'])
        assert row['actual_objective']['qop_prior'] == 'NONE'
    assert s['qualification'] == 'NOT_EVALUATED' and s['association'].startswith('UNKNOWN')


def test_quadratic_response_reports_deviation_without_physics_pass():
    s = analyze(*synthetic(nonlinear=True))
    rows = [x for x in s['rows'] if x['factor'] == .2]
    assert rows[0]['scaled_linearization_discrepancy'] == pytest.approx([.012, .008, .2, .28])
    assert all(x['relative_linearization_error'] > 0 for x in rows)
    assert s['classification'] == 'DIAGNOSTIC_ONLY_UNCALIBRATED_NONLINEAR_RESPONSE'


def test_missing_nonzero_official_state_is_preserved_not_filled():
    r, c, f, h = synthetic(); row = r['rows'][2]; row['has_value'] = False; row.pop('state')
    s = analyze(r, c, f, h)
    assert s['successful_calls'] == 15 and s['response_pattern'] == 'INCOMPLETE_OFFICIAL_RESPONSE'
    assert s['joint_objectives'][1]['actual_objective'] == 'UNKNOWN_MISSING_OFFICIAL_STATE'
    assert s['joint_objectives'][1]['missing_stations'] == [1]


def test_objective_matches_independent_correlated_covariance_solution():
    c = np.eye(4); c[0, 1] = c[1, 0] = .4
    r = np.arange(8)*.1; delta = np.arange(4)*.2
    d = np.zeros((8, 8)); d[:4, :4] = c; d[4:, 4:] = 2*c
    expected = r@np.linalg.solve(d, r)+delta@np.linalg.solve(c, delta)
    p = objective(r, delta, c, [c, 2*c], [1., 1., .001, .001])
    assert p['total'] == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize('bad', ['source', 'header', 'units', 'world', 'field', 'control', 'callcount', 'duplicate_call',
    'station', 'factor', 'seed_other', 'seed_qop', 'frame', 'negative_role', 'qop_units', 'start_direction',
    'start_position', 'covariance', 'missing_zero', 'nominal', 'returned_frame', 'returned_qop', 'nan',
    'sensor_id', 'sensor_transform', 'direction_order', 'factors'])
def test_identity_units_frame_and_charge_mutations_fail_closed(bad):
    r, c, f, h = synthetic(); row = r['rows'][2]
    if bad == 'source': r['identity']['input_xaod'] = '/wrong.root'
    elif bad == 'header': r['identity']['actual_event'] += 1
    elif bad == 'units': r['acts_MeV_unit'] *= 1000.
    elif bad == 'world': r['world']['name'] = 'wrong'
    elif bad == 'field': r['field_conditions']['scale'] = -1.
    elif bad == 'control': r['control_used']['seed'][0][0] += 1.
    elif bad == 'callcount': r['rows'].pop()
    elif bad == 'duplicate_call': row['input']['call_id'] = 1
    elif bad == 'station': row['input']['station'] = 2
    elif bad == 'factor': row['input']['factor'] = .3
    elif bad == 'seed_other': row['input']['seed'][0][0] += 1.
    elif bad == 'seed_qop': row['input']['seed'][4][0] *= 1000.
    elif bad == 'frame': row['input']['frame'][2][3] += 1.
    elif bad == 'negative_role': r['rows'][6]['input']['role'] = 'POSITIVE_QOP_DIAGNOSTIC'
    elif bad == 'qop_units': row['input']['start_state']['qop_acts'] *= 1000.
    elif bad == 'start_direction': row['input']['start_state']['global_direction'][0][0] += 1.
    elif bad == 'start_position': row['input']['start_state']['global_position_mm'][2][0] += 1.
    elif bad == 'covariance': row['input']['start_state']['covariance_present'] = True
    elif bad == 'missing_zero': r['rows'][0]['has_value'] = False; r['rows'][0].pop('state')
    elif bad == 'nominal': r['rows'][0]['state']['h'][0][0] += 1.
    elif bad == 'returned_frame': row['state']['global_position_mm'][2][0] += 1.
    elif bad == 'returned_qop': row['state']['qop_acts'] *= -1.
    elif bad == 'nan': row['state']['h'][0][0] = np.nan
    elif bad == 'sensor_id': r['sensors'][0]['strip'] += 1
    elif bad == 'sensor_transform': r['sensors'][0]['transform'][0][0] = 2.
    elif bad == 'direction_order': c['directions'].reverse(); r['control_used'] = copy.deepcopy(c)
    elif bad == 'factors': c['factors'][1] = .3; r['control_used'] = copy.deepcopy(c)
    with pytest.raises((ValueError, np.linalg.LinAlgError)): analyze(r, c, f, h)


@pytest.mark.parametrize('bad', ['negative', 'singular', 'asymmetric', 'nan', 'zero_scale'])
def test_objective_rejects_invalid_covariance_without_floor(bad):
    c = np.eye(4); scales = np.ones(4)
    if bad == 'negative': c[0, 0] = -1.
    elif bad == 'singular': c[0, 0] = 0.
    elif bad == 'asymmetric': c[0, 1] = .2
    elif bad == 'nan': c[0, 0] = np.nan
    elif bad == 'zero_scale': scales[0] = 0.
    with pytest.raises((ValueError, np.linalg.LinAlgError)): objective(np.ones(8), np.ones(4), c, [np.eye(4)]*2, scales)
