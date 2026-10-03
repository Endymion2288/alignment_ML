#!/usr/bin/env python3
"""Saved official logger snapshots; no end-state or causal inference from logs."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import read_public
from audit_wb117_response import check, array

APIS = ('propagate', 'propagationSteps')
ORDER = [(0., 0)] + [(m, sign) for m in (.25, .5, 1., 2.) for sign in (1, -1)]
CONSTRAINTS = ('actor_mm', 'aborter_mm', 'user_mm', 'accuracy_mm', 'effective_mm')


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def compress(sequence):
    out = []
    for item in sequence:
        if not out or item != out[-1]:
            out.append(item)
    return out


def compare_snapshots(a, b):
    ga = [(s['geometry_id'], s['surface_present']) for s in a]
    gb = [(s['geometry_id'], s['surface_present']) for s in b]
    ra = [s['rk_rejections_before_previous_accepted_step'] for s in a]
    rb = [s['rk_rejections_before_previous_accepted_step'] for s in b]
    ca = [s['step_constraints'] for s in a]; cb = [s['step_constraints'] for s in b]
    count_same = len(a) == len(b); geom_same = ga == gb; retry_same = ra == rb
    result = {'snapshot_count_difference': len(b)-len(a), 'geometry_sequence_exact': geom_same,
              'compressed_geometry_sequence_exact': compress(ga) == compress(gb),
              'first_geometry_difference_index': first_difference(ga, gb),
              'rk_rejection_sequence_exact': retry_same, 'first_rk_difference_index': first_difference(ra, rb),
              'constraints_sequence_exact': ca == cb, 'first_constraints_difference_index': first_difference(ca, cb),
              'discrete_branch_difference': not (count_same and geom_same and retry_same),
              'state_comparison': 'UNKNOWN_UNALIGNED_SNAPSHOTS'}
    if count_same and geom_same:
        positions = [np.abs(array(y['global_position_mm'], (3, 1)) - array(x['global_position_mm'], (3, 1))).reshape(3) for x, y in zip(a, b)]
        directions = [np.abs(array(y['global_direction'], (3, 1)) - array(x['global_direction'], (3, 1))).reshape(3) for x, y in zip(a, b)]
        result.update(state_comparison='ORDINAL_SNAPSHOT_COMPARISON_NOT_EQUAL_ARC',
                      max_abs_global_position_difference_mm=np.max(positions, axis=0).tolist(),
                      max_abs_global_direction_difference=np.max(directions, axis=0).tolist())
    return result


def validate_snapshot(s, i, nav):
    check(s['snapshot_index'] == i and s['navigation_direction'] == nav, 'snapshot identity/direction')
    p = array(s['momentum_MeV'], (3, 1)).reshape(3); u = array(s['global_direction'], (3, 1)).reshape(3)
    array(s['global_position_mm'], (3, 1))
    check(np.linalg.norm(p) > 0 and np.allclose(u, p/np.linalg.norm(p), rtol=0, atol=1e-12), 'snapshot momentum/direction')
    check(type(s['geometry_id']) is int and s['geometry_id'] >= 0 and type(s['surface_present']) is bool, 'snapshot geometry identity')
    check(s['surface_geometry_id'] == s['geometry_id'] if s['surface_present'] else s['surface_geometry_id'] is None, 'snapshot surface identity')
    check(type(s['rk_counter_unset']) is bool, 'RK sentinel type')
    n = s['rk_rejections_before_previous_accepted_step']
    check(n is None if s['rk_counter_unset'] else type(n) is int and n >= 0, 'RK sentinel/counter')
    check(s['rk_counter_unset'] is (i == 0), 'initial-only unset RK counter')
    c = s['step_constraints']; check(set(c) == set(CONSTRAINTS), 'constraint keys')
    for name, value in c.items():
        check(value is None or type(value) in (int, float) and np.isfinite(value) and abs(value) < np.finfo(float).max, 'constraint finite/unset')
    check(c['user_mm'] == 10000. and c['effective_mm'] is not None, 'unchanged production user cap')
    if c['accuracy_mm'] is not None: check(c['accuracy_mm'] > 0, 'positive accuracy constraint')
    v = min(np.finfo(float).max if c[name] is None else c[name] for name in ('actor_mm', 'aborter_mm', 'user_mm'))
    accuracy = np.finfo(float).max if c['accuracy_mm'] is None else c['accuracy_mm']
    effective = (-1. if np.signbit(v) else 1.) * min(abs(v), accuracy)
    check(c['effective_mm'] == effective, 'ConstrainedStep snapshot formula')


def analyze(runtime, control, fixture, historical, old_rows):
    check(runtime['schema'] == 'wb120_stepping_runtime_v1' and runtime['control_used'] == control, 'schema/control')
    check((fixture['index'], fixture['ordinal'], fixture['actual_run'], fixture['actual_event']) == (12, 2268, 100044, 2268), 'single seen event')
    check(runtime['identity'] == historical['identity'] == {k: fixture[k] for k in ('actual_run', 'actual_event', 'input_xaod', 'ordinal')}, 'event identity')
    for key in ('world', 'field_conditions', 'sensors', 'acts_MeV_unit', 'acts_T_unit'):
        check(runtime[key] == historical[key], 'cross-job '+key)
    previous_control = historical['control_used']
    for key in ('seed', 'seed_z_mm', 'targets', 'output_scales', 'qop_step_per_MeV'):
        check(control[key] == previous_control[key], 'nominal control '+key)
    seed = array(control['seed'], (5, 1)).reshape(5); scales = array(control['output_scales'], (4,))
    check(seed[4] == 1e-5 and control['qop_step_per_MeV'] == 1e-8 and np.all(scales > 0), 'original units/step')
    check(len(control['samples']) == 9 and runtime['official_calls'] == len(runtime['rows']) == 36, 'call matrix')
    check(control['max_official_calls'] == 36 and control['plain_calls'] == control['trace_calls'] == 18 and control['snapshot_budget_per_call'] == 10002, 'call/snapshot budget')
    old = {(r['sample_id'], r['station']): r for r in old_rows}; check(len(old) == len(old_rows) == 18, 'historical matrix')
    traces = {}; responses = []; profiles = []; missing = []; empty = []
    for sample_id, (multiple, sign) in enumerate(ORDER):
        sample = control['samples'][sample_id]; x = seed.copy(); x[4] += multiple*sign*control['qop_step_per_MeV']
        check((sample['sample_id'], sample['step_multiple'], sample['sign']) == (sample_id, multiple, sign) and
              sample['offset_multiple'] == multiple*sign and np.array_equal(array(sample['seed'], (5,)), x) and x[4] > 0, 'sample identity/seed')
        for ti, target in enumerate(control['targets']):
            station = ti+1; check(target['station'] == station, 'target order')
            reference = old[sample_id, station]; oi = reference['historical_input']
            check(oi['station'] == station and oi['step_multiple'] == multiple and oi['sign'] == sign and oi['repeat'] == 0 and
                  oi['call_id'] == reference['historical_call_id'] and oi['frame'] == target['frame'] and oi['seed_z_mm'] == control['seed_z_mm'] and
                  np.array_equal(array(oi['seed'], (5, 1)).reshape(5), x), 'historical input')
            check(historical['rows'][reference['historical_call_id']-1]['input'] == oi and historical['rows'][reference['historical_call_id']-1]['state']['h'] == reference['h'], 'actual historical source')
            frame = array(target['frame'], (4, 4)); u = np.r_[x[2:4], 1.]; u /= np.linalg.norm(u)
            nav = 1 if np.dot(frame[:3, 3]-np.r_[x[:2], control['seed_z_mm']], u) >= 0 else -1
            for ai, api in enumerate(APIS):
                call = 4*sample_id+2*ti+ai+1; row = runtime['rows'][call-1]; inp = row['input']; q = x[4]/runtime['acts_MeV_unit']
                check((inp['call_id'], inp['api'], inp['sample_id'], inp['station'], inp['step_multiple'], inp['sign']) == (call, api, sample_id, station, multiple, sign), 'call/API ordering')
                check(inp['factor'] == multiple*sign and inp['seed_z_mm'] == control['seed_z_mm'] and inp['frame'] == target['frame'] and
                      inp['role'] == 'POSITIVE_QOP_DIAGNOSTIC' and inp['navigation_direction'] == nav and
                      np.array_equal(array(inp['seed'], (5, 1)).reshape(5), x), 'call input identity')
                start = inp['start_state']
                check(start['covariance_present'] is False and start['time_acts'] == 0. and start['qop_acts'] == q and
                      np.max(np.abs(array(start['h'], (4, 1)).reshape(4)-x[:4])/scales) <= 1e-9 and
                      np.allclose(array(start['global_position_mm'], (3, 1)).reshape(3), np.r_[x[:2], control['seed_z_mm']], rtol=0, atol=1e-9) and
                      np.allclose(array(start['global_direction'], (3, 1)).reshape(3), u, rtol=0, atol=1e-12), 'start roundtrip/units')
                if api == 'propagate':
                    check(type(row['has_value']) is bool, 'optional type')
                    detail = {'call_id': call, 'sample_id': sample_id, 'station': station, 'step_multiple': multiple, 'sign': sign,
                              'historical_call_id': reference['historical_call_id'], 'exact': 'UNKNOWN_MISSING_STATE'}
                    if not row['has_value']:
                        check(multiple != 0 and 'state' not in row, 'nominal missing guard'); missing.append(call)
                    else:
                        state = row['state']; h = array(state['h'], (4, 1)).reshape(4)
                        check(state['qop_acts'] == q and state['covariance_present'] is False and np.isfinite(state['time_acts']), 'returned qop/covariance')
                        local = np.linalg.solve(frame, np.r_[array(state['global_position_mm'], (3, 1)).reshape(3), 1.])[:3]
                        direction = frame[:3, :3].T @ array(state['global_direction'], (3, 1)).reshape(3)
                        check(abs(local[2]) <= 1e-6 and abs(direction[2]) >= 1e-12 and
                              np.allclose(local, array(state['local_position_mm'], (3, 1)).reshape(3), rtol=0, atol=1e-8) and
                              np.allclose(np.r_[local[:2], direction[:2]/direction[2]], h, rtol=0, atol=1e-8), 'returned frame')
                        if multiple == 0: check(state['h'] == target['official_h'], 'nominal exact')
                        detail.update(exact=state['h'] == reference['h'], h=h.tolist(),
                                      scaled_difference=((h-array(reference['h'], (4, 1)).reshape(4))/scales).tolist())
                    responses.append(detail)
                else:
                    snaps = row['snapshots']; check(len(snaps) == row['snapshot_count'] <= 10002 and row['target_endpoint_exported'] is False and 'state' not in row, 'trace length/no target substitution')
                    check(row['trace_status'] == ('SNAPSHOTS_RETURNED' if snaps else 'UNKNOWN_EMPTY_TRACE'), 'trace status')
                    if not snaps: empty.append(call); continue
                    for i, snap in enumerate(snaps): validate_snapshot(snap, i, nav)
                    first = snaps[0]
                    check(np.allclose(array(first['global_position_mm'], (3, 1)), array(start['global_position_mm'], (3, 1)), rtol=0, atol=1e-9) and
                          np.allclose(array(first['global_direction'], (3, 1)), array(start['global_direction'], (3, 1)), rtol=0, atol=1e-12) and
                          np.isclose(np.linalg.norm(array(first['momentum_MeV'], (3, 1))), 1/x[4], rtol=1e-12, atol=0), 'trace initial units/state')
                    traces[station, multiple, sign] = snaps
                    seq = [[v['geometry_id'], v['surface_present']] for v in snaps]
                    profiles.append({'call_id': call, 'sample_id': sample_id, 'station': station, 'step_multiple': multiple, 'sign': sign,
                                     'snapshot_count': len(snaps), 'geometry_sequence': seq, 'compressed_geometry_sequence': compress(seq),
                                     'rk_rejection_sequence': [v['rk_rejections_before_previous_accepted_step'] for v in snaps],
                                     'snapshot_constraints_are_actual_accepted_lengths': False, 'target_endpoint_exported': False})
    comparisons = []
    for station in (1, 2):
        pairs = [('versus_nominal', (0., 0), key) for key in ORDER[1:]]
        pairs += [('plus_minus', (m, -1), (m, 1)) for m in (.25, .5, 1., 2.)]
        pairs += [('versus_full_step', (1., sign), (m, sign)) for sign in (1, -1) for m in (.25, .5, 2.)]
        for role, a, b in pairs:
            result = {'station': station, 'role': role, 'reference_step_sign': list(a), 'compared_step_sign': list(b), 'status': 'UNKNOWN_MISSING_TRACE'}
            if (station, *a) in traces and (station, *b) in traces:
                result.update(status='SNAPSHOT_COMPARISON_MEASURED', **compare_snapshots(traces[station, *a], traces[station, *b]))
            comparisons.append(result)
    nonexact = any(r['exact'] is False for r in responses)
    discrete = any(r.get('discrete_branch_difference', False) for r in comparisons)
    pattern = ('INCOMPLETE_OFFICIAL_OR_TRACE_RESPONSE' if missing or empty else 'RESPONSE_REPRODUCTION_DIFFERENCE' if nonexact else
               'OBSERVABLE_BRANCH_DIFFERENCE_MEASURED' if discrete else 'NO_OBSERVABLE_BRANCH_DIFFERENCE_IN_LOGGED_SNAPSHOTS')
    station_patterns = {str(st): 'UNKNOWN_INCOMPLETE_OR_PLAIN_RESPONSE_DIFFERENCE' if missing or empty or nonexact else
                        'OBSERVED' if any(v.get('discrete_branch_difference', False) for v in comparisons if v['station'] == st) else 'NOT_OBSERVED_IN_LOGGED_SNAPSHOTS' for st in (1, 2)}
    return {'schema': 'wb120_stepping_summary_v1', 'integrity_gate': 'PASS', 'classification': 'DIAGNOSTIC_ONLY_OFFICIAL_STEPPING_SNAPSHOTS',
            'response_pattern': pattern, 'station_observable_branch_status': station_patterns, 'official_calls': 36, 'plain_calls': 18, 'trace_calls': 18,
            'successful_plain_calls': 18-len(missing), 'nonempty_trace_calls': 18-len(empty), 'nominal_exact_guards': 2,
            'missing_plain_calls': missing, 'empty_trace_calls': empty, 'plain_response_checks': responses, 'trace_profiles': profiles,
            'comparisons': comparisons, 'logger_endpoint_fidelity': 'UNKNOWN_NOT_EXPORTED_BY_PUBLIC_API',
            'causal_attribution': 'UNVERIFIED', 'production_step_selected': False, 'production_tool_changed': False,
            'truth_momentum_used': False, 'held_out_access': False, 'new_reconstruction_calls': 0,
            'association': 'UNKNOWN_OR_AMBIGUOUS_WB114', 'covariance_calibration': 'UNVERIFIED', 'historical_generation_conditions': 'UNKNOWN',
            'qualification': 'NOT_EVALUATED', 'physics_screening': 'NOT_EVALUATED', 'final_oracle': 'NOT_EVALUATED'}


def audit(out):
    runtime = read_public(out/'event/response.json'); stream = [json.loads(l) for l in (out/'event/response.json.calls.ndjson').read_text().splitlines()]
    check(len(stream) == 73 and stream[-1] == {'record': 'terminal', 'status': 'COMPLETED', 'official_calls': 36}, 'complete callstream')
    for i, row in enumerate(runtime['rows']):
        check(stream[2*i] == {'record': 'before_official', 'input': row['input']} and stream[2*i+1] == {'record': 'after_official', 'row': row}, 'stream/runtime identity')
    return analyze(runtime, read_public(out/'control.json'), read_public(out/'fixture.json'), read_public(out/'historical_runtime.json'), read_public(out/'historical_responses.json'))
