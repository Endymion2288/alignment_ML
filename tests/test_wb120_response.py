"""Public logger semantics, source integration and scientific identity guards."""
import copy
import re
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb120_response import analyze, ORDER, compare_snapshots
from wb120_sources import files, athena_source
from wb120_contract import WORKBOOK, OUT
from test_wb119_response import synthetic as previous_synthetic


def synthetic():
    prior, oldc, fixture, _, _ = previous_synthetic()
    c = {k: copy.deepcopy(oldc[k]) for k in ('seed', 'seed_z_mm', 'targets', 'qop_step_per_MeV', 'output_scales')}
    c.update(samples=[], max_official_calls=36, plain_calls=18, trace_calls=18, snapshot_budget_per_call=10002)
    r = {k: copy.deepcopy(prior[k]) for k in ('identity', 'world', 'field_conditions', 'sensors', 'acts_MeV_unit', 'acts_T_unit')}
    r.update(schema='wb120_stepping_runtime_v1', control_used=c, rows=[], official_calls=36)
    old = []
    for sample_id, (multiple, sign) in enumerate(ORDER):
        sample = next(x for x in oldc['samples'] if (x['step_multiple'], x['sign'], x['repeat']) == (multiple, sign, 0))
        c['samples'].append({k: copy.deepcopy(sample[k]) for k in ('seed', 'step_multiple', 'sign', 'offset_multiple')})
        c['samples'][-1]['sample_id'] = sample_id
        for station in (1, 2):
            previous = next(x for x in prior['rows'] if (x['input']['step_multiple'], x['input']['sign'], x['input']['repeat'], x['input']['station']) == (multiple, sign, 0, station))
            old.append({'sample_id': sample_id, 'station': station, 'historical_input': copy.deepcopy(previous['input']),
                        'historical_call_id': previous['input']['call_id'], 'h': copy.deepcopy(previous['state']['h'])})
            for api in ('propagate', 'propagationSteps'):
                row = copy.deepcopy(previous); inp = row['input']; inp.pop('repeat'); inp.pop('direction')
                inp.update(api=api, call_id=len(r['rows'])+1, sample_id=sample_id, navigation_direction=1)
                if api == 'propagationSteps':
                    row.pop('state'); row.pop('has_value'); snaps = []
                    u = np.asarray(inp['start_state']['global_direction']); p = u/sample['seed'][4]
                    for i in range(3):
                        position = np.array([[sample['seed'][0]], [sample['seed'][1]], [i*station*10./3]])
                        constraints = {'actor_mm': None, 'aborter_mm': None, 'user_mm': 10000.,
                                       'accuracy_mm': None if i == 0 else 20., 'effective_mm': 10000. if i == 0 else 20.}
                        snaps.append({'snapshot_index': i, 'global_position_mm': position.tolist(), 'momentum_MeV': p.tolist(),
                                      'global_direction': u.tolist(), 'navigation_direction': 1, 'geometry_id': 0 if i == 0 else 100+station,
                                      'surface_present': i == 0, 'surface_geometry_id': 0 if i == 0 else None,
                                      'rk_counter_unset': i == 0, 'rk_rejections_before_previous_accepted_step': None if i == 0 else 0,
                                      'step_constraints': constraints})
                    row.update(snapshots=snaps, snapshot_count=3, trace_status='SNAPSHOTS_RETURNED', target_endpoint_exported=False)
                r['rows'].append(row)
    r['control_used'] = copy.deepcopy(c)
    return r, c, fixture, prior, old


def test_linear_response_with_equal_discrete_histories_does_not_claim_causality_or_endpoint():
    s = analyze(*synthetic())
    assert s['official_calls'] == 36 and s['plain_calls'] == s['trace_calls'] == 18
    assert s['nonempty_trace_calls'] == 18 and len(s['comparisons']) == 36
    assert all(x['exact'] for x in s['plain_response_checks'])
    assert s['response_pattern'] == 'NO_OBSERVABLE_BRANCH_DIFFERENCE_IN_LOGGED_SNAPSHOTS'
    assert all(x['state_comparison'] == 'ORDINAL_SNAPSHOT_COMPARISON_NOT_EQUAL_ARC' for x in s['comparisons'])
    assert s['logger_endpoint_fidelity'] == 'UNKNOWN_NOT_EXPORTED_BY_PUBLIC_API' and s['causal_attribution'] == 'UNVERIFIED'
    assert not s['production_step_selected'] and s['qualification'] == 'NOT_EVALUATED'


def test_retry_branch_change_is_observable_even_when_geometry_sequence_is_identical():
    data = synthetic(); data[0]['rows'][7]['snapshots'][1]['rk_rejections_before_previous_accepted_step'] = 2
    s = analyze(*data)
    assert s['response_pattern'] == 'OBSERVABLE_BRANCH_DIFFERENCE_MEASURED'
    assert s['station_observable_branch_status']['2'] == 'OBSERVED'
    assert s['station_observable_branch_status']['1'] == 'NOT_OBSERVED_IN_LOGGED_SNAPSHOTS'
    assert any(x['first_rk_difference_index'] == 1 and x['geometry_sequence_exact'] for x in s['comparisons'])


def test_geometry_change_preserves_all_data_and_prevents_ordinal_state_matching():
    data = synthetic(); data[0]['rows'][7]['snapshots'][1]['geometry_id'] += 1
    s = analyze(*data)
    assert any(x['discrete_branch_difference'] and x['state_comparison'] == 'UNKNOWN_UNALIGNED_SNAPSHOTS' for x in s['comparisons'])


def test_constraint_difference_is_continuous_telemetry_not_discrete_branch_claim():
    data = synthetic(); constraints = data[0]['rows'][7]['snapshots'][1]['step_constraints']
    constraints['accuracy_mm'] = constraints['effective_mm'] = 19.
    s = analyze(*data)
    assert s['response_pattern'] == 'NO_OBSERVABLE_BRANCH_DIFFERENCE_IN_LOGGED_SNAPSHOTS'
    assert any(not x['constraints_sequence_exact'] for x in s['comparisons'])


def test_snapshot_count_change_prevents_equal_index_comparison():
    data = synthetic(); row = data[0]['rows'][7]; row['snapshots'].pop(); row['snapshot_count'] -= 1
    s = analyze(*data)
    assert s['response_pattern'] == 'OBSERVABLE_BRANCH_DIFFERENCE_MEASURED'
    assert any(x['snapshot_count_difference'] != 0 for x in s['comparisons'])


def test_empty_trace_is_unknown_without_extra_calls_or_filled_endstate():
    data = synthetic(); row = data[0]['rows'][7]
    row.update(snapshots=[], snapshot_count=0, trace_status='UNKNOWN_EMPTY_TRACE')
    s = analyze(*data)
    assert s['response_pattern'] == 'INCOMPLETE_OFFICIAL_OR_TRACE_RESPONSE'
    assert s['empty_trace_calls'] == [8] and s['official_calls'] == 36


def test_missing_plain_response_is_unknown_without_changing_historical_evidence():
    data = synthetic(); row = data[0]['rows'][6]; row['has_value'] = False; row.pop('state')
    s = analyze(*data)
    assert s['missing_plain_calls'] == [7] and s['successful_plain_calls'] == 17
    assert s['response_pattern'] == 'INCOMPLETE_OFFICIAL_OR_TRACE_RESPONSE'


def test_nonzero_plain_historical_difference_remains_descriptive_reproduction_difference():
    data = synthetic(); state = data[0]['rows'][6]['state']
    state['h'][0][0] += .001; state['global_position_mm'][0][0] += .001; state['local_position_mm'][0][0] += .001
    s = analyze(*data)
    assert s['response_pattern'] == 'RESPONSE_REPRODUCTION_DIFFERENCE' and s['integrity_gate'] == 'PASS'
    assert any(x['exact'] is False for x in s['plain_response_checks'])


def test_actual_producer_consumer_schema_budget_and_both_public_apis():
    source = files()['WB120Diagnostic/BoundedResponse.cxx']; data = list(synthetic())
    data[0]['schema'] = re.search(r'Json out\{\{"schema","([^"]+)"\}', source).group(1)
    assert analyze(*data)['official_calls'] == 36
    assert 'if(count!=36)' in source and 'm_tool->propagate(ctx,fresh,*plane,direction)' in source
    assert 'm_tool->propagationSteps(ctx,fresh,*plane,direction)' in source and 'target_endpoint_exported' in source
    assert 'auto fresh=bound(seed,z,g)' in source and 'MaxStepSize=10.' in athena_source()
    assert 'FaserActsExtrapolationTool' in athena_source() and 'WB120.BoundedResponse' in athena_source()
    assert '_120_' in WORKBOOK.name and WORKBOOK.read_text().startswith('# Workbook 120 ')
    assert OUT.name == 'mc24_four_station_wb120_official_qop_stepping_trace_v1'


@pytest.mark.parametrize('bad', ['schema','source','units','sensor','field','world','call','api','order','frame','seed','qop','momentum_unit',
                                 'trace_start','sentinel','nan','snapshot_index','snapshot_direction','surface_id','constraint','cap','target_substitution',
                                 'historical_h','historical_input','nominal_missing'])
def test_scientific_identity_and_semantic_mutations_fail_closed(bad):
    r,c,f,h,old = synthetic(); trace = r['rows'][7]; snap = trace['snapshots'][1]
    if bad == 'schema': r['schema'] = 'wb119_bounded_response_runtime_v1'
    elif bad == 'source': r['identity']['input_xaod'] = '/other.root'
    elif bad == 'units': r['acts_MeV_unit'] *= 1000.
    elif bad == 'sensor': r['sensors'][0]['transform'][0][0] += 1.
    elif bad == 'field': r['field_conditions']['scale'] = -1.
    elif bad == 'world': r['world']['name'] = 'OTHER'
    elif bad == 'call': trace['input']['call_id'] = 1
    elif bad == 'api': trace['input']['api'] = 'propagate'
    elif bad == 'order': r['rows'][6],r['rows'][7] = r['rows'][7],r['rows'][6]
    elif bad == 'frame': trace['input']['frame'][2][3] += 1.
    elif bad == 'seed': trace['input']['seed'][0][0] += 1.
    elif bad == 'qop': trace['input']['start_state']['qop_acts'] *= 1000.
    elif bad == 'momentum_unit': trace['snapshots'][0]['momentum_MeV'] = (np.asarray(trace['snapshots'][0]['momentum_MeV'])*.001).tolist()
    elif bad == 'trace_start': trace['snapshots'][0]['global_position_mm'][2][0] += 1.
    elif bad == 'sentinel': snap['rk_rejections_before_previous_accepted_step'] = None
    elif bad == 'nan': snap['momentum_MeV'][0][0] = float('nan')
    elif bad == 'snapshot_index': snap['snapshot_index'] = 0
    elif bad == 'snapshot_direction': snap['navigation_direction'] = -1
    elif bad == 'surface_id': snap['surface_geometry_id'] = 45
    elif bad == 'constraint': snap['step_constraints']['effective_mm'] = 100.
    elif bad == 'cap': snap['step_constraints']['user_mm'] = 1000.
    elif bad == 'target_substitution': trace['state'] = copy.deepcopy(r['rows'][6]['state'])
    elif bad == 'historical_h': old[0]['h'][0][0] += 1.
    elif bad == 'historical_input': old[0]['historical_input']['sign'] = 1
    elif bad == 'nominal_missing': r['rows'][0]['has_value'] = False; r['rows'][0].pop('state')
    with pytest.raises(ValueError): analyze(r,c,f,h,old)
