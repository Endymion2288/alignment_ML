#!/usr/bin/env python3
"""Supplement WB125 with exact corresponding S0 oracles; never overwrite its gate.

Saved-only: no ROOT, Athena, propagation, source data read, tuning or rerun.
The original execution_contract remains UNKNOWN if the all-h oracle failed.
"""
import copy
import json
import re
from pathlib import Path

import numpy as np

from wb125_audit import audit_event
from wb125_contract import OUT, PARENT, ROOT, digest, frozen_runtime, verify


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def flat(value):
    return np.asarray(value, dtype=float).reshape(-1)


def boundary(response, old):
    row = next(r for r in response['responses']
               if (r['arm'], r['kind'], r['station']) == ('D', 'nominal', 0))
    nominal = next(t for t in old['targets'] if t['station'] == 0)
    require(np.array_equal(flat(nominal['h']), flat(response['seeds']['D'])[:4]),
            'old boundary h/input seed identity')
    require(np.array_equal(flat(row['state']['h']), flat(old['seed_roundtrip'])[:4]),
            'old/new boundary bound roundtrip identity')
    require(row['propagation_call'] is None, 'S0 must not propagate')
    require(len(nominal['sensors']) == len(response['targets'][0]['sensors']), 'S0 sensor coverage')
    for a, b in zip(nominal['sensors'], response['targets'][0]['sensors']):
        require(all(a[k] == b[k] for k in ('strip', 'wafer', 'geometry_id', 'transform')), 'S0 sensors')
        require(np.array_equal(a['delta'], np.eye(4)), 'old baseline delta identity')
    return {'input_seed_exact_old_h': True, 'readout_exact_old_roundtrip': True,
            'old_h_exact_new_readout': np.array_equal(flat(nominal['h']), flat(row['state']['h'])),
            'readout_minus_old_h': (flat(row['state']['h']) - flat(nominal['h'])).tolist()}


def main():
    freeze = verify()
    runtime = frozen_runtime()
    inputs = {}
    def tracked(path):
        inputs[str(Path(path).relative_to(ROOT))] = digest(path)
        return read(path)
    original = tracked(OUT / 'summary.json')
    results = []
    negative = []
    for i in (1, 4, 8, 12, 16, 20):
        event = OUT / 'events' / f'{i:02d}'
        fixture, provenance, response, historical, exit_receipt = [
            tracked(event / name) for name in
            ('fixture.json', 'provenance.json', 'response.json', 'summary.json', 'exit.json')]
        require(exit_receipt['exit_code'] == 0, 'worker exit')
        old = tracked(PARENT / 'events' / f'{i:02d}' / 'acts.json') if i != 12 else None
        b = boundary(response, old) if old else {'old_S3_failure_checked_by_frozen_auditor': True}
        # Keep every downstream oracle unchanged; explicitly audit S0 separately.
        downstream = copy.deepcopy(old)
        if downstream:
            downstream['targets'] = [t for t in downstream['targets'] if t['station'] > 0]
        supplemental = audit_event(fixture, provenance, response, downstream)
        if old:
            for t in old['targets']:
                require(len(t['sensors']) == len(response['targets'][t['station']]['sensors']), 'sensor coverage')
        parameters = provenance['persisted_tracks'][0]['matching_tracks'][0]['parameters']
        z = fixture['references'][0]['z_state_mm']
        selected = min(range(len(parameters)), key=lambda k: (abs(parameters[k]['global_position_native'][2] - z), k))
        require(selected == response['selection']['selected_index'], 'independent nearest state')
        native = parameters[selected]
        momentum = float(np.sqrt(sum(x*x for x in native['global_momentum_native'])))
        source_qop = native['charge_e'] / momentum
        require(abs(source_qop - response['selection']['source_qop_per_MeV']) <=
                4 * np.finfo(float).eps * abs(source_qop), 'independent q/p')
        for name in ('athena.log',):
            path = event / name
            inputs[str(path.relative_to(ROOT))] = digest(path)
        log = (event / 'athena.log').read_text(errors='replace')
        begins = [int(x) for x in re.findall(r'WB125_CALL_BEGIN id=(\d+)', log)]
        ends = [int(x) for x in re.findall(r'WB125_CALL_END id=(\d+)', log)]
        require(begins == ends == list(range(1, response['propagation_calls'] + 1)), 'call log closure')
        require('Reading folder /Tracker/Align from sqlite' in log and
                'Using FASER magnetic field service' in log, 'actual payload/field service')
        loaded = {Path(x).resolve() for x in response['loaded_libraries']}
        for path, identity in runtime.items():
            if '.so' in path:
                require(Path(path).resolve() in loaded and digest(path) == identity, 'default library')
        maps = re.findall(r'Initialized the field map from\s+([^\n]+)', log)
        require(bool(maps), 'actual field map')
        for path in maps:
            path = path.strip().strip('"')
            require(path in runtime and digest(path) == runtime[path], 'actual field map identity')
        old_log_path = PARENT / 'events' / f'{i:02d}' / 'athena.log'
        inputs[str(old_log_path.relative_to(ROOT))] = digest(old_log_path)
        pattern = r'initialized FaserFieldCacheCondObj and cache with scale factor ([^\s]+)'
        scales = re.findall(pattern, log)
        require(scales and scales == re.findall(pattern, old_log_path.read_text(errors='replace')), 'field scale')
        # Independent saved arithmetic, compared with the frozen auditor's values.
        metrics = []
        for row in response['responses']:
            if row['status'] == 'SUCCESS':
                y = flat(fixture['references'][row['station']]['fixed_z_state'])
                residual = y - flat(row['state']['h'])
                R = float(np.sqrt(sum((residual / [1., 1., .001, .001]) ** 2)))
                cell = [row['arm'], row['kind'], row['station'], float(row.get('qop_multiplier', 0.))]
                expected = next(m for m in supplemental['metrics'] if m['cell'] == cell)
                require(np.array_equal(residual, expected['residual']), 'independent residual')
                require(abs(R - expected['R']) <= 4 * np.finfo(float).eps * max(R, expected['R']), 'independent R')
                metrics.append({'cell': cell, 'residual': residual.tolist(), 'R': R})
        independent_fd = []
        for fd in supplemental['fd_diagnostics']:
            hs = {row['qop_multiplier']: flat(row['state']['h']) for row in response['responses']
                  if row['kind'] == 'fd' and row['station'] == fd['station'] and row['status'] == 'SUCCESS'}
            full = (hs[1.] - hs[-1.]) / (2 * 1e-8)
            half = (hs[.5] - hs[-.5]) / 1e-8
            require(np.array_equal(full, fd['J_full']) and np.array_equal(half, fd['J_half']), 'independent FD')
            effect = (half-full)*1e-8
            require(np.array_equal(effect, fd['step_scaled_difference']), 'independent FD effect')
            independent_fd.append({'station': fd['station'], 'step_scaled_difference': effect.tolist(),
                                   'max_abs_scaled': float(np.max(abs(effect / [1., 1., .001, .001])))})
        metadata = tracked(event / 'source_metadata.json')
        taginfo = [v.get('/TagInfo', {}) for v in metadata.values()]
        qop = supplemental['source_selection']['source_qop_per_MeV']
        supplemental.update({'boundary_oracles': b, 'original_integrity': historical['integrity'],
                             'original_error': historical.get('error'), 'actual_field_scales': scales,
                             'source_taginfo': taginfo, 'source_momentum_GeV': abs(1 / qop) / 1000,
                             'independent_selected_index': selected, 'independent_source_qop': source_qop,
                             'independent_fd': independent_fd,
                             'independent_metrics': metrics, 'runtime_and_call_closure': 'PASS',
                             'historical_raw_identity': tracked(OUT / 'raw_input_freeze' / f'{i:02d}.json')['historical_byte_identity']})
        if old:
            for label, source, oracle in [('boundary_readout', response, old), ('downstream_h', response, downstream)]:
                corrupt = copy.deepcopy(source)
                target_station = 0 if label == 'boundary_readout' else 1
                row = next(r for r in corrupt['responses'] if
                           (r['arm'], r['kind'], r['station']) == ('D', 'nominal', target_station))
                row['state']['h'][0][0] = float(np.nextafter(row['state']['h'][0][0], np.inf))
                rejected = False
                try:
                    if label == 'boundary_readout':
                        boundary(corrupt, oracle)
                    else:
                        audit_event(fixture, provenance, corrupt, oracle)
                except ValueError:
                    rejected = True
                require(rejected, 'one-ULP exact-oracle negative control')
                negative.append({'index': i, 'control': label, 'one_ulp_corruption_rejected': True})
        results.append(supplemental)
    total = sum(r['propagation_calls'] for r in results)
    require(total == original['propagation_calls'] == 88, 'budget closure')
    result = {'schema': 'wb125_saved_corresponding_boundary_oracle_audit_v1',
              'original_execution_contract': original['execution_contract'],
              'supplemental_corresponding_oracles': 'PASS',
              'scope': 'Corresponding S0 seed/roundtrip and unchanged downstream exact comparisons; not a replacement preregistered gate',
              'new_propagation_calls': 0, 'source_root_reads': 0, 'held_out_access': False,
              'propagation_calls_original_stage': total, 'events': results, 'negative_controls': negative,
              'physical_fixture_supported': all(r['physical_fixture_supported'] for r in results),
              'hypothesis': next(r['hypothesis'] for r in results if r['index'] == 12),
              'qualification': 'NOT_EVALUATED', 'freeze_sha256': digest(OUT / 'freeze.json'),
              'implementation_freeze_commit': freeze['git_commit'],
              'script_sha256': digest(Path(__file__)), 'input_sha256': inputs}
    with (OUT / 'saved_corresponding_boundary_audit.json').open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')
    verify()
    print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'input_sha256', 'negative_controls')}, indent=2))


if __name__ == '__main__':
    main()
