#!/usr/bin/env python3
"""Saved-only identity-interface recovery. Original WB126 UNKNOWN is immutable."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
sys.path[:] = [p for p in sys.path if 'madgraph' not in p.lower()]
import numpy as np
from wb126_contract import OUT, PARENT, ROOT, PROTOCOL, digest, read, write, verify
from wb126_information import audit_cell, hypothesis, require

RECOVERY = OUT/'saved_cluster_order_recovery_v1'
SCRIPT = Path(__file__)
TEST = ROOT/'tests/test_wb126_order_recovery.py'


def compare_reference(original, fixture):
    """Cluster selection is an unordered unique allowlist; numeric fields are exact."""
    a, b = original['clusters'], fixture['clusters']
    require(len(a) == len(set(a)) and len(b) == len(set(b)), 'duplicate cluster identity')
    require(sorted(a) == sorted(b), 'cluster membership changed')
    require(all(original[k] == fixture[k] for k in original if k != 'clusters'), 'numeric/reference identity changed')
    return {'cluster_membership_exact': True, 'all_other_fields_exact': True, 'ordering_differs': a != b}


def freeze():
    verify()
    require(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip(), 'committed recovery')
    RECOVERY.mkdir(exist_ok=False)
    hashes = {str(p): digest(p) for p in (SCRIPT, TEST, PROTOCOL, OUT/'summary.json', OUT/'independent_result_check.json')}
    for p in (OUT/'events').glob('*.json'):
        hashes[str(p)] = digest(p)
    receipt = read(ROOT/'outputs/wb126_order_reader_preflight.json')
    require(receipt['pytest_exit'] == 0 and receipt['script_sha256'] == digest(SCRIPT) and
            receipt['test_sha256'] == digest(TEST), 'order control identity')
    hashes[str(ROOT/'outputs/wb126_order_reader_preflight.json')] = digest(ROOT/'outputs/wb126_order_reader_preflight.json')
    write(RECOVERY/'freeze.json', {'hashes': hashes,
          'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'original_freeze_sha256': digest(OUT/'freeze.json'),
          'amendment': 'Only clusters compared as unique sorted ID lists, as WB92 fixture creation; every numeric field exact',
          'scientific_protocol_sha256': digest(PROTOCOL), 'original_execution_contract': 'UNKNOWN',
          'source_root_reads': 0, 'new_propagation_calls': 0})
    recovery_verify()


def recovery_verify():
    verify()
    for p, h in read(RECOVERY/'freeze.json')['hashes'].items():
        require(digest(p) == h, 'recovery frozen identity: '+p)


def run():
    recovery_verify()
    (RECOVERY/'execution_lock').mkdir(exist_ok=False)
    protocol = read(PROTOCOL); parent = read(PARENT/'saved_corresponding_boundary_audit.json')
    cells = []
    for i in protocol['indices']:
        directory = PARENT/'events'/f'{i:02d}'
        fixture, response = read(directory/'fixture.json'), read(directory/'response.json')
        identity = {k: fixture[k] for k in ('input_xaod', 'ordinal', 'actual_run', 'actual_event')}
        require(response['identity'] == identity, 'event identity')
        original_rows = [json.loads(line) for line in Path(fixture['wb91_repair_path']).read_text().splitlines()]
        require(all([r['run'], r['event']] == [fixture['actual_run'], fixture['actual_event']] for r in original_rows), 'repair identity')
        association = next(e for e in parent['events'] if e['index'] == i)
        for station in protocol['stations']:
            c = {'index': i, 'station': station, 'identity': identity, 'association': association['association'],
                 'physical_fixture_supported': association['physical_fixture_supported'], 'conditional_interpretation': True,
                 'downstream_membership_complete': association['downstream_membership_complete']}
            try:
                ref = fixture['references'][station]; require(ref['station'] == station, 'station')
                originals = [r for r in original_rows if r['station'] == station and r['state_index'] == 0]
                require(len(originals) == 1, 'reference multiplicity')
                c['reference_identity'] = compare_reference(originals[0], ref)
                rows = [r for r in response['responses'] if (r['arm'], r['kind'], r['station']) == ('P', 'nominal', station)]
                require(len(rows) == 1 and rows[0]['status'] == 'SUCCESS', 'P response')
                require(rows[0]['state']['on_surface'] and not rows[0]['state']['covariance_present'], 'bound/noC')
                c.update(audit_cell(ref, rows[0]['state']['h'], protocol))
            except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                c.update({'information_status': 'UNKNOWN', 'error': str(error)})
            cells.append(c)
    require(len(cells) == 24 and len({(c['index'], c['station']) for c in cells}) == 24, 'coverage')
    summary = {'schema': 'wb126_order_recovery_summary_v1', 'cells': cells,
               'execution_contract': 'PASS' if all(c['information_status'] == 'KNOWN' for c in cells) else 'UNKNOWN',
               'hypothesis': hypothesis(cells, protocol), 'original_execution_contract': read(OUT/'summary.json')['execution_contract'],
               'original_hypothesis': read(OUT/'summary.json')['hypothesis'], 'source_root_reads': 0,
               'new_propagation_calls': 0, 'covariance_calibration': 'NOT_EVALUATED', 'qualification': 'NOT_EVALUATED',
               'source_information': 'CONDITIONAL_FIXED_SELECTED_CLUSTER_WLS', 'independent_hit_rebuild': 'UNKNOWN_NOT_SAVED',
               'freeze_sha256': digest(RECOVERY/'freeze.json')}
    recovery_verify(); write(RECOVERY/'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'cells'}, indent=2))


def check():
    from scipy import linalg
    recovery_verify()
    summary = read(RECOVERY/'summary.json'); protocol = read(PROTOCOL); reports = []
    for c in summary['cells']:
        if c['information_status'] != 'KNOWN':
            reports.append({'index': c['index'], 'station': c['station'], 'status': 'ORIGINAL_UNKNOWN_PRESERVED'})
            continue
        f = read(PARENT/'events'/f'{c["index"]:02d}'/'fixture.json'); ref = f['references'][c['station']]
        response = read(PARENT/'events'/f'{c["index"]:02d}'/'response.json')
        row = next(r for r in response['responses'] if (r['arm'], r['kind'], r['station']) == ('P', 'nominal', c['station']))
        residual = np.array(ref['fixed_z_state']).reshape(4)-np.array(row['state']['h']).reshape(4)
        t = np.eye(4); t[0, 2] = t[1, 3] = ref['z_state_mm']-ref['z_center_mm']
        ti = linalg.solve(t, np.eye(4)); s = np.diag(protocol['scales'])
        n = s@ti.T@ref['raw_normal']@ti@s
        v = linalg.solve(s, residual)
        lam, vectors = linalg.eigh(n, driver='evd')
        mask = lam/lam[-1] <= protocol['weak_relative_information_cut']
        amplitudes = vectors.T@v
        fraction = float(np.linalg.norm(amplitudes[mask])**2/(v@v))
        raw_residual = linalg.solve(t, residual)
        q = float(raw_residual@ref['raw_normal']@raw_residual)
        expected = c['spectrum']
        errors = {'eigen': float(np.linalg.norm(lam-expected['eigenvalues'])/np.linalg.norm(lam)),
                  'fraction': abs(fraction-expected['weak_scaled_residual_fraction']),
                  'Q': abs(q-expected['Q'])/max(abs(q), np.finfo(float).tiny)}
        require(max(errors.values()) <= 1e-8, 'independent saved arithmetic')
        require(np.array_equal(residual, c['residual']), 'independent residual')
        reports.append({'index': c['index'], 'station': c['station'], 'status': 'PASS', 'errors': errors,
                        'independent_weak_fraction': fraction})
    # Check against the unchanged original classifier using independent fractions.
    surrogate = [{'index': r['index'], 'station': r['station'],
                  'information_status': 'KNOWN' if r['status'] == 'PASS' else 'UNKNOWN',
                  'spectrum': {'weak_scaled_residual_fraction': r.get('independent_weak_fraction')}} for r in reports]
    require(hypothesis(surrogate, protocol) == summary['hypothesis'], 'independent hypothesis')
    recovery_verify()
    write(RECOVERY/'independent_result_check.json', {'gate': 'PASS', 'cells': reports, 'python': sys.version,
          'numpy': np.__version__, 'summary_sha256': digest(RECOVERY/'summary.json'), 'new_propagation_calls': 0,
          'scope': 'SciPy independent eigen/lever-arm solve/quadratic, same saved data, no new events'})
    print('INDEPENDENT_SAVED_CHECK_PASS', len(reports))


def seal():
    recovery_verify(); summary = read(RECOVERY/'summary.json')
    require(read(RECOVERY/'independent_result_check.json')['gate'] == 'PASS', 'independent check')
    write(ROOT/'docs/wb126_tracklet_information_recovery_result_manifest.json',
          {'schema': 'wb126_order_recovery_result_manifest_v1', 'original_execution_contract': 'UNKNOWN',
           'original_hypothesis': 'UNKNOWN', 'recovery_execution_contract': summary['execution_contract'],
           'hypothesis': summary['hypothesis'], 'scientific_protocol_sha256': digest(PROTOCOL),
           'sources': {str(p.relative_to(ROOT)): digest(p) for p in (SCRIPT, TEST)},
           'artifacts': {str(p.relative_to(ROOT)): digest(p) for p in sorted(RECOVERY.rglob('*')) if p.is_file()},
           'reader_preflight': {'path': 'outputs/wb126_order_reader_preflight.json',
                                'sha256': digest(ROOT/'outputs/wb126_order_reader_preflight.json')},
           'new_propagation_calls': 0, 'raw_root_reads': 0, 'held_out_access': False,
           'covariance_calibration': 'NOT_EVALUATED', 'qualification': 'NOT_EVALUATED'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run', 'check', 'seal'))
    globals()[parser.parse_args().action]()
