#!/usr/bin/env python3
"""Independent SciPy saved recomputation, never running the producer again."""
import sys
sys.dont_write_bytecode = True
sys.path[:] = [p for p in sys.path if 'madgraph' not in p.lower()]
import json
from pathlib import Path
import numpy as np
from scipy import linalg
from wb126_contract import OUT, PARENT, digest, read, verify, write


def main():
    verify()
    summary = read(OUT/'summary.json')
    protocol = read(OUT/'protocol.json')
    reports = []
    for cell in summary['cells']:
        if cell['information_status'] != 'KNOWN':
            reports.append({'index': cell['index'], 'station': cell['station'],
                            'status': 'ORIGINAL_UNKNOWN_PRESERVED', 'error': cell.get('error')})
            continue
        fixture = read(PARENT/'events'/f'{cell["index"]:02d}'/'fixture.json')
        response = read(PARENT/'events'/f'{cell["index"]:02d}'/'response.json')
        station = cell['station']; ref = fixture['references'][station]
        row = next(r for r in response['responses'] if (r['arm'], r['kind'], r['station']) == ('P', 'nominal', station))
        residual = np.asarray(ref['fixed_z_state']).reshape(4)-np.asarray(row['state']['h']).reshape(4)
        scale = np.diag(protocol['scales'])
        t = np.eye(4); t[0, 2] = t[1, 3] = ref['z_state_mm']-ref['z_center_mm']
        ti = linalg.solve(t, np.eye(4))
        n = scale@ti.T@ref['raw_normal']@ti@scale
        v = linalg.solve(scale, residual)
        values, vectors = linalg.eigh(n, driver='evd')
        weak = values/values[-1] <= protocol['weak_relative_information_cut']
        fraction = float(np.linalg.norm(vectors[:, weak].T@v)**2/(v@v))
        raw_residual = linalg.solve(t, residual)
        q = float(raw_residual@ref['raw_normal']@raw_residual)
        expected = cell['spectrum']
        error_eigen = float(np.linalg.norm(values-expected['eigenvalues'])/np.linalg.norm(values))
        error_fraction = abs(fraction-expected['weak_scaled_residual_fraction'])
        error_q = abs(q-expected['Q'])/max(abs(q), np.finfo(float).tiny)
        assert error_eigen <= 1e-8 and error_fraction <= 1e-8 and error_q <= 1e-8
        assert np.array_equal(residual, cell['residual'])
        reports.append({'index': cell['index'], 'station': station, 'status': 'PASS',
                        'independent_weak_fraction': fraction, 'eigen_relative_error': error_eigen,
                        'weak_fraction_absolute_error': error_fraction, 'quadratic_relative_error': error_q})
    assert len(reports) == 24
    primary = [next(r for r in reports if [r['index'], r['station']] == key) for key in protocol['primary_cells']]
    known = [r for r in primary if r['status'] == 'PASS']
    independent_h = ('NOT_SUPPORTED' if any(r['independent_weak_fraction'] < .9 for r in known) else
                     'UNKNOWN' if len(known) != len(primary) else 'SUPPORTED_BUT_LIMITED')
    assert independent_h == summary['hypothesis']
    verify()
    write(OUT/'independent_result_check.json', {'gate': 'PASS', 'cells': reports, 'python': sys.version,
          'numpy': np.__version__, 'script_sha256': digest(Path(__file__)), 'summary_sha256': digest(OUT/'summary.json'),
          'hypothesis': independent_h, 'source_root_reads': 0, 'new_propagation_calls': 0,
          'scope': 'SciPy eigen decomposition and independent lever-arm/quadratic recomputation; same saved data, not independent events'})
    print(json.dumps({'gate': 'PASS', 'cells': 24, 'hypothesis': independent_h,
                      'max_fraction_error': max(r.get('weak_fraction_absolute_error', 0) for r in reports)}, indent=2))


if __name__ == '__main__':
    main()
