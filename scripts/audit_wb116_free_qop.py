#!/usr/bin/env python3
"""WB116 saved official-response diagnostic; no new physical backend calls."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alignment.wb90_measurement_contract import ROOT, read_public, write_new
from wb100_contract import digest, verify
from audit_wb112_shared_seed import references, inputs, BASE, HISTORICAL, SOURCE_FIXTURE

OUT = ROOT / 'outputs/mc24_four_station_wb116_free_qop_saved_response_v1'
PREVIOUS = ROOT / 'outputs/mc24_four_station_wb115_field_conditions_provenance_v2'
WB112 = ROOT / 'outputs/mc24_four_station_wb112_shared_seed_audit_v1'
WORKBOOK = ROOT / 'workbook/2026-10-03_116_四站保存响应的自由动量共享种子局部可辨识性审计.md'
AGREEMENT = 1e-8


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array(value, shape):
    out = np.asarray(value, dtype=np.float64)
    require(out.shape == shape and np.isfinite(out).all(), 'shape/finite')
    return out


def chol(value):
    require(np.allclose(value, value.T, rtol=1e-12, atol=1e-12), 'covariance symmetry')
    return np.linalg.cholesky(value)


def spectrum(design):
    values = np.linalg.svd(design, compute_uv=False)
    cutoff = np.finfo(np.float64).eps * max(design.shape) * values[0]
    rank = int(np.sum(values > cutoff))
    return {'singular_values': values.tolist(), 'machine_resolution_cutoff': float(cutoff),
            'machine_rank': rank, 'shape': list(design.shape),
            'condition_number': float(values[0] / values[-1]) if values[-1] > 0 else None,
            'scientific_identifiability': 'UNVERIFIED_UNCALIBRATED_COVARIANCE'}


def relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(1., float(np.linalg.norm(a)), float(np.linalg.norm(b))))


def profile(residual, jacobian, c0, cs, scales, qscale):
    """Conditional shared-covariance projection versus independent augmented QR."""
    r = array(residual, (8,)); j = array(jacobian, (8, 5))
    c0 = array(c0, (4, 4)); scales = array(scales, (4,))
    require(len(cs) == 2 and np.all(scales > 0) and np.isfinite(qscale) and qscale > 0, 'scales/targets')
    cs = [array(c, (4, 4)) for c in cs]
    outscale = np.tile(scales, 2); xscale = np.r_[scales, qscale]
    r = r / outscale; j = j / outscale[:, None] * xscale[None, :]
    c0 = c0 / scales[:, None] / scales[None, :]
    d = np.zeros((8, 8))
    for k, c in enumerate(cs):
        d[4*k:4*k+4, 4*k:4*k+4] = c / scales[:, None] / scales[None, :]
    l0 = chol(c0); ld = chol(d)
    j4, jq = j[:, :4], j[:, 4]
    shared = d + j4 @ c0 @ j4.T; chol(shared)
    solve_r = np.linalg.solve(shared, r); solve_q = np.linalg.solve(shared, jq)
    qfixed = float(r @ solve_r); info = float(jq @ solve_q)
    data_design = np.linalg.solve(ld, j)
    prior_design = np.c_[np.linalg.solve(l0, np.eye(4)), np.zeros(4)]
    design = np.vstack([data_design, prior_design]); rhs = np.r_[np.linalg.solve(ld, -r), np.zeros(4)]
    data_spec, augmented_spec = spectrum(data_design), spectrum(design)
    base = {'Q_fixed_qop_shared_seed': qfixed, 'conditional_information_scaled_qop': info,
            'data_only_spectrum': data_spec, 'augmented_spectrum': augmented_spec,
            'condition_shared_scaled': float(np.linalg.cond(shared)), 'qop_prior': 'NONE',
            'dummy_qop_variance_used': False, 'p_value': 'NOT_EVALUATED'}
    if info <= 0 or not np.isfinite(info) or augmented_spec['machine_rank'] != 5:
        return dict(base, status='UNKNOWN_NUMERICALLY_UNRESOLVED_QOP_DIRECTION')
    deltaq = -float(jq @ solve_r) / info
    shifted = r + jq * deltaq
    delta4 = -c0 @ j4.T @ np.linalg.solve(shared, shifted)
    delta = np.r_[delta4, deltaq]
    qrq, qrr = np.linalg.qr(design, mode='reduced')
    qrdelta = np.linalg.solve(qrr, qrq.T @ rhs)
    remaining = r + j @ delta
    prior_part = float(np.linalg.norm(np.linalg.solve(l0, delta4))**2)
    data_part = float(np.linalg.norm(np.linalg.solve(ld, remaining))**2)
    qfree = float(shifted @ np.linalg.solve(shared, shifted))
    qr_objective = float(np.linalg.norm(design @ qrdelta - rhs)**2)
    errors = {'delta_scaled': relative(delta, qrdelta),
              'Q_projection_vs_prior_data': relative(qfree, prior_part + data_part),
              'Q_projection_vs_augmented_QR': relative(qfree, qr_objective)}
    base.update({'conditional_information_MeV_squared': info / qscale**2,
                 'conditional_curvature_width_per_MeV': qscale / np.sqrt(info),
                 'linear_delta': (delta * xscale).tolist(), 'independent_QR_delta': (qrdelta*xscale).tolist(),
                 'Q_free_qop': qfree, 'Q_prior_part': prior_part, 'Q_data_part': data_part,
                 'Q_augmented_QR': qr_objective, 'Q_reduction': qfixed - qfree,
                 'linear_remaining_residual': (remaining*outscale).tolist(),
                 'delta4_in_seed_marginal_sigma': (delta4/np.sqrt(np.diag(c0))).tolist(),
                 'qr_diagonal': np.diag(qrr).tolist(), 'arithmetic_relative_errors': errors})
    if not all(np.isfinite(v) and v <= AGREEMENT for v in errors.values()):
        return dict(base, status='UNKNOWN_ARITHMETIC_DISAGREEMENT')
    require(qfree >= 0 and qfree <= qfixed + AGREEMENT*max(1., qfixed), 'projection monotonicity')
    return dict(base, status='RESOLVED_LOCAL_ARITHMETIC_ONLY')


def saved_response(fixture, control, trace):
    refs = fixture['references']; steps = array(fixture['protocol']['seed_steps'], (5,))
    scales = array(fixture['protocol']['output_scales'], (4,))
    require(np.all(steps > 0) and np.all(scales > 0), 'positive steps/scales')
    seed = np.r_[array(refs[0]['fixed_z_state'], (4, 1)).reshape(4), refs[0]['q_over_p_per_MeV']]
    require(np.array_equal(array(control['seed'], (5, 1)).reshape(5), seed), 'seed identity')
    require(control['seed_z_mm'] == refs[0]['z_state_mm'], 'seed z')
    require([t['station'] for t in control['targets']] == [1, 2, 3] and
            [t['call_id'] for t in control['targets']] == [24, 74, 124], 'nominal calls/stations')
    require(control['targets'][2]['official_has_value'] is False, 'historical station3 failure')
    ins = [x for x in trace if x['record'] == 'input']; outs = [x for x in trace if x['record'] == 'output']
    byid = {x['call_id']: x for x in ins}; outputs = {x['call_id']: x for x in outs}
    require(len(byid) == len(ins) and len(outputs) == len(outs), 'duplicate call identity')
    require(124 in byid and 124 not in outputs, 'station3 missing output contract')
    matrices = {'full': [], 'half': []}; residuals = []; details = []
    for station in (1, 2):
        target = control['targets'][station-1]; nominal = byid[target['call_id']]
        require(target['official_has_value'] is True and nominal['station'] == station and
                nominal['label'] == 'nominal' and nominal['frame'] == target['frame'] and
                np.array_equal(array(nominal['seed'], (5, 1)).reshape(5), seed) and
                nominal['seed_z_mm'] == control['seed_z_mm'], 'nominal identity')
        require(outputs[target['call_id']]['h'] == target['official_h'], 'nominal output')
        h = array(target['official_h'], (4, 1)).reshape(4)
        residuals.append(h - array(refs[station]['fixed_z_state'], (4, 1)).reshape(4))
        ids = []
        for name, factor, labels in (('full', 1., ('xi_plus', 'xi_minus')),
                                    ('half', .5, ('xi_half_plus', 'xi_half_minus'))):
            jac = np.zeros((4, 5))
            for axis in range(5):
                values = []
                for sign, label in zip((1., -1.), labels):
                    matches = [x for x in ins if (x['station'], x['axis'], x['label']) == (station, axis, label)]
                    require(len(matches) == 1, 'missing/duplicate axis response')
                    row = matches[0]; expected = seed.copy(); expected[axis] += sign*factor*steps[axis]
                    require(np.array_equal(array(row['seed'], (5, 1)).reshape(5), expected) and
                            row['seed_z_mm'] == control['seed_z_mm'] and row['frame'] == target['frame'],
                            'perturbation seed/z/frame/qop')
                    require(row['call_id'] in outputs, 'missing perturbation output')
                    values.append(array(outputs[row['call_id']]['h'], (4, 1)).reshape(4)); ids.append(row['call_id'])
                jac[:, axis] = (values[0] - values[1]) / (2*factor*steps[axis])
            matrices[name].append(jac)
        full, half = matrices['full'][-1], matrices['half'][-1]
        details.append({'station': station, 'used_call_ids': ids, 'Hxi_5D_full': full.tolist(),
                        'Hxi_5D_half': half.tolist(),
                        'full_qop_step_effect_scaled': (full[:, 4]*steps[4]/scales).tolist(),
                        'half_qop_step_effect_scaled': (half[:, 4]*steps[4]/scales).tolist(),
                        'halving_scaled_effect_difference': ((full-half)*steps[None, :]/scales[:, None]).tolist()})
    return np.concatenate(residuals), {k: np.vstack(v) for k, v in matrices.items()}, details, seed, steps, scales


def analyze(fixture, repair, control, trace, old):
    require((fixture['index'], fixture['ordinal'], fixture['actual_run'], fixture['actual_event']) ==
            (12, 2268, 100044, 2268), 'single seen-event allowlist')
    reference_evidence, _ = references(fixture, repair)
    r, matrices, details, seed, steps, scales = saved_response(fixture, control, trace)
    c0 = fixture['references'][0]['fixed_z_covariance']
    cs = [fixture['references'][s]['fixed_z_covariance'] for s in (1, 2)]
    profiles = {}
    for name, jac in matrices.items():
        row = profile(r, jac, c0, cs, scales, steps[4])
        expected = old['quadratic_diagnostics'][name]['Q_shared_seed']
        require(relative(row['Q_fixed_qop_shared_seed'], expected) <= AGREEMENT, 'WB112 fixed profile identity')
        row['WB112_fixed_Q_relative_difference'] = relative(row['Q_fixed_qop_shared_seed'], expected)
        if 'linear_delta' in row:
            delta = array(row['linear_delta'], (5,))
            row['delta_in_original_FD_steps'] = (delta/steps).tolist()
            row['all_deltas_inside_original_axis_box'] = bool(np.all(np.abs(delta) <= steps))
            row['algebraic_qop_after_delta_per_MeV'] = float(seed[4]+delta[4])
        profiles[name] = row
    comparisons = {}
    if all(row['status'] == 'RESOLVED_LOCAL_ARITHMETIC_ONLY' for row in profiles.values()):
        for key in ('Q_free_qop', 'conditional_information_scaled_qop', 'linear_delta'):
            comparisons[key] = relative(profiles['full'][key], profiles['half'][key])
        a, b = [array(profiles[n]['linear_delta'], (5,))/steps for n in ('full', 'half')]
        comparisons['delta_in_FD_steps_relative_difference'] = relative(a, b)
        comparisons['Q_free_full_over_half'] = profiles['full']['Q_free_qop']/profiles['half']['Q_free_qop'] if profiles['half']['Q_free_qop'] > 0 else None
    return {'schema': 'wb116_free_qop_saved_response_summary_v1',
            'classification': 'DIAGNOSTIC_ONLY_UNCALIBRATED_LOCAL_MODEL',
            'identity': {k: fixture[k] for k in ('index', 'ordinal', 'actual_run', 'actual_event', 'input_xaod')},
            'reference_evidence': reference_evidence, 'seed': seed.tolist(), 'original_FD_steps': steps.tolist(),
            'output_scales': scales.tolist(), 'derivative_details': details, 'profiles': profiles,
            'full_half_descriptive_comparisons': comparisons, 'arithmetic_agreement_tolerance': AGREEMENT,
            'station3_profile': 'UNKNOWN_MISSING_JACOBIAN', 'association': 'UNKNOWN_OR_AMBIGUOUS_WB114',
            'covariance_calibration': 'UNVERIFIED', 'physical_conditions_compatibility': 'UNKNOWN',
            'nonlinear_fit': 'NOT_EVALUATED', 'truth_momentum_used': False, 'dummy_qop_variance_used': False,
            'held_out_access': False, 'root_or_truth_access': False, 'new_propagation_calls': 0,
            'new_reconstruction_calls': 0, 'qualification': 'NOT_EVALUATED',
            'physics_screening': 'NOT_EVALUATED', 'final_oracle': 'NOT_EVALUATED'}


def freeze():
    require(subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == '4station', 'branch')
    require(not subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=ROOT, text=True).strip(), 'tracked changes')
    hashes = verify(PREVIOUS)['hashes'].copy()
    manifest = ROOT/'docs/wb115_field_conditions_result_manifest.json'
    m = read_public(manifest)
    require(m['execution_contract'] == m['integrity'] == 'PASS', 'WB115 integrity')
    for name, h in m['artifacts'].items():
        require(digest(ROOT/name) == h, 'WB115 artifact ' + name); hashes[str(ROOT/name)] = h
    previous_manifest = ROOT/'docs/wb112_shared_seed_result_manifest.json'
    for name, h in read_public(previous_manifest)['artifacts'].items():
        require(digest(ROOT/name) == h, 'WB112 artifact ' + name); hashes[str(ROOT/name)] = h
    original, fixture, repair = inputs()
    require(digest(repair) == original['wb91_repair_sha256'], 'repair identity')
    paths = [Path(__file__), ROOT/'tests/test_wb116_free_qop.py', manifest, previous_manifest,
             ROOT/'scripts/audit_wb112_shared_seed.py', SOURCE_FIXTURE, BASE/'fixture.json', BASE/'control.json',
             repair, WB112/'summary.json', HISTORICAL/'event/acts.json.calls.ndjson',
             ROOT/'scripts/setup_environment.sh', ROOT/'research/wb92/CommonSeedAudit.cxx',
             ROOT/'scripts/wb106_contract.py', ROOT/'scripts/wb109_sources.py',
             HISTORICAL/'isolated_source/WB107Diagnostic/CommonSeedAudit.cxx']
    for p in paths:
        hashes[str(p)] = digest(p)
    OUT.mkdir(exist_ok=False); shutil.copyfile(WORKBOOK, OUT/'contract_workbook.md')
    hashes[str(OUT/'contract_workbook.md')] = digest(OUT/'contract_workbook.md')
    write_new(OUT/'freeze.json', {'schema': 'wb116_free_qop_saved_response_freeze_v1', 'hashes': hashes,
              'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'branch': '4station', 'population': 1, 'new_propagation_calls': 0,
              'root_or_truth_access': False, 'held_out_access': False})


def rebuild():
    original, fixture, repair = inputs()
    require(digest(repair) == original['wb91_repair_sha256'], 'repair identity')
    rows = [json.loads(line) for line in repair.read_text().splitlines()]
    trace = [json.loads(line) for line in (HISTORICAL/'event/acts.json.calls.ndjson').read_text().splitlines()]
    return analyze(fixture, rows, read_public(BASE/'control.json'), trace, read_public(WB112/'summary.json'))


def run():
    start = time.monotonic(); frozen = verify(OUT); (OUT/'execution_lock').mkdir(exist_ok=False)
    summary = rebuild(); write_new(OUT/'summary.json', summary); verify(OUT)
    resolved = all(p['status'] == 'RESOLVED_LOCAL_ARITHMETIC_ONLY' for p in summary['profiles'].values())
    write_new(OUT/'execution_receipt.json', {'exit_code': 0, 'frozen_identities_verified': len(frozen['hashes']),
              'elapsed_seconds': time.monotonic()-start, 'mode': 'local_saved_values_only',
              'population': 1, 'profile_arithmetic': 'PASS' if resolved else 'UNKNOWN',
              'event_root_access': False, 'truth_access': False, 'new_propagation_calls': 0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run', 'verify'))
    action = parser.parse_args().action
    if action == 'verify':
        verify(OUT)
    else:
        globals()[action]()
