"""Complete local Jacobian scale diagnostic with explicit reference gates."""
import numpy as np
from alignment.wb93_transport_error import v, norm
from alignment.wb94_field_boundary import reference_check, check_state
from alignment.wb101_direction_acceptance import complete_metrics


def response(values, lam, p):
    a = np.asarray(values, dtype=float)
    if a.shape != (25, 4) or not np.isfinite(a).all() or lam not in p['lambdas']:
        raise ValueError('response shape, finite values or lambda')
    scale = np.asarray(p['output_scales'])
    k = np.stack([(a[1 + 4*j] - a[2 + 4*j]) / (lam * scale) for j in range(5)], axis=1)
    j = k * scale[:, None] / np.asarray(p['seed_steps'])[None, :]
    mixed = (a[23] - a[0]) / (.25 * lam * scale) - k @ np.array([1, -1, 1, -1, 1])
    return j, k, mixed


def budget(k, p):
    return p['jacobian_absolute_budget'] + p['jacobian_relative_budget'] * np.abs(k)


def decision(error, allowance):
    error, allowance = np.asarray(error), np.asarray(allowance)
    if error.shape != allowance.shape or not np.isfinite(error).all() or not np.isfinite(allowance).all():
        raise ValueError('decision shape or finite values')
    return {'error': error.tolist(), 'allowance': allowance.tolist(),
            'pass_matrix': (error <= allowance).tolist(), 'pass': bool(np.all(error <= allowance)),
            'max_ratio': float(np.max(error / allowance))}


def seeds(seed, lam, p):
    rows = [seed]
    for k in range(6):
        direction = np.eye(5)[k] if k < 5 else np.array([1, -1, 1, -1, 1])
        rows.extend(seed + lam * np.asarray(p['seed_steps']) * direction * m for m in (.5, -.5, .25, .125))
    return rows


def calls(setting):
    rows = [setting['entry_nominal']]
    for target in setting['targets']:
        rows.extend(target['samples'])
        rows.append(target['fixed_reference_start_nominal'])
    return rows


def physical_identity(raw, fixture, historic, p, lam):
    for key in ('input_xaod', 'ordinal', 'actual_run', 'actual_event', 'seed', 'seed_z_mm',
                'conditions', 'navigator', 'sensors', 'defaults', 'math_control',
                'wb101_node_evidence', 'wb101_muon_mass_Acts'):
        if raw[key] != historic[key]:
            raise ValueError('physical identity ' + key)
    if raw['wb102_lambda'] != lam or raw['wb101_calls'] != 1264:
        raise ValueError('physical lambda or population')
    order = [(tau, cap) for tau in p['arms'] for cap in p['max_step_sizes_m']]
    if [(s['direction_threshold'], s['cap_m']) for s in raw['settings']] != order:
        raise ValueError('complete arm/cap matrix')
    expected_seeds = seeds(v(raw['seed'], 5), lam, p)
    ids, failures, reproduction, stats, path_changes = [], [], [], [], []
    scale = np.asarray(p['output_scales'])
    for setting, old in zip(raw['settings'], historic['settings']):
        tau, cap = setting['direction_threshold'], setting['cap_m']
        if setting['tolerance'] != old['tolerance'] or len(setting['targets']) != 3:
            raise ValueError('tolerance/target multiplicity')
        baseline = next(s for s in raw['settings'] if s['direction_threshold'] == 0 and s['cap_m'] == cap)
        for target, previous, base in zip(setting['targets'], old['targets'], baseline['targets']):
            for key in ('station', 'frame', 'y', 'fixed_start'):
                if target[key] != previous[key]:
                    raise ValueError('target identity ' + key)
            if len(target['samples']) != 25:
                raise ValueError('sample population')
            for i, (r, oldrow, bs, xs) in enumerate(zip(target['samples'], previous['samples'], base['samples'], expected_seeds)):
                if r['label'] != oldrow['label'] or not np.array_equal(v(r['seed'], 5), xs):
                    raise ValueError('perturbation label/seed')
                if r['start_state']['q_over_p_per_MeV'] != xs[4]:
                    raise ValueError('start qop')
                if r['sensitive_sequence'] != bs['sensitive_sequence']:
                    path_changes.append({'lambda': lam, 'threshold': tau, 'cap_m': cap,
                                         'station': target['station'], 'sample': i})
        for index, (r, oldrow) in enumerate(zip(calls(setting), calls(old))):
            cid = r['call_id']
            ids.append(cid)
            identity = {'lambda': lam, 'threshold': tau, 'cap_m': cap, 'call_id': cid}
            if r['options'] != oldrow['options'] or r['target_frame'] != oldrow['target_frame']:
                raise ValueError('options or target frame')
            cc = r['direction_control']
            if cc['threshold'] != tau or cc['allowance'] != fixture['wb101_protocol']['direction_uncertainty_allowance']:
                raise ValueError('controller constants')
            if tau:
                if cc['accepted'] != r['accepted_steps'] or cc['trials'] != cc['accepted'] + cc['direction_rejected'] + cc['position_rejected'] or cc['max_accepted_budget'] > tau:
                    raise ValueError('controller accounting')
            elif any(cc[k] for k in ('trials', 'accepted', 'node_queries', 'direction_rejected', 'position_rejected')):
                raise ValueError('disabled intercepted step')
            counts = r['field_counts']
            expected = 3*r['accepted_steps'] + 2*r['rejected_trials'] + int(r['options']['loopProtection'])
            if counts['inside'] + counts['outside'] != counts['total'] or counts['gradient_calls'] or (r['status'] == 'PASS' and counts['total'] != expected):
                raise ValueError('field query accounting')
            if r['status'] != 'PASS':
                failures.append(identity | {'error': r.get('error_message')})
            else:
                check_state(r['state'], np.asarray(r['target_frame'])[2, 3], r['start_state']['q_over_p_per_MeV'])
                if r['accepted_steps'] != r['propagator_steps_counter'] + 1:
                    raise ValueError('accepted step count')
            if tau == 0:
                if 'official_default_state' not in r or r['status'] != 'PASS':
                    raise ValueError('disabled official comparison missing')
                error = norm((v(r['state']['h']) - v(r['official_default_state']['h'])) / scale)
                if error > p['historical_scaled_tolerance']:
                    raise ValueError('official backend reproduction')
                reproduction.append(identity | {'official_error': error})
            if lam == 1:
                for key in ('status', 'accepted_steps', 'rejected_trials', 'field_counts', 'sensitive_sequence',
                            'options', 'direction_control', 'start_state', 'target_frame'):
                    if r[key] != oldrow[key]:
                        raise ValueError('lambda=1 full historical reproduction ' + key)
                for key in ('h', 'position_mm', 'direction', 'bound_parameters'):
                    if r['state'][key] != oldrow['state'][key]:
                        raise ValueError('lambda=1 exact historical state ' + key)
                if r['state']['time_Acts'] != oldrow['state']['time_Acts'] or r['state']['q_over_p_per_MeV'] != oldrow['state']['q_over_p_per_MeV']:
                    raise ValueError('lambda=1 historical q/time')
            stats.append(identity | {key: r[key] for key in ('status', 'accepted_steps', 'rejected_trials',
                                                            'field_counts', 'direction_control', 'sensitive_sequence')})
    if ids != list(range(1264)) or len(reproduction) != 316:
        raise ValueError('call identity or official population')
    return {'gate': 'PASS', 'failures': failures, 'official_reproduction': reproduction,
            'call_stats': stats, 'sensitive_path_changes': path_changes,
            'historical_full_reproduction': 'PASS' if lam == 1 else 'NOT_APPLICABLE'}


def reference_identity(raw, fixture, historic, p, lam):
    for key in ('input_xaod', 'ordinal', 'actual_run', 'actual_event', 'seed', 'seed_z_mm',
                'conditions', 'mesh_mm', 'node_count', 'math_control', 'sensors', 'probes', 'domain_controls'):
        if raw[key] != historic[key]:
            raise ValueError('reference identity ' + key)
    if raw['wb102_lambda'] != lam or [m['name'] for m in raw['modes']] != p['reference_modes']:
        raise ValueError('reference lambda/modes')
    expected = seeds(v(raw['seed'], 5), lam, p)
    gates = {}
    controls = dict(fixture['wb95_protocol'], reference_rk4_dz_mm=p['reference_dz_mm'])
    for mode in raw['modes']:
        if [l['dz_mm'] for l in mode['ladders']] != p['reference_dz_mm']:
            raise ValueError('reference ladder')
        oldmode = next(m for m in historic['modes'] if m['name'] == mode['name'])
        for ladder in mode['ladders']:
            if len(ladder['samples']) != 25:
                raise ValueError('reference population')
            oldladder = next(l for l in oldmode['ladders'] if l['dz_mm'] == ladder['dz_mm'])
            for row, oldrow, xs in zip(ladder['samples'], oldladder['samples'], expected):
                if row['label'] != oldrow['label'] or not np.array_equal(v(row['seed'], 5), xs):
                    raise ValueError('reference perturbation')
                for point in ('entry', 'interior', 0, 1, 2):
                    state = row[point] if isinstance(point, str) else row['targets'][point]
                    previous = oldrow[point] if isinstance(point, str) else oldrow['targets'][point]
                    if state['z_mm'] != previous['z_mm'] or state['q_over_p_per_MeV'] != xs[4] or not np.isfinite(state['time_Acts']):
                        raise ValueError('reference plane/q/time')
                    v(state['h'], 4)
                if lam == 1 and row != oldrow:
                    raise ValueError('reference lambda=1 full historical reproduction')
        gates[mode['name']] = reference_check(mode['ladders'], controls)
    return {'gate': 'PASS', 'modes': gates, 'historical_full_reproduction': 'PASS' if lam == 1 else 'NOT_APPLICABLE'}


def analyze(raws, references, fixtures, historic, oldref, p):
    if list(raws) != p['lambdas'] or list(references) != p['lambdas'] or list(fixtures) != p['lambdas']:
        raise ValueError('complete lambda ladder')
    identity, refs, cells, diagnostics, failures = [], {}, [], [], []
    reference_checks, checks = [], []
    for lam in p['lambdas']:
        ident = physical_identity(raws[lam], fixtures[lam], historic, p, lam)
        ri = reference_identity(references[lam], fixtures[lam], oldref, p, lam)
        identity.append({'lambda': lam, 'physical': ident, 'reference': ri})
        failures.extend(ident['failures'])
        for mode in references[lam]['modes']:
            for station in (1, 2, 3):
                arrays = [np.array([v(r['targets'][station-1]['h'], 4) for r in l['samples']]) for l in mode['ladders']]
                responses = [response(a, lam, p) for a in arrays]
                _, k, m = responses[-1]
                uk, um = np.abs(k-responses[-2][1]), np.abs(m-responses[-2][2])
                b = budget(k, p)
                refs[lam, mode['name'], station] = {'arrays': arrays, 'K': k, 'M': m, 'U_K': uk, 'U_M': um, 'B': b}
                ref_check = decision(uk, p['reference_budget_fraction'] * b)
                reference_checks.append({'lambda': lam, 'mode': mode['name'], 'station': station,
                                         'metric': 'reference_U_K', **ref_check})
                for tau in p['arms']:
                    candidates = []
                    for setting in raws[lam]['settings']:
                        if setting['direction_threshold'] != tau:
                            continue
                        rows = setting['targets'][station-1]['samples']
                        if any(r['status'] != 'PASS' for r in rows):
                            continue
                        values = np.array([v(r['state']['h'], 4) for r in rows])
                        candidates.append(values)
                        jact, kact, mact = response(values, lam, p)
                        cell = {'lambda': lam, 'mode': mode['name'], 'station': station, 'threshold': tau,
                                'cap_m': setting['cap_m'], 'J': jact.tolist(), 'K': kact.tolist(),
                                'reference_K': k.tolist(), 'U_K': uk.tolist(), 'B': b.tolist(),
                                'mixed_closure': mact.tolist(), 'reference_mixed_closure': m.tolist(), 'U_M': um.tolist()}
                        cells.append(cell)
                        if mode['name'] == 'mesh_z_double' and tau in p['candidate_arms'] and lam in p['plateau_lambdas']:
                            identcell = {key: cell[key] for key in ('lambda', 'mode', 'station', 'threshold', 'cap_m')}
                            checks.append(identcell | {'metric': 'reference_agreement',
                                          **decision(np.abs(kact-k), b+p['uncertainty_factor']*uk)})
                            checks.append(identcell | {'metric': 'mixed_agreement',
                                          **decision(np.abs(mact-m), b.sum(axis=1)+p['uncertainty_factor']*um)})
                    if len(candidates) == 4:
                        metric = complete_metrics(candidates, arrays[-1], np.asarray(p['output_scales']))
                        diagnostics.append({'lambda': lam, 'mode': mode['name'], 'station': station, 'threshold': tau,
                                            **metric, 'T_over_lambda_squared': metric['taylor_excess']/lam**2,
                                            'Q_over_lambda_squared': metric['full_minus_two_half_excess']/lam**2})
    for a, b in zip(p['plateau_lambdas'][:-1], p['plateau_lambdas'][1:]):
        for station in (1, 2, 3):
            ra, rb = (refs[x, 'mesh_z_double', station] for x in (a, b))
            allowance = np.maximum(ra['B'], rb['B']) + p['uncertainty_factor']*(ra['U_K']+rb['U_K'])
            reference_checks.append({'lambdas': [a, b], 'mode': 'mesh_z_double', 'station': station,
                                     'metric': 'reference_plateau', **decision(np.abs(ra['K']-rb['K']), allowance)})
            for tau in p['candidate_arms']:
                for cap in p['max_step_sizes_m']:
                    selected = [next((c for c in cells if c['lambda'] == x and c['mode'] == 'mesh_z_double' and
                                      c['station'] == station and c['threshold'] == tau and c['cap_m'] == cap), None) for x in (a, b)]
                    if None in selected:
                        continue
                    checks.append({'lambdas': [a, b], 'mode': 'mesh_z_double', 'station': station, 'threshold': tau,
                                   'cap_m': cap, 'metric': 'scale_plateau',
                                   **decision(np.abs(np.array(selected[0]['K'])-selected[1]['K']), allowance)})
    ref_ok = all(i['reference']['modes']['mesh_z_double']['gate'] == 'NUMERICAL_REFERENCE_SUPPORTED' for i in identity)
    ref_ok = ref_ok and all(c['pass'] for c in reference_checks if c['mode'] == 'mesh_z_double')
    if not failures and (len(cells) != 384 or len(checks) != 192 or len(diagnostics) != 96):
        raise ValueError('incomplete scientific decision matrix')
    hypothesis = 'UNKNOWN' if not ref_ok else ('NOT_SUPPORTED' if failures or not all(c['pass'] for c in checks) else 'SUPPORTED_BUT_LIMITED')
    return {'schema': 'wb102_summary_v1', 'execution_contract': 'PASS', 'hypothesis': hypothesis,
            'reference_prerequisite': bool(ref_ok), 'identity': identity, 'cells': cells,
            'diagnostics': diagnostics, 'reference_checks': reference_checks, 'decision_checks': checks,
            'failures': failures, 'qualification': 'NOT_EVALUATED', 'physics_screening': 'NOT_EVALUATED',
            'final_oracle': 'NOT_EVALUATED', 'held_out_access': False, 'population': 1,
            'production_backend_changed': False, 'physical_field_accuracy': 'UNKNOWN'}
