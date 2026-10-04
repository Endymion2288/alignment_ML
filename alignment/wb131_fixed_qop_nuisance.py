"""Conditional four-parameter track diagnostic; never an alignment solver."""
from __future__ import annotations

import numpy as np


def derivative_request(export, protocol):
    seed = np.asarray(export['p_seed'], dtype=float).reshape(5)
    arms = [{'tag': 'nominal', 'seed': seed.tolist()}]
    for axis, step in enumerate(protocol['fd_steps']):
        for factor in protocol['fd_factors']:
            for sign in (-1, 1):
                shifted = seed.copy()
                shifted[axis] += sign * step * factor
                arms.append({'tag': f'{axis}:{factor}:{sign}', 'seed': shifted.tolist()})
    return {'schema': 'wb131_request_v1', 'arms': arms}


def prediction(arm, export):
    if len(arm['rows']) != len(export['rows']):
        raise ValueError('row count')
    values = []
    seed = np.asarray(arm['seed'], dtype=float).reshape(5)
    for row, original in zip(arm['rows'], export['rows']):
        for key in ('cluster_id', 'wafer_id', 'station'):
            if row[key] != original[key]:
                raise ValueError('row identity')
        if (row['status'] != 'SUCCESS' or not row['inside_bounds'] or
                not row['is_on_surface_with_bounds']):
            raise ValueError('invalid finite-surface propagation')
        state = row['state']
        pos = np.asarray(state['global_position_mm']).reshape(3)
        local = np.asarray(state['local_position_mm']).reshape(3)
        frame = np.asarray(original['sensor_transform'])
        if (np.linalg.norm(frame[:3, :3] @ local + frame[:3, 3] - pos) > 1e-9 or
                abs(local[2]) > 1e-5 or row['frame_roundtrip_mm'] > 1e-9):
            raise ValueError('frame round trip')
        qop = state['qop_per_MeV']
        if abs(qop - seed[4]) > max(1e-15, abs(seed[4]) * 1e-10):
            raise ValueError('qop changed')
        if state['covariance_present']:
            raise ValueError('unexpected covariance transport')
        value = row['predicted_loc0_mm']
        if abs(value - local[0]) > 1e-12:
            raise ValueError('loc0 representation')
        if abs(original['local_position'][0] - value - row['local_residual_mm']) > 1e-12:
            raise ValueError('residual sign')
        values.append(value)
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError('nonfinite prediction')
    return values


def stable_columns(full, half, sigma, protocol):
    scales = np.asarray(protocol['parameter_scales'])
    a, b = full * scales / sigma[:, None], half * scales / sigma[:, None]
    relative = np.linalg.norm(a - b, axis=0) / np.maximum(
        np.maximum(np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)), protocol['fd_norm_floor'])
    return relative, bool(np.all(relative <= protocol['fd_column_weighted_relative_tolerance']))


def conditional_fit(h, residual, sigma, protocol):
    scales = np.asarray(protocol['parameter_scales'])
    a, b = h * scales / sigma[:, None], residual / sigma
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(singular > protocol['rank_relative_cutoff'] * singular[0])) if singular[0] else 0
    result = {'rank': rank, 'singular_values': singular.tolist(), 'scaled_right_vectors': vt.tolist()}
    if rank != 4:
        return {**result, 'gate': 'UNKNOWN_RANK'}
    # All four modes retained. No normal inverse, ridge, prior or damping.
    delta = vt.T @ ((u.T @ b) / singular)
    projected = a @ delta
    result.update(delta_scaled=delta.tolist(), delta=(scales * delta).tolist(),
                  nominal_cost=float(b @ b), explained_cost=float(projected @ projected),
                  orthogonal_cost=float((b - projected) @ (b - projected)),
                  orthogonality_norm=float(np.linalg.norm(a.T @ (b - projected))),
                  linear_residual_mm=(residual - h @ (scales * delta)).tolist())
    result['gate'] = ('READY' if np.max(np.abs(delta)) <= protocol['max_absolute_scaled_update']
                      else 'UNKNOWN_UPDATE_CAP')
    return result


def fit_event(export, response, nominal, protocol):
    expected = derivative_request(export, protocol)
    if response['identity'] != export['identity'] or nominal['identity'] != export['identity']:
        raise ValueError('event identity')
    if [(a['tag'], a['seed']) for a in response['arms']] != [(a['tag'], a['seed']) for a in expected['arms']]:
        raise ValueError('arm definition')
    if response['official_calls'] != len(expected['arms']) * len(export['rows']):
        raise ValueError('call count')
    values = {a['tag']: prediction(a, export) for a in response['arms']}
    old = np.array([r['state']['local_position_mm'][0][0] for r in nominal['rows']])
    if [r['cluster_id'] for r in nominal['rows']] != [r['cluster_id'] for r in export['rows']]:
        raise ValueError('nominal row order')
    repeat = float(np.max(np.abs(values['nominal'] - old)))
    if repeat > protocol['nominal_repeat_max_mm']:
        return {'gate': 'UNKNOWN_NOMINAL_REPEAT', 'nominal_repeat_max_mm': repeat}
    h = []
    for factor in protocol['fd_factors']:
        h.append(np.column_stack([(values[f'{axis}:{factor}:1'] - values[f'{axis}:{factor}:-1']) /
                                  (2 * step * factor) for axis, step in enumerate(protocol['fd_steps'])]))
    sigma = np.sqrt([r['sigma_sq'] for r in export['rows']])
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError('measurement covariance')
    relative, stable = stable_columns(*h, sigma, protocol)
    measured = np.array([r['local_position'][0] for r in export['rows']])
    residual = measured - values['nominal']
    result = {'nominal_repeat_max_mm': repeat, 'fd_weighted_column_relative_difference': relative.tolist(),
              'fd_stable': stable, 'h_full': h[0].tolist(), 'h_half': h[1].tolist(),
              'nominal_prediction_mm': values['nominal'].tolist(), 'nominal_residual_mm': residual.tolist(),
              'fd_max_absolute_element_difference': np.max(np.abs(h[0] - h[1]), axis=0).tolist()}
    full = conditional_fit(h[0], residual, sigma, protocol)
    half = conditional_fit(h[1], residual, sigma, protocol)
    result.update(full_fit=full, half_fit=half)
    # Unstable derivatives may be inspected, but never authorize a replay.
    result['gate'] = full['gate'] if stable else 'UNKNOWN_FD_STABILITY'
    if full.get('nominal_cost', 0) == 0:
        result['gate'] = 'UNKNOWN_ZERO_COST' if stable and full['rank'] == 4 else result['gate']
    if result['gate'] == 'READY':
        updated = np.asarray(export['p_seed']).reshape(5).copy()
        updated[:4] += np.asarray(full['delta'])
        result['updated_seed'] = updated.tolist()
    return result


def replay_metrics(export, fit, response, protocol):
    if fit['gate'] != 'READY' or len(response['arms']) != 1:
        raise ValueError('unauthorized replay')
    arm = response['arms'][0]
    if (arm['tag'] != 'updated' or arm['seed'] != fit['updated_seed'] or
            response['identity'] != export['identity'] or response['official_calls'] != len(export['rows'])):
        raise ValueError('replay identity')
    actual = prediction(arm, export)
    nominal = np.asarray(fit['nominal_prediction_mm'])
    predicted_change = np.asarray(fit['h_full']) @ np.asarray(fit['full_fit']['delta'])
    error = actual - nominal - predicted_change
    sigma = np.sqrt([r['sigma_sq'] for r in export['rows']])
    measured = np.array([r['local_position'][0] for r in export['rows']])
    residual = measured - actual
    q0 = fit['full_fit']['nominal_cost']
    q1 = float(np.sum((residual / sigma) ** 2))
    ratio = q1 / q0
    relative = float(np.linalg.norm(error / sigma) / max(np.linalg.norm(predicted_change / sigma), protocol['fd_norm_floor']))
    rms = float(np.sqrt(np.mean(error ** 2)))
    linear_pass = relative <= protocol['nonlinear_weighted_relative_tolerance'] or rms <= protocol['nonlinear_absolute_rms_mm']
    cost_pass = ratio <= protocol['maximum_cost_ratio']
    stations = {}
    for station in sorted({r['station'] for r in export['rows']}):
        mask = np.array([r['station'] == station for r in export['rows']])
        stations[str(station)] = {'rows': int(mask.sum()), 'nominal_rms_mm': float(np.sqrt(np.mean((measured[mask]-nominal[mask])**2))),
                                 'actual_rms_mm': float(np.sqrt(np.mean(residual[mask]**2))),
                                 'actual_cost': float(np.sum((residual[mask]/sigma[mask])**2))}
    return {'verdict': 'PASS_CONDITIONAL_ONE_STEP' if linear_pass and cost_pass else 'FAIL_ONE_STEP_HYPOTHESIS',
            'linearization_pass': bool(linear_pass), 'cost_reduction_pass': bool(cost_pass),
            'cost_ratio': ratio, 'nominal_cost': q0, 'actual_cost': q1,
            'descriptive_cost_per_remaining_dof': q1 / (len(actual)-4),
            'nonlinear_weighted_relative_error': relative, 'nonlinear_error_rms_mm': rms,
            'nonlinear_error_mm': error.tolist(), 'actual_prediction_mm': actual.tolist(),
            'actual_residual_mm': residual.tolist(), 'stations': stations,
            'nominal_rms_mm': float(np.sqrt(np.mean((measured-nominal)**2))),
            'actual_rms_mm': float(np.sqrt(np.mean(residual**2)))}
