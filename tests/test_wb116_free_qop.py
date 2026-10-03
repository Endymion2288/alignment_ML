"""Synthetic independent projections and response identity controls, no actual qop fit."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_wb116_free_qop import profile, saved_response, spectrum


def model():
    j4 = np.vstack([np.eye(4)]*2)
    jq = np.r_[np.ones(4), -np.ones(4)]
    return jq, np.c_[j4, jq], np.eye(4), [np.eye(4)]*2


def test_free_qop_contrast_has_analytic_solution_and_keeps_shared_correlations():
    r, j, c, cs = model()
    p = profile(r, j, c, cs, np.ones(4), 1.)
    assert p['status'] == 'RESOLVED_LOCAL_ARITHMETIC_ONLY'
    assert p['Q_fixed_qop_shared_seed'] == pytest.approx(8.)
    assert p['Q_free_qop'] == pytest.approx(0., abs=1e-25)
    assert p['linear_delta'] == pytest.approx([0., 0., 0., 0., -1.])
    assert p['conditional_information_scaled_qop'] == pytest.approx(8.)
    assert p['conditional_curvature_width_per_MeV'] == pytest.approx(1./np.sqrt(8.))
    assert p['data_only_spectrum']['machine_rank'] == 5
    assert p['qop_prior'] == 'NONE' and p['dummy_qop_variance_used'] is False


def test_data_degenerate_direction_can_depend_on_seed_prior():
    j4 = np.vstack([np.eye(4)]*2); jq = j4[:, 0]
    p = profile(jq, np.c_[j4, jq], np.eye(4), [np.eye(4)]*2, np.ones(4), 1.)
    assert p['data_only_spectrum']['machine_rank'] == 4
    assert p['augmented_spectrum']['machine_rank'] == 5
    assert p['conditional_information_scaled_qop'] == pytest.approx(2./3.)
    assert p['linear_delta'][-1] == pytest.approx(-1.)


def test_same_direction_residual_retains_shared_seed_objective():
    _, j, c, cs = model()
    p = profile(np.ones(8), j, c, cs, np.ones(4), 1.)
    assert p['Q_fixed_qop_shared_seed'] == pytest.approx(8./3.)
    assert p['Q_free_qop'] == pytest.approx(8./3.)
    assert p['linear_delta'] == pytest.approx([-2./3.]*4 + [0.])


def test_zero_momentum_response_remains_unknown_without_floor_or_pinv():
    r, j, c, cs = model(); j[:, 4] = 0.
    p = profile(r, j, c, cs, np.ones(4), 1.)
    assert p['status'] == 'UNKNOWN_NUMERICALLY_UNRESOLVED_QOP_DIRECTION'
    assert p['augmented_spectrum']['machine_rank'] == 4
    assert 'linear_delta' not in p


def test_changing_units_preserves_objective_and_transforms_delta():
    r, j, c, cs = model(); r = r + np.arange(8)*.1
    a = profile(r, j, c, cs, np.ones(4), .02)
    u = np.array([1000., 1000., .001, .001]); v = np.tile(u, 2); w = np.r_[u, 1000.]
    b = profile(r*v, j*v[:, None]/w[None, :], c*u[:, None]*u[None, :],
                [x*u[:, None]*u[None, :] for x in cs], u, 20.)
    for key in ('Q_free_qop', 'Q_fixed_qop_shared_seed', 'Q_prior_part', 'Q_data_part'):
        assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)
    assert np.asarray(b['linear_delta']) == pytest.approx(np.asarray(a['linear_delta'])*w)


@pytest.mark.parametrize('bad', ['negative', 'singular', 'asymmetric', 'nan_response', 'infinite_residual', 'zero_scale', 'negative_qscale'])
def test_invalid_covariance_and_response_fail_closed(bad):
    r, j, c, cs = model(); scales = np.ones(4); qs = 1.
    if bad == 'negative': c[0, 0] = -1.
    elif bad == 'singular': c[0, 0] = 0.
    elif bad == 'asymmetric': c[0, 1] = .1
    elif bad == 'nan_response': j[0, 0] = np.nan
    elif bad == 'infinite_residual': r[0] = np.inf
    elif bad == 'zero_scale': scales[0] = 0.
    elif bad == 'negative_qscale': qs = -1.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        profile(r, j, c, cs, scales, qs)


def synthetic_trace():
    steps = np.array([.01, .01, 1e-5, 1e-5, 1e-7]); seed = np.r_[np.zeros(4), 1e-5]
    refs = [{'fixed_z_state': np.zeros((4, 1)).tolist(), 'z_state_mm': k*10., 'q_over_p_per_MeV': 1e-5} for k in range(4)]
    fixture = {'references': refs, 'protocol': {'seed_steps': steps.tolist(), 'output_scales': [1., 1., .001, .001]}}
    targets = []; trace = []; nextid = 200
    truth_j = np.c_[np.eye(4), np.arange(1., 5.)*1e6]
    for station, cid in zip((1, 2, 3), (24, 74, 124)):
        frame = np.eye(4); frame[2, 3] = station*10.
        t = {'station': station, 'call_id': cid, 'frame': frame.tolist(),
             'official_has_value': station != 3, 'official_h': np.zeros((4, 1)).tolist() if station != 3 else None}
        targets.append(t)
        trace.append({'record': 'input', 'call_id': cid, 'station': station, 'axis': -1,
                      'label': 'nominal', 'frame': frame.tolist(), 'seed': seed[:, None].tolist(), 'seed_z_mm': 0.})
        if station == 3: continue
        trace.append({'record': 'output', 'call_id': cid, 'h': copy.deepcopy(t['official_h'])})
        for axis in range(5):
            for label, sign, factor in (('xi_plus', 1., 1.), ('xi_minus', -1., 1.),
                                        ('xi_half_plus', 1., .5), ('xi_half_minus', -1., .5)):
                x = seed.copy(); x[axis] += sign*factor*steps[axis]
                trace.append({'record': 'input', 'call_id': nextid, 'station': station, 'axis': axis,
                              'label': label, 'frame': frame.tolist(), 'seed': x[:, None].tolist(), 'seed_z_mm': 0.})
                # Independent physical-coordinate derivative, not the analyzer formula.
                h = truth_j @ (x-seed)
                trace.append({'record': 'output', 'call_id': nextid, 'h': h[:, None].tolist()}); nextid += 1
    control = {'seed': seed[:, None].tolist(), 'seed_z_mm': 0., 'targets': targets}
    return fixture, control, trace, truth_j


def test_five_axis_parser_reconstructs_independent_linear_operator():
    f, c, t, j = synthetic_trace()
    _, matrices, details, _, steps, _ = saved_response(f, c, t)
    assert matrices['full'] == pytest.approx(np.vstack([j]*2))
    assert matrices['half'] == pytest.approx(np.vstack([j]*2))
    assert all(len(x['used_call_ids']) == 20 for x in details)
    assert steps[4] == 1e-7


@pytest.mark.parametrize('bad', ['missing_qop_input', 'missing_qop_output', 'duplicate_input', 'duplicate_output',
                               'wrong_qop', 'wrong_other_coordinate', 'wrong_axis', 'wrong_station',
                               'wrong_frame', 'wrong_seed_z', 'wrong_nominal', 'fake_station3_output'])
def test_fifth_axis_and_saved_identities_reject_mutations(bad):
    f, c, t, _ = synthetic_trace(); f, c, t = copy.deepcopy((f, c, t))
    row = next(x for x in t if x['record'] == 'input' and x['station'] == 1 and x['axis'] == 4 and x['label'] == 'xi_plus')
    out = next(x for x in t if x['record'] == 'output' and x['call_id'] == row['call_id'])
    if bad == 'missing_qop_input': t.remove(row)
    elif bad == 'missing_qop_output': t.remove(out)
    elif bad == 'duplicate_input': t.append(copy.deepcopy(row))
    elif bad == 'duplicate_output': t.append(copy.deepcopy(out))
    elif bad == 'wrong_qop': row['seed'][4][0] *= 1000.
    elif bad == 'wrong_other_coordinate': row['seed'][1][0] += 1.
    elif bad == 'wrong_axis': row['axis'] = 3
    elif bad == 'wrong_station': row['station'] = 2
    elif bad == 'wrong_frame': row['frame'][2][3] += 1.
    elif bad == 'wrong_seed_z': row['seed_z_mm'] += 1.
    elif bad == 'wrong_nominal': c['targets'][0]['official_h'][0][0] += 1.
    elif bad == 'fake_station3_output': t.append({'record': 'output', 'call_id': 124, 'h': np.zeros((4, 1)).tolist()})
    with pytest.raises(ValueError): saved_response(f, c, t)


def test_machine_rank_reports_resolution_without_physical_cut():
    s = spectrum(np.diag([1., 1e-18]))
    assert s['machine_rank'] == 1
    assert s['machine_resolution_cutoff'] == pytest.approx(np.finfo(np.float64).eps*2)
    assert s['scientific_identifiability'] == 'UNVERIFIED_UNCALIBRATED_COVARIANCE'
